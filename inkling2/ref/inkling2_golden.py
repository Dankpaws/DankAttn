"""Golden reference for thinkingmachines/Inkling-2-Mini (Inkling2LoopForConditionalGeneration, text tower).

Follows HuggingFace transformers modeling_inkling2_loop.py op-for-op directly from the
native NVFP4 / MXFP8 / BF16 safetensors checkpoint, without needing transformers' loader.

Also supports simulating DankAttn W8 quantization on attention / unembed projections to
measure exact KL divergence and top-1 agreement across the 3-pass looped core before
packing or running on Metal.

Usage:
  python3 inkling2_golden.py /path/to/Inkling-2-Mini --prompt "The capital of France is"
  python3 inkling2_golden.py /path/to/Inkling-2-Mini --ids ids.bin --out golden.npz --dump
  python3 inkling2_golden.py /path/to/Inkling-2-Mini --prompt "..." --w8-mode core_bf16
"""
import argparse
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.set_grad_enabled(False)

E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
_BYTE_IDX = torch.arange(256, dtype=torch.long)
_BYTE_TO_E2M1 = torch.stack((E2M1_VALUES[_BYTE_IDX & 0xF], E2M1_VALUES[_BYTE_IDX >> 4]), dim=-1)


class Weights:
    def __init__(self, root: str):
        self.root = root
        idx_path = os.path.join(root, "model.safetensors.index.json")
        self.map = json.load(open(idx_path))["weight_map"]
        self.handles = {}

    def _h(self, fname: str):
        if fname not in self.handles:
            self.handles[fname] = safe_open(os.path.join(self.root, fname), framework="pt", device="cpu")
        return self.handles[fname]

    def has(self, name: str) -> bool:
        return name in self.map

    def get(self, name: str, dtype=torch.float32) -> torch.Tensor:
        t = self._h(self.map[name]).get_tensor(name)
        return t if dtype is None else t.to(dtype)

    def dequant_expert(self, pfx: str, expert_idx: int, interleave_w13: bool = False) -> torch.Tensor:
        """Dequantize a single expert's NVFP4 or MXFP8 weight to float32 [rows, cols]."""
        w_sl = self._h(self.map[f"{pfx}_weight"]).get_slice(f"{pfx}_weight")
        s_sl = self._h(self.map[f"{pfx}_weight_scale"]).get_slice(f"{pfx}_weight_scale")
        w = w_sl[expert_idx]
        s = s_sl[expert_idx].float()
        if self.has(f"{pfx}_weight_amax"):
            amax = self._h(self.map[f"{pfx}_weight_amax"]).get_slice(f"{pfx}_weight_amax")[expert_idx].float()
            denom = self.get(f"{pfx}_weight_outer_scale_denominator", torch.float32)
            codes = w.to(torch.uint8).long()
            vals = _BYTE_TO_E2M1[codes].view(*s.shape, 16) * s[..., None]
            out = (vals * (amax / denom)).reshape(w.shape[0], w.shape[1] * 2)
        else:
            vals = w.float().view(*s.shape, 32) * torch.exp2(s - 127.0)[..., None]
            out = vals.reshape(w.shape[0], w.shape[1])
        if interleave_w13:
            # Checkpoint stores [gate_0, up_0, gate_1, up_1, ...]; deinterleave to [gate..., up...]
            out = out.reshape(out.shape[0] // 2, 2, out.shape[1]).transpose(0, 1).reshape(out.shape[0], out.shape[1])
        return out


def sim_w8(w: torch.Tensor, gs: int = 128) -> torch.Tensor:
    """Simulate DankAttn symmetric W8 group-wise quantization (gs=128 along K)."""
    N, K = w.shape
    w_g = w.float().reshape(N, K // gs, gs)
    amax = w_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / 127.0).to(torch.float16).float()
    q = (w_g / scale.clamp(min=1e-12)).round().clamp(-127, 127)
    return (q * scale).reshape(N, K)


def rms_norm(x: torch.Tensor, w: torch.Tensor | None, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return y if w is None else y * w.float()


def l2_clip(x: torch.Tensor, threshold: float, scale: torch.Tensor | None = None) -> torch.Tensor:
    xf = x.float()
    if scale is not None:
        xf = xf * scale.float()
    norm = torch.linalg.vector_norm(xf, dim=-1, keepdim=True)
    return xf * (threshold / (norm + 1e-6)).clamp(max=1.0)


def log_scale(x: torch.Tensor, pos: torch.Tensor, n_floor: float, alpha: float) -> torch.Tensor:
    tau = 1.0 + alpha * torch.log(((pos.float() + 1.0) / n_floor).clamp(min=1.0))
    return x.float() * tau[:, None, None]


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    return torch.tanh(x / cap) * cap


def block_attn_res(block_states: list[torch.Tensor], partial: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Block Attention Residual: query [H] or [nq, H] attends over RMSNorm(stack(block_states + [partial])).
    Returns [T, H] if query is 1D, or [nq, T, H] if query is 2D.
    """
    # stacked: [T, B, H]
    stacked = torch.stack(block_states + [partial.float()], dim=1)
    normed = rms_norm(stacked, None, eps=1e-6)
    H = normed.shape[-1]
    q = query.float()
    squeeze = q.ndim == 1
    if squeeze:
        q = q.unsqueeze(0)  # [1, H]
    # scores: [T, nq, B]
    scores = torch.einsum("qh,tbh->tqb", q, normed) * (H ** -0.5)
    probs = F.softmax(scores, dim=-1)
    out = torch.einsum("tqb,tbh->qth", probs, normed)  # [nq, T, H]
    return out[0] if squeeze else out


def static_sconv(x: torch.Tensor, weight: torch.Tensor, kernel_size: int = 4) -> torch.Tensor:
    """Inkling2LoopShortConvolution (full sequence, no cache): (conv1d(x) + x) / sqrt(K+1)."""
    xf = x.float()  # [T, C]
    T, C = xf.shape
    w = weight.float().reshape(C, 1, kernel_size)
    # causal conv1d: pad left by kernel_size - 1
    out = F.conv1d(xf.t().unsqueeze(0), w, padding=kernel_size - 1, groups=C)[0, :, :T].t()
    return (out + xf) / math.sqrt(kernel_size + 1)


def dynamic_sconv(
    x: torch.Tensor,
    dyn_in: torch.Tensor,
    static_w: torch.Tensor,
    weight_proj: torch.Tensor,
    num_heads: int,
    kernel_size: int = 4,
) -> torch.Tensor:
    """Inkling2LoopDynamicShortConvolution: (dyn_conv(x) + x) / sqrt(2K+1)."""
    T, C = x.shape
    hd = C // num_heads
    # dyn_w: [T, num_heads, 1, K]
    dyn_w = (dyn_in.float() @ weight_proj.float().t()).reshape(T, num_heads, 1, kernel_size)
    # static_w in ckpt: [K, C] -> [num_heads, hd, K]
    sw = static_w.float().t().reshape(num_heads, hd, kernel_size)
    w = sw[None, :, :, :] + dyn_w  # [T, num_heads, hd, K]

    xf = x.float()
    # causal windows: lag 0 = current token, lag k = token k steps back
    padded = F.pad(xf.t(), (kernel_size - 1, 0))  # [C, T + K - 1]
    windows = padded.unfold(-1, kernel_size, 1).flip(-1).permute(1, 0, 2)  # [T, C, K]
    windows = windows.reshape(T, num_heads, hd, kernel_size)
    out = (windows * w).sum(dim=-1).reshape(T, C)
    return (out + xf) / math.sqrt(2 * kernel_size + 1)


def message_gate(h: torch.Tensor, msg: torch.Tensor, w_down: torch.Tensor, w_forget: torch.Tensor, w_gain: torch.Tensor) -> torch.Tensor:
    hf, mf = h.float(), msg.float()
    cat = torch.cat([rms_norm(hf, None), rms_norm(mf, None)], dim=-1)
    low = cat.to(torch.bfloat16).float() @ w_down.float().t()
    fg = torch.tanh(low @ w_forget.float().t())
    mg = torch.tanh(low @ w_gain.float().t())
    return (1.0 - fg) * hf + mg * mf


def run_layer(
    W: Weights,
    tc: dict,
    phys_idx: int,
    h: torch.Tensor,
    block_states: list[torch.Tensor],
    pos: torch.Tensor,
    messages: torch.Tensor | None,
    w8_attn: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    lp = f"model.llm.layers.{phys_idx}"
    is_local = phys_idx in tc["local_layer_ids"]
    is_boundary = (phys_idx % tc["n_layers_per_attn_res_block"]) == 0
    nh = tc["swa_num_attention_heads"] if is_local else tc["num_attention_heads"]
    nkv = tc["swa_num_key_value_heads"] if is_local else tc["num_key_value_heads"]
    hd = tc["swa_head_dim"] if is_local else tc["head_dim"]
    d_rel = tc["d_rel"]
    rel_extent = tc["sliding_window_size"] if is_local else tc["rel_extent"]
    ks = tc["sconv_kernel_size"]
    eps = tc["rms_norm_eps"]
    T, H = h.shape

    def get_lin(name: str) -> torch.Tensor:
        w = W.get(name, torch.float32)
        return sim_w8(w) if w8_attn else w

    block_residual = h.float()
    # 1. Pre-attention Block Attention Residual
    if phys_idx == 0:
        qh = kh = vh = h.float()
    elif is_local:
        qh = kh = vh = block_attn_res(block_states, h, W.get(f"model.llm.attn_res.attn_queries.{phys_idx - 1}"))
    else:
        qkv = block_attn_res(block_states, h, W.get(f"model.llm.attn_res.attn_queries.{phys_idx - 1}"))
        qh, kh, vh = qkv[0], qkv[1], qkv[2]

    # 2. Message gates (passes >= 1 of core) + input RMSNorm
    if messages is not None:
        qm, km, vm, mm = messages[0], messages[1], messages[2], messages[3]
        qh = message_gate(qh, qm, W.get(f"{lp}.controllers.attn.down_weight"), W.get(f"{lp}.controllers.attn.forget_weight"), W.get(f"{lp}.controllers.attn.gain_weight"))
        kh = message_gate(kh, km, W.get(f"{lp}.controllers.k.down_weight"), W.get(f"{lp}.controllers.k.forget_weight"), W.get(f"{lp}.controllers.k.gain_weight"))
        vh = message_gate(vh, vm, W.get(f"{lp}.controllers.v.down_weight"), W.get(f"{lp}.controllers.v.forget_weight"), W.get(f"{lp}.controllers.v.gain_weight"))

    if is_local:
        wn = W.get(f"{lp}.attn_norm.weight")
        if messages is None:
            qn = kn = vn = rms_norm(qh, wn, eps)
        else:
            qn = rms_norm(qh, wn, eps)
            kn = rms_norm(kh, wn, eps)
            vn = rms_norm(vh, wn, eps)
    else:
        qn = rms_norm(qh, W.get(f"{lp}.q_input_norm.weight"), eps)
        kn = rms_norm(kh, W.get(f"{lp}.k_input_norm.weight"), eps)
        vn = rms_norm(vh, W.get(f"{lp}.v_input_norm.weight"), eps)

    # 3. Attention Q, K, V + sconv + norms
    q = (qn @ get_lin(f"{lp}.attn.wq_du.weight").t()).reshape(T, nh, hd)
    q = l2_clip(q, tc["qk_l2_clip_threshold"], W.get(f"{lp}.attn.q_l2_clip.scale"))
    if not is_local:
        q = log_scale(q, pos, tc["log_scaling_n_floor"], tc["log_scaling_alpha"])

    k_lin = kn @ get_lin(f"{lp}.attn.wk_dv.weight").t()
    v_lin = vn @ get_lin(f"{lp}.attn.wv_dv.weight").t()
    if is_local:
        k_conv = static_sconv(k_lin, W.get(f"{lp}.attn.k_sconv.weight"), ks)
        v_conv = static_sconv(v_lin, W.get(f"{lp}.attn.v_sconv.weight"), ks)
    else:
        k_conv = dynamic_sconv(k_lin, kn, W.get(f"{lp}.attn.k_sconv.static_w"), W.get(f"{lp}.attn.k_sconv.weight_proj"), nkv, ks)
        v_conv = dynamic_sconv(v_lin, vn, W.get(f"{lp}.attn.v_sconv.static_w"), W.get(f"{lp}.attn.v_sconv.weight_proj"), nkv, ks)

    k = l2_clip(k_conv.reshape(T, nkv, hd), tc["qk_l2_clip_threshold"])
    v = rms_norm(v_conv.reshape(T, nkv, hd), None, eps)

    # Relative position bias
    rel = (qn @ get_lin(f"{lp}.attn.wr_du.weight").t()).reshape(T, nh, d_rel)
    rel = l2_clip(rel, tc["rel_l2_clip_threshold"], W.get(f"{lp}.attn.rel_l2_clip.scale"))
    if not is_local:
        rel = log_scale(rel, pos, tc["log_scaling_n_floor"], tc["log_scaling_alpha"])
    rel_proj = W.get(f"{lp}.attn.rel_logits_proj.proj")  # [d_rel, rel_extent]
    rel_logits = torch.einsum("thd,de->hte", rel, rel_proj)  # [nh, T, rel_extent]

    dist = pos[:, None] - pos[None, :]  # [T, T]
    dist_clamped = dist.clamp(0, rel_extent - 1)
    pos_bias = rel_logits.gather(-1, dist_clamped.unsqueeze(0).expand(nh, -1, -1))
    pos_bias = pos_bias.masked_fill((dist < 0) | (dist >= rel_extent), 0.0)

    # Attention scores + causal/SWA mask + sinks
    n_rep = nh // nkv
    k_rep = k.repeat_interleave(n_rep, dim=1)  # [T, nh, hd]
    v_rep = v.repeat_interleave(n_rep, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k_rep) * (hd ** -0.5) + pos_bias
    mask = dist < 0
    if is_local:
        mask = mask | (dist >= tc["sliding_window_size"])
    scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))

    sinks = (qn @ W.get(f"{lp}.attn.attn_sink_proj.weight").t()).t().unsqueeze(-1)  # [nh, T, 1]
    probs = F.softmax(torch.cat([scores, sinks], dim=-1), dim=-1)[..., :-1]
    attn_out = torch.einsum("hqk,khd->qhd", probs, v_rep).reshape(T, nh * hd)

    # Output gate + wo_ud + post_attention_convolution
    gate = torch.sigmoid(qn @ get_lin(f"{lp}.attn.gate_attn.weight").t())
    attn_out = (attn_out * gate) @ get_lin(f"{lp}.attn.wo_ud.weight").t()
    attn_out = static_sconv(attn_out, W.get(f"{lp}.attn_sconv.weight"), ks)

    # 4. Block boundary update + pre-MLP Block Attention Residual
    new_block_states = list(block_states)
    if is_boundary:
        new_block_states.append(block_residual)
        residual = attn_out
    else:
        residual = block_residual + attn_out

    mh = block_attn_res(new_block_states, residual, W.get(f"model.llm.attn_res.mlp_queries.{phys_idx}"))
    if messages is not None:
        mh = message_gate(mh, mm, W.get(f"{lp}.controllers.mlp.down_weight"), W.get(f"{lp}.controllers.mlp.forget_weight"), W.get(f"{lp}.controllers.mlp.gain_weight"))
    mn = rms_norm(mh, W.get(f"{lp}.mlp_norm.weight"), eps)

    # 5. MoE Router (FP32) + SoftcappedSwiGLU Experts
    r_logits = mn @ W.get(f"{lp}.mlp.gate.weight").t()
    r_scores = torch.sigmoid(r_logits)
    r_bias = W.get(f"{lp}.mlp.gate.bias")
    top_idx = torch.topk(r_scores + r_bias, tc["num_experts_per_tok"], dim=-1, sorted=False).indices  # [T, 8]
    top_w = r_scores.gather(1, top_idx)
    top_w = (top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-20)) * tc["route_scale"]

    moe_out = torch.zeros(T, H, dtype=torch.float32)
    unique_e = top_idx.unique().tolist()
    for e in unique_e:
        tok_pos, k_pos = torch.where(top_idx == e)
        w13 = W.dequant_expert(f"{lp}.mlp.experts.w13", e, interleave_w13=True)
        w2 = W.dequant_expert(f"{lp}.mlp.experts.w2", e, interleave_w13=False)
        gu = mn[tok_pos] @ w13.t()
        g_part, u_part = gu.chunk(2, dim=-1)
        act = softcap(g_part, tc["mlp_gate_softcap"]) * torch.sigmoid(g_part) * softcap(u_part, tc["mlp_up_softcap"])
        dn = act @ w2.t()
        moe_out.index_add_(0, tok_pos, dn * top_w[tok_pos, k_pos, None])

    moe_out = static_sconv(moe_out, W.get(f"{lp}.mlp_sconv.weight"), ks)
    out = (residual + moe_out).to(torch.bfloat16).float()
    return out, new_block_states


def run_model(
    W: Weights,
    cfg: dict,
    ids: list[int],
    num_passes: int | None = None,
    w8_mode: str = "none",
    dump: bool = False,
) -> tuple[torch.Tensor, dict]:
    tc = cfg["text_config"]
    lc = tc["loop_config"]
    core_start = lc["core_start"]
    core_end = lc["core_end"]
    passes = lc["num_passes"] if num_passes is None else num_passes
    eps = tc["rms_norm_eps"]
    T = len(ids)
    pos = torch.arange(T, dtype=torch.long)

    # Embedding + embed_norm
    emb_sl = W._h(W.map["model.llm.embed.weight"]).get_slice("model.llm.embed.weight")
    h = torch.stack([emb_sl[i] for i in ids]).float()
    h = rms_norm(h, W.get("model.llm.embed_norm.weight"), eps).to(torch.bfloat16).float()

    # Cross-pass message queries: [12, 4, H] in order (q, k, v, mlp)
    cp_qm = W.get("model.llm.middle_cycle_cross_pass_queries").unflatten(0, (-1, 2))
    cp_kv = W.get("model.llm.middle_cycle_cross_pass_kv_queries").unflatten(0, (-1, 2))
    msg_queries = torch.cat([cp_qm[:, :1], cp_kv, cp_qm[:, 1:]], dim=1)  # [12, 4, H]

    block_states: list[torch.Tensor] = []
    dumped = {}
    logical = 0

    def use_w8(phys_idx: int) -> bool:
        if w8_mode == "all":
            return True
        if w8_mode == "outer":
            return phys_idx < core_start or phys_idx >= core_end
        return False

    # 1. Prelude (0 .. core_start-1)
    for l in range(core_start):
        h, block_states = run_layer(W, tc, l, h, block_states, pos, None, w8_attn=use_w8(l))
        if dump:
            dumped[f"L{logical:02d}_p{l}"] = h[-1].numpy()
        logical += 1

    pre_loop_h = h
    pre_loop_bs = list(block_states)
    n_pre_blocks = len(pre_loop_bs)

    # 2. First pass over core (core_start .. core_end-1)
    for l in range(core_start, core_end):
        h, block_states = run_layer(W, tc, l, h, block_states, pos, None, w8_attn=use_w8(l))
        if dump:
            dumped[f"L{logical:02d}_p{l}_pass0"] = h[-1].numpy()
        logical += 1

    # 3. Remaining passes over core
    for p in range(1, passes):
        prev_bs = block_states[n_pre_blocks + 1 :]
        prev_h = h
        h = pre_loop_h
        block_states = list(pre_loop_bs)
        for loop_i, l in enumerate(range(core_start, core_end)):
            msgs = block_attn_res(prev_bs, prev_h, msg_queries[loop_i])  # [4, T, H]
            h, block_states = run_layer(W, tc, l, h, block_states, pos, msgs, w8_attn=use_w8(l))
            if dump:
                dumped[f"L{logical:02d}_p{l}_pass{p}"] = h[-1].numpy()
            logical += 1

    # 4. Coda (core_end .. num_hidden_layers-1)
    for l in range(core_end, tc["num_hidden_layers"]):
        h, block_states = run_layer(W, tc, l, h, block_states, pos, None, w8_attn=use_w8(l))
        if dump:
            dumped[f"L{logical:02d}_p{l}"] = h[-1].numpy()
        logical += 1

    # 5. Final AttnRes + Norm + Unembed + Logit Softcap
    h = block_attn_res(block_states, h, W.get("model.llm.attn_res.output_query"))
    h = rms_norm(h, W.get("model.llm.norm.weight"), eps).to(torch.bfloat16).float()
    unembed = W.get("model.llm.unembed.weight")[: tc["unpadded_vocab_size"]]
    if w8_mode in ("all", "outer", "unembed"):
        unembed = sim_w8(unembed)
    logits = softcap(h @ unembed.t(), tc["final_logit_softcapping"])
    return logits, dumped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="Path to Inkling-2-Mini checkpoint directory")
    ap.add_argument("--prompt", type=str, default="The capital of France is")
    ap.add_argument("--ids", type=str, default=None, help="Optional int32 .bin or .npy token IDs")
    ap.add_argument("--num-passes", type=int, default=None)
    ap.add_argument("--w8-mode", choices=["none", "unembed", "outer", "all"], default="none")
    ap.add_argument("--compare-w8", action="store_true", help="Compare none vs outer vs all W8 KLD/top-1")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--dump", action="store_true")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.ckpt, "config.json")))
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(args.ckpt, "tokenizer.json"))

    if args.ids:
        ids = (np.load(args.ids) if args.ids.endswith(".npy") else np.fromfile(args.ids, dtype=np.int32)).tolist()
    else:
        ids = tok.encode(args.prompt).ids
    print(f"Prompt ({len(ids)} tokens): {ids[:16]}", flush=True)

    W = Weights(args.ckpt)
    t0 = time.time()
    logits, dumped = run_model(W, cfg, ids, num_passes=args.num_passes, w8_mode=args.w8_mode, dump=args.dump)
    dt = time.time() - t0
    top5 = torch.topk(logits[-1], 5)
    print(f"Done in {dt:.1f}s (w8_mode={args.w8_mode}). Next token top-5:", flush=True)
    for val, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        print(f"  id={idx:6d}  logit={val:7.3f}  text={tok.decode([idx])!r}")

    if args.out:
        save = {"ids": np.array(ids, dtype=np.int32), "logits": logits.numpy(), **dumped}
        np.savez(args.out, **save)
        print(f"Saved {args.out}")

    if args.compare_w8:
        ref_logp = F.log_softmax(logits, dim=-1)
        ref_p = ref_logp.exp()
        ref_top1 = logits.argmax(dim=-1)
        for mode in ("outer", "all"):
            t1 = time.time()
            l2, _ = run_model(W, cfg, ids, num_passes=args.num_passes, w8_mode=mode)
            q_logp = F.log_softmax(l2, dim=-1)
            kl = (ref_p * (ref_logp - q_logp)).sum(dim=-1).mean().item()
            top1 = (l2.argmax(dim=-1) == ref_top1).float().mean().item() * 100.0
            print(f"w8_mode={mode:6s}: mean KL={kl:.6f}, top-1 match={top1:.2f}% ({time.time()-t1:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
