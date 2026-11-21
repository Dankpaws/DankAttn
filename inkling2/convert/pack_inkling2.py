"""Inkling-2-Mini native NVFP4/MXFP8/BF16 checkpoint -> DankAttn inkling2 pack.

  python3 inkling2/convert/pack_inkling2.py /home/dank/models/Inkling-2-Mini /path/to/pack-inkling2-w8

Preserves MoE experts losslessly in their native trained formats:
  - Layers 0..19:  NVFP4 (E2M1, WBITS=40, GS=16, f16 combined scales)
  - Layers 20..23: MXFP8 (E4M3FN, WBITS=80, GS=32, f16 power-of-two scales)
Attention projections & lm_head are quantized to DankAttn W8 (GS=128)
(validated at ~0.00042 KL / 100% top-1 vs native BF16), with fused projections:
  L{i}.attn.qgrs : [wq_du (3072) | gate_attn (3072) | wr_du (384) | attn_sink_proj (24 + 8 pad)]
  L{i}.attn.k    : [wk_dv (1024)] (local) or [wk_dv (1024) | k_sconv.weight_proj (32)] (global)
  L{i}.attn.v    : [wv_dv (1024)] (local) or [wv_dv (1024) | v_sconv.weight_proj (32)] (global)
  L{i}.attn.o    : [wo_ud (2304 x 3072)]
  L{i}.moe.gu    : 128 experts x [2304, 2304], 16-gate | 16-up interleaved for EPI=3 SoftcappedSwiGLU
  L{i}.moe.dn    : 128 experts x [2304, 1152]

All FP32-strict tensors (attn_res, cross-pass queries, router weight/bias, sconv weights,
controller forget/gain weights) are stored raw in FP32 (or BF16 for embed & controller down_proj).
"""
import argparse
import json
import os
import shutil
import sys
import time
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "convert"))
import pack as P  # noqa: E402
from layout import pack_mxfp8_codes, pack_nvfp4_codes, pack_scales  # noqa: E402

torch.set_grad_enabled(False)


def raw_f32(W, name: str, t: torch.Tensor):
    W.raw(name, t.float().numpy(), dtype="f32")


def raw_bf16(W, name: str, t: torch.Tensor):
    W.raw(name, t.to(torch.bfloat16).view(torch.uint16).numpy(), dtype="bf16")


def pad_rows_32(w: torch.Tensor) -> torch.Tensor:
    rem = w.shape[0] % 32
    if rem == 0:
        return w
    return torch.cat([w, torch.zeros(32 - rem, w.shape[1], dtype=w.dtype)], dim=0)


def w13_ckpt_to_epi3_perm(I: int) -> np.ndarray:
    """Checkpoint w13 rows are [g0, u0, g1, u1, ...].
    DankAttn EPI=3 expects fragments of [g_{16f..16f+15}, u_{16f..16f+15}]."""
    perm = []
    for f in range(I // 16):
        perm.extend(2 * (16 * f + j) for j in range(16))
        perm.extend(2 * (16 * f + j) + 1 for j in range(16))
    return np.array(perm, dtype=np.int64)


def pack_grouped_nvfp4(W, name: str, w_u8: np.ndarray, s_f8_t: torch.Tensor,
                       amax: np.ndarray, denom: float, row_perm: np.ndarray | None = None):
    """Pack [E, N, K/2] uint8 NVFP4 + [E, N, K/16] fp8 scales into DankAttn grouped WBITS=40, GS=16."""
    E, N, K2 = w_u8.shape
    K = K2 * 2
    cb = N * K // 2  # bytes per expert (4 bits/weight)
    total_cb = E * cb
    if W.cur is None or W.size + total_cb + P.PAGE > W.max_file:
        W._open()
    s_f32 = s_f8_t.float().numpy()  # [E, N, K/16]
    # Pack codes contiguously so expert e sits at off + e * cb
    pad = (-W.size) % P.PAGE
    if pad:
        W.cur.write(b"\0" * pad)
        W.size += pad
    fc, oc = len(W.files) - 1, W.size
    scs = []
    for e in range(E):
        we = w_u8[e]
        se = s_f32[e]
        if row_perm is not None:
            we = we[row_perm]
            se = se[row_perm]
        u4 = np.empty((N, K), dtype=np.uint8)
        u4[:, 0::2] = we & 0x0F
        u4[:, 1::2] = we >> 4
        b = pack_nvfp4_codes(u4).tobytes()
        W.cur.write(b)
        W.size += len(b)
        se_comb = se * float(amax[e] / denom)
        scs.append(pack_scales(se_comb).tobytes())
    sdata = b"".join(scs)
    fs, os_ = W.put(sdata)
    W.tensors[name] = dict(
        kind="w", N=N, K=K, bits=40, gs=16, experts=E,
        file=fc, off=oc, bytes=total_cb, sfile=fs, soff=os_, sbytes=len(sdata), rows={},
    )


def pack_grouped_mxfp8(W, name: str, w_u8: np.ndarray, s_e8m0: np.ndarray,
                       row_perm: np.ndarray | None = None):
    """Pack [E, N, K] fp8_e4m3fn + [E, N, K/32] uint8 E8M0 scales into DankAttn grouped WBITS=80, GS=32."""
    E, N, K = w_u8.shape
    cb = N * K  # bytes per expert (8 bits/weight)
    total_cb = E * cb
    if W.cur is None or W.size + total_cb + P.PAGE > W.max_file:
        W._open()
    pad = (-W.size) % P.PAGE
    if pad:
        W.cur.write(b"\0" * pad)
        W.size += pad
    fc, oc = len(W.files) - 1, W.size
    scs = []
    for e in range(E):
        we = w_u8[e]
        se = s_e8m0[e]
        if row_perm is not None:
            we = we[row_perm]
            se = se[row_perm]
        b = pack_mxfp8_codes(we).tobytes()
        W.cur.write(b)
        W.size += len(b)
        se_f32 = np.exp2(se.astype(np.float32) - 127.0)
        scs.append(pack_scales(se_f32).tobytes())
    sdata = b"".join(scs)
    fs, os_ = W.put(sdata)
    W.tensors[name] = dict(
        kind="w", N=N, K=K, bits=80, gs=32, experts=E,
        file=fc, off=oc, bytes=total_cb, sfile=fs, soff=os_, sbytes=len(sdata), rows={},
    )


def pack_mtp(src: str, W, mtp_bits: int = 6):
    mtp_path = os.path.join(src, "mtp.safetensors")
    if not os.path.exists(mtp_path):
        return False
    print(f"Packing DSpark MTP drafter ({mtp_path}, W{mtp_bits})...", flush=True)
    with safe_open(mtp_path, framework="pt", device="cpu") as f:
        raw_f32(W, "mtp.fuse_weight", f.get_tensor("fuse_weight"))
        raw_bf16(W, "mtp.mask_embed", f.get_tensor("mask_embed"))
        raw_f32(W, "mtp.norm", f.get_tensor("norm.weight"))
        for l in range(3):
            p = f"layers.{l}"
            dp = f"mtp.L{l}"
            raw_f32(W, f"{dp}.dflare_norm", f.get_tensor(f"{p}.dflare_norm.weight"))
            raw_f32(W, f"{dp}.attn_norm", f.get_tensor(f"{p}.input_layernorm.weight"))
            raw_f32(W, f"{dp}.mlp_norm", f.get_tensor(f"{p}.post_attention_layernorm.weight"))
            raw_f32(W, f"{dp}.q_norm", f.get_tensor(f"{p}.self_attn.q_norm.weight"))
            raw_f32(W, f"{dp}.k_norm", f.get_tensor(f"{p}.self_attn.k_norm.weight"))
            raw_f32(W, f"{dp}.k_norm_ctx", f.get_tensor(f"{p}.self_attn.k_norm_ctx.weight"))

            qkv = torch.cat([
                f.get_tensor(f"{p}.self_attn.q_proj.weight").float(),
                f.get_tensor(f"{p}.self_attn.k_proj.weight").float(),
                f.get_tensor(f"{p}.self_attn.v_proj.weight").float(),
            ], dim=0)
            W.linear(f"{dp}.attn.qkv", qkv, 8, 128)
            kv_ctx = torch.cat([
                f.get_tensor(f"{p}.self_attn.k_proj_ctx.weight").float(),
                f.get_tensor(f"{p}.self_attn.v_proj_ctx.weight").float(),
            ], dim=0)
            W.linear(f"{dp}.attn.kv_ctx", kv_ctx, 8, 128)
            W.linear(f"{dp}.attn.o", f.get_tensor(f"{p}.self_attn.o_proj.weight").float(), 8, 128)

            g = f.get_tensor(f"{p}.mlp.gate_proj.weight").float()
            u = f.get_tensor(f"{p}.mlp.up_proj.weight").float()
            gu = torch.cat([g, u], dim=0)[P.gu_rows(g.shape[0])]
            W.linear(f"{dp}.mlp.gu", gu, mtp_bits, 128)
            W.linear(f"{dp}.mlp.dn", f.get_tensor(f"{p}.mlp.down_proj.weight").float(), mtp_bits, 128)

        raw_f32(W, "mtp.mh.dflare_norm", f.get_tensor("markov_head.dflare_norm.weight"))
        raw_bf16(W, "mtp.mh.hidden_out.w", f.get_tensor("markov_head.hidden_out.weight"))
        raw_f32(W, "mtp.mh.hidden_out.b", f.get_tensor("markov_head.hidden_out.bias"))
        raw_bf16(W, "mtp.mh.rnn_proj.w", f.get_tensor("markov_head.rnn_proj.weight"))
        raw_f32(W, "mtp.mh.rnn_proj.b", f.get_tensor("markov_head.rnn_proj.bias"))
        raw_bf16(W, "mtp.mh.markov_w1", f.get_tensor("markov_head.markov_w1.weight"))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="Path to Inkling-2-Mini checkpoint directory")
    ap.add_argument("out", help="Output DankAttn pack directory")
    ap.add_argument("--mtp-bits", type=int, default=6, choices=[6, 8], help="Bits for DSpark MTP MLP (default 6)")
    ap.add_argument("--no-mtp", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    cfg = json.load(open(os.path.join(args.src, "config.json")))
    tc = cfg["text_config"]
    lc = tc["loop_config"]
    idx = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def get(name: str) -> torch.Tensor:
        fn = idx[name]
        if fn not in handles:
            handles[fn] = safe_open(os.path.join(args.src, fn), framework="pt", device="cpu")
        return handles[fn].get_tensor(name)

    W = P.Writer(args.out)

    # 1. Global / shared tensors
    raw_bf16(W, "embed", get("model.llm.embed.weight"))
    raw_f32(W, "embed_norm", get("model.llm.embed_norm.weight"))
    raw_f32(W, "norm", get("model.llm.norm.weight"))
    raw_f32(W, "attn_res.output_query", get("model.llm.attn_res.output_query"))

    # Merge cross-pass message queries to [12, 4, H] in (q, k, v, mlp) order
    cp_qm = get("model.llm.middle_cycle_cross_pass_queries").float().unflatten(0, (-1, 2))
    cp_kv = get("model.llm.middle_cycle_cross_pass_kv_queries").float().unflatten(0, (-1, 2))
    msg_q = torch.cat([cp_qm[:, :1], cp_kv, cp_qm[:, 1:]], dim=1).contiguous()
    raw_f32(W, "loop.msg_queries", msg_q)

    # Unembed (unpadded_vocab_size padded to multiple of 32 for mm2)
    unembed = pad_rows_32(get("model.llm.unembed.weight")[: tc["unpadded_vocab_size"]].float())
    W.linear("lm_head", unembed, 8, 128)

    I_moe = tc["intermediate_size"]
    w13_perm = w13_ckpt_to_epi3_perm(I_moe)
    local_set = set(tc["local_layer_ids"])

    # 2. 24 physical layers
    for l in range(tc["num_hidden_layers"]):
        lt0 = time.time()
        lp = f"model.llm.layers.{l}"
        dp = f"L{l}"
        is_local = l in local_set
        is_loop = lc["core_start"] <= l < lc["core_end"]

        if l > 0:
            raw_f32(W, f"{dp}.ar.attn_q", get(f"model.llm.attn_res.attn_queries.{l - 1}"))
        raw_f32(W, f"{dp}.ar.mlp_q", get(f"model.llm.attn_res.mlp_queries.{l}"))

        if is_loop:
            for c_src, c_dst in (("attn", "q"), ("k", "k"), ("v", "v"), ("mlp", "mlp")):
                raw_bf16(W, f"{dp}.ctrl.{c_dst}.down", get(f"{lp}.controllers.{c_src}.down_weight"))
                raw_f32(W, f"{dp}.ctrl.{c_dst}.forget", get(f"{lp}.controllers.{c_src}.forget_weight"))
                raw_f32(W, f"{dp}.ctrl.{c_dst}.gain", get(f"{lp}.controllers.{c_src}.gain_weight"))

        if is_local:
            raw_f32(W, f"{dp}.attn_norm", get(f"{lp}.attn_norm.weight"))
        else:
            raw_f32(W, f"{dp}.q_norm", get(f"{lp}.q_input_norm.weight"))
            raw_f32(W, f"{dp}.k_norm", get(f"{lp}.k_input_norm.weight"))
            raw_f32(W, f"{dp}.v_norm", get(f"{lp}.v_input_norm.weight"))
        raw_f32(W, f"{dp}.mlp_norm", get(f"{lp}.mlp_norm.weight"))

        # Fused Q (3072) | gate (3072) | rel (384) | sink (24 + 8 pad)
        qgrs = torch.cat([
            get(f"{lp}.attn.wq_du.weight").float(),
            get(f"{lp}.attn.gate_attn.weight").float(),
            get(f"{lp}.attn.wr_du.weight").float(),
            pad_rows_32(get(f"{lp}.attn.attn_sink_proj.weight").float()),
        ], dim=0)
        W.linear(f"{dp}.attn.qgrs", qgrs, 8, 128)

        if is_local:
            W.linear(f"{dp}.attn.k", get(f"{lp}.attn.wk_dv.weight").float(), 8, 128)
            W.linear(f"{dp}.attn.v", get(f"{lp}.attn.wv_dv.weight").float(), 8, 128)
            raw_f32(W, f"{dp}.attn.k_sconv", get(f"{lp}.attn.k_sconv.weight").squeeze(1))
            raw_f32(W, f"{dp}.attn.v_sconv", get(f"{lp}.attn.v_sconv.weight").squeeze(1))
        else:
            wk = torch.cat([get(f"{lp}.attn.wk_dv.weight").float(), get(f"{lp}.attn.k_sconv.weight_proj").float()], dim=0)
            wv = torch.cat([get(f"{lp}.attn.wv_dv.weight").float(), get(f"{lp}.attn.v_sconv.weight_proj").float()], dim=0)
            W.linear(f"{dp}.attn.k", wk, 8, 128)
            W.linear(f"{dp}.attn.v", wv, 8, 128)
            raw_f32(W, f"{dp}.attn.k_sconv", get(f"{lp}.attn.k_sconv.static_w"))
            raw_f32(W, f"{dp}.attn.v_sconv", get(f"{lp}.attn.v_sconv.static_w"))

        W.linear(f"{dp}.attn.o", get(f"{lp}.attn.wo_ud.weight").float(), 8, 128)
        raw_f32(W, f"{dp}.attn.q_scale", get(f"{lp}.attn.q_l2_clip.scale"))
        raw_f32(W, f"{dp}.attn.rel_scale", get(f"{lp}.attn.rel_l2_clip.scale"))
        raw_bf16(W, f"{dp}.attn.rel_proj", get(f"{lp}.attn.rel_logits_proj.proj"))
        raw_f32(W, f"{dp}.attn_sconv", get(f"{lp}.attn_sconv.weight").squeeze(1))
        raw_f32(W, f"{dp}.mlp_sconv", get(f"{lp}.mlp_sconv.weight").squeeze(1))

        raw_f32(W, f"{dp}.moe.router_w", get(f"{lp}.mlp.gate.weight"))
        raw_f32(W, f"{dp}.moe.router_b", get(f"{lp}.mlp.gate.bias"))

        if f"{lp}.mlp.experts.w13_weight_amax" in idx:
            pack_grouped_nvfp4(
                W, f"{dp}.moe.gu",
                get(f"{lp}.mlp.experts.w13_weight").numpy(),
                get(f"{lp}.mlp.experts.w13_weight_scale"),
                get(f"{lp}.mlp.experts.w13_weight_amax").float().numpy(),
                float(get(f"{lp}.mlp.experts.w13_weight_outer_scale_denominator").float().item()),
                row_perm=w13_perm,
            )
            pack_grouped_nvfp4(
                W, f"{dp}.moe.dn",
                get(f"{lp}.mlp.experts.w2_weight").numpy(),
                get(f"{lp}.mlp.experts.w2_weight_scale"),
                get(f"{lp}.mlp.experts.w2_weight_amax").float().numpy(),
                float(get(f"{lp}.mlp.experts.w2_weight_outer_scale_denominator").float().item()),
            )
            fmt = "NVFP4"
        else:
            pack_grouped_mxfp8(
                W, f"{dp}.moe.gu",
                get(f"{lp}.mlp.experts.w13_weight").view(torch.uint8).numpy(),
                get(f"{lp}.mlp.experts.w13_weight_scale").numpy(),
                row_perm=w13_perm,
            )
            pack_grouped_mxfp8(
                W, f"{dp}.moe.dn",
                get(f"{lp}.mlp.experts.w2_weight").view(torch.uint8).numpy(),
                get(f"{lp}.mlp.experts.w2_weight_scale").numpy(),
            )
            fmt = "MXFP8"

        print(f"  L{l:02d} ({'SWA' if is_local else 'GLB'}, {'loop' if is_loop else 'once'}, {fmt}): {time.time() - lt0:.1f}s", flush=True)

    has_mtp = False if args.no_mtp else pack_mtp(args.src, W, mtp_bits=args.mtp_bits)
    if W.cur:
        W.cur.close()

    for fn in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "generation_config.json"):
        sp = os.path.join(args.src, fn)
        if os.path.exists(sp):
            shutil.copy2(sp, os.path.join(args.out, fn))

    manifest = {
        "format": "dankattn-inkling2-v1",
        "arch": "inkling2_loop",
        "has_mtp": has_mtp,
        "text_config": tc,
        "mtp_config": cfg.get("mtp_config"),
        "eos_token_id": cfg.get("eos_token_id"),
        "files": W.files,
        "tensors": W.tensors,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    total_gb = sum(os.path.getsize(os.path.join(args.out, s)) for s in W.files) / 1e9
    print(f"Packed {args.out}: {total_gb:.2f} GB across {len(W.files)} shards in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
