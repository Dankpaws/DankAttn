"""DFly2 draft reference (vLLM PR #68296 semantics, f32 torch, BF16 weights) on a golden dump.

  python3 dfly_ref.py <bf16 ckpt> <golden.npz with --dump> [anchor token]

Prompt = the dump's ids (T tokens, dense range only: T <= 2048), source row T - 1, anchor at position
T (default: the golden's argmax at row T - 1). Prints the 5 greedy drafts.
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy4_golden import Weights, rms, rope_tables, apply_rope  # noqa: E402

torch.set_grad_enabled(False)


def main(ckpt, dump, anchor=None):
    W = Weights(ckpt)
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    tc = cfg["text_config"]
    g = np.load(dump)
    T = len(g["ids"])
    assert T <= 2048, "reference covers the dense range only"
    h_src = torch.from_numpy(g["h_final"][T - 1]).float()
    if anchor is None:
        anchor = int(g["logits"][T - 1].argmax())
    eps = tc["rms_norm_eps"]
    nope, rope_d, vd, kvr = tc["qk_nope_head_dim"], tc["qk_rope_head_dim"], tc["v_head_dim"], tc["kv_lora_rank"]
    base = tc["rope_parameters"]["deepseek_sparse_attention"]["rope_theta"]
    Wb = cfg["speculative_config"]["num_speculative_tokens"] + 1
    mask = cfg["dfly_mask_token_id"]
    E = W.get("model.embed_tokens.weight", None)
    ids = torch.tensor([anchor] + [mask] * (Wb - 1))
    pos = torch.arange(T, T + Wb)
    x = E[ids].float()
    q = "model.dfly_layers."
    e0 = rms(x[0], W.get(q + "mtp_fusion.enorm.weight"), eps)
    hs = rms(h_src, W.get(q + "mtp_fusion.hnorm.weight"), eps)
    x[0] = x[0] + W.get(q + "mtp_fusion.gate") * (torch.cat([e0, hs]) @ W.get(q + "mtp_fusion.eh_proj.weight").t())
    cos, sin = rope_tables(pos, rope_d, base)
    for i, tl in enumerate(cfg["dfly_target_kv_layers"]):
        p = f"{q}{i}."
        a_ = p + "self_attn."
        hin = rms(x, W.get(p + "input_layernorm.weight"), eps)
        nh = W.get(a_ + "learnable_sink_param").shape[0]
        gate = (hin @ W.get(a_ + "linear_gate.weight").t()).view(Wb, nh, vd)
        qr_ = rms(hin @ W.get(a_ + "q_a_proj.weight").t(), W.get(a_ + "q_a_layernorm.weight"), eps)
        qq = (qr_ @ W.get(a_ + "q_b_proj.weight").t()).view(Wb, nh, nope + rope_d).transpose(0, 1)
        q_pass, q_rot = qq[..., :nope], apply_rope(qq[..., nope:], cos, sin)
        ckv = hin @ W.get(a_ + "kv_a_proj_with_mqa.weight").t()
        kp = rms(ckv[:, :kvr], W.get(a_ + "kv_a_layernorm.weight"), eps)
        kr = apply_rope(ckv[:, kvr:], cos, sin)
        lat_hist = torch.from_numpy(g[f"L{tl}.latent"]).float()  # [T, 576] target latents (normed | roped)
        lat = torch.cat([lat_hist, torch.cat([kp, kr], -1)], 0)  # history then the block
        kvb = W.get(a_ + "kv_b_proj.weight").view(nh, nope + vd, kvr)
        k_nope = torch.einsum("hdc,tc->htd", kvb[:, :nope], lat[:, :kvr])
        v = torch.einsum("hdc,tc->htd", kvb[:, nope:], lat[:, :kvr])
        kf = torch.cat([k_nope, lat[None, :, kvr:].expand(nh, -1, -1)], -1)
        s = (torch.cat([q_pass, q_rot], -1) @ kf.transpose(-1, -2)) * (nope + rope_d) ** -0.5  # [nh, Wb, T + Wb]
        allowed = torch.ones(Wb, T + Wb, dtype=torch.bool)
        for r in range(Wb):
            allowed[r, T + r + 1:] = False
        s = s.masked_fill(~allowed, float("-inf"))
        sinks = W.get(a_ + "learnable_sink_param")
        s = torch.cat([s, sinks[:, None, None].expand(nh, Wb, 1)], -1)
        pr = torch.softmax(s - s.max(-1, keepdim=True).values, -1)[..., :-1]
        o = (pr @ v).transpose(0, 1) * torch.sigmoid(gate)
        o = o.reshape(Wb, -1) @ W.get(a_ + "o_proj.weight").t()
        x = x + rms(o, W.get(a_ + "post_proj_norm.weight"), eps)
        m = rms(x, W.get(p + "post_attention_layernorm.weight"), eps)
        x = x + (F.silu(m @ W.get(p + "mlp.gate_proj.weight").t()) * (m @ W.get(p + "mlp.up_proj.weight").t())) @ W.get(p + "mlp.down_proj.weight").t()
    hd = rms(x, W.get(q + "final_norm.weight"), eps)
    lm = W.get("lm_head.weight")
    hc = q + "hidden_correction."
    prev, drafts = anchor, []
    for st in range(Wb - 1):
        cat = torch.cat([rms(hd[st + 1], W.get(hc + "hidden_norm.weight"), eps), rms(E[prev].float(), W.get(hc + "embed_norm.weight"), eps)])
        delta = (F.silu(cat @ W.get(hc + "gate_proj.weight").t()) * (cat @ W.get(hc + "up_proj.weight").t())) @ W.get(hc + "down_proj.weight").t()
        lg = (hd[st + 1] + delta) @ lm.t()
        prev = int(lg.argmax())
        drafts.append(prev)
    print("anchor", anchor, "drafts", drafts)
    return drafts


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
