"""PyTorch f32 reference for AngelSlim/Hy4-flash-DFly2-B6 (3L LatentMoE + AttentiveCorrection)
on a golden.npz dump (dense range T <= 2048).

  python3 hy4/ref/dfly2_b6_ref.py <target-bf16> <dfly2-b6-bf16> <golden.npz> [--causal]
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy4_golden import Weights, rms, rope_tables, apply_rope  # noqa: E402

torch.set_grad_enabled(False)


def run_ref(target_ckpt, dfly_dir, dump, causal=False, ac_on=True, anchor=None):
    TW = Weights(target_ckpt)
    cfg = json.load(open(os.path.join(dfly_dir, "config.json")))
    sf = safe_open(os.path.join(dfly_dir, "model.safetensors"), framework="pt")
    D = lambda k: sf.get_tensor(k).float()

    g = np.load(dump)
    T = len(g["ids"])
    assert T <= 2048
    h_src = torch.from_numpy(g["h_final"][T - 1]).float()
    if anchor is None:
        anchor = int(g["logits"][T - 1].argmax())
    eps = cfg["rms_norm_eps"]
    nope, rope_d, vd, kvr = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"], cfg["kv_lora_rank"]
    base = cfg["rope_parameters"]["rope_theta"]
    Wb = cfg["block_size"]
    mask = cfg["mask_token_id"]
    E_tok = TW.get("model.embed_tokens.weight", None)
    lm = TW.get("lm_head.weight")

    ids = torch.tensor([anchor] + [mask] * (Wb - 1))
    pos = torch.arange(T, T + Wb)
    x = E_tok[ids].float()
    e0 = rms(x[0], D("model.mtp_fusion.enorm.weight"), eps)
    hs = rms(h_src, D("model.mtp_fusion.hnorm.weight"), eps)
    x[0] = x[0] + D("model.mtp_fusion.gate") * (torch.cat([e0, hs]) @ D("model.mtp_fusion.eh_proj.weight").t())
    cos, sin = rope_tables(pos, rope_d, base)

    sw_lim = float(cfg.get("swiglu_limit", 10.0))
    rscale = float(cfg["router_scaling_factor"])
    topk = int(cfg["num_experts_per_tok"])

    for i, tl in enumerate(cfg["target_kv_layers"]):
        p = f"model.layers.{i}."
        a_ = p + "self_attn."
        m_ = p + "mlp."
        hin = rms(x, D(p + "input_layernorm.weight"), eps)
        nh = D(a_ + "learnable_sink_param").shape[0]
        gate = (hin @ D(a_ + "linear_gate.weight").t()).view(Wb, nh, vd)
        qr_ = rms(hin @ D(a_ + "q_a_proj.weight").t(), D(a_ + "q_a_layernorm.weight"), eps)
        qq = (qr_ @ D(a_ + "q_b_proj.weight").t()).view(Wb, nh, nope + rope_d).transpose(0, 1)
        q_pass, q_rot = qq[..., :nope], apply_rope(qq[..., nope:], cos, sin)
        ckv = hin @ D(a_ + "kv_a_proj_with_mqa.weight").t()
        kp = rms(ckv[:, :kvr], D(a_ + "kv_a_layernorm.weight"), eps)
        kr = apply_rope(ckv[:, kvr:], cos, sin)
        lat_hist = torch.from_numpy(g[f"L{tl}.latent"]).float()
        lat = torch.cat([lat_hist, torch.cat([kp, kr], -1)], 0)
        kvb = D(a_ + "kv_b_proj.weight").view(nh, nope + vd, kvr)
        k_nope = torch.einsum("hdc,tc->htd", kvb[:, :nope], lat[:, :kvr])
        v = torch.einsum("hdc,tc->htd", kvb[:, nope:], lat[:, :kvr])
        kf = torch.cat([k_nope, lat[None, :, kvr:].expand(nh, -1, -1)], -1)
        s = (torch.cat([q_pass, q_rot], -1) @ kf.transpose(-1, -2)) * (nope + rope_d) ** -0.5
        if causal:
            allowed = torch.ones(Wb, T + Wb, dtype=torch.bool)
            for r in range(Wb):
                allowed[r, T + r + 1:] = False
            s = s.masked_fill(~allowed, float("-inf"))
        sinks = D(a_ + "learnable_sink_param")
        s = torch.cat([s, sinks[:, None, None].expand(nh, Wb, 1)], -1)
        pr = torch.softmax(s - s.max(-1, keepdim=True).values, -1)[..., :-1]
        o = (pr @ v).transpose(0, 1) * torch.sigmoid(gate)
        o = o.reshape(Wb, -1) @ D(a_ + "o_proj.weight").t()
        x = x + rms(o, D(a_ + "post_proj_norm.weight"), eps)

        # LatentMoE
        hm = rms(x, D(p + "post_attention_layernorm.weight"), eps)
        sc = torch.sigmoid(hm @ D(m_ + "gate.weight").t())
        ch = sc + D(m_ + "expert_bias")
        topi = ch.topk(topk, dim=-1).indices
        topw = sc.gather(-1, topi)
        topw = topw / (topw.sum(-1, keepdim=True) + 1e-20) * rscale
        z = hm @ D(m_ + "routed_expert_down_proj.weight").t()  # [Wb, 1536]
        gu_w = D(m_ + "experts.gate_up_proj")  # [128, 3072, 1536]
        dn_w = D(m_ + "experts.down_proj")     # [128, 1536, 1536]
        z_out = torch.zeros_like(z)
        for r in range(Wb):
            for k in range(topk):
                e = int(topi[r, k])
                gu = z[r] @ gu_w[e].t()
                g_, u_ = gu.chunk(2, dim=-1)
                act = F.silu(g_.clamp(-sw_lim, sw_lim)) * u_.clamp(-sw_lim, sw_lim)
                z_out[r] += topw[r, k] * (act @ dn_w[e].t())
        routed = z_out @ D(m_ + "routed_expert_up_proj.weight").t()
        sh = (F.silu(hm @ D(m_ + "shared_experts.gate_proj.weight").t()) * (hm @ D(m_ + "shared_experts.up_proj.weight").t())) @ D(m_ + "shared_experts.down_proj.weight").t()
        x = x + rms(routed + sh, D(m_ + "combine_output_norm.weight"), eps)

    hd = rms(x, D("model.norm.weight"), eps)
    hc = "model.hidden_correction."
    ac = hc + "attention."
    lat_ac = torch.from_numpy(g[f"L{cfg['target_kv_layers'][-1]}.latent"]).float()  # [T, 576]
    ac_nh = D(ac + "learnable_sink_param").shape[0]
    ac_kvb = D(ac + "kv_b_proj.weight").view(ac_nh, nope + vd, kvr)
    w_kc, w_vc = ac_kvb[:, :nope], ac_kvb[:, nope:]
    ac_sinks = D(ac + "learnable_sink_param")
    ac_bsink = D(ac + "block_sink")
    ac_brel = D(ac + "block_rel_bias")
    ac_bgbias = D(ac + "block_gate_bias")

    # Slot 0 block key
    def make_blk(slot_i, prev_tok):
        cat = torch.cat([rms(hd[slot_i], D(hc + "hidden_norm.weight"), eps), rms(E_tok[prev_tok].float(), D(hc + "embed_norm.weight"), eps)])
        kv = cat @ D(ac + "kv_in.weight").t()
        kp = rms(kv[:kvr], D(ac + "kv_a_layernorm.weight"), eps)
        kr = apply_rope(kv[kvr:].view(1, 1, rope_d), cos[slot_i:slot_i+1], sin[slot_i:slot_i+1]).view(rope_d)
        return cat, torch.cat([kp, kr], -1)

    _, blk0 = make_blk(0, anchor)
    blks = [blk0]
    prev, drafts, probs = anchor, [], []
    for st in range(Wb - 1):
        slot_i = st + 1
        cat, blki = make_blk(slot_i, prev)
        blks.append(blki)
        delta = (F.silu(cat @ D(hc + "gate_proj.weight").t()) * (cat @ D(hc + "up_proj.weight").t())) @ D(hc + "down_proj.weight").t()
        if ac_on:
            qa = rms(cat @ D(ac + "q_in.weight").t(), D(ac + "q_a_layernorm.weight"), eps)
            qq = (qa @ D(ac + "q_b_proj.weight").t()).view(ac_nh, nope + rope_d)
            q_nope = qq[:, :nope]
            q_pe = apply_rope(qq[:, nope:].unsqueeze(1), cos[slot_i:slot_i+1], sin[slot_i:slot_i+1]).squeeze(1)
            q_lat = torch.einsum("hd,hdc->hc", q_nope, w_kc)  # [nh, 512]
            q_full = torch.cat([q_lat, q_pe], -1)  # [nh, 576]
            # Context attention
            sc_ctx = (q_full @ lat_ac.t()) * (nope + rope_d) ** -0.5  # [nh, T]
            sc_ctx = torch.cat([sc_ctx, ac_sinks[:, None]], -1)
            pr_ctx = torch.softmax(sc_ctx - sc_ctx.max(-1, keepdim=True).values, -1)[:, :-1]
            read = pr_ctx @ lat_ac[:, :kvr]  # [nh, 512]
            # Gated block attention
            B_mat = torch.stack(blks, 0)  # [slot_i+1, 576]
            sc_b = (q_full @ B_mat.t()) * (nope + rope_d) ** -0.5  # [nh, slot_i+1]
            for j in range(slot_i + 1):
                sc_b[:, j] += ac_brel[:, slot_i - j]
            sc_b = torch.cat([sc_b, ac_bsink[:, None]], -1)
            pr_b = torch.softmax(sc_b - sc_b.max(-1, keepdim=True).values, -1)[:, :-1]
            blk_read = pr_b @ B_mat[:, :kvr]
            g_b = torch.tanh(cat @ D(ac + "block_gate.weight").t() + ac_bgbias[:, slot_i])
            read = read + g_b[:, None] * blk_read
            out_v = torch.einsum("hc,hdc->hd", read, w_vc).reshape(-1)
            delta = delta + out_v @ D(ac + "o_proj.weight").t()
        lg = (hd[slot_i] + delta) @ lm.t()
        p_ = torch.softmax(lg, -1)
        prev = int(lg.argmax())
        drafts.append(prev)
        probs.append(round(float(p_[prev]), 4))
    print(f"causal={causal} ac={ac_on} | anchor={anchor} | drafts={drafts} | probs={probs}")
    return drafts


if __name__ == "__main__":
    tc, df, dm = sys.argv[1], sys.argv[2], sys.argv[3]
    run_ref(tc, df, dm, causal=False, ac_on=True)
    run_ref(tc, df, dm, causal=True, ac_on=True)
    run_ref(tc, df, dm, causal=False, ac_on=False)
