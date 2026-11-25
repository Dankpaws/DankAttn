#!/usr/bin/env python3
"""Fast, zero-swap 4-arm 2-bit quantization fidelity sweep for ERNIE-5.5-Flash-Preview.

Processes 1 expert at a time (L3-cache friendly, ~27 ms/matrix on CPU) over T=144
wikitext-2 tokens (crossing the HCA C128 compression boundary at t=128), measuring
exact NLL, PPL, KL divergence, and Top-1 agreement vs native MXFP4+FP8.
"""
import argparse
import json
import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ernie55_golden as G

torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count() or 16)


def fast_w2_nf2(w: torch.Tensor, clip_r: float = 0.78) -> torch.Tensor:
    g = w.reshape(2048, 32, 64)
    s = (g.abs().amax(-1, keepdim=True).clamp_min(1e-12) * clip_r).half().float().clamp_min(6e-8)
    xn = g / s
    sgn = torch.where(xn >= 0, 1.0, -1.0)
    q = torch.where(xn.abs() > 0.6499, sgn, sgn * 0.2998)
    return (q * s).reshape(2048, 2048)


def fast_w2_sym(w: torch.Tensor, clip_r: float = 0.80) -> torch.Tensor:
    g = w.reshape(2048, 32, 64)
    s = (g.abs().amax(-1, keepdim=True).clamp_min(1e-12) * clip_r).half().float().clamp_min(6e-8)
    xn = g / s
    sgn = torch.where(xn >= 0, 1.0, -1.0)
    q = torch.where(xn.abs() > 0.6667, sgn, sgn * (1.0 / 3.0))
    return (q * s).reshape(2048, 2048)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to ERNIE-5.5-Flash-Preview checkpoint")
    ap.add_argument("--wikitext", required=True, help="Path to wikitext-2 wiki.test.raw")
    ap.add_argument("--out_dir", default="./ernie55_eval")
    ap.add_argument("--tokens", type=int, default=144)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(args.ckpt, "tokenizer.json"))

    text = open(args.wikitext, encoding="utf-8", errors="ignore").read()[:8000]
    token_ids = [1] + tok.encode(text).ids[: args.tokens - 1]
    T = len(token_ids)
    np.array(token_ids, dtype=np.uint32).tofile(os.path.join(args.out_dir, f"eval_{T}.bin"))
    json.dump(token_ids, open(os.path.join(args.out_dir, f"eval_{T}.json"), "w"))

    arms = [
        ("ref_mxfp4",      "none", set()),
        ("w2_g64_nf2",     "nf2",  set()),
        ("w2_g64_sym",     "sym",  set()),
        ("w2_g64_nf2_hyb", "nf2",  {1, 2, 42}),
    ]
    print(f"Running {len(arms)}-arm fast sweep on T={T} wikitext-2 tokens...", flush=True)

    loader = G.CkptLoader(args.ckpt, device="cpu")
    cfg = loader.cfg
    dev = loader.device
    eps = cfg["rms_norm_eps"]
    hc_eps = cfg["hc_eps"]
    hc_mult = cfg["hc_mult"]
    sk_iters = cfg["hc_sinkhorn_iters"]
    n_layers = cfg["num_hidden_layers"]
    positions = torch.arange(T, dtype=torch.long, device=dev)

    rs = cfg["rope_scaling"]
    cos_hca, sin_hca = G.build_yarn_cos_sin(
        T + 16, cfg["qk_rope_head_dim"], cfg["compress_rope_theta"],
        rs["factor"], rs["original_max_position_embeddings"],
        rs["beta_fast"], rs["beta_slow"], dev,
    )
    cos_mla, sin_mla = G.build_plain_cos_sin(T + 16, cfg["hybrid_mla_qk_rope_head_dim"], cfg["rope_theta"], dev)

    emb_w = loader.raw("model.embed_tokens.weight", to_dev=False)
    h0 = emb_w[torch.tensor(token_ids, dtype=torch.long)].float()
    del emb_w
    res_base = h0.unsqueeze(1).expand(T, hc_mult, -1).contiguous()
    residuals = {name: res_base.clone() for name, _, _ in arms}

    t0 = time.time()
    for lid in range(n_layers):
        lt0 = time.time()
        lpfx = f"model.layers.{lid}"
        is_dsa = cfg["compress_ratios"][lid] <= 0
        is_moe = lid >= cfg["first_k_dense_replace"]

        ahc_fn = loader.raw(f"{lpfx}.self_attention_hyper_connection.mapping_proj.weight")
        ahc_base = loader.raw(f"{lpfx}.self_attention_hyper_connection.bias")
        ahc_scale = torch.cat([
            loader.raw(f"{lpfx}.self_attention_hyper_connection.alpha_pre"),
            loader.raw(f"{lpfx}.self_attention_hyper_connection.alpha_post"),
            loader.raw(f"{lpfx}.self_attention_hyper_connection.alpha_res"),
        ])
        attn_nw = loader.raw(f"{lpfx}.input_layernorm.weight")
        mhc_fn = loader.raw(f"{lpfx}.mlp_hyper_connection.mapping_proj.weight")
        mhc_base = loader.raw(f"{lpfx}.mlp_hyper_connection.bias")
        mhc_scale = torch.cat([
            loader.raw(f"{lpfx}.mlp_hyper_connection.alpha_pre"),
            loader.raw(f"{lpfx}.mlp_hyper_connection.alpha_post"),
            loader.raw(f"{lpfx}.mlp_hyper_connection.alpha_res"),
        ])
        ffn_nw = loader.raw(f"{lpfx}.post_attention_layernorm.weight")

        ffn_in, ffn_post = {}, {}
        for name, _, _ in arms:
            if lid == 0 and name != arms[0][0]:
                continue
            res = residuals[name]
            post_mix, comb, x = G.mhc_pre(res, ahc_fn, ahc_scale, ahc_base, eps, hc_eps, sk_iters)
            x = G.rms_norm(x, attn_nw, eps)
            if is_dsa:
                attn_out = G.run_dsa_layer(loader, lid, x, positions, cos_mla, sin_mla, cfg)
            else:
                attn_out = G.run_hca_layer(loader, lid, x, positions, cos_hca, sin_hca, cfg)
            res = G.mhc_post(attn_out, res, post_mix, comb)

            post_mix2, comb2, x2 = G.mhc_pre(res, mhc_fn, mhc_scale, mhc_base, eps, hc_eps, sk_iters)
            x2 = G.rms_norm(x2, ffn_nw, eps)
            residuals[name] = res
            ffn_in[name] = x2
            ffn_post[name] = (post_mix2, comb2)

        if not is_moe:
            ffn_out0 = G.run_dense_ffn(loader, lid, ffn_in[arms[0][0]], cfg)
            res0 = G.mhc_post(ffn_out0, residuals[arms[0][0]], *ffn_post[arms[0][0]])
            for name, _, _ in arms:
                residuals[name] = res0.clone()
            n_act = 0
        else:
            mpfx = f"{lpfx}.block_sparse_moe"
            wg0 = loader.raw(f"{mpfx}.gate.weight").float()
            wg1 = loader.raw(f"{mpfx}.gate.weight_1").float()
            bias = loader.raw(f"{mpfx}.e_score_correction_bias").float()
            rscale = loader.raw(f"{mpfx}.gate.routed_scaling_factor_param").float()
            w_fc1 = loader.linear_weight(f"{mpfx}.fc1_latent_proj")
            w_fc2 = loader.linear_weight(f"{mpfx}.fc2_latent_proj")
            lnorm_w = loader.raw(f"{mpfx}.latent_norm.weight")
            sw1 = loader.linear_weight(f"{mpfx}.shared_experts.w1")
            sw3 = loader.linear_weight(f"{mpfx}.shared_experts.w3")
            sw2 = loader.linear_weight(f"{mpfx}.shared_experts.w2")
            beta = cfg["activation_situ_beta"]
            lbeta = cfg["activation_situ_linear_beta"]
            topk = cfg["num_experts_per_tok"]

            arm_lat, arm_ids, arm_w, arm_lat_out = {}, {}, {}, {}
            union_e = set()
            for name, _, _ in arms:
                x = ffn_in[name]
                sc = torch.sigmoid(x @ wg0.t()) + torch.sigmoid(x @ wg1.t())
                ids = torch.topk(sc + bias.unsqueeze(0), topk, dim=-1, sorted=False).indices
                w = sc.gather(1, ids)
                w = (w / w.sum(dim=-1, keepdim=True).clamp_min(1e-20)) * rscale[ids]
                lat = x @ w_fc1.t()
                arm_lat[name] = lat
                arm_ids[name] = ids
                arm_w[name] = w
                arm_lat_out[name] = torch.zeros_like(lat)
                union_e.update(torch.unique(ids).tolist())

            n_act = len(union_e)
            for eid in sorted(union_e):
                epfx = f"{mpfx}.experts.{eid}"
                w1_ref = loader.expert_weight(f"{epfx}.w1")
                w3_ref = loader.expert_weight(f"{epfx}.w3")
                w2_ref = loader.expert_weight(f"{epfx}.w2")
                qcache = {"none": (w1_ref, w3_ref, w2_ref)}

                for name, qmode, keep_w4 in arms:
                    tok_idx, slot_idx = torch.where(arm_ids[name] == eid)
                    if tok_idx.numel() == 0:
                        continue
                    eff = "none" if lid in keep_w4 else qmode
                    if eff not in qcache:
                        fn = fast_w2_nf2 if eff == "nf2" else fast_w2_sym
                        qcache[eff] = (fn(w1_ref), fn(w3_ref), fn(w2_ref))
                    w1, w3, w2 = qcache[eff]
                    inp = arm_lat[name][tok_idx]
                    act = G.situ_and_mul(inp @ w1.t(), inp @ w3.t(), beta, lbeta)
                    out_e = act @ w2.t()
                    arm_lat_out[name].index_add_(0, tok_idx, out_e * arm_w[name][tok_idx, slot_idx].unsqueeze(-1))

            for name, _, _ in arms:
                x = ffn_in[name]
                routed_out = G.rms_norm(arm_lat_out[name], lnorm_w, eps) @ w_fc2.t()
                shared_out = G.situ_and_mul(x @ sw1.t(), x @ sw3.t(), beta, lbeta) @ sw2.t()
                residuals[name] = G.mhc_post(routed_out + shared_out, residuals[name], *ffn_post[name])

        if lid in (0, 1, 2, 7, 14, 21, 28, 35, 42) or (lid + 1) % 5 == 0:
            print(f"  L{lid:02d}/{n_layers-1} ({'DSA' if is_dsa else 'HCA'}, E={n_act:3d}) "
                  f"std_ref={residuals['ref_mxfp4'].std():.3f} "
                  f"std_nf2={residuals['w2_g64_nf2'].std():.3f} "
                  f"({time.time()-lt0:.1f}s, tot {time.time()-t0:.0f}s)", flush=True)

    h_fn = loader.raw("model.mhc_contract.hc_head_fn")
    h_base = loader.raw("model.mhc_contract.hc_head_base")
    h_scale = loader.raw("model.mhc_contract.hc_head_scale")
    norm_w = loader.raw("model.norm.weight")
    lm_w = loader.linear_weight("lm_head")
    mm_r = loader.raw("model.lm_head.multimax_ranges").float()
    mm_t = loader.raw("model.lm_head.multimax_ts").float()

    targets = torch.tensor(token_ids[1:], dtype=torch.long)
    logits_map = {}
    for name, _, _ in arms:
        h = G.mhc_head_contract(residuals[name], h_fn, h_scale, h_base, eps, hc_eps)
        h = G.rms_norm(h, norm_w, eps)
        lg = h @ lm_w.t()
        lg = (
            lg
            + mm_t[0] * F.relu(mm_r[0] - lg)
            + mm_t[1] * F.relu(lg - mm_r[1])
            + mm_t[2] * F.relu(mm_r[2] - lg).square()
            + mm_t[3] * F.relu(lg - mm_r[3]).square()
        )
        logits_map[name] = lg

    ref_lg = logits_map["ref_mxfp4"]
    torch.save({"token_ids": token_ids, "logits": ref_lg.half()}, os.path.join(args.out_dir, f"ref_{T}.pt"))
    p_ref = torch.softmax(ref_lg, dim=-1)
    log_p_ref = torch.log_softmax(ref_lg, dim=-1)
    ref_nll = F.cross_entropy(ref_lg[:-1], targets).item()

    print(f"\n================ 2-BIT QUANTIZATION SWEEP (T={T} wikitext-2) ================")
    print(f"{'Arm':<18} {'NLL':>7} {'PPL':>7} {'dNLL':>7} {'MeanKL':>8} {'MedKL':>8} {'Top1%':>7}")
    print("-" * 68)
    summary = {}
    for name, _, _ in arms:
        lg = logits_map[name]
        nll = F.cross_entropy(lg[:-1], targets).item()
        log_p_q = torch.log_softmax(lg, dim=-1)
        kl = (p_ref * (log_p_ref - log_p_q)).sum(dim=-1)
        top1 = (ref_lg.argmax(-1) == lg.argmax(-1)).float().mean().item() * 100.0
        print(f"{name:<18} {nll:7.4f} {math.exp(nll):7.2f} {nll-ref_nll:+7.4f} "
              f"{kl.mean().item():8.4f} {kl.median().item():8.4f} {top1:6.2f}%")
        summary[name] = dict(nll=nll, ppl=math.exp(nll), dnll=nll - ref_nll,
                             mean_kl=kl.mean().item(), med_kl=kl.median().item(), top1=top1)
    json.dump(summary, open(os.path.join(args.out_dir, "w2_sweep_summary.json"), "w"), indent=2)
    print("Saved summary to", os.path.join(args.out_dir, "w2_sweep_summary.json"))


if __name__ == "__main__":
    main()
