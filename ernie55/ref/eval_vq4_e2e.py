#!/usr/bin/env python3
"""End-to-end VQ4 vs NF2 2-bit expert quant for ERNIE-5.5-Flash-Preview (CPU).

VQ4: one global 256-entry x 4-D bf16 codebook (8-bit index per 4 weights = 2.0 bpw),
per-group RMS normalisation, least-squares f16 group scale. NF2: RTN13 clip search,
inner level 0.2998 (same as eval_gs32_codebook_sweep.py). All arms keep L1 and L42
at native MXFP4. Scored against the saved native reference ref_144.pt.

Codebook: k-means++ / Lloyd on RMS-normalised 4-tuples from expert 0 (w1, w3, w2)
of layers 2, 10, 21, 30, 42. Expert 0 is still evaluated if routed; it is <0.2%
of expert weight.
"""
import json
import math
import os
import sys
import time

# glibc returns the 16 MB temporaries to the OS and page-faults them back on every call:
# 3-5x slowdown after warm-up (measured 2026-11-27, gpu-desktop CPU). Pin them in-process.
if "MALLOC_MMAP_THRESHOLD_" not in os.environ:
    os.environ["MALLOC_MMAP_THRESHOLD_"] = os.environ["MALLOC_TRIM_THRESHOLD_"] = str(16 << 30)
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ernie55_golden as G

torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count() or 16)

CKPT = "/mnt/lmcache-nvme/ernie55/hf-mxfp4"
EVAL = "/mnt/lmcache-nvme/ernie55/eval"
CLIPS_13 = (1.0, 0.96, 0.92, 0.88, 0.84, 0.80, 0.76, 0.72, 0.68, 0.64, 0.60, 0.56, 0.52)
KEEP_W4 = {1, 42}
DEV = sys.argv[1] if len(sys.argv) > 1 else "cpu"  # e.g. cuda:0


def rtn_w2_nf2(w, gs, inner=0.2998):
    N, K = w.shape
    mid = 0.5 * (1.0 + inner)
    g = w.reshape(N, K // gs, gs)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    best_wq = best_err = None
    for r in CLIPS_13:
        s = (amax * r).half().float().clamp_min(6e-8)
        xn = g / s
        sgn = torch.where(xn >= 0, 1.0, -1.0)
        wq = torch.where(xn.abs() > mid, sgn, sgn * inner) * s
        err = (wq - g).square().sum(-1)
        if best_err is None:
            best_err, best_wq = err, wq
        else:
            m = err < best_err
            best_err = torch.where(m, err, best_err)
            best_wq = torch.where(m.unsqueeze(-1), wq, best_wq)
    return best_wq.reshape(N, K)


def norm_tuples(w, gs):
    N, K = w.shape
    g = w.reshape(N, K // gs, gs)
    rms = g.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
    return g, (g / rms).reshape(-1, 4).contiguous()


def train_vq4(samples, n_iters=30, seed=42):
    rng = np.random.default_rng(seed)
    M = samples.shape[0]
    sub_n = min(M, 65536)
    sub = samples[rng.choice(M, size=sub_n, replace=False)]
    C = np.empty((256, 4), dtype=np.float32)
    C[0] = sub[rng.integers(sub_n)]
    d2 = np.sum((sub - C[0]) ** 2, axis=1)
    for k in range(1, 256):
        C[k] = sub[rng.choice(sub_n, p=d2 / max(d2.sum(), 1e-12))]
        d2 = np.minimum(d2, np.sum((sub - C[k]) ** 2, axis=1))
    for _ in range(n_iters):
        assign = np.argmin(np.sum(C ** 2, axis=1)[None, :] - 2.0 * (sub @ C.T), axis=1)
        for k in range(256):
            m = assign == k
            if np.any(m):
                C[k] = sub[m].mean(axis=0)
    return torch.from_numpy(C).bfloat16().float()


def quant_vq4(w, gs, C, half_c2, bs=4096):
    N, K = w.shape
    g, gn = norm_tuples(w, gs)
    CT = C.t().contiguous()
    idx = torch.empty(gn.shape[0], dtype=torch.long, device=gn.device)
    for i in range(0, gn.shape[0], bs):  # cache-sized chunks: ~6x faster than 128k
        idx[i:i + bs] = (gn[i:i + bs] @ CT - half_c2).argmax(-1)
    rec = C[idx].reshape(N, K // gs, gs)
    s = ((g * rec).sum(-1, keepdim=True) / rec.square().sum(-1, keepdim=True).clamp_min(1e-12))
    return (rec * s.half().float().clamp_min(6e-8)).reshape(N, K)


def main():
    ref = torch.load(f"{EVAL}/ref_144.pt", map_location="cpu")
    token_ids = ref["token_ids"]
    ref_lg = ref["logits"].float().to(DEV)
    T = len(token_ids)
    loader = G.CkptLoader(CKPT, device=DEV)
    cfg = loader.cfg

    t0 = time.time()
    samp = {32: [], 64: []}
    for lid in (2, 10, 21, 30, 42):
        for part in ("w1", "w3", "w2"):
            w = loader.expert_weight(f"model.layers.{lid}.block_sparse_moe.experts.0.{part}").float()
            for gs in samp:
                gn = norm_tuples(w, gs)[1]
                samp[gs].append(gn[:: max(1, gn.shape[0] // 8192)].cpu().numpy())
    CB = {gs: train_vq4(np.concatenate(v)) for gs, v in samp.items()}
    torch.save({f"C{gs}": c for gs, c in CB.items()}, f"{EVAL}/vq4_codebooks_e2e.pt")
    CB = {gs: c.to(DEV) for gs, c in CB.items()}
    HC2 = {gs: c.square().sum(-1) / 2 for gs, c in CB.items()}
    print(f"codebooks trained in {time.time()-t0:.1f}s -> {EVAL}/vq4_codebooks_e2e.pt", flush=True)

    # name -> quantizer(w) ; quantizers shared across arms via cache key = name
    arms = {
        # 256-row blocks stay in cache: bit-identical, ~2.5x faster than whole-matrix
        "g64_nf2_L1_42": lambda w: torch.cat([rtn_w2_nf2(w[i:i + 256], 64) for i in range(0, w.shape[0], 256)]),
        "g64_vq4_L1_42": lambda w: quant_vq4(w, 64, CB[64], HC2[64]),
        "g32_vq4_L1_42": lambda w: quant_vq4(w, 32, CB[32], HC2[32]),
    }
    names = list(arms)

    eps, hc_eps = cfg["rms_norm_eps"], cfg["hc_eps"]
    sk_iters, n_layers = cfg["hc_sinkhorn_iters"], cfg["num_hidden_layers"]
    positions = torch.arange(T, dtype=torch.long, device=DEV)
    rs = cfg["rope_scaling"]
    cos_hca, sin_hca = G.build_yarn_cos_sin(
        T + 16, cfg["qk_rope_head_dim"], cfg["compress_rope_theta"],
        rs["factor"], rs["original_max_position_embeddings"], rs["beta_fast"], rs["beta_slow"], DEV)
    cos_mla, sin_mla = G.build_plain_cos_sin(T + 16, cfg["hybrid_mla_qk_rope_head_dim"], cfg["rope_theta"], DEV)

    emb_w = loader.raw("model.embed_tokens.weight", to_dev=False)
    h0 = emb_w[torch.tensor(token_ids, dtype=torch.long)].float().to(DEV)
    del emb_w
    res_base = h0.unsqueeze(1).expand(T, cfg["hc_mult"], -1).contiguous()
    residuals = {n: res_base.clone() for n in names}

    def hc(lpfx, kind):
        p = f"{lpfx}.{kind}_hyper_connection"
        return (loader.raw(f"{p}.mapping_proj.weight"),
                torch.cat([loader.raw(f"{p}.alpha_pre"), loader.raw(f"{p}.alpha_post"), loader.raw(f"{p}.alpha_res")]),
                loader.raw(f"{p}.bias"))

    t0 = time.time()
    for lid in range(n_layers):
        lt0 = time.time()
        lpfx = f"model.layers.{lid}"
        is_dsa = cfg["compress_ratios"][lid] <= 0
        a_fn, a_sc, a_b = hc(lpfx, "self_attention")
        m_fn, m_sc, m_b = hc(lpfx, "mlp")
        attn_nw = loader.raw(f"{lpfx}.input_layernorm.weight")
        ffn_nw = loader.raw(f"{lpfx}.post_attention_layernorm.weight")
        # Layers before the first quantized layer are identical across arms: compute once.
        run_names = names if lid > 1 else names[:1]

        ffn_in, ffn_post = {}, {}
        for n in run_names:
            pm, cb, x = G.mhc_pre(residuals[n], a_fn, a_sc, a_b, eps, hc_eps, sk_iters)
            x = G.rms_norm(x, attn_nw, eps)
            if is_dsa:
                ao = G.run_dsa_layer(loader, lid, x, positions, cos_mla, sin_mla, cfg)
            else:
                ao = G.run_hca_layer(loader, lid, x, positions, cos_hca, sin_hca, cfg)
            residuals[n] = G.mhc_post(ao, residuals[n], pm, cb)
            pm2, cb2, x2 = G.mhc_pre(residuals[n], m_fn, m_sc, m_b, eps, hc_eps, sk_iters)
            ffn_in[n] = G.rms_norm(x2, ffn_nw, eps)
            ffn_post[n] = (pm2, cb2)

        n_act = 0
        if lid < cfg["first_k_dense_replace"]:
            out = G.run_dense_ffn(loader, lid, ffn_in[names[0]], cfg)
            r0 = G.mhc_post(out, residuals[names[0]], *ffn_post[names[0]])
        else:
            mp = f"{lpfx}.block_sparse_moe"
            wg0 = loader.raw(f"{mp}.gate.weight").float()
            wg1 = loader.raw(f"{mp}.gate.weight_1").float()
            bias = loader.raw(f"{mp}.e_score_correction_bias").float()
            rscale = loader.raw(f"{mp}.gate.routed_scaling_factor_param").float()
            w_fc1 = loader.linear_weight(f"{mp}.fc1_latent_proj")
            w_fc2 = loader.linear_weight(f"{mp}.fc2_latent_proj")
            lnorm_w = loader.raw(f"{mp}.latent_norm.weight")
            sw1, sw3, sw2 = (loader.linear_weight(f"{mp}.shared_experts.{p}") for p in ("w1", "w3", "w2"))
            beta, lbeta, topk = cfg["activation_situ_beta"], cfg["activation_situ_linear_beta"], cfg["num_experts_per_tok"]

            lat, ids, rw, lat_out, union_e = {}, {}, {}, {}, set()
            for n in run_names:
                x = ffn_in[n]
                sc = torch.sigmoid(x @ wg0.t()) + torch.sigmoid(x @ wg1.t())
                ids[n] = torch.topk(sc + bias.unsqueeze(0), topk, dim=-1, sorted=False).indices
                w = sc.gather(1, ids[n])
                rw[n] = (w / w.sum(-1, keepdim=True).clamp_min(1e-20)) * rscale[ids[n]]
                lat[n] = x @ w_fc1.t()
                lat_out[n] = torch.zeros_like(lat[n])
                union_e.update(torch.unique(ids[n]).tolist())
            n_act = len(union_e)

            for eid in sorted(union_e):
                ep = f"{mp}.experts.{eid}"
                wref = [loader.expert_weight(f"{ep}.{p}").float() for p in ("w1", "w3", "w2")]
                for n in run_names:
                    tok, slot = torch.where(ids[n] == eid)
                    if tok.numel() == 0:
                        continue
                    w1, w3, w2 = wref if lid in KEEP_W4 else [arms[n](w) for w in wref]
                    inp = lat[n][tok]
                    oe = G.situ_and_mul(inp @ w1.t(), inp @ w3.t(), beta, lbeta) @ w2.t()
                    lat_out[n].index_add_(0, tok, oe * rw[n][tok, slot].unsqueeze(-1))

            for n in run_names:
                x = ffn_in[n]
                routed = G.rms_norm(lat_out[n], lnorm_w, eps) @ w_fc2.t()
                shared = G.situ_and_mul(x @ sw1.t(), x @ sw3.t(), beta, lbeta) @ sw2.t()
                residuals[n] = G.mhc_post(routed + shared, residuals[n], *ffn_post[n])
            r0 = residuals[names[0]]
        if len(run_names) == 1:
            for n in names:
                residuals[n] = r0.clone()

        print(f"  L{lid:02d}/{n_layers-1} ({'DSA' if is_dsa else 'HCA'}, E={n_act:3d}) "
              + " ".join(f"{n[:7]}={residuals[n].std():.3f}" for n in names)
              + f" ({time.time()-lt0:.1f}s, tot {time.time()-t0:.0f}s)", flush=True)

    h_fn = loader.raw("model.mhc_contract.hc_head_fn")
    h_base = loader.raw("model.mhc_contract.hc_head_base")
    h_scale = loader.raw("model.mhc_contract.hc_head_scale")
    norm_w = loader.raw("model.norm.weight")
    lm_w = loader.linear_weight("lm_head")
    mm_r = loader.raw("model.lm_head.multimax_ranges").float()
    mm_t = loader.raw("model.lm_head.multimax_ts").float()

    targets = torch.tensor(token_ids[1:], dtype=torch.long, device=DEV)
    p_ref = torch.softmax(ref_lg, -1)
    lp_ref = torch.log_softmax(ref_lg, -1)
    ref_nll = F.cross_entropy(ref_lg[:-1], targets).item()
    print(f"\n=== VQ4 vs NF2 end-to-end (T={T} wikitext-2, keep W4 {sorted(KEEP_W4)}) ===")
    print(f"{'Arm':<16} {'NLL':>7} {'PPL':>7} {'dNLL':>8} {'MeanKL':>8} {'MedKL':>8} {'Top1%':>7}")
    print(f"{'ref_144.pt':<16} {ref_nll:7.4f} {math.exp(ref_nll):7.2f}")
    summary = {}
    for n in names:
        h = G.rms_norm(G.mhc_head_contract(residuals[n], h_fn, h_scale, h_base, eps, hc_eps), norm_w, eps)
        lg = h @ lm_w.t()
        lg = (lg + mm_t[0] * F.relu(mm_r[0] - lg) + mm_t[1] * F.relu(lg - mm_r[1])
              + mm_t[2] * F.relu(mm_r[2] - lg).square() + mm_t[3] * F.relu(lg - mm_r[3]).square())
        nll = F.cross_entropy(lg[:-1], targets).item()
        kl = (p_ref * (lp_ref - torch.log_softmax(lg, -1))).sum(-1)
        top1 = (ref_lg.argmax(-1) == lg.argmax(-1)).float().mean().item() * 100
        print(f"{n:<16} {nll:7.4f} {math.exp(nll):7.2f} {nll-ref_nll:+8.4f} "
              f"{kl.mean().item():8.4f} {kl.median().item():8.4f} {top1:6.2f}%")
        summary[n] = dict(nll=nll, ppl=math.exp(nll), dnll=nll - ref_nll,
                          mean_kl=kl.mean().item(), med_kl=kl.median().item(), top1=top1)
    json.dump(summary, open(f"{EVAL}/vq4_e2e_summary.json", "w"), indent=2)
    print(f"saved {EVAL}/vq4_e2e_summary.json")


if __name__ == "__main__":
    main()
