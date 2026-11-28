#!/usr/bin/env python3
"""Fast 8-arm GPU sweep for ERNIE-5.5-Flash-Preview:
  1. W2-g32 (aligned 1:1 with native MXFP4 gs=32) vs W2-g64
  2. moe.dn-only W2-g32 vs full W2-g32
  3. E2M1-tuned inner codebook levels (0.25, 0.2998, 1/3, 0.35)
  4. Layer sensitivity: L1+L42 vs L1-only vs L1+L2 at W4
"""
import json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ernie55_golden as G

torch.set_grad_enabled(False)

CLIPS_13 = (1.0, 0.96, 0.92, 0.88, 0.84, 0.80, 0.76, 0.72, 0.68, 0.64, 0.60, 0.56, 0.52)


def rtn_w2_dequant(w: torch.Tensor, gs: int, inner: float) -> torch.Tensor:
    N, K = w.shape
    mid = 0.5 * (1.0 + inner)
    g = w.float().reshape(N, K // gs, gs)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    best_wq = None
    best_err = None
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


def main():
    ckpt = "/mnt/lmcache-nvme/ernie55/hf-mxfp4"
    ref_path = "/mnt/lmcache-nvme/ernie55/eval/ref_144.pt"

    ref = torch.load(ref_path, map_location="cpu")
    token_ids = ref["token_ids"]
    ref_lg = ref["logits"].float()
    T = len(token_ids)

    loader = G.CkptLoader(ckpt, device="cuda")
    cfg = loader.cfg
    dev = loader.device
    eps = cfg["rms_norm_eps"]
    hc_eps = cfg["hc_eps"]
    hc_mult = cfg["hc_mult"]
    sk_iters = cfg["hc_sinkhorn_iters"]
    n_layers = cfg["num_hidden_layers"]
    positions = torch.arange(T, dtype=torch.long, device=dev)

    # (name, gs_gu, gs_dn, inner, keep_w4)
    arms = [
        ("g64_nf2_L1_42",   64, 64, 0.2998,      {1, 42}),
        ("g32_dn_L1_42",    64, 32, 0.2998,      {1, 42}),
        ("g32_all_L1_42",   32, 32, 0.2998,      {1, 42}),
        ("g32_all_L1_only", 32, 32, 0.2998,      {1}),
        ("g32_all_L1_L2",   32, 32, 0.2998,      {1, 2}),
        ("g32_all_in25",    32, 32, 0.2500,      {1, 42}),
        ("g32_all_in333",   32, 32, 1.0 / 3.0,   {1, 42}),
        ("g32_all_in35",    32, 32, 0.3500,      {1, 42}),
    ]
    print(f"Running {len(arms)}-arm W2-g32 + codebook + layer sweep on T={T} (CUDA)...", flush=True)

    rs = cfg["rope_scaling"]
    cos_hca, sin_hca = G.build_yarn_cos_sin(
        T + 16, cfg["qk_rope_head_dim"], cfg["compress_rope_theta"],
        rs["factor"], rs["original_max_position_embeddings"],
        rs["beta_fast"], rs["beta_slow"], dev,
    )
    cos_mla, sin_mla = G.build_plain_cos_sin(T + 16, cfg["hybrid_mla_qk_rope_head_dim"], cfg["rope_theta"], dev)

    emb_w = loader.raw("model.embed_tokens.weight", to_dev=False)
    h0 = emb_w[torch.tensor(token_ids, dtype=torch.long)].float().to(dev)
    del emb_w
    res_base = h0.unsqueeze(1).expand(T, hc_mult, -1).contiguous()
    residuals = {a[0]: res_base.clone() for a in arms}
    layer_rel_err = {}

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
        for name, *_ in arms:
            if lid == 0 and name != arms[0][0]:
                continue
            res = residuals[name]
            post_mix, comb, x = G.mhc_pre(res, ahc_fn, ahc_scale, ahc_base, eps, hc_eps, sk_iters)
            x = G.rms_norm(x, attn_nw, eps)
            attn_out = G.run_dsa_layer(loader, lid, x, positions, cos_mla, sin_mla, cfg) if is_dsa \
                else G.run_hca_layer(loader, lid, x, positions, cos_hca, sin_hca, cfg)
            res = G.mhc_post(attn_out, res, post_mix, comb)
            post_mix2, comb2, x2 = G.mhc_pre(res, mhc_fn, mhc_scale, mhc_base, eps, hc_eps, sk_iters)
            x2 = G.rms_norm(x2, ffn_nw, eps)
            residuals[name] = res
            ffn_in[name] = x2
            ffn_post[name] = (post_mix2, comb2)

        if not is_moe:
            ffn_out0 = G.run_dense_ffn(loader, lid, ffn_in[arms[0][0]], cfg)
            res0 = G.mhc_post(ffn_out0, residuals[arms[0][0]], *ffn_post[arms[0][0]])
            for name, *_ in arms:
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
            for name, *_ in arms:
                x = ffn_in[name]
                sc = torch.sigmoid(x @ wg0.t()) + torch.sigmoid(x @ wg1.t())
                ids = torch.topk(sc + bias.unsqueeze(0), topk, dim=-1, sorted=False).indices
                w = sc.gather(1, ids)
                w = (w / w.sum(dim=-1, keepdim=True).clamp_min(1e-20)) * rscale[ids]
                lat = x @ w_fc1.t()
                arm_lat[name], arm_ids[name], arm_w[name] = lat, ids, w
                arm_lat_out[name] = torch.zeros_like(lat)
                union_e.update(torch.unique(ids).tolist())

            eids_sorted = sorted(union_e)
            n_act = len(eids_sorted)
            # Determine unique (gs, inner) pairs needed at this layer
            for b0 in range(0, n_act, 16):
                beids = eids_sorted[b0:b0 + 16]
                B = len(beids)
                w1s, w3s, w2s = [], [], []
                for eid in beids:
                    epfx = f"{mpfx}.experts.{eid}"
                    w1s.append(loader.expert_weight(f"{epfx}.w1"))
                    w3s.append(loader.expert_weight(f"{epfx}.w3"))
                    w2s.append(loader.expert_weight(f"{epfx}.w2"))
                Wgu_cat = torch.cat([torch.cat([w1, w3], dim=0) for w1, w3 in zip(w1s, w3s)], dim=0)
                Wdn_cat = torch.cat(w2s, dim=0)

                gu_cache = {("ref", 0): (w1s, w3s)}
                dn_cache = {("ref", 0): w2s}
                for _, gs_gu, gs_dn, inner, keep_w4 in arms:
                    if lid in keep_w4:
                        continue
                    k_gu = (gs_gu, inner)
                    if k_gu not in gu_cache:
                        qgu = rtn_w2_dequant(Wgu_cat, gs_gu, inner).view(B, 2, 2048, 2048)
                        gu_cache[k_gu] = ([qgu[i, 0] for i in range(B)], [qgu[i, 1] for i in range(B)])
                    k_dn = (gs_dn, inner)
                    if k_dn not in dn_cache:
                        qdn = rtn_w2_dequant(Wdn_cat, gs_dn, inner).view(B, 2048, 2048)
                        dn_cache[k_dn] = [qdn[i] for i in range(B)]

                for bi, eid in enumerate(beids):
                    for name, gs_gu, gs_dn, inner, keep_w4 in arms:
                        tok_idx, slot_idx = torch.where(arm_ids[name] == eid)
                        if tok_idx.numel() == 0:
                            continue
                        k_gu = ("ref", 0) if lid in keep_w4 else (gs_gu, inner)
                        k_dn = ("ref", 0) if lid in keep_w4 else (gs_dn, inner)
                        w1 = gu_cache[k_gu][0][bi]
                        w3 = gu_cache[k_gu][1][bi]
                        w2 = dn_cache[k_dn][bi]
                        inp = arm_lat[name][tok_idx]
                        act = G.situ_and_mul(inp @ w1.t(), inp @ w3.t(), beta, lbeta)
                        out_e = act @ w2.t()
                        arm_lat_out[name].index_add_(0, tok_idx, out_e * arm_w[name][tok_idx, slot_idx].unsqueeze(-1))

            for name, *_ in arms:
                x = ffn_in[name]
                routed_out = G.rms_norm(arm_lat_out[name], lnorm_w, eps) @ w_fc2.t()
                shared_out = G.situ_and_mul(x @ sw1.t(), x @ sw3.t(), beta, lbeta) @ sw2.t()
                residuals[name] = G.mhc_post(routed_out + shared_out, residuals[name], *ffn_post[name])

        if lid in (0, 1, 2, 10, 20, 30, 41, 42):
            print(f"  L{lid:02d}/{n_layers-1} (E={n_act:3d}) ({time.time()-lt0:.1f}s, tot {time.time()-t0:.0f}s)", flush=True)

    h_fn = loader.raw("model.mhc_contract.hc_head_fn")
    h_base = loader.raw("model.mhc_contract.hc_head_base")
    h_scale = loader.raw("model.mhc_contract.hc_head_scale")
    norm_w = loader.raw("model.norm.weight")
    lm_w = loader.linear_weight("lm_head")
    mm_r = loader.raw("model.lm_head.multimax_ranges").float().cpu()
    mm_t = loader.raw("model.lm_head.multimax_ts").float().cpu()

    targets = torch.tensor(token_ids[1:], dtype=torch.long)
    p_ref = torch.softmax(ref_lg, dim=-1)
    log_p_ref = torch.log_softmax(ref_lg, dim=-1)
    ref_nll = F.cross_entropy(ref_lg[:-1], targets).item()

    print(f"\n============= W2-G32 + CODEBOOK + LAYER SWEEP (T={T} wikitext-2) =============")
    print(f"{'Arm':<20} {'NLL':>7} {'PPL':>7} {'dNLL':>7} {'MeanKL':>8} {'MedKL':>8} {'Top1%':>7}")
    print("-" * 72)
    print(f"{'ref_mxfp4':<20} {ref_nll:7.4f} {math.exp(ref_nll):7.2f} {'+0.0000':>7} {'0.0000':>8} {'0.0000':>8} {'100.00%':>7}")
    summary = {}
    for name, *_ in arms:
        h = G.mhc_head_contract(residuals[name], h_fn, h_scale, h_base, eps, hc_eps)
        h = G.rms_norm(h, norm_w, eps)
        lg = (h @ lm_w.t()).cpu()
        lg = (
            lg
            + mm_t[0] * F.relu(mm_r[0] - lg)
            + mm_t[1] * F.relu(lg - mm_r[1])
            + mm_t[2] * F.relu(mm_r[2] - lg).square()
            + mm_t[3] * F.relu(lg - mm_r[3]).square()
        )
        nll = F.cross_entropy(lg[:-1], targets).item()
        kl = (p_ref * (log_p_ref - torch.log_softmax(lg, dim=-1))).sum(dim=-1)
        top1 = (ref_lg.argmax(-1) == lg.argmax(-1)).float().mean().item() * 100.0
        print(f"{name:<20} {nll:7.4f} {math.exp(nll):7.2f} {nll-ref_nll:+7.4f} "
              f"{kl.mean().item():8.4f} {kl.median().item():8.4f} {top1:6.2f}%")
        summary[name] = dict(nll=nll, ppl=math.exp(nll), dnll=nll - ref_nll,
                             mean_kl=kl.mean().item(), med_kl=kl.median().item(), top1=top1)
    json.dump(summary, open("/mnt/lmcache-nvme/ernie55/eval/gs32_codebook_sweep.json", "w"), indent=2)


if __name__ == "__main__":
    main()
