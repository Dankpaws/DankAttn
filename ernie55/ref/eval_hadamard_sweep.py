#!/usr/bin/env python3
"""4-arm Hadamard + GPTQ 2-bit sweep on T=144 wikitext-2 tokens (GPU).

Compares (all keeping L1 & L42 at ref/W4, L2..L41 at W2-g64 NF2):
  1. rtn_nf2         : 3-clip RTN NF2 on gu & dn
  2. gptq_gu_only    : shared-H_gu GPTQ on gu, clip-RTN on dn
  3. had_rtn_both    : 1D Hadamard (W_gu @ H) RTN on gu, 2D Hadamard (H @ W_dn @ H) RTN on dn
  4. had_gu_gptq_dn_rtn : 1D Hadamard GPTQ (W_gu @ H, H @ H_gu @ H) on gu, 2D Hadamard RTN on dn
"""
import json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "convert"))
import ernie55_golden as G
import gptq_w2 as GQ2

torch.set_grad_enabled(False)


def make_hadamard(n: int, device: torch.device) -> torch.Tensor:
    H = torch.tensor([[1.0]], dtype=torch.float32, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H * (1.0 / math.sqrt(n))


def rtn_dequant(w: torch.Tensor) -> torch.Tensor:
    q, s = GQ2.rtn_quant_w2(w, gs=64, mode="nf2")
    return GQ2.dequant_w2(q, s, gs=64, mode="nf2")


def main():
    ckpt = "/mnt/lmcache-nvme/ernie55/hf-mxfp4"
    calib_path = "/mnt/lmcache-nvme/ernie55/gptq_calib_8k.pt"
    ref_path = "/mnt/lmcache-nvme/ernie55/eval/ref_144.pt"

    ref = torch.load(ref_path, map_location="cpu")
    token_ids = ref["token_ids"]
    ref_lg = ref["logits"].float()
    T = len(token_ids)
    calib = torch.load(calib_path, map_location="cpu")["layers"]

    loader = G.CkptLoader(ckpt, device="cuda")
    cfg = loader.cfg
    dev = loader.device
    eps = cfg["rms_norm_eps"]
    hc_eps = cfg["hc_eps"]
    hc_mult = cfg["hc_mult"]
    sk_iters = cfg["hc_sinkhorn_iters"]
    n_layers = cfg["num_hidden_layers"]
    positions = torch.arange(T, dtype=torch.long, device=dev)

    H2048 = make_hadamard(2048, dev)
    keep_w4 = {1, 42}

    arms = [
        "rtn_nf2",
        "gptq_gu_only",
        "had_rtn_both",
        "had_gu_gptq_dn_rtn",
    ]
    print(f"Running 4-arm Hadamard+GPTQ sweep on T={T} wikitext-2 tokens (CUDA)...", flush=True)

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
    residuals = {a: res_base.clone() for a in arms}

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
        for a in arms:
            if lid == 0 and a != arms[0]:
                continue
            res = residuals[a]
            post_mix, comb, x = G.mhc_pre(res, ahc_fn, ahc_scale, ahc_base, eps, hc_eps, sk_iters)
            x = G.rms_norm(x, attn_nw, eps)
            attn_out = G.run_dsa_layer(loader, lid, x, positions, cos_mla, sin_mla, cfg) if is_dsa \
                else G.run_hca_layer(loader, lid, x, positions, cos_hca, sin_hca, cfg)
            res = G.mhc_post(attn_out, res, post_mix, comb)
            post_mix2, comb2, x2 = G.mhc_pre(res, mhc_fn, mhc_scale, mhc_base, eps, hc_eps, sk_iters)
            x2 = G.rms_norm(x2, ffn_nw, eps)
            residuals[a] = res
            ffn_in[a] = x2
            ffn_post[a] = (post_mix2, comb2)

        if not is_moe:
            ffn_out0 = G.run_dense_ffn(loader, lid, ffn_in[arms[0]], cfg)
            res0 = G.mhc_post(ffn_out0, residuals[arms[0]], *ffn_post[arms[0]])
            for a in arms:
                residuals[a] = res0.clone()
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
            for a in arms:
                x = ffn_in[a]
                sc = torch.sigmoid(x @ wg0.t()) + torch.sigmoid(x @ wg1.t())
                ids = torch.topk(sc + bias.unsqueeze(0), topk, dim=-1, sorted=False).indices
                w = sc.gather(1, ids)
                w = (w / w.sum(dim=-1, keepdim=True).clamp_min(1e-20)) * rscale[ids]
                lat = x @ w_fc1.t()
                arm_lat[a], arm_ids[a], arm_w[a] = lat, ids, w
                arm_lat_out[a] = torch.zeros_like(lat)
                union_e.update(torch.unique(ids).tolist())

            use_w4 = lid in keep_w4
            Hinv_plain, hdiag_plain = None, None
            Hinv_had, hdiag_had = None, None
            if not use_w4:
                H_gu = calib[lid]["H_gu"].to(dev)
                Hinv_plain, hdiag_plain = GQ2.prepare_hinv(H_gu, damp=0.01)
                H_gu_had = H2048 @ H_gu @ H2048
                Hinv_had, hdiag_had = GQ2.prepare_hinv(H_gu_had, damp=0.01)

            eids_sorted = sorted(union_e)
            n_act = len(eids_sorted)
            # Process in batches of 16 active experts for fast GPU GPTQ
            for b0 in range(0, n_act, 16):
                beids = eids_sorted[b0:b0 + 16]
                w1s, w3s, w2s = [], [], []
                for eid in beids:
                    epfx = f"{mpfx}.experts.{eid}"
                    w1s.append(loader.expert_weight(f"{epfx}.w1"))
                    w3s.append(loader.expert_weight(f"{epfx}.w3"))
                    w2s.append(loader.expert_weight(f"{epfx}.w2"))
                if use_w4:
                    arm_ws = {a: (w1s, w3s, w2s) for a in arms}
                else:
                    Wgu_cat = torch.cat([torch.cat([w1, w3], dim=0) for w1, w3 in zip(w1s, w3s)], dim=0)
                    Wdn_cat = torch.cat(w2s, dim=0)
                    B = len(beids)

                    # Arm 1: rtn_nf2
                    gu1 = rtn_dequant(Wgu_cat).view(B, 2, 2048, 2048)
                    dn1 = rtn_dequant(Wdn_cat).view(B, 2048, 2048)

                    # Arm 2: gptq_gu_only (shared H_gu GPTQ on gu, RTN on dn)
                    q2, s2 = GQ2.gptq_quant_w2(Wgu_cat, gs=64, mode="nf2", Hinv=Hinv_plain, hdiag=hdiag_plain)
                    gu2 = GQ2.dequant_w2(q2, s2, gs=64, mode="nf2").view(B, 2, 2048, 2048)
                    dn2 = dn1

                    # Arm 3: had_rtn_both (W_gu @ H RTN, H @ W_dn @ H RTN)
                    Wgu_h = Wgu_cat @ H2048
                    gu3 = (rtn_dequant(Wgu_h) @ H2048).view(B, 2, 2048, 2048)
                    Wdn_h = torch.stack([H2048 @ w2 @ H2048 for w2 in w2s], dim=0).reshape(B * 2048, 2048)
                    dn3_h = rtn_dequant(Wdn_h).view(B, 2048, 2048)
                    dn3 = torch.stack([H2048 @ dn3_h[bi] @ H2048 for bi in range(B)], dim=0)

                    # Arm 4: had_gu_gptq_dn_rtn (W_gu @ H GPTQ with H @ H_gu @ H, 2D Hadamard RTN on dn)
                    q4, s4 = GQ2.gptq_quant_w2(Wgu_h, gs=64, mode="nf2", Hinv=Hinv_had, hdiag=hdiag_had)
                    gu4 = (GQ2.dequant_w2(q4, s4, gs=64, mode="nf2") @ H2048).view(B, 2, 2048, 2048)
                    dn4 = dn3

                    arm_ws = {
                        "rtn_nf2":            ([gu1[i, 0] for i in range(B)], [gu1[i, 1] for i in range(B)], [dn1[i] for i in range(B)]),
                        "gptq_gu_only":       ([gu2[i, 0] for i in range(B)], [gu2[i, 1] for i in range(B)], [dn2[i] for i in range(B)]),
                        "had_rtn_both":       ([gu3[i, 0] for i in range(B)], [gu3[i, 1] for i in range(B)], [dn3[i] for i in range(B)]),
                        "had_gu_gptq_dn_rtn": ([gu4[i, 0] for i in range(B)], [gu4[i, 1] for i in range(B)], [dn4[i] for i in range(B)]),
                    }

                for bi, eid in enumerate(beids):
                    for a in arms:
                        tok_idx, slot_idx = torch.where(arm_ids[a] == eid)
                        if tok_idx.numel() == 0:
                            continue
                        w1 = arm_ws[a][0][bi]
                        w3 = arm_ws[a][1][bi]
                        w2 = arm_ws[a][2][bi]
                        inp = arm_lat[a][tok_idx]
                        act = G.situ_and_mul(inp @ w1.t(), inp @ w3.t(), beta, lbeta)
                        out_e = act @ w2.t()
                        arm_lat_out[a].index_add_(0, tok_idx, out_e * arm_w[a][tok_idx, slot_idx].unsqueeze(-1))

            for a in arms:
                x = ffn_in[a]
                routed_out = G.rms_norm(arm_lat_out[a], lnorm_w, eps) @ w_fc2.t()
                shared_out = G.situ_and_mul(x @ sw1.t(), x @ sw3.t(), beta, lbeta) @ sw2.t()
                residuals[a] = G.mhc_post(routed_out + shared_out, residuals[a], *ffn_post[a])

        if lid in (0, 1, 2, 10, 20, 30, 41, 42):
            print(f"  L{lid:02d}/{n_layers-1} (E={n_act:3d}) ({time.time()-lt0:.1f}s, tot {time.time()-t0:.0f}s)", flush=True)

    h_fn = loader.raw("model.mhc_contract.hc_head_fn")
    h_base = loader.raw("model.mhc_contract.hc_head_base")
    h_scale = loader.raw("model.mhc_contract.hc_head_scale")
    norm_w = loader.raw("model.norm.weight")
    lm_w = loader.linear_weight("lm_head")
    mm_r = loader.raw("model.lm_head.multimax_ranges").float()
    mm_t = loader.raw("model.lm_head.multimax_ts").float()

    targets = torch.tensor(token_ids[1:], dtype=torch.long)
    p_ref = torch.softmax(ref_lg, dim=-1)
    log_p_ref = torch.log_softmax(ref_lg, dim=-1)
    ref_nll = F.cross_entropy(ref_lg[:-1], targets).item()

    print(f"\n============= HADAMARD + GPTQ 2-BIT SWEEP (T={T} wikitext-2) =============")
    print(f"{'Arm':<22} {'NLL':>7} {'PPL':>7} {'dNLL':>7} {'MeanKL':>8} {'MedKL':>8} {'Top1%':>7}")
    print("-" * 74)
    print(f"{'ref_mxfp4':<22} {ref_nll:7.4f} {math.exp(ref_nll):7.2f} {'+0.0000':>7} {'0.0000':>8} {'0.0000':>8} {'100.00%':>7}")
    summary = {}
    for a in arms:
        h = G.mhc_head_contract(residuals[a], h_fn, h_scale, h_base, eps, hc_eps)
        h = G.rms_norm(h, norm_w, eps)
        lg = (h @ lm_w.t()).cpu()
        mm_rc, mm_tc = mm_r.cpu(), mm_t.cpu()
        lg = (
            lg
            + mm_tc[0] * F.relu(mm_rc[0] - lg)
            + mm_tc[1] * F.relu(lg - mm_rc[1])
            + mm_tc[2] * F.relu(mm_rc[2] - lg).square()
            + mm_tc[3] * F.relu(lg - mm_rc[3]).square()
        )
        nll = F.cross_entropy(lg[:-1], targets).item()
        log_p_q = torch.log_softmax(lg, dim=-1)
        kl = (p_ref * (log_p_ref - log_p_q)).sum(dim=-1)
        top1 = (ref_lg.argmax(-1) == lg.argmax(-1)).float().mean().item() * 100.0
        print(f"{a:<22} {nll:7.4f} {math.exp(nll):7.2f} {nll-ref_nll:+7.4f} "
              f"{kl.mean().item():8.4f} {kl.median().item():8.4f} {top1:6.2f}%")
        summary[a] = dict(nll=nll, ppl=math.exp(nll), dnll=nll - ref_nll,
                          mean_kl=kl.mean().item(), med_kl=kl.median().item(), top1=top1)
    json.dump(summary, open("/mnt/lmcache-nvme/ernie55/eval/hadamard_sweep_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
