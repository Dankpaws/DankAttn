#!/usr/bin/env python3
"""Collect per-layer latent Hessians H_gu [2048, 2048] and per-expert routing stats
for ERNIE-5.5-Flash-Preview 2-bit GPTQ quantization.

Runs one forward pass over calibration tokens (e.g., 4k-8k tokens of diverse text/code)
and saves gptq_calib.pt (~400 MB).
"""
import argparse
import os
import sys
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref"))
import ernie55_golden as G

torch.set_grad_enabled(False)


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to upstream ERNIE-5.5-Flash-Preview checkpoint")
    ap.add_argument("--out", default="gptq_calib.pt", help="Output calibration .pt path")
    ap.add_argument("--wikitext", default="", help="Optional path to wikitext-2 raw text file")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(args.ckpt, "tokenizer.json"))

    # Build diverse calibration text from optional wikitext-2 + DankAttn source/docs
    parts = []
    if args.wikitext and os.path.exists(args.wikitext):
        parts.append(open(args.wikitext, encoding="utf-8", errors="ignore").read()[:60000])
    for rel_path in [
        "engine/hy4/hy4.mm",
        "engine/kernels/mm2.metal",
        "ernie55/ref/ernie55_golden.py",
        "README.md",
    ]:
        src_path = os.path.join(repo_root, rel_path)
        if os.path.exists(src_path):
            parts.append(open(src_path, encoding="utf-8", errors="ignore").read()[:30000])

    # Interleave 512-token chunks so H_gu sees prose, code, Metal, Python, and markdown
    all_chunks = []
    for p in parts:
        ids = tok.encode(p).ids
        for i in range(0, len(ids) - 512, 512):
            all_chunks.append([1] + ids[i:i + 511])
    # Pick evenly across sources
    need_chunks = max(1, args.tokens // 512)
    step = max(1, len(all_chunks) // need_chunks)
    chunks = all_chunks[::step][:need_chunks]
    print(f"Collecting GPTQ Hessians over {len(chunks)} chunks x 512 = {len(chunks)*512} tokens", flush=True)

    loader = G.CkptLoader(args.ckpt, device=args.device)
    cfg = loader.cfg
    dev = loader.device
    eps = cfg["rms_norm_eps"]
    hc_eps = cfg["hc_eps"]
    hc_mult = cfg["hc_mult"]
    sk_iters = cfg["hc_sinkhorn_iters"]
    n_layers = cfg["num_hidden_layers"]
    T = 512

    rs = cfg["rope_scaling"]
    cos_hca, sin_hca = G.build_yarn_cos_sin(
        T + 16, cfg["qk_rope_head_dim"], cfg["compress_rope_theta"],
        rs["factor"], rs["original_max_position_embeddings"],
        rs["beta_fast"], rs["beta_slow"], dev,
    )
    cos_mla, sin_mla = G.build_plain_cos_sin(T + 16, cfg["hybrid_mla_qk_rope_head_dim"], cfg["rope_theta"], dev)
    positions = torch.arange(T, dtype=torch.long, device=dev)

    # Embed all chunks on CPU first (small: ~N*512*4096*2 bytes = ~32 MB)
    emb_w = loader.raw("model.embed_tokens.weight", to_dev=False)
    residuals = []
    for c in chunks:
        h0 = emb_w[torch.tensor(c, dtype=torch.long)].float()  # [512, 4096]
        residuals.append(h0.unsqueeze(1).expand(T, hc_mult, -1).contiguous())
    del emb_w

    layers_cal = {}
    t0 = time.time()
    for lid in range(n_layers):
        lpfx = f"model.layers.{lid}"
        is_dsa = cfg["compress_ratios"][lid] <= 0
        is_moe = lid >= cfg["first_k_dense_replace"]

        # Load layer-level non-expert weights once
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

        H_gu = torch.zeros((2048, 2048), dtype=torch.float64, device=dev) if is_moe else None
        lat_list = []
        top_list = []

        for ci in range(len(residuals)):
            res = residuals[ci].to(dev)
            post_mix, comb, x = G.mhc_pre(res, ahc_fn, ahc_scale, ahc_base, eps, hc_eps, sk_iters)
            x = G.rms_norm(x, attn_nw, eps)
            if is_dsa:
                attn_out = G.run_dsa_layer(loader, lid, x, positions, cos_mla, sin_mla, cfg)
            else:
                attn_out = G.run_hca_layer(loader, lid, x, positions, cos_hca, sin_hca, cfg)
            res = G.mhc_post(attn_out, res, post_mix, comb)

            post_mix, comb, x = G.mhc_pre(res, mhc_fn, mhc_scale, mhc_base, eps, hc_eps, sk_iters)
            x = G.rms_norm(x, ffn_nw, eps)
            if not is_moe:
                ffn_out = G.run_dense_ffn(loader, lid, x, cfg)
            else:
                # Capture latent input & routing for GPTQ
                mpfx = f"{lpfx}.block_sparse_moe"
                w_fc1 = loader.linear_weight(f"{mpfx}.fc1_latent_proj")
                lat = x @ w_fc1.t()  # [512, 2048]
                H_gu += (lat.double().t() @ lat.double())
                wg0 = loader.raw(f"{mpfx}.gate.weight").float()
                wg1 = loader.raw(f"{mpfx}.gate.weight_1").float()
                scores = torch.sigmoid(x @ wg0.t()) + torch.sigmoid(x @ wg1.t())
                bias = loader.raw(f"{mpfx}.e_score_correction_bias").float()
                ids = torch.topk(scores + bias.unsqueeze(0), cfg["num_experts_per_tok"], dim=-1, sorted=False).indices
                lat_list.append(lat.half().cpu())
                top_list.append(ids.to(torch.int16).cpu())
                ffn_out = G.run_moe_layer(loader, lid, x, cfg, expert_quant="none")
            res = G.mhc_post(ffn_out, res, post_mix, comb)
            residuals[ci] = res.cpu()

        if is_moe:
            layers_cal[lid] = {
                "H_gu": (H_gu / (len(residuals) * T)).float().cpu(),
                "lat": torch.cat(lat_list, dim=0),
                "top": torch.cat(top_list, dim=0),
            }
        print(f"  calib layer {lid:2d}/{n_layers-1} done ({time.time()-t0:.1f}s)", flush=True)

    torch.save({"layers": layers_cal, "tokens": len(residuals) * T}, args.out)
    print(f"Saved GPTQ calibration to {args.out}", flush=True)


if __name__ == "__main__":
    main()
