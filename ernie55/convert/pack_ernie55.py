#!/usr/bin/env python3
"""ERNIE-5.5-Flash-Preview (MXFP4+FP8) -> DankAttn ernie55 pack.

Packs:
  - embed:       bf16 raw [201216, 4096]
  - lm_head:     W8 g128 [201216, 4096]
  - dense/attn:  W8 g128 (or g64 for K=192)
  - routed MoE:  W2 g64 (NF2 or SYM2 codebook, 2.25 bpw, optional GPTQ)
                 + optional --keep_w4_layers kept at W4 g64

Usage:
  python3 pack_ernie55.py <hf_mxfp4_dir> <out_pack_dir> [--w2_mode nf2|sym2] [--gptq calib.pt] [--keep_w4 1,2,42]
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "convert"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pack as P
from layout import pack_codes, pack_scales
import ernie55_golden as G
import gptq_w2 as GQ2

torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count() or 16)


def fast_rtn_w2(w: torch.Tensor, gs: int = 64, mode: str = "nf2"):
    """Fast branchless 5-clip-search 2-bit quantizer (CPU L3-friendly, zero int64).
    Returns (uint8 codes [N, K] in 0..3, f32 scales [N, K//gs]).
    """
    N, K = w.shape
    inner = 0.2998 if mode == "nf2" else (1.0 / 3.0)
    mid = 0.5 * (1.0 + inner)
    g = w.float().reshape(N, K // gs, gs)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    best_s = (amax * 0.65).half().float().clamp_min(6e-8)
    best_err = None
    for r in (0.58, 0.66, 0.76):
        s = (amax * r).half().float().clamp_min(6e-8)
        xn = g / s
        sgn = torch.where(xn >= 0, 1.0, -1.0)
        wq = torch.where(xn.abs() > mid, sgn, sgn * inner) * s
        err = (wq - g).square().sum(-1)
        if best_err is None:
            best_err, best_s = err, s
        else:
            m = err < best_err
            best_err = torch.where(m, err, best_err)
            best_s = torch.where(m.unsqueeze(-1), s, best_s)
    xn = g / best_s
    # Map to codes 0..3: 0=-1, 1=-inner, 2=+inner, 3=+1
    pos = xn >= 0
    outer = xn.abs() > mid
    q = torch.where(pos, torch.where(outer, 3, 2), torch.where(outer, 0, 1)).to(torch.uint8)
    return q.reshape(N, K), best_s.squeeze(-1)


def quant_w8_dev(w: torch.Tensor, gs: int = 128):
    q, s = P.tquant(w, 8, gs, clip=True)
    return q.cpu(), s.cpu()


def quant_w4_dev(w: torch.Tensor, gs: int = 64):
    q, s = P.tquant(w, 4, gs, clip=True)
    return q.cpu(), s.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="HF MXFP4+FP8 checkpoint dir")
    ap.add_argument("out", help="Output DankAttn pack dir")
    ap.add_argument("--w2_mode", default="nf2", choices=["nf2", "sym2"])
    ap.add_argument("--w2_gs", type=int, default=64)
    ap.add_argument("--gptq", default=None, help="Optional gptq_calib.pt from collect_gptq.py")
    ap.add_argument("--gptq_damp", type=float, default=0.01)
    ap.add_argument("--gptq_damp_dn", type=float, default=0.10)
    ap.add_argument("--gptq_dn", action="store_true", help="Also run per-expert GPTQ on moe.dn")
    ap.add_argument("--keep_w4", default="", help="Comma-separated MoE layer ids to keep at W4-g64")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_layers", type=int, default=0, help="Debug: pack only first N layers")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    loader = G.CkptLoader(args.src, device=args.device)
    cfg = loader.cfg
    dev = loader.device
    keep_w4 = set(int(x) for x in args.keep_w4.split(",") if x.strip())
    w2_bits_tag = 2 if args.w2_mode == "nf2" else 20  # mm2.metal WBITS=2 (NF2) or 20 (SYM2)
    gptq_cal = torch.load(args.gptq, map_location="cpu")["layers"] if args.gptq else None

    W = P.Writer(args.out)

    def put_raw(name: str, t: torch.Tensor):
        t = t.cpu().contiguous()
        if t.dtype == torch.bfloat16:
            raw_bytes = t.view(torch.uint16).numpy().tobytes()
            dt = "bf16"
        elif t.dtype == torch.float32:
            raw_bytes = t.numpy().tobytes()
            dt = "f32"
        else:
            raw_bytes = t.float().numpy().tobytes()
            dt = "f32"
        f, o = W.put(raw_bytes)
        W.tensors[name] = dict(kind="r", dtype=dt, shape=list(t.shape), file=f, off=o, bytes=len(raw_bytes))

    def put_w8(name: str, w: torch.Tensor, gs: int = 128, rows: dict | None = None):
        N, K = w.shape
        q, s = quant_w8_dev(w, gs)
        cb = pack_codes(q.numpy(), 8)
        sb = pack_scales(s.numpy())
        f, o = W.put(cb.tobytes())
        fs, os_ = W.put(sb.tobytes())
        W.tensors[name] = dict(
            kind="w", N=N, K=K, bits=8, gs=gs, experts=1,
            file=f, off=o, bytes=cb.nbytes, sfile=fs, soff=os_, sbytes=sb.nbytes,
            rows=rows or {},
        )

    def put_w8_grouped(name: str, w3: torch.Tensor, gs: int = 128):
        E_, N, K = w3.shape
        cb_e = N * K
        if W.cur is None or W.size + E_ * cb_e + P.PAGE > W.max_file:
            W._open()
        first = None
        scs = []
        for e in range(E_):
            q, s = quant_w8_dev(w3[e], gs)
            f, o = W.put(pack_codes(q.numpy(), 8).tobytes())
            if first is None:
                first = (f, o)
            scs.append(pack_scales(s.numpy()))
        sc = np.concatenate(scs)
        fs, os_ = W.put(sc.tobytes())
        W.tensors[name] = dict(
            kind="w", N=N, K=K, bits=8, gs=gs, experts=E_,
            file=first[0], off=first[1], bytes=E_ * cb_e,
            sfile=fs, soff=os_, sbytes=sc.nbytes, rows={},
        )

    # 1. Global tensors
    print("Packing embed, norm, hch, lm_head, multimax...", flush=True)
    put_raw("embed", loader.raw("model.embed_tokens.weight", to_dev=False).to(torch.bfloat16))
    h_fn = loader.raw("model.mhc_contract.hc_head_fn", to_dev=False).float()
    if h_fn.shape[0] == cfg["hc_mult"] * cfg["hidden_size"]:
        h_fn = h_fn.t().contiguous()
    put_raw("hch.fn", h_fn)
    put_raw("hch.base", loader.raw("model.mhc_contract.hc_head_base", to_dev=False).float())
    put_raw("hch.scale", loader.raw("model.mhc_contract.hc_head_scale", to_dev=False).float())
    put_raw("norm", loader.raw("model.norm.weight", to_dev=False).to(torch.bfloat16))
    put_w8("lm_head", loader.linear_weight("lm_head"), gs=128)
    put_raw("multimax.r", loader.raw("model.lm_head.multimax_ranges", to_dev=False).float())
    put_raw("multimax.t", loader.raw("model.lm_head.multimax_ts", to_dev=False).float())

    n_layers = args.max_layers or cfg["num_hidden_layers"]
    E = cfg["n_routed_experts"]
    gu_order = P.gu_rows(cfg["moe_intermediate_size"]).to(dev)
    dense_gu_order = P.gu_rows(cfg["intermediate_size"]).to(dev)

    for lid in range(n_layers):
        lt0 = time.time()
        lpfx = f"model.layers.{lid}"
        dpfx = f"L{lid}"
        is_dsa = cfg["compress_ratios"][lid] <= 0
        is_moe = lid >= cfg["first_k_dense_replace"]

        # mHC
        for src_hc, dst_hc in [("self_attention_hyper_connection", "ahc"), ("mlp_hyper_connection", "mhc")]:
            fn = loader.raw(f"{lpfx}.{src_hc}.mapping_proj.weight", to_dev=False).float().t().contiguous()
            put_raw(f"{dpfx}.{dst_hc}.fn", fn)
            put_raw(f"{dpfx}.{dst_hc}.base", loader.raw(f"{lpfx}.{src_hc}.bias", to_dev=False).float())
            sc = torch.cat([
                loader.raw(f"{lpfx}.{src_hc}.alpha_pre", to_dev=False).float(),
                loader.raw(f"{lpfx}.{src_hc}.alpha_post", to_dev=False).float(),
                loader.raw(f"{lpfx}.{src_hc}.alpha_res", to_dev=False).float(),
            ])
            put_raw(f"{dpfx}.{dst_hc}.scale", sc)
        put_raw(f"{dpfx}.attn_norm", loader.raw(f"{lpfx}.input_layernorm.weight", to_dev=False).to(torch.bfloat16))
        put_raw(f"{dpfx}.ffn_norm", loader.raw(f"{lpfx}.post_attention_layernorm.weight", to_dev=False).to(torch.bfloat16))

        # Attention
        apfx = f"{lpfx}.self_attn"
        U = loader.raw(f"{apfx}.vha_postmix_U").float()
        V = loader.raw(f"{apfx}.vha_postmix_V").float()
        nh = cfg["hybrid_mla_num_attention_heads"] if is_dsa else cfg["num_attention_heads"]
        vha_m = torch.eye(nh, device=dev) + V @ U.t()
        put_raw(f"{dpfx}.attn.vha_m", vha_m)

        if not is_dsa:
            # HCA
            w_qd = loader.linear_weight(f"{apfx}.linear_q_down_proj")
            w_kv = loader.linear_weight(f"{apfx}.linear_kv_proj")
            w_ckv = loader.linear_weight(f"{apfx}.core_attention.compressor.linear_wkv")
            w_cgate = loader.linear_weight(f"{apfx}.core_attention.compressor.linear_wgate")
            w_in = torch.cat([w_qd, w_kv, w_ckv, w_cgate], dim=0)
            put_w8(f"{dpfx}.attn.in", w_in, gs=128, rows={
                "qd": [0, 1024], "kv": [1024, 512], "ckv": [1536, 512], "cgate": [2048, 512]
            })
            put_raw(f"{dpfx}.attn.q_norm", loader.raw(f"{apfx}.q_layernorm.weight", to_dev=False).to(torch.bfloat16))
            put_raw(f"{dpfx}.attn.kv_norm", loader.raw(f"{apfx}.kv_layernorm.weight", to_dev=False).to(torch.bfloat16))
            put_raw(f"{dpfx}.attn.comp_ape", loader.raw(f"{apfx}.core_attention.compressor.ape", to_dev=False).float())
            put_raw(f"{dpfx}.attn.comp_norm", loader.raw(f"{apfx}.core_attention.compressor.norm.weight", to_dev=False).to(torch.bfloat16))
            put_raw(f"{dpfx}.attn.sink", loader.raw(f"{apfx}.core_attention.attn_sink", to_dev=False).float())

            w_qu = loader.linear_weight(f"{apfx}.linear_q_up_proj")
            w_gt = loader.linear_weight(f"{apfx}.gate_proj")
            put_w8(f"{dpfx}.attn.qb", torch.cat([w_qu, w_gt], dim=0), gs=128, rows={
                "q": [0, 32768], "gate": [32768, 8192]
            })
            wo_a = loader.linear_weight(f"{apfx}.linear_o_group_proj").view(cfg["o_groups"], cfg["o_lora_rank"], -1)
            put_w8_grouped(f"{dpfx}.attn.wo_a", wo_a, gs=128)
            put_w8(f"{dpfx}.attn.wo_b", loader.linear_weight(f"{apfx}.o_proj"), gs=128)
        else:
            # DSA
            ipfx = f"{apfx}.core_attention.indexer"
            w_qa = loader.linear_weight(f"{apfx}.q_a_proj")
            w_kva = loader.linear_weight(f"{apfx}.kv_a_proj_with_mqa")
            w_ik = loader.linear_weight(f"{ipfx}.wk")
            w_iw = loader.linear_weight(f"{ipfx}.weights_proj")
            put_w8(f"{dpfx}.attn.in", torch.cat([w_qa, w_kva, w_ik, w_iw], dim=0), gs=128, rows={
                "qa": [0, 1024], "kva": [1024, 576], "ik": [1600, 128], "iw": [1728, 64]
            })
            put_raw(f"{dpfx}.attn.qa_norm", loader.raw(f"{apfx}.q_a_layernorm.weight", to_dev=False).to(torch.bfloat16))
            put_raw(f"{dpfx}.attn.kva_norm", loader.raw(f"{apfx}.kv_a_layernorm.weight", to_dev=False).to(torch.bfloat16))
            put_raw(f"{dpfx}.attn.idx_kn_w", loader.raw(f"{ipfx}.k_norm.weight", to_dev=False).float())
            put_raw(f"{dpfx}.attn.idx_kn_b", loader.raw(f"{ipfx}.k_norm.bias", to_dev=False).float())
            put_raw(f"{dpfx}.attn.sink", loader.raw(f"{apfx}.core_attention.softmax_offset", to_dev=False).float())

            w_qb = loader.linear_weight(f"{apfx}.q_b_proj")
            w_gt = loader.linear_weight(f"{apfx}.gate_proj")
            w_iqb = loader.linear_weight(f"{ipfx}.wq_b")
            put_w8(f"{dpfx}.attn.qb", torch.cat([w_qb, w_gt, w_iqb], dim=0), gs=128, rows={
                "q": [0, 16384], "gate": [16384, 16384], "iq": [32768, 8192]
            })
            nope = cfg["hybrid_mla_qk_nope_head_dim"]
            vdim = cfg["hybrid_mla_v_head_dim"]
            kvl = cfg["hybrid_mla_kv_lora_rank"]
            w_kvb = loader.linear_weight(f"{apfx}.kv_b_proj").view(nh, nope + vdim, kvl)
            # kb transposed: [64, 512, 192] (K=192 -> gs=64)
            put_w8_grouped(f"{dpfx}.attn.kb", w_kvb[:, :nope, :].transpose(1, 2).contiguous(), gs=64)
            put_w8_grouped(f"{dpfx}.attn.vb", w_kvb[:, nope:, :].contiguous(), gs=128)
            put_w8(f"{dpfx}.attn.o", loader.linear_weight(f"{apfx}.o_proj"), gs=128)

        # FFN / MoE
        if not is_moe:
            mpfx = f"{lpfx}.mlp"
            wg = loader.linear_weight(f"{mpfx}.gate_proj")
            wu = loader.linear_weight(f"{mpfx}.up_proj")
            wgu = torch.cat([wg, wu], dim=0)[dense_gu_order]
            put_w8(f"{dpfx}.mlp.gu", wgu, gs=128)
            put_w8(f"{dpfx}.mlp.dn", loader.linear_weight(f"{mpfx}.down_proj"), gs=128)
            print(f"L{lid:02d} (HCA dense) packed in {time.time()-lt0:.1f}s", flush=True)
        else:
            mpfx = f"{lpfx}.block_sparse_moe"
            wg0 = loader.raw(f"{mpfx}.gate.weight").float()
            wg1 = loader.raw(f"{mpfx}.gate.weight_1").float()
            w_fc1 = loader.linear_weight(f"{mpfx}.fc1_latent_proj")
            sw1 = loader.linear_weight(f"{mpfx}.shared_experts.w1")
            sw3 = loader.linear_weight(f"{mpfx}.shared_experts.w3")
            sgu = torch.cat([sw1, sw3], dim=0)[gu_order]
            put_w8(f"{dpfx}.moe.in", torch.cat([wg0, wg1, w_fc1, sgu], dim=0), gs=128, rows={
                "r0": [0, E], "r1": [E, E], "fc1": [2 * E, 2048], "sgu": [2 * E + 2048, 4096]
            })
            put_raw(f"{dpfx}.moe.bias", loader.raw(f"{mpfx}.e_score_correction_bias", to_dev=False).float())
            put_raw(f"{dpfx}.moe.rscale", loader.raw(f"{mpfx}.gate.routed_scaling_factor_param", to_dev=False).float())
            put_raw(f"{dpfx}.moe.lnorm", loader.raw(f"{mpfx}.latent_norm.weight", to_dev=False).to(torch.bfloat16))
            w_fc2 = loader.linear_weight(f"{mpfx}.fc2_latent_proj")
            sw2 = loader.linear_weight(f"{mpfx}.shared_experts.w2")
            put_w8(f"{dpfx}.moe.out", torch.cat([w_fc2, sw2], dim=1), gs=128)

            # Routed experts: W2-g64 (or W4-g64 if in keep_w4)
            use_w4 = lid in keep_w4
            bits = 4 if use_w4 else w2_bits_tag
            raw_bits = 4 if use_w4 else 2
            gs = 64 if use_w4 else args.w2_gs
            N_gu, K_gu = 4096, 2048
            N_dn, K_dn = 2048, 2048
            cb_gu = N_gu * K_gu * raw_bits // 8
            cb_dn = N_dn * K_dn * raw_bits // 8

            Hinv_gu, hdiag_gu = None, None
            if not use_w4 and gptq_cal is not None and lid in gptq_cal:
                Hinv_gu, hdiag_gu = GQ2.prepare_hinv(gptq_cal[lid]["H_gu"].to(dev), damp=args.gptq_damp)

            # Pack moe.gu (1 expert at a time for L3 cache efficiency on CPU)
            if W.cur is None or W.size + E * cb_gu + P.PAGE > W.max_file:
                W._open()
            first_gu, scs_gu = None, []
            num_gu, den_gu = 0.0, 0.0
            for eid in range(E):
                w1 = loader.expert_weight(f"{mpfx}.experts.{eid}.w1")
                w3 = loader.expert_weight(f"{mpfx}.experts.{eid}.w3")
                w_blk = torch.cat([w1, w3], dim=0)[gu_order]
                if use_w4:
                    q, s = quant_w4_dev(w_blk, gs=64)
                elif Hinv_gu is not None:
                    q, s = GQ2.gptq_quant_w2(w_blk, gs=gs, mode=args.w2_mode, Hinv=Hinv_gu, hdiag=hdiag_gu)
                    q, s = q.cpu(), s.cpu()
                else:
                    q, s = fast_rtn_w2(w_blk, gs=gs, mode=args.w2_mode)
                    q, s = q.cpu(), s.cpu()
                f, o = W.put(pack_codes(q.numpy(), raw_bits).tobytes())
                if first_gu is None:
                    first_gu = (f, o)
                scs_gu.append(pack_scales(s.numpy()))
            sc_gu = np.concatenate(scs_gu)
            fs, os_ = W.put(sc_gu.tobytes())
            W.tensors[f"{dpfx}.moe.gu"] = dict(
                kind="w", N=N_gu, K=K_gu, bits=bits, gs=gs, experts=E,
                file=first_gu[0], off=first_gu[1], bytes=E * cb_gu,
                sfile=fs, soff=os_, sbytes=sc_gu.nbytes, rows={},
            )

            # Pack moe.dn
            if W.cur is None or W.size + E * cb_dn + P.PAGE > W.max_file:
                W._open()
            first_dn, scs_dn = None, []
            num_dn, den_dn = 0.0, 0.0
            for eid in range(E):
                w_blk = loader.expert_weight(f"{mpfx}.experts.{eid}.w2")
                if use_w4:
                    q, s = quant_w4_dev(w_blk, gs=64)
                else:
                    q, s = fast_rtn_w2(w_blk, gs=gs, mode=args.w2_mode)
                    q, s = q.cpu(), s.cpu()
                f, o = W.put(pack_codes(q.numpy(), raw_bits).tobytes())
                if first_dn is None:
                    first_dn = (f, o)
                scs_dn.append(pack_scales(s.numpy()))
            sc_dn = np.concatenate(scs_dn)
            fs, os_ = W.put(sc_dn.tobytes())
            W.tensors[f"{dpfx}.moe.dn"] = dict(
                kind="w", N=N_dn, K=K_dn, bits=bits, gs=gs, experts=E,
                file=first_dn[0], off=first_dn[1], bytes=E * cb_dn,
                sfile=fs, soff=os_, sbytes=sc_dn.nbytes, rows={},
            )
            tag = "W4" if use_w4 else f"W2({args.w2_mode}{'+GPTQ' if Hinv_gu is not None else ''})"
            print(f"L{lid:02d} ({'DSA' if is_dsa else 'HCA'} {tag}) "
                  f"rel_gu={num_gu/max(den_gu,1e-30):.4f} rel_dn={num_dn/max(den_dn,1e-30):.4f} "
                  f"({time.time()-lt0:.1f}s)", flush=True)

    W.close()
    manifest = {
        "format": "dankattn-ernie55-v1",
        "model_type": cfg["model_type"],
        "H": cfg["hidden_size"],
        "HC": cfg["hc_mult"],
        "nL": n_layers,
        "vocab": cfg["vocab_size"],
        "E": E,
        "topk": cfg["num_experts_per_tok"],
        "I": cfg["intermediate_size"],
        "moe_I": cfg["moe_intermediate_size"],
        "moe_lat": cfg["moe_latent_size"],
        "nh": cfg["num_attention_heads"],
        "hd": cfg["head_dim"],
        "nope": cfg["qk_nope_head_dim"],
        "rdim": cfg["qk_rope_head_dim"],
        "o_groups": cfg["o_groups"],
        "o_rank": cfg["o_lora_rank"],
        "window": cfg["sliding_window"],
        "compress_ratios": cfg["compress_ratios"][:n_layers],
        "mla_nh": cfg["hybrid_mla_num_attention_heads"],
        "mla_q_rank": cfg["hybrid_mla_q_lora_rank"],
        "mla_kv_rank": cfg["hybrid_mla_kv_lora_rank"],
        "mla_nope": cfg["hybrid_mla_qk_nope_head_dim"],
        "mla_rdim": cfg["hybrid_mla_qk_rope_head_dim"],
        "mla_vdim": cfg["hybrid_mla_v_head_dim"],
        "idx_nh": cfg["index_n_heads"],
        "idx_hd": cfg["index_head_dim"],
        "idx_topk": cfg["index_topk"],
        "eps": cfg["rms_norm_eps"],
        "hc_eps": cfg["hc_eps"],
        "hc_sinkhorn_iters": cfg["hc_sinkhorn_iters"],
        "situ_beta": cfg["activation_situ_beta"],
        "situ_lbeta": cfg["activation_situ_linear_beta"],
        "rope_theta": cfg["rope_theta"],
        "compress_rope_theta": cfg["compress_rope_theta"],
        "rope_scaling": cfg["rope_scaling"],
        "bos": cfg["bos_token_id"],
        "eos": cfg["eos_token_id"],
        "pad": cfg["pad_token_id"],
        "w2_mode": args.w2_mode,
        "files": W.files,
        "tensors": W.tensors,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    # Copy tokenizer files for serving
    for fn in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "config.json"):
        sp = os.path.join(args.src, fn)
        if os.path.exists(sp):
            import shutil
            shutil.copy2(sp, os.path.join(args.out, fn))
    tot_gb = sum(os.path.getsize(os.path.join(args.out, fn)) for fn in W.files) / (1024 ** 3)
    print(f"DONE: {args.out} ({len(W.files)} shards, {tot_gb:.2f} GiB) in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
