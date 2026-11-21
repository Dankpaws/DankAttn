"""Hy4-flash BF16 checkpoint -> opus-hy4 pack.

  python3 pack_hy4.py <bf16 ckpt> <out pack> [importance stats.pt]

Env: HY4_EXPERT_BITS / HY4_EXPERT_GS (4 / 128), HY4_GATE_BITS, HY4_O_BITS (8: attention output
gate and o_proj, the largest dense reads), HY4_DEV (cpu | cuda, where quantization runs).

Layout (names are what engine/hy4 reads; W = register-direct mm2 weight, R = raw):
  embed R bf16 [V, H]; hch.fn/base/scale R f32; norm R; lm_head W8
  L{i}.ahc.* / L{i}.mhc.* R f32 (fn [8, 4H], base [8], scale [2]); L{i}.attn_norm, L{i}.ffn_norm R
  L{i}.attn.in   W8  K=H     rows qa | kva | (sparse) idxk | idxw (padded to 32)
  L{i}.attn.gate W8  K=H     N = nh*256
  L{i}.attn.qb   W8  K=1024  rows q (nh*256, per head [nope 192 | rope 64]) | (sparse) idxq 2048
  L{i}.attn.kb   W8  grouped over heads, N=512 K=192  (q_nope -> latent: kv_b key half, transposed)
  L{i}.attn.vb   W8  grouped over heads, N=256 K=512  (latent -> value: kv_b value half)
  L{i}.attn.o    W8  N=H K=nh*256
  R: attn.qa_norm, attn.kva_norm, attn.sinks, attn.post_norm; sparse: idx.k_norm, idx.kvp, idx.kgp,
     idx.ape, idx.key_norm (f32)
  dense L0: L0.mlp.gu W8 (gate|up interleaved by 16), L0.mlp.dn W8, L0.mlp.post_norm R
  MoE:  L{i}.moe.in  W8 K=H rows router 512 | ldn 1536 | sgu 3072 (shared gate|up interleaved by 16)
        L{i}.moe.bias R f32 [E]; L{i}.moe.gu W4 grouped (gate|up interleaved by 16), L{i}.moe.dn W4 grouped
        L{i}.moe.out W8 K = 1536 + 1536: [routed_up | shared_down] side by side; L{i}.moe.norm R
  Engram: L{i}.eng.kv W8 N=5H K=1536 rows key0..3 | value; R eng.bias [5, H], eng.lnorm [16, 96],
        eng.nk [4, H], eng.nq [4, H], eng.gamma [H], eng.hcs [4, H]; table rows in engram.bin (fp8, see
        manifest "engram")
"""
import json, math, os, sys, time
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "convert"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pack as P  # noqa: E402  (Writer, tquant, importance, gu_rows)
from layout import pack_codes, pack_scales  # noqa: E402
import gptq as GQ  # noqa: E402

torch.set_grad_enabled(False)
DEV = os.environ.get("HY4_DEV", "cpu")


_tq_cpu = P.tquant


def tquant_dev(w, bits, gs, clip=True, imp=None):
    q, s = _tq_cpu(w.to(DEV), bits, gs, clip, None if imp is None else imp.to(DEV))
    return q.cpu(), s.cpu()


if DEV != "cpu":
    P.tquant = tquant_dev


def main(src, out, stats=None):
    t0 = time.time()
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    cfg = json.load(open(os.path.join(src, "config.json")))
    tc = cfg["text_config"]
    if stats:
        P.IMP = torch.load(stats)
        print("importance stats:", len(P.IMP["s"]), "sets", flush=True)
    gptq_path = os.environ.get("HY4_GPTQ")
    gptq_cal = None
    gptq_mode = os.environ.get("HY4_GPTQ_MODE", "both")
    gptq_damp = float(os.environ.get("HY4_GPTQ_DAMP", "0.01"))
    gptq_damp_dn = float(os.environ.get("HY4_GPTQ_DAMP_DN", "0.05"))
    if gptq_path:
        gptq_cal = torch.load(gptq_path, map_location="cpu")["layers"]
        print(f"GPTQ calib: {gptq_path} ({len(gptq_cal)} MoE layers, mode={gptq_mode}, damp={gptq_damp}/{gptq_damp_dn})", flush=True)

    def pack_experts_gptq(layer_idx, d_pfx, gu3, dn3, bits, gs):
        """Pack routed experts (moe.gu, moe.dn) with GPTQ. Returns (rel_gu, rel_dn)."""
        E_, N_gu, K_gu = gu3.shape
        _, N_dn, K_dn = dn3.shape
        I_ = N_gu // 2
        cb_gu = N_gu * K_gu * bits // 8
        cb_dn = N_dn * K_dn * bits // 8
        cal = gptq_cal[layer_idx]
        Hinv_gu, hdiag_gu = GQ.prepare_hinv(cal["H_gu"], damp=gptq_damp)
        do_dn = gptq_mode == "both" and "lat" in cal and "top" in cal
        cal_lat = cal["lat"].float() if do_dn else None
        cal_top = cal["top"].long() if do_dn else None
        imp_dn_all = P.importance(d_pfx + ".moe.dn", K_dn, experts=E_)
        limit = tc["swiglu_limit"]

        # 1. Quantize + pack moe.gu in chunks of 16 experts (shares Hinv_gu across the chunk)
        if W.cur is None or W.size + E_ * cb_gu + P.PAGE > W.max_file:
            W._open()
        scs_gu, num_gu, den_gu, first_gu = [], 0.0, 0.0, None
        wgu_deq = [None] * E_ if do_dn else None
        chunk = 16
        assert (chunk * cb_gu) % P.PAGE == 0
        for e0 in range(0, E_, chunk):
            ce = min(chunk, E_ - e0)
            w = gu3[e0:e0 + ce].float().reshape(-1, K_gu)
            q, s = GQ.gptq_quant(w, bits=bits, gs=gs, clip=True, Hinv=Hinv_gu, hdiag=hdiag_gu)
            f, o = W.put(pack_codes(q.numpy(), bits).tobytes())
            first_gu = (f, o) if first_gu is None else first_gu
            scs_gu.append(pack_scales(s.numpy()))
            wq = (q.float().reshape(-1, K_gu // gs, gs) * s[..., None]).reshape(-1, K_gu)
            err = wq - w
            num_gu += float(err.pow(2).sum())
            den_gu += float(w.pow(2).sum())
            if do_dn:
                wq3 = wq.reshape(ce, N_gu, K_gu)
                for j in range(ce):
                    wgu_deq[e0 + j] = wq3[j].half()
        sc_gu = np.concatenate(scs_gu)
        fs, os_ = W.put(sc_gu.tobytes())
        W.tensors[d_pfx + ".moe.gu"] = dict(kind="w", N=N_gu, K=K_gu, bits=bits, gs=gs, experts=E_,
                                            file=first_gu[0], off=first_gu[1], bytes=E_ * cb_gu,
                                            sfile=fs, soff=os_, sbytes=sc_gu.nbytes, rows={})
        rel_gu = num_gu / max(den_gu, 1e-30)

        # 2. Quantize + pack moe.dn (sequential GPTQ if do_dn, else RTN)
        if not do_dn:
            rel_dn = W.linear_experts(d_pfx + ".moe.dn", dn3, bits, gs)
            return rel_gu, rel_dn
        if W.cur is None or W.size + E_ * cb_dn + P.PAGE > W.max_file:
            W._open()
        scs_dn, num_dn, den_dn, first_dn = [], 0.0, 0.0, None
        assert (chunk * cb_dn) % P.PAGE == 0
        for e0 in range(0, E_, chunk):
            ce = min(chunk, E_ - e0)
            qs, ss = [], []
            for j in range(ce):
                e = e0 + j
                w = dn3[e].float()
                crows, _ = torch.where(cal_top == e)
                if crows.numel() >= 8:
                    # wgu_deq[e] has rows interleaved by 16: [16 gate | 16 up] per block
                    cgu = cal_lat[crows] @ wgu_deq[e].float().t()
                    cgu_v = cgu.view(-1, I_ // 16, 2, 16)
                    cg = cgu_v[:, :, 0, :].reshape(-1, I_).clamp(max=limit)
                    cu = cgu_v[:, :, 1, :].reshape(-1, I_).clamp(-limit, limit)
                    ca = torch.nn.functional.silu(cg) * cu
                    H_dn = (ca.double().T @ ca.double()) / crows.numel()
                    q, s = GQ.gptq_quant(w, H=H_dn, bits=bits, gs=gs, damp=gptq_damp_dn, clip=True)
                else:
                    imp = None if imp_dn_all is None else imp_dn_all[e]
                    q, s = P.tquant(w, bits, gs, clip=True, imp=imp)
                wgu_deq[e] = None
                qs.append(q)
                ss.append(s)
                err = (q.float().reshape(N_dn, K_dn // gs, gs) * s[..., None]).reshape(N_dn, K_dn) - w
                num_dn += float(err.pow(2).sum())
                den_dn += float(w.pow(2).sum())
            q_cat = torch.cat(qs, 0)
            s_cat = torch.cat(ss, 0)
            f, o = W.put(pack_codes(q_cat.numpy(), bits).tobytes())
            first_dn = (f, o) if first_dn is None else first_dn
            scs_dn.append(pack_scales(s_cat.numpy()))
        sc_dn = np.concatenate(scs_dn)
        fs, os_ = W.put(sc_dn.tobytes())
        W.tensors[d_pfx + ".moe.dn"] = dict(kind="w", N=N_dn, K=K_dn, bits=bits, gs=gs, experts=E_,
                                            file=first_dn[0], off=first_dn[1], bytes=E_ * cb_dn,
                                            sfile=fs, soff=os_, sbytes=sc_dn.nbytes, rows={})
        return rel_gu, num_dn / max(den_dn, 1e-30)
    handles = {}

    def T(name, dtype=torch.float32):
        f = idx[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(src, f), framework="pt")
        t = handles[f].get_tensor(name)
        return t if dtype is None else t.to(dtype)

    W = P.Writer(out)
    rel = {}
    H, HC = tc["hidden_size"], tc["hc_mult"]
    I = tc["moe_intermediate_size"]
    LAT = tc["routed_expert_hidden_size"]
    E = tc["n_routed_experts"]
    nope, rope, vd, kvr = tc["qk_nope_head_dim"], tc["qk_rope_head_dim"], tc["v_head_dim"], tc["kv_lora_rank"]
    ebits = int(os.environ.get("HY4_EXPERT_BITS", "4"))
    egs = int(os.environ.get("HY4_EXPERT_GS", "128"))
    dbits = int(os.environ.get("HY4_DENSE_BITS", "8"))
    gbits = int(os.environ.get("HY4_GATE_BITS", str(dbits)))
    obits = int(os.environ.get("HY4_O_BITS", str(dbits)))
    lm_bits = int(os.environ.get("HY4_LMHEAD_BITS", "8"))

    def pad32(n):
        return 32 * math.ceil(n / 32)

    def lin(name, w, bits=None, gs=128, rows=None, imp=None):
        if bits is None:
            bits = dbits
        rel[name] = W.linear(name, w, bits, gs, n_pad=pad32(w.shape[0]), rows=rows, imp_key=imp)

    def cat_rows(parts):
        rows, n, ws = {}, 0, []
        for k, w in parts:
            rows[k] = [n, n + w.shape[0]]
            n += w.shape[0]
            ws.append(w)
        return torch.cat(ws, 0), rows

    def hc(dst, p):
        W.raw(dst + ".fn", T(p + "hc_fn").numpy())
        W.raw(dst + ".base", T(p + "hc_base").numpy())
        W.raw(dst + ".scale", T(p + "hc_scale").numpy())

    emb = T("model.embed_tokens.weight", torch.bfloat16)
    W.raw("embed", emb.view(torch.int16).numpy(), dtype="bf16")
    del emb
    eng_meta = {}
    if os.environ.get("HY4_LAYERS"):  # mini pack for engine tests against `hy4_golden.py --layers N`
        tc = dict(tc, num_hidden_layers=int(os.environ["HY4_LAYERS"]))
    for L in range(tc["num_hidden_layers"]):
        p = f"model.layers.{L}."
        d = f"L{L}"
        sliding = tc["layer_types"][L] == "sliding_attention"
        a = p + "self_attn."
        nh = tc["swa_num_attention_heads"] if sliding else tc["num_attention_heads"]
        hc(d + ".ahc", p + "hc_attn_layer.hc_pre.")
        hc(d + ".mhc", p + "hc_mlp_layer.hc_pre.")
        W.raw(d + ".attn_norm", T(p + "input_layernorm.weight").numpy())
        W.raw(d + ".ffn_norm", T(p + "post_attention_layernorm.weight").numpy())
        ci = a + "compressed_indexer."
        parts = [("qa", T(a + "q_a_proj.weight")), ("kva", T(a + "kv_a_proj_with_mqa.weight"))]
        if not sliding:
            parts += [("idxk", T(ci + "token_key_projection.weight")), ("idxw", T(ci + "head_weight_projection.weight"))]
        w, rows = cat_rows(parts)
        lin(d + ".attn.in", w, rows=rows, imp=d + ".attn.qa")
        lin(d + ".attn.gate", T(a + "linear_gate.weight"), gbits, 64 if gbits == 4 else 128, imp=d + ".attn.gate")
        parts = [("q", T(a + "q_b_proj.weight"))]
        if not sliding:
            parts.append(("idxq", T(ci + "query_projection.weight")))
        w, rows = cat_rows(parts)
        lin(d + ".attn.qb", w, rows=rows, imp=d + ".attn.qb")
        kvb = T(a + "kv_b_proj.weight").view(nh, nope + vd, kvr)
        kb = kvb[:, :nope, :].transpose(1, 2).contiguous()  # [nh, 512, 192]: y[c] = sum_d q[d] kvb[h, d, c]
        vb = kvb[:, nope:, :].contiguous()                   # [nh, 256, 512]
        rel[d + ".attn.kb"] = W.linear_experts(d + ".attn.kb", kb.bfloat16(), dbits, 64, chunk=8)
        rel[d + ".attn.vb"] = W.linear_experts(d + ".attn.vb", vb.bfloat16(), dbits, 128, chunk=8)
        lin(d + ".attn.o", T(a + "o_proj.weight"), obits, 64 if obits == 4 else 128, imp=d + ".attn.o")
        W.raw(d + ".attn.qa_norm", T(a + "q_a_layernorm.weight").numpy())
        W.raw(d + ".attn.kva_norm", T(a + "kv_a_layernorm.weight").numpy())
        W.raw(d + ".attn.sinks", T(a + "learnable_sink_param").numpy())
        W.raw(d + ".attn.post_norm", T(a + "post_proj_norm.weight").numpy())
        if not sliding:
            W.raw(d + ".idx.k_norm", T(ci + "token_key_norm.weight").numpy())
            W.raw(d + ".idx.kvp", T(ci + "key_value_projection.weight").numpy())
            W.raw(d + ".idx.kgp", T(ci + "key_gate_projection.weight").numpy())
            W.raw(d + ".idx.ape", T(ci + "intra_group_ape").numpy())
            W.raw(d + ".idx.key_norm", T(ci + "key_norm.weight").numpy())
        m = p + "mlp."
        if tc["mlp_layer_types"][L] == "dense":
            ff = T(m + "gate_proj.weight").shape[0]
            gu = torch.cat([T(m + "gate_proj.weight"), T(m + "up_proj.weight")], 0)[P.gu_rows(ff)]
            lin(d + ".mlp.gu", gu, imp=d + ".mlp.gu")
            lin(d + ".mlp.dn", T(m + "down_proj.weight"), imp=d + ".mlp.dn")
            W.raw(d + ".mlp.post_norm", T(m + "post_fc2_norm.weight").numpy())
        else:
            sgu = torch.cat([T(m + "shared_experts.gate_proj.weight"), T(m + "shared_experts.up_proj.weight")], 0)[P.gu_rows(I)]
            w, rows = cat_rows([("router", T(m + "gate.weight")), ("ldn", T(m + "routed_expert_down_proj.weight")), ("sgu", sgu)])
            lin(d + ".moe.in", w, rows=rows, imp=d + ".moe.router")
            W.raw(d + ".moe.bias", T(m + "gate.e_score_correction_bias").numpy())
            gu3 = T(m + "experts.gate_up_proj", torch.bfloat16)[:, P.gu_rows(I)]
            dn3 = T(m + "experts.down_proj", torch.bfloat16)
            if gptq_cal is not None and L in gptq_cal:
                rel[d + ".moe.gu"], rel[d + ".moe.dn"] = pack_experts_gptq(L, d, gu3, dn3, ebits, egs)
            else:
                rel[d + ".moe.gu"] = W.linear_experts(d + ".moe.gu", gu3, ebits, egs)
                rel[d + ".moe.dn"] = W.linear_experts(d + ".moe.dn", dn3, ebits, egs)
            del gu3, dn3
            mo = torch.cat([T(m + "routed_expert_up_proj.weight"), T(m + "shared_experts.down_proj.weight")], 1)
            if P.IMP is not None and (d + ".moe.lup") in P.IMP["s"]:
                P.IMP["s"][d + ".moe.out"] = torch.cat([P.IMP["s"][d + ".moe.lup"] / max(P.IMP["n"][d + ".moe.lup"], 1),
                                                       P.IMP["s"][d + ".moe.sdn"] / max(P.IMP["n"][d + ".moe.sdn"], 1)])
                P.IMP["n"][d + ".moe.out"] = 1
            lin(d + ".moe.out", mo, imp=d + ".moe.out")
            W.raw(d + ".moe.norm", T(m + "combine_output_norm.weight").numpy())
        if (L + 1) in tc["engram_layer_ids"]:
            e = p + "engram_layer."
            parts = [(f"key{s}", T(e + f"key_projs.{s}.weight")) for s in range(HC)] + [("value", T(e + "value_proj.weight"))]
            w, rows = cat_rows(parts)
            lin(d + ".eng.kv", w, rows=rows, imp=d + ".eng.kv")
            W.raw(d + ".eng.bias", torch.stack([T(e + f"key_projs.{s}.bias") for s in range(HC)] + [T(e + "value_proj.bias")]).numpy())
            nh_e = 2 * tc["engram_n_head_per_ngram"]
            W.raw(d + ".eng.lnorm", torch.stack([T(e + f"lookup_per_head_norms.{j}.weight") for j in range(nh_e)]).numpy())
            W.raw(d + ".eng.nk", torch.stack([T(e + f"norm_key.{s}.weight") for s in range(HC)]).numpy())
            W.raw(d + ".eng.nq", torch.stack([T(e + f"norm_query.{s}.weight") for s in range(HC)]).numpy())
            W.raw(d + ".eng.gamma", T(e + "unified_pre_gamma").numpy())
            W.raw(d + ".eng.hcs", torch.stack([T(e + f"unified_per_hc_scales.{s}") for s in range(HC)]).numpy())
            table = T(e + "lookup_table", None).numpy().astype(np.int32)
            W.raw(d + ".eng.map", table, dtype="i32")
            lid = L + 1
            half = max(1, np.iinfo(np.int64).max // (int(table.max()) + 1) // 2)  # HF _init_weights
            g = np.random.default_rng(tc["engram_seed"] + 10007 * lid)
            mult = (g.integers(0, half, size=tc["engram_max_ngram_size"], dtype=np.int64) * 2 + 1).tolist()
            sizes = [s for order in tc["engram_layer_head_vocab_sizes"][str(lid)] for s in order]
            eng_meta[d] = dict(mult=[int(x) for x in mult], primes=sizes, pad=int(table[tc["engram_pad_id"]]))
        print(f"layer {L:2d} done {time.time() - t0:6.0f}s  " + (f"experts rel.err gu {rel[d + '.moe.gu']:.2e} dn {rel[d + '.moe.dn']:.2e}" if d + ".moe.gu" in rel else ""), flush=True)
    for k in ("fn", "base", "scale"):
        W.raw("hch." + k, T("model.hc_head.hc_head_" + k).numpy())
    W.raw("norm", T("model.norm.weight").numpy())
    lin("lm_head", T("lm_head.weight"), bits=lm_bits, imp="lm_head")
    W.close()
    qdesc = f"GPTQ({gptq_mode})" if gptq_cal is not None else "importance-weighted clip search"
    man = dict(format="opus-hy4-pack-v0", source=dict(repo="tencent/Hy4-flash", revision="d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959"),
               precision=f"experts W{ebits}g{egs} {qdesc}; attn gate W{gbits}, o W{obits}; dense W{dbits}g128; lm_head W{lm_bits}; embed bf16; vectors f32",
               config=tc, files=W.files, tensors=W.tensors, rel_mse=rel, engram=eng_meta,
               layout={"gu": "gate_up_interleaved_16", "moe_out": "routed_up|shared_down along K"})
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print("done", time.time() - t0, "s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
