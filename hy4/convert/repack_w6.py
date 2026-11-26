"""Fast W6 repacker for an existing opus-hy4 pack.

Copies W4 routed experts (L*.moe.gu, L*.moe.dn), DFly2 draft weights (D*, DF.*, lm_head_w4),
all raw tensors (embed, norms, HC, indexer, engram vectors), engram.bin (hardlinked), and
tokenizer files verbatim from <src_pack>, and only re-quantizes selected dense projections
from the BF16 checkpoint to W6 (or keeps them copied verbatim if configured as W8).

Usage:
  python3 repack_w6.py <bf16_ckpt> <src_pack> <dst_pack> [importance_stats.pt]

Env overrides (default 6 for dense projections, 8 for lm_head):
  HY4_DENSE_BITS=6   default bits for dense projections
  HY4_GATE_BITS      override for L*.attn.gate
  HY4_O_BITS         override for L*.attn.o
  HY4_IN_BITS        override for L*.attn.in
  HY4_QB_BITS        override for L*.attn.qb
  HY4_KVB_BITS       override for L*.attn.kb and L*.attn.vb
  HY4_MOEIO_BITS     override for L*.moe.in and L*.moe.out
  HY4_MLP_BITS       override for L0.mlp.gu and L0.mlp.dn
  HY4_ENG_BITS       override for L*.eng.kv
  HY4_LMHEAD_BITS=8  override for lm_head (default 8)
"""
import json, math, os, shutil, sys, time
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "convert"))
import pack as P  # noqa: E402

torch.set_grad_enabled(False)


def main(ckpt, src_pack, dst_pack, stats=None):
    t0 = time.time()
    src_man = json.load(open(os.path.join(src_pack, "manifest.json")))
    tc = src_man["config"]
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    if stats:
        P.IMP = torch.load(stats, weights_only=True)
        print("importance stats:", len(P.IMP["s"]), "sets", flush=True)

    dbits = int(os.environ.get("HY4_DENSE_BITS", "6"))
    bits_map = dict(
        gate=int(os.environ.get("HY4_GATE_BITS", str(dbits))),
        o=int(os.environ.get("HY4_O_BITS", str(dbits))),
        attn_in=int(os.environ.get("HY4_IN_BITS", str(dbits))),
        qb=int(os.environ.get("HY4_QB_BITS", str(dbits))),
        kvb=int(os.environ.get("HY4_KVB_BITS", str(dbits))),
        moeio=int(os.environ.get("HY4_MOEIO_BITS", str(dbits))),
        mlp=int(os.environ.get("HY4_MLP_BITS", str(dbits))),
        eng=int(os.environ.get("HY4_ENG_BITS", str(dbits))),
        lm=int(os.environ.get("HY4_LMHEAD_BITS", "8")),
    )
    print("repack_w6 bits:", bits_map, flush=True)

    os.makedirs(dst_pack, exist_ok=True)
    src_fds = [open(os.path.join(src_pack, f), "rb") for f in src_man["files"]]
    W = P.Writer(dst_pack)
    rel = dict(src_man.get("rel_mse", {}))

    def copy_bytes(file_idx, off, nbytes):
        fd = src_fds[file_idx]
        fd.seek(off)
        if W.cur is None or W.size + nbytes + P.PAGE > W.max_file:
            W._open()
        pad = (-W.size) % P.PAGE
        if pad:
            W.cur.write(b"\0" * pad)
            W.size += pad
        out_off = W.size
        rem = nbytes
        chunk = 64 << 20
        while rem > 0:
            buf = fd.read(min(rem, chunk))
            if not buf:
                raise EOFError(f"short read on {src_man['files'][file_idx]}")
            W.cur.write(buf)
            rem -= len(buf)
        W.size += nbytes
        return len(W.files) - 1, out_off

    def copy_tensor(name):
        t = dict(src_man["tensors"][name])
        f, o = copy_bytes(t["file"], t["off"], t["bytes"])
        t["file"], t["off"] = f, o
        if t["kind"] == "w":
            fs, os_ = copy_bytes(t["sfile"], t["soff"], t["sbytes"])
            t["sfile"], t["soff"] = fs, os_
        W.tensors[name] = t

    handles = {}

    def T(name, dtype=torch.float32):
        f = idx[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(ckpt, f), framework="pt")
        t = handles[f].get_tensor(name)
        return t if dtype is None else t.to(dtype)

    def pad32(n):
        return 32 * math.ceil(n / 32)

    def cat_rows(parts):
        rows, n, ws = {}, 0, []
        for k, w in parts:
            rows[k] = [n, n + w.shape[0]]
            n += w.shape[0]
            ws.append(w)
        return torch.cat(ws, 0), rows

    def requant_or_copy(name, bits, build_fn, gs=128, imp=None):
        st = src_man["tensors"][name]
        if st["bits"] == bits and st["gs"] == gs:
            copy_tensor(name)
            return
        w, rows = build_fn()
        rel[name] = W.linear(name, w, bits, gs, n_pad=pad32(w.shape[0]), rows=rows, imp_key=imp)

    HC = tc["hc_mult"]
    I = tc["moe_intermediate_size"]
    nope, vd, kvr = tc["qk_nope_head_dim"], tc["v_head_dim"], tc["kv_lora_rank"]

    copied = set()

    def mark_copy(name):
        copy_tensor(name)
        copied.add(name)

    mark_copy("embed")

    for L in range(tc["num_hidden_layers"]):
        p = f"model.layers.{L}."
        d = f"L{L}"
        sliding = tc["layer_types"][L] == "sliding_attention"
        a = p + "self_attn."
        nh = tc["swa_num_attention_heads"] if sliding else tc["num_attention_heads"]
        ci = a + "compressed_indexer."

        for suffix in (".ahc.fn", ".ahc.base", ".ahc.scale", ".mhc.fn", ".mhc.base", ".mhc.scale",
                       ".attn_norm", ".ffn_norm"):
            mark_copy(d + suffix)

        def build_in():
            parts = [("qa", T(a + "q_a_proj.weight")), ("kva", T(a + "kv_a_proj_with_mqa.weight"))]
            if not sliding:
                parts += [("idxk", T(ci + "token_key_projection.weight")), ("idxw", T(ci + "head_weight_projection.weight"))]
            return cat_rows(parts)

        requant_or_copy(d + ".attn.in", bits_map["attn_in"], build_in, imp=d + ".attn.qa")
        copied.add(d + ".attn.in")

        gate_gs = int(os.environ.get("HY4_GATE_GS", "64" if bits_map["gate"] == 4 else "128"))
        requant_or_copy(d + ".attn.gate", bits_map["gate"], lambda: (T(a + "linear_gate.weight"), None),
                        gs=gate_gs, imp=d + ".attn.gate")
        copied.add(d + ".attn.gate")

        def build_qb():
            parts = [("q", T(a + "q_b_proj.weight"))]
            if not sliding:
                parts.append(("idxq", T(ci + "query_projection.weight")))
            return cat_rows(parts)

        requant_or_copy(d + ".attn.qb", bits_map["qb"], build_qb, imp=d + ".attn.qb")
        copied.add(d + ".attn.qb")

        if src_man["tensors"][d + ".attn.kb"]["bits"] == bits_map["kvb"]:
            mark_copy(d + ".attn.kb")
            mark_copy(d + ".attn.vb")
        else:
            kvb = T(a + "kv_b_proj.weight").view(nh, nope + vd, kvr)
            kb = kvb[:, :nope, :].transpose(1, 2).contiguous()
            vb = kvb[:, nope:, :].contiguous()
            rel[d + ".attn.kb"] = W.linear_experts(d + ".attn.kb", kb.bfloat16(), bits_map["kvb"], 64, chunk=8)
            rel[d + ".attn.vb"] = W.linear_experts(d + ".attn.vb", vb.bfloat16(), bits_map["kvb"], 128, chunk=8)
            copied.add(d + ".attn.kb")
            copied.add(d + ".attn.vb")

        requant_or_copy(d + ".attn.o", bits_map["o"], lambda: (T(a + "o_proj.weight"), None),
                        gs=64 if bits_map["o"] == 4 else 128, imp=d + ".attn.o")
        copied.add(d + ".attn.o")

        for suffix in (".attn.qa_norm", ".attn.kva_norm", ".attn.sinks", ".attn.post_norm"):
            mark_copy(d + suffix)
        if not sliding:
            for suffix in (".idx.k_norm", ".idx.kvp", ".idx.kgp", ".idx.ape", ".idx.key_norm"):
                mark_copy(d + suffix)

        m = p + "mlp."
        if tc["mlp_layer_types"][L] == "dense":
            def build_mlp_gu():
                g = T(m + "gate_proj.weight")
                u = T(m + "up_proj.weight")
                return torch.cat([g, u], 0)[P.gu_rows(g.shape[0])], None

            requant_or_copy(d + ".mlp.gu", bits_map["mlp"], build_mlp_gu, imp=d + ".mlp.gu")
            copied.add(d + ".mlp.gu")
            requant_or_copy(d + ".mlp.dn", bits_map["mlp"], lambda: (T(m + "down_proj.weight"), None), imp=d + ".mlp.dn")
            copied.add(d + ".mlp.dn")
            mark_copy(d + ".mlp.post_norm")
        else:
            def build_moe_in():
                sgu = torch.cat([T(m + "shared_experts.gate_proj.weight"), T(m + "shared_experts.up_proj.weight")], 0)[P.gu_rows(I)]
                return cat_rows([("router", T(m + "gate.weight")), ("ldn", T(m + "routed_expert_down_proj.weight")), ("sgu", sgu)])

            requant_or_copy(d + ".moe.in", bits_map["moeio"], build_moe_in, imp=d + ".moe.router")
            copied.add(d + ".moe.in")
            mark_copy(d + ".moe.bias")
            mark_copy(d + ".moe.gu")
            mark_copy(d + ".moe.dn")

            def build_moe_out():
                mo = torch.cat([T(m + "routed_expert_up_proj.weight"), T(m + "shared_experts.down_proj.weight")], 1)
                if P.IMP is not None and (d + ".moe.lup") in P.IMP["s"]:
                    P.IMP["s"][d + ".moe.out"] = torch.cat([
                        P.IMP["s"][d + ".moe.lup"] / max(P.IMP["n"][d + ".moe.lup"], 1),
                        P.IMP["s"][d + ".moe.sdn"] / max(P.IMP["n"][d + ".moe.sdn"], 1),
                    ])
                    P.IMP["n"][d + ".moe.out"] = 1
                return mo, None

            requant_or_copy(d + ".moe.out", bits_map["moeio"], build_moe_out, imp=d + ".moe.out")
            copied.add(d + ".moe.out")
            mark_copy(d + ".moe.norm")

        if (L + 1) in tc["engram_layer_ids"]:
            e = p + "engram_layer."

            def build_eng_kv():
                parts = [(f"key{s}", T(e + f"key_projs.{s}.weight")) for s in range(HC)] + [("value", T(e + "value_proj.weight"))]
                return cat_rows(parts)

            requant_or_copy(d + ".eng.kv", bits_map["eng"], build_eng_kv, imp=d + ".eng.kv")
            copied.add(d + ".eng.kv")
            for suffix in (".eng.bias", ".eng.lnorm", ".eng.nk", ".eng.nq", ".eng.gamma", ".eng.hcs", ".eng.map"):
                mark_copy(d + suffix)

        print(f"layer {L:2d} done {time.time() - t0:6.1f}s", flush=True)

    for name in ("hch.fn", "hch.base", "hch.scale", "norm"):
        mark_copy(name)
    requant_or_copy("lm_head", bits_map["lm"], lambda: (T("lm_head.weight"), None), imp="lm_head")
    copied.add("lm_head")

    # Copy any remaining tensors verbatim (e.g., DFly2 D*, DF.*, lm_head_w4)
    for name in src_man["tensors"]:
        if name not in copied:
            mark_copy(name)

    W.close()
    for fd in src_fds:
        fd.close()

    man = dict(src_man)
    man["files"] = W.files
    man["tensors"] = W.tensors
    man["rel_mse"] = rel
    man["precision"] = f"{src_man.get('precision', '')} | repack_w6({bits_map})"
    json.dump(man, open(os.path.join(dst_pack, "manifest.json"), "w"), indent=1)

    # Hardlink engram.bin and copy tokenizer/aux files
    eng_src = os.path.join(src_pack, "engram.bin")
    eng_dst = os.path.join(dst_pack, "engram.bin")
    if os.path.exists(eng_src):
        if os.path.exists(eng_dst):
            os.remove(eng_dst)
        os.link(eng_src, eng_dst)
    for fn in os.listdir(src_pack):
        if fn == "manifest.json" or fn.startswith("pack-") or fn == "engram.bin":
            continue
        sp = os.path.join(src_pack, fn)
        dp = os.path.join(dst_pack, fn)
        if os.path.isfile(sp):
            shutil.copy2(sp, dp)

    total_bytes = sum(os.path.getsize(os.path.join(dst_pack, f)) for f in W.files)
    src_bytes = sum(os.path.getsize(os.path.join(src_pack, f)) for f in src_man["files"])
    print(f"done in {time.time() - t0:.1f}s: {src_bytes/1e9:.2f} GB -> {total_bytes/1e9:.2f} GB (saved {(src_bytes-total_bytes)/1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
