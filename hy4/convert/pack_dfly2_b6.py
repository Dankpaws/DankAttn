"""Pack AngelSlim/Hy4-flash-DFly2-B6 (3-layer LatentMoE + AttentiveCorrection) as an overlay
on an existing DankAttn Hy4-flash pack.

  python3 hy4/convert/pack_dfly2_b6.py <dfly2-b6-bf16-dir> <base-pack-dir> <out-pack-dir>

Creates <out-pack-dir> with symlinks to the base pack's non-drafter files, copies `lm_head_w4`
from the base pack's old drafter file so that file can be dropped from manifest["files"],
and writes the new drafter into pack-dfly2b6-00.bin + manifest.json.

Env overrides:
  HY4_DFLY_BITS=6      dense/attn/correction linear bits (default 6)
  HY4_DFLY_GS=128      dense/attn/correction group size (default 128; 64 if bits==4)
  HY4_DFLY_EBITS=4     routed MoE expert bits (default 4)
  HY4_DFLY_EGS=64      routed MoE gu group size (default 64)
"""
import json, math, os, sys, time
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "convert"))
import pack as P  # noqa: E402

torch.set_grad_enabled(False)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV != "cpu":
    _orig_tquant = P.tquant
    def tquant_dev(w, bits, gs, imp=None):
        q, s = _orig_tquant(w.to(DEV), bits, gs, None if imp is None else imp.to(DEV))
        return q.cpu(), s.cpu()
    P.tquant = tquant_dev


class DflyWriter(P.Writer):
    def _open(self):
        if self.cur:
            self.cur.close()
        name = f"pack-dfly2b6-{len(self.files):02d}.bin"
        self.files.append(name)
        self.cur = open(os.path.join(self.out, name), "wb")
        self.size = 0

    def copy_linear_from_base(self, name, tmeta, base_dir, base_files):
        """Copy an already-packed linear (q + s blobs) verbatim from base_dir."""
        if not self.cur:
            self._open()
        for key_f, key_o, key_b in (("file", "off", "bytes"), ("sfile", "soff", "sbytes")):
            pass
        qf = os.path.join(base_dir, base_files[tmeta["file"]])
        sf = os.path.join(base_dir, base_files[tmeta["sfile"]])
        # Determine byte sizes from shape/dtype
        N, K, bits, gs = tmeta["N"], tmeta["K"], tmeta["bits"], tmeta["gs"]
        E = tmeta.get("E", 1)
        qbytes = E * N * K * bits // 8
        sbytes = E * N * (K // gs) * 2  # fp16 scales
        with open(qf, "rb") as f:
            f.seek(tmeta["off"])
            qdata = f.read(qbytes)
        with open(sf, "rb") as f:
            f.seek(tmeta["soff"])
            sdata = f.read(sbytes)
        fi, off = self.put(qdata)
        sfi, soff = self.put(sdata)
        nt = dict(tmeta)
        nt["file"], nt["off"] = fi, off
        nt["sfile"], nt["soff"] = sfi, soff
        self.tensors[name] = nt


def main(src, base, out):
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    cfg = json.load(open(os.path.join(src, "config.json")))
    man = json.load(open(os.path.join(base, "manifest.json")))
    base_files = list(man["files"])
    sf = safe_open(os.path.join(src, "model.safetensors"), framework="pt")

    def T(name, dtype=torch.float32):
        return sf.get_tensor(name).to(dtype)

    dbits = int(os.environ.get("HY4_DFLY_BITS", "6"))
    dgs = int(os.environ.get("HY4_DFLY_GS", "64" if dbits == 4 else "128"))
    ebits = int(os.environ.get("HY4_DFLY_EBITS", "4"))
    egs = int(os.environ.get("HY4_DFLY_EGS", "64"))
    moe_dn_had = int(man.get("layout", {}).get("moe_dn_had", 0))
    print(
        f"pack_dfly2_b6: dense W{dbits}g{dgs}, experts W{ebits}g{egs}, "
        f"moe_dn_had={moe_dn_had}, dev={DEV}",
        flush=True,
    )

    H128 = None
    if moe_dn_had == 128:
        H128 = torch.tensor([[1.0]], dtype=torch.float32)
        while H128.shape[0] < 128:
            H128 = torch.cat([torch.cat([H128, H128], 1), torch.cat([H128, -H128], 1)], 0)
        H128 = H128 * (1.0 / math.sqrt(128.0))

    def had128_right(w):
        s = w.shape
        return (w.view(*s[:-1], s[-1] // 128, 128) @ H128).reshape(s)

    # Drop old D* and DF.* tensors so we can identify which base files are still needed
    kept_tensors = {
        k: v for k, v in man["tensors"].items()
        if not (k.startswith("DF.") or (len(k) >= 2 and k[0] == "D" and k[1].isdigit()))
        and k != "lm_head_w4"
    }
    used_old_fi = set()
    for v in kept_tensors.values():
        if "file" in v:
            used_old_fi.add(v["file"])
        if "sfile" in v:
            used_old_fi.add(v["sfile"])
    used_old_fi = sorted(used_old_fi)
    fi_remap = {old_i: new_i for new_i, old_i in enumerate(used_old_fi)}
    new_base_files = [base_files[i] for i in used_old_fi]
    for v in kept_tensors.values():
        if "file" in v:
            v["file"] = fi_remap[v["file"]]
        if "sfile" in v:
            v["sfile"] = fi_remap[v["sfile"]]

    # Symlink kept base files + auxiliary files (engram.bin, tokenizer*, etc.)
    base_abs = os.path.abspath(base)
    for fn in new_base_files:
        dst = os.path.join(out, fn)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.join(base_abs, fn), dst)
    for fn in os.listdir(base_abs):
        if fn == "manifest.json" or fn.startswith("pack-"):
            continue
        dst = os.path.join(out, fn)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(base_abs, fn), dst)

    W = DflyWriter(out)
    rel = {}

    def lin(name, w, bits=None, gs=None, rows=None):
        b = dbits if bits is None else bits
        g = dgs if gs is None else gs
        rel[name] = W.linear(name, w, b, g, n_pad=32 * math.ceil(w.shape[0] / 32), rows=rows)

    def cat_rows(parts):
        rows, n, ws = {}, 0, []
        for k, w in parts:
            rows[k] = [n, n + w.shape[0]]
            n += w.shape[0]
            ws.append(w)
        return torch.cat(ws, 0), rows

    # Copy lm_head_w4 verbatim from base pack so old pack-11.bin is not wired
    W.copy_linear_from_base("lm_head_w4", man["tensors"]["lm_head_w4"], base, base_files)

    nope = cfg["qk_nope_head_dim"]
    vd = cfg["v_head_dim"]
    kvr = cfg["kv_lora_rank"]
    nl = cfg["num_hidden_layers"]
    I = cfg["moe_intermediate_size"]

    for i in range(nl):
        p = f"model.layers.{i}."
        a = p + "self_attn."
        m = p + "mlp."
        d = f"D{i}"
        qa, kva = T(a + "q_a_proj.weight"), T(a + "kv_a_proj_with_mqa.weight")
        w_in, r_in = cat_rows([("qa", qa), ("kva", kva)])
        lin(d + ".attn.in", w_in, rows=r_in)
        lin(d + ".attn.gate", T(a + "linear_gate.weight"))
        qb = T(a + "q_b_proj.weight")
        lin(d + ".attn.qb", qb, rows={"q": [0, qb.shape[0]]})
        nh = T(a + "learnable_sink_param").shape[0]
        kvb = T(a + "kv_b_proj.weight").view(nh, nope + vd, kvr)
        rel[d + ".attn.kb"] = W.linear_experts(
            d + ".attn.kb", kvb[:, :nope, :].transpose(1, 2).contiguous().bfloat16(), dbits, 64, chunk=8
        )
        rel[d + ".attn.vb"] = W.linear_experts(
            d + ".attn.vb", kvb[:, nope:, :].contiguous().bfloat16(), dbits, dgs, chunk=8
        )
        lin(d + ".attn.o", T(a + "o_proj.weight"))
        W.raw(d + ".attn.qa_norm", T(a + "q_a_layernorm.weight").numpy())
        W.raw(d + ".attn.kva_norm", T(a + "kv_a_layernorm.weight").numpy())
        W.raw(d + ".attn.sinks", T(a + "learnable_sink_param").numpy())
        W.raw(d + ".attn.post_norm", T(a + "post_proj_norm.weight").numpy())
        W.raw(d + ".in_norm", T(p + "input_layernorm.weight").numpy())
        W.raw(d + ".post_ln", T(p + "post_attention_layernorm.weight").numpy())

        # LatentMoE (identical layout to Hy4 target layers)
        sgu = torch.cat(
            [T(m + "shared_experts.gate_proj.weight"), T(m + "shared_experts.up_proj.weight")], 0
        )[P.gu_rows(I)]
        w_min, r_min = cat_rows([
            ("router", T(m + "gate.weight")),
            ("ldn", T(m + "routed_expert_down_proj.weight")),
            ("sgu", sgu),
        ])
        lin(d + ".moe.in", w_min, rows=r_min)
        W.raw(d + ".moe.bias", T(m + "expert_bias").numpy())
        gu3 = T(m + "experts.gate_up_proj", torch.bfloat16)[:, P.gu_rows(I)].contiguous()
        dn3 = T(m + "experts.down_proj", torch.bfloat16)
        rel[d + ".moe.gu"] = W.linear_experts(d + ".moe.gu", gu3, ebits, egs)
        dn_in = had128_right(dn3.float()).bfloat16() if moe_dn_had == 128 else dn3
        dn_gs = 128 if moe_dn_had == 128 else egs
        rel[d + ".moe.dn"] = W.linear_experts(d + ".moe.dn", dn_in, ebits, dn_gs)
        del gu3, dn3, dn_in
        mo = torch.cat([T(m + "routed_expert_up_proj.weight"), T(m + "shared_experts.down_proj.weight")], 1)
        lin(d + ".moe.out", mo)
        W.raw(d + ".moe.norm", T(m + "combine_output_norm.weight").numpy())
        print(f"{d} done ({time.time() - t0:.1f}s)", flush=True)

    # MTP fusion + final norm + hidden correction SwiGLU
    lin("DF.eh", T("model.mtp_fusion.eh_proj.weight"))
    for k, n in (
        ("DF.enorm", "model.mtp_fusion.enorm.weight"),
        ("DF.hnorm", "model.mtp_fusion.hnorm.weight"),
        ("DF.gate", "model.mtp_fusion.gate"),
        ("DF.hc.hnorm", "model.hidden_correction.hidden_norm.weight"),
        ("DF.hc.enorm", "model.hidden_correction.embed_norm.weight"),
        ("DF.norm", "model.norm.weight"),
    ):
        W.raw(k, T(n).numpy())
    hci = T("model.hidden_correction.gate_proj.weight").shape[0]
    lin(
        "DF.hc.gu",
        torch.cat(
            [T("model.hidden_correction.gate_proj.weight"), T("model.hidden_correction.up_proj.weight")], 0
        )[P.gu_rows(hci)],
    )
    lin("DF.hc.dn", T("model.hidden_correction.down_proj.weight"))

    # AttentiveCorrection (model.hidden_correction.attention.*)
    ac = "model.hidden_correction.attention."
    ac_qa, ac_kva = T(ac + "q_in.weight"), T(ac + "kv_in.weight")
    w_acin, r_acin = cat_rows([("qa", ac_qa), ("kva", ac_kva)])
    lin("DF.ac.in", w_acin, rows=r_acin)
    ac_qb = T(ac + "q_b_proj.weight")
    lin("DF.ac.qb", ac_qb, rows={"q": [0, ac_qb.shape[0]]})
    ac_nh = T(ac + "learnable_sink_param").shape[0]
    ac_kvb = T(ac + "kv_b_proj.weight").view(ac_nh, nope + vd, kvr)
    rel["DF.ac.kb"] = W.linear_experts(
        "DF.ac.kb", ac_kvb[:, :nope, :].transpose(1, 2).contiguous().bfloat16(), dbits, 64, chunk=8
    )
    rel["DF.ac.vb"] = W.linear_experts(
        "DF.ac.vb", ac_kvb[:, nope:, :].contiguous().bfloat16(), dbits, dgs, chunk=8
    )
    lin("DF.ac.o", T(ac + "o_proj.weight"))
    lin("DF.ac.bgate", T(ac + "block_gate.weight"), 8, 128)
    for k, n in (
        ("DF.ac.qa_norm", "q_a_layernorm.weight"),
        ("DF.ac.kva_norm", "kv_a_layernorm.weight"),
        ("DF.ac.sinks", "learnable_sink_param"),
        ("DF.ac.bsink", "block_sink"),
        ("DF.ac.brel", "block_rel_bias"),
        ("DF.ac.bgbias", "block_gate_bias"),
    ):
        W.raw(k, T(ac + n).numpy())

    W.close()

    nb = len(new_base_files)
    for k, t in W.tensors.items():
        t = dict(t)
        if "file" in t:
            t["file"] += nb
        if "sfile" in t:
            t["sfile"] += nb
        kept_tensors[k] = t

    man["files"] = new_base_files + W.files
    man["tensors"] = kept_tensors
    man.setdefault("rel_mse", {}).update(rel)
    ac_src = cfg.get("attentive_correction_source_layer", -1)
    if ac_src < 0:
        ac_src += nl
    man["dfly"] = dict(
        layers=nl,
        target_layers=cfg["target_kv_layers"],
        index_layers=cfg["target_index_layer"],
        mask_token_id=cfg["mask_token_id"],
        block=cfg["block_size"],
        intermediate=I,
        correction_intermediate=hci,
        mlp_type=cfg.get("draft_mlp_type", "moe"),
        num_experts=cfg["num_experts"],
        num_experts_per_tok=cfg["num_experts_per_tok"],
        moe_latent=cfg["routed_expert_hidden_size"],
        router_scaling_factor=cfg["router_scaling_factor"],
        swiglu_limit=cfg.get("swiglu_limit", 10.0),
        attentive_correction=bool(cfg.get("attentive_correction", False)),
        ac_source_layer=ac_src,
        ac_block_gated=cfg.get("attentive_correction_block_mode", "shared") == "gated",
    )
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    sz = sum(os.path.getsize(os.path.join(out, f)) for f in W.files)
    print(f"done in {time.time() - t0:.1f}s | overlay={W.files} ({sz / 1e9:.3f} GB)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
