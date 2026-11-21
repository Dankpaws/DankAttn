"""Append Hy4-flash's DFly2 drafter (model.dfly_layers.*) to an opus-hy4 pack.

  python3 pack_dfly.py <bf16 ckpt> <pack>

New files continue the pack's numbering; the manifest gains the tensors and a "dfly" entry:
  D{i}.attn.in W8 rows qa | kva; D{i}.attn.gate, .qb, .o W8; .kb / .vb grouped per head (as the
  target's); R D{i}.attn.qa_norm, kva_norm, sinks, post_norm, D{i}.in_norm, D{i}.post_ln;
  D{i}.mlp.gu W8 (gate|up interleaved by 16), D{i}.mlp.dn W8
  DF.eh W8 K = 2H ([enorm(embed) | hnorm(target hidden)]), R DF.enorm, DF.hnorm, DF.gate
  DF.hc.gu W8 K = 2H (gate|up interleaved), DF.hc.dn W8, R DF.hc.hnorm, DF.hc.enorm, DF.norm
  lm_head_w4: W4 g64 copy of the target head for the draft steps (drafts only; verify uses lm_head)
"""
import json, math, os, sys, time
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "convert"))
import pack as P  # noqa: E402

torch.set_grad_enabled(False)


def main(src, out):
    t0 = time.time()
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    cfg = json.load(open(os.path.join(src, "config.json")))
    tc = cfg["text_config"]
    man_p = os.path.join(out, "manifest.json")
    man = json.load(open(man_p))
    handles = {}

    def T(name, dtype=torch.float32):
        f = idx[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(src, f), framework="pt")
        return handles[f].get_tensor(name).to(dtype)

    W = P.Writer(out)
    W.files = list(man["files"])  # continue the numbering; Writer opens pack-NN.bin with NN = len(files)
    rel = {}
    dbits = int(os.environ.get("HY4_DFLY_BITS", "6"))
    dgs = int(os.environ.get("HY4_DFLY_GS", "64" if dbits == 4 else "128"))
    print(f"pack_dfly: D*/DF.* W{dbits}g{dgs} (lm_head_w4 W4g64)", flush=True)

    def lin(name, w, bits=None, gs=None, rows=None):
        b = dbits if bits is None else bits
        g = dgs if gs is None else gs
        rel[name] = W.linear(name, w, b, g, n_pad=32 * math.ceil(w.shape[0] / 32), rows=rows)

    nope, vd, kvr = tc["qk_nope_head_dim"], tc["v_head_dim"], tc["kv_lora_rank"]
    nl = cfg["dfly_num_hidden_layers"]
    for i in range(nl):
        p = f"model.dfly_layers.{i}."
        a = p + "self_attn."
        d = f"D{i}"
        qa, kva = T(a + "q_a_proj.weight"), T(a + "kv_a_proj_with_mqa.weight")
        lin(d + ".attn.in", torch.cat([qa, kva], 0), rows={"qa": [0, qa.shape[0]], "kva": [qa.shape[0], qa.shape[0] + kva.shape[0]]})
        lin(d + ".attn.gate", T(a + "linear_gate.weight"))
        lin(d + ".attn.qb", T(a + "q_b_proj.weight"), rows={"q": [0, T(a + "q_b_proj.weight").shape[0]]})
        nh = T(a + "learnable_sink_param").shape[0]
        kvb = T(a + "kv_b_proj.weight").view(nh, nope + vd, kvr)
        rel[d + ".attn.kb"] = W.linear_experts(d + ".attn.kb", kvb[:, :nope, :].transpose(1, 2).contiguous().bfloat16(), dbits, 64, chunk=8)
        rel[d + ".attn.vb"] = W.linear_experts(d + ".attn.vb", kvb[:, nope:, :].contiguous().bfloat16(), dbits, dgs, chunk=8)
        lin(d + ".attn.o", T(a + "o_proj.weight"))
        W.raw(d + ".attn.qa_norm", T(a + "q_a_layernorm.weight").numpy())
        W.raw(d + ".attn.kva_norm", T(a + "kv_a_layernorm.weight").numpy())
        W.raw(d + ".attn.sinks", T(a + "learnable_sink_param").numpy())
        W.raw(d + ".attn.post_norm", T(a + "post_proj_norm.weight").numpy())
        W.raw(d + ".in_norm", T(p + "input_layernorm.weight").numpy())
        W.raw(d + ".post_ln", T(p + "post_attention_layernorm.weight").numpy())
        ff = T(p + "mlp.gate_proj.weight").shape[0]
        lin(d + ".mlp.gu", torch.cat([T(p + "mlp.gate_proj.weight"), T(p + "mlp.up_proj.weight")], 0)[P.gu_rows(ff)])
        lin(d + ".mlp.dn", T(p + "mlp.down_proj.weight"))
        print(f"{d} done {time.time() - t0:.0f}s", flush=True)
    q = "model.dfly_layers."
    lin("DF.eh", T(q + "mtp_fusion.eh_proj.weight"))
    for k, n in (("DF.enorm", "mtp_fusion.enorm.weight"), ("DF.hnorm", "mtp_fusion.hnorm.weight"), ("DF.gate", "mtp_fusion.gate"),
                 ("DF.hc.hnorm", "hidden_correction.hidden_norm.weight"), ("DF.hc.enorm", "hidden_correction.embed_norm.weight"),
                 ("DF.norm", "final_norm.weight")):
        W.raw(k, T(q + n).numpy())
    hci = T(q + "hidden_correction.gate_proj.weight").shape[0]
    lin("DF.hc.gu", torch.cat([T(q + "hidden_correction.gate_proj.weight"), T(q + "hidden_correction.up_proj.weight")], 0)[P.gu_rows(hci)])
    lin("DF.hc.dn", T(q + "hidden_correction.down_proj.weight"))
    lin("lm_head_w4", T("lm_head.weight"), 4, 64)
    W.close()
    man["files"] = W.files
    man["tensors"].update(W.tensors)
    man.setdefault("rel_mse", {}).update(rel)
    man["dfly"] = dict(layers=nl, target_layers=cfg["dfly_target_kv_layers"], index_layers=cfg["dfly_target_index_layer"],
                       mask_token_id=cfg["dfly_mask_token_id"], block=cfg["speculative_config"]["num_speculative_tokens"] + 1,
                       intermediate=ff, correction_intermediate=hci)
    json.dump(man, open(man_p, "w"), indent=1)
    print("done", time.time() - t0, "s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
