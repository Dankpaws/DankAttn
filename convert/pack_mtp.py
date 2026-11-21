"""Append the native MTP head (mtp.*) to an existing opus pack: one extra pack file and
the tensors merged into manifest.json. Same precision plan as the trunk."""
import json, math, os, sys
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from convert.pack import Writer, hc_up_rows, gu_rows

src, out = sys.argv[1], sys.argv[2]
man = json.load(open(os.path.join(out, "manifest.json")))
idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
tc = man["config"]
H, HC, R = tc["hidden_size"], tc["hc_count"], tc["hc_lowrank"]
handles = {}


def T(name, dtype=torch.float32):
    f = idx[name]
    if f not in handles:
        handles[f] = safe_open(os.path.join(src, f), framework="pt")
    return handles[f].get_tensor(name).to(dtype)


class Appender(Writer):
    def _open(self):
        if self.cur:
            self.cur.close()
        name = f"pack-mtp-{len(self.files):02d}.bin"
        self.files.append(name)
        self.cur = open(os.path.join(self.out, name), "wb")
        self.size = 0


W = Appender(out)
rel = {}


def hc(dst, p):
    dn = T(p + "input_mix_weight_down.weight")
    rows = {"down": [0, R]}
    if (p + "block_inject_weight.weight") in idx:
        dn = torch.cat([dn, T(p + "block_inject_weight.weight")], 0)
        rows["inject"] = [R, R + HC]
    rel[dst + ".dn"] = W.linear(dst + ".dn", dn, 8, 128, n_pad=32 * math.ceil(dn.shape[0] / 32), rows=rows)
    up = T(p + "input_mix_weight_up.weight")
    if man.get("layout", {}).get("hc_up"):
        up = up[hc_up_rows(H, HC)]
    rel[dst + ".up"] = W.linear(dst + ".up", up, 8, 64)
    W.raw(dst + ".norm", T(p + "hc_norm.weight").numpy())


p = "mtp.layers.0."
d = "MTP"
hc(d + ".ahc", p + "attn_hyper_connection.")
hc(d + ".mhc", p + "mlp_hyper_connection.")
a = p + "self_attn."
ws = [T(a + "q_proj.weight"), T(a + "k_proj.weight"), T(a + "v_proj.weight"), T(a + "indexer.index_qk_proj.weight")]
n = [x.shape[0] for x in ws]
cat = torch.cat(ws, 0)
rel[d + ".attn.in"] = W.linear(d + ".attn.in", cat, 8, 128, n_pad=32 * math.ceil(cat.shape[0] / 32),
                               rows={"qg": [0, n[0]], "k": [n[0], sum(n[:2])], "v": [sum(n[:2]), sum(n[:3])], "idx": [sum(n[:3]), sum(n)]})
rel[d + ".attn.o"] = W.linear(d + ".attn.o", T(a + "o_proj.weight"), 8, 128)
for k in ["q_norm", "k_norm", "indexer.q_layernorm", "indexer.k_layernorm"]:
    W.raw(d + ".attn." + k, T(a + k + ".weight").numpy())
m = p + "mlp."
ws = [T(m + "gate.weight"), T(m + "shared_expert.gate_proj.weight"), T(m + "shared_expert.up_proj.weight"), T(m + "shared_expert_gate.weight")]
cat = torch.cat(ws, 0)
rel[d + ".moe.in"] = W.linear(d + ".moe.in", cat, 8, 128, n_pad=32 * math.ceil(cat.shape[0] / 32))
rel[d + ".moe.sdown"] = W.linear(d + ".moe.sdown", T(m + "shared_expert.down_proj.weight"), 8, 128)
egs = int(os.environ.get("OPUS_EXPERT_GS", "128"))
gu = T(m + "experts.gate_up_proj", torch.bfloat16)
if man.get("layout", {}).get("moe_gu"):
    gu = gu[:, gu_rows(tc["moe_intermediate_size"])]
rel[d + ".moe.gu"] = W.linear_experts(d + ".moe.gu", gu, 4, egs)
rel[d + ".moe.dn"] = W.linear_experts(d + ".moe.dn", T(m + "experts.down_proj", torch.bfloat16), 4, egs)
hc(d + ".mix", "mtp.hyper_connection_mixer.")
rel[d + ".fc_emb"] = W.linear(d + ".fc_emb", T("mtp.fc_embedding.weight"), 8, 128)
rel[d + ".fc_hid"] = W.linear(d + ".fc_hid", T("mtp.fc_hidden.weight"), 8, 128)
W.raw(d + ".norm_emb", T("mtp.pre_fc_norm_embedding.weight").numpy())
# draft-only lm_head copy at 4 bits: drafts are always verified by the trunk, so this
# cannot change output, only acceptance
rel[d + ".lm_head_w4"] = W.linear(d + ".lm_head_w4", T("lm_head.weight"), 4, 128)
W.raw(d + ".norm_hid", T("mtp.pre_fc_norm_hidden.weight").numpy())
W.close()
base = len(man["files"])
for k, t in W.tensors.items():
    t = dict(t)
    t["file"] += base
    if "sfile" in t:
        t["sfile"] += base
    man["tensors"][k] = t
man["files"] += W.files
man["rel_mse"].update(rel)
man["mtp"] = True
json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
print("mtp appended:", W.files, {k: f"{v:.2e}" for k, v in rel.items()})
