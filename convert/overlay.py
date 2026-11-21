"""Overlay pack: re-quantize selected tensors of an existing pack from BF16 into one extra
file and write a new manifest that points everything else at the base pack's files.
  python3 convert/overlay.py <bf16> <base_pack> <out_pack> <calib.pt> <bits> <gs> <regex>
The out dir gets manifest.json + pack-ovl-00.bin; the base files are referenced by path
(relative paths in manifest "files"), so nothing is copied.
"""
import json, math, os, re, sys
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import convert.pack as P

src, base, out, calib, bits, gs, pat = sys.argv[1:8]
bits, gs = int(bits), int(gs)
rx = re.compile(pat)
man = json.load(open(os.path.join(base, "manifest.json")))
idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
P.IMP = torch.load(calib)
tc = man["config"]
handles = {}


def T(name):
    f = idx[name]
    if f not in handles:
        handles[f] = safe_open(os.path.join(src, f), framework="pt")
    return handles[f].get_tensor(name).float()


pre = "model.language_model."
SRC = {  # pack tensor -> (bf16 tensors concatenated in order)
    "gdn.in": ["linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight", "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight"],
    "gdn.out": ["linear_attn.out_proj.weight"],
    "attn.in": ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.indexer.index_qk_proj.weight"],
    "attn.o": ["self_attn.o_proj.weight"],
    "moe.sdown": ["mlp.shared_expert.down_proj.weight"],
}


class Ovl(P.Writer):
    def _open(self):
        if self.cur:
            self.cur.close()
        name = f"pack-ovl-{len(self.files):02d}.bin"
        self.files.append(name)
        self.cur = open(os.path.join(self.out, name), "wb")
        self.size = 0


W = Ovl(out)
rel = {}
for name, t in man["tensors"].items():
    if not rx.fullmatch(name):
        continue
    m = re.fullmatch(r"L(\d+)\.(.+)", name)
    L, kind = int(m.group(1)), m.group(2)
    w = torch.cat([T(f"{pre}layers.{L}.{s}") for s in SRC[kind]], 0)
    rel[name] = W.linear(name, w, bits, gs, n_pad=t["N"], rows=t.get("rows", {}))
    print(name, f"{rel[name]:.2e}", flush=True)
W.close()
# the new manifest: base files by absolute path, overlay files appended
paths = [os.path.join(os.path.relpath(base, out), f) for f in man["files"]]
nb = len(paths)
for k, t in W.tensors.items():
    t = dict(t)
    t["file"] += nb
    t["sfile"] += nb
    man["tensors"][k] = t
man["files"] = paths + W.files  # relative to the pack directory
man["rel_mse"].update(rel)
man["precision"] += f"; overlay: /{pat}/ at W{bits} g{gs}"
json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
print("overlay", W.files)
