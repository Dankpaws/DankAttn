"""Engram table for an opus-hy4 pack: every layer's BF16 rows as fp8 e4m3 of row * 448 / amax(row),
exactly HF's _fp8_round_trip (torch's fp8 cast). The per-head RMSNorm after the lookup makes the
dropped per-row scale irrelevant. Layers are concatenated in order; writes engram.bin into the pack
and the global row offsets + engram_table entry into its manifest.

  python3 build_engram.py <bf16 ckpt> <pack>
"""
import json, os, sys, time
import torch
from safetensors import safe_open

ckpt, pack = sys.argv[1], sys.argv[2]
idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
man_p = os.path.join(pack, "manifest.json")
man = json.load(open(man_p))
layers = sorted(man["engram"], key=lambda d: int(d[1:]))
out = open(os.path.join(pack, "engram.bin"), "wb")
base, t0, CH = 0, time.time(), 1 << 21
for d in layers:
    L = int(d[1:])
    name = f"model.layers.{L}.engram_layer.multi_head_embedding.embedding.weight"
    sl = safe_open(os.path.join(ckpt, idx[name]), framework="pt").get_slice(name)
    n = sl.get_shape()[0]
    for r0 in range(0, n, CH):
        x = sl[r0:min(n, r0 + CH)].float()
        s = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
        q = (x / s).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        out.write(q.numpy().tobytes())
    sizes = man["engram"][d]["primes"]
    offs, o = [], base
    for sz in sizes:
        offs.append(o)
        o += sz
    assert o - base == n, (d, o - base, n)
    man["engram"][d]["offsets"] = offs
    base += n
    print(f"{d}: {n} rows, {time.time() - t0:.0f}s", flush=True)
out.close()
man["engram_table"] = dict(path="engram.bin", offset=0, rows=base, dim=96, dtype="f8_e4m3")
json.dump(man, open(man_p, "w"), indent=1)
print("done", base, "rows")
