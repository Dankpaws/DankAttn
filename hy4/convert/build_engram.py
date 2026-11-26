"""Engram table for an opus-hy4 pack: every layer's BF16 rows packed as either:
  - f8_e4m3 (96 B/row, 59.0 GiB): row * 448 / amax(row), matching HF's _fp8_round_trip
  - u4_g32  (60 B/row, 36.9 GiB): affine 4-bit group-32 (48 B packed nibbles + 3 x fp16 (scale, bias))

Layers are concatenated in order; writes engram.bin into the pack and updates manifest.json.

  python3 build_engram.py <bf16 ckpt> <pack> [--dtype f8_e4m3|u4_g32]
"""
import argparse, json, os, sys, time
import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("pack")
ap.add_argument("--dtype", choices=("f8_e4m3", "u4_g32"), default=os.environ.get("HY4_ENGRAM_DTYPE", "f8_e4m3"))
args = ap.parse_args()

ckpt, pack, dtype = args.ckpt, args.pack, args.dtype
idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
man_p = os.path.join(pack, "manifest.json")
man = json.load(open(man_p))
layers = sorted(man["engram"], key=lambda d: int(d[1:]))
out = open(os.path.join(pack, "engram.bin"), "wb")
base, t0, CH = 0, time.time(), 1 << 20
for d in layers:
    L = int(d[1:])
    name = f"model.layers.{L}.engram_layer.multi_head_embedding.embedding.weight"
    sl = safe_open(os.path.join(ckpt, idx[name]), framework="pt").get_slice(name)
    n = sl.get_shape()[0]
    for r0 in range(0, n, CH):
        x = sl[r0:min(n, r0 + CH)].float()
        if dtype == "f8_e4m3":
            s = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
            q = (x / s).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
            out.write(q.numpy().tobytes())
        else:
            xg = x.view(-1, 3, 32)
            mn_h = xg.amin(-1, keepdim=True).half()
            mx_h = xg.amax(-1, keepdim=True).half()
            sc_h = ((mx_h.float() - mn_h.float()) / 15.0).clamp(min=1e-12).half()
            q = ((xg - mn_h.float()) / sc_h.float().clamp(min=1e-12)).round().clamp(0, 15).to(torch.uint8)
            qf = q.view(-1, 48, 2)
            packed = qf[:, :, 0] | (qf[:, :, 1] << 4)
            sb = torch.stack([sc_h.squeeze(-1), mn_h.squeeze(-1)], dim=-1).contiguous().view(torch.uint8).view(-1, 12)
            out.write(torch.cat([packed, sb], dim=1).numpy().tobytes())
    sizes = man["engram"][d]["primes"]
    offs, o = [], base
    for sz in sizes:
        offs.append(o)
        o += sz
    assert o - base == n, (d, o - base, n)
    man["engram"][d]["offsets"] = offs
    base += n
    print(f"{d}: {n} rows ({dtype}), {time.time() - t0:.0f}s", flush=True)
out.close()
row_bytes = 60 if dtype == "u4_g32" else 96
man["engram_table"] = dict(path="engram.bin", offset=0, rows=base, dim=96, dtype=dtype, row_bytes=row_bytes)
json.dump(man, open(man_p, "w"), indent=1)
print(f"done {base} rows ({dtype}, {base * row_bytes / (1 << 30):.2f} GiB)")
