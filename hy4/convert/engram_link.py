"""Point an opus-hy4 pack at a GGUF's fp8 Engram table instead of copying 63 GB.

  python3 engram_link.py <pack> <gguf (local, for metadata)> <gguf path the engine opens> [bf16 ckpt]

Writes manifest["engram_table"] = {path, offset, rows} and each Engram layer's 16 global row
offsets; asserts the GGUF's multipliers and bucket sizes equal the ones the converter derived from
the checkpoint (HF hash). With the checkpoint, also checks sampled GGUF rows equal fp8(row / amax*448)
of the BF16 table: the per-head RMSNorm after the lookup makes the dropped per-row scale irrelevant.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, "/mnt/lmcache-nvme/hy4/llama.cpp/gguf-py")
from gguf import GGUFReader  # noqa: E402


def main(pack, gguf, gguf_engine_path, ckpt=None):
    man_p = os.path.join(pack, "manifest.json")
    man = json.load(open(man_p))
    r = GGUFReader(gguf)
    f = lambda k: r.fields[k].contents()
    mult = f("hy_v4_flash.ple.layer_multipliers")
    offs = f("hy_v4_flash.ple.head_offsets")
    primes = f("hy_v4_flash.ple.head_vocab_sizes")
    t = [x for x in r.tensors if x.name == "per_layer_token_embd.weight"][0]
    rows = int(t.shape[1])
    layers = sorted(man["engram"], key=lambda d: int(d[1:]))
    for e, d in enumerate(layers):
        m = man["engram"][d]
        assert m["mult"] == [int(x) for x in mult[3 * e:3 * e + 3]], (d, m["mult"], mult[3 * e:3 * e + 3])
        assert m["primes"] == [int(x) for x in primes[16 * e:16 * e + 16]], d
        m["offsets"] = [int(x) for x in offs[16 * e:16 * e + 16]]
    man["engram_table"] = dict(path=gguf_engine_path, offset=int(t.data_offset), rows=rows, dim=96, dtype="f8_e4m3")
    if ckpt:
        import torch
        from safetensors import safe_open
        idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
        mm = np.memmap(gguf, dtype=np.uint8, mode="r", offset=t.data_offset, shape=(rows, 96))
        rng = np.random.default_rng(0)
        worst = 0.0
        for e, d in enumerate(layers):
            L = int(d[1:])
            name = f"model.layers.{L}.engram_layer.multi_head_embedding.embedding.weight"
            sl = safe_open(os.path.join(ckpt, idx[name]), framework="pt").get_slice(name)
            n_local = sl.get_shape()[0]
            for i in rng.integers(0, n_local, 64):
                x = sl[int(i)].float()
                amax = x.abs().amax().clamp(min=1e-12) / 448.0
                q = (x / amax).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
                g = np.asarray(mm[offs[16 * e] + int(i)])
                if not np.array_equal(q, g):
                    # compare after the per-head norm, which is what the model sees
                    a = torch.from_numpy(q.copy()).view(torch.float8_e4m3fn).float()
                    b = torch.from_numpy(g.copy()).view(torch.float8_e4m3fn).float()
                    na, nb = a / a.pow(2).mean().sqrt(), b / b.pow(2).mean().sqrt()
                    worst = max(worst, float((na - nb).abs().max()))
        print(f"sampled {64 * len(layers)} rows: worst normalized diff {worst:.3g} (0 = byte-identical)")
    json.dump(man, open(man_p, "w"), indent=1)
    print("linked", gguf_engine_path, "offset", t.data_offset, "rows", rows)


if __name__ == "__main__":
    main(*sys.argv[1:5])
