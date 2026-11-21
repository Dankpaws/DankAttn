# Per-tensor shape census of the BF16 checkpoint from safetensors headers (HTTP range reads).
import json, struct, sys, urllib.request, collections, re
REPO, REV = "Qwen/Qwen3.8-Flash-Next", "7ecef55a2e549814c62143fcf8b5be457695e3b9"
base = f"https://huggingface.co/{REPO}/resolve/{REV}/"
idx = json.load(urllib.request.urlopen(base + "model.safetensors.index.json"))
files = sorted(set(idx["weight_map"].values()))
def rng(f, a, b):
    r = urllib.request.Request(base + f, headers={"Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(r).read()
out = {}
for f in files:
    n = struct.unpack("<Q", rng(f, 0, 7))[0]
    h = json.loads(rng(f, 8, 8 + n - 1))
    for k, v in h.items():
        if k != "__metadata__": out[k] = {"dtype": v["dtype"], "shape": v["shape"], "file": f}
json.dump(out, open(sys.argv[1], "w"))
print(len(out), "tensors")
