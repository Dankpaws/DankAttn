# Build ngram.json: where each PLE n-gram shard of an MLX 4-bit g32 pack lives on disk,
# plus the hash constants (prime head sizes, offsets, multipliers) from the HF rule.
import json, os, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
src, out, cfgp = sys.argv[1], sys.argv[2], sys.argv[3]
idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
hdrs = {}
def hdr(f):
    if f not in hdrs:
        with open(os.path.join(src, f), "rb") as h:
            n = struct.unpack("<Q", h.read(8))[0]
            hdrs[f] = (8 + n, json.loads(h.read(n)))
    return hdrs[f]
pre = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards."
nsh = 1 + max(int(k[len(pre):].split(".")[0]) for k in idx if k.startswith(pre))
shards = []
for i in range(nsh):
    rec = {}
    for part, key in (("weight", "w"), ("scales", "s"), ("biases", "b")):
        name = f"{pre}{i}.{part}"
        f = idx[name]
        base, h = hdr(f)
        rec[key] = base + h[name]["data_offsets"][0]
        rec["file"] = os.path.join(src, f)
        if part == "weight":
            rec["rows"] = h[name]["shape"][0]
            assert h[name]["shape"][1] == 20 and h[name]["dtype"] == "U32"
    shards.append(rec)
assert len({s["rows"] for s in shards}) == 1, "non-uniform shards"
# hash constants, same rule as transformers' Qwen4ExpTextNGramEmbedding
sys.path.insert(0, os.path.dirname(cfgp))
import importlib.util
spec = importlib.util.spec_from_file_location("golden", os.path.join(os.path.dirname(__file__), "..", "convert", "ngram_consts.py"))
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
tc = json.load(open(cfgp))["text_config"]
sizes, offs, mult = g.ngram_meta(tc, 0)
eos = tc["eos_token_id"][0] if isinstance(tc["eos_token_id"], list) else tc["eos_token_id"]
json.dump(dict(ngram_size=tc["ngram_size"], heads_per_ngram=tc["heads_per_ngram"], dim=160, group=32, eos=eos,
               head_sizes=sizes, head_offsets=offs, mult=mult, shards=shards), open(out, "w"))
print(nsh, "shards,", shards[0]["rows"], "rows each; head sizes", sizes[:3], "...")
