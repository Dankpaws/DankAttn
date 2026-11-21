"""Append a vocabulary-restricted draft head to a pack: the lm_head rows of the K most
frequent tokens of a corpus (plus every special token), W4 g128, and their ids. Drafts are
always verified by the full head, so this can change acceptance, never output.
  python3 convert/draft_head.py <bf16> <pack> <K> <files...>
"""
import collections, json, os, sys
import numpy as np
import torch
from safetensors import safe_open
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from convert.pack import Writer

src, out, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
files = sys.argv[4:]
tok = Tokenizer.from_file(os.path.join(out, "tokenizer.json")) if os.path.exists(os.path.join(out, "tokenizer.json")) \
    else Tokenizer.from_file(os.path.join(src, "tokenizer.json"))
c = collections.Counter()
for f in files:
    try:
        c.update(tok.encode(open(f, errors="ignore").read()).ids)
    except Exception:
        pass
special = [i for t, i in tok.get_vocab().items() if t.startswith("<") and t.endswith(">") and len(t) < 40]
ids = list(dict.fromkeys(special + [t for t, _ in c.most_common()]))[:K]
ids += [i for i in range(K * 4) if i not in set(ids)][: K - len(ids)]  # pad with low ids if the corpus is small
ids = sorted(ids[:K])
idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
h = safe_open(os.path.join(src, idx["lm_head.weight"]), framework="pt")
w = h.get_tensor("lm_head.weight")[torch.tensor(ids)].float()
man = json.load(open(os.path.join(out, "manifest.json")))


class App(Writer):
    def _open(self):
        if self.cur:
            self.cur.close()
        name = f"pack-dh-{len(self.files):02d}.bin"
        self.files.append(name)
        self.cur = open(os.path.join(self.out, name), "wb")
        self.size = 0


W = App(out)
r = W.linear("MTP.lm_head_sub", w, 4, 128)
W.raw("MTP.lm_head_sub_ids", np.array(ids, dtype=np.uint32), dtype="u32")
W.close()
base = len(man["files"])
for k, t in W.tensors.items():
    t = dict(t)
    t["file"] += base
    if "sfile" in t:
        t["sfile"] += base
    man["tensors"][k] = t
man["files"] += W.files
man["rel_mse"]["MTP.lm_head_sub"] = r
json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
print("draft head:", K, "tokens,", W.files, f"rel {r:.2e}")
