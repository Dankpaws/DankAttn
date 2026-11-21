"""Engine logits (opus-hy4 score, f16 [T][V]) vs the golden (npz logits [T][V]).

  python3 compare.py golden.npz engine.f16 [more.f16 ...]
Prints mean KL(golden || engine), top-1 agreement and both NLLs of the next token.
"""
import sys
import numpy as np


def lsm(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


import os
S0 = int(os.environ.get("CMP_FROM", "0"))  # score positions >= S0 only (e.g. after a prefilled span)
g = np.load(sys.argv[1])
G, ids = g["logits"][S0:].astype(np.float64), g["ids"][S0:]
T, V = G.shape
lg = lsm(G)
nll_g = -lg[np.arange(T - 1), ids[1:]].mean()
print(f"golden: {T} tokens, NLL {nll_g:.4f}")
for p in sys.argv[2:]:
    e = np.fromfile(p, dtype=np.float16).astype(np.float64).reshape(-1, V)[S0:S0 + T]
    le = lsm(e)
    kl = (np.exp(lg) * (lg - le)).sum(-1)
    top = (G.argmax(-1) == e.argmax(-1)).mean()
    nll_e = -le[np.arange(T - 1), ids[1:]].mean()
    print(f"{p}: KL {kl.mean():.4f} (median {np.median(kl):.4f}, max {kl.max():.3f})  top-1 {100 * top:.1f}%  NLL {nll_e:.4f}")
