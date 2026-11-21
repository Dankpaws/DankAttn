# compare_chunks.py <dir>: 24 chunks x 512 of wikitext-2: KL(golden || engine), top-1, NLL; all positions and the second half (llama-perplexity scores positions 256..511)
import numpy as np, sys
d = sys.argv[1]  # dir with golden/gNN.npz (golden_chunks.py) and eNN.f16 (opus-hy4 score of cNN.bin)
def lsm(x):
    x = x - x.max(-1, keepdims=True); return x - np.log(np.exp(x).sum(-1, keepdims=True))
R = {"all": [], "half": []}
for k in range(24):
    g = np.load(f"{d}/golden/g{k:02d}.npz"); G = g["logits"].astype(np.float64); ids = g["ids"]
    V = G.shape[1]
    E = np.fromfile(f"{d}/e{k:02d}.f16", dtype=np.float16).astype(np.float64).reshape(-1, V)
    lg, le = lsm(G), lsm(E)
    kl = (np.exp(lg) * (lg - le)).sum(-1); top = G.argmax(-1) == E.argmax(-1)
    nll_g = -lg[np.arange(511), ids[1:]]; nll_e = -le[np.arange(511), ids[1:]]
    for name, s in (("all", slice(0, 512)), ("half", slice(256, 512))):
        R[name].append((kl[s], top[s], nll_g[s][: (511 if name == "all" else 255)], nll_e[s][: (511 if name == "all" else 255)]))
for name, rows in R.items():
    kl = np.concatenate([r[0] for r in rows]); top = np.concatenate([r[1] for r in rows])
    ng = np.concatenate([r[2] for r in rows]); ne = np.concatenate([r[3] for r in rows])
    ck = np.array([r[0].mean() for r in rows])
    print(f"{name:4s}: {kl.size} positions  KL {kl.mean():.4f} (median {np.median(kl):.4f}, p99 {np.percentile(kl,99):.3f}, chunk means {ck.min():.3f}-{ck.max():.3f})  "
          f"top-1 {100*top.mean():.1f}%  PPL golden {np.exp(ng.mean()):.3f} engine {np.exp(ne.mean()):.3f}")
