# Compare two f16 [T][V] logit files: relative error, top-1 agreement, KL(golden || other).
import sys, numpy as np, torch
V = int(sys.argv[3]) if len(sys.argv) > 3 else 248320
a = torch.from_numpy(np.fromfile(sys.argv[1], dtype=np.float16).astype(np.float32)).view(-1, V)
b = torch.from_numpy(np.fromfile(sys.argv[2], dtype=np.float16).astype(np.float32)).view(-1, V)
T = min(len(a), len(b)); a, b = a[:T], b[:T]
rel = ((a - b).norm(dim=-1) / a.norm(dim=-1)).numpy()
la, lb = torch.log_softmax(a.double(), -1), torch.log_softmax(b.double(), -1)
kl = (la.exp() * (la - lb)).sum(-1).numpy()
top1 = (a.argmax(-1) == b.argmax(-1)).float().numpy()
print(f"T={T} rel.err mean {rel.mean():.2e} max {rel.max():.2e} (worst pos {rel.argmax()}) | top1 agree {top1.mean():.4f} | KL mean {kl.mean():.2e} max {kl.max():.2e}")
for lo, hi in [(0, 16), (16, 64), (64, 256), (256, 2048), (2048, 10**9)]:
    s = slice(lo, min(hi, T))
    if lo < T: print(f"  pos {lo}-{min(hi, T)}: rel {rel[s].mean():.2e} KL {kl[s].mean():.2e} top1 {top1[s].mean():.3f}")
