# Teacher-forced logits of an MLX pack (oQ4e baseline) for a uint32 id file -> f16 [T][V].
import sys, time, numpy as np
import mlx.core as mx
from mlx_vlm import load
path, idp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
ids = np.fromfile(idp, dtype=np.uint32).astype(np.int32)
t0 = time.time()
model, proc = load(path)
print("loaded", round(time.time() - t0, 1), "s", flush=True)
lm = model.language_model
x = mx.array(ids)[None]
out = lm(x)
logits = out.logits if hasattr(out, "logits") else out
logits = logits[0].astype(mx.float32)
mx.eval(logits)
np.array(logits).astype(np.float16).tofile(outp)
lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
nll = -mx.take_along_axis(lp[:-1], mx.array(ids[1:])[:, None], axis=-1).mean()
print("T", len(ids), "nll", float(nll), "time", round(time.time() - t0, 1), flush=True)
