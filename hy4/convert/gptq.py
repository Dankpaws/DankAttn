"""Group-wise symmetric GPTQ quantizer compatible with convert/pack.py (W4/W8, f16 group scales).

Produces (int8 codes [N, K], f32 scales [N, K // gs]) drop-in compatible with
convert/layout.py (pack_codes / pack_scales) and opus-hy4 Metal kernels.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "convert"))
import pack as P  # noqa: E402


def prepare_hinv(H: torch.Tensor, damp: float = 0.01):
    """Compute upper-triangular Cholesky factor of (H + lambda I)^-1 in float64 for stability."""
    H = H.double().clone()
    K = H.shape[0]
    diag = torch.diag(H)
    dead = diag <= 0
    if dead.any():
        H[dead, dead] = 1.0
    mean_d = diag[~dead].mean() if (~dead).any() else torch.tensor(1.0, dtype=torch.float64)
    lam = damp * mean_d.clamp_min(1e-12)
    idx = torch.arange(K, device=H.device)
    H[idx, idx] += lam
    # Retry with stronger damping if Cholesky fails on ill-conditioned H
    for mult in (1.0, 10.0, 100.0):
        try:
            L = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(L)
            Hinv = torch.linalg.cholesky(Hinv, upper=True)
            return Hinv.float(), diag.float().clamp_min(0.0)
        except RuntimeError:
            H[idx, idx] += lam * mult * 9.0
    # Fallback: diagonal inverse
    inv_d = torch.rsqrt(torch.diag(H).clamp_min(1e-12))
    return torch.diag(inv_d).float(), diag.float().clamp_min(0.0)


def _group_scale(w_blk: torch.Tensor, bits: int, clip: bool, imp_blk: torch.Tensor | None):
    """Importance-weighted clip-searched f16 group scale matching convert/pack.py."""
    qmax, qmin = (1 << (bits - 1)) - 1, -(1 << (bits - 1))
    amax = w_blk.abs().amax(-1).clamp_min(1e-12)
    best = amax / qmax
    if clip:
        clips = P.CLIPS_V1 if bits <= 4 else P.CLIPS_V1_W8
        berr = None
        for r in clips:
            s = (amax * r / qmax).half().float()
            q = torch.clamp(torch.round(w_blk / s[:, None]), qmin, qmax)
            d2 = (q * s[:, None] - w_blk) ** 2
            err = (d2 * imp_blk[None, :]).sum(-1) if imp_blk is not None else d2.sum(-1)
            if berr is None:
                berr, best = err, s
            else:
                m = err < berr
                berr = torch.where(m, err, berr)
                best = torch.where(m, s, best)
    return best.half().float().clamp_min(6e-8)


def gptq_quant(
    w: torch.Tensor,
    H: torch.Tensor | None = None,
    bits: int = 4,
    gs: int = 64,
    damp: float = 0.01,
    clip: bool = True,
    Hinv: torch.Tensor | None = None,
    hdiag: torch.Tensor | None = None,
):
    """Quantize w [N, K] with group-wise symmetric GPTQ.

    Pass either H [K, K] or precomputed (Hinv, hdiag) from prepare_hinv(H, damp)
    when quantizing many experts that share the same input Hessian (e.g. moe.gu).
    Returns (q int8 [N, K], scales f32 [N, K // gs]).
    """
    N, K = w.shape
    qmax, qmin = (1 << (bits - 1)) - 1, -(1 << (bits - 1))
    if Hinv is None:
        assert H is not None, "gptq_quant needs H or precomputed Hinv"
        Hinv, hdiag = prepare_hinv(H, damp=damp)

    W = w.float().clone()
    Q = torch.empty((N, K), dtype=torch.int8, device=w.device)
    S = torch.empty((N, K // gs), dtype=torch.float32, device=w.device)
    imp = None
    if hdiag is not None:
        m = hdiag / hdiag.mean().clamp_min(1e-30)
        imp = m + 1e-3

    for g0 in range(0, K, gs):
        g1 = g0 + gs
        W1 = W[:, g0:g1].clone()
        Hinv1 = Hinv[g0:g1, g0:g1]
        imp1 = None if imp is None else imp[g0:g1]
        s = _group_scale(W1, bits, clip, imp1)
        S[:, g0 // gs] = s
        Q1 = torch.empty((N, gs), dtype=torch.int8, device=w.device)
        Err1 = torch.empty((N, gs), dtype=torch.float32, device=w.device)
        inv_s = 1.0 / s
        for j in range(gs):
            wj = W1[:, j]
            qj = torch.clamp(torch.round(wj * inv_s), qmin, qmax)
            Q1[:, j] = qj.to(torch.int8)
            err = (wj - qj * s) / Hinv1[j, j]
            Err1[:, j] = err
            if j + 1 < gs:
                W1[:, j + 1:] -= err[:, None] * Hinv1[j, j + 1:][None, :]
        Q[:, g0:g1] = Q1
        if g1 < K:
            W[:, g1:] -= Err1 @ Hinv[g0:g1, g1:]

    return Q, S
