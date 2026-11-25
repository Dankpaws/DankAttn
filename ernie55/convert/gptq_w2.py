"""Group-wise 2-bit codebook (NF2 / SYM2) GPTQ quantizer for ERNIE-5.5 routed experts.

Produces (uint8 codes [N, K] in 0..3, f32 scales [N, K // gs]) drop-in compatible with
convert/layout.py (pack_codes(q, bits=2) / pack_scales(s)) and mm2.metal (WBITS=2 / WBITS=20).
"""
import torch

NF2_LUT = torch.tensor([-1.0, -0.2998, 0.2998, 1.0], dtype=torch.float32)
SYM2_LUT = torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=torch.float32)
CLIPS_W2 = (1.0, 0.96, 0.92, 0.88, 0.84, 0.80, 0.76, 0.72, 0.68, 0.64, 0.60, 0.56, 0.52)


def get_lut(mode: str, device: torch.device) -> torch.Tensor:
    lut = NF2_LUT if mode == "nf2" else SYM2_LUT
    return lut.to(device)


def prepare_hinv(H: torch.Tensor, damp: float = 0.01):
    """Compute upper-triangular Cholesky factor of (H + lambda I)^-1 in float64."""
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
    for mult in (1.0, 10.0, 100.0):
        try:
            L = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(L)
            Hinv = torch.linalg.cholesky(Hinv, upper=True)
            return Hinv.float(), diag.float().clamp_min(0.0)
        except RuntimeError:
            H[idx, idx] += lam * mult * 9.0
    inv_d = torch.rsqrt(torch.diag(H).clamp_min(1e-12))
    return torch.diag(inv_d).float(), diag.float().clamp_min(0.0)


def _group_scale_w2(w_blk: torch.Tensor, lut: torch.Tensor, imp_blk: torch.Tensor | None):
    """Importance-weighted clip-searched f16 group scale for 2-bit LUT quantization."""
    mids = 0.5 * (lut[:-1] + lut[1:])
    amax = w_blk.abs().amax(-1).clamp_min(1e-12)
    best_s = amax.half().float().clamp_min(6e-8)
    best_err = None
    for r in CLIPS_W2:
        s = (amax * r).half().float().clamp_min(6e-8)
        xn = w_blk / s[:, None]
        idx = (xn > mids[0]).long() + (xn > mids[1]).long() + (xn > mids[2]).long()
        d2 = (lut[idx] * s[:, None] - w_blk).square()
        err = (d2 * imp_blk[None, :]).sum(-1) if imp_blk is not None else d2.sum(-1)
        if best_err is None:
            best_err, best_s = err, s
        else:
            m = err < best_err
            best_err = torch.where(m, err, best_err)
            best_s = torch.where(m, s, best_s)
    return best_s


def rtn_quant_w2(w: torch.Tensor, gs: int = 64, mode: str = "nf2", imp: torch.Tensor | None = None):
    """Fast group-wise 2-bit RTN with clip search. Returns (uint8 codes [N, K], f32 scales [N, K//gs])."""
    N, K = w.shape
    lut = get_lut(mode, w.device)
    mids = 0.5 * (lut[:-1] + lut[1:])
    G = K // gs
    g = w.float().reshape(N, G, gs)
    amax = g.abs().amax(-1).clamp_min(1e-12)
    wi = None if imp is None else (imp if imp.dim() == 2 else imp[None, :].expand(N, K)).reshape(N, G, gs)
    best_s = amax.half().float().clamp_min(6e-8)
    best_err = None
    for r in CLIPS_W2:
        s = (amax * r).half().float().clamp_min(6e-8)
        xn = g / s.unsqueeze(-1)
        idx = (xn > mids[0]).long() + (xn > mids[1]).long() + (xn > mids[2]).long()
        d2 = (lut[idx] * s.unsqueeze(-1) - g).square()
        err = (d2 * wi).sum(-1) if wi is not None else d2.sum(-1)
        if best_err is None:
            best_err, best_s = err, s
        else:
            m = err < best_err
            best_err = torch.where(m, err, best_err)
            best_s = torch.where(m, s, best_s)
    xn = g / best_s.unsqueeze(-1)
    q = ((xn > mids[0]).long() + (xn > mids[1]).long() + (xn > mids[2]).long()).to(torch.uint8)
    return q.reshape(N, K), best_s


def gptq_quant_w2(
    w: torch.Tensor,
    H: torch.Tensor | None = None,
    gs: int = 64,
    mode: str = "nf2",
    damp: float = 0.01,
    Hinv: torch.Tensor | None = None,
    hdiag: torch.Tensor | None = None,
):
    """Quantize w [N, K] with group-wise 2-bit LUT GPTQ.
    Returns (uint8 codes [N, K] in 0..3, f32 scales [N, K // gs]).
    """
    N, K = w.shape
    lut = get_lut(mode, w.device)
    mids = 0.5 * (lut[:-1] + lut[1:])
    if Hinv is None:
        assert H is not None
        Hinv, hdiag = prepare_hinv(H, damp=damp)

    W = w.float().clone()
    Q = torch.empty((N, K), dtype=torch.uint8, device=w.device)
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
        s = _group_scale_w2(W1, lut, imp1)
        S[:, g0 // gs] = s
        Q1 = torch.empty((N, gs), dtype=torch.uint8, device=w.device)
        Err1 = torch.empty((N, gs), dtype=torch.float32, device=w.device)
        inv_s = 1.0 / s
        for j in range(gs):
            wj = W1[:, j]
            xn = wj * inv_s
            idx = (xn > mids[0]).long() + (xn > mids[1]).long() + (xn > mids[2]).long()
            Q1[:, j] = idx.to(torch.uint8)
            wq = lut[idx] * s
            err = (wj - wq) / Hinv1[j, j]
            Err1[:, j] = err
            if j + 1 < gs:
                W1[:, j + 1:] -= err[:, None] * Hinv1[j, j + 1:][None, :]
        Q[:, g0:g1] = Q1
        if g1 < K:
            W[:, g1:] -= Err1 @ Hinv[g0:g1, g1:]

    return Q, S


def dequant_w2(q: torch.Tensor, s: torch.Tensor, gs: int = 64, mode: str = "nf2") -> torch.Tensor:
    N, K = q.shape
    lut = get_lut(mode, q.device)
    return (lut[q.long()].reshape(N, K // gs, gs) * s.unsqueeze(-1)).reshape(N, K)


def _group_scale_w2_batched(w_blk: torch.Tensor, lut: torch.Tensor, imp_blk: torch.Tensor | None):
    """Batched importance-weighted clip-searched f16 group scale. w_blk: [B, N, gs], imp_blk: [B, gs]."""
    mids = 0.5 * (lut[:-1] + lut[1:])
    amax = w_blk.abs().amax(-1).clamp_min(1e-12)  # [B, N]
    best_s = amax.half().float().clamp_min(6e-8)
    best_err = None
    imp_w = None if imp_blk is None else imp_blk.unsqueeze(1)  # [B, 1, gs]
    for r in CLIPS_W2:
        s = (amax * r).half().float().clamp_min(6e-8)
        xn = w_blk / s.unsqueeze(-1)
        idx = (xn > mids[0]).long() + (xn > mids[1]).long() + (xn > mids[2]).long()
        d2 = (lut[idx] * s.unsqueeze(-1) - w_blk).square()
        err = (d2 * imp_w).sum(-1) if imp_w is not None else d2.sum(-1)
        if best_err is None:
            best_err, best_s = err, s
        else:
            m = err < best_err
            best_err = torch.where(m, err, best_err)
            best_s = torch.where(m, s, best_s)
    return best_s


def gptq_quant_w2_batched(
    w: torch.Tensor,
    Hinv: torch.Tensor,
    hdiag: torch.Tensor | None = None,
    gs: int = 64,
    mode: str = "nf2",
):
    """Batched group-wise 2-bit LUT GPTQ for B independent matrices with per-matrix Hessians.
    w: [B, N, K], Hinv: [B, K, K], hdiag: [B, K] or None.
    Returns (uint8 codes [B, N, K], f32 scales [B, N, K // gs]).
    """
    B, N, K = w.shape
    lut = get_lut(mode, w.device)
    mids = 0.5 * (lut[:-1] + lut[1:])

    W = w.float().clone()
    Q = torch.empty((B, N, K), dtype=torch.uint8, device=w.device)
    S = torch.empty((B, N, K // gs), dtype=torch.float32, device=w.device)
    imp = None
    if hdiag is not None:
        m = hdiag / hdiag.mean(dim=-1, keepdim=True).clamp_min(1e-30)
        imp = m + 1e-3

    for g0 in range(0, K, gs):
        g1 = g0 + gs
        W1 = W[:, :, g0:g1].clone()
        Hinv1 = Hinv[:, g0:g1, g0:g1]
        imp1 = None if imp is None else imp[:, g0:g1]
        s = _group_scale_w2_batched(W1, lut, imp1)  # [B, N]
        S[:, :, g0 // gs] = s
        Q1 = torch.empty((B, N, gs), dtype=torch.uint8, device=w.device)
        Err1 = torch.empty((B, N, gs), dtype=torch.float32, device=w.device)
        inv_s = 1.0 / s
        for j in range(gs):
            wj = W1[:, :, j]
            xn = wj * inv_s
            idx = (xn > mids[0]).long() + (xn > mids[1]).long() + (xn > mids[2]).long()
            Q1[:, :, j] = idx.to(torch.uint8)
            wq = lut[idx] * s
            err = (wj - wq) / Hinv1[:, j, j].unsqueeze(-1)
            Err1[:, :, j] = err
            if j + 1 < gs:
                W1[:, :, j + 1:] -= err.unsqueeze(-1) * Hinv1[:, j, j + 1:].unsqueeze(1)
        Q[:, :, g0:g1] = Q1
        if g1 < K:
            W[:, :, g1:] -= torch.bmm(Err1, Hinv[:, g0:g1, g1:])

    return Q, S
