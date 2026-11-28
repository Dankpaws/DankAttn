"""Register-direct fragment layouts shared by the converter, the reference interpreter
and the Metal kernels (engine/kernels/*.metal). Lane maps were dumped from the M5's
16x32x32 cooperative tensors by bench/layout.mm.
"""
import numpy as np


def _right_map():
    # element i of lane l of the right (K x N, 32 x 32) operand -> (k, n)
    k = np.zeros((32, 32), np.int64)
    n = np.zeros((32, 32), np.int64)
    for l in range(32):
        for i in range(32):
            k[l, i] = 4 * (l & 1) + 8 * ((l >> 3) & 1) + (i & 3) + 16 * (i >> 4)
            n[l, i] = ((l >> 1) & 3) + 4 * (l >> 4) + 8 * ((i >> 2) & 3)
    return k, n


def _left_map():
    # element i of lane l of the left (M x K, 16 x 32) operand -> (m, k)
    m = np.zeros((32, 16), np.int64)
    k = np.zeros((32, 16), np.int64)
    for l in range(32):
        for i in range(16):
            k[l, i] = 4 * (l & 1) + 8 * ((l >> 3) & 1) + (i & 3) + 16 * (i >> 3)
            m[l, i] = ((l >> 1) & 3) + 4 * (l >> 4) + 8 * ((i >> 2) & 1)
    return m, k


RK, RN = _right_map()
LM, LK = _left_map()


def pack_codes(q, bits):
    """q: int array [N, K] of signed codes. Returns uint8 bytes in register-direct order:
    blocks [N/32][K/32][32 lanes] of 16 (int4) or 32 (int8) bytes."""
    N, K = q.shape
    assert N % 32 == 0 and K % 32 == 0
    NF, KS = N // 32, K // 32
    q4 = q.reshape(NF, 32, KS, 32)
    # e[nf, ks, l, i] = q[nf*32 + RN[l,i], ks*32 + RK[l,i]]
    e = q4.transpose(0, 2, 1, 3)[:, :, RN, RK]  # [NF, KS, 32, 32]
    if bits == 8:
        return e.astype(np.int8).view(np.uint8).reshape(-1)
    if bits == 2:
        # 8 bytes per lane per k-step: 32 x 2-bit codes in 0..3 packed 4 per byte
        u = e.astype(np.uint8) & 0x03
        u = u.reshape(NF, KS, 32, 8, 4)
        return (u[..., 0] | (u[..., 1] << 2) | (u[..., 2] << 4) | (u[..., 3] << 6)).reshape(-1)
    if bits == 3:
        # 12 bytes per lane per k-step: 8 bytes lo2 (2 bits x 32) + 4 bytes hi1 (1 bit x 32)
        u = (e + 4).astype(np.uint8) & 0x07  # 0..7
        lo2 = (u & 0x03).reshape(NF, KS, 32, 8, 4)
        lo_bytes = (lo2[..., 0] | (lo2[..., 1] << 2) | (lo2[..., 2] << 4) | (lo2[..., 3] << 6))
        hi1 = ((u >> 2) & 0x01).reshape(NF, KS, 32, 4, 8)
        hi_bytes = (
            hi1[..., 0] | (hi1[..., 1] << 1) | (hi1[..., 2] << 2) | (hi1[..., 3] << 3)
            | (hi1[..., 4] << 4) | (hi1[..., 5] << 5) | (hi1[..., 6] << 6) | (hi1[..., 7] << 7)
        )
        return np.concatenate([lo_bytes, hi_bytes], axis=-1).reshape(-1)
    if bits == 6:
        # 24 bytes per lane per k-step: 16 bytes lo4 (same nibble order as W4) + 8 bytes hi2 (2 bits x 32)
        u = (e + 32).astype(np.uint8)  # 0..63
        lo4 = u & 0x0F
        lo4 = lo4.reshape(NF, KS, 32, 4, 8)
        lo_bytes = (lo4[..., :4] | (lo4[..., 4:] << 4)).reshape(NF, KS, 32, 16)
        hi2 = ((u >> 4) & 0x03).reshape(NF, KS, 32, 8, 4)
        hi_bytes = (hi2[..., 0] | (hi2[..., 1] << 2) | (hi2[..., 2] << 4) | (hi2[..., 3] << 6))
        return np.concatenate([lo_bytes, hi_bytes], axis=-1).reshape(-1)
    assert bits == 4
    u = (e + 8).astype(np.uint8)  # 0..15
    u = u.reshape(NF, KS, 32, 4, 8)  # word w, element j in 0..7
    lo, hi = u[..., :4], u[..., 4:]
    return (lo | (hi << 4)).reshape(-1)


def pack_scales(s):
    """s: [N, G] float -> f16 bytes laid out [N/32][G][32]."""
    N, G = s.shape
    return s.reshape(N // 32, 32, G).transpose(0, 2, 1).astype(np.float16).reshape(-1).view(np.uint8)


def pack_nvfp4_codes(u4):
    """u4: uint8 array [N, K] of E2M1 nibble codes in 0..15.
    Returns uint8 bytes in register-direct W4 order: [N/32][K/32][32 lanes][16 bytes]."""
    N, K = u4.shape
    assert N % 32 == 0 and K % 32 == 0
    NF, KS = N // 32, K // 32
    e = u4.reshape(NF, 32, KS, 32).transpose(0, 2, 1, 3)[:, :, RN, RK]  # [NF, KS, 32, 32]
    e = e.reshape(NF, KS, 32, 4, 8)
    return (e[..., :4] | (e[..., 4:] << 4)).astype(np.uint8).reshape(-1)


def pack_mxfp4_codes(u4):
    """u4: uint8 array [N, K] of E2M1 nibble codes in 0..15 (GS=32).
    Returns uint8 bytes in register-direct W4 order: [N/32][K/32][32 lanes][16 bytes]."""
    return pack_nvfp4_codes(u4)


E2M1_VALS = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)


def quantize_mxfp4(w, gs=32, e8m0=True):
    """Quantize f32 [N, K] to MXFP4 E2M1 codes [N, K] (uint8 0..15) and scales [N, K/gs] (f32 -> f16).
    When e8m0=True, scales are rounded to exact powers of two (OCP MXFP4 spec)."""
    N, K = w.shape
    assert K % gs == 0
    g = w.reshape(N, K // gs, gs).astype(np.float32)
    amax = np.abs(g).max(-1)
    amax = np.where(amax == 0, 1e-12, amax)
    raw_s = amax / 6.0
    if e8m0:
        # Power-of-two scale >= raw_s (clamped to f16 normal range)
        exp = np.ceil(np.log2(np.maximum(raw_s, 2.0 ** -14)))
        s = np.clip(2.0 ** exp, 2.0 ** -14, 2.0 ** 15).astype(np.float16).astype(np.float32)
    else:
        s = np.where(raw_s.astype(np.float16) == 0, np.float32(6e-8), raw_s.astype(np.float16).astype(np.float32))
    wn = g / s[..., None]
    # Nearest E2M1 code per element
    pos_vals = E2M1_VALS[:8]
    sign = (wn < 0).astype(np.uint8)
    mag = np.abs(wn)
    idx = np.argmin(np.abs(mag[..., None] - pos_vals), axis=-1).astype(np.uint8)
    codes = (sign << 3) | idx
    return codes.reshape(N, K), s


def pack_mxfp8_codes(u8):
    """u8: uint8 array [N, K] of raw FP8 E4M3FN bytes.
    Returns uint8 bytes in register-direct W8 order: [N/32][K/32][32 lanes][32 bytes]."""
    N, K = u8.shape
    assert N % 32 == 0 and K % 32 == 0
    NF, KS = N // 32, K // 32
    e = u8.reshape(NF, 32, KS, 32).transpose(0, 2, 1, 3)[:, :, RN, RK]  # [NF, KS, 32, 32]
    return np.ascontiguousarray(e, dtype=np.uint8).reshape(-1)


def quantize(w, bits, gs, clip_search=True):
    """Symmetric per-(row, group) quantization. w: f32 [N, K]. Returns codes, scales [N, K/gs].
    With clip_search, each group's scale minimises squared error over a small clip grid."""
    N, K = w.shape
    if bits == 3:
        qmin, qmax = -4, 3
    elif bits == 4:
        qmin, qmax = -8, 7
    elif bits == 6:
        qmin, qmax = -32, 31
    else:
        qmin, qmax = -128, 127
    g = w.reshape(N, K // gs, gs)
    amax = np.abs(g).max(-1)
    amax = np.where(amax == 0, 1e-12, amax)
    best_s = amax / qmax
    if clip_search and bits in (3, 4):
        best_err = None
        for r in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75):
            s = amax * r / qmax
            q = np.clip(np.rint(g / s[..., None]), qmin, qmax)
            err = ((q * s[..., None] - g) ** 2).sum(-1)
            if best_err is None:
                best_err, best_s = err, s
            else:
                m = err < best_err
                best_err = np.where(m, err, best_err)
                best_s = np.where(m, s, best_s)
    s16 = best_s.astype(np.float16).astype(np.float32)  # the stored scale is what the engine uses
    s16 = np.where(s16 == 0, np.float32(6e-8), s16)
    q = np.clip(np.rint(g / s16[..., None]), qmin, qmax).astype(np.int8)
    return q.reshape(N, K), s16


def dequant(q, s, gs):
    N, K = q.shape
    return (q.reshape(N, K // gs, gs).astype(np.float32) * s[..., None]).reshape(N, K)
