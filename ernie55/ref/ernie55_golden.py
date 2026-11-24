#!/usr/bin/env python3
"""Pure-PyTorch Golden Reference & 2-bit Quantization Evaluator for ERNIE-5.5-Flash-Preview.

Runs layer-by-layer on GPU (or CPU) directly from the MXFP4+FP8 HuggingFace checkpoint,
matching baidu/vLLM-ERNIE math exactly (HCA C128 + VHA + inverse RoPE + grouped wo_a,
DSA MLA + Lightning Indexer + SWA, Sinkhorn mHC, Multi-Router Latent MoE with SiTU-GLU,
and MultiMax lm_head).

Usage:
  python3 ernie55_golden.py --ckpt /mnt/lmcache-nvme/ernie55/hf-mxfp4 --text "The capital of France is"
  python3 ernie55_golden.py --ckpt ... --ids_file ... --save_ref ref.pt
  python3 ernie55_golden.py --ckpt ... --ids_file ... --compare_ref ref.pt --expert_quant w2_g64_nf2
"""
import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.set_grad_enabled(False)

E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

# Optimal 4-level symmetric Lloyd-Max codebook for N(0, 1), normalized to [-1, 1]
NF2_LUT = torch.tensor([-1.0, -0.2998, 0.2998, 1.0], dtype=torch.float32)
SYM2_LUT = torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=torch.float32)


class CkptLoader:
    """LRU-cached safetensors loader for MXFP4+FP8 ERNIE-5.5 checkpoints."""

    def __init__(self, ckpt_dir: str, device: str = "cuda", max_open: int = 8):
        self.ckpt_dir = ckpt_dir
        self.device = torch.device(device)
        with open(os.path.join(ckpt_dir, "config.json")) as f:
            self.cfg = json.load(f)
        with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
            self.wmap = json.load(f)["weight_map"]
        self._handles: OrderedDict[str, safe_open] = OrderedDict()
        self._max_open = max_open
        self._e2m1 = E2M1_LUT.to(self.device)

    def _sf(self, fname: str):
        if fname in self._handles:
            self._handles.move_to_end(fname)
            return self._handles[fname]
        if len(self._handles) >= self._max_open:
            self._handles.popitem(last=False)
        sf = safe_open(os.path.join(self.ckpt_dir, fname), framework="pt", device="cpu")
        self._handles[fname] = sf
        return sf

    def has(self, key: str) -> bool:
        return key in self.wmap

    def raw(self, key: str, to_dev: bool = True) -> torch.Tensor:
        t = self._sf(self.wmap[key]).get_tensor(key)
        return t.to(self.device, non_blocking=True) if to_dev else t

    def linear_weight(self, prefix: str, block=(128, 128)) -> torch.Tensor:
        """Load dense linear weight [N, K] as float32 on device (dequantizing FP8 if needed)."""
        w_key = prefix if self.has(prefix) else prefix + ".weight"
        s_key = (
            prefix + "_scale_inv"
            if self.has(prefix + "_scale_inv")
            else prefix + ".weight_scale_inv"
        )
        w = self.raw(w_key)
        if w.dtype == torch.float8_e4m3fn and self.has(s_key):
            s = self.raw(s_key).float()
            n, k = w.shape
            bn, bk = block
            scale = s.repeat_interleave(bn, 0)[:n].repeat_interleave(bk, 1)[:, :k]
            return w.float() * scale
        return w.float()

    def expert_weight(self, key_prefix: str) -> torch.Tensor:
        """Load and dequantize one MXFP4 expert weight [N, K] as float32 on device."""
        w = self.raw(key_prefix + ".weight")
        s = self.raw(key_prefix + ".weight_scale_inv")
        if w.dtype == torch.int8:
            u = w.view(torch.uint8)
            lo = (u & 0x0F).long()
            hi = ((u >> 4) & 0x0F).long()
            n, k2 = u.shape
            out = torch.empty((n, k2 * 2), dtype=torch.float32, device=self.device)
            out[:, 0::2] = self._e2m1[lo]
            out[:, 1::2] = self._e2m1[hi]
            # s is float8_e8m0fnu [N, K//32]
            scale = s.float().repeat_interleave(32, dim=-1)
            return out * scale
        elif w.dtype == torch.float8_e4m3fn:
            return self.linear_weight(key_prefix)
        return w.float()


# ---- 2-bit Quantization Simulators (fake-quant on-the-fly for active experts) ----

def quant_dequant_2bit_lut(w: torch.Tensor, gs: int, lut: torch.Tensor, clip: bool = True) -> torch.Tensor:
    """Group-wise symmetric 2-bit LUT quantizer with f16 group scale."""
    N, K = w.shape
    g = w.reshape(N, K // gs, gs)
    amax = g.abs().amax(-1).clamp_min(1e-12)
    lut = lut.to(w.device)
    # Decision boundaries are midpoints between sorted LUT levels
    mids = 0.5 * (lut[:-1] + lut[1:])  # [3]

    def _eval(s):
        # s: [N, G]
        xn = g / s.unsqueeze(-1)
        idx = (xn > mids[0]).long() + (xn > mids[1]).long() + (xn > mids[2]).long()
        wq = lut[idx] * s.unsqueeze(-1)
        return wq

    best_s = amax
    if clip:
        best_err = None
        for r in (1.0, 0.92, 0.84, 0.76, 0.68, 0.60, 0.52):
            s = (amax * r).half().float().clamp_min(6e-8)
            wq = _eval(s)
            err = (wq - g).square().sum(-1)
            if best_err is None:
                best_err, best_s = err, s
            else:
                m = err < best_err
                best_err = torch.where(m, err, best_err)
                best_s = torch.where(m, s, best_s)
    else:
        best_s = best_s.half().float().clamp_min(6e-8)
    return _eval(best_s).reshape(N, K)


def quant_dequant_2bit_asym(w: torch.Tensor, gs: int = 128, clip: bool = True) -> torch.Tensor:
    """Group-wise asymmetric 2-bit {0,1,2,3}*s + z with f16 scale and f16 zero."""
    N, K = w.shape
    g = w.reshape(N, K // gs, gs)
    wmin = g.amin(-1)
    wmax = g.amax(-1)
    rng = (wmax - wmin).clamp_min(1e-12)

    def _eval(s, z):
        q = torch.clamp(torch.round((g - z.unsqueeze(-1)) / s.unsqueeze(-1)), 0, 3)
        return q * s.unsqueeze(-1) + z.unsqueeze(-1)

    best_s = (rng / 3.0).half().float().clamp_min(6e-8)
    best_z = wmin.half().float()
    if clip:
        best_err = None
        mid = 0.5 * (wmin + wmax)
        half_r = 0.5 * rng
        for r in (1.0, 0.92, 0.84, 0.76, 0.68, 0.60):
            lo = (mid - half_r * r).half().float()
            s = (2.0 * half_r * r / 3.0).half().float().clamp_min(6e-8)
            wq = _eval(s, lo)
            err = (wq - g).square().sum(-1)
            if best_err is None:
                best_err, best_s, best_z = err, s, lo
            else:
                m = err < best_err
                best_err = torch.where(m, err, best_err)
                best_s = torch.where(m, s, best_s)
                best_z = torch.where(m, lo, best_z)
    return _eval(best_s, best_z).reshape(N, K)


def quant_dequant_w3_sym(w: torch.Tensor, gs: int = 64) -> torch.Tensor:
    """Symmetric 3-bit {-4..3} * f16 scale (for mixed-precision comparison)."""
    N, K = w.shape
    g = w.reshape(N, K // gs, gs)
    amax = g.abs().amax(-1).clamp_min(1e-12)
    best_s = (amax / 3.0).half().float().clamp_min(6e-8)
    best_err = None
    for r in (1.0, 0.92, 0.85, 0.78, 0.70):
        s = (amax * r / 3.0).half().float().clamp_min(6e-8)
        q = torch.clamp(torch.round(g / s.unsqueeze(-1)), -4, 3)
        err = (q * s.unsqueeze(-1) - g).square().sum(-1)
        if best_err is None:
            best_err, best_s = err, s
        else:
            m = err < best_err
            best_err = torch.where(m, err, best_err)
            best_s = torch.where(m, s, best_s)
    q = torch.clamp(torch.round(g / best_s.unsqueeze(-1)), -4, 3)
    return (q * best_s.unsqueeze(-1)).reshape(N, K)


def apply_expert_quant(w: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return w
    if mode == "w2_g64_sym":
        return quant_dequant_2bit_lut(w, 64, SYM2_LUT)
    if mode == "w2_g64_nf2":
        return quant_dequant_2bit_lut(w, 64, NF2_LUT)
    if mode == "w2_g32_nf2":
        return quant_dequant_2bit_lut(w, 32, NF2_LUT)
    if mode == "w2_g128_asym":
        return quant_dequant_2bit_asym(w, 128)
    if mode == "w2_g64_asym":
        return quant_dequant_2bit_asym(w, 64)
    if mode == "w3_g64_sym":
        return quant_dequant_w3_sym(w, 64)
    raise ValueError(f"Unknown expert_quant mode: {mode}")


# ---- Math Primitives ----

def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    xf = x.float()
    r = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    y = xf * r
    if weight is not None:
        y = y * weight.float()
    return y


def situ_and_mul(gate: torch.Tensor, up: torch.Tensor, beta: float = 4.0, linear_beta: float = 25.0) -> torch.Tensor:
    g = gate.float()
    u = up.float()
    g = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u / linear_beta)
    return g * u


def hadamard_128(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Walsh-Hadamard transform along last dim (size 128)."""
    orig_shape = x.shape
    h = x.float().reshape(-1, 128)
    # 7 butterfly stages
    step = 1
    for _ in range(7):
        h = h.view(-1, 128 // (2 * step), 2, step)
        a = h[:, :, 0, :]
        b = h[:, :, 1, :]
        h = torch.stack([a + b, a - b], dim=2).reshape(-1, 128)
        step *= 2
    return (h * (128.0 ** -0.5)).reshape(orig_shape)


def build_yarn_cos_sin(
    max_pos: int,
    dim: int,
    base: float,
    factor: float,
    orig_max_pos: int,
    beta_fast: float,
    beta_slow: float,
    device: torch.device,
):
    """DeepSeek YaRN inverse frequencies (mscale=0) matching vLLM build_deepseek_v4_rope."""
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    inv_freq_extrap = 1.0 / pos_freqs
    inv_freq_interp = 1.0 / (factor * pos_freqs)

    def find_corr_dim(num_rotations):
        return (dim * math.log(orig_max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(find_corr_dim(beta_fast)), 0)
    high = min(math.ceil(find_corr_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    linear = (torch.arange(dim // 2, dtype=torch.float32, device=device) - low) / (high - low)
    ramp = torch.clamp(linear, 0.0, 1.0)
    # ramp=0 -> high freq (extrap), ramp=1 -> low freq (interp)
    inv_freq = inv_freq_interp * ramp + inv_freq_extrap * (1.0 - ramp)

    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def build_plain_cos_sin(max_pos: int, dim: int, base: float, device: torch.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """GPT-J interleaved RoPE on x[..., D] where cos/sin are broadcastable to [..., D/2]."""
    even = x[..., 0::2]
    odd = x[..., 1::2]
    if not inverse:
        out_e = even * cos - odd * sin
        out_o = odd * cos + even * sin
    else:
        out_e = even * cos + odd * sin
        out_o = odd * cos - even * sin
    out = torch.empty_like(x)
    out[..., 0::2] = out_e
    out[..., 1::2] = out_o
    return out


def apply_rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """NeoX half-split RoPE on x[..., D]."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ---- mHC ----

def mhc_pre(residual: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor,
            rms_eps: float, hc_eps: float, sinkhorn_iters: int):
    """ERNIE 5.5 mHC pre: RMS factor is 1 / (sqrt(mean) + rms_eps)."""
    T, M, H = residual.shape
    x = residual.reshape(T, M * H).float()
    # fn in checkpoint is [16384, 24] -> transpose to [24, 16384]
    mixes = (x @ fn.float()) / (x.square().mean(dim=-1, keepdim=True).sqrt() + rms_eps)
    pre_mix = torch.sigmoid(mixes[:, :M] * scale[0] + base[:M]) + hc_eps
    post_mix = torch.sigmoid(mixes[:, M:2 * M] * scale[1] + base[M:2 * M]) * 2.0
    comb = mixes[:, 2 * M:].reshape(T, M, M) * scale[2] + base[2 * M:].reshape(1, M, M)
    comb = torch.softmax(comb, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    layer_input = (pre_mix.unsqueeze(-1) * residual.float()).sum(dim=1)
    return post_mix.unsqueeze(-1), comb, layer_input


def mhc_post(x: torch.Tensor, residual: torch.Tensor, post_mix: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    mixed = torch.einsum("tij,tih->tjh", comb.float(), residual.float())
    return mixed + post_mix.float() * x.float().unsqueeze(1)


def mhc_head_contract(residual: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor,
                      rms_eps: float, hc_eps: float) -> torch.Tensor:
    """Final mHC contraction before model.norm. Uses rsqrt(mean + rms_eps) per vLLM tilelang kernel."""
    T, M, H = residual.shape
    x = residual.reshape(T, M * H).float()
    # Check shape of fn: [16384, 4] or [4, 16384]
    if fn.shape[0] == M * H:
        mixes = x @ fn.float()
    else:
        mixes = x @ fn.float().t()
    r = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + rms_eps)
    pre_mix = torch.sigmoid(mixes * r * scale.float() + base.float()) + hc_eps
    return (pre_mix.unsqueeze(-1) * residual.float()).sum(dim=1)


# ---- Attention Layers ----

def run_hca_layer(loader: CkptLoader, lid: int, x: torch.Tensor, positions: torch.Tensor,
                  cos_hca: torch.Tensor, sin_hca: torch.Tensor, cfg: dict) -> torch.Tensor:
    pfx = f"model.layers.{lid}.self_attn"
    T, H = x.shape
    nh = cfg["num_attention_heads"]  # 64
    hd = cfg["head_dim"]  # 512
    nope = cfg["qk_nope_head_dim"]  # 448
    rdim = cfg["qk_rope_head_dim"]  # 64
    win = cfg["sliding_window"]  # 128
    cr = cfg["compress_ratios"][lid]  # 128
    eps = cfg["rms_norm_eps"]

    w_qd = loader.linear_weight(f"{pfx}.linear_q_down_proj")  # [1024, 4096]
    w_kv = loader.linear_weight(f"{pfx}.linear_kv_proj")      # [512, 4096]
    qr = rms_norm(x @ w_qd.t(), loader.raw(f"{pfx}.q_layernorm.weight"), eps)
    kv = rms_norm(x @ w_kv.t(), loader.raw(f"{pfx}.kv_layernorm.weight"), eps)  # [T, 512]

    w_qu = loader.linear_weight(f"{pfx}.linear_q_up_proj")    # [32768, 1024]
    q = (qr @ w_qu.t()).view(T, nh, hd)
    # Per-head unweighted RMSNorm on Q before RoPE
    q = rms_norm(q, None, eps)

    # Apply HCA YaRN RoPE on last 64 dims of Q and KV
    c_pos = cos_hca[positions]  # [T, 32]
    s_pos = sin_hca[positions]
    q_r = q.clone()
    q_r[..., nope:] = apply_rope_interleaved(q[..., nope:], c_pos.unsqueeze(1), s_pos.unsqueeze(1))
    kv_r = kv.clone()
    kv_r[..., nope:] = apply_rope_interleaved(kv[..., nope:], c_pos, s_pos)

    # C128 Compressor
    num_blocks = T // cr
    comp_kv = None
    if num_blocks > 0:
        w_ckv = loader.linear_weight(f"{pfx}.core_attention.compressor.linear_wkv")     # [512, 4096]
        w_cgate = loader.linear_weight(f"{pfx}.core_attention.compressor.linear_wgate") # [512, 4096]
        ape = loader.raw(f"{pfx}.core_attention.compressor.ape").float()                # [128, 512]
        cnorm_w = loader.raw(f"{pfx}.core_attention.compressor.norm.weight")            # [512]

        x_blk = x[:num_blocks * cr]
        c_kv_raw = (x_blk @ w_ckv.t()).view(num_blocks, cr, hd)
        c_sc_raw = (x_blk @ w_cgate.t()).view(num_blocks, cr, hd) + ape.unsqueeze(0)
        c_w = torch.softmax(c_sc_raw, dim=1)
        c_sum = (c_kv_raw * c_w).sum(dim=1)  # [num_blocks, 512]
        c_normed = rms_norm(c_sum, cnorm_w, eps)
        # RoPE at position block_idx * 128
        b_pos = torch.arange(num_blocks, device=x.device) * cr
        comp_kv = c_normed.clone()
        comp_kv[..., nope:] = apply_rope_interleaved(c_normed[..., nope:], cos_hca[b_pos], sin_hca[b_pos])

    # MQA Attention with compressed blocks + SWA-128 + attn_sink
    sink = loader.raw(f"{pfx}.core_attention.attn_sink").float()  # [64]
    scale = hd ** -0.5
    o = torch.empty((T, nh, hd), dtype=torch.float32, device=x.device)

    # Chunk over queries to keep memory small
    q_chunk = 128
    for t0 in range(0, T, q_chunk):
        t1 = min(t0 + q_chunk, T)
        for t in range(t0, t1):
            nc = (t + 1) // cr
            swa_s = max(t - win + 1, 0)
            k_swa = kv_r[swa_s:t + 1]  # [L_swa, 512]
            if nc > 0:
                k_all = torch.cat([comp_kv[:nc], k_swa], dim=0)
            else:
                k_all = k_swa
            # q_r[t]: [64, 512], k_all: [L, 512]
            scores = (q_r[t] @ k_all.t()) * scale  # [64, L]
            # Append sink logit (value = 0)
            scores_s = torch.cat([scores, sink.unsqueeze(1)], dim=1)  # [64, L+1]
            probs = torch.softmax(scores_s, dim=-1)[:, :-1]  # [64, L]
            o[t] = probs @ k_all  # [64, 512]

    # VHA PostMix: (I + V @ U^T) @ o
    U = loader.raw(f"{pfx}.vha_postmix_U").float()  # [64, 16]
    V = loader.raw(f"{pfx}.vha_postmix_V").float()  # [64, 16]
    M_vha = torch.eye(nh, device=x.device) + V @ U.t()  # [64, 64]
    o = torch.einsum("hg,tgd->thd", M_vha, o)

    # Inverse RoPE on last 64 dims
    o[..., nope:] = apply_rope_interleaved(o[..., nope:], c_pos.unsqueeze(1), s_pos.unsqueeze(1), inverse=True)

    # Grouped wo_a: [8, 1024, 4096]
    n_groups = cfg["o_groups"]  # 8
    o_rank = cfg["o_lora_rank"]  # 1024
    wo_a = loader.linear_weight(f"{pfx}.linear_o_group_proj").view(n_groups, o_rank, -1)  # [8, 1024, 4096]
    o_grp = o.reshape(T, n_groups, -1)  # [T, 8, 4096]
    z = torch.einsum("tgd,god->tgo", o_grp, wo_a).reshape(T, n_groups * o_rank)  # [T, 8192]

    # Low-rank output gate from qr
    w_gate = loader.linear_weight(f"{pfx}.gate_proj")  # [8192, 1024]
    gate = torch.sigmoid(qr @ w_gate.t())
    w_ob = loader.linear_weight(f"{pfx}.o_proj")       # [4096, 8192]
    return (z * gate) @ w_ob.t()


def run_dsa_layer(loader: CkptLoader, lid: int, x: torch.Tensor, positions: torch.Tensor,
                  cos_mla: torch.Tensor, sin_mla: torch.Tensor, cfg: dict) -> torch.Tensor:
    pfx = f"model.layers.{lid}.self_attn"
    T, H = x.shape
    nh = cfg["hybrid_mla_num_attention_heads"]  # 64
    q_rank = cfg["hybrid_mla_q_lora_rank"]      # 1024
    kv_rank = cfg["hybrid_mla_kv_lora_rank"]    # 512
    nope = cfg["hybrid_mla_qk_nope_head_dim"]   # 192
    rdim = cfg["hybrid_mla_qk_rope_head_dim"]   # 64
    vdim = cfg["hybrid_mla_v_head_dim"]         # 256
    qk_dim = nope + rdim                        # 256
    win = cfg["sliding_window"]                 # 128
    topk = cfg["index_topk"]                    # 2048
    eps = cfg["rms_norm_eps"]

    w_qa = loader.linear_weight(f"{pfx}.q_a_proj")             # [1024, 4096]
    w_kva = loader.linear_weight(f"{pfx}.kv_a_proj_with_mqa")  # [576, 4096]
    q_c = rms_norm(x @ w_qa.t(), loader.raw(f"{pfx}.q_a_layernorm.weight"), eps)
    kv_lora = x @ w_kva.t()
    kv_c = rms_norm(kv_lora[:, :kv_rank], loader.raw(f"{pfx}.kv_a_layernorm.weight"), eps)  # [T, 512]
    k_pe = kv_lora[:, kv_rank:]  # [T, 64]

    w_qb = loader.linear_weight(f"{pfx}.q_b_proj")  # [64*256, 1024]
    q = (q_c @ w_qb.t()).view(T, nh, qk_dim)

    c_pos = cos_mla[positions]  # [T, 32]
    s_pos = sin_mla[positions]
    q_nope = q[..., :nope]
    q_rope = apply_rope_interleaved(q[..., nope:], c_pos.unsqueeze(1), s_pos.unsqueeze(1))
    k_pe = apply_rope_interleaved(k_pe, c_pos, s_pos)  # [T, 64]

    # kv_b_proj: [64 * (192 + 256), 512] -> absorb into q_nope (for K) and apply after attn (for V)
    w_kvb = loader.linear_weight(f"{pfx}.kv_b_proj").view(nh, nope + vdim, kv_rank)
    w_kb = w_kvb[:, :nope, :]  # [64, 192, 512]
    w_vb = w_kvb[:, nope:, :]  # [64, 256, 512]
    # q_abs: [T, 64, 512] = q_nope @ w_kb
    q_abs = torch.einsum("thn,hnl->thl", q_nope, w_kb)
    q_full = torch.cat([q_abs, q_rope], dim=-1)      # [T, 64, 576]
    kv_full = torch.cat([kv_c, k_pe], dim=-1)        # [T, 576]

    # Lightning Indexer (only needed if T > topk + win)
    need_indexer = T > (topk + win)
    if need_indexer:
        ipfx = f"{pfx}.core_attention.indexer"
        ih = cfg["index_n_heads"]      # 64
        ihd = cfg["index_head_dim"]    # 128
        w_iqb = loader.linear_weight(f"{ipfx}.wq_b")          # [64*128, 1024]
        w_ik = loader.linear_weight(f"{ipfx}.wk")             # [128, 4096]
        w_iw = loader.linear_weight(f"{ipfx}.weights_proj")   # [64, 4096]
        ik_nw = loader.raw(f"{ipfx}.k_norm.weight").float()
        ik_nb = loader.raw(f"{ipfx}.k_norm.bias").float()

        iq = (q_c @ w_iqb.t()).view(T, ih, ihd)
        ik = F.layer_norm(x @ w_ik.t(), (ihd,), ik_nw, ik_nb, eps=1e-6)
        iw = (x @ w_iw.t()) * (ihd ** -0.5) * (ih ** -0.5)  # [T, 64]

        # Partial NeoX RoPE on leading 64 dims
        iq_r = iq.clone()
        iq_r[..., :rdim] = apply_rope_neox(iq[..., :rdim], c_pos.unsqueeze(1), s_pos.unsqueeze(1))
        ik_r = ik.clone()
        ik_r[..., :rdim] = apply_rope_neox(ik[..., :rdim], c_pos, s_pos)
        iq_h = hadamard_128(iq_r)
        ik_h = hadamard_128(ik_r)

    sink = loader.raw(f"{pfx}.core_attention.softmax_offset").float()  # [64]
    scale = qk_dim ** -0.5
    o_lat = torch.empty((T, nh, kv_rank), dtype=torch.float32, device=x.device)

    for t in range(T):
        if t + 1 <= topk + win:
            k_sel = kv_full[:t + 1]
            v_sel = kv_c[:t + 1]
        else:
            # Indexer scores 0 .. t - win
            n_idx = t + 1 - win
            # iq_h[t]: [64, 128], ik_h[:n_idx]: [n_idx, 128]
            isc = torch.relu(iq_h[t] @ ik_h[:n_idx].t())  # [64, n_idx]
            total_sc = (iw[t].unsqueeze(1) * isc).sum(dim=0)  # [n_idx]
            top_ids = torch.topk(total_sc, topk, dim=0, sorted=False).indices
            swa_ids = torch.arange(n_idx, t + 1, device=x.device)
            all_ids = torch.cat([top_ids, swa_ids], dim=0)
            k_sel = kv_full[all_ids]
            v_sel = kv_c[all_ids]

        scores = (q_full[t] @ k_sel.t()) * scale  # [64, L]
        scores_s = torch.cat([scores, sink.unsqueeze(1)], dim=1)
        probs = torch.softmax(scores_s, dim=-1)[:, :-1]
        o_lat[t] = probs @ v_sel  # [64, 512]

    # Project latent output through w_vb: [T, 64, 256]
    o = torch.einsum("thl,hvl->thv", o_lat, w_vb)

    # VHA PostMix
    U = loader.raw(f"{pfx}.vha_postmix_U").float()
    V = loader.raw(f"{pfx}.vha_postmix_V").float()
    M_vha = torch.eye(nh, device=x.device) + V @ U.t()
    o = torch.einsum("hg,tgd->thd", M_vha, o).reshape(T, nh * vdim)

    w_gate = loader.linear_weight(f"{pfx}.gate_proj")  # [16384, 1024]
    gate = torch.sigmoid(q_c @ w_gate.t())
    w_ob = loader.linear_weight(f"{pfx}.o_proj")       # [4096, 16384]
    return (o * gate) @ w_ob.t()


# ---- FFN / MoE Layers ----

def run_dense_ffn(loader: CkptLoader, lid: int, x: torch.Tensor, cfg: dict) -> torch.Tensor:
    pfx = f"model.layers.{lid}.mlp"
    wg = loader.linear_weight(f"{pfx}.gate_proj")
    wu = loader.linear_weight(f"{pfx}.up_proj")
    wd = loader.linear_weight(f"{pfx}.down_proj")
    act = situ_and_mul(x @ wg.t(), x @ wu.t(), cfg["activation_situ_beta"], cfg["activation_situ_linear_beta"])
    return act @ wd.t()


def run_moe_layer(loader: CkptLoader, lid: int, x: torch.Tensor, cfg: dict,
                  expert_quant: str = "none", keep_w4_layers: set[int] | None = None) -> torch.Tensor:
    pfx = f"model.layers.{lid}.block_sparse_moe"
    T, H = x.shape
    E = cfg["n_routed_experts"]       # 512
    topk = cfg["num_experts_per_tok"]  # 10
    beta = cfg["activation_situ_beta"]
    lbeta = cfg["activation_situ_linear_beta"]
    eps = cfg["rms_norm_eps"]

    # Multi-router: sigmoid(l0) + sigmoid(l1)
    wg0 = loader.raw(f"{pfx}.gate.weight").float()    # [512, 4096]
    wg1 = loader.raw(f"{pfx}.gate.weight_1").float()  # [512, 4096]
    scores = torch.sigmoid(x @ wg0.t()) + torch.sigmoid(x @ wg1.t())  # [T, 512]
    bias = loader.raw(f"{pfx}.e_score_correction_bias").float()
    rscale = loader.raw(f"{pfx}.gate.routed_scaling_factor_param").float()

    ids = torch.topk(scores + bias.unsqueeze(0), topk, dim=-1, sorted=False).indices  # [T, 10]
    w = scores.gather(1, ids)
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    w = w * rscale[ids]

    # Latent down-projection
    w_fc1 = loader.linear_weight(f"{pfx}.fc1_latent_proj")  # [2048, 4096]
    lat = x @ w_fc1.t()  # [T, 2048]

    # Execute only active experts!
    qmode = "none" if (keep_w4_layers and lid in keep_w4_layers) else expert_quant
    lat_out = torch.zeros_like(lat)
    active_experts = torch.unique(ids).tolist()
    for eid in active_experts:
        tok_mask = (ids == eid)  # [T, 10]
        tok_idx, slot_idx = torch.where(tok_mask)
        if tok_idx.numel() == 0:
            continue
        epfx = f"{pfx}.experts.{eid}"
        w1 = apply_expert_quant(loader.expert_weight(f"{epfx}.w1"), qmode)
        w3 = apply_expert_quant(loader.expert_weight(f"{epfx}.w3"), qmode)
        w2 = apply_expert_quant(loader.expert_weight(f"{epfx}.w2"), qmode)
        inp = lat[tok_idx]
        act = situ_and_mul(inp @ w1.t(), inp @ w3.t(), beta, lbeta)
        out_e = act @ w2.t()
        lat_out.index_add_(0, tok_idx, out_e * w[tok_idx, slot_idx].unsqueeze(-1))

    # Latent norm + up-projection
    lat_out = rms_norm(lat_out, loader.raw(f"{pfx}.latent_norm.weight"), eps)
    w_fc2 = loader.linear_weight(f"{pfx}.fc2_latent_proj")  # [4096, 2048]
    routed_out = lat_out @ w_fc2.t()

    # Shared expert
    sw1 = loader.linear_weight(f"{pfx}.shared_experts.w1")
    sw3 = loader.linear_weight(f"{pfx}.shared_experts.w3")
    sw2 = loader.linear_weight(f"{pfx}.shared_experts.w2")
    shared_out = situ_and_mul(x @ sw1.t(), x @ sw3.t(), beta, lbeta) @ sw2.t()
    return routed_out + shared_out


# ---- Full Forward ----

def forward_all(loader: CkptLoader, token_ids: list[int], expert_quant: str = "none",
                keep_w4_layers: set[int] | None = None, verbose: bool = True) -> torch.Tensor:
    cfg = loader.cfg
    dev = loader.device
    T = len(token_ids)
    ids = torch.tensor(token_ids, dtype=torch.long, device=dev)
    positions = torch.arange(T, dtype=torch.long, device=dev)
    eps = cfg["rms_norm_eps"]
    hc_eps = cfg["hc_eps"]
    hc_mult = cfg["hc_mult"]
    sk_iters = cfg["hc_sinkhorn_iters"]

    # Build RoPE tables
    rs = cfg["rope_scaling"]
    cos_hca, sin_hca = build_yarn_cos_sin(
        T + 16, cfg["qk_rope_head_dim"], cfg["compress_rope_theta"],
        rs["factor"], rs["original_max_position_embeddings"],
        rs["beta_fast"], rs["beta_slow"], dev,
    )
    cos_mla, sin_mla = build_plain_cos_sin(T + 16, cfg["hybrid_mla_qk_rope_head_dim"], cfg["rope_theta"], dev)

    # Embedding -> broadcast to 4 mHC streams
    emb_w = loader.raw("model.embed_tokens.weight", to_dev=False)
    h0 = emb_w[ids.cpu()].to(dev).float()  # [T, 4096]
    del emb_w
    residual = h0.unsqueeze(1).expand(T, hc_mult, -1).contiguous()

    t_start = time.time()
    n_layers = cfg["num_hidden_layers"]
    for lid in range(n_layers):
        lpfx = f"model.layers.{lid}"
        # Attention mHC pre
        ahc_fn = loader.raw(f"{lpfx}.self_attention_hyper_connection.mapping_proj.weight")
        ahc_base = loader.raw(f"{lpfx}.self_attention_hyper_connection.bias")
        ahc_scale = torch.cat([
            loader.raw(f"{lpfx}.self_attention_hyper_connection.alpha_pre"),
            loader.raw(f"{lpfx}.self_attention_hyper_connection.alpha_post"),
            loader.raw(f"{lpfx}.self_attention_hyper_connection.alpha_res"),
        ])
        post_mix, comb, x = mhc_pre(residual, ahc_fn, ahc_scale, ahc_base, eps, hc_eps, sk_iters)
        x = rms_norm(x, loader.raw(f"{lpfx}.input_layernorm.weight"), eps)

        is_dsa = cfg["compress_ratios"][lid] <= 0
        if is_dsa:
            attn_out = run_dsa_layer(loader, lid, x, positions, cos_mla, sin_mla, cfg)
        else:
            attn_out = run_hca_layer(loader, lid, x, positions, cos_hca, sin_hca, cfg)

        residual = mhc_post(attn_out, residual, post_mix, comb)

        # FFN mHC pre
        mhc_fn = loader.raw(f"{lpfx}.mlp_hyper_connection.mapping_proj.weight")
        mhc_base = loader.raw(f"{lpfx}.mlp_hyper_connection.bias")
        mhc_scale = torch.cat([
            loader.raw(f"{lpfx}.mlp_hyper_connection.alpha_pre"),
            loader.raw(f"{lpfx}.mlp_hyper_connection.alpha_post"),
            loader.raw(f"{lpfx}.mlp_hyper_connection.alpha_res"),
        ])
        post_mix, comb, x = mhc_pre(residual, mhc_fn, mhc_scale, mhc_base, eps, hc_eps, sk_iters)
        x = rms_norm(x, loader.raw(f"{lpfx}.post_attention_layernorm.weight"), eps)

        if lid < cfg["first_k_dense_replace"]:
            ffn_out = run_dense_ffn(loader, lid, x, cfg)
        else:
            ffn_out = run_moe_layer(loader, lid, x, cfg, expert_quant, keep_w4_layers)

        residual = mhc_post(ffn_out, residual, post_mix, comb)
        if verbose and (lid in (0, 1, 7, 21, 42) or (lid + 1) % 10 == 0):
            print(f"  layer {lid:2d}/{n_layers-1} ({'DSA' if is_dsa else 'HCA'}) "
                  f"res_std={residual.std().item():.4f} elapsed={time.time()-t_start:.1f}s", flush=True)

    # Final mHC head contraction + norm + lm_head + multimax
    h_fn = loader.raw("model.mhc_contract.hc_head_fn")
    h_base = loader.raw("model.mhc_contract.hc_head_base")
    h_scale = loader.raw("model.mhc_contract.hc_head_scale")
    h = mhc_head_contract(residual, h_fn, h_scale, h_base, eps, hc_eps)
    h = rms_norm(h, loader.raw("model.norm.weight"), eps)

    lm_w = loader.linear_weight("lm_head")  # [201216, 4096]
    logits = h @ lm_w.t()  # [T, V]

    # MultiMax
    if loader.has("model.lm_head.multimax_ranges"):
        r = loader.raw("model.lm_head.multimax_ranges").float()
        t = loader.raw("model.lm_head.multimax_ts").float()
        logits = (
            logits
            + t[0] * F.relu(r[0] - logits)
            + t[1] * F.relu(logits - r[1])
            + t[2] * F.relu(r[2] - logits).square()
            + t[3] * F.relu(logits - r[3]).square()
        )
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/lmcache-nvme/ernie55/hf-mxfp4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--text", default=None)
    ap.add_argument("--ids_file", default=None, help="JSON or .pt list of token ids")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--expert_quant", default="none")
    ap.add_argument("--keep_w4_layers", default="")
    ap.add_argument("--save_ref", default=None)
    ap.add_argument("--compare_ref", default=None)
    args = ap.parse_args()

    # Load tokenizer from ckpt (uses tokenizers library directly via tokenizer.json)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(args.ckpt, "tokenizer.json"))

    if args.ids_file:
        if args.ids_file.endswith(".json"):
            token_ids = json.load(open(args.ids_file))[:args.max_tokens]
        else:
            token_ids = torch.load(args.ids_file)[:args.max_tokens].tolist()
    else:
        text = args.text or "The capital of France is"
        token_ids = [1] + tok.encode(text).ids  # BOS = 1
        token_ids = token_ids[:args.max_tokens]

    keep_w4 = set(int(x) for x in args.keep_w4_layers.split(",") if x.strip())
    print(f"Running ERNIE-5.5-Flash-Preview: T={len(token_ids)} quant={args.expert_quant} keep_w4={sorted(keep_w4)}", flush=True)
    loader = CkptLoader(args.ckpt, device=args.device)
    logits = forward_all(loader, token_ids, expert_quant=args.expert_quant, keep_w4_layers=keep_w4)

    # Next-token predictions at last position
    last_probs = torch.softmax(logits[-1], dim=-1)
    topv, topi = torch.topk(last_probs, 10)
    print("\nTop-10 next tokens after prompt:")
    for p, tid in zip(topv.tolist(), topi.tolist()):
        print(f"  {tid:7d} ({p*100:5.2f}%): {tok.decode([tid])!r}")

    # Compute NLL / PPL over sequence (positions 0..T-2 predicting 1..T-1)
    if len(token_ids) > 1:
        targets = torch.tensor(token_ids[1:], dtype=torch.long, device=logits.device)
        nll = F.cross_entropy(logits[:-1], targets).item()
        print(f"\nSequence NLL = {nll:.4f}, PPL = {math.exp(nll):.2f}")

    if args.save_ref:
        torch.save({"token_ids": token_ids, "logits": logits.cpu().half()}, args.save_ref)
        print(f"Saved reference logits to {args.save_ref}")

    if args.compare_ref:
        ref = torch.load(args.compare_ref, map_location=logits.device)
        ref_logits = ref["logits"].float()
        assert ref["token_ids"] == token_ids
        p_ref = torch.softmax(ref_logits, dim=-1)
        log_p_ref = torch.log_softmax(ref_logits, dim=-1)
        log_p_q = torch.log_softmax(logits, dim=-1)
        kl = (p_ref * (log_p_ref - log_p_q)).sum(dim=-1)
        top1 = (ref_logits.argmax(-1) == logits.argmax(-1)).float().mean().item()
        print(f"\n=== vs Reference ({args.compare_ref}) ===")
        print(f"  Mean KL   : {kl.mean().item():.4f} (median {kl.median().item():.4f}, max {kl.max().item():.4f})")
        print(f"  Top-1 agr : {top1*100:.2f}%")
        if len(token_ids) > 1:
            targets = torch.tensor(token_ids[1:], dtype=torch.long, device=logits.device)
            ref_nll = F.cross_entropy(ref_logits[:-1], targets).item()
            q_nll = F.cross_entropy(logits[:-1], targets).item()
            print(f"  Ref NLL/PPL: {ref_nll:.4f} / {math.exp(ref_nll):.2f}")
            print(f"  Q   NLL/PPL: {q_nll:.4f} / {math.exp(q_nll):.2f} (dNLL = {q_nll - ref_nll:+.4f})")


if __name__ == "__main__":
    main()
