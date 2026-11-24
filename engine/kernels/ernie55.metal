// ernie55.metal: ERNIE-5.5-Flash-Preview Metal 4 ops (everything outside mm2/mmp/mmpg).
// All arithmetic in f32; bf16 only for matmul fragment operands and compact KV caches.
#include <metal_stdlib>
using namespace metal;

static inline uint frag_off(uint m, uint k, uint K) {
  const uint kk = k & 31, mm = m & 15;
  const uint l = ((kk >> 2) & 1) | ((mm & 3) << 1) | (((kk >> 3) & 1) << 3) | (((mm >> 2) & 1) << 4);
  const uint i = (kk & 3) | ((mm >> 3) << 2) | ((kk >> 4) << 3);
  return ((m >> 4) * (K >> 5) + (k >> 5)) * 512 + l * 16 + i;
}

static inline float sigm(float x) { return 1.0f / (1.0f + exp(-x)); }

static inline float situ_act(float g, float u, float beta, float lbeta) {
  float ga = beta * tanh(g / beta) * sigm(g);
  float ua = (lbeta > 0.0f) ? (lbeta * tanh(u / lbeta)) : u;
  return ga * ua;
}

static inline float tg_sum(float v, threadgroup float* sh, ushort tid, ushort ntg) {
  v = simd_sum(v);
  const ushort sg = tid / 32, ns = (ntg + 31) / 32;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((tid & 31) == 0) sh[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float s = 0.0f;
  for (ushort i = 0; i < ns; i++) s += sh[i];
  return s;
}

// ------------------------------------------------------------------ 1. Embedding
struct EmbArgs { uint M, H, HC; };
kernel void e5_embed(device const uint* ids [[buffer(0)]], device const bfloat* E [[buffer(1)]],
                     device float* h [[buffer(2)]], constant EmbArgs& a [[buffer(3)]],
                     uint2 g [[thread_position_in_grid]]) {
  const uint c = g.x, m = g.y;
  if (c >= a.H || m >= a.M) return;
  const float v = float(E[size_t(ids[m]) * a.H + c]);
  for (uint s = 0; s < a.HC; s++) h[(size_t(m) * a.HC + s) * a.H + c] = v;
}

// ------------------------------------------------------------------ 2. Sinkhorn mHC (post + pre fused)
// mode 0: sublayer mHC (24 mixes: 4 pre + 4 post + 16 comb, Sinkhorn 20 iters, RMS eps outside sqrt)
// mode 1: final head contraction (4 mixes, rsqrt(mean + eps), then model.norm)
struct MhcArgs {
  uint M, H, inject, mode, sk_iters;
  float eps, hc_eps;
};
kernel void e5_mhc_1pass(device float* h [[buffer(0)]],
                         device const float* y [[buffer(1)]],
                         device const float* postp [[buffer(2)]],
                         device const float* combp [[buffer(3)]],
                         device const float* fn [[buffer(4)]],
                         device const float* base [[buffer(5)]],
                         device const float* scale [[buffer(6)]],
                         device const bfloat* nw [[buffer(7)]],
                         device bfloat* xfrag [[buffer(8)]],
                         device float* xf [[buffer(9)]],
                         device float* post_out [[buffer(10)]],
                         device float* comb_out [[buffer(11)]],
                         constant MhcArgs& a [[buffer(12)]],
                         uint m [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]],
                         ushort ntg [[threads_per_threadgroup]]) {
  threadgroup float shp[34][32];  // up to 24 mixes + 10 Gram terms
  threadgroup float tot[34];
  threadgroup float pre_sh[4];
  const uint H = a.H, W = 4 * H;
  device float* hr = h + size_t(m) * W;
  const uint nmix = (a.mode == 0) ? 24u : 4u;

  float post_in[4];
  float comb_in[4][4];
  if (a.inject) {
    for (uint s = 0; s < 4; s++) post_in[s] = postp[m * 4 + s];
    for (uint i = 0; i < 4; i++)
      for (uint j = 0; j < 4; j++) comb_in[i][j] = combp[m * 16 + i * 4 + j];
  }

  float vreg[4][4];  // H=4096, ntg=1024 -> 4 channels per thread
  float acc[34];
  for (uint i = 0; i < 34; i++) acc[i] = 0.0f;

  uint kc = 0;
  for (uint c = tid; c < H; c += ntg, kc++) {
    float v[4];
    if (a.inject) {
      float oldv[4];
      for (uint s = 0; s < 4; s++) oldv[s] = hr[s * H + c];
      const float yc = y[size_t(m) * H + c];
      for (uint j = 0; j < 4; j++) {
        float accj = post_in[j] * yc;
        for (uint i = 0; i < 4; i++) accj = fma(comb_in[i][j], oldv[i], accj);
        v[j] = accj;
        hr[j * H + c] = accj;
      }
    } else {
      for (uint s = 0; s < 4; s++) v[s] = hr[s * H + c];
    }
    for (uint s = 0; s < 4; s++) vreg[kc][s] = v[s];
    for (uint j = 0; j < nmix; j++)
      for (uint s = 0; s < 4; s++) acc[j] = fma(fn[size_t(j) * W + s * H + c], v[s], acc[j]);
    uint q = 24;
    for (uint s = 0; s < 4; s++)
      for (uint u = s; u < 4; u++) { acc[q] = fma(v[s], v[u], acc[q]); q++; }
  }

  const ushort sgi = tid / 32, ns = (ntg + 31) / 32;
  for (uint i = 0; i < 34; i++) {
    if (i >= nmix && i < 24) continue;
    const float t = simd_sum(acc[i]);
    if ((tid & 31) == 0) shp[i][sgi] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 34 && (tid < nmix || tid >= 24)) {
    float t = 0.0f;
    for (ushort k = 0; k < ns; k++) t += shp[tid][k];
    tot[tid] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    float G[4][4];
    uint q = 24;
    for (uint s = 0; s < 4; s++)
      for (uint u = s; u < 4; u++) { G[s][u] = tot[q]; G[u][s] = tot[q]; q++; }
    const float tr = G[0][0] + G[1][1] + G[2][2] + G[3][3];
    const float mean_sq = tr / float(W);
    if (a.mode == 0) {
      // ERNIE 5.5 mHC pre: 1 / (sqrt(mean) + eps)
      const float r = 1.0f / (sqrt(mean_sq) + a.eps);
      for (uint s = 0; s < 4; s++) {
        pre_sh[s] = sigm(tot[s] * r * scale[0] + base[s]) + a.hc_eps;
        post_out[m * 4 + s] = 2.0f * sigm(tot[4 + s] * r * scale[1] + base[4 + s]);
      }
      // Sinkhorn-Knopp on 4x4 comb
      float C[4][4];
      for (uint i = 0; i < 4; i++) {
        float mx = -1e30f;
        for (uint j = 0; j < 4; j++) {
          C[i][j] = tot[8 + i * 4 + j] * r * scale[2] + base[8 + i * 4 + j];
          mx = max(mx, C[i][j]);
        }
        float sm = 0.0f;
        for (uint j = 0; j < 4; j++) { C[i][j] = exp(C[i][j] - mx); sm += C[i][j]; }
        for (uint j = 0; j < 4; j++) C[i][j] = C[i][j] / sm + a.hc_eps;
      }
      // Initial column norm
      for (uint j = 0; j < 4; j++) {
        float cs = C[0][j] + C[1][j] + C[2][j] + C[3][j] + a.hc_eps;
        for (uint i = 0; i < 4; i++) C[i][j] /= cs;
      }
      for (uint it = 1; it < a.sk_iters; it++) {
        for (uint i = 0; i < 4; i++) {
          float rs = C[i][0] + C[i][1] + C[i][2] + C[i][3] + a.hc_eps;
          for (uint j = 0; j < 4; j++) C[i][j] /= rs;
        }
        for (uint j = 0; j < 4; j++) {
          float cs = C[0][j] + C[1][j] + C[2][j] + C[3][j] + a.hc_eps;
          for (uint i = 0; i < 4; i++) C[i][j] /= cs;
        }
      }
      for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++) comb_out[m * 16 + i * 4 + j] = C[i][j];
    } else {
      // Final head contraction: rsqrt(mean + eps)
      const float r = rsqrt(mean_sq + a.eps);
      for (uint s = 0; s < 4; s++)
        pre_sh[s] = sigm(tot[s] * r * scale[0] + base[s]) + a.hc_eps;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float pre[4];
  for (uint s = 0; s < 4; s++) pre[s] = pre_sh[s];

  // Compute x = sum_s pre[s] * h[s] and its RMSNorm
  float x2 = 0.0f;
  float xreg[4];
  kc = 0;
  for (uint c = tid; c < H; c += ntg, kc++) {
    float x = 0.0f;
    for (uint s = 0; s < 4; s++) x = fma(pre[s], vreg[kc][s], x);
    xreg[kc] = x;
    x2 = fma(x, x, x2);
  }
  const float xr = rsqrt(tg_sum(x2, shp[0], tid, ntg) / float(H) + a.eps);
  kc = 0;
  for (uint c = tid; c < H; c += ntg, kc++) {
    const float v = xreg[kc] * xr * float(nw[c]);
    if (xf) xf[size_t(m) * H + c] = v;
    xfrag[frag_off(m, c, H)] = bfloat(v);
  }
}

// ------------------------------------------------------------------ 3. HCA Prep
// Splits attn.in [M, 2560] = qd(1024) | kv(512) | ckv(512) | cgate(512).
// Writes qr_frag (1024 bf16 frag), kv_swa ring [RING_HCA, 512] bf16, and comp partial [128, 512] f32.
struct HcaPrepArgs { uint M, p0, ring, cr, nope, rdim; float eps; };
kernel void e5_hca_prep(device const float* in_buf [[buffer(0)]],
                        device const bfloat* q_nw [[buffer(1)]],
                        device const bfloat* kv_nw [[buffer(2)]],
                        device const float* ape [[buffer(3)]],
                        device const float* cos_t [[buffer(4)]],
                        device const float* sin_t [[buffer(5)]],
                        device bfloat* qr_frag [[buffer(6)]],
                        device bfloat* swa_kv [[buffer(7)]],
                        device float* comp_kv [[buffer(8)]],
                        device float* comp_sc [[buffer(9)]],
                        constant HcaPrepArgs& a [[buffer(10)]],
                        uint m [[threadgroup_position_in_grid]],
                        ushort tid [[thread_index_in_threadgroup]],
                        ushort ntg [[threads_per_threadgroup]]) {
  threadgroup float sh[32];
  const uint pos = a.p0 + m;
  device const float* row = in_buf + size_t(m) * 2560;

  // 1) qr: RMSNorm(row[0..1023], q_nw) -> qr_frag
  float ss = 0.0f;
  for (uint c = tid; c < 1024; c += ntg) ss = fma(row[c], row[c], ss);
  const float rq = rsqrt(tg_sum(ss, sh, tid, ntg) / 1024.0f + a.eps);
  for (uint c = tid; c < 1024; c += ntg)
    qr_frag[frag_off(m, c, 1024)] = bfloat(row[c] * rq * float(q_nw[c]));

  // 2) kv: RMSNorm(row[1024..1535], kv_nw) -> HCA RoPE on [448..511] -> swa_kv[pos % ring]
  ss = 0.0f;
  for (uint c = tid; c < 512; c += ntg) {
    float v = row[1024 + c];
    ss = fma(v, v, ss);
  }
  const float rk = rsqrt(tg_sum(ss, sh, tid, ntg) / 512.0f + a.eps);
  device bfloat* dst_swa = swa_kv + size_t(pos % a.ring) * 512;
  for (uint c = tid; c < a.nope; c += ntg)
    dst_swa[c] = bfloat(row[1024 + c] * rk * float(kv_nw[c]));
  if (tid < a.rdim / 2) {
    const uint j = tid;
    const uint c0 = a.nope + 2 * j;
    const float e = row[1024 + c0] * rk * float(kv_nw[c0]);
    const float o = row[1024 + c0 + 1] * rk * float(kv_nw[c0 + 1]);
    const float cs = cos_t[size_t(pos) * (a.rdim / 2) + j];
    const float sn = sin_t[size_t(pos) * (a.rdim / 2) + j];
    dst_swa[c0] = bfloat(e * cs - o * sn);
    dst_swa[c0 + 1] = bfloat(o * cs + e * sn);
  }

  // 3) Compressor partial state at slot (pos % 128)
  // Note: for multi-token chunks crossing a 128 boundary, host dispatches chunks that don't wrap comp buffer,
  // or uses comp_kv[pos % comp_ring] where comp_ring >= MP + 128. Here comp_kv is indexed by (pos % a.ring).
  const uint cslot = pos % a.ring;
  const uint ape_r = pos % a.cr;
  for (uint c = tid; c < 512; c += ntg) {
    comp_kv[size_t(cslot) * 512 + c] = row[1536 + c];
    comp_sc[size_t(cslot) * 512 + c] = row[2048 + c] + ape[size_t(ape_r) * 512 + c];
  }
}

// ------------------------------------------------------------------ 4. HCA C128 Compress
// One threadgroup per completed 128-token block b (covering pos = b*128 .. b*128+127).
struct HcaCompArgs { uint b0, ring, cr, nope, rdim; float eps; };
kernel void e5_hca_compress(device const float* comp_kv [[buffer(0)]],
                            device const float* comp_sc [[buffer(1)]],
                            device const bfloat* cnw [[buffer(2)]],
                            device const float* cos_t [[buffer(3)]],
                            device const float* sin_t [[buffer(4)]],
                            device bfloat* comp_cache [[buffer(5)]],
                            constant HcaCompArgs& a [[buffer(6)]],
                            uint bi [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]],
                            ushort ntg [[threads_per_threadgroup]]) {
  threadgroup float sh[32];
  threadgroup float cvec[512];
  const uint b = a.b0 + bi;
  const uint p_start = b * a.cr;
  // Each thread handles 1 dim (ntg = 512)
  float val = 0.0f;
  if (tid < 512) {
    float mx = -1e30f;
    for (uint k = 0; k < a.cr; k++) {
      uint sl = (p_start + k) % a.ring;
      mx = max(mx, comp_sc[size_t(sl) * 512 + tid]);
    }
    float sm = 0.0f, acc = 0.0f;
    for (uint k = 0; k < a.cr; k++) {
      uint sl = (p_start + k) % a.ring;
      float w = exp(comp_sc[size_t(sl) * 512 + tid] - mx);
      sm += w;
      acc = fma(w, comp_kv[size_t(sl) * 512 + tid], acc);
    }
    val = acc / sm;
    cvec[tid] = val;
  }
  const float r = rsqrt(tg_sum(val * val, sh, tid, ntg) / 512.0f + a.eps);
  if (tid < 512) cvec[tid] = val * r * float(cnw[tid]);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device bfloat* dst = comp_cache + size_t(b) * 512;
  if (tid < a.nope) {
    dst[tid] = bfloat(cvec[tid]);
  } else if (tid >= a.nope && tid < a.nope + a.rdim / 2) {
    const uint j = tid - a.nope;
    const uint c0 = a.nope + 2 * j;
    const float e = cvec[c0], o = cvec[c0 + 1];
    const float cs = cos_t[size_t(p_start) * (a.rdim / 2) + j];
    const float sn = sin_t[size_t(p_start) * (a.rdim / 2) + j];
    dst[c0] = bfloat(e * cs - o * sn);
    dst[c0 + 1] = bfloat(o * cs + e * sn);
  }
}

// ------------------------------------------------------------------ 5. HCA Q-norm + RoPE
// q_in is inside attn.qb output [M, 40960] at offset 0 (stride 40960), 64 heads x 512.
struct HcaQArgs { uint M, p0, nh, hd, nope, rdim, stride; float eps; };
kernel void e5_hca_qnorm_rope(device const float* qb_out [[buffer(0)]],
                              device const float* cos_t [[buffer(1)]],
                              device const float* sin_t [[buffer(2)]],
                              device float* q_out [[buffer(3)]],
                              constant HcaQArgs& a [[buffer(4)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              ushort tid [[thread_index_in_threadgroup]],
                              uint2 ntg2 [[threads_per_threadgroup]]) {
  const ushort ntg = ushort(ntg2.x);
  threadgroup float sh[32];
  const uint h = tg.x, m = tg.y, pos = a.p0 + m;
  device const float* src = qb_out + size_t(m) * a.stride + h * a.hd;
  device float* dst = q_out + (size_t(m) * a.nh + h) * a.hd;
  float ss = 0.0f;
  for (uint c = tid; c < a.hd; c += ntg) {
    float v = src[c];
    ss = fma(v, v, ss);
  }
  const float r = rsqrt(tg_sum(ss, sh, tid, ntg) / float(a.hd) + a.eps);
  for (uint c = tid; c < a.nope; c += ntg) dst[c] = src[c] * r;
  if (tid < a.rdim / 2) {
    const uint j = tid, c0 = a.nope + 2 * j;
    const float e = src[c0] * r, o = src[c0 + 1] * r;
    const float cs = cos_t[size_t(pos) * (a.rdim / 2) + j];
    const float sn = sin_t[size_t(pos) * (a.rdim / 2) + j];
    dst[c0] = e * cs - o * sn;
    dst[c0 + 1] = o * cs + e * sn;
  }
}

// ------------------------------------------------------------------ 6. HCA Attention (MQA over comp + SWA + sink)
// Grid: (nh=64, M), 128 threads (4 simdgroups) per (h, m). Each thread holds 4 dims of hd=512!
struct HcaAttnArgs { uint M, p0, nh, hd, cr, win, ring; float scale; };
kernel void e5_hca_attn(device const float* q [[buffer(0)]],
                        device const bfloat* comp_cache [[buffer(1)]],
                        device const bfloat* swa_kv [[buffer(2)]],
                        device const float* sink [[buffer(3)]],
                        device float* o_out [[buffer(4)]],
                        constant HcaAttnArgs& a [[buffer(5)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        ushort tid [[thread_index_in_threadgroup]],
                        uint2 ntg2 [[threads_per_threadgroup]]) {
  const ushort ntg = ushort(ntg2.x);
  threadgroup float sh[32];
  const uint h = tg.x, m = tg.y, pos = a.p0 + m;
  const uint nc = (pos + 1) / a.cr;
  const uint swa_s = (pos + 1 > a.win) ? (pos + 1 - a.win) : 0u;
  const uint n_swa = pos + 1 - swa_s;
  const uint n_tot = nc + n_swa;

  device const float* qr = q + (size_t(m) * a.nh + h) * 512;
  float q0 = qr[tid], q1 = qr[tid + 128], q2 = qr[tid + 256], q3 = qr[tid + 384];
  float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
  // Initialize running softmax with sink logit (value = 0)
  float mx = sink[h];
  float lse = 1.0f;

  for (uint j = 0; j < n_tot; j++) {
    device const bfloat* kr = (j < nc)
      ? (comp_cache + size_t(j) * 512)
      : (swa_kv + size_t((swa_s + (j - nc)) % a.ring) * 512);
    float k0 = float(kr[tid]), k1 = float(kr[tid + 128]), k2 = float(kr[tid + 256]), k3 = float(kr[tid + 384]);
    float dot = fma(q0, k0, fma(q1, k1, fma(q2, k2, q3 * k3)));
    float sc = tg_sum(dot, sh, tid, ntg) * a.scale;
    float nmx = max(mx, sc);
    float alpha = exp(mx - nmx);
    float p = exp(sc - nmx);
    lse = lse * alpha + p;
    o0 = fma(p, k0, o0 * alpha);
    o1 = fma(p, k1, o1 * alpha);
    o2 = fma(p, k2, o2 * alpha);
    o3 = fma(p, k3, o3 * alpha);
    mx = nmx;
  }
  const float inv = 1.0f / lse;
  device float* dst = o_out + (size_t(m) * a.nh + h) * 512;
  dst[tid] = o0 * inv;
  dst[tid + 128] = o1 * inv;
  dst[tid + 256] = o2 * inv;
  dst[tid + 384] = o3 * inv;
}

// ------------------------------------------------------------------ 7. HCA VHA + Inverse RoPE -> grouped wo_a frags
// Grid: (nh=64, M), 256 threads. Packs [M, 8 groups, 4096] into grouped frags (stride_e = ceil(M/16)*128*512 bf16).
struct HcaVhaArgs { uint M, p0, nh, hd, nope, rdim, n_groups, g_stride_bf16; };
kernel void e5_hca_vha_invrope(device const float* o_in [[buffer(0)]],
                               device const float* vha_m [[buffer(1)]],
                               device const float* cos_t [[buffer(2)]],
                               device const float* sin_t [[buffer(3)]],
                               device bfloat* out_frags [[buffer(4)]],
                               constant HcaVhaArgs& a [[buffer(5)]],
                               uint2 tg [[threadgroup_position_in_grid]],
                               ushort tid [[thread_index_in_threadgroup]],
                               uint2 ntg2 [[threads_per_threadgroup]]) {
  const ushort ntg = ushort(ntg2.x);
  threadgroup float ovec[512];
  const uint h = tg.x, m = tg.y, pos = a.p0 + m;
  device const float* mrow = vha_m + h * a.nh;
  device const float* obase = o_in + size_t(m) * a.nh * a.hd;

  for (uint c = tid; c < a.hd; c += ntg) {
    float acc = 0.0f;
    for (uint g = 0; g < a.nh; g++) acc = fma(mrow[g], obase[g * a.hd + c], acc);
    ovec[c] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint hpg = a.nh / a.n_groups;  // 8
  const uint grp = h / hpg;
  const uint hig = h % hpg;
  device bfloat* gdst = out_frags + size_t(grp) * a.g_stride_bf16;
  const uint K = hpg * a.hd;  // 4096

  for (uint c = tid; c < a.nope; c += ntg)
    gdst[frag_off(m, hig * a.hd + c, K)] = bfloat(ovec[c]);
  if (tid < a.rdim / 2) {
    const uint j = tid, c0 = a.nope + 2 * j;
    const float e = ovec[c0], o = ovec[c0 + 1];
    const float cs = cos_t[size_t(pos) * (a.rdim / 2) + j];
    const float sn = sin_t[size_t(pos) * (a.rdim / 2) + j];
    // Inverse RoPE: (e*c + o*s, o*c - e*s)
    gdst[frag_off(m, hig * a.hd + c0, K)] = bfloat(e * cs + o * sn);
    gdst[frag_off(m, hig * a.hd + c0 + 1, K)] = bfloat(o * cs - e * sn);
  }
}

// ------------------------------------------------------------------ 8. Gate mul + frag pack (for HCA wo_b)
struct GatePackArgs { uint M, K, g_stride, g_off, z_gstride; };
kernel void e5_hca_gate_pack(device const float* z_grp [[buffer(0)]],
                             device const float* qb_out [[buffer(1)]],
                             device bfloat* dst_frag [[buffer(2)]],
                             constant GatePackArgs& a [[buffer(3)]],
                             uint2 g [[thread_position_in_grid]]) {
  const uint c = g.x, m = g.y;
  if (c >= a.K || m >= a.M) return;
  const uint grp = c / 1024, col = c % 1024;
  const float zv = z_grp[size_t(grp) * a.z_gstride + size_t(m) * 1024 + col];
  const float gv = qb_out[size_t(m) * a.g_stride + a.g_off + c];
  dst_frag[frag_off(m, c, a.K)] = bfloat(zv * sigm(gv));
}

// ------------------------------------------------------------------ 9. DSA Prep (qa, kva, indexer K + W)
static inline void hadamard_128_tg(threadgroup float* v, ushort tid) {
  for (uint step = 1; step < 128; step <<= 1) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64) {
      const uint blk = tid / step;
      const uint off = tid % step;
      const uint i0 = blk * 2 * step + off;
      const uint i1 = i0 + step;
      const float a = v[i0], b = v[i1];
      v[i0] = a + b;
      v[i1] = a - b;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 128) v[tid] *= 0.08838834764831845f;  // 1/sqrt(128)
}

struct DsaPrepArgs { uint M, p0, cap; float eps; };
kernel void e5_dsa_prep(device const float* in_buf [[buffer(0)]],
                        device const bfloat* qa_nw [[buffer(1)]],
                        device const bfloat* kva_nw [[buffer(2)]],
                        device const float* ik_nw [[buffer(3)]],
                        device const float* ik_nb [[buffer(4)]],
                        device const float* cos_t [[buffer(5)]],
                        device const float* sin_t [[buffer(6)]],
                        device bfloat* qc_frag [[buffer(7)]],
                        device uchar* lat_q8 [[buffer(8)]],   // [cap, 512] int8 + [cap, 8] f16 + [cap, 64] bf16 = 656 B/tok (aligned 672 B)
                        device uchar* idx_k8 [[buffer(9)]],   // [cap, 128] int8 + [cap, 1] f16 = 130 B (aligned 136 B)
                        device float* idx_w [[buffer(10)]],   // [M, 64]
                        constant DsaPrepArgs& a [[buffer(11)]],
                        uint m [[threadgroup_position_in_grid]],
                        ushort tid [[thread_index_in_threadgroup]],
                        ushort ntg [[threads_per_threadgroup]]) {
  threadgroup float sh[32];
  threadgroup float buf128[128];
  const uint pos = a.p0 + m;
  device const float* row = in_buf + size_t(m) * 1792;

  // 1) q_c: RMSNorm(row[0..1023], qa_nw) -> qc_frag
  float ss = 0.0f;
  for (uint c = tid; c < 1024; c += ntg) ss = fma(row[c], row[c], ss);
  const float rq = rsqrt(tg_sum(ss, sh, tid, ntg) / 1024.0f + a.eps);
  for (uint c = tid; c < 1024; c += ntg)
    qc_frag[frag_off(m, c, 1024)] = bfloat(row[c] * rq * float(qa_nw[c]));

  // 2) kv_c: RMSNorm(row[1024..1535], kva_nw) -> kv8 (8 groups of 64 int8 + 8 f16 scales + 64 bf16 rope = 672 B/tok)
  ss = 0.0f;
  for (uint c = tid; c < 512; c += ntg) {
    float v = row[1024 + c];
    ss = fma(v, v, ss);
  }
  const float rkv = rsqrt(tg_sum(ss, sh, tid, ntg) / 512.0f + a.eps);
  device uchar* lrow = lat_q8 + size_t(pos) * 672;
  device char* lq = (device char*)lrow;
  device half* ls = (device half*)(lrow + 512);
  device bfloat* lrope = (device bfloat*)(lrow + 528);
  // 8 groups of 64: use first 8 simdgroups (256 threads) or simple loop over 8 groups
  if (tid < 512) {
    float v = row[1024 + tid] * rkv * float(kva_nw[tid]);
    uint grp = tid >> 6;  // 0..7
    // 64 elements = 2 simdgroups; reduce via sh
    float am = simd_max(abs(v));
    if ((tid & 31) == 0) sh[tid >> 5] = am;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gmax = max(sh[grp * 2], sh[grp * 2 + 1]);
    float sc = max(gmax / 127.0f, 1e-7f);
    if ((tid & 63) == 0) ls[grp] = half(sc);
    lq[tid] = char(clamp(rint(v / float(half(sc))), -127.0f, 127.0f));
  }
  // k_pe: plain interleaved RoPE on row[1536..1599]
  if (tid < 32) {
    const uint c0 = 1536 + 2 * tid;
    const float e = row[c0], o = row[c0 + 1];
    const float cs = cos_t[size_t(pos) * 32 + tid];
    const float sn = sin_t[size_t(pos) * 32 + tid];
    lrope[2 * tid] = bfloat(e * cs - o * sn);
    lrope[2 * tid + 1] = bfloat(o * cs + e * sn);
  }

  // 3) Indexer K: LayerNorm(row[1600..1727]) -> partial NeoX RoPE on 0..63 -> Hadamard 128 -> int8 + f16 scale
  float kv = (tid < 128) ? row[1600 + tid] : 0.0f;
  const float mean = tg_sum(kv, sh, tid, ntg) / 128.0f;
  float diff = (tid < 128) ? (kv - mean) : 0.0f;
  const float rln = rsqrt(tg_sum(diff * diff, sh, tid, ntg) / 128.0f + 1e-6f);
  if (tid < 128) buf128[tid] = diff * rln * ik_nw[tid] + ik_nb[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 32) {
    const float x1 = buf128[tid], x2 = buf128[tid + 32];
    const float cs = cos_t[size_t(pos) * 32 + tid];
    const float sn = sin_t[size_t(pos) * 32 + tid];
    buf128[tid] = x1 * cs - x2 * sn;
    buf128[tid + 32] = x2 * cs + x1 * sn;
  }
  hadamard_128_tg(buf128, tid);
  float hv = (tid < 128) ? buf128[tid] : 0.0f;
  float am = simd_max(abs(hv));
  if ((tid & 31) == 0) sh[tid >> 5] = am;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 128) {
    float gmax = max(max(sh[0], sh[1]), max(sh[2], sh[3]));
    float sc = max(gmax / 127.0f, 1e-7f);
    device uchar* krow = idx_k8 + size_t(pos) * 136;
    if (tid == 0) *(device half*)(krow + 128) = half(sc);
    ((device char*)krow)[tid] = char(clamp(rint(hv / float(half(sc))), -127.0f, 127.0f));
  }

  // 4) Indexer weights: row[1728..1791] * (128^-0.5 * 64^-0.5)
  if (tid < 64) idx_w[size_t(m) * 64 + tid] = row[1728 + tid] * 0.011048543456039806f;
}

// ------------------------------------------------------------------ 10. DSA Q Prep
// Takes qb_out [M, 40960] = q(16384: 64x256) | gate(16384) | iq(8192: 64x128).
// Packs q_nope [64, M, 192] into grouped frags for attn.kb, writes q_rope [M, 64, 64] f32,
// and computes partial NeoX RoPE + Hadamard 128 on iq -> iq_h [M, 64, 128] f32.
struct DsaQPrepArgs { uint M, p0, kb_gstride_bf16; };
kernel void e5_dsa_q_prep(device const float* qb_out [[buffer(0)]],
                          device const float* cos_t [[buffer(1)]],
                          device const float* sin_t [[buffer(2)]],
                          device bfloat* qnope_frags [[buffer(3)]],
                          device float* qrope_out [[buffer(4)]],
                          device float* iq_out [[buffer(5)]],
                          constant DsaQPrepArgs& a [[buffer(6)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup float buf128[128];
  const uint h = tg.x, m = tg.y, pos = a.p0 + m;
  device const float* qsrc = qb_out + size_t(m) * 40960 + h * 256;
  device bfloat* gdst = qnope_frags + size_t(h) * a.kb_gstride_bf16;

  // Pack q_nope [192]
  if (tid < 192) gdst[frag_off(m, tid, 192)] = bfloat(qsrc[tid]);
  // Plain interleaved RoPE on q_rope [64]
  if (tid < 32) {
    const float e = qsrc[192 + 2 * tid], o = qsrc[192 + 2 * tid + 1];
    const float cs = cos_t[size_t(pos) * 32 + tid];
    const float sn = sin_t[size_t(pos) * 32 + tid];
    device float* rdst = qrope_out + (size_t(m) * 64 + h) * 64;
    rdst[2 * tid] = e * cs - o * sn;
    rdst[2 * tid + 1] = o * cs + e * sn;
  }
  // Indexer Q: partial NeoX RoPE on leading 64 dims + Hadamard 128
  device const float* isrc = qb_out + size_t(m) * 40960 + 32768 + h * 128;
  if (tid < 128) buf128[tid] = isrc[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 32) {
    const float x1 = buf128[tid], x2 = buf128[tid + 32];
    const float cs = cos_t[size_t(pos) * 32 + tid];
    const float sn = sin_t[size_t(pos) * 32 + tid];
    buf128[tid] = x1 * cs - x2 * sn;
    buf128[tid + 32] = x2 * cs + x1 * sn;
  }
  hadamard_128_tg(buf128, tid);
  if (tid < 128) iq_out[(size_t(m) * 64 + h) * 128 + tid] = buf128[tid];
}

// ------------------------------------------------------------------ 11. DSA Indexer Score & Top-K
// Scores all positions j < n_idx for each row m. Grid: (ceil(max_n_idx / 64), M), 64 threads.
struct DsaIdxArgs { uint M, p0, win, topk, cap; };
kernel void e5_dsa_idx_score(device const float* iq_h [[buffer(0)]],
                             device const float* iw [[buffer(1)]],
                             device const uchar* idx_k8 [[buffer(2)]],
                             device float* scores [[buffer(3)]],  // [M, cap]
                             constant DsaIdxArgs& a [[buffer(4)]],
                             uint2 g [[thread_position_in_grid]]) {
  const uint j = g.x, m = g.y;
  const uint pos = a.p0 + m;
  const uint n_idx = (pos + 1 > a.win) ? (pos + 1 - a.win) : 0u;
  if (m >= a.M || j >= n_idx) return;

  device const uchar* krow = idx_k8 + size_t(j) * 136;
  device const char* kq = (device const char*)krow;
  const float ksc = float(*(device const half*)(krow + 128));
  float kf[128];
  for (uint d = 0; d < 128; d++) kf[d] = float(kq[d]) * ksc;

  float tot = 0.0f;
  device const float* qm = iq_h + size_t(m) * 64 * 128;
  device const float* wm = iw + size_t(m) * 64;
  for (uint h = 0; h < 64; h++) {
    device const float* qh = qm + h * 128;
    float dot = 0.0f;
    for (uint d = 0; d < 128; d++) dot = fma(qh[d], kf[d], dot);
    if (dot > 0.0f) tot = fma(wm[h], dot, tot);
  }
  scores[size_t(m) * a.cap + j] = tot;
}

// Simple exact top-2048 selection via 16-bin radix/threshold or partial sort per row (1 TG of 256 threads per row).
// For n_idx <= topk (<= 2048), emits 0..n_idx-1 directly.
kernel void e5_dsa_idx_topk(device const float* scores [[buffer(0)]],
                            device int* sel_out [[buffer(1)]],    // [M, 2176]
                            device uint* sel_len [[buffer(2)]],   // [M]
                            constant DsaIdxArgs& a [[buffer(3)]],
                            uint m [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]],
                            ushort ntg [[threads_per_threadgroup]]) {
  threadgroup uint hist[256];
  threadgroup uint count_hi;
  const uint pos = a.p0 + m;
  const uint n_idx = (pos + 1 > a.win) ? (pos + 1 - a.win) : 0u;
  const uint swa_len = pos + 1 - n_idx;
  device int* dst = sel_out + size_t(m) * (a.topk + a.win);

  if (n_idx <= a.topk) {
    for (uint i = tid; i < pos + 1; i += ntg) dst[i] = int(i);
    if (tid == 0) sel_len[m] = pos + 1;
    return;
  }

  // Binary search float threshold (via order-preserving uint32 key) for exact top-k
  device const float* sc = scores + size_t(m) * a.cap;
  uint lo = 0u, hi = 0xFFFFFFFFu;
  for (uint bit = 0; bit < 32; bit++) {
    uint mid = lo + ((hi - lo) >> 1);
    uint local_gt = 0;
    for (uint j = tid; j < n_idx; j += ntg) {
      uint bits = as_type<uint>(sc[j]);
      uint key = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
      if (key > mid) local_gt++;
    }
    uint sg = simd_sum(local_gt);
    if ((tid & 31) == 0) hist[tid >> 5] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint tot = 0;
    for (uint k = 0; k < ntg / 32; k++) tot += hist[k];
    if (tot >= a.topk) lo = mid + 1; else hi = mid;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint thresh = lo;
  if (tid == 0) count_hi = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Collect keys > thresh first, then keys == thresh up to topk
  for (uint pass = 0; pass < 2; pass++) {
    for (uint j = tid; j < n_idx; j += ntg) {
      uint bits = as_type<uint>(sc[j]);
      uint key = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
      bool take = (pass == 0) ? (key > thresh) : (key == thresh);
      if (take) {
        uint slot = atomic_fetch_add_explicit((threadgroup atomic_uint*)&count_hi, 1u, memory_order_relaxed);
        if (slot < a.topk) dst[slot] = int(j);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // Append trailing SWA positions
  for (uint i = tid; i < swa_len; i += ntg) dst[a.topk + i] = int(n_idx + i);
  if (tid == 0) sel_len[m] = a.topk + swa_len;
}

// ------------------------------------------------------------------ 12. DSA Sparse MLA Attention
// Grid: (nh=64, M), 128 threads. Reads q_abs [64, M, 512] (grouped), q_rope [M, 64, 64],
// lat_q8 [cap, 672 B], sel [M, 2176], sink [64]. Writes o_lat into grouped bf16 frags [64, M, 512] for attn.vb!
struct DsaAttnArgs { uint M, topk_win, qabs_gstride, vb_gstride_bf16; float scale; };
kernel void e5_dsa_attn(device const float* q_abs [[buffer(0)]],
                        device const float* q_rope [[buffer(1)]],
                        device const uchar* lat_q8 [[buffer(2)]],
                        device const int* sel [[buffer(3)]],
                        device const uint* sel_len [[buffer(4)]],
                        device const float* sink [[buffer(5)]],
                        device bfloat* olat_frags [[buffer(6)]],
                        constant DsaAttnArgs& a [[buffer(7)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        ushort tid [[thread_index_in_threadgroup]],
                        uint2 ntg2 [[threads_per_threadgroup]]) {
  const ushort ntg = ushort(ntg2.x);
  threadgroup float sh[32];
  const uint h = tg.x, m = tg.y;
  const uint L = sel_len[m];
  device const int* sm = sel + size_t(m) * a.topk_win;
  device const float* qa = q_abs + size_t(h) * a.qabs_gstride + size_t(m) * 512;
  device const float* qr = q_rope + (size_t(m) * 64 + h) * 64;

  float q0 = qa[tid], q1 = qa[tid + 128], q2 = qa[tid + 256], q3 = qa[tid + 384];
  float qr_v = (tid < 64) ? qr[tid] : 0.0f;
  float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
  float mx = sink[h];
  float lse = 1.0f;

  const uint g0 = tid >> 6, g1 = (tid + 128) >> 6, g2 = (tid + 256) >> 6, g3 = (tid + 384) >> 6;
  for (uint j = 0; j < L; j++) {
    device const uchar* lrow = lat_q8 + size_t(sm[j]) * 672;
    device const char* lq = (device const char*)lrow;
    device const half* ls = (device const half*)(lrow + 512);
    device const bfloat* lr = (device const bfloat*)(lrow + 528);

    float k0 = float(lq[tid]) * float(ls[g0]);
    float k1 = float(lq[tid + 128]) * float(ls[g1]);
    float k2 = float(lq[tid + 256]) * float(ls[g2]);
    float k3 = float(lq[tid + 384]) * float(ls[g3]);
    float dot = fma(q0, k0, fma(q1, k1, fma(q2, k2, q3 * k3)));
    if (tid < 64) dot = fma(qr_v, float(lr[tid]), dot);

    float sc = tg_sum(dot, sh, tid, ntg) * a.scale;
    float nmx = max(mx, sc);
    float alpha = exp(mx - nmx);
    float p = exp(sc - nmx);
    lse = lse * alpha + p;
    o0 = fma(p, k0, o0 * alpha);
    o1 = fma(p, k1, o1 * alpha);
    o2 = fma(p, k2, o2 * alpha);
    o3 = fma(p, k3, o3 * alpha);
    mx = nmx;
  }
  const float inv = 1.0f / lse;
  device bfloat* gdst = olat_frags + size_t(h) * a.vb_gstride_bf16;
  gdst[frag_off(m, tid, 512)] = bfloat(o0 * inv);
  gdst[frag_off(m, tid + 128, 512)] = bfloat(o1 * inv);
  gdst[frag_off(m, tid + 256, 512)] = bfloat(o2 * inv);
  gdst[frag_off(m, tid + 384, 512)] = bfloat(o3 * inv);
}

// ------------------------------------------------------------------ 13. DSA VHA + Gate -> o_proj frags
// o_in is grouped [64, M, 256] (stride o_gstride), gate is qb_out[m, 16384..32767].
struct DsaVhaArgs { uint M, o_gstride; };
kernel void e5_dsa_vha_gate(device const float* o_grp [[buffer(0)]],
                            device const float* vha_m [[buffer(1)]],
                            device const float* qb_out [[buffer(2)]],
                            device bfloat* dst_frag [[buffer(3)]],
                            constant DsaVhaArgs& a [[buffer(4)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]]) {
  const uint h = tg.x, m = tg.y;
  if (tid >= 256) return;
  device const float* mrow = vha_m + h * 64;
  float acc = 0.0f;
  for (uint g = 0; g < 64; g++)
    acc = fma(mrow[g], o_grp[size_t(g) * a.o_gstride + size_t(m) * 256 + tid], acc);
  const uint col = h * 256 + tid;
  const float gv = qb_out[size_t(m) * 40960 + 16384 + col];
  dst_frag[frag_off(m, col, 16384)] = bfloat(acc * sigm(gv));
}

// ------------------------------------------------------------------ 14. SiTU-GLU + frag pack
// Interleaved by 16: each 32-col block has gate[0..15] then up[0..15].
struct SituArgs { uint M, I, src_stride, src_off, dst_K, dst_off; float beta, lbeta; };
kernel void e5_situ_pack(device const float* src [[buffer(0)]],
                         device bfloat* dst_frag [[buffer(1)]],
                         constant SituArgs& a [[buffer(2)]],
                         uint2 g [[thread_position_in_grid]]) {
  const uint c = g.x, m = g.y;
  if (c >= a.I || m >= a.M) return;
  const uint blk = c >> 4, off = c & 15u;
  device const float* row = src + size_t(m) * a.src_stride + a.src_off;
  const float gv = row[blk * 32 + off];
  const float uv = row[blk * 32 + 16 + off];
  dst_frag[frag_off(m, a.dst_off + c, a.dst_K)] = bfloat(situ_act(gv, uv, a.beta, a.lbeta));
}

// Pack plain f32 rows [M, K] (from offset src_off in stride src_stride) into bf16 frags [M, K]
struct PackF32Args { uint M, K, src_stride, src_off; };
kernel void e5_pack_f32(device const float* src [[buffer(0)]],
                        device bfloat* dst_frag [[buffer(1)]],
                        constant PackF32Args& a [[buffer(2)]],
                        uint2 g [[thread_position_in_grid]]) {
  const uint c = g.x, m = g.y;
  if (c >= a.K || m >= a.M) return;
  dst_frag[frag_off(m, c, a.K)] = bfloat(src[size_t(m) * a.src_stride + a.src_off + c]);
}

// ------------------------------------------------------------------ 15. Multi-Router Top-10
// moe_in [M, 7168] has r0[0..511] | r1[512..1023] | fc1[1024..3071] | sgu[3072..7167].
struct RouteArgs { uint M, E, topk; };
kernel void e5_route_topk(device const float* moe_in [[buffer(0)]],
                          device const float* bias [[buffer(1)]],
                          device const float* rscale [[buffer(2)]],
                          device uint* top_ids [[buffer(3)]],
                          device float* top_w [[buffer(4)]],
                          constant RouteArgs& a [[buffer(5)]],
                          uint m [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup float sc[512];
  threadgroup float sel[512];
  device const float* row = moe_in + size_t(m) * 7168;
  for (uint e = tid; e < 512; e += 256) {
    float s = sigm(row[e]) + sigm(row[512 + e]);
    sc[e] = s;
    sel[e] = s + bias[e];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint ids[10];
    float ws[10];
    float wsum = 0.0f;
    for (uint k = 0; k < 10; k++) {
      float best = -1e30f;
      uint bi = 0;
      for (uint e = 0; e < 512; e++) {
        if (sel[e] > best) { best = sel[e]; bi = e; }
      }
      ids[k] = bi;
      ws[k] = sc[bi];
      wsum += ws[k];
      sel[bi] = -1e30f;
    }
    const float inv = 1.0f / max(wsum, 1e-20f);
    for (uint k = 0; k < 10; k++) {
      top_ids[m * 10 + k] = ids[k];
      top_w[m * 10 + k] = ws[k] * inv * rscale[ids[k]];
    }
  }
}

// ------------------------------------------------------------------ 16. MoE Combine + LatentNorm + Concat Shared -> moe.out frags
// exp_out [M * 10, 2048] f32 (each slot k of token m), top_w [M, 10], lnorm [2048] bf16,
// moe_in [M, 7168] (sgu at offset 3072..7167). Writes moe_out_frag [M, 4096] = [routed_normed 2048 | shared_act 2048].
struct MoeCombArgs { uint M, topk; float eps, beta, lbeta; };
kernel void e5_moe_combine_pack(device const float* exp_out [[buffer(0)]],
                                device const float* top_w [[buffer(1)]],
                                device const bfloat* lnorm [[buffer(2)]],
                                device const float* moe_in [[buffer(3)]],
                                device bfloat* dst_frag [[buffer(4)]],
                                constant MoeCombArgs& a [[buffer(5)]],
                                uint m [[threadgroup_position_in_grid]],
                                ushort tid [[thread_index_in_threadgroup]],
                                ushort ntg [[threads_per_threadgroup]]) {
  threadgroup float sh[32];
  threadgroup float comb[2048];
  float ss = 0.0f;
  for (uint c = tid; c < 2048; c += ntg) {
    float acc = 0.0f;
    for (uint k = 0; k < a.topk; k++)
      acc = fma(top_w[m * a.topk + k], exp_out[(size_t(m) * a.topk + k) * 2048 + c], acc);
    comb[c] = acc;
    ss = fma(acc, acc, ss);
  }
  const float r = rsqrt(tg_sum(ss, sh, tid, ntg) / 2048.0f + a.eps);
  device const float* sgu = moe_in + size_t(m) * 7168 + 3072;
  for (uint c = tid; c < 2048; c += ntg) {
    dst_frag[frag_off(m, c, 4096)] = bfloat(comb[c] * r * float(lnorm[c]));
    const uint blk = c >> 4, off = c & 15u;
    const float sv = situ_act(sgu[blk * 32 + off], sgu[blk * 32 + 16 + off], a.beta, a.lbeta);
    dst_frag[frag_off(m, 2048 + c, 4096)] = bfloat(sv);
  }
}

// ------------------------------------------------------------------ 17. MultiMax + Argmax
struct MultimaxArgs { uint M, V; };
kernel void e5_multimax(device float* logits [[buffer(0)]],
                        device const float* r [[buffer(1)]],
                        device const float* t [[buffer(2)]],
                        constant MultimaxArgs& a [[buffer(3)]],
                        uint2 g [[thread_position_in_grid]]) {
  const uint v = g.x, m = g.y;
  if (v >= a.V || m >= a.M) return;
  const size_t idx = size_t(m) * a.V + v;
  const float x = logits[idx];
  const float a0 = max(r[0] - x, 0.0f);
  const float b0 = max(x - r[1], 0.0f);
  const float c0 = max(r[2] - x, 0.0f);
  const float d0 = max(x - r[3], 0.0f);
  logits[idx] = x + t[0] * a0 + t[1] * b0 + t[2] * (c0 * c0) + t[3] * (d0 * d0);
}
