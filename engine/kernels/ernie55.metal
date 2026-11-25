// ernie55.metal: ERNIE-5.5-Flash-Preview Metal 4 ops (everything outside mm2/mmp/mmpg).
// All arithmetic in f32; bf16 only for matmul fragment operands and compact KV caches.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

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
  uint M, H, inject, mode, sk_iters, npart;
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
struct HcaPrepArgs { uint M, p0, ring, cr, nope, rdim, comp_ring, use_tmp; float eps; };
// Fused M=1 HCA prep: 4 TGs x 32 threads (1 SG/TG, ZERO TG barriers, float4 vectorized!)
// TG 0: qr RMSNorm (1024) -> qr_frag; TG 1: kv RMSNorm (512) + RoPE -> swa_kv; TG 2: comp_kv; TG 3: comp_sc + ape.
kernel void e5_hca_prep_m1(device const float* in_buf [[buffer(0)]],
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
                           uint tg [[threadgroup_position_in_grid]],
                           ushort l [[thread_index_in_simdgroup]]) {
  const uint pos = a.p0;
  if (tg == 0u) {
    float4 x[8];
    float ss = 0.0f;
    for (uint k = 0; k < 8u; k++) {
      const uint c4 = (k * 32u + l) << 2;
      x[k] = *(device const float4*)(in_buf + c4);
      ss += dot(x[k], x[k]);
    }
    const float rq = rsqrt(simd_sum(ss) / 1024.0f + a.eps);
    for (uint k = 0; k < 8u; k++) {
      const uint c4 = (k * 32u + l) << 2;
      const bfloat4 w4 = *(device const bfloat4*)(q_nw + c4);
      const float4 v = x[k] * rq * float4(w4);
      *(device bfloat4*)(qr_frag + frag_off(0u, c4, 1024u)) = bfloat4(bfloat(v.x), bfloat(v.y), bfloat(v.z), bfloat(v.w));
    }
  } else if (tg == 1u) {
    const uint c4 = uint(l) * 16u;
    device const float* src = in_buf + 1024u + c4;
    const float4 v0 = *(device const float4*)(src + 0u);
    const float4 v1 = *(device const float4*)(src + 4u);
    const float4 v2 = *(device const float4*)(src + 8u);
    const float4 v3 = *(device const float4*)(src + 12u);
    const float rk = rsqrt(simd_sum(dot(v0, v0) + dot(v1, v1) + dot(v2, v2) + dot(v3, v3)) / 512.0f + a.eps);
    float4 r0 = v0 * rk * float4(*(device const bfloat4*)(kv_nw + c4 + 0u));
    float4 r1 = v1 * rk * float4(*(device const bfloat4*)(kv_nw + c4 + 4u));
    float4 r2 = v2 * rk * float4(*(device const bfloat4*)(kv_nw + c4 + 8u));
    float4 r3 = v3 * rk * float4(*(device const bfloat4*)(kv_nw + c4 + 12u));
    if (l >= 28u) {
      const uint j0 = (c4 - a.nope) >> 1;
      device const float* cp = cos_t + size_t(pos) * (a.rdim / 2u) + j0;
      device const float* sp = sin_t + size_t(pos) * (a.rdim / 2u) + j0;
      auto rot4 = [&](float4 v, uint jo) -> float4 {
        float c0 = cp[jo], s0 = sp[jo], c1 = cp[jo + 1], s1 = sp[jo + 1];
        return float4(v.x * c0 - v.y * s0, v.y * c0 + v.x * s0, v.z * c1 - v.w * s1, v.w * c1 + v.z * s1);
      };
      r0 = rot4(r0, 0u); r1 = rot4(r1, 2u); r2 = rot4(r2, 4u); r3 = rot4(r3, 6u);
    }
    device bfloat* dst = swa_kv + size_t(pos % a.ring) * 512u + c4;
    *(device bfloat4*)(dst + 0u) = bfloat4(bfloat(r0.x), bfloat(r0.y), bfloat(r0.z), bfloat(r0.w));
    *(device bfloat4*)(dst + 4u) = bfloat4(bfloat(r1.x), bfloat(r1.y), bfloat(r1.z), bfloat(r1.w));
    *(device bfloat4*)(dst + 8u) = bfloat4(bfloat(r2.x), bfloat(r2.y), bfloat(r2.z), bfloat(r2.w));
    *(device bfloat4*)(dst + 12u) = bfloat4(bfloat(r3.x), bfloat(r3.y), bfloat(r3.z), bfloat(r3.w));
  } else if (tg == 2u) {
    const uint cslot = pos % a.comp_ring;
    const uint c4 = uint(l) * 16u;
    device const float* src = in_buf + 1536u + c4;
    device float* dst = comp_kv + size_t(cslot) * 512u + c4;
    *(device float4*)(dst + 0u) = *(device const float4*)(src + 0u);
    *(device float4*)(dst + 4u) = *(device const float4*)(src + 4u);
    *(device float4*)(dst + 8u) = *(device const float4*)(src + 8u);
    *(device float4*)(dst + 12u) = *(device const float4*)(src + 12u);
  } else {
    const uint cslot = pos % a.comp_ring;
    const uint ape_r = pos % a.cr;
    const uint c4 = uint(l) * 16u;
    device const float* src = in_buf + 2048u + c4;
    device const float* ap = ape + size_t(ape_r) * 512u + c4;
    device float* dst = comp_sc + size_t(cslot) * 512u + c4;
    *(device float4*)(dst + 0u) = *(device const float4*)(src + 0u) + *(device const float4*)(ap + 0u);
    *(device float4*)(dst + 4u) = *(device const float4*)(src + 4u) + *(device const float4*)(ap + 4u);
    *(device float4*)(dst + 8u) = *(device const float4*)(src + 8u) + *(device const float4*)(ap + 8u);
    *(device float4*)(dst + 12u) = *(device const float4*)(src + 12u) + *(device const float4*)(ap + 12u);
  }
}

// Splits attn.in [M, 2560] = qd(1024) | kv(512) | ckv(512) | cgate(512).
// Writes qr_frag (1024 bf16 frag), kv_swa ring [RING_HCA, 512] bf16, and comp partial [128, 512] f32.
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
                        device float* comp_kv_tmp [[buffer(11)]],
                        device float* comp_sc_tmp [[buffer(12)]],
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

  // 3) Compressor partial state: in prefill (use_tmp != 0), write current chunk to shared scratch
  // AND persist the trailing < 128 incomplete tokens into the layer's compact comp_ring!
  const uint ape_r = pos % a.cr;
  const uint cslot = pos % a.comp_ring;
  const uint tail_start = ((a.p0 + a.M) / a.cr) * a.cr;
  if (a.use_tmp) {
    const bool keep = (pos >= tail_start);
    for (uint c = tid; c < 512; c += ntg) {
      const float kv_v = row[1536 + c];
      const float sc_v = row[2048 + c] + ape[size_t(ape_r) * 512 + c];
      comp_kv_tmp[size_t(m) * 512 + c] = kv_v;
      comp_sc_tmp[size_t(m) * 512 + c] = sc_v;
      if (keep) {
        comp_kv[size_t(cslot) * 512 + c] = kv_v;
        comp_sc[size_t(cslot) * 512 + c] = sc_v;
      }
    }
  } else {
    for (uint c = tid; c < 512; c += ntg) {
      comp_kv[size_t(cslot) * 512 + c] = row[1536 + c];
      comp_sc[size_t(cslot) * 512 + c] = row[2048 + c] + ape[size_t(ape_r) * 512 + c];
    }
  }
}

// ------------------------------------------------------------------ 4. HCA C128 Compress
// One threadgroup per completed 128-token block b (covering pos = b*128 .. b*128+127).
struct HcaCompArgs { uint b0, ring, cr, nope, rdim, p0, use_tmp; float eps; };
kernel void e5_hca_compress(device const float* comp_kv [[buffer(0)]],
                            device const float* comp_sc [[buffer(1)]],
                            device const bfloat* cnw [[buffer(2)]],
                            device const float* cos_t [[buffer(3)]],
                            device const float* sin_t [[buffer(4)]],
                            device bfloat* comp_cache [[buffer(5)]],
                            constant HcaCompArgs& a [[buffer(6)]],
                            device const float* comp_kv_tmp [[buffer(7)]],
                            device const float* comp_sc_tmp [[buffer(8)]],
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
      const uint p = p_start + k;
      const float sc = (a.use_tmp && p >= a.p0)
          ? comp_sc_tmp[size_t(p - a.p0) * 512 + tid]
          : comp_sc[size_t(p % a.ring) * 512 + tid];
      mx = max(mx, sc);
    }
    float sm = 0.0f, acc = 0.0f;
    for (uint k = 0; k < a.cr; k++) {
      const uint p = p_start + k;
      const bool from_tmp = (a.use_tmp && p >= a.p0);
      const float sc = from_tmp ? comp_sc_tmp[size_t(p - a.p0) * 512 + tid]
                                : comp_sc[size_t(p % a.ring) * 512 + tid];
      const float kv = from_tmp ? comp_kv_tmp[size_t(p - a.p0) * 512 + tid]
                                : comp_kv[size_t(p % a.ring) * 512 + tid];
      float w = exp(sc - mx);
      sm += w;
      acc = fma(w, kv, acc);
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

// ------------------------------------------------------------------ 5. HCA Q-norm + RoPE (1 SG/TG, float4 -> bfloat4!)
// q_in is inside attn.qb output [M, 40960] at offset 0 (stride 40960), 64 heads x 512.
// Writes q_out as bfloat [M, 64, 512] (halves DRAM traffic for hca_qnorm + hca_attn, bit-exact for matmul2d!).
struct HcaQArgs { uint M, p0, nh, hd, nope, rdim, stride; float eps; };
kernel void e5_hca_qnorm_rope(device const float* qb_out [[buffer(0)]],
                              device const float* cos_t [[buffer(1)]],
                              device const float* sin_t [[buffer(2)]],
                              device bfloat* q_out [[buffer(3)]],
                              constant HcaQArgs& a [[buffer(4)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              ushort l [[thread_index_in_simdgroup]]) {
  const uint h = tg.x, m = tg.y, pos = a.p0 + m;
  const uint d_off = uint(l) * 16u;
  device const float* src = qb_out + size_t(m) * a.stride + h * a.hd + d_off;
  float4 q0 = *(device const float4*)(src + 0u);
  float4 q1 = *(device const float4*)(src + 4u);
  float4 q2 = *(device const float4*)(src + 8u);
  float4 q3 = *(device const float4*)(src + 12u);
  const float r = rsqrt(simd_sum(dot(q0, q0) + dot(q1, q1) + dot(q2, q2) + dot(q3, q3)) / float(a.hd) + a.eps);
  q0 *= r; q1 *= r; q2 *= r; q3 *= r;
  if (l >= 28u) {
    const uint j0 = (d_off - a.nope) >> 1;
    device const float* cp = cos_t + size_t(pos) * (a.rdim / 2u) + j0;
    device const float* sp = sin_t + size_t(pos) * (a.rdim / 2u) + j0;
    auto rot4 = [&](float4 v, uint jo) -> float4 {
      float c0 = cp[jo], s0 = sp[jo], c1 = cp[jo + 1], s1 = sp[jo + 1];
      return float4(v.x * c0 - v.y * s0, v.y * c0 + v.x * s0, v.z * c1 - v.w * s1, v.w * c1 + v.z * s1);
    };
    q0 = rot4(q0, 0u); q1 = rot4(q1, 2u); q2 = rot4(q2, 4u); q3 = rot4(q3, 6u);
  }
  device bfloat* dst = q_out + (size_t(m) * a.nh + h) * a.hd + d_off;
  *(device bfloat4*)(dst + 0u) = bfloat4(bfloat(q0.x), bfloat(q0.y), bfloat(q0.z), bfloat(q0.w));
  *(device bfloat4*)(dst + 4u) = bfloat4(bfloat(q1.x), bfloat(q1.y), bfloat(q1.z), bfloat(q1.w));
  *(device bfloat4*)(dst + 8u) = bfloat4(bfloat(q2.x), bfloat(q2.y), bfloat(q2.z), bfloat(q2.w));
  *(device bfloat4*)(dst + 12u) = bfloat4(bfloat(q3.x), bfloat(q3.y), bfloat(q3.z), bfloat(q3.w));
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
                          device bfloat* iq_out [[buffer(5)]],
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
  // Indexer Q: partial NeoX RoPE on leading 64 dims + Hadamard 128 -> bf16
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
  if (tid < 128) iq_out[(size_t(m) * 64 + h) * 128 + tid] = bfloat(buf128[tid]);
}

// ------------------------------------------------------------------ 11. DSA Indexer Score & Top-K
// 11. DSA Indexer Score & Top-K (supports row0 sub-batching so idx_sc stays 32 MB)
struct DsaIdxArgs { uint M, p0, win, topk, cap, row0; };
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

static inline uint e5_fkey(float x) {
  uint b = as_type<uint>(x);
  return (b & 0x80000000u) ? ~b : (b ^ 0x80000000u);
}

// Fast 4-pass 256-bin radix top-2048 selection with TG candidate compaction (2 DRAM passes instead of 32).
// For n_idx <= topk (<= 2048), emits 0..pos directly.
kernel void e5_dsa_idx_topk(device const float* scores [[buffer(0)]],
                            device int* sel_out [[buffer(1)]],    // [M_tot, 2176]
                            device uint* sel_len [[buffer(2)]],   // [M_tot]
                            constant DsaIdxArgs& a [[buffer(3)]],
                            uint mb [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]],
                            ushort ntg [[threads_per_threadgroup]]) {
  threadgroup atomic_uint hist[256];
  threadgroup uint cnt_s[8];
  threadgroup uint pref, need, use_cand;
  threadgroup uint cand_k[2048];
  threadgroup atomic_uint cand_n, count_hi;
  const uint m = a.row0 + mb;
  const uint pos = a.p0 + m;
  const uint n_idx = (pos + 1 > a.win) ? (pos + 1 - a.win) : 0u;
  const uint swa_len = pos + 1 - n_idx;
  device int* dst = sel_out + size_t(m) * (a.topk + a.win);

  if (n_idx <= a.topk) {
    for (uint i = tid; i < pos + 1; i += ntg) dst[i] = int(i);
    if (tid == 0) sel_len[m] = pos + 1;
    return;
  }

  device const float* sc = scores + size_t(mb) * a.cap;
  if (tid == 0) {
    pref = 0u; need = a.topk; use_cand = 0u;
    atomic_store_explicit(&cand_n, 0u, memory_order_relaxed);
    atomic_store_explicit(&count_hi, 0u, memory_order_relaxed);
  }
  uint mask = 0u;
  for (int shift = 24; shift >= 0; shift -= 8) {
    atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint p = pref, nd = need, uc = use_cand;
    if (shift <= 8 && uc) {
      const uint nc = min(2048u, atomic_load_explicit(&cand_n, memory_order_relaxed));
      for (uint i = tid; i < nc; i += ntg) {
        const uint k = cand_k[i];
        if ((k & mask) == p) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
      }
    } else if (shift == 16 && uc) {
      for (uint j = tid; j < n_idx; j += ntg) {
        const uint k = e5_fkey(sc[j]);
        if ((k & mask) == p) {
          atomic_fetch_add_explicit(&hist[(k >> 16) & 255u], 1u, memory_order_relaxed);
          const uint ci = atomic_fetch_add_explicit(&cand_n, 1u, memory_order_relaxed);
          if (ci < 2048u) cand_k[ci] = k;
        }
      }
    } else {
      for (uint j = tid; j < n_idx; j += ntg) {
        const uint k = e5_fkey(sc[j]);
        if ((k & mask) == p) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint h = atomic_load_explicit(&hist[255u - tid], memory_order_relaxed);
    const uint inc = simd_prefix_inclusive_sum(h);
    if ((tid & 31u) == 31u) cnt_s[tid >> 5] = inc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint sg = tid >> 5;
    uint base = 0u;
    for (uint s = 0; s < sg; s++) base += cnt_s[s];
    const uint hi = base + inc;
    const uint lo = hi - h;
    if (lo < nd && hi >= nd) {
      pref = p | ((255u - tid) << shift);
      need = nd - lo;
      if (shift == 24) use_cand = (h <= 2048u) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mask |= 255u << shift;
  }
  const uint thresh = pref;

  // Collect keys > thresh first, then keys == thresh up to topk
  for (uint pass = 0; pass < 2; pass++) {
    for (uint j = tid; j < n_idx; j += ntg) {
      const uint key = e5_fkey(sc[j]);
      const bool take = (pass == 0) ? (key > thresh) : (key == thresh);
      if (take) {
        uint slot = atomic_fetch_add_explicit(&count_hi, 1u, memory_order_relaxed);
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

// Fast Walsh-Hadamard Transform (N=2048, normalized, self-inverse)
static inline void tg_fwht_2048(threadgroup float* buf, ushort tid, ushort ntg) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = 0; s < 11; s++) {
    const uint stride = 1u << s;
    const uint mask = stride - 1u;
    for (uint b = tid; b < 1024; b += ntg) {
      const uint i0 = ((b >> s) << (s + 1)) + (b & mask);
      const uint i1 = i0 + stride;
      const float a = buf[i0];
      const float c = buf[i1];
      buf[i0] = a + c;
      buf[i1] = a - c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  constexpr float inv_sq = 0.02209708691207961f; // 1 / sqrt(2048)
  for (uint c = tid; c < 2048; c += ntg) buf[c] *= inv_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Grid: (M_rows, 1, 1), 256 threads per threadgroup. Computes SiTU-GLU on I=2048, applies FWHT-2048, packs frags.
kernel void e5_situ_had_pack(device const float* src [[buffer(0)]],
                             device bfloat* dst_frag [[buffer(1)]],
                             constant SituArgs& a [[buffer(2)]],
                             uint m [[threadgroup_position_in_grid]],
                             ushort tid [[thread_index_in_threadgroup]],
                             ushort ntg [[threads_per_threadgroup]]) {
  threadgroup float buf[2048];
  device const float* row = src + size_t(m) * a.src_stride + a.src_off;
  for (uint c = tid; c < 2048; c += ntg) {
    const uint blk = c >> 4, off = c & 15u;
    buf[c] = situ_act(row[blk * 32 + off], row[blk * 32 + 16 + off], a.beta, a.lbeta);
  }
  tg_fwht_2048(buf, tid, ntg);
  for (uint c = tid; c < 2048; c += ntg) {
    dst_frag[frag_off(m, a.dst_off + c, a.dst_K)] = bfloat(buf[c]);
  }
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

// ------------------------------------------------------------------ 15. Multi-Router Top-10 (parallel SIMD)
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
  threadgroup float sh_v[8];
  threadgroup uint sh_i[8];
  threadgroup float sh_s[8];
  threadgroup float out_w[10];
  threadgroup uint out_i[10];
  device const float* row = moe_in + size_t(m) * 7168;
  const uint e0 = tid, e1 = tid + 256u;
  float sc0 = sigm(row[e0]) + sigm(row[512u + e0]);
  float sc1 = sigm(row[e1]) + sigm(row[512u + e1]);
  float v0 = sc0 + bias[e0];
  float v1 = sc1 + bias[e1];
  const ushort sgi = tid >> 5, lane = tid & 31u;

  for (uint k = 0; k < 10u; k++) {
    bool pick0 = (v0 >= v1);
    float mv = pick0 ? v0 : v1;
    uint mi = pick0 ? e0 : e1;
    float ms = pick0 ? sc0 : sc1;
    float sg_mx = simd_max(mv);
    uint sg_mi = simd_min((mv == sg_mx) ? mi : 999u);
    if (mi == sg_mi) {
      sh_v[sgi] = mv;
      sh_i[sgi] = mi;
      sh_s[sgi] = ms;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float bv = -1e30f;
      uint bi = 999u;
      float bs = 0.0f;
      for (uint g = 0; g < 8u; g++) {
        if (sh_v[g] > bv || (sh_v[g] == bv && sh_i[g] < bi)) {
          bv = sh_v[g]; bi = sh_i[g]; bs = sh_s[g];
        }
      }
      out_i[k] = bi;
      out_w[k] = bs;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint chosen = out_i[k];
    if (e0 == chosen) v0 = -1e30f;
    if (e1 == chosen) v1 = -1e30f;
  }
  if (tid == 0) {
    float wsum = 0.0f;
    for (uint k = 0; k < 10u; k++) wsum += out_w[k];
    const float inv = 1.0f / max(wsum, 1e-20f);
    for (uint k = 0; k < 10u; k++) {
      top_ids[m * 10u + k] = out_i[k];
      top_w[m * 10u + k] = out_w[k] * inv * rscale[out_i[k]];
    }
  }
}

// Fast M=1 route_topk + fc1 pack:
// TG 0 (32 threads = 1 SG, ZERO TG barriers!): 16 experts/lane in registers, 10x simd_max!
// TGs 1..8 (64 threads each): vectorized float4 -> bfloat4 pack of fc1[1024..3071] -> frag_a!
kernel void e5_route_topk_pack_m1(device const float* moe_in [[buffer(0)]],
                                  device const float* bias [[buffer(1)]],
                                  device const float* rscale [[buffer(2)]],
                                  device uint* top_ids [[buffer(3)]],
                                  device float* top_w [[buffer(4)]],
                                  device bfloat* fc1_frag [[buffer(5)]],
                                  constant RouteArgs& a [[buffer(6)]],
                                  uint tg [[threadgroup_position_in_grid]],
                                  ushort l [[thread_index_in_threadgroup]]) {
  if (tg > 0u) {
    const uint c4 = ((tg - 1u) * 64u + l) << 2;
    if (c4 >= 2048u) return;
    const float4 v = *(device const float4*)(moe_in + 1024u + c4);
    *(device bfloat4*)(fc1_frag + frag_off(0u, c4, 2048u)) = bfloat4(bfloat(v.x), bfloat(v.y), bfloat(v.z), bfloat(v.w));
    return;
  }
  if (l >= 32u) return;
  float v[16], sc[16];
  uint id[16];
  for (uint j = 0; j < 16u; j++) {
    const uint e = j * 32u + l;
    const float s = sigm(moe_in[e]) + sigm(moe_in[512u + e]);
    sc[j] = s;
    v[j] = s + bias[e];
    id[j] = e;
  }
  uint out_i[10];
  float out_w[10];
  for (uint k = 0; k < 10u; k++) {
    float mv = v[0];
    uint mi = id[0];
    float ms = sc[0];
    uint mj = 0u;
    for (uint j = 1; j < 16u; j++) {
      if (v[j] > mv) { mv = v[j]; mi = id[j]; ms = sc[j]; mj = j; }
    }
    const float sg_mx = simd_max(mv);
    const uint sg_mi = simd_min((mv == sg_mx) ? mi : 999u);
    const float sg_ms = simd_max((mi == sg_mi) ? ms : -1e30f);
    out_i[k] = sg_mi;
    out_w[k] = sg_ms;
    if (mi == sg_mi) v[mj] = -1e30f;
  }
  if (l == 0u) {
    float wsum = 0.0f;
    for (uint k = 0; k < 10u; k++) wsum += out_w[k];
    const float inv = 1.0f / max(wsum, 1e-20f);
    for (uint k = 0; k < 10u; k++) {
      top_ids[k] = out_i[k];
      top_w[k] = out_w[k] * inv * rscale[out_i[k]];
    }
  }
}

// ------------------------------------------------------------------ 16. MoE Combine + LatentNorm + Concat Shared -> moe.out frags
// exp_out [M * 10, 2048] f32 (each slot k of token m), top_w [M, 10], lnorm [2048] bf16,
// moe_in [M, 7168] (sgu at offset 3072..7167). Writes moe_out_frag [M, 4096] = [routed_normed 2048 | shared_act 2048].
struct MoeCombArgs { uint M, topk, hadamard; float eps, beta, lbeta; };
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
  if (a.hadamard) tg_fwht_2048(comb, tid, ntg);
  const float r = rsqrt(tg_sum(ss, sh, tid, ntg) / 2048.0f + a.eps);
  device const float* sgu = moe_in + size_t(m) * 7168 + 3072;
  for (uint c = tid; c < 2048; c += ntg) {
    dst_frag[frag_off(m, c, 4096)] = bfloat(comb[c] * r * float(lnorm[c]));
    const uint blk = c >> 4, off = c & 15u;
    const float sv = situ_act(sgu[blk * 32 + off], sgu[blk * 32 + 16 + off], a.beta, a.lbeta);
    dst_frag[frag_off(m, 2048 + c, 4096)] = bfloat(sv);
  }
}

// ------------------------------------------------------------------ 16b. MoE Plan Lists (for mmpg grouped prefill/verify)
struct PlanListArgs { uint M, E, topk, tile_rows, gx_gu, gx_dn; };
kernel void e5_plan_lists(device const uint* topi [[buffer(0)]], device uint* slots [[buffer(1)]],
                          device uint* nslots [[buffer(2)]], device uint* roff [[buffer(3)]],
                          device uint* rcnt [[buffer(4)]], device uint* rlist [[buffer(5)]],
                          device uint* cidx [[buffer(6)]], constant PlanListArgs& a [[buffer(7)]],
                          device uint* wtile [[buffer(8)]], device uint* ind [[buffer(9)]],
                          ushort tid [[thread_index_in_threadgroup]], ushort ntg [[threads_per_threadgroup]]) {
  threadgroup atomic_uint cnt[512];
  threadgroup uint off[512], sgs[32], sgt[32];
  const uint n = a.M * a.topk;
  for (uint e = tid; e < 512; e += ntg) atomic_store_explicit(&cnt[e], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < n; i += ntg) atomic_fetch_add_explicit(&cnt[topi[i]], 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint e = tid;
  const uint c = e < a.E ? atomic_load_explicit(&cnt[e], memory_order_relaxed) : 0u;
  const uint act = c ? 1u : 0u;
  const uint nt = c ? (c + a.tile_rows - 1) / a.tile_rows : 0u;
  const uint pk = (c << 10) | act;
  const uint inc = simd_prefix_inclusive_sum(pk);
  const uint inct = simd_prefix_inclusive_sum(nt);
  if ((tid & 31) == 31) { sgs[tid / 32] = inc; sgt[tid / 32] = inct; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint base = 0, baset = 0;
  for (uint g = 0; g < tid / 32; g++) { base += sgs[g]; baset += sgt[g]; }
  const uint ex = base + inc - pk, o = ex >> 10, z = ex & 1023u;
  const uint tb = baset + inct - nt;
  if (e < 512) off[e] = o;
  if (e < a.E && c) {
    slots[z] = e; roff[z] = o; rcnt[z] = c;
    for (uint t = 0; t < nt; t++) wtile[tb + t] = (z << 16) | t;
  }
  if (e == 511) {
    nslots[0] = (base + inc) & 1023u;
    const uint ttot = baset + inct;
    ind[0] = a.gx_gu; ind[1] = ttot; ind[2] = 1;
    ind[3] = a.gx_dn; ind[4] = ttot; ind[5] = 1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e2 = tid; e2 < 512; e2 += ntg) atomic_store_explicit(&cnt[e2], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < n; i += ntg) {
    const uint x = topi[i];
    const uint j = off[x] + atomic_fetch_add_explicit(&cnt[x], 1u, memory_order_relaxed);
    rlist[j] = i / a.topk;
    cidx[i] = j;
  }
}

// ------------------------------------------------------------------ 16c. MoE Combine (compact cidx) + LatentNorm + Concat Shared
kernel void e5_moe_combine_c_pack(device const float* exp_out [[buffer(0)]],
                                  device const uint* cidx [[buffer(1)]],
                                  device const float* top_w [[buffer(2)]],
                                  device const bfloat* lnorm [[buffer(3)]],
                                  device const float* moe_in [[buffer(4)]],
                                  device bfloat* dst_frag [[buffer(5)]],
                                  constant MoeCombArgs& a [[buffer(6)]],
                                  uint m [[threadgroup_position_in_grid]],
                                  ushort tid [[thread_index_in_threadgroup]],
                                  ushort ntg [[threads_per_threadgroup]]) {
  threadgroup float sh[32];
  threadgroup float comb[2048];
  float ss = 0.0f;
  for (uint c = tid; c < 2048; c += ntg) {
    float acc = 0.0f;
    for (uint k = 0; k < a.topk; k++)
      acc = fma(top_w[m * a.topk + k], exp_out[size_t(cidx[m * a.topk + k]) * 2048 + c], acc);
    comb[c] = acc;
    ss = fma(acc, acc, ss);
  }
  if (a.hadamard) tg_fwht_2048(comb, tid, ntg);
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

// ==================================================================
// 18. M5 matmul2d Accelerated Attention & Indexer Kernels
// ==================================================================

// 18a. DSA Indexer Score (matmul2d: 4 simdgroups = all 64 heads, 32 KV positions per TG)
// Reads bf16 iq_h and factors positive ksc out of ReLU(dot(q, int8_k * ksc)) = ksc * ReLU(dot(q, int8_k))!
// Grid: (ceil(max_n_idx / 32), M), 128 threads (4 simdgroups).
kernel void e5_dsa_idx_score_mma(device const bfloat* iq_h [[buffer(0)]],
                                 device const float* iw [[buffer(1)]],
                                 device const uchar* idx_k8 [[buffer(2)]],
                                 device float* scores [[buffer(3)]],
                                 constant DsaIdxArgs& a [[buffer(4)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 ushort sg [[simdgroup_index_in_threadgroup]],
                                 ushort l [[thread_index_in_simdgroup]]) {
  const uint j0 = tg.x * 32u, mb = tg.y, m = a.row0 + mb;
  const uint pos = a.p0 + m;
  const uint n_idx = (pos + 1 > a.win) ? (pos + 1 - a.win) : 0u;
  if (mb >= a.M || j0 >= n_idx) return;

  threadgroup bfloat Kt[32][128];
  threadgroup float ksc_sh[32];
  threadgroup float part[4][32];
  const ushort tid = sg * 32u + l;
  if (tid < 32u) {
    const uint j = j0 + tid;
    ksc_sh[tid] = (j < n_idx) ? float(*(device const half*)(idx_k8 + size_t(j) * 136u + 128u)) : 0.0f;
  }
  // Cooperatively unpack 32 KV rows x 128 int8 dims -> bfloat into Kt (zero multiplies!)
  for (uint idx = tid; idx < 32u * 32u; idx += 128u) {
    const uint jj = idx >> 5;
    const uint d4 = (idx & 31u) * 4u;
    const uint j = j0 + jj;
    if (j < n_idx) {
      const float4 f4 = float4(*(device const char4*)(idx_k8 + size_t(j) * 136u + d4));
      *(threadgroup bfloat4*)(&Kt[jj][d4]) = bfloat4(bfloat(f4.x), bfloat(f4.y), bfloat(f4.z), bfloat(f4.w));
    } else {
      *(threadgroup bfloat4*)(&Kt[jj][d4]) = bfloat4(bfloat(0.0f));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const ushort m0 = ((l >> 1) & 3) + 4 * (l >> 4), n0 = 4 * (l & 1) + 8 * ((l >> 3) & 1);
  constexpr auto desc =
      matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto qh = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto cs = op.template get_destination_cooperative_tensor<decltype(qh), decltype(wb), float>();
  for (int i = 0; i < 16; i++) cs[i] = 0.0f;

  device const bfloat* qm = iq_h + (size_t(m) * 64u + size_t(sg) * 16u) * 128u;
  for (uint ks = 0; ks < 4; ks++) {
    for (int i = 0; i < 16; i++) {
      const uint h = m0 + 8 * ((i >> 2) & 1), d = ks * 32u + n0 + (i & 3) + 16 * (i >> 3);
      qh[i] = qm[h * 128u + d];
    }
    for (int qq = 0; qq < 4; qq++) {
      const uint jj = m0 + 8 * qq;
      for (int hh = 0; hh < 2; hh++) {
        const uint d = ks * 32u + n0 + 16 * hh;
        const bfloat4 b4 = *(threadgroup const bfloat4*)(&Kt[jj][d]);
        wb[hh * 16 + qq * 4 + 0] = b4.x; wb[hh * 16 + qq * 4 + 1] = b4.y;
        wb[hh * 16 + qq * 4 + 2] = b4.z; wb[hh * 16 + qq * 4 + 3] = b4.w;
      }
    }
    op.run(qh, wb, cs);
  }
  device const float* wm = iw + size_t(m) * 64u + size_t(sg) * 16u;
  const float w0 = wm[m0], w1 = wm[m0 + 8];
  float sc[8];
  for (int j = 0; j < 8; j++) {
    const int i = (j & 3) + 8 * (j >> 2);
    sc[j] = fma(w0, max(cs[i], 0.0f), w1 * max(cs[i + 4], 0.0f));
  }
  for (int j = 0; j < 8; j++) {
    float v = sc[j];
    v += simd_shuffle_xor(v, 2);
    v += simd_shuffle_xor(v, 4);
    v += simd_shuffle_xor(v, 16);
    sc[j] = v;
  }
  if (m0 == 0) {
    for (int j = 0; j < 8; j++) {
      const uint jj = n0 + (j & 3) + 16 * (j >> 2);
      part[sg][jj] = sc[j];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 32u && j0 + tid < n_idx) {
    scores[size_t(mb) * a.cap + j0 + tid] = (part[0][tid] + part[1][tid] + part[2][tid] + part[3][tid]) * ksc_sh[tid];
  }
}

// 18b. HCA MQA Attention (matmul2d: 16 heads/TG, 4 simdgroups x 128 dims = 512 dims in registers!)
// Grid: (4 head_groups, M), 128 threads (4 simdgroups). Reads bf16 q, writes bf16 o_out!
kernel void e5_hca_attn_mma(device const bfloat* q [[buffer(0)]],
                            device const bfloat* comp_cache [[buffer(1)]],
                            device const bfloat* swa_kv [[buffer(2)]],
                            device const float* sink [[buffer(3)]],
                            device bfloat* o_out [[buffer(4)]],
                            constant HcaAttnArgs& a [[buffer(5)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort l [[thread_index_in_simdgroup]]) {
  threadgroup float Sp[4][16][32];
  const uint hg = tg.x, m = tg.y, pos = a.p0 + m;
  const uint h0 = hg * 16u;
  const uint nc = (pos + 1) / a.cr;
  const uint swa_s = (pos + 1 > a.win) ? (pos + 1 - a.win) : 0u;
  const uint n_swa = pos + 1 - swa_s;
  const uint n_tot = nc + n_swa;

  const ushort m0 = ((l >> 1) & 3) + 4 * (l >> 4), n0 = 4 * (l & 1) + 8 * ((l >> 3) & 1);
  constexpr auto desc =
      matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
#define LEFT op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>()
  auto q0 = LEFT; auto q1 = LEFT; auto q2 = LEFT; auto q3 = LEFT; auto pa = LEFT;
#undef LEFT
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto cs = op.template get_destination_cooperative_tensor<decltype(pa), decltype(wb), float>();
  auto o0 = op.template get_destination_cooperative_tensor<decltype(pa), decltype(wb), float>();
  auto o1 = op.template get_destination_cooperative_tensor<decltype(pa), decltype(wb), float>();
  auto o2 = op.template get_destination_cooperative_tensor<decltype(pa), decltype(wb), float>();
  auto o3 = op.template get_destination_cooperative_tensor<decltype(pa), decltype(wb), float>();

  // Load 16 heads x 128 dims of Q (bf16) into registers once (4 k-steps of 32 dims)
  device const bfloat* qbase = q + (size_t(m) * a.nh + h0) * 512u;
  for (int ks = 0; ks < 4; ks++) {
    for (int i = 0; i < 16; i++) {
      const uint hh = m0 + 8 * ((i >> 2) & 1), d = sg * 128u + ks * 32u + n0 + (i & 3) + 16 * (i >> 3);
      const bfloat v = qbase[hh * 512u + d];
      if (ks == 0) q0[i] = v; else if (ks == 1) q1[i] = v; else if (ks == 2) q2[i] = v; else q3[i] = v;
    }
  }
  for (int i = 0; i < 16; i++) { o0[i] = 0.0f; o1[i] = 0.0f; o2[i] = 0.0f; o3[i] = 0.0f; }
  float mx[2] = { sink[h0 + m0], sink[h0 + m0 + 8] };
  float lse[2] = { 1.0f, 1.0f };

  auto kv_ptr = [&](uint j) -> device const bfloat* {
    return (j < nc) ? (comp_cache + size_t(j) * 512u)
                    : (swa_kv + size_t((swa_s + (j - nc)) % a.ring) * 512u);
  };

  for (uint j0 = 0; j0 < n_tot; j0 += 32u) {
    for (int i = 0; i < 16; i++) cs[i] = 0.0f;
    // 1. Q @ K^T over this simdgroup's 128 dims
    device const bfloat* kp[4];
    for (int qq = 0; qq < 4; qq++) {
      const uint j = j0 + m0 + 8 * qq;
      kp[qq] = (j < n_tot) ? kv_ptr(j) : nullptr;
    }
    for (int ks = 0; ks < 4; ks++) {
      const uint d = sg * 128u + ks * 32u + n0;
      for (int qq = 0; qq < 4; qq++) {
        for (int hh = 0; hh < 2; hh++) {
          const bfloat4 b4 = kp[qq] ? *(device const bfloat4*)(kp[qq] + d + 16 * hh) : bfloat4(bfloat(0.0f));
          for (int e = 0; e < 4; e++) wb[hh * 16 + qq * 4 + e] = b4[e];
        }
      }
      if (ks == 0) op.run(q0, wb, cs);
      else if (ks == 1) op.run(q1, wb, cs);
      else if (ks == 2) op.run(q2, wb, cs);
      else op.run(q3, wb, cs);
    }
    // 2. Reduce scores across 4 simdgroups
    for (int i = 0; i < 16; i++) {
      const uint r = m0 + 8 * ((i >> 2) & 1), c = n0 + (i & 3) + 16 * (i >> 3);
      Sp[sg][r][c] = cs[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s[16];
    for (int i = 0; i < 16; i++) {
      const uint r = m0 + 8 * ((i >> 2) & 1), c = n0 + (i & 3) + 16 * (i >> 3);
      const uint j = j0 + c;
      s[i] = (j < n_tot) ? ((Sp[0][r][c] + Sp[1][r][c] + Sp[2][r][c] + Sp[3][r][c]) * a.scale) : -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Online softmax in registers (2 simd_shuffle_xors)
    float alpha[2];
    for (int r = 0; r < 2; r++) {
      float mb = -INFINITY;
      for (int i = 0; i < 16; i++) if (((i >> 2) & 1) == r) mb = max(mb, s[i]);
      mb = max(mb, simd_shuffle_xor(mb, 1));
      mb = max(mb, simd_shuffle_xor(mb, 8));
      const float mn = max(mx[r], mb);
      alpha[r] = exp(mx[r] - mn);
      float bs = 0.0f;
      for (int i = 0; i < 16; i++)
        if (((i >> 2) & 1) == r) { s[i] = exp(s[i] - mn); bs += s[i]; }
      bs += simd_shuffle_xor(bs, 1);
      bs += simd_shuffle_xor(bs, 8);
      lse[r] = lse[r] * alpha[r] + bs;
      mx[r] = mn;
    }
    for (int i = 0; i < 16; i++) {
      const float cc = alpha[(i >> 2) & 1];
      o0[i] *= cc; o1[i] *= cc; o2[i] *= cc; o3[i] *= cc;
      pa[i] = bfloat(s[i]);
    }

    // 4. P @ V (V == K) for this simdgroup's 128 dims (4 output chunks of 32 dims)
    device const bfloat* vp[8];
    for (int hh = 0; hh < 2; hh++)
      for (int e = 0; e < 4; e++) {
        const uint j = j0 + n0 + e + 16 * hh;
        vp[hh * 4 + e] = (j < n_tot) ? kv_ptr(j) : nullptr;
      }
    for (int nc_i = 0; nc_i < 4; nc_i++) {
      const uint d0 = sg * 128u + nc_i * 32u + m0;
      for (int hh = 0; hh < 2; hh++) {
        for (int e = 0; e < 4; e++) {
          device const bfloat* vrow = vp[hh * 4 + e];
          for (int qq = 0; qq < 4; qq++)
            wb[hh * 16 + qq * 4 + e] = vrow ? vrow[d0 + 8 * qq] : bfloat(0.0f);
        }
      }
      if (nc_i == 0) op.run(pa, wb, o0);
      else if (nc_i == 1) op.run(pa, wb, o1);
      else if (nc_i == 2) op.run(pa, wb, o2);
      else op.run(pa, wb, o3);
    }
  }

  // Write normalized output [M, 64, 512] bf16
  const float inv0 = 1.0f / lse[0], inv1 = 1.0f / lse[1];
  device bfloat* dst = o_out + (size_t(m) * a.nh + h0) * 512u;
  for (int nc_i = 0; nc_i < 4; nc_i++) {
    for (int i = 0; i < 16; i++) {
      const uint hh = m0 + 8 * ((i >> 2) & 1), d = sg * 128u + nc_i * 32u + n0 + (i & 3) + 16 * (i >> 3);
      const float v = (nc_i == 0) ? o0[i] : ((nc_i == 1) ? o1[i] : ((nc_i == 2) ? o2[i] : o3[i]));
      dst[hh * 512u + d] = bfloat(v * (((i >> 2) & 1) ? inv1 : inv0));
    }
  }
}

// 18c. DSA Sparse MLA Attention (matmul2d: 16 heads/TG, 4 simdgroups x 128 dims = 512 dims in registers!)
// Scale-folded Q@K^T and (P*Ks)@V_int8 (4x fewer dequant multiplies!), padded Kq[4][32][132], coalesced Kr[2][32][36]!
// Grid: (4 head_groups, M), 128 threads (4 simdgroups).
kernel void e5_dsa_attn_mma(device const float* q_abs [[buffer(0)]],
                            device const float* q_rope [[buffer(1)]],
                            device const uchar* lat_q8 [[buffer(2)]],
                            device const int* sel [[buffer(3)]],
                            device const uint* sel_len [[buffer(4)]],
                            device const float* sink [[buffer(5)]],
                            device bfloat* olat_frags [[buffer(6)]],
                            constant DsaAttnArgs& a [[buffer(7)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort l [[thread_index_in_simdgroup]]) {
  threadgroup float Sp[4][16][32];
  threadgroup char Kq[4][32][128];
  threadgroup half Ks[4][32][2];
  const uint hg = tg.x, m = tg.y;
  const uint h0 = hg * 16u;
  const uint L = sel_len[m];
  device const int* sm = sel + size_t(m) * a.topk_win;

  const ushort m0 = ((l >> 1) & 3) + 4 * (l >> 4), n0 = 4 * (l & 1) + 8 * ((l >> 3) & 1);
  const uint lr = l >> 3, lc = l & 7u;
  constexpr auto desc =
      matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
#define LEFT op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>()
  auto q0 = LEFT; auto q1 = LEFT; auto q2 = LEFT; auto q3 = LEFT; auto qr = LEFT; auto pa = LEFT;
#undef LEFT
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto cs = op.template get_destination_cooperative_tensor<decltype(pa), decltype(wb), float>();
  auto o0 = op.template get_destination_cooperative_tensor<decltype(pa), decltype(wb), float>();
  auto o1 = op.template get_destination_cooperative_tensor<decltype(pa), decltype(wb), float>();
  auto o2 = op.template get_destination_cooperative_tensor<decltype(pa), decltype(wb), float>();
  auto o3 = op.template get_destination_cooperative_tensor<decltype(pa), decltype(wb), float>();

  // Load 16 heads x 128 dims of q_abs into registers once
  for (int ks = 0; ks < 4; ks++) {
    for (int i = 0; i < 16; i++) {
      const uint hh = m0 + 8 * ((i >> 2) & 1), d = sg * 128u + ks * 32u + n0 + (i & 3) + 16 * (i >> 3);
      const bfloat v = bfloat(q_abs[size_t(h0 + hh) * a.qabs_gstride + size_t(m) * 512u + d]);
      if (ks == 0) q0[i] = v; else if (ks == 1) q1[i] = v; else if (ks == 2) q2[i] = v; else q3[i] = v;
    }
  }
  if (sg < 2) {
    device const float* qrb = q_rope + (size_t(m) * 64u + h0) * 64u;
    for (int i = 0; i < 16; i++) {
      const uint hh = m0 + 8 * ((i >> 2) & 1), d = sg * 32u + n0 + (i & 3) + 16 * (i >> 3);
      qr[i] = bfloat(qrb[hh * 64u + d]);
    }
  }
  for (int i = 0; i < 16; i++) { o0[i] = 0.0f; o1[i] = 0.0f; o2[i] = 0.0f; o3[i] = 0.0f; }
  float mx[2] = { sink[h0 + m0], sink[h0 + m0 + 8] };
  float lse[2] = { 1.0f, 1.0f };

  for (uint j0 = 0; j0 < L; j0 += 32u) {
    for (int i = 0; i < 16; i++) cs[i] = 0.0f;
    // 0. Coalesced single load of 32 KV rows x 128 dims into per-simdgroup Kq/Ks
    for (uint r0 = 0; r0 < 32u; r0 += 4u) {
      const uint jj = r0 + lr;
      const uint j = j0 + jj;
      if (j < L) {
        device const uchar* lrow = lat_q8 + size_t(sm[j]) * 672u;
        *(threadgroup uint4*)(&Kq[sg][jj][lc * 16u]) = *(device const uint4*)(lrow + sg * 128u + lc * 16u);
        if (lc < 2u) Ks[sg][jj][lc] = ((device const half*)(lrow + 512u))[sg * 2u + lc];
      } else {
        *(threadgroup uint4*)(&Kq[sg][jj][lc * 16u]) = uint4(0);
        if (lc < 2u) Ks[sg][jj][lc] = half(0.0h);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // 1. Q_abs @ K_lat^T from TG Kq/Ks + Q_rope @ K_rope^T
    for (int ks = 0; ks < 4; ks++) {
      const uint d_loc = ks * 32u + n0;
      const uint g_loc = ks >> 1;
      for (int qq = 0; qq < 4; qq++) {
        const uint jj = m0 + 8u * qq;
        const float ksc = float(Ks[sg][jj][g_loc]);
        for (int hh = 0; hh < 2; hh++) {
          const char4 c4 = *(threadgroup const char4*)(&Kq[sg][jj][d_loc + 16u * hh]);
          wb[hh * 16 + qq * 4 + 0] = bfloat(float(c4.x) * ksc);
          wb[hh * 16 + qq * 4 + 1] = bfloat(float(c4.y) * ksc);
          wb[hh * 16 + qq * 4 + 2] = bfloat(float(c4.z) * ksc);
          wb[hh * 16 + qq * 4 + 3] = bfloat(float(c4.w) * ksc);
        }
      }
      if (ks == 0) op.run(q0, wb, cs);
      else if (ks == 1) op.run(q1, wb, cs);
      else if (ks == 2) op.run(q2, wb, cs);
      else op.run(q3, wb, cs);
    }
    if (sg < 2) {
      const uint dr = sg * 32u + n0;
      for (int qq = 0; qq < 4; qq++) {
        const uint j = j0 + m0 + 8u * qq;
        device const bfloat* lr_p = (j < L) ? (device const bfloat*)(lat_q8 + size_t(sm[j]) * 672u + 528u) : nullptr;
        for (int hh = 0; hh < 2; hh++) {
          const bfloat4 b4 = lr_p ? *(device const bfloat4*)(lr_p + dr + 16 * hh) : bfloat4(bfloat(0.0f));
          wb[hh * 16 + qq * 4 + 0] = b4.x; wb[hh * 16 + qq * 4 + 1] = b4.y;
          wb[hh * 16 + qq * 4 + 2] = b4.z; wb[hh * 16 + qq * 4 + 3] = b4.w;
        }
      }
      op.run(qr, wb, cs);
    }

    // 2. Reduce scores across 4 simdgroups
    for (int i = 0; i < 16; i++) {
      const uint r = m0 + 8 * ((i >> 2) & 1), c = n0 + (i & 3) + 16 * (i >> 3);
      Sp[sg][r][c] = cs[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s[16];
    for (int i = 0; i < 16; i++) {
      const uint r = m0 + 8 * ((i >> 2) & 1), c = n0 + (i & 3) + 16 * (i >> 3);
      const uint j = j0 + c;
      s[i] = (j < L) ? ((Sp[0][r][c] + Sp[1][r][c] + Sp[2][r][c] + Sp[3][r][c]) * a.scale) : -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Online softmax in registers
    float alpha[2];
    for (int r = 0; r < 2; r++) {
      float mb = -INFINITY;
      for (int i = 0; i < 16; i++) if (((i >> 2) & 1) == r) mb = max(mb, s[i]);
      mb = max(mb, simd_shuffle_xor(mb, 1));
      mb = max(mb, simd_shuffle_xor(mb, 8));
      const float mn = max(mx[r], mb);
      alpha[r] = exp(mx[r] - mn);
      float bs = 0.0f;
      for (int i = 0; i < 16; i++)
        if (((i >> 2) & 1) == r) { s[i] = exp(s[i] - mn); bs += s[i]; }
      bs += simd_shuffle_xor(bs, 1);
      bs += simd_shuffle_xor(bs, 8);
      lse[r] = lse[r] * alpha[r] + bs;
      mx[r] = mn;
    }
    for (int i = 0; i < 16; i++) {
      const float cc = alpha[(i >> 2) & 1];
      o0[i] *= cc; o1[i] *= cc; o2[i] *= cc; o3[i] *= cc;
    }

    // 4. Scale-folded (P * Ks) @ V_int8 directly from TG Kq (zero dequant multiply in V loop!)
    for (int i = 0; i < 16; i++) {
      const uint c = n0 + (i & 3) + 16 * (i >> 3);
      pa[i] = bfloat(s[i] * float(Ks[sg][c][0]));
    }
    for (int nc_i = 0; nc_i < 2; nc_i++) {
      const uint d_loc = nc_i * 32u + m0;
      for (int hh = 0; hh < 2; hh++) {
        for (int e = 0; e < 4; e++) {
          const uint jj = n0 + e + 16u * hh;
          for (int qq = 0; qq < 4; qq++)
            wb[hh * 16 + qq * 4 + e] = bfloat(float(Kq[sg][jj][d_loc + 8u * qq]));
        }
      }
      if (nc_i == 0) op.run(pa, wb, o0); else op.run(pa, wb, o1);
    }
    for (int i = 0; i < 16; i++) {
      const uint c = n0 + (i & 3) + 16 * (i >> 3);
      pa[i] = bfloat(s[i] * float(Ks[sg][c][1]));
    }
    for (int nc_i = 2; nc_i < 4; nc_i++) {
      const uint d_loc = nc_i * 32u + m0;
      for (int hh = 0; hh < 2; hh++) {
        for (int e = 0; e < 4; e++) {
          const uint jj = n0 + e + 16u * hh;
          for (int qq = 0; qq < 4; qq++)
            wb[hh * 16 + qq * 4 + e] = bfloat(float(Kq[sg][jj][d_loc + 8u * qq]));
        }
      }
      if (nc_i == 2) op.run(pa, wb, o2); else op.run(pa, wb, o3);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write normalized output directly into grouped bf16 frags [64, M, 512] for attn.vb!
  const float inv0 = 1.0f / lse[0], inv1 = 1.0f / lse[1];
  for (int nc_i = 0; nc_i < 4; nc_i++) {
    for (int i = 0; i < 16; i++) {
      const uint hh = m0 + 8 * ((i >> 2) & 1), d = sg * 128u + nc_i * 32u + n0 + (i & 3) + 16 * (i >> 3);
      const float v = (nc_i == 0) ? o0[i] : ((nc_i == 1) ? o1[i] : ((nc_i == 2) ? o2[i] : o3[i]));
      olat_frags[size_t(h0 + hh) * a.vb_gstride_bf16 + frag_off(m, d, 512u)] = bfloat(v * (((i >> 2) & 1) ? inv1 : inv0));
    }
  }
}

// 18d. Fused M=1 SiTU-GLU pack for all topk=10 experts in 1 dispatch (replaces 10 serial launches)
// Grid: (2048, topk), 256 threads. Each expert slot k writes to frag_b + k * 32768 bf16.
struct SituM1Args { uint I, topk, slab_bf16; float beta, lbeta; };
kernel void e5_situ_pack_m1(device const float* src [[buffer(0)]],
                            device bfloat* dst_frag [[buffer(1)]],
                            constant SituM1Args& a [[buffer(2)]],
                            uint2 g [[thread_position_in_grid]]) {
  const uint c = g.x, k = g.y;
  if (c >= a.I || k >= a.topk) return;
  const uint blk = c >> 4, off = c & 15u;
  device const float* row = src + size_t(k) * (2u * a.I);
  const float gv = row[blk * 32 + off];
  const float uv = row[blk * 32 + 16 + off];
  dst_frag[size_t(k) * a.slab_bf16 + frag_off(0u, c, a.I)] = bfloat(situ_act(gv, uv, a.beta, a.lbeta));
}

// 18e. FlashDecoding Split-K DSA Attention (S=8 splits for M <= 4)
// Grid: (4 head_groups, M, S), 128 threads. Writes unnormalized O [M, S, 64, 512] + ml [M, S, 64, 2] to part.
struct DsaSplitArgs { uint M, topk_win, qabs_gstride, vb_gstride_bf16, S; float scale; };
kernel void e5_dsa_attn_mma_split(device const float* q_abs [[buffer(0)]],
                                  device const float* q_rope [[buffer(1)]],
                                  device const uchar* lat_q8 [[buffer(2)]],
                                  device const int* sel [[buffer(3)]],
                                  device const uint* sel_len [[buffer(4)]],
                                  device const float* sink [[buffer(5)]],
                                  device float* part [[buffer(6)]],
                                  constant DsaSplitArgs& a [[buffer(7)]],
                                  uint3 tg [[threadgroup_position_in_grid]],
                                  ushort sg [[simdgroup_index_in_threadgroup]],
                                  ushort l [[thread_index_in_simdgroup]]) {
  threadgroup float Sp[4][16][32];
  threadgroup char Kq[4][32][128];
  threadgroup half Ks[4][32][2];
  const uint hg = tg.x, m = tg.y, sp = tg.z;
  const uint h0 = hg * 16u;
  const uint L = sel_len[m];
  const uint per = ((L + a.S - 1u) / a.S + 31u) & ~31u;
  const uint j_begin = sp * per;
  const uint j_end = min(L, j_begin + per);
  device const int* sm = sel + size_t(m) * a.topk_win;

  const ushort m0 = ((l >> 1) & 3) + 4 * (l >> 4), n0 = 4 * (l & 1) + 8 * ((l >> 3) & 1);
  const uint lr = l >> 3, lc = l & 7u;
  constexpr auto desc =
      matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
#define LEFT op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>()
  auto q0 = LEFT; auto q1 = LEFT; auto q2 = LEFT; auto q3 = LEFT; auto qr = LEFT; auto pa = LEFT;
#undef LEFT
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto cs = op.template get_destination_cooperative_tensor<decltype(pa), decltype(wb), float>();
  auto o0 = op.template get_destination_cooperative_tensor<decltype(pa), decltype(wb), float>();
  auto o1 = op.template get_destination_cooperative_tensor<decltype(pa), decltype(wb), float>();
  auto o2 = op.template get_destination_cooperative_tensor<decltype(pa), decltype(wb), float>();
  auto o3 = op.template get_destination_cooperative_tensor<decltype(pa), decltype(wb), float>();

  for (int i = 0; i < 16; i++) { o0[i] = 0.0f; o1[i] = 0.0f; o2[i] = 0.0f; o3[i] = 0.0f; }
  float mx[2] = { (sp == 0) ? sink[h0 + m0] : -INFINITY, (sp == 0) ? sink[h0 + m0 + 8] : -INFINITY };
  float lse[2] = { (sp == 0) ? 1.0f : 0.0f, (sp == 0) ? 1.0f : 0.0f };

  if (j_begin < j_end) {
    for (int ks = 0; ks < 4; ks++) {
      for (int i = 0; i < 16; i++) {
        const uint hh = m0 + 8 * ((i >> 2) & 1), d = sg * 128u + ks * 32u + n0 + (i & 3) + 16 * (i >> 3);
        const bfloat v = bfloat(q_abs[size_t(h0 + hh) * a.qabs_gstride + size_t(m) * 512u + d]);
        if (ks == 0) q0[i] = v; else if (ks == 1) q1[i] = v; else if (ks == 2) q2[i] = v; else q3[i] = v;
      }
    }
    if (sg < 2) {
      device const float* qrb = q_rope + (size_t(m) * 64u + h0) * 64u;
      for (int i = 0; i < 16; i++) {
        const uint hh = m0 + 8 * ((i >> 2) & 1), d = sg * 32u + n0 + (i & 3) + 16 * (i >> 3);
        qr[i] = bfloat(qrb[hh * 64u + d]);
      }
    }

    for (uint j0 = j_begin; j0 < j_end; j0 += 32u) {
      for (int i = 0; i < 16; i++) cs[i] = 0.0f;
      // Coalesced single load of 32 KV rows x 128 dims into per-simdgroup Kq/Ks
      for (uint r0 = 0; r0 < 32u; r0 += 4u) {
        const uint jj = r0 + lr;
        const uint j = j0 + jj;
        if (j < j_end) {
          device const uchar* lrow = lat_q8 + size_t(sm[j]) * 672u;
          *(threadgroup uint4*)(&Kq[sg][jj][lc * 16u]) = *(device const uint4*)(lrow + sg * 128u + lc * 16u);
          if (lc < 2u) Ks[sg][jj][lc] = ((device const half*)(lrow + 512u))[sg * 2u + lc];
        } else {
          *(threadgroup uint4*)(&Kq[sg][jj][lc * 16u]) = uint4(0);
          if (lc < 2u) Ks[sg][jj][lc] = half(0.0h);
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);

      for (int ks = 0; ks < 4; ks++) {
        const uint d_loc = ks * 32u + n0;
        const uint g_loc = ks >> 1;
        for (int qq = 0; qq < 4; qq++) {
          const uint jj = m0 + 8u * qq;
          const float ksc = float(Ks[sg][jj][g_loc]);
          for (int hh = 0; hh < 2; hh++) {
            const char4 c4 = *(threadgroup const char4*)(&Kq[sg][jj][d_loc + 16u * hh]);
            for (int e = 0; e < 4; e++) wb[hh * 16 + qq * 4 + e] = bfloat(float(c4[e]) * ksc);
          }
        }
        if (ks == 0) op.run(q0, wb, cs);
        else if (ks == 1) op.run(q1, wb, cs);
        else if (ks == 2) op.run(q2, wb, cs);
        else op.run(q3, wb, cs);
      }
      if (sg < 2) {
        const uint dr = sg * 32u + n0;
        for (int qq = 0; qq < 4; qq++) {
          const uint j = j0 + m0 + 8u * qq;
          device const bfloat* lr_p = (j < j_end) ? (device const bfloat*)(lat_q8 + size_t(sm[j]) * 672u + 528u) : nullptr;
          for (int hh = 0; hh < 2; hh++) {
            const bfloat4 b4 = lr_p ? *(device const bfloat4*)(lr_p + dr + 16 * hh) : bfloat4(bfloat(0.0f));
            for (int e = 0; e < 4; e++) wb[hh * 16 + qq * 4 + e] = b4[e];
          }
        }
        op.run(qr, wb, cs);
      }
      for (int i = 0; i < 16; i++) {
        const uint r = m0 + 8 * ((i >> 2) & 1), c = n0 + (i & 3) + 16 * (i >> 3);
        Sp[sg][r][c] = cs[i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float s[16];
      for (int i = 0; i < 16; i++) {
        const uint r = m0 + 8 * ((i >> 2) & 1), c = n0 + (i & 3) + 16 * (i >> 3);
        const uint j = j0 + c;
        s[i] = (j < j_end) ? ((Sp[0][r][c] + Sp[1][r][c] + Sp[2][r][c] + Sp[3][r][c]) * a.scale) : -INFINITY;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      float alpha[2];
      for (int r = 0; r < 2; r++) {
        float mb = -INFINITY;
        for (int i = 0; i < 16; i++) if (((i >> 2) & 1) == r) mb = max(mb, s[i]);
        mb = max(mb, simd_shuffle_xor(mb, 1));
        mb = max(mb, simd_shuffle_xor(mb, 8));
        const float mn = max(mx[r], mb);
        alpha[r] = (mx[r] == -INFINITY) ? 0.0f : exp(mx[r] - mn);
        float bs = 0.0f;
        for (int i = 0; i < 16; i++)
          if (((i >> 2) & 1) == r) { s[i] = (s[i] == -INFINITY) ? 0.0f : exp(s[i] - mn); bs += s[i]; }
        bs += simd_shuffle_xor(bs, 1);
        bs += simd_shuffle_xor(bs, 8);
        lse[r] = lse[r] * alpha[r] + bs;
        mx[r] = mn;
      }
      for (int i = 0; i < 16; i++) {
        const float cc = alpha[(i >> 2) & 1];
        o0[i] *= cc; o1[i] *= cc; o2[i] *= cc; o3[i] *= cc;
      }

      // Scale-folded (P * Ks) @ V_int8 (eliminates 128 dequant multiplies/thread!)
      for (int i = 0; i < 16; i++) {
        const uint c = n0 + (i & 3) + 16 * (i >> 3);
        pa[i] = bfloat(s[i] * float(Ks[sg][c][0]));
      }
      for (int nc_i = 0; nc_i < 2; nc_i++) {
        const uint d_loc = nc_i * 32u + m0;
        for (int hh = 0; hh < 2; hh++) {
          for (int e = 0; e < 4; e++) {
            const uint jj = n0 + e + 16u * hh;
            for (int qq = 0; qq < 4; qq++)
              wb[hh * 16 + qq * 4 + e] = bfloat(float(Kq[sg][jj][d_loc + 8u * qq]));
          }
        }
        if (nc_i == 0) op.run(pa, wb, o0); else op.run(pa, wb, o1);
      }
      for (int i = 0; i < 16; i++) {
        const uint c = n0 + (i & 3) + 16 * (i >> 3);
        pa[i] = bfloat(s[i] * float(Ks[sg][c][1]));
      }
      for (int nc_i = 2; nc_i < 4; nc_i++) {
        const uint d_loc = nc_i * 32u + m0;
        for (int hh = 0; hh < 2; hh++) {
          for (int e = 0; e < 4; e++) {
            const uint jj = n0 + e + 16u * hh;
            for (int qq = 0; qq < 4; qq++)
              wb[hh * 16 + qq * 4 + e] = bfloat(float(Kq[sg][jj][d_loc + 8u * qq]));
          }
        }
        if (nc_i == 2) op.run(pa, wb, o2); else op.run(pa, wb, o3);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // Write partial O [M, S, 64, 512] and ml [M, S, 64, 2]
  const size_t o_base = (size_t(m) * a.S + sp) * 64u * 512u;
  for (int nc_i = 0; nc_i < 4; nc_i++) {
    for (int i = 0; i < 16; i++) {
      const uint hh = m0 + 8 * ((i >> 2) & 1), d = sg * 128u + nc_i * 32u + n0 + (i & 3) + 16 * (i >> 3);
      const float v = (nc_i == 0) ? o0[i] : ((nc_i == 1) ? o1[i] : ((nc_i == 2) ? o2[i] : o3[i]));
      part[o_base + size_t(h0 + hh) * 512u + d] = v;
    }
  }
  if (sg == 0 && n0 == 0) {
    const size_t ml_base = size_t(a.M) * a.S * 64u * 512u + (size_t(m) * a.S + sp) * 64u * 2u;
    part[ml_base + (h0 + m0) * 2u + 0] = mx[0];
    part[ml_base + (h0 + m0) * 2u + 1] = lse[0];
    part[ml_base + (h0 + m0 + 8u) * 2u + 0] = mx[1];
    part[ml_base + (h0 + m0 + 8u) * 2u + 1] = lse[1];
  }
}

// Grid: (64 heads, M), 128 threads. Merges S splits from part -> olat_frags.
kernel void e5_dsa_attn_merge(device const float* part [[buffer(0)]],
                              device bfloat* olat_frags [[buffer(1)]],
                              constant DsaSplitArgs& a [[buffer(2)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              ushort tid [[thread_index_in_threadgroup]]) {
  const uint h = tg.x, m = tg.y;
  const size_t ml_off0 = size_t(a.M) * a.S * 64u * 512u;
  float mn = -INFINITY;
  for (uint sp = 0; sp < a.S; sp++) {
    const float mx = part[ml_off0 + (size_t(m) * a.S + sp) * 64u * 2u + h * 2u];
    mn = max(mn, mx);
  }
  float lse = 0.0f, o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
  for (uint sp = 0; sp < a.S; sp++) {
    const size_t mlp = ml_off0 + (size_t(m) * a.S + sp) * 64u * 2u + h * 2u;
    const float mx = part[mlp], ls = part[mlp + 1];
    if (ls <= 0.0f) continue;
    const float w = exp(mx - mn);
    lse = fma(w, ls, lse);
    device const float* op = part + (size_t(m) * a.S + sp) * 64u * 512u + size_t(h) * 512u;
    o0 = fma(w, op[tid], o0);
    o1 = fma(w, op[tid + 128u], o1);
    o2 = fma(w, op[tid + 256u], o2);
    o3 = fma(w, op[tid + 384u], o3);
  }
  const float inv = 1.0f / max(lse, 1e-30f);
  device bfloat* gdst = olat_frags + size_t(h) * a.vb_gstride_bf16;
  gdst[frag_off(m, tid, 512u)] = bfloat(o0 * inv);
  gdst[frag_off(m, tid + 128u, 512u)] = bfloat(o1 * inv);
  gdst[frag_off(m, tid + 256u, 512u)] = bfloat(o2 * inv);
  gdst[frag_off(m, tid + 384u, 512u)] = bfloat(o3 * inv);
}

// 18f-fast. 1-SG-per-TG Split-H x Split-Mix mHC for M <= 4 (ZERO TG barriers in part, ZERO TG barriers in fin!)
// Grid: (32 K-chunks, mode==0 ? 4 : 1, M), TG: (32, 1, 1) (1 simdgroup).
// Each K-chunk p=0..31 covers 128 channels (32 lanes x float4). Each mg=0..3 computes 6 mixes (4 for mode==1).
// Only mg==0 computes the 10 Gram terms and writes h_out (4x inject instead of 25x!).
kernel void e5_mhc_part_sg(device const float* h_in [[buffer(0)]],
                           device const float* y [[buffer(1)]],
                           device const float* postp [[buffer(2)]],
                           device const float* combp [[buffer(3)]],
                           device const float* fn [[buffer(4)]],
                           device float* hcpart [[buffer(5)]],
                           constant MhcArgs& a [[buffer(6)]],
                           device float* h_out [[buffer(7)]],
                           uint3 tg [[threadgroup_position_in_grid]],
                           ushort l [[thread_index_in_simdgroup]]) {
  const uint p = tg.x, mg = tg.y, m = tg.z;
  const uint H = a.H, W = 4u * H;
  const uint c4 = (p * 32u + l) << 2;
  device const float* hr = h_in + size_t(m) * W + c4;

  float4 v0, v1, v2, v3;
  if (a.inject) {
    device const float* po = postp + m * 4u;
    device const float* cb = combp + m * 16u;
    const float4 yc = *(device const float4*)(y + size_t(m) * H + c4);
    const float4 o0 = *(device const float4*)(hr + 0u * H);
    const float4 o1 = *(device const float4*)(hr + 1u * H);
    const float4 o2 = *(device const float4*)(hr + 2u * H);
    const float4 o3 = *(device const float4*)(hr + 3u * H);
    v0 = fma(cb[12], o3, fma(cb[8], o2, fma(cb[4], o1, fma(cb[0], o0, po[0] * yc))));
    v1 = fma(cb[13], o3, fma(cb[9], o2, fma(cb[5], o1, fma(cb[1], o0, po[1] * yc))));
    v2 = fma(cb[14], o3, fma(cb[10], o2, fma(cb[6], o1, fma(cb[2], o0, po[2] * yc))));
    v3 = fma(cb[15], o3, fma(cb[11], o2, fma(cb[7], o1, fma(cb[3], o0, po[3] * yc))));
    if (mg == 0u) {
      device float* hw = h_out + size_t(m) * W + c4;
      *(device float4*)(hw + 0u * H) = v0;
      *(device float4*)(hw + 1u * H) = v1;
      *(device float4*)(hw + 2u * H) = v2;
      *(device float4*)(hw + 3u * H) = v3;
    }
  } else {
    v0 = *(device const float4*)(hr + 0u * H);
    v1 = *(device const float4*)(hr + 1u * H);
    v2 = *(device const float4*)(hr + 2u * H);
    v3 = *(device const float4*)(hr + 3u * H);
  }

  device float* out = hcpart + (size_t(m) * a.npart + p) * 34u;
  const uint j0 = (a.mode == 0) ? (mg * 6u) : 0u;
  const uint j1 = (a.mode == 0) ? (j0 + 6u) : 4u;
  for (uint j = j0; j < j1; j++) {
    device const float* fj = fn + size_t(j) * W + c4;
    float s = dot(*(device const float4*)(fj + 0u * H), v0) +
              dot(*(device const float4*)(fj + 1u * H), v1) +
              dot(*(device const float4*)(fj + 2u * H), v2) +
              dot(*(device const float4*)(fj + 3u * H), v3);
    s = simd_sum(s);
    if (l == 0u) out[j] = s;
  }
  if (mg == 0u) {
    const float4 vs[4] = {v0, v1, v2, v3};
    uint q = 0u;
    for (uint s = 0; s < 4u; s++) {
      for (uint u = s; u < 4u; u++) {
        float t = simd_sum(dot(vs[s], vs[u]));
        if (l == 0u) out[24u + q] = t;
        q++;
      }
    }
  }
}

// e5_mhc_fin_sg: Grid (32, M), 32 threads (1 simdgroup, ZERO TG barriers!).
// Reduces 32 partials, runs 16-lane SIMD-parallel Sinkhorn, writes post/comb (p==0) and xfrag.
kernel void e5_mhc_fin_sg(device const float* h [[buffer(0)]],
                          device const float* hcpart [[buffer(1)]],
                          device const float* base [[buffer(2)]],
                          device const float* scale [[buffer(3)]],
                          device const bfloat* nw [[buffer(4)]],
                          device bfloat* xfrag [[buffer(5)]],
                          device float* post_out [[buffer(6)]],
                          device float* comb_out [[buffer(7)]],
                          constant MhcArgs& a [[buffer(8)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          ushort l [[thread_index_in_simdgroup]]) {
  threadgroup float tot[34];
  const uint p = tg.x, m = tg.y;
  const uint H = a.H, W = 4u * H;
  const uint nmix = (a.mode == 0) ? 24u : 4u;

  {
    float s = 0.0f;
    if (l < nmix || l >= 24u) {
      for (uint k = 0; k < a.npart; k++) s += hcpart[(size_t(m) * a.npart + k) * 34u + l];
    }
    tot[l] = s;
    if (l < 2u) {
      float s2 = 0.0f;
      for (uint k = 0; k < a.npart; k++) s2 += hcpart[(size_t(m) * a.npart + k) * 34u + 32u + l];
      tot[32u + l] = s2;
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  if (p == 0 && a.mode == 0) {
    const float mean_sq = (tot[24] + tot[28] + tot[31] + tot[33]) / float(W);
    const float r = 1.0f / (sqrt(mean_sq) + a.eps);
    const float sr = scale[2];
    float val = (l < 16u) ? (tot[8u + l] * r * sr + base[8u + l]) : 0.0f;
    float mx = max(val, simd_shuffle_xor(val, 1));
    mx = max(mx, simd_shuffle_xor(mx, 2));
    val = exp(val - mx);
    float sm = val + simd_shuffle_xor(val, 1);
    sm += simd_shuffle_xor(sm, 2);
    val = val / sm + a.hc_eps;
    float cs = val + simd_shuffle_xor(val, 4);
    cs += simd_shuffle_xor(cs, 8);
    val /= (cs + a.hc_eps);
    for (uint it = 1; it < a.sk_iters; it++) {
      float rs = val + simd_shuffle_xor(val, 1);
      rs += simd_shuffle_xor(rs, 2);
      val /= (rs + a.hc_eps);
      cs = val + simd_shuffle_xor(val, 4);
      cs += simd_shuffle_xor(cs, 8);
      val /= (cs + a.hc_eps);
    }
    if (l < 16u) comb_out[m * 16u + l] = val;
  }

  const float mean_sq = (tot[24] + tot[28] + tot[31] + tot[33]) / float(W);
  float pre0, pre1, pre2, pre3;
  if (a.mode == 0) {
    const float r = 1.0f / (sqrt(mean_sq) + a.eps);
    const float sp = scale[0];
    pre0 = sigm(tot[0] * r * sp + base[0]) + a.hc_eps;
    pre1 = sigm(tot[1] * r * sp + base[1]) + a.hc_eps;
    pre2 = sigm(tot[2] * r * sp + base[2]) + a.hc_eps;
    pre3 = sigm(tot[3] * r * sp + base[3]) + a.hc_eps;
    if (p == 0 && l < 4u) {
      post_out[m * 4u + l] = 2.0f * sigm(tot[4u + l] * r * scale[1] + base[4u + l]);
    }
  } else {
    const float r = rsqrt(mean_sq + a.eps);
    const float sp = scale[0];
    pre0 = sigm(tot[0] * r * sp + base[0]) + a.hc_eps;
    pre1 = sigm(tot[1] * r * sp + base[1]) + a.hc_eps;
    pre2 = sigm(tot[2] * r * sp + base[2]) + a.hc_eps;
    pre3 = sigm(tot[3] * r * sp + base[3]) + a.hc_eps;
  }
  const float ss =
      pre0 * (pre0 * tot[24] + 2.0f * (pre1 * tot[25] + pre2 * tot[26] + pre3 * tot[27])) +
      pre1 * (pre1 * tot[28] + 2.0f * (pre2 * tot[29] + pre3 * tot[30])) +
      pre2 * (pre2 * tot[31] + 2.0f * (pre3 * tot[32])) +
      pre3 * pre3 * tot[33];
  const float inv_r = rsqrt(ss / float(H) + a.eps);

  const uint c4 = (p * 32u + l) << 2;
  device const float* hr = h + size_t(m) * W + c4;
  float4 x = pre0 * *(device const float4*)(hr + 0u * H) +
             pre1 * *(device const float4*)(hr + 1u * H) +
             pre2 * *(device const float4*)(hr + 2u * H) +
             pre3 * *(device const float4*)(hr + 3u * H);
  const bfloat4 n4 = *(device const bfloat4*)(nw + c4);
  const float4 v = x * inv_r * float4(n4);
  *(device bfloat4*)(xfrag + frag_off(m, c4, H)) = bfloat4(bfloat(v.x), bfloat(v.y), bfloat(v.z), bfloat(v.w));
}

// 18f. 200-TG Parallel Split-H x Split-Mix mHC for M <= 4 (8 K-chunks x (nmix + 1) TGs per row!)
// Each TG (128 threads x float4 = 512 channels, 8 K-chunks = 4096 = H) computes ONE mix j (tg.y < nmix)
// or the 10 Gram terms + h_out write (tg.y == nmix). Double-buffered h_in -> h_out when inject == 1 (zero race!).
// 100% DRAM bandwidth across all 64 cores, npart = 8 (8x fewer partials in e5_mhc_fin!).
kernel void e5_mhc_part(device const float* h_in [[buffer(0)]],
                        device const float* y [[buffer(1)]],
                        device const float* postp [[buffer(2)]],
                        device const float* combp [[buffer(3)]],
                        device const float* fn [[buffer(4)]],
                        device float* hcpart [[buffer(5)]],
                        constant MhcArgs& a [[buffer(6)]],
                        device float* h_out [[buffer(7)]],
                        uint3 tg [[threadgroup_position_in_grid]],
                        ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup float sh[4];
  // 2-mix-per-TG high-occupancy mHC part: Grid (8, nmix/2, M) x 128 threads (96 TGs = 384 SGs = 6 SGs/core!)
  // Halves h_in/y reads and inject work vs 25 TGs, while keeping full DRAM latency hiding!
  const uint p = tg.x, mg = tg.y, m = tg.z;
  const uint H = a.H, W = 4u * H;
  const uint c4 = (p * 128u + tid) << 2;
  device const float* hr = h_in + size_t(m) * W + c4;

  float4 v0, v1, v2, v3;
  if (a.inject) {
    device const float* po = postp + m * 4u;
    device const float* cb = combp + m * 16u;
    const float4 yc = *(device const float4*)(y + size_t(m) * H + c4);
    const float4 o0 = *(device const float4*)(hr + 0u * H);
    const float4 o1 = *(device const float4*)(hr + 1u * H);
    const float4 o2 = *(device const float4*)(hr + 2u * H);
    const float4 o3 = *(device const float4*)(hr + 3u * H);
    v0 = fma(cb[12], o3, fma(cb[8], o2, fma(cb[4], o1, fma(cb[0], o0, po[0] * yc))));
    v1 = fma(cb[13], o3, fma(cb[9], o2, fma(cb[5], o1, fma(cb[1], o0, po[1] * yc))));
    v2 = fma(cb[14], o3, fma(cb[10], o2, fma(cb[6], o1, fma(cb[2], o0, po[2] * yc))));
    v3 = fma(cb[15], o3, fma(cb[11], o2, fma(cb[7], o1, fma(cb[3], o0, po[3] * yc))));
    if (mg == 0u) {
      device float* hw = h_out + size_t(m) * W + c4;
      *(device float4*)(hw + 0u * H) = v0;
      *(device float4*)(hw + 1u * H) = v1;
      *(device float4*)(hw + 2u * H) = v2;
      *(device float4*)(hw + 3u * H) = v3;
    }
  } else {
    v0 = *(device const float4*)(hr + 0u * H);
    v1 = *(device const float4*)(hr + 1u * H);
    v2 = *(device const float4*)(hr + 2u * H);
    v3 = *(device const float4*)(hr + 3u * H);
  }

  const ushort sgi = tid >> 5, lane = tid & 31u;
  device float* out = hcpart + (size_t(m) * a.npart + p) * 34u;
  const uint mpt = (a.mode == 0u) ? 3u : 2u;
  const uint j0 = mg * mpt;
  device const float* fj0 = fn + size_t(j0) * W + c4;
  device const float* fj1 = fn + size_t(j0 + 1u) * W + c4;
  float s0 = dot(*(device const float4*)(fj0 + 0u * H), v0) +
             dot(*(device const float4*)(fj0 + 1u * H), v1) +
             dot(*(device const float4*)(fj0 + 2u * H), v2) +
             dot(*(device const float4*)(fj0 + 3u * H), v3);
  float s1 = dot(*(device const float4*)(fj1 + 0u * H), v0) +
             dot(*(device const float4*)(fj1 + 1u * H), v1) +
             dot(*(device const float4*)(fj1 + 2u * H), v2) +
             dot(*(device const float4*)(fj1 + 3u * H), v3);
  s0 = simd_sum(s0);
  s1 = simd_sum(s1);
  threadgroup float sh3[3][4];
  if (lane == 0) { sh3[0][sgi] = s0; sh3[1][sgi] = s1; }
  if (a.mode == 0u) {
    device const float* fj2 = fn + size_t(j0 + 2u) * W + c4;
    float s2 = dot(*(device const float4*)(fj2 + 0u * H), v0) +
               dot(*(device const float4*)(fj2 + 1u * H), v1) +
               dot(*(device const float4*)(fj2 + 2u * H), v2) +
               dot(*(device const float4*)(fj2 + 3u * H), v3);
    s2 = simd_sum(s2);
    if (lane == 0) sh3[2][sgi] = s2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < mpt) out[j0 + tid] = sh3[tid][0] + sh3[tid][1] + sh3[tid][2] + sh3[tid][3];
  if (mg == 0u) {
    threadgroup float shg[10][4];
    const float4 vs[4] = {v0, v1, v2, v3};
    uint q = 0u;
    for (uint s = 0; s < 4u; s++) {
      for (uint u = s; u < 4u; u++) {
        float t = simd_sum(dot(vs[s], vs[u]));
        if (lane == 0) shg[q][sgi] = t;
        q++;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 10u) {
      out[24u + tid] = shg[tid][0] + shg[tid][1] + shg[tid][2] + shg[tid][3];
    }
  }
}

// e5_mhc_fin: Grid (16, M), 64 threads (2 simdgroups, each thread handles float4 = 4 channels -> 256 channels/TG).
// Reduces a.npart partials, runs 16-lane SIMD-parallel Sinkhorn in SG0, writes post/comb (p==0) and xfrag.
kernel void e5_mhc_fin(device const float* h [[buffer(0)]],
                       device const float* hcpart [[buffer(1)]],
                       device const float* base [[buffer(2)]],
                       device const float* scale [[buffer(3)]],
                       device const bfloat* nw [[buffer(4)]],
                       device bfloat* xfrag [[buffer(5)]],
                       device float* xf [[buffer(6)]],
                       device float* post_out [[buffer(7)]],
                       device float* comb_out [[buffer(8)]],
                       constant MhcArgs& a [[buffer(9)]],
                       uint2 tg [[threadgroup_position_in_grid]],
                       ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup float tot[34];
  threadgroup float pre_sh[4];
  threadgroup float inv_r_sh;
  const uint p = tg.x, m = tg.y;
  const uint H = a.H, W = 4u * H;
  const uint nmix = (a.mode == 0) ? 24u : 4u;

  if (tid < 34u) {
    float s = 0.0f;
    if (tid < nmix || tid >= 24u) {
      for (uint k = 0; k < a.npart; k++) s += hcpart[(size_t(m) * a.npart + k) * 34u + tid];
    }
    tot[tid] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 16-lane SIMD-parallel Sinkhorn in simdgroup 0 (p == 0, mode == 0)
  if (p == 0 && a.mode == 0 && tid < 32u) {
    const float mean_sq = (tot[24] + tot[28] + tot[31] + tot[33]) / float(W);
    const float r = 1.0f / (sqrt(mean_sq) + a.eps);
    const float sr = scale[2];
    float val = (tid < 16u) ? (tot[8u + tid] * r * sr + base[8u + tid]) : 0.0f;
    float mx = max(val, simd_shuffle_xor(val, 1));
    mx = max(mx, simd_shuffle_xor(mx, 2));
    val = exp(val - mx);
    float sm = val + simd_shuffle_xor(val, 1);
    sm += simd_shuffle_xor(sm, 2);
    val = val / sm + a.hc_eps;
    float cs = val + simd_shuffle_xor(val, 4);
    cs += simd_shuffle_xor(cs, 8);
    val /= (cs + a.hc_eps);
    for (uint it = 1; it < a.sk_iters; it++) {
      float rs = val + simd_shuffle_xor(val, 1);
      rs += simd_shuffle_xor(rs, 2);
      val /= (rs + a.hc_eps);
      cs = val + simd_shuffle_xor(val, 4);
      cs += simd_shuffle_xor(cs, 8);
      val /= (cs + a.hc_eps);
    }
    if (tid < 16u) comb_out[m * 16u + tid] = val;
  }

  if (tid == 0) {
    const float mean_sq = (tot[24] + tot[28] + tot[31] + tot[33]) / float(W);
    float pre[4];
    if (a.mode == 0) {
      const float r = 1.0f / (sqrt(mean_sq) + a.eps);
      const float sp = scale[0], spo = scale[1];
      for (uint j = 0; j < 4; j++) {
        pre[j] = sigm(tot[j] * r * sp + base[j]) + a.hc_eps;
        pre_sh[j] = pre[j];
        if (p == 0) post_out[m * 4u + j] = 2.0f * sigm(tot[4 + j] * r * spo + base[4 + j]);
      }
    } else {
      const float r = rsqrt(mean_sq + a.eps);
      const float sp = scale[0];
      for (uint j = 0; j < 4; j++) {
        pre[j] = sigm(tot[j] * r * sp + base[j]) + a.hc_eps;
        pre_sh[j] = pre[j];
      }
    }
    const float G[4][4] = {
      {tot[24], tot[25], tot[26], tot[27]},
      {tot[25], tot[28], tot[29], tot[30]},
      {tot[26], tot[29], tot[31], tot[32]},
      {tot[27], tot[30], tot[32], tot[33]},
    };
    float ss = 0.0f;
    for (uint i = 0; i < 4; i++)
      for (uint j = 0; j < 4; j++) ss += pre[i] * pre[j] * G[i][j];
    inv_r_sh = rsqrt(ss / float(H) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint c4 = (p * 64u + tid) << 2;
  device const float* hr = h + size_t(m) * W + c4;
  float4 x = pre_sh[0] * *(device const float4*)(hr + 0u * H) +
             pre_sh[1] * *(device const float4*)(hr + 1u * H) +
             pre_sh[2] * *(device const float4*)(hr + 2u * H) +
             pre_sh[3] * *(device const float4*)(hr + 3u * H);
  const bfloat4 n4 = *(device const bfloat4*)(nw + c4);
  const float4 v = x * inv_r_sh * float4(n4);
  *(device bfloat4*)(xfrag + frag_off(m, c4, H)) = bfloat4(bfloat(v.x), bfloat(v.y), bfloat(v.z), bfloat(v.w));
}

// 18g. FlashDecoding Split-K HCA Attention (S=4 splits for M <= 4)
// Grid: (4 head_groups, M, S), 128 threads. Writes unnormalized O [M, S, 64, 512] + ml [M, S, 64, 2] to part.
struct HcaSplitArgs { uint M, p0, nh, hd, cr, win, ring, S; float scale; };
kernel void e5_hca_attn_mma_split(device const bfloat* q [[buffer(0)]],
                                  device const bfloat* comp_cache [[buffer(1)]],
                                  device const bfloat* swa_kv [[buffer(2)]],
                                  device const float* sink [[buffer(3)]],
                                  device float* part [[buffer(4)]],
                                  constant HcaSplitArgs& a [[buffer(5)]],
                                  uint3 tg [[threadgroup_position_in_grid]],
                                  ushort sg [[simdgroup_index_in_threadgroup]],
                                  ushort l [[thread_index_in_simdgroup]]) {
  threadgroup float Sp[4][16][32];
  threadgroup bfloat Vt[4][32][32];
  const uint hg = tg.x, m = tg.y, sp = tg.z, pos = a.p0 + m;
  const uint h0 = hg * 16u;
  const uint nc = (pos + 1) / a.cr;
  const uint swa_s = (pos + 1 > a.win) ? (pos + 1 - a.win) : 0u;
  const uint n_swa = pos + 1 - swa_s;
  const uint n_tot = nc + n_swa;
  const uint per = ((n_tot + a.S - 1u) / a.S + 31u) & ~31u;
  const uint j_begin = sp * per;
  const uint j_end = min(n_tot, j_begin + per);

  const ushort m0 = ((l >> 1) & 3) + 4 * (l >> 4), n0 = 4 * (l & 1) + 8 * ((l >> 3) & 1);
  const uint vlr = l >> 3, vlc = l & 7u;
  constexpr auto desc =
      matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
#define LEFT op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>()
  auto q0 = LEFT; auto q1 = LEFT; auto q2 = LEFT; auto q3 = LEFT; auto pa = LEFT;
#undef LEFT
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto cs = op.template get_destination_cooperative_tensor<decltype(pa), decltype(wb), float>();
  auto o0 = op.template get_destination_cooperative_tensor<decltype(pa), decltype(wb), float>();
  auto o1 = op.template get_destination_cooperative_tensor<decltype(pa), decltype(wb), float>();
  auto o2 = op.template get_destination_cooperative_tensor<decltype(pa), decltype(wb), float>();
  auto o3 = op.template get_destination_cooperative_tensor<decltype(pa), decltype(wb), float>();

  for (int i = 0; i < 16; i++) { o0[i] = 0.0f; o1[i] = 0.0f; o2[i] = 0.0f; o3[i] = 0.0f; }
  float mx[2] = { (sp == 0) ? sink[h0 + m0] : -INFINITY, (sp == 0) ? sink[h0 + m0 + 8] : -INFINITY };
  float lse[2] = { (sp == 0) ? 1.0f : 0.0f, (sp == 0) ? 1.0f : 0.0f };

  auto kv_ptr = [&](uint j) -> device const bfloat* {
    return (j < nc) ? (comp_cache + size_t(j) * 512u)
                    : (swa_kv + size_t((swa_s + (j - nc)) % a.ring) * 512u);
  };

  if (j_begin < j_end) {
    device const bfloat* qbase = q + (size_t(m) * a.nh + h0) * 512u;
    for (int ks = 0; ks < 4; ks++) {
      for (int i = 0; i < 16; i++) {
        const uint hh = m0 + 8 * ((i >> 2) & 1), d = sg * 128u + ks * 32u + n0 + (i & 3) + 16 * (i >> 3);
        const bfloat v = qbase[hh * 512u + d];
        if (ks == 0) q0[i] = v; else if (ks == 1) q1[i] = v; else if (ks == 2) q2[i] = v; else q3[i] = v;
      }
    }
    for (uint j0 = j_begin; j0 < j_end; j0 += 32u) {
      for (int i = 0; i < 16; i++) cs[i] = 0.0f;
      device const bfloat* kp[4];
      for (int qq = 0; qq < 4; qq++) {
        const uint j = j0 + m0 + 8 * qq;
        kp[qq] = (j < j_end) ? kv_ptr(j) : nullptr;
      }
      for (int ks = 0; ks < 4; ks++) {
        const uint d = sg * 128u + ks * 32u + n0;
        for (int qq = 0; qq < 4; qq++) {
          for (int hh = 0; hh < 2; hh++) {
            const bfloat4 b4 = kp[qq] ? *(device const bfloat4*)(kp[qq] + d + 16 * hh) : bfloat4(bfloat(0.0f));
            for (int e = 0; e < 4; e++) wb[hh * 16 + qq * 4 + e] = b4[e];
          }
        }
        if (ks == 0) op.run(q0, wb, cs);
        else if (ks == 1) op.run(q1, wb, cs);
        else if (ks == 2) op.run(q2, wb, cs);
        else op.run(q3, wb, cs);
      }
      for (int i = 0; i < 16; i++) {
        const uint r = m0 + 8 * ((i >> 2) & 1), c = n0 + (i & 3) + 16 * (i >> 3);
        Sp[sg][r][c] = cs[i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float s[16];
      for (int i = 0; i < 16; i++) {
        const uint r = m0 + 8 * ((i >> 2) & 1), c = n0 + (i & 3) + 16 * (i >> 3);
        const uint j = j0 + c;
        s[i] = (j < j_end) ? ((Sp[0][r][c] + Sp[1][r][c] + Sp[2][r][c] + Sp[3][r][c]) * a.scale) : -INFINITY;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      float alpha[2];
      for (int r = 0; r < 2; r++) {
        float mb = -INFINITY;
        for (int i = 0; i < 16; i++) if (((i >> 2) & 1) == r) mb = max(mb, s[i]);
        mb = max(mb, simd_shuffle_xor(mb, 1));
        mb = max(mb, simd_shuffle_xor(mb, 8));
        const float mn = max(mx[r], mb);
        alpha[r] = (mx[r] == -INFINITY) ? 0.0f : exp(mx[r] - mn);
        float bs = 0.0f;
        for (int i = 0; i < 16; i++)
          if (((i >> 2) & 1) == r) { s[i] = (s[i] == -INFINITY) ? 0.0f : exp(s[i] - mn); bs += s[i]; }
        bs += simd_shuffle_xor(bs, 1);
        bs += simd_shuffle_xor(bs, 8);
        lse[r] = lse[r] * alpha[r] + bs;
        mx[r] = mn;
      }
      for (int i = 0; i < 16; i++) {
        const float cc = alpha[(i >> 2) & 1];
        o0[i] *= cc; o1[i] *= cc; o2[i] *= cc; o3[i] *= cc;
        pa[i] = bfloat(s[i]);
      }

      for (int nc_i = 0; nc_i < 4; nc_i++) {
        const uint d_base = sg * 128u + nc_i * 32u;
        for (uint r0 = 0; r0 < 32u; r0 += 4u) {
          const uint jj = r0 + vlr;
          const uint j = j0 + jj;
          *(threadgroup bfloat4*)(&Vt[sg][jj][vlc * 4u]) =
              (j < j_end) ? *(device const bfloat4*)(kv_ptr(j) + d_base + vlc * 4u) : bfloat4(bfloat(0.0f));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int hh = 0; hh < 2; hh++) {
          for (int e = 0; e < 4; e++) {
            const uint jj = n0 + e + 16u * hh;
            for (int qq = 0; qq < 4; qq++)
              wb[hh * 16 + qq * 4 + e] = Vt[sg][jj][m0 + 8u * qq];
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (nc_i == 0) op.run(pa, wb, o0);
        else if (nc_i == 1) op.run(pa, wb, o1);
        else if (nc_i == 2) op.run(pa, wb, o2);
        else op.run(pa, wb, o3);
      }
    }
  }

  const size_t o_base = (size_t(m) * a.S + sp) * 64u * 512u;
  for (int nc_i = 0; nc_i < 4; nc_i++) {
    for (int i = 0; i < 16; i++) {
      const uint hh = m0 + 8 * ((i >> 2) & 1), d = sg * 128u + nc_i * 32u + n0 + (i & 3) + 16 * (i >> 3);
      const float v = (nc_i == 0) ? o0[i] : ((nc_i == 1) ? o1[i] : ((nc_i == 2) ? o2[i] : o3[i]));
      part[o_base + size_t(h0 + hh) * 512u + d] = v;
    }
  }
  if (sg == 0 && n0 == 0) {
    const size_t ml_base = size_t(a.M) * a.S * 64u * 512u + (size_t(m) * a.S + sp) * 64u * 2u;
    part[ml_base + (h0 + m0) * 2u + 0] = mx[0];
    part[ml_base + (h0 + m0) * 2u + 1] = lse[0];
    part[ml_base + (h0 + m0 + 8u) * 2u + 0] = mx[1];
    part[ml_base + (h0 + m0 + 8u) * 2u + 1] = lse[1];
  }
}

// Grid: (64 heads, M), 128 threads. Merges S splits from part -> o_out [M, 64, 512] bf16.
kernel void e5_hca_attn_merge(device const float* part [[buffer(0)]],
                              device bfloat* o_out [[buffer(1)]],
                              constant HcaSplitArgs& a [[buffer(2)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              ushort tid [[thread_index_in_threadgroup]]) {
  const uint h = tg.x, m = tg.y;
  const size_t ml_off0 = size_t(a.M) * a.S * 64u * 512u;
  float mn = -INFINITY;
  for (uint sp = 0; sp < a.S; sp++) {
    const float mx = part[ml_off0 + (size_t(m) * a.S + sp) * 64u * 2u + h * 2u];
    mn = max(mn, mx);
  }
  float lse = 0.0f, o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
  for (uint sp = 0; sp < a.S; sp++) {
    const size_t mlp = ml_off0 + (size_t(m) * a.S + sp) * 64u * 2u + h * 2u;
    const float mx = part[mlp], ls = part[mlp + 1];
    if (ls <= 0.0f) continue;
    const float w = exp(mx - mn);
    lse = fma(w, ls, lse);
    device const float* op = part + (size_t(m) * a.S + sp) * 64u * 512u + size_t(h) * 512u;
    o0 = fma(w, op[tid], o0);
    o1 = fma(w, op[tid + 128u], o1);
    o2 = fma(w, op[tid + 256u], o2);
    o3 = fma(w, op[tid + 384u], o3);
  }
  const float inv = 1.0f / max(lse, 1e-30f);
  device bfloat* dst = o_out + (size_t(m) * a.nh + h) * 512u;
  dst[tid] = bfloat(o0 * inv);
  dst[tid + 128u] = bfloat(o1 * inv);
  dst[tid + 256u] = bfloat(o2 * inv);
  dst[tid + 384u] = bfloat(o3 * inv);
}

// ==================================================================
// 19. Prefill-Optimized Tiled mHC & matmul2d VHA Kernels
// ==================================================================

// 19a. Vectorized float4 in-place mHC inject: h[m, s, c] = post[m, s]*y[m, c] + sum_i comb[m, i, s]*h[m, i, c]
// Grid: threads (H/4, M), TG (256, 1, 1).
kernel void e5_mhc_pre_inject(device float* h [[buffer(0)]],
                              device const float* y [[buffer(1)]],
                              device const float* postp [[buffer(2)]],
                              device const float* combp [[buffer(3)]],
                              constant MhcArgs& a [[buffer(4)]],
                              uint2 g [[thread_position_in_grid]]) {
  const uint H4 = a.H >> 2;
  if (g.x >= H4 || g.y >= a.M) return;
  const uint c4 = g.x << 2, m = g.y, H = a.H, W = 4u * H;
  device float* hr = h + size_t(m) * W + c4;
  const float4 yc = *(device const float4*)(y + size_t(m) * H + c4);
  const float4 old0 = *(device const float4*)(hr + 0u * H);
  const float4 old1 = *(device const float4*)(hr + 1u * H);
  const float4 old2 = *(device const float4*)(hr + 2u * H);
  const float4 old3 = *(device const float4*)(hr + 3u * H);
  device const float* po = postp + m * 4u;
  device const float* cb = combp + m * 16u;
  for (uint j = 0; j < 4; j++) {
    float4 acc = po[j] * yc;
    acc = fma(cb[0u * 4u + j], old0, acc);
    acc = fma(cb[1u * 4u + j], old1, acc);
    acc = fma(cb[2u * 4u + j], old2, acc);
    acc = fma(cb[3u * 4u + j], old3, acc);
    *(device float4*)(hr + j * H) = acc;
  }
}

// 19b. Fused Inject + Gram + matmul2d Prefill mHC (16 tokens/TG, 16 K-slices of 256 channels, 4 simdgroups = 4 streams!)
// Eliminates e5_mhc_pre_inject dispatch + second h DRAM read for Gram terms (~11.7 GB saved per chunk!).
// Grid: (16 K-slices, ceil(M / 16)), TG: (128, 1, 1) (4 simdgroups). Writes hcpart[m, p, 34] (p=0..15).
kernel void e5_mhc_pre_mma(device float* h [[buffer(0)]],
                           device const float* fn [[buffer(1)]],
                           device float* hcpart [[buffer(2)]],
                           constant MhcArgs& a [[buffer(3)]],
                           device const float* y [[buffer(4)]],
                           device const float* postp [[buffer(5)]],
                           device const float* combp [[buffer(6)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort l [[thread_index_in_simdgroup]]) {
  threadgroup float Sp[4][16][32];
  const uint p = tg.x, m0 = tg.y * 16u;
  const uint m_cnt = (m0 < a.M) ? min(16u, a.M - m0) : 0u;
  if (m_cnt == 0u) return;
  const uint H = a.H, W = 4u * H;
  const uint c0 = p * 256u;
  const uint nmix = (a.mode == 0) ? 24u : 4u;
  const ushort tid = sg * 32u + l;

  // Step 1: Fused in-place inject (if a.inject) + 10 Gram terms in one pass over h[m0..m0+m_cnt-1, c0..c0+255]!
  const uint c2 = c0 + uint(tid) * 2u;
  for (uint bm = 0; bm < m_cnt; bm++) {
    const uint m = m0 + bm;
    device float* hr = h + size_t(m) * W + c2;
    float2 vs[4];
    if (a.inject) {
      const float2 yc = *(device const float2*)(y + size_t(m) * H + c2);
      const float2 o0 = *(device const float2*)(hr + 0u * H);
      const float2 o1 = *(device const float2*)(hr + 1u * H);
      const float2 o2 = *(device const float2*)(hr + 2u * H);
      const float2 o3 = *(device const float2*)(hr + 3u * H);
      device const float* po = postp + m * 4u;
      device const float* cb = combp + m * 16u;
      vs[0] = fma(cb[12], o3, fma(cb[8], o2, fma(cb[4], o1, fma(cb[0], o0, po[0] * yc))));
      vs[1] = fma(cb[13], o3, fma(cb[9], o2, fma(cb[5], o1, fma(cb[1], o0, po[1] * yc))));
      vs[2] = fma(cb[14], o3, fma(cb[10], o2, fma(cb[6], o1, fma(cb[2], o0, po[2] * yc))));
      vs[3] = fma(cb[15], o3, fma(cb[11], o2, fma(cb[7], o1, fma(cb[3], o0, po[3] * yc))));
      *(device float2*)(hr + 0u * H) = vs[0];
      *(device float2*)(hr + 1u * H) = vs[1];
      *(device float2*)(hr + 2u * H) = vs[2];
      *(device float2*)(hr + 3u * H) = vs[3];
    } else {
      vs[0] = *(device const float2*)(hr + 0u * H);
      vs[1] = *(device const float2*)(hr + 1u * H);
      vs[2] = *(device const float2*)(hr + 2u * H);
      vs[3] = *(device const float2*)(hr + 3u * H);
    }
    uint q = 0u;
    for (uint s = 0; s < 4u; s++) {
      for (uint u = s; u < 4u; u++) {
        float t = simd_sum(dot(vs[s], vs[u]));
        if (l == 0) Sp[sg][bm][q] = t;
        q++;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint idx = tid; idx < m_cnt * 10u; idx += 128u) {
    const uint bm = idx / 10u, q = idx % 10u;
    hcpart[(size_t(m0 + bm) * a.npart + p) * 34u + 24u + q] = Sp[0][bm][q] + Sp[1][bm][q] + Sp[2][bm][q] + Sp[3][bm][q];
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

  // Step 2: matmul2d fn @ h (reads freshly updated h from L1/L2 cache!)
  const ushort rm0 = ((l >> 1) & 3) + 4 * (l >> 4), cn0 = 4 * (l & 1) + 8 * ((l >> 3) & 1);
  constexpr auto desc =
      matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto ha = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto cs = op.template get_destination_cooperative_tensor<decltype(ha), decltype(wb), float>();
  for (int i = 0; i < 16; i++) cs[i] = 0.0f;

  // Each simdgroup sg handles stream sg (256 contiguous channels = 8 k-steps of 32)
  for (uint ks = 0; ks < 8u; ks++) {
    const uint ck = c0 + ks * 32u;
    for (int i = 0; i < 16; i++) {
      const uint bm = rm0 + 8u * ((i >> 2) & 1u);
      const uint cc = ck + cn0 + (i & 3u) + 16u * (i >> 3u);
      ha[i] = (bm < m_cnt) ? bfloat(h[size_t(m0 + bm) * W + size_t(sg) * H + cc]) : bfloat(0.0f);
    }
    for (int qq = 0; qq < 4; qq++) {
      const uint j = rm0 + 8u * qq;
      device const float* fj = (j < nmix) ? (fn + size_t(j) * W + size_t(sg) * H + ck + cn0) : nullptr;
      for (int hh = 0; hh < 2; hh++) {
        const float4 f4 = fj ? *(device const float4*)(fj + 16u * hh) : float4(0.0f);
        wb[hh * 16 + qq * 4 + 0] = bfloat(f4.x);
        wb[hh * 16 + qq * 4 + 1] = bfloat(f4.y);
        wb[hh * 16 + qq * 4 + 2] = bfloat(f4.z);
        wb[hh * 16 + qq * 4 + 3] = bfloat(f4.w);
      }
    }
    op.run(ha, wb, cs);
  }
  for (int i = 0; i < 16; i++) {
    const uint r = rm0 + 8u * ((i >> 2) & 1u), c = cn0 + (i & 3u) + 16u * (i >> 3u);
    Sp[sg][r][c] = cs[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint idx = tid; idx < m_cnt * nmix; idx += 128u) {
    const uint bm = idx / nmix, j = idx % nmix;
    hcpart[(size_t(m0 + bm) * a.npart + p) * 34u + j] = Sp[0][bm][j] + Sp[1][bm][j] + Sp[2][bm][j] + Sp[3][bm][j];
  }
}

// 19e. Dedicated 1-Pass M=1 HCA Attention with FUSED Q-RMSNorm + RoPE!
// 64 TGs (1 per head = 100% of 64 cores!), 4 simdgroups striding KV,
// ZERO threadgroup barriers inside KV loop, coalesced bfloat4 loads, NO qnorm kernel, NO merge kernel!
struct HcaAttnM1Args { uint M, p0, nh, hd, cr, win, ring, nope, rdim; float scale, eps; };
kernel void e5_hca_attn_m1(device const float* qb_out [[buffer(0)]],
                           device const bfloat* comp_cache [[buffer(1)]],
                           device const bfloat* swa_kv [[buffer(2)]],
                           device const float* sink [[buffer(3)]],
                           device bfloat* o_out [[buffer(4)]],
                           constant HcaAttnM1Args& a [[buffer(5)]],
                           device const float* cos_t [[buffer(6)]],
                           device const float* sin_t [[buffer(7)]],
                           uint h [[threadgroup_position_in_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort l [[thread_index_in_simdgroup]]) {
  threadgroup float sh_mx[4];
  threadgroup float sh_lse[4];
  threadgroup float4 sh_o[4][32][4];
  const uint pos = a.p0;
  const uint nc = (pos + 1u) / a.cr;
  const uint swa_s = (pos + 1u > a.win) ? (pos + 1u - a.win) : 0u;
  const uint n_swa = pos + 1u - swa_s;
  const uint n_tot = nc + n_swa;

  const uint d_off = uint(l) * 16u;
  device const float* qr = qb_out + size_t(h) * 512u + d_off;
  float4 q0 = *(device const float4*)(qr + 0u);
  float4 q1 = *(device const float4*)(qr + 4u);
  float4 q2 = *(device const float4*)(qr + 8u);
  float4 q3 = *(device const float4*)(qr + 12u);
  const float qr_inv = rsqrt(simd_sum(dot(q0, q0) + dot(q1, q1) + dot(q2, q2) + dot(q3, q3)) / 512.0f + a.eps);
  q0 *= qr_inv; q1 *= qr_inv; q2 *= qr_inv; q3 *= qr_inv;
  if (l >= 28u) {
    const uint j0 = (d_off - a.nope) >> 1;
    device const float* cp = cos_t + size_t(pos) * (a.rdim / 2u) + j0;
    device const float* sp = sin_t + size_t(pos) * (a.rdim / 2u) + j0;
    auto rot4 = [&](float4 v, uint jo) -> float4 {
      float c0 = cp[jo], s0 = sp[jo], c1 = cp[jo + 1], s1 = sp[jo + 1];
      return float4(v.x * c0 - v.y * s0, v.y * c0 + v.x * s0, v.z * c1 - v.w * s1, v.w * c1 + v.z * s1);
    };
    q0 = rot4(q0, 0u); q1 = rot4(q1, 2u); q2 = rot4(q2, 4u); q3 = rot4(q3, 6u);
  }

  float4 o0 = float4(0.0f), o1 = float4(0.0f), o2 = float4(0.0f), o3 = float4(0.0f);
  float mx = (sg == 0) ? sink[h] : -INFINITY;
  float lse = (sg == 0) ? 1.0f : 0.0f;

  // Branchless 2-loop KV scan (1: comp_cache linear; 2: swa_kv with conditional wrap, ZERO % div!)
  uint j = sg;
  for (; j + 4u < nc; j += 8u) {
    device const bfloat* kr0 = comp_cache + size_t(j) * 512u + d_off;
    device const bfloat* kr1 = comp_cache + size_t(j + 4u) * 512u + d_off;
    const float4 k00 = float4(*(device const bfloat4*)(kr0 + 0u));
    const float4 k01 = float4(*(device const bfloat4*)(kr0 + 4u));
    const float4 k02 = float4(*(device const bfloat4*)(kr0 + 8u));
    const float4 k03 = float4(*(device const bfloat4*)(kr0 + 12u));
    const float4 k10 = float4(*(device const bfloat4*)(kr1 + 0u));
    const float4 k11 = float4(*(device const bfloat4*)(kr1 + 4u));
    const float4 k12 = float4(*(device const bfloat4*)(kr1 + 8u));
    const float4 k13 = float4(*(device const bfloat4*)(kr1 + 12u));
    const float sc0 = simd_sum(dot(q0, k00) + dot(q1, k01) + dot(q2, k02) + dot(q3, k03)) * a.scale;
    const float sc1 = simd_sum(dot(q0, k10) + dot(q1, k11) + dot(q2, k12) + dot(q3, k13)) * a.scale;
    const float nmx = max(mx, max(sc0, sc1));
    const float alpha = (mx == -INFINITY) ? 0.0f : exp(mx - nmx);
    const float p0 = exp(sc0 - nmx);
    const float p1 = exp(sc1 - nmx);
    lse = lse * alpha + p0 + p1;
    o0 = fma(p1, k10, fma(p0, k00, o0 * alpha));
    o1 = fma(p1, k11, fma(p0, k01, o1 * alpha));
    o2 = fma(p1, k12, fma(p0, k02, o2 * alpha));
    o3 = fma(p1, k13, fma(p0, k03, o3 * alpha));
    mx = nmx;
  }
  if (j < nc) {
    device const bfloat* kr = comp_cache + size_t(j) * 512u + d_off;
    const float4 k0 = float4(*(device const bfloat4*)(kr + 0u));
    const float4 k1 = float4(*(device const bfloat4*)(kr + 4u));
    const float4 k2 = float4(*(device const bfloat4*)(kr + 8u));
    const float4 k3 = float4(*(device const bfloat4*)(kr + 12u));
    const float sc = simd_sum(dot(q0, k0) + dot(q1, k1) + dot(q2, k2) + dot(q3, k3)) * a.scale;
    const float nmx = max(mx, sc);
    const float alpha = (mx == -INFINITY) ? 0.0f : exp(mx - nmx);
    const float p = exp(sc - nmx);
    lse = lse * alpha + p;
    o0 = fma(p, k0, o0 * alpha);
    o1 = fma(p, k1, o1 * alpha);
    o2 = fma(p, k2, o2 * alpha);
    o3 = fma(p, k3, o3 * alpha);
    mx = nmx;
  }

  const uint swa_s_mod = swa_s % a.ring;
  uint i = sg;
  for (; i + 4u < n_swa; i += 8u) {
    uint r0 = swa_s_mod + i; if (r0 >= a.ring) r0 -= a.ring;
    uint r1 = swa_s_mod + i + 4u; if (r1 >= a.ring) r1 -= a.ring;
    device const bfloat* kr0 = swa_kv + size_t(r0) * 512u + d_off;
    device const bfloat* kr1 = swa_kv + size_t(r1) * 512u + d_off;
    const float4 k00 = float4(*(device const bfloat4*)(kr0 + 0u));
    const float4 k01 = float4(*(device const bfloat4*)(kr0 + 4u));
    const float4 k02 = float4(*(device const bfloat4*)(kr0 + 8u));
    const float4 k03 = float4(*(device const bfloat4*)(kr0 + 12u));
    const float4 k10 = float4(*(device const bfloat4*)(kr1 + 0u));
    const float4 k11 = float4(*(device const bfloat4*)(kr1 + 4u));
    const float4 k12 = float4(*(device const bfloat4*)(kr1 + 8u));
    const float4 k13 = float4(*(device const bfloat4*)(kr1 + 12u));
    const float sc0 = simd_sum(dot(q0, k00) + dot(q1, k01) + dot(q2, k02) + dot(q3, k03)) * a.scale;
    const float sc1 = simd_sum(dot(q0, k10) + dot(q1, k11) + dot(q2, k12) + dot(q3, k13)) * a.scale;
    const float nmx = max(mx, max(sc0, sc1));
    const float alpha = (mx == -INFINITY) ? 0.0f : exp(mx - nmx);
    const float p0 = exp(sc0 - nmx);
    const float p1 = exp(sc1 - nmx);
    lse = lse * alpha + p0 + p1;
    o0 = fma(p1, k10, fma(p0, k00, o0 * alpha));
    o1 = fma(p1, k11, fma(p0, k01, o1 * alpha));
    o2 = fma(p1, k12, fma(p0, k02, o2 * alpha));
    o3 = fma(p1, k13, fma(p0, k03, o3 * alpha));
    mx = nmx;
  }
  if (i < n_swa) {
    uint r0 = swa_s_mod + i; if (r0 >= a.ring) r0 -= a.ring;
    device const bfloat* kr = swa_kv + size_t(r0) * 512u + d_off;
    const float4 k0 = float4(*(device const bfloat4*)(kr + 0u));
    const float4 k1 = float4(*(device const bfloat4*)(kr + 4u));
    const float4 k2 = float4(*(device const bfloat4*)(kr + 8u));
    const float4 k3 = float4(*(device const bfloat4*)(kr + 12u));
    const float sc = simd_sum(dot(q0, k0) + dot(q1, k1) + dot(q2, k2) + dot(q3, k3)) * a.scale;
    const float nmx = max(mx, sc);
    const float alpha = (mx == -INFINITY) ? 0.0f : exp(mx - nmx);
    const float p = exp(sc - nmx);
    lse = lse * alpha + p;
    o0 = fma(p, k0, o0 * alpha);
    o1 = fma(p, k1, o1 * alpha);
    o2 = fma(p, k2, o2 * alpha);
    o3 = fma(p, k3, o3 * alpha);
    mx = nmx;
  }

  if (l == 0) { sh_mx[sg] = mx; sh_lse[sg] = lse; }
  sh_o[sg][l][0] = o0; sh_o[sg][l][1] = o1; sh_o[sg][l][2] = o2; sh_o[sg][l][3] = o3;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0) {
    const float mn = max(max(sh_mx[0], sh_mx[1]), max(sh_mx[2], sh_mx[3]));
    float tot_lse = 0.0f;
    float4 r0 = float4(0.0f), r1 = float4(0.0f), r2 = float4(0.0f), r3 = float4(0.0f);
    for (uint s = 0; s < 4u; s++) {
      if (sh_lse[s] <= 0.0f) continue;
      const float w = exp(sh_mx[s] - mn);
      tot_lse = fma(w, sh_lse[s], tot_lse);
      r0 = fma(w, sh_o[s][l][0], r0);
      r1 = fma(w, sh_o[s][l][1], r1);
      r2 = fma(w, sh_o[s][l][2], r2);
      r3 = fma(w, sh_o[s][l][3], r3);
    }
    const float inv = 1.0f / max(tot_lse, 1e-30f);
    const float4 v0 = r0 * inv, v1 = r1 * inv, v2 = r2 * inv, v3 = r3 * inv;
    device bfloat* dst = o_out + size_t(h) * 512u + d_off;
    *(device bfloat4*)(dst + 0u) = bfloat4(bfloat(v0.x), bfloat(v0.y), bfloat(v0.z), bfloat(v0.w));
    *(device bfloat4*)(dst + 4u) = bfloat4(bfloat(v1.x), bfloat(v1.y), bfloat(v1.z), bfloat(v1.w));
    *(device bfloat4*)(dst + 8u) = bfloat4(bfloat(v2.x), bfloat(v2.y), bfloat(v2.z), bfloat(v2.w));
    *(device bfloat4*)(dst + 12u) = bfloat4(bfloat(v3.x), bfloat(v3.y), bfloat(v3.z), bfloat(v3.w));
  }
}

// 19f. Parallel 17-TG M=1 MoE Combine + Shared SiTU-GLU (float4 vectorized, zero comb shmem when !hadamard!):
// TG 0 (512 threads = 1 float4/thread) does routed combine + RMSNorm -> frag_a[0..2047].
// TGs 1..16 (128 threads each) compute sgu SiTU-GLU -> frag_a[2048..4095] in parallel across 16 cores!
kernel void e5_moe_combine_pack_m1(device const float* exp_out [[buffer(0)]],
                                   device const float* top_w [[buffer(1)]],
                                   device const bfloat* lnorm [[buffer(2)]],
                                   device const float* moe_in [[buffer(3)]],
                                   device bfloat* dst_frag [[buffer(4)]],
                                   constant MoeCombArgs& a [[buffer(5)]],
                                   uint tg [[threadgroup_position_in_grid]],
                                   ushort tid [[thread_index_in_threadgroup]],
                                   ushort ntg [[threads_per_threadgroup]]) {
  if (tg > 0u) {
    const uint c = (tg - 1u) * 128u + tid;
    if (tid >= 128u || c >= 2048u) return;
    device const float* sgu = moe_in + 3072u;
    const uint blk = c >> 4, off = c & 15u;
    const float sv = situ_act(sgu[blk * 32u + off], sgu[blk * 32u + 16u + off], a.beta, a.lbeta);
    dst_frag[frag_off(0u, 2048u + c, 4096u)] = bfloat(sv);
    return;
  }
  threadgroup float sh[32];
  float w[10];
  for (uint k = 0; k < 10u; k++) w[k] = top_w[k];
  const uint c4 = uint(tid) << 2;
  float4 acc = float4(0.0f);
  for (uint k = 0; k < 10u; k++)
    acc = fma(w[k], *(device const float4*)(exp_out + size_t(k) * 2048u + c4), acc);
  const float r = rsqrt(tg_sum(dot(acc, acc), sh, tid, ntg) / 2048.0f + a.eps);
  if (!a.hadamard) {
    const bfloat4 nw4 = *(device const bfloat4*)(lnorm + c4);
    const float4 v = acc * r * float4(nw4);
    *(device bfloat4*)(dst_frag + frag_off(0u, c4, 4096u)) = bfloat4(bfloat(v.x), bfloat(v.y), bfloat(v.z), bfloat(v.w));
  } else {
    threadgroup float comb[2048];
    comb[c4 + 0u] = acc.x; comb[c4 + 1u] = acc.y; comb[c4 + 2u] = acc.z; comb[c4 + 3u] = acc.w;
    tg_fwht_2048(comb, tid, ntg);
    const bfloat4 nw4 = *(device const bfloat4*)(lnorm + c4);
    const float4 v = float4(comb[c4], comb[c4 + 1u], comb[c4 + 2u], comb[c4 + 3u]) * r * float4(nw4);
    *(device bfloat4*)(dst_frag + frag_off(0u, c4, 4096u)) = bfloat4(bfloat(v.x), bfloat(v.y), bfloat(v.z), bfloat(v.w));
  }
}

// 19c-m1. 64-TG (1 SG/TG) M=1 HCA VHA + in-register inverse RoPE -> grouped wo_a frags [8, 1, 4096]
// Grid: (4 head_groups, 16 dim_chunks_of_32, 1), TG: (32, 1, 1) -> 64 TGs = 100% of 64 GPU cores, ZERO TG barriers!
kernel void e5_hca_vha_invrope_m1(device const bfloat* o_in [[buffer(0)]],
                                  device const float* vha_m [[buffer(1)]],
                                  device const float* cos_t [[buffer(2)]],
                                  device const float* sin_t [[buffer(3)]],
                                  device bfloat* out_frags [[buffer(4)]],
                                  constant HcaVhaArgs& a [[buffer(5)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  ushort l [[thread_index_in_simdgroup]]) {
  threadgroup bfloat Ot[64][32];
  const uint hg = tg.x, dc = tg.y;
  const uint h0 = hg * 16u, d0_blk = dc * 32u, pos = a.p0;
  // Cooperatively load bf16 o_in[0..63, d0_blk..d0_blk+31] into Ot (4 KB) with 1 SG (32 lanes x 16 bfloat4)
  const uint lr = l >> 3, lc = (l & 7u) << 2;
  for (uint r0 = 0; r0 < 64u; r0 += 4u) {
    const uint g = r0 + lr;
    *(threadgroup bfloat4*)(&Ot[g][lc]) = *(device const bfloat4*)(o_in + size_t(g) * 512u + d0_blk + lc);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  const ushort m0 = ((l >> 1) & 3) + 4 * (l >> 4), n0 = 4 * (l & 1) + 8 * ((l >> 3) & 1);
  constexpr auto desc =
      matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto vm = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto cs = op.template get_destination_cooperative_tensor<decltype(vm), decltype(wb), float>();
  for (int i = 0; i < 16; i++) cs[i] = 0.0f;

  for (uint ks = 0; ks < 2u; ks++) {
    const uint k0 = ks * 32u;
    for (int i = 0; i < 16; i++) {
      const uint hh = m0 + 8 * ((i >> 2) & 1), kk = k0 + n0 + (i & 3) + 16 * (i >> 3);
      vm[i] = bfloat(vha_m[(h0 + hh) * 64u + kk]);
    }
    for (int hh = 0; hh < 2; hh++) {
      for (int e = 0; e < 4; e++) {
        const uint kk = k0 + n0 + e + 16 * hh;
        for (int qq = 0; qq < 4; qq++)
          wb[hh * 16 + qq * 4 + e] = Ot[kk][m0 + 8 * qq];
      }
    }
    op.run(vm, wb, cs);
  }

  float outv[16];
  for (int i = 0; i < 16; i++) outv[i] = cs[i];
  if (dc >= 14u) {
    device const float* cp = cos_t + size_t(pos) * (a.rdim / 2u);
    device const float* sp = sin_t + size_t(pos) * (a.rdim / 2u);
    for (int p = 0; p < 8; p++) {
      const int i0 = p * 2, i1 = i0 + 1;
      const uint d = d0_blk + n0 + (i0 & 3) + 16 * (i0 >> 3);
      const uint j = (d - a.nope) >> 1;
      const float cs_v = cp[j], sn_v = sp[j];
      const float e_v = outv[i0], o_v = outv[i1];
      outv[i0] = e_v * cs_v + o_v * sn_v;
      outv[i1] = o_v * cs_v - e_v * sn_v;
    }
  }
  const uint hpg = a.nh / a.n_groups;
  const uint K = hpg * a.hd;
  for (int i = 0; i < 16; i++) {
    const uint h = h0 + m0 + 8 * ((i >> 2) & 1);
    const uint d = d0_blk + n0 + (i & 3) + 16 * (i >> 3);
    const uint grp = h / hpg, hig = h % hpg;
    out_frags[size_t(grp) * a.g_stride_bf16 + frag_off(0u, hig * a.hd + d, K)] = bfloat(outv[i]);
  }
}

// 19c. matmul2d HCA VHA + in-register inverse RoPE -> grouped wo_a frags [8, M, 4096]
// Grid: (4 head_groups, 4 dim_chunks, M), TG: (128, 1, 1) (4 simdgroups x 32 dims = 128 dims/TG).
kernel void e5_hca_vha_invrope_mma(device const bfloat* o_in [[buffer(0)]],
                                   device const float* vha_m [[buffer(1)]],
                                   device const float* cos_t [[buffer(2)]],
                                   device const float* sin_t [[buffer(3)]],
                                   device bfloat* out_frags [[buffer(4)]],
                                   constant HcaVhaArgs& a [[buffer(5)]],
                                   uint3 tg [[threadgroup_position_in_grid]],
                                   ushort sg [[simdgroup_index_in_threadgroup]],
                                   ushort l [[thread_index_in_simdgroup]]) {
  threadgroup bfloat Ot[64][128];
  const uint hg = tg.x, dc = tg.y, m = tg.z;
  const uint h0 = hg * 16u, d0_blk = dc * 128u, pos = a.p0 + m;
  const ushort tid = sg * 32u + l;

  // Cooperatively load bf16 o_in[m, 0..63, d0_blk..d0_blk+127] into Ot (16 KB) with coalesced uint4 (8 bf16/load!)
  device const bfloat* obase = o_in + size_t(m) * 64u * 512u + d0_blk;
  for (uint idx = tid; idx < 64u * 16u; idx += 128u) {
    const uint g = idx >> 4, col8 = (idx & 15u) * 8u;
    *(threadgroup uint4*)(&Ot[g][col8]) = *(device const uint4*)(obase + g * 512u + col8);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const ushort m0 = ((l >> 1) & 3) + 4 * (l >> 4), n0 = 4 * (l & 1) + 8 * ((l >> 3) & 1);
  constexpr auto desc =
      matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto vm = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto cs = op.template get_destination_cooperative_tensor<decltype(vm), decltype(wb), float>();
  for (int i = 0; i < 16; i++) cs[i] = 0.0f;

  const uint d_sg = sg * 32u;
  for (uint ks = 0; ks < 2u; ks++) {
    const uint k0 = ks * 32u;
    for (int i = 0; i < 16; i++) {
      const uint hh = m0 + 8 * ((i >> 2) & 1), kk = k0 + n0 + (i & 3) + 16 * (i >> 3);
      vm[i] = bfloat(vha_m[(h0 + hh) * 64u + kk]);
    }
    for (int hh = 0; hh < 2; hh++) {
      for (int e = 0; e < 4; e++) {
        const uint kk = k0 + n0 + e + 16 * hh;
        for (int qq = 0; qq < 4; qq++)
          wb[hh * 16 + qq * 4 + e] = Ot[kk][d_sg + m0 + 8 * qq];
      }
    }
    op.run(vm, wb, cs);
  }

  float outv[16];
  for (int i = 0; i < 16; i++) outv[i] = cs[i];
  // In-register inverse RoPE for dims 448..511 (dc == 3 && sg >= 2): pairs (2p, 2p+1) are adjacent in outv!
  if (dc == 3u && sg >= 2u) {
    device const float* cp = cos_t + size_t(pos) * (a.rdim / 2u);
    device const float* sp = sin_t + size_t(pos) * (a.rdim / 2u);
    for (int p = 0; p < 8; p++) {
      const int i0 = p * 2, i1 = i0 + 1;
      const uint d = d0_blk + d_sg + n0 + (i0 & 3) + 16 * (i0 >> 3);
      const uint j = (d - a.nope) >> 1;
      const float cs_v = cp[j], sn_v = sp[j];
      const float e_v = outv[i0], o_v = outv[i1];
      outv[i0] = e_v * cs_v + o_v * sn_v;
      outv[i1] = o_v * cs_v - e_v * sn_v;
    }
  }

  const uint hpg = a.nh / a.n_groups;  // 8
  const uint K = hpg * a.hd;           // 4096
  for (int i = 0; i < 16; i++) {
    const uint h = h0 + m0 + 8 * ((i >> 2) & 1);
    const uint d = d0_blk + d_sg + n0 + (i & 3) + 16 * (i >> 3);
    const uint grp = h / hpg, hig = h % hpg;
    out_frags[size_t(grp) * a.g_stride_bf16 + frag_off(m, hig * a.hd + d, K)] = bfloat(outv[i]);
  }
}

// 19d. matmul2d DSA VHA + Gate -> o_proj frags [M, 16384]
// Grid: (4 head_groups, 2 dim_chunks, M), TG: (128, 1, 1) (4 simdgroups x 32 dims = 128 dims/TG).
kernel void e5_dsa_vha_gate_mma(device const float* o_grp [[buffer(0)]],
                                device const float* vha_m [[buffer(1)]],
                                device const float* qb_out [[buffer(2)]],
                                device bfloat* dst_frag [[buffer(3)]],
                                constant DsaVhaArgs& a [[buffer(4)]],
                                uint3 tg [[threadgroup_position_in_grid]],
                                ushort sg [[simdgroup_index_in_threadgroup]],
                                ushort l [[thread_index_in_simdgroup]]) {
  threadgroup bfloat Ot[64][128];
  const uint hg = tg.x, dc = tg.y, m = tg.z;
  const uint h0 = hg * 16u, d0_blk = dc * 128u;
  const ushort tid = sg * 32u + l;

  for (uint idx = tid; idx < 64u * 32u; idx += 128u) {
    const uint g = idx >> 5, col4 = (idx & 31u) * 4u;
    const float4 v4 = *(device const float4*)(o_grp + size_t(g) * a.o_gstride + size_t(m) * 256u + d0_blk + col4);
    *(threadgroup bfloat4*)(&Ot[g][col4]) = bfloat4(bfloat(v4.x), bfloat(v4.y), bfloat(v4.z), bfloat(v4.w));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const ushort m0 = ((l >> 1) & 3) + 4 * (l >> 4), n0 = 4 * (l & 1) + 8 * ((l >> 3) & 1);
  constexpr auto desc =
      matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto vm = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto cs = op.template get_destination_cooperative_tensor<decltype(vm), decltype(wb), float>();
  for (int i = 0; i < 16; i++) cs[i] = 0.0f;

  const uint d_sg = sg * 32u;
  for (uint ks = 0; ks < 2u; ks++) {
    const uint k0 = ks * 32u;
    for (int i = 0; i < 16; i++) {
      const uint hh = m0 + 8 * ((i >> 2) & 1), kk = k0 + n0 + (i & 3) + 16 * (i >> 3);
      vm[i] = bfloat(vha_m[(h0 + hh) * 64u + kk]);
    }
    for (int hh = 0; hh < 2; hh++) {
      for (int e = 0; e < 4; e++) {
        const uint kk = k0 + n0 + e + 16 * hh;
        for (int qq = 0; qq < 4; qq++)
          wb[hh * 16 + qq * 4 + e] = Ot[kk][d_sg + m0 + 8 * qq];
      }
    }
    op.run(vm, wb, cs);
  }

  device const float* gate_row = qb_out + size_t(m) * 40960u + 16384u;
  for (int i = 0; i < 16; i++) {
    const uint h = h0 + m0 + 8 * ((i >> 2) & 1);
    const uint d = d0_blk + d_sg + n0 + (i & 3) + 16 * (i >> 3);
    const uint col = h * 256u + d;
    dst_frag[frag_off(m, col, 16384u)] = bfloat(cs[i] * sigm(gate_row[col]));
  }
}
