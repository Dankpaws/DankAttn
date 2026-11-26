// hy4: the Hy4-flash specific ops (everything that is not an mm2 / mmp matmul).
// Rows are decode / verify rows (M <= MAXM) unless a kernel says otherwise. f32 everywhere except
// the bf16 matmul operands (frag layout, see mm2.metal) and the bf16 latent cache.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#ifdef LANES
struct LaneState {
  device uchar* lat;
  device float* tokring;
  device bfloat* bk;
};
#define LANE_PARAMS , device const LaneState* lt [[buffer(12)]], device const uint* row_lane [[buffer(13)]], device const uint* row_pos [[buffer(14)]]
#define ROW_POS(m) (row_pos[m])
#else
#define LANE_PARAMS
#define ROW_POS(m) (a.p0 + (m))
#endif

static inline uint frag_off(uint m, uint k, uint K) {
  const uint kk = k & 31, mm = m & 15;
  const uint l = ((kk >> 2) & 1) | ((mm & 3) << 1) | (((kk >> 3) & 1) << 3) | (((mm >> 2) & 1) << 4);
  const uint i = (kk & 3) | ((mm >> 3) << 2) | ((kk >> 4) << 3);
  return ((m >> 4) * (K >> 5) + (k >> 5)) * 512 + l * 16 + i;
}
static inline float sigm(float x) { return 1.0f / (1.0f + exp(-x)); }
static inline float silu(float x) { return x / (1.0f + exp(-x)); }

// threadgroup sum of one float (any power-of-two size <= 1024); every thread gets the total
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

// ------------------------------------------------------------------ embedding
struct EmbArgs { uint M, H, HC; };
kernel void h4_embed(device const uint* ids [[buffer(0)]], device const bfloat* E [[buffer(1)]],
                     device float* h [[buffer(2)]], constant EmbArgs& a [[buffer(3)]],
                     uint2 g [[thread_position_in_grid]]) {
  const uint c = g.x, m = g.y;
  if (c >= a.H || m >= a.M) return;
  const float v = float(E[size_t(ids[m]) * a.H + c]);
  for (uint s = 0; s < a.HC; s++) h[(size_t(m) * a.HC + s) * a.H + c] = v;
}

// Multimodal embedding: scatter precomputed visual embeddings (vis_idx[m] >= 0) else token embed.
kernel void h4_embed_vis(device const uint* ids [[buffer(0)]], device const bfloat* E [[buffer(1)]],
                         device const float* vis [[buffer(2)]], device const int* vis_idx [[buffer(3)]],
                         device float* h [[buffer(4)]], constant EmbArgs& a [[buffer(5)]],
                         uint2 g [[thread_position_in_grid]]) {
  const uint c = g.x, m = g.y;
  if (c >= a.H || m >= a.M) return;
  const int vi = vis_idx[m];
  const float v = vi >= 0 ? vis[size_t(vi) * a.H + c] : float(E[size_t(ids[m]) * a.H + c]);
  for (uint s = 0; s < a.HC; s++) h[(size_t(m) * a.HC + s) * a.H + c] = v;
}

// ------------------------------------------------------------------ hyper-connections
// Two-pass hyper-connection read-in over NP channel slices (grid (NP, M), 256 threads): pass 1 injects and
// writes per-slice partials [m][slice][18] = 8 mix dots, then the 10 Gram terms <h_s, h_t> (s <= t);
// pass 2 reduces them in slice order (every threadgroup the same way), so |flat|^2 = trace(G) and
// |x|^2 = pre' G pre come without another pass over the streams.
// mode 0: sublayer read-in (8 mixes); mode 1: head (4 mixes); mode 2: inject only (before Engram).
struct HcArgs { uint M, H, inject, mode; float eps, hc_eps, mag; uint ynorm; };  // ynorm: y = rms(y) * w (h4_rms's bits)
// Single-pass Gram-trick hyper-connection read-in (1 threadgroup of 1024 threads per row):
// Each thread holds its H/1024 (<= 4, 3 for H=3072) channels of h[4] in registers across one 18-sum
// barrier, eliminating the hcpart global round-trip, the second h[] read, and half the HC dispatches.
kernel void h4_hc_1pass(device float* h [[buffer(0)]], device const float* y [[buffer(1)]],
                        device const float* postp [[buffer(2)]], device const float* fn [[buffer(3)]],
                        device const float* base [[buffer(4)]], device const float* scale [[buffer(5)]],
                        device const float* nw [[buffer(6)]], device bfloat* xfrag [[buffer(7)]],
                        device float* xf [[buffer(8)]], device float* post [[buffer(9)]],
                        constant HcArgs& a [[buffer(10)]], device const float* yw [[buffer(11)]],
                        uint m [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                        ushort ntg [[threads_per_threadgroup]]) {
  threadgroup float shp[18][32];
  threadgroup float sh_y[32];
  threadgroup float tot18[18];
  const uint H = a.H, W = 4 * H;
  device float* hr = h + size_t(m) * W;
  const uint nmix = a.mode == 0 ? 8 : 4;
  float yr = 1.0f;
  if (a.inject && a.ynorm) {
    device const float* xr = y + size_t(m) * H;
    float ss = 0.0f;
    for (uint c = tid; c < H; c += ntg) ss = fma(xr[c], xr[c], ss);
    yr = rsqrt(tg_sum(ss, sh_y, tid, ntg) / float(H) + a.eps);
  }
  float vreg[4][4];  // up to 4 channels per thread (H <= 4096; H=3072 uses 3)
  float acc[18];
  for (uint i = 0; i < 18; i++) acc[i] = 0.0f;
  uint kc = 0;
  for (uint c = tid; c < H; c += ntg, kc++) {
    const float yc = a.inject ? (a.ynorm ? y[size_t(m) * H + c] * yr * yw[c] : y[size_t(m) * H + c]) : 0.0f;
    float v[4];
    for (uint s = 0; s < 4; s++) {
      v[s] = hr[s * H + c];
      if (a.inject) { v[s] = fma(postp[m * 4 + s], yc, v[s]); hr[s * H + c] = v[s]; }
      vreg[kc][s] = v[s];
    }
    if (a.mode == 2) continue;
    for (uint j = 0; j < nmix; j++)
      for (uint s = 0; s < 4; s++) acc[j] = fma(fn[size_t(j) * W + s * H + c], v[s], acc[j]);
    uint q = 8;
    for (uint s = 0; s < 4; s++)
      for (uint t = s; t < 4; t++) { acc[q] = fma(v[s], v[t], acc[q]); q++; }
  }
  if (a.mode == 2) return;
  const ushort sgi = tid / 32, ns = (ntg + 31) / 32;
  for (uint i = 0; i < 18; i++) {
    const float t = simd_sum(acc[i]);
    if ((tid & 31) == 0) shp[i][sgi] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 18) {
    float t = 0.0f;
    for (ushort k = 0; k < ns; k++) t += shp[tid][k];
    tot18[tid] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float G[4][4];
  uint q = 8;
  for (uint s = 0; s < 4; s++)
    for (uint u = s; u < 4; u++) { G[s][u] = tot18[q]; G[u][s] = tot18[q]; q++; }
  const float tr = G[0][0] + G[1][1] + G[2][2] + G[3][3];
  const float r = rsqrt(tr / float(W) + a.eps);
  float pre[4];
  for (uint s = 0; s < 4; s++) pre[s] = sigm(tot18[s] * r * scale[0] + base[s]) + a.hc_eps;
  if (a.mode == 0 && tid < 4) post[m * 4 + tid] = a.mag * sigm(tot18[4 + tid] * r * scale[1] + base[4 + tid]) + a.hc_eps;
  float x2 = 0.0f;
  for (uint s = 0; s < 4; s++)
    for (uint u = 0; u < 4; u++) x2 = fma(pre[s] * pre[u], G[s][u], x2);
  const float xr = rsqrt(x2 / float(H) + a.eps);
  kc = 0;
  for (uint c = tid; c < H; c += ntg, kc++) {
    float x = 0.0f;
    for (uint s = 0; s < 4; s++) x = fma(pre[s], vreg[kc][s], x);
    const float v = x * xr * nw[c];
    xf[size_t(m) * H + c] = v;
    xfrag[frag_off(m, c, H)] = bfloat(v);
  }
}

kernel void h4_hc_part(device float* h [[buffer(0)]], device const float* y [[buffer(1)]],
                       device const float* postp [[buffer(2)]], device const float* fn [[buffer(3)]],
                       device float* part [[buffer(4)]], constant HcArgs& a [[buffer(5)]],
                       device const float* yw [[buffer(6)]],
                       uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       uint2 ntg2 [[threads_per_threadgroup]], uint2 ng [[threadgroups_per_grid]]) {
  const ushort ntg = ushort(ntg2.x);
  threadgroup float sh[32];
  const uint m = tg.y, H = a.H, W = 4 * H, per = H / ng.x, c0 = tg.x * per;
  device float* hr = h + size_t(m) * W;
  const uint nmix = a.mode == 0 ? 8 : 4;
  // ynorm (decode): the sublayer output's RMSNorm, as h4_rms computes it (same threads, order and
  // expression), instead of a separate dispatch; every slice of the row recomputes the row's scale
  float yr = 1.0f;
  if (a.inject && a.ynorm) {
    device const float* xr = y + size_t(m) * H;
    float ss = 0.0f;
    for (uint i = tid; i < H; i += ntg) ss = fma(xr[i], xr[i], ss);
    yr = rsqrt(tg_sum(ss, sh, tid, ntg) / float(H) + a.eps);
  }
  float acc[18];
  for (uint i = 0; i < 18; i++) acc[i] = 0.0f;
  for (uint c = c0 + tid; c < c0 + per; c += ntg) {
    float v[4];
    const float yc = a.inject ? (a.ynorm ? y[size_t(m) * H + c] * yr * yw[c] : y[size_t(m) * H + c]) : 0.0f;
    for (uint s = 0; s < 4; s++) {
      v[s] = hr[s * H + c];
      if (a.inject) { v[s] = fma(postp[m * 4 + s], yc, v[s]); hr[s * H + c] = v[s]; }
    }
    if (a.mode == 2) continue;
    for (uint j = 0; j < nmix; j++)
      for (uint s = 0; s < 4; s++) acc[j] = fma(fn[size_t(j) * W + s * H + c], v[s], acc[j]);
    uint q = 8;
    for (uint s = 0; s < 4; s++)
      for (uint t = s; t < 4; t++) { acc[q] = fma(v[s], v[t], acc[q]); q++; }
  }
  if (a.mode == 2) return;
  // the 18 sums of tg_sum (simd sums, then the simdgroups' in order from 0.0f) behind one barrier
  threadgroup float shp[18][32];
  const ushort sgi = tid / 32, ns = (ntg + 31) / 32;
  for (uint i = 0; i < 18; i++) {
    const float t = simd_sum(acc[i]);
    if ((tid & 31) == 0) shp[i][sgi] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 18) {
    float t = 0.0f;
    for (ushort k = 0; k < ns; k++) t += shp[tid][k];
    part[(size_t(m) * ng.x + tg.x) * 18 + tid] = t;
  }
  (void)sh;
}
kernel void h4_hc_fin(device const float* h [[buffer(0)]], device const float* part [[buffer(1)]],
                      device const float* base [[buffer(2)]], device const float* scale [[buffer(3)]],
                      device const float* nw [[buffer(4)]], device bfloat* xfrag [[buffer(5)]],
                      device float* xf [[buffer(6)]], device float* post [[buffer(7)]],
                      constant HcArgs& a [[buffer(8)]], constant uint& np [[buffer(9)]],
                      uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                      uint2 ntg2 [[threads_per_threadgroup]], uint2 ng [[threadgroups_per_grid]]) {
  const ushort ntg = ushort(ntg2.x);
  const uint m = tg.y, H = a.H, W = 4 * H;
  float t[18];
  for (uint i = 0; i < 18; i++) t[i] = 0.0f;
  for (uint p = 0; p < np; p++)
    for (uint i = 0; i < 18; i++) t[i] += part[(size_t(m) * np + p) * 18 + i];
  float G[4][4];
  uint q = 8;
  for (uint s = 0; s < 4; s++)
    for (uint u = s; u < 4; u++) { G[s][u] = t[q]; G[u][s] = t[q]; q++; }
  const float tr = G[0][0] + G[1][1] + G[2][2] + G[3][3];
  const float r = rsqrt(tr / float(W) + a.eps);
  float pre[4];
  for (uint s = 0; s < 4; s++) pre[s] = sigm(t[s] * r * scale[0] + base[s]) + a.hc_eps;
  if (a.mode == 0 && tg.x == 0 && tid < 4) post[m * 4 + tid] = a.mag * sigm(t[4 + tid] * r * scale[1] + base[4 + tid]) + a.hc_eps;
  float x2 = 0.0f;
  for (uint s = 0; s < 4; s++)
    for (uint u = 0; u < 4; u++) x2 = fma(pre[s] * pre[u], G[s][u], x2);
  const float xr = rsqrt(x2 / float(H) + a.eps);
  device const float* hr = h + size_t(m) * W;
  const uint per = H / ng.x;
  for (uint c = tg.x * per + tid; c < (tg.x + 1) * per; c += ntg) {
    float x = 0.0f;
    for (uint s = 0; s < 4; s++) x = fma(pre[s], hr[s * H + c], x);
    const float v = x * xr * nw[c];
    xf[size_t(m) * H + c] = v;
    xfrag[frag_off(m, c, H)] = bfloat(v);
  }
}

// ------------------------------------------------------------------ generic RMSNorm of a row slice
// out f32 rows [M][D] (stride ldo) and/or a bf16 frag [M][D]; eps per call. One threadgroup per row.
struct RmsArgs { uint M, D, ldi, off, ldo, want_f, want_frag; float eps; uint fo, fk; };  // frag column offset / width (0: D)
kernel void h4_rms(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                   device float* y [[buffer(2)]], device bfloat* yf [[buffer(3)]], constant RmsArgs& a [[buffer(4)]],
                   uint m [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                   ushort ntg [[threads_per_threadgroup]]) {
  threadgroup float sh[32];
  device const float* xr = x + size_t(m) * a.ldi + a.off;
  float ss = 0.0f;
  for (uint i = tid; i < a.D; i += ntg) ss = fma(xr[i], xr[i], ss);
  const float r = rsqrt(tg_sum(ss, sh, tid, ntg) / float(a.D) + a.eps);
  for (uint i = tid; i < a.D; i += ntg) {
    const float v = xr[i] * r * w[i];
    if (a.want_f) y[size_t(m) * a.ldo + i] = v;
    if (a.want_frag) yf[frag_off(m, a.fo + i, a.fk ? a.fk : a.D)] = bfloat(v);
  }
}

// ------------------------------------------------------------------ RoPE (interleaved pairs)
// rope tables from the host: [M][64] per base, cos[0..31] then sin[0..31] (pair j turns by inv_j)
static inline void rope_pair(thread float& x0, thread float& x1, float c, float s) {
  const float a = x0 * c - x1 * s, b = x1 * c + x0 * s;
  x0 = a; x1 = b;
}

// ------------------------------------------------------------------ latent cache rows
// bf16: 576 values (1152 B). kv8 (sparse layers' full caches): 576 int8 then 18 f16 scales, one per 32 dims,
// in a 640-byte row. Readers take the layer's mode from their args.
constant uint KV8_ROW = 640;
static inline float latv(device const uchar* lat, uint row, uint d, uint kv8) {
  if (kv8) {
    device const uchar* r = lat + size_t(row) * KV8_ROW;
    return float(as_type<char>(r[d])) * float(((device const half*)(r + 576))[d >> 5]);
  }
  return float(((device const bfloat*)lat)[size_t(row) * 576 + d]);
}
// 4 consecutive dims d..d+3 (one scale group) as bf16 operands
static inline bfloat4 latv4(device const uchar* lat, uint row, uint d, uint kv8) {
  if (kv8) {
    device const uchar* r = lat + size_t(row) * KV8_ROW;
    const char4 c = as_type<char4>(*(device const uint*)(r + d));
    const float sc = float(((device const half*)(r + 576))[d >> 5]);
    return bfloat4(bfloat(float(c.x) * sc), bfloat(float(c.y) * sc), bfloat(float(c.z) * sc), bfloat(float(c.w) * sc));
  }
  return as_type<bfloat4>(*(device const uint2*)((device const bfloat*)lat + size_t(row) * 576 + d));
}

// Latent rows are stored with the dims of every 32-dim chunk permuted, phys(w) = 4 (w % 8) + w / 8, so
// a lane's attention operands are 4 consecutive bytes/halves for K (dims n0.. of the MMA k index, in
// physical order, Q read through the inverse) and for V (logical dims m0 + 8 j, physical 4 m0 + j).
static inline uint lat_phys(uint x) { return (x & ~31u) | ((x & 7u) << 2) | ((x >> 3) & 3u); }
static inline uint lat_logic(uint p) { return (p & ~31u) | ((p & 3u) << 3) | ((p >> 2) & 7u); }

// ------------------------------------------------------------------ attention input
// From attn.in's rows [M][Nin]: q_a -> rms -> frag (K=1024) for q_b; kv_a -> rms(512) + rope(64)
// -> latent cache row; sparse layers: indexer token key -> rms -> ring, head weights.
// One threadgroup (256) per row.
constant uint TOKR = 4096;  // indexer token-key ring (positions), >= rows of a step + 3
struct KvArgs { uint M, Nin, p0, ring, sparse, ikoff, iwoff, kvoff; float eps; float iw_scale; uint kv8; };
kernel void h4_attn_in(device const float* Y [[buffer(0)]], device const float* qa_w [[buffer(1)]],
                       device const float* kva_w [[buffer(2)]], device const float* ik_w [[buffer(3)]],
                       device const float* rope [[buffer(4)]], device bfloat* qa_frag [[buffer(5)]],
                       device uchar* lat [[buffer(6)]], device float* tokring [[buffer(7)]],
                       device float* iw [[buffer(8)]], constant KvArgs& a [[buffer(9)]],
                       uint m [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       ushort ntg [[threads_per_threadgroup]] LANE_PARAMS) {
  threadgroup float kr[64];
  threadgroup float lv[576];
  threadgroup float gs[18];
  device const float* yr = Y + size_t(m) * a.Nin;
  const uint pos = ROW_POS(m);
#ifdef LANES
  lat = lt[row_lane[m]].lat;
  if (a.sparse) tokring = lt[row_lane[m]].tokring;
#endif
  // the three norms' sums of squares (q_a, kv latent, indexer key) behind one barrier, each summed as
  // tg_sum does (simd sums, then the simdgroups in order from 0.0f)
  device const float* kv = yr + a.kvoff;
  device const float* ik = yr + a.ikoff;
  float s3[3] = {0.0f, 0.0f, 0.0f};
  for (uint i = tid; i < 1024; i += ntg) s3[0] = fma(yr[i], yr[i], s3[0]);
  for (uint i = tid; i < 512; i += ntg) s3[1] = fma(kv[i], kv[i], s3[1]);
  if (a.sparse) for (uint i = tid; i < 128; i += ntg) s3[2] = fma(ik[i], ik[i], s3[2]);
  threadgroup float sh3[3][32];
  {
    const ushort sgi = tid / 32, ns = (ntg + 31) / 32;
    for (uint j = 0; j < 3; j++) { const float t = simd_sum(s3[j]); if ((tid & 31) == 0) sh3[j][sgi] = t; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < 3; j++) { float t = 0.0f; for (ushort k = 0; k < ns; k++) t += sh3[j][k]; s3[j] = t; }
  }
  float r = rsqrt(s3[0] / 1024.0f + a.eps);
  for (uint i = tid; i < 1024; i += ntg) qa_frag[frag_off(m, i, 1024)] = bfloat(yr[i] * r * qa_w[i]);
  // kv latent
  r = rsqrt(s3[1] / 512.0f + a.eps);
  const uint row = a.ring ? pos % a.ring : pos;
  for (uint i = tid; i < 512; i += ntg) lv[i] = kv[i] * r * kva_w[i];
  device const float* rp = rope + size_t(m) * 64;
  if (tid < 32) {
    float x0 = kv[512 + 2 * tid], x1 = kv[512 + 2 * tid + 1];
    rope_pair(x0, x1, rp[tid], rp[32 + tid]);
    kr[2 * tid] = x0; kr[2 * tid + 1] = x1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 64) lv[512 + tid] = kr[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (a.kv8) {
    device uchar* lr = lat + size_t(row) * KV8_ROW;
    if (tid < 18) {
      float mx = 0.0f;
      for (uint i = 0; i < 32; i++) mx = max(mx, abs(lv[tid * 32 + i]));
      const half sc = half(mx / 127.0f);
      ((device half*)(lr + 576))[tid] = sc;
      gs[tid] = float(sc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 576; i += ntg) {  // i: physical position
      const float sc = gs[i >> 5];
      lr[i] = as_type<uchar>(char(sc > 0.0f ? clamp(rint(lv[lat_logic(i)] / sc), -127.0f, 127.0f) : 0.0f));
    }
  } else {
    device bfloat* lr = (device bfloat*)lat + size_t(row) * 576;
    for (uint i = tid; i < 576; i += ntg) lr[i] = bfloat(lv[lat_logic(i)]);
  }
  if (!a.sparse) return;
  r = rsqrt(s3[2] / 128.0f + a.eps);
  for (uint i = tid; i < 128; i += ntg) tokring[(pos % TOKR) * 128 + i] = ik[i] * r * ik_w[i];
  if (tid < 16) iw[m * 16 + tid] = yr[a.iwoff + tid] * a.iw_scale;
}

// From q_b's rows [M][nh*256 (+2048)]: q_nope per head -> the kb matmul's per-head frag
// [head][16 rows][192]; q_rope (rotated) -> qr [M][nh][64]; sparse: indexer q (last 64 of each head
// rotated by the long base) -> iq [M][16][128]. Grid (nh + 16, M), 64 threads.
struct QArgs { uint M, nh, Nq, sparse, rs; };
kernel void h4_q_prep(device const float* Y [[buffer(0)]], device const float* rope [[buffer(1)]],
                      device const float* rope_idx [[buffer(2)]], device bfloat* qn_frag [[buffer(3)]],
                      device float* qr [[buffer(4)]], device float* iq [[buffer(5)]], constant QArgs& a [[buffer(6)]],
                      uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
  const uint hh = tg.x, m = tg.y;
  device const float* yr = Y + size_t(m) * a.Nq;
  if (hh < a.nh) {
    device const float* q = yr + hh * 256;
    device bfloat* fr = qn_frag + size_t(hh) * a.rs * 192;
    for (uint i = tid; i < 192; i += 64) fr[frag_off(m, i, 192)] = bfloat(q[i]);
    if (tid < 32) {
      device const float* rp = rope + size_t(m) * 64;
      float x0 = q[192 + 2 * tid], x1 = q[192 + 2 * tid + 1];
      rope_pair(x0, x1, rp[tid], rp[32 + tid]);
      qr[(size_t(m) * a.nh + hh) * 64 + 2 * tid] = x0;
      qr[(size_t(m) * a.nh + hh) * 64 + 2 * tid + 1] = x1;
    }
    return;
  }
  if (!a.sparse) return;
  const uint ih = hh - a.nh;
  device const float* q = yr + a.nh * 256 + ih * 128;
  device float* o = iq + (size_t(m) * 16 + ih) * 128;
  o[tid] = q[tid];
  if (tid < 32) {
    device const float* rp = rope_idx + size_t(m) * 64;
    float x0 = q[64 + 2 * tid], x1 = q[64 + 2 * tid + 1];
    rope_pair(x0, x1, rp[tid], rp[32 + tid]);
    o[64 + 2 * tid] = x0;
    o[64 + 2 * tid + 1] = x1;
  }
}

// ------------------------------------------------------------------ indexer
// Pool the groups completed by this step's rows: group g = pos / 4 when pos % 4 == 3. The token keys
// sit in the ring (32 positions). One threadgroup (128) per row; rows that close no group return.
struct PoolArgs { uint M, p0; float eps; };
kernel void h4_idx_pool(device const float* tokring [[buffer(0)]], device const float* kvp [[buffer(1)]],
                        device const float* kgp [[buffer(2)]], device const float* ape [[buffer(3)]],
                        device const float* knorm [[buffer(4)]], device const float* rope_tab [[buffer(5)]],
                        device bfloat* bk [[buffer(6)]], constant PoolArgs& a [[buffer(7)]],
                        uint m [[threadgroup_position_in_grid]], ushort c [[thread_index_in_threadgroup]] LANE_PARAMS) {
  threadgroup float tk[4][128];
  threadgroup float pooled[128];
  threadgroup float sh[32];
  const uint pos = ROW_POS(m);
  if (pos % 4 != 3) return;
#ifdef LANES
  tokring = lt[row_lane[m]].tokring;
  bk = lt[row_lane[m]].bk;
#endif
  const uint g = pos / 4;
  for (uint s = 0; s < 4; s++) tk[s][c] = tokring[((4 * g + s) % TOKR) * 128 + c];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float v[4] = {0.0f, 0.0f, 0.0f, 0.0f}, gt[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  device const float* kvp_c = kvp + size_t(c) * 128;
  device const float* kgp_c = kgp + size_t(c) * 128;
  for (uint k = 0; k < 128; k++) {
    const float wv = kvp_c[k], wg = kgp_c[k];
    for (uint s = 0; s < 4; s++) {
      const float t = tk[s][k];
      v[s] = fma(wv, t, v[s]);
      gt[s] = fma(wg, t, gt[s]);
    }
  }
  float mx = -INFINITY;
  for (uint s = 0; s < 4; s++) {
    gt[s] += ape[s * 128 + c];
    mx = max(mx, gt[s]);
  }
  float den = 0.0f, num = 0.0f;
  for (uint s = 0; s < 4; s++) { const float e = exp(gt[s] - mx); den += e; num = fma(e, v[s], num); }
  const float p = num / den;
  const float r = rsqrt(tg_sum(p * p, sh, c, 128) / 128.0f + a.eps);
  pooled[c] = p * r * knorm[c];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // rope on the last 64 at the group's first position: the host table row for that position
  float o = pooled[c];
  if (c >= 64) {
    const uint j = (c - 64) / 2;
    device const float* rp = rope_tab + size_t(m) * 64;  // row m holds position 4g (host-filled)
    const float x0 = pooled[64 + 2 * j], x1 = pooled[64 + 2 * j + 1];
    o = (c & 1) ? x1 * rp[j] + x0 * rp[32 + j] : x0 * rp[j] - x1 * rp[32 + j];
  }
  bk[size_t(g) * 128 + c] = bfloat(o);
}

// scores[m][g] = sum_h w_h relu(q_h . k_g) for g < pos / 4 on the matrix units: per simdgroup
// S[16 heads][32 groups] = q . k over 4 k-steps of 32 dims (q as bf16 hi + lo, keys bf16), then
// sum_h w_h relu(S) across the lanes (256 groups per threadgroup, 8 simdgroups of 32).
struct ScoreArgs { uint M, p0, dense_max, gmax, row0; };
kernel void h4_idx_score_mma(device const float* iq [[buffer(0)]], device const float* iw [[buffer(1)]],
                             device const bfloat* bk [[buffer(2)]], device float* scores [[buffer(3)]],
                             constant ScoreArgs& a [[buffer(4)]],
                             uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort l [[thread_index_in_simdgroup]] LANE_PARAMS) {
  const uint mb = tg.y, m = a.row0 + mb, pos = ROW_POS(m);
  if (pos < a.dense_max) return;
#ifdef LANES
  bk = lt[row_lane[m]].bk;
#endif
  const uint G = pos / 4, g0 = tg.x * 256 + sg * 32;
  if (g0 >= G) return;
  const ushort m0 = ((l >> 1) & 3) + 4 * (l >> 4), n0 = 4 * (l & 1) + 8 * ((l >> 3) & 1);
  constexpr auto desc =
      matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto qh = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto cs = op.template get_destination_cooperative_tensor<decltype(qh), decltype(wb), float>();
  for (int i = 0; i < 16; i++) cs[i] = 0.0f;
  device const float* q = iq + size_t(m) * 2048;
  for (uint ks = 0; ks < 4; ks++) {
    for (int i = 0; i < 16; i++) {
      const uint h = m0 + 8 * ((i >> 2) & 1), d = ks * 32 + n0 + (i & 3) + 16 * (i >> 3);
      qh[i] = bfloat(q[h * 128 + d]);
    }
    for (int qq = 0; qq < 4; qq++) {
      const uint g = g0 + m0 + 8 * qq;
      for (int hh = 0; hh < 2; hh++) {
        const bfloat4 b4 = g < G ? *(device const bfloat4*)(bk + size_t(g) * 128 + ks * 32 + n0 + 16 * hh) : bfloat4(bfloat(0.0f));
        for (int e = 0; e < 4; e++) wb[hh * 16 + qq * 4 + e] = b4[e];
      }
    }
    op.run(qh, wb, cs);
  }
  const float w0 = iw[m * 16 + m0], w1 = iw[m * 16 + m0 + 8];
  float sc[8];
  for (int j = 0; j < 8; j++) {  // group n0 + (j & 3) + 16 (j >> 2): elements i = (j & 3) + 8 (j >> 2) (+4: head + 8)
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
  if (m0 != 0) return;
  for (int j = 0; j < 8; j++) {
    const uint g = g0 + n0 + (j & 3) + 16 * (j >> 2);
    if (g < G) scores[size_t(mb) * a.gmax + g] = sc[j];
  }
}

// Top (511) groups by score among g < G, ties to the lower index; emits the selected token positions
// in ascending order: the picked groups' 4 tokens each, then the query's own group up to pos.
// One threadgroup (1024) per row. Rows inside the dense range emit 0..pos.
static inline uint fkey(float f) { const uint u = as_type<uint>(f); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
struct TopArgs { uint M, p0, dense_max, gmax, ktop, selmax, row0; };
kernel void h4_idx_topk(device const float* scores [[buffer(0)]], device uint* sel [[buffer(1)]],
                        device uint* nsel [[buffer(2)]], device uint* mark [[buffer(3)]], constant TopArgs& a [[buffer(4)]],
                        uint mb [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                        ushort ntg [[threads_per_threadgroup]] LANE_PARAMS) {
  const uint m = a.row0 + mb;
  threadgroup atomic_uint hist[256];
  threadgroup uint pref, need, cnt_s[1024 / 32];
  threadgroup uint base_out;
  threadgroup uint cand_k[2048];
  threadgroup atomic_uint cand_n;
  threadgroup uint use_cand;
  const uint pos = ROW_POS(m);
  device uint* so = sel + size_t(m) * a.selmax;
  if (pos < a.dense_max) {
    for (uint i = tid; i <= pos; i += ntg) so[i] = i;
    if (tid == 0) nsel[m] = pos + 1;
    return;
  }
  const uint G = pos / 4;
  device const float* sc = scores + size_t(mb) * a.gmax;
  device uint* mk = mark + size_t(mb) * (a.gmax / 32 + 1);
  const uint K = min(a.ktop, G);
  // radix select of the K-th largest key, 8 bits at a time from the top;
  // when pass 0's winning bucket has <= 2048 items, compact them into TG memory on pass 1
  // so passes 2 & 3 do zero DRAM reads.
  if (tid == 0) { pref = 0; need = K; use_cand = 0; atomic_store_explicit(&cand_n, 0u, memory_order_relaxed); }
  uint mask = 0;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (uint i = tid; i < 256; i += ntg) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint p = pref, nd = need;
    const uint uc = use_cand;
    if (shift <= 8 && uc) {
      const uint nc = min(2048u, atomic_load_explicit(&cand_n, memory_order_relaxed));
      for (uint i = tid; i < nc; i += ntg) {
        const uint k = cand_k[i];
        if ((k & mask) == p) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
      }
    } else if (shift == 16 && uc) {
      for (uint g = tid; g < G; g += ntg) {
        const uint k = fkey(sc[g]);
        if ((k & mask) == p) {
          atomic_fetch_add_explicit(&hist[(k >> 16) & 255u], 1u, memory_order_relaxed);
          const uint ci = atomic_fetch_add_explicit(&cand_n, 1u, memory_order_relaxed);
          if (ci < 2048u) cand_k[ci] = k;
        }
      }
    } else {
      for (uint g = tid; g < G; g += ntg) {
        const uint k = fkey(sc[g]);
        if ((k & mask) == p) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 256) {
      const uint h = atomic_load_explicit(&hist[255 - tid], memory_order_relaxed);
      const uint inc = simd_prefix_inclusive_sum(h);
      if ((tid & 31) == 31) cnt_s[tid / 32] = inc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 256) {
      const uint sg = tid / 32;
      uint base = 0;
      for (uint s = 0; s < sg; s++) base += cnt_s[s];
      const uint h = atomic_load_explicit(&hist[255 - tid], memory_order_relaxed);
      const uint hi = base + simd_prefix_inclusive_sum(h);
      const uint lo = hi - h;
      if (lo < nd && hi >= nd) {
        pref = p | ((255u - tid) << shift);
        need = nd - lo;
        if (shift == 24) use_cand = (h <= 2048u) ? 1u : 0u;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mask |= 255u << shift;
  }
  const uint thr = pref;      // the K-th largest key
  const uint ties = need;     // how many keys equal to thr to take (lowest indices first)
  // mark: keys > thr, plus the first `ties` keys == thr in index order. Each thread owns a
  // contiguous range of 32-group words; a prefix over the threads' == counts ranks the ties.
  const uint nw = (G + 31) / 32;
  const uint per = (nw + ntg - 1) / ntg, w0 = tid * per, w1 = min(nw, w0 + per);
  const uint lane = tid & 31, sg = tid / 32;
  uint neq = 0;
  for (uint wi = w0; wi < w1; wi++) {
    uint bits = 0;
    for (uint b = 0; b < 32; b++) {
      const uint g = wi * 32 + b;
      if (g >= G) break;
      const uint k = fkey(sc[g]);
      if (k > thr) bits |= 1u << b;
      neq += k == thr;
    }
    mk[wi] = bits;
  }
  {
    const uint inc = simd_prefix_inclusive_sum(neq);
    if (lane == 31) cnt_s[sg] = inc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      uint s = 0;
      for (uint i = 0; i < (ntg + 31) / 32; i++) { const uint t = cnt_s[i]; cnt_s[i] = s; s += t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint rank = cnt_s[sg] + inc - neq;  // ties before this thread's range
    if (neq > 0 && rank < ties) {
      for (uint wi = w0; wi < w1 && rank < ties; wi++)
        for (uint b = 0; b < 32 && rank < ties; b++) {
          const uint g = wi * 32 + b;
          if (g < G && fkey(sc[g]) == thr) { mk[wi] |= 1u << b; rank++; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup_barrier(mem_flags::mem_device);
  // ascending compaction: per-thread counts over the same word ranges, then a prefix
  uint c = 0;
  for (uint wi = w0; wi < w1; wi++) c += popcount(mk[wi]);
  uint incl = simd_prefix_inclusive_sum(c);
  if (lane == 31) cnt_s[sg] = incl;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint s = 0;
    for (uint i = 0; i < (ntg + 31) / 32; i++) { const uint t = cnt_s[i]; cnt_s[i] = s; s += t; }
    base_out = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint o = cnt_s[sg] + incl - c;
  for (uint wi = w0; wi < w1; wi++) {
    uint bits = mk[wi];
    while (bits) {
      const uint b = ctz(bits);
      bits &= bits - 1;
      const uint g = wi * 32 + b;
      for (uint t = 0; t < 4; t++) so[4 * o + t] = 4 * g + t;
      o++;
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  const uint tot = base_out * 4;
  for (uint t = tid; t <= pos - 4 * G; t += ntg) so[tot + t] = 4 * G + t;
  if (tid == 0) nsel[m] = tot + (pos - 4 * G + 1);
}

// ------------------------------------------------------------------ attention (matrix units)
// MLA as MQA (16x32x32 bf16 cooperative tensors, f32 accumulate): a threadgroup takes 16 heads of one
// row as the M side and walks the row's keys in 32-key tiles. 8 simdgroups: S = Q K^T over the 18
// 32-dim k-steps of the 576-wide latent row, partials summed in sg order in threadgroup memory;
// every simdgroup runs online softmax and accumulates O for dims sg*64..+64. Grid (nh / 16, M, splits), 256 threads.
struct AttnArgs { uint M, nh, p0, sliding, window, ring, splits, selmax; float scale; uint rs; uint kv8; uint direct, rsf; };
kernel void h4_attn_mma(device const float* qlat [[buffer(0)]], device const float* qr [[buffer(1)]],
                        device const uchar* lat [[buffer(2)]], device const uint* sel [[buffer(3)]],
                        device const uint* nsel [[buffer(4)]], device float* part [[buffer(5)]],
                        constant AttnArgs& a [[buffer(6)]],
                        device const float* sinks [[buffer(7)]], device bfloat* ofrag [[buffer(8)]],
                        uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort l [[thread_index_in_simdgroup]] LANE_PARAMS) {
  threadgroup float Sp[8][16][32];
  const uint m = tg.y, h0 = tg.x * 16, split = tg.z;
  const uint pos = ROW_POS(m);
#ifdef LANES
  lat = lt[row_lane[m]].lat;
#endif
#ifdef ATT_SLIDING
  constexpr bool sliding = (ATT_SLIDING != 0);
#else
  const bool sliding = a.sliding != 0;
#endif
#ifdef ATT_KV8
  constexpr uint kv8_ = ATT_KV8;
#else
  const uint kv8_ = a.kv8;
#endif
  uint nk, k0;
  if (sliding) { k0 = pos + 1 > a.window ? pos + 1 - a.window : 0; nk = pos + 1 - k0; }
  else { k0 = 0; nk = nsel[m]; }
  const uint per = (nk + a.splits - 1) / a.splits, j_begin = split * per, j_end = min(nk, j_begin + per);
  auto keyof = [&](uint j) -> uint {
    if (sliding) return (k0 + j) & (a.ring - 1u);
    return sel[size_t(m) * a.selmax + j];
  };
  const ushort m0 = ((l >> 1) & 3) + 4 * (l >> 4), n0 = 4 * (l & 1) + 8 * ((l >> 3) & 1);
  constexpr auto desc =
      matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
#define LEFT op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>()
  auto qh0 = LEFT; auto qh1 = LEFT; auto qh2 = LEFT;
  auto xa = LEFT;
#undef LEFT
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto cs = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), float>();
  auto o0 = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), float>();
  auto o1 = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), float>();
  const uint nks = sg < 2 ? 3 : 2;  // k-steps sg, sg + 8, sg + 16 (< 18)
  for (uint t = 0; t < nks; t++) {
    const uint ks = sg + 8 * t;
    for (int i = 0; i < 16; i++) {
      const uint hh = h0 + m0 + 8 * ((i >> 2) & 1), d = lat_logic(ks * 32 + n0 + (i & 3) + 16 * (i >> 3));
      const float v = d < 512 ? qlat[(size_t(hh) * a.rs + m) * 512 + d] : qr[(size_t(m) * a.nh + hh) * 64 + d - 512];
      const bfloat hi = bfloat(v);
      if (t == 0) qh0[i] = hi; else if (t == 1) qh1[i] = hi; else qh2[i] = hi;
    }
  }
  for (int i = 0; i < 16; i++) { o0[i] = 0.0f; o1[i] = 0.0f; }
  float mx[2] = {-INFINITY, -INFINITY}, lsum[2] = {0.0f, 0.0f};
  for (uint j0 = j_begin; j0 < j_end; j0 += 32) {
    for (int i = 0; i < 16; i++) cs[i] = 0.0f;
    uint tk[4], tv[8];
    for (int qq = 0; qq < 4; qq++) { const uint j = j0 + m0 + 8 * qq; tk[qq] = j < j_end ? keyof(j) : 0xFFFFFFFFu; }
    for (int hh = 0; hh < 2; hh++)
      for (int e = 0; e < 4; e++) { const uint j = j0 + n0 + e + 16 * hh; tv[hh * 4 + e] = j < j_end ? keyof(j) : 0xFFFFFFFFu; }
    for (uint t = 0; t < nks; t++) {
      const uint d = (sg + 8 * t) * 32 + n0;
      for (int qq = 0; qq < 4; qq++)
        for (int hh = 0; hh < 2; hh++) {
          const bfloat4 b4 = tk[qq] != 0xFFFFFFFFu ? latv4(lat, tk[qq], d + 16 * hh, kv8_) : bfloat4(bfloat(0.0f));
          for (int e = 0; e < 4; e++) wb[hh * 16 + qq * 4 + e] = b4[e];
        }
      if (t == 0) op.run(qh0, wb, cs); else if (t == 1) op.run(qh1, wb, cs); else op.run(qh2, wb, cs);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < 16; i++) Sp[sg][i][l] = cs[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s[16];
    for (int i = 0; i < 16; i++) {
      const uint j = j0 + n0 + (i & 3) + 16 * (i >> 3);
      float v = 0.0f;
      for (uint q = 0; q < 8; q++) v += Sp[q][i][l];
      s[i] = j < j_end ? v * a.scale : -INFINITY;
    }
    float c[2];
    for (int r = 0; r < 2; r++) {
      float mb = -INFINITY;
      for (int i = 0; i < 16; i++) if (((i >> 2) & 1) == r) mb = max(mb, s[i]);
      mb = max(mb, simd_shuffle_xor(mb, 1));
      mb = max(mb, simd_shuffle_xor(mb, 8));
      const float mn = max(mx[r], mb);
      c[r] = exp(mx[r] - mn);
      float bs = 0.0f;
      for (int i = 0; i < 16; i++)
        if (((i >> 2) & 1) == r) { s[i] = exp(s[i] - mn); bs += s[i]; }
      bs += simd_shuffle_xor(bs, 1);
      bs += simd_shuffle_xor(bs, 8);
      lsum[r] = lsum[r] * c[r] + bs;
      mx[r] = mn;
    }
    for (int i = 0; i < 16; i++) { const float cc = c[(i >> 2) & 1]; o0[i] *= cc; o1[i] *= cc; }
    for (int i = 0; i < 16; i++) xa[i] = bfloat(s[i]);
    // O += P V: right operand V (k = key, n = dim); lane element i -> key n0 + (i&3) + 16(i>>4), dim m0 + 8((i>>2)&3)
    for (int nc = 0; nc < 2; nc++) {
      const uint d = sg * 64 + nc * 32 + 4 * m0;  // physical: logical dims m0 + 8 jq of this chunk
      for (int hh = 0; hh < 2; hh++)
        for (int e = 0; e < 4; e++) {
          const uint kk = tv[hh * 4 + e];
          const bfloat4 b4 = kk != 0xFFFFFFFFu ? latv4(lat, kk, d, kv8_) : bfloat4(bfloat(0.0f));
          for (int jq = 0; jq < 4; jq++) wb[hh * 16 + jq * 4 + e] = b4[jq];
        }
      if (nc == 0) op.run(xa, wb, o0); else op.run(xa, wb, o1);
    }
  }
  if (a.direct) {
#pragma clang fp contract(off)  // h4_attn_merge rounds w * l before adding the sink term (a simd_sum between)
    for (int r = 0; r < 2; r++) {
      const uint hh = h0 + m0 + 8 * r;
      const float sk = sinks[hh];
      const float M_ = max(lsum[r] > 0.0f ? mx[r] : -INFINITY, sk);
      const float w = lsum[r] > 0.0f ? exp(mx[r] - M_) : 0.0f;
      const float inv = 1.0f / (w * lsum[r] + exp(sk - M_));
      device bfloat* fr = ofrag + size_t(hh) * a.rsf * 512;
      for (int i = 0; i < 16; i++) {
        if (((i >> 2) & 1) != r) continue;
        const uint dn = n0 + (i & 3) + 16 * (i >> 3);
        const float v0 = w != 0.0f ? 0.0f + w * o0[i] : 0.0f, v1 = w != 0.0f ? 0.0f + w * o1[i] : 0.0f;
        fr[frag_off(m, sg * 64 + dn, 512)] = bfloat(v0 * inv);
        fr[frag_off(m, sg * 64 + 32 + dn, 512)] = bfloat(v1 * inv);
      }
    }
    return;
  }
  for (int r = 0; r < 2; r++) {
    const uint hh = h0 + m0 + 8 * r;
    device float* pp = part + ((size_t(m) * a.nh + hh) * a.splits + split) * 514;
    if (sg == 0 && n0 == 0) { pp[0] = mx[r]; pp[1] = lsum[r]; }
    for (int i = 0; i < 16; i++) {
      if (((i >> 2) & 1) != r) continue;
      const uint dn = n0 + (i & 3) + 16 * (i >> 3);
      pp[2 + sg * 64 + dn] = o0[i];
      pp[2 + sg * 64 + 32 + dn] = o1[i];
    }
  }
}

// merge the splits with the sink and write the per-head frag for vb: [head][16 rows][512]
struct MergeArgs { uint M, nh, splits, rs; };
kernel void h4_attn_merge(device const float* part [[buffer(0)]], device const float* sinks [[buffer(1)]],
                          device bfloat* ofrag [[buffer(2)]], constant MergeArgs& a [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup float wsp[64];
  threadgroup float inv;
  const uint hh = tg.x, m = tg.y;
  device const float* pp = part + (size_t(m) * a.nh + hh) * a.splits * 514;
  if (tid < 32) {  // split weights once: lane = split
    const bool on = tid < a.splits;
    const float ms = on ? pp[tid * 514] : -INFINITY, ls = on ? pp[tid * 514 + 1] : 0.0f;
    const float M_ = max(simd_max(ls > 0.0f ? ms : -INFINITY), sinks[hh]);
    const float w = ls > 0.0f ? exp(ms - M_) : 0.0f;
    const float den = simd_sum(w * ls) + exp(sinks[hh] - M_);
    if (on) wsp[tid] = w;
    if (tid == 0) inv = 1.0f / den;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device bfloat* fr = ofrag + size_t(hh) * a.rs * 512;
  for (uint d = tid; d < 512; d += 128) {
    float v = 0.0f;
    for (uint s = 0; s < a.splits; s++) if (wsp[s] != 0.0f) v += wsp[s] * pp[s * 514 + 2 + d];
    fr[frag_off(m, d, 512)] = bfloat(v * inv);
  }
}

// o [M][nh*256] = vb_out[head][m][d] * sigmoid(gate[m][head*256 + d]) -> frag for o_proj
struct GateArgs { uint M, nh, rs; };
kernel void h4_gate(device const float* vb [[buffer(0)]], device const float* gate [[buffer(1)]],
                    device bfloat* ofrag [[buffer(2)]], constant GateArgs& a [[buffer(3)]],
                    uint2 g [[thread_position_in_grid]]) {
  const uint i = g.x, m = g.y, N = a.nh * 256;
  if (i >= N || m >= a.M) return;
  const uint hh = i / 256, d = i % 256;
  const float v = vb[(size_t(hh) * a.rs + m) * 256 + d] * sigm(gate[size_t(m) * N + i]);
  ofrag[frag_off(m, i, N)] = bfloat(v);
}

// ------------------------------------------------------------------ MoE
// Router (sigmoid + correction bias, top-k, renormalised, scaled), the latent input frag and the
// shared expert's activation. One threadgroup (512) per row. moe.in rows: router | ldn | sgu.
struct RouteArgs { uint M, E, topk, Nin, lat, I; float scale; };
kernel void h4_route(device const float* Y [[buffer(0)]], device const float* bias [[buffer(1)]],
                     device uint* topi [[buffer(2)]], device float* topw [[buffer(3)]],
                     device bfloat* lat_frag [[buffer(4)]], device bfloat* mo_frag [[buffer(5)]],
                     constant RouteArgs& a [[buffer(6)]],
                     uint m [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                     ushort ntg [[threads_per_threadgroup]]) {
  threadgroup float sc[512], ch[512];
  threadgroup uint pick[32];
  device const float* yr = Y + size_t(m) * a.Nin;
  for (uint e = tid; e < a.E; e += ntg) { const float s = sigm(yr[e]); sc[e] = s; ch[e] = s + bias[e]; }
  for (uint i = tid; i < a.lat; i += ntg) lat_frag[frag_off(m, i, a.lat)] = bfloat(yr[a.E + i]);
  device const float* sg = yr + a.E + a.lat;
  for (uint c = tid; c < a.I; c += ntg) {
    const uint f = c / 16, j = c % 16;
    mo_frag[frag_off(m, a.lat + c, a.lat + a.I)] = bfloat(silu(sg[32 * f + j]) * sg[32 * f + 16 + j]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // top-k by rank: expert e's rank = experts scoring higher, ties to the lower index; ranks < k are picked
  // in rank order, the same picks and order as repeated argmax
  if (a.E == 512 && ntg == 512 && a.topk <= 16) {
    // two stages, same picks: each simdgroup ranks its 32 experts and keeps its top 16 (every global top-16
    // expert is in its simdgroup's top 16), then a candidate's rank among the 256 candidates is its global
    // rank whenever that is < 16, and >= 16 otherwise
    threadgroup float cv[256];
    threadgroup uint ce[256], cr[512];
    const uint sgi = tid / 32;
    const float v = ch[tid];
    uint r = 0;
    for (ushort j = 0; j < 32; j++) {
      const float u = simd_shuffle(v, j);
      r += (u > v) || (u == v && sgi * 32 + j < tid);
    }
    if (r < 16) { cv[sgi * 16 + r] = v; ce[sgi * 16 + r] = tid; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint c = tid & 255, h = tid >> 8;
    const float vc = cv[c];
    const uint ec = ce[c];
    uint rr = 0;
    for (uint f = h * 128; f < h * 128 + 128; f++) { const float u = cv[f]; rr += (u > vc) || (u == vc && ce[f] < ec); }
    cr[tid] = rr;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 256) { const uint R = cr[tid] + cr[tid + 256]; if (R < a.topk) pick[R] = ec; }
  } else
  for (uint e = tid; e < a.E; e += ntg) {
    const float v = ch[e];
    uint r = 0;
    for (uint f = 0; f < a.E; f++) { const float u = ch[f]; r += (u > v) || (u == v && f < e); }
    if (r < a.topk) pick[r] = e;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float s = 0.0f;
    for (uint k = 0; k < a.topk; k++) s += sc[pick[k]];
    for (uint k = 0; k < a.topk; k++) {
      topi[m * a.topk + k] = pick[k];
      topw[m * a.topk + k] = sc[pick[k]] / (s + 1e-20f) * a.scale;
    }
  }
}

// distinct experts of the step's rows, ascending -> slots, nslots; slot_of[e] for the combine;
// rmask[slot * 4] = 16-bit bitmask of rows routed to that expert (for mm2 a.rm)
struct PlanArgs { uint M, topk, E; };
kernel void h4_plan(device const uint* topi [[buffer(0)]], device uint* slots [[buffer(1)]],
                    device uint* nslots [[buffer(2)]], device uint* slot_of [[buffer(3)]],
                    constant PlanArgs& a [[buffer(4)]], device uint* rmask [[buffer(5)]],
                    ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint used[512];
  for (uint e = tid; e < a.E; e += 512) atomic_store_explicit(&used[e], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < a.M * a.topk; i += 512)
    atomic_fetch_or_explicit(&used[topi[i]], 1u << (i / a.topk), memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup uint sgsum[16];
  const uint mask = tid < a.E ? atomic_load_explicit(&used[tid], memory_order_relaxed) : 0u;
  const uint u = mask ? 1u : 0u;
  const uint inc = simd_prefix_inclusive_sum(u);
  if ((tid & 31) == 31) sgsum[tid / 32] = inc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint base = 0;
  for (uint g = 0; g < tid / 32; g++) base += sgsum[g];
  const uint at = base + inc - u;
  if (tid < a.E) {
    if (u) { slots[at] = tid; rmask[at * 4] = mask; }
    slot_of[tid] = u ? at : 0xFFFFFFFFu;
  }
  if (tid == 511) nslots[0] = base + inc;
}

// routed latent output [m][c] = sum_k w_k * down[slot(e_k)][m][c] -> mo frag columns 0..lat
struct CombArgs { uint M, topk, lat, ystride; };
kernel void h4_combine(device const float* yd [[buffer(0)]], device const uint* topi [[buffer(1)]],
                       device const float* topw [[buffer(2)]], device const uint* slot_of [[buffer(3)]],
                       device bfloat* mo_frag [[buffer(4)]], constant CombArgs& a [[buffer(5)]],
                       uint2 g [[thread_position_in_grid]]) {
  const uint c = g.x, m = g.y;
  if (c >= a.lat || m >= a.M) return;
  float s = 0.0f;
  for (uint k = 0; k < a.topk; k++) {
    const uint sl = slot_of[topi[m * a.topk + k]];
    s = fma(topw[m * a.topk + k], yd[size_t(sl) * a.ystride + size_t(m) * a.lat + c], s);
  }
  mo_frag[frag_off(m, c, 2 * a.lat)] = bfloat(s);
}

// ------------------------------------------------------------------ Block-128 WHT on MoE act frags
struct WhtArgs { uint M, I; };
static inline void wht128_inplace(device bfloat* base, uint row, uint k0, uint I, ushort l) {
  // k0 is a multiple of 128, so (k0 + 4*l) is 4-aligned and frag_off(+0..+3) are 4 contiguous 8B-aligned bfloats.
  device bfloat4* p = (device bfloat4*)(base + frag_off(row, k0 + 4u * l, I));
  const bfloat4 v = *p;
  float a0 = float(v.x), a1 = float(v.y), a2 = float(v.z), a3 = float(v.w);
  // Stages 1 & 2 (strides 1, 2 in registers)
  { float t0 = a0 + a1, t1 = a0 - a1, t2 = a2 + a3, t3 = a2 - a3; a0 = t0; a1 = t1; a2 = t2; a3 = t3; }
  { float t0 = a0 + a2, t2 = a0 - a2, t1 = a1 + a3, t3 = a1 - a3; a0 = t0; a1 = t1; a2 = t2; a3 = t3; }
  // Stages 3..7 (strides 4, 8, 16, 32, 64 across the 32-lane simdgroup)
#pragma clang loop unroll(full)
  for (ushort mask = 1; mask <= 16; mask <<= 1) {
    const float sgn = (l & mask) ? -1.0f : 1.0f;
    const float b0 = simd_shuffle_xor(a0, mask);
    const float b1 = simd_shuffle_xor(a1, mask);
    const float b2 = simd_shuffle_xor(a2, mask);
    const float b3 = simd_shuffle_xor(a3, mask);
    a0 = fma(sgn, a0, b0);
    a1 = fma(sgn, a1, b1);
    a2 = fma(sgn, a2, b2);
    a3 = fma(sgn, a3, b3);
  }
  const float inv = 0.08838834764831845f;  // 1 / sqrt(128)
  *p = bfloat4(bfloat(a0 * inv), bfloat(a1 * inv), bfloat(a2 * inv), bfloat(a3 * inv));
}
// Prefill: act is compact frag_off(row, c, I) over row in [0, M*topk). 4 SGs (128 threads) = 4 rows/TG.
// Grid (I/128, ceil(M*topk / 4)), 128 threads.
kernel void h4_wht128_pf(device bfloat* act [[buffer(0)]], constant WhtArgs& a [[buffer(1)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort l [[thread_index_in_simdgroup]]) {
  const uint row = tg.y * 4u + sg;
  if (row < a.M) wht128_inplace(act, row, tg.x * 128u, a.I, l);
}
// Decode/verify: act is [smax][16 * I] frags; 1 SG per (block128, active slot) walks only set bits of rmask.
// Grid (I/128, smax), 32 threads.
kernel void h4_wht128_dec(device bfloat* act [[buffer(0)]], device const uint* nslots [[buffer(1)]],
                          device const uint* rmask [[buffer(2)]], constant WhtArgs& a [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]], ushort l [[thread_index_in_threadgroup]]) {
  const uint sl = tg.y;
  if (sl >= *nslots) return;
  uint mask = rmask[sl * 4u] & ((a.M >= 32u) ? 0xFFFFFFFFu : ((1u << a.M) - 1u));
  device bfloat* base = act + size_t(sl) * (a.I >> 5) * 512u;
  const uint k0 = tg.x * 128u;
  while (mask) {
    const uint m = ctz(mask);
    mask &= mask - 1u;
    wht128_inplace(base, m, k0, a.I, l);
  }
}

// ------------------------------------------------------------------ Engram
static inline float fp8_e4m3(uchar b) {
  const uint s = b >> 7, e = (b >> 3) & 15u, mnt = b & 7u;
  float v;
  if (e == 0) v = float(mnt) * 0.001953125f;            // 2^-9 subnormal step
  else if (e == 15 && mnt == 7) v = NAN;
  else v = ldexp(1.0f + float(mnt) * 0.125f, int(e) - 7);
  return s ? -v : v;
}
// rows [M][16][96] fp8 -> per-head rms * w -> frag [M][1536]. Grid (16, M), 96 threads.
struct EngRowArgs { uint M; float eps; };
kernel void h4_eng_rows(device const uchar* rows [[buffer(0)]], device const float* lnorm [[buffer(1)]],
                        device bfloat* frag [[buffer(2)]], constant EngRowArgs& a [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup float sh[32];
  const uint hh = tg.x, m = tg.y;
  const float v = fp8_e4m3(rows[(size_t(m) * 16 + hh) * 96 + tid]);
  const float r = rsqrt(tg_sum(v * v, sh, tid, 96) / 96.0f + a.eps);
  frag[frag_off(m, hh * 96 + tid, 1536)] = bfloat(v * r * lnorm[hh * 96 + tid]);
}
// rows [M][16][60] affine u4_g32 (48B packed nibbles + 3 x half2(scale, bias)) -> per-head rms * w -> frag [M][1536].
kernel void h4_eng_rows_q4(device const uchar* rows [[buffer(0)]], device const float* lnorm [[buffer(1)]],
                           device bfloat* frag [[buffer(2)]], constant EngRowArgs& a [[buffer(3)]],
                           uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup float sh[32];
  const uint hh = tg.x, m = tg.y;
  device const uchar* rp = rows + (size_t(m) * 16 + hh) * 60;
  const uchar b = rp[tid >> 1];
  const uint code = (tid & 1) ? (b >> 4) : (b & 15u);
  const half2 sb = ((device const half2*)(rp + 48))[tid >> 5];
  const float v = fma(float(code), float(sb.x), float(sb.y));
  const float r = rsqrt(tg_sum(v * v, sh, tid, 96) / 96.0f + a.eps);
  frag[frag_off(m, hh * 96 + tid, 1536)] = bfloat(v * r * lnorm[hh * 96 + tid]);
}
// h_s += (gate_s + hcs_s) * main, gate_s = sigmoid(signed sqrt(rms(k_s) . rms(h_s) / sqrt(H)))
struct EngMixArgs { uint M, H; float eps; };
kernel void h4_eng_mix(device float* h [[buffer(0)]], device const float* Y [[buffer(1)]],
                       device const float* bias [[buffer(2)]], device const float* nk [[buffer(3)]],
                       device const float* nq [[buffer(4)]], device const float* gamma [[buffer(5)]],
                       device const float* hcs [[buffer(6)]], device const uint* live [[buffer(7)]],
                       constant EngMixArgs& a [[buffer(8)]],
                       uint m [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       ushort ntg [[threads_per_threadgroup]]) {
  threadgroup float shp[13][32];
  threadgroup float tot13[13];
  if (!live[m]) return;
  const uint H = a.H;
  device const float* y = Y + size_t(m) * 5 * H;
  device float* hr = h + size_t(m) * 4 * H;
  float acc[13];
  for (uint i = 0; i < 13; i++) acc[i] = 0.0f;
  for (uint c = tid; c < H; c += ntg) {
    const float v = y[4 * H + c] + bias[4 * H + c];
    acc[0] = fma(v, v, acc[0]);
    for (uint s = 0; s < 4; s++) {
      const float k = y[s * H + c] + bias[s * H + c], q = hr[s * H + c];
      acc[1 + 3 * s] = fma(k, k, acc[1 + 3 * s]);
      acc[2 + 3 * s] = fma(q, q, acc[2 + 3 * s]);
      acc[3 + 3 * s] = fma(k * nk[s * H + c], q * nq[s * H + c], acc[3 + 3 * s]);
    }
  }
  const ushort sgi = tid / 32, ns = (ntg + 31) / 32;
  for (uint i = 0; i < 13; i++) {
    const float t = simd_sum(acc[i]);
    if ((tid & 31) == 0) shp[i][sgi] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 13) {
    float t = 0.0f;
    for (ushort k = 0; k < ns; k++) t += shp[tid][k];
    tot13[tid] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float vr = rsqrt(tot13[0] / float(H) + a.eps);
  float g[4];
  for (uint s = 0; s < 4; s++) {
    const float K = tot13[1 + 3 * s], Q = tot13[2 + 3 * s], D = tot13[3 + 3 * s];
    float d = D * rsqrt(K / float(H) + a.eps) * rsqrt(Q / float(H) + a.eps) * rsqrt(float(H));
    d = sqrt(max(abs(d), 1e-6f)) * (d >= 0.0f ? 1.0f : -1.0f);
    g[s] = sigm(d);
  }
  for (uint c = tid; c < H; c += ntg) {
    const float main = gamma[c] * (y[4 * H + c] + bias[4 * H + c]) * vr;
    for (uint s = 0; s < 4; s++) hr[s * H + c] = fma(g[s] + hcs[s * H + c], main, hr[s * H + c]);
  }
}

// ------------------------------------------------------------------ argmax
struct ArgArgs { uint M, N; };
kernel void h4_argmax(device const float* y [[buffer(0)]], device uint* out [[buffer(1)]],
                      constant ArgArgs& a [[buffer(2)]], uint m [[threadgroup_position_in_grid]],
                      ushort tid [[thread_index_in_threadgroup]], ushort ntg [[threads_per_threadgroup]]) {
  threadgroup float bv[32];
  threadgroup uint bi[32];
  device const float* r = y + size_t(m) * a.N;
  float v = -INFINITY;
  uint idx = 0;
  for (uint i = tid; i < a.N; i += ntg) if (r[i] > v) { v = r[i]; idx = i; }
  for (ushort off = 16; off > 0; off >>= 1) {
    const float ov = simd_shuffle_down(v, off);
    const uint oi = simd_shuffle_down(idx, off);
    if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
  }
  if ((tid & 31) == 0) { bv[tid / 32] = v; bi[tid / 32] = idx; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float b = bv[0]; uint bix = bi[0];
    for (uint s = 1; s < ntg / 32; s++) if (bv[s] > b || (bv[s] == b && bi[s] < bix)) { b = bv[s]; bix = bi[s]; }
    out[m] = bix;
  }
}

// ------------------------------------------------------------------ prefill MoE planning
// Per-expert row lists for mmpg. slots = active experts ascending, roff/rcnt per slot, rlist = the
// expert's rows, cidx[m * topk + k] = compact row of that pick. Counts and placement through
// threadgroup atomics over all picks (1024 threads), so the order of rows inside an expert's list
// is arbitrary: each compact row is computed on its own, so no output bit depends on it.
struct ListArgs { uint M, topk, E, tile_rows, gx_gu, gx_dn; };
kernel void h4_plan_lists(device const uint* topi [[buffer(0)]], device uint* slots [[buffer(1)]],
                          device uint* nslots [[buffer(2)]], device uint* roff [[buffer(3)]],
                          device uint* rcnt [[buffer(4)]], device uint* rlist [[buffer(5)]],
                          device uint* cidx [[buffer(6)]], constant ListArgs& a [[buffer(7)]],
                          device uint* wtile [[buffer(8)]], device uint* ind [[buffer(9)]],
                          ushort tid [[thread_index_in_threadgroup]], ushort ntg [[threads_per_threadgroup]]) {
  threadgroup atomic_uint cnt[512];
  threadgroup uint off[512], sgs[32], sgt[32];
  const uint n = a.M * a.topk;
  for (uint e = tid; e < 512; e += ntg) atomic_store_explicit(&cnt[e], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < n; i += ntg) atomic_fetch_add_explicit(&cnt[topi[i]], 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // exclusive prefix of counts (rows), active experts (slots), and row-tiles (wtile)
  const uint e = tid;
  const uint c = e < a.E ? atomic_load_explicit(&cnt[e], memory_order_relaxed) : 0u;
  const uint act = c ? 1u : 0u;
  const uint nt = c ? (c + a.tile_rows - 1) / a.tile_rows : 0u;
  const uint pk = (c << 10) | act;  // rows <= 16 * 1024 < 2^22, slots <= 512 < 2^10
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
// routed latent output of prefill rows from the compact down outputs
struct CombCArgs { uint M, topk, lat; };
kernel void h4_combine_c(device const float* yd [[buffer(0)]], device const uint* cidx [[buffer(1)]],
                         device const float* topw [[buffer(2)]], device bfloat* mo_frag [[buffer(3)]],
                         constant CombCArgs& a [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  const uint c = g.x, m = g.y;
  if (c >= a.lat || m >= a.M) return;
  float s = 0.0f;
  for (uint k = 0; k < a.topk; k++)
    s = fma(topw[m * a.topk + k], yd[size_t(cidx[m * a.topk + k]) * a.lat + c], s);
  mo_frag[frag_off(m, c, 2 * a.lat)] = bfloat(s);
}

// ------------------------------------------------------------------ DFly2 drafter
// x[m][c] += (g ? g[c] : 1) * y[m][c]  (residual adds; the slot-0 fusion's gate)
struct AddArgs { uint M, H, ldy, gated; };
kernel void h4_add(device float* x [[buffer(0)]], device const float* yv [[buffer(1)]], device const float* g [[buffer(2)]],
                   constant AddArgs& a [[buffer(3)]], uint2 t [[thread_position_in_grid]]) {
  const uint c = t.x, m = t.y;
  if (c >= a.H || m >= a.M) return;
  const float v = yv[size_t(m) * a.ldy + c];
  x[size_t(m) * a.H + c] += a.gated ? g[c] * v : v;
}
// draft key lists: row q of the block sees the target layer's selection of the source row (all
// positions < the anchor), then the block's own positions anchor..anchor+q
struct DselArgs { uint rows, src, anchor, selmax_t, selmax_d; };
kernel void h4_dsel(device const uint* tsel [[buffer(0)]], device const uint* tnsel [[buffer(1)]],
                    device uint* dsel [[buffer(2)]], device uint* dnsel [[buffer(3)]], constant DselArgs& a [[buffer(4)]],
                    uint2 t [[thread_position_in_grid]]) {
  const uint i = t.x, q = t.y;
  if (q >= a.rows) return;
  const uint nh = tnsel[a.src];
  device uint* o = dsel + size_t(q) * a.selmax_d;
  if (i < nh) o[i] = tsel[size_t(a.src) * a.selmax_t + i];
  else if (i <= nh + q) o[i] = a.anchor + (i - nh);
  if (i == 0) dnsel[q] = nh + q + 1;
}
// Fused DFly2 post-attn norm + residual add + post-ln -> frag (replaces 3 dispatches per layer with 1)
struct DPostArgs { uint M, H; float eps; };
kernel void h4_d_post_add_ln(device const float* Yo [[buffer(0)]], device const float* post_w [[buffer(1)]],
                             device float* dx [[buffer(2)]], device const float* ln_w [[buffer(3)]],
                             device bfloat* frag [[buffer(4)]], constant DPostArgs& a [[buffer(5)]],
                             uint m [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                             ushort ntg [[threads_per_threadgroup]]) {
  threadgroup float sh[32];
  device const float* yr = Yo + size_t(m) * a.H;
  device float* xr = dx + size_t(m) * a.H;
  float s0 = 0.0f;
  for (uint i = tid; i < a.H; i += ntg) s0 = fma(yr[i], yr[i], s0);
  const float r0 = rsqrt(tg_sum(s0, sh, tid, ntg) / float(a.H) + a.eps);
  float s1 = 0.0f;
  for (uint i = tid; i < a.H; i += ntg) {
    const float v = xr[i] + yr[i] * r0 * post_w[i];
    xr[i] = v;
    s1 = fma(v, v, s1);
  }
  const float r1 = rsqrt(tg_sum(s1, sh, tid, ntg) / float(a.H) + a.eps);
  for (uint i = tid; i < a.H; i += ntg) frag[frag_off(m, i, a.H)] = bfloat(xr[i] * r1 * ln_w[i]);
}

// out = x + y as a frag (no norm): the corrected draft hidden feeds lm_head directly
struct AddFArgs { uint M, H, ldx, offx; };
kernel void h4_add_frag(device const float* x [[buffer(0)]], device const float* yv [[buffer(1)]],
                        device bfloat* frag [[buffer(2)]], constant AddFArgs& a [[buffer(3)]],
                        device const uint* gate [[buffer(4)]], uint2 t [[thread_position_in_grid]]) {
  if (gate[0] == 0) return;
  const uint c = t.x, m = t.y;
  if (c >= a.H || m >= a.M) return;
  const float v = x[size_t(m) * a.ldx + a.offx + c] + yv[size_t(m) * a.H + c];
  frag[frag_off(m, c, a.H)] = bfloat(v);
}

// ------------------------------------------------------------------ sampling
// Gumbel-max: token = argmax_t(logit_t / T + G(seed, pos, t)), G from a counter hash, so a position's
// draw is fixed by (seed, position) alone: exact categorical sampling at temperature T, deterministic per
// seed, and speculation stays exact (a draft is accepted iff it equals the token drawn here).
static inline uint hmix(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
static inline float gumbel(uint s0, uint s1, uint pos, uint t) {
  const uint h = hmix(hmix(hmix(s0 ^ 0x9E3779B9u) ^ s1) ^ hmix(pos * 0x85EBCA6Bu + 0x27d4eb2fu) ^ (t * 0xC2B2AE35u));
  const float u = (float(h >> 8) + 0.5f) * (1.0f / 16777216.0f);  // (0, 1)
  return -log(-log(u));
}
static inline float fkey_inv(uint k) { return as_type<float>((k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k); }

// Compute the logit floor (in scaled-logit space r[i] * it) for min_p, top_k, and top_p.
// top_k uses a 4-pass 8-bit MSB radix select (exact, any range); top_p uses 1 pass of a 256-bin
// probability histogram (0.125 log-prob bins) to bracket the cutoff, then 8 binary-search passes.
static inline float trunc_floor(device const float* r, uint N, float it, float M_, float lmin,
                                float top_p, uint top_k, threadgroup float* bm,
                                threadgroup atomic_uint* hist, threadgroup uint* pref_need,
                                ushort tid, ushort ntg) {
  float floor_ = M_ + lmin;
  if (top_k > 0 && top_k < N) {
    if (tid == 0) { pref_need[0] = 0; pref_need[1] = top_k; }
    uint mask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
      for (uint i = tid; i < 256; i += ntg) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const uint p = pref_need[0];
      for (uint i = tid; i < N; i += ntg) {
        const uint k = fkey(r[i] * it);
        if ((k & mask) == p) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid == 0) {
        uint acc = 0, nd = pref_need[1];
        for (int b = 255; b >= 0; b--) {
          const uint h = atomic_load_explicit(&hist[b], memory_order_relaxed);
          if (acc + h >= nd) { pref_need[0] = p | (uint(b) << shift); pref_need[1] = nd - acc; break; }
          acc += h;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mask |= 255u << shift;
    }
    floor_ = max(floor_, fkey_inv(pref_need[0]));
  }
  if (top_p > 0.0f && top_p < 1.0f) {
    // Pass 1: coarse 256-bin probability histogram (0.125 log-prob bins, covering [M_-32, M_])
    for (uint i = tid; i < 256; i += ntg) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float z = 0.0f;
    for (uint i = tid; i < N; i += ntg) {
      const float l = r[i] * it;
      if (l >= floor_) {
        const float w = exp(l - M_);
        z += w;
        const uint b = uint(clamp(int((M_ - l) * 8.0f), 0, 255));
        atomic_fetch_add_explicit(&hist[b], uint(w * 65536.0f + 0.5f), memory_order_relaxed);
      }
    }
    const float Z = tg_sum(z, bm, tid, ntg);
    const float target = top_p * Z;
    if (tid == 0) {
      const ulong tgt_q = ulong(target * 65536.0f);
      ulong acc = 0;
      uint bstar = 255;
      for (uint b = 0; b < 256; b++) {
        const ulong h = atomic_load_explicit(&hist[b], memory_order_relaxed);
        if (acc + h >= tgt_q) { bstar = b; break; }
        acc += h;
      }
      pref_need[0] = bstar;
      pref_need[1] = uint(min(acc, ulong(0xFFFFFFFFu)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint bstar = pref_need[0];
    const float lo = bstar < 255 ? max(floor_, M_ - float(bstar + 1) * 0.125f) : max(floor_, M_ - 40.0f);
    const float hi = M_ - float(bstar) * 0.125f;
    if (hi > lo) {
      // Pass 2: fine 256-bin probability histogram inside [lo, hi] (step 0.000488, matching 8 binary search steps in 1 pass)
      for (uint i = tid; i < 256; i += ntg) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float inv_w = 256.0f / (hi - lo);
      float above = 0.0f;
      for (uint i = tid; i < N; i += ntg) {
        const float l = r[i] * it;
        if (l >= hi) {
          above += exp(l - M_);
        } else if (l >= lo) {
          const float w = exp(l - M_);
          const uint b = uint(clamp(int((hi - l) * inv_w), 0, 255));
          atomic_fetch_add_explicit(&hist[b], uint(w * 16777216.0f + 0.5f), memory_order_relaxed);
        }
      }
      const float A = tg_sum(above, bm, tid, ntg);
      if (tid == 0) {
        const float rem = max(0.0f, target - A);
        const ulong rem_q = ulong(rem * 16777216.0f);
        ulong acc = 0;
        uint fstar = 255;
        for (uint b = 0; b < 256; b++) {
          acc += atomic_load_explicit(&hist[b], memory_order_relaxed);
          if (acc >= rem_q) { fstar = b; break; }
        }
        pref_need[0] = fstar;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float fine_lo = hi - float(pref_need[0] + 1) * ((hi - lo) / 256.0f);
      floor_ = max(floor_, fine_lo);
    } else {
      floor_ = max(floor_, lo);
    }
  }
  return floor_;
}

struct SampArgs { uint M, N, p0, s0, s1; float inv_t; float lmin; float top_p; uint top_k; };  // lmin: log(min_p), -inf off
kernel void h4_sample(device const float* y [[buffer(0)]], device uint* out [[buffer(1)]],
                      constant SampArgs& a [[buffer(2)]], uint m [[threadgroup_position_in_grid]],
                      ushort tid [[thread_index_in_threadgroup]], ushort ntg [[threads_per_threadgroup]]) {
  threadgroup float bv[32], bm[32];
  threadgroup uint bi[32], pn[2];
  threadgroup atomic_uint hist[256];
  device const float* r = y + size_t(m) * a.N;
  const uint pos = a.p0 + m;
  float mx = -INFINITY;
  for (uint i = tid; i < a.N; i += ntg) mx = max(mx, r[i] * a.inv_t);
  mx = simd_max(mx);
  if ((tid & 31) == 0) bm[tid / 32] = mx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float M_ = bm[0];
  for (uint s = 1; s < ntg / 32; s++) M_ = max(M_, bm[s]);
  const float floor_ = trunc_floor(r, a.N, a.inv_t, M_, a.lmin, a.top_p, a.top_k, bm, hist, pn, tid, ntg);
  float v = -INFINITY;
  uint idx = 0;
  for (uint i = tid; i < a.N; i += ntg) {
    const float l = r[i] * a.inv_t;
    if (l < floor_) continue;
    const float s = l + gumbel(a.s0, a.s1, pos, i);
    if (s > v) { v = s; idx = i; }
  }
  for (ushort off = 16; off > 0; off >>= 1) {
    const float ov = simd_shuffle_down(v, off);
    const uint oi = simd_shuffle_down(idx, off);
    if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
  }
  if ((tid & 31) == 0) { bv[tid / 32] = v; bi[tid / 32] = idx; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float b = bv[0]; uint bix = bi[0];
    for (uint s = 1; s < ntg / 32; s++) if (bv[s] > b || (bv[s] == b && bi[s] < bix)) { b = bv[s]; bix = bi[s]; }
    out[m] = bix;
  }
}

// argmax / Gumbel draw that also writes the chosen token's softmax probability at temperature 1 (or T):
// the drafter's confidence, which decides how many drafts a round verifies
kernel void h4_pick_p(device const float* y [[buffer(0)]], device uint* out [[buffer(1)]],
                      constant SampArgs& a [[buffer(2)]], device float* prob [[buffer(3)]],
                      device float* pmax [[buffer(4)]], device const uint* gate [[buffer(5)]],
                      uint m [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                      ushort ntg [[threads_per_threadgroup]]) {
  if (gate[0] == 0) return;  // a dead draft (h4_dgate): nothing reads its token
  threadgroup float bv[32], bm[32], bs[32];
  threadgroup uint bi[32], pn[2];
  threadgroup atomic_uint hist[256];
  device const float* r = y + size_t(m) * a.N;
  const bool samp = a.inv_t > 0.0f;  // greedy callers pass inv_t < 0
  const float it = samp ? a.inv_t : 1.0f;
  float mx = -INFINITY;
  for (uint i = tid; i < a.N; i += ntg) mx = max(mx, r[i] * it);
  mx = simd_max(mx);
  if ((tid & 31) == 0) bm[tid / 32] = mx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float M_ = bm[0];
  for (uint s = 1; s < ntg / 32; s++) M_ = max(M_, bm[s]);
  const float floor_ = samp ? trunc_floor(r, a.N, it, M_, a.lmin, a.top_p, a.top_k, bm, hist, pn, tid, ntg) : -INFINITY;
  float v = -INFINITY, se = 0.0f;
  uint idx = 0;
  for (uint i = tid; i < a.N; i += ntg) {
    const float l = r[i] * it;
    se += exp(l - M_);
    if (l < floor_) continue;
    const float s = samp ? l + gumbel(a.s0, a.s1, a.p0 + m, i) : l;
    if (s > v) { v = s; idx = i; }
  }
  for (ushort off = 16; off > 0; off >>= 1) {
    const float ov = simd_shuffle_down(v, off);
    const uint oi = simd_shuffle_down(idx, off);
    if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
  }
  se = simd_sum(se);
  if ((tid & 31) == 0) { bv[tid / 32] = v; bi[tid / 32] = idx; bs[tid / 32] = se; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float b = bv[0]; uint bix = bi[0]; float S = 0.0f;
    for (uint s = 1; s < ntg / 32; s++) if (bv[s] > b || (bv[s] == b && bi[s] < bix)) { b = bv[s]; bix = bi[s]; }
    for (uint s = 0; s < ntg / 32; s++) S += bs[s];
    out[m] = bix;
    prob[m] = exp(r[bix] * it - M_) / S;
    pmax[m] = 1.0f / S;  // the distribution's top probability: a confidence that does not depend on the draw
  }
}

// gate[0] = 1 while the drafts so far keep their cumulative probability at or above the cut (the host's rule, so a
// draft the host would verify always has its head computed); probabilities of dead drafts are stale but <= 1
struct DgArgs { uint n; float cut; };
kernel void h4_dgate(device const float* prob [[buffer(0)]], device uint* gate [[buffer(1)]], constant DgArgs& a [[buffer(2)]]) {
  float c = 1.0f;
  for (uint i = 0; i < a.n; i++) c *= prob[i];
  gate[0] = c >= a.cut ? 1u : 0u;
}

// ------------------------------------------------------------------ batched multi-lane DFly2 drafting
struct DraftEhRow { device const float* xf; uint dx_row, xf_row; };
struct DraftEhArgs { uint n, H; float eps; };
// Pack n lanes' slot-0 fusion inputs: rms(dx[dx_row]) * enorm -> dfrag2[j, 0..H), rms(xf[xf_row]) * hnorm -> dfrag2[j, H..2H)
kernel void h4_draft_eh_rms(device const float* dx [[buffer(0)]], device const float* enorm [[buffer(1)]],
                            device const float* hnorm [[buffer(2)]], device bfloat* dfrag2 [[buffer(3)]],
                            device const DraftEhRow* rows [[buffer(4)]], constant DraftEhArgs& a [[buffer(5)]],
                            uint j [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                            ushort ntg [[threads_per_threadgroup]]) {
  threadgroup float sh[2][32];
  const DraftEhRow r = rows[j];
  device const float* xr = dx + size_t(r.dx_row) * a.H;
  device const float* hr = r.xf + size_t(r.xf_row) * a.H;
  float s0 = 0.0f, s1 = 0.0f;
  for (uint i = tid; i < a.H; i += ntg) {
    s0 = fma(xr[i], xr[i], s0);
    s1 = fma(hr[i], hr[i], s1);
  }
  const ushort sgi = tid / 32, ns = (ntg + 31) / 32;
  const float t0 = simd_sum(s0), t1 = simd_sum(s1);
  if ((tid & 31) == 0) { sh[0][sgi] = t0; sh[1][sgi] = t1; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot0 = 0.0f, tot1 = 0.0f;
  for (ushort k = 0; k < ns; k++) { tot0 += sh[0][k]; tot1 += sh[1][k]; }
  const float r0 = rsqrt(tot0 / float(a.H) + a.eps), r1 = rsqrt(tot1 / float(a.H) + a.eps);
  for (uint i = tid; i < a.H; i += ntg) {
    dfrag2[frag_off(j, i, 2 * a.H)] = bfloat(xr[i] * r0 * enorm[i]);
    dfrag2[frag_off(j, a.H + i, 2 * a.H)] = bfloat(hr[i] * r1 * hnorm[i]);
  }
}

// dx[row0[j]][c] += gate[c] * Yo[j][c]
struct DraftAddArgs { uint n, H; };
kernel void h4_draft_eh_add(device float* dx [[buffer(0)]], device const float* Yo [[buffer(1)]],
                            device const float* gate [[buffer(2)]], device const uint* row0 [[buffer(3)]],
                            constant DraftAddArgs& a [[buffer(4)]], uint2 t [[thread_position_in_grid]]) {
  const uint c = t.x, j = t.y;
  if (c >= a.H || j >= a.n) return;
  dx[size_t(row0[j]) * a.H + c] += gate[c] * Yo[size_t(j) * a.H + c];
}

// Per-row draft key lists across lanes: row m belongs to a lane with its own tsel/tnsel/src/anchor/local_q
struct DselLaneRow { device const uint* tsel; device const uint* tnsel; uint src, anchor, q; };
struct DselLaneArgs { uint M, selmax_t, selmax_d; };
kernel void h4_dsel_lanes(device const DselLaneRow* rows [[buffer(0)]], device uint* dsel [[buffer(1)]],
                          device uint* dnsel [[buffer(2)]], constant DselLaneArgs& a [[buffer(3)]],
                          uint2 t [[thread_position_in_grid]]) {
  const uint i = t.x, m = t.y;
  if (m >= a.M) return;
  const DselLaneRow r = rows[m];
  const uint nh = r.tnsel[r.src];
  device uint* o = dsel + size_t(m) * a.selmax_d;
  if (i < nh) o[i] = r.tsel[size_t(r.src) * a.selmax_t + i];
  else if (i <= nh + r.q) o[i] = r.anchor + (i - nh);
  if (i == 0) dnsel[m] = nh + r.q + 1;
}

// Fused hidden-correction input for step s2 across n lanes:
// rms(dh[row0[j] + s2 + 1]) * hchn -> dfrag2[j, 0..H), rms(embed[tok_j]) * hcen -> dfrag2[j, H..2H)
struct DraftHcArgs { uint n, H, s2; float eps; };
kernel void h4_draft_hc_in(device const float* dh [[buffer(0)]], device const bfloat* E [[buffer(1)]],
                           device const float* hchn [[buffer(2)]], device const float* hcen [[buffer(3)]],
                           device const uint* row0 [[buffer(4)]], device const uint* tok_in [[buffer(5)]],
                           device const uint* max_d [[buffer(6)]], device const uint* gate [[buffer(7)]],
                           device bfloat* dfrag2 [[buffer(8)]], constant DraftHcArgs& a [[buffer(9)]],
                           uint j [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                           ushort ntg [[threads_per_threadgroup]]) {
  if (gate[0] == 0 || a.s2 >= max_d[j]) return;
  threadgroup float sh[2][32];
  device const float* hr = dh + size_t(row0[j] + a.s2 + 1) * a.H;
  device const bfloat* er = E + size_t(tok_in[j]) * a.H;
  float s0 = 0.0f, s1 = 0.0f;
  for (uint i = tid; i < a.H; i += ntg) {
    const float hv = hr[i], ev = float(er[i]);
    s0 = fma(hv, hv, s0);
    s1 = fma(ev, ev, s1);
  }
  const ushort sgi = tid / 32, ns = (ntg + 31) / 32;
  const float t0 = simd_sum(s0), t1 = simd_sum(s1);
  if ((tid & 31) == 0) { sh[0][sgi] = t0; sh[1][sgi] = t1; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot0 = 0.0f, tot1 = 0.0f;
  for (ushort k = 0; k < ns; k++) { tot0 += sh[0][k]; tot1 += sh[1][k]; }
  const float r0 = rsqrt(tot0 / float(a.H) + a.eps), r1 = rsqrt(tot1 / float(a.H) + a.eps);
  for (uint i = tid; i < a.H; i += ntg) {
    dfrag2[frag_off(j, i, 2 * a.H)] = bfloat(hr[i] * r0 * hchn[i]);
    dfrag2[frag_off(j, a.H + i, 2 * a.H)] = bfloat(float(er[i]) * r1 * hcen[i]);
  }
}

// Per-lane hidden correction add -> frag (for lm_head_w4)
struct DraftCorrArgs { uint n, H, s2; };
kernel void h4_draft_corr(device const float* dh [[buffer(0)]], device const float* Yo [[buffer(1)]],
                          device const uint* row0 [[buffer(2)]], device const uint* max_d [[buffer(3)]],
                          device const uint* gate [[buffer(4)]], device bfloat* frag [[buffer(5)]],
                          constant DraftCorrArgs& a [[buffer(6)]], uint2 t [[thread_position_in_grid]]) {
  const uint c = t.x, j = t.y;
  if (c >= a.H || j >= a.n || gate[0] == 0 || a.s2 >= max_d[j]) return;
  const float v = dh[size_t(row0[j] + a.s2 + 1) * a.H + c] + Yo[size_t(j) * a.H + c];
  frag[frag_off(j, c, a.H)] = bfloat(v);
}

// Per-lane draft sampling (same bits as h4_pick_p, per-row SampRow)
struct SampRow { uint N, pos, s0, s1; float inv_t, lmin, top_p; uint top_k; };
kernel void h4_pick_p_lanes(device const float* y [[buffer(0)]], device uint* out [[buffer(1)]],
                            device const SampRow* sr [[buffer(2)]], device float* prob [[buffer(3)]],
                            device float* pmax [[buffer(4)]], device const uint* gate [[buffer(5)]],
                            device const uint* max_d [[buffer(6)]], constant uint& s2 [[buffer(7)]],
                            uint m [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                            ushort ntg [[threads_per_threadgroup]]) {
  if (gate[0] == 0 || s2 >= max_d[m]) return;
  threadgroup float bv[32], bm[32], bs[32];
  threadgroup uint bi[32], pn[2];
  threadgroup atomic_uint hist[256];
  const SampRow a = sr[m];
  device const float* r = y + size_t(m) * a.N;
  const bool samp = a.inv_t > 0.0f;
  const float it = samp ? a.inv_t : 1.0f;
  float mx = -INFINITY;
  for (uint i = tid; i < a.N; i += ntg) mx = max(mx, r[i] * it);
  mx = simd_max(mx);
  if ((tid & 31) == 0) bm[tid / 32] = mx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float M_ = bm[0];
  for (uint s = 1; s < ntg / 32; s++) M_ = max(M_, bm[s]);
  const float floor_ = samp ? trunc_floor(r, a.N, it, M_, a.lmin, a.top_p, a.top_k, bm, hist, pn, tid, ntg) : -INFINITY;
  float v = -INFINITY, se = 0.0f;
  uint idx = 0;
  for (uint i = tid; i < a.N; i += ntg) {
    const float l = r[i] * it;
    se += exp(l - M_);
    if (l < floor_) continue;
    const float s = samp ? l + gumbel(a.s0, a.s1, a.pos, i) : l;
    if (s > v) { v = s; idx = i; }
  }
  for (ushort off = 16; off > 0; off >>= 1) {
    const float ov = simd_shuffle_down(v, off);
    const uint oi = simd_shuffle_down(idx, off);
    if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
  }
  se = simd_sum(se);
  if ((tid & 31) == 0) { bv[tid / 32] = v; bi[tid / 32] = idx; bs[tid / 32] = se; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float b = bv[0]; uint bix = bi[0]; float S = 0.0f;
    for (uint s = 1; s < ntg / 32; s++) if (bv[s] > b || (bv[s] == b && bi[s] < bix)) { b = bv[s]; bix = bi[s]; }
    for (uint s = 0; s < ntg / 32; s++) S += bs[s];
    out[m] = bix;
    prob[m] = exp(r[bix] * it - M_) / S;
    pmax[m] = 1.0f / S;
  }
}

struct DgLaneArgs { uint n_lanes, step, stride; float cut; };
kernel void h4_dgate_lanes(device const float* prob [[buffer(0)]], device const uint* max_d [[buffer(1)]],
                           device uint* gate [[buffer(2)]], constant DgLaneArgs& a [[buffer(3)]]) {
  uint any = 0u;
  for (uint j = 0; j < a.n_lanes; j++) {
    if (a.step >= max_d[j]) continue;
    float c = 1.0f;
    for (uint i = 0; i < a.step; i++) c *= prob[i * a.stride + j];
    if (c >= a.cut) { any = 1u; break; }
  }
  gate[0] = any;
}

// Per-row sampling for multi-lane steps: each row carries its own pos, seed, temperature (inv_t <= 0: greedy),
// min_p, top_p, and top_k. Identical bits to h4_argmax (greedy) and h4_sample (sampled).
kernel void h4_sample_lanes(device const float* y [[buffer(0)]], device uint* out [[buffer(1)]],
                            device const SampRow* sr [[buffer(2)]], uint m [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]], ushort ntg [[threads_per_threadgroup]]) {
  threadgroup float bv[32], bm[32];
  threadgroup uint bi[32], pn[2];
  threadgroup atomic_uint hist[256];
  const SampRow a = sr[m];
  device const float* r = y + size_t(m) * a.N;
  float v = -INFINITY;
  uint idx = 0;
  if (a.inv_t <= 0.0f) {
    for (uint i = tid; i < a.N; i += ntg) if (r[i] > v) { v = r[i]; idx = i; }
  } else {
    float mx = -INFINITY;
    for (uint i = tid; i < a.N; i += ntg) mx = max(mx, r[i] * a.inv_t);
    mx = simd_max(mx);
    if ((tid & 31) == 0) bm[tid / 32] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float M_ = bm[0];
    for (uint s = 1; s < ntg / 32; s++) M_ = max(M_, bm[s]);
    const float floor_ = trunc_floor(r, a.N, a.inv_t, M_, a.lmin, a.top_p, a.top_k, bm, hist, pn, tid, ntg);
    for (uint i = tid; i < a.N; i += ntg) {
      const float l = r[i] * a.inv_t;
      if (l < floor_) continue;
      const float s = l + gumbel(a.s0, a.s1, a.pos, i);
      if (s > v) { v = s; idx = i; }
    }
  }
  for (ushort off = 16; off > 0; off >>= 1) {
    const float ov = simd_shuffle_down(v, off);
    const uint oi = simd_shuffle_down(idx, off);
    if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
  }
  if ((tid & 31) == 0) { bv[tid / 32] = v; bi[tid / 32] = idx; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float b = bv[0]; uint bix = bi[0];
    for (uint s = 1; s < ntg / 32; s++) if (bv[s] > b || (bv[s] == b && bi[s] < bix)) { b = bv[s]; bix = bi[s]; }
    out[m] = bix;
  }
}
