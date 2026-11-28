// DankAttn (Inkling-2-Mini): Metal 4 kernels for Inkling2LoopForConditionalGeneration (text tower).
// Wave 2A: vectorized float4/bfloat4, 2-stage parallel msg_gate, barrier-free simdgroup split-K
// attention (4 keys/TG in parallel, 0 inner barriers), and SIMD top-8 MoE routing.
#include <metal_stdlib>
using namespace metal;

static inline uint frag_off(uint m, uint k, uint K) {
  const uint kk = k & 31, mm = m & 15;
  const uint l = ((kk >> 2) & 1) | ((mm & 3) << 1) | (((kk >> 3) & 1) << 3) | (((mm >> 2) & 1) << 4);
  const uint i = (kk & 3) | ((mm >> 3) << 2) | ((kk >> 4) << 3);
  return ((m >> 4) * (K >> 5) + (k >> 5)) * 512 + l * 16 + i;
}

static inline float isigm(float x) { return 1.0f / (1.0f + exp(-x)); }

// 1. Embedding lookup (bf16 [V, H]) + embed_norm (f32 [H]) -> h_out (f32 [M, H], rounded via bf16).
// Grid: (M, 1, 1), TG: 256 threads.
// Depthwise short-conv value of one channel (shared by ink_post_sconv_add and the fused decode kernels so the
// arithmetic is one expression everywhere).
static inline float post_sconv_value(device const float* sconv_w, uint ch, uint ks, float x, float h2, float h1,
                                     float h0) {
  const float sw3 = sconv_w[ch * 4u + 0u];
  const float sw2 = sconv_w[ch * 4u + 1u];
  const float sw1 = sconv_w[ch * 4u + 2u];
  const float sw0 = sconv_w[ch * 4u + 3u];
  const float inv_denom = rsqrt(float(ks + 1u));
  return (sw0 * x + sw1 * h2 + sw2 * h1 + sw3 * h0 + x) * inv_denom;
}

kernel void ink_embed(device const bfloat4* emb [[buffer(0)]], device const float4* norm_w [[buffer(1)]],
                      device const uint* ids [[buffer(2)]], device float4* out [[buffer(3)]],
                      constant uint& H [[buffer(4)]], constant float& eps [[buffer(5)]],
                      uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint tpg [[threads_per_threadgroup]]) {
  threadgroup float sm[8];
  const uint H4 = H >> 2;
  const uint tok = ids[row];
  device const bfloat4* er = emb + size_t(tok) * H4;
  float s2 = 0.0f;
  for (uint c = tid; c < H4; c += tpg) {
    const float4 v = float4(er[c]);
    s2 += dot(v, v);
  }
  s2 = simd_sum(s2);
  if ((tid & 31u) == 0u) sm[tid >> 5] = s2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float tot = 0.0f;
    for (uint i = 0; i < (tpg >> 5); i++) tot += sm[i];
    sm[0] = rsqrt(tot / float(H) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = sm[0];
  device float4* dst = out + size_t(row) * H4;
  for (uint c = tid; c < H4; c += tpg) {
    const float4 v = float4(er[c]) * inv * norm_w[c];
    dst[c] = float4(bfloat4(v));
  }
}

// 2. Fast 2-barrier Block Attention Residual (float4 vectorized, register-cached query, handles nq = 1, 3, or 4).
// Grid: (M, nq, 1), TG: 256 threads.
struct AttnResArgs {
  uint H, M, n_prev, b_stride;
  float eps;
};
kernel void ink_attn_res(device const float4* blocks [[buffer(0)]], device const float4* partial [[buffer(1)]],
                         device const float4* queries [[buffer(2)]], device float4* out [[buffer(3)]],
                         constant AttnResArgs& a [[buffer(4)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint H4 = a.H >> 2, bs4 = a.b_stride >> 2;
  const uint row = tg.x, qi = tg.y;
  const uint B = a.n_prev + 1u;
  const float inv_sqrt_h = rsqrt(float(a.H));
  threadgroup float sm_s[12][8];
  threadgroup float sm_d[12][8];
  threadgroup float wts[12];
  device const float4* q = queries + size_t(qi) * H4;

  // H4 = 576, TG = 256 -> at most 3 float4s per thread; cache q in registers once.
  const bool has2 = (tid + 512u) < H4;
  const float4 q0 = q[tid];
  const float4 q1 = q[tid + 256u];
  const float4 q2 = has2 ? q[tid + 512u] : float4(0.0f);

  for (uint b = 0; b < B; b++) {
    device const float4* st = (b < a.n_prev) ? (blocks + size_t(b) * bs4 + size_t(row) * H4)
                                             : (partial + size_t(row) * H4);
    const float4 v0 = st[tid], v1 = st[tid + 256u];
    float s2 = dot(v0, v0) + dot(v1, v1);
    float dt = dot(v0, q0) + dot(v1, q1);
    if (has2) {
      const float4 v2 = st[tid + 512u];
      s2 += dot(v2, v2);
      dt += dot(v2, q2);
    }
    s2 = simd_sum(s2);
    dt = simd_sum(dt);
    if ((tid & 31u) == 0u) { sm_s[b][tid >> 5] = s2; sm_d[b][tid >> 5] = dt; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float inv_rms[12];
    float mx = -1e30f;
    for (uint b = 0; b < B; b++) {
      float t2 = 0.0f, td = 0.0f;
      for (uint i = 0; i < 8u; i++) { t2 += sm_s[b][i]; td += sm_d[b][i]; }
      const float ir = rsqrt(t2 / float(a.H) + a.eps);
      inv_rms[b] = ir;
      const float w = td * ir * inv_sqrt_h;
      wts[b] = w;
      mx = max(mx, w);
    }
    float den = 0.0f;
    for (uint b = 0; b < B; b++) { wts[b] = exp(wts[b] - mx); den += wts[b]; }
    const float inv_d = 1.0f / den;
    for (uint b = 0; b < B; b++) wts[b] *= inv_d * inv_rms[b];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f);
  for (uint b = 0; b < B; b++) {
    device const float4* st = (b < a.n_prev) ? (blocks + size_t(b) * bs4 + size_t(row) * H4)
                                             : (partial + size_t(row) * H4);
    const float wb = wts[b];
    acc0 = fma(st[tid], wb, acc0);
    acc1 = fma(st[tid + 256u], wb, acc1);
    if (has2) acc2 = fma(st[tid + 512u], wb, acc2);
  }
  device float4* dst = out + (size_t(qi) * a.M + row) * H4;
  dst[tid] = acc0;
  dst[tid + 256u] = acc1;
  if (has2) dst[tid + 512u] = acc2;
}

// 2b. Fused Block Attention Residual + RMSNorm -> bf16 fragment (+ optional f32 copy).
//     Eliminates DRAM round-trip + encoder barrier on all !has_msgs layers (24/48 layers) & final norm!
//     Handles nq=1 (local attn, MLP, final) and nq=3 (global Q/K/V with separate norm weights & frag outputs).
// Grid: (M, nq, 1), TG: 256 threads.
struct AttnResRmsArgs {
  uint H, M, n_prev, b_stride, write_f32;
  float eps;
};
kernel void ink_attn_res_rms(device const float4* blocks [[buffer(0)]], device const float4* partial [[buffer(1)]],
                             device const float4* queries [[buffer(2)]],
                             device const float4* w0 [[buffer(3)]], device const float4* w1 [[buffer(4)]],
                             device const float4* w2 [[buffer(5)]],
                             device bfloat* frag0 [[buffer(6)]], device bfloat* frag1 [[buffer(7)]],
                             device bfloat* frag2 [[buffer(8)]],
                             device float4* f32_out [[buffer(9)]],
                             constant AttnResRmsArgs& a [[buffer(10)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint H4 = a.H >> 2, bs4 = a.b_stride >> 2;
  const uint row = tg.x, qi = tg.y;
  const uint B = a.n_prev + 1u;
  const float inv_sqrt_h = rsqrt(float(a.H));
  threadgroup float sm_s[12][8];
  threadgroup float sm_d[12][8];
  threadgroup float wts[12];
  threadgroup float sm_r[8];
  device const float4* q = queries + size_t(qi) * H4;

  const bool has2 = (tid + 512u) < H4;
  const float4 q0 = q[tid];
  const float4 q1 = q[tid + 256u];
  const float4 q2 = has2 ? q[tid + 512u] : float4(0.0f);

  for (uint b = 0; b < B; b++) {
    device const float4* st = (b < a.n_prev) ? (blocks + size_t(b) * bs4 + size_t(row) * H4)
                                             : (partial + size_t(row) * H4);
    const float4 v0 = st[tid], v1 = st[tid + 256u];
    float s2 = dot(v0, v0) + dot(v1, v1);
    float dt = dot(v0, q0) + dot(v1, q1);
    if (has2) {
      const float4 v2 = st[tid + 512u];
      s2 += dot(v2, v2);
      dt += dot(v2, q2);
    }
    s2 = simd_sum(s2);
    dt = simd_sum(dt);
    if ((tid & 31u) == 0u) { sm_s[b][tid >> 5] = s2; sm_d[b][tid >> 5] = dt; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float inv_rms[12];
    float mx = -1e30f;
    for (uint b = 0; b < B; b++) {
      float t2 = 0.0f, td = 0.0f;
      for (uint i = 0; i < 8u; i++) { t2 += sm_s[b][i]; td += sm_d[b][i]; }
      const float ir = rsqrt(t2 / float(a.H) + a.eps);
      inv_rms[b] = ir;
      const float w = td * ir * inv_sqrt_h;
      wts[b] = w;
      mx = max(mx, w);
    }
    float den = 0.0f;
    for (uint b = 0; b < B; b++) { wts[b] = exp(wts[b] - mx); den += wts[b]; }
    const float inv_d = 1.0f / den;
    for (uint b = 0; b < B; b++) wts[b] *= inv_d * inv_rms[b];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f);
  for (uint b = 0; b < B; b++) {
    device const float4* st = (b < a.n_prev) ? (blocks + size_t(b) * bs4 + size_t(row) * H4)
                                             : (partial + size_t(row) * H4);
    const float wb = wts[b];
    acc0 = fma(st[tid], wb, acc0);
    acc1 = fma(st[tid + 256u], wb, acc1);
    if (has2) acc2 = fma(st[tid + 512u], wb, acc2);
  }

  float rs2 = dot(acc0, acc0) + dot(acc1, acc1) + (has2 ? dot(acc2, acc2) : 0.0f);
  rs2 = simd_sum(rs2);
  if ((tid & 31u) == 0u) sm_r[tid >> 5] = rs2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float t = 0.0f;
    for (uint i = 0; i < 8u; i++) t += sm_r[i];
    sm_r[0] = rsqrt(t / float(a.H) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = sm_r[0];

  device const float4* wn = (qi == 0u) ? w0 : ((qi == 1u) ? w1 : w2);
  device bfloat* frag = (qi == 0u) ? frag0 : ((qi == 1u) ? frag1 : frag2);

  auto emit = [&](uint c4, float4 v) {
    const bfloat4 bv = bfloat4(v * inv * wn[c4]);
    const uint c = c4 << 2;
    frag[frag_off(row, c + 0u, a.H)] = bv.x;
    frag[frag_off(row, c + 1u, a.H)] = bv.y;
    frag[frag_off(row, c + 2u, a.H)] = bv.z;
    frag[frag_off(row, c + 3u, a.H)] = bv.w;
    if (a.write_f32) f32_out[size_t(row) * H4 + c4] = float4(bv);
  };
  emit(tid, acc0);
  emit(tid + 256u, acc1);
  if (has2) emit(tid + 512u, acc2);
}

// 2b'. Decode/verify fusion: attention post-sconv (+ residual / block boundary) followed by the pre-MLP
//      ink_attn_res_rms (nq = 1, write_f32 = 1) for one lane's rows, in one phase instead of two. Each thread
//      computes the post-conv of exactly the float4 columns it reads afterwards, so the new partial and the
//      boundary block come from registers; arithmetic matches the two kernels it replaces.
// Grid: (m, 1, 1), TG: 256 threads.
struct ConvResRmsArgs {
  uint H, m, n_prev, b_stride, r0, ks, is_boundary, b_slot;
  float eps;
};
kernel void ink_post_sconv_attn_res_rms(device const float* o_out [[buffer(0)]],
                                        device const float* sconv_w [[buffer(1)]],
                                        device const float* conv_state_in [[buffer(2)]],
                                        device float* h_res [[buffer(3)]],
                                        device float* blocks_w [[buffer(4)]],
                                        device float* conv_hist [[buffer(5)]],
                                        device float* conv_state_out [[buffer(6)]],
                                        device const float4* queries [[buffer(7)]],
                                        device const float4* wn [[buffer(8)]],
                                        device bfloat* frag [[buffer(9)]],
                                        device float4* f32_out [[buffer(10)]],
                                        constant ConvResRmsArgs& a [[buffer(11)]],
                                        uint m [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint H4 = a.H >> 2, bs4 = a.b_stride >> 2;
  const uint row = a.r0 + m;
  const bool has2 = (tid + 512u) < H4;
  const uint nc = has2 ? 3u : 2u;
  const bool save_hist = (a.m <= 16u);

  // Post-attention sconv for this thread's columns (ink_post_sconv_add, is_mlp = 0).
  float4 part[3], bnd[3];
  for (uint i = 0u; i < nc; i++) {
    const uint c4 = tid + 256u * i;
    for (uint q = 0u; q < 4u; q++) {
      const uint ch = c4 * 4u + q;
      const float x  = o_out[size_t(row) * a.H + ch];
      const float h2 = (m >= 1u) ? o_out[size_t(row - 1u) * a.H + ch] : conv_state_in[2u * a.H + ch];
      const float h1 = (m >= 2u) ? o_out[size_t(row - 2u) * a.H + ch] : conv_state_in[(1u + m) * a.H + ch];
      const float h0 = (m >= 3u) ? o_out[size_t(row - 3u) * a.H + ch] : conv_state_in[(0u + m) * a.H + ch];
      if (save_hist) {
        if (m == 0u) {
          conv_hist[0u * a.H + ch] = h0;
          conv_hist[1u * a.H + ch] = h1;
          conv_hist[2u * a.H + ch] = h2;
        }
        conv_hist[(3u + m) * a.H + ch] = x;
      }
      if (m + 1u == a.m) {
        conv_state_out[0u * a.H + ch] = h1;
        conv_state_out[1u * a.H + ch] = h2;
        conv_state_out[2u * a.H + ch] = x;
      }
      const float cv = post_sconv_value(sconv_w, ch, a.ks, x, h2, h1, h0);
      const size_t hi = size_t(row) * a.H + ch;
      const float r_old = h_res[hi];
      float hn;
      if (a.is_boundary) {
        blocks_w[size_t(a.b_slot) * a.b_stride + hi] = r_old;
        hn = cv;
      } else {
        hn = r_old + cv;
      }
      h_res[hi] = hn;
      part[i][q] = hn;
      bnd[i][q] = r_old;
    }
  }
  if (!has2) { part[2] = float4(0.0f); bnd[2] = float4(0.0f); }

  // ink_attn_res_rms body (nq = 1), reading this thread's freshly written columns from registers.
  device const float4* blocks = (device const float4*)blocks_w;
  const uint B = a.n_prev + 1u;
  const float inv_sqrt_h = rsqrt(float(a.H));
  threadgroup float sm_s[12][8];
  threadgroup float sm_d[12][8];
  threadgroup float wts[12];
  threadgroup float sm_r[8];
  device const float4* q = queries;
  const float4 q0 = q[tid];
  const float4 q1 = q[tid + 256u];
  const float4 q2 = has2 ? q[tid + 512u] : float4(0.0f);
  auto ld = [&](uint b, uint i) -> float4 {
    if (b == a.n_prev) return part[i];
    if (a.is_boundary && b == a.b_slot) return bnd[i];
    return blocks[size_t(b) * bs4 + size_t(row) * H4 + tid + 256u * i];
  };

  for (uint b = 0; b < B; b++) {
    const float4 v0 = ld(b, 0u), v1 = ld(b, 1u);
    float s2 = dot(v0, v0) + dot(v1, v1);
    float dt = dot(v0, q0) + dot(v1, q1);
    if (has2) {
      const float4 v2 = ld(b, 2u);
      s2 += dot(v2, v2);
      dt += dot(v2, q2);
    }
    s2 = simd_sum(s2);
    dt = simd_sum(dt);
    if ((tid & 31u) == 0u) { sm_s[b][tid >> 5] = s2; sm_d[b][tid >> 5] = dt; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float inv_rms[12];
    float mx = -1e30f;
    for (uint b = 0; b < B; b++) {
      float t2 = 0.0f, td = 0.0f;
      for (uint i = 0; i < 8u; i++) { t2 += sm_s[b][i]; td += sm_d[b][i]; }
      const float ir = rsqrt(t2 / float(a.H) + a.eps);
      inv_rms[b] = ir;
      const float w = td * ir * inv_sqrt_h;
      wts[b] = w;
      mx = max(mx, w);
    }
    float den = 0.0f;
    for (uint b = 0; b < B; b++) { wts[b] = exp(wts[b] - mx); den += wts[b]; }
    const float inv_d = 1.0f / den;
    for (uint b = 0; b < B; b++) wts[b] *= inv_d * inv_rms[b];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f);
  for (uint b = 0; b < B; b++) {
    const float wb = wts[b];
    acc0 = fma(ld(b, 0u), wb, acc0);
    acc1 = fma(ld(b, 1u), wb, acc1);
    if (has2) acc2 = fma(ld(b, 2u), wb, acc2);
  }

  float rs2 = dot(acc0, acc0) + dot(acc1, acc1) + (has2 ? dot(acc2, acc2) : 0.0f);
  rs2 = simd_sum(rs2);
  if ((tid & 31u) == 0u) sm_r[tid >> 5] = rs2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float t = 0.0f;
    for (uint i = 0; i < 8u; i++) t += sm_r[i];
    sm_r[0] = rsqrt(t / float(a.H) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = sm_r[0];

  auto emit = [&](uint c4, float4 v) {
    const bfloat4 bv = bfloat4(v * inv * wn[c4]);
    const uint c = c4 << 2;
    frag[frag_off(row, c + 0u, a.H)] = bv.x;
    frag[frag_off(row, c + 1u, a.H)] = bv.y;
    frag[frag_off(row, c + 2u, a.H)] = bv.z;
    frag[frag_off(row, c + 3u, a.H)] = bv.w;
    f32_out[size_t(row) * H4 + c4] = float4(bv);
  };
  emit(tid, acc0);
  emit(tid + 256u, acc1);
  if (has2) emit(tid + 512u, acc2);
}

// 2b''. Same fusion for has_msgs layers: attention post-sconv + the pre-MLP ink_attn_res_mg_prep (q_shared = 0,
//       one slot). Grid: (m, 1, 1), TG: 256 threads.
kernel void ink_post_sconv_attn_res_mg_prep(device const float* o_out [[buffer(0)]],
                                            device const float* sconv_w [[buffer(1)]],
                                            device const float* conv_state_in [[buffer(2)]],
                                            device float* h_res [[buffer(3)]],
                                            device float* blocks_w [[buffer(4)]],
                                            device float* conv_hist [[buffer(5)]],
                                            device float* conv_state_out [[buffer(6)]],
                                            device const float4* queries [[buffer(7)]],
                                            device const float4* msgs [[buffer(8)]],
                                            device float4* h_out [[buffer(9)]],
                                            device bfloat4* cat_out [[buffer(10)]],
                                            constant ConvResRmsArgs& a [[buffer(11)]],
                                            uint m [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint H4 = a.H >> 2, bs4 = a.b_stride >> 2;
  const uint row = a.r0 + m;
  const bool has2 = (tid + 512u) < H4;
  const uint nc = has2 ? 3u : 2u;
  const bool save_hist = (a.m <= 16u);

  float4 part[3], bnd[3];
  for (uint i = 0u; i < nc; i++) {
    const uint c4 = tid + 256u * i;
    for (uint q = 0u; q < 4u; q++) {
      const uint ch = c4 * 4u + q;
      const float x  = o_out[size_t(row) * a.H + ch];
      const float h2 = (m >= 1u) ? o_out[size_t(row - 1u) * a.H + ch] : conv_state_in[2u * a.H + ch];
      const float h1 = (m >= 2u) ? o_out[size_t(row - 2u) * a.H + ch] : conv_state_in[(1u + m) * a.H + ch];
      const float h0 = (m >= 3u) ? o_out[size_t(row - 3u) * a.H + ch] : conv_state_in[(0u + m) * a.H + ch];
      if (save_hist) {
        if (m == 0u) {
          conv_hist[0u * a.H + ch] = h0;
          conv_hist[1u * a.H + ch] = h1;
          conv_hist[2u * a.H + ch] = h2;
        }
        conv_hist[(3u + m) * a.H + ch] = x;
      }
      if (m + 1u == a.m) {
        conv_state_out[0u * a.H + ch] = h1;
        conv_state_out[1u * a.H + ch] = h2;
        conv_state_out[2u * a.H + ch] = x;
      }
      const float cv = post_sconv_value(sconv_w, ch, a.ks, x, h2, h1, h0);
      const size_t hi = size_t(row) * a.H + ch;
      const float r_old = h_res[hi];
      float hn;
      if (a.is_boundary) {
        blocks_w[size_t(a.b_slot) * a.b_stride + hi] = r_old;
        hn = cv;
      } else {
        hn = r_old + cv;
      }
      h_res[hi] = hn;
      part[i][q] = hn;
      bnd[i][q] = r_old;
    }
  }
  if (!has2) { part[2] = float4(0.0f); bnd[2] = float4(0.0f); }

  device const float4* blocks = (device const float4*)blocks_w;
  const uint B = a.n_prev + 1u;
  const float inv_sqrt_h = rsqrt(float(a.H));
  threadgroup float sm_s[12][8];
  threadgroup float sm_d[12][8];
  threadgroup float wts[12];
  threadgroup float sm_h[8], sm_m[8];
  device const float4* q = queries;
  const float4 q0 = q[tid];
  const float4 q1 = q[tid + 256u];
  const float4 q2 = has2 ? q[tid + 512u] : float4(0.0f);
  auto ld = [&](uint b, uint i) -> float4 {
    if (b == a.n_prev) return part[i];
    if (a.is_boundary && b == a.b_slot) return bnd[i];
    return blocks[size_t(b) * bs4 + size_t(row) * H4 + tid + 256u * i];
  };

  for (uint b = 0; b < B; b++) {
    const float4 v0 = ld(b, 0u), v1 = ld(b, 1u);
    float s2 = dot(v0, v0) + dot(v1, v1);
    float dt = dot(v0, q0) + dot(v1, q1);
    if (has2) {
      const float4 v2 = ld(b, 2u);
      s2 += dot(v2, v2);
      dt += dot(v2, q2);
    }
    s2 = simd_sum(s2);
    dt = simd_sum(dt);
    if ((tid & 31u) == 0u) { sm_s[b][tid >> 5] = s2; sm_d[b][tid >> 5] = dt; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float inv_rms[12];
    float mx = -1e30f;
    for (uint b = 0; b < B; b++) {
      float t2 = 0.0f, td = 0.0f;
      for (uint i = 0; i < 8u; i++) { t2 += sm_s[b][i]; td += sm_d[b][i]; }
      const float ir = rsqrt(t2 / float(a.H) + a.eps);
      inv_rms[b] = ir;
      const float w = td * ir * inv_sqrt_h;
      wts[b] = w;
      mx = max(mx, w);
    }
    float den = 0.0f;
    for (uint b = 0; b < B; b++) { wts[b] = exp(wts[b] - mx); den += wts[b]; }
    const float inv_d = 1.0f / den;
    for (uint b = 0; b < B; b++) wts[b] *= inv_d * inv_rms[b];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f);
  for (uint b = 0; b < B; b++) {
    const float wb = wts[b];
    acc0 = fma(ld(b, 0u), wb, acc0);
    acc1 = fma(ld(b, 1u), wb, acc1);
    if (has2) acc2 = fma(ld(b, 2u), wb, acc2);
  }
  {
    device float4* dst = h_out + size_t(row) * H4;
    dst[tid] = acc0;
    dst[tid + 256u] = acc1;
    if (has2) dst[tid + 512u] = acc2;
  }

  device const float4* mr = msgs + size_t(row) * H4;
  const float4 m0 = mr[tid], m1 = mr[tid + 256u];
  const float4 m2 = has2 ? mr[tid + 512u] : float4(0.0f);
  float sh = dot(acc0, acc0) + dot(acc1, acc1) + (has2 ? dot(acc2, acc2) : 0.0f);
  float sv = dot(m0, m0) + dot(m1, m1) + (has2 ? dot(m2, m2) : 0.0f);
  sh = simd_sum(sh); sv = simd_sum(sv);
  if ((tid & 31u) == 0u) { sm_h[tid >> 5] = sh; sm_m[tid >> 5] = sv; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float th = 0.0f, tm = 0.0f;
    for (uint i = 0; i < 8u; i++) { th += sm_h[i]; tm += sm_m[i]; }
    sm_h[0] = rsqrt(th / float(a.H) + a.eps);
    sm_m[0] = rsqrt(tm / float(a.H) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv_h = sm_h[0], inv_m = sm_m[0];
  device bfloat4* cdst = cat_out + size_t(row) * (2u * H4);
  cdst[tid] = bfloat4(acc0 * inv_h);
  cdst[tid + 256u] = bfloat4(acc1 * inv_h);
  if (has2) cdst[tid + 512u] = bfloat4(acc2 * inv_h);
  cdst[H4 + tid] = bfloat4(m0 * inv_m);
  cdst[H4 + tid + 256u] = bfloat4(m1 * inv_m);
  if (has2) cdst[H4 + tid + 512u] = bfloat4(m2 * inv_m);
}

// 2c. Fused Block Attention Residual + Message Gate Stage 1 (mg_prep) for has_msgs layers.
//     Writes h_out[slot] AND normalized bf16 cat_out[slot] in one kernel (eliminates 48 barriers + 96 dispatches).
//     Grid: (M, n_slots, 1) where n_slots=3 for QKV (q_shared=1 on local, 0 on global) or 1 for MLP.
struct AttnResMgArgs {
  uint H, M, n_prev, b_stride, q_shared, cat_stride_b4;
  float eps;
};
kernel void ink_attn_res_mg_prep(device const float4* blocks [[buffer(0)]], device const float4* partial [[buffer(1)]],
                                 device const float4* queries [[buffer(2)]],
                                 device const float4* msgs [[buffer(3)]],
                                 device float4* h_out [[buffer(4)]],
                                 device bfloat4* cat_out [[buffer(5)]],
                                 constant AttnResMgArgs& a [[buffer(6)]],
                                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint H4 = a.H >> 2, bs4 = a.b_stride >> 2;
  const uint row = tg.x, slot = tg.y;
  const uint qi = a.q_shared ? 0u : slot;
  const uint B = a.n_prev + 1u;
  const float inv_sqrt_h = rsqrt(float(a.H));
  threadgroup float sm_s[12][8];
  threadgroup float sm_d[12][8];
  threadgroup float wts[12];
  threadgroup float sm_h[8], sm_m[8];
  device const float4* q = queries + size_t(qi) * H4;

  const bool has2 = (tid + 512u) < H4;
  const float4 q0 = q[tid];
  const float4 q1 = q[tid + 256u];
  const float4 q2 = has2 ? q[tid + 512u] : float4(0.0f);

  for (uint b = 0; b < B; b++) {
    device const float4* st = (b < a.n_prev) ? (blocks + size_t(b) * bs4 + size_t(row) * H4)
                                             : (partial + size_t(row) * H4);
    const float4 v0 = st[tid], v1 = st[tid + 256u];
    float s2 = dot(v0, v0) + dot(v1, v1);
    float dt = dot(v0, q0) + dot(v1, q1);
    if (has2) {
      const float4 v2 = st[tid + 512u];
      s2 += dot(v2, v2);
      dt += dot(v2, q2);
    }
    s2 = simd_sum(s2);
    dt = simd_sum(dt);
    if ((tid & 31u) == 0u) { sm_s[b][tid >> 5] = s2; sm_d[b][tid >> 5] = dt; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float inv_rms[12];
    float mx = -1e30f;
    for (uint b = 0; b < B; b++) {
      float t2 = 0.0f, td = 0.0f;
      for (uint i = 0; i < 8u; i++) { t2 += sm_s[b][i]; td += sm_d[b][i]; }
      const float ir = rsqrt(t2 / float(a.H) + a.eps);
      inv_rms[b] = ir;
      const float w = td * ir * inv_sqrt_h;
      wts[b] = w;
      mx = max(mx, w);
    }
    float den = 0.0f;
    for (uint b = 0; b < B; b++) { wts[b] = exp(wts[b] - mx); den += wts[b]; }
    const float inv_d = 1.0f / den;
    for (uint b = 0; b < B; b++) wts[b] *= inv_d * inv_rms[b];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f);
  for (uint b = 0; b < B; b++) {
    device const float4* st = (b < a.n_prev) ? (blocks + size_t(b) * bs4 + size_t(row) * H4)
                                             : (partial + size_t(row) * H4);
    const float wb = wts[b];
    acc0 = fma(st[tid], wb, acc0);
    acc1 = fma(st[tid + 256u], wb, acc1);
    if (has2) acc2 = fma(st[tid + 512u], wb, acc2);
  }
  if (!a.q_shared || slot == 0u) {
    device float4* dst = h_out + (size_t(qi) * a.M + row) * H4;
    dst[tid] = acc0;
    dst[tid + 256u] = acc1;
    if (has2) dst[tid + 512u] = acc2;
  }

  device const float4* mr = msgs + (size_t(slot) * a.M + row) * H4;
  const float4 m0 = mr[tid], m1 = mr[tid + 256u];
  const float4 m2 = has2 ? mr[tid + 512u] : float4(0.0f);
  float sh = dot(acc0, acc0) + dot(acc1, acc1) + (has2 ? dot(acc2, acc2) : 0.0f);
  float sv = dot(m0, m0) + dot(m1, m1) + (has2 ? dot(m2, m2) : 0.0f);
  sh = simd_sum(sh); sv = simd_sum(sv);
  if ((tid & 31u) == 0u) { sm_h[tid >> 5] = sh; sm_m[tid >> 5] = sv; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float th = 0.0f, tm = 0.0f;
    for (uint i = 0; i < 8u; i++) { th += sm_h[i]; tm += sm_m[i]; }
    sm_h[0] = rsqrt(th / float(a.H) + a.eps);
    sm_m[0] = rsqrt(tm / float(a.H) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv_h = sm_h[0], inv_m = sm_m[0];
  device bfloat4* cdst = cat_out + size_t(slot) * a.cat_stride_b4 + size_t(row) * (2u * H4);
  cdst[tid] = bfloat4(acc0 * inv_h);
  cdst[tid + 256u] = bfloat4(acc1 * inv_h);
  if (has2) cdst[tid + 512u] = bfloat4(acc2 * inv_h);
  cdst[H4 + tid] = bfloat4(m0 * inv_m);
  cdst[H4 + tid + 256u] = bfloat4(m1 * inv_m);
  if (has2) cdst[H4 + tid + 512u] = bfloat4(m2 * inv_m);
}

// 2d. Shared-query version of ink_attn_res_mg_prep for local has_msgs layers (q_shared=1):
//     Computes attn_res ONCE per row (3x fewer block DRAM reads!) and emits all 3 msg slots in one TG.
//     Grid: (M, 1, 1), TG: 256 threads.
kernel void ink_attn_res_mg_prep_shared(device const float4* blocks [[buffer(0)]],
                                        device const float4* partial [[buffer(1)]],
                                        device const float4* queries [[buffer(2)]],
                                        device const float4* msgs [[buffer(3)]],
                                        device float4* h_out [[buffer(4)]],
                                        device bfloat4* cat_out [[buffer(5)]],
                                        constant AttnResMgArgs& a [[buffer(6)]],
                                        uint row [[threadgroup_position_in_grid]],
                                        uint tid [[thread_index_in_threadgroup]]) {
  const uint H4 = a.H >> 2, bs4 = a.b_stride >> 2;
  const uint B = a.n_prev + 1u;
  const float inv_sqrt_h = rsqrt(float(a.H));
  threadgroup float sm_s[12][8];
  threadgroup float sm_d[12][8];
  threadgroup float wts[12];
  threadgroup float sm_h[8];
  threadgroup float sm_m[3][8];

  const bool has2 = (tid + 512u) < H4;
  const float4 q0 = queries[tid];
  const float4 q1 = queries[tid + 256u];
  const float4 q2 = has2 ? queries[tid + 512u] : float4(0.0f);

  for (uint b = 0; b < B; b++) {
    device const float4* st = (b < a.n_prev) ? (blocks + size_t(b) * bs4 + size_t(row) * H4)
                                             : (partial + size_t(row) * H4);
    const float4 v0 = st[tid], v1 = st[tid + 256u];
    float s2 = dot(v0, v0) + dot(v1, v1);
    float dt = dot(v0, q0) + dot(v1, q1);
    if (has2) {
      const float4 v2 = st[tid + 512u];
      s2 += dot(v2, v2);
      dt += dot(v2, q2);
    }
    s2 = simd_sum(s2);
    dt = simd_sum(dt);
    if ((tid & 31u) == 0u) { sm_s[b][tid >> 5] = s2; sm_d[b][tid >> 5] = dt; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float inv_rms[12];
    float mx = -1e30f;
    for (uint b = 0; b < B; b++) {
      float t2 = 0.0f, td = 0.0f;
      for (uint i = 0; i < 8u; i++) { t2 += sm_s[b][i]; td += sm_d[b][i]; }
      const float ir = rsqrt(t2 / float(a.H) + a.eps);
      inv_rms[b] = ir;
      const float w = td * ir * inv_sqrt_h;
      wts[b] = w;
      mx = max(mx, w);
    }
    float den = 0.0f;
    for (uint b = 0; b < B; b++) { wts[b] = exp(wts[b] - mx); den += wts[b]; }
    const float inv_d = 1.0f / den;
    for (uint b = 0; b < B; b++) wts[b] *= inv_d * inv_rms[b];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f);
  for (uint b = 0; b < B; b++) {
    device const float4* st = (b < a.n_prev) ? (blocks + size_t(b) * bs4 + size_t(row) * H4)
                                             : (partial + size_t(row) * H4);
    const float wb = wts[b];
    acc0 = fma(st[tid], wb, acc0);
    acc1 = fma(st[tid + 256u], wb, acc1);
    if (has2) acc2 = fma(st[tid + 512u], wb, acc2);
  }
  device float4* dst = h_out + size_t(row) * H4;
  dst[tid] = acc0;
  dst[tid + 256u] = acc1;
  if (has2) dst[tid + 512u] = acc2;

  float sh = simd_sum(dot(acc0, acc0) + dot(acc1, acc1) + (has2 ? dot(acc2, acc2) : 0.0f));
  if ((tid & 31u) == 0u) sm_h[tid >> 5] = sh;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float th = 0.0f;
    for (uint i = 0; i < 8u; i++) th += sm_h[i];
    sm_h[0] = rsqrt(th / float(a.H) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv_h = sm_h[0];
  const bfloat4 h0_bf = bfloat4(acc0 * inv_h);
  const bfloat4 h1_bf = bfloat4(acc1 * inv_h);
  const bfloat4 h2_bf = has2 ? bfloat4(acc2 * inv_h) : bfloat4(bfloat(0.0f));

  for (uint s = 0u; s < 3u; s++) {
    device const float4* mr = msgs + (size_t(s) * a.M + row) * H4;
    const float4 m0 = mr[tid], m1 = mr[tid + 256u];
    const float4 m2 = has2 ? mr[tid + 512u] : float4(0.0f);
    // One sm_m row per slot: reusing a row races the next slot's partial sums against this slot's inv_m reads.
    float sv = simd_sum(dot(m0, m0) + dot(m1, m1) + (has2 ? dot(m2, m2) : 0.0f));
    if ((tid & 31u) == 0u) sm_m[s][tid >> 5] = sv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float tm = 0.0f;
      for (uint i = 0; i < 8u; i++) tm += sm_m[s][i];
      sm_m[s][0] = rsqrt(tm / float(a.H) + a.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_m = sm_m[s][0];
    device bfloat4* cdst = cat_out + size_t(s) * a.cat_stride_b4 + size_t(row) * (2u * H4);
    cdst[tid] = h0_bf;
    cdst[tid + 256u] = h1_bf;
    if (has2) cdst[tid + 512u] = h2_bf;
    cdst[H4 + tid] = bfloat4(m0 * inv_m);
    cdst[H4 + tid + 256u] = bfloat4(m1 * inv_m);
    if (has2) cdst[H4 + tid + 512u] = bfloat4(m2 * inv_m);
  }
}

// 3a. Parallel Message Gate Stage 1: compute inv_rms(h), inv_rms(msg) -> inv_out[M, 2]
//     and normalized bf16 cat buffer cat_out[M, 2*H].
// Grid: (M, 1, 1), TG: 256 threads.
struct MsgGateArgs {
  uint H, R;
  float eps;
};
kernel void ink_mg_prep(device const float4* h_in [[buffer(0)]], device const float4* msg_in [[buffer(1)]],
                        device bfloat4* cat_out [[buffer(2)]],
                        constant MsgGateArgs& a [[buffer(3)]],
                        uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint tpg [[threads_per_threadgroup]]) {
  threadgroup float sm_h[8], sm_m[8];
  const uint H4 = a.H >> 2;
  device const float4* hr = h_in + size_t(row) * H4;
  device const float4* mr = msg_in + size_t(row) * H4;
  float sh = 0.0f, sv = 0.0f;
  for (uint c = tid; c < H4; c += tpg) {
    const float4 hv = hr[c], mv = mr[c];
    sh += dot(hv, hv);
    sv += dot(mv, mv);
  }
  sh = simd_sum(sh); sv = simd_sum(sv);
  if ((tid & 31u) == 0u) { sm_h[tid >> 5] = sh; sm_m[tid >> 5] = sv; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float th = 0.0f, tm = 0.0f;
    for (uint i = 0; i < (tpg >> 5); i++) { th += sm_h[i]; tm += sm_m[i]; }
    sm_h[0] = rsqrt(th / float(a.H) + a.eps);
    sm_m[0] = rsqrt(tm / float(a.H) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv_h = sm_h[0], inv_m = sm_m[0];
  device bfloat4* dst = cat_out + size_t(row) * (2u * H4);
  for (uint c = tid; c < H4; c += tpg) {
    dst[c] = bfloat4(hr[c] * inv_h);
    dst[H4 + c] = bfloat4(mr[c] * inv_m);
  }
}

// 3b. Parallel Message Gate Stage 2: low[M, R] = cat[M, 2H] @ w_down[R, 2H]^T.
// Grid: (M, R / 4, 1) -> 16 TGs per row, TG: 128 threads (4 simdgroups, 1 rank per simdgroup).
kernel void ink_mg_down(device const bfloat4* cat_in [[buffer(0)]],
                        device const bfloat4* w_down [[buffer(1)]],
                        device float* low_out [[buffer(2)]],
                        constant MsgGateArgs& a [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x;
  const uint sg = tid >> 5, lane = tid & 31u;
  const uint r = tg.y * 4u + sg;
  if (r >= a.R) return;
  const uint H2_4 = a.H >> 1;  // (2 * H) / 4
  device const bfloat4* cr = cat_in + size_t(row) * H2_4;
  device const bfloat4* wd = w_down + size_t(r) * H2_4;
  float acc = 0.0f;
  for (uint c = lane; c < H2_4; c += 32u) acc += dot(float4(cr[c]), float4(wd[c]));
  acc = simd_sum(acc);
  if (lane == 0u) low_out[size_t(row) * a.R + r] = acc;
}

// 3c. Parallel Message Gate Stage 3 (single slot, used for MLP): out[M, H] = (1 - tanh(w_f @ low)) * h + tanh(w_g @ low) * msg.
// Grid: (M, ceil(H / 64), 1), TG: 64 threads.
kernel void ink_mg_apply(device const float* h_in [[buffer(0)]], device const float* msg_in [[buffer(1)]],
                         device const float* low_in [[buffer(2)]],
                         device const float4* w_forget [[buffer(3)]],
                         device const float4* w_gain [[buffer(4)]],
                         device float* out [[buffer(5)]],
                         constant MsgGateArgs& a [[buffer(6)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x;
  const uint c = tg.y * 64u + tid;
  threadgroup float4 lv[16];  // R=64 -> 16 float4s
  const uint R4 = a.R >> 2;
  if (tid < R4) lv[tid] = ((device const float4*)(low_in + size_t(row) * a.R))[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (c >= a.H) return;
  device const float4* wf = w_forget + size_t(c) * R4;
  device const float4* wg = w_gain + size_t(c) * R4;
  float df = 0.0f, dg = 0.0f;
  for (uint r = 0; r < R4; r++) {
    const float4 x = lv[r];
    df += dot(wf[r], x);
    dg += dot(wg[r], x);
  }
  const size_t idx = size_t(row) * a.H + c;
  out[idx] = (1.0f - tanh(df)) * h_in[idx] + tanh(dg) * msg_in[idx];
}

// 3d. Batched Q/K/V Message Gate Stage 2: 1 dispatch for all 3 slots (tg.z in 0..2).
// Grid: (M, R / 4, 3), TG: 128 threads.
struct MsgGate3Args {
  uint H, R, M, q_shared, cat_stride_b4, low_stride_f;
};
kernel void ink_mg_down3(device const bfloat4* cat_in [[buffer(0)]],
                         device const bfloat4* wd0 [[buffer(1)]],
                         device const bfloat4* wd1 [[buffer(2)]],
                         device const bfloat4* wd2 [[buffer(3)]],
                         device float* low_out [[buffer(4)]],
                         constant MsgGate3Args& a [[buffer(5)]],
                         uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x, slot = tg.z;
  const uint sg = tid >> 5, lane = tid & 31u;
  const uint r = tg.y * 4u + sg;
  if (r >= a.R) return;
  const uint H2_4 = a.H >> 1;
  device const bfloat4* cr = cat_in + size_t(slot) * a.cat_stride_b4 + size_t(row) * H2_4;
  device const bfloat4* w_down = (slot == 0u) ? wd0 : ((slot == 1u) ? wd1 : wd2);
  device const bfloat4* wd = w_down + size_t(r) * H2_4;
  float acc = 0.0f;
  for (uint c = lane; c < H2_4; c += 32u) acc += dot(float4(cr[c]), float4(wd[c]));
  acc = simd_sum(acc);
  if (lane == 0u) low_out[size_t(slot) * a.low_stride_f + size_t(row) * a.R + r] = acc;
}

// 3e. Batched Q/K/V Message Gate Stage 3: 1 dispatch for all 3 slots -> b_qkv_out (zero aliasing/barrier!).
// Grid: (M, ceil(H / 64), 3), TG: 64 threads.
kernel void ink_mg_apply3(device const float* h_in [[buffer(0)]], device const float* msg_in [[buffer(1)]],
                          device const float* low_in [[buffer(2)]],
                          device const float4* wf0 [[buffer(3)]], device const float4* wg0 [[buffer(4)]],
                          device const float4* wf1 [[buffer(5)]], device const float4* wg1 [[buffer(6)]],
                          device const float4* wf2 [[buffer(7)]], device const float4* wg2 [[buffer(8)]],
                          device float* out [[buffer(9)]],
                          constant MsgGate3Args& a [[buffer(10)]],
                          uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x, slot = tg.z;
  const uint c = tg.y * 64u + tid;
  threadgroup float4 lv[16];
  const uint R4 = a.R >> 2;
  if (tid < R4) lv[tid] = ((device const float4*)(low_in + size_t(slot) * a.low_stride_f + size_t(row) * a.R))[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (c >= a.H) return;
  device const float4* wf = ((slot == 0u) ? wf0 : ((slot == 1u) ? wf1 : wf2)) + size_t(c) * R4;
  device const float4* wg = ((slot == 0u) ? wg0 : ((slot == 1u) ? wg1 : wg2)) + size_t(c) * R4;
  float df = 0.0f, dg = 0.0f;
  for (uint r = 0; r < R4; r++) {
    const float4 x = lv[r];
    df += dot(wf[r], x);
    dg += dot(wg[r], x);
  }
  const size_t stride_m = size_t(a.M) * a.H;
  const size_t h_slot = a.q_shared ? 0u : slot;
  const size_t idx_row = size_t(row) * a.H + c;
  out[slot * stride_m + idx_row] =
      (1.0f - tanh(df)) * h_in[h_slot * stride_m + idx_row] + tanh(dg) * msg_in[slot * stride_m + idx_row];
}

// 3f. Replicated-row stream fusions (M <= 16 decode/verify): eliminate dependent barrier phases without
//     starving GPU cores by running the small-state row prep (L2-resident) in each weight-reading TG.
//     Only tg.y == 0 writes shared side-effect buffers; b_h is double-buffered (h_in -> h_out) to avoid races.

// 3f-1. Plain-visit pre-MLP fusion: attn_res(mlp) + rms(mlp) + router scores in ONE phase!
//       Grid: (M, E / 8, 1) = (M, 16, 1), TG: 256 threads. 100% bit-identical to the 2 separate phases.
struct AttnResRmsScoresArgs {
  uint H, E, M, n_prev, b_stride;
  float eps;
};
kernel void ink_attn_res_rms_scores(device const float4* blocks [[buffer(0)]],
                                    device const float4* partial [[buffer(1)]],
                                    device const float4* queries [[buffer(2)]],
                                    device const float4* wn [[buffer(3)]],
                                    device bfloat* frag [[buffer(4)]],
                                    device float4* f32_out [[buffer(5)]],
                                    device const float4* rw [[buffer(6)]],
                                    device float* scores [[buffer(7)]],
                                    constant AttnResRmsScoresArgs& a [[buffer(8)]],
                                    uint2 tg [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x, eg = tg.y;
  const bool is_writer = (eg == 0u);
  const uint H4 = a.H >> 2, bs4 = a.b_stride >> 2;
  const uint B = a.n_prev + 1u;
  const float inv_sqrt_h = rsqrt(float(a.H));
  const bool has2 = (tid + 512u) < H4;

  threadgroup float sm_s[12][8];
  threadgroup float sm_d[12][8];
  threadgroup float wts[12];
  threadgroup float sm_r[8];
  threadgroup float4 sm_mn[576];

  const float4 q0 = queries[tid];
  const float4 q1 = queries[tid + 256u];
  const float4 q2 = has2 ? queries[tid + 512u] : float4(0.0f);

  for (uint b = 0u; b < B; b++) {
    device const float4* st = (b < a.n_prev) ? (blocks + size_t(b) * bs4 + size_t(row) * H4)
                                             : (partial + size_t(row) * H4);
    const float4 v0 = st[tid], v1 = st[tid + 256u];
    float s2 = dot(v0, v0) + dot(v1, v1);
    float dt = dot(v0, q0) + dot(v1, q1);
    if (has2) {
      const float4 v2 = st[tid + 512u];
      s2 += dot(v2, v2);
      dt += dot(v2, q2);
    }
    s2 = simd_sum(s2);
    dt = simd_sum(dt);
    if ((tid & 31u) == 0u) { sm_s[b][tid >> 5] = s2; sm_d[b][tid >> 5] = dt; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float inv_rms[12];
    float mx = -1e30f;
    for (uint b = 0u; b < B; b++) {
      float t2 = 0.0f, td = 0.0f;
      for (uint i = 0u; i < 8u; i++) { t2 += sm_s[b][i]; td += sm_d[b][i]; }
      const float ir = rsqrt(t2 / float(a.H) + a.eps);
      inv_rms[b] = ir;
      const float w = td * ir * inv_sqrt_h;
      wts[b] = w;
      mx = max(mx, w);
    }
    float den = 0.0f;
    for (uint b = 0u; b < B; b++) { wts[b] = exp(wts[b] - mx); den += wts[b]; }
    const float inv_d = 1.0f / den;
    for (uint b = 0u; b < B; b++) wts[b] *= inv_d * inv_rms[b];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f);
  for (uint b = 0u; b < B; b++) {
    device const float4* st = (b < a.n_prev) ? (blocks + size_t(b) * bs4 + size_t(row) * H4)
                                             : (partial + size_t(row) * H4);
    const float wb = wts[b];
    acc0 = fma(st[tid], wb, acc0);
    acc1 = fma(st[tid + 256u], wb, acc1);
    if (has2) acc2 = fma(st[tid + 512u], wb, acc2);
  }

  float rs2 = dot(acc0, acc0) + dot(acc1, acc1) + (has2 ? dot(acc2, acc2) : 0.0f);
  rs2 = simd_sum(rs2);
  if ((tid & 31u) == 0u) sm_r[tid >> 5] = rs2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float t = 0.0f;
    for (uint i = 0u; i < 8u; i++) t += sm_r[i];
    sm_r[0] = rsqrt(t / float(a.H) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = sm_r[0];

  auto emit = [&](uint c4, float4 v) {
    const bfloat4 bv = bfloat4(v * inv * wn[c4]);
    const float4 fv = float4(bv);
    sm_mn[c4] = fv;
    if (is_writer) {
      const uint c = c4 << 2;
      frag[frag_off(row, c + 0u, a.H)] = bv.x;
      frag[frag_off(row, c + 1u, a.H)] = bv.y;
      frag[frag_off(row, c + 2u, a.H)] = bv.z;
      frag[frag_off(row, c + 3u, a.H)] = bv.w;
      f32_out[size_t(row) * H4 + c4] = fv;
    }
  };
  emit(tid, acc0);
  emit(tid + 256u, acc1);
  if (has2) emit(tid + 512u, acc2);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint sg = tid >> 5, lane = tid & 31u;
  const uint e = eg * 8u + sg;
  if (e < a.E) {
    device const float4* wr = rw + size_t(e) * H4;
    float acc = 0.0f;
    for (uint c = lane; c < H4; c += 32u) acc += dot(sm_mn[c], wr[c]);
    acc = simd_sum(acc);
    if (lane == 0u) scores[size_t(row) * a.E + e] = isigm(acc);
  }
}

// 3f-2. Fused RMSNorm -> bf16 fragment + router scores (for has_msgs MLP after mg_apply).
//       Grid: (M, E / 8, 1) = (M, 16, 1), TG: 256 threads. 100% bit-identical.
kernel void ink_rms_scores(device const float4* x [[buffer(0)]],
                           device const float4* w [[buffer(1)]],
                           device bfloat* frag [[buffer(2)]],
                           device float4* f32_out [[buffer(3)]],
                           device const float4* rw [[buffer(4)]],
                           device float* scores [[buffer(5)]],
                           constant uint& H [[buffer(6)]],
                           constant uint& E [[buffer(7)]],
                           constant float& eps [[buffer(8)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x, eg = tg.y;
  const bool is_writer = (eg == 0u);
  const uint H4 = H >> 2;
  const bool has2 = (tid + 512u) < H4;
  threadgroup float sm[8];
  threadgroup float4 sm_mn[576];

  device const float4* xr = x + size_t(row) * H4;
  const float4 x0 = xr[tid];
  const float4 x1 = xr[tid + 256u];
  const float4 x2 = has2 ? xr[tid + 512u] : float4(0.0f);
  float s2 = dot(x0, x0) + dot(x1, x1) + (has2 ? dot(x2, x2) : 0.0f);
  s2 = simd_sum(s2);
  if ((tid & 31u) == 0u) sm[tid >> 5] = s2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float t = 0.0f;
    for (uint i = 0u; i < 8u; i++) t += sm[i];
    sm[0] = rsqrt(t / float(H) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = sm[0];

  auto emit = [&](uint c4, float4 v) {
    const bfloat4 bv = bfloat4(v * inv * w[c4]);
    const float4 fv = float4(bv);
    sm_mn[c4] = fv;
    if (is_writer) {
      const uint c = c4 << 2;
      frag[frag_off(row, c + 0u, H)] = bv.x;
      frag[frag_off(row, c + 1u, H)] = bv.y;
      frag[frag_off(row, c + 2u, H)] = bv.z;
      frag[frag_off(row, c + 3u, H)] = bv.w;
      f32_out[size_t(row) * H4 + c4] = fv;
    }
  };
  emit(tid, x0);
  emit(tid + 256u, x1);
  if (has2) emit(tid + 512u, x2);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint sg = tid >> 5, lane = tid & 31u;
  const uint e = eg * 8u + sg;
  if (e < E) {
    device const float4* wr = rw + size_t(e) * H4;
    float acc = 0.0f;
    for (uint c = lane; c < H4; c += 32u) acc += dot(sm_mn[c], wr[c]);
    acc = simd_sum(acc);
    if (lane == 0u) scores[size_t(row) * E + e] = isigm(acc);
  }
}

// 3f-3. has_msgs pre-MLP fusion: attn_res + mg_prep + mg_down in ONE phase!
//       Grid: (M, R / 8, 1) = (M, 8, 1), TG: 256 threads (all 8 simdgroups compute 8 ranks/TG). 100% bit-identical.
struct AttnResMgDownArgs {
  uint H, R, M, n_prev, b_stride;
  float eps;
};
kernel void ink_attn_res_mg_down(device const float4* blocks [[buffer(0)]],
                                 device const float4* partial [[buffer(1)]],
                                 device const float4* queries [[buffer(2)]],
                                 device const float4* msgs [[buffer(3)]],
                                 device float4* mh_out [[buffer(4)]],
                                 device const bfloat4* w_down [[buffer(5)]],
                                 device float* low_out [[buffer(6)]],
                                 constant AttnResMgDownArgs& a [[buffer(7)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x, rg = tg.y;
  const bool is_writer = (rg == 0u);
  const uint H4 = a.H >> 2, bs4 = a.b_stride >> 2;
  const uint B = a.n_prev + 1u;
  const float inv_sqrt_h = rsqrt(float(a.H));
  const bool has2 = (tid + 512u) < H4;

  threadgroup float sm_s[12][8];
  threadgroup float sm_d[12][8];
  threadgroup float wts[12];
  threadgroup float sm_h[8], sm_m[8];
  threadgroup bfloat4 sm_cat[1152];

  const float4 q0 = queries[tid];
  const float4 q1 = queries[tid + 256u];
  const float4 q2 = has2 ? queries[tid + 512u] : float4(0.0f);

  for (uint b = 0u; b < B; b++) {
    device const float4* st = (b < a.n_prev) ? (blocks + size_t(b) * bs4 + size_t(row) * H4)
                                             : (partial + size_t(row) * H4);
    const float4 v0 = st[tid], v1 = st[tid + 256u];
    float s2 = dot(v0, v0) + dot(v1, v1);
    float dt = dot(v0, q0) + dot(v1, q1);
    if (has2) {
      const float4 v2 = st[tid + 512u];
      s2 += dot(v2, v2);
      dt += dot(v2, q2);
    }
    s2 = simd_sum(s2);
    dt = simd_sum(dt);
    if ((tid & 31u) == 0u) { sm_s[b][tid >> 5] = s2; sm_d[b][tid >> 5] = dt; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float inv_rms[12];
    float mx = -1e30f;
    for (uint b = 0u; b < B; b++) {
      float t2 = 0.0f, td = 0.0f;
      for (uint i = 0u; i < 8u; i++) { t2 += sm_s[b][i]; td += sm_d[b][i]; }
      const float ir = rsqrt(t2 / float(a.H) + a.eps);
      inv_rms[b] = ir;
      const float w = td * ir * inv_sqrt_h;
      wts[b] = w;
      mx = max(mx, w);
    }
    float den = 0.0f;
    for (uint b = 0u; b < B; b++) { wts[b] = exp(wts[b] - mx); den += wts[b]; }
    const float inv_d = 1.0f / den;
    for (uint b = 0u; b < B; b++) wts[b] *= inv_d * inv_rms[b];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f);
  for (uint b = 0u; b < B; b++) {
    device const float4* st = (b < a.n_prev) ? (blocks + size_t(b) * bs4 + size_t(row) * H4)
                                             : (partial + size_t(row) * H4);
    const float wb = wts[b];
    acc0 = fma(st[tid], wb, acc0);
    acc1 = fma(st[tid + 256u], wb, acc1);
    if (has2) acc2 = fma(st[tid + 512u], wb, acc2);
  }
  if (is_writer) {
    device float4* dst = mh_out + size_t(row) * H4;
    dst[tid] = acc0;
    dst[tid + 256u] = acc1;
    if (has2) dst[tid + 512u] = acc2;
  }

  device const float4* mr = msgs + size_t(row) * H4;
  const float4 m0 = mr[tid], m1 = mr[tid + 256u];
  const float4 m2 = has2 ? mr[tid + 512u] : float4(0.0f);
  float sh = simd_sum(dot(acc0, acc0) + dot(acc1, acc1) + (has2 ? dot(acc2, acc2) : 0.0f));
  float sv = simd_sum(dot(m0, m0) + dot(m1, m1) + (has2 ? dot(m2, m2) : 0.0f));
  if ((tid & 31u) == 0u) { sm_h[tid >> 5] = sh; sm_m[tid >> 5] = sv; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float th = 0.0f, tm = 0.0f;
    for (uint i = 0u; i < 8u; i++) { th += sm_h[i]; tm += sm_m[i]; }
    sm_h[0] = rsqrt(th / float(a.H) + a.eps);
    sm_m[0] = rsqrt(tm / float(a.H) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv_h = sm_h[0], inv_m = sm_m[0];
  sm_cat[tid] = bfloat4(acc0 * inv_h);
  sm_cat[tid + 256u] = bfloat4(acc1 * inv_h);
  if (has2) sm_cat[tid + 512u] = bfloat4(acc2 * inv_h);
  sm_cat[H4 + tid] = bfloat4(m0 * inv_m);
  sm_cat[H4 + tid + 256u] = bfloat4(m1 * inv_m);
  if (has2) sm_cat[H4 + tid + 512u] = bfloat4(m2 * inv_m);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint sg = tid >> 5, lane = tid & 31u;
  const uint r = rg * 8u + sg;
  if (r < a.R) {
    const uint H2_4 = a.H >> 1;
    device const bfloat4* wd = w_down + size_t(r) * H2_4;
    float acc = 0.0f;
    for (uint c = lane; c < H2_4; c += 32u) acc += dot(float4(sm_cat[c]), float4(wd[c]));
    acc = simd_sum(acc);
    if (lane == 0u) low_out[size_t(row) * a.R + r] = acc;
  }
}

// 3f-4. has_msgs pre-Attn fusion: attn_res + mg_prep + mg_down3 in ONE phase!
//       Grid: (M, R / 4, 3) = (M, 16, 3), TG: 256 threads (sg < 4 computes 4 ranks/TG). 100% bit-identical.
struct AttnResMgDown3Args {
  uint H, R, M, n_prev, b_stride, q_shared, low_stride_f;
  float eps;
};
kernel void ink_attn_res_mg_down3(device const float4* blocks [[buffer(0)]],
                                  device const float4* partial [[buffer(1)]],
                                  device const float4* queries [[buffer(2)]],
                                  device const float4* msgs [[buffer(3)]],
                                  device float4* h_out [[buffer(4)]],
                                  device const bfloat4* wd0 [[buffer(5)]],
                                  device const bfloat4* wd1 [[buffer(6)]],
                                  device const bfloat4* wd2 [[buffer(7)]],
                                  device float* low_out [[buffer(8)]],
                                  constant AttnResMgDown3Args& a [[buffer(9)]],
                                  uint3 tg [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x, rg = tg.y, slot = tg.z;
  const bool is_writer = (rg == 0u);
  const uint H4 = a.H >> 2, bs4 = a.b_stride >> 2;
  const uint qi = a.q_shared ? 0u : slot;
  const uint B = a.n_prev + 1u;
  const float inv_sqrt_h = rsqrt(float(a.H));
  const bool has2 = (tid + 512u) < H4;

  threadgroup float sm_s[12][8];
  threadgroup float sm_d[12][8];
  threadgroup float wts[12];
  threadgroup float sm_h[8], sm_m[8];
  threadgroup bfloat4 sm_cat[1152];

  device const float4* q = queries + size_t(qi) * H4;
  const float4 q0 = q[tid];
  const float4 q1 = q[tid + 256u];
  const float4 q2 = has2 ? q[tid + 512u] : float4(0.0f);

  for (uint b = 0u; b < B; b++) {
    device const float4* st = (b < a.n_prev) ? (blocks + size_t(b) * bs4 + size_t(row) * H4)
                                             : (partial + size_t(row) * H4);
    const float4 v0 = st[tid], v1 = st[tid + 256u];
    float s2 = dot(v0, v0) + dot(v1, v1);
    float dt = dot(v0, q0) + dot(v1, q1);
    if (has2) {
      const float4 v2 = st[tid + 512u];
      s2 += dot(v2, v2);
      dt += dot(v2, q2);
    }
    s2 = simd_sum(s2);
    dt = simd_sum(dt);
    if ((tid & 31u) == 0u) { sm_s[b][tid >> 5] = s2; sm_d[b][tid >> 5] = dt; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float inv_rms[12];
    float mx = -1e30f;
    for (uint b = 0u; b < B; b++) {
      float t2 = 0.0f, td = 0.0f;
      for (uint i = 0u; i < 8u; i++) { t2 += sm_s[b][i]; td += sm_d[b][i]; }
      const float ir = rsqrt(t2 / float(a.H) + a.eps);
      inv_rms[b] = ir;
      const float w = td * ir * inv_sqrt_h;
      wts[b] = w;
      mx = max(mx, w);
    }
    float den = 0.0f;
    for (uint b = 0u; b < B; b++) { wts[b] = exp(wts[b] - mx); den += wts[b]; }
    const float inv_d = 1.0f / den;
    for (uint b = 0u; b < B; b++) wts[b] *= inv_d * inv_rms[b];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f);
  for (uint b = 0u; b < B; b++) {
    device const float4* st = (b < a.n_prev) ? (blocks + size_t(b) * bs4 + size_t(row) * H4)
                                             : (partial + size_t(row) * H4);
    const float wb = wts[b];
    acc0 = fma(st[tid], wb, acc0);
    acc1 = fma(st[tid + 256u], wb, acc1);
    if (has2) acc2 = fma(st[tid + 512u], wb, acc2);
  }
  if (is_writer && (!a.q_shared || slot == 0u)) {
    device float4* dst = h_out + (size_t(qi) * a.M + row) * H4;
    dst[tid] = acc0;
    dst[tid + 256u] = acc1;
    if (has2) dst[tid + 512u] = acc2;
  }

  device const float4* mr = msgs + (size_t(slot) * a.M + row) * H4;
  const float4 m0 = mr[tid], m1 = mr[tid + 256u];
  const float4 m2 = has2 ? mr[tid + 512u] : float4(0.0f);
  float sh = simd_sum(dot(acc0, acc0) + dot(acc1, acc1) + (has2 ? dot(acc2, acc2) : 0.0f));
  float sv = simd_sum(dot(m0, m0) + dot(m1, m1) + (has2 ? dot(m2, m2) : 0.0f));
  if ((tid & 31u) == 0u) { sm_h[tid >> 5] = sh; sm_m[tid >> 5] = sv; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float th = 0.0f, tm = 0.0f;
    for (uint i = 0u; i < 8u; i++) { th += sm_h[i]; tm += sm_m[i]; }
    sm_h[0] = rsqrt(th / float(a.H) + a.eps);
    sm_m[0] = rsqrt(tm / float(a.H) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv_h = sm_h[0], inv_m = sm_m[0];
  sm_cat[tid] = bfloat4(acc0 * inv_h);
  sm_cat[tid + 256u] = bfloat4(acc1 * inv_h);
  if (has2) sm_cat[tid + 512u] = bfloat4(acc2 * inv_h);
  sm_cat[H4 + tid] = bfloat4(m0 * inv_m);
  sm_cat[H4 + tid + 256u] = bfloat4(m1 * inv_m);
  if (has2) sm_cat[H4 + tid + 512u] = bfloat4(m2 * inv_m);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint sg = tid >> 5, lane = tid & 31u;
  const uint r = rg * 8u + sg;
  if (r < a.R) {
    const uint H2_4 = a.H >> 1;
    device const bfloat4* w_down = (slot == 0u) ? wd0 : ((slot == 1u) ? wd1 : wd2);
    device const bfloat4* wd = w_down + size_t(r) * H2_4;
    float acc = 0.0f;
    for (uint c = lane; c < H2_4; c += 32u) acc += dot(float4(sm_cat[c]), float4(wd[c]));
    acc = simd_sum(acc);
    if (lane == 0u) low_out[size_t(slot) * a.low_stride_f + size_t(row) * a.R + r] = acc;
  }
}

// 4. RMSNorm (float4) -> bf16 fragment (and optional f32 copy for router).
// Grid: (M, 1, 1), TG: 256 threads.
kernel void ink_rms_frag(device const float4* x [[buffer(0)]], device const float4* w [[buffer(1)]],
                         device bfloat* frag [[buffer(2)]], device float4* f32_out [[buffer(3)]],
                         constant uint& H [[buffer(4)]], constant float& eps [[buffer(5)]],
                         constant uint& write_f32 [[buffer(6)]],
                         uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint tpg [[threads_per_threadgroup]]) {
  threadgroup float sm[8];
  const uint H4 = H >> 2;
  device const float4* xr = x + size_t(row) * H4;
  float s2 = 0.0f;
  for (uint c = tid; c < H4; c += tpg) { const float4 v = xr[c]; s2 += dot(v, v); }
  s2 = simd_sum(s2);
  if ((tid & 31u) == 0u) sm[tid >> 5] = s2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float t = 0.0f; for (uint i = 0; i < (tpg >> 5); i++) t += sm[i];
    sm[0] = rsqrt(t / float(H) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = sm[0];
  for (uint c4 = tid; c4 < H4; c4 += tpg) {
    const bfloat4 bv = bfloat4(xr[c4] * inv * w[c4]);
    const uint c = c4 << 2;
    frag[frag_off(row, c + 0u, H)] = bv.x;
    frag[frag_off(row, c + 1u, H)] = bv.y;
    frag[frag_off(row, c + 2u, H)] = bv.z;
    frag[frag_off(row, c + 3u, H)] = bv.w;
    if (write_f32) f32_out[size_t(row) * H4 + c4] = float4(bv);
  }
}

// 4b. Batched 3-slot RMSNorm (float4) -> bf16 fragments for Q/K/V after mg_apply3 (1 dispatch instead of 3).
// Grid: (M, 3, 1), TG: 256 threads. 100% bit-identical to 3x ink_rms_frag.
kernel void ink_rms_frag3(device const float4* x [[buffer(0)]],
                          device const float4* w0 [[buffer(1)]],
                          device const float4* w1 [[buffer(2)]],
                          device const float4* w2 [[buffer(3)]],
                          device bfloat* frag0 [[buffer(4)]],
                          device bfloat* frag1 [[buffer(5)]],
                          device bfloat* frag2 [[buffer(6)]],
                          constant uint& H [[buffer(7)]],
                          constant uint& M [[buffer(8)]],
                          constant float& eps [[buffer(9)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x, slot = tg.y;
  const uint H4 = H >> 2;
  threadgroup float sm[8];
  device const float4* xr = x + (size_t(slot) * M + row) * H4;
  float s2 = 0.0f;
  for (uint c = tid; c < H4; c += 256u) { const float4 v = xr[c]; s2 += dot(v, v); }
  s2 = simd_sum(s2);
  if ((tid & 31u) == 0u) sm[tid >> 5] = s2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float t = 0.0f; for (uint i = 0u; i < 8u; i++) t += sm[i];
    sm[0] = rsqrt(t / float(H) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = sm[0];
  device const float4* w = (slot == 0u) ? w0 : ((slot == 1u) ? w1 : w2);
  device bfloat* frag = (slot == 0u) ? frag0 : ((slot == 1u) ? frag1 : frag2);
  for (uint c4 = tid; c4 < H4; c4 += 256u) {
    const bfloat4 bv = bfloat4(xr[c4] * inv * w[c4]);
    const uint c = c4 << 2;
    frag[frag_off(row, c + 0u, H)] = bv.x;
    frag[frag_off(row, c + 1u, H)] = bv.y;
    frag[frag_off(row, c + 2u, H)] = bv.z;
    frag[frag_off(row, c + 3u, H)] = bv.w;
  }
}

// 5. Q L2-clip (+ log_scale) + Rel L2-clip (+ log_scale) + rel_logits_proj (bfloat4) + gate sigmoid + sinks.
// Grid: (M, nh, 1), TG: 128 threads (1 head per TG).
struct QRelArgs {
  uint nh, hd, d_rel, rel_extent, qgrs_n, pos0, is_global;
  float qk_clip, rel_clip, log_n_floor, log_alpha;
};
kernel void ink_q_rel(device const float* qgrs [[buffer(0)]],
                      device const float* q_scale [[buffer(1)]],    // [nh, hd]
                      device const float* rel_scale [[buffer(2)]],  // [nh, d_rel]
                      device const bfloat4* rel_proj [[buffer(3)]], // [d_rel, rel_extent/4]
                      device float* Q_out [[buffer(4)]],            // [M, nh, hd]
                      device float4* rel_logits [[buffer(5)]],      // [M, nh, rel_extent/4]
                      device float* gate_out [[buffer(6)]],         // [M, nh * hd]
                      device float* sink_out [[buffer(7)]],         // [M, nh]
                      constant QRelArgs& a [[buffer(8)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x, h = tg.y;
  const uint pos = a.pos0 + row;
  const float tau = a.is_global
      ? (1.0f + a.log_alpha * log(max(float(pos + 1u) / a.log_n_floor, 1.0f)))
      : 1.0f;

  device const float* qr = qgrs + size_t(row) * a.qgrs_n;
  device const float* q_in = qr + h * a.hd;
  device const float* g_in = qr + a.nh * a.hd + h * a.hd;
  device const float* r_in = qr + 2u * a.nh * a.hd + h * a.d_rel;
  device const float* s_in = qr + 2u * a.nh * a.hd + a.nh * a.d_rel;

  threadgroup float sm[4];
  threadgroup float rv[16];
  const float qv = q_in[tid] * q_scale[h * a.hd + tid];
  float n2 = simd_sum(qv * qv);
  if ((tid & 31u) == 0u) sm[tid >> 5] = n2;
  if (tid < a.d_rel) rv[tid] = r_in[tid] * rel_scale[h * a.d_rel + tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    const float norm = sqrt(sm[0] + sm[1] + sm[2] + sm[3]);
    sm[0] = min(a.qk_clip / (norm + 1e-6f), 1.0f) * tau;
    float rn2 = 0.0f;
    for (uint d = 0; d < a.d_rel; d++) rn2 += rv[d] * rv[d];
    const float mult = min(a.rel_clip / (sqrt(rn2) + 1e-6f), 1.0f) * tau;
    for (uint d = 0; d < a.d_rel; d++) rv[d] = float(bfloat(rv[d] * mult));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  Q_out[(size_t(row) * a.nh + h) * a.hd + tid] = float(bfloat(qv * sm[0]));
  gate_out[(size_t(row) * a.nh + h) * a.hd + tid] = isigm(g_in[tid]);
  if (tid == 0u) sink_out[size_t(row) * a.nh + h] = s_in[h];

  // Vectorized bfloat4: only compute needed relative distances up to min(pos + 1, rel_extent)
  const uint ext4 = a.rel_extent >> 2;
  const uint ext4_need = (min(pos + 1u, a.rel_extent) + 3u) >> 2;
  device float4* rl = rel_logits + (size_t(row) * a.nh + h) * ext4;
  for (uint e4 = tid; e4 < ext4_need; e4 += 128u) {
    float4 acc = float4(0.0f);
    for (uint d = 0; d < a.d_rel; d++) acc = fma(rv[d], float4(rel_proj[d * ext4 + e4]), acc);
    rl[e4] = acc;
  }
}

// 5b. Prefill Q L2-clip + gate sigmoid + sinks + bf16 rel_vec[M*nh, d_rel] (feeds ink_q_rel_mma).
// Grid: (M, nh, 1), TG: 128 threads.
kernel void ink_q_rel_prep(device const float* qgrs [[buffer(0)]],
                           device const float* q_scale [[buffer(1)]],
                           device const float* rel_scale [[buffer(2)]],
                           device float* Q_out [[buffer(3)]],
                           device bfloat* rel_vec [[buffer(4)]],
                           device float* gate_out [[buffer(5)]],
                           device float* sink_out [[buffer(6)]],
                           constant QRelArgs& a [[buffer(7)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x, h = tg.y;
  const uint pos = a.pos0 + row;
  const float tau = a.is_global
      ? (1.0f + a.log_alpha * log(max(float(pos + 1u) / a.log_n_floor, 1.0f)))
      : 1.0f;

  device const float* qr = qgrs + size_t(row) * a.qgrs_n;
  device const float* q_in = qr + h * a.hd;
  device const float* g_in = qr + a.nh * a.hd + h * a.hd;
  device const float* r_in = qr + 2u * a.nh * a.hd + h * a.d_rel;
  device const float* s_in = qr + 2u * a.nh * a.hd + a.nh * a.d_rel;

  threadgroup float sm[4];
  threadgroup float rv[16];
  const float qv = q_in[tid] * q_scale[h * a.hd + tid];
  float n2 = simd_sum(qv * qv);
  if ((tid & 31u) == 0u) sm[tid >> 5] = n2;
  if (tid < a.d_rel) rv[tid] = r_in[tid] * rel_scale[h * a.d_rel + tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    const float norm = sqrt(sm[0] + sm[1] + sm[2] + sm[3]);
    sm[0] = min(a.qk_clip / (norm + 1e-6f), 1.0f) * tau;
    float rn2 = 0.0f;
    for (uint d = 0; d < a.d_rel; d++) rn2 += rv[d] * rv[d];
    const float mult = min(a.rel_clip / (sqrt(rn2) + 1e-6f), 1.0f) * tau;
    for (uint d = 0; d < a.d_rel; d++) rv[d] = float(bfloat(rv[d] * mult));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  Q_out[(size_t(row) * a.nh + h) * a.hd + tid] = float(bfloat(qv * sm[0]));
  gate_out[(size_t(row) * a.nh + h) * a.hd + tid] = isigm(g_in[tid]);
  if (tid == 0u) sink_out[size_t(row) * a.nh + h] = s_in[h];
  if (tid < a.d_rel) rel_vec[(size_t(row) * a.nh + h) * a.d_rel + tid] = bfloat(rv[tid]);
}

// 6. Parallel K/V short convolution (static on local, dynamic on global) + K L2-clip / V RMSNorm -> KV cache.
//    Also records raw inputs into conv_hist[3 + MAXM, C] for exact speculative rollback.
//    Reads conv_state_in (read-only) and row M-1 writes conv_state_out (double-buffered, zero race).
// Grid: (M, nkv, 1), TG: 128 threads.
struct KvConvArgs {
  uint M, nkv, hd, k_stride, pos0, cap, is_global, is_key, ks, kv8, fork_pos;
  float qk_clip, eps;
};
kernel void ink_kv_conv_cache(device const float* in_kv [[buffer(0)]],
                              device const float* sconv_w [[buffer(1)]],
                              device const float* conv_state_in [[buffer(2)]],
                              device bfloat* kv_cache [[buffer(3)]],
                              device char* kv_i8 [[buffer(4)]],
                              device half* kv_sc [[buffer(5)]],
                              device float* conv_hist [[buffer(6)]],
                              constant KvConvArgs& a [[buffer(7)]],
                              device float* conv_state_out [[buffer(8)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint m = tg.x, h = tg.y;
  const uint C = a.nkv * a.hd;
  const uint ch = h * a.hd + tid;
  const bool save_hist = (a.M <= 16u);

  const float x  = in_kv[size_t(m) * a.k_stride + ch];
  const float h2 = (m >= 1u) ? in_kv[size_t(m - 1u) * a.k_stride + ch] : conv_state_in[2u * C + ch];
  const float h1 = (m >= 2u) ? in_kv[size_t(m - 2u) * a.k_stride + ch] : conv_state_in[(1u + m) * C + ch];
  const float h0 = (m >= 3u) ? in_kv[size_t(m - 3u) * a.k_stride + ch] : conv_state_in[(0u + m) * C + ch];

  if (save_hist) {
    if (m == 0u) {
      conv_hist[0u * C + ch] = h0;
      conv_hist[1u * C + ch] = h1;
      conv_hist[2u * C + ch] = h2;
    }
    conv_hist[(3u + m) * C + ch] = x;
  }
  if (m + 1u == a.M) {
    conv_state_out[0u * C + ch] = h1;
    conv_state_out[1u * C + ch] = h2;
    conv_state_out[2u * C + ch] = x;
  }

  const float inv_denom = a.is_global ? rsqrt(float(2u * a.ks + 1u)) : rsqrt(float(a.ks + 1u));
  float y;
  if (!a.is_global) {
    const float sw3 = sconv_w[ch * 4u + 0u];
    const float sw2 = sconv_w[ch * 4u + 1u];
    const float sw1 = sconv_w[ch * 4u + 2u];
    const float sw0 = sconv_w[ch * 4u + 3u];
    y = (sw0 * x + sw1 * h2 + sw2 * h1 + sw3 * h0 + x) * inv_denom;
  } else {
    const float sw0 = sconv_w[0u * C + ch];
    const float sw1 = sconv_w[1u * C + ch];
    const float sw2 = sconv_w[2u * C + ch];
    const float sw3 = sconv_w[3u * C + ch];
    device const float* dw = in_kv + size_t(m) * a.k_stride + C + h * 4u;
    y = ((sw0 + dw[0]) * x + (sw1 + dw[1]) * h2 + (sw2 + dw[2]) * h1 + (sw3 + dw[3]) * h0 + x) * inv_denom;
  }

  threadgroup float sm[4], sm2[4];
  float n2 = simd_sum(y * y);
  if ((tid & 31u) == 0u) sm[tid >> 5] = n2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    const float tot = sm[0] + sm[1] + sm[2] + sm[3];
    sm[0] = a.is_key ? min(a.qk_clip / (sqrt(tot) + 1e-6f), 1.0f) : rsqrt(tot / float(a.hd) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float v = y * sm[0];
  const uint slot = ((a.pos0 + m) - a.fork_pos) % a.cap;
  if (a.kv8) {
    float am = simd_max(fabs(v));
    if ((tid & 31u) == 0u) sm2[tid >> 5] = am;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      const float mx = max(max(sm2[0], sm2[1]), max(sm2[2], sm2[3]));
      const half sc = half(max(mx / 127.0f, 1e-6f));
      sm2[0] = float(sc);
      kv_sc[size_t(slot) * a.nkv + h] = sc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int q = int(clamp(rint(v / sm2[0]), -127.0f, 127.0f));
    kv_i8[(size_t(slot) * a.nkv + h) * a.hd + tid] = char(q);
  } else {
    kv_cache[(size_t(slot) * a.nkv + h) * a.hd + tid] = bfloat(v);
  }
}

// 7a. Barrier-free Simdgroup Split-K GQA Attention (supports both BF16 and kv8 int8 KV cache).
//     Grid: (M, nh, n_splits), TG: 128 threads (4 simdgroups).
struct AttnArgs {
  uint M, nh, nkv, hd, rel_extent, pos0, cap, window, is_local, n_splits, kv8;
};
kernel void ink_attn_split(device const float4* Q [[buffer(0)]],         // [M, nh, 32]
                           device const bfloat4* K_cache [[buffer(1)]],  // [cap, nkv, 32]
                           device const bfloat4* V_cache [[buffer(2)]],  // [cap, nkv, 32]
                           device const float* rel_logits [[buffer(3)]], // [M, nh, rel_extent]
                           device float* part_out [[buffer(4)]],         // [M, nh, n_splits, 130]
                           device const char4* K_i8 [[buffer(5)]],
                           device const half* K_sc [[buffer(6)]],
                           device const char4* V_i8 [[buffer(7)]],
                           device const half* V_sc [[buffer(8)]],
                           constant AttnArgs& a [[buffer(9)]],
                           uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x, h = tg.y, sp = tg.z;
  const uint sg = tid >> 5, lane = tid & 31u;
  const uint kvh = h / (a.nh / a.nkv);
  const uint pos = a.pos0 + row;
  const uint t_start = (a.is_local && pos + 1u > a.window) ? (pos + 1u - a.window) : 0u;
  const uint total_k = pos - t_start + 1u;
  const uint per_sp = (total_k + a.n_splits - 1u) / a.n_splits;
  const uint t_lo = t_start + sp * per_sp;
  const uint t_hi = min(pos + 1u, t_lo + per_sp);

  const float scale = rsqrt(float(a.hd));
  const float4 q4 = Q[(size_t(row) * a.nh + h) * 32u + lane] * scale;
  device const float* rl = rel_logits + (size_t(row) * a.nh + h) * a.rel_extent;

  float m_max = -1e30f;
  float l_sum = 0.0f;
  float4 o4 = float4(0.0f);

  for (uint t = t_lo + sg; t < t_hi; t += 4u) {
    const uint slot = t % a.cap;
    const size_t sc_off = size_t(slot) * a.nkv + kvh;
    const size_t kv_off = sc_off * 32u + lane;
    float4 k4, v4;
    if (a.kv8) {
      k4 = float4(K_i8[kv_off]) * float(K_sc[sc_off]);
      v4 = float4(V_i8[kv_off]) * float(V_sc[sc_off]);
    } else {
      k4 = float4(K_cache[kv_off]);
      v4 = float4(V_cache[kv_off]);
    }
    float s = simd_sum(dot(q4, k4));
    const uint dist = pos - t;
    if (dist < a.rel_extent) s += rl[dist];
    const float m_new = max(m_max, s);
    const float alpha = (l_sum > 0.0f) ? exp(m_max - m_new) : 0.0f;
    const float p = exp(s - m_new);
    o4 = fma(p, v4, o4 * alpha);
    l_sum = l_sum * alpha + p;
    m_max = m_new;
  }

  threadgroup float sm_m[4], sm_l[4];
  threadgroup float4 sm_o[4][32];
  if (lane == 0u) { sm_m[sg] = m_max; sm_l[sg] = l_sum; }
  sm_o[sg][lane] = o4;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0u) {
    float fm = -1e30f, fl = 0.0f;
    float4 fo = float4(0.0f);
    for (uint s = 0; s < 4u; s++) {
      const float sl = sm_l[s];
      if (sl <= 0.0f) continue;
      const float smv = sm_m[s];
      const float mn = max(fm, smv);
      const float a0 = (fl > 0.0f) ? exp(fm - mn) : 0.0f;
      const float a1 = exp(smv - mn);
      fo = fo * a0 + sm_o[s][lane] * a1;
      fl = fl * a0 + sl * a1;
      fm = mn;
    }
    device float* dst = part_out + ((size_t(row) * a.nh + h) * a.n_splits + sp) * 130u;
    if (lane == 0u) { dst[0] = fm; dst[1] = fl; }
    ((device float4*)(dst + 2u))[lane] = fo;
  }
}

// 7b. Reduce n_splits attention partials + sink + sigmoid gate -> bf16 fragment for wo_ud.
// Grid: (M, nh, 1), TG: 32 threads (1 simdgroup, 1 float4 per lane = 128 dims!).
kernel void ink_attn_reduce(device const float* part_in [[buffer(0)]],  // [M, nh, n_splits, 130]
                            device const float4* gate [[buffer(1)]],    // [M, nh, 32]
                            device const float* sinks [[buffer(2)]],    // [M, nh]
                            device bfloat* out_frag [[buffer(3)]],
                            constant AttnArgs& a [[buffer(4)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  const uint row = tg.x, h = tg.y;
  float fm = sinks[size_t(row) * a.nh + h];
  float fl = 1.0f;  // sink has zero value vector
  float4 fo = float4(0.0f);
  device const float* base = part_in + (size_t(row) * a.nh + h) * a.n_splits * 130u;
  for (uint sp = 0; sp < a.n_splits; sp++) {
    device const float* src = base + sp * 130u;
    const float sl = src[1];
    if (sl <= 0.0f) continue;
    const float smv = src[0];
    const float mn = max(fm, smv);
    const float a0 = exp(fm - mn);
    const float a1 = exp(smv - mn);
    fo = fo * a0 + ((device const float4*)(src + 2u))[lane] * a1;
    fl = fl * a0 + sl * a1;
    fm = mn;
  }
  const float4 g4 = gate[(size_t(row) * a.nh + h) * 32u + lane];
  const bfloat4 o4 = bfloat4((fo / fl) * g4);
  const uint col0 = h * a.hd + lane * 4u;
  const uint K = a.nh * a.hd;
  out_frag[frag_off(row, col0 + 0u, K)] = o4.x;
  out_frag[frag_off(row, col0 + 1u, K)] = o4.y;
  out_frag[frag_off(row, col0 + 2u, K)] = o4.z;
  out_frag[frag_off(row, col0 + 3u, K)] = o4.w;
}

// 8. Parallel post-attention / post-MLP static sconv + residual add / block boundary save.
//    Reads conv_state_in (read-only) and row M-1 writes conv_state_out (double-buffered, zero race).
// Grid: (M, ceil(H / 256), 1), TG: 256 threads.
struct PostConvArgs {
  uint M, H, ks, is_boundary, is_mlp, b_slot, b_stride;
};
kernel void ink_post_sconv_add(device const float* y_in [[buffer(0)]],
                               device const float* sconv_w [[buffer(1)]],
                               device const float* conv_state_in [[buffer(2)]],
                               device float* h_res [[buffer(3)]],
                               device float* blocks [[buffer(4)]],
                               device float* conv_hist [[buffer(5)]],
                               constant PostConvArgs& a [[buffer(6)]],
                               device float* conv_state_out [[buffer(7)]],
                               uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint m = tg.x;
  const uint ch = tg.y * 256u + tid;
  if (ch >= a.H) return;

  const float x  = y_in[size_t(m) * a.H + ch];
  const float h2 = (m >= 1u) ? y_in[size_t(m - 1u) * a.H + ch] : conv_state_in[2u * a.H + ch];
  const float h1 = (m >= 2u) ? y_in[size_t(m - 2u) * a.H + ch] : conv_state_in[(1u + m) * a.H + ch];
  const float h0 = (m >= 3u) ? y_in[size_t(m - 3u) * a.H + ch] : conv_state_in[(0u + m) * a.H + ch];

  const bool save_hist = (a.M <= 16u);
  if (save_hist) {
    if (m == 0u) {
      conv_hist[0u * a.H + ch] = h0;
      conv_hist[1u * a.H + ch] = h1;
      conv_hist[2u * a.H + ch] = h2;
    }
    conv_hist[(3u + m) * a.H + ch] = x;
  }
  if (m + 1u == a.M) {
    conv_state_out[0u * a.H + ch] = h1;
    conv_state_out[1u * a.H + ch] = h2;
    conv_state_out[2u * a.H + ch] = x;
  }

  const float cv = post_sconv_value(sconv_w, ch, a.ks, x, h2, h1, h0);

  const float r_old = h_res[size_t(m) * a.H + ch];
  if (!a.is_mlp) {
    if (a.is_boundary) {
      blocks[size_t(a.b_slot) * a.b_stride + size_t(m) * a.H + ch] = r_old;
      h_res[size_t(m) * a.H + ch] = cv;
    } else {
      h_res[size_t(m) * a.H + ch] = r_old + cv;
    }
  } else {
    h_res[size_t(m) * a.H + ch] = float(bfloat(r_old + cv));
  }
}

// 9a. Parallel FP32 router logits (float4 vectorized): scores[m, e] = sigmoid(dot(mn[m], rw[e])).
// Grid: (M, E / 8, 1), TG: 256 threads (8 simdgroups, 1 expert per simdgroup).
kernel void ink_moe_scores(device const float4* mn [[buffer(0)]],
                           device const float4* rw [[buffer(1)]],
                           device float* scores [[buffer(2)]],
                           constant uint& H [[buffer(3)]], constant uint& E [[buffer(4)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint2 tpg2 [[threads_per_threadgroup]]) {
  const uint row = tg.x;
  const uint sg = tid >> 5, lane = tid & 31u;
  const uint e = tg.y * (tpg2.x >> 5) + sg;
  if (e >= E) return;
  const uint H4 = H >> 2;
  device const float4* mr = mn + size_t(row) * H4;
  device const float4* wr = rw + size_t(e) * H4;
  float acc = 0.0f;
  for (uint c = lane; c < H4; c += 32u) acc += dot(mr[c], wr[c]);
  acc = simd_sum(acc);
  if (lane == 0u) scores[size_t(row) * E + e] = isigm(acc);
}

// 9b. Fast SIMD Top-8 selection + renorm + compact active expert list (E=128 = 4 simdgroups x 32).
// Grid: (1, 1, 1), TG: 128 threads.
struct RouteArgs {
  uint M, H, E, topk;
  float route_scale;
};
kernel void ink_moe_route(device const float* scores [[buffer(0)]],
                          device const float* rb [[buffer(1)]],
                          device uint* top_idx [[buffer(2)]],
                          device float* top_w [[buffer(3)]],
                          device uint* slots [[buffer(4)]],
                          device uint* nslots [[buffer(5)]],
                          device uint* rmask [[buffer(6)]],
                          constant RouteArgs& a [[buffer(7)]],
                          uint e [[thread_index_in_threadgroup]]) {
  const uint sg = e >> 5, lane = e & 31u;
  threadgroup float sm_v[4];
  threadgroup uint sm_i[4];
  threadgroup uint picked_i[8];
  threadgroup float picked_s[8];
  threadgroup uint e_mask[128];
  e_mask[e] = 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint m = 0; m < a.M; m++) {
    const float s = scores[size_t(m) * a.E + e];
    float k_val = s + rb[e];

    for (uint k = 0; k < a.topk; k++) {
      float bv = k_val;
      uint bi = e;
      for (ushort d = 16; d > 0; d >>= 1) {
        const float ov = simd_shuffle_down(bv, d);
        const uint oi = simd_shuffle_down(bi, d);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      if (lane == 0u) { sm_v[sg] = bv; sm_i[sg] = bi; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (e == 0u) {
        float fv = sm_v[0]; uint fi = sm_i[0];
        for (uint g = 1; g < 4u; g++) {
          if (sm_v[g] > fv || (sm_v[g] == fv && sm_i[g] < fi)) { fv = sm_v[g]; fi = sm_i[g]; }
        }
        picked_i[k] = fi;
        e_mask[fi] |= (1u << m);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (e == picked_i[k]) { picked_s[k] = s; k_val = -1e30f; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (e < a.topk) {
      float wsum = 0.0f;
      for (uint k = 0; k < a.topk; k++) wsum += picked_s[k];
      const float inv = a.route_scale / (wsum + 1e-20f);
      top_idx[m * a.topk + e] = picked_i[e];
      top_w[m * a.topk + e] = picked_s[e] * inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (e == 0u) {
    uint ns = 0;
    for (uint i = 0; i < a.E; i++) {
      if (e_mask[i]) {
        slots[ns] = i;
        rmask[ns * 4u + 0u] = e_mask[i];
        rmask[ns * 4u + 1u] = 0u;
        rmask[ns * 4u + 2u] = 0u;
        rmask[ns * 4u + 3u] = 0u;
        ns++;
      }
    }
    nslots[0] = ns;
  }
}

// 9b'. Decode/verify router for M = 1..16 in one phase (replaces ink_moe_route, and ink_moe_topk_pf +
//      ink_moe_pack_dec). Each expert finds its rank under the same order the iterative max uses (value
//      descending, index ascending on ties) in one pass, so the picks, their order and the weight sum (still
//      added in pick order) are identical. Slots are built by a prefix count, which gives the ascending expert
//      order of the serial scan. Grid: (1, 1, 1), TG: E = 128 threads.
kernel void ink_moe_route_fast(device const float* scores [[buffer(0)]],
                               device const float* rb [[buffer(1)]],
                               device uint* top_idx [[buffer(2)]],
                               device float* top_w [[buffer(3)]],
                               device uint* slots [[buffer(4)]],
                               device uint* nslots [[buffer(5)]],
                               device uint* rmask [[buffer(6)]],
                               constant RouteArgs& a [[buffer(7)]],
                               uint e [[thread_index_in_threadgroup]]) {
  const uint sg = e >> 5, lane = e & 31u;
  threadgroup float kv[128];
  threadgroup uint picked_i[8];
  threadgroup float picked_s[8];
  threadgroup uint sg_cnt[4];
  uint mask = 0u;
  const float bias = rb[e];
  for (uint m = 0; m < a.M; m++) {
    const float s = scores[size_t(m) * a.E + e];
    const float k_val = s + bias;
    kv[e] = k_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r = 0u;
    for (uint j = 0; j < 128u; j++) {
      const float o = kv[j];
      r += (o > k_val || (o == k_val && j < e)) ? 1u : 0u;
    }
    if (r < a.topk) { picked_i[r] = e; picked_s[r] = s; mask |= (1u << m); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e < a.topk) {
      float wsum = 0.0f;
      for (uint k = 0; k < a.topk; k++) wsum += picked_s[k];
      const float inv = a.route_scale / (wsum + 1e-20f);
      top_idx[m * a.topk + e] = picked_i[e];
      top_w[m * a.topk + e] = picked_s[e] * inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint flag = mask ? 1u : 0u;
  const uint pre = simd_prefix_exclusive_sum(flag);
  const uint tot = simd_sum(flag);
  if (lane == 0u) sg_cnt[sg] = tot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint base = 0u;
  for (uint g = 0; g < sg; g++) base += sg_cnt[g];
  if (flag) {
    const uint p = base + pre;
    slots[p] = e;
    rmask[p * 4u + 0u] = mask;
    rmask[p * 4u + 1u] = 0u;
    rmask[p * 4u + 2u] = 0u;
    rmask[p * 4u + 3u] = 0u;
  }
  if (e == 0u) nslots[0] = sg_cnt[0] + sg_cnt[1] + sg_cnt[2] + sg_cnt[3];
}

// 9c. Pack active expert slots + rmask for multi-row verify (M = 2..16) after parallel ink_moe_topk_pf.
// Grid: (1, 1, 1), TG: 128 threads.
kernel void ink_moe_pack_dec(device const uint* top_idx [[buffer(0)]],
                             device uint* slots [[buffer(1)]],
                             device uint* nslots [[buffer(2)]],
                             device uint* rmask [[buffer(3)]],
                             constant RouteArgs& a [[buffer(4)]],
                             uint tid [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint e_mask[128];
  atomic_store_explicit(&e_mask[tid], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint tot = a.M * a.topk;
  for (uint i = tid; i < tot; i += 128u) {
    const uint m = i / a.topk;
    atomic_fetch_or_explicit(&e_mask[top_idx[i]], 1u << m, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0u) {
    uint ns = 0;
    for (uint i = 0; i < a.E; i++) {
      const uint mask = atomic_load_explicit(&e_mask[i], memory_order_relaxed);
      if (mask) {
        slots[ns] = i;
        rmask[ns * 4u + 0u] = mask;
        rmask[ns * 4u + 1u] = 0u;
        rmask[ns * 4u + 2u] = 0u;
        rmask[ns * 4u + 3u] = 0u;
        ns++;
      }
    }
    nslots[0] = ns;
  }
}

// Fast float4 device copy (keeps looped pass transitions inside one compute encoder).
kernel void ink_copy_f32(device const float4* src [[buffer(0)]], device float4* dst [[buffer(1)]],
                         constant uint& n4 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n4) dst[gid] = src[gid];
}

// 10. Accumulate routed expert outputs [nslots, M, H] -> moe_out [M, H] in FP32 (float4 + TG slot cache).
// Grid: (M, ceil(H/4 / 64), 1), TG: 64 threads.
kernel void ink_moe_accum(device const float4* exp_y [[buffer(0)]],
                          device const uint* slots [[buffer(1)]],
                          device const uint* nslots [[buffer(2)]],
                          device const uint* top_idx [[buffer(3)]],
                          device const float* top_w [[buffer(4)]],
                          device float4* out [[buffer(5)]],
                          constant uint& H [[buffer(6)]], constant uint& ystride [[buffer(7)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x;
  const uint H4 = H >> 2;
  const uint ys4 = ystride >> 2;
  const uint c4 = tg.y * 64u + tid;
  threadgroup uint s_idx[8];
  threadgroup float s_w[8];
  if (tid < 8u) {
    const uint e = top_idx[row * 8u + tid];
    const uint ns = nslots[0];
    uint found = 0u;
    float w = 0.0f;  // an expert missing from slots contributes 0, as before
    for (uint s = 0u; s < ns; s++) {
      if (slots[s] == e) { found = s; w = top_w[row * 8u + tid]; break; }
    }
    s_idx[tid] = found;
    s_w[tid] = w;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (c4 >= H4) return;
  float4 acc = float4(0.0f);
  for (uint k = 0u; k < 8u; k++) {
    acc = fma(exp_y[size_t(s_idx[k]) * ys4 + size_t(row) * H4 + c4], s_w[k], acc);
  }
  out[size_t(row) * H4 + c4] = acc;
}

// 10b. Decode/verify fusion of ink_moe_accum + ink_post_sconv_add(is_mlp=1) for one lane's rows: one phase
//      instead of two. Each thread owns 4 channels and recomputes the expert sum of its row and the (up to) 3
//      earlier rows the short conv needs, with ink_moe_accum's exact fma order, so the output is bit-identical.
// Grid: (m, ceil(H/4 / 64), 1), TG: 64 threads.
struct AccumConvArgs {
  uint m, H, ks, r0, ystride;
};
kernel void ink_moe_accum_post_sconv(device const float4* exp_y [[buffer(0)]],
                                     device const uint* slots [[buffer(1)]],
                                     device const uint* nslots [[buffer(2)]],
                                     device const uint* top_idx [[buffer(3)]],
                                     device const float* top_w [[buffer(4)]],
                                     device const float* sconv_w [[buffer(5)]],
                                     device const float* conv_state_in [[buffer(6)]],
                                     device float* h_res [[buffer(7)]],
                                     device float* conv_hist [[buffer(8)]],
                                     device float* conv_state_out [[buffer(9)]],
                                     constant AccumConvArgs& a [[buffer(10)]],
                                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint m = tg.x;                 // lane-local row
  const uint H4 = a.H >> 2, ys4 = a.ystride >> 2;
  const uint c4 = tg.y * 64u + tid;
  const uint nr = min(m, 3u) + 1u;     // rows m, m-1, .. needed
  threadgroup uint s_idx[4][8];
  threadgroup float s_w[4][8];
  if (tid < 8u * nr) {
    const uint j = tid >> 3, k = tid & 7u;
    const uint row = a.r0 + m - j;
    const uint e = top_idx[row * 8u + k];
    const uint ns = nslots[0];
    uint found = 0u;
    float w = 0.0f;
    for (uint s = 0u; s < ns; s++) {
      if (slots[s] == e) { found = s; w = top_w[row * 8u + k]; break; }
    }
    s_idx[j][k] = found;
    s_w[j][k] = w;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (c4 >= H4) return;
  float4 y[4];
  for (uint j = 0u; j < nr; j++) {
    const uint row = a.r0 + m - j;
    float4 acc = float4(0.0f);
    for (uint k = 0u; k < 8u; k++) acc = fma(exp_y[size_t(s_idx[j][k]) * ys4 + size_t(row) * H4 + c4], s_w[j][k], acc);
    y[j] = acc;
  }
  const bool save_hist = (a.m <= 16u);
  for (uint q = 0u; q < 4u; q++) {
    const uint ch = c4 * 4u + q;
    const float x  = y[0][q];
    const float h2 = (m >= 1u) ? y[1][q] : conv_state_in[2u * a.H + ch];
    const float h1 = (m >= 2u) ? y[2][q] : conv_state_in[(1u + m) * a.H + ch];
    const float h0 = (m >= 3u) ? y[3][q] : conv_state_in[(0u + m) * a.H + ch];
    if (save_hist) {
      if (m == 0u) {
        conv_hist[0u * a.H + ch] = h0;
        conv_hist[1u * a.H + ch] = h1;
        conv_hist[2u * a.H + ch] = h2;
      }
      conv_hist[(3u + m) * a.H + ch] = x;
    }
    if (m + 1u == a.m) {
      conv_state_out[0u * a.H + ch] = h1;
      conv_state_out[1u * a.H + ch] = h2;
      conv_state_out[2u * a.H + ch] = x;
    }
    const float cv = post_sconv_value(sconv_w, ch, a.ks, x, h2, h1, h0);
    const size_t hi = size_t(a.r0 + m) * a.H + ch;
    h_res[hi] = float(bfloat(h_res[hi] + cv));
  }
}

// 10c. Decode/verify NVFP4 expert GEMV over mm2's tiled bank, in place (technique after mlx-serve's inkling2
//      ups/downs, MIT, David Dalcu: dot the packed E2M1 values against x in f32 and scale once per 16-value
//      group, instead of unpacking every weight to bf16 for a 16-row tensor op that carries one row).
//      Bank layout (mm2, WBITS 40): per fragment nf (32 output columns) and k-step ks (32 k), lane l holds one
//      uint4 = 32 nibbles: word j, byte i, nibble h -> column m0 + 16 (j & 1) + 8 h, k n0 + i + 16 (j >> 1), with
//      m0 = ((l >> 1) & 3) + 4 (l >> 4), n0 = 4 (l & 1) + 8 ((l >> 3) & 1). Scales half [nf][K / 16][32].
//      Expert-major: one simdgroup = one fragment of one active expert; each k-step's nibbles are decoded once and
//      used for every routed row (4 rows a pass), so a row's arithmetic does not depend on M or on its neighbours.
//      Grid (ceil(N / 32 / 4), 1, slots), TG 128 threads (4 simdgroups = 4 fragments).
#ifndef MAXM_ROWS
#define MAXM_ROWS 16
#endif
struct GemvArgs {
  uint N, K, M, wstride, sstride, xstride, ystride, I;
};
static inline float2 e2m1x2_half(uint t) {  // nibbles at bits 0..3 and 16..19 -> their E2M1 values x 2^-14
  return float2(as_type<half2>(((t & 0x70007u) << 9) | ((t & 0x80008u) << 12)));
}
#ifndef GEMV_ROWS
#define GEMV_ROWS 2
#endif
// One lane's k-step for GEMV_ROWS rows (slots past nr idle): partial dots of its 4 columns (q = 0..3 -> column
// m0 + 8 q). Nibble pairs 16 bits apart share a column: in word j, bits (0, 16) = column q0 at k 0 and 2, (4, 20) =
// column q0 + 1 at k 0 and 2, (8, 24) = column q0 at k 1 and 3, (12, 28) = column q0 + 1 at k 1 and 3 (q0 = 2 (j & 1),
// k half j >> 1). Explicit fma chains: a row's bits do not depend on which batch slot it occupies.
static inline void fp4_kstep(uint4 w, thread const float4 (&xlo)[GEMV_ROWS], thread const float4 (&xhi)[GEMV_ROWS],
                             uint nr, thread const float (&sc)[2][4], thread float (&acc)[GEMV_ROWS][4]) {
  const uint wd[4] = {w.x, w.y, w.z, w.w};
#pragma clang loop unroll(full)
  for (uint j = 0; j < 4; j++) {
    const uint kh = j >> 1, q0 = 2u * (j & 1u);
    const float2 a0 = e2m1x2_half(wd[j]), a1 = e2m1x2_half(wd[j] >> 4);
    const float2 b0 = e2m1x2_half(wd[j] >> 8), b1 = e2m1x2_half(wd[j] >> 12);
#pragma clang loop unroll(full)
    for (uint r = 0; r < GEMV_ROWS; r++) {
      if (r < nr) {
        const float4 x = kh ? xhi[r] : xlo[r];
        const float d0 = fma(x.w, b0.y, fma(x.z, a0.y, fma(x.y, b0.x, x.x * a0.x)));
        const float d1 = fma(x.w, b1.y, fma(x.z, a1.y, fma(x.y, b1.x, x.x * a1.x)));
        acc[r][q0] = fma(d0, sc[kh][q0], acc[r][q0]);
        acc[r][q0 + 1] = fma(d1, sc[kh][q0 + 1], acc[r][q0 + 1]);
      }
    }
  }
}

// A threadgroup = one fragment of one active expert; its 4 simdgroups split the k-steps (sg owns KS / 4 of them),
// and their partials are summed in simdgroup order through threadgroup memory. Routed rows go GEMV_ROWS at a time
// (each k-step's nibbles decoded once for them): a row's arithmetic is fixed, the same at every M.
#define GEMV_PROLOGUE                                                                                             \
  if (tg.z >= nslots[0]) return;                                                                                  \
  const uint nf = tg.x;                                                                                           \
  const uint e = slots[tg.z];                                                                                     \
  const uint KS = a.K / 32u, G = a.K / 16u, KQ = KS / 4u, ks0 = sg * KQ;                                         \
  device const uint4* Wp = W + size_t(e) * a.wstride + size_t(nf) * KS * 32u + l;                                 \
  device const half* Sp = S + size_t(e) * a.sstride + size_t(nf) * G * 32u + ((l >> 1) & 3u) + 4u * (l >> 4);    \
  const uint m0 = ((l >> 1) & 3u) + 4u * (l >> 4), n0 = 4u * (l & 1u) + 8u * ((l >> 3) & 1u);                    \
  threadgroup float red[4][GEMV_ROWS][32];

#define GEMV_ROW_LOOP(XROW)                                                                                       \
    float acc[GEMV_ROWS][4] = {};                                                                                 \
    for (uint ks = ks0; ks < ks0 + KQ; ks++) {                                                                    \
      float sc[2][4];                                                                                             \
      for (uint kh = 0; kh < 2u; kh++)                                                                            \
        for (uint q = 0; q < 4u; q++) sc[kh][q] = float(Sp[(2u * ks + kh) * 32u + 8u * q]);                       \
      float4 xlo[GEMV_ROWS], xhi[GEMV_ROWS];                                                                      \
      _Pragma("clang loop unroll(full)")                                                                          \
      for (uint r = 0; r < GEMV_ROWS; r++) {                                                                      \
        if (r < nr) { const uint row = rows[r]; XROW }                                                            \
      }                                                                                                           \
      fp4_kstep(Wp[size_t(ks) * 32u], xlo, xhi, nr, sc, acc);                                                     \
    }                                                                                                             \
    _Pragma("clang loop unroll(full)")                                                                            \
    for (uint r = 0; r < GEMV_ROWS; r++) {                                                                        \
      for (uint q = 0; q < 4u; q++) {                                                                             \
        float v = acc[r][q];                                                                                      \
        v += simd_shuffle_xor(v, 1);                                                                              \
        v += simd_shuffle_xor(v, 8);                                                                              \
        if ((l & 9u) == 0u) red[sg][r][m0 + 8u * q] = v;                                                          \
      }                                                                                                           \
    }                                                                                                             \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                              \
    _Pragma("clang loop unroll(full)")                                                                            \
    for (uint r = 0; r < GEMV_ROWS; r++)                                                                          \
      for (uint q = 0; q < 4u; q++) {                                                                             \
        const uint c = m0 + 8u * q;                                                                               \
        acc[r][q] = (((red[0][r][c] + red[1][r][c]) + red[2][r][c]) + red[3][r][c]) * 16384.0f;                   \
      }                                                                                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);

#define GEMV_ROWS_BEGIN                                                                                           \
  for (uint todo = rmask[tg.z * 4u]; todo;) {                                                                     \
    uint rows[GEMV_ROWS], nr = 0;                                                                                 \
    _Pragma("clang loop unroll(full)")                                                                            \
    for (uint r = 0; r < GEMV_ROWS; r++) {                                                                        \
      rows[r] = 0u;                                                                                               \
      if (todo) { rows[r] = ctz(todo); todo &= todo - 1u; nr++; }                                                 \
    }

// Gate|up (interleaved by 16 inside each fragment) + the soft-capped SwiGLU -> act [slot][MAXM][I] bf16.
kernel void ink_moe_gu_gemv(device const uint4* W [[buffer(0)]], device const half* S [[buffer(1)]],
                            device const float* X [[buffer(2)]],           // [M, K] f32 (bf16-rounded values)
                            device bfloat* act [[buffer(3)]],
                            device const uint* slots [[buffer(4)]], device const uint* nslots [[buffer(5)]],
                            device const uint* rmask [[buffer(6)]],
                            constant GemvArgs& a [[buffer(7)]],
                            uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort l [[thread_index_in_simdgroup]]) {
  GEMV_PROLOGUE
  GEMV_ROWS_BEGIN
    GEMV_ROW_LOOP({
      device const float* xr = X + size_t(row) * a.K + ks * 32u + n0;
      xlo[r] = *(device const float4*)xr;
      xhi[r] = *(device const float4*)(xr + 16u);
    })
    if (sg == 0u && (l & 9u) == 0u) {
      for (uint r = 0; r < nr; r++)
        for (uint p = 0; p < 2u; p++) {  // gate column m0 + 8 p, its up column m0 + 8 p + 16
          const float g = acc[r][p], u = acc[r][p + 2u];
          act[(size_t(tg.z) * MAXM_ROWS + rows[r]) * a.I + nf * 16u + m0 + 8u * p] =
              bfloat((tanh(g / 4.0f) * 4.0f) * (1.0f / (1.0f + exp(-g))) * (tanh(u / 25.0f) * 25.0f));
        }
    }
  }
}

// Down projection: act [slot][MAXM][I] -> y [slot][MAXM][N] f32 (ink_moe_accum's input, ystride per slot).
kernel void ink_moe_dn_gemv(device const uint4* W [[buffer(0)]], device const half* S [[buffer(1)]],
                            device const bfloat* act [[buffer(2)]],
                            device float* Y [[buffer(3)]],
                            device const uint* slots [[buffer(4)]], device const uint* nslots [[buffer(5)]],
                            device const uint* rmask [[buffer(6)]],
                            constant GemvArgs& a [[buffer(7)]],
                            uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort l [[thread_index_in_simdgroup]]) {
  GEMV_PROLOGUE
  GEMV_ROWS_BEGIN
    GEMV_ROW_LOOP({
      device const bfloat* xr = act + (size_t(tg.z) * MAXM_ROWS + row) * a.K + ks * 32u + n0;
      xlo[r] = float4(*(device const bfloat4*)xr);
      xhi[r] = float4(*(device const bfloat4*)(xr + 16u));
    })
    if (sg == 0u && (l & 9u) == 0u)
      for (uint r = 0; r < nr; r++)
        for (uint q = 0; q < 4u; q++)
          Y[size_t(tg.z) * a.ystride + size_t(rows[r]) * a.N + nf * 32u + m0 + 8u * q] = acc[r][q];
  }
}

// 11. Parallel 2-stage final logit softcap (30.0 * tanh(x / 30.0)) + greedy argmax.
// Stage 1: Grid (M, 64, 1), TG 256 threads -> 16,384 threads/row!
kernel void ink_softcap_argmax_p1(device float* logits [[buffer(0)]],
                                  device float* part_v [[buffer(1)]],
                                  device uint* part_i [[buffer(2)]],
                                  constant uint& V_valid [[buffer(3)]], constant uint& V_pad [[buffer(4)]],
                                  constant float& cap [[buffer(5)]],
                                  uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float sm_v[8];
  threadgroup uint sm_i[8];
  const uint row = tg.x, blk = tg.y;
  device float* lr = logits + size_t(row) * V_pad;
  float bv = -1e30f;
  uint bi = 0u;
  const float inv_cap = 1.0f / cap;
  const uint stride = 64u * 256u;
  for (uint v = blk * 256u + tid; v < V_valid; v += stride) {
    const float val = tanh(lr[v] * inv_cap) * cap;
    lr[v] = val;
    if (val > bv || (val == bv && v < bi)) { bv = val; bi = v; }
  }
  for (ushort d = 16; d > 0; d >>= 1) {
    const float ov = simd_shuffle_down(bv, d);
    const uint oi = simd_shuffle_down(bi, d);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  if ((tid & 31u) == 0u) { sm_v[tid >> 5] = bv; sm_i[tid >> 5] = bi; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float fv = sm_v[0]; uint fi = sm_i[0];
    for (uint i = 1; i < 8u; i++) {
      if (sm_v[i] > fv || (sm_v[i] == fv && sm_i[i] < fi)) { fv = sm_v[i]; fi = sm_i[i]; }
    }
    part_v[size_t(row) * 64u + blk] = fv;
    part_i[size_t(row) * 64u + blk] = fi;
  }
}

// Stage 2: Grid (M, 1, 1), TG 64 threads -> reduces 64 partials to tok_out[row].
kernel void ink_softcap_argmax_p2(device const float* part_v [[buffer(0)]],
                                  device const uint* part_i [[buffer(1)]],
                                  device uint* tok_out [[buffer(2)]],
                                  uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float sm_v[2];
  threadgroup uint sm_i[2];
  float bv = part_v[size_t(row) * 64u + tid];
  uint bi = part_i[size_t(row) * 64u + tid];
  for (ushort d = 16; d > 0; d >>= 1) {
    const float ov = simd_shuffle_down(bv, d);
    const uint oi = simd_shuffle_down(bi, d);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  if ((tid & 31u) == 0u) { sm_v[tid >> 5] = bv; sm_i[tid >> 5] = bi; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    const bool pick1 = (sm_v[1] > sm_v[0]) || (sm_v[1] == sm_v[0] && sm_i[1] < sm_i[0]);
    tok_out[row] = pick1 ? sm_i[1] : sm_i[0];
  }
}

// ============================================================================
// DSpark MTP Drafter Kernels (3 dense DFlare layers + rnn_hidden Markov head)
// ============================================================================

// M1. Mix 4 tapped target hidden states [4, M_stride, H] with alpha[4] + RMSNorm -> bf16 frag / f32 out.
// Grid: (M, 1, 1), TG: 256 threads.
struct MtpMixArgs {
  uint H, M_stride, write_frag, write_f32, row_off;
  float eps;
};
kernel void ink_mtp_mix_norm(device const float4* taps [[buffer(0)]],   // [4][M_stride][H/4]
                             device const float* alpha [[buffer(1)]],   // [4]
                             device const float4* norm_w [[buffer(2)]], // [H/4]
                             device bfloat* frag_out [[buffer(3)]],
                             device float4* f32_out [[buffer(4)]],
                             constant MtpMixArgs& a [[buffer(5)]],
                             uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint tpg [[threads_per_threadgroup]]) {
  threadgroup float sm[8];
  const uint H4 = a.H >> 2;
  const uint src_row = row + a.row_off;
  const size_t tap_stride = size_t(a.M_stride) * H4;
  const float a0 = alpha[0], a1 = alpha[1], a2 = alpha[2], a3 = alpha[3];

  float s2 = 0.0f;
  for (uint c = tid; c < H4; c += tpg) {
    const size_t o = size_t(src_row) * H4 + c;
    const float4 v = taps[0u * tap_stride + o] * a0 + taps[1u * tap_stride + o] * a1 +
                     taps[2u * tap_stride + o] * a2 + taps[3u * tap_stride + o] * a3;
    s2 += dot(v, v);
  }
  s2 = simd_sum(s2);
  if ((tid & 31u) == 0u) sm[tid >> 5] = s2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float tot = 0.0f;
    for (uint i = 0; i < (tpg >> 5); i++) tot += sm[i];
    sm[0] = rsqrt(tot / float(a.H) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = sm[0];

  for (uint c4 = tid; c4 < H4; c4 += tpg) {
    const size_t o = size_t(src_row) * H4 + c4;
    const float4 v = (taps[0u * tap_stride + o] * a0 + taps[1u * tap_stride + o] * a1 +
                      taps[2u * tap_stride + o] * a2 + taps[3u * tap_stride + o] * a3) * inv * norm_w[c4];
    const bfloat4 bv = bfloat4(v);
    if (a.write_frag) {
      const uint c = c4 << 2;
      frag_out[frag_off(row, c + 0u, a.H)] = bv.x;
      frag_out[frag_off(row, c + 1u, a.H)] = bv.y;
      frag_out[frag_off(row, c + 2u, a.H)] = bv.z;
      frag_out[frag_off(row, c + 3u, a.H)] = bv.w;
    }
    if (a.write_f32) f32_out[size_t(row) * H4 + c4] = float4(bv);
  }
}

// M2. MTP context KV write: k_norm_ctx (RMSNorm hd=128) + half-split RoPE (theta=1e6) -> MTP KV cache.
// Grid: (M, nkv, 1), TG: 128 threads.
struct MtpKvArgs {
  uint nkv, hd, pos0, cap, fork_pos;
  float eps, rope_theta;
};
kernel void ink_mtp_kv_ctx_cache(device const float* kv_ctx [[buffer(0)]],  // [M, 2 * nkv * hd]
                                 device const float* k_norm [[buffer(1)]],  // [hd]
                                 device bfloat* k_cache [[buffer(2)]],      // [cap, nkv, hd]
                                 device bfloat* v_cache [[buffer(3)]],      // [cap, nkv, hd]
                                 constant MtpKvArgs& a [[buffer(4)]],
                                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x, h = tg.y;
  const uint kv_dim = a.nkv * a.hd;
  device const float* kr = kv_ctx + size_t(row) * (2u * kv_dim) + h * a.hd;
  device const float* vr = kv_ctx + size_t(row) * (2u * kv_dim) + kv_dim + h * a.hd;

  threadgroup float sm[4];
  threadgroup float kn[128];
  const float x = kr[tid];
  float s2 = simd_sum(x * x);
  if ((tid & 31u) == 0u) sm[tid >> 5] = s2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) sm[0] = rsqrt((sm[0] + sm[1] + sm[2] + sm[3]) / float(a.hd) + a.eps);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  kn[tid] = x * sm[0] * k_norm[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Half-split RoPE (NeoX style: pair (d, d + hd/2))
  const uint half_d = a.hd >> 1;
  const uint d = tid % half_d;
  const float freq = float(a.pos0 + row) * pow(a.rope_theta, -float(2u * d) / float(a.hd));
  const float cs = cos(freq), sn = sin(freq);
  const float k_rot = (tid < half_d) ? (kn[d] * cs - kn[d + half_d] * sn)
                                     : (kn[d + half_d] * cs + kn[d] * sn);
  const uint slot = ((a.pos0 + row) - a.fork_pos) % a.cap;
  const size_t out_off = (size_t(slot) * a.nkv + h) * a.hd + tid;
  k_cache[out_off] = bfloat(k_rot);
  v_cache[out_off] = bfloat(vr[tid]);
}

// M3. Build 8-token MTP draft input: row 0 = embed_norm(embed[anchor]), rows 1..7 = mask_embed.
// Grid: (B, 1, 1), TG: 256 threads.
kernel void ink_mtp_embed(device const bfloat4* emb [[buffer(0)]], device const float4* emb_norm [[buffer(1)]],
                          device const bfloat4* mask_emb [[buffer(2)]], device float4* out [[buffer(3)]],
                          constant uint& anchor_tok [[buffer(4)]], constant uint& H [[buffer(5)]],
                          constant float& eps [[buffer(6)]],
                          uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint tpg [[threads_per_threadgroup]]) {
  const uint H4 = H >> 2;
  device float4* dst = out + size_t(row) * H4;
  if (row > 0u) {
    for (uint c = tid; c < H4; c += tpg) dst[c] = float4(mask_emb[c]);
    return;
  }
  threadgroup float sm[8];
  device const bfloat4* er = emb + size_t(anchor_tok) * H4;
  float s2 = 0.0f;
  for (uint c = tid; c < H4; c += tpg) { const float4 v = float4(er[c]); s2 += dot(v, v); }
  s2 = simd_sum(s2);
  if ((tid & 31u) == 0u) sm[tid >> 5] = s2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float tot = 0.0f; for (uint i = 0; i < (tpg >> 5); i++) tot += sm[i];
    sm[0] = rsqrt(tot / float(H) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = sm[0];
  for (uint c = tid; c < H4; c += tpg) dst[c] = float4(bfloat4(float4(er[c]) * inv * emb_norm[c]));
}

// M4. MTP block QKV split + q_norm/k_norm + RoPE -> Q_blk [B, nh, hd], K_blk [B, nkv, hd], V_blk [B, nkv, hd].
// Grid: (B, nh + nkv, 1), TG: 128 threads.
struct MtpQkvArgs {
  uint nh, nkv, hd, pos0;
  float eps, rope_theta;
};
kernel void ink_mtp_qkv_rope(device const float* qkv [[buffer(0)]],
                             device const float* q_norm [[buffer(1)]],
                             device const float* k_norm [[buffer(2)]],
                             device float* Q_out [[buffer(3)]],
                             device bfloat* K_out [[buffer(4)]],
                             device bfloat* V_out [[buffer(5)]],
                             constant MtpQkvArgs& a [[buffer(6)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x, idx = tg.y;
  const uint q_dim = a.nh * a.hd, kv_dim = a.nkv * a.hd;
  const uint tot_dim = q_dim + 2u * kv_dim;
  const bool is_q = (idx < a.nh);
  const uint h = is_q ? idx : (idx - a.nh);
  device const float* src = qkv + size_t(row) * tot_dim + (is_q ? (h * a.hd) : (q_dim + h * a.hd));
  device const float* nw = is_q ? q_norm : k_norm;

  threadgroup float sm[4];
  threadgroup float xn[128];
  const float x = src[tid];
  float s2 = simd_sum(x * x);
  if ((tid & 31u) == 0u) sm[tid >> 5] = s2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) sm[0] = rsqrt((sm[0] + sm[1] + sm[2] + sm[3]) / float(a.hd) + a.eps);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  xn[tid] = x * sm[0] * nw[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint half_d = a.hd >> 1;
  const uint d = tid % half_d;
  const float freq = float(a.pos0 + row) * pow(a.rope_theta, -float(2u * d) / float(a.hd));
  const float cs = cos(freq), sn = sin(freq);
  const float rot = (tid < half_d) ? (xn[d] * cs - xn[d + half_d] * sn)
                                   : (xn[d + half_d] * cs + xn[d] * sn);
  if (is_q) {
    Q_out[(size_t(row) * a.nh + h) * a.hd + tid] = rot;
  } else {
    K_out[(size_t(row) * a.nkv + h) * a.hd + tid] = bfloat(rot);
    device const float* vsrc = qkv + size_t(row) * tot_dim + q_dim + kv_dim + h * a.hd;
    V_out[(size_t(row) * a.nkv + h) * a.hd + tid] = bfloat(vsrc[tid]);
  }
}

// M5. GQA G=3 Shared-KV MTP Non-Causal Block Attention over [cached ctx] U [B block K/V].
// Grid: (B, nkv, 1), TG: 128 threads (4 simdgroups, 3x fewer KV cache reads!).
struct MtpAttnArgs {
  uint B, nh, nkv, hd, ctx_len, cap, window, is_local, fork_pos, out_row0;
};
kernel void ink_mtp_attn(device const float4* Q [[buffer(0)]],
                         device const bfloat4* K_ctx [[buffer(1)]],
                         device const bfloat4* V_ctx [[buffer(2)]],
                         device const bfloat4* K_blk [[buffer(3)]],
                         device const bfloat4* V_blk [[buffer(4)]],
                         device bfloat* out_frag [[buffer(5)]],
                         constant MtpAttnArgs& a [[buffer(6)]],
                         device const bfloat4* K_pref [[buffer(7)]],
                         device const bfloat4* V_pref [[buffer(8)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x, kvh = tg.y;
  const uint sg = tid >> 5, lane = tid & 31u;
  const uint h0 = kvh * 3u;
  const uint t_start = (a.is_local && a.ctx_len > a.window) ? (a.ctx_len - a.window) : 0u;
  const float scale = rsqrt(float(a.hd));
  const size_t q_base = size_t(row) * a.nh + h0;
  const float4 q0 = Q[(q_base + 0u) * 32u + lane] * scale;
  const float4 q1 = Q[(q_base + 1u) * 32u + lane] * scale;
  const float4 q2 = Q[(q_base + 2u) * 32u + lane] * scale;

  float m0 = -1e30f, l0 = 0.0f; float4 o0 = float4(0.0f);
  float m1 = -1e30f, l1 = 0.0f; float4 o1 = float4(0.0f);
  float m2 = -1e30f, l2 = 0.0f; float4 o2 = float4(0.0f);

  // 1) Attend to cached context KV (supports zero-copy Split-KV prefix when !is_local && fork_pos > 0)
  uint t = t_start + sg;
  if (!a.is_local && a.fork_pos > 0u) {
    for (; t < a.ctx_len; t += 4u) {
      const bool in_pref = (t < a.fork_pos);
      const uint sl = in_pref ? t : ((t - a.fork_pos) % a.cap);
      const size_t kv_off = (size_t(sl) * a.nkv + kvh) * 32u + lane;
      const float4 k4 = float4(in_pref ? K_pref[kv_off] : K_ctx[kv_off]);
      const float4 v4 = float4(in_pref ? V_pref[kv_off] : V_ctx[kv_off]);
      {
        const float s = simd_sum(dot(q0, k4));
        const float mn = max(m0, s);
        const float al = (l0 > 0.0f) ? exp(m0 - mn) : 0.0f;
        const float p = exp(s - mn);
        o0 = fma(p, v4, o0 * al); l0 = l0 * al + p; m0 = mn;
      }
      {
        const float s = simd_sum(dot(q1, k4));
        const float mn = max(m1, s);
        const float al = (l1 > 0.0f) ? exp(m1 - mn) : 0.0f;
        const float p = exp(s - mn);
        o1 = fma(p, v4, o1 * al); l1 = l1 * al + p; m1 = mn;
      }
      {
        const float s = simd_sum(dot(q2, k4));
        const float mn = max(m2, s);
        const float al = (l2 > 0.0f) ? exp(m2 - mn) : 0.0f;
        const float p = exp(s - mn);
        o2 = fma(p, v4, o2 * al); l2 = l2 * al + p; m2 = mn;
      }
    }
  } else {
  uint slot = (t < a.ctx_len) ? (t % a.cap) : 0u;
  for (; t < a.ctx_len; t += 4u) {
    const size_t kv_off = (size_t(slot) * a.nkv + kvh) * 32u + lane;
    const float4 k4 = float4(K_ctx[kv_off]);
    const float4 v4 = float4(V_ctx[kv_off]);
    {
      const float s = simd_sum(dot(q0, k4));
      const float mn = max(m0, s);
      const float al = (l0 > 0.0f) ? exp(m0 - mn) : 0.0f;
      const float p = exp(s - mn);
      o0 = fma(p, v4, o0 * al); l0 = l0 * al + p; m0 = mn;
    }
    {
      const float s = simd_sum(dot(q1, k4));
      const float mn = max(m1, s);
      const float al = (l1 > 0.0f) ? exp(m1 - mn) : 0.0f;
      const float p = exp(s - mn);
      o1 = fma(p, v4, o1 * al); l1 = l1 * al + p; m1 = mn;
    }
    {
      const float s = simd_sum(dot(q2, k4));
      const float mn = max(m2, s);
      const float al = (l2 > 0.0f) ? exp(m2 - mn) : 0.0f;
      const float p = exp(s - mn);
      o2 = fma(p, v4, o2 * al); l2 = l2 * al + p; m2 = mn;
    }
    slot += 4u;
    if (slot >= a.cap) slot -= a.cap;
  }
  }
  // 2) Attend to all B block tokens (non-causal!)
  for (uint b = sg; b < a.B; b += 4u) {
    const size_t kv_off = (size_t(b) * a.nkv + kvh) * 32u + lane;
    const float4 k4 = float4(K_blk[kv_off]);
    const float4 v4 = float4(V_blk[kv_off]);
    {
      const float s = simd_sum(dot(q0, k4));
      const float mn = max(m0, s);
      const float al = (l0 > 0.0f) ? exp(m0 - mn) : 0.0f;
      const float p = exp(s - mn);
      o0 = fma(p, v4, o0 * al); l0 = l0 * al + p; m0 = mn;
    }
    {
      const float s = simd_sum(dot(q1, k4));
      const float mn = max(m1, s);
      const float al = (l1 > 0.0f) ? exp(m1 - mn) : 0.0f;
      const float p = exp(s - mn);
      o1 = fma(p, v4, o1 * al); l1 = l1 * al + p; m1 = mn;
    }
    {
      const float s = simd_sum(dot(q2, k4));
      const float mn = max(m2, s);
      const float al = (l2 > 0.0f) ? exp(m2 - mn) : 0.0f;
      const float p = exp(s - mn);
      o2 = fma(p, v4, o2 * al); l2 = l2 * al + p; m2 = mn;
    }
  }

  threadgroup float sm_m[3][4], sm_l[3][4];
  threadgroup float4 sm_o[3][4][32];
  if (lane == 0u) {
    sm_m[0][sg] = m0; sm_l[0][sg] = l0;
    sm_m[1][sg] = m1; sm_l[1][sg] = l1;
    sm_m[2][sg] = m2; sm_l[2][sg] = l2;
  }
  sm_o[0][sg][lane] = o0;
  sm_o[1][sg][lane] = o1;
  sm_o[2][sg][lane] = o2;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg < 3u) {
    const uint g = sg;
    float fm = -1e30f, fl = 0.0f;
    float4 fo = float4(0.0f);
    for (uint s = 0; s < 4u; s++) {
      const float sl = sm_l[g][s];
      if (sl <= 0.0f) continue;
      const float smv = sm_m[g][s];
      const float mn = max(fm, smv);
      const float a0 = (fl > 0.0f) ? exp(fm - mn) : 0.0f;
      const float a1 = exp(smv - mn);
      fo = fo * a0 + sm_o[g][s][lane] * a1;
      fl = fl * a0 + sl * a1;
      fm = mn;
    }
    const bfloat4 res = bfloat4(fo / fl);
    const uint col0 = (h0 + g) * a.hd + lane * 4u;
    const uint K = a.nh * a.hd;
    const uint orow = a.out_row0 + row;
    out_frag[frag_off(orow, col0 + 0u, K)] = res.x;
    out_frag[frag_off(orow, col0 + 1u, K)] = res.y;
    out_frag[frag_off(orow, col0 + 2u, K)] = res.z;
    out_frag[frag_off(orow, col0 + 3u, K)] = res.w;
  }
}

// M6. Fused residual add (bf16-rounded) + RMSNorm -> bf16 frag (+ optional f32 copy).
// Grid: (M, 1, 1), TG: 256 threads.
kernel void ink_add_rms_frag(device float4* h [[buffer(0)]], device const float4* y [[buffer(1)]],
                             device const float4* w [[buffer(2)]],
                             device bfloat* frag [[buffer(3)]], device float4* f32_out [[buffer(4)]],
                             constant uint& H [[buffer(5)]], constant float& eps [[buffer(6)]],
                             constant uint& write_f32 [[buffer(7)]],
                             uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint tpg [[threads_per_threadgroup]]) {
  threadgroup float sm[8];
  const uint H4 = H >> 2;
  device float4* hr = h + size_t(row) * H4;
  device const float4* yr = y + size_t(row) * H4;
  float s2 = 0.0f;
  for (uint c = tid; c < H4; c += tpg) {
    const float4 v = float4(bfloat4(hr[c] + yr[c]));
    hr[c] = v;
    s2 += dot(v, v);
  }
  s2 = simd_sum(s2);
  if ((tid & 31u) == 0u) sm[tid >> 5] = s2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float t = 0.0f;
    for (uint i = 0; i < (tpg >> 5); i++) t += sm[i];
    sm[0] = rsqrt(t / float(H) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = sm[0];
  for (uint c4 = tid; c4 < H4; c4 += tpg) {
    const bfloat4 bv = bfloat4(hr[c4] * inv * w[c4]);
    const uint c = c4 << 2;
    frag[frag_off(row, c + 0u, H)] = bv.x;
    frag[frag_off(row, c + 1u, H)] = bv.y;
    frag[frag_off(row, c + 2u, H)] = bv.z;
    frag[frag_off(row, c + 3u, H)] = bv.w;
    if (write_f32) f32_out[size_t(row) * H4 + c4] = float4(bv);
  }
}

// M7a-static. Precompute static Markov RNN projection ONCE for all gamma steps:
//             static_out[k, out_r] = dot(base_h[k+1], W_bh[out_r]) + dot(f_rnn, W_fr[out_r]) + rnn_b[out_r].
//             Reads W_bh & W_fr (7.08 MB) ONCE instead of 7 times (saves 42.5 MB/round!).
// Grid: ((3*R + 3) / 4, 1, 1), TG: 128 threads (4 simdgroups, 1 out_r per simdgroup).
struct RnnArgs {
  uint H, R, step;  // R=256, H=2304; for static, step = gamma (1..7)
};
kernel void ink_mtp_rnn_static(device const float* base_h [[buffer(0)]],  // [B, H]
                               device const float* f_rnn [[buffer(1)]],   // [H]
                               device const bfloat4* rnn_w [[buffer(2)]], // [3*R, (2*R+2*H)/4]
                               device const float* rnn_b [[buffer(3)]],   // [3*R]
                               device float* static_out [[buffer(4)]],    // [gamma, 3*R]
                               constant RnnArgs& a [[buffer(5)]],
                               uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint sg = tid >> 5, lane = tid & 31u;
  const uint out_r = tg * 4u + sg;
  if (out_r >= 3u * a.R) return;

  const uint R4 = a.R >> 2, H4 = a.H >> 2;
  const uint in4 = 2u * R4 + 2u * H4;
  const uint gamma = a.step;
  device const bfloat4* wr = rnn_w + size_t(out_r) * in4;
  device const float4* fr4 = (device const float4*)f_rnn;

  float f_acc = 0.0f;
  for (uint c = lane; c < H4; c += 32u) f_acc += dot(fr4[c], float4(wr[2u * R4 + H4 + c]));
  const float f_dot = simd_sum(f_acc) + rnn_b[out_r];

  float acc[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint c = lane; c < H4; c += 32u) {
    const float4 wv = float4(wr[2u * R4 + c]);
    for (uint k = 0; k < gamma; k++) {
      device const float4* bh4 = (device const float4*)(base_h + size_t(k + 1u) * a.H);
      acc[k] += dot(bh4[c], wv);
    }
  }
  for (uint k = 0; k < gamma; k++) {
    const float v = simd_sum(acc[k]);
    if (lane == 0u) static_out[size_t(k) * (3u * a.R) + out_r] = v + f_dot;
  }
}

// M7a-dyn. Dynamic Markov RNN projection (only state[256] + markov_w1[prev_tok][256] = 0.78 MB -> 10x smaller!).
// Grid: ((3*R + 3) / 4, 1, 1), TG: 128 threads.
kernel void ink_mtp_rnn_dyn(device const float* state [[buffer(0)]],
                            device const bfloat* markov_w1 [[buffer(1)]],
                            device const uint* prev_tok_buf [[buffer(2)]],
                            device const float* static_in [[buffer(3)]],  // [gamma, 3*R]
                            device const bfloat4* rnn_w [[buffer(4)]],    // [3*R, (2*R+2*H)/4]
                            device float* proj_out [[buffer(5)]],         // [3*R]
                            constant RnnArgs& a [[buffer(6)]],
                            uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint sg = tid >> 5, lane = tid & 31u;
  const uint out_r = tg * 4u + sg;
  if (out_r >= 3u * a.R) return;

  const uint R4 = a.R >> 2, H4 = a.H >> 2;
  const uint in4 = 2u * R4 + 2u * H4;
  const uint tok = prev_tok_buf[0];
  device const float4* st4 = (device const float4*)state;
  device const bfloat4* mw4 = (device const bfloat4*)(markov_w1 + size_t(tok) * a.R);
  device const bfloat4* wr = rnn_w + size_t(out_r) * in4;

  float acc = 0.0f;
  for (uint c = lane; c < R4; c += 32u) {
    acc += dot(st4[c], float4(wr[c])) + dot(float4(mw4[c]), float4(wr[R4 + c]));
  }
  acc = simd_sum(acc);
  if (lane == 0u) proj_out[out_r] = acc + static_in[size_t(a.step) * (3u * a.R) + out_r];
}

// M7b. RNN gates update: state' = sigm(g)*state + (1-sigm(g))*tanh(c), latent = tanh(o) -> bf16 frag (row_idx, K=256).
// Grid: (1, 1, 1), TG: 256 threads.
kernel void ink_mtp_rnn_gates(device const float* proj [[buffer(0)]],
                              device float* state [[buffer(1)]],
                              device bfloat* lat_frag [[buffer(2)]],
                              constant uint& R [[buffer(3)]],
                              constant uint& row_idx [[buffer(4)]],
                              uint tid [[thread_index_in_threadgroup]]) {
  if (tid >= R) return;
  const float g = float(bfloat(isigm(proj[tid])));
  const float c = float(bfloat(tanh(proj[R + tid])));
  const float o = tanh(proj[2u * R + tid]);
  const float kept = float(bfloat(g * state[tid]));
  const float blend = float(bfloat(float(bfloat(1.0f - g)) * c));
  state[tid] = kept + blend;
  lat_frag[frag_off(row_idx, tid, R)] = bfloat(o);
}

// Pack L*gamma rows (base_h[i * B + 1 + k]) -> bf16 frag[i * gamma + k, H] for single batched lm_head.
// Grid: (L * gamma, 1, 1), TG: 256 threads.
struct PackFragArgs { uint H, B, gamma; };
kernel void ink_pack_frag_rows(device const float4* src [[buffer(0)]], device bfloat* dst_frag [[buffer(1)]],
                               constant PackFragArgs& a [[buffer(2)]],
                               uint out_row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint lane_i = out_row / a.gamma;
  const uint k = out_row % a.gamma;
  const uint src_row = lane_i * a.B + 1u + k;
  const uint H4 = a.H >> 2;
  device const float4* sr = src + size_t(src_row) * H4;
  for (uint c4 = tid; c4 < H4; c4 += 256u) {
    const bfloat4 bv = bfloat4(sr[c4]);
    const uint c = c4 << 2;
    dst_frag[frag_off(out_row, c + 0u, a.H)] = bv.x;
    dst_frag[frag_off(out_row, c + 1u, a.H)] = bv.y;
    dst_frag[frag_off(out_row, c + 2u, a.H)] = bv.z;
    dst_frag[frag_off(out_row, c + 3u, a.H)] = bv.w;
  }
}

// M8. Parallel 2-stage add base_logits[step] + bias + fold_b -> softcap(30) -> argmax + online p_max.
// Stage 1: Grid (64, 1, 1), TG 256 threads -> 16,384 threads!
// ============================================================================
// Sampling (ported from Hy4): Gumbel-max keyed by (seed, position) with min_p / top_k / top_p truncation.
// A position's draw is fixed by (seed, position) alone, so speculation stays exact at any temperature
// (a draft is accepted iff it equals the token drawn here). Logits arrive softcapped (ink_softcap_argmax_p1).
// ============================================================================
#include "sampling.metal"


struct MtpPickArgs {
  uint V_valid, V_pad, step;
  float cap;
  float inv_t;      // > 0: draw with the target's Gumbel keys (seed, pos0 + step) at temperature 1 / inv_t
  uint s0, s1, pos0;
};
kernel void ink_mtp_add_argmax_p1(device const float* base_logits [[buffer(0)]],
                                  device const float* bias [[buffer(1)]],
                                  device const float* fold_b [[buffer(2)]],
                                  device float* part_v [[buffer(3)]],
                                  device uint* part_i [[buffer(4)]],
                                  device float* part_l [[buffer(5)]],
                                  constant MtpPickArgs& a [[buffer(6)]],
                                  device float* part_x [[buffer(7)]],  // sampled: [64] max scaled logit, [64] chosen's
                                  uint blk [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float sm_v[8];
  threadgroup uint sm_i[8];
  threadgroup float sm_l[8];
  threadgroup float sm_m[8], sm_b[8];
  device const float* bl = base_logits + size_t(a.step) * a.V_pad;
  float bv = -1e30f;
  uint bi = 0u;
  float lsum = 0.0f;
  const float inv_cap = 1.0f / a.cap;
  const uint stride = 64u * 256u;
  if (a.inv_t > 0.0f) {
    // best Gumbel score (bv, bi) and its scaled logit (bl_), plus an online softmax (mx, lsum) for its probability
    float mx = -1e30f, bl_ = -1e30f;
    for (uint v = blk * 256u + tid; v < a.V_valid; v += stride) {
      const float l = tanh((bl[v] + bias[v] + fold_b[v]) * inv_cap) * a.cap * a.inv_t;
      if (l > mx) { lsum = lsum * exp(mx - l) + 1.0f; mx = l; } else { lsum += exp(l - mx); }
      const float sc = l + gumbel(a.s0, a.s1, a.pos0 + a.step, v);
      if (sc > bv) { bv = sc; bi = v; bl_ = l; }
    }
    for (ushort d = 16; d > 0; d >>= 1) {
      const float ov = simd_shuffle_down(bv, d);
      const uint oi = simd_shuffle_down(bi, d);
      const float ob = simd_shuffle_down(bl_, d);
      const float om = simd_shuffle_down(mx, d);
      const float ol = simd_shuffle_down(lsum, d);
      const float mn = max(mx, om);
      lsum = lsum * exp(mx - mn) + ol * exp(om - mn);
      mx = mn;
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; bl_ = ob; }
    }
    if ((tid & 31u) == 0u) { sm_v[tid >> 5] = bv; sm_i[tid >> 5] = bi; sm_l[tid >> 5] = lsum; sm_m[tid >> 5] = mx; sm_b[tid >> 5] = bl_; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float fv = sm_v[0], fl = sm_l[0], fm = sm_m[0], fb = sm_b[0];
      uint fi = sm_i[0];
      for (uint i = 1; i < 8u; i++) {
        const float mn = max(fm, sm_m[i]);
        fl = fl * exp(fm - mn) + sm_l[i] * exp(sm_m[i] - mn);
        fm = mn;
        if (sm_v[i] > fv || (sm_v[i] == fv && sm_i[i] < fi)) { fv = sm_v[i]; fi = sm_i[i]; fb = sm_b[i]; }
      }
      part_v[blk] = fv;
      part_i[blk] = fi;
      part_l[blk] = fl;
      part_x[blk] = fm;
      part_x[64u + blk] = fb;
    }
    return;
  }
  for (uint v = blk * 256u + tid; v < a.V_valid; v += stride) {
    const float val = tanh((bl[v] + bias[v] + fold_b[v]) * inv_cap) * a.cap;
    if (val > bv) {
      lsum = lsum * exp(bv - val) + 1.0f;
      bv = val;
      bi = v;
    } else {
      lsum += exp(val - bv);
    }
  }
  for (ushort d = 16; d > 0; d >>= 1) {
    const float ov = simd_shuffle_down(bv, d);
    const uint oi = simd_shuffle_down(bi, d);
    const float ol = simd_shuffle_down(lsum, d);
    const float mn = max(bv, ov);
    lsum = lsum * exp(bv - mn) + ol * exp(ov - mn);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  if ((tid & 31u) == 0u) { sm_v[tid >> 5] = bv; sm_i[tid >> 5] = bi; sm_l[tid >> 5] = lsum; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float fv = sm_v[0], fl = sm_l[0];
    uint fi = sm_i[0];
    for (uint i = 1; i < 8u; i++) {
      const float mn = max(fv, sm_v[i]);
      fl = fl * exp(fv - mn) + sm_l[i] * exp(sm_v[i] - mn);
      if (sm_v[i] > fv || (sm_v[i] == fv && sm_i[i] < fi)) { fv = sm_v[i]; fi = sm_i[i]; }
    }
    part_v[blk] = fv;
    part_i[blk] = fi;
    part_l[blk] = fl;
  }
}

// Stage 2: Grid (1, 1, 1), TG 64 threads -> reduces 64 online-softmax partials.
kernel void ink_mtp_add_argmax_p2(device const float* part_v [[buffer(0)]],
                                  device const uint* part_i [[buffer(1)]],
                                  device const float* part_l [[buffer(2)]],
                                  device uint* prev_tok_buf [[buffer(3)]],
                                  device uint* draft_out [[buffer(4)]],
                                  device float* draft_probs [[buffer(5)]],
                                  constant MtpPickArgs& a [[buffer(6)]],
                                  device const float* part_x [[buffer(7)]],
                                  uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float sm_v[2];
  threadgroup uint sm_i[2];
  threadgroup float sm_l[2];
  threadgroup float sm_m[2], sm_b[2];
  if (a.inv_t > 0.0f) {
    float bv = part_v[tid], lsum = part_l[tid], mx = part_x[tid], bl_ = part_x[64u + tid];
    uint bi = part_i[tid];
    for (ushort d = 16; d > 0; d >>= 1) {
      const float ov = simd_shuffle_down(bv, d);
      const uint oi = simd_shuffle_down(bi, d);
      const float ob = simd_shuffle_down(bl_, d);
      const float om = simd_shuffle_down(mx, d);
      const float ol = simd_shuffle_down(lsum, d);
      const float mn = max(mx, om);
      lsum = lsum * exp(mx - mn) + ol * exp(om - mn);
      mx = mn;
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; bl_ = ob; }
    }
    if ((tid & 31u) == 0u) { sm_v[tid >> 5] = bv; sm_i[tid >> 5] = bi; sm_l[tid >> 5] = lsum; sm_m[tid >> 5] = mx; sm_b[tid >> 5] = bl_; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      const float mn = max(sm_m[0], sm_m[1]);
      const float fl = sm_l[0] * exp(sm_m[0] - mn) + sm_l[1] * exp(sm_m[1] - mn);
      const bool pick1 = (sm_v[1] > sm_v[0]) || (sm_v[1] == sm_v[0] && sm_i[1] < sm_i[0]);
      const uint fi = pick1 ? sm_i[1] : sm_i[0];
      const float fb = pick1 ? sm_b[1] : sm_b[0];
      prev_tok_buf[0] = fi;
      draft_out[a.step] = fi;
      draft_probs[a.step] = exp(fb - mn) / max(fl, 1e-30f);
    }
    return;
  }
  float bv = part_v[tid];
  uint bi = part_i[tid];
  float lsum = part_l[tid];
  for (ushort d = 16; d > 0; d >>= 1) {
    const float ov = simd_shuffle_down(bv, d);
    const uint oi = simd_shuffle_down(bi, d);
    const float ol = simd_shuffle_down(lsum, d);
    const float mn = max(bv, ov);
    lsum = lsum * exp(bv - mn) + ol * exp(ov - mn);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  if ((tid & 31u) == 0u) { sm_v[tid >> 5] = bv; sm_i[tid >> 5] = bi; sm_l[tid >> 5] = lsum; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    const float mn = max(sm_v[0], sm_v[1]);
    const float fl = sm_l[0] * exp(sm_v[0] - mn) + sm_l[1] * exp(sm_v[1] - mn);
    const bool pick1 = (sm_v[1] > sm_v[0]) || (sm_v[1] == sm_v[0] && sm_i[1] < sm_i[0]);
    const uint fi = pick1 ? sm_i[1] : sm_i[0];
    prev_tok_buf[0] = fi;
    draft_out[a.step] = fi;
    draft_probs[a.step] = 1.0f / max(fl, 1.0f);
  }
}

// ============================================================================
// Prefill MoE Routing & Indirect-Dispatch Planning (M <= 256, mmpg CH=4 NF=2)
// ============================================================================

// P1. Parallel per-row top-8 selection for prefill: grid (M, 1, 1), TG 128 threads.
kernel void ink_moe_topk_pf(device const float* scores [[buffer(0)]],
                            device const float* rb [[buffer(1)]],
                            device uint* top_idx [[buffer(2)]],
                            device float* top_w [[buffer(3)]],
                            constant RouteArgs& a [[buffer(4)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint e [[thread_index_in_threadgroup]]) {
  const uint sg = e >> 5, lane = e & 31u;
  threadgroup float sm_v[4];
  threadgroup uint sm_i[4];
  threadgroup uint picked_i[8];
  threadgroup float picked_s[8];

  const float s = scores[size_t(row) * a.E + e];
  float k_val = s + rb[e];
  for (uint k = 0; k < a.topk; k++) {
    float bv = k_val;
    uint bi = e;
    for (ushort d = 16; d > 0; d >>= 1) {
      const float ov = simd_shuffle_down(bv, d);
      const uint oi = simd_shuffle_down(bi, d);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0u) { sm_v[sg] = bv; sm_i[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0u) {
      float fv = sm_v[0]; uint fi = sm_i[0];
      for (uint g = 1; g < 4u; g++) {
        if (sm_v[g] > fv || (sm_v[g] == fv && sm_i[g] < fi)) { fv = sm_v[g]; fi = sm_i[g]; }
      }
      picked_i[k] = fi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == picked_i[k]) { picked_s[k] = s; k_val = -1e30f; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (e < a.topk) {
    float wsum = 0.0f;
    for (uint k = 0; k < a.topk; k++) wsum += picked_s[k];
    const float inv = a.route_scale / (wsum + 1e-20f);
    top_idx[size_t(row) * a.topk + e] = picked_i[e];
    top_w[size_t(row) * a.topk + e] = picked_s[e] * inv;
  }
}

// P2. Parallel 128-thread compact grouped MoE worklist planner for mmpg (CH=4 -> 64 rows/tile, NF=2 -> 64 cols/TG).
// Grid: (1, 1, 1), TG: 128 threads.
struct PlanArgs {
  uint M, E, topk, gx_gu, gx_dn;
};
kernel void ink_moe_plan_pf(device const uint* top_idx [[buffer(0)]],  // [M, 8]
                            device uint* slots [[buffer(1)]],          // [E]
                            device uint* nslots [[buffer(2)]],         // [1]
                            device uint* roff [[buffer(3)]],           // [E]
                            device uint* rcnt [[buffer(4)]],           // [E]
                            device uint* rlist [[buffer(5)]],          // [M * 8]
                            device uint* rpos [[buffer(6)]],           // [M * 8] -> compact row index
                            device uint* wtile [[buffer(7)]],          // [(M*8)/64 + E]
                            device uint* ind_gu [[buffer(8)]],         // MTLDispatchThreadgroupsIndirectArguments (3 uint)
                            device uint* ind_dn [[buffer(9)]],
                            constant PlanArgs& a [[buffer(10)]],
                            uint tid [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint tg_cnt[128];
  threadgroup uint tg_e2s[128];
  threadgroup atomic_uint tg_cur[128];

  atomic_store_explicit(&tg_cnt[tid], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint tot = a.M * a.topk;
  for (uint i = tid; i < tot; i += 128u) {
    atomic_fetch_add_explicit(&tg_cnt[top_idx[i]], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0u) {
    uint ns = 0, off = 0, nt = 0;
    for (uint e = 0; e < a.E; e++) {
      const uint c = atomic_load_explicit(&tg_cnt[e], memory_order_relaxed);
      if (c > 0u) {
        tg_e2s[e] = ns;
        slots[ns] = e;
        roff[ns] = off;
        rcnt[ns] = c;
        atomic_store_explicit(&tg_cur[ns], off, memory_order_relaxed);
        const uint tiles = (c + 63u) >> 6;
        for (uint ty = 0; ty < tiles; ty++) wtile[nt++] = (ns << 16) | ty;
        off += (c + 15u) & ~15u;
        ns++;
      }
    }
    nslots[0] = ns;
    ind_gu[0] = a.gx_gu; ind_gu[1] = nt; ind_gu[2] = 1u;
    ind_dn[0] = a.gx_dn; ind_dn[1] = nt; ind_dn[2] = 1u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < tot; i += 128u) {
    const uint m = i / a.topk;
    const uint s = tg_e2s[top_idx[i]];
    const uint pos = atomic_fetch_add_explicit(&tg_cur[s], 1u, memory_order_relaxed);
    rlist[pos] = m;
    rpos[i] = pos;
  }
}

// P3. Prefill MoE gather-accumulate (float4 + TG rpos/w cache): moe_out[m, c] = sum_k exp_y[rpos[m, k], c] * top_w[m, k].
// Grid: (M, ceil(H/4 / 256), 1), TG: 256 threads.
kernel void ink_moe_accum_pf(device const float4* exp_y [[buffer(0)]],
                             device const uint* rpos [[buffer(1)]],
                             device const float* top_w [[buffer(2)]],
                             device float4* out [[buffer(3)]],
                             constant uint& H [[buffer(4)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x;
  const uint H4 = H >> 2;
  const uint c4 = tg.y * 256u + tid;
  threadgroup uint rp[8];
  threadgroup float tw[8];
  if (tid < 8u) {
    rp[tid] = rpos[size_t(row) * 8u + tid];
    tw[tid] = top_w[size_t(row) * 8u + tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (c4 >= H4) return;
  float4 acc = float4(0.0f);
  for (uint k = 0u; k < 8u; k++) {
    acc = fma(exp_y[size_t(rp[k]) * H4 + c4], tw[k], acc);
  }
  out[size_t(row) * H4 + c4] = acc;
}

// Per-row sampling over softcapped logits (row stride ld) + Self-Certainty confidence (KL from uniform).
// Rows with inv_t <= 0 keep the argmax already in out[] but still emit Self-Certainty into conf_out[m].
// Grid: (M, 1, 1), TG: 1024 threads.
struct SampRow { uint N, ld, pos, s0, s1; float inv_t, lmin, top_p; uint top_k; };
kernel void ink_sample_rows(device const float* y [[buffer(0)]], device uint* out [[buffer(1)]],
                            device const SampRow* sr [[buffer(2)]], device float* conf_out [[buffer(3)]],
                            uint m [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]], ushort ntg [[threads_per_threadgroup]]) {
  const SampRow a = sr[m];
  threadgroup float bv[32], bm[32], bse[32], bsl[32];
  threadgroup uint bi[32], pn[2];
  threadgroup atomic_uint hist[256];
  device const float* r = y + size_t(m) * a.ld;
  const float scale = (a.inv_t > 0.0f) ? a.inv_t : 1.0f;
  float mx = -INFINITY;
  for (uint i = tid; i < a.N; i += ntg) mx = max(mx, r[i] * scale);
  mx = simd_max(mx);
  if ((tid & 31) == 0) bm[tid / 32] = mx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float M_ = bm[0];
  for (uint s = 1; s < ntg / 32; s++) M_ = max(M_, bm[s]);
  const float M_raw = (a.inv_t > 0.0f) ? (M_ / a.inv_t) : M_;

  float se = 0.0f, sl = 0.0f;
  if (a.inv_t <= 0.0f) {
    for (uint i = tid; i < a.N; i += ntg) {
      const float raw = r[i];
      se += exp(raw - M_raw);
      sl += raw;
    }
    se = simd_sum(se);
    sl = simd_sum(sl);
    if ((tid & 31) == 0) { bse[tid / 32] = se; bsl[tid / 32] = sl; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float tse = 0.0f, tsl = 0.0f;
      for (uint s = 0; s < ntg / 32; s++) { tse += bse[s]; tsl += bsl[s]; }
      conf_out[m] = (M_raw + log(max(tse, 1e-30f))) - (tsl / float(a.N)) - log(float(a.N));
    }
    return;
  }

  const float floor_ = trunc_floor(r, a.N, a.inv_t, M_, a.lmin, a.top_p, a.top_k, bm, hist, pn, tid, ntg);
  float v = -INFINITY;
  uint idx = 0;
  for (uint i = tid; i < a.N; i += ntg) {
    const float raw = r[i];
    se += exp(raw - M_raw);
    sl += raw;
    const float l = raw * a.inv_t;
    if (l < floor_) continue;
    const float sc = l + gumbel(a.s0, a.s1, a.pos, i);
    if (sc > v) { v = sc; idx = i; }
  }
  se = simd_sum(se);
  sl = simd_sum(sl);
  for (ushort off = 16; off > 0; off >>= 1) {
    const float ov = simd_shuffle_down(v, off);
    const uint oi = simd_shuffle_down(idx, off);
    if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((tid & 31) == 0) { bv[tid / 32] = v; bi[tid / 32] = idx; bse[tid / 32] = se; bsl[tid / 32] = sl; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float b = bv[0]; uint bix = bi[0];
    float tse = 0.0f, tsl = 0.0f;
    for (uint s = 0; s < ntg / 32; s++) {
      tse += bse[s]; tsl += bsl[s];
      if (s > 0 && (bv[s] > b || (bv[s] == b && bi[s] < bix))) { b = bv[s]; bix = bi[s]; }
    }
    out[m] = bix;
    conf_out[m] = (M_raw + log(max(tse, 1e-30f))) - (tsl / float(a.N)) - log(float(a.N));
  }
}

// ============================================================================
// DFlash 2 Block-Diffusion Drafter Kernels (incoai/Inkling-2-Mini-DFlash2)
// ============================================================================

// D1. Concatenate 4 target taps [4, MP, H] f32 -> bf16 matmul fragment [n_commit, 4*H] for df2.fc.
// Grid: (n_commit, 1, 1), TG: 256 threads.
struct Df2CatArgs { uint H, MP, row0; };
kernel void ink_df2_cat_taps(device const float* taps [[buffer(0)]],
                             device bfloat* out_frag [[buffer(1)]],
                             constant Df2CatArgs& a [[buffer(2)]],
                             uint r [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint tpg [[threads_per_threadgroup]]) {
  const uint K = 4u * a.H;
  const uint src_r = a.row0 + r;
  for (uint t = 0u; t < 4u; t++) {
    device const float* src = taps + (size_t(t) * a.MP + src_r) * a.H;
    const uint k0 = t * a.H;
    for (uint c = tid; c < a.H; c += tpg) {
      out_frag[frag_off(r, k0 + c, K)] = bfloat(src[c]);
    }
  }
}

// D2. RMSNorm writing both f32 output (optional) and bf16 matmul fragment (optional).
// Grid: (M, 1, 1), TG: 256 threads.
struct Df2RmsArgs { uint H, row_off, write_f32, write_frag; float eps; };
kernel void ink_df2_rms(device const float* x [[buffer(0)]],
                        device const float* w [[buffer(1)]],
                        device float* y_f32 [[buffer(2)]],
                        device bfloat* y_frag [[buffer(3)]],
                        constant Df2RmsArgs& a [[buffer(4)]],
                        uint r [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint tpg [[threads_per_threadgroup]]) {
  device const float* xr = x + size_t(a.row_off + r) * a.H;
  threadgroup float sm[8];
  float s2 = 0.0f;
  for (uint c = tid; c < a.H; c += tpg) { const float v = xr[c]; s2 += v * v; }
  s2 = simd_sum(s2);
  if ((tid & 31u) == 0u) sm[tid >> 5] = s2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float tot = 0.0f;
    for (uint i = 0u; i < 8u; i++) tot += sm[i];
    sm[0] = rsqrt(tot / float(a.H) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = sm[0];
  device float* yr = y_f32 + size_t(r) * a.H;
  for (uint c = tid; c < a.H; c += tpg) {
    const float v = float(bfloat(xr[c] * inv * w[c]));
    if (a.write_f32) yr[c] = v;
    if (a.write_frag) y_frag[frag_off(r, c, a.H)] = bfloat(v);
  }
}

// D3. Grouped 2-tap dynamic causal convolution within each B-token draft block (supports L*B batched rows).
// stage=0 (prepare): reads x [L*B, H], dyn [L*B, 576], base [2, 2, H] -> writes bf16 frag out_frag [L*B, H].
// stage=1 (finish):  reads x [L*B, H], dyn [L*B, 576], base [2, 2, H], res [L*B, H] -> writes f32 out_f32 = res + y.
// Grid: (L*B, 1, 1), TG: 256 threads.
struct Df2ConvArgs { uint H, group_size, stage, B; };
kernel void ink_df2_conv(device const float* x [[buffer(0)]],
                         device const float* dyn [[buffer(1)]],
                         device const float* base [[buffer(2)]],
                         device const float* res [[buffer(3)]],
                         device float* out_f32 [[buffer(4)]],
                         device bfloat* out_frag [[buffer(5)]],
                         constant Df2ConvArgs& a [[buffer(6)]],
                         uint r [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint tpg [[threads_per_threadgroup]]) {
  const uint G = a.H / a.group_size;
  const uint dyn_stride = 4u * G;
  const uint r_blk = r % a.B;
  device const float* dr = dyn + size_t(r) * dyn_stride + size_t(a.stage) * (2u * G);
  device const float* b0 = base + size_t(a.stage) * (2u * a.H);
  device const float* b1 = b0 + a.H;
  device const float* x0 = x + size_t(r) * a.H;
  device const float* x1 = (r_blk > 0u) ? (x + size_t(r - 1u) * a.H) : x0;
  device const float* rr = res + size_t(r) * a.H;
  device float* yr = out_f32 + size_t(r) * a.H;
  for (uint c = tid; c < a.H; c += tpg) {
    const uint g = c / a.group_size;
    const float v0 = x0[c];
    const float v1 = (r_blk > 0u) ? x1[c] : 0.0f;
    const float y = (b0[c] + dr[g]) * v0 + (b1[c] + dr[G + g]) * v1;
    if (a.stage == 0u) out_frag[frag_off(r, c, a.H)] = bfloat(y);
    else yr[c] = rr[c] + y;
  }
}

// D4b. Split-K GQA G=4 Shared-KV Non-Causal SWA-2048 Block Attention for DFlash 2 (8 key splits -> 512 TGs).
// Grid: (B, nkv, n_splits), TG: 128 threads (4 simdgroups).
struct Df2AttnSplitArgs {
  uint B, nh, nkv, hd, ctx_len, cap, window, n_splits, out_row0;
};
kernel void ink_df2_attn_split(device const float4* Q [[buffer(0)]],
                               device const bfloat4* K_ctx [[buffer(1)]],
                               device const bfloat4* V_ctx [[buffer(2)]],
                               device const bfloat4* K_blk [[buffer(3)]],
                               device const bfloat4* V_blk [[buffer(4)]],
                               device float* part_out [[buffer(5)]],  // [B, nh, n_splits, 130]
                               constant Df2AttnSplitArgs& a [[buffer(6)]],
                               uint3 tg [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x, kvh = tg.y, sp = tg.z;
  const uint sg = tid >> 5, lane = tid & 31u;
  const uint h0 = kvh * 4u;
  const uint q_abs = a.ctx_len + row;
  const uint t_start = (q_abs + 1u > a.window) ? (q_abs + 1u - a.window) : 0u;
  const uint total_ctx = (a.ctx_len > t_start) ? (a.ctx_len - t_start) : 0u;
  const uint chunk = (total_ctx + a.n_splits - 1u) / a.n_splits;
  const uint t0 = t_start + sp * chunk;
  const uint t1 = min(a.ctx_len, t0 + chunk);

  const float scale = rsqrt(float(a.hd));
  const size_t q_base = size_t(row) * a.nh + h0;
  const float4 q0 = Q[(q_base + 0u) * 32u + lane] * scale;
  const float4 q1 = Q[(q_base + 1u) * 32u + lane] * scale;
  const float4 q2 = Q[(q_base + 2u) * 32u + lane] * scale;
  const float4 q3 = Q[(q_base + 3u) * 32u + lane] * scale;

  float m0 = -1e30f, l0 = 0.0f; float4 o0 = float4(0.0f);
  float m1 = -1e30f, l1 = 0.0f; float4 o1 = float4(0.0f);
  float m2 = -1e30f, l2 = 0.0f; float4 o2 = float4(0.0f);
  float m3 = -1e30f, l3 = 0.0f; float4 o3 = float4(0.0f);

  for (uint t = t0 + sg; t < t1; t += 4u) {
    const uint sl = t % a.cap;
    const size_t kv_off = (size_t(sl) * a.nkv + kvh) * 32u + lane;
    const float4 k4 = float4(K_ctx[kv_off]);
    const float4 v4 = float4(V_ctx[kv_off]);
    { const float s = simd_sum(dot(q0, k4)); const float mn = max(m0, s); const float al = (l0 > 0.0f) ? exp(m0 - mn) : 0.0f; const float p = exp(s - mn); o0 = fma(p, v4, o0 * al); l0 = l0 * al + p; m0 = mn; }
    { const float s = simd_sum(dot(q1, k4)); const float mn = max(m1, s); const float al = (l1 > 0.0f) ? exp(m1 - mn) : 0.0f; const float p = exp(s - mn); o1 = fma(p, v4, o1 * al); l1 = l1 * al + p; m1 = mn; }
    { const float s = simd_sum(dot(q2, k4)); const float mn = max(m2, s); const float al = (l2 > 0.0f) ? exp(m2 - mn) : 0.0f; const float p = exp(s - mn); o2 = fma(p, v4, o2 * al); l2 = l2 * al + p; m2 = mn; }
    { const float s = simd_sum(dot(q3, k4)); const float mn = max(m3, s); const float al = (l3 > 0.0f) ? exp(m3 - mn) : 0.0f; const float p = exp(s - mn); o3 = fma(p, v4, o3 * al); l3 = l3 * al + p; m3 = mn; }
  }
  if (sp == 0u) {
    for (uint j = sg; j < a.B; j += 4u) {
      const size_t kv_off = (size_t(j) * a.nkv + kvh) * 32u + lane;
      const float4 k4 = float4(K_blk[kv_off]);
      const float4 v4 = float4(V_blk[kv_off]);
      { const float s = simd_sum(dot(q0, k4)); const float mn = max(m0, s); const float al = (l0 > 0.0f) ? exp(m0 - mn) : 0.0f; const float p = exp(s - mn); o0 = fma(p, v4, o0 * al); l0 = l0 * al + p; m0 = mn; }
      { const float s = simd_sum(dot(q1, k4)); const float mn = max(m1, s); const float al = (l1 > 0.0f) ? exp(m1 - mn) : 0.0f; const float p = exp(s - mn); o1 = fma(p, v4, o1 * al); l1 = l1 * al + p; m1 = mn; }
      { const float s = simd_sum(dot(q2, k4)); const float mn = max(m2, s); const float al = (l2 > 0.0f) ? exp(m2 - mn) : 0.0f; const float p = exp(s - mn); o2 = fma(p, v4, o2 * al); l2 = l2 * al + p; m2 = mn; }
      { const float s = simd_sum(dot(q3, k4)); const float mn = max(m3, s); const float al = (l3 > 0.0f) ? exp(m3 - mn) : 0.0f; const float p = exp(s - mn); o3 = fma(p, v4, o3 * al); l3 = l3 * al + p; m3 = mn; }
    }
  }

  threadgroup float sm_m[4][4], sm_l[4][4];
  threadgroup float4 sm_o[4][4][32];
  if (lane == 0u) {
    sm_m[0][sg] = m0; sm_l[0][sg] = l0;
    sm_m[1][sg] = m1; sm_l[1][sg] = l1;
    sm_m[2][sg] = m2; sm_l[2][sg] = l2;
    sm_m[3][sg] = m3; sm_l[3][sg] = l3;
  }
  sm_o[0][sg][lane] = o0;
  sm_o[1][sg][lane] = o1;
  sm_o[2][sg][lane] = o2;
  sm_o[3][sg][lane] = o3;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint qh = h0 + sg;
  float mm = max(max(sm_m[sg][0], sm_m[sg][1]), max(sm_m[sg][2], sm_m[sg][3]));
  float ll = 0.0f;
  float4 oo = float4(0.0f);
  for (uint s = 0u; s < 4u; s++) {
    const float w = (sm_l[sg][s] > 0.0f) ? exp(sm_m[sg][s] - mm) : 0.0f;
    ll += sm_l[sg][s] * w;
    oo += sm_o[sg][s][lane] * w;
  }
  device float* dst = part_out + ((size_t(row) * a.nh + qh) * a.n_splits + sp) * 130u;
  dst[lane * 4u + 0u] = oo.x;
  dst[lane * 4u + 1u] = oo.y;
  dst[lane * 4u + 2u] = oo.z;
  dst[lane * 4u + 3u] = oo.w;
  if (lane == 0u) { dst[128u] = mm; dst[129u] = ll; }
}

// D4c. Reduce n_splits partials -> out_frag [out_row0 + B, nh * hd]. Grid: (B, nh, 1), TG: 32 threads.
kernel void ink_df2_attn_reduce(device const float* part_out [[buffer(0)]],
                                device bfloat* out_frag [[buffer(1)]],
                                constant Df2AttnSplitArgs& a [[buffer(2)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_threadgroup]]) {
  const uint row = tg.x, qh = tg.y;
  device const float* base = part_out + (size_t(row) * a.nh + qh) * a.n_splits * 130u;
  float mm = -1e30f;
  for (uint sp = 0u; sp < a.n_splits; sp++) {
    if (base[sp * 130u + 129u] > 0.0f) mm = max(mm, base[sp * 130u + 128u]);
  }
  float ll = 0.0f;
  float4 oo = float4(0.0f);
  for (uint sp = 0u; sp < a.n_splits; sp++) {
    const float ls = base[sp * 130u + 129u];
    if (ls <= 0.0f) continue;
    const float w = exp(base[sp * 130u + 128u] - mm);
    ll += ls * w;
    const float4 v4 = float4(base[sp * 130u + lane * 4u + 0u], base[sp * 130u + lane * 4u + 1u],
                             base[sp * 130u + lane * 4u + 2u], base[sp * 130u + lane * 4u + 3u]);
    oo = fma(w, v4, oo);
  }
  const float4 out4 = oo / max(ll, 1e-30f);
  const uint K_tot = a.nh * a.hd;
  const uint k0 = qh * a.hd + lane * 4u;
  const uint out_r = a.out_row0 + row;
  out_frag[frag_off(out_r, k0 + 0u, K_tot)] = bfloat(out4.x);
  out_frag[frag_off(out_r, k0 + 1u, K_tot)] = bfloat(out4.y);
  out_frag[frag_off(out_r, k0 + 2u, K_tot)] = bfloat(out4.z);
  out_frag[frag_off(out_r, k0 + 3u, K_tot)] = bfloat(out4.w);
}

// D4. GQA G=4 Shared-KV Non-Causal SWA-2048 Block Attention for DFlash 2 (nh=32, nkv=8, hd=128).
// Grid: (B, nkv, 1), TG: 128 threads (4 simdgroups).
kernel void ink_df2_attn(device const float4* Q [[buffer(0)]],
                         device const bfloat4* K_ctx [[buffer(1)]],
                         device const bfloat4* V_ctx [[buffer(2)]],
                         device const bfloat4* K_blk [[buffer(3)]],
                         device const bfloat4* V_blk [[buffer(4)]],
                         device bfloat* out_frag [[buffer(5)]],
                         constant MtpAttnArgs& a [[buffer(6)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x, kvh = tg.y;
  const uint sg = tid >> 5, lane = tid & 31u;
  const uint h0 = kvh * 4u;
  const uint q_abs = a.ctx_len + row;
  const uint t_start = (q_abs + 1u > a.window) ? (q_abs + 1u - a.window) : 0u;
  const float scale = rsqrt(float(a.hd));
  const size_t q_base = size_t(row) * a.nh + h0;
  const float4 q0 = Q[(q_base + 0u) * 32u + lane] * scale;
  const float4 q1 = Q[(q_base + 1u) * 32u + lane] * scale;
  const float4 q2 = Q[(q_base + 2u) * 32u + lane] * scale;
  const float4 q3 = Q[(q_base + 3u) * 32u + lane] * scale;

  float m0 = -1e30f, l0 = 0.0f; float4 o0 = float4(0.0f);
  float m1 = -1e30f, l1 = 0.0f; float4 o1 = float4(0.0f);
  float m2 = -1e30f, l2 = 0.0f; float4 o2 = float4(0.0f);
  float m3 = -1e30f, l3 = 0.0f; float4 o3 = float4(0.0f);

  for (uint t = t_start + sg; t < a.ctx_len; t += 4u) {
    const uint sl = t % a.cap;
    const size_t kv_off = (size_t(sl) * a.nkv + kvh) * 32u + lane;
    const float4 k4 = float4(K_ctx[kv_off]);
    const float4 v4 = float4(V_ctx[kv_off]);
    { const float s = simd_sum(dot(q0, k4)); const float mn = max(m0, s); const float al = (l0 > 0.0f) ? exp(m0 - mn) : 0.0f; const float p = exp(s - mn); o0 = fma(p, v4, o0 * al); l0 = l0 * al + p; m0 = mn; }
    { const float s = simd_sum(dot(q1, k4)); const float mn = max(m1, s); const float al = (l1 > 0.0f) ? exp(m1 - mn) : 0.0f; const float p = exp(s - mn); o1 = fma(p, v4, o1 * al); l1 = l1 * al + p; m1 = mn; }
    { const float s = simd_sum(dot(q2, k4)); const float mn = max(m2, s); const float al = (l2 > 0.0f) ? exp(m2 - mn) : 0.0f; const float p = exp(s - mn); o2 = fma(p, v4, o2 * al); l2 = l2 * al + p; m2 = mn; }
    { const float s = simd_sum(dot(q3, k4)); const float mn = max(m3, s); const float al = (l3 > 0.0f) ? exp(m3 - mn) : 0.0f; const float p = exp(s - mn); o3 = fma(p, v4, o3 * al); l3 = l3 * al + p; m3 = mn; }
  }
  for (uint j = sg; j < a.B; j += 4u) {
    const size_t kv_off = (size_t(j) * a.nkv + kvh) * 32u + lane;
    const float4 k4 = float4(K_blk[kv_off]);
    const float4 v4 = float4(V_blk[kv_off]);
    { const float s = simd_sum(dot(q0, k4)); const float mn = max(m0, s); const float al = (l0 > 0.0f) ? exp(m0 - mn) : 0.0f; const float p = exp(s - mn); o0 = fma(p, v4, o0 * al); l0 = l0 * al + p; m0 = mn; }
    { const float s = simd_sum(dot(q1, k4)); const float mn = max(m1, s); const float al = (l1 > 0.0f) ? exp(m1 - mn) : 0.0f; const float p = exp(s - mn); o1 = fma(p, v4, o1 * al); l1 = l1 * al + p; m1 = mn; }
    { const float s = simd_sum(dot(q2, k4)); const float mn = max(m2, s); const float al = (l2 > 0.0f) ? exp(m2 - mn) : 0.0f; const float p = exp(s - mn); o2 = fma(p, v4, o2 * al); l2 = l2 * al + p; m2 = mn; }
    { const float s = simd_sum(dot(q3, k4)); const float mn = max(m3, s); const float al = (l3 > 0.0f) ? exp(m3 - mn) : 0.0f; const float p = exp(s - mn); o3 = fma(p, v4, o3 * al); l3 = l3 * al + p; m3 = mn; }
  }

  threadgroup float sm_m[4][4], sm_l[4][4];
  threadgroup float4 sm_o[4][4][32];
  if (lane == 0u) {
    sm_m[0][sg] = m0; sm_l[0][sg] = l0;
    sm_m[1][sg] = m1; sm_l[1][sg] = l1;
    sm_m[2][sg] = m2; sm_l[2][sg] = l2;
    sm_m[3][sg] = m3; sm_l[3][sg] = l3;
  }
  sm_o[0][sg][lane] = o0;
  sm_o[1][sg][lane] = o1;
  sm_o[2][sg][lane] = o2;
  sm_o[3][sg][lane] = o3;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 4 simdgroups each finalize one of the 4 Q heads!
  const uint qh = h0 + sg;
  float mm = max(max(sm_m[sg][0], sm_m[sg][1]), max(sm_m[sg][2], sm_m[sg][3]));
  float ll = 0.0f;
  float4 oo = float4(0.0f);
  for (uint s = 0u; s < 4u; s++) {
    const float w = (sm_l[sg][s] > 0.0f) ? exp(sm_m[sg][s] - mm) : 0.0f;
    ll += sm_l[sg][s] * w;
    oo += sm_o[sg][s][lane] * w;
  }
  const float4 out4 = oo / max(ll, 1e-30f);
  const uint K_tot = a.nh * a.hd;
  const uint k0 = qh * a.hd + lane * 4u;
  const uint out_r = a.out_row0 + row;
  out_frag[frag_off(out_r, k0 + 0u, K_tot)] = bfloat(out4.x);
  out_frag[frag_off(out_r, k0 + 1u, K_tot)] = bfloat(out4.y);
  out_frag[frag_off(out_r, k0 + 2u, K_tot)] = bfloat(out4.z);
  out_frag[frag_off(out_r, k0 + 3u, K_tot)] = bfloat(out4.w);
}

// D5. Exact Top-16 over valid_vocab_size raw logits (Stage 1: Grid (rows, 64, 1), TG 256).
struct Df2TopkArgs { uint V_valid, V_pad; };
kernel void ink_df2_topk16_p1(device const float* logits [[buffer(0)]],
                              device float* part_v [[buffer(1)]],
                              device uint* part_i [[buffer(2)]],
                              constant Df2TopkArgs& a [[buffer(3)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x, blk = tg.y;
  device const float* lr = logits + size_t(row) * a.V_pad;
  const uint stride = 64u * 256u;
  float lv[13];
  uint li[13];
  uint n = 0u;
  for (uint v = blk * 256u + tid; v < a.V_valid && n < 13u; v += stride) {
    lv[n] = lr[v];
    li[n] = v;
    n++;
  }
  threadgroup float sm_v[8];
  threadgroup uint sm_i[8];
  const size_t out_base = (size_t(row) * 64u + blk) * 16u;
  for (uint k = 0u; k < 16u; k++) {
    float bv = -1e30f; uint bi = 0xFFFFFFFFu; uint bj = 13u;
    for (uint j = 0u; j < n; j++) {
      if (lv[j] > bv || (lv[j] == bv && li[j] < bi)) { bv = lv[j]; bi = li[j]; bj = j; }
    }
    float sv = bv; uint si = bi;
    for (ushort d = 16; d > 0; d >>= 1) {
      const float ov = simd_shuffle_down(sv, d);
      const uint oi = simd_shuffle_down(si, d);
      if (ov > sv || (ov == sv && oi < si)) { sv = ov; si = oi; }
    }
    if ((tid & 31u) == 0u) { sm_v[tid >> 5] = sv; sm_i[tid >> 5] = si; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float fv = sm_v[0]; uint fi = sm_i[0];
      for (uint s = 1u; s < 8u; s++) {
        if (sm_v[s] > fv || (sm_v[s] == fv && sm_i[s] < fi)) { fv = sm_v[s]; fi = sm_i[s]; }
      }
      sm_v[0] = fv; sm_i[0] = fi;
      part_v[out_base + k] = fv;
      part_i[out_base + k] = fi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (bj < n && bi == sm_i[0]) lv[bj] = -1e30f;
  }
}

// D5b. Exact Top-16 Stage 2: merge 64 x 16 = 1024 partials -> [rows, 16]. Grid (rows, 1, 1), TG 256.
kernel void ink_df2_topk16_p2(device const float* part_v [[buffer(0)]],
                              device const uint* part_i [[buffer(1)]],
                              device float* top_v [[buffer(2)]],
                              device uint* top_i [[buffer(3)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
  const size_t in_base = size_t(row) * 1024u;
  float lv[4];
  uint li[4];
  for (uint j = 0u; j < 4u; j++) {
    const uint idx = j * 256u + tid;
    lv[j] = part_v[in_base + idx];
    li[j] = part_i[in_base + idx];
  }
  threadgroup float sm_v[8];
  threadgroup uint sm_i[8];
  const size_t out_base = size_t(row) * 16u;
  for (uint k = 0u; k < 16u; k++) {
    float bv = -1e30f; uint bi = 0xFFFFFFFFu; uint bj = 4u;
    for (uint j = 0u; j < 4u; j++) {
      if (lv[j] > bv || (lv[j] == bv && li[j] < bi)) { bv = lv[j]; bi = li[j]; bj = j; }
    }
    float sv = bv; uint si = bi;
    for (ushort d = 16; d > 0; d >>= 1) {
      const float ov = simd_shuffle_down(sv, d);
      const uint oi = simd_shuffle_down(si, d);
      if (ov > sv || (ov == sv && oi < si)) { sv = ov; si = oi; }
    }
    if ((tid & 31u) == 0u) { sm_v[tid >> 5] = sv; sm_i[tid >> 5] = si; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float fv = sm_v[0]; uint fi = sm_i[0];
      for (uint s = 1u; s < 8u; s++) {
        if (sm_v[s] > fv || (sm_v[s] == fv && sm_i[s] < fi)) { fv = sm_v[s]; fi = sm_i[s]; }
      }
      sm_v[0] = fv; sm_i[0] = fi;
      top_v[out_base + k] = fv;
      top_i[out_base + k] = fi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (bj < 4u && bi == sm_i[0]) lv[bj] = -1e30f;
  }
}

// D6. DFlash 2 Candidate Path Selector (rank=256, top_k=16, post_softcap=30.0, greedy + coupled Gumbel-max).
// Grid: (1, 1, 1), TG: 256 threads (1 thread per rank dim).
struct Df2SelArgs {
  uint gamma, anchor_tok, pos0, seed0, seed1, top_k;
  float inv_temp, log_min_p, top_p, softcap;
};
kernel void ink_df2_select(device const float* unary [[buffer(0)]],       // [gamma, 16]
                           device const uint* cand_ids [[buffer(1)]],     // [gamma, 16]
                           device const float* h_sel [[buffer(2)]],       // [gamma, 256]
                           device const bfloat* pred_cb [[buffer(3)]],    // [V_valid, 256]
                           device const bfloat* succ_cb [[buffer(4)]],    // [V_valid, 256]
                           device uint* draft_out [[buffer(5)]],          // [gamma]
                           device float* draft_probs [[buffer(6)]],       // [gamma]
                           constant Df2SelArgs& a [[buffer(7)]],
                           uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float sm[8];
  threadgroup float scores[16];
  threadgroup uint prev_tok;
  if (tid == 0u) prev_tok = a.anchor_tok;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float inv_cap = (a.softcap > 0.0f) ? (1.0f / a.softcap) : 0.0f;
  for (uint p = 0u; p < a.gamma; p++) {
    const uint pt = prev_tok;
    const float ph = float(pred_cb[size_t(pt) * 256u + tid]) * h_sel[size_t(p) * 256u + tid];
    for (uint k = 0u; k < 16u; k++) {
      const uint cid = cand_ids[size_t(p) * 16u + k];
      float prod = ph * float(succ_cb[size_t(cid) * 256u + tid]);
      prod = simd_sum(prod);
      if ((tid & 31u) == 0u) sm[tid >> 5] = prod;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid == 0u) {
        float dot = 0.0f;
        for (uint s = 0u; s < 8u; s++) dot += sm[s];
        float sc = unary[size_t(p) * 16u + k] + dot;
        if (a.softcap > 0.0f) sc = tanh(sc * inv_cap) * a.softcap;
        scores[k] = sc;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
      float mx = -1e30f;
      uint bi = 0u;
      for (uint k = 0u; k < 16u; k++) {
        if (scores[k] > mx) { mx = scores[k]; bi = k; }
      }
      const float it = (a.inv_temp > 0.0f) ? a.inv_temp : 1.0f;
      float se = 0.0f;
      for (uint k = 0u; k < 16u; k++) se += exp((scores[k] - mx) * it);
      uint pick = bi;
      if (a.inv_temp > 0.0f) {
        const float floor_v = mx * it + a.log_min_p;
        float bg = -1e30f;
        for (uint k = 0u; k < 16u; k++) {
          const float sc = scores[k] * it;
          if (sc >= floor_v) {
            const float g = sc + gumbel(a.seed0, a.seed1, a.pos0 + 1u + p, cand_ids[size_t(p) * 16u + k]);
            if (g > bg) { bg = g; pick = k; }
          }
        }
      }
      const uint tok = cand_ids[size_t(p) * 16u + pick];
      const float prob = exp((scores[pick] - mx) * it) / max(se, 1e-30f);
      draft_out[p] = tok;
      draft_probs[p] = prob;
      prev_tok = tok;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
