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

// 2. Block Attention Residual (float4 vectorized, handles nq = 1, 3, or 4 queries).
// Grid: (M, nq, 1), TG: 256 threads.
struct AttnResArgs {
  uint H, M, n_prev, b_stride;
  float eps;
};
kernel void ink_attn_res(device const float4* blocks [[buffer(0)]], device const float4* partial [[buffer(1)]],
                         device const float4* queries [[buffer(2)]], device float4* out [[buffer(3)]],
                         constant AttnResArgs& a [[buffer(4)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint2 tpg2 [[threads_per_threadgroup]]) {
  const uint tpg = tpg2.x;
  const uint H4 = a.H >> 2, bs4 = a.b_stride >> 2;
  const uint row = tg.x, qi = tg.y;
  const uint B = a.n_prev + 1u;
  const float inv_sqrt_h = rsqrt(float(a.H));
  threadgroup float sm_s[8];
  threadgroup float sm_d[8];
  threadgroup float inv_rms[16];
  threadgroup float wts[16];
  device const float4* q = queries + size_t(qi) * H4;

  for (uint b = 0; b < B; b++) {
    device const float4* st = (b < a.n_prev) ? (blocks + size_t(b) * bs4 + size_t(row) * H4)
                                             : (partial + size_t(row) * H4);
    float s2 = 0.0f, dt = 0.0f;
    for (uint c = tid; c < H4; c += tpg) {
      const float4 v = st[c];
      s2 += dot(v, v);
      dt += dot(v, q[c]);
    }
    s2 = simd_sum(s2);
    dt = simd_sum(dt);
    if ((tid & 31u) == 0u) { sm_s[tid >> 5] = s2; sm_d[tid >> 5] = dt; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float t2 = 0.0f, td = 0.0f;
      for (uint i = 0; i < (tpg >> 5); i++) { t2 += sm_s[i]; td += sm_d[i]; }
      const float ir = rsqrt(t2 / float(a.H) + a.eps);
      inv_rms[b] = ir;
      wts[b] = td * ir * inv_sqrt_h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0u) {
    float mx = wts[0];
    for (uint b = 1; b < B; b++) mx = max(mx, wts[b]);
    float den = 0.0f;
    for (uint b = 0; b < B; b++) { wts[b] = exp(wts[b] - mx); den += wts[b]; }
    const float inv_d = 1.0f / den;
    for (uint b = 0; b < B; b++) wts[b] *= inv_d * inv_rms[b];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device float4* dst = out + (size_t(qi) * a.M + row) * H4;
  for (uint c = tid; c < H4; c += tpg) {
    float4 acc = float4(0.0f);
    for (uint b = 0; b < B; b++) {
      device const float4* st = (b < a.n_prev) ? (blocks + size_t(b) * bs4 + size_t(row) * H4)
                                               : (partial + size_t(row) * H4);
      acc = fma(st[c], wts[b], acc);
    }
    dst[c] = acc;
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

// 3c. Parallel Message Gate Stage 3: out[M, H] = (1 - tanh(w_f @ low)) * h + tanh(w_g @ low) * msg.
// Grid: (M, ceil(H / 64), 1) -> 36 TGs per row, TG: 64 threads (1 channel per thread, float4 loads on R=64).
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
  const float qv = q_in[tid] * q_scale[h * a.hd + tid];
  float n2 = simd_sum(qv * qv);
  if ((tid & 31u) == 0u) sm[tid >> 5] = n2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    const float norm = sqrt(sm[0] + sm[1] + sm[2] + sm[3]);
    sm[0] = min(a.qk_clip / (norm + 1e-6f), 1.0f) * tau;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  Q_out[(size_t(row) * a.nh + h) * a.hd + tid] = float(bfloat(qv * sm[0]));

  gate_out[(size_t(row) * a.nh + h) * a.hd + tid] = isigm(g_in[tid]);
  if (tid == 0u) sink_out[size_t(row) * a.nh + h] = s_in[h];

  threadgroup float rv[16];
  if (tid < a.d_rel) rv[tid] = r_in[tid] * rel_scale[h * a.d_rel + tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float rn2 = 0.0f;
    for (uint d = 0; d < a.d_rel; d++) rn2 += rv[d] * rv[d];
    const float mult = min(a.rel_clip / (sqrt(rn2) + 1e-6f), 1.0f) * tau;
    for (uint d = 0; d < a.d_rel; d++) rv[d] = float(bfloat(rv[d] * mult));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Vectorized bfloat4: rel_extent=1024 -> 256 float4s -> 2 iterations per thread!
  const uint ext4 = a.rel_extent >> 2;
  device float4* rl = rel_logits + (size_t(row) * a.nh + h) * ext4;
  for (uint e4 = tid; e4 < ext4; e4 += 128u) {
    float4 acc = float4(0.0f);
    for (uint d = 0; d < a.d_rel; d++) acc = fma(rv[d], float4(rel_proj[d * ext4 + e4]), acc);
    rl[e4] = acc;
  }
}

// 6. K/V short convolution (static on local, dynamic on global) + K L2-clip / V RMSNorm -> KV cache.
// Grid: (nkv, 1, 1), TG: 128 threads.
struct KvConvArgs {
  uint M, nkv, hd, k_stride, pos0, cap, is_global, is_key, ks;
  float qk_clip, eps;
};
kernel void ink_kv_conv_cache(device const float* in_kv [[buffer(0)]],
                              device const float* sconv_w [[buffer(1)]],
                              device float* conv_state [[buffer(2)]],
                              device bfloat* kv_cache [[buffer(3)]],
                              constant KvConvArgs& a [[buffer(4)]],
                              uint h [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint C = a.nkv * a.hd;
  const uint ch = h * a.hd + tid;
  float h0 = conv_state[0u * C + ch];
  float h1 = conv_state[1u * C + ch];
  float h2 = conv_state[2u * C + ch];
  const float inv_denom = a.is_global ? rsqrt(float(2u * a.ks + 1u)) : rsqrt(float(a.ks + 1u));

  float sw0, sw1, sw2, sw3;
  if (!a.is_global) {
    sw3 = sconv_w[ch * 4u + 0u];
    sw2 = sconv_w[ch * 4u + 1u];
    sw1 = sconv_w[ch * 4u + 2u];
    sw0 = sconv_w[ch * 4u + 3u];
  } else {
    sw0 = sconv_w[0u * C + ch];
    sw1 = sconv_w[1u * C + ch];
    sw2 = sconv_w[2u * C + ch];
    sw3 = sconv_w[3u * C + ch];
  }

  threadgroup float sm[4];
  for (uint m = 0; m < a.M; m++) {
    const float x = in_kv[size_t(m) * a.k_stride + ch];
    float y;
    if (!a.is_global) {
      y = (sw0 * x + sw1 * h2 + sw2 * h1 + sw3 * h0 + x) * inv_denom;
    } else {
      device const float* dw = in_kv + size_t(m) * a.k_stride + C + h * 4u;
      y = ((sw0 + dw[0]) * x + (sw1 + dw[1]) * h2 + (sw2 + dw[2]) * h1 + (sw3 + dw[3]) * h0 + x) * inv_denom;
    }
    h0 = h1; h1 = h2; h2 = x;

    float n2 = simd_sum(y * y);
    if ((tid & 31u) == 0u) sm[tid >> 5] = n2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      const float tot = sm[0] + sm[1] + sm[2] + sm[3];
      sm[0] = a.is_key ? min(a.qk_clip / (sqrt(tot) + 1e-6f), 1.0f) : rsqrt(tot / float(a.hd) + a.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint slot = (a.pos0 + m) % a.cap;
    kv_cache[(size_t(slot) * a.nkv + h) * a.hd + tid] = bfloat(y * sm[0]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  conv_state[0u * C + ch] = h0;
  conv_state[1u * C + ch] = h1;
  conv_state[2u * C + ch] = h2;
}

// 7a. Barrier-free Simdgroup Split-K GQA Attention (hd=128 = 32 lanes x float4).
//     Each simdgroup (4 per TG) processes independent keys with ZERO inner-loop barriers!
//     Grid: (M, nh, n_splits), TG: 128 threads (4 simdgroups).
//     Outputs partial [M, nh, n_splits, 130] floats = [m_max, l_sum, o_acc[128]].
struct AttnArgs {
  uint M, nh, nkv, hd, rel_extent, pos0, cap, window, is_local, n_splits;
};
kernel void ink_attn_split(device const float4* Q [[buffer(0)]],         // [M, nh, 32]
                           device const bfloat4* K_cache [[buffer(1)]],  // [cap, nkv, 32]
                           device const bfloat4* V_cache [[buffer(2)]],  // [cap, nkv, 32]
                           device const float* rel_logits [[buffer(3)]], // [M, nh, rel_extent]
                           device float* part_out [[buffer(4)]],         // [M, nh, n_splits, 130]
                           constant AttnArgs& a [[buffer(5)]],
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

  // Each simdgroup handles keys strided by 4 — ZERO barriers in the hot loop!
  for (uint t = t_lo + sg; t < t_hi; t += 4u) {
    const uint slot = t % a.cap;
    const size_t kv_off = (size_t(slot) * a.nkv + kvh) * 32u + lane;
    float s = simd_sum(dot(q4, float4(K_cache[kv_off])));
    const uint dist = pos - t;
    if (dist < a.rel_extent) s += rl[dist];
    const float m_new = max(m_max, s);
    const float alpha = (l_sum > 0.0f) ? exp(m_max - m_new) : 0.0f;
    const float p = exp(s - m_new);
    o4 = fma(p, float4(V_cache[kv_off]), o4 * alpha);
    l_sum = l_sum * alpha + p;
    m_max = m_new;
  }

  // Merge the 4 simdgroups in fixed order 0..3
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

// 8. Post-attention / Post-MLP static sconv + residual add / block boundary save.
// Grid: (ceil(H / 256), 1, 1), TG: 256 threads.
struct PostConvArgs {
  uint M, H, ks, is_boundary, is_mlp, b_slot, b_stride;
};
kernel void ink_post_sconv_add(device const float* y_in [[buffer(0)]],
                               device const float* sconv_w [[buffer(1)]],
                               device float* conv_state [[buffer(2)]],
                               device float* h_res [[buffer(3)]],
                               device float* blocks [[buffer(4)]],
                               constant PostConvArgs& a [[buffer(5)]],
                               uint ch [[thread_position_in_grid]]) {
  if (ch >= a.H) return;
  float h0 = conv_state[0u * a.H + ch];
  float h1 = conv_state[1u * a.H + ch];
  float h2 = conv_state[2u * a.H + ch];
  const float sw3 = sconv_w[ch * 4u + 0u];
  const float sw2 = sconv_w[ch * 4u + 1u];
  const float sw1 = sconv_w[ch * 4u + 2u];
  const float sw0 = sconv_w[ch * 4u + 3u];
  const float inv_denom = rsqrt(float(a.ks + 1u));

  for (uint m = 0; m < a.M; m++) {
    const float x = y_in[size_t(m) * a.H + ch];
    const float cv = (sw0 * x + sw1 * h2 + sw2 * h1 + sw3 * h0 + x) * inv_denom;
    h0 = h1; h1 = h2; h2 = x;

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
  conv_state[0u * a.H + ch] = h0;
  conv_state[1u * a.H + ch] = h1;
  conv_state[2u * a.H + ch] = h2;
}

// 9a. Parallel FP32 router logits (float4 vectorized): scores[m, e] = sigmoid(dot(mn[m], rw[e])).
// Grid: (M, E / 8, 1), TG: 256 threads (8 simdgroups, 1 expert per simdgroup).
kernel void ink_moe_scores(device const float4* mn [[buffer(0)]],
                           device const float4* rw [[buffer(1)]],
                           device float* scores [[buffer(2)]],
                           constant uint& H [[buffer(3)]], constant uint& E [[buffer(4)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x;
  const uint sg = tid >> 5, lane = tid & 31u;
  const uint e = tg.y * 8u + sg;
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
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

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

// Fast float4 device copy (keeps looped pass transitions inside one compute encoder).
kernel void ink_copy_f32(device const float4* src [[buffer(0)]], device float4* dst [[buffer(1)]],
                         constant uint& n4 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n4) dst[gid] = src[gid];
}

// 10. Accumulate routed expert outputs [nslots, M, H] -> moe_out [M, H] in FP32.
// Grid: (M, ceil(H / 256), 1), TG: 256 threads.
kernel void ink_moe_accum(device const float* exp_y [[buffer(0)]],
                          device const uint* slots [[buffer(1)]],
                          device const uint* nslots [[buffer(2)]],
                          device const uint* top_idx [[buffer(3)]],
                          device const float* top_w [[buffer(4)]],
                          device float* out [[buffer(5)]],
                          constant uint& H [[buffer(6)]], constant uint& ystride [[buffer(7)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x;
  const uint c = tg.y * 256u + tid;
  if (c >= H) return;
  const uint ns = nslots[0];
  float acc = 0.0f;
  for (uint k = 0; k < 8u; k++) {
    const uint e = top_idx[row * 8u + k];
    const float w = top_w[row * 8u + k];
    for (uint s = 0; s < ns; s++) {
      if (slots[s] == e) {
        acc = fma(exp_y[size_t(s) * ystride + size_t(row) * H + c], w, acc);
        break;
      }
    }
  }
  out[size_t(row) * H + c] = acc;
}

// 11. Final logit softcap (30.0 * tanh(x / 30.0)) + greedy argmax.
// Grid: (M, 1, 1), TG: 256 threads.
kernel void ink_softcap_argmax(device float* logits [[buffer(0)]], device uint* tok_out [[buffer(1)]],
                               constant uint& V_valid [[buffer(2)]], constant uint& V_pad [[buffer(3)]],
                               constant float& cap [[buffer(4)]],
                               uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                               uint tpg [[threads_per_threadgroup]]) {
  threadgroup float sm_v[8];
  threadgroup uint sm_i[8];
  device float* lr = logits + size_t(row) * V_pad;
  float bv = -1e30f;
  uint bi = 0u;
  const float inv_cap = 1.0f / cap;
  for (uint v = tid; v < V_valid; v += tpg) {
    const float val = tanh(lr[v] * inv_cap) * cap;
    lr[v] = val;
    if (val > bv) { bv = val; bi = v; }
  }
  for (ushort d = 16; d > 0; d >>= 1) {
    const float ov = simd_shuffle_down(bv, d);
    const uint oi = simd_shuffle_down(bi, d);
    if (ov > bv) { bv = ov; bi = oi; }
  }
  if ((tid & 31u) == 0u) { sm_v[tid >> 5] = bv; sm_i[tid >> 5] = bi; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float fv = sm_v[0]; uint fi = sm_i[0];
    for (uint i = 1; i < (tpg >> 5); i++) if (sm_v[i] > fv) { fv = sm_v[i]; fi = sm_i[i]; }
    tok_out[row] = fi;
  }
}
