// DankAttn (Inkling-2-Mini): Metal 4 matmul2d tiled FlashAttention for chunked prefill (M <= 256).
// 16 query rows x 32 keys per tile, hd=128 split across 4 simdgroups (32 dims/simdgroup).
// Supports causal masking, SWA-1024 (ATT_LOCAL), kv8 int8 KV cache (ATT_KV8), rel_logits, sinks,
// and sigmoid output gating writing directly to bf16 matmul fragments (zero intermediate partials).
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#ifndef ATT_LOCAL
#define ATT_LOCAL 0
#endif
#ifndef ATT_KV8
#define ATT_KV8 0
#endif

static inline uint frag_off(uint m, uint k, uint K) {
  const uint kk = k & 31, mm = m & 15;
  const uint l = ((kk >> 2) & 1) | ((mm & 3) << 1) | (((kk >> 3) & 1) << 3) | (((mm >> 2) & 1) << 4);
  const uint i = (kk & 3) | ((mm >> 3) << 2) | ((kk >> 4) << 3);
  return ((m >> 4) * (K >> 5) + (k >> 5)) * 512 + l * 16 + i;
}

#ifndef ATT_FAST
#define ATT_FAST 1
#endif

struct AttnPfArgs {
  uint M, nh, nkv, hd, rel_extent, pos0, cap, window;
};

// Grid: (ceil(M / 16), nh, 1), TG: 128 threads (4 simdgroups x 32 lanes).
kernel void ink_attn_pf_mma(device const float* Q [[buffer(0)]],          // [M, nh, 128]
                            device const bfloat* K_cache [[buffer(1)]],   // [cap, nkv, 128]
                            device const bfloat* V_cache [[buffer(2)]],   // [cap, nkv, 128]
                            device const float* rel_logits [[buffer(3)]], // [M, nh, rel_extent]
                            device const float* gate [[buffer(4)]],       // [M, nh, 128]
                            device const float* sinks [[buffer(5)]],      // [M, nh]
                            device bfloat* out_frag [[buffer(6)]],        // frag [M, nh * 128]
                            device const char* K_i8 [[buffer(7)]],
                            device const half* K_sc [[buffer(8)]],
                            device const char* V_i8 [[buffer(9)]],
                            device const half* V_sc [[buffer(10)]],
                            constant AttnPfArgs& a [[buffer(11)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort l [[thread_index_in_simdgroup]]) {
  constexpr uint HD = 128u;
  const uint row0 = tg.x * 16u;
  const uint h = tg.y;
  const uint kvh = h / (a.nh / a.nkv);
  const uint m_end = min(a.M, row0 + 16u);
  const uint pos_lo = a.pos0 + row0;
  const uint pos_hi = a.pos0 + m_end - 1u;

#if ATT_LOCAL
  const uint t_start = (pos_lo + 1u > a.window) ? (pos_lo + 1u - a.window) : 0u;
#else
  const uint t_start = 0u;
#endif
  const uint t_end = pos_hi + 1u;
  const float scale = rsqrt(float(HD));

  const ushort m0 = ((l >> 1) & 3) + 4 * (l >> 4);
  const ushort n0 = 4 * (l & 1) + 8 * ((l >> 3) & 1);
  const uint d_sg = uint(sg) * 32u;

  constexpr auto desc =
      matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto qh = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto xa = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto cs = op.template get_destination_cooperative_tensor<decltype(qh), decltype(wb), float>();
  auto o0 = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), float>();

  // Load Q tile [16, 32] for this simdgroup's 32 dims (Q is already bf16-rounded in ink_q_rel)
  for (int i = 0; i < 16; i++) {
    const uint rr = m0 + 8 * ((i >> 2) & 1);
    const uint dd = d_sg + n0 + (i & 3) + 16 * (i >> 3);
    const uint qr = row0 + rr;
    qh[i] = (qr < a.M) ? bfloat(Q[(size_t(qr) * a.nh + h) * HD + dd]) : bfloat(0.0f);
    o0[i] = 0.0f;
  }

  // Initialize online softmax with attention sinks (sink has zero value vector -> lsum = 1.0)
  float m_max[2], l_sum[2];
  for (int r = 0; r < 2; r++) {
    const uint qr = row0 + m0 + 8 * r;
    m_max[r] = (qr < a.M) ? sinks[size_t(qr) * a.nh + h] : -INFINITY;
    l_sum[r] = 1.0f;
  }

  // Transposed Sp[4][16][32] avoids bank conflicts (each lane hits its own bank);
  // Vt[32][128] stages 32 keys x 128 dims coalesced from DRAM for P @ V.
  threadgroup float Sp[4][16][32];
  threadgroup bfloat Vt[32][132];  // +4 stride padding prevents 32-bank conflicts on column reads

  const size_t tstride = size_t(a.nkv) * HD;
#if ATT_KV8
  device const char* K8 = K_i8 + kvh * HD;
  device const char* V8 = V_i8 + kvh * HD;
#else
  device const bfloat* Kh = K_cache + kvh * HD;
  device const bfloat* Vh = V_cache + kvh * HD;
#endif

  uint slot0 = t_start % a.cap;
  for (uint j0 = t_start; j0 < t_end; j0 += 32u) {
    // 1. Load K tile [32 keys, 32 dims] into wb and compute S partial = Q @ K^T
    for (int i = 0; i < 16; i++) cs[i] = 0.0f;
    uint tk[4];
    for (int qq = 0; qq < 4; qq++) {
      const uint off = m0 + 8u * uint(qq);
      if (j0 + off < t_end) {
        uint sl = slot0 + off;
        if (sl >= a.cap) sl -= a.cap;
        tk[qq] = sl;
      } else {
        tk[qq] = 0xFFFFFFFFu;
      }
    }
#if ATT_KV8
    for (int qq = 0; qq < 4; qq++) {
      const uint sl = tk[qq];
      for (int hh = 0; hh < 2; hh++) {
        uint w = 0u;
        if (sl != 0xFFFFFFFFu) w = *(device const uint*)(K8 + size_t(sl) * tstride + d_sg + n0 + 16 * hh);
        const char4 c4 = as_type<char4>(w);
        for (int e = 0; e < 4; e++) wb[hh * 16 + qq * 4 + e] = bfloat(float(c4[e]));
      }
    }
#else
    for (int qq = 0; qq < 4; qq++) {
      const uint sl = tk[qq];
      for (int hh = 0; hh < 2; hh++) {
        uint2 w = uint2(0u);
        if (sl != 0xFFFFFFFFu) w = *(device const uint2*)(Kh + size_t(sl) * tstride + d_sg + n0 + 16 * hh);
        const bfloat4 b4 = as_type<bfloat4>(w);
        for (int e = 0; e < 4; e++) wb[hh * 16 + qq * 4 + e] = b4[e];
      }
    }
#endif
    op.run(qh, wb, cs);

    // Sync before overwriting Sp and Vt from the previous tile
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < 16; i++) Sp[sg][i][l] = cs[i];

    // Stage V tile [32 keys, 128 dims] using vectorized bfloat4/char4 loads (4 simdgroups = 4 keys/iter -> 8 iters!)
    const uint d4 = uint(l) * 4u;
    for (uint kk = uint(sg); kk < 32u; kk += 4u) {
      if (j0 + kk < t_end) {
        uint sl = slot0 + kk;
        if (sl >= a.cap) sl -= a.cap;
#if ATT_KV8
        const char4 v4 = *(device const char4*)(V8 + size_t(sl) * tstride + d4);
        Vt[kk][d4 + 0u] = bfloat(float(v4.x));
        Vt[kk][d4 + 1u] = bfloat(float(v4.y));
        Vt[kk][d4 + 2u] = bfloat(float(v4.z));
        Vt[kk][d4 + 3u] = bfloat(float(v4.w));
#else
        const bfloat4 v4 = *(device const bfloat4*)(Vh + size_t(sl) * tstride + d4);
        Vt[kk][d4 + 0u] = v4.x;
        Vt[kk][d4 + 1u] = v4.y;
        Vt[kk][d4 + 2u] = v4.z;
        Vt[kk][d4 + 3u] = v4.w;
#endif
      } else {
        Vt[kk][d4 + 0u] = bfloat(0.0f);
        Vt[kk][d4 + 1u] = bfloat(0.0f);
        Vt[kk][d4 + 2u] = bfloat(0.0f);
        Vt[kk][d4 + 3u] = bfloat(0.0f);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Reduce S across 4 simdgroups, apply kv8 K_sc + scale + causal/window mask + rel_logits
#if ATT_KV8
    float ksc[8], vsc[8];
    for (int hh = 0; hh < 2; hh++)
      for (int e = 0; e < 4; e++) {
        const uint off = n0 + uint(e) + 16u * uint(hh);
        if (j0 + off < t_end) {
          uint sl = slot0 + off;
          if (sl >= a.cap) sl -= a.cap;
          const size_t soff = size_t(sl) * a.nkv + kvh;
          ksc[hh * 4 + e] = float(K_sc[soff]);
          vsc[hh * 4 + e] = float(V_sc[soff]);
        } else {
          ksc[hh * 4 + e] = 0.0f;
          vsc[hh * 4 + e] = 0.0f;
        }
      }
#endif

    const bool maybe_rel = (pos_hi >= j0) && ((pos_lo < j0 + a.rel_extent) || (pos_hi - j0 < a.rel_extent));
#if ATT_LOCAL
    const bool full_tile = ATT_FAST && (j0 + 31u <= pos_lo) && (pos_hi - j0 < a.window) &&
                           (pos_hi - j0 < a.rel_extent) && (row0 + 16u <= a.M);
#else
    const bool full_tile = ATT_FAST && (!maybe_rel) && (j0 + 31u <= pos_lo) && (row0 + 16u <= a.M);
#endif
    float s[16];
    if (full_tile) {
#if ATT_LOCAL
      const size_t rbase0 = (size_t(row0 + m0) * a.nh + h) * a.rel_extent;
      const size_t rbase1 = (size_t(row0 + m0 + 8u) * a.nh + h) * a.rel_extent;
      const uint d0 = a.pos0 + row0 + m0 - j0 - n0;
      const uint d1 = d0 + 8u;
#endif
      for (int i = 0; i < 16; i++) {
        float sv = ((Sp[0][i][l] + Sp[1][i][l]) + (Sp[2][i][l] + Sp[3][i][l])) * scale;
#if ATT_KV8
        sv *= ksc[(i >> 3) * 4 + (i & 3)];
#endif
#if ATT_LOCAL
        const uint sub = uint((i & 3) + 16 * (i >> 3));
        sv += ((i >> 2) & 1) ? rel_logits[rbase1 + (d1 - sub)] : rel_logits[rbase0 + (d0 - sub)];
#endif
        s[i] = sv;
      }
    } else {
      for (int i = 0; i < 16; i++) {
        const uint rr = m0 + 8 * ((i >> 2) & 1);
        const uint qr = row0 + rr;
        const uint qpos = a.pos0 + qr;
        const uint kpos = j0 + n0 + (i & 3) + 16 * (i >> 3);
        bool ok = (qr < a.M) && (kpos <= qpos);
#if ATT_LOCAL
        ok = ok && ((qpos - kpos) < a.window);
#endif
        if (ok) {
          float sv = ((Sp[0][i][l] + Sp[1][i][l]) + (Sp[2][i][l] + Sp[3][i][l])) * scale;
#if ATT_KV8
          sv *= ksc[(i >> 3) * 4 + (i & 3)];
#endif
          if (maybe_rel) {
            const uint dist = qpos - kpos;
            if (dist < a.rel_extent) {
              sv += rel_logits[(size_t(qr) * a.nh + h) * a.rel_extent + dist];
            }
          }
          s[i] = sv;
        } else {
          s[i] = -INFINITY;
        }
      }
    }

    // 3. Online softmax per row r in {0, 1}
    float c[2];
    for (int r = 0; r < 2; r++) {
      float mb = -INFINITY;
      for (int i = 0; i < 16; i++) if (((i >> 2) & 1) == r) mb = max(mb, s[i]);
      mb = max(mb, simd_shuffle_xor(mb, 1));
      mb = max(mb, simd_shuffle_xor(mb, 8));
      const float mn = max(m_max[r], mb);
      c[r] = exp(m_max[r] - mn);
      float bs = 0.0f;
      for (int i = 0; i < 16; i++) {
        if (((i >> 2) & 1) == r) {
          const float p = (s[i] > -1e29f) ? exp(s[i] - mn) : 0.0f;
          s[i] = p;
          bs += p;
        }
      }
      bs += simd_shuffle_xor(bs, 1);
      bs += simd_shuffle_xor(bs, 8);
      l_sum[r] = l_sum[r] * c[r] + bs;
      m_max[r] = mn;
    }
    for (int i = 0; i < 16; i++) o0[i] *= c[(i >> 2) & 1];

    // 4. O += P @ V
    for (int i = 0; i < 16; i++) {
#if ATT_KV8
      const float ps = s[i] * vsc[(i >> 3) * 4 + (i & 3)];
#else
      const float ps = s[i];
#endif
      xa[i] = bfloat(ps);
    }
    for (int hh = 0; hh < 2; hh++)
      for (int e = 0; e < 4; e++) {
        const uint k_loc = n0 + e + 16 * hh;
        for (int qq = 0; qq < 4; qq++) {
          wb[hh * 16 + qq * 4 + e] = Vt[k_loc][d_sg + m0 + 8 * qq];
        }
      }
    op.run(xa, wb, o0);

    slot0 += 32u;
    if (slot0 >= a.cap) slot0 -= a.cap;
  }

  // 5. Finalize: divide by l_sum, apply sigmoid gate, write directly to bf16 fragment
  const uint K_tot = a.nh * HD;
  for (int i = 0; i < 16; i++) {
    const uint rr = m0 + 8 * ((i >> 2) & 1);
    const uint qr = row0 + rr;
    if (qr >= a.M) continue;
    const uint dd = d_sg + n0 + (i & 3) + 16 * (i >> 3);
    const float g = gate[(size_t(qr) * a.nh + h) * HD + dd];
    const float val = (o0[i] / l_sum[(i >> 2) & 1]) * g;
    out_frag[frag_off(qr, h * HD + dd, K_tot)] = bfloat(val);
  }
}

// GQA G=3 Shared-KV Split-K Decode/Verify Attention (compile-time ATT_LOCAL / ATT_KV8, modulo-free ring).
// Each TG handles 1 KV head x 3 Q heads -> 3x fewer KV cache DRAM reads, 100% bit-exact across M.
// Grid: (M, nkv, n_splits), TG: 128 threads (4 simdgroups).
struct AttnDecArgs {
  uint M, nh, nkv, hd, rel_extent, pos0, cap, window, n_splits;
};
kernel void ink_attn_split_gqa(device const float4* Q [[buffer(0)]],         // [M, nh, 32]
                               device const bfloat4* K_cache [[buffer(1)]],  // [cap, nkv, 32]
                               device const bfloat4* V_cache [[buffer(2)]],  // [cap, nkv, 32]
                               device const float* rel_logits [[buffer(3)]], // [M, nh, rel_extent]
                               device float* part_out [[buffer(4)]],         // [M, nh, n_splits, 130]
                               device const char4* K_i8 [[buffer(5)]],
                               device const half* K_sc [[buffer(6)]],
                               device const char4* V_i8 [[buffer(7)]],
                               device const half* V_sc [[buffer(8)]],
                               constant AttnDecArgs& a [[buffer(9)]],
                               uint3 tg [[threadgroup_position_in_grid]],
                               ushort tid [[thread_index_in_threadgroup]]) {
  const uint row = tg.x, kvh = tg.y, sp = tg.z;
  const uint sg = tid >> 5, lane = tid & 31u;
  const uint h0 = kvh * 3u;
  const uint pos = a.pos0 + row;
#if ATT_LOCAL
  const uint t_start = (pos + 1u > a.window) ? (pos + 1u - a.window) : 0u;
#else
  const uint t_start = 0u;
#endif
  const uint total_k = pos - t_start + 1u;
  const uint per_sp = (total_k + a.n_splits - 1u) / a.n_splits;
  const uint t_lo = t_start + sp * per_sp;
  const uint t_hi = min(pos + 1u, t_lo + per_sp);

  const float scale = rsqrt(float(a.hd));
  const size_t q_base = size_t(row) * a.nh + h0;
  const float4 q0 = Q[(q_base + 0u) * 32u + lane] * scale;
  const float4 q1 = Q[(q_base + 1u) * 32u + lane] * scale;
  const float4 q2 = Q[(q_base + 2u) * 32u + lane] * scale;
  device const float* rl0 = rel_logits + (q_base + 0u) * a.rel_extent;
  device const float* rl1 = rel_logits + (q_base + 1u) * a.rel_extent;
  device const float* rl2 = rel_logits + (q_base + 2u) * a.rel_extent;

  float m0 = -1e30f, l0 = 0.0f; float4 o0 = float4(0.0f);
  float m1 = -1e30f, l1 = 0.0f; float4 o1 = float4(0.0f);
  float m2 = -1e30f, l2 = 0.0f; float4 o2 = float4(0.0f);

  uint t = t_lo + sg;
  uint slot = (t < t_hi) ? (t % a.cap) : 0u;
  for (; t < t_hi; t += 4u) {
    const size_t sc_off = size_t(slot) * a.nkv + kvh;
    const size_t kv_off = sc_off * 32u + lane;
#if ATT_KV8
    const float4 k4 = float4(K_i8[kv_off]);
    const float ksc = float(K_sc[sc_off]);
    float s0 = simd_sum(dot(q0, k4)) * ksc;
    float s1 = simd_sum(dot(q1, k4)) * ksc;
    float s2 = simd_sum(dot(q2, k4)) * ksc;
#else
    const float4 k4 = float4(K_cache[kv_off]);
    float s0 = simd_sum(dot(q0, k4));
    float s1 = simd_sum(dot(q1, k4));
    float s2 = simd_sum(dot(q2, k4));
#endif
    const uint dist = pos - t;
    if (dist < a.rel_extent) {
      s0 += rl0[dist];
      s1 += rl1[dist];
      s2 += rl2[dist];
    }
#if ATT_KV8
    const float4 v4 = float4(V_i8[kv_off]);
    const float vsc = float(V_sc[sc_off]);
#else
    const float4 v4 = float4(V_cache[kv_off]);
#endif
    {
      const float mn = max(m0, s0);
      const float al = (l0 > 0.0f) ? exp(m0 - mn) : 0.0f;
      const float p = exp(s0 - mn);
#if ATT_KV8
      o0 = fma(p * vsc, v4, o0 * al);
#else
      o0 = fma(p, v4, o0 * al);
#endif
      l0 = l0 * al + p;
      m0 = mn;
    }
    {
      const float mn = max(m1, s1);
      const float al = (l1 > 0.0f) ? exp(m1 - mn) : 0.0f;
      const float p = exp(s1 - mn);
#if ATT_KV8
      o1 = fma(p * vsc, v4, o1 * al);
#else
      o1 = fma(p, v4, o1 * al);
#endif
      l1 = l1 * al + p;
      m1 = mn;
    }
    {
      const float mn = max(m2, s2);
      const float al = (l2 > 0.0f) ? exp(m2 - mn) : 0.0f;
      const float p = exp(s2 - mn);
#if ATT_KV8
      o2 = fma(p * vsc, v4, o2 * al);
#else
      o2 = fma(p, v4, o2 * al);
#endif
      l2 = l2 * al + p;
      m2 = mn;
    }
    slot += 4u;
    if (slot >= a.cap) slot -= a.cap;
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
    device float* dst = part_out + ((size_t(row) * a.nh + h0 + g) * a.n_splits + sp) * 130u;
    if (lane == 0u) { dst[0] = fm; dst[1] = fl; }
    ((device float4*)(dst + 2u))[lane] = fo;
  }
}

// ============================================================================
// Metal 4 matmul2d Prefill Message Gates (M > 16): replaces ALU dot products
// with hardware tensor ops -> ~10x faster on Base M5 & ~4x faster on M5 Ultra!
// ============================================================================
struct MgMmaArgs {
  uint H, R, M, q_shared, cat_stride_b, low_stride_f;
};

// Grid: (ceil(M / 16), R / 32, n_slots), TG: 128 threads (4 simdgroups splitting K = 2*H).
kernel void ink_mg_down_mma(device const bfloat* cat_in [[buffer(0)]],
                            device const bfloat* wd0 [[buffer(1)]],
                            device const bfloat* wd1 [[buffer(2)]],
                            device const bfloat* wd2 [[buffer(3)]],
                            device float* low_out [[buffer(4)]],
                            constant MgMmaArgs& a [[buffer(5)]],
                            uint3 tg [[threadgroup_position_in_grid]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort l [[thread_index_in_simdgroup]]) {
  const uint row0 = tg.x * 16u, n_tile = tg.y * 32u, slot = tg.z;
  const uint K = 2u * a.H;  // 4608 = 144 * 32 -> 36 k-steps per simdgroup
  const uint ks_tot = K >> 5;
  const uint ks_per = ks_tot >> 2;
  const uint ks_lo = uint(sg) * ks_per;
  const uint ks_hi = (sg == 3u) ? ks_tot : (ks_lo + ks_per);

  const ushort m0 = ((l >> 1) & 3) + 4 * (l >> 4);
  const ushort n0 = 4 * (l & 1) + 8 * ((l >> 3) & 1);

  constexpr auto desc =
      matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto xa = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto acc = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), float>();
  for (int i = 0; i < 16; i++) acc[i] = 0.0f;

  device const bfloat* cbase = cat_in + size_t(slot) * a.cat_stride_b;
  device const bfloat* wbase = (slot == 0u) ? wd0 : ((slot == 1u) ? wd1 : wd2);

  for (uint ks = ks_lo; ks < ks_hi; ks++) {
    const uint k0 = ks * 32u;
    for (int hh = 0; hh < 2; hh++) {
      const uint kc = k0 + n0 + 16u * uint(hh);
      for (int r = 0; r < 2; r++) {
        const uint qr = row0 + m0 + 8u * uint(r);
        uint2 w = uint2(0u);
        if (qr < a.M) w = *(device const uint2*)(cbase + size_t(qr) * K + kc);
        const bfloat4 b4 = as_type<bfloat4>(w);
        for (int e = 0; e < 4; e++) xa[hh * 8 + r * 4 + e] = b4[e];
      }
      for (int qq = 0; qq < 4; qq++) {
        const uint nc = n_tile + m0 + 8u * uint(qq);
        uint2 w = uint2(0u);
        if (nc < a.R) w = *(device const uint2*)(wbase + size_t(nc) * K + kc);
        const bfloat4 b4 = as_type<bfloat4>(w);
        for (int e = 0; e < 4; e++) wb[hh * 16 + qq * 4 + e] = b4[e];
      }
    }
    op.run(xa, wb, acc);
  }

  threadgroup float Sp[4][16][32];
  for (int i = 0; i < 16; i++) Sp[sg][i][l] = acc[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0u) {
    device float* dst = low_out + size_t(slot) * a.low_stride_f;
    for (int i = 0; i < 16; i++) {
      const uint qr = row0 + m0 + 8u * uint((i >> 2) & 1);
      const uint nc = n_tile + n0 + uint(i & 3) + 16u * uint(i >> 3);
      if (qr < a.M && nc < a.R) {
        dst[size_t(qr) * a.R + nc] = (Sp[0][i][l] + Sp[1][i][l]) + (Sp[2][i][l] + Sp[3][i][l]);
      }
    }
  }
}

// Grid: (ceil(M / 64), ceil(H / 32), n_slots), TG: 128 threads (4 simdgroups x 16 rows = 64 rows/TG).
kernel void ink_mg_apply_mma(device const float* h_in [[buffer(0)]], device const float* msg_in [[buffer(1)]],
                             device const float* low_in [[buffer(2)]],
                             device const float* wf0 [[buffer(3)]], device const float* wg0 [[buffer(4)]],
                             device const float* wf1 [[buffer(5)]], device const float* wg1 [[buffer(6)]],
                             device const float* wf2 [[buffer(7)]], device const float* wg2 [[buffer(8)]],
                             device float* out [[buffer(9)]],
                             constant MgMmaArgs& a [[buffer(10)]],
                             uint3 tg [[threadgroup_position_in_grid]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort l [[thread_index_in_simdgroup]]) {
  const uint row0 = (tg.x * 4u + uint(sg)) * 16u;
  if (row0 >= a.M) return;
  const uint n_tile = tg.y * 32u, slot = tg.z;
  const ushort m0 = ((l >> 1) & 3) + 4 * (l >> 4);
  const ushort n0 = 4 * (l & 1) + 8 * ((l >> 3) & 1);

  constexpr auto desc =
      matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto xa0 = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto xa1 = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto df = op.template get_destination_cooperative_tensor<decltype(xa0), decltype(wb), float>();
  auto dg = op.template get_destination_cooperative_tensor<decltype(xa0), decltype(wb), float>();
  for (int i = 0; i < 16; i++) { df[i] = 0.0f; dg[i] = 0.0f; }

  device const float* lbase = low_in + size_t(slot) * a.low_stride_f;
  // Load low[16, 64] into xa0 (cols 0..31) and xa1 (cols 32..63) once
  for (int hh = 0; hh < 2; hh++) {
    for (int r = 0; r < 2; r++) {
      const uint qr = row0 + m0 + 8u * uint(r);
      const float4 v0 = (qr < a.M) ? *(device const float4*)(lbase + size_t(qr) * a.R + n0 + 16u * uint(hh)) : float4(0.0f);
      const float4 v1 = (qr < a.M) ? *(device const float4*)(lbase + size_t(qr) * a.R + 32u + n0 + 16u * uint(hh)) : float4(0.0f);
      const bfloat4 b0 = bfloat4(v0), b1 = bfloat4(v1);
      for (int e = 0; e < 4; e++) {
        xa0[hh * 8 + r * 4 + e] = b0[e];
        xa1[hh * 8 + r * 4 + e] = b1[e];
      }
    }
  }

  device const float* wf = (slot == 0u) ? wf0 : ((slot == 1u) ? wf1 : wf2);
  device const float* wg = (slot == 0u) ? wg0 : ((slot == 1u) ? wg1 : wg2);

  // Forget projection: df = low @ wf^T (2 k-steps of 32)
  for (int ks = 0; ks < 2; ks++) {
    const uint k0 = uint(ks) * 32u;
    for (int hh = 0; hh < 2; hh++) {
      const uint kc = k0 + n0 + 16u * uint(hh);
      for (int qq = 0; qq < 4; qq++) {
        const uint nc = n_tile + m0 + 8u * uint(qq);
        const float4 v = (nc < a.H) ? *(device const float4*)(wf + size_t(nc) * a.R + kc) : float4(0.0f);
        const bfloat4 b4 = bfloat4(v);
        for (int e = 0; e < 4; e++) wb[hh * 16 + qq * 4 + e] = b4[e];
      }
    }
    if (ks == 0) op.run(xa0, wb, df);
    else op.run(xa1, wb, df);
  }

  // Gain projection: dg = low @ wg^T (2 k-steps of 32)
  for (int ks = 0; ks < 2; ks++) {
    const uint k0 = uint(ks) * 32u;
    for (int hh = 0; hh < 2; hh++) {
      const uint kc = k0 + n0 + 16u * uint(hh);
      for (int qq = 0; qq < 4; qq++) {
        const uint nc = n_tile + m0 + 8u * uint(qq);
        const float4 v = (nc < a.H) ? *(device const float4*)(wg + size_t(nc) * a.R + kc) : float4(0.0f);
        const bfloat4 b4 = bfloat4(v);
        for (int e = 0; e < 4; e++) wb[hh * 16 + qq * 4 + e] = b4[e];
      }
    }
    if (ks == 0) op.run(xa0, wb, dg);
    else op.run(xa1, wb, dg);
  }

  const size_t stride_m = size_t(a.M) * a.H;
  const size_t h_slot = a.q_shared ? 0u : slot;
  for (int i = 0; i < 16; i++) {
    const uint qr = row0 + m0 + 8u * uint((i >> 2) & 1);
    const uint nc = n_tile + n0 + uint(i & 3) + 16u * uint(i >> 3);
    if (qr < a.M && nc < a.H) {
      const size_t idx = size_t(qr) * a.H + nc;
      out[slot * stride_m + idx] =
          (1.0f - tanh(df[i])) * h_in[h_slot * stride_m + idx] + tanh(dg[i]) * msg_in[slot * stride_m + idx];
    }
  }
}

// Metal 4 matmul2d Prefill MoE Router Scores (M > 16): scores[M, E] = sigmoid(mn[M, H] @ rw[E, H]^T).
// Uses hi+lo bf16 split on rw for fp24 precision matching FP32 routing.
// Grid: (ceil(M / 16), E / 32, 1), TG: 128 threads (4 simdgroups splitting K = H).
kernel void ink_moe_scores_mma(device const float* mn [[buffer(0)]],      // [M, H] (already bf16-rounded)
                               device const float* rw [[buffer(1)]],      // [E, H] f32
                               device float* scores [[buffer(2)]],        // [M, E]
                               constant uint& M [[buffer(3)]], constant uint& H [[buffer(4)]],
                               constant uint& E [[buffer(5)]],
                               uint2 tg [[threadgroup_position_in_grid]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort l [[thread_index_in_simdgroup]]) {
  const uint row0 = tg.x * 16u, e0 = tg.y * 32u;
  const uint ks_tot = H >> 5;  // 2304 / 32 = 72 -> 18 k-steps per simdgroup
  const uint ks_per = ks_tot >> 2;
  const uint ks_lo = uint(sg) * ks_per;
  const uint ks_hi = (sg == 3u) ? ks_tot : (ks_lo + ks_per);

  const ushort m0 = ((l >> 1) & 3) + 4 * (l >> 4);
  const ushort n0 = 4 * (l & 1) + 8 * ((l >> 3) & 1);

  constexpr auto desc =
      matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto xa = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto wbh = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto wbl = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto acc = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wbh), float>();
  for (int i = 0; i < 16; i++) acc[i] = 0.0f;

  for (uint ks = ks_lo; ks < ks_hi; ks++) {
    const uint k0 = ks * 32u;
    for (int hh = 0; hh < 2; hh++) {
      const uint kc = k0 + n0 + 16u * uint(hh);
      for (int r = 0; r < 2; r++) {
        const uint qr = row0 + m0 + 8u * uint(r);
        const float4 v = (qr < M) ? *(device const float4*)(mn + size_t(qr) * H + kc) : float4(0.0f);
        const bfloat4 b4 = bfloat4(v);
        for (int e = 0; e < 4; e++) xa[hh * 8 + r * 4 + e] = b4[e];
      }
      for (int qq = 0; qq < 4; qq++) {
        const uint ec = e0 + m0 + 8u * uint(qq);
        const float4 v = (ec < E) ? *(device const float4*)(rw + size_t(ec) * H + kc) : float4(0.0f);
        const bfloat4 bh = bfloat4(v);
        const bfloat4 bl = bfloat4(v - float4(bh));
        for (int e = 0; e < 4; e++) {
          wbh[hh * 16 + qq * 4 + e] = bh[e];
          wbl[hh * 16 + qq * 4 + e] = bl[e];
        }
      }
    }
    op.run(xa, wbh, acc);
    op.run(xa, wbl, acc);
  }

  threadgroup float Sp[4][16][32];
  for (int i = 0; i < 16; i++) Sp[sg][i][l] = acc[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0u) {
    for (int i = 0; i < 16; i++) {
      const uint qr = row0 + m0 + 8u * uint((i >> 2) & 1);
      const uint ec = e0 + n0 + uint(i & 3) + 16u * uint(i >> 3);
      if (qr < M && ec < E) {
        const float v = (Sp[0][i][l] + Sp[1][i][l]) + (Sp[2][i][l] + Sp[3][i][l]);
        scores[size_t(qr) * E + ec] = 1.0f / (1.0f + exp(-v));
      }
    }
  }
}
