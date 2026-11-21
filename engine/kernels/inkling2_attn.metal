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
    float s[16];
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
