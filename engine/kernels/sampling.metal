// Shared bit-exact GPU sampling helpers (threadgroup sum, radix float keys, Gumbel draw,
// min_p / top_k / top_p truncation floor, and GPU OPMK bitmask + logprobs kernel)
// included by hy4.metal / inkling2.metal and compiled standalone by engine/core/gpu_sample.h.
// Math and reduction order are intentionally fixed so decode, verify, and speculation stay bit-identical.
#ifdef DANK_SAMPLING_STANDALONE
#include <metal_stdlib>
using namespace metal;
#endif

// Threadgroup sum of one float (any power-of-two size <= 1024); every thread gets the total.
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

static inline uint fkey(float f) {
  const uint u = as_type<uint>(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

static inline float fkey_inv(uint k) {
  return as_type<float>((k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k);
}

static inline uint hmix(uint x) {
  x ^= x >> 16; x *= 0x7feb352du;
  x ^= x >> 15; x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// Deterministic Gumbel noise fixed by (seed, position, token): exact categorical sampling at temperature T.
static inline float gumbel(uint s0, uint s1, uint pos, uint t) {
  const uint h = hmix(hmix(hmix(s0 ^ 0x9E3779B9u) ^ s1) ^ hmix(pos * 0x85EBCA6Bu + 0x27d4eb2fu) ^ (t * 0xC2B2AE35u));
  const float u = (float(h >> 8) + 0.5f) * (1.0f / 16777216.0f);  // (0, 1)
  return -log(-log(u));
}

// Compute the logit floor (in scaled-logit space r[i] * it) for min_p, top_k (4-pass 8-bit MSB radix select,
// exact), and top_p (coarse 256-bin 0.125 log-prob histogram + fine 256-bin pass inside the bracket).
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

#ifdef DANK_SAMPLING_STANDALONE

struct DankSampleArgs {
  uint V;          // valid vocab size
  uint use_mask;   // 1: apply bitmask[i >> 5] & (1u << (i & 31))
  uint want_lp;    // 1: compute chosen_lp + top n_top logprobs
  uint n_top;      // 0..20
  float temp;      // <= 0: greedy
  float top_p;     // (0, 1)
  uint top_k;      // > 0
  float min_p;     // > 0
  uint s0, s1;     // seed lo/hi
  uint pos;        // token position
};

struct DankSampleOut {
  uint chosen;
  float chosen_lp;
  uint n_top;
  uint _pad;
  uint top_ids[24];
  float top_lps[24];
};

static inline bool mask_ok(device const uint* mask, uint use_mask, uint i) {
  return !use_mask || (((mask[i >> 5] >> (i & 31u)) & 1u) != 0u);
}

// 4-pass 8-bit MSB radix select returning the exact k-th largest value of `r[i] * scale`
// over tokens passing `mask_ok`.
static inline float radix_kth_masked(device const float* r, device const uint* mask, uint use_mask,
                                     uint N, float scale, uint k_want,
                                     threadgroup atomic_uint* hist, threadgroup uint* pref_need,
                                     ushort tid, ushort ntg) {
  if (tid == 0) { pref_need[0] = 0u; pref_need[1] = k_want; }
  uint m = 0u;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (uint i = tid; i < 256u; i += ntg) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint p = pref_need[0];
    for (uint i = tid; i < N; i += ntg) {
      if (!mask_ok(mask, use_mask, i)) continue;
      const uint k = fkey(r[i] * scale);
      if ((k & m) == p) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      uint acc = 0u, nd = pref_need[1];
      uint chosen_b = 0u;
      for (int b = 255; b >= 0; b--) {
        const uint h = atomic_load_explicit(&hist[b], memory_order_relaxed);
        if (acc + h >= nd) { chosen_b = uint(b); pref_need[1] = nd - acc; break; }
        acc += h;
      }
      pref_need[0] = p | (chosen_b << shift);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m |= 255u << shift;
  }
  return fkey_inv(pref_need[0]);
}

// Single-threadgroup (1024 threads) GPU masked sampler + exact logprobs / top-K logprobs.
// Eliminates the 800 KB logits memcpy and 200k-element CPU scan on OPMK and logprobs requests.
kernel void dank_sample_masked_lp(device const float* r [[buffer(0)]],
                                  device const uint* mask [[buffer(1)]],
                                  device DankSampleOut* out [[buffer(2)]],
                                  constant DankSampleArgs& a [[buffer(3)]],
                                  ushort tid [[thread_index_in_threadgroup]]) {
  constexpr ushort NTG = 1024;
  threadgroup float sh_v[32];
  threadgroup uint sh_i[32];
  threadgroup atomic_uint hist[256];
  threadgroup uint pref_need[2];
  threadgroup atomic_uint top_cnt;
  threadgroup uint top_ids[64];
  threadgroup float top_vals[64];

  const ushort sg = tid >> 5;
  const ushort lane = tid & 31u;

  // Helper: reduce (val, id) max across 1024 threads with smallest-id tie-break into (sh_v[0], sh_i[0]).
  // Pass 1: find max_l and best_id over allowed tokens.
  float lv = -INFINITY;
  uint li = 0u;
  for (uint i = tid; i < a.V; i += NTG) {
    if (!mask_ok(mask, a.use_mask, i)) continue;
    const float v = r[i];
    if (v > lv || (v == lv && i < li)) { lv = v; li = i; }
  }
  for (ushort off = 16; off > 0; off >>= 1) {
    const float ov = simd_shuffle_down(lv, off);
    const uint oi = simd_shuffle_down(li, off);
    if (ov > lv || (ov == lv && oi < li)) { lv = ov; li = oi; }
  }
  if (lane == 0) { sh_v[sg] = lv; sh_i[sg] = li; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float bv = sh_v[0];
    uint bi = sh_i[0];
    for (ushort s = 1; s < 32; s++) {
      if (sh_v[s] > bv || (sh_v[s] == bv && sh_i[s] < bi)) { bv = sh_v[s]; bi = sh_i[s]; }
    }
    sh_v[0] = bv;
    sh_i[0] = bi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float max_l = sh_v[0];
  uint chosen = sh_i[0];

  // Pass 2: temperature + min_p / top_k / top_p + deterministic Gumbel-max sampling.
  if (a.temp > 0.0f && isfinite(max_l)) {
    const float it = 1.0f / a.temp;
    const float M_ = max_l * it;
    const float lmin = (a.min_p > 0.0f) ? log(a.min_p) : -1e30f;
    float floor_ = M_ + lmin;
    if (a.top_k > 0u && a.top_k < a.V) {
      floor_ = max(floor_, radix_kth_masked(r, mask, a.use_mask, a.V, it, a.top_k, hist, pref_need, tid, NTG));
    }
    if (a.top_p > 0.0f && a.top_p < 1.0f) {
      for (uint i = tid; i < 256u; i += NTG) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float z = 0.0f;
      for (uint i = tid; i < a.V; i += NTG) {
        if (!mask_ok(mask, a.use_mask, i)) continue;
        const float l = r[i] * it;
        if (l >= floor_) {
          const float w = exp(l - M_);
          z += w;
          const uint b = uint(clamp(int((M_ - l) * 8.0f), 0, 255));
          atomic_fetch_add_explicit(&hist[b], uint(w * 65536.0f + 0.5f), memory_order_relaxed);
        }
      }
      const float Z = tg_sum(z, sh_v, tid, NTG);
      const float target = a.top_p * Z;
      if (tid == 0) {
        const ulong tgt_q = ulong(target * 65536.0f);
        ulong acc = 0;
        uint bstar = 255u;
        for (uint b = 0; b < 256u; b++) {
          const ulong h = atomic_load_explicit(&hist[b], memory_order_relaxed);
          if (acc + h >= tgt_q) { bstar = b; break; }
          acc += h;
        }
        pref_need[0] = bstar;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const uint bstar = pref_need[0];
      const float lo = bstar < 255u ? max(floor_, M_ - float(bstar + 1u) * 0.125f) : max(floor_, M_ - 40.0f);
      const float hi = M_ - float(bstar) * 0.125f;
      if (hi > lo) {
        for (uint i = tid; i < 256u; i += NTG) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv_w = 256.0f / (hi - lo);
        float above = 0.0f;
        for (uint i = tid; i < a.V; i += NTG) {
          if (!mask_ok(mask, a.use_mask, i)) continue;
          const float l = r[i] * it;
          if (l >= hi) {
            above += exp(l - M_);
          } else if (l >= lo) {
            const float w = exp(l - M_);
            const uint b = uint(clamp(int((hi - l) * inv_w), 0, 255));
            atomic_fetch_add_explicit(&hist[b], uint(w * 16777216.0f + 0.5f), memory_order_relaxed);
          }
        }
        const float A = tg_sum(above, sh_v, tid, NTG);
        if (tid == 0) {
          const float rem = max(0.0f, target - A);
          const ulong rem_q = ulong(rem * 16777216.0f);
          ulong acc = 0;
          uint fstar = 255u;
          for (uint b = 0; b < 256u; b++) {
            acc += atomic_load_explicit(&hist[b], memory_order_relaxed);
            if (acc >= rem_q) { fstar = b; break; }
          }
          pref_need[0] = fstar;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        floor_ = max(floor_, hi - float(pref_need[0] + 1u) * ((hi - lo) / 256.0f));
      } else {
        floor_ = max(floor_, lo);
      }
    }

    float bg = -INFINITY;
    uint bi = chosen;
    for (uint i = tid; i < a.V; i += NTG) {
      if (!mask_ok(mask, a.use_mask, i)) continue;
      const float ls = r[i] * it;
      if (ls >= floor_) {
        const float g = ls + gumbel(a.s0, a.s1, a.pos, i);
        if (g > bg || (g == bg && i < bi)) { bg = g; bi = i; }
      }
    }
    for (ushort off = 16; off > 0; off >>= 1) {
      const float og = simd_shuffle_down(bg, off);
      const uint oi = simd_shuffle_down(bi, off);
      if (og > bg || (og == bg && oi < bi)) { bg = og; bi = oi; }
    }
    if (lane == 0) { sh_v[sg] = bg; sh_i[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float best_g = sh_v[0];
      uint best_id = sh_i[0];
      for (ushort s = 1; s < 32; s++) {
        if (sh_v[s] > best_g || (sh_v[s] == best_g && sh_i[s] < best_id)) { best_g = sh_v[s]; best_id = sh_i[s]; }
      }
      sh_i[0] = best_id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    chosen = sh_i[0];
  }

  // Pass 3: exact log-softmax logprob + top-K logprobs over the (masked) distribution.
  float chosen_lp = 0.0f;
  uint n_out = 0u;
  if (a.want_lp && isfinite(max_l)) {
    float z = 0.0f;
    for (uint i = tid; i < a.V; i += NTG) {
      if (!mask_ok(mask, a.use_mask, i)) continue;
      z += exp(r[i] - max_l);
    }
    const float Z = tg_sum(z, sh_v, tid, NTG);
    const float lse = max_l + log(max(Z, 1e-30f));
    chosen_lp = r[chosen] - lse;

    const uint k_want = min(a.n_top, 20u);
    if (k_want > 0u) {
      const float kth = radix_kth_masked(r, mask, a.use_mask, a.V, 1.0f, k_want, hist, pref_need, tid, NTG);
      if (tid == 0) atomic_store_explicit(&top_cnt, 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // Collect strictly greater first so ties at kth can never crowd out higher logits
      for (uint i = tid; i < a.V; i += NTG) {
        if (!mask_ok(mask, a.use_mask, i)) continue;
        const float v = r[i];
        if (v > kth) {
          const uint slot = atomic_fetch_add_explicit(&top_cnt, 1u, memory_order_relaxed);
          if (slot < 64u) { top_ids[slot] = i; top_vals[slot] = v; }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = tid; i < a.V; i += NTG) {
        if (!mask_ok(mask, a.use_mask, i)) continue;
        const float v = r[i];
        if (v == kth) {
          const uint slot = atomic_fetch_add_explicit(&top_cnt, 1u, memory_order_relaxed);
          if (slot < 64u) { top_ids[slot] = i; top_vals[slot] = v; }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid == 0) {
        const uint cnt = min(atomic_load_explicit(&top_cnt, memory_order_relaxed), 64u);
        for (uint i = 1; i < cnt; i++) {
          const float kv = top_vals[i];
          const uint ki = top_ids[i];
          int j = int(i) - 1;
          while (j >= 0 && (kv > top_vals[j] || (kv == top_vals[j] && ki < top_ids[j]))) {
            top_vals[j + 1] = top_vals[j];
            top_ids[j + 1] = top_ids[j];
            j--;
          }
          top_vals[j + 1] = kv;
          top_ids[j + 1] = ki;
        }
        n_out = min(cnt, k_want);
        for (uint i = 0; i < n_out; i++) {
          out->top_ids[i] = top_ids[i];
          out->top_lps[i] = top_vals[i] - lse;
        }
      }
    }
  }
  if (tid == 0) {
    out->chosen = chosen;
    out->chosen_lp = chosen_lp;
    out->n_top = n_out;
  }
}

#endif  // DANK_SAMPLING_STANDALONE
