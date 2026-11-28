// Shared bit-exact GPU sampling helpers (threadgroup sum, radix float keys, Gumbel draw,
// min_p / top_k / top_p truncation floor) included by hy4.metal and inkling2.metal.
// Math and reduction order are intentionally fixed so decode, verify, and speculation stay bit-identical.

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
