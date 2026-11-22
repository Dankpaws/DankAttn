// mm2: register-direct W4/W8 x bf16 matmul for M <= 16 * TILES rows, two-level split-K.
//
// A threadgroup of SG simdgroups owns one 32-wide output fragment and one of `ksplit`
// K ranges; its simdgroups take consecutive group-aligned slices of that range. Slices
// are reduced in threadgroup memory in simdgroup order, then the ksplit threadgroup
// partials in split order by whichever threadgroup arrives last. Both orders are fixed
// per tensor (never by M or timing), so a row's bits are the same at every M.
//
// Weights: converter layout (convert/layout.py). Activations: bf16 left-operand frags.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#ifndef WBITS
#define WBITS 4
#endif
#ifndef GS
#define GS 128
#endif
#if WBITS == 6
#define WLOAD uint2
#define WV 3            // uint2 per lane per k-step (24 B = 16 B lo4 + 8 B hi2)
#elif WBITS == 40       // NVFP4 (E2M1, GS=16, in-unpack scaling)
#define WLOAD uint4
#define WV 1            // uint4 per lane per k-step (16 B = 32 E2M1 nibbles)
#elif WBITS == 80       // MXFP8 (E4M3FN, GS=32, f16 power-of-two scales)
#define WLOAD uint4
#define WV 2            // uint4 per lane per k-step (32 B = 32 FP8 bytes)
#else
#define WLOAD uint4
#define WV (WBITS / 4)  // uint4 per lane per k-step
#endif
#ifndef XMODE
#define XMODE 0         // 0: bf16 frags; 1: f32 rows, silu(x / xdiv); 2: f32 rows [gate | up], silu(g) * u
#endif
#ifndef PF
#define PF 1            // k-steps of weights in flight per simdgroup
#endif

static inline float xsilu(float x) { return x / (1.0f + exp(-x)); }

#if WBITS == 40
constant float E2M1_LUT[16] = {
  0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};
#elif WBITS == 80
static inline float decode_fp8_e4m3(uint b) {
  const uint s = (b >> 7) & 1u, e = (b >> 3) & 15u, m = b & 7u;
  const float v = (e == 0u) ? (float(m) * (1.0f / 512.0f)) : ldexp(1.0f + float(m) * 0.125f, int(e) - 7);
  return s ? -v : v;
}
#endif

// Unpack one lane's k-step of W4 / W6 / W8 / NVFP4(40) / MXFP8(80) weights into 32 bfloats.
// For NVFP4 (WBITS == 40), sp0/sp1 are the two GS=16 half scale slabs for this 32-wide k-step
// and nbase = ((l >> 1) & 3) + 4 * (l >> 4) is this lane's base output column in 0..7.
template<typename Dst>
static inline void unpack_w(thread const WLOAD* v, thread Dst& out,
                            device const half* sp0 = nullptr, device const half* sp1 = nullptr,
                            ushort nbase = 0) {
#if WBITS == 4
  const uint w4[4] = {v[0].x, v[0].y, v[0].z, v[0].w};
#pragma clang loop unroll(full)
  for (int j = 0; j < 4; j++)
#pragma clang loop unroll(full)
    for (int i = 0; i < 4; i++) {
      out[8 * j + i] = bfloat(float(int((w4[j] >> (8 * i)) & 15u) - 8));
      out[8 * j + 4 + i] = bfloat(float(int((w4[j] >> (8 * i + 4)) & 15u) - 8));
    }
#elif WBITS == 40
  const uint w4[4] = {v[0].x, v[0].y, v[0].z, v[0].w};
  const float sc[2][4] = {
    {float(sp0[nbase]), float(sp0[nbase + 8]), float(sp0[nbase + 16]), float(sp0[nbase + 24])},
    {float(sp1[nbase]), float(sp1[nbase + 8]), float(sp1[nbase + 16]), float(sp1[nbase + 24])}
  };
#pragma clang loop unroll(full)
  for (int j = 0; j < 4; j++) {
    const int kh = j >> 1, g0 = (j & 1) << 1;
    const float s_lo = sc[kh][g0], s_hi = sc[kh][g0 + 1];
#pragma clang loop unroll(full)
    for (int i = 0; i < 4; i++) {
      out[8 * j + i] = bfloat(E2M1_LUT[(w4[j] >> (8 * i)) & 15u] * s_lo);
      out[8 * j + 4 + i] = bfloat(E2M1_LUT[(w4[j] >> (8 * i + 4)) & 15u] * s_hi);
    }
  }
#elif WBITS == 6
  const uint w4[4] = {v[0].x, v[0].y, v[1].x, v[1].y};
  const uint h2[2] = {v[2].x, v[2].y};
#pragma clang loop unroll(full)
  for (int j = 0; j < 4; j++) {
    const uint hw = h2[j >> 1] >> (16 * (j & 1));
#pragma clang loop unroll(full)
    for (int i = 0; i < 4; i++) {
      out[8 * j + i] = bfloat(float(int(((w4[j] >> (8 * i)) & 15u) | (((hw >> (2 * i)) & 3u) << 4)) - 32));
      out[8 * j + 4 + i] = bfloat(float(int(((w4[j] >> (8 * i + 4)) & 15u) | (((hw >> (2 * i + 8)) & 3u) << 4)) - 32));
    }
  }
#elif WBITS == 80
  const uint w8[8] = {v[0].x, v[0].y, v[0].z, v[0].w, v[1].x, v[1].y, v[1].z, v[1].w};
#pragma clang loop unroll(full)
  for (int j = 0; j < 8; j++)
#pragma clang loop unroll(full)
    for (int i = 0; i < 4; i++) out[4 * j + i] = bfloat(decode_fp8_e4m3((w8[j] >> (8 * i)) & 255u));
#else
  const uint w8[8] = {v[0].x, v[0].y, v[0].z, v[0].w, v[1].x, v[1].y, v[1].z, v[1].w};
#pragma clang loop unroll(full)
  for (int j = 0; j < 8; j++)
#pragma clang loop unroll(full)
    for (int i = 0; i < 4; i++) out[4 * j + i] = bfloat(float(int(char((w8[j] >> (8 * i)) & 255u))));
#endif
}

struct MmArgs {
  uint N, K, M;
  uint ksplit;            // threadgroup-level K splits
  uint wstride, sstride;  // grouped: per expert, uint4 / half
  uint xstride;           // grouped: per slot, uint4 (0: shared input)
  uint pstride;           // grouped: per slot, floats of partials
  uint ystride;           // grouped: per slot, floats of output
  uint sgs;               // simdgroups per fragment (K slices); the threadgroup holds nsg / sgs fragments
  uint ldx;               // XMODE 1/2: floats per input row (XMODE 2: up half starts at K)
  float xdiv;             // XMODE 1 divisor
  uint psplit;            // floats between split slabs of the partials
  // epilogue (EPI): 1 hc down -> silu(y / xdiv) frag for n < R; 2 hc up (streams interleaved
  // by 8) -> x = mean_s sigmoid(y) * hn, f32 and frag; 3 expert gate|up (interleaved by 16)
  // -> silu(g) * u frag per slot
  uint R, H, HC;
  // ride-along prefetch (ungrouped mm2 only): threadgroups with z >= pf_z0 do no matmul work;
  // each thread reads 16 uint4 of the NEXT op's weights (RAW) so that op finds them in cache.
  uint pf_z0, pf_gx, pf_gy, pf_tpt, pf_n;  // first prefetch z, grid x/y, threads per tg, uint4 in range
  // ksplit > 1 without atomics: red 1 = store this split's partials and stop; red 2 = the
  // follow-up dispatch (one simdgroup per fragment) that sums them in split order + epilogue
  uint red;
  uint rm;   // GROUPED decode: store / post-process only the routed rows (rmask from moe_route)
};

static inline uint frag_off(uint m, uint k, uint K) {
  const uint kk = k & 31, mm = m & 15;
  const uint l = ((kk >> 2) & 1) | ((mm & 3) << 1) | (((kk >> 3) & 1) << 3) | (((mm >> 2) & 1) << 4);
  const uint i = (kk & 3) | ((mm >> 3) << 2) | ((kk >> 4) << 3);
  return ((m >> 4) * (K >> 5) + (k >> 5)) * 512 + l * 16 + i;
}
static inline float esigm(float x) { return 1.0f / (1.0f + exp(-x)); }
#ifndef EPI
#define EPI 0
#endif

// Epilogue over one finished 32-column fragment (rows m < Mloc of this block, global row0).
static inline void epilogue(device const float* Y, uint N, uint nb, uint Mloc, uint row0, ushort l,
                            device bfloat* e0, device const float* e1, device float* e2, constant MmArgs& a,
                            uint live = 0xFFFFu
#if EPI == 4
                            , device float* e3 = nullptr, device float* e4 = nullptr, device float* e5 = nullptr
#endif
                            ) {
#if EPI != 0
  simdgroup_barrier(mem_flags::mem_device);
#if EPI == 1
  for (uint idx = l; idx < Mloc * 32; idx += 32) {
    const uint m = idx >> 5, n = nb + (idx & 31);
    if (n < a.R) e0[frag_off(row0 + m, n, a.R)] = bfloat(xsilu(Y[m * N + n] / a.xdiv));
  }
#elif EPI == 2
  // fragment holds channels c0..c0+7 for each stream s at local column s*8 + j
  const uint c0 = (nb / 32) * 8, W = a.H * a.HC;
  for (uint idx = l; idx < Mloc * 8; idx += 32) {
    const uint m = idx >> 3, j = idx & 7, c = c0 + j;
    float s = 0.0f;
    for (uint t = 0; t < a.HC; t++) s += esigm(Y[m * N + nb + t * 8 + j]) * e1[size_t(row0 + m) * W + t * a.H + c];
    s = s / float(a.HC);
    e2[size_t(row0 + m) * a.H + c] = s;
    e0[frag_off(row0 + m, c, a.H)] = bfloat(s);
  }
#elif EPI == 3
  // fragment f: gate of channels 16f.. at local 0..15, up at 16..31; e0 is this slot's frag
  const uint c0 = (nb / 32) * 16;
  for (uint idx = l; idx < Mloc * 16; idx += 32) {
    const uint m = idx >> 4, j = idx & 15;
    if (!((live >> m) & 1u)) continue;
#ifdef SWIGLU_LIMIT  // Hy4: gate clamped above, up both ways, before silu(gate) * up
    e0[frag_off(row0 + m, c0 + j, a.R)] = bfloat(xsilu(min(Y[m * N + nb + j], float(SWIGLU_LIMIT))) *
                                                 clamp(Y[m * N + nb + 16 + j], -float(SWIGLU_LIMIT), float(SWIGLU_LIMIT)));
#elif defined(SWIGLU_SOFTCAP_G)  // Inkling2: softcap(g, G) * sigmoid(g) * softcap(u, U)
    const float gv = Y[m * N + nb + j], uv = Y[m * N + nb + 16 + j];
    e0[frag_off(row0 + m, c0 + j, a.R)] = bfloat(
        (tanh(gv / float(SWIGLU_SOFTCAP_G)) * float(SWIGLU_SOFTCAP_G)) * esigm(gv) *
        (tanh(uv / float(SWIGLU_SOFTCAP_U)) * float(SWIGLU_SOFTCAP_U)));
#else
    e0[frag_off(row0 + m, c0 + j, a.R)] = bfloat(xsilu(Y[m * N + nb + j]) * Y[m * N + nb + 16 + j]);
#endif
  }
#elif EPI == 4
  // Inkling2 o_proj + post-attention sconv + residual/boundary save (single-lane M <= 16, row0 == 0).
  // e0 = sconv_w, e1 = conv_state_in, e2 = h_res, e3 = blocks, e4 = conv_hist, e5 = conv_state_out.
  // a.H = is_boundary, a.HC = b_slot, a.R = b_stride, ks = 4 (inv_denom = rsqrt(5.0f)).
  const uint ch = nb + l;
  if (ch < N) {
    device const float* sconv_w = (device const float*)e0;
    const float sw3 = sconv_w[ch * 4u + 0u];
    const float sw2 = sconv_w[ch * 4u + 1u];
    const float sw1 = sconv_w[ch * 4u + 2u];
    const float sw0 = sconv_w[ch * 4u + 3u];
    const float inv_denom = rsqrt(5.0f);
    float h0 = e1[0u * N + ch];
    float h1 = e1[1u * N + ch];
    float h2 = e1[2u * N + ch];
    e4[0u * N + ch] = h0;
    e4[1u * N + ch] = h1;
    e4[2u * N + ch] = h2;
    for (uint m = 0u; m < Mloc; m++) {
      const float x = Y[size_t(m) * N + ch];
      e4[(3u + m) * N + ch] = x;
      const float cv = (sw0 * x + sw1 * h2 + sw2 * h1 + sw3 * h0 + x) * inv_denom;
      const size_t hi = size_t(m) * N + ch;
      const float r_old = e2[hi];
      if (a.H) {
        e3[size_t(a.HC) * a.R + hi] = r_old;
        e2[hi] = cv;
      } else {
        e2[hi] = r_old + cv;
      }
      h0 = h1; h1 = h2; h2 = x;
    }
    e5[0u * N + ch] = h0;
    e5[1u * N + ch] = h1;
    e5[2u * N + ch] = h2;
  }
#endif
#endif
}



// EPI 3 straight from a lane's 16 outputs (prefill mmp / mmpg): element i < 8 is gate column n0 + (i & 3)
// of row m0 + 8 ((i >> 2) & 1), element i + 8 its up column (+16): the same values the memory epilogue
// reloads, so the same bits, without the f32 [rows][N] round trip (or its buffer).
static inline void epi3_regs(thread const float* r2, uint nb, uint Mloc, uint row0, ushort n0, ushort m0,
                             device bfloat* e0, constant MmArgs& a, uint live = 0xFFFFu) {
#if EPI == 3
  const uint c0 = (nb / 32) * 16;
  for (int i = 0; i < 8; i++) {
    const uint m = m0 + 8 * ((i >> 2) & 1), j = n0 + (i & 3);
    if (m >= Mloc || !((live >> m) & 1u)) continue;
    const float g = r2[i], u = r2[i + 8];
#ifdef SWIGLU_LIMIT
    e0[frag_off(row0 + m, c0 + j, a.R)] = bfloat(xsilu(min(g, float(SWIGLU_LIMIT))) * clamp(u, -float(SWIGLU_LIMIT), float(SWIGLU_LIMIT)));
#elif defined(SWIGLU_SOFTCAP_G)
    e0[frag_off(row0 + m, c0 + j, a.R)] = bfloat(
        (tanh(g / float(SWIGLU_SOFTCAP_G)) * float(SWIGLU_SOFTCAP_G)) * esigm(g) *
        (tanh(u / float(SWIGLU_SOFTCAP_U)) * float(SWIGLU_SOFTCAP_U)));
#else
    e0[frag_off(row0 + m, c0 + j, a.R)] = bfloat(xsilu(g) * u);
#endif
  }
#endif
}

// TILES (1, 2 or 4): 16-row tiles per threadgroup that share every unpacked weight operand, so a
// step of up to 16 * TILES rows reads each weight once. Tile t is rows 16t.. of this threadgroup's
// rows; its arithmetic (and so its bits) is that of the same 16 rows with TILES 1.
#ifndef TILES
#define TILES 1
#endif
#if TILES > 1 && defined(GATHER)
#error "GATHER tiles are gathered rows; TILES applies to decode/verify rows only"
#endif

kernel void mm2(device const WLOAD* W [[buffer(0)]], device const half* S [[buffer(1)]],
                device const uint4* X [[buffer(2)]], device float* P [[buffer(3)]],
                device float* Y [[buffer(4)]], constant MmArgs& a [[buffer(5)]],
                device const uint* slots [[buffer(6)]], device const uint* nslots [[buffer(7)]],
                device atomic_uint* cnt [[buffer(8)]],
                device bfloat* E0 [[buffer(9)]], device const float* E1 [[buffer(10)]], device float* E2 [[buffer(11)]],
                threadgroup float* red [[threadgroup(0)]],  // [TILES][simdgroups][16][32], sized by the host
                uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                ushort nsg [[simdgroups_per_threadgroup]], ushort l [[thread_index_in_simdgroup]],
#if EPI == 4
                device float* e3_buf [[buffer(12)]],
                device float* e4_buf [[buffer(13)]], device float* e5_buf [[buffer(14)]],
#else
                device const uint* trows [[buffer(12)]],    // GATHER: tile rows (16 per tile)
                device const uint4* RAW [[buffer(13)]], device uint* pfsink [[buffer(14)]],
#endif
                device const uint* rmask [[buffer(15)]]) {
#if !defined(GROUPED) && EPI != 4
  if (tg.z >= a.pf_z0) {
    const uint i = (((tg.z - a.pf_z0) * a.pf_gy + tg.y) * a.pf_gx + tg.x) * a.pf_tpt + uint(sg) * 32 + l;
    const uint b = i * 16;
    if (b >= a.pf_n) return;
    uint4 s = uint4(0);
    for (uint j = 0; j < 16; j++) s += RAW[b + j];
    if (s.x == 0xFFFFFFFFu && s.y == 0x9E3779B9u) pfsink[0] = s.z;  // keeps the reads alive; scratch buffer only
    return;
  }
#endif
  const ushort nft = nsg / a.sgs, fl = sg / a.sgs;   // fragments per threadgroup, this simdgroup's fragment
  const uint nf = tg.x * nft + fl, split = tg.y;
  const ushort sl = sg % a.sgs;                      // this simdgroup's slice of the fragment
  uint Mloc[TILES];  // rows of each 16-row tile
  const uint KF = (a.K / 32) * 64;                   // uint4 per 16-row block of a bf16 frag
#ifdef GROUPED
  if (tg.z >= nslots[0]) return;
  {
    const uint e = slots[tg.z];
    W += size_t(e) * a.wstride;
    S += size_t(e) * a.sstride;
#ifndef GATHER
    X += size_t(tg.z) * a.xstride;
#endif
    P += size_t(tg.z) * a.pstride;
    Y += size_t(tg.z) * a.ystride;
    cnt += tg.z * (a.N / 32);
    E0 += size_t(tg.z) * 16 * TILES * a.R;
  }
  const uint row0 = 0;
  for (uint t = 0; t < TILES; t++) Mloc[t] = TILES == 1 ? a.M : (a.M > 16 * t ? min(16u, a.M - 16 * t) : 0u);
#else
  // row blocks: tg.z picks 16 * TILES rows; their outputs, partials and counters are disjoint
  if (a.red == 2) {  // the follow-up dispatch of red 1: tg.z is one 16-row block whatever TILES is
    const uint rb = tg.z;
    Y += size_t(rb) * 16 * a.N;
    P += size_t(rb) * 16 * a.N;
    const uint Mb = min(16u, a.M - rb * 16), row0 = rb * 16;
    const uint f = tg.x * nsg + sg;
    if (f >= a.N / 32) return;
    const uint fb = f * 32;
    for (uint idx = l; idx < Mb * 32; idx += 32) {
      const uint m = idx >> 5, n = fb + (idx & 31);
      float v[16];  // issue every load first, then sum in split order
#pragma clang loop unroll(full)
      for (uint p = 0; p < 16; p++) v[p] = p < a.ksplit ? P[size_t(p) * a.psplit + m * a.N + n] : 0.0f;
      float s = 0.0f;
#pragma clang loop unroll(full)
      for (uint p = 0; p < 16; p++) if (p < a.ksplit) s += v[p];
      Y[m * a.N + n] = s;
    }
    epilogue(Y, a.N, fb, Mb, row0, l, E0, E1, E2, a);
    return;
  }
  {
    const uint rb = tg.z * TILES;
    X += size_t(rb) * KF;  // (as before for XMODE 1/2 too: their f32 rows only meet rb 0, in decode)
    Y += size_t(rb) * 16 * a.N;
    P += size_t(rb) * 16 * a.N;
    cnt += tg.z * (a.N / 32);
    for (uint t = 0; t < TILES; t++) Mloc[t] = a.M > (rb + t) * 16 ? min(16u, a.M - (rb + t) * 16) : 0u;
  }
  const uint row0 = tg.z * TILES * 16;
#endif
  constexpr auto desc =
      matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto xa = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto c0 = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), float>();
#if TILES > 1
  auto c1 = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), float>();
#endif
#if TILES > 2
  auto c2 = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), float>();
  auto c3 = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), float>();
#endif

#if WBITS == 40
  const uint KS = a.K / 32, G = a.K / 16;
  const uint ksper = KS / (a.ksplit * a.sgs);
  const uint ks0 = (split * a.sgs + sl) * ksper, nks = ksper;
#else
  const uint KS = a.K / 32, G = a.K / GS, KPG = GS / 32;
  const uint gper = G / (a.ksplit * a.sgs);            // groups per simdgroup slice
  const uint ks0 = (split * a.sgs + sl) * gper * KPG, nks = gper * KPG;
#endif
  const ushort n0 = 4 * (l & 1) + 8 * ((l >> 3) & 1), m0 = ((l >> 1) & 3) + 4 * (l >> 4);
  device const WLOAD* Wp = W + (size_t(nf) * KS * 32 + l) * WV;
  device const half* Sp = S + size_t(nf) * G * 32;

#ifdef GATHER
  const uint grow[2] = {trows[tg.z * 16 + m0], trows[tg.z * 16 + m0 + 8]};
  // frag_off(src, 32 ks + n0 + 16 h, K) = goff[h][r] + 512 ks: the row and lane part, once
  uint goff[2][2];
  for (int h = 0; h < 2; h++)
    for (int r = 0; r < 2; r++) goff[h][r] = grow[r] == 0xFFFFFFFFu ? 0u : frag_off(grow[r], n0 + 16 * h, a.K);
#endif
  float acc[TILES][16];
  for (int t = 0; t < TILES; t++)
    for (int i = 0; i < 16; i++) acc[t][i] = 0.0f;
  for (int i = 0; i < 16; i++) {
    c0[i] = 0.0f;
#if TILES > 1
    c1[i] = 0.0f;
#endif
#if TILES > 2
    c2[i] = 0.0f;
    c3[i] = 0.0f;
#endif
  }
  // PF k-steps of weights in flight; the ring index is a compile-time constant after
  // unrolling, so the ring stays in registers.
  WLOAD ring[PF][WV];
#pragma clang loop unroll(full)
  for (uint u = 0; u < PF; u++)
    if (u < nks)
      for (int v = 0; v < WV; v++) ring[u][v] = Wp[size_t(ks0 + u) * 32 * WV + v];
  for (uint t = 0; t < nks; t += PF) {
#pragma clang loop unroll(full)
    for (uint u = 0; u < PF; u++) {
      if (t + u >= nks) break;
      const uint ks = ks0 + t + u;
      // unpack straight into the operand (no int8 temporary: fewer live registers), once for every tile
#if WBITS == 40
      unpack_w(ring[u], wb, Sp + (2 * ks) * 32, Sp + (2 * ks + 1) * 32, m0);
#else
      unpack_w(ring[u], wb);
#endif
      if (t + u + PF < nks)
        for (int v = 0; v < WV; v++) ring[u][v] = Wp[size_t(ks + PF) * 32 * WV + v];
      // tile tt's left operand, then its multiply into c
#if XMODE == 0 && defined(GATHER)
#define LOAD_X(tt)                                                                                         \
      for (int h = 0; h < 2; h++)                                                                          \
        for (int r = 0; r < 2; r++) {                                                                      \
          const uint src = grow[r];                                                                        \
          const uint2 w = src == 0xFFFFFFFFu ? uint2(0) : *(const device uint2*)((const device bfloat*)X + goff[h][r] + ks * 512); \
          const bfloat4 b4 = as_type<bfloat4>(w);                                                          \
          for (int e = 0; e < 4; e++) xa[h * 8 + r * 4 + e] = b4[e];                                       \
        }
#elif XMODE == 0
#define LOAD_X(tt)                                                                                         \
      {                                                                                                    \
        const device bfloat* xp = (const device bfloat*)(X + size_t(tt) * KF + (ks * 32 + l) * 2);         \
        for (int i = 0; i < 16; i++) xa[i] = xp[i];                                                        \
      }
#else
      // f32 rows, transformed exactly as the standalone kernels did before fusion
#if XMODE == 1
#define XVAL(m, k) bfloat(xsilu(xr[(m) * a.ldx + (k)] / a.xdiv))
#else
#define XVAL(m, k) bfloat(xsilu(xr[(m) * a.ldx + (k)]) * xr[(m) * a.ldx + a.K + (k)])
#endif
#define LOAD_X(tt)                                                                                         \
      {                                                                                                    \
        const device float* xr = (const device float*)X;                                                   \
        for (int i = 0; i < 16; i++) {                                                                     \
          const uint m = 16 * (tt) + m0 + 8 * ((i >> 2) & 1), k = ks * 32 + n0 + (i & 3) + 16 * (i >> 3); \
          xa[i] = m < a.M ? XVAL(m, k) : bfloat(0.0f);                                                     \
        }                                                                                                  \
      }
#endif
      LOAD_X(0)
      op.run(xa, wb, c0);
#if TILES > 1
      if (Mloc[1]) { LOAD_X(1) op.run(xa, wb, c1); }
#endif
#if TILES > 2
      if (Mloc[2]) { LOAD_X(2) op.run(xa, wb, c2); }
      if (Mloc[3]) { LOAD_X(3) op.run(xa, wb, c3); }
#endif
#undef LOAD_X
#if WBITS != 40
      if ((ks + 1) % KPG == 0) {
        const device half* sp = Sp + (ks / KPG) * 32;
        float s8[8];
        for (int q4 = 0; q4 < 4; q4++) { s8[q4] = float(sp[n0 + q4]); s8[4 + q4] = float(sp[n0 + 16 + q4]); }
        for (int i = 0; i < 16; i++) {
          const float sc = s8[(i & 3) + 4 * (i >> 3)];
          acc[0][i] = fma(c0[i], sc, acc[0][i]); c0[i] = 0.0f;
#if TILES > 1
          acc[1][i] = fma(c1[i], sc, acc[1][i]); c1[i] = 0.0f;
#endif
#if TILES > 2
          acc[2][i] = fma(c2[i], sc, acc[2][i]); c2[i] = 0.0f;
          acc[3][i] = fma(c3[i], sc, acc[3][i]); c3[i] = 0.0f;
#endif
        }
      }
#endif
    }
  }
#if WBITS == 40
  for (int i = 0; i < 16; i++) {
    acc[0][i] = c0[i];
#if TILES > 1
    acc[1][i] = c1[i];
#endif
#if TILES > 2
    acc[2][i] = c2[i];
    acc[3][i] = c3[i];
#endif
  }
#endif
  // level 1: simdgroups of this threadgroup, in order (each tile on its own)
  if (a.sgs > 1) {
    for (int t = 0; t < TILES; t++)
      for (int i = 0; i < 16; i++) red[((t * nsg + sg) * 16 + i) * 32 + l] = acc[t][i];  // [.][i][lane]: lanes on distinct banks
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sl != 0) return;
    for (int t = 0; t < TILES; t++)
      for (int i = 0; i < 16; i++) {
        float s = 0.0f;
        for (ushort k = 0; k < a.sgs; k++) s += red[((t * nsg + sg + k) * 16 + i) * 32 + l];
        acc[t][i] = s;
      }
  }
  // level 2: threadgroups across ksplit, in split order, by the last arrival
  const uint nb = nf * 32;
  if (a.ksplit == 1) {
    for (uint t = 0; t < TILES; t++) {
      if (!Mloc[t]) continue;
      // routed rows of this tile, read here rather than at entry so the mask is not live across the k-loop
      const uint live = a.rm ? rmask[tg.z * 4 + t] : 0xFFFFu;
#if EPI == 3
      epi3_regs(acc[t], nb, Mloc[t], row0 + 16 * t, n0, m0, E0, a, live);
#else
      device float* Yt = Y + size_t(t) * 16 * a.N;
      for (int i = 0; i < 16; i++) {
        const uint m = m0 + 8 * ((i >> 2) & 1), n = nb + n0 + (i & 3) + 16 * (i >> 3);
        if (m < Mloc[t] && ((live >> m) & 1u)) Yt[m * a.N + n] = acc[t][i];
      }
#if EPI == 4
      epilogue(Yt, a.N, nb, Mloc[t], row0 + 16 * t, l, E0, E1, E2, a, live, e3_buf, e4_buf, e5_buf);
#else
      epilogue(Yt, a.N, nb, Mloc[t], row0 + 16 * t, l, E0, E1, E2, a, live);
#endif
#endif
    }
    return;
  }
  if (a.red == 1) {
    for (uint t = 0; t < TILES; t++)
      for (int i = 0; i < 16; i++) {
        const uint m = m0 + 8 * ((i >> 2) & 1), n = nb + n0 + (i & 3) + 16 * (i >> 3);
        if (m < Mloc[t]) P[size_t(split) * a.psplit + (16 * t + m) * a.N + n] = acc[t][i];
      }
    return;
  }
  device atomic_float* Pa = (device atomic_float*)P;
  for (uint t = 0; t < TILES; t++) {
    const uint live = a.rm ? rmask[tg.z * 4 + t] : 0xFFFFu;
    for (int i = 0; i < 16; i++) {
      const uint m = m0 + 8 * ((i >> 2) & 1), n = nb + n0 + (i & 3) + 16 * (i >> 3);
      if (m < Mloc[t] && ((live >> m) & 1u))
        atomic_store_explicit(&Pa[size_t(split) * a.psplit + (16 * t + m) * a.N + n], acc[t][i], memory_order_relaxed);
    }
  }
  // Every lane releases its own partial stores through its own increment (a fence only orders the
  // thread that executes it), and every lane of the last simdgroup acquires through its own RMW on
  // the counter, which reads the final value. With one counting lane, lanes 1-31's stores were not
  // formally ordered: shader validation's timing made the reduce read stale partials.
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  const uint arrivals = 32 * a.ksplit;
  const bool mine = atomic_fetch_add_explicit(&cnt[nf], 1u, memory_order_relaxed) == arrivals - 1;
  if (!simd_any(mine)) return;
  (void)atomic_fetch_add_explicit(&cnt[nf], 0u, memory_order_relaxed);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  for (uint t = 0; t < TILES; t++) {
    if (!Mloc[t]) continue;
    const uint live = a.rm ? rmask[tg.z * 4 + t] : 0xFFFFu;
    device float* Yt = Y + size_t(t) * 16 * a.N;
    for (uint idx = l; idx < Mloc[t] * 32; idx += 32) {
      const uint m = idx >> 5, n = nb + (idx & 31);
      if (!((live >> m) & 1u)) continue;
      float v[16];  // issue every load first, then sum in split order
#pragma clang loop unroll(full)
      for (uint p = 0; p < 16; p++)
        v[p] = p < a.ksplit ? atomic_load_explicit(&Pa[size_t(p) * a.psplit + (16 * t + m) * a.N + n], memory_order_relaxed) : 0.0f;
      float s = 0.0f;
#pragma clang loop unroll(full)
      for (uint p = 0; p < 16; p++) if (p < a.ksplit) s += v[p];
      Yt[m * a.N + n] = s;
    }
  }
  simdgroup_barrier(mem_flags::mem_device);  // every lane's acquiring RMW precedes the reset
  if (l == 0) atomic_store_explicit(&cnt[nf], 0u, memory_order_relaxed);
  for (uint t = 0; t < TILES; t++)
    if (Mloc[t]) {
      const uint live = a.rm ? rmask[tg.z * 4 + t] : 0xFFFFu;
#if EPI == 4
      epilogue(Y + size_t(t) * 16 * a.N, a.N, nb, Mloc[t], row0 + 16 * t, l, E0, E1, E2, a, live, e3_buf, e4_buf, e5_buf);
#else
      epilogue(Y + size_t(t) * 16 * a.N, a.N, nb, Mloc[t], row0 + 16 * t, l, E0, E1, E2, a, live);
#endif
    }
}

// mmp: prefill twin of mm2 for dense tensors (XMODE 0). A threadgroup owns one 32-column
// fragment and 8 row blocks (128 rows), one per simdgroup. Each k-step's weight operand is
// unpacked once into threadgroup memory and shared by the 8 simdgroups (mm2 unpacks it once
// per 16 rows); each simdgroup walks the whole K range itself and replays mm2's reductions
// exactly: k-steps into c, c scaled into acc per group, slices summed from 0.0f in order,
// splits summed from 0.0f in order. So every output bit equals mm2's, with no split-K
// partials in memory.
#ifndef GROUPED
kernel void mmp(device const WLOAD* W [[buffer(0)]], device const half* S [[buffer(1)]],
                device const uint4* X [[buffer(2)]], device float* P [[buffer(3)]],
                device float* Y [[buffer(4)]], constant MmArgs& a [[buffer(5)]],
                device const uint* slots [[buffer(6)]], device const uint* nslots [[buffer(7)]],
                device atomic_uint* cnt [[buffer(8)]],
                device bfloat* E0 [[buffer(9)]], device const float* E1 [[buffer(10)]], device float* E2 [[buffer(11)]],
                uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                ushort l [[thread_index_in_simdgroup]]) {
#ifndef MMP_CH
#define MMP_CH 8
#endif
#ifndef MMP_NF
#define MMP_NF 1
#endif
#ifndef MMP_FAST
#define MMP_FAST 0
#endif
#if MMP_NF == 2 && MMP_FAST == 1
  // 2x N-fragments per threadgroup (128 rows x 64 cols, 8 simdgroups) without split-K replay:
  // only c0/c1 + acc0/acc1 (64 floats/thread, matching baseline NF=1 register pressure), Wt = 16 KB
  // (4 k-steps x 2 fragments unpacked in parallel by sg 0..3 and sg 4..7), halving X device loads.
  constexpr uint CH = 8;
  constexpr uint KC = 4;
  threadgroup bfloat4 Wt[2][KC][8][32];
  const uint N32 = a.N / 32;
  const uint nf0 = tg.x * 2;
  const bool has1 = (nf0 + 1) < N32;
  const uint rb = tg.y * CH + sg;
  const uint KS = a.K / 32, G = a.K / (WBITS == 40 ? 16 : GS);
  const uint KPG = (WBITS == 40) ? 1u : (GS / 32u);
  const bool live = rb * 16 < a.M;
  const uint Mloc = live ? min(16u, a.M - rb * 16) : 0;
  const uint row0 = rb * 16;
  X += size_t(rb) * KS * 64;
  Y += size_t(rb) * 16 * a.N;
  constexpr auto desc =
      matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto xa = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto c0 = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), float>();
  auto c1 = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), float>();
  const ushort n0 = 4 * (l & 1) + 8 * ((l >> 3) & 1), m0 = ((l >> 1) & 3) + 4 * (l >> 4);
  const uint uf = sg >> 2, uk = sg & 3u;
  device const WLOAD* Wpu = W + (size_t(nf0 + (uf && has1 ? 1u : 0u)) * KS * 32 + l) * WV;
  device const half* Sp0 = S + size_t(nf0) * G * 32;
  device const half* Sp1 = has1 ? S + size_t(nf0 + 1) * G * 32 : Sp0;
  float acc0[16], acc1[16];
  for (int i = 0; i < 16; i++) { acc0[i] = 0.0f; acc1[i] = 0.0f; c0[i] = 0.0f; c1[i] = 0.0f; }
  for (uint kc = 0; kc < KS; kc += KC) {
    const uint nch = min(KC, KS - kc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (uk < nch && (uf == 0 || has1)) {
      WLOAD v[WV];
      for (int q = 0; q < WV; q++) v[q] = Wpu[size_t(kc + uk) * 32 * WV + q];
      bfloat e[32];
#if WBITS == 40
      device const half* Spu = uf ? Sp1 : Sp0;
      const uint ks_u = kc + uk;
      unpack_w(v, e, Spu + (2 * ks_u) * 32, Spu + (2 * ks_u + 1) * 32, m0);
#else
      unpack_w(v, e);
#endif
      for (int q = 0; q < 8; q++)
        Wt[uf][uk][q][l] = bfloat4(e[4 * q], e[4 * q + 1], e[4 * q + 2], e[4 * q + 3]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live) continue;
    for (uint u = 0; u < nch; u++) {
      const uint ks = kc + u;
      const device bfloat4* xp4 = (const device bfloat4*)(X + (ks * 32 + l) * 2);
      for (int q = 0; q < 4; q++) {
        const bfloat4 xv = xp4[q];
        xa[4 * q] = xv.x; xa[4 * q + 1] = xv.y; xa[4 * q + 2] = xv.z; xa[4 * q + 3] = xv.w;
      }
      for (int q = 0; q < 8; q++) {
        const bfloat4 wv = Wt[0][u][q][l];
        wb[4 * q] = wv.x; wb[4 * q + 1] = wv.y; wb[4 * q + 2] = wv.z; wb[4 * q + 3] = wv.w;
      }
      op.run(xa, wb, c0);
      if (has1) {
        for (int q = 0; q < 8; q++) {
          const bfloat4 wv = Wt[1][u][q][l];
          wb[4 * q] = wv.x; wb[4 * q + 1] = wv.y; wb[4 * q + 2] = wv.z; wb[4 * q + 3] = wv.w;
        }
        op.run(xa, wb, c1);
      }
#if WBITS != 40
      if ((ks + 1) % KPG == 0) {
        const uint gi = ks / KPG;
        const device half* sp0 = Sp0 + gi * 32;
        float s8[8];
        for (int q4 = 0; q4 < 4; q4++) { s8[q4] = float(sp0[n0 + q4]); s8[4 + q4] = float(sp0[n0 + 16 + q4]); }
        for (int i = 0; i < 16; i++) { acc0[i] = fma(c0[i], s8[(i & 3) + 4 * (i >> 3)], acc0[i]); c0[i] = 0.0f; }
        if (has1) {
          const device half* sp1 = Sp1 + gi * 32;
          for (int q4 = 0; q4 < 4; q4++) { s8[q4] = float(sp1[n0 + q4]); s8[4 + q4] = float(sp1[n0 + 16 + q4]); }
          for (int i = 0; i < 16; i++) { acc1[i] = fma(c1[i], s8[(i & 3) + 4 * (i >> 3)], acc1[i]); c1[i] = 0.0f; }
        }
      }
#endif
    }
  }
  if (!live) return;
#if WBITS == 40
  for (int i = 0; i < 16; i++) { acc0[i] = c0[i]; if (has1) acc1[i] = c1[i]; }
#endif
  const uint nb0 = nf0 * 32;
#if EPI == 3
  epi3_regs(acc0, nb0, Mloc, row0, n0, m0, E0, a);
  if (has1) epi3_regs(acc1, nb0 + 32, Mloc, row0, n0, m0, E0, a);
  return;
#endif
  for (int i = 0; i < 16; i++) {
    const uint m = m0 + 8 * ((i >> 2) & 1), n = nb0 + n0 + (i & 3) + 16 * (i >> 3);
    if (m < Mloc) Y[m * a.N + n] = acc0[i];
  }
  epilogue(Y, a.N, nb0, Mloc, row0, l, E0, E1, E2, a);
  if (has1) {
    const uint nb1 = nb0 + 32;
    for (int i = 0; i < 16; i++) {
      const uint m = m0 + 8 * ((i >> 2) & 1), n = nb1 + n0 + (i & 3) + 16 * (i >> 3);
      if (m < Mloc) Y[m * a.N + n] = acc1[i];
    }
    epilogue(Y, a.N, nb1, Mloc, row0, l, E0, E1, E2, a);
  }
#else
  constexpr uint CH = MMP_CH;               // k-steps per chunk = simdgroups per threadgroup
  threadgroup bfloat Wt[CH][4][32][8];      // [k-step][8-element quarter][lane][8]
  const uint nf = tg.x, rb = tg.y * CH + sg;
  const uint KS = a.K / 32, G = a.K / (WBITS == 40 ? 16 : GS);
  const uint KPG = (WBITS == 40) ? 1u : (GS / 32u);
#if MMP_FAST == 0
  const uint sks = G / (a.ksplit * a.sgs) * KPG;  // k-steps per mm2 slice
#endif
  const bool live = rb * 16 < a.M;
  const uint Mloc = live ? min(16u, a.M - rb * 16) : 0;
  const uint row0 = rb * 16;
  X += size_t(rb) * KS * 64;
  Y += size_t(rb) * 16 * a.N;
  constexpr auto desc =
      matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto xa = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto c = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), float>();
  const ushort n0 = 4 * (l & 1) + 8 * ((l >> 3) & 1), m0 = ((l >> 1) & 3) + 4 * (l >> 4);
  device const WLOAD* Wp = W + (size_t(nf) * KS * 32 + l) * WV;
  device const half* Sp = S + size_t(nf) * G * 32;
#if MMP_FAST == 1
  float acc[16];
  for (int i = 0; i < 16; i++) { acc[i] = 0.0f; c[i] = 0.0f; }
#else
  float acc[16], r1[16], r2[16];
  for (int i = 0; i < 16; i++) { acc[i] = 0.0f; r1[i] = 0.0f; r2[i] = 0.0f; c[i] = 0.0f; }
#endif
  for (uint kc = 0; kc < KS; kc += CH) {
    const uint nch = min(CH, KS - kc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < nch) {
      WLOAD v[WV];
      for (int q = 0; q < WV; q++) v[q] = Wp[size_t(kc + sg) * 32 * WV + q];
      bfloat e[32];
      unpack_w(v, e);
      for (int q = 0; q < 4; q++)
        for (int i = 0; i < 8; i++) Wt[sg][q][l][i] = e[q * 8 + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live) continue;
    for (uint u = 0; u < nch; u++) {
      const uint ks = kc + u;
      const device bfloat* xp = (const device bfloat*)(X + (ks * 32 + l) * 2);
      for (int i = 0; i < 16; i++) xa[i] = xp[i];
      for (int q = 0; q < 4; q++)
        for (int i = 0; i < 8; i++) wb[q * 8 + i] = Wt[u][q][l][i];
      op.run(xa, wb, c);
      if ((ks + 1) % KPG == 0) {
        const device half* sp = Sp + (ks / KPG) * 32;
        float s8[8];
        for (int q4 = 0; q4 < 4; q4++) { s8[q4] = float(sp[n0 + q4]); s8[4 + q4] = float(sp[n0 + 16 + q4]); }
        for (int i = 0; i < 16; i++) { acc[i] = fma(c[i], s8[(i & 3) + 4 * (i >> 3)], acc[i]); c[i] = 0.0f; }
      }
#if MMP_FAST == 0
      if ((ks + 1) % sks == 0) {  // end of mm2 slice q = (split, sl)
        const uint q = ks / sks, sl = q % a.sgs, split = q / a.sgs;
        if (a.sgs > 1) {
          if (sl == 0) for (int i = 0; i < 16; i++) r1[i] = 0.0f;
          for (int i = 0; i < 16; i++) r1[i] += acc[i];
        } else {
          for (int i = 0; i < 16; i++) r1[i] = acc[i];
        }
        for (int i = 0; i < 16; i++) acc[i] = 0.0f;
        if (sl == a.sgs - 1) {
          if (a.ksplit > 1) {
            if (split == 0) for (int i = 0; i < 16; i++) r2[i] = 0.0f;
            for (int i = 0; i < 16; i++) r2[i] += r1[i];
          } else {
            for (int i = 0; i < 16; i++) r2[i] = r1[i];
          }
        }
      }
#endif
    }
  }
  if (!live) return;
  const uint nb = nf * 32;
#if MMP_FAST == 1
#define MMP_OUT acc
#else
#define MMP_OUT r2
#endif
#if EPI == 3
  epi3_regs(MMP_OUT, nb, Mloc, row0, n0, m0, E0, a);
  return;
#endif
  for (int i = 0; i < 16; i++) {
    const uint m = m0 + 8 * ((i >> 2) & 1), n = nb + n0 + (i & 3) + 16 * (i >> 3);
    if (m < Mloc) Y[m * a.N + n] = MMP_OUT[i];
  }
  epilogue(Y, a.N, nb, Mloc, row0, l, E0, E1, E2, a);
#undef MMP_OUT
#endif
}
#endif

// mmpg: mmp for grouped weights (experts, or one weight per attention head) over row lists, for
// prefill. Slot z (weight slots[z]) multiplies rows i < count of its list: source row = list[off + i]
// (ldx 0), i of a per-slot input X + z * xstride (ldx 1, count M, outputs at z * M + i), or off + i
// (ldx 2). Outputs are compact: Y row = off + i (ldx 0/2) or z * M + i (ldx 1); EPI 3 writes its frag
// at the same row. Same reduction order as mm2 GROUPED, so bits equal the decode path's.
#ifdef GROUPED
kernel void mmpg(device const WLOAD* W [[buffer(0)]], device const half* S [[buffer(1)]],
                 device const uint4* X [[buffer(2)]], device float* P [[buffer(3)]],
                 device float* Y [[buffer(4)]], constant MmArgs& a [[buffer(5)]],
                 device const uint* slots [[buffer(6)]], device const uint* nslots [[buffer(7)]],
                 device atomic_uint* cnt [[buffer(8)]],
                 device bfloat* E0 [[buffer(9)]], device const float* E1 [[buffer(10)]], device float* E2 [[buffer(11)]],
                 device const uint* roff [[buffer(12)]], device const uint* rcnt [[buffer(13)]],
                 device const uint* rlist [[buffer(14)]], device const uint* wtile [[buffer(15)]],
                 uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                 ushort l [[thread_index_in_simdgroup]]) {
#ifndef MMP_CH
#define MMP_CH 8
#endif
#ifndef MMP_NF
#define MMP_NF 1
#endif
#ifndef MMP_FAST
#define MMP_FAST 0
#endif
#if MMP_NF == 2 && MMP_FAST == 1
  // 2x N-fragments per threadgroup (CH*16 rows x 64 cols, CH simdgroups) without split-K replay:
  // KC = CH/2 k-steps x 2 fragments unpacked in parallel into Wt[2][KC][8][32] (8 KB at CH=4),
  // halving scattered X gathers and TG launch overhead for MoE prefill (moe_gu / moe_dn).
  constexpr uint CH = MMP_CH;
  constexpr uint KC = CH / 2;
  threadgroup bfloat4 Wt[2][KC][8][32];
  uint z = tg.z, ty = tg.y;
  if (a.rm) { const uint wt = wtile[tg.y]; z = wt >> 16; ty = wt & 0xFFFFu; }
  if (z >= nslots[0]) return;
  const uint count = a.ldx == 1 ? a.M : rcnt[z], off = a.ldx == 1 ? z * a.M : roff[z];
  if (ty * CH * 16 >= count) return;
  W += size_t(slots[z]) * a.wstride;
  S += size_t(slots[z]) * a.sstride;
  if (a.ldx == 1) X += size_t(z) * a.xstride;
  const uint N32 = a.N / 32;
  const uint nf0 = tg.x * 2;
  const bool has1 = (nf0 + 1) < N32;
  const uint rb = ty * CH + sg;
  const uint KS = a.K / 32, G = a.K / (WBITS == 40 ? 16 : GS);
  const uint KPG = (WBITS == 40) ? 1u : (GS / 32u);
  const bool live = rb * 16 < count;
  const uint Mloc = live ? min(16u, count - rb * 16) : 0;
  constexpr auto desc =
      matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto xa = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto c0 = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), float>();
  auto c1 = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), float>();
  const ushort n0 = 4 * (l & 1) + 8 * ((l >> 3) & 1), m0 = ((l >> 1) & 3) + 4 * (l >> 4);
#ifdef CONTIG_X
  X += size_t((a.ldx == 2 ? (off >> 4) : 0u) + rb) * KS * 64;
#else
  uint goff[2][2];
  bool gok[2];
  for (int r = 0; r < 2; r++) {
    const uint i = rb * 16 + m0 + 8 * r;
    gok[r] = i < count;
    const uint src = !gok[r] ? 0u : a.ldx == 0 ? rlist[off + i] : a.ldx == 1 ? i : off + i;
    for (int hh = 0; hh < 2; hh++) goff[hh][r] = frag_off(src, n0 + 16 * hh, a.K);
  }
#endif
  const uint uf = sg / KC, uk = sg % KC;
  device const WLOAD* Wpu = W + (size_t(nf0 + (uf && has1 ? 1u : 0u)) * KS * 32 + l) * WV;
  device const half* Sp0 = S + size_t(nf0) * G * 32;
  device const half* Sp1 = has1 ? S + size_t(nf0 + 1) * G * 32 : Sp0;
  float acc0[16], acc1[16];
  for (int i = 0; i < 16; i++) { acc0[i] = 0.0f; acc1[i] = 0.0f; c0[i] = 0.0f; c1[i] = 0.0f; }
  for (uint kc = 0; kc < KS; kc += KC) {
    const uint nch = min(KC, KS - kc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (uk < nch && (uf == 0 || has1)) {
      WLOAD v[WV];
      for (int q = 0; q < WV; q++) v[q] = Wpu[size_t(kc + uk) * 32 * WV + q];
      bfloat e[32];
#if WBITS == 40
      device const half* Spu = uf ? Sp1 : Sp0;
      const uint ks_u = kc + uk;
      unpack_w(v, e, Spu + (2 * ks_u) * 32, Spu + (2 * ks_u + 1) * 32, m0);
#else
      unpack_w(v, e);
#endif
      for (int q = 0; q < 8; q++)
        Wt[uf][uk][q][l] = bfloat4(e[4 * q], e[4 * q + 1], e[4 * q + 2], e[4 * q + 3]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live) continue;
    for (uint u = 0; u < nch; u++) {
      const uint ks = kc + u;
#ifdef CONTIG_X
      const device bfloat4* xp4 = (const device bfloat4*)(X + (ks * 32 + l) * 2);
      for (int q = 0; q < 4; q++) {
        const bfloat4 xv = xp4[q];
        xa[4 * q] = xv.x; xa[4 * q + 1] = xv.y; xa[4 * q + 2] = xv.z; xa[4 * q + 3] = xv.w;
      }
#else
      for (int hh = 0; hh < 2; hh++)
        for (int r = 0; r < 2; r++) {
          const uint2 w = gok[r] ? *(const device uint2*)((const device bfloat*)X + goff[hh][r] + ks * 512) : uint2(0);
          const bfloat4 b4 = as_type<bfloat4>(w);
          for (int e = 0; e < 4; e++) xa[hh * 8 + r * 4 + e] = b4[e];
        }
#endif
      for (int q = 0; q < 8; q++) {
        const bfloat4 wv = Wt[0][u][q][l];
        wb[4 * q] = wv.x; wb[4 * q + 1] = wv.y; wb[4 * q + 2] = wv.z; wb[4 * q + 3] = wv.w;
      }
      op.run(xa, wb, c0);
      if (has1) {
        for (int q = 0; q < 8; q++) {
          const bfloat4 wv = Wt[1][u][q][l];
          wb[4 * q] = wv.x; wb[4 * q + 1] = wv.y; wb[4 * q + 2] = wv.z; wb[4 * q + 3] = wv.w;
        }
        op.run(xa, wb, c1);
      }
#if WBITS != 40
      if ((ks + 1) % KPG == 0) {
        const uint gi = ks / KPG;
        const device half* sp0 = Sp0 + gi * 32;
        float s8[8];
        for (int q4 = 0; q4 < 4; q4++) { s8[q4] = float(sp0[n0 + q4]); s8[4 + q4] = float(sp0[n0 + 16 + q4]); }
        for (int i = 0; i < 16; i++) { acc0[i] = fma(c0[i], s8[(i & 3) + 4 * (i >> 3)], acc0[i]); c0[i] = 0.0f; }
        if (has1) {
          const device half* sp1 = Sp1 + gi * 32;
          for (int q4 = 0; q4 < 4; q4++) { s8[q4] = float(sp1[n0 + q4]); s8[4 + q4] = float(sp1[n0 + 16 + q4]); }
          for (int i = 0; i < 16; i++) { acc1[i] = fma(c1[i], s8[(i & 3) + 4 * (i >> 3)], acc1[i]); c1[i] = 0.0f; }
        }
      }
#endif
    }
  }
  if (!live) return;
#if WBITS == 40
  for (int i = 0; i < 16; i++) { acc0[i] = c0[i]; if (has1) acc1[i] = c1[i]; }
#endif
  const uint nb0 = nf0 * 32, row0 = off + rb * 16;
#if EPI == 3
  epi3_regs(acc0, nb0, Mloc, row0, n0, m0, E0, a);
  if (has1) epi3_regs(acc1, nb0 + 32, Mloc, row0, n0, m0, E0, a);
  return;
#endif
  device float* Yb = Y + size_t(row0) * a.N;
  for (int i = 0; i < 16; i++) {
    const uint m = m0 + 8 * ((i >> 2) & 1), n = nb0 + n0 + (i & 3) + 16 * (i >> 3);
    if (m < Mloc) Yb[m * a.N + n] = acc0[i];
  }
  epilogue(Yb, a.N, nb0, Mloc, row0, l, E0 - 0, E1, E2, a);
  if (has1) {
    const uint nb1 = nb0 + 32;
    for (int i = 0; i < 16; i++) {
      const uint m = m0 + 8 * ((i >> 2) & 1), n = nb1 + n0 + (i & 3) + 16 * (i >> 3);
      if (m < Mloc) Yb[m * a.N + n] = acc1[i];
    }
    epilogue(Yb, a.N, nb1, Mloc, row0, l, E0 - 0, E1, E2, a);
  }
#else
  constexpr uint CH = MMP_CH;
  threadgroup bfloat Wt[CH][4][32][8];
  uint z = tg.z, ty = tg.y;
  if (a.rm) { const uint wt = wtile[tg.y]; z = wt >> 16; ty = wt & 0xFFFFu; }
  if (z >= nslots[0]) return;
  const uint count = a.ldx == 1 ? a.M : rcnt[z], off = a.ldx == 1 ? z * a.M : roff[z];
  if (ty * CH * 16 >= count) return;
  W += size_t(slots[z]) * a.wstride;
  S += size_t(slots[z]) * a.sstride;
  if (a.ldx == 1) X += size_t(z) * a.xstride;
  const uint nf = tg.x, rb = ty * CH + sg;
  const uint KS = a.K / 32, G = a.K / (WBITS == 40 ? 16 : GS);
  const uint KPG = (WBITS == 40) ? 1u : (GS / 32u);
#ifndef MMP_FAST
#define MMP_FAST 0
#endif
#if MMP_FAST == 0
  const uint sks = G / (a.ksplit * a.sgs) * KPG;
#endif
  const bool live = rb * 16 < count;
  const uint Mloc = live ? min(16u, count - rb * 16) : 0;
  constexpr auto desc =
      matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto xa = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto c = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), float>();
  const ushort n0 = 4 * (l & 1) + 8 * ((l >> 3) & 1), m0 = ((l >> 1) & 3) + 4 * (l >> 4);
  // this lane's two source rows (m0, m0 + 8 of the block) and their frag offsets at k-step 0
  uint goff[2][2];
  bool gok[2];
  for (int r = 0; r < 2; r++) {
    const uint i = rb * 16 + m0 + 8 * r;
    gok[r] = i < count;
    const uint src = !gok[r] ? 0u : a.ldx == 0 ? rlist[off + i] : a.ldx == 1 ? i : off + i;
    for (int hh = 0; hh < 2; hh++) goff[hh][r] = frag_off(src, n0 + 16 * hh, a.K);
  }
  device const WLOAD* Wp = W + (size_t(nf) * KS * 32 + l) * WV;
  device const half* Sp = S + size_t(nf) * G * 32;
#if MMP_FAST == 1
  float acc[16];
  for (int i = 0; i < 16; i++) { acc[i] = 0.0f; c[i] = 0.0f; }
#else
  float acc[16], r1[16], r2[16];
  for (int i = 0; i < 16; i++) { acc[i] = 0.0f; r1[i] = 0.0f; r2[i] = 0.0f; c[i] = 0.0f; }
#endif
  for (uint kc = 0; kc < KS; kc += CH) {
    const uint nch = min(CH, KS - kc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < nch) {
      WLOAD v[WV];
      for (int q = 0; q < WV; q++) v[q] = Wp[size_t(kc + sg) * 32 * WV + q];
      bfloat e[32];
      unpack_w(v, e);
      for (int q = 0; q < 4; q++)
        for (int i = 0; i < 8; i++) Wt[sg][q][l][i] = e[q * 8 + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live) continue;
    for (uint u = 0; u < nch; u++) {
      const uint ks = kc + u;
      for (int hh = 0; hh < 2; hh++)
        for (int r = 0; r < 2; r++) {
          const uint2 w = gok[r] ? *(const device uint2*)((const device bfloat*)X + goff[hh][r] + ks * 512) : uint2(0);
          const bfloat4 b4 = as_type<bfloat4>(w);
          for (int e = 0; e < 4; e++) xa[hh * 8 + r * 4 + e] = b4[e];
        }
      for (int q = 0; q < 4; q++)
        for (int i = 0; i < 8; i++) wb[q * 8 + i] = Wt[u][q][l][i];
      op.run(xa, wb, c);
      if ((ks + 1) % KPG == 0) {
        const device half* sp = Sp + (ks / KPG) * 32;
        float s8[8];
        for (int q4 = 0; q4 < 4; q4++) { s8[q4] = float(sp[n0 + q4]); s8[4 + q4] = float(sp[n0 + 16 + q4]); }
        for (int i = 0; i < 16; i++) { acc[i] = fma(c[i], s8[(i & 3) + 4 * (i >> 3)], acc[i]); c[i] = 0.0f; }
      }
#if MMP_FAST == 0
      if ((ks + 1) % sks == 0) {
        const uint q = ks / sks, sl = q % a.sgs, split = q / a.sgs;
        if (a.sgs > 1) {
          if (sl == 0) for (int i = 0; i < 16; i++) r1[i] = 0.0f;
          for (int i = 0; i < 16; i++) r1[i] += acc[i];
        } else {
          for (int i = 0; i < 16; i++) r1[i] = acc[i];
        }
        for (int i = 0; i < 16; i++) acc[i] = 0.0f;
        if (sl == a.sgs - 1) {
          if (a.ksplit > 1) {
            if (split == 0) for (int i = 0; i < 16; i++) r2[i] = 0.0f;
            for (int i = 0; i < 16; i++) r2[i] += r1[i];
          } else {
            for (int i = 0; i < 16; i++) r2[i] = r1[i];
          }
        }
      }
#endif
    }
  }
  if (!live) return;
  const uint nb = nf * 32, row0 = off + rb * 16;
#if MMP_FAST == 1
#define MMPG_OUT acc
#else
#define MMPG_OUT r2
#endif
#if EPI == 3
  epi3_regs(MMPG_OUT, nb, Mloc, row0, n0, m0, E0, a);
  return;
#endif
  device float* Yb = Y + size_t(row0) * a.N;
  for (int i = 0; i < 16; i++) {
    const uint m = m0 + 8 * ((i >> 2) & 1), n = nb + n0 + (i & 3) + 16 * (i >> 3);
    if (m < Mloc) Yb[m * a.N + n] = MMPG_OUT[i];
  }
  // epilogue rows are compact rows row0.. of E0 (EPI 3 frag of K = R), not per slot
  epilogue(Yb, a.N, nb, Mloc, row0, l, E0 - 0, E1, E2, a);
#undef MMPG_OUT
#endif
}
#endif
