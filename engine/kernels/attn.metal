#include <metal_simdgroup_matrix>
// QSA (indexed) attention for qwen4_exp decode/verify rows.
// attn_prep: q/k norms + partial RoPE, KV and indexer-key cache append, block keys.
// qsa_select: per row, the top `budget/ratio` complete blocks by indexer score (only when
//   more blocks exist than the budget); ties go to the lower block index.
// attn_rows: one threadgroup per (row, query head); tokens in fixed chunks, online softmax
//   in f32, partials merged in simdgroup order.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant float EPS = 1e-6f;

static inline uint frag_off(uint m, uint k, uint K) {
  const uint kk = k & 31, mm = m & 15;
  const uint l = ((kk >> 2) & 1) | ((mm & 3) << 1) | (((kk >> 3) & 1) << 3) | (((mm >> 2) & 1) << 4);
  const uint i = (kk & 3) | ((mm >> 3) << 2) | ((kk >> 4) << 3);
  return ((m >> 4) * (K >> 5) + (k >> 5)) * 512 + l * 16 + i;
}
// V cache stores each 32-dim chunk of a head as [d % 8][d / 8], so a lane's right-operand
// elements for O += P V (dims d0, d0+8, d0+16, d0+24) are one 8-byte load.
static inline uint vperm(uint d) { return (d & ~31u) + (d & 7u) * 4 + ((d >> 3) & 3u); }
static inline float sigm(float x) { return 1.0f / (1.0f + exp(-x)); }
static inline float psum(device const float* y, uint split, uint M, uint N, uint m, uint n) {
  float s = 0.0f;
  for (uint p = 0; p < split; p++) s += y[(size_t(p) * M + m) * N + n];
  return s;
}

struct AttnArgs {
  uint M, split, N;      // proj partials [split][M][N]: q+gate (nh*2hd) | k | v | indexer q (inh*ihd) | indexer k (ihd)
  uint nh, nkv, hd;      // 24, 2, 256
  uint inh, ihd;         // 4, 128
  uint rot;              // rotary dims (64)
  uint p0;               // position of row 0
  uint ratio;            // indexer compress ratio (4)
  uint cap;              // cache capacity (tokens)
  uint budget;           // indexer token budget (2048)
  uint row0;             // lanes: output-frag row of this dispatch's row 0 (attn_merge only)
  uint kv8;              // K/V caches hold int8 + a half scale per 32 dims (Q8_0); see kv8_scales
};

// kv8 layout: Kc/Vc are [cap][nkv][hd] int8, then [cap][nkv][hd/32] half scales (absmax / 127).
// A 32-dim group is one simdgroup in attn_prep and one k-step / n-chunk in attn_rows_mma.
static inline device half* kv8_scales(device void* c, constant AttnArgs& a) {
  return (device half*)((device char*)c + size_t(a.cap) * a.nkv * a.hd);
}
static inline device const half* kv8_scales(device const void* c, constant AttnArgs& a) {
  return (device const half*)((device const char*)c + size_t(a.cap) * a.nkv * a.hd);
}

// LANES: one dispatch covers the rows of every lane in a joint step. Rows are global (the step's
// scratch is laid out by row); each row finds its lane's cache and position in the lane table. The
// rope tables hold every row (M = the step's rows). Per row, the arithmetic is the per-lane dispatch's.
#ifdef LANES
struct LaneAttn { device bfloat* Kc; device bfloat* Vc; device bfloat* rawk; device float* bk; uint p0, row0, pad0, pad1; };
#define LANE_PARAMS , device const LaneAttn* lt [[buffer(20)]], device const uchar* rowlane [[buffer(21)]]
#define ROW_POS(row) (lt[rowlane[row]].p0 + (row) - lt[rowlane[row]].row0)
#else
#define LANE_PARAMS
#define ROW_POS(row) (a.p0 + (row))
#endif

// Threadgroup per row, 256 threads. rope: [M][rot] cos then [M][rot] sin (host f64 -> f32).
kernel void attn_prep(device const float* y [[buffer(0)]], device const float* rope [[buffer(1)]],
                      device const float* qn_w [[buffer(2)]], device const float* kn_w [[buffer(3)]],
                      device const float* iqn_w [[buffer(4)]], device const float* ikn_w [[buffer(5)]],
                      device float* q [[buffer(6)]],          // [M][nh][hd]
                      device float* gate [[buffer(7)]],       // [M][nh*hd]
                      device bfloat* Kc [[buffer(8)]], device bfloat* Vc [[buffer(9)]],  // [cap][nkv][hd]
                      device float* iq [[buffer(10)]],        // [M][inh][ihd]
                      device bfloat* rawk [[buffer(11)]],     // [cap][ihd]
                      device float* bk [[buffer(12)]],        // [cap/ratio][ihd]
                      device const float* rope_blk [[buffer(13)]],  // [M][2][rot] cos/sin at block start (pos-ratio+1)
                      constant AttnArgs& a [[buffer(14)]],
                      uint2 gp [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]] LANE_PARAMS) {
  threadgroup float buf[256];
  threadgroup float tmp[8];
  // one threadgroup per (row, unit): units are q heads, then kv heads, then indexer heads
  const uint row = gp.x, unit = gp.y;
  const uint pos = ROW_POS(row);
#ifdef LANES
  Kc = lt[rowlane[row]].Kc; Vc = lt[rowlane[row]].Vc; rawk = lt[rowlane[row]].rawk;
#endif
  device const float* cs = rope + row * a.rot;
  device const float* sn = rope + (a.M + row) * a.rot;
  const uint half_r = a.rot / 2;
  // one vector (len D <= 256) normalised + roped; result in buf
  auto norm_rope = [&](float v, uint D, device const float* w) {
    float ss = simd_sum(t < D ? v * v : 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) tmp[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; i++) tot += tmp[i];
    float nv = t < D ? v * rsqrt(tot / float(D) + EPS) * (1.0f + w[t]) : 0.0f;
    if (t < D) buf[t] = nv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = nv;
    if (t < a.rot) {
      const float other = t < half_r ? -buf[t + half_r] : buf[t - half_r];
      r = nv * cs[t] + other * sn[t];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
  };
  const uint ko = a.nh * 2 * a.hd, vo = ko + a.nkv * a.hd, iqo = vo + a.nkv * a.hd, iko = iqo + a.inh * a.ihd;
  if (unit < a.nh) {
    const uint h = unit, base = h * 2 * a.hd;
    float v = psum(y, a.split, a.M, a.N, row, base + t);
    float r = norm_rope(v, a.hd, qn_w);
    q[(row * a.nh + h) * a.hd + t] = r;
    gate[row * a.nh * a.hd + h * a.hd + t] = psum(y, a.split, a.M, a.N, row, base + a.hd + t);
  } else if (unit < a.nh + a.nkv) {
    const uint h = unit - a.nh;
    float v = psum(y, a.split, a.M, a.N, row, ko + h * a.hd + t);
    float r = norm_rope(v, a.hd, kn_w);
    const float vv = psum(y, a.split, a.M, a.N, row, vo + h * a.hd + t);
    if (a.kv8) {  // simdgroup sg = dims sg*32..+32 = one group (vperm stays inside its 32-dim chunk)
      const half ks = half(min(simd_max(fabs(r)) / 127.0f, 65504.0f)), vs = half(min(simd_max(fabs(vv)) / 127.0f, 65504.0f));
      const float kd = float(ks), vd = float(vs);
      const size_t base = (size_t(pos) * a.nkv + h) * a.hd;
      ((device char*)Kc)[base + t] = char(kd > 0.0f ? clamp(rint(r / kd), -127.0f, 127.0f) : 0.0f);
      ((device char*)Vc)[base + vperm(t)] = char(vd > 0.0f ? clamp(rint(vv / vd), -127.0f, 127.0f) : 0.0f);
      if (lane == 0) {
        kv8_scales(Kc, a)[(size_t(pos) * a.nkv + h) * (a.hd / 32) + sg] = ks;
        kv8_scales(Vc, a)[(size_t(pos) * a.nkv + h) * (a.hd / 32) + sg] = vs;
      }
    } else {
      Kc[(size_t(pos) * a.nkv + h) * a.hd + t] = bfloat(r);
      Vc[(size_t(pos) * a.nkv + h) * a.hd + vperm(t)] = bfloat(vv);
    }
    if (h == 0 && t < a.ihd) rawk[size_t(pos) * a.ihd + t] = bfloat(psum(y, a.split, a.M, a.N, row, iko + t));
  } else {
    const uint h = unit - a.nh - a.nkv;
    float v = t < a.ihd ? psum(y, a.split, a.M, a.N, row, iqo + h * a.ihd + t) : 0.0f;
    float r = norm_rope(v, a.ihd, iqn_w);
    if (t < a.ihd) iq[(row * a.inh + h) * a.ihd + t] = r;
  }
}

// Block keys for blocks completed by this step's rows. Separate dispatch so every row's
// raw key is in the cache first. Threadgroup per row, 128 threads.
kernel void attn_blocks(device const bfloat* rawk [[buffer(0)]], device const float* ikn_w [[buffer(1)]],
                        device float* bk [[buffer(2)]], device const float* rope_blk [[buffer(3)]],
                        constant AttnArgs& a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                        ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]] LANE_PARAMS) {
  threadgroup float buf[128];
  threadgroup float tmp[4];
  const uint pos = ROW_POS(row);
  if ((pos + 1) % a.ratio != 0) return;
#ifdef LANES
  rawk = lt[rowlane[row]].rawk; bk = lt[rowlane[row]].bk;
#endif
  const uint b = pos / a.ratio, s0 = b * a.ratio;
  float m = 0.0f;
  for (uint j = 0; j < a.ratio; j++) m += float(rawk[size_t(s0 + j) * a.ihd + t]);
  m = float(bfloat(m / float(a.ratio)));  // HF: mean in f32, cast back to the cache dtype
  float ss = simd_sum(m * m);
  if (lane == 0) tmp[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = tmp[0] + tmp[1] + tmp[2] + tmp[3];
  float nv = m * rsqrt(tot / float(a.ihd) + EPS) * (1.0f + ikn_w[t]);
  buf[t] = nv;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint half_r = a.rot / 2;
  device const float* cs = rope_blk + row * 2 * a.rot;
  device const float* sn = cs + a.rot;
  float r = nv;
  if (t < a.rot) r = nv * cs[t] + (t < half_r ? -buf[t + half_r] : buf[t - half_r]) * sn[t];
  bk[size_t(b) * a.ihd + t] = r;
}

// Block scores for rows with more complete blocks than the budget, on the matrix units:
// C[row*inh + h][b] = q[row][h] . bk[b] (8x8 fp32 tiles, k in fixed 8-chunks), then
// score = sum over h (in order) of relu(C). Every (row, block) result depends only on its own
// operands, so it is the same for any M and any chunking. Threadgroup: 4 simdgroups, each
// 32/inh query rows x 32 blocks; inh must divide 8, ihd a multiple of 8.
kernel void qsa_scores(device const float* iq [[buffer(0)]], device const float* bk [[buffer(1)]],
                       device float* scores [[buffer(2)]], constant AttnArgs& a [[buffer(3)]],
                       uint2 g [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float ct[4][16][32];
  const uint rpt = 16 / a.inh;                 // query rows per tile
  const uint r0 = g.y * rpt, b0 = (g.x * 4 + sg) * 32;
  const uint k = a.budget / a.ratio;
  const uint maxnb = (a.p0 + min(a.M, r0 + rpt)) / a.ratio;
  if (b0 >= maxnb || maxnb <= k) return;
  simdgroup_float8x8 acc[2][4];
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  device const float* A = iq + size_t(r0) * a.inh * a.ihd;
  device const float* B = bk + size_t(b0) * a.ihd;
  for (uint kk = 0; kk < a.ihd; kk += 8) {
    simdgroup_float8x8 xa[2], xb[4];
    for (int i = 0; i < 2; i++) simdgroup_load(xa[i], A + size_t(i * 8) * a.ihd + kk, a.ihd);
    for (int j = 0; j < 4; j++) simdgroup_load(xb[j], B + size_t(j * 8) * a.ihd + kk, a.ihd, ulong2(0, 0), true);
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], xa[i], xb[j], acc[i][j]);
  }
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4; j++) simdgroup_store(acc[i][j], &ct[sg][i * 8][j * 8], 32);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const uint b = b0 + lane;
  for (uint rr = 0; rr < rpt; rr++) {
    const uint row = r0 + rr;
    if (row >= a.M) break;
    const uint nb = (a.p0 + row + 1) / a.ratio;
    if (nb <= k || b >= nb) continue;
    float s = 0.0f;
    for (uint h = 0; h < a.inh; h++) s += max(ct[sg][rr * a.inh + h][lane], 0.0f);
    scores[size_t(row) * (a.cap / a.ratio) + b] = s / sqrt(float(a.ihd));
  }
}

// Pick the top budget/ratio blocks per row: exact k-th largest score by a 4-pass 8-bit radix
// select (threadgroup histograms), then ascending compaction keeping ties lowest index first.
// sel[row][0..n) ascending; nsel = 0xFFFFFFFF means "all tokens" (few enough blocks).
// Threadgroup per row, 1024 threads.
kernel void qsa_select(device const float* iq [[buffer(0)]], device const float* bk [[buffer(1)]],
                       device float* scores [[buffer(2)]], device uint* sel [[buffer(3)]],
                       device uint* nsel [[buffer(4)]], constant AttnArgs& a [[buffer(5)]],
                       uint row [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup atomic_uint hist[256];
  threadgroup uint lo_hi[2];
  threadgroup uint sgc[2][32];
  const uint pos = a.p0 + row;
  const uint nb = (pos + 1) / a.ratio, k = a.budget / a.ratio;
  if (nb <= k) { if (t == 0) nsel[row] = 0xFFFFFFFFu; return; }
  device const float* sc = scores + size_t(row) * (a.cap / a.ratio);
  uint prefix = 0, mask = 0, need = k;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint b = t; b < nb; b += 1024) {
      const uint v = as_type<uint>(sc[b]);
      if ((v & mask) == prefix) atomic_fetch_add_explicit(&hist[(v >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      // lane l owns digits 8l..8l+7: the digit where the count of larger keys first reaches need
      uint cnt[8], owned = 0;
      for (uint j = 0; j < 8; j++) {
        cnt[j] = atomic_load_explicit(&hist[8 * lane + j], memory_order_relaxed);
        owned += cnt[j];
      }
      const uint above = simd_sum(owned) - simd_prefix_inclusive_sum(owned);
      if (above < need && need <= above + owned) {
        uint cum = above;
        for (int j = 7; j >= 0; j--) {
          if (cum + cnt[j] >= need) { lo_hi[0] = 8 * lane + uint(j); lo_hi[1] = need - cum; break; }
          cum += cnt[j];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix |= lo_hi[0] << shift;
    mask |= 255u << shift;
    need = lo_hi[1];
  }
  const uint thr = prefix;
  // Parallel, ascending compaction: thread tt owns blocks [tt*R, (tt+1)*R). Everything above
  // thr is kept; ties at thr are kept lowest index first until k are selected. The counts of
  // the threads before tt: whole simdgroups from threadgroup memory, lanes by a prefix sum.
  const uint tt = uint(sg) * 32 + lane;
  const uint R = (nb + 1023) / 1024, b0 = tt * R, b1 = min(nb, b0 + R);
  uint na = 0, nt = 0;
  for (uint b = b0; b < b1; b++) {
    const uint v = as_type<uint>(sc[b]);
    na += v > thr ? 1u : 0u;
    nt += v == thr ? 1u : 0u;
  }
  const uint sga = simd_sum(na), sgt = simd_sum(nt);
  if (lane == 0) { sgc[0][sg] = sga; sgc[1][sg] = sgt; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint va = sgc[0][lane], vt = sgc[1][lane];
  const uint above_before = simd_shuffle(simd_prefix_exclusive_sum(va), sg) + simd_prefix_exclusive_sum(na);
  uint tie_before = simd_shuffle(simd_prefix_exclusive_sum(vt), sg) + simd_prefix_exclusive_sum(nt);
  const uint ties_needed = k - simd_sum(va);
  uint out = above_before + min(tie_before, ties_needed);
  for (uint b = b0; b < b1; b++) {
    const uint v = as_type<uint>(sc[b]);
    if (v > thr || (v == thr && tie_before < ties_needed)) {
      if (v == thr) tie_before++;
      sel[size_t(row) * k + out++] = b;
    }
  }
  if (t == 0) nsel[row] = k;
}

// Threadgroup per (row, kv head, split); simdgroup g handles query head kvh * G + g.
// A row's tokens are cut into S = min(8, ceil(ntok / 128)) contiguous ranges (fixed by the
// row's position alone), each walked in 16-token tiles staged once in threadgroup memory
// and shared by the G heads. Partial (max, sum, acc) per split; attn_merge combines them in
// split order.
constant uint SMAX = 16;
static inline uint nsplits(uint ntok) { return min(SMAX, (ntok + 127) / 128); }

kernel void attn_rows(device const float* q [[buffer(0)]], device const float* gate [[buffer(1)]],
                      device const bfloat* Kc [[buffer(2)]], device const bfloat* Vc [[buffer(3)]],
                      device const uint* sel [[buffer(4)]], device const uint* nsel [[buffer(5)]],
                      device float* part [[buffer(6)]], constant AttnArgs& a [[buffer(7)]],
                      uint3 g [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                      uint3 ntg [[threads_per_threadgroup]]) {
  constexpr uint TT = 16;
  threadgroup bfloat Ks[TT][256], Vs[TT][256];
  threadgroup uint toks[TT];
  const uint ntid = ntg.x;
  const uint row = g.x, kvh = g.y, sp = g.z, G = a.nh / a.nkv, h = kvh * G + sg;
  const uint pos = a.p0 + row;
  const uint k = a.budget / a.ratio;
  const bool all = nsel[row] == 0xFFFFFFFFu;
  const uint nb_sel = all ? 0 : nsel[row];
  const uint tail0 = ((pos + 1) / a.ratio) * a.ratio;
  const uint ntok = all ? pos + 1 : nb_sel * a.ratio + (pos + 1 - tail0);
  const uint S = nsplits(ntok);
  if (sp >= S) return;
  const uint per = (ntok + S - 1) / S, j_begin = sp * per, j_end = min(ntok, j_begin + per);
  const float scale = rsqrt(float(a.hd));
  float qv[8];
  for (int i = 0; i < 8; i++) qv[i] = q[(row * a.nh + h) * a.hd + lane * 8 + i];
  float m = -INFINITY, lsum = 0.0f, acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (uint j0 = j_begin; j0 < j_end; j0 += TT) {
    const uint nt = min(TT, j_end - j0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < nt) {
      const uint j = j0 + t;
      uint tok;
      if (all) tok = j;
      else if (j < nb_sel * a.ratio) tok = sel[size_t(row) * k + j / a.ratio] * a.ratio + j % a.ratio;
      else tok = tail0 + (j - nb_sel * a.ratio);
      toks[t] = tok;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = t; e < nt * (a.hd / 8); e += ntid) {
      const uint jj = e / (a.hd / 8), d8 = e % (a.hd / 8);
      const size_t base = (size_t(toks[jj]) * a.nkv + kvh) * a.hd + d8 * 8;
      *(threadgroup uint4*)&Ks[jj][d8 * 8] = *(device const uint4*)(Kc + base);
      *(threadgroup uint4*)&Vs[jj][d8 * 8] = *(device const uint4*)(Vc + base);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint jj = 0; jj < nt; jj++) {
      float d = 0.0f;
      for (int i = 0; i < 8; i++) d = fma(qv[i], float(Ks[jj][lane * 8 + i]), d);
      d = simd_sum(d) * scale;
      const float mn = max(m, d);
      const float c = exp(m - mn), p = exp(d - mn);
      lsum = lsum * c + p;
      for (int i = 0; i < 8; i++) acc[i] = acc[i] * c + p * float(Vs[jj][vperm(lane * 8 + i)]);
      m = mn;
    }
  }
  // partials: [row][h][split] -> (m, l, acc[hd])
  device float* pp = part + ((size_t(row) * a.nh + h) * SMAX + sp) * (a.hd + 2);
  if (lane == 0) { pp[0] = m; pp[1] = lsum; }
  for (int i = 0; i < 8; i++) pp[2 + lane * 8 + i] = acc[i];
}

// The same attention on the M5 matrix units (16x32x32 bf16 cooperative tensors, f32
// accumulate). Per row the GQA group's heads (padded to 16) are the M side; keys go in
// 32-key tiles. Simdgroup sg computes S over dims sg*64..+64; the four partials are summed
// in sg order in threadgroup memory, every simdgroup runs the same online softmax, and
// P stays in registers (the S destination layout is the PV left-operand layout) while sg
// accumulates O for its 64 dims. Q and P enter as bf16 hi + lo pairs, so products carry
// ~16 mantissa bits; K and V are bf16 already. Only this row's data enters its result, so
// prefill, verify and decode stay bit-identical. hd == 256, G <= 16, 128 threads.
kernel void attn_rows_mma(device const float* q [[buffer(0)]], device const float* gate [[buffer(1)]],
                          device const bfloat* Kc [[buffer(2)]], device const bfloat* Vc [[buffer(3)]],
                          device const uint* sel [[buffer(4)]], device const uint* nsel [[buffer(5)]],
                          device float* part [[buffer(6)]], constant AttnArgs& a [[buffer(7)]],
                          uint3 g [[threadgroup_position_in_grid]],
                          ushort sg [[simdgroup_index_in_threadgroup]], ushort l [[thread_index_in_simdgroup]] LANE_PARAMS) {
  constexpr uint TT = 32, HD = 256;
  threadgroup float Sp[4][16][32];
  const uint row = g.x, kvh = g.y, sp = g.z, G = a.nh / a.nkv;
  const uint pos = ROW_POS(row);
#ifdef LANES
  Kc = lt[rowlane[row]].Kc; Vc = lt[rowlane[row]].Vc;
#endif
  const uint k = a.budget / a.ratio;
  const bool all = nsel[row] == 0xFFFFFFFFu;
  const uint nb_sel = all ? 0 : nsel[row];
  const uint tail0 = ((pos + 1) / a.ratio) * a.ratio;
  const uint ntok = all ? pos + 1 : nb_sel * a.ratio + (pos + 1 - tail0);
  const uint S = nsplits(ntok);
  if (sp >= S) return;
  const uint per = (ntok + S - 1) / S, j_begin = sp * per, j_end = min(ntok, j_begin + per);
  const float scale = rsqrt(float(HD));
  auto tokof = [&](uint j) -> uint {
    if (all) return j;
    if (j < nb_sel * a.ratio) return sel[size_t(row) * k + j / a.ratio] * a.ratio + j % a.ratio;
    return tail0 + (j - nb_sel * a.ratio);
  };
  // cooperative-tensor lane maps (convert/layout.py): left/dest element i -> (m0 + 8((i>>2)&1),
  // n0 + (i&3) + 16(i>>3)); right element i -> (k: n0 + (i&3) + 16(i>>4), n: m0 + 8((i>>2)&3))
  const ushort m0 = ((l >> 1) & 3) + 4 * (l >> 4), n0 = 4 * (l & 1) + 8 * ((l >> 3) & 1);
  constexpr auto desc =
      matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
#define LEFT op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>()
  auto q0h = LEFT; auto q0l = LEFT; auto q1h = LEFT; auto q1l = LEFT; auto xa = LEFT; auto xl = LEFT;
#undef LEFT
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto cs = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), float>();
  auto c8 = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), float>();  // kv8: one group's S
  auto o0 = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), float>();
  auto o1 = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), float>();
  // Q hi/lo for this simdgroup's two 32-dim k-steps
  for (int ks = 0; ks < 2; ks++)
    for (int i = 0; i < 16; i++) {
      const uint hh = m0 + 8 * ((i >> 2) & 1), d = sg * 64 + ks * 32 + n0 + (i & 3) + 16 * (i >> 3);
      const float v = hh < G ? q[(size_t(row) * a.nh + kvh * G + hh) * HD + d] : 0.0f;
      const bfloat hi = bfloat(v), lo = bfloat(v - float(hi));
      if (ks == 0) { q0h[i] = hi; q0l[i] = lo; } else { q1h[i] = hi; q1l[i] = lo; }
    }
  for (int i = 0; i < 16; i++) { o0[i] = 0.0f; o1[i] = 0.0f; }
  float m[2] = {-INFINITY, -INFINITY}, lsum[2] = {0.0f, 0.0f};
  device const bfloat* Kh = Kc + kvh * HD;
  device const bfloat* Vh = Vc + kvh * HD;
  const size_t tstride = size_t(a.nkv) * HD;
  // kv8: int8 values enter the matrix units exactly (|v| <= 127 is exact in bf16); a key's scale multiplies its
  // group's S partial, and a value's scale multiplies P before the hi/lo split
  device const char* K8 = (device const char*)Kc + kvh * HD;
  device const char* V8 = (device const char*)Vc + kvh * HD;
  device const half* Ksc = kv8_scales(Kc, a);
  device const half* Vsc = kv8_scales(Vc, a);
  for (uint j0 = j_begin; j0 < j_end; j0 += TT) {
    // S partial over dims sg*64..+64: right operand K^T (k = dim, n = key)
    for (int i = 0; i < 16; i++) cs[i] = 0.0f;
    uint tv[8];  // this lane's keys as P / S columns: element i -> key n0 + (i&3) + 16(i>>3) = tv[(i>>3)*4 + (i&3)]
    for (int hh = 0; hh < 2; hh++)
      for (int e = 0; e < 4; e++) {
        const uint j = j0 + n0 + e + 16 * hh;
        tv[hh * 4 + e] = j < j_end ? tokof(j) : 0xFFFFFFFFu;
      }
    if (a.kv8) {
      uint tk[4];
      for (int qq = 0; qq < 4; qq++) {
        const uint j = j0 + m0 + 8 * qq;
        tk[qq] = j < j_end ? tokof(j) : 0xFFFFFFFFu;
      }
      for (int ks = 0; ks < 2; ks++) {
        const uint d = sg * 64 + ks * 32 + n0, grp = sg * 2 + ks;
        for (int qq = 0; qq < 4; qq++)
          for (int hh = 0; hh < 2; hh++) {
            uint w = 0;
            if (tk[qq] != 0xFFFFFFFFu) w = *(device const uint*)(K8 + tk[qq] * tstride + d + 16 * hh);
            const char4 c4 = as_type<char4>(w);
            for (int e = 0; e < 4; e++) wb[hh * 16 + qq * 4 + e] = bfloat(float(c4[e]));
          }
        for (int i = 0; i < 16; i++) c8[i] = 0.0f;
        if (ks == 0) { op.run(q0h, wb, c8); op.run(q0l, wb, c8); }
        else { op.run(q1h, wb, c8); op.run(q1l, wb, c8); }
        for (int i = 0; i < 16; i++) {
          const uint tt = tv[(i >> 3) * 4 + (i & 3)];
          const float sc = tt != 0xFFFFFFFFu ? float(Ksc[(size_t(tt) * a.nkv + kvh) * (HD / 32) + grp]) : 0.0f;
          cs[i] = fma(c8[i], sc, cs[i]);
        }
      }
    } else {
      uint tk[4];
      for (int qq = 0; qq < 4; qq++) {
        const uint j = j0 + m0 + 8 * qq;
        tk[qq] = j < j_end ? tokof(j) : 0xFFFFFFFFu;
      }
      for (int ks = 0; ks < 2; ks++) {
        const uint d = sg * 64 + ks * 32 + n0;
        for (int qq = 0; qq < 4; qq++)
          for (int hh = 0; hh < 2; hh++) {
            uint2 w = 0;
            if (tk[qq] != 0xFFFFFFFFu) w = *(device const uint2*)(Kh + tk[qq] * tstride + d + 16 * hh);
            const bfloat4 b4 = as_type<bfloat4>(w);
            for (int e = 0; e < 4; e++) wb[hh * 16 + qq * 4 + e] = b4[e];
          }
        if (ks == 0) { op.run(q0h, wb, cs); op.run(q0l, wb, cs); }
        else { op.run(q1h, wb, cs); op.run(q1l, wb, cs); }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // every simdgroup done reading the last tile's S
    for (int i = 0; i < 16; i++) Sp[sg][i][l] = cs[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s[16];
    for (int i = 0; i < 16; i++) {
      const uint j = j0 + n0 + (i & 3) + 16 * (i >> 3);
      s[i] = j < j_end ? (((Sp[0][i][l] + Sp[1][i][l]) + Sp[2][i][l]) + Sp[3][i][l]) * scale : -INFINITY;
    }
    // online softmax per row r (elements with (i>>2)&1 == r); a row's 32 keys sit in lanes l, l^1, l^8, l^9
    float c[2];
    for (int r = 0; r < 2; r++) {
      float mb = -INFINITY;
      for (int i = 0; i < 16; i++) if (((i >> 2) & 1) == r) mb = max(mb, s[i]);
      mb = max(mb, simd_shuffle_xor(mb, 1));
      mb = max(mb, simd_shuffle_xor(mb, 8));
      const float mn = max(m[r], mb);
      c[r] = exp(m[r] - mn);
      float bs = 0.0f;
      for (int i = 0; i < 16; i++)
        if (((i >> 2) & 1) == r) { s[i] = exp(s[i] - mn); bs += s[i]; }
      bs += simd_shuffle_xor(bs, 1);
      bs += simd_shuffle_xor(bs, 8);
      lsum[r] = lsum[r] * c[r] + bs;
      m[r] = mn;
    }
    for (int i = 0; i < 16; i++) {
      const float cc = c[(i >> 2) & 1];
      o0[i] *= cc;
      o1[i] *= cc;
    }
    // O += P V for dims sg*64 + nc*32: right operand V (k = key, n = dim), V stored vperm'd
    if (a.kv8) {
      for (int nc = 0; nc < 2; nc++) {
        const uint d = sg * 64 + nc * 32 + m0 * 4, grp = sg * 2 + nc;
        for (int i = 0; i < 16; i++) {
          const uint tt = tv[(i >> 3) * 4 + (i & 3)];
          const float ps = tt != 0xFFFFFFFFu ? s[i] * float(Vsc[(size_t(tt) * a.nkv + kvh) * (HD / 32) + grp]) : 0.0f;
          const bfloat hi = bfloat(ps);
          xa[i] = hi;
          xl[i] = bfloat(ps - float(hi));
        }
        for (int hh = 0; hh < 2; hh++)
          for (int e = 0; e < 4; e++) {
            uint w = 0;
            if (tv[hh * 4 + e] != 0xFFFFFFFFu) w = *(device const uint*)(V8 + tv[hh * 4 + e] * tstride + d);
            const char4 c4 = as_type<char4>(w);
            for (int jq = 0; jq < 4; jq++) wb[hh * 16 + jq * 4 + e] = bfloat(float(c4[jq]));
          }
        if (nc == 0) { op.run(xa, wb, o0); op.run(xl, wb, o0); }
        else { op.run(xa, wb, o1); op.run(xl, wb, o1); }
      }
      continue;
    }
    for (int i = 0; i < 16; i++) {
      const bfloat hi = bfloat(s[i]);
      xa[i] = hi;
      xl[i] = bfloat(s[i] - float(hi));
    }
    {
      for (int nc = 0; nc < 2; nc++) {
        const uint d = sg * 64 + nc * 32 + m0 * 4;
        for (int hh = 0; hh < 2; hh++)
          for (int e = 0; e < 4; e++) {
            uint2 w = 0;
            if (tv[hh * 4 + e] != 0xFFFFFFFFu) w = *(device const uint2*)(Vh + tv[hh * 4 + e] * tstride + d);
            const bfloat4 b4 = as_type<bfloat4>(w);
            for (int jq = 0; jq < 4; jq++) wb[hh * 16 + jq * 4 + e] = b4[jq];
          }
        if (nc == 0) { op.run(xa, wb, o0); op.run(xl, wb, o0); }
        else { op.run(xa, wb, o1); op.run(xl, wb, o1); }
      }
    }
  }
  // partials: [row][h][split] -> (m, l, acc[hd])
  for (int r = 0; r < 2; r++) {
    const uint hh = m0 + 8 * r;
    if (hh >= G) continue;
    device float* pp = part + ((size_t(row) * a.nh + kvh * G + hh) * SMAX + sp) * (HD + 2);
    if (sg == 0 && n0 == 0) { pp[0] = m[r]; pp[1] = lsum[r]; }
    for (int i = 0; i < 16; i++) {
      if (((i >> 2) & 1) != r) continue;
      const uint dn = n0 + (i & 3) + 16 * (i >> 3);
      pp[2 + sg * 64 + dn] = o0[i];
      pp[2 + sg * 64 + 32 + dn] = o1[i];
    }
  }
}

// Merge a row's splits in split order, apply the output gate -> frag. (row, head) per TG.
kernel void attn_merge(device const float* part [[buffer(0)]], device const float* gate [[buffer(1)]],
                       device const uint* nsel [[buffer(2)]], device bfloat* of [[buffer(3)]],
                       constant AttnArgs& a [[buffer(4)]], uint2 g [[threadgroup_position_in_grid]],
                       ushort t [[thread_index_in_threadgroup]] LANE_PARAMS) {
  const uint row = g.x, h = g.y, pos = ROW_POS(row);
  const bool all = nsel[row] == 0xFFFFFFFFu;
  const uint nb_sel = all ? 0 : nsel[row];
  const uint tail0 = ((pos + 1) / a.ratio) * a.ratio;
  const uint ntok = all ? pos + 1 : nb_sel * a.ratio + (pos + 1 - tail0);
  const uint S = nsplits(ntok);
  device const float* pp = part + (size_t(row) * a.nh + h) * SMAX * (a.hd + 2);
  float M_ = -INFINITY;
  for (uint s = 0; s < S; s++) M_ = max(M_, pp[s * (a.hd + 2)]);
  float L = 0.0f, o = 0.0f;
  for (uint s = 0; s < S; s++) {
    const float c = exp(pp[s * (a.hd + 2)] - M_);
    L += pp[s * (a.hd + 2) + 1] * c;
    o += pp[s * (a.hd + 2) + 2 + t] * c;
  }
  o = o / L * sigm(gate[row * a.nh * a.hd + h * a.hd + t]);
  of[frag_off(a.row0 + row, h * a.hd + t, a.nh * a.hd)] = bfloat(o);
}
