// opus row ops for qwen4_exp. Every kernel processes rows independently (or, for the
// GatedDeltaNet recurrence, strictly in row order), with a fixed threadgroup shape and a
// fixed reduction order, so a row's bits never depend on how many rows share the step.
//
// Matmul inputs are written straight into the register-direct left-operand layout
// ("frag"): for row m and column k of a K-wide input, element offset
//   ((m / 16) * (K / 32) + k / 32) * 512 + lane * 16 + elem
// with lane/elem from frag_lane_elem below. All other activations are f32.
#include <metal_stdlib>
using namespace metal;
#ifndef HCT
#define HCT 256
#endif

constant float EPS = 1e-6f;

static inline uint frag_off(uint m, uint k, uint K) {
  const uint kk = k & 31, mm = m & 15;
  const uint l = ((kk >> 2) & 1) | ((mm & 3) << 1) | (((kk >> 3) & 1) << 3) | (((mm >> 2) & 1) << 4);
  const uint i = (kk & 3) | ((mm >> 3) << 2) | ((kk >> 4) << 3);
  return ((m >> 4) * (K >> 5) + (k >> 5)) * 512 + l * 16 + i;
}

static inline float silu(float x) { return x / (1.0f + exp(-x)); }
static inline float sigm(float x) { return 1.0f / (1.0f + exp(-x)); }
static inline float softplus(float x) { return x > 20.0f ? x : log(1.0f + exp(x)); }

// Deterministic threadgroup sum: simd_sum per simdgroup, then the simdgroup partials
// in index order. `tmp` needs one float per simdgroup.
static inline float tg_sum(float v, threadgroup float* tmp, ushort sg, ushort lane, ushort nsg) {
  v = simd_sum(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) tmp[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float s = 0.0f;
  for (ushort i = 0; i < nsg; i++) s += tmp[i];
  return s;
}

static inline float tg_max(float v, threadgroup float* tmp, ushort sg, ushort lane, ushort nsg) {
  v = simd_max(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) tmp[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float s = tmp[0];
  for (ushort i = 1; i < nsg; i++) s = max(s, tmp[i]);
  return s;
}

// Sum ksplit partials [split][M][N] for (m, n) in split order.
static inline float psum(device const float* y, uint split, uint M, uint N, uint m, uint n) {
  float s = 0.0f;
  for (uint p = 0; p < split; p++) s += y[(size_t(p) * M + m) * N + n];
  return s;
}

#define TGARGS                                                                     \
  uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], \
      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]], \
      ushort ntid [[threads_per_threadgroup]]

// Ride-along prefetch: threadgroups with z >= 1 of a latency-bound dispatch do none of its work;
// each thread reads 16 uint4 of an upcoming op's weights (RAW) so that op finds them in cache.
struct PfArgs { uint gx, gy, tpt, n; };  // work grid x/y, threads per threadgroup, uint4 in range
#define PF_PARAMS device const uint4* RAW [[buffer(PFB)]], device uint* pfsink [[buffer(PFB + 1)]], constant PfArgs& pfa [[buffer(PFB + 2)]]
#define PF_RIDE(G3, TID)                                                                   \
  if (G3.z >= 1) {                                                                         \
    const uint pi_ = (((G3.z - 1) * pfa.gy + G3.y) * pfa.gx + G3.x) * pfa.tpt + uint(TID); \
    const uint pb_ = pi_ * 16;                                                             \
    if (pb_ < pfa.n) {                                                                     \
      uint4 ps_ = uint4(0);                                                                \
      for (uint pj_ = 0; pj_ < 16; pj_++) ps_ += RAW[pb_ + pj_];                           \
      if (ps_.x == 0xFFFFFFFFu && ps_.y == 0x9E3779B9u) pfsink[0] = ps_.z;                 \
    }                                                                                      \
    return;                                                                                \
  }
#define PFB 20

struct HcArgs {
  uint M, H, HC;
  uint inject;          // 0 none; 1 o rows [M][H]; 2 MoE combine (experts in ascending id + shared)
  uint Nd, R, topk;     // previous hc down output [M][Nd]: inject weights at [R, R + HC)
};

// h += o (x) inj (optional); hn = groupnorm(h) * (1 + w); writes hn (f32) and its frag (bf16).
// One threadgroup (256 threads) per (row, stream): the norm groups are the streams. The
// inject weights 2 sigmoid(y / HC) and the MoE combine are computed here, with the same
// arithmetic the standalone hc_mid / moe_combine kernels used.
kernel void hc_pre(device float* h [[buffer(0)]], device const float* o [[buffer(1)]],
                   device const float* ydn [[buffer(2)]], device const float* w [[buffer(3)]],
                   device float* hn [[buffer(4)]], device bfloat* hnf [[buffer(5)]],
                   constant HcArgs& a [[buffer(6)]],
                   device const float* yd [[buffer(7)]], device const float* ysh [[buffer(8)]],
                   device const uint* topi [[buffer(9)]], device const float* topw [[buffer(10)]],
                   device const uint* slot_of [[buffer(11)]], device const float* sgate [[buffer(12)]],
                   uint3 g3 [[threadgroup_position_in_grid]],
                   ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]], PF_PARAMS) {
  threadgroup float tmp[32];
  PF_RIDE(g3, tid)
  const uint2 g = g3.xy;
  const uint row = g.x, s = g.y, W = a.H * a.HC;
  device float* hr = h + size_t(row) * W + s * a.H;
  const float iv = a.inject ? 2.0f * sigm(ydn[size_t(row) * a.Nd + a.R + s] / float(a.HC)) : 0.0f;
  uint sl[16];
  float wt[16];
  float sgt = 0.0f;
  if (a.inject == 2) {
    uint id[16];
    for (uint k = 0; k < a.topk; k++) { id[k] = topi[row * a.topk + k]; sl[k] = slot_of[row * a.topk + k]; wt[k] = topw[row * a.topk + k]; }
    for (uint i = 1; i < a.topk; i++)  // insertion sort by expert id (ids are distinct)
      for (uint j = i; j > 0 && id[j] < id[j - 1]; j--) {
        uint t = id[j]; id[j] = id[j - 1]; id[j - 1] = t;
        t = sl[j]; sl[j] = sl[j - 1]; sl[j - 1] = t;
        float f = wt[j]; wt[j] = wt[j - 1]; wt[j - 1] = f;
      }
    sgt = sgate[row];
  }
  const ushort NT = HCT;  // threads per (row, stream)
  float v[(2560 + HCT - 1) / HCT];
  float ss = 0.0f;
  for (uint j = 0; j < (a.H + NT - 1) / NT; j++) {
    const uint c = tid + NT * j;
    if (c >= a.H) break;
    float x = hr[c];
    if (a.inject) {
      float oc;
      if (a.inject == 1) {
        oc = o[size_t(row) * a.H + c];
      } else {
        oc = 0.0f;
        for (uint q = 0; q < a.topk; q++) oc += yd[(size_t(sl[q]) * a.M + row) * a.H + c] * wt[q];
        oc += sgt * ysh[size_t(row) * a.H + c];
      }
      x = fma(oc, iv, x);
      hr[c] = x;
    }
    v[j] = x;
    ss = fma(x, x, ss);
  }
  const float inv = rsqrt(tg_sum(ss, tmp, sg, lane, NT / 32) / float(a.H) + EPS);
  for (uint j = 0; j < (a.H + NT - 1) / NT; j++) {
    const uint c = tid + NT * j, e = s * a.H + c;
    if (c >= a.H) break;
    const float y = v[j] * inv * (1.0f + w[e]);
    hn[size_t(row) * W + e] = y;
    hnf[frag_off(row, e, W)] = bfloat(y);
  }
}

struct MidArgs {
  uint M, split, N;  // y partials [split][M][N]
  uint R, HC;        // low rank, streams; y[0:R] down, y[R:R+HC] inject
  uint combine;
};

// d = silu(down / HC) -> frag (K = R); inj = 2 sigmoid(inject / HC). One thread per output.
kernel void hc_mid(device const float* y [[buffer(0)]], device bfloat* df [[buffer(1)]],
                   device float* inj [[buffer(2)]], constant MidArgs& a [[buffer(3)]],
                   uint2 g [[thread_position_in_grid]]) {
  const uint row = g.y, j = g.x;
  if (j < a.R) df[frag_off(row, j, a.R)] = bfloat(silu(y[size_t(row) * a.N + j] / float(a.HC)));
  else if (a.combine && j < a.R + a.HC) inj[row * a.HC + (j - a.R)] = 2.0f * sigm(y[size_t(row) * a.N + j] / float(a.HC));
}

struct MixArgs {
  uint M, split, H, HC;  // mw partials [split][M][H*HC]
};

// x = mean_s(sigmoid(mw_s) * hn_s) -> f32 and frag (K = H). One thread per output.
kernel void hc_mix(device const float* mw [[buffer(0)]], device const float* hn [[buffer(1)]],
                   device float* x [[buffer(2)]], device bfloat* xf [[buffer(3)]],
                   constant MixArgs& a [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  const uint row = g.y, c = g.x, W = a.H * a.HC;
  if (c >= a.H) return;
  float s = 0.0f;
  for (uint t = 0; t < a.HC; t++) s += sigm(mw[size_t(row) * W + t * a.H + c]) * hn[size_t(row) * W + t * a.H + c];
  s = s / float(a.HC);
  x[size_t(row) * a.H + c] = s;
  xf[frag_off(row, c, a.H)] = bfloat(s);
}

// ------------------------------------------------------------------ GatedDeltaNet
struct GdnArgs {
  uint M, split, N;          // proj partials [split][M][N]: q | k | v | z | b | a
  uint nk, nv, dk, dv;       // 16, 48, 128, 128
  uint conv_par;             // which conv history buffer to read (writes the other)
  uint mode;                 // 0 update (outputs + state), 1 verify (outputs only), 2 commit (state only)
  uint row0;                 // lanes: row of the output frag that this dispatch's row 0 is (inputs: buffer offset)
};

// One state column of the delta rule for one row. Columns are independent; a simdgroup
// holds a column with lane l owning dk rows [4l, 4l+4) (dk == 128), so both reductions are
// one simd_sum. gdn_step and gdn_scan both go through here, so their bits match.
#ifndef GDN_NC
#define GDN_NC 4
#endif
static inline float gdn_col(thread float* st, float4 k4, float4 q4, float v, float beta, float decay) {
  float kv = 0.0f;
  for (int j = 0; j < 4; j++) {
    st[j] *= decay;
    kv = fma(st[j], k4[j], kv);
  }
  const float delta = (v - simd_sum(kv)) * beta;
  float ov = 0.0f;
  for (int j = 0; j < 4; j++) {
    st[j] = fma(k4[j], delta, st[j]);
    ov = fma(st[j], q4[j], ov);
  }
  return simd_sum(ov);
}

// One threadgroup (1024 threads) per value head. Simdgroup sg owns state columns
// [4 sg, 4 sg + 4) (gdn_col layout). Rows are processed strictly in order.
static inline void gdn_head(device const float* y, device float* convh, device const float* convw, device float* S,
                            device const float* A_log, device const float* dt_bias, device const float* nw,
                            device bfloat* of, constant GdnArgs& a, uint M, uint conv_par, uint row0, uint hv,
                            threadgroup float* qs, threadgroup float* ks, threadgroup float* vs, threadgroup float* zs,
                            threadgroup float* os, threadgroup float* tmp, threadgroup float* sc,
                            ushort t, ushort sg, ushort lane) {
  const uint C = 2 * a.nk * a.dk + a.nv * a.dv;
  const uint hk = hv / (a.nv / a.nk);
  // channels this head needs: q[hk], k[hk], v[hv]
  const uint chq = hk * a.dk, chk = a.nk * a.dk + hk * a.dk, chv = 2 * a.nk * a.dk + hv * a.dv;
  device const float* hin = convh + conv_par * 3 * C;
  device float* hout = convh + (1 - conv_par) * 3 * C;
  float st[4][4];
  device float* Sh = S + size_t(hv) * a.dk * a.dv;
  for (int cc = 0; cc < 4; cc++) {
    const float4 v = *(device const float4*)(Sh + (sg * 4 + cc) * a.dk + 4 * lane);
    for (int j = 0; j < 4; j++) st[cc][j] = v[j];
  }
  const float Aexp = exp(A_log[hv]);

  for (uint r = 0; r < M; r++) {
    // causal conv (kernel 4) over [hist0, hist1, hist2, rows 0..r]
    if (t < 384) {
      const uint which = t / 128, cc = t & 127;
      const uint ch = which == 0 ? chq + cc : which == 1 ? chk + cc : chv + cc;
      float acc = 0.0f;
      for (int j = 0; j < 4; j++) {
        const int src = int(r) - 3 + j;  // < 0: from history
        float xv = src < 0 ? hin[(3 + src) * C + ch] : psum(y, a.split, M, a.N, uint(src), ch);
        acc = fma(convw[ch * 4 + j], xv, acc);
      }
      acc = silu(acc);
      if (which == 0) qs[cc] = acc; else if (which == 1) ks[cc] = acc; else vs[cc] = acc;
    } else if (t < 512) {
      zs[t - 384] = psum(y, a.split, M, a.N, r, C + hv * a.dv + (t - 384));
    }
    if (t == 512) {
      const float bb = psum(y, a.split, M, a.N, r, C + a.nv * a.dv + hv);
      const float aa = psum(y, a.split, M, a.N, r, C + a.nv * a.dv + a.nv + hv);
      sc[0] = sigm(bb);                                  // beta
      sc[1] = exp(-Aexp * softplus(aa + dt_bias[hv]));   // decay
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // l2norm q, k (simdgroups 0..3 hold q, 4..7 hold k)
    float qq = t < 128 ? qs[t] * qs[t] : 0.0f, kk2 = t < 128 ? ks[t] * ks[t] : 0.0f;
    float nq = tg_sum(qq, tmp, sg, lane, 32);
    float nk2 = tg_sum(kk2, tmp, sg, lane, 32);
    const float iq = rsqrt(nq + 1e-6f) / sqrt(float(a.dk)), ik = rsqrt(nk2 + 1e-6f);
    const float beta = sc[0], decay = sc[1];
    float4 k4, q4;
    for (int j = 0; j < 4; j++) { k4[j] = ks[4 * lane + j] * ik; q4[j] = qs[4 * lane + j] * iq; }
    for (int cc = 0; cc < 4; cc++) {
      const float o = gdn_col(st[cc], k4, q4, vs[sg * 4 + cc], beta, decay);
      if (lane == cc) os[sg * 4 + cc] = o;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float o = t < 128 ? os[t] : 0.0f;
    // gated rmsnorm over dv
    float o2 = tg_sum(o * o, tmp, sg, lane, 32);
    if (t < 128 && a.mode != 2) {
      float v = o * rsqrt(o2 / float(a.dv) + EPS) * nw[t] * sigm(zs[t]);
      of[frag_off(row0 + r, hv * a.dv + t, a.nv * a.dv)] = bfloat(v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (a.mode == 1) return;
  for (int cc = 0; cc < 4; cc++)
    *(device float4*)(Sh + (sg * 4 + cc) * a.dk + 4 * lane) = float4(st[cc][0], st[cc][1], st[cc][2], st[cc][3]);
  // new conv history: last 3 of [hist, rows] for this head's channels
  if (t < 384) {
    const uint which = t / 128, cc = t & 127;
    const uint ch = which == 0 ? chq + cc : which == 1 ? chk + cc : chv + cc;
    if (which == 2 || hv % (a.nv / a.nk) == 0) {
      for (int j = 0; j < 3; j++) {
        const int src = int(M) - 3 + j;
        hout[j * C + ch] = src < 0 ? hin[(3 + src) * C + ch] : psum(y, a.split, M, a.N, uint(src), ch);
      }
    }
  }
}

kernel void gdn_step(device const float* y [[buffer(0)]],
                     device float* convh [[buffer(1)]],      // [2][3][C] conv input history
                     device const float* convw [[buffer(2)]],// [C][4]
                     device float* S [[buffer(3)]],          // [nv][dv][dk] (column-major per head)
                     device const float* A_log [[buffer(4)]], device const float* dt_bias [[buffer(5)]],
                     device const float* nw [[buffer(6)]],   // [dv]
                     device bfloat* of [[buffer(7)]],        // frag, K = nv * dv
                     constant GdnArgs& a [[buffer(8)]],
                     uint3 g3 [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]], PF_PARAMS) {
  PF_RIDE(g3, t)
  threadgroup float qs[128], ks[128], vs[128], zs[128], os[128];
  threadgroup float tmp[32];
  threadgroup float sc[2];
  gdn_head(y, convh, convw, S, A_log, dt_bias, nw, of, a, a.M, a.conv_par, a.row0, g3.x, qs, ks, vs, zs, os, tmp, sc, t, sg, lane);
}

// gdn_step for every lane of a joint step in one dispatch: threadgroup (head, 0, lane). Lane k's rows
// are rows row0.. of the layer input (split 1), its state, conv history and parity from the table.
struct LaneGdn { device float* S; device float* convh; uint M, row0, conv_par, pad; };
kernel void gdn_lanes(device const float* y [[buffer(0)]], device const LaneGdn* lt [[buffer(1)]],
                      device const float* convw [[buffer(2)]],
                      device const float* A_log [[buffer(4)]], device const float* dt_bias [[buffer(5)]],
                      device const float* nw [[buffer(6)]], device bfloat* of [[buffer(7)]],
                      constant GdnArgs& a [[buffer(8)]],
                      uint3 g3 [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float qs[128], ks[128], vs[128], zs[128], os[128];
  threadgroup float tmp[32];
  threadgroup float sc[2];
  const LaneGdn L = lt[g3.z];
  gdn_head(y + size_t(L.row0) * a.N, L.convh, convw, L.S, A_log, dt_bias, nw, of, a, L.M, L.conv_par, L.row0, g3.x,
           qs, ks, vs, zs, os, tmp, sc, t, sg, lane);
}

// ------------------------------------------------------------------ embeddings / logits
struct EmbArgs { uint M, H, HC; };
kernel void embed(device const uint* ids [[buffer(0)]], device const bfloat* E [[buffer(1)]],
                  device float* h [[buffer(2)]], constant EmbArgs& a [[buffer(3)]], TGARGS) {
  const uint id = ids[row];
  for (uint e = tid; e < a.H * a.HC; e += ntid) h[size_t(row) * a.H * a.HC + e] = float(E[size_t(id) * a.H + e % a.H]);
}

struct ArgArgs { uint M, split, N; };
// argmax per row over partials; ties go to the lowest index.
kernel void argmax_rows(device const float* y [[buffer(0)]], device uint* out [[buffer(1)]],
                        device float* outv [[buffer(2)]], constant ArgArgs& a [[buffer(3)]], TGARGS,
                        device float* outm [[buffer(4)]]) {  // outm: best minus runner-up (draft confidence)
  threadgroup float tv[32];
  threadgroup uint ti[32];
  threadgroup float ts[32];
  float best = -INFINITY, sec = -INFINITY;
  uint bi = 0xFFFFFFFFu;
  for (uint n = tid; n < a.N; n += ntid) {
    float v = psum(y, a.split, a.M, a.N, row, n);
    if (v > best) { sec = best; best = v; bi = n; }
    else if (v > sec) sec = v;
  }
  for (ushort off = 16; off > 0; off >>= 1) {
    float ov = simd_shuffle_down(best, off);
    uint oi = simd_shuffle_down(bi, off);
    float os = simd_shuffle_down(sec, off);
    if (ov > best || (ov == best && oi < bi)) { sec = max(best, os); best = ov; bi = oi; }
    else sec = max(sec, ov);
  }
  if (lane == 0) { tv[sg] = best; ti[sg] = bi; ts[sg] = sec; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    for (ushort i = 1; i < ntid / 32; i++)
      if (tv[i] > best || (tv[i] == best && ti[i] < bi)) { best = tv[i]; bi = ti[i]; }
    best = tv[0]; bi = ti[0];
    for (ushort i = 1; i < ntid / 32; i++)
      if (tv[i] > best || (tv[i] == best && ti[i] < bi)) { best = tv[i]; bi = ti[i]; }
    if (bi == 0xFFFFFFFFu) bi = 0;  // no finite logit (all NaN / -inf): a valid token id, not the sentinel
    out[row] = bi;
    outv[row] = best;
    float s2 = -INFINITY;  // runner-up: the winner group's own second, every other group's best
    for (ushort i = 0; i < ntid / 32; i++) s2 = max(s2, ti[i] == bi ? ts[i] : tv[i]);
    outm[row] = best - s2;
  }
}

// ------------------------------------------------------------------ MoE
struct RouteArgs {
  uint M, split, N;        // fused partials [split][M][N]: router E | shared gate I | shared up I | shared_gate 1
  uint E, I, topk;
};

// Per row: softmax over E router logits (f32), top-k with ties to the lower expert id,
// renormalised; shared gate. One simdgroup per row (E <= 512: 16 logits per lane).
// Then the same threadgroup builds the expert plan (moe_plan's job), after a barrier.
kernel void moe_route(device const float* y [[buffer(0)]], device uint* topi [[buffer(1)]],
                      device float* topw [[buffer(2)]], device bfloat* shf [[buffer(3)]],
                      device float* sgate [[buffer(4)]], constant RouteArgs& a [[buffer(5)]],
                      device uint* slots [[buffer(6)]], device uint* nslots [[buffer(7)]],
                      device uint* slot_of [[buffer(8)]], device uint* rmask [[buffer(9)]],
                      ushort sgi [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                      ushort tid [[thread_index_in_threadgroup]], uint3 ntid3 [[threads_per_threadgroup]],
                      uint3 g3 [[threadgroup_position_in_grid]], PF_PARAMS) {
  PF_RIDE(g3, tid)
  const ushort ntid = ushort(ntid3.x);
  threadgroup uint used[512];
  threadgroup uint pos[512];
  threadgroup uint chosen[64 * 16];  // rows <= 64, topk <= 16
  // per-row top-k scratch [weights 64x16 | ids 64x16]; once the picks are made it is reused as
  // um[expert][tile]: rows routed to each expert (bit m = row 16 tile + m). Keeps the kernel at 32 KB.
  threadgroup uint scratch[2 * 64 * 16];
  threadgroup atomic_uint* um = (threadgroup atomic_uint*)scratch;
  for (uint e = tid; e < 512; e += ntid) used[e] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // shared-expert activation: every thread of the (wider) threadgroup helps
  for (uint idx = tid; idx < a.M * a.I; idx += ntid) {
    const uint rr = idx / a.I, j = idx % a.I;
    device const float* yq = y + size_t(rr) * a.N;
    shf[frag_off(rr, j, a.I)] = bfloat(silu(yq[a.E + j]) * yq[a.E + a.I + j]);
  }
  for (uint row = sgi; row < a.M; row += ntid / 32) {  // one simdgroup per row (rows past 32: again)
  device const float* yr = y + size_t(row) * a.N;
  float p[16];
  float mx = -INFINITY;
#pragma clang loop unroll(full)
  for (uint j = 0; j < 16; j++) {
    const uint e = lane + 32 * j;
    p[j] = e < a.E ? yr[e] : -INFINITY;
    mx = max(mx, p[j]);
  }
  mx = simd_max(mx);
  float den = 0.0f;
#pragma clang loop unroll(full)
  for (uint j = 0; j < 16; j++) {
    const uint e = lane + 32 * j;
    p[j] = e < a.E ? exp(p[j] - mx) : 0.0f;
    den += p[j];
  }
  den = simd_sum(den);
#pragma clang loop unroll(full)
  for (uint j = 0; j < 16; j++) p[j] = (lane + 32 * j) < a.E ? p[j] / den : -1.0f;
  threadgroup float* cw = (threadgroup float*)(scratch + row * 16);  // runtime-indexed: threadgroup memory, not spilled registers
  threadgroup uint* ci = scratch + 64 * 16 + row * 16;
  float tot = 0.0f;
  for (uint k = 0; k < a.topk; k++) {
    float bv = -2.0f;
    uint bi = 0xFFFFFFFFu;
#pragma clang loop unroll(full)
    for (uint j = 0; j < 16; j++)
      if (p[j] > bv) { bv = p[j]; bi = lane + 32 * j; }   // ascending j: first max = lowest id on this lane
    // the round's pick: largest value, lowest id among its holders (two reductions, the
    // same pick as a pairwise (value, id) tree)
    {
      const float m = simd_max(bv);
      bi = simd_min(bv == m ? bi : 0xFFFFFFFFu);
      bv = m;
    }
    cw[k] = bv;
    ci[k] = bi;
    tot += bv;
#pragma clang loop unroll(full)
    for (uint j = 0; j < 16; j++)
      if (lane + 32 * j == bi) p[j] = -2.0f;
  }
  if (lane == 0) {
    for (uint k = 0; k < a.topk; k++) {
      topi[row * a.topk + k] = ci[k];
      topw[row * a.topk + k] = cw[k] / tot;
      chosen[row * a.topk + k] = ci[k];
    }
    sgate[row] = sigm(yr[a.E + 2 * a.I]);
  }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < 512 * 4; e += ntid) atomic_store_explicit(&um[e], 0u, memory_order_relaxed);  // top-k scratch is dead
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < a.M * a.topk; i += ntid) {
    const uint r = i / a.topk;
    used[chosen[i]] = 1;
    atomic_fetch_or_explicit(&um[chosen[i] * 4 + (r >> 4)], 1u << (r & 15), memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 32) {  // ascending compaction: lane owns experts [16 lane, 16 lane + 16)
    uint c = 0;
    for (uint j = 0; j < 16; j++) c += (16 * tid + j < a.E && used[16 * tid + j]) ? 1u : 0u;
    uint n = simd_prefix_exclusive_sum(c);
    for (uint j = 0; j < 16; j++) {
      const uint e = 16 * tid + j;
      if (e >= a.E) break;
      pos[e] = n;
      if (used[e]) slots[n++] = e;
    }
    if (tid == 31) nslots[0] = n;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < a.M * a.topk; i += ntid) slot_of[i] = pos[chosen[i]];
  // rmask[slot][tile]: bit m set when row 16 tile + m is routed to the slot's expert; the
  // grouped decode matmuls store and post-process only those rows
  for (uint e = tid; e < a.E; e += ntid)
    if (used[e])
      for (uint t = 0; t < 4; t++) rmask[pos[e] * 4 + t] = atomic_load_explicit(&um[e * 4 + t], memory_order_relaxed);
}

struct PlanArgs { uint M, topk, E; };
// Unique routed experts across rows, ascending: slots[0..n), slot_of[row][k].
kernel void moe_plan(device const uint* topi [[buffer(0)]], device uint* slots [[buffer(1)]],
                     device uint* nslots [[buffer(2)]], device uint* slot_of [[buffer(3)]],
                     constant PlanArgs& a [[buffer(4)]], ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup uint used[512];
  threadgroup uint pos[512];
  if (tid < a.E) used[tid] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < a.M * a.topk) used[topi[tid]] = 1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint n = 0;
    for (uint e = 0; e < a.E; e++) { pos[e] = n; if (used[e]) slots[n++] = e; }
    nslots[0] = n;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < a.M * a.topk) slot_of[tid] = pos[topi[tid]];
}

struct ActArgs { uint M, split, N, I; };  // gate_up partials [slot][split][M][N = 2I]
kernel void moe_act(device const float* y [[buffer(0)]], device bfloat* af [[buffer(1)]],
                    device const uint* nslots [[buffer(2)]], constant ActArgs& a [[buffer(3)]],
                    uint2 g [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                    uint2 ntg [[threads_per_threadgroup]]) {
  const uint slot = g.y, row = g.x, ntid = ntg.x;
  if (slot >= nslots[0]) return;
  device const float* ys = y + size_t(slot) * a.split * a.M * a.N;
  device bfloat* as = af + size_t(slot) * 16 * a.I;
  for (uint j = tid; j < a.I; j += ntid) {
    float gg = psum(ys, a.split, a.M, a.N, row, j), u = psum(ys, a.split, a.M, a.N, row, a.I + j);
    as[frag_off(row, j, a.I)] = bfloat(silu(gg) * u);
  }
}

struct CombArgs {
  uint M, split, H, topk;  // expert down partials [slot][split][M][H]
  uint ssplit;             // shared down partials [ssplit][M][H]
};
// o[row] = sum over the row's experts in ascending id of w * y + sgate * shared (HF order).
// One thread per (row, column).
kernel void moe_combine(device const float* yd [[buffer(0)]], device const float* ysh [[buffer(1)]],
                        device const uint* topi [[buffer(2)]], device const float* topw [[buffer(3)]],
                        device const uint* slot_of [[buffer(4)]], device const float* sgate [[buffer(5)]],
                        device float* o [[buffer(6)]], constant CombArgs& a [[buffer(7)]],
                        uint3 g3 [[threadgroup_position_in_grid]], ushort lt [[thread_index_in_threadgroup]], PF_PARAMS) {
  PF_RIDE(g3, lt)
  const uint2 g = g3.xy;
  // the row's experts sorted by id once per threadgroup (ids are distinct), then one column
  // per thread: same summation order as before
  threadgroup uint sl[16];
  threadgroup float wt[16];
  const uint row = g.y, c = g.x * 256 + lt;
  if (lt == 0) {
    uint id[16];
    for (uint k = 0; k < a.topk; k++) { id[k] = topi[row * a.topk + k]; sl[k] = slot_of[row * a.topk + k]; wt[k] = topw[row * a.topk + k]; }
    for (uint i = 1; i < a.topk; i++)
      for (uint j = i; j > 0 && id[j] < id[j - 1]; j--) {
        uint t = id[j]; id[j] = id[j - 1]; id[j - 1] = t;
        t = sl[j]; sl[j] = sl[j - 1]; sl[j - 1] = t;
        float f = wt[j]; wt[j] = wt[j - 1]; wt[j - 1] = f;
      }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (c >= a.H) return;
  float s = 0.0f;
  for (uint q = 0; q < a.topk; q++) s += yd[(size_t(sl[q]) * a.M + row) * a.H + c] * wt[q];
  s += sgate[row] * ysh[size_t(row) * a.H + c];
  o[size_t(row) * a.H + c] = s;
}

// ------------------------------------------------------------------ PLE (layer 2)
struct PleArgs {
  uint M, split, N;   // key|value partials [split][M][N], N = HC*H + H
  uint H, HC;
  uint par;           // conv history buffer to read (writes the other)
  uint hist;          // history rows kept = (kernel-1)*dilation (9)
  uint dil, ksz;      // 3, 4
  uint mode;          // 0 update, 1 verify (h only), 2 commit (history only)
};

// Per row: gate each stream by its normalised key . normalised query; gv, gvn -> f32.
kernel void ple_gate(device const float* y [[buffer(0)]], device const float* h [[buffer(1)]],
                     device const float* nkw [[buffer(2)]], device const float* nqw [[buffer(3)]],
                     device const float* ncw [[buffer(4)]], device float* gv [[buffer(5)]],
                     device float* gvn [[buffer(6)]], constant PleArgs& a [[buffer(7)]], TGARGS) {
  threadgroup float tmp[32];
  const uint W = a.H * a.HC;
  device const float* hr = h + size_t(row) * W;
  float sk[4] = {0, 0, 0, 0}, sq[4] = {0, 0, 0, 0};
  for (uint e = tid; e < W; e += ntid) {
    const float kv = psum(y, a.split, a.M, a.N, row, e), qv = hr[e];
    sk[e / a.H] += kv * kv;
    sq[e / a.H] += qv * qv;
  }
  float ik[4], iq[4];
  for (uint s = 0; s < a.HC; s++) {
    ik[s] = rsqrt(tg_sum(sk[s], tmp, sg, lane, ntid / 32) / float(a.H) + EPS);
    iq[s] = rsqrt(tg_sum(sq[s], tmp, sg, lane, ntid / 32) / float(a.H) + EPS);
  }
  float dot[4] = {0, 0, 0, 0};
  for (uint e = tid; e < W; e += ntid) {
    const uint s = e / a.H;
    const float kn = psum(y, a.split, a.M, a.N, row, e) * ik[s] * (1.0f + nkw[e]);
    const float qn = hr[e] * iq[s] * (1.0f + nqw[e]);
    dot[s] = fma(kn, qn, dot[s]);
  }
  float gate[4];
  for (uint s = 0; s < a.HC; s++) {
    float gg = tg_sum(dot[s], tmp, sg, lane, ntid / 32) / sqrt(float(a.H));
    gg = sqrt(max(fabs(gg), 1e-6f)) * sign(gg);
    gate[s] = sigm(gg);
  }
  float sv[4] = {0, 0, 0, 0};
  for (uint e = tid; e < W; e += ntid) {
    const float v = gate[e / a.H] * psum(y, a.split, a.M, a.N, row, W + e % a.H);
    gv[size_t(row) * W + e] = v;
    sv[e / a.H] += v * v;
  }
  float iv[4];
  for (uint s = 0; s < a.HC; s++) iv[s] = rsqrt(tg_sum(sv[s], tmp, sg, lane, ntid / 32) / float(a.H) + EPS);
  for (uint e = tid; e < W; e += ntid) gvn[size_t(row) * W + e] = gv[size_t(row) * W + e] * iv[e / a.H] * (1.0f + ncw[e]);
}

// Per channel, rows in order: h += gv + silu(dilated causal conv over gvn history).
kernel void ple_conv(device float* h [[buffer(0)]], device const float* gv [[buffer(1)]],
                     device const float* gvn [[buffer(2)]], device float* hist [[buffer(3)]],  // [2][hist][W]
                     device const float* cw [[buffer(4)]], constant PleArgs& a [[buffer(5)]],
                     uint ch [[thread_position_in_grid]]) {
  const uint W = a.H * a.HC;
  if (ch >= W) return;
  device const float* hin = hist + a.par * a.hist * W;
  device float* hout = hist + (1 - a.par) * a.hist * W;
  for (uint r = 0; r < a.M && a.mode != 2; r++) {
    float acc = 0.0f;
    for (uint j = 0; j < a.ksz; j++) {
      const int src = int(r) - int((a.ksz - 1 - j) * a.dil);
      const float xv = src < 0 ? hin[(int(a.hist) + src) * W + ch] : gvn[size_t(src) * W + ch];
      acc = fma(cw[ch * a.ksz + j], xv, acc);
    }
    h[size_t(r) * W + ch] += gv[size_t(r) * W + ch] + silu(acc);
  }
  if (a.mode == 1) return;
  for (uint j = 0; j < a.hist; j++) {
    const int src = int(a.M) - int(a.hist) + int(j);
    hout[j * W + ch] = src < 0 ? hin[(int(a.hist) + src) * W + ch] : gvn[size_t(src) * W + ch];
  }
}

// ------------------------------------------------------------------ MTP input fusion, copies
struct RmsArgs { uint D, K, src_stride, src_row0; };
// Per row: x * rsqrt(mean(x^2) + eps) * (1 + w) over D, written to a K-wide frag with
// element e at frag row r * (D / K) + e / K (D = K: plain; D = HC*K: one row per stream).
kernel void rms_frag(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                     device bfloat* f [[buffer(2)]], constant RmsArgs& a [[buffer(3)]], TGARGS) {
  threadgroup float tmp[32];
  device const float* xr = x + size_t(a.src_row0 + row) * a.src_stride;
  float ss = 0.0f;
  for (uint e = tid; e < a.D; e += ntid) ss = fma(xr[e], xr[e], ss);
  const float inv = rsqrt(tg_sum(ss, tmp, sg, lane, ntid / 32) / float(a.D) + EPS);
  const uint per = a.D / a.K;
  for (uint e = tid; e < a.D; e += ntid) f[frag_off(row * per + e / a.K, e % a.K, a.K)] = bfloat(xr[e] * inv * (1.0f + w[e]));
}

struct FuseArgs { uint H, HC; };
// h[r][s*H + c] = pe[r][c] + ph[r*HC + s][c]
kernel void mtp_fuse(device const float* pe [[buffer(0)]], device const float* ph [[buffer(1)]],
                     device float* h [[buffer(2)]], constant FuseArgs& a [[buffer(3)]],
                     uint2 g [[thread_position_in_grid]]) {
  const uint r = g.y, e = g.x;
  if (e >= a.H * a.HC) return;
  const uint s = e / a.H, c = e % a.H;
  h[size_t(r) * a.H * a.HC + e] = pe[size_t(r) * a.H + c] + ph[(size_t(r) * a.HC + s) * a.H + c];
}

struct CopyArgs { uint n, src0, dst0, rows; };  // floats per row, first source/dest row
kernel void copy_rows(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                      constant CopyArgs& a [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= a.n || g.y >= a.rows) return;
  dst[size_t(a.dst0 + g.y) * a.n + g.x] = src[size_t(a.src0 + g.y) * a.n + g.x];
}

// ------------------------------------------------------------------ prefill MoE (tiles)
// Routing for many rows: one simdgroup per row, same arithmetic as moe_route.
kernel void moe_route_rows(device const float* y [[buffer(0)]], device uint* topi [[buffer(1)]],
                           device float* topw [[buffer(2)]], device bfloat* shf [[buffer(3)]],
                           device float* sgate [[buffer(4)]], constant RouteArgs& a [[buffer(5)]],
                           uint tgi [[threadgroup_position_in_grid]], ushort sgi [[simdgroup_index_in_threadgroup]],
                           ushort nsg [[simdgroups_per_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  const uint row = tgi * nsg + sgi;
  if (row >= a.M) return;
  device const float* yr = y + size_t(row) * a.N;
  float p[16];
  float mx = -INFINITY;
#pragma clang loop unroll(full)
  for (uint j = 0; j < 16; j++) {
    const uint e = lane + 32 * j;
    p[j] = e < a.E ? yr[e] : -INFINITY;
    mx = max(mx, p[j]);
  }
  mx = simd_max(mx);
  float den = 0.0f;
#pragma clang loop unroll(full)
  for (uint j = 0; j < 16; j++) {
    const uint e = lane + 32 * j;
    p[j] = e < a.E ? exp(p[j] - mx) : 0.0f;
    den += p[j];
  }
  den = simd_sum(den);
#pragma clang loop unroll(full)
  for (uint j = 0; j < 16; j++) p[j] = (lane + 32 * j) < a.E ? p[j] / den : -1.0f;
  float cw[16];
  uint ci[16];
  float tot = 0.0f;
  for (uint k = 0; k < a.topk; k++) {
    float bv = -2.0f;
    uint bi = 0xFFFFFFFFu;
#pragma clang loop unroll(full)
    for (uint j = 0; j < 16; j++)
      if (p[j] > bv) { bv = p[j]; bi = lane + 32 * j; }
    for (ushort off = 16; off > 0; off >>= 1) {
      const float ov = simd_shuffle_xor(bv, off);
      const uint oi = simd_shuffle_xor(bi, off);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    cw[k] = bv;
    ci[k] = bi;
    tot += bv;
#pragma clang loop unroll(full)
    for (uint j = 0; j < 16; j++)
      if (lane + 32 * j == bi) p[j] = -2.0f;
  }
  if (lane == 0) {
    for (uint k = 0; k < a.topk; k++) {
      topi[row * a.topk + k] = ci[k];
      topw[row * a.topk + k] = cw[k] / tot;
    }
    sgate[row] = sigm(yr[a.E + 2 * a.I]);
  }
  for (uint j = lane; j < a.I; j += 32) shf[frag_off(row, j, a.I)] = bfloat(silu(yr[a.E + j]) * yr[a.E + a.I + j]);
}

struct TileArgs { uint M, topk, E, maxtiles; };
// Rows of each expert, in row order, cut into 16-row tiles (tiles ascending by expert).
// Stable counting sort with one thread per expert scanning the assignments in order, so
// the plan is identical to a serial pass. M * topk <= 5120.
kernel void moe_plan_tiles(device const uint* topi [[buffer(0)]], device uint* tile_exp [[buffer(1)]],
                           device uint* ntiles [[buffer(2)]], device uint* tile_rows [[buffer(3)]],
                           device uint* assign_pos [[buffer(4)]], constant TileArgs& a [[buffer(5)]],
                           ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup ushort tp[10240];
  threadgroup uint first_tile[513];
  const uint NA = a.M * a.topk;
  for (uint i = tid; i < NA; i += 1024) tp[i] = ushort(topi[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint c = 0;
  if (tid < a.E) {
    const threadgroup ushort4* tp4 = (const threadgroup ushort4*)tp;
    const ushort4 me = ushort4(ushort(tid));
    for (uint i = 0; i < NA / 4; i++) {
      const bool4 m = tp4[i] == me;
      c += uint(m.x) + uint(m.y) + uint(m.z) + uint(m.w);
    }
    for (uint i = NA / 4 * 4; i < NA; i++) c += tp[i] == tid ? 1u : 0u;
  }
  if (tid < a.E) first_tile[tid] = (c + 15) / 16;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint t = 0;
    for (uint e = 0; e < a.E; e++) { const uint n = first_tile[e]; first_tile[e] = t; t += n; }
    first_tile[a.E] = t;
    ntiles[0] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < a.E) {
    const uint t0 = first_tile[tid], t1 = first_tile[tid + 1];
    for (uint t = t0; t < t1; t++) {
      tile_exp[t] = tid;
      for (uint r = 0; r < 16; r++) tile_rows[t * 16 + r] = 0xFFFFFFFFu;
    }
    uint k = 0;
    for (uint i = 0; i < NA; i++)
      if (tp[i] == tid) {
        const uint slot = t0 * 16 + k++;
        tile_rows[slot] = i / a.topk;
        assign_pos[i] = slot;
      }
  }
}

struct GatherArgs { uint K; };
// tile frags (16 rows each) <- rows of the layer input frag. In the frag layout k, k+1, k+2,
// k+3 (k % 4 == 0) of a row are adjacent, so each thread moves 8 bytes.
kernel void moe_gather(device const bfloat* xf [[buffer(0)]], device bfloat* xg [[buffer(1)]],
                       device const uint* tile_rows [[buffer(2)]], device const uint* ntiles [[buffer(3)]],
                       constant GatherArgs& a [[buffer(4)]], uint3 g [[thread_position_in_grid]]) {
  const uint k = g.x * 4, i = g.y, t = g.z;
  if (t >= ntiles[0] || k >= a.K) return;
  const uint row = tile_rows[t * 16 + i];
  *(device uint2*)(xg + size_t(t) * 16 * a.K + frag_off(i, k, a.K)) =
      row == 0xFFFFFFFFu ? uint2(0) : *(device const uint2*)(xf + frag_off(row, k, a.K));
}

// o[row] = sum over the row's experts in ascending id of w * y + sgate * shared, from tiles
kernel void moe_combine_tiles(device const float* yd [[buffer(0)]], device const float* ysh [[buffer(1)]],
                              device const uint* topi [[buffer(2)]], device const float* topw [[buffer(3)]],
                              device const uint* assign_pos [[buffer(4)]], device const float* sgate [[buffer(5)]],
                              device float* o [[buffer(6)]], constant CombArgs& a [[buffer(7)]],
                              uint2 g [[threadgroup_position_in_grid]], ushort lt [[thread_index_in_threadgroup]]) {
  // the row's experts sorted by id once per threadgroup (ids are distinct), then one column
  // per thread: same summation order as before
  threadgroup uint sl[16];
  threadgroup float wt[16];
  const uint row = g.y, c = g.x * 256 + lt;
  if (lt == 0) {
    uint id[16];
    for (uint k = 0; k < a.topk; k++) { id[k] = topi[row * a.topk + k]; sl[k] = assign_pos[row * a.topk + k]; wt[k] = topw[row * a.topk + k]; }
    for (uint i = 1; i < a.topk; i++)
      for (uint j = i; j > 0 && id[j] < id[j - 1]; j--) {
        uint t = id[j]; id[j] = id[j - 1]; id[j - 1] = t;
        t = sl[j]; sl[j] = sl[j - 1]; sl[j - 1] = t;
        float f = wt[j]; wt[j] = wt[j - 1]; wt[j - 1] = f;
      }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (c >= a.H) return;
  float s = 0.0f;
  for (uint q = 0; q < a.topk; q++) s += yd[size_t(sl[q]) * a.H + c] * wt[q];
  s += sgate[row] * ysh[size_t(row) * a.H + c];
  o[size_t(row) * a.H + c] = s;
}

// ------------------------------------------------------------------ sampling
// Exact top-k / top-p / temperature sampling by Gumbel-max with noise keyed by
// (seed, predicted position, token): sample = argmax over the kept tokens of
// logit / T + Gumbel(hash(seed, pos, token)). Deterministic for a seed, and the same
// rule for verify rows and MTP drafts, so speculation stays lossless.
struct SampleArgs {
  uint M, N;
  float temp, top_p;
  uint top_k;            // <= 64
  uint seed_lo, seed_hi;
  uint pos0;             // position predicted by row 0 (row r predicts pos0 + r)
  uint mapped;           // logits index i is token idmap[i] (restricted draft head)
  uint per_row;          // settings and position per row from RowSamp[row] (joint steps); temp <= 0 keeps argmax
};
struct RowSamp { float temp, top_p; uint top_k, seed_lo, seed_hi, pos; };

static inline ulong mix64(ulong v) {
  v += 0x9E3779B97F4A7C15ul;
  v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ul;
  v = (v ^ (v >> 27)) * 0x94D049BB133111EBul;
  return v ^ (v >> 31);
}
static inline uint fkey(float f) {  // monotonic float -> uint; NaN ranks below -inf
  if (isnan(f)) return 0u;
  const uint u = as_type<uint>(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

kernel void sample_rows(device const float* lg [[buffer(0)]], device uint* out [[buffer(1)]],
                        constant SampleArgs& a [[buffer(2)]], device const uint* idmap [[buffer(3)]],
                        constant RowSamp* rs [[buffer(4)]], TGARGS) {
  float temp = a.temp, top_p = a.top_p;
  uint top_k = a.top_k, seed_lo = a.seed_lo, seed_hi = a.seed_hi, pos = a.pos0 + row;
  if (a.per_row) {
    const RowSamp s = rs[row];
    if (s.temp <= 0.0f) return;  // a greedy lane: its argmax stands (uniform per threadgroup, before any barrier)
    temp = s.temp; top_p = s.top_p; top_k = s.top_k; seed_lo = s.seed_lo; seed_hi = s.seed_hi; pos = s.pos;
  }
  threadgroup atomic_uint hist[256];
  threadgroup uint bcast[2];
  threadgroup uint cnt_above[1024], cnt_tie[1024];
  threadgroup uint cand_i[64];
  threadgroup float cand_v[64];
  device const float* r = lg + size_t(row) * a.N;
  const uint K = min(top_k, 64u);
  // exact K-th largest key: 4-pass 8-bit radix select
  uint prefix = 0, mask = 0, need = K;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (tid < 256) atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint n = tid; n < a.N; n += ntid) {
      const uint kk = fkey(r[n]);
      if ((kk & mask) == prefix) atomic_fetch_add_explicit(&hist[(kk >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      uint cum = 0, d = 0;
      for (int i = 255; i >= 0; i--) {
        const uint hc = atomic_load_explicit(&hist[i], memory_order_relaxed);
        if (cum + hc >= need) { d = uint(i); break; }
        cum += hc;
      }
      bcast[0] = d;
      bcast[1] = need - cum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix |= bcast[0] << shift;
    mask |= 255u << shift;
    need = bcast[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint thr = prefix;
  // parallel ascending compaction: keys above thr, then ties at thr lowest index first, K total
  const uint R = (a.N + ntid - 1) / ntid, n0 = tid * R, n1 = min(a.N, n0 + R);
  uint na = 0, nt = 0;
  for (uint n = n0; n < n1; n++) {
    const uint kk = fkey(r[n]);
    na += kk > thr ? 1u : 0u;
    nt += kk == thr ? 1u : 0u;
  }
  cnt_above[tid] = na;
  cnt_tie[tid] = nt;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint sa = 0, st = 0;
    for (uint i = 0; i < ntid; i++) {
      const uint a2 = cnt_above[i], t2 = cnt_tie[i];
      cnt_above[i] = sa; cnt_tie[i] = st;
      sa += a2; st += t2;
    }
    bcast[0] = sa;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint ties_needed = K - bcast[0];
  {
    uint tb = cnt_tie[tid], above = cnt_above[tid];
    // candidates go in ascending index order: above-threshold slots first, then ties
    for (uint n = n0; n < n1; n++) {
      const uint kk = fkey(r[n]);
      if (kk > thr) { cand_i[above] = n; cand_v[above] = r[n]; above++; }
      else if (kk == thr && tb < ties_needed) { cand_i[bcast[0] + tb] = n; cand_v[bcast[0] + tb] = r[n]; tb++; }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    const uint n = K;
    // sort by logit desc, ties by index asc (insertion; n <= 64)
    for (uint i = 1; i < n; i++)
      for (uint j = i; j > 0 && (cand_v[j] > cand_v[j - 1] || (cand_v[j] == cand_v[j - 1] && cand_i[j] < cand_i[j - 1])); j--) {
        float tv = cand_v[j]; cand_v[j] = cand_v[j - 1]; cand_v[j - 1] = tv;
        uint ti = cand_i[j]; cand_i[j] = cand_i[j - 1]; cand_i[j - 1] = ti;
      }
    const float mx = cand_v[0] / temp;
    float den = 0.0f;
    for (uint i = 0; i < n; i++) den += exp(cand_v[i] / temp - mx);
    float cum = 0.0f;
    uint keep = n;
    for (uint i = 0; i < n; i++) {
      cum += exp(cand_v[i] / temp - mx) / den;
      if (cum >= top_p) { keep = i + 1; break; }
    }
    const ulong seed = (ulong(seed_hi) << 32) | seed_lo;
    const ulong p = pos;
    float best = -INFINITY;
    uint bi = cand_i[0];
    for (uint i = 0; i < keep; i++) {
      const uint tokid = a.mapped ? idmap[cand_i[i]] : cand_i[i];
      const ulong h = mix64(seed ^ mix64(p * 0x9E3779B97F4A7C15ul ^ (ulong(tokid) << 1)));
      // 23 bits: x + 0.5 is exact, so u stays inside (0, 1) (24 bits rounded 2^24 - 0.5 up to u = 1, g = +inf)
      const float u = (float((h >> 41) & 0x7FFFFFul) + 0.5f) / 8388608.0f;
      const float g = -log(-log(u));
      const float sc = cand_v[i] / temp + g;
      if (sc > best) { best = sc; bi = cand_i[i]; }
    }
    out[row] = bi;
  }
}

// ------------------------------------------------------------------ GatedDeltaNet, split
// Same arithmetic as gdn_step, in three passes: gdn_pre (row-parallel: conv, silu, l2 norms,
// beta, decay), gdn_scan (per head, rows in order: the state recurrence only) and gdn_post
// (row-parallel: gated rmsnorm). Reductions keep gdn_step's simdgroup trees, so the bits match.
kernel void gdn_pre(device const float* y [[buffer(0)]], device const float* convh [[buffer(1)]],
                    device const float* convw [[buffer(2)]], device const float* A_log [[buffer(3)]],
                    device const float* dt_bias [[buffer(4)]], device float* qn [[buffer(5)]],
                    device float* kn [[buffer(6)]], device float* vv [[buffer(7)]], device float* bd [[buffer(8)]],
                    constant GdnArgs& a [[buffer(9)]], uint2 g [[threadgroup_position_in_grid]],
                    ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                    ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float tmp[32];
  const uint r = g.x, hv = g.y;
  const uint C = 2 * a.nk * a.dk + a.nv * a.dv;
  const uint hk = hv / (a.nv / a.nk);
  const uint chq = hk * a.dk, chk = a.nk * a.dk + hk * a.dk, chv = 2 * a.nk * a.dk + hv * a.dv;
  device const float* hin = convh + a.conv_par * 3 * C;
  float o3[3];
  for (uint which = 0; which < 3; which++) {
    const uint ch = which == 0 ? chq + t : which == 1 ? chk + t : chv + t;
    float acc = 0.0f;
    for (int j = 0; j < 4; j++) {
      const int src = int(r) - 3 + j;
      float xv = src < 0 ? hin[(3 + src) * C + ch] : psum(y, a.split, a.M, a.N, uint(src), ch);
      acc = fma(convw[ch * 4 + j], xv, acc);
    }
    o3[which] = silu(acc);
  }
  // gdn_step's tg_sum over 32 simdgroups where only simdgroups 0..3 are non-zero
  float pq = simd_sum(o3[0] * o3[0]), pk = simd_sum(o3[1] * o3[1]);
  if (lane == 0) tmp[sg] = pq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float nq = 0.0f;
  for (ushort i = 0; i < 4; i++) nq += tmp[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) tmp[sg] = pk;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float nk2 = 0.0f;
  for (ushort i = 0; i < 4; i++) nk2 += tmp[i];
  const float iq = rsqrt(nq + 1e-6f) / sqrt(float(a.dk)), ik = rsqrt(nk2 + 1e-6f);
  const size_t o = (size_t(r) * a.nv + hv) * a.dv + t;
  qn[o] = o3[0] * iq;
  kn[o] = o3[1] * ik;
  vv[o] = o3[2];
  if (t == 0) {
    const float bb = psum(y, a.split, a.M, a.N, r, C + a.nv * a.dv + hv);
    const float aa = psum(y, a.split, a.M, a.N, r, C + a.nv * a.dv + a.nv + hv);
    bd[(size_t(r) * a.nv + hv) * 2] = sigm(bb);
    bd[(size_t(r) * a.nv + hv) * 2 + 1] = exp(-exp(A_log[hv]) * softplus(aa + dt_bias[hv]));
  }
}

// Threadgroup (head hv, block of 4 GDN_NC columns); simdgroup sg takes GDN_NC columns. No barriers:
// columns are independent, so the row loop is one simdgroup's dependency chain.
kernel void gdn_scan(device const float* y [[buffer(0)]], device float* convh [[buffer(1)]],
                     device float* S [[buffer(2)]], device const float* qn [[buffer(3)]],
                     device const float* kn [[buffer(4)]], device const float* vv [[buffer(5)]],
                     device const float* bd [[buffer(6)]], device float* oraw [[buffer(7)]],
                     constant GdnArgs& a [[buffer(8)]], uint2 g [[threadgroup_position_in_grid]],
                     ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                     ushort lane [[thread_index_in_simdgroup]]) {
  const uint C = 2 * a.nk * a.dk + a.nv * a.dv;
  const uint hv = g.x, c0 = (g.y * 4 + sg) * GDN_NC;
  float st[GDN_NC][4];
  device float* Sh = S + size_t(hv) * a.dk * a.dv;
  for (int cc = 0; cc < GDN_NC; cc++) {
    const float4 v = *(device const float4*)(Sh + (c0 + cc) * a.dk + 4 * lane);
    for (int j = 0; j < 4; j++) st[cc][j] = v[j];
  }
  for (uint r = 0; r < a.M; r++) {
    const size_t base = (size_t(r) * a.nv + hv) * a.dv;
    const float beta = bd[(size_t(r) * a.nv + hv) * 2], decay = bd[(size_t(r) * a.nv + hv) * 2 + 1];
    const float4 k4 = *(device const float4*)(kn + base + 4 * lane), q4 = *(device const float4*)(qn + base + 4 * lane);
    const float vl = lane < GDN_NC ? vv[base + c0 + lane] : 0.0f;
    float mine = 0.0f;
    for (int cc = 0; cc < GDN_NC; cc++) {
      const float o = gdn_col(st[cc], k4, q4, simd_shuffle(vl, ushort(cc)), beta, decay);
      if (lane == cc) mine = o;
    }
    if (lane < GDN_NC && a.mode != 2) oraw[base + c0 + lane] = mine;
  }
  if (a.mode == 1) return;
  for (int cc = 0; cc < GDN_NC; cc++)
    *(device float4*)(Sh + (c0 + cc) * a.dk + 4 * lane) = float4(st[cc][0], st[cc][1], st[cc][2], st[cc][3]);
  if (g.y == 0) {
    const uint hk = hv / (a.nv / a.nk);
    const uint chq = hk * a.dk, chk = a.nk * a.dk + hk * a.dk, chv = 2 * a.nk * a.dk + hv * a.dv;
    device const float* hin = convh + a.conv_par * 3 * C;
    device float* hout = convh + (1 - a.conv_par) * 3 * C;
    for (uint which = 0; which < 3; which++) {
      const uint ch = which == 0 ? chq + t : which == 1 ? chk + t : chv + t;
      if (which == 2 || hv % (a.nv / a.nk) == 0) {
        for (int j = 0; j < 3; j++) {
          const int src = int(a.M) - 3 + j;
          hout[j * C + ch] = src < 0 ? hin[(3 + src) * C + ch] : psum(y, a.split, a.M, a.N, uint(src), ch);
        }
      }
    }
  }
}

kernel void gdn_post(device const float* y [[buffer(0)]], device const float* oraw [[buffer(1)]],
                     device const float* nw [[buffer(2)]], device bfloat* of [[buffer(3)]],
                     constant GdnArgs& a [[buffer(4)]], uint2 g [[threadgroup_position_in_grid]],
                     ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                     ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float tmp[4];
  const uint r = g.x, hv = g.y;
  const uint C = 2 * a.nk * a.dk + a.nv * a.dv;
  const float o = oraw[(size_t(r) * a.nv + hv) * a.dv + t];
  const float p = simd_sum(o * o);
  if (lane == 0) tmp[sg] = p;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float o2 = 0.0f;
  for (ushort i = 0; i < 4; i++) o2 += tmp[i];
  const float z = psum(y, a.split, a.M, a.N, r, C + hv * a.dv + t);
  const float v = o * rsqrt(o2 / float(a.dv) + EPS) * nw[t] * sigm(z);
  of[frag_off(r, hv * a.dv + t, a.nv * a.dv)] = bfloat(v);
}

// argmax / sample index -> token id for the restricted draft head
kernel void map_ids(device uint* v [[buffer(0)]], device const uint* idmap [[buffer(1)]],
                    constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) v[i] = idmap[v[i]];
}

// Parallel tile plan. Placement of a row inside its expert's tiles does not affect any
// result (rows are independent through the expert matmuls and the combine sums each row's
// experts in id order), so slots are taken with atomics.
kernel void plan_count(device const uint* topi [[buffer(0)]], device atomic_uint* cnt [[buffer(1)]],
                       constant TileArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < a.M * a.topk) atomic_fetch_add_explicit(&cnt[topi[i]], 1u, memory_order_relaxed);
}
kernel void plan_prefix(device atomic_uint* cnt [[buffer(0)]], device uint* first_tile [[buffer(1)]],
                        device uint* tile_exp [[buffer(2)]], device uint* ntiles [[buffer(3)]],
                        device uint* tile_rows [[buffer(4)]], constant TileArgs& a [[buffer(5)]],
                        ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup uint nt[512], ft[513];
  if (tid < a.E) nt[tid] = (atomic_load_explicit(&cnt[tid], memory_order_relaxed) + 15) / 16;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint t = 0;
    for (uint e = 0; e < a.E; e++) { ft[e] = t; t += nt[e]; }
    ft[a.E] = t;
    ntiles[0] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < a.E) {
    first_tile[tid] = ft[tid];
    for (uint t = ft[tid]; t < ft[tid + 1]; t++) {
      tile_exp[t] = tid;
      for (uint r = 0; r < 16; r++) tile_rows[t * 16 + r] = 0xFFFFFFFFu;
    }
    atomic_store_explicit(&cnt[tid], 0u, memory_order_relaxed);  // reused as fill counters
  }
}
kernel void plan_scatter(device const uint* topi [[buffer(0)]], device atomic_uint* fill [[buffer(1)]],
                         device const uint* first_tile [[buffer(2)]], device uint* tile_rows [[buffer(3)]],
                         device uint* assign_pos [[buffer(4)]], constant TileArgs& a [[buffer(5)]],
                         uint i [[thread_position_in_grid]]) {
  if (i >= a.M * a.topk) return;
  const uint e = topi[i];
  const uint slot = first_tile[e] * 16 + atomic_fetch_add_explicit(&fill[e], 1u, memory_order_relaxed);
  tile_rows[slot] = i / a.topk;
  assign_pos[i] = slot;
}

kernel void zero_u32(device uint* v [[buffer(0)]], uint i [[thread_position_in_grid]]) { v[i] = 0; }

// whole words, unlike copy_rows' floats (small integers are denormal floats and may flush)
kernel void copy_u32(device const uint* src [[buffer(0)]], device uint* dst [[buffer(1)]],
                     constant uint& dst0 [[buffer(2)]], uint i [[thread_position_in_grid]]) { dst[dst0 + i] = src[i]; }
