// hy4_vit: Native Metal 4 kernels for Hy4-flash's SigLIP2-adapted Vision Transformer + PatchMerger.
// Linear projections (patch_embed, qkv, o, fc1, fc2, merge.fc1, merge.fc2) reuse DankAttn's
// existing W8g128 mmp kernel (engine/kernels/mm2.metal) with zero extra matmul shaders.
#include <metal_stdlib>
using namespace metal;

static inline uint frag_off(uint m, uint k, uint K) {
  const uint kk = k & 31, mm = m & 15;
  const uint l = ((kk >> 2) & 1) | ((mm & 3) << 1) | (((kk >> 3) & 1) << 3) | (((mm >> 2) & 1) << 4);
  const uint i = (kk & 3) | ((mm >> 3) << 2) | ((kk >> 4) << 3);
  return ((m >> 4) * (K >> 5) + (k >> 5)) * 512 + l * 16 + i;
}

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

// Accurate erf(x) (Abramowitz & Stegun 7.1.26, max error < 1.5e-7) for exact PyTorch GELU.
static inline float vit_erf(float x) {
  const float s = x < 0.0f ? -1.0f : 1.0f;
  const float a = fabs(x);
  const float t = 1.0f / (1.0f + 0.3275911f * a);
  const float p = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
  return s * (1.0f - p * exp(-a * a));
}

static inline float vit_gelu(float x) {
  return 0.5f * x * (1.0f + vit_erf(x * 0.7071067811865475244f));
}

// 1. Pack row-major bfloat [M][K] into mmp left-operand bf16 frag layout [Mp][K] (zero-padding m >= M).
struct FragArgs { uint M, Mp, K; };
kernel void vit_to_frag(device const bfloat* src [[buffer(0)]], device bfloat* dst [[buffer(1)]],
                        constant FragArgs& a [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const uint k = g.x, m = g.y;
  if (k >= a.K || m >= a.Mp) return;
  dst[frag_off(m, k, a.K)] = m < a.M ? src[size_t(m) * a.K + k] : bfloat(0.0f);
}

// 2. Add patch_embedding bias + bilinearly interpolated 2D position_embedding (128x128 grid).
// Matches PyTorch F.interpolate(..., scale_factor=((h+0.1)/edge, (w+0.1)/edge), mode='bilinear', align_corners=False)
// including PyTorch's identity short-circuit when h == edge && w == edge.
struct PatchPosArgs { uint T, h, w, H, edge; };
kernel void vit_patch_pos(device float* x [[buffer(0)]], device const float* bias [[buffer(1)]],
                          device const bfloat* pos [[buffer(2)]], constant PatchPosArgs& a [[buffer(3)]],
                          uint2 g [[thread_position_in_grid]]) {
  const uint c = g.x, row = g.y;
  const uint N = a.h * a.w;
  if (c >= a.H || row >= a.T * N) return;
  const uint loc = row % N;
  const uint py = loc / a.w, px = loc % a.w;
  float pe;
  if (a.h == a.edge && a.w == a.edge) {
    pe = float(pos[(size_t(py) * a.edge + px) * a.H + c]);
  } else {
    const float sh = (float(a.h) + 0.1f) / float(a.edge);
    const float sw = (float(a.w) + 0.1f) / float(a.edge);
    const float sy = (a.h == a.edge) ? float(py) : clamp((float(py) + 0.5f) / sh - 0.5f, 0.0f, float(a.edge - 1));
    const float sx = (a.w == a.edge) ? float(px) : clamp((float(px) + 0.5f) / sw - 0.5f, 0.0f, float(a.edge - 1));
    const uint y0 = uint(floor(sy)), x0 = uint(floor(sx));
    const uint y1 = min(y0 + 1u, a.edge - 1u), x1 = min(x0 + 1u, a.edge - 1u);
    const float wy = sy - float(y0), wx = sx - float(x0);
    const float p00 = float(pos[(size_t(y0) * a.edge + x0) * a.H + c]);
    const float p01 = float(pos[(size_t(y0) * a.edge + x1) * a.H + c]);
    const float p10 = float(pos[(size_t(y1) * a.edge + x0) * a.H + c]);
    const float p11 = float(pos[(size_t(y1) * a.edge + x1) * a.H + c]);
    pe = (p00 * (1.0f - wx) + p01 * wx) * (1.0f - wy) + (p10 * (1.0f - wx) + p11 * wx) * wy;
  }
  x[size_t(row) * a.H + c] += bias[c] + pe;
}

// 3. LayerNorm(x, w, b, eps) writing directly to bf16 frag layout [Mp][H] (1 TG of 256 threads per row).
struct LnArgs { uint M, Mp, H; float eps; };
kernel void vit_layernorm_frag(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                               device const float* b [[buffer(2)]], device bfloat* dst [[buffer(3)]],
                               constant LnArgs& a [[buffer(4)]], uint m [[threadgroup_position_in_grid]],
                               ushort tid [[thread_index_in_threadgroup]], ushort ntg [[threads_per_threadgroup]]) {
  if (m >= a.Mp) return;
  if (m >= a.M) {
    for (uint c = tid; c < a.H; c += ntg) dst[frag_off(m, c, a.H)] = bfloat(0.0f);
    return;
  }
  threadgroup float sh[32];
  device const float* xr = x + size_t(m) * a.H;
  float sum = 0.0f;
  for (uint c = tid; c < a.H; c += ntg) sum += xr[c];
  const float mean = tg_sum(sum, sh, tid, ntg) / float(a.H);
  float var = 0.0f;
  for (uint c = tid; c < a.H; c += ntg) {
    const float d = xr[c] - mean;
    var = fma(d, d, var);
  }
  const float inv = rsqrt(tg_sum(var, sh, tid, ntg) / float(a.H) + a.eps);
  for (uint c = tid; c < a.H; c += ntg) {
    dst[frag_off(m, c, a.H)] = bfloat((xr[c] - mean) * inv * w[c] + b[c]);
  }
}

// 4. Bidirectional FlashAttention for ViT (D=72, nh=16).
// Reads QKV[M][3*H] + qkv_bias[3*H] directly (no separate Q/K/V transpose), writes output
// directly into bf16 frag layout [Mp][H] ready for V{i}.attn.o mmp.
// Grid: (ceil(N / 4), nh, T), 128 threads/TG (4 simdgroups = 4 query rows per TG, BK = 32 keys).
struct VitAttnArgs { uint T, N, Mp, H, nh, hd; float scale; };
kernel void vit_attn_fwd(device const float* qkv [[buffer(0)]], device const float* qkv_b [[buffer(1)]],
                         device bfloat* out_frag [[buffer(2)]], constant VitAttnArgs& a [[buffer(3)]],
                         uint3 tgid [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
  constexpr uint BK = 32;
  constexpr uint HD = 72;
  const ushort sg = tid >> 5;
  const ushort lane = tid & 31;
  const uint frame = tgid.z;
  const uint head = tgid.y;
  const uint q_loc = tgid.x * 4u + sg;
  const uint base_row = frame * a.N;
  const uint q_row = base_row + q_loc;
  const uint stride = 3u * a.H;
  const uint h_off = head * HD;
  const bool valid_q = q_loc < a.N;

  // Load query vector (3 floats/lane cover 72 dims: lane, lane+32, lane+64).
  float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f;
  if (valid_q) {
    device const float* qp = qkv + size_t(q_row) * stride + h_off;
    device const float* qb = qkv_b + h_off;
    q0 = qp[lane] + qb[lane];
    q1 = qp[lane + 32u] + qb[lane + 32u];
    if (lane + 64u < HD) q2 = qp[lane + 64u] + qb[lane + 64u];
  }

  threadgroup float shK[BK][HD];
  threadgroup float shV[BK][HD];

  float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f;
  float mi = -INFINITY, li = 0.0f;

  for (uint k0 = 0; k0 < a.N; k0 += BK) {
    const uint klen = min(BK, a.N - k0);
    // Cooperative load of K[klen][72] and V[klen][72] (128 threads x 18 elements each).
    for (uint idx = tid; idx < klen * HD; idx += 128u) {
      const uint kr = idx / HD, d = idx % HD;
      const size_t row_off = size_t(base_row + k0 + kr) * stride;
      shK[kr][d] = qkv[row_off + a.H + h_off + d] + qkv_b[a.H + h_off + d];
      shV[kr][d] = qkv[row_off + 2u * a.H + h_off + d] + qkv_b[2u * a.H + h_off + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid_q) {
      for (uint kr = 0; kr < klen; kr++) {
        float dot = q0 * shK[kr][lane] + q1 * shK[kr][lane + 32u];
        if (lane + 64u < HD) dot = fma(q2, shK[kr][lane + 64u], dot);
        const float s = simd_sum(dot) * a.scale;
        const float mn = max(mi, s);
        const float alpha = exp(mi - mn);
        const float p = exp(s - mn);
        li = li * alpha + p;
        o0 = fma(p, shV[kr][lane], o0 * alpha);
        o1 = fma(p, shV[kr][lane + 32u], o1 * alpha);
        if (lane + 64u < HD) o2 = fma(p, shV[kr][lane + 64u], o2 * alpha);
        mi = mn;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (valid_q) {
    const float inv_l = 1.0f / li;
    out_frag[frag_off(q_row, h_off + lane, a.H)] = bfloat(o0 * inv_l);
    out_frag[frag_off(q_row, h_off + lane + 32u, a.H)] = bfloat(o1 * inv_l);
    if (lane + 64u < HD) out_frag[frag_off(q_row, h_off + lane + 64u, a.H)] = bfloat(o2 * inv_l);
  }
  // Zero-pad tail rows (M .. Mp-1) for head 0..nh-1.
  if (frame == a.T - 1u && q_loc >= a.N && base_row + q_loc < a.Mp) {
    out_frag[frag_off(q_row, h_off + lane, a.H)] = bfloat(0.0f);
    out_frag[frag_off(q_row, h_off + lane + 32u, a.H)] = bfloat(0.0f);
    if (lane + 64u < HD) out_frag[frag_off(q_row, h_off + lane + 64u, a.H)] = bfloat(0.0f);
  }
}

// 5. Residual add + bias: x[m, c] += y[m, c] + b[c].
struct ResArgs { uint M, H; };
kernel void vit_bias_res(device float* x [[buffer(0)]], device const float* y [[buffer(1)]],
                         device const float* b [[buffer(2)]], constant ResArgs& a [[buffer(3)]],
                         uint2 g [[thread_position_in_grid]]) {
  const uint c = g.x, m = g.y;
  if (c >= a.H || m >= a.M) return;
  x[size_t(m) * a.H + c] += y[size_t(m) * a.H + c] + b[c];
}

// 6. Bias + GELU writing directly to bf16 frag layout [Mp][Npad], zeroing c >= Nreal and m >= M.
struct GeluFragArgs { uint M, Mp, Nreal, Npad; };
kernel void vit_bias_gelu_frag(device const float* y [[buffer(0)]], device const float* b [[buffer(1)]],
                               device bfloat* dst [[buffer(2)]], constant GeluFragArgs& a [[buffer(3)]],
                               uint2 g [[thread_position_in_grid]]) {
  const uint c = g.x, m = g.y;
  if (c >= a.Npad || m >= a.Mp) return;
  if (m >= a.M || c >= a.Nreal) {
    dst[frag_off(m, c, a.Npad)] = bfloat(0.0f);
    return;
  }
  dst[frag_off(m, c, a.Npad)] = bfloat(vit_gelu(y[size_t(m) * a.Npad + c] + b[c]));
}

// 7. PatchMerger pre-RMSNorm + 2x2 spatial gather directly into bf16 frag layout [Mp_merge][4*H].
// 1 TG of 256 threads per input patch (frame, py, px) computes RMSNorm(x) and scatters into
// merged row (frame*(h/2)*(w/2) + (py/2)*(w/2) + (px/2)) at channel ((py&1)*2 + (px&1))*H + c.
struct MergeGatherArgs { uint T, h, w, H, Mp_merge; float eps; };
kernel void vit_merge_gather_frag(device const float* x [[buffer(0)]], device const float* nw [[buffer(1)]],
                                  device bfloat* dst [[buffer(2)]], constant MergeGatherArgs& a [[buffer(3)]],
                                  uint m [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                                  ushort ntg [[threads_per_threadgroup]]) {
  const uint N = a.h * a.w;
  const uint total = a.T * N;
  const uint K4 = 4u * a.H;
  if (m < total) {
    threadgroup float sh[32];
    device const float* xr = x + size_t(m) * a.H;
    float ss = 0.0f;
    for (uint c = tid; c < a.H; c += ntg) ss = fma(xr[c], xr[c], ss);
    const float inv = rsqrt(tg_sum(ss, sh, tid, ntg) / float(a.H) + a.eps);
    const uint frame = m / N, loc = m % N;
    const uint py = loc / a.w, px = loc % a.w;
    const uint hm = a.h >> 1, wm = a.w >> 1;
    const uint m_out = (frame * hm + (py >> 1)) * wm + (px >> 1);
    const uint sub = ((py & 1u) << 1) | (px & 1u);
    const uint k0 = sub * a.H;
    for (uint c = tid; c < a.H; c += ntg) {
      dst[frag_off(m_out, k0 + c, K4)] = bfloat(xr[c] * inv * nw[c]);
    }
  } else {
    // Zero-pad tail merged rows [M_merge .. Mp_merge - 1] (dispatched as extra TGs after total).
    const uint m_merge = a.T * (a.h >> 1) * (a.w >> 1);
    const uint m_out = m_merge + (m - total);
    if (m_out < a.Mp_merge) {
      for (uint k = tid; k < K4; k += ntg) dst[frag_off(m_out, k, K4)] = bfloat(0.0f);
    }
  }
}

// 8. PatchMerger output: add bias + insert learned image_newline at the end of every merged row.
// Input y: [T * (h/2) * (w/2)][out_H]; Output out: [T * (h/2) * (w/2 + 1)][out_H].
struct MergeOutArgs { uint T, hm, wm, out_H; };
kernel void vit_merge_out(device const float* y [[buffer(0)]], device const float* b [[buffer(1)]],
                          device const float* newline [[buffer(2)]], device float* out [[buffer(3)]],
                          constant MergeOutArgs& a [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  const uint c = g.x, row = g.y;
  const uint cols = a.wm + 1u;
  const uint total_out = a.T * a.hm * cols;
  if (c >= a.out_H || row >= total_out) return;
  const uint col = row % cols;
  if (col == a.wm) {
    out[size_t(row) * a.out_H + c] = newline[c];
  } else {
    const uint y_row = (row / cols) * a.wm + col;
    out[size_t(row) * a.out_H + c] = y[size_t(y_row) * a.out_H + c] + b[c];
  }
}
