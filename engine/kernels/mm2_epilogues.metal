// Model-specific matmul epilogues included by engine/kernels/mm2.metal.
// Keeping these out of mm2.metal isolates per-architecture activations and fused post-ops
// (Flash-Next HC, SwiGLU/SiTU-GLU variants, Inkling2 sconv, ERNIE-5.5 HCA wo_a) from the
// generic register-direct W2/W4/W6/W8/NVFP4/MXFP8 matmul kernels.

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
#elif EPI == 5
                            , uint z = 0u
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
#elif defined(SITU_BETA)  // ERNIE-5.5: beta * tanh(g / beta) * sigmoid(g) * (lbeta > 0 ? lbeta * tanh(u / lbeta) : u)
    const float gv = Y[m * N + nb + j], uv = Y[m * N + nb + 16 + j];
    const float ga = float(SITU_BETA) * tanh(gv / float(SITU_BETA)) * esigm(gv);
    const float ua = (float(SITU_LBETA) > 0.0f) ? (float(SITU_LBETA) * tanh(uv / float(SITU_LBETA))) : uv;
    e0[frag_off(row0 + m, c0 + j, a.R)] = bfloat(ga * ua);
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
#elif EPI == 5
  // ERNIE-5.5 HCA wo_a fused gate*sigm + frag_a pack (z = group 0..7, a.R = 8192, a.H = 40960, a.HC = 32768)
  for (uint idx = l; idx < Mloc * 32; idx += 32) {
    const uint m = idx >> 5, n = nb + (idx & 31);
    const uint col = z * N + n;
    const float gv = e1[size_t(row0 + m) * a.H + a.HC + col];
    e0[frag_off(row0 + m, col, a.R)] = bfloat(Y[m * N + n] * esigm(gv));
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
#elif defined(SITU_BETA)
    const float ga = float(SITU_BETA) * tanh(g / float(SITU_BETA)) * esigm(g);
    const float ua = (float(SITU_LBETA) > 0.0f) ? (float(SITU_LBETA) * tanh(u / float(SITU_LBETA))) : u;
    e0[frag_off(row0 + m, c0 + j, a.R)] = bfloat(ga * ua);
#else
    e0[frag_off(row0 + m, c0 + j, a.R)] = bfloat(xsilu(g) * u);
#endif
  }
#endif
}
