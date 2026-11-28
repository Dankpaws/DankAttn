// Standalone Metal 4 unit test for WBITS=3 (W3) and WBITS=41 (MXFP4, GS=32) in mm2.metal.
// Verifies mm1 (M=1), mm2 (M=1, 4, 16), and mmp (M=64) against f32 CPU reference + M=1==M=16 bit-identity.
// Build: clang++ -std=c++20 -fobjc-arc -O2 bench/quant_test.mm -framework Metal -framework Foundation -o build/quant_test
#include "../engine/mtl.h"
#include "../engine/mm_args.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static constexpr float E2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static inline uint32_t rk(uint32_t l, uint32_t i) { return 4 * (l & 1) + 8 * ((l >> 3) & 1) + (i & 3) + 16 * (i >> 4); }
static inline uint32_t rn(uint32_t l, uint32_t i) { return ((l >> 1) & 3) + 4 * (l >> 4) + 8 * ((i >> 2) & 3); }
static inline uint32_t frag_off(uint32_t m, uint32_t k, uint32_t K) {
  const uint32_t kk = k & 31, mm = m & 15;
  const uint32_t l = ((kk >> 2) & 1) | ((mm & 3) << 1) | (((kk >> 3) & 1) << 3) | (((mm >> 2) & 1) << 4);
  const uint32_t i = (kk & 3) | ((mm >> 3) << 2) | ((kk >> 4) << 3);
  return ((m >> 4) * (K >> 5) + (k >> 5)) * 512 + l * 16 + i;
}
static inline uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return uint16_t(u >> 16); }
static inline float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; memcpy(&f, &u, 4); return f; }

static int run_tier(mtl::Ctx& C, const std::string& mm2_src, int wbits, uint32_t gs, const char* name) {
  const uint32_t N = 256, K = 512, NF = N / 32, KS = K / 32, G = K / gs, Mmax = 64;
  std::vector<float> Xf(Mmax * K);
  for (size_t i = 0; i < Xf.size(); i++) {
    float v = float(int((i * 1103515245u + 12345u) >> 16) % 21 - 10) * 0.125f;
    Xf[i] = bf2f(f2bf(v));  // exact bf16 values
  }
  // Pack bf16 frags for X
  id<MTLBuffer> xbuf = C.buf(Mmax * K * 2);
  uint16_t* xp = (uint16_t*)xbuf.contents;
  for (uint32_t m = 0; m < Mmax; m++)
    for (uint32_t k = 0; k < K; k++)
      xp[frag_off(m, k, K)] = f2bf(Xf[m * K + k]);

  // Generate codes & scales
  std::vector<int8_t> codes(N * K);
  std::vector<float> scales(N * G);
  for (uint32_t n = 0; n < N; n++) {
    for (uint32_t g = 0; g < G; g++) scales[n * G + g] = 0.0625f * float(1 + ((n * 7 + g * 13) & 7));
    for (uint32_t k = 0; k < K; k++) {
      uint32_t h = (n * 2654435761u ^ k * 40503u) >> 16;
      codes[n * K + k] = (wbits == 3) ? int8_t((h & 7u) - 4) : int8_t(h & 15u);
    }
  }
  // Pack scales [NF][G][32] f16
  id<MTLBuffer> sbuf = C.buf(N * G * 2);
  __fp16* sp = (__fp16*)sbuf.contents;
  for (uint32_t nf = 0; nf < NF; nf++)
    for (uint32_t g = 0; g < G; g++)
      for (uint32_t j = 0; j < 32; j++)
        sp[(nf * G + g) * 32 + j] = __fp16(scales[(nf * 32 + j) * G + g]);

  // Pack weights in register-direct layout
  const size_t wbytes = (wbits == 3) ? size_t(NF) * KS * 32 * 12 : size_t(NF) * KS * 32 * 16;
  id<MTLBuffer> wbuf = C.buf(wbytes);
  memset(wbuf.contents, 0, wbytes);
  if (wbits == 3) {
    uint32_t* wp = (uint32_t*)wbuf.contents;
    for (uint32_t nf = 0; nf < NF; nf++)
      for (uint32_t ks = 0; ks < KS; ks++)
        for (uint32_t l = 0; l < 32; l++) {
          uint32_t lo0 = 0, lo1 = 0, hi = 0;
          for (uint32_t i = 0; i < 32; i++) {
            uint32_t u = uint32_t(codes[(nf * 32 + rn(l, i)) * K + ks * 32 + rk(l, i)] + 4) & 7u;
            if (i < 16) lo0 |= (u & 3u) << (2 * i);
            else lo1 |= (u & 3u) << (2 * (i - 16));
            hi |= ((u >> 2) & 1u) << i;
          }
          uint32_t* dst = wp + ((nf * KS + ks) * 32 + l) * 3;
          dst[0] = lo0; dst[1] = lo1; dst[2] = hi;
        }
  } else {
    uint8_t* wp = (uint8_t*)wbuf.contents;
    for (uint32_t nf = 0; nf < NF; nf++)
      for (uint32_t ks = 0; ks < KS; ks++)
        for (uint32_t l = 0; l < 32; l++) {
          uint8_t* dst = wp + ((nf * KS + ks) * 32 + l) * 16;
          for (uint32_t w = 0; w < 4; w++)
            for (uint32_t j = 0; j < 4; j++) {
              uint8_t lo = uint8_t(codes[(nf * 32 + rn(l, w * 8 + j)) * K + ks * 32 + rk(l, w * 8 + j)]) & 15u;
              uint8_t hi = uint8_t(codes[(nf * 32 + rn(l, w * 8 + 4 + j)) * K + ks * 32 + rk(l, w * 8 + 4 + j)]) & 15u;
              dst[w * 4 + j] = lo | (hi << 4);
            }
        }
  }

  // CPU reference
  std::vector<float> ref(Mmax * N, 0.0f);
  for (uint32_t m = 0; m < Mmax; m++)
    for (uint32_t n = 0; n < N; n++) {
      float acc = 0.0f;
      for (uint32_t g = 0; g < G; g++) {
        float c = 0.0f;
        for (uint32_t u = 0; u < gs; u++) {
          uint32_t k = g * gs + u;
          float wv = (wbits == 3) ? float(codes[n * K + k]) : E2M1[uint8_t(codes[n * K + k]) & 15u];
          c += Xf[m * K + k] * wv;
        }
        acc += c * float(__fp16(scales[n * G + g]));
      }
      ref[m * N + n] = acc;
    }

  std::map<std::string, std::string> defs = {
      {"WBITS", std::to_string(wbits)}, {"GS", std::to_string(gs)},
      {"TILES", "1"}, {"EPI", "0"}, {"XMODE", "0"}, {"PF", "1"},
  };
  id<MTLComputePipelineState> p_mm2 = C.pipe(mm2_src, "mm2", defs);
  id<MTLComputePipelineState> p_mm1 = C.pipe(mm2_src, "mm1", defs);
  auto mmp_defs = defs;
  mmp_defs["MMP_NF"] = "2"; mmp_defs["MMP_CH"] = "4"; mmp_defs["MMP_FAST"] = "1";
  id<MTLComputePipelineState> p_mmp = C.pipe(mm2_src, "mmp", mmp_defs);

  id<MTLBuffer> ybuf = C.buf(Mmax * N * 4);
  id<MTLBuffer> pbuf = C.buf(Mmax * N * 4 * 4);
  id<MTLBuffer> cnt = C.buf(NF * 4);
  id<MTLBuffer> dum = C.buf(64);
  memset(cnt.contents, 0, NF * 4);

  auto run_mm2 = [&](uint32_t M, std::vector<float>& out) {
    memset(ybuf.contents, 0, M * N * 4);
    MmArgs a{};
    a.N = N; a.K = K; a.M = M; a.ksplit = 1; a.sgs = 4; a.psplit = M * N;
    id<MTLCommandBuffer> cb = [C.q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:p_mm2];
    [e setBuffer:wbuf offset:0 atIndex:0];
    [e setBuffer:sbuf offset:0 atIndex:1];
    [e setBuffer:xbuf offset:0 atIndex:2];
    [e setBuffer:pbuf offset:0 atIndex:3];
    [e setBuffer:ybuf offset:0 atIndex:4];
    [e setBytes:&a length:sizeof(a) atIndex:5];
    for (int i = 6; i <= 15; i++) [e setBuffer:(i == 8 ? cnt : dum) offset:0 atIndex:i];
    [e setThreadgroupMemoryLength:4 * 16 * 32 * 4 atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(NF, 1, 1) threadsPerThreadgroup:MTLSizeMake(4 * 32, 1, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    out.assign((float*)ybuf.contents, (float*)ybuf.contents + M * N);
  };

  std::vector<float> y1, y4, y16;
  run_mm2(1, y1);
  run_mm2(4, y4);
  run_mm2(16, y16);

  // Check bit-identity of row 0 across M=1, M=4, M=16
  int bit_diffs = 0;
  float max_err_mm2 = 0.0f;
  for (uint32_t n = 0; n < N; n++) {
    if (memcmp(&y1[n], &y4[n], 4) != 0 || memcmp(&y1[n], &y16[n], 4) != 0) bit_diffs++;
    max_err_mm2 = std::max(max_err_mm2, std::fabs(y16[n] - ref[n]));
  }
  for (uint32_t i = 0; i < 16 * N; i++) max_err_mm2 = std::max(max_err_mm2, std::fabs(y16[i] - ref[i]));

  // Check mm1 (M=1)
  memset(ybuf.contents, 0, N * 4);
  {
    MmArgs a{};
    a.N = N; a.K = K; a.M = 1; a.ksplit = 1; a.sgs = 4;
    id<MTLCommandBuffer> cb = [C.q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:p_mm1];
    [e setBuffer:wbuf offset:0 atIndex:0];
    [e setBuffer:sbuf offset:0 atIndex:1];
    [e setBuffer:xbuf offset:0 atIndex:2];
    [e setBuffer:ybuf offset:0 atIndex:3];
    [e setBytes:&a length:sizeof(a) atIndex:4];
    [e setBuffer:dum offset:0 atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(NF, 1, 1) threadsPerThreadgroup:MTLSizeMake(4 * 32, 1, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
  }
  float max_err_mm1 = 0.0f;
  for (uint32_t n = 0; n < N; n++) max_err_mm1 = std::max(max_err_mm1, std::fabs(((float*)ybuf.contents)[n] - ref[n]));

  // Check mmp (M=64)
  memset(ybuf.contents, 0, Mmax * N * 4);
  {
    MmArgs a{};
    a.N = N; a.K = K; a.M = Mmax; a.ksplit = 1; a.sgs = 4;
    id<MTLCommandBuffer> cb = [C.q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:p_mmp];
    [e setBuffer:wbuf offset:0 atIndex:0];
    [e setBuffer:sbuf offset:0 atIndex:1];
    [e setBuffer:xbuf offset:0 atIndex:2];
    [e setBuffer:pbuf offset:0 atIndex:3];
    [e setBuffer:ybuf offset:0 atIndex:4];
    [e setBytes:&a length:sizeof(a) atIndex:5];
    for (int i = 6; i <= 11; i++) [e setBuffer:dum offset:0 atIndex:i];
    [e dispatchThreadgroups:MTLSizeMake(NF / 2, (Mmax + 127) / 128, 1) threadsPerThreadgroup:MTLSizeMake(8 * 32, 1, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
  }
  float max_err_mmp = 0.0f;
  for (uint32_t i = 0; i < Mmax * N; i++) max_err_mmp = std::max(max_err_mmp, std::fabs(((float*)ybuf.contents)[i] - ref[i]));

  printf("%-10s (WBITS=%2d GS=%3u): mm2_err=%.3e mm1_err=%.3e mmp_err=%.3e M1_M16_bit_diffs=%d\n",
         name, wbits, gs, max_err_mm2, max_err_mm1, max_err_mmp, bit_diffs);
  return (bit_diffs == 0 && max_err_mm2 < 1e-3f && max_err_mm1 < 1e-3f && max_err_mmp < 1e-3f) ? 0 : 1;
}

int main() {
  mtl::Ctx C;
  const std::string src = mtl::slurp(mtl::tree_root() + "/engine/kernels/mm2.metal");
  int rc = 0;
  rc |= run_tier(C, src, 3, 64, "W3-g64");
  rc |= run_tier(C, src, 3, 128, "W3-g128");
  rc |= run_tier(C, src, 41, 32, "MXFP4-g32");
  C.flush_shader_cache();
  printf("shader_cache: hits=%u misses=%u\n", C.archive_hits, C.archive_misses);
  return rc;
}
