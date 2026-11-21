// attn_rows vs attn_rows_mma against an f64 reference on synthetic q / K / V (all tokens
// attended, nsel = all). Partials are merged in f64 on the CPU, so only the row kernels
// are compared. Run from ~/opus/src: attntest [qscale]
#include "../engine/mtl.h"
#include <cmath>
#include <random>
#include <vector>

struct AttnArgs { uint32_t M, split, N, nh, nkv, hd, inh, ihd, rot, p0, ratio, cap, budget; };
static const uint32_t NH = 24, NKV = 2, HD = 256, SMAX = 16;
static uint32_t vperm(uint32_t d) { return (d & ~31u) + (d & 7u) * 4 + ((d >> 3) & 3u); }
static float bf(float x) { uint32_t u; memcpy(&u, &x, 4); u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000u; memcpy(&x, &u, 4); return x; }
static uint16_t bfbits(float x) { uint32_t u; x = bf(x); memcpy(&u, &x, 4); return u >> 16; }

int main(int argc, char** argv) {
  @autoreleasepool {
    const float qs = argc > 1 ? atof(argv[1]) : 2.0f;
    mtl::Ctx C;
    std::string at = mtl::slurp("engine/kernels/attn.metal");
    for (auto [M, p0] : {std::pair<uint32_t, uint32_t>{256, 0}, {32, 3000}}) {
      const uint32_t cap = p0 + M;
      std::mt19937 rng(11 + p0);
      std::normal_distribution<float> nd;
      std::vector<float> q(size_t(M) * NH * HD), K(size_t(cap) * NKV * HD), V(K.size());
      for (auto& x : q) x = nd(rng) * qs;
      for (auto& x : K) x = bf(nd(rng) * (1.0f + 2.0f * (rng() % 16 == 0)));
      for (auto& x : V) x = bf(nd(rng));
      id<MTLBuffer> bq = C.buf(q.size() * 4), bg = C.buf(q.size() * 4), bK = C.buf(K.size() * 2), bV = C.buf(V.size() * 2),
                    bsel = C.buf(16), bns = C.buf(M * 4), bp = C.buf(size_t(M) * NH * SMAX * (HD + 2) * 4);
      memcpy(bq.contents, q.data(), q.size() * 4);
      uint16_t* k16 = (uint16_t*)bK.contents;
      uint16_t* v16 = (uint16_t*)bV.contents;
      for (size_t t = 0; t < cap * NKV; t++)
        for (uint32_t d = 0; d < HD; d++) {
          k16[t * HD + d] = bfbits(K[t * HD + d]);
          v16[t * HD + vperm(d)] = bfbits(V[t * HD + d]);
        }
      for (uint32_t r = 0; r < M; r++) ((uint32_t*)bns.contents)[r] = 0xFFFFFFFFu;
      AttnArgs a{M, 1, 0, NH, NKV, HD, 4, 128, 64, p0, 4, cap, 2048};
      // f64 reference
      std::vector<double> ref(size_t(M) * NH * HD);
      for (uint32_t r = 0; r < M; r++)
        for (uint32_t h = 0; h < NH; h++) {
          const uint32_t kvh = h / (NH / NKV), n = p0 + r + 1;
          std::vector<double> s(n);
          double mx = -1e300;
          for (uint32_t j = 0; j < n; j++) {
            double d = 0;
            for (uint32_t e = 0; e < HD; e++) d += double(q[(size_t(r) * NH + h) * HD + e]) * K[(size_t(j) * NKV + kvh) * HD + e];
            s[j] = d / 16.0;
            mx = std::max(mx, s[j]);
          }
          double L = 0;
          for (auto& x : s) { x = exp(x - mx); L += x; }
          for (uint32_t e = 0; e < HD; e++) {
            double o = 0;
            for (uint32_t j = 0; j < n; j++) o += s[j] * V[(size_t(j) * NKV + kvh) * HD + e];
            ref[(size_t(r) * NH + h) * HD + e] = o / L;
          }
        }
      for (const char* fn : {"attn_rows", "attn_rows_mma"}) {
        id<MTLComputePipelineState> p = C.pipe(at, fn);
        memset(bp.contents, 0, bp.length);
        id<MTLCommandBuffer> cb = [C.q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p];
        [e setBuffer:bq offset:0 atIndex:0];
        [e setBuffer:bg offset:0 atIndex:1];
        [e setBuffer:bK offset:0 atIndex:2];
        [e setBuffer:bV offset:0 atIndex:3];
        [e setBuffer:bsel offset:0 atIndex:4];
        [e setBuffer:bns offset:0 atIndex:5];
        [e setBuffer:bp offset:0 atIndex:6];
        [e setBytes:&a length:sizeof a atIndex:7];
        const bool mma = strcmp(fn, "attn_rows_mma") == 0;
        [e dispatchThreadgroups:MTLSizeMake(M, NKV, SMAX) threadsPerThreadgroup:MTLSizeMake(mma ? 128 : 32 * (NH / NKV), 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const float* pp = (const float*)bp.contents;
        double worst = 0, sum = 0, bias = 0;
        size_t cnt = 0;
        for (uint32_t r = 0; r < M; r++)
          for (uint32_t h = 0; h < NH; h++) {
            const uint32_t ntok = p0 + r + 1, S = std::min(SMAX, (ntok + 127) / 128);
            const float* b = pp + (size_t(r) * NH + h) * SMAX * (HD + 2);
            double Mx = -1e300;
            for (uint32_t s = 0; s < S; s++) Mx = std::max(Mx, double(b[s * (HD + 2)]));
            double L = 0;
            std::vector<double> o(HD, 0.0);
            for (uint32_t s = 0; s < S; s++) {
              const double c = exp(double(b[s * (HD + 2)]) - Mx);
              L += b[s * (HD + 2) + 1] * c;
              for (uint32_t d = 0; d < HD; d++) o[d] += b[s * (HD + 2) + 2 + d] * c;
            }
            double en = 0, rn = 0, dot = 0;
            for (uint32_t d = 0; d < HD; d++) {
              const double x = o[d] / L, y = ref[(size_t(r) * NH + h) * HD + d];
              en += (x - y) * (x - y);
              rn += y * y;
              dot += (x - y) * y;
            }
            const double rel = sqrt(en / rn);
            worst = std::max(worst, rel);
            sum += rel;
            bias += dot / rn;
            cnt++;
          }
        printf("M=%u p0=%u %-14s rel.err mean %.3g max %.3g  projection on ref (bias) %.3g\n", M, p0, fn, sum / cnt, worst, bias / cnt);
      }
    }
  }
}
