// What a cache-resident weight set is worth to a chain of dependent small matmuls: 48 serial
// mm2 dispatches of one decode shape (M = 1), each stage with its own weights (cold DRAM)
// vs all stages on one weight set (SLC-resident after the first). Run from ~/opus/src.
#include "../engine/mtl.h"
#include <random>

struct MmArgs { uint32_t N, K, M, ksplit, wstride, sstride, xstride, pstride, ystride, sgs, ldx; float xdiv; uint32_t psplit, R, H, HC; };

static void pick_split(uint32_t N, uint32_t K, uint32_t gs, uint32_t& sgs, uint32_t& ksplit) {
  const uint32_t G = K / gs, nf = N / 32;
  uint32_t s_tot = G;
  for (uint32_t s = 1; s <= G; s++)
    if (G % s == 0 && nf * s >= 1024) { s_tot = s; break; }
  for (;;) {
    sgs = 1;
    for (uint32_t c = 8; c >= 2; c--)
      if (s_tot % c == 0) { sgs = c; break; }
    ksplit = s_tot / sgs;
    if (ksplit <= 16) break;
    uint32_t nx = 1;
    for (uint32_t s = s_tot - 1; s >= 1; s--)
      if (G % s == 0) { nx = s; break; }
    s_tot = nx;
  }
}

int main() {
  @autoreleasepool {
    mtl::Ctx C;
    std::string src = mtl::slurp("engine/kernels/mm2.metal");
    struct Shape { const char* nm; uint32_t N, K, bits, gs; } shapes[] = {
        {"hc.dn", 352, 10240, 8, 128}, {"hc.up", 10240, 320, 8, 64}, {"moe.sdown", 2560, 640, 8, 128},
        {"gdn.out", 2560, 6144, 8, 128}, {"gdn.in", 16480, 2560, 8, 128}};
    const int NS = 48;
    std::mt19937 rng(5);
    for (auto& sh : shapes) {
      uint32_t sgs, ksplit;
      pick_split(sh.N, sh.K, sh.gs, sgs, ksplit);
     for (int pf : {1, 2, 4}) {
      id<MTLComputePipelineState> p = C.pipe(src, "mm2", {{"WBITS", std::to_string(sh.bits)}, {"GS", std::to_string(sh.gs)}, {"PF", std::to_string(pf)}});
      const size_t wb = size_t(sh.N) * sh.K * sh.bits / 8, sbytes = size_t(sh.N) * (sh.K / sh.gs) * 2;
      std::vector<id<MTLBuffer>> W(NS), S(NS);
      for (int i = 0; i < NS; i++) {
        W[i] = C.buf(wb);
        S[i] = C.buf(sbytes);
        uint32_t* w = (uint32_t*)W[i].contents;
        for (size_t j = 0; j < wb / 4; j++) w[j] = rng();
        uint16_t* s = (uint16_t*)S[i].contents;
        for (size_t j = 0; j < sbytes / 2; j++) s[j] = 0x1400;  // small half
      }
      // stage outputs feed nothing (the chain is ordered by the serial encoder alone)
      id<MTLBuffer> X = C.buf(size_t(sh.K) * 32 * 2), Y = C.buf(size_t(sh.N) * 16 * 4),
                    P = C.buf(size_t(16) * 16 * sh.N * 4), cnt = C.buf(size_t(sh.N / 32) * 4 + 64), d = C.buf(64);
      memset(X.contents, 0, X.length);
      memset(cnt.contents, 0, cnt.length);
      uint32_t nft = 1;
      while (sgs * nft < 4 && (sh.N / 32) % (nft * 2) == 0) nft *= 2;
      MmArgs a{sh.N, sh.K, 1, ksplit, 0, 0, 0, ksplit * sh.N, 0, sgs, 0, 1.0f, 16 * sh.N, 0, 0, 0};
      for (int same = 0; same < 2; same++) {
        double best = 1e9;
        for (int rep = 0; rep < 5; rep++) {
          id<MTLCommandBuffer> cb = [C.q commandBuffer];
          id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
          for (int i = 0; i < NS; i++) {
            const int wi = same ? 0 : i;
            [e setComputePipelineState:p];
            [e setBuffer:W[wi] offset:0 atIndex:0];
            [e setBuffer:S[wi] offset:0 atIndex:1];
            [e setBuffer:X offset:0 atIndex:2];
            [e setBuffer:P offset:0 atIndex:3];
            [e setBuffer:Y offset:0 atIndex:4];
            [e setBytes:&a length:sizeof a atIndex:5];
            [e setBuffer:d offset:0 atIndex:6];
            [e setBuffer:d offset:0 atIndex:7];
            [e setBuffer:cnt offset:0 atIndex:8];
            [e setBuffer:d offset:0 atIndex:9];
            [e setBuffer:d offset:0 atIndex:10];
            [e setBuffer:d offset:0 atIndex:11];
            [e setThreadgroupMemoryLength:(sgs > 1 ? size_t(sgs) * nft * 32 * 16 * 4 : 16) atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake(sh.N / 32 / nft, ksplit, 1) threadsPerThreadgroup:MTLSizeMake(32 * sgs * nft, 1, 1)];
          }
          [e endEncoding];
          [cb commit];
          [cb waitUntilCompleted];
          best = std::min(best, mtl::gpu_seconds(cb));
        }
        const double us = best / NS * 1e6;
        printf("%-10s %6.2f MB  sgs %u ksplit %u PF %d  %s: %6.1f us/stage  %6.0f GB/s\n", sh.nm, wb / 1e6, sgs, ksplit, pf,
               same ? "same weights (cached)" : "distinct weights     ", us, wb / (us * 1e-6) / 1e9);
      }
     }
    }
  }
}
