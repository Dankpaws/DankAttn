// Serial dependent mm2 dispatches: weights cycling through DRAM-sized copies vs one copy
// that stays cache-resident. Tells how much an SLC prefetcher could buy per op.
#include "../engine/mtl.h"
#include <random>
struct MmArgs { uint32_t N, K, M, ksplit, wstride, sstride, xstride, pstride, ystride, sgs; };
int main() {
  @autoreleasepool {
    mtl::Ctx C;
    std::string src = mtl::slurp("engine/kernels/mm2.metal");
    id<MTLComputePipelineState> p8 = C.pipe(src, "mm2", {{"WBITS", "8"}, {"GS", "128"}});
    std::mt19937 rng(1);
    struct Sh { const char* n; uint32_t N, K, sgs, ksplit; } shapes[] = {
        {"hc.dn/10", 352, 10240, 8, 10}};
    for (int warm = 0; warm < 2; warm++)
    for (auto& s : shapes) {
      size_t wb = size_t(s.N) * s.K, sb = size_t(s.N) * (s.K / 128) * 2;
      for (size_t big : {size_t(8) << 20, size_t(16) << 20, size_t(32) << 20, size_t(48) << 20, size_t(64) << 20, size_t(96) << 20, size_t(128) << 20, size_t(256) << 20, size_t(1) << 30}) {
        size_t nc = std::max<size_t>(1, big / wb);
        id<MTLBuffer> W = C.buf(wb * nc), S = C.buf(sb), X = C.buf(16 * s.K * 2), P = C.buf(size_t(s.ksplit) * 16 * s.N * 4),
                      Y = C.buf(16 * s.N * 4), cnt = C.buf(s.N / 32 * 4 + 64), sl = C.buf(64);
        memset(cnt.contents, 0, cnt.length);
        for (size_t i = 0; i < wb * nc / 4; i += 97) ((uint32_t*)W.contents)[i] = rng();
        MmArgs a{s.N, s.K, 1, s.ksplit, 0, 0, 0, 0, 0, s.sgs};
        uint32_t nft = 1;
        while (s.sgs * nft < 4 && (s.N / 32) % (nft * 2) == 0) nft *= 2;
        const int reps = 200;
        id<MTLCommandBuffer> cb = [C.q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p8];
        for (int r = 0; r < reps; r++) {
          [e setBuffer:W offset:(r % nc) * wb atIndex:0];
          [e setBuffer:S offset:0 atIndex:1]; [e setBuffer:X offset:0 atIndex:2]; [e setBuffer:P offset:0 atIndex:3];
          [e setBuffer:Y offset:0 atIndex:4]; [e setBytes:&a length:sizeof a atIndex:5]; [e setBuffer:sl offset:0 atIndex:6];
          [e setBuffer:sl offset:0 atIndex:7]; [e setBuffer:cnt offset:0 atIndex:8];
          [e dispatchThreadgroups:MTLSizeMake(s.N / 32 / nft, s.ksplit, 1) threadsPerThreadgroup:MTLSizeMake(32 * s.sgs * nft, 1, 1)];
        }
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        double us = mtl::gpu_seconds(cb) / reps * 1e6;
        if (warm) printf("%-8s working set %5zu MB: %6.1f us/op\n", s.n, nc * wb >> 20, us);
      }
    }
  }
}
