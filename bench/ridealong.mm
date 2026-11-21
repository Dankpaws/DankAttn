// Ride-along prefetch: can a latency-bound small dispatch carry extra threadgroups that read
// the NEXT dispatch's weights, so that the next (bandwidth-bound) dispatch finds part of them
// in cache? Each stage is [small op: SPIN iterations of arithmetic on 16 threadgroups]
// then [big op: stream BIG MB of this stage's own weights]. With prefetch, the small op's
// grid grows by threadgroups that read the first PRE MB of the big op's range and discard it.
// 48 stages over distinct ranges, so every big read is cold unless prefetched.
// Run from ~/opus-next/src: ridealong
#include "../engine/mtl.h"

static const char* kSrc = R"(
#include <metal_stdlib>
using namespace metal;
struct A { uint per, base, work_tgs, spin, slot; };
// threadgroups < work_tgs: the "small op" (pure arithmetic, no memory traffic);
// threadgroups >= work_tgs: read `per` uint4 each from W starting at base and drop them.
kernel void small_op(device const uint4* W [[buffer(0)]], device float* out [[buffer(1)]],
                     constant A& a [[buffer(2)]],
                     uint g [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
  if (g < a.work_tgs) {
    float x = out[a.slot * 65536 + t] + float(t);
    for (uint i = 0; i < a.spin; i++) x = x * 1.0000001f + 0.5f;
    out[(a.slot + 1) * 65536 + g * 1024 + t] = x;
    return;
  }
  device const uint4* p = W + a.base + (size_t(g - a.work_tgs) * 1024 + t) * a.per;
  uint4 s = uint4(0);
  for (uint i = 0; i < a.per; i++) s += p[i];
  if (s.x == 0xFFFFFFFFu) out[0] = 1.0f;  // keep the reads alive
}
kernel void big_op(device const uint4* W [[buffer(0)]], device float* out [[buffer(1)]],
                   constant A& a [[buffer(2)]],
                   uint g [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
  const float dep = out[(a.slot + 1) * 65536];
  device const uint4* p = W + a.base + (size_t(g) * 1024 + t) * a.per;
  uint4 s = uint4(uint(dep) & 1u);
  for (uint i = 0; i < a.per; i++) s += p[i];
  out[(a.slot + 1) * 65536 + g * 1024 + t] = float(s.x & 255u);
}
)";

int main() {
  @autoreleasepool {
    mtl::Ctx C;
    id<MTLComputePipelineState> pS = C.pipe(kSrc, "small_op"), pB = C.pipe(kSrc, "big_op");
    const int NL = 48;
    const size_t MAXBIG = size_t(48) << 20;
    id<MTLBuffer> W = C.buf(MAXBIG * NL), out = C.buf(size_t(64) * 65536 * 4), ev = C.buf(size_t(2) << 30);
    memset(W.contents, 1, W.length);
    memset(ev.contents, 1, ev.length);
    memset(out.contents, 0, out.length);
    struct A { uint32_t per, base, work_tgs, spin, slot; };
    auto evict = [&] {
      id<MTLCommandBuffer> cb = [C.q commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:pB];
      A a{uint32_t(ev.length / 16 / (64 * 1024)), 0, 0, 0, 62};
      [e setBuffer:ev offset:0 atIndex:0];
      [e setBuffer:out offset:0 atIndex:1];
      [e setBytes:&a length:sizeof a atIndex:2];
      [e dispatchThreadgroups:MTLSizeMake(64, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
      [e endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
    };
    auto run = [&](size_t big_mb, size_t pre_mb, uint32_t spin, uint32_t pre_tgs, bool small_only) {
      evict();
      id<MTLCommandBuffer> cb = [C.q commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setBuffer:W offset:0 atIndex:0];
      [e setBuffer:out offset:0 atIndex:1];
      const uint32_t btg = 32, wtg = 16;
      for (int i = 0; i < NL; i++) {
        const uint32_t base = uint32_t(size_t(i) * MAXBIG / 16);
        A s{pre_mb ? uint32_t((pre_mb << 20) / 16 / (pre_tgs * 1024)) : 0, base, wtg, spin, uint32_t(i % 60)};
        [e setComputePipelineState:pS];
        [e setBytes:&s length:sizeof s atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(wtg + (pre_mb ? pre_tgs : 0), 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        if (small_only) continue;
        A b{uint32_t((big_mb << 20) / 16 / (btg * 1024)), base, 0, 0, uint32_t(i % 60)};
        [e setComputePipelineState:pB];
        [e setBytes:&b length:sizeof b atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(btg, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
      }
      [e endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      return mtl::gpu_seconds(cb) * 1e6 / NL;
    };
    auto best = [&](size_t big_mb, size_t pre_mb, uint32_t spin, uint32_t pre_tgs, bool so) {
      double b = 1e9;
      for (int r = 0; r < 5; r++) b = std::min(b, run(big_mb, pre_mb, spin, pre_tgs, so));
      return b;
    };
    for (int i = 0; i < 3; i++) evict();
    for (uint32_t spin : {400u, 1200u}) {
      const double so = best(0, 0, spin, 16, true);
      printf("small op alone (spin %u): %.1f us\n", spin, so);
      for (size_t big : {size_t(16), size_t(42)}) {
        const double base = best(big, 0, spin, 16, false);
        printf("  big %2zu MB: no prefetch %.1f us/stage (big op %.1f us, %.0f GB/s)\n", big, base, base - so,
               double(big << 20) / ((base - so) * 1e-6) / 1e9);
        for (size_t pre : {size_t(4), size_t(8), size_t(16), size_t(32), size_t(42)}) {
          if (pre > big) continue;
          for (uint32_t ptg : {8u, 32u}) {
            const double t = best(big, pre, spin, ptg, false);
            printf("    prefetch %2zu MB on %2u tgs: %.1f us/stage (%+.1f us, %+.0f%%)\n", pre, ptg, t, t - base, 100 * (t - base) / base);
          }
        }
      }
    }
  }
}
