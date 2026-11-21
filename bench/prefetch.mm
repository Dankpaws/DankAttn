// Two untested ways to get weights out of DRAM ahead of a dependent chain.
//  1. CPU prefetch: does a buffer the CPU has just read arrive faster on the GPU? (Only if the
//     CPU clusters and the GPU share a cache level.) GPU reads a 32 MB target cold (after a
//     2 GB eviction stream), warm (read twice), and after the CPU summed it.
//  2. Second command queue: does a stream on queue B overlap a dependent chain on queue A?
//     (bench/rbarrier showed an encoder cannot overlap them within one command buffer.)
// Run from ~/opus-next/src: prefetch
#include "../engine/mtl.h"
#include <chrono>
#include <dispatch/dispatch.h>

static const char* kSrc = R"(
#include <metal_stdlib>
using namespace metal;
// 1024 threads per threadgroup; each threadgroup sums `per` bytes*16 starting at its own slice
kernel void stream(device const uint4* W [[buffer(0)]], device uint* out [[buffer(1)]],
                   constant uint2& a [[buffer(2)]],  // {uint4 per thread, base offset in uint4}
                   uint g [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
  const uint per = a.x;
  device const uint4* p = W + a.y + (size_t(g) * 1024 + t) * per;
  uint4 s = uint4(0);
  for (uint i = 0; i < per; i++) s += p[i];
  out[g * 1024 + t] = s.x + s.y + s.z + s.w;
}
// dependent link: reads the previous link's output word, then streams its own slice
kernel void link(device const uint4* W [[buffer(0)]], device uint* out [[buffer(1)]],
                 constant uint4& a [[buffer(2)]],  // {uint4 per thread, base, prev slot, this slot}
                 uint g [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
  const uint per = a.x;
  const uint dep = out[a.z * 65536] & 1u;
  device const uint4* p = W + a.y + (size_t(g) * 1024 + t) * per + dep * 0u;
  uint4 s = uint4(dep);
  for (uint i = 0; i < per; i++) s += p[i];
  out[a.w * 65536 + g * 1024 + t] = s.x + s.y + s.z + s.w;
}
)";

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main() {
  @autoreleasepool {
    mtl::Ctx C;
    id<MTLCommandQueue> q2 = [C.dev newCommandQueueWithMaxCommandBufferCount:64];
    id<MTLComputePipelineState> ps = C.pipe(kSrc, "stream"), pl = C.pipe(kSrc, "link");
    const size_t EV = size_t(2) << 30, TG = size_t(32) << 20, CH = size_t(8) << 20;
    const int NL = 48;
    id<MTLBuffer> ev = C.buf(EV), tgt = C.buf(TG), chw = C.buf(CH * NL), out = C.buf(size_t(64) * 65536 * 4);
    for (id<MTLBuffer> b : {ev, tgt, chw}) memset(b.contents, 1, b.length);

    auto stream = [&](id<MTLCommandQueue> q, id<MTLBuffer> b, size_t bytes, bool wait) {
      const uint32_t ntg = 64, per = uint32_t(bytes / 16 / (ntg * 1024));
      uint32_t a[2] = {per, 0};
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:ps];
      [e setBuffer:b offset:0 atIndex:0];
      [e setBuffer:out offset:0 atIndex:1];
      [e setBytes:a length:8 atIndex:2];
      [e dispatchThreadgroups:MTLSizeMake(ntg, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
      [e endEncoding];
      [cb commit];
      if (wait) [cb waitUntilCompleted];
      return cb;
    };
    auto chain = [&](id<MTLCommandQueue> q, bool wait) {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:pl];
      [e setBuffer:chw offset:0 atIndex:0];
      [e setBuffer:out offset:0 atIndex:1];
      const uint32_t ntg = 16, per = uint32_t(CH / 16 / (ntg * 1024));
      for (int i = 0; i < NL; i++) {
        uint32_t a[4] = {per, uint32_t(size_t(i) * CH / 16), uint32_t(i), uint32_t(i + 1)};
        [e setBytes:a length:16 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(ntg, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
      }
      [e endEncoding];
      [cb commit];
      if (wait) [cb waitUntilCompleted];
      return cb;
    };
    for (int i = 0; i < 3; i++) stream(C.q, ev, EV, true);  // GPU clocks up

    printf("1. CPU prefetch of a 32 MB buffer\n");
    double cold = 1e9, hot = 1e9, cpuw = 1e9, cpu_ms = 0;
    for (int rep = 0; rep < 6; rep++) {
      stream(C.q, ev, EV, true);
      cold = std::min(cold, mtl::gpu_seconds(stream(C.q, tgt, TG, true)));
      hot = std::min(hot, mtl::gpu_seconds(stream(C.q, tgt, TG, true)));
      stream(C.q, ev, EV, true);
      const double t0 = now_s();
      __block uint64_t sink = 0;
      const uint64_t* w = (const uint64_t*)tgt.contents;
      dispatch_apply(16, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t k) {
        uint64_t s = 0;
        const size_t n = TG / 8 / 16;
        for (size_t i = 0; i < n; i++) s += w[k * n + i];
        __sync_fetch_and_add(&sink, s);
      });
      cpu_ms = (now_s() - t0) * 1e3;
      cpuw = std::min(cpuw, mtl::gpu_seconds(stream(C.q, tgt, TG, true)));
    }
    printf("   GPU read cold %.1f us (%.0f GB/s), hot %.1f us (%.0f GB/s), after CPU read %.1f us (%.0f GB/s); CPU read took %.2f ms\n",
           cold * 1e6, TG / cold / 1e9, hot * 1e6, TG / hot / 1e9, cpuw * 1e6, TG / cpuw / 1e9, cpu_ms);

    printf("2. second command queue: 48-link dependent chain (8 MB per link) beside a 1 GB stream\n");
    double ta = 1e9, tb = 1e9, tboth = 1e9, tsame = 1e9;
    for (int rep = 0; rep < 6; rep++) {
      stream(C.q, ev, EV, true);
      double t0 = now_s();
      chain(C.q, true);
      ta = std::min(ta, now_s() - t0);
      t0 = now_s();
      stream(q2, ev, EV / 2, true);
      tb = std::min(tb, now_s() - t0);
      stream(C.q, ev, EV, true);
      t0 = now_s();
      id<MTLCommandBuffer> b1 = stream(q2, ev, EV / 2, false), b2 = chain(C.q, false);
      [b1 waitUntilCompleted];
      [b2 waitUntilCompleted];
      tboth = std::min(tboth, now_s() - t0);
      stream(C.q, ev, EV, true);
      t0 = now_s();
      b1 = stream(C.q, ev, EV / 2, false);
      b2 = chain(C.q, false);
      [b1 waitUntilCompleted];
      [b2 waitUntilCompleted];
      tsame = std::min(tsame, now_s() - t0);
    }
    printf("   chain alone %.0f us, stream alone %.0f us; both on two queues %.0f us, both on one queue %.0f us (overlap if ~max, serial if ~sum %.0f)\n",
           ta * 1e6, tb * 1e6, tboth * 1e6, tsame * 1e6, (ta + tb) * 1e6);

    printf("3. chain with each link's weights read by a second-queue stream just before (prefetched)\n");
    double tc = 1e9, tp = 1e9;
    for (int rep = 0; rep < 6; rep++) {
      stream(C.q, ev, EV, true);
      double t0 = now_s();
      tc = std::min(tc, mtl::gpu_seconds(chain(C.q, true)));
      stream(C.q, ev, EV, true);
      stream(q2, chw, CH * 4, true);  // warm only the first 4 links' weights (32 MB), then run
      id<MTLCommandBuffer> cb = chain(C.q, true);
      tp = std::min(tp, mtl::gpu_seconds(cb));
      (void)t0;
    }
    printf("   chain GPU time cold %.0f us (%.1f us/link); with the first 4 links pre-read %.0f us\n", tc * 1e6, tc * 1e6 / NL, tp * 1e6);
  }
}
