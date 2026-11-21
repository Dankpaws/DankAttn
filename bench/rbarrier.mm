// Does a resource-scoped barrier on a concurrent encoder let a dependent chain run beside an
// unrelated long dispatch? Chain of tiny dependent dispatches on buffer A, with and without
// a long read-only stream over buffer B encoded first.
#include "../engine/mtl.h"

static const char* kSrc = R"(
#include <metal_stdlib>
using namespace metal;
kernel void tiny(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] = x[i] * 0.5f + 1.0f; }
kernel void stream(device const uint4* p [[buffer(0)]], device uint* sink [[buffer(1)]], constant uint& n [[buffer(2)]],
                   uint i [[thread_position_in_grid]], uint ng [[threads_per_grid]]) {
  uint acc = 0;
  for (uint j = i; j < n; j += ng) acc ^= p[j].x;
  if (acc == 0x9E3779B9u) sink[0] = acc;
}
)";

int main() {
  @autoreleasepool {
    mtl::Ctx C;
    id<MTLComputePipelineState> pt = C.pipe(kSrc, "tiny"), ps = C.pipe(kSrc, "stream");
    id<MTLBuffer> A = C.buf(1 << 20), B = C.buf(size_t(512) << 20), sink = C.buf(16);
    const uint32_t n16 = uint32_t(B.length / 16);
    auto run = [&](bool with_stream, bool resource_barrier, int chain) {
      double best = 1e9;
      for (int rep = 0; rep < 5; rep++) {
        id<MTLCommandBuffer> cb = [C.q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        if (with_stream) {
          [e setComputePipelineState:ps];
          [e setBuffer:B offset:0 atIndex:0];
          [e setBuffer:sink offset:0 atIndex:1];
          [e setBytes:&n16 length:4 atIndex:2];
          [e dispatchThreads:MTLSizeMake(65536, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [e setComputePipelineState:pt];
        [e setBuffer:A offset:0 atIndex:0];
        for (int i = 0; i < chain; i++) {
          if (i) {
            if (resource_barrier) [e memoryBarrierWithResources:&A count:1];
            else [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
          }
          [e dispatchThreads:MTLSizeMake(4096, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        best = std::min(best, mtl::gpu_seconds(cb));
      }
      return best * 1e6;
    };
    // barrier over many resources (what an engine-wide resource barrier would name)
    std::vector<id<MTLResource>> many{A};
    for (int i = 0; i < 99; i++) many.push_back(C.buf(4096));
    for (size_t nres : {size_t(1), size_t(10), size_t(50), size_t(100)}) {
      double best = 1e9;
      for (int rep = 0; rep < 5; rep++) {
        id<MTLCommandBuffer> cb = [C.q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        [e setComputePipelineState:pt];
        [e setBuffer:A offset:0 atIndex:0];
        for (int i = 0; i < 1000; i++) {
          if (i) [e memoryBarrierWithResources:many.data() count:nres];
          [e dispatchThreads:MTLSizeMake(4096, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        best = std::min(best, mtl::gpu_seconds(cb));
      }
      printf("resource barrier over %3zu buffers: %.2f us per dependent dispatch\n", nres, best / 1000 * 1e6);
    }
    const double s_only = run(true, false, 0);
    for (int rb = 0; rb < 2; rb++) {
      const double c_only = run(false, rb, 200), both = run(true, rb, 200);
      printf("%s barrier: stream alone %.0f us, chain alone %.0f us, both %.0f us (overlap if ~max)\n",
             rb ? "resource" : "scope   ", s_only, c_only, both);
    }
  }
}
