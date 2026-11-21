// Cost of a dependent dispatch on a serial encoder: chains of trivial kernels at several
// grid sizes, and the same chain on a concurrent encoder with buffer barriers.
#include "../engine/mtl.h"

static const char* kSrc = R"(
#include <metal_stdlib>
using namespace metal;
kernel void tiny(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] = x[i] * 0.5f + 1.0f; }
)";

int main() {
  @autoreleasepool {
    mtl::Ctx C;
    id<MTLComputePipelineState> p = C.pipe(kSrc, "tiny");
    id<MTLBuffer> b = C.buf(size_t(1) << 26);
    const int NS = 1000;
    for (int conc = 0; conc < 2; conc++)
      for (uint32_t tgs : {1u, 64u, 1024u, 8192u}) {
        double best = 1e9;
        for (int rep = 0; rep < 5; rep++) {
          id<MTLCommandBuffer> cb = [C.q commandBuffer];
          id<MTLComputeCommandEncoder> e = conc ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] : [cb computeCommandEncoder];
          [e setComputePipelineState:p];
          [e setBuffer:b offset:0 atIndex:0];
          for (int i = 0; i < NS; i++) {
            if (conc && i) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [e dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          }
          [e endEncoding];
          [cb commit];
          [cb waitUntilCompleted];
          best = std::min(best, mtl::gpu_seconds(cb));
        }
        printf("%s encoder, %5u threadgroups x 256: %.2f us per dependent dispatch\n", conc ? "concurrent+barrier" : "serial            ", tgs, best / NS * 1e6);
      }
  }
}
