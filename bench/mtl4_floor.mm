// Head-to-head microbench: Metal 3 (serial / concurrent) vs Metal 4 (MTL4CommandBuffer + MTL4ArgumentTable).
// Measures CPU encode time, GPU/wall dispatch floor, and compute+blit transition overhead over 600 serialized dispatches
// (matching ERNIE-5.5's decode dispatch count on M5 Ultra).
// Build: clang++ -std=c++20 -fobjc-arc -O2 bench/mtl4_floor.mm -framework Metal -framework Foundation -o build/mtl4_floor
#include "../engine/mtl.h"
#include "../engine/core/pack.h"
#include <cstdio>
#include <vector>

static const char* kSrc = R"(
#include <metal_stdlib>
using namespace metal;
kernel void bump(device const float* in [[buffer(0)]],
                 device float* out [[buffer(1)]],
                 constant uint& n [[buffer(2)]],
                 uint tid [[thread_position_in_grid]]) {
  if (tid < n) out[tid] = in[tid] + 1.0f;
}
)";

int main() {
  mtl::Ctx C;
  id<MTLDevice> dev = C.dev;
  id<MTLComputePipelineState> pso = C.pipe(kSrc, "bump");

  constexpr uint32_t N = 4096;
  constexpr uint32_t OPS = 600;
  constexpr int WARMUP = 5, ITERS = 30;

  id<MTLBuffer> b0 = C.buf(N * 4);
  id<MTLBuffer> b1 = C.buf(N * 4);
  id<MTLBuffer> bn = C.buf(16);
  *(uint32_t*)bn.contents = N;
  memset(b0.contents, 0, N * 4);
  memset(b1.contents, 0, N * 4);

  // Residency set for Metal 4
  MTLResidencySetDescriptor* rsd = [MTLResidencySetDescriptor new];
  rsd.label = @"mtl4-floor";
  NSError* err = nil;
  id<MTLResidencySet> rs = [dev newResidencySetWithDescriptor:rsd error:&err];
  if (!rs) mtl::die("residency set", err);
  [rs addAllocation:b0];
  [rs addAllocation:b1];
  [rs addAllocation:bn];
  [rs commit];
  [rs requestResidency];

  // --- 1. Metal 3 serial ---
  auto run_m3 = [&](bool conc) {
    double enc_ms = 0, wall_ms = 0, gpu_ms = 0;
    for (int it = -WARMUP; it < ITERS; it++) {
      double t0 = dankattn::host_now();
      id<MTLCommandBuffer> cb = [C.q commandBuffer];
      id<MTLComputeCommandEncoder> e = conc
          ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
          : [cb computeCommandEncoder];
      [e setComputePipelineState:pso];
      for (uint32_t k = 0; k < OPS; k++) {
        if (conc && k > 0) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [e setBuffer:(k & 1) ? b1 : b0 offset:0 atIndex:0];
        [e setBuffer:(k & 1) ? b0 : b1 offset:0 atIndex:1];
        [e setBuffer:bn offset:0 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(N / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      [e endEncoding];
      double t1 = dankattn::host_now();
      [cb commit];
      [cb waitUntilCompleted];
      double t2 = dankattn::host_now();
      if (it >= 0) {
        enc_ms += (t1 - t0) * 1e3;
        wall_ms += (t2 - t0) * 1e3;
        gpu_ms += (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
      }
    }
    enc_ms /= ITERS; wall_ms /= ITERS; gpu_ms /= ITERS;
    printf("Metal 3 %-10s (%u ops): encode=%.3f ms (%.2f us/op)  gpu=%.3f ms (%.2f us/op)  wall=%.3f ms\n",
           conc ? "concurrent" : "serial", OPS, enc_ms, enc_ms * 1e3 / OPS, gpu_ms, gpu_ms * 1e3 / OPS, wall_ms);
  };

  run_m3(false);
  run_m3(true);

  // --- 2. Metal 4 (MTL4CommandQueue + MTL4CommandAllocator + MTL4ArgumentTable) ---
  id<MTL4CommandQueue> q4 = [dev newMTL4CommandQueue];
  if (!q4) mtl::die("newMTL4CommandQueue");
  [q4 addResidencySet:rs];

  id<MTL4CommandAllocator> alloc = [dev newCommandAllocator];
  id<MTL4CommandBuffer> cb4 = [dev newCommandBuffer];
  id<MTLSharedEvent> ev = [dev newSharedEvent];
  uint64_t ev_val = 0;

  // Pre-bake 2 argument tables (even/odd ping-pong) once outside the hot loop!
  MTL4ArgumentTableDescriptor* atd = [MTL4ArgumentTableDescriptor new];
  atd.maxBufferBindCount = 4;
  id<MTL4ArgumentTable> at_dyn = [dev newArgumentTableWithDescriptor:atd error:&err];
  id<MTL4ArgumentTable> at0 = [dev newArgumentTableWithDescriptor:atd error:&err];
  id<MTL4ArgumentTable> at1 = [dev newArgumentTableWithDescriptor:atd error:&err];
  if (!at_dyn || !at0 || !at1) mtl::die("newArgumentTable", err);
  [at0 setAddress:b0.gpuAddress atIndex:0];
  [at0 setAddress:b1.gpuAddress atIndex:1];
  [at0 setAddress:bn.gpuAddress atIndex:2];
  [at1 setAddress:b1.gpuAddress atIndex:0];
  [at1 setAddress:b0.gpuAddress atIndex:1];
  [at1 setAddress:bn.gpuAddress atIndex:2];

  // Test Metal 4 with barrierAfterEncoderStages (None vs Device visibility, dynamic vs prebaked, 1-CB vs 4-CB early commit)
  auto run_m4 = [&](const char* label, bool prebaked, MTL4VisibilityOptions vis, uint32_t n_cbs) {
    std::vector<id<MTL4CommandAllocator>> allocs(n_cbs);
    std::vector<id<MTL4CommandBuffer>> cbs(n_cbs);
    for (uint32_t c = 0; c < n_cbs; c++) {
      allocs[c] = [dev newCommandAllocator];
      cbs[c] = [dev newCommandBuffer];
    }
    double enc_ms = 0, wall_ms = 0;
    const uint32_t per_cb = (OPS + n_cbs - 1u) / n_cbs;
    for (int it = -WARMUP; it < ITERS; it++) {
      for (uint32_t c = 0; c < n_cbs; c++) [allocs[c] reset];
      double t0 = dankattn::host_now();
      double enc_sum = 0.0;
      for (uint32_t c = 0; c < n_cbs; c++) {
        const uint32_t k0 = c * per_cb;
        const uint32_t k1 = std::min(OPS, k0 + per_cb);
        if (k0 >= k1) break;
        double e0 = dankattn::host_now();
        [cbs[c] beginCommandBufferWithAllocator:allocs[c]];
        id<MTL4ComputeCommandEncoder> e = [cbs[c] computeCommandEncoder];
        [e setComputePipelineState:pso];
        for (uint32_t k = k0; k < k1; k++) {
          if (k > k0) {
            [e barrierAfterEncoderStages:MTLStageDispatch
                     beforeEncoderStages:MTLStageDispatch
                       visibilityOptions:vis];
          }
          if (prebaked) {
            [e setArgumentTable:(k & 1) ? at1 : at0];
          } else {
            [at_dyn setAddress:((k & 1) ? b1 : b0).gpuAddress atIndex:0];
            [at_dyn setAddress:((k & 1) ? b0 : b1).gpuAddress atIndex:1];
            [at_dyn setAddress:bn.gpuAddress atIndex:2];
            [e setArgumentTable:at_dyn];
          }
          [e dispatchThreadgroups:MTLSizeMake(N / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [e endEncoding];
        [cbs[c] endCommandBuffer];
        enc_sum += (dankattn::host_now() - e0) * 1e3;
        id<MTL4CommandBuffer> one[1] = {cbs[c]};
        [q4 commit:one count:1];
        if (c + 1 < n_cbs) {
          const uint64_t mid = ++ev_val;
          [q4 signalEvent:ev value:mid];
          [q4 waitForEvent:ev value:mid];
        }
      }
      const uint64_t target = ++ev_val;
      [q4 signalEvent:ev value:target];
      [ev waitUntilSignaledValue:target timeoutMS:5000];
      double t2 = dankattn::host_now();
      if (it >= 0) {
        enc_ms += enc_sum;
        wall_ms += (t2 - t0) * 1e3;
      }
    }
    enc_ms /= ITERS; wall_ms /= ITERS;
    printf("Metal 4 %-20s (%u ops): encode=%.3f ms (%.2f us/op)  wall=%.3f ms (%.2f us/op)\n",
           label, OPS, enc_ms, enc_ms * 1e3 / OPS, wall_ms, wall_ms * 1e3 / OPS);
    // Verify exactness on the last configuration
    memset(b0.contents, 0, N * 4);
    memset(b1.contents, 0, N * 4);
    for (uint32_t c = 0; c < n_cbs; c++) [allocs[c] reset];
    for (uint32_t c = 0; c < n_cbs; c++) {
      const uint32_t k0 = c * per_cb;
      const uint32_t k1 = std::min(OPS, k0 + per_cb);
      if (k0 >= k1) break;
      [cbs[c] beginCommandBufferWithAllocator:allocs[c]];
      id<MTL4ComputeCommandEncoder> e = [cbs[c] computeCommandEncoder];
      [e setComputePipelineState:pso];
      for (uint32_t k = k0; k < k1; k++) {
        if (k > k0) [e barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:vis];
        [e setArgumentTable:(k & 1) ? at1 : at0];
        [e dispatchThreadgroups:MTLSizeMake(N / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      [e endEncoding];
      [cbs[c] endCommandBuffer];
      id<MTL4CommandBuffer> one[1] = {cbs[c]};
      [q4 commit:one count:1];
      if (c + 1 < n_cbs) {
        const uint64_t mid = ++ev_val;
        [q4 signalEvent:ev value:mid];
        [q4 waitForEvent:ev value:mid];
      }
    }
    const uint64_t vt = ++ev_val;
    [q4 signalEvent:ev value:vt];
    while (ev.signaledValue < vt) { [ev waitUntilSignaledValue:vt timeoutMS:50]; }
    float got = ((float*)b0.contents)[0];
    if (got != float(OPS)) {
      printf("  VERIFY FAIL (%s): b0[0]=%.1f (expected %.1f)\n", label, got, float(OPS));
      exit(1);
    }
  };

  run_m4("dyn (None, 1CB)",       false, MTL4VisibilityOptionNone,   1);
  run_m4("prebaked (None, 1CB)",  true,  MTL4VisibilityOptionNone,   1);
  run_m4("prebaked (Dev, 1CB)",   true,  MTL4VisibilityOptionDevice, 1);
  run_m4("prebaked (None, 4CB)",  true,  MTL4VisibilityOptionNone,   4);
  printf("all verify checks: b0[0]=%.1f OK\n", float(OPS));
  return 0;
}
