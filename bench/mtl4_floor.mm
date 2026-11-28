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
  id<MTL4ArgumentTable> at0 = [dev newArgumentTableWithDescriptor:atd error:&err];
  id<MTL4ArgumentTable> at1 = [dev newArgumentTableWithDescriptor:atd error:&err];
  if (!at0 || !at1) mtl::die("newArgumentTable", err);
  [at0 setAddress:b0.gpuAddress atIndex:0];
  [at0 setAddress:b1.gpuAddress atIndex:1];
  [at0 setAddress:bn.gpuAddress atIndex:2];
  [at1 setAddress:b1.gpuAddress atIndex:0];
  [at1 setAddress:b0.gpuAddress atIndex:1];
  [at1 setAddress:bn.gpuAddress atIndex:2];

  // Test Metal 4 with barrierAfterEncoderStages (same serialization as Metal 3)
  auto run_m4 = [&](bool prebaked) {
    double enc_ms = 0, wall_ms = 0;
    for (int it = -WARMUP; it < ITERS; it++) {
      [alloc reset];
      double t0 = dankattn::host_now();
      [cb4 beginCommandBufferWithAllocator:alloc];
      id<MTL4ComputeCommandEncoder> e = [cb4 computeCommandEncoder];
      [e setComputePipelineState:pso];
      for (uint32_t k = 0; k < OPS; k++) {
        if (k > 0) {
          [e barrierAfterEncoderStages:MTLStageDispatch
                   beforeEncoderStages:MTLStageDispatch
                     visibilityOptions:MTL4VisibilityOptionDevice];
        }
        if (prebaked) {
          [e setArgumentTable:(k & 1) ? at1 : at0];
        } else {
          [at0 setAddress:((k & 1) ? b1 : b0).gpuAddress atIndex:0];
          [at0 setAddress:((k & 1) ? b0 : b1).gpuAddress atIndex:1];
          [at0 setAddress:bn.gpuAddress atIndex:2];
          [e setArgumentTable:at0];
        }
        [e dispatchThreadgroups:MTLSizeMake(N / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      [e endEncoding];
      [cb4 endCommandBuffer];
      double t1 = dankattn::host_now();
      id<MTL4CommandBuffer> cbs[1] = {cb4};
      [q4 commit:cbs count:1];
      const uint64_t target = ++ev_val;
      [q4 signalEvent:ev value:target];
      [ev waitUntilSignaledValue:target timeoutMS:5000];
      double t2 = dankattn::host_now();
      if (it >= 0) {
        enc_ms += (t1 - t0) * 1e3;
        wall_ms += (t2 - t0) * 1e3;
      }
    }
    enc_ms /= ITERS; wall_ms /= ITERS;
    printf("Metal 4 %-10s (%u ops): encode=%.3f ms (%.2f us/op)  wall=%.3f ms (%.2f us/op)\n",
           prebaked ? "prebaked" : "dynamic", OPS, enc_ms, enc_ms * 1e3 / OPS, wall_ms, wall_ms * 1e3 / OPS);
  };

  run_m4(false);
  run_m4(true);

  // Verify correctness: 600 bumps from 0 -> 600.0f
  memset(b0.contents, 0, N * 4);
  memset(b1.contents, 0, N * 4);
  [alloc reset];
  [cb4 beginCommandBufferWithAllocator:alloc];
  id<MTL4ComputeCommandEncoder> e = [cb4 computeCommandEncoder];
  [e setComputePipelineState:pso];
  for (uint32_t k = 0; k < OPS; k++) {
    if (k > 0) [e barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
    [e setArgumentTable:(k & 1) ? at1 : at0];
    [e dispatchThreadgroups:MTLSizeMake(N / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
  [e endEncoding];
  [cb4 endCommandBuffer];
  id<MTL4CommandBuffer> cbs[1] = {cb4};
  [q4 commit:cbs count:1];
  const uint64_t target = ++ev_val;
  [q4 signalEvent:ev value:target];
  while (ev.signaledValue < target) { [ev waitUntilSignaledValue:target timeoutMS:50]; }
  float got = ((float*)b0.contents)[0];
  printf("verify: b0[0]=%.1f (expected %.1f) %s\n", got, float(OPS), got == float(OPS) ? "OK" : "FAIL");
  return got == float(OPS) ? 0 : 1;
}
