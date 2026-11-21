// Where the fixed ~11-15 us of a small cache-resident mm2 dispatch goes (hc.up shape: 320
// threadgroups x 160 threads, 2 k-steps per simdgroup; bench/chain measures 13 us/stage against
// a 2.6 us dispatch floor). A chain of dependent dispatches with the kernel grown one piece at
// a time:
//   V0  read the previous stage's word, write one word            (floor at this grid)
//   V1  + matmul2d op declared, NK op.run on register operands, 16 results written
//   V2  + X (16 bf16) and W (32 bytes, int -> bf16) loaded per k-step from small hot buffers
//   V3  + per-group scale fma into a float accumulator
//   V4  + threadgroup-memory reduce across simdgroups behind a threadgroup barrier
//   V5  + cross-threadgroup atomics (store, fence, counter, last arrival re-reads) [ksplit 2]
// Run from ~/opus-next/src: fixed
#include "../engine/mtl.h"

static const char* kSrc = R"(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct A { uint nk, prev, cur, ntg; };
kernel void stage(device float* out [[buffer(0)]], constant A& a [[buffer(1)]],
                  device const uint4* W [[buffer(2)]], device const bfloat* X [[buffer(3)]],
                  device const half* S [[buffer(4)]], device atomic_uint* cnt [[buffer(5)]],
                  device float* P [[buffer(6)]],
                  threadgroup float* red [[threadgroup(0)]],
                  ushort l [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                  ushort nsg [[simdgroups_per_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
  const float dep = out[a.prev * 65536 * 16];
#if VAR == 0
  if (l == 0 && sg == 0) out[a.cur * 65536 * 16 + g] = dep + 1.0f;
#else
  constexpr auto desc = matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto xa = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto c = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), float>();
  float acc[16];
  for (int i = 0; i < 16; i++) { c[i] = 0.0f; acc[i] = 0.0f; xa[i] = bfloat(dep); }
  for (int i = 0; i < 32; i++) wb[i] = bfloat(float((l + i) % 3));
  const uint base = (g * nsg + sg) & 63u;
  for (uint t = 0; t < a.nk; t++) {
    const uint ks = (base + t) & 63u;
#if VAR >= 2
    {
      const device bfloat* xp = X + (ks * 32 + l) * 16;
      for (int i = 0; i < 16; i++) xa[i] = xp[i];
      const uint4 v0 = W[(ks * 32 + l) * 2], v1 = W[(ks * 32 + l) * 2 + 1];
      const uint w8[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma clang loop unroll(full)
      for (int j = 0; j < 8; j++)
#pragma clang loop unroll(full)
        for (int i = 0; i < 4; i++) wb[4 * j + i] = bfloat(float(int(char((w8[j] >> (8 * i)) & 255u))));
    }
#endif
    op.run(xa, wb, c);
#if VAR >= 3
    if ((t & 1u) == 1u) {
      const device half* sp = S + (ks >> 1) * 32;
      for (int i = 0; i < 16; i++) { acc[i] = fma(c[i], float(sp[i & 7]), acc[i]); c[i] = 0.0f; }
    }
#endif
  }
#if VAR < 3
  for (int i = 0; i < 16; i++) acc[i] = c[i];
#endif
#if VAR >= 4
  for (int i = 0; i < 16; i++) red[(sg * 32 + l) * 16 + i] = acc[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  for (int i = 0; i < 16; i++) {
    float s = 0.0f;
    for (ushort k = 0; k < nsg; k++) s += red[(k * 32 + l) * 16 + i];
    acc[i] = s;
  }
#endif
#if VAR >= 5
  {
    const uint nf = g >> 1, split = g & 1u;
    device atomic_float* Pa = (device atomic_float*)P;
    for (int i = 0; i < 16; i++) atomic_store_explicit(&Pa[split * 65536 * 8 + (nf * 32 + l) * 16 + i], acc[i], memory_order_relaxed);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    simdgroup_barrier(mem_flags::mem_device);
    uint last = 0;
    if (l == 0) last = atomic_fetch_add_explicit(&cnt[nf], 1u, memory_order_relaxed) == 1u;
    last = simd_broadcast(last, 0);
    if (!last) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    for (int i = 0; i < 16; i++)
      acc[i] = atomic_load_explicit(&Pa[(nf * 32 + l) * 16 + i], memory_order_relaxed) +
               atomic_load_explicit(&Pa[65536 * 8 + (nf * 32 + l) * 16 + i], memory_order_relaxed);
    if (l == 0) atomic_store_explicit(&cnt[nf], 0u, memory_order_relaxed);
  }
#endif
  // each simdgroup that reaches here writes its 16 x 32 results
  for (int i = 0; i < 16; i++) out[a.cur * 65536 * 16 + ((g * nsg + sg) * 32 + l) * 16 + i] = acc[i];
#endif
}
)";

int main(int argc, char** argv) {
  @autoreleasepool {
    mtl::Ctx C;
    const int NS = 480;
    id<MTLBuffer> out = C.buf(size_t(8) * 65536 * 16 * 4), W = C.buf(64 * 32 * 32), X = C.buf(64 * 32 * 16 * 2),
                  S = C.buf(64 * 32 * 2), cnt = C.buf(65536 * 4), P = C.buf(size_t(2) * 65536 * 8 * 4);
    for (size_t i = 0; i < W.length; i++) ((uint8_t*)W.contents)[i] = uint8_t(i * 37 + 11);
    for (size_t i = 0; i < X.length / 2; i++) ((uint16_t*)X.contents)[i] = 0x3F80;
    for (size_t i = 0; i < S.length / 2; i++) ((uint16_t*)S.contents)[i] = 0x3C00;
    memset(cnt.contents, 0, cnt.length);
    struct Shape { const char* nm; uint32_t ntg, sgs, nk; } shapes[] = {
        {"hc.up   320 tg x 5 sg, 2 k-steps", 320, 5, 2},
        {"hc.dn   110 tg x 8 sg, 4 k-steps", 110, 8, 4},
        {"moe.in   57 tg x 8 sg, 10 k-steps", 57, 8, 10},
        {"tiny     16 tg x 4 sg, 2 k-steps", 16, 4, 2}};
    for (auto& sh : shapes) {
      printf("%s\n", sh.nm);
      for (int v = 0; v <= 5; v++) {
        id<MTLComputePipelineState> p = C.pipe(kSrc, "stage", {{"VAR", std::to_string(v)}});
        double best = 1e9;
        for (int rep = 0; rep < 6; rep++) {
          id<MTLCommandBuffer> cb = [C.q commandBuffer];
          id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
          [e setComputePipelineState:p];
          [e setBuffer:out offset:0 atIndex:0];
          [e setBuffer:W offset:0 atIndex:2];
          [e setBuffer:X offset:0 atIndex:3];
          [e setBuffer:S offset:0 atIndex:4];
          [e setBuffer:cnt offset:0 atIndex:5];
          [e setBuffer:P offset:0 atIndex:6];
          [e setThreadgroupMemoryLength:size_t(sh.sgs) * 32 * 16 * 4 atIndex:0];
          for (int i = 0; i < NS; i++) {
            struct { uint32_t nk, prev, cur, ntg; } a{sh.nk, uint32_t(i & 7), uint32_t((i + 1) & 7), sh.ntg};
            [e setBytes:&a length:sizeof a atIndex:1];
            [e dispatchThreadgroups:MTLSizeMake(sh.ntg & ~1u, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * sh.sgs, 1, 1)];
          }
          [e endEncoding];
          [cb commit];
          [cb waitUntilCompleted];
          best = std::min(best, mtl::gpu_seconds(cb));
        }
        printf("  V%d  %.2f us per dependent dispatch\n", v, best / NS * 1e6);
      }
    }
  }
}
