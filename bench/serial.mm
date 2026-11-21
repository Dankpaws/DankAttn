// Alternatives to mm2's two reductions for small matmuls, with COLD weights (every stage of
// the chain reads its own weight range, as in a decode step). bench/fixed showed that with hot
// weights the matmul itself is ~4 us, the threadgroup-barrier reduce +3.4-5.6 us and the
// cross-threadgroup atomics +6-15 us. Each stage computes F fragments x SL slices x NK k-steps.
//   A  current layout: threadgroup = SGS simdgroups (one slice each) + barrier reduce; the
//      remaining SL / SGS splits are separate threadgroups joined by atomics
//   B  like A, but the splits are joined by a second tiny dispatch instead of atomics
//   C  serial: one simdgroup per fragment walks all SL slices itself (RING loads in flight),
//      no reduction at all; threadgroups of 8 fragments
//   D  C + PRE extra threadgroups of 1024 threads that only read this stage's weights
//   E  serial per split: threadgroup = SL / SGS simdgroups, each walks SGS slices; barrier reduce
//   F  E + PRE prefetch threadgroups
// Run from ~/opus-next/src: serial
#include "../engine/mtl.h"

static const char* kSrc = R"(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct A { uint nk, prev, cur, wbase, F, SL, SGS, work_tgs, pre_per, mode; };
// mode: 0 = A/B stage 1 (ATOM selects A), 1 = C/D, 2 = E/F, 3 = B's reduce dispatch
kernel void stage(device float* out [[buffer(0)]], constant A& a [[buffer(1)]],
                  device const uint4* W [[buffer(2)]], device const bfloat* X [[buffer(3)]],
                  device const half* S [[buffer(4)]], device atomic_uint* cnt [[buffer(5)]],
                  device float* P [[buffer(6)]],
                  threadgroup float* red [[threadgroup(0)]],
                  ushort l [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                  ushort nsg [[simdgroups_per_threadgroup]], uint g [[threadgroup_position_in_grid]],
                  ushort tid [[thread_index_in_threadgroup]]) {
  const float dep = out[a.prev * 65536 * 16];
  if (g >= a.work_tgs) {  // prefetch threadgroups
    device const uint4* p = W + a.wbase + (size_t(g - a.work_tgs) * 1024 + tid) * a.pre_per;
    uint4 s = uint4(uint(dep) & 1u);
    for (uint i = 0; i < a.pre_per; i++) s += p[i];
    if (s.x == 0xFFFFFFFFu && s.y == 0x9E3779B9u) out[7 * 65536 * 16 + 1] = 1.0f;
    return;
  }
  if (a.mode == 3) {  // reduce dispatch: one simdgroup per fragment sums the split partials
    const uint f = g * nsg + sg, nsp = a.SL / a.SGS;
    for (int i = 0; i < 16; i++) {
      float s = 0.0f;
      for (uint p = 0; p < nsp; p++) s += P[p * 65536 * 8 + (f * 32 + l) * 16 + i];
      out[a.cur * 65536 * 16 + (f * 32 + l) * 16 + i] = s + dep * 0.0f;
    }
    return;
  }
  constexpr auto desc = matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto xa = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto c = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), float>();
  // which fragment / first slice / how many slices this simdgroup walks
  uint f, s0, ns;
  if (a.mode == 0) { const uint nsp = a.SL / a.SGS; f = g / nsp; s0 = (g % nsp) * a.SGS + sg; ns = 1; }
  else if (a.mode == 1) { f = g * nsg + sg; s0 = 0; ns = a.SL; }
  else { f = g; s0 = sg * a.SGS; ns = a.SGS; }
  float tot[16], acc[16];
  for (int i = 0; i < 16; i++) { tot[i] = 0.0f; c[i] = 0.0f; xa[i] = bfloat(dep); }
  const uint nkt = ns * a.nk;
  device const uint4* Wp = W + a.wbase + ((size_t(f) * a.SL + s0) * a.nk * 32 + l) * 2;
  uint4 ring[RING][2];
#pragma clang loop unroll(full)
  for (uint u = 0; u < RING; u++)
    if (u < nkt) { ring[u][0] = Wp[size_t(u) * 64]; ring[u][1] = Wp[size_t(u) * 64 + 1]; }
  for (uint t = 0; t < nkt; t += RING) {
#pragma clang loop unroll(full)
    for (uint u = 0; u < RING; u++) {
      if (t + u >= nkt) break;
      const uint ks = t + u;
      if (ks % a.nk == 0) for (int i = 0; i < 16; i++) acc[i] = 0.0f;
      const device bfloat* xp = X + ((ks & 63u) * 32 + l) * 16;
      for (int i = 0; i < 16; i++) xa[i] = xp[i];
      const uint w8[8] = {ring[u][0].x, ring[u][0].y, ring[u][0].z, ring[u][0].w, ring[u][1].x, ring[u][1].y, ring[u][1].z, ring[u][1].w};
#pragma clang loop unroll(full)
      for (int j = 0; j < 8; j++)
#pragma clang loop unroll(full)
        for (int i = 0; i < 4; i++) wb[4 * j + i] = bfloat(float(int(char((w8[j] >> (8 * i)) & 255u))));
      if (ks + RING < nkt) { ring[u][0] = Wp[size_t(ks + RING) * 64]; ring[u][1] = Wp[size_t(ks + RING) * 64 + 1]; }
      op.run(xa, wb, c);
      if ((ks & 1u) == 1u) {
        const device half* sp = S + ((ks >> 1) & 31u) * 32;
        for (int i = 0; i < 16; i++) { acc[i] = fma(c[i], float(sp[i & 7]), acc[i]); c[i] = 0.0f; }
      }
      if ((ks + 1) % a.nk == 0) for (int i = 0; i < 16; i++) tot[i] += acc[i];
    }
  }
  if (a.mode == 1) {
    for (int i = 0; i < 16; i++) out[a.cur * 65536 * 16 + (f * 32 + l) * 16 + i] = tot[i];
    return;
  }
  for (int i = 0; i < 16; i++) red[(sg * 32 + l) * 16 + i] = tot[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  for (int i = 0; i < 16; i++) {
    float s = 0.0f;
    for (ushort k = 0; k < nsg; k++) s += red[(k * 32 + l) * 16 + i];
    tot[i] = s;
  }
  if (a.mode == 2 || a.SL == a.SGS) {
    for (int i = 0; i < 16; i++) out[a.cur * 65536 * 16 + (f * 32 + l) * 16 + i] = tot[i];
    return;
  }
  const uint nsp = a.SL / a.SGS, split = g % nsp;
#if ATOM
  device atomic_float* Pa = (device atomic_float*)P;
  for (int i = 0; i < 16; i++) atomic_store_explicit(&Pa[split * 65536 * 8 + (f * 32 + l) * 16 + i], tot[i], memory_order_relaxed);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  simdgroup_barrier(mem_flags::mem_device);
  uint last = 0;
  if (l == 0) last = atomic_fetch_add_explicit(&cnt[f], 1u, memory_order_relaxed) == nsp - 1;
  last = simd_broadcast(last, 0);
  if (!last) return;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  for (int i = 0; i < 16; i++) {
    float s = 0.0f;
    for (uint p = 0; p < nsp; p++) s += atomic_load_explicit(&Pa[p * 65536 * 8 + (f * 32 + l) * 16 + i], memory_order_relaxed);
    out[a.cur * 65536 * 16 + (f * 32 + l) * 16 + i] = s;
  }
  if (l == 0) atomic_store_explicit(&cnt[f], 0u, memory_order_relaxed);
#else
  for (int i = 0; i < 16; i++) P[split * 65536 * 8 + (f * 32 + l) * 16 + i] = tot[i];
#endif
}
)";

int main() {
  @autoreleasepool {
    mtl::Ctx C;
    const int NS = 240;
    const size_t STRIDE = size_t(8) << 20;  // bytes of weights reserved per stage
    id<MTLBuffer> out = C.buf(size_t(8) * 65536 * 16 * 4), W = C.buf(STRIDE * NS), X = C.buf(64 * 32 * 16 * 2),
                  S = C.buf(64 * 32 * 2), cnt = C.buf(65536 * 4), P = C.buf(size_t(16) * 65536 * 8 * 4);
    for (size_t i = 0; i < W.length; i += 8) ((uint64_t*)W.contents)[i / 8] = 0x0B25F1A3C7E91D0Bull * (i + 1);
    for (size_t i = 0; i < X.length / 2; i++) ((uint16_t*)X.contents)[i] = 0x3F80;
    for (size_t i = 0; i < S.length / 2; i++) ((uint16_t*)S.contents)[i] = 0x3C00;
    memset(cnt.contents, 0, cnt.length);
    struct A { uint32_t nk, prev, cur, wbase, F, SL, SGS, work_tgs, pre_per, mode; };
    struct Shape { const char* nm; uint32_t F, SL, SGS, nk; } shapes[] = {
        {"hc.up  320 fragments, 5 slices x 2 k-steps (3.3 MB)", 320, 5, 5, 2},
        {"sdown   80 fragments, 5 slices x 4 k-steps (1.6 MB)", 80, 5, 5, 4},
        {"hc.dn   11 fragments, 80 slices x 4 k-steps (3.6 MB)", 11, 80, 8, 4},
        {"moe.in  57 fragments, 20 slices x 4 k-steps (4.7 MB)", 57, 20, 5, 4}};
    auto run = [&](const Shape& sh, int mode, bool atom, bool second, uint32_t ring, uint32_t pre_tgs) {
      id<MTLComputePipelineState> p = C.pipe(kSrc, "stage", {{"RING", std::to_string(ring)}, {"ATOM", atom ? "1" : "0"}});
      const uint32_t bytes = sh.F * sh.SL * sh.nk * 32 * 32;
      double best = 1e9;
      for (int rep = 0; rep < 5; rep++) {
        id<MTLCommandBuffer> cb = [C.q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p];
        [e setBuffer:out offset:0 atIndex:0];
        [e setBuffer:W offset:0 atIndex:2];
        [e setBuffer:X offset:0 atIndex:3];
        [e setBuffer:S offset:0 atIndex:4];
        [e setBuffer:cnt offset:0 atIndex:5];
        [e setBuffer:P offset:0 atIndex:6];
        for (int i = 0; i < NS; i++) {
          uint32_t tgs, tpt;
          if (mode == 0) { tgs = sh.F * (sh.SL / sh.SGS); tpt = 32 * sh.SGS; }
          else if (mode == 1) { tgs = (sh.F + 7) / 8; tpt = 32 * 8; }
          else { tgs = sh.F; tpt = 32 * (sh.SL / sh.SGS); }
          if (mode == 1 && sh.F % 8) { tgs = sh.F; tpt = 32; }
          A a{sh.nk, uint32_t(i & 3), uint32_t((i + 1) & 3), uint32_t(size_t(i) * STRIDE / 16), sh.F, sh.SL, sh.SGS, tgs,
              pre_tgs ? bytes / 16 / (pre_tgs * 1024) : 0, uint32_t(mode)};
          [e setBytes:&a length:sizeof a atIndex:1];
          [e setThreadgroupMemoryLength:size_t(tpt) * 16 * 4 atIndex:0];
          if (pre_tgs) {
            // prefetch threadgroups need 1024 threads: dispatch them with the work grid's
            // thread count rounded up is not possible, so the work threadgroups use 1024 too
            // only when they already do; otherwise prefetch per thread scales with tpt
            a.pre_per = bytes / 16 / (pre_tgs * tpt);
            [e setBytes:&a length:sizeof a atIndex:1];
          }
          [e dispatchThreadgroups:MTLSizeMake(tgs + pre_tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
          if (second) {
            A b = a;
            b.mode = 3;
            b.work_tgs = (sh.F + 7) / 8;
            b.prev = a.cur;
            [e setBytes:&b length:sizeof b atIndex:1];
            [e dispatchThreadgroups:MTLSizeMake(sh.F % 8 ? sh.F : sh.F / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(sh.F % 8 ? 32 : 256, 1, 1)];
          }
        }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        best = std::min(best, mtl::gpu_seconds(cb));
      }
      return best / NS * 1e6;
    };
    for (auto& sh : shapes) {
      printf("%s\n", sh.nm);
      const bool split = sh.SL != sh.SGS;
      printf("  A  current (barrier%s)                 %6.2f us\n", split ? " + atomics" : "", run(sh, 0, true, false, 1, 0));
      printf("  A  current, 4 loads in flight            %6.2f us\n", run(sh, 0, true, false, 4, 0));
      if (split) printf("  B  second dispatch instead of atomics    %6.2f us\n", run(sh, 0, false, true, 1, 0));
      for (uint32_t ring : {4u, 10u, 20u})
        printf("  C  serial per fragment, ring %2u           %6.2f us\n", ring, run(sh, 1, false, false, ring, 0));
      for (uint32_t pre : {64u, 256u})
        printf("  D  C (ring 10) + %3u prefetch tgs         %6.2f us\n", pre, run(sh, 1, false, false, 10, pre));
      if (split) {
        for (uint32_t ring : {4u, 16u})
          printf("  E  serial per split + barrier, ring %2u    %6.2f us\n", ring, run(sh, 2, false, false, ring, 0));
        for (uint32_t pre : {16u, 64u})
          printf("  F  E (ring 16) + %3u prefetch tgs         %6.2f us\n", pre, run(sh, 2, false, false, 16, pre));
      }
    }
  }
}
