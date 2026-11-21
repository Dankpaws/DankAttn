// mmp's sharing pattern alone: 8 simdgroups each stage one slice into threadgroup memory, a
// threadgroup barrier, then every simdgroup reads all 8 slices; repeated over 64 chunks. No matrix
// units, no device atomics. Deterministic by construction; run plain and under MTL_SHADER_VALIDATION=1.
//   bench/tgbar [reps] [mpp 0|1]
#include "../engine/mtl.h"
#include <cstring>
#include <random>

static const char* SRC = R"(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
kernel void tgb(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                ushort l [[thread_index_in_simdgroup]]) {
  threadgroup float T[8][32][8];
  float acc = 0.0f;
  for (uint kc = 0; kc < 64; kc++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < 8; i++) T[sg][l][i] = in[((size_t(tg) * 64 + kc) * 8 + sg) * 256 + l * 8 + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int u = 0; u < 8; u++)
      for (int i = 0; i < 8; i++) acc = fma(T[u][l][i], 1.0001f, acc);
  }
  out[tg * 256 + sg * 32 + l] = acc;
}
// same staging, but the shared slice feeds the 16x32x32 matrix unit as in mmp
kernel void tgm(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                ushort l [[thread_index_in_simdgroup]]) {
  threadgroup bfloat T[8][32][32];
  constexpr auto desc = matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto xa = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto wb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto c = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), float>();
  for (int i = 0; i < 16; i++) { c[i] = 0.0f; xa[i] = bfloat(in[(size_t(tg) * 8 + sg) * 512 + l * 16 + i]); }
  for (uint kc = 0; kc < 64; kc++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < 32; i++) T[sg][l][i] = bfloat(in[((size_t(tg) * 64 + kc) * 8 + sg) * 1024 + l * 32 + i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int u = 0; u < 8; u++) {
      for (int i = 0; i < 32; i++) wb[i] = T[u][l][i];
      op.run(xa, wb, c);
    }
  }
  for (int i = 0; i < 16; i++) out[(size_t(tg) * 8 + sg) * 512 + l * 16 + i] = c[i];
}
)";

int main(int argc, char** argv) {
  @autoreleasepool {
    mtl::Ctx C;
    const int reps = argc > 1 ? atoi(argv[1]) : 100;
    const bool mpp = argc > 2 && atoi(argv[2]);
    const uint32_t NTG = 512;
    id<MTLComputePipelineState> p = C.pipe(SRC, mpp ? "tgm" : "tgb");
    const size_t nin = size_t(NTG) * 64 * 8 * (mpp ? 1024 : 256), nout = size_t(NTG) * 8 * (mpp ? 512 : 32);
    id<MTLBuffer> in = C.buf(nin * 4), out = C.buf(nout * 4);
    std::mt19937 rng(5);
    for (size_t i = 0; i < nin; i++) ((float*)in.contents)[i] = (int(rng() % 2001) - 1000) / 1000.0f;
    std::vector<float> ref;
    int bad = 0;
    for (int r = 0; r < reps; r++) {
      memset(out.contents, 0xff, out.length);
      id<MTLCommandBuffer> cb = [C.q commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:p];
      [e setBuffer:in offset:0 atIndex:0];
      [e setBuffer:out offset:0 atIndex:1];
      [e dispatchThreadgroups:MTLSizeMake(NTG, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
      if (cb.error) { fprintf(stderr, "command buffer error: %s\n", cb.error.localizedDescription.UTF8String); return 2; }
      std::vector<float> y((float*)out.contents, (float*)out.contents + nout);
      if (r == 0) ref = y; else if (memcmp(y.data(), ref.data(), y.size() * 4)) bad++;
    }
    printf("tgbar %s: %d/%d runs differ from run 0\n", mpp ? "staged slice -> matrix unit (mmp-like)" : "staged slice -> scalar fma", bad, reps - 1);
  }
}
