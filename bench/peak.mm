// Peak matrix-unit throughput: each simdgroup runs `iters` dependent-free 16x32x32
// matmul2d ops from registers (NACC independent accumulators). Reports TFLOPS (bf16, int8).
#include "../engine/mtl.h"

static const char* kSrc = R"(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
kernel void peak(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                 ushort l [[thread_index_in_simdgroup]], uint g [[threadgroup_position_in_grid]]) {
  constexpr auto desc = matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto a = op.template get_left_input_cooperative_tensor<AT, AT, CT>();
  auto b = op.template get_right_input_cooperative_tensor<AT, AT, CT>();
  auto c0 = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), CT>();
  auto c1 = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), CT>();
  auto c2 = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), CT>();
  auto c3 = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), CT>();
  for (int i = 0; i < 16; i++) a[i] = AT(int(l + i) % 3);
  for (int i = 0; i < 32; i++) b[i] = AT(int(l * i) % 3);
  for (int i = 0; i < 16; i++) { c0[i] = 0; c1[i] = 0; c2[i] = 0; c3[i] = 0; }
  for (uint t = 0; t < iters; t++) {
    op.run(a, b, c0); op.run(a, b, c1); op.run(a, b, c2); op.run(a, b, c3);
  }
  float s = 0;
  for (int i = 0; i < 16; i++) s += float(c0[i]) + float(c1[i]) + float(c2[i]) + float(c3[i]);
  out[g * 32 + l] = s;
}
)";

int main() {
  @autoreleasepool {
    mtl::Ctx C;
    struct { const char* nm; std::map<std::string, std::string> d; } cfg[] = {
        {"bf16->f32", {{"AT", "bfloat"}, {"CT", "float"}}},
        {"f16->f32", {{"AT", "half"}, {"CT", "float"}}},
        {"int8->i32", {{"AT", "int8_t"}, {"CT", "int32_t"}}}};
    for (auto& c : cfg) {
      id<MTLComputePipelineState> p = C.pipe(kSrc, "peak", c.d);
      for (uint32_t sgs : {1u, 4u, 8u}) {
        const uint32_t iters = 4096, ntg = 4096;
        id<MTLBuffer> o = C.buf(size_t(ntg) * sgs * 32 * 4);
        double best = 1e9;
        for (int rep = 0; rep < 3; rep++) {
          id<MTLCommandBuffer> cb = [C.q commandBuffer];
          id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
          [e setComputePipelineState:p];
          [e setBuffer:o offset:0 atIndex:0];
          [e setBytes:&iters length:4 atIndex:1];
          [e dispatchThreadgroups:MTLSizeMake(ntg, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * sgs, 1, 1)];
          [e endEncoding];
          [cb commit];
          [cb waitUntilCompleted];
          best = std::min(best, mtl::gpu_seconds(cb));
        }
        const double flops = double(ntg) * sgs * iters * 4 * 16 * 32 * 32 * 2;
        printf("%-10s %u sg/tg: %.1f TFLOPS (%.2f ms)\n", c.nm, sgs, flops / best / 1e12, best * 1e3);
      }
    }
  }
}
