// Where the matmul inner loop loses throughput against the matrix-unit peak: one 16x32x32
// matmul2d per k-step, with the operand preparation of the real kernels switched on one piece
// at a time. All buffers are small and stay cache-resident, so this is compute only.
//   LOADX 0/1   left operand from registers / 16 elements per lane loaded per k-step
//   WMODE 0     right operand constant in registers (peak)
//         1     W8 bytes loaded, int -> float -> bfloat per element (mm2, WBITS 8)
//         2     W4 nibbles loaded, int -> float -> bfloat per element (mm2, WBITS 4)
//         3     right operand loaded as stored bfloat (no conversion; 64 bytes per lane)
//         4     W8 bytes loaded and used as int8 directly (int8 x int8 -> i32)
//         5     W4 nibbles loaded, -8, used as int8 directly
//         6     W8 bytes, int8 -> bfloat through as_type of a 256-entry LUT-free bit trick (see below)
//   SCALE 0/1   per-group fma into a float accumulator every 4 k-steps
// Run from ~/opus-next/src: innerloop
#include "../engine/mtl.h"

static const char* kSrc = R"(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#if WMODE == 4 || WMODE == 5
#define AT int8_t
#define CT int32_t
#else
#define AT bfloat
#define CT float
#endif
kernel void inner(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                  device const uint4* W [[buffer(2)]], device const AT* X [[buffer(3)]],
                  device const bfloat* WB [[buffer(4)]], device const half* S [[buffer(5)]],
                  ushort l [[thread_index_in_simdgroup]], uint g [[threadgroup_position_in_grid]]) {
  constexpr auto desc = matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto xa = op.template get_left_input_cooperative_tensor<AT, AT, CT>();
  auto wb = op.template get_right_input_cooperative_tensor<AT, AT, CT>();
  auto c = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wb), CT>();
  for (int i = 0; i < 16; i++) xa[i] = AT(int(l + i) % 3);
  for (int i = 0; i < 32; i++) wb[i] = AT(int(l * i) % 3);
  float acc[16];
  for (int i = 0; i < 16; i++) { c[i] = 0; acc[i] = 0.0f; }
  for (uint t = 0; t < iters; t++) {
    const uint ks = t & 63u;
#if LOADX
    {
      const device AT* xp = X + (ks * 32 + l) * 16;
      for (int i = 0; i < 16; i++) xa[i] = xp[i];
    }
#endif
#if WMODE == 1
    {
      const uint4 v0 = W[(ks * 32 + l) * 2], v1 = W[(ks * 32 + l) * 2 + 1];
      const uint w8[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma clang loop unroll(full)
      for (int j = 0; j < 8; j++)
#pragma clang loop unroll(full)
        for (int i = 0; i < 4; i++) wb[4 * j + i] = bfloat(float(int(char((w8[j] >> (8 * i)) & 255u))));
    }
#elif WMODE == 2
    {
      const uint4 v0 = W[ks * 32 + l];
      const uint w4[4] = {v0.x, v0.y, v0.z, v0.w};
#pragma clang loop unroll(full)
      for (int j = 0; j < 4; j++)
#pragma clang loop unroll(full)
        for (int i = 0; i < 4; i++) {
          wb[8 * j + i] = bfloat(float(int((w4[j] >> (8 * i)) & 15u) - 8));
          wb[8 * j + 4 + i] = bfloat(float(int((w4[j] >> (8 * i + 4)) & 15u) - 8));
        }
    }
#elif WMODE == 3
    {
      const device bfloat* wp = WB + (ks * 32 + l) * 32;
      for (int i = 0; i < 32; i++) wb[i] = wp[i];
    }
#elif WMODE == 4
    {
      const device char* wp = (const device char*)(W + (ks * 32 + l) * 2);
      for (int i = 0; i < 32; i++) wb[i] = wp[i];
    }
#elif WMODE == 5
    {
      const uint4 v0 = W[ks * 32 + l];
      const uint w4[4] = {v0.x, v0.y, v0.z, v0.w};
#pragma clang loop unroll(full)
      for (int j = 0; j < 4; j++) {
        const char4 lo = as_type<char4>((w4[j] & 0x0F0F0F0Fu) - 0x08080808u + ((~w4[j] & 0x08080808u) << 0) * 0u);
        const char4 hi = as_type<char4>(((w4[j] >> 4) & 0x0F0F0F0Fu));
#pragma clang loop unroll(full)
        for (int i = 0; i < 4; i++) { wb[8 * j + i] = char(int(as_type<uchar4>(w4[j] & 0x0F0F0F0Fu)[i]) - 8); wb[8 * j + 4 + i] = char(int(as_type<uchar4>((w4[j] >> 4) & 0x0F0F0F0Fu)[i]) - 8); }
        (void)lo; (void)hi;
      }
    }
#elif WMODE == 6
    {
      // int8 -> bfloat without an int->float convert: integers -128..127 are exact in bf16, and
      // the 256 bit patterns are loaded from a table held in the first 512 bytes of WB.
      const device uchar* wp = (const device uchar*)(W + (ks * 32 + l) * 2);
      for (int i = 0; i < 32; i++) wb[i] = WB[wp[i]];
    }
#endif
    op.run(xa, wb, c);
#if SCALE
    if ((t & 3u) == 3u) {
      const device half* sp = S + (ks >> 2) * 32;
      for (int i = 0; i < 16; i++) { acc[i] = fma(float(c[i]), float(sp[i & 7]), acc[i]); c[i] = 0; }
    }
#endif
  }
  float s = 0;
  for (int i = 0; i < 16; i++) s += float(c[i]) + acc[i];
  out[g * 32 + l] = s;
}
)";

int main() {
  @autoreleasepool {
    mtl::Ctx C;
    const uint32_t iters = 4096, ntg = 8192;
    id<MTLBuffer> o = C.buf(size_t(ntg) * 32 * 4);
    id<MTLBuffer> W = C.buf(64 * 32 * 32), X = C.buf(64 * 32 * 16 * 2), WB = C.buf(64 * 32 * 32 * 2), S = C.buf(16 * 32 * 2);
    for (size_t i = 0; i < W.length; i++) ((uint8_t*)W.contents)[i] = uint8_t(i * 37 + 11);
    for (size_t i = 0; i < X.length / 2; i++) ((uint16_t*)X.contents)[i] = 0x3F80;   // bf16 1.0 (int8 view: small ints)
    for (size_t i = 0; i < WB.length / 2; i++) ((uint16_t*)WB.contents)[i] = 0x3F80;
    for (size_t i = 0; i < S.length / 2; i++) ((uint16_t*)S.contents)[i] = 0x3C00;   // half 1.0
    struct { const char* nm; int loadx, wmode, scale; } cfg[] = {
        {"peak bf16 (registers)", 0, 0, 0},
        {"+ load X", 1, 0, 0},
        {"+ W8 unpack int->bf16", 1, 1, 0},
        {"+ W8 unpack, + group scale", 1, 1, 1},
        {"W4 unpack int->bf16, loadX, scale", 1, 2, 1},
        {"W stored bf16, loadX, scale", 1, 3, 1},
        {"W8 via 256-entry bf16 table, loadX, scale", 1, 6, 1},
        {"peak int8 (registers)", 0, 4, 0},
        {"int8: W8 bytes direct, loadX", 1, 4, 0},
        {"int8: W8 bytes direct, loadX, scale", 1, 4, 1},
        {"int8: W4 nibbles, loadX, scale", 1, 5, 1}};
    for (auto& c : cfg) {
      std::map<std::string, std::string> d{{"LOADX", std::to_string(c.loadx)}, {"WMODE", std::to_string(c.wmode)},
                                           {"SCALE", std::to_string(c.scale)}};
      id<MTLComputePipelineState> p = C.pipe(kSrc, "inner", d);
      double best = 1e9;
      for (int rep = 0; rep < 4; rep++) {
        id<MTLCommandBuffer> cb = [C.q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p];
        [e setBuffer:o offset:0 atIndex:0];
        [e setBytes:&iters length:4 atIndex:1];
        [e setBuffer:W offset:0 atIndex:2];
        [e setBuffer:X offset:0 atIndex:3];
        [e setBuffer:WB offset:0 atIndex:4];
        [e setBuffer:S offset:0 atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(ntg, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        best = std::min(best, mtl::gpu_seconds(cb));
      }
      const double runs = double(ntg) * iters;
      printf("%-44s %6.1f TFLOPS  %5.2f ns per k-step per simdgroup-slot\n", c.nm, runs * 16 * 32 * 32 * 2 / best / 1e12,
             best / iters * 1e9);
    }
  }
}
