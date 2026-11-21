// Dump which (row, col) of each MPP cooperative tensor every simd lane holds, for the
// int8 fragment shapes the register-direct GEMM would use.
#include "../engine/mtl.h"

static const char* kSrc = R"(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
kernel void lay(device int* out [[buffer(0)]], ushort l [[thread_index_in_simdgroup]]) {
  constexpr auto desc = matmul2d_descriptor(FM, FN, FK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto a = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto b = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto c = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), int32_t>();
  int base = 0;
  #define DUMP(t) { for (ushort i = 0; i < 64; i++) { int v = (i < t.get_capacity() && t.is_valid_element(i)); \
     int i0 = -1, i1 = -1; if (v) { auto ix = t.get_multidimensional_index(i); i0 = ix[0]; i1 = ix[1]; } \
     out[base + (l * 64 + i) * 2] = i0; out[base + (l * 64 + i) * 2 + 1] = i1; } base += 32 * 64 * 2; }
  DUMP(a) DUMP(b) DUMP(c)
  out[3 * 32 * 64 * 2 + l * 3 + 0] = a.get_capacity();
  out[3 * 32 * 64 * 2 + l * 3 + 1] = b.get_capacity();
  out[3 * 32 * 64 * 2 + l * 3 + 2] = c.get_capacity();
}
)";

int main(int argc, char** argv) {
  @autoreleasepool {
    mtl::Ctx C;
    int shapes[][3] = {{16, 32, 32}, {32, 16, 32}, {16, 16, 32}, {8, 32, 32}, {32, 32, 32}, {16, 32, 64}};
    for (auto& s : shapes) {
      NSError* e = nil;
      MTLCompileOptions* o = [MTLCompileOptions new];
      o.languageVersion = MTLLanguageVersion4_0;
      o.preprocessorMacros = @{@"FM" : @(s[0]), @"FN" : @(s[1]), @"FK" : @(s[2])};
      id<MTLLibrary> lib = [C.dev newLibraryWithSource:@(kSrc) options:o error:&e];
      if (!lib) { printf("%dx%dx%d: compile failed: %.300s\n", s[0], s[1], s[2], [[e description] UTF8String]); continue; }
      id<MTLComputePipelineState> p = [C.dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"lay"] error:&e];
      if (!p) { printf("%dx%dx%d: pipeline failed\n", s[0], s[1], s[2]); continue; }
      size_t n = 3 * 32 * 64 * 2 + 32 * 3;
      id<MTLBuffer> b = C.buf(n * 4);
      id<MTLCommandBuffer> cb = [C.q commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:p];
      [enc setBuffer:b offset:0 atIndex:0];
      [enc dispatchThreads:MTLSizeMake(32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      int* r = (int*)b.contents;
      int* cap = r + 3 * 32 * 64 * 2;
      printf("== %dx%dx%d caps a=%d b=%d c=%d\n", s[0], s[1], s[2], cap[0], cap[1], cap[2]);
      const char* nm[3] = {"left(MxK)", "right(KxN)", "dest(MxN)"};
      for (int t = 0; t < 3; t++) {
        for (int lane : {0, 1, 2, 3, 4, 8, 16}) {
          printf("  %s lane%-2d:", nm[t], lane);
          for (int i = 0; i < cap[t] && i < 32; i++) {
            int* q = r + t * 32 * 64 * 2 + (lane * 64 + i) * 2;
            printf(" (%d,%d)", q[0], q[1]);
          }
          printf("\n");
        }
      }
    }
  }
}
