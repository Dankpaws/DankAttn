// Precision of one 16x32x32 bf16 matmul2d (f32 destination): f32 A split into bf16 hi (+ lo)
// against an f64 reference, with elements placed by get_multidimensional_index.
#include "../engine/mtl.h"
#include <cmath>
#include <random>

static const char* kSrc = R"(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
// A [16][32] f32, B [32 k][32 n] (bf16-exact f32), C out [3][16][32]: hi, hi+lo, idx check
kernel void prec(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                 device float* C [[buffer(2)]], ushort l [[thread_index_in_simdgroup]]) {
  constexpr auto desc = matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto ah = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto al = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto b = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto c0 = op.template get_destination_cooperative_tensor<decltype(ah), decltype(b), float>();
  auto c1 = op.template get_destination_cooperative_tensor<decltype(ah), decltype(b), float>();
  for (ushort i = 0; i < ah.get_capacity(); i++) {
    auto ix = ah.get_multidimensional_index(i);  // (k, m) or (m, k)?
    const float v = A[MIDX];
    ah[i] = bfloat(v);
    al[i] = bfloat(v - float(bfloat(v)));
  }
  for (ushort i = 0; i < b.get_capacity(); i++) {
    auto ix = b.get_multidimensional_index(i);
    b[i] = bfloat(B[BIDX]);
  }
  for (ushort i = 0; i < c0.get_capacity(); i++) { c0[i] = 0.0f; c1[i] = 0.0f; }
  op.run(ah, b, c0);
  op.run(ah, b, c1);
  op.run(al, b, c1);
  for (ushort i = 0; i < c0.get_capacity(); i++) {
    auto ix = c0.get_multidimensional_index(i);
    C[CIDX] = c0[i];
    C[512 + CIDX] = c1[i];
  }
}
)";

int main() {
  @autoreleasepool {
    mtl::Ctx C;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd;
    std::vector<float> A(16 * 32), B(32 * 32);
    for (auto& x : A) x = nd(rng);
    for (auto& x : B) { float f = nd(rng); uint32_t u; memcpy(&u, &f, 4); u &= 0xFFFF0000u; memcpy(&x, &u, 4); }
    // candidate index conventions: ix[0] is column (x) and ix[1] row (y), or the reverse
    const char* conv[2][3] = {{"ix[1]*32+ix[0]", "ix[1]*32+ix[0]", "ix[1]*32+ix[0]"},
                              {"ix[0]*32+ix[1]", "ix[0]*32+ix[1]", "ix[0]*32+ix[1]"}};
    for (int cv = 0; cv < 2; cv++) {
      NSError* e = nil;
      MTLCompileOptions* o = [MTLCompileOptions new];
      o.languageVersion = MTLLanguageVersion4_0;
      o.preprocessorMacros = @{@"MIDX" : @(conv[cv][0]), @"BIDX" : @(conv[cv][1]), @"CIDX" : @(conv[cv][2])};
      id<MTLLibrary> lib = [C.dev newLibraryWithSource:@(kSrc) options:o error:&e];
      if (!lib) { printf("compile failed: %.400s\n", [[e description] UTF8String]); return 1; }
      id<MTLComputePipelineState> p = [C.dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"prec"] error:&e];
      id<MTLBuffer> bA = C.buf(A.size() * 4), bB = C.buf(B.size() * 4), bC = C.buf(1024 * 4);
      memcpy(bA.contents, A.data(), A.size() * 4);
      memcpy(bB.contents, B.data(), B.size() * 4);
      id<MTLCommandBuffer> cb = [C.q commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:p];
      [enc setBuffer:bA offset:0 atIndex:0];
      [enc setBuffer:bB offset:0 atIndex:1];
      [enc setBuffer:bC offset:0 atIndex:2];
      [enc dispatchThreads:MTLSizeMake(32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      float* r = (float*)bC.contents;
      // try both B orientations for the reference: C = A B (B[k][n]) and A B^T (B[n][k])
      for (int bt = 0; bt < 2; bt++) {
        double e0 = 0, e1 = 0, eh = 0, mx = 0;
        for (int m = 0; m < 16; m++)
          for (int n = 0; n < 32; n++) {
            double s = 0, sh = 0;
            for (int k = 0; k < 32; k++) {
              const double b = bt ? B[n * 32 + k] : B[k * 32 + n];
              s += double(A[m * 32 + k]) * b;
              uint32_t u; float f = A[m * 32 + k]; memcpy(&u, &f, 4);
              u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000u; memcpy(&f, &u, 4);
              sh += double(f) * b;
            }
            mx = fmax(mx, fabs(s));
            e0 = fmax(e0, fabs(r[m * 32 + n] - s));
            e1 = fmax(e1, fabs(r[512 + m * 32 + n] - s));
            eh = fmax(eh, fabs(r[m * 32 + n] - sh));
          }
        printf("conv %d, B%s: max|C| %.3g  hi err %.3g (vs exact-hi ref %.3g)  hi+lo err %.3g\n", cv, bt ? "^T" : "", mx, e0, eh, e1);
      }
    }
  }
}
