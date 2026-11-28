// Isolated MoE bandwidth & geometry probe for engine/kernels/mm2.metal on Apple Silicon.
// Benchmarks exact Hy4 moe_gu (N=3072, K=1536, W4g64, EPI=3) and moe_dn (N=1536, K=1536, W4g128, EPI=0)
// shapes across (sgs, ksplit, nsg, PF) geometries at M=1 (16E), M=5 (56E), M=8 (72E), M=16 (96E),
// verifying 0.000e+00 bit-exact output against the current default config.
#include "../engine/mm_args.h"
#include "../engine/mtl.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
  @autoreleasepool {
    mtl::Ctx C;
    const std::string src = mtl::slurp(mtl::tree_root() + "/engine/kernels/mm2.metal");

    // 128 experts in pool (~300 MB for moe_gu) so rotating expert IDs busts M5 Ultra SLC/L2 completely
    constexpr uint32_t E = 128;
    constexpr uint32_t N_GU = 3072, K_GU = 1536, GS_GU = 64;
    constexpr uint32_t N_DN = 1536, K_DN = 1536, GS_DN = 128;
    constexpr uint32_t MAX_SLOTS = 128;

    const size_t w_gu_e = size_t(N_GU) * K_GU / 2u;  // 2.36 MB/expert
    const size_t s_gu_e = size_t(N_GU) * (K_GU / GS_GU) * 2u;
    id<MTLBuffer> W = C.buf(size_t(E) * w_gu_e);
    id<MTLBuffer> S = C.buf(size_t(E) * s_gu_e);
    id<MTLBuffer> X = C.buf(size_t(MAX_SLOTS) * 16u * K_GU * 2u);
    id<MTLBuffer> P = C.buf(size_t(MAX_SLOTS) * 8u * 16u * N_GU * 4u);
    id<MTLBuffer> Y = C.buf(size_t(MAX_SLOTS) * 16u * N_GU * 4u);
    id<MTLBuffer> E0 = C.buf(size_t(MAX_SLOTS) * 16u * (N_GU / 2u) * 2u);
    id<MTLBuffer> cnt = C.buf(size_t(MAX_SLOTS) * (N_GU / 32u) * 4u);
    id<MTLBuffer> slots = C.buf(size_t(MAX_SLOTS) * 4u);
    id<MTLBuffer> nslots = C.buf(16u);
    id<MTLBuffer> rmask = C.buf(size_t(MAX_SLOTS) * 4u);
    id<MTLBuffer> dummy = C.buf(256u);

    uint32_t* wp = (uint32_t*)W.contents;
    for (size_t i = 0; i < (size_t(E) * w_gu_e) / 4u; i++) wp[i] = uint32_t(i * 2654435761u ^ (i >> 7));
    uint16_t* sp = (uint16_t*)S.contents;
    for (size_t i = 0; i < (size_t(E) * s_gu_e) / 2u; i++) sp[i] = 0x3C00u;  // 1.0 in f16
    uint16_t* xp = (uint16_t*)X.contents;
    for (size_t i = 0; i < (size_t(MAX_SLOTS) * 16u * K_GU); i++) xp[i] = 0x3F80u;  // 1.0 in bf16
    memset(cnt.contents, 0, cnt.length);

    auto run_case = [&](const char* name, uint32_t N, uint32_t K, uint32_t gs, uint32_t epi,
                        uint32_t M, uint32_t ns, uint32_t sgs, uint32_t ksplit, uint32_t nsg,
                        uint32_t pf, uint32_t rm, std::vector<uint8_t>* out_ref, bool check_ref) {
      if ((K / gs) % (sgs * ksplit) != 0) return;
      if (nsg % sgs != 0) return;
      const uint32_t nft = nsg / sgs;
      if ((N / 32u) % nft != 0) return;

      std::map<std::string, std::string> defs = {
          {"WBITS", "4"}, {"GS", std::to_string(gs)}, {"GROUPED", "1"},
          {"EPI", std::to_string(epi)}, {"TILES", "1"}, {"PF", std::to_string(pf)},
      };
      if (epi == 3) defs["SWIGLU_LIMIT"] = "7.0f";
      id<MTLComputePipelineState> pso = C.pipe(src, "mm2", defs);

      uint32_t* sl = (uint32_t*)slots.contents;
      uint32_t* rmp = (uint32_t*)rmask.contents;
      for (uint32_t i = 0; i < ns; i++) {
        sl[i] = (i * 7u + 3u) % E;
        rmp[i] = (M == 1) ? 1u : (1u << (i % M));
      }
      *(uint32_t*)nslots.contents = ns;

      MmArgs a{};
      a.N = N; a.K = K; a.M = M;
      a.ksplit = ksplit;
      a.wstride = uint32_t((size_t(N) * K / 2u) / 16u);
      a.sstride = N * (K / gs);
      a.xstride = (epi == 0) ? (16u * K / 8u) : 0u;
      a.pstride = ksplit * 16u * N;
      a.ystride = 16u * N;
      a.sgs = sgs;
      a.psplit = 16u * N;
      a.R = N / 2u;
      a.rm = rm;

      const uint32_t gx = (N / 32u) / nft;
      const size_t tgm = (sgs > 1) ? (size_t(nsg) * 16u * 32u * 4u) : 16u;

      constexpr int WARM = 3, ITERS = 15;
      double best_s = 1e9, sum_s = 0.0;
      for (int it = -WARM; it < ITERS; it++) {
        for (uint32_t i = 0; i < ns; i++) sl[i] = (uint32_t(it + WARM) * ns + i) % E;
        id<MTLCommandBuffer> cb = [C.q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:W offset:0 atIndex:0];
        [e setBuffer:S offset:0 atIndex:1];
        [e setBuffer:X offset:0 atIndex:2];
        [e setBuffer:P offset:0 atIndex:3];
        [e setBuffer:Y offset:0 atIndex:4];
        [e setBytes:&a length:sizeof(a) atIndex:5];
        [e setBuffer:slots offset:0 atIndex:6];
        [e setBuffer:nslots offset:0 atIndex:7];
        [e setBuffer:cnt offset:0 atIndex:8];
        [e setBuffer:E0 offset:0 atIndex:9];
        [e setBuffer:dummy offset:0 atIndex:10];
        [e setBuffer:dummy offset:0 atIndex:11];
        [e setBuffer:dummy offset:0 atIndex:12];
        [e setBuffer:dummy offset:0 atIndex:13];
        [e setBuffer:dummy offset:0 atIndex:14];
        [e setBuffer:rmask offset:0 atIndex:15];
        [e setThreadgroupMemoryLength:tgm atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(gx, ksplit, ns) threadsPerThreadgroup:MTLSizeMake(32u * nsg, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) mtl::die("moe_bw GPU error", cb.error);
        if (it >= 0) {
          const double s = mtl::gpu_seconds(cb);
          best_s = std::min(best_s, s);
          sum_s += s;
        }
      }
      const double avg_s = sum_s / ITERS;
      const double bytes = double(ns) * (double(N) * K * 0.5 + double(N) * (K / gs) * 2.0);
      const double gbs_avg = bytes / avg_s / 1e9;
      const double gbs_best = bytes / best_s / 1e9;

      const size_t chk_bytes = (epi == 3) ? (size_t(ns) * 16u * (N / 2u) * 2u) : (size_t(ns) * 16u * N * 4u);
      const void* ptr = (epi == 3) ? E0.contents : Y.contents;
      int diffs = 0;
      if (out_ref && !check_ref) {
        out_ref->resize(chk_bytes);
        memcpy(out_ref->data(), ptr, chk_bytes);
      } else if (out_ref && check_ref) {
        diffs = (memcmp(out_ref->data(), ptr, chk_bytes) == 0) ? 0 : 1;
      }
      printf("%-24s (sgs=%u ks=%u nsg=%u pf=%u rm=%u): avg=%.3f ms (%5.0f GB/s) best=%.3f ms (%5.0f GB/s) 42L=%.2f ms%s\n",
             name, sgs, ksplit, nsg, pf, rm, avg_s * 1e3, gbs_avg, best_s * 1e3, gbs_best,
             avg_s * 42.0 * 1e3, check_ref ? (diffs ? " DIFF!" : " ok") : "");
    };

    const std::vector<std::array<uint32_t, 4>> cfgs = {
        // {sgs, ks, nsg, pf}
        {2, 1, 4, 1},  // CURRENT hy4.mm default
        {2, 1, 8, 1},
        {2, 1, 4, 2},
        {2, 1, 8, 2},
        {1, 1, 4, 1},
        {1, 1, 8, 1},
        {1, 1, 4, 2},
        {1, 1, 8, 2},
        {1, 2, 4, 1},
        {4, 1, 4, 1},
        {4, 1, 8, 1},
    };

    printf("=== HY4 moe_gu (N=3072 K=1536 W4g64 EPI=3, rm=1) ===\n");
    for (auto [M, ns] : std::vector<std::pair<uint32_t, uint32_t>>{{1, 16}, {5, 56}, {16, 96}}) {
      printf("-- M=%u (%u experts) --\n", M, ns);
      std::vector<uint8_t> ref;
      for (size_t i = 0; i < cfgs.size(); i++) {
        auto [sgs, ks, nsg, pf] = cfgs[i];
        run_case("moe_gu", N_GU, K_GU, GS_GU, 3, M, ns, sgs, ks, nsg, pf, 1, &ref, i > 0);
      }
    }

    printf("\n=== HY4 moe_dn (N=1536 K=1536 W4g128 EPI=0, rm=1) ===\n");
    for (auto [M, ns] : std::vector<std::pair<uint32_t, uint32_t>>{{1, 16}, {5, 56}, {16, 96}}) {
      printf("-- M=%u (%u experts) --\n", M, ns);
      std::vector<uint8_t> ref;
      for (size_t i = 0; i < cfgs.size(); i++) {
        auto [sgs, ks, nsg, pf] = cfgs[i];
        run_case("moe_dn", N_DN, K_DN, GS_DN, 0, M, ns, sgs, ks, nsg, pf, 1, &ref, i > 0);
      }
    }
  }
  return 0;
}
