// Determinism of mm2's split-K reduce in isolation: the hc down shape (K 10240 -> N 352, W8 g128,
// 10 K splits x 8 simdgroups) run R times on fixed random inputs; every output must equal the first,
// bit for bit. red 0 = the atomic last-arrival reduce (what decode uses), red 2 = partials joined by a
// second dispatch (OPUS_RED2), as a control. Run plain and under MTL_SHADER_VALIDATION=1.
//   bench/valrace [kernel tree] [reps] [M]
#include "../engine/mtl.h"
#include <random>
#include <cstring>

struct MmArgs { uint32_t N, K, M, ksplit, wstride, sstride, xstride, pstride, ystride, sgs, ldx; float xdiv; uint32_t psplit, R, H, HC; uint32_t pf_z0 = ~0u, pf_gx = 0, pf_gy = 0, pf_tpt = 0, pf_n = 0, red = 0, rm = 0; };

int main(int argc, char** argv) {
  @autoreleasepool {
    mtl::Ctx C;
    const std::string tree = argc > 1 ? argv[1] : ".";
    const int reps = argc > 2 ? atoi(argv[2]) : 300;
    const uint32_t M = argc > 3 ? atoi(argv[3]) : 1;
    const uint32_t N = 352, K = 10240, GSZ = 128, ksplit = 10, sgs = 8;
    std::string src = mtl::slurp(tree + "/engine/kernels/mm2.metal");
    id<MTLComputePipelineState> p = C.pipe(src, "mm2", {{"WBITS", "8"}, {"GS", "128"}});
    std::mt19937 rng(11);
    auto fill = [&](id<MTLBuffer> b) { uint32_t* u = (uint32_t*)b.contents; for (size_t i = 0; i < b.length / 4; i++) u[i] = rng(); };
    id<MTLBuffer> W = C.buf(size_t(N) * K), S = C.buf(size_t(N) * (K / GSZ) * 2), X = C.buf(size_t(16) * K * 2);
    fill(W);
    {  // scales: small positive halves; activations: bf16 in [-1, 1)
      uint16_t* s = (uint16_t*)S.contents;
      for (size_t i = 0; i < S.length / 2; i++) s[i] = 0x2000 + (rng() & 0x3ff);
      uint16_t* x = (uint16_t*)X.contents;
      for (size_t i = 0; i < X.length / 2; i++) { float f = (int(rng() % 2001) - 1000) / 1000.0f; uint32_t b; memcpy(&b, &f, 4); x[i] = uint16_t(b >> 16); }
    }
    id<MTLBuffer> P = C.buf(size_t(ksplit) * 16 * N * 4), Y = C.buf(size_t(16) * N * 4), cnt = C.buf(N / 32 * 4 * 16);
    id<MTLBuffer> dummy = C.buf(1 << 16);
    memset(cnt.contents, 0, cnt.length);
    auto run = [&](uint32_t red, std::vector<float>& out) {
      MmArgs a{N, K, M, ksplit, 0, 0, 0, 0, 0, sgs, 0, 1.0f, 16 * N, 0, 0, 0};
      id<MTLCommandBuffer> cb = [C.q commandBuffer];
      auto enc = [&](uint32_t r, MTLSize grid, MTLSize tpt, size_t tgm) {
        a.red = r;
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p];
        id<MTLBuffer> bs[16] = {W, S, X, P, Y, nil, dummy, dummy, cnt, dummy, dummy, dummy, dummy, dummy, dummy, dummy};
        for (int i = 0; i < 16; i++) if (i != 5) [e setBuffer:bs[i] offset:0 atIndex:i];
        [e setBytes:&a length:sizeof a atIndex:5];
        [e setThreadgroupMemoryLength:tgm atIndex:0];
        [e dispatchThreadgroups:grid threadsPerThreadgroup:tpt];
        [e endEncoding];
      };
      memset(Y.contents, 0xff, Y.length);
      if (red == 0) {
        enc(0, MTLSizeMake(N / 32, ksplit, 1), MTLSizeMake(32 * sgs, 1, 1), size_t(sgs) * 32 * 16 * 4);
      } else {
        enc(1, MTLSizeMake(N / 32, ksplit, 1), MTLSizeMake(32 * sgs, 1, 1), size_t(sgs) * 32 * 16 * 4);
        enc(2, MTLSizeMake((N / 32 + 7) / 8, 1, 1), MTLSizeMake(256, 1, 1), 16);
      }
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.error) { fprintf(stderr, "command buffer error: %s\n", cb.error.localizedDescription.UTF8String); exit(2); }
      out.assign((float*)Y.contents, (float*)Y.contents + size_t(M) * N);
    };
    if (argc > 4 && std::string(argv[4]) == "mmp") {  // prefill twin: Mp rows of one dispatch, same weights
      const uint32_t Mp = 2048;
      id<MTLComputePipelineState> pp = C.pipe(src, "mmp", {{"WBITS", "8"}, {"GS", "128"}, {"EPI", "0"}});
      id<MTLBuffer> Xp = C.buf(size_t(Mp) * K * 2), Yp = C.buf(size_t(Mp) * N * 4);
      { uint16_t* x = (uint16_t*)Xp.contents;
        for (size_t i = 0; i < Xp.length / 2; i++) { float f = (int(rng() % 2001) - 1000) / 1000.0f; uint32_t b; memcpy(&b, &f, 4); x[i] = uint16_t(b >> 16); } }
      std::vector<float> ref;
      int bad = 0;
      for (int r = 0; r < reps; r++) {
        MmArgs a{N, K, Mp, ksplit, 0, 0, 0, 0, 0, sgs, 0, 1.0f, 0, 0, 0, 0};
        memset(Yp.contents, 0xff, Yp.length);
        id<MTLCommandBuffer> cb = [C.q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pp];
        id<MTLBuffer> bs[12] = {W, S, Xp, P, Yp, nil, dummy, dummy, cnt, dummy, dummy, dummy};
        for (int i = 0; i < 12; i++) if (i != 5) [e setBuffer:bs[i] offset:0 atIndex:i];
        [e setBytes:&a length:sizeof a atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(N / 32, (Mp + 127) / 128, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.error) { fprintf(stderr, "command buffer error: %s\n", cb.error.localizedDescription.UTF8String); exit(2); }
        std::vector<float> y((float*)Yp.contents, (float*)Yp.contents + size_t(Mp) * N);
        if (r == 0) ref = y; else if (memcmp(y.data(), ref.data(), y.size() * 4)) bad++;
      }
      printf("valrace mmp %u rows: %d/%d runs differ from run 0\n", Mp, bad, reps - 1);
      return 0;
    }
    for (uint32_t red : {0u, 2u}) {
      std::vector<float> ref, y;
      run(red, ref);
      int bad = 0, first = -1;
      size_t worst_n = 0;
      for (int r = 1; r < reps; r++) {
        run(red, y);
        size_t nd = 0;
        for (size_t i = 0; i < y.size(); i++) if (memcmp(&y[i], &ref[i], 4)) nd++;
        if (nd) { bad++; if (first < 0) first = r; worst_n = std::max(worst_n, nd); }
      }
      uint32_t cz = 0;
      for (uint32_t i = 0; i < N / 32; i++) cz |= ((uint32_t*)cnt.contents)[i];
      printf("valrace M=%u %s: %d/%d runs differ from run 0 (first at %d, up to %zu of %zu outputs); counters %s\n", M,
             red ? "red2 (two dispatches)" : "atomic last arrival", bad, reps - 1, first, worst_n, ref.size(), cz ? "NOT reset" : "reset");
    }
  }
}
