// DankAttn (Inkling-2-Mini) CLI driver: test, vtest, gen, bench.
#import "inkling2.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static std::vector<uint32_t> read_ids_bin(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) mtl::die(("cannot read " + p).c_str());
  size_t sz = size_t(f.tellg());
  f.seekg(0);
  std::vector<uint32_t> ids(sz / 4);
  f.read((char*)ids.data(), sz);
  return ids;
}

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 3) {
      fprintf(stderr,
              "Usage:\n"
              "  %s <pack> smoke                  # 5-token Paris check + 32-token greedy gen\n"
              "  %s <pack> vtest                  # verify M=1 vs M=5 bit-exactness\n"
              "  %s <pack> bench <n_ctx> <n_gen>  # decode speed benchmark\n"
              "  %s <pack> gen <ids.bin> <n_gen>  # greedy generation from int32 ids.bin\n"
              "Env: INK_CAP=32768  INK_PASSES=3\n",
              argv[0], argv[0], argv[0], argv[0]);
      return 1;
    }
    std::string pack = argv[1];
    std::string cmd = argv[2];
    uint32_t cap = getenv("INK_CAP") ? uint32_t(atoi(getenv("INK_CAP"))) : 32768u;
    uint32_t passes = getenv("INK_PASSES") ? uint32_t(atoi(getenv("INK_PASSES"))) : 0u;

    ink2::Model M;
    M.load(pack, cap, passes);

    // Default 5-token prompt: "The capital of France is"
    std::vector<uint32_t> paris_ids = {371, 151451, 149879, 151194, 149884};

    if (cmd == "smoke") {
      double ms = 0;
      std::vector<float> logits;
      auto out = M.step(paris_ids, &ms, &logits);
      printf("prefill 5 tok: %.2f ms GPU, next_tok=%u (expected 152717=' Paris')\n", ms, out.back());
      // Print top-5 logits
      std::vector<std::pair<float, uint32_t>> top;
      top.reserve(logits.size());
      for (uint32_t i = 0; i < logits.size(); i++) top.push_back({logits[i], i});
      std::partial_sort(top.begin(), top.begin() + 5, top.end(),
                        [](auto& a, auto& b) { return a.first > b.first; });
      for (int i = 0; i < 5; i++) printf("  top%d: id=%u logit=%.3f\n", i + 1, top[i].second, top[i].first);

      uint32_t tok = out.back();
      double tot_ms = 0;
      printf("gen 32 tokens:");
      for (int i = 0; i < 32; i++) {
        printf(" %u", tok);
        double dms = 0;
        tok = M.step({tok}, &dms)[0];
        tot_ms += dms;
      }
      printf("\ndecode 32 tok: %.2f ms/tok GPU (%.1f tok/s)\n", tot_ms / 32.0, 32000.0 / tot_ms);
      return 0;
    }

    if (cmd == "vtest") {
      std::vector<float> l_batch, l_seq;
      auto ob = M.step(paris_ids, nullptr, &l_batch);
      M.reset();
      uint32_t last_seq = 0;
      for (size_t i = 0; i < paris_ids.size(); i++) {
        last_seq = M.step({paris_ids[i]}, nullptr, (i + 1 == paris_ids.size()) ? &l_seq : nullptr)[0];
      }
      float max_diff = 0.0f;
      for (size_t i = 0; i < l_batch.size(); i++) max_diff = std::max(max_diff, std::fabs(l_batch[i] - l_seq[i]));
      printf("vtest: batch_tok=%u seq_tok=%u max_logit_diff=%.3e (%s)\n",
             ob.back(), last_seq, max_diff, (ob.back() == last_seq && max_diff == 0.0f) ? "BIT-EXACT" : "DIFF");
      return (ob.back() == last_seq) ? 0 : 1;
    }

    if (cmd == "dprof") {
      uint32_t n_ctx = argc > 3 ? uint32_t(atoi(argv[3])) : 128u;
      std::vector<uint32_t> chunk(16);
      for (uint32_t i = 0; i < n_ctx; i += 16) {
        uint32_t n = std::min(16u, n_ctx - i);
        chunk.assign(n, 371u);
        for (uint32_t j = 0; j < n; j++) chunk[j] = paris_ids[(i + j) % paris_ids.size()];
        M.step(chunk);
      }
      // Warmup 4 decode steps
      uint32_t tok = 152717;
      for (int i = 0; i < 4; i++) tok = M.step({tok})[0];
      double unprof_ms = 0;
      tok = M.step({tok}, &unprof_ms)[0];
      std::map<std::string, double> prof;
      M.step({tok}, nullptr, nullptr, &prof);
      std::vector<std::pair<double, std::string>> rows;
      double tot = 0;
      for (auto& [k, v] : prof) { rows.push_back({v, k}); tot += v; }
      std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.first > b.first; });
      printf("dprof (ctx=%u, unprofiled=%.2f ms = %.1f tok/s, profiled_sum=%.2f ms):\n",
             n_ctx, unprof_ms, 1000.0 / unprof_ms, tot);
      for (auto& [v, k] : rows) printf("  %-14s %6.2f ms (%4.1f%%)\n", k.c_str(), v, 100.0 * v / tot);
      return 0;
    }

    if (cmd == "vbench") {
      uint32_t n_ctx = argc > 3 ? uint32_t(atoi(argv[3])) : 512u;
      for (uint32_t M_test : {1u, 4u, 8u, 16u}) {
        M.reset();
        std::vector<uint32_t> chunk(16);
        for (uint32_t i = 0; i < n_ctx; i += 16) {
          uint32_t n = std::min(16u, n_ctx - i);
          chunk.assign(n, 371u);
          for (uint32_t j = 0; j < n; j++) chunk[j] = paris_ids[(i + j) % paris_ids.size()];
          M.step(chunk);
        }
        std::vector<uint32_t> row(M_test, 152717u);
        // Warmup 2 steps
        M.step(row);
        M.step(row);
        double tot = 0;
        const int iters = 16;
        for (int i = 0; i < iters; i++) {
          double ms = 0;
          M.step(row, &ms);
          tot += ms;
        }
        printf("vbench (ctx=%u, M=%2u): %6.2f ms/step GPU (%6.1f tok/s if all accepted)\n",
               n_ctx, M_test, tot / iters, 1000.0 * M_test * iters / tot);
      }
      return 0;
    }

    if (cmd == "bench") {
      uint32_t n_ctx = argc > 3 ? uint32_t(atoi(argv[3])) : 512u;
      uint32_t n_gen = argc > 4 ? uint32_t(atoi(argv[4])) : 64u;
      std::vector<uint32_t> chunk(16);
      for (uint32_t i = 0; i < n_ctx; i += 16) {
        uint32_t n = std::min(16u, n_ctx - i);
        chunk.assign(n, 371u);
        for (uint32_t j = 0; j < n; j++) chunk[j] = paris_ids[(i + j) % paris_ids.size()];
        M.step(chunk);
      }
      double tot_ms = 0;
      auto t0 = std::chrono::steady_clock::now();
      uint32_t tok = 152717;
      for (uint32_t i = 0; i < n_gen; i++) {
        double ms = 0;
        tok = M.step({tok}, &ms)[0];
        tot_ms += ms;
      }
      double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      printf("bench (ctx=%u, gen=%u, passes=%u): %.2f ms/tok GPU (%.1f tok/s GPU, %.1f tok/s wall)\n",
             n_ctx, n_gen, M.num_passes, tot_ms / n_gen, 1000.0 * n_gen / tot_ms, 1000.0 * n_gen / wall_ms);
      return 0;
    }

    if (cmd == "gen") {
      if (argc < 5) mtl::die("gen requires <ids.bin> <n_gen>");
      auto ids = read_ids_bin(argv[3]);
      uint32_t n_gen = uint32_t(atoi(argv[4]));
      uint32_t tok = 0;
      for (size_t i = 0; i < ids.size(); i += 16) {
        size_t n = std::min<size_t>(16, ids.size() - i);
        std::vector<uint32_t> ch(ids.begin() + i, ids.begin() + i + n);
        tok = M.step(ch).back();
      }
      double tot_ms = 0;
      for (uint32_t i = 0; i < n_gen; i++) {
        printf("%u%s", tok, (i + 1 == n_gen) ? "\n" : " ");
        double ms = 0;
        tok = M.step({tok}, &ms)[0];
        tot_ms += ms;
      }
      fprintf(stderr, "gen %u tok: %.2f ms/tok GPU (%.1f tok/s)\n", n_gen, tot_ms / n_gen, 1000.0 * n_gen / tot_ms);
      return 0;
    }

    mtl::die(("unknown command: " + cmd).c_str());
  }
}
