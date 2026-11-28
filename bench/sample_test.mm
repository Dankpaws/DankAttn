// GPU unit test & latency benchmark for engine/core/gpu_sample.h (dank_sample_masked_lp)
// vs CPU reference engine/core/sample.h across V = 200,704 logits (greedy, masked OPMK,
// temperature+top_k+min_p, and logprobs + top-20 logprobs).
#include "../engine/core/gpu_sample.h"
#include "../engine/core/pack.h"
#include "../engine/core/sample.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main() {
  mtl::Ctx C;
  dankattn::GpuSampler gs;

  constexpr uint32_t V = 200704;
  id<MTLBuffer> lg_buf = C.buf(size_t(V) * 4u);
  float* lg = (float*)lg_buf.contents;
  std::mt19937 rng(42);
  std::normal_distribution<float> nd(0.0f, 3.0f);
  for (uint32_t i = 0; i < V; i++) lg[i] = nd(rng);
  // Plant distinct high logits + an exact tie to exercise tie-breaking
  lg[12345] = 18.5f;
  lg[54321] = 18.5f;
  lg[99999] = 17.2f;
  lg[77777] = 16.8f;

  std::vector<uint32_t> mask_ids;
  for (uint32_t i = 0; i < V; i += 17) mask_ids.push_back(i);
  mask_ids.push_back(54321);
  mask_ids.push_back(99999);

  auto check = [&](const char* name, const std::vector<uint32_t>* allowed,
                   float temp, float top_p, uint32_t top_k, float min_p,
                   uint64_t seed, uint32_t pos, bool want_lp, uint32_t n_top) {
    float cpu_lp = 0, gpu_lp = 0;
    std::vector<std::pair<uint32_t, float>> cpu_tops, gpu_tops;
    uint32_t ct = dankattn::sample::sample_and_logprobs(
        lg, V, allowed, temp, top_p, top_k, min_p, seed, pos, want_lp, n_top, cpu_lp, cpu_tops);
    uint32_t gt = gs.sample(
        C, lg_buf, 0, V, allowed, temp, top_p, top_k, min_p, seed, pos, want_lp, n_top, gpu_lp, gpu_tops);
    if (ct != gt) {
      fprintf(stderr, "FAIL %s: tok cpu=%u gpu=%u\n", name, ct, gt);
      exit(1);
    }
    if (want_lp) {
      const float lp_diff = std::fabs(cpu_lp - gpu_lp);
      if (lp_diff > 1e-4f) {
        fprintf(stderr, "FAIL %s: chosen_lp cpu=%.6f gpu=%.6f diff=%.6f\n", name, cpu_lp, gpu_lp, lp_diff);
        exit(1);
      }
      if (cpu_tops.size() != gpu_tops.size()) {
        fprintf(stderr, "FAIL %s: n_top cpu=%zu gpu=%zu\n", name, cpu_tops.size(), gpu_tops.size());
        exit(1);
      }
      for (size_t i = 0; i < cpu_tops.size(); i++) {
        if (cpu_tops[i].first != gpu_tops[i].first || std::fabs(cpu_tops[i].second - gpu_tops[i].second) > 1e-4f) {
          fprintf(stderr, "FAIL %s top[%zu]: cpu=(%u,%.6f) gpu=(%u,%.6f)\n", name, i,
                  cpu_tops[i].first, cpu_tops[i].second, gpu_tops[i].first, gpu_tops[i].second);
          exit(1);
        }
      }
    }
    printf("PASS %-24s tok=%u lp=%.4f tops=%zu\n", name, gt, gpu_lp, gpu_tops.size());
  };

  check("greedy_unmasked_lp20", nullptr, 0.0f, 1.0f, 0, 0.0f, 1, 10, true, 20);
  check("greedy_masked_lp10",   &mask_ids, 0.0f, 1.0f, 0, 0.0f, 1, 10, true, 10);
  check("temp_topk_unmasked",   nullptr, 0.8f, 1.0f, 40, 0.02f, 12345, 77, true, 5);
  check("temp_topk_masked",     &mask_ids, 0.8f, 1.0f, 40, 0.02f, 12345, 77, true, 5);
  check("temp_topp_masked",     &mask_ids, 0.7f, 0.9f, 0, 0.01f, 999, 42, true, 5);

  // Latency benchmark: CPU vs GPU (unmasked + logprobs top-20, and masked OPMK)
  constexpr int ITERS = 50;
  float lp = 0;
  std::vector<std::pair<uint32_t, float>> tops;
  double t0 = dankattn::host_now();
  for (int i = 0; i < ITERS; i++)
    dankattn::sample::sample_and_logprobs(lg, V, nullptr, 0.8f, 0.95f, 40, 0.02f, 42, uint32_t(i), true, 20, lp, tops);
  double cpu_lp_ms = (dankattn::host_now() - t0) * 1e3 / ITERS;

  t0 = dankattn::host_now();
  for (int i = 0; i < ITERS; i++)
    gs.sample(C, lg_buf, 0, V, nullptr, 0.8f, 0.95f, 40, 0.02f, 42, uint32_t(i), true, 20, lp, tops);
  double gpu_lp_ms = (dankattn::host_now() - t0) * 1e3 / ITERS;

  t0 = dankattn::host_now();
  for (int i = 0; i < ITERS; i++)
    dankattn::sample::sample_and_logprobs(lg, V, &mask_ids, 0.0f, 1.0f, 0, 0.0f, 42, uint32_t(i), false, 0, lp, tops);
  double cpu_mk_ms = (dankattn::host_now() - t0) * 1e3 / ITERS;

  t0 = dankattn::host_now();
  for (int i = 0; i < ITERS; i++)
    gs.sample(C, lg_buf, 0, V, &mask_ids, 0.0f, 1.0f, 0, 0.0f, 42, uint32_t(i), false, 0, lp, tops);
  double gpu_mk_ms = (dankattn::host_now() - t0) * 1e3 / ITERS;

  printf("BENCH V=%u (temp+top_p+lp20): CPU=%.3f ms  GPU=%.3f ms  (%.1fx faster)\n",
         V, cpu_lp_ms, gpu_lp_ms, cpu_lp_ms / std::max(gpu_lp_ms, 1e-6));
  printf("BENCH V=%u (OPMK masked greedy): CPU=%.3f ms  GPU=%.3f ms  (%.1fx faster)\n",
         V, cpu_mk_ms, gpu_mk_ms, cpu_mk_ms / std::max(gpu_mk_ms, 1e-6));
  printf("SAMPLE_TEST_OK\n");
  return 0;
}
