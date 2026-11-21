// DankAttn (Inkling-2-Mini) CLI driver: smoke, vtest, dprof, pfbench, spec, bench, vbench, gen.
#import "inkling2.h"
#include "../lookup.h"
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
    bool kv8 = getenv("INK_KV8") ? (atoi(getenv("INK_KV8")) != 0) : true;

    ink2::Model M;
    M.load(pack, cap, passes, kv8);

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
      std::vector<uint32_t> chunk;
      for (uint32_t i = 0; i < n_ctx; i += ink2::MP) {
        uint32_t n = std::min(ink2::MP, n_ctx - i);
        chunk.resize(n);
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

    if (cmd == "pfbench") {
      uint32_t n_tok = argc > 3 ? uint32_t(atoi(argv[3])) : 2048u;
      // 1. Verify M=64 mmp/mmpg prefill against sequential M=1
      std::vector<uint32_t> chk(64);
      for (uint32_t i = 0; i < 64; i++) chk[i] = paris_ids[i % paris_ids.size()];
      std::vector<float> l_pf, l_seq;
      auto opf = M.step(chk, nullptr, &l_pf);
      M.reset();
      uint32_t last_seq = 0;
      for (uint32_t i = 0; i < 64; i++) {
        last_seq = M.step({chk[i]}, nullptr, (i + 1 == 64) ? &l_seq : nullptr)[0];
      }
      float max_diff = 0.0f;
      for (size_t i = 0; i < l_pf.size(); i++) max_diff = std::max(max_diff, std::fabs(l_pf[i] - l_seq[i]));
      printf("pfcheck (M=64 vs seq): pf_tok=%u seq_tok=%u max_logit_diff=%.3e (%s)\n",
             opf.back(), last_seq, max_diff, (opf.back() == last_seq) ? "EXACT TOK" : "DIFF");

      // 2. Benchmark chunked prefill (MP=256)
      M.reset();
      std::vector<uint32_t> chunk;
      double tot_gpu_ms = 0;
      auto t0 = std::chrono::steady_clock::now();
      for (uint32_t i = 0; i < n_tok; i += ink2::MP) {
        uint32_t n = std::min(ink2::MP, n_tok - i);
        chunk.resize(n);
        for (uint32_t j = 0; j < n; j++) chunk[j] = paris_ids[(i + j) % paris_ids.size()];
        double ms = 0;
        M.step(chunk, &ms);
        tot_gpu_ms += ms;
      }
      double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      printf("pfbench (n_tok=%u, chunk=%u): %.1f ms GPU (%.1f tok/s GPU), %.1f ms wall (%.1f tok/s wall)\n",
             n_tok, ink2::MP, tot_gpu_ms, 1000.0 * n_tok / tot_gpu_ms,
             wall_ms, 1000.0 * n_tok / wall_ms);
      return (opf.back() == last_seq) ? 0 : 1;
    }

    if (cmd == "spec") {
      if (!M.has_mtp) mtl::die("spec requires DSpark overlay (mtp_manifest.json)");
      uint32_t n_gen = argc > 3 ? uint32_t(atoi(argv[3])) : 128u;
      std::vector<uint32_t> prompt_ids = (argc > 4 && std::string(argv[4]) != "nocheck")
          ? read_ids_bin(argv[4]) : paris_ids;
      bool verify_exact = (argc <= 5 || std::string(argv[5]) != "nocheck");

      const uint32_t draft_k = getenv("INK_DRAFT_K") ? uint32_t(atoi(getenv("INK_DRAFT_K"))) : 7u;
      const float draft_cut = getenv("INK_DRAFT_CUT") ? float(atof(getenv("INK_DRAFT_CUT"))) : 0.50f;
      const bool use_lookup = getenv("INK_LOOKUP") ? (atoi(getenv("INK_LOOKUP")) != 0) : true;

      // 1. Prefill prompt + commit MTP context KV (chunked by MP=256)
      double pf_gpu_ms = 0;
      uint32_t anchor = 0;
      for (size_t i = 0; i < prompt_ids.size(); i += ink2::MP) {
        size_t n = std::min<size_t>(ink2::MP, prompt_ids.size() - i);
        std::vector<uint32_t> ch(prompt_ids.begin() + i, prompt_ids.begin() + i + n);
        double ms = 0;
        anchor = M.step(ch, &ms).back();
        pf_gpu_ms += ms;
        M.mtp_commit(uint32_t(n));
      }

      std::vector<uint32_t> hist = prompt_ids;
      opus_lookup::Lookup lk;

      std::vector<uint32_t> spec_toks;
      spec_toks.reserve(n_gen + 16);
      spec_toks.push_back(anchor);

      double t_draft = 0, t_verify = 0, t_commit = 0;
      uint32_t rounds = 0, total_acc = 0, total_drafted = 0, lk_hits = 0;
      auto t0 = std::chrono::steady_clock::now();

      while (spec_toks.size() < n_gen) {
        double dms = 0, vms = 0, cms = 0;
        std::vector<uint32_t> drafts;

        // Optional 3-gram context lookup before DSpark
        if (use_lookup && hist.size() >= 4) {
          lk.sync(hist, false);
          std::vector<uint32_t> tail;
          const size_t tlen = std::min<size_t>(hist.size(), 63);
          tail.insert(tail.end(), hist.end() - long(tlen), hist.end());
          tail.push_back(anchor);
          uint32_t mlen = 0;
          uint32_t mpos = lk.find(hist, tail, &mlen);
          if (mpos != opus_lookup::Lookup::NONE && mlen >= 4) {
            const uint32_t cap_lk = std::min<uint32_t>(ink2::MAXM - 1u, uint32_t(hist.size() - mpos));
            for (uint32_t i = 0; i < cap_lk; i++) drafts.push_back(hist[mpos + i]);
            lk_hits++;
          }
        }

        if (drafts.empty()) {
          std::vector<float> dprobs;
          drafts = M.draft(anchor, draft_k, &dprobs, &dms);
          t_draft += dms;
          if (draft_cut > 0.0f) {
            uint32_t keep = 0;
            while (keep < drafts.size() && dprobs[keep] >= draft_cut) keep++;
            drafts.resize(keep);
          }
        }

        // Verify [anchor, d0, ..., d_{k-1}] (M = 1 + drafts.size() <= 16)
        std::vector<uint32_t> v_in;
        v_in.reserve(1 + drafts.size());
        v_in.push_back(anchor);
        for (uint32_t d : drafts) v_in.push_back(d);
        total_drafted += uint32_t(drafts.size());

        auto v_out = M.step(v_in, &vms);
        t_verify += vms;

        // Count longest accepted prefix: v_out[i] predicts the token AFTER v_in[i]
        uint32_t acc = 0;
        while (acc < drafts.size() && v_out[acc] == drafts[acc]) acc++;
        const uint32_t n_adv = acc + 1u;
        M.rollback(n_adv, uint32_t(v_in.size()));
        M.mtp_commit(n_adv, &cms);
        t_commit += cms;

        hist.push_back(anchor);
        for (uint32_t i = 0; i < acc; i++) {
          hist.push_back(drafts[i]);
          if (spec_toks.size() < n_gen) spec_toks.push_back(drafts[i]);
        }
        anchor = v_out[acc];
        if (spec_toks.size() < n_gen) spec_toks.push_back(anchor);
        total_acc += n_adv;
        rounds++;
      }
      double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      uint32_t gen_n = uint32_t(spec_toks.size()) - 1u;
      printf("spec (K=%u cut=%.2f lk=%d | %u tok, %u rounds, %.2f tok/rnd, avg_draft=%.2f, lk_hits=%u): "
             "%.1f tok/s wall (draft=%.2f ms, verify=%.2f ms, commit=%.2f ms | pf=%zu tok in %.1f ms)\n",
             draft_k, draft_cut, use_lookup ? 1 : 0, gen_n, rounds,
             double(total_acc) / rounds, double(total_drafted) / rounds, lk_hits,
             1000.0 * gen_n / wall_ms, t_draft / rounds, t_verify / rounds, t_commit / rounds,
             prompt_ids.size(), pf_gpu_ms);

      if (verify_exact) {
        M.reset();
        uint32_t tok = 0;
        for (size_t i = 0; i < prompt_ids.size(); i += ink2::MP) {
          size_t n = std::min<size_t>(ink2::MP, prompt_ids.size() - i);
          std::vector<uint32_t> ch(prompt_ids.begin() + i, prompt_ids.begin() + i + n);
          tok = M.step(ch).back();
        }
        uint32_t match = 0;
        for (uint32_t i = 0; i < spec_toks.size(); i++) {
          if (tok == spec_toks[i]) match++;
          if (i + 1 < spec_toks.size()) tok = M.step({tok})[0];
        }
        printf("spec exact check: %u/%zu greedy tokens match (%s)\n",
               match, spec_toks.size(), (match == spec_toks.size()) ? "100% EXACT" : "MISMATCH");
        return (match == spec_toks.size()) ? 0 : 1;
      }
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
      std::vector<uint32_t> chunk;
      for (uint32_t i = 0; i < n_ctx; i += ink2::MP) {
        uint32_t n = std::min(ink2::MP, n_ctx - i);
        chunk.resize(n);
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
