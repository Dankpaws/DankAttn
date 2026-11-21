// DankAttn (Inkling-2-Mini) CLI driver: smoke, vtest, dprof, pfbench, spec, bench, vbench, gen.
#import "inkling2.h"
#include "../lookup.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
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
    // Sampling for every lane (INK_TEMP <= 0: greedy). Lane k uses seed INK_SEED + k.
    ink2::Model::Samp gsamp;
    if (getenv("INK_TEMP")) gsamp.temp = float(atof(getenv("INK_TEMP")));
    if (getenv("INK_TOP_P")) gsamp.top_p = float(atof(getenv("INK_TOP_P")));
    if (getenv("INK_TOP_K")) gsamp.top_k = uint32_t(atoi(getenv("INK_TOP_K")));
    if (getenv("INK_MIN_P")) gsamp.min_p = float(atof(getenv("INK_MIN_P")));
    if (getenv("INK_SEED")) gsamp.seed = uint64_t(strtoull(getenv("INK_SEED"), nullptr, 10));
    auto lane_samp = [&](uint32_t k) { ink2::Model::Samp s = gsamp; s.seed += k; return s; };
    M.samp = gsamp;

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

    if (cmd == "ltest" || cmd == "lbench") {
      // ltest <ids.bin> <lanes> <n_gen>: lanes with different prompts decode jointly via step_lanes; each lane's
      //   tokens and per-step logits must match the same sequence run alone with step() (bit-exact).
      //   Then a mixed-width round (lane k feeds 1 + k % 4 known tokens) is checked against single-lane step().
      // lbench <ids.bin> <lanes> <ctx> <n_gen>: every lane prefills ctx tokens, then joint one-row decode.
      if (argc < 6) mtl::die("ltest <ids.bin> <lanes> <n_gen> | lbench <ids.bin> <lanes> <ctx> <n_gen>");
      const bool bench = (cmd == "lbench");
      auto all = read_ids_bin(argv[3]);
      const uint32_t nl = uint32_t(atoi(argv[4]));
      const uint32_t ctx = bench ? uint32_t(atoi(argv[5])) : 0u;
      const uint32_t n_gen = uint32_t(atoi(argv[bench ? 6 : 5]));
      auto prompt = [&](uint32_t k) {
        const uint32_t len = bench ? ctx : 97u + 151u * k;
        std::vector<uint32_t> p(len);
        for (uint32_t i = 0; i < len; i++) p[i] = all[(i + 211u * k) % all.size()];
        return p;
      };
      auto prefill = [&](const std::vector<uint32_t>& p) {
        uint32_t t = 0;
        for (size_t i = 0; i < p.size(); i += ink2::MP) {
          std::vector<uint32_t> c(p.begin() + i, p.begin() + std::min(p.size(), i + ink2::MP));
          t = M.step(c).back();
        }
        return t;
      };
      while (M.lanes_.size() < nl) M.add_lane();
      const uint32_t V = M.vocab_valid;

      std::vector<std::vector<uint32_t>> ref_tok(nl);
      std::vector<std::vector<float>> ref_lg(nl);
      std::vector<std::vector<float>> ref_mix(nl);
      if (!bench) {
        for (uint32_t k = 0; k < nl; k++) {
          M.use_lane(0);
          M.reset();
          uint32_t t = prefill(prompt(k));
          std::vector<float> lg;
          for (uint32_t g = 0; g < n_gen; g++) {
            ref_tok[k].push_back(t);
            t = M.step({t}, nullptr, &lg)[0];
            ref_lg[k].insert(ref_lg[k].end(), lg.begin(), lg.end());
          }
          ref_tok[k].push_back(t);
          // mixed-width round: feed the next (1 + k % 4) tokens of a fixed pattern at once
          std::vector<uint32_t> mix(1 + k % 4);
          for (uint32_t i = 0; i < mix.size(); i++) mix[i] = all[(997u * k + 13u * i) % all.size()];
          std::vector<float> lm;
          M.step(mix, nullptr, &lm);  // last row's logits
          ref_mix[k] = lm;
        }
      }

      for (uint32_t k = 0; k < nl; k++) {
        M.use_lane(int(k));
        M.reset();
      }
      std::vector<uint32_t> cur(nl);
      for (uint32_t k = 0; k < nl; k++) {
        M.use_lane(int(k));
        cur[k] = prefill(prompt(k));
      }
      M.use_lane(0);
      uint32_t bad_tok = 0, bad_lg = 0;
      double gpu_tot = 0;
      auto t0 = std::chrono::steady_clock::now();
      std::vector<float> lg;
      for (uint32_t g = 0; g < n_gen; g++) {
        std::vector<std::pair<int, std::vector<uint32_t>>> b;
        for (uint32_t k = 0; k < nl; k++) {
          if (!bench && cur[k] != ref_tok[k][g]) bad_tok++;
          b.push_back({int(k), {cur[k]}});
        }
        double ms = 0;
        auto out = M.step_lanes(b, &ms, bench ? nullptr : &lg);
        gpu_tot += ms;
        for (uint32_t k = 0; k < nl; k++) {
          if (!bench && memcmp(lg.data() + size_t(k) * V, ref_lg[k].data() + size_t(g) * V, V * 4) != 0) bad_lg++;
          cur[k] = out[k];
        }
      }
      const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (bench && getenv("INK_LPROF")) {
        std::vector<std::pair<int, std::vector<uint32_t>>> b;
        for (uint32_t k = 0; k < nl; k++) b.push_back({int(k), {cur[k]}});
        std::map<std::string, double> prof;
        M.step_lanes(b, nullptr, nullptr, &prof);
        double tot = 0;
        std::vector<std::pair<double, std::string>> rows;
        for (auto& [k, v] : prof) { rows.push_back({v, k}); tot += v; }
        std::sort(rows.begin(), rows.end(), [](auto& x, auto& y) { return x.first > y.first; });
        printf("lprof lanes=%u (profiled_sum=%.2f ms):\n", nl, tot);
        for (auto& [v, k] : rows) printf("  %-14s %6.2f ms (%4.1f%%)\n", k.c_str(), v, 100.0 * v / tot);
      }
      if (bench) {
        printf("lbench lanes=%u ctx=%u gen=%u: %.2f ms/step GPU, %.1f tok/s GPU total (%.1f per lane), %.1f tok/s wall total\n",
               nl, ctx, n_gen, gpu_tot / n_gen, 1000.0 * nl * n_gen / gpu_tot, 1000.0 * n_gen / gpu_tot,
               1000.0 * nl * n_gen / wall);
        return 0;
      }
      uint32_t bad_end = 0;
      for (uint32_t k = 0; k < nl; k++) bad_end += (cur[k] != ref_tok[k][n_gen]);
      // mixed-width joint round
      std::vector<std::pair<int, std::vector<uint32_t>>> b;
      for (uint32_t k = 0; k < nl; k++) {
        std::vector<uint32_t> mix(1 + k % 4);
        for (uint32_t i = 0; i < mix.size(); i++) mix[i] = all[(997u * k + 13u * i) % all.size()];
        b.push_back({int(k), mix});
      }
      uint32_t rows = 0;
      for (auto& x : b) rows += uint32_t(x.second.size());
      uint32_t bad_mix = 0;
      if (rows <= ink2::MAXM) {
        M.step_lanes(b, nullptr, &lg);
        uint32_t r = 0;
        for (uint32_t k = 0; k < nl; k++) {
          r += uint32_t(b[k].second.size());
          if (memcmp(lg.data() + size_t(r - 1u) * V, ref_mix[k].data(), V * 4) != 0) bad_mix++;
        }
      }
      printf("ltest lanes=%u gen=%u: token mismatches=%u, logit mismatches=%u/%u steps x lanes, mixed-width (%u rows) mismatches=%u/%u -> %s\n",
             nl, n_gen, bad_tok + bad_end, bad_lg, nl * n_gen, rows, bad_mix, rows <= ink2::MAXM ? nl : 0u,
             (bad_tok + bad_end + bad_lg + bad_mix) == 0 ? "BIT-EXACT" : "DIFF");
      return (bad_tok + bad_end + bad_lg + bad_mix) == 0 ? 0 : 1;
    }

    if (cmd == "jspec") {
      // jspec <ids.bin> <lanes> <n_gen>: joint speculation. Each round every lane drafts (3-gram lookup, else
      // DSpark with p_max cut), one step_lanes() verifies all lanes' rows (<= MAXM, split evenly), then each lane
      // rolls back and commits on its own. Every lane's output is checked against its own greedy M=1 run.
      if (argc < 6) mtl::die("jspec <ids.bin> <lanes> <n_gen>");
      auto all = read_ids_bin(argv[3]);
      const uint32_t nl = uint32_t(atoi(argv[4]));
      const uint32_t n_gen = uint32_t(atoi(argv[5]));
      const uint32_t draft_k = getenv("INK_DRAFT_K") ? uint32_t(atoi(getenv("INK_DRAFT_K"))) : 7u;
      const float draft_cut = getenv("INK_DRAFT_CUT") ? float(atof(getenv("INK_DRAFT_CUT"))) : 0.50f;
      const bool use_lookup = getenv("INK_LOOKUP") ? (atoi(getenv("INK_LOOKUP")) != 0) : true;
      const bool check = !(getenv("INK_JCHECK") && atoi(getenv("INK_JCHECK")) == 0);
      auto prompt = [&](uint32_t k) {
        const uint32_t len = 97u + 151u * k;
        std::vector<uint32_t> p(len);
        for (uint32_t i = 0; i < len; i++) p[i] = all[(i + 211u * k) % all.size()];
        return p;
      };
      auto prefill = [&](const std::vector<uint32_t>& p, bool commit) {
        uint32_t t = 0;
        for (size_t i = 0; i < p.size(); i += ink2::MP) {
          std::vector<uint32_t> c(p.begin() + i, p.begin() + std::min(p.size(), i + ink2::MP));
          t = M.step(c).back();
          if (commit) M.mtp_commit(uint32_t(c.size()));
        }
        return t;
      };
      while (M.lanes_.size() < nl) M.add_lane();

      std::vector<std::vector<uint32_t>> ref(nl);
      if (check) {
        for (uint32_t k = 0; k < nl; k++) {
          M.use_lane(0);
          M.reset();
          M.samp = lane_samp(k);
          uint32_t t = prefill(prompt(k), false);
          for (uint32_t g = 0; g < n_gen; g++) { ref[k].push_back(t); t = M.step({t})[0]; }
        }
      }

      struct LS {
        std::vector<uint32_t> hist, out;
        uint32_t anchor = 0, pending = 0, pending_r0 = 0, rounds = 0, acc = 0;
        opus_lookup::Lookup lk;
      };
      std::vector<LS> ls(nl);
      for (uint32_t k = 0; k < nl; k++) {
        M.use_lane(int(k));
        M.reset();
        M.samp = lane_samp(k);
        ls[k].hist = prompt(k);
        ls[k].anchor = prefill(ls[k].hist, true);
        ls[k].out.push_back(ls[k].anchor);
      }
      double t_draft = 0, t_verify = 0;
      uint32_t rounds = 0, lk_hits = 0, rows_tot = 0;
      auto t0 = std::chrono::steady_clock::now();
      for (;;) {
        std::vector<uint32_t> act;
        for (uint32_t k = 0; k < nl; k++) if (ls[k].out.size() < n_gen) act.push_back(k);
        if (act.empty()) break;
        const uint32_t budget = std::max(1u, ink2::MAXM / uint32_t(act.size()));  // rows per lane this round
        std::vector<std::pair<int, std::vector<uint32_t>>> batches;
        auto td = std::chrono::steady_clock::now();
        for (uint32_t k : act) {
          auto& L = ls[k];
          M.use_lane(int(k));
          M.tap_row0 = L.pending_r0;
          std::vector<uint32_t> drafts;
          if (budget > 1u && use_lookup && L.hist.size() >= 4) {
            L.lk.sync(L.hist, false);
            std::vector<uint32_t> tail;
            const size_t tlen = std::min<size_t>(L.hist.size(), 63);
            tail.insert(tail.end(), L.hist.end() - long(tlen), L.hist.end());
            tail.push_back(L.anchor);
            uint32_t mlen = 0;
            uint32_t mpos = L.lk.find(L.hist, tail, &mlen);
            if (mpos != opus_lookup::Lookup::NONE && mlen >= 4) {
              const uint32_t cap_lk = std::min<uint32_t>(budget - 1u, uint32_t(L.hist.size() - mpos));
              for (uint32_t i = 0; i < cap_lk; i++) drafts.push_back(L.hist[mpos + i]);
              lk_hits++;
            }
          }
          if (drafts.empty() && budget > 1u) {
            std::vector<float> dprobs;
            drafts = M.draft(L.anchor, std::min(draft_k, budget - 1u), &dprobs, nullptr, L.pending);
            L.pending = 0;
            uint32_t keep = 0;
            while (keep < drafts.size() && (draft_cut <= 0.0f || dprobs[keep] >= draft_cut)) keep++;
            drafts.resize(std::min<uint32_t>(keep, budget - 1u));
          } else if (L.pending > 0) {
            M.mtp_commit(L.pending);
            L.pending = 0;
          }
          std::vector<uint32_t> v{L.anchor};
          v.insert(v.end(), drafts.begin(), drafts.end());
          batches.push_back({int(k), v});
        }
        t_draft += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td).count();
        double vms = 0;
        auto out = M.step_lanes(batches, &vms);
        t_verify += vms;
        uint32_t r0 = 0;
        for (auto& [k, v] : batches) {
          auto& L = ls[k];
          const uint32_t m = uint32_t(v.size());
          uint32_t acc = 0;
          while (acc + 1u < m && out[r0 + acc] == v[acc + 1u]) acc++;
          const uint32_t n_adv = acc + 1u;
          M.use_lane(k);
          M.rollback(n_adv, m);
          L.pending = n_adv;
          L.pending_r0 = r0;
          L.hist.push_back(L.anchor);
          for (uint32_t i = 0; i < acc; i++) {
            L.hist.push_back(v[i + 1u]);
            if (L.out.size() < n_gen) L.out.push_back(v[i + 1u]);
          }
          L.anchor = out[r0 + acc];
          if (L.out.size() < n_gen) L.out.push_back(L.anchor);
          L.acc += n_adv;
          L.rounds++;
          r0 += m;
        }
        rows_tot += r0;
        rounds++;
      }
      const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      uint32_t gen = 0, match = 0, acc = 0, lr = 0;
      for (uint32_t k = 0; k < nl; k++) {
        gen += uint32_t(ls[k].out.size()) - 1u;
        acc += ls[k].acc;
        lr += ls[k].rounds;
        if (check)
          for (uint32_t i = 0; i < n_gen; i++) match += (ls[k].out[i] == ref[k][i]);
      }
      printf("jspec lanes=%u K<=%u cut=%.2f lk=%d | %u tok, %u rounds, %.2f tok/lane-round, %.1f rows/step, lk_hits=%u: "
             "%.1f tok/s wall total (%.1f per lane) | draft %.2f ms/round wall, verify %.2f ms/round GPU\n",
             nl, draft_k, draft_cut, use_lookup ? 1 : 0, gen, rounds, double(acc) / lr, double(rows_tot) / rounds, lk_hits,
             1000.0 * gen / wall, 1000.0 * gen / wall / nl, t_draft / rounds, t_verify / rounds);
      if (check) printf("jspec exact check: %u/%u tokens match plain single-lane decode (%s)\n", match, nl * n_gen,
                        match == nl * n_gen ? "100% EXACT" : "MISMATCH");
      return (!check || match == nl * n_gen) ? 0 : 1;
    }

    if (cmd == "lgdump") {
      // lgdump <ids.bin> <n> <out.bin>: final-row logits after chunked prefill (MP) and after sequential M=1.
      if (argc < 6) mtl::die("lgdump <ids.bin> <n> <out.bin>");
      auto all = read_ids_bin(argv[3]);
      const uint32_t n = std::min<uint32_t>(uint32_t(atoi(argv[4])), uint32_t(all.size()));
      std::vector<float> l_pf, l_seq;
      std::vector<uint32_t> chunk;
      uint32_t pf_tok = 0;
      for (uint32_t i = 0; i < n; i += ink2::MP) {
        const uint32_t c = std::min(ink2::MP, n - i);
        chunk.assign(all.begin() + i, all.begin() + i + c);
        pf_tok = M.step(chunk, nullptr, (i + c == n) ? &l_pf : nullptr).back();
      }
      M.reset();
      uint32_t seq_tok = 0;
      for (uint32_t i = 0; i < n; i++) seq_tok = M.step({all[i]}, nullptr, (i + 1 == n) ? &l_seq : nullptr)[0];
      float md = 0.0f;
      for (size_t i = 0; i < l_pf.size(); i++) md = std::max(md, std::fabs(l_pf[i] - l_seq[i]));
      FILE* f = fopen(argv[5], "wb");
      fwrite(l_pf.data(), 4, l_pf.size(), f);
      fwrite(l_seq.data(), 4, l_seq.size(), f);
      fclose(f);
      printf("lgdump n=%u pf_tok=%u seq_tok=%u max_logit_diff=%.3e\n", n, pf_tok, seq_tok, md);
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
      uint32_t m_step = argc > 4 ? uint32_t(atoi(argv[4])) : 1u;
      std::vector<uint32_t> chunk;
      for (uint32_t i = 0; i < n_ctx; i += ink2::MP) {
        uint32_t n = std::min(ink2::MP, n_ctx - i);
        chunk.resize(n);
        for (uint32_t j = 0; j < n; j++) chunk[j] = paris_ids[(i + j) % paris_ids.size()];
        M.step(chunk);
      }
      // Warmup
      std::vector<uint32_t> probe(m_step);
      for (uint32_t i = 0; i < m_step; i++) probe[i] = paris_ids[(n_ctx + i) % paris_ids.size()];
      if (m_step == 1u) {
        uint32_t tok = 152717;
        for (int i = 0; i < 4; i++) tok = M.step({tok})[0];
        probe[0] = tok;
      } else {
        M.step(probe);
      }
      double unprof_ms = 0;
      M.step(probe, &unprof_ms);
      std::map<std::string, double> prof;
      M.step(probe, nullptr, nullptr, &prof);
      std::vector<std::pair<double, std::string>> rows;
      double tot = 0;
      for (auto& [k, v] : prof) { rows.push_back({v, k}); tot += v; }
      std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.first > b.first; });
      printf("dprof (ctx=%u, M=%u, unprofiled=%.2f ms = %.1f tok/s, profiled_sum=%.2f ms):\n",
             n_ctx, m_step, unprof_ms, 1000.0 * m_step / unprof_ms, tot);
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
      uint32_t pending_commit = 0;
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
          drafts = M.draft(anchor, draft_k, &dprobs, &dms, pending_commit);
          pending_commit = 0;
          t_draft += dms;
          if (draft_cut > 0.0f) {
            uint32_t keep = 0;
            while (keep < drafts.size() && dprobs[keep] >= draft_cut) keep++;
            drafts.resize(keep);
          }
        } else if (pending_commit > 0) {
          M.mtp_commit(pending_commit, &cms);
          pending_commit = 0;
          t_commit += cms;
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
        pending_commit = n_adv;

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
        printf("spec exact check: %u/%zu plain-decode tokens match (%s)\n",
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
