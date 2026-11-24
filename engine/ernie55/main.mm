// DankAttn ERNIE-5.5 CLI:
//   dankattn-ernie55 <pack_or_http_url> score <ids.bin> <logits.f16>
//   dankattn-ernie55 <pack_or_http_url> gen <ids.bin> <n_tokens>
//   dankattn-ernie55 <pack_or_http_url> bench <ids.bin>
#include "ernie55.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

static std::vector<uint32_t> read_ids(const char* p) {
  std::string s = mtl::slurp(p);
  std::vector<uint32_t> v(s.size() / 4);
  memcpy(v.data(), s.data(), v.size() * 4);
  return v;
}

static uint32_t argmax(const float* p, uint32_t n) {
  uint32_t best = 0;
  float bv = p[0];
  for (uint32_t i = 1; i < n; i++) {
    if (p[i] > bv) { bv = p[i]; best = i; }
  }
  return best;
}

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 4) {
      fprintf(stderr, "usage: dankattn-ernie55 <pack|url> score|gen|bench <ids.bin> [arg]\n");
      return 2;
    }
    const std::string pack = argv[1];
    const std::string mode = argv[2];
    const uint32_t cap = getenv("E5_CAP") ? uint32_t(atoi(getenv("E5_CAP"))) : 262144u;
    ernie55::Model m(pack, cap);
    std::vector<uint32_t> ids;
    if (mode != "suite") ids = read_ids(argv[3]);

    if (mode == "score") {
      if (argc < 5) mtl::die("score needs <out_logits.f16>");
      FILE* out = fopen(argv[4], "wb");
      if (!out) mtl::die("fopen out");
      double tot_ms = 0;
      std::vector<__fp16> row16(m.vocab);
      for (size_t i = 0; i < ids.size(); i += ernie55::MAXM) {
        std::vector<uint32_t> chunk(ids.begin() + i, ids.begin() + std::min(ids.size(), i + size_t(ernie55::MAXM)));
        double gms = 0;
        const float* lg = m.step(chunk, &gms);
        tot_ms += gms;
        for (size_t r = 0; r < chunk.size(); r++) {
          const float* src = lg + r * m.vocab;
          for (uint32_t v = 0; v < m.vocab; v++) row16[v] = __fp16(src[v]);
          fwrite(row16.data(), 2, m.vocab, out);
        }
      }
      fclose(out);
      fprintf(stderr, "score: %zu tokens in %.1f ms GPU (%.1f tok/s)\n",
              ids.size(), tot_ms, ids.size() / (tot_ms / 1e3));
      return 0;
    }

    if (mode == "gen") {
      const uint32_t ngen = argc > 4 ? uint32_t(atoi(argv[4])) : 64u;
      double pms = 0;
      const float* lg = nullptr;
      for (size_t i = 0; i < ids.size(); i += ernie55::MAXM) {
        std::vector<uint32_t> chunk(ids.begin() + i, ids.begin() + std::min(ids.size(), i + size_t(ernie55::MAXM)));
        double gms = 0;
        lg = m.step(chunk, &gms);
        pms += gms;
      }
      uint32_t last_r = uint32_t((ids.size() - 1) % ernie55::MAXM);
      uint32_t next_tok = argmax(lg + size_t(last_r) * m.vocab, m.vocab);
      fprintf(stderr, "prefill %zu tok: %.1f ms GPU (%.1f tok/s)\n", ids.size(), pms, ids.size() / (pms / 1e3));

      std::vector<uint32_t> gen_ids;
      gen_ids.push_back(next_tok);
      double dms = 0;
      auto w0 = std::chrono::steady_clock::now();
      for (uint32_t g = 1; g < ngen; g++) {
        if (next_tok == m.eos) break;
        double gms = 0;
        lg = m.step({next_tok}, &gms);
        dms += gms;
        next_tok = argmax(lg, m.vocab);
        gen_ids.push_back(next_tok);
      }
      double wms = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() * 1e3;
      fprintf(stderr, "decode %zu tok: %.1f ms GPU (%.1f tok/s), wall %.1f ms (%.1f tok/s)\n",
              gen_ids.size(), dms, (gen_ids.size() > 1 ? (gen_ids.size() - 1) / (dms / 1e3) : 0.0),
              wms, (gen_ids.size() > 1 ? (gen_ids.size() - 1) / (wms / 1e3) : 0.0));
      for (size_t i = 0; i < gen_ids.size(); i++)
        printf("%u%s", gen_ids[i], (i + 1 < gen_ids.size()) ? " " : "\n");
      return 0;
    }

    if (mode == "bench") {
      double pms = 0;
      auto pw0 = std::chrono::steady_clock::now();
      for (size_t i = 0; i < ids.size(); i += ernie55::MAXM) {
        std::vector<uint32_t> chunk(ids.begin() + i, ids.begin() + std::min(ids.size(), i + size_t(ernie55::MAXM)));
        double gms = 0;
        m.step(chunk, &gms);
        pms += gms;
      }
      double pw = std::chrono::duration<double>(std::chrono::steady_clock::now() - pw0).count() * 1e3;
      fprintf(stderr, "bench prefill %zu tok: GPU %.0f ms (%.1f tok/s), wall %.0f ms (%.1f tok/s)\n",
              ids.size(), pms, ids.size() / (pms / 1e3), pw, ids.size() / (pw / 1e3));
      uint32_t p = m.pos;
      std::vector<uint32_t> one = {ids.back()};
      for (int w = 0; w < 3; w++) { m.step(one); m.truncate(p); }
      double gt = 0;
      auto dw0 = std::chrono::steady_clock::now();
      for (int w = 0; w < 10; w++) { double g = 0; m.step(one, &g); gt += g; m.truncate(p); }
      double dw = std::chrono::duration<double>(std::chrono::steady_clock::now() - dw0).count() * 1e3;
      fprintf(stderr, "bench decode (pos=%u): GPU %.2f ms/tok (%.1f tok/s), wall %.2f ms/tok (%.1f tok/s)\n",
              p, gt / 10.0, 10000.0 / gt, dw / 10.0, 10000.0 / dw);
      return 0;
    }

    if (mode == "suite") {
      // argv[3] = suite dir containing eval_144.bin + p*.bin; writes results to stdout + full_logits.f16
      const std::string sdir = argv[3];
      // 1. Score eval_144.bin
      {
        auto eids = read_ids((sdir + "/eval_144.bin").c_str());
        m.reset();
        FILE* out = fopen((sdir + "/full_logits.f16").c_str(), "wb");
        double tot_ms = 0;
        std::vector<__fp16> row16(m.vocab);
        for (size_t i = 0; i < eids.size(); i += ernie55::MAXM) {
          std::vector<uint32_t> chunk(eids.begin() + i, eids.begin() + std::min(eids.size(), i + size_t(ernie55::MAXM)));
          double gms = 0;
          const float* lg = m.step(chunk, &gms);
          tot_ms += gms;
          for (size_t r = 0; r < chunk.size(); r++) {
            const float* src = lg + r * m.vocab;
            for (uint32_t v = 0; v < m.vocab; v++) row16[v] = __fp16(src[v]);
            fwrite(row16.data(), 2, m.vocab, out);
          }
        }
        fclose(out);
        fprintf(stderr, "[suite] score eval_144: %.1f ms GPU (%.1f tok/s)\n", tot_ms, eids.size() / (tot_ms / 1e3));
      }

      // 2. Generation battery
      const std::vector<std::pair<std::string, uint32_t>> prompts = {
        {"p1_paris", 16},
        {"p2_math", 128},
        {"p3_code", 160},
        {"p4_moe", 128},
        {"p5_needle4k", 48},
        {"p6_needle16k", 48},
      };
      for (auto& [name, ngen] : prompts) {
        std::string ppath = sdir + "/" + name + ".bin";
        FILE* chk = fopen(ppath.c_str(), "rb");
        if (!chk) continue;
        fclose(chk);
        auto pids = read_ids(ppath.c_str());
        m.reset();
        double pms = 0;
        const float* lg = nullptr;
        auto pw0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < pids.size(); i += ernie55::MAXM) {
          std::vector<uint32_t> chunk(pids.begin() + i, pids.begin() + std::min(pids.size(), i + size_t(ernie55::MAXM)));
          double gms = 0;
          lg = m.step(chunk, &gms);
          pms += gms;
        }
        double pw = std::chrono::duration<double>(std::chrono::steady_clock::now() - pw0).count() * 1e3;
        uint32_t last_r = uint32_t((pids.size() - 1) % ernie55::MAXM);
        uint32_t next_tok = argmax(lg + size_t(last_r) * m.vocab, m.vocab);
        std::vector<uint32_t> gids = {next_tok};
        double dms = 0;
        auto dw0 = std::chrono::steady_clock::now();
        for (uint32_t g = 1; g < ngen; g++) {
          if (next_tok == m.eos) break;
          double gms = 0;
          lg = m.step({next_tok}, &gms);
          dms += gms;
          next_tok = argmax(lg, m.vocab);
          gids.push_back(next_tok);
        }
        double dw = std::chrono::duration<double>(std::chrono::steady_clock::now() - dw0).count() * 1e3;
        double dtoks = gids.size() > 1 ? double(gids.size() - 1) : 1.0;
        fprintf(stderr, "[suite] %-12s prompt=%zu tok prefill=%.1f tok/s (GPU %.0fms) | gen=%zu tok decode=%.1f tok/s GPU (%.1f wall)\n",
                name.c_str(), pids.size(), pids.size() / (pms / 1e3), pms,
                gids.size(), dtoks / (dms / 1e3), dtoks / (dw / 1e3));
        printf("PROMPT %s:", name.c_str());
        for (uint32_t t : gids) printf(" %u", t);
        printf("\n");
        fflush(stdout);
      }
      return 0;
    }

    fprintf(stderr, "unknown mode: %s\n", mode.c_str());
    return 2;
  }
}
