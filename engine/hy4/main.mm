// opus-hy4 CLI
//   opus-hy4 <pack> score <ids.bin> <logits.f16>   teacher-forced logits of every position (16-row steps)
//   opus-hy4 <pack> gen <ids.bin> <n>               prefill in 16-row steps, then greedy decode
//   opus-hy4 <pack> prof <ids.bin>                  per-op GPU ms of one decode step after the prompt
//   opus-hy4 <pack> vtest <ids.bin>                 rows of a 16-row step == the same tokens one at a time
#include "hy4.h"
#include <map>
#include "spec.inc"
#include "serve.inc"
#include <algorithm>
#include <chrono>
#include <cmath>

static std::vector<uint32_t> read_ids(const char* p) {
  std::string s = mtl::slurp(p);
  std::vector<uint32_t> v(s.size() / 4);
  memcpy(v.data(), s.data(), v.size() * 4);
  return v;
}

static void prefill(hy4::Model& m, const std::vector<uint32_t>& ids, double* ms, uint32_t chunk = 0) {
  double t = 0;
  if (!chunk) chunk = m.MR;
  for (size_t i = 0; i < ids.size(); i += chunk) {
    std::vector<uint32_t> c(ids.begin() + i, ids.begin() + std::min(ids.size(), i + chunk));
    if (i + chunk < ids.size()) m.next_hint.assign(ids.begin() + i + chunk, ids.begin() + std::min(ids.size(), i + 2 * chunk));
    double g;
    m.step(c, &g);
    t += g;
  }
  if (ms) *ms = t;
}

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 3) { fprintf(stderr, "usage: opus-hy4 <pack> serve | score|gen|spec|prof|pprof|vtest|ptest <ids.bin> ...\n"); return 2; }
    const std::string mode = argv[2];
    if (mode == "serve") {  // opus-hy4 <pack> serve  (HY4_CAP tokens)
      hy4::Model m(argv[1], getenv("HY4_CAP") ? atoi(getenv("HY4_CAP")) : 262144);
      return serve::main(m);
    }
    if (mode == "vittest") {  // opus-hy4 <pack> vittest <golden.bin>
      hy4::Model m(argv[1], getenv("HY4_CAP") ? atoi(getenv("HY4_CAP")) : 4096);
      if (!m.has_vision) { fprintf(stderr, "vittest: pack has no vision tower\n"); return 2; }
      std::string raw = mtl::slurp(argv[3]);
      if (raw.size() < 16) mtl::die("vittest: truncated golden file");
      uint32_t hdr[4];
      memcpy(hdr, raw.data(), 16);
      const uint32_t T = hdr[0], h = hdr[1], w = hdr[2], n_out = hdr[3];
      const size_t n_pix = size_t(T) * h * w * 3 * m.v_ps * m.v_ps;
      const size_t n_emb = size_t(n_out) * m.H;
      const uint16_t* pix = (const uint16_t*)(raw.data() + 16);
      const float* ref_bf16 = (const float*)(raw.data() + 16 + n_pix * 2);
      const float* ref_w8 = ref_bf16 + n_emb;
      double gms = 0;
      auto t0 = std::chrono::steady_clock::now();
      auto got = m.encode_vision(pix, T, h, w, &gms);
      const double wms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
      if (got.size() != n_emb) mtl::die("vittest: output size mismatch");
      auto cmp = [&](const char* tag, const float* ref) {
        double dot = 0, na = 0, nb = 0, err2 = 0, ma = 0;
        for (size_t i = 0; i < n_emb; i++) {
          double a = got[i], b = ref[i], d = std::abs(a - b);
          dot += a * b; na += a * a; nb += b * b; err2 += d * d;
          if (d > ma) ma = d;
        }
        fprintf(stderr, "vittest vs %s: cos=%.6f rel_L2=%.5f max_abs=%.5f\n",
                tag, dot / std::sqrt(na * nb), std::sqrt(err2 / nb), ma);
      };
      fprintf(stderr, "vittest: grid=(%u,%u,%u) -> %u tokens, GPU %.2f ms, wall %.2f ms\n", T, h, w, n_out, gms, wms);
      cmp("ref_bf16", ref_bf16);
      cmp("ref_w8  ", ref_w8);
      return 0;
    }
    auto ids = read_ids(argv[3]);
    const uint32_t cap = getenv("HY4_CAP") ? atoi(getenv("HY4_CAP")) : 32768;
    hy4::Model m(argv[1], cap);
    if (getenv("HY4_TEMP")) { m.temp = float(atof(getenv("HY4_TEMP"))); m.seed = getenv("HY4_SEED") ? strtoull(getenv("HY4_SEED"), nullptr, 10) : 1; }
    if (mode == "bench") {
      // opus-hy4 <pack> bench <ids.bin> [rows,...]: prefill the prompt in MR-row chunks (GPU tok/s), then the
      // mean GPU and wall ms of an R-row step after it (verify-sized steps, truncated back), 3 warm + 10 timed
      double pms;
      std::vector<uint32_t> pre(ids.begin(), ids.end() - 16);
      auto w0 = std::chrono::steady_clock::now();
      prefill(m, pre, &pms);
      const double pw = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() * 1e3;
      fprintf(stderr, "bench: prefill %zu tok: GPU %.0f ms (%.0f tok/s), wall %.0f ms (%.0f tok/s)\n", pre.size(), pms,
              pre.size() / (pms / 1e3), pw, pre.size() / (pw / 1e3));
      std::string rl = argc > 4 ? argv[4] : "1,4,6";
      const uint32_t p = m.pos;
      for (size_t s = 0; s < rl.size();) {
        size_t e = rl.find(',', s);
        if (e == std::string::npos) e = rl.size();
        const uint32_t R = atoi(rl.substr(s, e - s).c_str());
        s = e + 1;
        std::vector<uint32_t> rows(ids.end() - 16, ids.end() - 16 + R);
        for (int w = 0; w < 3; w++) { m.step(rows); m.truncate(p); }
        double g = 0, gt = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int w = 0; w < 10; w++) { m.step(rows, &g); gt += g; m.truncate(p); }
        const double wt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
        fprintf(stderr, "bench: %u rows at %u: GPU %.2f ms, wall %.2f ms\n", R, p, gt / 10, wt / 10);
      }
      return 0;
    }
    if (mode == "score") {
      FILE* f = fopen(argv[4], "wb");
      double gms = 0;
      unsigned long long xh = 1469598103934665603ull;
      // HY4_SCORE_PREFILL=N: positions < N go through the prefill path (MR-row chunks); their logits are
      // written as zeros, so the file keeps one row per position
      const size_t np = getenv("HY4_SCORE_PREFILL") ? std::min<size_t>(atoi(getenv("HY4_SCORE_PREFILL")), ids.size()) / 16 * 16 : 0;
      if (np) {
        prefill(m, std::vector<uint32_t>(ids.begin(), ids.begin() + np), nullptr);
        std::vector<__fp16> z(size_t(m.vocab), __fp16(0.0f));
        for (size_t i = 0; i < np; i++) fwrite(z.data(), 2, z.size(), f);
      }
      for (size_t i = np; i < ids.size(); i += hy4::MAXM) {
        std::vector<uint32_t> c(ids.begin() + i, ids.begin() + std::min(ids.size(), i + hy4::MAXM));
        std::vector<float> lg;
        double ms;
        m.step(c, &ms, &lg);
        gms += ms;
        for (size_t j = 0; j < lg.size() * 4; j++) { xh ^= ((const unsigned char*)lg.data())[j]; xh *= 1099511628211ull; }
        std::vector<__fp16> hh(lg.size());
        for (size_t j = 0; j < lg.size(); j++) hh[j] = __fp16(lg[j]);
        fwrite(hh.data(), 2, hh.size(), f);
      }
      fclose(f);
      fprintf(stderr, "scored %zu tokens, GPU %.1f ms, logits hash %016llx\n", ids.size(), gms, xh);
    } else if (mode == "gen") {
      const uint32_t n = argc > 4 ? atoi(argv[4]) : 64;
      double pms;
      std::vector<uint32_t> pre(ids.begin(), ids.end() - 1);
      prefill(m, pre, &pms);
      uint32_t t = ids.back();
      std::vector<uint32_t> out;
      double dms = 0;
      auto w0 = std::chrono::steady_clock::now();
      for (uint32_t i = 0; i < n; i++) {
        double g;
        t = m.step({t}, &g)[0];
        dms += g;
        out.push_back(t);
        if (t == m.eos) break;
      }
      const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() * 1e3;
      fprintf(stderr, "prefill %zu tok %.0f ms GPU (%.0f tok/s); decode %zu tok %.2f ms/tok GPU, %.2f ms/tok wall (%.1f tok/s wall)\n",
              pre.size(), pms, pre.size() / (pms / 1e3), out.size(), dms / out.size(), wall / out.size(), out.size() / (wall / 1e3));
      for (auto x : out) printf("%u ", x);
      printf("\n");
    } else if (mode == "prof") {
      prefill(m, ids, nullptr);
      std::map<std::string, double> pr;
      // HY4_PROF_ROWS rows per profiled step (a verify step), default 1
      const uint32_t R = getenv("HY4_PROF_ROWS") ? atoi(getenv("HY4_PROF_ROWS")) : 1;
      const std::vector<uint32_t> rows(ids.end() - R, ids.end());
      const uint32_t p = m.pos;
      for (int w = 0; w < 3; w++) { m.step(rows); m.truncate(p); }
      double g;
      m.step(rows, &g, nullptr, &pr);
      std::vector<std::pair<double, std::string>> v;
      double tot = 0;
      for (auto& [k, ms] : pr) { v.push_back({ms, k}); tot += ms; }
      std::sort(v.rbegin(), v.rend());
      for (auto& [ms, k] : v) fprintf(stderr, "%-16s %8.3f ms  %5.1f%%\n", k.c_str(), ms, 100 * ms / tot);
      fprintf(stderr, "sum of ops %.3f ms (profiled step)\n", tot);
    } else if (mode == "spec") {
      // DFly2 speculation vs plain greedy decoding from the same prompt: tokens must be identical
      const uint32_t n = argc > 4 ? atoi(argv[4]) : 128;
      // reference: the same prefill (the whole prompt in MR-row chunks), then plain decoding
      std::vector<uint32_t> ref;
      uint32_t t = 0;
      for (size_t i = 0; i < ids.size(); i += m.MR) {
        if (i + m.MR < ids.size()) m.next_hint.assign(ids.begin() + i + m.MR, ids.begin() + std::min(ids.size(), i + 2 * m.MR));
        t = m.step(std::vector<uint32_t>(ids.begin() + i, ids.begin() + std::min(ids.size(), i + m.MR))).back();
      }
      double plain = 0;
      ref.push_back(t);
      for (uint32_t i = 1; i < n; i++) { double g; t = m.step({t}, &g)[0]; plain += g; ref.push_back(t); }
      m.reset();
      // prefill the whole prompt; its last chunk leaves the source row for the first draft
      uint32_t b = 0, sx = 0, ss = 0;
      for (size_t i = 0; i < ids.size(); i += m.MR) {
        std::vector<uint32_t> c(ids.begin() + i, ids.begin() + std::min(ids.size(), i + m.MR));
        if (i + m.MR < ids.size()) m.next_hint.assign(ids.begin() + i + m.MR, ids.begin() + std::min(ids.size(), i + 2 * m.MR));
        auto o = m.step(c);
        b = o.back();
        sx = ss = c.size() > hy4::MAXM ? 0 : uint32_t(c.size()) - 1;
      }
      std::vector<uint32_t> out;
      hy4spec::Spec sp;
      sp.begin(m);
      const double ew0 = m.eng_wait_ms, er0 = m.eng_rest_ms;
      auto t0 = std::chrono::steady_clock::now();
      while (out.size() < n) sp.round(m, b, sx, ss, out, [&](uint32_t t) { return t == m.eos; });
      const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
      const auto& st = sp.st;
      const uint32_t rounds = st.rounds, acc = st.accepted;
      fprintf(stderr, "spec: engram wait %.2f ms / round (%.2f of it before the second Engram layer)\n", (m.eng_wait_ms - ew0) / rounds,
              (m.eng_rest_ms - er0) / rounds);
      fprintf(stderr, "spec: lookup %s: %.2f rows / round, %u drafter calls, %u lookup-only rounds, lookup proposed %u, tested %u, accepted %u\n",
              sp.on ? "on" : "off", double(st.rows) / rounds, st.drafts, st.lk_only, st.lk_proposed, st.lk_tested, st.lk_accepted);
      // both start with the prompt's next token
      size_t same = 0;
      while (same < out.size() && same < ref.size() && out[same] == ref[same]) same++;
      fprintf(stderr, "spec: %u rounds, %.2f drafts accepted / round (%.2f tokens / round); draft %.2f ms, verify %.2f ms per round; "
              "%.1f tok/s wall vs plain %.1f tok/s (GPU); %zu / %zu tokens match plain greedy\n",
              rounds, double(acc) / rounds, double(acc + rounds) / rounds, st.draft_ms / rounds, st.verify_ms / rounds,
              (acc + rounds) / (wall / 1e3), (n - 1) / (plain / 1e3), same, std::min<size_t>(ref.size(), out.size()));
    } else if (mode == "dtest") {
      // drafts after the prompt and a given anchor token (default: the model's own next token)
      uint32_t b = 0, sx = 0, ss = 0;
      for (size_t i = 0; i < ids.size(); i += m.MR) {
        std::vector<uint32_t> c(ids.begin() + i, ids.begin() + std::min(ids.size(), i + m.MR));
        b = m.step(c).back();
        sx = ss = c.size() > hy4::MAXM ? 0 : uint32_t(c.size()) - 1;
      }
      const uint32_t anchor = argc > 4 ? atoi(argv[4]) : b;
      auto d = m.draft(anchor, sx, ss);
      fprintf(stderr, "dtest: model's next %u, anchor %u, drafts", b, anchor);
      for (auto t : d) fprintf(stderr, " %u", t);
      fprintf(stderr, "\n");
    } else if (mode == "dprof") {
      // per-op GPU ms of one draft call after the prompt
      uint32_t b = 0, sx = 0, ss = 0;
      for (size_t i = 0; i < ids.size(); i += m.MR) {
        std::vector<uint32_t> c(ids.begin() + i, ids.begin() + std::min(ids.size(), i + m.MR));
        b = m.step(c).back();
        sx = ss = c.size() > hy4::MAXM ? 0 : uint32_t(c.size()) - 1;
      }
      for (int w = 0; w < 3; w++) m.draft(b, sx, ss);
      double g = 0;
      std::map<std::string, double> pr;
      m.draft(b, sx, ss, &g, &pr);
      double g2 = 0;
      for (int w = 0; w < 10; w++) { double x; m.draft(b, sx, ss, &x); g2 += x; }
      std::vector<std::pair<double, std::string>> v;
      double tot = 0;
      for (auto& [k, ms] : pr) { v.push_back({ms, k}); tot += ms; }
      std::sort(v.rbegin(), v.rend());
      for (auto& [ms, k] : v) fprintf(stderr, "%-16s %8.3f ms  %5.1f%%\n", k.c_str(), ms, 100 * ms / tot);
      fprintf(stderr, "draft: sum of ops %.3f ms (profiled), unprofiled GPU %.3f ms\n", tot, g2 / 10);
    } else if (mode == "pprof") {
      // per-op GPU ms of the last MR-row prefill chunk of the prompt
      const size_t n = ids.size() / m.MR * m.MR;
      std::vector<uint32_t> pre(ids.begin(), ids.begin() + n - m.MR), last(ids.begin() + n - m.MR, ids.begin() + n);
      prefill(m, pre, nullptr);
      std::map<std::string, double> pr;
      double g;
      m.step(last, &g, nullptr, &pr);
      std::vector<std::pair<double, std::string>> v;
      double tot = 0;
      for (auto& [k, ms] : pr) { v.push_back({ms, k}); tot += ms; }
      std::sort(v.rbegin(), v.rend());
      for (auto& [ms, k] : v) fprintf(stderr, "%-16s %8.3f ms  %5.1f%%\n", k.c_str(), ms, 100 * ms / tot);
      fprintf(stderr, "chunk of %u rows at %zu: sum of ops %.2f ms\n", m.MR, pre.size(), tot);
    } else if (mode == "ptest") {
      // prefill in MR-row chunks (mmp / mmpg, one attention split) vs 16-row steps: next-step logits
      std::vector<uint32_t> pre(ids.begin(), ids.end() - 1);
      std::vector<float> a, b;
      double ta, tb;
      prefill(m, pre, &ta);
      m.step({ids.back()}, nullptr, &a);
      m.reset();
      prefill(m, pre, &tb, hy4::MAXM);
      m.step({ids.back()}, nullptr, &b);
      double mx = 0, ref = 0;
      size_t ia = 0, ib = 0;
      for (size_t i = 0; i < a.size(); i++) {
        mx = std::max(mx, (double)std::fabs(a[i] - b[i])); ref = std::max(ref, (double)std::fabs(b[i]));
        if (a[i] > a[ia]) ia = i;
        if (b[i] > b[ib]) ib = i;
      }
      fprintf(stderr, "ptest: %zu tokens, chunked %.0f ms vs 16-row %.0f ms; next-token logits max |diff| %.3g (max |logit| %.3g), argmax %zu / %zu\n",
              pre.size(), ta, tb, mx, ref, ia, ib);
    } else if (mode == "ctest") {
      // prefix-cache file round trip: logits of 16-row steps after a restore (at the saved length, and after a
      // truncation 300 tokens below it) == the same steps on the live state; bit for bit
      const size_t n = ids.size() - 16;
      std::vector<uint32_t> pre(ids.begin(), ids.begin() + n), tail(ids.end() - 16, ids.end());
      prefill(m, pre, nullptr);
      const char* path = "/tmp/hy4-ctest.hy4p";
      double sms = 0, lms = 0;
      if (!m.save_prefix(path, &sms)) { fprintf(stderr, "ctest: save failed\n"); return 1; }
      const uint32_t p = m.pos, back = std::min<uint32_t>(p, 300);
      std::vector<float> a, b, a2, b2;
      m.step(tail, nullptr, &a);
      m.truncate(p - back);
      m.step(tail, nullptr, &a2);
      m.reset();
      for (auto& l : m.L)  // no stale rows can help
        for (id<MTLBuffer> b : {l.lat, l.tokring, l.bk}) if (b) memset(b.contents, 0x5A, b.length);
      if (m.load_prefix(path, &lms) != p) { fprintf(stderr, "ctest: load failed\n"); return 1; }
      m.step(tail, nullptr, &b);
      m.truncate(p - back);
      m.step(tail, nullptr, &b2);
      size_t d1 = 0, d2 = 0;
      for (size_t i = 0; i < a.size(); i++) d1 += a[i] != b[i];
      for (size_t i = 0; i < a2.size(); i++) d2 += a2[i] != b2[i];
      fprintf(stderr, "ctest: %u tokens, save %.0f ms, load %.0f ms (floor %u); differing logits at the length %zu, %u below it %zu (%s)\n", p,
              sms, lms, m.ring_floor, d1, back, d2, d1 + d2 ? "FAIL" : "bit-identical");
      unlink(path);
      return d1 + d2 ? 1 : 0;
    } else if (mode == "ntest") {
      // multi-lane bit-identity test: K lanes with distinct prompts & lengths (spanning < dense_max and > dense_max),
      // running both 1-row/lane decode and multi-row joint-verify steps vs single-lane m.step()
      const uint32_t K = argc > 4 ? std::min<uint32_t>(hy4::MAXM, atoi(argv[4])) : 4;
      for (uint32_t k = 1; k < K; k++) m.add_slot(cap);
      std::vector<uint32_t> p0(K);
      std::vector<std::vector<uint32_t>> tails(K);
      const size_t ntail = 4;
      for (uint32_t k = 0; k < K; k++) {
        m.use_slot(int(k));
        // vary length across lanes (some > 2048 sparse, some < 2048 dense) and rotate tokens
        const size_t base = ids.size() > ntail + 64 ? ids.size() - ntail : ids.size() / 2;
        const size_t len = (k % 2 == 0) ? (base - (k * 37) % std::min<size_t>(base / 4, 200))
                                        : std::min<size_t>(base, 500 + k * 211);
        std::vector<uint32_t> pre(ids.begin(), ids.begin() + len);
        if (k > 0 && pre.size() > 8) pre[0] = ids[(k * 13) % ids.size()];
        prefill(m, pre, nullptr);
        p0[k] = m.pos;
        tails[k].assign(ids.end() - ntail, ids.end());
        if (k > 0) std::rotate(tails[k].begin(), tails[k].begin() + (k % ntail), tails[k].end());
      }
      // 1) 1 token per lane (4 steps, mixed greedy & sampled)
      std::vector<std::vector<float>> lg_lane(K);
      std::vector<std::vector<uint32_t>> tok_lane(K);
      for (size_t s = 0; s < ntail; s++) {
        std::vector<hy4::Model::LaneBatch> bs;
        for (uint32_t k = 0; k < K; k++) {
          hy4::Model::LaneBatch b;
          b.slot = int(k);
          b.toks = {tails[k][s]};
          b.temp = (k % 2 == 1) ? 0.9f : 0.0f;
          b.min_p = 0.02f;
          b.top_p = (k % 3 == 1) ? 0.85f : 1.0f;
          b.top_k = (k % 3 == 2) ? 40u : 0u;
          b.seed = 100 + k;
          bs.push_back(std::move(b));
        }
        std::vector<float> lg;
        auto out = m.step_lanes(bs, nullptr, &lg);
        for (uint32_t k = 0; k < K; k++) {
          tok_lane[k].push_back(out[k]);
          lg_lane[k].insert(lg_lane[k].end(), lg.begin() + size_t(k) * m.vocab, lg.begin() + size_t(k + 1) * m.vocab);
        }
      }
      size_t diff_lg = 0, diff_tok = 0;
      for (uint32_t k = 0; k < K; k++) {
        m.use_slot(int(k));
        m.truncate(p0[k]);
        m.temp = (k % 2 == 1) ? 0.9f : 0.0f;
        m.min_p = 0.02f;
        m.top_p = (k % 3 == 1) ? 0.85f : 1.0f;
        m.top_k = (k % 3 == 2) ? 40u : 0u;
        m.seed = 100 + k;
        std::vector<float> ref_lg, one;
        for (size_t s = 0; s < ntail; s++) {
          auto o = m.step({tails[k][s]}, nullptr, &one);
          diff_tok += (o[0] != tok_lane[k][s]);
          ref_lg.insert(ref_lg.end(), one.begin(), one.end());
        }
        for (size_t i = 0; i < ref_lg.size(); i++) diff_lg += (ref_lg[i] != lg_lane[k][i]);
        m.truncate(p0[k]);
      }
      // 2) multi-row joint verify (up to 16 total rows across K lanes)
      size_t diff_jlg = 0;
      {
        const uint32_t per = std::max<uint32_t>(1, std::min<uint32_t>(ntail, hy4::MAXM / K));
        std::vector<hy4::Model::LaneBatch> bs;
        for (uint32_t k = 0; k < K; k++) {
          hy4::Model::LaneBatch b;
          b.slot = int(k);
          b.toks.assign(tails[k].begin(), tails[k].begin() + per);
          bs.push_back(std::move(b));
        }
        std::vector<float> jlg;
        m.step_lanes(bs, nullptr, &jlg);
        for (uint32_t k = 0; k < K; k++) {
          m.use_slot(int(k));
          m.truncate(p0[k]);
          m.temp = 0.0f;
          std::vector<float> ref;
          m.step(std::vector<uint32_t>(tails[k].begin(), tails[k].begin() + per), nullptr, &ref);
          const float* got = jlg.data() + size_t(k) * per * m.vocab;
          for (size_t i = 0; i < ref.size(); i++) diff_jlg += (ref[i] != got[i]);
        }
      }
      fprintf(stderr, "ntest: %u lanes (1-row x %zu + joint %u-row/lane): %zu diff logits, %zu diff tokens, %zu diff joint logits (%s)\n",
              K, ntail, std::max<uint32_t>(1, std::min<uint32_t>(ntail, hy4::MAXM / K)), diff_lg, diff_tok, diff_jlg,
              (diff_lg + diff_tok + diff_jlg) ? "FAIL" : "bit-identical");
      return (diff_lg + diff_tok + diff_jlg) ? 1 : 0;
    } else if (mode == "jtest") {
      // joint speculation: jtest <ids.bin> <n> [more ids.bin ...], one lane per prompt (HY4_TEMP: lane k samples
      // with seed + k). Each lane's n tokens from joint rounds must equal the lane decoded alone (plain one-row
      // steps); wall tok/s against plain one-row lane steps over the same lanes. HY4_JS_ROWS: the row budget.
      const uint32_t n = argc > 4 ? atoi(argv[4]) : 200;
      std::vector<std::vector<uint32_t>> P{ids};
      for (int i = 5; i < argc; i++) P.push_back(read_ids(argv[i]));
      const uint32_t K = std::min<uint32_t>(hy4::MAXM, uint32_t(P.size()));
      const uint32_t rows = hy4::cfg().js_rows;
      for (uint32_t k = 1; k < K; k++) m.add_slot(cap);
      const float temp = m.temp;
      const uint64_t seed = m.seed;
      std::vector<hy4::Model::LaneBatch> samp(K);
      for (uint32_t k = 0; k < K; k++) { samp[k].temp = temp; samp[k].min_p = m.default_min_p; samp[k].seed = seed + k; }
      auto use = [&](uint32_t k) {
        m.use_slot(int(k));
        m.temp = samp[k].temp; m.min_p = samp[k].min_p; m.top_p = 1.0f; m.top_k = 0; m.seed = samp[k].seed;
      };
      std::vector<uint32_t> p0(K), first(K);
      std::vector<std::vector<uint32_t>> ref(K);
      for (uint32_t k = 0; k < K; k++) {
        use(k);
        uint32_t t = 0;
        for (size_t i = 0; i < P[k].size(); i += m.MR) {
          if (i + m.MR < P[k].size()) m.next_hint.assign(P[k].begin() + i + m.MR, P[k].begin() + std::min(P[k].size(), i + 2 * m.MR));
          t = m.step(std::vector<uint32_t>(P[k].begin() + i, P[k].begin() + std::min(P[k].size(), i + m.MR))).back();
        }
        p0[k] = m.pos;
        first[k] = t;
        ref[k].push_back(t);
        for (uint32_t i = 1; i < n; i++) { t = m.step({t})[0]; ref[k].push_back(t); }
      }
      // joint rounds (the lanes start without drafter sources: the prefills ran one after another)
      hy4spec::Spec sp;
      std::vector<hy4spec::Lane> ln(K);
      std::vector<std::vector<uint32_t>> got(K);
      for (uint32_t k = 0; k < K; k++) {
        use(k);
        m.truncate(p0[k]);
        sp.begin(m);
        ln[k].slot = int(k);
        ln[k].b = first[k];
        ln[k].samp = samp[k];
        ln[k].is_stop = [&m](uint32_t t) { return t == m.eos; };
      }
      uint32_t jrounds = 0;
      double jgpu = 0;
      auto t0 = std::chrono::steady_clock::now();
      for (;;) {
        std::vector<hy4spec::Lane*> lp;
        for (uint32_t k = 0; k < K; k++) if (got[k].size() < n) lp.push_back(&ln[k]);
        if (lp.empty()) break;
        double g = 0;
        sp.joint(m, lp, rows, &g);
        jgpu += g;
        jrounds++;
        for (uint32_t k = 0; k < K; k++)
          if (got[k].size() < n) got[k].insert(got[k].end(), ln[k].out.begin(), ln[k].out.end());
      }
      const double jwall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
      size_t jdiff = 0, jtok = 0;
      for (uint32_t k = 0; k < K; k++) {
        got[k].resize(std::min<size_t>(got[k].size(), n));
        for (uint32_t i = 0; i < n; i++) jdiff += i >= got[k].size() || got[k][i] != ref[k][i];
        jtok += got[k].size();
      }
      // plain one-row lane steps from the same states
      for (uint32_t k = 0; k < K; k++) { use(k); m.truncate(p0[k]); }
      std::vector<std::vector<uint32_t>> pl(K);
      std::vector<uint32_t> cur(first);
      t0 = std::chrono::steady_clock::now();
      for (uint32_t i = 1; i < n; i++) {
        std::vector<hy4::Model::LaneBatch> bs(samp);
        for (uint32_t k = 0; k < K; k++) { bs[k].slot = int(k); bs[k].toks = {cur[k]}; pl[k].push_back(cur[k]); }
        auto o = m.step_lanes(bs);
        for (uint32_t k = 0; k < K; k++) cur[k] = o[k];
      }
      const double pwall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
      size_t pdiff = 0;
      for (uint32_t k = 0; k < K; k++) {
        pl[k].push_back(cur[k]);
        for (uint32_t i = 0; i < n; i++) pdiff += pl[k][i] != ref[k][i];
        const auto& st = ln[k].st;
        fprintf(stderr, "jtest: lane %u (prompt %zu): %u rounds, %.2f accepted / round, %.2f rows / round, %u drafter calls, %u lookup-only, lookup %u/%u\n",
                k, P[k].size(), st.rounds, double(st.accepted) / std::max(1u, st.rounds), double(st.rows) / std::max(1u, st.rounds), st.drafts,
                st.lk_only, st.lk_accepted, st.lk_tested);
      }
      fprintf(stderr, "jtest: %u lanes, %u tokens each, rows %u, temp %.2f: joint %u rounds, %.1f ms GPU + %.1f ms other / round, %.1f tok/s wall; "
              "plain lanes %.1f tok/s wall (%.2fx); %zu joint / %zu plain tokens differ from single-lane decoding (%s)\n",
              K, n, rows, temp, jrounds, jgpu / jrounds, (jwall - jgpu) / jrounds, jtok / (jwall / 1e3), double(K) * (n - 1) / (pwall / 1e3),
              (jtok / (jwall / 1e3)) / (double(K) * (n - 1) / (pwall / 1e3)), jdiff, pdiff, jdiff + pdiff ? "FAIL" : "exact");
      return jdiff + pdiff ? 1 : 0;
    } else if (mode == "btest") {
      // multi-lane decode throughput benchmark: btest <ids.bin> [lanes,...]
      std::string ll = argc > 4 ? argv[4] : "1,2,4,8,16";
      std::vector<uint32_t> ks;
      uint32_t kmax = 1;
      for (size_t s = 0; s < ll.size();) {
        size_t e = ll.find(',', s);
        if (e == std::string::npos) e = ll.size();
        uint32_t k = std::min<uint32_t>(hy4::MAXM, atoi(ll.substr(s, e - s).c_str()));
        ks.push_back(k); kmax = std::max(kmax, k);
        s = e + 1;
      }
      for (uint32_t k = 1; k < kmax; k++) m.add_slot(cap);
      std::vector<uint32_t> pre(ids.begin(), ids.end() - 16);
      // prefill slot 0 once, then copy its KV/indexer state to slots 1..kmax-1 so setup is instant
      prefill(m, pre, nullptr);
      const uint32_t p = m.pos;
      for (uint32_t k = 1; k < kmax; k++) {
        m.slots_[k].pos = p;
        m.slots_[k].history = m.history;
        for (size_t li = 0; li < m.L.size(); li++) {
          if (m.slots_[k].lat[li]) memcpy(m.slots_[k].lat[li].contents, m.slots_[0].lat[li].contents, m.slots_[0].lat[li].length);
          if (m.slots_[k].tokring[li]) memcpy(m.slots_[k].tokring[li].contents, m.slots_[0].tokring[li].contents, m.slots_[0].tokring[li].length);
          if (m.slots_[k].bk[li]) memcpy(m.slots_[k].bk[li].contents, m.slots_[0].bk[li].contents, m.slots_[0].bk[li].length);
        }
      }
      for (uint32_t K : ks) {
        std::vector<hy4::Model::LaneBatch> bs;
        for (uint32_t k = 0; k < K; k++) bs.push_back({int(k), {ids[ids.size() - 16 + k]}});
        auto reset_lanes = [&] {
          for (uint32_t k = 0; k < K; k++) { m.use_slot(int(k)); m.truncate(p); }
        };
        for (int w = 0; w < 3; w++) { m.step_lanes(bs); reset_lanes(); }
        double gt = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int w = 0; w < 10; w++) { double g = 0; m.step_lanes(bs, &g); gt += g; reset_lanes(); }
        const double wt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
        fprintf(stderr, "btest: %2u lanes at %u: GPU %.2f ms/step (%.1f tok/s), wall %.2f ms/step (%.1f tok/s)\n",
                K, p, gt / 10, K / (gt / 10 / 1e3), wt / 10, K / (wt / 10 / 1e3));
      }
      return 0;
    } else if (mode == "vtest") {
      const size_t n = std::min<size_t>(ids.size(), 16);
      std::vector<uint32_t> pre(ids.begin(), ids.end() - n), tail(ids.end() - n, ids.end());
      prefill(m, pre, nullptr);
      const uint32_t p = m.pos;
      std::vector<float> a, b, one;
      m.step(tail, nullptr, &a);
      m.truncate(p);
      for (auto t : tail) { m.step({t}, nullptr, &one); b.insert(b.end(), one.begin(), one.end()); }
      size_t diff = 0;
      for (size_t i = 0; i < a.size(); i++) diff += a[i] != b[i];
      fprintf(stderr, "vtest: %zu rows, %zu differing logits (%s)\n", n, diff, diff ? "FAIL" : "bit-identical");
      return diff ? 1 : 0;
    }
  }
  return 0;
}
