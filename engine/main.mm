#include <unistd.h>
// opus CLI: teacher-forced scoring and greedy generation on an opus pack.
//   opus <pack> score <ids.bin> <logits_out.f16>   all-position logits (f16) for fidelity checks
//   opus <pack> gen <ids.bin> <n>                    prefill in 16-row steps, then greedy decode
#include "model.h"
#include <chrono>
#include <memory>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <thread>

static std::vector<uint32_t> read_ids(const char* p) {
  std::string s = mtl::slurp(p);
  std::vector<uint32_t> v(s.size() / 4);
  memcpy(v.data(), s.data(), v.size() * 4);
  return v;
}

#include "serve.inc"

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 3) { fprintf(stderr, "usage: opus <pack> score|gen|spec|vtest|prof|serve ...\n"); return 2; }
    std::string mode = argv[2];
    if (mode == "serve") return serve::main(argc, argv);
    auto ids = read_ids(argv[3]);
    opus::Model m(argv[1], getenv("OPUS_CLI_CAP") ? atoi(getenv("OPUS_CLI_CAP")) : 32768);
    if (mode == "score") {
      FILE* f = fopen(argv[4], "wb");
      double gms = 0;
      unsigned long long xh = 1469598103934665603ull;  // FNV-1a over the raw f32 logits
      for (size_t i = 0; i < ids.size(); i += opus::MAXM) {
        std::vector<uint32_t> chunk(ids.begin() + i, ids.begin() + std::min(ids.size(), i + opus::MAXM));
        std::vector<float> lg;
        double ms;
        m.step(chunk, &ms, &lg);
        gms += ms;
        for (size_t j = 0; j < lg.size() * 4; j++) { xh ^= ((const unsigned char*)lg.data())[j]; xh *= 1099511628211ull; }
        std::vector<__fp16> h(lg.size());
        for (size_t j = 0; j < lg.size(); j++) h[j] = __fp16(lg[j]);
        fwrite(h.data(), 2, h.size(), f);
      }
      fclose(f);
      fprintf(stderr, "scored %zu tokens, GPU %.1f ms, logits hash %016llx\n", ids.size(), gms, xh);
    } else if (mode == "nll") {
      // teacher-forced mean NLL of a token sequence: opus <pack> nll <ids.bin>
      // The quality number for any lossy path (tolerant accept, kv8, gdn8, low-bit weights):
      // same units as the golden NLL, so a change can be compared against a reorder.
      const size_t NV = m.lm_head.N;
      double s = 0;
      size_t cnt = 0;
      for (size_t i = 0; i + 1 < ids.size(); i += opus::MAXM) {
        const size_t e = std::min(ids.size() - 1, i + opus::MAXM);
        std::vector<uint32_t> chunk(ids.begin() + i, ids.begin() + e);
        std::vector<float> lg;
        double ms;
        m.step(chunk, &ms, &lg);
        for (size_t r = 0; r < chunk.size(); r++) {
          const float* row = lg.data() + r * NV;
          double mx = -1e30;
          for (size_t v = 0; v < NV; v++) mx = std::max(mx, double(row[v]));
          double z = 0;
          for (size_t v = 0; v < NV; v++) z += std::exp(double(row[v]) - mx);
          s -= double(row[ids[i + r + 1]]) - mx - std::log(z);
          cnt++;
        }
      }
      printf("nll %.4f over %zu tokens\n", cnt ? s / double(cnt) : 0.0, cnt);
    } else if (mode == "spec") {
      // speculative (MTP) vs plain greedy: opus <pack> spec <ids.bin> <n>
      int n = argc > 4 ? atoi(argv[4]) : 64;
      if (argc > 5) { m.temp = atof(argv[5]); m.top_p = 0.95f; m.top_k = 20; m.sample_seed = 0x1234ABCDull; }
      std::vector<uint32_t> prompt(ids.begin(), ids.end());
      auto t0 = std::chrono::steady_clock::now();
      opus::SpecStats st;
      m.t_encode = m.t_host = 0;
      auto sp = m.generate_spec(prompt, n, &st);
      if (const char* p = getenv("OPUS_SPEC_IDS")) {
        if (FILE* g = fopen(p, "wb")) { fwrite(sp.data(), 4, sp.size(), g); fclose(g); }
      }
      fprintf(stderr, "spec CPU: encode %.1f ms, host prep %.1f ms\n", m.t_encode, m.t_host);
      auto t1 = std::chrono::steady_clock::now();
      // plain greedy
      m.reset();
      std::vector<uint32_t> last;
      for (size_t i = 0; i < prompt.size(); i += opus::MP)
        last = m.step(std::vector<uint32_t>(prompt.begin() + i, prompt.begin() + std::min(prompt.size(), i + opus::MP)));
      std::vector<uint32_t> pl{last.back()};
      double dec_ms = 0;
      auto t2 = std::chrono::steady_clock::now();
      while ((int)pl.size() < n) { double ms; pl.push_back(m.step({pl.back()}, &ms)[0]); dec_ms += ms; }
      auto t3 = std::chrono::steady_clock::now();
      size_t same = 0;
      while (same < pl.size() && sp[same] == pl[same]) same++;
      const double ws = std::chrono::duration<double>(t1 - t0).count(), wp = std::chrono::duration<double>(t3 - t2).count();
      printf("spec == plain for %zu/%zu tokens%s\n", same, pl.size(), same == pl.size() ? " (identical)" : "  MISMATCH");
      printf("plain: %.2f ms/tok GPU, %.1f tok/s wall (decode only)\n", dec_ms / (n - 1), (n - 1) / wp);
      printf("spec : %u rounds, %.2f drafts/round, %.2f accepted drafts/round, %.2f tok/round; verify %.2f ms/round, drafts %.2f ms/round; %.1f tok/s wall incl. prefill %.0f ms\n",
             st.rounds, double(st.drafted) / std::max(1u, st.rounds), double(st.accepted) / std::max(1u, st.rounds), double(n - 1) / std::max(1u, st.rounds),
             st.verify_ms / std::max(1u, st.rounds), st.draft_ms / std::max(1u, st.rounds), n / ws, st.prefill_ms);
      printf("chain: %.2f rows/round; lookup proposed %u, tested %u, accepted %u, overrides of MTP %u; decode-only %.1f tok/s (GPU verify+draft)\n",
             double(st.rows) / std::max(1u, st.rounds), st.lk_proposed, st.lk_tested, st.lk_accepted, st.lk_override,
             (n - 1) / ((st.verify_ms + st.draft_ms) * 1e-3));
      printf("wall : decode loop %.1f ms for %d tokens = %.1f tok/s wall; %.2f ms/round wall vs %.2f ms/round GPU\n", st.wall_ms, n - 1,
             (n - 1) / (st.wall_ms * 1e-3), st.wall_ms / std::max(1u, st.rounds), (st.verify_ms + st.draft_ms) / std::max(1u, st.rounds));
      printf("streak: %u chunks (%u rolled back), %u tokens, %.1f ms wall\n", st.streak_chunks, st.streak_fail, st.streak_tokens, st.streak_ms);
      printf("tol  : %u tokens accepted within tolerance\n", st.tol_accepted);
      printf("conf : %.2f drafts made/round, %u follow-up draft batches; calibration %.2f %.2f %.2f %.2f %.2f %.2f\n",
             double(st.drafts_made) / std::max(1u, st.rounds), st.ext_batches, m.conf_rate[0], m.conf_rate[1], m.conf_rate[2],
             m.conf_rate[3], m.conf_rate[4], m.conf_rate[5]);
      for (int i = 0; i < std::min(n, 24); i++) printf("%u ", sp[i]);
      printf("\n");
      return same == pl.size() ? 0 : 1;
    } else if (mode == "ptest") {
      // prefill (one big chunk) vs decode-path chunks (16 rows): the next-token logits and the
      // continuation must be bit-identical. opus <pack> ptest <ids.bin> <T> <gen>
      const size_t T = std::min<size_t>(ids.size(), argc > 4 ? atoi(argv[4]) : 300);
      const int G = argc > 5 ? atoi(argv[5]) : 8;
      std::vector<uint32_t> p(ids.begin(), ids.begin() + T);
      auto run = [&](size_t chunk, std::vector<float>& lg) {
        m.reset();
        std::vector<uint32_t> last;
        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < T; i += chunk) {
          std::vector<uint32_t> c(p.begin() + i, p.begin() + std::min(T, i + chunk));
          last = m.forward(c, 0, nullptr, &lg);
        }
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::vector<float> keep(lg.end() - m.vocab, lg.end());
        lg = keep;
        std::vector<uint32_t> out{last.back()};
        for (int g = 1; g < G; g++) out.push_back(m.forward({out.back()}, 0)[0]);
        fprintf(stderr, "  chunk %4zu: prefill %.0f tok/s\n", chunk, T / s);
        return out;
      };
      std::vector<float> a, b;
      auto oa = run(opus::MAXM, a), ob = run(opus::MP, b);
      bool same = memcmp(a.data(), b.data(), a.size() * 4) == 0 && oa == ob;
      if (const char* dp = getenv("OPUS_PTEST_DUMP")) {  // debug: <dp>.a / <dp>.b final logits, continuations on stderr
        FILE* fa = fopen((std::string(dp) + ".a").c_str(), "wb"); fwrite(a.data(), 4, a.size(), fa); fclose(fa);
        FILE* fb = fopen((std::string(dp) + ".b").c_str(), "wb"); fwrite(b.data(), 4, b.size(), fb); fclose(fb);
        for (auto* o : {&oa, &ob}) { for (auto t : *o) fprintf(stderr, " %u", t); fprintf(stderr, "\n"); }
      }
      printf("ptest: %zu-token prompt, 16-row vs %u-row prefill: logits %s, continuation %s\n", T, opus::MP,
             memcmp(a.data(), b.data(), a.size() * 4) == 0 ? "bit-identical" : "DIFFER", oa == ob ? "identical" : "DIFFERS");
      return same ? 0 : 1;
    } else if (mode == "ctest") {
      // checkpoint resume: conversation A, then prompt B sharing only a prefix with A; the
      // continuation of B must equal a fresh run of B. opus <pack> ctest <ids.bin> <spec 0|1>
      const bool sp = argc > 4 && atoi(argv[4]);
      std::vector<uint32_t> A(ids.begin(), ids.begin() + 5000), B(ids.begin(), ids.begin() + 4500);
      B.insert(B.end(), ids.begin() + 6000, ids.begin() + 6500);
      auto run = [&](const std::vector<uint32_t>& p, opus::SpecStats* st) {
        std::vector<uint32_t> o;
        m.generate(p, 24, sp, [&](uint32_t t) { o.push_back(t); return true; }, st);
        return o;
      };
      m.reset();
      run(A, nullptr);
      opus::SpecStats st;
      auto b1 = run(B, &st);
      m.reset();
      auto b2 = run(B, nullptr);
      printf("ctest(spec=%d): resumed from %u of %zu shared tokens; continuation %s\n", sp, st.reused, size_t(4500),
             b1 == b2 ? "identical" : "DIFFERS");
      return b1 == b2 ? 0 : 1;
    } else if (mode == "btest") {
      // effective weight bandwidth of joint decode steps: opus <pack> btest <ids.bin> [T] [warm]
      // Independent lanes (lane k at ids[k * stride ..], warmed with `warm` tokens), T joint one-row steps per
      // lane count (BTEST_LANES, default 1,2,4,8,12,16). Bytes per step: every dense matmul's weights (counted
      // in mm, per op tag), distinct experts per layer x the expert gate|up / down bytes, the KV and block keys
      // attention reads; reported against GPU time. BTEST_PROF=1 adds a profiled pass per lane count: each op
      // tag's GPU time (per-op timestamps: an op's own duration, they do not sum to the step), bytes and GB/s.
      // OPUS_SKIP=<tag> (timing ablation) gives an op's in-chain cost as the drop in ms/step.
      const uint32_t T = argc > 4 ? atoi(argv[4]) : 24, WARM = argc > 5 ? atoi(argv[5]) : 1024;
      const opus::Layer& l0 = m.L[0];
      auto lb = [](const opus::Lin& w) { return double(w.N) * w.K * w.bits / 8 + double(w.N) * (w.K / w.gs) * 2; };
      const double gub = lb(l0.moe_gu), dnb = lb(l0.moe_dn), ebytes = gub + dnb;
      uint32_t nattn = 0;
      for (auto& l : m.L) nattn += l.attn;
      std::vector<uint32_t> counts;
      {
        std::string s = getenv("BTEST_LANES") ? getenv("BTEST_LANES") : "1,2,4,8,12,16";
        for (size_t i = 0; i < s.size();) { size_t j = s.find(',', i); if (j == std::string::npos) j = s.size(); counts.push_back(atoi(s.substr(i, j - i).c_str())); i = j + 1; }
      }
      const uint32_t NMAX = *std::max_element(counts.begin(), counts.end());
      if (NMAX > uint32_t(opus::NSLOT) || NMAX > opus::MAXM) mtl::die("btest: more lanes than NSLOT or MAXM");
      const size_t STR = (ids.size() - WARM - T - 1) / NMAX;
      std::vector<opus::Model*> L{&m};
      for (uint32_t k = 1; k < NMAX; k++) L.push_back(new opus::Model(argv[1], m.cap, &m));
      m.dump_nslots = m.C.buf(size_t(m.nL) * 4);
      const bool prof = getenv("BTEST_PROF") != nullptr;
      printf("btest: one expert %.2f MB (gate|up + down), %u attention layers, warm %u tokens%s%s\n", ebytes / 1e6, nattn, WARM,
             getenv("OPUS_SKIP") ? ", OPUS_SKIP=" : "", getenv("OPUS_SKIP") ? getenv("OPUS_SKIP") : "");
      for (uint32_t NL : counts) {
        for (uint32_t k = 0; k < NL; k++) {  // every lane count starts from the same warm state
          L[k]->reset();
          L[k]->feed(std::vector<uint32_t>(ids.begin() + k * STR, ids.begin() + k * STR + WARM), false, nullptr);
        }
        std::vector<opus::Model*> act(L.begin(), L.begin() + NL);
        double gms = 0, dense = 0, experts = 0, kv = 0, ex_layers = 0;
        std::map<std::string, double> tag_bytes, tag_ms;
        for (uint32_t t = 0; t < T; t++) {
          std::vector<std::vector<uint32_t>> tk(NL);
          for (uint32_t k = 0; k < NL; k++) tk[k] = {ids[k * STR + act[k]->pos]};
          double kvs = 0;
          for (uint32_t k = 0; k < NL; k++)  // keys each lane's attention reads: all below the QSA budget, else the budget
            kvs += double(nattn) * (std::min<uint32_t>(act[k]->pos + 1, m.budget) * m.nkv * m.hd * 2 * 2 +
                                    double(act[k]->pos / m.ratio) * m.ihd * 4);
          kv += kvs;
          m.count_bytes = true;
          m.dense_bytes = 0;
          m.bytes_by_tag.clear();
          double ms = 0;
          m.forward_lanes(act, tk, 0, nullptr, &ms);
          m.count_bytes = false;
          gms += ms;
          dense += m.dense_bytes;
          for (auto& [k, b] : m.bytes_by_tag) tag_bytes[k] += b;
          double ns_sum = 0;
          const uint32_t* ns = (const uint32_t*)m.dump_nslots.contents;
          for (uint32_t li = 0; li < m.nL; li++) ns_sum += ns[li];
          experts += ns_sum * ebytes;
          ex_layers += ns_sum;
          tag_bytes["mm.moe.gu"] += ns_sum * gub;
          tag_bytes["mm.moe.dn"] += ns_sum * dnb;
          tag_bytes["attn (kv)"] += kvs;
        }
        const double bytes = (dense + experts + kv) / T, ms = gms / T;
        printf("btest %2u lanes: %6.2f ms/step GPU | %5.1f distinct experts/layer | GB/step: dense %.2f, experts %.2f, kv %.2f, total %.2f | %4.0f GB/s | %5.1f tok/s\n",
               NL, ms, ex_layers / T / m.nL, dense / T / 1e9, experts / T / 1e9, kv / T / 1e9, bytes / 1e9, bytes / ms / 1e6, NL * 1000.0 / ms);
        if (!prof) continue;
        // profiled pass: the same steps again from the same warm state, per-op timestamps
        for (uint32_t k = 0; k < NL; k++) {
          L[k]->reset();
          L[k]->feed(std::vector<uint32_t>(ids.begin() + k * STR, ids.begin() + k * STR + WARM), false, nullptr);
        }
        for (uint32_t t = 0; t < T; t++) {
          std::vector<std::vector<uint32_t>> tk(NL);
          for (uint32_t k = 0; k < NL; k++) tk[k] = {ids[k * STR + act[k]->pos]};
          m.forward_lanes(act, tk, 0, nullptr, nullptr, &tag_ms);
        }
        std::vector<std::pair<double, std::string>> v;
        double tot = 0;
        for (auto& [k, x] : tag_ms) { v.push_back({x / T, k}); tot += x / T; }
        std::sort(v.rbegin(), v.rend());
        printf("  %-18s %8s %6s %8s %7s\n", "op", "ms/step", "share", "GB/step", "GB/s");
        for (auto& [x, k] : v) {
          // attention ops share the KV bytes; they are reported once, on attn_rows
          const double b = k == "attn_rows" ? tag_bytes["attn (kv)"] / T : (tag_bytes.count(k) ? tag_bytes[k] / T : 0.0);
          if (b > 0) printf("  %-18s %8.3f %5.1f%% %8.3f %7.0f\n", k.c_str(), x, 100 * x / tot, b / 1e9, b / x / 1e6);
          else printf("  %-18s %8.3f %5.1f%%\n", k.c_str(), x, 100 * x / tot);
        }
        printf("  sum of profiled ops %.2f ms vs %.2f ms unprofiled step\n", tot, ms);
      }
      return 0;
    } else if (mode == "dtest") {
      // per-layer dump: opus <pack> dtest <ids.bin> <out> [P] [M]. Prefills ids[0:P], then one mode-0 step
      // of ids[P:P+M] writing every layer's residual streams and pending MoE output (f32) to <out>, then the
      // step's logits. Compare two builds (e.g. OPUS_MAXM 16 vs 32) layer by layer.
      const uint32_t P = argc > 5 ? atoi(argv[5]) : 300, M = argc > 6 ? atoi(argv[6]) : 32;
      const size_t W = size_t(m.H) * m.HC;
      m.dump_h = m.C.buf(size_t(m.nL) * M * W * 4);
      m.dump_moe = m.C.buf(size_t(m.nL) * M * m.H * 4);
      id<MTLBuffer> dh = m.dump_h, dm = m.dump_moe;
      m.dump_h = nil;
      m.feed(std::vector<uint32_t>(ids.begin(), ids.begin() + P), false, nullptr);
      m.dump_h = dh;
      m.want_all = true;
      std::vector<float> lg;
      m.forward(std::vector<uint32_t>(ids.begin() + P, ids.begin() + P + M), 0, nullptr, &lg);
      FILE* f = fopen(argv[4], "wb");
      fwrite(dh.contents, 1, dh.length, f);
      fwrite(dm.contents, 1, dm.length, f);
      fwrite(lg.data(), 4, lg.size(), f);
      fclose(f);
      printf("dtest: %u layers x %u rows, W %zu, H %u, logits rows %zu\n", m.nL, M, W, m.H, lg.size() / m.vocab);
      return 0;
    } else if (mode == "ftest") {
      // fork: opus <pack> ftest <ids.bin> [T]. For several fork points f, slot A prefills ids[0:f], slot B
      // becomes its copy (fork_from); both then prefill ids[f:f+300] and decode T teacher-forced tokens.
      // B's logits must equal A's bit for bit at every position.
      const uint32_t T = argc > 4 ? atoi(argv[4]) : 32;
      opus::Model B(argv[1], m.cap, &m);
      const uint32_t V = m.vocab;
      int bad = 0;
      for (size_t f : {size_t(1), size_t(700), size_t(1024), size_t(2049), size_t(4321)}) {
        if (f + 300 + T + 8 > m.cap || f + 300 + T > ids.size()) mtl::die("ftest: ids or cap too short");
        m.reset();
        B.reset();
        m.feed(std::vector<uint32_t>(ids.begin(), ids.begin() + f), false, nullptr);
        if (!getenv("OPUS_FTEST_NOFORK")) B.fork_from(m);  // negative control: B starts empty
        std::vector<uint32_t> tail(ids.begin() + f, ids.begin() + f + 300);
        m.feed(tail, false, nullptr);
        B.feed(tail, false, nullptr);
        size_t ok = 0;
        for (uint32_t i = 0; i < T; i++) {
          std::vector<float> la, lb;
          m.forward({ids[f + 300 + i]}, 0, nullptr, &la);
          B.forward({ids[f + 300 + i]}, 0, nullptr, &lb);
          ok += la.size() == V && memcmp(la.data(), lb.data(), size_t(V) * 4) == 0;
        }
        printf("ftest fork at %zu: %zu/%u positions bit-identical\n", f, ok, T);
        bad |= ok != T;
      }
      return bad;
    } else if (mode == "kvtest") {
      // kv8 fidelity: opus <pack> kvtest <ids.bin> [N] [every]. Two models on the same weights, A with bf16
      // K/V caches and B with kv8, read ids[0:N] teacher-forced in 16-row steps; on every `every`-th step,
      // per row KL(A || B) and top-1 agreement over the full vocabulary, by context depth.
      const size_t N = argc > 4 ? size_t(atol(argv[4])) : ids.size();
      const uint32_t every = argc > 5 ? uint32_t(atoi(argv[5])) : 4;
      if (N > ids.size() || N + 16 > m.cap) mtl::die("kvtest: N exceeds ids or OPUS_CLI_CAP");
      opus::kv8_force = 0;
      opus::Model A(argv[1], m.cap, &m);
      opus::kv8_force = 1;
      opus::Model B(argv[1], m.cap, &m);
      opus::kv8_force = -1;
      if (A.kv8 || !B.kv8) mtl::die("kvtest: kv8 override did not take");
      const uint32_t V = m.vocab;
      const size_t edges[] = {2048, 8192, 32768, 131072, SIZE_MAX};
      double kl[5] = {0}, klmax[5] = {0};
      size_t rows[5] = {0}, agree[5] = {0};
      std::vector<double> la_(V), lb_(V);
      for (size_t i = 0, step = 0; i + 16 <= N; i += 16, step++) {
        std::vector<uint32_t> c(ids.begin() + i, ids.begin() + i + 16);
        if (step % every) { A.forward(c, 0); B.forward(c, 0); continue; }
        std::vector<float> la, lb;
        A.forward(c, 0, nullptr, &la);
        B.forward(c, 0, nullptr, &lb);
        for (uint32_t r = 0; r < 16; r++) {
          const float* pa = la.data() + size_t(r) * V;
          const float* pb = lb.data() + size_t(r) * V;
          double ma = -1e30, mb = -1e30;
          uint32_t ta = 0, tb = 0;
          for (uint32_t v = 0; v < V; v++) {
            if (pa[v] > ma) { ma = pa[v]; ta = v; }
            if (pb[v] > mb) { mb = pb[v]; tb = v; }
          }
          double za = 0, zb = 0;
          for (uint32_t v = 0; v < V; v++) { za += exp(double(pa[v]) - ma); zb += exp(double(pb[v]) - mb); }
          const double lza = log(za) + ma, lzb = log(zb) + mb;
          double d = 0;
          for (uint32_t v = 0; v < V; v++) {
            const double lpa = double(pa[v]) - lza, lpb = double(pb[v]) - lzb;
            d += exp(lpa) * (lpa - lpb);
          }
          size_t b = 0;
          while (i + r >= edges[b]) b++;
          kl[b] += d; klmax[b] = std::max(klmax[b], d); rows[b]++; agree[b] += ta == tb;
        }
      }
      const char* names[] = {"0-2k", "2k-8k", "8k-32k", "32k-128k", "128k+"};
      for (int b = 0; b < 5; b++)
        if (rows[b]) printf("kvtest %-9s %6zu rows: mean KL %.5f, max %.4f, top-1 agree %.2f%%\n", names[b], rows[b],
                            kl[b] / rows[b], klmax[b], 100.0 * agree[b] / rows[b]);
      return 0;
    } else if (mode == "kftest") {
      // fork from a checkpoint: opus <pack> kftest <ids.bin> [T]. For several prompt splits f, model A
      // prefills ids[0:f+1500] (it runs ahead of the fork point, as a busy donor lane does), B forks from
      // A's newest checkpoint at or below f (fork_prefix) and prefills the rest of ids[0:f+300]; reference
      // R prefills ids[0:f+300] from scratch. B and R then decode T teacher-forced tokens: logits must be
      // equal bit for bit at every position.
      const uint32_t T = argc > 4 ? atoi(argv[4]) : 32;
      opus::Model B(argv[1], m.cap, &m), R(argv[1], m.cap, &m);
      const uint32_t V = m.vocab;
      int bad = 0;
      for (size_t f : {size_t(2100), size_t(3000), size_t(4500), size_t(5333)}) {
        if (f + 1500 + 8 > m.cap || f + 300 + T > ids.size()) mtl::die("kftest: ids or cap too short");
        m.reset(); B.reset(); R.reset();
        m.feed(std::vector<uint32_t>(ids.begin(), ids.begin() + f + 1500), false, nullptr);
        const size_t p = B.fork_prefix(m, f);
        if (!p) mtl::die("kftest: no checkpoint below the fork point");
        B.feed(std::vector<uint32_t>(ids.begin() + p, ids.begin() + f + 300), false, nullptr);
        R.feed(std::vector<uint32_t>(ids.begin(), ids.begin() + f + 300), false, nullptr);
        size_t ok = 0;
        for (uint32_t i = 0; i < T; i++) {
          std::vector<float> lb, lr;
          B.forward({ids[f + 300 + i]}, 0, nullptr, &lb);
          R.forward({ids[f + 300 + i]}, 0, nullptr, &lr);
          ok += lb.size() == V && memcmp(lb.data(), lr.data(), size_t(V) * 4) == 0;
        }
        printf("kftest split %zu, forked at %zu: %zu/%u positions bit-identical\n", f, p, ok, T);
        bad |= ok != T;
      }
      return bad;
    } else if (mode == "ntest") {
      // n lanes: opus <pack> ntest <idsA.bin> <idsB.bin> [NL] [T] [rows] [stride] [OFF]
      // Lane 0 reads idsA[i]; lane k>0 reads idsB[OFF + k*STR + i] after a solo warmup of OFF tokens, so the
      // lanes sit at unrelated positions. Each lane is first run ALONE (decode + RA-row verify blocks) to
      // record the reference logits at every position. Then all lanes run TOGETHER: joint decode, then joint
      // verify blocks with per-lane commits. Every joint position must equal the solo reference bit for bit.
      const uint32_t NL = argc > 5 ? atoi(argv[5]) : 2, T = argc > 6 ? atoi(argv[6]) : 32;
      const uint32_t RA = argc > 7 ? atoi(argv[7]) : 1, STR = argc > 8 ? atoi(argv[8]) : 32;
      const uint32_t OFF = argc > 9 ? atoi(argv[9]) : 5;
      auto idsB = read_ids(argv[4]);
      if (NL < 2 || NL > uint32_t(opus::NSLOT)) mtl::die("ntest: 2..NSLOT lanes");
      if (uint64_t(NL) * RA > opus::MAXM) mtl::die("ntest: NL * rows exceeds MAXM");
      const uint32_t V = m.vocab;
      auto tok_of = [&](uint32_t k, size_t i) -> uint32_t {
        if (k == 0) return ids[i];
        const size_t j = OFF + size_t(k) * STR + i;
        if (j >= idsB.size()) mtl::die("ntest: lane past the end of idsB");
        return idsB[j];
      };
      std::vector<opus::Model*> L;
      for (uint32_t k = 0; k < NL; k++) L.push_back(k ? new opus::Model(argv[1], m.cap, &m) : &m);
      // ---- solo reference: decode T positions, then verify the same positions in RA-row blocks
      std::vector<std::vector<std::vector<float>>> S(NL), SV(NL);
      for (uint32_t k = 0; k < NL; k++) { S[k].resize(T); SV[k].resize(T); }
      double solo_decode = 0;
      for (uint32_t k = 0; k < NL; k++) {
        L[k]->reset();
        for (size_t i = 0; i < OFF && k; i++) L[k]->forward({tok_of(k, i)}, 0);
        for (size_t i = 0; i < T; i++) {
          double one = 0;
          L[k]->forward({tok_of(k, i + (k ? OFF : 0))}, 0, &one, &S[k][i]);
          solo_decode += one;
        }
      }
      // verify reference: fresh lane, warmup, then RA-row verify blocks (state only advanced by commit of 1)
      for (uint32_t k = 0; k < NL; k++) {
        L[k]->reset();
        for (size_t i = 0; i < OFF && k; i++) L[k]->forward({tok_of(k, i)}, 0);
        for (size_t b = 0; b < T; b += RA) {
          const uint32_t nb = uint32_t(std::min<size_t>(RA, T - b));
          std::vector<uint32_t> t;
          for (uint32_t r = 0; r < nb; r++) t.push_back(tok_of(k, b + r + (k ? OFF : 0)));
          std::vector<float> lg;
          L[k]->forward(t, 1, nullptr, &lg);
          for (uint32_t r = 0; r < nb; r++)
            SV[k][b + r].assign(lg.begin() + size_t(r) * V, lg.begin() + size_t(r + 1) * V);
          L[k]->commit(nb);  // solo accepts every row
        }
      }
      const double solo_avg = solo_decode / NL / T;
      // ---- joint decode: every lane one row per joint step
      for (auto* pr : L) pr->reset();
      for (uint32_t k = 1; k < NL; k++)
        for (size_t i = 0; i < OFF; i++) L[k]->forward({tok_of(k, i)}, 0);
      size_t ok = 0, tot = 0;
      double jsum = 0, maxms = 0;
      for (size_t i = 0; i < T; i++) {
        std::vector<std::vector<uint32_t>> tk(NL);
        for (uint32_t k = 0; k < NL; k++) tk[k] = {tok_of(k, i + (k ? OFF : 0))};
        std::vector<std::vector<float>> lg;
        double one = 0;
        m.forward_lanes(L, tk, 0, &lg, &one);
        jsum += one;
        maxms = std::max(maxms, one);
        for (uint32_t k = 0; k < NL; k++) { ok += memcmp(S[k][i].data(), lg[k].data(), size_t(V) * 4) == 0; tot++; }
      }
      const double jdec = jsum / T;
      printf("ntest decode %2u lanes: %zu/%zu bit-identical | joint %6.2f ms/step vs %6.2f ms summed solo (%.2fx)\n",
             NL, ok, tot, jdec, solo_avg * NL, solo_avg * NL / jdec);
      // ---- joint verify with per-lane commits
      for (auto* pr : L) pr->reset();
      for (uint32_t k = 1; k < NL; k++)
        for (size_t i = 0; i < OFF; i++) L[k]->forward({tok_of(k, i)}, 0);
      std::vector<size_t> ptr(NL, 0);
      for (uint32_t k = 0; k < NL; k++) ptr[k] = k ? OFF : 0;
      size_t vok = 0, vtot = 0, vsteps = 0;
      double vsum = 0;
      uint32_t seedr = 777;
      while ([&] { for (uint32_t k = 0; k < NL; k++) if (ptr[k] < T) return true; return false; }()) {
        std::vector<opus::Model*> ln;
        std::vector<std::vector<uint32_t>> tk;
        std::vector<uint32_t> nb(NL, 0), row0(NL, 0), lidx(NL, 0);
        uint32_t acc = 0;
        for (uint32_t k = 0; k < NL; k++) {
          if (ptr[k] >= T) continue;
          nb[k] = uint32_t(std::min<size_t>(RA, T - ptr[k]));
          std::vector<uint32_t> t;
          for (uint32_t r = 0; r < nb[k]; r++) t.push_back(tok_of(k, ptr[k] + r));
          row0[k] = acc; lidx[k] = uint32_t(ln.size()); acc += nb[k];
          ln.push_back(L[k]);
          tk.push_back(t);
        }
        std::vector<std::vector<float>> lg;
        double one = 0;
        m.forward_lanes(ln, tk, 1, &lg, &one);
        vsum += one; vsteps++;
        for (uint32_t k = 0; k < NL; k++) {
          if (ptr[k] >= T) continue;
          const size_t base = k ? ptr[k] - OFF : ptr[k];  // position index into S/SV
          for (uint32_t r = 0; r < nb[k]; r++)
            vok += memcmp(SV[k][base + r].data(), lg[lidx[k]].data() + size_t(r) * V, size_t(V) * 4) == 0;
          vtot += nb[k];
          const uint32_t j = 1 + (seedr = seedr * 1103515245u + 12345u, (seedr >> 16) % nb[k]);
          L[k]->commit_lane(m, row0[k], j);
          ptr[k] += j;
        }
      }
      printf("ntest verify %2u lanes x %u rows: %zu/%zu bit-identical | joint %6.2f ms/step  (%zu joint steps)\n",
             NL, RA, vok, vtot, vsum / std::max<size_t>(vsteps, 1), vsteps);
      return (vok == vtot && ok == tot) ? 0 : 1;
    } else if (mode == "laneprof") {
      // step-time curve: opus <pack> laneprof <idsA.bin> <idsB.bin> [rows per lane] [STR] [OFF] [throwoff]
      // Times a joint step at N = 1..NSLOT lanes with `rows` rows each, no correctness work, then the same
      // N lanes served one after another (N x solo step). throwoff=0 enumerates tl_list lanes properly.
      auto idsB = read_ids(argv[4]);
      const uint32_t R = argc > 5 ? atoi(argv[5]) : 1, STR = argc > 6 ? atoi(argv[6]) : 40;
      const uint32_t OFF = argc > 7 ? atoi(argv[7]) : 6;
      const size_t T = 64;
      std::vector<opus::Model*> L;
      for (uint32_t k = 0; k < uint32_t(opus::NSLOT); k++) L.push_back(k ? new opus::Model(argv[1], m.cap, &m) : &m);
      auto tok_of = [&](uint32_t k, size_t i) -> uint32_t {
        if (k == 0) return ids[i];
        const size_t j = OFF + size_t(k) * STR + i;
        return j < idsB.size() ? idsB[j] : ids[i];
      };
      for (auto* p : L) p->reset();
      for (uint32_t k = 1; k < uint32_t(opus::NSLOT); k++)
        for (size_t i = 0; i < OFF; i++) L[k]->forward({tok_of(k, i)}, 0);
      printf("laneprof: %u rows per lane, MAXM %u\n", R, opus::MAXM);
      double prev = 0;
      for (uint32_t N = 1; N <= uint32_t(opus::NSLOT); N++) {
        if (uint64_t(N) * R > opus::MAXM) break;
        std::vector<opus::Model*> ln(L.begin(), L.begin() + N);
        auto batch = [&](uint32_t base) {
          std::vector<std::vector<uint32_t>> tk(N);
          for (uint32_t k = 0; k < N; k++)
            for (uint32_t r = 0; r < R; r++) tk[k].push_back(tok_of(k, base + r));
          return tk;
        };
        std::vector<std::vector<float>> lg;
        double wms;
        m.forward_lanes(ln, batch(3), 1, &lg, &wms);  // warm
        double jsum = 0;
        const int REPS = 5;
        for (int rep = 0; rep < REPS; rep++) {
          m.forward_lanes(ln, batch(3), 1, &lg, &wms);
          jsum += wms;
        }
        const double joint = jsum / REPS;
        printf("N=%u rows=%u  joint %6.2f ms  per-lane %6.2f ms  aggregate %5.1f%%  marginal %+6.2f ms/lane\n",
               N, N * R, joint, joint / N, 100.0 * N * R / joint, N == 1 ? 0.0 : joint - prev);
        prev = joint;
      }
      return 0;
    } else if (mode == "jtest") {
      // joint speculation: opus <pack> jtest <ids.bin> <n> [NL] [spec 0|1]
      // NL lanes each generate n tokens from a distinct slice of ids: first alone through generate()
      // (the reference), then together through generate_spec_lanes. Token streams must be identical.
      const uint32_t ntok = argc > 4 ? atoi(argv[4]) : 48;
      const uint32_t NL = argc > 5 ? atoi(argv[5]) : 2;
      const bool sp = argc > 6 ? atoi(argv[6]) : 1;
      const size_t SL = 48;  // prompt length per lane
      if (ids.size() < SL * NL) mtl::die("jtest: prompt file too short for NL lanes");
      std::vector<std::vector<uint32_t>> prompts(NL);
      for (uint32_t k = 0; k < NL; k++) prompts[k].assign(ids.begin() + k * SL, ids.begin() + k * SL + SL);
      std::vector<opus::Model*> L;
      for (uint32_t k = 0; k < NL; k++) L.push_back(k ? new opus::Model(argv[1], m.cap, &m) : &m);
      // reference: each lane alone, spec or plain
      std::vector<std::vector<uint32_t>> ref(NL);
      double solo_wall = 0;
      for (uint32_t k = 0; k < NL; k++) {
        opus::SpecStats s1;
        L[k]->reset();
        const auto t0 = std::chrono::steady_clock::now();
        if (sp) ref[k] = L[k]->generate_spec(prompts[k], ntok, &s1);
        else {
          L[k]->generate(prompts[k], ntok, false, [&](uint32_t t) { ref[k].push_back(t); return true; }, &s1);
        }
        solo_wall += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      }
      // joint decoding stops a lane at end-of-sequence; the reference runs on past it
      for (auto& r : ref)
        if (auto it = std::find(r.begin(), r.end(), m.eos); it != r.end()) r.erase(it + 1, r.end());
      // joint
      opus::SpecStats sj;
      m.t_encode = m.t_host = 0;
      const auto j0 = std::chrono::steady_clock::now();
      auto got = m.generate_spec_lanes(m, L, prompts, ntok, &sj);
      const double joint_wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - j0).count();
      size_t ok = 0, tot = 0;
      for (uint32_t k = 0; k < NL; k++) {
        const size_t nn = std::min(ref[k].size(), got[k].size());
        for (size_t i = 0; i < nn; i++) { ok += ref[k][i] == got[k][i]; tot++; }
        ok += (ref[k].size() == got[k].size()) * 1;
        tot += 1;
      }
      for (uint32_t k = 0; k < NL; k++) {
        size_t i = 0;
        while (i < ref[k].size() && i < got[k].size() && ref[k][i] == got[k][i]) i++;
        if (i < ref[k].size() || i < got[k].size())
          fprintf(stderr, "LANE %u: ref %zu tokens, joint %zu, first difference at %zu (ref %d, joint %d, eos %u)\n", k, ref[k].size(),
                  got[k].size(), i, i < ref[k].size() ? int(ref[k][i]) : -1, i < got[k].size() ? int(got[k][i]) : -1, m.eos);
      }
      for (uint32_t k = 0; k < NL; k++) {
        fprintf(stderr, "OUT ref[%u] ", k);
        for (size_t i = 0; i < std::min<size_t>(ref[k].size(), 12); i++) fprintf(stderr, "%u ", ref[k][i]);
        fprintf(stderr, "\nOUT got[%u] ", k);
        for (size_t i = 0; i < std::min<size_t>(got[k].size(), 12); i++) fprintf(stderr, "%u ", got[k][i]);
        fprintf(stderr, "\n");
      }
      printf("jtest spec=%d %u lanes x %u tokens: %zu/%zu match | rounds %u, %.2f tok/round, %.2f drafts/round, %.2f accepted/round\n",
             int(sp), NL, ntok, ok, tot, sj.rounds, NL * ntok / double(std::max(1u, sj.rounds)),
             double(sj.drafted) / std::max(1u, sj.rounds), double(sj.accepted) / std::max(1u, sj.rounds));
      printf("jtest GPU per round: drafts %.2f ms, verify %.2f ms (%.1f rows), commit %.2f ms\n", sj.draft_ms / std::max(1u, sj.rounds),
             sj.verify_ms / std::max(1u, sj.rounds), double(sj.rows) / std::max(1u, sj.rounds), sj.commit_ms / std::max(1u, sj.rounds));
      printf("jtest wall: solo %.0f ms (%.1f tok/s/lane), joint %.0f ms (%.1f tok/s/lane, %.1f aggregate)\n",
             solo_wall, NL * ntok * 1000.0 / solo_wall / NL, joint_wall, NL * ntok * 1000.0 / joint_wall / NL,
             NL * ntok * 1000.0 / joint_wall);
      return ok == tot ? 0 : 1;
    } else if (mode == "ltest") {
      // lanes: opus <pack> ltest <idsA.bin> <idsB.bin> [T] [rowsA] [rowsB]
      // Two conversations, each run alone (one token per step), then together: (1) one token per lane
      // per joint step, (2) joint verify blocks of rowsA / rowsB rows with pseudo-random acceptance and
      // per-lane commits. Every position's logits must equal the solo run's bit for bit.
      auto idsB = read_ids(argv[4]);
      const size_t T = std::min<size_t>(std::min(ids.size(), idsB.size() - (argc > 8 ? atoi(argv[8]) : 5)), argc > 5 ? atoi(argv[5]) : 32);
      const uint32_t RA = argc > 6 ? atoi(argv[6]) : 3, RB = argc > 7 ? atoi(argv[7]) : 5, OFF = argc > 8 ? atoi(argv[8]) : 5;
      opus::Model m2(argv[1], m.cap, &m);
      const uint32_t V = m.vocab;
      std::vector<std::vector<float>> SA(T), SB(T + OFF);
      double solo_ms = 0, ms = 0;
      m.reset();
      for (size_t i = 0; i < T; i++) { m.forward({ids[i]}, 0, &ms, &SA[i]); solo_ms += ms; }
      m2.reset();
      for (size_t i = 0; i < T + OFF; i++) { m2.forward({idsB[i]}, 0, &ms, &SB[i]); if (i >= OFF) solo_ms += ms; }
      auto same = [&](const std::vector<float>& a, const float* b) { return memcmp(a.data(), b, size_t(V) * 4) == 0; };
      // (1) joint decode, lane B five positions ahead of lane A
      m.reset(); m2.reset();
      for (size_t i = 0; i < OFF; i++) m2.forward({idsB[i]}, 0);
      size_t okA = 0, okB = 0;
      double joint_ms = 0;
      for (size_t i = 0; i < T; i++) {
        std::vector<std::vector<float>> lg;
        m.forward_lanes({&m, &m2}, {{ids[i]}, {idsB[OFF + i]}}, 0, &lg, &ms);
        joint_ms += ms;
        okA += same(SA[i], lg[0].data());
        okB += same(SB[OFF + i], lg[1].data());
      }
      printf("ltest decode: lane A %zu/%zu, lane B %zu/%zu bit-identical; %.2f ms per joint step vs %.2f ms for the two solo steps\n",
             okA, T, okB, T, joint_ms / T, solo_ms / T);
      // (2) joint verify + per-lane commit
      m.reset(); m2.reset();
      for (size_t i = 0; i < OFF; i++) m2.forward({idsB[i]}, 0);
      size_t iA = 0, iB = 0, vA = 0, vB = 0, steps = 0;
      uint32_t seedr = 12345;
      double vms = 0;
      while (iA < T || iB < T) {
        std::vector<opus::Model*> ln;
        std::vector<std::vector<uint32_t>> tk;
        uint32_t nA = 0, nB = 0;
        if (iA < T) { nA = uint32_t(std::min<size_t>(RA, T - iA)); ln.push_back(&m); tk.emplace_back(ids.begin() + iA, ids.begin() + iA + nA); }
        if (iB < T) { nB = uint32_t(std::min<size_t>(RB, T - iB)); ln.push_back(&m2); tk.emplace_back(idsB.begin() + OFF + iB, idsB.begin() + OFF + iB + nB); }
        std::vector<std::vector<float>> lg;
        m.forward_lanes(ln, tk, 1, &lg, &ms);
        vms += ms;
        steps++;
        size_t k = 0;
        uint32_t row0 = 0;
        if (nA) {
          seedr = seedr * 1103515245u + 12345u;
          const uint32_t j = 1 + (seedr >> 16) % nA;
          for (uint32_t r = 0; r < j; r++) vA += same(SA[iA + r], lg[k].data() + size_t(r) * V);
          m.commit_lane(m, row0, j);
          iA += j; row0 += nA; k++;
        }
        if (nB) {
          seedr = seedr * 1103515245u + 12345u;
          const uint32_t j = 1 + (seedr >> 16) % nB;
          for (uint32_t r = 0; r < j; r++) vB += same(SB[OFF + iB + r], lg[k].data() + size_t(r) * V);
          m2.commit_lane(m, row0, j);
          iB += j;
        }
      }
      printf("ltest verify: lane A %zu/%zu, lane B %zu/%zu bit-identical over %zu joint verify steps (%u + %u rows), %.2f ms per step\n",
             vA, T, vB, T, steps, RA, RB, vms / steps);
      return okA == T && okB == T && vA == T && vB == T ? 0 : 1;
    } else if (mode == "vtest") {
      // verify/commit invariance: logits of every position fed one token at a time vs fed
      // through 4-row verify blocks with pseudo-random acceptance; must match bit for bit.
      const size_t T = std::min<size_t>(ids.size(), argc > 4 ? atoi(argv[4]) : 64);
      const uint32_t V = m.vocab;
      std::vector<std::vector<float>> A(T), B(T);
      m.reset();
      for (size_t i = 0; i < T; i++) { std::vector<float> lg; m.forward({ids[i]}, 0, nullptr, &lg); A[i] = lg; }
      m.reset();
      uint32_t seedr = 12345;
      size_t i = 0, blocks = 0;
      while (i < T) {
        const uint32_t BS = argc > 5 ? atoi(argv[5]) : 4;
        const uint32_t nb = uint32_t(std::min<size_t>(BS, T - i));
        std::vector<uint32_t> blk(ids.begin() + i, ids.begin() + i + nb);
        std::vector<float> lg;
        m.forward(blk, 1, nullptr, &lg);
        seedr = seedr * 1103515245u + 12345u;
        const uint32_t j = 1 + (seedr >> 16) % nb;
        for (uint32_t r = 0; r < j; r++) B[i + r].assign(lg.begin() + size_t(r) * V, lg.begin() + size_t(r + 1) * V);
        m.commit(j);
        i += j;
        blocks++;
      }
      size_t same = 0, worst = 0;
      double maxd = 0;
      for (size_t t = 0; t < T; t++) {
        bool eq = memcmp(A[t].data(), B[t].data(), size_t(V) * 4) == 0;
        same += eq;
        for (uint32_t v = 0; v < V; v++) {
          double d = fabs(double(A[t][v]) - double(B[t][v]));
          if (d > maxd) { maxd = d; worst = t; }
        }
      }
      printf("vtest: %zu positions, %zu verify blocks: %zu/%zu bit-identical, max |diff| %.3g (pos %zu)\n", T, blocks, same, T, maxd, worst);
      return same == T ? 0 : 1;
    } else if (mode == "rtest") {
      // expert sharing inside a block of M rows (last layer's routing): opus <pack> rtest <ids.bin> <M> <ctx>
      const uint32_t M = argc > 4 ? atoi(argv[4]) : 16, ctx = argc > 5 ? atoi(argv[5]) : 512;
      for (size_t i = 0; i < ctx; i += opus::MP)
        m.step(std::vector<uint32_t>(ids.begin() + i, ids.begin() + std::min<size_t>(i + opus::MP, ctx)));
      double sum = 0; uint32_t blocks = 0, mn = 1u << 30, mx = 0;
      for (size_t i = ctx; i + M <= ids.size() && blocks < 40; i += M, blocks++) {
        m.step(std::vector<uint32_t>(ids.begin() + i, ids.begin() + i + M));
        const uint32_t* t = (const uint32_t*)m.topi.contents;
        std::vector<uint32_t> v(t, t + M * m.topk);
        std::sort(v.begin(), v.end());
        const uint32_t d = uint32_t(std::unique(v.begin(), v.end()) - v.begin());
        sum += d; mn = std::min(mn, d); mx = std::max(mx, d);
      }
      printf("rtest M=%u: %u blocks, distinct experts per block mean %.1f (min %u, max %u) of %u slots -> %.0f%% of the slot path's expert bytes\n",
             M, blocks, sum / blocks, mn, mx, M * m.topk, 100.0 * sum / blocks / (M * m.topk));
    } else if (mode == "mprof") {
      // per-op GPU time of one single-row MTP draft call: opus <pack> mprof <ids.bin> [steps]
      const uint32_t steps = argc > 4 ? atoi(argv[4]) : 40;
      for (size_t i = 0; i < 256 && i < ids.size(); i += opus::MP)
        m.step(std::vector<uint32_t>(ids.begin() + i, ids.begin() + std::min<size_t>(i + opus::MP, 256)));
      std::map<std::string, double> prof;
      double plain = 0;
      for (uint32_t s = 0; s < steps; s++) {
        double ms;
        m.mtp_forward({ids[s]}, m.seed_h, 0, &ms);
        plain += ms;
        m.mtp_profile = &prof;
        m.mtp_forward({ids[s]}, m.seed_h, 0, nullptr);
        m.mtp_profile = nullptr;
      }
      double tot = 0;
      std::vector<std::pair<double, std::string>> v;
      for (auto& [k, ms] : prof) { v.push_back({ms / steps, k}); tot += ms / steps; }
      std::sort(v.rbegin(), v.rend());
      printf("MTP draft call %.3f ms (unprofiled)  sum of ops %.3f ms (profiled)\n", plain / steps, tot);
      for (auto& [ms, k] : v) printf("  %-22s %8.3f ms  %5.1f%%\n", k.c_str(), ms, 100 * ms / tot);
    } else if (mode == "prof") {
      // per-op GPU time of decode steps at M rows: opus <pack> prof <ids.bin> <M> <steps>
      uint32_t M = argc > 4 ? atoi(argv[4]) : 1, steps = argc > 5 ? atoi(argv[5]) : 20;
      const uint32_t warm = getenv("OPUS_PROF_CTX") ? atoi(getenv("OPUS_PROF_CTX")) : 256;
      for (size_t i = 0; i < warm && i < ids.size(); i += opus::MP)
        m.step(std::vector<uint32_t>(ids.begin() + i, ids.begin() + std::min<size_t>(i + opus::MP, warm)));
      std::map<std::string, double> prof;
      double plain = 0;
      for (uint32_t s = 0; s < steps; s++) {
        std::vector<uint32_t> t(ids.begin(), ids.begin() + M);
        double ms;
        m.step(t, &ms);
        plain += ms;
        m.pos -= M; m.history.resize(m.history.size() - M);
        m.step(t, nullptr, nullptr, &prof);
        m.pos -= M; m.history.resize(m.history.size() - M);
      }
      double tot = 0;
      std::vector<std::pair<double, std::string>> v;
      for (auto& [k, ms] : prof) { v.push_back({ms / steps, k}); tot += ms / steps; }
      std::sort(v.rbegin(), v.rend());
      printf("M=%u  step %.3f ms (unprofiled)  sum of ops %.3f ms (profiled)\n", M, plain / steps, tot);
      for (auto& [ms, k] : v) printf("  %-22s %8.3f ms  %5.1f%%\n", k.c_str(), ms, 100 * ms / tot);
    } else if (mode == "idletest") {
      // first-step latency after an idle gap: opus <pack> idletest <ids.bin> <idle seconds>
      const int idle = argc > 4 ? atoi(argv[4]) : 20;
      m.step(std::vector<uint32_t>(ids.begin(), ids.begin() + 64));
      auto timed = [&](const char* tag) {
        for (int i = 0; i < 4; i++) {
          auto t0 = std::chrono::steady_clock::now();
          m.step({ids[64 + i]});
          m.pos -= 1; m.history.resize(m.history.size() - 1);
          printf("%s step %d: %.1f ms wall\n", tag, i,
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
      };
      timed("warm");
      sleep(idle);
      timed("after idle");
    } else if (mode == "gen") {
      int n = argc > 4 ? atoi(argv[4]) : 32;
      double pre_ms = 0;
      std::vector<uint32_t> last;
      auto t0 = std::chrono::steady_clock::now();
      for (size_t i = 0; i < ids.size(); i += opus::MP) {
        std::vector<uint32_t> chunk(ids.begin() + i, ids.begin() + std::min(ids.size(), i + opus::MP));
        double ms;
        last = m.step(chunk, &ms);
        pre_ms += ms;
      }
      auto t1 = std::chrono::steady_clock::now();
      uint32_t tok = last.back();
      std::vector<uint32_t> out{tok};
      double dec_ms = 0;
      for (int i = 1; i < n; i++) {
        double ms;
        tok = m.step({tok}, &ms)[0];
        dec_ms += ms;
        out.push_back(tok);
      }
      auto t2 = std::chrono::steady_clock::now();
      for (auto t : out) printf("%u ", t);
      printf("\n");
      double wall_pre = std::chrono::duration<double>(t1 - t0).count(), wall_dec = std::chrono::duration<double>(t2 - t1).count();
      fprintf(stderr, "prefill %zu tok: %.0f ms GPU (%.0f tok/s wall)\ndecode %d tok: %.2f ms/tok GPU, %.1f tok/s wall\n",
              ids.size(), pre_ms, ids.size() / wall_pre, n - 1, dec_ms / std::max(1, n - 1), (n - 1) / wall_dec);
    }
  }
}
