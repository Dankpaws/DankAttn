// DankAttn ERNIE-5.5 CLI & binary server:
//   dankattn-ernie55 <pack_or_http_url> score <ids.bin> <logits.f16>
//   dankattn-ernie55 <pack_or_http_url> gen <ids.bin> <n_tokens>
//   dankattn-ernie55 <pack_or_http_url> bench <ids.bin>
//   dankattn-ernie55 <pack_or_http_url> suite <suite_dir>
//   dankattn-ernie55 <pack_or_http_url> serve
#include "ernie55.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

static std::vector<uint32_t> read_ids(const char* p) {
  std::string s = mtl::slurp(p);
  std::vector<uint32_t> v(s.size() / 4);
  memcpy(v.data(), s.data(), v.size() * 4);
  return v;
}

static uint32_t argmax(const float* p, uint32_t n) {
  float m0 = p[0], m1 = p[0], m2 = p[0], m3 = p[0];
  uint32_t i = 0;
  for (; i + 3 < n; i += 4) {
    if (p[i] > m0) m0 = p[i];
    if (p[i + 1] > m1) m1 = p[i + 1];
    if (p[i + 2] > m2) m2 = p[i + 2];
    if (p[i + 3] > m3) m3 = p[i + 3];
  }
  float bv = std::max(std::max(m0, m1), std::max(m2, m3));
  for (; i < n; i++) if (p[i] > bv) bv = p[i];
  for (uint32_t j = 0; j < n; j++) if (p[j] == bv) return j;
  return 0;
}

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
  uint64_t next_u64() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  float next_f32() { return float((next_u64() >> 40) * (1.0 / 16777216.0)); }
};

struct SampleCfg {
  float temp = 0.0f;
  float top_p = 1.0f;
  uint32_t top_k = 0;
  float min_p = 0.0f;
};

// Build filtered probability distribution (sparse or dense) and sample; also supports exact spec verify.
static void build_probs(const float* logits, uint32_t V, const SampleCfg& sc,
                        std::vector<std::pair<float, uint32_t>>& cand, float& p_sum) {
  cand.clear();
  float mx = logits[0];
  for (uint32_t i = 1; i < V; i++) if (logits[i] > mx) mx = logits[i];
  const float inv_t = 1.0f / sc.temp;
  // Fast threshold: anything with (logit - mx)/temp < -18 contributes < 1.5e-8
  const float cut = mx - 18.0f * sc.temp;
  for (uint32_t i = 0; i < V; i++) {
    if (logits[i] >= cut) cand.emplace_back(std::exp((logits[i] - mx) * inv_t), i);
  }
  std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  if (sc.top_k > 0 && cand.size() > sc.top_k) cand.resize(sc.top_k);
  if (sc.min_p > 0.0f && !cand.empty()) {
    const float mp = cand[0].first * sc.min_p;
    size_t keep = 1;
    while (keep < cand.size() && cand[keep].first >= mp) keep++;
    cand.resize(keep);
  }
  double sum = 0.0;
  for (const auto& c : cand) sum += c.first;
  if (sc.top_p < 1.0f && cand.size() > 1) {
    double cum = 0.0;
    const double lim = sc.top_p * sum;
    size_t keep = 0;
    for (; keep < cand.size(); keep++) {
      cum += cand[keep].first;
      if (cum >= lim) { keep++; break; }
    }
    cand.resize(std::max<size_t>(1, keep));
    sum = 0.0;
    for (const auto& c : cand) sum += c.first;
  }
  p_sum = float(sum);
}

static uint32_t sample_logits(const float* logits, uint32_t V, const SampleCfg& sc, Rng& rng,
                              std::vector<std::pair<float, uint32_t>>& cand, uint32_t exclude = 0xFFFFFFFFu) {
  if (sc.temp <= 0.0f) return argmax(logits, V);
  float p_sum = 0.0f;
  build_probs(logits, V, sc, cand, p_sum);
  if (exclude != 0xFFFFFFFFu) {
    float rem = 0.0f;
    for (auto& c : cand) {
      if (c.second == exclude) c.first = 0.0f;
      else rem += c.first;
    }
    if (rem > 0.0f) p_sum = rem;
    else return argmax(logits, V);
  }
  float r = rng.next_f32() * p_sum;
  float acc = 0.0f;
  for (const auto& c : cand) {
    acc += c.first;
    if (r <= acc) return c.second;
  }
  return cand.empty() ? argmax(logits, V) : cand.back().second;
}

static float prob_of(const std::vector<std::pair<float, uint32_t>>& cand, float p_sum, uint32_t tok) {
  for (const auto& c : cand) if (c.second == tok) return c.first / std::max(p_sum, 1e-30f);
  return 0.0f;
}

// 3-gram lookup drafter (prompt + output PLD)
struct NgramTable {
  std::unordered_map<uint64_t, uint32_t> tab;
  static inline uint64_t key3(uint32_t a, uint32_t b, uint32_t c) {
    return ((uint64_t(a) * 1000003ULL + b) * 1000003ULL) + c;
  }
  void clear() { tab.clear(); }
  // Index 3-grams ending at positions [start .. end_excl-1]
  void index_range(const std::vector<uint32_t>& h, size_t start, size_t end_excl) {
    size_t i0 = std::max<size_t>(2, start);
    for (size_t i = i0; i < end_excl; i++) {
      tab[key3(h[i - 2], h[i - 1], h[i])] = uint32_t(i);
    }
  }
  // Draft up to max_k tokens matching the 3-gram ending at h.back()
  void draft(const std::vector<uint32_t>& h, uint32_t max_k, std::vector<uint32_t>& out) const {
    out.clear();
    if (max_k == 0 || h.size() < 3) return;
    const size_t n = h.size();
    const uint32_t a = h[n - 3], b = h[n - 2], c = h[n - 1];
    auto it = tab.find(key3(a, b, c));
    if (it == tab.end()) return;
    const uint32_t idx = it->second;
    if (idx < 2 || idx + 1 >= n) return;
    if (h[idx - 2] != a || h[idx - 1] != b || h[idx] != c) return;
    for (uint32_t j = 0; j < max_k && idx + 1 + j < n; j++) {
      out.push_back(h[idx + 1 + j]);
    }
  }
};

// Fast chunked prefill of toks[start..end-1]; returns last token's logits [vocab]
static const float* prefill_fast(ernie55::Model& m, const std::vector<uint32_t>& toks, size_t start, double& pms) {
  static const size_t chunk_sz = []() -> size_t {
    if (const char* e = getenv("E5_CHUNK")) {
      size_t v = size_t(atoi(e));
      if (v > 0 && v <= ernie55::PREFILL_M) return v;
    }
    return size_t(ernie55::PREFILL_M);
  }();
  const float* lg = nullptr;
  for (size_t i = start; i < toks.size(); i += chunk_sz) {
    const size_t e = std::min(toks.size(), i + chunk_sz);
    const bool is_last = (e == toks.size());
    std::vector<uint32_t> chunk(toks.begin() + i, toks.begin() + e);
    double gms = 0;
    lg = m.step(chunk, &gms, /*need_logits=*/is_last, /*last_only=*/true);
    pms += gms;
  }
  return lg;
}

// Binary framing helpers for serve mode (matches server/engine_frontend.py)
static bool read_exact(FILE* f, void* dst, size_t n) {
  return fread(dst, 1, n, f) == n;
}

struct ServeReq {
  uint32_t rid = 0;
  std::vector<uint32_t> ids;
  uint32_t max_new = 4096;
  uint32_t flags = 0;
  std::vector<uint32_t> stops;
  SampleCfg sc;
  uint64_t seed = 0;
  uint32_t think_budget = 0;
  uint32_t think_end = 0;
  std::vector<uint32_t> think_force;
};

static void send_u32_pair(uint32_t a, uint32_t b) {
  uint32_t buf[2] = {a, b};
  fwrite(buf, 4, 2, stdout);
  fflush(stdout);
}

static void send_error(uint32_t rid, const std::string& msg) {
  uint32_t hdr[3] = {rid, 0xFFFFFFFEu, uint32_t(msg.size())};
  fwrite(hdr, 4, 3, stdout);
  if (!msg.empty()) fwrite(msg.data(), 1, msg.size(), stdout);
  fflush(stdout);
}

static void send_done(uint32_t rid, uint32_t reused, uint32_t prompt, uint32_t generated,
                      uint32_t rounds, uint32_t accepted, uint32_t dflags, float prefill_ms, float wall_ms) {
  struct {
    uint32_t rid, tag, reused, prompt, generated, rounds, accepted, dflags;
    float prefill_ms, wall_ms;
  } pkt{rid, 0xFFFFFFFFu, reused, prompt, generated, rounds, accepted, dflags, prefill_ms, wall_ms};
  fwrite(&pkt, sizeof(pkt), 1, stdout);
  fflush(stdout);
}

static int run_serve(ernie55::Model& m) {
  const uint32_t k_draft = getenv("E5_DRAFT_K") ? uint32_t(atoi(getenv("E5_DRAFT_K"))) : 7u;
  std::mutex mu;
  std::condition_variable cv;
  std::deque<ServeReq> q;
  std::atomic<bool> eof{false};
  std::atomic<uint32_t> cancel_rid{0};

  // Signal ready to engine_frontend.py
  send_u32_pair(0u, 0xFFFFFFFDu);

  std::thread reader([&]() {
    while (true) {
      uint32_t magic = 0;
      if (!read_exact(stdin, &magic, 4)) break;
      if (magic == 0x5843504Fu) {
        uint32_t rid = 0;
        if (!read_exact(stdin, &rid, 4)) break;
        cancel_rid.store(rid, std::memory_order_relaxed);
        continue;
      }
      if (magic != 0x5355504Fu) {
        fprintf(stderr, "ernie55 serve: bad magic 0x%08x\n", magic);
        break;
      }
      ServeReq r;
      uint32_t n_ids = 0, n_stops = 0;
      if (!read_exact(stdin, &r.rid, 4) || !read_exact(stdin, &n_ids, 4)) break;
      r.ids.resize(n_ids);
      if (n_ids && !read_exact(stdin, r.ids.data(), size_t(n_ids) * 4)) break;
      if (!read_exact(stdin, &r.max_new, 4) || !read_exact(stdin, &r.flags, 4) || !read_exact(stdin, &n_stops, 4)) break;
      r.stops.resize(n_stops);
      if (n_stops && !read_exact(stdin, r.stops.data(), size_t(n_stops) * 4)) break;
      if (!read_exact(stdin, &r.sc.temp, 4) || !read_exact(stdin, &r.sc.top_p, 4) ||
          !read_exact(stdin, &r.sc.top_k, 4) || !read_exact(stdin, &r.seed, 8)) break;
      if (r.flags & 2u) {
        uint32_t grp[3];
        if (!read_exact(stdin, grp, 12)) break;
      }
      if (r.flags & 4u) {
        uint32_t nf = 0;
        if (!read_exact(stdin, &r.think_budget, 4) || !read_exact(stdin, &r.think_end, 4) || !read_exact(stdin, &nf, 4)) break;
        r.think_force.resize(nf);
        if (nf && !read_exact(stdin, r.think_force.data(), size_t(nf) * 4)) break;
      }
      if (r.flags & 16u) {
        if (!read_exact(stdin, &r.sc.min_p, 4)) break;
      }
      if (r.flags & 32u) {
        // Drain vision payload so stream stays aligned, then mark error
        uint32_t nv = 0;
        if (!read_exact(stdin, &nv, 4)) break;
        for (uint32_t vi = 0; vi < nv; vi++) {
          struct { uint32_t T, h, w, tok_off; uint64_t hash; } vh;
          if (!read_exact(stdin, &vh, sizeof(vh))) break;
          size_t nb = size_t(vh.T) * vh.h * vh.w * 3 * 2;
          std::vector<uint8_t> skip(nb);
          if (!read_exact(stdin, skip.data(), nb)) break;
        }
      }
      {
        std::lock_guard<std::mutex> lk(mu);
        q.push_back(std::move(r));
      }
      cv.notify_one();
    }
    eof.store(true);
    cv.notify_all();
  });

  std::vector<uint32_t> cached_ids;
  NgramTable ngram;
  std::vector<uint32_t> draft_toks;
  std::vector<std::pair<float, uint32_t>> cand;

  while (true) {
    ServeReq r;
    {
      std::unique_lock<std::mutex> lk(mu);
      cv.wait(lk, [&]() { return !q.empty() || eof.load(); });
      if (q.empty()) break;
      r = std::move(q.front());
      q.pop_front();
    }
    if (r.flags & 32u) {
      send_error(r.rid, "ERNIE-5.5 text engine does not support vision inputs");
      continue;
    }
    if (r.ids.empty()) {
      send_error(r.rid, "empty prompt");
      continue;
    }
    if (r.ids.size() >= m.cap) {
      send_error(r.rid, "prompt exceeds context capacity");
      continue;
    }

    auto tw0 = std::chrono::steady_clock::now();
    // Exact prefix reuse (safe when rollback distance < COMP_RING - 128 so HCA comp/swa rings are intact)
    size_t lcp = 0;
    const size_t max_lcp = std::min(cached_ids.size(), r.ids.size() - 1);
    while (lcp < max_lcp && cached_ids[lcp] == r.ids[lcp]) lcp++;
    if (lcp > 0 && (m.pos - uint32_t(lcp)) >= (ernie55::COMP_RING - 128u)) {
      lcp = 0;
    }
    if (lcp > 0) {
      m.truncate(uint32_t(lcp));
      cached_ids.resize(lcp);
    } else {
      m.reset();
      cached_ids.clear();
    }

    double pms = 0.0;
    const float* lg = prefill_fast(m, r.ids, lcp, pms);
    cached_ids = r.ids;

    std::unordered_set<uint32_t> stop_set(r.stops.begin(), r.stops.end());
    stop_set.insert(m.eos);
    Rng rng(r.seed);
    const bool use_spec = (r.flags & 1u) != 0 && k_draft > 0;
    ngram.clear();
    if (use_spec && cached_ids.size() >= 3) {
      ngram.index_range(cached_ids, 2, cached_ids.size() - 1);
    }

    uint32_t generated = 0, rounds = 0, accepted = 0, dflags = 0;
    bool hit_stop = false;
    bool in_think = (r.think_budget > 0);
    uint32_t think_toks = 0;

    uint32_t next_tok = sample_logits(lg, m.vocab, r.sc, rng, cand);
    auto emit_tok = [&](uint32_t t) -> bool {
      if (stop_set.count(t)) { hit_stop = true; return false; }
      send_u32_pair(r.rid, t);
      generated++;
      if (in_think) {
        if (t == r.think_end) in_think = false;
        else think_toks++;
      }
      return generated < r.max_new;
    };

    if (emit_tok(next_tok)) {
      while (generated < r.max_new && m.pos + 1 < m.cap && !hit_stop) {
        if (cancel_rid.load(std::memory_order_relaxed) == r.rid) break;

        // Enforce thinking budget if configured
        if (in_think && think_toks >= r.think_budget && !r.think_force.empty()) {
          in_think = false;
          std::vector<uint32_t> fseq;
          fseq.push_back(next_tok);
          for (uint32_t ft : r.think_force) {
            if (generated >= r.max_new) break;
            send_u32_pair(r.rid, ft);
            generated++;
            fseq.push_back(ft);
          }
          if (m.pos + fseq.size() >= m.cap) break;
          double gms = 0;
          lg = m.step(fseq, &gms, true, true);
          cached_ids.insert(cached_ids.end(), fseq.begin(), fseq.end());
          if (use_spec && cached_ids.size() >= 3) {
            ngram.index_range(cached_ids, cached_ids.size() - fseq.size(), cached_ids.size() - 1);
          }
          next_tok = sample_logits(lg, m.vocab, r.sc, rng, cand);
          if (!emit_tok(next_tok)) break;
          continue;
        }

        // Append next_tok to history for 3-gram lookup
        const size_t hist_prev = cached_ids.size();
        cached_ids.push_back(next_tok);

        draft_toks.clear();
        if (use_spec) {
          uint32_t avail = std::min<uint32_t>({k_draft, ernie55::MAXM - 1u, r.max_new - generated, m.cap - m.pos - 2u});
          if (avail > 0) ngram.draft(cached_ids, avail, draft_toks);
          if (hist_prev >= 2) ngram.index_range(cached_ids, hist_prev, cached_ids.size());
        }

        if (draft_toks.empty()) {
          double gms = 0;
          lg = m.step({next_tok}, &gms, true, false);
          rounds++;
          next_tok = sample_logits(lg, m.vocab, r.sc, rng, cand);
          if (!emit_tok(next_tok)) break;
        } else {
          // Speculative verify: step({next_tok, d[0], ..., d[k-1]})
          const uint32_t pos0 = m.pos;
          std::vector<uint32_t> vseq;
          vseq.reserve(1 + draft_toks.size());
          vseq.push_back(next_tok);
          for (uint32_t dt : draft_toks) vseq.push_back(dt);
          double gms = 0;
          lg = m.step(vseq, &gms, true, false);
          rounds++;

          uint32_t n_acc = 0;
          uint32_t bonus_tok = 0;
          bool all_acc = true;
          for (uint32_t i = 0; i < draft_toks.size(); i++) {
            const float* row = lg + size_t(i) * m.vocab;
            if (r.sc.temp <= 0.0f) {
              uint32_t t = argmax(row, m.vocab);
              if (t == draft_toks[i]) {
                n_acc++;
              } else {
                bonus_tok = t;
                all_acc = false;
                break;
              }
            } else {
              float p_sum = 0.0f;
              build_probs(row, m.vocab, r.sc, cand, p_sum);
              float pd = prob_of(cand, p_sum, draft_toks[i]);
              if (rng.next_f32() < pd) {
                n_acc++;
              } else {
                bonus_tok = sample_logits(row, m.vocab, r.sc, rng, cand, draft_toks[i]);
                all_acc = false;
                break;
              }
            }
          }
          if (all_acc) {
            bonus_tok = sample_logits(lg + size_t(draft_toks.size()) * m.vocab, m.vocab, r.sc, rng, cand);
          }
          accepted += n_acc;
          // Roll back unaccepted draft tokens (at most k_draft <= 7, always inside HCA_RING)
          m.truncate(pos0 + 1u + n_acc);

          // Emit accepted draft tokens + bonus token
          bool cont = true;
          for (uint32_t i = 0; i < n_acc; i++) {
            uint32_t dt = draft_toks[i];
            if (!emit_tok(dt)) { cont = false; break; }
            cached_ids.push_back(dt);
            if (cached_ids.size() >= 3) ngram.index_range(cached_ids, cached_ids.size() - 1, cached_ids.size());
          }
          if (!cont) break;
          next_tok = bonus_tok;
          if (!emit_tok(next_tok)) break;
        }
      }
    }

    if (!hit_stop && generated >= r.max_new) dflags |= 1u;
    float wall_ms = float(std::chrono::duration<double>(std::chrono::steady_clock::now() - tw0).count() * 1e3);
    send_done(r.rid, uint32_t(lcp), uint32_t(r.ids.size()), generated, rounds, accepted, dflags, float(pms), wall_ms);
  }

  reader.join();
  return 0;
}

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 3) {
      fprintf(stderr, "usage: dankattn-ernie55 <pack|url> score|gen|bench|suite|serve [args]\n");
      return 2;
    }
    const std::string pack = argv[1];
    const std::string mode = argv[2];
    const uint32_t cap = getenv("E5_CAP") ? uint32_t(atoi(getenv("E5_CAP"))) : 262144u;
    ernie55::Model m(pack, cap);

    if (mode == "serve") {
      return run_serve(m);
    }

    if (argc < 4) {
      fprintf(stderr, "usage: dankattn-ernie55 <pack|url> score|gen|bench|suite <path> [arg]\n");
      return 2;
    }
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
        const float* lg = m.step(chunk, &gms, true, false);
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
      const bool use_spec = getenv("E5_SPEC") && atoi(getenv("E5_SPEC")) != 0;
      double pms = 0;
      const float* lg = prefill_fast(m, ids, 0, pms);
      uint32_t next_tok = argmax(lg, m.vocab);
      fprintf(stderr, "prefill %zu tok: %.1f ms GPU (%.1f tok/s)\n", ids.size(), pms, ids.size() / (pms / 1e3));

      std::vector<uint32_t> hist = ids;
      NgramTable ngram;
      if (use_spec && hist.size() >= 3) ngram.index_range(hist, 2, hist.size() - 1);
      std::vector<uint32_t> gen_ids = {next_tok};
      std::vector<uint32_t> draft_toks;
      double dms = 0;
      uint32_t rounds = 0, accepted = 0;
      auto w0 = std::chrono::steady_clock::now();
      while (gen_ids.size() < ngen && next_tok != m.eos) {
        const size_t hp = hist.size();
        hist.push_back(next_tok);
        draft_toks.clear();
        if (use_spec) {
          uint32_t avail = std::min<uint32_t>({7u, ngen - uint32_t(gen_ids.size())});
          ngram.draft(hist, avail, draft_toks);
          if (hp >= 2) ngram.index_range(hist, hp, hist.size());
        }
        if (draft_toks.empty()) {
          double gms = 0;
          lg = m.step({next_tok}, &gms, true, false);
          dms += gms; rounds++;
          next_tok = argmax(lg, m.vocab);
          gen_ids.push_back(next_tok);
        } else {
          uint32_t pos0 = m.pos;
          std::vector<uint32_t> vseq = {next_tok};
          vseq.insert(vseq.end(), draft_toks.begin(), draft_toks.end());
          double gms = 0;
          lg = m.step(vseq, &gms, true, false);
          dms += gms; rounds++;
          uint32_t n_acc = 0, bonus = 0;
          bool all_acc = true;
          for (uint32_t i = 0; i < draft_toks.size(); i++) {
            uint32_t t = argmax(lg + size_t(i) * m.vocab, m.vocab);
            if (t == draft_toks[i]) n_acc++;
            else { bonus = t; all_acc = false; break; }
          }
          if (all_acc) bonus = argmax(lg + size_t(draft_toks.size()) * m.vocab, m.vocab);
          accepted += n_acc;
          m.truncate(pos0 + 1u + n_acc);
          for (uint32_t i = 0; i < n_acc && gen_ids.size() < ngen; i++) {
            gen_ids.push_back(draft_toks[i]);
            hist.push_back(draft_toks[i]);
            if (hist.size() >= 3) ngram.index_range(hist, hist.size() - 1, hist.size());
            if (draft_toks[i] == m.eos) { next_tok = m.eos; break; }
          }
          if (next_tok != m.eos && gen_ids.size() < ngen) {
            next_tok = bonus;
            gen_ids.push_back(next_tok);
          }
        }
      }
      double wms = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() * 1e3;
      double dtoks = gen_ids.size() > 1 ? double(gen_ids.size() - 1) : 1.0;
      fprintf(stderr, "decode %zu tok (rounds=%u acc=%u): %.1f ms GPU (%.1f tok/s), wall %.1f ms (%.1f tok/s)\n",
              gen_ids.size(), rounds, accepted, dms, dtoks / (dms / 1e3), wms, dtoks / (wms / 1e3));
      for (size_t i = 0; i < gen_ids.size(); i++)
        printf("%u%s", gen_ids[i], (i + 1 < gen_ids.size()) ? " " : "\n");
      return 0;
    }

    if (mode == "bench") {
      double pms = 0;
      auto pw0 = std::chrono::steady_clock::now();
      prefill_fast(m, ids, 0, pms);
      double pw = std::chrono::duration<double>(std::chrono::steady_clock::now() - pw0).count() * 1e3;
      fprintf(stderr, "bench prefill %zu tok: GPU %.0f ms (%.1f tok/s), wall %.0f ms (%.1f tok/s)\n",
              ids.size(), pms, ids.size() / (pms / 1e3), pw, ids.size() / (pw / 1e3));
      uint32_t p = m.pos;
      std::vector<uint32_t> one = {ids.back()};
      for (int w = 0; w < 3; w++) { m.step(one, nullptr, true, false); m.truncate(p); }
      double gt = 0;
      auto dw0 = std::chrono::steady_clock::now();
      for (int w = 0; w < 10; w++) { double g = 0; m.step(one, &g, true, false); gt += g; m.truncate(p); }
      double dw = std::chrono::duration<double>(std::chrono::steady_clock::now() - dw0).count() * 1e3;
      fprintf(stderr, "bench decode (pos=%u): GPU %.2f ms/tok (%.1f tok/s), wall %.2f ms/tok (%.1f tok/s)\n",
              p, gt / 10.0, 10000.0 / gt, dw / 10.0, 10000.0 / dw);
      return 0;
    }

    if (mode == "suite") {
      const std::string sdir = argv[3];
      const bool use_spec = getenv("E5_SPEC") && atoi(getenv("E5_SPEC")) != 0;
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
          const float* lg = m.step(chunk, &gms, true, false);
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
        const float* lg = prefill_fast(m, pids, 0, pms);
        uint32_t next_tok = argmax(lg, m.vocab);
        std::vector<uint32_t> hist = pids;
        NgramTable ngram;
        if (use_spec && hist.size() >= 3) ngram.index_range(hist, 2, hist.size() - 1);
        std::vector<uint32_t> gids = {next_tok};
        std::vector<uint32_t> draft_toks;
        double dms = 0;
        uint32_t rounds = 0, accepted = 0;
        auto dw0 = std::chrono::steady_clock::now();
        while (gids.size() < ngen && next_tok != m.eos) {
          const size_t hp = hist.size();
          hist.push_back(next_tok);
          draft_toks.clear();
          if (use_spec) {
            uint32_t avail = std::min<uint32_t>({7u, ngen - uint32_t(gids.size())});
            ngram.draft(hist, avail, draft_toks);
            if (hp >= 2) ngram.index_range(hist, hp, hist.size());
          }
          if (draft_toks.empty()) {
            double gms = 0;
            lg = m.step({next_tok}, &gms, true, false);
            dms += gms; rounds++;
            next_tok = argmax(lg, m.vocab);
            gids.push_back(next_tok);
          } else {
            uint32_t pos0 = m.pos;
            std::vector<uint32_t> vseq = {next_tok};
            vseq.insert(vseq.end(), draft_toks.begin(), draft_toks.end());
            double gms = 0;
            lg = m.step(vseq, &gms, true, false);
            dms += gms; rounds++;
            uint32_t n_acc = 0, bonus = 0;
            bool all_acc = true;
            for (uint32_t i = 0; i < draft_toks.size(); i++) {
              uint32_t t = argmax(lg + size_t(i) * m.vocab, m.vocab);
              if (t == draft_toks[i]) n_acc++;
              else { bonus = t; all_acc = false; break; }
            }
            if (all_acc) bonus = argmax(lg + size_t(draft_toks.size()) * m.vocab, m.vocab);
            accepted += n_acc;
            m.truncate(pos0 + 1u + n_acc);
            for (uint32_t i = 0; i < n_acc && gids.size() < ngen; i++) {
              gids.push_back(draft_toks[i]);
              hist.push_back(draft_toks[i]);
              if (hist.size() >= 3) ngram.index_range(hist, hist.size() - 1, hist.size());
              if (draft_toks[i] == m.eos) { next_tok = m.eos; break; }
            }
            if (next_tok != m.eos && gids.size() < ngen) {
              next_tok = bonus;
              gids.push_back(next_tok);
            }
          }
        }
        double dw = std::chrono::duration<double>(std::chrono::steady_clock::now() - dw0).count() * 1e3;
        double dtoks = gids.size() > 1 ? double(gids.size() - 1) : 1.0;
        fprintf(stderr, "[suite] %-12s prompt=%zu tok prefill=%.1f tok/s (GPU %.0fms) | gen=%zu tok (r=%u acc=%u) decode=%.1f tok/s GPU (%.1f wall)\n",
                name.c_str(), pids.size(), pids.size() / (pms / 1e3), pms,
                gids.size(), rounds, accepted, dtoks / (dms / 1e3), dtoks / (dw / 1e3));
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
