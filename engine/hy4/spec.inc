// opus-hy4 speculation round, shared by `spec` (main.mm) and `serve` (serve.inc): DFly2 drafts merged with
// context-lookup drafts. The lookup indexes every 3-gram of the committed history; where the current tail
// occurred before, what followed it there is a proposal (OpenCode traffic copies a lot: file contents, tool
// arguments, earlier code). Every proposal goes through the exact verify, so output equals plain decoding;
// proposals change only speed. Ported from the Flash-Next engine's hybrid chain (engine/step.inc).
// joint(): the same per lane for 2+ concurrent requests, all lanes' rows verified in one step_lanes().
#include <functional>
#include "../lookup.h"

namespace hy4spec {

using Lookup = opus_lookup::Lookup;

struct Stats {
  uint32_t rounds = 0, accepted = 0, rows = 0, drafts = 0, lk_only = 0, lk_proposed = 0, lk_tested = 0, lk_accepted = 0;
  double draft_ms = 0, verify_ms = 0;
  // per round: drafts accepted (a) and candidates verified (c.size()), 0..MAXM
  uint32_t acc_hist[hy4::MAXM + 1] = {}, cand_hist[hy4::MAXM + 1] = {}, full = 0;
};

// one round's candidates after the anchor, and what acceptance needs to score them
struct Prop {
  uint32_t p0 = 0;
  std::vector<uint32_t> c, clen;
  std::vector<uint8_t> src;
  Lookup* L = nullptr;
  double draft_ms = 0;
};

// one concurrent request in a joint round
struct Lane {
  int slot = 0;
  uint32_t b = 0, sx = 0, ss = 0;  // anchor (at the slot's pos, not yet fed) and the drafter's source rows
  bool src = false;                // sx / ss are rows of the last target step (xf / lsel still hold them)
  bool spec = true;                // propose candidates (else one row a round)
  hy4::Model::LaneBatch samp;      // sampling parameters (toks unused)
  std::function<bool(uint32_t)> is_stop;
  Stats st;
  std::vector<uint32_t> out;       // the round's tokens: b, then the accepted candidates
};

struct Spec {
  std::vector<Lookup> lk;  // one per conversation slot
  // HY4_LOOKUP=0: DFly2 drafts only
  bool on = hy4::cfg().lookup;
  // rows past the drafter's drafts are verified while the chain's estimated probability stays >= pmin
  double pmin = hy4::cfg().lk_pmin;  // swept 0.05-0.5 2026-11-19: 0.05-0.15 flat, 0.3+ cuts lookup-only chains short
  // a lookup-only round (the drafter is not run) when the match is at least skip_len tokens long and that
  // length's measured acceptance is at least skip_rate
  uint32_t skip_len = hy4::cfg().lk_skip;
  double skip_rate = hy4::cfg().lk_skip_rate;
  uint32_t emax = std::min(hy4::cfg().lk_emax, hy4::MAXM - 1);
  Stats st;

  Lookup& cur(hy4::Model& m) {
    if (lk.size() < m.slots_.size()) lk.resize(m.slots_.size());
    return lk[m.cur_slot];
  }
  // a new request on this slot: the index follows the slot's history (truncated to the common prefix)
  void begin(hy4::Model& m) { if (on) cur(m).sync(m.history, true); }
  // rows a round may feed (the caller falls back to plain steps near the capacity)
  uint32_t max_rows(const hy4::Model& m) const { return std::max(emax, m.dblock - 1) + 1; }

  // Merge DFly2 drafts (d, dprobs) with lookup continuation starting at (cp, cl), up to cmax candidates.
  void merge_candidates(const hy4::Model& m, Prop& pr, const std::vector<uint32_t>& d, const std::vector<float>& dprobs,
                        std::vector<uint32_t>& tail, uint32_t cp, uint32_t cl, uint32_t cmax,
                        const std::function<bool(uint32_t)>& is_stop) const {
    const uint32_t K = uint32_t(d.size());
    auto& c = pr.c;
    if (!on) {
      c.assign(d.begin(), d.begin() + std::min(K, cmax));
      pr.src.assign(c.size(), 0);
      pr.clen.assign(c.size(), 0);
      return;
    }
    Lookup* L = pr.L;
    double p = 1.0;
    bool dr_ok = true;
    while (c.size() < cmax) {
      const uint32_t i = uint32_t(c.size());
      const bool hm = dr_ok && i < K;
      const bool hl = cp != Lookup::NONE && cp < m.history.size();
      if (!hm && !hl) break;
      const double pl = hl ? L->rate[Lookup::bucket(cl)] : 0.0;
      const double pm = hm ? double(dprobs[i]) : 0.0;
      uint32_t t;
      uint8_t s;
      if (hm && hl && d[i] == m.history[cp]) { t = d[i]; s = 2; p *= std::max(pm, pl); }
      else if (hm && (!hl || pl <= pm)) { t = d[i]; s = 0; p *= pm; }
      else { t = m.history[cp]; s = 1; p *= pl; dr_ok = false; }
      if (i >= K && p < pmin) break;
      c.push_back(t);
      pr.src.push_back(s);
      pr.clen.push_back(cl);
      tail.push_back(t);
      if (is_stop(t)) break;
      if (hl && m.history[cp] == t) { cp++; cl++; }
      else cp = L->find(m.history, tail, &cl);
    }
  }

  // Candidates after anchor b (the token at m.pos of the current slot): DFly2 drafts when the drafter's source
  // rows are valid (src) merged with lookup proposals, at most lim of them. prefetch: start the anchor's Engram
  // fetch behind the draft (single-lane only: step() consumes it, step_lanes() does not).
  Prop propose(hy4::Model& m, uint32_t b, uint32_t sx, uint32_t ss, bool src, uint32_t lim, bool prefetch,
               const std::function<bool(uint32_t)>& is_stop) {
    Prop pr;
    pr.p0 = m.pos;
    const uint32_t cmax = std::min(emax, lim);
    if (cmax == 0) return pr;
    std::vector<uint32_t> tail;
    uint32_t cp = Lookup::NONE, cl = 0;
    if (on) {
      pr.L = &cur(m);
      pr.L->sync(m.history, false);
      tail.assign(m.history.end() - long(std::min<size_t>(m.history.size(), Lookup::MAXLEN)), m.history.end());
      tail.push_back(b);
      cp = pr.L->find(m.history, tail, &cl);
    }
    const bool lk_only = cp != Lookup::NONE && cl >= skip_len && pr.L->rate[Lookup::bucket(cl)] >= skip_rate;
    std::vector<uint32_t> d;
    if (lk_only) st.lk_only++;
    else if (src) {
      if (prefetch) m.prefetch_anchor(b);
      d = m.draft(b, sx, ss, &pr.draft_ms);
      st.drafts++;
    }
    merge_candidates(m, pr, d, m.last_probs, tail, cp, cl, cmax, is_stop);
    return pr;
  }

  // After the verify step: g holds the step's tokens for this request's rows (b, then pr.c), which start at
  // row0 of the step. Appends b and the accepted candidates to out, drops the rejected rows from the current
  // slot; b, sx, ss become the next round's anchor and drafter source rows.
  void finish(hy4::Model& m, const Prop& pr, const uint32_t* g, uint32_t row0, double verify_ms, uint32_t& b, uint32_t& sx,
              uint32_t& ss, std::vector<uint32_t>& out) {
    const auto& c = pr.c;
    uint32_t a = 0;
    while (a < c.size() && c[a] == g[a]) a++;
    const uint32_t nb = g[a];
    if (pr.L)
      for (uint32_t i = 0; i < c.size() && i <= a; i++) {
        if (pr.src[i] != 1) continue;
        double& r = pr.L->rate[Lookup::bucket(pr.clen[i])];
        r = 0.9 * r + 0.1 * (i < a ? 1.0 : 0.0);
        st.lk_tested++;
        st.lk_accepted += i < a;
      }
    for (auto s : pr.src) st.lk_proposed += s == 1;
    out.push_back(b);
    out.insert(out.end(), c.begin(), c.begin() + a);
    m.truncate(pr.p0 + a + 1);
    b = nb;
    sx = ss = row0 + a;
    st.rounds++;
    st.accepted += a;
    st.rows += uint32_t(c.size()) + 1;
    st.acc_hist[std::min<uint32_t>(a, hy4::MAXM)]++;
    st.cand_hist[std::min<size_t>(c.size(), hy4::MAXM)]++;
    st.full += c.size() > 0 && a == c.size();
    st.draft_ms += pr.draft_ms;
    st.verify_ms += verify_ms;
  }

  // One round from anchor b (the token at m.pos, not yet fed): drafts, the verify step, acceptance. Appends
  // b and the accepted drafts to out; b, sx, ss become the next round's anchor and drafter source rows.
  void round(hy4::Model& m, uint32_t& b, uint32_t& sx, uint32_t& ss, std::vector<uint32_t>& out,
             const std::function<bool(uint32_t)>& is_stop) {
    const Prop pr = propose(m, b, sx, ss, true, emax, true, is_stop);
    std::vector<uint32_t> rows{b};
    rows.insert(rows.end(), pr.c.begin(), pr.c.end());
    double g2 = 0;
    auto g = m.step(rows, &g2);
    const size_t o0 = out.size();
    finish(m, pr, g.data(), 0, g2, b, sx, ss, out);
    // HY4_SPEC_DUMP=path: one JSON line per round with every draft's confidence signals and the accepted count
    // (with HY4_DRAFT_CUT=0 HY4_DRAFT_ADAPT=0 HY4_LOOKUP=0 every draft is verified: offline policy replay)
    static FILE* dump = getenv("HY4_SPEC_DUMP") ? fopen(getenv("HY4_SPEC_DUMP"), "a") : nullptr;
    if (dump && !m.last_probs.empty()) {
      auto arr = [&](const char* k, const std::vector<float>& v) {
        fprintf(dump, ",\"%s\":[", k);
        for (size_t i = 0; i < v.size(); i++) fprintf(dump, "%s%.6g", i ? "," : "", std::isfinite(v[i]) ? v[i] : 1e9f);
        fprintf(dump, "]");
      };
      fprintf(dump, "{\"a\":%zu,\"k\":%zu,\"T\":%.3f", out.size() - o0 - 1, pr.c.size(), m.temp);
      arr("p", m.last_probs);
      arr("pm", m.last_pmax);
      arr("mg", m.last_marg);
      fprintf(dump, "}\n");
      fflush(dump);
    }
  }

  // Joint speculation: each lane proposes within a dynamic share of `rows` (its anchor included), one
  // draft_lanes() drafts all needing lanes together while prefetching all anchors' Engram rows, leftover
  // rows from draft_cut / non-spec lanes are redistributed to lookup-extending lanes, one step_lanes()
  // verifies every lane's rows, and each lane accepts its own prefix. Every lane's output equals plain decoding.
  void joint(hy4::Model& m, const std::vector<Lane*>& lanes, uint32_t rows, double* gpu_ms = nullptr) {
    const uint32_t n = uint32_t(lanes.size());
    const uint32_t budget = std::max<uint32_t>(n, std::min(rows, hy4::MAXM));
    std::vector<Prop> pr(n);
    std::vector<hy4::Model::LaneBatch> bs(n);
    std::vector<std::vector<uint32_t>> tails(n);
    std::vector<uint32_t> cps(n, Lookup::NONE), cls(n, 0), cmaxs(n, 0);
    std::vector<bool> can_spec(n, false), is_lk_only(n, false), drafted_flag(n, false);
    std::vector<hy4::Model::DraftReq> dreqs;
    std::vector<uint32_t> dmap;
    std::vector<hy4::Model::AnchorLaneReq> areqs;
    areqs.reserve(n);
    const Stats keep = st;
    uint32_t n_spec = 0;
    for (uint32_t j = 0; j < n; j++) {
      Lane& l = *lanes[j];
      m.use_slot(l.slot);
      pr[j].p0 = m.pos;
      areqs.push_back({l.slot, l.b});
      if (l.spec && m.pos + max_rows(m) + 1 < m.cap) { can_spec[j] = true; n_spec++; }
    }
    const uint32_t spec_budget = budget - (n - n_spec);
    const uint32_t base_share = n_spec ? std::max<uint32_t>(1, spec_budget / n_spec) : 1;
    uint32_t rem_share = n_spec ? (spec_budget - base_share * n_spec) : 0;
    for (uint32_t j = 0; j < n; j++) {
      Lane& l = *lanes[j];
      if (!can_spec[j]) continue;
      m.use_slot(l.slot);
      const uint32_t share = base_share + (rem_share ? (rem_share--, 1u) : 0u);
      const uint32_t cmax = std::min(emax, share - 1);
      cmaxs[j] = cmax;
      if (cmax == 0) continue;
      if (on) {
        pr[j].L = &cur(m);
        pr[j].L->sync(m.history, false);
        tails[j].assign(m.history.end() - long(std::min<size_t>(m.history.size(), Lookup::MAXLEN)), m.history.end());
        tails[j].push_back(l.b);
        cps[j] = pr[j].L->find(m.history, tails[j], &cls[j]);
      }
      const bool lk_only = cps[j] != Lookup::NONE && cls[j] >= skip_len && pr[j].L->rate[Lookup::bucket(cls[j])] >= skip_rate;
      is_lk_only[j] = lk_only;
      if (lk_only) {
        l.st.lk_only++;
      } else if (l.src) {
        hy4::Model::DraftReq rq;
        rq.slot = l.slot; rq.anchor = l.b; rq.src_x = l.sx; rq.src_s = l.ss;
        rq.max_drafts = std::min(m.dblock - 1, cmax);
        rq.temp = l.samp.temp; rq.min_p = l.samp.min_p; rq.top_p = l.samp.top_p; rq.top_k = l.samp.top_k; rq.seed = l.samp.seed;
        dreqs.push_back(rq);
        dmap.push_back(j);
        drafted_flag[j] = true;
        l.st.drafts++;
      }
    }
    std::vector<std::vector<uint32_t>> d_toks(n);
    std::vector<std::vector<float>> d_probs(n);
    if (!dreqs.empty()) m.prefetch_anchors_lanes(areqs);
    if (dreqs.size() == 1) {
      const uint32_t j = dmap[0];
      Lane& l = *lanes[j];
      m.use_slot(l.slot);
      m.temp = l.samp.temp; m.min_p = l.samp.min_p; m.top_p = l.samp.top_p; m.top_k = l.samp.top_k; m.seed = l.samp.seed;
      d_toks[j] = m.draft(l.b, l.sx, l.ss, &pr[j].draft_ms);
      d_probs[j] = m.last_probs;
    } else if (dreqs.size() >= 2) {
      double dl_ms = 0;
      auto dres = m.draft_lanes(dreqs, &dl_ms);
      const double per_ms = dl_ms / double(dreqs.size());
      for (size_t k = 0; k < dreqs.size(); k++) {
        const uint32_t j = dmap[k];
        d_toks[j] = std::move(dres[k].toks);
        d_probs[j] = std::move(dres[k].probs);
        pr[j].draft_ms = per_ms;
      }
    }
    // Reclaim unused rows from draft_cut / non-lookup lanes and give them to lookup-extending lanes
    uint32_t spare = budget - n;
    for (uint32_t j = 0; j < n; j++) {
      if (drafted_flag[j] && cps[j] == Lookup::NONE)
        cmaxs[j] = std::min<uint32_t>(cmaxs[j], uint32_t(d_toks[j].size()));
      spare -= std::min(spare, cmaxs[j]);
    }
    while (spare > 0) {
      bool gave = false;
      for (uint32_t pass = 0; pass < 2 && spare > 0; pass++) {
        for (uint32_t j = 0; j < n && spare > 0; j++) {
          if (!can_spec[j] || cmaxs[j] >= emax) continue;
          if (pass == 0 ? is_lk_only[j] : (cps[j] != Lookup::NONE)) {
            cmaxs[j]++;
            spare--;
            gave = true;
          }
        }
      }
      if (!gave) break;
    }
    for (uint32_t j = 0; j < n; j++) {
      Lane& l = *lanes[j];
      m.use_slot(l.slot);
      if (cmaxs[j] > 0) merge_candidates(m, pr[j], d_toks[j], d_probs[j], tails[j], cps[j], cls[j], cmaxs[j], l.is_stop);
      bs[j] = l.samp;
      bs[j].slot = l.slot;
      bs[j].toks.assign(1, l.b);
      bs[j].toks.insert(bs[j].toks.end(), pr[j].c.begin(), pr[j].c.end());
    }
    double g2 = 0;
    const auto g = m.step_lanes(bs, &g2);
    if (gpu_ms) *gpu_ms = g2;
    uint32_t r0 = 0;
    for (uint32_t j = 0; j < n; j++) {
      Lane& l = *lanes[j];
      m.use_slot(l.slot);
      st = l.st;
      l.out.clear();
      finish(m, pr[j], g.data() + r0, r0, g2, l.b, l.sx, l.ss, l.out);
      l.st = st;
      l.src = true;
      r0 += uint32_t(bs[j].toks.size());
    }
    st = keep;
  }
};

}  // namespace hy4spec
