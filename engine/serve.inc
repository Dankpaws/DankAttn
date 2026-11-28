// serve mode: requests from the HTTP front end on stdin, tokens back on stdout.
//
// Protocol (all little-endian u32 unless noted):
//   in,  request: 'OPUS', rid, n, ids[n], max_new, flags (1 = speculate, 2 = group member), nstop, stops[nstop],
//                 f32 temperature (<= 0 greedy), f32 top_p, u32 top_k, u64 seed,
//                 then for a group member (flags & 2): group id, group size, member index,
//                 then for a thinking budget (flags & 4): budget, end-of-thinking token, nforce, force[nforce];
//                 flags & 8: exact acceptance for this request (no OPUS_TOL), no extra fields
//   in,  cancel:  'OPCX', rid
//   out: records of (rid, word): a token; 0xFFFFFFFF then the stats {u32 reused, n_prompt, n_gen, rounds,
//        accepted, flags (1 = the reply reached its token limit: max_new or the capacity), f32 prefill_ms_gpu,
//        wall_ms}; 0xFFFFFFFE then u32 length and an error message; (0, 0xFFFFFFFD) once, when ready to serve
//
// A lone request runs alone on a conversation slot (OPUS_SLOTS), speculation included. A group (the
// branches of one fanout, see the front end's start_group) waits until all its members have arrived,
// then runs on the lane slots (OPUS_LANES, each OPUS_LANE_CAP tokens): the members' longest common
// prefix is prefilled once on lane 0 (resuming from what lane 0 already holds), forked into the other
// lanes, each lane prefills its own tail, and every lane then decodes one row per joint step
// (forward_lanes, bit-identical to decoding alone). Lone requests wait while a group runs.
//
// Pool mode (OPUS_POOL=1): lone requests also run on the lanes, so independent conversations (a lead's
// subagents, several chats) decode together. A request goes to a conversation slot when a slot already
// holds its conversation or it does not fit a lane; to the lane that holds its conversation; otherwise
// to a lane while the slots are busy with a recently used conversation (OPUS_SLOT_PROTECT_S) or lanes are
// running, else to a slot. Lane requests join the running joint step between steps: the lane resumes
// from what it holds (prefill_to), its last prompt token gives the first token, then it decodes one row
// per step; each row samples with its own request's settings, so a reply equals the same request run
// alone. A slot request runs whole while the lanes wait; a group waits until no lane is running.
//
// Thinking budget: the reply starts inside a thinking block; after `budget` thinking tokens without the
// end-of-thinking token, the `force` tokens (the front end's closing text) are fed and emitted in place of
// the model's next tokens, and the answer follows.
#include "core/scheduler.h"

namespace serve {

using dankattn::proto::END;
using dankattn::proto::ERR;
using dankattn::proto::READY;
using dankattn::sched::Req;
using dankattn::sched::ReqP;

static dankattn::sched::Inbox& inbox_state() { static dankattn::sched::Inbox in; return in; }

std::mutex out_mu;
void put(uint32_t rid, uint32_t w) { dankattn::proto::put(rid, w, false); }
void flush() { fflush(stdout); }
void error(const Req& q, const std::string& msg) {
  std::lock_guard<std::mutex> g(out_mu);
  dankattn::proto::send_err(q.rid, msg);
  fprintf(stderr, "opus: rid %u refused: %s\n", q.rid, msg.c_str());
}
void end(const Req& q, uint32_t reused, uint32_t n, uint32_t rounds, uint32_t accepted, float prefill_ms, float wall_ms,
         bool length = false) {
  std::lock_guard<std::mutex> g(out_mu);
  dankattn::proto::send_end(q.rid, reused, uint32_t(q.ids.size()), n, rounds, accepted, length ? 1u : 0u, prefill_ms, wall_ms);
}
void forget(uint32_t rid) { inbox_state().forget(rid); }

// Where a reply stands in its thinking block, for the budget.
struct Think {
  bool on = false;  // still thinking under a budget
  uint32_t n = 0, budget = 0, end = 0;
  explicit Think(const Req& q) : on(q.flags & 4), budget(q.budget), end(q.think_end) {}
  // after emitting t: true when the budget is spent (the caller feeds the force tokens next)
  bool spent(uint32_t t) {
    if (!on) return false;
    if (t == end) { on = false; return false; }
    if (++n < budget) return false;
    on = false;
    return true;
  }
};

void set_sampling(opus::Model& m, const Req& q) {
  m.temp = q.temp;
  m.top_p = q.top_p;
  m.top_k = q.top_k ? std::min<uint32_t>(q.top_k, 64) : 64;
  m.sample_seed = q.seed;
  m.stop_ids = q.stops;
  m.tol_off = q.flags & 8;
}

// Bring m to a start of `prefix` without feeding: keep what it holds when that is a prefix, else resume
// from a checkpoint, else start over. Returns the tokens kept (m.pos).
uint32_t resume_to(opus::Model& m, const std::vector<uint32_t>& prefix) {
  size_t c = 0;
  while (c < m.history.size() && c < prefix.size() && m.history[c] == prefix[c]) c++;
  if (c == m.history.size() && c <= prefix.size()) {
    // m holds a prefix of `prefix`
  } else if (!m.restore_ckpt(c)) {
    m.reset();
  }
  return m.pos;
}

// Bring m to exactly `prefix` (no MTP). Returns the tokens reused.
uint32_t prefill_to(opus::Model& m, const std::vector<uint32_t>& prefix, opus::SpecStats* st) {
  const uint32_t reused = resume_to(m, prefix);
  if (m.pos < prefix.size()) m.feed(std::vector<uint32_t>(prefix.begin() + m.pos, prefix.end()), false, st);
  return reused;
}

struct Server {
  std::vector<std::unique_ptr<opus::Model>> slots, lanes;
  std::vector<uint64_t> used;
  uint64_t tick = 0;
  size_t KEEP_MIN = 2048, LOSE_MAX = 256;
  double group_wait_s = 60;
  // pool mode
  bool pool = false;
  double slot_protect_s = 600;
  size_t lane_min_out = 2048;  // a lane takes a request only with this much room left for its reply
  bool share_prefix = true;    // OPUS_SHARE_PREFIX: a new lane request may fork from another model's checkpoint
  bool lane_spec = true;       // OPUS_LANE_SPEC: joint MTP speculation on the lanes (at most MAXM/2 running)
  uint64_t spec_rounds = 0;
  std::vector<std::chrono::steady_clock::time_point> slot_at;  // last use of each slot
  std::vector<uint64_t> lane_used;  // LRU tick per lane
  struct Run {  // a lone request decoding on a lane
    ReqP q;
    uint32_t lane = 0, x = 0, n = 0, max_new = 0, reused = 0, steps = 0, forked = 0;
    float pre_ms = 0, fork_ms = 0;
    bool spec = false;  // admitted with the MTP pass: may run joint speculation rounds
    std::vector<uint32_t> acc;  // joint speculation: pairs awaiting the next accept call
    uint32_t accepted = 0, rounds = 0;
    Think th;
    std::deque<uint32_t> force;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    explicit Run(ReqP r) : q(std::move(r)), th(*q) {}
  };
  std::vector<std::unique_ptr<Run>> runs;
  uint64_t pool_steps = 0, pool_rows = 0;

  static size_t common(const std::vector<uint32_t>& h, const std::vector<uint32_t>& ids) {
    size_t c = 0;
    while (c < h.size() && c < ids.size() && h[c] == ids[c]) c++;
    return c;
  }
  // m holds a conversation this request continues (at most LOSE_MAX of its tokens dropped)
  bool extends(const opus::Model& m, const Req& q) const {
    const auto& h = m.history;
    return !h.empty() && h.size() - common(h, q.ids) <= LOSE_MAX;
  }
  bool lane_busy(uint32_t k) const {
    for (auto& r : runs) if (r->lane == k) return true;
    return false;
  }
  bool fits_lane(const Req& q) const { return q.ids.size() + 9 + lane_min_out <= lanes[0]->cap; }
  // the idle lane holding this request's conversation (longest shared prefix), or -1
  int lane_of(const Req& q) const {
    int best = -1;
    size_t bc = 0;
    for (uint32_t k = 0; k < lanes.size(); k++)
      if (!lane_busy(k) && extends(*lanes[k], q)) {
        const size_t c = common(lanes[k]->history, q.ids);
        if (best < 0 || c > bc) { best = int(k); bc = c; }
      }
    return best;
  }
  // crowded: other lone requests are already waiting, so this one would queue behind them on a slot
  bool to_lane(const Req& q, bool crowded) const {
    if (!pool || lanes.empty() || !fits_lane(q)) return false;
    for (auto& s : slots) if (extends(*s, q)) return false;
    if (lane_of(q) >= 0 || !runs.empty() || crowded) return true;
    const auto now = std::chrono::steady_clock::now();
    for (size_t s = 0; s < slots.size(); s++) {
      const bool held = slots[s]->history.size() >= KEEP_MIN &&
                        std::chrono::duration<double>(now - slot_at[s]).count() < slot_protect_s;
      if (!held) return false;  // a free slot: the request gets speculation there
    }
    return true;
  }
  int idle_lane(const Req& q) const {
    if (int k = lane_of(q); k >= 0) return k;
    int best = -1;
    for (uint32_t k = 0; k < lanes.size(); k++)
      if (!lane_busy(k) && (best < 0 || lane_used[k] < lane_used[size_t(best)])) best = int(k);
    return best;
  }
  // emit run r's next token x (or a forced one); false once the run is over
  bool emit(Run& r) {
    if (!r.force.empty()) { r.x = r.force.front(); r.force.pop_front(); }
    put(r.q->rid, r.x);
    if (r.th.spent(r.x)) r.force.assign(r.q->force.begin(), r.q->force.end());
    const bool stop = r.force.empty() && (r.x == lanes[0]->eos ||
                      std::find(r.q->stops.begin(), r.q->stops.end(), r.x) != r.q->stops.end());
    return !(stop || ++r.n >= r.max_new || r.q->cancel);
  }
  void finish_run(Run& r) {
    const float wall = std::chrono::duration<float>(std::chrono::steady_clock::now() - r.t0).count() * 1e3f;
    lane_used[r.lane] = ++tick;
    fprintf(stderr, "opus: lane %u req prompt %zu (reused %u%s) gen %u in %u joint steps, prefill %.0f ms, %.1f tok/s wall%s\n",
            r.lane, r.q->ids.size(), r.reused,
            r.forked ? (", forked " + std::to_string(r.forked) + " in " + std::to_string(int(r.fork_ms)) + " ms").c_str() : "",
            r.n, r.steps, r.pre_ms, r.n / std::max(1e-9, wall * 1e-3),
            r.q->cancel ? " (cancelled)" : "");
    end(*r.q, r.reused, r.n, r.rounds ? r.rounds : r.steps, r.accepted, r.pre_ms, wall, r.n >= r.max_new);
    forget(r.q->rid);
  }
  bool is_stop(const Run& r, uint32_t t) const {
    return t == lanes[0]->eos || std::find(r.q->stops.begin(), r.q->stops.end(), t) != r.q->stops.end();
  }
  // emit run r's forced tokens (a spent thinking budget) and feed them, with the MTP pass, after r.x;
  // the model's next token becomes r.x. false once the run is over.
  bool feed_force(Run& r) {
    std::vector<uint32_t> seq{r.x};
    bool on = true;
    {
      std::lock_guard<std::mutex> g(out_mu);
      while (!r.force.empty() && on) {
        const uint32_t f = r.force.front();
        r.force.pop_front();
        put(r.q->rid, f);
        seq.push_back(f);
        on = ++r.n < r.max_new && !r.q->cancel;
      }
      flush();
    }
    if (!on) return false;
    opus::Model& m = *lanes[r.lane];
    m.mtp_pos = m.pos;
    opus::SpecStats st;
    r.x = m.feed(seq, true, &st);
    memcpy(m.seed_h_lane[r.lane].contents, m.seed_h.contents, m.seed_h.length);
    m.spec_level = 0;
    r.acc.clear();
    std::lock_guard<std::mutex> g(out_mu);
    on = emit(r);
    flush();
    return on;
  }
  // one joint speculation round over every running lane (all admitted with the MTP pass)
  void spec_step() {
    const uint32_t NL = uint32_t(lanes.size());
    std::vector<opus::Model*> L;
    for (auto& l : lanes) L.push_back(l.get());
    std::vector<Run*> by(NL, nullptr);
    std::vector<bool> live(NL, false);
    std::vector<uint32_t> x(NL, 0);
    std::vector<std::vector<uint32_t>> acc(NL), out(NL);
    for (auto& r : runs) { by[r->lane] = r.get(); live[r->lane] = true; x[r->lane] = r->x; acc[r->lane] = r->acc; }
    // keep the round's tokens up to and including the first that ends the run or spends the budget
    std::function<uint32_t(uint32_t, const std::vector<uint32_t>&)> cut = [&](uint32_t k, const std::vector<uint32_t>& e) {
      Run& r = *by[k];
      Think th = r.th;
      uint32_t n = r.n;
      for (uint32_t j = 0; j < e.size(); j++)
        if (is_stop(r, e[j]) || th.spent(e[j]) || ++n >= r.max_new || r.q->cancel) return j + 1;
      return uint32_t(e.size());
    };
    lanes[0]->spec_lanes_round(*lanes[0], L, x, acc, out, live, UINT32_MAX, nullptr, &cut);
    spec_rounds++;
    std::vector<Run*> forcing, done;
    {
      std::lock_guard<std::mutex> g(out_mu);
      for (auto& rp : runs) {
        Run& r = *rp;
        const auto& o = out[r.lane];
        r.rounds++;
        r.steps++;
        r.accepted += uint32_t(o.size()) - 1;
        r.acc = acc[r.lane];
        bool on = true;
        for (uint32_t t : o) {  // the cut guarantees an ending token is the round's last
          r.x = t;
          put(r.q->rid, t);
          if (r.th.spent(t)) r.force.assign(r.q->force.begin(), r.q->force.end());
          if ((r.force.empty() && is_stop(r, t)) || ++r.n >= r.max_new || r.q->cancel) on = false;
        }
        if (!on) done.push_back(&r);
        else if (!r.force.empty()) forcing.push_back(&r);
      }
      flush();
    }
    for (Run* r : forcing)
      if (!feed_force(*r)) done.push_back(r);
    for (Run* r : done) {
      finish_run(*r);
      runs.erase(std::find_if(runs.begin(), runs.end(), [&](auto& p) { return p.get() == r; }));
    }
  }
  // start q on lane k: resume what the lane holds, feed the prompt's last token for the first token
  void admit(ReqP q, uint32_t k) {
    auto r = std::make_unique<Run>(q);
    r->lane = k;
    if (q->ids.empty() || !q->max_new) { error(*q, "empty prompt or max_tokens 0"); forget(q->rid); return; }
    r->max_new = std::min<uint32_t>(q->max_new, uint32_t(lanes[k]->cap - q->ids.size() - 8));
    opus::Model& m = *lanes[k];
    set_sampling(m, *q);
    opus::SpecStats st;
    const std::vector<uint32_t> head(q->ids.begin(), q->ids.end() - 1);
    // what this lane can resume by itself: an exact extension, else its newest checkpoint inside the prefix
    const size_t c_own = common(m.history, head);
    const size_t own = c_own == m.history.size() ? c_own : m.ckpt_below(c_own);
    // another lane or slot may hold a longer start of this prompt (subagents share their system prompt):
    // fork from its newest checkpoint inside the shared part
    opus::Model* donor = nullptr;
    size_t dp = 0;
    if (share_prefix) {
      auto consider = [&](opus::Model& d) {
        if (&d == &m) return;
        const size_t p = d.ckpt_below(common(d.history, head));
        if (p > dp) { dp = p; donor = &d; }
      };
      for (auto& l : lanes) consider(*l);
      for (auto& s : slots) consider(*s);
    }
    if (donor && dp >= own + 1024) {
      const auto tf = std::chrono::steady_clock::now();
      m.fork_prefix(*donor, dp);
      r->forked = uint32_t(dp);
      r->fork_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - tf).count();
    }
    r->spec = lane_spec && m.has_mtp && (q->flags & 1);
    if (r->spec) {
      // as generate() with speculation: the remaining prompt goes through the MTP pass too, whose last row
      // drafts the first token; the draft cache picks up at the resume point
      r->reused = resume_to(m, head);
      m.mtp_pos = m.pos;
      r->x = m.feed(std::vector<uint32_t>(q->ids.begin() + m.pos, q->ids.end()), true, &st);
      memcpy(m.seed_h_lane[k].contents, m.seed_h.contents, m.seed_h.length);
      m.spec_level = 0;
    } else {
      r->reused = prefill_to(m, head, &st);
      r->x = m.feed({q->ids.back()}, false, &st);
    }
    r->pre_ms = float(st.prefill_ms);
    lane_used[k] = ++tick;
    bool on;
    {
      std::lock_guard<std::mutex> g(out_mu);
      on = emit(*r);
      flush();
    }
    if (on) runs.push_back(std::move(r));
    else finish_run(*r);
  }
  // one joint step over every running lane
  void pool_step() {
    std::vector<opus::Model*> act;
    std::vector<std::vector<uint32_t>> tk;
    for (auto& r : runs) { act.push_back(lanes[r->lane].get()); tk.push_back({r->x}); }
    auto out = lanes[0]->forward_lanes(act, tk, 0);
    pool_steps++;
    pool_rows += runs.size();
    for (auto& r : runs)
      if (r->spec) {  // a plain step between speculation rounds: the draft cache skips this position
        opus::Model& m = *lanes[r->lane];
        r->acc.clear();
        m.mtp_pos = m.pos;
        m.spec_level = 0;
      }
    std::vector<size_t> done;
    {
      std::lock_guard<std::mutex> g(out_mu);
      for (size_t j = 0; j < runs.size(); j++) {
        runs[j]->x = out[j][0];
        runs[j]->steps++;
        if (!emit(*runs[j])) done.push_back(j);
      }
      flush();
    }
    for (size_t i = done.size(); i-- > 0;) {
      finish_run(*runs[done[i]]);
      runs.erase(runs.begin() + long(done[i]));
    }
  }

  void lone(Req& q) {
    const uint32_t NS = uint32_t(slots.size());
    // slot choice: a slot this request extends (at most LOSE_MAX of its tokens dropped) with
    // the longest shared prefix; otherwise the slot where prefill now plus the loss of the
    // most recent conversation is smallest
    uint32_t si = 0;
    {
      size_t best_ext = 0, best_cost = ~size_t(0);
      bool have_ext = false;
      uint64_t mru = 0;
      for (uint32_t s = 0; s < NS; s++) mru = std::max(mru, used[s]);
      for (uint32_t s = 0; s < NS; s++) {
        if (q.ids.size() + q.max_new + 8 > slots[s]->cap) continue;  // does not fit this slot
        const auto& h = slots[s]->history;
        size_t c = 0;
        while (c < h.size() && c < q.ids.size() && h[c] == q.ids[c]) c++;
        const size_t lost = h.size() - c;
        if (lost <= LOSE_MAX && !h.empty()) {
          if (!have_ext || c > best_ext) { have_ext = true; best_ext = c; si = s; }
        } else if (!have_ext) {
          // losing under KEEP_MIN tokens of a conversation is cheaper than a slot switch (~50 ms)
          const size_t pen = lost < KEEP_MIN ? 0 : (used[s] == mru && tick ? lost : (used[s] ? lost / 4 : 0));
          const size_t cost = (q.ids.size() - c) + pen + (used[s] == mru ? 0 : 64);
          if (cost < best_cost) { best_cost = cost; si = s; }
        }
      }
    }
    // No slot holds prompt + max_new: slot 0 is the largest. The reply is cut at its capacity
    // (finish "length"); a prompt that does not fit at all is refused, not fatal.
    uint32_t max_new = q.max_new;
    if (q.ids.size() + max_new + 8 > slots[si]->cap) {
      si = 0;
      if (q.ids.size() + 9 > slots[0]->cap) {
        error(q, "prompt of " + std::to_string(q.ids.size()) + " tokens exceeds the context capacity of " +
                     std::to_string(slots[0]->cap));
        return;
      }
      max_new = uint32_t(slots[0]->cap - q.ids.size() - 8);
    }
    if (q.ids.empty() || !max_new) { error(q, "empty prompt or max_tokens 0"); return; }
    used[si] = ++tick;
    if (si < slot_at.size()) slot_at[si] = std::chrono::steady_clock::now();
    opus::Model& m = *slots[si];
    set_sampling(m, q);
    opus::SpecStats st;
    const auto t0 = std::chrono::steady_clock::now();
    Think th(q);
    bool spent = false;
    std::vector<uint32_t> said;  // emitted tokens, for the second phase of a spent budget
    auto say = [&](uint32_t t) {
      std::lock_guard<std::mutex> g(out_mu);
      put(q.rid, t);
      flush();
    };
    uint32_t n = m.generate(q.ids, max_new, q.flags & 1, [&](uint32_t t) {
      say(t);
      said.push_back(t);
      if (th.spent(t)) { spent = true; return false; }
      return !q.cancel && std::find(q.stops.begin(), q.stops.end(), t) == q.stops.end();
    }, &st);
    // budget spent: emit the closing tokens and continue the same conversation with them (it extends
    // what the slot holds, so nothing is prefilled again). generate does not count the token its
    // callback refused, so count what was emitted.
    if (spent) n = uint32_t(said.size());
    if (spent && !q.cancel && n + q.force.size() < max_new) {
      std::vector<uint32_t> p2 = q.ids;
      p2.insert(p2.end(), said.begin(), said.end());
      p2.insert(p2.end(), q.force.begin(), q.force.end());
      for (uint32_t t : q.force) say(t);
      n += uint32_t(q.force.size());
      opus::SpecStats st2;
      n += m.generate(p2, max_new - n, q.flags & 1, [&](uint32_t t) {
        say(t);
        return !q.cancel && std::find(q.stops.begin(), q.stops.end(), t) == q.stops.end();
      }, &st2);
      st.rounds += st2.rounds;
      st.accepted += st2.accepted;
      st.prefill_ms += st2.prefill_ms;
      st.verify_ms += st2.verify_ms;
      st.draft_ms += st2.draft_ms;
      st.wall_ms += st2.wall_ms;
    }
    const float wall = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count() * 1e3f;
    fprintf(stderr, "opus: slot %u req prompt %zu (reused %u) gen %u in %u rounds (%.2f tok/round, %.2f rows/round), lookup %u/%u accepted, streak %u chunks (%u rolled back, %u tokens), prefill %.0f ms, decode %.1f tok/s GPU, %.1f tok/s wall%s\n",
            si, q.ids.size(), st.reused, n, st.rounds, double(n) / std::max(1u, st.rounds), double(st.rows) / std::max(1u, st.rounds),
            st.lk_accepted, st.lk_tested, st.streak_chunks, st.streak_fail, st.streak_tokens, st.prefill_ms,
            n / std::max(1e-9, (st.verify_ms + st.draft_ms) * 1e-3), n / std::max(1e-9, st.wall_ms * 1e-3),
            q.cancel ? " (cancelled)" : "");
    if (st.tol_accepted) fprintf(stderr, "opus: tolerant acceptance took %u of %u tokens\n", st.tol_accepted, n);
    end(q, st.reused, n, st.rounds, st.accepted, float(st.prefill_ms), wall, n >= max_new);
  }

  void group(std::vector<ReqP> g) {
    std::sort(g.begin(), g.end(), [](const ReqP& a, const ReqP& b) { return a->grp2 < b->grp2; });
    const uint32_t N = uint32_t(g.size());
    auto refuse = [&](const std::string& why) { for (auto& q : g) error(*q, "group of " + std::to_string(N) + " refused: " + why); };
    if (N > lanes.size()) { refuse("the server has " + std::to_string(lanes.size()) + " lanes (OPUS_LANES)"); return; }
    const uint32_t cap = lanes[0]->cap;
    std::vector<uint32_t> max_new(N);
    for (uint32_t k = 0; k < N; k++) {
      const Req& q = *g[k];
      if (q.ids.empty() || !q.max_new) { refuse("member " + std::to_string(k) + " has an empty prompt or max_tokens 0"); return; }
      if (q.ids.size() + 9 > cap) {
        refuse("member " + std::to_string(k) + " prompt of " + std::to_string(q.ids.size()) +
               " tokens exceeds the lane capacity of " + std::to_string(cap) + " (OPUS_LANE_CAP)");
        return;
      }
      max_new[k] = std::min<uint32_t>(q.max_new, uint32_t(cap - q.ids.size() - 8));
    }
    // longest common prefix, leaving every member at least one token of its own
    size_t P = g[0]->ids.size() - 1;
    for (auto& q : g) {
      P = std::min(P, q->ids.size() - 1);
      size_t c = 0;
      while (c < P && q->ids[c] == g[0]->ids[c]) c++;
      P = c;
    }
    const auto t0 = std::chrono::steady_clock::now();
    opus::Model& drv = *lanes[0];
    set_sampling(drv, *g[0]);  // one setting for every row of a joint step
    opus::SpecStats pst;
    const uint32_t reused = prefill_to(drv, std::vector<uint32_t>(g[0]->ids.begin(), g[0]->ids.begin() + P), &pst);
    const double ms_prefix = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (uint32_t k = 1; k < N; k++) lanes[k]->fork_from(drv);
    const double ms_fork = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() - ms_prefix;
    // each lane's own tail; its last row gives the first token
    std::vector<uint32_t> x(N), n(N, 0);
    std::vector<bool> on(N, true);
    std::vector<Think> th;
    std::vector<std::deque<uint32_t>> force(N);  // tokens fed and emitted in place of the model's
    for (auto& q : g) th.emplace_back(*q);
    std::vector<float> pre_ms(N, 0);
    std::vector<opus::Model*> L;
    for (uint32_t k = 0; k < N; k++) {
      opus::Model& m = *lanes[k];
      if (k) set_sampling(m, *g[k]);
      opus::SpecStats st;
      x[k] = m.feed(std::vector<uint32_t>(g[k]->ids.begin() + P, g[k]->ids.end()), false, &st);
      pre_ms[k] = float(st.prefill_ms + (k ? 0 : pst.prefill_ms));
      L.push_back(&m);
    }
    const auto td = std::chrono::steady_clock::now();
    const double ms_tails = std::chrono::duration<double, std::milli>(td - t0).count() - ms_prefix - ms_fork;
    auto stop_tok = [&](uint32_t k, uint32_t t) {
      return t == drv.eos || std::find(g[k]->stops.begin(), g[k]->stops.end(), t) != g[k]->stops.end();
    };
    auto emit = [&](uint32_t k) {  // x[k] is lane k's next token
      if (!force[k].empty()) { x[k] = force[k].front(); force[k].pop_front(); }
      put(g[k]->rid, x[k]);
      if (th[k].spent(x[k])) force[k].assign(g[k]->force.begin(), g[k]->force.end());
      if ((force[k].empty() && stop_tok(k, x[k])) || ++n[k] >= max_new[k] || g[k]->cancel) on[k] = false;
    };
    uint32_t steps = 0;
    size_t total = 0;
    {
      std::lock_guard<std::mutex> lk(out_mu);
      for (uint32_t k = 0; k < N; k++) emit(k);
      flush();
    }
    auto finish_lane = [&](uint32_t k) {
      const float wall = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count() * 1e3f;
      end(*g[k], k ? uint32_t(P) : reused, n[k], steps, 0, pre_ms[k], wall, n[k] >= max_new[k]);
      total += n[k];
    };
    for (uint32_t k = 0; k < N; k++) if (!on[k]) finish_lane(k);
    for (;;) {
      std::vector<opus::Model*> act;
      std::vector<uint32_t> idx;
      std::vector<std::vector<uint32_t>> tk;
      for (uint32_t k = 0; k < N; k++)
        if (on[k]) { act.push_back(L[k]); idx.push_back(k); tk.push_back({x[k]}); }
      if (act.empty()) break;
      auto out = drv.forward_lanes(act, tk, 0);
      steps++;
      std::vector<uint32_t> done;
      {
        std::lock_guard<std::mutex> lk(out_mu);
        for (size_t j = 0; j < idx.size(); j++) {
          x[idx[j]] = out[j][0];
          emit(idx[j]);
          if (!on[idx[j]]) done.push_back(idx[j]);
        }
        flush();
      }
      for (uint32_t k : done) finish_lane(k);
    }
    const double ms_dec = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td).count();
    fprintf(stderr, "opus: group %u x%u: shared prefix %zu (reused %u), prefix %.0f ms, fork %.0f ms, tails %.0f ms; "
                    "%zu tokens in %u joint steps, %.1f ms/step, decode %.1f tok/s aggregate, %.1f tok/s overall\n",
            g[0]->grp0, N, P, reused, ms_prefix, ms_fork, ms_tails, total, steps, ms_dec / std::max(1u, steps),
            total / std::max(1e-9, ms_dec * 1e-3),
            total / std::max(1e-9, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()));
  }

  void run_pool() {
    auto& in = inbox_state();
    std::deque<ReqP> slot_q, lane_q;
    std::map<uint32_t, std::vector<ReqP>> pending;  // group id -> members so far
    auto drop = [](const ReqP& q) { end(*q, 0, 0, 0, 0, 0, 0); forget(q->rid); };
    for (;;) {
      std::vector<ReqP> fresh;
      {
        std::unique_lock<std::mutex> lk(in.mu);
        const bool busy = !runs.empty() || !slot_q.empty() || !lane_q.empty();
        if (in.queue.empty() && !busy && !in.closed) in.cv.wait_for(lk, std::chrono::seconds(pending.empty() ? 3600 : 1));
        if (in.closed && in.queue.empty() && !busy) break;
        for (auto& q : in.queue) {
          if (q->flags & 2) pending[q->grp0].push_back(q);
          else fresh.push_back(q);
        }
        in.queue.clear();
      }
      // several new requests at once (a lead's subagents): none takes the slot, which would hold the lanes
      // for its whole reply
      for (auto& q : fresh) {
        if (q->cancel) { drop(q); continue; }
        (to_lane(*q, !slot_q.empty() || !lane_q.empty() || fresh.size() > 1) ? lane_q : slot_q).push_back(q);
      }
      // a complete group runs when no lane is busy: it takes lanes 0..N-1 from scratch
      if (runs.empty() && lane_q.empty()) {
        bool ran = false;
        for (auto it = pending.begin(); it != pending.end(); ++it) {
          auto& m = it->second;
          if (m.size() < m[0]->grp1) continue;
          auto grp = std::move(m);
          pending.erase(it);
          group(grp);
          for (auto& q : grp) forget(q->rid);
          ran = true;
          break;
        }
        if (ran) continue;
      }
      // waiting lane requests join on idle lanes, between steps
      for (auto it = lane_q.begin(); it != lane_q.end();) {
        if ((*it)->cancel) { drop(*it); it = lane_q.erase(it); continue; }
        const int k = idle_lane(**it);
        if (k < 0) break;
        auto q = *it;
        it = lane_q.erase(it);
        admit(q, uint32_t(k));
      }
      // a slot request runs whole; the lanes keep their state and wait
      if (!slot_q.empty()) {
        auto q = slot_q.front();
        slot_q.pop_front();
        if (q->cancel) drop(q);
        else { lone(*q); forget(q->rid); }
        continue;
      }
      if (!runs.empty()) {
        const bool all_spec = std::all_of(runs.begin(), runs.end(), [](auto& r) { return r->spec && r->force.empty(); });
        if (lane_spec && all_spec && runs.size() * 2 <= opus::MAXM) spec_step();
        else pool_step();
      }
      const auto now = std::chrono::steady_clock::now();
      for (auto it = pending.begin(); it != pending.end();) {
        auto& m = it->second;
        const bool stale = std::chrono::duration<double>(now - m[0]->arrived).count() > group_wait_s;
        const bool gone = std::all_of(m.begin(), m.end(), [](const ReqP& q) { return q->cancel.load(); });
        if (stale || gone) {
          for (auto& q : m) {
            error(*q, "group " + std::to_string(q->grp0) + ": " + std::to_string(m.size()) + " of " +
                          std::to_string(q->grp1) + " members arrived");
            forget(q->rid);
          }
          it = pending.erase(it);
        } else ++it;
      }
    }
  }

  void run() {
    auto& in = inbox_state();
    in.start_reader();
    if (pool) { run_pool(); in.join(); return; }
    std::deque<ReqP> lone_q;
    std::map<uint32_t, std::vector<ReqP>> pending;  // group id -> members so far
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(in.mu);
        if (in.queue.empty() && lone_q.empty() && !in.closed)
          in.cv.wait_for(lk, std::chrono::seconds(pending.empty() ? 3600 : 1));
        if (in.closed && in.queue.empty() && lone_q.empty()) break;
        for (auto& q : in.queue) {
          if (q->flags & 2) pending[q->grp0].push_back(q);
          else lone_q.push_back(q);
        }
        in.queue.clear();
      }
      // a complete group first: its members are all waiting; then one lone request
      bool ran = false;
      for (auto it = pending.begin(); it != pending.end(); ++it) {
        auto& m = it->second;
        if (m.size() < m[0]->grp1) continue;
        auto grp = std::move(m);
        pending.erase(it);
        group(grp);
        for (auto& q : grp) forget(q->rid);
        ran = true;
        break;
      }
      if (!ran && !lone_q.empty()) {
        auto q = lone_q.front();
        lone_q.pop_front();
        if (!q->cancel) lone(*q);
        else end(*q, 0, 0, 0, 0, 0, 0);
        forget(q->rid);
      }
      // a group that never completes is refused rather than holding lanes and its members forever
      const auto now = std::chrono::steady_clock::now();
      for (auto it = pending.begin(); it != pending.end();) {
        auto& m = it->second;
        const bool stale = std::chrono::duration<double>(now - m[0]->arrived).count() > group_wait_s;
        const bool gone = std::all_of(m.begin(), m.end(), [](const ReqP& q) { return q->cancel.load(); });
        if (stale || gone) {
          for (auto& q : m) {
            error(*q, "group " + std::to_string(q->grp0) + ": " + std::to_string(m.size()) + " of " +
                          std::to_string(q->grp1) + " members arrived");
            forget(q->rid);
          }
          it = pending.erase(it);
        } else ++it;
      }
    }
    in.join();
  }
};

int main(int argc, char** argv) {
  uint32_t cap = argc > 3 ? atoi(argv[3]) : 131072;
  auto env = [](const char* k, long d) { return getenv(k) ? atol(getenv(k)) : d; };
  Server S;
  // Conversation slots (OPUS_SLOTS, default 1): each keeps one conversation's caches, so
  // requests that alternate between conversations do not re-prefill each other's context.
  // All slots share the weights and the per-step scratch; a slot costs only its own state.
  const uint32_t NS = uint32_t(std::max(1L, env("OPUS_SLOTS", 1)));
  S.KEEP_MIN = size_t(env("OPUS_SLOT_KEEP", 2048));
  S.LOSE_MAX = size_t(env("OPUS_SLOT_LOSE", 256));
  S.group_wait_s = double(env("OPUS_GROUP_WAIT_S", 60));
  S.slot_protect_s = double(env("OPUS_SLOT_PROTECT_S", 600));
  S.lane_min_out = size_t(env("OPUS_LANE_MIN_OUT", 2048));
  S.share_prefix = env("OPUS_SHARE_PREFIX", 1) != 0;
  S.lane_spec = env("OPUS_LANE_SPEC", 1) != 0;
  S.slots.emplace_back(new opus::Model(argv[1], cap));
  // extra slots hold shorter conversations by default (OPUS_SLOT_CAP tokens, default cap / 2)
  const uint32_t cap2 = getenv("OPUS_SLOT_CAP") ? std::min<uint32_t>(cap, atoi(getenv("OPUS_SLOT_CAP"))) : cap / 2;
  for (uint32_t i = 1; i < NS; i++) S.slots.emplace_back(new opus::Model(argv[1], cap2, S.slots[0].get()));
  // several slots share the RAM: fewer checkpoints each (branch points stay pinned regardless)
  if (NS > 1) for (auto& sl : S.slots) { sl->ckpt_keep = 4; sl->ckpt_far = 16384; }
  // checkpoint retention (~120 MB each): OPUS_CKPT_KEEP newest plus one every OPUS_CKPT_FAR tokens per slot,
  // OPUS_LANE_CKPT_KEEP / OPUS_LANE_CKPT_FAR per lane (far must be a multiple of the 1,024 spacing)
  for (auto& sl : S.slots) {
    sl->ckpt_keep = size_t(env("OPUS_CKPT_KEEP", long(sl->ckpt_keep)));
    sl->ckpt_far = size_t(env("OPUS_CKPT_FAR", long(sl->ckpt_far)));
  }
  // lane slots for groups (OPUS_LANES, at most NSLOT, all OPUS_LANE_CAP tokens: a joint step needs equal capacities)
  const uint32_t NL = uint32_t(std::min<long>(env("OPUS_LANES", 0), opus::NSLOT));
  const uint32_t lcap = uint32_t(std::min<long>(env("OPUS_LANE_CAP", 16384), cap));
  for (uint32_t i = 0; i < NL; i++) {
    S.lanes.emplace_back(new opus::Model(argv[1], lcap, S.slots[0].get()));
    S.lanes.back()->ckpt_keep = size_t(env("OPUS_LANE_CKPT_KEEP", 2));
    S.lanes.back()->ckpt_far = size_t(env("OPUS_LANE_CKPT_FAR", 16384));
  }
  S.used.assign(NS, 0);
  S.slot_at.assign(NS, std::chrono::steady_clock::now() - std::chrono::hours(24));
  S.lane_used.assign(NL, 0);
  S.pool = NL > 0 && env("OPUS_POOL", 0) != 0;
  {
    size_t tot = S.slots[0]->C.allocated;
    for (size_t i = 1; i < NS; i++) tot += S.slots[i]->C.allocated - S.slots[0]->C.allocated;
    for (auto& l : S.lanes) tot += l->C.allocated - S.slots[0]->C.allocated;
    fprintf(stderr, "opus: state buffers %.1f GiB in %u slot(s) and %u lane(s) of %u tokens\n", tot / 1073741824.0, NS, NL, lcap);
  }
  if (env("OPUS_WARM", 1) != 0) {
    // first use of each pipeline and code path costs ~0.3 s: pay it now, not on the first request
    const auto tw = std::chrono::steady_clock::now();
    std::vector<uint32_t> w(2100);
    for (size_t i = 0; i < w.size(); i++) w[i] = 1000 + uint32_t(i * 7919 % 50000);
    for (size_t n : {size_t(40), w.size()}) {
      S.slots[0]->generate(std::vector<uint32_t>(w.begin(), w.begin() + n), 6, true, [](uint32_t) { return true; }, nullptr);
      S.slots[0]->reset();
    }
    if (NL > 1) {
      std::vector<opus::Model*> L;
      std::vector<std::vector<uint32_t>> t;
      for (auto& l : S.lanes) { l->feed({w.begin(), w.begin() + 40}, false); L.push_back(l.get()); t.push_back({w[40]}); }
      S.lanes[0]->forward_lanes(L, t, 0);
      for (auto& l : S.lanes) l->reset();
    }
    fprintf(stderr, "opus: warm-up %.0f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw).count());
  }
  fprintf(stderr, "opus: serving (cap %u tokens, mtp %s, %u conversation slots, %u lanes%s)\n", cap,
          S.slots[0]->has_mtp ? "on" : "off", NS, NL, S.pool ? ", pool" : "");
  {
    std::lock_guard<std::mutex> g(out_mu);
    put(0, READY);
    flush();
  }
  S.run();
  return 0;
}

}  // namespace serve
