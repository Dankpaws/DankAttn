// DankAttn Inkling2 serve: the DankAttn binary protocol (same framing as engine/hy4/serve.inc, spoken by
// server/engine_frontend.py) with continuous multi-lane batching.
//   in,  request: 'OPUS', rid, n, ids[n], max_new, flags (1 = speculate, 4 = thinking budget, 16 = min_p), nstop, stops[nstop],
//                 f32 temperature (<= 0 greedy), f32 top_p, u32 top_k, u64 seed, then for flags & 4: budget,
//                 end-of-thinking token, nforce, force[nforce]; for flags & 16: f32 min_p.
//                 flags & 2 (fanout groups) and & 32 (images) are refused.
//   in,  cancel:  'OPCX', rid
//   out: (rid, token) records; (rid, 0xFFFFFFFF) then u32 {reused, n_prompt, n_gen, rounds, accepted, length}
//        and f32 {prefill_ms_gpu, wall_ms}; (rid, 0xFFFFFFFE), u32 length, message; (0, 0xFFFFFFFD) when ready.
// Every lane is an independent sequence (Model::add_lane). Each loop iteration prefills one chunk of one
// waiting request, then runs one speculation round over every decoding lane: each lane drafts (3-gram lookup,
// else DSpark with the p_max cut), one step_lanes() verifies all lanes' rows (MAXM rows split evenly), and
// each lane accepts, rolls back and commits its DSpark context on its own. One decoding lane gets all MAXM rows.
// Prompt reuse: a lane resumes a prompt that extends exactly what it has processed, or rewinds to the end of
// its last prefill (sconv states are checkpointed there; the local KV rings must still hold the window before
// it, so at most MP + INK_RING_SLACK tokens may have followed). Clients that drop reasoning from earlier turns
// land on the checkpoint. Anything else re-prefills from zero.
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

namespace iserve {

constexpr uint32_t END = 0xFFFFFFFFu, ERR = 0xFFFFFFFEu, READY = 0xFFFFFFFDu;

struct Req {
  uint32_t rid = 0, max_new = 0, flags = 0;
  std::vector<uint32_t> ids, stops, force;
  uint32_t budget = 0, think_end = 0;
  float temp = 0, top_p = 1, min_p = -1.0f;
  uint32_t top_k = 0;
  uint64_t seed = 0;
  std::atomic<bool> cancel{false};
};
using ReqP = std::shared_ptr<Req>;

std::mutex mu;
std::condition_variable cv;
std::deque<ReqP> inbox;
std::map<uint32_t, ReqP> live;
bool closed = false;

void put(uint32_t rid, uint32_t w) { uint32_t r[2] = {rid, w}; fwrite(r, 4, 2, stdout); }
void error(uint32_t rid, const std::string& msg) {
  put(rid, ERR);
  const uint32_t n = uint32_t(msg.size());
  fwrite(&n, 4, 1, stdout);
  fwrite(msg.data(), 1, n, stdout);
  fflush(stdout);
  fprintf(stderr, "inkling2: rid %u refused: %s\n", rid, msg.c_str());
}

void reader() {
  auto rd = [](void* p, size_t n) { return fread(p, 1, n, stdin) == n; };
  for (;;) {
    uint32_t magic;
    if (!rd(&magic, 4)) break;
    if (magic == 0x5843504F) {  // 'OPCX'
      uint32_t rid;
      if (!rd(&rid, 4)) break;
      std::lock_guard<std::mutex> g(mu);
      if (auto it = live.find(rid); it != live.end()) it->second->cancel = true;
      continue;
    }
    if (magic != 0x5355504F) break;
    auto q = std::make_shared<Req>();
    uint32_t hdr[2], tail[3];
    if (!rd(hdr, 8)) break;
    q->rid = hdr[0];
    q->ids.resize(hdr[1]);
    if (!rd(q->ids.data(), q->ids.size() * 4) || !rd(tail, 12)) break;
    q->max_new = tail[0];
    q->flags = tail[1];
    q->stops.resize(tail[2]);
    if (!rd(q->stops.data(), q->stops.size() * 4)) break;
    struct { float temp, top_p; uint32_t top_k; uint64_t seed; } __attribute__((packed)) sp;
    if (!rd(&sp, sizeof sp)) break;
    q->temp = sp.temp; q->top_p = sp.top_p; q->top_k = sp.top_k; q->seed = sp.seed;
    if (q->flags & 2) { uint32_t g3[3]; if (!rd(g3, 12)) break; }
    if (q->flags & 4) {
      uint32_t t3[3];
      if (!rd(t3, 12)) break;
      q->budget = t3[0]; q->think_end = t3[1];
      q->force.resize(t3[2]);
      if (!rd(q->force.data(), q->force.size() * 4)) break;
    }
    if (q->flags & 16) {
      if (!rd(&q->min_p, 4)) break;
    }
    if (q->flags & 32) {  // images: consume, then refuse at admission
      uint32_t n_imgs = 0;
      if (!rd(&n_imgs, 4)) break;
      bool ok = true;
      for (uint32_t k = 0; k < n_imgs && ok; k++) {
        uint32_t hdr4[4];
        uint64_t ih = 0;
        if (!rd(hdr4, 16) || !rd(&ih, 8)) { ok = false; break; }
        std::vector<uint16_t> px(size_t(hdr4[0]) * hdr4[1] * hdr4[2] * 768);
        if (!px.empty() && !rd(px.data(), px.size() * 2)) ok = false;
      }
      if (!ok) break;
    }
    std::lock_guard<std::mutex> g(mu);
    live[q->rid] = q;
    inbox.push_back(q);
    cv.notify_one();
  }
  std::lock_guard<std::mutex> g(mu);
  closed = true;
  cv.notify_one();
}

struct Active {
  ReqP q;
  int lane = -1;
  std::chrono::steady_clock::time_point t0;
  std::vector<uint32_t> rest;  // prompt tokens still to prefill
  size_t pref_off = 0;
  uint32_t reused = 0;
  double pms = 0.0;
  uint32_t b = 0;  // next token to emit (sampled, not yet fed)
  bool prefilling = true, spec = false, thinking = false, length = false;
  uint32_t nthink = 0, n = 0, rounds = 0, accepted = 0;
  std::unique_ptr<opus_lookup::Lookup> lk;
  bool is_stop(const ink2::Model& m, uint32_t t) const {
    return t == m.eos || std::find(q->stops.begin(), q->stops.end(), t) != q->stops.end();
  }
};

// Tokens each lane has processed (its KV / sconv state covers exactly these, pos == size()).
static std::vector<std::vector<uint32_t>>& lane_hist() {
  static std::vector<std::vector<uint32_t>> h;
  return h;
}
// Per lane: sconv states right after its last prefill, at position pos.
struct Ckpt {
  uint32_t pos = 0;
  std::vector<uint8_t> conv;
};
static std::vector<Ckpt>& lane_ckpt() {
  static std::vector<Ckpt> c;
  return c;
}

static bool emit_tok(ink2::Model& m, Active& a, uint32_t t) {
  put(a.q->rid, t);
  a.n++;
  if (a.thinking) { if (t == a.q->think_end) a.thinking = false; else a.nthink++; }
  if (a.is_stop(m, t)) return true;
  if (a.n >= a.q->max_new || m.lane_pos(a.lane) + ink2::MAXM + 1 >= m.cap) { a.length = true; return true; }
  return a.q->cancel.load();
}

static void finish_active(Active& a, std::vector<bool>& busy) {
  const float wall = std::chrono::duration<float>(std::chrono::steady_clock::now() - a.t0).count() * 1e3f;
  put(a.q->rid, END);
  uint32_t s6[6] = {a.reused, uint32_t(a.q->ids.size()), a.n, a.rounds, a.accepted, a.length ? 1u : 0u};
  float f2[2] = {float(a.pms), wall};
  fwrite(s6, 4, 6, stdout);
  fwrite(f2, 4, 2, stdout);
  fflush(stdout);
  char sb[160] = "";
  if (a.spec && a.rounds)
    snprintf(sb, sizeof sb, " spec %u rounds, %.2f tok/round", a.rounds, double(a.accepted) / a.rounds);
  fprintf(stderr, "inkling2: rid %u lane %d prompt %zu (reused %u, prefill %.0f ms) gen %u in %.0f ms%s\n",
          a.q->rid, a.lane, a.q->ids.size(), a.reused, a.pms, a.n, wall, sb);
  if (a.lane >= 0) busy[a.lane] = false;
  std::lock_guard<std::mutex> g(mu);
  live.erase(a.q->rid);
}

// Longest free-lane reuse: a lane whose processed tokens are a strict prefix of ids. Ties and no reuse prefer
// the lane with the least to lose (shortest history), so a long conversation stays resumable.
static int pick_lane(const ink2::Model& m, const std::vector<uint32_t>& ids, const std::vector<bool>& busy,
                     uint32_t* reuse, bool* rewind) {
  int best = -1;
  uint32_t bc = 0;
  size_t bl = 0;
  bool brw = false;
  auto& H = lane_hist();
  for (int i = 0; i < int(busy.size()); i++) {
    if (busy[i]) continue;
    const auto& h = H[i];
    const auto& ck = lane_ckpt()[i];
    uint32_t c = 0;
    bool rw = false;
    if (!h.empty() && h.size() < ids.size() && std::equal(h.begin(), h.end(), ids.begin())) {
      c = uint32_t(h.size());
    } else if (ck.pos > 0 && ck.pos < ids.size() && ck.pos <= h.size() && h.size() - ck.pos <= m.rewind_slack() &&
               std::equal(h.begin(), h.begin() + ck.pos, ids.begin())) {
      c = ck.pos;
      rw = true;
    }
    if (best < 0 || c > bc || (c == bc && h.size() < bl)) { best = i; bc = c; bl = h.size(); brw = rw; }
  }
  *reuse = bc;
  *rewind = brw;
  return best;
}

static void admit_requests(ink2::Model& m, std::vector<Active>& active, std::vector<bool>& busy) {
  for (;;) {
    ReqP q;
    {
      std::lock_guard<std::mutex> g(mu);
      if (inbox.empty()) return;
      if (std::find(busy.begin(), busy.end(), false) == busy.end()) return;
      q = inbox.front();
      inbox.pop_front();
    }
    std::string why;
    if (q->cancel.load()) why = "";
    else if (q->flags & 2) why = "inkling2: fanout groups (flags & 2) are not supported";
    else if (q->flags & 32) why = "inkling2: image input is not supported by this engine";
    else if (q->ids.empty()) why = "empty prompt";
    else if (q->ids.size() + ink2::MAXM + 1 >= m.cap)
      why = "prompt of " + std::to_string(q->ids.size()) + " tokens exceeds the context capacity of " + std::to_string(m.cap);
    if (q->cancel.load() || !why.empty()) {
      if (!why.empty()) error(q->rid, why);
      std::lock_guard<std::mutex> g(mu);
      live.erase(q->rid);
      continue;
    }
    uint32_t reuse = 0;
    bool rw = false;
    const int ln = pick_lane(m, q->ids, busy, &reuse, &rw);
    m.use_lane(ln);
    auto& h = lane_hist()[ln];
    if (reuse == 0) {
      m.reset();
      h.clear();
    } else if (rw) {
      m.rewind(reuse, lane_ckpt()[ln].conv);
      h.resize(reuse);
    }
    ink2::Model::Samp sp;
    sp.temp = q->temp > 0 ? q->temp : 0.0f;
    sp.top_p = (q->top_p > 0.0f && q->top_p <= 1.0f) ? q->top_p : 1.0f;
    sp.top_k = q->top_k;
    sp.min_p = q->min_p >= 0.0f ? q->min_p : 0.0f;
    sp.seed = q->seed;
    m.samp = sp;
    Active a;
    a.q = q;
    a.lane = ln;
    a.t0 = std::chrono::steady_clock::now();
    a.reused = reuse;
    a.rest.assign(q->ids.begin() + reuse, q->ids.end());
    a.spec = (q->flags & 1) && m.has_mtp;
    a.thinking = (q->flags & 4) != 0;
    a.lk = std::make_unique<opus_lookup::Lookup>();
    busy[ln] = true;
    active.push_back(std::move(a));
  }
}

// Feed toks on lane a (prefill chunks or a forced span), committing DSpark context right away
// (b_taps holds this step's rows only until the next step). Returns the token after the last one fed.
static uint32_t feed(ink2::Model& m, Active& a, const std::vector<uint32_t>& toks, double* gpu) {
  m.use_lane(a.lane);
  uint32_t b = 0;
  for (size_t i = 0; i < toks.size(); i += ink2::MP) {
    std::vector<uint32_t> c(toks.begin() + long(i), toks.begin() + long(std::min(toks.size(), i + ink2::MP)));
    double g = 0;
    b = m.step(c, &g).back();
    if (gpu) *gpu += g;
    if (m.has_mtp) m.mtp_commit(uint32_t(c.size()));
    auto& h = lane_hist()[a.lane];
    h.insert(h.end(), c.begin(), c.end());
  }
  return b;
}

int main(ink2::Model& m) {
  const uint32_t n_lanes = std::max<uint32_t>(1u, std::min<uint32_t>(ink2::MAXM,
      getenv("INK_LANES") ? uint32_t(atoi(getenv("INK_LANES"))) : 4u));
  while (m.lanes_.size() < n_lanes) m.add_lane();
  lane_hist().assign(n_lanes, {});
  lane_ckpt().assign(n_lanes, {});
  const uint32_t draft_k = getenv("INK_DRAFT_K") ? uint32_t(atoi(getenv("INK_DRAFT_K"))) : 7u;
  const float draft_cut = getenv("INK_DRAFT_CUT") ? float(atof(getenv("INK_DRAFT_CUT"))) : 0.50f;
  const bool use_lookup = getenv("INK_LOOKUP") ? (atoi(getenv("INK_LOOKUP")) != 0) : true;
  const uint32_t pf_dec_chunk = getenv("INK_PF_DEC_CHUNK") ? uint32_t(atoi(getenv("INK_PF_DEC_CHUNK"))) : 256u;
  fprintf(stderr, "inkling2: serving %u lanes x %u tokens (rewind slack %u), GPU state %.2f GB, DSpark %s\n", n_lanes,
          m.cap, m.rewind_slack(), m.C.allocated / 1e9, m.has_mtp ? "on" : "off");
  std::thread rt(reader);
  put(0, READY);
  fflush(stdout);

  std::vector<Active> active;
  std::vector<bool> busy(n_lanes, false);

  for (;;) {
    admit_requests(m, active, busy);
    if (active.empty()) {
      std::unique_lock<std::mutex> g(mu);
      cv.wait(g, [&] { return closed || !inbox.empty(); });
      if (inbox.empty() && closed) break;
      continue;
    }
    @autoreleasepool {
      for (size_t i = 0; i < active.size();) {
        if (active[i].q->cancel.load()) {
          finish_active(active[i], busy);
          active.erase(active.begin() + long(i));
        } else i++;
      }
      if (active.empty()) continue;

      // 1) One prefill chunk for the first prefilling request (smaller while other lanes decode, to bound ITL)
      bool any_decoding = false;
      for (const auto& a : active) if (!a.prefilling) { any_decoding = true; break; }
      const uint32_t chunk = any_decoding ? std::max<uint32_t>(ink2::MAXM + 1u, std::min(ink2::MP, pf_dec_chunk)) : ink2::MP;
      for (auto& a : active) {
        if (!a.prefilling) continue;
        const size_t end = std::min(a.rest.size(), a.pref_off + chunk);
        std::vector<uint32_t> c(a.rest.begin() + long(a.pref_off), a.rest.begin() + long(end));
        a.b = feed(m, a, c, &a.pms);
        a.pref_off = end;
        if (a.pref_off >= a.rest.size()) {
          a.prefilling = false;
          a.rest.clear();
          auto& ck = lane_ckpt()[a.lane];
          m.use_lane(a.lane);
          ck.pos = m.pos;
          m.save_conv(ck.conv);
        }
        break;
      }

      std::vector<size_t> dec;
      for (size_t i = 0; i < active.size(); i++) if (!active[i].prefilling) dec.push_back(i);
      if (dec.empty()) continue;

      // 2) Thinking-budget exhaustion: force the end-of-thinking span
      bool forced = false;
      for (size_t i : dec) {
        Active& a = active[i];
        if (a.thinking && a.nthink >= a.q->budget && !a.q->force.empty()) {
          a.thinking = false;
          bool stop = false;
          for (auto t : a.q->force) if ((stop = emit_tok(m, a, t))) break;
          fflush(stdout);
          if (stop) a.q->cancel = true;
          else a.b = feed(m, a, a.q->force, nullptr);
          forced = true;
        }
      }
      if (forced) continue;

      // 3) Emit every decoding lane's pending token; lanes that stop on it leave the round
      std::vector<bool> done(active.size(), false);
      std::vector<size_t> step_idx;
      for (size_t i : dec) {
        if (emit_tok(m, active[i], active[i].b)) done[i] = true;
        else step_idx.push_back(i);
      }
      fflush(stdout);

      if (!step_idx.empty()) {
        // 4) Drafts per lane, one joint verify
        const uint32_t budget = std::max(1u, ink2::MAXM / uint32_t(step_idx.size()));
        std::vector<std::pair<int, std::vector<uint32_t>>> batches;
        for (size_t i : step_idx) {
          Active& a = active[i];
          m.use_lane(a.lane);
          std::vector<uint32_t> drafts;
          auto& h = lane_hist()[a.lane];
          if (a.spec && budget > 1u) {
            if (use_lookup && h.size() >= 4) {
              a.lk->sync(h, false);
              std::vector<uint32_t> tail;
              const size_t tlen = std::min<size_t>(h.size(), 63);
              tail.insert(tail.end(), h.end() - long(tlen), h.end());
              tail.push_back(a.b);
              uint32_t mlen = 0;
              const uint32_t mpos = a.lk->find(h, tail, &mlen);
              if (mpos != opus_lookup::Lookup::NONE && mlen >= 4) {
                const uint32_t cap_lk = std::min<uint32_t>(budget - 1u, uint32_t(h.size() - mpos));
                for (uint32_t k = 0; k < cap_lk; k++) drafts.push_back(h[mpos + k]);
              }
            }
            if (drafts.empty()) {
              std::vector<float> dprobs;
              drafts = m.draft(a.b, std::min(draft_k, budget - 1u), &dprobs, nullptr, 0u);
              uint32_t keep = 0;
              while (keep < drafts.size() && (draft_cut <= 0.0f || dprobs[keep] >= draft_cut)) keep++;
              drafts.resize(std::min<uint32_t>(keep, budget - 1u));
            }
          }
          std::vector<uint32_t> v{a.b};
          v.insert(v.end(), drafts.begin(), drafts.end());
          batches.push_back({a.lane, std::move(v)});
        }
        const auto out = m.step_lanes(batches);

        // 5) Per lane: accept, emit, roll back to what was emitted, commit DSpark context
        uint32_t r0 = 0;
        for (size_t j = 0; j < step_idx.size(); j++) {
          Active& a = active[step_idx[j]];
          const auto& v = batches[j].second;
          const uint32_t mrows = uint32_t(v.size());
          uint32_t acc = 0;
          while (acc + 1u < mrows && out[r0 + acc] == v[acc + 1u]) acc++;
          bool stop = false;
          uint32_t emitted = 0;  // accepted drafts emitted (the anchor v[0] went out in step 3)
          while (emitted < acc && !stop) stop = emit_tok(m, a, v[1u + emitted++]);
          const uint32_t keep = 1u + emitted;  // rows the lane keeps: anchor + emitted drafts
          m.use_lane(a.lane);
          m.rollback(keep, mrows);
          if (m.has_mtp) {
            m.tap_row0 = r0;
            m.mtp_commit(keep);
            m.tap_row0 = 0;
          }
          auto& h = lane_hist()[a.lane];
          h.insert(h.end(), v.begin(), v.begin() + keep);
          a.b = out[r0 + keep - 1u];
          a.rounds++;
          a.accepted += keep;
          if (stop) done[step_idx[j]] = true;
          r0 += mrows;
        }
        fflush(stdout);
      }
      for (size_t i = active.size(); i-- > 0;) {
        if (done[i]) {
          finish_active(active[i], busy);
          active.erase(active.begin() + long(i));
        }
      }
    }
  }
  rt.join();
  return 0;
}

}  // namespace iserve
