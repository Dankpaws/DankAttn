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
#include <dirent.h>
#include <memory>
#include <mutex>
#include <sys/mount.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace iserve {

// Persistent SSD prefix cache for Lane 0 (mirrors Hy4's PrefixCache: atomic .tmp writes, pack fingerprint, LRU + free-disk guard).
// Env: INK_PCACHE=<dir> (default <tree>/pcache_ink, 0 = off), INK_PCACHE_GB (default 20), INK_PCACHE_FREE_GB (default 15).
struct PrefixCache {
  static constexpr uint32_t PC_MIN = 2048, PC_GAIN = 1024;
  struct Entry { std::string path; std::vector<uint32_t> toks; size_t bytes = 0; uint64_t used = 0; };
  std::string dir;
  size_t budget = 0, reserve = 0;
  bool on = false;
  uint64_t clock = 0;
  std::vector<Entry> ents;

  static uint32_t common(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, size_t n) {
    uint32_t c = 0;
    const size_t lim = std::min({a.size(), b.size(), n});
    while (c < lim && a[c] == b[c]) c++;
    return c;
  }

  void init(const ink2::Model& m, const std::string& def) {
    const char* env_d = getenv("INK_PCACHE");
    dir = env_d ? std::string(env_d) : def;
    on = !dir.empty() && dir != "0";
    const double gb = getenv("INK_PCACHE_GB") ? atof(getenv("INK_PCACHE_GB")) : 20.0;
    const double fgb = getenv("INK_PCACHE_FREE_GB") ? atof(getenv("INK_PCACHE_FREE_GB")) : 15.0;
    budget = size_t(std::max(0.0, gb) * 1e9);
    reserve = size_t(std::max(0.0, fgb) * 1e9);
    if (!on) return;
    mkdir(dir.c_str(), 0755);
    DIR* d = opendir(dir.c_str());
    if (!d) { on = false; fprintf(stderr, "inkling2: pcache: cannot open %s, off\n", dir.c_str()); return; }
    std::vector<std::pair<double, Entry>> found;
    while (dirent* de = readdir(d)) {
      const std::string n = de->d_name;
      if (n.size() < 6 || n.substr(n.size() - 5) != ".inkp") {
        if (n.size() > 4 && n.substr(n.size() - 4) == ".tmp") unlink((dir + "/" + n).c_str());
        continue;
      }
      Entry en;
      en.path = dir + "/" + n;
      struct stat st{};
      if (!ink2::Model::prefix_tokens(en.path, m.fingerprint, en.toks, &en.bytes) || stat(en.path.c_str(), &st) != 0) {
        unlink(en.path.c_str());
        continue;
      }
      found.push_back({double(st.st_mtimespec.tv_sec) + st.st_mtimespec.tv_nsec * 1e-9, std::move(en)});
    }
    closedir(d);
    std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    size_t tot = 0;
    for (auto& [t, en] : found) { en.used = ++clock; tot += en.bytes; ents.push_back(std::move(en)); }
    fprintf(stderr, "inkling2: pcache %s: %zu files, %.2f GB, budget %.0f GB\n", dir.c_str(), ents.size(), tot / 1e9, budget / 1e9);
  }

  // Find longest exact-prefix file on disk (ids.size() > file.pos && prefix matches all file.pos tokens).
  int best(const std::vector<uint32_t>& ids, uint32_t* keep) const {
    int bi = -1;
    *keep = 0;
    if (!on || ids.size() <= 1) return -1;
    for (int i = 0; i < int(ents.size()); i++) {
      const uint32_t p = uint32_t(ents[i].toks.size());
      if (p < ids.size() && p > *keep && common(ents[i].toks, ids, p) == p) {
        *keep = p;
        bi = i;
      }
    }
    return bi;
  }

  bool save(ink2::Model& m, const std::vector<uint32_t>& hist, const std::string& protect = "") {
    if (!on || m.pos < PC_MIN || m.fork_pos != 0 || hist.size() < m.pos) return false;
    for (auto& e : ents) {
      if (e.toks.size() == m.pos && common(e.toks, hist, m.pos) == m.pos) {
        e.used = ++clock;
        return false;
      }
    }
    const size_t need = size_t(m.pos) * 22000u + (size_t(190) << 20);
    auto total = [&] { size_t t = 0; for (const auto& e : ents) t += e.bytes; return t; };
    auto free_disk = [&]() -> size_t { struct statfs f{}; return statfs(dir.c_str(), &f) == 0 ? size_t(f.f_bavail) * f.f_bsize : 0; };
    auto drop_oldest = [&] {
      long o = -1;
      for (size_t i = 0; i < ents.size(); i++)
        if (ents[i].path != protect && (o < 0 || ents[i].used < ents[size_t(o)].used)) o = long(i);
      if (o < 0) return false;
      unlink(ents[size_t(o)].path.c_str());
      ents.erase(ents.begin() + o);
      return true;
    };
    while ((total() + need > budget || free_disk() < reserve + need) && drop_oldest()) {}
    if (total() + need > budget || free_disk() < reserve + need) {
      fprintf(stderr, "inkling2: pcache: no room for %u tokens (%.1f GB free, reserve %.0f GB)\n",
              m.pos, free_disk() / 1e9, reserve / 1e9);
      return false;
    }
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = 0; i < m.pos; i++) { h ^= hist[i]; h *= 1099511628211ull; }
    char nm[64];
    snprintf(nm, sizeof nm, "/%016llx-%u.inkp", (unsigned long long)h, m.pos);
    Entry en;
    en.path = dir + nm;
    double ms = 0;
    if (!m.save_prefix(en.path, hist, &ms, &en.bytes)) {
      fprintf(stderr, "inkling2: pcache: save of %u tokens failed\n", m.pos);
      return false;
    }
    en.toks.assign(hist.begin(), hist.begin() + m.pos);
    en.used = ++clock;
    for (size_t i = 0; i < ents.size();) {
      if (ents[i].path != en.path && ents[i].toks.size() < en.toks.size() &&
          common(ents[i].toks, en.toks, ents[i].toks.size()) == ents[i].toks.size()) {
        unlink(ents[i].path.c_str());
        ents.erase(ents.begin() + long(i));
      } else i++;
    }
    ents.push_back(std::move(en));
    fprintf(stderr, "inkling2: pcache: saved %u tokens (%.2f GB) in %.0f ms (%zu files, %.2f GB)\n",
            m.pos, ents.back().bytes / 1e9, ms, ents.size(), total() / 1e9);
    return true;
  }

  uint32_t restore(ink2::Model& m, int idx, std::vector<uint32_t>& hist, uint32_t rid, uint32_t c0) {
    if (idx < 0 || idx >= int(ents.size())) return 0;
    double ms = 0;
    std::vector<uint32_t> toks;
    const uint32_t got = m.load_prefix(ents[idx].path, &toks, &ms);
    if (got) {
      ents[idx].used = ++clock;
      hist = std::move(toks);
    }
    fprintf(stderr, "inkling2: rid %u pcache: %s %u tokens in %.0f ms (slot kept %u)\n",
            rid, got ? "restored" : "restore failed", got, ms, c0);
    return got;
  }
};

static PrefixCache& pcache() { static PrefixCache p; return p; }

constexpr uint32_t END = 0xFFFFFFFFu, ERR = 0xFFFFFFFEu, READY = 0xFFFFFFFDu, CAND_SEP = 0xFFFFFFFCu;

struct Req {
  uint32_t rid = 0, max_new = 0, flags = 0;
  uint32_t fpr_lanes = 0, fpr_quorum = 0, fpr_opts = 0;  // flags & 2: Fork-Prune-Resolve / Parallel Fill
  std::vector<std::vector<uint32_t>> fpr_suffixes;       // mode == 2: per-lane section prompts
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
    if (q->flags & 2) {
      uint32_t g3[3];
      if (!rd(g3, 12)) break;
      q->fpr_lanes = g3[0];
      q->fpr_quorum = g3[1];
      q->fpr_opts = g3[2];
      if ((q->fpr_opts & 0xFu) == 2u) {
        q->fpr_suffixes.resize(q->fpr_lanes);
        bool ok = true;
        for (uint32_t k = 0; k < q->fpr_lanes; k++) {
          uint32_t slen = 0;
          if (!rd(&slen, 4)) { ok = false; break; }
          q->fpr_suffixes[k].resize(slen);
          if (slen && !rd(q->fpr_suffixes[k].data(), size_t(slen) * 4u)) { ok = false; break; }
        }
        if (!ok) break;
      }
    }
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
  const uint32_t lpos = m.lane_pos(a.lane);
  const uint32_t fpos = m.lane_fork_pos(a.lane);
  const uint32_t used = (lpos >= fpos) ? (lpos - fpos) : lpos;
  if (a.n >= a.q->max_new || lpos + ink2::MAXM + 1 >= m.cap || used + ink2::MAXM + 1 >= m.lane_global_cap(a.lane)) {
    a.length = true;
    return true;
  }
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

// Highest fork_pos among currently active side lanes reading Lane 0's global KV.
// Lane 0 must not reset or rewind below this floor while any such side lane is busy.
static uint32_t active_fork_floor(const ink2::Model& m, const std::vector<bool>& busy) {
  uint32_t fl = 0;
  for (int i = 1; i < int(busy.size()); i++) {
    if (busy[i] && m.lane_fork_src(i) == 0) fl = std::max(fl, m.lane_fork_pos(i));
  }
  return fl;
}

// Clear any idle side lane whose forked prefix in Lane 0 is about to be overwritten.
static void invalidate_stale_forks(ink2::Model& m, const std::vector<bool>& busy, uint32_t new_pos0) {
  auto& H = lane_hist();
  auto& CK = lane_ckpt();
  for (int i = 1; i < int(busy.size()); i++) {
    if (!busy[i] && m.lane_fork_src(i) == 0 && m.lane_fork_pos(i) > new_pos0) {
      m.clear_lane_fork(i);
      H[i].clear();
      CK[i] = {};
    }
  }
}

// Longest free-lane reuse: a lane whose processed tokens are a strict prefix of ids. Ties and no reuse prefer
// the lane with the least to lose (shortest history), so a long conversation stays resumable.
// When Lane 0 holds a matching prefix (live h0 or rewindable ck0) and is busy (or a side lane gives strictly
// better reuse), an idle side lane can Split-KV fork from Lane 0 if the suffix fits its side cap.
// fork_mode: 0 = direct, 1 = fork_lane(0, ln) at h0.size(), 2 = fork_lane_at(0, ln, ck0.pos, ck0.conv).
static int pick_lane(const ink2::Model& m, const std::vector<uint32_t>& ids, const std::vector<bool>& busy,
                     uint32_t* reuse, bool* rewind, int* fork_mode) {
  int best = -1;
  uint32_t bc = 0;
  size_t bl = 0;
  bool brw = false;
  int bfm = 0;
  auto& H = lane_hist();
  auto& CK = lane_ckpt();
  const uint32_t floor0 = active_fork_floor(m, busy);

  for (int i = 0; i < int(busy.size()); i++) {
    if (busy[i]) continue;
    const auto& h = H[i];
    const auto& ck = CK[i];
    const uint32_t fpos = (m.lane_fork_src(i) == 0) ? m.lane_fork_pos(i) : 0u;
    uint32_t c = 0;
    bool rw = false;
    if (!h.empty() && h.size() < ids.size() && std::equal(h.begin(), h.end(), ids.begin())) {
      c = uint32_t(h.size());
    } else if (ck.pos > 0 && ck.pos >= fpos && ck.pos < ids.size() && ck.pos <= h.size() &&
               h.size() - ck.pos <= m.rewind_slack() &&
               std::equal(h.begin(), h.begin() + ck.pos, ids.begin())) {
      c = ck.pos;
      rw = true;
    }
    if (i == 0 && floor0 > 0 && c < floor0) continue;  // protect active forked side lanes
    const uint32_t eff_used = (c > 0 && fpos > 0) ? uint32_t(ids.size() - fpos) : uint32_t(ids.size());
    if (eff_used + ink2::MAXM + 1u >= m.lane_global_cap(i)) continue;
    if (best < 0 || c > bc || (c == bc && h.size() < bl)) {
      best = i; bc = c; bl = h.size(); brw = rw; bfm = 0;
    }
  }

  // Split-KV fork from Lane 0 -> an idle side lane (works even while Lane 0 is busy decoding!)
  static const bool allow_splitkv = !(getenv("INK_SPLITKV_SERVE") && atoi(getenv("INK_SPLITKV_SERVE")) == 0);
  if (allow_splitkv && busy.size() > 1 && ids.size() + ink2::MAXM + 1u < m.cap) {
    const auto& h0 = H[0];
    const auto& ck0 = CK[0];
    uint32_t fc = 0;
    int fm = 0;
    if (h0.size() >= 64u && h0.size() < ids.size() && std::equal(h0.begin(), h0.end(), ids.begin())) {
      fc = uint32_t(h0.size());
      fm = 1;
    } else if (ck0.pos >= 64u && ck0.pos < ids.size() && ck0.pos <= h0.size() &&
               h0.size() - ck0.pos <= m.rewind_slack() &&
               std::equal(h0.begin(), h0.begin() + ck0.pos, ids.begin())) {
      fc = ck0.pos;
      fm = 2;
    }
    if (fc > 0 && (best < 0 || fc > bc)) {
      const uint32_t suffix_len = uint32_t(ids.size() - fc);
      int fk = -1;
      size_t fkl = 0;
      for (int i = 1; i < int(busy.size()); i++) {
        if (busy[i]) continue;
        if (suffix_len + ink2::MAXM + 1u >= m.lane_global_cap(i)) continue;
        if (fk < 0 || H[i].size() < fkl) { fk = i; fkl = H[i].size(); }
      }
      if (fk >= 0) {
        best = fk; bc = fc; brw = false; bfm = fm;
      }
    }
  }

  *reuse = bc;
  *rewind = brw;
  *fork_mode = bfm;
  return best;
}

static void run_fpr(ink2::Model& m, const ReqP& q, uint32_t n_lanes, uint32_t draft_k, float draft_cut, bool use_lookup) {
  const auto t0 = std::chrono::steady_clock::now();
  const uint32_t K = std::max(2u, std::min(q->fpr_lanes ? q->fpr_lanes : n_lanes, n_lanes));
  const uint32_t Q = (q->fpr_quorum == 0u) ? K : std::max(1u, std::min(q->fpr_quorum, K));
  const uint32_t mode = q->fpr_opts & 0xFu;        // 0 = select, 1 = candidates
  const uint32_t prune_lvl = (q->fpr_opts >> 4) & 0xFu;
  const bool spec = (q->flags & 1) && m.has_mtp;
  auto is_stop = [&](uint32_t t) {
    return t == m.eos || std::find(q->stops.begin(), q->stops.end(), t) != q->stops.end();
  };

  // Stage 0: Prefill once on Lane 0 (with full prefix reuse / rewind!)
  m.use_lane(0);
  auto& h0 = lane_hist()[0];
  auto& ck0 = lane_ckpt()[0];
  uint32_t reuse = 0;
  bool rw = false;
  if (!h0.empty() && h0.size() < q->ids.size() && std::equal(h0.begin(), h0.end(), q->ids.begin())) {
    reuse = uint32_t(h0.size());
  } else if (ck0.pos > 0 && ck0.pos < q->ids.size() && ck0.pos <= h0.size() &&
             h0.size() - ck0.pos <= m.rewind_slack() &&
             std::equal(h0.begin(), h0.begin() + ck0.pos, q->ids.begin())) {
    reuse = ck0.pos;
    rw = true;
  }
  if (pcache().on) {
    uint32_t dkeep = 0;
    const int di = pcache().best(q->ids, &dkeep);
    const bool want_disk = (di >= 0 && dkeep > reuse + PrefixCache::PC_GAIN);
    if (reuse == 0 || want_disk) {
      pcache().save(m, h0, want_disk ? pcache().ents[di].path : "");
    }
    if (want_disk) {
      const uint32_t got = pcache().restore(m, di, h0, q->rid, reuse);
      if (got) { reuse = got; rw = false; }
    }
  }
  if (reuse == 0) { m.reset(); h0.clear(); }
  else if (rw) { m.rewind(reuse, ck0.conv); h0.resize(reuse); }

  ink2::Model::Samp base_sp;
  base_sp.temp = q->temp > 0 ? q->temp : 0.0f;
  base_sp.top_p = (q->top_p > 0.0f && q->top_p <= 1.0f) ? q->top_p : 1.0f;
  base_sp.top_k = q->top_k;
  base_sp.min_p = q->min_p >= 0.0f ? q->min_p : 0.0f;
  base_sp.seed = q->seed;
  m.samp = base_sp;

  double pms = 0.0;
  uint32_t b0 = 0;
  float conf0 = 0.0f;
  for (size_t i = reuse; i < q->ids.size(); i += ink2::MP) {
    if (q->cancel.load()) break;
    std::vector<uint32_t> c(q->ids.begin() + long(i), q->ids.begin() + long(std::min(q->ids.size(), i + ink2::MP)));
    double g = 0;
    b0 = m.step(c, &g).back();
    conf0 = m.last_conf.empty() ? 0.0f : m.last_conf.back();
    pms += g;
    if (m.has_mtp) m.mtp_commit(uint32_t(c.size()));
    h0.insert(h0.end(), c.begin(), c.end());
  }
  if (q->cancel.load()) {
    const float wall = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count() * 1e3f;
    put(q->rid, END);
    uint32_t s6[6] = {reuse, uint32_t(q->ids.size()), 0u, 0u, 0u, 0u};
    float f2[2] = {float(pms), wall};
    fwrite(s6, 4, 6, stdout); fwrite(f2, 4, 2, stdout); fflush(stdout);
    std::lock_guard<std::mutex> g(mu); live.erase(q->rid);
    return;
  }

  const uint32_t fork_pos = m.pos;
  ck0.pos = fork_pos;
  m.save_conv(ck0.conv);

  // Zero-copy Split-KV fork from Lane 0 -> Lanes 1..K-1
  for (uint32_t k = 1; k < K; k++) {
    m.fork_lane(0, int(k));
    ink2::Model::Samp spk = base_sp;
    if (mode != 2u && spk.temp <= 0.0f) { spk.temp = 0.8f; if (spk.min_p <= 0.0f) spk.min_p = 0.02f; }
    spk.seed = q->seed + k;
    m.lanes_[k].samp = spk;
    lane_hist()[k] = h0;
  }

  struct FprLane {
    int lane = 0;
    uint32_t b = 0;
    float b_conf = 0.0f;
    std::vector<uint32_t> toks;
    std::vector<float> confs;
    bool done = false, pruned = false, length = false;
    uint32_t rounds = 0, accepted = 0;
    std::unique_ptr<opus_lookup::Lookup> lk;
  };
  std::vector<FprLane> fls(K);
  for (uint32_t k = 0; k < K; k++) {
    fls[k].lane = int(k);
    fls[k].b = b0;
    fls[k].b_conf = conf0;
    fls[k].lk = std::make_unique<opus_lookup::Lookup>();
  }

  // Mode 2 (Parallel Section Fill): batch all K lanes' short section prompts via step_lanes() + mtp_commit_lanes()!
  if (mode == 2u) {
    std::vector<size_t> soff(K, 0);
    for (;;) {
      std::vector<std::pair<int, std::vector<uint32_t>>> batches;
      std::vector<uint32_t> bk;
      uint32_t rem_budget = ink2::MAXM;
      for (uint32_t k = 0; k < K && rem_budget > 0u; k++) {
        if (k >= q->fpr_suffixes.size() || soff[k] >= q->fpr_suffixes[k].size()) continue;
        const size_t rem = q->fpr_suffixes[k].size() - soff[k];
        const uint32_t take = uint32_t(std::min<size_t>(rem, rem_budget));
        std::vector<uint32_t> c(q->fpr_suffixes[k].begin() + long(soff[k]),
                                q->fpr_suffixes[k].begin() + long(soff[k] + take));
        soff[k] += take;
        rem_budget -= take;
        batches.push_back({int(k), std::move(c)});
        bk.push_back(k);
      }
      if (batches.empty()) break;
      double g = 0;
      auto out = m.step_lanes(batches, &g);
      pms += g;
      std::vector<ink2::Model::MtpCommitReq> creqs;
      uint32_t r0 = 0;
      for (size_t j = 0; j < batches.size(); j++) {
        const uint32_t k = bk[j];
        const auto& c = batches[j].second;
        const uint32_t mrows = uint32_t(c.size());
        fls[k].b = out[r0 + mrows - 1u];
        fls[k].b_conf = (r0 + mrows - 1u < m.last_conf.size()) ? m.last_conf[r0 + mrows - 1u] : 0.0f;
        if (m.has_mtp) creqs.push_back({int(k), r0, mrows});
        auto& hk = lane_hist()[k];
        hk.insert(hk.end(), c.begin(), c.end());
        r0 += mrows;
      }
      if (m.has_mtp && !creqs.empty()) m.mtp_commit_lanes(creqs, r0);
    }
    m.use_lane(0);
  }

  uint32_t n_emit = 0;
  uint32_t next_flush = 1u;  // for mode == 2: lane 0 streams live, lanes 1..K-1 flush in order as soon as ready
  if (mode == 2u) {
    const float zf = 0.0f; const uint32_t zu = 0u;
    put(q->rid, CAND_SEP);
    fwrite(&zf, 4, 1, stdout);
    fwrite(&zu, 4, 1, stdout);
    fflush(stdout);
  }

  auto push_tok = [&](FprLane& fl, uint32_t t, float cf) -> bool {
    fl.toks.push_back(t);
    fl.confs.push_back(cf);
    if (mode == 2u && fl.lane == 0) {
      put(q->rid, t);
      n_emit++;
    }
    if (is_stop(t)) { fl.done = true; return true; }
    const uint32_t lcap = m.lane_global_cap(fl.lane);
    const uint32_t used = (fl.lane == 0) ? m.lane_pos(0) : (m.lane_pos(fl.lane) - fork_pos);
    if (fl.toks.size() >= q->max_new || used + ink2::MAXM + 1u >= lcap) {
      fl.length = true;
      fl.done = true;
      return true;
    }
    return false;
  };

  auto flush_ready_sections = [&]() {
    if (mode != 2u || !fls[0].done) return;
    while (next_flush < K && fls[next_flush].done) {
      const auto& fl = fls[next_flush];
      const float tc = 0.0f;
      const uint32_t cf = fl.length ? 1u : 0u;
      put(q->rid, CAND_SEP);
      fwrite(&tc, 4, 1, stdout);
      fwrite(&cf, 4, 1, stdout);
      for (uint32_t t : fl.toks) { put(q->rid, t); n_emit++; }
      next_flush++;
    }
  };

  uint32_t n_finished = 0, n_pruned = 0;
  uint32_t tot_rounds = 0, tot_acc = 0;

  // Stage 1: Parallel Exploration + Online DeepConf Pruning + Quorum
  while (n_finished < Q && !q->cancel.load()) {
    std::vector<uint32_t> step_idx;
    for (uint32_t k = 0; k < K; k++) {
      if (fls[k].done || fls[k].pruned) continue;
      if (push_tok(fls[k], fls[k].b, fls[k].b_conf)) n_finished++;
      else step_idx.push_back(k);
    }
    if (n_finished >= Q || step_idx.empty()) break;

    const uint32_t budget = std::max(1u, ink2::MAXM / uint32_t(step_idx.size()));
    const uint32_t cmax = budget > 1u ? (budget - 1u) : 0u;
    std::vector<std::vector<uint32_t>> lane_drafts(step_idx.size());
    std::vector<ink2::LookupProp> lprops(step_idx.size());
    std::vector<ink2::Model::MtpDraftReq> dreqs;
    std::vector<size_t> dreq_map;
    for (size_t j = 0; j < step_idx.size(); j++) {
      FprLane& fl = fls[step_idx[j]];
      auto& h = lane_hist()[fl.lane];
      if (spec && cmax > 0u) {
        lprops[j] = ink2::lookup_prepare(*fl.lk, h, fl.b, cmax, use_lookup);
        if (lprops[j].lk_only) {
          lane_drafts[j] = ink2::lookup_merge(*fl.lk, h, lprops[j], {}, {}, cmax, draft_cut, m.big_gpu, uint32_t(step_idx.size()));
        } else {
          dreqs.push_back({fl.lane, fl.b});
          dreq_map.push_back(j);
        }
      }
    }
    if (!dreqs.empty()) {
      std::vector<std::vector<uint32_t>> d_out;
      std::vector<std::vector<float>> d_probs;
      m.draft_lanes(dreqs, std::min(draft_k, cmax), d_out, &d_probs);
      for (size_t di = 0; di < dreqs.size(); di++) {
        const size_t j = dreq_map[di];
        FprLane& fl = fls[step_idx[j]];
        lane_drafts[j] = ink2::lookup_merge(*fl.lk, lane_hist()[fl.lane], lprops[j], d_out[di], d_probs[di],
                                            cmax, draft_cut, m.big_gpu, uint32_t(step_idx.size()));
      }
    }
    std::vector<std::pair<int, std::vector<uint32_t>>> batches;
    for (size_t j = 0; j < step_idx.size(); j++) {
      FprLane& fl = fls[step_idx[j]];
      std::vector<uint32_t> v{fl.b};
      v.insert(v.end(), lane_drafts[j].begin(), lane_drafts[j].end());
      batches.push_back({fl.lane, std::move(v)});
    }
    const auto out = m.step_lanes(batches);

    uint32_t r0 = 0;
    std::vector<ink2::Model::MtpCommitReq> creqs;
    for (size_t j = 0; j < step_idx.size(); j++) {
      FprLane& fl = fls[step_idx[j]];
      const auto& v = batches[j].second;
      const uint32_t mrows = uint32_t(v.size());
      uint32_t acc = 0;
      while (acc + 1u < mrows && out[r0 + acc] == v[acc + 1u]) acc++;
      if (spec) ink2::lookup_update(*fl.lk, lprops[j], acc);
      bool stop = false;
      uint32_t emitted = 0;
      while (emitted < acc && !stop) {
        const float cf = (r0 + emitted < m.last_conf.size()) ? m.last_conf[r0 + emitted] : 0.0f;
        stop = push_tok(fl, v[1u + emitted], cf);
        emitted++;
      }
      const uint32_t keep = 1u + emitted;
      m.use_lane(fl.lane);
      m.rollback(keep, mrows);
      if (m.has_mtp) creqs.push_back({fl.lane, r0, keep});
      auto& h = lane_hist()[fl.lane];
      h.insert(h.end(), v.begin(), v.begin() + keep);
      fl.b = out[r0 + keep - 1u];
      fl.b_conf = (r0 + keep - 1u < m.last_conf.size()) ? m.last_conf[r0 + keep - 1u] : 0.0f;
      fl.rounds++;
      fl.accepted += keep;
      tot_rounds++;
      tot_acc += keep;
      if (stop) n_finished++;
      r0 += mrows;
    }
    if (m.has_mtp && !creqs.empty()) m.mtp_commit_lanes(creqs, r0);
    if (mode == 2u) {
      flush_ready_sections();
      fflush(stdout);
    }

    // Online DeepConf confidence pruning (drop clearly spiraling lanes early when n_alive > remaining needed)
    if (prune_lvl > 0u) {
      std::vector<std::pair<float, uint32_t>> alive_sc;
      uint32_t min_len = 0xFFFFFFFFu;
      for (uint32_t k = 0; k < K; k++) {
        if (fls[k].done || fls[k].pruned) continue;
        min_len = std::min(min_len, uint32_t(fls[k].confs.size()));
        const size_t n = fls[k].confs.size();
        const size_t w = std::min<size_t>(n, 64u);
        float s = 0.0f;
        for (size_t i = n - w; i < n; i++) s += fls[k].confs[i];
        alive_sc.push_back({w ? s / float(w) : 0.0f, k});
      }
      const uint32_t need = (Q > n_finished) ? (Q - n_finished) : 0u;
      if (min_len >= 128u && alive_sc.size() > need && alive_sc.size() >= 2u) {
        std::sort(alive_sc.begin(), alive_sc.end());
        const float med = alive_sc[alive_sc.size() / 2].first;
        const float thr = (prune_lvl == 1u ? 0.72f : 0.82f) * med;
        if (alive_sc[0].first < thr) {
          fls[alive_sc[0].second].pruned = true;
          n_pruned++;
        }
      }
    }
  }

  // Stage 2: Resolve
  auto tail_conf = [](const FprLane& fl) {
    const size_t n = fl.confs.size();
    const size_t w = std::min<size_t>(n, 128u);
    if (w == 0) return 0.0f;
    double s = 0.0;
    for (size_t i = n - w; i < n; i++) s += fl.confs[i];
    return float(s / double(w));
  };

  std::vector<uint32_t> cands;
  if (mode == 2u) {
    for (uint32_t k = 0; k < K; k++) cands.push_back(k);  // exact section order 0..K-1
  } else {
    for (uint32_t k = 0; k < K; k++) if (fls[k].done && !fls[k].pruned) cands.push_back(k);
    if (cands.empty()) {
      for (uint32_t k = 0; k < K; k++) if (!fls[k].pruned) cands.push_back(k);
    }
    if (cands.empty()) cands.push_back(0u);
  }

  bool out_len = false;
  int win_lane = int(cands[0]);
  float win_conf = 0.0f;

  if (q->cancel.load()) {
    m.use_lane(0);
    m.rewind(fork_pos, ck0.conv);
    lane_hist()[0] = q->ids;
  } else if (mode == 2u) {
    // Flush any remaining buffered sections in order 1..K-1 (lane 0 was streamed live!)
    while (next_flush < K) {
      const auto& fl = fls[next_flush];
      const float tc = tail_conf(fl);
      const uint32_t cf = fl.length ? 1u : 0u;
      put(q->rid, CAND_SEP);
      fwrite(&tc, 4, 1, stdout);
      fwrite(&cf, 4, 1, stdout);
      for (uint32_t t : fl.toks) { put(q->rid, t); n_emit++; }
      next_flush++;
    }
    m.use_lane(0);
    m.rewind(fork_pos, ck0.conv);
    lane_hist()[0] = q->ids;
  } else if (mode == 0u) {
    // Mode 0: in-engine Self-Certainty + Consensus selection
    auto tail_sig = [&](const FprLane& fl) {
      std::vector<uint32_t> sig;
      for (uint32_t t : fl.toks) if (!is_stop(t)) sig.push_back(t);
      if (sig.size() > 48u) sig.erase(sig.begin(), sig.end() - 48);
      return sig;
    };
    std::vector<std::vector<uint32_t>> sigs;
    for (uint32_t k : cands) sigs.push_back(tail_sig(fls[k]));
    float best_sc = -1e30f;
    size_t best_i = 0;
    for (size_t i = 0; i < cands.size(); i++) {
      const auto& fl = fls[cands[i]];
      const float tc = tail_conf(fl);
      uint32_t peers = 0;
      for (size_t j = 0; j < cands.size(); j++)
        if (j != i && !sigs[i].empty() && sigs[i] == sigs[j]) peers++;
      const float sc = tc + 0.5f * float(peers) - (fl.length ? 10.0f : 0.0f);
      if (sc > best_sc) { best_sc = sc; best_i = i; win_conf = tc; }
    }
    const auto& win = fls[cands[best_i]];
    win_lane = win.lane;
    out_len = win.length;
    for (uint32_t t : win.toks) { put(q->rid, t); n_emit++; }
    if (win.lane != 0) {
      m.promote_lane(win.lane, 0);
      lane_hist()[0] = lane_hist()[win.lane];
    }
    m.use_lane(0);
  } else {
    // Mode 1: emit all surviving candidates separated by CAND_SEP, rewind Lane 0 to fork_pos for zero-prefill judge/synth!
    for (uint32_t k : cands) {
      const auto& fl = fls[k];
      const float tc = tail_conf(fl);
      const uint32_t cf = fl.length ? 1u : 0u;
      put(q->rid, CAND_SEP);
      fwrite(&tc, 4, 1, stdout);
      fwrite(&cf, 4, 1, stdout);
      for (uint32_t t : fl.toks) { put(q->rid, t); n_emit++; }
    }
    m.use_lane(0);
    m.rewind(fork_pos, ck0.conv);
    lane_hist()[0] = q->ids;
  }

  for (uint32_t k = 1; k < n_lanes; k++) {
    m.clear_lane_fork(int(k));
    lane_hist()[k].clear();
    lane_ckpt()[k] = {};
  }
  const float wall = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count() * 1e3f;
  put(q->rid, END);
  uint32_t s6[6] = {reuse, uint32_t(q->ids.size()), n_emit, tot_rounds, tot_acc, out_len ? 1u : 0u};
  float f2[2] = {float(pms), wall};
  fwrite(s6, 4, 6, stdout);
  fwrite(f2, 4, 2, stdout);
  fflush(stdout);
  fprintf(stderr, "inkling2: rid %u FPR(%s) K=%u Q=%u pruned=%u prompt %zu (reused %u, prefill %.0f ms) emit %u (agg %u) in %.0f ms (win=L%d sc=%.2f)\n",
          q->rid, mode == 0u ? "select" : (mode == 2u ? "fast" : "cands"), K, Q, n_pruned, q->ids.size(), reuse, pms, n_emit, tot_acc, wall, win_lane, win_conf);
  std::lock_guard<std::mutex> g(mu);
  live.erase(q->rid);
}

static void admit_requests(ink2::Model& m, std::vector<Active>& active, std::vector<bool>& busy,
                           uint32_t n_lanes, uint32_t draft_k, float draft_cut, bool use_lookup) {
  for (;;) {
    ReqP q;
    {
      std::lock_guard<std::mutex> g(mu);
      if (inbox.empty()) return;
      if (inbox.front()->flags & 2) {
        if (!active.empty()) return;  // wait until all lanes are idle for FPR
      } else {
        if (std::find(busy.begin(), busy.end(), false) == busy.end()) return;
      }
      q = inbox.front();
      inbox.pop_front();
    }
    std::string why;
    if (q->cancel.load()) why = "";
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
    if (q->flags & 2) {
      run_fpr(m, q, n_lanes, draft_k, draft_cut, use_lookup);
      continue;
    }
    uint32_t reuse = 0;
    bool rw = false;
    int fork_mode = 0;
    const int ln = pick_lane(m, q->ids, busy, &reuse, &rw, &fork_mode);
    if (ln < 0) {
      std::lock_guard<std::mutex> g(mu);
      inbox.push_front(q);
      return;
    }
    auto& h = lane_hist()[ln];
    if (fork_mode > 0) {
      const auto& h0 = lane_hist()[0];
      if (fork_mode == 1) {
        m.fork_lane(0, ln);
        h = h0;
      } else {
        m.fork_lane_at(0, ln, reuse, lane_ckpt()[0].conv);
        h.assign(h0.begin(), h0.begin() + reuse);
      }
      lane_ckpt()[ln] = {};
      m.use_lane(ln);
      fprintf(stderr, "inkling2: rid %u lane %d split-kv forked from L0 @ %u (mode=%d, suffix=%zu)\n",
              q->rid, ln, reuse, fork_mode, q->ids.size() - reuse);
    } else {
      m.use_lane(ln);
      if (ln == 0 && pcache().on && active_fork_floor(m, busy) == 0u) {
        uint32_t dkeep = 0;
        const int di = pcache().best(q->ids, &dkeep);
        const bool want_disk = (di >= 0 && dkeep > reuse + PrefixCache::PC_GAIN);
        if (reuse == 0 || want_disk) {
          pcache().save(m, h, want_disk ? pcache().ents[di].path : "");
        }
        if (want_disk) {
          const uint32_t got = pcache().restore(m, di, h, q->rid, reuse);
          if (got) {
            invalidate_stale_forks(m, busy, 0u);
            reuse = got;
            rw = false;
          }
        }
      }
      if (reuse == 0) {
        if (ln == 0) invalidate_stale_forks(m, busy, 0u);
        m.reset();
        h.clear();
      } else if (rw) {
        if (ln == 0) invalidate_stale_forks(m, busy, reuse);
        m.rewind(reuse, lane_ckpt()[ln].conv);
        h.resize(reuse);
      }
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
  const uint32_t side_cap = getenv("INK_SIDE_CAP") ? uint32_t(atoi(getenv("INK_SIDE_CAP"))) : 0u;
  while (m.lanes_.size() < n_lanes) m.add_lane(side_cap);
  lane_hist().assign(n_lanes, {});
  lane_ckpt().assign(n_lanes, {});
  const uint32_t draft_k = getenv("INK_DRAFT_K") ? uint32_t(atoi(getenv("INK_DRAFT_K"))) : 7u;
  const float draft_cut = getenv("INK_DRAFT_CUT") ? float(atof(getenv("INK_DRAFT_CUT"))) : 0.50f;
  const bool use_lookup = getenv("INK_LOOKUP") ? (atoi(getenv("INK_LOOKUP")) != 0) : true;
  const uint32_t pf_dec_chunk = getenv("INK_PF_DEC_CHUNK") ? uint32_t(atoi(getenv("INK_PF_DEC_CHUNK"))) : 256u;
  pcache().init(m, mtl::tree_root() + "/pcache_ink");
  fprintf(stderr, "inkling2: serving %u lanes (L0=%u, side=%u, rewind slack %u), GPU state %.2f GB, DSpark %s, power %s\n", n_lanes,
          m.cap, side_cap ? side_cap : m.cap, m.rewind_slack(), m.C.allocated / 1e9, m.has_mtp ? "on" : "off",
          ink2::power_mode_name());
  std::thread rt(reader);
  put(0, READY);
  fflush(stdout);

  std::vector<Active> active;
  std::vector<bool> busy(n_lanes, false);

  for (;;) {
    admit_requests(m, active, busy, n_lanes, draft_k, draft_cut, use_lookup);
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
      const uint32_t max_pf = ink2::eco_mode() ? std::min<uint32_t>(ink2::MP, 256u) : ink2::MP;
      const uint32_t chunk = any_decoding ? std::max<uint32_t>(ink2::MAXM + 1u, std::min(max_pf, pf_dec_chunk)) : max_pf;
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
        // 4) Batched drafts across lanes, one joint verify
        const uint32_t budget = std::max(1u, ink2::MAXM / uint32_t(step_idx.size()));
        const uint32_t cmax = budget > 1u ? (budget - 1u) : 0u;
        std::vector<std::vector<uint32_t>> lane_drafts(step_idx.size());
        std::vector<ink2::LookupProp> lprops(step_idx.size());
        std::vector<ink2::Model::MtpDraftReq> dreqs;
        std::vector<size_t> dreq_map;
        for (size_t j = 0; j < step_idx.size(); j++) {
          Active& a = active[step_idx[j]];
          auto& h = lane_hist()[a.lane];
          if (a.spec && cmax > 0u) {
            lprops[j] = ink2::lookup_prepare(*a.lk, h, a.b, cmax, use_lookup);
            if (lprops[j].lk_only) {
              lane_drafts[j] = ink2::lookup_merge(*a.lk, h, lprops[j], {}, {}, cmax, draft_cut, m.big_gpu, uint32_t(step_idx.size()));
            } else {
              dreqs.push_back({a.lane, a.b});
              dreq_map.push_back(j);
            }
          }
        }
        if (!dreqs.empty()) {
          std::vector<std::vector<uint32_t>> d_out;
          std::vector<std::vector<float>> d_probs;
          m.draft_lanes(dreqs, std::min(draft_k, cmax), d_out, &d_probs);
          for (size_t di = 0; di < dreqs.size(); di++) {
            const size_t j = dreq_map[di];
            Active& a = active[step_idx[j]];
            lane_drafts[j] = ink2::lookup_merge(*a.lk, lane_hist()[a.lane], lprops[j], d_out[di], d_probs[di],
                                                cmax, draft_cut, m.big_gpu, uint32_t(step_idx.size()));
          }
        }
        std::vector<std::pair<int, std::vector<uint32_t>>> batches;
        for (size_t j = 0; j < step_idx.size(); j++) {
          Active& a = active[step_idx[j]];
          std::vector<uint32_t> v{a.b};
          v.insert(v.end(), lane_drafts[j].begin(), lane_drafts[j].end());
          batches.push_back({a.lane, std::move(v)});
        }
        const auto out = m.step_lanes(batches);

        // 5) Per lane: accept, emit, roll back, then ONE batched DSpark commit across all lanes
        uint32_t r0 = 0;
        std::vector<ink2::Model::MtpCommitReq> creqs;
        for (size_t j = 0; j < step_idx.size(); j++) {
          Active& a = active[step_idx[j]];
          const auto& v = batches[j].second;
          const uint32_t mrows = uint32_t(v.size());
          uint32_t acc = 0;
          while (acc + 1u < mrows && out[r0 + acc] == v[acc + 1u]) acc++;
          if (a.spec) ink2::lookup_update(*a.lk, lprops[j], acc);
          bool stop = false;
          uint32_t emitted = 0;  // accepted drafts emitted (the anchor v[0] went out in step 3)
          while (emitted < acc && !stop) stop = emit_tok(m, a, v[1u + emitted++]);
          const uint32_t keep = 1u + emitted;  // rows the lane keeps: anchor + emitted drafts
          m.use_lane(a.lane);
          m.rollback(keep, mrows);
          if (m.has_mtp) creqs.push_back({a.lane, r0, keep});
          auto& h = lane_hist()[a.lane];
          h.insert(h.end(), v.begin(), v.begin() + keep);
          a.b = out[r0 + keep - 1u];
          a.rounds++;
          a.accepted += keep;
          if (stop) done[step_idx[j]] = true;
          r0 += mrows;
        }
        if (m.has_mtp && !creqs.empty()) m.mtp_commit_lanes(creqs, r0);
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
  m.use_lane(0);
  pcache().save(m, lane_hist()[0]);
  rt.join();
  return 0;
}

}  // namespace iserve
