// DankAttn Hy4 serve: the DankAttn binary protocol (engine/serve.inc) with continuous multi-lane batching.
//   in,  request: 'OPUS', rid, n, ids[n], max_new, flags (1 = speculate, 4 = thinking budget, 16 = min_p), nstop, stops[nstop],
//                 f32 temperature (<= 0 greedy), f32 top_p, u32 top_k, u64 seed, then for flags & 4: budget,
//                 end-of-thinking token, nforce, force[nforce]; for flags & 16: f32 min_p.
//   in,  cancel:  'OPCX', rid
//   out: (rid, token) records; (rid, 0xFFFFFFFF) then u32 {reused, n_prompt, n_gen, rounds, accepted, length}
//        and f32 {prefill_ms_gpu, wall_ms}; (rid, 0xFFFFFFFE), u32 length, message; (0, 0xFFFFFFFD) when ready.
// temperature, min_p, top_k, and top_p are applied on GPU (both target sampling and DFly2 drafting).
// 1 active request uses full DFly2 + context-lookup speculation; 2+ active requests speculate jointly: each lane
// drafts, one step_lanes() verifies every lane's rows (HY4_JS=0: plain one-row lanes).
#include "../core/pcache.h"
#include "../core/proto.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

namespace serve {

using dankattn::proto::END;
using dankattn::proto::ERR;
using dankattn::proto::READY;
using dankattn::proto::VisImg;
using Req = dankattn::proto::OpusReq;
using ReqP = std::shared_ptr<Req>;

std::mutex mu;
std::condition_variable cv;
std::deque<ReqP> inbox;
std::map<uint32_t, ReqP> live;
bool closed = false;

void put(uint32_t rid, uint32_t w) { dankattn::proto::put(rid, w, false); }
void error(uint32_t rid, const std::string& msg) {
  dankattn::proto::send_err(rid, msg);
  fprintf(stderr, "hy4: rid %u refused: %s\n", rid, msg.c_str());
}

void reader() {
  const dankattn::proto::ReadOpts ropts{.keep_images = true, .parse_fpr_suffixes = false};
  for (;;) {
    auto q = std::make_shared<Req>();
    uint32_t cancel_rid = 0;
    const auto kind = dankattn::proto::read_packet(*q, cancel_rid, ropts);
    if (kind == dankattn::proto::PacketKind::Eof) break;
    if (kind == dankattn::proto::PacketKind::Cancel) {
      std::lock_guard<std::mutex> g(mu);
      if (auto it = live.find(cancel_rid); it != live.end()) it->second->cancel = true;
      continue;
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

// feed toks (prefill chunks or one verify / decode step); returns the next token and where the source
// row of the last fed token sits for the drafter. Checks cancel between chunks.
static uint32_t feed(hy4::Model& m, const std::vector<uint32_t>& toks, uint32_t& sx, uint32_t& ss, double* gpu,
                     const std::atomic<bool>* cancel = nullptr) {
  uint32_t b = 0;
  for (size_t i = 0; i < toks.size(); i += m.MR) {
    if (cancel && cancel->load()) return 0;
    std::vector<uint32_t> c(toks.begin() + i, toks.begin() + std::min(toks.size(), i + m.MR));
    if (i + m.MR < toks.size()) m.next_hint.assign(toks.begin() + i + m.MR, toks.begin() + std::min(toks.size(), i + 2 * m.MR));
    double g = 0;
    b = m.step(c, &g).back();
    if (gpu) *gpu += g;
    sx = ss = c.size() > hy4::MAXM ? 0 : uint32_t(c.size()) - 1;
  }
  return b;
}

// SSD prefix cache for the main slot. Multiple sessions and subagents take turns on slot 0, and each
// switch re-prefilled the whole conversation (17-32 s at 20-36k tokens, minutes past 200k). Before a request
// drops more than PC_MIN tokens of slot 0's state, the state goes to a file (Model::save_prefix, ~8.5 KB a
// token); a request whose prompt continues a file's history restores it and prefills only the rest.
// HY4_PCACHE=<dir> (default <tree>/pcache, 0 = off), HY4_PCACHE_GB (LRU budget, default 40), HY4_PCACHE_FREE_GB
// (the disk keeps at least this much free, default 15: the Mac had 21 GiB free on 2026-11-19).
struct PrefixCache {
  static constexpr uint32_t PC_MIN = 4096, PC_GAIN = 1024;
  struct Entry { std::string path; std::vector<uint32_t> toks; size_t bytes = 0; uint64_t used = 0; uint32_t floor = 0; };
  std::string dir;
  size_t budget = 0, reserve = 0;
  bool on = false;
  uint64_t clock = 0;
  std::vector<Entry> ents;
  std::mutex pc_mu;
  std::atomic<bool> saving{false};
  std::thread save_th;
  std::atomic<bool> loading{false};
  std::thread load_th;
  hy4::Model::PrefixLoad load_res;
  std::string load_path;
  uint32_t load_rid = 0, load_c0 = 0;
  void wait_save() {
    if (save_th.joinable()) save_th.join();
  }
  void wait_load() {
    if (load_th.joinable()) load_th.join();
  }
  bool start_load(hy4::Model& m, const std::string& path, uint32_t rid, uint32_t c0, bool async) {
    wait_save();
    wait_load();
    load_path = path;
    load_rid = rid;
    load_c0 = c0;
    load_res = hy4::Model::PrefixLoad{};
    auto do_read = [this, &m, path]() {
      m.load_prefix_into(0, path, load_res);
      { std::lock_guard<std::mutex> g(mu); loading.store(false); }
      cv.notify_one();
    };
    if (async) {
      loading.store(true);
      load_th = std::thread(do_read);
      return true;
    }
    do_read();
    finish_load(m);
    return false;
  }
  uint32_t finish_load(hy4::Model& m) {
    wait_load();
    const double ms = load_res.ms;
    const uint32_t got = m.apply_prefix_load(0, std::move(load_res));
    if (got) {
      std::lock_guard<std::mutex> lg(pc_mu);
      for (auto& e : ents) if (e.path == load_path) { e.used = ++clock; break; }
    }
    fprintf(stderr, "hy4: rid %u pcache: %s %u tokens in %.0f ms (slot kept %u)\n", load_rid,
            got ? "restored" : "restore failed", got, ms, load_c0);
    return got;
  }
  static uint32_t common(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, size_t n) {
    return dankattn::proto::common_prefix(a, b, n);
  }
  void init(const hy4::Model& m, const std::string& def) {
    dir = hy4::cfg().pcache_set ? hy4::cfg().pcache_dir : def;
    on = !dir.empty() && dir != "0";
    budget = size_t(hy4::cfg().pcache_gb * 1e9);
    reserve = size_t(hy4::cfg().pcache_free_gb * 1e9);
    if (!on) return;
    size_t tot = 0;
    if (!dankattn::pcache::scan_dir(dir, ".hy4p", ents, clock, tot, [&](const std::string& p, Entry& en) {
          return hy4::Model::prefix_tokens(p, m.fingerprint, en.toks, &en.bytes, &en.floor);
        })) {
      on = false;
      fprintf(stderr, "hy4: pcache: cannot open %s, off\n", dir.c_str());
      return;
    }
    fprintf(stderr, "hy4: pcache %s: %zu files, %.2f GB, budget %.0f GB\n", dir.c_str(), ents.size(), tot / 1e9, budget / 1e9);
  }
  // the file a prompt can restore (the prompt continues its history at least to the file's floor) and the
  // tokens that would be kept; -1 none
  int best(const std::vector<uint32_t>& ids, uint32_t* keep) const {
    int bi = -1;
    *keep = 0;
    for (int i = 0; i < int(ents.size()); i++) {
      const uint32_t c = common(ents[i].toks, ids, ids.size() - 1);
      if (c >= ents[i].floor && c > *keep) { *keep = c; bi = i; }
    }
    return bi;
  }
  // Prepare save of slot 0 (already saved -> false; evicts oldest to make room -> false if no room).
  // When async is true, launches a background writer thread reading slot 0's immutable buffers while
  // slot_busy[0] stays true so active side lanes never stall. Returns true iff an async save was started.
  bool save(hy4::Model& m, const std::string& protect = "", bool async = false) {
    wait_save();
    if (!on) return false;
    auto snap = std::make_shared<hy4::Model::PrefixSnap>(m.snapshot_prefix(0));
    if (snap->pos < PC_MIN) return false;
    std::string path;
    {
      std::lock_guard<std::mutex> lg(pc_mu);
      for (auto& e : ents)
        if (e.toks.size() == snap->pos && common(e.toks, snap->history, snap->pos) == snap->pos) {
          e.used = ++clock;
          return false;
        }
      const size_t need = size_t(snap->pos) * 8600 + (size_t(140) << 20);
      if (!dankattn::pcache::make_room(ents, dir, need, budget, reserve, protect)) {
        fprintf(stderr, "hy4: pcache: no room for %u tokens (%.1f GB free, reserve %.0f GB), not saved\n",
                snap->pos, dankattn::pcache::free_disk_bytes(dir) / 1e9, reserve / 1e9);
        return false;
      }
      const uint64_t h = dankattn::proto::fnv1a_tokens(snap->history.data(), snap->pos);
      char nm[64];
      snprintf(nm, sizeof nm, "/%016llx-%u.hy4p", (unsigned long long)h, snap->pos);
      path = dir + nm;
    }
    auto do_write = [this, &m, snap, path]() {
      Entry en;
      en.path = path;
      double ms = 0;
      const bool ok = m.save_prefix_snap(*snap, en.path, &ms, &en.bytes, &en.floor);
      if (!ok) {
        fprintf(stderr, "hy4: pcache: save of %u tokens failed\n", snap->pos);
      } else {
        en.toks = std::move(snap->history);
        size_t nfiles = 0, tot = 0;
        {
          std::lock_guard<std::mutex> lg(pc_mu);
          en.used = ++clock;
          dankattn::pcache::prune_subprefixes(ents, en);
          ents.push_back(std::move(en));
          nfiles = ents.size();
          tot = dankattn::pcache::total_bytes(ents);
        }
        const size_t fdsk = dankattn::pcache::free_disk_bytes(dir);
        fprintf(stderr, "hy4: pcache: saved %u tokens (%.2f GB) in %.0f ms; %zu files, %.2f GB, %.1f GB free\n",
                snap->pos, en.bytes / 1e9, ms, nfiles, tot / 1e9, fdsk / 1e9);
      }
      { std::lock_guard<std::mutex> g(mu); saving.store(false); }
      cv.notify_one();
    };
    if (async) {
      saving.store(true);
      save_th = std::thread(do_write);
      return true;
    }
    do_write();
    return false;
  }
};

static PrefixCache& pcache() { static PrefixCache p; return p; }

// the lookup indexes outlive requests (one per slot, synced to the slot's history at each request)
static hy4spec::Spec& spec_state() { static hy4spec::Spec s; return s; }

struct Active {
  ReqP q;
  int slot = -1;
  std::chrono::steady_clock::time_point t0;
  std::vector<uint32_t> rest;
  std::vector<float> vis_embeds;
  std::vector<int32_t> vis_idx;       // length == q->ids.size(), -1 for text
  std::vector<hy4::Pos3D> req_pos3d;  // length == q->ids.size() when multimodal
  bool vis_ready = false;
  size_t pref_off = 0;
  uint32_t reused = 0;
  double pms = 0.0;
  uint32_t b = 0, sx = 0, ss = 0;
  bool prefilling = true, spec = false, thinking = false, length = false;
  bool lk_synced = false;
  uint32_t nthink = 0, n = 0;
  hy4spec::Stats st;
  bool is_stop(const hy4::Model& m, uint32_t t) const {
    return t == m.eos || std::find(q->stops.begin(), q->stops.end(), t) != q->stops.end();
  }
};

// Per-slot image fingerprint list [(tok_off, hash)] so two different images of the same resolution
// never falsely hit each other's KV prefix cache on identical <image_pad> token IDs.
static std::vector<std::vector<std::pair<uint32_t, uint64_t>>>& slot_imgs() {
  static std::vector<std::vector<std::pair<uint32_t, uint64_t>>> s;
  return s;
}

static bool emit_tok(hy4::Model& m, Active& a, uint32_t t) {
  put(a.q->rid, t);
  a.n++;
  if (a.thinking) { if (t == a.q->think_end) a.thinking = false; else a.nthink++; }
  if (a.is_stop(m, t)) return true;
  const uint32_t spos = (a.slot == m.cur_slot) ? m.pos : m.slots_[a.slot].pos;
  const uint32_t scap = (a.slot == m.cur_slot) ? m.cap : m.slots_[a.slot].cap;
  if (a.n >= a.q->max_new || spos + 1 >= scap) { a.length = true; return true; }
  return a.q->cancel.load();
}

static void finish_active(Active& a, std::vector<bool>& slot_busy) {
  const float wall = std::chrono::duration<float>(std::chrono::steady_clock::now() - a.t0).count() * 1e3f;
  put(a.q->rid, END);
  const auto& st = a.st;
  uint32_t s6[6] = {a.reused, uint32_t(a.q->ids.size()), a.n, st.rounds, st.accepted, a.length ? 1u : 0u};
  float f2[2] = {float(a.pms), wall};
  fwrite(s6, 4, 6, stdout);
  fwrite(f2, 4, 2, stdout);
  fflush(stdout);
  char sb[256] = "";
  if (a.spec && st.rounds)
    snprintf(sb, sizeof sb, " spec %u rounds, %u accepted, %.1f rows/round, draft %.1f / verify %.1f ms/round, %u lookup-only, lookup %u/%u",
             st.rounds, st.accepted, double(st.rows) / st.rounds, st.draft_ms / st.rounds, st.verify_ms / st.rounds, st.lk_only,
             st.lk_accepted, st.lk_tested);
  fprintf(stderr, "hy4: rid %u slot %d prompt %zu (reused %u, prefill %.0f ms) gen %u in %.0f ms%s\n",
          a.q->rid, a.slot, a.q->ids.size(), a.reused, a.pms, a.n, wall, sb);
  if (a.slot >= 0) slot_busy[a.slot] = false;
  std::lock_guard<std::mutex> g(mu);
  live.erase(a.q->rid);
}

static void apply_sampling(hy4::Model& m, const Req& q) {
  m.temp = q.temp > 0 ? q.temp : 0.0f;
  m.top_p = (q.top_p > 0.0f && q.top_p <= 1.0f) ? q.top_p : 1.0f;
  m.top_k = q.top_k;
  m.min_p = q.min_p >= 0.0f ? q.min_p : m.default_min_p;
  m.seed = q.seed;
}

static int pick_free_slot(hy4::Model& m, const Req& q, const std::vector<bool>& busy) {
  const uint32_t side_min = 256;
  int best = -1;
  uint32_t bc = 0;
  for (int i = 0; i < int(m.slots_.size()); i++) {
    if (busy[i] || q.ids.size() + 1 > m.slots_[i].cap) continue;
    const uint32_t c = m.common_with(i, q.ids);
    if (best < 0 || c > bc) { bc = c; best = i; }
  }
  if (best < 0) return -1;
  if (bc >= side_min) return best;
  // Low prefix overlap: prefer a free side lane that fits so slot 0's big conversation isn't evicted
  int side_best = -1;
  uint32_t side_bc = 0;
  for (int i = 1; i < int(m.slots_.size()); i++) {
    if (busy[i] || q.ids.size() + 1 > m.slots_[i].cap) continue;
    const uint32_t c = m.common_with(i, q.ids);
    if (side_best < 0 || c > side_bc) { side_bc = c; side_best = i; }
  }
  if (side_best >= 0) return side_best;
  return !busy[0] && q.ids.size() + 1 <= m.slots_[0].cap ? 0 : -1;
}

static void admit_requests(hy4::Model& m, std::vector<Active>& active, std::vector<bool>& slot_busy) {
  for (;;) {
    ReqP q;
    {
      std::lock_guard<std::mutex> g(mu);
      if (inbox.empty()) return;
      // Validate front request or find first request that fits a free slot
      bool found = false;
      for (auto it = inbox.begin(); it != inbox.end();) {
        ReqP cand = *it;
        if (cand->cancel.load()) {
          live.erase(cand->rid);
          it = inbox.erase(it);
          continue;
        }
        if (cand->flags & 2) {
          error(cand->rid, "hy4: fanout groups (flags & 2) are not supported");
          live.erase(cand->rid);
          it = inbox.erase(it);
          continue;
        }
        if ((cand->flags & 32) && !cand->imgs.empty() && !m.has_vision) {
          error(cand->rid, "hy4: pack has no vision tower");
          live.erase(cand->rid);
          it = inbox.erase(it);
          continue;
        }
        if (cand->ids.empty()) {
          error(cand->rid, "empty prompt");
          live.erase(cand->rid);
          it = inbox.erase(it);
          continue;
        }
        if (cand->ids.size() + 1 > m.slots_[0].cap) {
          error(cand->rid, "prompt of " + std::to_string(cand->ids.size()) + " tokens exceeds the context capacity of " +
                               std::to_string(m.slots_[0].cap));
          live.erase(cand->rid);
          it = inbox.erase(it);
          continue;
        }
        if (pick_free_slot(m, *cand, slot_busy) >= 0) {
          q = cand;
          inbox.erase(it);
          found = true;
          break;
        }
        ++it;
      }
      if (!found) return;
    }
    const int sl = pick_free_slot(m, *q, slot_busy);
    if (sl < 0) {
      std::lock_guard<std::mutex> g(mu);
      inbox.push_front(q);
      return;
    }
    PrefixCache& pc = pcache();
    if (sl == 0 && pc.on) {
      m.use_slot(0);
      const uint32_t c0 = m.resumable(q->ids);
      uint32_t keep = 0;
      const int e = pc.best(q->ids, &keep);
      const bool restore = e >= 0 && keep > c0 + PrefixCache::PC_GAIN;
      if (restore || m.pos > c0 + PrefixCache::PC_MIN) {
        // When other lanes are active, save slot 0 asynchronously so side lanes never stall;
        // slot_busy[0] stays true until the writer finishes, and q waits at the front of inbox.
        const bool use_async = !active.empty();
        if (pc.save(m, restore ? pc.ents[e].path : "", use_async)) {
          slot_busy[0] = true;
          std::lock_guard<std::mutex> g(mu);
          inbox.push_front(q);
          return;
        }
      }
      if (restore) {
        const int ei = pc.best(q->ids, &keep);
        if (ei >= 0) {
          const bool use_async = !active.empty();
          if (pc.start_load(m, pc.ents[ei].path, q->rid, c0, use_async)) {
            slot_busy[0] = true;
            std::lock_guard<std::mutex> g(mu);
            inbox.push_front(q);
            return;
          }
        }
      }
    }
    m.use_slot(sl);
    m.slots_[sl].has_df_src = false;
    if (slot_imgs().size() < m.slots_.size()) slot_imgs().resize(m.slots_.size());
    Active a;
    a.q = q;
    a.slot = sl;
    a.t0 = std::chrono::steady_clock::now();
    // Clamp prefix reuse at the first image span that differs from what is cached in this slot.
    uint32_t img_reuse_cap = uint32_t(q->ids.size());
    {
      const auto& prev = slot_imgs()[sl];
      const size_t n_cur = q->imgs.size(), n_prv = prev.size();
      const size_t nm = std::min(n_cur, n_prv);
      size_t k = 0;
      while (k < nm && prev[k].first == q->imgs[k].tok_off && prev[k].second == q->imgs[k].hash) k++;
      if (k < n_cur) img_reuse_cap = std::min(img_reuse_cap, q->imgs[k].tok_off);
      if (k < n_prv) img_reuse_cap = std::min(img_reuse_cap, prev[k].first);
    }
    a.reused = m.resume(q->ids);
    if (a.reused > img_reuse_cap) {
      if (m.can_truncate(img_reuse_cap)) {
        m.truncate(img_reuse_cap);
        a.reused = img_reuse_cap;
      } else {
        m.reset();
        a.reused = 0;
      }
    }
    slot_imgs()[sl].clear();
    for (const auto& im : q->imgs) slot_imgs()[sl].push_back({im.tok_off, im.hash});
    a.rest.assign(q->ids.begin() + a.reused, q->ids.end());
    a.spec = (q->flags & 1) && m.has_dfly;
    a.thinking = (q->flags & 4) != 0;
    slot_busy[sl] = true;
    active.push_back(std::move(a));
  }
}

int main(hy4::Model& m) {
  const uint32_t n_side = std::min<uint32_t>(hy4::MAXM - 1, hy4::cfg().side_lanes);
  const uint32_t side_cap = hy4::cfg().side_cap;
  if (side_cap && n_side) {
    for (uint32_t i = 0; i < n_side; i++) m.add_slot(std::min(side_cap, m.slots_[0].cap));
    fprintf(stderr, "hy4: %u side lanes x %u tokens (%zu total slots), state now %.2f GB\n",
            n_side, std::min(side_cap, m.slots_[0].cap), m.slots_.size(), m.C.allocated / 1e9);
    mtl::check_wired_limit(m.mapped_bytes + m.C.allocated, m.C.dev);
  }
  pcache().init(m, mtl::tree_root() + "/pcache");
  std::thread rt(reader);
  put(0, READY);
  fflush(stdout);

  std::vector<Active> active;
  std::vector<bool> slot_busy(m.slots_.size(), false);
  std::vector<uint32_t> out;

  for (;;) {
    // Release slot 0 if an async prefix-cache save or load finished (and no Active request owns slot 0)
    if (slot_busy[0] && !pcache().saving.load() && !pcache().loading.load()) {
      bool owned = false;
      for (const auto& a : active) if (a.slot == 0) { owned = true; break; }
      if (!owned) {
        if (pcache().save_th.joinable()) pcache().wait_save();
        if (pcache().load_th.joinable()) pcache().finish_load(m);
        slot_busy[0] = false;
      }
    }
    admit_requests(m, active, slot_busy);
    if (active.empty()) {
      std::unique_lock<std::mutex> g(mu);
      cv.wait(g, [&] { return closed || (!inbox.empty() && !pcache().saving.load() && !pcache().loading.load()); });
      if (inbox.empty() && closed) break;
      continue;
    }
    @autoreleasepool {
      // 1) Drop any cancelled active requests before spending GPU work
      for (size_t i = 0; i < active.size();) {
        if (active[i].q->cancel.load()) {
          finish_active(active[i], slot_busy);
          active.erase(active.begin() + long(i));
        } else i++;
      }
      if (active.empty()) continue;

      // 2) Prefill one chunk for the first prefilling request (smaller chunk when decode lanes are active to bound ITL)
      bool any_decoding = false;
      for (const auto& a : active) if (!a.prefilling) { any_decoding = true; break; }
      const uint32_t pf_chunk = any_decoding
          ? std::min(m.MR, std::max<uint32_t>(hy4::MAXM + 1, (hy4::cfg().pf_dec_chunk / 16) * 16))
          : m.MR;
      for (size_t i = 0; i < active.size(); i++) {
        if (!active[i].prefilling) continue;
        Active& a = active[i];
        m.use_slot(a.slot);
        apply_sampling(m, *a.q);
        // Prepare 3D MRoPE positions & encode any uncached images once before the first prefill chunk.
        if (!a.vis_ready && !a.q->imgs.empty()) {
          const size_t L = a.q->ids.size();
          a.req_pos3d.resize(L);
          a.vis_idx.assign(L, -1);
          uint32_t cur_pos = 0;
          size_t tok_i = 0;
          for (const auto& im : a.q->imgs) {
            while (tok_i < im.tok_off && tok_i < L) {
              a.req_pos3d[tok_i++] = hy4::Pos3D{cur_pos, cur_pos, cur_pos};
              cur_pos++;
            }
            const uint32_t hm = im.h / m.v_merge, wm = im.w / m.v_merge, cols = wm + 1u;
            const uint32_t step = std::max(hm, cols);
            const uint32_t n_toks = im.T * hm * cols;
            const bool need_enc = (im.tok_off + n_toks > a.reused);
            int32_t base_vi = -1;
            if (need_enc) {
              double vms = 0;
              auto emb = m.encode_vision(im.pixels.data(), im.T, im.h, im.w, &vms);
              a.pms += vms;
              base_vi = int32_t(a.vis_embeds.size() / m.H);
              a.vis_embeds.insert(a.vis_embeds.end(), emb.begin(), emb.end());
            }
            for (uint32_t t = 0; t < im.T; t++) {
              for (uint32_t r = 0; r < hm; r++) {
                for (uint32_t c0 = 0; c0 < cols; c0++) {
                  if (tok_i < L) {
                    a.req_pos3d[tok_i] = hy4::Pos3D{cur_pos + t * step, cur_pos + t * step + r, cur_pos + t * step + c0};
                    if (need_enc) a.vis_idx[tok_i] = base_vi + int32_t((t * hm + r) * cols + c0);
                    tok_i++;
                  }
                }
              }
            }
            cur_pos += im.T * step;
          }
          while (tok_i < L) {
            a.req_pos3d[tok_i++] = hy4::Pos3D{cur_pos, cur_pos, cur_pos};
            cur_pos++;
          }
          a.vis_ready = true;
        }
        const size_t end = std::min(a.rest.size(), a.pref_off + pf_chunk);
        std::vector<uint32_t> c(a.rest.begin() + long(a.pref_off), a.rest.begin() + long(end));
        if (end < a.rest.size())
          m.next_hint.assign(a.rest.begin() + long(end), a.rest.begin() + long(std::min(a.rest.size(), end + pf_chunk)));
        double g = 0;
        const size_t g_off = a.reused + a.pref_off;
        if (!a.req_pos3d.empty()) {
          const uint32_t n_vis = uint32_t(a.vis_embeds.size() / m.H);
          a.b = m.step(c, &g, nullptr, nullptr, n_vis ? a.vis_embeds.data() : nullptr, n_vis,
                       n_vis ? a.vis_idx.data() + g_off : nullptr, a.req_pos3d.data() + g_off).back();
        } else {
          a.b = m.step(c, &g).back();
        }
        a.pms += g;
        a.pref_off = end;
        if (a.pref_off >= a.rest.size()) {
          a.prefilling = false;
          a.rest.clear();
          a.vis_embeds.clear();
          a.vis_idx.clear();
          a.req_pos3d.clear();
          m.save_dfly_src(a.slot, c.size() > hy4::MAXM ? 0 : uint32_t(c.size()) - 1);
          m.save_ckpt();
        }
        break;  // one chunk per loop iteration so arriving requests / active decodes stay responsive
      }

      // 3) Count ready-to-decode requests
      std::vector<size_t> dec_idx;
      for (size_t i = 0; i < active.size(); i++)
        if (!active[i].prefilling) dec_idx.push_back(i);
      if (dec_idx.empty()) continue;

      // Handle thinking-budget exhaustion on any decoding request first
      bool did_force = false;
      for (size_t idx : dec_idx) {
        Active& a = active[idx];
        if (a.thinking && a.nthink >= a.q->budget && !a.q->force.empty()) {
          a.thinking = false;
          bool stop = false;
          for (auto t : a.q->force) if ((stop = emit_tok(m, a, t))) break;
          fflush(stdout);
          if (stop) {
            a.q->cancel = true;
          } else {
            m.use_slot(a.slot);
            apply_sampling(m, *a.q);
            a.b = feed(m, a.q->force, a.sx, a.ss, nullptr, &a.q->cancel);
            if (!a.q->cancel.load()) m.save_dfly_src(a.slot, a.sx);
          }
          did_force = true;
        }
      }
      if (did_force) continue;

      // 4a) Single-request fast path: full DFly2 + context-lookup speculation (identical to single-slot mode)
      if (dec_idx.size() == 1) {
        Active& a = active[dec_idx[0]];
        m.use_slot(a.slot);
        apply_sampling(m, *a.q);
        hy4spec::Spec& sp = spec_state();
        bool stop = false;
        if (!a.spec || !m.slots_[a.slot].has_df_src || m.pos + sp.max_rows(m) + 1 >= m.cap) {
          stop = emit_tok(m, a, a.b);
          if (!stop) {
            a.b = m.step({a.b}).back();
            m.save_dfly_src(a.slot, 0);
          }
        } else {
          // Emit the anchor immediately (before draft+verify) to cut TTFT and match multi-lane (4b)
          stop = emit_tok(m, a, a.b);
          fflush(stdout);
          if (!stop) {
            if (!a.lk_synced) { sp.begin(m); a.lk_synced = true; }
            sp.st = a.st;
            out.clear();
            uint32_t sx = ~0u, ss = ~0u;
            sp.round(m, a.b, sx, ss, out, [&](uint32_t t) { return a.is_stop(m, t); });
            m.save_dfly_src(a.slot, sx);
            a.st = sp.st;
            // out[0] is the anchor already emitted above
            for (size_t i = 1; i < out.size() && !stop; i++) stop = emit_tok(m, a, out[i]);
          }
        }
        fflush(stdout);
        if (stop) {
          finish_active(a, slot_busy);
          active.erase(active.begin() + long(dec_idx[0]));
        }
        continue;
      }

      // 4b) 2..MAXM concurrent decoding requests: each anchor goes out first (a request that ends on it leaves
      // the step), then one joint round over the rest: per-lane drafts, one step_lanes() verify, per-lane acceptance
      const bool js_on = hy4::cfg().js;
      const uint32_t js_rows = hy4::cfg().js_rows;
      const uint32_t js_lanes = hy4::cfg().js_lanes;
      std::vector<size_t> step_idx;
      std::vector<bool> done(active.size(), false);
      for (size_t idx : dec_idx) {
        Active& a = active[idx];
        if (emit_tok(m, a, a.b)) {
          done[idx] = true;
        } else {
          step_idx.push_back(idx);
        }
      }
      fflush(stdout);
      if (!step_idx.empty()) {
        hy4spec::Spec& sp = spec_state();
        const bool joint = js_on && m.has_dfly && step_idx.size() <= js_lanes;
        std::vector<hy4spec::Lane> ln(step_idx.size());
        std::vector<hy4spec::Lane*> lp;
        for (size_t j = 0; j < step_idx.size(); j++) {
          Active& a = active[step_idx[j]];
          hy4spec::Lane& l = ln[j];
          l.slot = a.slot;
          l.b = a.b; l.sx = ~0u; l.ss = ~0u;
          l.src = m.slots_[a.slot].has_df_src;
          l.spec = joint && a.spec;
          l.samp.temp = a.q->temp > 0 ? a.q->temp : 0.0f;
          l.samp.min_p = a.q->min_p >= 0.0f ? a.q->min_p : m.default_min_p;
          l.samp.top_p = (a.q->top_p > 0.0f && a.q->top_p <= 1.0f) ? a.q->top_p : 1.0f;
          l.samp.top_k = a.q->top_k;
          l.samp.seed = a.q->seed;
          l.is_stop = [&m, &a](uint32_t t) { return a.is_stop(m, t); };
          l.st = a.st;
          if (l.spec && !a.lk_synced) { m.use_slot(a.slot); sp.begin(m); a.lk_synced = true; }
          lp.push_back(&l);
        }
        sp.joint(m, lp, js_rows);
        for (size_t j = 0; j < step_idx.size(); j++) {
          Active& a = active[step_idx[j]];
          const hy4spec::Lane& l = ln[j];
          a.b = l.b;
          m.save_dfly_src(a.slot, l.sx);
          a.st = l.st;
          // out[0] is the anchor emitted above
          bool stop = false;
          for (size_t i = 1; i < l.out.size() && !stop; i++) stop = emit_tok(m, a, l.out[i]);
          if (stop) done[step_idx[j]] = true;
        }
        fflush(stdout);
      }
      for (size_t i = active.size(); i-- > 0;) {
        if (done[i]) {
          finish_active(active[i], slot_busy);
          active.erase(active.begin() + long(i));
        }
      }
    }
  }
  rt.join();
  // a clean stop (stdin closed) keeps the main conversation for the next start
  pcache().wait_save();
  pcache().wait_load();
  m.use_slot(0);
  pcache().save(m);
  return 0;
}

}  // namespace serve
