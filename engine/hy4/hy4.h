// DankAttn (Hy4): Hy4-flash (hy_v4_flash text tower) on a DankAttn hy4 pack, M <= 16 rows per step.
// Shares the register-direct matmuls (kernels/mm2.metal) and the Metal runtime (mtl.h) with the
// Flash-Next engine; everything model-specific is here and in kernels/hy4.metal.
#pragma once
#include "../core/kv_radix.h"
#include "../core/pack.h"
#include "../mtl.h"
#include "config.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <future>
#include <atomic>
#include <memory>
#include <tuple>

namespace hy4 {

constexpr uint32_t MAXM = 16;     // rows of a decode / verify step (mm2); more rows go through mmp / mmpg

// Fan-out-aware expected-tok/ms draft length selector (exactness-preserving: only trims k in 0..max_k).
// Models Hy4's 512-expert top-16 MoE verify cost T(M) = t0 + c_moe * (1 - (496/512)^M) + t_draft
// fitted to M5 Ultra 12k measurements (1/5/8/16 rows: 19.08/27.61/33.26/44.76 ms, draft 3.22 ms).
// Set HY4_DRAFT_ADAPT=0 to keep the legacy cumulative draft_cut length max_k.
inline size_t select_draft_k(const std::vector<float>& probs, size_t max_k, uint32_t active_lanes = 1) {
  const size_t lim = std::min(max_k, probs.size());
  if (!cfg().draft_adapt || lim == 0) return lim;
  constexpr double kMiss = 496.0 / 512.0;  // 31/32
  const double t0 = 16.9;
  const double c_moe = 70.0 / (active_lanes > 1 ? (1.0 + 0.35 * double(active_lanes - 1)) : 1.0);
  const double t_draft = 3.22 / double(active_lanes > 0 ? active_lanes : 1);
  const double p_scale = cfg().draft_pscale;
  double best_rate = 1.0 / (t_draft + t0 + c_moe * (1.0 - kMiss));
  size_t best_k = 0;
  double cum_p = 1.0, exp_tok = 1.0, miss = kMiss;
  for (size_t k = 0; k < lim; k++) {
    const double pk = std::min(1.0, std::max(0.0, double(probs[k]) * p_scale));
    cum_p *= pk;
    exp_tok += cum_p;
    miss *= kMiss;
    const double rate = exp_tok / (t_draft + t0 + c_moe * (1.0 - miss));
    if (rate > best_rate) {
      best_rate = rate;
      best_k = k + 1;
    }
  }
  return best_k;
}
constexpr uint32_t RING = 4096;   // sliding-window latent ring (window 512 + a prefill chunk + rollback slack)
constexpr uint32_t SELMAX = 2048; // tokens a sparse layer attends past the dense range
constexpr uint32_t TOKR = 4096;   // indexer token-key ring (positions); kernels/hy4.metal TOKR
constexpr uint32_t ASPLITS = 32;  // attention key splits per row (fixed, so a row has the same bits at every M)
constexpr uint32_t HC_NP = 12;    // channel slices of the two-pass hyper-connection read-in
constexpr uint32_t SB = 64;       // indexer score / top-k rows per block (scratch is SB x cap/4 floats)

using dankattn::Lin;
using dankattn::Raw;

struct Layer {
  bool sparse = false, dense = false, eng = false;
  bool kv8 = false;  // latent cache rows int8 + f16 scales (640 B) instead of bf16 (1152 B); sparse layers
  uint32_t nh = 0;
  Raw ahc_fn, ahc_base, ahc_scale, mhc_fn, mhc_base, mhc_scale, attn_norm, ffn_norm;
  Lin in, gate, qb, kb, vb, o;
  Raw qa_norm, kva_norm, sinks, post_norm;
  Raw idx_k_norm, idx_kvp, idx_kgp, idx_ape, idx_key_norm;
  Lin mlp_gu, mlp_dn;
  Raw mlp_post;
  Lin moe_in, moe_gu, moe_dn, moe_out;
  Raw moe_bias, moe_norm;
  Lin eng_kv;
  Raw eng_bias, eng_lnorm, eng_nk, eng_nq, eng_gamma, eng_hcs;
  // Engram hashing (host)
  std::vector<int32_t> eng_map;
  uint64_t mult[3] = {0, 0, 0};
  uint64_t prime[16], off[16];
  int64_t eng_pad = 0;
  uint32_t eng_slot = 0;  // which of the step's engram row blocks
  // state
  id<MTLBuffer> lat;      // latent rows (576 bf16): sparse [cap], sliding [RING]
  id<MTLBuffer> tokring;  // sparse: indexer token keys, TOKR positions x 128 f32
  id<MTLBuffer> bk;       // sparse: pooled group keys [cap / 4][128] f32
  id<MTLBuffer> lsel, lnsel;  // DFly2 target layers: this layer's own selection (else the shared one)
};

struct DLayer {  // one DFly2 draft layer (dense or LatentMoE MLA, attends to its target layer's latent cache)
  uint32_t nh = 0, tl = 0;
  bool moe = false;
  Lin in, gate, qb, kb, vb, o, gu, dn;
  Lin moe_in, moe_gu, moe_dn, moe_out;
  Raw qa_norm, kva_norm, sinks, post_norm, in_norm, post_ln, moe_bias, moe_norm;
};

struct Pos3D {
  uint32_t t = 0, h = 0, w = 0;
};

struct VLayer {  // one SigLIP2-adapted ViT block (H=1152, nh=16, hd=72, I=4304->4352)
  Raw ln1_w, ln1_b, ln2_w, ln2_b;
  Lin qkv, o, fc1, fc2;
  Raw qkv_b, o_b, fc1_b, fc2_b;
};

struct Rec;

struct Model {
  mtl::Ctx C;
  std::string dir;
  uint32_t H = 0, HC = 4, nL = 0, vocab = 0, E = 0, topk = 0, I = 0, LAT = 0;
  uint32_t window = 512, ratio = 4, gtopk = 512, eos = 0, pad_id = 0;
  float eps = 1e-5f, hc_eps = 1e-6f, hc_mag = 2.0f, rscale = 1.0f, swiglu = 10.0f;
  double base_swa = 1e4, base_dsa = 1e7;
  double inv_swa[32] = {}, inv_dsa[32] = {};
  std::vector<id<MTLBuffer>> files;
  size_t mapped_bytes = 0;
  std::vector<Layer> L;
  Raw embed, hch_fn, hch_base, hch_scale, norm;
  Lin lm_head;
  // Engram table (fp8 or u4_g32 rows of 96), mapped from the pack or a GGUF
  const uint8_t* eng_tab = nullptr;
  size_t eng_rows = 0;
  uint32_t eng_dim = 96, eng_row_bytes = 96, n_eng = 0, moe_dn_had = 0;
  bool eng_q4 = false;
  int eng_fd = -1, eng_fd_pf = -1;
  size_t eng_off = 0;  // byte offset of row 0 in the table file
  // Pages of the table this process read lately (2 s epoch of the last read per 16 KB page, 0 never): in decode
  // and verify steps their rows are copied from the mapping on the calling thread (~0.01 us a row) instead of a
  // pread through the I/O pool (~0.5 us warm plus the pool's wake-ups, measured 2026-11-19); only the rest go
  // through the pool. Pages older than ~8 s may have been evicted and go through pread. HY4_ENG_WARM=0 off.
  std::unique_ptr<std::atomic<uint8_t>[]> eng_seen;
  bool eng_warm = cfg().eng_warm;
  // Userspace lock-free direct-mapped Engram row cache (seqlock per slot, 0 = disabled via HY4_ENG_CACHE_MB=0).
  // Stride is 68 B for u4_g32 (seq + row + 60 B) or 104 B for fp8 (seq + row + 96 B).
  std::unique_ptr<uint8_t[]> eng_cache_buf;
  size_t eng_cache_mask = 0;    // (power-of-2 slots) - 1; 0 when disabled
  size_t eng_cache_stride = 0;
  bool eng_cache_get(uint64_t row, uint8_t* dst) const;
  void eng_cache_put(uint64_t row, const uint8_t* src);
  // Prefill: the next chunk's rows are fetched while the GPU runs this one. A caller that knows the next
  // chunk sets next_hint before step(); the step after uses the prefetched rows if its tokens match.
  std::vector<uint32_t> next_hint;
  // the last step()'s host timeline (ms): entry -> first commit, entry -> last commit, first commit -> GPU start,
  // GPU span, GPU end -> host wake (spec trace: which side a slow round's host time sits on)
  struct StepLat { double first_commit = 0, last_commit = 0, queue = 0, gpu = 0, wake = 0; } last_lat, last_dlat;  // step(), draft()
  double eng_wait_ms = 0, eng_rest_ms = 0;  // host time spent in engram_rows (the GPU runs layer 0 meanwhile), cumulative
  std::future<void> pf_job;
  std::vector<uint32_t> pf_toks;
  size_t pf_p0 = 0;
  id<MTLBuffer> eng_rows_pf, eng_live_pf;
  // Overlap the anchor token's cold Engram fetch with DFly2 drafting (reuses eng_rows_pf / eng_live_pf).
  std::future<void> anchor_job;
  uint32_t anchor_tok = 0;
  size_t anchor_p0 = ~size_t(0);
  void prefetch_anchor(uint32_t tok);
  struct AnchorLaneReq { int slot = 0; uint32_t tok = 0; };
  std::future<void> anchor_lanes_job;
  std::vector<std::tuple<int, size_t, uint32_t>> anchor_lanes_meta;  // (slot, pos, tok)
  void prefetch_anchors_lanes(const std::vector<AnchorLaneReq>& reqs);

  // DFly2 drafter
  bool has_dfly = false, df_moe = false, df_has_ac = false, df_ac_gated = false;
  std::vector<DLayer> D;
  Lin df_eh, df_hcgu, df_hcdn, lm_head_w4;
  Lin df_ac_in, df_ac_qb, df_ac_kb, df_ac_vb, df_ac_o, df_ac_bgate;
  Raw df_enorm, df_hnorm, df_gate, df_hchn, df_hcen, df_norm;
  Raw df_ac_qa_norm, df_ac_kva_norm, df_ac_sinks, df_ac_bsink, df_ac_brel, df_ac_bgbias;
  uint32_t dblock = 6, mask_id = 0, dff = 0, dhci = 0;
  uint32_t d_E = 0, d_topk = 0, d_I = 0, d_LAT = 0, df_ac_src = 0, df_ac_nh = 32;
  float d_rscale = 1.0f;
  static constexpr uint32_t DSELMAX = 2176;
  id<MTLBuffer> dx, dh, de, dfrag2, dsel, dnsel, rope_draft, dids, dout, dprob, dpmax, dmarg;
  id<MTLBuffer> ac_blk, ac_bg, ac_nh_gate, dl_ac_rope, dl_ac_sel_b, dl_ac_rlane;
  id<MTLBuffer> dl_eh_b, dl_row0_b, dl_maxd_b, dl_anchor_b, dl_dsel_b, dl_samp_b;
  std::vector<float> last_probs;  // the last draft's per-token probabilities (draft head)
  std::vector<float> last_pmax;   // and each draft distribution's top probability
  std::vector<float> last_marg;   // and the chosen token's (Gumbel-perturbed when sampling) score margin over the runner-up
  // GPU-side draft cut: dgate[s] holds 1 while draft s can still pass draft_cut (dgate[0] is always 1).
  std::vector<id<MTLBuffer>> dgate;
  uint32_t as_draft = cfg().as_draft;
  float draft_cut = cfg().draft_cut;
  // Draft dblock - 1 tokens after `anchor` (the token at position pos, not yet fed): the target's
  // last step left the source row's final hidden at xf row src_x and its selections at row src_s.
  std::vector<uint32_t> draft(uint32_t anchor, uint32_t src_x, uint32_t src_s, double* gpu_ms = nullptr,
                              std::map<std::string, double>* profile = nullptr, bool cut = true);
  // Batched DFly2 drafting across concurrent lanes (total rows sum(max_drafts + 1) <= MAXM).
  struct DraftReq {
    int slot = 0;
    uint32_t anchor = 0, src_x = 0, src_s = 0, max_drafts = 0;
    float temp = 0.0f, min_p = 0.0f, top_p = 1.0f;
    uint32_t top_k = 0;
    uint64_t seed = 0;
  };
  struct DraftRes {
    std::vector<uint32_t> toks;
    std::vector<float> probs, pmax, marg;
  };
  std::vector<DraftRes> draft_lanes(const std::vector<DraftReq>& reqs, double* gpu_ms = nullptr, bool cut = true);

  // Vision tower (optional pack overlay: man["vision"], ~500 MB W8g128, 0 persistent scratch)
  bool has_vision = false;
  uint32_t v_nL = 0, v_H = 0, v_I = 0, v_Ipad = 0, v_nh = 0, v_hd = 0, v_ps = 16, v_merge = 2, v_edge = 128;
  float v_eps = 1e-5f;
  uint32_t img_start_id = 120118, img_end_id = 120119, img_tok_id = 120120, img_nl_id = 120121;
  uint32_t vid_start_id = 120122, vid_end_id = 120123, vid_tok_id = 120124;
  Lin v_patch_w, v_merge_fc1, v_merge_fc2;
  Raw v_patch_b, v_pos_embed, v_merge_norm, v_merge_fc1_b, v_merge_fc2_b, v_merge_nl;
  std::vector<VLayer> VL;
  id<MTLBuffer> vis_idx_b = nil;
  // Encode one image/video (pixels_bf16: [T*h*w, 3*ps*ps]) -> float32 [T*(h/2)*(w/2+1), H].
  std::vector<float> encode_vision(const uint16_t* pixels_bf16, uint32_t T, uint32_t h, uint32_t w,
                                   double* gpu_ms = nullptr);

  // pipelines (mm2/mmp/mmpg compiled lazily for only the (bits, gs) pairs present in this pack)
  std::map<std::string, id<MTLComputePipelineState>> P;
  id<MTLComputePipelineState> pipe(const std::string& k) const;
  // scratch (phase-aliased intermediates live in scratch_heap)
  id<MTLHeap> scratch_heap = nil;
  id<MTLResidencySet> slot_rs = nil;
  std::vector<id<MTLBuffer>> lane_tab;  // per layer: LaneState[slots_.size()] GPU address table
  id<MTLBuffer> row_lane_b, row_pos_b, samp_rows_b;
  bool lane_pipes_ready = false;
  id<MTLBuffer> ids, h, y, post_a, post_f, xn_frag, xf, Yin, Yg, Yq, Yo, qa_frag, qn_frag, qr, iq, iw, qlat, part,
      ofrag, yv, gfrag, Ym, topi, topw, lat_frag, mo_frag, slots, nslots, slot_of, Ygu, act, yd, Ymlp, mlp_frag,
      eng_rows_b, eng_frag, Ye, eng_live, logits, amax, rope_swa, rope_dsa, rope_pool, scores, mark, sel, nsel, mmpart,
      counters, head_slots, nslots32, nslots64, hcpart, roff, rcnt, rlist, cidx, rmask, wtile, moe_ind;
  uint32_t MR = 1024;
  uint32_t gch_ = cfg().gch;  // prefill grouped matmul: simdgroups per threadgroup
  uint32_t mmp_nf_ = 2u;
  uint32_t grpmmp_nf_ = gch_ >= 2 ? 2u : 1u;
  uint32_t selmax = SELMAX;  // selection list stride (HY4_DENSE: the capacity)
  bool kv8 = cfg().kv8;  // sparse layers' latent caches in int8 (default on)
  bool paged_kv = cfg().paged_kv;  // shared 128-token paged KV block pool + radix prefix tree
  dankattn::KvRadixPool kv_pool;
  void ensure_kv_blocks(int slot, uint32_t need_pos);
  void commit_kv_blocks(int slot);
  void release_kv_blocks(int slot);
  size_t lat_row(const Layer& l) const { return l.kv8 ? 640 : 1152; }
  const Raw* ynorm_pending = nullptr;
  uint32_t cap = 0, pos = 0;
  // sampling: temp <= 0 greedy; else Gumbel-max keyed by (seed, position); drafts draw with the same keys
  float temp = 0.0f;
  // min_p / top_p / top_k truncation of sampled draws; default_min_p comes from HY4_MIN_P (0.02)
  float default_min_p = cfg().min_p;
  float min_p = default_min_p;
  float top_p = 1.0f;
  uint32_t top_k = 0;
  uint64_t seed = 0;
  std::vector<uint32_t> history;
  std::vector<Pos3D> pos3d;
  int32_t mrope_delta = 0;

  Model(const std::string& pack, uint32_t capacity);
  // Feed toks at positions pos..; returns argmax per row; logits_out gets [M][vocab] when given.
  // Optional multimodal inputs: vis_embeds [n_vis][H], vis_idx [M] (-1 for text, else row in vis_embeds),
  // pos3d_in [M] (3D interleaved MRoPE coords; nullptr uses 1D pos + mrope_delta).
  std::vector<uint32_t> step(const std::vector<uint32_t>& toks, double* gpu_ms = nullptr,
                             std::vector<float>* logits_out = nullptr,
                             std::map<std::string, double>* profile = nullptr,
                             const float* vis_embeds = nullptr, uint32_t n_vis = 0,
                             const int32_t* vis_idx = nullptr,
                             const Pos3D* pos3d_in = nullptr);
  // Multi-lane batched decode / joint-verify step (total rows across batches <= MAXM).
  // Each batch appends toks to its slot's history/pos and returns one sampled/argmax token per row.
  struct LaneBatch {
    int slot = 0;
    std::vector<uint32_t> toks;
    float temp = 0.0f, min_p = 0.0f, top_p = 1.0f;
    uint32_t top_k = 0;
    uint64_t seed = 0;
  };
  std::vector<uint32_t> step_lanes(const std::vector<LaneBatch>& batches, double* gpu_ms = nullptr,
                                   std::vector<float>* logits_out = nullptr);
  void reset() {
    if (anchor_job.valid()) anchor_job.wait();
    if (anchor_lanes_job.valid()) anchor_lanes_job.wait();
    anchor_p0 = ~size_t(0);
    anchor_lanes_meta.clear();
    pos = 0; history.clear(); pos3d.clear(); mrope_delta = 0; ckpts.clear(); ring_floor = 0;
    if (paged_kv && cur_slot >= 0 && cur_slot < int(slots_.size())) release_kv_blocks(cur_slot);
    if (cur_slot >= 0 && cur_slot < int(slots_.size())) slots_[cur_slot].has_df_src = false;
  }
  // drop the newest tokens (keep the first n): valid while the sliding rings still hold n - window and the
  // indexer ring the tokens of n's group, and not below ring_floor: after a checkpoint or prefix-file restore
  // the rings hold only the rows that came with it (before 2026-11-19 a truncation below a restored
  // checkpoint read stale ring rows)
  void truncate(uint32_t n);
  uint32_t ring_floor = 0;
  bool can_truncate(uint32_t n) const { return n <= pos && pos - n <= RING - window - MR && n >= ring_floor; }
  // the tokens resume(prompt) would keep, without changing anything
  uint32_t resumable(const std::vector<uint32_t>& prompt) const;
  // Checkpoints: what positions < pos depend on besides the sparse layers' full caches: each sliding
  // layer's last `window` latent rows and each sparse layer's last 4 indexer token keys.
  struct Ckpt { uint32_t pos; id<MTLBuffer> b; };
  std::vector<Ckpt> ckpts;
  size_t max_ckpts = 8;
  void save_ckpt();
  // Make the cache hold the longest prefix of `prompt` it can (at most prompt.size() - 1 tokens, so the
  // last one is always fed): truncate, else restore the newest usable checkpoint, else start over.
  // Returns the tokens kept.
  uint32_t resume(const std::vector<uint32_t>& prompt);
  // Conversation slots: every per-conversation buffer (latent caches and rings, indexer keys) plus the
  // position, history and checkpoints. Weights and step scratch are shared; switching swaps pointers.
  struct Slot {
    uint32_t cap = 0, pos = 0, ring_floor = 0;
    int32_t mrope_delta = 0;
    std::vector<uint32_t> history;
    std::vector<Pos3D> pos3d;
    std::vector<Ckpt> ckpts;
    std::vector<id<MTLBuffer>> lat, tokring, bk;
    std::vector<uint32_t> kv_blocks;  // physical block IDs in kv_pool when paged_kv is enabled
    id<MTLBuffer> kv_tab = nil;       // [cap / 128] uint32 page table on GPU when paged_kv is enabled
    // Saved DFly2 source state (~52 KB/slot): last accepted xf row + target layers' lsel/lnsel rows,
    // so interleaved prefill chunks or other slots' steps never invalidate this slot's drafter source.
    id<MTLBuffer> df_xf = nil;
    std::vector<id<MTLBuffer>> df_lsel, df_lnsel;
    bool has_df_src = false;
  };
  std::vector<Slot> slots_;
  int cur_slot = 0;
  int add_slot(uint32_t capacity);  // returns its index; slot 0 is the one the constructor built
  void save_dfly_src(int slot, uint32_t row);
  // Prefix cache files (serve's SSD cache). save_prefix writes the current slot's state at pos: header, history,
  // a checkpoint blob (sliding rings' last window rows, indexer tokens' last 4), then each sparse layer's
  // latent rows [0, pos) and pooled keys [0, ceil(pos / 4)). load_prefix puts a file's state into the current
  // slot and returns its length (0: the file is not for this pack and layout, or does not fit the slot).
  // fingerprint: the manifest's bytes and the state layout; bump PREFIX_LAYOUT when the cache layout changes.
  static constexpr uint32_t PREFIX_LAYOUT = 2;
  uint64_t fingerprint = 0;
  struct PrefixSnap {
    uint32_t pos = 0, ring_floor = 0;
    std::vector<uint32_t> history;
    std::vector<id<MTLBuffer>> lat, tokring, bk;
  };
  PrefixSnap snapshot_prefix(int slot) const;
  bool save_prefix_snap(const PrefixSnap& snap, const std::string& path, double* ms = nullptr,
                        size_t* out_bytes = nullptr, uint32_t* out_floor = nullptr) const;
  bool save_prefix(const std::string& path, double* ms = nullptr, size_t* out_bytes = nullptr,
                   uint32_t* out_floor = nullptr);  // the rings' rows go along as far back as a live slot could truncate (window + 2,560)
  struct PrefixLoad {
    uint32_t pos = 0, ring_floor = 0;
    std::vector<uint32_t> history;
    double ms = 0.0;
    bool ok = false;
  };
  bool load_prefix_into(int slot, const std::string& path, PrefixLoad& out) const;
  uint32_t apply_prefix_load(int slot, PrefixLoad&& pl);
  uint32_t load_prefix(const std::string& path, double* ms = nullptr);
  // a file's history (header check only) and the lowest length it can be truncated to after loading; false
  // when it is not a cache file of this fingerprint
  static bool prefix_tokens(const std::string& path, uint64_t fp, std::vector<uint32_t>& toks, size_t* bytes, uint32_t* floor);
  void use_slot(int i);
  // tokens of `prompt` slot i would reuse (common prefix with its history)
  uint32_t common_with(int i, const std::vector<uint32_t>& prompt) const;

 private:
  void mm(Rec& rc, const char* tag, const Lin& w, id<MTLBuffer> X, id<MTLBuffer> Y, uint32_t M,
          id<MTLBuffer> slots = nil, id<MTLBuffer> nslots = nil, uint32_t nz = 0, uint32_t xstride = 0,
          uint32_t ystride = 0, bool epi3 = false, id<MTLBuffer> e0 = nil, uint32_t epiR = 0, uint32_t ldx = 1,
          bool rm = false, bool barrier = true, id<MTLBuffer> e1 = nil);
  void engram_rows(uint32_t M, const uint32_t* toks);
  // row ids of tokens ctx[p0 .. p0 + M) (ctx holds every earlier token too), fetched into dst / live
  // engram layer slots [s0, s1) only (live may be null: rows' pad flags already written)
  void engram_read_rows(const std::vector<uint64_t>& rows, uint32_t M, uint8_t* dst, uint32_t dst_r0, bool allow_warm,
                        const uint32_t* dst_rows = nullptr);
  void engram_fetch(const std::vector<uint32_t>& ctx, size_t p0, uint32_t M, uint8_t* dst, uint32_t* live,
                    uint32_t s0 = 0, uint32_t s1 = ~0u, uint32_t dst_r0 = 0);
  std::future<void> eng_rest;
  void ensure_lane_pipes();
  void rebuild_lane_tables();
  void engram_fetch_lanes(const std::vector<const std::vector<uint32_t>*>& rhist, const std::vector<size_t>& rpos,
                          uint8_t* dst, uint32_t* live, uint32_t s0 = 0, uint32_t s1 = ~0u,
                          const uint32_t* dst_rows = nullptr);
  void hc_read(Rec& rc, const Raw& fn, const Raw& base, const Raw& scale, const Raw& nw, id<MTLBuffer> postp,
               id<MTLBuffer> post, uint32_t inject, uint32_t mode, uint32_t M, const char* tag);
  void encode_layer(Rec& rc, Layer& l, uint32_t li, uint32_t M, uint32_t p0, bool first, bool lanes = false,
                    uint32_t max_pos = 0);
  void restore_rings(const Ckpt& c);  // a checkpoint's ring rows back into the current slot's rings
};

}  // namespace hy4
