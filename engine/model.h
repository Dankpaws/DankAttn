// DankAttn: Qwen3.8-Flash-Next (qwen4_exp) on a DankAttn pack, M <= 16 rows per step.
#pragma once
#include "mtl.h"
#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <map>
#include <array>
#include <dispatch/dispatch.h>
#include <memory>
#include <unordered_map>

namespace opus {

#ifndef OPUS_MAXM
#define OPUS_MAXM 16
#endif
constexpr uint32_t MAXM = OPUS_MAXM;   // decode / verify rows per step
constexpr uint32_t ASPLIT = 16; // attention key splits per row (SMAX in attn.metal)
#ifndef OPUS_MP
#define OPUS_MP 2048
#endif
constexpr uint32_t MP = OPUS_MP;  // prefill rows per step
constexpr int NSLOT = 16;       // per-call RoPE / token slots inside one command buffer
constexpr uint32_t MAXD = 7;    // most MTP drafts per round

struct Lin {  // register-direct matmul weight
  id<MTLBuffer> w, s;
  size_t woff = 0, soff = 0;
  uint32_t N = 0, K = 0, bits = 0, gs = 0, experts = 1, ksplit = 1, sgs = 1;
  std::map<std::string, std::pair<uint32_t, uint32_t>> rows;
};

struct Raw {
  id<MTLBuffer> b;
  size_t off = 0, bytes = 0;
};

struct Layer {
  bool attn = false, ple = false;
  Lin ahc_dn, ahc_up, mhc_dn, mhc_up, in, out, moe_in, moe_sdown, moe_gu, moe_dn, ple_kv;
  Raw ahc_norm, mhc_norm;
  Raw conv, A_log, dt_bias, gnorm;                 // GDN
  Raw q_norm, k_norm, iq_norm, ik_norm;            // attention
  Raw ple_nk, ple_nq, ple_nc, ple_conv;            // PLE
  // state
  id<MTLBuffer> S, convh;                          // GDN
  id<MTLBuffer> Kc, Vc, rawk, bk;                  // attention cache
  id<MTLBuffer> plehist;                           // PLE conv history
  id<MTLBuffer> gin;                               // GDN layer input rows (kept for commit)
};

struct Checkpoint {  // recurrent state at a prefill boundary (pos multiple of CKPT)
  uint32_t pos = 0, conv_par = 0, ple_par = 0;
  id<MTLBuffer> blob;                      // per layer S, convh, plehist, then the last-row hidden
  uint64_t pinned = 0;                     // > 0: a later request branched from here (kept; larger = more recent)
};

struct SpecStats {
  double prefill_ms = 0, draft_ms = 0, verify_ms = 0;
  uint32_t rounds = 0, accepted = 0, reused = 0, drafted = 0;
  double cpu_encode_ms = 0, cpu_host_ms = 0, wall_ms = 0;
  double commit_ms = 0;  // joint speculation: the lanes' commit command buffer
  // hybrid chain: verify rows fed, context-lookup tokens proposed / tested / accepted
  uint32_t rows = 0, lk_proposed = 0, lk_tested = 0, lk_accepted = 0, lk_override = 0;
  // streak mode: chunks verified through the prefill path, how many rolled back, tokens gained
  uint32_t streak_chunks = 0, streak_fail = 0, streak_tokens = 0, ext_batches = 0, drafts_made = 0, tol_accepted = 0;
  double streak_ms = 0;
};

// Context-continuation drafts: an index of every LK_N-gram of the committed history. A draft
// chain may continue from the place the current tail last occurred. Drafts only: every
// proposed token still goes through the exact verify, so a stale or wrong entry costs rows,
// never correctness.
struct Lookup {
  static constexpr uint32_t LK_N = 3, CAND = 48, MAXLEN = 64, NB = 4;
  static constexpr uint32_t NONE = 0xFFFFFFFFu;
  std::unordered_map<uint64_t, std::vector<uint32_t>> at;  // key -> continuation positions, ascending
  size_t n = 0;                                            // continuation positions < n are indexed
  double rate[NB] = {0.30, 0.50, 0.70, 0.85};              // EMA acceptance by match-length bucket
  static uint64_t key(const uint32_t* t) {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = 0; i < LK_N; i++) { h ^= t[i]; h *= 1099511628211ull; }
    return h;
  }
  static uint32_t bucket(uint32_t len) { return len <= 4 ? 0 : len <= 8 ? 1 : len <= 16 ? 2 : 3; }
  void clear() { at.clear(); n = 0; }
  void sync(const std::vector<uint32_t>& hist);
  // Drop the entries for continuation positions >= newsize; call BEFORE hist is shortened.
  void truncate(const std::vector<uint32_t>& hist, size_t newsize);
  // Longest backward match of `tail` ending inside hist; returns the continuation position
  // (hist[p] is the proposed next token) or NONE, and the match length.
  uint32_t find(const std::vector<uint32_t>& hist, const std::vector<uint32_t>& tail, uint32_t* len) const;
};

struct GdnArgs { uint32_t M, split, N, nk, nv, dk, dv, conv_par, mode, row0 = 0; };
struct NgramTable;
struct Rec;  // CPU-side PLE n-gram rows (engine/ngram.mm)

// kv8 for Models built from now on: -1 = from OPUS_KV8, else 0/1 (kvtest builds one of each)
inline int kv8_force = -1;
inline bool kv8_default() {
  if (kv8_force >= 0) return kv8_force != 0;
  const char* e = getenv("OPUS_KV8");
  return e && atoi(e) != 0;
}

struct Model {
  mtl::Ctx C;
  std::string dir;
  // config
  uint32_t H, HC, R, nL, vocab, E, topk, I, eos;
  uint32_t nh, nkv, hd, inh, ihd, rot, ratio, budget;
  uint32_t nk, nv, dk, dv;
  double rope_theta;
  std::vector<int> ple_layers;
  // weights
  std::vector<id<MTLBuffer>> files;
  std::vector<Layer> L;
  Lin mix_dn, mix_up, lm_head;
  Raw mix_norm, embed;
  NgramTable* ngram = nullptr;
  // kernels
  id<MTLComputePipelineState> k_mm_w8g128, k_mm_w8g64, k_mm_w4g128, k_mm_w4g128_grp, k_mm_w4g64_grp, k_mm_w4g64, k_mm_w4g128_grp_x2, k_mm_w8g64_x1, k_hc_pre, k_hc_mid, k_hc_mix, k_gdn,
      k_embed, k_argmax, k_route, k_plan, k_act, k_comb, k_attn_prep, k_attn_blocks, k_qsa, k_attn_rows,
      k_ple_gate, k_ple_conv;
  // scratch
  id<MTLBuffer> ids, h, hn, hnf, ydn, df, inj, ymw, x, xf, yin, of, yo, topi, topw, sgate, shf, slots, nslots,
      slot_of, rmask, ygu, af, yd, ysh, moe_o, logits, amax_i, amax_v, rope, rope_blk, q, gate, iq, scores, sel, nsel,
      ple_e, ple_gv, ple_gvn, mix_x, part, counters, h_tgt, hsrc, mtp_h, seed_h, erows, pe, ph;
  id<MTLComputePipelineState> k_copy, k_rms_frag, k_mtp_fuse, k_route_rows, k_plan_tiles, k_gather, k_comb_tiles;
  bool fused_layout = false, attn_mma = false;
  bool kv8 = kv8_default();  // K/V caches as int8 + half scale per 32 dims (attn.metal kv8_scales)
  size_t kv_bytes(size_t ntok) const { return kv8 ? ntok * nkv * hd + ntok * nkv * (hd / 32) * 2 : ntok * nkv * hd * 2; }
  // blit src's K/V cache rows [0, P) into dst (caps may differ: kv8 scales sit after cap rows)
  void kv_copy(id<MTLBlitCommandEncoder> bl, const struct Layer& s, uint32_t scap, struct Layer& d, size_t P) const;
  id<MTLComputePipelineState> k_plan_count, k_plan_prefix, k_plan_scatter, k_zero;
  id<MTLBuffer> plan_cnt, plan_first;
  id<MTLBuffer> nsel_trunk, nsel_mtp;
  id<MTLSharedEvent> ple_event;
  uint64_t ple_wait = 0, ple_counter = 0;  // > 0: wait for ple_event >= ple_wait before PLE
  id<MTLComputePipelineState> k_qsa_scores;
  uint32_t hct = 256;
  id<MTLComputePipelineState> k_gdn_pre, k_gdn_scan, k_gdn_post;
  id<MTLBuffer> g_qn, g_kn, g_vv, g_o, g_bd;
  void encode_gdn(Rec& rc, Layer& l, id<MTLBuffer> gi, const GdnArgs& ga, uint32_t M);
  id<MTLComputePipelineState> k_epi1[2][2], k_epi2[2][2], k_epi3[2];
  id<MTLComputePipelineState> k_mmp[2][2][3];  // prefill dense matmul [w4][gs64][epi]
  uint32_t mmp_min = 512;
  uint32_t mmp_ch = 8;  // mmp row blocks per threadgroup (one simdgroup each): 8 = 128 rows per weight read
  // ride-along prefetch: the next ungrouped mm2 dispatches carry extra threadgroups that read
  // pf (the weights of an upcoming bandwidth-bound op), pf_mb bytes each (0 disables)
  struct Pf { id<MTLBuffer> b = nil; size_t off = 0, bytes = 0; } pf;
  size_t pf_mb = size_t(16) << 20, pf_take = 0;
  id<MTLBuffer> pf_sink;
  size_t pf_hc = 0, pf_gdn = 0, pf_route = 0, pf_comb = 0;  // bytes carried per op (OPUS_PF_HC/GDN/ROUTE/COMB, MB)
  // acceptance estimate of an MTP draft by argmax margin (best logit - runner-up), calibrated online
  double conf_rate[6] = {0.35, 0.42, 0.58, 0.85, 0.90, 0.97};
  static uint32_t conf_bin(float m) { return m < 1.0f ? 0 : m < 2.0f ? 1 : m < 3.0f ? 2 : m < 5.0f ? 3 : m < 8.0f ? 4 : 5; }
  std::map<std::string, double>* mtp_profile = nullptr;  // per-op times of mtp_forward (mprof mode)
  std::vector<uint16_t> pre_frag;  // n-gram rows of the next prefill chunk, fetched in the background
  uint32_t pre_frag_M = 0;
  bool pf_mm = false;
  // mm2 TILES 2 / 4 variants of each decode pipeline (builds with MAXM > 16): steps of 17..64 rows
  std::map<void*, std::array<id<MTLComputePipelineState>, 2>> mm2_tiled;
  static uint32_t tiles_for(uint32_t M) { return M <= 16 ? 1 : M <= 32 ? 2 : 4; }
  bool rmask_on = false;  // decode MoE: moe_route wrote rmask for this step's slots (OPUS_RMASK=0 disables)
  bool rmask_ok = true;
  bool pipe_is_gather(id<MTLComputePipelineState> p) const { return p == k_epi3_ga[0] || p == k_epi3_ga[1] || p == k_grp_ga[0] || p == k_grp_ga[1]; }
  bool red2 = false;   // join mm2's ksplit partials with a second dispatch instead of atomics (OPUS_RED2)  // also ride on ungrouped mm2 dispatches (measured: no net gain)
  // binds the prefetch buffers of an ops.metal kernel whose work grid is (gx, gy, 1) and returns
  // the extra z layers to dispatch; takes up to `take` bytes from pf
  uint32_t pf_ride(id<MTLComputeCommandEncoder> e, uint32_t gx, uint32_t gy, uint32_t tpt, size_t take) {
    struct { uint32_t gx, gy, tpt, n; } a{gx, gy, tpt, 0};
    take = std::min(take, pf.bytes);
    uint32_t pz = 0;
    if (take) {
      const size_t tl = size_t(gx) * gy * tpt;
      a.n = uint32_t(take / 16);
      pz = uint32_t((take / 256 + tl - 1) / tl);
      [e setBuffer:pf.b offset:pf.off atIndex:20];
      pf.off += take;
      pf.bytes -= take;
    } else {
      [e setBuffer:pf_sink offset:0 atIndex:20];
    }
    [e setBuffer:pf_sink offset:0 atIndex:21];
    [e setBytes:&a length:sizeof a atIndex:22];
    return pz;
  }
  void pf_set(const Lin& w, size_t skip = 0) {
    const size_t n = (size_t(w.N) * w.K * w.bits / 8) & ~size_t(4095);
    pf = skip < n ? Pf{w.w, w.woff + skip, n - skip} : Pf{};
  }   // rows from which the prefill matmul twin (mmp) beats mm2 row blocks
  static constexpr uint32_t SROWS = 256;  // most rows of one streak chunk (per-row logits)
  bool want_all = false;    // prefill-path forward: logits / argmax for every row, not just the last
  std::vector<uint32_t> stop_ids;  // draft chains end at these (plus eos); drafts only
  Checkpoint snapshot();                 // recurrent state now (not kept in ckpts)
  void rollback(const Checkpoint& c);
  // Become a copy of `src` at its current position (src.pos <= cap): the attention caches up to src.pos,
  // the recurrent state and the history. Both must be built on the same pack. MTP state is not copied
  // (mtp_synced = false), so a speculative request on this slot re-prefills unless it is a lane.
  void fork_from(const Model& src);
  // start from src's history[0:p], where p is src's newest checkpoint at or below `common` (src keeps
  // running: its cache rows below p are final). Returns p, 0 when src has no such checkpoint.
  size_t fork_prefix(Model& src, size_t common);
  size_t ckpt_below(size_t common) const;  // newest checkpoint position <= common, 0 if none    // back to a snapshot taken earlier in this conversation
  bool use_mmp = true, gather_mode = false;  // gather_mode: grouped matmul reads tile rows via tile_rows
  id<MTLComputePipelineState> k_epi3_ga[2], k_grp_ga[2];
  id<MTLBuffer> attn_part, rope_slot[NSLOT], rope_blk_slot[NSLOT], draft_out, mtp_ids[NSLOT], h_last_row, amax_m, draft_m;
  id<MTLBuffer> seed_h_lane[NSLOT];  // joint speculation: each lane's draft seed
  id<MTLComputePipelineState> k_attn_merge;
  id<MTLBuffer> tile_exp, ntiles, tile_rows, assign_pos, xg;
  uint32_t maxtiles = 0;
  // MTP head
  bool has_mtp = false;
  Layer mtp;
  Lin mtp_mix_dn, mtp_mix_up, mtp_fc_emb, mtp_fc_hid, mtp_head;  // mtp_head: draft-only 4-bit lm_head
  bool has_draft_head = false, has_sub_head = false;
  Raw sub_ids;
  id<MTLComputePipelineState> k_map_ids;
  double t_encode = 0, t_host = 0;
  std::chrono::steady_clock::time_point enc_t0;  // cumulative CPU ms: encoding, host-side prep (rope, n-grams)
  Raw mtp_mix_norm, mtp_norm_emb, mtp_norm_hid;
  uint32_t mtp_pos = 0, seed = 0, prev_dn_N = 0;
  uint32_t spec_level = 0;  // joint speculation: log2 of this lane's chain depth
  uint32_t acc_src0 = 0;
  id<MTLBuffer> dump_h = nil, dump_moe = nil;
  // btest: weight bytes the step reads. Dense matmuls are counted on the host; each layer's distinct
  // expert count (the router's nslots) is copied to dump_nslots[layer]
  bool count_bytes = false;
  double dense_bytes = 0;
  std::map<std::string, double> bytes_by_tag;  // btest: dense weight bytes per op tag
  id<MTLBuffer> dump_nslots = nil;
  id<MTLComputePipelineState> k_copy_u32;  // dtest: every layer's streams and pending MoE output    // joint speculation: driver h_tgt row of this lane's first pending accept pair
  bool skip_attn_extras = true;
  std::vector<uint32_t> pending_toks;
  // sampling (temp <= 0: greedy)
  float temp = 0.0f, top_p = 1.0f;
  bool tol_off = false;  // this request asked for exact acceptance (request flags & 8): OPUS_TOL does not apply
  uint32_t top_k = 20;
  uint64_t sample_seed = 0;
  // joint steps: each row sampled with its own lane's settings and position (forward_lanes fills this
  // while it encodes, and only when some lane samples; layout matches RowSamp in ops.metal)
  struct RowSamp { float temp, top_p; uint32_t top_k, seed_lo, seed_hi, pos; };
  std::vector<RowSamp> row_samp;
  bool lane_rows = false;  // a forward_lanes step is being encoded: sampling comes from row_samp only
  id<MTLComputePipelineState> k_sample;
  bool mtp_synced = false;
  static constexpr uint32_t CKPT = 1024;  // checkpoint spacing; older ones thinned to 8192
  size_t ckpt_keep = 8, ckpt_far = 8192, ckpt_pins = 4;
  uint64_t pin_tick = 0;
  std::vector<size_t> branch_pts;  // where recent requests left this conversation's history (pin the checkpoint below)  // checkpoint retention (fewer when several slots share the RAM)
  std::vector<Checkpoint> ckpts;
  std::vector<id<MTLBuffer>> ckpt_pool;
  size_t ckpt_bytes() const;
  void save_ckpt();
  bool restore_ckpt(size_t common);  // largest checkpoint <= common; false if none
  std::vector<dispatch_source_t> state_timers;  // residency refresh (OPUS_RESIDENCY_MS)
  uint32_t cap = 0;        // KV capacity (tokens)
  uint32_t pos = 0;        // tokens already in the cache
  uint32_t conv_par = 0, ple_par = 0;
  std::vector<uint32_t> history;  // all tokens fed so far (for PLE n-grams)
  Lookup lk;                      // context-continuation draft index over history

  // share: another Model on the same pack whose device, pipelines and (read-only) weight buffers
  // this one reuses; it then only owns its own conversation state
  Model(const std::string& pack_dir, uint32_t capacity, const Model* share = nullptr);
  // Feed M tokens at positions pos..pos+M-1; returns greedy next token per row.
  std::vector<uint32_t> step(const std::vector<uint32_t>& toks, double* gpu_ms = nullptr,
                             std::vector<float>* logits_out = nullptr,
                             std::map<std::string, double>* profile = nullptr);  // per-op GPU ms
  void reset();
  // mode 0: feed and commit; mode 1: verify (commit later with commit(j))
  std::vector<uint32_t> forward(const std::vector<uint32_t>& toks, int mode, double* gpu_ms = nullptr,
                                std::vector<float>* logits_out = nullptr,
                                std::map<std::string, double>* profile = nullptr);
  void commit(uint32_t j);
  // Lanes: several conversations (Models on the same pack, built with `share`) advance in ONE step.
  // The rows of all lanes are concatenated (at most MAXM in total); every stateless op runs once over
  // all of them, the stateful ops (GatedDeltaNet, attention, PLE conv) once per lane on that lane's own
  // state and position. Called on the driver, whose scratch buffers the step uses; a lane's bits are the
  // same as when it runs alone. mode as forward(); after mode 1 call commit_lane on each lane.
  struct LaneRef { Model* m; uint32_t row0, M, p0; int slot; };
  std::vector<LaneRef> jl;  // non-empty while a joint step is being encoded
  std::vector<std::vector<uint32_t>> forward_lanes(const std::vector<Model*>& lanes,
                                                   const std::vector<std::vector<uint32_t>>& toks, int mode,
                                                   std::vector<std::vector<float>>* logits_out = nullptr,
                                                   double* gpu_ms = nullptr,
                                                   std::map<std::string, double>* profile = nullptr);
  void commit_lane(Model& drv, uint32_t row0, uint32_t j);  // this lane's first j rows of the joint verify
  void encode_commit_lane(Rec& rc, Model& drv, uint32_t row0, uint32_t j);
  // Joint speculation. Every lane runs its own strict MTP chain (K = 1 << level, never > MAXD); the lanes'
  // draft calls share one command buffer, the verify is one joint step, the commit is per lane. `drv` is the
  // driver: the stateful kernels read each lane's own state, the driver's scratch holds the joint rows.
  // out[k] = lane k's tokens, bit-identical to generate_spec(prompts[k]) on that lane alone.
  std::vector<std::vector<uint32_t>> generate_spec_lanes(Model& drv, const std::vector<Model*>& lanes,
                                                        const std::vector<std::vector<uint32_t>>& prompts,
                                                        uint32_t n, SpecStats* st = nullptr);
  void spec_lanes_round(Model& drv, const std::vector<Model*>& lanes, std::vector<uint32_t>& x,
                        std::vector<std::vector<uint32_t>>& acc, std::vector<std::vector<uint32_t>>& out,
                        std::vector<bool>& live, uint32_t n, SpecStats* st,
                        const std::function<uint32_t(uint32_t, const std::vector<uint32_t>&)>* cut = nullptr);
  // cut (pool serving): given lane k and the tokens this round would emit, how many to keep (>= 1); the
  // round then accepts only those, exactly as plain decoding would stop there (stop token, thinking
  // budget, max_tokens)
  // Lanes through the MTP head in one pass (joint speculation's drafts); see step.inc
  std::vector<uint32_t> mtp_lanes(const std::vector<Model*>& lanes, const std::vector<std::vector<uint32_t>>& toks,
                                  const std::vector<std::pair<id<MTLBuffer>, uint32_t>>& src,
                                  const std::vector<id<MTLBuffer>>& seed, double* gpu_ms = nullptr);
  void encode_ple_lanes(Rec& rc, Layer& l, size_t li, int mode);
  void encode_gdn_lanes(Rec& rc, Layer& l, size_t li, int mode);
  void encode_attn_lanes(Rec& rc, Layer& l, size_t li);
  // Lane-batched stateful ops (OPUS_LANE_BATCH, default on): one dispatch per layer for every lane of a
  // joint step; each lane's state is found through a table of buffer addresses, each row's lane in rowlane
  bool lane_batch = true;
  id<MTLComputePipelineState> k_gdn_lanes, k_attn_prep_l, k_attn_blocks_l, k_attn_rows_l, k_attn_merge_l;
  id<MTLBuffer> lane_attn_tab, lane_gdn_tab, rowlane, rope_lanes, rope_blk_lanes;
  void lanes_tables();  // rowlane and the step's rope tables, from jl
  void encode_gdn_lanes_b(Rec& rc, Layer& l, size_t li, int mode);
  void encode_attn_lanes_b(Rec& rc, Layer& l, size_t li);
  std::vector<uint32_t> mtp_forward(const std::vector<uint32_t>& toks, id<MTLBuffer> src, uint32_t src0,
                                    double* gpu_ms = nullptr);
  uint32_t feed(const std::vector<uint32_t>& toks, bool with_mtp, SpecStats* st = nullptr);
  uint32_t generate(const std::vector<uint32_t>& prompt, uint32_t max_new, bool spec,
                    const std::function<bool(uint32_t)>& cb, SpecStats* st = nullptr);
  std::vector<uint32_t> generate_spec(const std::vector<uint32_t>& prompt, uint32_t n, SpecStats* st = nullptr);

 private:
  std::vector<id<MTLCounterSampleBuffer>> prof_pool;  // per-op timestamps, PROF_OPS ops per buffer
  void mm(Rec& rc, const char* tag, const Lin& w, id<MTLBuffer> X, id<MTLBuffer> Y, uint32_t M,
          uint32_t slots_max = 0, uint32_t xstride = 0, uint32_t ystride = 0,
          id<MTLComputePipelineState> pipe = nil, uint32_t ldx = 0, float xdiv = 1.0f, bool barrier = true,
          int epi = 0, id<MTLBuffer> e0 = nil, id<MTLBuffer> e1 = nil, id<MTLBuffer> e2 = nil, uint32_t epiR = 0);
  void set_rope(uint32_t p0, uint32_t M, int slot = 0);
  void encode_commit(Rec& rc, uint32_t j);
  uint32_t encode_trunk(Rec& rc, uint32_t M, int mode, uint32_t p0);
  void encode_mtp(Rec& rc, uint32_t M, const uint32_t* host_toks, id<MTLBuffer> src, uint32_t src0, int slot,
                  int out_slot);
  void copy(Rec& rc, id<MTLBuffer> src, uint32_t src0, id<MTLBuffer> dst, uint32_t dst0, uint32_t rows, uint32_t n);
  void copy_enc(id<MTLComputeCommandEncoder> e, id<MTLBuffer> src, uint32_t src0, id<MTLBuffer> dst,
                       uint32_t dst0, uint32_t rows, uint32_t n);
  void encode_layer(Rec& rc, Layer& l, uint32_t M, uint32_t p0, int mode, bool& pending);
  void encode_head(Rec& rc, const Lin& dn, const Lin& up, const Raw& norm, uint32_t M, bool pending, uint32_t p0,
                   const Lin* head = nullptr);
  void begin(Rec& rc, std::map<std::string, double>* profile);
  double finish(Rec& rc, std::map<std::string, double>* profile);
  void copy_mtp_to_seed(uint32_t row = 0);
  void hc_block(Rec& rc, const Lin& dn, const Lin& up, const Raw& norm, uint32_t M, uint32_t inject, id<MTLBuffer> o);
};

}  // namespace opus
