// DankAttn (Inkling-2-Mini): Inkling2LoopForConditionalGeneration (text tower) + DSpark MTP on Metal 4.
// Shares register-direct matmuls (kernels/mm2.metal: W8, W6, native NVFP4 WBITS=40, native MXFP8 WBITS=80)
// and mtl.h with Hy4 and Flash-Next.
#pragma once
#include "../mtl.h"
#include "../mm_args.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ink2 {

constexpr uint32_t MAXM = 16;   // rows per decode / verify step (mm2)
constexpr uint32_t MTP_MAXL = 8; // max lanes in one batched DSpark draft call
constexpr uint32_t MTP_MAXB = MTP_MAXL * 8u; // 64 rows across 8 lanes x block_size 8
constexpr uint32_t MP = 512;    // max rows per chunked prefill step (mmp / mmpg)
constexpr uint32_t PF_COMPACT = 6144;  // max compact padded MoE rows for MP=512 (512*8 + 128*15 = 6016)
constexpr uint32_t MAX_BLOCKS = 12;
constexpr uint32_t ASPLITS = 64; // max attention key splits (fixed per layer -> bit-exact across M)

struct Lin {
  id<MTLBuffer> w = nil, s = nil;
  size_t woff = 0, soff = 0;
  uint32_t N = 0, K = 0, bits = 0, gs = 0, experts = 1, ksplit = 1, sgs = 1;
  uint32_t pslot = 0;  // split-K scratch region: tensors that run concurrently (qgrs || k || v) need their own
};

struct Raw {
  id<MTLBuffer> b = nil;
  size_t off = 0, bytes = 0;
};

struct Ctrl {
  Raw down, forget, gain;
};

struct PhysLayer {
  bool is_local = false;
  bool is_loop = false;
  bool is_boundary = false;
  Raw ar_attn_q, ar_mlp_q;
  Ctrl ctrl_q, ctrl_k, ctrl_v, ctrl_mlp;
  Raw attn_norm, q_norm, k_norm, v_norm, mlp_norm;
  Lin qgrs, k_proj, v_proj, o_proj;
  Raw q_scale, rel_scale, rel_proj;
  Raw k_sconv, v_sconv, attn_sconv, mlp_sconv;
  Raw router_w, router_b;
  Lin moe_gu, moe_dn;
};

struct LogicalState {
  uint32_t phys_idx = 0;
  uint32_t cap = 0;
  bool kv8 = false;
  id<MTLBuffer> k_cache = nil;
  id<MTLBuffer> v_cache = nil;
  id<MTLBuffer> k_i8 = nil, k_sc = nil;
  id<MTLBuffer> v_i8 = nil, v_sc = nil;
  id<MTLBuffer> k_conv = nil, k_conv_next = nil, k_hist = nil;       // [3, C] and [3 + MAXM, C] f32
  id<MTLBuffer> v_conv = nil, v_conv_next = nil, v_hist = nil;
  id<MTLBuffer> attn_conv = nil, attn_conv_next = nil, attn_hist = nil;
  id<MTLBuffer> mlp_conv = nil, mlp_conv_next = nil, mlp_hist = nil;
};

struct MtpLayer {
  bool is_local = false;
  uint32_t cap = 0;
  Raw dflare_norm, attn_norm, mlp_norm, q_norm, k_norm, k_norm_ctx;
  Lin qkv, kv_ctx, o, mlp_gu, mlp_dn;
  id<MTLBuffer> k_cache = nil, v_cache = nil;
};

struct Model {
  mtl::Ctx C;
  std::string dir;

  // Config
  uint32_t H = 2304, nL = 24, vocab = 201024, vocab_valid = 200058, vocab_pad = 200064;
  uint32_t nh = 24, nkv = 8, hd = 128, d_rel = 16, rel_extent = 1024, window = 1024;
  uint32_t E = 128, topk = 8, I_moe = 1152, ks = 4, n_per_block = 6;
  uint32_t core_start = 6, core_end = 18, num_passes = 3, gate_rank = 64;
  uint32_t eos = 200006, cap = 32768;
  uint32_t nsp_local = 8, nsp_global = 8;
  bool big_gpu = false;
  uint32_t ring_extra = 0;  // INK_RING_SLACK: extra local-ring rows, so a lane can rewind MP + ring_extra tokens
  bool qrel_mma = true;  // INK_QREL_MMA: prefill rel_logits via matmul2d (ink_q_rel_prep + ink_q_rel_mma)
  bool use_kv8 = true;
  float eps = 1e-6f, qk_clip = 22.627417f, rel_clip = 8.0f;
  float log_n_floor = 8192.0f, log_alpha = 0.1f, route_scale = 8.0f;
  float gate_softcap = 4.0f, up_softcap = 25.0f, logit_softcap = 30.0f;

  // Mapped shard files
  std::vector<id<MTLBuffer>> files;
  std::vector<id<MTLBuffer>> mtp_files;
  size_t mapped_bytes = 0;

  // Weights
  Raw embed, embed_norm, norm, ar_out_q, loop_msg_q;
  Lin lm_head;
  std::vector<PhysLayer> L;

  // Sampling: temp <= 0 is greedy argmax; else Gumbel-max keyed by (seed, input position) after min_p / top_k /
  // top_p truncation, on GPU. DSpark drafts draw with the same keys, so speculation stays exact.
  struct Samp {
    float temp = 0.0f, min_p = 0.0f, top_p = 1.0f;
    uint32_t top_k = 0;
    uint64_t seed = 0;
  };
  Samp samp;  // active lane's

  // Per-logical-layer state of the active lane (`S`, `pos`); other lanes are parked in `lanes_`.
  std::vector<LogicalState> S;
  uint32_t pos = 0;
  uint32_t fork_pos = 0;
  int fork_src = -1;
  struct Lane {
    std::vector<LogicalState> S;  // empty while the lane is active (its state lives in Model::S)
    uint32_t pos = 0;
    uint32_t fork_pos = 0;
    int fork_src = -1;
    // DSpark context state (KV of the 3 MTP layers, context position, mixed feature); live copy in MtpL / Model
    std::vector<id<MTLBuffer>> mtp_k, mtp_v;
    std::vector<uint32_t> mtp_cap;
    uint32_t mtp_ctx_pos = 0;
    id<MTLBuffer> mtp_f_rnn = nil;
    Samp samp;
  };
  // First row of the active lane's rows in the last step: mtp_commit / draft(n_commit) read its b_taps rows.
  uint32_t tap_row0 = 0;
  std::vector<Lane> lanes_;
  int active_lane = 0;
  void alloc_state(std::vector<LogicalState>& SS, uint32_t global_cap = 0);
  // Add an independent sequence (own KV rings, sconv state, rollback history); returns its lane id.
  // global_cap > 0 allocates smaller global KV/MTP suffix buffers (for Split-KV forked lanes).
  int add_lane(uint32_t global_cap = 0);
  // Make lane k the target of step / reset / rollback / draft.
  void use_lane(int k);
  // Zero-copy Split-KV fork on global layers + fast local SWA/sconv/MTP copy from src -> dst at src's current pos.
  void fork_lane(int src, int dst);
  // Promote a forked lane `src` (forked from `dst`) into standalone `dst` by copying its generated suffix [fork_pos..pos).
  void promote_lane(int src, int dst);
  uint32_t lane_pos(int k) const { return k == active_lane ? pos : lanes_[k].pos; }
  uint32_t lane_global_cap(int k) const;
  // Rewind support for the active lane: the sconv states at a position (saved right after a prefill) plus the
  // KV rings still holding the window before it. rewind(c) needs pos - c <= rewind_slack().
  void save_conv(std::vector<uint8_t>& out) const;
  void rewind(uint32_t c, const std::vector<uint8_t>& conv);
  uint32_t rewind_slack() const { return MP + ring_extra; }
  struct LaneRef {
    std::vector<LogicalState>* S;
    std::vector<LogicalState>* pref_S;
    uint32_t* pos;
    uint32_t fork_pos;
    uint32_t r0, m;  // rows [r0, r0 + m) of the step belong to this lane
    const Samp* sp;
  };

  // DSpark MTP drafter (optional overlay: mtp_manifest.json + pack-mtp-00.bin)
  bool has_mtp = false;
  uint32_t mtp_nL = 3, mtp_I = 9216, mtp_block = 8, mtp_rank = 256;
  float mtp_rope_theta = 1000000.0f;
  uint32_t mtp_taps[4] = {5, 23, 41, 48};
  uint32_t mtp_ctx_pos = 0;
  Raw mtp_fuse_alpha, mtp_mask_embed, mtp_norm;
  Raw mtp_mh_dflare_norm, mtp_mh_markov_w1, mtp_mh_rnn_w, mtp_mh_rnn_b, mtp_mh_fold_b;
  Lin mtp_mh_fold_w;
  std::vector<MtpLayer> MtpL;

  // Pipelines
  id<MTLComputePipelineState> p_embed, p_attn_res, p_attn_res_rms, p_attn_res_mg_prep, p_attn_res_mg_prep_shared;
  id<MTLComputePipelineState> p_mg_prep, p_mg_down, p_mg_apply, p_mg_down3, p_mg_apply3, p_rms_frag;
  id<MTLComputePipelineState> p_mg_down_mma, p_mg_apply_mma;
  id<MTLComputePipelineState> p_q_rel, p_q_rel_prep, p_q_rel_mma, p_kv_conv, p_attn_split, p_attn_reduce, p_post_conv;
  id<MTLComputePipelineState> p_attn_pf_local, p_attn_pf_global_bf16, p_attn_pf_global_kv8;
  id<MTLComputePipelineState> p_attn_dec_local, p_attn_dec_global_bf16, p_attn_dec_global_kv8;
  id<MTLComputePipelineState> p_scores, p_scores_mma, p_route, p_pack_dec, p_moe_accum, p_softcap_argmax_p1, p_softcap_argmax_p2, p_copy;
  id<MTLComputePipelineState> p_topk_pf, p_plan_pf, p_moe_accum_pf;
  id<MTLComputePipelineState> p_route_fast;      // decode router: rank top-k + prefix slots, one phase
  id<MTLComputePipelineState> p_moe_accum_conv;  // decode fusion: moe_accum + mlp post_sconv
  id<MTLComputePipelineState> p_conv_res_rms;    // decode fusion: attn post_sconv + pre-MLP attn_res_rms
  id<MTLComputePipelineState> p_conv_res_mg;     // decode fusion: attn post_sconv + pre-MLP attn_res_mg_prep
  id<MTLComputePipelineState> p_mm_w8, p_mm_w6_gu, p_mm_w6_dn;
  id<MTLComputePipelineState> p_mmp_w8, p_mmp_w8_mtp, p_mmp_w6_gu, p_mmp_w6_dn;
  id<MTLComputePipelineState> p_moe_fp4_gu, p_moe_fp4_dn;
  id<MTLComputePipelineState> p_moe_fp8_gu, p_moe_fp8_dn;
  id<MTLComputePipelineState> p_mmpg_fp4_gu, p_mmpg_fp4_dn;
  id<MTLComputePipelineState> p_mmpg_fp8_gu, p_mmpg_fp8_dn;
  // DSpark pipelines
  id<MTLComputePipelineState> p_mtp_mix, p_mtp_kv_ctx, p_mtp_embed, p_mtp_qkv_rope;
  id<MTLComputePipelineState> p_mtp_attn, p_add_rms_frag, p_mtp_rnn_static, p_mtp_rnn_dyn, p_mtp_rnn_gates;
  id<MTLComputePipelineState> p_mtp_add_argmax_p1, p_mtp_add_argmax_p2, p_pack_frag_rows, p_sample_rows;
  id<MTLResidencySet> rs = nil;
  id<MTLHeap> scratch_heap = nil;

  // Scratch buffers (sized for MAXM = 16)
  id<MTLBuffer> b_ids, b_h, b_pre_h, b_prev_h, b_blocks, b_pre_blocks, b_prev_blocks;
  id<MTLBuffer> b_qkv_in, b_qkv_out, b_msgs, b_mh, b_mn_f32, b_rscores;
  id<MTLBuffer> b_mg_cat, b_mg_low, b_attn_part;
  id<MTLBuffer> b_frag_q, b_frag_k, b_frag_v, b_frag_o, b_frag_m, b_frag_gu;
  id<MTLBuffer> b_qgrs, b_k_raw, b_v_raw, b_Q, b_rel_logits, b_rel_vec, b_gate, b_sinks, b_o_out;
  id<MTLBuffer> b_top_idx, b_top_w, b_slots, b_nslots, b_rmask;
  id<MTLBuffer> b_roff, b_rcnt, b_rlist, b_rpos, b_wtile, b_ind_gu, b_ind_dn;
  id<MTLBuffer> b_moe_y, b_moe_out, b_logits, b_tok_out, b_argmax_v, b_argmax_i, b_conf_out;
  id<MTLBuffer> b_dummy;
  // Per-row Self-Certainty (KL from uniform over softcapped logits) from the last step / step_lanes call.
  std::vector<float> last_conf;

  // DSpark scratch buffers
  id<MTLBuffer> b_taps;          // [4, MP, H] f32 captured target hidden states
  id<MTLBuffer> b_mtp_f_rnn;     // [MP, H] f32 mixed context feature for Markov RNN
  id<MTLBuffer> b_mtp_h, b_mtp_frag, b_mtp_qkv, b_mtp_kv_ctx, b_mtp_Q, b_mtp_K_blk, b_mtp_V_blk;
  id<MTLBuffer> b_mtp_o_frag, b_mtp_o_out, b_mtp_gu_frag, b_mtp_dn_out;
  id<MTLBuffer> b_mtp_base_h, b_mtp_base_logits, b_mtp_bias;
  id<MTLBuffer> b_mtp_rnn_state, b_mtp_rnn_static, b_mtp_rnn_proj, b_mtp_lat_frag;
  id<MTLBuffer> b_mtp_prev_tok, b_mtp_draft_out, b_mtp_draft_probs, b_mtp_part_l, b_mtp_part_x, b_samp_rows;

  void load(const std::string& pack_dir, uint32_t max_cap = 32768, uint32_t passes_override = 0, bool kv8 = true);
  void reset();
  // Run M <= MP tokens starting at current `pos`, advancing `pos += M`.
  // M <= MAXM uses bit-exact mm2 decode/verify; M > MAXM uses mmp/mmpg chunked prefill (computes last row's logits).
  // All rows' logits of one step (decode/verify path), for A/B dumps.
  std::vector<uint32_t> step_all(const std::vector<uint32_t>& ids, std::vector<float>* all_logits);
  std::vector<uint32_t> step(const std::vector<uint32_t>& ids, double* gpu_ms = nullptr,
                             std::vector<float>* last_logits = nullptr,
                             std::map<std::string, double>* prof = nullptr);
  // Multi-lane batched decode / verify: each (lane, toks) appends toks at that lane's pos; total rows <= MAXM.
  // Stateless ops (projections, MoE, norms, message gates) run once over all rows; q_rel, K/V sconv + cache
  // writes, decode attention and the post-attention / post-MLP sconv dispatch per lane against its own state.
  // Returns argmax per row; all_logits gets [M][vocab_valid] when given.
  std::vector<uint32_t> step_lanes(const std::vector<std::pair<int, std::vector<uint32_t>>>& batches,
                                   double* gpu_ms = nullptr, std::vector<float>* all_logits = nullptr,
                                   std::map<std::string, double>* prof = nullptr);
  std::vector<uint32_t> step_core(const std::vector<uint32_t>& ids, const std::vector<LaneRef>& lanes, double* gpu_ms,
                                  std::vector<float>* last_logits, std::vector<float>* all_logits,
                                  std::map<std::string, double>* prof);
  // Roll back sconv state & `pos` after verifying `M_step` rows and accepting `accepted` (1 <= accepted <= M_step).
  void rollback(uint32_t accepted, uint32_t M_step);
  // Commit `n_commit` accepted tokens' tapped target features (from the last step) into DSpark KV cache + f_rnn.
  void mtp_commit(uint32_t n_commit, double* gpu_ms = nullptr);
  struct MtpCommitReq { int lane; uint32_t tap_row0; uint32_t n_commit; };
  // Batched DSpark commit across lanes in one command buffer (reads kv_ctx weights once over total_tap_rows <= MAXM).
  void mtp_commit_lanes(const std::vector<MtpCommitReq>& reqs, uint32_t total_tap_rows, double* gpu_ms = nullptr);
  // Draft up to `max_drafts` (1..7, default 7) tokens starting from `anchor_tok` at current `pos`.
  // If `n_commit > 0`, fuses `mtp_commit(n_commit)` into the same command buffer (zero extra GPU sync!).
  std::vector<uint32_t> draft(uint32_t anchor_tok, uint32_t max_drafts = 7,
                              std::vector<float>* draft_probs = nullptr, double* gpu_ms = nullptr,
                              uint32_t n_commit = 0);
  struct MtpDraftReq { int lane; uint32_t anchor_tok; };
  // Batched DSpark drafting across L lanes (L <= MTP_MAXL) in one command buffer; bit-exact vs sequential draft().
  void draft_lanes(const std::vector<MtpDraftReq>& reqs, uint32_t max_drafts,
                   std::vector<std::vector<uint32_t>>& out_drafts,
                   std::vector<std::vector<float>>* out_probs = nullptr, double* gpu_ms = nullptr);
};

}  // namespace ink2
