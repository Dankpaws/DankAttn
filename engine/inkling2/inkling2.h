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
constexpr uint32_t MP = 512;    // max rows per chunked prefill step (mmp / mmpg)
constexpr uint32_t PF_COMPACT = 6144;  // max compact padded MoE rows for MP=512 (512*8 + 128*15 = 6016)
constexpr uint32_t MAX_BLOCKS = 12;
constexpr uint32_t ASPLITS = 24; // max attention key splits (fixed per layer -> bit-exact across M)

struct Lin {
  id<MTLBuffer> w = nil, s = nil;
  size_t woff = 0, soff = 0;
  uint32_t N = 0, K = 0, bits = 0, gs = 0, experts = 1, ksplit = 1, sgs = 1;
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

  // Per-logical-layer state
  std::vector<LogicalState> S;
  uint32_t pos = 0;

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
  id<MTLComputePipelineState> p_embed, p_attn_res, p_attn_res_rms, p_attn_res_mg_prep;
  id<MTLComputePipelineState> p_mg_prep, p_mg_down, p_mg_apply, p_mg_down3, p_mg_apply3, p_rms_frag;
  id<MTLComputePipelineState> p_mg_down_mma, p_mg_apply_mma;
  id<MTLComputePipelineState> p_q_rel, p_kv_conv, p_attn_split, p_attn_reduce, p_post_conv;
  id<MTLComputePipelineState> p_attn_pf_local, p_attn_pf_global_bf16, p_attn_pf_global_kv8;
  id<MTLComputePipelineState> p_attn_dec_local, p_attn_dec_global_bf16, p_attn_dec_global_kv8;
  id<MTLComputePipelineState> p_scores, p_route, p_pack_dec, p_moe_accum, p_softcap_argmax_p1, p_softcap_argmax_p2, p_copy;
  id<MTLComputePipelineState> p_topk_pf, p_plan_pf, p_moe_accum_pf;
  id<MTLComputePipelineState> p_mm_w8, p_mm_w6_gu, p_mm_w6_dn;
  id<MTLComputePipelineState> p_mmp_w8;
  id<MTLComputePipelineState> p_moe_fp4_gu, p_moe_fp4_dn;
  id<MTLComputePipelineState> p_moe_fp8_gu, p_moe_fp8_dn;
  id<MTLComputePipelineState> p_mmpg_fp4_gu, p_mmpg_fp4_dn;
  id<MTLComputePipelineState> p_mmpg_fp8_gu, p_mmpg_fp8_dn;
  // DSpark pipelines
  id<MTLComputePipelineState> p_mtp_mix, p_mtp_kv_ctx, p_mtp_embed, p_mtp_qkv_rope;
  id<MTLComputePipelineState> p_mtp_attn, p_add_rms_frag, p_mtp_rnn_static, p_mtp_rnn_dyn, p_mtp_rnn_gates;
  id<MTLComputePipelineState> p_mtp_add_argmax_p1, p_mtp_add_argmax_p2;
  id<MTLResidencySet> rs = nil;

  // Scratch buffers (sized for MAXM = 16)
  id<MTLBuffer> b_ids, b_h, b_pre_h, b_prev_h, b_blocks, b_pre_blocks, b_prev_blocks;
  id<MTLBuffer> b_qkv_in, b_qkv_out, b_msgs, b_mh, b_mn_f32, b_rscores;
  id<MTLBuffer> b_mg_cat, b_mg_low, b_attn_part;
  id<MTLBuffer> b_frag_q, b_frag_k, b_frag_v, b_frag_o, b_frag_m, b_frag_gu;
  id<MTLBuffer> b_qgrs, b_k_raw, b_v_raw, b_Q, b_rel_logits, b_gate, b_sinks, b_o_out;
  id<MTLBuffer> b_top_idx, b_top_w, b_slots, b_nslots, b_rmask;
  id<MTLBuffer> b_roff, b_rcnt, b_rlist, b_rpos, b_wtile, b_ind_gu, b_ind_dn;
  id<MTLBuffer> b_moe_y, b_moe_out, b_logits, b_tok_out, b_argmax_v, b_argmax_i;
  id<MTLBuffer> b_dummy;

  // DSpark scratch buffers
  id<MTLBuffer> b_taps;          // [4, MP, H] f32 captured target hidden states
  id<MTLBuffer> b_mtp_f_rnn;     // [MP, H] f32 mixed context feature for Markov RNN
  id<MTLBuffer> b_mtp_h, b_mtp_frag, b_mtp_qkv, b_mtp_kv_ctx, b_mtp_Q, b_mtp_K_blk, b_mtp_V_blk;
  id<MTLBuffer> b_mtp_o_frag, b_mtp_o_out, b_mtp_gu_frag, b_mtp_dn_out;
  id<MTLBuffer> b_mtp_base_h, b_mtp_base_logits, b_mtp_bias;
  id<MTLBuffer> b_mtp_rnn_state, b_mtp_rnn_static, b_mtp_rnn_proj, b_mtp_lat_frag;
  id<MTLBuffer> b_mtp_prev_tok, b_mtp_draft_out, b_mtp_draft_probs, b_mtp_part_l;

  void load(const std::string& pack_dir, uint32_t max_cap = 32768, uint32_t passes_override = 0, bool kv8 = true);
  void reset();
  // Run M <= MP tokens starting at current `pos`, advancing `pos += M`.
  // M <= MAXM uses bit-exact mm2 decode/verify; M > MAXM uses mmp/mmpg chunked prefill (computes last row's logits).
  std::vector<uint32_t> step(const std::vector<uint32_t>& ids, double* gpu_ms = nullptr,
                             std::vector<float>* last_logits = nullptr,
                             std::map<std::string, double>* prof = nullptr);
  // Roll back sconv state & `pos` after verifying `M_step` rows and accepting `accepted` (1 <= accepted <= M_step).
  void rollback(uint32_t accepted, uint32_t M_step);
  // Commit `n_commit` accepted tokens' tapped target features (from the last step) into DSpark KV cache + f_rnn.
  void mtp_commit(uint32_t n_commit, double* gpu_ms = nullptr);
  // Draft up to `max_drafts` (1..7, default 7) tokens starting from `anchor_tok` at current `pos`.
  std::vector<uint32_t> draft(uint32_t anchor_tok, uint32_t max_drafts = 7,
                              std::vector<float>* draft_probs = nullptr, double* gpu_ms = nullptr);
};

}  // namespace ink2
