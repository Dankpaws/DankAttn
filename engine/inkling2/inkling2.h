// DankAttn (Inkling-2-Mini): Inkling2LoopForConditionalGeneration (text tower) on Metal 4.
// Shares register-direct matmuls (kernels/mm2.metal, including native NVFP4 WBITS=40 & MXFP8 WBITS=80)
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
constexpr uint32_t MAX_BLOCKS = 12;
constexpr uint32_t ASPLITS = 8; // fixed attention key splits (bit-exact across M)

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
  id<MTLBuffer> k_cache = nil;
  id<MTLBuffer> v_cache = nil;
  id<MTLBuffer> k_conv = nil;     // [3, nkv * hd] f32
  id<MTLBuffer> v_conv = nil;     // [3, nkv * hd] f32
  id<MTLBuffer> attn_conv = nil;  // [3, H] f32
  id<MTLBuffer> mlp_conv = nil;   // [3, H] f32
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
  float eps = 1e-6f, qk_clip = 22.627417f, rel_clip = 8.0f;
  float log_n_floor = 8192.0f, log_alpha = 0.1f, route_scale = 8.0f;
  float gate_softcap = 4.0f, up_softcap = 25.0f, logit_softcap = 30.0f;

  // Mapped shard files
  std::vector<id<MTLBuffer>> files;
  size_t mapped_bytes = 0;

  // Weights
  Raw embed, embed_norm, norm, ar_out_q, loop_msg_q;
  Lin lm_head;
  std::vector<PhysLayer> L;

  // Per-logical-layer state (size = 24 + (num_passes - 1) * (core_end - core_start))
  std::vector<LogicalState> S;
  uint32_t pos = 0;

  // Pipelines
  id<MTLComputePipelineState> p_embed, p_attn_res, p_mg_prep, p_mg_down, p_mg_apply, p_rms_frag;
  id<MTLComputePipelineState> p_q_rel, p_kv_conv, p_attn_split, p_attn_reduce, p_post_conv;
  id<MTLComputePipelineState> p_scores, p_route, p_moe_accum, p_softcap_argmax, p_copy;
  id<MTLComputePipelineState> p_mm_w8, p_mm_w8_red2;
  id<MTLComputePipelineState> p_moe_fp4_gu, p_moe_fp4_dn;
  id<MTLComputePipelineState> p_moe_fp8_gu, p_moe_fp8_dn;
  id<MTLResidencySet> rs = nil;

  // Scratch buffers (sized for MAXM = 16)
  id<MTLBuffer> b_ids, b_h, b_pre_h, b_prev_h, b_blocks, b_pre_blocks, b_prev_blocks;
  id<MTLBuffer> b_qkv_in, b_msgs, b_mh, b_mn_f32, b_rscores;
  id<MTLBuffer> b_mg_cat, b_mg_low, b_attn_part;
  id<MTLBuffer> b_frag_q, b_frag_k, b_frag_v, b_frag_o, b_frag_m, b_frag_gu;
  id<MTLBuffer> b_qgrs, b_k_raw, b_v_raw, b_Q, b_rel_logits, b_gate, b_sinks, b_o_out;
  id<MTLBuffer> b_top_idx, b_top_w, b_slots, b_nslots, b_rmask;
  id<MTLBuffer> b_moe_y, b_moe_out, b_logits, b_tok_out;
  id<MTLBuffer> b_partials, b_cnt;
  id<MTLBuffer> b_dummy;

  void load(const std::string& pack_dir, uint32_t max_cap = 32768, uint32_t passes_override = 0);
  void reset();
  // Run M <= MAXM tokens starting at current `pos`, advancing `pos += M`.
  // Returns greedy argmax token for each row; optionally writes last row's logits, timing, and per-op profile.
  std::vector<uint32_t> step(const std::vector<uint32_t>& ids, double* gpu_ms = nullptr,
                             std::vector<float>* last_logits = nullptr,
                             std::map<std::string, double>* prof = nullptr);
};

}  // namespace ink2
