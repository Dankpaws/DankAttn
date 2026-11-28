// DankAttn (ERNIE-5.5): Metal 4 engine for baidu/ERNIE-5.5-Flash-Preview (W2-g64 experts, W8 dense).
// Supports local mmap packs AND zero-disk HTTP streaming packs (http://host:port) into unified RAM.
#pragma once
#include "../core/pack.h"
#include "../mtl.h"
#include "../core/mtl4.h"
#include "../mm_args.h"
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace ernie55 {

constexpr uint32_t MAXM = 16;        // rows per decode/verify step (mm2)
constexpr uint32_t PREFILL_M = 1024; // max rows per prefill chunk (mmp/mmpg)
constexpr uint32_t IDX_SUB = 16;     // sub-batch rows for DSA indexer scores (keeps idx_sc 16 MB)
constexpr uint32_t HCA_RING = 1152;  // >= 128 SWA + PREFILL_M
constexpr uint32_t COMP_RING = 256;  // persistent per-layer HCA compressor ring (2 blocks of 128)

using dankattn::Lin;
using dankattn::Raw;

struct Layer {
  bool is_dsa = false;
  bool is_moe = false;
  Raw ahc_fn, ahc_base, ahc_scale, mhc_fn, mhc_base, mhc_scale;
  Raw attn_norm, ffn_norm;
  Raw vha_m, sink;

  // HCA weights
  Lin attn_in, attn_qb, wo_a, wo_b;
  Raw q_norm, kv_norm, comp_ape, comp_norm;

  // DSA weights (reuses attn_in, attn_qb)
  Lin kb, vb, attn_o;
  Raw qa_norm, kva_norm, idx_kn_w, idx_kn_b;

  // Dense L0 FFN
  Lin mlp_gu, mlp_dn;

  // MoE L1..42
  Lin moe_in, moe_gu, moe_dn, moe_out;
  Raw moe_bias, moe_rscale, moe_lnorm;

  // KV / Compressor state
  id<MTLBuffer> hca_comp_cache = nil; // [cap / 128, 512] bf16
  id<MTLBuffer> hca_swa_kv = nil;     // [HCA_RING, 512] bf16
  id<MTLBuffer> hca_comp_kv = nil;    // [HCA_RING, 512] f32
  id<MTLBuffer> hca_comp_sc = nil;    // [HCA_RING, 512] f32

  id<MTLBuffer> dsa_lat_q8 = nil;     // [cap, 672 B]
  id<MTLBuffer> dsa_idx_k8 = nil;     // [cap, 136 B]
};

struct Model {
  mtl::Ctx C;
  std::string dir;
  uint32_t cap = 262144;
  uint32_t pos = 0;

  uint32_t H = 4096, HC = 4, nL = 43, vocab = 201216, E = 512, topk = 10;
  uint32_t I = 14336, moe_I = 2048, moe_lat = 2048;
  uint32_t nh = 64, hd = 512, nope = 448, rdim = 64, o_groups = 8, o_rank = 1024;
  uint32_t win = 128, idx_topk = 2048, sk_iters = 20;
  uint32_t bos = 1, eos = 2;
  float eps = 1e-6f, hc_eps = 1e-6f, situ_beta = 4.0f, situ_lbeta = 25.0f;
  bool hadamard = false;

  std::vector<id<MTLBuffer>> files;
  size_t mapped_bytes = 0;
  size_t state_bytes = 0;

  Raw embed, hch_fn, hch_base, hch_scale, norm, mm_r, mm_t;
  Lin lm_head;
  std::vector<Layer> L;

  // RoPE tables [cap, 32] f32
  id<MTLBuffer> cos_hca = nil, sin_hca = nil;
  id<MTLBuffer> cos_mla = nil, sin_mla = nil;

  // Scratch buffers (sized for MAXM=16)
  id<MTLBuffer> ids_buf = nil;
  id<MTLBuffer> h_buf = nil;       // [PREFILL_M, 4, H] f32
  id<MTLBuffer> h_tmp = nil;       // [4, 4, H] f32 (256 KB double-buffer for M <= 4 parallel mHC)
  id<MTLBuffer> post_buf = nil;    // [PREFILL_M, 4] f32
  id<MTLBuffer> comb_buf = nil;    // [MAXM, 16] f32
  id<MTLBuffer> x_f32 = nil;       // [MAXM, H] f32
  id<MTLBuffer> y_f32 = nil;       // [MAXM, H] f32
  id<MTLBuffer> frag_a = nil;      // general bf16 frag buffer (up to MAXM * 32768 bf16)
  id<MTLBuffer> frag_b = nil;      // second bf16 frag buffer (grouped: up to 64 * MAXM * 512 bf16)
  id<MTLBuffer> big_a = nil;       // f32 work buffer (up to MAXM * 40960 f32)
  id<MTLBuffer> big_b = nil;       // f32 work buffer (up to MAXM * 32768 f32)
  id<MTLBuffer> big_c = nil;       // f32 work buffer (up to MAXM * 16384 f32)
  id<MTLBuffer> q_rope = nil;      // [MAXM, 64, 64] f32
  id<MTLBuffer> iq_h = nil;        // [MAXM, 64, 128] f32
  id<MTLBuffer> idx_w = nil;       // [MAXM, 64] f32
  id<MTLBuffer> idx_sc = nil;      // [MAXM, cap] f32
  id<MTLBuffer> idx_sel = nil;     // [MAXM, 2176] int32
  id<MTLBuffer> idx_len = nil;     // [MAXM] uint32
  id<MTLBuffer> top_ids = nil;     // [MAXM * 10] uint32
  id<MTLBuffer> top_w = nil;       // [MAXM * 10] f32
  id<MTLBuffer> slots_buf = nil;   // [64] uint32 (0..63 for grouped wo_a / kb / vb)
  id<MTLBuffer> nslots_buf = nil;  // [1] uint32
  id<MTLBuffer> moe_slots = nil;   // [512] uint32 (active experts for mmpg)
  id<MTLBuffer> moe_nslots = nil;  // [1] uint32
  id<MTLBuffer> roff = nil;        // [512] uint32
  id<MTLBuffer> rcnt = nil;        // [512] uint32
  id<MTLBuffer> rlist = nil;       // [PREFILL_M * 10] uint32
  id<MTLBuffer> cidx = nil;        // [PREFILL_M * 10] uint32
  id<MTLBuffer> wtile = nil;       // [512 + PREFILL_M * 10 / 16 + 16] uint32
  id<MTLBuffer> moe_ind = nil;     // [6] uint32 (2 indirect dispatch structs)
  id<MTLBuffer> mmpart = nil;      // mm2 split-K partials & HCA temp o_out
  id<MTLBuffer> hcpart = nil;      // [PREFILL_M, 16, 34] f32 split-H mHC partials
  id<MTLBuffer> comp_kv_tmp = nil; // [PREFILL_M, 512] f32 shared prefill HCA comp scratch
  id<MTLBuffer> comp_sc_tmp = nil; // [PREFILL_M, 512] f32 shared prefill HCA comp scratch
  id<MTLBuffer> counters = nil;    // mm2 split-K counters
  id<MTLBuffer> logits_buf = nil;  // [MAXM, vocab] f32

  std::string mm2_src, e5_src;
  dankattn::Mtl4Runtime mtl4;
  std::unordered_map<std::string, id<MTLComputePipelineState>> e5_pcache;
  std::unordered_map<uint64_t, id<MTLComputePipelineState>> mm_pcache;

  Model(const std::string& pack_dir_or_url, uint32_t capacity = 262144);
  void reset() { pos = 0; }
  void truncate(uint32_t p) { pos = p; }

  // Run M <= PREFILL_M tokens starting at current pos.
  // If need_logits == false, skips final head + lm_head + multimax and returns nullptr.
  // If need_logits == true && last_only == true, computes logits only for token M-1 (at logits_buf[0..vocab-1]).
  // Otherwise (need_logits && !last_only), M must be <= MAXM and returns logits [M, vocab] f32.
  const float* step(const std::vector<uint32_t>& toks, double* gpu_ms = nullptr,
                    bool need_logits = true, bool last_only = false);

private:
  id<MTLComputePipelineState> mm1_pipe(uint32_t bits, uint32_t gs, bool grouped, uint32_t fuse = 0);
  id<MTLComputePipelineState> mm2_pipe(uint32_t bits, uint32_t gs, bool grouped, bool multi_slot, uint32_t epi = 0);
  id<MTLComputePipelineState> mmp_pipe(uint32_t bits, uint32_t gs, uint32_t epi = 0);
  id<MTLComputePipelineState> mmpg_pipe(uint32_t bits, uint32_t gs, bool contig_x, uint32_t ch,
                                        uint32_t epi = 0, uint32_t nf = 2);
  id<MTLComputePipelineState> e5_pipe(const std::string& fn);
  void dispatch_mm1(id<MTLComputeCommandEncoder> e, const Lin& w, id<MTLBuffer> X, id<MTLBuffer> Y,
                    id<MTLBuffer> sl, size_t sl_off, uint32_t nz, uint32_t xstride_u4, uint32_t ystride_f32,
                    uint32_t fuse = 0, id<MTLBuffer> extra_buf = nil, size_t extra_off = 0);
  void dispatch_mm(id<MTLComputeCommandEncoder> e, const Lin& w, id<MTLBuffer> X, id<MTLBuffer> Y,
                   uint32_t M, id<MTLBuffer> sl, uint32_t nz, uint32_t xstride_u4, uint32_t ystride_f32,
                   uint32_t epi = 0, id<MTLBuffer> E0 = nil, uint32_t R = 0,
                   id<MTLBuffer> E1 = nil, uint32_t epi_H = 0, uint32_t epi_HC = 0);
  void dispatch_mmpg_moe(id<MTLComputeCommandEncoder> e, const Lin& w, id<MTLBuffer> X, id<MTLBuffer> Y,
                         uint32_t M, uint32_t ldx, size_t ind_off,
                         uint32_t epi = 0, id<MTLBuffer> E0 = nil, uint32_t R = 0);
};

}  // namespace ernie55
