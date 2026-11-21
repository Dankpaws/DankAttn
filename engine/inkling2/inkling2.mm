// DankAttn (Inkling-2-Mini): model loader, kv8 global KV, Wave 2A forward step, rollback, and DSpark MTP drafter.
#import "inkling2.h"
#import <Foundation/Foundation.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ink2 {

namespace {

std::string read_text(const std::string& p) {
  std::ifstream f(p);
  if (!f) mtl::die(("cannot read " + p).c_str());
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

id<MTLBuffer> mmap_file(id<MTLDevice> dev, const std::string& path, size_t& out_bytes) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) mtl::die(("open " + path).c_str());
  struct stat st;
  if (fstat(fd, &st) != 0) mtl::die(("stat " + path).c_str());
  size_t sz = size_t(st.st_size);
  size_t pg = size_t(getpagesize());
  size_t msz = (sz + pg - 1) & ~(pg - 1);
  void* ptr = mmap(nullptr, msz, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) mtl::die(("mmap " + path).c_str());
  id<MTLBuffer> buf = [dev newBufferWithBytesNoCopy:ptr length:msz options:MTLResourceStorageModeShared
                                        deallocator:^(void* p, NSUInteger l) { munmap(p, l); }];
  if (!buf) mtl::die(("metal mmap buffer " + path).c_str());
  out_bytes += sz;
  return buf;
}

// Kernel argument structs matching engine/kernels/inkling2.metal
struct AttnResArgs { uint32_t H, M, n_prev, b_stride; float eps; };
struct MsgGateArgs { uint32_t H, R; float eps; };
struct QRelArgs {
  uint32_t nh, hd, d_rel, rel_extent, qgrs_n, pos0, is_global;
  float qk_clip, rel_clip, log_n_floor, log_alpha;
};
struct KvConvArgs {
  uint32_t M, nkv, hd, k_stride, pos0, cap, is_global, is_key, ks, kv8;
  float qk_clip, eps;
};
struct AttnArgs { uint32_t M, nh, nkv, hd, rel_extent, pos0, cap, window, is_local, n_splits, kv8; };
struct PostConvArgs { uint32_t M, H, ks, is_boundary, is_mlp, b_slot, b_stride; };
struct RouteArgs { uint32_t M, H, E, topk; float route_scale; };
struct MtpMixArgs { uint32_t H, M_stride, write_frag, write_f32; float eps; };
struct MtpKvArgs { uint32_t nkv, hd, pos0, cap; float eps, rope_theta; };
struct MtpQkvArgs { uint32_t nh, nkv, hd, pos0; float eps, rope_theta; };
struct MtpAttnArgs { uint32_t B, nh, nkv, hd, ctx_len, cap, window, is_local; };
struct RnnArgs { uint32_t H, R, step; };
struct MtpPickArgs { uint32_t V_valid, V_pad, step; float cap; };

void enc_mm2(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipe,
             const Lin& lin, id<MTLBuffer> X, size_t xoff, id<MTLBuffer> Y, size_t yoff,
             uint32_t M, id<MTLBuffer> E0, id<MTLBuffer> dummy,
             id<MTLBuffer> slots, id<MTLBuffer> nslots, id<MTLBuffer> rmask,
             bool grouped, uint32_t R_epi = 0, uint32_t xstride_u4 = 0, uint32_t ystride_f = 0) {
  MmArgs a{};
  a.N = lin.N;
  a.K = lin.K;
  a.M = M;
  a.ksplit = 1;
  a.sgs = lin.sgs;
  a.R = R_epi;
  a.pf_z0 = ~0u;
  if (grouped) {
    const uint32_t cb_exp = (lin.bits == 40 || lin.bits == 4) ? (lin.N * lin.K / 2)
                          : (lin.bits == 6) ? (lin.N * lin.K * 3 / 4) : (lin.N * lin.K);
    a.wstride = cb_exp / 16u;
    a.sstride = lin.N * (lin.K / lin.gs);
    a.xstride = xstride_u4;
    a.ystride = ystride_f;
    a.rm = 1;
  }
  const uint32_t nsg = ((lin.N / 32u) % 2u == 0u && lin.sgs <= 4u) ? (lin.sgs * 2u) : lin.sgs;
  const uint32_t nft = nsg / lin.sgs;
  const uint32_t gx = (lin.N / 32u + nft - 1u) / nft;
  const uint32_t gz = grouped ? std::min<uint32_t>(M * 8u, lin.experts) : 1u;

  [enc setComputePipelineState:pipe];
  [enc setBuffer:lin.w offset:lin.woff atIndex:0];
  [enc setBuffer:lin.s offset:lin.soff atIndex:1];
  [enc setBuffer:X offset:xoff atIndex:2];
  [enc setBuffer:dummy offset:0 atIndex:3];
  [enc setBuffer:Y offset:yoff atIndex:4];
  [enc setBytes:&a length:sizeof(a) atIndex:5];
  [enc setBuffer:(grouped ? slots : dummy) offset:0 atIndex:6];
  [enc setBuffer:(grouped ? nslots : dummy) offset:0 atIndex:7];
  [enc setBuffer:dummy offset:0 atIndex:8];
  [enc setBuffer:(E0 ? E0 : dummy) offset:0 atIndex:9];
  [enc setBuffer:dummy offset:0 atIndex:10];
  [enc setBuffer:dummy offset:0 atIndex:11];
  [enc setBuffer:dummy offset:0 atIndex:12];
  [enc setBuffer:dummy offset:0 atIndex:13];
  [enc setBuffer:dummy offset:0 atIndex:14];
  [enc setBuffer:(grouped ? rmask : dummy) offset:0 atIndex:15];
  [enc setThreadgroupMemoryLength:size_t(nsg) * 16u * 32u * sizeof(float) atIndex:0];
  [enc dispatchThreadgroups:MTLSizeMake(gx, 1, gz) threadsPerThreadgroup:MTLSizeMake(nsg * 32u, 1, 1)];
}

}  // namespace

void Model::load(const std::string& pack_dir, uint32_t max_cap, uint32_t passes_override, bool kv8) {
  dir = pack_dir;
  cap = max_cap;
  use_kv8 = kv8;

  NSData* jd = [NSData dataWithContentsOfFile:@((dir + "/manifest.json").c_str())];
  if (!jd) mtl::die(("cannot read " + dir + "/manifest.json").c_str());
  NSError* err = nil;
  NSDictionary* man = [NSJSONSerialization JSONObjectWithData:jd options:0 error:&err];
  if (!man) mtl::die("parse manifest.json", err);

  NSDictionary* tc = man[@"text_config"];
  H = [tc[@"hidden_size"] unsignedIntValue];
  nL = [tc[@"num_hidden_layers"] unsignedIntValue];
  vocab = [tc[@"vocab_size"] unsignedIntValue];
  vocab_valid = [tc[@"unpadded_vocab_size"] unsignedIntValue];
  vocab_pad = (vocab_valid + 31u) & ~31u;
  nh = [tc[@"num_attention_heads"] unsignedIntValue];
  nkv = [tc[@"num_key_value_heads"] unsignedIntValue];
  hd = [tc[@"head_dim"] unsignedIntValue];
  d_rel = [tc[@"d_rel"] unsignedIntValue];
  rel_extent = [tc[@"rel_extent"] unsignedIntValue];
  window = [tc[@"sliding_window_size"] unsignedIntValue];
  E = [tc[@"n_routed_experts"] unsignedIntValue];
  topk = [tc[@"num_experts_per_tok"] unsignedIntValue];
  I_moe = [tc[@"intermediate_size"] unsignedIntValue];
  ks = [tc[@"sconv_kernel_size"] unsignedIntValue];
  n_per_block = [tc[@"n_layers_per_attn_res_block"] unsignedIntValue];
  eps = [tc[@"rms_norm_eps"] floatValue];
  qk_clip = [tc[@"qk_l2_clip_threshold"] floatValue];
  rel_clip = [tc[@"rel_l2_clip_threshold"] floatValue];
  log_n_floor = [tc[@"log_scaling_n_floor"] floatValue];
  log_alpha = [tc[@"log_scaling_alpha"] floatValue];
  route_scale = [tc[@"route_scale"] floatValue];
  gate_softcap = [tc[@"mlp_gate_softcap"] floatValue];
  up_softcap = [tc[@"mlp_up_softcap"] floatValue];
  logit_softcap = [tc[@"final_logit_softcapping"] floatValue];
  eos = [man[@"eos_token_id"] unsignedIntValue];

  NSDictionary* lc = tc[@"loop_config"];
  core_start = [lc[@"core_start"] unsignedIntValue];
  core_end = [lc[@"core_end"] unsignedIntValue];
  num_passes = passes_override ? passes_override : [lc[@"num_passes"] unsignedIntValue];
  gate_rank = [lc[@"gate_rank"] unsignedIntValue];

  std::vector<bool> is_local_vec(nL, false);
  for (NSNumber* lid in tc[@"local_layer_ids"]) {
    uint32_t i = [lid unsignedIntValue];
    if (i < nL) is_local_vec[i] = true;
  }

  NSArray* flist = man[@"files"];
  files.reserve([flist count]);
  for (NSString* fn in flist) {
    files.push_back(mmap_file(C.dev, dir + "/" + [fn UTF8String], mapped_bytes));
  }

  auto load_raw_from = [&](NSDictionary* tens, const std::vector<id<MTLBuffer>>& fvec, const std::string& name) -> Raw {
    NSDictionary* d = tens[@(name.c_str())];
    if (!d) mtl::die(("missing raw tensor " + name).c_str());
    Raw r;
    r.b = fvec[[d[@"file"] unsignedIntValue]];
    r.off = [d[@"off"] unsignedLongLongValue];
    r.bytes = [d[@"bytes"] unsignedLongLongValue];
    return r;
  };
  auto load_lin_from = [&](NSDictionary* tens, const std::vector<id<MTLBuffer>>& fvec, const std::string& name) -> Lin {
    NSDictionary* d = tens[@(name.c_str())];
    if (!d) mtl::die(("missing linear tensor " + name).c_str());
    Lin ln;
    ln.w = fvec[[d[@"file"] unsignedIntValue]];
    ln.woff = [d[@"off"] unsignedLongLongValue];
    ln.s = fvec[[d[@"sfile"] unsignedIntValue]];
    ln.soff = [d[@"soff"] unsignedLongLongValue];
    ln.N = [d[@"N"] unsignedIntValue];
    ln.K = [d[@"K"] unsignedIntValue];
    ln.bits = [d[@"bits"] unsignedIntValue];
    ln.gs = [d[@"gs"] unsignedIntValue];
    ln.experts = [d[@"experts"] unsignedIntValue];
    const uint32_t units = (ln.bits == 40) ? (ln.K / 32u) : (ln.K / ln.gs);
    if (units % 4u == 0u) ln.sgs = 4;
    else if (units % 3u == 0u) ln.sgs = 3;
    else if (units % 2u == 0u) ln.sgs = 2;
    else ln.sgs = 1;
    return ln;
  };

  NSDictionary* tens = man[@"tensors"];
  auto get_raw = [&](const std::string& n) { return load_raw_from(tens, files, n); };
  auto get_lin = [&](const std::string& n) { return load_lin_from(tens, files, n); };

  embed = get_raw("embed");
  embed_norm = get_raw("embed_norm");
  norm = get_raw("norm");
  ar_out_q = get_raw("attn_res.output_query");
  loop_msg_q = get_raw("loop.msg_queries");
  lm_head = get_lin("lm_head");

  L.resize(nL);
  for (uint32_t l = 0; l < nL; l++) {
    auto& ly = L[l];
    std::string p = "L" + std::to_string(l);
    ly.is_local = is_local_vec[l];
    ly.is_loop = (l >= core_start && l < core_end);
    ly.is_boundary = (l % n_per_block) == 0;
    if (l > 0) ly.ar_attn_q = get_raw(p + ".ar.attn_q");
    ly.ar_mlp_q = get_raw(p + ".ar.mlp_q");
    if (ly.is_loop) {
      auto load_c = [&](const std::string& sub, Ctrl& c) {
        c.down = get_raw(p + ".ctrl." + sub + ".down");
        c.forget = get_raw(p + ".ctrl." + sub + ".forget");
        c.gain = get_raw(p + ".ctrl." + sub + ".gain");
      };
      load_c("q", ly.ctrl_q);
      load_c("k", ly.ctrl_k);
      load_c("v", ly.ctrl_v);
      load_c("mlp", ly.ctrl_mlp);
    }
    if (ly.is_local) {
      ly.attn_norm = get_raw(p + ".attn_norm");
    } else {
      ly.q_norm = get_raw(p + ".q_norm");
      ly.k_norm = get_raw(p + ".k_norm");
      ly.v_norm = get_raw(p + ".v_norm");
    }
    ly.mlp_norm = get_raw(p + ".mlp_norm");
    ly.qgrs = get_lin(p + ".attn.qgrs");
    ly.k_proj = get_lin(p + ".attn.k");
    ly.v_proj = get_lin(p + ".attn.v");
    ly.o_proj = get_lin(p + ".attn.o");
    ly.q_scale = get_raw(p + ".attn.q_scale");
    ly.rel_scale = get_raw(p + ".attn.rel_scale");
    ly.rel_proj = get_raw(p + ".attn.rel_proj");
    ly.k_sconv = get_raw(p + ".attn.k_sconv");
    ly.v_sconv = get_raw(p + ".attn.v_sconv");
    ly.attn_sconv = get_raw(p + ".attn_sconv");
    ly.mlp_sconv = get_raw(p + ".mlp_sconv");
    ly.router_w = get_raw(p + ".moe.router_w");
    ly.router_b = get_raw(p + ".moe.router_b");
    ly.moe_gu = get_lin(p + ".moe.gu");
    ly.moe_dn = get_lin(p + ".moe.dn");
  }

  // Allocate per-logical-layer KV + sconv + rollback history
  const uint32_t n_core = core_end - core_start;
  const uint32_t n_logical = nL + (num_passes - 1u) * n_core;
  S.resize(n_logical);
  auto init_logical = [&](uint32_t log_idx, uint32_t phys_idx) {
    auto& st = S[log_idx];
    st.phys_idx = phys_idx;
    st.cap = L[phys_idx].is_local ? (window + MAXM) : cap;
    st.kv8 = (!L[phys_idx].is_local) && use_kv8;
    if (st.kv8) {
      st.k_i8 = C.buf(size_t(st.cap) * nkv * hd);
      st.v_i8 = C.buf(size_t(st.cap) * nkv * hd);
      st.k_sc = C.buf(size_t(st.cap) * nkv * sizeof(uint16_t));
      st.v_sc = C.buf(size_t(st.cap) * nkv * sizeof(uint16_t));
    } else {
      size_t kv_bytes = size_t(st.cap) * nkv * hd * sizeof(uint16_t);
      st.k_cache = C.buf(kv_bytes);
      st.v_cache = C.buf(kv_bytes);
    }
    st.k_conv = C.buf(3u * nkv * hd * sizeof(float));
    st.v_conv = C.buf(3u * nkv * hd * sizeof(float));
    st.attn_conv = C.buf(3u * H * sizeof(float));
    st.mlp_conv = C.buf(3u * H * sizeof(float));
    st.k_hist = C.buf((3u + MAXM) * nkv * hd * sizeof(float));
    st.v_hist = C.buf((3u + MAXM) * nkv * hd * sizeof(float));
    st.attn_hist = C.buf((3u + MAXM) * H * sizeof(float));
    st.mlp_hist = C.buf((3u + MAXM) * H * sizeof(float));
  };
  uint32_t li = 0;
  for (uint32_t l = 0; l < core_start; l++) init_logical(li++, l);
  for (uint32_t p = 0; p < num_passes; p++)
    for (uint32_t l = core_start; l < core_end; l++) init_logical(li++, l);
  for (uint32_t l = core_end; l < nL; l++) init_logical(li++, l);

  // Allocate main scratch buffers
  b_ids = C.buf(MAXM * sizeof(uint32_t));
  b_h = C.buf(MAXM * H * sizeof(float));
  b_pre_h = C.buf(MAXM * H * sizeof(float));
  b_prev_h = C.buf(MAXM * H * sizeof(float));
  b_blocks = C.buf(MAX_BLOCKS * MAXM * H * sizeof(float));
  b_pre_blocks = C.buf(MAX_BLOCKS * MAXM * H * sizeof(float));
  b_prev_blocks = C.buf(MAX_BLOCKS * MAXM * H * sizeof(float));
  b_qkv_in = C.buf(3u * MAXM * H * sizeof(float));
  b_msgs = C.buf(4u * MAXM * H * sizeof(float));
  b_mh = C.buf(MAXM * H * sizeof(float));
  b_mn_f32 = C.buf(MAXM * H * sizeof(float));
  b_rscores = C.buf(MAXM * E * sizeof(float));
  b_mg_cat = C.buf(3u * MAXM * 2u * H * sizeof(uint16_t));
  b_mg_low = C.buf(3u * MAXM * gate_rank * sizeof(float));
  b_attn_part = C.buf(size_t(MAXM) * nh * ASPLITS * 130u * sizeof(float));

  b_frag_q = C.buf(MAXM * H * sizeof(uint16_t));
  b_frag_k = C.buf(MAXM * H * sizeof(uint16_t));
  b_frag_v = C.buf(MAXM * H * sizeof(uint16_t));
  b_frag_o = C.buf(MAXM * nh * hd * sizeof(uint16_t));
  b_frag_m = C.buf(MAXM * H * sizeof(uint16_t));
  b_frag_gu = C.buf(size_t(E) * MAXM * I_moe * sizeof(uint16_t));

  b_qgrs = C.buf(MAXM * L[0].qgrs.N * sizeof(float));
  b_k_raw = C.buf(MAXM * (nkv * hd + 32u) * sizeof(float));
  b_v_raw = C.buf(MAXM * (nkv * hd + 32u) * sizeof(float));
  b_Q = C.buf(MAXM * nh * hd * sizeof(float));
  b_rel_logits = C.buf(MAXM * nh * rel_extent * sizeof(float));
  b_gate = C.buf(MAXM * nh * hd * sizeof(float));
  b_sinks = C.buf(MAXM * nh * sizeof(float));
  b_o_out = C.buf(MAXM * H * sizeof(float));

  b_top_idx = C.buf(MAXM * topk * sizeof(uint32_t));
  b_top_w = C.buf(MAXM * topk * sizeof(float));
  b_slots = C.buf(E * sizeof(uint32_t));
  b_nslots = C.buf(16);
  b_rmask = C.buf(E * 4u * sizeof(uint32_t));
  b_moe_y = C.buf(size_t(E) * MAXM * H * sizeof(float));
  b_moe_out = C.buf(MAXM * H * sizeof(float));
  b_logits = C.buf(MAXM * vocab_pad * sizeof(float));
  b_tok_out = C.buf(MAXM * sizeof(uint32_t));
  b_dummy = C.buf(64);

  // Optional DSpark MTP drafter overlay
  NSData* mjd = [NSData dataWithContentsOfFile:@((dir + "/mtp_manifest.json").c_str())];
  if (mjd && num_passes == 3) {
    NSDictionary* mman = [NSJSONSerialization JSONObjectWithData:mjd options:0 error:&err];
    if (mman) {
      has_mtp = true;
      NSDictionary* mc = mman[@"mtp_config"];
      mtp_nL = [mc[@"num_hidden_layers"] unsignedIntValue];
      mtp_I = [mc[@"intermediate_size"] unsignedIntValue];
      mtp_block = [mc[@"block_size"] unsignedIntValue];
      mtp_rank = [mc[@"markov_rank"] unsignedIntValue];
      mtp_rope_theta = [mc[@"rope_theta"] floatValue];
      NSArray* tids = mc[@"target_layer_ids"];
      for (uint32_t i = 0; i < 4 && i < [tids count]; i++) mtp_taps[i] = [tids[i] unsignedIntValue];

      std::vector<bool> mtp_local(mtp_nL, false);
      for (NSNumber* lid in mc[@"local_layer_ids"]) {
        uint32_t i = [lid unsignedIntValue];
        if (i < mtp_nL) mtp_local[i] = true;
      }

      NSArray* mflist = mman[@"files"];
      for (NSString* fn in mflist) {
        mtp_files.push_back(mmap_file(C.dev, dir + "/" + [fn UTF8String], mapped_bytes));
      }
      NSDictionary* mtens = mman[@"tensors"];
      auto mraw = [&](const std::string& n) { return load_raw_from(mtens, mtp_files, n); };
      auto mlin = [&](const std::string& n) { return load_lin_from(mtens, mtp_files, n); };

      mtp_fuse_alpha = mraw("mtp.fuse_alpha");
      mtp_mask_embed = mraw("mtp.mask_embed");
      mtp_norm = mraw("mtp.norm");
      mtp_mh_dflare_norm = mraw("mtp.mh.dflare_norm");
      mtp_mh_markov_w1 = mraw("mtp.mh.markov_w1");
      mtp_mh_rnn_w = mraw("mtp.mh.rnn_proj.w");
      mtp_mh_rnn_b = mraw("mtp.mh.rnn_proj.b");
      mtp_mh_fold_w = mlin("mtp.mh.fold_w");
      mtp_mh_fold_b = mraw("mtp.mh.fold_b");

      MtpL.resize(mtp_nL);
      for (uint32_t l = 0; l < mtp_nL; l++) {
        auto& ml = MtpL[l];
        std::string p = "mtp.L" + std::to_string(l);
        ml.is_local = mtp_local[l];
        ml.cap = ml.is_local ? (window + MAXM) : cap;
        ml.dflare_norm = mraw(p + ".dflare_norm");
        ml.attn_norm = mraw(p + ".attn_norm");
        ml.mlp_norm = mraw(p + ".mlp_norm");
        ml.q_norm = mraw(p + ".q_norm");
        ml.k_norm = mraw(p + ".k_norm");
        ml.k_norm_ctx = mraw(p + ".k_norm_ctx");
        ml.qkv = mlin(p + ".attn.qkv");
        ml.kv_ctx = mlin(p + ".attn.kv_ctx");
        ml.o = mlin(p + ".attn.o");
        ml.mlp_gu = mlin(p + ".mlp.gu");
        ml.mlp_dn = mlin(p + ".mlp.dn");
        size_t kvb = size_t(ml.cap) * nkv * hd * sizeof(uint16_t);
        ml.k_cache = C.buf(kvb);
        ml.v_cache = C.buf(kvb);
      }

      b_taps = C.buf(4u * MAXM * H * sizeof(float));
      b_mtp_f_rnn = C.buf(MAXM * H * sizeof(float));
      b_mtp_h = C.buf(MAXM * H * sizeof(float));
      b_mtp_frag = C.buf(MAXM * H * sizeof(uint16_t));
      b_mtp_qkv = C.buf(MAXM * (nh * hd + 2u * nkv * hd) * sizeof(float));
      b_mtp_kv_ctx = C.buf(MAXM * 2u * nkv * hd * sizeof(float));
      b_mtp_Q = C.buf(MAXM * nh * hd * sizeof(float));
      b_mtp_K_blk = C.buf(MAXM * nkv * hd * sizeof(uint16_t));
      b_mtp_V_blk = C.buf(MAXM * nkv * hd * sizeof(uint16_t));
      b_mtp_o_frag = C.buf(MAXM * nh * hd * sizeof(uint16_t));
      b_mtp_o_out = C.buf(MAXM * H * sizeof(float));
      b_mtp_gu_frag = C.buf(MAXM * mtp_I * sizeof(uint16_t));
      b_mtp_dn_out = C.buf(MAXM * H * sizeof(float));
      b_mtp_base_h = C.buf(MAXM * H * sizeof(float));
      b_mtp_base_logits = C.buf(MAXM * vocab_pad * sizeof(float));
      b_mtp_bias = C.buf(vocab_pad * sizeof(float));
      b_mtp_rnn_state = C.buf(mtp_rank * sizeof(float));
      b_mtp_rnn_proj = C.buf(3u * mtp_rank * sizeof(float));
      b_mtp_lat_frag = C.buf(16u * mtp_rank * sizeof(uint16_t));
      b_mtp_prev_tok = C.buf(16);
      b_mtp_draft_out = C.buf(MAXM * sizeof(uint32_t));
    }
  }

  // Compile Metal 4 kernels
  std::string kdir = mtl::tree_root() + "/engine/kernels/";
  std::string ink_src = read_text(kdir + "inkling2.metal");
  std::string mm_src = read_text(kdir + "mm2.metal");

  p_embed = C.pipe(ink_src, "ink_embed");
  p_attn_res = C.pipe(ink_src, "ink_attn_res");
  p_mg_prep = C.pipe(ink_src, "ink_mg_prep");
  p_mg_down = C.pipe(ink_src, "ink_mg_down");
  p_mg_apply = C.pipe(ink_src, "ink_mg_apply");
  p_rms_frag = C.pipe(ink_src, "ink_rms_frag");
  p_q_rel = C.pipe(ink_src, "ink_q_rel");
  p_kv_conv = C.pipe(ink_src, "ink_kv_conv_cache");
  p_attn_split = C.pipe(ink_src, "ink_attn_split");
  p_attn_reduce = C.pipe(ink_src, "ink_attn_reduce");
  p_post_conv = C.pipe(ink_src, "ink_post_sconv_add");
  p_scores = C.pipe(ink_src, "ink_moe_scores");
  p_route = C.pipe(ink_src, "ink_moe_route");
  p_moe_accum = C.pipe(ink_src, "ink_moe_accum");
  p_softcap_argmax = C.pipe(ink_src, "ink_softcap_argmax");
  p_copy = C.pipe(ink_src, "ink_copy_f32");

  p_mm_w8 = C.pipe(mm_src, "mm2", {{"WBITS", "8"}, {"GS", "128"}, {"TILES", "1"}});
  p_moe_fp4_gu = C.pipe(mm_src, "mm2", {
      {"WBITS", "40"}, {"GS", "16"}, {"TILES", "1"}, {"GROUPED", "1"},
      {"EPI", "3"}, {"SWIGLU_SOFTCAP_G", "4.0"}, {"SWIGLU_SOFTCAP_U", "25.0"}});
  p_moe_fp4_dn = C.pipe(mm_src, "mm2", {
      {"WBITS", "40"}, {"GS", "16"}, {"TILES", "1"}, {"GROUPED", "1"}});
  p_moe_fp8_gu = C.pipe(mm_src, "mm2", {
      {"WBITS", "80"}, {"GS", "32"}, {"TILES", "1"}, {"GROUPED", "1"},
      {"EPI", "3"}, {"SWIGLU_SOFTCAP_G", "4.0"}, {"SWIGLU_SOFTCAP_U", "25.0"}});
  p_moe_fp8_dn = C.pipe(mm_src, "mm2", {
      {"WBITS", "80"}, {"GS", "32"}, {"TILES", "1"}, {"GROUPED", "1"}});

  if (has_mtp) {
    p_mtp_mix = C.pipe(ink_src, "ink_mtp_mix_norm");
    p_mtp_kv_ctx = C.pipe(ink_src, "ink_mtp_kv_ctx_cache");
    p_mtp_embed = C.pipe(ink_src, "ink_mtp_embed");
    p_mtp_qkv_rope = C.pipe(ink_src, "ink_mtp_qkv_rope");
    p_mtp_attn = C.pipe(ink_src, "ink_mtp_attn");
    p_add_f32 = C.pipe(ink_src, "ink_add_f32");
    p_mtp_rnn_proj = C.pipe(ink_src, "ink_mtp_rnn_proj");
    p_mtp_rnn_gates = C.pipe(ink_src, "ink_mtp_rnn_gates");
    p_mtp_add_argmax = C.pipe(ink_src, "ink_mtp_add_argmax");
    const std::string mb = std::to_string(MtpL[0].mlp_gu.bits);
    p_mm_w6_gu = C.pipe(mm_src, "mm2", {{"WBITS", mb}, {"GS", "128"}, {"TILES", "1"}, {"EPI", "3"}});
    p_mm_w6_dn = C.pipe(mm_src, "mm2", {{"WBITS", mb}, {"GS", "128"}, {"TILES", "1"}});
  }

  // Attach residency set for mapped weight shards + KV caches
  {
    MTLResidencySetDescriptor* rsd = [MTLResidencySetDescriptor new];
    rsd.label = @"inkling2-weights";
    NSError* rerr = nil;
    rs = [C.dev newResidencySetWithDescriptor:rsd error:&rerr];
    if (rs) {
      for (id<MTLBuffer> fb : files) [rs addAllocation:fb];
      for (id<MTLBuffer> fb : mtp_files) [rs addAllocation:fb];
      for (auto& st : S) {
        if (st.kv8) {
          [rs addAllocation:st.k_i8]; [rs addAllocation:st.v_i8];
          [rs addAllocation:st.k_sc]; [rs addAllocation:st.v_sc];
        } else {
          [rs addAllocation:st.k_cache]; [rs addAllocation:st.v_cache];
        }
      }
      for (auto& ml : MtpL) {
        [rs addAllocation:ml.k_cache]; [rs addAllocation:ml.v_cache];
      }
      [rs commit];
      [rs requestResidency];
      [C.q addResidencySet:rs];
    }
  }

  reset();
  fprintf(stderr, "inkling2: loaded %s (%.2f GB mapped, %.2f GB GPU state, %u phys -> %u logical, %u passes, cap=%u, kv8=%d, dspark=%d)\n",
          dir.c_str(), mapped_bytes / 1e9, C.allocated / 1e9, nL, n_logical, num_passes, cap, int(use_kv8), int(has_mtp));
}

void Model::reset() {
  pos = 0;
  mtp_ctx_pos = 0;
  for (auto& st : S) {
    memset([st.k_conv contents], 0, [st.k_conv length]);
    memset([st.v_conv contents], 0, [st.v_conv length]);
    memset([st.attn_conv contents], 0, [st.attn_conv length]);
    memset([st.mlp_conv contents], 0, [st.mlp_conv length]);
  }
}

void Model::rollback(uint32_t accepted, uint32_t M_step) {
  if (accepted >= M_step) return;
  pos -= (M_step - accepted);
  const size_t kv_row = size_t(nkv) * hd * sizeof(float);
  const size_t h_row = size_t(H) * sizeof(float);
  for (auto& st : S) {
    memcpy([st.k_conv contents], (const uint8_t*)[st.k_hist contents] + accepted * kv_row, 3u * kv_row);
    memcpy([st.v_conv contents], (const uint8_t*)[st.v_hist contents] + accepted * kv_row, 3u * kv_row);
    memcpy([st.attn_conv contents], (const uint8_t*)[st.attn_hist contents] + accepted * h_row, 3u * h_row);
    memcpy([st.mlp_conv contents], (const uint8_t*)[st.mlp_hist contents] + accepted * h_row, 3u * h_row);
  }
}

std::vector<uint32_t> Model::step(const std::vector<uint32_t>& ids, double* gpu_ms,
                                  std::vector<float>* last_logits,
                                  std::map<std::string, double>* prof) {
  const uint32_t M = uint32_t(ids.size());
  if (M == 0 || M > MAXM) mtl::die("step: M must be in 1..16");
  memcpy([b_ids contents], ids.data(), M * sizeof(uint32_t));

  id<MTLCommandBuffer> cb = [C.q commandBuffer];
  const bool conc = (prof == nullptr);
  auto new_enc = [&]() -> id<MTLComputeCommandEncoder> {
    return conc ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                : [cb computeCommandEncoder];
  };
  id<MTLComputeCommandEncoder> enc = new_enc();
  auto barrier = [&]() {
    if (conc) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
  };

  auto flush_prof = [&](const char* tag) {
    if (!prof) return;
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) mtl::die("GPU step error", cb.error);
    (*prof)[tag] += (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
    cb = [C.q commandBuffer];
    enc = new_enc();
  };

  auto copy_f32 = [&](id<MTLBuffer> src, size_t soff, id<MTLBuffer> dst, size_t doff, size_t bytes) {
    const uint32_t n4 = uint32_t(bytes / 16u);
    [enc setComputePipelineState:p_copy];
    [enc setBuffer:src offset:soff atIndex:0];
    [enc setBuffer:dst offset:doff atIndex:1];
    [enc setBytes:&n4 length:4 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((n4 + 255u) / 256u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  };

  const uint32_t b_stride = MAXM * H;
  const uint32_t n_core = core_end - core_start;

  // 1. Embedding + embed_norm -> b_h
  [enc setComputePipelineState:p_embed];
  [enc setBuffer:embed.b offset:embed.off atIndex:0];
  [enc setBuffer:embed_norm.b offset:embed_norm.off atIndex:1];
  [enc setBuffer:b_ids offset:0 atIndex:2];
  [enc setBuffer:b_h offset:0 atIndex:3];
  [enc setBytes:&H length:4 atIndex:4];
  [enc setBytes:&eps length:4 atIndex:5];
  [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  barrier();

  auto maybe_capture = [&](uint32_t log_idx) {
    if (!has_mtp) return;
    for (uint32_t ti = 0; ti < 3u; ti++) {
      if (log_idx == mtp_taps[ti]) {
        copy_f32(b_h, 0, b_taps, size_t(ti) * MAXM * H * sizeof(float), size_t(M) * H * sizeof(float));
        barrier();
      }
    }
  };

  auto run_layer = [&](uint32_t log_idx, uint32_t phys_idx, uint32_t& n_blocks, bool has_msgs) {
    maybe_capture(log_idx);
    const auto& ly = L[phys_idx];
    const auto& st = S[log_idx];

    // A. Pre-attention Block Attention Residual -> b_qkv_in
    id<MTLBuffer> q_in_buf = b_qkv_in, k_in_buf = b_qkv_in, v_in_buf = b_qkv_in;
    size_t q_in_off = 0, k_in_off = 0, v_in_off = 0;

    if (phys_idx == 0) {
      q_in_buf = k_in_buf = v_in_buf = b_h;
    } else {
      const uint32_t nq = ly.is_local ? 1u : 3u;
      AttnResArgs ara{H, M, n_blocks, b_stride, eps};
      [enc setComputePipelineState:p_attn_res];
      [enc setBuffer:b_blocks offset:0 atIndex:0];
      [enc setBuffer:b_h offset:0 atIndex:1];
      [enc setBuffer:ly.ar_attn_q.b offset:ly.ar_attn_q.off atIndex:2];
      [enc setBuffer:b_qkv_in offset:0 atIndex:3];
      [enc setBytes:&ara length:sizeof(ara) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake(M, nq, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      if (!ly.is_local) {
        k_in_off = size_t(M) * H * sizeof(float);
        v_in_off = 2u * size_t(M) * H * sizeof(float);
      }
      barrier();
      flush_prof("attn_res");
    }

    // B. Cross-pass message gates (passes >= 1 of core)
    if (has_msgs) {
      MsgGateArgs mga{H, gate_rank, eps};
      const size_t stride_m = size_t(M) * H * sizeof(float);
      const size_t cat_stride = size_t(MAXM) * 2u * H * sizeof(uint16_t);
      const size_t low_stride = size_t(MAXM) * gate_rank * sizeof(float);

      auto mg_prep = [&](uint32_t slot, id<MTLBuffer> hin, size_t hoff, size_t moff) {
        [enc setComputePipelineState:p_mg_prep];
        [enc setBuffer:hin offset:hoff atIndex:0];
        [enc setBuffer:b_msgs offset:moff atIndex:1];
        [enc setBuffer:b_mg_cat offset:(slot * cat_stride) atIndex:2];
        [enc setBytes:&mga length:sizeof(mga) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      };
      auto mg_down = [&](uint32_t slot, const Ctrl& c) {
        [enc setComputePipelineState:p_mg_down];
        [enc setBuffer:b_mg_cat offset:(slot * cat_stride) atIndex:0];
        [enc setBuffer:c.down.b offset:c.down.off atIndex:1];
        [enc setBuffer:b_mg_low offset:(slot * low_stride) atIndex:2];
        [enc setBytes:&mga length:sizeof(mga) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(M, gate_rank / 4u, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      };
      auto mg_apply = [&](uint32_t slot, id<MTLBuffer> hin, size_t hoff, size_t moff, const Ctrl& c, size_t out_off) {
        [enc setComputePipelineState:p_mg_apply];
        [enc setBuffer:hin offset:hoff atIndex:0];
        [enc setBuffer:b_msgs offset:moff atIndex:1];
        [enc setBuffer:b_mg_low offset:(slot * low_stride) atIndex:2];
        [enc setBuffer:c.forget.b offset:c.forget.off atIndex:3];
        [enc setBuffer:c.gain.b offset:c.gain.off atIndex:4];
        [enc setBuffer:b_qkv_in offset:out_off atIndex:5];
        [enc setBytes:&mga length:sizeof(mga) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(M, (H + 63u) / 64u, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      };

      mg_prep(0, q_in_buf, q_in_off, 0u * stride_m);
      mg_prep(1, k_in_buf, k_in_off, 1u * stride_m);
      mg_prep(2, v_in_buf, v_in_off, 2u * stride_m);
      barrier();
      mg_down(0, ly.ctrl_q);
      mg_down(1, ly.ctrl_k);
      mg_down(2, ly.ctrl_v);
      barrier();
      mg_apply(1, k_in_buf, k_in_off, 1u * stride_m, ly.ctrl_k, 1u * stride_m);
      mg_apply(2, v_in_buf, v_in_off, 2u * stride_m, ly.ctrl_v, 2u * stride_m);
      if (ly.is_local) barrier();
      mg_apply(0, q_in_buf, q_in_off, 0u * stride_m, ly.ctrl_q, 0u * stride_m);
      barrier();

      q_in_buf = k_in_buf = v_in_buf = b_qkv_in;
      q_in_off = 0;
      k_in_off = 1u * stride_m;
      v_in_off = 2u * stride_m;
      flush_prof("msg_gate");
    }

    // C. Input RMSNorm -> bf16 fragments
    const uint32_t zero_u = 0;
    auto enc_rms = [&](id<MTLBuffer> x, size_t xoff, const Raw& w, id<MTLBuffer> frag,
                       id<MTLBuffer> f32o, uint32_t wf32) {
      [enc setComputePipelineState:p_rms_frag];
      [enc setBuffer:x offset:xoff atIndex:0];
      [enc setBuffer:w.b offset:w.off atIndex:1];
      [enc setBuffer:frag offset:0 atIndex:2];
      [enc setBuffer:(f32o ? f32o : b_dummy) offset:0 atIndex:3];
      [enc setBytes:&H length:4 atIndex:4];
      [enc setBytes:&eps length:4 atIndex:5];
      [enc setBytes:&wf32 length:4 atIndex:6];
      [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };

    id<MTLBuffer> fk = b_frag_q, fv = b_frag_q;
    if (ly.is_local && !has_msgs) {
      enc_rms(q_in_buf, q_in_off, ly.attn_norm, b_frag_q, nil, zero_u);
    } else {
      enc_rms(q_in_buf, q_in_off, ly.is_local ? ly.attn_norm : ly.q_norm, b_frag_q, nil, zero_u);
      enc_rms(k_in_buf, k_in_off, ly.is_local ? ly.attn_norm : ly.k_norm, b_frag_k, nil, zero_u);
      enc_rms(v_in_buf, v_in_off, ly.is_local ? ly.attn_norm : ly.v_norm, b_frag_v, nil, zero_u);
      fk = b_frag_k;
      fv = b_frag_v;
    }
    barrier();
    flush_prof("rms");

    // D. Attention projections (W8): qgrs || k || v concurrently
    enc_mm2(enc, p_mm_w8, ly.qgrs, b_frag_q, 0, b_qgrs, 0, M, nil, b_dummy, nil, nil, nil, false);
    if (prof) flush_prof("attn_qgrs");
    enc_mm2(enc, p_mm_w8, ly.k_proj, fk, 0, b_k_raw, 0, M, nil, b_dummy, nil, nil, nil, false);
    enc_mm2(enc, p_mm_w8, ly.v_proj, fv, 0, b_v_raw, 0, M, nil, b_dummy, nil, nil, nil, false);
    barrier();
    flush_prof("attn_kv");

    // E + F. Q+Rel || K sconv || V sconv concurrently
    QRelArgs qra{nh, hd, d_rel, rel_extent, ly.qgrs.N, pos, ly.is_local ? 0u : 1u,
                 qk_clip, rel_clip, log_n_floor, log_alpha};
    [enc setComputePipelineState:p_q_rel];
    [enc setBuffer:b_qgrs offset:0 atIndex:0];
    [enc setBuffer:ly.q_scale.b offset:ly.q_scale.off atIndex:1];
    [enc setBuffer:ly.rel_scale.b offset:ly.rel_scale.off atIndex:2];
    [enc setBuffer:ly.rel_proj.b offset:ly.rel_proj.off atIndex:3];
    [enc setBuffer:b_Q offset:0 atIndex:4];
    [enc setBuffer:b_rel_logits offset:0 atIndex:5];
    [enc setBuffer:b_gate offset:0 atIndex:6];
    [enc setBuffer:b_sinks offset:0 atIndex:7];
    [enc setBytes:&qra length:sizeof(qra) atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake(M, nh, 1) threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
    if (prof) flush_prof("q_rel");

    KvConvArgs kca{M, nkv, hd, ly.k_proj.N, pos, st.cap, ly.is_local ? 0u : 1u, 1u, ks, st.kv8 ? 1u : 0u, qk_clip, eps};
    [enc setComputePipelineState:p_kv_conv];
    [enc setBuffer:b_k_raw offset:0 atIndex:0];
    [enc setBuffer:ly.k_sconv.b offset:ly.k_sconv.off atIndex:1];
    [enc setBuffer:st.k_conv offset:0 atIndex:2];
    [enc setBuffer:(st.k_cache ? st.k_cache : b_dummy) offset:0 atIndex:3];
    [enc setBuffer:(st.k_i8 ? st.k_i8 : b_dummy) offset:0 atIndex:4];
    [enc setBuffer:(st.k_sc ? st.k_sc : b_dummy) offset:0 atIndex:5];
    [enc setBuffer:st.k_hist offset:0 atIndex:6];
    [enc setBytes:&kca length:sizeof(kca) atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake(nkv, 1, 1) threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];

    KvConvArgs vca{M, nkv, hd, ly.v_proj.N, pos, st.cap, ly.is_local ? 0u : 1u, 0u, ks, st.kv8 ? 1u : 0u, qk_clip, eps};
    [enc setBuffer:b_v_raw offset:0 atIndex:0];
    [enc setBuffer:ly.v_sconv.b offset:ly.v_sconv.off atIndex:1];
    [enc setBuffer:st.v_conv offset:0 atIndex:2];
    [enc setBuffer:(st.v_cache ? st.v_cache : b_dummy) offset:0 atIndex:3];
    [enc setBuffer:(st.v_i8 ? st.v_i8 : b_dummy) offset:0 atIndex:4];
    [enc setBuffer:(st.v_sc ? st.v_sc : b_dummy) offset:0 atIndex:5];
    [enc setBuffer:st.v_hist offset:0 atIndex:6];
    [enc setBytes:&vca length:sizeof(vca) atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake(nkv, 1, 1) threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
    barrier();
    flush_prof("kv_conv");

    // G. Barrier-free Simdgroup Split-K GQA Attention (BF16 or kv8)
    AttnArgs aa{M, nh, nkv, hd, rel_extent, pos, st.cap, window, ly.is_local ? 1u : 0u, ASPLITS, st.kv8 ? 1u : 0u};
    [enc setComputePipelineState:p_attn_split];
    [enc setBuffer:b_Q offset:0 atIndex:0];
    [enc setBuffer:(st.k_cache ? st.k_cache : b_dummy) offset:0 atIndex:1];
    [enc setBuffer:(st.v_cache ? st.v_cache : b_dummy) offset:0 atIndex:2];
    [enc setBuffer:b_rel_logits offset:0 atIndex:3];
    [enc setBuffer:b_attn_part offset:0 atIndex:4];
    [enc setBuffer:(st.k_i8 ? st.k_i8 : b_dummy) offset:0 atIndex:5];
    [enc setBuffer:(st.k_sc ? st.k_sc : b_dummy) offset:0 atIndex:6];
    [enc setBuffer:(st.v_i8 ? st.v_i8 : b_dummy) offset:0 atIndex:7];
    [enc setBuffer:(st.v_sc ? st.v_sc : b_dummy) offset:0 atIndex:8];
    [enc setBytes:&aa length:sizeof(aa) atIndex:9];
    [enc dispatchThreadgroups:MTLSizeMake(M, nh, ASPLITS) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    barrier();

    [enc setComputePipelineState:p_attn_reduce];
    [enc setBuffer:b_attn_part offset:0 atIndex:0];
    [enc setBuffer:b_gate offset:0 atIndex:1];
    [enc setBuffer:b_sinks offset:0 atIndex:2];
    [enc setBuffer:b_frag_o offset:0 atIndex:3];
    [enc setBytes:&aa length:sizeof(aa) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(M, nh, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    barrier();
    flush_prof("attn");

    enc_mm2(enc, p_mm_w8, ly.o_proj, b_frag_o, 0, b_o_out, 0, M, nil, b_dummy, nil, nil, nil, false);
    barrier();
    flush_prof("attn_o");

    // H. Post-attention sconv + block boundary / residual add
    PostConvArgs pca{M, H, ks, ly.is_boundary ? 1u : 0u, 0u, n_blocks, b_stride};
    [enc setComputePipelineState:p_post_conv];
    [enc setBuffer:b_o_out offset:0 atIndex:0];
    [enc setBuffer:ly.attn_sconv.b offset:ly.attn_sconv.off atIndex:1];
    [enc setBuffer:st.attn_conv offset:0 atIndex:2];
    [enc setBuffer:b_h offset:0 atIndex:3];
    [enc setBuffer:b_blocks offset:0 atIndex:4];
    [enc setBuffer:st.attn_hist offset:0 atIndex:5];
    [enc setBytes:&pca length:sizeof(pca) atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake((H + 255u) / 256u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    if (ly.is_boundary) n_blocks++;
    barrier();
    flush_prof("post_conv");

    // I. Pre-MLP Block Attention Residual + optional 3-stage parallel message gate + mlp_norm
    AttnResArgs mara{H, M, n_blocks, b_stride, eps};
    [enc setComputePipelineState:p_attn_res];
    [enc setBuffer:b_blocks offset:0 atIndex:0];
    [enc setBuffer:b_h offset:0 atIndex:1];
    [enc setBuffer:ly.ar_mlp_q.b offset:ly.ar_mlp_q.off atIndex:2];
    [enc setBuffer:b_mh offset:0 atIndex:3];
    [enc setBytes:&mara length:sizeof(mara) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();
    flush_prof("attn_res");

    if (has_msgs) {
      MsgGateArgs mga{H, gate_rank, eps};
      const size_t m_off = 3u * size_t(M) * H * sizeof(float);
      [enc setComputePipelineState:p_mg_prep];
      [enc setBuffer:b_mh offset:0 atIndex:0];
      [enc setBuffer:b_msgs offset:m_off atIndex:1];
      [enc setBuffer:b_mg_cat offset:0 atIndex:2];
      [enc setBytes:&mga length:sizeof(mga) atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      barrier();

      [enc setComputePipelineState:p_mg_down];
      [enc setBuffer:b_mg_cat offset:0 atIndex:0];
      [enc setBuffer:ly.ctrl_mlp.down.b offset:ly.ctrl_mlp.down.off atIndex:1];
      [enc setBuffer:b_mg_low offset:0 atIndex:2];
      [enc setBytes:&mga length:sizeof(mga) atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake(M, gate_rank / 4u, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      barrier();

      [enc setComputePipelineState:p_mg_apply];
      [enc setBuffer:b_mh offset:0 atIndex:0];
      [enc setBuffer:b_msgs offset:m_off atIndex:1];
      [enc setBuffer:b_mg_low offset:0 atIndex:2];
      [enc setBuffer:ly.ctrl_mlp.forget.b offset:ly.ctrl_mlp.forget.off atIndex:3];
      [enc setBuffer:ly.ctrl_mlp.gain.b offset:ly.ctrl_mlp.gain.off atIndex:4];
      [enc setBuffer:b_mh offset:0 atIndex:5];
      [enc setBytes:&mga length:sizeof(mga) atIndex:6];
      [enc dispatchThreadgroups:MTLSizeMake(M, (H + 63u) / 64u, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      barrier();
      flush_prof("msg_gate");
    }

    enc_rms(b_mh, 0, ly.mlp_norm, b_frag_m, b_mn_f32, 1u);
    barrier();
    flush_prof("rms");

    // J. MoE Router + Grouped NVFP4/MXFP8 Experts + Accum
    [enc setComputePipelineState:p_scores];
    [enc setBuffer:b_mn_f32 offset:0 atIndex:0];
    [enc setBuffer:ly.router_w.b offset:ly.router_w.off atIndex:1];
    [enc setBuffer:b_rscores offset:0 atIndex:2];
    [enc setBytes:&H length:4 atIndex:3];
    [enc setBytes:&E length:4 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(M, E / 8u, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();

    RouteArgs ra{M, H, E, topk, route_scale};
    [enc setComputePipelineState:p_route];
    [enc setBuffer:b_rscores offset:0 atIndex:0];
    [enc setBuffer:ly.router_b.b offset:ly.router_b.off atIndex:1];
    [enc setBuffer:b_top_idx offset:0 atIndex:2];
    [enc setBuffer:b_top_w offset:0 atIndex:3];
    [enc setBuffer:b_slots offset:0 atIndex:4];
    [enc setBuffer:b_nslots offset:0 atIndex:5];
    [enc setBuffer:b_rmask offset:0 atIndex:6];
    [enc setBytes:&ra length:sizeof(ra) atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(E, 1, 1)];
    barrier();
    flush_prof("moe_route");

    const bool is_fp4 = (ly.moe_gu.bits == 40);
    enc_mm2(enc, is_fp4 ? p_moe_fp4_gu : p_moe_fp8_gu, ly.moe_gu, b_frag_m, 0, b_dummy, 0, M,
            b_frag_gu, b_dummy, b_slots, b_nslots, b_rmask, true, I_moe, 0u, 0u);
    barrier();
    flush_prof("moe_gu");
    const uint32_t xstride_dn = (MAXM * I_moe * sizeof(uint16_t)) / 16u;
    const uint32_t ystride_dn = MAXM * H;
    enc_mm2(enc, is_fp4 ? p_moe_fp4_dn : p_moe_fp8_dn, ly.moe_dn, b_frag_gu, 0, b_moe_y, 0, M,
            nil, b_dummy, b_slots, b_nslots, b_rmask, true, 0u, xstride_dn, ystride_dn);
    barrier();
    flush_prof("moe_dn");

    [enc setComputePipelineState:p_moe_accum];
    [enc setBuffer:b_moe_y offset:0 atIndex:0];
    [enc setBuffer:b_slots offset:0 atIndex:1];
    [enc setBuffer:b_nslots offset:0 atIndex:2];
    [enc setBuffer:b_top_idx offset:0 atIndex:3];
    [enc setBuffer:b_top_w offset:0 atIndex:4];
    [enc setBuffer:b_moe_out offset:0 atIndex:5];
    [enc setBytes:&H length:4 atIndex:6];
    [enc setBytes:&ystride_dn length:4 atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake(M, (H + 255u) / 256u, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();
    flush_prof("moe_accum");

    // K. Post-MLP sconv + residual add -> b_h (bf16-rounded)
    PostConvArgs mpca{M, H, ks, 0u, 1u, 0u, b_stride};
    [enc setComputePipelineState:p_post_conv];
    [enc setBuffer:b_moe_out offset:0 atIndex:0];
    [enc setBuffer:ly.mlp_sconv.b offset:ly.mlp_sconv.off atIndex:1];
    [enc setBuffer:st.mlp_conv offset:0 atIndex:2];
    [enc setBuffer:b_h offset:0 atIndex:3];
    [enc setBuffer:b_blocks offset:0 atIndex:4];
    [enc setBuffer:st.mlp_hist offset:0 atIndex:5];
    [enc setBytes:&mpca length:sizeof(mpca) atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake((H + 255u) / 256u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();
    flush_prof("post_conv");
  };

  uint32_t n_blocks = 0;
  uint32_t log_idx = 0;

  // 2. Prelude
  for (uint32_t l = 0; l < core_start; l++) run_layer(log_idx++, l, n_blocks, false);

  const uint32_t n_pre_blocks = n_blocks;
  copy_f32(b_h, 0, b_pre_h, 0, size_t(M) * H * sizeof(float));
  if (n_pre_blocks > 0) copy_f32(b_blocks, 0, b_pre_blocks, 0, size_t(n_pre_blocks) * b_stride * sizeof(float));
  barrier();

  // 3. Pass 0 over core
  for (uint32_t l = core_start; l < core_end; l++) run_layer(log_idx++, l, n_blocks, false);

  // 4. Passes 1 .. num_passes - 1
  for (uint32_t p = 1; p < num_passes; p++) {
    const uint32_t n_prev_bs = (n_blocks > n_pre_blocks + 1u) ? (n_blocks - (n_pre_blocks + 1u)) : 0u;
    copy_f32(b_h, 0, b_prev_h, 0, size_t(M) * H * sizeof(float));
    if (n_prev_bs > 0) {
      copy_f32(b_blocks, size_t(n_pre_blocks + 1u) * b_stride * sizeof(float),
               b_prev_blocks, 0, size_t(n_prev_bs) * b_stride * sizeof(float));
    }
    barrier();
    copy_f32(b_pre_h, 0, b_h, 0, size_t(M) * H * sizeof(float));
    if (n_pre_blocks > 0) copy_f32(b_pre_blocks, 0, b_blocks, 0, size_t(n_pre_blocks) * b_stride * sizeof(float));
    barrier();
    n_blocks = n_pre_blocks;

    for (uint32_t li = 0; li < n_core; li++) {
      AttnResArgs mara{H, M, n_prev_bs, b_stride, eps};
      [enc setComputePipelineState:p_attn_res];
      [enc setBuffer:b_prev_blocks offset:0 atIndex:0];
      [enc setBuffer:b_prev_h offset:0 atIndex:1];
      [enc setBuffer:loop_msg_q.b offset:(loop_msg_q.off + size_t(li) * 4u * H * sizeof(float)) atIndex:2];
      [enc setBuffer:b_msgs offset:0 atIndex:3];
      [enc setBytes:&mara length:sizeof(mara) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake(M, 4, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      barrier();
      run_layer(log_idx++, core_start + li, n_blocks, true);
    }
  }

  // 5. Coda
  for (uint32_t l = core_end; l < nL; l++) run_layer(log_idx++, l, n_blocks, false);

  // 6. Final AttnRes + Norm (+ capture tap 3 = final normalized hidden) + lm_head + softcap + argmax
  AttnResArgs fara{H, M, n_blocks, b_stride, eps};
  [enc setComputePipelineState:p_attn_res];
  [enc setBuffer:b_blocks offset:0 atIndex:0];
  [enc setBuffer:b_h offset:0 atIndex:1];
  [enc setBuffer:ar_out_q.b offset:ar_out_q.off atIndex:2];
  [enc setBuffer:b_mh offset:0 atIndex:3];
  [enc setBytes:&fara length:sizeof(fara) atIndex:4];
  [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  barrier();

  const uint32_t wf32_final = has_mtp ? 1u : 0u;
  [enc setComputePipelineState:p_rms_frag];
  [enc setBuffer:b_mh offset:0 atIndex:0];
  [enc setBuffer:norm.b offset:norm.off atIndex:1];
  [enc setBuffer:b_frag_m offset:0 atIndex:2];
  [enc setBuffer:(has_mtp ? b_taps : b_dummy) offset:(has_mtp ? (3u * size_t(MAXM) * H * sizeof(float)) : 0) atIndex:3];
  [enc setBytes:&H length:4 atIndex:4];
  [enc setBytes:&eps length:4 atIndex:5];
  [enc setBytes:&wf32_final length:4 atIndex:6];
  [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  barrier();

  enc_mm2(enc, p_mm_w8, lm_head, b_frag_m, 0, b_logits, 0, M, nil, b_dummy, nil, nil, nil, false);
  barrier();
  flush_prof("lm_head");

  [enc setComputePipelineState:p_softcap_argmax];
  [enc setBuffer:b_logits offset:0 atIndex:0];
  [enc setBuffer:b_tok_out offset:0 atIndex:1];
  [enc setBytes:&vocab_valid length:4 atIndex:2];
  [enc setBytes:&vocab_pad length:4 atIndex:3];
  [enc setBytes:&logit_softcap length:4 atIndex:4];
  [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  flush_prof("argmax");

  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.error) mtl::die("GPU step error", cb.error);
  if (gpu_ms) *gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;

  pos += M;
  std::vector<uint32_t> out(M);
  memcpy(out.data(), [b_tok_out contents], M * sizeof(uint32_t));
  if (last_logits) {
    last_logits->resize(vocab_valid);
    const float* lp = (const float*)[b_logits contents] + size_t(M - 1u) * vocab_pad;
    memcpy(last_logits->data(), lp, vocab_valid * sizeof(float));
  }
  return out;
}

void Model::mtp_commit(uint32_t n_commit, double* gpu_ms) {
  if (!has_mtp || n_commit == 0) return;
  id<MTLCommandBuffer> cb = [C.q commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
  auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };

  // 1. Project & cache context KV for each of the 3 MTP layers
  for (uint32_t l = 0; l < mtp_nL; l++) {
    auto& ml = MtpL[l];
    MtpMixArgs ma{H, MAXM, 1u, 0u, eps};
    [enc setComputePipelineState:p_mtp_mix];
    [enc setBuffer:b_taps offset:0 atIndex:0];
    [enc setBuffer:mtp_fuse_alpha.b offset:(mtp_fuse_alpha.off + size_t(l) * 4u * sizeof(float)) atIndex:1];
    [enc setBuffer:ml.dflare_norm.b offset:ml.dflare_norm.off atIndex:2];
    [enc setBuffer:b_mtp_frag offset:0 atIndex:3];
    [enc setBuffer:b_dummy offset:0 atIndex:4];
    [enc setBytes:&ma length:sizeof(ma) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(n_commit, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();

    enc_mm2(enc, p_mm_w8, ml.kv_ctx, b_mtp_frag, 0, b_mtp_kv_ctx, 0, n_commit, nil, b_dummy, nil, nil, nil, false);
    barrier();

    MtpKvArgs ka{nkv, hd, mtp_ctx_pos, ml.cap, eps, mtp_rope_theta};
    [enc setComputePipelineState:p_mtp_kv_ctx];
    [enc setBuffer:b_mtp_kv_ctx offset:0 atIndex:0];
    [enc setBuffer:ml.k_norm_ctx.b offset:ml.k_norm_ctx.off atIndex:1];
    [enc setBuffer:ml.k_cache offset:0 atIndex:2];
    [enc setBuffer:ml.v_cache offset:0 atIndex:3];
    [enc setBytes:&ka length:sizeof(ka) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(n_commit, nkv, 1) threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
    barrier();
  }

  // 2. Compute f_rnn for the last committed token (row = n_commit - 1)
  MtpMixArgs rma{H, MAXM, 0u, 1u, eps};
  [enc setComputePipelineState:p_mtp_mix];
  [enc setBuffer:b_taps offset:0 atIndex:0];
  [enc setBuffer:mtp_fuse_alpha.b offset:(mtp_fuse_alpha.off + 3u * 4u * sizeof(float)) atIndex:1];
  [enc setBuffer:mtp_mh_dflare_norm.b offset:mtp_mh_dflare_norm.off atIndex:2];
  [enc setBuffer:b_dummy offset:0 atIndex:3];
  [enc setBuffer:b_mtp_f_rnn offset:0 atIndex:4];
  [enc setBytes:&rma length:sizeof(rma) atIndex:5];
  [enc dispatchThreadgroups:MTLSizeMake(n_commit, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.error) mtl::die("GPU mtp_commit error", cb.error);
  if (gpu_ms) *gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;

  // Move row (n_commit - 1) of b_mtp_f_rnn to row 0 so draft() always reads offset 0
  if (n_commit > 1u) {
    memcpy([b_mtp_f_rnn contents],
           (const uint8_t*)[b_mtp_f_rnn contents] + size_t(n_commit - 1u) * H * sizeof(float),
           size_t(H) * sizeof(float));
  }
  mtp_ctx_pos += n_commit;
}

std::vector<uint32_t> Model::draft(uint32_t anchor_tok, double* gpu_ms) {
  if (!has_mtp) return {};
  const uint32_t B = mtp_block;  // 8
  const uint32_t gamma = B - 1u; // 7 draft tokens
  *(uint32_t*)[b_mtp_prev_tok contents] = anchor_tok;
  memset([b_mtp_rnn_state contents], 0, size_t(mtp_rank) * sizeof(float));

  id<MTLCommandBuffer> cb = [C.q commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
  auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };

  // 1. Build 8-token draft input block: [embed_norm(embed[anchor]), mask_embed x 7]
  [enc setComputePipelineState:p_mtp_embed];
  [enc setBuffer:embed.b offset:embed.off atIndex:0];
  [enc setBuffer:embed_norm.b offset:embed_norm.off atIndex:1];
  [enc setBuffer:mtp_mask_embed.b offset:mtp_mask_embed.off atIndex:2];
  [enc setBuffer:b_mtp_h offset:0 atIndex:3];
  [enc setBytes:&anchor_tok length:4 atIndex:4];
  [enc setBytes:&H length:4 atIndex:5];
  [enc setBytes:&eps length:4 atIndex:6];
  [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  barrier();

  const uint32_t zero_u = 0, one_u = 1;
  const uint32_t bh4 = (B * H) / 4u;

  // 2. Run 3 dense DFlare MTP layers
  for (uint32_t l = 0; l < mtp_nL; l++) {
    auto& ml = MtpL[l];
    // attn_norm -> b_mtp_frag
    [enc setComputePipelineState:p_rms_frag];
    [enc setBuffer:b_mtp_h offset:0 atIndex:0];
    [enc setBuffer:ml.attn_norm.b offset:ml.attn_norm.off atIndex:1];
    [enc setBuffer:b_mtp_frag offset:0 atIndex:2];
    [enc setBuffer:b_dummy offset:0 atIndex:3];
    [enc setBytes:&H length:4 atIndex:4];
    [enc setBytes:&eps length:4 atIndex:5];
    [enc setBytes:&zero_u length:4 atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();

    enc_mm2(enc, p_mm_w8, ml.qkv, b_mtp_frag, 0, b_mtp_qkv, 0, B, nil, b_dummy, nil, nil, nil, false);
    barrier();

    MtpQkvArgs qa{nh, nkv, hd, pos, eps, mtp_rope_theta};
    [enc setComputePipelineState:p_mtp_qkv_rope];
    [enc setBuffer:b_mtp_qkv offset:0 atIndex:0];
    [enc setBuffer:ml.q_norm.b offset:ml.q_norm.off atIndex:1];
    [enc setBuffer:ml.k_norm.b offset:ml.k_norm.off atIndex:2];
    [enc setBuffer:b_mtp_Q offset:0 atIndex:3];
    [enc setBuffer:b_mtp_K_blk offset:0 atIndex:4];
    [enc setBuffer:b_mtp_V_blk offset:0 atIndex:5];
    [enc setBytes:&qa length:sizeof(qa) atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(B, nh + nkv, 1) threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
    barrier();

    MtpAttnArgs aa{B, nh, nkv, hd, mtp_ctx_pos, ml.cap, window, ml.is_local ? 1u : 0u};
    [enc setComputePipelineState:p_mtp_attn];
    [enc setBuffer:b_mtp_Q offset:0 atIndex:0];
    [enc setBuffer:ml.k_cache offset:0 atIndex:1];
    [enc setBuffer:ml.v_cache offset:0 atIndex:2];
    [enc setBuffer:b_mtp_K_blk offset:0 atIndex:3];
    [enc setBuffer:b_mtp_V_blk offset:0 atIndex:4];
    [enc setBuffer:b_mtp_o_frag offset:0 atIndex:5];
    [enc setBytes:&aa length:sizeof(aa) atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(B, nh, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    barrier();

    enc_mm2(enc, p_mm_w8, ml.o, b_mtp_o_frag, 0, b_mtp_o_out, 0, B, nil, b_dummy, nil, nil, nil, false);
    barrier();

    [enc setComputePipelineState:p_add_f32];
    [enc setBuffer:b_mtp_h offset:0 atIndex:0];
    [enc setBuffer:b_mtp_o_out offset:0 atIndex:1];
    [enc setBytes:&bh4 length:4 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((bh4 + 255u) / 256u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();

    // MLP: mlp_norm -> mlp_gu (EPI=3 SiLU SwiGLU) -> mlp_dn -> residual add
    [enc setComputePipelineState:p_rms_frag];
    [enc setBuffer:b_mtp_h offset:0 atIndex:0];
    [enc setBuffer:ml.mlp_norm.b offset:ml.mlp_norm.off atIndex:1];
    [enc setBuffer:b_mtp_frag offset:0 atIndex:2];
    [enc setBuffer:b_dummy offset:0 atIndex:3];
    [enc setBytes:&H length:4 atIndex:4];
    [enc setBytes:&eps length:4 atIndex:5];
    [enc setBytes:&zero_u length:4 atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();

    enc_mm2(enc, p_mm_w6_gu, ml.mlp_gu, b_mtp_frag, 0, b_dummy, 0, B, b_mtp_gu_frag, b_dummy, nil, nil, nil, false, mtp_I);
    barrier();

    enc_mm2(enc, p_mm_w6_dn, ml.mlp_dn, b_mtp_gu_frag, 0, b_mtp_dn_out, 0, B, nil, b_dummy, nil, nil, nil, false);
    barrier();

    [enc setComputePipelineState:p_add_f32];
    [enc setBuffer:b_mtp_h offset:0 atIndex:0];
    [enc setBuffer:b_mtp_dn_out offset:0 atIndex:1];
    [enc setBytes:&bh4 length:4 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((bh4 + 255u) / 256u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();
  }

  // 3. Final MTP norm -> b_mtp_frag (all B rows) & b_mtp_base_h (f32 for RNN)
  [enc setComputePipelineState:p_rms_frag];
  [enc setBuffer:b_mtp_h offset:0 atIndex:0];
  [enc setBuffer:mtp_norm.b offset:mtp_norm.off atIndex:1];
  [enc setBuffer:b_mtp_frag offset:0 atIndex:2];
  [enc setBuffer:b_mtp_base_h offset:0 atIndex:3];
  [enc setBytes:&H length:4 atIndex:4];
  [enc setBytes:&eps length:4 atIndex:5];
  [enc setBytes:&one_u length:4 atIndex:6];
  [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  barrier();

  // Base logits for all B rows (we use rows 1..7)
  enc_mm2(enc, p_mm_w8, lm_head, b_mtp_frag, 0, b_mtp_base_logits, 0, B, nil, b_dummy, nil, nil, nil, false);
  barrier();

  // 4. Sequential Markov RNN head for 7 draft tokens (all inside this single command buffer!)
  for (uint32_t k = 0; k < gamma; k++) {
    RnnArgs ra{H, mtp_rank, k};
    [enc setComputePipelineState:p_mtp_rnn_proj];
    [enc setBuffer:b_mtp_rnn_state offset:0 atIndex:0];
    [enc setBuffer:mtp_mh_markov_w1.b offset:mtp_mh_markov_w1.off atIndex:1];
    [enc setBuffer:b_mtp_prev_tok offset:0 atIndex:2];
    [enc setBuffer:b_mtp_base_h offset:0 atIndex:3];
    [enc setBuffer:b_mtp_f_rnn offset:0 atIndex:4];
    [enc setBuffer:mtp_mh_rnn_w.b offset:mtp_mh_rnn_w.off atIndex:5];
    [enc setBuffer:mtp_mh_rnn_b.b offset:mtp_mh_rnn_b.off atIndex:6];
    [enc setBuffer:b_mtp_rnn_proj offset:0 atIndex:7];
    [enc setBytes:&ra length:sizeof(ra) atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake((3u * mtp_rank + 3u) / 4u, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    barrier();

    [enc setComputePipelineState:p_mtp_rnn_gates];
    [enc setBuffer:b_mtp_rnn_proj offset:0 atIndex:0];
    [enc setBuffer:b_mtp_rnn_state offset:0 atIndex:1];
    [enc setBuffer:b_mtp_lat_frag offset:0 atIndex:2];
    [enc setBytes:&mtp_rank length:4 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();

    // Folded Markov LM head: [200064, 256] W8 x [1, 256] -> b_mtp_bias
    enc_mm2(enc, p_mm_w8, mtp_mh_fold_w, b_mtp_lat_frag, 0, b_mtp_bias, 0, 1u, nil, b_dummy, nil, nil, nil, false);
    barrier();

    MtpPickArgs pa{vocab_valid, vocab_pad, k, logit_softcap};
    [enc setComputePipelineState:p_mtp_add_argmax];
    [enc setBuffer:b_mtp_base_logits offset:(size_t(vocab_pad) * sizeof(float)) atIndex:0];  // row 1..7!
    [enc setBuffer:b_mtp_bias offset:0 atIndex:1];
    [enc setBuffer:mtp_mh_fold_b.b offset:mtp_mh_fold_b.off atIndex:2];
    [enc setBuffer:b_mtp_prev_tok offset:0 atIndex:3];
    [enc setBuffer:b_mtp_draft_out offset:0 atIndex:4];
    [enc setBytes:&pa length:sizeof(pa) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();
  }

  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.error) mtl::die("GPU draft error", cb.error);
  if (gpu_ms) *gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;

  std::vector<uint32_t> out(gamma);
  memcpy(out.data(), [b_mtp_draft_out contents], gamma * sizeof(uint32_t));
  return out;
}

}  // namespace ink2
