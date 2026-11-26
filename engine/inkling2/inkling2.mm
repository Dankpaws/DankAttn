// DankAttn (Inkling-2-Mini): model loader, kv8 global KV, Wave 2A forward step, rollback, and DSpark MTP drafter.
#import "inkling2.h"
#import <Foundation/Foundation.h>
#import <IOKit/ps/IOPowerSources.h>
#import <IOKit/ps/IOPSKeys.h>
#include <algorithm>
#include <chrono>
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

// Native macOS power awareness (INK_POWER=auto|performance|eco; default auto).
static int parse_power_policy() {
  const char* env = getenv("INK_POWER");
  if (!env || !*env || strcmp(env, "auto") == 0) return 0;  // auto
  if (strcmp(env, "eco") == 0 || strcmp(env, "battery") == 0 || strcmp(env, "1") == 0) return 1;  // forced eco
  return 2;  // forced performance
}

static bool query_macos_battery_or_lpm() {
  if ([NSProcessInfo processInfo].lowPowerModeEnabled) return true;
  CFTypeRef info = IOPSCopyPowerSourcesInfo();
  if (!info) return false;
  CFStringRef src = IOPSGetProvidingPowerSourceType(info);
  const bool on_batt = (src != nullptr && CFEqual(src, CFSTR(kIOPSBatteryPowerValue)));
  CFRelease(info);
  return on_batt;
}

bool eco_mode() {
  static const int policy = parse_power_policy();
  if (policy == 1) return true;
  if (policy == 2) return false;
  static auto last_check = std::chrono::steady_clock::now() - std::chrono::seconds(10);
  static bool cached = false;
  static bool init = false;
  const auto now = std::chrono::steady_clock::now();
  if (!init || std::chrono::duration_cast<std::chrono::seconds>(now - last_check).count() >= 5) {
    const bool next = query_macos_battery_or_lpm();
    if (init && next != cached) {
      fprintf(stderr, "inkling2: power state changed -> %s\n", next ? "eco (battery/LPM)" : "performance (AC)");
    }
    cached = next;
    last_check = now;
    init = true;
  }
  return cached;
}

const char* power_mode_name() {
  const int p = parse_power_policy();
  if (p == 1) return "eco";
  if (p == 2) return "performance";
  return eco_mode() ? "auto(eco)" : "auto(performance)";
}

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
struct AttnResRmsArgs { uint32_t H, M, n_prev, b_stride, write_f32; float eps; };
struct AttnResMgArgs { uint32_t H, M, n_prev, b_stride, q_shared, cat_stride_b4; float eps; };
struct MsgGateArgs { uint32_t H, R; float eps; };
struct MsgGate3Args { uint32_t H, R, M, q_shared, cat_stride_b4, low_stride_f; };
struct AttnResRmsScoresArgs { uint32_t H, E, M, n_prev, b_stride; float eps; };
struct AttnResMgDownArgs { uint32_t H, R, M, n_prev, b_stride; float eps; };
struct AttnResMgDown3Args { uint32_t H, R, M, n_prev, b_stride, q_shared, low_stride_f; float eps; };
struct MgMmaArgs { uint32_t H, R, M, q_shared, cat_stride_b, low_stride_f; };
struct QRelArgs {
  uint32_t nh, hd, d_rel, rel_extent, qgrs_n, pos0, is_global;
  float qk_clip, rel_clip, log_n_floor, log_alpha;
};
struct KvConvArgs {
  uint32_t M, nkv, hd, k_stride, pos0, cap, is_global, is_key, ks, kv8, fork_pos;
  float qk_clip, eps;
};
struct AttnArgs { uint32_t M, nh, nkv, hd, rel_extent, pos0, cap, window, is_local, n_splits, kv8; };
struct AttnPfArgs { uint32_t M, nh, nkv, hd, rel_extent, pos0, cap, window, fork_pos; };
struct AttnDecArgs { uint32_t M, nh, nkv, hd, rel_extent, pos0, cap, window, n_splits, fork_pos; };
struct PostConvArgs { uint32_t M, H, ks, is_boundary, is_mlp, b_slot, b_stride; };
struct AccumConvArgs { uint32_t m, H, ks, r0, ystride; };
struct GemvArgs { uint32_t N, K, M, wstride, sstride, xstride, ystride, I; };
struct ConvResRmsArgs { uint32_t H, m, n_prev, b_stride, r0, ks, is_boundary, b_slot; float eps; };
struct RouteArgs { uint32_t M, H, E, topk; float route_scale; };
struct PlanArgs { uint32_t M, E, topk, gx_gu, gx_dn; };
struct MtpMixArgs { uint32_t H, M_stride, write_frag, write_f32, row_off; float eps; };
struct MtpKvArgs { uint32_t nkv, hd, pos0, cap, fork_pos; float eps, rope_theta; };
struct MtpQkvArgs { uint32_t nh, nkv, hd, pos0; float eps, rope_theta; };
struct MtpAttnArgs { uint32_t B, nh, nkv, hd, ctx_len, cap, window, is_local, fork_pos, out_row0; };
struct PackFragArgs { uint32_t H, B, gamma; };
struct RnnArgs { uint32_t H, R, step; };
struct MtpPickArgs { uint32_t V_valid, V_pad, step; float cap; float inv_t; uint32_t s0, s1, pos0; };
struct SampRow { uint32_t N, ld, pos, s0, s1; float inv_t, lmin, top_p; uint32_t top_k; };

// Split-K (Lin.ksplit > 1, dense M <= 16 only): per-split partials and self-resetting last-arrival counters for
// mm2's reduce, allocated at load.
static id<MTLBuffer> g_mmpart = nil, g_mmcnt = nil;
static bool g_big_gpu = false;
static constexpr size_t kSplitPartBytes = size_t(16) * 16u * 8192u * sizeof(float);  // per pslot: ksplit <= 16, N <= 8192
static constexpr size_t kSplitCntBytes = size_t(1) << 14;                            // per pslot

void enc_mm2(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipe,
             const Lin& lin, id<MTLBuffer> X, size_t xoff, id<MTLBuffer> Y, size_t yoff,
             uint32_t M, id<MTLBuffer> E0, id<MTLBuffer> dummy,
             id<MTLBuffer> slots, id<MTLBuffer> nslots, id<MTLBuffer> rmask,
             bool grouped, uint32_t R_epi = 0, uint32_t xstride_u4 = 0, uint32_t ystride_f = 0,
             bool prefer_nft1 = false) {
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
  // INK_NFT: 32-column fragments per threadgroup (layout only, same bits). Default 1 on Ultra/Max (measured -0.08 ms
  // at ctx 512 M=1), 0 on Base M5 (per-call choice, nft=2 where possible).
  static const int nft_env = getenv("INK_NFT") ? atoi(getenv("INK_NFT")) : -1;
  if (nft_env == 1 || (nft_env == -1 && g_big_gpu)) prefer_nft1 = true;
  else if (nft_env == 2) prefer_nft1 = false;
  const bool use_nft2 = !prefer_nft1 && ((lin.N / 32u) % 2u == 0u && lin.sgs <= 4u);
  const uint32_t nsg = use_nft2 ? (lin.sgs * 2u) : lin.sgs;
  const uint32_t nft = nsg / lin.sgs;
  const uint32_t gx = (lin.N / 32u + nft - 1u) / nft;
  const uint32_t gz = grouped ? std::min<uint32_t>(M * 8u, lin.experts) : 1u;
  const uint32_t ks = grouped ? 1u : lin.ksplit;
  if (ks > 1u) {
    a.ksplit = ks;
    a.psplit = 16u * lin.N;  // one 16-row block (M <= 16)
    if (size_t(ks) * a.psplit * sizeof(float) > kSplitPartBytes) mtl::die("mm2: split partials buffer too small");
  }

  [enc setComputePipelineState:pipe];
  [enc setBuffer:lin.w offset:lin.woff atIndex:0];
  [enc setBuffer:lin.s offset:lin.soff atIndex:1];
  [enc setBuffer:X offset:xoff atIndex:2];
  [enc setBuffer:(ks > 1u ? g_mmpart : dummy) offset:(ks > 1u ? lin.pslot * kSplitPartBytes : 0) atIndex:3];
  [enc setBuffer:Y offset:yoff atIndex:4];
  [enc setBytes:&a length:sizeof(a) atIndex:5];
  [enc setBuffer:(grouped ? slots : dummy) offset:0 atIndex:6];
  [enc setBuffer:(grouped ? nslots : dummy) offset:0 atIndex:7];
  [enc setBuffer:(ks > 1u ? g_mmcnt : dummy) offset:(ks > 1u ? lin.pslot * kSplitCntBytes : 0) atIndex:8];
  [enc setBuffer:(E0 ? E0 : dummy) offset:0 atIndex:9];
  [enc setBuffer:dummy offset:0 atIndex:10];
  [enc setBuffer:dummy offset:0 atIndex:11];
  [enc setBuffer:dummy offset:0 atIndex:12];
  [enc setBuffer:dummy offset:0 atIndex:13];
  [enc setBuffer:dummy offset:0 atIndex:14];
  [enc setBuffer:(grouped ? rmask : dummy) offset:0 atIndex:15];
  [enc setThreadgroupMemoryLength:size_t(nsg) * 16u * 32u * sizeof(float) atIndex:0];
  [enc dispatchThreadgroups:MTLSizeMake(gx, ks, gz) threadsPerThreadgroup:MTLSizeMake(nsg * 32u, 1, 1)];
}

// Dispatch mmp (dense W8 prefill, MMP_CH=8, MMP_NF=2, MMP_FAST=1).
void enc_mmp(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipe,
             const Lin& lin, id<MTLBuffer> X, size_t xoff, id<MTLBuffer> Y, size_t yoff,
             uint32_t M, id<MTLBuffer> dummy) {
  MmArgs a{};
  a.N = lin.N;
  a.K = lin.K;
  a.M = M;
  a.ksplit = lin.ksplit;  // replay mm2's split order
  a.sgs = lin.sgs;
  const uint32_t gx = (lin.N / 32u + 1u) / 2u;       // MMP_NF = 2
  const uint32_t gy = (M + 127u) / 128u;             // MMP_CH = 8 -> 128 rows/TG
  [enc setComputePipelineState:pipe];
  [enc setBuffer:lin.w offset:lin.woff atIndex:0];
  [enc setBuffer:lin.s offset:lin.soff atIndex:1];
  [enc setBuffer:X offset:xoff atIndex:2];
  [enc setBuffer:dummy offset:0 atIndex:3];
  [enc setBuffer:Y offset:yoff atIndex:4];
  [enc setBytes:&a length:sizeof(a) atIndex:5];
  [enc setBuffer:dummy offset:0 atIndex:6];
  [enc setBuffer:dummy offset:0 atIndex:7];
  [enc setBuffer:dummy offset:0 atIndex:8];
  [enc setBuffer:dummy offset:0 atIndex:9];
  [enc setBuffer:dummy offset:0 atIndex:10];
  [enc setBuffer:dummy offset:0 atIndex:11];
  [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1) threadsPerThreadgroup:MTLSizeMake(8u * 32u, 1, 1)];
}

// Dispatch mmp (MMP_CH=4, MMP_NF=1, MMP_FAST=0 -> bit-exact with mm2, supports W6/W8 + EPI=3 SwiGLU!).
void enc_mmp_nf1(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipe,
                 const Lin& lin, id<MTLBuffer> X, size_t xoff, id<MTLBuffer> Y, size_t yoff,
                 uint32_t M, id<MTLBuffer> E0, id<MTLBuffer> dummy, uint32_t R_epi = 0) {
  MmArgs a{};
  a.N = lin.N;
  a.K = lin.K;
  a.M = M;
  a.ksplit = lin.ksplit;  // replay mm2's split order
  a.sgs = lin.sgs;
  a.R = R_epi;
  const uint32_t gx = lin.N / 32u;
  const uint32_t gy = (M + 63u) / 64u;
  [enc setComputePipelineState:pipe];
  [enc setBuffer:lin.w offset:lin.woff atIndex:0];
  [enc setBuffer:lin.s offset:lin.soff atIndex:1];
  [enc setBuffer:X offset:xoff atIndex:2];
  [enc setBuffer:dummy offset:0 atIndex:3];
  [enc setBuffer:Y offset:yoff atIndex:4];
  [enc setBytes:&a length:sizeof(a) atIndex:5];
  [enc setBuffer:dummy offset:0 atIndex:6];
  [enc setBuffer:dummy offset:0 atIndex:7];
  [enc setBuffer:dummy offset:0 atIndex:8];
  [enc setBuffer:(E0 ? E0 : dummy) offset:0 atIndex:9];
  [enc setBuffer:dummy offset:0 atIndex:10];
  [enc setBuffer:dummy offset:0 atIndex:11];
  [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1) threadsPerThreadgroup:MTLSizeMake(4u * 32u, 1, 1)];
}

// Dispatch mmpg (grouped NVFP4/MXFP8 MoE prefill, MMP_CH=4, MMP_NF=2, MMP_FAST=1, indirect worklist).
void enc_mmpg(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipe,
              const Lin& lin, id<MTLBuffer> X, id<MTLBuffer> Y, id<MTLBuffer> E0,
              id<MTLBuffer> slots, id<MTLBuffer> nslots, id<MTLBuffer> roff, id<MTLBuffer> rcnt,
              id<MTLBuffer> rlist, id<MTLBuffer> wtile, id<MTLBuffer> ind_buf, id<MTLBuffer> dummy,
              uint32_t M, uint32_t ldx, uint32_t R_epi) {
  MmArgs a{};
  a.N = lin.N;
  a.K = lin.K;
  a.M = M;
  a.ksplit = lin.ksplit;  // replay mm2's split order
  a.sgs = lin.sgs;
  a.ldx = ldx;   // 0: gather via rlist (moe_gu); 2: compact off+i (moe_dn)
  a.R = R_epi;
  a.rm = 1;      // use wtile indirect worklist
  const uint32_t cb_exp = (lin.bits == 40 || lin.bits == 4) ? (lin.N * lin.K / 2) : (lin.N * lin.K);
  a.wstride = cb_exp / 16u;
  a.sstride = lin.N * (lin.K / lin.gs);
  [enc setComputePipelineState:pipe];
  [enc setBuffer:lin.w offset:lin.woff atIndex:0];
  [enc setBuffer:lin.s offset:lin.soff atIndex:1];
  [enc setBuffer:X offset:0 atIndex:2];
  [enc setBuffer:dummy offset:0 atIndex:3];
  [enc setBuffer:Y offset:0 atIndex:4];
  [enc setBytes:&a length:sizeof(a) atIndex:5];
  [enc setBuffer:slots offset:0 atIndex:6];
  [enc setBuffer:nslots offset:0 atIndex:7];
  [enc setBuffer:dummy offset:0 atIndex:8];
  [enc setBuffer:(E0 ? E0 : dummy) offset:0 atIndex:9];
  [enc setBuffer:dummy offset:0 atIndex:10];
  [enc setBuffer:dummy offset:0 atIndex:11];
  [enc setBuffer:roff offset:0 atIndex:12];
  [enc setBuffer:rcnt offset:0 atIndex:13];
  [enc setBuffer:rlist offset:0 atIndex:14];
  [enc setBuffer:wtile offset:0 atIndex:15];
  [enc dispatchThreadgroupsWithIndirectBuffer:ind_buf indirectBufferOffset:0
                        threadsPerThreadgroup:MTLSizeMake(4u * 32u, 1, 1)];
}

}  // namespace

void Model::load(const std::string& pack_dir, uint32_t max_cap, uint32_t passes_override, bool kv8) {
  dir = pack_dir;
  cap = max_cap;
  use_kv8 = kv8;
  {
    NSString* dname = [C.dev name];
    big_gpu = dname && ([dname containsString:@"Ultra"] || [dname containsString:@"Max"]);
    g_big_gpu = big_gpu;
    // Decode attention key splits (one serial key chain per simdgroup, so more splits = shorter chains). M5 Ultra
    // 2026-11-22, 12/24 -> 24/64: ctx 512 +0.05 ms, 8k -0.39, 30k -1.18 (13.55 -> 12.37 ms); 96/128 global splits
    // lose to the longer reduce. Base M5 keeps 8/8 (not re-measured).
    nsp_local = big_gpu ? 24u : 8u;
    nsp_global = big_gpu ? 64u : 8u;
    // INK_NSP_LOCAL / INK_NSP_GLOBAL: decode attention key splits per (row, kv head), 1..ASPLITS.
    if (getenv("INK_NSP_LOCAL")) nsp_local = std::clamp<uint32_t>(uint32_t(atoi(getenv("INK_NSP_LOCAL"))), 1u, ASPLITS);
    if (getenv("INK_NSP_GLOBAL")) nsp_global = std::clamp<uint32_t>(uint32_t(atoi(getenv("INK_NSP_GLOBAL"))), 1u, ASPLITS);
  }

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
    std::string s = [fn UTF8String];
    mtl::check_pack_filename(s);
    files.push_back(mmap_file(C.dev, dir + "/" + s, mapped_bytes));
  }

  auto load_raw_from = [&](NSDictionary* tens, const std::vector<id<MTLBuffer>>& fvec, const std::string& name) -> Raw {
    NSDictionary* d = tens[@(name.c_str())];
    if (!d) mtl::die(("missing raw tensor " + name).c_str());
    Raw r;
    r.off = [d[@"off"] unsignedLongLongValue];
    r.bytes = [d[@"bytes"] unsignedLongLongValue];
    r.b = mtl::checked_file(fvec, [d[@"file"] unsignedIntValue], r.off, r.bytes, name);
    return r;
  };
  auto load_lin_from = [&](NSDictionary* tens, const std::vector<id<MTLBuffer>>& fvec, const std::string& name) -> Lin {
    NSDictionary* d = tens[@(name.c_str())];
    if (!d) mtl::die(("missing linear tensor " + name).c_str());
    Lin ln;
    ln.woff = [d[@"off"] unsignedLongLongValue];
    ln.w = mtl::checked_file(fvec, [d[@"file"] unsignedIntValue], ln.woff, 0, name);
    ln.soff = [d[@"soff"] unsignedLongLongValue];
    ln.s = mtl::checked_file(fvec, [d[@"sfile"] unsignedIntValue], ln.soff, 0, name + ".scales");
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
    // Decode split table (default on for Ultra/Max, off for Base M5 where 10 cores lose ~1.4 ms to 12-way o_proj
    // split atomics; INK_SPLIT_TAB=0/1 overrides): measured 2026-11-22 on M5 Ultra, ctx 512 M=1 (-0.73 ms).
    const bool use_split_tab = getenv("INK_SPLIT_TAB") ? (atoi(getenv("INK_SPLIT_TAB")) != 0) : big_gpu;
    if (use_split_tab) {
      static const uint32_t tab[][5] = {{8, 2304, 3072, 1, 12}, {40, 2304, 2304, 2, 1}, {40, 2304, 1152, 2, 1}};
      for (const auto& r : tab)
        if (r[0] == ln.bits && r[1] == ln.N && r[2] == ln.K && units % (r[3] * r[4]) == 0) {
          ln.sgs = r[3];
          ln.ksplit = r[4];
        }
    }
    // INK_SPLITS="bits:N:K:sgs:ksplit,..." (screen): K-split of that tensor shape across simdgroups (sgs) and, for
    // dense tensors, threadgroups (ksplit). Changes the reduction tree for decode, verify and prefill alike
    // (mmp/mmpg replay mm2's slices and splits), so those paths stay mutually exact.
    if (const char* ov = getenv("INK_SPLITS")) {
      std::string spec(ov);
      size_t at = 0;
      while (at < spec.size()) {
        size_t e = spec.find(',', at);
        if (e == std::string::npos) e = spec.size();
        unsigned b = 0, n = 0, k = 0, g = 0, q = 0;
        if (sscanf(spec.substr(at, e - at).c_str(), "%u:%u:%u:%u:%u", &b, &n, &k, &g, &q) == 5 && b == ln.bits &&
            n == ln.N && k == ln.K && g >= 1 && g <= 12 && q >= 1 && q <= 16 && units % (g * q) == 0 &&
            (q == 1 || (ln.experts <= 1 && ln.N <= 8192))) {
          ln.sgs = g;
          ln.ksplit = q;
        }
        at = e + 1;
      }
    }
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
    ly.qgrs.pslot = 1;  // qgrs || k || v share a phase: disjoint split-K scratch
    ly.k_proj.pslot = 2;
    ly.v_proj.pslot = 3;
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

  ring_extra = getenv("INK_RING_SLACK") ? uint32_t(atoi(getenv("INK_RING_SLACK"))) : 0u;
  // Allocate per-logical-layer KV + sconv + rollback history (lane 0)
  alloc_state(S);
  lanes_.assign(1, Lane{});
  active_lane = 0;

  // Allocate main scratch buffers (sized for MP = 512 chunked prefill + MAXM = 16 decode)
  // Always-live step buffers:
  b_ids = C.buf(MP * sizeof(uint32_t));
  b_h = C.buf(MP * H * sizeof(float));
  b_pre_h = C.buf(MP * H * sizeof(float));
  b_prev_h = C.buf(MP * H * sizeof(float));
  b_blocks = C.buf(MAX_BLOCKS * MP * H * sizeof(float));
  b_pre_blocks = C.buf(MAX_BLOCKS * MP * H * sizeof(float));
  b_prev_blocks = C.buf(MAX_BLOCKS * MP * H * sizeof(float));
  b_msgs = C.buf(4u * MP * H * sizeof(float));
  b_frag_q = C.buf(MP * H * sizeof(uint16_t));
  b_frag_k = C.buf(MP * H * sizeof(uint16_t));
  b_frag_v = C.buf(MP * H * sizeof(uint16_t));

  // Phase-aliased MTLHeap (MTLHazardTrackingModeTracked) for non-overlapping Attention / MoE / Logits phases:
  // Saves ~160 MB of wired GPU state (crucial for 32 GB MacBook Pro at 128k context).
  auto align16k = [](size_t n) -> size_t { return (n + 16383u) & ~size_t(16383u); };
  const size_t sz_mg_cat    = align16k(3u * MP * 2u * H * sizeof(uint16_t));
  const size_t sz_mg_low    = align16k(3u * MP * gate_rank * sizeof(float));
  const size_t sz_qkv_in    = align16k(3u * MP * H * sizeof(float));
  const size_t sz_qkv_out   = align16k(3u * MP * H * sizeof(float));
  const size_t off_mg_end   = sz_mg_cat + sz_mg_low;
  const size_t sz_phase1a   = off_mg_end + sz_qkv_in + sz_qkv_out;

  const size_t sz_qgrs      = align16k(size_t(MP) * L[0].qgrs.N * sizeof(float));
  const size_t sz_k_raw     = align16k(size_t(MP) * (nkv * hd + 32u) * sizeof(float));
  const size_t sz_v_raw     = align16k(size_t(MP) * (nkv * hd + 32u) * sizeof(float));
  const size_t sz_Q         = align16k(size_t(MP) * nh * hd * sizeof(float));
  const size_t sz_rel       = align16k(size_t(MP) * nh * rel_extent * sizeof(float));
  const size_t sz_gate      = align16k(size_t(MP) * nh * hd * sizeof(float));
  const size_t sz_sinks     = align16k(size_t(MP) * nh * sizeof(float));
  const size_t sz_o_out     = align16k(size_t(MP) * H * sizeof(float));
  const size_t sz_attn_part = align16k(size_t(MAXM) * nh * ASPLITS * 130u * sizeof(float));
  const size_t sz_frag_o    = align16k(size_t(MP) * nh * hd * sizeof(uint16_t));
  const size_t sz_phase1b   = sz_qgrs + sz_k_raw + sz_v_raw + sz_Q + sz_rel + sz_gate + sz_sinks + sz_o_out + sz_attn_part + sz_frag_o;

  const size_t sz_mh        = align16k(size_t(MP) * H * sizeof(float));
  const size_t sz_mn_f32    = align16k(size_t(MP) * H * sizeof(float));
  const size_t sz_frag_m    = align16k(size_t(MP) * H * sizeof(uint16_t));
  const size_t sz_rscores   = align16k(size_t(MP) * E * sizeof(float));
  const size_t sz_frag_gu   = align16k(size_t(PF_COMPACT) * I_moe * sizeof(uint16_t));
  const size_t sz_moe_y     = align16k(size_t(PF_COMPACT) * H * sizeof(float));
  const size_t sz_moe_out   = align16k(size_t(MP) * H * sizeof(float));
  const size_t sz_phase2    = off_mg_end + sz_mh + sz_mn_f32 + sz_frag_m + sz_rscores + sz_frag_gu + sz_moe_y + sz_moe_out;

  const size_t sz_logits    = align16k(size_t(MAXM) * vocab_pad * sizeof(float));
  const size_t heap_sz      = std::max({sz_phase1a, sz_phase1b, sz_phase2, sz_logits});
  scratch_heap = C.heap(heap_sz);
  auto hbuf = [&](size_t& off, size_t sz) -> id<MTLBuffer> {
    id<MTLBuffer> b = [scratch_heap newBufferWithLength:sz options:MTLResourceStorageModeShared offset:off];
    if (!b) mtl::die("heap buffer alloc");
    off += sz;
    return b;
  };

  // Phase 1a (Pre-Attn msg_gate):
  {
    size_t o = 0;
    b_mg_cat  = hbuf(o, sz_mg_cat);
    b_mg_low  = hbuf(o, sz_mg_low);
    b_qkv_in  = hbuf(o, sz_qkv_in);
    b_qkv_out = hbuf(o, sz_qkv_out);
  }
  b_rel_vec = C.buf(size_t(MP) * nh * 16u * sizeof(uint16_t));
  b_samp_rows = C.buf(MAXM * sizeof(SampRow));
  // Phase 1b (Attention projections + FlashAttn + O-proj):
  {
    size_t o = 0;
    b_qgrs       = hbuf(o, sz_qgrs);
    b_k_raw      = hbuf(o, sz_k_raw);
    b_v_raw      = hbuf(o, sz_v_raw);
    b_Q          = hbuf(o, sz_Q);
    b_rel_logits = hbuf(o, sz_rel);
    b_gate       = hbuf(o, sz_gate);
    b_sinks      = hbuf(o, sz_sinks);
    b_o_out      = hbuf(o, sz_o_out);
    b_attn_part  = hbuf(o, sz_attn_part);
    b_frag_o     = hbuf(o, sz_frag_o);
  }
  // Phase 2 (MLP / MoE; starts after b_mg_cat + b_mg_low so MLP msg_gate never overlaps):
  {
    size_t o = off_mg_end;
    b_mh      = hbuf(o, sz_mh);
    b_mn_f32  = hbuf(o, sz_mn_f32);
    b_frag_m  = hbuf(o, sz_frag_m);
    b_rscores = hbuf(o, sz_rscores);
    b_frag_gu = hbuf(o, sz_frag_gu);
    b_moe_y   = hbuf(o, sz_moe_y);
    b_moe_out = hbuf(o, sz_moe_out);
  }
  // Phase 3 (Final logits at offset 0 < off_mg_end, so never overlaps b_frag_m/b_mh):
  {
    size_t o = 0;
    b_logits = hbuf(o, sz_logits);
  }

  b_top_idx = C.buf(MP * topk * sizeof(uint32_t));
  b_top_w = C.buf(MP * topk * sizeof(float));
  b_slots = C.buf(E * sizeof(uint32_t));
  b_nslots = C.buf(16);
  b_rmask = C.buf(E * 4u * sizeof(uint32_t));
  b_roff = C.buf(E * sizeof(uint32_t));
  b_rcnt = C.buf(E * sizeof(uint32_t));
  b_rlist = C.buf(PF_COMPACT * sizeof(uint32_t));
  b_rpos = C.buf(MP * topk * sizeof(uint32_t));
  b_wtile = C.buf((PF_COMPACT / 64u + E) * sizeof(uint32_t));
  b_ind_gu = C.buf(16);
  b_ind_dn = C.buf(16);
  b_tok_out = C.buf(MAXM * sizeof(uint32_t));
  b_argmax_v = C.buf(MAXM * 64u * sizeof(float));
  b_argmax_i = C.buf(MAXM * 64u * sizeof(uint32_t));
  b_conf_out = C.buf(MAXM * sizeof(float));
  b_samp_rows = C.buf(MAXM * sizeof(SampRow));
  b_dummy = C.buf(64);
  b_moe_act = C.buf(size_t(128) * MAXM * 1152u * sizeof(uint16_t));  // [slot][MAXM][I] bf16 (NVFP4 GEMV)
  g_mmpart = C.buf(4u * kSplitPartBytes);  // pslots 0..3
  g_mmcnt = C.buf(4u * kSplitCntBytes);
  memset([g_mmcnt contents], 0, 4u * kSplitCntBytes);  // last-arrival counters start at 0; mm2 resets each after use

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
        std::string s = [fn UTF8String];
        mtl::check_pack_filename(s);
        mtp_files.push_back(mmap_file(C.dev, dir + "/" + s, mapped_bytes));
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
        ml.cap = ml.is_local ? (window + MP + ring_extra) : cap;
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

      b_taps = C.buf(4u * MP * H * sizeof(float));
      b_mtp_f_rnn = C.buf(MP * H * sizeof(float));
      b_mtp_h = C.buf(MTP_MAXB * H * sizeof(float));
      b_mtp_frag = C.buf(MP * H * sizeof(uint16_t));
      b_mtp_qkv = C.buf(MTP_MAXB * (nh * hd + 2u * nkv * hd) * sizeof(float));
      b_mtp_kv_ctx = C.buf(MP * 2u * nkv * hd * sizeof(float));
      b_mtp_Q = C.buf(MTP_MAXB * nh * hd * sizeof(float));
      b_mtp_K_blk = C.buf(MTP_MAXB * nkv * hd * sizeof(uint16_t));
      b_mtp_V_blk = C.buf(MTP_MAXB * nkv * hd * sizeof(uint16_t));
      b_mtp_o_frag = C.buf(MTP_MAXB * nh * hd * sizeof(uint16_t));
      b_mtp_o_out = C.buf(MTP_MAXB * H * sizeof(float));
      b_mtp_gu_frag = C.buf(MTP_MAXB * mtp_I * sizeof(uint16_t));
      b_mtp_dn_out = C.buf(MTP_MAXB * H * sizeof(float));
      b_mtp_base_h = C.buf(MTP_MAXB * H * sizeof(float));
      b_mtp_base_logits = C.buf(MAXM * vocab_pad * sizeof(float));
      b_mtp_bias = C.buf(MTP_MAXL * vocab_pad * sizeof(float));
      b_mtp_rnn_state = C.buf(MTP_MAXL * mtp_rank * sizeof(float));
      b_mtp_rnn_static = C.buf(MTP_MAXL * 7u * 3u * mtp_rank * sizeof(float));
      b_mtp_rnn_proj = C.buf(MTP_MAXL * 3u * mtp_rank * sizeof(float));
      b_mtp_lat_frag = C.buf(16u * mtp_rank * sizeof(uint16_t));
      b_mtp_prev_tok = C.buf(MTP_MAXL * 16u);
      b_mtp_draft_out = C.buf(MTP_MAXL * MAXM * sizeof(uint32_t));
      b_mtp_draft_probs = C.buf(MTP_MAXL * MAXM * sizeof(float));
      b_mtp_part_l = C.buf(MTP_MAXL * 64u * sizeof(float));
      b_mtp_part_x = C.buf(MTP_MAXL * 128u * sizeof(float));
    }
  }

  // Compile Metal 4 kernels
  std::string kdir = mtl::tree_root() + "/engine/kernels/";
  std::string ink_src = read_text(kdir + "inkling2.metal");
  std::string attn_src = read_text(kdir + "inkling2_attn.metal");
  std::string mm_src = read_text(kdir + "mm2.metal");

  const char* pf_fast = (getenv("INK_PF_FAST") && atoi(getenv("INK_PF_FAST")) == 0) ? "0" : "1";
  p_attn_pf_local = C.pipe(attn_src, "ink_attn_pf_mma", {{"ATT_LOCAL", "1"}, {"ATT_KV8", "0"}, {"ATT_FAST", pf_fast}});
  p_attn_pf_global_bf16 = C.pipe(attn_src, "ink_attn_pf_mma", {{"ATT_LOCAL", "0"}, {"ATT_KV8", "0"}, {"ATT_FAST", pf_fast}});
  p_attn_pf_global_kv8 = C.pipe(attn_src, "ink_attn_pf_mma", {{"ATT_LOCAL", "0"}, {"ATT_KV8", "1"}, {"ATT_FAST", pf_fast}});
  p_attn_pf_global_bf16_fork = C.pipe(attn_src, "ink_attn_pf_mma", {{"ATT_LOCAL", "0"}, {"ATT_KV8", "0"}, {"ATT_FAST", pf_fast}, {"ATT_FORK", "1"}});
  p_attn_pf_global_kv8_fork = C.pipe(attn_src, "ink_attn_pf_mma", {{"ATT_LOCAL", "0"}, {"ATT_KV8", "1"}, {"ATT_FAST", pf_fast}, {"ATT_FORK", "1"}});
  p_attn_dec_local = C.pipe(attn_src, "ink_attn_split_gqa", {{"ATT_LOCAL", "1"}, {"ATT_KV8", "0"}});
  p_attn_dec_global_bf16 = C.pipe(attn_src, "ink_attn_split_gqa", {{"ATT_LOCAL", "0"}, {"ATT_KV8", "0"}});
  p_attn_dec_global_kv8 = C.pipe(attn_src, "ink_attn_split_gqa", {{"ATT_LOCAL", "0"}, {"ATT_KV8", "1"}});

  p_embed = C.pipe(ink_src, "ink_embed");
  p_attn_res = C.pipe(ink_src, "ink_attn_res");
  p_attn_res_rms = C.pipe(ink_src, "ink_attn_res_rms");
  p_attn_res_mg_prep = C.pipe(ink_src, "ink_attn_res_mg_prep");
  p_attn_res_mg_prep_shared = C.pipe(ink_src, "ink_attn_res_mg_prep_shared");
  p_mg_prep = C.pipe(ink_src, "ink_mg_prep");
  p_mg_down = C.pipe(ink_src, "ink_mg_down");
  p_mg_apply = C.pipe(ink_src, "ink_mg_apply");
  p_mg_down3 = C.pipe(ink_src, "ink_mg_down3");
  p_mg_apply3 = C.pipe(ink_src, "ink_mg_apply3");
  p_attn_res_rms_scores = C.pipe(ink_src, "ink_attn_res_rms_scores");
  p_rms_scores = C.pipe(ink_src, "ink_rms_scores");
  p_attn_res_mg_down = C.pipe(ink_src, "ink_attn_res_mg_down");
  p_attn_res_mg_down3 = C.pipe(ink_src, "ink_attn_res_mg_down3");
  p_mg_down_mma = C.pipe(attn_src, "ink_mg_down_mma");
  p_mg_apply_mma = C.pipe(attn_src, "ink_mg_apply_mma");
  p_rms_frag = C.pipe(ink_src, "ink_rms_frag");
  p_rms_frag3 = C.pipe(ink_src, "ink_rms_frag3");
  p_q_rel = C.pipe(ink_src, "ink_q_rel");
  p_q_rel_prep = C.pipe(ink_src, "ink_q_rel_prep");
  p_q_rel_mma = C.pipe(attn_src, "ink_q_rel_mma");
  qrel_mma = !(getenv("INK_QREL_MMA") && atoi(getenv("INK_QREL_MMA")) == 0) && d_rel == 16u;
  p_kv_conv = C.pipe(ink_src, "ink_kv_conv_cache");
  p_attn_split = C.pipe(ink_src, "ink_attn_split");
  p_attn_reduce = C.pipe(ink_src, "ink_attn_reduce");
  p_post_conv = C.pipe(ink_src, "ink_post_sconv_add");
  p_scores = C.pipe(ink_src, "ink_moe_scores");
  p_scores_mma = C.pipe(attn_src, "ink_moe_scores_mma");
  p_route = C.pipe(ink_src, "ink_moe_route");
  p_pack_dec = C.pipe(ink_src, "ink_moe_pack_dec");
  p_moe_accum = C.pipe(ink_src, "ink_moe_accum");
  p_route_fast = C.pipe(ink_src, "ink_moe_route_fast");
  p_moe_gu_gemv = C.pipe(ink_src, "ink_moe_gu_gemv", {{"MAXM_ROWS", std::to_string(MAXM)}, {"GEMV_ROWS", getenv("INK_GEMV_ROWS") ? getenv("INK_GEMV_ROWS") : "1"}});
  p_moe_dn_gemv = C.pipe(ink_src, "ink_moe_dn_gemv", {{"MAXM_ROWS", std::to_string(MAXM)}, {"GEMV_ROWS", getenv("INK_GEMV_ROWS") ? getenv("INK_GEMV_ROWS") : "1"}});
  p_moe_accum_conv = C.pipe(ink_src, "ink_moe_accum_post_sconv");
  p_conv_res_rms = C.pipe(ink_src, "ink_post_sconv_attn_res_rms");
  p_conv_res_mg = C.pipe(ink_src, "ink_post_sconv_attn_res_mg_prep");
  p_softcap_argmax_p1 = C.pipe(ink_src, "ink_softcap_argmax_p1");
  p_softcap_argmax_p2 = C.pipe(ink_src, "ink_softcap_argmax_p2");
  p_copy = C.pipe(ink_src, "ink_copy_f32");
  p_topk_pf = C.pipe(ink_src, "ink_moe_topk_pf");
  p_plan_pf = C.pipe(ink_src, "ink_moe_plan_pf");
  p_moe_accum_pf = C.pipe(ink_src, "ink_moe_accum_pf");

  // INK_PF_W8 / INK_PF_MOE: k-steps of weights in flight per simdgroup in the M <= 16 mm2 matmuls (load
  // scheduling only: the accumulation order, so every output bit, is unchanged).
  const std::string pf_w8 = getenv("INK_PF_W8") ? getenv("INK_PF_W8") : "1";
  const std::string pf_moe = getenv("INK_PF_MOE") ? getenv("INK_PF_MOE") : "1";
  p_mm_w8 = C.pipe(mm_src, "mm2", {{"WBITS", "8"}, {"GS", "128"}, {"TILES", "1"}, {"PF", pf_w8}});
  p_mm_w8_oconv = C.pipe(mm_src, "mm2", {{"WBITS", "8"}, {"GS", "128"}, {"TILES", "1"}, {"PF", pf_w8}, {"EPI", "4"}});
  p_mmp_w8 = C.pipe(mm_src, "mmp", {{"WBITS", "8"}, {"GS", "128"}, {"MMP_CH", "8"}, {"MMP_NF", "2"}, {"MMP_FAST", "1"}});
  p_moe_fp4_gu = C.pipe(mm_src, "mm2", {
      {"WBITS", "40"}, {"GS", "16"}, {"TILES", "1"}, {"GROUPED", "1"}, {"PF", pf_moe},
      {"EPI", "3"}, {"SWIGLU_SOFTCAP_G", "4.0"}, {"SWIGLU_SOFTCAP_U", "25.0"}});
  p_moe_fp4_dn = C.pipe(mm_src, "mm2", {
      {"WBITS", "40"}, {"GS", "16"}, {"TILES", "1"}, {"GROUPED", "1"}, {"PF", pf_moe}});
  p_moe_fp8_gu = C.pipe(mm_src, "mm2", {
      {"WBITS", "80"}, {"GS", "32"}, {"TILES", "1"}, {"GROUPED", "1"}, {"PF", pf_moe},
      {"EPI", "3"}, {"SWIGLU_SOFTCAP_G", "4.0"}, {"SWIGLU_SOFTCAP_U", "25.0"}});
  p_moe_fp8_dn = C.pipe(mm_src, "mm2", {
      {"WBITS", "80"}, {"GS", "32"}, {"TILES", "1"}, {"GROUPED", "1"}, {"PF", pf_moe}});
  p_mmpg_fp4_gu = C.pipe(mm_src, "mmpg", {
      {"WBITS", "40"}, {"GS", "16"}, {"GROUPED", "1"}, {"MMP_CH", "4"}, {"MMP_NF", "2"}, {"MMP_FAST", "1"},
      {"EPI", "3"}, {"SWIGLU_SOFTCAP_G", "4.0"}, {"SWIGLU_SOFTCAP_U", "25.0"}});
  p_mmpg_fp4_dn = C.pipe(mm_src, "mmpg", {
      {"WBITS", "40"}, {"GS", "16"}, {"GROUPED", "1"}, {"MMP_CH", "4"}, {"MMP_NF", "2"}, {"MMP_FAST", "1"},
      {"CONTIG_X", "1"}});
  p_mmpg_fp8_gu = C.pipe(mm_src, "mmpg", {
      {"WBITS", "80"}, {"GS", "32"}, {"GROUPED", "1"}, {"MMP_CH", "4"}, {"MMP_NF", "2"}, {"MMP_FAST", "1"},
      {"EPI", "3"}, {"SWIGLU_SOFTCAP_G", "4.0"}, {"SWIGLU_SOFTCAP_U", "25.0"}});
  p_mmpg_fp8_dn = C.pipe(mm_src, "mmpg", {
      {"WBITS", "80"}, {"GS", "32"}, {"GROUPED", "1"}, {"MMP_CH", "4"}, {"MMP_NF", "2"}, {"MMP_FAST", "1"},
      {"CONTIG_X", "1"}});

  p_sample_rows = C.pipe(ink_src, "ink_sample_rows");
  if (has_mtp) {
    p_mtp_mix = C.pipe(ink_src, "ink_mtp_mix_norm");
    p_mtp_kv_ctx = C.pipe(ink_src, "ink_mtp_kv_ctx_cache");
    p_mtp_embed = C.pipe(ink_src, "ink_mtp_embed");
    p_mtp_qkv_rope = C.pipe(ink_src, "ink_mtp_qkv_rope");
    p_mtp_attn = C.pipe(ink_src, "ink_mtp_attn");
    p_add_rms_frag = C.pipe(ink_src, "ink_add_rms_frag");
    p_mtp_rnn_static = C.pipe(ink_src, "ink_mtp_rnn_static");
    p_mtp_rnn_dyn = C.pipe(ink_src, "ink_mtp_rnn_dyn");
    p_mtp_rnn_gates = C.pipe(ink_src, "ink_mtp_rnn_gates");
    p_mtp_add_argmax_p1 = C.pipe(ink_src, "ink_mtp_add_argmax_p1");
    p_mtp_add_argmax_p2 = C.pipe(ink_src, "ink_mtp_add_argmax_p2");
    p_pack_frag_rows = C.pipe(ink_src, "ink_pack_frag_rows");
    const std::string mb = std::to_string(MtpL[0].mlp_gu.bits);
    p_mm_w6_gu = C.pipe(mm_src, "mm2", {{"WBITS", mb}, {"GS", "128"}, {"TILES", "1"}, {"EPI", "3"}});
    p_mm_w6_dn = C.pipe(mm_src, "mm2", {{"WBITS", mb}, {"GS", "128"}, {"TILES", "1"}});
    p_mmp_w8_mtp = C.pipe(mm_src, "mmp", {{"WBITS", "8"}, {"GS", "128"}, {"MMP_CH", "4"}, {"MMP_NF", "1"}, {"MMP_FAST", "0"}});
    p_mmp_w6_gu = C.pipe(mm_src, "mmp", {{"WBITS", mb}, {"GS", "128"}, {"MMP_CH", "4"}, {"MMP_NF", "1"}, {"MMP_FAST", "0"}, {"EPI", "3"}});
    p_mmp_w6_dn = C.pipe(mm_src, "mmp", {{"WBITS", mb}, {"GS", "128"}, {"MMP_CH", "4"}, {"MMP_NF", "1"}, {"MMP_FAST", "0"}});
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
  // Compute 64-bit FNV-1a fingerprint of pack & KV layout for SSD prefix cache validation.
  {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
    for (unsigned char c : dir) mix(c);
    mix(mapped_bytes); mix(H); mix(nL); mix(num_passes); mix(window);
    mix(use_kv8 ? 1u : 0u); mix(has_mtp ? 1u : 0u); mix(uint64_t(S.size()));
    fingerprint = h;
  }
  fprintf(stderr, "inkling2: loaded %s (%.2f GB mapped, %.2f GB GPU state, %u phys -> %u logical, %u passes, cap=%u, kv8=%d, dspark=%d)\n",
          dir.c_str(), mapped_bytes / 1e9, C.allocated / 1e9, nL, uint32_t(S.size()), num_passes, cap, int(use_kv8), int(has_mtp));
}

void Model::alloc_state(std::vector<LogicalState>& SS, uint32_t global_cap) {
  const uint32_t gcap = global_cap ? global_cap : cap;
  const uint32_t n_core = core_end - core_start;
  const uint32_t n_logical = nL + (num_passes - 1u) * n_core;
  SS.resize(n_logical);
  auto init_logical = [&](uint32_t log_idx, uint32_t phys_idx) {
    auto& st = SS[log_idx];
    st.phys_idx = phys_idx;
    st.cap = L[phys_idx].is_local ? (window + MP + ring_extra) : gcap;
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
    st.k_conv_next = C.buf(3u * nkv * hd * sizeof(float));
    st.v_conv = C.buf(3u * nkv * hd * sizeof(float));
    st.v_conv_next = C.buf(3u * nkv * hd * sizeof(float));
    st.attn_conv = C.buf(3u * H * sizeof(float));
    st.attn_conv_next = C.buf(3u * H * sizeof(float));
    st.mlp_conv = C.buf(3u * H * sizeof(float));
    st.mlp_conv_next = C.buf(3u * H * sizeof(float));
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
}

int Model::add_lane(uint32_t global_cap) {
  const uint32_t gcap = global_cap ? global_cap : cap;
  Lane ln;
  const size_t before = C.allocated;
  alloc_state(ln.S, gcap);
  static const bool quiet = getenv("INK_QUIET") != nullptr;
  if (!quiet) fprintf(stderr, "inkling2: lane %zu added (%.2f GB state, cap=%u)\n", lanes_.size(), (C.allocated - before) / 1e9, gcap);
  for (auto& st : ln.S) {
    memset([st.k_conv contents], 0, [st.k_conv length]);
    memset([st.v_conv contents], 0, [st.v_conv length]);
    memset([st.attn_conv contents], 0, [st.attn_conv length]);
    memset([st.mlp_conv contents], 0, [st.mlp_conv length]);
  }
  if (rs) {
    for (auto& st : ln.S) {
      if (st.kv8) {
        [rs addAllocation:st.k_i8]; [rs addAllocation:st.v_i8];
        [rs addAllocation:st.k_sc]; [rs addAllocation:st.v_sc];
      } else {
        [rs addAllocation:st.k_cache]; [rs addAllocation:st.v_cache];
      }
    }
    [rs commit];
    [rs requestResidency];
  }
  if (has_mtp) {
    for (auto& ml : MtpL) {
      const uint32_t mcap = ml.is_local ? ml.cap : gcap;
      const size_t kvb = size_t(mcap) * nkv * hd * sizeof(uint16_t);
      ln.mtp_k.push_back(C.buf(kvb));
      ln.mtp_v.push_back(C.buf(kvb));
      ln.mtp_cap.push_back(mcap);
      if (rs) { [rs addAllocation:ln.mtp_k.back()]; [rs addAllocation:ln.mtp_v.back()]; }
    }
    ln.mtp_f_rnn = C.buf(MP * H * sizeof(float));
    if (rs) [rs commit];
  }
  lanes_.push_back(std::move(ln));
  return int(lanes_.size()) - 1;
}

uint32_t Model::lane_global_cap(int k) const {
  const auto& SS = (k == active_lane) ? S : lanes_[k].S;
  for (const auto& st : SS) if (!L[st.phys_idx].is_local) return st.cap;
  return cap;
}

void Model::use_lane(int k) {
  if (k < 0 || k >= int(lanes_.size())) mtl::die("use_lane: invalid lane");
  if (k == active_lane) return;
  auto park = [&](Lane& ln) {  // swap the live state with a lane's parked copy
    std::swap(S, ln.S);
    std::swap(pos, ln.pos);
    std::swap(fork_pos, ln.fork_pos);
    std::swap(fork_src, ln.fork_src);
    std::swap(samp, ln.samp);
    if (has_mtp) {
      if (ln.mtp_k.size() != MtpL.size()) {
        ln.mtp_k.assign(MtpL.size(), nil);
        ln.mtp_v.assign(MtpL.size(), nil);
        ln.mtp_cap.assign(MtpL.size(), 0u);
      }
      for (size_t l = 0; l < MtpL.size(); l++) {
        std::swap(MtpL[l].k_cache, ln.mtp_k[l]);
        std::swap(MtpL[l].v_cache, ln.mtp_v[l]);
        std::swap(MtpL[l].cap, ln.mtp_cap[l]);
      }
      std::swap(mtp_ctx_pos, ln.mtp_ctx_pos);
      std::swap(b_mtp_f_rnn, ln.mtp_f_rnn);
    }
  };
  park(lanes_[active_lane]);  // live -> parked slot of the old lane (which held the empty placeholders)
  park(lanes_[k]);            // parked state of k -> live; k's slot now holds the placeholders
  active_lane = k;
  tap_row0 = 0;
}

void Model::fork_lane(int src, int dst) {
  if (src < 0 || src >= int(lanes_.size()) || dst < 0 || dst >= int(lanes_.size()) || src == dst)
    mtl::die("fork_lane: invalid lane");
  auto& src_S = (src == active_lane) ? S : lanes_[src].S;
  auto& dst_S = (dst == active_lane) ? S : lanes_[dst].S;
  const uint32_t P = (src == active_lane) ? pos : lanes_[src].pos;
  const int src_fsrc = (src == active_lane) ? fork_src : lanes_[src].fork_src;
  if (src_fsrc >= 0) mtl::die("fork_lane: cannot fork from an already-forked lane");
  (dst == active_lane ? pos : lanes_[dst].pos) = P;
  (dst == active_lane ? fork_pos : lanes_[dst].fork_pos) = P;
  (dst == active_lane ? fork_src : lanes_[dst].fork_src) = (P > 0u) ? src : -1;

  for (size_t i = 0; i < src_S.size(); i++) {
    const auto& ss = src_S[i];
    auto& ds = dst_S[i];
    memcpy([ds.k_conv contents], [ss.k_conv contents], [ss.k_conv length]);
    memcpy([ds.v_conv contents], [ss.v_conv contents], [ss.v_conv length]);
    memcpy([ds.attn_conv contents], [ss.attn_conv contents], [ss.attn_conv length]);
    memcpy([ds.mlp_conv contents], [ss.mlp_conv contents], [ss.mlp_conv length]);
  }

  id<MTLCommandBuffer> cb = [C.q commandBuffer];
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  const size_t kv_row_bf16 = size_t(nkv) * hd * sizeof(uint16_t);
  auto blit_ring = [&](id<MTLBuffer> sb, id<MTLBuffer> db, uint32_t len, uint32_t rcap) {
    if (len == 0u) return;
    const uint32_t t0 = (len > window) ? (len - window) : 0u;
    const uint32_t n = len - t0;
    const uint32_t s0 = t0 % rcap;
    const uint32_t first = std::min(n, rcap - s0);
    [blit copyFromBuffer:sb sourceOffset:size_t(s0) * kv_row_bf16
                toBuffer:db destinationOffset:size_t(s0) * kv_row_bf16
                    size:size_t(first) * kv_row_bf16];
    if (n > first) {
      [blit copyFromBuffer:sb sourceOffset:0 toBuffer:db destinationOffset:0
                      size:size_t(n - first) * kv_row_bf16];
    }
  };
  for (size_t i = 0; i < src_S.size(); i++) {
    if (!L[src_S[i].phys_idx].is_local) continue;
    blit_ring(src_S[i].k_cache, dst_S[i].k_cache, P, src_S[i].cap);
    blit_ring(src_S[i].v_cache, dst_S[i].v_cache, P, src_S[i].cap);
  }
  if (has_mtp) {
    const uint32_t mp = (src == active_lane) ? mtp_ctx_pos : lanes_[src].mtp_ctx_pos;
    (dst == active_lane ? mtp_ctx_pos : lanes_[dst].mtp_ctx_pos) = mp;
    id<MTLBuffer> sf = (src == active_lane) ? b_mtp_f_rnn : lanes_[src].mtp_f_rnn;
    id<MTLBuffer> df = (dst == active_lane) ? b_mtp_f_rnn : lanes_[dst].mtp_f_rnn;
    memcpy([df contents], [sf contents], size_t(H) * sizeof(float));
    for (size_t l = 0; l < MtpL.size(); l++) {
      if (!MtpL[l].is_local) continue;
      id<MTLBuffer> sk = (src == active_lane) ? MtpL[l].k_cache : lanes_[src].mtp_k[l];
      id<MTLBuffer> sv = (src == active_lane) ? MtpL[l].v_cache : lanes_[src].mtp_v[l];
      id<MTLBuffer> dk = (dst == active_lane) ? MtpL[l].k_cache : lanes_[dst].mtp_k[l];
      id<MTLBuffer> dv = (dst == active_lane) ? MtpL[l].v_cache : lanes_[dst].mtp_v[l];
      const uint32_t rcap = (src == active_lane) ? MtpL[l].cap : lanes_[src].mtp_cap[l];
      blit_ring(sk, dk, mp, rcap);
      blit_ring(sv, dv, mp, rcap);
    }
  }
  [blit endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.error) mtl::die("GPU fork_lane error", cb.error);
}

void Model::fork_lane_at(int src, int dst, uint32_t P, const std::vector<uint8_t>& conv) {
  if (src < 0 || src >= int(lanes_.size()) || dst < 0 || dst >= int(lanes_.size()) || src == dst)
    mtl::die("fork_lane_at: invalid lane");
  auto& src_S = (src == active_lane) ? S : lanes_[src].S;
  auto& dst_S = (dst == active_lane) ? S : lanes_[dst].S;
  const uint32_t src_pos = (src == active_lane) ? pos : lanes_[src].pos;
  const int src_fsrc = (src == active_lane) ? fork_src : lanes_[src].fork_src;
  if (src_fsrc >= 0) mtl::die("fork_lane_at: cannot fork from an already-forked lane");
  if (P > src_pos || src_pos - P > rewind_slack()) mtl::die("fork_lane_at: P beyond rewind slack");
  (dst == active_lane ? pos : lanes_[dst].pos) = P;
  (dst == active_lane ? fork_pos : lanes_[dst].fork_pos) = P;
  (dst == active_lane ? fork_src : lanes_[dst].fork_src) = (P > 0u) ? src : -1;

  size_t off = 0;
  for (size_t i = 0; i < dst_S.size(); i++) {
    auto& ds = dst_S[i];
    for (id<MTLBuffer> b : {ds.k_conv, ds.v_conv, ds.attn_conv, ds.mlp_conv}) {
      if (off + [b length] > conv.size()) mtl::die("fork_lane_at: conv snapshot size mismatch");
      memcpy([b contents], conv.data() + off, [b length]);
      off += [b length];
    }
  }

  id<MTLCommandBuffer> cb = [C.q commandBuffer];
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  const size_t kv_row_bf16 = size_t(nkv) * hd * sizeof(uint16_t);
  auto blit_ring = [&](id<MTLBuffer> sb, id<MTLBuffer> db, uint32_t len, uint32_t rcap) {
    if (len == 0u) return;
    const uint32_t t0 = (len > window) ? (len - window) : 0u;
    const uint32_t n = len - t0;
    const uint32_t s0 = t0 % rcap;
    const uint32_t first = std::min(n, rcap - s0);
    [blit copyFromBuffer:sb sourceOffset:size_t(s0) * kv_row_bf16
                toBuffer:db destinationOffset:size_t(s0) * kv_row_bf16
                    size:size_t(first) * kv_row_bf16];
    if (n > first) {
      [blit copyFromBuffer:sb sourceOffset:0 toBuffer:db destinationOffset:0
                      size:size_t(n - first) * kv_row_bf16];
    }
  };
  for (size_t i = 0; i < src_S.size(); i++) {
    if (!L[src_S[i].phys_idx].is_local) continue;
    blit_ring(src_S[i].k_cache, dst_S[i].k_cache, P, src_S[i].cap);
    blit_ring(src_S[i].v_cache, dst_S[i].v_cache, P, src_S[i].cap);
  }
  if (has_mtp) {
    (dst == active_lane ? mtp_ctx_pos : lanes_[dst].mtp_ctx_pos) = P;
    for (size_t l = 0; l < MtpL.size(); l++) {
      if (!MtpL[l].is_local) continue;
      id<MTLBuffer> sk = (src == active_lane) ? MtpL[l].k_cache : lanes_[src].mtp_k[l];
      id<MTLBuffer> sv = (src == active_lane) ? MtpL[l].v_cache : lanes_[src].mtp_v[l];
      id<MTLBuffer> dk = (dst == active_lane) ? MtpL[l].k_cache : lanes_[dst].mtp_k[l];
      id<MTLBuffer> dv = (dst == active_lane) ? MtpL[l].v_cache : lanes_[dst].mtp_v[l];
      const uint32_t rcap = (src == active_lane) ? MtpL[l].cap : lanes_[src].mtp_cap[l];
      blit_ring(sk, dk, P, rcap);
      blit_ring(sv, dv, P, rcap);
    }
  }
  [blit endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.error) mtl::die("GPU fork_lane_at error", cb.error);
}

void Model::promote_lane(int src, int dst) {
  if (src < 0 || src >= int(lanes_.size()) || dst < 0 || dst >= int(lanes_.size()) || src == dst)
    mtl::die("promote_lane: invalid lane");
  const int fsrc = (src == active_lane) ? fork_src : lanes_[src].fork_src;
  const uint32_t P = (src == active_lane) ? fork_pos : lanes_[src].fork_pos;
  const uint32_t E = (src == active_lane) ? pos : lanes_[src].pos;
  if (fsrc != dst || E < P) mtl::die("promote_lane: src is not forked from dst");
  auto& src_S = (src == active_lane) ? S : lanes_[src].S;
  auto& dst_S = (dst == active_lane) ? S : lanes_[dst].S;
  const uint32_t n_gen = E - P;
  if (n_gen > lane_global_cap(src) || E > lane_global_cap(dst)) mtl::die("promote_lane: suffix exceeds cap");

  (dst == active_lane ? pos : lanes_[dst].pos) = E;
  (dst == active_lane ? fork_pos : lanes_[dst].fork_pos) = 0;
  (dst == active_lane ? fork_src : lanes_[dst].fork_src) = -1;

  for (size_t i = 0; i < src_S.size(); i++) {
    const auto& ss = src_S[i];
    auto& ds = dst_S[i];
    memcpy([ds.k_conv contents], [ss.k_conv contents], [ss.k_conv length]);
    memcpy([ds.v_conv contents], [ss.v_conv contents], [ss.v_conv length]);
    memcpy([ds.attn_conv contents], [ss.attn_conv contents], [ss.attn_conv length]);
    memcpy([ds.mlp_conv contents], [ss.mlp_conv contents], [ss.mlp_conv length]);
  }

  id<MTLCommandBuffer> cb = [C.q commandBuffer];
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  const size_t kv_row_bf16 = size_t(nkv) * hd * sizeof(uint16_t);
  const size_t kv_row_i8 = size_t(nkv) * hd;
  const size_t sc_row = size_t(nkv) * sizeof(uint16_t);
  auto blit_ring = [&](id<MTLBuffer> sb, id<MTLBuffer> db, uint32_t len, uint32_t rcap) {
    if (len == 0u) return;
    const uint32_t t0 = (len > window) ? (len - window) : 0u;
    const uint32_t n = len - t0;
    const uint32_t s0 = t0 % rcap;
    const uint32_t first = std::min(n, rcap - s0);
    [blit copyFromBuffer:sb sourceOffset:size_t(s0) * kv_row_bf16
                toBuffer:db destinationOffset:size_t(s0) * kv_row_bf16
                    size:size_t(first) * kv_row_bf16];
    if (n > first) {
      [blit copyFromBuffer:sb sourceOffset:0 toBuffer:db destinationOffset:0
                      size:size_t(n - first) * kv_row_bf16];
    }
  };
  for (size_t i = 0; i < src_S.size(); i++) {
    const auto& ss = src_S[i];
    auto& ds = dst_S[i];
    if (L[ss.phys_idx].is_local) {
      blit_ring(ss.k_cache, ds.k_cache, E, ss.cap);
      blit_ring(ss.v_cache, ds.v_cache, E, ss.cap);
    } else if (n_gen > 0u) {
      if (ss.kv8) {
        [blit copyFromBuffer:ss.k_i8 sourceOffset:0 toBuffer:ds.k_i8 destinationOffset:size_t(P) * kv_row_i8 size:size_t(n_gen) * kv_row_i8];
        [blit copyFromBuffer:ss.v_i8 sourceOffset:0 toBuffer:ds.v_i8 destinationOffset:size_t(P) * kv_row_i8 size:size_t(n_gen) * kv_row_i8];
        [blit copyFromBuffer:ss.k_sc sourceOffset:0 toBuffer:ds.k_sc destinationOffset:size_t(P) * sc_row size:size_t(n_gen) * sc_row];
        [blit copyFromBuffer:ss.v_sc sourceOffset:0 toBuffer:ds.v_sc destinationOffset:size_t(P) * sc_row size:size_t(n_gen) * sc_row];
      } else {
        [blit copyFromBuffer:ss.k_cache sourceOffset:0 toBuffer:ds.k_cache destinationOffset:size_t(P) * kv_row_bf16 size:size_t(n_gen) * kv_row_bf16];
        [blit copyFromBuffer:ss.v_cache sourceOffset:0 toBuffer:ds.v_cache destinationOffset:size_t(P) * kv_row_bf16 size:size_t(n_gen) * kv_row_bf16];
      }
    }
  }
  if (has_mtp) {
    const uint32_t mp = (src == active_lane) ? mtp_ctx_pos : lanes_[src].mtp_ctx_pos;
    (dst == active_lane ? mtp_ctx_pos : lanes_[dst].mtp_ctx_pos) = mp;
    id<MTLBuffer> sf = (src == active_lane) ? b_mtp_f_rnn : lanes_[src].mtp_f_rnn;
    id<MTLBuffer> df = (dst == active_lane) ? b_mtp_f_rnn : lanes_[dst].mtp_f_rnn;
    memcpy([df contents], [sf contents], size_t(H) * sizeof(float));
    for (size_t l = 0; l < MtpL.size(); l++) {
      id<MTLBuffer> sk = (src == active_lane) ? MtpL[l].k_cache : lanes_[src].mtp_k[l];
      id<MTLBuffer> sv = (src == active_lane) ? MtpL[l].v_cache : lanes_[src].mtp_v[l];
      id<MTLBuffer> dk = (dst == active_lane) ? MtpL[l].k_cache : lanes_[dst].mtp_k[l];
      id<MTLBuffer> dv = (dst == active_lane) ? MtpL[l].v_cache : lanes_[dst].mtp_v[l];
      if (MtpL[l].is_local) {
        const uint32_t rcap = (src == active_lane) ? MtpL[l].cap : lanes_[src].mtp_cap[l];
        blit_ring(sk, dk, mp, rcap);
        blit_ring(sv, dv, mp, rcap);
      } else if (mp > P) {
        const uint32_t mg = mp - P;
        [blit copyFromBuffer:sk sourceOffset:0 toBuffer:dk destinationOffset:size_t(P) * kv_row_bf16 size:size_t(mg) * kv_row_bf16];
        [blit copyFromBuffer:sv sourceOffset:0 toBuffer:dv destinationOffset:size_t(P) * kv_row_bf16 size:size_t(mg) * kv_row_bf16];
      }
    }
  }
  [blit endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.error) mtl::die("GPU promote_lane error", cb.error);
}

std::vector<uint32_t> Model::step_lanes(const std::vector<std::pair<int, std::vector<uint32_t>>>& batches,
                                        double* gpu_ms, std::vector<float>* all_logits,
                                        std::map<std::string, double>* prof) {
  std::vector<uint32_t> ids;
  std::vector<LaneRef> refs;
  for (auto& [k, toks] : batches) {
    if (k < 0 || k >= int(lanes_.size())) mtl::die("step_lanes: invalid lane");
    if (toks.empty()) continue;
    auto* kS = (k == active_lane ? &S : &lanes_[k].S);
    for (auto& r : refs)
      if (r.S == kS) mtl::die("step_lanes: lane listed twice");
    const int fsrc = (k == active_lane ? fork_src : lanes_[k].fork_src);
    const uint32_t fpos = (k == active_lane ? fork_pos : lanes_[k].fork_pos);
    auto* pS = (fsrc >= 0) ? (fsrc == active_lane ? &S : &lanes_[fsrc].S) : nullptr;
    refs.push_back({kS, pS, k == active_lane ? &pos : &lanes_[k].pos, fpos,
                    uint32_t(ids.size()), uint32_t(toks.size()), k == active_lane ? &samp : &lanes_[k].samp});
    ids.insert(ids.end(), toks.begin(), toks.end());
  }
  if (ids.empty() || ids.size() > MAXM) mtl::die("step_lanes: total rows must be in 1..MAXM");
  return step_core(ids, refs, gpu_ms, nullptr, all_logits, prof);
}

void Model::save_conv(std::vector<uint8_t>& out) const {
  out.clear();
  for (const auto& st : S)
    for (id<MTLBuffer> b : {st.k_conv, st.v_conv, st.attn_conv, st.mlp_conv})
      out.insert(out.end(), (const uint8_t*)[b contents], (const uint8_t*)[b contents] + [b length]);
}

void Model::rewind(uint32_t c, const std::vector<uint8_t>& conv) {
  if (c > pos || pos - c > rewind_slack()) mtl::die("rewind: beyond the KV ring slack");
  size_t off = 0;
  for (auto& st : S)
    for (id<MTLBuffer> b : {st.k_conv, st.v_conv, st.attn_conv, st.mlp_conv}) {
      if (off + [b length] > conv.size()) mtl::die("rewind: conv snapshot size mismatch");
      memcpy([b contents], conv.data() + off, [b length]);
      off += [b length];
    }
  pos = c;
  if (has_mtp) mtp_ctx_pos = c;
}

void Model::reset() {
  pos = 0;
  fork_pos = 0;
  fork_src = -1;
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

std::vector<uint32_t> Model::step_all(const std::vector<uint32_t>& ids, std::vector<float>* all_logits) {
  tap_row0 = 0;
  auto* pS = (fork_src >= 0) ? (fork_src == active_lane ? &S : &lanes_[fork_src].S) : nullptr;
  return step_core(ids, {{&S, pS, &pos, fork_pos, 0u, uint32_t(ids.size()), &samp}}, nullptr, nullptr, all_logits, nullptr);
}

std::vector<uint32_t> Model::step(const std::vector<uint32_t>& ids, double* gpu_ms,
                                  std::vector<float>* last_logits,
                                  std::map<std::string, double>* prof) {
  tap_row0 = 0;
  auto* pS = (fork_src >= 0) ? (fork_src == active_lane ? &S : &lanes_[fork_src].S) : nullptr;
  return step_core(ids, {{&S, pS, &pos, fork_pos, 0u, uint32_t(ids.size()), &samp}}, gpu_ms, last_logits, nullptr, prof);
}

std::vector<uint32_t> Model::step_core(const std::vector<uint32_t>& ids, const std::vector<LaneRef>& lanes,
                                       double* gpu_ms, std::vector<float>* last_logits,
                                       std::vector<float>* all_logits, std::map<std::string, double>* prof) {
  const uint32_t M = uint32_t(ids.size());
  if (M == 0 || M > MP) mtl::die("step: M must be in 1..512");
  const bool pf = (M > MAXM);
  if (lanes.empty() || (pf && lanes.size() != 1u)) mtl::die("step: chunked prefill runs one lane at a time");
  // Single-lane paths (prefill) and per-step structural args use lane 0; per-lane ops use lr.pos / lr.S.
  const uint32_t pos = *lanes[0].pos;
  memcpy([b_ids contents], ids.data(), M * sizeof(uint32_t));

  id<MTLCommandBuffer> cb = [C.q commandBuffer];
  const bool conc = (prof == nullptr);
  auto new_enc = [&]() -> id<MTLComputeCommandEncoder> {
    return conc ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                : [cb computeCommandEncoder];
  };
  id<MTLComputeCommandEncoder> enc = new_enc();
  // INK_XBAR=n (screen only): n extra empty dispatch + barrier phases after every barrier, to price one phase.
  static const uint32_t xbar = getenv("INK_XBAR") ? uint32_t(atoi(getenv("INK_XBAR"))) : 0u;
  // INK_SKIP=mask (screen only, wrong output): drop op groups + their barriers from the stream to time them.
  static const uint32_t skipm = getenv("INK_SKIP") ? uint32_t(strtoul(getenv("INK_SKIP"), nullptr, 0)) : 0u;
  // INK_DUP=mask (screen only): run those op groups 1 + INK_DUPN times; the extra time prices one instance in-stream.
  static const uint32_t dupm = getenv("INK_DUP") ? uint32_t(strtoul(getenv("INK_DUP"), nullptr, 0)) : 0u;
  static const uint32_t dupn = getenv("INK_DUPN") ? uint32_t(atoi(getenv("INK_DUPN"))) : 1u;
  auto barrier = [&]() {
    if (!conc) return;
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    for (uint32_t k = 0; k < xbar; k++) {
      const uint32_t zero = 0u;
      [enc setComputePipelineState:p_copy];
      [enc setBuffer:b_dummy offset:0 atIndex:0];
      [enc setBuffer:b_dummy offset:0 atIndex:1];
      [enc setBytes:&zero length:4 atIndex:2];
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
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

  // Early commits (INK_EARLY, default on): hand finished segments to the GPU while the CPU encodes the rest.
  // Same queue = in-order execution, so outputs are unchanged; only encode latency is hidden.
  static const bool early = !(getenv("INK_EARLY") && atoi(getenv("INK_EARLY")) == 0);
  // Decode fusions (INK_FUSE=1; default off): bit-identical, but measured +0.2 ms at ctx 512 M=1 on M5 Ultra
  // (the single-TG fused kernels cost more than the ~4 us post-sconv phases they absorb).
  static const bool fuse = getenv("INK_FUSE") && atoi(getenv("INK_FUSE")) != 0;
  // Batch all n_core*4 loop messages per pass + skip tap-capture barriers + fused final attn_res_rms (default on, bit-identical).
  static const bool batch_msg = !(getenv("INK_BATCH_MSG") && atoi(getenv("INK_BATCH_MSG")) == 0);
  // Replicated-row stream fusions for M <= 16 (default 15 = all 4 on; wins on both Ultra and Base M5, bit-identical):
  //   1: attn_res+rms+scores (plain), 2: rms+scores (msg), 4: attn_res+mg_down (msg mlp), 8: attn_res+mg_down3 (msg attn).
  static const uint32_t stream_fuse = getenv("INK_STREAM_FUSE") ? uint32_t(atoi(getenv("INK_STREAM_FUSE"))) : 15u;
  // Fuse post-attention sconv + residual/boundary into o_proj mm2 epilogue (single-lane, bit-identical):
  //   0: off, 1 (default): M == 1 only, 2: M <= 16 (measured +0.13 ms at M=8 due to serial M-loop).
  static const uint32_t fuse_oconv = getenv("INK_FUSE_OCONV") ? uint32_t(atoi(getenv("INK_FUSE_OCONV"))) : 1u;
  // Fuse MoE accum + post-MLP sconv + residual add (ink_moe_accum_post_sconv, bit-identical):
  //   0: off, 1 (default): M == 1 only, 2: all M <= 16.
  static const uint32_t fuse_mconv = getenv("INK_FUSE_MCONV") ? uint32_t(atoi(getenv("INK_FUSE_MCONV"))) : 1u;
  // Decode router in one phase (INK_ROUTE_FAST, default on; same picks, weights and slots).
  static const bool route_fast = !(getenv("INK_ROUTE_FAST") && atoi(getenv("INK_ROUTE_FAST")) == 0);
  std::vector<id<MTLCommandBuffer>> done_cbs;
  auto early_commit = [&]() {
    if (prof || !early) return;
    [enc endEncoding];
    [cb commit];
    done_cbs.push_back(cb);
    cb = [C.q commandBuffer];
    enc = new_enc();
  };

  // NVFP4 expert GEMV (INK_MOE_GEMV=1, INK_GEMV_ROWS rows a pass, default 1): ink_moe_gu_gemv / ink_moe_dn_gemv over
  // the tiled bank, decode/verify only. M5 Ultra 2026-11-22, vbench2 ctx 2048: M=1 11.52 -> 11.13 ms, M=8 verify
  // 20.56 -> 23.1 (rows re-read the bank; R=2: 11.31 / 22.17). Off by default: DSpark verify is the serving path.
  static const bool moe_gemv = getenv("INK_MOE_GEMV") && atoi(getenv("INK_MOE_GEMV")) != 0;
  auto enc_moe_gemv = [&](id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> pipe, const Lin& lin,
                          id<MTLBuffer> X, id<MTLBuffer> Y, uint32_t M_, uint32_t ystride, uint32_t I_) {
    GemvArgs ga{lin.N, lin.K, M_, (lin.N * lin.K / 2u) / 16u, lin.N * (lin.K / lin.gs), 0u, ystride, I_};
    [e setComputePipelineState:pipe];
    [e setBuffer:lin.w offset:lin.woff atIndex:0];
    [e setBuffer:lin.s offset:lin.soff atIndex:1];
    [e setBuffer:X offset:0 atIndex:2];
    [e setBuffer:Y offset:0 atIndex:3];
    [e setBuffer:b_slots offset:0 atIndex:4];
    [e setBuffer:b_nslots offset:0 atIndex:5];
    [e setBuffer:b_rmask offset:0 atIndex:6];
    [e setBytes:&ga length:sizeof(ga) atIndex:7];
    const uint32_t gz = std::min<uint32_t>(M_ * 8u, lin.experts);
    [e dispatchThreadgroups:MTLSizeMake(lin.N / 32u, 1, gz) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  };

  auto copy_f32 = [&](id<MTLBuffer> src, size_t soff, id<MTLBuffer> dst, size_t doff, size_t bytes) {
    const uint32_t n4 = uint32_t(bytes / 16u);
    [enc setComputePipelineState:p_copy];
    [enc setBuffer:src offset:soff atIndex:0];
    [enc setBuffer:dst offset:doff atIndex:1];
    [enc setBytes:&n4 length:4 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((n4 + 255u) / 256u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  };

  const uint32_t b_stride = MP * H;
  const uint32_t n_core = core_end - core_start;

  // Sampled rows (lanes with temp > 0) redraw over the softcapped logits; greedy rows keep the argmax.
  // Always runs so every row gets its GPU Self-Certainty score in b_conf_out.
  auto enc_sample = [&](uint32_t n_rows, uint32_t pf_pos) {
    std::vector<SampRow> rows(n_rows);
    for (const auto& lr : lanes) {
      const Samp& sp = *lr.sp;
      for (uint32_t i = 0; i < lr.m; i++) {
        const uint32_t r = pf ? 0u : lr.r0 + i;
        if (pf && i + 1u != lr.m) continue;
        rows[r] = SampRow{vocab_valid, vocab_pad, pf ? pf_pos : *lr.pos + i, uint32_t(sp.seed), uint32_t(sp.seed >> 32),
                          sp.temp > 0.0f ? 1.0f / sp.temp : -1.0f, sp.min_p > 0.0f ? logf(sp.min_p) : -INFINITY,
                          sp.top_p, sp.top_k};
      }
    }
    memcpy([b_samp_rows contents], rows.data(), n_rows * sizeof(SampRow));
    barrier();
    [enc setComputePipelineState:p_sample_rows];
    [enc setBuffer:b_logits offset:0 atIndex:0];
    [enc setBuffer:b_tok_out offset:0 atIndex:1];
    [enc setBuffer:b_samp_rows offset:0 atIndex:2];
    [enc setBuffer:b_conf_out offset:0 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(n_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    flush_prof("sample");
  };

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
        copy_f32(b_h, 0, b_taps, size_t(ti) * MP * H * sizeof(float), size_t(M) * H * sizeof(float));
        if (!batch_msg) barrier();
      }
    }
  };

  auto run_layer = [&](uint32_t log_idx, uint32_t phys_idx, uint32_t& n_blocks, bool has_msgs, size_t msg_off = 0) {
    maybe_capture(log_idx);
    const auto& ly = L[phys_idx];
    auto& st = (*lanes[0].S)[log_idx];  // structure (cap, kv8) is identical across lanes

    // A + B + C. Pre-attention Block Attention Residual + optional Message Gate + Input RMSNorm -> bf16 fragments
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
    for (uint32_t _r = 0, _n = (((skipm & (1u << 0)) || (!has_msgs && (skipm & (1u << 16))) || (has_msgs && (skipm & (1u << 17)))) ? 0u : 1u + ((dupm >> 0) & 1u) * dupn); _r < _n; _r++) {  // INK_SKIP/INK_DUP bit 0 (16: plain, 17: msg): pre-attn attn_res/mg/rms
      if (_r > 0u) barrier();
    if (phys_idx == 0) {
      enc_rms(b_h, 0, ly.attn_norm, b_frag_q, nil, zero_u);
      barrier();
      flush_prof("rms");
    } else if (!has_msgs) {
      // Fused attn_res + rms_frag (zero DRAM round-trip, zero intermediate barrier!)
      const uint32_t nq = ly.is_local ? 1u : 3u;
      AttnResRmsArgs arra{H, M, n_blocks, b_stride, 0u, eps};
      [enc setComputePipelineState:p_attn_res_rms];
      [enc setBuffer:b_blocks offset:0 atIndex:0];
      [enc setBuffer:b_h offset:0 atIndex:1];
      [enc setBuffer:ly.ar_attn_q.b offset:ly.ar_attn_q.off atIndex:2];
      const Raw& w0 = ly.is_local ? ly.attn_norm : ly.q_norm;
      const Raw& w1 = ly.is_local ? ly.attn_norm : ly.k_norm;
      const Raw& w2 = ly.is_local ? ly.attn_norm : ly.v_norm;
      [enc setBuffer:w0.b offset:w0.off atIndex:3];
      [enc setBuffer:w1.b offset:w1.off atIndex:4];
      [enc setBuffer:w2.b offset:w2.off atIndex:5];
      [enc setBuffer:b_frag_q offset:0 atIndex:6];
      [enc setBuffer:b_frag_k offset:0 atIndex:7];
      [enc setBuffer:b_frag_v offset:0 atIndex:8];
      [enc setBuffer:b_dummy offset:0 atIndex:9];
      [enc setBytes:&arra length:sizeof(arra) atIndex:10];
      [enc dispatchThreadgroups:MTLSizeMake(M, nq, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      if (!ly.is_local) { fk = b_frag_k; fv = b_frag_v; }
      barrier();
      flush_prof("attn_res");
    } else {
      // Fused attn_res + mg_prep -> batched mg_down3 -> batched mg_apply3 (zero local barrier!) -> rms
      const uint32_t cat_stride_b4 = (MP * 2u * H) / 4u;
      const uint32_t low_stride_f = MP * gate_rank;
      const uint32_t q_shared = ly.is_local ? 1u : 0u;

      if (!pf && (stream_fuse & 8u)) {
        AttnResMgDown3Args armd{H, gate_rank, M, n_blocks, b_stride, q_shared, low_stride_f, eps};
        [enc setComputePipelineState:p_attn_res_mg_down3];
        [enc setBuffer:b_blocks offset:0 atIndex:0];
        [enc setBuffer:b_h offset:0 atIndex:1];
        [enc setBuffer:ly.ar_attn_q.b offset:ly.ar_attn_q.off atIndex:2];
        [enc setBuffer:b_msgs offset:msg_off atIndex:3];
        [enc setBuffer:b_qkv_in offset:0 atIndex:4];
        [enc setBuffer:ly.ctrl_q.down.b offset:ly.ctrl_q.down.off atIndex:5];
        [enc setBuffer:ly.ctrl_k.down.b offset:ly.ctrl_k.down.off atIndex:6];
        [enc setBuffer:ly.ctrl_v.down.b offset:ly.ctrl_v.down.off atIndex:7];
        [enc setBuffer:b_mg_low offset:0 atIndex:8];
        [enc setBytes:&armd length:sizeof(armd) atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake(M, gate_rank / 8u, 3) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();
        flush_prof("attn_res");

        MsgGate3Args mg3{H, gate_rank, M, q_shared, cat_stride_b4, low_stride_f};
        [enc setComputePipelineState:p_mg_apply3];
        [enc setBuffer:b_qkv_in offset:0 atIndex:0];
        [enc setBuffer:b_msgs offset:msg_off atIndex:1];
        [enc setBuffer:b_mg_low offset:0 atIndex:2];
        [enc setBuffer:ly.ctrl_q.forget.b offset:ly.ctrl_q.forget.off atIndex:3];
        [enc setBuffer:ly.ctrl_q.gain.b offset:ly.ctrl_q.gain.off atIndex:4];
        [enc setBuffer:ly.ctrl_k.forget.b offset:ly.ctrl_k.forget.off atIndex:5];
        [enc setBuffer:ly.ctrl_k.gain.b offset:ly.ctrl_k.gain.off atIndex:6];
        [enc setBuffer:ly.ctrl_v.forget.b offset:ly.ctrl_v.forget.off atIndex:7];
        [enc setBuffer:ly.ctrl_v.gain.b offset:ly.ctrl_v.gain.off atIndex:8];
        [enc setBuffer:b_qkv_out offset:0 atIndex:9];
        [enc setBytes:&mg3 length:sizeof(mg3) atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake(M, (H + 63u) / 64u, 3) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      } else {
      AttnResMgArgs arm{H, M, n_blocks, b_stride, q_shared, cat_stride_b4, eps};
      [enc setComputePipelineState:(q_shared ? p_attn_res_mg_prep_shared : p_attn_res_mg_prep)];
      [enc setBuffer:b_blocks offset:0 atIndex:0];
      [enc setBuffer:b_h offset:0 atIndex:1];
      [enc setBuffer:ly.ar_attn_q.b offset:ly.ar_attn_q.off atIndex:2];
      [enc setBuffer:b_msgs offset:msg_off atIndex:3];
      [enc setBuffer:b_qkv_in offset:0 atIndex:4];
      [enc setBuffer:b_mg_cat offset:0 atIndex:5];
      [enc setBytes:&arm length:sizeof(arm) atIndex:6];
      [enc dispatchThreadgroups:MTLSizeMake(M, q_shared ? 1u : 3u, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      barrier();
      flush_prof("attn_res");

      if (pf) {
        MgMmaArgs mgm{H, gate_rank, M, q_shared, MP * 2u * H, low_stride_f};
        [enc setComputePipelineState:p_mg_down_mma];
        [enc setBuffer:b_mg_cat offset:0 atIndex:0];
        [enc setBuffer:ly.ctrl_q.down.b offset:ly.ctrl_q.down.off atIndex:1];
        [enc setBuffer:ly.ctrl_k.down.b offset:ly.ctrl_k.down.off atIndex:2];
        [enc setBuffer:ly.ctrl_v.down.b offset:ly.ctrl_v.down.off atIndex:3];
        [enc setBuffer:b_mg_low offset:0 atIndex:4];
        [enc setBytes:&mgm length:sizeof(mgm) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((M + 15u) / 16u, gate_rank / 32u, 3) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        barrier();

        [enc setComputePipelineState:p_mg_apply_mma];
        [enc setBuffer:b_qkv_in offset:0 atIndex:0];
        [enc setBuffer:b_msgs offset:msg_off atIndex:1];
        [enc setBuffer:b_mg_low offset:0 atIndex:2];
        [enc setBuffer:ly.ctrl_q.forget.b offset:ly.ctrl_q.forget.off atIndex:3];
        [enc setBuffer:ly.ctrl_q.gain.b offset:ly.ctrl_q.gain.off atIndex:4];
        [enc setBuffer:ly.ctrl_k.forget.b offset:ly.ctrl_k.forget.off atIndex:5];
        [enc setBuffer:ly.ctrl_k.gain.b offset:ly.ctrl_k.gain.off atIndex:6];
        [enc setBuffer:ly.ctrl_v.forget.b offset:ly.ctrl_v.forget.off atIndex:7];
        [enc setBuffer:ly.ctrl_v.gain.b offset:ly.ctrl_v.gain.off atIndex:8];
        [enc setBuffer:b_qkv_out offset:0 atIndex:9];
        [enc setBytes:&mgm length:sizeof(mgm) atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake((M + 63u) / 64u, (H + 31u) / 32u, 3) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      } else {
        MsgGate3Args mg3{H, gate_rank, M, q_shared, cat_stride_b4, low_stride_f};
        [enc setComputePipelineState:p_mg_down3];
        [enc setBuffer:b_mg_cat offset:0 atIndex:0];
        [enc setBuffer:ly.ctrl_q.down.b offset:ly.ctrl_q.down.off atIndex:1];
        [enc setBuffer:ly.ctrl_k.down.b offset:ly.ctrl_k.down.off atIndex:2];
        [enc setBuffer:ly.ctrl_v.down.b offset:ly.ctrl_v.down.off atIndex:3];
        [enc setBuffer:b_mg_low offset:0 atIndex:4];
        [enc setBytes:&mg3 length:sizeof(mg3) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(M, gate_rank / 4u, 3) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        barrier();

        [enc setComputePipelineState:p_mg_apply3];
        [enc setBuffer:b_qkv_in offset:0 atIndex:0];
        [enc setBuffer:b_msgs offset:msg_off atIndex:1];
        [enc setBuffer:b_mg_low offset:0 atIndex:2];
        [enc setBuffer:ly.ctrl_q.forget.b offset:ly.ctrl_q.forget.off atIndex:3];
        [enc setBuffer:ly.ctrl_q.gain.b offset:ly.ctrl_q.gain.off atIndex:4];
        [enc setBuffer:ly.ctrl_k.forget.b offset:ly.ctrl_k.forget.off atIndex:5];
        [enc setBuffer:ly.ctrl_k.gain.b offset:ly.ctrl_k.gain.off atIndex:6];
        [enc setBuffer:ly.ctrl_v.forget.b offset:ly.ctrl_v.forget.off atIndex:7];
        [enc setBuffer:ly.ctrl_v.gain.b offset:ly.ctrl_v.gain.off atIndex:8];
        [enc setBuffer:b_qkv_out offset:0 atIndex:9];
        [enc setBytes:&mg3 length:sizeof(mg3) atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake(M, (H + 63u) / 64u, 3) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      }
      }
      barrier();
      flush_prof("msg_gate");

      if (!pf && stream_fuse) {
        const Raw& w0 = ly.is_local ? ly.attn_norm : ly.q_norm;
        const Raw& w1 = ly.is_local ? ly.attn_norm : ly.k_norm;
        const Raw& w2 = ly.is_local ? ly.attn_norm : ly.v_norm;
        [enc setComputePipelineState:p_rms_frag3];
        [enc setBuffer:b_qkv_out offset:0 atIndex:0];
        [enc setBuffer:w0.b offset:w0.off atIndex:1];
        [enc setBuffer:w1.b offset:w1.off atIndex:2];
        [enc setBuffer:w2.b offset:w2.off atIndex:3];
        [enc setBuffer:b_frag_q offset:0 atIndex:4];
        [enc setBuffer:b_frag_k offset:0 atIndex:5];
        [enc setBuffer:b_frag_v offset:0 atIndex:6];
        [enc setBytes:&H length:4 atIndex:7];
        [enc setBytes:&M length:4 atIndex:8];
        [enc setBytes:&eps length:4 atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake(M, 3, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      } else {
        const size_t stride_m = size_t(M) * H * sizeof(float);
        enc_rms(b_qkv_out, 0u, ly.is_local ? ly.attn_norm : ly.q_norm, b_frag_q, nil, zero_u);
        enc_rms(b_qkv_out, 1u * stride_m, ly.is_local ? ly.attn_norm : ly.k_norm, b_frag_k, nil, zero_u);
        enc_rms(b_qkv_out, 2u * stride_m, ly.is_local ? ly.attn_norm : ly.v_norm, b_frag_v, nil, zero_u);
      }
      fk = b_frag_k;
      fv = b_frag_v;
      barrier();
      flush_prof("rms");
    }

    }
    for (uint32_t _r = 0, _n = ((skipm & (1u << 1)) ? 0u : 1u + ((dupm >> 1) & 1u) * dupn); _r < _n; _r++) {  // INK_SKIP/INK_DUP bit 1: qkv proj
      if (_r > 0u) barrier();
    // D. Attention projections (W8): qgrs || k || v concurrently
    if (pf) {
      enc_mmp(enc, p_mmp_w8, ly.qgrs, b_frag_q, 0, b_qgrs, 0, M, b_dummy);
      if (prof) flush_prof("attn_qgrs");
      enc_mmp(enc, p_mmp_w8, ly.k_proj, fk, 0, b_k_raw, 0, M, b_dummy);
      enc_mmp(enc, p_mmp_w8, ly.v_proj, fv, 0, b_v_raw, 0, M, b_dummy);
    } else {
      enc_mm2(enc, p_mm_w8, ly.qgrs, b_frag_q, 0, b_qgrs, 0, M, nil, b_dummy, nil, nil, nil, false);
      if (prof) flush_prof("attn_qgrs");
      enc_mm2(enc, p_mm_w8, ly.k_proj, fk, 0, b_k_raw, 0, M, nil, b_dummy, nil, nil, nil, false, 0, 0, 0, big_gpu);
      enc_mm2(enc, p_mm_w8, ly.v_proj, fv, 0, b_v_raw, 0, M, nil, b_dummy, nil, nil, nil, false, 0, 0, 0, big_gpu);
    }
    barrier();
    flush_prof("attn_kv");

    }
    for (uint32_t _r = 0, _n = ((skipm & (1u << 2)) ? 0u : 1u + ((dupm >> 2) & 1u) * dupn); _r < _n; _r++) {  // INK_SKIP/INK_DUP bit 2: q_rel + kv_conv
      if (_r > 0u) barrier();
    // E + F. Q+Rel || K sconv || V sconv concurrently
    QRelArgs qra{nh, hd, d_rel, rel_extent, ly.qgrs.N, pos, ly.is_local ? 0u : 1u,
                 qk_clip, rel_clip, log_n_floor, log_alpha};
    const bool qrel_pf = pf && qrel_mma;
    if (qrel_pf) {
      [enc setComputePipelineState:p_q_rel_prep];
      [enc setBuffer:b_qgrs offset:0 atIndex:0];
      [enc setBuffer:ly.q_scale.b offset:ly.q_scale.off atIndex:1];
      [enc setBuffer:ly.rel_scale.b offset:ly.rel_scale.off atIndex:2];
      [enc setBuffer:b_Q offset:0 atIndex:3];
      [enc setBuffer:b_rel_vec offset:0 atIndex:4];
      [enc setBuffer:b_gate offset:0 atIndex:5];
      [enc setBuffer:b_sinks offset:0 atIndex:6];
      [enc setBytes:&qra length:sizeof(qra) atIndex:7];
      [enc dispatchThreadgroups:MTLSizeMake(M, nh, 1) threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
    } else {
      for (const auto& lr : lanes) {
        QRelArgs qrl{nh, hd, d_rel, rel_extent, ly.qgrs.N, *lr.pos, ly.is_local ? 0u : 1u,
                     qk_clip, rel_clip, log_n_floor, log_alpha};
        const size_t r0 = lr.r0;
        [enc setComputePipelineState:p_q_rel];
        [enc setBuffer:b_qgrs offset:r0 * ly.qgrs.N * sizeof(float) atIndex:0];
        [enc setBuffer:ly.q_scale.b offset:ly.q_scale.off atIndex:1];
        [enc setBuffer:ly.rel_scale.b offset:ly.rel_scale.off atIndex:2];
        [enc setBuffer:ly.rel_proj.b offset:ly.rel_proj.off atIndex:3];
        [enc setBuffer:b_Q offset:r0 * nh * hd * sizeof(float) atIndex:4];
        [enc setBuffer:b_rel_logits offset:r0 * nh * rel_extent * sizeof(float) atIndex:5];
        [enc setBuffer:b_gate offset:r0 * nh * hd * sizeof(float) atIndex:6];
        [enc setBuffer:b_sinks offset:r0 * nh * sizeof(float) atIndex:7];
        [enc setBytes:&qrl length:sizeof(qrl) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(lr.m, nh, 1) threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
      }
    }
    if (prof) flush_prof("q_rel");

    for (const auto& lr : lanes) {
      auto& sl = (*lr.S)[log_idx];
      const uint32_t fpos = (!ly.is_local && lr.pref_S) ? lr.fork_pos : 0u;
      KvConvArgs kca{lr.m, nkv, hd, ly.k_proj.N, *lr.pos, sl.cap, ly.is_local ? 0u : 1u, 1u, ks, sl.kv8 ? 1u : 0u, fpos, qk_clip, eps};
      [enc setComputePipelineState:p_kv_conv];
      [enc setBuffer:b_k_raw offset:size_t(lr.r0) * ly.k_proj.N * sizeof(float) atIndex:0];
      [enc setBuffer:ly.k_sconv.b offset:ly.k_sconv.off atIndex:1];
      [enc setBuffer:sl.k_conv offset:0 atIndex:2];
      [enc setBuffer:(sl.k_cache ? sl.k_cache : b_dummy) offset:0 atIndex:3];
      [enc setBuffer:(sl.k_i8 ? sl.k_i8 : b_dummy) offset:0 atIndex:4];
      [enc setBuffer:(sl.k_sc ? sl.k_sc : b_dummy) offset:0 atIndex:5];
      [enc setBuffer:sl.k_hist offset:0 atIndex:6];
      [enc setBytes:&kca length:sizeof(kca) atIndex:7];
      [enc setBuffer:sl.k_conv_next offset:0 atIndex:8];
      [enc dispatchThreadgroups:MTLSizeMake(lr.m, nkv, 1) threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
      std::swap(sl.k_conv, sl.k_conv_next);

      KvConvArgs vca{lr.m, nkv, hd, ly.v_proj.N, *lr.pos, sl.cap, ly.is_local ? 0u : 1u, 0u, ks, sl.kv8 ? 1u : 0u, fpos, qk_clip, eps};
      [enc setBuffer:b_v_raw offset:size_t(lr.r0) * ly.v_proj.N * sizeof(float) atIndex:0];
      [enc setBuffer:ly.v_sconv.b offset:ly.v_sconv.off atIndex:1];
      [enc setBuffer:sl.v_conv offset:0 atIndex:2];
      [enc setBuffer:(sl.v_cache ? sl.v_cache : b_dummy) offset:0 atIndex:3];
      [enc setBuffer:(sl.v_i8 ? sl.v_i8 : b_dummy) offset:0 atIndex:4];
      [enc setBuffer:(sl.v_sc ? sl.v_sc : b_dummy) offset:0 atIndex:5];
      [enc setBuffer:sl.v_hist offset:0 atIndex:6];
      [enc setBytes:&vca length:sizeof(vca) atIndex:7];
      [enc setBuffer:sl.v_conv_next offset:0 atIndex:8];
      [enc dispatchThreadgroups:MTLSizeMake(lr.m, nkv, 1) threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
      std::swap(sl.v_conv, sl.v_conv_next);
    }
    barrier();
    flush_prof("kv_conv");

    if (qrel_pf) {
      // rel_logits[M * nh, ext] = rel_vec[M * nh, 16] @ rel_proj[16, rel_extent], ext = distances this chunk can reach
      const uint32_t R = M * nh;
      const uint32_t ext = std::min(rel_extent, pos + M);
      [enc setComputePipelineState:p_q_rel_mma];
      [enc setBuffer:b_rel_vec offset:0 atIndex:0];
      [enc setBuffer:ly.rel_proj.b offset:ly.rel_proj.off atIndex:1];
      [enc setBuffer:b_rel_logits offset:0 atIndex:2];
      [enc setBytes:&R length:4 atIndex:3];
      [enc setBytes:&rel_extent length:4 atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake((R + 63u) / 64u, (ext + 31u) / 32u, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      barrier();
      flush_prof("q_rel");
    }

    }
    for (uint32_t _r = 0, _n = ((skipm & (1u << 3)) ? 0u : 1u + ((dupm >> 3) & 1u) * dupn); _r < _n; _r++) {  // INK_SKIP/INK_DUP bit 3: attention
      if (_r > 0u) barrier();
    // G. Attention: Metal 4 matmul2d FlashAttention for prefill (pf), split-K GQA for decode/verify
    if (pf) {
      const auto* pl = (!ly.is_local && lanes[0].pref_S) ? &(*lanes[0].pref_S)[log_idx] : nullptr;
      const uint32_t fpos = pl ? lanes[0].fork_pos : 0u;
      AttnPfArgs pfa{M, nh, nkv, hd, rel_extent, pos, st.cap, window, fpos};
      id<MTLComputePipelineState> p_pf = ly.is_local ? p_attn_pf_local
                                                     : (fpos > 0u
                                                            ? (st.kv8 ? p_attn_pf_global_kv8_fork : p_attn_pf_global_bf16_fork)
                                                            : (st.kv8 ? p_attn_pf_global_kv8 : p_attn_pf_global_bf16));
      [enc setComputePipelineState:p_pf];
      [enc setBuffer:b_Q offset:0 atIndex:0];
      [enc setBuffer:(st.k_cache ? st.k_cache : b_dummy) offset:0 atIndex:1];
      [enc setBuffer:(st.v_cache ? st.v_cache : b_dummy) offset:0 atIndex:2];
      [enc setBuffer:b_rel_logits offset:0 atIndex:3];
      [enc setBuffer:b_gate offset:0 atIndex:4];
      [enc setBuffer:b_sinks offset:0 atIndex:5];
      [enc setBuffer:b_frag_o offset:0 atIndex:6];
      [enc setBuffer:(st.k_i8 ? st.k_i8 : b_dummy) offset:0 atIndex:7];
      [enc setBuffer:(st.k_sc ? st.k_sc : b_dummy) offset:0 atIndex:8];
      [enc setBuffer:(st.v_i8 ? st.v_i8 : b_dummy) offset:0 atIndex:9];
      [enc setBuffer:(st.v_sc ? st.v_sc : b_dummy) offset:0 atIndex:10];
      [enc setBytes:&pfa length:sizeof(pfa) atIndex:11];
      if (fpos > 0u) {
        [enc setBuffer:(pl->k_cache ? pl->k_cache : b_dummy) offset:0 atIndex:12];
        [enc setBuffer:(pl->v_cache ? pl->v_cache : b_dummy) offset:0 atIndex:13];
        [enc setBuffer:(pl->k_i8 ? pl->k_i8 : b_dummy) offset:0 atIndex:14];
        [enc setBuffer:(pl->k_sc ? pl->k_sc : b_dummy) offset:0 atIndex:15];
        [enc setBuffer:(pl->v_i8 ? pl->v_i8 : b_dummy) offset:0 atIndex:16];
        [enc setBuffer:(pl->v_sc ? pl->v_sc : b_dummy) offset:0 atIndex:17];
      }
      [enc dispatchThreadgroups:MTLSizeMake((M + 15u) / 16u, nh, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      barrier();
      flush_prof("attn");
    } else {
      const uint32_t nsp = ly.is_local ? nsp_local : nsp_global;
      id<MTLComputePipelineState> p_dec = ly.is_local ? p_attn_dec_local
                                                      : (st.kv8 ? p_attn_dec_global_kv8 : p_attn_dec_global_bf16);
      for (const auto& lr : lanes) {
        auto& sl = (*lr.S)[log_idx];
        const auto* pl = (!ly.is_local && lr.pref_S) ? &(*lr.pref_S)[log_idx] : nullptr;
        const uint32_t fpos = pl ? lr.fork_pos : 0u;
        const size_t r0 = lr.r0;
        AttnDecArgs da{lr.m, nh, nkv, hd, rel_extent, *lr.pos, sl.cap, window, nsp, fpos};
        [enc setComputePipelineState:p_dec];
        [enc setBuffer:b_Q offset:r0 * nh * hd * sizeof(float) atIndex:0];
        [enc setBuffer:(sl.k_cache ? sl.k_cache : b_dummy) offset:0 atIndex:1];
        [enc setBuffer:(sl.v_cache ? sl.v_cache : b_dummy) offset:0 atIndex:2];
        [enc setBuffer:b_rel_logits offset:r0 * nh * rel_extent * sizeof(float) atIndex:3];
        [enc setBuffer:b_attn_part offset:r0 * nh * nsp * 130u * sizeof(float) atIndex:4];
        [enc setBuffer:(sl.k_i8 ? sl.k_i8 : b_dummy) offset:0 atIndex:5];
        [enc setBuffer:(sl.k_sc ? sl.k_sc : b_dummy) offset:0 atIndex:6];
        [enc setBuffer:(sl.v_i8 ? sl.v_i8 : b_dummy) offset:0 atIndex:7];
        [enc setBuffer:(sl.v_sc ? sl.v_sc : b_dummy) offset:0 atIndex:8];
        [enc setBytes:&da length:sizeof(da) atIndex:9];
        [enc setBuffer:(pl && pl->k_cache ? pl->k_cache : b_dummy) offset:0 atIndex:10];
        [enc setBuffer:(pl && pl->v_cache ? pl->v_cache : b_dummy) offset:0 atIndex:11];
        [enc setBuffer:(pl && pl->k_i8 ? pl->k_i8 : b_dummy) offset:0 atIndex:12];
        [enc setBuffer:(pl && pl->k_sc ? pl->k_sc : b_dummy) offset:0 atIndex:13];
        [enc setBuffer:(pl && pl->v_i8 ? pl->v_i8 : b_dummy) offset:0 atIndex:14];
        [enc setBuffer:(pl && pl->v_sc ? pl->v_sc : b_dummy) offset:0 atIndex:15];
        [enc dispatchThreadgroups:MTLSizeMake(lr.m, nkv, nsp) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      }
      barrier();

      AttnArgs aa{M, nh, nkv, hd, rel_extent, pos, st.cap, window, ly.is_local ? 1u : 0u, nsp, st.kv8 ? 1u : 0u};
      [enc setComputePipelineState:p_attn_reduce];
      [enc setBuffer:b_attn_part offset:0 atIndex:0];
      [enc setBuffer:b_gate offset:0 atIndex:1];
      [enc setBuffer:b_sinks offset:0 atIndex:2];
      [enc setBuffer:b_frag_o offset:0 atIndex:3];
      [enc setBytes:&aa length:sizeof(aa) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake(M, nh, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      barrier();
      flush_prof("attn");
    }

    }
    const bool use_oconv = !pf && !fuse && ((fuse_oconv >= 2u && M <= 16u) || (fuse_oconv == 1u && M == 1u)) && lanes.size() == 1u && ks == 4u;
    for (uint32_t _r = 0, _n = ((skipm & (1u << 4)) ? 0u : 1u + ((dupm >> 4) & 1u) * dupn); _r < _n; _r++) {  // INK_SKIP/INK_DUP bit 4: o_proj
      if (_r > 0u) barrier();
    if (pf) {
      enc_mmp(enc, p_mmp_w8, ly.o_proj, b_frag_o, 0, b_o_out, 0, M, b_dummy);
    } else if (use_oconv) {
      auto& sl = (*lanes[0].S)[log_idx];
      MmArgs a{};
      a.N = ly.o_proj.N; a.K = ly.o_proj.K; a.M = M;
      a.sgs = ly.o_proj.sgs;
      a.ksplit = ly.o_proj.ksplit;
      a.psplit = 16u * ly.o_proj.N;
      a.R = b_stride;
      a.H = ly.is_boundary ? 1u : 0u;
      a.HC = n_blocks;
      a.pf_z0 = ~0u;
      static const int nft_env = getenv("INK_NFT") ? atoi(getenv("INK_NFT")) : -1;
      bool pref1 = big_gpu;
      if (nft_env == 1) pref1 = true;
      else if (nft_env == 2) pref1 = false;
      const bool use_nft2 = !pref1 && ((a.N / 32u) % 2u == 0u && a.sgs <= 4u);
      const uint32_t nsg = use_nft2 ? (a.sgs * 2u) : a.sgs;
      const uint32_t nft = nsg / a.sgs;
      const uint32_t gx = (a.N / 32u + nft - 1u) / nft;
      [enc setComputePipelineState:p_mm_w8_oconv];
      [enc setBuffer:ly.o_proj.w offset:ly.o_proj.woff atIndex:0];
      [enc setBuffer:ly.o_proj.s offset:ly.o_proj.soff atIndex:1];
      [enc setBuffer:b_frag_o offset:0 atIndex:2];
      [enc setBuffer:(a.ksplit > 1u ? g_mmpart : b_dummy) offset:(a.ksplit > 1u ? ly.o_proj.pslot * kSplitPartBytes : 0) atIndex:3];
      [enc setBuffer:b_o_out offset:0 atIndex:4];
      [enc setBytes:&a length:sizeof(a) atIndex:5];
      [enc setBuffer:b_dummy offset:0 atIndex:6];
      [enc setBuffer:b_dummy offset:0 atIndex:7];
      [enc setBuffer:(a.ksplit > 1u ? g_mmcnt : b_dummy) offset:(a.ksplit > 1u ? ly.o_proj.pslot * kSplitCntBytes : 0) atIndex:8];
      [enc setBuffer:ly.attn_sconv.b offset:ly.attn_sconv.off atIndex:9];
      [enc setBuffer:sl.attn_conv offset:0 atIndex:10];
      [enc setBuffer:b_h offset:0 atIndex:11];
      [enc setBuffer:b_blocks offset:0 atIndex:12];
      [enc setBuffer:sl.attn_hist offset:0 atIndex:13];
      [enc setBuffer:sl.attn_conv_next offset:0 atIndex:14];
      [enc setBuffer:b_dummy offset:0 atIndex:15];
      [enc setThreadgroupMemoryLength:size_t(nsg) * 16u * 32u * sizeof(float) atIndex:0];
      [enc dispatchThreadgroups:MTLSizeMake(gx, a.ksplit, 1) threadsPerThreadgroup:MTLSizeMake(nsg * 32u, 1, 1)];
      std::swap(sl.attn_conv, sl.attn_conv_next);
    } else {
      enc_mm2(enc, p_mm_w8, ly.o_proj, b_frag_o, 0, b_o_out, 0, M, nil, b_dummy, nil, nil, nil, false, 0, 0, 0, big_gpu);
    }
    barrier();
    flush_prof("attn_o");

    }
    const bool use_mid_plain = !pf && !has_msgs && ((stream_fuse & 1u) != 0u);
    const bool use_rms_scores = !pf && has_msgs && ((stream_fuse & 2u) != 0u);
    const bool use_mid_msg_down = !pf && has_msgs && ((stream_fuse & 4u) != 0u);
    const bool fuse_h = !pf && fuse;
    const bool skip_h = fuse_h || use_oconv;
    const uint32_t h_slot = n_blocks;
    for (uint32_t _r = 0, _n = ((skipm & (1u << 5)) ? 0u : 1u + ((dupm >> 5) & 1u) * dupn); _r < _n; _r++) {  // INK_SKIP/INK_DUP bit 5: post_conv attn
      if (_r > 0u) barrier();
    // H. Post-attention sconv + block boundary / residual add (decode plain layers with fuse: done inside the
    //    pre-MLP attn_res_rms phase below; with use_oconv: done inside o_proj mm2 epilogue above)
    if (!skip_h)
    for (const auto& lr : lanes) {
      auto& sl = (*lr.S)[log_idx];
      const size_t r0 = lr.r0;
      PostConvArgs pca{lr.m, H, ks, ly.is_boundary ? 1u : 0u, 0u, n_blocks, b_stride};
      [enc setComputePipelineState:p_post_conv];
      [enc setBuffer:b_o_out offset:r0 * H * sizeof(float) atIndex:0];
      [enc setBuffer:ly.attn_sconv.b offset:ly.attn_sconv.off atIndex:1];
      [enc setBuffer:sl.attn_conv offset:0 atIndex:2];
      [enc setBuffer:b_h offset:r0 * H * sizeof(float) atIndex:3];
      [enc setBuffer:b_blocks offset:r0 * H * sizeof(float) atIndex:4];
      [enc setBuffer:sl.attn_hist offset:0 atIndex:5];
      [enc setBytes:&pca length:sizeof(pca) atIndex:6];
      [enc setBuffer:sl.attn_conv_next offset:0 atIndex:7];
      [enc dispatchThreadgroups:MTLSizeMake(lr.m, (H + 255u) / 256u, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      std::swap(sl.attn_conv, sl.attn_conv_next);
    }
    }
    if (ly.is_boundary) n_blocks++;
    if (!skip_h) {
      barrier();
      flush_prof("post_conv");
    }

    for (uint32_t _r = 0, _n = (((skipm & (1u << 6)) || (!has_msgs && (skipm & (1u << 14))) || (has_msgs && (skipm & (1u << 15)))) ? 0u : 1u + ((dupm >> 6) & 1u) * dupn); _r < _n; _r++) {  // INK_SKIP/INK_DUP bit 6 (14: plain layers only, 15: msg layers only): pre-mlp attn_res/mg/rms
      if (_r > 0u) barrier();
    // I. Pre-MLP Block Attention Residual + optional Message Gate + mlp_norm
    if (use_mid_plain) {
      AttnResRmsScoresArgs ars{H, E, M, n_blocks, b_stride, eps};
      [enc setComputePipelineState:p_attn_res_rms_scores];
      [enc setBuffer:b_blocks offset:0 atIndex:0];
      [enc setBuffer:b_h offset:0 atIndex:1];
      [enc setBuffer:ly.ar_mlp_q.b offset:ly.ar_mlp_q.off atIndex:2];
      [enc setBuffer:ly.mlp_norm.b offset:ly.mlp_norm.off atIndex:3];
      [enc setBuffer:b_frag_m offset:0 atIndex:4];
      [enc setBuffer:b_mn_f32 offset:0 atIndex:5];
      [enc setBuffer:ly.router_w.b offset:ly.router_w.off atIndex:6];
      [enc setBuffer:b_rscores offset:0 atIndex:7];
      [enc setBytes:&ars length:sizeof(ars) atIndex:8];
      [enc dispatchThreadgroups:MTLSizeMake(M, E / 8u, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      barrier();
      flush_prof("attn_res");
    } else if (fuse_h && !has_msgs) {
      for (const auto& lr : lanes) {
        auto& sl = (*lr.S)[log_idx];
        ConvResRmsArgs cra{H, lr.m, n_blocks, b_stride, lr.r0, ks, ly.is_boundary ? 1u : 0u, h_slot, eps};
        [enc setComputePipelineState:p_conv_res_rms];
        [enc setBuffer:b_o_out offset:0 atIndex:0];
        [enc setBuffer:ly.attn_sconv.b offset:ly.attn_sconv.off atIndex:1];
        [enc setBuffer:sl.attn_conv offset:0 atIndex:2];
        [enc setBuffer:b_h offset:0 atIndex:3];
        [enc setBuffer:b_blocks offset:0 atIndex:4];
        [enc setBuffer:sl.attn_hist offset:0 atIndex:5];
        [enc setBuffer:sl.attn_conv_next offset:0 atIndex:6];
        [enc setBuffer:ly.ar_mlp_q.b offset:ly.ar_mlp_q.off atIndex:7];
        [enc setBuffer:ly.mlp_norm.b offset:ly.mlp_norm.off atIndex:8];
        [enc setBuffer:b_frag_m offset:0 atIndex:9];
        [enc setBuffer:b_mn_f32 offset:0 atIndex:10];
        [enc setBytes:&cra length:sizeof(cra) atIndex:11];
        [enc dispatchThreadgroups:MTLSizeMake(lr.m, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        std::swap(sl.attn_conv, sl.attn_conv_next);
      }
      barrier();
      flush_prof("attn_res");
    } else if (!has_msgs) {
      AttnResRmsArgs mar{H, M, n_blocks, b_stride, 1u, eps};
      [enc setComputePipelineState:p_attn_res_rms];
      [enc setBuffer:b_blocks offset:0 atIndex:0];
      [enc setBuffer:b_h offset:0 atIndex:1];
      [enc setBuffer:ly.ar_mlp_q.b offset:ly.ar_mlp_q.off atIndex:2];
      [enc setBuffer:ly.mlp_norm.b offset:ly.mlp_norm.off atIndex:3];
      [enc setBuffer:ly.mlp_norm.b offset:ly.mlp_norm.off atIndex:4];
      [enc setBuffer:ly.mlp_norm.b offset:ly.mlp_norm.off atIndex:5];
      [enc setBuffer:b_frag_m offset:0 atIndex:6];
      [enc setBuffer:b_frag_m offset:0 atIndex:7];
      [enc setBuffer:b_frag_m offset:0 atIndex:8];
      [enc setBuffer:b_mn_f32 offset:0 atIndex:9];
      [enc setBytes:&mar length:sizeof(mar) atIndex:10];
      [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      barrier();
      static const uint32_t dup_mlp_ar = getenv("INK_DUP_MLPAR") ? uint32_t(atoi(getenv("INK_DUP_MLPAR"))) : 0u;
      for (uint32_t k = 0; k < dup_mlp_ar; k++) {  // screen only: repeat the (idempotent) dispatch
        [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();
      }
      flush_prof("attn_res");
    } else {
      const uint32_t cat_stride_b4 = (MP * 2u * H) / 4u;
      const size_t m_off = msg_off + 3u * size_t(M) * H * sizeof(float);
      if (use_mid_msg_down) {
        AttnResMgDownArgs armd{H, gate_rank, M, n_blocks, b_stride, eps};
        [enc setComputePipelineState:p_attn_res_mg_down];
        [enc setBuffer:b_blocks offset:0 atIndex:0];
        [enc setBuffer:b_h offset:0 atIndex:1];
        [enc setBuffer:ly.ar_mlp_q.b offset:ly.ar_mlp_q.off atIndex:2];
        [enc setBuffer:b_msgs offset:m_off atIndex:3];
        [enc setBuffer:b_mh offset:0 atIndex:4];
        [enc setBuffer:ly.ctrl_mlp.down.b offset:ly.ctrl_mlp.down.off atIndex:5];
        [enc setBuffer:b_mg_low offset:0 atIndex:6];
        [enc setBytes:&armd length:sizeof(armd) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(M, gate_rank / 8u, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();
        flush_prof("attn_res");

        MsgGateArgs mga{H, gate_rank, eps};
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
      } else {
      if (fuse_h) {
        for (const auto& lr : lanes) {
          auto& sl = (*lr.S)[log_idx];
          ConvResRmsArgs cra{H, lr.m, n_blocks, b_stride, lr.r0, ks, ly.is_boundary ? 1u : 0u, h_slot, eps};
          [enc setComputePipelineState:p_conv_res_mg];
          [enc setBuffer:b_o_out offset:0 atIndex:0];
          [enc setBuffer:ly.attn_sconv.b offset:ly.attn_sconv.off atIndex:1];
          [enc setBuffer:sl.attn_conv offset:0 atIndex:2];
          [enc setBuffer:b_h offset:0 atIndex:3];
          [enc setBuffer:b_blocks offset:0 atIndex:4];
          [enc setBuffer:sl.attn_hist offset:0 atIndex:5];
          [enc setBuffer:sl.attn_conv_next offset:0 atIndex:6];
          [enc setBuffer:ly.ar_mlp_q.b offset:ly.ar_mlp_q.off atIndex:7];
          [enc setBuffer:b_msgs offset:m_off atIndex:8];
          [enc setBuffer:b_mh offset:0 atIndex:9];
          [enc setBuffer:b_mg_cat offset:0 atIndex:10];
          [enc setBytes:&cra length:sizeof(cra) atIndex:11];
          [enc dispatchThreadgroups:MTLSizeMake(lr.m, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          std::swap(sl.attn_conv, sl.attn_conv_next);
        }
      } else {
      AttnResMgArgs arm{H, M, n_blocks, b_stride, 0u, cat_stride_b4, eps};
      [enc setComputePipelineState:p_attn_res_mg_prep];
      [enc setBuffer:b_blocks offset:0 atIndex:0];
      [enc setBuffer:b_h offset:0 atIndex:1];
      [enc setBuffer:ly.ar_mlp_q.b offset:ly.ar_mlp_q.off atIndex:2];
      [enc setBuffer:b_msgs offset:m_off atIndex:3];
      [enc setBuffer:b_mh offset:0 atIndex:4];
      [enc setBuffer:b_mg_cat offset:0 atIndex:5];
      [enc setBytes:&arm length:sizeof(arm) atIndex:6];
      [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      barrier();
      flush_prof("attn_res");

      if (pf) {
        MgMmaArgs mgm{H, gate_rank, M, 0u, MP * 2u * H, MP * gate_rank};
        [enc setComputePipelineState:p_mg_down_mma];
        [enc setBuffer:b_mg_cat offset:0 atIndex:0];
        [enc setBuffer:ly.ctrl_mlp.down.b offset:ly.ctrl_mlp.down.off atIndex:1];
        [enc setBuffer:ly.ctrl_mlp.down.b offset:ly.ctrl_mlp.down.off atIndex:2];
        [enc setBuffer:ly.ctrl_mlp.down.b offset:ly.ctrl_mlp.down.off atIndex:3];
        [enc setBuffer:b_mg_low offset:0 atIndex:4];
        [enc setBytes:&mgm length:sizeof(mgm) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((M + 15u) / 16u, gate_rank / 32u, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        barrier();

        [enc setComputePipelineState:p_mg_apply_mma];
        [enc setBuffer:b_mh offset:0 atIndex:0];
        [enc setBuffer:b_msgs offset:m_off atIndex:1];
        [enc setBuffer:b_mg_low offset:0 atIndex:2];
        [enc setBuffer:ly.ctrl_mlp.forget.b offset:ly.ctrl_mlp.forget.off atIndex:3];
        [enc setBuffer:ly.ctrl_mlp.gain.b offset:ly.ctrl_mlp.gain.off atIndex:4];
        [enc setBuffer:ly.ctrl_mlp.forget.b offset:ly.ctrl_mlp.forget.off atIndex:5];
        [enc setBuffer:ly.ctrl_mlp.gain.b offset:ly.ctrl_mlp.gain.off atIndex:6];
        [enc setBuffer:ly.ctrl_mlp.forget.b offset:ly.ctrl_mlp.forget.off atIndex:7];
        [enc setBuffer:ly.ctrl_mlp.gain.b offset:ly.ctrl_mlp.gain.off atIndex:8];
        [enc setBuffer:b_mh offset:0 atIndex:9];
        [enc setBytes:&mgm length:sizeof(mgm) atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake((M + 63u) / 64u, (H + 31u) / 32u, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      } else {
        MsgGateArgs mga{H, gate_rank, eps};
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
      }
      barrier();
      flush_prof("msg_gate");
      }

      if (use_rms_scores) {
        [enc setComputePipelineState:p_rms_scores];
        [enc setBuffer:b_mh offset:0 atIndex:0];
        [enc setBuffer:ly.mlp_norm.b offset:ly.mlp_norm.off atIndex:1];
        [enc setBuffer:b_frag_m offset:0 atIndex:2];
        [enc setBuffer:b_mn_f32 offset:0 atIndex:3];
        [enc setBuffer:ly.router_w.b offset:ly.router_w.off atIndex:4];
        [enc setBuffer:b_rscores offset:0 atIndex:5];
        [enc setBytes:&H length:4 atIndex:6];
        [enc setBytes:&E length:4 atIndex:7];
        [enc setBytes:&eps length:4 atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(M, E / 8u, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();
        flush_prof("rms");
      } else {
        enc_rms(b_mh, 0, ly.mlp_norm, b_frag_m, b_mn_f32, 1u);
        barrier();
        flush_prof("rms");
      }
    }

    }
    if (!use_mid_plain && !use_rms_scores)
    for (uint32_t _r = 0, _n = ((skipm & (1u << 7)) ? 0u : 1u + ((dupm >> 7) & 1u) * dupn); _r < _n; _r++) {  // INK_SKIP/INK_DUP bit 7: router scores
      if (_r > 0u) barrier();
    // J. MoE Router + Grouped NVFP4/MXFP8 Experts + Accum
    if (pf) {
      [enc setComputePipelineState:p_scores_mma];
      [enc setBuffer:b_mn_f32 offset:0 atIndex:0];
      [enc setBuffer:ly.router_w.b offset:ly.router_w.off atIndex:1];
      [enc setBuffer:b_rscores offset:0 atIndex:2];
      [enc setBytes:&M length:4 atIndex:3];
      [enc setBytes:&H length:4 atIndex:4];
      [enc setBytes:&E length:4 atIndex:5];
      [enc dispatchThreadgroups:MTLSizeMake((M + 15u) / 16u, E / 32u, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
      [enc setComputePipelineState:p_scores];
      [enc setBuffer:b_mn_f32 offset:0 atIndex:0];
      [enc setBuffer:ly.router_w.b offset:ly.router_w.off atIndex:1];
      [enc setBuffer:b_rscores offset:0 atIndex:2];
      [enc setBytes:&H length:4 atIndex:3];
      [enc setBytes:&E length:4 atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake(M, E / 8u, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    barrier();

    }
    RouteArgs ra{M, H, E, topk, route_scale};
    const bool is_fp4 = (ly.moe_gu.bits == 40);
    if (pf) {
      [enc setComputePipelineState:p_topk_pf];
      [enc setBuffer:b_rscores offset:0 atIndex:0];
      [enc setBuffer:ly.router_b.b offset:ly.router_b.off atIndex:1];
      [enc setBuffer:b_top_idx offset:0 atIndex:2];
      [enc setBuffer:b_top_w offset:0 atIndex:3];
      [enc setBytes:&ra length:sizeof(ra) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(E, 1, 1)];
      barrier();

      const uint32_t gx_gu = (ly.moe_gu.N / 32u + 1u) / 2u;
      const uint32_t gx_dn = (ly.moe_dn.N / 32u + 1u) / 2u;
      PlanArgs pa{M, E, topk, gx_gu, gx_dn};
      [enc setComputePipelineState:p_plan_pf];
      [enc setBuffer:b_top_idx offset:0 atIndex:0];
      [enc setBuffer:b_slots offset:0 atIndex:1];
      [enc setBuffer:b_nslots offset:0 atIndex:2];
      [enc setBuffer:b_roff offset:0 atIndex:3];
      [enc setBuffer:b_rcnt offset:0 atIndex:4];
      [enc setBuffer:b_rlist offset:0 atIndex:5];
      [enc setBuffer:b_rpos offset:0 atIndex:6];
      [enc setBuffer:b_wtile offset:0 atIndex:7];
      [enc setBuffer:b_ind_gu offset:0 atIndex:8];
      [enc setBuffer:b_ind_dn offset:0 atIndex:9];
      [enc setBytes:&pa length:sizeof(pa) atIndex:10];
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      barrier();
      flush_prof("moe_route");

      enc_mmpg(enc, is_fp4 ? p_mmpg_fp4_gu : p_mmpg_fp8_gu, ly.moe_gu, b_frag_m, b_dummy, b_frag_gu,
               b_slots, b_nslots, b_roff, b_rcnt, b_rlist, b_wtile, b_ind_gu, b_dummy, M, 0u, I_moe);
      barrier();
      flush_prof("moe_gu");

      enc_mmpg(enc, is_fp4 ? p_mmpg_fp4_dn : p_mmpg_fp8_dn, ly.moe_dn, b_frag_gu, b_moe_y, nil,
               b_slots, b_nslots, b_roff, b_rcnt, b_rlist, b_wtile, b_ind_dn, b_dummy, M, 2u, 0u);
      barrier();
      flush_prof("moe_dn");

      [enc setComputePipelineState:p_moe_accum_pf];
      [enc setBuffer:b_moe_y offset:0 atIndex:0];
      [enc setBuffer:b_rpos offset:0 atIndex:1];
      [enc setBuffer:b_top_w offset:0 atIndex:2];
      [enc setBuffer:b_moe_out offset:0 atIndex:3];
      [enc setBytes:&H length:4 atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake(M, ((H >> 2) + 255u) / 256u, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      barrier();
      flush_prof("moe_accum");
    } else {
    for (uint32_t _r = 0, _n = ((skipm & (1u << 7)) ? 0u : 1u + ((dupm >> 7) & 1u) * dupn); _r < _n; _r++) {  // INK_SKIP/INK_DUP bit 7: router top-k
      if (_r > 0u) barrier();
      if (route_fast) {
        [enc setComputePipelineState:p_route_fast];
        [enc setBuffer:b_rscores offset:0 atIndex:0];
        [enc setBuffer:ly.router_b.b offset:ly.router_b.off atIndex:1];
        [enc setBuffer:b_top_idx offset:0 atIndex:2];
        [enc setBuffer:b_top_w offset:0 atIndex:3];
        [enc setBuffer:b_slots offset:0 atIndex:4];
        [enc setBuffer:b_nslots offset:0 atIndex:5];
        [enc setBuffer:b_rmask offset:0 atIndex:6];
        [enc setBytes:&ra length:sizeof(ra) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(E, 1, 1)];
      } else if (M == 1u) {
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
      } else {
        [enc setComputePipelineState:p_topk_pf];
        [enc setBuffer:b_rscores offset:0 atIndex:0];
        [enc setBuffer:ly.router_b.b offset:ly.router_b.off atIndex:1];
        [enc setBuffer:b_top_idx offset:0 atIndex:2];
        [enc setBuffer:b_top_w offset:0 atIndex:3];
        [enc setBytes:&ra length:sizeof(ra) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(E, 1, 1)];
        barrier();

        [enc setComputePipelineState:p_pack_dec];
        [enc setBuffer:b_top_idx offset:0 atIndex:0];
        [enc setBuffer:b_slots offset:0 atIndex:1];
        [enc setBuffer:b_nslots offset:0 atIndex:2];
        [enc setBuffer:b_rmask offset:0 atIndex:3];
        [enc setBytes:&ra length:sizeof(ra) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(E, 1, 1)];
      }
      barrier();
      flush_prof("moe_route");

    }
    for (uint32_t _r = 0, _n = ((skipm & (1u << 8)) ? 0u : 1u + ((dupm >> 8) & 1u) * dupn); _r < _n; _r++) {  // INK_SKIP/INK_DUP bit 8: moe gate/up
      if (_r > 0u) barrier();
      if (moe_gemv && is_fp4) {
        enc_moe_gemv(enc, p_moe_gu_gemv, ly.moe_gu, b_mn_f32, b_moe_act, M, 0u, I_moe);
      } else {
      enc_mm2(enc, is_fp4 ? p_moe_fp4_gu : p_moe_fp8_gu, ly.moe_gu, b_frag_m, 0, b_dummy, 0, M,
              b_frag_gu, b_dummy, b_slots, b_nslots, b_rmask, true, I_moe, 0u, 0u);
      }
      barrier();
      flush_prof("moe_gu");
    }
      const uint32_t xstride_dn = (MAXM * I_moe * sizeof(uint16_t)) / 16u;
      const uint32_t ystride_dn = MAXM * H;
    for (uint32_t _r = 0, _n = ((skipm & (1u << 9)) ? 0u : 1u + ((dupm >> 9) & 1u) * dupn); _r < _n; _r++) {  // INK_SKIP/INK_DUP bit 9: moe down
      if (_r > 0u) barrier();
      if (moe_gemv && is_fp4) {
        enc_moe_gemv(enc, p_moe_dn_gemv, ly.moe_dn, b_moe_act, b_moe_y, M, ystride_dn, I_moe);
      } else {
      enc_mm2(enc, is_fp4 ? p_moe_fp4_dn : p_moe_fp8_dn, ly.moe_dn, b_frag_gu, 0, b_moe_y, 0, M,
              nil, b_dummy, b_slots, b_nslots, b_rmask, true, 0u, xstride_dn, ystride_dn);
      }
      barrier();
      flush_prof("moe_dn");

    }
    const bool use_mconv = !pf && (fuse || (fuse_mconv >= 2u && M <= 16u) || (fuse_mconv == 1u && M == 1u));
    if (!use_mconv && !(skipm & (1u << 10))) {  // INK_SKIP bit 10: moe accum (fused into section K when use_mconv)
      [enc setComputePipelineState:p_moe_accum];
      [enc setBuffer:b_moe_y offset:0 atIndex:0];
      [enc setBuffer:b_slots offset:0 atIndex:1];
      [enc setBuffer:b_nslots offset:0 atIndex:2];
      [enc setBuffer:b_top_idx offset:0 atIndex:3];
      [enc setBuffer:b_top_w offset:0 atIndex:4];
      [enc setBuffer:b_moe_out offset:0 atIndex:5];
      [enc setBytes:&H length:4 atIndex:6];
      [enc setBytes:&ystride_dn length:4 atIndex:7];
      [enc dispatchThreadgroups:MTLSizeMake(M, ((H >> 2) + 63u) / 64u, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      barrier();
      flush_prof("moe_accum");
    }
    }

    const bool use_mconv_k = !pf && (fuse || (fuse_mconv >= 2u && M <= 16u) || (fuse_mconv == 1u && M == 1u));
    for (uint32_t _r = 0, _n = ((skipm & (1u << 11)) ? 0u : 1u + ((dupm >> 11) & 1u) * dupn); _r < _n; _r++) {  // INK_SKIP/INK_DUP bit 11: post_conv mlp
      if (_r > 0u) barrier();
    // K. Post-MLP sconv + residual add -> b_h (bf16-rounded); decode with use_mconv: + moe_accum in the same phase
    if (use_mconv_k) {
      const uint32_t ystride_dn = MAXM * H;
      for (const auto& lr : lanes) {
        auto& sl = (*lr.S)[log_idx];
        AccumConvArgs aca{lr.m, H, ks, lr.r0, ystride_dn};
        [enc setComputePipelineState:p_moe_accum_conv];
        [enc setBuffer:b_moe_y offset:0 atIndex:0];
        [enc setBuffer:b_slots offset:0 atIndex:1];
        [enc setBuffer:b_nslots offset:0 atIndex:2];
        [enc setBuffer:b_top_idx offset:0 atIndex:3];
        [enc setBuffer:b_top_w offset:0 atIndex:4];
        [enc setBuffer:ly.mlp_sconv.b offset:ly.mlp_sconv.off atIndex:5];
        [enc setBuffer:sl.mlp_conv offset:0 atIndex:6];
        [enc setBuffer:b_h offset:0 atIndex:7];
        [enc setBuffer:sl.mlp_hist offset:0 atIndex:8];
        [enc setBuffer:sl.mlp_conv_next offset:0 atIndex:9];
        [enc setBytes:&aca length:sizeof(aca) atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake(lr.m, ((H >> 2) + 63u) / 64u, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        std::swap(sl.mlp_conv, sl.mlp_conv_next);
      }
    } else
    for (const auto& lr : lanes) {
      auto& sl = (*lr.S)[log_idx];
      const size_t r0 = lr.r0;
      PostConvArgs mpca{lr.m, H, ks, 0u, 1u, 0u, b_stride};
      [enc setComputePipelineState:p_post_conv];
      [enc setBuffer:b_moe_out offset:r0 * H * sizeof(float) atIndex:0];
      [enc setBuffer:ly.mlp_sconv.b offset:ly.mlp_sconv.off atIndex:1];
      [enc setBuffer:sl.mlp_conv offset:0 atIndex:2];
      [enc setBuffer:b_h offset:r0 * H * sizeof(float) atIndex:3];
      [enc setBuffer:b_blocks offset:r0 * H * sizeof(float) atIndex:4];
      [enc setBuffer:sl.mlp_hist offset:0 atIndex:5];
      [enc setBytes:&mpca length:sizeof(mpca) atIndex:6];
      [enc setBuffer:sl.mlp_conv_next offset:0 atIndex:7];
      [enc dispatchThreadgroups:MTLSizeMake(lr.m, (H + 255u) / 256u, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      std::swap(sl.mlp_conv, sl.mlp_conv_next);
    }
    barrier();
    flush_prof("post_conv");
    }
  };

  uint32_t n_blocks = 0;
  uint32_t log_idx = 0;

  // 2. Prelude
  for (uint32_t l = 0; l < core_start; l++) run_layer(log_idx++, l, n_blocks, false);

  const uint32_t n_pre_blocks = n_blocks;
  copy_f32(b_h, 0, b_pre_h, 0, size_t(M) * H * sizeof(float));
  if (n_pre_blocks > 0) copy_f32(b_blocks, 0, b_pre_blocks, 0, size_t(n_pre_blocks) * b_stride * sizeof(float));
  barrier();

  early_commit();
  // 3. Pass 0 over core
  for (uint32_t l = core_start; l < core_end; l++) run_layer(log_idx++, l, n_blocks, false);
  early_commit();

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
    if (!pf && batch_msg) {
      // Batch all n_core * 4 loop messages in one dispatch, concurrently with the restore copies above!
      AttnResArgs mara{H, M, n_prev_bs, b_stride, eps};
      [enc setComputePipelineState:p_attn_res];
      [enc setBuffer:b_prev_blocks offset:0 atIndex:0];
      [enc setBuffer:b_prev_h offset:0 atIndex:1];
      [enc setBuffer:loop_msg_q.b offset:loop_msg_q.off atIndex:2];
      [enc setBuffer:b_msgs offset:0 atIndex:3];
      [enc setBytes:&mara length:sizeof(mara) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake(M, n_core * 4u, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    barrier();
    n_blocks = n_pre_blocks;

    for (uint32_t li = 0; li < n_core; li++) {
    if (pf || !batch_msg) {
    for (uint32_t _r = 0, _n = ((skipm & (1u << 13)) ? 0u : 1u + ((dupm >> 13) & 1u) * dupn); _r < _n; _r++) {  // INK_SKIP/INK_DUP bit 13: loop attn_res
      if (_r > 0u) barrier();
      AttnResArgs mara{H, M, n_prev_bs, b_stride, eps};
      [enc setComputePipelineState:p_attn_res];
      [enc setBuffer:b_prev_blocks offset:0 atIndex:0];
      [enc setBuffer:b_prev_h offset:0 atIndex:1];
      [enc setBuffer:loop_msg_q.b offset:(loop_msg_q.off + size_t(li) * 4u * H * sizeof(float)) atIndex:2];
      [enc setBuffer:b_msgs offset:0 atIndex:3];
      [enc setBytes:&mara length:sizeof(mara) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake(M, 4, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      barrier();
    }
    }
      const size_t msg_off = (!pf && batch_msg) ? (size_t(li) * 4u * size_t(M) * H * sizeof(float)) : 0u;
      run_layer(log_idx++, core_start + li, n_blocks, true, msg_off);
    }
    early_commit();
  }

  // 5. Coda
  for (uint32_t l = core_end; l < nL; l++) run_layer(log_idx++, l, n_blocks, false);

  // 6. Final AttnRes + Norm (+ capture tap 3 = final normalized hidden) + lm_head + softcap + argmax
  const uint32_t wf32_final = has_mtp ? 1u : 0u;
  if (pf || !batch_msg) {
    AttnResArgs fara{H, M, n_blocks, b_stride, eps};
    [enc setComputePipelineState:p_attn_res];
    [enc setBuffer:b_blocks offset:0 atIndex:0];
    [enc setBuffer:b_h offset:0 atIndex:1];
    [enc setBuffer:ar_out_q.b offset:ar_out_q.off atIndex:2];
    [enc setBuffer:b_mh offset:0 atIndex:3];
    [enc setBytes:&fara length:sizeof(fara) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();
  }

  if (pf) {
    if (has_mtp) {
      [enc setComputePipelineState:p_rms_frag];
      [enc setBuffer:b_mh offset:0 atIndex:0];
      [enc setBuffer:norm.b offset:norm.off atIndex:1];
      [enc setBuffer:b_frag_m offset:0 atIndex:2];
      [enc setBuffer:b_taps offset:(3u * size_t(MP) * H * sizeof(float)) atIndex:3];
      [enc setBytes:&H length:4 atIndex:4];
      [enc setBytes:&eps length:4 atIndex:5];
      [enc setBytes:&wf32_final length:4 atIndex:6];
      [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      barrier();
    }
    const uint32_t zero_u = 0u;
    [enc setComputePipelineState:p_rms_frag];
    [enc setBuffer:b_mh offset:(size_t(M - 1u) * H * sizeof(float)) atIndex:0];
    [enc setBuffer:norm.b offset:norm.off atIndex:1];
    [enc setBuffer:b_frag_q offset:0 atIndex:2];
    [enc setBuffer:b_dummy offset:0 atIndex:3];
    [enc setBytes:&H length:4 atIndex:4];
    [enc setBytes:&eps length:4 atIndex:5];
    [enc setBytes:&zero_u length:4 atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();

    enc_mm2(enc, p_mm_w8, lm_head, b_frag_q, 0, b_logits, 0, 1u, nil, b_dummy, nil, nil, nil, false);
    barrier();
    flush_prof("lm_head");

    [enc setComputePipelineState:p_softcap_argmax_p1];
    [enc setBuffer:b_logits offset:0 atIndex:0];
    [enc setBuffer:b_argmax_v offset:0 atIndex:1];
    [enc setBuffer:b_argmax_i offset:0 atIndex:2];
    [enc setBytes:&vocab_valid length:4 atIndex:3];
    [enc setBytes:&vocab_pad length:4 atIndex:4];
    [enc setBytes:&logit_softcap length:4 atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(1, 64, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();
    [enc setComputePipelineState:p_softcap_argmax_p2];
    [enc setBuffer:b_argmax_v offset:0 atIndex:0];
    [enc setBuffer:b_argmax_i offset:0 atIndex:1];
    [enc setBuffer:b_tok_out offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    flush_prof("argmax");
    enc_sample(1u, pos + M - 1u);
  } else {
    if (batch_msg) {
      AttnResRmsArgs far{H, M, n_blocks, b_stride, wf32_final, eps};
      [enc setComputePipelineState:p_attn_res_rms];
      [enc setBuffer:b_blocks offset:0 atIndex:0];
      [enc setBuffer:b_h offset:0 atIndex:1];
      [enc setBuffer:ar_out_q.b offset:ar_out_q.off atIndex:2];
      [enc setBuffer:norm.b offset:norm.off atIndex:3];
      [enc setBuffer:norm.b offset:norm.off atIndex:4];
      [enc setBuffer:norm.b offset:norm.off atIndex:5];
      [enc setBuffer:b_frag_m offset:0 atIndex:6];
      [enc setBuffer:b_frag_m offset:0 atIndex:7];
      [enc setBuffer:b_frag_m offset:0 atIndex:8];
      [enc setBuffer:(has_mtp ? b_taps : b_dummy) offset:(has_mtp ? (3u * size_t(MP) * H * sizeof(float)) : 0) atIndex:9];
      [enc setBytes:&far length:sizeof(far) atIndex:10];
      [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
      [enc setComputePipelineState:p_rms_frag];
      [enc setBuffer:b_mh offset:0 atIndex:0];
      [enc setBuffer:norm.b offset:norm.off atIndex:1];
      [enc setBuffer:b_frag_m offset:0 atIndex:2];
      [enc setBuffer:(has_mtp ? b_taps : b_dummy) offset:(has_mtp ? (3u * size_t(MP) * H * sizeof(float)) : 0) atIndex:3];
      [enc setBytes:&H length:4 atIndex:4];
      [enc setBytes:&eps length:4 atIndex:5];
      [enc setBytes:&wf32_final length:4 atIndex:6];
      [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    barrier();

    for (uint32_t _r = 0, _n = ((skipm & (1u << 12)) ? 0u : 1u + ((dupm >> 12) & 1u) * dupn); _r < _n; _r++) {  // INK_SKIP/INK_DUP bit 12: lm_head + argmax
      if (_r > 0u) barrier();
    enc_mm2(enc, p_mm_w8, lm_head, b_frag_m, 0, b_logits, 0, M, nil, b_dummy, nil, nil, nil, false);
    barrier();
    flush_prof("lm_head");

    [enc setComputePipelineState:p_softcap_argmax_p1];
    [enc setBuffer:b_logits offset:0 atIndex:0];
    [enc setBuffer:b_argmax_v offset:0 atIndex:1];
    [enc setBuffer:b_argmax_i offset:0 atIndex:2];
    [enc setBytes:&vocab_valid length:4 atIndex:3];
    [enc setBytes:&vocab_pad length:4 atIndex:4];
    [enc setBytes:&logit_softcap length:4 atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(M, 64, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();
    [enc setComputePipelineState:p_softcap_argmax_p2];
    [enc setBuffer:b_argmax_v offset:0 atIndex:0];
    [enc setBuffer:b_argmax_i offset:0 atIndex:1];
    [enc setBuffer:b_tok_out offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    flush_prof("argmax");
    }
    enc_sample(M, 0u);
  }

  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  for (id<MTLCommandBuffer> d : done_cbs) {
    [d waitUntilCompleted];
    if (d.error) mtl::die("GPU step error", d.error);
  }
  if (cb.error) mtl::die("GPU step error", cb.error);
  if (gpu_ms) *gpu_ms = (cb.GPUEndTime - (done_cbs.empty() ? cb : done_cbs.front()).GPUStartTime) * 1000.0;

  for (const auto& lr : lanes) *lr.pos += lr.m;
  std::vector<uint32_t> out(M, 0u);
  if (pf) {
    out[M - 1u] = *(const uint32_t*)[b_tok_out contents];
    last_conf.assign(1u, *(const float*)[b_conf_out contents]);
    if (last_logits) {
      last_logits->resize(vocab_valid);
      memcpy(last_logits->data(), [b_logits contents], vocab_valid * sizeof(float));
    }
  } else {
    memcpy(out.data(), [b_tok_out contents], M * sizeof(uint32_t));
    last_conf.resize(M);
    memcpy(last_conf.data(), [b_conf_out contents], M * sizeof(float));
    if (last_logits) {
      last_logits->resize(vocab_valid);
      const float* lp = (const float*)[b_logits contents] + size_t(M - 1u) * vocab_pad;
      memcpy(last_logits->data(), lp, vocab_valid * sizeof(float));
    }
    if (all_logits) {
      all_logits->resize(size_t(M) * vocab_valid);
      for (uint32_t r = 0; r < M; r++)
        memcpy(all_logits->data() + size_t(r) * vocab_valid, (const float*)[b_logits contents] + size_t(r) * vocab_pad,
               vocab_valid * sizeof(float));
    }
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
    MtpMixArgs ma{H, MP, 1u, 0u, 0u, eps};
    [enc setComputePipelineState:p_mtp_mix];
    [enc setBuffer:b_taps offset:size_t(tap_row0) * H * sizeof(float) atIndex:0];
    [enc setBuffer:mtp_fuse_alpha.b offset:(mtp_fuse_alpha.off + size_t(l) * 4u * sizeof(float)) atIndex:1];
    [enc setBuffer:ml.dflare_norm.b offset:ml.dflare_norm.off atIndex:2];
    [enc setBuffer:b_mtp_frag offset:0 atIndex:3];
    [enc setBuffer:b_dummy offset:0 atIndex:4];
    [enc setBytes:&ma length:sizeof(ma) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(n_commit, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();

    if (n_commit > MAXM) {
      enc_mmp(enc, p_mmp_w8, ml.kv_ctx, b_mtp_frag, 0, b_mtp_kv_ctx, 0, n_commit, b_dummy);
    } else {
      enc_mm2(enc, p_mm_w8, ml.kv_ctx, b_mtp_frag, 0, b_mtp_kv_ctx, 0, n_commit, nil, b_dummy, nil, nil, nil, false, 0, 0, 0, big_gpu);
    }
    barrier();

    const uint32_t mfpos = (!ml.is_local && fork_src >= 0) ? fork_pos : 0u;
    MtpKvArgs ka{nkv, hd, mtp_ctx_pos, ml.cap, mfpos, eps, mtp_rope_theta};
    [enc setComputePipelineState:p_mtp_kv_ctx];
    [enc setBuffer:b_mtp_kv_ctx offset:0 atIndex:0];
    [enc setBuffer:ml.k_norm_ctx.b offset:ml.k_norm_ctx.off atIndex:1];
    [enc setBuffer:ml.k_cache offset:0 atIndex:2];
    [enc setBuffer:ml.v_cache offset:0 atIndex:3];
    [enc setBytes:&ka length:sizeof(ka) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(n_commit, nkv, 1) threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
    barrier();
  }

  // 2. Compute f_rnn ONLY for the last committed token (row_off = n_commit - 1 -> row 0, zero CPU memcpy!)
  MtpMixArgs rma{H, MP, 0u, 1u, n_commit - 1u, eps};
  [enc setComputePipelineState:p_mtp_mix];
  [enc setBuffer:b_taps offset:size_t(tap_row0) * H * sizeof(float) atIndex:0];
  [enc setBuffer:mtp_fuse_alpha.b offset:(mtp_fuse_alpha.off + 3u * 4u * sizeof(float)) atIndex:1];
  [enc setBuffer:mtp_mh_dflare_norm.b offset:mtp_mh_dflare_norm.off atIndex:2];
  [enc setBuffer:b_dummy offset:0 atIndex:3];
  [enc setBuffer:b_mtp_f_rnn offset:0 atIndex:4];
  [enc setBytes:&rma length:sizeof(rma) atIndex:5];
  [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.error) mtl::die("GPU mtp_commit error", cb.error);
  if (gpu_ms) *gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
  mtp_ctx_pos += n_commit;
}

std::vector<uint32_t> Model::draft(uint32_t anchor_tok, uint32_t max_drafts,
                                   std::vector<float>* draft_probs, double* gpu_ms,
                                   uint32_t n_commit) {
  if (!has_mtp) return {};
  const uint32_t B = mtp_block;  // 8
  const uint32_t gamma = std::max(1u, std::min(max_drafts, B - 1u));
  *(uint32_t*)[b_mtp_prev_tok contents] = anchor_tok;
  memset([b_mtp_rnn_state contents], 0, size_t(mtp_rank) * sizeof(float));

  id<MTLCommandBuffer> cb = [C.q commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
  auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };

  // Optional fused mtp_commit(n_commit) inside the same command buffer (eliminates 1 GPU sync round-trip!)
  if (n_commit > 0u) {
    for (uint32_t l = 0; l < mtp_nL; l++) {
      auto& ml = MtpL[l];
      MtpMixArgs ma{H, MP, 1u, 0u, 0u, eps};
      [enc setComputePipelineState:p_mtp_mix];
      [enc setBuffer:b_taps offset:size_t(tap_row0) * H * sizeof(float) atIndex:0];
      [enc setBuffer:mtp_fuse_alpha.b offset:(mtp_fuse_alpha.off + size_t(l) * 4u * sizeof(float)) atIndex:1];
      [enc setBuffer:ml.dflare_norm.b offset:ml.dflare_norm.off atIndex:2];
      [enc setBuffer:b_mtp_frag offset:0 atIndex:3];
      [enc setBuffer:b_dummy offset:0 atIndex:4];
      [enc setBytes:&ma length:sizeof(ma) atIndex:5];
      [enc dispatchThreadgroups:MTLSizeMake(n_commit, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      barrier();

      if (n_commit > MAXM) {
        enc_mmp(enc, p_mmp_w8, ml.kv_ctx, b_mtp_frag, 0, b_mtp_kv_ctx, 0, n_commit, b_dummy);
      } else {
        enc_mm2(enc, p_mm_w8, ml.kv_ctx, b_mtp_frag, 0, b_mtp_kv_ctx, 0, n_commit, nil, b_dummy, nil, nil, nil, false, 0, 0, 0, big_gpu);
      }
      barrier();

      const uint32_t mfpos = (!ml.is_local && fork_src >= 0) ? fork_pos : 0u;
    MtpKvArgs ka{nkv, hd, mtp_ctx_pos, ml.cap, mfpos, eps, mtp_rope_theta};
      [enc setComputePipelineState:p_mtp_kv_ctx];
      [enc setBuffer:b_mtp_kv_ctx offset:0 atIndex:0];
      [enc setBuffer:ml.k_norm_ctx.b offset:ml.k_norm_ctx.off atIndex:1];
      [enc setBuffer:ml.k_cache offset:0 atIndex:2];
      [enc setBuffer:ml.v_cache offset:0 atIndex:3];
      [enc setBytes:&ka length:sizeof(ka) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake(n_commit, nkv, 1) threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
      barrier();
    }
    MtpMixArgs rma{H, MP, 0u, 1u, n_commit - 1u, eps};
    [enc setComputePipelineState:p_mtp_mix];
    [enc setBuffer:b_taps offset:size_t(tap_row0) * H * sizeof(float) atIndex:0];
    [enc setBuffer:mtp_fuse_alpha.b offset:(mtp_fuse_alpha.off + 3u * 4u * sizeof(float)) atIndex:1];
    [enc setBuffer:mtp_mh_dflare_norm.b offset:mtp_mh_dflare_norm.off atIndex:2];
    [enc setBuffer:b_dummy offset:0 atIndex:3];
    [enc setBuffer:b_mtp_f_rnn offset:0 atIndex:4];
    [enc setBytes:&rma length:sizeof(rma) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    mtp_ctx_pos += n_commit;
  }

  // 1. Build 8-token draft input block (runs concurrently with f_rnn above!)
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

  // Initial attn_norm for layer 0 -> b_mtp_frag
  [enc setComputePipelineState:p_rms_frag];
  [enc setBuffer:b_mtp_h offset:0 atIndex:0];
  [enc setBuffer:MtpL[0].attn_norm.b offset:MtpL[0].attn_norm.off atIndex:1];
  [enc setBuffer:b_mtp_frag offset:0 atIndex:2];
  [enc setBuffer:b_dummy offset:0 atIndex:3];
  [enc setBytes:&H length:4 atIndex:4];
  [enc setBytes:&eps length:4 atIndex:5];
  [enc setBytes:&zero_u length:4 atIndex:6];
  [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  barrier();

  // 2. Run 3 dense DFlare MTP layers (fused add + next RMSNorm!)
  for (uint32_t l = 0; l < mtp_nL; l++) {
    auto& ml = MtpL[l];
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

    const uint32_t mfpos = (!ml.is_local && fork_src >= 0) ? fork_pos : 0u;
    id<MTLBuffer> pk = (mfpos > 0u) ? (fork_src == active_lane ? ml.k_cache : lanes_[fork_src].mtp_k[l]) : b_dummy;
    id<MTLBuffer> pv = (mfpos > 0u) ? (fork_src == active_lane ? ml.v_cache : lanes_[fork_src].mtp_v[l]) : b_dummy;
    MtpAttnArgs aa{B, nh, nkv, hd, mtp_ctx_pos, ml.cap, window, ml.is_local ? 1u : 0u, mfpos, 0u};
    [enc setComputePipelineState:p_mtp_attn];
    [enc setBuffer:b_mtp_Q offset:0 atIndex:0];
    [enc setBuffer:ml.k_cache offset:0 atIndex:1];
    [enc setBuffer:ml.v_cache offset:0 atIndex:2];
    [enc setBuffer:b_mtp_K_blk offset:0 atIndex:3];
    [enc setBuffer:b_mtp_V_blk offset:0 atIndex:4];
    [enc setBuffer:b_mtp_o_frag offset:0 atIndex:5];
    [enc setBytes:&aa length:sizeof(aa) atIndex:6];
    [enc setBuffer:pk offset:0 atIndex:7];
    [enc setBuffer:pv offset:0 atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake(B, nkv, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    barrier();

    enc_mm2(enc, p_mm_w8, ml.o, b_mtp_o_frag, 0, b_mtp_o_out, 0, B, nil, b_dummy, nil, nil, nil, false);
    barrier();

    // Fused residual add + mlp_norm -> b_mtp_frag
    [enc setComputePipelineState:p_add_rms_frag];
    [enc setBuffer:b_mtp_h offset:0 atIndex:0];
    [enc setBuffer:b_mtp_o_out offset:0 atIndex:1];
    [enc setBuffer:ml.mlp_norm.b offset:ml.mlp_norm.off atIndex:2];
    [enc setBuffer:b_mtp_frag offset:0 atIndex:3];
    [enc setBuffer:b_dummy offset:0 atIndex:4];
    [enc setBytes:&H length:4 atIndex:5];
    [enc setBytes:&eps length:4 atIndex:6];
    [enc setBytes:&zero_u length:4 atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();

    enc_mm2(enc, p_mm_w6_gu, ml.mlp_gu, b_mtp_frag, 0, b_dummy, 0, B, b_mtp_gu_frag, b_dummy, nil, nil, nil, false, mtp_I);
    barrier();

    enc_mm2(enc, p_mm_w6_dn, ml.mlp_dn, b_mtp_gu_frag, 0, b_mtp_dn_out, 0, B, nil, b_dummy, nil, nil, nil, false);
    barrier();

    // Fused residual add + next layer attn_norm (or final mtp_norm + b_mtp_base_h)
    const bool last_l = (l + 1u == mtp_nL);
    const Raw& next_nw = last_l ? mtp_norm : MtpL[l + 1u].attn_norm;
    const uint32_t wf32 = last_l ? one_u : zero_u;
    [enc setComputePipelineState:p_add_rms_frag];
    [enc setBuffer:b_mtp_h offset:0 atIndex:0];
    [enc setBuffer:b_mtp_dn_out offset:0 atIndex:1];
    [enc setBuffer:next_nw.b offset:next_nw.off atIndex:2];
    [enc setBuffer:b_mtp_frag offset:0 atIndex:3];
    [enc setBuffer:(last_l ? b_mtp_base_h : b_dummy) offset:0 atIndex:4];
    [enc setBytes:&H length:4 atIndex:5];
    [enc setBytes:&eps length:4 atIndex:6];
    [enc setBytes:&wf32 length:4 atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();
  }

  // 3. Base logits for all B rows || Precompute static Markov RNN projections concurrently!
  enc_mm2(enc, p_mm_w8, lm_head, b_mtp_frag, 0, b_mtp_base_logits, 0, B, nil, b_dummy, nil, nil, nil, false);
  {
    RnnArgs sra{H, mtp_rank, gamma};
    [enc setComputePipelineState:p_mtp_rnn_static];
    [enc setBuffer:b_mtp_base_h offset:0 atIndex:0];
    [enc setBuffer:b_mtp_f_rnn offset:0 atIndex:1];
    [enc setBuffer:mtp_mh_rnn_w.b offset:mtp_mh_rnn_w.off atIndex:2];
    [enc setBuffer:mtp_mh_rnn_b.b offset:mtp_mh_rnn_b.off atIndex:3];
    [enc setBuffer:b_mtp_rnn_static offset:0 atIndex:4];
    [enc setBytes:&sra length:sizeof(sra) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((3u * mtp_rank + 3u) / 4u, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  }
  barrier();

  // 4. Sequential Markov RNN head for gamma draft tokens (10x smaller rnn_dyn + 64-TG parallel argmax!)
  for (uint32_t k = 0; k < gamma; k++) {
    RnnArgs ra{H, mtp_rank, k};
    [enc setComputePipelineState:p_mtp_rnn_dyn];
    [enc setBuffer:b_mtp_rnn_state offset:0 atIndex:0];
    [enc setBuffer:mtp_mh_markov_w1.b offset:mtp_mh_markov_w1.off atIndex:1];
    [enc setBuffer:b_mtp_prev_tok offset:0 atIndex:2];
    [enc setBuffer:b_mtp_rnn_static offset:0 atIndex:3];
    [enc setBuffer:mtp_mh_rnn_w.b offset:mtp_mh_rnn_w.off atIndex:4];
    [enc setBuffer:b_mtp_rnn_proj offset:0 atIndex:5];
    [enc setBytes:&ra length:sizeof(ra) atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake((3u * mtp_rank + 3u) / 4u, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    barrier();

    [enc setComputePipelineState:p_mtp_rnn_gates];
    [enc setBuffer:b_mtp_rnn_proj offset:0 atIndex:0];
    [enc setBuffer:b_mtp_rnn_state offset:0 atIndex:1];
    [enc setBuffer:b_mtp_lat_frag offset:0 atIndex:2];
    [enc setBytes:&mtp_rank length:4 atIndex:3];
    [enc setBytes:&zero_u length:4 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();

    // Folded Markov LM head: [200064, 256] W8 x [1, 256] -> b_mtp_bias
    enc_mm2(enc, p_mm_w8, mtp_mh_fold_w, b_mtp_lat_frag, 0, b_mtp_bias, 0, 1u, nil, b_dummy, nil, nil, nil, false);
    barrier();

    MtpPickArgs pa{vocab_valid, vocab_pad, k, logit_softcap, samp.temp > 0.0f ? 1.0f / samp.temp : -1.0f,
                   uint32_t(samp.seed), uint32_t(samp.seed >> 32), pos};
    [enc setComputePipelineState:p_mtp_add_argmax_p1];
    [enc setBuffer:b_mtp_base_logits offset:(size_t(vocab_pad) * sizeof(float)) atIndex:0];  // row 1..7!
    [enc setBuffer:b_mtp_bias offset:0 atIndex:1];
    [enc setBuffer:mtp_mh_fold_b.b offset:mtp_mh_fold_b.off atIndex:2];
    [enc setBuffer:b_argmax_v offset:0 atIndex:3];
    [enc setBuffer:b_argmax_i offset:0 atIndex:4];
    [enc setBuffer:b_mtp_part_l offset:0 atIndex:5];
    [enc setBytes:&pa length:sizeof(pa) atIndex:6];
    [enc setBuffer:b_mtp_part_x offset:0 atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake(64, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();

    [enc setComputePipelineState:p_mtp_add_argmax_p2];
    [enc setBuffer:b_argmax_v offset:0 atIndex:0];
    [enc setBuffer:b_argmax_i offset:0 atIndex:1];
    [enc setBuffer:b_mtp_part_l offset:0 atIndex:2];
    [enc setBuffer:b_mtp_prev_tok offset:0 atIndex:3];
    [enc setBuffer:b_mtp_draft_out offset:0 atIndex:4];
    [enc setBuffer:b_mtp_draft_probs offset:0 atIndex:5];
    [enc setBytes:&pa length:sizeof(pa) atIndex:6];
    [enc setBuffer:b_mtp_part_x offset:0 atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    barrier();
  }

  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.error) mtl::die("GPU draft error", cb.error);
  if (gpu_ms) *gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;

  std::vector<uint32_t> out(gamma);
  memcpy(out.data(), [b_mtp_draft_out contents], gamma * sizeof(uint32_t));
  if (draft_probs) {
    draft_probs->resize(gamma);
    memcpy(draft_probs->data(), [b_mtp_draft_probs contents], gamma * sizeof(float));
  }
  return out;
}

void Model::mtp_commit_lanes(const std::vector<MtpCommitReq>& reqs, uint32_t total_tap_rows, double* gpu_ms) {
  if (!has_mtp || reqs.empty() || total_tap_rows == 0u) return;
  if (reqs.size() == 1u) {
    use_lane(reqs[0].lane);
    tap_row0 = reqs[0].tap_row0;
    mtp_commit(reqs[0].n_commit, gpu_ms);
    tap_row0 = 0;
    return;
  }
  if (total_tap_rows > MAXM) mtl::die("mtp_commit_lanes: total_tap_rows > MAXM");
  id<MTLCommandBuffer> cb = [C.q commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
  auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };

  // 1. Project & cache context KV for all lanes in ONE kv_ctx matmul per MTP layer!
  for (uint32_t l = 0; l < mtp_nL; l++) {
    auto& ml = MtpL[l];
    MtpMixArgs ma{H, MP, 1u, 0u, 0u, eps};
    [enc setComputePipelineState:p_mtp_mix];
    [enc setBuffer:b_taps offset:0 atIndex:0];
    [enc setBuffer:mtp_fuse_alpha.b offset:(mtp_fuse_alpha.off + size_t(l) * 4u * sizeof(float)) atIndex:1];
    [enc setBuffer:ml.dflare_norm.b offset:ml.dflare_norm.off atIndex:2];
    [enc setBuffer:b_mtp_frag offset:0 atIndex:3];
    [enc setBuffer:b_dummy offset:0 atIndex:4];
    [enc setBytes:&ma length:sizeof(ma) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(total_tap_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();

    enc_mm2(enc, p_mm_w8, ml.kv_ctx, b_mtp_frag, 0, b_mtp_kv_ctx, 0, total_tap_rows, nil, b_dummy, nil, nil, nil, false, 0, 0, 0, big_gpu);
    barrier();

    for (const auto& rq : reqs) {
      if (rq.n_commit == 0u) continue;
      const int k = rq.lane;
      id<MTLBuffer> kc = (k == active_lane) ? ml.k_cache : lanes_[k].mtp_k[l];
      id<MTLBuffer> vc = (k == active_lane) ? ml.v_cache : lanes_[k].mtp_v[l];
      const uint32_t mcap = (k == active_lane) ? ml.cap : lanes_[k].mtp_cap[l];
      const uint32_t mpos = (k == active_lane) ? mtp_ctx_pos : lanes_[k].mtp_ctx_pos;
      const int fsrc = (k == active_lane) ? fork_src : lanes_[k].fork_src;
      const uint32_t fpos = (k == active_lane) ? fork_pos : lanes_[k].fork_pos;
      const uint32_t mfpos = (!ml.is_local && fsrc >= 0) ? fpos : 0u;
      MtpKvArgs ka{nkv, hd, mpos, mcap, mfpos, eps, mtp_rope_theta};
      [enc setComputePipelineState:p_mtp_kv_ctx];
      [enc setBuffer:b_mtp_kv_ctx offset:size_t(rq.tap_row0) * 2u * nkv * hd * sizeof(float) atIndex:0];
      [enc setBuffer:ml.k_norm_ctx.b offset:ml.k_norm_ctx.off atIndex:1];
      [enc setBuffer:kc offset:0 atIndex:2];
      [enc setBuffer:vc offset:0 atIndex:3];
      [enc setBytes:&ka length:sizeof(ka) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake(rq.n_commit, nkv, 1) threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
    }
    barrier();
  }

  // 2. Compute f_rnn for each lane's last committed token concurrently
  for (const auto& rq : reqs) {
    if (rq.n_commit == 0u) continue;
    const int k = rq.lane;
    id<MTLBuffer> fr = (k == active_lane) ? b_mtp_f_rnn : lanes_[k].mtp_f_rnn;
    MtpMixArgs rma{H, MP, 0u, 1u, rq.tap_row0 + rq.n_commit - 1u, eps};
    [enc setComputePipelineState:p_mtp_mix];
    [enc setBuffer:b_taps offset:0 atIndex:0];
    [enc setBuffer:mtp_fuse_alpha.b offset:(mtp_fuse_alpha.off + 3u * 4u * sizeof(float)) atIndex:1];
    [enc setBuffer:mtp_mh_dflare_norm.b offset:mtp_mh_dflare_norm.off atIndex:2];
    [enc setBuffer:b_dummy offset:0 atIndex:3];
    [enc setBuffer:fr offset:0 atIndex:4];
    [enc setBytes:&rma length:sizeof(rma) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }

  [enc endEncoding];
  [cb commit];
  static const bool sync_commit = getenv("INK_SYNC_COMMIT") && atoi(getenv("INK_SYNC_COMMIT")) != 0;
  if (gpu_ms || sync_commit) {
    [cb waitUntilCompleted];
    if (cb.error) mtl::die("GPU mtp_commit_lanes error", cb.error);
    if (gpu_ms) *gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
  }
  for (const auto& rq : reqs) {
    if (rq.lane == active_lane) mtp_ctx_pos += rq.n_commit;
    else lanes_[rq.lane].mtp_ctx_pos += rq.n_commit;
  }
}

void Model::draft_lanes(const std::vector<MtpDraftReq>& reqs, uint32_t max_drafts,
                        std::vector<std::vector<uint32_t>>& out_drafts,
                        std::vector<std::vector<float>>* out_probs, double* gpu_ms) {
  out_drafts.assign(reqs.size(), {});
  if (out_probs) out_probs->assign(reqs.size(), {});
  if (!has_mtp || reqs.empty()) return;
  const uint32_t B = mtp_block;  // 8
  const uint32_t gamma = std::max(1u, std::min(max_drafts, B - 1u));
  if (reqs.size() == 1u) {
    use_lane(reqs[0].lane);
    std::vector<float> dp;
    out_drafts[0] = draft(reqs[0].anchor_tok, gamma, out_probs ? &dp : nullptr, gpu_ms, 0u);
    if (out_probs) (*out_probs)[0] = std::move(dp);
    return;
  }

  // Max lanes per batch so L <= MTP_MAXL and L * gamma <= MAXM (fits b_mtp_base_logits[MAXM])
  const uint32_t max_batch = std::max(1u, std::min<uint32_t>(MTP_MAXL, MAXM / gamma));
  double tot_ms = 0.0;
  for (size_t off = 0; off < reqs.size(); off += max_batch) {
    const uint32_t L = uint32_t(std::min<size_t>(max_batch, reqs.size() - off));
    if (L == 1u) {
      use_lane(reqs[off].lane);
      std::vector<float> dp;
      double g = 0.0;
      out_drafts[off] = draft(reqs[off].anchor_tok, gamma, out_probs ? &dp : nullptr, &g, 0u);
      if (out_probs) (*out_probs)[off] = std::move(dp);
      tot_ms += g;
      continue;
    }

    const uint32_t MB = L * B;
    for (uint32_t i = 0; i < L; i++) {
      *(uint32_t*)((uint8_t*)[b_mtp_prev_tok contents] + size_t(i) * 16u) = reqs[off + i].anchor_tok;
    }
    memset([b_mtp_rnn_state contents], 0, size_t(L) * mtp_rank * sizeof(float));

    id<MTLCommandBuffer> cb = [C.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
    const uint32_t zero_u = 0, one_u = 1;

    // 1. Build 8-token draft input block per lane -> b_mtp_h[L * B, H]
    for (uint32_t i = 0; i < L; i++) {
      const uint32_t atok = reqs[off + i].anchor_tok;
      [enc setComputePipelineState:p_mtp_embed];
      [enc setBuffer:embed.b offset:embed.off atIndex:0];
      [enc setBuffer:embed_norm.b offset:embed_norm.off atIndex:1];
      [enc setBuffer:mtp_mask_embed.b offset:mtp_mask_embed.off atIndex:2];
      [enc setBuffer:b_mtp_h offset:size_t(i) * B * H * sizeof(float) atIndex:3];
      [enc setBytes:&atok length:4 atIndex:4];
      [enc setBytes:&H length:4 atIndex:5];
      [enc setBytes:&eps length:4 atIndex:6];
      [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    barrier();

    // Initial attn_norm for layer 0 over all MB rows -> b_mtp_frag
    [enc setComputePipelineState:p_rms_frag];
    [enc setBuffer:b_mtp_h offset:0 atIndex:0];
    [enc setBuffer:MtpL[0].attn_norm.b offset:MtpL[0].attn_norm.off atIndex:1];
    [enc setBuffer:b_mtp_frag offset:0 atIndex:2];
    [enc setBuffer:b_dummy offset:0 atIndex:3];
    [enc setBytes:&H length:4 atIndex:4];
    [enc setBytes:&eps length:4 atIndex:5];
    [enc setBytes:&zero_u length:4 atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(MB, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();

    // 2. Run 3 dense DFlare MTP layers ONCE over all MB = L*8 rows!
    for (uint32_t l = 0; l < mtp_nL; l++) {
      auto& ml = MtpL[l];
      enc_mmp_nf1(enc, p_mmp_w8_mtp, ml.qkv, b_mtp_frag, 0, b_mtp_qkv, 0, MB, nil, b_dummy);
      barrier();

      const size_t qkv_stride = size_t(nh * hd + 2u * nkv * hd) * sizeof(float);
      for (uint32_t i = 0; i < L; i++) {
        const int k = reqs[off + i].lane;
        const uint32_t lpos = (k == active_lane) ? pos : lanes_[k].pos;
        MtpQkvArgs qa{nh, nkv, hd, lpos, eps, mtp_rope_theta};
        [enc setComputePipelineState:p_mtp_qkv_rope];
        [enc setBuffer:b_mtp_qkv offset:size_t(i) * B * qkv_stride atIndex:0];
        [enc setBuffer:ml.q_norm.b offset:ml.q_norm.off atIndex:1];
        [enc setBuffer:ml.k_norm.b offset:ml.k_norm.off atIndex:2];
        [enc setBuffer:b_mtp_Q offset:size_t(i) * B * nh * hd * sizeof(float) atIndex:3];
        [enc setBuffer:b_mtp_K_blk offset:size_t(i) * B * nkv * hd * sizeof(uint16_t) atIndex:4];
        [enc setBuffer:b_mtp_V_blk offset:size_t(i) * B * nkv * hd * sizeof(uint16_t) atIndex:5];
        [enc setBytes:&qa length:sizeof(qa) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(B, nh + nkv, 1) threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
      }
      barrier();

      for (uint32_t i = 0; i < L; i++) {
        const int k = reqs[off + i].lane;
        id<MTLBuffer> kc = (k == active_lane) ? ml.k_cache : lanes_[k].mtp_k[l];
        id<MTLBuffer> vc = (k == active_lane) ? ml.v_cache : lanes_[k].mtp_v[l];
        const uint32_t mcap = (k == active_lane) ? ml.cap : lanes_[k].mtp_cap[l];
        const uint32_t mpos = (k == active_lane) ? mtp_ctx_pos : lanes_[k].mtp_ctx_pos;
        const int fsrc = (k == active_lane) ? fork_src : lanes_[k].fork_src;
        const uint32_t fpos = (k == active_lane) ? fork_pos : lanes_[k].fork_pos;
        const uint32_t mfpos = (!ml.is_local && fsrc >= 0) ? fpos : 0u;
        id<MTLBuffer> pk = (mfpos > 0u) ? (fsrc == active_lane ? ml.k_cache : lanes_[fsrc].mtp_k[l]) : b_dummy;
        id<MTLBuffer> pv = (mfpos > 0u) ? (fsrc == active_lane ? ml.v_cache : lanes_[fsrc].mtp_v[l]) : b_dummy;
        MtpAttnArgs aa{B, nh, nkv, hd, mpos, mcap, window, ml.is_local ? 1u : 0u, mfpos, i * B};
        [enc setComputePipelineState:p_mtp_attn];
        [enc setBuffer:b_mtp_Q offset:size_t(i) * B * nh * hd * sizeof(float) atIndex:0];
        [enc setBuffer:kc offset:0 atIndex:1];
        [enc setBuffer:vc offset:0 atIndex:2];
        [enc setBuffer:b_mtp_K_blk offset:size_t(i) * B * nkv * hd * sizeof(uint16_t) atIndex:3];
        [enc setBuffer:b_mtp_V_blk offset:size_t(i) * B * nkv * hd * sizeof(uint16_t) atIndex:4];
        [enc setBuffer:b_mtp_o_frag offset:0 atIndex:5];
        [enc setBytes:&aa length:sizeof(aa) atIndex:6];
        [enc setBuffer:pk offset:0 atIndex:7];
        [enc setBuffer:pv offset:0 atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(B, nkv, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      }
      barrier();

      enc_mmp_nf1(enc, p_mmp_w8_mtp, ml.o, b_mtp_o_frag, 0, b_mtp_o_out, 0, MB, nil, b_dummy);
      barrier();

      [enc setComputePipelineState:p_add_rms_frag];
      [enc setBuffer:b_mtp_h offset:0 atIndex:0];
      [enc setBuffer:b_mtp_o_out offset:0 atIndex:1];
      [enc setBuffer:ml.mlp_norm.b offset:ml.mlp_norm.off atIndex:2];
      [enc setBuffer:b_mtp_frag offset:0 atIndex:3];
      [enc setBuffer:b_dummy offset:0 atIndex:4];
      [enc setBytes:&H length:4 atIndex:5];
      [enc setBytes:&eps length:4 atIndex:6];
      [enc setBytes:&zero_u length:4 atIndex:7];
      [enc dispatchThreadgroups:MTLSizeMake(MB, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      barrier();

      enc_mmp_nf1(enc, p_mmp_w6_gu, ml.mlp_gu, b_mtp_frag, 0, b_dummy, 0, MB, b_mtp_gu_frag, b_dummy, mtp_I);
      barrier();

      enc_mmp_nf1(enc, p_mmp_w6_dn, ml.mlp_dn, b_mtp_gu_frag, 0, b_mtp_dn_out, 0, MB, nil, b_dummy);
      barrier();

      const bool last_l = (l + 1u == mtp_nL);
      const Raw& next_nw = last_l ? mtp_norm : MtpL[l + 1u].attn_norm;
      const uint32_t wf32 = last_l ? one_u : zero_u;
      [enc setComputePipelineState:p_add_rms_frag];
      [enc setBuffer:b_mtp_h offset:0 atIndex:0];
      [enc setBuffer:b_mtp_dn_out offset:0 atIndex:1];
      [enc setBuffer:next_nw.b offset:next_nw.off atIndex:2];
      [enc setBuffer:b_mtp_frag offset:0 atIndex:3];
      [enc setBuffer:(last_l ? b_mtp_base_h : b_dummy) offset:0 atIndex:4];
      [enc setBytes:&H length:4 atIndex:5];
      [enc setBytes:&eps length:4 atIndex:6];
      [enc setBytes:&wf32 length:4 atIndex:7];
      [enc dispatchThreadgroups:MTLSizeMake(MB, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      barrier();
    }

    // 3. Pack ONLY the L*gamma (<= 16) active draft rows -> single batched lm_head || static RNN per lane!
    {
      PackFragArgs pfa{H, B, gamma};
      [enc setComputePipelineState:p_pack_frag_rows];
      [enc setBuffer:b_mtp_base_h offset:0 atIndex:0];
      [enc setBuffer:b_mtp_frag offset:0 atIndex:1];
      [enc setBytes:&pfa length:sizeof(pfa) atIndex:2];
      [enc dispatchThreadgroups:MTLSizeMake(L * gamma, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    for (uint32_t i = 0; i < L; i++) {
      const int k = reqs[off + i].lane;
      id<MTLBuffer> fr = (k == active_lane) ? b_mtp_f_rnn : lanes_[k].mtp_f_rnn;
      RnnArgs sra{H, mtp_rank, gamma};
      [enc setComputePipelineState:p_mtp_rnn_static];
      [enc setBuffer:b_mtp_base_h offset:size_t(i) * B * H * sizeof(float) atIndex:0];
      [enc setBuffer:fr offset:0 atIndex:1];
      [enc setBuffer:mtp_mh_rnn_w.b offset:mtp_mh_rnn_w.off atIndex:2];
      [enc setBuffer:mtp_mh_rnn_b.b offset:mtp_mh_rnn_b.off atIndex:3];
      [enc setBuffer:b_mtp_rnn_static offset:size_t(i) * 7u * 3u * mtp_rank * sizeof(float) atIndex:4];
      [enc setBytes:&sra length:sizeof(sra) atIndex:5];
      [enc dispatchThreadgroups:MTLSizeMake((3u * mtp_rank + 3u) / 4u, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    barrier();

    enc_mm2(enc, p_mm_w8, lm_head, b_mtp_frag, 0, b_mtp_base_logits, 0, L * gamma, nil, b_dummy, nil, nil, nil, false);
    barrier();

    // 4. Sequential Markov RNN head across all L lanes simultaneously (1 batched fold_w matmul per step!)
    for (uint32_t step_k = 0; step_k < gamma; step_k++) {
      for (uint32_t i = 0; i < L; i++) {
        RnnArgs ra{H, mtp_rank, step_k};
        [enc setComputePipelineState:p_mtp_rnn_dyn];
        [enc setBuffer:b_mtp_rnn_state offset:size_t(i) * mtp_rank * sizeof(float) atIndex:0];
        [enc setBuffer:mtp_mh_markov_w1.b offset:mtp_mh_markov_w1.off atIndex:1];
        [enc setBuffer:b_mtp_prev_tok offset:size_t(i) * 16u atIndex:2];
        [enc setBuffer:b_mtp_rnn_static offset:size_t(i) * 7u * 3u * mtp_rank * sizeof(float) atIndex:3];
        [enc setBuffer:mtp_mh_rnn_w.b offset:mtp_mh_rnn_w.off atIndex:4];
        [enc setBuffer:b_mtp_rnn_proj offset:size_t(i) * 3u * mtp_rank * sizeof(float) atIndex:5];
        [enc setBytes:&ra length:sizeof(ra) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake((3u * mtp_rank + 3u) / 4u, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      }
      barrier();

      for (uint32_t i = 0; i < L; i++) {
        [enc setComputePipelineState:p_mtp_rnn_gates];
        [enc setBuffer:b_mtp_rnn_proj offset:size_t(i) * 3u * mtp_rank * sizeof(float) atIndex:0];
        [enc setBuffer:b_mtp_rnn_state offset:size_t(i) * mtp_rank * sizeof(float) atIndex:1];
        [enc setBuffer:b_mtp_lat_frag offset:0 atIndex:2];
        [enc setBytes:&mtp_rank length:4 atIndex:3];
        [enc setBytes:&i length:4 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      barrier();

      // ONE batched folded Markov LM head over all L lanes!
      enc_mm2(enc, p_mm_w8, mtp_mh_fold_w, b_mtp_lat_frag, 0, b_mtp_bias, 0, L, nil, b_dummy, nil, nil, nil, false);
      barrier();

      for (uint32_t i = 0; i < L; i++) {
        const int k = reqs[off + i].lane;
        const Samp& sp = (k == active_lane) ? samp : lanes_[k].samp;
        const uint32_t lpos = (k == active_lane) ? pos : lanes_[k].pos;
        MtpPickArgs pa{vocab_valid, vocab_pad, step_k, logit_softcap, sp.temp > 0.0f ? 1.0f / sp.temp : -1.0f,
                       uint32_t(sp.seed), uint32_t(sp.seed >> 32), lpos};
        [enc setComputePipelineState:p_mtp_add_argmax_p1];
        [enc setBuffer:b_mtp_base_logits offset:size_t(i) * gamma * vocab_pad * sizeof(float) atIndex:0];
        [enc setBuffer:b_mtp_bias offset:size_t(i) * vocab_pad * sizeof(float) atIndex:1];
        [enc setBuffer:mtp_mh_fold_b.b offset:mtp_mh_fold_b.off atIndex:2];
        [enc setBuffer:b_argmax_v offset:size_t(i) * 64u * sizeof(float) atIndex:3];
        [enc setBuffer:b_argmax_i offset:size_t(i) * 64u * sizeof(uint32_t) atIndex:4];
        [enc setBuffer:b_mtp_part_l offset:size_t(i) * 64u * sizeof(float) atIndex:5];
        [enc setBytes:&pa length:sizeof(pa) atIndex:6];
        [enc setBuffer:b_mtp_part_x offset:size_t(i) * 128u * sizeof(float) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(64, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      barrier();

      for (uint32_t i = 0; i < L; i++) {
        const int k = reqs[off + i].lane;
        const Samp& sp = (k == active_lane) ? samp : lanes_[k].samp;
        const uint32_t lpos = (k == active_lane) ? pos : lanes_[k].pos;
        MtpPickArgs pa{vocab_valid, vocab_pad, step_k, logit_softcap, sp.temp > 0.0f ? 1.0f / sp.temp : -1.0f,
                       uint32_t(sp.seed), uint32_t(sp.seed >> 32), lpos};
        [enc setComputePipelineState:p_mtp_add_argmax_p2];
        [enc setBuffer:b_argmax_v offset:size_t(i) * 64u * sizeof(float) atIndex:0];
        [enc setBuffer:b_argmax_i offset:size_t(i) * 64u * sizeof(uint32_t) atIndex:1];
        [enc setBuffer:b_mtp_part_l offset:size_t(i) * 64u * sizeof(float) atIndex:2];
        [enc setBuffer:b_mtp_prev_tok offset:size_t(i) * 16u atIndex:3];
        [enc setBuffer:b_mtp_draft_out offset:size_t(i) * MAXM * sizeof(uint32_t) atIndex:4];
        [enc setBuffer:b_mtp_draft_probs offset:size_t(i) * MAXM * sizeof(float) atIndex:5];
        [enc setBytes:&pa length:sizeof(pa) atIndex:6];
        [enc setBuffer:b_mtp_part_x offset:size_t(i) * 128u * sizeof(float) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      }
      barrier();
    }

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) mtl::die("GPU draft_lanes error", cb.error);
    tot_ms += (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;

    for (uint32_t i = 0; i < L; i++) {
      out_drafts[off + i].resize(gamma);
      memcpy(out_drafts[off + i].data(), (const uint32_t*)[b_mtp_draft_out contents] + size_t(i) * MAXM, gamma * sizeof(uint32_t));
      if (out_probs) {
        (*out_probs)[off + i].resize(gamma);
        memcpy((*out_probs)[off + i].data(), (const float*)[b_mtp_draft_probs contents] + size_t(i) * MAXM, gamma * sizeof(float));
      }
    }
  }
  if (gpu_ms) *gpu_ms = tot_ms;
}

namespace {
struct InkPrefixHdr {
  uint32_t magic;       // 0x504B4E49u ("INKP")
  uint32_t version;     // 1
  uint64_t fingerprint;
  uint32_t pos;
  uint32_t n_log;
  uint32_t use_kv8;
  uint32_t has_mtp;
  uint32_t mtp_ctx_pos;
  uint32_t window;
};
constexpr uint32_t kInkPrefixMagic = 0x504B4E49u;
constexpr uint32_t kInkPrefixVer = 1u;

bool write_all(FILE* f, const void* p, size_t n) {
  return n == 0 || fwrite(p, 1, n, f) == n;
}
bool read_all(FILE* f, void* p, size_t n) {
  return n == 0 || fread(p, 1, n, f) == n;
}
}  // namespace

bool Model::prefix_tokens(const std::string& path, uint64_t fp, std::vector<uint32_t>& toks, size_t* bytes) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  InkPrefixHdr hdr{};
  bool ok = read_all(f, &hdr, sizeof(hdr)) && hdr.magic == kInkPrefixMagic &&
            hdr.version == kInkPrefixVer && hdr.fingerprint == fp && hdr.pos > 0;
  if (ok) {
    toks.resize(hdr.pos);
    ok = read_all(f, toks.data(), size_t(hdr.pos) * sizeof(uint32_t));
  }
  if (ok && bytes) {
    struct stat st{};
    *bytes = (fstat(fileno(f), &st) == 0) ? size_t(st.st_size) : 0;
  }
  fclose(f);
  if (!ok) toks.clear();
  return ok;
}

bool Model::save_prefix(const std::string& path, const std::vector<uint32_t>& history,
                        double* ms, size_t* out_bytes) const {
  if (pos == 0 || fork_pos != 0 || history.size() < pos) return false;
  const double t0 = CFAbsoluteTimeGetCurrent();
  const std::string tmp = path + ".tmp";
  int fd = open(tmp.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0600);
  if (fd < 0) return false;
  FILE* f = fdopen(fd, "wb");
  if (!f) { close(fd); return false; }

  InkPrefixHdr hdr{kInkPrefixMagic, kInkPrefixVer, fingerprint, pos, uint32_t(S.size()),
                   use_kv8 ? 1u : 0u, has_mtp ? 1u : 0u, mtp_ctx_pos, window};
  bool ok = write_all(f, &hdr, sizeof(hdr)) &&
            write_all(f, history.data(), size_t(pos) * sizeof(uint32_t));

  const size_t c_kv = size_t(nkv) * hd;
  const size_t kv_conv_bytes = 3u * c_kv * sizeof(float);
  const size_t h_conv_bytes = 3u * size_t(H) * sizeof(float);
  const size_t kv_row_bf16 = c_kv * sizeof(uint16_t);
  const size_t kv_row_i8 = c_kv * sizeof(int8_t);
  const size_t kv_row_sc = size_t(nkv) * sizeof(uint16_t);

  auto write_ring = [&](id<MTLBuffer> buf, uint32_t P, uint32_t rcap, size_t row_bytes) -> bool {
    const uint32_t n = std::min(P, window);
    if (n == 0) return true;
    const uint32_t start = (P - n) % rcap;
    const uint32_t first = std::min(n, rcap - start);
    const uint8_t* p = (const uint8_t*)[buf contents];
    if (!write_all(f, p + size_t(start) * row_bytes, size_t(first) * row_bytes)) return false;
    if (n > first && !write_all(f, p, size_t(n - first) * row_bytes)) return false;
    return true;
  };

  for (size_t i = 0; ok && i < S.size(); i++) {
    const auto& st = S[i];
    ok = write_all(f, [st.k_conv contents], kv_conv_bytes) &&
         write_all(f, [st.v_conv contents], kv_conv_bytes) &&
         write_all(f, [st.attn_conv contents], h_conv_bytes) &&
         write_all(f, [st.mlp_conv contents], h_conv_bytes);
    if (!ok) break;
    if (L[st.phys_idx].is_local) {
      ok = write_ring(st.k_cache, pos, st.cap, kv_row_bf16) &&
           write_ring(st.v_cache, pos, st.cap, kv_row_bf16);
    } else if (st.kv8) {
      ok = write_all(f, [st.k_i8 contents], size_t(pos) * kv_row_i8) &&
           write_all(f, [st.k_sc contents], size_t(pos) * kv_row_sc) &&
           write_all(f, [st.v_i8 contents], size_t(pos) * kv_row_i8) &&
           write_all(f, [st.v_sc contents], size_t(pos) * kv_row_sc);
    } else {
      ok = write_all(f, [st.k_cache contents], size_t(pos) * kv_row_bf16) &&
           write_all(f, [st.v_cache contents], size_t(pos) * kv_row_bf16);
    }
  }

  if (ok && has_mtp) {
    ok = write_all(f, [b_mtp_f_rnn contents], size_t(H) * sizeof(float));
    for (size_t l = 0; ok && l < MtpL.size(); l++) {
      const auto& ml = MtpL[l];
      if (ml.is_local) {
        ok = write_ring(ml.k_cache, mtp_ctx_pos, ml.cap, kv_row_bf16) &&
             write_ring(ml.v_cache, mtp_ctx_pos, ml.cap, kv_row_bf16);
      } else {
        ok = write_all(f, [ml.k_cache contents], size_t(mtp_ctx_pos) * kv_row_bf16) &&
             write_all(f, [ml.v_cache contents], size_t(mtp_ctx_pos) * kv_row_bf16);
      }
    }
  }

  if (ok) ok = (fflush(f) == 0);
  size_t sz = 0;
  if (ok) {
    struct stat st{};
    if (fstat(fileno(f), &st) == 0) sz = size_t(st.st_size);
  }
  fclose(f);
  if (ok && rename(tmp.c_str(), path.c_str()) != 0) ok = false;
  if (!ok) unlink(tmp.c_str());
  if (ms) *ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0;
  if (out_bytes) *out_bytes = ok ? sz : 0;
  return ok;
}

uint32_t Model::load_prefix(const std::string& path, std::vector<uint32_t>* out_toks, double* ms) {
  const double t0 = CFAbsoluteTimeGetCurrent();
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return 0;
  InkPrefixHdr hdr{};
  bool ok = read_all(f, &hdr, sizeof(hdr)) && hdr.magic == kInkPrefixMagic &&
            hdr.version == kInkPrefixVer && hdr.fingerprint == fingerprint &&
            hdr.n_log == S.size() && hdr.pos > 0 && hdr.pos <= cap &&
            hdr.use_kv8 == (use_kv8 ? 1u : 0u) && hdr.has_mtp == (has_mtp ? 1u : 0u) &&
            hdr.window == window;
  std::vector<uint32_t> toks;
  if (ok) {
    toks.resize(hdr.pos);
    ok = read_all(f, toks.data(), size_t(hdr.pos) * sizeof(uint32_t));
  }

  const size_t c_kv = size_t(nkv) * hd;
  const size_t kv_conv_bytes = 3u * c_kv * sizeof(float);
  const size_t h_conv_bytes = 3u * size_t(H) * sizeof(float);
  const size_t kv_row_bf16 = c_kv * sizeof(uint16_t);
  const size_t kv_row_i8 = c_kv * sizeof(int8_t);
  const size_t kv_row_sc = size_t(nkv) * sizeof(uint16_t);

  auto read_ring = [&](id<MTLBuffer> buf, uint32_t P, uint32_t rcap, size_t row_bytes) -> bool {
    const uint32_t n = std::min(P, window);
    if (n == 0) return true;
    const uint32_t start = (P - n) % rcap;
    const uint32_t first = std::min(n, rcap - start);
    uint8_t* p = (uint8_t*)[buf contents];
    if (!read_all(f, p + size_t(start) * row_bytes, size_t(first) * row_bytes)) return false;
    if (n > first && !read_all(f, p, size_t(n - first) * row_bytes)) return false;
    return true;
  };

  for (size_t i = 0; ok && i < S.size(); i++) {
    auto& st = S[i];
    ok = read_all(f, [st.k_conv contents], kv_conv_bytes) &&
         read_all(f, [st.v_conv contents], kv_conv_bytes) &&
         read_all(f, [st.attn_conv contents], h_conv_bytes) &&
         read_all(f, [st.mlp_conv contents], h_conv_bytes);
    if (!ok) break;
    if (L[st.phys_idx].is_local) {
      ok = read_ring(st.k_cache, hdr.pos, st.cap, kv_row_bf16) &&
           read_ring(st.v_cache, hdr.pos, st.cap, kv_row_bf16);
    } else if (st.kv8) {
      ok = read_all(f, [st.k_i8 contents], size_t(hdr.pos) * kv_row_i8) &&
           read_all(f, [st.k_sc contents], size_t(hdr.pos) * kv_row_sc) &&
           read_all(f, [st.v_i8 contents], size_t(hdr.pos) * kv_row_i8) &&
           read_all(f, [st.v_sc contents], size_t(hdr.pos) * kv_row_sc);
    } else {
      ok = read_all(f, [st.k_cache contents], size_t(hdr.pos) * kv_row_bf16) &&
           read_all(f, [st.v_cache contents], size_t(hdr.pos) * kv_row_bf16);
    }
  }

  if (ok && has_mtp) {
    ok = read_all(f, [b_mtp_f_rnn contents], size_t(H) * sizeof(float));
    for (size_t l = 0; ok && l < MtpL.size(); l++) {
      auto& ml = MtpL[l];
      if (ml.is_local) {
        ok = read_ring(ml.k_cache, hdr.mtp_ctx_pos, ml.cap, kv_row_bf16) &&
             read_ring(ml.v_cache, hdr.mtp_ctx_pos, ml.cap, kv_row_bf16);
      } else {
        ok = read_all(f, [ml.k_cache contents], size_t(hdr.mtp_ctx_pos) * kv_row_bf16) &&
             read_all(f, [ml.v_cache contents], size_t(hdr.mtp_ctx_pos) * kv_row_bf16);
      }
    }
  }
  fclose(f);
  if (!ok) {
    reset();
    return 0;
  }
  pos = hdr.pos;
  fork_pos = 0;
  fork_src = -1;
  mtp_ctx_pos = hdr.mtp_ctx_pos;
  if (out_toks) *out_toks = std::move(toks);
  if (ms) *ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0;
  return pos;
}

}  // namespace ink2
