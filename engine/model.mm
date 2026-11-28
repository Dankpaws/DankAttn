#include "model.h"
#include "core/runtime.h"
#include "mm_args.h"
#include "ngram.h"
#include <cmath>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>

namespace opus {

using dankattn::jload;
using dankattn::new_sample_buffer;

struct Args4 { uint32_t N, K, M, ksplit, wstride, sstride, xstride, ystride, yfstride; };
struct HcArgs { uint32_t M, H, HC, inject, Nd, R, topk; };
struct MidArgs { uint32_t M, split, N, R, HC, combine; };
struct MixArgs { uint32_t M, split, H, HC; };
struct EmbArgs { uint32_t M, H, HC; };
struct ArgArgs { uint32_t M, split, N; };
struct RouteArgs { uint32_t M, split, N, E, I, topk; };
struct PlanArgs { uint32_t M, topk, E; };
struct ActArgs { uint32_t M, split, N, I; };
struct CombArgs { uint32_t M, split, H, topk, ssplit; };
struct AttnArgs { uint32_t M, split, N, nh, nkv, hd, inh, ihd, rot, p0, ratio, cap, budget, row0 = 0, kv8 = 0; };
struct PleArgs { uint32_t M, split, N, H, HC, par, hist, dil, ksz, mode; };

constexpr size_t PROF_OPS = dankattn::Rec::PROF_OPS;

struct Rec : dankattn::Rec {
  Rec() {
    conc = false;
    static const std::string skip = getenv("OPUS_SKIP") ? "," + std::string(getenv("OPUS_SKIP")) + "," : "";
    skip_csv = &skip;
  }
};

// Slices per output fragment (simdgroups splitting K) for >= ~1024 simdgroups in flight,
// as threadgroup simdgroups x threadgroup splits. Fixed per tensor, never by M.
// Slices per output fragment (simdgroups splitting K) for >= ~1024 simdgroups in flight,
// as threadgroup simdgroups (<= 8, reduced in threadgroup memory) x threadgroup splits
// (reduced through atomics). Fixed per tensor, never by M.
// Measured splits for the dense decode shapes (bench/mmchain SWEEP, 2026-11-14, M5 Ultra, cold weights,
// scored M1 + 2 M4 + M8): fewer simdgroups per threadgroup and more threadgroup splits beat the
// generic pick, because the threadgroup-level reduce (a barrier on the slowest simdgroup) costs more
// than the atomic last-arrival join. {N, K, gs, sgs, ksplit, grouped}; OPUS_SPLIT_TAB=0 restores the generic pick.
static const uint32_t split_tab[][6] = {
    {352, 10240, 128, 4, 10, 0},   // hc down: 13.4/15.2/17.6 -> 11.9/13.7/16.1 us at 1/4/8 rows
    {10240, 320, 64, 1, 1, 0},     // hc up: 9.8/9.9/10.0 -> 9.2/9.2/9.3
    {1824, 2560, 128, 1, 10, 0},   // moe.in: 12.0/13.3/15.0 -> 9.4/11.2/14.1
    {2560, 6144, 128, 1, 8, 0},    // attention / GDN out: 27.2/28.4/29.8 -> 19.9/21.5/23.7
    {16480, 2560, 128, 1, 5, 0},   // GDN in: 54.8/55.1/55.4 -> 45.8/47.4/49.8
    {13952, 2560, 128, 2, 1, 0},   // attention in: 41.1/41.4/41.7 -> 38.3/38.5/38.9
    {248320, 2560, 128, 5, 1, 0},  // lm_head: 586/587/595 -> 566/569/576
    {2560, 2560, 128, 1, 10, 0},   // MTP fc: 13.9/15.3/13.4 -> 10.9/12.7/11.9
    {320, 10240, 128, 4, 10, 0},   // mixer down: 13.2/14.9/11.9 -> 11.9/13.6/10.6
    {2560, 640, 64, 1, 1, 1},      // routed expert down (10 / 36 / 65 experts): 15.4/49.4/79.0 -> 16.1/44.1/75.0
};

static void pick_split(uint32_t N, uint32_t K, uint32_t gs, uint32_t slots, uint32_t& sgs, uint32_t& ksplit) {
  const uint32_t G = K / gs, nf = N / 32 * slots;
  static const bool tab = !(getenv("OPUS_SPLIT_TAB") && atoi(getenv("OPUS_SPLIT_TAB")) == 0);
  if (tab)
    for (auto& r : split_tab)
      if (r[0] == N && r[1] == K && r[2] == gs && (r[5] != 0) == (slots > 1) && G % (r[3] * r[4]) == 0) { sgs = r[3]; ksplit = r[4]; return; }
  uint32_t s_tot = G;
  for (uint32_t s = 1; s <= G; s++)
    if (G % s == 0 && nf * s >= 1024) { s_tot = s; break; }
  // experiment: override the slice count of the hc down (K = 4H) / up (N = 4H) matmuls
  if (getenv("OPUS_DN_S") && K == 10240 && G % atoi(getenv("OPUS_DN_S")) == 0) s_tot = atoi(getenv("OPUS_DN_S"));
  if (getenv("OPUS_UP_S") && N == 10240 && G % atoi(getenv("OPUS_UP_S")) == 0) s_tot = atoi(getenv("OPUS_UP_S"));
  for (;;) {
    sgs = 1;
    for (uint32_t c = 8; c >= 2; c--)
      if (s_tot % c == 0) { sgs = c; break; }
    ksplit = s_tot / sgs;
    if (ksplit <= 16) break;
    uint32_t nx = 1;  // the cross-threadgroup reduce holds <= 16 partials
    for (uint32_t s = s_tot - 1; s >= 1; s--)
      if (G % s == 0) { nx = s; break; }
    s_tot = nx;
  }
}

Model::Model(const std::string& pack_dir, uint32_t capacity, const Model* share) : dir(pack_dir), cap(capacity) {
  if (share) C = share->C;
  C.made.clear();
  C.track = false;  // the state set measured no gain over the weights alone (OPUS_RESIDENCY_STATE=1 adds it)
  if (getenv("OPUS_RESIDENCY_STATE")) C.track = !getenv("OPUS_RESIDENCY_MS") || atoll(getenv("OPUS_RESIDENCY_MS")) > 0;
  NSDictionary* man = jload(dir + "/manifest.json");
  NSDictionary* cfg = man[@"config"];
  H = [cfg[@"hidden_size"] unsignedIntValue];
  HC = [cfg[@"hc_count"] unsignedIntValue];
  R = [cfg[@"hc_lowrank"] unsignedIntValue];
  nL = [cfg[@"num_hidden_layers"] unsignedIntValue];
  vocab = [cfg[@"vocab_size"] unsignedIntValue];
  E = [cfg[@"num_experts"] unsignedIntValue];
  topk = [cfg[@"num_experts_per_tok"] unsignedIntValue];
  I = [cfg[@"moe_intermediate_size"] unsignedIntValue];
  id eosv = cfg[@"eos_token_id"];
  eos = [eosv isKindOfClass:[NSArray class]] ? [eosv[0] unsignedIntValue] : [eosv unsignedIntValue];
  nh = [cfg[@"num_attention_heads"] unsignedIntValue];
  nkv = [cfg[@"num_key_value_heads"] unsignedIntValue];
  hd = [cfg[@"head_dim"] unsignedIntValue];
  inh = [cfg[@"indexer_n_heads"] unsignedIntValue];
  ihd = [cfg[@"indexer_head_dim"] unsignedIntValue];
  ratio = [cfg[@"indexer_compress_ratio"] unsignedIntValue];
  budget = [cfg[@"indexer_budget"] unsignedIntValue];
  nk = [cfg[@"linear_num_key_heads"] unsignedIntValue];
  nv = [cfg[@"linear_num_value_heads"] unsignedIntValue];
  dk = [cfg[@"linear_key_head_dim"] unsignedIntValue];
  dv = [cfg[@"linear_value_head_dim"] unsignedIntValue];
  NSDictionary* rp = cfg[@"rope_parameters"];
  rope_theta = [rp[@"rope_theta"] doubleValue];
  rot = uint32_t(hd * [rp[@"partial_rotary_factor"] doubleValue]);
  for (NSNumber* n in cfg[@"ple_layer_ids"]) ple_layers.push_back([n intValue] - 1);

  // map pack files without copying; keep them resident
  auto t0 = std::chrono::steady_clock::now();
  size_t total = 0;
  if (share) {
    files = share->files;
  } else {
  id<MTLResidencySet> rs;
  {
    MTLResidencySetDescriptor* d = [MTLResidencySetDescriptor new];
    d.label = @"opus-weights";
    NSError* e = nil;
    rs = [C.dev newResidencySetWithDescriptor:d error:&e];
    if (!rs) mtl::die("residency set", e);
  }
  for (NSString* f in man[@"files"]) {
    std::string fn = [f UTF8String];
    mtl::check_pack_filename(fn);
    std::string p = dir + "/" + fn;
    int fd = open(p.c_str(), O_RDONLY);
    if (fd < 0) mtl::die(("open " + p).c_str());
    struct stat st;
    fstat(fd, &st);
    size_t len = (size_t(st.st_size) + 16383) & ~size_t(16383);
    void* m = mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (m == MAP_FAILED) mtl::die(("mmap " + p).c_str());
    id<MTLBuffer> b = [C.dev newBufferWithBytesNoCopy:m length:len options:MTLResourceStorageModeShared deallocator:nil];
    if (!b) mtl::die(("nocopy buffer " + p).c_str());
    [rs addAllocation:b];
    files.push_back(b);
    total += len;
  }
  [rs commit];
  [rs requestResidency];
  [C.q addResidencySet:rs];
  // macOS drops the wiring a few seconds after the GPU goes idle, so the first step of the next
  // request pays for wiring the pack again. OPUS_RESIDENCY_MS > 0 asks again on a timer.
  {  // default 1000 ms; OPUS_RESIDENCY_MS=0 turns the refresh off
    const char* ms = getenv("OPUS_RESIDENCY_MS");
    const int64_t iv = (ms ? atoll(ms) : 1000) * int64_t(NSEC_PER_MSEC);
    if (iv > 0) {
      static dispatch_source_t tm;
      tm = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, dispatch_get_global_queue(QOS_CLASS_UTILITY, 0));
      dispatch_source_set_timer(tm, dispatch_time(DISPATCH_TIME_NOW, iv), uint64_t(iv), uint64_t(iv / 10));
      dispatch_source_set_event_handler(tm, ^{ if (mtl::keep_warm()) [rs requestResidency]; });
      dispatch_resume(tm);
    }
  }
  }

  NSDictionary* T = man[@"tensors"];
  fused_layout = man[@"layout"] != nil;
  auto lin = [&](const std::string& name, uint32_t slots = 1) {
    NSDictionary* t = T[@(name.c_str())];
    if (!t) mtl::die(("missing tensor " + name).c_str());
    Lin l;
    l.woff = [t[@"off"] unsignedLongLongValue];
    l.w = mtl::checked_file(files, [t[@"file"] unsignedIntValue], l.woff, 0, name);
    l.soff = [t[@"soff"] unsignedLongLongValue];
    l.s = mtl::checked_file(files, [t[@"sfile"] unsignedIntValue], l.soff, 0, name + ".scales");
    l.N = [t[@"N"] unsignedIntValue];
    l.K = [t[@"K"] unsignedIntValue];
    l.bits = [t[@"bits"] unsignedIntValue];
    l.gs = [t[@"gs"] unsignedIntValue];
    l.experts = [t[@"experts"] unsignedIntValue];
    NSDictionary* rows = t[@"rows"];
    for (NSString* k in rows) l.rows[[k UTF8String]] = {[rows[k][0] unsignedIntValue], [rows[k][1] unsignedIntValue]};
    pick_split(l.N, l.K, l.gs, slots, l.sgs, l.ksplit);
    return l;
  };
  auto raw = [&](const std::string& name) {
    NSDictionary* t = T[@(name.c_str())];
    if (!t) mtl::die(("missing tensor " + name).c_str());
    Raw r;
    r.off = [t[@"off"] unsignedLongLongValue];
    r.bytes = [t[@"bytes"] unsignedLongLongValue];
    r.b = mtl::checked_file(files, [t[@"file"] unsignedIntValue], r.off, r.bytes, name);
    return r;
  };
  NSArray* lt = cfg[@"layer_types"];
  L.resize(nL);
  for (uint32_t i = 0; i < nL; i++) {
    Layer& l = L[i];
    std::string d = "L" + std::to_string(i);
    l.attn = ![lt[i] isEqualToString:@"linear_attention"];
    l.ahc_dn = lin(d + ".ahc.dn");
    l.ahc_up = lin(d + ".ahc.up");
    l.ahc_norm = raw(d + ".ahc.norm");
    l.mhc_dn = lin(d + ".mhc.dn");
    l.mhc_up = lin(d + ".mhc.up");
    l.mhc_norm = raw(d + ".mhc.norm");
    if (l.attn) {
      l.in = lin(d + ".attn.in");
      l.out = lin(d + ".attn.o");
      l.q_norm = raw(d + ".attn.q_norm");
      l.k_norm = raw(d + ".attn.k_norm");
      l.iq_norm = raw(d + ".attn.indexer.q_layernorm");
      l.ik_norm = raw(d + ".attn.indexer.k_layernorm");
      l.Kc = C.buf(kv_bytes(cap));
      l.Vc = C.buf(kv_bytes(cap));
      l.rawk = C.buf(size_t(cap) * ihd * 2);
      l.bk = C.buf(size_t(cap / ratio + 128) * ihd * 4);  // +128: score tiles read whole 32-block groups
    } else {
      l.in = lin(d + ".gdn.in");
      l.out = lin(d + ".gdn.out");
      l.conv = raw(d + ".gdn.conv");
      l.A_log = raw(d + ".gdn.A_log");
      l.dt_bias = raw(d + ".gdn.dt_bias");
      l.gnorm = raw(d + ".gdn.norm");
      l.S = C.buf(size_t(nv) * dk * dv * 4);
      // decode/verify rows only (prefill reads yin). Joint speculation would need MP rows here,
      // 4.6 GiB per slot, so it stays off until it is correct (generate_spec_lanes refuses).
      // only one step is in flight and lanes commit from the driver's, so share-mode slots use the driver's
      l.gin = share ? share->L[i].gin : C.buf(size_t(MAXM) * l.in.N * 4);
      l.convh = C.buf(size_t(2) * 3 * (2 * nk * dk + nv * dv) * 4);
    }
    l.moe_in = lin(d + ".moe.in");
    l.moe_sdown = lin(d + ".moe.sdown");
    // split sized for one row (topk experts); fixed for every M so row bits never change
    l.moe_gu = lin(d + ".moe.gu", topk);
    l.moe_dn = lin(d + ".moe.dn", topk);
    for (int pl : ple_layers)
      if (pl == int(i)) {
        l.ple = true;
        l.ple_kv = lin(d + ".ple.kv");
        l.ple_nk = raw(d + ".ple.norm_key");
        l.ple_nq = raw(d + ".ple.norm_query");
        l.ple_nc = raw(d + ".ple.norm_conv");
        l.ple_conv = raw(d + ".ple.conv");
        l.plehist = C.buf(size_t(2) * 9 * H * HC * 4);
      }
  }
  if (T[@"MTP.attn.in"]) {
    has_mtp = true;
    Layer& l = mtp;
    l.attn = true;
    l.ahc_dn = lin("MTP.ahc.dn");
    l.ahc_up = lin("MTP.ahc.up");
    l.ahc_norm = raw("MTP.ahc.norm");
    l.mhc_dn = lin("MTP.mhc.dn");
    l.mhc_up = lin("MTP.mhc.up");
    l.mhc_norm = raw("MTP.mhc.norm");
    l.in = lin("MTP.attn.in");
    l.out = lin("MTP.attn.o");
    l.q_norm = raw("MTP.attn.q_norm");
    l.k_norm = raw("MTP.attn.k_norm");
    l.iq_norm = raw("MTP.attn.indexer.q_layernorm");
    l.ik_norm = raw("MTP.attn.indexer.k_layernorm");
    l.Kc = C.buf(kv_bytes(cap));
    l.Vc = C.buf(kv_bytes(cap));
    l.rawk = C.buf(size_t(cap) * ihd * 2);
    l.bk = C.buf(size_t(cap / ratio + 128) * ihd * 4);  // +128: score tiles read whole 32-block groups
    l.moe_in = lin("MTP.moe.in");
    l.moe_sdown = lin("MTP.moe.sdown");
    l.moe_gu = lin("MTP.moe.gu", topk);
    l.moe_dn = lin("MTP.moe.dn", topk);
    mtp_mix_dn = lin("MTP.mix.dn");
    mtp_mix_up = lin("MTP.mix.up");
    mtp_mix_norm = raw("MTP.mix.norm");
    mtp_fc_emb = lin("MTP.fc_emb");
    mtp_fc_hid = lin("MTP.fc_hid");
    mtp_norm_emb = raw("MTP.norm_emb");
    mtp_norm_hid = raw("MTP.norm_hid");
    if (T[@"MTP.lm_head_sub"] && !getenv("OPUS_FULL_DRAFT_HEAD")) {
      mtp_head = lin("MTP.lm_head_sub");
      sub_ids = raw("MTP.lm_head_sub_ids");
      has_draft_head = has_sub_head = true;
    } else if (T[@"MTP.lm_head_w4"]) {
      mtp_head = lin("MTP.lm_head_w4");
      has_draft_head = true;
    }
  }
  mix_dn = lin("mix.dn");
  mix_up = lin("mix.up");
  mix_norm = raw("mix.norm");
  lm_head = lin("lm_head");
  embed = raw("embed");

  // kernels
  const std::string kdir = mtl::tree_root() + "/engine/kernels/";
  std::string mm = mtl::slurp(kdir + "mm2.metal");
  std::string ops = mtl::slurp(kdir + "ops.metal");
  std::string at = mtl::slurp(kdir + "attn.metal");
  std::map<std::string, std::string> b16 = {};
  if (getenv("OPUS_PF")) b16["PF"] = getenv("OPUS_PF");
  auto with = [&](std::map<std::string, std::string> m, std::map<std::string, std::string> extra) {
    for (auto& [k, v] : extra) m[k] = v;
    return m;
  };
  // an mm2 pipeline, plus its TILES 2 / 4 twins when steps can exceed 16 rows (not for gathered tiles)
  auto mm2pipe = [&](std::map<std::string, std::string> d) {
    id<MTLComputePipelineState> p = C.pipe(mm, "mm2", d);
    if (MAXM > 16 && !d.count("GATHER"))
      mm2_tiled[(__bridge void*)p] = {C.pipe(mm, "mm2", with(d, {{"TILES", "2"}})), C.pipe(mm, "mm2", with(d, {{"TILES", "4"}}))};
    return p;
  };
  k_mm_w8g128 = mm2pipe(with(b16, {{"WBITS", "8"}, {"GS", "128"}}));
  k_mm_w8g64 = mm2pipe(with(b16, {{"WBITS", "8"}, {"GS", "64"}}));
  k_mm_w4g128 = mm2pipe(with(b16, {{"WBITS", "4"}, {"GS", "128"}}));
  k_mm_w4g128_grp = mm2pipe(with(b16, {{"WBITS", "4"}, {"GS", "128"}, {"GROUPED", "1"}}));
  k_mm_w4g64_grp = mm2pipe(with(b16, {{"WBITS", "4"}, {"GS", "64"}, {"GROUPED", "1"}}));
  k_mm_w4g64 = mm2pipe(with(b16, {{"WBITS", "4"}, {"GS", "64"}}));
  for (int b4 = 0; b4 < 2; b4++)
    for (int gi = 0; gi < 2; gi++) {
      std::map<std::string, std::string> d = with(b16, {{"WBITS", b4 ? "4" : "8"}, {"GS", gi ? "64" : "128"}});
      k_epi1[b4][gi] = mm2pipe(with(d, {{"EPI", "1"}}));
      k_epi2[b4][gi] = mm2pipe(with(d, {{"EPI", "2"}}));
    }
  use_mmp = !getenv("OPUS_NO_MMP");
  pf_mm = getenv("OPUS_PF_MM") != nullptr;
  red2 = getenv("OPUS_RED2") ? atoi(getenv("OPUS_RED2")) != 0 : red2;
  rmask_ok = !(getenv("OPUS_RMASK") && !atoi(getenv("OPUS_RMASK")));
  {
    auto mbv = [](const char* k, double d) { return size_t((getenv(k) ? atof(getenv(k)) : d) * 1048576) & ~size_t(4095); };
    pf_hc = mbv("OPUS_PF_HC", 2), pf_gdn = mbv("OPUS_PF_GDN", 2), pf_route = mbv("OPUS_PF_ROUTE", 2), pf_comb = mbv("OPUS_PF_COMB", 2);
    if (getenv("OPUS_RIDE") && !atoi(getenv("OPUS_RIDE"))) pf_hc = pf_gdn = pf_route = pf_comb = 0;
  }
  if (getenv("OPUS_PF_MB")) pf_mb = size_t(atof(getenv("OPUS_PF_MB")) * 1048576) & ~size_t(4095);
  if (getenv("OPUS_MMP_MIN")) mmp_min = uint32_t(atoi(getenv("OPUS_MMP_MIN")));
  if (getenv("OPUS_MMP_CH")) mmp_ch = uint32_t(atoi(getenv("OPUS_MMP_CH")));
  for (int b4 = 0; b4 < 2; b4++)
    for (int gi = 0; gi < 2; gi++)
      for (int ep = 0; ep < 3; ep++)
        k_mmp[b4][gi][ep] = C.pipe(mm, "mmp", with(b16, {{"WBITS", b4 ? "4" : "8"}, {"GS", gi ? "64" : "128"}, {"EPI", std::to_string(ep)}, {"MMP_CH", std::to_string(mmp_ch)}}));

  for (int gi = 0; gi < 2; gi++)
    k_epi3[gi] = mm2pipe(with(with(b16, {{"WBITS", "4"}, {"GS", gi ? "64" : "128"}, {"GROUPED", "1"}}), {{"EPI", "3"}}));
  for (int gi = 0; gi < 2; gi++) {
    const auto d = with(b16, {{"WBITS", "4"}, {"GS", gi ? "64" : "128"}, {"GROUPED", "1"}, {"GATHER", "1"}});
    k_epi3_ga[gi] = mm2pipe(with(d, {{"EPI", "3"}}));
    k_grp_ga[gi] = mm2pipe(d);
  }
  k_mm_w4g128_grp_x2 = mm2pipe(with(b16, {{"WBITS", "4"}, {"GS", "128"}, {"GROUPED", "1"}, {"XMODE", "2"}}));
  k_mm_w8g64_x1 = mm2pipe(with(b16, {{"WBITS", "8"}, {"GS", "64"}, {"XMODE", "1"}}));
  hct = getenv("OPUS_HCT") ? atoi(getenv("OPUS_HCT")) : 1024;
  if (getenv("OPUS_PIPEINFO")) {
    for (auto [n, pp] : std::vector<std::pair<const char*, id<MTLComputePipelineState>>>{{"w8g128", k_mm_w8g128}, {"w4g64grp", k_mm_w4g64_grp}, {"epi1w8", k_epi1[0][0]}, {"epi2w8g64", k_epi2[0][1]}, {"epi3g64", k_epi3[1]}})
      fprintf(stderr, "pipe %-10s maxThreads %lu simdWidth %lu\n", n, (unsigned long)pp.maxTotalThreadsPerThreadgroup, (unsigned long)pp.threadExecutionWidth);
  }
  k_hc_pre = C.pipe(ops, "hc_pre", {{"HCT", std::to_string(hct)}});
  k_hc_mid = C.pipe(ops, "hc_mid");
  k_hc_mix = C.pipe(ops, "hc_mix");
  k_gdn = C.pipe(ops, "gdn_step");
  k_embed = C.pipe(ops, "embed");
  k_argmax = C.pipe(ops, "argmax_rows");
  k_route = C.pipe(ops, "moe_route");
  k_plan = C.pipe(ops, "moe_plan");
  k_act = C.pipe(ops, "moe_act");
  k_comb = C.pipe(ops, "moe_combine");
  k_ple_gate = C.pipe(ops, "ple_gate");
  k_ple_conv = C.pipe(ops, "ple_conv");
  k_attn_prep = C.pipe(at, "attn_prep");
  k_attn_blocks = C.pipe(at, "attn_blocks");
  k_qsa = C.pipe(at, "qsa_select");
  k_qsa_scores = C.pipe(at, "qsa_scores");
  attn_mma = !getenv("OPUS_ATTN_SCALAR") && hd == 256 && nh / nkv <= 16;
  if (kv8 && !attn_mma) mtl::die("OPUS_KV8 needs the matrix-unit attention (hd 256, no OPUS_ATTN_SCALAR)");
  k_attn_rows = C.pipe(at, attn_mma ? "attn_rows_mma" : "attn_rows");
  k_attn_merge = C.pipe(at, "attn_merge");
  lane_batch = !(getenv("OPUS_LANE_BATCH") && atoi(getenv("OPUS_LANE_BATCH")) == 0) && attn_mma;
  if (lane_batch) {
    k_gdn_lanes = C.pipe(ops, "gdn_lanes");
    const std::map<std::string, std::string> ld = {{"LANES", "1"}};
    k_attn_prep_l = C.pipe(at, "attn_prep", ld);
    k_attn_blocks_l = C.pipe(at, "attn_blocks", ld);
    k_attn_rows_l = C.pipe(at, "attn_rows_mma", ld);
    k_attn_merge_l = C.pipe(at, "attn_merge", ld);
    lane_attn_tab = C.buf(size_t(nL + 1) * NSLOT * 48);  // per attention layer (the MTP layer last), per lane
    lane_gdn_tab = C.buf(size_t(nL) * NSLOT * 32);
    rowlane = C.buf(MAXM);
    rope_lanes = C.buf(size_t(2) * MAXM * rot * 4);
    rope_blk_lanes = C.buf(size_t(2) * MAXM * rot * 4);
  }
  k_copy = C.pipe(ops, "copy_rows");
  k_rms_frag = C.pipe(ops, "rms_frag");
  k_mtp_fuse = C.pipe(ops, "mtp_fuse");
  k_gdn_pre = C.pipe(ops, "gdn_pre");
  k_gdn_scan = C.pipe(ops, "gdn_scan");
  k_gdn_post = C.pipe(ops, "gdn_post");
  // Per-step scratch (nothing in it outlives one step) is the driver's in a slot built with `share`:
  // steps of different slots are encoded one at a time on the serving thread, never interleaved.
  // Per-conversation state (caches, recurrent state, h_last_row, MTP seed/drafts, gin) stays per slot.
  g_qn = share ? share->g_qn : C.buf(size_t(MP) * nv * dv * 4);
  g_kn = share ? share->g_kn : C.buf(size_t(MP) * nv * dv * 4);
  g_vv = share ? share->g_vv : C.buf(size_t(MP) * nv * dv * 4);
  g_o = share ? share->g_o : C.buf(size_t(MP) * nv * dv * 4);
  g_bd = share ? share->g_bd : C.buf(size_t(MP) * nv * 2 * 4);
  k_route_rows = C.pipe(ops, "moe_route_rows");
  k_plan_tiles = C.pipe(ops, "moe_plan_tiles");
  k_plan_count = C.pipe(ops, "plan_count");
  k_plan_prefix = C.pipe(ops, "plan_prefix");
  k_plan_scatter = C.pipe(ops, "plan_scatter");
  k_zero = C.pipe(ops, "zero_u32");
  k_copy_u32 = C.pipe(ops, "copy_u32");
  plan_cnt = C.buf(512 * 4);
  memset(plan_cnt.contents, 0, 512 * 4);
  plan_first = C.buf(513 * 4);
  k_gather = C.pipe(ops, "moe_gather");
  k_comb_tiles = C.pipe(ops, "moe_combine_tiles");
  k_sample = C.pipe(ops, "sample_rows");
  k_map_ids = C.pipe(ops, "map_ids");

  // scratch
  const uint32_t W = H * HC, M = MP, SL = MAXM * topk;
  maxtiles = MP * topk / 16 + E;
  auto maxsplitN = [&](std::initializer_list<const Lin*> ls) {
    size_t m = 0;
    for (auto* l : ls) m = std::max(m, size_t(l->ksplit) * l->N);
    return m;
  };
  auto maxN = [&](std::initializer_list<const Lin*> ls) {
    size_t m = 0;
    for (auto* l : ls) m = std::max(m, size_t(l->N));
    return m;
  };
  size_t yin_n = 0, ydn_n = 0, ymw_n = 0, yo_n = 0, part_n = 0, gpart = 0;
  for (auto& l : L) {
    yin_n = std::max(yin_n, maxN({&l.in, &l.moe_in}));
    if (l.ple) yin_n = std::max(yin_n, maxN({&l.ple_kv}));
    ydn_n = std::max(ydn_n, maxN({&l.ahc_dn, &l.mhc_dn}));
    ymw_n = std::max(ymw_n, maxN({&l.ahc_up, &l.mhc_up}));
    yo_n = std::max(yo_n, maxN({&l.out, &l.moe_sdown}));
    part_n = std::max(part_n, maxsplitN({&l.in, &l.moe_in, &l.ahc_dn, &l.mhc_dn, &l.ahc_up, &l.mhc_up, &l.out, &l.moe_sdown}));
    // grouped partials hold one decode step: at most min(SL, E) expert slots of MAXM rows each
    gpart = std::max({gpart, std::min<size_t>(SL, E) * MAXM * l.moe_gu.ksplit * l.moe_gu.N,
                      std::min<size_t>(SL, E) * MAXM * l.moe_dn.ksplit * l.moe_dn.N});
    if (l.ple) part_n = std::max(part_n, maxsplitN({&l.ple_kv}));
  }
  ydn_n = std::max(ydn_n, maxN({&mix_dn}));
  ymw_n = std::max(ymw_n, maxN({&mix_up}));
  part_n = std::max(part_n, maxsplitN({&mix_dn, &mix_up, &lm_head}));
  part = share ? share->part : C.buf(std::max<size_t>(size_t(M) * part_n, gpart) * 4 + size_t(maxtiles) * 16 * 2560 * 4 * 5);
  counters = C.buf(size_t(64) << 12);
  memset(counters.contents, 0, counters.length);
  ids = C.buf(M * 4);
  h = share ? share->h : C.buf(size_t(M) * W * 4);
  hn = share ? share->hn : C.buf(size_t(M) * W * 4);
  hnf = share ? share->hnf : C.buf(size_t(M) * W * 2);
  ydn = share ? share->ydn : C.buf(M * ydn_n * 4);
  df = share ? share->df : C.buf(size_t(M) * R * 2);
  inj = share ? share->inj : C.buf(M * HC * 4);
  ymw = share ? share->ymw : C.buf(M * ymw_n * 4);
  x = share ? share->x : C.buf(size_t(M) * H * 4);
  xf = share ? share->xf : C.buf(size_t(M) * H * 2);
  yin = share ? share->yin : C.buf(M * yin_n * 4);
  of = share ? share->of : C.buf(size_t(M) * nv * dv * 2 > size_t(M) * nh * hd * 2 ? size_t(M) * nv * dv * 2 : size_t(M) * nh * hd * 2);
  yo = share ? share->yo : C.buf(M * yo_n * 4);
  topi = C.buf(size_t(MP) * topk * 4);
  topw = C.buf(size_t(MP) * topk * 4);
  sgate = C.buf(M * 4);
  shf = share ? share->shf : C.buf(size_t(M) * I * 2);
  slots = C.buf(SL * 4);
  nslots = C.buf(16);
  slot_of = C.buf(size_t(MP) * topk * 4);
  rmask = C.buf(size_t(E) * 4 * 4);
  // a step's expert slots are distinct experts: at most min(MAXM * topk, E), each MAXM rows
  const size_t ex_rows = std::max<size_t>(std::min<size_t>(SL, E) * MAXM, size_t(maxtiles) * 16);
  ygu = share ? share->ygu : C.buf(ex_rows * L[0].moe_gu.N * 4);
  af = share ? share->af : C.buf(ex_rows * I * 2);
  yd = share ? share->yd : C.buf(ex_rows * L[0].moe_dn.N * 4);
  tile_exp = C.buf(size_t(maxtiles) * 4);
  ntiles = C.buf(16);
  tile_rows = C.buf(size_t(maxtiles) * 16 * 4);
  assign_pos = C.buf(size_t(MP) * topk * 4);
  xg = share ? share->xg : C.buf(size_t(maxtiles) * 16 * H * 2);
  ysh = share ? share->ysh : C.buf(M * yo_n * 4);
  moe_o = share ? share->moe_o : C.buf(size_t(M) * H * 4);
  logits = share ? share->logits : C.buf(size_t(std::max(MAXM, SROWS)) * lm_head.N * 4);
  pf_sink = C.buf(16384);
  amax_i = C.buf(M * 4);
  amax_v = C.buf(M * 4);
  amax_m = C.buf(M * 4);
  draft_m = C.buf(64 * 4);
  rope = C.buf(size_t(2) * M * rot * 4);
  rope_blk = C.buf(size_t(2) * M * rot * 4);
  rope_slot[0] = rope;
  rope_blk_slot[0] = rope_blk;
  for (int i = 1; i < NSLOT; i++) {
    rope_slot[i] = C.buf(size_t(2) * MAXM * rot * 4);
    rope_blk_slot[i] = C.buf(size_t(2) * MAXM * rot * 4);
  }
  draft_out = C.buf(64);
  h_last_row = C.buf(size_t(W) * 4);
  for (int i = 0; i < NSLOT; i++) mtp_ids[i] = C.buf(MP * 4);
  q = share ? share->q : C.buf(size_t(M) * nh * hd * 4);
  attn_part = share ? share->attn_part : C.buf(size_t(M) * nh * ASPLIT * (hd + 2) * 4);
  gate = share ? share->gate : C.buf(size_t(M) * nh * hd * 4);
  iq = share ? share->iq : C.buf(size_t(M) * inh * ihd * 4);
  scores = share ? share->scores : C.buf(size_t(M) * (cap / ratio + 1) * 4);
  sel = C.buf(size_t(M) * (budget / ratio) * 4);
  nsel_trunk = C.buf(M * 4);
  nsel_mtp = C.buf(M * 4);
  nsel = nsel_trunk;
  ple_event = [C.dev newSharedEvent];
  ple_e = C.buf(size_t(M) * H * 2);
  h_tgt = C.buf(size_t(M) * W * 4);
  hsrc = C.buf(size_t(M) * W * 4);
  mtp_h = C.buf(size_t(M) * W * 4);
  seed_h = C.buf(size_t(W) * 4);
  for (int i = 0; i < NSLOT; i++) seed_h_lane[i] = C.buf(size_t(W) * 4);
  erows = share ? share->erows : C.buf(size_t(M) * H * 4);
  pe = share ? share->pe : C.buf(size_t(M) * H * 4);
  ph = share ? share->ph : C.buf(size_t(M) * H * 4);
  ple_gv = share ? share->ple_gv : C.buf(size_t(M) * W * 4);
  ple_gvn = share ? share->ple_gvn : C.buf(size_t(M) * W * 4);
  ngram = share ? share->ngram : new NgramTable(dir + "/ngram.json", H);
  if (C.track) {
    // this conversation's state and scratch: kept wired the same way as the weights
    MTLResidencySetDescriptor* d = [MTLResidencySetDescriptor new];
    d.label = @"opus-state";
    NSError* e = nil;
    id<MTLResidencySet> ss = [C.dev newResidencySetWithDescriptor:d error:&e];
    if (!ss) mtl::die("residency set", e);
    for (id<MTLBuffer> b : C.made) [ss addAllocation:b];
    [ss commit];
    [ss requestResidency];
    [C.q addResidencySet:ss];
    const int64_t iv = (getenv("OPUS_RESIDENCY_MS") ? atoll(getenv("OPUS_RESIDENCY_MS")) : 1000) * int64_t(NSEC_PER_MSEC);
    dispatch_source_t tm = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, dispatch_get_global_queue(QOS_CLASS_UTILITY, 0));
    dispatch_source_set_timer(tm, dispatch_time(DISPATCH_TIME_NOW, iv), uint64_t(iv), uint64_t(iv / 10));
    dispatch_source_set_event_handler(tm, ^{ if (mtl::keep_warm()) [ss requestResidency]; });
    dispatch_resume(tm);
    state_timers.push_back(tm);
    C.track = false;
    C.made.clear();
  }
  reset();
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  fprintf(stderr, "opus: mapped %.1f GB in %zu files, %u layers, cap %u tokens, ready in %.1fs (kernels: %s)\n", total / 1e9,
          files.size(), nL, cap, secs, kdir.c_str());
}

void Model::reset() {
  for (auto& c : ckpts) ckpt_pool.push_back(c.blob);
  ckpts.clear();
  pos = 0;
  mtp_pos = 0;
  mtp_synced = false;
  conv_par = ple_par = 0;
  history.clear();
  lk.clear();
  for (auto& l : L) {
    if (l.S) memset(l.S.contents, 0, l.S.length);
    if (l.convh) memset(l.convh.contents, 0, l.convh.length);
    if (l.plehist) memset(l.plehist.contents, 0, l.plehist.length);
  }
}

// MmArgs is shared with engine/hy4/hy4.mm via engine/mm_args.h (matches engine/kernels/mm2.metal).

void Model::mm(Rec& rc, const char* tag, const Lin& w, id<MTLBuffer> X, id<MTLBuffer> Y, uint32_t M,
               uint32_t slots_max, uint32_t xstride, uint32_t ystride, id<MTLComputePipelineState> pipe,
               uint32_t ldx, float xdiv, bool barrier, int epi, id<MTLBuffer> e0, id<MTLBuffer> e1, id<MTLBuffer> e2,
               uint32_t epiR) {
  if (count_bytes && !slots_max) {
    const double b = double(w.N) * w.K * w.bits / 8 + double(w.N) * (w.K / w.gs) * 2;
    dense_bytes += b;
    bytes_by_tag[tag] += b;
  }
  id<MTLComputeCommandEncoder> e = rc.op(tag, barrier);
  if (use_mmp && !slots_max && M > MAXM && M >= mmp_min && (!pipe || epi) && epi <= 2) {
    // prefill: same bits as mm2 (see mmp in mm2.metal), weights unpacked once per mmp_ch * 16 rows
    MmArgs a{w.N, w.K, M, w.ksplit, 0, 0, xstride, 0, ystride, w.sgs, ldx, xdiv, 0, epiR, H, HC};
    [e setComputePipelineState:k_mmp[w.bits == 4][w.gs == 64][epi]];
    [e setBuffer:w.w offset:w.woff atIndex:0];
    [e setBuffer:w.s offset:w.soff atIndex:1];
    [e setBuffer:X offset:0 atIndex:2];
    [e setBuffer:part offset:0 atIndex:3];
    [e setBuffer:Y offset:0 atIndex:4];
    [e setBytes:&a length:sizeof a atIndex:5];
    [e setBuffer:slots offset:0 atIndex:6];
    [e setBuffer:nslots offset:0 atIndex:7];
    [e setBuffer:counters offset:0 atIndex:8];
    [e setBuffer:(e0 ? e0 : part) offset:0 atIndex:9];
    [e setBuffer:(e1 ? e1 : part) offset:0 atIndex:10];
    [e setBuffer:(e2 ? e2 : part) offset:0 atIndex:11];
    [e dispatchThreadgroups:MTLSizeMake(w.N / 32, (M + 16 * mmp_ch - 1) / (16 * mmp_ch), 1) threadsPerThreadgroup:MTLSizeMake(32 * mmp_ch, 1, 1)];
    return;
  }
  if (epi) {
    const int gi = w.gs == 64 ? 1 : 0;
    pipe = epi == 1 ? k_epi1[w.bits == 4][gi] : epi == 2 ? k_epi2[w.bits == 4][gi] : gather_mode ? k_epi3_ga[gi] : k_epi3[gi];
  } else if (gather_mode && slots_max && !pipe) {
    pipe = k_grp_ga[w.gs == 64];
  }
  id<MTLComputePipelineState> p =
      pipe ? pipe
      : slots_max ? (w.gs == 64 ? k_mm_w4g64_grp : k_mm_w4g128_grp)
      : w.bits == 4 ? (w.gs == 64 ? k_mm_w4g64 : k_mm_w4g128)
                    : (w.gs == 64 ? k_mm_w8g64 : k_mm_w8g128);
  const uint32_t RB = slots_max ? 1 : (M + 15) / 16;
  // a decode/verify step past 16 rows: tiles of 16 rows share each weight read (TILES variants)
  const uint32_t T = (M > 16 && M <= MAXM && !(red2 && w.ksplit > 1 && !slots_max)) ? tiles_for(M) : 1;
  if (T > 1) {
    auto it = mm2_tiled.find((__bridge void*)p);
    if (it == mm2_tiled.end()) mtl::die("mm: no TILES variant of this mm2 pipeline");
    p = it->second[T == 2 ? 0 : 1];
  }
  const uint32_t NZ = slots_max ? slots_max : (RB + T - 1) / T;  // threadgroups along z
  MmArgs a{w.N, w.K, M, w.ksplit, 0, 0, xstride, w.ksplit * M * w.N, ystride, w.sgs, ldx, xdiv,
           slots_max ? M * w.N : RB * 16 * w.N, epiR, H, HC};
  if (slots_max) {
    a.wstride = uint32_t(size_t(w.N) * w.K * w.bits / 8 / 16);
    a.sstride = uint32_t(size_t(w.N) * (w.K / w.gs));
  }
  const bool two = red2 && w.ksplit > 1 && !slots_max;  // split partials joined by a second dispatch
  if (two) a.red = 1;
  if (slots_max && rmask_on && !gather_mode && !pipe_is_gather(p)) a.rm = 1;
  auto bind = [&](id<MTLComputeCommandEncoder> e) {
    [e setComputePipelineState:p];
    [e setBuffer:w.w offset:w.woff atIndex:0];
    [e setBuffer:w.s offset:w.soff atIndex:1];
    [e setBuffer:X offset:0 atIndex:2];
    [e setBuffer:part offset:0 atIndex:3];
    [e setBuffer:Y offset:0 atIndex:4];
    [e setBytes:&a length:sizeof a atIndex:5];
    [e setBuffer:slots offset:0 atIndex:6];
    [e setBuffer:nslots offset:0 atIndex:7];
    [e setBuffer:counters offset:0 atIndex:8];
    [e setBuffer:(e0 ? e0 : part) offset:0 atIndex:9];
    [e setBuffer:(e1 ? e1 : part) offset:0 atIndex:10];
    [e setBuffer:(e2 ? e2 : part) offset:0 atIndex:11];
    [e setBuffer:tile_rows offset:0 atIndex:12];
    [e setBuffer:pf_sink offset:0 atIndex:13];
    [e setBuffer:pf_sink offset:0 atIndex:14];
    [e setBuffer:rmask offset:0 atIndex:15];
  };
  bind(e);
  // fill threadgroups to 4+ simdgroups with neighbouring fragments when K slices are few
  uint32_t nft = 1;
  while (w.sgs * nft < 4 && (w.N / 32) % (nft * 2) == 0) nft *= 2;
  [e setThreadgroupMemoryLength:(w.sgs > 1 ? size_t(T) * w.sgs * nft * 32 * 16 * 4 : 16) atIndex:0];
  uint32_t pz = 0;
  if (pf_mm && pf.bytes && pf_mb && !slots_max) {
    // ride-along prefetch: extra z layers of this dispatch read the next op's weights
    const size_t take = std::min(pf.bytes, pf_take ? pf_take : pf_mb), tl = size_t(w.N) * w.ksplit * w.sgs;
    a.pf_z0 = NZ;
    a.pf_gx = w.N / 32 / nft;
    a.pf_gy = w.ksplit;
    a.pf_tpt = 32 * w.sgs * nft;
    a.pf_n = uint32_t(take / 16);
    pz = uint32_t((take / 256 + tl - 1) / tl);
    [e setBuffer:pf.b offset:pf.off atIndex:13];
    [e setBytes:&a length:sizeof a atIndex:5];
    pf.off += take;
    pf.bytes -= take;
  } else {
    [e setBuffer:pf_sink offset:0 atIndex:13];
  }
  [e dispatchThreadgroups:MTLSizeMake(w.N / 32 / nft, w.ksplit, NZ + pz)
     threadsPerThreadgroup:MTLSizeMake(32 * w.sgs * nft, 1, 1)];
  if (two) {
    a.red = 2;
    a.pf_z0 = ~0u;
    e = rc.op(tag, true);
    bind(e);
    [e setThreadgroupMemoryLength:16 atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake((w.N / 32 + 7) / 8, 1, RB) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
}

// Hyper-connection read-in: (inject the pending block output into h) -> norm -> down/inject
// -> up (silu fused into its input load) -> mix -> x. inject: 0 none, 1 rows of o, 2 MoE.
void Model::hc_block(Rec& rc, const Lin& dn, const Lin& up, const Raw& norm, uint32_t M, uint32_t inject,
                     id<MTLBuffer> o) {
  HcArgs ha{M, H, HC, inject, prev_dn_N, R, topk};
  id<MTLComputeCommandEncoder> e = rc.op("hc_pre");
  [e setComputePipelineState:k_hc_pre];
  [e setBuffer:h offset:0 atIndex:0];
  [e setBuffer:(o ? o : h) offset:0 atIndex:1];
  [e setBuffer:ydn offset:0 atIndex:2];
  [e setBuffer:norm.b offset:norm.off atIndex:3];
  [e setBuffer:hn offset:0 atIndex:4];
  [e setBuffer:hnf offset:0 atIndex:5];
  [e setBytes:&ha length:sizeof ha atIndex:6];
  [e setBuffer:yd offset:0 atIndex:7];
  [e setBuffer:ysh offset:0 atIndex:8];
  [e setBuffer:topi offset:0 atIndex:9];
  [e setBuffer:topw offset:0 atIndex:10];
  [e setBuffer:slot_of offset:0 atIndex:11];
  [e setBuffer:sgate offset:0 atIndex:12];
  [e dispatchThreadgroups:MTLSizeMake(M, HC, 1 + pf_ride(e, M, HC, hct, pf_take)) threadsPerThreadgroup:MTLSizeMake(hct, 1, 1)];
  if (fused_layout) {
    // down: silu(y / HC) frag in the epilogue; up (streams interleaved): the mix in the epilogue
    mm(rc, "mm.dn", dn, hnf, ydn, M, 0, 0, 0, nil, 0, float(HC), true, 1, df, nil, nil, R);
    prev_dn_N = dn.N;
    mm(rc, "mm.up", up, df, ymw, M, 0, 0, 0, nil, 0, 1.0f, true, 2, xf, hn, x, 0);
    return;
  }
  mm(rc, "mm.dn", dn, hnf, ydn, M);
  prev_dn_N = dn.N;
  MidArgs ma{M, 1, dn.N, R, HC, 0};
  e = rc.op("hc_mid");
  [e setComputePipelineState:k_hc_mid];
  [e setBuffer:ydn offset:0 atIndex:0];
  [e setBuffer:df offset:0 atIndex:1];
  [e setBuffer:inj offset:0 atIndex:2];
  [e setBytes:&ma length:sizeof ma atIndex:3];
  [e dispatchThreads:MTLSizeMake(R, M, 1) threadsPerThreadgroup:MTLSizeMake(R, 1, 1)];
  mm(rc, "mm.up", up, df, ymw, M);
  MixArgs xa{M, 1, H, HC};
  e = rc.op("hc_mix");
  [e setComputePipelineState:k_hc_mix];
  [e setBuffer:ymw offset:0 atIndex:0];
  [e setBuffer:hn offset:0 atIndex:1];
  [e setBuffer:x offset:0 atIndex:2];
  [e setBuffer:xf offset:0 atIndex:3];
  [e setBytes:&xa length:sizeof xa atIndex:4];
  [e dispatchThreads:MTLSizeMake(H, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void rope_rows(double theta, uint32_t rot, const std::vector<uint32_t>& positions, float* out) {
  // out: [M][rot] cos then [M][rot] sin; angles from exact integer positions in f64
  const size_t M = positions.size();
  for (size_t r = 0; r < M; r++)
    for (uint32_t i = 0; i < rot; i++) {
      const double inv = 1.0 / std::pow(theta, double(2 * (i % (rot / 2))) / double(rot));
      const double ang = double(positions[r]) * inv;
      out[r * rot + i] = float(std::cos(ang));
      out[(M + r) * rot + i] = float(std::sin(ang));
    }
}

#include "step.inc"

