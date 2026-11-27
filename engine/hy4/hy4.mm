#include "hy4.h"
#include "../mm_args.h"
#include <cmath>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <mach/mach_time.h>
#include <atomic>
#include <thread>
#include <array>
#include <memory>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <set>

namespace hy4 {

static NSDictionary* jload(const std::string& p) {
  NSData* d = [NSData dataWithContentsOfFile:@(p.c_str())];
  if (!d) mtl::die(("read " + p).c_str());
  NSError* e = nil;
  NSDictionary* j = [NSJSONSerialization JSONObjectWithData:d options:0 error:&e];
  if (!j) mtl::die(("parse " + p).c_str(), e);
  return j;
}

// per-op encoders with timestamps when profiling, else one serial encoder per command buffer
static id<MTLCounterSampleBuffer> new_sample_buffer(id<MTLDevice> dev) {
  MTLCounterSampleBufferDescriptor* d = [MTLCounterSampleBufferDescriptor new];
  for (id<MTLCounterSet> cs in dev.counterSets)
    if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) d.counterSet = cs;
  d.storageMode = MTLStorageModeShared;
  d.sampleCount = 4096;
  NSError* err = nil;
  id<MTLCounterSampleBuffer> sb = [dev newCounterSampleBufferWithDescriptor:d error:&err];
  if (!sb) mtl::die("counter sample buffer", err);
  return sb;
}

// host clock in seconds on the base of MTLCommandBuffer GPUStartTime / GPUEndTime (CACurrentMediaTime's)
static double host_now() {
  static mach_timebase_info_data_t tb;
  if (!tb.denom) mach_timebase_info(&tb);
  return double(mach_absolute_time()) * tb.numer / tb.denom / 1e9;
}

struct Rec {
  id<MTLCommandBuffer> cb;
  id<MTLComputeCommandEncoder> e = nil;
  bool prof = false;
  bool conc = cfg().conc;
  std::vector<id<MTLCounterSampleBuffer>> pool;
  std::vector<std::string> names;
  id<MTLComputeCommandEncoder> op(const char* name, bool barrier = true) {
    // timing ablation: HY4_SKIP=<op>[,<op>...] encodes those ops into a buffer that never runs (values go stale:
    // bench / prof only; ops that steer routing or indexer selection can mislead)
    static const std::string skip = getenv("HY4_SKIP") ? "," + std::string(getenv("HY4_SKIP")) + "," : "";
    if (!skip.empty() && !prof && skip.find("," + std::string(name) + ",") != std::string::npos) {
      static id<MTLCommandBuffer> dcb = nil;
      static id<MTLComputeCommandEncoder> de = nil;
      if (!de) { dcb = [cb.commandQueue commandBuffer]; de = [dcb computeCommandEncoder]; }
      return de;
    }
    if (!prof) {
      if (!e) {
        e = conc ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] : [cb computeCommandEncoder];
      } else if (conc && barrier) {
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
      return e;
    }
    if (e) [e endEncoding];
    const size_t b = names.size() / 2048, i = names.size() % 2048;
    while (pool.size() <= b) pool.push_back(new_sample_buffer(cb.device));
    MTLComputePassDescriptor* d = [MTLComputePassDescriptor computePassDescriptor];
    d.sampleBufferAttachments[0].sampleBuffer = pool[b];
    d.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2 * i;
    d.sampleBufferAttachments[0].endOfEncoderSampleIndex = 2 * i + 1;
    e = [cb computeCommandEncoderWithDescriptor:d];
    names.push_back(name);
    return e;
  }
  void resolve(std::map<std::string, double>& out) {
    for (size_t b = 0; b * 2048 < names.size(); b++) {
      const size_t n = std::min<size_t>(2048, names.size() - b * 2048);
      NSData* d = [pool[b] resolveCounterRange:NSMakeRange(0, 2 * n)];
      const MTLCounterResultTimestamp* ts = (const MTLCounterResultTimestamp*)d.bytes;
      for (size_t i = 0; i < n; i++) out[names[b * 2048 + i]] += (ts[2 * i + 1].timestamp - ts[2 * i].timestamp) / 1e6;
    }
  }
};

// Per-tensor overrides {N, K, gs, grouped, sgs, ksplit}; HY4_SPLITS="N:K:gs:grouped:sgs:ksplit,..." adds
// (and wins over) rows of the table. HY4_SPLIT_TAB defaults to auto (on for Ultra/Max, off for Base/Pro);
// 0/1 overrides. Fixed per tensor, never by M.
static std::vector<std::array<uint32_t, 6>> split_rows(bool use_split_tab) {
  std::vector<std::array<uint32_t, 6>> t;
  if (!cfg().splits.empty()) {
    const std::string& s = cfg().splits;
    for (size_t p = 0; p < s.size();) {
      size_t q = s.find(',', p);
      if (q == std::string::npos) q = s.size();
      std::array<uint32_t, 6> r{};
      if (sscanf(s.substr(p, q - p).c_str(), "%u:%u:%u:%u:%u:%u", &r[0], &r[1], &r[2], &r[3], &r[4], &r[5]) == 6) t.push_back(r);
      p = q + 1;
    }
  }
  if (use_split_tab)
    for (auto& r : std::vector<std::array<uint32_t, 6>>{
             // measured end to end (bench, 3k context, 2026-11-18, M5 Ultra): moe.in, moe.out, routed down;
             // the three together 22.69/32.08 -> 21.72/31.62 ms at 1/5 rows. Other dense shapes moved < 0.2 ms.
             {5120, 3072, 128, 0, 1, 6},
             {3072, 3072, 128, 0, 1, 8},
             {1536, 1536, 64, 1, 1, 2},
             // measured 2026-11-20 on M5 Ultra (12k context, pack-g64-gptq-v3): attn.o, attn.in, moe.gu;
             // 1/5/8/16 rows 20.30/29.88/35.40/47.51 -> 20.09/29.13/35.01/47.15 ms (ksplit=1 keeps moe.gu epi3_regs)
             {3072, 16384, 128, 0, 1, 8},
             {3072, 8192, 128, 0, 1, 8},
             {1760, 3072, 128, 0, 1, 8},
             {1600, 3072, 128, 0, 1, 8},
             {3072, 1536, 64, 1, 2, 1},
         })
      t.push_back(r);
  return t;
}

static void pick_split(uint32_t N, uint32_t K, uint32_t gs, uint32_t slots,
                       const std::vector<std::array<uint32_t, 6>>& tab, uint32_t& sgs, uint32_t& ksplit) {
  const uint32_t G = K / gs, nf = N / 32 * slots;
  for (auto& r : tab)
    if (r[0] == N && r[1] == K && r[2] == gs && (r[3] != 0) == (slots > 1) && G % (r[4] * r[5]) == 0 && r[5] <= 16) {
      sgs = r[4]; ksplit = r[5];
      return;
    }
  uint32_t s_tot = G;
  for (uint32_t s = 1; s <= G; s++)
    if (G % s == 0 && nf * s >= 1024) { s_tot = s; break; }
  for (;;) {
    sgs = 1;
    for (uint32_t c = 8; c >= 2; c--)
      if (s_tot % c == 0) { sgs = c; break; }
    ksplit = s_tot / sgs;
    if (ksplit <= 16) break;
    uint32_t nx = 1;
    for (uint32_t s = s_tot - 1; s >= 1; s--)
      if (G % s == 0) { nx = s; break; }
    s_tot = nx;
  }
}

id<MTLComputePipelineState> Model::pipe(const std::string& k) const {
  auto it = P.find(k);
  if (it == P.end()) mtl::die(("no pipeline " + k).c_str());
  return it->second;
}

Model::Model(const std::string& pack, uint32_t capacity) : dir(pack), cap(capacity) {
  auto t0 = std::chrono::steady_clock::now();
  NSDictionary* man = jload(dir + "/manifest.json");
  {
    const std::string mb = mtl::slurp((dir + "/manifest.json").c_str());
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : mb) { h ^= c; h *= 1099511628211ull; }
    fingerprint = h;
  }
  NSDictionary* cfg = man[@"config"];
  H = [cfg[@"hidden_size"] unsignedIntValue];
  HC = [cfg[@"hc_mult"] unsignedIntValue];
  nL = [cfg[@"num_hidden_layers"] unsignedIntValue];
  vocab = [cfg[@"vocab_size"] unsignedIntValue];
  E = [cfg[@"n_routed_experts"] unsignedIntValue];
  topk = [cfg[@"num_experts_per_tok"] unsignedIntValue];
  I = [cfg[@"moe_intermediate_size"] unsignedIntValue];
  LAT = [cfg[@"routed_expert_hidden_size"] unsignedIntValue];
  window = [cfg[@"sliding_window"] unsignedIntValue];
  ratio = [cfg[@"compressed_indexer_compress_ratio"] unsignedIntValue];
  gtopk = [cfg[@"compressed_indexer_group_topk"] unsignedIntValue];
  eos = [cfg[@"eos_token_id"] unsignedIntValue];
  pad_id = [cfg[@"engram_pad_id"] unsignedIntValue];
  eps = [cfg[@"rms_norm_eps"] floatValue];
  hc_eps = [cfg[@"hc_eps"] floatValue];
  hc_mag = [cfg[@"hc_magnitude"] floatValue];
  rscale = [cfg[@"routed_scaling_factor"] floatValue];
  swiglu = [cfg[@"swiglu_limit"] floatValue];
  moe_dn_had = [man[@"layout"][@"moe_dn_had"] unsignedIntValue];
  if (moe_dn_had != 0 && moe_dn_had != 128) mtl::die("unsupported moe_dn_had (expected 0 or 128)");
  NSDictionary* rp = cfg[@"rope_parameters"];
  base_swa = [rp[@"sliding_attention"][@"rope_theta"] doubleValue];
  base_dsa = [rp[@"deepseek_sparse_attention"][@"rope_theta"] doubleValue];
  for (uint32_t j = 0; j < 32; j++) {
    inv_swa[j] = 1.0 / std::pow(base_swa, double(2 * j) / 64.0);
    inv_dsa[j] = 1.0 / std::pow(base_dsa, double(2 * j) / 64.0);
  }
  if (cap % 128) mtl::die("capacity must be a multiple of 128");

  size_t total = 0;
  id<MTLResidencySet> rs;
  {
    MTLResidencySetDescriptor* d = [MTLResidencySetDescriptor new];
    d.label = @"hy4-weights";
    NSError* e = nil;
    rs = [C.dev newResidencySetWithDescriptor:d error:&e];
    if (!rs) mtl::die("residency set", e);
  }
  // Identify files referenced ONLY by vision tensors (V.* / V{i}.*) — and optionally DFly2 tensors when
  // HY4_DFLY_UNWIRED=1 — so they stay mmap'd in unified page cache without pushing permanent wired residency.
  const bool is_moe_dfly = man[@"dfly"] && man[@"dfly"][@"mlp_type"] && [man[@"dfly"][@"mlp_type"] isEqualToString:@"moe"];
  const bool dfly_unwired = hy4::cfg().dfly_unwired >= 0 ? (hy4::cfg().dfly_unwired != 0) : is_moe_dfly;
  std::set<uint32_t> text_files;
  NSDictionary* ts_map = man[@"tensors"];
  for (NSString* k in ts_map) {
    const std::string ks = [k UTF8String];
    const bool is_vis = ks.size() >= 2 && ks[0] == 'V' && (ks[1] == '.' || (ks[1] >= '0' && ks[1] <= '9'));
    if (is_vis) continue;
    if (dfly_unwired) {
      const bool is_df = (ks.size() >= 3 && ks[0] == 'D' && ks[1] == 'F' && ks[2] == '.') ||
                         (ks.size() >= 2 && ks[0] == 'D' && ks[1] >= '0' && ks[1] <= '9') ||
                         ks == "lm_head_w4";
      if (is_df) continue;
    }
    NSDictionary* td = ts_map[k];
    if (td[@"file"]) text_files.insert([td[@"file"] unsignedIntValue]);
    if (td[@"sfile"]) text_files.insert([td[@"sfile"] unsignedIntValue]);
  }
  uint32_t fi = 0;
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
    if (text_files.empty() || text_files.count(fi)) {
      [rs addAllocation:b];
      total += len;
    }
    files.push_back(b);
    fi++;
  }
  [rs commit];
  [rs requestResidency];
  [C.q addResidencySet:rs];

  NSString* dname = [C.dev name];
  const bool big_gpu = dname && ([dname containsString:@"Ultra"] || [dname containsString:@"Max"]);
  const bool use_split_tab = hy4::cfg().split_tab < 0 ? big_gpu : (hy4::cfg().split_tab != 0);
  const auto stab = split_rows(use_split_tab);

  NSDictionary* T = man[@"tensors"];
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
    pick_split(l.N, l.K, l.gs, slots, stab, l.sgs, l.ksplit);
    if (hy4::cfg().split_dump) {
      static std::set<std::string> seen;
      char b[160];
      snprintf(b, sizeof b, "%u:%u:%u:%u", l.N, l.K, l.gs, slots > 1 ? 1u : 0u);
      if (seen.insert(b).second) fprintf(stderr, "split %-16s %s -> sgs %u ksplit %u (slots %u)\n", name.c_str(), b, l.sgs, l.ksplit, slots);
    }
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
  NSArray* mt = cfg[@"mlp_layer_types"];
  NSDictionary* eng = man[@"engram"];
  L.resize(nL);
  uint32_t eslot = 0;
  for (uint32_t i = 0; i < nL; i++) {
    Layer& l = L[i];
    const std::string d = "L" + std::to_string(i);
    l.sparse = ![lt[i] isEqualToString:@"sliding_attention"];
    l.dense = [mt[i] isEqualToString:@"dense"];
    l.ahc_fn = raw(d + ".ahc.fn"); l.ahc_base = raw(d + ".ahc.base"); l.ahc_scale = raw(d + ".ahc.scale");
    l.mhc_fn = raw(d + ".mhc.fn"); l.mhc_base = raw(d + ".mhc.base"); l.mhc_scale = raw(d + ".mhc.scale");
    l.attn_norm = raw(d + ".attn_norm");
    l.ffn_norm = raw(d + ".ffn_norm");
    l.in = lin(d + ".attn.in");
    l.gate = lin(d + ".attn.gate");
    l.qb = lin(d + ".attn.qb");
    l.o = lin(d + ".attn.o");
    l.nh = l.gate.N / 256;
    l.kb = lin(d + ".attn.kb", l.nh);
    l.vb = lin(d + ".attn.vb", l.nh);
    l.qa_norm = raw(d + ".attn.qa_norm");
    l.kva_norm = raw(d + ".attn.kva_norm");
    l.sinks = raw(d + ".attn.sinks");
    l.post_norm = raw(d + ".attn.post_norm");
    if (l.sparse) {
      l.idx_k_norm = raw(d + ".idx.k_norm");
      l.idx_kvp = raw(d + ".idx.kvp");
      l.idx_kgp = raw(d + ".idx.kgp");
      l.idx_ape = raw(d + ".idx.ape");
      l.idx_key_norm = raw(d + ".idx.key_norm");
      l.kv8 = kv8;
      l.lat = C.buf(size_t(cap) * lat_row(l));
      l.tokring = C.buf(size_t(TOKR) * 128 * 4);
      l.bk = C.buf(size_t(cap / 4) * 128 * 2);
    } else {
      l.lat = C.buf(size_t(RING) * 576 * 2);
    }
    if (l.dense) {
      l.mlp_gu = lin(d + ".mlp.gu");
      l.mlp_dn = lin(d + ".mlp.dn");
      l.mlp_post = raw(d + ".mlp.post_norm");
    } else {
      l.moe_in = lin(d + ".moe.in");
      l.moe_gu = lin(d + ".moe.gu", topk);
      l.moe_dn = lin(d + ".moe.dn", topk);
      l.moe_out = lin(d + ".moe.out");
      l.moe_bias = raw(d + ".moe.bias");
      l.moe_norm = raw(d + ".moe.norm");
    }
    NSDictionary* em = eng[@(d.c_str())];
    if (em) {
      l.eng = true;
      l.eng_slot = eslot++;
      l.eng_kv = lin(d + ".eng.kv");
      l.eng_bias = raw(d + ".eng.bias");
      l.eng_lnorm = raw(d + ".eng.lnorm");
      l.eng_nk = raw(d + ".eng.nk");
      l.eng_nq = raw(d + ".eng.nq");
      l.eng_gamma = raw(d + ".eng.gamma");
      l.eng_hcs = raw(d + ".eng.hcs");
      Raw mp = raw(d + ".eng.map");
      const int32_t* mv = (const int32_t*)((const char*)mp.b.contents + mp.off);
      l.eng_map.assign(mv, mv + mp.bytes / 4);
      for (int k = 0; k < 3; k++) l.mult[k] = [em[@"mult"][k] unsignedLongLongValue];
      for (int k = 0; k < 16; k++) {
        l.prime[k] = [em[@"primes"][k] unsignedLongLongValue];
        l.off[k] = [em[@"offsets"][k] unsignedLongLongValue];
      }
      l.eng_pad = [em[@"pad"] longLongValue];
    }
  }
  n_eng = eslot;
  embed = raw("embed");
  hch_fn = raw("hch.fn"); hch_base = raw("hch.base"); hch_scale = raw("hch.scale");
  norm = raw("norm");
  lm_head = lin("lm_head");
  NSDictionary* dfm = man[@"dfly"];
  if (dfm && !hy4::cfg().no_dfly) {
    has_dfly = true;
    df_moe = dfm[@"mlp_type"] && [dfm[@"mlp_type"] isEqualToString:@"moe"];
    df_has_ac = [dfm[@"attentive_correction"] boolValue];
    df_ac_gated = [dfm[@"ac_block_gated"] boolValue];
    dblock = [dfm[@"block"] unsignedIntValue];
    // For the built-in 5-layer dense DFly2, 8 rather than 6 gives +1-15% on code/quoting.
    // For block-gated AttentiveCorrection (trained for exact block=6), keep trained block unless HY4_DBLOCK overrides.
    if (hy4::cfg().dblock) dblock = std::min<uint32_t>(MAXM, hy4::cfg().dblock);
    else if (!df_has_ac) dblock = std::min<uint32_t>(MAXM, std::max(dblock, 8u));
    mask_id = [dfm[@"mask_token_id"] unsignedIntValue];
    dff = [dfm[@"intermediate"] unsignedIntValue];
    dhci = [dfm[@"correction_intermediate"] unsignedIntValue];
    if (df_moe) {
      d_E = [dfm[@"num_experts"] unsignedIntValue];
      d_topk = [dfm[@"num_experts_per_tok"] unsignedIntValue];
      d_I = dff;
      d_LAT = dfm[@"moe_latent"] ? [dfm[@"moe_latent"] unsignedIntValue] : LAT;
      d_rscale = dfm[@"router_scaling_factor"] ? [dfm[@"router_scaling_factor"] floatValue] : rscale;
    }
    NSArray* tls = dfm[@"target_layers"];
    D.resize([dfm[@"layers"] unsignedIntValue]);
    for (uint32_t i = 0; i < D.size(); i++) {
      DLayer& d = D[i];
      const std::string n = "D" + std::to_string(i);
      d.tl = [tls[i] unsignedIntValue];
      d.moe = df_moe;
      d.in = lin(n + ".attn.in"); d.gate = lin(n + ".attn.gate"); d.qb = lin(n + ".attn.qb"); d.o = lin(n + ".attn.o");
      d.nh = d.gate.N / 256;
      d.kb = lin(n + ".attn.kb", d.nh); d.vb = lin(n + ".attn.vb", d.nh);
      if (d.moe) {
        d.moe_in = lin(n + ".moe.in");
        d.moe_gu = lin(n + ".moe.gu", d_topk);
        d.moe_dn = lin(n + ".moe.dn", d_topk);
        d.moe_out = lin(n + ".moe.out");
        d.moe_bias = raw(n + ".moe.bias");
        d.moe_norm = raw(n + ".moe.norm");
      } else {
        d.gu = lin(n + ".mlp.gu"); d.dn = lin(n + ".mlp.dn");
      }
      d.qa_norm = raw(n + ".attn.qa_norm"); d.kva_norm = raw(n + ".attn.kva_norm"); d.sinks = raw(n + ".attn.sinks");
      d.post_norm = raw(n + ".attn.post_norm"); d.in_norm = raw(n + ".in_norm"); d.post_ln = raw(n + ".post_ln");
      if (d.tl >= nL || !L[d.tl].sparse) mtl::die("dfly: target layer is not a sparse layer of this pack");
    }
    df_eh = lin("DF.eh"); df_hcgu = lin("DF.hc.gu"); df_hcdn = lin("DF.hc.dn"); lm_head_w4 = lin("lm_head_w4");
    df_enorm = raw("DF.enorm"); df_hnorm = raw("DF.hnorm"); df_gate = raw("DF.gate");
    df_hchn = raw("DF.hc.hnorm"); df_hcen = raw("DF.hc.enorm"); df_norm = raw("DF.norm");
    if (df_has_ac) {
      df_ac_src = dfm[@"ac_source_layer"] ? [dfm[@"ac_source_layer"] unsignedIntValue] : uint32_t(D.size() - 1);
      df_ac_nh = D[df_ac_src].nh;
      df_ac_in = lin("DF.ac.in");
      df_ac_qb = lin("DF.ac.qb");
      df_ac_kb = lin("DF.ac.kb", df_ac_nh);
      df_ac_vb = lin("DF.ac.vb", df_ac_nh);
      df_ac_o = lin("DF.ac.o");
      df_ac_bgate = lin("DF.ac.bgate");
      df_ac_qa_norm = raw("DF.ac.qa_norm");
      df_ac_kva_norm = raw("DF.ac.kva_norm");
      df_ac_sinks = raw("DF.ac.sinks");
      df_ac_bsink = raw("DF.ac.bsink");
      df_ac_brel = raw("DF.ac.brel");
      df_ac_bgbias = raw("DF.ac.bgbias");
    }
  }
  NSDictionary* vm = man[@"vision"];
  if (vm) {
    has_vision = true;
    v_nL = [vm[@"layers"] unsignedIntValue];
    v_H = [vm[@"hidden_size"] unsignedIntValue];
    v_I = [vm[@"intermediate_size"] unsignedIntValue];
    v_Ipad = [vm[@"intermediate_padded"] unsignedIntValue];
    v_nh = [vm[@"num_heads"] unsignedIntValue];
    v_hd = [vm[@"head_dim"] unsignedIntValue];
    v_ps = [vm[@"patch_size"] unsignedIntValue];
    v_merge = [vm[@"spatial_merge_size"] unsignedIntValue];
    v_edge = [vm[@"position_edge"] unsignedIntValue];
    v_eps = [vm[@"eps"] floatValue];
    if (vm[@"image_start_token_id"]) img_start_id = [vm[@"image_start_token_id"] unsignedIntValue];
    if (vm[@"image_end_token_id"]) img_end_id = [vm[@"image_end_token_id"] unsignedIntValue];
    if (vm[@"image_token_id"]) img_tok_id = [vm[@"image_token_id"] unsignedIntValue];
    if (vm[@"image_newline_token_id"]) img_nl_id = [vm[@"image_newline_token_id"] unsignedIntValue];
    if (vm[@"video_start_token_id"]) vid_start_id = [vm[@"video_start_token_id"] unsignedIntValue];
    if (vm[@"video_end_token_id"]) vid_end_id = [vm[@"video_end_token_id"] unsignedIntValue];
    if (vm[@"video_token_id"]) vid_tok_id = [vm[@"video_token_id"] unsignedIntValue];
    v_patch_w = lin("V.patch.w"); v_patch_b = raw("V.patch.b"); v_pos_embed = raw("V.pos_embed");
    VL.resize(v_nL);
    for (uint32_t i = 0; i < v_nL; i++) {
      VLayer& v = VL[i];
      const std::string n = "V" + std::to_string(i);
      v.ln1_w = raw(n + ".ln1.w"); v.ln1_b = raw(n + ".ln1.b");
      v.ln2_w = raw(n + ".ln2.w"); v.ln2_b = raw(n + ".ln2.b");
      v.qkv = lin(n + ".attn.qkv"); v.qkv_b = raw(n + ".attn.qkv.b");
      v.o = lin(n + ".attn.o"); v.o_b = raw(n + ".attn.o.b");
      v.fc1 = lin(n + ".mlp.fc1"); v.fc1_b = raw(n + ".mlp.fc1.b");
      v.fc2 = lin(n + ".mlp.fc2"); v.fc2_b = raw(n + ".mlp.fc2.b");
    }
    v_merge_norm = raw("V.merge.norm");
    v_merge_fc1 = lin("V.merge.fc1"); v_merge_fc1_b = raw("V.merge.fc1.b");
    v_merge_fc2 = lin("V.merge.fc2"); v_merge_fc2_b = raw("V.merge.fc2.b");
    v_merge_nl = raw("V.merge.newline");
  }

  // Engram table
  NSDictionary* et = man[@"engram_table"];
  if (n_eng) {
    if (!et) mtl::die("pack has Engram layers but no engram_table");
    std::string p = [et[@"path"] UTF8String];
    if (p[0] != '/') p = dir + "/" + p;
    const size_t off = [et[@"offset"] unsignedLongLongValue];
    eng_rows = [et[@"rows"] unsignedLongLongValue];
    NSString* dt = et[@"dtype"];
    eng_q4 = dt && [dt isEqualToString:@"u4_g32"];
    eng_row_bytes = eng_q4 ? 60u : 96u;
    int fd = open(p.c_str(), O_RDONLY);
    if (fd < 0) mtl::die(("open " + p).c_str());
    const size_t pg = off & ~size_t(16383);
    const size_t len = off - pg + eng_rows * eng_row_bytes;
    void* m = mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, pg);
    close(fd);
    if (m == MAP_FAILED) mtl::die(("mmap " + p).c_str());
    madvise(m, len, MADV_RANDOM);
    eng_tab = (const uint8_t*)m + (off - pg);
    eng_fd = open(p.c_str(), O_RDONLY);
    // rows are random bytes: no readahead around them (HY4_ENG_RDAHEAD=1 keeps the default)
    if (!hy4::cfg().eng_rdahead) fcntl(eng_fd, F_RDAHEAD, 0);
    // Separate F_NOCACHE fd for prefill chunks (M > MAXM) so tens of thousands of random Engram reads
    // don't evict wired/hot model pages from the OS page cache during long prefills.
    eng_fd_pf = open(p.c_str(), O_RDONLY);
    if (!hy4::cfg().eng_rdahead) fcntl(eng_fd_pf, F_RDAHEAD, 0);
    if (hy4::cfg().eng_pf_nocache) fcntl(eng_fd_pf, F_NOCACHE, 1);
    eng_off = off;
    eng_seen.reset(new std::atomic<uint8_t>[(off + eng_rows * eng_row_bytes) / 16384 + 2]());
    if (hy4::cfg().eng_cache_mb > 0) {
      eng_cache_stride = 8 + eng_row_bytes;  // uint32_t seq + uint32_t row + eng_row_bytes (4B-aligned)
      const size_t max_bytes = size_t(hy4::cfg().eng_cache_mb) * 1024 * 1024;
      size_t slots = 1024;
      while (slots * 2 * eng_cache_stride <= max_bytes) slots *= 2;
      eng_cache_mask = slots - 1;
      eng_cache_buf.reset(new uint8_t[slots * eng_cache_stride]());
    }
  }

  // kernels: compile only the mm2/mmp/mmpg (bits, gs) variants actually used by this pack's tensors
  const std::string kdir = mtl::tree_root() + "/engine/kernels/";
  const std::string mm2 = mtl::slurp(kdir + "mm2.metal"), hk = mtl::slurp(kdir + "hy4.metal");
  const std::string lim = std::to_string(int(swiglu));
  const std::string mnf = std::to_string(mmp_nf_), gch = std::to_string(gch_), gnf = std::to_string(grpmmp_nf_);
  std::set<std::pair<uint32_t, uint32_t>> need_dec, need_dec_epi3, need_grp, need_grp_epi3, need_grp_epi3_nc, need_grp_epi5, need_mmp, need_mmp_epi3, need_grpmmp, need_grpmmp_epi3, need_grpmmp_cx_epi5;
  auto reg = [&](const Lin& w, bool grp, bool pf, bool epi3) {
    auto bg = std::make_pair(w.bits, w.gs);
    if (grp) {
      if (epi3) need_grp_epi3.insert(bg); else need_grp.insert(bg);
      if (pf) { if (epi3) need_grpmmp_epi3.insert(bg); else need_grpmmp.insert(bg); }
    } else {
      need_dec.insert(bg);
      if (epi3) need_dec_epi3.insert(bg);
      if (pf) { if (epi3) need_mmp_epi3.insert(bg); else need_mmp.insert(bg); }
    }
  };
  for (const auto& l : L) {
    reg(l.in, false, true, false); reg(l.gate, false, true, false); reg(l.qb, false, true, false); reg(l.o, false, true, false);
    reg(l.kb, true, true, false); reg(l.vb, true, true, false);
    need_grp_epi5.insert({l.vb.bits, l.vb.gs});
    need_grpmmp_cx_epi5.insert({l.vb.bits, l.vb.gs});
    if (l.dense) { reg(l.mlp_gu, false, true, true); reg(l.mlp_dn, false, true, false); }
    else { reg(l.moe_in, false, true, false); reg(l.moe_gu, true, true, true); reg(l.moe_dn, true, true, false); reg(l.moe_out, false, true, false); }
    if (l.eng) reg(l.eng_kv, false, true, false);
  }
  reg(lm_head, false, false, false);
  if (has_dfly) {
    for (const auto& d : D) {
      reg(d.in, false, false, false); reg(d.gate, false, false, false); reg(d.qb, false, false, false); reg(d.o, false, false, false);
      reg(d.kb, true, false, false); reg(d.vb, true, false, false);
      need_grp_epi5.insert({d.vb.bits, d.vb.gs});
      if (d.moe) {
        reg(d.moe_in, false, false, false);
        reg(d.moe_gu, true, false, true);
        reg(d.moe_dn, true, false, false);
        reg(d.moe_out, false, false, false);
      } else {
        reg(d.gu, false, false, false); need_dec_epi3.insert({d.gu.bits, d.gu.gs}); reg(d.dn, false, false, false);
      }
    }
    reg(df_eh, false, false, false);
    reg(df_hcgu, false, false, false); need_dec_epi3.insert({df_hcgu.bits, df_hcgu.gs}); need_grp_epi3_nc.insert({df_hcgu.bits, df_hcgu.gs});
    reg(df_hcdn, false, false, false); reg(df_hcdn, true, false, false);
    if (df_has_ac) {
      reg(df_ac_in, false, false, false); reg(df_ac_in, true, false, false);
      reg(df_ac_qb, false, false, false); reg(df_ac_qb, true, false, false);
      reg(df_ac_kb, true, false, false); reg(df_ac_vb, true, false, false);
      reg(df_ac_o, false, false, false); reg(df_ac_o, true, false, false);
      reg(df_ac_bgate, false, false, false); reg(df_ac_bgate, true, false, false);
    }
    reg(lm_head_w4, false, false, false);
    reg(lm_head_w4, true, false, false);  // GPU-gated draft head uses grouped mm2 with 1 slot
  }
  if (has_vision) {
    reg(v_patch_w, false, true, false);
    for (const auto& v : VL) {
      reg(v.qkv, false, true, false); reg(v.o, false, true, false);
      reg(v.fc1, false, true, false); reg(v.fc2, false, true, false);
    }
    reg(v_merge_fc1, false, true, false); reg(v_merge_fc2, false, true, false);
  }
  auto mk = [&](std::pair<uint32_t, uint32_t> bg) { return "w" + std::to_string(bg.first) + "g" + std::to_string(bg.second); };
  // HY4_PF / HY4_PF_GRP: k-steps of weights in flight per simdgroup in decode / verify mm2 (all, grouped only);
  // load scheduling only, so the bits are unchanged
  auto pfd = [](std::map<std::string, std::string> d, bool grp) {
    const char* e = grp && getenv("HY4_PF_GRP") ? getenv("HY4_PF_GRP") : getenv("HY4_PF");
    if (e) d["PF"] = e;
    return d;
  };
  for (auto bg : need_dec)
    P[mk(bg)] = C.pipe(mm2, "mm2", pfd({{"WBITS", std::to_string(bg.first)}, {"GS", std::to_string(bg.second)}}, false));
  for (auto bg : need_dec_epi3)
    P[mk(bg) + "_epi3"] = C.pipe(mm2, "mm2", pfd({{"WBITS", std::to_string(bg.first)}, {"GS", std::to_string(bg.second)}, {"EPI", "3"}}, false));
  for (auto bg : need_grp)
    P[mk(bg) + "grp"] = C.pipe(mm2, "mm2", pfd({{"WBITS", std::to_string(bg.first)}, {"GS", std::to_string(bg.second)}, {"GROUPED", "1"}}, true));
  for (auto bg : need_grp_epi3)
    P[mk(bg) + "grp_epi3"] = C.pipe(mm2, "mm2", pfd({{"WBITS", std::to_string(bg.first)}, {"GS", std::to_string(bg.second)}, {"GROUPED", "1"}, {"EPI", "3"}, {"SWIGLU_LIMIT", lim}}, true));
  for (auto bg : need_grp_epi3_nc)
    P[mk(bg) + "grp_epi3_nc"] = C.pipe(mm2, "mm2", pfd({{"WBITS", std::to_string(bg.first)}, {"GS", std::to_string(bg.second)}, {"GROUPED", "1"}, {"EPI", "3"}}, true));
  for (auto bg : need_grp_epi5)
    P[mk(bg) + "grp_epi5"] = C.pipe(mm2, "mm2", pfd({{"WBITS", std::to_string(bg.first)}, {"GS", std::to_string(bg.second)}, {"GROUPED", "1"}, {"EPI", "5"}}, true));
  for (auto bg : need_mmp)
    P[mk(bg) + "mmp"] = C.pipe(mm2, "mmp", {{"WBITS", std::to_string(bg.first)}, {"GS", std::to_string(bg.second)}, {"MMP_FAST", "1"}, {"MMP_NF", mnf}});
  for (auto bg : need_mmp_epi3)
    P[mk(bg) + "mmp_epi3"] = C.pipe(mm2, "mmp", {{"WBITS", std::to_string(bg.first)}, {"GS", std::to_string(bg.second)}, {"EPI", "3"}, {"MMP_FAST", "1"}, {"MMP_NF", mnf}});
  for (auto bg : need_grpmmp) {
    P[mk(bg) + "grpmmp"] = C.pipe(mm2, "mmpg", {{"WBITS", std::to_string(bg.first)}, {"GS", std::to_string(bg.second)}, {"GROUPED", "1"}, {"MMP_CH", gch}, {"MMP_FAST", "1"}, {"MMP_NF", gnf}});
    P[mk(bg) + "grpmmp_cx"] = C.pipe(mm2, "mmpg", {{"WBITS", std::to_string(bg.first)}, {"GS", std::to_string(bg.second)}, {"GROUPED", "1"}, {"CONTIG_X", "1"}, {"MMP_CH", "8"}, {"MMP_FAST", "1"}, {"MMP_NF", "2"}});
  }
  for (auto bg : need_grpmmp_cx_epi5)
    P[mk(bg) + "grpmmp_cx_epi5"] = C.pipe(mm2, "mmpg", {{"WBITS", std::to_string(bg.first)}, {"GS", std::to_string(bg.second)}, {"GROUPED", "1"}, {"CONTIG_X", "1"}, {"MMP_CH", "8"}, {"MMP_FAST", "1"}, {"MMP_NF", "2"}, {"EPI", "5"}});
  if (hy4::cfg().mm1 || hy4::cfg().mm1_draft) {
    auto add_mm1 = [&](std::pair<uint32_t, uint32_t> bg, const char* suf, std::map<std::string, std::string> defs) {
      if (bg.first != 4 && bg.first != 8) return;
      defs["WBITS"] = std::to_string(bg.first);
      defs["GS"] = std::to_string(bg.second);
      P[mk(bg) + suf] = C.pipe(mm2, "mm1", defs);
    };
    for (auto bg : need_dec) add_mm1(bg, "mm1", {});
    for (auto bg : need_dec_epi3) add_mm1(bg, "mm1_f3", {{"FUSE", "3"}});
    for (auto bg : need_grp) add_mm1(bg, "mm1grp", {{"GROUPED", "1"}, {"MM1_GATED", "1"}});
    for (auto bg : need_grp_epi3) add_mm1(bg, "mm1grp_f3", {{"GROUPED", "1"}, {"MM1_GATED", "1"}, {"FUSE", "3"}, {"SWIGLU_LIMIT", lim}});
    for (auto bg : need_grp_epi3_nc) add_mm1(bg, "mm1grp_f3_nc", {{"GROUPED", "1"}, {"MM1_GATED", "1"}, {"FUSE", "3"}});
    for (auto bg : need_grp_epi5) add_mm1(bg, "mm1grp_f1", {{"GROUPED", "1"}, {"MM1_GATED", "1"}, {"FUSE", "1"}});
  }
  for (auto bg : need_grpmmp_epi3)
    P[mk(bg) + "grpmmp_epi3"] = C.pipe(mm2, "mmpg", {{"WBITS", std::to_string(bg.first)}, {"GS", std::to_string(bg.second)}, {"GROUPED", "1"}, {"EPI", "3"}, {"SWIGLU_LIMIT", lim}, {"MMP_CH", gch}, {"MMP_FAST", "1"}, {"MMP_NF", gnf}});
  for (const char* k : {"h4_embed", "h4_embed_vis", "h4_hc_1pass", "h4_hc_part", "h4_hc_fin", "h4_rms", "h4_attn_in", "h4_q_prep", "h4_idx_pool",
                        "h4_idx_topk", "h4_attn_merge", "h4_gate", "h4_route", "h4_plan", "h4_combine",
                        "h4_eng_rows", "h4_eng_rows_q4", "h4_eng_mix", "h4_argmax", "h4_plan_lists", "h4_combine_c",
                        "h4_wht128_pf", "h4_wht128_dec",
                        "h4_add", "h4_dsel", "h4_add_frag", "h4_add_frag2", "h4_d_post_add_ln", "h4_draft_hc_in",
                        "h4_ac_block_attn", "h4_yv_to_frag", "h4_ac_gate_nsel",
                        "h4_ac_blk0_in", "h4_ac_in_lanes", "h4_ac_sel_lanes", "h4_draft_corr2",
                        "h4_sample", "h4_pick_p", "h4_idx_score_mma", "h4_dgate"})
    P[k] = C.pipe(hk, k);
  if (has_vision) {
    const std::string vk = mtl::slurp(kdir + "hy4_vit.metal");
    for (const char* k : {"vit_to_frag", "vit_patch_pos", "vit_layernorm_frag", "vit_attn_fwd",
                          "vit_bias_res", "vit_bias_gelu_frag", "vit_merge_gather_frag", "vit_merge_out"})
      P[k] = C.pipe(vk, k);
  }
  P["h4_attn_mma_swa"] = C.pipe(hk, "h4_attn_mma", {{"ATT_SLIDING", "1"}, {"ATT_KV8", "0"}});
  P["h4_attn_mma_dsa"] = C.pipe(hk, "h4_attn_mma", {{"ATT_SLIDING", "0"}, {"ATT_KV8", kv8 ? "1" : "0"}});

  // scratch (M <= MR rows; non-overlapping Engram / Attention / FFN phases alias in scratch_heap)
  if (hy4::cfg().dense_attn) selmax = cap;
  MR = std::max<uint32_t>(MAXM, hy4::cfg().mp / 16 * 16);
  const uint32_t MRp = MR;
  auto B = [&](size_t n) { return C.buf(n); };
  auto A16 = [](size_t n) -> size_t { return ((n ? n : 16) + 16383) & ~size_t(16383); };
  const uint32_t nhmax = 64, Nin = 1024 + 576 + 128 + 32;
  const uint32_t Nmoe = E + LAT + 2 * I;
  const uint32_t smax = MRp * topk;  // compact rows (prefill); decode slots <= 16 * topk
  const size_t crow = std::max<size_t>(size_t(std::min(E, MAXM * topk)) * 16, smax);
  bool need_ygu = false;
  for (auto& l : L) if (!l.dense && l.moe_gu.ksplit > 1) need_ygu = true;
  uint32_t ffd = 0;
  for (auto& l : L) if (l.dense) ffd = std::max(ffd, l.mlp_dn.K);
  const uint32_t max_splits = std::max({hy4::cfg().as_swa, hy4::cfg().as_dsa, as_draft, 1u});

  // Compute aliased phase offsets inside one MTLHeap:
  //   Phase 1 (Engram):    Ye + eng_frag
  //   Phase 2 (Attention): Yg + SlotA(Yq | qlat | yv) + SlotB(scores+mark | ofrag | gfrag) + qn_frag + qr + iq + Yin + qa_frag + sel + part
  //   Phase 3 (FFN / MoE): Ym + lat_frag + mo_frag + act + yd + mlp_frag + Ygu + Ymlp
  const size_t sz_Ye = size_t(MRp) * 5 * H * 4, sz_eng_frag = size_t(MRp) * 1536 * 2;
  const size_t p1_end = A16(sz_Ye) + A16(sz_eng_frag);

  const size_t sz_Yg = size_t(MRp) * nhmax * 256 * 4;
  const size_t sz_Yq = size_t(MRp) * (nhmax * 256 + 2048) * 4;
  const size_t sz_qlat = size_t(nhmax) * MRp * 512 * 4;
  const size_t sz_yv = size_t(nhmax) * MRp * 256 * 4;
  const size_t sz_slotA = std::max({A16(sz_Yq), A16(sz_qlat), A16(sz_yv)});

  const size_t sz_scores = size_t(SB) * (cap / 4) * 4;
  const size_t sz_mark = size_t(SB) * (cap / 128 + 1) * 4;
  const size_t sz_ofrag = size_t(nhmax) * MRp * 512 * 2;
  const size_t sz_gfrag = size_t(MRp) * nhmax * 256 * 2;
  const size_t sz_slotB = std::max({A16(sz_scores) + A16(sz_mark), A16(sz_ofrag), A16(sz_gfrag)});

  const size_t sz_qn_frag = size_t(nhmax) * MRp * 192 * 2;
  const size_t sz_qr = size_t(MRp) * nhmax * 64 * 4;
  const size_t sz_iq = size_t(MRp) * 16 * 128 * 4;
  const size_t sz_Yin = size_t(MRp) * Nin * 4;
  const size_t sz_qa_frag = size_t(MRp) * 1024 * 2;
  const size_t sz_sel = size_t(MRp) * selmax * 4;
  const size_t sz_part = size_t(MAXM) * max_splits * nhmax * 514 * 4;

  const size_t off_Yg = 0;
  const size_t off_slotA = off_Yg + A16(sz_Yg);
  const size_t off_slotB = off_slotA + sz_slotA;
  const size_t off_qn = off_slotB + sz_slotB;
  const size_t off_qr = off_qn + A16(sz_qn_frag);
  const size_t off_iq = off_qr + A16(sz_qr);
  const size_t off_Yin = off_iq + A16(sz_iq);
  const size_t off_qa = off_Yin + A16(sz_Yin);
  const size_t off_sel = off_qa + A16(sz_qa_frag);
  const size_t off_part = off_sel + A16(sz_sel);
  const size_t p2_end = off_part + A16(sz_part);

  const size_t sz_Ym = size_t(MRp) * Nmoe * 4;
  const size_t sz_lat_frag = size_t(MRp) * LAT * 2;
  const size_t sz_mo_frag = size_t(MRp) * (LAT + I) * 2;
  const size_t sz_act = crow * I * 2;
  const size_t sz_yd = crow * LAT * 4;
  const size_t sz_mlp_frag = size_t(MRp) * std::max(ffd, 32u) * 2;
  const size_t sz_Ygu = need_ygu ? size_t(std::min(E, MAXM * topk)) * 16 * 2 * I * 4 : 16;
  const size_t sz_Ymlp = size_t(std::max<uint32_t>(MAXM, 16)) * 2 * std::max({ffd, dff, dhci}) * 4;
  const size_t off_Ym = 0;
  const size_t off_lat_frag = off_Ym + A16(sz_Ym);
  const size_t off_mo_frag = off_lat_frag + A16(sz_lat_frag);
  const size_t off_act = off_mo_frag + A16(sz_mo_frag);
  const size_t off_yd = off_act + A16(sz_act);
  const size_t off_mlp_frag = off_yd + A16(sz_yd);
  const size_t off_Ygu = off_mlp_frag + A16(sz_mlp_frag);
  const size_t off_Ymlp = off_Ygu + A16(sz_Ygu);
  const size_t p3_end = off_Ymlp + A16(sz_Ymlp);

  scratch_heap = C.heap(std::max({p1_end, p2_end, p3_end}));
  auto HBuf = [&](size_t len, size_t off) -> id<MTLBuffer> {
    id<MTLBuffer> b = [scratch_heap newBufferWithLength:(len ? len : 16)
                                                options:(MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked)
                                                 offset:off];
    if (!b) mtl::die("heap sub-alloc");
    return b;
  };

  // Phase-aliased buffers
  Ye = HBuf(sz_Ye, 0);
  eng_frag = HBuf(sz_eng_frag, A16(sz_Ye));
  Yg = HBuf(sz_Yg, off_Yg);
  Yq = HBuf(sz_Yq, off_slotA);
  qlat = HBuf(sz_qlat, off_slotA);
  yv = HBuf(sz_yv, off_slotA);
  scores = HBuf(sz_scores, off_slotB);
  mark = HBuf(sz_mark, off_slotB + A16(sz_scores));
  ofrag = HBuf(sz_ofrag, off_slotB);
  gfrag = HBuf(sz_gfrag, hy4::cfg().fuse_gate ? off_slotA : off_slotB);
  qn_frag = HBuf(sz_qn_frag, off_qn);
  qr = HBuf(sz_qr, off_qr);
  iq = HBuf(sz_iq, off_iq);
  Yin = HBuf(sz_Yin, off_Yin);
  qa_frag = HBuf(sz_qa_frag, off_qa);
  sel = HBuf(sz_sel, off_sel);
  part = HBuf(sz_part, off_part);
  Ym = HBuf(sz_Ym, off_Ym);
  lat_frag = HBuf(sz_lat_frag, off_lat_frag);
  mo_frag = HBuf(sz_mo_frag, off_mo_frag);
  act = HBuf(sz_act, off_act);
  yd = HBuf(sz_yd, off_yd);
  mlp_frag = HBuf(sz_mlp_frag, off_mlp_frag);
  Ygu = HBuf(sz_Ygu, off_Ygu);
  Ymlp = HBuf(sz_Ymlp, off_Ymlp);

  // Persistent step / runtime scratch
  ids = B(MRp * 4);
  h = B(size_t(MRp) * HC * H * 4);
  y = B(size_t(MRp) * H * 4);
  post_a = B(MRp * 4 * 4);
  post_f = B(MRp * 4 * 4);
  xn_frag = B(size_t(MRp) * H * 2);
  xf = B(size_t(MRp) * H * 4);
  Yo = B(size_t(MRp) * H * 4);
  iw = B(MRp * 16 * 4);
  topi = B(MRp * topk * 4);
  topw = B(MRp * topk * 4);
  slots = B(E * 4);
  nslots = B(16);
  slot_of = B(E * 4);
  rmask = B(E * 4 * 4);
  eng_rows_b = B(size_t(std::max(n_eng, 1u)) * MRp * 16 * eng_row_bytes);
  eng_rows_pf = B(size_t(std::max(n_eng, 1u)) * MRp * 16 * eng_row_bytes);
  eng_live_pf = B(MRp * 4);
  eng_live = B(MRp * 4);
  logits = B(size_t(MAXM) * lm_head.N * 4);
  amax = B(MAXM * 4);
  rope_swa = B(MRp * 64 * 4);
  rope_dsa = B(MRp * 64 * 4);
  rope_pool = B(MRp * 64 * 4);
  if (has_vision) vis_idx_b = B(MRp * 4);
  nsel = B(MRp * 4);
  mmpart = B(size_t(64) << 20);
  counters = B(size_t(1) << 22);
  roff = B(E * 4); rcnt = B(E * 4); rlist = B(size_t(MRp) * topk * 4); cidx = B(size_t(MRp) * topk * 4);
  wtile = B((E + size_t(MRp) * topk / 16 + 16) * 4); moe_ind = B(32);
  hcpart = B(size_t(MRp) * HC_NP * 18 * 4);
  head_slots = B(nhmax * 4);
  for (auto& d : D) {
    Layer& t = L[d.tl];
    if (!t.lsel) { t.lsel = B(size_t(MAXM) * selmax * 4); t.lnsel = B(size_t(MAXM) * 4); }
  }
  if (has_dfly) {
    dx = B(size_t(MAXM) * H * 4); dh = B(size_t(MAXM) * H * 4); de = B(size_t(H) * 4);
    dfrag2 = B(size_t(MAXM) * 2 * H * 2);
    dsel = B(size_t(MAXM) * DSELMAX * 4); dnsel = B(MAXM * 4);
    rope_draft = B(MAXM * 64 * 4); dids = B(MAXM * 4);
    dout = B(size_t(MAXM) * MAXM * 4); dprob = B(size_t(MAXM) * MAXM * 4); dpmax = B(size_t(MAXM) * MAXM * 4); dmarg = B(size_t(MAXM) * MAXM * 4);
    for (uint32_t i = 0; i < MAXM; i++) { dgate.push_back(B(16)); ((uint32_t*)dgate.back().contents)[0] = 1; }
    dl_eh_b = B(MAXM * 16);
    dl_row0_b = B(MAXM * 4);
    dl_maxd_b = B(MAXM * 4);
    dl_anchor_b = B(MAXM * 4);
    dl_dsel_b = B(D.size() * MAXM * 32);
    dl_samp_b = B(MAXM * MAXM * 32);
    if (df_has_ac) {
      ac_blk = B(size_t(MAXM) * 16 * 576 * 2);
      ac_bg = B(size_t(MAXM) * 64 * 4);
      ac_nh_gate = B(16);
      dl_ac_rope = B(size_t(MAXM) * MAXM * 64 * 4);
      dl_ac_sel_b = B(size_t(MAXM) * 32);
      dl_ac_rlane = B(size_t(MAXM) * 4);
    }
  }
  for (uint32_t i = 0; i < nhmax; i++) ((uint32_t*)head_slots.contents)[i] = i;
  nslots32 = B(16);
  nslots64 = B(16);
  ((uint32_t*)nslots32.contents)[0] = 32;
  ((uint32_t*)nslots64.contents)[0] = 64;
  {  // slot 0: the buffers just allocated
    Slot s0;
    s0.cap = cap;
    for (auto& l : L) { s0.lat.push_back(l.lat); s0.tokring.push_back(l.tokring); s0.bk.push_back(l.bk); }
    if (has_dfly) {
      s0.df_xf = B(size_t(H) * 4);
      for (size_t di = 0; di < D.size(); di++) {
        s0.df_lsel.push_back(B(size_t(selmax) * 4));
        s0.df_lnsel.push_back(B(16));
      }
    }
    slots_.push_back(s0);
  }
  row_lane_b = B(MAXM * 4);
  row_pos_b = B(MAXM * 4);
  samp_rows_b = B(MAXM * 32);
  {
    MTLResidencySetDescriptor* d = [MTLResidencySetDescriptor new];
    d.label = @"hy4-slots";
    NSError* e = nil;
    slot_rs = [C.dev newResidencySetWithDescriptor:d error:&e];
    if (!slot_rs) mtl::die("slot residency set", e);
    for (auto& l : L) {
      if (l.lat) [slot_rs addAllocation:l.lat];
      if (l.tokring) [slot_rs addAllocation:l.tokring];
      if (l.bk) [slot_rs addAllocation:l.bk];
    }
    if (has_dfly) {
      [slot_rs addAllocation:xf];
      [slot_rs addAllocation:slots_[0].df_xf];
      for (size_t di = 0; di < D.size(); di++) {
        [slot_rs addAllocation:L[D[di].tl].lsel];
        [slot_rs addAllocation:L[D[di].tl].lnsel];
        [slot_rs addAllocation:slots_[0].df_lsel[di]];
        [slot_rs addAllocation:slots_[0].df_lnsel[di]];
      }
    }
    [slot_rs commit];
    [slot_rs requestResidency];
    [C.q addResidencySet:slot_rs];
  }
  {  // macOS drops the wiring soon after the GPU idles; ask again every second while recently used (mtl::keep_warm)
    const int64_t iv = hy4::cfg().residency_ms * int64_t(NSEC_PER_MSEC);
    if (iv > 0) {
      static dispatch_source_t tm;
      id<MTLResidencySet> srs = slot_rs;
      tm = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, dispatch_get_global_queue(QOS_CLASS_UTILITY, 0));
      dispatch_source_set_timer(tm, dispatch_time(DISPATCH_TIME_NOW, iv), uint64_t(iv), uint64_t(iv / 10));
      dispatch_source_set_event_handler(tm, ^{ if (mtl::keep_warm()) { [rs requestResidency]; if (srs) [srs requestResidency]; } });
      dispatch_resume(tm);
    }
  }
  for (uint64_t v : {uint64_t(PREFIX_LAYOUT), uint64_t(kv8), uint64_t(window), uint64_t(RING), uint64_t(TOKR), uint64_t(nL)}) {
    fingerprint ^= v + 0x9E3779B97F4A7C15ull + (fingerprint << 6) + (fingerprint >> 2);
  }
  mapped_bytes = total;
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  fprintf(stderr, "hy4: mapped %.1f GB in %zu files, %u layers (%u Engram), cap %u, state %.2f GB, ready in %.1fs\n",
          total / 1e9, files.size(), nL, n_eng, cap, C.allocated / 1e9, secs);
  mtl::check_wired_limit(mapped_bytes + C.allocated, C.dev);
}

void Model::mm(Rec& rc, const char* tag, const Lin& w, id<MTLBuffer> X, id<MTLBuffer> Y, uint32_t M,
               id<MTLBuffer> sl, id<MTLBuffer> ns, uint32_t nz, uint32_t xstride, uint32_t ystride, bool epi3,
               id<MTLBuffer> e0, uint32_t epiR, uint32_t ldx, bool rm, bool barrier, id<MTLBuffer> e1) {
  const bool grouped = sl != nil;
  const bool epi5 = !epi3 && e1 != nil;
  if (M > MAXM) {
    // prefill: mmp (dense) / mmpg (grouped over row lists; CONTIG_X + CH=8 for per-head kb/vb)
    const bool contig_x = grouped && !rm && ldx == 1 && !epi3;
    std::string k = "w" + std::to_string(w.bits) + "g" + std::to_string(w.gs) +
                    (contig_x ? "grpmmp_cx" : (grouped ? "grpmmp" : "mmp")) +
                    (epi3 ? "_epi3" : (epi5 ? "_epi5" : ""));
    const uint32_t CH = (grouped && !contig_x) ? gch_ : 8, rows = M;
    const uint32_t nf_tg = (grouped && !contig_x) ? grpmmp_nf_ : mmp_nf_;
    const uint32_t gx = (w.N / 32 + nf_tg - 1) / nf_tg;
    MmArgs a{w.N, w.K, M, w.ksplit, 0, 0, xstride, 0, 0, w.sgs, ldx, 1.0f, 0, epiR, epi5 ? epiR : H, epi5 ? 0u : HC};
    a.rm = (grouped && rm) ? 1u : 0u;
    if (grouped) {
      const uint32_t wunit = w.bits == 6 ? 8u : 16u;
      a.wstride = uint32_t(size_t(w.N) * w.K * w.bits / 8 / wunit);
      a.sstride = uint32_t(size_t(w.N) * (w.K / w.gs));
    }
    id<MTLComputeCommandEncoder> e = rc.op(tag, barrier);
    [e setComputePipelineState:pipe(k)];
    [e setBuffer:w.w offset:w.woff atIndex:0];
    [e setBuffer:w.s offset:w.soff atIndex:1];
    [e setBuffer:X offset:0 atIndex:2];
    [e setBuffer:mmpart offset:0 atIndex:3];
    [e setBuffer:Y offset:0 atIndex:4];
    [e setBytes:&a length:sizeof a atIndex:5];
    [e setBuffer:(grouped ? sl : slots) offset:0 atIndex:6];
    [e setBuffer:(grouped ? ns : nslots) offset:0 atIndex:7];
    [e setBuffer:counters offset:0 atIndex:8];
    [e setBuffer:(e0 ? e0 : mmpart) offset:0 atIndex:9];
    [e setBuffer:(e1 ? e1 : mmpart) offset:0 atIndex:10];
    [e setBuffer:mmpart offset:0 atIndex:11];
    [e setBuffer:roff offset:0 atIndex:12];
    [e setBuffer:rcnt offset:0 atIndex:13];
    [e setBuffer:rlist offset:0 atIndex:14];
    [e setBuffer:wtile offset:0 atIndex:15];
    if (a.rm) {
      [e dispatchThreadgroupsWithIndirectBuffer:moe_ind indirectBufferOffset:(ldx == 0 ? 0 : 12)
                          threadsPerThreadgroup:MTLSizeMake(32 * CH, 1, 1)];
    } else {
      [e dispatchThreadgroups:MTLSizeMake(gx, (rows + 16 * CH - 1) / (16 * CH), grouped ? nz : 1)
         threadsPerThreadgroup:MTLSizeMake(32 * CH, 1, 1)];
    }
    return;
  }
  const bool is_draft_m1 = (tag[0] == 'd' && tag[1] == '_');
  const bool use_mm1 = (M == 1u) && (w.bits == 4u || w.bits == 8u) &&
                       (is_draft_m1 ? cfg().mm1_draft : cfg().mm1);
  if (use_mm1) {
    std::string k = "w" + std::to_string(w.bits) + "g" + std::to_string(w.gs) + "mm1" + (grouped ? "grp" : "") +
                    (epi3 ? (grouped && ldx == 3 ? "_f3_nc" : "_f3") : (epi5 ? "_f1" : ""));
    const uint32_t G = w.K / w.gs;
    const uint32_t nf_tot = (w.N / 32u) * (grouped ? nz : 1u);
    uint32_t sgs1 = 1u;
    if (nf_tot >= 2048u) sgs1 = 1u;
    else if (nf_tot <= 128u && G % 16u == 0) sgs1 = 16u;
    else if (nf_tot < 256u && G % 8u == 0) sgs1 = 8u;
    else if (nf_tot < 1024u && G % 4u == 0) sgs1 = 4u;
    else if (nf_tot < 2048u && G % 2u == 0) sgs1 = 2u;
    MmArgs a{};
    a.N = w.N;
    a.K = w.K;
    a.M = 1u;
    a.ksplit = 1u;
    if (grouped) {
      a.wstride = uint32_t(size_t(w.N) * w.K * w.bits / 8 / 16u);
      a.sstride = uint32_t(size_t(w.N) * G);
    }
    a.xstride = xstride;
    a.ystride = epi5 ? w.N : (epi3 ? 16u * epiR : ystride);
    a.sgs = sgs1;
    id<MTLComputeCommandEncoder> e = rc.op(tag, barrier);
    [e setComputePipelineState:pipe(k)];
    [e setBuffer:w.w offset:w.woff atIndex:0];
    [e setBuffer:w.s offset:w.soff atIndex:1];
    [e setBuffer:X offset:0 atIndex:2];
    [e setBuffer:((epi3 || epi5) ? e0 : Y) offset:0 atIndex:3];
    [e setBytes:&a length:sizeof a atIndex:4];
    [e setBuffer:(grouped ? sl : slots) offset:0 atIndex:5];
    if (epi5) [e setBuffer:e1 offset:0 atIndex:6];
    if (grouped) [e setBuffer:ns offset:0 atIndex:7];
    [e dispatchThreadgroups:MTLSizeMake(w.N / 32u, 1, grouped ? nz : 1)
      threadsPerThreadgroup:MTLSizeMake(32u * sgs1, 1, 1)];
    return;
  }
  std::string k = "w" + std::to_string(w.bits) + "g" + std::to_string(w.gs) + (grouped ? "grp" : "") +
                  (epi3 ? (grouped && ldx == 3 ? "_epi3_nc" : "_epi3") : (epi5 ? "_epi5" : ""));
  id<MTLComputePipelineState> p = pipe(k);
  const uint32_t RB = grouped ? 1 : (M + 15) / 16;
  const uint32_t NZ = grouped ? nz : RB;
  MmArgs a{w.N, w.K, M, w.ksplit, 0, 0, xstride, w.ksplit * M * w.N, ystride, w.sgs, 0, 1.0f,
           grouped ? M * w.N : RB * 16 * w.N, epiR, epi5 ? epiR : H, epi5 ? 0u : HC};
  a.rm = (grouped && rm) ? 1u : 0u;
  if (grouped) {
    const uint32_t wunit = w.bits == 6 ? 8u : 16u;
    a.wstride = uint32_t(size_t(w.N) * w.K * w.bits / 8 / wunit);
    a.sstride = uint32_t(size_t(w.N) * (w.K / w.gs));
  }
  if (size_t(NZ) * w.ksplit * (grouped ? M : RB * 16) * w.N * 4 > mmpart.length) mtl::die("mm: partials buffer too small");
  id<MTLComputeCommandEncoder> e = rc.op(tag, barrier);
  [e setComputePipelineState:p];
  [e setBuffer:w.w offset:w.woff atIndex:0];
  [e setBuffer:w.s offset:w.soff atIndex:1];
  [e setBuffer:X offset:0 atIndex:2];
  [e setBuffer:mmpart offset:0 atIndex:3];
  [e setBuffer:Y offset:0 atIndex:4];
  [e setBytes:&a length:sizeof a atIndex:5];
  [e setBuffer:(grouped ? sl : slots) offset:0 atIndex:6];
  [e setBuffer:(grouped ? ns : nslots) offset:0 atIndex:7];
  [e setBuffer:counters offset:0 atIndex:8];
  [e setBuffer:(e0 ? e0 : mmpart) offset:0 atIndex:9];
  [e setBuffer:(e1 ? e1 : mmpart) offset:0 atIndex:10];
  [e setBuffer:mmpart offset:0 atIndex:11];
  [e setBuffer:mmpart offset:0 atIndex:12];
  [e setBuffer:counters offset:0 atIndex:13];
  [e setBuffer:counters offset:0 atIndex:14];
  [e setBuffer:(a.rm ? rmask : counters) offset:0 atIndex:15];
  uint32_t nft = 1;
  while (w.sgs * nft < 4 && (w.N / 32) % (nft * 2) == 0) nft *= 2;
  [e setThreadgroupMemoryLength:(w.sgs > 1 ? size_t(w.sgs) * nft * 32 * 16 * 4 : 16) atIndex:0];
  [e dispatchThreadgroups:MTLSizeMake(w.N / 32 / nft, w.ksplit, NZ) threadsPerThreadgroup:MTLSizeMake(32 * w.sgs * nft, 1, 1)];
}

static void rope_rows(const double* inv, const std::vector<uint32_t>& pos, float* out) {
  // [M][64]: cos of pairs 0..31, then sin; pair j turns by pos * base^(-2j/64), exact positions in f64
  for (size_t r = 0; r < pos.size(); r++) {
    const double pr = double(pos[r]);
    for (uint32_t j = 0; j < 32; j++) {
      const double ang = pr * inv[j];
      out[r * 64 + j] = float(std::cos(ang));
      out[r * 64 + 32 + j] = float(std::sin(ang));
    }
  }
}

// Interleaved 3D MRoPE (mrope_section = [12, 10, 10]): pairs j < 30 interleave (T, H, W) at stride 3,
// pairs 30..31 use T. When t == h == w (text), bit-identical to rope_rows().
static void rope_rows_3d(const double* inv, const std::vector<Pos3D>& pos, float* out) {
  for (size_t r = 0; r < pos.size(); r++) {
    const Pos3D& p3 = pos[r];
    for (uint32_t j = 0; j < 32; j++) {
      uint32_t p = p3.t;
      if (j < 30) {
        if (j % 3 == 1) p = p3.h;
        else if (j % 3 == 2) p = p3.w;
      }
      const double ang = double(p) * inv[j];
      out[r * 64 + j] = float(std::cos(ang));
      out[r * 64 + 32 + j] = float(std::sin(ang));
    }
  }
}

// Persistent blocking-I/O pool for the Engram row reads of decode / verify steps: every read of a step in
// flight at once (cold rows come from SSD; GCD's pool runs about one thread per core, so ~100 reads took
// several latency waves). The caller works too and returns when every index is done.
struct IoPool {
  struct Batch {
    std::function<void(size_t)> fn;
    size_t n = 0;
    std::atomic<size_t> next{0}, fin{0};
  };
  std::vector<std::thread> ws;
  std::mutex mu;
  std::condition_variable cv, dn;
  std::shared_ptr<Batch> cur;  // a late worker holds its own run's batch, whose indices are spent
  uint64_t gen = 0;
  explicit IoPool(size_t k) {
    for (size_t t = 0; t < k; t++)
      ws.emplace_back([this] {
        uint64_t seen = 0;
        for (;;) {
          std::shared_ptr<Batch> b;
          {
            std::unique_lock<std::mutex> g(mu);
            cv.wait(g, [&] { return gen != seen; });
            seen = gen; b = cur;
          }
          work(*b);
        }
      });
  }
  void work(Batch& b) {
    for (size_t i; (i = b.next.fetch_add(1)) < b.n;) {
      b.fn(i);
      if (b.fin.fetch_add(1) + 1 == b.n) { std::lock_guard<std::mutex> g(mu); dn.notify_all(); }
    }
  }
  void run(size_t n, std::function<void(size_t)> f) {
    auto b = std::make_shared<Batch>();
    b->fn = std::move(f); b->n = n;
    { std::lock_guard<std::mutex> g(mu); cur = b; gen++; }
    cv.notify_all();
    work(*b);
    std::unique_lock<std::mutex> g(mu);
    dn.wait(g, [&] { return b->fin.load() >= n; });
  }
};

static inline size_t eng_cache_bucket(uint64_t row, size_t mask) {
  uint64_t h = row * 0x9e3779b97f4a7c15ull;
  h ^= h >> 32;
  return size_t(h) & mask;
}

bool Model::eng_cache_get(uint64_t row, uint8_t* dst) const {
  if (!eng_cache_mask) return false;
  const uint8_t* slot = eng_cache_buf.get() + eng_cache_bucket(row, eng_cache_mask) * eng_cache_stride;
  auto* seq = reinterpret_cast<const std::atomic<uint32_t>*>(slot);
  const uint32_t s1 = seq->load(std::memory_order_acquire);
  if ((s1 & 1u) || s1 == 0) return false;
  uint32_t tag;
  memcpy(&tag, slot + 4, 4);
  if (tag != uint32_t(row)) return false;
  memcpy(dst, slot + 8, eng_row_bytes);
  std::atomic_thread_fence(std::memory_order_acquire);
  return seq->load(std::memory_order_relaxed) == s1;
}

void Model::eng_cache_put(uint64_t row, const uint8_t* src) {
  if (!eng_cache_mask) return;
  uint8_t* slot = eng_cache_buf.get() + eng_cache_bucket(row, eng_cache_mask) * eng_cache_stride;
  auto* seq = reinterpret_cast<std::atomic<uint32_t>*>(slot);
  uint32_t s = seq->load(std::memory_order_relaxed);
  if (s & 1u) return;
  if (!seq->compare_exchange_strong(s, s + 1, std::memory_order_acquire, std::memory_order_relaxed)) return;
  const uint32_t tag = uint32_t(row);
  memcpy(slot + 4, &tag, 4);
  memcpy(slot + 8, src, eng_row_bytes);
  uint32_t next = s + 2;
  if (next == 0) next = 2;  // reserve 0 for empty slot
  seq->store(next, std::memory_order_release);
}

void Model::engram_read_rows(const std::vector<uint64_t>& rows, uint32_t M, uint8_t* dst, uint32_t dst_r0, bool allow_warm,
                             const uint32_t* dst_rows) {
  // rows[] is [engram layer][row][16], the upload buffer's order. The table sits on SSD and every row is
  // its own page read, so throughput is queue depth: many threads each pread their share.
  const size_t rb = eng_row_bytes;
  auto at = [&](size_t i) {
    const size_t e = i / (size_t(M) * 16), rk = i % (size_t(M) * 16);
    const size_t dr = dst_rows ? size_t(dst_rows[rk / 16]) : (size_t(dst_r0) + rk / 16);
    return dst + (e * MR * 16 + dr * 16 + (rk % 16)) * rb;
  };
  // decode / verify steps: rows on pages read in the last ~8 s are copied now, the rest (idx) go through pread.
  // Not for prefill chunks: tens of thousands of rows copied serially through the mapping (first-touch faults)
  // made 64k prefill 1,147 -> 757 tok/s (2026-11-19); their parallel preads are faster.
  const bool wp = eng_warm && allow_warm;
  const uint8_t cur = uint8_t(1 + (mtl::now_s() / 2) % 255);
  auto pg = [&](size_t i, int end) { return (eng_off + rows[i] * rb + (end ? rb - 1 : 0)) >> 14; };
  auto warm = [&](size_t p) { const uint8_t v = eng_seen[p].load(std::memory_order_relaxed); return v && (cur + 255 - v) % 255 < 4; };
  struct Miss { uint64_t row; uint32_t i; };
  std::vector<Miss> misses;
  misses.reserve(rows.size());
  for (size_t i = 0; i < rows.size(); i++) {
    uint8_t* d = at(i);
    if (eng_cache_get(rows[i], d)) continue;
    if (wp && warm(pg(i, 0)) && warm(pg(i, 1))) {
      memcpy(d, eng_tab + rows[i] * rb, rb);
      eng_cache_put(rows[i], d);
    } else {
      misses.push_back({rows[i], uint32_t(i)});
    }
  }
  const int rfd = (!allow_warm && eng_fd_pf >= 0) ? eng_fd_pf : eng_fd;
  if (eng_warm && allow_warm && !misses.empty()) {
    for (const auto& m : misses) {
      eng_seen[pg(m.i, 0)].store(cur, std::memory_order_relaxed);
      eng_seen[pg(m.i, 1)].store(cur, std::memory_order_relaxed);
    }
  }
  if (misses.empty()) return;

  if (cfg().eng_dedup) {
    // Sort misses by file offset and deduplicate identical row IDs so each unique cold row is pread once.
    std::sort(misses.begin(), misses.end(), [](const Miss& a, const Miss& b) { return a.row < b.row; });
    struct Group { uint64_t row; uint32_t start, count; };
    std::vector<Group> groups;
    groups.reserve(misses.size());
    for (size_t k = 0; k < misses.size();) {
      size_t j = k + 1;
      while (j < misses.size() && misses[j].row == misses[k].row) j++;
      groups.push_back({misses[k].row, uint32_t(k), uint32_t(j - k)});
      k = j;
    }
    const size_t ng = groups.size();
    auto do_group = [&](size_t g) {
      const auto& gr = groups[g];
      uint8_t* d0 = at(misses[gr.start].i);
      if (pread(rfd, d0, rb, off_t(eng_off + gr.row * rb)) != ssize_t(rb)) memcpy(d0, eng_tab + gr.row * rb, rb);
      eng_cache_put(gr.row, d0);
      for (uint32_t k = 1; k < gr.count; k++) memcpy(at(misses[gr.start + k].i), d0, rb);
    };
    const size_t maxth = cfg().io_threads;
    const size_t th = std::min(maxth, std::max<size_t>(1, ng / 16));
    if (th <= 1) { for (size_t g = 0; g < ng; g++) do_group(g); return; }
    if (ng <= 2048) {
      static IoPool& pool = *new IoPool(cfg().io_pool);
      static std::mutex pool_mu;
      std::lock_guard<std::mutex> g(pool_mu);
      pool.run(ng, [&](size_t i) { do_group(i); });
      return;
    }
    std::atomic<size_t> next{0};
    std::vector<std::thread> ws;
    for (size_t t = 0; t < th; t++)
      ws.emplace_back([&] { for (size_t i; (i = next.fetch_add(16)) < ng;) for (size_t k = i; k < std::min(ng, i + 16); k++) do_group(k); });
    for (auto& w : ws) w.join();
    return;
  }

  const size_t n = misses.size();
  auto one = [&](size_t j) {
    const size_t i = misses[j].i;
    uint8_t* d = at(i);
    if (pread(rfd, d, rb, off_t(eng_off + rows[i] * rb)) != ssize_t(rb)) memcpy(d, eng_tab + rows[i] * rb, rb);
    eng_cache_put(rows[i], d);
  };
  const size_t maxth = cfg().io_threads;
  const size_t th = std::min(maxth, std::max<size_t>(1, n / 16));
  if (th <= 1) { for (size_t i = 0; i < n; i++) one(i); return; }
  if (n <= 2048) {
    static IoPool& pool = *new IoPool(cfg().io_pool);
    static std::mutex pool_mu;
    std::lock_guard<std::mutex> g(pool_mu);
    pool.run(n, [&](size_t i) { one(i); });
    return;
  }
  std::atomic<size_t> next{0};
  std::vector<std::thread> ws;
  for (size_t t = 0; t < th; t++)
    ws.emplace_back([&] { for (size_t i; (i = next.fetch_add(16)) < n;) for (size_t k = i; k < std::min(n, i + 16); k++) one(k); });
  for (auto& w : ws) w.join();
}

// Engram row ids on the host (HF hash_ngrams over build_ngram_tokens), then the fp8 rows copied
// from the mapped table into the step's upload buffer [engram layer][row][16][96].
void Model::engram_fetch(const std::vector<uint32_t>& ctx, size_t p0, uint32_t M, uint8_t* dst, uint32_t* live,
                         uint32_t s0, uint32_t s1, uint32_t dst_r0) {
  if (live) for (uint32_t r = 0; r < M; r++) {
    const uint32_t t = ctx[p0 + r];
    live[dst_r0 + r] = (t != pad_id && t != img_tok_id && t != vid_tok_id);
  }
  s1 = std::min(s1, n_eng);
  std::vector<uint64_t> rows;
  rows.reserve(size_t(s1 - s0) * M * 16);
  dst += size_t(s0) * MR * 16 * eng_row_bytes;
  for (auto& l : L) {
    if (!l.eng || l.eng_slot < s0 || l.eng_slot >= s1) continue;
    const auto cid = [&](int64_t t) -> uint64_t {
      const int64_t tsz = int64_t(l.eng_map.size());
      return uint64_t(t >= 0 && t < tsz ? l.eng_map[t] : l.eng_map[pad_id]);
    };
    for (uint32_t r = 0; r < M; r++) {
      const size_t p = p0 + r;
      uint64_t c[3];
      for (int s = 0; s < 3; s++) c[s] = p >= size_t(s) ? cid(ctx[p - s]) : cid(pad_id);
      const uint64_t m0 = c[0] * l.mult[0], m1 = c[1] * l.mult[1], m2 = c[2] * l.mult[2];
      const uint64_t bi = m0 ^ m1, tri = m0 ^ m1 ^ m2;
      for (int k = 0; k < 16; k++) {
        const uint64_t row = (k < 8 ? bi : tri) % l.prime[k] + l.off[k];
        if (row >= eng_rows) mtl::die("engram row out of range");
        rows.push_back(row);
      }
    }
  }
  engram_read_rows(rows, M, dst, dst_r0, M <= MAXM);
}

void Model::prefetch_anchor(uint32_t tok) {
  if (!n_eng) return;
  if (pf_job.valid()) return;
  if (anchor_job.valid()) anchor_job.wait();
  if (anchor_lanes_job.valid()) { anchor_lanes_job.wait(); anchor_lanes_meta.clear(); }
  std::vector<uint32_t> ctx3(3, pad_id);
  if (history.size() >= 2) ctx3[0] = history[history.size() - 2];
  if (history.size() >= 1) ctx3[1] = history[history.size() - 1];
  ctx3[2] = tok;
  anchor_tok = tok;
  anchor_p0 = history.size();
  uint8_t* d = (uint8_t*)eng_rows_pf.contents;
  uint32_t* lv = (uint32_t*)eng_live_pf.contents;
  anchor_job = std::async(std::launch::async, [this, ctx3 = std::move(ctx3), d, lv] {
    engram_fetch(ctx3, 2, 1, d, lv, 0, n_eng, 0);
  });
}

void Model::prefetch_anchors_lanes(const std::vector<AnchorLaneReq>& reqs) {
  if (!n_eng || reqs.empty()) return;
  if (pf_job.valid()) return;
  if (anchor_job.valid()) { anchor_job.wait(); anchor_p0 = ~size_t(0); }
  if (anchor_lanes_job.valid()) anchor_lanes_job.wait();
  anchor_lanes_meta.clear();
  auto ctxs = std::make_shared<std::vector<std::vector<uint32_t>>>();
  ctxs->reserve(reqs.size());
  for (const auto& rq : reqs) {
    if (rq.slot < 0 || rq.slot >= int(slots_.size())) continue;
    const auto& sh = (rq.slot == cur_slot) ? history : slots_[rq.slot].history;
    std::vector<uint32_t> ctx3(3, pad_id);
    if (sh.size() >= 2) ctx3[0] = sh[sh.size() - 2];
    if (sh.size() >= 1) ctx3[1] = sh[sh.size() - 1];
    ctx3[2] = rq.tok;
    anchor_lanes_meta.emplace_back(rq.slot, sh.size(), rq.tok);
    ctxs->push_back(std::move(ctx3));
  }
  if (ctxs->empty()) return;
  uint8_t* d = (uint8_t*)eng_rows_pf.contents;
  uint32_t* lv = (uint32_t*)eng_live_pf.contents;
  anchor_lanes_job = std::async(std::launch::async, [this, ctxs, d, lv] {
    std::vector<const std::vector<uint32_t>*> rh(ctxs->size());
    std::vector<size_t> rp(ctxs->size(), 2);
    for (size_t i = 0; i < ctxs->size(); i++) rh[i] = &(*ctxs)[i];
    engram_fetch_lanes(rh, rp, d, lv, 0, n_eng);
  });
}

void Model::engram_rows(uint32_t M, const uint32_t* toks) {
  if (!n_eng) return;
  const size_t p0 = history.size() - M;
  if (pf_job.valid()) {
    pf_job.wait();
    if (pf_p0 == p0 && pf_toks.size() == M && std::equal(pf_toks.begin(), pf_toks.end(), toks)) {
      std::swap(eng_rows_b, eng_rows_pf);
      std::swap(eng_live, eng_live_pf);
      pf_toks.clear();
      return;
    }
    pf_toks.clear();
  }
  bool have_anchor = false;
  if (anchor_job.valid()) {
    anchor_job.wait();
    if (anchor_p0 == p0 && anchor_tok == toks[0]) {
      uint8_t* dst = (uint8_t*)eng_rows_b.contents;
      const uint8_t* src = (const uint8_t*)eng_rows_pf.contents;
      for (uint32_t e = 0; e < n_eng; e++)
        memcpy(dst + size_t(e) * MR * 16 * eng_row_bytes, src + size_t(e) * MR * 16 * eng_row_bytes, 16 * eng_row_bytes);
      ((uint32_t*)eng_live.contents)[0] = ((const uint32_t*)eng_live_pf.contents)[0];
      have_anchor = true;
    }
    anchor_p0 = ~size_t(0);
  }
  if (have_anchor && M == 1) return;
  const uint32_t r0 = have_anchor ? 1u : 0u;
  const uint32_t nM = M - r0;
  if (n_eng > 1 && M <= MAXM && !cfg().eng_sync) {
    // decode / verify: the first Engram layer's rows now, the later layers' rows behind the GPU's next
    // layers in one parallel SSD wave (eng_rest, waited for before the second Engram layer is encoded)
    engram_fetch(history, p0 + r0, nM, (uint8_t*)eng_rows_b.contents, (uint32_t*)eng_live.contents, 0, 1, r0);
    uint8_t* d = (uint8_t*)eng_rows_b.contents;
    eng_rest = std::async(std::launch::async, [this, p0, r0, nM, d] {
      engram_fetch(history, p0 + r0, nM, d, nullptr, 1, n_eng, r0);
    });
    return;
  }
  engram_fetch(history, p0 + r0, nM, (uint8_t*)eng_rows_b.contents, (uint32_t*)eng_live.contents, 0, n_eng, r0);
}

void Model::truncate(uint32_t n) {
  if (n >= pos) return;
  if (!can_truncate(n)) mtl::die("truncate: past the sliding rings (needs a checkpoint or a re-prefill)");
  pos = n;
  history.resize(n);
  if (pos3d.size() > n) pos3d.resize(n);
  if (pos3d.empty()) mrope_delta = 0;
  else {
    const Pos3D& b = pos3d.back();
    mrope_delta = int32_t(std::max({b.t, b.h, b.w}) + 1u) - int32_t(n);
  }
  while (!ckpts.empty() && ckpts.back().pos > n) ckpts.pop_back();
}

// ring rows of positions [p0, p0 + n) <-> a contiguous blob (two blits when the range wraps)
static void ring_copy(id<MTLBlitCommandEncoder> bl, id<MTLBuffer> ring, uint32_t ringN, size_t row, id<MTLBuffer> blob,
                      size_t boff, uint32_t p0, uint32_t n, bool save) {
  uint32_t done = 0;
  while (done < n) {
    const uint32_t r = (p0 + done) % ringN, k = std::min(n - done, ringN - r);
    if (save) [bl copyFromBuffer:ring sourceOffset:size_t(r) * row toBuffer:blob destinationOffset:boff + size_t(done) * row size:size_t(k) * row];
    else [bl copyFromBuffer:blob sourceOffset:boff + size_t(done) * row toBuffer:ring destinationOffset:size_t(r) * row size:size_t(k) * row];
    done += k;
  }
}

void Model::save_ckpt() {
  if (!ckpts.empty() && ckpts.back().pos == pos) return;
  const uint32_t nw = std::min(pos, window), nt = std::min(pos, 4u);
  size_t bytes = 0;
  for (auto& l : L) bytes += l.sparse ? size_t(4) * 128 * 4 : size_t(window) * 1152;
  Ckpt c{pos, ckpts.size() >= max_ckpts ? ckpts.front().b : C.buf(bytes)};
  if (ckpts.size() >= max_ckpts) ckpts.erase(ckpts.begin());
  id<MTLCommandBuffer> cb = [C.q commandBuffer];
  id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
  size_t off = 0;
  for (auto& l : L) {
    if (l.sparse) { ring_copy(bl, l.tokring, TOKR, 512, c.b, off, pos - nt, nt, true); off += size_t(4) * 512; }
    else { ring_copy(bl, l.lat, RING, 1152, c.b, off, pos - nw, nw, true); off += size_t(window) * 1152; }
  }
  [bl endEncoding];
  [cb commit];
  ckpts.push_back(c);
}

uint32_t Model::resume(const std::vector<uint32_t>& prompt) {
  uint32_t common = 0;
  const uint32_t lim = std::min<uint32_t>(uint32_t(history.size()), uint32_t(prompt.size()) - 1);
  while (common < lim && history[common] == prompt[common]) common++;
  if (can_truncate(common)) { truncate(common); return common; }
  while (!ckpts.empty() && ckpts.back().pos > common) ckpts.pop_back();
  if (ckpts.empty()) { reset(); return 0; }
  const Ckpt c = ckpts.back();
  restore_rings(c);
  pos = c.pos;
  ring_floor = c.pos;
  history.resize(c.pos);
  return c.pos;
}

uint32_t Model::resumable(const std::vector<uint32_t>& prompt) const {
  if (prompt.empty()) return 0;
  uint32_t common = 0;
  const uint32_t lim = std::min<uint32_t>(uint32_t(history.size()), uint32_t(prompt.size()) - 1);
  while (common < lim && history[common] == prompt[common]) common++;
  if (can_truncate(common)) return common;
  for (size_t i = ckpts.size(); i-- > 0;)
    if (ckpts[i].pos <= common) return ckpts[i].pos;
  return 0;
}

void Model::restore_rings(const Ckpt& c) {
  const uint32_t nw = std::min(c.pos, window), nt = std::min(c.pos, 4u);
  id<MTLCommandBuffer> cb = [C.q commandBuffer];
  id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
  size_t off = 0;
  for (auto& l : L) {
    if (l.sparse) { ring_copy(bl, l.tokring, TOKR, 512, c.b, off, c.pos - nt, nt, false); off += size_t(4) * 512; }
    else { ring_copy(bl, l.lat, RING, 1152, c.b, off, c.pos - nw, nw, false); off += size_t(window) * 1152; }
  }
  [bl endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
}

// ---- prefix cache files
struct PrefixHdr {
  uint32_t magic = 0x50345948, layout = Model::PREFIX_LAYOUT;  // 'HY4P'
  uint64_t fp = 0;
  uint32_t pos = 0, nL = 0;
  uint32_t nw = 0, nt = 0;   // sliding-ring rows [pos - nw, pos), indexer-ring rows [pos - nt, pos)
  uint64_t state_bytes = 0;  // everything after the history
  uint32_t floor() const {  // the lowest length the loaded state can be truncated to
    const uint32_t fs = pos > nw ? pos - nw + 512 : 0, ft = pos > nt ? pos - nt + 4 : 0;
    return std::min(pos, std::max(fs, ft));
  }
};

static bool xfer(int fd, void* p, size_t n, off_t off, bool wr) {
  uint8_t* b = (uint8_t*)p;
  while (n) {
    const size_t k = std::min<size_t>(n, size_t(1) << 30);
    const ssize_t r = wr ? pwrite(fd, b, k, off) : pread(fd, b, k, off);
    if (r <= 0) return false;
    b += r; n -= size_t(r); off += r;
  }
  return true;
}

// ring rows of positions [p0, p0 + n) <-> the file at off (two transfers when the range wraps)
static bool ring_xfer(int fd, id<MTLBuffer> ring, uint32_t ringN, size_t row, uint32_t p0, uint32_t n, off_t off, bool wr) {
  uint32_t done = 0;
  while (done < n) {
    const uint32_t r = (p0 + done) % ringN, k = std::min(n - done, ringN - r);
    if (!xfer(fd, (uint8_t*)ring.contents + size_t(r) * row, size_t(k) * row, off + off_t(size_t(done) * row), wr)) return false;
    done += k;
  }
  return true;
}

Model::PrefixSnap Model::snapshot_prefix(int slot) const {
  PrefixSnap s;
  if (slot < 0 || slot >= int(slots_.size())) return s;
  if (slot == cur_slot) {
    s.pos = pos;
    s.ring_floor = ring_floor;
    s.history.assign(history.begin(), history.begin() + std::min<size_t>(pos, history.size()));
  } else {
    const Slot& sl = slots_[slot];
    s.pos = sl.pos;
    s.ring_floor = sl.ring_floor;
    s.history.assign(sl.history.begin(), sl.history.begin() + std::min<size_t>(sl.pos, sl.history.size()));
  }
  s.lat = slots_[slot].lat;
  s.tokring = slots_[slot].tokring;
  s.bk = slots_[slot].bk;
  return s;
}

bool Model::save_prefix_snap(const PrefixSnap& snap, const std::string& path, double* ms, size_t* out_bytes,
                             uint32_t* out_floor) const {
  auto t0 = std::chrono::steady_clock::now();
  const uint32_t sp = snap.pos, rf = snap.ring_floor;
  if (sp == 0 || window != 512 || snap.history.size() < sp) return false;
  // ring rows valid at pos: back to where a live truncation could still reach, and not below ring_floor's
  const uint32_t back = RING - MR;  // pos - (RING - window - MR) - window
  const uint32_t lo_s = std::max<int64_t>(0, std::max<int64_t>(int64_t(sp) - back, rf ? int64_t(rf) - window : 0));
  const uint32_t lo_t = std::max<int64_t>(0, std::max<int64_t>(int64_t(sp) - back, rf ? int64_t(rf) - 4 : 0));
  PrefixHdr h;
  h.fp = fingerprint; h.pos = sp; h.nL = nL; h.nw = sp - lo_s; h.nt = sp - lo_t;
  for (size_t li = 0; li < L.size(); li++)
    h.state_bytes += L[li].sparse ? size_t(h.nt) * 512 + size_t(sp) * lat_row(L[li]) + size_t((sp + 3) / 4) * 256 : size_t(h.nw) * 1152;
  const std::string tmp = path + ".tmp";
  int fd = open(tmp.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0600);
  if (fd < 0) return false;
  fcntl(fd, F_NOCACHE, 1);  // a cache file must not push the Engram table out of the page cache
  off_t off = 0;
  bool ok = xfer(fd, &h, sizeof h, off, true);
  off += sizeof h;
  ok = ok && xfer(fd, const_cast<uint32_t*>(snap.history.data()), size_t(sp) * 4, off, true);
  off += off_t(sp) * 4;
  for (size_t li = 0; li < L.size(); li++) {
    if (!ok) break;
    const Layer& l = L[li];
    if (l.sparse) {
      const size_t a = size_t(sp) * lat_row(l), b = size_t((sp + 3) / 4) * 256;
      ok = ring_xfer(fd, snap.tokring[li], TOKR, 512, lo_t, h.nt, off, true) &&
           xfer(fd, snap.lat[li].contents, a, off + off_t(size_t(h.nt) * 512), true) &&
           xfer(fd, snap.bk[li].contents, b, off + off_t(size_t(h.nt) * 512 + a), true);
      off += off_t(size_t(h.nt) * 512 + a + b);
    } else {
      ok = ring_xfer(fd, snap.lat[li], RING, 1152, lo_s, h.nw, off, true);
      off += off_t(size_t(h.nw) * 1152);
    }
  }
  ok = close(fd) == 0 && ok;
  if (!ok || rename(tmp.c_str(), path.c_str()) != 0) { unlink(tmp.c_str()); return false; }
  if (out_bytes) *out_bytes = sizeof h + size_t(h.pos) * 4 + h.state_bytes;
  if (out_floor) *out_floor = h.floor();
  if (ms) *ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
  return true;
}

bool Model::save_prefix(const std::string& path, double* ms, size_t* out_bytes, uint32_t* out_floor) {
  return save_prefix_snap(snapshot_prefix(cur_slot), path, ms, out_bytes, out_floor);
}

bool Model::prefix_tokens(const std::string& path, uint64_t fp, std::vector<uint32_t>& toks, size_t* bytes, uint32_t* floor) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return false;
  PrefixHdr h;
  bool ok = xfer(fd, &h, sizeof h, 0, false) && h.magic == PrefixHdr{}.magic && h.layout == PREFIX_LAYOUT && h.fp == fp;
  if (ok) {
    toks.resize(h.pos);
    ok = xfer(fd, toks.data(), size_t(h.pos) * 4, sizeof h, false);
    if (bytes) *bytes = sizeof h + size_t(h.pos) * 4 + h.state_bytes;
    if (floor) *floor = h.floor();
  }
  close(fd);
  return ok;
}

bool Model::load_prefix_into(int slot, const std::string& path, PrefixLoad& out) const {
  out = PrefixLoad{};
  if (slot < 0 || slot >= int(slots_.size())) return false;
  const uint32_t scap = slots_[slot].cap;
  auto t0 = std::chrono::steady_clock::now();
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return false;
  fcntl(fd, F_NOCACHE, 1);
  PrefixHdr h;
  if (!xfer(fd, &h, sizeof h, 0, false) || h.magic != PrefixHdr{}.magic || h.layout != PREFIX_LAYOUT || h.fp != fingerprint ||
      h.nL != nL || h.pos == 0 || h.pos + 1 >= scap || h.nw > h.pos || h.nt > h.pos || h.nw > RING || h.nt > TOKR || window != 512) {
    close(fd);
    return false;
  }
  std::vector<uint32_t> toks(h.pos);
  off_t off = sizeof h;
  bool ok = xfer(fd, toks.data(), size_t(h.pos) * 4, off, false);
  off += off_t(h.pos) * 4;
  const Slot& sl = slots_[slot];
  for (size_t li = 0; li < L.size(); li++) {
    if (!ok) break;
    const Layer& l = L[li];
    if (l.sparse) {
      const size_t a = size_t(h.pos) * lat_row(l), b = size_t((h.pos + 3) / 4) * 256;
      ok = ring_xfer(fd, sl.tokring[li], TOKR, 512, h.pos - h.nt, h.nt, off, false) &&
           xfer(fd, sl.lat[li].contents, a, off + off_t(size_t(h.nt) * 512), false) &&
           xfer(fd, sl.bk[li].contents, b, off + off_t(size_t(h.nt) * 512 + a), false);
      off += off_t(size_t(h.nt) * 512 + a + b);
    } else {
      ok = ring_xfer(fd, sl.lat[li], RING, 1152, h.pos - h.nw, h.nw, off, false);
      off += off_t(size_t(h.nw) * 1152);
    }
  }
  close(fd);
  out.ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
  if (!ok) return false;
  out.pos = h.pos;
  out.ring_floor = h.floor();
  out.history = std::move(toks);
  out.ok = true;
  return true;
}

uint32_t Model::apply_prefix_load(int slot, PrefixLoad&& pl) {
  use_slot(slot);
  if (pf_job.valid()) pf_job.wait();
  if (anchor_job.valid()) anchor_job.wait();
  if (anchor_lanes_job.valid()) anchor_lanes_job.wait();
  anchor_p0 = ~size_t(0);
  anchor_lanes_meta.clear();
  pf_toks.clear();
  if (!pl.ok) { reset(); return 0; }  // the slot's caches may be partly overwritten
  history = std::move(pl.history);
  pos = pl.pos;
  ring_floor = pl.ring_floor;
  ckpts.clear();
  slots_[cur_slot].has_df_src = false;
  return pl.pos;
}

uint32_t Model::load_prefix(const std::string& path, double* ms) {
  PrefixLoad pl;
  load_prefix_into(cur_slot, path, pl);
  if (ms) *ms = pl.ms;
  return apply_prefix_load(cur_slot, std::move(pl));
}

int Model::add_slot(uint32_t capacity) {
  if (capacity > slots_[0].cap || capacity % 128) mtl::die("add_slot: capacity must be <= slot 0's and a multiple of 128");
  Slot s;
  s.cap = capacity;
  for (auto& l : L) {
    id<MTLBuffer> la = C.buf(size_t(l.sparse ? capacity : RING) * lat_row(l));
    id<MTLBuffer> tr = l.sparse ? C.buf(size_t(TOKR) * 128 * 4) : nil;
    id<MTLBuffer> bk = l.sparse ? C.buf(size_t(capacity / 4) * 128 * 2) : nil;
    s.lat.push_back(la);
    s.tokring.push_back(tr);
    s.bk.push_back(bk);
    if (slot_rs) {
      if (la) [slot_rs addAllocation:la];
      if (tr) [slot_rs addAllocation:tr];
      if (bk) [slot_rs addAllocation:bk];
    }
  }
  if (has_dfly) {
    s.df_xf = C.buf(size_t(H) * 4);
    if (slot_rs) [slot_rs addAllocation:s.df_xf];
    for (size_t di = 0; di < D.size(); di++) {
      id<MTLBuffer> ds = C.buf(size_t(selmax) * 4);
      id<MTLBuffer> dn = C.buf(16);
      s.df_lsel.push_back(ds);
      s.df_lnsel.push_back(dn);
      if (slot_rs) { [slot_rs addAllocation:ds]; [slot_rs addAllocation:dn]; }
    }
  }
  if (slot_rs) { [slot_rs commit]; [slot_rs requestResidency]; }
  slots_.push_back(s);
  if (!lane_tab.empty()) rebuild_lane_tables();
  return int(slots_.size()) - 1;
}

void Model::save_dfly_src(int slot, uint32_t row) {
  if (!has_dfly || slot < 0 || slot >= int(slots_.size())) return;
  Slot& s = slots_[slot];
  memcpy(s.df_xf.contents, (const float*)xf.contents + size_t(row) * H, size_t(H) * 4);
  for (size_t di = 0; di < D.size(); di++) {
    const Layer& t = L[D[di].tl];
    const uint32_t n = ((const uint32_t*)t.lnsel.contents)[row];
    ((uint32_t*)s.df_lnsel[di].contents)[0] = n;
    if (n) memcpy(s.df_lsel[di].contents, (const uint32_t*)t.lsel.contents + size_t(row) * selmax, size_t(n) * 4);
  }
  s.has_df_src = true;
}

void Model::use_slot(int i) {
  if (i == cur_slot) return;
  if (pf_job.valid()) pf_job.wait();
  if (anchor_job.valid()) anchor_job.wait();
  anchor_p0 = ~size_t(0);
  pf_toks.clear();
  Slot& o = slots_[cur_slot];
  o.pos = pos; o.ring_floor = ring_floor; o.mrope_delta = mrope_delta;
  o.history.swap(history); o.pos3d.swap(pos3d); o.ckpts.swap(ckpts);
  Slot& n = slots_[i];
  for (size_t k = 0; k < L.size(); k++) { L[k].lat = n.lat[k]; L[k].tokring = n.tokring[k]; L[k].bk = n.bk[k]; }
  cap = n.cap; pos = n.pos; ring_floor = n.ring_floor; mrope_delta = n.mrope_delta;
  history.swap(n.history); pos3d.swap(n.pos3d); ckpts.swap(n.ckpts);
  cur_slot = i;
}

uint32_t Model::common_with(int i, const std::vector<uint32_t>& prompt) const {
  const std::vector<uint32_t>& h = i == cur_slot ? history : slots_[i].history;
  uint32_t c = 0;
  while (c < h.size() && c < prompt.size() && h[c] == prompt[c]) c++;
  return c;
}

struct HcArgs { uint32_t M, H, inject, mode; float eps, hc_eps, mag; uint32_t ynorm = 0; };
struct RmsArgs { uint32_t M, D, ldi, off, ldo, want_f, want_frag; float eps; uint32_t fo = 0, fk = 0; };
struct KvArgs { uint32_t M, Nin, p0, ring, sparse, ikoff, iwoff, kvoff; float eps; float iw_scale; uint32_t kv8; };
struct QArgs { uint32_t M, nh, Nq, sparse, rs; };
struct PoolArgs { uint32_t M, p0; float eps; };
struct ScoreArgs { uint32_t M, p0, dense_max, gmax, row0; };
struct TopArgs { uint32_t M, p0, dense_max, gmax, ktop, selmax, row0; };
struct AttnArgs { uint32_t M, nh, p0, sliding, window, ring, splits, selmax; float scale; uint32_t rs; uint32_t kv8; uint32_t direct = 0, rsf = 0; };
struct MergeArgs { uint32_t M, nh, splits, rs; };
struct GateArgs { uint32_t M, nh, rs; };
struct RouteArgs { uint32_t M, E, topk, Nin, lat, I; float scale; };
struct PlanArgs { uint32_t M, topk, E; };
struct CombArgs { uint32_t M, topk, lat, ystride; };
struct WhtArgs { uint32_t M, I; };
struct EngRowArgs { uint32_t M; float eps; };
struct EngMixArgs { uint32_t M, H; float eps; };
struct ArgArgs { uint32_t M, N; };
struct SampArgs { uint32_t M, N, p0, s0, s1; float inv_t; float lmin; float top_p; uint32_t top_k; };
struct ListArgs { uint32_t M, topk, E, tile_rows, gx_gu, gx_dn; };
struct CombCArgs { uint32_t M, topk, lat; };

static void setb(id<MTLComputeCommandEncoder> e, const Raw& r, int i) { [e setBuffer:r.b offset:r.off atIndex:i]; }

void Model::hc_read(Rec& rc, const Raw& fn, const Raw& base, const Raw& scale, const Raw& nw, id<MTLBuffer> postp,
                    id<MTLBuffer> post, uint32_t inject, uint32_t mode, uint32_t M, const char* tag) {
  id<MTLComputeCommandEncoder> e;
  HcArgs a{M, H, inject, mode, eps, hc_eps, hc_mag};
  // a pending sublayer norm (decode): injected as rms(Yo) * w inside hc instead of y
  const Raw* yn = inject ? ynorm_pending : nullptr;
  ynorm_pending = nullptr;
  a.ynorm = yn ? 1u : 0u;
  if (M > MAXM && H <= 4096) {
    e = rc.op(tag);
    [e setComputePipelineState:pipe("h4_hc_1pass")];
    [e setBuffer:h offset:0 atIndex:0];
    [e setBuffer:(yn ? Yo : y) offset:0 atIndex:1];
    [e setBuffer:postp offset:0 atIndex:2];
    setb(e, fn, 3); setb(e, base, 4); setb(e, scale, 5); setb(e, nw, 6);
    [e setBuffer:xn_frag offset:0 atIndex:7];
    [e setBuffer:xf offset:0 atIndex:8];
    [e setBuffer:post offset:0 atIndex:9];
    [e setBytes:&a length:sizeof a atIndex:10];
    if (yn) setb(e, *yn, 11); else [e setBuffer:y offset:0 atIndex:11];
    [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    return;
  }
  const uint32_t np = HC_NP;
  e = rc.op(tag);
  [e setComputePipelineState:pipe("h4_hc_part")];
  [e setBuffer:h offset:0 atIndex:0];
  [e setBuffer:(yn ? Yo : y) offset:0 atIndex:1];
  [e setBuffer:postp offset:0 atIndex:2];
  setb(e, fn, 3);
  [e setBuffer:hcpart offset:0 atIndex:4];
  [e setBytes:&a length:sizeof a atIndex:5];
  if (yn) setb(e, *yn, 6); else [e setBuffer:y offset:0 atIndex:6];
  [e dispatchThreadgroups:MTLSizeMake(np, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  if (mode == 2) return;
  e = rc.op(tag);
  [e setComputePipelineState:pipe("h4_hc_fin")];
  [e setBuffer:h offset:0 atIndex:0];
  [e setBuffer:hcpart offset:0 atIndex:1];
  setb(e, base, 2); setb(e, scale, 3); setb(e, nw, 4);
  [e setBuffer:xn_frag offset:0 atIndex:5];
  [e setBuffer:xf offset:0 atIndex:6];
  [e setBuffer:post offset:0 atIndex:7];
  [e setBytes:&a length:sizeof a atIndex:8];
  [e setBytes:&np length:4 atIndex:9];
  [e dispatchThreadgroups:MTLSizeMake(np, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void Model::ensure_lane_pipes() {
  if (lane_pipes_ready) return;
  const std::string hk = mtl::slurp(mtl::tree_root() + "/engine/kernels/hy4.metal");
  for (const char* k : {"h4_attn_in", "h4_idx_pool", "h4_idx_score_mma", "h4_idx_topk"})
    P[std::string(k) + "_lanes"] = C.pipe(hk, k, {{"LANES", "1"}});
  P["h4_attn_mma_swa_lanes"] = C.pipe(hk, "h4_attn_mma", {{"LANES", "1"}, {"ATT_SLIDING", "1"}, {"ATT_KV8", "0"}});
  P["h4_attn_mma_dsa_lanes"] = C.pipe(hk, "h4_attn_mma", {{"LANES", "1"}, {"ATT_SLIDING", "0"}, {"ATT_KV8", kv8 ? "1" : "0"}});
  P["h4_sample_lanes"] = C.pipe(hk, "h4_sample_lanes");
  if (has_dfly) {
    for (const char* k : {"h4_draft_eh_rms", "h4_draft_eh_add", "h4_dsel_lanes",
                          "h4_draft_corr", "h4_pick_p_lanes", "h4_dgate_lanes"})
      P[k] = C.pipe(hk, k);
  }
  lane_pipes_ready = true;
}

void Model::rebuild_lane_tables() {
  struct LaneStateHost { uint64_t lat, tokring, bk; };
  lane_tab.resize(nL);
  const size_t ns = slots_.size();
  for (uint32_t li = 0; li < nL; li++) {
    lane_tab[li] = C.buf(ns * sizeof(LaneStateHost));
    LaneStateHost* t = (LaneStateHost*)lane_tab[li].contents;
    for (size_t s = 0; s < ns; s++) {
      t[s].lat = slots_[s].lat[li] ? slots_[s].lat[li].gpuAddress : 0;
      t[s].tokring = slots_[s].tokring[li] ? slots_[s].tokring[li].gpuAddress : 0;
      t[s].bk = slots_[s].bk[li] ? slots_[s].bk[li].gpuAddress : 0;
    }
  }
}

void Model::engram_fetch_lanes(const std::vector<const std::vector<uint32_t>*>& rhist, const std::vector<size_t>& rpos,
                               uint8_t* dst, uint32_t* live, uint32_t s0, uint32_t s1, const uint32_t* dst_rows) {
  const uint32_t M = uint32_t(rhist.size());
  if (M == 0) return;
  if (live) for (uint32_t r = 0; r < M; r++) {
    const uint32_t t = (*rhist[r])[rpos[r]];
    live[dst_rows ? dst_rows[r] : r] = (t != pad_id && t != img_tok_id && t != vid_tok_id);
  }
  s1 = std::min(s1, n_eng);
  std::vector<uint64_t> rows;
  rows.reserve(size_t(s1 - s0) * M * 16);
  dst += size_t(s0) * MR * 16 * eng_row_bytes;
  for (auto& l : L) {
    if (!l.eng || l.eng_slot < s0 || l.eng_slot >= s1) continue;
    const auto cid = [&](int64_t t) -> uint64_t {
      const int64_t tsz = int64_t(l.eng_map.size());
      return uint64_t(t >= 0 && t < tsz ? l.eng_map[t] : l.eng_map[pad_id]);
    };
    for (uint32_t r = 0; r < M; r++) {
      const auto& ctx = *rhist[r];
      const size_t p = rpos[r];
      uint64_t c[3];
      for (int s = 0; s < 3; s++) c[s] = p >= size_t(s) ? cid(ctx[p - s]) : cid(pad_id);
      const uint64_t m0 = c[0] * l.mult[0], m1 = c[1] * l.mult[1], m2 = c[2] * l.mult[2];
      const uint64_t bi = m0 ^ m1, tri = m0 ^ m1 ^ m2;
      for (int k = 0; k < 16; k++) {
        const uint64_t row = (k < 8 ? bi : tri) % l.prime[k] + l.off[k];
        if (row >= eng_rows) mtl::die("engram row out of range");
        rows.push_back(row);
      }
    }
  }
  engram_read_rows(rows, M, dst, 0, true, dst_rows);
}

void Model::encode_layer(Rec& rc, Layer& l, uint32_t li, uint32_t M, uint32_t p0, bool first, bool lanes, uint32_t max_pos) {
  const float feps = 1.1920929e-07f;
  id<MTLComputeCommandEncoder> e;
  const bool pf = M > MAXM;                          // prefill chunk: mmp / mmpg, one key split
  const uint32_t rsf = pf ? (M + 15) / 16 * 16 : 16;  // per-head frag rows
  const uint32_t rsy = pf ? M : 16;                  // per-head f32 output rows
  auto hc = [&](const Raw& fn, const Raw& base, const Raw& scale, const Raw& nw, id<MTLBuffer> postp, id<MTLBuffer> post,
                uint32_t inject, uint32_t mode, const char* tag) { hc_read(rc, fn, base, scale, nw, postp, post, inject, mode, M, tag); };
  auto rms = [&](id<MTLBuffer> X, const Raw& w, id<MTLBuffer> Yf, uint32_t D, uint32_t ldi, uint32_t off, const char* tag) {
    if (!pf && X == Yo && Yf == y && D == H && ldi == H && off == 0) { ynorm_pending = &w; return; }
    RmsArgs a{M, D, ldi, off, D, 1, 0, eps};
    e = rc.op(tag);
    [e setComputePipelineState:pipe("h4_rms")];
    [e setBuffer:X offset:0 atIndex:0];
    setb(e, w, 1);
    [e setBuffer:Yf offset:0 atIndex:2];
    [e setBuffer:xn_frag offset:0 atIndex:3];
    [e setBytes:&a length:sizeof a atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  };

  // ---- Engram: the previous layer's FFN output goes into the streams first
  if (l.eng) {
    if (!first) hc(l.ahc_fn, l.ahc_base, l.ahc_scale, l.attn_norm, post_f, post_a, 1, 2, "hc_inject");
    EngRowArgs ra{M, feps};
    e = rc.op("eng_rows");
    [e setComputePipelineState:pipe(eng_q4 ? "h4_eng_rows_q4" : "h4_eng_rows")];
    [e setBuffer:eng_rows_b offset:size_t(l.eng_slot) * MR * 16 * eng_row_bytes atIndex:0];
    setb(e, l.eng_lnorm, 1);
    [e setBuffer:eng_frag offset:0 atIndex:2];
    [e setBytes:&ra length:sizeof ra atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(16, M, 1) threadsPerThreadgroup:MTLSizeMake(96, 1, 1)];
    mm(rc, "mm.eng", l.eng_kv, eng_frag, Ye, M);
    EngMixArgs ma{M, H, feps};
    e = rc.op("eng_mix");
    [e setComputePipelineState:pipe("h4_eng_mix")];
    [e setBuffer:h offset:0 atIndex:0];
    [e setBuffer:Ye offset:0 atIndex:1];
    setb(e, l.eng_bias, 2); setb(e, l.eng_nk, 3); setb(e, l.eng_nq, 4); setb(e, l.eng_gamma, 5); setb(e, l.eng_hcs, 6);
    [e setBuffer:eng_live offset:0 atIndex:7];
    [e setBytes:&ma length:sizeof ma atIndex:8];
    [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
  }
  // ---- attention read-in (inject the previous FFN output unless Engram already did)
  hc(l.ahc_fn, l.ahc_base, l.ahc_scale, l.attn_norm, post_f, post_a, (first || l.eng) ? 0 : 1, 0, "hc_attn");
  mm(rc, "mm.attn_in", l.in, xn_frag, Yin, M);
  const bool ov_gate = pf || l.in.ksplit == 1 || l.gate.ksplit == 1;
  mm(rc, "mm.gate", l.gate, xn_frag, Yg, M, nil, nil, 0, 0, 0, false, nil, 0, 1, false, !ov_gate);
  const uint32_t Nin = l.in.N, kvoff = l.in.rows.at("kva").first;
  {
    KvArgs a{M, Nin, p0, l.sparse ? 0u : RING, l.sparse ? 1u : 0u, l.sparse ? l.in.rows.at("idxk").first : 0,
             l.sparse ? l.in.rows.at("idxw").first : 0, kvoff, eps, 1.0f / sqrtf(16.0f * 128.0f), l.kv8 ? 1u : 0u};
    e = rc.op("attn_in");
    [e setComputePipelineState:pipe(lanes ? "h4_attn_in_lanes" : "h4_attn_in")];
    [e setBuffer:Yin offset:0 atIndex:0];
    setb(e, l.qa_norm, 1); setb(e, l.kva_norm, 2);
    if (l.sparse) setb(e, l.idx_k_norm, 3); else setb(e, l.qa_norm, 3);
    [e setBuffer:(l.sparse ? rope_dsa : rope_swa) offset:0 atIndex:4];
    [e setBuffer:qa_frag offset:0 atIndex:5];
    [e setBuffer:l.lat offset:0 atIndex:6];
    [e setBuffer:(l.sparse ? l.tokring : iw) offset:0 atIndex:7];
    [e setBuffer:iw offset:0 atIndex:8];
    [e setBytes:&a length:sizeof a atIndex:9];
    if (lanes) {
      [e setBuffer:lane_tab[li] offset:0 atIndex:12];
      [e setBuffer:row_lane_b offset:0 atIndex:13];
      [e setBuffer:row_pos_b offset:0 atIndex:14];
    }
    [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
  mm(rc, "mm.qb", l.qb, qa_frag, Yq, M);
  if (l.sparse) {
    // idx_pool depends only on tokring (from attn_in), so it overlaps with mm.qb
    PoolArgs pa{M, p0, eps};
    e = rc.op("idx_pool", false);
    [e setComputePipelineState:pipe(lanes ? "h4_idx_pool_lanes" : "h4_idx_pool")];
    [e setBuffer:l.tokring offset:0 atIndex:0];
    setb(e, l.idx_kvp, 1); setb(e, l.idx_kgp, 2); setb(e, l.idx_ape, 3); setb(e, l.idx_key_norm, 4);
    [e setBuffer:rope_pool offset:0 atIndex:5];
    [e setBuffer:l.bk offset:0 atIndex:6];
    [e setBytes:&pa length:sizeof pa atIndex:7];
    if (lanes) {
      [e setBuffer:lane_tab[li] offset:0 atIndex:12];
      [e setBuffer:row_lane_b offset:0 atIndex:13];
      [e setBuffer:row_pos_b offset:0 atIndex:14];
    }
    [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  }
  {
    QArgs a{M, l.nh, l.qb.N, l.sparse ? 1u : 0u, rsf};
    e = rc.op("q_prep");
    [e setComputePipelineState:pipe("h4_q_prep")];
    [e setBuffer:Yq offset:0 atIndex:0];
    [e setBuffer:(l.sparse ? rope_dsa : rope_swa) offset:0 atIndex:1];
    [e setBuffer:rope_dsa offset:0 atIndex:2];
    [e setBuffer:qn_frag offset:0 atIndex:3];
    [e setBuffer:qr offset:0 atIndex:4];
    [e setBuffer:iq offset:0 atIndex:5];
    [e setBytes:&a length:sizeof a atIndex:6];
    [e dispatchThreadgroups:MTLSizeMake(l.nh + (l.sparse ? 16 : 0), M, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  }
  id<MTLBuffer> nsh = l.nh == 64 ? nslots64 : nslots32;
  mm(rc, "mm.kb", l.kb, qn_frag, qlat, M, head_slots, nsh, l.nh, rsf * 192 / 8, rsy * 512);
  // positions below see every earlier token; HY4_DENSE (diagnostic) attends densely everywhere
  const uint32_t dense_max = cfg().dense_attn ? 0xFFFFFFFFu : gtopk * ratio;
  if (l.sparse) {
    const uint32_t gmax = slots_[0].cap / 4;
    for (uint32_t r0 = 0; r0 < M; r0 += SB) {
      const uint32_t mb = std::min(SB, M - r0);
      const uint32_t top_pos = lanes ? max_pos : (p0 + r0 + mb - 1);
      const bool scored = top_pos >= dense_max;
      if (scored) {
        ScoreArgs sa{M, p0, dense_max, gmax, r0};
        e = rc.op("idx_score", r0 > 0);  // r0 == 0 overlaps with mm.kb
        [e setComputePipelineState:pipe(lanes ? "h4_idx_score_mma_lanes" : "h4_idx_score_mma")];
        [e setBuffer:iq offset:0 atIndex:0];
        [e setBuffer:iw offset:0 atIndex:1];
        [e setBuffer:l.bk offset:0 atIndex:2];
        [e setBuffer:scores offset:0 atIndex:3];
        [e setBytes:&sa length:sizeof sa atIndex:4];
        if (lanes) {
          [e setBuffer:lane_tab[li] offset:0 atIndex:12];
          [e setBuffer:row_lane_b offset:0 atIndex:13];
          [e setBuffer:row_pos_b offset:0 atIndex:14];
        }
        [e dispatchThreadgroups:MTLSizeMake(((top_pos + 1) / 4 + 255) / 256, mb, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      // DFly2 target layers keep every decode row's selection in l.lsel (lane steps too: joint speculation
      // drafts each lane from its row of the last lane step)
      TopArgs ta{M, p0, dense_max, gmax, gtopk - 1, selmax, r0};
      e = rc.op("idx_topk", scored || r0 > 0);  // dense r0 == 0 overlaps with mm.kb
      [e setComputePipelineState:pipe(lanes ? "h4_idx_topk_lanes" : "h4_idx_topk")];
      [e setBuffer:scores offset:0 atIndex:0];
      [e setBuffer:((l.lsel && !pf) ? l.lsel : sel) offset:0 atIndex:1];
      [e setBuffer:((l.lnsel && !pf) ? l.lnsel : nsel) offset:0 atIndex:2];
      [e setBuffer:mark offset:0 atIndex:3];
      [e setBytes:&ta length:sizeof ta atIndex:4];
      if (lanes) {
        [e setBuffer:lane_tab[li] offset:0 atIndex:12];
        [e setBuffer:row_lane_b offset:0 atIndex:13];
        [e setBuffer:row_pos_b offset:0 atIndex:14];
      }
      [e dispatchThreadgroups:MTLSizeMake(mb, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    if (l.lsel && pf && !lanes) {  // DFly2 only needs the last prefill row's selection (at row 0 of the 16-row l.lsel)
      if (rc.e) { [rc.e endEncoding]; rc.e = nil; }
      id<MTLBlitCommandEncoder> bl = [rc.cb blitCommandEncoder];
      [bl copyFromBuffer:sel sourceOffset:size_t(M - 1) * selmax * 4 toBuffer:l.lsel destinationOffset:0 size:size_t(selmax) * 4];
      [bl copyFromBuffer:nsel sourceOffset:size_t(M - 1) * 4 toBuffer:l.lnsel destinationOffset:0 size:4];
      [bl endEncoding];
    }
  }
  {
    // fixed per layer type for every decode M (verify rows keep decode bits); sliding layers see <= 512 keys
    // (sweep 2026-11-18, 3k context: 8 / 16 beat 32 / 32 by 0.4 / 3.9 / 4.4 ms at 1 / 4 / 6 rows)
    const uint32_t splits = pf ? 1 : (l.sparse ? cfg().as_dsa : cfg().as_swa);
    AttnArgs a{M, l.nh, p0, l.sparse ? 0u : 1u, window, l.sparse ? 0u : RING, splits, selmax, 1.0f / 16.0f, rsy, l.kv8 ? 1u : 0u};
    const bool direct = pf;  // prefill: the one split's merge inside the attention kernel
    a.direct = direct ? 1u : 0u;
    a.rsf = rsf;
    e = rc.op("attn");
    const char* ap = l.sparse ? (lanes ? "h4_attn_mma_dsa_lanes" : "h4_attn_mma_dsa")
                              : (lanes ? "h4_attn_mma_swa_lanes" : "h4_attn_mma_swa");
    [e setComputePipelineState:pipe(ap)];
    [e setBuffer:qlat offset:0 atIndex:0];
    [e setBuffer:qr offset:0 atIndex:1];
    [e setBuffer:l.lat offset:0 atIndex:2];
    [e setBuffer:((l.lsel && !pf) ? l.lsel : sel) offset:0 atIndex:3];
    [e setBuffer:((l.lnsel && !pf) ? l.lnsel : nsel) offset:0 atIndex:4];
    [e setBuffer:part offset:0 atIndex:5];
    [e setBytes:&a length:sizeof a atIndex:6];
    setb(e, l.sinks, 7);
    [e setBuffer:ofrag offset:0 atIndex:8];
    if (lanes) {
      [e setBuffer:lane_tab[li] offset:0 atIndex:12];
      [e setBuffer:row_lane_b offset:0 atIndex:13];
      [e setBuffer:row_pos_b offset:0 atIndex:14];
    }
    [e dispatchThreadgroups:MTLSizeMake(l.nh / 16, M, splits) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    if (!direct) {
    MergeArgs ma{M, l.nh, splits, rsf};
    e = rc.op("attn_merge");
    [e setComputePipelineState:pipe("h4_attn_merge")];
    [e setBuffer:part offset:0 atIndex:0];
    setb(e, l.sinks, 1);
    [e setBuffer:ofrag offset:0 atIndex:2];
    [e setBytes:&ma length:sizeof ma atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(l.nh, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
  }
  if (cfg().fuse_gate) {
    mm(rc, "mm.vb", l.vb, ofrag, yv, M, head_slots, nsh, l.nh, rsf * 512 / 8, rsy * 256, false, gfrag, l.nh * 256, 1, false, true, Yg);
  } else {
    mm(rc, "mm.vb", l.vb, ofrag, yv, M, head_slots, nsh, l.nh, rsf * 512 / 8, rsy * 256);
    {
      GateArgs a{M, l.nh, rsy};
      e = rc.op("gate");
      [e setComputePipelineState:pipe("h4_gate")];
      [e setBuffer:yv offset:0 atIndex:0];
      [e setBuffer:Yg offset:0 atIndex:1];
      [e setBuffer:gfrag offset:0 atIndex:2];
      [e setBytes:&a length:sizeof a atIndex:3];
      [e dispatchThreads:MTLSizeMake(l.nh * 256, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
  }
  mm(rc, "mm.o", l.o, gfrag, Yo, M);
  rms(Yo, l.post_norm, y, H, H, 0, "attn_post_norm");
  // ---- FFN read-in
  hc(l.mhc_fn, l.mhc_base, l.mhc_scale, l.ffn_norm, post_a, post_f, 1, 0, "hc_ffn");
  if (l.dense) {
    const uint32_t ff = l.mlp_dn.K;
    // SwiGLU fused in mm2/mmp EPI=3 into mlp_frag (both prefill and decode)
    mm(rc, "mm.mlp_gu", l.mlp_gu, xn_frag, Ymlp, M, nil, nil, 0, 0, 0, true, mlp_frag, ff);
    mm(rc, "mm.mlp_dn", l.mlp_dn, mlp_frag, Yo, M);
    rms(Yo, l.mlp_post, y, H, H, 0, "mlp_post_norm");
    return;
  }
  mm(rc, "mm.moe_in", l.moe_in, xn_frag, Ym, M);
  {
    RouteArgs a{M, E, topk, l.moe_in.N, LAT, I, rscale};
    e = rc.op("route");
    [e setComputePipelineState:pipe("h4_route")];
    [e setBuffer:Ym offset:0 atIndex:0];
    setb(e, l.moe_bias, 1);
    [e setBuffer:topi offset:0 atIndex:2];
    [e setBuffer:topw offset:0 atIndex:3];
    [e setBuffer:lat_frag offset:0 atIndex:4];
    [e setBuffer:mo_frag offset:0 atIndex:5];
    [e setBytes:&a length:sizeof a atIndex:6];
    [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
    if (pf) {
      const uint32_t gx_gu = (l.moe_gu.N / 32 + grpmmp_nf_ - 1) / grpmmp_nf_;
      const uint32_t gx_dn = (l.moe_dn.N / 32 + grpmmp_nf_ - 1) / grpmmp_nf_;
      ListArgs la{M, topk, E, 16 * gch_, gx_gu, gx_dn};
      e = rc.op("plan_lists");
      [e setComputePipelineState:pipe("h4_plan_lists")];
      [e setBuffer:topi offset:0 atIndex:0];
      [e setBuffer:slots offset:0 atIndex:1];
      [e setBuffer:nslots offset:0 atIndex:2];
      [e setBuffer:roff offset:0 atIndex:3];
      [e setBuffer:rcnt offset:0 atIndex:4];
      [e setBuffer:rlist offset:0 atIndex:5];
      [e setBuffer:cidx offset:0 atIndex:6];
      [e setBytes:&la length:sizeof la atIndex:7];
      [e setBuffer:wtile offset:0 atIndex:8];
      [e setBuffer:moe_ind offset:0 atIndex:9];
      [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    } else {
      PlanArgs pa{M, topk, E};
      e = rc.op("plan");
      [e setComputePipelineState:pipe("h4_plan")];
      [e setBuffer:topi offset:0 atIndex:0];
      [e setBuffer:slots offset:0 atIndex:1];
      [e setBuffer:nslots offset:0 atIndex:2];
      [e setBuffer:slot_of offset:0 atIndex:3];
      [e setBytes:&pa length:sizeof pa atIndex:4];
      [e setBuffer:rmask offset:0 atIndex:5];
      [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
    }
  }
  if (pf) {
    // experts over per-expert row lists: gate|up (clamped SwiGLU into compact frags), optional WHT128, then down
    mm(rc, "mm.moe_gu", l.moe_gu, lat_frag, Ygu, M, slots, nslots, E, 0, 0, true, act, I, 0, true);
    if (moe_dn_had == 128) {
      WhtArgs wa{M * topk, I};
      e = rc.op("wht128");
      [e setComputePipelineState:pipe("h4_wht128_pf")];
      [e setBuffer:act offset:0 atIndex:0];
      [e setBytes:&wa length:sizeof wa atIndex:1];
      [e dispatchThreadgroups:MTLSizeMake(I / 128, (M * topk + 3u) / 4u, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    mm(rc, "mm.moe_dn", l.moe_dn, act, yd, M, slots, nslots, E, 0, 0, false, nil, 0, 2, true);
    CombCArgs a{M, topk, LAT};
    e = rc.op("combine");
    [e setComputePipelineState:pipe("h4_combine_c")];
    [e setBuffer:yd offset:0 atIndex:0];
    [e setBuffer:cidx offset:0 atIndex:1];
    [e setBuffer:topw offset:0 atIndex:2];
    [e setBuffer:mo_frag offset:0 atIndex:3];
    [e setBytes:&a length:sizeof a atIndex:4];
    [e dispatchThreads:MTLSizeMake(LAT, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  } else {
    const uint32_t smax = std::min(E, M * topk);
    mm(rc, "mm.moe_gu", l.moe_gu, lat_frag, Ygu, M, slots, nslots, smax, 0, 16 * 2 * I, true, act, I, 1, true);
    if (moe_dn_had == 128) {
      WhtArgs wa{M, I};
      e = rc.op("wht128");
      [e setComputePipelineState:pipe("h4_wht128_dec")];
      [e setBuffer:act offset:0 atIndex:0];
      [e setBuffer:nslots offset:0 atIndex:1];
      [e setBuffer:rmask offset:0 atIndex:2];
      [e setBytes:&wa length:sizeof wa atIndex:3];
      [e dispatchThreadgroups:MTLSizeMake(I / 128, smax, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    mm(rc, "mm.moe_dn", l.moe_dn, act, yd, M, slots, nslots, smax, (I / 32) * 64, 16 * LAT, false, nil, 0, 1, true);
    CombArgs a{M, topk, LAT, 16 * LAT};
    e = rc.op("combine");
    [e setComputePipelineState:pipe("h4_combine")];
    [e setBuffer:yd offset:0 atIndex:0];
    [e setBuffer:topi offset:0 atIndex:1];
    [e setBuffer:topw offset:0 atIndex:2];
    [e setBuffer:slot_of offset:0 atIndex:3];
    [e setBuffer:mo_frag offset:0 atIndex:4];
    [e setBytes:&a length:sizeof a atIndex:5];
    [e dispatchThreads:MTLSizeMake(LAT, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
  mm(rc, "mm.moe_out", l.moe_out, mo_frag, Yo, M);
  rms(Yo, l.moe_norm, y, H, H, 0, "moe_norm");
  (void)li;
}

std::vector<uint32_t> Model::step(const std::vector<uint32_t>& toks, double* gpu_ms, std::vector<float>* logits_out,
                                  std::map<std::string, double>* profile,
                                  const float* vis_embeds, uint32_t n_vis, const int32_t* vis_idx,
                                  const Pos3D* pos3d_in) {
  const uint32_t M = uint32_t(toks.size());
  if (M == 0 || M > MR) mtl::die("step: too many rows (HY4_MP)");
  if (logits_out && M > MAXM) mtl::die("step: all-row logits only for <= 16 rows");
  if (pos + M > cap) mtl::die("step: KV capacity exceeded");
  const uint32_t p0 = pos;
  const double h_entry = host_now();
  double h_commit0 = 0;
  mtl::last_use() = mtl::now_s();
  memcpy(ids.contents, toks.data(), M * 4);
  history.insert(history.end(), toks.begin(), toks.end());
  // Maintain 3D MRoPE positions parallel to history (text-only fast path keeps t==h==w).
  if (pos3d.size() < p0) {
    const size_t old = pos3d.size();
    pos3d.resize(p0);
    for (size_t k = old; k < p0; k++) {
      const uint32_t pk = uint32_t(int64_t(k) + mrope_delta);
      pos3d[k] = Pos3D{pk, pk, pk};
    }
  }
  pos3d.resize(p0 + M);
  std::vector<Pos3D> pp3(M), pg3(M);
  for (uint32_t r = 0; r < M; r++) {
    Pos3D pr;
    if (pos3d_in) {
      pr = pos3d_in[r];
    } else {
      const uint32_t p1 = uint32_t(int64_t(p0 + r) + mrope_delta);
      pr = Pos3D{p1, p1, p1};
    }
    pos3d[p0 + r] = pr;
    pp3[r] = pr;
  }
  {
    const Pos3D& b = pos3d[p0 + M - 1];
    mrope_delta = int32_t(std::max({b.t, b.h, b.w}) + 1u) - int32_t(p0 + M);
  }
  for (uint32_t r = 0; r < M; r++) {
    const uint32_t g0 = (p0 + r) / 4 * 4;
    pg3[r] = pos3d[g0];
  }
  rope_rows_3d(inv_swa, pp3, (float*)rope_swa.contents);
  rope_rows_3d(inv_dsa, pp3, (float*)rope_dsa.contents);
  rope_rows_3d(inv_dsa, pg3, (float*)rope_pool.contents);

  id<MTLBuffer> vis_buf = nil;
  if (vis_embeds && n_vis > 0 && vis_idx) {
    vis_buf = [C.dev newBufferWithBytes:vis_embeds length:size_t(n_vis) * H * 4 options:MTLResourceStorageModeShared];
    if (!vis_idx_b) vis_idx_b = C.buf(MR * 4);
    memcpy(vis_idx_b.contents, vis_idx, M * 4);
  }

  Rec rc;
  rc.cb = [C.q commandBuffer];
  rc.prof = profile != nullptr;
  std::vector<id<MTLCommandBuffer>> sent;  // early commits: the GPU starts while later layers are encoded
  const uint32_t chunk_layers = cfg().commit_layers;
  auto commit_now = [&]() {
    if (rc.e) { [rc.e endEncoding]; rc.e = nil; }
    [rc.cb commit];
    if (!h_commit0) h_commit0 = host_now();
    sent.push_back(rc.cb);
    rc.cb = [C.q commandBuffer];
  };
  id<MTLComputeCommandEncoder> e = rc.op("embed");
  {
    struct { uint32_t M, H, HC; } a{M, H, HC};
    if (vis_buf) {
      [e setComputePipelineState:pipe("h4_embed_vis")];
      [e setBuffer:ids offset:0 atIndex:0];
      setb(e, embed, 1);
      [e setBuffer:vis_buf offset:0 atIndex:2];
      [e setBuffer:vis_idx_b offset:0 atIndex:3];
      [e setBuffer:h offset:0 atIndex:4];
      [e setBytes:&a length:sizeof a atIndex:5];
    } else {
      [e setComputePipelineState:pipe("h4_embed")];
      [e setBuffer:ids offset:0 atIndex:0];
      setb(e, embed, 1);
      [e setBuffer:h offset:0 atIndex:2];
      [e setBytes:&a length:sizeof a atIndex:3];
    }
    [e dispatchThreads:MTLSizeMake(H, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
  for (uint32_t i = 0; i < nL; i++) {
    if (i == 1 || (i == 0 && L[0].eng)) {
      // the layers before the first Engram layer run while the host fetches the rows
      if (i == 1) commit_now();
      const bool trows = cfg().time_rows;
      auto r0 = std::chrono::steady_clock::now();
      engram_rows(M, toks.data());
      if (n_eng && !next_hint.empty() && next_hint.size() <= MR) {
        // the next chunk's rows, fetched while the GPU runs this one (its context: history + next_hint)
        pf_toks = next_hint;
        pf_p0 = history.size();
        auto ctx = std::make_shared<std::vector<uint32_t>>(history);
        ctx->insert(ctx->end(), next_hint.begin(), next_hint.end());
        const uint32_t nm = uint32_t(next_hint.size());
        uint8_t* d = (uint8_t*)eng_rows_pf.contents;
        uint32_t* lv = (uint32_t*)eng_live_pf.contents;
        const size_t q0 = pf_p0;
        pf_job = std::async(std::launch::async, [this, ctx, q0, nm, d, lv] { engram_fetch(*ctx, q0, nm, d, lv); });
      }
      next_hint.clear();
      eng_wait_ms += std::chrono::duration<double>(std::chrono::steady_clock::now() - r0).count() * 1e3;
      if (trows) fprintf(stderr, "engram rows: %u tokens, %.1f ms\n", M, std::chrono::duration<double>(std::chrono::steady_clock::now() - r0).count() * 1e3);
    } else if (L[i].eng && eng_rest.valid()) {
      commit_now();  // the GPU runs what is encoded while the later layers' rows arrive
      auto r0 = std::chrono::steady_clock::now();
      eng_rest.wait();
      eng_rest = {};
      const double w = std::chrono::duration<double>(std::chrono::steady_clock::now() - r0).count() * 1e3;
      eng_wait_ms += w;
      eng_rest_ms += w;
    } else if (chunk_layers && i > 1 && i % chunk_layers == 0) {
      commit_now();
    }
    encode_layer(rc, L[i], i, M, p0, i == 0);
  }
  // prefill: only the last row goes through the head (its streams moved to row 0 first)
  const uint32_t MH = M > MAXM ? 1 : M;
  if (M > MAXM) {
    id<MTLBlitCommandEncoder> bl;
    if (rc.e) { [rc.e endEncoding]; rc.e = nil; }
    bl = [rc.cb blitCommandEncoder];
    [bl copyFromBuffer:h sourceOffset:size_t(M - 1) * HC * H * 4 toBuffer:h destinationOffset:0 size:size_t(HC) * H * 4];
    [bl copyFromBuffer:(ynorm_pending ? Yo : y) sourceOffset:size_t(M - 1) * H * 4 toBuffer:(ynorm_pending ? Yo : y) destinationOffset:0 size:size_t(H) * 4];
    [bl copyFromBuffer:post_f sourceOffset:size_t(M - 1) * 16 toBuffer:post_f destinationOffset:0 size:16];
    [bl endEncoding];
  }
  hc_read(rc, hch_fn, hch_base, hch_scale, norm, post_f, post_a, 1, 1, MH, "hc_head");
  mm(rc, "mm.lm_head", lm_head, xn_frag, logits, MH);
  {
    ArgArgs a{MH, lm_head.N};
    SampArgs sa{MH, lm_head.N, p0 + M - MH, uint32_t(seed), uint32_t(seed >> 32), temp > 0 ? 1.0f / temp : 1.0f, min_p > 0 ? logf(min_p) : -INFINITY, top_p, top_k};
    e = rc.op("argmax");
    [e setComputePipelineState:pipe(temp > 0 ? "h4_sample" : "h4_argmax")];
    [e setBuffer:logits offset:0 atIndex:0];
    [e setBuffer:amax offset:0 atIndex:1];
    if (temp > 0) [e setBytes:&sa length:sizeof sa atIndex:2]; else [e setBytes:&a length:sizeof a atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(MH, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
  }
  if (rc.e) [rc.e endEncoding];
  [rc.cb commit];
  const double h_commit1 = host_now();
  if (!h_commit0) h_commit0 = h_commit1;
  [rc.cb waitUntilCompleted];
  const double h_wake = host_now();
  if (rc.cb.error) mtl::die("step: command buffer failed", rc.cb.error);
  {
    // GPU span from the first buffer's start to the last one's end (the buffers run back to back)
    double t0 = rc.cb.GPUStartTime, t1 = rc.cb.GPUEndTime;
    for (id<MTLCommandBuffer> b : sent) { t0 = std::min(t0, b.GPUStartTime); t1 = std::max(t1, b.GPUEndTime); }
    if (gpu_ms) *gpu_ms = (t1 - t0) * 1e3;
    last_lat = {(h_commit0 - h_entry) * 1e3, (h_commit1 - h_entry) * 1e3, (t0 - h_commit0) * 1e3, (t1 - t0) * 1e3,
                (h_wake - t1) * 1e3};
  }
  if (profile) rc.resolve(*profile);
  pos += M;
  std::vector<uint32_t> out(MH);
  memcpy(out.data(), amax.contents, MH * 4);
  if (logits_out) {
    logits_out->resize(size_t(M) * vocab);
    for (uint32_t r = 0; r < M; r++) memcpy(logits_out->data() + size_t(r) * vocab, (float*)logits.contents + size_t(r) * lm_head.N, vocab * 4);
  }
  return out;
}

std::vector<uint32_t> Model::step_lanes(const std::vector<LaneBatch>& batches, double* gpu_ms,
                                        std::vector<float>* logits_out) {
  ensure_lane_pipes();
  if (lane_tab.empty()) rebuild_lane_tables();
  struct SampRow { uint32_t N, pos, s0, s1; float inv_t, lmin, top_p; uint32_t top_k; };
  std::vector<uint32_t> toks, rlane, rpos;
  std::vector<Pos3D> rpos3, pg3;
  std::vector<const std::vector<uint32_t>*> rhist;
  std::vector<size_t> rpos_sz;
  std::vector<SampRow> srows;
  uint32_t max_pos = 0;
  for (const auto& b : batches) {
    if (b.slot < 0 || b.slot >= int(slots_.size())) mtl::die("step_lanes: invalid slot");
    if (b.toks.empty()) continue;
    uint32_t& sp = (b.slot == cur_slot) ? pos : slots_[b.slot].pos;
    std::vector<uint32_t>& sh = (b.slot == cur_slot) ? history : slots_[b.slot].history;
    std::vector<Pos3D>& sp3 = (b.slot == cur_slot) ? pos3d : slots_[b.slot].pos3d;
    const int32_t sdelta = (b.slot == cur_slot) ? mrope_delta : slots_[b.slot].mrope_delta;
    const uint32_t scap = (b.slot == cur_slot) ? cap : slots_[b.slot].cap;
    if (sp + b.toks.size() > scap) mtl::die("step_lanes: lane KV capacity exceeded");
    const uint32_t p0 = sp;
    sh.insert(sh.end(), b.toks.begin(), b.toks.end());
    if (sp3.size() < p0) {
      const size_t old = sp3.size();
      sp3.resize(p0);
      for (size_t k = old; k < p0; k++) {
        const uint32_t pk = uint32_t(int64_t(k) + sdelta);
        sp3[k] = Pos3D{pk, pk, pk};
      }
    }
    for (size_t j = 0; j < b.toks.size(); j++) {
      const uint32_t p = p0 + uint32_t(j);
      const uint32_t rp = uint32_t(int64_t(p) + sdelta);
      const Pos3D pr{rp, rp, rp};
      sp3.push_back(pr);
      toks.push_back(b.toks[j]);
      rlane.push_back(uint32_t(b.slot));
      rpos.push_back(p);
      rpos3.push_back(pr);
      rpos_sz.push_back(p);
      pg3.push_back(sp3[p / 4 * 4]);
      rhist.push_back(&sh);
      max_pos = std::max(max_pos, p);
      srows.push_back(SampRow{lm_head.N, p, uint32_t(b.seed), uint32_t(b.seed >> 32),
                              b.temp > 0 ? 1.0f / b.temp : -1.0f, b.min_p > 0 ? logf(b.min_p) : -INFINITY,
                              b.top_p, b.top_k});
    }
  }
  const uint32_t M = uint32_t(toks.size());
  if (M == 0 || M > MAXM) mtl::die("step_lanes: M must be in 1..MAXM");
  mtl::last_use() = mtl::now_s();
  memcpy(ids.contents, toks.data(), M * 4);
  memcpy(row_lane_b.contents, rlane.data(), M * 4);
  memcpy(row_pos_b.contents, rpos.data(), M * 4);
  memcpy(samp_rows_b.contents, srows.data(), M * sizeof(SampRow));
  rope_rows_3d(inv_swa, rpos3, (float*)rope_swa.contents);
  rope_rows_3d(inv_dsa, rpos3, (float*)rope_dsa.contents);
  rope_rows_3d(inv_dsa, pg3, (float*)rope_pool.contents);

  Rec rc;
  rc.cb = [C.q commandBuffer];
  std::vector<id<MTLCommandBuffer>> sent;
  const uint32_t chunk_layers = cfg().commit_layers;
  auto commit_now = [&]() {
    if (rc.e) { [rc.e endEncoding]; rc.e = nil; }
    [rc.cb commit];
    sent.push_back(rc.cb);
    rc.cb = [C.q commandBuffer];
  };
  id<MTLComputeCommandEncoder> e = rc.op("embed");
  {
    struct { uint32_t M, H, HC; } a{M, H, HC};
    [e setComputePipelineState:pipe("h4_embed")];
    [e setBuffer:ids offset:0 atIndex:0];
    setb(e, embed, 1);
    [e setBuffer:h offset:0 atIndex:2];
    [e setBytes:&a length:sizeof a atIndex:3];
    [e dispatchThreads:MTLSizeMake(H, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
  for (uint32_t i = 0; i < nL; i++) {
    if (n_eng && (i == 1 || (i == 0 && L[0].eng))) {
      if (i == 1) commit_now();
      std::vector<const std::vector<uint32_t>*> miss_hist;
      std::vector<size_t> miss_pos;
      std::vector<uint32_t> miss_dst;
      if (anchor_lanes_job.valid()) {
        anchor_lanes_job.wait();
        uint8_t* dst = (uint8_t*)eng_rows_b.contents;
        const uint8_t* src = (const uint8_t*)eng_rows_pf.contents;
        uint32_t* dlv = (uint32_t*)eng_live.contents;
        const uint32_t* slv = (const uint32_t*)eng_live_pf.contents;
        for (uint32_t r = 0; r < M; r++) {
          int hit = -1;
          for (size_t k = 0; k < anchor_lanes_meta.size(); k++) {
            auto [sl, sp, tk] = anchor_lanes_meta[k];
            if (sl == int(rlane[r]) && sp == rpos_sz[r] && tk == toks[r]) { hit = int(k); break; }
          }
          if (hit >= 0) {
            for (uint32_t e = 0; e < n_eng; e++)
              memcpy(dst + (size_t(e) * MR * 16 + size_t(r) * 16) * eng_row_bytes,
                     src + (size_t(e) * MR * 16 + size_t(hit) * 16) * eng_row_bytes, 16 * eng_row_bytes);
            dlv[r] = slv[hit];
          } else {
            miss_hist.push_back(rhist[r]);
            miss_pos.push_back(rpos_sz[r]);
            miss_dst.push_back(r);
          }
        }
        anchor_lanes_meta.clear();
      } else {
        miss_hist = rhist;
        miss_pos = rpos_sz;
        miss_dst.resize(M);
        for (uint32_t r = 0; r < M; r++) miss_dst[r] = r;
      }
      if (!miss_hist.empty()) {
        if (n_eng > 1 && !cfg().eng_sync) {
          engram_fetch_lanes(miss_hist, miss_pos, (uint8_t*)eng_rows_b.contents, (uint32_t*)eng_live.contents, 0, 1, miss_dst.data());
          uint8_t* d = (uint8_t*)eng_rows_b.contents;
          eng_rest = std::async(std::launch::async, [this, miss_hist, miss_pos, miss_dst, d] {
            engram_fetch_lanes(miss_hist, miss_pos, d, nullptr, 1, n_eng, miss_dst.data());
          });
        } else {
          engram_fetch_lanes(miss_hist, miss_pos, (uint8_t*)eng_rows_b.contents, (uint32_t*)eng_live.contents, 0, n_eng, miss_dst.data());
        }
      }
    } else if (L[i].eng && eng_rest.valid()) {
      commit_now();
      eng_rest.wait();
      eng_rest = {};
    } else if (chunk_layers && i > 1 && i % chunk_layers == 0) {
      commit_now();
    }
    encode_layer(rc, L[i], i, M, 0, i == 0, true, max_pos);
  }
  hc_read(rc, hch_fn, hch_base, hch_scale, norm, post_f, post_a, 1, 1, M, "hc_head");
  mm(rc, "mm.lm_head", lm_head, xn_frag, logits, M);
  {
    e = rc.op("sample_lanes");
    [e setComputePipelineState:pipe("h4_sample_lanes")];
    [e setBuffer:logits offset:0 atIndex:0];
    [e setBuffer:amax offset:0 atIndex:1];
    [e setBuffer:samp_rows_b offset:0 atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
  }
  if (rc.e) [rc.e endEncoding];
  [rc.cb commit];
  [rc.cb waitUntilCompleted];
  if (rc.cb.error) mtl::die("step_lanes: command buffer failed", rc.cb.error);
  if (gpu_ms) {
    double t0 = rc.cb.GPUStartTime, t1 = rc.cb.GPUEndTime;
    for (id<MTLCommandBuffer> b : sent) { t0 = std::min(t0, b.GPUStartTime); t1 = std::max(t1, b.GPUEndTime); }
    *gpu_ms = (t1 - t0) * 1e3;
  }
  for (const auto& b : batches) {
    uint32_t& sp = (b.slot == cur_slot) ? pos : slots_[b.slot].pos;
    sp += uint32_t(b.toks.size());
  }
  std::vector<uint32_t> out(M);
  memcpy(out.data(), amax.contents, M * 4);
  if (logits_out) {
    logits_out->resize(size_t(M) * vocab);
    for (uint32_t r = 0; r < M; r++) memcpy(logits_out->data() + size_t(r) * vocab, (float*)logits.contents + size_t(r) * lm_head.N, vocab * 4);
  }
  return out;
}

struct AddArgs { uint32_t M, H, ldy, gated; };
struct DselArgs { uint32_t rows, src, anchor, selmax_t, selmax_d; };
struct AddFArgs { uint32_t M, H, ldx, offx; };

std::vector<uint32_t> Model::draft(uint32_t anchor, uint32_t src_x, uint32_t src_s, double* gpu_ms,
                                   std::map<std::string, double>* profile, bool cut) {
  if (!has_dfly) mtl::die("draft: pack has no DFly2 drafter");
  const double h_entry = host_now();
  const uint32_t W = dblock, p = pos;
  if (p + W > cap) return {};
  {
    uint32_t* di = (uint32_t*)dids.contents;
    di[0] = anchor;
    for (uint32_t i = 1; i < W; i++) di[i] = mask_id;
    std::vector<uint32_t> pp(W);
    for (uint32_t i = 0; i < W; i++) pp[i] = uint32_t(int64_t(p + i) + mrope_delta);
    rope_rows(inv_dsa, pp, (float*)rope_draft.contents);
  }
  Rec rc;
  rc.cb = [C.q commandBuffer];
  rc.prof = profile != nullptr;
  id<MTLComputeCommandEncoder> e;
  auto embed_rows = [&](id<MTLBuffer> idb, size_t idoff, uint32_t M, id<MTLBuffer> out, bool barrier = true) {
    struct { uint32_t M, H, HC; } a{M, H, 1};
    e = rc.op("d_embed", barrier);
    [e setComputePipelineState:pipe("h4_embed")];
    [e setBuffer:idb offset:idoff atIndex:0];
    setb(e, embed, 1);
    [e setBuffer:out offset:0 atIndex:2];
    [e setBytes:&a length:sizeof a atIndex:3];
    [e dispatchThreads:MTLSizeMake(H, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  };
  // rms(x rows)*w -> f32 out and/or frag (columns fo.. of a fk-wide frag)
  auto rms = [&](id<MTLBuffer> X, uint32_t M, uint32_t ldi, uint32_t off, const Raw& w, id<MTLBuffer> Yf, id<MTLBuffer> frag,
                 uint32_t fo, uint32_t fk, bool barrier = true) {
    RmsArgs a{M, H, ldi, off, H, Yf ? 1u : 0u, frag ? 1u : 0u, eps, fo, fk};
    e = rc.op("d_rms", barrier);
    [e setComputePipelineState:pipe("h4_rms")];
    [e setBuffer:X offset:0 atIndex:0];
    setb(e, w, 1);
    [e setBuffer:(Yf ? Yf : y) offset:0 atIndex:2];
    [e setBuffer:(frag ? frag : xn_frag) offset:0 atIndex:3];
    [e setBytes:&a length:sizeof a atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  };
  auto add = [&](id<MTLBuffer> X, id<MTLBuffer> Yv, uint32_t M, uint32_t ldy, const Raw* g) {
    AddArgs a{M, H, ldy, g ? 1u : 0u};
    e = rc.op("d_add");
    [e setComputePipelineState:pipe("h4_add")];
    [e setBuffer:X offset:0 atIndex:0];
    [e setBuffer:Yv offset:0 atIndex:1];
    if (g) setb(e, *g, 2); else [e setBuffer:Yv offset:0 atIndex:2];
    [e setBytes:&a length:sizeof a atIndex:3];
    [e dispatchThreads:MTLSizeMake(H, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  };
  // ---- block embeddings; slot 0 fuses the anchor with the target's final hidden of the source row
  const bool use_saved = (src_x == ~0u);
  embed_rows(dids, 0, W, dx);
  rms(dx, 1, H, 0, df_enorm, nil, dfrag2, 0, 2 * H);
  rms(use_saved ? slots_[cur_slot].df_xf : xf, 1, H, use_saved ? 0 : src_x * H, df_hnorm, nil, dfrag2, H, 2 * H, false);
  mm(rc, "d_eh", df_eh, dfrag2, Yo, 1);
  add(dx, Yo, 1, H, &df_gate);
  bool xn_ready = false;
  for (size_t di = 0; di < D.size(); di++) {
    auto& d = D[di];
    Layer& t = L[d.tl];
    if (!xn_ready) rms(dx, W, H, 0, d.in_norm, nil, xn_frag, 0, 0);
    xn_ready = false;
    mm(rc, "d_attn_in", d.in, xn_frag, Yin, W);
    const bool ov_dgate = d.in.ksplit == 1 || d.gate.ksplit == 1;
    mm(rc, "d_gate", d.gate, xn_frag, Yg, W, nil, nil, 0, 0, 0, false, nil, 0, 1, false, !ov_dgate);
    {
      KvArgs a{W, d.in.N, p, 0, 0, 0, 0, d.in.rows.at("kva").first, eps, 1.0f, t.kv8 ? 1u : 0u};
      e = rc.op("d_attn_in_k");
      [e setComputePipelineState:pipe("h4_attn_in")];
      [e setBuffer:Yin offset:0 atIndex:0];
      setb(e, d.qa_norm, 1); setb(e, d.kva_norm, 2); setb(e, d.qa_norm, 3);
      [e setBuffer:rope_draft offset:0 atIndex:4];
      [e setBuffer:qa_frag offset:0 atIndex:5];
      [e setBuffer:t.lat offset:0 atIndex:6];   // the block's latent rows go into the target layer's cache
      [e setBuffer:iw offset:0 atIndex:7];
      [e setBuffer:iw offset:0 atIndex:8];
      [e setBytes:&a length:sizeof a atIndex:9];
      [e dispatchThreadgroups:MTLSizeMake(W, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    mm(rc, "d_qb", d.qb, qa_frag, Yq, W);
    {
      QArgs a{W, d.nh, d.qb.N, 0, 16};
      e = rc.op("d_q_prep");
      [e setComputePipelineState:pipe("h4_q_prep")];
      [e setBuffer:Yq offset:0 atIndex:0];
      [e setBuffer:rope_draft offset:0 atIndex:1];
      [e setBuffer:rope_draft offset:0 atIndex:2];
      [e setBuffer:qn_frag offset:0 atIndex:3];
      [e setBuffer:qr offset:0 atIndex:4];
      [e setBuffer:iq offset:0 atIndex:5];
      [e setBytes:&a length:sizeof a atIndex:6];
      [e dispatchThreadgroups:MTLSizeMake(d.nh, W, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
    id<MTLBuffer> nsh = d.nh == 64 ? nslots64 : nslots32;
    mm(rc, "d_kb", d.kb, qn_frag, qlat, W, head_slots, nsh, d.nh, 16 * 192 / 8, 16 * 512);
    {
      const bool use_saved_s = (src_s == ~0u);
      DselArgs a{W, use_saved_s ? 0u : src_s, p, selmax, DSELMAX};
      e = rc.op("d_dsel", false);
      [e setComputePipelineState:pipe("h4_dsel")];
      [e setBuffer:(use_saved_s ? slots_[cur_slot].df_lsel[di] : t.lsel) offset:0 atIndex:0];
      [e setBuffer:(use_saved_s ? slots_[cur_slot].df_lnsel[di] : t.lnsel) offset:0 atIndex:1];
      [e setBuffer:dsel offset:0 atIndex:2];
      [e setBuffer:dnsel offset:0 atIndex:3];
      [e setBytes:&a length:sizeof a atIndex:4];
      [e dispatchThreads:MTLSizeMake(DSELMAX, W, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    {
      AttnArgs a{W, d.nh, p, 0, window, 0, as_draft, DSELMAX, 1.0f / 16.0f, 16, t.kv8 ? 1u : 0u};
      e = rc.op("d_attn");
      [e setComputePipelineState:pipe("h4_attn_mma_dsa")];
      [e setBuffer:qlat offset:0 atIndex:0];
      [e setBuffer:qr offset:0 atIndex:1];
      [e setBuffer:t.lat offset:0 atIndex:2];
      [e setBuffer:dsel offset:0 atIndex:3];
      [e setBuffer:dnsel offset:0 atIndex:4];
      [e setBuffer:part offset:0 atIndex:5];
      [e setBytes:&a length:sizeof a atIndex:6];
      setb(e, d.sinks, 7);
      [e setBuffer:ofrag offset:0 atIndex:8];
      [e dispatchThreadgroups:MTLSizeMake(d.nh / 16, W, as_draft) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      MergeArgs ma{W, d.nh, as_draft, 16};
      e = rc.op("d_attn_merge");
      [e setComputePipelineState:pipe("h4_attn_merge")];
      [e setBuffer:part offset:0 atIndex:0];
      setb(e, d.sinks, 1);
      [e setBuffer:ofrag offset:0 atIndex:2];
      [e setBytes:&ma length:sizeof ma atIndex:3];
      [e dispatchThreadgroups:MTLSizeMake(d.nh, W, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    if (cfg().fuse_gate) {
      mm(rc, "d_vb", d.vb, ofrag, yv, W, head_slots, nsh, d.nh, 16 * 512 / 8, 16 * 256, false, gfrag, d.nh * 256, 1, false, true, Yg);
    } else {
      mm(rc, "d_vb", d.vb, ofrag, yv, W, head_slots, nsh, d.nh, 16 * 512 / 8, 16 * 256);
      {
        GateArgs a{W, d.nh, 16};
        e = rc.op("d_gate_mul");
        [e setComputePipelineState:pipe("h4_gate")];
        [e setBuffer:yv offset:0 atIndex:0];
        [e setBuffer:Yg offset:0 atIndex:1];
        [e setBuffer:gfrag offset:0 atIndex:2];
        [e setBytes:&a length:sizeof a atIndex:3];
        [e dispatchThreads:MTLSizeMake(d.nh * 256, W, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
    }
    mm(rc, "d_o", d.o, gfrag, Yo, W);
    {
      struct { uint32_t M, H; float eps; } pa{W, H, eps};
      e = rc.op("d_post_add_ln");
      [e setComputePipelineState:pipe("h4_d_post_add_ln")];
      [e setBuffer:Yo offset:0 atIndex:0];
      setb(e, d.post_norm, 1);
      [e setBuffer:dx offset:0 atIndex:2];
      setb(e, d.post_ln, 3);
      [e setBuffer:xn_frag offset:0 atIndex:4];
      [e setBytes:&pa length:sizeof pa atIndex:5];
      [e dispatchThreadgroups:MTLSizeMake(W, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    if (d.moe) {
      mm(rc, "d_moe_in", d.moe_in, xn_frag, Ym, W);
      {
        RouteArgs a{W, d_E, d_topk, d.moe_in.N, d_LAT, d_I, d_rscale};
        e = rc.op("d_route");
        [e setComputePipelineState:pipe("h4_route")];
        [e setBuffer:Ym offset:0 atIndex:0];
        setb(e, d.moe_bias, 1);
        [e setBuffer:topi offset:0 atIndex:2];
        [e setBuffer:topw offset:0 atIndex:3];
        [e setBuffer:lat_frag offset:0 atIndex:4];
        [e setBuffer:mo_frag offset:0 atIndex:5];
        [e setBytes:&a length:sizeof a atIndex:6];
        [e dispatchThreadgroups:MTLSizeMake(W, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
        PlanArgs pa{W, d_topk, d_E};
        e = rc.op("d_plan");
        [e setComputePipelineState:pipe("h4_plan")];
        [e setBuffer:topi offset:0 atIndex:0];
        [e setBuffer:slots offset:0 atIndex:1];
        [e setBuffer:nslots offset:0 atIndex:2];
        [e setBuffer:slot_of offset:0 atIndex:3];
        [e setBytes:&pa length:sizeof pa atIndex:4];
        [e setBuffer:rmask offset:0 atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
      }
      const uint32_t dsmax = std::min(d_E, W * d_topk);
      mm(rc, "d_moe_gu", d.moe_gu, lat_frag, Ygu, W, slots, nslots, dsmax, 0, 16 * 2 * d_I, true, act, d_I, 1, true);
      if (moe_dn_had == 128) {
        WhtArgs wa{W, d_I};
        e = rc.op("d_wht128");
        [e setComputePipelineState:pipe("h4_wht128_dec")];
        [e setBuffer:act offset:0 atIndex:0];
        [e setBuffer:nslots offset:0 atIndex:1];
        [e setBuffer:rmask offset:0 atIndex:2];
        [e setBytes:&wa length:sizeof wa atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(d_I / 128, dsmax, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      }
      mm(rc, "d_moe_dn", d.moe_dn, act, yd, W, slots, nslots, dsmax, (d_I / 32) * 64, 16 * d_LAT, false, nil, 0, 1, true);
      {
        CombArgs a{W, d_topk, d_LAT, 16 * d_LAT};
        e = rc.op("d_combine");
        [e setComputePipelineState:pipe("h4_combine")];
        [e setBuffer:yd offset:0 atIndex:0];
        [e setBuffer:topi offset:0 atIndex:1];
        [e setBuffer:topw offset:0 atIndex:2];
        [e setBuffer:slot_of offset:0 atIndex:3];
        [e setBuffer:mo_frag offset:0 atIndex:4];
        [e setBytes:&a length:sizeof a atIndex:5];
        [e dispatchThreads:MTLSizeMake(d_LAT, W, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      mm(rc, "d_moe_out", d.moe_out, mo_frag, Yo, W);
      if (di + 1 < D.size()) {
        struct { uint32_t M, H; float eps; } pa{W, H, eps};
        e = rc.op("d_moe_post_ln");
        [e setComputePipelineState:pipe("h4_d_post_add_ln")];
        [e setBuffer:Yo offset:0 atIndex:0];
        setb(e, d.moe_norm, 1);
        [e setBuffer:dx offset:0 atIndex:2];
        setb(e, D[di + 1].in_norm, 3);
        [e setBuffer:xn_frag offset:0 atIndex:4];
        [e setBytes:&pa length:sizeof pa atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(W, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        xn_ready = true;
      } else {
        rms(Yo, W, H, 0, d.moe_norm, y, nil, 0, 0);
        add(dx, y, W, H, nullptr);
      }
    } else {
      mm(rc, "d_mlp_gu", d.gu, xn_frag, Ymlp, W, nil, nil, 0, 0, 0, true, mlp_frag, dff);
      mm(rc, "d_mlp_dn", d.dn, mlp_frag, Yo, W);
      add(dx, Yo, W, H, nullptr);
    }
  }
  rms(dx, W, H, 0, df_norm, dh, nil, 0, 0);
  // ---- drafts: hidden correction (+ optional AttentiveCorrection) from previous token, then draft head
  const bool gate = draft_cut > 0;
  ((uint32_t*)dl_row0_b.contents)[0] = 0;
  ((uint32_t*)dl_maxd_b.contents)[0] = W - 1;
  if (df_has_ac && df_ac_gated) {
    // Populate slot 0's block latent ac_blk[0] from [norm(dh[0]); norm(embed(anchor))] at pos p
    rms(dh, 1, H, 0, df_hchn, nil, dfrag2, 0, 2 * H);
    embed_rows(dids, 0, 1, de, false);
    rms(de, 1, H, 0, df_hcen, nil, dfrag2, H, 2 * H);
    mm(rc, "d_ac_in0", df_ac_in, dfrag2, Yin, 1);
    {
      KvArgs a{1, df_ac_in.N, 0, 0, 0, 0, 0, df_ac_in.rows.at("kva").first, eps, 1.0f, 0u};
      e = rc.op("d_ac_blk0");
      [e setComputePipelineState:pipe("h4_attn_in")];
      [e setBuffer:Yin offset:0 atIndex:0];
      setb(e, df_ac_qa_norm, 1); setb(e, df_ac_kva_norm, 2); setb(e, df_ac_qa_norm, 3);
      [e setBuffer:rope_draft offset:0 atIndex:4];
      [e setBuffer:qa_frag offset:0 atIndex:5];
      [e setBuffer:ac_blk offset:0 atIndex:6];
      [e setBuffer:iw offset:0 atIndex:7];
      [e setBuffer:iw offset:0 atIndex:8];
      [e setBytes:&a length:sizeof a atIndex:9];
      [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
  }
  for (uint32_t s2 = 0; s2 + 1 < W; s2++) {
    id<MTLBuffer> gb = dgate[gate ? s2 : 0];
    {
      struct { uint32_t n, H, s2; float eps; } ha{1, H, s2, eps};
      e = rc.op("d_hc_in");
      [e setComputePipelineState:pipe("h4_draft_hc_in")];
      [e setBuffer:dh offset:0 atIndex:0];
      setb(e, embed, 1); setb(e, df_hchn, 2); setb(e, df_hcen, 3);
      [e setBuffer:dl_row0_b offset:0 atIndex:4];
      if (s2 == 0) [e setBuffer:dids offset:0 atIndex:5];
      else [e setBuffer:dout offset:(s2 - 1) * 4 atIndex:5];
      [e setBuffer:dl_maxd_b offset:0 atIndex:6];
      [e setBuffer:gb offset:0 atIndex:7];
      [e setBuffer:dfrag2 offset:0 atIndex:8];
      [e setBytes:&ha length:sizeof ha atIndex:9];
      [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    if (gate) {
      mm(rc, "d_hc_gu", df_hcgu, dfrag2, Ymlp, 1, head_slots, gb, 1, 0, 0, true, mlp_frag, dhci, 3);
      mm(rc, "d_hc_dn", df_hcdn, mlp_frag, Yo, 1, head_slots, gb, 1);
    } else {
      mm(rc, "d_hc_gu", df_hcgu, dfrag2, Ymlp, 1, nil, nil, 0, 0, 0, true, mlp_frag, dhci);
      mm(rc, "d_hc_dn", df_hcdn, mlp_frag, Yo, 1);
    }
    if (df_has_ac) {
      const uint32_t slot = s2 + 1;
      const size_t rope_off = size_t(slot) * 64 * 4;
      Layer& t_ac = L[D[df_ac_src].tl];
      if (gate) {
        mm(rc, "d_ac_in", df_ac_in, dfrag2, Yin, 1, head_slots, gb, 1);
        if (df_ac_gated) mm(rc, "d_ac_bg", df_ac_bgate, dfrag2, ac_bg, 1, head_slots, gb, 1, 0, 0, false, nil, 0, 1, false, false);
      } else {
        mm(rc, "d_ac_in", df_ac_in, dfrag2, Yin, 1);
        if (df_ac_gated) mm(rc, "d_ac_bg", df_ac_bgate, dfrag2, ac_bg, 1, nil, nil, 0, 0, 0, false, nil, 0, 1, false, false);
      }
      {
        KvArgs a{1, df_ac_in.N, slot, 0, 0, 0, 0, df_ac_in.rows.at("kva").first, eps, 1.0f, 0u};
        e = rc.op("d_ac_in_k");
        [e setComputePipelineState:pipe("h4_attn_in")];
        [e setBuffer:Yin offset:0 atIndex:0];
        setb(e, df_ac_qa_norm, 1); setb(e, df_ac_kva_norm, 2); setb(e, df_ac_qa_norm, 3);
        [e setBuffer:rope_draft offset:rope_off atIndex:4];
        [e setBuffer:qa_frag offset:0 atIndex:5];
        [e setBuffer:ac_blk offset:0 atIndex:6];
        [e setBuffer:iw offset:0 atIndex:7];
        [e setBuffer:iw offset:0 atIndex:8];
        [e setBytes:&a length:sizeof a atIndex:9];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      if (gate) mm(rc, "d_ac_qb", df_ac_qb, qa_frag, Yq, 1, head_slots, gb, 1);
      else mm(rc, "d_ac_qb", df_ac_qb, qa_frag, Yq, 1);
      {
        QArgs a{1, df_ac_nh, df_ac_qb.N, 0, 16};
        e = rc.op("d_ac_qp");
        [e setComputePipelineState:pipe("h4_q_prep")];
        [e setBuffer:Yq offset:0 atIndex:0];
        [e setBuffer:rope_draft offset:rope_off atIndex:1];
        [e setBuffer:rope_draft offset:rope_off atIndex:2];
        [e setBuffer:qn_frag offset:0 atIndex:3];
        [e setBuffer:qr offset:0 atIndex:4];
        [e setBuffer:iq offset:0 atIndex:5];
        [e setBytes:&a length:sizeof a atIndex:6];
        [e dispatchThreadgroups:MTLSizeMake(df_ac_nh, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      }
      id<MTLBuffer> nsh_ac = df_ac_nh == 64 ? nslots64 : nslots32;
      const bool use_saved_s = (src_s == ~0u);
      id<MTLBuffer> sb = use_saved_s ? slots_[cur_slot].df_lsel[df_ac_src] : t_ac.lsel;
      id<MTLBuffer> nb = use_saved_s ? slots_[cur_slot].df_lnsel[df_ac_src] : t_ac.lnsel;
      const size_t soff = use_saved_s ? 0 : size_t(src_s) * selmax * 4;
      const size_t noff = use_saved_s ? 0 : size_t(src_s) * 4;
      id<MTLBuffer> nb_use = nb;
      size_t noff_use = noff;
      if (gate) {
        struct { uint32_t nh; } ga{df_ac_nh};
        e = rc.op("d_ac_gnsel", false);
        [e setComputePipelineState:pipe("h4_ac_gate_nsel")];
        [e setBuffer:nb offset:noff atIndex:0];
        [e setBuffer:gb offset:0 atIndex:1];
        [e setBuffer:dnsel offset:0 atIndex:2];
        [e setBuffer:ac_nh_gate offset:0 atIndex:3];
        [e setBytes:&ga length:sizeof ga atIndex:4];
        [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        nb_use = dnsel;
        noff_use = 0;
        nsh_ac = ac_nh_gate;
      }
      mm(rc, "d_ac_kb", df_ac_kb, qn_frag, qlat, 1, head_slots, nsh_ac, df_ac_nh, 16 * 192 / 8, 16 * 512);
      {
        AttnArgs a{1, df_ac_nh, p + slot, 0, window, 0, as_draft, selmax, 1.0f / 16.0f, 16, t_ac.kv8 ? 1u : 0u};
        e = rc.op("d_ac_attn");
        [e setComputePipelineState:pipe("h4_attn_mma_dsa")];
        [e setBuffer:qlat offset:0 atIndex:0];
        [e setBuffer:qr offset:0 atIndex:1];
        [e setBuffer:t_ac.lat offset:0 atIndex:2];
        [e setBuffer:sb offset:soff atIndex:3];
        [e setBuffer:nb_use offset:noff_use atIndex:4];
        [e setBuffer:part offset:0 atIndex:5];
        [e setBytes:&a length:sizeof a atIndex:6];
        setb(e, df_ac_sinks, 7);
        [e setBuffer:ofrag offset:0 atIndex:8];
        [e dispatchThreadgroups:MTLSizeMake(df_ac_nh / 16, 1, as_draft) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        MergeArgs ma{1, df_ac_nh, as_draft, 16};
        e = rc.op("d_ac_merge");
        [e setComputePipelineState:pipe("h4_attn_merge")];
        [e setBuffer:part offset:0 atIndex:0];
        setb(e, df_ac_sinks, 1);
        [e setBuffer:ofrag offset:0 atIndex:2];
        [e setBytes:&ma length:sizeof ma atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(df_ac_nh, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      }
      if (df_ac_gated) {
        struct { uint32_t M, nh, rs, slot, blk_stride; float scale; } ba{1, df_ac_nh, 16, slot, 16, 1.0f / 16.0f};
        e = rc.op("d_ac_blk");
        [e setComputePipelineState:pipe("h4_ac_block_attn")];
        [e setBuffer:qlat offset:0 atIndex:0];
        [e setBuffer:qr offset:0 atIndex:1];
        [e setBuffer:ac_blk offset:0 atIndex:2];
        [e setBuffer:ac_bg offset:0 atIndex:3];
        setb(e, df_ac_brel, 4); setb(e, df_ac_bsink, 5); setb(e, df_ac_bgbias, 6);
        [e setBuffer:ofrag offset:0 atIndex:7];
        [e setBuffer:gb offset:0 atIndex:8];
        [e setBytes:&ba length:sizeof ba atIndex:9];
        [e dispatchThreadgroups:MTLSizeMake(df_ac_nh, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      }
      mm(rc, "d_ac_vb", df_ac_vb, ofrag, yv, 1, head_slots, nsh_ac, df_ac_nh, 16 * 512 / 8, 16 * 256);
      {
        struct { uint32_t M, nh, rs; } ya{1, df_ac_nh, 16};
        e = rc.op("d_ac_yv");
        [e setComputePipelineState:pipe("h4_yv_to_frag")];
        [e setBuffer:yv offset:0 atIndex:0];
        [e setBuffer:gfrag offset:0 atIndex:1];
        [e setBuffer:gb offset:0 atIndex:2];
        [e setBytes:&ya length:sizeof ya atIndex:3];
        [e dispatchThreads:MTLSizeMake(df_ac_nh * 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      if (gate) mm(rc, "d_ac_o", df_ac_o, gfrag, y, 1, head_slots, gb, 1);
      else mm(rc, "d_ac_o", df_ac_o, gfrag, y, 1);
      {
        AddFArgs a{1, H, H, (s2 + 1) * H};
        e = rc.op("d_corr");
        [e setComputePipelineState:pipe("h4_add_frag2")];
        [e setBuffer:dh offset:0 atIndex:0];
        [e setBuffer:Yo offset:0 atIndex:1];
        [e setBuffer:y offset:0 atIndex:2];
        [e setBuffer:xn_frag offset:0 atIndex:3];
        [e setBytes:&a length:sizeof a atIndex:4];
        [e setBuffer:gb offset:0 atIndex:5];
        [e dispatchThreads:MTLSizeMake(H, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
    } else {
      AddFArgs a{1, H, H, (s2 + 1) * H};
      e = rc.op("d_corr");
      [e setComputePipelineState:pipe("h4_add_frag")];
      [e setBuffer:dh offset:0 atIndex:0];
      [e setBuffer:Yo offset:0 atIndex:1];
      [e setBuffer:xn_frag offset:0 atIndex:2];
      [e setBytes:&a length:sizeof a atIndex:3];
      [e setBuffer:gb offset:0 atIndex:4];
      [e dispatchThreads:MTLSizeMake(H, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    // a dead draft (gate 0) skips the head: one grouped slot whose count is the gate
    const Lin& dhd = cfg().dhead_w8 ? lm_head : lm_head_w4;
    if (gate) mm(rc, "d_head", dhd, xn_frag, logits, 1, head_slots, dgate[s2], 1);
    else mm(rc, "d_head", dhd, xn_frag, logits, 1);
    SampArgs sa{1, lm_head_w4.N, p + s2, uint32_t(seed), uint32_t(seed >> 32), temp > 0 ? 1.0f / temp : -1.0f, min_p > 0 ? logf(min_p) : -INFINITY, top_p, top_k};
    e = rc.op("d_argmax");
    [e setComputePipelineState:pipe("h4_pick_p")];
    [e setBuffer:logits offset:0 atIndex:0];
    [e setBuffer:dout offset:s2 * 4 atIndex:1];
    [e setBytes:&sa length:sizeof sa atIndex:2];
    [e setBuffer:dprob offset:s2 * 4 atIndex:3];
    [e setBuffer:dpmax offset:s2 * 4 atIndex:4];
    [e setBuffer:dgate[gate ? s2 : 0] offset:0 atIndex:5];
    [e setBuffer:dmarg offset:s2 * 4 atIndex:6];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    if (gate && s2 + 2 < W) {
      struct { uint32_t n; float cut; } ga{s2 + 1, draft_cut};
      e = rc.op("d_gate");
      [e setComputePipelineState:pipe("h4_dgate")];
      [e setBuffer:dprob offset:0 atIndex:0];
      [e setBuffer:dgate[s2 + 1] offset:0 atIndex:1];
      [e setBytes:&ga length:sizeof ga atIndex:2];
      [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }
  }
  if (rc.e) [rc.e endEncoding];
  [rc.cb commit];
  const double h_commit = host_now();
  [rc.cb waitUntilCompleted];
  const double h_wake = host_now();
  if (rc.cb.error) mtl::die("draft: command buffer failed", rc.cb.error);
  if (gpu_ms) *gpu_ms = mtl::gpu_seconds(rc.cb) * 1e3;
  last_dlat = {(h_commit - h_entry) * 1e3, (h_commit - h_entry) * 1e3, (rc.cb.GPUStartTime - h_commit) * 1e3,
               (rc.cb.GPUEndTime - rc.cb.GPUStartTime) * 1e3, (h_wake - rc.cb.GPUEndTime) * 1e3};
  if (profile) rc.resolve(*profile);
  std::vector<uint32_t> out(W - 1);
  memcpy(out.data(), dout.contents, (W - 1) * 4);
  last_probs.assign((const float*)dprob.contents, (const float*)dprob.contents + (W - 1));
  last_pmax.assign((const float*)dpmax.contents, (const float*)dpmax.contents + (W - 1));
  last_marg.assign((const float*)dmarg.contents, (const float*)dmarg.contents + (W - 1));
  // verify only the drafts whose cumulative probability stays above draft_cut (at least one),
  // then optionally trim unprofitable tail drafts via the fan-out-aware tok/ms selector.
  if (cut) {
    size_t k_gpu = out.size();
    if (draft_cut > 0) {
      float c = 1.0f;  // f32 in h4_dgate's order, so the host verifies only drafts the GPU finished
      size_t k = 0;
      while (k < out.size() && c * last_probs[k] >= draft_cut) { c *= last_probs[k]; k++; }
      k_gpu = std::max<size_t>(k, 1);
    }
    const size_t keep = select_draft_k(last_probs, k_gpu, 1);
    out.resize(keep);
  }
  return out;
}

std::vector<Model::DraftRes> Model::draft_lanes(const std::vector<DraftReq>& reqs, double* gpu_ms, bool cut) {
  if (!has_dfly) mtl::die("draft_lanes: pack has no DFly2 drafter");
  const uint32_t n = uint32_t(reqs.size());
  std::vector<DraftRes> res(n);
  if (n == 0) return res;
  ensure_lane_pipes();
  if (lane_tab.empty()) rebuild_lane_tables();

  struct DraftEhRowHost { uint64_t xf; uint32_t dx_row, xf_row; };
  struct DselLaneRowHost { uint64_t tsel, tnsel; uint32_t src, anchor, q, _pad; };
  struct SampRow { uint32_t N, pos, s0, s1; float inv_t, lmin, top_p; uint32_t top_k; };
  static_assert(sizeof(DraftEhRowHost) == 16 && sizeof(DselLaneRowHost) == 32 && sizeof(SampRow) == 32);

  std::vector<uint32_t> dids_v, rlane, rpos, rope_rpos, row0(n), maxd(n), anchors(n), lane_p(n);
  std::vector<DraftEhRowHost> eh_rows(n);
  uint32_t max_D = 0;
  for (uint32_t j = 0; j < n; j++) {
    const auto& rq = reqs[j];
    const uint32_t sp = (rq.slot == cur_slot) ? pos : slots_[rq.slot].pos;
    const uint32_t scap = (rq.slot == cur_slot) ? cap : slots_[rq.slot].cap;
    const uint32_t md = (sp + 2 <= scap) ? std::min({rq.max_drafts, dblock - 1, scap - sp - 1}) : 0u;
    const uint32_t Wj = md + 1;
    row0[j] = uint32_t(dids_v.size());
    maxd[j] = md;
    anchors[j] = rq.anchor;
    lane_p[j] = sp;
    max_D = std::max(max_D, md);
    const bool use_saved = (rq.src_x == ~0u);
    eh_rows[j] = DraftEhRowHost{use_saved ? slots_[rq.slot].df_xf.gpuAddress : xf.gpuAddress, row0[j], use_saved ? 0u : rq.src_x};
    const int32_t sdelta = (rq.slot == cur_slot) ? mrope_delta : slots_[rq.slot].mrope_delta;
    for (uint32_t q = 0; q < Wj; q++) {
      dids_v.push_back(q == 0 ? rq.anchor : mask_id);
      rlane.push_back(uint32_t(rq.slot));
      rpos.push_back(sp + q);
      rope_rpos.push_back(uint32_t(int64_t(sp + q) + sdelta));
    }
  }
  const uint32_t M = uint32_t(dids_v.size());
  if (M == 0 || M > MAXM || max_D == 0) return res;

  memcpy(dids.contents, dids_v.data(), M * 4);
  memcpy(row_lane_b.contents, rlane.data(), M * 4);
  memcpy(row_pos_b.contents, rpos.data(), M * 4);
  memcpy(dl_eh_b.contents, eh_rows.data(), n * sizeof(DraftEhRowHost));
  memcpy(dl_row0_b.contents, row0.data(), n * 4);
  memcpy(dl_maxd_b.contents, maxd.data(), n * 4);
  memcpy(dl_anchor_b.contents, anchors.data(), n * 4);
  rope_rows(inv_dsa, rope_rpos, (float*)rope_draft.contents);

  // Per-DFly-layer dsel row descriptors
  for (size_t di = 0; di < D.size(); di++) {
    const Layer& t = L[D[di].tl];
    DselLaneRowHost* dr = (DselLaneRowHost*)dl_dsel_b.contents + di * MAXM;
    uint32_t m = 0;
    for (uint32_t j = 0; j < n; j++) {
      const auto& rq = reqs[j];
      const bool use_saved = (rq.src_s == ~0u);
      const uint64_t ts = use_saved ? slots_[rq.slot].df_lsel[di].gpuAddress : t.lsel.gpuAddress;
      const uint64_t tn = use_saved ? slots_[rq.slot].df_lnsel[di].gpuAddress : t.lnsel.gpuAddress;
      const uint32_t sr = use_saved ? 0u : rq.src_s;
      for (uint32_t q = 0; q <= maxd[j]; q++)
        dr[m++] = DselLaneRowHost{ts, tn, sr, lane_p[j], q, 0};
    }
  }
  if (df_has_ac) {
    const Layer& t_ac = L[D[df_ac_src].tl];
    DselLaneRowHost* ar = (DselLaneRowHost*)dl_ac_sel_b.contents;
    uint32_t* al = (uint32_t*)dl_ac_rlane.contents;
    for (uint32_t j = 0; j < n; j++) {
      const auto& rq = reqs[j];
      const bool use_saved = (rq.src_s == ~0u);
      const uint64_t ts = use_saved ? slots_[rq.slot].df_lsel[df_ac_src].gpuAddress : t_ac.lsel.gpuAddress;
      const uint64_t tn = use_saved ? slots_[rq.slot].df_lnsel[df_ac_src].gpuAddress : t_ac.lnsel.gpuAddress;
      ar[j] = DselLaneRowHost{ts, tn, use_saved ? 0u : rq.src_s, lane_p[j], 0, 0};
      al[j] = uint32_t(rq.slot);
    }
    // Pre-pack [slot][lane][64] RoPE table for slots 0 .. max_D
    for (uint32_t slot = 0; slot <= max_D; slot++) {
      std::vector<uint32_t> sp(n);
      for (uint32_t j = 0; j < n; j++) {
        const auto& rq = reqs[j];
        const int32_t sdelta = (rq.slot == cur_slot) ? mrope_delta : slots_[rq.slot].mrope_delta;
        sp[j] = uint32_t(int64_t(lane_p[j] + slot) + sdelta);
      }
      rope_rows(inv_dsa, sp, (float*)dl_ac_rope.contents + size_t(slot) * n * 64);
    }
  }
  // Per-(step, lane) SampRow descriptors
  for (uint32_t s2 = 0; s2 < max_D; s2++) {
    SampRow* sr = (SampRow*)dl_samp_b.contents + size_t(s2) * MAXM;
    for (uint32_t j = 0; j < n; j++) {
      const auto& rq = reqs[j];
      sr[j] = SampRow{lm_head_w4.N, lane_p[j] + s2, uint32_t(rq.seed), uint32_t(rq.seed >> 32),
                      rq.temp > 0 ? 1.0f / rq.temp : -1.0f, rq.min_p > 0 ? logf(rq.min_p) : -INFINITY,
                      rq.top_p, rq.top_k};
    }
  }

  Rec rc;
  rc.cb = [C.q commandBuffer];
  id<MTLComputeCommandEncoder> e;
  auto rms = [&](id<MTLBuffer> X, uint32_t MR_, const Raw& w, id<MTLBuffer> Yf, id<MTLBuffer> frag) {
    RmsArgs a{MR_, H, H, 0, H, Yf ? 1u : 0u, frag ? 1u : 0u, eps, 0, 0};
    e = rc.op("dl_rms");
    [e setComputePipelineState:pipe("h4_rms")];
    [e setBuffer:X offset:0 atIndex:0];
    setb(e, w, 1);
    [e setBuffer:(Yf ? Yf : y) offset:0 atIndex:2];
    [e setBuffer:(frag ? frag : xn_frag) offset:0 atIndex:3];
    [e setBytes:&a length:sizeof a atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(MR_, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  };
  auto add = [&](id<MTLBuffer> X, id<MTLBuffer> Yv, uint32_t MR_) {
    AddArgs a{MR_, H, H, 0};
    e = rc.op("dl_add");
    [e setComputePipelineState:pipe("h4_add")];
    [e setBuffer:X offset:0 atIndex:0];
    [e setBuffer:Yv offset:0 atIndex:1];
    [e setBuffer:Yv offset:0 atIndex:2];
    [e setBytes:&a length:sizeof a atIndex:3];
    [e dispatchThreads:MTLSizeMake(H, MR_, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  };

  // ---- Step A: block embeddings + batched slot-0 fusion across all n lanes
  {
    struct { uint32_t M, H, HC; } a{M, H, 1};
    e = rc.op("dl_embed");
    [e setComputePipelineState:pipe("h4_embed")];
    [e setBuffer:dids offset:0 atIndex:0];
    setb(e, embed, 1);
    [e setBuffer:dx offset:0 atIndex:2];
    [e setBytes:&a length:sizeof a atIndex:3];
    [e dispatchThreads:MTLSizeMake(H, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
  {
    struct { uint32_t n, H; float eps; } a{n, H, eps};
    e = rc.op("dl_eh_rms");
    [e setComputePipelineState:pipe("h4_draft_eh_rms")];
    [e setBuffer:dx offset:0 atIndex:0];
    setb(e, df_enorm, 1); setb(e, df_hnorm, 2);
    [e setBuffer:dfrag2 offset:0 atIndex:3];
    [e setBuffer:dl_eh_b offset:0 atIndex:4];
    [e setBytes:&a length:sizeof a atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
  mm(rc, "dl_eh", df_eh, dfrag2, Yo, n);
  {
    struct { uint32_t n, H; } a{n, H};
    e = rc.op("dl_eh_add");
    [e setComputePipelineState:pipe("h4_draft_eh_add")];
    [e setBuffer:dx offset:0 atIndex:0];
    [e setBuffer:Yo offset:0 atIndex:1];
    setb(e, df_gate, 2);
    [e setBuffer:dl_row0_b offset:0 atIndex:3];
    [e setBytes:&a length:sizeof a atIndex:4];
    [e dispatchThreads:MTLSizeMake(H, n, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }

  // ---- Step B: DFly2 layers over all M <= 16 rows at once
  bool dl_xn_ready = false;
  for (size_t di = 0; di < D.size(); di++) {
    auto& d = D[di];
    Layer& t = L[d.tl];
    if (!dl_xn_ready) rms(dx, M, d.in_norm, nil, xn_frag);
    dl_xn_ready = false;
    mm(rc, "dl_attn_in", d.in, xn_frag, Yin, M);
    const bool ov_dgate = d.in.ksplit == 1 || d.gate.ksplit == 1;
    mm(rc, "dl_gate", d.gate, xn_frag, Yg, M, nil, nil, 0, 0, 0, false, nil, 0, 1, false, !ov_dgate);
    {
      KvArgs a{M, d.in.N, 0, 0, 0, 0, 0, d.in.rows.at("kva").first, eps, 1.0f, t.kv8 ? 1u : 0u};
      e = rc.op("dl_attn_in_k");
      [e setComputePipelineState:pipe("h4_attn_in_lanes")];
      [e setBuffer:Yin offset:0 atIndex:0];
      setb(e, d.qa_norm, 1); setb(e, d.kva_norm, 2); setb(e, d.qa_norm, 3);
      [e setBuffer:rope_draft offset:0 atIndex:4];
      [e setBuffer:qa_frag offset:0 atIndex:5];
      [e setBuffer:t.lat offset:0 atIndex:6];
      [e setBuffer:iw offset:0 atIndex:7];
      [e setBuffer:iw offset:0 atIndex:8];
      [e setBytes:&a length:sizeof a atIndex:9];
      [e setBuffer:lane_tab[d.tl] offset:0 atIndex:12];
      [e setBuffer:row_lane_b offset:0 atIndex:13];
      [e setBuffer:row_pos_b offset:0 atIndex:14];
      [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    mm(rc, "dl_qb", d.qb, qa_frag, Yq, M);
    {
      QArgs a{M, d.nh, d.qb.N, 0, 16};
      e = rc.op("dl_q_prep");
      [e setComputePipelineState:pipe("h4_q_prep")];
      [e setBuffer:Yq offset:0 atIndex:0];
      [e setBuffer:rope_draft offset:0 atIndex:1];
      [e setBuffer:rope_draft offset:0 atIndex:2];
      [e setBuffer:qn_frag offset:0 atIndex:3];
      [e setBuffer:qr offset:0 atIndex:4];
      [e setBuffer:iq offset:0 atIndex:5];
      [e setBytes:&a length:sizeof a atIndex:6];
      [e dispatchThreadgroups:MTLSizeMake(d.nh, M, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
    id<MTLBuffer> nsh = d.nh == 64 ? nslots64 : nslots32;
    mm(rc, "dl_kb", d.kb, qn_frag, qlat, M, head_slots, nsh, d.nh, 16 * 192 / 8, 16 * 512);
    {
      struct { uint32_t M, selmax_t, selmax_d; } a{M, selmax, DSELMAX};
      e = rc.op("dl_dsel", false);
      [e setComputePipelineState:pipe("h4_dsel_lanes")];
      [e setBuffer:dl_dsel_b offset:di * MAXM * sizeof(DselLaneRowHost) atIndex:0];
      [e setBuffer:dsel offset:0 atIndex:1];
      [e setBuffer:dnsel offset:0 atIndex:2];
      [e setBytes:&a length:sizeof a atIndex:3];
      [e dispatchThreads:MTLSizeMake(DSELMAX, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    {
      AttnArgs a{M, d.nh, 0, 0, window, 0, as_draft, DSELMAX, 1.0f / 16.0f, 16, t.kv8 ? 1u : 0u};
      e = rc.op("dl_attn");
      [e setComputePipelineState:pipe("h4_attn_mma_dsa_lanes")];
      [e setBuffer:qlat offset:0 atIndex:0];
      [e setBuffer:qr offset:0 atIndex:1];
      [e setBuffer:t.lat offset:0 atIndex:2];
      [e setBuffer:dsel offset:0 atIndex:3];
      [e setBuffer:dnsel offset:0 atIndex:4];
      [e setBuffer:part offset:0 atIndex:5];
      [e setBytes:&a length:sizeof a atIndex:6];
      setb(e, d.sinks, 7);
      [e setBuffer:ofrag offset:0 atIndex:8];
      [e setBuffer:lane_tab[d.tl] offset:0 atIndex:12];
      [e setBuffer:row_lane_b offset:0 atIndex:13];
      [e setBuffer:row_pos_b offset:0 atIndex:14];
      [e dispatchThreadgroups:MTLSizeMake(d.nh / 16, M, as_draft) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      MergeArgs ma{M, d.nh, as_draft, 16};
      e = rc.op("dl_attn_merge");
      [e setComputePipelineState:pipe("h4_attn_merge")];
      [e setBuffer:part offset:0 atIndex:0];
      setb(e, d.sinks, 1);
      [e setBuffer:ofrag offset:0 atIndex:2];
      [e setBytes:&ma length:sizeof ma atIndex:3];
      [e dispatchThreadgroups:MTLSizeMake(d.nh, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    if (cfg().fuse_gate) {
      mm(rc, "dl_vb", d.vb, ofrag, yv, M, head_slots, nsh, d.nh, 16 * 512 / 8, 16 * 256, false, gfrag, d.nh * 256, 1, false, true, Yg);
    } else {
      mm(rc, "dl_vb", d.vb, ofrag, yv, M, head_slots, nsh, d.nh, 16 * 512 / 8, 16 * 256);
      {
        GateArgs a{M, d.nh, 16};
        e = rc.op("dl_gate_mul");
        [e setComputePipelineState:pipe("h4_gate")];
        [e setBuffer:yv offset:0 atIndex:0];
        [e setBuffer:Yg offset:0 atIndex:1];
        [e setBuffer:gfrag offset:0 atIndex:2];
        [e setBytes:&a length:sizeof a atIndex:3];
        [e dispatchThreads:MTLSizeMake(d.nh * 256, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
    }
    mm(rc, "dl_o", d.o, gfrag, Yo, M);
    {
      struct { uint32_t M, H; float eps; } pa{M, H, eps};
      e = rc.op("dl_post_add_ln");
      [e setComputePipelineState:pipe("h4_d_post_add_ln")];
      [e setBuffer:Yo offset:0 atIndex:0];
      setb(e, d.post_norm, 1);
      [e setBuffer:dx offset:0 atIndex:2];
      setb(e, d.post_ln, 3);
      [e setBuffer:xn_frag offset:0 atIndex:4];
      [e setBytes:&pa length:sizeof pa atIndex:5];
      [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    if (d.moe) {
      mm(rc, "dl_moe_in", d.moe_in, xn_frag, Ym, M);
      {
        RouteArgs a{M, d_E, d_topk, d.moe_in.N, d_LAT, d_I, d_rscale};
        e = rc.op("dl_route");
        [e setComputePipelineState:pipe("h4_route")];
        [e setBuffer:Ym offset:0 atIndex:0];
        setb(e, d.moe_bias, 1);
        [e setBuffer:topi offset:0 atIndex:2];
        [e setBuffer:topw offset:0 atIndex:3];
        [e setBuffer:lat_frag offset:0 atIndex:4];
        [e setBuffer:mo_frag offset:0 atIndex:5];
        [e setBytes:&a length:sizeof a atIndex:6];
        [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
        PlanArgs pa{M, d_topk, d_E};
        e = rc.op("dl_plan");
        [e setComputePipelineState:pipe("h4_plan")];
        [e setBuffer:topi offset:0 atIndex:0];
        [e setBuffer:slots offset:0 atIndex:1];
        [e setBuffer:nslots offset:0 atIndex:2];
        [e setBuffer:slot_of offset:0 atIndex:3];
        [e setBytes:&pa length:sizeof pa atIndex:4];
        [e setBuffer:rmask offset:0 atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
      }
      const uint32_t dsmax = std::min(d_E, M * d_topk);
      mm(rc, "dl_moe_gu", d.moe_gu, lat_frag, Ygu, M, slots, nslots, dsmax, 0, 16 * 2 * d_I, true, act, d_I, 1, true);
      if (moe_dn_had == 128) {
        WhtArgs wa{M, d_I};
        e = rc.op("dl_wht128");
        [e setComputePipelineState:pipe("h4_wht128_dec")];
        [e setBuffer:act offset:0 atIndex:0];
        [e setBuffer:nslots offset:0 atIndex:1];
        [e setBuffer:rmask offset:0 atIndex:2];
        [e setBytes:&wa length:sizeof wa atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(d_I / 128, dsmax, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      }
      mm(rc, "dl_moe_dn", d.moe_dn, act, yd, M, slots, nslots, dsmax, (d_I / 32) * 64, 16 * d_LAT, false, nil, 0, 1, true);
      {
        CombArgs a{M, d_topk, d_LAT, 16 * d_LAT};
        e = rc.op("dl_combine");
        [e setComputePipelineState:pipe("h4_combine")];
        [e setBuffer:yd offset:0 atIndex:0];
        [e setBuffer:topi offset:0 atIndex:1];
        [e setBuffer:topw offset:0 atIndex:2];
        [e setBuffer:slot_of offset:0 atIndex:3];
        [e setBuffer:mo_frag offset:0 atIndex:4];
        [e setBytes:&a length:sizeof a atIndex:5];
        [e dispatchThreads:MTLSizeMake(d_LAT, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      mm(rc, "dl_moe_out", d.moe_out, mo_frag, Yo, M);
      if (di + 1 < D.size()) {
        struct { uint32_t M, H; float eps; } pa{M, H, eps};
        e = rc.op("dl_moe_post_ln");
        [e setComputePipelineState:pipe("h4_d_post_add_ln")];
        [e setBuffer:Yo offset:0 atIndex:0];
        setb(e, d.moe_norm, 1);
        [e setBuffer:dx offset:0 atIndex:2];
        setb(e, D[di + 1].in_norm, 3);
        [e setBuffer:xn_frag offset:0 atIndex:4];
        [e setBytes:&pa length:sizeof pa atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        dl_xn_ready = true;
      } else {
        rms(Yo, M, d.moe_norm, y, nil);
        add(dx, y, M);
      }
    } else {
      mm(rc, "dl_mlp_gu", d.gu, xn_frag, Ymlp, M, nil, nil, 0, 0, 0, true, mlp_frag, dff);
      mm(rc, "dl_mlp_dn", d.dn, mlp_frag, Yo, M);
      add(dx, Yo, M);
    }
  }
  rms(dx, M, df_norm, dh, nil);

  // ---- Step C: batched hidden correction (+ optional AttentiveCorrection) + draft head across all n lanes
  const bool gate = draft_cut > 0;
  const Lin& dhd = cfg().dhead_w8 ? lm_head : lm_head_w4;
  if (df_has_ac && df_ac_gated) {
    struct { uint32_t n, H; float eps; } a0{n, H, eps};
    e = rc.op("dl_ac_blk0_in");
    [e setComputePipelineState:pipe("h4_ac_blk0_in")];
    [e setBuffer:dh offset:0 atIndex:0];
    setb(e, embed, 1); setb(e, df_hchn, 2); setb(e, df_hcen, 3);
    [e setBuffer:dl_row0_b offset:0 atIndex:4];
    [e setBuffer:dl_anchor_b offset:0 atIndex:5];
    [e setBuffer:dfrag2 offset:0 atIndex:6];
    [e setBytes:&a0 length:sizeof a0 atIndex:7];
    [e dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    mm(rc, "dl_ac_in0", df_ac_in, dfrag2, Yin, n);
    {
      struct { uint32_t M, Nin, kvoff, slot, blk_stride; float eps; } ia{n, df_ac_in.N, df_ac_in.rows.at("kva").first, 0, 16, eps};
      e = rc.op("dl_ac_blk0");
      [e setComputePipelineState:pipe("h4_ac_in_lanes")];
      [e setBuffer:Yin offset:0 atIndex:0];
      setb(e, df_ac_qa_norm, 1); setb(e, df_ac_kva_norm, 2);
      [e setBuffer:dl_ac_rope offset:0 atIndex:3];
      [e setBuffer:qa_frag offset:0 atIndex:4];
      [e setBuffer:ac_blk offset:0 atIndex:5];
      [e setBytes:&ia length:sizeof ia atIndex:6];
      [e dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
  }
  for (uint32_t s2 = 0; s2 < max_D; s2++) {
    id<MTLBuffer> gb = dgate[gate ? s2 : 0];
    {
      struct { uint32_t n, H, s2; float eps; } a{n, H, s2, eps};
      e = rc.op("dl_hc_in");
      [e setComputePipelineState:pipe("h4_draft_hc_in")];
      [e setBuffer:dh offset:0 atIndex:0];
      setb(e, embed, 1); setb(e, df_hchn, 2); setb(e, df_hcen, 3);
      [e setBuffer:dl_row0_b offset:0 atIndex:4];
      if (s2 == 0) [e setBuffer:dl_anchor_b offset:0 atIndex:5];
      else [e setBuffer:dout offset:size_t(s2 - 1) * MAXM * 4 atIndex:5];
      [e setBuffer:dl_maxd_b offset:0 atIndex:6];
      [e setBuffer:gb offset:0 atIndex:7];
      [e setBuffer:dfrag2 offset:0 atIndex:8];
      [e setBytes:&a length:sizeof a atIndex:9];
      [e dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    if (gate) {
      mm(rc, "dl_hc_gu", df_hcgu, dfrag2, Ymlp, n, head_slots, gb, 1, 0, 0, true, mlp_frag, dhci, 3);
      mm(rc, "dl_hc_dn", df_hcdn, mlp_frag, Yo, n, head_slots, gb, 1);
    } else {
      mm(rc, "dl_hc_gu", df_hcgu, dfrag2, Ymlp, n, nil, nil, 0, 0, 0, true, mlp_frag, dhci);
      mm(rc, "dl_hc_dn", df_hcdn, mlp_frag, Yo, n);
    }
    if (df_has_ac) {
      const uint32_t slot = s2 + 1;
      const size_t rope_off = size_t(slot) * n * 64 * 4;
      const uint32_t tl_ac = D[df_ac_src].tl;
      Layer& t_ac = L[tl_ac];
      if (gate) {
        mm(rc, "dl_ac_in", df_ac_in, dfrag2, Yin, n, head_slots, gb, 1);
        if (df_ac_gated) mm(rc, "dl_ac_bg", df_ac_bgate, dfrag2, ac_bg, n, head_slots, gb, 1, 0, 0, false, nil, 0, 1, false, false);
      } else {
        mm(rc, "dl_ac_in", df_ac_in, dfrag2, Yin, n);
        if (df_ac_gated) mm(rc, "dl_ac_bg", df_ac_bgate, dfrag2, ac_bg, n, nil, nil, 0, 0, 0, false, nil, 0, 1, false, false);
      }
      {
        struct { uint32_t M, Nin, kvoff, slot, blk_stride; float eps; } ia{n, df_ac_in.N, df_ac_in.rows.at("kva").first, slot, 16, eps};
        e = rc.op("dl_ac_in_k");
        [e setComputePipelineState:pipe("h4_ac_in_lanes")];
        [e setBuffer:Yin offset:0 atIndex:0];
        setb(e, df_ac_qa_norm, 1); setb(e, df_ac_kva_norm, 2);
        [e setBuffer:dl_ac_rope offset:rope_off atIndex:3];
        [e setBuffer:qa_frag offset:0 atIndex:4];
        [e setBuffer:ac_blk offset:0 atIndex:5];
        [e setBytes:&ia length:sizeof ia atIndex:6];
        [e dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      if (gate) mm(rc, "dl_ac_qb", df_ac_qb, qa_frag, Yq, n, head_slots, gb, 1);
      else mm(rc, "dl_ac_qb", df_ac_qb, qa_frag, Yq, n);
      {
        QArgs a{n, df_ac_nh, df_ac_qb.N, 0, 16};
        e = rc.op("dl_ac_qp");
        [e setComputePipelineState:pipe("h4_q_prep")];
        [e setBuffer:Yq offset:0 atIndex:0];
        [e setBuffer:dl_ac_rope offset:rope_off atIndex:1];
        [e setBuffer:dl_ac_rope offset:rope_off atIndex:2];
        [e setBuffer:qn_frag offset:0 atIndex:3];
        [e setBuffer:qr offset:0 atIndex:4];
        [e setBuffer:iq offset:0 atIndex:5];
        [e setBytes:&a length:sizeof a atIndex:6];
        [e dispatchThreadgroups:MTLSizeMake(df_ac_nh, n, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      }
      {
        struct { uint32_t M, selmax_t, selmax_d, s2, nh; } sa{n, selmax, DSELMAX, s2, df_ac_nh};
        e = rc.op("dl_ac_sel", false);
        [e setComputePipelineState:pipe("h4_ac_sel_lanes")];
        [e setBuffer:dl_ac_sel_b offset:0 atIndex:0];
        [e setBuffer:dsel offset:0 atIndex:1];
        [e setBuffer:dnsel offset:0 atIndex:2];
        [e setBuffer:gb offset:0 atIndex:3];
        [e setBuffer:dl_maxd_b offset:0 atIndex:4];
        [e setBuffer:ac_nh_gate offset:0 atIndex:5];
        [e setBytes:&sa length:sizeof sa atIndex:6];
        [e dispatchThreads:MTLSizeMake(selmax, n, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      id<MTLBuffer> nsh_ac = gate ? ac_nh_gate : (df_ac_nh == 64 ? nslots64 : nslots32);
      mm(rc, "dl_ac_kb", df_ac_kb, qn_frag, qlat, n, head_slots, nsh_ac, df_ac_nh, 16 * 192 / 8, 16 * 512);
      {
        AttnArgs a{n, df_ac_nh, 0, 0, window, 0, as_draft, DSELMAX, 1.0f / 16.0f, 16, t_ac.kv8 ? 1u : 0u};
        e = rc.op("dl_ac_attn");
        [e setComputePipelineState:pipe("h4_attn_mma_dsa_lanes")];
        [e setBuffer:qlat offset:0 atIndex:0];
        [e setBuffer:qr offset:0 atIndex:1];
        [e setBuffer:t_ac.lat offset:0 atIndex:2];
        [e setBuffer:dsel offset:0 atIndex:3];
        [e setBuffer:dnsel offset:0 atIndex:4];
        [e setBuffer:part offset:0 atIndex:5];
        [e setBytes:&a length:sizeof a atIndex:6];
        setb(e, df_ac_sinks, 7);
        [e setBuffer:ofrag offset:0 atIndex:8];
        [e setBuffer:lane_tab[tl_ac] offset:0 atIndex:12];
        [e setBuffer:dl_ac_rlane offset:0 atIndex:13];
        [e setBuffer:row_pos_b offset:0 atIndex:14];
        [e dispatchThreadgroups:MTLSizeMake(df_ac_nh / 16, n, as_draft) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        MergeArgs ma{n, df_ac_nh, as_draft, 16};
        e = rc.op("dl_ac_merge");
        [e setComputePipelineState:pipe("h4_attn_merge")];
        [e setBuffer:part offset:0 atIndex:0];
        setb(e, df_ac_sinks, 1);
        [e setBuffer:ofrag offset:0 atIndex:2];
        [e setBytes:&ma length:sizeof ma atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(df_ac_nh, n, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      }
      if (df_ac_gated) {
        struct { uint32_t M, nh, rs, slot, blk_stride; float scale; } ba{n, df_ac_nh, 16, slot, 16, 1.0f / 16.0f};
        e = rc.op("dl_ac_blk");
        [e setComputePipelineState:pipe("h4_ac_block_attn")];
        [e setBuffer:qlat offset:0 atIndex:0];
        [e setBuffer:qr offset:0 atIndex:1];
        [e setBuffer:ac_blk offset:0 atIndex:2];
        [e setBuffer:ac_bg offset:0 atIndex:3];
        setb(e, df_ac_brel, 4); setb(e, df_ac_bsink, 5); setb(e, df_ac_bgbias, 6);
        [e setBuffer:ofrag offset:0 atIndex:7];
        [e setBuffer:gb offset:0 atIndex:8];
        [e setBytes:&ba length:sizeof ba atIndex:9];
        [e dispatchThreadgroups:MTLSizeMake(df_ac_nh, n, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      }
      mm(rc, "dl_ac_vb", df_ac_vb, ofrag, yv, n, head_slots, nsh_ac, df_ac_nh, 16 * 512 / 8, 16 * 256);
      {
        struct { uint32_t M, nh, rs; } ya{n, df_ac_nh, 16};
        e = rc.op("dl_ac_yv");
        [e setComputePipelineState:pipe("h4_yv_to_frag")];
        [e setBuffer:yv offset:0 atIndex:0];
        [e setBuffer:gfrag offset:0 atIndex:1];
        [e setBuffer:gb offset:0 atIndex:2];
        [e setBytes:&ya length:sizeof ya atIndex:3];
        [e dispatchThreads:MTLSizeMake(df_ac_nh * 256, n, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      if (gate) mm(rc, "dl_ac_o", df_ac_o, gfrag, y, n, head_slots, gb, 1);
      else mm(rc, "dl_ac_o", df_ac_o, gfrag, y, n);
      {
        struct { uint32_t n, H, s2; } a{n, H, s2};
        e = rc.op("dl_corr");
        [e setComputePipelineState:pipe("h4_draft_corr2")];
        [e setBuffer:dh offset:0 atIndex:0];
        [e setBuffer:Yo offset:0 atIndex:1];
        [e setBuffer:y offset:0 atIndex:2];
        [e setBuffer:dl_row0_b offset:0 atIndex:3];
        [e setBuffer:dl_maxd_b offset:0 atIndex:4];
        [e setBuffer:gb offset:0 atIndex:5];
        [e setBuffer:xn_frag offset:0 atIndex:6];
        [e setBytes:&a length:sizeof a atIndex:7];
        [e dispatchThreads:MTLSizeMake(H, n, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
    } else {
      struct { uint32_t n, H, s2; } a{n, H, s2};
      e = rc.op("dl_corr");
      [e setComputePipelineState:pipe("h4_draft_corr")];
      [e setBuffer:dh offset:0 atIndex:0];
      [e setBuffer:Yo offset:0 atIndex:1];
      [e setBuffer:dl_row0_b offset:0 atIndex:2];
      [e setBuffer:dl_maxd_b offset:0 atIndex:3];
      [e setBuffer:gb offset:0 atIndex:4];
      [e setBuffer:xn_frag offset:0 atIndex:5];
      [e setBytes:&a length:sizeof a atIndex:6];
      [e dispatchThreads:MTLSizeMake(H, n, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    if (gate) mm(rc, "dl_head", dhd, xn_frag, logits, n, head_slots, gb, 1);
    else mm(rc, "dl_head", dhd, xn_frag, logits, n);
    {
      e = rc.op("dl_pick");
      [e setComputePipelineState:pipe("h4_pick_p_lanes")];
      [e setBuffer:logits offset:0 atIndex:0];
      [e setBuffer:dout offset:size_t(s2) * MAXM * 4 atIndex:1];
      [e setBuffer:dl_samp_b offset:size_t(s2) * MAXM * sizeof(SampRow) atIndex:2];
      [e setBuffer:dprob offset:size_t(s2) * MAXM * 4 atIndex:3];
      [e setBuffer:dpmax offset:size_t(s2) * MAXM * 4 atIndex:4];
      [e setBuffer:gb offset:0 atIndex:5];
      [e setBuffer:dl_maxd_b offset:0 atIndex:6];
      [e setBytes:&s2 length:4 atIndex:7];
      [e setBuffer:dmarg offset:size_t(s2) * MAXM * 4 atIndex:8];
      [e dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    }
    if (gate && s2 + 1 < max_D) {
      struct { uint32_t n_lanes, step, stride; float cut; } ga{n, s2 + 1, MAXM, draft_cut};
      e = rc.op("dl_gate");
      [e setComputePipelineState:pipe("h4_dgate_lanes")];
      [e setBuffer:dprob offset:0 atIndex:0];
      [e setBuffer:dl_maxd_b offset:0 atIndex:1];
      [e setBuffer:dgate[s2 + 1] offset:0 atIndex:2];
      [e setBytes:&ga length:sizeof ga atIndex:3];
      [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }
  }
  if (rc.e) [rc.e endEncoding];
  [rc.cb commit];
  [rc.cb waitUntilCompleted];
  if (rc.cb.error) mtl::die("draft_lanes: command buffer failed", rc.cb.error);
  if (gpu_ms) *gpu_ms = mtl::gpu_seconds(rc.cb) * 1e3;

  const uint32_t* po = (const uint32_t*)dout.contents;
  const float* pp = (const float*)dprob.contents;
  const float* pm = (const float*)dpmax.contents;
  const float* pg = (const float*)dmarg.contents;
  for (uint32_t j = 0; j < n; j++) {
    const uint32_t md = maxd[j];
    res[j].toks.resize(md);
    res[j].probs.resize(md);
    res[j].pmax.resize(md);
    res[j].marg.resize(md);
    for (uint32_t s2 = 0; s2 < md; s2++) {
      res[j].toks[s2] = po[s2 * MAXM + j];
      res[j].probs[s2] = pp[s2 * MAXM + j];
      res[j].pmax[s2] = pm[s2 * MAXM + j];
      res[j].marg[s2] = pg[s2 * MAXM + j];
    }
    if (cut && md > 0) {
      size_t k_gpu = md;
      if (draft_cut > 0) {
        float c = 1.0f;
        size_t k = 0;
        while (k < md && c * res[j].probs[k] >= draft_cut) { c *= res[j].probs[k]; k++; }
        k_gpu = std::max<size_t>(k, 1);
      }
      const size_t keep = select_draft_k(res[j].probs, k_gpu, n);
      res[j].toks.resize(keep);
      res[j].probs.resize(keep);
      res[j].pmax.resize(keep);
      res[j].marg.resize(keep);
    }
  }
  return res;
}

std::vector<float> Model::encode_vision(const uint16_t* pixels_bf16, uint32_t T, uint32_t h, uint32_t w,
                                        double* gpu_ms) {
  if (!has_vision) mtl::die("encode_vision: pack has no vision tower");
  if (T == 0 || h == 0 || w == 0 || (h % v_merge) != 0 || (w % v_merge) != 0)
    mtl::die("encode_vision: invalid grid_thw (h and w must be positive multiples of spatial_merge_size)");
  const uint32_t N = h * w;
  const uint32_t M = T * N;
  const uint32_t Mp = (M + 15u) / 16u * 16u;
  const uint32_t hm = h / v_merge, wm = w / v_merge;
  const uint32_t M_merge = T * hm * wm;
  const uint32_t Mp_merge = (M_merge + 15u) / 16u * 16u;
  const uint32_t M_out = T * hm * (wm + 1u);
  const uint32_t K_patch = 3u * v_ps * v_ps;  // 768
  const uint32_t K_merge = v_merge * v_merge * v_H;  // 4608

  mtl::last_use() = mtl::now_s();
  @autoreleasepool {
    auto tmp = [&](size_t bytes) -> id<MTLBuffer> {
      id<MTLBuffer> b = [C.dev newBufferWithLength:std::max<size_t>(bytes, 16) options:MTLResourceStorageModeShared];
      if (!b) mtl::die("encode_vision: failed to allocate transient Metal buffer");
      return b;
    };
    id<MTLBuffer> pix_b = [C.dev newBufferWithBytes:pixels_bf16 length:size_t(M) * K_patch * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> x_b = tmp(size_t(Mp) * v_H * 4);
    const size_t frag_elems = std::max<size_t>({size_t(Mp) * K_patch, size_t(Mp) * v_Ipad, size_t(Mp_merge) * K_merge});
    const size_t y_elems = std::max<size_t>({size_t(Mp) * 3u * v_H, size_t(Mp) * v_Ipad, size_t(Mp_merge) * K_merge});
    id<MTLBuffer> frag_b = tmp(frag_elems * 2);
    id<MTLBuffer> y_b = tmp(y_elems * 4);
    id<MTLBuffer> out_b = tmp(size_t(M_out) * H * 4);

    Rec rc;
    rc.cb = [C.q commandBuffer];
    id<MTLComputeCommandEncoder> e;

    // 1. Patch embedding: pixel_values [M, 768] -> frag -> mmp(V.patch.w) -> x_b + bias + interp pos_embed
    {
      struct { uint32_t M, Mp, K; } a{M, Mp, K_patch};
      e = rc.op("vit.to_frag");
      [e setComputePipelineState:pipe("vit_to_frag")];
      [e setBuffer:pix_b offset:0 atIndex:0];
      [e setBuffer:frag_b offset:0 atIndex:1];
      [e setBytes:&a length:sizeof a atIndex:2];
      [e dispatchThreads:MTLSizeMake(K_patch, Mp, 1) threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
    }
    mm(rc, "vit.patch", v_patch_w, frag_b, x_b, M);
    {
      struct { uint32_t T, h, w, H, edge; } a{T, h, w, v_H, v_edge};
      e = rc.op("vit.patch_pos");
      [e setComputePipelineState:pipe("vit_patch_pos")];
      [e setBuffer:x_b offset:0 atIndex:0];
      setb(e, v_patch_b, 1);
      setb(e, v_pos_embed, 2);
      [e setBytes:&a length:sizeof a atIndex:3];
      [e dispatchThreads:MTLSizeMake(v_H, M, 1) threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
    }

    // 2. 27 ViT blocks
    const uint32_t attn_gx = ((N + 15u) / 16u) * 4u;  // guarantees tail rows M..Mp-1 are zeroed on frame T-1
    for (uint32_t i = 0; i < v_nL; i++) {
      const VLayer& v = VL[i];
      // LN1 -> frag
      {
        struct { uint32_t M, Mp, H; float eps; } a{M, Mp, v_H, v_eps};
        e = rc.op("vit.ln1");
        [e setComputePipelineState:pipe("vit_layernorm_frag")];
        [e setBuffer:x_b offset:0 atIndex:0];
        setb(e, v.ln1_w, 1); setb(e, v.ln1_b, 2);
        [e setBuffer:frag_b offset:0 atIndex:3];
        [e setBytes:&a length:sizeof a atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(Mp, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      // QKV projection -> y_b [M, 3*v_H]
      mm(rc, "vit.qkv", v.qkv, frag_b, y_b, M);
      // Bidirectional FlashAttention (D=72, nh=16) -> frag_b [Mp, v_H]
      {
        struct { uint32_t T, N, Mp, H, nh, hd; float scale; } a{T, N, Mp, v_H, v_nh, v_hd, 1.0f / std::sqrt(float(v_hd))};
        e = rc.op("vit.attn");
        [e setComputePipelineState:pipe("vit_attn_fwd")];
        [e setBuffer:y_b offset:0 atIndex:0];
        setb(e, v.qkv_b, 1);
        [e setBuffer:frag_b offset:0 atIndex:2];
        [e setBytes:&a length:sizeof a atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(attn_gx, v_nh, T) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      }
      // O projection -> y_b [M, v_H], then x_b += y_b + o_b
      mm(rc, "vit.o", v.o, frag_b, y_b, M);
      {
        struct { uint32_t M, H; } a{M, v_H};
        e = rc.op("vit.o_res");
        [e setComputePipelineState:pipe("vit_bias_res")];
        [e setBuffer:x_b offset:0 atIndex:0];
        [e setBuffer:y_b offset:0 atIndex:1];
        setb(e, v.o_b, 2);
        [e setBytes:&a length:sizeof a atIndex:3];
        [e dispatchThreads:MTLSizeMake(v_H, M, 1) threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
      }
      // LN2 -> frag
      {
        struct { uint32_t M, Mp, H; float eps; } a{M, Mp, v_H, v_eps};
        e = rc.op("vit.ln2");
        [e setComputePipelineState:pipe("vit_layernorm_frag")];
        [e setBuffer:x_b offset:0 atIndex:0];
        setb(e, v.ln2_w, 1); setb(e, v.ln2_b, 2);
        [e setBuffer:frag_b offset:0 atIndex:3];
        [e setBytes:&a length:sizeof a atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(Mp, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      // MLP fc1 -> y_b [M, v_Ipad], bias + GELU -> frag_b [Mp, v_Ipad], fc2 -> y_b [M, v_H], x_b += y_b + fc2_b
      mm(rc, "vit.fc1", v.fc1, frag_b, y_b, M);
      {
        struct { uint32_t M, Mp, Nreal, Npad; } a{M, Mp, v_I, v_Ipad};
        e = rc.op("vit.gelu");
        [e setComputePipelineState:pipe("vit_bias_gelu_frag")];
        [e setBuffer:y_b offset:0 atIndex:0];
        setb(e, v.fc1_b, 1);
        [e setBuffer:frag_b offset:0 atIndex:2];
        [e setBytes:&a length:sizeof a atIndex:3];
        [e dispatchThreads:MTLSizeMake(v_Ipad, Mp, 1) threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
      }
      mm(rc, "vit.fc2", v.fc2, frag_b, y_b, M);
      {
        struct { uint32_t M, H; } a{M, v_H};
        e = rc.op("vit.fc2_res");
        [e setComputePipelineState:pipe("vit_bias_res")];
        [e setBuffer:x_b offset:0 atIndex:0];
        [e setBuffer:y_b offset:0 atIndex:1];
        setb(e, v.fc2_b, 2);
        [e setBytes:&a length:sizeof a atIndex:3];
        [e dispatchThreads:MTLSizeMake(v_H, M, 1) threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
      }
    }

    // 3. PatchMerger: RMSNorm + 2x2 gather -> frag_b [Mp_merge, 4608] -> fc1 -> GELU -> fc2 -> out_b + image_newline
    {
      struct { uint32_t T, h, w, H, Mp_merge; float eps; } a{T, h, w, v_H, Mp_merge, v_eps};
      e = rc.op("vit.merge_gather");
      [e setComputePipelineState:pipe("vit_merge_gather_frag")];
      [e setBuffer:x_b offset:0 atIndex:0];
      setb(e, v_merge_norm, 1);
      [e setBuffer:frag_b offset:0 atIndex:2];
      [e setBytes:&a length:sizeof a atIndex:3];
      [e dispatchThreadgroups:MTLSizeMake(M + (Mp_merge - M_merge), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    mm(rc, "vit.merge_fc1", v_merge_fc1, frag_b, y_b, M_merge);
    {
      struct { uint32_t M, Mp, Nreal, Npad; } a{M_merge, Mp_merge, K_merge, K_merge};
      e = rc.op("vit.merge_gelu");
      [e setComputePipelineState:pipe("vit_bias_gelu_frag")];
      [e setBuffer:y_b offset:0 atIndex:0];
      setb(e, v_merge_fc1_b, 1);
      [e setBuffer:frag_b offset:0 atIndex:2];
      [e setBytes:&a length:sizeof a atIndex:3];
      [e dispatchThreads:MTLSizeMake(K_merge, Mp_merge, 1) threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
    }
    mm(rc, "vit.merge_fc2", v_merge_fc2, frag_b, y_b, M_merge);
    {
      struct { uint32_t T, hm, wm, out_H; } a{T, hm, wm, H};
      e = rc.op("vit.merge_out");
      [e setComputePipelineState:pipe("vit_merge_out")];
      [e setBuffer:y_b offset:0 atIndex:0];
      setb(e, v_merge_fc2_b, 1);
      setb(e, v_merge_nl, 2);
      [e setBuffer:out_b offset:0 atIndex:3];
      [e setBytes:&a length:sizeof a atIndex:4];
      [e dispatchThreads:MTLSizeMake(H, M_out, 1) threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
    }

    if (rc.e) [rc.e endEncoding];
    [rc.cb commit];
    [rc.cb waitUntilCompleted];
    if (rc.cb.error) mtl::die("encode_vision: command buffer failed", rc.cb.error);
    if (gpu_ms) *gpu_ms = (rc.cb.GPUEndTime - rc.cb.GPUStartTime) * 1e3;

    std::vector<float> res(size_t(M_out) * H);
    memcpy(res.data(), out_b.contents, res.size() * 4);
    return res;
  }
}

}  // namespace hy4
