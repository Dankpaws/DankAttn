// DankAttn (ERNIE-5.5): Metal 4 engine implementation.
#include "ernie55.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>

namespace ernie55 {

static bool is_http(const std::string& s) {
  return s.rfind("http://", 0) == 0 || s.rfind("https://", 0) == 0;
}

static std::string http_get_string(const std::string& url) {
  std::string cmd = "curl -fsSL '" + url + "'";
  FILE* fp = popen(cmd.c_str(), "r");
  if (!fp) mtl::die(("popen curl " + url).c_str());
  std::string out;
  char buf[65536];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) out.append(buf, n);
  int rc = pclose(fp);
  if (rc != 0) mtl::die(("curl failed " + url).c_str());
  return out;
}

static id<MTLBuffer> http_get_buffer(id<MTLDevice> dev, const std::string& url, size_t& out_len) {
  // Query Content-Length via HEAD
  std::string hcmd = "curl -fsSI '" + url + "' | grep -i '^content-length:' | awk '{print $2}' | tr -d '\\r\\n'";
  FILE* hfp = popen(hcmd.c_str(), "r");
  if (!hfp) mtl::die(("curl HEAD " + url).c_str());
  char lbuf[64] = {};
  fgets(lbuf, sizeof(lbuf), hfp);
  pclose(hfp);
  size_t sz = strtoull(lbuf, nullptr, 10);
  if (sz == 0) mtl::die(("zero content-length for " + url).c_str());
  size_t aligned = (sz + 16383) & ~size_t(16383);
  id<MTLBuffer> b = [dev newBufferWithLength:aligned options:MTLResourceStorageModeShared];
  if (!b) mtl::die(("alloc buffer for " + url).c_str());
  uint8_t* dst = (uint8_t*)[b contents];
  std::string cmd = "curl -fsSL '" + url + "'";
  FILE* fp = popen(cmd.c_str(), "r");
  if (!fp) mtl::die(("curl GET " + url).c_str());
  size_t got = 0;
  while (got < sz) {
    size_t n = fread(dst + got, 1, std::min<size_t>(1 << 20, sz - got), fp);
    if (n == 0) break;
    got += n;
  }
  int rc = pclose(fp);
  if (rc != 0 || got != sz) mtl::die(("short read " + url).c_str());
  out_len = aligned;
  return b;
}

static NSDictionary* parse_json(const std::string& s) {
  NSData* d = [NSData dataWithBytes:s.data() length:s.size()];
  NSError* e = nil;
  id obj = [NSJSONSerialization JSONObjectWithData:d options:0 error:&e];
  if (!obj) mtl::die("json parse", e);
  return (NSDictionary*)obj;
}

static void pick_split(uint32_t N, uint32_t K, uint32_t gs, uint32_t slots, uint32_t& sgs, uint32_t& ksplit) {
  const uint32_t ng = K / gs;
  if (slots > 1) {
    sgs = (ng >= 4 && (ng % 4 == 0)) ? 4u : ((ng >= 2 && (ng % 2 == 0)) ? 2u : 1u);
    ksplit = 1;
    return;
  }
  sgs = (ng >= 8 && (ng % 8 == 0)) ? 8u : ((ng >= 4 && (ng % 4 == 0)) ? 4u : 1u);
  const uint32_t nf = N / 32;
  uint32_t rem = ng / sgs;
  if (nf <= 64 && rem >= 4 && (rem % 4 == 0)) ksplit = 4;
  else if (nf <= 128 && rem >= 2 && (rem % 2 == 0)) ksplit = 2;
  else ksplit = 1;
}

Model::Model(const std::string& pack, uint32_t capacity) : dir(pack), cap(capacity) {
  auto t0 = std::chrono::steady_clock::now();
  const bool remote = is_http(dir);
  std::string man_str = remote ? http_get_string(dir + "/manifest.json")
                               : mtl::slurp((dir + "/manifest.json").c_str());
  NSDictionary* man = parse_json(man_str);

  H = [man[@"H"] unsignedIntValue];
  HC = [man[@"HC"] unsignedIntValue];
  nL = [man[@"nL"] unsignedIntValue];
  vocab = [man[@"vocab"] unsignedIntValue];
  E = [man[@"E"] unsignedIntValue];
  topk = [man[@"topk"] unsignedIntValue];
  I = [man[@"I"] unsignedIntValue];
  moe_I = [man[@"moe_I"] unsignedIntValue];
  moe_lat = [man[@"moe_lat"] unsignedIntValue];
  nh = [man[@"nh"] unsignedIntValue];
  hd = [man[@"hd"] unsignedIntValue];
  nope = [man[@"nope"] unsignedIntValue];
  rdim = [man[@"rdim"] unsignedIntValue];
  o_groups = [man[@"o_groups"] unsignedIntValue];
  o_rank = [man[@"o_rank"] unsignedIntValue];
  win = [man[@"window"] unsignedIntValue];
  idx_topk = [man[@"idx_topk"] unsignedIntValue];
  sk_iters = [man[@"hc_sinkhorn_iters"] unsignedIntValue];
  bos = [man[@"bos"] unsignedIntValue];
  eos = [man[@"eos"] unsignedIntValue];
  eps = [man[@"eps"] floatValue];
  hc_eps = [man[@"hc_eps"] floatValue];
  situ_beta = [man[@"situ_beta"] floatValue];
  situ_lbeta = [man[@"situ_lbeta"] floatValue];

  if (cap % 128) mtl::die("capacity must be a multiple of 128");

  // Load weight shards (mmap locally, or HTTP stream into unified RAM)
  MTLResidencySetDescriptor* rsd = [MTLResidencySetDescriptor new];
  rsd.label = @"ernie55-weights";
  NSError* err = nil;
  id<MTLResidencySet> rs = [C.dev newResidencySetWithDescriptor:rsd error:&err];
  if (!rs) mtl::die("residency set", err);

  NSArray* flist = man[@"files"];
  uint32_t fi = 0;
  for (NSString* f in flist) {
    std::string fn = [f UTF8String];
    size_t len = 0;
    id<MTLBuffer> b = nil;
    if (remote) {
      fprintf(stderr, "\rloading shard %u/%lu (%s)...", fi + 1, (unsigned long)[flist count], fn.c_str());
      b = http_get_buffer(C.dev, dir + "/" + fn, len);
    } else {
      std::string p = dir + "/" + fn;
      int fd = open(p.c_str(), O_RDONLY);
      if (fd < 0) mtl::die(("open " + p).c_str());
      struct stat st;
      fstat(fd, &st);
      len = (size_t(st.st_size) + 16383) & ~size_t(16383);
      void* m = mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, 0);
      close(fd);
      if (m == MAP_FAILED) mtl::die(("mmap " + p).c_str());
      b = [C.dev newBufferWithBytesNoCopy:m length:len options:MTLResourceStorageModeShared deallocator:nil];
      if (!b) mtl::die(("nocopy " + p).c_str());
    }
    [rs addAllocation:b];
    files.push_back(b);
    mapped_bytes += len;
    fi++;
  }
  if (remote) fprintf(stderr, " done (%.2f GiB)\n", mapped_bytes / double(1024.0 * 1024 * 1024));
  [rs commit];
  [rs requestResidency];
  [C.q addResidencySet:rs];

  NSDictionary* T = man[@"tensors"];
  auto lin = [&](const std::string& name, uint32_t slots = 1) {
    NSDictionary* t = T[@(name.c_str())];
    if (!t) mtl::die(("missing tensor " + name).c_str());
    Lin l;
    l.w = files[[t[@"file"] unsignedIntValue]];
    l.woff = [t[@"off"] unsignedLongLongValue];
    l.s = files[[t[@"sfile"] unsignedIntValue]];
    l.soff = [t[@"soff"] unsignedLongLongValue];
    l.N = [t[@"N"] unsignedIntValue];
    l.K = [t[@"K"] unsignedIntValue];
    l.bits = [t[@"bits"] unsignedIntValue];
    l.gs = [t[@"gs"] unsignedIntValue];
    l.experts = [t[@"experts"] unsignedIntValue];
    pick_split(l.N, l.K, l.gs, slots, l.sgs, l.ksplit);
    return l;
  };
  auto raw = [&](const std::string& name) {
    NSDictionary* t = T[@(name.c_str())];
    if (!t) mtl::die(("missing tensor " + name).c_str());
    Raw r;
    r.b = files[[t[@"file"] unsignedIntValue]];
    r.off = [t[@"off"] unsignedLongLongValue];
    r.bytes = [t[@"bytes"] unsignedLongLongValue];
    return r;
  };

  embed = raw("embed");
  hch_fn = raw("hch.fn");
  hch_base = raw("hch.base");
  hch_scale = raw("hch.scale");
  norm = raw("norm");
  lm_head = lin("lm_head");
  mm_r = raw("multimax.r");
  mm_t = raw("multimax.t");

  NSArray* crs = man[@"compress_ratios"];
  L.resize(nL);
  const size_t alloc0 = C.allocated;
  for (uint32_t i = 0; i < nL; i++) {
    Layer& l = L[i];
    std::string p = "L" + std::to_string(i);
    l.is_dsa = [crs[i] intValue] <= 0;
    l.is_moe = (i >= 1);

    l.ahc_fn = raw(p + ".ahc.fn");
    l.ahc_base = raw(p + ".ahc.base");
    l.ahc_scale = raw(p + ".ahc.scale");
    l.mhc_fn = raw(p + ".mhc.fn");
    l.mhc_base = raw(p + ".mhc.base");
    l.mhc_scale = raw(p + ".mhc.scale");
    l.attn_norm = raw(p + ".attn_norm");
    l.ffn_norm = raw(p + ".ffn_norm");
    l.vha_m = raw(p + ".attn.vha_m");
    l.sink = raw(p + ".attn.sink");
    l.attn_in = lin(p + ".attn.in");
    l.attn_qb = lin(p + ".attn.qb");

    if (!l.is_dsa) {
      l.q_norm = raw(p + ".attn.q_norm");
      l.kv_norm = raw(p + ".attn.kv_norm");
      l.comp_ape = raw(p + ".attn.comp_ape");
      l.comp_norm = raw(p + ".attn.comp_norm");
      l.wo_a = lin(p + ".attn.wo_a", o_groups);
      l.wo_b = lin(p + ".attn.wo_b");

      l.hca_comp_cache = C.buf(size_t(cap / 128) * 512 * 2);
      l.hca_swa_kv = C.buf(size_t(HCA_RING) * 512 * 2);
      l.hca_comp_kv = C.buf(size_t(HCA_RING) * 512 * 4);
      l.hca_comp_sc = C.buf(size_t(HCA_RING) * 512 * 4);
    } else {
      l.qa_norm = raw(p + ".attn.qa_norm");
      l.kva_norm = raw(p + ".attn.kva_norm");
      l.idx_kn_w = raw(p + ".attn.idx_kn_w");
      l.idx_kn_b = raw(p + ".attn.idx_kn_b");
      l.kb = lin(p + ".attn.kb", nh);
      l.vb = lin(p + ".attn.vb", nh);
      l.attn_o = lin(p + ".attn.o");

      l.dsa_lat_q8 = C.buf(size_t(cap) * 672);
      l.dsa_idx_k8 = C.buf(size_t(cap) * 136);
    }

    if (!l.is_moe) {
      l.mlp_gu = lin(p + ".mlp.gu");
      l.mlp_dn = lin(p + ".mlp.dn");
    } else {
      l.moe_in = lin(p + ".moe.in");
      l.moe_bias = raw(p + ".moe.bias");
      l.moe_rscale = raw(p + ".moe.rscale");
      l.moe_lnorm = raw(p + ".moe.lnorm");
      l.moe_gu = lin(p + ".moe.gu", topk);
      l.moe_dn = lin(p + ".moe.dn", topk);
      l.moe_out = lin(p + ".moe.out");
    }
  }

  // Build RoPE tables (HCA YaRN + DSA plain)
  cos_hca = C.buf(size_t(cap) * 32 * 4);
  sin_hca = C.buf(size_t(cap) * 32 * 4);
  cos_mla = C.buf(size_t(cap) * 32 * 4);
  sin_mla = C.buf(size_t(cap) * 32 * 4);
  {
    const double base_hca = [man[@"compress_rope_theta"] doubleValue];
    const double base_mla = [man[@"rope_theta"] doubleValue];
    NSDictionary* rsc = man[@"rope_scaling"];
    const double factor = [rsc[@"factor"] doubleValue];
    const double orig_max = [rsc[@"original_max_position_embeddings"] doubleValue];
    const double beta_fast = [rsc[@"beta_fast"] doubleValue];
    const double beta_slow = [rsc[@"beta_slow"] doubleValue];

    auto find_corr = [&](double nrot) {
      return (64.0 * std::log(orig_max / (nrot * 2.0 * M_PI))) / (2.0 * std::log(base_hca));
    };
    double low = std::max(std::floor(find_corr(beta_fast)), 0.0);
    double high = std::min(std::ceil(find_corr(beta_slow)), 63.0);
    if (low == high) high += 0.001;

    double inv_hca[32], inv_mla[32];
    for (int j = 0; j < 32; j++) {
      double pf = std::pow(base_hca, double(2 * j) / 64.0);
      double ramp = std::clamp((double(j) - low) / (high - low), 0.0, 1.0);
      inv_hca[j] = (1.0 / (factor * pf)) * ramp + (1.0 / pf) * (1.0 - ramp);
      inv_mla[j] = 1.0 / std::pow(base_mla, double(2 * j) / 64.0);
    }
    float* ch = (float*)[cos_hca contents];
    float* sh = (float*)[sin_hca contents];
    float* cm = (float*)[cos_mla contents];
    float* sm = (float*)[sin_mla contents];
    for (uint32_t t = 0; t < cap; t++) {
      for (int j = 0; j < 32; j++) {
        double ah = double(t) * inv_hca[j];
        double am = double(t) * inv_mla[j];
        ch[size_t(t) * 32 + j] = float(std::cos(ah));
        sh[size_t(t) * 32 + j] = float(std::sin(ah));
        cm[size_t(t) * 32 + j] = float(std::cos(am));
        sm[size_t(t) * 32 + j] = float(std::sin(am));
      }
    }
  }

  // Scratch buffers
  ids_buf = C.buf(MAXM * 4);
  h_buf = C.buf(size_t(MAXM) * HC * H * 4);
  post_buf = C.buf(size_t(MAXM) * HC * 4);
  comb_buf = C.buf(size_t(MAXM) * HC * HC * 4);
  x_f32 = C.buf(size_t(MAXM) * H * 4);
  y_f32 = C.buf(size_t(MAXM) * H * 4);
  frag_a = C.buf(size_t(MAXM) * 32768 * 2);
  frag_b = C.buf(size_t(64) * MAXM * 512 * 2);
  big_a = C.buf(size_t(MAXM) * 40960 * 4);
  big_b = C.buf(size_t(MAXM) * 32768 * 4);
  big_c = C.buf(size_t(MAXM) * 16384 * 4);
  q_rope = C.buf(size_t(MAXM) * 64 * 64 * 4);
  iq_h = C.buf(size_t(MAXM) * 64 * 128 * 4);
  idx_w = C.buf(size_t(MAXM) * 64 * 4);
  idx_sc = C.buf(size_t(MAXM) * cap * 4);
  idx_sel = C.buf(size_t(MAXM) * (idx_topk + win) * 4);
  idx_len = C.buf(size_t(MAXM) * 4);
  top_ids = C.buf(size_t(MAXM) * topk * 4);
  top_w = C.buf(size_t(MAXM) * topk * 4);
  slots_buf = C.buf(64 * 4);
  nslots_buf = C.buf(4);
  for (uint32_t i = 0; i < 64; i++) ((uint32_t*)[slots_buf contents])[i] = i;
  *(uint32_t*)[nslots_buf contents] = 64;
  mmpart = C.buf(size_t(MAXM) * 201216 * 4 * 4); // generous partials
  counters = C.buf(65536 * 4);
  memset([counters contents], 0, 65536 * 4);
  logits_buf = C.buf(size_t(MAXM) * vocab * 4);

  state_bytes = C.allocated - alloc0;

  // Load Metal sources
  const std::string root = mtl::tree_root();
  mm2_src = mtl::slurp((root + "/engine/kernels/mm2.metal").c_str());
  e5_src = mtl::slurp((root + "/engine/kernels/ernie55.metal").c_str());

  double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
  fprintf(stderr, "ernie55 loaded: %u layers, cap=%u, weights=%.2f GiB, state=%.2f GiB (%.0f ms)\n",
          nL, cap, mapped_bytes / double(1024.0 * 1024 * 1024),
          state_bytes / double(1024.0 * 1024 * 1024), ms);
}

id<MTLComputePipelineState> Model::e5_pipe(const std::string& fn) {
  return C.pipe(e5_src, fn, {});
}

id<MTLComputePipelineState> Model::mm2_pipe(uint32_t bits, uint32_t gs, bool grouped, bool multi_slot) {
  std::map<std::string, std::string> defs;
  defs["WBITS"] = std::to_string(bits);
  defs["GS"] = std::to_string(gs);
  defs["TILES"] = "1";
  defs["XMODE"] = "0";
  defs["PF"] = "1";
  if (grouped) defs["GROUPED"] = "1";
  if (multi_slot) defs["SLOTS"] = "1";
  return C.pipe(mm2_src, "mm2", defs);
}

void Model::dispatch_mm2(id<MTLComputeCommandEncoder> e, const Lin& w, id<MTLBuffer> X, id<MTLBuffer> Y,
                         uint32_t M, id<MTLBuffer> sl, uint32_t nz, uint32_t xstride_u4, uint32_t ystride_f32) {
  const bool grouped = (sl != nil);
  const uint32_t raw_bits = (w.bits == 2 || w.bits == 20) ? 2u : w.bits;
  const uint32_t wunit = (raw_bits == 2 || raw_bits == 6) ? 8u : 16u;
  MmArgs a{};
  a.N = w.N;
  a.K = w.K;
  a.M = M;
  a.ksplit = w.ksplit;
  a.wstride = grouped ? uint32_t(size_t(w.N) * w.K * raw_bits / 8 / wunit) : 0u;
  a.sstride = grouped ? uint32_t(size_t(w.N) * (w.K / w.gs)) : 0u;
  a.xstride = xstride_u4;
  a.pstride = uint32_t(size_t(M) * w.N * w.ksplit);
  a.ystride = ystride_f32;
  a.sgs = w.sgs;
  a.psplit = grouped ? (M * w.N) : (16u * w.N);
  [e setComputePipelineState:mm2_pipe(w.bits, w.gs, grouped, false)];
  [e setBuffer:w.w offset:w.woff atIndex:0];
  [e setBuffer:w.s offset:w.soff atIndex:1];
  [e setBuffer:X offset:0 atIndex:2];
  [e setBuffer:mmpart offset:0 atIndex:3];
  [e setBuffer:Y offset:0 atIndex:4];
  [e setBytes:&a length:sizeof(a) atIndex:5];
  [e setBuffer:(grouped ? sl : slots_buf) offset:0 atIndex:6];
  [e setBuffer:nslots_buf offset:0 atIndex:7];
  [e setBuffer:counters offset:0 atIndex:8];
  [e setBuffer:mmpart offset:0 atIndex:9];
  [e setBuffer:mmpart offset:0 atIndex:10];
  [e setBuffer:mmpart offset:0 atIndex:11];
  [e setBuffer:mmpart offset:0 atIndex:12];
  [e setBuffer:counters offset:0 atIndex:13];
  [e setBuffer:counters offset:0 atIndex:14];
  [e setBuffer:counters offset:0 atIndex:15];
  [e setThreadgroupMemoryLength:(w.sgs > 1 ? size_t(w.sgs) * 32 * 16 * 4 : 16) atIndex:0];
  [e dispatchThreadgroups:MTLSizeMake(w.N / 32, w.ksplit, grouped ? nz : 1)
    threadsPerThreadgroup:MTLSizeMake(32 * w.sgs, 1, 1)];
}

const float* Model::step(const std::vector<uint32_t>& toks, double* gpu_ms) {
  const uint32_t M = uint32_t(toks.size());
  if (M == 0 || M > MAXM) mtl::die("step M out of range");
  if (pos + M > cap) mtl::die("context capacity exceeded");
  memcpy([ids_buf contents], toks.data(), M * 4);

  id<MTLCommandBuffer> cb = [C.q commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];

  auto barr = [&]() { [e memoryBarrierWithScope:MTLBarrierScopeBuffers]; };

  // 1. Embed
  {
    struct { uint32_t M, H, HC; } ea{M, H, HC};
    [e setComputePipelineState:e5_pipe("e5_embed")];
    [e setBuffer:ids_buf offset:0 atIndex:0];
    [e setBuffer:embed.b offset:embed.off atIndex:1];
    [e setBuffer:h_buf offset:0 atIndex:2];
    [e setBytes:&ea length:sizeof(ea) atIndex:3];
    [e dispatchThreads:MTLSizeMake(H, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barr();
  }

  for (uint32_t lid = 0; lid < nL; lid++) {
    Layer& l = L[lid];
    // Attention mHC pre (inject previous FFN output for lid > 0)
    {
      struct { uint32_t M, H, inject, mode, sk_iters; float eps, hc_eps; } ma{
        M, H, lid > 0 ? 1u : 0u, 0u, sk_iters, eps, hc_eps
      };
      [e setComputePipelineState:e5_pipe("e5_mhc_1pass")];
      [e setBuffer:h_buf offset:0 atIndex:0];
      [e setBuffer:y_f32 offset:0 atIndex:1];
      [e setBuffer:post_buf offset:0 atIndex:2];
      [e setBuffer:comb_buf offset:0 atIndex:3];
      [e setBuffer:l.ahc_fn.b offset:l.ahc_fn.off atIndex:4];
      [e setBuffer:l.ahc_base.b offset:l.ahc_base.off atIndex:5];
      [e setBuffer:l.ahc_scale.b offset:l.ahc_scale.off atIndex:6];
      [e setBuffer:l.attn_norm.b offset:l.attn_norm.off atIndex:7];
      [e setBuffer:frag_a offset:0 atIndex:8];
      [e setBuffer:x_f32 offset:0 atIndex:9];
      [e setBuffer:post_buf offset:0 atIndex:10];
      [e setBuffer:comb_buf offset:0 atIndex:11];
      [e setBytes:&ma length:sizeof(ma) atIndex:12];
      [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
      barr();
    }

    // Attention
    if (!l.is_dsa) {
      // HCA: attn_in -> big_a [M, 2560]
      dispatch_mm2(e, l.attn_in, frag_a, big_a, M, nil, 1, 0, 0);
      barr();
      // hca_prep: writes qr_frag (frag_a), swa_kv, comp_kv, comp_sc
      {
        struct { uint32_t M, p0, ring, cr, nope, rdim; float eps; } pa{M, pos, HCA_RING, 128u, nope, rdim, eps};
        [e setComputePipelineState:e5_pipe("e5_hca_prep")];
        [e setBuffer:big_a offset:0 atIndex:0];
        [e setBuffer:l.q_norm.b offset:l.q_norm.off atIndex:1];
        [e setBuffer:l.kv_norm.b offset:l.kv_norm.off atIndex:2];
        [e setBuffer:l.comp_ape.b offset:l.comp_ape.off atIndex:3];
        [e setBuffer:cos_hca offset:0 atIndex:4];
        [e setBuffer:sin_hca offset:0 atIndex:5];
        [e setBuffer:frag_a offset:0 atIndex:6];
        [e setBuffer:l.hca_swa_kv offset:0 atIndex:7];
        [e setBuffer:l.hca_comp_kv offset:0 atIndex:8];
        [e setBuffer:l.hca_comp_sc offset:0 atIndex:9];
        [e setBytes:&pa length:sizeof(pa) atIndex:10];
        [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barr();
      }
      // Check if any 128-token block completed in [pos, pos+M-1]
      uint32_t b_start = pos / 128;
      uint32_t b_end = (pos + M) / 128;
      if (b_end > b_start) {
        struct { uint32_t b0, ring, cr, nope, rdim; float eps; } ca{b_start, HCA_RING, 128u, nope, rdim, eps};
        [e setComputePipelineState:e5_pipe("e5_hca_compress")];
        [e setBuffer:l.hca_comp_kv offset:0 atIndex:0];
        [e setBuffer:l.hca_comp_sc offset:0 atIndex:1];
        [e setBuffer:l.comp_norm.b offset:l.comp_norm.off atIndex:2];
        [e setBuffer:cos_hca offset:0 atIndex:3];
        [e setBuffer:sin_hca offset:0 atIndex:4];
        [e setBuffer:l.hca_comp_cache offset:0 atIndex:5];
        [e setBytes:&ca length:sizeof(ca) atIndex:6];
        [e dispatchThreadgroups:MTLSizeMake(b_end - b_start, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
        barr();
      }
      // attn_qb: frag_a [M, 1024] -> big_a [M, 40960] = q(32768) | gate(8192)
      dispatch_mm2(e, l.attn_qb, frag_a, big_a, M, nil, 1, 0, 0);
      barr();
      // q_norm + RoPE -> big_b [M, 64, 512]
      {
        struct { uint32_t M, p0, nh, hd, nope, rdim, stride; float eps; } qa{M, pos, nh, hd, nope, rdim, 40960u, eps};
        [e setComputePipelineState:e5_pipe("e5_hca_qnorm_rope")];
        [e setBuffer:big_a offset:0 atIndex:0];
        [e setBuffer:cos_hca offset:0 atIndex:1];
        [e setBuffer:sin_hca offset:0 atIndex:2];
        [e setBuffer:big_b offset:0 atIndex:3];
        [e setBytes:&qa length:sizeof(qa) atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(nh, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barr();
      }
      // HCA attention -> big_c (needs M * 32768 floats, use big_b? No, big_b holds q -> wait, big_c is 16384*MAXM, use mmpart as temp o_out!)
      {
        struct { uint32_t M, p0, nh, hd, cr, win, ring; float scale; } aa{M, pos, nh, hd, 128u, win, HCA_RING, 1.0f / std::sqrt(float(hd))};
        [e setComputePipelineState:e5_pipe("e5_hca_attn")];
        [e setBuffer:big_b offset:0 atIndex:0];
        [e setBuffer:l.hca_comp_cache offset:0 atIndex:1];
        [e setBuffer:l.hca_swa_kv offset:0 atIndex:2];
        [e setBuffer:l.sink.b offset:l.sink.off atIndex:3];
        [e setBuffer:mmpart offset:0 atIndex:4];
        [e setBytes:&aa length:sizeof(aa) atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(nh, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        barr();
      }
      // VHA + inverse RoPE -> frag_b [8 groups, M, 4096]
      const uint32_t g_stride_bf16 = 128u * 512u; // for M <= 16: (4096/32)*512 = 65536 bf16
      {
        struct { uint32_t M, p0, nh, hd, nope, rdim, n_groups, g_stride_bf16; } va{
          M, pos, nh, hd, nope, rdim, o_groups, g_stride_bf16
        };
        [e setComputePipelineState:e5_pipe("e5_hca_vha_invrope")];
        [e setBuffer:mmpart offset:0 atIndex:0];
        [e setBuffer:l.vha_m.b offset:l.vha_m.off atIndex:1];
        [e setBuffer:cos_hca offset:0 atIndex:2];
        [e setBuffer:sin_hca offset:0 atIndex:3];
        [e setBuffer:frag_b offset:0 atIndex:4];
        [e setBytes:&va length:sizeof(va) atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(nh, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barr();
      }
      // Grouped wo_a: frag_b [8, M, 4096] -> big_b [8, M, 1024]
      dispatch_mm2(e, l.wo_a, frag_b, big_b, M, slots_buf, o_groups, g_stride_bf16 / 8, M * 1024);
      barr();
      // Gate mul + pack -> frag_a [M, 8192]
      {
        struct { uint32_t M, K, g_stride, g_off, z_gstride; } ga{M, 8192u, 40960u, 32768u, M * 1024u};
        [e setComputePipelineState:e5_pipe("e5_hca_gate_pack")];
        [e setBuffer:big_b offset:0 atIndex:0];
        [e setBuffer:big_a offset:0 atIndex:1];
        [e setBuffer:frag_a offset:0 atIndex:2];
        [e setBytes:&ga length:sizeof(ga) atIndex:3];
        [e dispatchThreads:MTLSizeMake(8192, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barr();
      }
      // wo_b: frag_a [M, 8192] -> y_f32 [M, 4096]
      dispatch_mm2(e, l.wo_b, frag_a, y_f32, M, nil, 1, 0, 0);
      barr();
    } else {
      // DSA: attn_in -> big_a [M, 1792]
      dispatch_mm2(e, l.attn_in, frag_a, big_a, M, nil, 1, 0, 0);
      barr();
      // dsa_prep: writes qc_frag (frag_a), dsa_lat_q8, dsa_idx_k8, idx_w
      {
        struct { uint32_t M, p0, cap; float eps; } pa{M, pos, cap, eps};
        [e setComputePipelineState:e5_pipe("e5_dsa_prep")];
        [e setBuffer:big_a offset:0 atIndex:0];
        [e setBuffer:l.qa_norm.b offset:l.qa_norm.off atIndex:1];
        [e setBuffer:l.kva_norm.b offset:l.kva_norm.off atIndex:2];
        [e setBuffer:l.idx_kn_w.b offset:l.idx_kn_w.off atIndex:3];
        [e setBuffer:l.idx_kn_b.b offset:l.idx_kn_b.off atIndex:4];
        [e setBuffer:cos_mla offset:0 atIndex:5];
        [e setBuffer:sin_mla offset:0 atIndex:6];
        [e setBuffer:frag_a offset:0 atIndex:7];
        [e setBuffer:l.dsa_lat_q8 offset:0 atIndex:8];
        [e setBuffer:l.dsa_idx_k8 offset:0 atIndex:9];
        [e setBuffer:idx_w offset:0 atIndex:10];
        [e setBytes:&pa length:sizeof(pa) atIndex:11];
        [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
        barr();
      }
      // attn_qb: frag_a [M, 1024] -> big_a [M, 40960] = q(16384) | gate(16384) | iq(8192)
      dispatch_mm2(e, l.attn_qb, frag_a, big_a, M, nil, 1, 0, 0);
      barr();
      // dsa_q_prep: q_nope -> frag_b [64, M, 192], q_rope -> q_rope, iq_h -> iq_h
      const uint32_t kb_gstride_bf16 = (192u / 32u) * 512u; // 3072 bf16 per head
      {
        struct { uint32_t M, p0, kb_gstride_bf16; } qa{M, pos, kb_gstride_bf16};
        [e setComputePipelineState:e5_pipe("e5_dsa_q_prep")];
        [e setBuffer:big_a offset:0 atIndex:0];
        [e setBuffer:cos_mla offset:0 atIndex:1];
        [e setBuffer:sin_mla offset:0 atIndex:2];
        [e setBuffer:frag_b offset:0 atIndex:3];
        [e setBuffer:q_rope offset:0 atIndex:4];
        [e setBuffer:iq_h offset:0 atIndex:5];
        [e setBytes:&qa length:sizeof(qa) atIndex:6];
        [e dispatchThreadgroups:MTLSizeMake(64, M, 1) threadsPerThreadgroup:MTLSizeMake(192, 1, 1)];
        barr();
      }
      // Indexer score (if needed) + topk
      {
        struct { uint32_t M, p0, win, topk, cap; } ia{M, pos, win, idx_topk, cap};
        uint32_t max_n_idx = (pos + M > win) ? (pos + M - win) : 0u;
        if (max_n_idx > idx_topk) {
          [e setComputePipelineState:e5_pipe("e5_dsa_idx_score")];
          [e setBuffer:iq_h offset:0 atIndex:0];
          [e setBuffer:idx_w offset:0 atIndex:1];
          [e setBuffer:l.dsa_idx_k8 offset:0 atIndex:2];
          [e setBuffer:idx_sc offset:0 atIndex:3];
          [e setBytes:&ia length:sizeof(ia) atIndex:4];
          [e dispatchThreads:MTLSizeMake(max_n_idx, M, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
          barr();
        }
        [e setComputePipelineState:e5_pipe("e5_dsa_idx_topk")];
        [e setBuffer:idx_sc offset:0 atIndex:0];
        [e setBuffer:idx_sel offset:0 atIndex:1];
        [e setBuffer:idx_len offset:0 atIndex:2];
        [e setBytes:&ia length:sizeof(ia) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barr();
      }
      // Grouped kb: frag_b [64, M, 192] -> big_b [64, M, 512]
      dispatch_mm2(e, l.kb, frag_b, big_b, M, slots_buf, 64, kb_gstride_bf16 / 8, M * 512);
      barr();
      // DSA attention -> frag_b [64, M, 512] bf16 frags for vb
      const uint32_t vb_gstride_bf16 = (512u / 32u) * 512u; // 8192 bf16 per head
      {
        struct { uint32_t M, topk_win, qabs_gstride, vb_gstride_bf16; float scale; } da{
          M, idx_topk + win, M * 512u, vb_gstride_bf16, 1.0f / std::sqrt(256.0f)
        };
        [e setComputePipelineState:e5_pipe("e5_dsa_attn")];
        [e setBuffer:big_b offset:0 atIndex:0];
        [e setBuffer:q_rope offset:0 atIndex:1];
        [e setBuffer:l.dsa_lat_q8 offset:0 atIndex:2];
        [e setBuffer:idx_sel offset:0 atIndex:3];
        [e setBuffer:idx_len offset:0 atIndex:4];
        [e setBuffer:l.sink.b offset:l.sink.off atIndex:5];
        [e setBuffer:frag_b offset:0 atIndex:6];
        [e setBytes:&da length:sizeof(da) atIndex:7];
        [e dispatchThreadgroups:MTLSizeMake(64, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        barr();
      }
      // Grouped vb: frag_b [64, M, 512] -> big_c [64, M, 256]
      dispatch_mm2(e, l.vb, frag_b, big_c, M, slots_buf, 64, vb_gstride_bf16 / 8, M * 256);
      barr();
      // VHA + gate -> frag_a [M, 16384]
      {
        struct { uint32_t M, o_gstride; } va{M, M * 256u};
        [e setComputePipelineState:e5_pipe("e5_dsa_vha_gate")];
        [e setBuffer:big_c offset:0 atIndex:0];
        [e setBuffer:l.vha_m.b offset:l.vha_m.off atIndex:1];
        [e setBuffer:big_a offset:0 atIndex:2];
        [e setBuffer:frag_a offset:0 atIndex:3];
        [e setBytes:&va length:sizeof(va) atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(64, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barr();
      }
      // attn_o: frag_a [M, 16384] -> y_f32 [M, 4096]
      dispatch_mm2(e, l.attn_o, frag_a, y_f32, M, nil, 1, 0, 0);
      barr();
    }

    // FFN mHC pre (inject attention output y_f32)
    {
      struct { uint32_t M, H, inject, mode, sk_iters; float eps, hc_eps; } ma{
        M, H, 1u, 0u, sk_iters, eps, hc_eps
      };
      [e setComputePipelineState:e5_pipe("e5_mhc_1pass")];
      [e setBuffer:h_buf offset:0 atIndex:0];
      [e setBuffer:y_f32 offset:0 atIndex:1];
      [e setBuffer:post_buf offset:0 atIndex:2];
      [e setBuffer:comb_buf offset:0 atIndex:3];
      [e setBuffer:l.mhc_fn.b offset:l.mhc_fn.off atIndex:4];
      [e setBuffer:l.mhc_base.b offset:l.mhc_base.off atIndex:5];
      [e setBuffer:l.mhc_scale.b offset:l.mhc_scale.off atIndex:6];
      [e setBuffer:l.ffn_norm.b offset:l.ffn_norm.off atIndex:7];
      [e setBuffer:frag_a offset:0 atIndex:8];
      [e setBuffer:x_f32 offset:0 atIndex:9];
      [e setBuffer:post_buf offset:0 atIndex:10];
      [e setBuffer:comb_buf offset:0 atIndex:11];
      [e setBytes:&ma length:sizeof(ma) atIndex:12];
      [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
      barr();
    }

    // FFN / MoE
    if (!l.is_moe) {
      // Dense L0: mlp_gu -> big_a [M, 28672]
      dispatch_mm2(e, l.mlp_gu, frag_a, big_a, M, nil, 1, 0, 0);
      barr();
      {
        struct { uint32_t M, I, src_stride, src_off, dst_K, dst_off; float beta, lbeta; } sa{
          M, I, 2 * I, 0u, I, 0u, situ_beta, situ_lbeta
        };
        [e setComputePipelineState:e5_pipe("e5_situ_pack")];
        [e setBuffer:big_a offset:0 atIndex:0];
        [e setBuffer:frag_a offset:0 atIndex:1];
        [e setBytes:&sa length:sizeof(sa) atIndex:2];
        [e dispatchThreads:MTLSizeMake(I, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barr();
      }
      dispatch_mm2(e, l.mlp_dn, frag_a, y_f32, M, nil, 1, 0, 0);
      barr();
    } else {
      // MoE: moe_in -> big_a [M, 7168]
      dispatch_mm2(e, l.moe_in, frag_a, big_a, M, nil, 1, 0, 0);
      barr();
      // Router top-10
      {
        struct { uint32_t M, E, topk; } ra{M, E, topk};
        [e setComputePipelineState:e5_pipe("e5_route_topk")];
        [e setBuffer:big_a offset:0 atIndex:0];
        [e setBuffer:l.moe_bias.b offset:l.moe_bias.off atIndex:1];
        [e setBuffer:l.moe_rscale.b offset:l.moe_rscale.off atIndex:2];
        [e setBuffer:top_ids offset:0 atIndex:3];
        [e setBuffer:top_w offset:0 atIndex:4];
        [e setBytes:&ra length:sizeof(ra) atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barr();
      }
      // Run each token m's 10 experts (grouped M=1, nz=10) -> big_b [M*10, 2048]
      // Note: simple & exact! For each token m in 0..M-1:
      for (uint32_t m = 0; m < M; m++) {
        // Pack fc1[m] (offset 1024 in big_a row m) -> frag_a [1, 2048]
        {
          struct { uint32_t M, K, src_stride, src_off; } pa{1u, 2048u, 7168u, m * 7168u + 1024u};
          [e setComputePipelineState:e5_pipe("e5_pack_f32")];
          [e setBuffer:big_a offset:0 atIndex:0];
          [e setBuffer:frag_a offset:0 atIndex:1];
          [e setBytes:&pa length:sizeof(pa) atIndex:2];
          [e dispatchThreads:MTLSizeMake(2048, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          barr();
        }
        // moe_gu: shared X=frag_a (xstride=0), sl=top_ids + m*10*4, nz=10 -> big_c [10, 4096]
        {
          const uint32_t raw_bits = (l.moe_gu.bits == 2 || l.moe_gu.bits == 20) ? 2u : l.moe_gu.bits;
          const uint32_t wunit = (raw_bits == 2 || raw_bits == 6) ? 8u : 16u;
          MmArgs a{};
          a.N = l.moe_gu.N; a.K = l.moe_gu.K; a.M = 1; a.ksplit = l.moe_gu.ksplit;
          a.wstride = uint32_t(size_t(l.moe_gu.N) * l.moe_gu.K * raw_bits / 8 / wunit);
          a.sstride = uint32_t(size_t(l.moe_gu.N) * (l.moe_gu.K / l.moe_gu.gs));
          a.xstride = 0; a.pstride = l.moe_gu.N * l.moe_gu.ksplit; a.ystride = 4096u; a.sgs = l.moe_gu.sgs;
          a.psplit = l.moe_gu.N;
          [e setComputePipelineState:mm2_pipe(l.moe_gu.bits, l.moe_gu.gs, true, false)];
          [e setBuffer:l.moe_gu.w offset:l.moe_gu.woff atIndex:0];
          [e setBuffer:l.moe_gu.s offset:l.moe_gu.soff atIndex:1];
          [e setBuffer:frag_a offset:0 atIndex:2];
          [e setBuffer:mmpart offset:0 atIndex:3];
          [e setBuffer:big_c offset:0 atIndex:4];
          [e setBytes:&a length:sizeof(a) atIndex:5];
          [e setBuffer:top_ids offset:m * topk * 4 atIndex:6];
          [e setBuffer:nslots_buf offset:0 atIndex:7];
          [e setBuffer:counters offset:0 atIndex:8];
          [e setBuffer:mmpart offset:0 atIndex:9];
          [e setBuffer:mmpart offset:0 atIndex:10];
          [e setBuffer:mmpart offset:0 atIndex:11];
          [e setBuffer:mmpart offset:0 atIndex:12];
          [e setBuffer:counters offset:0 atIndex:13];
          [e setBuffer:counters offset:0 atIndex:14];
          [e setBuffer:counters offset:0 atIndex:15];
          [e setThreadgroupMemoryLength:(l.moe_gu.sgs > 1 ? size_t(l.moe_gu.sgs) * 32 * 16 * 4 : 16) atIndex:0];
          [e dispatchThreadgroups:MTLSizeMake(l.moe_gu.N / 32, l.moe_gu.ksplit, topk)
            threadsPerThreadgroup:MTLSizeMake(32 * l.moe_gu.sgs, 1, 1)];
          barr();
        }
        // SiTU-GLU on big_c [10, 4096] -> frag_b [10, 2048] (each slot k gets its own 1-row frag slab: stride (2048/32)*512 = 32768 bf16)
        for (uint32_t k = 0; k < topk; k++) {
          struct { uint32_t M, I, src_stride, src_off, dst_K, dst_off; float beta, lbeta; } sa{
            1u, 2048u, 4096u, k * 4096u, 2048u, 0u, situ_beta, situ_lbeta
          };
          [e setComputePipelineState:e5_pipe("e5_situ_pack")];
          [e setBuffer:big_c offset:0 atIndex:0];
          [e setBuffer:frag_b offset:size_t(k) * 32768 * 2 atIndex:1];
          [e setBytes:&sa length:sizeof(sa) atIndex:2];
          [e dispatchThreads:MTLSizeMake(2048, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        barr();
        // moe_dn: frag_b [10, 2048] -> big_b + m*10*2048*4
        {
          const uint32_t raw_bits = (l.moe_dn.bits == 2 || l.moe_dn.bits == 20) ? 2u : l.moe_dn.bits;
          const uint32_t wunit = (raw_bits == 2 || raw_bits == 6) ? 8u : 16u;
          MmArgs a{};
          a.N = l.moe_dn.N; a.K = l.moe_dn.K; a.M = 1; a.ksplit = l.moe_dn.ksplit;
          a.wstride = uint32_t(size_t(l.moe_dn.N) * l.moe_dn.K * raw_bits / 8 / wunit);
          a.sstride = uint32_t(size_t(l.moe_dn.N) * (l.moe_dn.K / l.moe_dn.gs));
          a.xstride = 32768u / 8u; // in uint4 (16 bytes = 8 bf16)
          a.pstride = l.moe_dn.N * l.moe_dn.ksplit; a.ystride = 2048u; a.sgs = l.moe_dn.sgs;
          a.psplit = l.moe_dn.N;
          [e setComputePipelineState:mm2_pipe(l.moe_dn.bits, l.moe_dn.gs, true, false)];
          [e setBuffer:l.moe_dn.w offset:l.moe_dn.woff atIndex:0];
          [e setBuffer:l.moe_dn.s offset:l.moe_dn.soff atIndex:1];
          [e setBuffer:frag_b offset:0 atIndex:2];
          [e setBuffer:mmpart offset:0 atIndex:3];
          [e setBuffer:big_b offset:size_t(m) * topk * 2048 * 4 atIndex:4];
          [e setBytes:&a length:sizeof(a) atIndex:5];
          [e setBuffer:top_ids offset:m * topk * 4 atIndex:6];
          [e setBuffer:nslots_buf offset:0 atIndex:7];
          [e setBuffer:counters offset:0 atIndex:8];
          [e setBuffer:mmpart offset:0 atIndex:9];
          [e setBuffer:mmpart offset:0 atIndex:10];
          [e setBuffer:mmpart offset:0 atIndex:11];
          [e setBuffer:mmpart offset:0 atIndex:12];
          [e setBuffer:counters offset:0 atIndex:13];
          [e setBuffer:counters offset:0 atIndex:14];
          [e setBuffer:counters offset:0 atIndex:15];
          [e setThreadgroupMemoryLength:(l.moe_dn.sgs > 1 ? size_t(l.moe_dn.sgs) * 32 * 16 * 4 : 16) atIndex:0];
          [e dispatchThreadgroups:MTLSizeMake(l.moe_dn.N / 32, l.moe_dn.ksplit, topk)
            threadsPerThreadgroup:MTLSizeMake(32 * l.moe_dn.sgs, 1, 1)];
          barr();
        }
      }
      // Combine + latent_norm + concat shared_act -> frag_a [M, 4096]
      {
        struct { uint32_t M, topk; float eps, beta, lbeta; } ca{M, topk, eps, situ_beta, situ_lbeta};
        [e setComputePipelineState:e5_pipe("e5_moe_combine_pack")];
        [e setBuffer:big_b offset:0 atIndex:0];
        [e setBuffer:top_w offset:0 atIndex:1];
        [e setBuffer:l.moe_lnorm.b offset:l.moe_lnorm.off atIndex:2];
        [e setBuffer:big_a offset:0 atIndex:3];
        [e setBuffer:frag_a offset:0 atIndex:4];
        [e setBytes:&ca length:sizeof(ca) atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
        barr();
      }
      // moe_out: frag_a [M, 4096] -> y_f32 [M, 4096]
      dispatch_mm2(e, l.moe_out, frag_a, y_f32, M, nil, 1, 0, 0);
      barr();
    }
  }

  // Final mHC head contraction (inject last layer's FFN output) + model.norm -> frag_a [M, H]
  {
    struct { uint32_t M, H, inject, mode, sk_iters; float eps, hc_eps; } ma{
      M, H, 1u, 1u, sk_iters, eps, hc_eps
    };
    [e setComputePipelineState:e5_pipe("e5_mhc_1pass")];
    [e setBuffer:h_buf offset:0 atIndex:0];
    [e setBuffer:y_f32 offset:0 atIndex:1];
    [e setBuffer:post_buf offset:0 atIndex:2];
    [e setBuffer:comb_buf offset:0 atIndex:3];
    [e setBuffer:hch_fn.b offset:hch_fn.off atIndex:4];
    [e setBuffer:hch_base.b offset:hch_base.off atIndex:5];
    [e setBuffer:hch_scale.b offset:hch_scale.off atIndex:6];
    [e setBuffer:norm.b offset:norm.off atIndex:7];
    [e setBuffer:frag_a offset:0 atIndex:8];
    [e setBuffer:x_f32 offset:0 atIndex:9];
    [e setBuffer:post_buf offset:0 atIndex:10];
    [e setBuffer:comb_buf offset:0 atIndex:11];
    [e setBytes:&ma length:sizeof(ma) atIndex:12];
    [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    barr();
  }

  // lm_head -> logits_buf [M, vocab]
  dispatch_mm2(e, lm_head, frag_a, logits_buf, M, nil, 1, 0, 0);
  barr();

  // MultiMax
  {
    struct { uint32_t M, V; } ma{M, vocab};
    [e setComputePipelineState:e5_pipe("e5_multimax")];
    [e setBuffer:logits_buf offset:0 atIndex:0];
    [e setBuffer:mm_r.b offset:mm_r.off atIndex:1];
    [e setBuffer:mm_t.b offset:mm_t.off atIndex:2];
    [e setBytes:&ma length:sizeof(ma) atIndex:3];
    [e dispatchThreads:MTLSizeMake(vocab, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }

  [e endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if ([cb status] != MTLCommandBufferStatusCompleted) mtl::die("command buffer failed", [cb error]);
  if (gpu_ms) *gpu_ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1e3;

  pos += M;
  return (const float*)[logits_buf contents];
}

}  // namespace ernie55
