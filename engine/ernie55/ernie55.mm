// DankAttn (ERNIE-5.5): Metal 4 engine implementation.
#define DANKATTN_MTL4_IMPL
#include "ernie55.h"
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <chrono>

extern char** environ;

namespace ernie55 {

static bool is_http(const std::string& s) {
  return s.rfind("http://", 0) == 0 || s.rfind("https://", 0) == 0;
}

static FILE* spawn_curl(const std::vector<std::string>& args, pid_t& out_pid) {
  int pfd[2];
  if (pipe(pfd) != 0) mtl::die("pipe for curl");
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_addclose(&fa, pfd[0]);
  posix_spawn_file_actions_adddup2(&fa, pfd[1], STDOUT_FILENO);
  posix_spawn_file_actions_addclose(&fa, pfd[1]);
  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
  argv.push_back(nullptr);
  pid_t pid = 0;
  int rc = posix_spawn(&pid, "/usr/bin/curl", &fa, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&fa);
  close(pfd[1]);
  if (rc != 0) {
    close(pfd[0]);
    mtl::die("posix_spawn /usr/bin/curl");
  }
  FILE* fp = fdopen(pfd[0], "r");
  if (!fp) {
    close(pfd[0]);
    waitpid(pid, nullptr, 0);
    mtl::die("fdopen curl pipe");
  }
  out_pid = pid;
  return fp;
}

static int finish_curl(FILE* fp, pid_t pid) {
  fclose(fp);
  int status = 0;
  while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
  return (WIFEXITED(status) && WEXITSTATUS(status) == 0) ? 0 : -1;
}

static std::string http_get_string(const std::string& url) {
  pid_t pid = 0;
  FILE* fp = spawn_curl({"curl", "-fsSL", "--", url}, pid);
  std::string out;
  char buf[65536];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) out.append(buf, n);
  if (finish_curl(fp, pid) != 0) mtl::die(("curl failed " + url).c_str());
  return out;
}

static id<MTLBuffer> http_get_buffer(id<MTLDevice> dev, const std::string& url, size_t& out_len) {
  // Query Content-Length via HEAD without invoking a shell
  pid_t hpid = 0;
  FILE* hfp = spawn_curl({"curl", "-fsSIL", "--", url}, hpid);
  size_t sz = 0;
  char lbuf[1024];
  while (fgets(lbuf, sizeof(lbuf), hfp)) {
    if (strncasecmp(lbuf, "content-length:", 15) == 0) {
      sz = strtoull(lbuf + 15, nullptr, 10);
    }
  }
  if (finish_curl(hfp, hpid) != 0 || sz == 0) mtl::die(("zero content-length for " + url).c_str());
  size_t aligned = (sz + 16383) & ~size_t(16383);
  id<MTLBuffer> b = [dev newBufferWithLength:aligned options:MTLResourceStorageModeShared];
  if (!b) mtl::die(("alloc buffer for " + url).c_str());
  uint8_t* dst = (uint8_t*)[b contents];
  pid_t pid = 0;
  FILE* fp = spawn_curl({"curl", "-fsSL", "--", url}, pid);
  size_t got = 0;
  while (got < sz) {
    size_t n = fread(dst + got, 1, std::min<size_t>(1 << 20, sz - got), fp);
    if (n == 0) break;
    got += n;
  }
  if (finish_curl(fp, pid) != 0 || got != sz) mtl::die(("short read " + url).c_str());
  out_len = aligned;
  return b;
}

using dankattn::parse_json;

static void pick_split(uint32_t N, uint32_t K, uint32_t gs, uint32_t slots, uint32_t& sgs, uint32_t& ksplit) {
  const uint32_t G = K / gs;
  const uint32_t nf = (N / 32u) * slots;
  // Hy4-style M5 Ultra split picker: hit >= 1024 active 32-col slices, prefer low sgs (no TG reduction when nf>=1024)
  uint32_t s_tot = G;
  for (uint32_t s = 1; s <= G; s++) {
    if (G % s == 0 && nf * s >= 1024u) { s_tot = s; break; }
  }
  // For grouped weights (slots > 1), mm2 GROUPED with ksplit > 1 needs per-slot counters; keep ksplit=1 for grouped
  if (slots > 1) {
    if (nf >= 512u) { sgs = 1u; ksplit = 1u; return; }
    sgs = 1u;
    for (uint32_t c = 4u; c >= 1u; c--) {
      if (s_tot >= c && G % c == 0) { sgs = c; break; }
    }
    ksplit = 1u;
    return;
  }
  // Prefer ksplit over large sgs to minimize shared-memory reduction overhead while keeping ksplit <= 8
  ksplit = 1u;
  for (uint32_t k = 8u; k >= 1u; k--) {
    if (s_tot % k == 0 && (s_tot / k) <= 4u) { ksplit = k; break; }
  }
  sgs = s_tot / ksplit;
  if (sgs > 4u) {
    sgs = 4u;
    ksplit = std::min(8u, G / sgs);
    while (G % (sgs * ksplit) != 0 && ksplit > 1u) ksplit--;
  }
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
  hadamard = man[@"hadamard"] ? [man[@"hadamard"] boolValue] : false;

  if (cap % 128) mtl::die("capacity must be a multiple of 128");
  const bool want_mtl4 = dankattn::env_b("DANKATTN_MTL4", false);
  if (want_mtl4) C.track = true;

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
    mtl::check_pack_filename(fn);
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
  if (want_mtl4) {
    mtl4.init(C.dev);
    mtl4.add_residency_set(rs);
  }

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
      l.hca_comp_kv = C.buf(size_t(COMP_RING) * 512 * 4);
      l.hca_comp_sc = C.buf(size_t(COMP_RING) * 512 * 4);
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

  // Scratch buffers (sized for PREFILL_M=640; alias big_c with mmpart when !hadamard to save 42 MB!)
  ids_buf = C.buf(PREFILL_M * 4);
  h_buf = C.buf(size_t(PREFILL_M) * HC * H * 4);
  h_tmp = C.buf(size_t(8) * HC * H * 4);
  post_buf = C.buf(size_t(PREFILL_M) * HC * 4);
  comb_buf = C.buf(size_t(PREFILL_M) * HC * HC * 4);
  x_f32 = nil;  // unused (removed dead write)
  y_f32 = C.buf(size_t(PREFILL_M) * H * 4);
  frag_a = C.buf(size_t(PREFILL_M) * 16384 * 2);
  frag_b = C.buf(size_t(64) * PREFILL_M * 512 * 2);
  big_a = C.buf(size_t(PREFILL_M) * 40960 * 4);
  big_b = C.buf(size_t(PREFILL_M) * 32768 * 4);
  mmpart = C.buf(std::max(size_t(MAXM) * 201216 * 4 * 4, size_t(PREFILL_M) * 16384 * 4));
  big_c = hadamard ? C.buf(size_t(PREFILL_M) * 40960u * 4) : mmpart;
  q_rope = C.buf(size_t(PREFILL_M) * 64 * 64 * 4);
  iq_h = mmpart;  // aliased with mmpart (only live during dsa_q_prep -> dsa_idx_score, saves ~29 MB!)
  idx_w = C.buf(size_t(PREFILL_M) * 64 * 4);
  idx_sc = C.buf(size_t(IDX_SUB) * cap * 4);
  idx_sel = C.buf(size_t(PREFILL_M) * (idx_topk + win) * 4);
  idx_len = C.buf(size_t(PREFILL_M) * 4);
  top_ids = C.buf(size_t(PREFILL_M) * topk * 4);
  top_w = C.buf(size_t(PREFILL_M) * topk * 4);
  slots_buf = C.buf(64 * 4);
  nslots_buf = C.buf(4);
  for (uint32_t i = 0; i < 64; i++) ((uint32_t*)[slots_buf contents])[i] = i;
  *(uint32_t*)[nslots_buf contents] = 64;
  moe_slots = C.buf(512 * 4);
  moe_nslots = C.buf(4);
  roff = C.buf(512 * 4);
  rcnt = C.buf(512 * 4);
  rlist = C.buf(size_t(PREFILL_M) * topk * 4);
  cidx = C.buf(size_t(PREFILL_M) * topk * 4);
  wtile = C.buf((512 + size_t(PREFILL_M) * topk / 16 + 16) * 4);
  moe_ind = C.buf(32);
  hcpart = C.buf(size_t(PREFILL_M) * 16u * 34u * 4u);
  comp_kv_tmp = C.buf(size_t(PREFILL_M) * 512u * 4u);
  comp_sc_tmp = C.buf(size_t(PREFILL_M) * 512u * 4u);
  counters = C.buf(65536 * 4);
  memset([counters contents], 0, 65536 * 4);
  logits_buf = C.buf(size_t(MAXM) * vocab * 4);

  state_bytes = C.allocated - alloc0;
  if (want_mtl4) {
    mtl4.add_buffers(C.made);
    fprintf(stderr, "ernie55: Metal 4 command encoding enabled (DANKATTN_MTL4=1, %zu tracked bufs)\n", C.made.size());
  }

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
  auto it = e5_pcache.find(fn);
  if (it != e5_pcache.end()) return it->second;
  id<MTLComputePipelineState> p = C.pipe(e5_src, fn, {});
  e5_pcache[fn] = p;
  return p;
}

id<MTLComputePipelineState> Model::mm1_pipe(uint32_t bits, uint32_t gs, bool grouped, uint32_t fuse) {
  const uint64_t key = (1ULL << 56) | (uint64_t(bits) << 32) | (uint64_t(gs) << 16) | (uint64_t(grouped) << 8) | fuse;
  auto it = mm_pcache.find(key);
  if (it != mm_pcache.end()) return it->second;
  std::map<std::string, std::string> defs;
  defs["WBITS"] = std::to_string(bits);
  defs["GS"] = std::to_string(gs);
  if (grouped) defs["GROUPED"] = "1";
  if (fuse) defs["FUSE"] = std::to_string(fuse);
  static const bool w2_bit = getenv("E5_W2_BIT") && atoi(getenv("E5_W2_BIT")) != 0;
  if (w2_bit) defs["W2_BIT"] = "1";
  id<MTLComputePipelineState> p = C.pipe(mm2_src, "mm1", defs);
  mm_pcache[key] = p;
  return p;
}

void Model::dispatch_mm1(id<MTLComputeCommandEncoder> e, const Lin& w, id<MTLBuffer> X, id<MTLBuffer> Y,
                         id<MTLBuffer> sl, size_t sl_off, uint32_t nz, uint32_t xstride_u4, uint32_t ystride_f32,
                         uint32_t fuse, id<MTLBuffer> extra_buf, size_t extra_off) {
  const bool grouped = (sl != nil);
  const uint32_t raw_bits = (w.bits == 2 || w.bits == 20) ? 2u : w.bits;
  const uint32_t wunit = (raw_bits == 2 || raw_bits == 6) ? 8u : 16u;
  const uint32_t G = w.K / w.gs;
  const uint32_t nf_tot = (w.N / 32u) * (grouped ? nz : 1u);
  static const uint32_t max_nft = getenv("E5_MM1_NFT") ? uint32_t(atoi(getenv("E5_MM1_NFT"))) : 1u;
  static const uint32_t sgs_thr = getenv("E5_MM1_SGSTHR") ? uint32_t(atoi(getenv("E5_MM1_SGSTHR"))) : 1024u;
  static const uint32_t max_sgs = getenv("E5_MM1_MAXSGS") ? uint32_t(atoi(getenv("E5_MM1_MAXSGS"))) : 8u;
  static const uint32_t s1_thr = getenv("E5_MM1_S1THR") ? uint32_t(atoi(getenv("E5_MM1_S1THR"))) : 2048u;
  uint32_t sgs1 = 1u;
  if (nf_tot >= s1_thr) sgs1 = 1u;
  else if (nf_tot <= 128u && G % 32u == 0 && max_sgs >= 32u) sgs1 = 32u;
  else if (nf_tot < sgs_thr / 4u && G % 16u == 0 && max_sgs >= 16u) sgs1 = 16u;
  else if (nf_tot < sgs_thr / 2u && G % 8u == 0 && max_sgs >= 8u) sgs1 = 8u;
  else if (nf_tot < sgs_thr && G % 4u == 0 && max_sgs >= 4u) sgs1 = 4u;
  else if (nf_tot < sgs_thr * 2u && G % 2u == 0 && max_sgs >= 2u) sgs1 = 2u;
  uint32_t nft1 = 1u;
  while (sgs1 * nft1 < 4u && nft1 < max_nft && (w.N / 32u) % (nft1 * 2u) == 0) nft1 *= 2u;
  MmArgs a{};
  a.N = w.N;
  a.K = w.K;
  a.M = 1u;
  a.ksplit = 1u;
  a.wstride = grouped ? uint32_t(size_t(w.N) * w.K * raw_bits / 8 / wunit) : 0u;
  a.sstride = grouped ? uint32_t(size_t(w.N) * (w.K / w.gs)) : 0u;
  a.xstride = xstride_u4;
  a.ystride = ystride_f32;
  a.sgs = sgs1;
  [e setComputePipelineState:mm1_pipe(w.bits, w.gs, grouped, fuse)];
  [e setBuffer:w.w offset:w.woff atIndex:0];
  [e setBuffer:w.s offset:w.soff atIndex:1];
  [e setBuffer:X offset:0 atIndex:2];
  [e setBuffer:Y offset:0 atIndex:3];
  [e setBytes:&a length:sizeof(a) atIndex:4];
  [e setBuffer:(grouped ? sl : slots_buf) offset:(grouped ? sl_off : 0) atIndex:5];
  if (fuse == 1u) {
    [e setBuffer:extra_buf offset:extra_off atIndex:6];
  } else if (fuse == 2u) {
    float situ_p[2] = {situ_beta, situ_lbeta};
    [e setBytes:situ_p length:sizeof(situ_p) atIndex:6];
  }
  [e dispatchThreadgroups:MTLSizeMake(w.N / 32u / nft1, 1, grouped ? nz : 1)
    threadsPerThreadgroup:MTLSizeMake(32u * sgs1 * nft1, 1, 1)];
}

id<MTLComputePipelineState> Model::mm2_pipe(uint32_t bits, uint32_t gs, bool grouped, bool multi_slot, uint32_t epi) {
  const uint64_t key = (2ULL << 56) | (uint64_t(bits) << 32) | (uint64_t(gs) << 16) |
                       (uint64_t(grouped) << 8) | (uint64_t(multi_slot) << 4) | epi;
  auto it = mm_pcache.find(key);
  if (it != mm_pcache.end()) return it->second;
  std::map<std::string, std::string> defs;
  defs["WBITS"] = std::to_string(bits);
  defs["GS"] = std::to_string(gs);
  static const std::string pf_s = getenv("E5_PF") ? std::string(getenv("E5_PF")) : "1";
  defs["TILES"] = "1";
  defs["XMODE"] = "0";
  defs["PF"] = pf_s;
  if (grouped) defs["GROUPED"] = "1";
  if (multi_slot) defs["SLOTS"] = "1";
  if (epi) defs["EPI"] = std::to_string(epi);
  id<MTLComputePipelineState> p = C.pipe(mm2_src, "mm2", defs);
  mm_pcache[key] = p;
  return p;
}

id<MTLComputePipelineState> Model::mmp_pipe(uint32_t bits, uint32_t gs, uint32_t epi) {
  const uint64_t key = (3ULL << 56) | (uint64_t(bits) << 32) | (uint64_t(gs) << 16) | epi;
  auto it = mm_pcache.find(key);
  if (it != mm_pcache.end()) return it->second;
  std::map<std::string, std::string> defs;
  defs["WBITS"] = std::to_string(bits);
  defs["GS"] = std::to_string(gs);
  defs["MMP_FAST"] = "1";
  defs["MMP_NF"] = "2";
  defs["MMP_CH"] = "8";
  if (epi) {
    defs["EPI"] = std::to_string(epi);
    if (epi == 3u) {
      defs["SITU_BETA"] = std::to_string(situ_beta);
      defs["SITU_LBETA"] = std::to_string(situ_lbeta);
    }
  }
  id<MTLComputePipelineState> p = C.pipe(mm2_src, "mmp", defs);
  mm_pcache[key] = p;
  return p;
}

id<MTLComputePipelineState> Model::mmpg_pipe(uint32_t bits, uint32_t gs, bool contig_x, uint32_t ch,
                                             uint32_t epi, uint32_t nf) {
  const uint64_t key = (4ULL << 56) | (uint64_t(bits) << 32) | (uint64_t(gs) << 16) |
                       (uint64_t(contig_x) << 12) | (uint64_t(ch) << 8) | (uint64_t(nf) << 4) | epi;
  auto it = mm_pcache.find(key);
  if (it != mm_pcache.end()) return it->second;
  std::map<std::string, std::string> defs;
  defs["WBITS"] = std::to_string(bits);
  defs["GS"] = std::to_string(gs);
  defs["GROUPED"] = "1";
  defs["MMP_FAST"] = "1";
  defs["MMP_NF"] = std::to_string(nf);
  defs["MMP_CH"] = std::to_string(ch);
  if (contig_x) defs["CONTIG_X"] = "1";
  if (epi) {
    defs["EPI"] = std::to_string(epi);
    if (epi == 3u) {
      defs["SITU_BETA"] = std::to_string(situ_beta);
      defs["SITU_LBETA"] = std::to_string(situ_lbeta);
    }
  }
  id<MTLComputePipelineState> p = C.pipe(mm2_src, "mmpg", defs);
  mm_pcache[key] = p;
  return p;
}

void Model::dispatch_mm(id<MTLComputeCommandEncoder> e, const Lin& w, id<MTLBuffer> X, id<MTLBuffer> Y,
                        uint32_t M, id<MTLBuffer> sl, uint32_t nz, uint32_t xstride_u4, uint32_t ystride_f32,
                        uint32_t epi, id<MTLBuffer> E0, uint32_t R,
                        id<MTLBuffer> E1, uint32_t epi_H, uint32_t epi_HC) {
  static const bool no_mm1 = getenv("E5_NO_MM1") && atoi(getenv("E5_NO_MM1")) != 0;
  if (M == 1u && !no_mm1 && epi == 0u) {
    dispatch_mm1(e, w, X, Y, sl, 0, nz, xstride_u4, ystride_f32);
    return;
  }
  const bool grouped = (sl != nil);
  const uint32_t raw_bits = (w.bits == 2 || w.bits == 20) ? 2u : w.bits;
  const uint32_t wunit = (raw_bits == 2 || raw_bits == 6) ? 8u : 16u;
  if (M > MAXM) {
    // Prefill path: mmp (dense) or mmpg CONTIG_X (grouped per-head/per-group)
    constexpr uint32_t CH = 8u;
    constexpr uint32_t NF = 2u;
    const uint32_t gx = (w.N / 32 + NF - 1) / NF;
    MmArgs a{};
    a.N = w.N;
    a.K = w.K;
    a.M = M;
    a.ksplit = w.ksplit;
    a.wstride = grouped ? uint32_t(size_t(w.N) * w.K * raw_bits / 8 / wunit) : 0u;
    a.sstride = grouped ? uint32_t(size_t(w.N) * (w.K / w.gs)) : 0u;
    a.xstride = xstride_u4;
    a.sgs = w.sgs;
    a.ldx = grouped ? 1u : 0u;
    a.rm = 0u;
    a.R = R;
    a.H = epi_H;
    a.HC = epi_HC;
    [e setComputePipelineState:(grouped ? mmpg_pipe(w.bits, w.gs, true, CH, epi, NF) : mmp_pipe(w.bits, w.gs, epi))];
    [e setBuffer:w.w offset:w.woff atIndex:0];
    [e setBuffer:w.s offset:w.soff atIndex:1];
    [e setBuffer:X offset:0 atIndex:2];
    [e setBuffer:mmpart offset:0 atIndex:3];
    [e setBuffer:Y offset:0 atIndex:4];
    [e setBytes:&a length:sizeof(a) atIndex:5];
    [e setBuffer:(grouped ? sl : slots_buf) offset:0 atIndex:6];
    [e setBuffer:nslots_buf offset:0 atIndex:7];
    [e setBuffer:counters offset:0 atIndex:8];
    [e setBuffer:(E0 ? E0 : mmpart) offset:0 atIndex:9];
    [e setBuffer:(E1 ? E1 : mmpart) offset:0 atIndex:10];
    [e setBuffer:mmpart offset:0 atIndex:11];
    [e setBuffer:roff offset:0 atIndex:12];
    [e setBuffer:rcnt offset:0 atIndex:13];
    [e setBuffer:rlist offset:0 atIndex:14];
    [e setBuffer:wtile offset:0 atIndex:15];
    [e dispatchThreadgroups:MTLSizeMake(gx, (M + 16 * CH - 1) / (16 * CH), grouped ? nz : 1)
      threadsPerThreadgroup:MTLSizeMake(32 * CH, 1, 1)];
    return;
  }
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
  a.R = R;
  a.H = epi_H;
  a.HC = epi_HC;
  [e setComputePipelineState:mm2_pipe(w.bits, w.gs, grouped, false, epi)];
  [e setBuffer:w.w offset:w.woff atIndex:0];
  [e setBuffer:w.s offset:w.soff atIndex:1];
  [e setBuffer:X offset:0 atIndex:2];
  [e setBuffer:mmpart offset:0 atIndex:3];
  [e setBuffer:Y offset:0 atIndex:4];
  [e setBytes:&a length:sizeof(a) atIndex:5];
  [e setBuffer:(grouped ? sl : slots_buf) offset:0 atIndex:6];
  [e setBuffer:nslots_buf offset:0 atIndex:7];
  [e setBuffer:counters offset:0 atIndex:8];
  [e setBuffer:(E0 ? E0 : mmpart) offset:0 atIndex:9];
  [e setBuffer:(E1 ? E1 : mmpart) offset:0 atIndex:10];
  [e setBuffer:mmpart offset:0 atIndex:11];
  [e setBuffer:mmpart offset:0 atIndex:12];
  [e setBuffer:counters offset:0 atIndex:13];
  [e setBuffer:counters offset:0 atIndex:14];
  [e setBuffer:counters offset:0 atIndex:15];
  uint32_t nft = 1u;
  while (w.sgs * nft < 4u && (w.N / 32u) % (nft * 2u) == 0) nft *= 2u;
  [e setThreadgroupMemoryLength:(w.sgs > 1 ? size_t(w.sgs) * nft * 32 * 16 * 4 : 16) atIndex:0];
  [e dispatchThreadgroups:MTLSizeMake(w.N / 32u / nft, w.ksplit, grouped ? nz : 1)
    threadsPerThreadgroup:MTLSizeMake(32u * w.sgs * nft, 1, 1)];
}

void Model::dispatch_mmpg_moe(id<MTLComputeCommandEncoder> e, const Lin& w, id<MTLBuffer> X, id<MTLBuffer> Y,
                              uint32_t M, uint32_t ldx, size_t ind_off,
                              uint32_t epi, id<MTLBuffer> E0, uint32_t R) {
  static const int mmpg_mode = getenv("E5_MMPG_MODE") ? atoi(getenv("E5_MMPG_MODE")) : 0;
  // mode 0: CH=1 NF=4 (16 rows x 128 cols, 4 SGs); mode 1: CH=4 NF=2 (64 rows x 64 cols, 4 SGs); mode 2: CH=2 NF=2 (32 rows x 64 cols, 2 SGs)
  const uint32_t CH = (mmpg_mode == 1) ? 4u : ((mmpg_mode == 2) ? 2u : 1u);
  const uint32_t NF = (mmpg_mode == 0) ? 4u : 2u;
  const uint32_t nsg = (mmpg_mode == 0) ? 4u : CH;
  const uint32_t raw_bits = (w.bits == 2 || w.bits == 20) ? 2u : w.bits;
  const uint32_t wunit = (raw_bits == 2 || raw_bits == 6) ? 8u : 16u;
  MmArgs a{};
  a.N = w.N;
  a.K = w.K;
  a.M = M;
  a.ksplit = w.ksplit;
  a.wstride = uint32_t(size_t(w.N) * w.K * raw_bits / 8 / wunit);
  a.sstride = uint32_t(size_t(w.N) * (w.K / w.gs));
  a.sgs = w.sgs;
  a.ldx = ldx;
  a.rm = 1u;
  a.R = R;
  [e setComputePipelineState:mmpg_pipe(w.bits, w.gs, false, CH, epi, NF)];
  [e setBuffer:w.w offset:w.woff atIndex:0];
  [e setBuffer:w.s offset:w.soff atIndex:1];
  [e setBuffer:X offset:0 atIndex:2];
  [e setBuffer:mmpart offset:0 atIndex:3];
  [e setBuffer:Y offset:0 atIndex:4];
  [e setBytes:&a length:sizeof(a) atIndex:5];
  [e setBuffer:moe_slots offset:0 atIndex:6];
  [e setBuffer:moe_nslots offset:0 atIndex:7];
  [e setBuffer:counters offset:0 atIndex:8];
  [e setBuffer:(E0 ? E0 : mmpart) offset:0 atIndex:9];
  [e setBuffer:mmpart offset:0 atIndex:10];
  [e setBuffer:mmpart offset:0 atIndex:11];
  [e setBuffer:roff offset:0 atIndex:12];
  [e setBuffer:rcnt offset:0 atIndex:13];
  [e setBuffer:rlist offset:0 atIndex:14];
  [e setBuffer:wtile offset:0 atIndex:15];
  [e dispatchThreadgroupsWithIndirectBuffer:moe_ind indirectBufferOffset:ind_off
                      threadsPerThreadgroup:MTLSizeMake(32u * nsg, 1, 1)];
}

const float* Model::step(const std::vector<uint32_t>& toks, double* gpu_ms,
                         bool need_logits, bool last_only) {
  const uint32_t M = uint32_t(toks.size());
  if (M == 0 || M > PREFILL_M) mtl::die("step M out of range");
  if (need_logits && !last_only && M > MAXM) mtl::die("all_logits requires M <= MAXM");
  if (pos + M > cap) mtl::die("context capacity exceeded");
  const uint32_t rb = (M + 15) / 16;
  memcpy([ids_buf contents], toks.data(), M * 4);

  static const int prof = getenv("E5_PROF") ? atoi(getenv("E5_PROF")) : 0;
  const bool use_m4 = mtl4.enabled && !prof;
  std::map<std::string, double> pt;
  std::vector<id<MTLCommandBuffer>> cbs;
  id<MTLCommandBuffer> cb = use_m4 ? nil : [C.q commandBuffer];
  double m4_t0 = use_m4 ? dankattn::host_now() : 0.0;
  id<MTLComputeCommandEncoder> e = use_m4 ? mtl4.begin(/*auto_barrier=*/true) : [cb computeCommandEncoder];

  static const bool no_barr = getenv("E5_NO_BARR") && atoi(getenv("E5_NO_BARR")) != 0;
  auto barr = [&]() { if (!use_m4 && !no_barr) [e memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
  auto flush_cb = [&]() {
    if (use_m4) return;
    [e endEncoding];
    [cb commit];
    cbs.push_back(cb);
    cb = [C.q commandBuffer];
    e = [cb computeCommandEncoder];
  };
  auto mark = [&](const char* tag, int lvl = 1) {
    if (prof < lvl) return;
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    pt[tag] += ([cb GPUEndTime] - [cb GPUStartTime]) * 1e3;
    cb = [C.q commandBuffer];
    e = [cb computeCommandEncoder];
  };

  id<MTLBuffer> h_cur = h_buf;
  auto dispatch_mhc = [&](uint32_t m_rows, uint32_t inject, uint32_t mode,
                          const Raw& fn, const Raw& base, const Raw& scale, const Raw& nw,
                          size_t row_off = 0u) {
    const uint32_t npart = (m_rows <= 8u) ? 8u : 16u;
    struct { uint32_t M, H, inject, mode, sk_iters, npart; float eps, hc_eps; } ma{
      m_rows, H, inject, mode, sk_iters, npart, eps, hc_eps
    };
    if (m_rows <= 8u) {
      // 64-TG (M=1 exact 1.0 wave!) split-H x split-mix mHC (8 K-chunks x 8 mix-groups of 3 mixes, npart=8)
      id<MTLBuffer> h_in = h_cur;
      id<MTLBuffer> h_out = inject ? (h_cur == h_buf ? h_tmp : h_buf) : h_cur;
      const size_t h_in_off = row_off * HC * H * 4;
      const size_t h_out_off = (h_out == h_buf) ? (row_off * HC * H * 4) : 0u;
      [e setComputePipelineState:e5_pipe("e5_mhc_part")];
      [e setBuffer:h_in offset:h_in_off atIndex:0];
      [e setBuffer:y_f32 offset:row_off * H * 4 atIndex:1];
      [e setBuffer:post_buf offset:row_off * HC * 4 atIndex:2];
      [e setBuffer:comb_buf offset:row_off * HC * HC * 4 atIndex:3];
      [e setBuffer:fn.b offset:fn.off atIndex:4];
      [e setBuffer:hcpart offset:0 atIndex:5];
      [e setBytes:&ma length:sizeof(ma) atIndex:6];
      [e setBuffer:h_out offset:h_out_off atIndex:7];
      [e dispatchThreadgroups:MTLSizeMake(8u, (mode == 0u) ? 8u : 2u, m_rows) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      barr();
      if (inject) h_cur = h_out;
      const size_t h_fin_off = (h_cur == h_buf) ? (row_off * HC * H * 4) : 0u;
      [e setComputePipelineState:e5_pipe("e5_mhc_fin")];
      [e setBuffer:h_cur offset:h_fin_off atIndex:0];
      [e setBuffer:hcpart offset:0 atIndex:1];
      [e setBuffer:base.b offset:base.off atIndex:2];
      [e setBuffer:scale.b offset:scale.off atIndex:3];
      [e setBuffer:nw.b offset:nw.off atIndex:4];
      [e setBuffer:frag_a offset:0 atIndex:5];
      [e setBuffer:frag_a offset:0 atIndex:6];
      [e setBuffer:post_buf offset:row_off * HC * 4 atIndex:7];
      [e setBuffer:comb_buf offset:row_off * HC * HC * 4 atIndex:8];
      [e setBytes:&ma length:sizeof(ma) atIndex:9];
      [e dispatchThreadgroups:MTLSizeMake(16u, m_rows, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      barr();
    } else {
      // Fused inject + Gram + matmul2d prefill mHC (eliminates e5_mhc_pre_inject + second h read!) + e5_mhc_fin
      [e setComputePipelineState:e5_pipe("e5_mhc_pre_mma")];
      [e setBuffer:h_buf offset:row_off * HC * H * 4 atIndex:0];
      [e setBuffer:fn.b offset:fn.off atIndex:1];
      [e setBuffer:hcpart offset:0 atIndex:2];
      [e setBytes:&ma length:sizeof(ma) atIndex:3];
      [e setBuffer:y_f32 offset:row_off * H * 4 atIndex:4];
      [e setBuffer:post_buf offset:row_off * HC * 4 atIndex:5];
      [e setBuffer:comb_buf offset:row_off * HC * HC * 4 atIndex:6];
      [e dispatchThreadgroups:MTLSizeMake(16u, (m_rows + 15u) / 16u, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      barr();
      [e setComputePipelineState:e5_pipe("e5_mhc_fin")];
      [e setBuffer:h_buf offset:row_off * HC * H * 4 atIndex:0];
      [e setBuffer:hcpart offset:0 atIndex:1];
      [e setBuffer:base.b offset:base.off atIndex:2];
      [e setBuffer:scale.b offset:scale.off atIndex:3];
      [e setBuffer:nw.b offset:nw.off atIndex:4];
      [e setBuffer:frag_a offset:0 atIndex:5];
      [e setBuffer:frag_a offset:0 atIndex:6];
      [e setBuffer:post_buf offset:row_off * HC * 4 atIndex:7];
      [e setBuffer:comb_buf offset:row_off * HC * HC * 4 atIndex:8];
      [e setBytes:&ma length:sizeof(ma) atIndex:9];
      [e dispatchThreadgroups:MTLSizeMake(16u, m_rows, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      barr();
    }
  };

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
    dispatch_mhc(M, lid > 0 ? 1u : 0u, 0u, l.ahc_fn, l.ahc_base, l.ahc_scale, l.attn_norm);
    mark("mhc");

    // Attention
    if (!l.is_dsa) {
      // HCA: attn_in -> big_a [M, 2560]
      dispatch_mm(e, l.attn_in, frag_a, big_a, M, nil, 1, 0, 0);
      barr();
      mark("hca_in", 2);
      // hca_prep: writes qr_frag (frag_a), swa_kv, comp_kv, comp_sc
      {
        const uint32_t use_tmp = (M > 128u) ? 1u : 0u;
        struct { uint32_t M, p0, ring, cr, nope, rdim, comp_ring, use_tmp; float eps; } pa{
          M, pos, HCA_RING, 128u, nope, rdim, COMP_RING, use_tmp, eps
        };
        [e setComputePipelineState:e5_pipe(M == 1u ? "e5_hca_prep_m1" : "e5_hca_prep")];
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
        if (M != 1u) {
          [e setBuffer:comp_kv_tmp offset:0 atIndex:11];
          [e setBuffer:comp_sc_tmp offset:0 atIndex:12];
        }
        if (M == 1u) {
          [e dispatchThreadgroups:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        } else {
          [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        barr();
      }
      // Check if any 128-token block completed in [pos, pos+M-1]
      uint32_t b_start = pos / 128;
      uint32_t b_end = (pos + M) / 128;
      if (b_end > b_start) {
        const uint32_t use_tmp = (M > 128u) ? 1u : 0u;
        struct { uint32_t b0, ring, cr, nope, rdim, p0, use_tmp; float eps; } ca{
          b_start, COMP_RING, 128u, nope, rdim, pos, use_tmp, eps
        };
        [e setComputePipelineState:e5_pipe("e5_hca_compress")];
        [e setBuffer:l.hca_comp_kv offset:0 atIndex:0];
        [e setBuffer:l.hca_comp_sc offset:0 atIndex:1];
        [e setBuffer:l.comp_norm.b offset:l.comp_norm.off atIndex:2];
        [e setBuffer:cos_hca offset:0 atIndex:3];
        [e setBuffer:sin_hca offset:0 atIndex:4];
        [e setBuffer:l.hca_comp_cache offset:0 atIndex:5];
        [e setBytes:&ca length:sizeof(ca) atIndex:6];
        [e setBuffer:comp_kv_tmp offset:0 atIndex:7];
        [e setBuffer:comp_sc_tmp offset:0 atIndex:8];
        [e dispatchThreadgroups:MTLSizeMake(b_end - b_start, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
        barr();
      }
      mark("hca_prep", 2);
      // attn_qb: frag_a [M, 1024] -> big_a [M, 40960] = q(32768) | gate(8192)
      dispatch_mm(e, l.attn_qb, frag_a, big_a, M, nil, 1, 0, 0);
      barr();
      mark("hca_qb", 2);
      // q_norm + RoPE -> big_b [M, 64, 512] (fused into e5_hca_attn_m1 when M == 1!)
      static const bool old_attn_hca = getenv("E5_OLD_ATTN") && atoi(getenv("E5_OLD_ATTN")) != 0;
      if (old_attn_hca || M != 1u) {
        struct { uint32_t M, p0, nh, hd, nope, rdim, stride; float eps; } qa{M, pos, nh, hd, nope, rdim, 40960u, eps};
        [e setComputePipelineState:e5_pipe("e5_hca_qnorm_rope")];
        [e setBuffer:big_a offset:0 atIndex:0];
        [e setBuffer:cos_hca offset:0 atIndex:1];
        [e setBuffer:sin_hca offset:0 atIndex:2];
        [e setBuffer:big_b offset:0 atIndex:3];
        [e setBytes:&qa length:sizeof(qa) atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(nh, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        barr();
      }
      // HCA attention -> mmpart [M, 32768] f32
      mark("hca_qnorm", 1);
      {
        const uint32_t approx_ntot = (pos + M) / 128u + std::min(pos + M, win);
        if (!old_attn_hca && M == 1u) {
          // Dedicated 1-pass M=1 HCA attention with FUSED Q-RMSNorm + RoPE (reads big_a directly!)
          struct { uint32_t M, p0, nh, hd, cr, win, ring, nope, rdim; float scale, eps; } aa{
            1u, pos, nh, hd, 128u, win, HCA_RING, nope, rdim, 1.0f / std::sqrt(float(hd)), eps
          };
          [e setComputePipelineState:e5_pipe("e5_hca_attn_m1")];
          [e setBuffer:big_a offset:0 atIndex:0];
          [e setBuffer:l.hca_comp_cache offset:0 atIndex:1];
          [e setBuffer:l.hca_swa_kv offset:0 atIndex:2];
          [e setBuffer:l.sink.b offset:l.sink.off atIndex:3];
          [e setBuffer:mmpart offset:0 atIndex:4];
          [e setBytes:&aa length:sizeof(aa) atIndex:5];
          [e setBuffer:cos_hca offset:0 atIndex:6];
          [e setBuffer:sin_hca offset:0 atIndex:7];
          [e dispatchThreadgroups:MTLSizeMake(nh, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
          barr();
        } else if (!old_attn_hca && M <= 8u && approx_ntot >= 64u) {
          const uint32_t S = std::min(std::max(2u, 16u / M), std::max(2u, (approx_ntot + 31u) / 32u));
          struct { uint32_t M, p0, nh, hd, cr, win, ring, S; float scale; } aa{
            M, pos, nh, hd, 128u, win, HCA_RING, S, 1.0f / std::sqrt(float(hd))
          };
          [e setComputePipelineState:e5_pipe("e5_hca_attn_mma_split")];
          [e setBuffer:big_b offset:0 atIndex:0];
          [e setBuffer:l.hca_comp_cache offset:0 atIndex:1];
          [e setBuffer:l.hca_swa_kv offset:0 atIndex:2];
          [e setBuffer:l.sink.b offset:l.sink.off atIndex:3];
          [e setBuffer:frag_b offset:0 atIndex:4];
          [e setBytes:&aa length:sizeof(aa) atIndex:5];
          [e dispatchThreadgroups:MTLSizeMake(nh / 16u, M, S) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
          barr();
          [e setComputePipelineState:e5_pipe("e5_hca_attn_merge")];
          [e setBuffer:frag_b offset:0 atIndex:0];
          [e setBuffer:mmpart offset:0 atIndex:1];
          [e setBytes:&aa length:sizeof(aa) atIndex:2];
          [e dispatchThreadgroups:MTLSizeMake(nh, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
          barr();
        } else {
          struct { uint32_t M, p0, nh, hd, cr, win, ring; float scale; } aa{M, pos, nh, hd, 128u, win, HCA_RING, 1.0f / std::sqrt(float(hd))};
          [e setComputePipelineState:e5_pipe(old_attn_hca ? "e5_hca_attn" : "e5_hca_attn_mma")];
          [e setBuffer:big_b offset:0 atIndex:0];
          [e setBuffer:l.hca_comp_cache offset:0 atIndex:1];
          [e setBuffer:l.hca_swa_kv offset:0 atIndex:2];
          [e setBuffer:l.sink.b offset:l.sink.off atIndex:3];
          [e setBuffer:mmpart offset:0 atIndex:4];
          [e setBytes:&aa length:sizeof(aa) atIndex:5];
          [e dispatchThreadgroups:MTLSizeMake(old_attn_hca ? nh : (nh / 16u), M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
          barr();
        }
      }
      mark("hca_attn");
      // VHA + inverse RoPE -> frag_b [8 groups, M, 4096] (matmul2d accelerated)
      const uint32_t g_stride_bf16 = rb * 128u * 512u;
      {
        struct { uint32_t M, p0, nh, hd, nope, rdim, n_groups, g_stride_bf16; } va{
          M, pos, nh, hd, nope, rdim, o_groups, g_stride_bf16
        };
        if (M == 1u) {
          [e setComputePipelineState:e5_pipe("e5_hca_vha_invrope_m1")];
          [e setBuffer:mmpart offset:0 atIndex:0];
          [e setBuffer:l.vha_m.b offset:l.vha_m.off atIndex:1];
          [e setBuffer:cos_hca offset:0 atIndex:2];
          [e setBuffer:sin_hca offset:0 atIndex:3];
          [e setBuffer:frag_b offset:0 atIndex:4];
          [e setBytes:&va length:sizeof(va) atIndex:5];
          [e dispatchThreadgroups:MTLSizeMake(4u, 16u, 1u) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        } else {
          [e setComputePipelineState:e5_pipe("e5_hca_vha_invrope_mma")];
          [e setBuffer:mmpart offset:0 atIndex:0];
          [e setBuffer:l.vha_m.b offset:l.vha_m.off atIndex:1];
          [e setBuffer:cos_hca offset:0 atIndex:2];
          [e setBuffer:sin_hca offset:0 atIndex:3];
          [e setBuffer:frag_b offset:0 atIndex:4];
          [e setBytes:&va length:sizeof(va) atIndex:5];
          [e dispatchThreadgroups:MTLSizeMake(4u, 4u, M) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        barr();
      }
      mark("hca_vha", 2);
      // Grouped wo_a + gate*sigm + frag_a pack (fused in mm1 for M==1, fused EPI=5 in mmpg for M>MAXM)
      static const bool no_mm1_hca = getenv("E5_NO_MM1") && atoi(getenv("E5_NO_MM1")) != 0;
      if (M == 1u && !no_mm1_hca) {
        dispatch_mm1(e, l.wo_a, frag_b, frag_a, slots_buf, 0, o_groups, g_stride_bf16 / 8, 1024u,
                     1u, big_a, 32768u * 4u);
        barr();
      } else {
        // Fused wo_a + gate*sigm + frag_a pack (EPI=5 in mm2 for M<=16 and mmpg for M>16!)
        dispatch_mm(e, l.wo_a, frag_b, big_b, M, slots_buf, o_groups, g_stride_bf16 / 8, M * 1024,
                    5u, frag_a, 8192u, big_a, 40960u, 32768u);
        barr();
      }
      mark("hca_wo_a", 2);
      // wo_b: frag_a [M, 8192] -> y_f32 [M, 4096]
      dispatch_mm(e, l.wo_b, frag_a, y_f32, M, nil, 1, 0, 0);
      barr();
      mark("hca_wo_b", 1);
    } else {
      // DSA: attn_in -> big_a [M, 1792]
      dispatch_mm(e, l.attn_in, frag_a, big_a, M, nil, 1, 0, 0);
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
      dispatch_mm(e, l.attn_qb, frag_a, big_a, M, nil, 1, 0, 0);
      barr();
      // dsa_q_prep: q_nope -> frag_b [64, M, 192], q_rope -> q_rope, iq_h -> iq_h
      const uint32_t kb_gstride_bf16 = rb * (192u / 32u) * 512u;
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
      // Indexer score (sub-batched by IDX_SUB=32 so idx_sc stays 32 MB) + topk
      mark("dsa_pre_mm");
      {
        static const bool old_attn = getenv("E5_OLD_ATTN") && atoi(getenv("E5_OLD_ATTN")) != 0;
        for (uint32_t r0 = 0; r0 < M; r0 += IDX_SUB) {
          const uint32_t rM = std::min(IDX_SUB, M - r0);
          struct { uint32_t M, p0, win, topk, cap, row0; } ia{rM, pos, win, idx_topk, cap, r0};
          uint32_t max_n_idx = (pos + r0 + rM > win) ? (pos + r0 + rM - win) : 0u;
          if (max_n_idx > idx_topk) {
            if (old_attn) {
              [e setComputePipelineState:e5_pipe("e5_dsa_idx_score")];
              [e setBuffer:iq_h offset:size_t(r0) * 64 * 128 * 4 atIndex:0];
              [e setBuffer:idx_w offset:size_t(r0) * 64 * 4 atIndex:1];
              [e setBuffer:l.dsa_idx_k8 offset:0 atIndex:2];
              [e setBuffer:idx_sc offset:0 atIndex:3];
              [e setBytes:&ia length:sizeof(ia) atIndex:4];
              [e dispatchThreads:MTLSizeMake(max_n_idx, rM, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            } else {
              [e setComputePipelineState:e5_pipe("e5_dsa_idx_score_mma")];
              [e setBuffer:iq_h offset:0 atIndex:0];
              [e setBuffer:idx_w offset:0 atIndex:1];
              [e setBuffer:l.dsa_idx_k8 offset:0 atIndex:2];
              [e setBuffer:idx_sc offset:0 atIndex:3];
              [e setBytes:&ia length:sizeof(ia) atIndex:4];
              [e dispatchThreadgroups:MTLSizeMake((max_n_idx + 31u) / 32u, rM, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            barr();
          }
          [e setComputePipelineState:e5_pipe("e5_dsa_idx_topk")];
          [e setBuffer:idx_sc offset:0 atIndex:0];
          [e setBuffer:idx_sel offset:0 atIndex:1];
          [e setBuffer:idx_len offset:0 atIndex:2];
          [e setBytes:&ia length:sizeof(ia) atIndex:3];
          [e dispatchThreadgroups:MTLSizeMake(rM, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          barr();
        }
      }
      mark("dsa_idx");
      // Grouped kb: frag_b [64, M, 192] -> big_b [64, M, 512]
      dispatch_mm(e, l.kb, frag_b, big_b, M, slots_buf, 64, kb_gstride_bf16 / 8, M * 512);
      barr();
      mark("dsa_kb");
      // DSA attention -> frag_b [64, M, 512] bf16 frags for vb
      const uint32_t vb_gstride_bf16 = rb * (512u / 32u) * 512u;
      {
        static const bool old_attn = getenv("E5_OLD_ATTN") && atoi(getenv("E5_OLD_ATTN")) != 0;
        if (!old_attn && M <= 8u && (pos + M) >= 512) {
          const uint32_t S = (M == 1u) ? 16u : ((M <= 4u) ? 8u : 4u);
          struct { uint32_t M, topk_win, qabs_gstride, vb_gstride_bf16, S; float scale; } da{
            M, idx_topk + win, M * 512u, vb_gstride_bf16, S, 1.0f / std::sqrt(256.0f)
          };
          [e setComputePipelineState:e5_pipe("e5_dsa_attn_mma_split")];
          [e setBuffer:big_b offset:0 atIndex:0];
          [e setBuffer:q_rope offset:0 atIndex:1];
          [e setBuffer:l.dsa_lat_q8 offset:0 atIndex:2];
          [e setBuffer:idx_sel offset:0 atIndex:3];
          [e setBuffer:idx_len offset:0 atIndex:4];
          [e setBuffer:l.sink.b offset:l.sink.off atIndex:5];
          [e setBuffer:mmpart offset:0 atIndex:6];
          [e setBytes:&da length:sizeof(da) atIndex:7];
          [e dispatchThreadgroups:MTLSizeMake(4u, M, S) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
          barr();
          [e setComputePipelineState:e5_pipe("e5_dsa_attn_merge")];
          [e setBuffer:mmpart offset:0 atIndex:0];
          [e setBuffer:frag_b offset:0 atIndex:1];
          [e setBytes:&da length:sizeof(da) atIndex:2];
          [e dispatchThreadgroups:MTLSizeMake(64u, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
          barr();
        } else {
          struct { uint32_t M, topk_win, qabs_gstride, vb_gstride_bf16; float scale; } da{
            M, idx_topk + win, M * 512u, vb_gstride_bf16, 1.0f / std::sqrt(256.0f)
          };
          [e setComputePipelineState:e5_pipe(old_attn ? "e5_dsa_attn" : "e5_dsa_attn_mma")];
          [e setBuffer:big_b offset:0 atIndex:0];
          [e setBuffer:q_rope offset:0 atIndex:1];
          [e setBuffer:l.dsa_lat_q8 offset:0 atIndex:2];
          [e setBuffer:idx_sel offset:0 atIndex:3];
          [e setBuffer:idx_len offset:0 atIndex:4];
          [e setBuffer:l.sink.b offset:l.sink.off atIndex:5];
          [e setBuffer:frag_b offset:0 atIndex:6];
          [e setBytes:&da length:sizeof(da) atIndex:7];
          [e dispatchThreadgroups:MTLSizeMake(old_attn ? 64u : 4u, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
          barr();
        }
      }
      mark("dsa_attn");
      // Grouped vb: frag_b [64, M, 512] -> big_c [64, M, 256]
      dispatch_mm(e, l.vb, frag_b, big_c, M, slots_buf, 64, vb_gstride_bf16 / 8, M * 256);
      barr();
      // VHA + gate -> frag_a [M, 16384] (matmul2d accelerated)
      {
        struct { uint32_t M, o_gstride; } va{M, M * 256u};
        [e setComputePipelineState:e5_pipe("e5_dsa_vha_gate_mma")];
        [e setBuffer:big_c offset:0 atIndex:0];
        [e setBuffer:l.vha_m.b offset:l.vha_m.off atIndex:1];
        [e setBuffer:big_a offset:0 atIndex:2];
        [e setBuffer:frag_a offset:0 atIndex:3];
        [e setBytes:&va length:sizeof(va) atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(4u, 2u, M) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        barr();
      }
      // attn_o: frag_a [M, 16384] -> y_f32 [M, 4096]
      dispatch_mm(e, l.attn_o, frag_a, y_f32, M, nil, 1, 0, 0);
      barr();
      mark("dsa_post_mm");
    }

    // FFN mHC pre (inject attention output y_f32)
    dispatch_mhc(M, 1u, 0u, l.mhc_fn, l.mhc_base, l.mhc_scale, l.ffn_norm);
    mark("mhc");

    // FFN / MoE
    if (!l.is_moe) {
      // Dense L0: mlp_gu + fused SiTU-GLU EPI=3 in prefill (M > MAXM), writes to frag_b so X=frag_a isn't clobbered
      if (M > MAXM) {
        dispatch_mm(e, l.mlp_gu, frag_a, big_a, M, nil, 1, 0, 0, 3u, frag_b, I);
        barr();
        dispatch_mm(e, l.mlp_dn, frag_b, y_f32, M, nil, 1, 0, 0);
        barr();
      } else {
        dispatch_mm(e, l.mlp_gu, frag_a, big_a, M, nil, 1, 0, 0);
        barr();
        struct { uint32_t M, I, src_stride, src_off, dst_K, dst_off; float beta, lbeta; } sa{
          M, I, 2 * I, 0u, I, 0u, situ_beta, situ_lbeta
        };
        [e setComputePipelineState:e5_pipe("e5_situ_pack")];
        [e setBuffer:big_a offset:0 atIndex:0];
        [e setBuffer:frag_a offset:0 atIndex:1];
        [e setBytes:&sa length:sizeof(sa) atIndex:2];
        [e dispatchThreads:MTLSizeMake(I, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barr();
        dispatch_mm(e, l.mlp_dn, frag_a, y_f32, M, nil, 1, 0, 0);
        barr();
      }
    } else {
      // MoE: moe_in -> big_a [M, 7168]
      dispatch_mm(e, l.moe_in, frag_a, big_a, M, nil, 1, 0, 0);
      barr();
      mark("moe_in_mm", 2);
      // Router top-10 (fused with fc1 -> frag_a pack for M == 1)
      {
        struct { uint32_t M, E, topk; } ra{M, E, topk};
        if (M == 1) {
          [e setComputePipelineState:e5_pipe("e5_route_topk_pack_m1")];
          [e setBuffer:big_a offset:0 atIndex:0];
          [e setBuffer:l.moe_bias.b offset:l.moe_bias.off atIndex:1];
          [e setBuffer:l.moe_rscale.b offset:l.moe_rscale.off atIndex:2];
          [e setBuffer:top_ids offset:0 atIndex:3];
          [e setBuffer:top_w offset:0 atIndex:4];
          [e setBuffer:frag_a offset:0 atIndex:5];
          [e setBytes:&ra length:sizeof(ra) atIndex:6];
          [e dispatchThreadgroups:MTLSizeMake(9, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
          barr();
        } else {
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
      }
      mark("moe_route", 2);
      if (M == 1) {
        mark("moe_in");
        // Fast single-token decode path (mm1 SIMD matvec grouped, nz=10; frag_a already packed)
        static const bool no_mm1 = getenv("E5_NO_MM1") && atoi(getenv("E5_NO_MM1")) != 0;
        if (!no_mm1 && !hadamard) {
          // Fused moe_gu + SiTU-GLU + frag_b pack (eliminates big_c round-trip & e5_situ_pack_m1)
          dispatch_mm1(e, l.moe_gu, frag_a, frag_b, top_ids, 0, topk, 0u, 32768u, 2u);
          barr();
        } else if (!no_mm1) {
          dispatch_mm1(e, l.moe_gu, frag_a, big_c, top_ids, 0, topk, 0u, 4096u);
          barr();
        } else {
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
          [e setBuffer:top_ids offset:0 atIndex:6];
          [e setBuffer:nslots_buf offset:0 atIndex:7];
          [e setBuffer:counters offset:0 atIndex:8];
          [e setBuffer:mmpart offset:0 atIndex:9];
          [e setBuffer:mmpart offset:0 atIndex:10];
          [e setBuffer:mmpart offset:0 atIndex:11];
          [e setBuffer:mmpart offset:0 atIndex:12];
          [e setBuffer:counters offset:0 atIndex:13];
          [e setBuffer:counters offset:0 atIndex:14];
          [e setBuffer:counters offset:0 atIndex:15];
          uint32_t nft = 1u;
          while (l.moe_gu.sgs * nft < 4u && (l.moe_gu.N / 32u) % (nft * 2u) == 0) nft *= 2u;
          [e setThreadgroupMemoryLength:(l.moe_gu.sgs > 1 ? size_t(l.moe_gu.sgs) * nft * 32 * 16 * 4 : 16) atIndex:0];
          [e dispatchThreadgroups:MTLSizeMake(l.moe_gu.N / 32u / nft, l.moe_gu.ksplit, topk)
            threadsPerThreadgroup:MTLSizeMake(32u * l.moe_gu.sgs * nft, 1, 1)];
          barr();
        }
        if (hadamard) {
          for (uint32_t k = 0; k < topk; k++) {
            struct { uint32_t M, I, src_stride, src_off, dst_K, dst_off; float beta, lbeta; } sa{
              1u, 2048u, 4096u, k * 4096u, 2048u, 0u, situ_beta, situ_lbeta
            };
            [e setComputePipelineState:e5_pipe("e5_situ_had_pack")];
            [e setBuffer:big_c offset:0 atIndex:0];
            [e setBuffer:frag_b offset:size_t(k) * 32768 * 2 atIndex:1];
            [e setBytes:&sa length:sizeof(sa) atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          }
          barr();
        } else if (no_mm1) {
          struct { uint32_t I, topk, slab_bf16; float beta, lbeta; } sa{
            2048u, topk, 32768u, situ_beta, situ_lbeta
          };
          [e setComputePipelineState:e5_pipe("e5_situ_pack_m1")];
          [e setBuffer:big_c offset:0 atIndex:0];
          [e setBuffer:frag_b offset:0 atIndex:1];
          [e setBytes:&sa length:sizeof(sa) atIndex:2];
          [e dispatchThreads:MTLSizeMake(2048, topk, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          barr();
        }
        mark("moe_gu");
        if (!no_mm1) {
          dispatch_mm1(e, l.moe_dn, frag_b, big_b, top_ids, 0, topk, 32768u / 8u, 2048u);
          barr();
        } else {
          const uint32_t raw_bits = (l.moe_dn.bits == 2 || l.moe_dn.bits == 20) ? 2u : l.moe_dn.bits;
          const uint32_t wunit = (raw_bits == 2 || raw_bits == 6) ? 8u : 16u;
          MmArgs a{};
          a.N = l.moe_dn.N; a.K = l.moe_dn.K; a.M = 1; a.ksplit = l.moe_dn.ksplit;
          a.wstride = uint32_t(size_t(l.moe_dn.N) * l.moe_dn.K * raw_bits / 8 / wunit);
          a.sstride = uint32_t(size_t(l.moe_dn.N) * (l.moe_dn.K / l.moe_dn.gs));
          a.xstride = 32768u / 8u;
          a.pstride = l.moe_dn.N * l.moe_dn.ksplit; a.ystride = 2048u; a.sgs = l.moe_dn.sgs;
          a.psplit = l.moe_dn.N;
          [e setComputePipelineState:mm2_pipe(l.moe_dn.bits, l.moe_dn.gs, true, false)];
          [e setBuffer:l.moe_dn.w offset:l.moe_dn.woff atIndex:0];
          [e setBuffer:l.moe_dn.s offset:l.moe_dn.soff atIndex:1];
          [e setBuffer:frag_b offset:0 atIndex:2];
          [e setBuffer:mmpart offset:0 atIndex:3];
          [e setBuffer:big_b offset:0 atIndex:4];
          [e setBytes:&a length:sizeof(a) atIndex:5];
          [e setBuffer:top_ids offset:0 atIndex:6];
          [e setBuffer:nslots_buf offset:0 atIndex:7];
          [e setBuffer:counters offset:0 atIndex:8];
          [e setBuffer:mmpart offset:0 atIndex:9];
          [e setBuffer:mmpart offset:0 atIndex:10];
          [e setBuffer:mmpart offset:0 atIndex:11];
          [e setBuffer:mmpart offset:0 atIndex:12];
          [e setBuffer:counters offset:0 atIndex:13];
          [e setBuffer:counters offset:0 atIndex:14];
          [e setBuffer:counters offset:0 atIndex:15];
          uint32_t nft = 1u;
          while (l.moe_dn.sgs * nft < 4u && (l.moe_dn.N / 32u) % (nft * 2u) == 0) nft *= 2u;
          [e setThreadgroupMemoryLength:(l.moe_dn.sgs > 1 ? size_t(l.moe_dn.sgs) * nft * 32 * 16 * 4 : 16) atIndex:0];
          [e dispatchThreadgroups:MTLSizeMake(l.moe_dn.N / 32u / nft, l.moe_dn.ksplit, topk)
            threadsPerThreadgroup:MTLSizeMake(32u * l.moe_dn.sgs * nft, 1, 1)];
          barr();
        }
        mark("moe_dn_m1", 2);
        {
          struct { uint32_t M, topk, hadamard; float eps, beta, lbeta; } ca{1u, topk, hadamard ? 1u : 0u, eps, situ_beta, situ_lbeta};
          [e setComputePipelineState:e5_pipe("e5_moe_combine_pack_m1")];
          [e setBuffer:big_b offset:0 atIndex:0];
          [e setBuffer:top_w offset:0 atIndex:1];
          [e setBuffer:l.moe_lnorm.b offset:l.moe_lnorm.off atIndex:2];
          [e setBuffer:big_a offset:0 atIndex:3];
          [e setBuffer:frag_a offset:0 atIndex:4];
          [e setBytes:&ca length:sizeof(ca) atIndex:5];
          [e dispatchThreadgroups:MTLSizeMake(17, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
          barr();
        }
      } else {
        // Batched MoE path (prefill & speculative verify): plan_lists + mmpg (CH=1 NF=4 + fused EPI=3!)
        {
          struct { uint32_t M, K, src_stride, src_off; } pa{M, 2048u, 7168u, 1024u};
          [e setComputePipelineState:e5_pipe("e5_pack_f32")];
          [e setBuffer:big_a offset:0 atIndex:0];
          [e setBuffer:frag_a offset:0 atIndex:1];
          [e setBytes:&pa length:sizeof(pa) atIndex:2];
          [e dispatchThreads:MTLSizeMake(2048, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          barr();
        }
        {
          static const int mmpg_mode = getenv("E5_MMPG_MODE") ? atoi(getenv("E5_MMPG_MODE")) : 0;
          const uint32_t tile_r = (mmpg_mode == 1) ? 64u : ((mmpg_mode == 2) ? 32u : 16u);
          const uint32_t nf = (mmpg_mode == 0) ? 4u : 2u;
          struct { uint32_t M, E, topk, tile_rows, gx_gu, gx_dn; } la{
            M, E, topk, tile_r, (l.moe_gu.N / 32u + nf - 1u) / nf, (l.moe_dn.N / 32u + nf - 1u) / nf
          };
          [e setComputePipelineState:e5_pipe("e5_plan_lists")];
          [e setBuffer:top_ids offset:0 atIndex:0];
          [e setBuffer:moe_slots offset:0 atIndex:1];
          [e setBuffer:moe_nslots offset:0 atIndex:2];
          [e setBuffer:roff offset:0 atIndex:3];
          [e setBuffer:rcnt offset:0 atIndex:4];
          [e setBuffer:rlist offset:0 atIndex:5];
          [e setBuffer:cidx offset:0 atIndex:6];
          [e setBytes:&la length:sizeof(la) atIndex:7];
          [e setBuffer:wtile offset:0 atIndex:8];
          [e setBuffer:moe_ind offset:0 atIndex:9];
          [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
          barr();
        }
        mark("moe_plan", 2);
        if (!hadamard) {
          // Fused moe_gu + SiTU-GLU EPI=3 straight from registers into frag_b!
          dispatch_mmpg_moe(e, l.moe_gu, frag_a, big_c, M, 0u, 0u, 3u, frag_b, 2048u);
          barr();
        } else {
          dispatch_mmpg_moe(e, l.moe_gu, frag_a, big_c, M, 0u, 0u);
          barr();
          struct { uint32_t M, I, src_stride, src_off, dst_K, dst_off; float beta, lbeta; } sa{
            M * topk, 2048u, 4096u, 0u, 2048u, 0u, situ_beta, situ_lbeta
          };
          [e setComputePipelineState:e5_pipe("e5_situ_had_pack")];
          [e setBuffer:big_c offset:0 atIndex:0];
          [e setBuffer:frag_b offset:0 atIndex:1];
          [e setBytes:&sa length:sizeof(sa) atIndex:2];
          [e dispatchThreadgroups:MTLSizeMake(M * topk, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          barr();
        }
        mark("moe_gu_p", 2);
        dispatch_mmpg_moe(e, l.moe_dn, frag_b, big_b, M, 2u, 12u);
        barr();
        mark("moe_dn_p", 2);
        {
          struct { uint32_t M, topk, hadamard; float eps, beta, lbeta; } ca{M, topk, hadamard ? 1u : 0u, eps, situ_beta, situ_lbeta};
          [e setComputePipelineState:e5_pipe("e5_moe_combine_c_pack")];
          [e setBuffer:big_b offset:0 atIndex:0];
          [e setBuffer:cidx offset:0 atIndex:1];
          [e setBuffer:top_w offset:0 atIndex:2];
          [e setBuffer:l.moe_lnorm.b offset:l.moe_lnorm.off atIndex:3];
          [e setBuffer:big_a offset:0 atIndex:4];
          [e setBuffer:frag_a offset:0 atIndex:5];
          [e setBytes:&ca length:sizeof(ca) atIndex:6];
          [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
          barr();
        }
      }
      // moe_out: frag_a [M, 4096] -> y_f32 [M, 4096]
      mark("moe_comb", 1);
      dispatch_mm(e, l.moe_out, frag_a, y_f32, M, nil, 1, 0, 0);
      barr();
      mark("moe_out");
    }
    if (!prof && ((M > MAXM && (lid == 4 || lid == 14 || lid == 24 || lid == 34)) || (M <= MAXM && lid == 21))) flush_cb();
  }

  if (need_logits) {
    const uint32_t lm_M = last_only ? 1u : M;
    const size_t row_off = last_only ? size_t(M - 1) : 0u;
    // Final mHC head contraction (inject last layer's FFN output) + model.norm -> frag_a [lm_M, H]
    dispatch_mhc(lm_M, 1u, 1u, hch_fn, hch_base, hch_scale, norm, row_off);

    // lm_head -> logits_buf [lm_M, vocab]
    dispatch_mm(e, lm_head, frag_a, logits_buf, lm_M, nil, 1, 0, 0);
    barr();

    // MultiMax
    {
      struct { uint32_t M, V; } ma{lm_M, vocab};
      [e setComputePipelineState:e5_pipe("e5_multimax")];
      [e setBuffer:logits_buf offset:0 atIndex:0];
      [e setBuffer:mm_r.b offset:mm_r.off atIndex:1];
      [e setBuffer:mm_t.b offset:mm_t.off atIndex:2];
      [e setBytes:&ma length:sizeof(ma) atIndex:3];
      [e dispatchThreads:MTLSizeMake(vocab, lm_M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
  }

  if (use_m4) {
    mtl4.commit_and_wait();
    if (gpu_ms) *gpu_ms = (dankattn::host_now() - m4_t0) * 1e3;
    pos += M;
    return need_logits ? (const float*)[logits_buf contents] : nullptr;
  }

  [e endEncoding];
  [cb commit];
  cbs.push_back(cb);
  [cb waitUntilCompleted];
  for (id<MTLCommandBuffer> c : cbs) {
    if ([c status] != MTLCommandBufferStatusCompleted) mtl::die("command buffer failed", [c error]);
  }
  if (prof) {
    pt["tail"] += ([cb GPUEndTime] - [cb GPUStartTime]) * 1e3;
    double tot = 0;
    for (auto& kv : pt) tot += kv.second;
    if (gpu_ms) *gpu_ms = tot;
    static int printed = 0;
    if (printed < 4 || (M == 1 && printed < 6)) {
      printed++;
      fprintf(stderr, "[PROF M=%u pos=%u tot=%.2fms]", M, pos, tot);
      for (auto& kv : pt) fprintf(stderr, " %s=%.1f", kv.first.c_str(), kv.second);
      fprintf(stderr, "\n");
    }
  } else {
    if (gpu_ms) {
      double t_sum = 0.0;
      for (id<MTLCommandBuffer> c : cbs) t_sum += ([c GPUEndTime] - [c GPUStartTime]) * 1e3;
      *gpu_ms = t_sum;
    }
  }

  pos += M;
  return need_logits ? (const float*)[logits_buf contents] : nullptr;
}

}  // namespace ernie55
