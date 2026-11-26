// Minimal Metal runtime for DankAttn: device, buffers, runtime-compiled pipelines, timing.
// Kernels are compiled from source at startup with math mode pinned to safe, so the
// numeric contract does not depend on fast-math reassociation.
#pragma once
#include <atomic>
#include <chrono>
#include <mach-o/dyld.h>
#include <sys/sysctl.h>
#include <stdlib.h>
#include <limits.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace mtl {

[[noreturn]] inline void die(const char* what, NSError* e = nil) {
  fprintf(stderr, "fatal: %s%s%s\n", what, e ? ": " : "", e ? [[e description] UTF8String] : "");
  exit(1);
}

struct Ctx {
  id<MTLDevice> dev;
  id<MTLCommandQueue> q;
  std::map<std::string, id<MTLLibrary>> lib_cache;
  std::map<std::string, id<MTLComputePipelineState>> cache;

  Ctx() {
    dev = MTLCreateSystemDefaultDevice();
    if (!dev) die("no Metal device");
    q = [dev newCommandQueueWithMaxCommandBufferCount:64];
  }

  size_t allocated = 0;  // bytes handed out by buf()
  bool track = false;                 // collect the buffers buf() makes (for a residency set)
  std::vector<id<MTLBuffer>> made;
  id<MTLBuffer> buf(size_t bytes, int line = __builtin_LINE()) {
    id<MTLBuffer> b = [dev newBufferWithLength:(bytes ? bytes : 16) options:MTLResourceStorageModeShared];
    if (!b) die("buffer alloc");
    allocated += bytes;
    if (track) made.push_back(b);
    static const bool log = getenv("DANKATTN_BUFLOG") != nullptr || getenv("OPUS_BUFLOG") != nullptr;
    if (log && bytes >= (size_t(48) << 20)) fprintf(stderr, "buf line %d: %.0f MB\n", line, bytes / 1048576.0);
    return b;
  }

  id<MTLHeap> heap(size_t bytes) {
    MTLHeapDescriptor* d = [MTLHeapDescriptor new];
    d.type = MTLHeapTypePlacement;
    d.storageMode = MTLStorageModeShared;
    d.hazardTrackingMode = MTLHazardTrackingModeTracked;
    d.size = (bytes + 16383) & ~size_t(16383);
    id<MTLHeap> h = [dev newHeapWithDescriptor:d];
    if (!h) die("heap alloc");
    allocated += d.size;
    return h;
  }

  // Compile `src` with the given preprocessor defines and return pipeline `fn`.
  id<MTLComputePipelineState> pipe(const std::string& src, const std::string& fn,
                                   const std::map<std::string, std::string>& defs = {}) {
    std::string lib_key = std::to_string(std::hash<std::string>{}(src));
    for (auto& [k, v] : defs) lib_key += "|" + k + "=" + v;
    std::string key = fn + "|" + lib_key;
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    NSError* e = nil;
    id<MTLLibrary> lib = nil;
    auto lit = lib_cache.find(lib_key);
    if (lit != lib_cache.end()) {
      lib = lit->second;
    } else {
      MTLCompileOptions* o = [MTLCompileOptions new];
      o.languageVersion = MTLLanguageVersion4_0;
      o.mathMode = MTLMathModeSafe;
      o.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
      NSMutableDictionary* m = [NSMutableDictionary new];
      for (auto& [k, v] : defs) m[@(k.c_str())] = @(v.c_str());
      o.preprocessorMacros = m;
      lib = [dev newLibraryWithSource:@(src.c_str()) options:o error:&e];
      if (!lib) die(("compile " + fn).c_str(), e);
      lib_cache[lib_key] = lib;
    }
    id<MTLFunction> f = [lib newFunctionWithName:@(fn.c_str())];
    if (!f) die(("no function " + fn).c_str());
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&e];
    if (!p) die(("pipeline " + fn).c_str(), e);
    cache[key] = p;
    return p;
  }
};

// The source tree this binary belongs to: DANKATTN_ROOT (or OPUS_ROOT), else the parent of the executable's
// directory (build/dankattn-hy4 -> the tree). Kernels are read from there, never from the working
// directory, so a binary cannot pick up another tree's kernels.
inline const std::string& tree_root() {
  static const std::string r = [] {
    if (const char* e = getenv("DANKATTN_ROOT")) return std::string(e);
    if (const char* e = getenv("OPUS_ROOT")) return std::string(e);
    char buf[4096], real[4096];
    uint32_t n = sizeof buf;
    if (_NSGetExecutablePath(buf, &n) != 0 || !realpath(buf, real)) return std::string(".");
    std::string p(real);
    p = p.substr(0, p.rfind('/'));
    return p.substr(0, p.rfind('/'));
  }();
  return r;
}

// Seconds (steady clock) of the last GPU work in this process: the residency refresh keeps the
// pack wired only for a window after it (OPUS_RESIDENCY_WINDOW_S, default 900; 0 = always).
inline std::atomic<int64_t>& last_use() { static std::atomic<int64_t> t{0}; return t; }
inline int64_t now_s() { return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
inline bool keep_warm() {
  static const int64_t win = [] {
    if (const char* e = getenv("DANKATTN_RESIDENCY_WINDOW_S")) return atoll(e);
    if (const char* e = getenv("OPUS_RESIDENCY_WINDOW_S")) return atoll(e);
    return int64_t(900);
  }();
  return win <= 0 || now_s() - last_use().load() < win;
}

inline std::string slurp(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) die(("open " + path).c_str());
  std::string s;
  char b[1 << 16];
  size_t n;
  while ((n = fread(b, 1, sizeof b, f)) > 0) s.append(b, n);
  fclose(f);
  return s;
}

// GPU time of one command buffer in seconds.
inline double gpu_seconds(id<MTLCommandBuffer> cb) { return cb.GPUEndTime - cb.GPUStartTime; }

// Warn loudly at startup if total wired footprint (mapped weights + state) exceeds iogpu.wired_limit_mb
// (or recommendedMaxWorkingSetSize when the sysctl is unset/0, e.g. after a reboot).
inline void check_wired_limit(size_t need_bytes, id<MTLDevice> dev) {
  uint64_t mb = 0;
  size_t sz = sizeof(mb);
  if (sysctlbyname("iogpu.wired_limit_mb", &mb, &sz, nullptr, 0) != 0) mb = 0;
  const uint64_t rec = dev ? uint64_t(dev.recommendedMaxWorkingSetSize) : 0;
  const uint64_t lim = mb > 0 ? mb * 1048576ull : rec;
  if (lim > 0 && need_bytes > lim) {
    const uint64_t suggest_mb = (need_bytes / 1048576ull) + 2048ull;
    fprintf(stderr,
            "WARNING: wired footprint %.2f GiB exceeds %s (%.2f GiB) — expect swap/slowdown or OOM!\n"
            "         Run: sudo sysctl iogpu.wired_limit_mb=%llu\n",
            need_bytes / 1073741824.0, mb > 0 ? "iogpu.wired_limit_mb" : "recommendedMaxWorkingSetSize",
            lim / 1073741824.0, (unsigned long long)suggest_mb);
  }
}

inline void check_pack_filename(const std::string& fn) {
  if (fn.empty() || fn == "." || fn == ".." || fn.find('/') != std::string::npos ||
      fn.find('\\') != std::string::npos || fn.find("..") != std::string::npos) {
    die(("invalid manifest filename: " + fn).c_str());
  }
  for (unsigned char c : fn) {
    if (c < 0x20 || c == 0x7f) die(("invalid control char in manifest filename: " + fn).c_str());
  }
}

inline id<MTLBuffer> checked_file(const std::vector<id<MTLBuffer>>& files, uint32_t idx,
                                  uint64_t off, uint64_t bytes, const std::string& name) {
  if (idx >= files.size()) die(("manifest file index out of range for " + name).c_str());
  id<MTLBuffer> b = files[idx];
  const uint64_t len = uint64_t(b.length);
  if (off > len || bytes > len - off) die(("manifest tensor offset out of range for " + name).c_str());
  return b;
}

}  // namespace mtl
