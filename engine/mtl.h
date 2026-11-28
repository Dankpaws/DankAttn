// Minimal Metal runtime for DankAttn: device, buffers, runtime-compiled pipelines, timing.
// Kernels are compiled from source at startup with math mode pinned to safe, so the
// numeric contract does not depend on fast-math reassociation.
#pragma once
#include <atomic>
#include <chrono>
#include <mach-o/dyld.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <stdlib.h>
#include <limits.h>
#include <unistd.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
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
  std::map<std::string, id<MTLBinaryArchive>> bin_archives;
  std::set<std::string> dirty_archives;
  std::string shader_cache_dir;
  std::string device_os_tag;
  uint32_t archive_hits = 0, archive_misses = 0;

  static uint64_t fnv1a_str(const std::string& s, uint64_t h = 1469598103934665603ull) {
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
  }

  static void mkdir_p(const std::string& dir) {
    for (size_t i = 1; i < dir.size(); i++) {
      if (dir[i] == '/') mkdir(dir.substr(0, i).c_str(), 0700);
    }
    mkdir(dir.c_str(), 0700);
  }

  Ctx() {
    dev = MTLCreateSystemDefaultDevice();
    if (!dev) die("no Metal device");
    q = [dev newCommandQueueWithMaxCommandBufferCount:64];
    const char* sc = getenv("DANKATTN_SHADER_CACHE");
    if (sc) {
      if (std::string(sc) != "0" && *sc) shader_cache_dir = sc;
    } else if (const char* home = getenv("HOME")) {
      shader_cache_dir = std::string(home) + "/Library/Caches/DankAttn/metallib";
    }
    if (!shader_cache_dir.empty()) {
      char osver[128] = {};
      size_t osz = sizeof(osver) - 1;
      if (sysctlbyname("kern.osversion", osver, &osz, nullptr, 0) != 0) osver[0] = 0;
      const char* dname = dev.name ? [dev.name UTF8String] : "metal";
      device_os_tag = std::string(dname) + "|" + std::string(osver) + "|msl4_safe_v1";
      mkdir_p(shader_cache_dir);
    }
  }

  ~Ctx() { flush_shader_cache(); }

  void flush_shader_cache() {
    if (shader_cache_dir.empty() || dirty_archives.empty()) return;
    for (const auto& path : dirty_archives) {
      auto it = bin_archives.find(path);
      if (it == bin_archives.end() || !it->second) continue;
      const std::string tmp = path + ".tmp." + std::to_string(getpid());
      NSURL* url = [NSURL fileURLWithPath:@(tmp.c_str())];
      NSError* err = nil;
      if ([it->second serializeToURL:url error:&err]) {
        rename(tmp.c_str(), path.c_str());
      } else {
        unlink(tmp.c_str());
      }
    }
    dirty_archives.clear();
  }

  id<MTLBinaryArchive> get_archive(const std::string& lib_key, std::string& out_path) {
    if (shader_cache_dir.empty()) return nil;
    const uint64_t h = fnv1a_str(lib_key, fnv1a_str(device_os_tag));
    char buf[32];
    snprintf(buf, sizeof buf, "/%016llx.metallib", (unsigned long long)h);
    out_path = shader_cache_dir + buf;
    auto it = bin_archives.find(out_path);
    if (it != bin_archives.end()) return it->second;
    MTLBinaryArchiveDescriptor* d = [MTLBinaryArchiveDescriptor new];
    struct stat st{};
    if (stat(out_path.c_str(), &st) == 0 && st.st_size > 0) {
      d.url = [NSURL fileURLWithPath:@(out_path.c_str())];
    }
    NSError* err = nil;
    id<MTLBinaryArchive> ba = [dev newBinaryArchiveWithDescriptor:d error:&err];
    if (!ba && d.url) {
      // Corrupt or stale archive from an OS update: recreate cleanly.
      unlink(out_path.c_str());
      d.url = nil;
      ba = [dev newBinaryArchiveWithDescriptor:d error:&err];
    }
    if (ba) bin_archives[out_path] = ba;
    return ba;
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
    id<MTLComputePipelineState> p = nil;
    std::string apath;
    id<MTLBinaryArchive> ba = get_archive(lib_key, apath);
    if (ba) {
      MTLComputePipelineDescriptor* pd = [MTLComputePipelineDescriptor new];
      pd.computeFunction = f;
      pd.binaryArchives = @[ba];
      NSError* herr = nil;
      p = [dev newComputePipelineStateWithDescriptor:pd
                                             options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                          reflection:nil
                                               error:&herr];
      if (p) {
        archive_hits++;
      } else {
        archive_misses++;
        p = [dev newComputePipelineStateWithDescriptor:pd options:MTLPipelineOptionNone reflection:nil error:&e];
        if (!p) die(("pipeline " + fn).c_str(), e);
        NSError* aerr = nil;
        if ([ba addComputePipelineFunctionsWithDescriptor:pd error:&aerr]) {
          dirty_archives.insert(apath);
        }
      }
    } else {
      p = [dev newComputePipelineStateWithFunction:f error:&e];
      if (!p) die(("pipeline " + fn).c_str(), e);
    }
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

// Read a file (and recursively inline local `#include "..."` directives relative to its directory)
// so Metal shader sources compiled via newLibraryWithSource can share modular headers.
inline std::string slurp(const std::string& path, int depth = 0) {
  if (depth > 8) die(("include cycle in " + path).c_str());
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) die(("open " + path).c_str());
  std::string raw;
  char b[1 << 16];
  size_t n;
  while ((n = fread(b, 1, sizeof b, f)) > 0) raw.append(b, n);
  fclose(f);
  if (raw.find("#include \"") == std::string::npos) return raw;
  const size_t slash = path.rfind('/');
  const std::string dir = (slash == std::string::npos) ? "." : path.substr(0, slash);
  std::string out;
  out.reserve(raw.size());
  size_t pos = 0;
  while (pos < raw.size()) {
    size_t nl = raw.find('\n', pos);
    const size_t end = (nl == std::string::npos) ? raw.size() : nl + 1;
    size_t p = pos;
    while (p < end && (raw[p] == ' ' || raw[p] == '\t')) p++;
    if (end - p >= 11 && raw.compare(p, 10, "#include \"") == 0) {
      size_t q = raw.find('"', p + 10);
      if (q != std::string::npos && q < end) {
        const std::string rel = raw.substr(p + 10, q - (p + 10));
        out.append(slurp(dir + "/" + rel, depth + 1));
        if (!out.empty() && out.back() != '\n') out.push_back('\n');
        pos = end;
        continue;
      }
    }
    out.append(raw, pos, end - pos);
    pos = end;
  }
  return out;
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
