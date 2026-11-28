// Shared pack, weight-descriptor, mmap, JSON, and host-clock primitives for DankAttn engines.
#pragma once
#include "../mtl.h"
#import <Foundation/Foundation.h>
#include <cstdint>
#include <fcntl.h>
#include <mach/mach_time.h>
#include <map>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace dankattn {

// Quantized register-direct matmul weight descriptor (shared across all DankAttn models).
struct Lin {
  id<MTLBuffer> w = nil, s = nil;
  size_t woff = 0, soff = 0;
  uint32_t N = 0, K = 0, bits = 0, gs = 0, experts = 1, ksplit = 1, sgs = 1;
  uint32_t pslot = 0;  // split-K scratch region for concurrent dispatches (e.g. Inkling2 qgrs || k || v)
  std::map<std::string, std::pair<uint32_t, uint32_t>> rows;  // optional named row slices (Hy4 / Flash-Next)
};

// Raw tensor slice inside a pack file buffer.
struct Raw {
  id<MTLBuffer> b = nil;
  size_t off = 0, bytes = 0;
};

// Load and parse a JSON file into an NSDictionary.
inline NSDictionary* jload(const std::string& p) {
  NSData* d = [NSData dataWithContentsOfFile:@(p.c_str())];
  if (!d) mtl::die(("read " + p).c_str());
  NSError* e = nil;
  NSDictionary* j = [NSJSONSerialization JSONObjectWithData:d options:0 error:&e];
  if (!j) mtl::die(("parse " + p).c_str(), e);
  return j;
}

// Parse an in-memory JSON string into an NSDictionary.
inline NSDictionary* parse_json(const std::string& s) {
  NSData* d = [NSData dataWithBytes:s.data() length:s.size()];
  NSError* e = nil;
  NSDictionary* j = [NSJSONSerialization JSONObjectWithData:d options:0 error:&e];
  if (!j) mtl::die("parse json", e);
  return j;
}

// Memory-map a pack file into a zero-copy MTLBuffer (page-aligned, munmap on dealloc).
inline id<MTLBuffer> mmap_file(id<MTLDevice> dev, const std::string& path, size_t& out_bytes,
                               int map_flags = MAP_PRIVATE) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) mtl::die(("open " + path).c_str());
  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    mtl::die(("stat " + path).c_str());
  }
  const size_t sz = size_t(st.st_size);
  const size_t pg = size_t(getpagesize());
  const size_t msz = (sz + pg - 1) & ~(pg - 1);
  void* ptr = mmap(nullptr, msz, PROT_READ, map_flags, fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) mtl::die(("mmap " + path).c_str());
  id<MTLBuffer> buf = [dev newBufferWithBytesNoCopy:ptr
                                             length:msz
                                            options:MTLResourceStorageModeShared
                                        deallocator:^(void* p, NSUInteger l) { munmap(p, l); }];
  if (!buf) {
    munmap(ptr, msz);
    mtl::die(("metal mmap buffer " + path).c_str());
  }
  out_bytes += sz;
  return buf;
}

// Host clock in seconds on the same timebase as MTLCommandBuffer GPUStartTime / GPUEndTime.
inline double host_now() {
  static mach_timebase_info_data_t tb;
  if (!tb.denom) mach_timebase_info(&tb);
  return double(mach_absolute_time()) * tb.numer / tb.denom / 1e9;
}

}  // namespace dankattn
