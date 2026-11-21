#include "ngram.h"
#import <Foundation/Foundation.h>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <map>
#include <dispatch/dispatch.h>

namespace opus {

static inline float bf2f(uint16_t v) { uint32_t u = uint32_t(v) << 16; float f; memcpy(&f, &u, 4); return f; }
static inline uint16_t f2bf(float f) {
  uint32_t u; memcpy(&u, &f, 4);
  return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

NgramTable::NgramTable(const std::string& p, uint32_t hidden) : H(hidden) {
  NSDictionary* j = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@(p.c_str())] options:0 error:nil];
  if (!j) { fprintf(stderr, "fatal: ngram table %s\n", p.c_str()); exit(1); }
  ngram_size = [j[@"ngram_size"] unsignedIntValue];
  heads_per = [j[@"heads_per_ngram"] unsignedIntValue];
  dim = [j[@"dim"] unsignedIntValue];
  group = [j[@"group"] unsignedIntValue];
  eos = [j[@"eos"] unsignedIntValue];
  for (NSNumber* n in j[@"head_sizes"]) head_size.push_back([n longLongValue]);
  for (NSNumber* n in j[@"head_offsets"]) head_off.push_back([n longLongValue]);
  for (NSNumber* n in j[@"mult"]) mult.push_back([n longLongValue]);
  std::map<std::string, const uint8_t*> maps;
  for (NSDictionary* s in j[@"shards"]) {
    std::string f = [s[@"file"] UTF8String];
    if (!maps.count(f)) {
      int fd = open(f.c_str(), O_RDONLY);
      struct stat st;
      fstat(fd, &st);
      void* m = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
      close(fd);
      if (m == MAP_FAILED) { fprintf(stderr, "fatal: mmap %s\n", f.c_str()); exit(1); }
      maps[f] = (const uint8_t*)m;
    }
    const uint8_t* base = maps[f];
    Shard sh{base + [s[@"w"] unsignedLongLongValue], base + [s[@"s"] unsignedLongLongValue],
             base + [s[@"b"] unsignedLongLongValue], [s[@"rows"] unsignedLongLongValue]};
    shards.push_back(sh);
  }
  rows_per_shard = shards[0].rows;
}

void NgramTable::rows_f32(const std::vector<uint32_t>& hist, uint32_t M, float* out) const {
  const uint32_t heads = (ngram_size - 1) * heads_per, words = dim / 8, groups = dim / group;
  const size_t n = hist.size();
  // hash every (row, head) first, then fetch the rows concurrently: each fetch is a random
  // read from an SSD-resident table, so overlapping them hides the latency
  std::vector<uint64_t> rowid(size_t(M) * heads);
  for (uint32_t r = 0; r < M; r++) {
    const size_t t = n - M + r;
    int64_t sh[8];
    for (uint32_t s = 0; s < ngram_size; s++) {
      bool ok = true;
      for (uint32_t q = 1; q <= s; q++) {
        if (t < q || hist[t - q] == eos) { ok = false; break; }
      }
      sh[s] = (s == 0) ? int64_t(hist[t]) : (ok ? int64_t(hist[t - s]) : int64_t(eos));
    }
    for (uint32_t g = 2; g <= ngram_size; g++) {
      uint64_t mixed = uint64_t(sh[0]) * uint64_t(mult[0]);  // int64 wrap-around
      for (uint32_t q = 1; q < g; q++) mixed ^= uint64_t(sh[q]) * uint64_t(mult[q]);
      const int64_t sm = int64_t(mixed);
      for (uint32_t hh = 0; hh < heads_per; hh++) {
        const uint32_t head = (g - 2) * heads_per + hh;
        int64_t rem = sm % head_size[head];
        if (rem < 0) rem += head_size[head];  // torch.remainder
        rowid[size_t(r) * heads + head] = uint64_t(rem + head_off[head]);
      }
    }
  }
  // OPUS_NG_WILLNEED=0 turns it off. Ask for every page the rows touch (codes, scales, biases: three regions) before
  // reading any, so the SSD gets them all at once instead of one blocked fetch thread per page
  static const bool willneed = !(getenv("OPUS_NG_WILLNEED") && !atoi(getenv("OPUS_NG_WILLNEED")));
  if (willneed) {
    const uintptr_t pg = uintptr_t(getpagesize());
    for (const uint64_t row : rowid) {
      const Shard& S = shards[row / rows_per_shard];
      const uint64_t o = row % rows_per_shard;
      const uint8_t* ps[3] = {S.w + o * words * 4, S.s + o * groups * 2, S.b + o * groups * 2};
      for (const uint8_t* a : ps) madvise((void*)(uintptr_t(a) & ~(pg - 1)), pg, MADV_WILLNEED);
    }
  }
  // OPUS_NG_SERIAL=n: fetch serially when there are at most n rows (the pool's wake-ups can cost
  // more than a few page-cache hits)
  static const size_t serial_max = getenv("OPUS_NG_SERIAL") ? size_t(atoll(getenv("OPUS_NG_SERIAL"))) : 0;
  auto fetch = ^(size_t i) {
    const uint64_t row = rowid[i];
    const Shard& S = shards[row / rows_per_shard];
    const uint64_t o = row % rows_per_shard;
    const uint32_t* w = (const uint32_t*)(S.w + o * words * 4);
    const uint16_t* sc = (const uint16_t*)(S.s + o * groups * 2);
    const uint16_t* bi = (const uint16_t*)(S.b + o * groups * 2);
    float* dst = out + (i / heads) * heads * dim + (i % heads) * dim;
    for (uint32_t k = 0; k < dim; k++) {
      const uint32_t qv = (w[k / 8] >> (4 * (k % 8))) & 15u;
      dst[k] = bf2f(f2bf(bf2f(sc[k / group]) * float(qv) + bf2f(bi[k / group])));
    }
  };
  if (rowid.size() <= serial_max * heads)
    for (size_t i = 0; i < rowid.size(); i++) fetch(i);
  else
    dispatch_apply(rowid.size(), dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), fetch);
}

void NgramTable::rows_frag(const std::vector<uint32_t>& hist, uint32_t M, uint16_t* frag) const {
  const uint32_t K = (ngram_size - 1) * heads_per * dim;
  std::vector<float> rows(size_t(M) * K);
  rows_f32(hist, M, rows.data());
  memset(frag, 0, size_t((M + 15) / 16 * 16) * K * 2);
  for (uint32_t m = 0; m < M; m++)
    for (uint32_t k = 0; k < K; k++) {
      const uint32_t kk = k & 31, mm = m & 15;
      const uint32_t l = ((kk >> 2) & 1) | ((mm & 3) << 1) | (((kk >> 3) & 1) << 3) | (((mm >> 2) & 1) << 4);
      const uint32_t i = (kk & 3) | ((mm >> 3) << 2) | ((kk >> 4) << 3);
      frag[((m >> 4) * (K >> 5) + (k >> 5)) * 512 + l * 16 + i] = f2bf(rows[size_t(m) * K + k]);
    }
}

void NgramTable::rows_frag_at(const std::vector<uint32_t>& hist, uint32_t M, uint32_t row0, uint16_t* frag) const {
  const uint32_t K = (ngram_size - 1) * heads_per * dim;
  std::vector<float> rows(size_t(M) * K);
  rows_f32(hist, M, rows.data());
  for (uint32_t r = 0; r < M; r++)
    for (uint32_t k = 0; k < K; k++) {
      const uint32_t m = row0 + r, kk = k & 31, mm = m & 15;
      const uint32_t l = ((kk >> 2) & 1) | ((mm & 3) << 1) | (((kk >> 3) & 1) << 3) | (((mm >> 2) & 1) << 4);
      const uint32_t i = (kk & 3) | ((mm >> 3) << 2) | ((kk >> 4) << 3);
      frag[((m >> 4) * (K >> 5) + (k >> 5)) * 512 + l * 16 + i] = f2bf(rows[size_t(r) * K + k]);
    }
}

}  // namespace opus
