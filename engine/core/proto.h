// Shared OPUS/OPCX binary framing and token-prefix helpers for DankAttn serve loops.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace dankattn::proto {

constexpr uint32_t END = 0xFFFFFFFFu;
constexpr uint32_t ERR = 0xFFFFFFFEu;
constexpr uint32_t READY = 0xFFFFFFFDu;
constexpr uint32_t CAND_SEP = 0xFFFFFFFCu;

inline bool rd(void* p, size_t n) { return fread(p, 1, n, stdin) == n; }
inline bool wr(const void* p, size_t n) { return fwrite(p, 1, n, stdout) == n; }

inline void put(uint32_t rid, uint32_t word) {
  uint32_t b[2] = {rid, word};
  fwrite(b, 4, 2, stdout);
  fflush(stdout);
}

inline void send_err(uint32_t rid, const std::string& msg) {
  uint32_t b[3] = {rid, ERR, uint32_t(msg.size())};
  fwrite(b, 4, 3, stdout);
  fwrite(msg.data(), 1, msg.size(), stdout);
  fflush(stdout);
}

inline void send_end(uint32_t rid, uint32_t reused, uint32_t n_prompt, uint32_t n_gen,
                     uint32_t rounds, uint32_t accepted, uint32_t flags,
                     float prefill_ms, float wall_ms) {
  uint32_t h[8] = {rid, END, reused, n_prompt, n_gen, rounds, accepted, flags};
  float f[2] = {prefill_ms, wall_ms};
  fwrite(h, 4, 8, stdout);
  fwrite(f, 4, 2, stdout);
  fflush(stdout);
}

inline uint32_t common_prefix(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, size_t n) {
  uint32_t c = 0;
  const size_t lim = std::min({a.size(), b.size(), n});
  while (c < lim && a[c] == b[c]) c++;
  return c;
}

inline uint64_t fnv1a_tokens(const uint32_t* toks, size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; i++) {
    h ^= toks[i];
    h *= 1099511628211ull;
  }
  return h;
}

}  // namespace dankattn::proto
