// Shared CLI helpers (binary token-ID loading, FNV-1a logit hashing, CSV parsing) for DankAttn binaries.
#pragma once
#include "../mtl.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace dankattn::cli {

inline std::vector<uint32_t> read_ids(const std::string& path) {
  std::string s = mtl::slurp(path);
  std::vector<uint32_t> v(s.size() / 4);
  if (!v.empty()) memcpy(v.data(), s.data(), v.size() * 4);
  return v;
}

inline uint64_t fnv1a_f32(const float* p, size_t n, uint64_t h = 1469598103934665603ull) {
  const auto* b = reinterpret_cast<const unsigned char*>(p);
  const size_t nb = n * 4;
  for (size_t i = 0; i < nb; i++) {
    h ^= b[i];
    h *= 1099511628211ull;
  }
  return h;
}

inline std::vector<uint32_t> parse_u32_csv(const std::string& csv) {
  std::vector<uint32_t> out;
  for (size_t s = 0; s < csv.size();) {
    size_t e = csv.find(',', s);
    if (e == std::string::npos) e = csv.size();
    if (e > s) out.push_back(uint32_t(atoi(csv.substr(s, e - s).c_str())));
    s = e + 1;
  }
  return out;
}

inline void print_topk(const float* logits, uint32_t V, uint32_t k = 5) {
  const uint32_t kk = std::min(k, V);
  std::vector<std::pair<float, uint32_t>> top;
  top.reserve(V);
  for (uint32_t i = 0; i < V; i++) top.push_back({logits[i], i});
  std::partial_sort(top.begin(), top.begin() + kk, top.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
  for (uint32_t i = 0; i < kk; i++) {
    printf("  top%u: id=%u logit=%.3f\n", i + 1, top[i].second, top[i].first);
  }
}

}  // namespace dankattn::cli
