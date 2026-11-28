// Shared CPU masked-logit sampler and exact log-softmax logprobs helper for DankAttn engines.
// Used only when a request enables constrained decoding (flags & 64, OPMK) or logprobs (flags & 128);
// normal fast-path GPU sampling is completely untouched.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace dankattn::sample {

inline uint32_t hmix(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352du;
  x ^= x >> 15; x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

inline float gumbel_noise(uint64_t seed, uint32_t pos, uint32_t tok) {
  const uint32_t s0 = uint32_t(seed & 0xFFFFFFFFu), s1 = uint32_t(seed >> 32);
  const uint32_t h = hmix(hmix(hmix(s0 ^ 0x9E3779B9u) ^ s1) ^ hmix(pos * 0x85EBCA6Bu + 0x27d4eb2fu) ^ (tok * 0xC2B2AE35u));
  const float u = (float(h >> 8) + 0.5f) * (1.0f / 16777216.0f);
  return -std::log(-std::log(u));
}

// Sample one token from `logits[0..V-1]`, optionally restricting to `allowed` (non-empty) and
// optionally computing exact log-softmax logprob + top-K logprobs over the (masked) distribution.
inline uint32_t sample_and_logprobs(
    const float* logits, uint32_t V,
    const std::vector<uint32_t>* allowed,
    float temp, float top_p, uint32_t top_k, float min_p,
    uint64_t seed, uint32_t pos,
    bool want_lp, uint32_t n_top,
    float& out_lp, std::vector<std::pair<uint32_t, float>>& out_tops) {
  out_lp = 0.0f;
  out_tops.clear();
  if (V == 0) return 0;

  const bool use_mask = (allowed != nullptr && !allowed->empty());
  float max_l = -std::numeric_limits<float>::infinity();
  uint32_t best_id = 0;

  if (use_mask) {
    for (uint32_t id : *allowed) {
      if (id >= V) continue;
      const float v = logits[id];
      if (v > max_l || (v == max_l && id < best_id)) { max_l = v; best_id = id; }
    }
  } else {
    for (uint32_t id = 0; id < V; id++) {
      const float v = logits[id];
      if (v > max_l) { max_l = v; best_id = id; }
    }
  }

  uint32_t chosen = best_id;
  if (temp > 0.0f) {
    const float inv_t = 1.0f / temp;
    const float max_s = max_l * inv_t;
    const float min_floor = (min_p > 0.0f) ? (max_s + std::log(min_p)) : -std::numeric_limits<float>::infinity();
    std::vector<std::pair<float, uint32_t>> cands;
    if (use_mask) {
      cands.reserve(allowed->size());
      for (uint32_t id : *allowed) {
        if (id >= V) continue;
        const float ls = logits[id] * inv_t;
        if (ls >= min_floor) cands.emplace_back(ls, id);
      }
    } else {
      const float cut = std::max(min_floor, max_s - 18.0f);
      for (uint32_t id = 0; id < V; id++) {
        const float ls = logits[id] * inv_t;
        if (ls >= cut) cands.emplace_back(ls, id);
      }
    }
    if (!cands.empty()) {
      std::sort(cands.begin(), cands.end(), [](const auto& a, const auto& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
      });
      if (top_k > 0 && cands.size() > top_k) cands.resize(top_k);
      if (top_p > 0.0f && top_p < 1.0f && cands.size() > 1) {
        double sum = 0.0;
        for (const auto& c : cands) sum += std::exp(double(c.first - max_s));
        const double lim = double(top_p) * sum;
        double acc = 0.0;
        size_t keep = 0;
        for (; keep < cands.size(); keep++) {
          acc += std::exp(double(cands[keep].first - max_s));
          if (acc >= lim) { keep++; break; }
        }
        cands.resize(std::max<size_t>(1, keep));
      }
      float best_g = -std::numeric_limits<float>::infinity();
      for (const auto& [ls, id] : cands) {
        const float g = ls + gumbel_noise(seed, pos, id);
        if (g > best_g || (g == best_g && id < chosen)) { best_g = g; chosen = id; }
      }
    }
  }

  if (want_lp) {
    // 200k * exp(-24) < 8e-6 (below f32 log-softmax precision); skipping deeply negative tail logits
    // avoids ~95%+ of scalar exp() calls while preserving exact logprobs to < 1e-5.
    double z = 0.0;
    const uint32_t k = std::min<uint32_t>(n_top, use_mask ? uint32_t(allowed->size()) : V);
    constexpr uint32_t MAX_TOP = 24;
    const uint32_t k_cap = std::min(k, MAX_TOP);
    float top_v[MAX_TOP];
    uint32_t top_i[MAX_TOP];
    uint32_t n_cur = 0;
    float kth_v = -std::numeric_limits<float>::infinity();
    uint32_t kth_i = 0xFFFFFFFFu;

    auto push_top = [&](float v, uint32_t id) {
      if (n_cur == k_cap && (v < kth_v || (v == kth_v && id >= kth_i))) return;
      uint32_t pos_ins = n_cur < k_cap ? n_cur++ : (k_cap - 1u);
      while (pos_ins > 0u && (v > top_v[pos_ins - 1u] || (v == top_v[pos_ins - 1u] && id < top_i[pos_ins - 1u]))) {
        top_v[pos_ins] = top_v[pos_ins - 1u];
        top_i[pos_ins] = top_i[pos_ins - 1u];
        pos_ins--;
      }
      top_v[pos_ins] = v;
      top_i[pos_ins] = id;
      if (n_cur == k_cap) { kth_v = top_v[k_cap - 1u]; kth_i = top_i[k_cap - 1u]; }
    };

    if (use_mask) {
      for (uint32_t id : *allowed) {
        if (id >= V) continue;
        const float v = logits[id];
        const float d = v - max_l;
        if (d > -24.0f) z += std::exp(double(d));
        if (k_cap > 0u) push_top(v, id);
      }
    } else {
      for (uint32_t id = 0; id < V; id++) {
        const float v = logits[id];
        const float d = v - max_l;
        if (d > -24.0f) z += std::exp(double(d));
        if (k_cap > 0u) push_top(v, id);
      }
    }
    const double lse = double(max_l) + std::log(std::max(z, 1e-30));
    out_lp = float(double(logits[chosen]) - lse);
    if (k_cap > 0u) {
      out_tops.reserve(n_cur);
      for (uint32_t i = 0; i < n_cur; i++) out_tops.emplace_back(top_i[i], float(double(top_v[i]) - lse));
    }
  }
  return chosen;
}

}  // namespace dankattn::sample
