// Flat zero-allocation 3-gram context-continuation index shared across DankAttn models.
// Replaces unordered_map<uint64_t, vector<uint32_t>> with a fixed open-bucket head[] table and
// per-position prev[] intrusive chain: O(1) insert, O(dropped) exact LIFO truncate, zero per-node
// heap allocations, and most-recent-first candidate iteration.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace opus_lookup {

struct Lookup {
  static constexpr uint32_t N = 3, CAND = 48, MAXLEN = 64, NB = 4;
  static constexpr uint32_t NONE = 0xFFFFFFFFu;
  static constexpr uint32_t HBITS = 19, HSIZE = 1u << HBITS, HMASK = HSIZE - 1u;  // 512k buckets (2 MB)

  std::vector<uint32_t> head;   // lazily allocated to HSIZE on first sync
  std::vector<uint32_t> prev;   // prev[p] = previous continuation position in same hash bucket
  std::vector<uint32_t> tok;    // tokens currently indexed
  double rate[NB] = {0.30, 0.50, 0.70, 0.85};  // EMA acceptance by match-length bucket

  static uint64_t key(const uint32_t* t) {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = 0; i < N; i++) { h ^= t[i]; h *= 1099511628211ull; }
    return h;
  }
  static uint32_t bucket_of(const uint32_t* t) {
    const uint64_t h = key(t);
    return uint32_t((h ^ (h >> 32)) & HMASK);
  }
  static uint32_t bucket(uint32_t len) { return len <= 4 ? 0 : len <= 8 ? 1 : len <= 16 ? 2 : 3; }

  void clear() {
    if (!head.empty()) std::fill(head.begin(), head.end(), NONE);
    prev.clear();
    tok.clear();
  }

  void truncate(size_t c) {
    if (c >= tok.size()) return;
    if (!head.empty()) {
      for (size_t p = tok.size(); p-- > std::max<size_t>(c, N);) {
        const uint32_t b = bucket_of(&tok[p - N]);
        if (head[b] == uint32_t(p)) head[b] = prev[p];
      }
    }
    tok.resize(c);
    prev.resize(c);
  }

  void sync(const std::vector<uint32_t>& hist, bool check) {
    size_t c = std::min(tok.size(), hist.size());
    if (check) {
      size_t k = 0;
      while (k < c && tok[k] == hist[k]) k++;
      c = k;
    }
    if (c < tok.size()) truncate(c);
    if (hist.size() > tok.size()) {
      if (head.empty()) head.assign(HSIZE, NONE);
      const size_t old_sz = tok.size();
      tok.insert(tok.end(), hist.begin() + long(old_sz), hist.end());
      prev.resize(hist.size(), NONE);
      for (size_t p = std::max<size_t>(old_sz, N); p < hist.size(); p++) {
        const uint32_t b = bucket_of(&hist[p - N]);
        prev[p] = head[b];
        head[b] = uint32_t(p);
      }
    }
  }

  uint32_t find(const std::vector<uint32_t>& hist, const std::vector<uint32_t>& tail, uint32_t* len) const {
    *len = 0;
    if (tail.size() < N || head.empty()) return NONE;
    const uint32_t b = bucket_of(&tail[tail.size() - N]);
    uint32_t best = NONE, bl = 0, seen = 0;
    for (uint32_t p = head[b]; p != NONE && seen < CAND; p = prev[p], seen++) {
      if (p >= hist.size()) continue;
      uint32_t m = 0;
      while (m < MAXLEN && m < p && m < tail.size() && hist[p - 1 - m] == tail[tail.size() - 1 - m]) m++;
      if (m > bl) { bl = m; best = p; }
    }
    if (bl < N) return NONE;
    *len = bl;
    return best;
  }
};

}  // namespace opus_lookup
