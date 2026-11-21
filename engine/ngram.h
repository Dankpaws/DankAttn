// PLE n-gram embedding rows, gathered on the CPU from an SSD-resident table.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace opus {

struct NgramTable {
  struct Shard { const uint8_t *w, *s, *b; uint64_t rows; };
  std::vector<Shard> shards;
  std::vector<int64_t> head_size, head_off, mult;
  uint32_t ngram_size = 3, heads_per = 8, dim = 160, group = 32, eos = 0, H = 0;
  uint64_t rows_per_shard = 0;

  NgramTable(const std::string& json_path, uint32_t hidden);
  // Embedding rows for the last M tokens of `hist`, written as bf16 in the matmul
  // left-operand fragment layout (K = heads * dim).
  void rows_frag(const std::vector<uint32_t>& hist, uint32_t M, uint16_t* frag) const;
  // Same rows written at rows row0.. of a frag the caller has zeroed (row0 + M <= 16).
  void rows_frag_at(const std::vector<uint32_t>& hist, uint32_t M, uint32_t row0, uint16_t* frag) const;
  // Plain [M][heads * dim] f32 rows (tests).
  void rows_f32(const std::vector<uint32_t>& hist, uint32_t M, float* out) const;
};

}  // namespace opus
