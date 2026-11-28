// Unit test for dankattn::KvRadixPool (block-table allocator + radix prefix tree + LRU eviction).
// Build: clang++ -std=c++20 -O2 bench/kv_radix_test.mm -o build/kv_radix_test
#include "../engine/core/kv_radix.h"
#include <cassert>
#include <cstdio>
#include <vector>

static std::vector<uint32_t> make_seq(uint32_t seed, size_t n) {
  std::vector<uint32_t> v(n);
  for (size_t i = 0; i < n; i++) v[i] = seed * 100000u + uint32_t(i);
  return v;
}

int main() {
  using dankattn::KvRadixPool;
  constexpr uint32_t B = KvRadixPool::BLOCK_SIZE;

  // 1. Fresh pool: sequential block allocation, radix insert, and zero-copy multi-lane prefix sharing.
  KvRadixPool pool(8);
  auto p1 = make_seq(1, 3 * B + 20);  // 3 full blocks + 20 tail tokens
  auto m0 = pool.match_prefix(p1);
  assert(m0.matched_tokens == 0 && m0.blocks.empty());

  std::vector<uint32_t> b1;
  for (int i = 0; i < 4; i++) b1.push_back(pool.alloc_block());
  assert(b1[0] == 0 && b1[1] == 1 && b1[2] == 2 && b1[3] == 3);
  pool.insert_blocks(p1, b1);  // indexes first 3 full blocks; block 3 is partial (unindexed)

  // Second concurrent lane shares first 2 blocks of p1 + diverges in block 2
  auto p2 = p1;
  p2[2 * B + 5] = 999999u;
  auto m2 = pool.match_prefix(p2);
  assert(m2.matched_tokens == 2 * B);
  assert(m2.blocks.size() == 2 && m2.blocks[0] == 0 && m2.blocks[1] == 1);

  auto st = pool.stats();
  assert(st.pinned_blocks == 4 && st.free_blocks == 4 && st.prefix_hits == 2);

  // Release lane 1: blocks 0,1 stay pinned by lane 2; block 2 stays cached (unpinned); block 3 returns to free_list
  pool.release_blocks(b1);
  st = pool.stats();
  assert(st.pinned_blocks == 2 && st.cached_blocks == 1 && st.free_blocks == 5);

  // Release lane 2: blocks 0,1,2 all stay cached in radix tree
  pool.release_blocks(m2.blocks);
  st = pool.stats();
  assert(st.pinned_blocks == 0 && st.cached_blocks == 3 && st.free_blocks == 5);

  // 2. LRU eviction under pressure: allocate all 8 blocks -> must evict cached leaves (2, then 1, then 0)
  std::vector<uint32_t> all8;
  for (int i = 0; i < 8; i++) {
    uint32_t blk = pool.alloc_block();
    assert(blk != KvRadixPool::INVALID_BLOCK);
    all8.push_back(blk);
  }
  assert(pool.alloc_block() == KvRadixPool::INVALID_BLOCK);  // pool exhausted while all 8 pinned
  st = pool.stats();
  assert(st.pinned_blocks == 8 && st.evictions == 3);

  pool.release_blocks(all8);
  st = pool.stats();
  assert(st.free_blocks == 8 && st.pinned_blocks == 0 && st.cached_blocks == 0);

  printf("KV_RADIX_TEST_OK (total=%u hits=%llu evictions=%llu)\n",
         st.total_blocks, (unsigned long long)st.prefix_hits, (unsigned long long)st.evictions);
  return 0;
}
