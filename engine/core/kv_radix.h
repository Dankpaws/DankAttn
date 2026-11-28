// Shared Block-Table Paged KV Allocator + Radix Prefix Tree for DankAttn engines.
//
// Manages a fixed pool of physical KV blocks (default 128 tokens/block, matching Hy4's cap % 128
// and indexer 4-token group alignment). Multiple concurrent lanes sharing a common prompt prefix
// reference the exact same physical blocks (zero-copy, refcounted). Unpinned blocks (refcnt == 0)
// remain cached in the radix tree for instant reuse by future requests and are evicted LRU-first
// from leaf nodes when free blocks are needed.
#pragma once
#include "proto.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace dankattn {

class KvRadixPool {
 public:
  static constexpr uint32_t BLOCK_SIZE = 128;
  static constexpr uint32_t BLOCK_SHIFT = 7;
  static constexpr uint32_t BLOCK_MASK = BLOCK_SIZE - 1u;
  static constexpr uint32_t INVALID_BLOCK = 0xFFFFFFFFu;

  struct Stats {
    uint32_t total_blocks = 0;
    uint32_t free_blocks = 0;
    uint32_t pinned_blocks = 0;
    uint32_t cached_blocks = 0;
    uint64_t prefix_hits = 0;
    uint64_t evictions = 0;
  };

  struct MatchResult {
    uint32_t matched_tokens = 0;
    std::vector<uint32_t> blocks;  // pinned (refcnt incremented) physical block IDs
  };

 private:
  struct Node {
    uint64_t hash = 0;
    std::vector<uint32_t> toks;  // exact BLOCK_SIZE tokens
    uint32_t phys_block = INVALID_BLOCK;
    uint64_t last_used = 0;
    Node* parent = nullptr;
    // Keyed by hash; bucket handles theoretical hash collisions via exact token compare.
    std::unordered_map<uint64_t, std::vector<std::unique_ptr<Node>>> children;
  };

  uint32_t num_blocks_ = 0;
  uint64_t clock_ = 0;
  uint64_t hits_ = 0;
  uint64_t evictions_ = 0;
  std::vector<uint32_t> refcnt_;
  std::vector<uint32_t> free_list_;
  std::vector<Node*> block_owner_;  // physical block -> radix node (nullptr if unindexed)
  std::unique_ptr<Node> root_;

  static uint64_t hash_block(const uint32_t* toks) {
    return proto::fnv1a_tokens(toks, BLOCK_SIZE);
  }

  Node* find_child(Node* parent, uint64_t h, const uint32_t* toks) const {
    auto it = parent->children.find(h);
    if (it == parent->children.end()) return nullptr;
    for (const auto& up : it->second) {
      if (up->toks.size() == BLOCK_SIZE && std::equal(up->toks.begin(), up->toks.end(), toks)) {
        return up.get();
      }
    }
    return nullptr;
  }

  void collect_evictable_leaves(Node* n, std::vector<Node*>& leaves) const {
    if (n->children.empty()) {
      if (n != root_.get() && n->phys_block != INVALID_BLOCK && refcnt_[n->phys_block] == 0) {
        leaves.push_back(n);
      }
      return;
    }
    for (const auto& [h, vec] : n->children) {
      for (const auto& ch : vec) collect_evictable_leaves(ch.get(), leaves);
    }
  }

  bool evict_one_lru() {
    std::vector<Node*> leaves;
    collect_evictable_leaves(root_.get(), leaves);
    if (leaves.empty()) return false;
    Node* victim = *std::min_element(leaves.begin(), leaves.end(),
                                     [](const Node* a, const Node* b) { return a->last_used < b->last_used; });
    const uint32_t blk = victim->phys_block;
    block_owner_[blk] = nullptr;
    free_list_.push_back(blk);
    evictions_++;

    Node* p = victim->parent;
    const uint64_t vh = victim->hash;
    auto& vec = p->children[vh];
    vec.erase(std::remove_if(vec.begin(), vec.end(),
                             [victim](const std::unique_ptr<Node>& up) { return up.get() == victim; }),
              vec.end());
    if (vec.empty()) p->children.erase(vh);
    return true;
  }

 public:
  explicit KvRadixPool(uint32_t num_blocks = 0) { init(num_blocks); }

  void init(uint32_t num_blocks) {
    num_blocks_ = num_blocks;
    clock_ = 0;
    hits_ = 0;
    evictions_ = 0;
    refcnt_.assign(num_blocks, 0);
    block_owner_.assign(num_blocks, nullptr);
    free_list_.resize(num_blocks);
    // Stack order: pop_back() yields block 0, 1, 2, ... sequentially on a fresh pool.
    for (uint32_t i = 0; i < num_blocks; i++) free_list_[i] = num_blocks - 1u - i;
    root_ = std::make_unique<Node>();
  }

  uint32_t capacity_blocks() const { return num_blocks_; }
  uint32_t capacity_tokens() const { return num_blocks_ * BLOCK_SIZE; }

  // Find longest full-block prefix match in the radix tree and pin all matched blocks.
  // Stops strictly before `tokens.size()` (never matches the final token so at least 1 token is fed).
  MatchResult match_prefix(const std::vector<uint32_t>& tokens) {
    MatchResult r;
    if (tokens.size() <= BLOCK_SIZE) return r;
    const uint32_t max_full = uint32_t((tokens.size() - 1u) / BLOCK_SIZE);
    Node* cur = root_.get();
    for (uint32_t b = 0; b < max_full; b++) {
      const uint32_t* ptr = tokens.data() + size_t(b) * BLOCK_SIZE;
      Node* nxt = find_child(cur, hash_block(ptr), ptr);
      if (!nxt || nxt->phys_block == INVALID_BLOCK) break;
      nxt->last_used = ++clock_;
      refcnt_[nxt->phys_block]++;
      r.blocks.push_back(nxt->phys_block);
      r.matched_tokens += BLOCK_SIZE;
      hits_++;
      cur = nxt;
    }
    return r;
  }

  // Allocate one physical block (from free list, or by evicting the LRU unpinned radix leaf).
  // Returned block has refcnt == 1 (pinned). Returns INVALID_BLOCK if all blocks are pinned.
  uint32_t alloc_block() {
    if (free_list_.empty() && !evict_one_lru()) return INVALID_BLOCK;
    const uint32_t blk = free_list_.back();
    free_list_.pop_back();
    refcnt_[blk] = 1;
    block_owner_[blk] = nullptr;
    return blk;
  }

  void pin_block(uint32_t blk) {
    if (blk < num_blocks_) {
      refcnt_[blk]++;
      if (block_owner_[blk]) block_owner_[blk]->last_used = ++clock_;
    }
  }

  void release_block(uint32_t blk) {
    if (blk >= num_blocks_ || refcnt_[blk] == 0) return;
    refcnt_[blk]--;
    if (refcnt_[blk] == 0) {
      if (block_owner_[blk]) {
        // Keep in radix tree as an unpinned cached leaf for future reuse.
        block_owner_[blk]->last_used = ++clock_;
      } else {
        free_list_.push_back(blk);
      }
    }
  }

  void release_blocks(const std::vector<uint32_t>& blocks) {
    for (uint32_t blk : blocks) release_block(blk);
  }

  // Register full 128-token blocks of `tokens` into the radix tree.
  // If a block is already indexed under the same prefix, keeps the existing physical block
  // (updating its LRU clock) and leaves `blocks[b]` unindexed so it frees on release.
  void insert_blocks(const std::vector<uint32_t>& tokens, const std::vector<uint32_t>& blocks) {
    const uint32_t n_full = std::min<uint32_t>(uint32_t(tokens.size() / BLOCK_SIZE), uint32_t(blocks.size()));
    Node* cur = root_.get();
    for (uint32_t b = 0; b < n_full; b++) {
      const uint32_t* ptr = tokens.data() + size_t(b) * BLOCK_SIZE;
      const uint64_t h = hash_block(ptr);
      Node* nxt = find_child(cur, h, ptr);
      if (nxt) {
        nxt->last_used = ++clock_;
        cur = nxt;
        continue;
      }
      const uint32_t blk = blocks[b];
      if (blk >= num_blocks_ || block_owner_[blk] != nullptr) break;
      auto nn = std::make_unique<Node>();
      nn->hash = h;
      nn->toks.assign(ptr, ptr + BLOCK_SIZE);
      nn->phys_block = blk;
      nn->last_used = ++clock_;
      nn->parent = cur;
      Node* raw = nn.get();
      cur->children[h].push_back(std::move(nn));
      block_owner_[blk] = raw;
      cur = raw;
    }
  }

  Stats stats() const {
    Stats s;
    s.total_blocks = num_blocks_;
    s.free_blocks = uint32_t(free_list_.size());
    s.prefix_hits = hits_;
    s.evictions = evictions_;
    for (uint32_t i = 0; i < num_blocks_; i++) {
      if (refcnt_[i] > 0) s.pinned_blocks++;
      else if (block_owner_[i] != nullptr) s.cached_blocks++;
    }
    return s;
  }
};

}  // namespace dankattn
