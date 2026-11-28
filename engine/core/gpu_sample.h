// Shared GPU OPMK bitmask + exact logprobs sampler for DankAttn engines.
// Uploads a compact uint32_t bitmask (~25 KB for 200k vocab) and runs `dank_sample_masked_lp`
// directly on the engine's GPU logits buffer, eliminating the 800 KB logits memcpy and
// 200k-element CPU scan.
#pragma once
#include "../mtl.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace dankattn {

struct GpuSampler {
  struct Args {
    uint32_t V;
    uint32_t use_mask;
    uint32_t want_lp;
    uint32_t n_top;
    float temp;
    float top_p;
    uint32_t top_k;
    float min_p;
    uint32_t s0, s1;
    uint32_t pos;
  };
  struct Out {
    uint32_t chosen;
    float chosen_lp;
    uint32_t n_top;
    uint32_t _pad;
    uint32_t top_ids[24];
    float top_lps[24];
  };

  id<MTLComputePipelineState> pso = nil;
  id<MTLBuffer> mask_buf = nil;
  id<MTLBuffer> out_buf = nil;
  uint32_t cap_words = 0;

  void ensure_init(mtl::Ctx& C, uint32_t vocab) {
    if (!pso) {
      const std::string src = mtl::slurp(mtl::tree_root() + "/engine/kernels/sampling.metal");
      pso = C.pipe(src, "dank_sample_masked_lp", {{"DANK_SAMPLING_STANDALONE", "1"}});
    }
    const uint32_t words = (vocab + 31u) / 32u;
    if (!mask_buf || words > cap_words) {
      cap_words = std::max(words, 64u);
      mask_buf = C.buf(size_t(cap_words) * 4u);
    }
    if (!out_buf) out_buf = C.buf(sizeof(Out));
  }

  uint32_t sample(
      mtl::Ctx& C, id<MTLBuffer> logits_buf, size_t logits_byte_off, uint32_t vocab,
      const std::vector<uint32_t>* allowed,
      float temp, float top_p, uint32_t top_k, float min_p,
      uint64_t seed, uint32_t pos,
      bool want_lp, uint32_t n_top,
      float& out_lp, std::vector<std::pair<uint32_t, float>>& out_tops) {
    out_lp = 0.0f;
    out_tops.clear();
    if (vocab == 0 || !logits_buf) return 0;
    ensure_init(C, vocab);

    const bool use_mask = (allowed != nullptr && !allowed->empty());
    if (use_mask) {
      const uint32_t words = (vocab + 31u) / 32u;
      uint32_t* mp = (uint32_t*)mask_buf.contents;
      memset(mp, 0, size_t(words) * 4u);
      for (uint32_t id : *allowed) {
        if (id < vocab) mp[id >> 5] |= (1u << (id & 31u));
      }
    }

    Args a{
        vocab,
        use_mask ? 1u : 0u,
        want_lp ? 1u : 0u,
        want_lp ? std::min<uint32_t>(n_top, 20u) : 0u,
        temp > 0.0f ? temp : 0.0f,
        (top_p > 0.0f && top_p <= 1.0f) ? top_p : 1.0f,
        top_k,
        min_p > 0.0f ? min_p : 0.0f,
        uint32_t(seed & 0xFFFFFFFFu),
        uint32_t(seed >> 32),
        pos,
    };

    id<MTLCommandBuffer> cb = [C.q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:pso];
    [e setBuffer:logits_buf offset:logits_byte_off atIndex:0];
    [e setBuffer:mask_buf offset:0 atIndex:1];
    [e setBuffer:out_buf offset:0 atIndex:2];
    [e setBytes:&a length:sizeof(a) atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) mtl::die("GpuSampler error", cb.error);

    const Out* res = (const Out*)out_buf.contents;
    out_lp = res->chosen_lp;
    if (want_lp && res->n_top > 0u) {
      out_tops.reserve(res->n_top);
      for (uint32_t i = 0; i < res->n_top; i++) {
        out_tops.emplace_back(res->top_ids[i], res->top_lps[i]);
      }
    }
    return res->chosen;
  }
};

}  // namespace dankattn
