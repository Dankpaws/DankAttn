// Shared serving inbox, OPMK mask queue, cancellation, adaptive prefill chunking,
// masked/logprobs sampling, and request completion helpers for all DankAttn engines.
#pragma once
#include "gpu_sample.h"
#include "proto.h"
#include "sample.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace dankattn::sched {

using Req = proto::OpusReq;
using ReqP = std::shared_ptr<Req>;

// Thread-safe OPUS/OPCX/OPMK packet reader, request inbox, cancel tracker, and mask queue.
struct Inbox {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<ReqP> queue;
  std::map<uint32_t, ReqP> live;
  std::map<uint32_t, std::deque<std::vector<uint32_t>>> mask_q;
  bool closed = false;
  std::thread reader_th;

  void start_reader(proto::ReadOpts opts = {}, FILE* in = stdin) {
    reader_th = std::thread([this, opts, in]() {
      for (;;) {
        auto q = std::make_shared<Req>();
        uint32_t ctrl_rid = 0;
        std::vector<uint32_t> mask_ids;
        const auto kind = proto::read_packet(*q, ctrl_rid, opts, in, &mask_ids);
        if (kind == proto::PacketKind::Eof) break;
        if (kind == proto::PacketKind::Cancel) {
          std::lock_guard<std::mutex> g(mu);
          if (auto it = live.find(ctrl_rid); it != live.end()) it->second->cancel = true;
          cv.notify_all();
          continue;
        }
        if (kind == proto::PacketKind::Mask) {
          std::lock_guard<std::mutex> g(mu);
          mask_q[ctrl_rid].push_back(std::move(mask_ids));
          cv.notify_all();
          continue;
        }
        std::lock_guard<std::mutex> g(mu);
        live[q->rid] = q;
        queue.push_back(std::move(q));
        cv.notify_all();
      }
      {
        std::lock_guard<std::mutex> g(mu);
        closed = true;
      }
      cv.notify_all();
    });
  }

  void join() {
    if (reader_th.joinable()) reader_th.join();
  }

  void notify_all() { cv.notify_all(); }

  void forget(uint32_t rid) {
    std::lock_guard<std::mutex> g(mu);
    live.erase(rid);
    mask_q.erase(rid);
  }

  void push_front(ReqP q) {
    std::lock_guard<std::mutex> g(mu);
    queue.push_front(std::move(q));
  }

  // Block until an OPMK allowed-token list arrives for `rid`, or until cancelled / EOF.
  bool pop_mask(uint32_t rid, const std::atomic<bool>& cancel, std::vector<uint32_t>& out) {
    std::unique_lock<std::mutex> g(mu);
    cv.wait(g, [&] {
      if (closed || cancel.load()) return true;
      auto it = mask_q.find(rid);
      return it != mask_q.end() && !it->second.empty();
    });
    auto it = mask_q.find(rid);
    if (it == mask_q.end() || it->second.empty()) return false;
    out = std::move(it->second.front());
    it->second.pop_front();
    return true;
  }

  // Wait while idle until either EOF or a queued request is ready (and `extra_ready()` is true).
  // Returns false when the stream is closed and the inbox is empty.
  template <typename Pred>
  bool wait_for_work(Pred&& extra_ready) {
    std::unique_lock<std::mutex> g(mu);
    cv.wait(g, [&] { return closed || (!queue.empty() && extra_ready()); });
    return !(queue.empty() && closed);
  }

  bool wait_for_work() {
    return wait_for_work([] { return true; });
  }
};

inline bool is_stop_token(uint32_t eos, const Req& q, uint32_t tok) {
  return tok == eos || std::find(q.stops.begin(), q.stops.end(), tok) != q.stops.end();
}

// Adaptive prefill chunk size: full `max_pf` when no lanes are decoding; shrinks to `dec_chunk`
// (or half `dec_chunk` when 2+ lanes decode) to bound inter-token latency on active streams.
inline uint32_t adaptive_pf_chunk(uint32_t max_pf, uint32_t dec_chunk, uint32_t min_chunk,
                                  uint32_t n_decoding, uint32_t align = 1u) {
  if (n_decoding == 0u) return max_pf;
  const uint32_t cap = (n_decoding >= 2u) ? std::max<uint32_t>(64u, dec_chunk / 2u) : dec_chunk;
  const uint32_t aligned = (align > 1u) ? ((cap / align) * align) : cap;
  return std::min(max_pf, std::max(min_chunk, aligned));
}

// CPU masked-logit sampling (OPMK, flags & 64) and/or exact logprobs (flags & 128).
// Returns false if constrained decoding was cancelled/closed before a mask arrived.
inline bool sample_masked_or_logprobs(
    Inbox& inbox, const Req& req, const float* logits, uint32_t vocab,
    float default_min_p, uint32_t pos, uint32_t& out_tok, float logit_softcap = 0.0f) {
  const bool constrained = (req.flags & 64u) != 0u;
  const bool want_lp = (req.flags & 128u) != 0u;
  std::vector<uint32_t> allowed;
  if (constrained && !inbox.pop_mask(req.rid, req.cancel, allowed)) return false;

  const float* lg = logits;
  std::vector<float> capped;
  if (logit_softcap > 0.0f) {
    capped.resize(vocab);
    const float inv_sc = 1.0f / logit_softcap;
    for (uint32_t i = 0; i < vocab; i++) capped[i] = logit_softcap * std::tanh(logits[i] * inv_sc);
    lg = capped.data();
  }

  float lp_val = 0.0f;
  std::vector<std::pair<uint32_t, float>> tops;
  out_tok = sample::sample_and_logprobs(
      lg, vocab, constrained ? &allowed : nullptr,
      req.temp > 0.0f ? req.temp : 0.0f,
      (req.top_p > 0.0f && req.top_p <= 1.0f) ? req.top_p : 1.0f,
      req.top_k, req.min_p >= 0.0f ? req.min_p : default_min_p,
      req.seed, pos, want_lp, req.top_logprobs, lp_val, tops);
  if (want_lp) proto::send_logprob(req.rid, out_tok, lp_val, tops);
  return true;
}

// Zero-copy UMA fast-path masked-logit sampling (OPMK, flags & 64) and/or exact logprobs (flags & 128).
// Reads directly from the engine's shared GPU logits buffer (zero 800 KB memcpy, zero extra prefill row).
// Defaults to zero-copy UMA CPU scan (~5 us sparse OPMK, ~0.3-0.6 ms full V=200k logprobs, avoiding the
// ~0.2-0.3 ms separate Metal command-buffer round-trip); set DANKATTN_GPU_SAMPLE=1 to run the GPU kernel.
inline bool sample_masked_or_logprobs_gpu(
    Inbox& inbox, const Req& req, GpuSampler& gs, mtl::Ctx& C,
    id<MTLBuffer> logits_buf, size_t logits_byte_off, uint32_t vocab,
    float default_min_p, uint32_t pos, uint32_t& out_tok) {
  const bool constrained = (req.flags & 64u) != 0u;
  const bool want_lp = (req.flags & 128u) != 0u;
  std::vector<uint32_t> allowed;
  if (constrained && !inbox.pop_mask(req.rid, req.cancel, allowed)) return false;

  static const bool use_gpu = getenv("DANKATTN_GPU_SAMPLE") && atoi(getenv("DANKATTN_GPU_SAMPLE")) != 0;
  float lp_val = 0.0f;
  std::vector<std::pair<uint32_t, float>> tops;
  const float eff_temp = req.temp > 0.0f ? req.temp : 0.0f;
  const float eff_top_p = (req.top_p > 0.0f && req.top_p <= 1.0f) ? req.top_p : 1.0f;
  const float eff_min_p = req.min_p >= 0.0f ? req.min_p : default_min_p;
  if (use_gpu) {
    out_tok = gs.sample(
        C, logits_buf, logits_byte_off, vocab,
        constrained ? &allowed : nullptr,
        eff_temp, eff_top_p, req.top_k, eff_min_p,
        req.seed, pos, want_lp, req.top_logprobs, lp_val, tops);
  } else {
    const float* lg = (const float*)((const uint8_t*)logits_buf.contents + logits_byte_off);
    out_tok = sample::sample_and_logprobs(
        lg, vocab, constrained ? &allowed : nullptr,
        eff_temp, eff_top_p, req.top_k, eff_min_p,
        req.seed, pos, want_lp, req.top_logprobs, lp_val, tops);
  }
  if (want_lp) proto::send_logprob(req.rid, out_tok, lp_val, tops);
  return true;
}

inline void finish_request(
    Inbox& inbox, const Req& req, uint32_t reused, uint32_t n_gen,
    uint32_t rounds, uint32_t accepted, bool length, double prefill_ms,
    std::chrono::steady_clock::time_point t0, const char* tag,
    const char* slot_kind, int slot_id, const char* extra_log = "") {
  const float wall = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count() * 1e3f;
  proto::send_end(req.rid, reused, uint32_t(req.ids.size()), n_gen, rounds, accepted,
                  length ? 1u : 0u, float(prefill_ms), wall);
  if (tag && *tag) {
    fprintf(stderr, "%s: rid %u %s %d prompt %zu (reused %u, prefill %.0f ms) gen %u in %.0f ms%s\n",
            tag, req.rid, slot_kind ? slot_kind : "slot", slot_id, req.ids.size(),
            reused, prefill_ms, n_gen, wall, extra_log ? extra_log : "");
  }
  inbox.forget(req.rid);
}

}  // namespace dankattn::sched
