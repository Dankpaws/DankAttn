// Shared OPUS/OPCX binary framing, request parser, and token-prefix helpers for DankAttn serve loops.
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace dankattn::proto {

constexpr uint32_t MAGIC_REQ = 0x5355504Fu;     // 'OPUS'
constexpr uint32_t MAGIC_CANCEL = 0x5843504Fu;  // 'OPCX'

constexpr uint32_t END = 0xFFFFFFFFu;
constexpr uint32_t ERR = 0xFFFFFFFEu;
constexpr uint32_t READY = 0xFFFFFFFDu;
constexpr uint32_t CAND_SEP = 0xFFFFFFFCu;

inline bool rd(void* p, size_t n, FILE* f = stdin) { return n == 0 || fread(p, 1, n, f) == n; }
inline bool wr(const void* p, size_t n, FILE* f = stdout) { return n == 0 || fwrite(p, 1, n, f) == n; }

inline void put(uint32_t rid, uint32_t word, bool flush_now = false) {
  uint32_t b[2] = {rid, word};
  fwrite(b, 4, 2, stdout);
  if (flush_now) fflush(stdout);
}

inline void send_err(uint32_t rid, const std::string& msg) {
  uint32_t b[3] = {rid, ERR, uint32_t(msg.size())};
  fwrite(b, 4, 3, stdout);
  if (!msg.empty()) fwrite(msg.data(), 1, msg.size(), stdout);
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

struct VisImg {
  uint32_t T = 0, h = 0, w = 0, tok_off = 0;
  uint64_t hash = 0;
  std::vector<uint16_t> pixels;  // bf16 [T * h * w * 768]
};

struct OpusReq {
  uint32_t rid = 0, max_new = 0, flags = 0;
  uint32_t grp0 = 0, grp1 = 0, grp2 = 0;  // flags & 2: (gid, gsize, gindex) or (fpr_lanes, fpr_quorum, fpr_opts)
  std::vector<std::vector<uint32_t>> fpr_suffixes;  // inkling2 mode == 2 per-lane section prompts
  std::vector<uint32_t> ids, stops, force;
  std::vector<VisImg> imgs;
  uint32_t budget = 0, think_end = 0;
  float temp = 0.0f, top_p = 1.0f, min_p = -1.0f;
  uint32_t top_k = 0;
  uint64_t seed = 0;
  std::atomic<bool> cancel{false};
  std::chrono::steady_clock::time_point arrived = std::chrono::steady_clock::now();
};

enum class PacketKind { Eof, Cancel, Request };

struct ReadOpts {
  bool keep_images = false;         // true for vision-capable engines (hy4); false drains stream cleanly
  bool parse_fpr_suffixes = false;  // true for inkling2 (reads per-lane suffixes when (grp2 & 0xF) == 2)
};

// Read the next OPUS or OPCX packet from `in`. On Cancel, sets `out_cancel_rid`.
// On Request, populates `out_req`. Returns Eof on clean EOF or framing error.
inline PacketKind read_packet(OpusReq& out_req, uint32_t& out_cancel_rid,
                              const ReadOpts& opts = {}, FILE* in = stdin) {
  uint32_t magic = 0;
  if (!rd(&magic, 4, in)) return PacketKind::Eof;
  if (magic == MAGIC_CANCEL) {
    if (!rd(&out_cancel_rid, 4, in)) return PacketKind::Eof;
    return PacketKind::Cancel;
  }
  if (magic != MAGIC_REQ) return PacketKind::Eof;

  uint32_t hdr[2], tail[3];
  if (!rd(hdr, 8, in)) return PacketKind::Eof;
  out_req.rid = hdr[0];
  out_req.ids.resize(hdr[1]);
  if (!rd(out_req.ids.data(), out_req.ids.size() * 4, in) || !rd(tail, 12, in)) return PacketKind::Eof;
  out_req.max_new = tail[0];
  out_req.flags = tail[1];
  out_req.stops.resize(tail[2]);
  if (!rd(out_req.stops.data(), out_req.stops.size() * 4, in)) return PacketKind::Eof;

  struct { float temp, top_p; uint32_t top_k; uint64_t seed; } __attribute__((packed)) sp;
  if (!rd(&sp, sizeof sp, in)) return PacketKind::Eof;
  out_req.temp = sp.temp;
  out_req.top_p = sp.top_p;
  out_req.top_k = sp.top_k;
  out_req.seed = sp.seed;

  if (out_req.flags & 2u) {
    uint32_t g3[3];
    if (!rd(g3, 12, in)) return PacketKind::Eof;
    out_req.grp0 = g3[0];
    out_req.grp1 = g3[1];
    out_req.grp2 = g3[2];
    if (opts.parse_fpr_suffixes && (out_req.grp2 & 0xFu) == 2u) {
      out_req.fpr_suffixes.resize(out_req.grp0);
      for (uint32_t k = 0; k < out_req.grp0; k++) {
        uint32_t slen = 0;
        if (!rd(&slen, 4, in)) return PacketKind::Eof;
        out_req.fpr_suffixes[k].resize(slen);
        if (!rd(out_req.fpr_suffixes[k].data(), size_t(slen) * 4u, in)) return PacketKind::Eof;
      }
    }
  }
  if (out_req.flags & 4u) {
    uint32_t t3[3];
    if (!rd(t3, 12, in)) return PacketKind::Eof;
    out_req.budget = t3[0];
    out_req.think_end = t3[1];
    out_req.force.resize(t3[2]);
    if (!rd(out_req.force.data(), out_req.force.size() * 4, in)) return PacketKind::Eof;
  }
  if (out_req.flags & 16u) {
    if (!rd(&out_req.min_p, 4, in)) return PacketKind::Eof;
  }
  if (out_req.flags & 32u) {
    uint32_t n_imgs = 0;
    if (!rd(&n_imgs, 4, in)) return PacketKind::Eof;
    if (opts.keep_images) out_req.imgs.resize(n_imgs);
    std::vector<uint16_t> scratch;
    for (uint32_t k = 0; k < n_imgs; k++) {
      uint32_t hdr4[4];
      uint64_t ih = 0;
      if (!rd(hdr4, 16, in) || !rd(&ih, 8, in)) return PacketKind::Eof;
      const size_t npix = size_t(hdr4[0]) * hdr4[1] * hdr4[2] * 768u;
      if (opts.keep_images) {
        auto& im = out_req.imgs[k];
        im.T = hdr4[0]; im.h = hdr4[1]; im.w = hdr4[2]; im.tok_off = hdr4[3]; im.hash = ih;
        im.pixels.resize(npix);
        if (!rd(im.pixels.data(), npix * 2u, in)) return PacketKind::Eof;
      } else if (npix) {
        scratch.resize(npix);
        if (!rd(scratch.data(), npix * 2u, in)) return PacketKind::Eof;
      }
    }
  }
  return PacketKind::Request;
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
