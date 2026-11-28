// Shared Metal command-buffer recorder and per-op GPU timestamp profiler for DankAttn engines.
#pragma once
#include "../mtl.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace dankattn {

inline id<MTLCounterSampleBuffer> new_sample_buffer(id<MTLDevice> dev) {
  MTLCounterSampleBufferDescriptor* d = [MTLCounterSampleBufferDescriptor new];
  for (id<MTLCounterSet> cs in dev.counterSets)
    if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) d.counterSet = cs;
  d.storageMode = MTLStorageModeShared;
  d.sampleCount = 4096;
  NSError* err = nil;
  id<MTLCounterSampleBuffer> sb = [dev newCounterSampleBufferWithDescriptor:d error:&err];
  if (!sb) mtl::die("counter sample buffer", err);
  return sb;
}

// Per-op compute encoder wrapper:
// - When `prof == false`: reuses one serial or concurrent encoder per command buffer, inserting
//   MTLBarrierScopeBuffers between ops when `conc && barrier`. Supports optional early commits
//   (`chunk0` / `chunk`) and optional per-op skip ablation (`skip_list` comma-delimited).
// - When `prof == true`: opens a fresh compute pass with start/end GPU timestamp samples per op.
struct Rec {
  static constexpr size_t PROF_OPS = 2048;

  id<MTLCommandBuffer> cb = nil;
  id<MTLComputeCommandEncoder> e = nil;
  id<MTLComputeCommandEncoder> mtl4_e = nil;
  bool mtl4_dispatched = false;
  bool prof = false;
  bool conc = true;
  std::vector<id<MTLCounterSampleBuffer>> pool;
  std::vector<id<MTLCounterSampleBuffer>>* ext_pool = nullptr;  // optional external sample-buffer pool
  std::vector<std::string> names;

  // Optional early-commit chunking (used by Flash-Next).
  uint32_t chunk0 = 0, chunk = 0, nops = 0;
  std::vector<id<MTLCommandBuffer>> sent;

  // Optional comma-delimited op skip list (e.g. ",moe_gu,moe_dn,") for timing ablations.
  const std::string* skip_csv = nullptr;

  void flush() {
    if (e) [e endEncoding];
    e = nil;
    [cb commit];
    sent.push_back(cb);
    cb = [cb.commandQueue commandBuffer];
  }

  id<MTLComputeCommandEncoder> op(const char* name, bool barrier = true) {
    if (skip_csv && !skip_csv->empty() && !prof &&
        skip_csv->find("," + std::string(name) + ",") != std::string::npos) {
      static id<MTLCommandBuffer> dcb = nil;
      static id<MTLComputeCommandEncoder> de = nil;
      if (!de) {
        dcb = [cb.commandQueue commandBuffer];
        de = [dcb computeCommandEncoder];
      }
      return de;
    }
    if (!prof) {
      if (mtl4_e) {
        if (mtl4_dispatched && barrier) [mtl4_e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        mtl4_dispatched = true;
        return mtl4_e;
      }
      if (chunk0 && e && (++nops == chunk0 || (chunk && nops > chunk0 && (nops - chunk0) % chunk == 0))) flush();
      if (!e) {
        e = conc ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] : [cb computeCommandEncoder];
      } else if (conc && barrier) {
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
      return e;
    }
    if (e) [e endEncoding];
    auto& p = ext_pool ? *ext_pool : pool;
    const size_t b = names.size() / PROF_OPS, i = names.size() % PROF_OPS;
    while (p.size() <= b) p.push_back(new_sample_buffer(cb.device));
    MTLComputePassDescriptor* d = [MTLComputePassDescriptor computePassDescriptor];
    d.sampleBufferAttachments[0].sampleBuffer = p[b];
    d.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2 * i;
    d.sampleBufferAttachments[0].endOfEncoderSampleIndex = 2 * i + 1;
    e = [cb computeCommandEncoderWithDescriptor:d];
    names.push_back(name);
    return e;
  }

  void resolve(std::map<std::string, double>& out) {
    auto& p = ext_pool ? *ext_pool : pool;
    for (size_t b = 0; b * PROF_OPS < names.size(); b++) {
      const size_t n = std::min<size_t>(PROF_OPS, names.size() - b * PROF_OPS);
      NSData* d = [p[b] resolveCounterRange:NSMakeRange(0, 2 * n)];
      const MTLCounterResultTimestamp* ts = (const MTLCounterResultTimestamp*)d.bytes;
      for (size_t i = 0; i < n; i++)
        out[names[b * PROF_OPS + i]] += (ts[2 * i + 1].timestamp - ts[2 * i].timestamp) / 1e6;
    }
  }
};

}  // namespace dankattn
