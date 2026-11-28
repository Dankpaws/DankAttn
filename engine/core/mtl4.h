// True Metal 4 Command Encoding backend (MTL4CommandQueue + MTL4CommandBuffer ring +
// MTL4ArgumentTable with redundant-bind elision & pre-baked static tables + bump-allocated
// constant buffer + MTL4VisibilityOptionNone intra-encoder barriers + serialized early commits).
//
// Exposes an Objective-C proxy (`DankMTL4Encoder`) implementing the exact subset of
// MTLComputeCommandEncoder selectors used by DankAttn engines so models can switch between
// Metal 3 and Metal 4 via DANKATTN_MTL4=1 for bit-exact A/B benchmarking without duplicating
// dispatch code.
#pragma once
#include "../mtl.h"
#include "env.h"
#include <cstring>
#include <unordered_map>

@interface DankMTL4Encoder : NSObject {
 @public
  id<MTL4ComputeCommandEncoder> _enc;
  id<MTL4ArgumentTable> _at;
  id<MTL4ArgumentTable> _baked_at;       // non-nil when a pre-baked table is bound for the next dispatch
  id<MTLComputePipelineState> _bound_pso;
  uint64_t _bound_addr[32];
  uint32_t _bound_mask;
  uint8_t* _cbuf_ptr;
  uint64_t _cbuf_gpu;
  size_t _cbuf_off;
  size_t _cbuf_cap;
  bool _auto_barrier;  // true for serial-equivalent mode (barrier before every dispatch > 0)
  bool _dispatched;
  bool _barrier_done;  // true if an explicit barrier was already emitted since the last dispatch
  bool _at_dirty;
  MTL4VisibilityOptions _vis;
}
- (void)resetState;
- (void)setBakedTable:(id<MTL4ArgumentTable>)table;
- (void)setComputePipelineState:(id<MTLComputePipelineState>)state;
- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)setBytes:(const void*)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)setThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)memoryBarrierWithScope:(MTLBarrierScope)scope;
- (void)dispatchThreadgroups:(MTLSize)grid threadsPerThreadgroup:(MTLSize)tg;
- (void)dispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)tg;
- (void)dispatchThreadgroupsWithIndirectBuffer:(id<MTLBuffer>)buf
                          indirectBufferOffset:(NSUInteger)off
                         threadsPerThreadgroup:(MTLSize)tg;
- (void)endEncoding;
@end

#ifdef DANKATTN_MTL4_IMPL
@implementation DankMTL4Encoder
- (void)resetState {
  _baked_at = nil;
  _bound_pso = nil;
  _bound_mask = 0u;
  _dispatched = false;
  _barrier_done = false;
  _at_dirty = true;
}
- (void)setBakedTable:(id<MTL4ArgumentTable>)table {
  _baked_at = table;
  _at_dirty = true;
}
- (void)setComputePipelineState:(id<MTLComputePipelineState>)state {
  if (state == _bound_pso) return;
  _bound_pso = state;
  [_enc setComputePipelineState:state];
}
- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
  _baked_at = nil;
  const uint64_t addr = buffer ? (buffer.gpuAddress + offset) : 0ull;
  if (index < 32u) {
    const uint32_t bit = 1u << index;
    if ((_bound_mask & bit) && _bound_addr[index] == addr) return;
    _bound_mask |= bit;
    _bound_addr[index] = addr;
  }
  [_at setAddress:addr atIndex:index];
  _at_dirty = true;
}
- (void)setBytes:(const void*)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
  _baked_at = nil;
  const size_t aligned = (length + 15u) & ~size_t(15u);
  if (_cbuf_off + aligned > _cbuf_cap) mtl::die("MTL4 constant bump buffer overflow");
  memcpy(_cbuf_ptr + _cbuf_off, bytes, length);
  const uint64_t addr = _cbuf_gpu + _cbuf_off;
  _cbuf_off += aligned;
  if (index < 32u) {
    _bound_mask |= (1u << index);
    _bound_addr[index] = addr;
  }
  [_at setAddress:addr atIndex:index];
  _at_dirty = true;
}
- (void)setThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index {
  [_enc setThreadgroupMemoryLength:length atIndex:index];
}
- (void)memoryBarrierWithScope:(MTLBarrierScope)scope {
  (void)scope;
  if (!_dispatched || _barrier_done) return;
  [_enc barrierAfterEncoderStages:MTLStageDispatch
              beforeEncoderStages:MTLStageDispatch
                visibilityOptions:_vis];
  _barrier_done = true;
}
- (void)dispatchThreadgroups:(MTLSize)grid threadsPerThreadgroup:(MTLSize)tg {
  if (_auto_barrier && _dispatched && !_barrier_done) {
    [_enc barrierAfterEncoderStages:MTLStageDispatch
                beforeEncoderStages:MTLStageDispatch
                  visibilityOptions:_vis];
  }
  if (_baked_at) {
    [_enc setArgumentTable:_baked_at];
    _baked_at = nil;
    _at_dirty = true;
  } else if (_at_dirty) {
    [_enc setArgumentTable:_at];
    _at_dirty = false;
  }
  [_enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  _dispatched = true;
  _barrier_done = false;
}
- (void)dispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)tg {
  if (_auto_barrier && _dispatched && !_barrier_done) {
    [_enc barrierAfterEncoderStages:MTLStageDispatch
                beforeEncoderStages:MTLStageDispatch
                  visibilityOptions:_vis];
  }
  if (_baked_at) {
    [_enc setArgumentTable:_baked_at];
    _baked_at = nil;
    _at_dirty = true;
  } else if (_at_dirty) {
    [_enc setArgumentTable:_at];
    _at_dirty = false;
  }
  [_enc dispatchThreads:threads threadsPerThreadgroup:tg];
  _dispatched = true;
  _barrier_done = false;
}
- (void)dispatchThreadgroupsWithIndirectBuffer:(id<MTLBuffer>)buf
                          indirectBufferOffset:(NSUInteger)off
                         threadsPerThreadgroup:(MTLSize)tg {
  if (_auto_barrier && _dispatched && !_barrier_done) {
    [_enc barrierAfterEncoderStages:MTLStageDispatch
                beforeEncoderStages:MTLStageDispatch
                  visibilityOptions:_vis];
  }
  if (_baked_at) {
    [_enc setArgumentTable:_baked_at];
    _baked_at = nil;
    _at_dirty = true;
  } else if (_at_dirty) {
    [_enc setArgumentTable:_at];
    _at_dirty = false;
  }
  [_enc dispatchThreadgroupsWithIndirectBuffer:(buf.gpuAddress + off) threadsPerThreadgroup:tg];
  _dispatched = true;
  _barrier_done = false;
}
- (void)endEncoding {
  if (_enc) {
    [_enc endEncoding];
    _enc = nil;
  }
  [self resetState];
}
@end
#endif  // DANKATTN_MTL4_IMPL

namespace dankattn {

struct Mtl4Runtime {
  static constexpr uint32_t MAX_CBS = 16;

  bool enabled = false;
  id<MTLDevice> dev = nil;
  id<MTL4CommandQueue> q4 = nil;
  id<MTL4CommandAllocator> allocs[MAX_CBS] = {};
  id<MTL4CommandBuffer> cbs[MAX_CBS] = {};
  uint32_t cb_idx = 0;
  uint32_t max_used_cb = 0;
  id<MTL4ArgumentTable> at = nil;
  id<MTLSharedEvent> ev = nil;
  id<MTLBuffer> cbuf = nil;
  id<MTLResidencySet> own_rs = nil;
  id<MTLResidencySet> unwired_rs = nil;
  uint64_t ev_val = 0;
  DankMTL4Encoder* proxy = nil;
  std::unordered_map<uint64_t, id<MTL4ArgumentTable>> baked_tables;

  void init(id<MTLDevice> d, size_t cbuf_bytes = 4u << 20, uint32_t max_bufs = 32) {
    enabled = true;
    dev = d;
    q4 = [dev newMTL4CommandQueue];
    if (!q4) mtl::die("newMTL4CommandQueue failed");
    for (uint32_t i = 0; i < MAX_CBS; i++) {
      allocs[i] = [dev newCommandAllocator];
      cbs[i] = [dev newCommandBuffer];
    }
    ev = [dev newSharedEvent];
    NSError* err = nil;
    MTL4ArgumentTableDescriptor* atd = [MTL4ArgumentTableDescriptor new];
    atd.maxBufferBindCount = max_bufs;
    at = [dev newArgumentTableWithDescriptor:atd error:&err];
    if (!at) mtl::die("newArgumentTable failed", err);
    cbuf = [dev newBufferWithLength:cbuf_bytes options:MTLResourceStorageModeShared];
    if (!cbuf) mtl::die("MTL4 cbuf alloc failed");
    MTLResidencySetDescriptor* rsd = [MTLResidencySetDescriptor new];
    rsd.label = @"dankattn-mtl4-internal";
    own_rs = [dev newResidencySetWithDescriptor:rsd error:&err];
    if (!own_rs) mtl::die("MTL4 own_rs failed", err);
    [own_rs addAllocation:cbuf];
    [own_rs commit];
    [own_rs requestResidency];
    [q4 addResidencySet:own_rs];
    proxy = [DankMTL4Encoder new];
    proxy->_at = at;
    proxy->_cbuf_ptr = (uint8_t*)cbuf.contents;
    proxy->_cbuf_gpu = cbuf.gpuAddress;
    proxy->_cbuf_cap = cbuf_bytes;
    [proxy resetState];
  }

  // Bake a static argument table once (e.g., per-layer weight + fixed scratch bindings).
  id<MTL4ArgumentTable> bake_table(const uint64_t* addrs, uint32_t count) {
    if (!enabled || !dev) return nil;
    NSError* err = nil;
    MTL4ArgumentTableDescriptor* atd = [MTL4ArgumentTableDescriptor new];
    atd.maxBufferBindCount = count;
    id<MTL4ArgumentTable> tbl = [dev newArgumentTableWithDescriptor:atd error:&err];
    if (!tbl) mtl::die("bake_table newArgumentTable failed", err);
    for (uint32_t i = 0; i < count; i++) [tbl setAddress:addrs[i] atIndex:i];
    return tbl;
  }

  void add_residency_set(id<MTLResidencySet> rs) {
    if (enabled && q4 && rs) [q4 addResidencySet:rs];
  }

  void add_buffers(const std::vector<id<MTLBuffer>>& bufs) {
    if (!enabled || !own_rs || bufs.empty()) return;
    for (id<MTLBuffer> b : bufs) if (b) [own_rs addAllocation:b];
    [own_rs commit];
    [own_rs requestResidency];
  }

  // Register buffers with the MTL4 command queue without pinning them in wired RAM
  // (e.g. unwired drafter or vision weight files).
  void add_unwired_buffers(const std::vector<id<MTLBuffer>>& bufs) {
    if (!enabled || !q4 || bufs.empty()) return;
    if (!unwired_rs) {
      NSError* err = nil;
      MTLResidencySetDescriptor* rsd = [MTLResidencySetDescriptor new];
      rsd.label = @"dankattn-mtl4-unwired";
      unwired_rs = [dev newResidencySetWithDescriptor:rsd error:&err];
      if (!unwired_rs) mtl::die("MTL4 unwired_rs failed", err);
      [q4 addResidencySet:unwired_rs];
    }
    for (id<MTLBuffer> b : bufs) if (b) [unwired_rs addAllocation:b];
    [unwired_rs commit];
  }

  void add_heap(id<MTLHeap> h) {
    if (!enabled || !own_rs || !h) return;
    [own_rs addAllocation:h];
    [own_rs commit];
    [own_rs requestResidency];
  }

  id<MTLComputeCommandEncoder> begin(bool auto_barrier,
                                     MTL4VisibilityOptions vis = MTL4VisibilityOptionNone) {
    for (uint32_t i = 0; i <= max_used_cb && i < MAX_CBS; i++) [allocs[i] reset];
    cb_idx = 0;
    max_used_cb = 0;
    [cbs[0] beginCommandBufferWithAllocator:allocs[0]];
    proxy->_enc = [cbs[0] computeCommandEncoder];
    proxy->_cbuf_off = 0;
    proxy->_auto_barrier = auto_barrier;
    proxy->_vis = vis;
    [proxy resetState];
    return (id<MTLComputeCommandEncoder>)proxy;
  }

  // Early-commit the current MTL4CommandBuffer and open the next one in the ring, inserting
  // an explicit queue event barrier so GPU execution stays strictly in-order while CPU encoding
  // overlaps with GPU execution of earlier layers.
  id<MTLComputeCommandEncoder> flush() {
    if (!enabled || cb_idx + 1u >= MAX_CBS) return (id<MTLComputeCommandEncoder>)proxy;
    const bool ab = proxy->_auto_barrier;
    [proxy endEncoding];
    [cbs[cb_idx] endCommandBuffer];
    id<MTL4CommandBuffer> one[1] = {cbs[cb_idx]};
    [q4 commit:one count:1];
    const uint64_t mid = ++ev_val;
    [q4 signalEvent:ev value:mid];
    [q4 waitForEvent:ev value:mid];
    cb_idx++;
    if (cb_idx > max_used_cb) max_used_cb = cb_idx;
    [cbs[cb_idx] beginCommandBufferWithAllocator:allocs[cb_idx]];
    proxy->_enc = [cbs[cb_idx] computeCommandEncoder];
    proxy->_auto_barrier = ab;
    [proxy resetState];
    return (id<MTLComputeCommandEncoder>)proxy;
  }

  void copy_buffer(id<MTLBuffer> src, size_t soff, id<MTLBuffer> dst, size_t doff, size_t sz) {
    if (!proxy->_enc) return;
    [proxy->_enc barrierAfterEncoderStages:MTLStageDispatch
                       beforeEncoderStages:MTLStageBlit
                         visibilityOptions:(MTL4VisibilityOptionDevice | MTL4VisibilityOptionResourceAlias)];
    [proxy->_enc copyFromBuffer:src sourceOffset:soff toBuffer:dst destinationOffset:doff size:sz];
    [proxy->_enc barrierAfterEncoderStages:MTLStageBlit
                       beforeEncoderStages:MTLStageDispatch
                         visibilityOptions:(MTL4VisibilityOptionDevice | MTL4VisibilityOptionResourceAlias)];
  }

  void commit_and_wait() {
    [proxy endEncoding];
    [cbs[cb_idx] endCommandBuffer];
    id<MTL4CommandBuffer> one[1] = {cbs[cb_idx]};
    [q4 commit:one count:1];
    const uint64_t target = ++ev_val;
    [q4 signalEvent:ev value:target];
    while (ev.signaledValue < target) {
      [ev waitUntilSignaledValue:target timeoutMS:100];
    }
  }
};

}  // namespace dankattn
