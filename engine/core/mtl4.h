// True Metal 4 Command Encoding backend (MTL4CommandQueue + MTL4CommandBuffer + MTL4ArgumentTable
// + bump-allocated constant buffer + MTL4VisibilityOptionNone intra-encoder barriers).
//
// Exposes an Objective-C proxy (`DankMTL4Encoder`) implementing the exact subset of
// MTLComputeCommandEncoder selectors used by DankAttn engines so models can switch between
// Metal 3 and Metal 4 via DANKATTN_MTL4=1 for bit-exact A/B benchmarking without duplicating
// dispatch code.
#pragma once
#include "../mtl.h"
#include "env.h"
#include <cstring>

@interface DankMTL4Encoder : NSObject {
 @public
  id<MTL4ComputeCommandEncoder> _enc;
  id<MTL4ArgumentTable> _at;
  uint8_t* _cbuf_ptr;
  uint64_t _cbuf_gpu;
  size_t _cbuf_off;
  size_t _cbuf_cap;
  bool _auto_barrier;  // true for serial-equivalent mode (barrier before every dispatch > 0)
  bool _dispatched;
}
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
- (void)setComputePipelineState:(id<MTLComputePipelineState>)state {
  [_enc setComputePipelineState:state];
}
- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
  [_at setAddress:(buffer ? (buffer.gpuAddress + offset) : 0ull) atIndex:index];
}
- (void)setBytes:(const void*)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
  const size_t aligned = (length + 15u) & ~size_t(15u);
  if (_cbuf_off + aligned > _cbuf_cap) mtl::die("MTL4 constant bump buffer overflow");
  memcpy(_cbuf_ptr + _cbuf_off, bytes, length);
  [_at setAddress:(_cbuf_gpu + _cbuf_off) atIndex:index];
  _cbuf_off += aligned;
}
- (void)setThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index {
  [_enc setThreadgroupMemoryLength:length atIndex:index];
}
- (void)memoryBarrierWithScope:(MTLBarrierScope)scope {
  (void)scope;
  [_enc barrierAfterEncoderStages:MTLStageDispatch
              beforeEncoderStages:MTLStageDispatch
                visibilityOptions:MTL4VisibilityOptionNone];
}
- (void)dispatchThreadgroups:(MTLSize)grid threadsPerThreadgroup:(MTLSize)tg {
  if (_auto_barrier && _dispatched) {
    [_enc barrierAfterEncoderStages:MTLStageDispatch
                beforeEncoderStages:MTLStageDispatch
                  visibilityOptions:MTL4VisibilityOptionNone];
  }
  [_enc setArgumentTable:_at];
  [_enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  _dispatched = true;
}
- (void)dispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)tg {
  if (_auto_barrier && _dispatched) {
    [_enc barrierAfterEncoderStages:MTLStageDispatch
                beforeEncoderStages:MTLStageDispatch
                  visibilityOptions:MTL4VisibilityOptionNone];
  }
  [_enc setArgumentTable:_at];
  [_enc dispatchThreads:threads threadsPerThreadgroup:tg];
  _dispatched = true;
}
- (void)dispatchThreadgroupsWithIndirectBuffer:(id<MTLBuffer>)buf
                          indirectBufferOffset:(NSUInteger)off
                         threadsPerThreadgroup:(MTLSize)tg {
  if (_auto_barrier && _dispatched) {
    [_enc barrierAfterEncoderStages:MTLStageDispatch
                beforeEncoderStages:MTLStageDispatch
                  visibilityOptions:MTL4VisibilityOptionNone];
  }
  [_enc setArgumentTable:_at];
  [_enc dispatchThreadgroupsWithIndirectBuffer:(buf.gpuAddress + off) threadsPerThreadgroup:tg];
  _dispatched = true;
}
- (void)endEncoding {
  if (_enc) {
    [_enc endEncoding];
    _enc = nil;
  }
}
@end
#endif  // DANKATTN_MTL4_IMPL

namespace dankattn {

struct Mtl4Runtime {
  bool enabled = false;
  id<MTL4CommandQueue> q4 = nil;
  id<MTL4CommandAllocator> alloc = nil;
  id<MTL4CommandBuffer> cb4 = nil;
  id<MTL4ArgumentTable> at = nil;
  id<MTLSharedEvent> ev = nil;
  id<MTLBuffer> cbuf = nil;
  id<MTLResidencySet> own_rs = nil;
  uint64_t ev_val = 0;
  DankMTL4Encoder* proxy = nil;

  void init(id<MTLDevice> dev, size_t cbuf_bytes = 4u << 20, uint32_t max_bufs = 32) {
    enabled = true;
    q4 = [dev newMTL4CommandQueue];
    if (!q4) mtl::die("newMTL4CommandQueue failed");
    alloc = [dev newCommandAllocator];
    cb4 = [dev newCommandBuffer];
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

  void add_heap(id<MTLHeap> h) {
    if (!enabled || !own_rs || !h) return;
    [own_rs addAllocation:h];
    [own_rs commit];
    [own_rs requestResidency];
  }

  id<MTLComputeCommandEncoder> begin(bool auto_barrier) {
    [alloc reset];
    [cb4 beginCommandBufferWithAllocator:alloc];
    proxy->_enc = [cb4 computeCommandEncoder];
    proxy->_cbuf_off = 0;
    proxy->_auto_barrier = auto_barrier;
    proxy->_dispatched = false;
    return (id<MTLComputeCommandEncoder>)proxy;
  }

  void copy_buffer(id<MTLBuffer> src, size_t soff, id<MTLBuffer> dst, size_t doff, size_t sz) {
    if (!proxy->_enc) return;
    [proxy->_enc barrierAfterEncoderStages:MTLStageDispatch
                       beforeEncoderStages:MTLStageBlit
                         visibilityOptions:MTL4VisibilityOptionNone];
    [proxy->_enc copyFromBuffer:src sourceOffset:soff toBuffer:dst destinationOffset:doff size:sz];
    [proxy->_enc barrierAfterEncoderStages:MTLStageBlit
                       beforeEncoderStages:MTLStageDispatch
                         visibilityOptions:MTL4VisibilityOptionNone];
  }

  void commit_and_wait() {
    [proxy endEncoding];
    [cb4 endCommandBuffer];
    id<MTL4CommandBuffer> cbs[1] = {cb4};
    [q4 commit:cbs count:1];
    const uint64_t target = ++ev_val;
    [q4 signalEvent:ev value:target];
    while (ev.signaledValue < target) {
      [ev waitUntilSignaledValue:target timeoutMS:100];
    }
  }
};

}  // namespace dankattn
