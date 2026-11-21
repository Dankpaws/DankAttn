// Shared MmArgs layout for host C++ (engine/model.mm, engine/hy4/hy4.mm) matching engine/kernels/mm2.metal.
#pragma once
#include <cstdint>

struct MmArgs {
  uint32_t N = 0, K = 0, M = 0;
  uint32_t ksplit = 1;
  uint32_t wstride = 0, sstride = 0;
  uint32_t xstride = 0;
  uint32_t pstride = 0;
  uint32_t ystride = 0;
  uint32_t sgs = 1;
  uint32_t ldx = 0;
  float xdiv = 1.0f;
  uint32_t psplit = 0;
  uint32_t R = 0, H = 0, HC = 0;
  uint32_t pf_z0 = ~0u, pf_gx = 0, pf_gy = 0, pf_tpt = 0, pf_n = 0;
  uint32_t red = 0;
  uint32_t rm = 0;
};
static_assert(sizeof(MmArgs) == 23 * 4, "MmArgs must stay ABI-identical to engine/kernels/mm2.metal");
