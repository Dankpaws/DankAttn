# ERNIE-5.5-Flash-Preview in DankAttn: Architecture & Performance

This document covers the Metal 4 implementation, memory budget, measured M5 Ultra benchmarks, and key kernel/hardware lessons for `engine/ernie55/` (`ERNIE-5.5-Flash-Preview`, `ernie5_5_moe`).

## Architecture Overview

- **43 layers** (`H=4096`, 4-stream Sinkhorn mHC residual mixing with 20 Sinkhorn-Knopp iterations per sublayer):
  - **37 HCA layers**: sliding-window attention (`win=1024`) + C128 compressed-KV memory (`cr=128`), 64 heads × 512 dim (`nope=448`, `rdim=64`), VHA + inverse RoPE (`8` output groups × `1024` rank).
  - **6 DSA layers** (every 7th layer: L7, L14, L21, L28, L35, L42): DeepSeek-style sparse latent attention (`idx_topk=2048` + `win=128`), 64 heads, `kv8` (int8 symmetric per-64-dim latent + bf16 RoPE).
- **FFN / MoE**:
  - **L0**: Dense SiTU-GLU MLP (`I=14336`, W8-g128).
  - **L1–L42**: 512 routed experts (top-10, `moe_I=2048`, `moe_lat=2048`) + shared SiTU-GLU expert (`beta=4.0`, `lbeta=25.0`).
  - L1 experts are W4-g64; L2–L21 experts are W2-g32 NF2 and L22–L42 experts are W2-g64 NF2 (13-clip NF2 RTN).
- **Output head**: `lm_head` (`vocab=201216`, W8-g128) + MultiMax quadratic head.
- **Speculative decoding**: 3-gram prompt-lookup decoding (PLD, `k_draft=7`) is enabled by default in `server/ernie55_server.py` (disable with `--no-spec`; enable in CLI `gen`/`suite` with `E5_SPEC=1`).

## Memory Footprint (`cap=262144`)

| Component | Size | Notes |
| :--- | ---: | :--- |
| Mapped weights (`pack-ernie55-w2-g32-w8attn`) | **85.51 GiB** | W2-g32 NF2 MoE (L2–21) + W2-g64 NF2 MoE (L22–42) + W4-g64 (L1) + W8-g128 dense/attn/unembed |
| Wired GPU state (`PREFILL_M=1024`, `cap=262144`) | **2.03 GiB** | Compact `COMP_RING=256` + shared prefill scratch + phase-aliased `iq_h`/`mmpart` |
| **Total** | **87.54 GiB** | Fits under `iogpu.wired_limit_mb=90112` (88.0 GiB) on 96 GB Apple Silicon |

## Measured Performance (M5 Ultra, 96 GB)

| Benchmark | Pre-Opt | Current | Notes |
| :--- | :---: | :---: | :--- |
| 4k prefill (`p5_needle4k`, 3,866 tok) | 1,603 tok/s GPU | **1,797–1,804 tok/s GPU** | `PREFILL_M=1024` (1,776 tok/s wall) |
| 16k prefill (`p6_needle16k`, 15,864 tok) | 1,517 tok/s GPU (705 wall) | **1,682–1,687 tok/s GPU** (**1,677 wall**) | 2.38x wall prefill speedup |
| 4k decode `M=1` | 15.74 ms / 63.6 tok/s | **15.24 ms / 65.6 tok/s** | `mm1` pipelined W8 + double-buffered W2 |
| 16k decode `M=1` | 16.21 ms / 61.7 tok/s (56.6 wall) | **15.75 ms / 63.5 tok/s** (**59.4 wall**) | 2-CB decode + branchless `e5_hca_attn_m1` |
| 4k decode `E5_SPEC=1` | — | **87.1 tok/s GPU** (**80.4 wall**) | Fused `wo_a EPI=5` + `M<=8` split-K attn |
| 16k decode `E5_SPEC=1` | — | **84.1 tok/s GPU** (**78.0 wall**) | `k_draft=7` 3-gram PLD |
| `eval_144` NLL / PPL | 1.5320 / 4.6276 | **1.4795 / 4.3907** | `0.000000` run-to-run max abs diff |

## Key Optimizations & M5 Ultra Roofline

1. **Prefill (`PREFILL_M=1024`)**:
   - Compact `COMP_RING=256` per-layer compressor ring + shared prefill scratch (-116 MB), phase-aliased `iq_h = mmpart`, and trimmed scratch buffers keep `state = 2.03 GiB` safely under macOS's 88 GiB wired limit.
   - `bf16` HCA Q/O buffers, scale-folded `(P * Ks) @ V_int8` in DSA attention, positive-scale factoring + `bf16` `iq_h` in DSA indexer, and fused inject+Gram+`matmul2d` in `e5_mhc_pre_mma`.
2. **Decode `M=1` Roofline**:
   - Active weight bytes per token: **~9.97 GB** (`lm_head` alone saturates **1,030–1,040 GB/s** → **9.6 ms** pure DRAM floor).
   - ~600 serialized compute dispatches/token × ~5.5 µs ramp/drain floor = **~3.3 ms**, plus attention/mHC/routing/VHA = **~2.4 ms** → **15.24 ms** measured.
   - Continuous cross-group W8 software pipelining + 2-deep double-buffered W2 in `mm1`, exact 64-TG (1.0-wave) 3-mix `e5_mhc_part`, branchless 2-loop `e5_hca_attn_m1` with fused Q-RMSNorm+RoPE, 64-TG `e5_hca_vha_invrope_m1`, barrier-free 1-SG `e5_route_topk_pack_m1`, and 2-CB decode (`flush_cb` at `lid == 21`).
3. **Speculative Verify (`M=2..8`)**:
   - Fused `wo_a` gate×sigmoid + `frag_a` pack (`EPI=5`) directly in `mm2`, plus split-K FlashDecoding (`e5_hca_attn_mma_split`, `e5_dsa_attn_mma_split`) and `e5_mhc_part` extended to `M <= 8` so 7-draft (`M=8`) verify steps saturate all 64 GPU cores.
4. **Ruled-Out Experiments (avoid repeating)**:
   - `half4` W8 `mm1` overflows on ERNIE outlier activations (`|x| > 128` → NaN); keep `float4`.
   - `mm1` split-K (both atomic `seq_cst` and non-atomic `mm1_red`) and atomic single-kernel `e5_mhc_m1_fused` all regressed vs Metal's near-free in-encoder `MTLBarrierScopeBuffers`.
   - `W2_BIT` bitwise decode in `mm1` regressed vs `W2_LUT256` constant-memory lookup.

## CLI & Environment Variables

- `dankattn pull ernie55` — download `Dankpaws/ERNIE-5.5-Flash-Preview-DankAttn-w2-g32`
- `dankattn serve ernie55 start|stop|status|smoke|logs` — manage `:8343` OpenAI-compatible daemon
- `E5_PACK` — override pack directory
- `E5_PORT` — override HTTP port (default `8343`)
- `E5_CAP` — override context capacity (default `262144`)
- `E5_DRAFT_K` — max 3-gram PLD draft length (default `7`)
- `E5_SPEC=1` — enable 3-gram PLD in offline `gen` / `suite` CLI modes (server has it ON by default; pass `--no-spec` to disable)
- `E5_PROF=1|2` — print per-op GPU profiling breakdown
