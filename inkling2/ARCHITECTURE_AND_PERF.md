# DankAttn (Inkling-2-Mini) Architecture, Benchmarks & Engineering Log

> Last updated: 2026-11-21 (branch `inkling2-mini`, HEAD `7ba4268`; benchmarked on Apple M5 Ultra Mac Studio 96 GB and Base M5 MacBook Pro 32 GB).

## 1. Model & Pack Architecture (`pack-inkling2-mini-w8`)

- **Target Model**: [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (`Inkling2LoopForConditionalGeneration`), **26.4B total** / **3.4B physical active** / **5.4B effective active** parameters via a 3-pass looped core.
- **Looped Layer Topology**: 24 physical layers execute as **48 logical visits**:
  - **Prelude**: physical layers `0..5` (6 logical layers)
  - **Looped Core**: physical layers `6..17` × **3 passes** (36 logical layers). Passes 1 and 2 restart from the pre-loop `(h, block_states)` snapshot and gate in cross-pass Block Attention Residual messages through rank-64 low-rank controllers (`ctrl_q`, `ctrl_k`, `ctrl_v`, `ctrl_mlp`).
  - **Coda**: physical layers `18..23` (6 logical layers)
- **Attention & KV Geometry**:
  - **40 logical SWA-1024 layers** (sliding window `1024`, BF16 KV ring buffer) + **only 8 logical global layers** (`5, 11p0, 17p0, 11p1, 17p1, 11p2, 17p2, 23`).
  - No RoPE in the main text tower; uses `d_rel=16` (`rel_extent=1024`) relative positional bias, depthwise short convolutions (`sconv`, `kernel_size=4`) on K, V, post-attention, and post-MLP, plus `log_scaling` on global layers.
  - Global KV caches default to **`kv8`** (`INK_KV8=1`), keeping memory footprint small even at 128k context.
- **Native Lossless MoE Precision (Avoiding Loop-Compounding Drift)**:
  - Looped models amplify small quantization errors across passes (standard low-bit GGUF re-quantization degrades top-1 match noticeably on Inkling-2-Mini).
  - DankAttn keeps the MoE experts in their **native checkpoint formats with zero re-quantization**:
    - Layers `0..19`: native **NVFP4** (`WBITS=40`, `E2M1`, `GS=16`, in-register `E2M1_LUT[16]` decode inside `mm2.metal`).
    - Layers `20..23`: native **MXFP8** (`WBITS=80`, `E4M3FN`, `GS=32`, exact FP16 E8M0 scales).
  - Attention projections (`qgrs`, `k_proj`, `v_proj`, `o_proj`) and `lm_head` use DankAttn **W8 (`GS=128`)**; `attn_res`, routers, `sconv`, and cross-pass controllers remain **FP32**.
  - **Measured quality vs. BF16 golden reference** (`inkling2/ref/inkling2_golden.py`): `KL = 0.000420`, **100% top-1 token match**.
- **DSpark MTP Speculative Drafter (`mtp_manifest.json`, 382.3 MB overlay)**:
  - 3 dense Qwen3-like layers (`L0, L2` SWA-1024, `L1` global, NeoX RoPE `theta=1e6`, `I=9216` W6) tapping logical layers `[5, 23, 41]` (pre-layer) and `48` (final normalized hidden state).
  - `block_size = 8` (`[anchor, mask_1 .. mask_7]` → up to **7 draft tokens** per call) followed by a sequential rank-256 Markov RNN head with pre-folded `fold_w = lm_head @ hidden_out.weight` (`[200064, 256]` W8, 51 MB).
  - Combined with online softmax `p_max` confidence early-cutoff (`INK_DRAFT_CUT=0.50`) and zero-allocation 3-gram context lookup (`engine/lookup.h`, up to `M=16` on context matches).
- **Memory Footprint (`pack-inkling2-mini-w8`)**:
  - **Mapped weights**: `19.52 GB` (3 shards) + `0.38 GB` (DSpark overlay) = **`19.91 GB`**.
  - **GPU state (`cap=32768`, `kv8=1`, `dspark=1`)**: **`1.22 GB`** → **`21.13 GB` total**.
  - **GPU state (`cap=131072`, `kv8=1`, `dspark=1`)**: **`2.99 GB`** → **`22.90 GB` total** (fits comfortably inside a 32 GB MacBook Pro).

---

## 2. Metal 4 Engine Highlights (`engine/inkling2/`, `engine/kernels/inkling2.metal`)

- **Register-direct NVFP4 / MXFP8 / W8 matmuls (`engine/kernels/mm2.metal`)**: Shared cooperative `mpp::tensor_ops::matmul2d` kernels with fused `SoftcappedSwiGLU` (`SWIGLU_SOFTCAP_G=4.0`, `SWIGLU_SOFTCAP_U=25.0`, `EPI=3`).
- **Two-regime execution (`M <= 16` vs. `M <= 256`)**:
  - **Decode & Speculative Verify (`M <= 16`)**: `mm2` (two-level split-K) + SIMD top-8 MoE routing (`ink_moe_route`), bit-exact across `M=1..16` (`vtest` `0.000e+00`).
  - **Chunked Prefill (`MP = 256`)**: `mmp` (`MMP_CH=8, MMP_NF=2, MMP_FAST=1`) for dense W8 attention projections + parallel per-row routing (`ink_moe_topk_pf`), 16-row-aligned compact indirect worklist (`ink_moe_plan_pf`), `mmpg` (`MMP_CH=4, MMP_NF=2, MMP_FAST=1`) for NVFP4/MXFP8 MoE, and gather-accumulate (`ink_moe_accum_pf`). Only the final row runs `lm_head` during prefill.
- **Concurrent GPU dispatch & exact rollback**:
  - `MTLDispatchTypeConcurrent` overlaps `qgrs || k_proj || v_proj`, `q_rel || k_sconv || v_sconv`, and the 3-stage parallel cross-pass message gate (`ink_mg_prep` → `ink_mg_down` → `ink_mg_apply`).
  - Barrier-free simdgroup split-K GQA attention (`ink_attn_split` + `ink_attn_reduce`).
  - Per-logical-layer `conv_hist` ring snapshots enable exact `sconv` state rollback on partial speculative acceptance (100% greedy-exact match vs. `M=1`).

---

## 3. Verified Benchmarks: M5 Ultra Mac Studio (96 GB) vs. Base M5 MacBook Pro (32 GB)

All runs use `pack-inkling2-mini-w8` (`19.91 GB` mapped + `1.22 GB` GPU state, `cap=32768`, `kv8=1`, `dspark=1`, 3 passes).

| Workload | M5 Ultra Mac Studio (96 GB) | Base M5 MacBook Pro (32 GB, 153 GB/s) | Studio / MBP |
| :--- | :---: | :---: | :---: |
| **`vtest` (`M=5` vs `M=1`)** | `0.000e+00` (**BIT-EXACT**) | `0.000e+00` (**BIT-EXACT**) | — |
| **Prefill check (`M=64` vs `M=1`)** | Exact next-token (`149884`) | Exact next-token (`149884`) | — |
| **Prefill 2,048 tok (`MP=256`)** | **1,610.9 tok/s GPU** (1,601.9 wall, 1.28 s) | **314.2 tok/s GPU** (313.9 wall, 6.52 s) | 5.1× |
| **Prefill 8,192 tok (`MP=256`)** | **1,272.5 tok/s GPU** (1,266.8 wall, 6.47 s) | **236.2 tok/s GPU** (236.0 wall, 34.71 s) | 5.4× |
| **Decode `M=1` @ `ctx=512`** | **75.9 tok/s GPU** (71.1 wall, 13.18 ms) | **24.2 tok/s GPU** (23.7 wall, 41.36 ms) | 3.1× |
| **Decode `M=1` @ `ctx=2,048`** | **73.2 tok/s GPU** (68.7 wall, 13.66 ms) | **23.4 tok/s GPU** (23.0 wall, 42.67 ms) | 3.1× |
| **Decode `M=1` @ `ctx=8,192`** | **69.0 tok/s GPU** (65.0 wall, 14.50 ms) | **22.4 tok/s GPU** (22.0 wall, 44.59 ms) | 3.1× |
| **Speculative (`K=7, cut=0.50, lk=1`)** | **167.8 tok/s wall** (2.36× vs `M=1`) | **41.4 tok/s wall** (1.75× vs `M=1`) | 4.1× |
| **Spec exactness (`code_prompt.bin`)** | **256/256 (100% EXACT)** | **256/256 (100% EXACT)** | — |

### Speculation Sweep & the MoE Expert Fan-Out Tax (`spec 256 code_prompt.bin`)

On a 128-expert MoE (top-8 per token), verifying `M = 1 + K` diverse draft tokens loads ~45–50 unique experts/layer at `M=8` (vs. 8 experts/layer at `M=1`). Trimming low-confidence tail drafts (`p_max < 0.50`) eliminates wasted expert loads without losing accepted tokens:

| Config (`lk=1`) | Tok/Round | Avg Drafts | Studio Verify | Studio Wall | MBP Verify | MBP Wall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `K=7, cut=0.00` (`lk=0` baseline) | 4.35 | 7.00 | 22.34 ms | 153.7 tok/s | 103.53 ms | 36.3 tok/s |
| `K=3, cut=0.35` | 3.38 | 5.14 | 19.95 ms | 144.7 tok/s | 88.69 ms | 34.5 tok/s |
| `K=4, cut=0.35` | 3.81 | 5.67 | 20.77 ms | 156.0 tok/s | 88.86 ms | 38.9 tok/s |
| `K=5, cut=0.35` | 4.15 | 6.15 | 21.42 ms | 161.8 tok/s | 93.52 ms | 39.8 tok/s |
| `K=7, cut=0.00` | 4.58 | 8.11 | 23.34 ms | 158.9 tok/s | 107.73 ms | 37.5 tok/s |
| **`K=7, cut=0.50` (default)** | **4.58** | **6.82** | **21.92 ms** | **167.8 tok/s** | **96.81 ms** | **41.4 tok/s** |
| `K=7, cut=0.65` | 4.28 | 5.54 | 20.37 ms | 166.3 tok/s | 94.22 ms | 39.1 tok/s |

### Hardware Scaling Breakdown
- **Plain `M=1` decode (3.1×)**: Memory-bandwidth bound on both chips; scales directly with memory bandwidth.
- **Long-context decode (`512 → 8,192`)**: Under 9% slowdown across 16× more context on both machines thanks to 40 SWA-1024 layers + only 8 `kv8` global layers.
- **Chunked prefill `MP=256` (5.1–5.4×)**: Compute-bound (scales with GPU core count). The ~21–25% dip from 2k to 8k comes from using the decode-style split-K attention kernel (`ink_attn_split`) on the 8 global layers during prefill.
- **Speculative decode (4.1×)**: Mixed bandwidth/compute; M5 Ultra absorbs the `M=8` MoE expert fan-out cleanly (2.36× speedup), while Base M5 still achieves 1.75× speedup (41.4 tok/s wall) in 32 GB RAM.

---

## 4. Quickstart, Packing & CLI Reference

### Convert & Pack (Linux / Python)
```sh
# 1. Pack main Inkling-2-Mini weights (native NVFP4/MXFP8 MoE + DankAttn W8 attn/unembed, ~19.52 GB)
python3 inkling2/convert/pack_inkling2.py \
  --model /path/to/Inkling-2-Mini \
  --out /path/to/pack-inkling2-mini-w8

# 2. Pack DSpark MTP drafter overlay (~382 MB) into the same pack directory
python3 inkling2/convert/pack_mtp.py \
  --model /path/to/Inkling-2-Mini \
  --out /path/to/pack-inkling2-mini-w8
```

### Build & Run (macOS, Metal 4)
```sh
sh inkling2/build_inkling2.sh

./build/dankattn-inkling2 "$PACK" smoke                      # 5-token Paris sanity check + 32-token gen
./build/dankattn-inkling2 "$PACK" vtest                      # M=5 vs M=1 bit-exactness check (0.000e+00)
./build/dankattn-inkling2 "$PACK" pfbench 2048               # M=64 check + MP=256 chunked prefill bench
./build/dankattn-inkling2 "$PACK" bench 2048 64              # Plain M=1 decode at ctx=2048
./build/dankattn-inkling2 "$PACK" spec 256 prompt_ids.bin    # DSpark + 3-gram lookup + exact greedy check
./build/dankattn-inkling2 "$PACK" dprof 512                  # Per-op GPU timing breakdown
```

### Environment Variables

| Variable | Default | Description |
| :--- | :---: | :--- |
| `INK_CAP` | `32768` | Global KV cache capacity in tokens (up to `131072` in 32 GB RAM with `kv8=1`) |
| `INK_PASSES` | `3` | Number of core loop passes (`1..3`, default from manifest) |
| `INK_KV8` | `1` | `1`: int8 quantized global KV cache (`kv8`); `0`: BF16 global KV |
| `INK_DRAFT_K` | `7` | Max DSpark draft tokens per round (`1..7`, verify `M = 1 + K`) |
| `INK_DRAFT_CUT` | `0.50` | Early-cut DSpark drafts when `p_max < INK_DRAFT_CUT` (`0.0` disables) |
| `INK_LOOKUP` | `1` | `1`: enable 3-gram context-lookup drafting (`engine/lookup.h`, up to `M=16`) |

---

## 5. Engineering Log & Backlog

### Milestones
- **`298618b`**: Native NVFP4/MXFP8 + `SoftcappedSwiGLU` in `mm2.metal` / `layout.py`, `pack_inkling2.py`, golden reference `inkling2_golden.py`, and Wave 2A Metal 4 engine (`vtest` `0.000e+00` bit-exact after fixing missing threadgroup barriers in `ink_msg_gate` / `ink_kv_conv_cache` and odd `N/32` `nft=2` bounds in `enc_mm2`).
- **`6848e0a`**: `kv8` global KV cache, exact `sconv` rollback (`conv_hist`), `pack_mtp.py`, and full DSpark MTP speculative decoding (`spec`).
- **`7ba4268`**: `mmp`/`mmpg` chunked prefill (`MP=256`, `pfbench`), online softmax `p_max` confidence cutoff (`INK_DRAFT_CUT`), and `engine/lookup.h` 3-gram context lookup.

### Planned Next Steps (Not Yet Started)
- **OpenAI-compatible `serve` + staggered-pass multi-lane batching** (Item 3, deferred until metrics review).
- **Tiled FlashAttention prefill kernel** for the 8 global layers to flatten 8k–32k prefill throughput.
