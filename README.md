# DankAttn

![DankAttn](assets/dankattn-hero.jpg)

DankAttn is a from-scratch Metal 4 inference engine for Apple Silicon, written in Objective-C++ and Metal Shading Language with a thin Python OpenAI-compatible HTTP front end.

It currently targets three architectures:

- **Inkling-2-Mini (`engine/inkling2/`)**: `thinkingmachines/Inkling-2-Mini` (`Inkling2LoopForConditionalGeneration`): 26.4B total / 3.4B physical active / 5.4B effective active via a 3-pass looped core (24 physical -> 48 logical layers: 40 SWA-1024 + 8 `kv8` global, native lossless NVFP4/MXFP8 MoE, DankAttn W8 attention/unembed, DSpark MTP drafter + 3-gram lookup). Fits in **32 GB Apple Silicon** (`~21.1 GB` total at 32k context; `~22.9 GB` at 128k).
- **Hy4 (`engine/hy4/`)**: `tencent/Hy4-flash` (`HYV4FlashForCausalLM`): 42 layers (30 sliding-window MLA in bf16, 12 sparse DSA with `kv8` latent caches, 512-expert MoE, 4 SSD-offloaded FP8 Engram layers, and a 5-layer DFly2 drafter).
- **Flash-Next (`engine/`)**: `qwen4_exp` (GDN + sparse attention + MoE + PLE n-gram tables + MTP drafter).

Everything runs directly on Apple Silicon Metal without PyTorch, MLX, or llama.cpp in the inference path. Kernels compile from source at startup with `MTLMathModeSafe` and `MTLMathFloatingPointFunctionsPrecise` so decode, verify, multi-lane batching, and speculation remain bit-identical.

## Requirements

- **Hardware**: Apple Silicon with Metal 4 / `MetalPerformancePrimitives` (`mpp::tensor_ops` cooperative `matmul2d`). Developed and benchmarked on an M5 Ultra. Other Apple Silicon chips with Metal 4 support should work but have not been benchmarked here.
- **OS & toolchain**: macOS with Metal 4.0 support and Xcode Command Line Tools (`clang++` with C++20 and Objective-C++ ARC).
- **Memory**:
  - For **Inkling-2-Mini (`pack-inkling2-mini-w8`)**: ~19.91 GB mapped weights (including DSpark overlay) plus ~1.33 GB GPU state at `cap=32k` (~3.37 GB at `cap=128k`), fitting comfortably on a **32 GB MacBook Pro**.
  - For **Hy4 (`pack-g64-gptq-v3`)**: ~87.3 GB mapped weights plus ~6.5 GB wired GPU state for 4 concurrent lanes (512k main slot + 3x 32k side lanes), plus fast local NVMe for the 63 GB FP8 Engram table.
- **Wired memory limit**: macOS defaults can cap wired GPU memory below what a 90+ GB model needs. The engine checks `iogpu.wired_limit_mb` at startup and prints the exact `sysctl` command if your limit is too low:
  ```sh
  sudo sysctl iogpu.wired_limit_mb=<MB>
  ```
- **Python (front end only)**: Python 3.11+ with the packages in `requirements.txt`.

## Quickstart (Hy4)

1. Build the engine binary:
   ```sh
   sh hy4/build_hy4.sh
   ```
   This produces `build/dankattn-hy4` in a couple of seconds. Metal kernels compile at runtime from `engine/kernels/`.

2. Install front-end dependencies:
   ```sh
   python3 -m venv .venv
   . .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Point `HY4_PACK` at a DankAttn Hy4 pack directory (containing `manifest.json`, weight shards, `engram.bin`, `tokenizer.json`, and `chat_template.jinja`) and start the OpenAI-compatible server on `:8341`:
   ```sh
   HY4_PACK=/path/to/pack-g64-gptq-v3 sh hy4/serve_hy4.sh start
   sh hy4/serve_hy4.sh smoke
   ```
   Or run the front end directly:
   ```sh
   python3 server/hy4_server.py --pack /path/to/pack-g64-gptq-v3 --engine ./build/dankattn-hy4 --port 8341 --cap 524288
   ```

## Benchmarks (M5 Ultra, Hy4 `pack-g64-gptq-v3`)

Measured at 12,272-token context (`pack-g64-gptq-v3`, `kv8` on sparse layers, W4g64 GPTQ experts, W8 dense/attention projections):

| Workload | Result | Notes |
| :--- | ---: | :--- |
| Prefill (12,272 tok) | 1,700 tok/s GPU (1,694 wall) | `mmp` / `mmpg` indirect-dispatch MoE |
| Decode 1 row | 19.08 ms GPU (52.4 tok/s) | Plain single-lane |
| Verify 5 / 8 / 16 rows | 27.61 / 33.24 / 44.76 ms GPU | Bit-identical to 1-row decode |
| Multi-lane decode (`btest`, 16 lanes) | 376.3 tok/s GPU (42.52 ms/step) | 7.38x over 1 lane |
| Speculation (`ids12k`, 400 tok) | 128.1 tok/s wall | 400/400 exact greedy match |
| Speculation (`chat_code_no_think`, 400 tok) | 116.9 tok/s wall | 400/400 exact greedy match |
| Speculation (`edit_spec`, 400 tok) | 357.3 tok/s wall | 400/400 exact greedy match |
| 4-lane joint speculation (`jtest`, greedy) | 199.9 tok/s wall (1.30x) | 0 tokens differ vs single-lane |
| Wired GPU state (512k + 3x 32k lanes) | 6.49 GB | Phase-aliased `MTLHeap` scratch |
| Cold shader-compile startup | 0.7 - 1.0 s | `MTLLibrary` cache in `mtl.h` |

Quality of `pack-g64-gptq-v3` against the bf16 reference: `wikitext-2` (24x 512-token cold-start chunks) KL `0.287` / top-1 `81.9%`; `-28.8%` held-out KL vs uncalibrated RTN (`calib2048` single-document KL `~0.103`); `kv8` vs bf16 KV cache `97.4% - 98.4%` top-1.

See [hy4/ARCHITECTURE_AND_PERF.md](hy4/ARCHITECTURE_AND_PERF.md) for per-op profiles, optimization history, and experiments that were tested and rejected.

## Benchmarks & Quickstart (Inkling-2-Mini, `pack-inkling2-mini-w8`)

Native NVFP4 (L0–19) / MXFP8 (L20–23) MoE + DankAttn W8 attention/unembed (`KL = 0.00042` / 100% top-1 vs BF16; MMLU **86.23%** vs SGLang BF16-KV 85.96%), `kv8` on the 8 global layers, fan-out-aware adaptive DSpark MTP + 3-gram lookup, zero-copy Split-KV multi-lane serving (`:4x-fast` / `:8x-fast` / `:4x-select`), and persistent SSD prefix caching (`19.91 GB` mapped + `1.38 GB` GPU state at `cap=32k`, fits in **32 GB RAM** up to **128k context**):

| Workload | M5 Ultra Mac Studio (96 GB) | Base M5 MacBook Pro (32 GB) |
| :--- | :---: | :---: |
| `vtest` (`M=5` vs `M=1`) / `ctest` (SSD prefix) | `0.000e+00` (**BIT-EXACT**) | `0.000e+00` (**BIT-EXACT**) |
| Prefill 2,048 / 8,192 / 32,768 tok (`MP=512`) | **4,588 / 4,288 / 3,371 tok/s GPU** | **730 / 719 tok/s GPU** (2k / 8k) |
| Decode `M=1` (`ctx=512 / 2k / 8k`) | **93.3 / 90.6 / 89.4 tok/s GPU** (`90.6 / 88.2 / 86.9` wall) | **25.8 / 24.8 / 24.5 tok/s GPU** |
| Speculative (`code_prompt.bin`, 256 tok) | **204.9+ tok/s wall** (256/256 exact) | **46.3 tok/s wall** (256/256 exact) |
| Multi-lane (`jspec` 4 lanes / `lbench` 16 lanes) | **317.1 tok/s wall** (1024/1024 exact) / **501 tok/s GPU** | **65.8 tok/s wall** (4 lanes) |
| Parallel Section Fill (`:4x-fast` / `:8x-fast`, live SSE) | **319.0 / 343.6 eff visible tok/s** (TTFT 1.55–2.15 s) | — |
| SSD Prefix Cache restore (`ctest`, 8,000 tok) | **20.9 ms** (346 MB, `0.000e+00` exact) | — |

```sh
# Build & verify
sh inkling2/build_inkling2.sh
./build/dankattn-inkling2 "$INK_PACK" vtest
./build/dankattn-inkling2 "$INK_PACK" ctest prompt_ids.bin 1500
./build/dankattn-inkling2 "$INK_PACK" pfbench 2048
./build/dankattn-inkling2 "$INK_PACK" spec 256 prompt_ids.bin

# Start OpenAI-compatible server on :8342 (8 lanes, 128k L0 + 16k Split-KV side lanes, SSD prefix cache)
INK_PACK=/path/to/pack-inkling2-mini-w8 INK_CAP=131072 INK_LANES=8 INK_SIDE_CAP=16384 \
  sh server/inkling2_serve.sh start
sh server/inkling2_serve.sh smoke
```

See [inkling2/ARCHITECTURE_AND_PERF.md](inkling2/ARCHITECTURE_AND_PERF.md) for architecture details, head-to-head benchmarks, MoE fan-out analysis, packing instructions, CLI flags, and environment variables.

## Architecture Highlights

- **Register-direct quantized matmuls (`engine/kernels/mm2.metal`)**: W4, W6, and W8 weights unpacked directly into `mpp::tensor_ops::matmul2d` cooperative tensors with two-level split-K (`mm2`), shared-weight prefill (`mmp`), and indirect-dispatch grouped MoE prefill (`mmpg`) with fused SwiGLU epilogues (`EPI=3`).
- **Exact multi-lane batching & joint speculation (`engine/hy4/spec.inc`, `serve.inc`)**: Up to 16 rows per step across concurrent slots. Each lane proposes DFly2 + zero-allocation 3-gram context-lookup drafts (`engine/lookup.h`), `draft_lanes()` batches drafting while prefetching Engram anchors, and `step_lanes()` verifies all lanes in one forward pass. Sampling uses counter-hashed Gumbel-max keyed by `(seed, pos)`, keeping speculation exact at any temperature.
- **Concurrent GPU dispatch & phase-aliased scratch**: `MTLDispatchTypeConcurrent` overlaps independent sub-ops (`attn_in || gate`, `qb || idx_pool`, `kb || idx_score/topk`). Non-overlapping Engram, Attention, and FFN scratch buffers alias inside one `MTLHeapTypePlacement` heap.
- **SSD prefix cache**: Slot 0 saves and restores long-context KV/indexer state asynchronously to local SSD (`HY4_PCACHE`, `F_NOCACHE`) so side lanes do not stall on disk I/O.
- **Interleaved chunked prefill**: Prefill yields between chunks and drops chunk size (`HY4_PF_DEC_CHUNK`, default 256) whenever decode lanes are active to bound inter-token latency.

## Repository Layout

```
engine/
  mtl.h                 Metal runtime, library cache, wired-memory check
  mm_args.h             Shared MmArgs struct (host/kernel ABI)
  lookup.h              Zero-allocation 3-gram context-lookup table
  kernels/
    mm2.metal           NVFP4/MXFP8/W4/W6/W8 decode, prefill, and grouped MoE matmuls
    inkling2.metal      Inkling-2-Mini looped core, message gates, split-K GQA, MoE, DSpark MTP
    hy4.metal           Hy4 attention, indexer, MoE routing, Engram, DFly2, sampling
    attn.metal          Flash-Next attention kernels
    ops.metal           Flash-Next GDN and MoE kernels
  inkling2/
    inkling2.h, inkling2.mm  Inkling-2-Mini loader, step/step_lanes, Split-KV, rollback, DSpark, SSD prefix cache
    serve.inc           Binary protocol (OPUS/OPCX), multi-lane scheduler, FPR modes, SSD prefix cache
    main.mm             CLI driver (smoke, vtest, ctest, forktest, ltest, pfbench, bench, vbench2, spec, jspec, serve)
  hy4/
    config.h            All HY4_* environment variables and defaults
    hy4.h, hy4.mm       Hy4 model loader, step/step_lanes, draft/draft_lanes, prefix cache
    spec.inc            Single-lane and joint multi-lane speculation
    serve.inc           Binary protocol (OPUS/OPCX) + continuous multi-lane scheduler
    main.mm             CLI driver (bench, spec, vtest, ntest, ctest, jtest, pprof, dprof, serve)
server/
  engine_frontend.py    Shared subprocess supervisor, binary framing, prefix splicer, HTTP helpers
  inkling2_server.py    OpenAI-compatible FastAPI server for Inkling-2-Mini (tools, reasoning, :Nx-fast / :Nx-select)
  inkling2_serve.sh     Start/stop/status/smoke wrapper for :8342
  hy4_server.py         OpenAI-compatible FastAPI server for Hy4
  dankattn_server.py    OpenAI-compatible FastAPI server for Flash-Next
inkling2/
  ARCHITECTURE_AND_PERF.md  Inkling-2-Mini architecture, benchmarks, and CLI reference
  build_inkling2.sh     Build script for build/dankattn-inkling2
  convert/              Native NVFP4/MXFP8/W8 packer (pack_inkling2.py) & DSpark packer (pack_mtp.py)
  ref/                  Standalone PyTorch golden reference (inkling2_golden.py)
hy4/
  build_hy4.sh          Build script for build/dankattn-hy4
  serve_hy4.sh          Start/stop/status/smoke wrapper for :8341
  test_gate.sh          7-step regression gate
  convert/              Checkpoint conversion, GPTQ calibration, Engram & DFly2 packing
  ref/                  Reference comparison scripts
```

## Converting a Checkpoint

Conversion scripts for `tencent/Hy4-flash` live in `hy4/convert/`:

- `collect_gptq.py` / `gptq.py`: calibration Hessian collection and GPTQ quantization.
- `pack_hy4.py`: packs weights, manifest, and scales into DankAttn's layout.
- `pack_dfly.py`: packs the 5-layer DFly2 drafter.
- `build_engram.py`: builds the FP8 Engram table (`engram.bin`).

## Verification & Tests

The engine includes built-in bit-exactness and performance tests:

```sh
./build/dankattn-hy4 "$PACK" vtest ids3k.bin             # 16-row verify == 1-row decode (0 diff logits)
./build/dankattn-hy4 "$PACK" ntest ids3k.bin 4           # 4-lane batch == single-lane (0 diff logits/tokens)
./build/dankattn-hy4 "$PACK" ctest ids3k.bin             # SSD prefix cache save/load + truncate round trip
./build/dankattn-hy4 "$PACK" bench ids12k.bin 1,5,8,16   # Prefill + decode/verify latency
./build/dankattn-hy4 "$PACK" pprof ids12k.bin            # Per-op GPU timestamp breakdown
```

`hy4/test_gate.sh` runs the full regression suite (`vtest`, `ntest 4`, `ctest`, `jtest`, `bench`, `spec`) and restarts `:8341` with a smoke check.
