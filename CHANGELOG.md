# Changelog

## v2026.11.28

### Engine Core & Refactoring
- **Shared C++ runtime (`engine/core/`):** extracted `env.h`, `pack.h`, `runtime.h`, `proto.h`, `cli.h`, `sample.h`, `pcache.h`, and `scheduler.h`, deduplicating `Lin`, `Raw`, `jload`, `parse_json`, `mmap_file`, `host_now`, `new_sample_buffer`, `Rec`, `PrefixCache`, the `OPUS`/`OPCX`/`OPMK` reader/inbox/cancel/mask-queue loop, adaptive prefill chunking, masked/logprobs sampling, and CLI helpers across Hy4, Inkling2, ERNIE-5.5, and Flash-Next.
- **Unified multi-engine build (`build.sh`):** `sh build.sh [all|hy4|inkling2|ernie55|flash|bench]` builds all four engines and GPU unit tests from one script; per-model `build_*.sh` wrappers delegate to it.
- **Modular Metal shaders (`engine/mtl.h`, `engine/kernels/mm2_epilogues.metal`, `engine/kernels/sampling.metal`):** added recursive local `#include "..."` expansion in `mtl::slurp()`, split model-specific epilogues (`EPI=1..5`, `epi3_regs`) out of `engine/kernels/mm2.metal`, and consolidated GPU top-k/top-p/min-p sampling into `engine/kernels/sampling.metal`.
- **Opt-in Metal 4 command encoding (`engine/core/mtl4.h`, `DANKATTN_MTL4=1`):** `MTL4CommandQueue` + 16-deep serialized `MTL4CommandBuffer` ring (`flush()` early-commit with `signalEvent`/`waitForEvent` + encoder-tracked `_dispatched` barrier scoping with `MTL4VisibilityOptionNone` intra-encoder barriers and `add_unwired_buffers()` for unwired drafter/vision files) + `MTL4ArgumentTable` with redundant-bind/PSO elision and pre-baked static tables across ERNIE-5.5, Hy4 (both `step()` and DFly2 `draft()`), and Inkling2 (`bench/mtl4_floor.mm` built by `sh build.sh bench`). On M5 Ultra Hy4 (`pack-dfly2b6`, 12k ctx), `DANKATTN_MTL4=1` now beats Metal 3 end-to-end (`125.6` vs `122.4 tok/s` on `ids12k`, `220.7` vs `211.4 tok/s` on `chat_code_no_think`, `0` queue stalls, 100% bit-exact `vtest` + `400/400`).
- **Unified Python chat frontend (`server/engine_frontend.py`):** consolidated `parse_chat_params`, `find_stop_cut`, `finish_reason_of`, `make_sse_chunk`, `make_usage`, and `run_chat_completion` across `hy4_server.py`, `ernie55_server.py`, `inkling2_server.py`, and `dankattn_server.py`.

### Shader Cache, Quantization & Paged KV
- **Disk-cached `MTLBinaryArchive` (`engine/mtl.h`):** caches compiled Metal pipeline states in `~/Library/Caches/DankAttn/metallib` keyed by device, OS, source hash, and defines (`DANKATTN_SHADER_CACHE=0` to disable), cutting warm Hy4 server startup to ~1.5–7s.
- **`W3` and `MXFP4` quantization tiers (`engine/kernels/mm2.metal`, `convert/layout.py`, `bench/quant_test.mm`):** added `WBITS=3` (`gs=64/128`, 12 B/lane) and `WBITS=41` (`MXFP4`, E2M1, `gs=32`) kernels, packing layouts, and GPU unit tests verified bit-exact (`0.000e+00` error across `mm1`/`mm2`/`mmp`, `M1==M16` `0` bit diffs).
- **Lossless Paged KV + Radix Prefix Tree (`engine/core/kv_radix.h`, `bench/kv_radix_test.mm`):** 128-token block-table pool with refcounted zero-copy prefix sharing, LRU leaf eviction, and `paged_kv` prefix cache save/restore; wired into Hy4 behind opt-in `HY4_PAGED_KV=1` (`0.000e+00` diff on `vtest`/`ntest 4`/`calib2048`).
- **Opt-in TurboQuant 4-bit KV compression (`HY4_KV4=1`, `engine/kernels/hy4.metal`, `engine/hy4/`):** hybrid Rademacher + `H_32` Walsh-Hadamard + 16-level Lloyd-Max Gaussian quantization on the 512-dim MLA latent (`kv_nope`) with 8-bit `k_rope` (`368 B/row` vs `640 B` `kv8` and `1152 B` `bf16` — 42.5% smaller than `kv8`, 68% smaller than `bf16`; cuts 131k-cap Hy4 state from `1.83 GB` → `1.39 GB`/slot). `Q` is rotated once before the key loop and `O` is inverse-rotated once after `h4_attn_merge` (zero Hadamard ops in the hot attention loop); verified bit-exact `0 diff` on `vtest` (M3, `DANKATTN_MTL4=1`, `HY4_PAGED_KV=1`), `400/400` exact greedy match on `spec`, and `KL=0.150` / `88.5%` top-1 vs `bf16` at 12k context.
- **Adaptive prefill chunking:** scales prefill chunk size dynamically based on available memory headroom.
- **Hy4 MoE split tuning & bandwidth probe (`engine/hy4/hy4.mm`, `bench/moe_bw.mm`):** added `bench/moe_bw.mm` (`sh build.sh bench`) to benchmark grouped MoE W4 unpack and geometry on M5 Ultra (confirming stock ALU unpack beats 256/16-entry LUTs and hits ~988 GB/s at `M=16`), added missing `moe_dn` `gs=128` (`sgs=1, ksplit=2`) override and bumped `moe_gu` `gs=64` to `sgs=4, ksplit=1`, cutting Hy4 12k decode/verify latency from `19.85 / 28.23 / 33.66 ms` to `19.58 / 27.90 / 33.25 ms` GPU (`1 / 5 / 8` rows, bit-exact `0` diffs on `vtest`).

### API & Serving
- **Anthropic Messages API (`server/anthropic_compat.py`):** native `/v1/messages` (non-streaming + SSE streaming) and `/v1/messages/count_tokens` registered automatically across all DankAttn servers.
- **Zero-dependency structured outputs with Jump-Forward Decoding (`server/json_schema_dfa.py`, `OPMK` + `OPFF`):** `response_format: {"type": "json_schema", ...}` with first-character bucketed `VocabMasker`, engine-side `OPMK` logit masking, and `JsonSchemaDFA.forced_prefix()` + `OPFF` batched jump-forward decoding (skips decode steps on deterministic JSON syntax, keys, `const`, and single-value `enum` spans across Hy4, ERNIE-5.5, and Inkling2).
- **Zero-copy UMA + GPU `OPMK`/logprobs sampling (`engine/core/sample.h`, `engine/core/gpu_sample.h`, `engine/kernels/sampling.metal`, `bench/sample_test.mm`):** eliminated the per-token 800 KB logits copy, extra 1-row prefill step, and 1.6 MB `partial_sort` allocation on `OPMK`/logprobs requests across Hy4, ERNIE-5.5, and Inkling2 (stack top-K + `-24` tail cut on direct UMA `logits_buf.contents`; optional GPU bitmask kernel `dank_sample_masked_lp` behind `DANKATTN_GPU_SAMPLE=1`, verified bit-exact in `sample_test`).
- **Single-port multi-model router (`server/router.py`):** serves multiple DankAttn engines behind one port with on-demand backend switching.
- **Consolidated pack auto-detection (`hy4/serve_hy4.sh`):** auto-detects `models/pack-dfly2b6` vs `models/pack-v4-h128-w6` and sets matching `HY4_CAP` / `HY4_SIDE_CAP`.

## v2026.11.26

### Hy4 DFly2-B6 & Inkling2 DFlash2
- **Hy4 DFly2-B6 drafter:** added `DFly2-B6` LatentMoE + Attention-Conditioned speculative drafter, made `HY4_QOS=1` default, and added bit-exact audit logging.
- **Inkling-2-Mini DFlash2:** added `DFlash2` block-diffusion drafter overlay, split-K SWA-2048 attention, and updated `v2026.11.26` model cards/docs.
- **ERNIE-5.5 docs:** corrected quantization description to hybrid `W2-g32`/`W2-g64` NF2 RTN.

## v2026.11.25

### Hy4 Engine & `pack-v4-h128-w6`
- **New default Hy4 pack (`Dankpaws/Hy4-flash-DankAttn-v4-h128-w6`, 119 GB):** saves ~32 GB of SSD storage over `v3` (`151 GB`) while maintaining held-out quality parity (`81.9%` WikiText-2 top-1 vs BF16 golden). Legacy `pack-g64-gptq-v3` remains available via `dankattn pull hy4-v3` (and an optional `u4-engram` revision on Hugging Face).
- **`u4_g32` 4-bit Engram table (`h4_eng_rows_q4`, `hy4/convert/build_engram.py`):** compresses `engram.bin` from `59.0 GiB` (`96 B/row` FP8) to **`36.9 GiB` (`60 B/row` affine `u4_g32`)**.
- **Block-128 Walsh-Hadamard rotation (`h4_wht128_pf`, `h4_wht128_dec`) & mixed-bit `moe.dn`:** adds `moe_dn_had: 128` support (`W4g128` + `W6g128@[1, 40, 41]`) with vectorized `bfloat4` loads/stores, 4-SG prefill TGs (`3.08 ms`/1024-row chunk), and `ctz(mask)` active-row decode dispatch (0-diff bit-exact).
- **Engram I/O acceleration (`engine/hy4/hy4.mm`):**
  - Lock-free 128 MB userspace direct-mapped row cache (`HY4_ENG_CACHE_MB`, seqlock per 68 B slot, ~1M hot `u4_g32` rows).
  - Sorted deduplicated `pread`s (`HY4_ENG_DEDUP=1`) eliminating redundant reads within a batch/chunk.
  - Dedicated `F_NOCACHE` prefill file descriptor (`HY4_ENG_PF_NOCACHE=1`) preventing long-context prefill Engram reads from evicting model weight pages.
  - Measured on M5 Ultra (`pack-v4-h128-w6`): **+3.2% 64k prefill** (`1,651 → 1,704 tok/s` wall) and **−31% to −64% Engram wait** in speculative decode (`0.58–0.95 ms/round`).
- **Speculation, kernels & serving:** fan-out-aware adaptive draft length (`HY4_DRAFT_ADAPT`), auto `HY4_SPLIT_TAB` per Apple Silicon tier, fused `EPI=5` & `mm1_h4_swiglu` in `engine/kernels/mm2.metal`, and `--idle-unload` support in `server/hy4_server.py` / `bin/dankattn`.

### Security & Hardening (All Engines & Servers)
- **HTTP front-end security (`server/engine_frontend.py` + all servers):** `Host`/`Origin` allowlist (`--allow-host` / `DANKATTN_ALLOW_HOSTS`), 128 MB request body cap (`413 Payload Too Large`), filtered `chat_template_kwargs`, and constant-time bearer auth.
- **Vision SSRF protection (`server/vision_preproc.py`):** public-IP-only remote image fetching with redirect, byte-size, and per-request image count limits.
- **Manifest & runtime safety (`engine/mtl.h`, `hy4`, `inkling2`, `ernie55`, `model.mm`):** manifest filename path-traversal validation (`check_pack_filename`), bounds-checked buffer offsets (`checked_file`), `0700` prefix cache directories, `posix_spawn` for `curl` in `ernie55.mm`, vendored `server/ernie_chat/` (`--trust-remote-code` opt-in), and `torch.load(..., weights_only=True)` across conversion scripts.

## v2026.11.23
- Added one-liner `install.sh` and unified `dankattn` CLI (`pull`, `run`, `serve`, `doctor`, `update`).
- Added ERNIE-5.5-Flash-Preview (`engine/ernie55/`) and Inkling-2-Mini (`engine/inkling2/`) Metal 4 engines.

## v2026.11.22
- Added `update.sh` one-liner updater and README Updating section.
