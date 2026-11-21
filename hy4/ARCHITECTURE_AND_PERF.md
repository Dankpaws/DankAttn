# DankAttn (Hy4) Architecture, Benchmarks & Optimization Log

> Last updated: 2026-11-20 (HEAD after Wave 1–3 latency/multi-lane/indexer wins, M5 Ultra, `pack-g64-gptq-v3`, live on `:8341`).

## 1. Current Architecture & Live Defaults (`hy4/serve_hy4.sh`)

- **Model / Pack**: `pack-g64-gptq-v3` (42 layers: 30 sliding-window MLA with `nh=64, window=512` in bf16; 12 sparse DSA with `nh=32, selmax=2048` in `kv8`; 4 Engram layers; 5-layer DFly2 drafter).
- **Serving Topology (`:8341`)**: Continuous multi-lane batching (`engine/hy4/serve.inc`):
  - Slot `0`: `HY4_CAP=524288` main conversation slot backed by SSD `PrefixCache` (`HY4_PCACHE`, 40 GB budget, 15 GB free-disk reserve).
  - Side lanes: `HY4_SIDE_LANES=3` × `HY4_SIDE_CAP=32768` (4 total concurrent slots; **6.49 GB** total wired GPU state).
  - **1 active decoding lane**: full DFly2 (`HY4_DBLOCK=8`, `HY4_DRAFT_CUT=0.2`) + zero-allocation flat 3-gram context-lookup (`engine/lookup.h`) speculation.
  - **2+ active decoding lanes**: joint speculation (`Spec::joint` in `engine/hy4/spec.inc`, `5fa46ae`): each lane proposes DFly2 + lookup drafts within an even share of 16 rows, one `Model::step_lanes()` (`LANES` Metal kernels reading per-layer `LaneState { lat, tokring, bk }` GPU pointer tables via `MTLResidencySet slot_rs`) verifies every lane's rows, each lane accepts its own prefix. Lane steps write the DFly2 target layers' selections to `lsel`, so every lane drafts from its row of the last step; `serve` drops a lane's drafter source once anything else stepped (`step_gen`). `HY4_JS=0`: plain one-row lanes.
  - Prefill interleaves one `MR=1024` chunk per loop iteration and checks `q.cancel` between chunks.
- **Sampling (`engine/kernels/hy4.metal`)**:
  - `trunc_floor()` shared by `h4_sample`, `h4_pick_p` (DFly2 drafter), and `h4_sample_lanes`: 4-pass 8-bit MSB radix select for `top_k`, 1-pass 256-bin probability histogram + 8-pass bracketed search for `top_p`, plus `min_p` (`HY4_MIN_P=0.02`) and counter-hashed Gumbel-max.
- **Memory & Runtime**:
  - `mtl::Ctx` (`engine/mtl.h`) caches compiled `id<MTLLibrary>` by `(src_hash, defs)` (~3× faster cold startup).
  - Phase-aliased `MTLHeapTypePlacement` scratch arena (`MTLHazardTrackingModeTracked`) aliases non-overlapping Engram, Attention (`Yg`, `SlotA: Yq|qlat|yv`, `SlotB: scores+mark|ofrag|gfrag`, `part`), and FFN/MoE phases, plus 16-row DFly2 `t.lsel` sizing (**-560 MB** wired state at `512k`).
  - `hy4::Rec` uses `MTLDispatchTypeConcurrent` (`HY4_CONC=0` disables) with explicit buffer barriers, overlapping `attn_in ∥ gate`, `qb ∥ idx_pool`, `kb ∥ idx_score/topk`, and draft `d_attn_in ∥ d_gate`, `d_kb ∥ d_dsel`.
  - Shared host headers: `engine/mm_args.h` (`MmArgs`, `static_assert(sizeof == 92)`) and `engine/lookup.h`.
- **Front End (`server/hy4_server.py`)**:
  - Offloads `splice()` tokenization/bisection to `asyncio.to_thread`, tracks tool calls in `O(1)` (`in_tool_call`), supports user `stop` sequences (single-token GPU stops + streaming/non-streaming string stops), and reports live lane count on `/v1/load`.

---

## 2. Verified Benchmarks (M5 Ultra, `pack-g64-gptq-v3`, 12,272-token context)

| Metric | Baseline (`af65b1e`) | Current (`069fc37`) | Delta |
| :--- | ---: | ---: | ---: |
| **Prefill 12,272 tok (GPU)** | 9,192 ms (**1,335 tok/s**) | 7,273 ms (**1,687 tok/s**) | **+26.4%** |
| **Prefill 12,272 tok (wall)** | 9,244 ms (**1,328 tok/s**) | 7,303 ms (**1,680 tok/s**) | **+26.5%** |
| **Decode 1 row (GPU)** | 20.10 ms | **19.46 ms** | **-3.2%** |
| **Verify 5 rows (GPU)** | 29.04 ms | **28.13 ms** | **-3.1%** |
| **Verify 8 rows (GPU)** | 34.85 ms | **33.72 ms** | **-3.2%** |
| **Verify 16 rows (GPU)** | 46.96 ms | **45.27 ms** | **-3.6%** |
| **DFly2 draft (`dprof` unprofiled GPU)** | 3.686 ms | **3.415 ms** | **-7.4%** |
| **Prefill `attn` (`pprof` 1024-row chunk)** | 238.95 ms | **165.24 ms** | **-30.8%** |
| **Prefill `moe_gu` + `moe_dn` (`pprof`)** | 216.87 ms | **203.17 ms** | **-6.3%** |
| **Prefill `eng_mix` (`pprof`)** | 2.44 ms | **0.95 ms** | **-61.0%** |
| **Prefill `idx_score` (`pprof`)** | 5.86 ms | **4.88 ms** | **-16.7%** |
| **Wired state (`cap=32k`)** | 1.53 GB | **1.00 GB** | **-530 MB** |
| **Wired state (`cap=512k + 16k` side)** | 6.02 GB | **5.46 GB** | **-560 MB** |
| **Wired state (`cap=512k + 3×32k` live)** | — | **6.49 GB** | 4 concurrent lanes |
| **Cold shader-compile startup** | ~2.7 s | **0.8–1.0 s** | **~3× faster** |

### Multi-Lane Batched Decode (`btest` at 12,272-token context/lane, `ntest` 2/4/8/16 bit-identical)
- **1 lane**: `19.46 ms/step` (**51.4 tok/s** GPU)
- **2 lanes**: `22.01 ms/step` (**90.9 tok/s** GPU, 1.77×)
- **4 lanes**: `25.61 ms/step` (**156.2 tok/s** GPU, 3.05×)
- **8 lanes**: `33.26 ms/step` (**240.5 tok/s** GPU, 4.71×)
- **16 lanes**: `42.52 ms/step` (**376.3 tok/s** GPU, 7.38×)

### Speculation (`spec`, 400 tokens each, 100% exact greedy match)
- `ids12k`: **127.5 tok/s** wall (400/400)
- `chat_code_no_think`: **116.7 tok/s** wall (400/400)
- `edit_spec`: **355.8 tok/s** wall (400/400)

---

## 3. Optimization History & Dead Ends (Do Not Retread)

### What Won (Kept)
- **`3cd5610`**: Indirect-dispatch MoE prefill (`h4_plan_lists` compact `(slot, tile_y)` worklist + `dispatchThreadgroupsWithIndirectBuffer` eliminating ~93% empty MoE TGs: `1,335 → 1,504 tok/s`), `MTLLibrary` caching in `mtl.h`, 1-barrier `h4_eng_mix`, `sh_y[32]` race fix in `h4_hc_1pass`, RoPE `inv_freq` hoisting.
- **`aea50a5`**: `MTLDispatchTypeConcurrent` in `hy4::Rec` + single-bf16 `h4_idx_score_mma` (dropped `ql` pass).
- **`b6ffcb7`**: DFly2 `EPI=3` fused SwiGLU on `d_mlp_gu`/`d_hc_gu` + `dgate` early-exit gating on `d_hc_gu`/`d_hc_dn`.
- **`a5f7eb8` & `6b1977f`**: Transposed `Sp[8][32][16] → Sp[8][16][32]` in `h4_attn_mma` (and `Sp[4][32][16] → Sp[4][16][32]` in `attn.metal` for main `opus`), fixing 16-way threadgroup memory bank conflicts, plus compile-time `ATT_SLIDING`/`ATT_KV8` specialization (`attn` **-31%** in prefill, **-20%** in decode/draft).
- **`f169ab4`**: Radix-histogram `trunc_floor()` shared by `h4_sample` and `h4_pick_p`, prefill cancellation, `save_prefix` returning `bytes`/`floor`, dynamic side-slot check, async `splice()`, `O(1)` tool-call buffering, and user `stop` sequences.
- **`47aa57c`**: `MTLHeap` (`Tracked`) scratch arena (-560 MB), 16-row `t.lsel`, flat `engine/lookup.h`, shared `engine/mm_args.h`.
- **`047cf62`**: Pruned dead `ATT_NOLO`/`ATT_NOV` ifdefs, `HY4_DUMP_CONF`, duplicate `hy4/mk.sh`, unused `ngram.mm` link, `persist.metal`, and legacy `opus_{iter,next}_serve.sh`.
- **`6bc4b09` & `069fc37`**: Multi-lane `step_lanes()` core + `ntest`/`btest` + Phase 2 continuous multi-lane batching in `serve.inc`.
- **`5fa46ae`**: Joint speculation across lanes + `jtest` (0 differing tokens vs single-lane decoding, greedy and T 0.9). Aggregate wall, 300 tok/lane: 2 chat lanes 92.1 → 121.2 tok/s, 2 chat + edit 118.9 → 160.7, 3 chat + edit 145.2 → 178.7, 4 chat 150.7 → 153.6. Single-lane `spec` unchanged (126 / 116 / 353).
- **`058fe20`**: Per-slot saved DFly2 source state (`df_xf`, `df_lsel`, `df_lnsel`, ~52 KB/slot — eliminates `step_gen` invalidation during interleaved prefill), batched multi-lane DFly2 drafting (`Model::draft_lanes()`), unified `engram_read_rows` / single 128-thread `IoPool` (-128 idle threads), and `slot_rs` residency refresh. `jtest` 300 tok/lane: 2 chat 121.2 → **127.6 tok/s** (1.39×), 4 chat greedy 153.6 → **192.5 tok/s** (+25.3%, 1.27×), 4 chat T 0.9 152.2 → **198.5 tok/s** (+30.4%, 1.31×), 0 diff.
- **`d5bc0ae`**: Parallel 256-thread `h4_idx_topk` (`idx_topk` 10.61 → **4.00 ms**, -62.3%), 2-pass coarse+fine 256-bin `trunc_floor` for `top_p` (9 → 2 vocab scans), `mmpg CONTIG_X` + `CH=8` for prefill `kb`/`vb`, and decode dense `mlp_gu` `EPI=3` (pruned `h4_swiglu`). 12k prefill: **1,704 tok/s** GPU (`1,698` wall); 1/5/8/16-row decode: **19.16 / 27.77 / 33.34 / 44.86 ms**; `vtest` & `ntest 4` bit-identical.
- **`71d37e3`**: Fused DFly2 `h4_d_post_add_ln` (`post_norm + add + post_ln`) and gated single-lane `h4_draft_hc_in` / `h4_add_frag` (removed dead `dc` buffer): `dprof` unprofiled GPU 3.415 → **3.216 ms** (-5.8%).
- **Wave 1+2 (`64ec048`)**: Single-lane anchor emit before `sp.round()` (-30 ms TTFT), dynamic prefill chunk cap when decode lanes are active (`HY4_PF_DEC_CHUNK=256`, bounds live-stream ITL stalls), async non-blocking `PrefixCache::load` (`load_prefix_into` + `apply_prefix_load`), cached RoPE `inv_freq`, non-blocking `save_ckpt()`, multi-lane Engram anchor prefetching behind `draft_lanes()`, and dynamic joint-speculation row budgeting / early-cut redistribution (`jtest` 4 chat greedy: 192.5 → **199.4 tok/s** wall, 1.30×, 0 diff).
- **Wave 3 (`h4_idx_pool` + `h4_idx_topk`)**: Swapped `h4_idx_pool` loop order for 4× `kvp`/`kgp` weight reuse (`idx_pool` 4.12 → **3.87 ms**, -6.0%), and added TG-memory candidate compaction (passes 2–3 zero DRAM reads) + zero-tie thread skip in `h4_idx_topk` (`idx_topk` 4.01 → **3.88 ms** at 12k, fewer DRAM passes at 64k–512k). Decode/verify at 12k: **19.08 / 27.61 / 33.26 / 44.76 ms** GPU; `vtest` & `ntest 4` bit-identical.

### What Was Tested & Rejected (Do Not Retry)
1. **Runtime `if (a.ldx == 1)` branch inside `mmpg`'s inner K-loop**: Regressed MoE prefill (`moe_gu` +17.6 ms, `moe_dn` +5.8 ms) due to inner-loop branch/register pressure. If optimizing `attn.kb`/`attn.vb` prefill later, use a separate compile-time `#ifdef` kernel, never a runtime branch inside `mmpg`.
2. **Fusing `ynorm_pending` into `h4_hc_1pass` in prefill**: Added register pressure + barriers inside `h4_hc_1pass` (+0.87 ms net slower than keeping separate `h4_rms` + `h4_hc_1pass`).
3. **32-head `h4_attn_mma32` (2×16 head blocks per TG)**: Exceeded the M5 register sweet spot (~144 live floats/thread) and spilled to thread-local memory (`attn` 239 → 343 ms).
4. **`MTLHazardTrackingModeUntracked` on `MTLHeapTypePlacement`**: Caused a +67 ms/chunk (+18–24 ms/op) prefill penalty on Apple Silicon Metal. Always use `MTLHazardTrackingModeTracked` on `scratch_heap`.
5. **Vectorized `bfloat4` `LOAD_X` in `mm2` decode**: Wash / +0.07 ms noise vs scalar `bfloat` loads.
6. **Inner-loop `kv8` scale hoisting + conditional barrier in `h4_attn_mma`**: Added register/scheduling pressure inside the hot `QK^T` loop and regressed 12k prefill (`1,704 → 1,461 tok/s`) and `d_attn` (`0.497 → 0.548 ms`).
7. **Prefill SWA query-row transpose (`h4_attn_mma_swa_pf`, 16 query rows × 1 head on MMA M-axis)**: Same TG count as baseline (`64×64 = 4,096` vs `4×1024 = 4,096`) while adding +1 tile iteration (`527/32 = 17` vs `16`) and per-element window/causal masking in the inner loop; the 590 KB sliding window is already L2/SLC-resident, so `attn` regressed `165.1 → 171.6 ms` (12k prefill `1,700 → 1,494 tok/s`).

---

## 4. Remaining Future Ideas (Backlog)
- **Prefill sliding-window `Bq × Bk` query-row tiling**: Only if a register-neutral design can tile query rows without spilling (remember 32-head `h4_attn_mma32` spilled!).
- **Engine C++ core extraction (`engine/runtime.h`, `engine/proto.h`)**: Deduplicate `Rec`, `Lin`/`Raw`, residency timer, and the `OPUS`/`OPCX` binary protocol between Flash-Next (`engine/model.mm`, `engine/serve.inc`) and Hy4 (`engine/hy4/`).
