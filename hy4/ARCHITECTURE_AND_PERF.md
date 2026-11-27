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
- **2026-11-26 (`hy4-dfly2-b6` branch, staged in `~/opus-hy4-dev`, not yet promoted to `:8341`) — `AngelSlim/Hy4-flash-DFly2-B6` (3-layer LatentMoE + `AttentiveCorrection`)**:
  - **Architecture & packer (`hy4/convert/pack_dfly2_b6.py`)**: Overlay pack (`pack-dfly2b6-00.bin`, **1.935 GB** incl. `lm_head_w4`; W6g128 dense/attn/correction + W4g64/W4g128 `H128` 3×128 LatentMoE experts, drops old `pack-11.bin`). PyTorch f32 reference (`hy4/ref/dfly2_b6_ref.py`) confirmed `causal=True` intra-block attention (matching `h4_dsel`) and that `AttentiveCorrection` uses slot-0 anchor block key (`block_gate_bias` non-zero at cols 1..5, `block_rel_bias` at 0..5). Watch out: `h4_attn_in` writes `ac_blk` permuted via `lat_logic(i)`, so `h4_ac_block_attn` must index via `lat_phys(d)`.
  - **Key memory lever (`HY4_DFLY_UNWIRED`, default `-1` = auto-on for MoE drafters)**: Keeping `pack-dfly2b6-00.bin` mmap'd in page cache instead of wired in `MTLResidencySet` dropped wired residency `88.5 → 86.6 GB` (**−1.9 GB**, 0.8 GB *less* than the 5L dense pack's `87.4 GB`), eliminated the ~4% wired-limit plain-decode penalty (`49.5 → 51.8 tok/s`), and cut Engram SSD wait ~3× (`1.34–1.45 → 0.35–0.38 ms/round`) by leaving page-cache headroom — with zero draft slowdown and zero precision loss.
  - **Draft kernel wins**: `h4_ac_gate_nsel` + `ac_nh_gate` gating skipped dead tail slots in `d_ac_attn` (`0.432 → 0.300 ms`, −30.6%), `d_ac_kb`/`d_ac_vb` (−26% to −30%), and `d_ac_merge` (−22.5%); chained `h4_d_post_add_ln` MoE post-norm fusion cut `d_rms` (`0.143 → 0.094 ms`, −34.3%); concurrent `d_ac_bg` overlap. Unprofiled GPU draft: `3.459 →` **`3.220 ms`** (parity with 5L dense `3.216 ms`).
  - **Batched multi-lane `draft_lanes()`** (`h4_ac_blk0_in`, `h4_ac_in_lanes`, `h4_ac_sel_lanes`, `h4_draft_corr2`): `0/0` token diffs vs single-lane across all `jtest` suites.
  - **Clean 100% bit-exact A/B vs 5L dense (`pack-v4-h128-w6`, same `HY4_QOS=1` binary, M5 Ultra, warm Engram cache, `vtest`/`ntest`/`ctest`/`jtest` = `0` diff, `400/400` exact on all `spec` suites)**:
    - Wired RAM: `87.4 →` **`86.6 GB`** (`−0.8 GB`); `bench 12k` 1/5/8/16 rows: `19.83/28.25/33.73/44.86 ms` vs `19.90/28.33/33.81/45.01 ms` (parity).
    - Single-lane `spec` (400 tok): `chat_code_high` (`T=0.9`) `91.2 →` **`95.3 tok/s`** (**+4.5%**); `chat_code_no_think` (greedy) `118.3 →` **`119.9 tok/s`** (**+1.4%**); `chat_doc_high` (`T=0.9`) `77.9 → 77.7` (tie); `edit_spec` `357.2 → 350.1` (`−2.0%`); `qa_doc` `162.2 → 152.5` (`−6.0%`); `ids12k` `156.4 → 133.7` (`−14.5%`; 5L dense uses `block=8` vs B6's trained `block=6`).
    - 4-lane `jtest` (200 tok/lane, `0` diff): greedy `210.1 →` **`215.0 tok/s`** (**+2.3%**, `1.44×`); `T=0.9` `205.9 →` **`208.9 tok/s`** (**+1.5%**, `1.40×`).
  - **`HY4_QOS=1` default (`engine/hy4/main.mm`)**: `QOS_CLASS_USER_INTERACTIVE` on the engine thread is now default-on (100% bit-exact, neutral on warm steady-state ABBA `208.3` vs `208.4 tok/s`, guards against 60 Hz frame-boundary scheduling stalls under load). `HY4_DFLY_UNWIRED=1` on 5L dense (`pack-v4-h128-w6`) was rejected (caused 16–25 ms Metal re-wiring stalls on `draft queue`); stays auto-on for MoE drafters only.
  - **2026-11-26 afternoon correction — `HY4_DFLY_UNWIRED` stalls the draft at production state (do not promote B6 with it on).** Per-round host timeline (`spec` trace: verify/draft encode, commit→GPU-start queue, GPU-end→wake) shows every host stall is **draft command-buffer queue latency of 16.9 or 33 ms** (one or two 60 Hz frames), encode and wake 0.1 ms: Metal re-making the unwired drafter's pages resident at submit. `quick` suites × 6 each, `HY4_CAP=32768`: unwired ~40% of runs stalled, wired 0/12. At **`HY4_CAP=524288`** (5.15 GB state): unwired **53 stalls in 12/12 runs, `code_high` T0.9 58.0 / `code_nothink` 75.1 tok/s**; wired 0 stalls, **97.4 / 122.6**; production 5L pack 91.5 / 118.2. One wired run hit the wired-limit cliff instead (verify 24.5 → 34.4 ms GPU), and production adds 3 × 32k side lanes (6.49 GB state), so wired B6 needs ~1.1 GB freed elsewhere (or a higher `iogpu.wired_limit_mb`, which costs Engram page cache) before it is safe. Not host QoS: `HY4_QOS=1` (user-interactive engine thread) and `taskpolicy -t 0 -l 0` did not change the stall rate.
  - **Draft length is not the lever (offline replay, `HY4_SPEC_DUMP` with cut 0 / adapt 0 / lookup 0, 3,690 rounds, 7 prompts × greedy + T0.9 s1/s2, all exact).** Conditional acceptance by draft position tracks the drafter's mean prob within a few points at every position and both temperatures (calibrated; no port bug). Leave-one-prompt-out calibrated policies on prob, Gumbel margin (`h4_pick_p` now writes `dmarg`), prob+margin, position+margin: all within −1.5…+0.1% (geomean) of the live cut-0.2 + `select_draft_k` policy; perfect-foresight oracle +10%. Joint knobs (`HY4_DRAFT_PSCALE` 1.5/2.0, `HY4_DRAFT_ADAPT=0`) are within ±2% of base on four `jtest` mixes once host-stalled runs are excluded; the earlier "+9% at 4 lanes" was a stalled base run.
  - **Where the verify step's time goes (12k, `HY4_SKIP=<op>` unprofiled ablation, 1 row = 19.85 ms):** ~9.2 GB of weights a 1-row step (6.7 dense/attn + 2.45 experts) ≈ 9.3 ms at ~1 TB/s, so the step runs at ~46% of roofline (~58% at 4 rows). `mm.gate` 1.72 ms (~1,070 GB/s) and `mm.o` 2.07 (~890) are near peak; `moe_gu` 3.72 ms (~440 GB/s) and `moe_dn` 1.55 (~520) are the biggest gap, then latency-bound `attn` 2.0, `hc_attn` + `hc_ffn` 2.1, `idx_topk` 0.87. 1 → 6 rows adds 10.1 ms, 74% of it `moe_gu` + `moe_dn`. `HY4_PF` / `HY4_PF_GRP` (mm2 k-steps in flight) 2 and 4 are slower at every row count (bits unchanged, `vtest` 0 diff): keep PF 1. Skip ablations at 6 rows are unreliable (stale activations collapse routing).
  - **Harness:** `hy4/dfly_bench.sh LABEL [BIN] [PACK]` (SUITES quick/core/full/joint; one JSONL line per suite with exactness, tok/round, draft/verify ms, host p50/p90/max, stall count, `tps_clean`), `hy4/dfly_sweep.sh CONFIGS` (one `:8341` stop/restart around many configs; always restarts from `~/opus-hy4`, never the dev tree). `spec` prints accepted/candidate histograms and the worst rounds' host timeline.

### What Was Tested & Rejected (Do Not Retry)
1. **Runtime `if (a.ldx == 1)` branch inside `mmpg`'s inner K-loop**: Regressed MoE prefill (`moe_gu` +17.6 ms, `moe_dn` +5.8 ms) due to inner-loop branch/register pressure. If optimizing `attn.kb`/`attn.vb` prefill later, use a separate compile-time `#ifdef` kernel, never a runtime branch inside `mmpg`.
2. **Fusing `ynorm_pending` into `h4_hc_1pass` in prefill**: Added register pressure + barriers inside `h4_hc_1pass` (+0.87 ms net slower than keeping separate `h4_rms` + `h4_hc_1pass`).
3. **32-head `h4_attn_mma32` (2×16 head blocks per TG)**: Exceeded the M5 register sweet spot (~144 live floats/thread) and spilled to thread-local memory (`attn` 239 → 343 ms).
4. **`MTLHazardTrackingModeUntracked` on `MTLHeapTypePlacement`**: Caused a +67 ms/chunk (+18–24 ms/op) prefill penalty on Apple Silicon Metal. Always use `MTLHazardTrackingModeTracked` on `scratch_heap`.
5. **Vectorized `bfloat4` `LOAD_X` in `mm2` decode**: Wash / +0.07 ms noise vs scalar `bfloat` loads.
6. **Inner-loop `kv8` scale hoisting + conditional barrier in `h4_attn_mma`**: Added register/scheduling pressure inside the hot `QK^T` loop and regressed 12k prefill (`1,704 → 1,461 tok/s`) and `d_attn` (`0.497 → 0.548 ms`).
7. **Prefill SWA query-row transpose (`h4_attn_mma_swa_pf`, 16 query rows × 1 head on MMA M-axis)**: Same TG count as baseline (`64×64 = 4,096` vs `4×1024 = 4,096`) while adding +1 tile iteration (`527/32 = 17` vs `16`) and per-element window/causal masking in the inner loop; the 590 KB sliding window is already L2/SLC-resident, so `attn` regressed `165.1 → 171.6 ms` (12k prefill `1,700 → 1,494 tok/s`).
8. **Non-bit-exact speed levers tested & rejected on 2026-11-26 (`pack-v4-h128-w6` & `pack-dfly2b6`, M5 Ultra; not committed because all reduce Top-1 vs Base below 100%)**:
   - **`DANKATTN_FAST_MATH=1` (`MTLMathModeFast` + `MTLMathFloatingPointFunctionsFast`)**: Only −3.9% GPU on `calib2048` (`5,504 → 5,288 ms`), +2.0% prefill (`1,768 → 1,803 tok/s`), −0.2 to −0.45 ms on decode/verify — but caused real precision damage: `max KL = 8.19` alone (`top-1 = 94.97%`, `ΔNLL = +0.0096`), and combined with `HY4_M_SPLITS=1` pushed `calib2048` true-token NLL `2.6079 → 2.6383` (`ΔNLL = +0.0304`, PPL `13.57 → 13.99`, +3.1%). Do **not** enable fast-math.
   - **3-bucket M-adaptive attention splits (`HY4_M_SPLITS=1`: `M=1 → 8/16`, `M=2..8 → 4/8`, `M=9..16 → 2/4`, `direct=1` when `splits=1`)**: Under `MTLMathModeSafe`, cut 8/16-row verify by −3.4% / −6.8% (`33.77 → 32.63`, `44.89 → 41.95 ms` at 12k; `42.08 → 39.22 ms` at 64k), `edit_spec` `359.2 → 385.4 tok/s` (+7.3%), `qa_doc` `162.6 → 174.6` (+7.4%), and `calib2048` GPU `5,504 → 5,023 ms` (−8.7%) with small NLL drift (`2.6079 → 2.6140`, `ΔNLL = +0.0061`), but 1-ULP reduction-order flips dropped `calib2048` Top-1 vs Base to `94.53%` (`mean KL = 0.0228`, `median KL = 0.0047`).
   - **Verify matmul split-K reduction (`HY4_SPLITS_V`, lowering dense `ksplit` from `6..8` to `1..4`)**: Regressed 5/8-row verify by +1.6 to +2.9 ms (`27.78 → 29.39..30.64 ms`). Existing dense `stab` is already optimal for M5 Ultra.
   - **`HY4_MM1=1` (M=1 SIMD matvec on main-model decode)**: Cut 1-row decode `19.85 → 18.74 ms` (−5.6%), `vtest` `mean KL = 0.00176`, `top-1 = 16/16` (16 rows), but breaks M=1↔M=16 bit-identity (`1,933,312` differing logits).
   - **Tolerant speculative acceptance (`HY4_TOL=0.25 / 0.50 / 0.75` with stop + `min_p`/`top_p`/`top_k` guards)**: On `pack-dfly2b6` (3 seeds × 400 tok, `T=0.9`), `tol=0.50` raised `chat_code_high` `91.4 → 109.8 tok/s` (+20.1%) and `chat_doc_high` `71.8 → 83.9 tok/s` (+16.9%); on `pack-v4-h128-w6`, 4-lane `jtest` hit `232.2 tok/s` greedy (+27.4%) and `220.1 tok/s` at `T=0.9` (+24.0%). Past `0.50`, `tol=0.75` plateaued on code (`108.8 tok/s`) and `HY4_DRAFT_PSCALE=1.15 / HY4_DRAFT_CUT=0.12` regressed doc (`−0.9 tok/s`). Rejected as default because it intentionally substitutes near-tie non-matching tokens (not exact). Also noted: in 1:1 published-suite A/B, existing `pack-v4-h128-w6` beat `pack-dfly2b6` on 5/6 suites (`chat_code_no_think`, `edit_spec`, `qa_doc`, `jtest` greedy & `T=0.9`).

---

## 4. Remaining Future Ideas (Backlog)
- **Prefill sliding-window `Bq × Bk` query-row tiling**: Only if a register-neutral design can tile query rows without spilling (remember 32-head `h4_attn_mma32` spilled!).
- **Engine C++ core extraction (`engine/runtime.h`, `engine/proto.h`)**: Deduplicate `Rec`, `Lin`/`Raw`, residency timer, and the `OPUS`/`OPCX` binary protocol between Flash-Next (`engine/model.mm`, `engine/serve.inc`) and Hy4 (`engine/hy4/`).
