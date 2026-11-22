# DankAttn (Inkling-2-Mini) Architecture, Benchmarks & Engineering Log

> Last updated: 2026-11-22 (branch `inkling2-decode-fusion`, HEAD `a501d5d`: decode tuning on top of `inkling2-lanes-fpr` `9bcfe2b`, see *Decode Tuning*; earlier tables measured at the commits they name; benchmarked on Apple M5 Ultra Mac Studio 96 GB and Base M5 MacBook Pro 32 GB).

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
  - **GPU state (`cap=32768`, `kv8=1`, `dspark=1`)**: **`1.33 GB`** → **`21.24 GB` total**.
  - **GPU state (`cap=131072`, `kv8=1`, `dspark=1`)**: **`3.37 GB`** → **`23.28 GB` total** (fits inside a 32 GB MacBook Pro). Both grew ~0.1–0.4 GB with `MP=512` scratch (Wave 9).

---

## 2. Metal 4 Engine Highlights (`engine/inkling2/`, `engine/kernels/inkling2.metal`)

- **Register-direct NVFP4 / MXFP8 / W8 matmuls (`engine/kernels/mm2.metal`)**: Shared cooperative `mpp::tensor_ops::matmul2d` kernels with fused `SoftcappedSwiGLU` (`SWIGLU_SOFTCAP_G=4.0`, `SWIGLU_SOFTCAP_U=25.0`, `EPI=3`).
- **Two-regime execution (`M <= 16` vs. `M <= 512`)**:
  - **Decode & Speculative Verify (`M <= 16`)**: `mm2` (two-level split-K) + SIMD top-8 MoE routing (`ink_moe_route`), bit-exact across `M=1..16` (`vtest` `0.000e+00`).
  - **Chunked Prefill (`MP = 512`)**: `mmp` (`MMP_CH=8, MMP_NF=2, MMP_FAST=1`) for dense W8 attention projections + parallel per-row routing (`ink_moe_topk_pf`), 16-row-aligned compact indirect worklist (`ink_moe_plan_pf`), `mmpg` (`MMP_CH=4, MMP_NF=2, MMP_FAST=1`) for NVFP4/MXFP8 MoE, and gather-accumulate (`ink_moe_accum_pf`). Attention is a Metal 4 `matmul2d` FlashAttention kernel (`ink_attn_pf_mma`, local and global layers, interior-tile fast path) fed by `matmul2d` relative logits (`ink_q_rel_prep` + `ink_q_rel_mma`); message gates and the MoE router also run as `matmul2d` in prefill. Only the final row runs `lm_head` during prefill.
- **Concurrent GPU dispatch & exact rollback**:
  - `MTLDispatchTypeConcurrent` overlaps `qgrs || k_proj || v_proj`, `q_rel || k_sconv || v_sconv`, and the 3-stage parallel cross-pass message gate (`ink_mg_prep` → `ink_mg_down` → `ink_mg_apply`).
  - Barrier-free simdgroup split-K GQA attention (`ink_attn_split` + `ink_attn_reduce`).
  - Per-logical-layer `conv_hist` ring snapshots enable exact `sconv` state rollback on partial speculative acceptance (100% greedy-exact match vs. `M=1`).

---

## 3. Verified Benchmarks: M5 Ultra Mac Studio (96 GB) vs. Base M5 MacBook Pro (32 GB)

All runs use `pack-inkling2-mini-w8` (`19.91 GB` mapped + `1.33 GB` GPU state, `cap=32768`, `kv8=1`, `dspark=1`, 3 passes). Measured 2026-11-21 at `cf05d1f` (Studio macOS 27.0.1, MBP macOS 27.0); the `7ba4268` column is the pre-optimization baseline.

| Workload | M5 Ultra Mac Studio (96 GB) | Base M5 MacBook Pro (32 GB, 153 GB/s) | Studio / MBP | vs. `7ba4268` (Studio / MBP) |
| :--- | :---: | :---: | :---: | :---: |
| **`vtest` (`M=5` vs `M=1`)** | `0.000e+00` (**BIT-EXACT**) | `0.000e+00` (**BIT-EXACT**) | — | — |
| **Prefill check (`M=64` vs `M=1`)** | Exact next-token (`149884`) | Exact next-token (`149884`) | — | — |
| **Prefill 2,048 tok (`MP=512`)** | **4,607.0 tok/s GPU** (4,572.8 wall, 0.44 s) | **729.7 tok/s GPU** (728.7 wall, 2.81 s) | 6.3× | 2.86× / 2.32× |
| **Prefill 8,192 tok (`MP=512`)** | **4,335.3 tok/s GPU** (4,304.8 wall, 1.89 s) | **719.0 tok/s GPU** (718.2 wall, 11.39 s) | 6.0× | 3.41× / 3.04× |
| **Decode `M=1` @ `ctx=512`** | **82.3 tok/s GPU** (76.9 wall, 12.15 ms) | **25.8 tok/s GPU** (25.3 wall, 38.82 ms) | 3.2× | +8% / +7% |
| **Decode `M=1` @ `ctx=2,048`** | **79.2 tok/s GPU** (74.2 wall, 12.63 ms) | **24.8 tok/s GPU** (24.2 wall, 40.39 ms) | 3.2× | +8% / +6% |
| **Decode `M=1` @ `ctx=8,192`** | **76.7 tok/s GPU** (72.0 wall, 13.04 ms) | **24.5 tok/s GPU** (24.0 wall, 40.89 ms) | 3.1× | +11% / +9% |
| **Speculative (`K=7, cut=0.50, lk=1`)** | **204.9 tok/s wall** (2.66× vs `M=1` wall) | **46.3 tok/s wall** (1.83× vs `M=1` wall) | 4.4× | +22% / +12% |
| **Spec exactness (`code_prompt.bin`)** | **256/256 (100% EXACT)** | **256/256 (100% EXACT)** | — | — |

MBP prefill heat-soaks: the first run after idle reads 2–4% faster than steady state, so compare MBP variants interleaved, not back-to-back.

### Lanes & Joint Speculation (`0d84d10`, 2026-11-21)

Lanes are independent sequences with their own logical-layer KV rings, `sconv` state, rollback history, `pos`, and DSpark context; `use_lane(k)` swaps one into `Model::S`. `step_lanes` runs the stateless ops once over every lane's rows and dispatches only `q_rel`, K/V `sconv` + cache writes, decode attention split and the two post-`sconv`s per lane at row offsets (`ink_attn_reduce` runs once). The 3 loop passes can't share weight reads with each other: pass `p` builds its messages from pass `p-1`'s complete output (`b_prev_h` / `b_prev_blocks`), so lanes are the batching axis and "staggered passes" add nothing on top.

| Total tok/s | 1 lane | 2 | 4 | 8 | 16 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Studio `lbench` (ctx 2,048, GPU) | 79 | 133 | 223 | 340 | 501 |
| MBP `lbench` (ctx 2,048, GPU, `INK_CAP=8192`) | 25 | 37 | 49 | 65 | OOM |
| Studio `jspec` (wall) | 158 | 244 | 324 | — | — |
| MBP `jspec` (wall, `INK_CAP=8192`) | 45 | 57.5 | 65.8 | — | — |

Every `ltest` / `jspec` run is bit-exact per lane vs. the same sequence alone. At 16 lanes (Studio) the step is 32.2 ms vs. 22.6 ms for 16 rows of one sequence; ~8 ms of the gap is MoE fan-out (16 unrelated contexts route to more experts), per-lane dispatch ≤ ~1.7 ms. Per-lane state is 0.83 GB at `cap=32768` and 0.42 GB at 8,192 (MBP fits ~8 lanes at 8k). `jspec` prompts are doc/source slices (they draft worse than `code_prompt.bin`); per-lane DSpark drafting is sequential (4.5 ms/round at 4 lanes on the Studio, 18 ms on the MBP) and is the next thing to batch.

### Sampling & Serving (`c570a20`, `45031b4`, `ac82145`)

- **Sampling** (`ink_sample_rows`, ported from Hy4): Gumbel-max keyed by `(seed, input position)` after `min_p` / `top_k` / `top_p` truncation, over the softcapped logits. DSpark picks draw with the same keys and report the drawn token's probability for the `p_max` cut, so speculation stays exact at any temperature (`spec` at T=1.0: 199.5 tok/s, 4.06 tok/round, 256/256 equal to plain sampled decode). Greedy rows keep the argmax bits.
- **`serve`** (`engine/inkling2/serve.inc`): Hy4's OPUS/OPCX binary protocol, so `server/engine_frontend.py` drives it unchanged. Each loop prefills one chunk of one waiting request (256 rows while others decode), then one round over every decoding lane (lookup / DSpark drafts, one `step_lanes` verify, per-lane accept + rollback + DSpark commit right after the verify). A stop inside accepted drafts rolls the lane back to what was emitted. Prompt reuse: exact extension of a lane's history, or **rewind** to its last prefill end from saved `sconv` states (`save_conv` / `rewind`) when ≤ `MP + INK_RING_SLACK` tokens followed; clients that drop reasoning land there.
- **Front end** (`server/inkling2_server.py`, `server/inkling2_serve.sh`): parses message tokens by id (`<|content_thinking|>` → `reasoning_content`, `<|content_text|>` → `content`, `<|content_invoke_tool_json|>` → `tool_calls`); strips the transformers `{% generation %}` tag and honours `tojson(sort_keys, separators)`; temperature 1.0 and `min_p` 0.02 by default.

Studio, 4 lanes × 131,072 (`INK_RING_SLACK=4096`, 15.25 GB state), T=1.0, 2026-11-21: 27,193-token prompt prefill 3,262 tok/s; code effort low 178 tok/s decode (3.7 tok/round), effort high 141 tok/s; 4 concurrent chats 190 tok/s total; same-seed requests identical alone vs. concurrent (12/12); an OpenCode agent turn reused 16,913 / 20,025 prompt tokens (prefill 1.1 s vs. 4.7 s cold).

### End-to-End Quality vs. SGLang (`9bcfe2b` live server, 2026-11-21)

`sgl-eval` greedy through the OpenAI front end (8 lanes, `INK_CAP=131072`, side 16,384), same settings as the SGLang (CUDA, BF16 KV) runs on gpu-desktop; per-question pairing with `~/experiments/inkling2-mini/paired.py`. Reference is SGLang on the base checkpoint, except RULER (SGLang only ran the uncensored one).

| Eval | SGLang | DankAttn | Paired |
| :--- | :---: | :---: | :--- |
| MMLU 1,140 (effort none, 4,096 tok) | 85.96% | **86.14%** | 30 vs 32 discordant, z = +0.25; 91.39% both on 1,057 finished |
| GSM8K 1,319 | 93.40% | 92.87% | 23 vs 16, z = −1.12 |
| HumanEval 164 | 90.2% | 93.9% | 2 vs 8, p ≈ 0.11 |
| RULER2 32k / 64k / 128k (12 × 3) | 85.4 / 74.5 / 66.7% | 83.3 / 81.0 / 83.3% | 30 identical items: 82.5→90, 73→80, 60→80 |

W8 attention + `kv8` global KV + chunked prefill show no measurable quality loss. The 128k RULER gap is mostly SGLang truncating at the 4,096 cap (both-finished: 90 vs 95%); DankAttn generated 28% fewer tokens there, cause not isolated. n = 3 per RULER subtask. RULER 128k needs `INK_CAP=139264` (prompts average 127,878 tokens + 4,096 answer).

### Decode Tuning (`ac3b49a`, `a501d5d`, branch `inkling2-decode-fusion`, 2026-11-22)

Goal: close the gap to ddalcu's unreleased mlx-serve Inkling build (card: 118 tok/s plain decode on M5 Ultra). A token reads ~3.4 GB (experts 1.84, attention 0.93, `lm_head` 0.48 GB), a ~350 tok/s roofline at 1.2 TB/s; plain decode sat at 12.21 ms (~23% of peak), and Studio/MBP scaled 3.2x against 7.8x bandwidth, so the step is latency-bound, not bandwidth-bound.

**Method.** Per-op in-stream cost = extra time when one op group runs twice (`INK_DUP=<mask>`, idempotent groups); `INK_SKIP` (drop a group) is unreliable here because stale buffers change routing downstream. An empty dispatch + barrier phase costs ~1.7 µs (`INK_XBAR`). Clean profile at ctx 512, M=1, µs per layer visit (48/token): MoE gate/up 39, MoE down 22, pre-attn attn_res/msg-gate/rms 21.5, pre-MLP 21.3, router 20.7, q/k/v proj 20.6, attention 19.3, o_proj 18.9, q_rel + kv_conv 14.1, post-sconvs ~4 each; `lm_head` 0.45 ms/token. W8 matmuls with many threadgroups reach ~1.0–1.17 TB/s (`lm_head`, q/k/v); NVFP4 MoE runs at ~570 GB/s and is bound by unpack instruction count, small-N W8 (o_proj) and single-threadgroup kernels by latency.

**Kept** (each A/B'd interleaved on a clean GPU):

| Change | Effect | Exactness |
| :--- | :--- | :--- |
| Early command-buffer commits (`INK_EARLY`, 4 CBs per step) | GPU unchanged, wall 76.4 → 80.0 tok/s (ctx 512), 71.6 → 74.5 (8k) | bit-identical |
| `ink_moe_route_fast`: rank top-k + prefix slots, one phase for M ≤ 16 (`INK_ROUTE_FAST`) | −0.31 ms | bit-identical |
| Split table: attn.o 1 sgs × 12 ksplit, NVFP4 moe gu/dn 2 sgs (`INK_SPLIT_TAB`, `INK_SPLITS`); mm2 split-K partials/counters per `pslot` (qgrs ∥ k ∥ v need disjoint ones); prefill replays ksplit | −0.73 ms | reorder; vtest/jspec/ltest/forktest exact |
| One 32-column fragment per threadgroup (`INK_NFT`, default 1) | −0.08 ms | bit-identical |
| Attention key splits 12/24 → 24 local / 64 global (`INK_NSP_LOCAL/GLOBAL`, `ASPLITS` 64) | ctx 512 +0.05, 8k −0.39, 30k −1.18 ms | reorder |

**Rejected** (measured, then removed or left off): decode fusions accum + MLP post-sconv and attn post-sconv + pre-MLP attn_res (`INK_FUSE=1`, bit-identical, +0.2 ms: the fused single-TG kernels cost more than the ~4 µs phases they absorb); last-arrival phase merges for router scores + top-k (−0.04 ms) and attention split + reduce (+1.35 ms: device atomics for partials are expensive); `PF` 2/4/8 weight prefetch (+0.3 to +8 ms, register pressure); NVFP4 E2M1 from bits instead of the constant LUT (+1.9 ms) and scale slab via simd shuffles (+0.6 ms); more simdgroups per MoE threadgroup (sgs 8/12: +0.5–1.3 ms); q/k/v and `lm_head` split changes (±0.05); attn_res block loop with loads hoisted ahead of the simd sums (no change, the compiler already overlaps them); commits every k layers (no change); 96/128 global attention splits (the reduce gets longer).

**Result** (M5 Ultra; engine `bench`, `INK_CAP=32768`; server = `inkling2_server.py` production profile, bake-off `bench.py`):

| | Before (`d4b551e`) | After (`a501d5d`) |
| :--- | :---: | :---: |
| Engine M=1, ctx 512: GPU ms / wall tok/s | 12.21 / 76.4 | 11.20 / 86.8 |
| Engine M=1 GPU ms, ctx 8k / 30k | 13.11 / 13.55 | 11.66 / 12.37 |
| Server plain decode (`--no-spec`) | 77.4 tok/s | 87.0 tok/s |
| Server DSpark decode, code / prose | 117.2 / 105.1 | 136.1 / 114.9 |
| Server 4 / 8 streams, total | 255 / 342 | 266 / 361 |
| Server cold prefill 104k | 1,956 tok/s | 1,972 tok/s |

Quality: MMLU 1,140 86.23% vs 86.14% before (McNemar z = +0.13; 85.96% SGLang base). Teacher-forced NLL (`nll`, 16-row decode steps, two 8k-token texts) 1.88589 / 2.17662 nats vs 1.88344 / 2.17753; any summation reorder moves it by up to ~0.003 nats in either direction (adding noise to logits raises expected NLL), so these are within that floor. `jspec` acceptance rose 2.09 → 2.19 tok/lane-round on the A/B prompt.

**Verify (M = 8) check.** Split choices compared with `spec 256` look 3% apart, but that is content: changed numerics change the drafts, so the verified rows and their expert fan-out differ. On fixed real-text rows (`vbench2`, ctx 2048, rolled back each step) every MoE/o_proj split variant lands within 20.41–20.61 ms per M = 8 step (default 20.55; old sgs heuristic 20.60) while M = 1 keeps the table's lead (11.50 vs 12.10 ms), so the table stands for both. An 8-row verify routes ~45 distinct experts per layer, ~200 MB of NVFP4 weights per visit, ~7.4 ms per step at peak bandwidth for the experts alone: verify is fan-out bound.

**Head-to-head with mlx-serve 26.11.4-pre-release.1** (ddalcu's Inkling build, released 2026-11-22 05:27 UTC, commit `2fe9be83`; pack `ddalcu/Inkling-2-Mini-MLX-8bit`: release experts + 8-bit attention, embedding and head) on the same Studio, same bake-off `bench.py`, ctx 131,072: plain decode **117.1 / 115.9** tok/s code / prose (DankAttn `a501d5d` 87.0); with its drafter on 121.2 / 115.0, but its adaptive DSpark policy probes and then skips drafting on these prompts, so that is mostly plain decode (DankAttn DSpark 136.1 / 114.9, bit-exact against plain); 4 / 8 streams 219 / 302 tok/s (DankAttn 266 / 361); cold prefill 2k / 8k / 32k / 104k 1,671 / 3,008 / 3,189 / 2,068 tok/s (DankAttn 2,645 / 3,804 / 3,076 / 1,972). His card's 118 tok/s reproduces.

What his `src/plugins/inkling2` does differently (read at `2fe9be83`, MIT): one-to-few-row MoE reads the packed NVFP4 bank row-major in place (`ups.metal` = gate + up + soft-capped SwiGLU, `downs.metal` = down + weighted sum + the MLP short conv + residual for one row), an f32 dot per 16-value chunk with E2M1 and e4m3 decoded through half bit tricks (2^14 and 2^8 folded back once) and the scale applied once per 16 values; ~10 dependent launches a plain visit (`pre`, q/k/v, `prep`, attention, `oproj` with the gate folded in, `mid` = post-sconv + residual + depth attention + norm + router rows, `topk`, `ups`, `downs`); stream kernels up to 768 threads with each threadgroup redoing a row's sums. His notes state the same bound measured here: a one-row visit costs its dependent launches (5–7 µs each). His drafted verify is not bit-exact against plain decode (GPU proofs compare forwards of the same width).

Tried here from that: the in-place NVFP4 GEMV over mm2's *tiled* bank (`INK_MOE_GEMV`, `ink_moe_gu_gemv` / `ink_moe_dn_gemv`; lane l, word j, byte i, nibble h = column m0 + 16 (j & 1) + 8 h, k n0 + i + 16 (j >> 1); expert-major, k-steps split over 4 simdgroups, fixed per-row arithmetic so vtest/jspec/ltest/forktest stay exact, NLL 1.88468 / 2.17839) gives M = 1 11.52 → 11.13 ms but M = 8 verify 20.56 → 23.1 ms, so it is off by default. The tiled layout leaves each lane 4 values per (column, 16-value group): 8 scale loads and multiplies per 32 weights against his 2 — matching him needs a row-major expert copy (~17 GB) or a new pack. Wide (576-thread) `attn_res_rms` and wide post-sconv fusions: no change (11.47 / 11.53 vs 11.49 ms).

**Remaining gap** to ~118 tok/s plain (8.5 ms): the NVFP4 MoE matmuls (~3 ms/token, unpack-bound; a dedicated M = 1 GEMV would break the M = 1 vs verify bit-exactness speculation relies on), the msg-gate/attn_res blocks (~2 ms, chains of small single-threadgroup kernels), and attention (~1 ms at short context). ddalcu's build is not public, so whether it keeps decode/verify exactness is unknown.

A/B tooling: `rdump` (all-row logits of one M-row step), `nll`, `Model::step_all`; harness scripts `check.sh` / `cmp.sh` / `kl.py` / `dupprof.sh` / `sweep.sh` live in the Mac's `~/dankattn-inkling2-dev` (not in the repo).

### Speculation Sweep & the MoE Expert Fan-Out Tax (`spec 256 code_prompt.bin`)

Measured at `7ba4268`, before Waves 1–13; the ordering of configs still holds but absolute numbers are now higher (see the table above).

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
- **Chunked prefill `MP=512` (6.0–6.3×)**: Compute-bound (scales with GPU core count). The 2k→8k dip is down from ~21–25% to 6% (Studio) / 1.5% (MBP) since the 8 global layers moved from decode-style split-K attention to `ink_attn_pf_mma` (Wave 3). At `M=512`, `ctx=7,680` the Studio profile is attention 29%, MoE gate/up 26%, MoE down 13%.
- **Speculative decode (4.4×)**: Mixed bandwidth/compute; M5 Ultra absorbs the `M=8` MoE expert fan-out cleanly (2.66× over `M=1`), while Base M5 reaches 1.83× (46.3 tok/s wall) in 32 GB RAM. On the MBP, `M=8` verify is 63% MoE gate/up + down, so fan-out is the ceiling there.

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
./build/dankattn-inkling2 "$PACK" pfbench 2048               # M=64 check + MP=512 chunked prefill bench
./build/dankattn-inkling2 "$PACK" bench 2048 64              # Plain M=1 decode at ctx=2048
./build/dankattn-inkling2 "$PACK" spec 256 prompt_ids.bin    # DSpark + 3-gram lookup + exact greedy check
./build/dankattn-inkling2 "$PACK" dprof 512 8                # Per-op GPU timing breakdown (ctx, M)
./build/dankattn-inkling2 "$PACK" lgdump ids.bin 1600 out.bin # final-row logits: chunked prefill then M=1, for A/B diffs
./build/dankattn-inkling2 "$PACK" ltest ids.bin 4 16           # 4 lanes joint decode + mixed-width round, bit-exact vs alone
./build/dankattn-inkling2 "$PACK" lbench ids.bin 8 2048 32     # 8 lanes, ctx 2048 each, 32 joint decode steps (INK_LPROF=1: op profile)
./build/dankattn-inkling2 "$PACK" jspec ids.bin 2 256          # joint speculation, 2 lanes, exact check (INK_JCHECK=0 skips)
./build/dankattn-inkling2 "$PACK" rdump ids.bin 6000 8 out.bin  # chunked prefill of 6000, then one M=8 step: all rows' logits
./build/dankattn-inkling2 "$PACK" nll ids.bin 8000              # teacher-forced mean NLL through the decode/verify path
./build/dankattn-inkling2 "$PACK" vbench2 ids.bin 2048 8 32     # verify-step GPU ms on fixed real-text rows (rolled back)
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
| `INK_TEMP` / `INK_TOP_P` / `INK_TOP_K` / `INK_MIN_P` / `INK_SEED` | greedy | CLI sampling for every lane (lane `k` uses seed + `k`) |
| `INK_LANES` | `4` | `serve`: lanes (independent sequences) at `INK_CAP` each |
| `INK_SIDE_CAP` | `0` (same as `INK_CAP`) | `serve`: smaller global KV/MTP suffix cap for side lanes `1..INK_LANES-1` (Split-KV forked lanes; e.g. `16384` gives 8 lanes in 13.49 GB at `INK_CAP=131072`) |
| `INK_RING_SLACK` | `0` (`4096` in `inkling2_serve.sh`) | Extra local-ring rows per layer so a lane can rewind `512 + slack` tokens (~0.17 GB per lane per 1k) |
| `INK_PF_DEC_CHUNK` | `256` | `serve`: prefill chunk while other lanes decode |
| `INK_PF_FAST` | `1` | `0`: compile out the interior-tile fast path in `ink_attn_pf_mma` (bit-identical either way; for A/B) |
| `INK_QREL_MMA` | `1` | `0`: prefill relative logits via scalar `ink_q_rel` instead of `ink_q_rel_prep` + `ink_q_rel_mma` (rounding-level difference) |
| `INK_EARLY` | `1` | `0`: encode the whole step into one command buffer (no early commits; same output) |
| `INK_ROUTE_FAST` | `1` | `0`: old decode router (`ink_moe_route` / `topk_pf` + `pack_dec`; same output) |
| `INK_SPLIT_TAB` / `INK_SPLITS` | `1` / unset | `0` drops the decode split table; `"bits:N:K:sgs:ksplit,..."` overrides per tensor shape (reorders sums) |
| `INK_NFT` | `1` | 32-column fragments per mm2 threadgroup: `1`, `2`, or `0` = per-call choice (same output) |
| `INK_NSP_LOCAL` / `INK_NSP_GLOBAL` | `24` / `64` (Ultra/Max), `8` / `8` | decode attention key splits per (row, kv head), up to `ASPLITS` = 64 |
| `INK_FUSE` | `0` | `1`: decode fusions accum + MLP post-sconv, attn post-sconv + pre-MLP attn_res (same output, +0.2 ms) |
| `INK_XBAR` / `INK_SKIP` / `INK_DUP` (+`INK_DUPN`) / `INK_PF_W8` / `INK_PF_MOE` | — | profiling screens (empty phases; drop / repeat op groups; mm2 weight prefetch depth) |

---

## 5. Engineering Log & Backlog

### Milestones
- **`298618b`**: Native NVFP4/MXFP8 + `SoftcappedSwiGLU` in `mm2.metal` / `layout.py`, `pack_inkling2.py`, golden reference `inkling2_golden.py`, and Wave 2A Metal 4 engine (`vtest` `0.000e+00` bit-exact after fixing missing threadgroup barriers in `ink_msg_gate` / `ink_kv_conv_cache` and odd `N/32` `nft=2` bounds in `enc_mm2`).
- **`6848e0a`**: `kv8` global KV cache, exact `sconv` rollback (`conv_hist`), `pack_mtp.py`, and full DSpark MTP speculative decoding (`spec`).
- **`7ba4268`**: `mmp`/`mmpg` chunked prefill (`MP=256`, `pfbench`), online softmax `p_max` confidence cutoff (`INK_DRAFT_CUT`), and `engine/lookup.h` 3-gram context lookup.

- **`56fb0ad`..`180081e` (Waves 1–11)**: parallel `sconv`/`plan_pf`, `CONTIG_X` `mmpg` MoE down; `matmul2d` FlashAttention prefill (2.05–2.47×); GQA G=3 shared-KV decode attention and a `kv8` `sm2` race fix; fused `attn_res`+rms/`mg_prep`, batched `mg3`, parallel-M `moe_route`; DSpark static RNN precompute, 64-TG argmax, fused add+rms; `matmul2d` prefill message gates (6.2× MBP / 2.7× Studio on that op), `big_gpu` `nft=1`; `MP=512`; phase-aliased `MTLHeap` (−140 MB) and `matmul2d` prefill router; `mtp_commit` fused into the draft command buffer.
- **`f68efe2` + `5ecec5d`**: interior-tile fast paths in `ink_attn_pf_mma` (+3–5% prefill). As first committed, its `(full_tile || s > -1e29)` softmax guard made base M5 / macOS 27.0 produce different prefill logits even on inputs where the fast branch never runs (`pfcheck` max diff 0.70 → 5.89); M5 Ultra / 27.0.1 was unaffected. `5ecec5d` restores the plain guard: prefill is now bit-identical fast-on vs fast-off and Studio vs MBP (`lgdump` at n=23/600/1,600).
- **`cf05d1f` (Wave 13)**: `ink_attn_res_mg_prep_shared` (one TG per row on local `has_msgs` layers instead of 3), float4 `moe_accum`, `matmul2d` prefill `q_rel`. The shared `mg_prep` first shipped with a threadgroup-memory race (one `sm_m` row reused across the 3 slots), which made prefill nondeterministic; fixed before commit and verified bit-identical to `5ecec5d`. Studio prefill +1.0% @2k / +1.4% @8k; MBP neutral.
- **`0c35fca`..`9bcfe2b` (branch `inkling2-lanes-fpr` — Single-Output Multi-Lane Engine, 2026-11-21)**:
  - **Zero-copy Split-KV (`fork_lane` / `promote_lane`) & `INK_SIDE_CAP`**: Forked lanes read Lane 0's immutable global KV (`t < fork_pos`) in `ink_attn_split_gqa` and `ink_mtp_attn` while writing only to small suffix buffers (`INK_SIDE_CAP`), blitting only the 40 local SWA-1024 rings. `forktest` (`pref=1500, lanes=4, gen=16`): **100% BIT-EXACT** (`bad_tok=0, bad_lg=0, bad_draft=0, bad_prom=0`, GPU Self-Certainty `conf_err=3.42e-06`) — Studio `fork=4.79 ms`, `promote=0.78 ms`; MBP `fork=16.11 ms`, `promote=3.39 ms` (`0.32 GB` per side lane at `cap=2048`). Live Studio profile: 8 lanes (`L0=131072, side=16384`) in **13.49 GB GPU state** (less RAM than 4 × 131k = 15.25 GB).
  - **Free GPU Self-Certainty**: Computed inside `ink_sample_rows` over the same softcapped logit pass (zero extra DRAM reads).
  - **Batched DSpark (`draft_lanes` + `mtp_commit_lanes`, `f26a2fb`)**: Runs all 3 MTP layers once over `L × 8` rows (`mmp`, `MMP_CH=4, MMP_NF=1, MMP_FAST=0`), packs only the `L × gamma` (`≤ 16`) active rows for one `lm_head` matmul (`ink_pack_frag_rows`), and batches `mtp_mh_fold_w` over all `L` lanes per Markov RNN step. `jspec` exact check: **100% EXACT** on both Macs — Studio 4-lane draft **4.5 → 2.02 ms/round** (`512/512`), 8-lane draft **1.95 ms/round** (**428.8 tok/s wall**); MBP 4-lane draft **18 → 8.65 ms/round** (`256/256`).
  - **In-Engine Single-Output Modes (`flags & 2` in `serve.inc` + OpenAI aliases in `inkling2_server.py`)**:
    - **Parallel Test-Time Compute (`:Nx-select`, `:Nx-genselect`, `:Nx-synth`)**: Fork → parallel decode + DeepConf online confidence pruning + early quorum → resolve in-engine (`select` promotes winning lane to Lane 0; `genselect`/`synth` rewind Lane 0 for 100% prompt prefix reuse). On an 8-problem hard math/code suite: 1-lane **6/8 (75%)** (2 rambling failures) vs. 4-lane Majority & GenSelect **8/8 (100%)**; `:4x-select` hit **302–341 agg tok/s**, `:8x-select` **438.4 agg tok/s**; multi-turn prefix reuse verified (`3,031/3,052` tokens reused, 46 ms prefill).
    - **Parallel Section Fill (`:4x-fast`, `:8x-fast`, `2a2788c` + `9bcfe2b`)**: Terse JSON outline (prefilling `<|content_text|>{"sections":[` to skip thinking) → Split-KV fork with batched per-lane suffix prefill (`step_lanes`) → live Section-1 streaming + progressive flush of Sections `2..K` (auto-falls back to 1-lane on short or tool-call turns). Live SSE on M5 Ultra:
      - 4 Python data structures (`:4x-fast`): **TTFT 1.55 s**, outline 1.11 s, **4,887 visible tok in 15.32 s → 319.0 eff tok/s** (`344.6` fill tok/s; vs. 1-lane 1,321 visible tok in 27.0 s).
      - 8-section systems guide (`:8x-fast`): **TTFT 2.15 s**, outline 1.42 s, **8,048 visible tok in 23.42 s → 343.6 eff tok/s** (`366.0` fill tok/s; **3.0×** 1-lane's 114.0 eff tok/s).
  - **Empirical Dead Ends / Parked Experiments**:
    - **`INK_PASSES=1/2` (`is_varloop: false`)**: Tested on MBP (`smoke`) — both 1-pass (49.6 tok/s) and 2-pass (34.6 tok/s) collapse into tight 7-token repetition loops and disable DSpark (`dspark=0`). All lanes must run 3 passes on `Inkling-2-Mini`; heterogeneous-pass scout lanes only apply to `varloop` checkpoints.
    - **Top-2 Markov Beam-Tree DSpark (`tspec`, parked on branch `inkling2-tree-spec`, `e83d6b5`)**: 4-beam Top-2 tree in one command buffer (`draft` flat at 1.63 vs. 1.60 ms) lifted acceptance +12–23% per round, but 128-expert top-8 MoE fan-out raised verify time +9–13% (`15.7 → 17.1–17.8 ms`), capping net single-stream speedup at **+1.1% to +4.6%** (`132 → 138 tok/s` code, `131 → 132 tok/s` prose).

### Open Questions
- Why DankAttn reasons shorter than SGLang at 128k (1,277 vs 1,778 mean generated tokens on RULER): base vs. uncensored checkpoint, or numerics. A base-checkpoint SGLang RULER run would separate them.
- Chunked prefill vs. `M=1` final-row KL is 1e-5..4e-3 depending on prompt length (`lgdump`, real text, 23–1,600 tok), which is above the `0.00042` golden KL quoted in §1. It predates Waves 1–13 (same band at `5ecec5d`). Worth one `inkling2_golden.py --ids` run at ~600 tokens to see which path sits closer to BF16.
- Base M5 on macOS 27.0 miscompiled (or otherwise mis-executed) a logically neutral edit in `ink_attn_pf_mma`. Re-check with `lgdump` Studio-vs-MBP bit equality after any attention-kernel change or OS update; `pfcheck`'s max diff alone is too noisy to catch it.

### Planned Next Steps (Not Yet Started)
- **Prefix cache that survives engine restarts** (SSD prefix cache, like Hy4).
- **MoE expert fan-out at `M=8` verify** (MBP: `moe_gu` + `moe_dn` = 63% of verify time) is the main lever for MBP speculation.
- **Prefill attention** (29% at `M=512` on the Studio) and **MoE gate/up** (26%) are the remaining large prefill ops.
