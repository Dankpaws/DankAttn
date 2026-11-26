---
base_model: tencent/Hy4-flash
base_model_relation: quantized
license: other
license_name: tencent-hy-license
license_link: LICENSE
library_name: dankattn
pipeline_tag: text-generation
tags:
  - apple-silicon
  - metal
  - gptq
  - hadamard
  - moe
  - speculative-decoding
  - dankattn
---

# Hy4-flash (DankAttn Pack: `pack-v4-h128-w6`)

Pre-packed weights for [`tencent/Hy4-flash`](https://huggingface.co/tencent/Hy4-flash) (revision `d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959`) formatted for the [DankAttn](https://github.com/Dankpaws/DankAttn) Metal 4 inference engine on Apple Silicon (requires DankAttn **`v2026.11.25+`** for `moe_dn_had: 128` and `u4_g32` Engram).

Includes the main 42-layer model, the 5-layer DFly2 speculative drafter (`pack-11.bin`), the SigLIP2 vision encoder (`pack-12.bin`), the 4-bit `u4_g32` Engram n-gram table (`engram.bin`, **36.9 GiB** vs 59.0 GiB FP8 — saving ~22 GB on SSD), the pack index (`manifest.json`), `tokenizer.json`, and `chat_template.jinja`. The weights are re-quantized and re-packed into DankAttn's own format (`pack-*.bin`), so they do not load in Transformers, MLX, or llama.cpp.

*(The legacy FP8-Engram `pack-g64-gptq-v3` pack remains available at [`Dankpaws/Hy4-flash-DankAttn-g64-gptq-v3`](https://huggingface.co/Dankpaws/Hy4-flash-DankAttn-g64-gptq-v3), with an optional `u4-engram` revision.)*

## Quantization & Pack Layout

- **Routed MoE experts (512 experts, top-16, 16k-calibrated GPTQ):**
  - `moe_gu`: W4, group size 64
  - `moe_dn`: W4, group size 128 + block-128 Walsh-Hadamard rotation (`H128`) on 38 layers; **W6, group size 128 + `H128`** on sensitive layers `L1`, `L40`, `L41`
- **Attention `gate` & `o` projections**: W8
- **Dense layers, shared experts, attention `in`/`qb`/`kb`/`vb`, `lm_head`**: W8, group size 128
- **DFly2 drafter layers (`pack-11.bin`)**: W6, group size 128 (`lm_head_w4`: W4g64)
- **SigLIP2 vision encoder (`pack-12.bin`)**: W8, group size 128
- **Embeddings**: bf16
- **Norms, biases, hyper-connection weights, sinks**: f32
- **Engram table (`engram.bin`)**: affine **`u4_g32`** (660,006,416 rows × 96 dims, 60 B/row, **36.9 GiB**, streamed from SSD on demand with a 128 MB lock-free userspace row cache)

### Quality vs bf16 Reference

Evaluated end-to-end against the unquantized bf16 `tencent/Hy4-flash` checkpoint:

- **`wikitext-2` (24 × 512-token cold-start chunks, 12,288 positions)**: mean KL **`0.293`** (median `0.071`), top-1 agreement **`81.9%`** (parity with `pack-g64-gptq-v3`'s `0.287` / `81.9%`).
- **`ids96` / `calib2048`**: KL **`0.0481`** (`92.7%` top-1) on `ids96`; KL **`0.0730`** (`89.9%` top-1, max KL `13.03`) on `calib2048`.
- **Engine bit-exactness**: `vtest` (16-row verify vs 1-row decode), `ntest 4` (4-lane batch vs single-lane), and `jtest 4` (4-lane joint speculation) all pass with **0 differing logits / tokens**.

## Hardware & Memory Requirements

- **Apple Silicon with Metal 4 / `MetalPerformancePrimitives`** (tested on M5 Ultra).
- **Unified Memory**: `87.4 GB` mapped weights + `~6.5 GB` wired GPU state for 4 concurrent lanes (512k main slot + 3× 32k side lanes), or `1.00 GB` state at 32k single-slot (`1.83 GB` at 128k). A 128 GB+ Mac is recommended for full multi-lane serving; 96 GB works with smaller context caps (`HY4_CAP=32768 HY4_SIDE_LANES=0`).
- **Storage**: **`119 GB`** total on fast local NVMe SSD (`~82 GB` weight shards + `36.9 GiB` `engram.bin`).
- **Wired limit**: macOS may cap wired GPU memory below what the model needs. Raise it if prompted at startup:
  ```sh
  sudo sysctl iogpu.wired_limit_mb=90112
  ```

## Quickstart

```sh
# 1. Install or update DankAttn (v2026.11.25+)
curl -fsSL https://raw.githubusercontent.com/Dankpaws/DankAttn/main/install.sh | sh

# 2. Download this pack (~119 GB)
dankattn pull hy4

# 3. Serve OpenAI-compatible API on :8341
dankattn run hy4
# or background daemon:
dankattn serve hy4 start
dankattn serve hy4 smoke
```

## Benchmarks (M5 Ultra, `pack-v4-h128-w6`)

| Workload | Throughput / Latency | Notes |
| :--- | ---: | :--- |
| Prefill (12,272 tok) | **1,718 tok/s GPU** (1,711 wall) | `wht128` 3.08 ms/chunk (0.5%) |
| Prefill (65,520 tok, `HY4_CAP=131072`) | **1,707 tok/s GPU** (**1,704 wall**) | Dedicated `F_NOCACHE` prefill Engram fd |
| Decode 1 row (12k / 64k) | 19.85 / 20.28 ms GPU | Plain single-lane |
| Verify 5 / 8 / 16 rows (12k) | 28.30 / 33.74 / 44.94 ms GPU | Bit-identical to 1-row decode |
| Verify 5 / 8 / 16 rows (64k) | 28.00 / 32.58 / 42.14 ms GPU | Bit-identical to 1-row decode |
| Single-lane speculation (`ids12k` / `chat_code` / `edit_spec` / `qa_doc`) | **157.1 / 120.1 / 358.7 / 163.7 tok/s wall** | 400/400 exact greedy match; 0.58–0.95 ms/rd Engram wait |
| 4-lane joint speculation (`jtest`, greedy) | **211.3 tok/s wall** (`1.41×`) | 0 tokens differ vs single-lane |

## Attribution & License

Derived from [`tencent/Hy4-flash`](https://huggingface.co/tencent/Hy4-flash) (commit `d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959`). These are modified (quantized and re-packed) weights. Use is subject to the upstream `tencent/Hy4-flash` license, included here as [`LICENSE`](LICENSE). The DankAttn engine itself is MIT-licensed.
