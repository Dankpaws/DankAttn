---
base_model: tencent/Hy4-flash
base_model_relation: quantized
library_name: dankattn
pipeline_tag: text-generation
tags:
  - apple-silicon
  - metal
  - gptq
  - moe
  - speculative-decoding
  - dankattn
---

# Hy4-flash (DankAttn Pack: `pack-g64-gptq-v3`)

Pre-packed weights for [`tencent/Hy4-flash`](https://huggingface.co/tencent/Hy4-flash) (revision `d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959`) formatted for the [DankAttn](https://github.com/Dankpaws/DankAttn) Metal 4 inference engine on Apple Silicon.

Includes the main 42-layer model, the 5-layer DFly2 speculative drafter, the FP8 Engram n-gram table (`engram.bin`), `tokenizer.json`, and `chat_template.jinja`.

## Quantization & Pack Layout

- **Routed MoE experts (512 experts, top-16)**: W4, group size 64, GPTQ (`moe_gu` and `moe_dn`)
- **Attention `gate` & `o` projections**: W8
- **Dense layers, shared experts, attention `in`/`qb`/`kb`/`vb`, `lm_head`**: W8, group size 128
- **DFly2 draft head (`lm_head_w4`)**: W4
- **Embeddings**: bf16
- **Norms, biases, hyper-connection weights, sinks**: f32
- **Engram table (`engram.bin`)**: FP8 E4M3 (660,006,416 rows x 96 dims, ~63.4 GB, streamed from SSD on demand)

### Quality vs bf16 Reference

Evaluated end-to-end against the unquantized bf16 `tencent/Hy4-flash` checkpoint:

- **`wikitext-2` (24 independent 512-token cold-start prose chunks)**: mean KL `0.287`, top-1 agreement `81.9%`. (Short, high-entropy prose chunks naturally exhibit more top-1/top-2 flips than long-context or code prompts.)
- **16k-calibrated GPTQ (`pack-g64-gptq-v3`)**: `-28.8%` held-out KL reduction vs uncalibrated RTN (`calib2048` single-document KL `~0.103`).
- **`kv8` sparse latent cache vs bf16 KV**: `0.006 - 0.011` KL, `97.4% - 98.4%` top-1 agreement across 2k - 64k context.

## Hardware & Memory Requirements

- **Apple Silicon with Metal 4 / `MetalPerformancePrimitives`** (tested on M5 Ultra).
- **Unified Memory**: ~87.3 GB mapped weights + ~6.5 GB wired GPU state for 4 concurrent lanes (512k main slot + 3x 32k side lanes), or ~1.0 GB state at 32k single-slot. A 128 GB+ Mac is recommended for full multi-lane serving; 96 GB works with smaller context caps (`HY4_CAP=32768 HY4_SIDE_LANES=0`).
- **Storage**: ~151 GB total on fast local NVMe SSD (~87.3 GB weight shards + ~63.4 GB `engram.bin`).
- **Wired limit**: macOS may cap wired GPU memory below what the model needs. Raise it if prompted at startup:
  ```sh
  sudo sysctl iogpu.wired_limit_mb=104448
  ```

## Quickstart

```sh
# 1. Clone and build DankAttn
git clone https://github.com/Dankpaws/DankAttn.git
cd DankAttn
sh hy4/build_hy4.sh

# 2. Install Python front-end dependencies
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# 3. Download this pack
hf download <your-hf-namespace>/Hy4-flash-DankAttn-g64-gptq-v3 --local-dir ./pack-g64-gptq-v3

# 4. Serve OpenAI-compatible API on :8341
HY4_PACK=./pack-g64-gptq-v3 sh hy4/serve_hy4.sh start
sh hy4/serve_hy4.sh smoke
```

## Benchmarks (M5 Ultra, 12,272-token context)

| Workload | Throughput / Latency |
| :--- | ---: |
| Prefill (12,272 tok) | 1,700 tok/s GPU (1,694 wall) |
| Decode 1 row | 19.08 ms GPU (52.4 tok/s) |
| Verify 5 / 8 / 16 rows | 27.61 / 33.24 / 44.76 ms GPU |
| Multi-lane decode (16 lanes, `btest`) | 376.3 tok/s GPU |
| Single-lane speculation (`ids12k` / `chat_code` / `edit_spec`) | 128.1 / 116.9 / 357.3 tok/s wall (400/400 exact greedy) |
| 4-lane joint speculation (`jtest`, greedy) | 199.9 tok/s wall (0 diff vs single-lane) |

## Attribution & License

Derived from [`tencent/Hy4-flash`](https://huggingface.co/tencent/Hy4-flash) (commit `d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959`). Use of these weights is subject to the upstream `tencent/Hy4-flash` license terms.
