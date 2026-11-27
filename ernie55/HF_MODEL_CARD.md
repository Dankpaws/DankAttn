---
base_model: baidu/ERNIE-5.5-Flash-Preview
base_model_relation: quantized
license: apache-2.0
library_name: dankattn
pipeline_tag: text-generation
tags:
  - apple-silicon
  - metal
  - moe
  - 2-bit
  - nf2
  - speculative-decoding
  - dankattn
---

# ERNIE-5.5-Flash-Preview (DankAttn Pack: `pack-ernie55-w2-g32-w8attn`)

Pre-packed weights for `ERNIE-5.5-Flash-Preview` (`ernie5_5_moe`) formatted for the [DankAttn](https://github.com/Dankpaws/DankAttn) Metal 4 inference engine on Apple Silicon.

Includes the 43-layer hybrid architecture (37 HCA sliding-window + C128 compressed-KV attention layers + 6 sparse DSA `kv8` latent attention layers, 4-stream Sinkhorn mHC residual mixing, dense L0 MLP + 42 layers of 512-expert top-10 MoE with shared expert, MultiMax output head), the pack index (`manifest.json`), `tokenizer.json`, `encoding_ernie.py`, and `decoding_ernie.py`. The weights are repacked into DankAttn's own format (`pack-*.bin` / `p1_pack-*.bin`), so they do not load in Transformers, MLX, or llama.cpp.

![DankAttn ERNIE-5.5-Flash-Preview M5 Ultra Overview](assets/dankattn-ernie55-overview.png)

## Quantization & Pack Layout

Mixed-precision quantization tailored to fit the full 43-layer 512-expert MoE + **256k context** inside **96 GB Apple Silicon unified memory** (`85.51 GiB` mapped weights + `2.03 GiB` wired GPU state):

- **Routed MoE experts, layers 2–21 (512 experts, top-10)**: DankAttn **W2-g32 NF2** (2-bit NormalFloat codebook, group size 32 aligned 1:1 with upstream MXFP4 blocks, 13-clip RTN)
- **Routed MoE experts, layers 22–42 (512 experts, top-10)**: DankAttn **W2-g64 NF2** (2-bit NormalFloat codebook, group size 64, 13-clip RTN)
- **Routed MoE experts, layer 1**: DankAttn **W4-g64** (4-bit higher-precision first MoE layer)
- **Dense L0 MLP (`mlp_gu`, `mlp_dn`), all attention projections (`attn_in`, `attn_qb`, `wo_a`, `wo_b`, `kb`, `vb`, `attn_o`), `moe_in`, `moe_out`, & `lm_head`**: DankAttn **W8-g128**
- **Embeddings & small projections (`embed`, `vha_m`, norms)**: BF16 / f32
- **mHC Sinkhorn mixing & MultiMax head**: f32
- **KV caches**: runtime `bf16` HCA SWA-1024 + C128 compressed KV (37 layers) and `kv8` (int8 symmetric per-64-dim latent + bf16 RoPE) on the 6 DSA layers

> **Why hybrid W2-g32 / W2-g64 NF2?** Only the routed MoE experts on layers 2–42 are 2-bit; every attention projection, dense L0 MLP, router, shared expert path, and `lm_head` stays at **W8-g128** (and L1 MoE experts stay at **W4-g64**). Layers 2–21 use group size 32 (aligned 1:1 with upstream MXFP4 blocks) and layers 22–42 use group size 64, both quantized with a 13-clip NF2 RTN codebook (empirical sweeps showed clipped NF2 RTN outperformed 2-bit MoE GPTQ on held-out PPL) so the full 512-expert model + **256k context** fits cleanly inside **96 GB unified memory**. Users on 192–512 GB Macs can also pack higher-bit experts (W4-g64) with `ernie55/convert/pack_ernie55.py`.

### Quality & Determinism

- **`eval_144` held-out NLL / PPL**: **1.4795 / 4.3907**
- **Determinism**: `0.000000` max abs diff across repeated runs (`run1 vs run2`)
- **Speculative decoding (3-gram PLD, ON by default in live serving)**: exact speculative rejection sampling (`k_draft = 7`) for both greedy (`T=0`) and sampled (`T>0`) generation

## Hardware & Memory Requirements

- **Apple Silicon with Metal 4 / `MetalPerformancePrimitives`** (benchmarked on **M5 Ultra Mac Studio, 96 GB**).
- **Unified Memory**: **85.51 GiB** mapped weights + **2.03 GiB** wired GPU state at `cap=262144` (**87.54 GiB** total — fits cleanly in **96 GB unified memory**).
- **Wired memory limit**: Raise macOS's wired GPU limit to 88 GiB (`90112 MB`) before launching:
  ```sh
  sudo sysctl iogpu.wired_limit_mb=90112
  ```
- **Storage**: ~86 GiB on local SSD (`pack-00.bin`..`pack-06.bin` + `p1_pack-00.bin`..`p1_pack-05.bin`).

## Quickstart

```sh
# 1. Install DankAttn (compiles Metal 4 binaries in ~4s)
curl -fsSL https://raw.githubusercontent.com/Dankpaws/DankAttn/main/install.sh | sh

# 2. Raise wired GPU memory limit (96 GB Mac Studio)
sudo sysctl iogpu.wired_limit_mb=90112

# 3. Download this pack (~86 GiB)
dankattn pull ernie55

# 4. Serve OpenAI-compatible API on :8343 (256k context, 3-gram PLD speculation ON)
dankattn serve ernie55 start
dankattn serve ernie55 smoke
```

## Benchmarks (M5 Ultra Mac Studio, 96 GB)

Measured on `pack-ernie55-w2-g32-w8attn` (`cap=262144`, `85.51 GiB` mapped weights + `2.03 GiB` GPU state):

| Workload | Result (GPU) | Result (Wall) | Notes |
| :--- | :---: | :---: | :--- |
| Prefill 4k (`p5_needle4k`, 3,866 tok) | **1,797–1,804 tok/s** | **1,776 tok/s** | `PREFILL_M=1024`, `mmp` / `mmpg` |
| Prefill 16k (`p6_needle16k`, 15,864 tok) | **1,682–1,687 tok/s** | **1,677 tok/s** | Chunked `PREFILL_M=1024` |
| Plain Decode `M=1` (short ctx, 7–36 tok) | **67.5–69.4 tok/s** | **62.7–64.1 tok/s** | `mm1` SIMD matvec |
| Plain Decode `M=1` (4k / 16k ctx) | **65.6 / 63.5 tok/s** | **60.0 / 59.4 tok/s** | `15.24 / 15.75 ms/tok` GPU |
| Speculative 3-gram PLD (`p5_needle4k`, 4k ctx) | **87.1 tok/s** | **80.4 tok/s** | `k_draft=7`, ON by default in server |
| Speculative 3-gram PLD (`p6_needle16k`, 16k ctx) | **84.1 tok/s** | **78.0 tok/s** | `k_draft=7`, ON by default in server |
| `eval_144` NLL / PPL | **1.4795 / 4.3907** | — | `0.000000` run-to-run max abs diff |

## Attribution & License

Derived from `ERNIE-5.5-Flash-Preview`, repacked into DankAttn mixed-precision format (W2-g32 NF2 RTN routed experts L2–21, W2-g64 NF2 RTN L22–42, W4-g64 L1, W8-g128 dense/attention/unembed). Licensed under **Apache 2.0** (matching upstream). The DankAttn engine itself is MIT-licensed.
