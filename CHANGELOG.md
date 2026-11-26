# Changelog

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
