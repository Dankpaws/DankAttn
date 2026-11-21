# Contributing to DankAttn

Contributions are welcome! DankAttn is a from-scratch Metal 4 inference engine for Apple Silicon built to stay **fast, lean, and bit-exact**. These guidelines exist to keep the codebase easy to work on as new models and optimizations are added.

## Before You Code

- **Bug fixes, docs, and small cleanups:** Go straight to a Pull Request.
- **New model architectures, quantization formats, or major refactors:** Open an Issue first using the **New Model / Feature Proposal** template so we can agree on scope and layout before you spend time implementing it.

## AI-Assisted Contributions

Using AI tools to write, port, or optimize code is **welcome and encouraged** here—DankAttn itself is built with heavy AI assistance. That said, you are still the author of your PR:

- **Own the diff:** Understand what the code does and be ready to explain design choices during review.
- **Run it yourself:** Never submit untested AI-generated code or "blind translations."
- **Keep PRs focused:** One logical change per PR. Avoid drive-by rewrites or formatting churn in unrelated files.

## Architecture & Modularity (Keeping It Lean)

DankAttn avoids a heavy "universal framework" layer so each engine stays fast, readable, and easy to tune. Follow the existing separation between shared primitives and per-architecture code:

- **Shared core** (`engine/mtl.h`, `engine/mm_args.h`, `engine/lookup.h`, `engine/kernels/mm2.metal`, `server/engine_frontend.py`): Keep this layer minimal. Changes here must preserve compatibility with existing architectures and must not regress their speed or memory footprint.
- **Per-architecture modules:** New model architectures should be self-contained rather than adding branches throughout existing engines:
  - `engine/<arch>/` — model loader, forward pass, CLI driver, config
  - `engine/kernels/<arch>.metal` — architecture-specific Metal kernels (reuse `mm2.metal` where it fits)
  - `<arch>/` — build script (`build_<arch>.sh` producing `build/dankattn-<arch>`), conversion/packing scripts, and reference tests
  - `server/<arch>_server.py` — thin OpenAI-compatible front end reusing `server/engine_frontend.py` (if serving is supported)
- **Zero framework dependencies in the inference path:** Inference stays pure Objective-C++ and Metal Shading Language—no PyTorch, MLX, or llama.cpp in the runtime. Python is for conversion scripts, reference checks, and the HTTP front end.

## Hardware & Verification Requirements

All PRs **must be built and verified on Apple Silicon** by the contributor, with actual command output and results included in the PR description. Untested PRs or unverified draft ports will not be merged.

Include your test environment in the PR:
- Apple Silicon chip, unified memory, and macOS version
- Exact commands run and their output

### Changes to existing engines (Hy4 / Flash-Next)
- **Correctness:** Pass the built-in bit-exactness checks (`vtest`, `ntest`, `ctest`, `jtest`, or `hy4/test_gate.sh` / `tests/gate.sh` as applicable).
- **Performance & memory:** If you touch kernels, scheduling, or GPU state/scratch buffers, include before/after benchmark numbers (`bench`, `spec`, or `pprof`) and confirm wired GPU memory did not silently grow.

### Adding a new model architecture
New architectures (including smaller models that make DankAttn easier to test on lower-RAM Macs) are welcome as long as they are cleanly isolated and verified:
- **Reference verification:** Validate output numerically against a known-good reference (e.g., PyTorch, HuggingFace, or llama.cpp)—greedy token match or logit/KL comparison, not just "it generates text."
- **Self-contained tooling:** Include a build script, checkpoint conversion/packing script (or clear instructions), and at least one repeatable correctness/smoke test.
- **Baseline numbers:** Include prefill/decode tok/s and memory usage on your Apple Silicon machine.
- **No hot-path impact:** Existing binaries (`dankattn-hy4`, Flash-Next) must still build cleanly with zero regressions.

## Repo Hygiene & Licensing

- **No large binaries or weights:** Never commit model weights, `.bin` packs, calibration dumps, or local machine paths/secrets.
- **Docs & config:** If you add environment variables, CLI flags, or a new architecture, document them alongside the code (e.g., `config.h` and `README.md`).
- **License:** By submitting a PR, you agree that your contribution is licensed under the repository's [MIT License](LICENSE). Do not copy code from projects with incompatible licenses.
