# Contributing to Turbine

Thanks for your interest in Turbine. This page says how to build and test the project, what a change has to pass, and the rules every change follows.

## License

Turbine is licensed under the [Apache License 2.0](LICENSE). By submitting a contribution you agree that it is licensed under the same terms (Apache-2.0 section 5), and that you have the right to submit it. Code ported from another project keeps its own license: add the upstream file, commit, license and what you changed to the top of the ported file, put the license text in `kernels/rocm/third_party/LICENSES/`, and add the project to [NOTICE](NOTICE).

## Getting started

Toolchain: Rust 1.97 (edition 2024). No GPU, ROCm install or model weights are needed to build or run the host tests; GPU libraries are loaded at runtime and GPU tests are `#[ignore]`d.

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

Run the server without a GPU (CPU backend, for API and scheduler work):

```sh
cargo run -p turbine-server -- --config examples/turbine.yaml --check-config
cargo run -p turbine-server -- --support-matrix
```

### GPU tests

The HIP kernel library builds on a ROCm host:

```sh
cmake -S kernels/rocm -B build/rocm -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc -DGPU_TARGETS=gfx1201
cmake --build build/rocm
```

GPU and weights tests read `TURBINE_TEST_BACKEND=hip`, `TURBINE_KERNEL_LIBRARY=<build>/libturbine_hip.so` and `TURBINE_TEST_MODEL_DIR=<model dir>`, and run with `cargo test -- --include-ignored`. A missing variable fails the test; tests never download weights.

## What a change must pass

- `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and the tests of every crate it touches.
- A change to GPU-facing code (`turbine-kernels`, `turbine-model`, `turbine-device`, the kernel ABI, `kernels/rocm`) runs the GPU tests on a supported card.
- A change that can affect numerics or throughput runs the golden compare (`turbine-golden compare`, at concurrency 1 and 16) and a throughput bench, and reports the numbers in the pull request.

## Rules

The binding engineering rules are in [AGENTS.md](AGENTS.md#engineering-rules-spec-21--binding) and the code conventions in [docs/standards.md](docs/standards.md). In short:

- **Every optimized path has a reference test.** Never trade correctness or stability for throughput by default; measure before specializing a hot path.
- **Pluggable by construction.** Model families, tool and reasoning formats, weight and KV formats, kernels, backends, card profiles, scheduling and eviction policies, speculative proposers, collectives and transports are one file plus one registry entry each. Follow the page for the extension point under [docs/extending/](docs/extending/README.md); never add conditionals for one implementation across crates.
- **Refuse, don't guess.** An unsupported combination is refused at startup with a reason code and a support-matrix row, never served by a silent fallback.
- **Bounded everything.** Every queue and cache is bounded; cancellation releases resources promptly, with a test.
- **No Python in the build or serving path.** Python is allowed only for offline fixture generation (`scripts/golden`, `scripts/eval`).
- **Reuse kernels first.** Before writing a kernel, evaluate Composable Kernel, hipBLASLt, vLLM, llama.cpp and AITER kernels; write our own only when none works or all are measurably slower.
- **Unsafe and FFI stay isolated**, and every `unsafe` block has a `SAFETY:` comment.

## Commits and pull requests

- One logical change per commit, with a conventional message (`feat(kv): …`, `fix(scheduler): …`, `docs: …`, `perf(kernels): …`).
- Open a pull request against `main`; describe what changed, why, and the evidence (tests, golden and bench numbers for numeric or perf changes).
- Design questions that change behaviour are recorded in `.procoder/ask/decisions.md`; cross-crate names (config keys, metrics, reason codes, ABI) live in `.procoder/contract/interfaces.md`.

## Reporting issues

Open a GitHub issue with the Turbine commit, the config (`--check-config` output), the device (`/turbine/v1/status`), and the log lines around the failure (`event=` lines). For a numeric difference, include the prompt and the `turbine-golden` output.
