# Contributing

Thank you for considering a contribution. The full guide is [CONTRIBUTING.md](https://github.com/piwi3910/Turbine/blob/main/CONTRIBUTING.md) in the repository. This page summarises what a change has to pass and where the rules are.

## Getting started

Toolchain: Rust 1.97 (edition 2024). A GPU, ROCm and model weights are not needed to build or run the host tests.

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

Run the server without a GPU to check configuration and the support matrix:

```sh
cargo run -p turbine-server -- --config examples/turbine.yaml --check-config
cargo run -p turbine-server -- --support-matrix
```

The GPU build, the GPU test variables and the model setup are in [Install and build](getting-started/install.md) and [Testing and quality gates](testing.md).

## What a change must pass

- `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and the tests of every crate the change touches.
- A change to GPU-facing code (`turbine-kernels`, `turbine-model`, `turbine-device`, the kernel ABI, `kernels/rocm`) runs the GPU tests on a supported card.
- A change that can affect numerics or throughput runs golden compare at concurrency 1 and 16, and a throughput benchmark, and reports the numbers in the pull request.

## Rules

The binding engineering rules are in [AGENTS.md](https://github.com/piwi3910/Turbine/blob/main/AGENTS.md) and the code conventions are in [Engineering standards](standards.md). In short:

- **Every optimised path has a reference test.** Do not trade correctness or stability for throughput by default. Measure before specialising a hot path.
- **Pluggable by construction.** A new model family, format, kernel, backend, policy, proposer, collective or transport is one file plus one registry entry. Follow the page for the extension point under [Extension points](extending/README.md). Do not add conditionals for one implementation across crates.
- **Refuse, do not guess.** An unsupported combination is refused at startup with a reason code and a support-matrix row. It is never served by a silent fallback.
- **Bounded everything.** Every queue and cache is bounded. Cancellation releases resources promptly, with a test.
- **No Python in the build or serving path.** Python is allowed only for offline fixture generation.
- **Reuse kernels first.** Before writing a kernel, evaluate Composable Kernel, hipBLASLt, vLLM, llama.cpp and AITER kernels. Write our own only when none works or all are measurably slower.
- **Unsafe code and FFI stay isolated.** Every `unsafe` block has a `SAFETY:` comment.

## Commits and pull requests

- One logical change per commit, with a conventional message: `feat(kv): …`, `fix(scheduler): …`, `docs: …`, `perf(kernels): …`.
- Open the pull request against `main`. Describe what changed, why, and the evidence: tests, and golden and benchmark numbers for numeric or performance changes.
- Behaviour-changing design decisions are recorded in the repository's decision log. Cross-crate names (configuration keys, metrics, reason codes and the ABI) are kept in the interface contract.

## Reporting issues

Open a GitHub issue with:

- the Turbine commit,
- the configuration (the `--check-config` output),
- the device (`/turbine/v1/status`),
- the log lines around the failure, using the `event=` lines,
- for a numeric difference, the prompt and the `turbine-golden` output.

## Licence

By submitting a contribution you agree that it is licensed under the Apache License 2.0, and that you have the right to submit it. Code ported from another project keeps its own licence. Add the upstream file, commit, licence and what you changed to the top of the ported file, put the licence text in `kernels/rocm/third_party/LICENSES/`, and add the project to [NOTICE](https://github.com/piwi3910/Turbine/blob/main/NOTICE). See [License and notices](license.md).
