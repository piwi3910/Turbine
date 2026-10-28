# Testing and quality gates

Turbine's tests fall into four groups: host tests that need no GPU, GPU tests that need a ROCm host and model weights, numerics gates that compare generations with a reference, and load and soak runs that test behaviour over time. This page lists what each group checks and how to run it. Contributors need only the host tests to submit a change; the rest run on a GPU host.

## Host tests

These need only a Rust toolchain (1.97 or newer):

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

Run one crate or one test:

```sh
cargo test -p turbine-core
cargo test -p turbine-core config::tests::byte_size_parsing
cargo test -p turbine-api --test api route_table_phase0
```

Checks that guard the design and that run as part of the host tests:

| Check                      | Command                                                                      | What it guards                                                                                               |
| -------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Docs stay true to the tree | `cargo test -p turbine-model --test docs_extending`                          | Every extension-point page exists and names real paths and tests; the support-matrix page carries every row. |
| Registry conformance       | `cargo test --workspace` (each crate's `registry_conformance` tests)         | Every registered module passes its extension point's suite.                                                  |
| Unsafe code confined       | `cargo test -p turbine-kernels` (`unsafe_isolation`)                         | Only the kernel and device crates contain `unsafe`.                                                          |
| Vendor-neutral ABI         | `cargo test -p turbine-kernels` (`vendor_neutral_api`, `abi_header_neutral`) | The Rust API and the C header name no vendor.                                                                |
| No vendor GPU crate        | `cargo tree --workspace \| grep -Ei 'hip\|rocm\|cuda'` must print nothing    | GPU libraries are loaded at run time, never linked as crates.                                                |
| Support rows               | `cargo test -p turbine-core` (`support::tests`)                              | The resolution of every key, and the Phase 7 gate rows.                                                      |

## Scheduler and KV simulators

The scheduler runs without a GPU as a deterministic simulator, so overload, memory recovery and cancellation are tested as repeatable scenarios:

```sh
cargo test -p turbine-scheduler --test overload_sim
cargo test -p turbine-scheduler --test kv_sim
cargo test -p turbine-reliability
```

`overload_sim` includes deterministic 10× overload, out-of-memory recovery, cancellation, and named regressions for SURVIVAL liveness. The liveness sweep prints the outcome for seeds 1 to 12:

```sh
cargo test -p turbine-scheduler --test overload_sim survival_liveness_sweep -- --ignored --nocapture
```

Fault injection needs a build with the `fault-injection` feature. A default build refuses the fault section of the configuration:

```sh
cargo test -p turbine-server --features fault-injection --test fault
```

The offline KV policy simulator replays a workload through an eviction policy:

```sh
cargo run --release -p turbine-bench --bin turbine-bench -- kv-sim \
  --workload multi-turn --policy cost_aware --l0-blocks 256 --l1-blocks 1024 --output json
```

## Building the kernel library

The HIP kernel library builds on a ROCm host with `hipcc` and CMake:

```sh
cmake -S kernels/rocm -B build/rocm -DCMAKE_HIP_COMPILER="$(command -v hipcc)" -DGPU_TARGETS=gfx1201
cmake --build build/rocm
```

See [Install and build](getting-started/install.md).

## GPU tests

GPU and weights tests are marked `#[ignore]` so that a host build skips them. Run them with the variables below. A missing variable fails the test rather than skipping it, and no test downloads a model.

| Variable                                           | Meaning                                   |
| -------------------------------------------------- | ----------------------------------------- |
| `TURBINE_TEST_BACKEND=hip`                         | Selects the HIP backend.                  |
| `TURBINE_KERNEL_LIBRARY=<build>/libturbine_hip.so` | The kernel library to load.               |
| `TURBINE_TEST_MODEL_DIR=<model dir>`               | A Llama-3.2-3B-Instruct directory (BF16). |
| `TURBINE_TEST_MOE_MODEL_DIR=<model dir>`           | An OLMoE directory, for the MoE tests.    |

```sh
TURBINE_TEST_BACKEND=hip \
TURBINE_KERNEL_LIBRARY=build/rocm/libturbine_hip.so \
TURBINE_TEST_MODEL_DIR=/models/llama-3.2-3b-instruct \
cargo test --workspace -- --include-ignored
```

Named GPU suites include `hip_ops` in `turbine-kernels` (kernels against the reference), `kv_gpu` in `turbine-server` (prefix reuse against cold runs, and the NVMe round trip), `pinned_round_trip` in `turbine-kernels` (host copies), and the `tiny_model` and `golden` tests in `turbine-model` and `turbine-bench`.

## Numerics: golden compare

Golden compare replays fixed prompts greedily and checks the server's logprobs against a reference. The reference is generated once, offline, by Hugging Face transformers (`scripts/golden/hf_reference.py`); it is committed, and the server never runs Python.

Run the comparison against a running server:

```sh
cargo run --release -p turbine-bench --bin turbine-golden -- compare \
  --url http://127.0.0.1:8000 \
  --reference tests/golden/llama-3.2-3b-instruct/reference.jsonl --concurrency 1
```

- `--concurrency 1` is the strict gate. The tolerance for Llama holds each reference top-5 candidate to 0.15 nats when its reference logprob is above −2, and to 0.55 nats otherwise.
- `--concurrency 16` keeps 16 prompts in flight. Batch composition changes GEMM rounding, so the looser batched bounds apply (0.25 and 0.75 nats).
- A golden gate is both runs.
- `--multi-gpu-bounds` judges multi-GPU MoE runs with their calibrated bounds.
- `turbine-golden positions --url <base> --prompt-id <id>` prints one prompt's per-position differences, for diagnosis.

The Llama tolerance is in `tests/golden/llama-3.2-3b-instruct/tolerance.json`. The OLMoE bounds were calibrated against the spread between transformers' own runs (`scripts/golden/self_spread.py`).

Captures record another engine alongside the reference:

```sh
cargo run --release -p turbine-bench --bin turbine-golden -- capture \
  --url <base> --prompts tests/golden/prompts.jsonl --out <reference.jsonl>
```

## Accuracy: GSM8K

The 200-item GSM8K set (`tests/eval/gsm8k-200.jsonl`) measures task accuracy. Run it at concurrency 16 and compare with a baseline:

```sh
cargo run -q -p turbine-bench --bin turbine-golden -- eval \
  --url <base> --tasks tests/eval/gsm8k-200.jsonl --concurrency 16 --output json \
  > tests/eval/<model-slug>/<engine>.json

cargo run -q -p turbine-bench --bin turbine-golden -- eval-compare \
  --baseline <bf16.json> --candidate <quantized.json>
```

`eval-compare` exits with code 1 when the drop exceeds `--max-drop`. The default is 0.01. Quantized rows use the per-format limit in the checkpoint's `gate.json`: 0.02 for FP8 and 0.04 for 4-bit formats, when no reference engine runs the same checkpoint. Base checkpoints without a chat template use the completion-form set, `tests/eval/gsm8k-200-completion.jsonl`.

For KV formats, a shared-prefix variant (`tests/eval/gsm8k-200-shared-prefix.jsonl`) exercises the reuse path. The candidate run must also report lossy cached tokens, with `--min-lossy-cached-ratio`, so that the lossy blocks were actually reused.

Accuracy results are compared pairwise over several server runs, with a McNemar test to separate noise from a real drop.

## Load tests and benchmarks

`turbine-bench` drives an OpenAI-compatible endpoint:

```sh
cargo run --release -p turbine-bench --bin turbine-bench -- \
  --url http://127.0.0.1:8000 --concurrency 16 --requests 200 \
  --prompt-words 512 --max-tokens 256 --ignore-eos --output json
```

It also has a multi-turn profile that reports cached-token ratios and time to first token on the first and later turns:

```sh
turbine-bench --url <base> --profile multi-turn --sessions 16 --turns 8 \
  --shared-prefix-words 2000 --concurrency 8 --session-hints
```

`turbine-bench --help` lists every option. Results must be measured on a quiet GPU. Use the [performance](performance.md) page for the method and caveats of published figures.

## Overload soak

The soak runs the server under overload, then cools down, and writes a verdict file. The verdict covers the server's behaviour under overload and during the cool-down. A ten-minute run is the phase gate; a four-hour run is the release gate. The soak needs a GPU host with the model; the project's script for it is `scripts/overload-soak.sh`.

## Gates by kind of change

| Change                                                                                                 | Required                                                                                                                 |
| ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| Any code                                                                                               | `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and the tests of each crate touched. |
| GPU-facing code (`turbine-kernels`, `turbine-model`, `turbine-device`, the kernel ABI, `kernels/rocm`) | The GPU tests on a supported card.                                                                                       |
| Anything that can change numerics or throughput                                                        | Golden compare at concurrency 1 and 16, and a throughput benchmark. Report the numbers in the pull request.              |
| A new quantized format or KV format                                                                    | The GSM8K gate for its budget, the golden compare, and, for a KV format, the shared-prefix eval.                         |
| Release                                                                                                | The overload soak, the full GPU test run, and the golden and GSM8K gates for each supported row.                         |

The repository's `scripts/` directory holds the project's lab tooling for its GPU hosts, including the commit gate and the benchmark and soak runners. It is not needed to contribute, and it assumes access to those hosts.

## Reporting a failing test

A failing test should be reported with the command, the commit, the device (`/turbine/v1/status`), and the log lines around the failure. For a numeric failure, include the prompt and the `turbine-golden` output. See [Contributing](contributing.md).
