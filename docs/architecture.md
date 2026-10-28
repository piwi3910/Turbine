# Architecture

This page describes how a request moves through Turbine, which crate owns which part, how the Rust code reaches the GPU, and how the process is organised. It describes the code at the time of writing; the repository's [engineering rules](https://github.com/piwi3910/Turbine/blob/main/AGENTS.md) are the binding source for design decisions.

## Design in one paragraph

Turbine owns orchestration: request handling, admission, scheduling, the KV cache, placement, reliability and observability. It does not own GPU kernels. Kernels come from existing libraries behind a vendor-neutral C ABI, and a HIP library that Turbine loads at run time implements the calls. The scheduler and the KV code never depend on a specific kernel provider. A pressure controller and a circuit breaker react to memory and device signals, and every extension point is a trait with a static registry.

## Request path

```text
 client
   |  HTTP (OpenAI routes; /turbine/v1 diagnostics)
   v
 turbine-api        axum routes, JSON and SSE, error codes
   |
   v
 request manager    tokenise, chat template, tools, grammar setup     (turbine-model)
   |
   v
 admission          reserves worst-case KV; Admit | Queue | Reject    (turbine-reliability)
   |  pressure state, circuit state, admission queue
   v
 scheduler          continuous batching, chunked prefill, policy       (turbine-scheduler)
   |  <-->  KV orchestrator: block pool, prefix directory, tiers,
   |        eviction policy, planner, sessions                         (turbine-kv)
   v
 batch builder /    builds each iteration's batch; plans the step     (turbine-scheduler,
 execution planner                                                     turbine-server engine)
   |
   v
 model executor     families, layers, attention, MoE, sampling        (turbine-model)
   |
   v
 kernel registry    selects an implementation per operation           (turbine-kernels)
   |  C ABI (kernels/include/turbine_kernels.h)
   v
 libturbine_hip.so  Turbine HIP kernels, Composable Kernel, hipBLASLt  (kernels/rocm)
   |
   v
 GPU
   |
   v
 sampler, logits processors (grammar mask, penalties)  ->  detokenise  ->  SSE or JSON
```

A request leaves the scheduler once per iteration with its tokens, and returns to the API as events. Output tokens stream as they are produced.

## Crates

| Crate                   | Role                                                                                                                                                                                        |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `turbine-api`           | OpenAI routes, the `/turbine/v1` diagnostics, error codes and the HTTP-to-engine backend.                                                                                                   |
| `turbine-core`          | Configuration and its validation, shared vocabulary types, the reason-code table, the support matrix and the generic registry.                                                              |
| `turbine-device`        | GPU discovery and telemetry through NVML and amd-smi, loaded at run time. Contains `unsafe` FFI.                                                                                            |
| `turbine-distributed`   | Collectives, rank runtime, tensor-parallel sharding and data-parallel routing.                                                                                                              |
| `turbine-kernels`       | Capability traits, the kernel registry, the CPU reference provider and the bindings to the shim. Contains `unsafe` FFI.                                                                     |
| `turbine-kv`            | The L0 block pool, block tables, block identity, the prefix directory, tiers, eviction, the planner and sessions.                                                                           |
| `turbine-model`         | Hugging Face configuration, safetensors, the tokenizer, chat templates, the executor, sampling, model families, weight formats, tool-call and reasoning parsers, and speculative proposers. |
| `turbine-observability` | Tracing setup, Prometheus registry and HTTP metrics.                                                                                                                                        |
| `turbine-reliability`   | The memory budget and ledger, the pressure state machine, admission, throttling, recovery and the circuit breaker.                                                                          |
| `turbine-scheduler`     | Request lifecycle, queues, continuous batching, scheduling policies and the deterministic simulator.                                                                                        |
| `turbine-server`        | The binary: command line, startup order, engine thread and exit codes.                                                                                                                      |
| `turbine-tensor`        | Device-memory handles and tensors.                                                                                                                                                          |
| `benches/turbine-bench` | `turbine-bench` (load generation and `kv-sim`) and `turbine-golden` (compare, capture, eval).                                                                                               |

The dependency direction is one-way: `turbine-server` depends on everything, and the libraries do not depend on the server. Unsafe code and FFI are confined to `turbine-kernels` and `turbine-device`, and tests check this.

## Kernels and the C ABI

### The ABI

`kernels/include/turbine_kernels.h` declares the C interface between Rust and a kernel library. Its version is `TURBINE_ABI_VERSION` 2. Minor versions are additive: a new minor version adds symbols and never changes an existing one. Examples:

- v2.5 adds the copy streams used for host-to-device transfers between tiers.
- v2.9 adds `quantize_act` and `qgemm`, used by quantized layers.
- v2.11 adds the KV transcode between formats, and the promotion copy kernel.

The Rust side probes for each symbol. A missing symbol either degrades a feature with a warning, or refuses the configuration with a reason code, depending on the feature.

### The HIP library

`kernels/rocm/` builds `libturbine_hip.so` with CMake and `hipcc` for the target architecture (gfx1201 for the R9700). Its sources cover:

- attention and paged attention: Turbine's own kernels and Composable Kernel's `fmha_fwd_pagedkv` and `fmha_fwd_splitkv`, the latter sharing each K/V page across the query heads of its KV head;
- GEMM through hipBLASLt, with a tuned table per card;
- quantized GEMM, activation quantization and FP8 block kernels;
- mixture-of-experts kernels;
- the gated delta net recurrence used by the Qwen3.5 hybrids;
- KV transcode and TurboQuant encode and decode;
- host staging, copy streams and graph capture.

Some sources are ported from other projects under their licences. The [NOTICE](https://github.com/piwi3910/Turbine/blob/main/NOTICE) file names each project and the licence texts are in `kernels/rocm/third_party/LICENSES/`.

### Loading

The kernel library is loaded at startup from the first of these locations that is set or present:

1. `execution.kernel_library` in the configuration,
2. the `TURBINE_KERNEL_LIBRARY` environment variable,
3. `libturbine_hip.so` beside the `turbine-server` executable,
4. the dynamic loader's search path.

An explicitly configured library that fails to load exits with code 1. The `cpu` backend needs no library.

### Kernel selection

Each card profile lists a preference order of implementations for each operation. The kernel registry picks the first implementation whose `supports` check accepts the shapes and types. Every choice is recorded and reported on `/turbine/v1/status` under `kernels`, so a running server can be checked against the expected kernels.

## Threading

- The HTTP server runs on the tokio runtime in `turbine-api`.
- One dedicated engine thread per server owns the scheduler, the KV pool and the GPU stream. HTTP talks to it over a bounded channel.
- Tensor parallelism runs one thread per rank in local mode. Pipeline parallelism runs one thread per stage.
- Data parallelism runs one engine per replica, with a router in front.
- Overlap scheduling prepares the next iteration's host work while the GPU runs the current one. It is on by default. It is ignored under a tensor- or expert-parallel group.

Queues are bounded at each boundary.

## Startup

The process starts in a fixed order, and each step can refuse:

1. Parse the command line and the configuration. Unknown keys and out-of-range values exit with code 2.
2. Validate the module names against the registries. An unknown name exits with code 2 before any port is bound.
3. Resolve the support row from the backend, the model's `architecture` and the configured formats. An `unsupported` row exits with code 2. `--check-config` stops here.
4. Discover devices through the device library. The row is refined with the device architecture.
5. Load the kernel library and check its ABI.
6. Check the memory budget against the model's weights and the minimum KV. A budget that cannot hold the model exits with code 1.
7. Bind the port. A bind failure exits with code 1.
8. Load the weights, apply quantization and parallel placement, and warm up. Until this finishes, `/ready` answers 503 `loading_model`. Then it answers 200.

## Registries and extension points

Every pluggable part is a trait with a static registry and a conformance suite. The points are: model family, tool-call format, reasoning parser, weight format, KV format, kernel implementation, card profile, execution backend, logits processor, scheduling policy, eviction policy, speculative proposer, collective backend, rank transport, data-parallel router policy and device discovery.

Adding a module means adding one file and one registry entry. The per-point pages in [extension points](extending/README.md) give the steps.

## Exit codes

The process exits with one of four codes. These are the same for every subcommand that starts the server.

| Code | Meaning                                       | Examples                                                                                                                                                                                                                                                                   |
| ---- | --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0    | Clean shutdown.                               | SIGINT or SIGTERM, with the grace period completed or the rest cancelled.                                                                                                                                                                                                  |
| 1    | Runtime or startup failure.                   | Port bind failure; an explicitly configured kernel library that fails to load; a model directory or kernel library that cannot be used; a memory budget that cannot hold the model; `speculative_budget`; `kv_transcode_unavailable`; `promotion_copy_kernel_unavailable`. |
| 2    | Invalid configuration or refused combination. | An unknown module name; an `unsupported` support row; `kv_tq2_l0_refused`; `speculative_parallel_unsupported`; a refused parallel combination such as `combination_unsupported:pp+tp`; an invalid key or value.                                                            |
| 3    | Fatal circuit.                                | A sticky device error; a pressure-controller failure; an engine panic. A supervisor restarts the process.                                                                                                                                                                  |

Startup refusals are logged as `event="startup_refused"` with a `reason` before the process exits, and the message is printed on stderr.

## Configuration and status

The configuration is one YAML file with the top-level sections `server`, `model`, `execution`, `scheduler`, `kv`, `parallel`, `reliability`, `speculative`, `structured_output`, `logging`, `devices` and `quality`. The keys are in the [configuration reference](getting-started/configuration.md). The server's resolved state, including the support row, the modules and the kernels, is reported on `/turbine/v1/status`.
