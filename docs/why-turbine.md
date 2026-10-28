# Why Turbine

Turbine starts from three ideas about inference serving. This page states each one, what is built today and what is not, so that a reader can judge the project on its evidence.

## 1. KV cache as a managed resource

In most engines the KV cache is disposable GPU memory: a block is kept while a request needs it and dropped when the request ends or memory runs short. Turbine treats KV as a tiered resource.

- **Tiers.** GPU memory (L0), pinned host memory (L1) and NVMe slab files (L2) hold KV blocks. Dropping a block and recomputing it is a virtual tier, and a planner chooses between retrieving a block from a tier and recomputing it.
- **Prefix sharing.** Requests share full-block prefixes by default. Token-granular reuse of the last partial block is on by default too. A request can isolate its cache with `x-turbine-cache-salt`, and the whole feature can be turned off with `kv.prefix_sharing: false`.
- **Sessions.** Clients can pass session hints (`prompt_cache_key` with `x-turbine-session-resume-within` and `x-turbine-session-end`). Idle sessions move down the tiers on configurable timers.
- **Cost-aware eviction.** The default eviction policy is cost-aware rather than LRU. LRU stays available as the benchmark baseline.
- **Compression by tier.** Each lower tier can store blocks in its own format. FP8 and TurboQuant formats are lossy, so reuse of lossy blocks is traceable: lineage keys mark them, a request can opt out with `x-turbine-kv-lossy: deny`, and `usage.prompt_tokens_details.lossy_cached_tokens` reports how many prompt tokens came from lossy blocks.

Measured results are in [performance](performance.md) and the [KV guide](guide/kv-cache.md).

## 2. Reliability first

Turbine tries to avoid out-of-memory failures and device collapse rather than report them afterwards.

- **Pressure controller.** Device and host signals move the server through GREEN, YELLOW, ORANGE, RED and SURVIVAL, with hysteresis so that it does not flap. Each state changes behaviour in a documented way, and every transition is logged and exported as a metric.
- **Predictive admission.** Every admitted request reserves its worst-case KV footprint. A request that does not fit waits in a bounded queue, and a request that waits too long is rejected with a reason code.
- **Emergency reserve.** A VRAM reserve is excluded from normal scheduling and is released only in SURVIVAL.
- **Circuit breaker.** After device errors the breaker stops new work, drains, and probes the device before it accepts requests again. A sticky device error exits the process with code 3 so that a supervisor can restart it.
- **Tested by simulation and soak.** The scheduler runs as a deterministic simulator without a GPU, so overload, out-of-memory recovery and cancellation are tested as repeatable scenarios. Overload soak runs on a GPU host check the behaviour over time.

See the [reliability guide](guide/reliability.md) for the states and their limits.

## 3. Support is declared, not implied

Every combination of vendor, device architecture, model architecture, weight format, KV format and speculation maps to a row of the support matrix with a status: `supported`, `experimental` or `unsupported`.

- An `unsupported` combination is refused at startup with exit code 2 and a reason code, before the port is bound. Nothing falls back silently.
- An `experimental` combination serves, and the server logs a warning and reports the status in `/turbine/v1/status`.
- Each `supported` row carries evidence: golden compare results, GSM8K accuracy against a reference, throughput, and soak results, depending on the row. The [feature matrix](features.md) lists the evidence per row.

`turbine-server --support-matrix` prints every row, and `--check-config` prints the row that the configuration resolves to.

## Also part of the design

- **No Python in the serving path.** The server is Rust from HTTP to the kernel FFI. Hugging Face transformers is used only to generate golden reference files, and the Python scripts that do so are not part of the server.
- **Fast startup.** Turbine reports ready about 6 to 8 seconds after process start for Llama-3.2-3B. vLLM-ROCm took about 98 seconds in the same comparison (see [performance](performance.md) for the setup).
- **Pluggable modules.** Model families, tool-call and reasoning parsers, weight and KV formats, scheduling and eviction policies, backends, card profiles, collectives, routers and speculative proposers each sit behind a trait with a static registry. Adding one does not require conditionals in other crates. See [extension points](extending/README.md).
- **Kernels from existing libraries.** Turbine owns scheduling, memory, KV, placement and reliability. Attention and GEMM kernels come from Composable Kernel and hipBLASLt where they work, and the vendor-neutral C ABI lets the kernel library be replaced without changing the Rust side. The HIP library is loaded at run time, and no vendor GPU crate is in the build.
- **Correctness gates on every optimisation.** Golden compare is strict at concurrency 1 and uses batched bounds at concurrency 16. Accuracy budgets apply per quantized format. A change that moves numerics or throughput is measured before it lands.
- **Unsafe code is confined.** Only `turbine-kernels` and `turbine-device` contain `unsafe` code, and every block carries a safety comment.

## What is not built

- Multi-node serving and advanced distribution are deferred (phases 9 and 10 in the project plan).
- NVIDIA execution is deferred. The CUDA shim and the NVFP4 weight formats wait for that work.
- Speculative decoding has one validated pairing, a Llama-3.2-1B draft for Llama-3.2-3B. Its measured speed-up is below the bar that would make it `supported`, so it is `experimental`. Multi-token-prediction heads are in progress and not served.
- Prefill/decode disaggregation, which the original vision lists, is not implemented.

The [vision specification](https://github.com/piwi3910/Turbine/blob/main/turbine-spec.md) describes the original product scope. Where it differs from the phase specifications under `.procoder/specs/`, the phase specifications are the current plan.
