# Feature matrix

This page records, for each feature, how mature it is and what evidence supports that. It describes the repository as of 2026-10-29.

## Authority

The support matrix in code is the authority. `turbine_core::support::SUPPORT_MATRIX` (in [`crates/turbine-core/src/support.rs`](https://github.com/piwi3910/Turbine/blob/main/crates/turbine-core/src/support.rs)) decides what the server serves. The command below prints it, and the server reports the row it resolved for its configuration:

```sh
cargo run -p turbine-server -- --support-matrix
cargo run -p turbine-server -- --support-matrix --output json
```

The [support matrix](support-matrix.md) page is a hand-kept snapshot of the same table. It lags the code: its "Every row" table still lists Llama L0 `tq4` as experimental, and it shows the gfx1201 Qwen3, Mistral, Mixtral and Qwen3-MoE rows as pending, while the code lists them as supported. Where the two disagree, this page and the code are correct.

## Levels

| Level        | Meaning                                                                                                                                                                    |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Supported    | The combination passed its gate. The evidence is named in the row: golden compare, GSM8K accuracy, throughput, soak, as the feature requires. It serves without a warning. |
| Experimental | Served, with a startup warning (`event="support_matrix"`). Either the gate has not closed, or the feature is kept on purpose for tests.                                    |
| Unsupported  | Refused at startup with exit code 2 and a reason code, before any port is bound.                                                                                           |
| Deferred     | Work is on hold by project decision. The configuration is refused with the name of the deferred phase.                                                                     |
| Planned      | Specified or in progress, not served.                                                                                                                                      |

## Hardware and backends

| Feature                                   | Level        | Note                                                                                     |
| ----------------------------------------- | ------------ | ---------------------------------------------------------------------------------------- |
| AMD Radeon AI PRO R9700 (gfx1201), HIP    | Supported    | The primary target. Every gated row is on gfx1201.                                       |
| Other AMD architectures                   | Experimental | Family rows fall back to "gate pending" outside gfx1201.                                 |
| CPU backend                               | Experimental | The reference and test backend, for tiny checkpoints and fixtures. Not a serving target. |
| NVIDIA (GB10, sm_121)                     | Deferred     | Refused before any row is consulted (`DEFERRED_VENDORS`).                                |
| NVFP4 and ModelOpt weight formats         | Deferred     | Reserved for the NVIDIA work.                                                            |
| Execution backend registry                | Supported    | `cpu` and `hip`, chosen with `execution.backend`.                                        |
| Card profiles                             | Supported    | `execution.card_profile`; `auto` selects the profile of the discovered architecture.     |
| Device discovery through NVML and amd-smi | Supported    | Loaded at run time; no vendor GPU crate in the build.                                    |
| Multi-node                                | Deferred     | Phase 9 of the project plan.                                                             |

Sources: `crates/turbine-core/src/support.rs`, `crates/turbine-kernels/src/backends/mod.rs`, `crates/turbine-kernels/src/cards/mod.rs`, `crates/turbine-device/src/discovery/mod.rs`.

## Model families and placements

Placements are one GPU, tensor parallel 2 (tp2), expert parallel 2 (ep2), data parallel 2 (dp2) and pipeline parallel (pp).

| Family (architecture)                                                    | 1 GPU                                                                                                                | tp2                                                   | ep2                      | dp2           | pp                          |
| ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ------------------------ | ------------- | --------------------------- |
| Llama-3.2-3B-Instruct (`LlamaForCausalLM`, BF16)                         | Supported. About 864 tok/s at concurrency 16.                                                                        | Supported. Tensor-parallel prefill overlap is opt-in. | Not applicable (dense).  | Supported.    | Supported with local ranks. |
| OLMoE-1B-7B-0125-Instruct (`OlmoeForCausalLM`, BF16)                     | Supported. About 639 tok/s at concurrency 16.                                                                        | Supported; gated against the committed reference.     | Supported.               | Supported.    | Supported with local ranks. |
| Qwen3 dense, Qwen3-4B (`Qwen3ForCausalLM`, BF16)                         | Supported. GSM8K 0.915 against vLLM's 0.920; throughput 0.95 times vLLM.                                             | Experimental (not gated).                             | Not applicable.          | Experimental. | Experimental.               |
| Qwen3-MoE (`Qwen3MoeForCausalLM`, `fp8_block`)                           | Supported as a matrix row; no single-GPU gate.                                                                       | Supported.                                            | Supported.               | Experimental. | Experimental.               |
| Qwen3-MoE, BF16                                                          | Experimental.                                                                                                        | Experimental.                                         | Experimental.            | Experimental. | Experimental.               |
| Mistral-7B (`MistralForCausalLM`, BF16)                                  | Supported. Throughput 0.99 times vLLM; GSM8K +0.02.                                                                  | Experimental.                                         | Not applicable.          | Experimental. | Experimental.               |
| Mixtral (`MixtralForCausalLM`, `fp8`)                                    | Supported as a matrix row; no single-GPU gate.                                                                       | Supported.                                            | Supported.               | Experimental. | Experimental.               |
| Mixtral, BF16                                                            | Experimental.                                                                                                        | Experimental.                                         | Experimental.            | Experimental. | Experimental.               |
| gpt-oss-20b (`GptOssForCausalLM`, `mxfp4`, `openai_mxfp4` packaging)     | Supported. GSM8K 0.925 against vLLM's 0.930.                                                                         | Experimental.                                         | Experimental.            | Experimental. | Experimental.               |
| gpt-oss, BF16                                                            | Unsupported. No BF16 checkpoint exists: the experts are always stored fused in MXFP4.                                |                                                       |                          |               |                             |
| Qwen3.5 hybrid (Gated DeltaNet, `Qwen3_5ForConditionalGeneration`, BF16) | Experimental. Golden compare 9 of 16 at concurrency 1. The recurrence amplifies BF16 rounding, and the gate is open. | Experimental.                                         | Not applicable.          | Experimental. | Experimental.               |
| Qwen3.6 (`Qwen3_5MoeForConditionalGeneration`, BF16 or `fp8_block`)      | Experimental.                                                                                                        | Experimental, gate open.                              | Experimental, gate open. | Experimental. | Experimental.               |

Every family is registered in `crates/turbine-model/src/families/mod.rs`. The Phase 7 placements are gated by `PHASE7_ROWS` in `support.rs`, and a row turns supported only with a matching closing line in the performance log (`support::tests::phase7_rows`).

Throughput figures in this table are the Phase 7 exit values. See [Performance](performance.md) for the methods and the caveats.

## Weight formats

| Format (packagings)                                                                 | Level                    | Note                                                                                             |
| ----------------------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------ |
| BF16                                                                                | Supported                | The default for Llama, OLMoE, Qwen3 and Mistral.                                                 |
| FP8 per-tensor (`ct_fp8`, `hf_fp8`) on Llama                                        | Supported                | Golden 16 of 16. The GSM8K drop of 0.0129 is within noise (McNemar p 0.152).                     |
| FP8 block (`fp8_block`) on Llama                                                    | Supported                | Concurrency 16 at 1061 tok/s, 1.57 times vLLM. GSM8K drop 0.005. A 10-minute soak passed.        |
| FP8 block on Qwen3-MoE                                                              | Supported at tp2 and ep2 | Qwen3.6 is experimental.                                                                         |
| AWQ INT4 (`awq`, `ct_pack_int4`) on Llama                                           | Supported                | Concurrency 16 at 1.50 times the BF16 tokens per second. GSM8K drop 0.030 against a 0.04 budget. |
| GPTQ INT4 (`gptq`) on Llama                                                         | Supported                | An AutoRound checkpoint. GSM8K drop 0.0235. Golden 16 of 16. `act_order` is refused.             |
| MXFP4 (`ct_mxfp4`, `quark_mxfp4`) on Llama                                          | Experimental             | The results are close to the gate, and the MXFP4 prefill kernel is a recorded follow-up.         |
| MXFP4 weights with MXFP4 activations (`mxfp4_a4`) on Llama                          | Experimental             | Golden 10 of 16.                                                                                 |
| MXFP4 on gpt-oss (`openai_mxfp4`)                                                   | Supported                | See the model table.                                                                             |
| Quantized formats on any other row                                                  | Unsupported              | Refused with the quantization reason for the row.                                                |
| Quantized formats on the CPU backend                                                | Experimental             | Host tests and fixtures.                                                                         |
| NVFP4 and ModelOpt (`modelopt_nvfp4`, `modelopt_fp8`, `modelopt_mixed`, `ct_nvfp4`) | Deferred                 | Reserved for the NVIDIA work.                                                                    |
| Weight format registry                                                              | Supported                | Nine packagings, chosen from `config.json`. `crates/turbine-model/src/weights/mod.rs`.           |

## KV formats, L0 and lower tiers

| Format                   | As the L0 format (`kv.dtype`)                                                                                                                                                                          | As a lower-tier format (`kv.cpu.format`, `kv.nvme.format`)                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| BF16 (`l0`)              | Supported (default).                                                                                                                                                                                   | Supported (default).                                                                         |
| `fp8_e4m3`               | Supported on Llama with BF16 weights: GSM8K at concurrency 16 at 0.7885, against 0.7801 for BF16; golden 16 of 16. Supported on OLMoE with calibrated scales (drop 0.0030). Unsupported on other rows. | Supported. Shared-prefix GSM8K median drop 0.005, McNemar p at least 0.55 on all pairs.      |
| `tq4` (TurboQuant 4-bit) | Supported on gfx1201 Llama: shared-prefix pair 0.765 against 0.770 and 0.760. Experimental on OLMoE (shared-prefix drop 0.045) and on the CPU backend. Unsupported elsewhere.                          | Supported. Multi-turn cached ratio Llama 0.9075 against 0.9007, OLMoE 0.8625 against 0.8624. |
| `tq2` (TurboQuant 2-bit) | Refused with `kv_tq2_l0_refused`, exit 2. Shared-prefix GSM8K collapses to 0.15 to 0.20.                                                                                                               | Experimental. Fails the shared-prefix eval (Llama 0.720, OLMoE 0.610). It is a ladder rung.  |

The format registry is `crates/turbine-kv/src/codec/mod.rs`. Tier formats are checked against `TIER_FORMAT_REFUSALS`.

## KV tiers, prefix sharing and the compression ladder

| Feature                     | Level             | Note                                                                                                                                                                                                                            |
| --------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| L0 paged GPU pool           | Supported         | `kv.block_tokens` defaults to 128. Composable Kernel paged and split-KV attention. Other sizes use the slower kernel and log `paged_attention_fallback`.                                                                        |
| L1 pinned host memory       | Supported         | Slabs of 128 MiB up to `kv.cpu.max_bytes`. Discrete-VRAM devices only.                                                                                                                                                          |
| L2 NVMe slab files          | Supported         | Wiped at startup. Capped by free disk space.                                                                                                                                                                                    |
| Recompute as a virtual tier | Supported         | The planner weighs retrieval cost against recompute cost. `crates/turbine-kv/src/planner.rs`.                                                                                                                                   |
| Eviction policies           | Supported         | `cost_aware` (default) and `lru`.                                                                                                                                                                                               |
| Full-block prefix sharing   | Supported         | On by default. `x-turbine-cache-salt` isolates a tenant; `kv.prefix_sharing: false` turns it off.                                                                                                                               |
| Token-granular prefix reuse | Supported         | `kv.partial_prefix_reuse: true` by default.                                                                                                                                                                                     |
| Session hints               | Supported         | `prompt_cache_key`, `x-turbine-session-resume-within`, `x-turbine-session-end`.                                                                                                                                                 |
| Prefetch                    | Supported         | `POST /turbine/v1/kv/prefetch`.                                                                                                                                                                                                 |
| Promotion copy kernel       | Supported         | The ABI v2.11 copy kernel is about 1.8 times the copy engine, with no decode stall.                                                                                                                                             |
| Lossy reuse with lineage    | Supported         | `kv.lossy_reuse`, `kv.lossy_penalty`, the `x-turbine-kv-lossy` header and `lossy_cached_tokens`.                                                                                                                                |
| Compression ladder          | Supported, opt-in | `l0 -> fp8_e4m3 -> tq4 -> tq2 -> evict`. The full ladder measured 0.909 times the tokens per second with 6.2 % fewer recomputed tokens. With `kv.ladder.l0: false` it measured 0.952 times, with inter-token latency unchanged. |
| Offline KV policy simulator | Supported         | `turbine-bench kv-sim`.                                                                                                                                                                                                         |

## Parallelism

| Feature                              | Level                 | Note                                                                                                                      |
| ------------------------------------ | --------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Tensor parallel                      | Supported             | `parallel.tensor_parallel_size`.                                                                                          |
| Data parallel with a router          | Supported             | `prefix_affinity` and `least_loaded`.                                                                                     |
| Pipeline parallel                    | Supported             | Local ranks only.                                                                                                         |
| Expert parallel                      | Supported             | MoE families.                                                                                                             |
| Accepted combinations                | Supported             | Tensor × data; pipeline × data with tensor and expert at 1; expert × data with tensor at 1 or equal to the expert degree. |
| Refused combinations                 | Unsupported           | `combination_unsupported:pp+tp`, `pp+ep`, `pp+static`.                                                                    |
| Collective backends                  | Supported             | `host`, `hostmem`, `rccl`. NCCL is deferred with NVIDIA.                                                                  |
| Tensor-parallel prefill overlap      | Supported, opt-in     | Bitwise equal to the unsplit run. +10 % tokens per second and −38 % TTFT at tensor parallel 2 on Llama.                   |
| Collective verification              | Supported, diagnostic | `parallel.collective.verify` fails the step with `collective_corrupt`.                                                    |
| Rank transport                       | Supported             | `tcp`.                                                                                                                    |
| Refusal table                        | Empty                 | No combination is refused by the parallel refusal table today.                                                            |
| Multi-node and advanced distribution | Deferred              | Phases 9 and 10.                                                                                                          |

See [Multi-GPU parallelism](guide/parallelism.md) for the details.

## Serving and API

| Feature                                                 | Level                    | Note                                                                                                              |
| ------------------------------------------------------- | ------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `/v1/chat/completions`, `/v1/completions`, `/v1/models` | Supported                |                                                                                                                   |
| Server-sent event streaming and `stream_options`        | Supported                |                                                                                                                   |
| Sampling                                                | Supported                | Temperature, top-p, top-k, seed, stop, stop token IDs, minimum tokens, ignore-EOS, n.                             |
| Logprobs                                                | Supported                | `logprobs`, `top_logprobs`, `echo`, `return_tokens_as_token_ids`.                                                 |
| Penalties and bias                                      | Supported                | Presence, frequency and repetition penalties, and `logit_bias`, as logits processors.                             |
| Structured output                                       | Supported                | `response_format` with `json_object` or `json_schema`, enforced by llguidance.                                    |
| Jump-forward structured output                          | Supported, on by default | Forced tokens are verified as one step.                                                                           |
| Tool calls                                              | Supported                | Parsers `llama3_json`, `hermes`, `mistral`, `harmony`, `qwen3_xml`. `tool_choice` and `parallel_tool_calls`.      |
| Reasoning content                                       | Supported                | `qwen_think` and `harmony`. `model.reasoning_parser` `auto` or `none`. `reasoning_effort` for Harmony.            |
| Chat-template kwargs and overrides                      | Supported                | `chat_template_kwargs` and `model.chat_template`.                                                                 |
| Priority                                                | Supported                | The `priority` request field.                                                                                     |
| Cached-token usage                                      | Supported                | `usage.prompt_tokens_details.cached_tokens` and `lossy_cached_tokens`.                                            |
| Diagnostic endpoints                                    | Supported                | `/turbine/v1/{status, devices, kv, kv/prefetch, pressure, scheduler, topology}`.                                  |
| Fast startup                                            | Supported                | Ready 6 to 8 seconds after start for Llama-3.2-3B, against about 98 seconds for vLLM-ROCm in the same comparison. |

## Scheduling

| Feature                                  | Level                    | Note                                                                                                 |
| ---------------------------------------- | ------------------------ | ---------------------------------------------------------------------------------------------------- |
| Continuous batching with chunked prefill | Supported                | One engine thread, with a bounded number of running requests.                                        |
| Overlap scheduling                       | Supported, on by default | Measured TTFT 1.00 times, tokens per second +1.2 % (Llama) and +3.6 % (OLMoE).                       |
| Scheduling policies                      | Supported                | `default` (default) and `cache_aware`, which has a starvation bound.                                 |
| Deterministic simulator                  | Supported                | Arrivals, overload, pipeline and workloads, without a GPU. Tests `sim`, `kv_sim` and `overload_sim`. |
| Bounded queues everywhere                | Supported                | The HTTP-to-engine channel and the admission queue are both bounded.                                 |

## Reliability

| Feature                                             | Level                   | Note                                                                                                                     |
| --------------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Pressure states with hysteresis                     | Supported               | GREEN, YELLOW, ORANGE, RED, SURVIVAL.                                                                                    |
| Circuit breaker                                     | Supported               | HEALTHY, DEGRADED, CIRCUIT_OPEN, DRAINING, PROBING. Answers 503 `circuit_open`. A sticky device error exits with code 3. |
| Predictive admission with worst-case KV reservation | Supported               | 429 `queue_full`, 503 `queue_timeout`, 503 `overloaded`.                                                                 |
| Emergency VRAM reserve                              | Supported               | Released only in SURVIVAL.                                                                                               |
| Host memory reserve                                 | Supported               | `reliability.memory.host_reserve_bytes`, 8 GiB by default.                                                               |
| SURVIVAL liveness                                   | Supported               | `requeue_unstarted` (default) or `continue_prefills`.                                                                    |
| Fault-injection build                               | Supported, test feature | Pool reservation failures, device out-of-memory errors and kernel errors. A default build refuses the section.           |
| Graceful shutdown                                   | Supported               | 503 `shutting_down`, then `server.shutdown_grace`, then exit 0.                                                          |
| Overload soak                                       | Supported               | Ten minutes per phase; four hours per release.                                                                           |

See [Reliability](guide/reliability.md).

## Speculative decoding

| Feature                                               | Level        | Note                                                                                                                                                                                                                  |
| ----------------------------------------------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Draft model, Llama-3.2-1B for Llama-3.2-3B on gfx1201 | Experimental | The gate missed its 1.3 times decode bar at 0.937 times, with 17.1 % acceptance. Golden compare, the A/B, TTFT, concurrency 16, the comparison with vLLM and the soak passed. Accepted as experimental on 2026-10-27. |
| Draft on the CPU backend                              | Experimental | Host tests.                                                                                                                                                                                                           |
| Any other draft combination                           | Unsupported  | Only the validated pairing is served.                                                                                                                                                                                 |
| Draft with tensor or pipeline parallelism above 1     | Unsupported  | `speculative_parallel_unsupported`, exit 2. An undersized budget fails with `speculative_budget`, exit 1.                                                                                                             |
| Multi-token-prediction (MTP) heads                    | Planned      | Not served.                                                                                                                                                                                                           |

See [Speculative decoding](guide/speculative-decoding.md).

## Observability

| Feature                           | Level     | Note                                                                                                                                                                |
| --------------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Prometheus `/metrics`             | Supported | Admission, circuit, pressure, KV, iteration time, HTTP, build information, support status, speculation, jump-forward and overlap. Families are prefixed `turbine_`. |
| Structured logs with reason codes | Supported | Events such as `support_matrix`, `quant_refused`, `kv_promotion_copy`, `paged_attention_fallback`, `module_selected` and `startup_refused`.                         |
| Status document                   | Supported | `/turbine/v1/status` reports the support row, modules, kernels, quantization, ladder and speculation.                                                               |
| Host share of the iteration       | Supported | `turbine-bench stage-share`.                                                                                                                                        |
| Golden numerics tooling           | Supported | `turbine-golden compare`, `capture` and `positions`, and an operation-by-operation trace.                                                                           |

See [Observability](guide/observability.md).

## Engineering properties

| Property                      | Status            | Evidence                                                                                                                                         |
| ----------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| No Python in the serving path | Enforced          | The serving path is Rust from HTTP to the kernel FFI. Python is used only for offline fixture scripts under `scripts/golden` and `scripts/eval`. |
| Declared support              | Enforced          | Every combination maps to a row with a status and a reason. Unsupported rows are refused before any port binds.                                  |
| Unsafe code confined          | Enforced by tests | Only `turbine-kernels` and `turbine-device` contain `unsafe`. Tests in `crates/turbine-kernels/tests/` check it.                                 |
| Vendor-neutral kernel ABI     | Enforced by tests | `kernels/include/turbine_kernels.h` with additive minor versions.                                                                                |
| Reuse of existing kernels     | Practice          | Composable Kernel and hipBLASLt are used before custom kernels.                                                                                  |
| Golden correctness gate       | Practice          | Strict at concurrency 1; batched bounds at concurrency 16. See [Testing and quality gates](testing.md).                                          |
