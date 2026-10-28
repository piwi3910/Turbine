# Configuration reference

Turbine reads one YAML file, passed with `--config`. Any key can be overridden on the command line with `--set <dotted.key>=<value>`, where the value is YAML. The repository's [`examples/turbine.yaml`](https://github.com/piwi3910/Turbine/blob/main/examples/turbine.yaml) is the reference file: it sets every section, and its comments state the rules for each key.

## How the file is used

- `turbine-server --config <file> --check-config` validates the file, resolves the support row and exits. It does not read the model.
- `turbine-server --config <file> --set <key>=<value>` starts the server with overrides applied.
- An unknown key, an out-of-range value or an unregistered module name exits with code 2 before any port is bound. The error names the key.
- Durations are an integer followed by `ms`, `s`, `m` or `h` (for example `600s`). Byte sizes use binary suffixes such as `8MiB`, `64GiB` or `2GiB`.
- Module names (`model.tool_call_parser`, `execution.backend`, `scheduler.policy`, `kv.policy` and the rest) are looked up in the registries. An unknown name exits with code 2 and lists the registered names. See [extension points](../extending/README.md).

Two code defaults differ from the example file and are worth knowing:

- `server.listen` defaults to `0.0.0.0:8000` when the key is absent. The example binds `127.0.0.1:8000`.
- The admission and circuit-breaker keys are not in the example file. Their code defaults are listed under [Reliability](#reliability) below.

## server

| Key                          | Example          | Meaning                                                                      |
| ---------------------------- | ---------------- | ---------------------------------------------------------------------------- |
| `server.listen`              | `127.0.0.1:8000` | Address and port for the HTTP server.                                        |
| `server.max_request_bytes`   | `8MiB`           | Largest accepted request body.                                               |
| `server.request_timeout`     | `600s`           | Total request deadline. Expiry answers 504 `request_timeout`.                |
| `server.slow_client_timeout` | `30s`            | How long a request may wait on a full output channel before it is cancelled. |
| `server.shutdown_grace`      | `30s`            | Drain time for running requests after SIGINT or SIGTERM.                     |

## model

| Key                      | Example                         | Meaning                                                                                                                           |
| ------------------------ | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `model.path`             | `/models/llama-3.2-3b-instruct` | Model directory: must contain `config.json`, the safetensors weights and the tokenizer. Loaded at startup.                        |
| `model.dtype`            | `bf16`                          | Compute dtype.                                                                                                                    |
| `model.served_name`      | `null`                          | The id clients use. `null` takes the last path component, or `<org>/<name>` for a Hugging Face snapshot path.                     |
| `model.max_seq_len`      | `null`                          | Context limit. `null` is the smaller of 32768 and the model's `max_position_embeddings`.                                          |
| `model.tool_call_parser` | `null`                          | A tool-call format name (`llama3_json`, `hermes`, `mistral`, `harmony`, `qwen3_xml`), or `none`. `null` takes the family default. |
| `model.reasoning_parser` | `auto`                          | `qwen_think`, `harmony`, `none`, or `auto` for the family default. Fills `reasoning_content`.                                     |
| `model.chat_template`    | not set                         | Overrides the checkpoint's chat template with a file. Used to pin a template for reproducible tests.                              |
| `model.rope_scaling`     | not set                         | Overrides RoPE scaling, for example YaRN for a longer context.                                                                    |

The weight format is detected from `config.json`. You do not set it. See [Quantization](../guide/quantization.md).

## kv

KV cache settings. The tiers are described in [KV cache tiers and compression](../guide/kv-cache.md).

| Key                                | Example               | Meaning                                                                                                                                                                              |
| ---------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `kv.block_tokens`                  | `128`                 | Tokens per KV page. On HIP, a multiple of 128 runs Composable Kernel paged attention. Other sizes fall back to the slower Turbine kernel and log `event="paged_attention_fallback"`. |
| `kv.dtype`                         | `bf16`                | The L0 (GPU) format. `bf16` is exact. `fp8_e4m3` and `tq4` are lossy; `tq2` exits 2 (`kv_tq2_l0_refused`).                                                                           |
| `kv.gpu.enabled`                   | `true`                | L0 is required.                                                                                                                                                                      |
| `kv.gpu.max_bytes`                 | `null`                | L0 pool cap. `null` is the pool left by the memory budget.                                                                                                                           |
| `kv.cpu.enabled`                   | `true`                | L1 pinned host tier. Discrete-VRAM devices only. Otherwise a warning and no L1.                                                                                                      |
| `kv.cpu.max_bytes`                 | `64GiB`               | Ceiling for L1. Checked against host memory minus `reliability.memory.host_reserve_bytes`; exit 2 when it does not fit.                                                              |
| `kv.cpu.slab_bytes`                | `128MiB`              | Growth step of the L1 pool.                                                                                                                                                          |
| `kv.cpu.format`                    | `l0`                  | L1 format: `l0`, `fp8_e4m3`, `tq4` or `tq2`.                                                                                                                                         |
| `kv.nvme.enabled`                  | `false`               | Enables L2 slab files.                                                                                                                                                               |
| `kv.nvme.path`                     | `/var/lib/turbine/kv` | Absolute directory for slab files. Created if absent. Wiped at startup.                                                                                                              |
| `kv.nvme.max_bytes`                | `64GiB`               | Ceiling for L2, at most the free space at startup minus 10 %.                                                                                                                        |
| `kv.nvme.slab_bytes`               | `1GiB`                | Slab file size.                                                                                                                                                                      |
| `kv.nvme.max_queue_depth`          | `64`                  | I/O queue depth (code default 64).                                                                                                                                                   |
| `kv.nvme.io_threads`               | `4`                   | I/O threads for L2.                                                                                                                                                                  |
| `kv.nvme.format`                   | `l0`                  | L2 format, same choices as `kv.cpu.format`.                                                                                                                                          |
| `kv.policy`                        | `cost_aware`          | Eviction policy: `cost_aware` or `lru`.                                                                                                                                              |
| `kv.demote_min_value`              | `0.0`                 | Blocks scoring below this are dropped instead of demoted.                                                                                                                            |
| `kv.prefix_sharing`                | `true`                | `false` turns prefix lookup and reuse off.                                                                                                                                           |
| `kv.partial_prefix_reuse`          | `true`                | Token-granular reuse of the last prefix block through a private copy. Ignored when prefix sharing is off.                                                                            |
| `kv.transfer.max_inflight_bytes`   | `1GiB`                | Bound on bytes in flight between tiers.                                                                                                                                              |
| `kv.transfer.promotion_copy`       | not set               | `kernel` or `sdma`. Unset uses the kernel library's copy kernel when present.                                                                                                        |
| `kv.session.max_sessions`          | `10000`               | Bound on tracked sessions.                                                                                                                                                           |
| `kv.session.hot_ttl`               | `60s`                 | Idle time before a session demotes from L0 to L1.                                                                                                                                    |
| `kv.session.warm_ttl`              | `10m`                 | Idle time before a session demotes from L1 to L2. Must exceed `hot_ttl`.                                                                                                             |
| `kv.session.max_idle`              | `1h`                  | Idle time after which session metadata is dropped.                                                                                                                                   |
| `kv.prefetch.lead_time`            | `2s`                  | Lead time for prefetch promotions.                                                                                                                                                   |
| `kv.prefetch.max_queue`            | `256`                 | Bound on queued prefetches.                                                                                                                                                          |
| `kv.policy_weights.session_active` | `0.5`                 | Reuse boost for blocks of an active session, 0 to 1.                                                                                                                                 |
| `kv.policy_weights.hit_half_life`  | `60s`                 | Half-life of the hit signal used by the eviction policy.                                                                                                                             |
| `kv.lossless_tail_blocks`          | `1`                   | The newest full blocks of each sequence stay at the L0 format when they leave L0.                                                                                                    |
| `kv.lossy_reuse`                   | `allow`               | Whether requests that send no `x-turbine-kv-lossy` header may reuse lossy blocks. `deny` makes them opt in with `x-turbine-kv-lossy: allow`.                                         |
| `kv.lossy_penalty`                 | `null`                | Retrieval-cost multiplier per codec. Defaults: `fp8_e4m3` 0.1, `tq4` 0.5, `tq2` 1.0.                                                                                                 |
| `kv.recent_window_blocks`          | `1`                   | With `kv.dtype: tq4`, the newest full blocks of each live sequence that stay BF16.                                                                                                   |
| `kv.ladder.enabled`                | `false`               | Pressure-driven compression ladder. Off by default.                                                                                                                                  |
| `kv.ladder.l0`                     | `true`                | Puts L0 in the ladder. Set `false` for the cheaper L1/L2-only ladder.                                                                                                                |
| `kv.ladder.max_format`             | `tq4`                 | Lossiest rung the ladder may reach.                                                                                                                                                  |
| `kv.ladder.high_water`             | `0.95`                | An upper tier compresses above this fill.                                                                                                                                            |
| `kv.ladder.low_water`              | `0.85`                | At YELLOW the lowest tier compresses until its fill is back at this level.                                                                                                           |

## execution

| Key                            | Example | Meaning                                                                                                                             |
| ------------------------------ | ------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `execution.backend`            | `hip`   | Kernel backend: `hip` or `cpu`.                                                                                                     |
| `execution.device`             | `0`     | Device index from the inventory. Its vendor must match the backend.                                                                 |
| `execution.card_profile`       | `auto`  | Card profile. `auto` picks the profile of the device architecture.                                                                  |
| `execution.kernel_library`     | `null`  | Path to `libturbine_hip.so`. `null` uses the search order in [Install and build](install.md#locate-the-kernel-library-at-run-time). |
| `execution.gemm_autotune`      | `true`  | Uses the tuned GEMM table for the card. `false` takes hipBLASLt's first heuristic answer.                                           |
| `execution.decode_graphs`      | `true`  | Captures decode-only iterations into graphs and replays them.                                                                       |
| `execution.device_sampling`    | `true`  | Reduces logits on the device when the request allows it.                                                                            |
| `execution.fused_ops`          | `true`  | Fused QKV and gate/up projections and fused residual add with RMSNorm.                                                              |
| `execution.sampler_threads`    | `4`     | Threads that sample rows in parallel (1 to 64).                                                                                     |
| `execution.overlap_scheduling` | `true`  | Prepares the next iteration on the host while the GPU runs the current one.                                                         |

## scheduler

| Key                              | Example   | Meaning                                                                               |
| -------------------------------- | --------- | ------------------------------------------------------------------------------------- |
| `scheduler.continuous_batching`  | `true`    | `false` forces one running request.                                                   |
| `scheduler.chunked_prefill`      | `true`    | `false` rejects prompts longer than `max_batch_tokens`.                               |
| `scheduler.max_running_requests` | `64`      | Requests running at once. Further requests wait in the admission queue.               |
| `scheduler.max_batch_tokens`     | `8192`    | Per-iteration token budget (decodes plus prefill chunks).                             |
| `scheduler.prefill_chunk_tokens` | `2048`    | Largest prefill chunk per iteration.                                                  |
| `scheduler.max_queued_requests`  | `256`     | Capacity of the HTTP-to-engine channel. The waiting queue is `reliability.admission`. |
| `scheduler.policy`               | `default` | Scheduling policy: `default` or `cache_aware`.                                        |
| `scheduler.cache_aware_window`   | `1s`      | Ordering window of the `cache_aware` policy.                                          |

## structured_output

| Key                                         | Example | Meaning                                                                |
| ------------------------------------------- | ------- | ---------------------------------------------------------------------- |
| `structured_output.max_schema_bytes`        | `64KiB` | Largest accepted JSON schema or tool definition (1 KiB to 1 MiB).      |
| `structured_output.compile_timeout`         | `5s`    | Longest grammar compilation. Expiry answers 400 `invalid_json_schema`. |
| `structured_output.jump_forward`            | `true`  | Forced tokens are decoded without a forward pass.                      |
| `structured_output.jump_forward_max_tokens` | `32`    | Longest forced run in one step (1 to 256).                             |

## parallel

| Key                                | Example           | Meaning                                                                             |
| ---------------------------------- | ----------------- | ----------------------------------------------------------------------------------- |
| `parallel.tensor_parallel_size`    | `1`               | Tensor parallel degree, a power of two that divides the attention heads, or `auto`. |
| `parallel.data_parallel_size`      | `1`               | Data-parallel replicas, 1 to 64.                                                    |
| `parallel.pipeline_parallel_size`  | `1`               | Pipeline stages. Local ranks only.                                                  |
| `parallel.expert_parallel_size`    | `1`               | Expert parallel degree for MoE families.                                            |
| `parallel.devices`                 | `auto`            | `auto`, or a list of device indices.                                                |
| `parallel.collective_backend`      | `auto`            | `auto` (`rccl` on AMD, `host` for one device), `rccl`, `nccl` or `host`.            |
| `parallel.router`                  | `prefix_affinity` | Data-parallel router: `prefix_affinity` or `least_loaded`.                          |
| `parallel.plan_queue_depth`        | `2`               | Bound on queued step plans between the leader and workers (1 to 16).                |
| `parallel.collective.init_timeout` | `120s`            | Bound on communicator setup.                                                        |
| `parallel.collective.op_timeout`   | `30s`             | Bound on one collective step.                                                       |
| `parallel.collective.verify`       | `false`           | Checksums every collective across ranks. A diagnostic mode; it is slow.             |
| `parallel.ranks.mode`              | `local`           | `local`: one thread per device in this process. `static`: one process per rank.     |
| `parallel.ranks.transport`         | `tcp`             | Rank transport for static mode.                                                     |

Accepted and refused combinations are described in [Multi-GPU parallelism](../guide/parallelism.md).

## speculative

| Key                            | Example | Meaning                                                                 |
| ------------------------------ | ------- | ----------------------------------------------------------------------- |
| `speculative.method`           | `none`  | `none` or `draft`.                                                      |
| `speculative.num_tokens`       | `4`     | Draft tokens proposed per round (1 to 8).                               |
| `speculative.draft_model_path` | `null`  | Draft checkpoint directory. Required with `draft`.                      |
| `speculative.min_acceptance`   | `0.4`   | A request whose acceptance falls below this stops speculating (0 to 1). |

## reliability

| Key                                          | Default             | Meaning                                                                                                     |
| -------------------------------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------- |
| `reliability.enabled`                        | `true`              | Turns the reliability layer on.                                                                             |
| `reliability.emergency_vram_reserve`         | `2GiB`              | VRAM held back from normal scheduling. Released only in SURVIVAL.                                           |
| `reliability.adaptive_admission`             | `true`              | Enables adaptive admission.                                                                                 |
| `reliability.memory.host_reserve_bytes`      | `8GiB`              | Host memory kept free of the L1 tier.                                                                       |
| `reliability.pressure.escalate_samples`      | `2`                 | Consecutive samples over a threshold before the state escalates.                                            |
| `reliability.pressure.deescalate_dwell`      | `10s`               | Time below a threshold before the state steps back down. Must be at least the sampling interval.            |
| `reliability.admission.max_queue`            | `256`               | Admission queue capacity. A full queue answers 429 `queue_full`.                                            |
| `reliability.admission.queue_timeout`        | `30s`               | Longest wait in the queue. Expiry answers 503 `queue_timeout`.                                              |
| `reliability.admission.large_prefill_tokens` | `2048`              | Prefills above this many new tokens count as expensive. Must be at least `kv.block_tokens`.                 |
| `reliability.admission.max_bypass`           | `8`                 | Smaller requests that may overtake a queue head that does not fit.                                          |
| `reliability.recovery.max_retries`           | `3`                 | Retries after an allocation failure (0 to 10).                                                              |
| `reliability.recovery.backoff`               | `50ms`              | Delay between retries.                                                                                      |
| `reliability.recovery.survival_liveness`     | `requeue_unstarted` | How SURVIVAL releases KV: `requeue_unstarted` or `continue_prefills`.                                       |
| `reliability.circuit.oom_recoveries_to_open` | `3`                 | Recovered out-of-memory events that open the circuit.                                                       |
| `reliability.circuit.cooldown`               | `30s`               | Time the circuit stays open before probing.                                                                 |
| `reliability.circuit.probe_successes`        | `3`                 | Successful probes that close the circuit.                                                                   |
| `reliability.fault_injection`                | not set             | Fault-injection section. Refused with exit 2 in a default build; needs the `fault-injection` build feature. |

The states and their effects are in [Reliability](../guide/reliability.md).

## logging, devices and quality

| Key                         | Example | Meaning                                                                        |
| --------------------------- | ------- | ------------------------------------------------------------------------------ |
| `logging.format`            | `text`  | `text` or `json`.                                                              |
| `logging.level`             | `info`  | Log level.                                                                     |
| `devices.nvml_library`      | `null`  | Explicit NVML library for device discovery. `null` uses the default search.    |
| `devices.amd_smi_library`   | `null`  | Explicit amd-smi library for device discovery. `null` uses the default search. |
| `quality.max_accuracy_drop` | `0.01`  | Accuracy budget used by the quality comparison tools.                          |

## Next

- [Quickstart](quickstart.md): a worked example that uses these keys.
- [Observability](../guide/observability.md): the status document that reports the resolved configuration.
