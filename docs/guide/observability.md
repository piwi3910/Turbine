# Observability: metrics, logs and status

Turbine reports through three channels: Prometheus metrics on `/metrics`, structured log events, and the status and pressure documents under `/turbine/v1/*`. Every automatic decision carries a reason code, and most decisions are visible in all three.

## Prometheus metrics

`GET /metrics` serves the Prometheus text format. Metric names below are the names registered in the code. Counters are exposed with a `_total` suffix, and histograms with the usual `_bucket`, `_sum` and `_count` series. The list covers the families Turbine registers, not every label value.

Example scrape configuration:

```yaml
scrape_configs:
  - job_name: turbine
    metrics_path: /metrics
    static_configs:
      - targets: ["127.0.0.1:8000"]
```

### Requests and tokens

- `turbine_requests`: requests.
- `turbine_request_ttft_seconds`, `turbine_request_itl_seconds`, `turbine_request_e2e_seconds`: time to first token, inter-token latency and end-to-end latency.
- `turbine_tokens`: tokens generated.
- `turbine_stream_paused`: streams paused by a slow client.
- `turbine_http_requests` and `turbine_http_request_duration_seconds`: the HTTP layer.
- `turbine_build_info`: version and build information.

### Scheduling and engine

- `turbine_requests_active` and `turbine_requests_queued`: running and waiting requests.
- `turbine_queue_wait_seconds`: time spent in the admission queue.
- `turbine_iteration_seconds`, `turbine_engine_iteration_seconds`: duration of each engine iteration.
- `turbine_iteration_tokens` and `turbine_batch_requests`: batch composition.
- `turbine_preemptions` and `turbine_requests_cancelled`.
- `turbine_logits_rows`: logits rows sampled.
- `turbine_decode_graph`: decode-graph captures and replays.
- `turbine_overlap_serial_turns`: iterations that ran serially under overlap scheduling.
- `turbine_jump_forward_steps` and `turbine_jump_forward_tokens`: jump-forward structured output.

### Reliability

- `turbine_pressure_state`, `turbine_pressure_transitions`, `turbine_pressure_dwell_seconds`, `turbine_pressure_signal`, `turbine_pressure_signal_level` and `turbine_pressure_exhaustion_horizon_seconds`: the pressure controller.
- `turbine_admission_decisions`, `turbine_admission_queue_depth` and `turbine_admission_queue_wait_seconds`: admission.
- `turbine_circuit_state` and `turbine_circuit_transitions`: the circuit breaker, transitions labelled by reason.
- `turbine_request_failures`: failed requests, by reason.
- `turbine_device_budget_bytes`, `turbine_memory_pool_bytes`, `turbine_emergency_reserve_held` and `turbine_emergency_reserve_releases`: memory accounting.
- `turbine_throttle_plan` and `turbine_reclaim_bytes`.
- `turbine_allocation_failures`, `turbine_recoveries` and `turbine_recovery_retries`.
- `turbine_group_pressure_state` and `turbine_group_limiting_device`: pressure across a device group.

### KV cache

- `turbine_kv_blocks` and `turbine_kv_bytes`: blocks and bytes by tier and format.
- `turbine_kv_lookups`, `turbine_kv_prefix_cached_tokens`, `turbine_kv_prompt_tokens`: prefix lookups and cache hits.
- `turbine_kv_lossy_cached_tokens` and `turbine_kv_lossy_denied`: lossy reuse.
- `turbine_kv_partial_reuse`, `turbine_kv_partial_reused_tokens` and `turbine_kv_partial_entries`: token-granular reuse.
- `turbine_kv_promotions`, `turbine_kv_demotions`, `turbine_kv_evictions`, `turbine_kv_drops` and `turbine_kv_recompute_tokens`: tier movement.
- `turbine_kv_plans`: planner decisions.
- `turbine_kv_transfer_seconds`, `turbine_kv_transfer_bytes` and `turbine_kv_transfer_bandwidth_bytes_per_second`: transfers between tiers.
- `turbine_kv_prefetch` and `turbine_kv_sessions`.
- `turbine_kv_tier_degraded`: a tier that has been degraded.
- `turbine_storage_queue_depth` and `turbine_storage_latency_seconds`: L2 I/O.
- `turbine_kv_ladder_rung`, `turbine_kv_ladder_actions` (labels `tier`, `from`, `to`, `reason`) and `turbine_kv_ladder_action_seconds`: the compression ladder.
- `turbine_kv_copy_ahead`, `turbine_kv_queued_prefix_detached_blocks` and `turbine_kv_slab_reformat`.

### Speculative decoding

- `turbine_speculative_rounds`, `turbine_speculative_proposed_tokens`, `turbine_speculative_accepted_tokens`, `turbine_speculative_accepted_per_round` and `turbine_speculative_disabled`.

### Model and kernels

- `turbine_model_load_seconds` and `turbine_model_weight_bytes`: loading.
- `turbine_forward_seconds`: forward pass duration.
- `turbine_weight_format_info`: the detected weight format.
- `turbine_grammar_compile_seconds` and `turbine_token_mask_seconds`: structured output.
- `turbine_tool_calls` and `turbine_reasoning_tokens`.
- `turbine_kernel_provider_selected`: which kernel provider was chosen for each operation.
- `turbine_support_matrix_status{status}`: a gauge with the resolved support status, `supported`, `experimental` or `unsupported`.
- `turbine_hybrid_state_bytes` and `turbine_hybrid_state_snapshots`: recurrent state of hybrid models.

### Devices and host telemetry

- `turbine_devices`: the device inventory.
- `turbine_gpu_temperature_celsius`, `turbine_gpu_clock_mhz`, `turbine_gpu_power_watts`, `turbine_gpu_utilization_ratio`, `turbine_gpu_memory_bytes` and `turbine_gpu_throttle_active`.
- `turbine_host_memory_available_bytes`, `turbine_host_psi_memory_some_avg10` and `turbine_host_swap_in_pages_per_second`.
- `turbine_telemetry_stale` and `turbine_telemetry_call_duration_seconds`: health of the telemetry reads.

### Parallelism

- `turbine_parallel_info`, `turbine_topology_link_gbps`.
- `turbine_collective_duration_seconds`, `turbine_collective_bytes`, `turbine_collective_errors` and `turbine_collective_route`.
- `turbine_dp_routed` and `turbine_dp_routed_total`: requests the data-parallel router sent to each replica.
- `turbine_expert_rank_tokens` and `turbine_expert_imbalance_ratio`: expert load on MoE models.
- `turbine_moe_experts`.
- `turbine_pipeline_stage_duration_seconds` and `turbine_pipeline_bubble_ratio`.

## Structured log events

Log lines carry `event=` fields. Set the format with `logging.format` (`text` by default, or `json`) and the level with `logging.level` (`info` in the example).

| Event                      | When                                                                                      |
| -------------------------- | ----------------------------------------------------------------------------------------- |
| `support_matrix`           | At startup, with the resolved row. A WARN when the row is `experimental`.                 |
| `module_selected`          | A registered module was chosen for an extension point, with `point`, `name` and `reason`. |
| `startup_refused`          | Startup stopped before the port was bound, with a `reason` code.                          |
| `quant_refused`            | A checkpoint's quantization is refused at load.                                           |
| `paged_attention_fallback` | `kv.block_tokens` is not a multiple of 128 on HIP, so the slower kernel runs.             |
| `kv_promotion_copy`        | The promotion copy kernel is missing and the copy engine is used instead.                 |

Refusals and failures are logged with reason codes before the process exits, so a supervisor or a log search can find them.

## Status document

`GET /turbine/v1/status` is the one document that describes the running server. It reports:

- `support`: the resolved support row for the device, model, weights and KV format.
- `modules`: the module chosen for each extension point, such as the model family, the tool-call format, the reasoning parser, the scheduling policy and the eviction policy.
- `kernels`: every kernel choice made by the kernel registry.
- `quantization`: the detected weight format, `tier_formats` for each lower tier, and the `ladder` state.
- `speculative`: `method`, `draft_model_path`, `num_tokens`, `min_acceptance`, `active` and `acceptance`.

Use it to confirm what a running server is actually doing, which can differ from the configuration file when a row is resolved against a device.

## Other documents

- `GET /turbine/v1/pressure`: the pressure state and admission.
- `GET /turbine/v1/kv`: KV blocks per tier and format, cache counters and the ladder.
- `GET /turbine/v1/devices`: the device inventory and telemetry.
- `GET /turbine/v1/scheduler`: the scheduler state.
- `GET /turbine/v1/topology`: the device and link topology.

The routes are described in [OpenAI API and diagnostics](openai-api.md).

## Exit codes

Process exit codes tell a supervisor why the process stopped. The codes are listed in [Architecture](../architecture.md#exit-codes).

## Reason codes

Every refusal and failure has a snake_case reason code. The codes appear in error responses, in logs and, for startup refusals, in the `startup_refused` event. The codes that matter most for operations are these:

- Admission and load: `queue_full`, `queue_timeout`, `overloaded`, `circuit_open`, `shutting_down`, `loading_model`.
- Configuration: `kv_tq2_l0_refused`, `speculative_parallel_unsupported`, `combination_unsupported`.
- Quantization: `quant_scheme_unsupported`, `gptq_act_order`, `quant_moe_phase7`, `quant_shard_misaligned`, `quant_pipeline_unsupported`.
- KV transfer and copy: `kv_transcode_unavailable`, `promotion_copy_kernel_unavailable`.
- Parallelism and speculation budget: `collective_corrupt`, `speculative_budget`.
- Structured output: `invalid_json_schema`.
