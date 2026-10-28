# KV cache tiers and compression

Turbine stores attention keys and values in fixed-size blocks, and moves those blocks between tiers instead of discarding them. This page describes the tiers, how requests reuse blocks, the formats a tier can store, and the pressure-driven compression ladder.

## Tiers

| Tier      | Where                                     | Configured by                         | Status                                                                                                                                     |
| --------- | ----------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| L0        | GPU memory, paged block pool              | `kv.block_tokens`, `kv.gpu.max_bytes` | Supported                                                                                                                                  |
| L1        | Pinned host memory, grown in slabs        | `kv.cpu.*`                            | Supported on discrete-VRAM devices with a kernel library that exports the copy streams (ABI v2.5 or later). Otherwise a warning and no L1. |
| L2        | NVMe slab files under `kv.nvme.path`      | `kv.nvme.*`                           | Supported. Files are wiped at startup.                                                                                                     |
| Recompute | Not stored; the prompt is prefilled again | planner                               | Supported. A planner weighs the cost of retrieving a block from a tier against the cost of recomputing it.                                 |

L0 is required. The example configuration enables L1 and leaves L2 off.

### Blocks and pages

`kv.block_tokens` (128 by default) is the number of tokens per page. On HIP, a multiple of 128 runs paged attention on Composable Kernel's `fmha_fwd_pagedkv`, and for grouped-query models such as Llama it runs `fmha_fwd_splitkv`, which shares each K/V page across the query heads that use it. Other sizes fall back to Turbine's own paged-attention kernel, which is slower, and the server logs `event="paged_attention_fallback"` at startup.

Block tables map each sequence's logical positions to pages in whatever tier holds them.

### Host and NVMe tiers

```yaml
kv:
  cpu:
    enabled: true
    max_bytes: 64GiB # checked against host memory minus reliability.memory.host_reserve_bytes
    slab_bytes: 128MiB # growth step of the L1 pool
  nvme:
    enabled: true
    path: /var/lib/turbine/kv # absolute; created if absent; wiped at startup
    max_bytes: 64GiB # at most the free space at startup minus 10 %
    slab_bytes: 1GiB
```

An L1 `max_bytes` above host memory minus `reliability.memory.host_reserve_bytes` is refused with exit code 2. The L2 size is limited by free disk space at startup and exits with code 1 when the configured size cannot be held. Nothing persists across restarts: L2 is a spill area, not a cache that survives a process.

## Reuse

Requests share full-block prefixes globally by default. Three mechanisms control reuse:

- **Full-block prefix sharing.** A block whose contents match an existing block is reused. Disable it with `kv.prefix_sharing: false` to A/B test.
- **Token-granular reuse.** With `kv.partial_prefix_reuse: true` (the default), the last, partly filled prefix block is reused through a private copy-on-write tail. A request that shares 1000 tokens with an earlier one reuses all 1000, not only the full blocks.
- **Isolation by salt.** A request with `x-turbine-cache-salt` shares prefixes only with requests carrying the same salt.

The number of tokens served from cache is reported in `usage.prompt_tokens_details.cached_tokens`. The counters `turbine_kv_prefix_cached_tokens_total` and `turbine_kv_prompt_tokens_total` give the ratio over time.

## Sessions and prefetch

Clients that talk in multi-turn sessions can pass hints:

- `prompt_cache_key` names the session.
- `x-turbine-session-resume-within` (1 to 86400 seconds) keeps the session's blocks boosted after a response.
- `x-turbine-session-end: true` releases the session after its last response.

Idle sessions move down the tiers on timers: `kv.session.hot_ttl` (60 s) from L0 to L1, `kv.session.warm_ttl` (10 min) from L1 to L2, and `kv.session.max_idle` (1 h) after which the metadata is dropped. `kv.session.max_sessions` (10000) bounds the number of tracked sessions.

`POST /turbine/v1/kv/prefetch` queues promotions of a prefix into L0 ahead of a request. The queue is bounded by `kv.prefetch.max_queue` (256), and a prefetch is refused with 409 `pressure_too_high` at ORANGE or above.

## Eviction

`kv.policy` selects the eviction policy. `cost_aware` (the default) ranks blocks by their reuse value against the cost of retrieving or recomputing them. `lru` is kept as the benchmark baseline. Blocks scoring below `kv.demote_min_value` are dropped instead of demoted.

Policies are pluggable. See [Eviction policy](../extending/eviction-policy.md).

## Formats

A tier stores its blocks in a format from the KV format registry. The L0 format is set by `kv.dtype`, and each lower tier has its own format: `kv.cpu.format` and `kv.nvme.format`. A lower tier may not store a format more precise than the one above it.

| Format                    | As the L0 format (`kv.dtype`)                                                                                                                                                          | As a lower-tier format                                                                                                                                | Status and evidence |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| `l0` (BF16, exact)        | default                                                                                                                                                                                | default                                                                                                                                               | Supported.          |
| `fp8_e4m3`                | Supported for Llama with BF16 weights (full GSM8K at concurrency 16: 0.7885 against 0.7801 for BF16) and for OLMoE with calibrated scales (GSM8K drop 0.0030). Other rows are refused. | Supported. Shared-prefix GSM8K median drop 0.005, against a budget of 0.01.                                                                           | Lossy               |
| `tq4` (TurboQuant, 4-bit) | Supported for Llama on gfx1201. Experimental for OLMoE, whose shared-prefix pair drops 0.045. Experimental on the CPU backend.                                                         | Supported. Multi-turn cached ratio on Llama 0.9075 against 0.9007 for `l0`, on OLMoE 0.8625 against 0.8624.                                           | Lossy               |
| `tq2` (TurboQuant, 2-bit) | Refused with exit code 2, reason `kv_tq2_l0_refused`.                                                                                                                                  | Experimental. It fails the shared-prefix eval (Llama 0.720, OLMoE 0.610 against 0.780 and 0.635 for BF16). It is a ladder rung, not a serving format. | Lossy               |

Lower tiers store lossy formats only when the kernel library provides the ABI v2.11 KV transcode. Without it, a lossy `kv.cpu.format` or `kv.nvme.format` exits with code 1, reason `kv_transcode_unavailable`.

The support matrix is the authority for each row. See the [feature matrix](../features.md#kv-formats-l0-and-lower-tiers).

### Lossy blocks

Blocks stored in a lossy format carry lineage keys, so the server knows which cached blocks are lossy:

- `kv.lossy_reuse` (`allow` by default) decides whether requests that send no header may reuse them. `deny` makes them opt in.
- `x-turbine-kv-lossy: deny` on one request forbids its reuse, and `allow` permits it. Any other value is refused with 400.
- `kv.lossy_penalty` multiplies the retrieval cost of a lossy block when the planner chooses between retrieving and recomputing. Defaults: `fp8_e4m3` 0.1, `tq4` 0.5, `tq2` 1.0.
- `kv.lossless_tail_blocks` (1 by default) keeps the newest full blocks of each sequence at the L0 format when they leave L0.
- `usage.prompt_tokens_details.lossy_cached_tokens` reports the cached tokens that came from lossy blocks.

### L0 `tq4`

`kv.dtype: tq4` stores every L0 block as TurboQuant except the newest `kv.recent_window_blocks` (1 by default) full blocks of each live sequence, which stay BF16. A block that leaves the window is recompressed in place. Support is per row: see the feature matrix.

## The compression ladder

The ladder moves a tier's oldest and least reusable blocks one format down when memory pressure rises:

```text
l0 -> fp8_e4m3 -> tq4 -> tq2 -> evict
```

It is off by default:

```yaml
kv:
  ladder:
    enabled: false # opt in
    l0: true # true: L0 takes part in the ladder; false: L1 and L2 only
    max_format: tq4 # the lossiest rung the ladder may reach
    high_water: 0.95 # an upper tier compresses above this fill
    low_water: 0.85 # at YELLOW the lowest tier compresses only down to this fill
```

The ladder starts at YELLOW and under capacity pressure, rewrites at most 32 blocks per tick, and steps a rung back up only at GREEN, after `reliability.pressure.deescalate_dwell` below `low_water`.

The ladder trades throughput for memory. The measured A/B runs on the R9700 with the setup described in [Performance](../performance.md) were:

- With `kv.ladder.l0: true`, 0.909 times the tokens per second of the ladder-off arm, 1.69 times as many samples at ORANGE, and 6.2 % fewer recomputed tokens.
- With `kv.ladder.l0: false` (L1 and L2 rungs only), 0.952 times the tokens per second, with inter-token latency equal to the ladder-off arm.
- In a multi-turn run, the L0 step recomputed 23 % fewer tokens and raised the cached ratio, but lowered tokens per second and lengthened later-turn tail latency. The serving-cost work on the L0 step is still open.

Use the ladder when memory, not latency, is the binding constraint.

## Metrics and documents

- `GET /turbine/v1/kv` reports the blocks per format per tier, the cache and transfer counters, and the ladder state.
- `GET /turbine/v1/status` reports `quantization.tier_formats` and `quantization.ladder`.
- Metrics include `turbine_kv_blocks`, `turbine_kv_bytes`, `turbine_kv_lookups`, `turbine_kv_promotions`, `turbine_kv_demotions`, `turbine_kv_evictions`, `turbine_kv_drops`, `turbine_kv_recompute_tokens`, `turbine_kv_transfer_seconds`, `turbine_kv_transfer_bytes`, `turbine_kv_ladder_rung`, `turbine_kv_ladder_actions` (labels `tier`, `from`, `to`, `reason`), `turbine_kv_lossy_cached_tokens` and `turbine_kv_lossy_denied`. The full list is in [Observability](observability.md).

## Promotion copies

Moving a block from pinned host memory to L0 uses the kernel library's copy kernel when the library provides ABI v2.11. That kernel uses two workgroups and does not stall decode. Measured on the R9700, it moves data about 1.8 times faster than the copy engine. `kv.transfer.promotion_copy: sdma` forces the copy engine, and `kernel` on a library without the kernel exits with code 1 (`promotion_copy_kernel_unavailable`). Left unset, a library without the kernel falls back to the copy engine with a warning (`event="kv_promotion_copy"`).

## Offline simulation

`turbine-bench kv-sim` replays a workload through the policies without a GPU:

```sh
cargo run --release -p turbine-bench --bin turbine-bench -- kv-sim \
  --workload multi-turn --policy cost_aware --l0-blocks 256 --l1-blocks 1024 --seed 1 --output json
```

The workloads are `multi-turn`, `shared-system` and `mixed`. `--policy` takes a registered eviction policy. `--seed` fixes the generated workload. Results can still differ between runs in attach order, a known follow-up in the simulator.
