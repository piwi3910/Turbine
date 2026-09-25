# phase-4-kv-intelligence

Status: complete

Source: `turbine-spec.md` §19 Phase 4 (KV intelligence), with §8 (hierarchical KV subsystem), §14 (observability), §16 (security), §17 (testing), §20 (definition of done) and §21 (engineering rules). Sections of that document are cited as "TS §N". Decisions and their options are recorded in `.procoder/ask/decisions.md`. Builds on `phase-2-serving-runtime` (GPU paged KV) and `phase-3-reliability` (memory budget, pressure states, `KvReclaimer` hook, admission `ResourceEstimate`); where this spec depends on a question those specs leave open, it says so instead of re-asking.

## Problem

After Phase 3 the engine survives overload, but KV is still disposable GPU memory: every request re-prefills its whole prompt even when the previous turn of the same conversation or a thousand other requests share the same system prompt, finished sequences' KV is thrown away, and the only answer to KV pressure is to queue work or free blocks. On the lab's workloads (agentic multi-turn chat with long shared prefixes; Llama-3.2-3B-Instruct runs up to 128k context) prefill dominates TTFT and GPU time, and vLLM already reuses prefixes in GPU memory. TS §1 pillar 1 and TS §20 require more: KV as a managed, tiered resource — a working pinned-CPU tier, safe prefix sharing, cost-aware eviction instead of LRU, a planner that chooses between retrieving KV from a slower tier and recomputing it, session-aware retention and prefetch, and then an NVMe tier. Phase 4 builds that subsystem with the separate boundaries TS §21 rule 6 demands (identity, directory, tiers, policy, planner, transfer, metrics) and wires it into Phase 3's admission and reclaim hooks so that "demote KV" in the degradation ladder becomes real. The primary hardware is novanas's discrete-VRAM R9700 (gfx1201, 32 GB), where L0 (VRAM) and L1 (pinned host RAM) are physically separate; the unified-memory GB10 Sparks follow, with L1 disabled because there it would be the same memory as L0.

## Users

- **API clients running multi-turn and agentic workloads:** need lower TTFT on turns that share a prefix with earlier requests, with outputs identical to a cold run, and a way to say "this is session X, I will be back" (session hints) without leaving the OpenAI SDKs.
- **Operators:** need `/turbine/v1/kv` and metrics showing bytes/blocks and hit rate per tier, prefix reuse, promotions/demotions/evictions/recomputes, transfer latency/bandwidth and prefetch effectiveness (TS §14), and a few config keys to size and enable tiers.
- **Turbine developers:** need each KV boundary testable on macOS without a GPU (in-memory tier doubles, fake clock, trace-driven policy simulation — TS §17 items 1 and 3), GPU round-trip tests on the lab hosts, and a pluggable eviction policy they can benchmark against LRU (TS §8).
- **Benchmark runners:** need a multi-turn/shared-prefix load profile in `turbine-bench` and an offline policy simulator to compare hit rate and recompute cost across policies.

## In scope

- [S-1] KV identity in `turbine-kv` (module `identity`): a 128-bit block key (type `KvKey`; `BlockKey` is an alias of it, and the `KvBlock` model field is the model fingerprint — CONFLICT C-17) computed as a hash chain over the parent block key and the block's token ids, rooted in a namespace key derived from model identity (model path's resolved config hash and weights index hash), KV format (dtype, layout, _kv.block_tokens_, layer/head/head-dim description from the model) and the cache salt. Prefix sharing is global by default (empty salt: every request can reuse every other request's prefixes); a request may opt into isolation by sending a cache salt in the `x-turbine-cache-salt` header, which enters the namespace key so only requests carrying the same salt share blocks (TS §16 isolation hook). Only full blocks are keyed and shareable. Each cached block stores its token ids, and a lookup verifies them, so a hash collision is a miss, never wrong KV.
- [S-2] Local KV directory (module `directory`): block key → metadata (`KvBlock` of TS §8: token range, format, size, locations per tier, access count, decayed hit rate, last access, ref count, priority, recompute cost, session id, parent key, child count), with longest-prefix lookup across all tiers for a token sequence. The directory is the only owner of block metadata; tiers own bytes.
- [S-3] Prefix sharing in L0 (TS §20 "share reusable prefixes safely"): at admission a request's prompt is matched against the directory; matched full blocks are attached to the sequence by reference (ref count + 1) instead of prefilled; shared blocks are immutable and a sequence only writes its private tail block. At least one prompt token is always prefilled so the first output token has logits. `cached_prefix_tokens` feeds Phase 3's `ResourceEstimate`, and `usage.prompt_tokens_details.cached_tokens` is returned in OpenAI responses.
- [S-4] Tier abstraction (module `tier`): a `KvTier` trait with capacity, used bytes, pressure level, estimated latency and bandwidth, and `contains / put / get / evict` over block slots, implemented by L0 (the Phase 2 GPU block pool), L1 (pinned host memory) and L2 (NVMe); plus an in-memory test tier with injectable latency, bandwidth and faults. Hot-path GPU operations (attaching L0 blocks to a batch) stay synchronous and do not go through async trait methods (TS §8).
- [S-5] L1 pinned CPU tier: page-locked host memory allocated through the kernel shim C ABI v3 (`hipHostMalloc` inside the HIP shim, `cudaHostAlloc` inside the CUDA shim; Rust wrappers `PinnedMemory`/`CopyEngine` implemented for the shim context in `turbine-kernels`; `turbine-device` gains no vendor-runtime binding — CONFLICT C-6), grown lazily in 1 GiB slabs up to `kv.cpu.max_bytes` and released slab-by-slab under Phase 3 host-memory pressure (RED and above) when a slab is empty. L1 exists only on discrete-VRAM devices (R9700). On unified-memory devices (GB10) L1 is disabled: the L1 settings (`kv.cpu.enabled`, `kv.cpu.max_bytes`) are ignored with one WARN at startup, the tier reports `enabled: false`, and L0 demotes straight to L2 (or drops when L2 is disabled).
- [S-6] Transfer engine (module `transfer`): asynchronous block copies L0↔L1 on a dedicated copy stream per device (discrete devices), L0↔L2 through a bounce buffer on unified devices with completion events (copy streams, async memcpy and events come from kernel C ABI v3 in both shims, CONFLICT C-6), and L1↔L2 file I/O on a bounded thread pool; in-flight bytes bounded by `kv.transfer.max_inflight_bytes`; per-path bandwidth and latency EWMAs, seeded at startup by a 64 MiB calibration copy per path and logged.
- [S-7] Cost-aware eviction (module `policy`, TS §8): a pluggable `EvictionPolicy` trait scoring each evictable block; the default `cost_aware` policy implements value ≈ reuse_probability × recompute_cost × priority × retrieval_cost / memory_cost (amends TS §8, which divides by retrieval_cost — user decision 2026-09-25: blocks cheap to bring back are evicted first) with the concrete terms under Interfaces; an `lru` policy exists as the benchmark baseline. Selected by _kv.policy_. Blocks with ref count > 0 are never evicted; a block is not evicted from a tier while a descendant of it is still cached in the same or a faster tier (leaf-first).
- [S-8] Demotion and promotion: under L0 pressure (Phase 3 YELLOW/ORANGE `KvReclaimer::demote`, or an allocation needing blocks) the lowest-value unreferenced L0 blocks are copied to L1 (or, where L1 is disabled on unified memory, to L2; or dropped when their value is below _kv.demote_min_value_ or no lower tier has room) and then freed; L1 victims go to L2 when enabled, else are dropped. Promotion copies a block to L0 before the sequence's next prefill chunk that needs it. A block may have copies in several tiers; the directory tracks all of them.
- [S-9] Recompute-vs-retrieve planner (module `planner`, TS §8 "recompute as a virtual tier"): for a request's matched prefix, choose a cutoff k minimising cost(k) = Σ retrieval cost of blocks 0..k from their fastest tier + recompute cost of tokens k..n, using measured transfer bandwidths and the Phase 3 prefill-throughput EWMA; retrieval is also rejected when L0 cannot take the blocks at the current pressure state. Every plan is logged with its inputs.
- [S-10] Sessions and prefetch: a request may carry a session id in OpenAI's `prompt_cache_key` body field, plus the optional hints `x-turbine-session-resume-within` and `x-turbine-session-end` headers (semantics under Interfaces); the session table (bounded by `kv.session.max_sessions`) records the session's tail block key, last activity and an EWMA of inter-turn gaps. Session blocks get the `session_active` boost while the session is hot; idle sessions demote L0→L1 after `kv.session.hot_ttl` (L0→L2 on unified devices) and L1→L2 after `kv.session.warm_ttl`. Prefetch (L2/L1 → L0) starts on: arrival of a session request while it is queued, an explicit `POST /turbine/v1/kv/prefetch`, or predicted resume (last activity + gap EWMA − lead time) when pressure is GREEN or YELLOW. Prefetches are cancellable and counted as used or wasted.
- [S-11] L2 NVMe tier (TS §19 "then NVMe tier"), part of Phase 4: fixed-size slab files under `kv.nvme.path` (on the lab hosts `/home/piwi/turbine-kv` on the root NVMe, `kv.nvme.max_bytes` 64 GiB per host) opened with `O_DIRECT`, 4 KiB-aligned block slots, an in-memory slot index, a CRC32C per block verified on read (mismatch → drop and recompute), bounded queue depth, capacity `kv.nvme.max_bytes`, and the directory wiped at startup (not persistent across restarts). Storage queue depth, latency and bandwidth feed Phase 3 as new pressure signals.
- [S-12] Cancellation and failure containment: cancelling a request drops its block references and cancels its queued promotions/prefetches; transfers already in flight complete into the cache or are discarded, never into the cancelled sequence; a tier I/O or copy error marks that tier degraded, falls back to recompute and never fails the request.
- [S-13] Diagnostics: `GET /turbine/v1/kv` returns the KV document (see Data) instead of 501; `POST /turbine/v1/kv/prefetch` is added under the diagnostics prefix.
- [S-14] KV metrics and structured logs per TS §14 (list under Interfaces).
- [S-15] Configuration keys under Interfaces, validated at startup.
- [S-16] Benchmarks: `turbine-bench` multi-turn profile (sessions × turns with a shared system prefix, think time, session hints via `prompt_cache_key`) and an offline `turbine-bench kv-sim` subcommand replaying seeded synthetic workloads through the real directory, policy and planner against in-memory tiers, reporting hit rate per tier, recomputed tokens and transferred bytes per policy.
- [S-17] Correctness: KV produced via prefix reuse, L1 round trip or L2 round trip yields the same greedy tokens as a cold run of the same request, and block bytes are identical after each round trip (TS §21 rule 1; lossy KV formats are out of scope).

## Out of scope

- L3 cluster RAM/NVMe and L4 external/object tiers, the cluster-wide KV directory, KV transfer between nodes (phases 6–7).
- Lossy KV transforms (FP8 on CPU, compressed/quantized KV on slower tiers) — TS §8 requires quality validation first.
- Persisting the NVMe tier across restarts, and sharing it between processes.
- Multi-GPU KV placement (phase-5).
- Per-tenant cache quotas and accounting (TS §16 hooks only: the opt-in cache salt is the isolation hook).
- An L1 tier on unified-memory devices (GB10): it would copy between two budgets of the same physical memory; L1 is proven on the R9700.
- Hybrid (linear-attention) prefix snapshots: the Phase 1–2 models (Llama-3.2-3B, OLMoE-1B-7B) have per-token KV only; hybrid families arrive with the Phase 8 families track.
- Changing Phase 3 admission policy beyond supplying `cached_prefix_tokens` and planner costs.
- A kernel-level change to attention: prefix reuse uses the Phase 2 paged-attention block tables unchanged.

## Constraints

- Boundaries (TS §21 rule 6): `turbine-kv` modules `identity`, `directory`, `tier`, `policy`, `planner`, `transfer`, `session`, `metrics`, each with its own public API and unit tests; no module reaches into another's internals. Pinned-memory allocation and copy streams live in the kernel shims behind C ABI v3, wrapped in `turbine-kernels` (FFI, `// SAFETY:` comments stating who owns each host buffer and when a copy may be reused; no HIP/CUDA runtime binding outside the shim — CONFLICT C-6); `turbine-kv` stays `unsafe_code = "forbid"`.
- Everything except the GPU/pinned-memory/NVMe backends builds and tests on macOS arm64 with no GPU; the NVMe tier's file layer is tested on macOS against a temp directory without `O_DIRECT` (`F_NOCACHE` is not required).
- New dependencies with reasons: `blake3` (block hashing: fast, keyed, stable across platforms), `crc32c` (per-block checksum with hardware acceleration on arm64 and amd64). No other new runtime dependency.
- Bounded everything (TS §21 rule 8): directory entries bounded by the sum of tier capacities in blocks; session table by `kv.session.max_sessions`; transfer in-flight bytes; NVMe queue depth; prefetch queue (`kv.prefetch.max_queue`).
- The eviction policy and planner are pure functions of directory state, tier estimates and a clock, so they are deterministic under the simulator and benchmarkable (TS §8).
- No decision without a reason (TS §21 rule 7): every eviction, demotion, promotion, plan and prefetch carries a reason code and is counted.
- Memory: L0 capacity comes from the Phase 3 `kv` pool; L1 draws host memory that Phase 3's host signals watch and exists only on discrete-VRAM devices. On GB10 unified memory the Phase 3 budget (`MemAvailable` at startup − host reserve, capped by `reliability.memory.device_budget_bytes`) bounds L0 alone.
- Model: per-token KV only. GPU tests use `meta-llama/Llama-3.2-3B-Instruct` in BF16 (28 layers × 8 KV heads × 128 head dim, 112 KiB of BF16 KV per token, 1,835,008 bytes per 16-token block) from `TURBINE_TEST_MODEL_DIR` (`/home/piwi/turbine-models/<slug>`); tests never download weights. KV dtype is BF16 only (Phase 2 decision).
- Hardware order: GPU tests run first on novanas (one R9700 via a k3s Job requesting `amd.com/gpu: 1`, ROCm 7.14.1 at `/opt/rocm/rocm`), the only device where L1 is physically separate memory, then on the Sparks (GB10) with L1 disabled.
- Asking first: the implementer asks the user before any run that needs production workloads moved or memory freed on any host (a free R9700 on novanas; room beside production vLLM on a Spark). Lab runs never stop or starve production vLLM themselves; runs on a Spark stay inside containers with a hard `--memory` cap.
- Disk: the NVMe tier on the lab hosts lives at `/home/piwi/turbine-kv` on each host's root NVMe, capped at 64 GiB per host (the Sparks have about 133–153 GB free on 83–85 % used 916 GB disks; novanas has about 571 GB free on its ext4 root). Where the filesystem refuses `O_DIRECT` the tier falls back to buffered I/O with a WARN.

## Interfaces

### Configuration (new and changed keys)

| Key                                | Type     | Default               | Validation / meaning                                                                     |
| ---------------------------------- | -------- | --------------------- | ---------------------------------------------------------------------------------------- |
| `kv.gpu.enabled`                   | bool     | true                  | false is rejected in Phase 4 ("L0 is required")                                          |
| `kv.cpu.enabled`                   | bool     | true                  | ignored with a WARN on unified-memory devices (L1 disabled there)                        |
| `kv.cpu.max_bytes`                 | bytes    | 64GiB                 | > 0 when enabled; ≤ host MemTotal − `reliability.memory.host_reserve_bytes`, else exit 2 |
| `kv.nvme.enabled`                  | bool     | false                 | —                                                                                        |
| `kv.nvme.path`                     | string   | `/var/lib/turbine/kv` | absolute; created if absent; must be writable, else exit 1; lab: `/home/piwi/turbine-kv` |
| `kv.nvme.max_bytes`                | bytes    | 64GiB                 | > 0; ≤ free space at startup minus 10 %, else exit 1 naming the free space; lab: 64GiB   |
| `kv.nvme.slab_bytes`               | bytes    | 1GiB                  | multiple of block size rounded to 4 KiB                                                  |
| `kv.nvme.max_queue_depth`          | integer  | 64                    | 1..1024                                                                                  |
| `kv.nvme.io_threads`               | integer  | 4                     | 1..64                                                                                    |
| _kv.policy_                        | enum     | `cost_aware`          | `cost_aware` or `lru`                                                                    |
| _kv.demote_min_value_              | number   | 0.0                   | ≥ 0; blocks scoring below this are dropped instead of demoted                            |
| _kv.prefix_sharing_                | bool     | true                  | false: no lookup, no reuse (A/B switch)                                                  |
| `kv.transfer.max_inflight_bytes`   | bytes    | 1GiB                  | ≥ one block                                                                              |
| `kv.session.max_sessions`          | integer  | 10000                 | 1..1000000                                                                               |
| `kv.session.hot_ttl`               | duration | 60s                   | —                                                                                        |
| `kv.session.warm_ttl`              | duration | 10m                   | > `hot_ttl`                                                                              |
| `kv.session.max_idle`              | duration | 1h                    | session metadata dropped after this                                                      |
| `kv.prefetch.lead_time`            | duration | 2s                    | —                                                                                        |
| `kv.prefetch.max_queue`            | integer  | 256                   | —                                                                                        |
| `kv.policy_weights.session_active` | number   | 0.5                   | 0..1                                                                                     |
| `kv.policy_weights.hit_half_life`  | duration | 60s                   | —                                                                                        |

Durations and byte sizes use the Phase 0 / Phase 3 formats.

### Cost-aware value terms

For a block b in tier t at time now:

- reuse_probability = min(1, `session_active`·[b's session is hot] + h / (1 + h)), where h is b's hit count decayed with half-life `hit_half_life`, plus prefix popularity (child count / (1 + child count)) scaled by 0.25, capped at 1.
- recompute_cost = seconds to prefill b's tokens at its depth: block_tokens / prefill_tps × (1 + depth_tokens / 8192), from the Phase 3 prefill EWMA.
- retrieval_cost = seconds to bring b back to L0 from the tier it would be demoted to: latency + size / bandwidth of that path (recompute cost when it would be dropped); floor 1 µs.
- memory_cost = size_bytes / tier capacity × (1 + tier pressure level), pressure level 0 (GREEN) … 4 (SURVIVAL).
- priority = 1.0 normal, 2.0 high, 0.5 low (Phase 2 request priority of the most recent user, lower integer = higher priority: < 0 high, 0 normal, > 0 low — CONFLICT C-10).
- Tie-break: older last access first, then larger depth first (leaves).

### Plan decision

`KvPlan { reuse_l0: n, promote: [(tier, n)], recompute_tokens: n, reason }` with reasons `all_l0`, `retrieve_cheaper`, `recompute_cheaper`, `l0_pressure`, `tier_degraded`, `no_match`.

### Session hints

| Carrier                                   | Where                                                            | Meaning                                                                                             | Invalid → 400 `invalid_request_error` code |
| ----------------------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| `prompt_cache_key`                        | OpenAI body field (chat completions, completions)                | session id: 1–128 visible ASCII chars                                                               | `invalid_session_id`                       |
| `x-turbine-session-resume-within: <secs>` | request header                                                   | integer 1..86400; the session's blocks keep the `session_active` boost until now + secs             | `invalid_session_hint`                     |
| `x-turbine-session-end: true`             | request header                                                   | release the session boost and drop the session entry right after this response (`false` is a no-op) | `invalid_session_hint`                     |
| `x-turbine-cache-salt: <salt>`            | request header (also honoured by `POST /turbine/v1/kv/prefetch`) | 1–128 visible ASCII chars; enters the namespace key, so only same-salt requests share prefixes      | `invalid_cache_salt`                       |

Either session header without `prompt_cache_key` → 400 `invalid_session_hint`. A session remembers the salt of its first request; a later request with the same `prompt_cache_key` and a different salt starts a new session entry (no cross-salt sharing). Without any of these, requests still share prefixes globally; they simply have no session boost or predicted-resume prefetch.

### HTTP routes (changes from Phase 3)

| Route                          | Phase 4 behaviour                                                                                                                                                                                                                                                                                        |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET /turbine/v1/kv`           | `200` KV document (see Data)                                                                                                                                                                                                                                                                             |
| `POST /turbine/v1/kv/prefetch` | body `{"session_id":"…"}` (a `prompt_cache_key` value) or `{"prompt":"…"}` / `{"messages":[…]}` (tokenized like the OpenAI routes); `202` `{"blocks_queued":n,"blocks_resident":n}`; `404` code `session_not_found`; `429` code `prefetch_queue_full`; `409` code `pressure_too_high` at ORANGE or above |
| OpenAI inference routes        | responses add `usage.prompt_tokens_details.cached_tokens` (also in the final streamed usage chunk)                                                                                                                                                                                                       |

### Metrics (label values from closed sets)

- `turbine_kv_blocks{tier,state}` (the Phase 2 label set, `state` ∈ `used`, `free`; the per-tier total is the sum — CONFLICT C-4) and `turbine_kv_bytes{tier,kind}` (`kind` ∈ `capacity`, `used`) gauges; `tier` ∈ `l0`, `l1`, `l2` (a disabled tier reports 0; Phase 6 adds `l3`).
- `turbine_kv_lookups_total{result}` (`l0`, `l1`, `l2`, `miss`) counter per block; `turbine_kv_prefix_cached_tokens_total`, `turbine_kv_prompt_tokens_total` counters (hit rate = ratio).
- `turbine_kv_promotions_total{from,to}`, `turbine_kv_demotions_total{from,to}`, `turbine_kv_evictions_total{tier,reason}` (`capacity`, `pressure`, `session_expired`, `checksum`, `tier_degraded`), `turbine_kv_drops_total{reason}` counters.
- `turbine_kv_recompute_tokens_total{reason}` and `turbine_kv_plans_total{reason}` counters.
- `turbine_kv_transfer_seconds{path}` histogram and `turbine_kv_transfer_bytes_total{path}` counter, `path` ∈ `l0_to_l1`, `l1_to_l0`, `l1_to_l2`, `l2_to_l1`, `l0_to_l2`, `l2_to_l0` (the last two only on unified devices); `turbine_kv_transfer_bandwidth_bytes_per_second{path}` gauge (EWMA).
- `turbine_kv_prefetch_total{outcome}` (`used`, `wasted`, `cancelled`, `rejected`) counter; `turbine_kv_sessions` gauge.
- `turbine_kv_tier_degraded{tier}` gauge; `turbine_storage_queue_depth`, `turbine_storage_latency_seconds` (histogram) for L2.

### Structured log events

`kv_plan` (DEBUG, request id, matched blocks per tier, costs, chosen cutoff, reason), `kv_evict` / `kv_demote` / `kv_promote` (DEBUG, aggregated per iteration at INFO), `kv_prefetch` (INFO), `kv_tier_degraded` (WARN), `kv_checksum_mismatch` (WARN), `kv_calibration` (INFO at startup with measured bandwidths).

### `turbine-bench` additions

```
turbine-bench ... --profile multi-turn --sessions <n> --turns <n> --shared-prefix-words <n>
                  [--think-time <min>..<max>] [--session-hints]
turbine-bench kv-sim --workload <multi-turn|shared-system|mixed> --policy <cost_aware|lru>
                     --l0-blocks <n> [--l1-blocks <n>] [--l2-blocks <n>] [--seed <u64>] [--output text|json]
```

- Multi-turn: each session sends `--turns` requests, each turn's prompt = shared prefix + the session's full history + a new user message, sequential within a session, sessions concurrent up to `--concurrency`; `--session-hints` sends the session id as `prompt_cache_key` and `x-turbine-session-resume-within` set to the maximum think time; the last turn sends `x-turbine-session-end: true`. Report adds `cached_tokens_ratio` (from `usage.prompt_tokens_details.cached_tokens`) and TTFT split by turn index (first vs later).
- `kv-sim` runs the real `identity`, `directory`, `policy` and `planner` with in-memory tiers and a fake clock; JSON keys `hit_rate_by_tier`, `recompute_tokens`, `transfer_bytes`, `evictions`, `simulated_prefill_seconds`.

## Data

- Nothing persists across restarts. At startup L2 slab files under _kv.nvme.path_ matching `turbine-kv-*.slab` are deleted and recreated; other files in the directory are left alone.
- Block key: 16 bytes = first 128 bits of BLAKE3(namespace_key ‖ parent_key ‖ token ids as little-endian u32). namespace_key = BLAKE3 of a canonical JSON of `{model_config_hash, weights_index_hash, kv_format, block_tokens, cache_salt}` with `cache_salt` the empty string when no salt is sent. Namespace keys are computed per distinct salt and memoised. Displayed as 32 lowercase hex characters.
- L2 slab file layout: slab header (4 KiB: magic `TKVSLAB1`, format version, namespace key, slot size, slot count) followed by slots; each slot holds one block's bytes padded to 4 KiB. The in-memory index maps block key → (slab, slot, crc32c, length).
- KV document (`GET /turbine/v1/kv`):

```json
{
  "policy": "cost_aware",
  "prefix_sharing": true,
  "block_tokens": 16,
  "block_bytes": 1835008,
  "tiers": [
    {
      "tier": "l0",
      "enabled": true,
      "capacity_bytes": 21474836480,
      "used_bytes": 7516192768,
      "blocks": 4096,
      "referenced_blocks": 3100,
      "pressure": "YELLOW",
      "degraded": false,
      "est_latency_seconds": 0.0,
      "est_bandwidth_bytes_per_second": null
    },
    {
      "tier": "l1",
      "enabled": true,
      "capacity_bytes": 68719476736,
      "used_bytes": 2147483648,
      "blocks": 1170,
      "referenced_blocks": 0,
      "pressure": "GREEN",
      "degraded": false,
      "est_latency_seconds": 0.00002,
      "est_bandwidth_bytes_per_second": 50000000000
    }
  ],
  "unified_memory": false,
  "hit_rate": {
    "window_seconds": 300,
    "prompt_tokens": 1200000,
    "cached_tokens": 840000
  },
  "sessions": { "active": 12, "max": 10000 },
  "prefetch": { "queued": 0, "used": 118, "wasted": 9, "cancelled": 2 },
  "transfers": { "inflight_bytes": 0, "max_inflight_bytes": 1073741824 }
}
```

- The directory, tiers and session table are owned by the KV orchestrator task in `turbine-server`; the scheduler holds block references, never block metadata.

## Edge cases

- Prompt exactly a multiple of block size and fully cached (the last block is recomputed so at least one token is prefilled); prompt shorter than one block (no sharing).
- Two requests with the same new prefix arriving together: the second waits on the first's in-progress blocks (registered as pending in the directory) instead of computing them twice, bounded by a 2 s wait after which it computes its own.
- Hash collision or corrupted directory entry: token ids mismatch → treated as miss, counted, logged.
- Same prompt text under a different chat template, sampling seed or model revision: different tokens or different namespace → no sharing (sampling parameters do not affect KV and do not enter the key).
- Eviction candidate is the parent of a still-cached child (leaf-first rule); a block referenced by a running sequence under SURVIVAL (never evicted; Phase 3 preemption releases it first).
- Promotion needed while L0 is at RED or above (planner recomputes or waits per plan reason `l0_pressure`; running generations are never evicted to promote).
- Block demoted to L1 while a promotion of the same block is in flight; request cancelled while its promotion is in flight; session prefetch for a session that is then ended.
- L1 slab allocation fails (pinned memory limit, `ulimit -l`): L1 stays at its current size, logged, not fatal.
- NVMe full, read-only, slow (p99 latency > 10× calibration) or returning checksum mismatches: tier degraded, recompute path; path on a filesystem without `O_DIRECT` support (e.g. tmpfs, some ZFS versions) → buffered I/O with a WARN.
- Same prompt with and without `x-turbine-cache-salt`, or with two different salts: different namespace keys, no sharing; a prefetch by `session_id` uses the session's recorded salt.
- `x-turbine-session-end: true` on a request whose session has another request in flight: the entry is dropped after the last in-flight request finishes.
- Unified-memory device with `kv.nvme.enabled: false`: no tier below L0; demotion always drops, and the planner only chooses between `all_l0` and recompute.
- Session id reused concurrently by two in-flight requests (both proceed; the session tail becomes whichever finishes last).
- Many distinct sessions exceeding `max_sessions` (least recently active session metadata dropped; its blocks lose the boost only).

## Failure modes

- **Unified-memory device:** L1 disabled by design with one WARN (`kv_l1_disabled_unified`), not an error.
- **Pinned-memory API unavailable (no GPU library, macOS):** L1 disabled with a WARN, engine runs with L0 only; explicit `kv.cpu.enabled: true` on a GPU host where pinned allocation of the first slab fails → exit 1 naming the error.
- **Copy stream error during transfer:** the block copy is discarded, the tier is marked degraded if 3 errors occur within 60 s, the request falls back to recompute; a sticky device error follows Phase 3 (`device_fatal`).
- **NVMe I/O error or checksum mismatch:** block dropped, `evictions_total{reason="checksum"}` counted, request recomputes; repeated errors (3 in 60 s) mark L2 degraded for 5 minutes, after which one probe write/read decides.
- **NVMe slow:** storage latency signal escalates Phase 3 pressure; demotions to L2 pause at ORANGE storage pressure and blocks are dropped instead.
- **Directory inconsistency detected (location points at a freed slot):** the location is removed, the lookup is a miss, an error is logged with the block key; debug builds assert.
- **Prefetch overload:** prefetch queue full → new prefetches rejected (`prefetch_total{outcome="rejected"}`), never blocking requests.
- **Calibration copy fails at startup:** bandwidth estimates fall back to conservative constants (L0↔L1 8 GB/s, L1↔L2 and L0↔L2 1 GB/s) with a WARN.

## Acceptance criteria

- [ ] [S-1] `cargo test -p turbine-kv identity::tests::keys_are_stable_and_scoped` exits 0; it asserts committed golden keys for a fixed token sequence, that changing any token, the parent, the model config hash, block size or cache salt changes the key, that the empty salt (no header) yields one global namespace shared by all requests, and that a partial block has no key; fails if key derivation changes silently or ignores a scoping input.
- [ ] [S-1] [S-2] `cargo test -p turbine-kv directory::tests::collision_is_a_miss` exits 0; it forces two token sequences to the same key via a test hasher and asserts the second lookup is a miss counted in lookups `miss` with a logged mismatch; fails if a collision returns another sequence's KV.
- [ ] [S-2] `cargo test -p turbine-kv directory::tests::longest_prefix_across_tiers` exits 0; it places blocks 0–3 in L0, 4–5 in L1 and 6 in L2 and asserts the lookup returns 7 matched blocks with their tiers, and stops at the first missing block; fails if lookup skips a gap or ignores a tier.
- [ ] [S-3] `cargo test -p turbine-scheduler --test kv_sim prefix_reuse_refcounts` exits 0; it runs 50 simulated requests sharing a 64-block prefix and asserts the prefix is prefilled once, ref counts rise and fall to 0 after completion, shared blocks are never written, a fully cached prompt still prefills one token, and `cached_prefix_tokens` reaches admission; fails if a shared block is mutated or a ref leaks.
- [ ] [S-3] [S-17] Lab test `cargo test -p turbine-server --test kv_gpu prefix_reuse_matches_cold -- --ignored` exits 0 under `scripts/lab-test.sh novanas` and, once the NVIDIA path exists, under `scripts/lab-test.sh dgx-spark` (each run after the user confirms the host may be used); with Llama-3.2-3B-Instruct BF16 from `TURBINE_TEST_MODEL_DIR` it runs 5 prompts cold, then again with warm prefixes, greedy 64 tokens each, and asserts identical token ids and `cached_tokens` > 0 on the warm run; fails if reuse changes outputs or is not used.
- [ ] [S-1] [S-3] `cargo test -p turbine-api --test api cache_salt_isolates` exits 0; against the simulated KV stack it sends the same prompt with no salt twice (second has `cached_tokens` > 0), then with `x-turbine-cache-salt: a` (0 cached), again with `a` (> 0), and with `b` (0), and asserts a 129-char salt returns 400 `invalid_cache_salt`; fails if salted requests share with unsalted or other-salt requests.
- [ ] [S-4] `cargo test -p turbine-kv tier::tests::contract_suite` exits 0; one generic suite runs against the in-memory tier and the L2 file tier on a temp directory and asserts put/get/contains/evict semantics, capacity enforcement and byte-identical round trips; fails if a tier implementation diverges from the trait contract.
- [ ] [S-5] [S-6] [S-17] Lab test `cargo test -p turbine-kernels --test lab pinned_round_trip -- --ignored` (CONFLICT C-6) under `scripts/lab-test.sh novanas` (R9700, after the user confirms it is free) exits 0; it allocates two 1 GiB L1 slabs through the ABI v3 pinned allocation (`hipHostMalloc` in the HIP shim), copies 1,000 random GPU blocks to L1 and back through the copy stream, asserts byte equality, and logs measured bandwidth per direction; fails if any byte differs or a copy completes before its event signals.
- [ ] [S-5] `cargo test -p turbine-kv tier::tests::l1_grows_and_shrinks` exits 0; with a fake allocator it asserts slabs are allocated only on demand up to `kv.cpu.max_bytes`, a failed slab allocation leaves L1 usable at its current size, and empty slabs are released when host pressure reaches RED; with a unified-memory device description it asserts L1 reports `enabled: false`, one WARN is logged, and demotion targets L2 directly; fails if L1 preallocates, cannot shrink, or runs on unified memory.
- [ ] [S-6] `cargo test -p turbine-kv transfer::tests::inflight_bounded` exits 0; it queues 10 GiB of simulated copies with `max_inflight_bytes: 1GiB` and asserts in-flight bytes never exceed 1 GiB and all copies complete; fails if the bound is exceeded.
- [ ] [S-7] `cargo test -p turbine-kv policy::tests::cost_aware_ordering` exits 0; it asserts, for blocks differing in one term at a time, that higher reuse, higher recompute cost and higher priority raise value while larger size, higher tier pressure and cheaper retrieval lower it; that referenced blocks and parents of cached children are never candidates; fails if any term has the wrong direction or a protected block is evicted.
- [ ] [S-7] [S-16] `cargo run -p turbine-bench -- kv-sim --workload mixed --policy cost_aware --l0-blocks 2048 --l1-blocks 8192 --seed 1 --output json` and the same with `--policy lru` both exit 0, and `cargo test -p turbine-bench --test kv_sim cost_aware_beats_lru` exits 0 asserting `simulated_prefill_seconds` for `cost_aware` ≤ 0.9 × `lru` on `multi-turn` and `mixed` and ≤ 1.0 × on `shared-system`; fails if the default policy is not at least as good as LRU.
- [ ] [S-8] `cargo test -p turbine-scheduler --test kv_sim demotion_under_pressure` exits 0; it drives Phase 3 pressure to ORANGE with a full L0 and asserts unreferenced blocks move to L1 (copy then free), blocks below `demote_min_value` are dropped, L1 victims go to L2 when enabled, and a later request promotes them before its prefill chunk; the same scenario with a unified device asserts L0 blocks go straight to L2 (`demotions_total{from="l0",to="l2"}`); fails if demotion frees before the copy completes, promotion is skipped, or a unified device uses L1.
- [ ] [S-9] `cargo test -p turbine-kv planner::tests::cutoff_minimises_cost` exits 0; it asserts for fixed bandwidths and prefill rate that a 1,000-block L1 prefix is retrieved, a 4-block L2 prefix behind a slow disk is recomputed, the chosen cutoff equals a brute-force minimum over all k for 200 random cases, and L0 at RED yields reason `l0_pressure`; fails if the planner ever picks a costlier plan than brute force.
- [ ] [S-10] `cargo test -p turbine-kv session::tests::lifecycle_and_prefetch` exits 0; with a fake clock it asserts a session's blocks demote L0→L1 after `hot_ttl` and L1→L2 after `warm_ttl`, predicted-resume prefetch starts at last activity + gap EWMA − `lead_time` only at GREEN/YELLOW, `x-turbine-session-end` removes the boost and the entry, `x-turbine-session-resume-within: 600` keeps the boost for 600 s past `hot_ttl`, the table evicts beyond `max_sessions`, and unused prefetched blocks count as `wasted`; fails if session state is unbounded or prefetch ignores pressure.
- [ ] [S-10] [S-13] `cargo test -p turbine-api --test api kv_routes` exits 0; it asserts `GET /turbine/v1/kv` returns every key of the Data example, `POST /turbine/v1/kv/prefetch` returns 202/404/429/409 per the route table, that a request with `prompt_cache_key` creates a session visible in `sessions.active`, and that a 129-char `prompt_cache_key` returns 400 `invalid_session_id`, `x-turbine-session-resume-within: 0` returns 400 `invalid_session_hint`, and `x-turbine-session-end: true` without `prompt_cache_key` returns 400 `invalid_session_hint`; fails if the route still returns 501, a status differs, or `prompt_cache_key` is rejected by the OpenAI schema.
- [ ] [S-11] `cargo test -p turbine-kv tier::tests::nvme_checksum_and_restart` exits 0; on a temp directory it writes blocks, corrupts one byte of one slot, and asserts that read returns a checksum miss counted under `evictions_total{reason="checksum"}`; restarting the tier deletes only `turbine-kv-*.slab` files; fails if a corrupted block is returned or unrelated files are deleted.
- [ ] [S-11] [S-17] Lab test `cargo test -p turbine-server --test kv_gpu nvme_round_trip_matches_cold -- --ignored` with `kv.nvme.path: /home/piwi/turbine-kv` and `kv.nvme.max_bytes: 64GiB` exits 0 under `scripts/lab-test.sh novanas` (forcing a prefix L0→L1→L2 and back) and, once the NVIDIA path exists, under `scripts/lab-test.sh dgx-spark` (forcing L0→L2 and back, L1 disabled); each asserts identical greedy tokens to the cold run, byte-identical blocks, and that no more than 64 GiB of slab files exist under the path; fails if the NVMe path corrupts KV or exceeds its cap.
- [ ] [S-12] `cargo test -p turbine-scheduler --test kv_sim cancellation_releases_kv` exits 0; it cancels 100 requests during in-flight promotions and prefetches and asserts all their references drop within one iteration, no completed transfer is written into a cancelled sequence, and L0 referenced blocks return to baseline; fails if cancellation leaks references or transfers.
- [ ] [S-12] `cargo test -p turbine-kv tier::tests::faulty_tier_degrades_to_recompute` exits 0; it injects 3 read errors in 60 s into the in-memory tier and asserts the tier is marked degraded, plans switch to reason `tier_degraded`, and no request fails; fails if a tier error surfaces to the client.
- [ ] [S-14] `cargo test -p turbine-api --test api kv_metrics_bounded` exits 0; it exercises hits in every tier, demotions, evictions, recomputes and prefetch outcomes against the simulated KV stack and asserts every metric family under Interfaces appears with only documented label values; fails if a family is missing or a label is unbounded.
- [ ] [S-15] `cargo test -p turbine-core config::tests::kv_config_validation` exits 0; it asserts the defaults in the configuration table and rejects, naming the key: `kv.gpu.enabled: false`, `kv.policy: lfu`, `kv.nvme.max_bytes: 0`, `kv.session.warm_ttl: 30s` with `hot_ttl: 60s`, `kv.nvme.path: rel/kv` with NVMe enabled, `kv.transfer.max_inflight_bytes: 1`; fails if an impossible KV config is accepted.
- [ ] [S-16] Manual lab run (after the user confirms an R9700 is free): `turbine-bench --url http://<turbine on novanas> --profile multi-turn --sessions 16 --turns 8 --shared-prefix-words 2000 --concurrency 8 --session-hints --output json` exits 0 with _kv.prefix_sharing_ on and off; the pasted evidence shows `cached_tokens_ratio` ≥ 0.6 with sharing on and TTFT p50 for turns ≥ 2 at most 0.5× the sharing-off value; the same pair is then recorded on dgx-spark once the NVIDIA path exists; fails if prefix sharing does not cut later-turn TTFT.

## Open questions

<!-- None: decisions recorded in .procoder/ask/decisions.md -->
