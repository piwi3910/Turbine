# phase-3-reliability

Status: complete

Source: `turbine-spec.md` §19 Phase 3 (reliability), with §9 (reliability and pressure controller), §14 (observability), §17 (testing), §18 (benchmarking) and §21 (engineering rules). Sections of that document are cited as "TS §N". Decisions and their options are recorded in `.procoder/ask/decisions.md`. Builds on `phase-0-skeleton`, `phase-1-single-request` and `phase-2-serving-runtime`; where this spec depends on a question those specs leave open, it says so instead of re-asking.

## Problem

After Phase 2 Turbine serves concurrent streaming requests with continuous batching, chunked prefill and a paged GPU KV pool, but nothing stops it from accepting more work than the device can hold. A burst of long prompts exhausts KV blocks mid-generation, a workspace allocation fails and the iteration aborts, host memory on a unified-memory GB10 is squeezed by co-tenant processes (both DGX Sparks run production vLLM that holds 80–90 GB of the ~121 GB pool), and thermal throttling silently doubles inter-token latency. vLLM and SGLang answer overload by preempting, OOMing or timing out; Turbine's first differentiator (TS §1 pillar 2) is to predict pressure and degrade gracefully instead. Phase 3 is the phase that makes the TS §20 promise "survive deliberate overload by throttling/queuing instead of OOMing" true and provable: explicit memory budgets and reservations, live pressure telemetry feeding a hysteretic state machine, predictive admission, a throttling ladder the scheduler obeys, bounded recovery from allocation failure, a circuit breaker, and an overload soak benchmark whose pass criterion is "no worker death, no OOM, recovery to GREEN". The primary hardware is one discrete-VRAM R9700 (gfx1201, 32 GB) on novanas, where Phases 1–2 were built; the same controller then runs on the unified-memory GB10 DGX Sparks, whose NVIDIA path arrives right after Phase 2 behind the same vendor-neutral device and kernel traits.

## Users

- **Operators:** need `/turbine/v1/pressure` and Prometheus metrics that say which state the engine is in, why (reason codes, dominant signal, predicted exhaustion horizon), what it is doing about it (throttle plan), and whether the circuit is open; need `/ready` to go false while the circuit is open so a load balancer stops routing; need a small set of overridable knobs with safe defaults (TS §9 self-tuning).
- **API clients:** need overload to surface as well-formed, retryable responses (`429`/`503` with `Retry-After` and a stable error `code`) rather than dropped connections, stalls or a dead server; need already-streaming generations to keep streaming while new work is queued.
- **Turbine developers (humans and AI agents):** need the controller, admission and circuit logic as pure, clock-injected code testable on macOS without a GPU (TS §17 items 1, 2, 7), a fault-injection build to force allocation failures and device errors, and a soak script that runs the overload benchmark on the lab hosts.
- **Benchmark runners:** need `turbine-bench` to generate open-loop overload (arrival rate above service rate) for a fixed duration and to record the pressure timeline alongside latency (TS §18 "overload stability, recovery from pressure").

## In scope

- [S-1] New crate `crates/turbine-reliability` (`unsafe_code = "forbid"`, no GPU dependency) holding the memory budget and reservation ledger, pressure signals, the pressure state machine, the admission predictor, the throttle planner, the recovery controller and the circuit breaker. Every time-dependent component takes a `Clock` trait object so tests drive time deterministically.
- [S-2] Memory budget and pools: at startup, after weights load, compute a per-device budget split into the pools `weights`, `kv`, `workspace`, `runtime` and `reserve` (TS §9 "resource reservations"). On dedicated-memory devices (R9700) the budget is the device memory measured free at startup (plus what Turbine itself already holds for weights); on unified-memory devices (GB10) it is `MemAvailable` at startup minus `reliability.memory.host_reserve_bytes`. In both cases `reliability.memory.device_budget_bytes`, when set, caps it. Refuse to start when the budget cannot hold weights plus workspace plus runtime overhead plus the emergency reserve plus the minimum KV (one full-context sequence). The KV pool size that Phase 2 derived becomes the `kv` pool capacity from this budget: `kv.gpu.max_bytes` now defaults to null, meaning the `kv` pool is the remainder of the budget, and an explicit value caps the `kv` pool (CONFLICT C-8).
- [S-3] Reservation ledger: every KV block, workspace buffer and optional buffer is obtained through a pool reservation (`reserve → commit → release`) represented by an RAII guard, so cancellation, request failure and preemption release reservations on drop (TS §21 rule 9). The pools never hand out more than their capacity.
- [S-4] Emergency reserve: _reliability.emergency_vram_reserve_ bytes are allocated on the device at startup and held outside normal scheduling; only the recovery controller may release them (entering SURVIVAL), and they are re-acquired before the state may drop below RED. Reserve use is counted and exposed.
- [S-5] Live telemetry in `turbine-device`: a sampler with two cadences — a fast tick every `reliability.telemetry.interval` (default 100 ms) reading the host `/proc` files below and the allocator/pool ledger, and a vendor tick every `reliability.telemetry.vendor_interval` (default 1 s) reading, per device, used/free memory (dedicated devices), temperature, SM/GFX clock, throttle/clock-event reasons, power and utilisation from amd-smi (AMD, the amd-smi shared library from the ROCm install at /opt/rocm/rocm, runtime-loaded) and NVML (NVIDIA, runtime-loaded). The fast tick reads the host `/proc/meminfo` (MemAvailable, SwapTotal/SwapFree), `/proc/vmstat` (pswpin/pswpout) and `/proc/pressure/memory` (PSI `some avg10`). Every vendor call has a deadline; a missed deadline marks that source stale. Samples are published through a lock-free latest-value cell.
- [S-6] Pressure signals: each sample is converted into named signals (table under Interfaces) — KV pool utilisation including reservations, device memory utilisation, host available memory, PSI memory, swap-in rate, predicted KV exhaustion horizon, admission queue fill, ITL/step-time drift against a learned baseline, thermal state, telemetry staleness and recent allocation failures — each mapped to a level GREEN…SURVIVAL by fixed, documented default thresholds (the signal table) that the operator may override per signal; deriving thresholds per device from calibration is deferred to TS §9 self-tuning.
- [S-7] Pressure state machine `GREEN → YELLOW → ORANGE → RED → SURVIVAL` with hysteresis (TS §9): escalation after _reliability.pressure.escalate_samples_ consecutive samples at or above a level (immediately for allocation failure), de-escalation one level at a time only after every signal has stayed below that level's threshold minus the exit margin for the dwell time. Every transition records `from`, `to`, the dominant signal, its value and the threshold crossed, as a structured log event, a metric and an entry in a bounded transition history.
- [S-8] Exhaustion horizon predictor: estimates seconds until the KV pool is exhausted from the committed growth of admitted sequences (remaining `max_tokens` of each running sequence) and the observed decode rate; exposed as a signal and a metric.
- [S-9] Predictive admission (TS §9): before a request enters the scheduler, compute a `ResourceEstimate` (prompt tokens, cached prefix tokens — always 0 until phase-4 fills it — new prefill tokens, max output tokens, projected KV blocks, workspace bytes, estimated prefill and decode seconds from observed-throughput EWMAs) and return `Admit`, `Queue { reason }` or `Reject { reason }`. Admitted requests reserve their worst-case KV: blocks for prompt + `max_tokens`, or prompt + the remaining _model.max_seq_len_ context when `max_tokens` is absent, so a running sequence can never run out of KV below SURVIVAL. Overcommit is not offered in Phase 3. Requests that can never fit are rejected immediately; the admission queue is bounded by _reliability.admission.max_queue_ and each queued request by _reliability.admission.queue_timeout_. With adaptive admission, in GREEN, YELLOW, ORANGE and RED an admission (or a refill from the queue) also waits (`kv_reservation`) when its worst-case reservation would lift `kv_utilization` past the `kv_utilization` threshold of the next state up (at GREEN: RED's threshold, 0.90, so a burst of admissions inside one controller sample cannot reach RED or SURVIVAL; amendment 2026-10-02, 6b decision "which cap the GREEN admission headroom uses", A; lab: 14 admissions reserved 0.9946 and jumped GREEN to SURVIVAL), so admissions alone never escalate the state past that (KV headroom, amendment 2026-09-27: without it the backlog queued at the end of an overload re-escalated a de-escalating engine to RED and missed the 60 s recovery criterion in the simulation, seed 1: 64 s). The queue is ordered by the scheduling policy's admission key (Phase 2m `scheduling_policy`; the `default` policy is priority, then arrival). This admission queue is the one waiting queue from Phase 3 on (CONFLICT C-1): it replaces the Phase 2 waiting-queue bound, _scheduler.max_queued_requests_ remains only the HTTP→engine submission-channel capacity, and `scheduler.queue_timeout` is removed and rejected as a removed key.
- [S-10] Throttle planner: each scheduler iteration reads a `ThrottlePlan` derived from the current pressure state (table under Interfaces) — batch growth limit, prefill token budget fraction, prefill chunk size, admission mode and reclaim actions — implementing the TS §9 degradation ladder "reduce batch growth → throttle prefill → demote KV → shrink chunks → queue requests → stop admission → drain/recover". Phase 3 reclaims by freeing unreferenced cached KV blocks and optional buffers; demotion to other tiers is a `KvReclaimer` trait that phase-4 implements. Batch growth limits the admitted count (running requests plus those holding a KV reservation that wait for their first prefill chunk); in RED queued requests refill the slots of finished requests, so sustained overload keeps serving at a fixed batch instead of draining the queue only through timeouts (user decision 2026-09-26).
- [S-11] Recovery from allocation failure (TS §9): SURVIVAL keeps the engine live (decision "Phase 3: SURVIVAL liveness fix", 2026-09-27, provisional A pending user review, switchable by _reliability.recovery.survival_liveness_): under `requeue_unstarted` (A, default) admitted requests that have not started (no KV written) go back to the admission queue on entering SURVIVAL and drop their KV reservations (at their original turn and queue-wait start; answered `503 overloaded` when the queue is full), so running requests drain the pool; under `continue_prefills` (B) prefills already in progress continue in SURVIVAL at RED's budget and chunk floor. A device out-of-memory error from the executor or an allocator enters SURVIVAL, reclaims (optional buffers, cached blocks, emergency reserve), shrinks the failed iteration's batch and retries it up to _reliability.recovery.max_retries_ times with backoff; when retries are exhausted the requests in that batch fail with a `resource_exhausted` error and the worker keeps running. Never retries indefinitely.
- [S-12] Circuit breaker `HEALTHY → DEGRADED → CIRCUIT_OPEN → DRAINING → PROBING → HEALTHY` (TS §9) triggered by repeated OOM recoveries, non-OOM kernel/device errors, severe latency drift and thermal degradation, with the transition table under Interfaces. `GET /ready` returns 503 while the circuit is `CIRCUIT_OPEN`, `DRAINING` or `PROBING`. Sticky (context-corrupting) device errors open the circuit, drain, and exit the process with code 3; an external supervisor (k3s Job/Deployment restart policy, Docker restart policy or systemd) restarts it. These circuit rules replace the Phase 1 (3 consecutive failed requests) and Phase 2 (3 consecutive failed iterations) exit rules, and the `device_error` readiness reason is retired (CONFLICT C-25).
- [S-13] `GET /turbine/v1/pressure` returns the pressure document (see Data) instead of 501; `GET /turbine/v1/status` gains `pressure_state` and `circuit_state`; OpenAI routes return the admission error responses under Interfaces.
- [S-14] Reliability metrics and structured logs per TS §14: pressure state and transitions, signals, exhaustion horizon, admission outcomes and queue time, throttle actions, pool bytes, allocation failures and recoveries, emergency-reserve use, circuit state, and GPU temperature/clocks/throttle reasons and host memory.
- [S-15] Configuration: the `reliability` section extended with the keys under Interfaces, validated at startup (TS §15 "fail early on impossible configurations"); _reliability.enabled_ and _reliability.adaptive_admission_ gain their real meaning.
- [S-16] Fault injection: a Cargo feature `fault-injection` on `turbine-server` enabling a _reliability.fault_injection_ config section that fails every Nth pool allocation, raises a device OOM at a given iteration, raises a non-OOM kernel error, overrides telemetry temperature/throttle reasons and delays telemetry calls. Without the feature, the section is an unknown-key error.
- [S-17] Deterministic overload simulation without GPUs (TS §17 item 2): a test harness driving the real scheduler, admission and pressure controller against a simulated executor and simulated KV pool with a fake clock, at arrival rates up to 10× service capacity.
- [S-18] `turbine-bench` overload mode: `--duration`, open-loop seeded Poisson `--rate`, per-request ranges for prompt words and max tokens, a `--pressure-timeline` file sampling `/turbine/v1/pressure` once per second, and a report broken down by HTTP status and error `code`.
- [S-19] Overload soak script `scripts/overload-soak.sh <host>` (novanas first, then the Sparks) running calibrate → overload → cool-down phases against a freshly started `turbine-server` on a lab host and evaluating the pass criteria under Acceptance criteria (TS §18 "success is graceful queuing/throttling and recovery without worker death/OOM").

## Out of scope

- Pinned CPU, NVMe and remote KV tiers, prefix sharing, cost-aware eviction, recompute-vs-retrieve and session prefetch (phase-4). Phase 3 defines the `KvReclaimer` hook and reclaims only by freeing unreferenced GPU blocks.
- Multi-GPU pressure accounting, per-device admission across replicas, rebalancing between devices (phase-5), node failure handling (phase-6).
- Self-tuning against objectives such as a p99 TTFT target (TS §9 "long term"), including deriving pressure thresholds per device from calibration. Phase 3 thresholds are fixed configuration with documented defaults.
- KV overcommit at admission (reserving less than the worst case); it can be revisited once phase-4 demotion and recompute exist.
- Automatic in-process restart of a device context after a sticky device error; the process exits and an external supervisor restarts it.
- Per-tenant quotas and accounting (TS §16 hooks), request priorities beyond the priority field Phase 2 defines.
- Storage-tier signals (NVMe queue depth/latency/bandwidth); they arrive with the NVMe tier in phase-4.
- Overload comparison runs against vLLM/SGLang: the only vLLM instances in the lab serve production traffic and must not be overloaded.
- A supervisor, systemd unit, container image or deployment manifest for `turbine-server`.

## Constraints

- Rust only (TS §21 rule 4). `turbine-reliability` has no GPU, FFI or `unsafe` code; telemetry FFI stays in `turbine-device` with `// SAFETY:` comments (TS §21 rule 10). New runtime dependencies: none beyond the Phase 0–2 set; the Poisson arrival sampler in `turbine-bench` uses its existing seeded PRNG.
- Everything in `turbine-reliability`, the telemetry parsers for `/proc` files, the admission and circuit logic and the overload simulation build and pass on macOS arm64 with no GPU; `/proc` readers are given file contents by tests and report the host source as `unavailable` on macOS.
- No magic pressure behaviour (TS §21 rule 7): every state transition, admission decision other than `Admit`, throttle change, recovery attempt and circuit transition carries a reason code from a closed enum, a metric and a structured log event with the request id where one exists.
- Bounded everything (TS §21 rule 8): admission queue length, queue wait, transition history (32 entries), recovery retries, drain time and telemetry call time are all bounded by configuration with defaults.
- Protect running generations (TS §9): no pressure state below SURVIVAL may preempt or fail a running sequence to make room for a new request; below SURVIVAL, new work waits.
- The controller is not on the per-token hot path: the scheduler reads the current `ThrottlePlan` from a single atomic snapshot per iteration; the state machine evaluates on the telemetry tick, not per token.
- Memory budgeting uses memory measured free (dedicated) or `MemAvailable` (unified) at startup, never device total, because the R9700s may hold other workloads and the Sparks share their unified pool with production vLLM. On unified devices `device_memory` is not computed and the host signals (`host_available`, PSI, swap) stand in for device memory.
- Asking first: the implementer asks the user before any run that needs production workloads moved or memory freed on any host (the soak and fault-injection runs on novanas need a free R9700; any GPU run on a Spark needs room beside production vLLM). Lab runs never stop, restart or starve the production vLLM containers themselves; any memory-pressure run on a Spark runs inside a container with a hard memory limit (`docker run --memory`).
- Hardware order: GPU-backed tests run first on novanas (one R9700 via a k3s Job requesting `amd.com/gpu: 1`, ROCm 7.14.1 at `/opt/rocm/rocm`) and then on the Sparks (GB10) once the NVIDIA path from right after Phase 2 exists; telemetry and budget tests run on all three hosts.
- Model: all GPU-backed tests use the Phase 1 target, `meta-llama/Llama-3.2-3B-Instruct` in BF16 (dense, GQA, 28 layers, 8 KV heads × 128 head dim, so 112 KiB of BF16 KV per token), read from `TURBINE_TEST_MODEL_DIR` (`/home/piwi/turbine-models/<slug>` on each host); tests never download weights. OLMoE-1B-7B (BF16 MoE, Phase 2) must also budget correctly; its KV layout is taken from its config like any other model.
- Durations in configuration are strings `<integer><unit>` with unit `ms`, `s`, `m` or `h`, no space, case-sensitive; fractions are decimal numbers in the open or closed interval stated per key.

## Interfaces

### Configuration (new and changed keys)

All keys are optional; unknown keys remain errors. Two-part keys are in italics.

| Key                                          | Type     | Default           | Validation / meaning                                                                                                                                         |
| -------------------------------------------- | -------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| _reliability.enabled_                        | bool     | true              | false: state fixed at GREEN, `"enabled": false` in the pressure document; recovery and queue bounds remain                                                   |
| _reliability.emergency_vram_reserve_         | bytes    | 2GiB              | < budgeted device memory; 0 allowed (reserve disabled, logged at WARN)                                                                                       |
| _reliability.adaptive_admission_             | bool     | true              | false: admission checks only hard capacity and queue bounds                                                                                                  |
| `reliability.memory.workspace_bytes`         | bytes    | 1GiB              | > 0; execution workspace pool                                                                                                                                |
| `reliability.memory.runtime_overhead_bytes`  | bytes    | 1GiB              | ≥ 0; context, streams, graphs, allocator slack                                                                                                               |
| `reliability.memory.device_budget_bytes`     | bytes?   | null              | optional hard cap on everything Turbine claims on a device (co-tenancy); applies to dedicated and unified devices                                            |
| `reliability.memory.host_reserve_bytes`      | bytes    | 8GiB              | host memory Turbine never plans to use; drives the host signal; on unified devices the budget is `MemAvailable` at startup minus this                        |
| `reliability.telemetry.interval`             | duration | 100ms             | 50ms ≤ value ≤ 10s; fast tick: `/proc` files and allocator/pool ledger                                                                                       |
| `reliability.telemetry.vendor_interval`      | duration | 1s                | ≥ `reliability.telemetry.interval`, ≤ 60s; vendor tick: amd-smi / NVML temperature, clocks, throttle reasons, power, utilisation, device memory              |
| `reliability.telemetry.call_timeout`         | duration | 500ms             | < 10s; per vendor call                                                                                                                                       |
| `reliability.telemetry.stale_after`          | duration | 5s                | > `reliability.telemetry.vendor_interval`                                                                                                                    |
| `reliability.pressure.escalate_samples`      | integer  | 2                 | 1 ≤ value ≤ 20                                                                                                                                               |
| `reliability.pressure.deescalate_dwell`      | duration | 10s               | ≥ interval                                                                                                                                                   |
| `reliability.pressure.exit_margin`           | fraction | 0.05              | 0 ≤ value < 0.5, relative to the threshold                                                                                                                   |
| `reliability.pressure.thresholds.<signal>`   | list     | table below       | four ascending (or descending, for "lower is worse" signals) values for YELLOW, ORANGE, RED, SURVIVAL                                                        |
| `reliability.admission.max_queue`            | integer  | 256               | 1 ≤ value ≤ 65536                                                                                                                                            |
| `reliability.admission.queue_timeout`        | duration | 30s               | 1s ≤ value ≤ 1h                                                                                                                                              |
| `reliability.admission.large_prefill_tokens` | integer  | 2048              | ≥ block size; prefills above this are "expensive" (TS §9 ORANGE)                                                                                             |
| `reliability.admission.max_bypass`           | integer  | 8                 | smaller requests that may overtake a waiting queue head before it blocks the queue                                                                           |
| `reliability.recovery.max_retries`           | integer  | 3                 | 0 ≤ value ≤ 10                                                                                                                                               |
| `reliability.recovery.backoff`               | duration | 50ms              | doubled per retry                                                                                                                                            |
| `reliability.recovery.survival_liveness`     | enum     | requeue_unstarted | `requeue_unstarted` (A) or `continue_prefills` (B): how SURVIVAL keeps KV draining (S-11)                                                                    |
| `reliability.circuit.oom_recoveries_to_open` | integer  | 3                 | within `reliability.circuit.window`                                                                                                                          |
| `reliability.circuit.window`                 | duration | 60s               | —                                                                                                                                                            |
| `reliability.circuit.latency_drift_degraded` | fraction | 2.0               | ratio of windowed p95 step time to baseline; > 1                                                                                                             |
| `reliability.circuit.latency_drift_open`     | fraction | 4.0               | > `latency_drift_degraded`                                                                                                                                   |
| `reliability.circuit.cooldown`               | duration | 30s               | time in CIRCUIT_OPEN before DRAINING completes into PROBING                                                                                                  |
| `reliability.circuit.drain_timeout`          | duration | 120s              | running sequences still active after this fail with `circuit_open`                                                                                           |
| `reliability.circuit.probe_successes`        | integer  | 3                 | consecutive successful probes to return to HEALTHY                                                                                                           |
| `reliability.fault_injection.*`              | section  | absent            | only with Cargo feature `fault-injection`: `alloc_fail_every`, `oom_at_iteration`, `kernel_error_at_iteration`, `telemetry_temperature_c`, `telemetry_delay` |

Startup validation (exit 2, message naming the key): thresholds not monotonic; `latency_drift_open` ≤ `latency_drift_degraded`; `deescalate_dwell` shorter than `interval`; `vendor_interval` shorter than `interval`; `stale_after` not longer than `vendor_interval`; `fault_injection` present without the feature. After discovery and weight load (exit 1, message naming every pool and its bytes): budget cannot hold weights + workspace + runtime overhead + emergency reserve + one full-context sequence of KV.

### Pressure signals and default thresholds

Levels are YELLOW / ORANGE / RED / SURVIVAL; a signal below the YELLOW threshold is GREEN. The overall state is the maximum level over non-stale signals. These defaults are fixed and documented; each may be overridden per signal with `reliability.pressure.thresholds.<signal>`. Signals derived from `/proc` and the ledger update on the fast tick; `device_memory`, `thermal` and vendor staleness update on the vendor tick and hold their last level in between.

| Signal                  | Definition                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Y     | O    | R    | S     |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- | ---- | ---- | ----- |
| `kv_utilization`        | (used + reserved KV blocks) / KV pool capacity; used counts every block the pool holds, attached cached prefixes included (amendment 2026-10-01, P6b)                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | 0.70  | 0.82 | 0.90 | 0.97  |
| `device_memory`         | (device used − idle pre-allocated bytes) / budgeted device bytes (dedicated devices only; not computed on unified); idle pre-allocated = free `kv` pool bytes + the emergency reserve while held                                                                                                                                                                                                                                                                                                                                                                                                                        | 0.85  | 0.90 | 0.95 | 0.98  |
| `host_available`        | MemAvailable as a multiple of `host_reserve_bytes` (lower is worse)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | 4.0   | 2.0  | 1.0  | 0.5   |
| `psi_memory_some_avg10` | `/proc/pressure/memory` `some avg10`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 5     | 10   | 25   | 50    |
| `swap_in_rate`          | pages/s swapped in, from `/proc/vmstat` deltas                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 1     | 100  | 1000 | 10000 |
| `exhaustion_horizon`    | predicted seconds to KV exhaustion (lower is worse)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | 60    | 20   | 5    | 1     |
| `queue_fill`            | admission queue length / `max_queue`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 0.50  | 0.80 | 0.95 | —     |
| `step_time_drift`       | windowed p95 of each pure decode iteration's time over the calm baseline of its shape bucket (exact decoding rows × total context tokens in half-powers of two; learned in GREEN + HEALTHY, judged once the bucket has 8 calm steps, unseen shapes not judged; a decode step that overlapped an in-flight KV tier copy is neither judged nor learned and is counted in `turbine_decode_steps_unjudged_total{reason="kv_copy"}`, decision "6b: step-time drift during KV promotions" C; a judged step leaves the window after 10 s); not computed while nothing runs or fewer than 20 steps were judged in the last 10 s | 1.5   | 2.0  | 3.0  | —     |
| `thermal`               | 1 = within 5 °C of the vendor slowdown temperature, 2 = thermal throttle reason active                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 1     | 2    | —    | —     |
| `telemetry_stale`       | any source stale for longer than `stale_after`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | stale | —    | —    | —     |
| `allocation_failure`    | a device OOM in the last `deescalate_dwell`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | —     | —    | —    | any   |

`device_memory` counts pre-allocated bytes that hold no data as free (decision "Phase 3: device_memory counts idle pre-allocated pools as free", 2026-09-27, provisional pending user review): the `kv` pool and the emergency reserve are allocated up front, so the device reports them as used from startup. With `kv.gpu.max_bytes` at its default (null) the `kv` pool is the rest of the budget and an idle R9700 measured 0.972 of its budget used, RED before the first request; admission then queued every request until `queue_timeout` (the first 10-minute soak on novanas failed its calibration this way). Occupied KV blocks (committed or reserved) still count as used, so a full pool reads as before; releasing or re-acquiring the reserve leaves the value unchanged.

### Throttle plan per state

| State    | Batch growth                                                                   | Prefill budget                                                                 | Prefill chunk                  | Admission                                                                                                                                            | Reclaim                                                                                                                                          |
| -------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| GREEN    | unlimited                                                                      | 100 %                                                                          | configured                     | open                                                                                                                                                 | none                                                                                                                                             |
| YELLOW   | at most +1 admitted request/iteration                                          | 100 %                                                                          | configured                     | open                                                                                                                                                 | `KvReclaimer::demote` of idle blocks to a target of the YELLOW threshold (no-op in Phase 3)                                                      |
| ORANGE   | frozen at the current admitted count; finished requests may be replaced        | 50 %                                                                           | halved, floor 4 × block tokens | requests with new prefill > `large_prefill_tokens` queue (`pressure_orange`)                                                                         | free unreferenced cached blocks down to the ORANGE threshold; demote aggressively (phase-4)                                                      |
| RED      | frozen: queued requests only refill slots of finished requests (no net growth) | 50 %; new prefills start only in refilled slots; in-progress prefills continue | floor                          | every new request queues (`pressure_red`); the queue refills finished slots under ORANGE's rule (new prefill > `large_prefill_tokens` keeps waiting) | free all unreferenced cached blocks and optional buffers                                                                                         |
| SURVIVAL | shrink only; admitted requests that have not started return to the queue (A)   | 0 (A); in-progress prefills continue at 50 % (B)                               | — (A); floor (B)               | stopped: new requests rejected `503 overloaded`; the queue is kept but not drained                                                                   | release emergency reserve to the allocator; preempt (recompute) the most recently admitted sequence only if the next decode step cannot allocate |

Batch growth counts **admitted** requests: running ones plus those the admission gate admitted with their worst-case KV reservation that wait for their first prefill chunk. The gate moves queued requests into admitted slots only within the growth limit, measured against the admitted count after the previous iteration; admitted requests start as the state's prefill budget allows. Work-conserving floor (decision "Phase 3: soak stall — drift per iteration, idle floor", 2026-09-27, provisional): while nothing is admitted and the state is below SURVIVAL, the gate admits the first queued request that passes the KV checks (capacity, reservation, headroom) whatever its pressure reason (`pressure_orange`, `pressure_red`, `circuit_degraded`, `prefill_budget`), logged as `admission_decision` with reason `idle_floor`; those rules protect running work, and an idle device relieves no pressure. Without it a slot that a blocked queue head could not refill is lost (the next limit is measured from the lower count), the admitted count drains to 0, and ORANGE/RED then freeze an idle engine behind a full queue whose `queue_fill` holds the state. Counting reservations, not only running sequences, keeps a state change from stranding requests that already hold KV (the RED/ORANGE freeze would otherwise lock in the running count of the ramp-up, 3 of about 18 KV-bound slots in the 10× overload simulation).

RED refills finished slots (user decision 2026-09-26): its `pressure_red` queueing applies to new arrivals; a queued request offered a freed slot is judged by ORANGE's rules, so expensive prefills keep waiting. The admitted count never rises in RED and every refill holds its worst-case KV reservation, so KV stays within capacity. In the simulation, admit-nothing RED completed 16 of 7,019 requests in 600 s at 10× load; refilling completes 491, and refilling expensive prefills too completes only 329 (their prefill pushes KV into SURVIVAL).

Returning below RED requires the emergency reserve to be re-acquired.

SURVIVAL liveness (decision 2026-09-27, provisional A pending user review): SURVIVAL runs no new prefill, so without a release the worst-case reservations of admitted-but-unstarted requests and of prefills in progress can keep `kv_utilization` above SURVIVAL's exit threshold forever (the simulation's seed 6 jumped GREEN → SURVIVAL during the ramp and stayed there, 18 unstarted and 3 prefilling requests holding 0.934 of the pool). Under `reliability.recovery.survival_liveness: requeue_unstarted` (A) the SURVIVAL plan requeues the unstarted requests (a `queue` decision with reason `survival_requeue`; `503 overloaded` when the queue is full) and drops their reservations; under `continue_prefills` (B) the in-progress prefills finish at RED's prefill budget and chunk floor. Measured over seeds 1–12 with the KV headroom rule (S-9): every seed returns to GREEN + HEALTHY 42–49 s after the load stops under either option; seed 6 recovers in 42.9 s (A, 247 completions) or 46.9 s (B, 342).

### Admission decisions and HTTP mapping

Decision reason codes (closed enums, used as metric labels):

- `Queue` reasons: `kv_reservation` (not enough unreserved KV, or not enough KV headroom below the next state's threshold, S-9), `pressure_orange`, `pressure_red`, `circuit_degraded`, `prefill_budget`, `survival_requeue` (an admitted request that had not started returned to the queue in SURVIVAL, S-11).
- `Reject` reasons and responses (OpenAI error shape from Phase 0):

| Reason                        | Status | `type`                  | `code`                        | `Retry-After`                                |
| ----------------------------- | ------ | ----------------------- | ----------------------------- | -------------------------------------------- |
| `context_exceeds_kv_capacity` | 400    | `invalid_request_error` | `context_exceeds_kv_capacity` | —                                            |
| `queue_full`                  | 429    | `rate_limit_error`      | `queue_full`                  | estimated queue drain seconds, clamped 1..60 |
| `queue_timeout`               | 503    | `service_unavailable`   | `queue_timeout`               | same                                         |
| `survival`                    | 503    | `service_unavailable`   | `overloaded`                  | same                                         |
| `circuit_open`                | 503    | `service_unavailable`   | `circuit_open`                | remaining cooldown seconds, min 1            |

- A running streaming request that fails after retries are exhausted receives one final SSE event `data: {"error":{"message":…,"type":"server_error","code":"resource_exhausted"}}` followed by `data: [DONE]`; a non-streaming one receives `503` with code `resource_exhausted`.
- Queue order: FIFO within the Phase 2 priority; a smaller request that fits may overtake a head that does not fit at most `max_bypass` times, after which the head blocks the queue (starvation guard).

### Circuit breaker transitions

| From         | To           | Trigger (reason code)                                                                                                                       |
| ------------ | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| HEALTHY      | DEGRADED     | `latency_drift` ≥ `latency_drift_degraded`; `thermal_throttle`; one `oom_recovered`; `telemetry_stale`                                      |
| DEGRADED     | HEALTHY      | no trigger for `window`                                                                                                                     |
| any          | CIRCUIT_OPEN | `oom_recoveries_to_open` recoveries in `window` (`repeated_oom`); `latency_drift` ≥ `latency_drift_open`; `recovery_failed`; `device_error` |
| CIRCUIT_OPEN | DRAINING     | immediately: admission stops, queue is rejected with `circuit_open`, running sequences continue                                             |
| DRAINING     | PROBING      | running sequences finished (or failed at `drain_timeout`) and `cooldown` elapsed                                                            |
| PROBING      | HEALTHY      | `probe_successes` consecutive probe requests (internal 16-token greedy generation of a fixed prompt) succeed within 2 × baseline latency    |
| PROBING      | CIRCUIT_OPEN | a probe fails or times out (`probe_failed`)                                                                                                 |

`latency_drift` feeds the circuit only while the pressure state is GREEN (decision "Phase 3: soak config max_batch_tokens", answer option 2, 2026-09-27, provisional): the circuit owns device health and the pressure controller owns load, so above GREEN a slower step (bigger batches, longer contexts) raises only the `step_time_drift` pressure signal. The probe and drain logic are unchanged.

DEGRADED raises the pressure floor to YELLOW. A sticky device error (HIP `hipErrorIllegalAddress` / `hipErrorLaunchFailure`, NVIDIA `cudaErrorIllegalAddress`-class — any error the vendor marks as context-corrupting) goes to CIRCUIT_OPEN with reason `device_fatal`, drains without running further device work (running sequences fail with `resource_exhausted`), and exits the process with code 3.

### HTTP routes (changes from Phase 2)

| Route                      | Phase 3 behaviour                                                                                  |
| -------------------------- | -------------------------------------------------------------------------------------------------- |
| `GET /turbine/v1/pressure` | `200` pressure document (see Data)                                                                 |
| `GET /turbine/v1/status`   | adds `"pressure_state"` and `"circuit_state"`                                                      |
| `GET /ready`               | `503` `{"ready":false,"reason":"circuit_open"}` while circuit is CIRCUIT_OPEN, DRAINING or PROBING |
| OpenAI inference routes    | admission errors per the table above                                                               |

### Metrics (all label values from closed sets)

- `turbine_pressure_state{state}` gauge (1 for the current state, 0 otherwise); `turbine_pressure_transitions_total{from,to,signal}` counter; `turbine_pressure_signal{signal}` gauge; `turbine_pressure_signal_level{signal}` gauge (0–4); `turbine_pressure_exhaustion_horizon_seconds` gauge (+Inf when not growing).
- `turbine_admission_decisions_total{decision,reason}` counter (`reason="none"` for `Admit`); `turbine_admission_queue_depth` gauge; `turbine_admission_queue_wait_seconds` histogram.
- `turbine_throttle_plan{field}` gauge (`batch_growth_limit`, `prefill_budget_fraction`, `prefill_chunk_tokens`); `turbine_reclaim_bytes_total{action}` counter.
- `turbine_memory_pool_bytes{device,pool,kind}` gauge (`kind` ∈ `capacity`, `used`, `reserved`); `turbine_emergency_reserve_held{device}` gauge; `turbine_emergency_reserve_releases_total{device}` counter.
- `turbine_allocation_failures_total{device,pool}`, `turbine_recoveries_total{outcome}` (`recovered`, `failed`), `turbine_recovery_retries_total` counters.
- `turbine_circuit_state{state}` gauge; `turbine_circuit_transitions_total{from,to,reason}` counter.
- `turbine_gpu_temperature_celsius{device}`, `turbine_gpu_clock_mhz{device}`, `turbine_gpu_power_watts{device}`, `turbine_gpu_utilization_ratio{device}`, `turbine_gpu_memory_bytes{device,kind}` (`used`, `free`; dedicated only), `turbine_gpu_throttle_active{device,reason}` (`thermal`, `power`, `other`) gauges; `turbine_host_memory_available_bytes`, `turbine_host_psi_memory_some_avg10`, `turbine_host_swap_in_pages_per_second` gauges; `turbine_telemetry_stale{source}` gauge; `turbine_telemetry_call_duration_seconds{source}` histogram. `device` is the Phase 0 global index.

### Structured log events

`pressure_transition` (INFO), `admission_decision` (DEBUG for `Admit`, INFO otherwise), `throttle_plan_changed` (INFO), `reclaim` (INFO), `allocation_failure` (WARN), `recovery_attempt` / `recovery_outcome` (WARN), `circuit_transition` (WARN), `telemetry_stale` (WARN), `emergency_reserve` (WARN). Each carries its reason code and the numeric inputs to the decision.

### `turbine-bench` additions

```
turbine-bench ... [--duration <dur>] [--rate <req/s>] [--prompt-words-range <min>..<max>]
                  [--max-tokens-range <min>..<max>] [--pressure-timeline <file.jsonl>]
```

- `--duration` runs until the duration elapses instead of `--requests`; `--rate` switches to open-loop arrivals with exponential inter-arrival times from the seeded PRNG, with `--concurrency` as the cap on outstanding requests (arrivals beyond it are counted `client_dropped`). Ranges draw uniformly per request from the seeded PRNG.
- `--pressure-timeline` polls `<url>/turbine/v1/pressure` every second and writes one JSON line per sample (`t`, `state`, `circuit`, `dominant_signal`, `queue`, `kv_utilization`); a failed poll writes `{"t":…,"error":…}`.
- The report adds `by_status` (`{"200":n,"429":n,…}`), `by_error_code` (`{"queue_full":n,…}`), `client_dropped`, and for 200 responses `streams_incomplete` (no `[DONE]`).

### `scripts/overload-soak.sh`

```
scripts/overload-soak.sh <novanas|dgx-spark|dgx-spark2> [--duration <dur>] [--model <path>]
```

0. Prints the host, the GPU it will use and the memory it will claim, and refuses to start (exit 1, `precondition`) unless the target R9700 reports less than 1 GiB VRAM in use (novanas) or `MemAvailable` exceeds the container cap plus `host_reserve_bytes` (Spark); freeing a host is the user's decision, asked beforehand, never the script's.
1. Builds and starts `turbine-server` in a container on the host (novanas: k3s Job with `amd.com/gpu: 1`, the `rust:1.97-trixie` image and ROCm mounted read-only from `/opt/rocm/rocm` as in phase-0/phase-1 (CONFLICT C-7); Spark: `docker run --gpus all --memory 24g`), with `--model` defaulting to `/home/piwi/turbine-models/llama-3.2-3b-instruct` mounted read-only, waiting for `/ready`.
2. Calibrate (2 min): closed-loop at `--concurrency 4`, recording baseline ITL p99 and request throughput R.
3. Overload (`--duration`, default `10m`; the one pre-exit long run uses `--duration 4h`): open-loop `--rate 4R`, `--prompt-words-range 64..6000`, `--max-tokens-range 16..1024`, `--concurrency 1024`, pressure timeline on.
4. Cool-down (5 min): no load, pressure timeline on.
5. Evaluates the pass criteria, prints a JSON verdict, exits 0 on pass and 1 on fail, and always removes its container/Job.

## Data

- Nothing is persisted. The budget, ledger, signal history (ring buffer of the last 600 samples per signal), transition history (32) and circuit history (32) live in memory in `turbine-reliability`.
- Pressure document (`GET /turbine/v1/pressure`):

```json
{
  "enabled": true,
  "state": "ORANGE",
  "since": "2026-09-25T18:02:11.482Z",
  "dominant_signal": "kv_utilization",
  "exhaustion_horizon_seconds": 14.2,
  "signals": [
    {
      "name": "kv_utilization",
      "value": 0.84,
      "level": "ORANGE",
      "stale": false
    },
    {
      "name": "host_available",
      "value": 3.1,
      "level": "YELLOW",
      "stale": false
    }
  ],
  "throttle": {
    "batch_growth_limit": 0,
    "prefill_budget_fraction": 0.5,
    "prefill_chunk_tokens": 256,
    "admission": "expensive_queued"
  },
  "memory": [
    {
      "device": 0,
      "memory_kind": "dedicated",
      "budget_bytes": 30064771072,
      "pools": [
        {
          "name": "weights",
          "capacity_bytes": 0,
          "used_bytes": 0,
          "reserved_bytes": 0
        },
        {
          "name": "kv",
          "capacity_bytes": 0,
          "used_bytes": 0,
          "reserved_bytes": 0
        },
        {
          "name": "workspace",
          "capacity_bytes": 0,
          "used_bytes": 0,
          "reserved_bytes": 0
        },
        {
          "name": "runtime",
          "capacity_bytes": 0,
          "used_bytes": 0,
          "reserved_bytes": 0
        },
        {
          "name": "reserve",
          "capacity_bytes": 2147483648,
          "used_bytes": 2147483648,
          "reserved_bytes": 0
        }
      ],
      "emergency_reserve_held": true
    }
  ],
  "admission": {
    "queued": 17,
    "max_queue": 256,
    "decisions": {
      "admit": 1200,
      "queue": { "pressure_orange": 40 },
      "reject": { "queue_full": 3 }
    }
  },
  "circuit": {
    "state": "HEALTHY",
    "since": "2026-09-25T17:40:00.000Z",
    "last_reason": null
  },
  "transitions": [
    {
      "at": "2026-09-25T18:02:11.482Z",
      "from": "YELLOW",
      "to": "ORANGE",
      "signal": "kv_utilization",
      "value": 0.83,
      "threshold": 0.82
    }
  ]
}
```

- `ResourceEstimate` (in-process): `prompt_tokens`, `cached_prefix_tokens`, `new_prefill_tokens`, `max_output_tokens` (request `max_tokens`, else context length minus prompt), `projected_kv_blocks` = ceil((prompt + max_output) / block tokens) − cached full blocks, `workspace_bytes`, `est_prefill_seconds`, `est_decode_seconds`. The EWMAs (α = 0.1) of prefill tokens/s and decode step time are owned by the controller and seeded from the first 32 iterations; before seeding, estimates use the calibration step's figures and are marked `estimated: false` in logs.
- Soak outputs: the bench JSON report, the pressure timeline JSONL and the verdict JSON are written by `scripts/overload-soak.sh` to `target/soak/<host>-<timestamp>/` on the workstation.

## Edge cases

- Request whose prompt alone exceeds KV capacity, or whose prompt + `max_tokens` exceeds it (reject `context_exceeds_kv_capacity`), versus one that exceeds only current free capacity (queue `kv_reservation`).
- Queue head that never fits while small requests keep arriving (bypass limit), and a queued request whose client disconnects (removed from the queue, reservation never taken, counted as cancelled).
- Signal oscillating around a threshold (hysteresis must hold the level); two signals crossing opposite directions in the same tick; a signal staying stale forever (floor YELLOW, circuit DEGRADED, never GREEN).
- Allocation failure during prefill of a request that already streamed nothing versus during decode of a sequence that already streamed tokens; allocation failure during the recovery retry itself; allocation failure while the reserve is already released.
- Emergency reserve re-acquisition fails because another process took the memory (stay at RED, log, retry each tick).
- `emergency_vram_reserve: 0`; `max_retries: 0`; `reliability.enabled: false` with an OOM (still bounded recovery).
- Request without `max_tokens` on a 128k-context model: the worst-case reservation is prompt + remaining _model.max_seq_len_ tokens (about 14 GiB of KV for Llama-3.2-3B at 128k), which on a 32 GB R9700 means only one or two such requests run at once and the rest queue with `kv_reservation`; operators lower _model.max_seq_len_ or clients send `max_tokens`. The estimator takes per-token KV bytes from the model's KV layout description (GQA heads, layers), never from assumed dimensions.
- Unified memory (GB10): NVML memory queries return not-supported; device free memory is unknown, the budget is `MemAvailable` at startup − `host_reserve_bytes` (capped by `device_budget_bytes`), `device_memory` is not computed and the host signals carry the load.
- Co-tenant process on the same device grows after startup (MemAvailable or device free falls without Turbine allocating).
- Telemetry vendor call hangs; `/proc/pressure/memory` absent (kernel without PSI); swap disabled (swap signal reports 0).
- Thermal throttle reported while the device is idle (ignore when no work ran in the last window: drift and thermal circuit triggers require at least one iteration in the window).
- Latency baseline learned while already under pressure (baseline taken only from GREEN iterations; reset after a circuit returns to HEALTHY).
- SIGTERM while in SURVIVAL or DRAINING (graceful shutdown still completes within its Phase 2 bound).
- Probe request itself hits admission (probes bypass the queue and are not counted as client requests).

## Failure modes

- **Budget impossible at startup:** exit 1 before binding the API port, listing every pool and its bytes and the measured free memory.
- **Telemetry library missing or failing:** that source is `unavailable`; the corresponding signals are omitted (not stale) and logged once at WARN; the controller runs on the remaining signals. A source that worked and then stops answering is `stale`.
- **Device OOM:** SURVIVAL, reclaim, bounded retries; on exhaustion fail the batch's requests with `resource_exhausted`, count `recoveries_total{outcome="failed"}`, open the circuit if repeated.
- **Sticky device error:** CIRCUIT_OPEN (`device_fatal`), fail running sequences, `/ready` 503, exit code 3 within `drain_timeout`; the external supervisor restarts the process, which reloads weights and re-measures the budget.
- **Host memory pressure (swap, PSI):** escalates through the host signals; SURVIVAL releases optional host buffers; the engine never allocates host memory past `host_reserve_bytes` headroom intentionally.
- **Clock skew or suspended process:** the controller uses a monotonic clock; a tick gap larger than 10 × interval discards rate-based signals for that tick instead of computing huge rates.
- **Controller panic:** the controller task is supervised; a panic sets the state to SURVIVAL and circuit to CIRCUIT_OPEN with reason `controller_failed` and the process exits with code 3 after drain — pressure control is never silently lost.
- **Soak script dependencies missing (SSH, Docker, kubectl) or server fails to become ready within 10 min:** exit 1 naming the step; containers/Jobs it started are removed.

## Acceptance criteria

- [ ] [S-1] `cargo build --workspace`, `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --all --check` exit 0 on macOS arm64 with no GPU, and `cargo tree -p turbine-reliability` lists no `nvml-wrapper`, `libloading` or GPU crate; fails if the reliability crate gains a GPU dependency or `unsafe`.
- [ ] [S-2] `cargo test -p turbine-reliability budget::tests::pools_partition_free_memory` exits 0; it asserts that for a dedicated device with measured free 30 GiB, weights 6 GiB, workspace 1 GiB, overhead 1 GiB and reserve 2 GiB the KV pool is 20 GiB and the pools sum to the budget, and that `device_budget_bytes: 20GiB` caps the budget (KV pool 10 GiB); fails if pools overlap or exceed free memory.
- [ ] [S-2] `cargo test -p turbine-reliability budget::tests::unified_budget_from_mem_available` exits 0; it asserts that for a unified device with `MemAvailable` 42 GiB and `host_reserve_bytes: 8GiB` the budget is 34 GiB, that `device_budget_bytes: 24GiB` lowers it to 24 GiB, that a larger cap does not raise it, and that no `device_memory` signal is produced; fails if a unified budget uses device total or ignores the host reserve or the cap.
- [ ] [S-2] `cargo test -p turbine-reliability budget::tests::impossible_budget_rejected` exits 0; it asserts a budget that cannot hold one full-context sequence returns an error naming each pool and its bytes; fails if the engine would start with a KV pool too small for one sequence.
- [ ] [S-3] `cargo test -p turbine-reliability ledger::tests::reservations_never_exceed_capacity` exits 0; it runs 10,000 seeded random reserve/commit/release/drop operations from 8 threads and asserts used + reserved ≤ capacity at every step and 0 after all guards drop; fails if a dropped guard leaks or the pool over-commits.
- [ ] [S-3] [S-9] `cargo test -p turbine-scheduler --test overload_sim cancellation_releases_reservations` exits 0; it cancels 100 queued and 100 running simulated requests and asserts the KV and workspace ledgers return to their idle values within one scheduler iteration; fails if cancellation leaves a reservation behind.
- [ ] [S-4] `cargo test -p turbine-reliability reserve::tests::reserve_only_released_in_survival` exits 0; it asserts the reserve cannot be released in GREEN–RED, is released on entering SURVIVAL via recovery, and that the state cannot drop below RED until re-acquisition succeeds; fails if normal scheduling can consume the reserve.
- [ ] [S-5] `cargo test -p turbine-device telemetry::tests::proc_parsers` exits 0; it parses fixture `/proc/meminfo`, `/proc/vmstat` and `/proc/pressure/memory` files and asserts MemAvailable bytes, pswpin deltas and `some avg10`; and asserts a missing PSI file yields source `unavailable`; fails if a parser mis-scales kB or crashes on a missing file.
- [ ] [S-5] `cargo test -p turbine-device telemetry::tests::two_cadences` exits 0; with a fake clock, fake `/proc` source and counting vendor backend at defaults it asserts 100 ± 1 fast ticks and 10 ± 1 vendor calls per device in 10 simulated seconds, and that `kv_utilization` reflects a ledger change within one fast tick; fails if vendor calls run on the fast tick or memory signals wait for the vendor tick.
- [ ] [S-5] `cargo test -p turbine-device telemetry::tests::hung_call_marks_stale` exits 0; it injects a vendor backend that blocks for 5 s with `call_timeout: 500ms` and asserts the sample returns within 600 ms marked stale and later samples continue; fails if a hung vendor call blocks the sampler.
- [ ] [S-5] [S-14] `cargo test -p turbine-device --test lab live_telemetry -- --ignored` exits 0 under `scripts/lab-test.sh dgx-spark`, `scripts/lab-test.sh dgx-spark2` and `scripts/lab-test.sh novanas`; it samples 10 vendor ticks and asserts non-stale temperature and clock on every device, `used`/`free` memory on the two R9700s, no device memory on the GB10s, and host MemAvailable > 0 on each host; telemetry is read-only and needs no memory freed; fails if a real device yields no temperature or clock.
- [ ] [S-6] [S-7] `cargo test -p turbine-reliability state::tests::hysteresis_holds` exits 0; it feeds `kv_utilization` oscillating 0.81/0.83 around the 0.82 ORANGE threshold for 60 simulated seconds and asserts exactly one transition to ORANGE and none back until the value stays below 0.779 for `deescalate_dwell`; fails if the state flaps.
- [ ] [S-7] `cargo test -p turbine-reliability state::tests::escalation_and_stepwise_deescalation` exits 0; it asserts allocation failure moves GREEN→SURVIVAL in one sample, other signals need `escalate_samples` samples, and recovery descends SURVIVAL→RED→ORANGE→YELLOW→GREEN one level per dwell; fails if de-escalation skips levels or escalation ignores the sample count.
- [ ] [S-7] [S-14] `cargo test -p turbine-reliability state::tests::transitions_explained` exits 0; it asserts each transition produces a history entry, a `turbine_pressure_transitions_total` increment and a `pressure_transition` log event carrying signal, value and threshold, and that history is capped at 32; fails if any transition lacks a reason.
- [ ] [S-8] `cargo test -p turbine-reliability horizon::tests::predicts_exhaustion` exits 0; it simulates 20 sequences each with 500 remaining tokens growing at 50 tokens/s against 200 free blocks of 16 tokens and asserts the horizon is 3.2 s ± 10 %, and +Inf with no running sequences; fails if the horizon ignores committed growth.
- [ ] [S-9] `cargo test -p turbine-reliability admission::tests::decision_table` exits 0; it asserts `Reject(context_exceeds_kv_capacity)` for prompt + max_tokens above pool capacity, `Queue(kv_reservation)` when only free capacity is short, `Queue(pressure_orange)` for an expensive prefill in ORANGE while a cheap one is admitted, `Queue(pressure_red)` for every request in RED, `Reject(survival)` in SURVIVAL, `Reject(queue_full)` at `max_queue`, admission bypassing pressure checks when `adaptive_admission: false`, and that an admitted request without `max_tokens` reserves ceil(max_seq_len / block tokens) blocks (worst case) while one with `max_tokens: 100` reserves ceil((prompt + 100) / block tokens); fails if any row of the decision table changes or a reservation is below the worst case.
- [ ] [S-9] [S-10] `cargo test -p turbine-reliability admission::tests::red_refill_from_queue` exits 0; it asserts that in RED a new request queues (`pressure_red`) while `Admission::evaluate_refill` admits a queued cheap request into a finished slot, keeps an expensive one queued (`pressure_orange`, as in ORANGE), still answers `Queue(kv_reservation)` when unreserved KV is short, `Reject(survival)` in SURVIVAL and `Reject(circuit_open)` with an open circuit, and records no decision metric; fails if RED refills an expensive prefill or a refill bypasses the KV reservation.
- [ ] [S-9] `cargo test -p turbine-reliability admission::tests::bypass_bounded` exits 0; it asserts a non-fitting queue head is overtaken exactly `max_bypass` times before smaller requests wait behind it; fails if the head can starve.
- [ ] [S-10] `cargo test -p turbine-reliability throttle::tests::plan_per_state` exits 0; it asserts every cell of the throttle-plan table, including the chunk floor of 4 × block tokens and the call to `KvReclaimer::demote` in YELLOW and ORANGE; fails if a state's plan differs from the table.
- [ ] [S-10] [S-17] `cargo test -p turbine-scheduler --test overload_sim ten_x_overload` exits 0; it drives 10× service-rate arrivals (seed 7) for 600 simulated seconds then 120 s of silence and asserts: simulated KV never exceeds capacity, no running sequence is preempted below SURVIVAL, the state reaches RED, the admitted count never rises between two consecutive RED iterations, every request ends completed, cancelled or rejected with a code from the table, the queue never exceeds `max_queue`, completions reach at least 50 % of the capacity bound (`OverloadConfig::service_rate()` × 600 s = 688 requests), and the state is GREEN within 60 s of the load stopping; the run repeats for every registered scheduling policy (Phase 2m); fails if any assertion breaks. The 50 % floor sits below the measured 72 % (493 completions) and above the two failure modes it guards against: admit-nothing RED (16, 2 %) and freezing the running rather than the admitted count (166, 24 %); across seeds 1–12 the design measures 137–581 completions (20–84 %) — seed 6's SURVIVAL liveness gap is fixed by S-11 and tested below.
- [ ] [S-11] [S-17] `cargo test -p turbine-scheduler --test overload_sim survival_` exits 0: `survival_liveness_seed_6` and `survival_liveness_seed_1` run the `ten_x_overload` workload with seeds 6 and 1 under `survival_liveness: requeue_unstarted` and assert every overload invariant (KV within capacity, no preemption below SURVIVAL, no growth in RED, every request an outcome from the reject table, queue within `max_queue`) plus GREEN + HEALTHY within 60 s of the load stopping (seed 6 must pass through SURVIVAL); `survival_liveness_option_b` asserts the same for both seeds under `continue_prefills`; `survival_requeues_unstarted_admitted` asserts that an OOM-triggered SURVIVAL returns the admitted requests that had not started to the admission queue with their reservations released, the started ones keep theirs, and all complete afterwards; fails if SURVIVAL can hold KV that nothing releases (seed 6 stuck in SURVIVAL, as before the fix) or recovery takes longer than 60 s (seed 1 took 64 s before the KV headroom rule).
- [ ] [S-9] `cargo test -p turbine-reliability admission::tests::kv_headroom` and `recovery::tests::survival_plan_switch` exit 0: in YELLOW / ORANGE / RED an admission or refill that would lift `kv_utilization` past the next state's threshold queues `kv_reservation` (at GREEN past RED's threshold, `admission::tests::kv_headroom_green_burst`; never without adaptive admission); `survival_liveness` changes only the SURVIVAL row (A: no prefill, requeue; B: in-progress prefills at 50 % and the chunk floor); fails if admissions can escalate the state or the switch does not reach the plan.
- [ ] [S-10] [S-17] `cargo test -p turbine-scheduler --test overload_sim active_generations_protected` exits 0; it starts 8 long decodes, then floods 2,000 large prefills and asserts the 8 decodes' simulated step time stays within 1.5× their pre-flood value and all 8 complete; fails if new work starves running generations.
- [ ] [S-11] `cargo test -p turbine-scheduler --test overload_sim oom_recovery_bounded` exits 0; it injects device OOM on the next 2 iterations and asserts recovery shrinks the batch and succeeds on retry 3; then injects persistent OOM and asserts exactly `max_retries` retries, the batch's requests fail with `resource_exhausted`, the SSE stream ends with the error event then `[DONE]`, and the worker keeps serving new requests; fails if recovery retries forever or kills the worker.
- [ ] [S-12] `cargo test -p turbine-reliability circuit::tests::transition_table` exits 0; it asserts every row of the circuit transition table with a fake clock, including DEGRADED→HEALTHY after `window`, PROBING→CIRCUIT_OPEN on a failed probe, and drift/thermal triggers ignored with no iterations in the window; fails if any row differs.
- [ ] [S-12] [S-13] `cargo test -p turbine-api --test api ready_follows_circuit` exits 0; it drives a stub controller through CIRCUIT_OPEN, DRAINING, PROBING and HEALTHY and asserts `/ready` is 503 with reason `circuit_open` for the first three and 200 after, and that chat completions return `503 circuit_open` with `Retry-After` ≥ 1 while open; fails if readiness ignores the circuit.
- [ ] [S-12] [S-16] Lab run (after the user confirms an R9700 is free): `scripts/lab-test.sh novanas` with `--features fault-injection` exits 0 including `cargo test -p turbine-server --test fault sticky_device_error_exits_3 -- --ignored`, which starts the server with `kernel_error_at_iteration` configured as a sticky error (named with the backend's own sticky error name, Phase 2m `ExecutionBackend::sticky_error_prefixes`), sends a request, and asserts `/ready` 503 then process exit code 3 within `drain_timeout`; the same test then passes under `scripts/lab-test.sh dgx-spark` once the NVIDIA path exists; fails if the process keeps serving on a corrupted context or exits with another code.
- [ ] [S-13] `cargo test -p turbine-api --test api pressure_document_shape` exits 0; it asserts `GET /turbine/v1/pressure` returns 200 with every key of the Data example and types matching, `/turbine/v1/status` includes `pressure_state` and `circuit_state`, and `reliability.enabled: false` yields `"enabled": false` and state GREEN; fails if the route still returns 501 or a key is missing.
- [ ] [S-13] `cargo test -p turbine-api --test api admission_error_mapping` exits 0; it asserts status, `type`, `code` and `Retry-After` for each row of the reject table; fails if any overload response lacks a stable code or retry hint.
- [ ] [S-14] `cargo test -p turbine-api --test api reliability_metrics_bounded` exits 0; it drives a stub controller through every state and reason and asserts all metric families under Interfaces appear in `/metrics` and that no label value outside the documented closed sets appears; fails if a family is missing or a label is unbounded.
- [ ] [S-15] `cargo test -p turbine-core config::tests::reliability_config_validation` exits 0; it asserts the defaults in the configuration table, durations `100ms`/`10s`/`1h` parse, and each of: non-monotonic thresholds, `latency_drift_open: 1.5` with degraded 2.0, `deescalate_dwell: 10ms`, `vendor_interval: 50ms` with `interval: 100ms`, `stale_after: 1s` with `vendor_interval: 1s`, `max_queue: 0`, the removed keys `reliability.admission.kv_overcommit` and `scheduler.queue_timeout` (CONFLICT C-1), and `reliability.fault_injection` without the feature is rejected naming the key; fails if an impossible reliability config is accepted.
- [ ] [S-16] `cargo test -p turbine-server --features fault-injection --test fault alloc_fail_every_injects` exits 0 on macOS using the simulated executor; it asserts `alloc_fail_every: 5` fails exactly every fifth pool allocation and each failure is counted in `turbine_allocation_failures_total`; fails if injection is inaccurate or compiled into default builds (a default build must reject the section).
- [ ] [S-18] `cargo test -p turbine-bench --test bench open_loop_rate_and_breakdown` exits 0; it runs `--rate 50 --duration 4s --seed 3` against a mock server returning 200, 429 `queue_full` and 503 `overloaded` in a fixed pattern and asserts 200 ± 30 arrivals, `by_status` and `by_error_code` counts matching the mock, identical arrival schedules for the same seed, and a pressure timeline with 4 ± 1 lines; fails if arrivals are closed-loop or error codes are not broken down.
- [ ] [S-19] `scripts/overload-soak.sh novanas` run on a host where the R9700 still holds another workload exits 1 with step `precondition` and starts no Job; fails if the script claims a busy GPU or frees memory itself.
- [ ] [S-19] [S-4] [S-11] Manual lab runs, each after the user confirms the host may be used: `scripts/overload-soak.sh novanas` (10 min overload) exits 0; then, before the Phase 3 exit, `scripts/overload-soak.sh novanas --duration 4h` exits 0; and `scripts/overload-soak.sh dgx-spark` (10 min, 24 GiB container cap) exits 0 once the NVIDIA path exists. Each verdict JSON is pasted into the task evidence showing: the server process never exited or restarted and its container was not OOM-killed; zero HTTP 5xx other than `503` with codes `overloaded`, `queue_timeout`, `circuit_open`; zero `streams_incomplete`; admitted requests' ITL p99 during overload ≤ 2× the calibration ITL p99; the timeline reached at least ORANGE during overload; state GREEN and circuit HEALTHY within 60 s of cool-down start; KV `used_bytes` and `reserved_bytes` back to the idle values and `emergency_reserve_held: true` at the end; fails if any of these does not hold.

## Open questions

<!-- None: decisions recorded in .procoder/ask/decisions.md -->
