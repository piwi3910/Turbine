# phase-3-reliability — implementation plan

Status: draft
Spec: .procoder/specs/phase-3-reliability.md

## Goal

Make overload survivable and provable: explicit memory budgets and reservations, live telemetry feeding a hysteretic pressure state machine, predictive admission with worst-case KV reservation, a throttle ladder the scheduler obeys, bounded OOM recovery, a circuit breaker that exits with code 3 on sticky device errors, and an overload soak whose pass criterion is "no worker death, no OOM, recovery to GREEN".

## Architecture

A new GPU-free crate `turbine-reliability` owns the budget, reservation ledger, emergency reserve, signals, state machine, exhaustion horizon, admission (with its bounded queue), throttle planner, recovery controller, circuit breaker and the `PressureController` that ties them together on the telemetry fast tick and publishes one atomic `Snapshot` (read through `ControllerHandle`) that the engine reads once per iteration. `turbine-device` gains the two-cadence `TelemetrySampler` (`/proc` + ledger every 100 ms, NVML/amd-smi every 1 s with a deadline per call) publishing through a lock-free `LatestSample`; the shared vocabulary (`PressureSignal`, telemetry samples, the `reliability` config) lives in `turbine-core` so `turbine-reliability` never depends on `turbine-device`. `turbine-kv`, `turbine-scheduler`, `turbine-api`, `turbine-server` and `turbine-bench` are extended to pay every KV block from a ledger reservation, replace "admit waiting" by admission decisions, serve `/turbine/v1/pressure` and the admission errors, wire controller/sampler/recovery/circuit into the engine thread, and generate open-loop overload; `scripts/overload-soak.sh` runs calibrate → overload → cool-down on the lab hosts.

## Constraints

Copied verbatim from the spec (§Constraints):

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

Inherited from the contract (`.procoder/contract/interfaces.md`, binding):

- Edition 2024, `rust-version = "1.97"`, Apache-2.0; public enums a later phase extends are `#[non_exhaustive]`; config structs are `#[serde(deny_unknown_fields, default)]`; one `thiserror` error enum per crate.
- Every time-dependent component takes `Arc<dyn turbine_core::clock::Clock>` and never calls `Instant::now()` directly.
- Metric label values come only from closed enums' `as_str()` or the Phase 0 device index; `PressureState`/`CircuitState` serialise `SCREAMING_SNAKE_CASE` everywhere (CONFLICT C-5).
- One waiting queue: from Phase 3 it is the admission queue bounded by `reliability.admission.max_queue` / `queue_timeout`; `scheduler.max_queued_requests` is only the HTTP→engine channel capacity; `scheduler.queue_timeout` is rejected as a removed key (CONFLICT C-1).
- `kv.gpu.max_bytes` default null = the `kv` pool remainder of the budget; an explicit value caps the `kv` pool (CONFLICT C-8).
- Mid-stream errors are one `data: {"error":…}` event followed by `data: [DONE]` (CONFLICT C-3); the P2 `device_error` readiness reason and 3-failed-iterations exit are retired in favour of the circuit breaker (CONFLICT C-25).
- `unsafe` only in `crates/turbine-device/src` and `crates/turbine-kernels/src`, each block preceded by `// SAFETY:`.
- Lab: novanas runs k3s Jobs, the Sparks `docker run --gpus all` containers named `turbine-lab-*` (CONFLICT C-24); no `docker run` outside the lab scripts.
- Gate for every task: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`; tasks touching the `fault-injection` feature also run `cargo clippy --workspace --all-targets --all-features -- -D warnings`.
- prometheus-client appends `_total` to counters: counter families are registered through `MetricsRegistry::register` without the suffix (e.g. `turbine_pressure_transitions`, rendered `turbine_pressure_transitions_total`).

## Port onto the Phase 2m layout (2026-09-27)

Tasks 1–18 were first built on the pre-refactor tree (branch `runahead/p3-reliability`) and ported onto main after Phase 2m (registries for families, tool formats, weight formats, backends, card profiles, kernel implementations, logits processors and scheduling policies). Where Phase 3 touches an extension point it goes through the registry; no family, backend or vendor branch was added:

- Task 11: the vendor telemetry backends are one file each (`crates/turbine-device/src/telemetry/{nvml.rs, amd_smi.rs}`) and are opened through the `device_discovery` registry: `DiscoveryKind::telemetry(&DiscoveryOptions)`; `telemetry::vendor::vendor_backends(opts: &DiscoveryOptions)` iterates the registry (was `vendor_backends(nvml, amd_smi)`); `docs/extending/backend.md` names the new method.
- Task 12: the admission gate's queue is ordered by the scheduling policy's `AdmissionKey` (`AdmissionQueue<T, K>`, `push_keyed`; the gate keeps each request's `submit_no` so an admitted request keeps its order), `Scheduler::with_policy` applies to gated requests, `OverloadConfig::policy` names a registered policy and `ten_x_overload` runs over every registered policy; `docs/extending/scheduling-policy.md` notes that the key also orders the gate.
- Task 14: error classification comes from the backend registry (`KernelError::is_oom`, `is_sticky` over `ExecutionBackend::sticky_error_prefixes`); `FaultyExecutor::new(inner, injector, sticky_name)` names an injected sticky error with the opened backend's first sticky prefix (the first registered backend's when the backend has none, e.g. `cpu`) instead of a hard-coded HIP name, and forwards the Phase 2c `launch` / `collect` / decode-graph methods. The Phase 2c overlap loop (`execution.overlap_scheduling`) got the same reliability rules: an out-of-memory launch finishes the iteration in flight and runs the plan through the serial recovery path; any other launch error, or an error collecting an iteration the scheduler already completed ahead, fails its requests and goes to the circuit breaker (`engine::loop::tests::{overlap_failed_steps_fail_their_requests_only, overlap_oom_launch_recovers_serially}`). `model::load` reads the weight format from the family registry (`turbine_model_weight_bytes{format}`), decode graphs stay wired, and `PreparedModel` keeps `tool_format` / `modules` next to the P3 budget fields.
- Task 18: `lab_scripts phase3_soak_config_loads` compares the module name (`execution.backend` is an open `ModuleName` since Phase 2m).

## Task 1: Reliability configuration, pressure vocabulary and telemetry types in turbine-core

Files: `crates/turbine-core/src/config/reliability.rs` (new: `reliability` section structs + static validation), `crates/turbine-core/src/config/mod.rs` (re-export, `Config::validate` hook, `scheduler.queue_timeout` removed-key, test), `crates/turbine-core/src/types.rs` (`PressureSignal`; helpers on `PressureState`/`CircuitState`), `crates/turbine-core/src/telemetry.rs` (new: telemetry vocabulary), `crates/turbine-core/src/lib.rs` (module), `crates/turbine-core/Cargo.toml` (feature `fault-injection = []`)
Interfaces:

- `pub struct ReliabilityConfig { enabled, emergency_vram_reserve: ByteSize, adaptive_admission, memory: ReliabilityMemoryConfig, telemetry: ReliabilityTelemetryConfig, pressure: PressureConfig, admission: AdmissionConfig, recovery: RecoveryConfig, circuit: CircuitConfig, fault_injection }`
- `pub fn ReliabilityConfig::validate(&self, block_tokens: u32) -> Result<(), ConfigError>`
- `pub struct PressureConfig { escalate_samples: u32, deescalate_dwell: HumanDuration, exit_margin: f64, thresholds: BTreeMap<PressureSignal, [Option<f64>; 4]> }` (map holds overrides only)
- `pub struct AdmissionConfig { max_queue: u32, queue_timeout: HumanDuration, large_prefill_tokens: u32, max_bypass: u32, kv_overcommit: Option<serde_norway::Value> /* removed key, always rejected */ }`
- `#[cfg(feature = "fault-injection")] pub struct FaultInjectionConfig { alloc_fail_every: Option<u32>, oom_at_iteration: Option<u64>, kernel_error_at_iteration: Option<u64>, kernel_error_sticky: bool, telemetry_temperature_c: Option<f64>, telemetry_delay: Option<HumanDuration> }`; without the feature the field is `Option<serde_norway::Value>` rejected by `validate`
- `pub enum PressureSignal { KvUtilization, DeviceMemory, HostAvailable, PsiMemorySomeAvg10, SwapInRate, ExhaustionHorizon, QueueFill, StepTimeDrift, Thermal, TelemetryStale, AllocationFailure }` + `ALL`, `as_str()`, `lower_is_worse()`
- `impl PressureState { ALL, as_u8(), as_str(), from_level(u8), lower() }`; `impl CircuitState { ALL, as_str(), blocks_readiness() }`
- `pub struct TelemetrySample { at_mono_ns: u64, host: HostSample, devices: Vec<DeviceSample>, ledger: LedgerSample, storage: Option<StorageSample> }`, `pub struct LedgerSample { kv_utilization: f64, queue_fill: f64 }`, `DeviceSample::empty(DeviceId, SourceStatus)`, `SourceStatus::{Ok, Unavailable, Stale}`, `pub trait LedgerProbe { fn kv_utilization(&self) -> f64; fn queue_fill(&self) -> f64; }`

Covers: S-15, S-16 (configuration half); `config::tests::reliability_config_validation`.
Depends on: phase-0 plan (`Config`, `ByteSize`, `ConfigError`, `ReliabilityConfig` P0 keys), phase-2 plan (`HumanDuration`, `SchedulerConfig`, `Clock`).

- [ ] Write failing test `crates/turbine-core/src/config/mod.rs` `config::tests::reliability_config_validation`: asserts every default of the P3 configuration table (`workspace_bytes` 1GiB, `host_reserve_bytes` 8GiB, `interval` 100ms, `vendor_interval` 1s, `call_timeout` 500ms, `stale_after` 5s, `escalate_samples` 2, `deescalate_dwell` 10s, `exit_margin` 0.05, `max_queue` 256, `queue_timeout` 30s, `large_prefill_tokens` 2048, `max_bypass` 8, `max_retries` 3, `backoff` 50ms, circuit 3/60s/2.0/4.0/30s/120s/3), that `100ms`/`10s`/`1h` and an override `kv_utilization: [0.6, 0.8, 0.9, null]` parse, and that each of `kv_utilization: [0.9, 0.8, 0.95, 0.97]`, `host_available: [1.0, 2.0, 4.0, 8.0]`, `latency_drift_open: 1.5` with degraded 2.0, `deescalate_dwell: 10ms`, `vendor_interval: 50ms` with `interval: 100ms`, `stale_after: 1s` with `vendor_interval: 1s`, `max_queue: 0`, `kv_overcommit: 1.5`, `scheduler.queue_timeout: 60s` and (default build) `fault_injection: {alloc_fail_every: 5}` fails `Config::validate` with `ConfigError::key()` equal to the full dotted key.
- [ ] Run: `cargo test -p turbine-core config::tests::reliability_config_validation` — expect FAIL
- [ ] Implement the `reliability` section in `config/reliability.rs` (replacing the P0 three-key struct) with hand-written `Default`s from the table and `validate(block_tokens)` returning `ConfigError::Invalid { key, reason }`; thresholds must be strictly ascending (descending when `lower_is_worse`) over their non-null values; removed keys are accepted by serde as `Option<serde_norway::Value>` with `#[serde(skip_serializing)]` so the error names the key rather than a serde "unknown field".
- [ ] Implement `PressureSignal` and the state helpers in `types.rs`, the telemetry vocabulary in `telemetry.rs`, and call `self.reliability.validate(self.kv.block_tokens)` plus the `scheduler.queue_timeout` removed-key check from `Config::validate`; replace every P2 read of `scheduler.queue_timeout` by `reliability.admission.queue_timeout`.
- [ ] Run: `cargo test -p turbine-core config::tests::reliability_config_validation && cargo test -p turbine-core --features fault-injection config::tests::reliability_config_validation` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(core): reliability config, pressure signals and telemetry vocabulary`

## Task 2: turbine-reliability crate, metric families, signal table and memory budget

Files: `Cargo.toml` (workspace dep `arc-swap = "1.9"`, see OPEN), `crates/turbine-reliability/Cargo.toml` (new crate: deps core, observability, arc-swap, prometheus-client, serde, serde_json, thiserror, tracing; feature `fault-injection = ["turbine-core/fault-injection"]`; `unsafe_code = "forbid"`), `crates/turbine-reliability/src/lib.rs` (module root, re-exports `Clock`, `PressureState`, `CircuitState`), `crates/turbine-reliability/src/metrics.rs` (all reliability families), `crates/turbine-reliability/src/signals.rs` (threshold table + evaluator), `crates/turbine-reliability/src/budget.rs` (budget and pools)
Interfaces:

- `pub struct ReliabilityMetrics { pressure_state, pressure_transitions, pressure_signal, pressure_signal_level, exhaustion_horizon_seconds, admission_decisions, admission_queue_depth, admission_queue_wait_seconds, throttle_plan, reclaim_bytes, memory_pool_bytes, emergency_reserve_held, emergency_reserve_releases, allocation_failures, recoveries, recovery_retries, circuit_state, circuit_transitions }` (prometheus-client `Family<LabelStruct, Gauge|Counter>`, label structs with `&'static str` / `u32 device` fields)
- `pub fn ReliabilityMetrics::register(reg: &MetricsRegistry) -> Self`; `pub fn ReliabilityMetrics::unregistered() -> Self`
- `pub struct SignalThresholds { pub levels: [Option<f64>; 4], pub lower_is_worse: bool }` with `level(f64) -> PressureState`, `threshold(PressureState) -> Option<f64>`, `below_exit(f64, PressureState, margin: f64) -> bool`, `exit_threshold(PressureState, f64) -> Option<f64>`
- `pub fn default_thresholds() -> BTreeMap<PressureSignal, SignalThresholds>`; `pub fn effective_thresholds(cfg: &PressureConfig) -> BTreeMap<PressureSignal, SignalThresholds>`; `pub fn active_signals(kind: MemoryKind) -> Vec<PressureSignal>`
- `pub struct SignalValue { pub signal: PressureSignal /* serde "name" */, pub value: f64, pub level: PressureState, pub stale: bool }`
- `pub struct SignalInputs<'a> { sample: &'a TelemetrySample, host_reserve_bytes: u64, devices: &'a [DeviceMemoryInput], exhaustion_horizon_seconds: f64, step_time_drift: Option<f64>, allocation_failure_recent: bool }`
- `pub struct DeviceMemoryInput { device: DeviceId, memory_kind: MemoryKind, budget_bytes: u64, idle_preallocated_bytes: u64 }` (amended 2026-09-27, Task 12b)
- `pub fn SignalEvaluator::new(thresholds, interval: Duration, stale_after: Duration) -> Self`; `pub fn evaluate(&mut self, inp: &SignalInputs<'_>, now: Duration) -> Vec<SignalValue>`
- `pub enum PoolKind { Weights, Kv, Workspace, Runtime, Reserve }` + `ALL`, `as_str()`
- `pub struct BudgetInputs { device, memory_kind, measured_free_bytes: Option<u64>, already_held_bytes: u64, host_mem_available_bytes: Option<u64>, weights_bytes, kv_bytes_per_token, max_seq_len: u32, block_bytes }`
- `pub struct DeviceBudget { device, memory_kind, budget_bytes, pools: Vec<(PoolKind, u64)> }` with `pool(PoolKind) -> u64`, `tracks_device_memory() -> bool`
- `pub fn compute_budget(inp: &BudgetInputs, cfg: &ReliabilityConfig, kv_cap: Option<ByteSize>) -> Result<DeviceBudget, BudgetError>`; `pub struct BudgetError { pub breakdown: String }`

Covers: S-1, S-2, S-6; `budget::tests::pools_partition_free_memory`, `budget::tests::unified_budget_from_mem_available`, `budget::tests::impossible_budget_rejected`, and the S-1 acceptance criterion (workspace build/test/clippy/fmt on macOS; `cargo tree -p turbine-reliability` without `nvml-wrapper`, `libloading` or GPU crates).
Depends on: Task 1; phase-0 plan (`MetricsRegistry`).

- [ ] Write failing test `crates/turbine-reliability/src/budget.rs` `budget::tests::pools_partition_free_memory`: dedicated device, measured free 30 GiB (and equivalently 24 GiB free + 6 GiB already held), weights 6 GiB, defaults → budget 30 GiB, `kv` pool 20 GiB, pools sum to the budget; `device_budget_bytes: 20GiB` → budget 20 GiB, `kv` 10 GiB; `kv_cap` 4 GiB → `kv` 4 GiB.
- [ ] Write failing test `budget::tests::unified_budget_from_mem_available`: unified device with measured free 100 GiB (ignored) and `MemAvailable` 42 GiB → budget 34 GiB; cap 24 GiB → 24 GiB; cap 64 GiB → still 34 GiB; `tracks_device_memory()` false and `active_signals(Unified)` lacks `DeviceMemory` while `active_signals(Dedicated)` has it.
- [ ] Write failing test `budget::tests::impossible_budget_rejected`: Llama-3.2-3B layout (114,688 B/token, 16-token blocks, `max_seq_len` 32768 → 3.5 GiB minimum KV) with 10 GiB and 13 GiB free fails with a breakdown naming `budget=`, `weights=`, `workspace=`, `runtime=`, `reserve=`, `kv=`, `measured_free=` and `32768-token`; 14 GiB succeeds.
- [ ] Run: `cargo test -p turbine-reliability budget::tests` — expect FAIL
- [ ] Implement `metrics.rs` (every P3 reliability family registered on the given `MetricsRegistry`), `signals.rs` (the P3 signal table verbatim; the evaluator omits signals of `Unavailable` sources, flags `Stale` ones, emits `telemetry_stale = 1` once any source stayed stale for `stale_after`, computes `swap_in_rate` from `pswpin` deltas and discards it when the tick gap exceeds 10 × interval, `thermal` = 2 on a thermal throttle reason, 1 within 5 °C of the slowdown temperature) and `compute_budget` (dedicated: measured free + already held; unified: `MemAvailable` − `host_reserve_bytes`; both `min` the cap; `kv` = remainder − fixed pools, then `min(kv_cap)`; error when reserve ≥ budget, fixed pools exceed the budget or `kv` < `ceil(max_seq_len × bytes_per_token / block_bytes) × block_bytes`).
- [ ] Run: `cargo test -p turbine-reliability budget::tests && cargo tree -p turbine-reliability -e normal | grep -E 'nvml|libloading|cudarc'; test $? -eq 1` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings (then `cargo build --workspace && cargo test --workspace` on macOS)
- [ ] Commit: `feat(reliability): crate skeleton, metric families, signal table and memory budget`

## Task 3: Reservation ledger and fault injector

Files: `crates/turbine-reliability/src/ledger.rs` (per-device pools, RAII reservations), `crates/turbine-reliability/src/fault.rs` (feature-gated injector), `crates/turbine-reliability/src/lib.rs` (modules)
Interfaces:

- `pub struct PoolUsage { pub capacity: u64, pub used: u64, pub reserved: u64 }` with `available()`, `utilization() -> f64`
- `pub enum LedgerError { Exhausted { pool: PoolKind, requested: u64, available: u64 }, Injected }`
- `pub fn Ledger::new(budget: &DeviceBudget) -> Arc<Ledger>`; `pub fn reserve(self: &Arc<Self>, device: DeviceId, pool: PoolKind, bytes: u64) -> Result<Reservation, LedgerError>`; `pub fn usage(&self, device: DeviceId, pool: PoolKind) -> PoolUsage`; `pub fn set_metrics(&self, m: ReliabilityMetrics)`; `#[cfg(feature = "fault-injection")] pub fn set_fault_injector(&self, f: Arc<FaultInjector>)`
- `pub struct Reservation` with `commit(&mut self)`, `commit_bytes(&mut self, bytes: u64)`, `bytes() -> u64`, `committed() -> u64`, `pool() -> PoolKind`; `impl Drop for Reservation`
- `pub fn FaultInjector::new(cfg: FaultInjectionConfig) -> Self`; `pub fn should_fail_alloc(&self) -> bool`; `pub fn iteration_fault(&self, iteration: u64) -> Option<InjectedIterationFault>`; `pub enum InjectedIterationFault { DeviceOom, KernelError { sticky: bool } }`

Covers: S-3, S-16 (injector); `ledger::tests::reservations_never_exceed_capacity`.
Depends on: Task 2.

- [ ] Write failing test `crates/turbine-reliability/src/ledger.rs` `ledger::tests::reservations_never_exceed_capacity`: 8 threads × 1,250 seeded (splitmix64) random reserve / partial commit / drop operations on a 1,000,000-byte `kv` pool assert `used + reserved ≤ capacity` after every operation, `(used, reserved) == (0, 0)` after all guards drop, and a 1,000,001-byte request returns `Exhausted { requested: 1000001, available: 1000000 }`.
- [ ] Run: `cargo test -p turbine-reliability ledger::tests::reservations_never_exceed_capacity` — expect FAIL
- [ ] Implement the ledger as a `Mutex<BTreeMap<(DeviceId, PoolKind), PoolUsage>>`; uncommitted bytes count as `reserved`, committed as `used`; `Drop` returns both; every change updates `turbine_memory_pool_bytes{device,pool,kind}` and every refusal increments `turbine_allocation_failures_total{device,pool}` and logs `allocation_failure` (WARN). Implement `FaultInjector` (1-based atomic counter: every `alloc_fail_every`-th reservation returns `LedgerError::Injected`) with a unit test `fault::tests::every_fifth_reservation_fails` (reservations 5, 10, 15, 20 of 20 fail, counter = 4).
- [ ] Run: `cargo test -p turbine-reliability ledger::tests && cargo test -p turbine-reliability --features fault-injection fault::tests` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings (plus `--all-features`)
- [ ] Commit: `feat(reliability): reservation ledger with RAII guards and allocation fault injector`

## Task 4: Pressure state machine with hysteresis

Files: `crates/turbine-reliability/src/state.rs` (machine, bounded history, test helpers), `crates/turbine-reliability/src/lib.rs` (module)
Interfaces:

- `pub const TRANSITION_HISTORY: usize = 32`
- `pub struct Transition { pub at: Duration, pub at_wall: SystemTime, pub from: PressureState, pub to: PressureState, pub signal: PressureSignal, pub value: f64, pub threshold: f64 }`
- `pub struct Gates { pub floor: PressureState, pub floor_signal: PressureSignal, pub reserve_held: bool }` (Default: GREEN, `TelemetryStale`, true)
- `pub struct MachineConfig { enabled, escalate_samples: u32, deescalate_dwell: Duration, exit_margin: f64 }`; `pub fn MachineConfig::from_config(cfg: &ReliabilityConfig) -> Self`
- `pub fn PressureMachine::new(cfg: MachineConfig, thresholds: BTreeMap<PressureSignal, SignalThresholds>, metrics: ReliabilityMetrics, clock: Arc<dyn Clock>) -> Self`
- `pub fn evaluate(&mut self, signals: &[SignalValue], gates: Gates) -> Option<Transition>`; `state()`, `since_wall()`, `history() -> impl DoubleEndedIterator<Item = &Transition> + ExactSizeIterator`

Covers: S-6, S-7, S-14 (transition metric + log); `state::tests::hysteresis_holds`, `state::tests::escalation_and_stepwise_deescalation`, `state::tests::transitions_explained`.
Depends on: Tasks 2–3.

- [ ] Write failing test `state::tests::hysteresis_holds`: on a `FakeClock` with 100 ms ticks, `kv_utilization` alternating 0.83/0.81 every 500 ms for 60 s produces exactly one transition (GREEN→ORANGE); then 0.77 produces ORANGE→YELLOW exactly 10 s after it started, and 0.78 (inside the 5 % exit margin, threshold 0.779) never de-escalates.
- [ ] Write failing test `state::tests::escalation_and_stepwise_deescalation`: one `allocation_failure = 1` sample moves GREEN→SURVIVAL; all-clear signals then give SURVIVAL→RED→ORANGE→YELLOW→GREEN at 10 s, 20 s, 30 s, 40 s (+ one tick); with `escalate_samples: 3` two RED samples, a GREEN sample, and three RED samples transition only on the last.
- [ ] Write failing test `state::tests::transitions_explained`: with `escalate_samples: 1` and dwell 100 ms, 400 ticks alternating 0.91/0.10 every second produce > 32 transitions; history holds exactly 32; the `turbine_pressure_transitions_total{from,to,signal="kv_utilization"}` counters sum to the transition count; one `pressure_transition` event per transition carries `from`, `to`, `signal`, `value`, `threshold` equal to the history entries (events captured by a process-wide capture layer installed once, after `tracing::callsite::rebuild_interest_cache()`, because scoped subscribers race with callsite-interest caching in parallel tests).
- [ ] Run: `cargo test -p turbine-reliability state::tests` — expect FAIL
- [ ] Implement `PressureMachine`: target = max level of non-stale signals and `gates.floor`; `AllocationFailure` at SURVIVAL escalates immediately; otherwise escalate to the minimum level seen across `escalate_samples` consecutive samples above the current state; de-escalate one level when every live signal satisfies `below_exit(value, state, exit_margin)` continuously for `deescalate_dwell` (dwell restarts at each step), never below `gates.floor`, and never from RED while `!gates.reserve_held`; each transition pushes history (cap 32), increments the counter, sets `turbine_pressure_state{state}` and logs `pressure_transition` (INFO); `enabled: false` keeps GREEN.
- [ ] Run: `cargo test -p turbine-reliability state::tests` (10 consecutive runs) — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(reliability): hysteretic pressure state machine with explained transitions`

## Task 5: Emergency reserve

Files: `crates/turbine-reliability/src/reserve.rs` (reserve outside normal scheduling), `crates/turbine-reliability/src/lib.rs` (module)
Interfaces:

- `pub trait ReserveAllocator: Send { fn allocate(&mut self, bytes: u64) -> Result<(), String>; fn free(&mut self); }`
- `pub enum ReserveError { NotSurvival(PressureState), NotHeld, Allocation { bytes: u64, detail: String } }`
- `pub fn EmergencyReserve::acquire(device: DeviceId, bytes: u64, ledger: &Arc<Ledger>, allocator: Box<dyn ReserveAllocator>, metrics: ReliabilityMetrics) -> Result<Self, ReserveError>`
- `pub fn release_for_recovery(&mut self, state: PressureState) -> Result<u64, ReserveError>`; `pub fn try_reacquire(&mut self) -> bool`; `held()`, `bytes()`, `releases()`

Covers: S-4; `reserve::tests::reserve_only_released_in_survival`.
Depends on: Tasks 3–4.

- [ ] Write failing test `reserve::tests::reserve_only_released_in_survival`: a 2 GiB reserve fully uses the `reserve` pool (a 1-byte reservation there fails); `release_for_recovery` in GREEN, YELLOW, ORANGE and RED returns `NotSurvival` and keeps it held; after an `allocation_failure` sample drives the machine to SURVIVAL it releases 2 GiB (`releases() == 1`); with the allocator refusing ("memory taken by another process") the machine stays at RED for 60 s of all-clear ticks; after a successful `try_reacquire` it steps RED→ORANGE one dwell later.
- [ ] Run: `cargo test -p turbine-reliability reserve::tests::reserve_only_released_in_survival` — expect FAIL
- [ ] Implement `EmergencyReserve`: the bytes are a committed `Reserve`-pool reservation plus the real allocation from `ReserveAllocator`; `bytes == 0` counts as held (disabled); release/re-acquire update `turbine_emergency_reserve_held{device}`, `turbine_emergency_reserve_releases_total{device}` and log `emergency_reserve` (WARN) with reason `released` / `acquired` / `reacquire_failed`.
- [ ] Run: `cargo test -p turbine-reliability reserve::tests` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(reliability): emergency reserve released only in SURVIVAL`

## Task 6: KV exhaustion horizon

Files: `crates/turbine-reliability/src/horizon.rs` (predictor), `crates/turbine-reliability/src/lib.rs` (module)
Interfaces:

- `pub fn ExhaustionHorizon::predict(running: &[u32], decode_tokens_per_s: f64, free_blocks: u32, block_tokens: u32) -> f64`

Covers: S-8; `horizon::tests::predicts_exhaustion`.
Depends on: Task 2.

- [ ] Write failing test `horizon::tests::predicts_exhaustion`: 20 sequences × 500 remaining tokens at 50 tokens/s against 200 free 16-token blocks → 3.2 s ± 10 %; no running sequences → `+∞`; 20 × 100 remaining (fits) → `+∞`; 10 × 10 + 10 × 1000 remaining → 6.2 s.
- [ ] Run: `cargo test -p turbine-reliability horizon::tests::predicts_exhaustion` — expect FAIL
- [ ] Implement the piecewise-linear solve of Σ min(rᵢ, rate·t) = free tokens over sorted remaining lengths (sequences stop growing at their `max_tokens`); `+∞` when the total committed growth fits or nothing grows.
- [ ] Run: `cargo test -p turbine-reliability horizon::tests` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(reliability): KV exhaustion horizon from committed growth`

## Task 7: Predictive admission and the bounded admission queue

Files: `crates/turbine-reliability/src/admission.rs` (estimate, decision table, KV reservation, EWMAs, queue), `crates/turbine-reliability/src/lib.rs` (module)
Interfaces:

- `pub enum PressureReason { KvReservation, PressureOrange, PressureRed, CircuitDegraded, PrefillBudget }` + `ALL`, `as_str()`
- `pub enum RejectionReason { ContextExceedsKvCapacity, QueueFull, QueueTimeout, Survival, CircuitOpen }` + `ALL`, `as_str()`, `http(&self) -> (u16, &'static str, &'static str)`
- `pub enum AdmissionDecision { Admit, Queue { reason: PressureReason }, Reject { reason: RejectionReason } }`; `labels(&self) -> (&'static str, &'static str)`
- `pub struct Calibration { prefill_tokens_per_s: f64, decode_step_s: f64 }`; `pub struct AdmissionParams { device, adaptive, max_queue, large_prefill_tokens, block_bytes, prefill_chunk_tokens, workspace_bytes_per_token, calibration }`
- `pub fn Admission::new(params: AdmissionParams, ledger: Arc<Ledger>, metrics: ReliabilityMetrics) -> Self`
- `pub fn estimate(&self, prompt_tokens: u32, cached_prefix_tokens: u32, max_tokens: Option<u32>, layout: &KvLayout, max_seq_len: u32) -> ResourceEstimate`
- `pub fn decide(&mut self, est: &ResourceEstimate, state: PressureState, circuit: CircuitState, queue_len: usize) -> AdmissionDecision`; `pub fn evaluate(&self, …same…) -> AdmissionDecision` (records nothing)
- `pub fn evaluate_refill(&self, est: &ResourceEstimate, state: PressureState, circuit: CircuitState) -> AdmissionDecision` — the decision for a queued request offered an admitted slot (gate pump): RED is judged by ORANGE's rules (`pressure_red` binds new arrivals only), never `queue_full`, records nothing (amendment 2026-09-26: RED refills finished slots)
- `pub fn reserve_kv(&self, est: &ResourceEstimate) -> Result<Reservation, LedgerError>`; `expensive_prefill_started(&mut self)`, `expensive_prefill_finished(&mut self)`; `observe_iteration(&mut self, prefill_tokens: u32, prefill_seconds: f64, decode_step_seconds: Option<f64>)`
- `pub struct Queued<T> { id, priority, estimate, reason, enqueued_at, bypassed, payload: T }`
- `pub fn AdmissionQueue::<T>::new(max_queue: u32, queue_timeout: Duration, max_bypass: u32) -> Self`; `push(&mut self, id, priority, estimate, reason, now, payload: T) -> Result<(), T>`; `remove(&mut self, id) -> Option<Queued<T>>`; `expire(&mut self, now) -> Vec<Queued<T>>`; `drain_all(&mut self) -> Vec<Queued<T>>`; `pump(&mut self, try_admit: impl FnMut(&Queued<T>) -> bool) -> Vec<Queued<T>>`; `estimated_drain_seconds(&self, max_running: u32) -> u64`; `len`, `is_empty`, `iter`

Covers: S-9; `admission::tests::decision_table`, `admission::tests::bypass_bounded`, `admission::tests::red_refill_from_queue`.
Depends on: Tasks 2–3; phase-2 plan (`ResourceEstimate`, `KvLayout`, `Priority`, `RequestId`).

- [ ] Write failing test `admission::tests::decision_table`: with a 1,024-block Llama `kv` pool: prompt 4000 + `max_tokens` 13000 → `Reject(ContextExceedsKvCapacity)`; cheap (100 + 100) in GREEN → `Admit`; expensive prefill (4000 tokens) in ORANGE → `Queue(PressureOrange)` while cheap is admitted; any request in RED → `Queue(PressureRed)`, at `queue_len == max_queue` (4) → `Reject(QueueFull)`; SURVIVAL → `Reject(Survival)`; CIRCUIT_OPEN → `Reject(CircuitOpen)`; DEGRADED + expensive → `Queue(CircuitDegraded)`; YELLOW + expensive while one expensive prefill runs → `Queue(PrefillBudget)`; with 1,020 blocks held elsewhere → `Queue(KvReservation)`; `adaptive: false` admits in RED and SURVIVAL but still rejects the oversize request; no `max_tokens` with `max_seq_len` 8192 reserves 512 blocks; `max_tokens: 100` with a 100-token prompt reserves ceil(200/16) = 13 blocks; `http()` of the five reject reasons equals the P3 reject table.
- [ ] Write failing test `admission::tests::bypass_bounded`: a head needing 1,000 blocks and 20 one-block requests with 10 free blocks → the first `pump` admits exactly requests 1–8 (`max_bypass` 8) in FIFO order, the second admits nothing, the head shows `bypassed == 8`, and once everything fits the head goes first; priority `-1` sorts before `0`, FIFO within a priority; `max_queue` bounds `push`; entries older than `queue_timeout` are returned by `expire`.
- [ ] Write failing test `admission::tests::red_refill_from_queue` (amendment 2026-09-26): with the 1,024-block pool, cheap (100 + 100) in RED → `decide`/`evaluate` `Queue(PressureRed)` but `evaluate_refill` `Admit`; expensive (4000) in RED and ORANGE → `evaluate_refill` `Queue(PressureOrange)`; cheap in ORANGE and expensive in GREEN → `Admit`; SURVIVAL → `Reject(Survival)`; CIRCUIT_OPEN → `Reject(CircuitOpen)`; with 1,020 blocks held elsewhere → `Queue(KvReservation)`; `turbine_admission_decisions_total{decision="admit"}` unchanged by `evaluate_refill`.
- [ ] Run: `cargo test -p turbine-reliability admission::tests` — expect FAIL
- [ ] Implement admission: `projected_kv_blocks = ceil((prompt + max_output)/block_tokens) − cached_full_blocks` with `max_output = max_tokens` else remaining context (worst case, no overcommit); decision order hard capacity → circuit → (adaptive) SURVIVAL/RED/ORANGE-expensive/DEGRADED-expensive/YELLOW-prefill-budget → unreserved KV → `queue_full` at `max_queue`; each decision increments `turbine_admission_decisions_total{decision,reason}` (`reason="none"` for admit) and logs `admission_decision` (DEBUG for admit, INFO otherwise); EWMAs (α = 0.1) of prefill tokens/s and decode step time are seeded by the mean of the first 32 iterations and use `Calibration` before that (`estimated: false`); the queue is sorted by `(priority, arrival seq)` and `pump` counts bypasses only against the current head.
- [ ] Run: `cargo test -p turbine-reliability admission::tests` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(reliability): predictive admission with worst-case KV reservation and bounded queue`

## Task 8: Throttle planner and the KvReclaimer hook

Files: `crates/turbine-reliability/src/throttle.rs` (plan per state, reclaim, plan gauges), `crates/turbine-reliability/src/lib.rs` (module)
Interfaces:

- `pub enum AdmissionMode { Open, ExpensiveQueued, AllQueued, Stopped }` (serde snake_case)
- `pub enum ReclaimAction { None, DemoteIdle, FreeCachedToOrange, FreeAllCachedAndOptional, ReleaseReserveAndPreemptIfNeeded }`
- `pub struct ThrottlePlan { state, batch_growth_limit: Option<u32>, shrink_only: bool, prefill_budget_fraction: f64, prefill_chunk_tokens: Option<u32>, start_new_prefills: bool, admission: AdmissionMode, reclaim: ReclaimAction }`
- `pub struct SchedulerLimits { prefill_chunk_tokens: u32, block_tokens: u32 }`; `pub fn plan_for(state: PressureState, cfg: &SchedulerLimits) -> ThrottlePlan`
- `pub trait KvReclaimer: Send + Sync { fn demote(&self, target_utilization: f64) -> u64; fn free_unreferenced(&self, target_utilization: f64) -> u64; fn free_optional(&self) -> u64 { 0 } }`
- `pub fn apply_reclaim(plan: &ThrottlePlan, reclaimer: &dyn KvReclaimer, kv: &SignalThresholds, metrics: &ReliabilityMetrics) -> Vec<(&'static str, u64)>`; `pub fn publish_plan(plan: &ThrottlePlan, metrics: &ReliabilityMetrics)`

Covers: S-10; `throttle::tests::plan_per_state`.
Depends on: Task 2.

- [ ] Write failing test `throttle::tests::plan_per_state`: with chunk 2048 and 16-token blocks: GREEN (unlimited, 1.0, 2048, open, none); YELLOW (+1, 1.0, 2048, open, demote); ORANGE (0, 0.5, 1024, expensive_queued, free cached to ORANGE); RED (0, not shrink-only — finished slots are refilled, 0.5, 64, new prefills start in refilled slots, all_queued, free all cached + optional); SURVIVAL (0, shrink-only, 0.0, no chunk, stopped, release reserve); chunk 96 halves to the 64 floor; a recording `KvReclaimer` sees `demote(0.70)` in YELLOW, `free_unreferenced(0.82)` then `demote(0.82)` in ORANGE, `free_unreferenced(0.0)` + `free_optional()` in RED and nothing in GREEN/SURVIVAL; freed bytes land in `turbine_reclaim_bytes_total{action="free_cached"}`.
- [ ] Run: `cargo test -p turbine-reliability throttle::tests::plan_per_state` — expect FAIL
- [ ] Implement `plan_for` from the P3 table (chunk floor 4 × block tokens), `apply_reclaim` with the `kv_utilization` YELLOW/ORANGE thresholds as targets (metric + `reclaim` INFO log per non-zero action), and `publish_plan` (`turbine_throttle_plan{field}`; unlimited growth published as −1).
- [ ] Run: `cargo test -p turbine-reliability throttle::tests` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(reliability): throttle plan per pressure state and reclaim hook`

## Task 9: Circuit breaker

Files: `crates/turbine-reliability/src/circuit.rs` (transition table), `crates/turbine-reliability/src/lib.rs` (module)
Interfaces:

- `pub enum CircuitReason { LatencyDrift, ThermalThrottle, OomRecovered, TelemetryStale, RepeatedOom, RecoveryFailed, DeviceError, ProbeFailed, DeviceFatal, ControllerFailed, NoTrigger, DrainStarted, DrainComplete, ProbesSucceeded }` + `as_str()`
- `pub enum CircuitEvent { Iteration, LatencyDrift { ratio: f64 }, ThermalThrottle, TelemetryStale, OomRecovered, RecoveryFailed, DeviceError { sticky: bool }, ControllerFailed, ProbeSucceeded { latency_ratio: f64 }, ProbeFailed, Tick { running: u32 } }`
- `pub type CircuitTransition = (CircuitState, CircuitState, CircuitReason)`
- `pub fn CircuitBreaker::new(cfg: &CircuitConfig, metrics: ReliabilityMetrics, now: Duration) -> Self`; `pub fn on_event(&mut self, ev: CircuitEvent, now: Duration) -> Option<CircuitTransition>`; `state()`, `since()`, `last_reason()`, `is_fatal()`, `retry_after_secs() -> u64`, `drain_expired() -> bool`, `baseline_reset_due(t: &CircuitTransition) -> bool`

Covers: S-12 (state machine); `circuit::tests::transition_table`.
Depends on: Tasks 1–2.

- [ ] Write failing test `circuit::tests::transition_table`: with defaults and explicit `now`: drift 5.0 and thermal are ignored with no `Iteration` in the window; drift 2.0 → DEGRADED, back to HEALTHY on the tick 60 s after the last trigger (`no_trigger`); thermal / one OOM recovery / telemetry stale → DEGRADED; 3 OOM recoveries within 60 s → CIRCUIT_OPEN (`repeated_oom`), but not when spread over > 60 s; drift ≥ 4.0, `RecoveryFailed`, non-sticky device error → CIRCUIT_OPEN; OPEN → DRAINING on the next tick with `retry_after_secs() == 30`; DRAINING → PROBING only when running == 0 and 30 s cooldown elapsed; `drain_expired()` at 120 s; three probes ≤ 2 × baseline → HEALTHY (baseline reset due); a failed or 2.5×-slow probe → CIRCUIT_OPEN (`probe_failed`); sticky device error → CIRCUIT_OPEN (`device_fatal`), fatal, stays DRAINING; controller failure → fatal.
- [ ] Run: `cargo test -p turbine-reliability circuit::tests::transition_table` — expect FAIL
- [ ] Implement the breaker from the P3 transition table (sliding OOM window as a `VecDeque<Duration>`; OPEN→DRAINING, DRAINING→PROBING and PROBING→HEALTHY use the added reasons `drain_started`, `drain_complete`, `probes_succeeded`); each transition increments `turbine_circuit_transitions_total{from,to,reason}`, sets `turbine_circuit_state{state}` and logs `circuit_transition` (WARN).
- [ ] Run: `cargo test -p turbine-reliability circuit::tests` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(reliability): circuit breaker transition table`

## Task 10: Recovery controller, pressure document and PressureController

Files: `crates/turbine-reliability/src/recovery.rs` (bounded OOM retries), `crates/turbine-reliability/src/document.rs` (`GET /turbine/v1/pressure` body), `crates/turbine-reliability/src/controller.rs` (tick loop, snapshot, handle), `crates/turbine-reliability/src/lib.rs` (modules)
Interfaces:

- `pub enum RecoveryOutcome { Recovered { retries: u32 }, Failed }` + `as_str()`; `pub enum RecoveryStep { Retry { attempt: u32, backoff: Duration, batch_limit: usize }, GiveUp }`
- `pub fn RecoveryController::new(cfg: &RecoveryConfig, metrics: ReliabilityMetrics) -> Self`; `on_oom(&mut self, batch_len: usize) -> RecoveryStep`; `on_success(&mut self) -> Option<RecoveryOutcome>`; `in_recovery() -> bool`
- `pub struct PressureDocument { enabled, state, since: String, dominant_signal: Option<PressureSignal>, exhaustion_horizon_seconds: Option<f64>, signals: Vec<SignalValue>, throttle: ThrottleDoc, memory: Vec<MemoryDoc>, admission: AdmissionDoc, circuit: CircuitDoc, transitions: Vec<TransitionDoc> }`; `pub fn rfc3339_millis(t: SystemTime) -> String`; `pub fn memory_doc(&DeviceBudget, &Ledger, reserve_held: bool) -> MemoryDoc`; `pub fn decisions_doc(&ReliabilityMetrics) -> DecisionsDoc`
- `pub struct EngineStats { running_remaining_tokens: Vec<u32>, free_kv_blocks: u32, block_tokens: u32, decode_tokens_per_s: f64, step_time_p95_s: Option<f64>, queue_len: u32, iterations: u64 }`
- `pub struct Snapshot { state, circuit, throttle: ThrottlePlan, circuit_retry_after_secs: u64, fatal: bool, drain_expired: bool, document: PressureDocument }`
- `#[derive(Clone)] pub struct ControllerHandle` with `snapshot() -> Arc<Snapshot>`, `throttle()`, `state()`, `circuit()`, `document()` (backed by `arc_swap::ArcSwap<Snapshot>`)
- `pub fn PressureController::new(cfg: &ReliabilityConfig, limits: SchedulerLimits, budget: DeviceBudget, ledger: Arc<Ledger>, reserve: EmergencyReserve, reclaimer: Arc<dyn KvReclaimer>, metrics: ReliabilityMetrics, clock: Arc<dyn Clock>) -> (Self, ControllerHandle)`
- `pub fn tick(&mut self, sample: &TelemetrySample, stats: &EngineStats) -> Option<Transition>`; `pub fn on_oom(&mut self) -> u64`; `pub fn on_recovery(&mut self, outcome: RecoveryOutcome)`; `pub fn on_circuit_event(&mut self, ev: CircuitEvent) -> Option<CircuitTransition>`; `pub struct NoReclaim` (`KvReclaimer` returning 0)

Covers: S-11 (retry/backoff logic), S-13 (document model), S-14 (signal, horizon, plan, reserve, reclaim gauges); unit tests `recovery::tests::retries_are_bounded_and_back_off`, `document::tests::rfc3339_formats_utc_millis`, `controller::tests::simulated_overload_cycle`, `controller::tests::disabled_stays_green` (the S-11/S-13 acceptance tests are delivered by Tasks 12–14).
Depends on: Tasks 2–9.

- [ ] Write failing test `recovery::tests::retries_are_bounded_and_back_off`: defaults give `Retry { attempt: 1, backoff: 50ms, batch_limit: 4 }` for a batch of 8, then `(2, 100ms, 2)`, `on_success() == Some(Recovered { retries: 2 })`; a fourth OOM after three retries returns `GiveUp`, `turbine_recovery_retries_total == 5`, `turbine_recoveries_total{outcome="failed"} == 1`; `max_retries: 0` gives up at once.
- [ ] Write failing test `document::tests::rfc3339_formats_utc_millis`: 1,790,359,331,482 ms after the epoch → `2026-09-25T18:02:11.482Z`; epoch → `1970-01-01T00:00:00.000Z`; 951,782,400 s → `2000-02-29T00:00:00.000Z`.
- [ ] Write failing test `controller::tests::simulated_overload_cycle`: deterministic simulated telemetry (dedicated device at 40 °C of a 90 °C slowdown, 64 GiB MemAvailable, no PSI/swap) on a `FakeClock`: `kv_utilization` 0.5 for 2 s → GREEN; 0.85 for 1 s → ORANGE with chunk 1024 and dominant `kv_utilization`; `on_oom()` returns 2 GiB, state SURVIVAL, document shows `emergency_reserve_held: false` and reserve pool used 0; `on_recovery(Recovered)` → circuit DEGRADED; 45 s of 0.1 → YELLOW (DEGRADED floor) with the reserve re-acquired; 40 s more → HEALTHY and GREEN; the serialised document has every key of the P3 §Data example, `state: "GREEN"` and ≥ 6 transitions.
- [ ] Write failing test `controller::tests::disabled_stays_green`: `enabled: false` with `kv_utilization` 0.99 for 5 s keeps GREEN and serialises `"enabled": false`.
- [ ] Run: `cargo test -p turbine-reliability -- recovery::tests document::tests controller::tests` — expect FAIL
- [ ] Implement the recovery controller (backoff = `backoff × 2^(attempt−1)`, batch halved with floor 1, `recovery_attempt` / `recovery_outcome` WARN logs), the document (serde structs with P3 key names; +∞ horizon → `null`; civil-from-days RFC 3339 formatting, no date-time dependency; admission decision counts read back from the counters) and `PressureController::tick` (horizon → drift against a baseline learned only in GREEN + HEALTHY → signals → circuit events (drift, thermal ≥ 2, telemetry stale, tick) → reserve re-acquisition when at or below RED → machine with DEGRADED floor YELLOW and reserve gate → plan + reclaim → publish signal/horizon gauges and a new `Snapshot` via `ArcSwap::store`); `on_oom` evaluates an immediate allocation-failure sample and releases the reserve (`turbine_reclaim_bytes_total{action="release_reserve"}`).
- [ ] Run: `cargo test -p turbine-reliability && cargo test -p turbine-reliability --features fault-injection` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings (plus `--all-features`)
- [ ] Commit: `feat(reliability): recovery controller, pressure document and pressure controller`

## Task 11: Two-cadence telemetry sampler in turbine-device

Files: `crates/turbine-device/src/telemetry/mod.rs` (config, `VendorTelemetry`, `LatestSample`, `SamplerCore`, `TelemetrySampler`, `TelemetryMetrics`, tests), `crates/turbine-device/src/telemetry/proc.rs` (`/proc` parsers + `FsProc`), `crates/turbine-device/src/telemetry/vendor.rs` (NVML and amd-smi backends), `crates/turbine-device/src/lib.rs` (module), `crates/turbine-device/Cargo.toml` (`arc-swap`, `prometheus-client`), `crates/turbine-device/tests/fixtures/proc/{meminfo,vmstat_a,vmstat_b,pressure_memory}` (fixtures: dgx-spark `/proc/meminfo` captured 2026-09-25 trimmed at `SwapFree`, `MemAvailable: 44950556 kB`, `SwapTotal`/`SwapFree: 16777212 kB`; `pswpin` 1200 then 1450; PSI `some avg10=12.50 …` / `full avg10=4.02 …`)
Interfaces:

- `pub struct TelemetryConfig { interval, vendor_interval, call_timeout, stale_after: Duration }`; `pub fn TelemetryConfig::from_config(cfg: &ReliabilityTelemetryConfig) -> Self`
- `pub trait VendorTelemetry: Send { fn vendor(&self) -> Vendor; fn sample(&mut self, device: &DeviceInfo) -> Result<DeviceSample, String>; }`
- `#[derive(Clone)] pub struct LatestSample` with `new()`, `load(&self) -> Arc<TelemetrySample>` (`ArcSwap`)
- `pub fn read_host(proc: &dyn ProcSource) -> HostRead`; `pub struct HostRead { sample: HostSample, vmstat: SourceStatus, psi: SourceStatus }`
- `pub fn SamplerCore::new(cfg: TelemetryConfig, inventory: &DeviceInventory, vendor: Vec<Box<dyn VendorTelemetry>>, proc: Box<dyn ProcSource>, ledger: Arc<dyn LedgerProbe>, clock: Arc<dyn Clock>) -> Self`; `with_metrics(self, TelemetryMetrics) -> Self`; `latest()`, `fast_ticks()`, `vendor_ticks()`, `poll(&mut self) -> Duration`
- `pub fn TelemetrySampler::spawn(cfg, inventory, vendor, proc, ledger, clock) -> (TelemetrySampler, LatestSample)`; `pub fn spawn_core(core: SamplerCore) -> (TelemetrySampler, LatestSample)` (stops and joins on `Drop`)
- `pub fn TelemetryMetrics::register(reg: &MetricsRegistry) -> Self`; `pub fn record_device(&self, d: &DeviceSample, kind: MemoryKind)`
- `proc::{ProcFile::{Meminfo, Vmstat, PressureMemory}, trait ProcSource { fn read(&self, file: ProcFile) -> std::io::Result<String>; }, FsProc { root: PathBuf }, parse_meminfo(&str) -> Result<MemInfo, ParseError>, parse_vmstat(&str) -> Result<VmStat, ParseError>, parse_psi(&str) -> Result<PsiMemory, ParseError>}`
- `vendor::{NvmlTelemetry::open(path: Option<&Path>) -> Result<Self, String>, AmdSmiTelemetry::open(path: Option<&Path>) -> Result<Self, String>, vendor_backends(opts: &DiscoveryOptions) -> Vec<Box<dyn VendorTelemetry>>}` (Phase 2m port: one per registered `DiscoveryKind`, whose `telemetry(&DiscoveryOptions)` opens `nvml::NvmlTelemetry` / `amd_smi::AmdSmiTelemetry`)

Covers: S-5, S-14 (GPU/host/telemetry gauges); `telemetry::tests::proc_parsers`, `telemetry::tests::two_cadences`, `telemetry::tests::hung_call_marks_stale`.
Depends on: Task 1; phase-0 plan (`DeviceInfo`, `DeviceInventory`, `nvml-wrapper` 0.13, `libloading` 0.9, `DevicesConfig.{nvml_library, amd_smi_library}`).

- [ ] Write failing test `telemetry::tests::proc_parsers`: the fixture meminfo gives `MemAvailable` 44,950,556 × 1024 bytes and swap totals ×1024 (kB scaling); `MemTotal`-only input → `ParseError::Missing { field: "MemAvailable" }`; `MemAvailable: lots kB` → `Malformed`; `pswpin` delta between the two vmstat fixtures is 250; PSI `some_avg10 == 12.5`, `full_avg10 == Some(4.02)`; a source without `pressure/memory` yields PSI `Unavailable` with host `Ok`; `FsProc { root: "/nonexistent-proc-root" }` yields host `Unavailable` without panicking.
- [ ] Write failing test `telemetry::tests::two_cadences`: `FakeClock` advanced 10 ms per `poll` for 10 s with defaults, 2 devices and a counting vendor backend → 100 ± 1 fast ticks, 10 ± 1 vendor calls per device, `/proc` reads = 3 × fast ticks, and a `LedgerProbe` change (0.10 → 0.75) visible in `latest().ledger.kv_utilization` within 100 ms.
- [ ] Write failing test `telemetry::tests::hung_call_marks_stale`: a vendor backend sleeping 5 s with `call_timeout: 500ms` → the first `poll` returns within 600 ms with the device `Stale`; over the next 1.5 s every `poll` returns within 600 ms, fast ticks advance by ≥ 10, and the host sample keeps updating.
- [ ] Run: `cargo test -p turbine-device telemetry::tests` — expect FAIL
- [ ] Implement the parsers (meminfo kB → bytes), `read_host`, `SamplerCore` (fast tick: `/proc` + `LedgerProbe` → new `TelemetrySample` stored in `LatestSample`; vendor tick: one worker thread per vendor library fed by a `sync_channel(1)`; the sampler waits `recv_timeout(call_timeout)`, a missed deadline marks the device `Stale` and marks the worker busy so later ticks skip it via `try_recv` instead of blocking; an error before the first success is `Unavailable` (WARN once), after it `Stale`; unified devices drop memory fields; missed ticks after a stall of > 10 intervals are skipped), `TelemetrySampler` (thread calling `poll`, sleeping the returned wait clamped to 1–50 ms), and the metrics (`turbine_gpu_*`, `turbine_host_*`, `turbine_telemetry_stale{source}`, `turbine_telemetry_call_duration_seconds{source}`).
- [ ] Implement the vendor backends: NVML through `nvml_wrapper::Nvml::{init, builder().lib_path(..).init()}` and `Device::{temperature(TemperatureSensor::Gpu), temperature_threshold(TemperatureThreshold::Slowdown), clock_info(Clock::SM), power_usage() /* mW */, utilization_rates(), memory_info() /* NotSupported on GB10 */, current_throttle_reasons()}` mapping `SW_/HW_THERMAL_SLOWDOWN` → thermal, `SW_POWER_CAP`/`HW_POWER_BRAKE_SLOWDOWN` → power, `HW_SLOWDOWN` → other; amd-smi through `libloading` resolving `amdsmi_init(uint64_t)` (flag `AMDSMI_INIT_AMD_GPUS = 1 << 1`), `amdsmi_get_socket_handles`, `amdsmi_get_processor_handles` (vendor_index = enumeration order), `amdsmi_get_temp_metric(h, AMDSMI_TEMPERATURE_TYPE_HOTSPOT = 1, AMDSMI_TEMP_CURRENT = 0 | AMDSMI_TEMP_CRITICAL = 5, int64_t*)` (Celsius), `amdsmi_get_clock_info(h, AMDSMI_CLK_TYPE_GFX = 0, amdsmi_clk_info_t*)`, `amdsmi_get_power_info(h, amdsmi_power_info_t*)` (`current_socket_power`, else `average_socket_power`, W), `amdsmi_get_gpu_activity` (`gfx_activity` %), `amdsmi_get_gpu_memory_usage/_total(h, AMDSMI_MEM_TYPE_VRAM = 0, uint64_t*)`, optional `amdsmi_get_violation_status` into an 8 KiB zeroed buffer (struct is 6,016 bytes with `AMDSMI_MAX_NUM_XCP = AMDSMI_MAX_NUM_XCC = 8`) reading `active_prochot_thrm`/`active_socket_thrm` (bytes 120/122 → thermal) and `active_ppt_pwr` (121 → power), `amdsmi_shut_down` on `Drop` — layouts checked read-only against novanas `/opt/rocm/rocm/include/amd_smi/amdsmi.h` (lines 3158, 3215, 3280, 4302, 4331, 7623, 7641, 7661, 7699, 7737); `unsafe impl Send for AmdSmiTelemetry` and every call carry `// SAFETY:`.
- [ ] Run: `cargo test -p turbine-device telemetry::tests && cargo test -p turbine-kernels --test unsafe_isolation` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(device): two-cadence telemetry sampler with /proc parsers and NVML/amd-smi backends`

## Task 12: Ledger-backed KV blocks, admission gate, throttle overlay and overload simulation

Files: `crates/turbine-kv/Cargo.toml` (dep `turbine-reliability`), `crates/turbine-kv/src/pool.rs` (`with_ledger`, `allocate_reserved`, `PoolError::ReservationShort`), `crates/turbine-kv/src/reclaim.rs` (new: L0 `KvReclaimer`), `crates/turbine-kv/src/lib.rs` (module), `crates/turbine-scheduler/Cargo.toml` (dep `turbine-reliability`), `crates/turbine-scheduler/src/gate.rs` (new: admission gate owning the single waiting queue), `crates/turbine-scheduler/src/scheduler.rs` (gate, per-request reservation, throttle overlay, SURVIVAL-only preemption, snapshot helpers), `crates/turbine-scheduler/src/lib.rs` (`SubmitError::Rejected`, `IterationLimits.allow_preempt`, `From<&ThrottlePlan>`), `crates/turbine-scheduler/src/sim/overload.rs` (new: overload harness), `crates/turbine-scheduler/src/sim/mod.rs` (module), `crates/turbine-scheduler/tests/overload_sim.rs` (new: acceptance tests)
Interfaces:

- `pub fn BlockPool::with_ledger(self, ledger: Arc<Ledger>, device: DeviceId) -> Self`; `pub fn allocate_reserved(&mut self, n: u32, reservation: &mut Reservation) -> Result<SmallVec<[BlockId; 8]>, PoolError>`; `PoolError::ReservationShort { needed: u64, uncommitted: u64 }`
- `pub struct turbine_kv::reclaim::L0Reclaimer` (`KvReclaimer`; returns 0 — the P2 pool keeps no unreferenced cached blocks; phase-4 replaces it)
- `pub enum GateOutcome { Admitted(SchedRequest, Reservation), Queued }`
- `pub fn AdmissionGate::new(admission: Admission, queue: AdmissionQueue<SchedRequest>, controller: ControllerHandle, clock: Arc<dyn Clock>, max_running: u32) -> Self`; `offer(&mut self, r: SchedRequest) -> Result<GateOutcome, SubmitError>`; `expire(&mut self) -> Vec<SchedRequest>`; `pump(&mut self, max_new: usize) -> Vec<(SchedRequest, Reservation)>`; `remove(&mut self, id: RequestId) -> Option<SchedRequest>`; `reject_all(&mut self) -> Vec<SchedRequest>`; `retry_after_secs(&self, reason: RejectionReason) -> u64`; `queue_len()`; `admission_mut()`
- `SubmitError::Rejected { reason: RejectionReason, retry_after_secs: u64 }`; `IterationLimits { …, allow_preempt: bool }`; `impl From<&ThrottlePlan> for IterationLimits`; `pub fn Scheduler::with_gate(self, gate: AdmissionGate) -> Self`; `pub fn Scheduler::queue_len(&self) -> usize`; `pub fn Scheduler::admitted_count(&self) -> usize` (running + waiting with a reservation: the count batch growth limits)
- `SchedulerSnapshot::{token_limit(&self, SeqId) -> Option<u32>, remaining_tokens(&self) -> Vec<u32>, running_ids(&self) -> Vec<RequestId>, queued_ids(&self) -> Vec<RequestId>, is_idle(&self) -> bool}`
- `sim::overload::{OverloadConfig { seed, pool_blocks, max_seq_len, params: SchedulerParams, cost: CostModel, reliability: ReliabilityConfig, rate_multiple, prompt_range, max_tokens_range }, Outcome::{Completed, Cancelled, Rejected(String), Failed(String)}, OverloadReport { kv_capacity_bytes, max_kv_committed_plus_reserved, preempted_below_survival, red_growth /* RED plans after a RED plan whose admitted count rose; must be 0 */, outcomes, max_queue_len, states, green_after_stop, final_circuit }, IterationReport { recovery, attempt_batch_sizes, failed_requests }, OverloadSim::{new, config() -> &OverloadConfig, submit_now(prompt, max_tokens: Option<u32>) -> RequestId, cancel, step_iteration() -> IterationReport, run_for, run_until_done, run_load(load, quiet) -> OverloadReport, inject_oom_attempts(u32), outcome, stream_tail, running_ids, queued_ids, is_queued, pool_used_blocks, ledger_idle, mean_step_time, report}}`

Covers: S-3 (scheduler side), S-9 (engine admission), S-10, S-11, S-17; `overload_sim cancellation_releases_reservations`, `overload_sim ten_x_overload`, `overload_sim active_generations_protected`, `overload_sim oom_recovery_bounded`.
Depends on: Tasks 3, 7, 8, 10; phase-2 plan (`Scheduler`, `BlockPool`, `sim::{SimExecutor, CostModel, ArrivalProcess}`, `CancelReason::{QueueTimeout, CircuitOpen}`, `PreemptReason::SurvivalDecodeAlloc`, `turbine_tensor::host::HostMemory`).

- [ ] Write failing test `crates/turbine-scheduler/tests/overload_sim.rs` `cancellation_releases_reservations`: with a 1,024-block pool, 100 queued and 100 running simulated requests (plus 100 short ones) are cancelled; after one `step_iteration` the `kv` ledger `used + reserved == 0`, workspace reservations 0 and `pool_used_blocks() == 0`.
- [ ] Write failing test `ten_x_overload`: arrivals at 10 × service rate (prompts 64–6000, max tokens 16–1024, seed 7) for 600 virtual seconds then 120 s silence: `max_kv_committed_plus_reserved ≤ kv_capacity_bytes`, `preempted_below_survival == 0`, every outcome is completed/cancelled/rejected with a reject-table code (at least one `queue_full`/`queue_timeout`/`overloaded`), `max_queue_len ≤ 256`, the state reaches RED, `red_growth == 0`, completions ≥ 50 % of the capacity bound `config().service_rate() × 600` (688; amendment 2026-09-26 — measured 491 = 71 % with RED refilling finished slots, 16 with admit-nothing RED, 166 when the freeze counts only running requests), GREEN + HEALTHY within 60 s of the load stopping; the test prints served / `queue_full` / `queue_timeout` / `overloaded` / recovery time.
- [ ] Write failing test `active_generations_protected`: 8 decodes of 4,000 tokens run 5 s, then 2,000 prefills of 6,000 tokens flood in; the 8 decodes' mean step time stays ≤ 1.5 × the pre-flood value, all 8 complete, no preemption below SURVIVAL.
- [ ] Write failing test `oom_recovery_bounded`: OOM on the next 2 attempts → one `step_iteration` reports `Recovered { retries: 2 }` with 3 strictly shrinking attempt batch sizes; persistent OOM → 4 attempts (original + `max_retries` 3), `Failed`, every batch request `Failed("resource_exhausted")` with stream tail `data: {"error":{…"type":"server_error","code":"resource_exhausted"}}` then `data: [DONE]`; a fresh request afterwards completes.
- [ ] Run: `cargo test -p turbine-scheduler --test overload_sim` — expect FAIL
- [ ] Implement the KV side: `with_ledger` asserts the L0 pool fits the `kv` pool capacity; `allocate_reserved` allocates via the P2 `allocate` and `commit_bytes(n × block_bytes)` on the request's worst-case reservation (saturating, so blocks re-allocated after a recompute are not paid twice); releasing blocks never touches the ledger — dropping the request's `Reservation` does.
- [ ] Implement the scheduler side: `AdmissionGate::offer` decides with the controller snapshot, reserves KV for an immediate admit only when the queue is empty and an admitted slot is free (`admitted < max_running` and within the plan's batch growth over `prev_admitted`, amendment 2026-09-26 — otherwise requests behind a full batch were admitted with reservations that never time out), otherwise enqueues (full → `Rejected(QueueFull)`); `plan` first turns `expire()` into `dropped` with `CancelReason::QueueTimeout` and, when the circuit blocks readiness, `reject_all()` into `CancelReason::CircuitOpen`; P2 rule (2) preempts only when `limits.allow_preempt` (SURVIVAL, reason `SurvivalDecodeAlloc`); rule (3) scales chunk and prefill budget by the plan and replaces "admit waiting" by `gate.pump(max_new)` with `max_new = 0` when `!admit_new || !start_new_prefills || shrink_only` (SURVIVAL), else `(prev_admitted + batch_growth_limit − admitted).min(max_running − admitted)` where `admitted = running + waiting` (every waiting request holds its reservation) and `prev_admitted` is the admitted count after the previous plan — so `Some(0)` (ORANGE, RED) refills finished slots without growth; `admit_new` is false only for `AdmissionMode::Stopped`; admitted requests start prefill as the budget allows, without a further growth check. Without a gate the P2 behaviour is unchanged (`IterationLimits::default().allow_preempt == true`).
- [ ] Implement `sim::overload`: the real `Scheduler` + `AdmissionGate` + `PressureController` + `RecoveryController` on a `FakeClock`, a `BlockPool` over `HostMemory` with a 2-layer 2-head 64-dim BF16 layout (16,384 B blocks) linked to the ledger, virtual time advanced by the P2 `CostModel`, seeded `ChaCha8Rng` Poisson arrivals, synthetic telemetry every 100 ms from the ledger (`kv_utilization`, `queue_fill`), OOM injection that walks the same `on_oom` → `RecoveryStep` → `on_recovery` path the engine uses (Task 14).
- [ ] Run: `cargo test -p turbine-scheduler --test overload_sim && cargo test -p turbine-scheduler && cargo tree -p turbine-scheduler | grep -E 'turbine-kernels|turbine-model'; test $? -eq 1` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(scheduler): admission gate, throttle overlay, ledger-backed KV and overload simulation`

## Task 12a: SURVIVAL liveness and KV headroom (amendment 2026-09-27)

Files: `crates/turbine-core/src/config/reliability.rs` (`SurvivalLiveness`, `reliability.recovery.survival_liveness`), `crates/turbine-core/src/config/tests.rs`, `crates/turbine-reliability/src/{throttle.rs, recovery.rs, controller.rs, admission.rs}`, `crates/turbine-scheduler/src/{scheduler.rs, gate.rs, request.rs, sim/overload.rs}`, `crates/turbine-scheduler/tests/overload_sim.rs`, `crates/turbine-server/src/{engine/loop.rs, reliability.rs}`, `crates/turbine-api/tests/api.rs`, `scripts/lab/phase3-novanas-soak.yaml`
Interfaces:

- `pub enum SurvivalLiveness { RequeueUnstarted /* default, option A */, ContinuePrefills /* option B */ }` (`turbine_core::config`, serde snake_case)
- `ThrottlePlan.requeue_unstarted: bool`; `pub fn throttle::plan_with(state, &SchedulerLimits, SurvivalLiveness) -> ThrottlePlan`; `pub fn recovery::survival_plan(plan, &SchedulerLimits, SurvivalLiveness) -> ThrottlePlan`
- `IterationLimits.requeue_unstarted: bool`; `CancelReason::Overloaded` (`"overloaded"`); `PressureReason::SurvivalRequeue` (`"survival_requeue"`)
- `pub fn AdmissionGate::requeue(&mut self, r: SchedRequest, key: AdmissionKey, submit_no: u64) -> bool`; `pub fn Admission::record_requeue(&self, id: RequestId, est: &ResourceEstimate)`; `pub fn Admission::with_kv_headroom(self, kv: SignalThresholds) -> Self`

Covers: S-9 (KV headroom), S-11 (SURVIVAL liveness); `overload_sim survival_liveness_seed_6`, `survival_liveness_seed_1`, `survival_liveness_option_b`, `survival_requeues_unstarted_admitted`, `admission::tests::kv_headroom`, `recovery::tests::survival_plan_switch`.
Depends on: Tasks 7, 8, 10, 12, 14; decision "Phase 3: SURVIVAL liveness fix" (provisional A, pending user review).

- [ ] Write failing tests `overload_sim survival_liveness_seed_6` (the `ten_x_overload` workload, seed 6, passes through SURVIVAL and is GREEN + HEALTHY within 60 s of the load stopping with every overload invariant) and `survival_liveness_seed_1` (seed 1, same criterion), `survival_liveness_option_b` (both seeds under `continue_prefills`), `survival_requeues_unstarted_admitted` (an OOM-triggered SURVIVAL requeues exactly the admitted requests that had not started and releases their reservations; all complete afterwards).
- [ ] Run: `cargo test -p turbine-scheduler --test overload_sim survival_` — expect FAIL (seed 6 stays in SURVIVAL with 0.934 of the pool reserved; seed 1 recovers in 64 s: its end-of-load backlog re-escalates YELLOW → RED).
- [ ] Implement option A behind `reliability.recovery.survival_liveness` (default `requeue_unstarted`): the SURVIVAL throttle plan sets `requeue_unstarted`; `Scheduler::plan` then returns each admitted request that was never given a running slot to the gate's queue at its policy key and original arrival, dropping its reservation (`survival_requeue` queue decision, INFO `survival_requeue`), or answers it `overloaded` when the queue is full; option B (`continue_prefills`) instead gives SURVIVAL RED's prefill budget and chunk floor with no new starts.
- [ ] Implement the KV headroom rule: with adaptive admission, in YELLOW, ORANGE and RED `decide`, `evaluate` and `evaluate_refill` answer `Queue(kv_reservation)` when the worst-case reservation would lift `kv_utilization` past the next state's `kv_utilization` threshold (`with_kv_headroom(effective_thresholds(..)[KvUtilization])` in the server and the simulator).
- [ ] Run: `cargo test -p turbine-scheduler --test overload_sim && cargo test -p turbine-reliability && cargo test -p turbine-core config::tests::reliability_config_validation` — expect PASS; `cargo test --release -p turbine-scheduler --test overload_sim survival_liveness_sweep -- --ignored --nocapture` prints seeds 1–12 under both options (measured: every seed GREEN + HEALTHY 42–49 s after the load stops; seed 6: 42.9 s under A, 46.9 s under B).
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings (plus `--all-features`)
- [ ] Commit: `fix(reliability): SURVIVAL requeues unstarted admitted requests; admissions keep KV headroom`

## Task 12b: device_memory counts idle pre-allocated pools as free (amendment 2026-09-27)

Files: `crates/turbine-reliability/src/{signals.rs, controller.rs}`, `crates/turbine-scheduler/src/sim/overload.rs`
Interfaces:

- `pub struct DeviceMemoryInput { device: DeviceId, memory_kind: MemoryKind, budget_bytes: u64, idle_preallocated_bytes: u64 }` replaces the `(DeviceId, MemoryKind, u64)` tuple of `SignalInputs.devices`; `device_memory` = (used − `idle_preallocated_bytes`) / `budget_bytes`
- `PressureController::tick` fills `idle_preallocated_bytes` from the ledger: `kv` pool `available()` + `reserve` pool `used`

Covers: S-6 (`device_memory` definition), S-19 (the soak's calibration); `signals::tests::device_memory_counts_idle_preallocated_bytes_as_free`, `controller::tests::idle_full_budget_kv_pool_stays_green`, `overload_sim` with device memory reported.
Depends on: Tasks 2, 10, 12, 12a; decision "Phase 3: device_memory counts idle pre-allocated pools as free" (provisional, pending user review).

- [ ] Write failing tests `signals::tests::device_memory_counts_idle_preallocated_bytes_as_free` (the novanas soak startup figures: budget 33,908,850,688, KV pool 23,188,383,744, reserve 2 GiB, 0.972 of the budget measured used → GREEN when idle, 0.972 RED when nothing is idle), `controller::tests::idle_full_budget_kv_pool_stays_green` (30 GiB budget, KV pool the remainder, 28.7 GiB used at idle → GREEN, value = weights + runtime) and make the overload simulation's device sample report memory used (the whole KV pool, the reserve while held, half the workspace).
- [ ] Run: `cargo test -p turbine-reliability && cargo test -p turbine-scheduler --test overload_sim` — expect FAIL (both unit tests; `ten_x_overload` and the three `survival_liveness_*` cases never return to GREEN).
- [ ] Implement `DeviceMemoryInput` and the subtraction in `SignalEvaluator::evaluate`; the controller computes the idle bytes from the ledger each tick.
- [ ] Run: the same — expect PASS; `survival_liveness_sweep -- --ignored --nocapture` unchanged (seeds 1–12 GREEN + HEALTHY 42–49 s after the load stops).
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings (plus the `fault-injection` feature)
- [ ] Run: `scripts/bench-lock.sh scripts/overload-soak.sh novanas` — expect exit 0 and `"pass": true`.
- [ ] Commit: `fix(reliability): device_memory counts idle pre-allocated KV and reserve as free`

## Task 12c: Soak stall — drift per iteration, stale drift, idle floor (amendment 2026-09-27)

Files: `crates/turbine-reliability/src/{step_window.rs (new), controller.rs, admission.rs, lib.rs}`, `crates/turbine-scheduler/src/{gate.rs, scheduler.rs, sim/overload.rs}`, `crates/turbine-scheduler/tests/overload_sim.rs`, `crates/turbine-server/src/engine/loop.rs`
Interfaces:

- `pub struct step_window::DecodeStepWindow { fn new(), fn observe(prefill_tokens: u32, decode_tokens: u32, secs: f64), fn p95() -> Option<f64> }` (window of 64 pure decode iterations; the engine and the simulator both feed it)
- `pub fn Admission::evaluate_idle(&self, est, state, circuit) -> AdmissionDecision`; `pub fn AdmissionGate::pump(&mut self, max_new: usize, idle: bool)`
- `OverloadConfig.degraded_during_load: bool`; `OverloadReport.max_idle_with_queue: Duration`; the simulator sends `CircuitEvent::Iteration` like the server's pressure thread

Covers: S-6 (`step_time_drift`), S-10 (work-conserving floor), S-12 (latency drift), S-19; `step_window::tests::batch_size_is_not_drift`, `controller::tests::drift_is_not_judged_while_idle`, `overload_sim soak_workload_keeps_serving`, `degraded_circuit_keeps_serving`, `ten_x_overload` (idle-with-queue bound).
Depends on: Tasks 10, 12, 12a, 12b, 14; decision "Phase 3: soak stall — drift per iteration, idle floor" (provisional, pending user review).

- [ ] Write failing tests `step_window::tests::batch_size_is_not_drift` (the same iteration time at batch 16 and batch 1 is the same step time), `controller::tests::drift_is_not_judged_while_idle` (YELLOW on 1.95 × baseline drift; with nothing running the state returns to GREEN and no drift signal is emitted), `overload_sim degraded_circuit_keeps_serving` (the soak workload at 4× with the circuit held DEGRADED: never idle > 1 s with requests queued) and `soak_workload_keeps_serving` (seeds 1 and 7); make the simulator send `CircuitEvent::Iteration` and feed `DecodeStepWindow`.
- [ ] Run: `cargo test --release -p turbine-reliability -p turbine-scheduler` — expect FAIL (the window test; the circuit ends DEGRADED in the SURVIVAL liveness cases; `degraded_circuit_keeps_serving` idles 4.2 s with requests queued; the drift test stays YELLOW).
- [ ] Implement: the window records the iteration time (not divided by the batch); the controller ignores drift and does not learn its baseline while nothing runs; the scheduler pumps at least one slot with `idle = true` while nothing is admitted, and the gate decides those with `evaluate_idle` (INFO `admission_decision` reason `idle_floor`); the simulator's RED-growth count exempts 0 → 1.
- [ ] Run: the same — expect PASS; `survival_liveness_sweep` unchanged (42–49 s).
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings (plus the `fault-injection` feature)
- [ ] Run: `scripts/bench-lock.sh scripts/overload-soak.sh novanas` — expect exit 0 and `"pass": true`.
- [ ] Commit: `fix(reliability): drift per decode iteration, none while idle; admission floor when nothing is admitted`
- [ ] Soak config (decision "Phase 3: soak config max_batch_tokens"): `scripts/lab/phase3-novanas-soak.yaml` `scheduler.max_batch_tokens: 2048` (the Phase 2c value its header names); commit `fix(scripts): soak config uses the Phase 2c max_batch_tokens`.

## Task 12d: Latency drift feeds the circuit only in GREEN (amendment 2026-09-27)

Files: `crates/turbine-reliability/src/controller.rs`
Interfaces: none changed; `PressureController::tick` sends `CircuitEvent::LatencyDrift` only while the pressure state is GREEN.

Covers: S-12 (latency drift trigger), S-19; `controller::tests::drift_under_pressure_leaves_the_circuit`.
Depends on: Task 12c; decision "Phase 3: soak config max_batch_tokens" (answer: option 1 tried and reverted, option 2 provisional).

- [ ] Write failing test `controller::tests::drift_under_pressure_leaves_the_circuit`: with pressure ORANGE from KV, a 2.5× and then a 5× drift spike leave the circuit HEALTHY; the same 2.5× spike in GREEN makes it DEGRADED.
- [ ] Run: `cargo test -p turbine-reliability controller::tests` — expect FAIL (the circuit goes DEGRADED under pressure).
- [ ] Implement: `tick` gates the `LatencyDrift` circuit event on `machine.state() == Green`; the `step_time_drift` signal is unchanged.
- [ ] Run: `cargo test --workspace` and `overload_sim -- --include-ignored` — expect PASS (sweep unchanged, 42–49 s).
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings (plus the `fault-injection` feature)
- [ ] Run: `scripts/bench-lock.sh scripts/overload-soak.sh novanas` — expect exit 0 and `"pass": true`.
- [ ] Commit: `fix(reliability): latency drift feeds the circuit only in GREEN`

## Task 12e: Per-shape drift baselines (amendment 2026-09-27)

Files: `crates/turbine-reliability/src/{step_window.rs, controller.rs}`, `crates/turbine-scheduler/src/{scheduler.rs, sim/overload.rs}`, `crates/turbine-server/src/engine/loop.rs`
Interfaces:

- `pub struct step_window::StepSample { prefill_tokens: u32, rows: u32, context_tokens: u64, secs: f64 }`; `DecodeStepWindow::observe(&mut self, StepSample, calm: bool)`, `reset()`, `p95()` (ratio over the bucket baseline); `pub const MIN_BUCKET_SAMPLES: u32 = 8`
- `EngineStats.step_time_p95` (was `step_time_p95_s`, seconds): the ratio; the controller keeps no baseline of its own. `pub fn IterationPlan::decode_context_tokens(&self) -> u64`
- PROBING → HEALTHY resets the window (the engine on its next snapshot, the simulator on the probe's circuit transition)

Covers: S-6 (`step_time_drift`), S-12; `step_window::tests::moe_full_batch_is_not_drift`, `same_bucket_slowdown_is_drift`, `unseen_shapes_and_prefills_are_not_judged`, `context_buckets`.
Depends on: Tasks 12c, 12d; decision "Phase 3: per-shape drift baselines (OLMoE landing regression)" (provisional).

- [ ] Write failing tests: an MoE-like step (time grows with rows) at a full batch of 16 reads 1.0 after a calm ramp over 1..16 rows; the same shape at twice the time reads 2.0; a shape without 8 calm steps, an unseen shape and a prefill are not judged.
- [ ] Run: `cargo test -p turbine-reliability step_window` — expect FAIL.
- [ ] Implement the buckets (exact rows × `floor(2 · log2(context/1024 + 1))`), an EWMA baseline per bucket (α 0.1) from calm steps, the ratio judged before the update; the controller's drift is the window's p95 while decoding.
- [ ] Run: `cargo test --workspace`, `overload_sim -- --include-ignored` (sweep unchanged); mutation check: one bucket for every shape fails the MoE, unseen-shape and bucket tests.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings (plus the `fault-injection` feature)
- [ ] Run: `scripts/lab-bench.sh --gpu 1 --model olmoe` (≥ 599 tok/s, no `step_time_drift` transition in `metrics.txt`) and `--model llama`; then `scripts/bench-lock.sh scripts/overload-soak.sh novanas`.
- [ ] Commit: `fix(reliability): drift baselines per step shape`

## Task 13: Pressure route, readiness, admission errors and metrics in turbine-api

Files: `crates/turbine-api/src/routes/diagnostics.rs` (pressure route 200, status fields), `crates/turbine-api/src/routes/health.rs` (`circuit_open` readiness), `crates/turbine-api/src/error.rs` (overload `ApiError` constructors, `Retry-After`), `crates/turbine-api/src/backend.rs` (`NotReadyReason::CircuitOpen`, `readiness_for_circuit`), `crates/turbine-api/Cargo.toml` (dev-deps `turbine-reliability`, `turbine-device`), `crates/turbine-api/tests/api.rs` (four tests)
Interfaces:

- `NotReadyReason::CircuitOpen` (body `{"ready":false,"reason":"circuit_open"}`); `pub fn readiness_for_circuit(circuit: CircuitState, base: ReadyState) -> ReadyState`
- `pub fn ApiError::overload(code: ErrorCode, retry_after: Option<u64>) -> ApiError` (status/type from the §14.3 table: `context_exceeds_kv_capacity` 400 `invalid_request_error`; `queue_full` 429 `rate_limit_error`; `queue_timeout`, `overloaded`, `circuit_open`, `resource_exhausted` 503 `service_unavailable`)
- `ErrorCode::{Overloaded, CircuitOpen, ResourceExhausted}` (P3 codes in `turbine_core::request`)
- `Diagnostics::pressure(&self) -> Result<serde_json::Value, ApiError>` now returns the document; status JSON gains `pressure_state`, `circuit_state`

Covers: S-12 (readiness), S-13, S-14; `api ready_follows_circuit`, `api pressure_document_shape`, `api admission_error_mapping`, `api reliability_metrics_bounded`.
Depends on: Tasks 7, 10, 11; phase-0 plan (`router`, `ApiState`, `Readiness`, `Diagnostics`, `ApiError`), phase-2 plan (routes).

- [ ] Write failing test `api ready_follows_circuit`: a stub `Readiness` driven through CIRCUIT_OPEN, DRAINING, PROBING and HEALTHY returns `/ready` 503 `{"ready":false,"reason":"circuit_open"}` for the first three and 200 after; a stub backend returning `ApiError::overload(CircuitOpen, Some(12))` makes `POST /v1/chat/completions` answer 503 code `circuit_open` with `retry-after: 12` (≥ 1).
- [ ] Write failing test `api pressure_document_shape`: a stub `Diagnostics` serving a `PressureController` document (Task 10) → `GET /turbine/v1/pressure` 200 with every key of the P3 §Data example and matching JSON types (strings, numbers, arrays, objects, nullable `last_reason`); `/turbine/v1/status` includes `pressure_state` and `circuit_state`; with `reliability.enabled: false` the document has `"enabled": false` and `"state": "GREEN"`.
- [ ] Write failing test `api admission_error_mapping`: for each reject row (`context_exceeds_kv_capacity` 400/no Retry-After, `queue_full` 429 Retry-After 7, `queue_timeout` 503 Retry-After 7, `overloaded` 503 Retry-After 7, `circuit_open` 503 Retry-After 1) the response status, `error.type`, `error.code` and `retry-after` header match, and they equal `RejectionReason::http()` for all five reasons.
- [ ] Write failing test `api reliability_metrics_bounded`: registers `ReliabilityMetrics` and `TelemetryMetrics` on the router's registry, drives a state machine through every pressure state, the circuit through DEGRADED/OPEN/DRAINING/PROBING/OPEN, every queue and reject reason, recovery `recovered`/`failed`, reserve release, reclaim, pool gauges and one device sample; `GET /metrics` has a `# TYPE` line for each P3 family and every label value belongs to its closed set (states, signals, decisions, reasons, fields, actions, pools, kinds, outcomes, device index < 64, source `host`/`storage`/index).
- [ ] Run: `cargo test -p turbine-api --test api -- ready_follows_circuit pressure_document_shape admission_error_mapping reliability_metrics_bounded` — expect FAIL
- [ ] Implement the route/readiness/error changes: `/turbine/v1/pressure` returns `diagnostics.pressure()` instead of 501; `readiness_for_circuit` maps `blocks_readiness()` to `NotReady { CircuitOpen }`; `ApiError::overload` fills status/type from the code and sets `Retry-After` when given; the status document fields come from the server (Task 14).
- [ ] Run: `cargo test -p turbine-api --test api` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(api): pressure document, circuit readiness and admission error mapping`

## Task 14: Server wiring: budget, reserve, controller, sampler, recovery, circuit and fault injection

Files: `crates/turbine-server/Cargo.toml` (deps `turbine-reliability`; feature `fault-injection = ["turbine-core/fault-injection", "turbine-reliability/fault-injection"]`), `crates/turbine-server/src/startup.rs` (`compute_budget` after weights with the P1 pre-check before, ledger, reserve allocation, controller/sampler threads), `crates/turbine-server/src/reliability.rs` (new: `DeviceReserve` allocator, `EngineLedgerProbe`, controller supervision thread, probes, rejection → `ApiError` mapping, fault wrappers), `crates/turbine-server/src/engine/mod.rs` (gate, `IterationLimits::from(&handle.throttle())`, OOM recovery loop, device-error/sticky handling, drain, exit 3), `crates/turbine-server/src/exit.rs` (`ExitCode::DeviceFatal = 3`), `crates/turbine-server/tests/fault.rs` (new: `alloc_fail_every_injects`, `sticky_device_error_exits_3`)
Interfaces:

- `pub struct DeviceReserve` (`ReserveAllocator` over `DeviceBuffer::alloc(&Arc<dyn DeviceMemory>, bytes)`; `free` drops the buffer)
- `pub struct EngineLedgerProbe { ledger: Arc<Ledger>, device: DeviceId, queue_len: Arc<AtomicU32>, max_queue: u32 }` (`LedgerProbe`)
- `pub fn api_error_for(reason: RejectionReason, retry_after_secs: u64) -> ApiError`
- `pub fn spawn_controller(controller: Arc<Mutex<PressureController>>, latest: LatestSample, stats: Arc<ArcSwap<EngineStats>>, interval: Duration, wake: WeakSender<EngineCommand>, stop: Arc<AtomicBool>) -> io::Result<JoinHandle<()>>` (amended 2026-09-26: the engine thread shares the controller behind one mutex for `on_oom` / `on_recovery` / circuit events — never per token — and the thread wakes the idle engine on a circuit change so PROBING starts its probes; runs `tick` per fast tick under `catch_unwind`, reporting new engine iterations as `CircuitEvent::Iteration`; a panic calls `on_circuit_event(ControllerFailed)` → CIRCUIT_OPEN `controller_failed`, fatal: the engine drains up to `drain_timeout` and the process exits 3)
- `#[cfg(feature = "fault-injection")] pub struct FaultyVendor { inner: Box<dyn VendorTelemetry>, temperature_c: Option<f64>, delay: Option<Duration> }` and `FaultyExecutor` wrapping `ModelExecutor::forward` with `FaultInjector::iteration_fault`
- `ExitCode::DeviceFatal = 3`; `KernelError::is_sticky()`, `KernelError::is_oom()` (P3 in turbine-kernels, contract §7.1)

Covers: S-2/S-4 (startup), S-11 (engine), S-12 (sticky → exit 3), S-16; `fault alloc_fail_every_injects`.
Depends on: Tasks 1–13; phase-1/2 plans (`startup` order, engine thread, `ShimContext`, `ModelExecutor`, `turbine_model::budget` pre-check), phase-0 plan (`discover`, exit codes).

- [ ] Write failing test `crates/turbine-server/tests/fault.rs` `alloc_fail_every_injects` (`--features fault-injection`, macOS, CPU backend with the P1 tiny checkpoint and simulated executor): config `reliability.fault_injection.alloc_fail_every: 5` → exactly every fifth pool reservation fails (5th, 10th, … of the first 20 counted through the ledger), each counted in `turbine_allocation_failures_total` scraped from `/metrics`; the same config under a default build exits 2 naming `reliability.fault_injection`.
- [ ] Run: `cargo test -p turbine-server --features fault-injection --test fault alloc_fail_every_injects` — expect FAIL
- [ ] Implement startup: after weights load, `compute_budget` with `measured_free_bytes` from `DeviceMemory::mem_info` (dedicated) or `MemAvailable` (unified, from `read_host(&FsProc::default())`) and `already_held_bytes` = weights; exit 1 before binding with `BudgetError.breakdown`; `Ledger::new`, weights reservation committed, `EmergencyReserve::acquire` with `DeviceReserve` (0 bytes → WARN), `L0Reclaimer`, `PressureController`, `TelemetrySampler::spawn(TelemetryConfig::from_config(..), &inventory, vendor_backends(devices.nvml_library, devices.amd_smi_library), Box::new(FsProc::default()), probe, clock)`, `spawn_controller`; `BlockPool::with_ledger`; the engine's `Scheduler::with_gate` built from `Admission` + `AdmissionQueue`.
- [ ] Implement the engine loop: `IterationLimits::from(&handle.throttle())` once per iteration; publish `EngineStats`; on `KernelError::is_oom()` call `controller.on_oom()` then `RecoveryController::on_oom(batch)` — `Retry` sleeps `backoff` and re-plans with at most `batch_limit` sequences, `GiveUp` fails the batch's requests with `GenerationEvent::Error { code: ResourceExhausted }` (stream: error event then `[DONE]`; non-stream 503) and keeps serving; success reports `on_recovery`; a non-OOM error → `CircuitEvent::DeviceError { sticky: err.is_sticky() }`; while the circuit is OPEN/DRAINING admission rejects `circuit_open` and running sequences continue until `drain_expired()` (then fail `circuit_open`); PROBING runs internal 16-token greedy probes of a fixed prompt that bypass the queue and report `ProbeSucceeded { latency_ratio }` / `ProbeFailed`; when `snapshot.fatal` the engine stops device work, fails running sequences `resource_exhausted`, sets `/ready` 503 and exits with `ExitCode::DeviceFatal` (3) within `drain_timeout`; SIGTERM keeps the P2 graceful bound in every state.
- [ ] Implement fault injection behind the feature: `Ledger::set_fault_injector`, `FaultyExecutor` raising `KernelError::OutOfMemory` at `oom_at_iteration` and `KernelError::Device { message: "hipErrorIllegalAddress: injected" | "injected kernel error" }` at `kernel_error_at_iteration` (sticky per `kernel_error_sticky`), `FaultyVendor` overriding temperature and delaying calls.
- [ ] Run: `cargo test -p turbine-server --features fault-injection --test fault alloc_fail_every_injects && cargo test --workspace` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings (plus `--all-features`)
- [ ] Commit: `feat(server): wire memory budget, pressure controller, telemetry, recovery and circuit breaker`

## Task 15: Open-loop overload mode in turbine-bench

Files: `benches/turbine-bench/src/open_loop.rs` (new: seeded Poisson arrivals, ranges, breakdown, pressure timeline), `benches/turbine-bench/src/main.rs` (flags `--duration`, `--rate`, `--prompt-words-range`, `--max-tokens-range`, `--pressure-timeline`; report keys), `benches/turbine-bench/src/report.rs` (`by_status`, `by_error_code`, `client_dropped`, `streams_incomplete`), `benches/turbine-bench/tests/bench.rs` (new test)
Interfaces:

- `pub struct RangeArg { pub min: u32, pub max: u32 }` (`FromStr` for `<min>..<max>`)
- `pub struct OpenLoopRng` with `new(seed: u64)`, `inter_arrival(&mut self, rate: f64) -> Duration`, `draw(&mut self, r: RangeArg) -> u32` (`rand_chacha::ChaCha8Rng`, the bench's seeded PRNG)
- `pub fn arrival_schedule(rate: f64, duration: Duration, seed: u64) -> Vec<Duration>`
- `pub struct Breakdown { by_status: BTreeMap<String, u64>, by_error_code: BTreeMap<String, u64>, client_dropped: u64, streams_incomplete: u64 }` with `record(status: u16, error_code: Option<&str>, stream_done: bool)`, `dropped()`
- `pub fn error_code(body: &str) -> Option<String>`; `pub fn timeline_line(t: f64, doc: Result<serde_json::Value, String>) -> String`; `pub async fn pressure_timeline(client: reqwest::Client, url: String, out: PathBuf, stop: tokio::sync::watch::Receiver<bool>) -> std::io::Result<()>`

Covers: S-18; `bench open_loop_rate_and_breakdown`.
Depends on: phase-0 plan (`turbine-bench` CLI, report, mock-endpoint test helpers).

- [ ] Write failing test `benches/turbine-bench/tests/bench.rs` `open_loop_rate_and_breakdown`: an in-process axum mock answering chat completions 200 (streaming, with `[DONE]`), 429 `queue_full` and 503 `overloaded` in a fixed 3-cycle pattern plus a `/turbine/v1/pressure` document; `turbine-bench --url <mock> --rate 50 --duration 4s --seed 3 --concurrency 64 --pressure-timeline <tmp>/t.jsonl --output json` → 200 ± 30 arrivals, `by_status` and `by_error_code` equal to the mock's counts, identical `arrival_schedule(50, 4 s, 3)` on two calls, and 4 ± 1 timeline lines each with `t`, `state`, `circuit`, `dominant_signal`, `queue`, `kv_utilization`.
- [ ] Run: `cargo test -p turbine-bench --test bench open_loop_rate_and_breakdown` — expect FAIL
- [ ] Implement open-loop mode: exponential inter-arrival times `−ln(U)/rate` from the seeded `ChaCha8Rng`, `--concurrency` caps outstanding requests (excess arrivals count `client_dropped`), ranges drawn uniformly per request, `--duration` replaces `--requests`, the timeline task polls once per second and writes `{"t":…,"error":…}` on failure, and the report adds the four keys.
- [ ] Run: `cargo test -p turbine-bench` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(bench): open-loop overload mode with status breakdown and pressure timeline`

## Task 16: Live telemetry lab test on all three hosts

Files: `crates/turbine-device/tests/lab.rs` (add ignored test `live_telemetry`), `AGENTS.md` (Commands: reliability tests, `--features fault-injection`, lab commands)
Interfaces:

- consumes `SamplerCore`, `vendor_backends`, `FsProc`, `TelemetryConfig` (Task 11), `turbine_device::discover` (P0), `TURBINE_EXPECT_NVIDIA` (P0)

Covers: S-5, S-14; `turbine-device --test lab live_telemetry`.
Depends on: Task 11; phase-0 plan (`scripts/lab-test.sh`, `lab inventory_matches_expectation`); phase-2b plan (Spark lab image).

- [ ] Write failing test `crates/turbine-device/tests/lab.rs` `live_telemetry` (`#[ignore]`): discovers the inventory, runs 10 vendor ticks with the real `SystemClock`, and asserts every device has non-stale `temperature_c` and `clock_mhz`, AMD devices have `memory_used_bytes`/`memory_free_bytes`, unified (GB10) devices have none, and the host `mem_available_bytes > 0`; telemetry is read-only (no GPU memory allocated).
- [ ] Run: `cargo test -p turbine-device --test lab live_telemetry -- --ignored` on macOS — expect FAIL (no devices)
- [ ] Implement nothing beyond the test (Task 11 provides the sampler); update the `AGENTS.md` Commands section with `cargo test -p turbine-reliability`, `cargo test -p turbine-scheduler --test overload_sim`, `cargo test -p turbine-server --features fault-injection --test fault`, `scripts/overload-soak.sh <host>`.
- [ ] Run: `scripts/lab-test.sh novanas` — expect exit 0 and the log line `test live_telemetry ... ok` (read-only: no memory needs freeing).
- [ ] Run: `scripts/lab-test.sh dgx-spark` and `scripts/lab-test.sh dgx-spark2` — expect exit 0 and `test live_telemetry ... ok` (proceed only when the Phase 2b MemAvailable precondition passes; otherwise ASK THE USER FIRST).
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `test(device): live telemetry lab test on novanas and the Sparks`

## Task 17: Sticky device error exits 3 (lab)

Files: `crates/turbine-server/tests/fault.rs` (ignored test `sticky_device_error_exits_3`), `scripts/lab-test.sh` (optional `--features <list>` forwarded to `cargo test`)
Interfaces:

- `scripts/lab-test.sh <host> [--features fault-injection]`; config keys `reliability.fault_injection.kernel_error_at_iteration`, `reliability.fault_injection.kernel_error_sticky: true`

Covers: S-12, S-16; `fault sticky_device_error_exits_3`.
Depends on: Task 14; phase-1 plan (novanas GPU lab path), phase-2b plan (Spark CUDA path).

- [ ] Write failing test `sticky_device_error_exits_3` (`#[ignore]`, `require_backend` per `TURBINE_TEST_BACKEND`): starts `turbine-server` on the real Llama-3.2-3B with `kernel_error_at_iteration: 3` and `kernel_error_sticky: true`, sends one streaming request, and asserts the stream ends with an error event then `[DONE]`, `/ready` returns 503 `circuit_open`, and the process exits with code 3 within `drain_timeout` (120 s).
- [ ] Run: `cargo test -p turbine-server --features fault-injection --test fault sticky_device_error_exits_3 -- --ignored` on macOS — expect FAIL (no GPU backend)
- [ ] Implement the `--features` pass-through in `scripts/lab-test.sh` (appended to the in-container `cargo test --workspace` invocation only).
- [ ] Run: ASK THE USER FIRST to confirm an R9700 on novanas is free, then `scripts/lab-test.sh novanas --features fault-injection` — expect exit 0 with `test sticky_device_error_exits_3 ... ok`.
- [ ] Run: once the NVIDIA path exists and after asking the user if MemAvailable is short, `scripts/lab-test.sh dgx-spark --features fault-injection` — expect exit 0 with `test sticky_device_error_exits_3 ... ok`.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings (plus `--all-features`)
- [ ] Commit: `test(server): sticky device error opens the circuit and exits 3`

## Task 18: Overload soak script and soak runs

Files: `scripts/overload-soak.sh` (new: precondition → build/start → calibrate → overload → cool-down → verdict → cleanup), `scripts/lab/phase3-novanas-soak.yaml` (new, amended 2026-09-26: the server config the soak serves through `scripts/lab-serve.sh novanas`, whose k3s Job `turbine-lab-serve-<run id>` already is namespace `turbine-ci`, `amd.com/gpu: 1`, ROCm hostPath `/opt/rocm/rocm`, model read-only — instead of a second Job template; the trap stops only that run with `lab-serve.sh novanas --stop <run id>`; the novanas precondition reads amdgpu sysfs `mem_info_vram_used`), `benches/turbine-bench/tests/lab_scripts.rs` (test `soak_precondition_refuses_busy_gpu`)
Interfaces:

- `scripts/overload-soak.sh <novanas|dgx-spark|dgx-spark2> [--duration <dur>=10m] [--model <path>=/home/piwi/turbine-models/llama-3.2-3b-instruct]`
- shell function `soak_precondition <host> <vram_used_bytes|mem_available_bytes>` (exit 1 printing `precondition`); outputs under `target/soak/<host>-<timestamp>/{calibrate.json,overload.json,timeline.jsonl,verdict.json}`

Covers: S-19, S-4, S-11; `scripts/overload-soak.sh novanas` precondition refusal (AC S-19), manual soak runs (AC S-19/S-4/S-11).
Depends on: Tasks 14–15; phase-0/1 plans (`scripts/lab-test.sh`, `scripts/lab-serve.sh`, k3s namespace `turbine-ci`), phase-2b plan (Spark image).

- [ ] Write failing test `benches/turbine-bench/tests/lab_scripts.rs` `soak_precondition_refuses_busy_gpu`: sourcing `scripts/overload-soak.sh` with `SOAK_SOURCE_ONLY=1`, `soak_precondition novanas 4294967296` (4 GiB VRAM in use) exits 1 with `precondition` on stderr and starts nothing; `soak_precondition novanas 536870912` passes; `soak_precondition dgx-spark 26843545600` (25 GiB MemAvailable < 24 GiB cap + 8 GiB reserve) fails and 42 GiB passes.
- [ ] Run: `cargo test -p turbine-bench --test lab_scripts soak_precondition_refuses_busy_gpu` — expect FAIL
- [ ] Implement the script: step 0 prints host, GPU and claimed memory and checks the precondition (novanas: `amd-smi metric --mem-usage` over SSH < 1 GiB used on the target R9700; Sparks: `MemAvailable` > 24 GiB cap + `host_reserve_bytes`), never freeing anything; step 1 starts `turbine-server` (novanas: `kubectl apply` of the Job; Sparks: `docker run --gpus all --memory 24g --name turbine-lab-soak` from the Phase 2b lab image) and waits ≤ 10 min for `/ready`; step 2 calibrates 2 min closed-loop `--concurrency 4` (baseline ITL p99, throughput R); step 3 overload `--rate 4R --prompt-words-range 64..6000 --max-tokens-range 16..1024 --concurrency 1024 --pressure-timeline`; step 4 cool-down 5 min with the timeline; step 5 evaluates the pass criteria (no restart/OOM kill, 5xx only 503 `overloaded`/`queue_timeout`/`circuit_open`, `streams_incomplete == 0`, overload ITL p99 ≤ 2 × calibration, timeline ≥ ORANGE, GREEN + HEALTHY within 60 s of cool-down, KV `used_bytes`/`reserved_bytes` at idle values, `emergency_reserve_held: true`), prints `verdict.json`, exits 0/1, and a `trap` always removes its Job/container (only `turbine-lab-*`).
- [ ] Run: `cargo test -p turbine-bench --test lab_scripts soak_precondition_refuses_busy_gpu && bash -n scripts/overload-soak.sh` — expect PASS
- [ ] Run: `scripts/overload-soak.sh novanas` while the R9700 still holds another workload — expect exit 1 with `precondition` and no Job created (`kubectl -n turbine-ci get jobs` unchanged).
- [ ] Run: ASK THE USER FIRST to free an R9700, then `scripts/overload-soak.sh novanas` — expect exit 0 and `"pass": true` in `verdict.json`; paste the verdict into task evidence.
- [ ] Run: ASK THE USER FIRST (before the Phase 3 exit), then `scripts/overload-soak.sh novanas --duration 4h` — expect exit 0 and `"pass": true`; paste the verdict.
- [ ] Run: ASK THE USER FIRST (GB10 compute is shared with production vLLM), then once the NVIDIA path exists `scripts/overload-soak.sh dgx-spark` (24 GiB container cap) — expect exit 0 and `"pass": true`; paste the verdict.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(scripts): overload soak with precondition, pressure timeline and pass verdict`

## Open questions

- Decided 2026-09-25 (user): add `arc-swap` 1.x for the lock-free latest-value cell and atomic plan snapshot — the one runtime dependency added beyond Phases 0–2.
