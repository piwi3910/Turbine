# phase-2-serving-runtime — implementation plan

Status: draft
Spec: .procoder/specs/phase-2-serving-runtime.md

## Goal

Turn the Phase 1 single-request server into a serving runtime on the novanas R9700: continuous batching with separate prefill/decode queues and chunked prefill over a paged BF16 KV block pool, bounded queues with backpressure, prompt cancellation and preemption by recompute, a deterministic scheduler simulator, the OpenAI fields agents need (n, penalties, bias, tools, JSON-schema output via llguidance), and OLMoE-1B-7B as the first MoE model — all validated against the Phase 1 golden tolerance under concurrency.

## Architecture

Two new GPU-free crates carry the runtime logic: `turbine-kv` (`BlockPool` accounting over one preallocated device buffer, `BlockTable`, `KvMetrics`, `KvDocument`) and `turbine-scheduler` (request lifecycle, waiting/prefill/decode queues, per-iteration `plan`/`complete`, preemption, the `sim` module with `SimExecutor` and virtual time). `turbine-model` gains ragged paged batches in `ModelExecutor::forward`, the `OlmoeExecutor`, penalties and token masks in the sampler, `structured` (llguidance matchers) and `tools` (llama3_json parser and grammar); `turbine-kernels` and both sides of the C ABI move to v2 (paged attention, `copy_blocks`, `moe_route`, `moe_experts`, `turbine_ctx_get_info`). `turbine-server` replaces the Phase 1 generation thread by a dedicated engine OS thread that owns the context, scheduler, pool, executor, samplers and matchers, fed by a bounded `EngineCommand` channel and emitting into bounded per-request channels; grammar compilation runs on Tokio's blocking pool behind a 4-permit semaphore.

## Constraints

From the spec (verbatim):

- Scheduler logic depends on no GPU, kernel provider or model crate; it sees requests, token counts, block counts and a cost model only (TS §21 rules 6 and 11). `turbine-scheduler` and `turbine-kv` build and test on macOS with no ROCm.
- KV boundaries (TS §21 rule 6): block pool, block table and KV metrics are separate modules in `turbine-kv`; no module named or behaving as a catch-all "KV manager".
- The HTTP runtime (Tokio) never blocks on the engine: submissions go through a bounded channel (capacity _scheduler.max_queued_requests_), outputs through bounded per-request channels; the engine thread never awaits a client. Grammar compilation runs on Tokio's blocking pool behind a semaphore of 4 permits with a timeout of _structured_output.compile_timeout_, never on the engine thread; per-step mask computation runs on the engine thread within llguidance's step limits.
- Startup budget (extends Phase 1): weights + `kv.gpu.max_bytes` + workspace for _scheduler.max_batch_tokens_ (activations, logits, MoE permutation buffers) + _reliability.emergency_vram_reserve_ ≤ available device memory (device free memory on the R9700); violation exits 1 before weights load, naming each term.
- Lab: novanas (192.168.10.203), one R9700 (`gfx1201`, 32 GB) per k3s Job, ROCm 7.14.1; the user empties the GPUs for Turbine work. Any run that needs production workloads moved or memory freed on any host — including the vLLM-ROCm baseline run and the overload run — is preceded by asking the user; the user moves workloads. Scripts never stop, restart or reconfigure other workloads.
- Models: `meta-llama/Llama-3.2-3B-Instruct` and `allenai/OLMoE-1B-7B-0125-Instruct`, BF16, under `/home/piwi/turbine-models/<slug>`, downloaded once over SSH by Claude with a Hugging Face token the user supplies at that time (never stored in the repository or on the host); tests read `TURBINE_TEST_MODEL_DIR` (and `TURBINE_TEST_MOE_MODEL_DIR` for OLMoE) and never download.
- New dependencies beyond Phase 1: `llguidance` and `toktrie_hf_tokenizers` (runtime; exact versions pinned in the workspace manifest, MIT license recorded in the manifest comment); `proptest` and `jsonschema` (dev-dependencies: scheduler invariants, validating structured outputs in tests).
- Bounds on constrained decoding (TS §21 rule 8): schema or tool-definition JSON over _structured_output.max_schema_bytes_ is rejected; grammar compilation over _structured_output.compile_timeout_ is rejected; llguidance's per-step parser limits stay at their defaults and a step that exceeds them fails only that request.

From the interface contract (binding):

- Names in `.procoder/contract/interfaces.md` §3, §7, §9–§12, §14, §16–§20 are used verbatim. §23 picks override the spec: C-1 (Phase 2 uses `scheduler.max_queued_requests` / `scheduler.queue_timeout`), C-2 (error code and metric reason `context_exceeds_kv_capacity`, not `kv_capacity_exceeded`), C-3 (timeout / slow-client / shutdown stream errors are an error event **followed by** `data: [DONE]`), C-4 (KV tier label and document tier `l0`, not `gpu`), C-8 (`kv.gpu.max_bytes: Option<ByteSize>`, default `8GiB` in P2), C-10 (`Priority(i32)`, lower first), C-14 (one duration parser: `ms`, `s`, `m`, `h`), C-15 (slug `olmoe-1b-7b-0125-instruct`), C-25 (3 consecutive failed iterations → `/ready` 503 `device_error`, exit 1).
- Kernel C ABI moves to `TURBINE_ABI_VERSION 2u` / `TURBINE_KERNELS_ABI_VERSION = 2` with exactly the §9.3 v2 declarations.
- Contract §20.2 note: the Phase 1 test `tiny_server single_slot_and_cancel` asserts `429 engine_busy`, which Phase 2 removes; this plan rewrites that test to assert queueing (Task 15).
- Builds on `.procoder/plans/phase-0-skeleton.md` and `.procoder/plans/phase-1-single-request.md` (crates `turbine-tensor`, `turbine-kernels`, `turbine-model`, the P1 server, `turbine-golden`, lab scripts).
- Every task ends gate-clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`, then `cargo test --workspace`.

## Task 1: Phase 2 core vocabulary

Files: `crates/turbine-core/src/clock.rs` (new: `Clock`, `SystemClock`, `FakeClock`), `crates/turbine-core/src/lib.rs` (declare `clock`), `crates/turbine-core/src/types.rs` (`SeqId`, `BlockId`, `Priority`, `PriorityClass`), `crates/turbine-core/src/request.rs` (P2 fields and variants)
Interfaces:

- `pub trait Clock: Send + Sync { fn now_mono(&self) -> Duration; fn now_wall(&self) -> SystemTime; }`, `pub struct SystemClock`, `pub struct FakeClock` with `new(start: Duration)`, `advance(&self, d: Duration)`, `set(&self, t: Duration)`
- `pub struct SeqId(pub u64)`, `pub struct BlockId(pub u32)`, `pub struct Priority(pub i32)` (default 0) with `fn class(self) -> PriorityClass`, `pub enum PriorityClass { High, Normal, Low }` (<0, 0, >0 — C-10)
- `SamplingParams` gains `presence_penalty: f32, frequency_penalty: f32, repetition_penalty: f32 (default 1.0), logit_bias: Vec<(u32, f32)>, min_tokens: u32`; `StopConditions` gains `stop_token_ids: Vec<u32>`; `FinishReason::ToolCalls` (`"tool_calls"`)
- `GenerationRequest` gains `n: u32, priority: Priority, echo: bool, constraint: Option<ConstraintSpec>, deadline_ms: u64`
- `pub enum ConstraintSpec { JsonObject, JsonSchema { schema: serde_json::Value }, ToolCall { grammar_source: String } }`
- `GenerationEvent::ToolCalls { choice: u32, calls: Vec<ToolCallOut> }`, `pub struct ToolCallOut { pub index: u32, pub id: String, pub name: String, pub arguments: String }`
- `pub struct ResourceEstimate { prompt_tokens, cached_prefix_tokens, new_prefill_tokens, max_output_tokens, projected_kv_blocks: u32, workspace_bytes: u64, est_prefill_seconds, est_decode_seconds: f64, estimated: bool }` with `fn for_request(prompt: u32, max_output: u32, block_tokens: u32) -> ResourceEstimate` (P2 fills the token fields and `projected_kv_blocks = ceil((prompt+max_output)/block_tokens)`)
- `ErrorCode` gains `QueueFull`, `QueueTimeout`, `ContextExceedsKvCapacity`, `InvalidJsonSchema`, `ToolsNotSupported`, `UnknownTool`, `ShuttingDown`, `RequestTimeout`, `SlowClient`
  Covers: S-2 (request vocabulary), S-10/S-17/S-18 (field types)
  Depends on: Phase 1 plan Task 1

- [ ] Write failing test `turbine-core request::tests::resource_estimate_blocks`: `ResourceEstimate::for_request(100, 60, 16).projected_kv_blocks == 10`, `Priority(-1).class() == High`, `Priority(3).class() == Low`, and a `FakeClock` advanced by 250 ms reports `now_mono() == start + 250 ms`. Run: `cargo test -p turbine-core request::tests::resource_estimate_blocks` — expect FAIL
- [ ] Implement the additions; every extended enum stays `#[non_exhaustive]`; `ErrorCode::as_str` covers the new variants in snake_case.
- [ ] Run: `cargo test -p turbine-core` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-core): phase 2 vocabulary, clock and resource estimate`

## Task 2: Phase 2 configuration keys and durations

Files: `crates/turbine-core/src/config/duration.rs` (new: `HumanDuration`), `crates/turbine-core/src/config/mod.rs` (new keys, `StructuredOutputConfig`, validation), `crates/turbine-core/src/config/tests.rs`
Interfaces:

- `pub struct HumanDuration(pub std::time::Duration)` — `<integer><ms|s|m|h>`, no space; errors name the key (C-14)
- `ServerConfig` gains `request_timeout` (`10m`), `slow_client_timeout` (`30s`), `shutdown_grace` (`30s`, ≥ 0)
- `SchedulerConfig` gains `max_running_requests: u32` (64, 1..=1024), `max_batch_tokens: u32` (8192, ≥ max_running_requests and ≥ `kv.block_tokens`), `prefill_chunk_tokens: u32` (2048, 1..=max_batch_tokens), `max_queued_requests: u32` (256, 1..=65536), `queue_timeout: HumanDuration` (`60s`, > 0)
- `KvGpuConfig` gains `max_bytes: Option<ByteSize>` (default `Some(8GiB)`, C-8); `kv.gpu.enabled: false` → startup failure (Task 15)
- `ModelConfig` gains `tool_call_parser: Option<ToolCallParserKind>` with `pub enum ToolCallParserKind { Llama3Json, None }` (`llama3_json` | `none`)
- `pub struct StructuredOutputConfig { max_schema_bytes: ByteSize (64KiB, 1KiB..=1MiB), compile_timeout: HumanDuration (5s, > 0) }` as `Config::structured_output`
- `scheduler.continuous_batching: false` forces `max_running_requests` to 1 (`Config::effective_max_running(&self) -> u32`, contract addition)
  Covers: S-7 (bounds are configurable), §Configuration additions and changes
  Depends on: Task 1

- [ ] Write failing test `turbine-core config::tests::phase2_keys`: defaults equal the spec table; `scheduler.queue_timeout: 1.5s`, `2 s`, `10d` are rejected naming the key; `250ms`, `1h` parse; `max_batch_tokens: 32` with `max_running_requests: 64` is rejected naming `scheduler.max_batch_tokens`; `structured_output.max_schema_bytes: 2MiB` is rejected; `continuous_batching: false` makes `effective_max_running()` 1; `model.tool_call_parser: llama3_json` parses. Run: `cargo test -p turbine-core config::tests::phase2_keys` — expect FAIL
- [ ] Implement `HumanDuration` (serde from string, integer part `u64`, units exact-case) and the validation rules with `invalid(key, reason)`.
- [ ] Run: `cargo test -p turbine-core config::tests` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-core): phase 2 scheduler, kv, timeout and structured-output keys`

## Task 3: `turbine-kv` block pool and block tables

Files: `crates/turbine-kv/Cargo.toml` (new crate, `unsafe_code = "forbid"`, deps core/observability/tensor/smallvec/serde/thiserror; dev-dep `proptest`), `crates/turbine-kv/src/lib.rs`, `crates/turbine-kv/src/pool.rs` (`BlockPool`, `BlockPoolConfig`, `PoolError`, test), `crates/turbine-kv/src/table.rs` (`BlockTable`, `blocks_for_tokens`), `crates/turbine-tensor/src/kv_view.rs` (new: `KvPoolView`), `Cargo.toml` (members, `proptest`)
Interfaces:

- `pub struct KvPoolView<'a> { pub storage: &'a DeviceBuffer, pub layout: KvLayout, pub num_blocks: u32, pub layer_stride_bytes: u64 }` in `turbine_tensor` (contract addition: `turbine-model` may not depend on `turbine-kv`; layer `l` occupies `[num_blocks, 2, block_tokens, kv_heads, head_dim]` BF16 at `l × layer_stride_bytes`)
- `pub struct BlockPoolConfig { pub layout: KvLayout, pub num_blocks: u32 }`
- `BlockPool::new(cfg: BlockPoolConfig, mem: Arc<dyn DeviceMemory>) -> Result<BlockPool, PoolError>` (one `DeviceBuffer` of `num_blocks × block_bytes`), `free_blocks`, `total_blocks`, `used_blocks(&self) -> u32`, `allocate(&mut self, n: u32) -> Result<SmallVec<[BlockId; 8]>, PoolError>`, `incref(&mut self, b: BlockId)`, `release(&mut self, blocks: &[BlockId])`, `fork(&mut self, table: &BlockTable) -> Result<(BlockTable, Option<(BlockId, BlockId)>), PoolError>` (shares full blocks by refcount; returns the partial-tail `(src, dst)` pair the engine copies with `copy_blocks` — contract addition to the return type), `view(&self) -> KvPoolView<'_>`
- `pub enum PoolError { Exhausted { requested: u32, available: u32 }, Memory(MemoryError) }`
- `pub struct BlockTable { pub blocks: SmallVec<[BlockId; 16]>, pub tokens: u32 }` with `fn blocks_needed(&self, extra_tokens: u32, block_tokens: u32) -> u32`; `pub fn blocks_for_tokens(tokens: u32, block_tokens: u32) -> u32`
  Covers: S-5 AC `pool::tests::blocks_conserved`; S-1 (crate boundary)
  Depends on: Task 1, Phase 1 plan Task 2

- [ ] Write failing test `turbine-kv pool::tests::blocks_conserved`: a `proptest` over random sequences of allocate / append tokens / free / fork on a 64-block pool backed by `HostMemory` asserts after every step that `used + free == total`, that every block's refcount equals the number of tables holding it, and that releasing a table returns exactly the blocks whose refcount reached 0. Run: `cargo test -p turbine-kv pool::tests::blocks_conserved` — expect FAIL
- [ ] Implement the pool (free list + refcount vector; `allocate` is all-or-nothing and never calls the device allocator after construction) and table.
- [ ] Run: `cargo test -p turbine-kv` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-kv): preallocated block pool and block tables`

## Task 4: KV metrics and the `/turbine/v1/kv` document

Files: `crates/turbine-kv/src/metrics.rs` (`KvMetrics`), `crates/turbine-kv/src/document.rs` (`KvDocument`, `KvTierDocument`)
Interfaces:

- `KvMetrics::register(reg: &MetricsRegistry) -> KvMetrics`, `fn record(&self, pool: &BlockPool)` — `turbine_kv_blocks{tier="l0",state="used"|"free"}` (C-4)
- `#[derive(Serialize)] pub struct KvDocument { pub tiers: Vec<KvTierDocument> }`, `pub struct KvTierDocument { tier: &'static str /* "l0" */, dtype: &'static str /* "bf16" */, block_tokens: u32, block_bytes: u64, blocks_total: u32, blocks_used: u32, blocks_free: u32 }`, `KvDocument::from_pool(pool: &BlockPool) -> KvDocument`
- Startup log line (INFO, `event="kv_pool"`): `block_bytes`, `num_blocks` (Llama-3.2-3B 16-token blocks: 1 835 008 B, 4 681 blocks in 8 GiB; OLMoE: 2 097 152 B, 4 096 blocks)
  Covers: S-5 (KV accounting), S-11 (KV document shape), S-14 (`turbine_kv_blocks`)
  Depends on: Task 3

- [ ] Write failing test `turbine-kv document::tests::document_and_metrics_agree`: a pool with Llama-3.2-3B layout and 8 GiB of blocks reports `blocks_total: 4681`, `block_bytes: 1835008`; after allocating 1024 blocks the document shows 1024 used / 3657 free, `tier: "l0"`, and `/metrics` shows `turbine_kv_blocks{tier="l0",state="used"} 1024`. Run: `cargo test -p turbine-kv document::tests::document_and_metrics_agree` — expect FAIL
- [ ] Implement both modules (no pool internals exposed beyond `used_blocks`/`free_blocks`/`total_blocks`).
- [ ] Run: `cargo test -p turbine-kv` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-kv): kv metrics and diagnostics document`

## Task 5: Scheduler request lifecycle and queues

Files: `crates/turbine-scheduler/Cargo.toml` (new crate, deps core/observability/kv/smallvec/serde/thiserror/tracing/prometheus-client only; dev-dep `proptest`), `crates/turbine-scheduler/src/lib.rs`, `crates/turbine-scheduler/src/request.rs` (`RequestState`, `CancelReason`, `PreemptReason`, `SchedRequest`, per-seq bookkeeping, test), `crates/turbine-scheduler/src/queue.rs` (waiting queue ordered by priority then arrival, preempted requests at the front)
Interfaces:

- `pub enum RequestState { Waiting, Prefilling, Decoding, Paused, Finished, Cancelled, Failed }` (serde lowercase), `fn can_transition(self, to: RequestState) -> bool`
- `pub enum CancelReason { ClientDisconnect, RequestTimeout, SlowClient, Shutdown, QueueTimeout }` (`QueueTimeout` introduced here rather than P3 because P2 drops timed-out waiters; `as_str` = metric labels)
- `pub enum PreemptReason { KvExhausted }`
- `pub struct SchedRequest { pub id: RequestId, pub seqs: SmallVec<[SeqId; 1]>, pub prompt_len: u32, pub max_new_tokens: u32, pub priority: Priority, pub estimate: ResourceEstimate, pub arrival: Duration, pub cancel: CancelFlag, pub constrained: bool }` (`constrained` is a contract addition feeding the snapshot's `constrained` count)
- `pub enum SchedError { IllegalTransition { from: RequestState, to: RequestState } }`
  Covers: S-2 AC `request::tests::lifecycle_transitions`
  Depends on: Tasks 1, 3

- [ ] Write failing test `turbine-scheduler request::tests::lifecycle_transitions`: every allowed edge (waiting→prefilling, prefilling→decoding, decoding↔paused, prefilling/decoding→waiting on preemption, any live state→finished/cancelled/failed) succeeds; `finished → decoding`, `cancelled → waiting` and `failed → prefilling` return `SchedError::IllegalTransition`; a 100-token prompt with `max_tokens: 60` and 16-token blocks estimates 10 blocks. Run: `cargo test -p turbine-scheduler request::tests::lifecycle_transitions` — expect FAIL
- [ ] Implement the lifecycle table and the queue (a `BTreeMap<(Priority, arrival, seq_no), RequestId>` plus a front deque for preempted requests).
- [ ] Run: `cargo test -p turbine-scheduler request::tests` — expect PASS; `cargo tree -p turbine-scheduler | grep -E 'turbine-(kernels|model)|hip|cuda'` — expect no output
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-scheduler): request lifecycle and waiting queue`

## Task 6: Per-iteration planning, preemption, cancellation and scheduler metrics

Files: `crates/turbine-scheduler/src/scheduler.rs` (`Scheduler`, `SchedulerParams`, `IterationLimits`, `IterationPlan`, `BatchItem`, `BatchKind`, `IterationOutcome`, `SubmitError`, `SchedulerSnapshot`, unit tests), `crates/turbine-scheduler/src/metrics.rs` (`SchedulerMetrics`)
Interfaces:

- `pub struct SchedulerParams { max_running_requests, max_batch_tokens, prefill_chunk_tokens, max_queued_requests: u32, chunked_prefill: bool, block_tokens: u32, free_watermark: f64, max_seq_len: u32, queue_timeout: Duration }` (`max_seq_len`, `queue_timeout` are contract additions)
- `Scheduler::new(p: SchedulerParams, clock: Arc<dyn Clock>) -> Scheduler`, `submit(&mut self, r: SchedRequest, pool_total_blocks: u32) -> Result<(), SubmitError>` (pool size argument is a contract addition for the capacity check), `cancel(&mut self, id: RequestId, reason: CancelReason)`, `plan(&mut self, pool: &mut BlockPool, limits: &IterationLimits) -> IterationPlan`, `complete(&mut self, pool: &mut BlockPool, outcome: IterationOutcome)`, `pause(&mut self, seq: SeqId)`, `resume(&mut self, seq: SeqId)`, `begin_shutdown(&mut self)`, `snapshot(&self) -> SchedulerSnapshot`, `is_idle(&self) -> bool`
- `#[derive(Clone, Copy, Default)] pub struct IterationLimits { batch_growth_limit: Option<u32>, shrink_only: bool, prefill_budget_fraction: f64, prefill_chunk_tokens: Option<u32>, admit_new: bool, start_new_prefills: bool }` (P2 passes `Default` = GREEN: 1.0, true, true)
- `pub struct IterationPlan { iteration: u64, items: Vec<BatchItem>, preempted: Vec<(SeqId, PreemptReason)>, dropped: Vec<(RequestId, CancelReason)>, forks: Vec<ForkOp> }`, `pub struct ForkOp { src: SeqId, dst: SeqId, copy: Option<(BlockId, BlockId)> }` (`forks` is a contract addition for `n > 1`)
- `pub struct BatchItem { seq: SeqId, kind: BatchKind, block_table: BlockTable }`, `pub enum BatchKind { Prefill { start: u32, len: u32 }, Decode }`
- `pub struct IterationOutcome { iteration: u64, finished: Vec<(SeqId, FinishReason)>, appended: Vec<(SeqId, u32)>, failed: Option<IterationFailure> }`
- `pub enum SubmitError { QueueFull, ContextLengthExceeded, ContextExceedsKvCapacity, PromptTooLong, ShuttingDown }`
- Rules exactly P2 §Scheduling rules: (1) drop cancelled/queue-timed-out, free blocks; (2) decode set = every decoding, unpaused seq; preempt lowest priority then most recently admitted until blocks suffice, never preempting a request whose full KV fits the empty pool into livelock; preempted requests return to the queue front and later re-prefill `[0, prompt+generated)`; (3) prefill budget = `max_batch_tokens − decodes`, continue prefilling oldest first with `min(remaining, chunk, budget)`, then admit by priority then arrival while running < max and the pool covers the first chunk plus the 1 % watermark; forks wait for blocks without re-prefilling; (4) nothing to do → the caller sleeps until a submission or cancellation
- `SchedulerMetrics::register(reg) -> SchedulerMetrics`: `turbine_requests_active{state}`, `turbine_requests_queued`, `turbine_admission_total{outcome,reason}` (reasons incl. `context_exceeds_kv_capacity`, C-2), `turbine_queue_wait_seconds`, `turbine_iteration_seconds`, `turbine_iteration_tokens{phase}`, `turbine_batch_requests`, `turbine_preemptions_total{reason}`, `turbine_requests_cancelled_total{reason}`; INFO logs `event` ∈ `reject`, `preempt`, `pause`, `cancel` with `request_id` and `reason`
- `SchedulerSnapshot` serialises the P2 §Data scheduler JSON
  Covers: S-3, S-4, S-6, S-7 (scheduler side), S-8, S-11 (snapshot), S-14 (scheduler metrics and reason logs); the named sim ACs are delivered in Task 7
  Depends on: Task 5

- [ ] Write failing test `turbine-scheduler scheduler::tests::decode_first_then_chunked_prefill`: with `max_batch_tokens: 64`, `prefill_chunk_tokens: 32`, two decoding requests and one waiting 100-token prompt, `plan` returns the two decodes then one 32-token prefill chunk and a 30-token chunk; submitting to a full queue returns `QueueFull`; a request needing more blocks than the pool returns `ContextExceedsKvCapacity`. Run: `cargo test -p turbine-scheduler scheduler::tests` — expect FAIL
- [ ] Implement `plan`/`complete`/`cancel`/`pause`/`resume` with the rules above; timestamps only from the injected `Clock`.
- [ ] Run: `cargo test -p turbine-scheduler` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-scheduler): continuous batching plan, preemption and metrics`

## Task 7: Deterministic simulator and scheduler invariants

Files: `crates/turbine-scheduler/src/sim/mod.rs` (`Simulation`, `SimReport`, tests), `crates/turbine-scheduler/src/sim/executor.rs` (`SimExecutor`, `CostModel`), `crates/turbine-scheduler/src/sim/arrivals.rs` (`ArrivalProcess`)
Interfaces:

- `pub struct CostModel { pub per_prefill_token_s: f64, pub per_decode_step_s: f64, pub per_seq_s: f64 }`, `pub struct SimExecutor { pub cost: CostModel }`
- `ArrivalProcess::poisson(rate: f64, seed: u64) -> ArrivalProcess` (ChaCha8; prompt/output lengths from a seeded mix), `ArrivalProcess::scripted(arrivals: Vec<SimArrival>) -> ArrivalProcess`
- `Simulation::new(params: SchedulerParams, pool: BlockPool, exec: SimExecutor, arrivals: ArrivalProcess) -> Simulation`, `run(&mut self, until: Duration) -> SimReport` (virtual time via `FakeClock`, no sleeps), `SimReport { iterations: Vec<IterationTrace>, rejected: Vec<(RequestId, SubmitError)>, completed: u32, max_waiting: u32, max_running: u32 }` (`Serialize`, compared byte-for-byte)
- Hooks for tests: `cancel_at(iteration, id, reason)`, `pause_at(iteration, seq)`
  Covers: S-3 AC `sim::tests::ts_section7_iteration_pattern`; S-3/S-12 AC `sim::tests::decode_never_starved`; S-4 AC `sim::tests::chunk_budget_respected`; S-6 AC `sim::tests::preemption_by_recompute`; S-8 AC `sim::tests::cancellation_frees_within_one_iteration`; S-12 AC `sim::tests::bounded_under_overload`
  Depends on: Task 6

- [ ] Write failing test `sim::tests::ts_section7_iteration_pattern`: replays the TS §7 example (A: 3-chunk prompt; B: 1-chunk prompt arriving after iteration 1; C and D decoding) and asserts the three iteration compositions exactly as TS §7 lists them. Run: `cargo test -p turbine-scheduler sim::tests::ts_section7_iteration_pattern` — expect FAIL
- [ ] Write failing test `sim::tests::decode_never_starved`: 1,000 Poisson arrivals (seed 42) of mixed prompt lengths; every decoding, unpaused seq gets a token in every iteration, and two runs serialise to identical bytes. Run: `cargo test -p turbine-scheduler sim::tests::decode_never_starved` — expect FAIL
- [ ] Write failing test `sim::tests::chunk_budget_respected`: no iteration exceeds `max_batch_tokens`, no chunk exceeds `prefill_chunk_tokens`, and with `chunked_prefill: false` a prompt longer than `max_batch_tokens` is rejected at submission with `PromptTooLong`. Run: `cargo test -p turbine-scheduler sim::tests::chunk_budget_respected` — expect FAIL
- [ ] Write failing test `sim::tests::preemption_by_recompute`: with a pool too small for all running requests the lowest-priority, most recently admitted request is preempted, re-queued at the front, re-prefills prompt + generated tokens, its appended-token count never repeats a position (no duplicate token), and a request whose full KV fits the empty pool completes. Run: `cargo test -p turbine-scheduler sim::tests::preemption_by_recompute` — expect FAIL
- [ ] Write failing test `sim::tests::cancellation_frees_within_one_iteration`: requests cancelled while waiting, between prefill chunks, decoding and paused have all their blocks free before the next `plan` returns. Run: `cargo test -p turbine-scheduler sim::tests::cancellation_frees_within_one_iteration` — expect FAIL
- [ ] Write failing test `sim::tests::bounded_under_overload`: arrivals at 10× the simulated service rate for 10,000 virtual seconds never exceed `max_queued_requests` waiting or `max_running_requests` running, and excess arrivals are rejected with `QueueFull`. Run: `cargo test -p turbine-scheduler sim::tests::bounded_under_overload` — expect FAIL
- [ ] Implement the simulator driving the real `Scheduler` and `BlockPool` (on `HostMemory` with zero-byte-per-block layout) against `SimExecutor` costs.
- [ ] Run: `cargo test -p turbine-scheduler sim::tests` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-scheduler): deterministic simulator and scheduling invariants`

## Task 8: Kernel ABI v2 on the Rust side and CPU reference ops

Files: `kernels/include/turbine_kernels.h` (v2 block of contract §9.3, `TURBINE_ABI_VERSION 2u`), `crates/turbine-kernels/src/lib.rs` (`TURBINE_KERNELS_ABI_VERSION = 2`), `crates/turbine-kernels/src/ops/mod.rs` (P2 kinds, configs, contexts, traits), `crates/turbine-kernels/src/ffi.rs` (`CtxInfo`, `AttentionPagedDesc`, `CopyBlocksDesc`, `MoeRouteDesc`, `MoeExpertsDesc`, new symbols), `crates/turbine-kernels/src/shim.rs` (v2 provider methods, `ContextInfo`), `crates/turbine-kernels/src/registry.rs` (new `OpConfig` variants and accessors), `crates/turbine-kernels/src/cpu/paged.rs`, `crates/turbine-kernels/src/cpu/moe.rs` (new), `crates/turbine-kernels/stub/stub_shim.c` (v2 symbols), `crates/turbine-kernels/tests/abi_header_neutral.rs` (version 2)
Interfaces:

- `OpKind` gains `AttentionPrefillPaged`, `AttentionDecodePaged`, `CopyBlocks`, `MoeRoute`, `MoeExperts` (and `ALL` lists them); `AttentionKind` gains `PrefillPaged`, `DecodePaged` (`block_tokens: Some(_)`)
- `pub struct PagedAttentionContext<'a> { cfg, q, k_new, v_new, out, kv_layer: TensorView<'a>, block_table, q_indptr, kv_lens: TensorView<'a>, max_q_len, max_kv_len, max_blocks_per_seq: u32, scale: f32 }`; `AttentionKernel::execute_paged(&self, ctx: &mut PagedAttentionContext<'_>) -> Result<(), KernelError>` (contract addition: second method on the same trait)
- `pub trait KvCopyKernel { fn supports(&self, cfg: &KvCopyConfig) -> bool; fn implementation(&self, cfg: &KvCopyConfig) -> String; fn execute(&self, ctx: &mut KvCopyContext<'_>) -> Result<(), KernelError>; }` with `KvCopyContext { pool: DeviceSlice<'a>, layer_stride_bytes, block_bytes: u64, num_layers: u32, pairs: &'a [(BlockId, BlockId)] }`
- `pub trait MoeKernel { fn supports_route(&self, cfg: &MoeRouteConfig) -> bool; fn supports_experts(&self, cfg: &MoeExpertsConfig) -> bool; fn implementation_route/experts(…) -> String; fn route(&self, ctx: &mut MoeRouteContext<'_>) -> Result<(), KernelError>; fn experts(&self, ctx: &mut MoeExpertsContext<'_>) -> Result<(), KernelError>; }` — route: softmax F32, top-k with ties to the lower id, `renormalize` flag (OLMoE 0), `sorted_rows` grouped by expert, `expert_offsets`; experts: gate/up/down per selected expert, SiLU·up, weighted accumulate into `out` in fixed order
- `KernelProvider` gains `fn kv_copy(&self) -> Option<&dyn KvCopyKernel>`, `fn moe(&self) -> Option<&dyn MoeKernel>`
- `ShimContext::info(&self) -> ContextInfo` (`ContextInfo { workspace_bytes: u64, compute_capability: Option<(i32, i32)>, device_arch: String }` from `turbine_ctx_get_info`)
  Covers: S-5 (paged attention ops, `copy_blocks`), S-16 (`moe_route`, `moe_experts` with CPU reference)
  Depends on: Phase 1 plan Tasks 3–6

- [ ] Write failing test `turbine-kernels cpu::tests::paged_attention_equals_contiguous_and_moe_route_ties`: a 3-sequence ragged batch (q lengths 5, 1, 3; kv lengths 20, 9, 3) over a shuffled 16-token block table gives the same outputs as the contiguous `attention` op per sequence; `moe_route` on logits `[1,1,0,…]` with top-2 selects experts 0 and 1 in that order with unrenormalised softmax weights; `copy_blocks` duplicates block 3 into block 7 on every layer. Run: `cargo test -p turbine-kernels cpu::tests` — expect FAIL
- [ ] Implement the header, descriptors, provider methods and CPU reference ops; `abi_header_neutral` now expects `TURBINE_ABI_VERSION 2u` and the new trios; the stub libraries export the v2 symbols.
- [ ] Run: `cargo test -p turbine-kernels` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-kernels): kernel ABI v2 with paged attention, block copy and MoE ops`

## Task 9: OLMoE config, weight slots and tiny OLMoE checkpoint

Files: `crates/turbine-model/src/config.rs` (`Architecture::Olmoe`, `MoeConfig`, `moe`, `qk_norm`), `crates/turbine-model/src/loader.rs` (`olmoe_slots`), `crates/turbine-model/src/testing/tiny.rs` (`write_tiny_olmoe`), `crates/turbine-model/tests/fixtures/olmoe-1b-7b-0125-instruct/{config.json,generation_config.json}`
Interfaces:

- `Architecture::Olmoe` (`"OlmoeForCausalLM"`); `ModelArchConfig` gains `pub moe: Option<MoeConfig>`, `pub qk_norm: bool`; `pub struct MoeConfig { num_experts: u32, experts_per_token: u32, expert_intermediate: u32, norm_topk_prob: bool }`
- `pub fn olmoe_slots(cfg: &ModelArchConfig) -> Vec<WeightSlot>` (per layer `self_attn.{q,k,v,o}_proj`, `self_attn.q_norm.weight` [heads·hd], `self_attn.k_norm.weight` [kv·hd], `mlp.gate.weight` [experts, hidden], `mlp.experts.<e>.{gate,up}_proj` [expert_inter, hidden], `mlp.experts.<e>.down_proj` [hidden, expert_inter], norms, `lm_head.weight`)
- `pub fn write_tiny_olmoe(dir: &Path, seed: u64) -> TinySpec` (2 layers, hidden 64, 4/4 heads, head_dim 16, 8 experts, top-2, expert width 32, `norm_topk_prob: false`, theta 10000, untied, the tiny tokenizer, a template without `tools`; loads in transformers `OlmoeForCausalLM`)
- Allowlist message now lists `LlamaForCausalLM, OlmoeForCausalLM`
  Covers: S-16 AC `config::tests::parses_olmoe_config`
  Depends on: Phase 1 plan Tasks 7, 9

- [ ] Fetch the fixture (ungated): `curl -fsSL -o crates/turbine-model/tests/fixtures/olmoe-1b-7b-0125-instruct/config.json https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct/resolve/main/config.json` and the same for `generation_config.json`, then replace `main` by the commit `sha` returned by `curl -s https://huggingface.co/api/models/allenai/OLMoE-1B-7B-0125-Instruct` in the command recorded in the commit message.
- [ ] Write failing test `turbine-model config::tests::parses_olmoe_config`: the fixture gives 16 layers, hidden 2048, 16/16 heads, head_dim 128, 64 experts, top-8, expert width 1024, `norm_topk_prob: false`, rope theta 10000, vocab 50304, untied embeddings, `qk_norm: true`. Run: `cargo test -p turbine-model config::tests::parses_olmoe_config` — expect FAIL
- [ ] Implement parsing (`num_experts`, `num_experts_per_tok`, `intermediate_size` as expert width, `norm_topk_prob`), slots and the tiny writer.
- [ ] Run: `cargo test -p turbine-model config::tests` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-model): OLMoE config, weight slots and tiny checkpoint`

## Task 10: Ragged paged batches in the Llama executor

Files: `crates/turbine-model/src/executor/mod.rs` (P2 `BatchInput`, `SeqSlice`, `copy_blocks` on the trait), `crates/turbine-model/src/executor/llama.rs` (ragged forward over the pool), `crates/turbine-model/src/executor/batch.rs` (new: host-side packing of positions, `q_indptr`, `kv_lens`, block tables into device buffers)
Interfaces:

- `pub struct BatchInput<'a> { pub tokens: &'a [u32], pub positions: &'a [u32], pub seqs: &'a [SeqSlice<'a>], pub kv: &'a KvPoolView<'a> }`
- `pub struct SeqSlice<'a> { pub seq: SeqId, pub q_start: u32, pub q_len: u32, pub kv_len: u32, pub block_table: &'a [BlockId] }` (`kv_len` after this step's append)
- `ModelExecutor::forward(&mut self, batch: &BatchInput<'_>) -> Result<Logits, ModelError>` returns one FP32 row per sequence (last position), one device-to-host copy per iteration; `fn copy_blocks(&mut self, src: &[BlockId], dst: &[BlockId]) -> Result<(), ModelError>`
- `LlamaExecutor::new(cfg, weights, registry, mem, max_batch_tokens: u32, max_seqs: u32) -> Result<LlamaExecutor, ModelError>` (no contiguous cache; KV lives in the pool); `requirements` now lists `attention_prefill_paged`, `attention_decode_paged` (with `block_tokens`) and `copy_blocks`; `workspace_bytes(cfg, max_batch_tokens)`
  Covers: S-9 (batched execution, one D2H copy), S-5 (executor on the paged pool); the named paged/chunked ACs are delivered in Task 11
  Depends on: Tasks 3, 8

- [ ] Write failing test `turbine-model --test tiny_model paged_llama_single_sequence`: on the tiny Llama checkpoint a 40-token prompt prefilled into the pool then decoded 10 steps gives the same logits (≤ 1e-4) as the Phase 1 contiguous run recorded by driving the same executor with one sequence per batch. Run: `cargo test -p turbine-model --test tiny_model paged_llama_single_sequence` — expect FAIL
- [ ] Implement the ragged forward: all tokens of all sequences go through the GEMMs as one `[total_tokens, hidden]` batch; attention uses the paged op per layer with `kv_layer` = layer `l` of `KvPoolView`; the final RMSNorm and LM head run only on each sequence's last row.
- [ ] Run: `cargo test -p turbine-model --test tiny_model` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-model): ragged paged batches for the Llama executor`

## Task 11: OLMoE executor and batching equivalence tests

Files: `crates/turbine-model/src/executor/olmoe.rs` (new `OlmoeExecutor`), `crates/turbine-model/src/executor/mod.rs` (re-export, `build_executor(cfg, …) -> Box<dyn ModelExecutor>`), `crates/turbine-model/tests/tiny_model.rs` (`olmoe_cpu_forward_matches_naive`, `chunked_prefill_matches_unchunked`, `paged_matches_contiguous`)
Interfaces:

- `OlmoeExecutor::requirements(cfg: &ModelArchConfig) -> Vec<OpRequirement>` (adds `rmsnorm` over `heads·hd` and `kv·hd` for Q/K norm, `moe_route`, `moe_experts`), `OlmoeExecutor::workspace_bytes(cfg, max_batch_tokens) -> u64` (includes MoE permutation buffers), `OlmoeExecutor::new(cfg, weights, registry, mem, max_batch_tokens, max_seqs) -> Result<OlmoeExecutor, ModelError>`
- Layer: RMSNorm → Q/K/V → RMSNorm over the full Q and K projections → RoPE (standard, theta 10000) → paged attention → O → residual → RMSNorm → router GEMM (F32 logits) → `moe_route` (softmax, top-8, no renormalisation when `norm_topk_prob` is false) → `moe_experts` → residual
- `pub fn build_executor(cfg: &ModelArchConfig, weights: LoadedWeights, registry: Arc<KernelRegistry>, mem: Arc<dyn DeviceMemory>, max_batch_tokens: u32, max_seqs: u32) -> Result<Box<dyn ModelExecutor>, ModelError>` (contract addition: dispatch on `Architecture`)
  Covers: S-16 AC `tiny_model olmoe_cpu_forward_matches_naive`; S-4/S-9 AC `tiny_model chunked_prefill_matches_unchunked`; S-5/S-9 AC `tiny_model paged_matches_contiguous`
  Depends on: Tasks 9, 10

- [ ] Write failing test `tiny_model olmoe_cpu_forward_matches_naive`: tiny OLMoE (8 experts, top-2) CPU logits equal an independent naive f32 implementation in the test within 1e-4, including Q/K norm and unrenormalised routing weights. Run: `cargo test -p turbine-model --test tiny_model olmoe_cpu_forward_matches_naive` — expect FAIL
- [ ] Write failing test `tiny_model chunked_prefill_matches_unchunked`: for both tiny checkpoints, a 300-token prompt prefilled in chunks of 64 gives last-position logits within 1e-4 of a single-chunk prefill. Run: `cargo test -p turbine-model --test tiny_model chunked_prefill_matches_unchunked` — expect FAIL
- [ ] Write failing test `tiny_model paged_matches_contiguous`: for both tiny checkpoints, batched paged decoding of 4 sequences with prompt lengths 3, 17, 40 and 65 gives logits within 1e-4 of four single-sequence runs. Run: `cargo test -p turbine-model --test tiny_model paged_matches_contiguous` — expect FAIL
- [ ] Implement `OlmoeExecutor` and `build_executor`; expert weights are uploaded as stacked `[experts, inter, hidden]` tensors to match the `moe_experts` descriptor.
- [ ] Run: `cargo test -p turbine-model --test tiny_model` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-model): OLMoE executor and batching equivalence tests`

## Task 12: Sampler penalties, token masks and llguidance structured output

Files: `crates/turbine-model/src/sampler.rs` (penalties, bias, `min_tokens`, mask, `SamplerState`), `crates/turbine-model/src/structured.rs` (new), `crates/turbine-model/src/lib.rs` (`ModelError::Constraint`), `crates/turbine-model/Cargo.toml` (`llguidance = "=1.8.0"`, `toktrie_hf_tokenizers = "=1.8.0"` — MIT, recorded in a comment; dev-dep `jsonschema`), `Cargo.toml` (workspace pins)
Interfaces:

- `Sampler::new(params: &SamplingParams, prompt_tokens: &[u32], eos_token_ids: &[u32]) -> Sampler` (signature extended for repetition penalty and `min_tokens`), `sample(&mut self, logits: &mut [f32], mask: Option<&TokenMask>) -> SampledToken`, `observe(&mut self, token: u32)`, `state(&self) -> SamplerState`, `from_state(params, state: SamplerState) -> Sampler`; order: logit_bias → presence/frequency (OpenAI) and repetition (HF) penalties → EOS suppressed while generated < `min_tokens` → mask (the mask wins over `logit_bias`) → greedy/sample; reported logprobs stay the raw log-softmax
- `#[derive(Serialize, Deserialize)] pub struct SamplerState { pub seed: [u8; 32], pub word_pos: u128, pub generated: Vec<u32> }` (ChaCha8 seed and `get_word_pos`)
- `pub struct TokenMask` (bitset): `new_all(vocab)`, `new_none(vocab)`, `allow(id)`, `is_allowed(id) -> bool`, `apply(&self, logits: &mut [f32])` (disallowed → −∞)
- `pub trait TokenMatcher: Send { fn allowed(&mut self, mask: &mut TokenMask) -> Result<(), ModelError>; fn commit(&mut self, token: u32) -> Result<(), ModelError>; fn accepts_eos(&self) -> bool; }`
- `pub struct GrammarLimits { pub max_schema_bytes: usize }`; `GrammarCompiler::new(tokenizer: &Tokenizer, eos_token_ids: &[u32]) -> Result<GrammarCompiler, ModelError>` (token trie built once with `toktrie_hf_tokenizers::ByteTokenizerEnv` from the same `tokenizers` 0.21 crate Phase 1 pinned); `compile(&self, spec: &ConstraintSpec, limits: &GrammarLimits) -> Result<Box<dyn TokenMatcher>, ModelError>` — JSON with bounded natural whitespace (llguidance JSON options `,`/`:` separators and `whitespace_pattern` `[\x20\x0A\x0D\x09]{1,16}` = `JSON_MAX_WHITESPACE`, amended 2026-09-26); unsupported keyword or size over the bound → `ModelError::Constraint` naming it
- `ModelError::Constraint(String)` (`"constraint: {0}"`)
- Metrics on `ModelMetrics`: `turbine_grammar_compile_seconds{kind}`, `turbine_token_mask_seconds`
  Covers: S-17 AC `structured::tests::mask_applied_before_sampling`; S-10 (penalties, bias, min_tokens, stop_token_ids in sampling); S-6 (sampler state kept across preemption)
  Depends on: Task 1, Phase 1 plan Task 14

- [ ] Write failing test `turbine-model structured::tests::mask_applied_before_sampling`: a mock matcher allowing only ids {5, 9} yields only 5 or 9 under greedy and 200 seeded samples even with `logit_bias` of +100 on id 3, and EOS id 9 is masked until `accepts_eos()` turns true. Run: `cargo test -p turbine-model structured::tests::mask_applied_before_sampling` — expect FAIL
- [ ] Write failing test `turbine-model structured::tests::json_schema_matcher_on_llama_tokenizer`: compiling `{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}` with the committed Llama tokenizer and forcing greedy choices from seeded random logits produces text that parses and validates with `jsonschema` and has no whitespace run outside strings longer than `JSON_MAX_WHITESPACE`; `structured::tests::json_whitespace_natural_and_bounded` accepts `{"name": "John", …}`, compact and pretty-printed objects and refuses a gap of `JSON_MAX_WHITESPACE + 1` (amended 2026-09-26); a schema using an unsupported keyword errors naming it. Run: `cargo test -p turbine-model structured::tests::json_schema_matcher_on_llama_tokenizer` — expect FAIL
- [ ] Implement the sampler changes, `TokenMask`, `LlguidanceMatcher` (wrapping `llguidance::Matcher`: `compute_mask`, `consume_token`, `is_accepting`) and `GrammarCompiler`.
- [ ] Run: `cargo test -p turbine-model` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-model): sampler penalties, token masks and llguidance structured output`

## Task 13: Llama-3 tool-call parser, tool grammar and tool rendering

Files: `crates/turbine-model/src/tools.rs` (new), `crates/turbine-model/src/chat_template.rs` (test only), `crates/turbine-model/tests/fixtures/llama-3.2-3b-instruct/expected_renders.json` (regenerated with the tools case by `scripts/golden/render_fixture.py`), `scripts/golden/render_fixture.py` (adds the tools conversation)
Interfaces:

- `pub enum ToolParse { Calls(Vec<ToolCallOut>), Content(String) }`; `pub trait ToolCallParser: Send + Sync { fn parse(&self, text: &str) -> ToolParse; }`; `pub struct Llama3JsonParser` (optional `<|python_tag|>`, one or more `{"name":…,"parameters":{…}}` separated by `;`; `arguments` = compact JSON of `parameters`; ids `call_` + 24 alphanumerics)
- `pub enum ToolChoice { None, Auto, Required, Named(String) }`
- `pub fn tool_call_grammar(tools: &[serde_json::Value], choice: &ToolChoice, parallel: bool) -> Result<ConstraintSpec, ModelError>` (llguidance Lark grammar: optional `<|python_tag|>`, `{"name": <enum of allowed names>, "parameters": <that tool's JSON schema>}`, `;`-separated repetition when `parallel`; parameters and the `;` separator take the same bounded whitespace as S-17 JSON, amended 2026-09-26; `auto` → `start: text | calls` with `text: /[\x20\x0A\x0D\x09]{0,16}[^{\x20\x0A\x0D\x09](?s:.)*/` and `calls` the `required` grammar, amended 2026-09-26; `none` → `ModelError::Constraint`)
- `pub fn new_call_id(rng: &mut impl RngCore) -> String`
- Metric `turbine_tool_calls_total{parser="llama3_json",outcome="parsed"|"parse_failed"}` on `ModelMetrics`
  Covers: S-18 AC `tools::tests::llama3_json_parser`, `tools::tests::auto_grammar_on_llama_tokenizer`, `chat_template::tests::renders_llama_tools`
  Depends on: Task 12, Phase 1 plan Task 12

- [ ] Write failing test `turbine-model tools::tests::llama3_json_parser`: one call, two `;`-separated calls, a call preceded by `<|python_tag|>` parse into `ToolCallOut`s with `arguments` equal to the compact `parameters` string; plain text and `{"a": 1}` return `Content` unchanged. Run: `cargo test -p turbine-model tools::tests::llama3_json_parser` — expect FAIL
- [ ] Write failing test `turbine-model tools::tests::auto_grammar_on_llama_tokenizer`: on the committed Llama-3.2 tokenizer the `auto` grammar accepts plain text (also with `{` later in the text), valid calls with and without `<|python_tag|>`, and refuses text starting with `{` that is not a call, an unlisted function name and a string where the schema wants a number. Run: `cargo test -p turbine-model tools::tests::auto_grammar_on_llama_tokenizer` — expect FAIL
- [ ] Write failing test `turbine-model chat_template::tests::renders_llama_tools`: the Llama-3.2 template with a `get_weather` tool (required `location` string, optional `unit` enum), a user message, an assistant `tool_calls` message with arguments `{"location":"Paris"}` and a `tool` message `{"temperature": 21}`, `date_string: "26 Jul 2024"`, renders exactly the transformers string stored under `tools` in `expected_renders.json`. Run: `cargo test -p turbine-model chat_template::tests::renders_llama_tools` — expect FAIL
- [ ] Regenerate the fixture: `uv run --with 'transformers==4.57.1' --with jinja2 python3 scripts/golden/render_fixture.py crates/turbine-model/tests/fixtures/llama-3.2-3b-instruct`
- [ ] Implement the parser (find the first `{` after an optional python tag; `serde_json::Deserializer::into_iter` over `;`-separated objects; every object must have exactly `name` string and `parameters` object) and the grammar builder.
- [ ] Run: `cargo test -p turbine-model tools::tests chat_template::tests` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-model): llama3_json tool-call parser and tool grammar`

## Task 14: OpenAI Phase 2 fields, tool calls and new errors in `turbine-api`

Files: `crates/turbine-api/src/openai/request.rs` (P2 fields and validation), `crates/turbine-api/src/openai/response.rs` (choices by index, `message.tool_calls`), `crates/turbine-api/src/openai/stream.rs` (per-choice chunks, `delta.tool_calls`, timeout error events), `crates/turbine-api/src/error.rs` (new constructors), `crates/turbine-api/src/backend.rs` (`NotReadyReason::ShuttingDown`), `crates/turbine-api/tests/openai.rs`
Interfaces:

- Accepted now: `n` ≥ 1, `presence_penalty`, `frequency_penalty` (−2..=2), `repetition_penalty` (> 0), `logit_bias` (−100..=100, ids < vocab checked by the backend), `min_tokens`, `stop_token_ids`, `priority`, `echo: true` (completions), `user`, `response_format` (`text`, `json_object`, `json_schema{name,schema,strict}`), `tools`, `tool_choice`, `parallel_tool_calls`, assistant `tool_calls`, `tool` role with `tool_call_id`
- `response_format` other than `text` with `tool_choice` other than `none` → 400 `unsupported_parameter`
- `ApiError::{queue_full(), queue_timeout(), context_exceeds_kv_capacity(msg), invalid_json_schema(msg), tools_not_supported(model), unknown_tool(name), shutting_down(), request_timeout()}` per contract §14.3 (`queue_full`: 429 `retry-after: 1`; `request_timeout`: 504 type `timeout`)
- Streaming ends for server-side cancellation (`request_timeout`, `slow_client`, `shutting_down`) with an error event then `data: [DONE]` (C-3); `ToolCalls` → one `delta.tool_calls` entry per call with `index`, `id`, `type: "function"`, `function.name`, complete `function.arguments`, and `finish_reason: "tool_calls"`
- `NotReadyReason::ShuttingDown` (`"shutting_down"`)
  Covers: S-10, S-13 (readiness reason), S-17/S-18 (API surface); end-to-end assertions in Tasks 15–17
  Depends on: Task 1, Phase 1 plan Tasks 15–16

- [ ] Write failing test `turbine-api --test openai phase2_shapes`: a scripted backend emitting two choices renders `choices` with indices 0 and 1 (streaming and not); a `ToolCalls` event renders `message.tool_calls[0].function.arguments == "{\"location\":\"Paris\"}"` and `finish_reason: "tool_calls"`; `response_format: {"type":"json_object"}` with `tool_choice: "required"` is 400 `unsupported_parameter`; a `request_timeout` error mid-stream is followed by `[DONE]`. Run: `cargo test -p turbine-api --test openai phase2_shapes` — expect FAIL
- [ ] Implement the fields, validation and rendering.
- [ ] Run: `cargo test -p turbine-api` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-api): phase 2 OpenAI fields, tool calls and timeout errors`

## Task 15: Engine thread with continuous batching, bounded channels and diagnostics

Files: `crates/turbine-server/src/engine/mod.rs` (new: `EngineHandle`, `EngineCommand`, thread spawn), `crates/turbine-server/src/engine/loop.rs` (iteration loop), `crates/turbine-server/src/engine/requests.rs` (per-request host state: token history, sampler, detokenizer, stop state, output sender), `crates/turbine-server/src/generation.rs` (removed; `ModelBackend::submit` now forwards to the engine), `crates/turbine-server/src/model.rs` (pool sizing, startup budget with `kv.gpu.max_bytes` and `max_batch_tokens` workspace), `crates/turbine-server/src/startup.rs` (diagnostics `scheduler`/`kv` return documents), `crates/turbine-server/Cargo.toml` (add turbine-kv, turbine-scheduler), `crates/turbine-server/tests/tiny_server.rs` (rewrite `single_slot_and_cancel`, add P2 tests)
Interfaces:

- `pub struct EngineHandle { pub submit_tx: tokio::sync::mpsc::Sender<EngineCommand> }` (capacity `scheduler.max_queued_requests`), `pub enum EngineCommand { Submit(Box<GenerationRequest>, mpsc::Sender<GenerationEvent> /* 256 */), Cancel { id: RequestId, reason: CancelReason }, Shutdown }`
- Loop per iteration: drain commands (non-blocking; block on `recv` only when `Scheduler::is_idle`) → `plan` → execute `forks` via `ModelExecutor::copy_blocks` → build `BatchInput` from items and `BlockPool::view()` → one `forward` → per-seq sample (masks from matchers) → emit events with `try_send`; a full channel pauses the seq (`Scheduler::pause`, `turbine_stream_paused_total`) and resumes when drained; a closed channel cancels with `ClientDisconnect` → `complete` → update metrics
- Submission checks in `ModelBackend::submit` before queueing (P2 §Scheduling rules): `context_length_exceeded`, `context_exceeds_kv_capacity`, `queue_full` (429, retry-after 1), `shutting_down` (503); `kv.gpu.enabled: false` → startup exit 1
- 3 consecutive failed iterations → every request in them gets `internal_error`, `/ready` 503 `device_error`, exit 1 (C-25); an engine panic is caught (`std::panic::catch_unwind` around the loop body), logged with the iteration's request ids, all in-flight requests get `internal_error`, exit 1
- `/turbine/v1/scheduler` = `Scheduler::snapshot()`, `/turbine/v1/kv` = `KvDocument::from_pool`, `/turbine/v1/pressure` stays 501
  Covers: S-1 (engine thread in `turbine-server`), S-7 AC `tiny_server queue_full_429`; S-8 AC `tiny_server disconnect_releases_kv`; S-11 AC `tiny_server diagnostics_shapes`; S-3 (continuous batching end to end)
  Depends on: Tasks 2, 4, 7, 11, 14

- [ ] Rewrite `tiny_server single_slot_and_cancel` (contract §20.2): a second concurrent request is queued and completes after the first instead of getting 429; dropping the first client mid-stream frees its blocks and `turbine_requests_total{outcome="cancelled"} 1`. Run: `cargo test -p turbine-server --test tiny_server single_slot_and_cancel` — expect FAIL
- [ ] Write failing test `tiny_server queue_full_429`: with `scheduler.max_queued_requests: 2`, `max_running_requests: 1`, 6 concurrent long requests give 3 completions and 3 × 429 `queue_full` with `retry-after`, and `turbine_admission_total{outcome="rejected",reason="queue_full"} 3`. Run: `cargo test -p turbine-server --test tiny_server queue_full_429` — expect FAIL
- [ ] Write failing test `tiny_server disconnect_releases_kv`: 8 streams open, 4 clients dropped; within 1 s `/turbine/v1/kv` `blocks_used` equals the remaining requests' usage, 0 after all finish, and `turbine_requests_cancelled_total{reason="client_disconnect"} 4`. Run: `cargo test -p turbine-server --test tiny_server disconnect_releases_kv` — expect FAIL
- [ ] Write failing test `tiny_server diagnostics_shapes`: under load `/turbine/v1/scheduler` has every P2 §Data key with counts matching `turbine_requests_active`/`turbine_requests_queued`, `/turbine/v1/kv` has the P2 tier object with `tier: "l0"`, and `/turbine/v1/pressure` is 501. Run: `cargo test -p turbine-server --test tiny_server diagnostics_shapes` — expect FAIL
- [ ] Implement the engine thread, submission checks and diagnostics as listed.
- [ ] Run: `cargo test -p turbine-server` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-server): engine thread with continuous batching and paged KV`

## Task 16: Timeouts, slow clients and graceful shutdown

Files: `crates/turbine-server/src/engine/deadlines.rs` (new: request deadline, queue timeout, slow-client timer driven by `SystemClock`), `crates/turbine-server/src/engine/loop.rs` (cancellation reasons), `crates/turbine-server/src/startup.rs` (SIGINT/SIGTERM: readiness `shutting_down`, grace, cancel, exit 0), `crates/turbine-server/tests/tiny_server.rs`, `crates/turbine-server/tests/server_cli.rs`
Interfaces:

- `server.request_timeout` → non-streaming 504 `request_timeout`, streaming error event `request_timeout` then `[DONE]` (C-3), `CancelReason::RequestTimeout`
- `scheduler.queue_timeout` → 503 `queue_timeout` (`CancelReason::QueueTimeout`, `turbine_admission_total{outcome="rejected",reason="queue_timeout"}`)
- Paused longer than `server.slow_client_timeout` → cancelled with `slow_client` (stream error event then `[DONE]`), blocks freed; resuming reads before the timeout resumes decoding
- Shutdown: new requests 503 `shutting_down`, `/ready` 503 `shutting_down`, running requests continue up to `server.shutdown_grace`, then `EngineCommand::Shutdown` cancels the rest with `shutdown` and the process exits 0
  Covers: S-7 AC `tiny_server slow_client_paused_then_cancelled`, `tiny_server request_and_queue_timeouts`; S-13 AC `server_cli sigterm_drains_then_cancels`; S-8 (timeout and shutdown cancellation)
  Depends on: Task 15

- [ ] Write failing test `tiny_server slow_client_paused_then_cancelled`: with `slow_client_timeout: 1s` a client that stops reading appears as `paused` in `/turbine/v1/scheduler` while another stream keeps progressing, and after 1 s it is cancelled with reason `slow_client` and its blocks are free. Run: `cargo test -p turbine-server --test tiny_server slow_client_paused_then_cancelled` — expect FAIL
- [ ] Write failing test `tiny_server request_and_queue_timeouts`: `request_timeout: 1s` ends a long stream with an error event `request_timeout` followed by `[DONE]` and a non-streaming one with 504; with `max_running_requests: 1` and `queue_timeout: 1s` a waiting request gets 503 `queue_timeout`. Run: `cargo test -p turbine-server --test tiny_server request_and_queue_timeouts` — expect FAIL
- [ ] Write failing test `server_cli sigterm_drains_then_cancels`: with `shutdown_grace: 2s`, one short and one very long generation, SIGTERM → new requests 503 `shutting_down`, the short one completes, the long one ends with error code `shutting_down`, exit 0 within 3 s. Run: `cargo test -p turbine-server --test server_cli sigterm_drains_then_cancels` — expect FAIL
- [ ] Implement deadlines (checked at every iteration boundary and while queued) and the shutdown sequence.
- [ ] Run: `cargo test -p turbine-server` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-server): request/queue timeouts, slow clients and graceful shutdown`

## Task 17: Phase 2 OpenAI fields, preemption, structured output and tools in the engine

Files: `crates/turbine-server/src/engine/requests.rs` (n > 1 forks, penalties, `min_tokens`, `stop_token_ids`, `echo`, priority, matchers, tool parsing), `crates/turbine-server/src/engine/grammar.rs` (new: compile on `spawn_blocking` behind `Semaphore::new(4)` with `tokio::time::timeout(compile_timeout)`), `crates/turbine-server/src/model.rs` (`GrammarCompiler` built once at startup, `tool_call_parser` default resolution via `ChatTemplate::renders_tools`), `crates/turbine-server/tests/tiny_server.rs`
Interfaces:

- `n > 1`: one prefill, then `ForkOp`s copy the tail block; each choice has its own sampler (seed + choice index) and matcher; `turbine_tokens_total{kind="prompt"}` grows once
- Preemption keeps `SamplerState`, detokenizer offsets and the matcher on the host; the re-prefill of prompt + generated tokens emits no events
- `response_format` / constrained tool grammar compiled before queueing: failure, unsupported keyword, over `structured_output.max_schema_bytes` or over `compile_timeout` → 400 `invalid_json_schema` (`turbine_admission_total{reason="invalid_json_schema"}`); per-step matcher error → that request ends with `internal_error`, log reason `constraint_error`
- Tools: parser `none` → 400 `tools_not_supported`; named function absent → 400 `unknown_tool`; `required`/named/`auto` → `tool_call_grammar` (amended 2026-09-26: `auto` is constrained to text or schema-valid calls); `auto` still buffers output starting with `<|python_tag|>` or `{` and parses it at finish (`tool_call_parse_failed` log on failure, raw text returned); `tool_choice: "none"` renders without tools
  Covers: S-10 AC `tiny_server openai_phase2_fields`; S-6/S-9 AC `tiny_server preempted_output_unchanged`; S-17 AC `tiny_server response_format_json_schema`; S-18 AC `tiny_server tool_choice_modes`; S-14 AC `tiny_server phase2_metrics_and_reasons`
  Depends on: Tasks 12, 13, 16

- [ ] Write failing test `tiny_server openai_phase2_fields`: `n: 3` returns choices 0–2 (streaming and not) with the prompt counted once; penalties and `logit_bias` change greedy output exactly as the host sampler predicts on captured logits; `min_tokens: 5` suppresses EOS; `stop_token_ids: [<id>]` stops; `echo: true` prepends the prompt; of two queued requests the one with `priority: -1` starts first. Run: `cargo test -p turbine-server --test tiny_server openai_phase2_fields` — expect FAIL
- [ ] Write failing test `tiny_server preempted_output_unchanged`: with `kv.gpu.max_bytes` sized for 3 sequences, 8 seeded concurrent requests (two with a `json_schema` response format) produce the same tokens as the same requests run one at a time, and `turbine_preemptions_total{reason="kv_exhausted"}` is > 0. Run: `cargo test -p turbine-server --test tiny_server preempted_output_unchanged` — expect FAIL
- [ ] Write failing test `tiny_server response_format_json_schema`: 20 seeded `temperature: 1.0` requests with a schema of a boolean, an integer enum and a string enum (no additional properties) and 5 `json_object` requests all finish with `stop`, parse, validate with `jsonschema` and keep whitespace outside strings within `JSON_MAX_WHITESPACE` per gap; a schema with an unsupported keyword and one over `max_schema_bytes` get 400 `invalid_json_schema`. Run: `cargo test -p turbine-server --test tiny_server response_format_json_schema` — expect FAIL
- [ ] Write failing test `tiny_server tool_choice_modes`: with `model.tool_call_parser: llama3_json` a named `tool_choice` returns exactly one call to that function whose arguments validate (streaming and not, ids matching `call_[A-Za-z0-9]{24}`, `finish_reason: "tool_calls"`); `required` + `parallel_tool_calls: false` returns exactly one call; `auto` + `parallel_tool_calls: false` returns content not starting with `{` or exactly one valid call, and a valid call when `logit_bias` favours `{`; `none` returns content only; an unknown named function is 400; a tiny checkpoint written with `template_with_tools: false` answers `tools` with 400 `tools_not_supported`. Run: `cargo test -p turbine-server --test tiny_server tool_choice_modes` — expect FAIL
- [ ] Write failing test `tiny_server phase2_metrics_and_reasons`: after a run that queues, rejects, preempts, cancels and serves a constrained request, every §Metrics (added) family is present with the expected counts, and the JSON log has a record with `reason` for each reject, preemption and cancellation. Run: `cargo test -p turbine-server --test tiny_server phase2_metrics_and_reasons` — expect FAIL
- [ ] Implement the request features as listed.
- [ ] Run: `cargo test -p turbine-server` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-server): n, penalties, preemption state, structured output and tool calls`

## Task 18: `libturbine_hip.so` ABI v2 — paged attention, block copy and MoE

Files: `kernels/rocm/src/context.cpp` (`turbine_ctx_get_info`, ABI 2), `kernels/rocm/src/paged_attention.cpp` (new), `kernels/rocm/src/paged_attention.hip` (new Turbine paged kernel), `kernels/rocm/src/copy_blocks.cpp` (new), `kernels/rocm/src/moe.cpp` + `kernels/rocm/src/moe.hip` (new), `kernels/rocm/CMakeLists.txt` (add the group-mode `pagedkv_prefill` instances to the generator call), `kernels/rocm/src/turbine_hip.hpp` + `kernels/rocm/src/bf16.hpp` (internal declarations, BF16 device helpers), `crates/turbine-kernels/tests/hip_ops.rs` (`paged_and_moe_ops`)
Interfaces:

- Paged attention: every call first writes `k_new`/`v_new` into their page slots with a Turbine append kernel (CK `fmha_fwd_appendkv` is batch mode with one `seqlen_knew` for all sequences, so it cannot take a ragged batch in one call); then, when `block_tokens % 128 == 0`, CK `fmha_fwd_pagedkv` in group mode for prefill and decode alike (`seqstart_q = q_indptr`, `seqlen_k = kv_lens`, `block_table_ptr`), K/V strides `stride = Hkv·128`, `nhead_stride = 128`, `batch_stride = 2·block_tokens·Hkv·128`, `v_ptr = k_ptr + block_tokens·Hkv·128`, bottom-right causal mask (`_impl` `ck_tile_fmha_pagedkv`); otherwise (the 16-token default) the Turbine HIP paged kernel (`_impl` `turbine_hip`) — CK requires a page size that is a multiple of 128 on this tag. `fmha_fwd_splitkv` (split-K decode with an `lse_acc`/`o_acc` workspace and a combine pass) is a decode-throughput optimisation left to a benchmark-driven follow-up
- `copy_blocks`: one `hipMemcpyAsync(…, hipMemcpyDeviceToDevice, stream)` per layer per pair (`_impl` `hip_memcpy_d2d`)
- `moe_route`: Turbine kernel (F32 softmax, top-k ties to the lower id, counting sort into `sorted_rows`/`expert_offsets`)
- `moe_experts`: `hipblaslt_ext::GroupedGemm` has no gfx1201 solutions in ROCm 7.14.1 (all installed kernels are non-grouped), so the shim gathers each expert's rows into the workspace and runs per-expert `hipblasLtMatmul` gate/up/down with the Turbine SiLU·up and a weighted scatter-add in fixed order (`_impl` `hipblaslt_per_expert`); the grouped path (`_impl` `hipblaslt_grouped`) is selected automatically if `GroupedGemm::algoGetHeuristic` returns a solution when the context is created. Intermediates use the caller's workspace when it is large enough, otherwise a context-owned scratch grown on demand
  Covers: S-5, S-16 (HIP paged-KV and MoE providers); the ignored `hip_ops paged_and_moe_ops` is written here and accepted by its lab run in Task 20
  Depends on: Task 8, Phase 1 plan Task 20

- [ ] Write the ignored test `turbine-kernels --test hip_ops paged_and_moe_ops`: `require_backend("hip")`; seeded ragged batches compare `attention_prefill_paged`, `attention_decode_paged`, `copy_blocks`, `moe_route` and `moe_experts` with the CPU reference (Phase 1 tolerances; `moe_route` selections identical), including a batch where every token selects the same 8 experts and one where an expert receives no token; prints each `_impl`. Run: `scripts/lab-test.sh novanas` — expect FAIL until the v2 library is built
- [ ] Implement the v2 shim sources and CMake additions.
- [ ] Run: `scripts/lab-test.sh novanas` — expect PASS with `test paged_and_moe_ops ... ok` and the log naming `hipblaslt_per_expert` and the paged `_impl` in use
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(kernels): hip ABI v2 paged attention, block copy and MoE ops`

## Task 19: `turbine-golden compare --concurrency`

Files: `benches/turbine-bench/src/bin/turbine-golden.rs` (flag), `benches/turbine-bench/src/golden/client.rs` (bounded concurrent replay), `benches/turbine-bench/tests/golden.rs`; amended 2026-09-26 (decision "Golden at concurrency 16"): `benches/turbine-bench/src/golden/{fixture,compare,mod}.rs` (batched bounds), `tests/golden/*/tolerance.json`, `crates/turbine-model/tests/golden.rs` (its `deny_unknown_fields` tolerance parser accepts the new keys)
Interfaces:

- `turbine-golden compare … [--concurrency <n>]` (default 1): at most `n` prompts in flight (`futures_util::stream::iter(..).buffer_unordered(n)`), results reported in prompt order; exit codes unchanged
- `Tolerance { …, max_abs_logprob_diff_likely_batched: Option<f32>, max_abs_logprob_diff_tail_batched: Option<f32> }` (serde default, omitted when absent); `Tolerance::logprob_bounds(concurrency) -> LogprobBounds { batched, max_abs_logprob_diff_likely, max_abs_logprob_diff_tail }` (strict at 1; above 1 each tier's batched bound, else the strict one) and `Tolerance::at_concurrency(concurrency)`; `CompareReport` gains `concurrency` and `logprob_bounds`, `CompareReport::new(prompts, tolerance, concurrency)`; the token rule is unchanged
  Covers: §Interfaces `turbine-golden compare (change)`; S-15 AC `cargo test -p turbine-bench --test golden batched`
  Depends on: Phase 1 plan Task 18

- [ ] Write failing test `turbine-bench --test golden compare_concurrency_bounded`: against the mock endpoint that records its maximum simultaneous requests, `compare --concurrency 3` over 8 prompts exits 0 and the mock saw at most 3 and at least 2 concurrent requests. Run: `cargo test -p turbine-bench --test golden compare_concurrency_bounded` — expect FAIL
- [ ] Implement the flag.
- [ ] Write failing tests `turbine-bench --test golden compare_batched_bounds_apply_only_above_concurrency_1` (a likely logprob moved by 0.2 fails at `--concurrency 1`, passes at `--concurrency 2` with batched 0.25, report `logprob_bounds.batched` true; a 2-nat flip still fails) and `compare_without_batched_keys_falls_back_to_strict_bounds`. Run: `cargo test -p turbine-bench --test golden batched` — expect FAIL (unknown field `max_abs_logprob_diff_likely_batched`)
- [ ] Implement the batched bounds; add `max_abs_logprob_diff_likely_batched` 0.25 and `max_abs_logprob_diff_tail_batched` 0.75 to both committed `tolerance.json` files.
- [ ] Run: `cargo test -p turbine-bench` — expect PASS
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `feat(turbine-bench): concurrent golden comparison`

## Task 20: Lab configs, OLMoE weights and fixtures, tool/JSON lab test on novanas

Files: `scripts/lab/phase2-novanas-llama.yaml`, `scripts/lab/phase2-novanas-olmoe.yaml` (new), `scripts/lab/novanas-test-job.yaml` (`TURBINE_TEST_MOE_MODEL_DIR`), `scripts/lab/novanas-vllm-job.yaml` (new), `scripts/lab-serve.sh` (`novanas --vllm <slug>`), `tests/golden/olmoe-1b-7b-0125-instruct/{reference.jsonl,tolerance.json}`, `tests/golden/tools/requests.jsonl` (new), `crates/turbine-server/tests/lab_openai.rs` (new)
Interfaces:

- `phase2-novanas-llama.yaml` / `phase2-novanas-olmoe.yaml`: the Phase 1 lab config with `model.path: /models/llama-3.2-3b-instruct` / `/models/olmoe-1b-7b-0125-instruct` and the scheduler defaults
- `novanas-vllm-job.yaml`: namespace `turbine-ci`, image `rocm/vllm` at a tag pinned after checking `curl -s 'https://hub.docker.com/v2/repositories/rocm/vllm/tags?page_size=25'`, `amd.com/gpu: 1`, `hostNetwork: true`, port 18100, weights read-only, args `--dtype bfloat16 --kv-cache-dtype auto`; `scripts/lab-serve.sh novanas --vllm <slug>` waits for `/v1/models` or exits 1 printing the pod log
- `tests/golden/tools/requests.jsonl`: chat requests with `tools` (`required`, named, `auto`) and `response_format` `json_schema`
- `lab_openai tools_and_json_schema` (ignored): starts `turbine-server` on the real Llama-3.2-3B with the HIP backend, runs every request greedily, asserts constrained calls and JSON outputs parse and validate and `auto` results never leak raw call JSON into `content`
- `tolerance.json` for OLMoE identical to Llama's (amended 2026-09-26: replaced by the calibrated self-spread tolerance, see `tests/golden/olmoe-1b-7b-0125-instruct/README.md`)
  Covers: S-15/S-17/S-18 AC `lab_openai tools_and_json_schema`; S-5/S-16/S-15 `hip_ops paged_and_moe_ops` (lab run); S-15 (lab configs and jobs)
  Depends on: Tasks 17, 18

- [ ] Download once with the hf CLI the user logged in on novanas (decision 2026-09-26; the token is never read or passed): `ssh piwi@192.168.10.203 '~/.local/bin/hf download allenai/OLMoE-1B-7B-0125-Instruct --revision b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e --local-dir /home/piwi/turbine-models/olmoe-1b-7b-0125-instruct'`
- [ ] ASK THE USER FIRST that novanas CPU time is available, then generate the OLMoE reference: `ssh piwi@192.168.10.203 'cd /home/piwi/turbine-ci/src && uv run scripts/golden/hf_reference.py --model-dir /home/piwi/turbine-models/olmoe-1b-7b-0125-instruct --prompts tests/golden/prompts.jsonl --out tests/golden/olmoe-1b-7b-0125-instruct/reference.jsonl --top-logprobs 20 --device cpu'` and copy it back.
- [ ] Write the ignored test `turbine-server --test lab_openai tools_and_json_schema` and the request fixture. Run: `scripts/lab-test.sh novanas` — expect FAIL until the configs and env exist
- [ ] Implement configs, Job changes, the vLLM Job and the `--vllm` mode; `bash -n scripts/lab-serve.sh && shellcheck scripts/lab-serve.sh`.
- [ ] ASK THE USER FIRST that an R9700 is free. Run: `scripts/lab-test.sh novanas` — expect PASS with `test paged_and_moe_ops ... ok`, `test tools_and_json_schema ... ok`, Job exit 0
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `test(lab): phase 2 configs, OLMoE reference and tool/JSON lab test`

## Task 21: Concurrency golden runs, baseline, overload and acceptance on macOS

Files: `AGENTS.md` (Commands: `scripts/lab-serve.sh novanas --vllm <slug>`, `turbine-golden compare --concurrency`), `examples/turbine.yaml` (scheduler, timeout and structured-output keys)
Interfaces:

- Consumes `scripts/lab-serve.sh`, `turbine-golden`, `turbine-bench` (Phase 0/1)
  Covers: S-1 AC (macOS build/test/clippy/fmt and `cargo tree -p turbine-scheduler`); S-15/S-16 AC manual golden runs at `--concurrency 1` (strict) and `--concurrency 16` (batched bounds); S-15 AC manual baseline run; S-15 AC manual overload run
  Depends on: Tasks 19, 20

- [ ] Run on the macOS workstation: `cargo build --workspace && cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check && ! cargo tree -p turbine-scheduler | grep -E 'turbine-(kernels|model)|hip|cuda|nvml'` — expect PASS
- [ ] ASK THE USER FIRST that an R9700 is free. For each of `scripts/lab/phase2-novanas-llama.yaml` and `scripts/lab/phase2-novanas-olmoe.yaml`: `scripts/lab-serve.sh novanas <config>` (expect `/ready` 200), then `cargo run --release -p turbine-bench --bin turbine-golden -- compare --url http://192.168.10.203:18000 --reference tests/golden/<slug>/reference.jsonl` — expect exit 0 under the strict bounds (verdict line `strict bounds (concurrency 1)`), and the same with `--concurrency 16` — expect exit 0 under the token rule with the batched bounds (verdict line `batched bounds (concurrency 16)`); paste all four outputs into evidence; `scripts/lab-serve.sh novanas --stop`.
- [ ] ASK THE USER FIRST. For each model: serve it, run `cargo run --release -p turbine-bench -- --url http://192.168.10.203:18000 --concurrency 16 --requests 200 --prompt-words 512 --max-tokens 256 --ignore-eos --output json` — expect exit 0 with `"requests_failed": 0`; stop; then `scripts/lab-serve.sh novanas --vllm <slug>` and the same command against `http://192.168.10.203:18100` — record the JSON or "vLLM-ROCm did not run" with the pod log excerpt, the ROCm version, image tag and card.
- [ ] ASK THE USER FIRST. Serve the Llama config with `scheduler.max_queued_requests: 256`, run `cargo run --release -p turbine-bench -- --url http://192.168.10.203:18000 --concurrency 512 --requests 2000 --max-tokens 256 --ignore-eos --output json` — expect exit 0, some 429s in the report, the pod still `Running` with `/ready` 200, and `/turbine/v1/kv` `blocks_used: 0` once idle.
- [ ] Implement the AGENTS.md and `examples/turbine.yaml` updates.
- [ ] Gate: cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
- [ ] Commit: `docs: phase 2 commands and lab evidence`
