# phase-5p-serving-efficiency — implementation plan

Status: draft
Spec: .procoder/specs/phase-5p-serving-efficiency.md

## Goal

Make serving cheaper on one novanas R9700 without changing outputs: measure the host share of a step at c1 and c16 (and build overlap scheduling only if c1 exceeds 5 %), then reuse cached prefixes token by token through a copy-on-write tail block, order waiting requests by cached prefix with a starvation bound, and jump forward over llguidance's forced tokens in one verified step — each landed alone and measured.

## Architecture

Measurement first: `turbine-bench stage-share` turns two `/metrics` scrapes into a host share, and `scripts/lab-bench.sh` gains `--dry-run`, `--concurrency`, `--multi-turn` and `--multi-turn-sessions` so every landing step records the standard bench, the host share and the Phase 4 multi-turn profile on GPU 0. Token-granular reuse stays inside the Phase 4 boundaries: `turbine_kv::directory` gains L0-only partial entries and a token-by-token comparison of the last block, `planner` a token-granular cap that keeps `MIN_RECOMPUTE_TOKENS`, `hierarchy` a `TailCopy` that the scheduler plans as `IterationPlan.tail_copies` and the engine executes with the fork copies through `ModelExecutor::copy_blocks` — the Phase 5 `TpExecutor` already fans that call out to every rank. The cache-aware policy is one file in the `scheduling_policy` registry, fed by two new `AdmissionInfo` fields; jump-forward adds a multi-token append (`Scheduler::extend`, `IterationOutcome.rolled_back`, `SeqSlice.logits_rows`) and a verification loop in the engine that treats llguidance's forced tokens as a draft.

## Constraints

Copied verbatim from the spec (Constraints):

- Order and landing (decision "Phase 5p"): S-1/S-2 (measure) first, then S-3 only under its rule, then S-4–S-6, then S-7–S-8, then S-9–S-10. Each item lands alone and is measured before the next: `scripts/lab-bench.sh` with golden c1 and the throughput bench per landing step; for S-4–S-8 also the multi-turn profile before and after (`--multi-turn`, and `--multi-turn-sessions 64` for S-7); for S-10 the JSON-schema golden of Interfaces.
- Test tiers (AGENTS.md "Test tiers", decision 2026-09-27): per landing step `scripts/gate.sh`, plus `scripts/lab-test.sh novanas --tier quick` when the change touches GPU-facing code (`turbine-model` for S-9, the `kv_gpu` lab tests for S-5/S-6), plus `scripts/lab-bench.sh --quick` for an iteration read and the full 200-request `scripts/lab-bench.sh` for the landing number; phase exit: `scripts/gate.sh --full`, `scripts/lab-test.sh novanas --tier full`, `scripts/lab-bench.sh --golden16` for both models and the 10-minute `scripts/overload-soak.sh novanas --duration 10m`.
- Prefix-exact prefill invariance (decisions "Pre-Phase-5 #1 follow-up" and "(A)"): a warm request that reuses a token-granular prefix produces bit-identical tokens and logprobs to the cold run of the same prompt when the reused KV was written by a prefill (the Phase 4 `kv_gpu` rule); Llama's prefill steps keep the invariant GEMM rows, the planner keeps `MIN_RECOMPUTE_TOKENS` = 2, and a warm prefill that starts mid-block must stay prefill-shaped. KV written by decode steps (a previous turn's generated tokens) is reused as in Phase 4: identical within the golden tolerance, not bitwise, because decode steps use the speed-tuned rows.
- KV tiers (decision constraint): L1 and L2 hold full blocks only; a partial entry never leaves L0; the Phase 4 reuse-evidence gates, `DEMOTION_INFLIGHT` and capacity demotion are unchanged for full blocks.
- Correctness bar unchanged: the golden tolerances of `tests/golden/<slug>/tolerance.json` (strict at concurrency 1, batched bounds at 16), the OLMoE calibration, the `kv_gpu` bit-exact prefix checks, the Phase 3 reservations (a tail copy's block is reserved; a shared block is never written) and cancellation releasing every reference within one iteration.
- Scheduling policies are stateless, clock-free and deterministic (Phase 2m); every scheduler change is covered by the deterministic simulator (TS §17) with no GPU.
- Bounded everything (TS §21 rule 8): children compared per lookup (64), partial entries per parent (8), partial entries in total (they are L0 blocks, reclaimed like cached blocks), forced tokens per step (_structured_output.jump_forward_max_tokens_ ≤ 256), logits rows per step (≤ max_batch_tokens).
- Every automatic decision exposes a reason code and a metric (TS §21 rule 7): partial reuse outcomes, jump-forward accept/reject, policy selection (`module_selected`).
- Phase 5 (merged before this phase starts): every item works at tp 1, TP (local and static rank modes), EP and DP as S-6 states; lab checks at tp 2 run through `scripts/lab-cluster.sh tp2-novanas` (both R9700s, under `scripts/bench-lock.sh`) at phase exit, not per landing step.
- Unsafe Rust and FFI stay in `crates/turbine-kernels/src`; `turbine-kv` and `turbine-scheduler` gain no GPU or model dependency; no new external dependency (llguidance 1.8.0 already provides `compute_ff_tokens`); no vendor GPU crate.
- Lab: novanas only, GPU 0 for every throughput, golden and multi-turn number (GPU 1 is not comparable), native runs under `scripts/bench-lock.sh`; the standing novanas approvals (2026-09-25/26) cover `lab-test.sh`, `lab-serve.sh` and the golden/bench runs; if `amd.com/gpu` is held by another workload, stop and ask the user; soaks are asked for first.
- Every run is recorded: `lab-bench.sh` uploads to labbook (set `phase-5p`), and each landing step adds a row to `.procoder/perf-log.md`.

From the interface contract and the work in flight (binding):

- Names in `.procoder/contract/interfaces.md` §11 (`turbine-kv`), §12 (`turbine-scheduler`), §10 (`turbine-model`), §16A (`turbine-bench`), §17 (metrics) and §24 (registries) are used verbatim; this phase's additions are listed in the contract's §25 "Phase 5p additions" and in the spec's Interfaces. No kernel ABI change: `TURBINE_ABI_VERSION` stays `2u`, the minor stays Phase 5's.
- Toolchain edition 2024, `rust-version = "1.97"`; `#[non_exhaustive]` on enums later phases extend (`EvictReason`, `PlanReason`); config structs `#[serde(deny_unknown_fields, default)]`; time through `Arc<dyn Clock>`; metric labels from closed enums rendered with `as_str()`.
- Starting point: Phase 5 (`phase-5-multi-gpu`) merged into `main`. Its changes this phase builds on: `turbine_server::engine::tp::{TpExecutor, WorkerRank}` (`copy_blocks` runs on every rank through a copies-only `StepPlan`), `turbine_distributed::rank::{StepPlan { copies, .. }, StepSeq}` (several tokens per sequence already), `turbine_kv::tier::sharded::ShardedL1Tier`, `KvFormat.shards`, `Scheduler::with_micro_batches` (plans completing out of order, matched by `outcome.iteration`) and group reservations (`Reservation::with_members`). If Phase 5 has not merged when a task starts, the task stops and the coordinator asks the user.
- Builds and tests run on novanas through `scripts/remote-cargo.sh` (no local target directories); every task ends with `scripts/gate.sh` printing `gate: ok`; GPU-facing tasks add `scripts/lab-test.sh novanas --tier quick`; every task that changes serving code ends with `scripts/lab-bench.sh --model llama` and `--model olmoe` (GPU 0, under `scripts/bench-lock.sh`, labbook set `phase-5p`) and a row in `.procoder/perf-log.md` — one change, then measure; the next task starts only after that row.
- Lab runs of this phase fall under the standing novanas approvals (2026-09-25, 2026-09-26): `lab-test.sh`, `lab-bench.sh`, golden and bench runs while the R9700s are free; if `amd.com/gpu` is held by another workload, stop and ask the user; the overload soak is asked for first.

## Task 1: `turbine-bench stage-share`

Files: `benches/turbine-bench/src/stage_share.rs` (new: parse two scrapes, compute the share), `benches/turbine-bench/src/lib.rs` (`pub mod stage_share`), `benches/turbine-bench/src/main.rs` (dispatch `stage-share` like `kv-sim`), `benches/turbine-bench/tests/bench.rs` (test), `benches/turbine-bench/tests/fixtures/stage_share_before.txt`, `benches/turbine-bench/tests/fixtures/stage_share_after.txt` (two committed scrapes of a tiny-server run)
Interfaces:

- `pub struct StageShare { pub iterations: u64, pub seconds: BTreeMap<String, f64>, pub host_share: f64 }` (serde `Serialize`, the spec's Data shape)
- `pub fn stage_share(before: &str, after: &str) -> Result<StageShare, StageShareError>`; `StageShareError::{MissingHistogram(String), NoIterations}`
- `host_share` = Σ `turbine_engine_iteration_seconds_sum{stage}` deltas over every stage except `device_wait` ÷ Σ over all eight; `iterations` = the `stage="schedule"` count delta
- CLI: `turbine-bench stage-share --before <file> --after <file> [--output text|json]`; exit 0, or 2 on `StageShareError`
  Covers: S-1 AC `bench stage_share_from_metrics`
  Depends on: nothing in this phase

- [ ] Write failing test `bench stage_share_from_metrics`: load the two fixtures, assert each stage's seconds, `iterations` and `host_share` equal the values computed by hand in the test (1e-9), that moving 1 s from `launch` to `device_wait` in a copy of the after-scrape lowers the share, and that `stage-share` with `--before` = `--after` exits 2 naming `NoIterations`. Run: `scripts/remote-cargo.sh test -p turbine-bench --test bench stage_share_from_metrics` — expect FAIL (module missing)
- [ ] Implement the parser over the Prometheus text format (lines `turbine_engine_iteration_seconds_sum{stage="…"} <v>` and `_count`), the formula and the subcommand.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-bench` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(bench): stage-share, the host share of engine iterations from two scrapes`

## Task 2: `lab-bench.sh` — dry run, concurrency, host share, multi-turn

Files: `scripts/lab-bench.sh` (`--dry-run`, `--concurrency`, `--multi-turn`, `--multi-turn-sessions`, `stage-share` call, new `BENCH` fields and labbook values), `benches/turbine-bench/tests/lab_scripts.rs` (test through the existing `dry_run` helper), `AGENTS.md` (Commands: the lab-bench line)
Interfaces:

- `scripts/lab-bench.sh [--dry-run] [--gpu 0] [--model llama|olmoe] [--label L] [--with-tests] [--skip-tests] [--golden16] [--quick] [--concurrency <n>] [--multi-turn] [--multi-turn-sessions <n>] [-- --set k=v ...]`
- `BENCH` line gains `conc=<n> host_share=<x.xxxx>` always and `mt_cached= mt_ttft_first_p50= mt_ttft_first_p99= mt_ttft_later_p50= mt_tok_s=` with a multi-turn flag; `labbook-values.json` gains `host_share`, `concurrency` and the `mt_*` keys
- Multi-turn command (standard): `turbine-bench --url <url> --profile multi-turn --sessions 16 --turns 8 --shared-prefix-words 2000 --concurrency 8 --session-hints --output json`; saturated: `--sessions <n> --concurrency <n>`; both run after the throughput bench under the same `bench-lock.sh` hold
  Covers: S-1 AC `bash -n` / `shellcheck` / `lab_scripts lab_bench_dry_run_passes_flags`
  Depends on: Task 1

- [ ] Write failing test `lab_scripts lab_bench_dry_run_passes_flags`: `scripts/lab-bench.sh --dry-run --concurrency 1 --quick --multi-turn-sessions 64 -- --set scheduler.policy=cache_aware` exits 0 and prints, in order, the remote build, the server start with `--set scheduler.policy=cache_aware`, `turbine-golden compare … --concurrency 1`, `turbine-bench … --concurrency 1 --requests 64 …`, `turbine-bench stage-share --before … --after …`, `turbine-bench … --profile multi-turn --sessions 64 … --concurrency 64 …` and the server stop, and contacts no host (no `ssh` executed: `PATH` holds a failing `ssh` stub). Run: `scripts/remote-cargo.sh test -p turbine-bench --test lab_scripts lab_bench_dry_run_passes_flags` — expect FAIL
- [ ] Implement the flags (a `run` wrapper that echoes under `--dry-run`), the host-share call and the multi-turn step; keep the default command line and `BENCH` fields byte-compatible except for the two new always-present fields.
- [ ] Run: `bash -n scripts/lab-bench.sh && shellcheck scripts/lab-bench.sh && scripts/remote-cargo.sh test -p turbine-bench --test lab_scripts` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(lab): lab-bench dry run, concurrency, host share and the multi-turn profile`

## Task 3: Measure first — host share and multi-turn baselines (S-2)

Files: `.procoder/perf-log.md` (new table "Phase 5p" with the baseline rows), `.procoder/ask/decisions.md` (entry "Phase 5p: host share measured — overlap scheduling go/no-go", with the numbers and the rule's outcome)
Interfaces:

- Consumes Task 2's `BENCH` fields; produces the S-2 baselines every later task compares with: Llama and OLMoE tok/s, TTFT p50, `host_share` at c1 and c16; Llama `mt_cached`, `mt_ttft_later_p50` (standard) and the saturated `mt_*` values with `scheduler.max_running_requests=16`
  Covers: S-2 AC (manual lab measurement)
  Depends on: Task 2; Phase 5 merged

- [ ] Confirm `main` contains the Phase 5 merge (`git log --oneline main | grep -i "phase 5"`) and branch `phase-5p` from it; if not, stop and ask.
- [ ] Run (GPU 0): `scripts/lab-bench.sh --model llama --concurrency 1 --quick`, `scripts/lab-bench.sh --model llama --multi-turn`, `scripts/lab-bench.sh --model olmoe --concurrency 1 --quick`, `scripts/lab-bench.sh --model olmoe` — expect exit 0, `golden1=PASS`, four `host_share=` values (c16 expected ≈ 0.015 from `.procoder/perf-profile-2026-09-27.md`)
- [ ] Run: `scripts/lab-bench.sh --model llama --multi-turn-sessions 64 -- --set scheduler.max_running_requests=16` three times — expect exit 0; its median `mt_*` values are the saturated baseline
- [ ] Run the standard multi-turn twice more (`scripts/lab-bench.sh --model llama --multi-turn`) so the Task 8 comparison has a median of 3
- [ ] Record every `BENCH` line in `.procoder/perf-log.md`; write the decisions entry: overlap work (Task 4) is built iff a c1 `host_share` > 0.05 (decision "Phase 5p", item 4)
- [ ] Commit: `docs(perf): phase 5p baselines — host share at c1/c16 and multi-turn`

## Task 4: Conditional — overlap scheduling without the TTFT penalty (S-3)

Files: (only if Task 3 recorded a c1 host share > 0.05) `crates/turbine-server/src/engine/loop.rs` (`turn_overlap` / `launch_next`: launch ahead only while no admitted request waits for its first prefill chunk), `crates/turbine-scheduler/src/scheduler.rs` (`Scheduler::has_unstarted_admitted(&self) -> bool`), `crates/turbine-core/src/config/mod.rs` and `examples/turbine.yaml` (default flip only after the A/B), `.procoder/perf-log.md`
Interfaces:

- `Scheduler::has_unstarted_admitted(&self) -> bool` — an admitted request whose prefill has not started exists (read before launching ahead)
- `execution.overlap_scheduling` default `false` → `true` only when the A/B passes
  Covers: S-3 AC (conditional)
  Depends on: Task 3

- [ ] If every c1 `host_share` in Task 3 is ≤ 0.05: mark this task done with a pointer to the Task 3 decisions entry; no code, no commit. Otherwise continue.
- [ ] Write failing test `engine::r#loop::tests::overlap_waits_for_new_prefills`: with overlap on (CPU backend, tiny Llama), a request submitted while step N+1 is launched ahead has its first prefill chunk in the plan right after N+1, not one later, and streams equal the serial run. Run: `scripts/remote-cargo.sh test -p turbine-server --bin turbine-server engine::r#loop::tests::overlap_waits_for_new_prefills` — expect FAIL
- [ ] Implement the rule in `launch_next`; keep the Phase 2c overlap tests passing.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-server --bin turbine-server engine::r#loop::tests && scripts/remote-cargo.sh test -p turbine-server --test tiny_server overlap_scheduling_matches_serial` — expect PASS
- [ ] Lab A/B (GPU 0): `scripts/lab-bench.sh --model <llama|olmoe> [--concurrency 1 --quick] -- --set execution.overlap_scheduling=<true|false>` for both models at c1 and c16 — flip the default only if tok/s rises and TTFT p50 ≤ 1.10 × the serial run for both models; record all eight lines
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `perf(server): overlap scheduling waits for new prefills` (plus `feat(core): overlap scheduling on by default` only if the A/B passed)

## Task 5: Phase 5p configuration keys

Files: `crates/turbine-core/src/config/mod.rs` (`KvConfig.partial_prefix_reuse`, `SchedulerConfig.cache_aware_window`, `StructuredOutputConfig.{jump_forward, jump_forward_max_tokens}`, validation), `crates/turbine-core/src/config/tests.rs` (test), `examples/turbine.yaml` (the four keys, commented), `crates/turbine-scheduler/src/scheduler.rs` (`SchedulerParams.cache_aware_window`), `crates/turbine-server/src/engine/mod.rs` (params from config)
Interfaces:

- `KvConfig.partial_prefix_reuse: bool` (true), `SchedulerConfig.cache_aware_window: HumanDuration` (1s, 10ms..=60s), `StructuredOutputConfig.jump_forward: bool` (true), `StructuredOutputConfig.jump_forward_max_tokens: u32` (32, 1..=256, and + 1 ≤ `scheduler.max_batch_tokens`)
- `SchedulerParams.cache_aware_window: Duration` (read by Task 9's policy); the other keys are read by Tasks 8 and 12 (until then they change nothing)
  Covers: S-7/S-11 AC `config::tests::phase5p_keys`
  Depends on: Task 3

- [ ] Write failing test `config::tests::phase5p_keys`: defaults true / 1s / true / 32; `scheduler.cache_aware_window: 5ms` and `61s`, `structured_output.jump_forward_max_tokens: 0` and `257`, and `jump_forward_max_tokens: 2048` with `max_batch_tokens: 2048` are rejected naming the key; `--set kv.partial_prefix_reuse=false` applies; `examples/turbine.yaml` loads. Run: `scripts/remote-cargo.sh test -p turbine-core config::tests::phase5p_keys` — expect FAIL
- [ ] Implement the fields with serde defaults and the checks through the existing `invalid(key, reason)` helper; thread `cache_aware_window` into `SchedulerParams`.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-core && scripts/remote-cargo.sh test -p turbine-scheduler` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(core): phase 5p configuration keys`

## Task 6: Token-granular prefix match in the directory and the planner (S-4)

Files: `crates/turbine-kv/src/directory.rs` (`PartialMatch`, `PrefixMatch.partial`, `insert_partial`, `KvBlock::is_partial`, the last-block comparison in `lookup`, `PARTIAL_SCAN_LIMIT`, `PARTIAL_PER_PARENT`, tests), `crates/turbine-kv/src/planner.rs` (`PlanInputs.partial_tokens`, `KvPlan.partial_tokens`, `reuse_cap_tokens`, tests), `crates/turbine-kv/src/metrics.rs` (`EvictReason::PartialL0Only`, `PartialOutcome` for `turbine_kv_partial_reuse_total{outcome}`)
Interfaces:

- `pub struct PartialMatch { pub key: KvKey, pub block: BlockId, pub tokens: u32 }`; `PrefixMatch.partial: Option<PartialMatch>`
- `KvDirectory::insert_partial(&mut self, block: KvBlock) -> Result<(), DirectoryError>`; `KvBlock::is_partial(&self, block_tokens: u32) -> bool`; `pub const PARTIAL_SCAN_LIMIT: usize = 64; pub const PARTIAL_PER_PARENT: usize = 8;`
- `lookup` keeps its signature; after the full-block walk it compares the remaining prompt tokens with the L0-resident children of the last matched key (`ROOT_PARENT` when none), most recently used first, and fills `partial` with the longest common prefix (≥ 1 token); an L1/L2-only best candidate is reported as outcome `not_resident`
- `pub fn reuse_cap_tokens(prompt_tokens: u32) -> u32` (= `prompt − MIN_RECOMPUTE_TOKENS`); `plan_prefix` reuses the partial tokens only when its cutoff keeps every matched full block, capped so full + partial ≤ `reuse_cap_tokens`
- `pub enum PartialOutcome { Reused, NoCandidate, NotResident, PlannerCutoff, Disabled }` with `as_str()`
  Covers: S-4 ACs `directory::tests::token_granular_match`, `planner::tests::token_granular_cap`
  Depends on: Task 5

- [ ] Write failing test `directory::tests::token_granular_match` exactly as the spec's S-4 criterion lists (3-block chain + 40-token partial entry; 25-token partial match; divergence at token 70 of a full block; colliding test hasher → 0; a prompt shorter than one block under `ROOT_PARENT`; L1-only → `not_resident`; 64-child scan limit; ninth partial entry evicts the least recently used). Run: `scripts/remote-cargo.sh test -p turbine-kv directory::tests::token_granular_match` — expect FAIL
- [ ] Write failing test `planner::tests::token_granular_cap`: prompts of 129, 130, 255, 256, 257 tokens fully cached at 128-token blocks reuse 0+127, 1+0, 1+125, 1+126, 1+127 (full blocks + partial tokens); with no partial candidate the plan equals the brute-force whole-block minimum over 200 random cases; the warm prefill is never shorter than 2 tokens. Run: `scripts/remote-cargo.sh test -p turbine-kv planner::tests::token_granular_cap` — expect FAIL
- [ ] Implement; keep every Phase 4 test in `turbine-kv` unchanged and passing (`collision_is_a_miss`, `longest_prefix_across_tiers`, `cutoff_minimises_cost`).
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-kv` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(kv): token-granular match of the last prefix block`

## Task 7: Copy-on-write attach, tail copies in the plan, partial publication (S-5, host side)

Files: `crates/turbine-kv/src/hierarchy.rs` (`TailCopy`, `PrefixAttach.tail_copy`, `attach_prefix` allocating the private block and holding `src`, `publish_tail`, `tail_copied`, `HierarchyConfig.partial_prefix_reuse`, partial entries excluded from demotion/prefetch/reuse evidence, reclaim reason `partial_l0_only`, tests), `crates/turbine-scheduler/src/request.rs` (`SchedRequest::attach_prefix`: token-granular `cached_prefix_tokens`, `projected_kv_blocks` minus full blocks only), `crates/turbine-scheduler/src/scheduler.rs` (`IterationPlan.tail_copies`, planned in the request's first iteration; table = shared blocks + private block; `tail_copied` after `complete`; release on drop/requeue), `crates/turbine-scheduler/src/sim/mod.rs` (`KvSimDriver` executes tail copies and publishes tails), `crates/turbine-scheduler/tests/kv_sim.rs` (tests)
Interfaces:

- `pub struct TailCopy { pub src: BlockId, pub dst: BlockId, pub tokens: u32 }`; `PrefixAttach.tail_copy: Option<TailCopy>`; `cached_tokens` = blocks × block_tokens + tail tokens
- `KvHierarchy::publish_tail(&mut self, pool: &mut BlockPool, request: RequestId, block: BlockId, tokens: &[u32])`, `KvHierarchy::tail_copied(&mut self, pool: &mut BlockPool, request: RequestId)`
- `IterationPlan.tail_copies: Vec<(SeqId, BlockId, BlockId)>` (executed before the forward, with `forks`); `IterationPlan::is_empty` also checks it
- The server still passes `partial_prefix_reuse: false` to the hierarchy in this task (Task 8 wires the executor side and turns it on)
  Covers: S-5 ACs `kv_sim partial_tail_reuse`, `kv_sim cancellation_before_tail_copy`, `kv_sim cancellation_releases_kv`, `kv_sim demotion_under_pressure`
  Depends on: Task 6

- [ ] Write failing test `kv_sim partial_tail_reuse` as the spec's S-5 criterion lists (multi-turn simulated workload; reuse up to the last written token, capped at `prompt − 2`; one tail copy per reusing request in its first iteration; shared blocks never written; the private block reserved; token-granular `cached_prefix_tokens`; no partial entry demoted with L1 on; every reference back to 0; with the switch off the run equals the Phase 4 result via `SimReport::plan_digest`). Run: `scripts/remote-cargo.sh test -p turbine-scheduler --test kv_sim partial_tail_reuse` — expect FAIL
- [ ] Write failing test `kv_sim cancellation_before_tail_copy`: requests cancelled, timed out and SURVIVAL-requeued between attach and their first iteration release `src` and `dst` within one iteration. Run: `scripts/remote-cargo.sh test -p turbine-scheduler --test kv_sim cancellation_before_tail_copy` — expect FAIL
- [ ] Implement; `publish_tail` stores only tokens whose KV is written (prompt + generated − the newest, never-fed token) and only for a normal finish; partial entries go through `insert_partial`.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-kv && scripts/remote-cargo.sh test -p turbine-scheduler` — expect PASS (including `prefix_reuse_refcounts`, `demotion_under_pressure`, `cancellation_releases_kv`, `one_off_overload_does_not_demote`, `overload_sim`)
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(kv,scheduler): copy-on-write tail block for token-granular prefix reuse`

## Task 8: Token-granular reuse in the engine, on every rank — landing (S-5, S-6)

Files: `crates/turbine-server/src/engine/loop.rs` (execute `plan.tail_copies` in `fork_copies` through `ModelExecutor::copy_blocks`; `publish_tail` for choice 0 on a normal finish; `tail_copied` after `complete`; `kv_commits` unchanged for full blocks; tests), `crates/turbine-server/src/kv_orchestrator.rs` (`partial_prefix_reuse` from config, `publish_tail`, the KV document fields), `crates/turbine-server/src/metrics.rs` (`turbine_kv_partial_reuse_total{outcome}`, `turbine_kv_partial_reused_tokens_total`, `turbine_kv_partial_entries`), `crates/turbine-kv/src/document.rs` (`partial_prefix_reuse`, `l0.partial_entries`), `crates/turbine-api/tests/api.rs` (`kv_metrics_bounded` extended), `crates/turbine-server/tests/tiny_server.rs` (`tp2_partial_prefix_reuse`), `crates/turbine-server/tests/kv_gpu.rs` (`prefix_reuse_partial_block_matches_cold`; `prefix_reuse_suffix_lengths_match_cold`'s expected `cached_tokens` becomes token-granular), `.procoder/perf-log.md`
Interfaces:

- Tail copies share the fork-copy path: one `copy_blocks(&pool.view(), &src, &dst)` call per iteration with fork pairs first, then tail pairs; under TP `TpExecutor::copy_blocks` runs it on rank 0 and every `WorkerRank` (local mode) or sends it in the `StepPlan.copies` of that step (static mode)
- Metric `outcome` values from `turbine_kv::metrics::PartialOutcome`; KV document keys `partial_prefix_reuse`, `tiers[l0].partial_entries`
  Covers: S-5 ACs `engine::r#loop::tests::partial_prefix_reuse_matches_cold`, `api kv_metrics_bounded`, lab `kv_gpu prefix_reuse_partial_block_matches_cold`; S-6 AC `tiny_server tp2_partial_prefix_reuse`; S-4/S-5/S-13 landing measurement
  Depends on: Task 7

- [ ] Write failing test `engine::r#loop::tests::partial_prefix_reuse_matches_cold` (CPU backend, tiny Llama): prompts reusing 1, 2, 17, 100 and 127 tokens of a partial block and one diverging inside a full block give tokens and logprobs bitwise equal to `kv.partial_prefix_reuse=false` and to a cold engine, with `cached_tokens` = reused tokens. Run: `scripts/remote-cargo.sh test -p turbine-server --bin turbine-server engine::r#loop::tests::partial_prefix_reuse_matches_cold` — expect FAIL
- [ ] Write failing test `tiny_server tp2_partial_prefix_reuse`: tp 2 on the CPU backend, `local` and `static` (loopback `tcp`), each rank's copy counter shows the tail copy once before the forward, outputs equal tp 1, pools return to the same free count. Run: `scripts/remote-cargo.sh test -p turbine-server --test tiny_server tp2_partial_prefix_reuse` — expect FAIL
- [ ] Extend `api kv_metrics_bounded` with the three new families and the two KV document keys — expect FAIL until implemented
- [ ] Write the ignored lab test `kv_gpu prefix_reuse_partial_block_matches_cold` (prefill-written prefixes ending 1, 2, 17, 64, 100, 126, 127 tokens into a block and one divergence inside a full block; greedy 64 tokens, `top_logprobs` 5; text and every logprob bit-equal to cold) and update `prefix_reuse_suffix_lengths_match_cold`'s expected `cached_tokens` to `prompt − 2` capped by the cached length.
- [ ] Implement the engine and orchestrator wiring; turn the hierarchy's `partial_prefix_reuse` on from config.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-server && scripts/remote-cargo.sh test -p turbine-api --test api` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Lab: `scripts/lab-test.sh novanas --tier quick` and `scripts/lab-test.sh novanas -- -p turbine-server --test kv_gpu` — expect exit 0 with `prefix_reuse_partial_block_matches_cold`, `prefix_reuse_matches_cold`, `prefix_reuse_suffix_lengths_match_cold` and `nvme_round_trip_matches_cold` ok
- [ ] Landing measurement (GPU 0): `scripts/lab-bench.sh --model llama --multi-turn` three times and `scripts/lab-bench.sh --model olmoe` once — expect golden1 PASS, tok/s ≥ 0.98 × the Task 3 baseline, TTFT p50 ≤ 1.10 ×, median `mt_cached` ≥ baseline + 0.01 and median `mt_ttft_later_p50` ≤ 0.95 × baseline; record the lines in `.procoder/perf-log.md`. If a target is missed, keep the code, set `kv.partial_prefix_reuse` false in the lab configs and report to the coordinator before Task 9.
- [ ] Commit: `perf(server): token-granular prefix reuse through a copy-on-write tail block`

## Task 9: `cache_aware` scheduling policy — landing (S-7, S-8)

Files: `crates/turbine-scheduler/src/policy/mod.rs` (`AdmissionInfo.{prompt_tokens, cached_prefix_tokens, window}`, `AdmissionKey.affinity`, registry entry, unit test), `crates/turbine-scheduler/src/policy/default.rs` (`affinity: 0`), `crates/turbine-scheduler/src/policy/cache_aware.rs` (new), `crates/turbine-scheduler/src/scheduler.rs` (fill the new `AdmissionInfo` fields from `SchedRequest.estimate` and `SchedulerParams.cache_aware_window` in `submit` and `enqueue`), `crates/turbine-scheduler/src/sim/mod.rs` (`KvSimDriver::with_policy`, `SimReport.admission_inversions`), `crates/turbine-scheduler/tests/kv_sim.rs` (`cache_aware_starvation_bound` with a test-only unbounded variant), `docs/extending/scheduling-policy.md` (the `cache_aware` example, the new fields, the pitfall "the key is computed once at push"), `.procoder/perf-log.md`
Interfaces:

- `AdmissionInfo` gains `pub prompt_tokens: u32, pub cached_prefix_tokens: u32, pub window: Duration`; `AdmissionKey { tier, priority, arrival, affinity: u64, order }`
- `pub struct CacheAwarePolicy;` `name()` = `"cache_aware"`; `admission_key`: preempted → as `DefaultPolicy`; else `tier 1, priority, arrival = Duration::from_nanos(⌊arrival / window⌋ × window), affinity = u32::MAX − cached_prefix_tokens, order = submit_no`; `preemption_rank`, `pick_victim`, `chunk_cap` delegate to `DefaultPolicy`
- Registry: `Registry::new("scheduling_policy", &[&DefaultPolicy, &CacheAwarePolicy])`
- `SimReport.admission_inversions: Vec<(RequestId, RequestId)>` — pairs (earlier, later) of equal priority, never preempted, arrival gap ≥ window, where the later was admitted first
  Covers: S-7/S-8 ACs `registry_conformance`, `policy::tests::cache_aware_orders_by_window_then_prefix`, `kv_sim cache_aware_starvation_bound`, `docs_extending`; S-7 lab A/B
  Depends on: Tasks 5, 8

- [ ] Write failing test `policy::tests::cache_aware_orders_by_window_then_prefix` (within a window the larger `cached_prefix_tokens` first; across windows arrival order; priority dominates; preempted as `default`; `default`'s keys unchanged, including the existing `default_policy_orders_like_main`). Run: `scripts/remote-cargo.sh test -p turbine-scheduler policy::tests` — expect FAIL
- [ ] Write failing test `kv_sim cache_aware_starvation_bound` (seeded; a flood of fully cached requests, one uncached every 50 arrivals, `max_running_requests` 4, 2,000 requests, window 1 s): `admission_inversions` empty and every request completes for `cache_aware`; the test-only unbounded variant (window `Duration::MAX`) has inversions and a larger maximum wait for the uncached requests. Run: `scripts/remote-cargo.sh test -p turbine-scheduler --test kv_sim cache_aware_starvation_bound` — expect FAIL
- [ ] Implement; `registry_conformance` runs the Phase 2m suite over both policies with no change to the suite.
- [ ] Update `docs/extending/scheduling-policy.md`. Run: `scripts/remote-cargo.sh test -p turbine-scheduler && scripts/remote-cargo.sh test -p turbine-model --test docs_extending && scripts/remote-cargo.sh test -p turbine-server` — expect PASS (the server's `modules.scheduling_policy` shows `cache_aware` under `--set scheduler.policy=cache_aware`)
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Landing (GPU 0): `scripts/lab-bench.sh --model llama` and `--model olmoe` (default policy: standard-bench bound); then three runs each of `scripts/lab-bench.sh --model llama --multi-turn-sessions 64 -- --set scheduler.max_running_requests=16 --set scheduler.policy=cache_aware` and the same with `scheduler.policy=default` on the same day — expect medians `mt_ttft_later_p50` ≤ 0.85 × default, `mt_cached` ≥ default, `mt_tok_s` ≥ 0.98 × default, `mt_ttft_first_p99` ≤ 1.5 × default, golden1 PASS; record in `.procoder/perf-log.md`; the coordinator asks the user whether `cache_aware` becomes the default.
- [ ] Commit: `feat(scheduler): cache_aware scheduling policy with a windowed starvation bound`

## Task 10: Multi-token append for running sequences and multi-row logits (S-9)

Files: `crates/turbine-scheduler/src/scheduler.rs` (`Scheduler::extend`, the extension item in the decode stage, `IterationOutcome.rolled_back` in `complete`, `SchedError::NotDecoding`, preemption drops a pending extension, tests), `crates/turbine-scheduler/src/sim/mod.rs` (random extensions in the invariant runs), `crates/turbine-model/src/executor/mod.rs` (`SeqSlice.logits_rows`, `Logits::rows_of`), `crates/turbine-model/src/executor/decoder/mod.rs` and `crates/turbine-model/src/executor/batch.rs` (gather the last `logits_rows` positions for the LM head; such rows are never reduced), `crates/turbine-model/tests/tiny_model.rs` (test), `crates/turbine-server/src/engine/tp.rs` (the leader returns every row)
Interfaces:

- `Scheduler::extend(&mut self, seq: SeqId, tokens: u32) -> Result<(), SchedError>` — the next plan feeds `1 + tokens` positions of the decoding sequence as `BatchKind::Prefill { start, len }` from the budget left after decodes, before prefill chunks; not coverable → a plain decode and the extension dropped
- `IterationOutcome.rolled_back: Vec<(SeqId, u32)>` — `complete` shortens the table by that many tokens and frees blocks left empty
- `SeqSlice.logits_rows: u32` (1 ≤ rows ≤ `q_len`); `Logits::rows_of(&self, seq_index: usize) -> &[f32]` (row-major, position order); existing callers set 1
  Covers: S-9 ACs `scheduler::tests::extend_and_roll_back`, `tiny_model multi_row_logits_match_single_steps`
  Depends on: Task 9

- [ ] Write failing test `scheduler::tests::extend_and_roll_back` as the spec lists (6-token item within budget; roll-back of 3 frees an empty block; budget fallback to decode; `NotDecoding` for a prefilling sequence; preemption drops the extension). Run: `scripts/remote-cargo.sh test -p turbine-scheduler scheduler::tests::extend_and_roll_back` — expect FAIL
- [ ] Write failing test `tiny_model multi_row_logits_match_single_steps` (both tiny checkpoints, CPU provider: 1 + 7 tokens with `logits_rows` 8 equal the rows of 8 single steps within the Phase 1 CPU tolerance; reduced decode rows in the same batch unchanged; `logits_rows` > `q_len` refused). Run: `scripts/remote-cargo.sh test -p turbine-model --test tiny_model multi_row_logits_match_single_steps` — expect FAIL
- [ ] Implement; add random extensions to the simulator's invariant runs (`decode_never_starved`, budgets) and keep them passing.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-scheduler && scripts/remote-cargo.sh test -p turbine-model && scripts/remote-cargo.sh test -p turbine-server` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Lab: `scripts/lab-test.sh novanas --tier quick` (GPU-facing: `turbine-model`) — expect exit 0; `scripts/lab-bench.sh --model llama --quick` and `--model olmoe --quick` — expect the standard-bench bound (nothing uses extensions yet)
- [ ] Commit: `feat(scheduler,model): multi-token append for running sequences and multi-row logits`

## Task 11: Forced tokens from the grammar matcher (S-10, matcher side)

Files: `crates/turbine-model/src/structured.rs` (`TokenMatcher::forced_tokens` with a default empty implementation; `LlguidanceMatcher` over `Matcher::compute_ff_tokens`, capped; tests)
Interfaces:

- `fn forced_tokens(&mut self, max: usize) -> Vec<u32>` on `TokenMatcher` (default `Vec::new()`); does not commit anything — the engine commits accepted tokens one at a time with `commit`
  Covers: S-10 AC `structured::tests::forced_tokens_follow_llguidance`
  Depends on: Task 10

- [ ] Write failing test `structured::tests::forced_tokens_follow_llguidance` (tiny tokenizer, `small_schema`: equals `compute_ff_tokens` capped by `max`; empty inside a free string; never re-tokenizes the last committed token; calling it twice without a commit returns the same tokens). Run: `scripts/remote-cargo.sh test -p turbine-model structured::tests::forced_tokens_follow_llguidance` — expect FAIL
- [ ] Implement over llguidance 1.8.0's `Matcher::compute_ff_tokens` (no new dependency).
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-model` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(model): forced tokens from the llguidance matcher`

## Task 12: Verified jump-forward in the engine — landing (S-10)

Files: `crates/turbine-server/src/engine/loop.rs` (after a constrained choice samples: ask `forced_tokens`, cap by `jump_forward_max_tokens`, `max_tokens`, `model.max_seq_len`; `Scheduler::extend`; `logits_rows = k + 1` on that slice; on collect verify row by row through `ActiveRequest::step`, stop at the first disagreement, report `rolled_back`; tests), `crates/turbine-server/src/engine/requests.rs` (`ActiveRequest::step_forced(choice, rows, forced) -> ForcedOutcome`: runs the normal step per row, keeps the sampler's uniform stream, stop strings, EOS, `max_tokens` and tool-call holding per accepted token), `crates/turbine-server/src/metrics.rs` (`turbine_jump_forward_tokens_total{outcome}`, `turbine_jump_forward_steps_total`), `crates/turbine-server/tests/tiny_server.rs` (`jump_forward_matches_token_by_token`), `crates/turbine-server/tests/lab_openai.rs` (`json_schema_jump_forward_matches`), `.procoder/perf-log.md`
Interfaces:

- `pub struct ForcedOutcome { pub accepted: u32, pub next: Option<SampledToken>, pub finished: Option<FinishReason> }`
- Engine flow per constrained choice: sample `t` → `commit(t)` → `forced_tokens(max)` → `Scheduler::extend(seq, k)`; next step feeds `[t, f1..fk]` with `logits_rows = k + 1`; row j is stepped with the mask and sampler as a token-by-token step; accepted while the sampled token equals `f(j+1)`; at the first disagreement the model's token is taken and `rolled_back = k − accepted`; all accepted → row k yields the next token; `kv_commits` counts kept tokens only
- Constrained choices keep finishing serially under overlap scheduling (Phase 2c S-16), so a jump-forward step is never launched ahead
  Covers: S-10 ACs `tiny_server jump_forward_matches_token_by_token`, lab `lab_openai json_schema_jump_forward_matches`; S-13 landing
  Depends on: Task 11

- [ ] Write failing test `tiny_server jump_forward_matches_token_by_token` as the spec lists (JSON schema, `json_object`, tool call; greedy with and without `top_logprobs`, seeded `temperature` 0.8, a stop string inside a forced span, `max_tokens` cutting a span; identical streams, logprobs and finish reasons with `structured_output.jump_forward` true and false; `accepted` > 0; a test grammar forcing a non-canonical split is rejected at that token). Run: `scripts/remote-cargo.sh test -p turbine-server --test tiny_server jump_forward_matches_token_by_token` — expect FAIL
- [ ] Write the ignored lab test `lab_openai json_schema_jump_forward_matches`: the four JSON cases of `tests/golden/tools/requests.jsonl` at temperature 0 with `logprobs` and `top_logprobs` 5, jump-forward on and off (two servers, `--set structured_output.jump_forward=false` for the second); identical token ids (divergence excused only at a near-tie under the golden token rule of `tests/golden/llama-3.2-3b-instruct/tolerance.json`), logprobs within its strict bounds, schema-valid content; print `forward_steps_per_token` and wall time per case for both settings and assert ≤ 0.8 × steps and ≤ 0.9 × time summed over the four.
- [ ] Implement.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-server && scripts/remote-cargo.sh test -p turbine-model` — expect PASS (including `response_format_json_schema`, `matcher_masks_each_choice_and_fails_alone`, the overlap tests)
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Lab: `scripts/lab-test.sh novanas --tier quick` and `scripts/lab-test.sh novanas -- -p turbine-server --test lab_openai` — expect exit 0 with `json_schema_jump_forward_matches` and `tools_and_json_schema` ok; paste the printed steps and times into the perf log
- [ ] Landing (GPU 0): `scripts/lab-bench.sh --model llama` and `--model olmoe` — expect the standard-bench bound and golden1 PASS; record the lines
- [ ] Commit: `perf(server): verified jump-forward over the grammar's forced tokens`

## Task 13: Phase exit — full suites, multi-GPU, soak, docs

Files: `AGENTS.md` (Commands: `stage-share`, the new `lab-bench.sh` flags, `scheduler.policy: cache_aware`, `kv.partial_prefix_reuse`, `structured_output.jump_forward*`; project state: Phase 5p done), `examples/turbine.yaml` (final comments), `.procoder/perf-log.md` (phase summary row), `.procoder/specs/phase-5p-serving-efficiency.md` (tick criteria with evidence)
Interfaces:

- Consumes every earlier task; produces the phase-exit evidence
  Covers: S-11 AC `--check-config`; S-6 phase-exit lab (`lab-cluster.sh tp2-novanas`); S-13 phase-exit AC
  Depends on: Tasks 1–12

- [ ] Run: `cargo run -p turbine-server -- --config examples/turbine.yaml --check-config --set scheduler.policy=cache_aware --set kv.partial_prefix_reuse=false --set structured_output.jump_forward=false` — expect `config ok`; with `--set scheduler.policy=lpm` — expect exit 2 naming `default` and `cache_aware`
- [ ] Run: `scripts/gate.sh --full` — expect `gate: ok`
- [ ] Run: `scripts/lab-test.sh novanas --tier full` — expect exit 0
- [ ] Run: `scripts/bench-lock.sh scripts/lab-cluster.sh tp2-novanas` — expect exit 0 with golden c1 PASS and `kv_gpu prefix_reuse_partial_block_matches_cold` bit-exact at tp 2
- [ ] Run (GPU 0): `scripts/lab-bench.sh --model llama --golden16 --multi-turn` and `scripts/lab-bench.sh --model olmoe --golden16` — expect golden1 and golden16 PASS within the standard-bench bound
- [ ] Ask the user, then run `scripts/overload-soak.sh novanas --duration 10m` — expect `verdict.json` passing (and again with `cache_aware` if the user made it the default)
- [ ] Update `AGENTS.md`, `examples/turbine.yaml` and the perf log; tick the spec's criteria with evidence.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `docs: phase 5p serving efficiency — commands and acceptance evidence`
