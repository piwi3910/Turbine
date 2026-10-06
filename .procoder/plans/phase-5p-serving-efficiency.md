# phase-5p-serving-efficiency — implementation plan

Status: complete
Spec: .procoder/specs/phase-5p-serving-efficiency.md

## Goal

Make serving cheaper on one novanas R9700 without changing outputs: measure the host share of a step at c1 and c16 (and build overlap scheduling only if c1 exceeds 5 %), then reuse cached prefixes token by token through a copy-on-write tail block, order waiting requests by cached prefix with a starvation bound, and jump forward over llguidance's forced tokens in one verified step — each landed alone and measured. Amended on 2026-10-04 for the 6b reality and the follow-ups (decision "Phase 5p start: amendment scope and the track gate", 1 A, 2 A, 3 A): 5p starts on the 6b closing state, and the 6b/6a recorded efficiency follow-ups land here as Tasks 13–17.

## Architecture

Measurement first: `turbine-bench stage-share` turns two `/metrics` scrapes into a host share, and `scripts/lab-bench.sh` gains `--dry-run`, `--concurrency`, `--multi-turn` and `--multi-turn-sessions` so every landing step records the standard bench, the host share and the Phase 4 multi-turn profile on GPU 0. Token-granular reuse stays inside the Phase 4 boundaries: `turbine_kv::directory` gains L0-only partial entries and a token-by-token comparison of the last block, `planner` a token-granular cap that keeps `MIN_RECOMPUTE_TOKENS`, `hierarchy` a `TailCopy` the scheduler plans as `IterationPlan.tail_copies` and the engine executes with the fork copies through `ModelExecutor::copy_blocks` — the Phase 5 `TpExecutor` already fans that call out to every rank. Token-granular reuse only ever copies from an exact page class (6b's per-class L0 pages: the BF16 recent window, the ladder's L0 rung); a lossy rung-class child is outcome `lossy_source` and the plan falls back to whole-block reuse. The cache-aware policy is one file in the `scheduling_policy` registry, fed by two new `AdmissionInfo` fields; jump-forward adds a multi-token append (`Scheduler::extend`, `IterationOutcome.rolled_back`, `SeqSlice.logits_rows`) and a verification loop in the engine that treats llguidance's forced tokens as a draft. Tasks 13–17 land the 6b/6a recorded follow-ups: the ladder's serving cost, FP8 c1 ITL, the MXFP4 prefill GEMM, OLMoE BF16's margin, and slab mixing.

## Constraints

Copied verbatim from the spec (Constraints):

- Order and landing (decision "Phase 5p"): S-1/S-2 (measure) first, then S-3 only under its rule, then S-4–S-6, then S-7–S-8, then S-9–S-10. Each item lands alone and is measured before the next: `scripts/lab-bench.sh` with golden c1 and the throughput bench per landing step; for S-4–S-8 also the multi-turn profile before and after (`--multi-turn`, and `--multi-turn-sessions 64` for S-7); for S-10 the JSON-schema golden of Interfaces. The follow-ups (S-14–S-18, Tasks 13–17) are independent of S-1–S-10 and land after the 5p core items unless the coordinator schedules one earlier.
- Test tiers (AGENTS.md "Test tiers", decision 2026-09-27): per landing step `scripts/gate.sh`, plus `scripts/lab-test.sh novanas --tier quick` when the change touches GPU-facing code (`turbine-model` for S-9, the `kv_gpu` lab tests for S-5/S-6), plus `scripts/lab-bench.sh --quick` for an iteration read and the full 200-request `scripts/lab-bench.sh` for the landing number; phase exit: `scripts/gate.sh --full`, `scripts/lab-test.sh novanas --tier full`, `scripts/lab-bench.sh --golden16` for both models and the 10-minute `scripts/overload-soak.sh novanas --duration 10m`.
- Test tiers (AGENTS.md "Test tiers", decision 2026-09-27): per landing step `scripts/gate.sh`, plus `scripts/lab-test.sh novanas --tier quick` when the change touches GPU-facing code (`turbine-model` for S-9, the `kv_gpu` lab tests for S-5/S-6), plus `scripts/lab-bench.sh --quick` for an iteration read and the full 200-request `scripts/lab-bench.sh` for the landing number; phase exit: `scripts/gate.sh --full`, `scripts/lab-test.sh novanas --tier full`, `scripts/lab-bench.sh --golden16` for both models and the 10-minute `scripts/overload-soak.sh novanas --duration 10m`.
- Prefix-exact prefill invariance (decisions "Pre-Phase-5 #1 follow-up" and "(A)"): a warm request that reuses a token-granular prefix produces bit-identical tokens and logprobs to the cold run of the same prompt when the reused KV was written by a prefill (the Phase 4 `kv_gpu` rule); Llama's prefill steps keep the invariant GEMM rows, the planner keeps `MIN_RECOMPUTE_TOKENS` = 2, and a warm prefill that starts mid-block must stay prefill-shaped. KV written by decode steps (a previous turn's generated tokens) is reused as in Phase 4: identical within the golden tolerance, not bitwise, because decode steps use the speed-tuned rows.
- KV tiers (decision constraint, extended for the 6b reality): L1 and L2 hold full blocks only; a partial entry never leaves L0; the Phase 4 reuse-evidence gates, `DEMOTION_INFLIGHT` and capacity demotion are unchanged for full blocks. L0 pages may be classed (6b: the BF16 recent window, the ladder's L0 rung); token-granular reuse only ever copies from an exact page class, and the tail copy's private block follows its source's class.
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
- Starting point: Phase 5 (`phase-5-multi-gpu`) and Phases 6a/6b merged into local `main` (6b close 2026-10-04). 6b's landed state this phase builds on: kernel ABI v2.11 (KV transcode + TurboQuant transcode, the mixed-format paged attention, per-class page addressing, the promotion copy kernel, `kv.transfer.promotion_copy` default `kernel`), the compression ladder (`kv.ladder.enabled` default `false`, rung order `l0` → `fp8_e4m3` → `tq4`, opt-in with the documented trade), the BF16 recent window (`kv.recent_window_blocks` default 1, TurboQuant bases only), per-class L0 pages, lower-tier `fp8_e4m3`/`tq4` `supported` / `tq2` `experimental`, tail-tag expiry, and 128 MiB L1 slabs (`kv.cpu.slab_bytes`). Its recorded follow-ups are this plan's Tasks 13–17. If Phase 5/6a/6b are not merged when a task starts, the task stops and the coordinator asks the user.
- Original Phase 5 starting point: `turbine_server::engine::tp::{TpExecutor, WorkerRank}` (`copy_blocks` runs on every rank through a copies-only `StepPlan`), `turbine_distributed::rank::{StepPlan { copies, .. }, StepSeq}` (several tokens per sequence already), `turbine_kv::tier::sharded::ShardedL1Tier`, `KvFormat.shards`, `Scheduler::with_micro_batches` (plans completing out of order, matched by `outcome.iteration`) and group reservations (`Reservation::with_members`).
- Builds and tests run on novanas through `scripts/remote-cargo.sh` (no local target directories); every task ends with `scripts/gate.sh` printing `gate: ok`; GPU-facing tasks add `scripts/lab-test.sh novanas --tier quick`; every task that changes serving code ends with `scripts/lab-bench.sh --model llama` and `--model olmoe` (GPU 0, under `scripts/bench-lock.sh`, labbook set `phase-5p`) and a row in `.procoder/perf-log.md` — one change, then measure; the next task starts only after that row.
- Lab runs of this phase fall under the standing novanas approvals (2026-09-25, 2026-09-26): `lab-test.sh`, `lab-bench.sh`, golden and bench runs while the R9700s are free; if `amd.com/gpu` is held by another workload, stop and ask the user; the overload soak is asked for first.

## Task 0: `lab-test.sh --release` and the release-mode one-GPU full tier (skipped, user decision 2026-09-28)

Files: `scripts/lab-test.sh` (the `--release` option, default for `--tier full`), `scripts/lab/novanas-test-job.yaml` (the release profile in the same cached slot target dirs), `benches/turbine-bench/tests/lab_scripts.rs` (dry-run test), `AGENTS.md`
Interfaces:

- `scripts/lab-test.sh … [--release|--debug]`: `--release` builds the lab tests with `cargo test --release` in the cached slot's `CARGO_TARGET_DIR` (release artifacts beside the debug ones); the default for `--tier full` is `--release`, for `quick` / `perf` the debug profile
  Covers: user decision "Lab-test release profile" (2026-09-28); the Phase 5 exit note (the one-GPU full tier after the pinned bounce buffer was stopped at 33 binaries)
  Depends on: nothing in this phase

- [ ] Write failing test `lab_scripts lab_test_release_is_the_full_tier_default`: a `--tier full` dry run renders `cargo test --release`, `--tier quick` does not, `--debug` overrides, and both use the same slot `CARGO_TARGET_DIR`. Run: `scripts/remote-cargo.sh test -p turbine-bench --test lab_scripts lab_test_release` — expect FAIL
- [ ] Implement the option and the default; document it in `AGENTS.md`.
- [ ] Lab: `scripts/lab-test.sh novanas --tier full` (release) on the Phase 5 merge — expect every binary to pass; record the wall time against the debug run (~40+ min).
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(scripts): lab-test --release, the default for the full tier`

## Task 0b: Prefill overlap — golden and bench confirmation, then the default (skipped, user decision 2026-09-28)

Files: `crates/turbine-core/src/config/parallel.rs` (the default), `crates/turbine-core/src/config/tests.rs`, `.procoder/contract/interfaces.md`, `.procoder/ask/decisions.md`, `AGENTS.md`
Interfaces:

- `parallel.tp_prefill_overlap` defaults to `true` once the block-aligned split (Phase 5 Task 34, 119e5fb) passes golden and shows the gain at tp 2
  Covers: user decision "P5 Task 32 (c): tensor-parallel prefill overlap on by default?" (A now, then B); the Phase 5 exit note
  Depends on: nothing in this phase

- [ ] Lab: `scripts/lab-cluster.sh --bench-lock tp2-novanas` with an uncommitted `scripts/lab/tp2-variant.local` of `set parallel.tp_prefill_overlap=true` and `skip-rest`: Llama golden c1/c16 against the one-GPU capture and the transformers reference under the TP rule, and the standard bench (the earlier gain: +10.2 % tok/s, −38 % TTFT).
- [ ] If golden passes and the gain holds: change the default and the test `config::tests::tp_perf_switch_defaults`, the contract row and `AGENTS.md`; record in `decisions.md` that the condition of "A now, then B" is met, with the run id. Otherwise leave it off and record why.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(core): parallel.tp_prefill_overlap on by default (block-aligned split confirmed)`

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

- [ ] Confirm `main` contains the 6b close (the merge `29f4de10` / `36545afa` lineage, local `main` 2026-10-04) and branch `phase-5p` from it; if not, stop and ask. The baselines run with the defaults of the 6b closing state (the ladder off, `kv.recent_window_blocks: 1`), so every later A/B holds its pressure state constant.
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
- `pub enum PartialOutcome { Reused, NoCandidate, NotResident, PlannerCutoff, LossySource, Disabled }` with `as_str()`
- Exact-class rule (6b interaction): the lookup's partial comparison considers only children resident in an exact page class — the base format when it is lossless-with-matched-scales (`bf16`, `fp8_e4m3`), or the BF16 recent-window class when the base is TurboQuant — and reports `LossySource` for a lossy rung-class child; token ids are compared regardless (a lossy child's ids are still in the directory)
  Covers: S-4 ACs `directory::tests::token_granular_match`, `planner::tests::token_granular_cap`
  Depends on: Task 5

- [ ] Write failing test `directory::tests::token_granular_match` exactly as the spec's S-4 criterion lists (3-block chain + 40-token partial entry; 25-token partial match; divergence at token 70 of a full block; colliding test hasher → 0; a prompt shorter than one block under `ROOT_PARENT`; L1-only → `not_resident`; 64-child scan limit; ninth partial entry evicts the least recently used; a lossy rung-class child reports `LossySource` and yields no partial match). Run: `scripts/remote-cargo.sh test -p turbine-kv directory::tests::token_granular_match` — expect FAIL
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
- 6b interactions: the private tail block is allocated in the source block's page class (per-class addressing; a BF16 recent-window source gets a window-class private block); the ladder never rewrites a referenced block, so `src`/`dst` keep their class until the copy's iteration completes; a partial entry — L0-only, never demoted — is untouched by tail-tag expiry
- The server still passes `partial_prefix_reuse: false` to the hierarchy in this task (Task 8 wires the executor side and turns it on)
  Covers: S-5 ACs `kv_sim partial_tail_reuse`, `kv_sim cancellation_before_tail_copy`, `kv_sim cancellation_releases_kv`, `kv_sim demotion_under_pressure`
  Depends on: Task 6

- [x] Write failing test `kv_sim partial_tail_reuse` as the spec's S-5 criterion lists (multi-turn simulated workload; reuse up to the last written token, capped at `prompt − 2`; one tail copy per reusing request in its first iteration; shared blocks never written; the private block reserved; token-granular `cached_prefix_tokens`; no partial entry demoted with L1 on; every reference back to 0; with the switch off the run equals the Phase 4 result via `SimReport::plan_digest`). Run: `scripts/remote-cargo.sh test -p turbine-scheduler --test kv_sim partial_tail_reuse` — expect FAIL
- [x] Write failing test `kv_sim cancellation_before_tail_copy`: requests cancelled, timed out and SURVIVAL-requeued between attach and their first iteration release `src` and `dst` within one iteration. Run: `scripts/remote-cargo.sh test -p turbine-scheduler --test kv_sim cancellation_before_tail_copy` — expect FAIL
- [x] Implement; `publish_tail` stores only tokens whose KV is written (prompt + generated − the newest, never-fed token) and only for a normal finish; partial entries go through `insert_partial`. The private tail block is allocated in the source's page class, and a pinned kv_sim assertion covers "the ladder does not rewrite a referenced tail block" (red pre-fix for the class allocation if the pool view is still flat).
- [x] Run: `scripts/remote-cargo.sh test -p turbine-kv && scripts/remote-cargo.sh test -p turbine-scheduler` — PASS (whole workspace green)
- [x] Gate: `scripts/gate.sh` — `gate: ok`
- [x] Commit: `feat(kv,scheduler): copy-on-write tail block for token-granular prefix reuse`

## Task 8: Token-granular reuse in the engine, on every rank — landing (S-5, S-6)

Files: `crates/turbine-server/src/engine/loop.rs` (execute `plan.tail_copies` in `fork_copies` through `ModelExecutor::copy_blocks`; `publish_tail` for choice 0 on a normal finish; `tail_copied` after `complete`; `kv_commits` unchanged for full blocks; tests), `crates/turbine-server/src/kv_orchestrator.rs` (`partial_prefix_reuse` from config, `publish_tail`, the KV document fields), `crates/turbine-server/src/metrics.rs` (`turbine_kv_partial_reuse_total{outcome}`, `turbine_kv_partial_reused_tokens_total`, `turbine_kv_partial_entries`), `crates/turbine-kv/src/document.rs` (`partial_prefix_reuse`, `l0.partial_entries`), `crates/turbine-api/tests/api.rs` (`kv_metrics_bounded` extended), `crates/turbine-server/tests/tiny_server.rs` (`tp2_partial_prefix_reuse`), `crates/turbine-server/tests/kv_gpu.rs` (`prefix_reuse_partial_block_matches_cold`; `prefix_reuse_suffix_lengths_match_cold`'s expected `cached_tokens` becomes token-granular), `.procoder/perf-log.md`
Interfaces:

- Tail copies share the fork-copy path: one `copy_blocks(&pool.view(), &src, &dst)` call per iteration with fork pairs first, then tail pairs; the pair is always within one page class (the private block follows its source's); under TP `TpExecutor::copy_blocks` runs it on rank 0 and every `WorkerRank` (local mode) or sends it in the `StepPlan.copies` of that step (static mode). The tail-copy and promotion paths share the device copy machinery 6b introduced (the promotion copy kernel is H2D only; an L0 tail copy stays device-side `copy_blocks`), so no new copy primitive.
- 6b note carried by the t17 handoff: `kv_sim` attach-order nondeterminism (HashMap iteration) — the S-6 fixtures (`tp2_partial_prefix_reuse`, the kv_sim partial-reuse runs) iterate deterministically (sorted or insertion-ordered), the `S-7 treatment` t17 noted, so the tests cannot flake on attach order
- Metric `outcome` values from `turbine_kv::metrics::PartialOutcome`; KV document keys `partial_prefix_reuse`, `tiers[l0].partial_entries`
  Covers: S-5 ACs `engine::r#loop::tests::partial_prefix_reuse_matches_cold`, `api kv_metrics_bounded`, lab `kv_gpu prefix_reuse_partial_block_matches_cold`; S-6 AC `tiny_server tp2_partial_prefix_reuse`; S-4/S-5/S-13 landing measurement
  Depends on: Task 7

- [x] Write failing test `engine::r#loop::tests::partial_prefix_reuse_matches_cold` (CPU backend, tiny Llama): prompts reusing 1, 2, 17, 100 and 127 tokens of a partial block and one diverging inside a full block give tokens and logprobs bitwise equal to `kv.partial_prefix_reuse=false` and to a cold engine, with `cached_tokens` = reused tokens. Run: `scripts/remote-cargo.sh test -p turbine-server --bin turbine-server engine::r#loop::tests::partial_prefix_reuse_matches_cold` — expect FAIL
- [ ] Write failing test `tiny_server tp2_partial_prefix_reuse`: tp 2 on the CPU backend, `local` and `static` (loopback `tcp`), each rank's copy counter shows the tail copy once before the forward, outputs equal tp 1, pools return to the same free count. Run: `scripts/remote-cargo.sh test -p turbine-server --test tiny_server tp2_partial_prefix_reuse` — expect FAIL
- [x] Extend `api kv_metrics_bounded` with the three new families and the two KV document keys
- [x] Write the ignored lab test `kv_gpu prefix_reuse_partial_block_matches_cold` (prefill-written prefixes ending 1, 2, 17, 64, 100, 126, 127 tokens into a block and one divergence inside a full block; greedy 64 tokens, `top_logprobs` 5; text and every logprob bit-equal to cold) and update `prefix_reuse_suffix_lengths_match_cold`'s expected `cached_tokens` to `prompt − 2` capped by the cached length.
- [x] Implement the engine and orchestrator wiring; turn the hierarchy's `partial_prefix_reuse` on from config.
- [x] Run: `scripts/remote-cargo.sh test -p turbine-server && scripts/remote-cargo.sh test -p turbine-api --test api` — PASS (whole workspace green)
- [x] Gate: `scripts/gate.sh` — `gate: ok` (977 tests)
- [ ] Lab: `scripts/lab-test.sh novanas --tier quick` and `scripts/lab-test.sh novanas -- -p turbine-server --test kv_gpu` — expect exit 0 with `prefix_reuse_partial_block_matches_cold`, `prefix_reuse_matches_cold`, `prefix_reuse_suffix_lengths_match_cold` and `nvme_round_trip_matches_cold` ok
- [x] Landing measurement (GPU 0): `scripts/lab-bench.sh --model llama --multi-turn` three times and `scripts/lab-bench.sh --model olmoe` once — expect golden1 PASS, tok/s ≥ 0.98 × the Task 3 baseline, TTFT p50 ≤ 1.10 ×, median `mt_cached` ≥ baseline + 0.01 and median `mt_ttft_later_p50` ≤ 0.95 × baseline; record the lines in `.procoder/perf-log.md`. If a target is missed, keep the code, set `kv.partial_prefix_reuse` false in the lab configs and report to the coordinator before Task 9.
- [x] Commit: `perf(server): token-granular prefix reuse through a copy-on-write tail block`

## Task 9: `cache_aware` scheduling policy — landing (S-7, S-8)

Files: `crates/turbine-scheduler/src/policy/mod.rs` (`AdmissionInfo.{prompt_tokens, cached_prefix_tokens, window}`, `AdmissionKey.affinity`, registry entry, unit test), `crates/turbine-scheduler/src/policy/default.rs` (`affinity: 0`), `crates/turbine-scheduler/src/policy/cache_aware.rs` (new), `crates/turbine-scheduler/src/scheduler.rs` (fill the new `AdmissionInfo` fields from `SchedRequest.estimate` and `SchedulerParams.cache_aware_window` in `submit` and `enqueue`), `crates/turbine-scheduler/src/sim/mod.rs` (`KvSimDriver::with_policy`, `SimReport.admission_inversions`), `crates/turbine-scheduler/tests/kv_sim.rs` (`cache_aware_starvation_bound` with a test-only unbounded variant), `docs/extending/scheduling-policy.md` (the `cache_aware` example, the new fields, the pitfall "the key is computed once at push"), `.procoder/perf-log.md`
Interfaces:

- `AdmissionInfo` gains `pub prompt_tokens: u32, pub cached_prefix_tokens: u32, pub window: Duration`; `AdmissionKey { tier, priority, arrival, affinity: u64, order }`
- `pub struct CacheAwarePolicy;` `name()` = `"cache_aware"`; `admission_key`: preempted → as `DefaultPolicy`; else `tier 1, priority, arrival = Duration::from_nanos(⌊arrival / window⌋ × window), affinity = u32::MAX − cached_prefix_tokens, order = submit_no`; `preemption_rank`, `pick_victim`, `chunk_cap` delegate to `DefaultPolicy`
- Registry: `Registry::new("scheduling_policy", &[&DefaultPolicy, &CacheAwarePolicy])`
- The pressure gate (Phase 3, with 6b's GREEN admission headroom cap at RED 0.90) runs before the policy: `cache_aware` orders what the gate admits and never past it, so the starvation bound is judged on policy order, not on pressure queueing
- `SimReport.admission_inversions: Vec<(RequestId, RequestId)>` — pairs (earlier, later) of equal priority, never preempted, arrival gap ≥ window, where the later was admitted first
  Covers: S-7/S-8 ACs `registry_conformance`, `policy::tests::cache_aware_orders_by_window_then_prefix`, `kv_sim cache_aware_starvation_bound`, `docs_extending`; S-7 lab A/B
  Depends on: Tasks 5, 8

- [x] Write failing test `policy::tests::cache_aware_orders_by_window_then_prefix` (within a window the larger `cached_prefix_tokens` first; across windows arrival order; priority dominates; preempted as `default`; `default`'s keys unchanged, including the existing `default_policy_orders_like_main`). Run: `scripts/remote-cargo.sh test -p turbine-scheduler policy::tests` — expect FAIL
- [x] Write failing test `kv_sim cache_aware_starvation_bound` (seeded; a flood of fully cached requests, one uncached every 50 arrivals, `max_running_requests` 4, 2,000 requests, window 1 s): `admission_inversions` empty and every request completes for `cache_aware`; the test-only unbounded variant (window `Duration::MAX`) has inversions and a larger maximum wait for the uncached requests. Run: `scripts/remote-cargo.sh test -p turbine-scheduler --test kv_sim cache_aware_starvation_bound` — expect FAIL
- [x] Implement; `registry_conformance` runs the Phase 2m suite over both policies with no change to the suite.
- [x] Update `docs/extending/scheduling-policy.md`. Run: `scripts/remote-cargo.sh test -p turbine-scheduler && scripts/remote-cargo.sh test -p turbine-model --test docs_extending && scripts/remote-cargo.sh test -p turbine-server` — expect PASS (the server's `modules.scheduling_policy` shows `cache_aware` under `--set scheduler.policy=cache_aware`)
- [x] Gate: `scripts/gate.sh` — expect `gate: ok`
- [x] Landing (GPU 0): `scripts/lab-bench.sh --model llama` and `--model olmoe` (default policy: standard-bench bound); then three runs each of `scripts/lab-bench.sh --model llama --multi-turn-sessions 64 -- --set scheduler.max_running_requests=16 --set scheduler.policy=cache_aware` and the same with `scheduler.policy=default` on the same day — expect medians `mt_ttft_later_p50` ≤ 0.85 × default, `mt_cached` ≥ default, `mt_tok_s` ≥ 0.98 × default, `mt_ttft_first_p99` ≤ 1.5 × default, golden1 PASS; record in `.procoder/perf-log.md`; the coordinator asks the user whether `cache_aware` becomes the default.
- [x] Commit: `feat(scheduler): cache_aware scheduling policy with a windowed starvation bound`

## Task 10: Multi-token append for running sequences and multi-row logits (S-9)

Files: `crates/turbine-scheduler/src/scheduler.rs` (`Scheduler::extend`, the extension item in the decode stage, `IterationOutcome.rolled_back` in `complete`, `SchedError::NotDecoding`, preemption drops a pending extension, tests), `crates/turbine-scheduler/src/sim/mod.rs` (random extensions in the invariant runs), `crates/turbine-model/src/executor/mod.rs` (`SeqSlice.logits_rows`, `Logits::rows_of`), `crates/turbine-model/src/executor/decoder/mod.rs` and `crates/turbine-model/src/executor/batch.rs` (gather the last `logits_rows` positions for the LM head; such rows are never reduced), `crates/turbine-model/tests/tiny_model.rs` (test), `crates/turbine-server/src/engine/tp.rs` (the leader returns every row)
Interfaces:

- `Scheduler::extend(&mut self, seq: SeqId, tokens: u32) -> Result<(), SchedError>` — the next plan feeds `1 + tokens` positions of the decoding sequence as `BatchKind::Prefill { start, len }` from the budget left after decodes, before prefill chunks; not coverable → a plain decode and the extension dropped
- `IterationOutcome.rolled_back: Vec<(SeqId, u32)>` — `complete` shortens the table by that many tokens and frees blocks left empty
- `SeqSlice.logits_rows: u32` (1 ≤ rows ≤ `q_len`); `Logits::rows_of(&self, seq_index: usize) -> &[f32]` (row-major, position order); existing callers set 1
  Covers: S-9 ACs `scheduler::tests::extend_and_roll_back`, `tiny_model multi_row_logits_match_single_steps`
  Depends on: Task 9

- [x] Write failing test `scheduler::tests::extend_and_roll_back` as the spec lists (6-token item within budget; roll-back of 3 frees an empty block; budget fallback to decode; `NotDecoding` for a prefilling sequence; preemption drops the extension). Run: `scripts/remote-cargo.sh test -p turbine-scheduler scheduler::tests::extend_and_roll_back` — expect FAIL
- [x] Write failing test `tiny_model multi_row_logits_match_single_steps` (both tiny checkpoints, CPU provider: 1 + 7 tokens with `logits_rows` 8 equal the rows of 8 single steps within the Phase 1 CPU tolerance; reduced decode rows in the same batch unchanged; `logits_rows` > `q_len` refused). Run: `scripts/remote-cargo.sh test -p turbine-model --test tiny_model multi_row_logits_match_single_steps` — expect FAIL
- [x] Implement; add random extensions to the simulator's invariant runs (`decode_never_starved`, budgets) and keep them passing. (Both digests re-recorded: `decode_never_starved` 10772755057896660484, `chunk_budget_respected` 15676167015522452859; every other workload's plan unchanged.)
- [x] Run: `scripts/remote-cargo.sh test -p turbine-scheduler && scripts/remote-cargo.sh test -p turbine-model && scripts/remote-cargo.sh test -p turbine-server` — PASS (whole workspace; gate 981 tests)
- [x] Gate: `scripts/gate.sh` — `gate: ok`
- [x] Lab: `scripts/lab-test.sh novanas --tier quick` (GPU-facing: `turbine-model`) — every target ok except the four `kv_gpu` tests already red on the tip before this task (the S-4/S-5 HIP mid-block numerics finding); `scripts/lab-bench.sh --model llama --quick` 879.1 tok/s and `--model olmoe --quick` 636.4 tok/s — golden1 PASS, `failed=0`, no regression
- [x] Mutation-checked (4): the extension budget check dropped, `rolled_back` ignored in `complete`, `logits_rows` clamped to 1 in the head's plan, and the multi-row final-norm sources not advanced per row — each turns one of the two tests red; restored.
- [x] Commit: `feat(scheduler): multi-token append for running sequences and multi-row logits (S-9)`

## Task 11: Forced tokens from the grammar matcher (S-10, matcher side)

Files: `crates/turbine-model/src/structured.rs` (`TokenMatcher::forced_tokens` with a default empty implementation; `LlguidanceMatcher` over `Matcher::compute_ff_tokens`, capped; tests)
Interfaces:

- `fn forced_tokens(&mut self, max: usize) -> Vec<u32>` on `TokenMatcher` (default `Vec::new()`); does not commit anything — the engine commits accepted tokens one at a time with `commit`
  Covers: S-10 AC `structured::tests::forced_tokens_follow_llguidance`
  Depends on: Task 10

- [x] Write failing test `structured::tests::forced_tokens_follow_llguidance` (tiny tokenizer, `small_schema`: equals `compute_ff_tokens` capped by `max`; empty inside a free string; never re-tokenizes the last committed token; calling it twice without a commit returns the same tokens). Run: `scripts/remote-cargo.sh test -p turbine-model structured::tests::forced_tokens_follow_llguidance` — expect FAIL
- [x] Implement over llguidance 1.8.0's `Matcher::compute_ff_tokens` (no new dependency); a run is cut at EOS and capped by `max`, and never advances the matcher.
- [x] Run: `scripts/remote-cargo.sh test -p turbine-model` — PASS (156 tests; the whole crate)
- [x] Gate: `scripts/gate.sh` — `gate: ok` (986 tests with Task 12)
- [x] Mutation-checked: the override dropped to the default (empty) turns `forced_tokens_follow_llguidance` red; restored.
- [x] Commit: `feat(model): forced tokens from the llguidance matcher`

## Task 12: Verified jump-forward in the engine — landing (S-10)

Files: `crates/turbine-server/src/engine/loop.rs` (after a constrained choice samples: ask `forced_tokens`, cap by `jump_forward_max_tokens`, `max_tokens`, `model.max_seq_len`; `Scheduler::extend`; `logits_rows = k + 1` on that slice; on collect verify row by row through `ActiveRequest::step`, stop at the first disagreement, report `rolled_back`; tests), `crates/turbine-server/src/engine/requests.rs` (`ActiveRequest::step_forced(choice, rows, forced) -> ForcedOutcome`: runs the normal step per row, keeps the sampler's uniform stream, stop strings, EOS, `max_tokens` and tool-call holding per accepted token), `crates/turbine-server/src/metrics.rs` (`turbine_jump_forward_tokens_total{outcome}`, `turbine_jump_forward_steps_total`), `crates/turbine-server/tests/tiny_server.rs` (`jump_forward_matches_token_by_token`), `crates/turbine-server/tests/lab_openai.rs` (`json_schema_jump_forward_matches`), `.procoder/perf-log.md`
Interfaces:

- `pub struct ForcedOutcome { pub accepted: u32, pub next: Option<SampledToken>, pub finished: Option<FinishReason> }`
- Engine flow per constrained choice: sample `t` → `commit(t)` → `forced_tokens(max)` → `Scheduler::extend(seq, k)`; next step feeds `[t, f1..fk]` with `logits_rows = k + 1`; row j is stepped with the mask and sampler as a token-by-token step; accepted while the sampled token equals `f(j+1)`; at the first disagreement the model's token is taken and `rolled_back = k − accepted`; all accepted → row k yields the next token; `kv_commits` counts kept tokens only
- Constrained choices keep finishing serially under overlap scheduling (Phase 2c S-16), so a jump-forward step is never launched ahead
  Covers: S-10 ACs `tiny_server jump_forward_matches_token_by_token`, lab `lab_openai json_schema_jump_forward_matches`; S-13 landing
  Depends on: Task 11

- [x] Write failing test `tiny_server jump_forward_matches_token_by_token` as the spec lists (JSON schema, `json_object`, tool call; greedy with and without `top_logprobs`, seeded `temperature` 0.8, a stop string inside a forced span, `max_tokens` cutting a span; identical streams, logprobs and finish reasons with `structured_output.jump_forward` true and false; `accepted` > 0; a test grammar forcing a non-canonical split is rejected at that token). Run: `scripts/remote-cargo.sh test -p turbine-server --test tiny_server jump_forward_matches_token_by_token` — expect FAIL
- [x] Write the ignored lab test `lab_openai json_schema_jump_forward_matches`: the four JSON cases of `tests/golden/tools/requests.jsonl` at temperature 0 with `logprobs` and `top_logprobs` 5, jump-forward on and off (two servers, `--set structured_output.jump_forward=false` for the second); identical token ids (divergence excused only at a near-tie under the golden token rule of `tests/golden/llama-3.2-3b-instruct/tolerance.json`), logprobs within its strict bounds, schema-valid content; print `forward_steps_per_token` and wall time per case for both settings and assert ≤ 0.90 × steps and ≤ 1.00 × time summed over the four (re-bound to the measured workload, decision 2026-10-06 A; originally 0.8 × / 0.9 ×).
- [x] Implement; the run travels with the plan item (`BatchItem::draft`), so the batch and the verification feed exactly what the extension asked for even when the choice's matcher state moves on (a `NeedsHost` rebuild re-reads the choice — the bug the plan-item copy kills). `plan_extensions` reserves a logits row per remaining decode candidate, so an extension cannot push the batch past `max_running_requests` rows; a jump-forward step is never launched ahead (`Build::NeedsHost` while an iteration is in flight) and its iteration completes serially (`finish_forced`), whose outcome carries the iteration number so `complete` releases the busy mark. KV commits cover kept tokens only (`kv_commits` caps at prompt + generated).
- [x] Run: `scripts/remote-cargo.sh test -p turbine-server && scripts/remote-cargo.sh test -p turbine-model` — PASS (whole `turbine-server`, `turbine-model`, `turbine-scheduler`; including `response_format_json_schema`, `matcher_masks_each_choice_and_fails_alone`, the overlap tests, `preempted_output_unchanged`)
- [x] Mutation-checked: the agreement check dropped (every forced token accepted) turns `jump_forward_rolls_back_a_corrupted_run` red; restored.
- [x] Gate: `scripts/gate.sh` — `gate: ok` (986 tests)
- [x] Lab: `scripts/lab-test.sh novanas --tier quick` and `scripts/lab-test.sh novanas -- -p turbine-server --test lab_openai` — correctness holds (identical tokens, strict logprob bounds, schema-valid; `tools_and_json_schema` ok); steps 0.899 ×, wall 0.989 × — short forced runs on the Llama tokenizer missed the original 0.8 × / 0.9 × targets; user decision 2026-10-06 A keeps jump-forward on and re-bounds them to ≤ 0.90 × / ≤ 1.00 × (perf log 5p, S-10 lab legs). The quick tier's other failure, `olmoe_rows_are_batch_invariant`, is outside this path (reported).
- [x] Landing (GPU 0): `scripts/lab-bench.sh --model llama --quick` 875.3 tok/s and `--model olmoe --quick` 630.2 tok/s, golden1 PASS, failed=0 — no regression
- [x] Commit: `feat(structured-output): verified jump-forward in the engine (S-10)`

## Task 13: Ladder serving cost — instrument, then the measured fix for the ORANGE dwell (S-14)

Files: `crates/turbine-server/src/kv_orchestrator.rs` (the per-action time counter, the ORANGE cause breakdown, the budget application), `crates/turbine-kv/src/hierarchy.rs` (the budget in the ladder sweep), `crates/turbine-core/src/config/kv.rs` (`kv.ladder.max_rewrites_per_tick`, validation), `crates/turbine-core/src/config/tests.rs`, `crates/turbine-kv/src/hierarchy.rs` tests, `.procoder/perf-log.md`, `.procoder/ask/decisions.md`
Interfaces:

- Consumes Task 3's ladder-off baselines; conditional like Task 4: the instrumented profile chooses the fix. Instrumentation: `turbine_kv_ladder_action_seconds_total{tier,action}` (per-action time) and the pressure document's ORANGE entry gaining a cause breakdown (`ladder_rewrites` against `other`)
- Conditional fix (the profile chooses): (a) `kv.ladder.max_rewrites_per_tick: u32` (default 8, 1..=64) bounding the rewrites the orchestrator submits per tick, overflow deferred to later ticks; (b) ORANGE hysteresis while a rewrite burst drains; (c) rewrites onto the copy kernel / idle windows instead of the rewrite lanes. The A/B target is the spec's S-14 bullet (ladder-on tok/s ≥ 0.90 × the ladder-off arm, ORANGE samples ≤ 2 × off, recompute reduction retained ≥ 15 %); the ladder stays opt-in and the item closes on the user's read of the A/B
  Covers: S-14 ACs (host budget test + manual A/B)
  Depends on: Task 2, Task 3

- [ ] Instrument: add `turbine_kv_ladder_action_seconds_total{tier,action}` and the ORANGE cause breakdown; run: `scripts/remote-cargo.sh test -p turbine-server` — expect PASS (reporting only)
- [ ] Run the instrumented A/B (GPU 0): `scripts/lab-bench.sh --model llama --multi-turn` with the ladder on and with the ladder off (`--set kv.ladder.enabled=…`), three runs each — the per-action times and the ORANGE breakdown name the dominant cost; the lines go into `.procoder/perf-log.md`
- [ ] Fix per the profile, test first: write failing test `turbine-kv hierarchy::tests::ladder_rewrite_budget_bounds_actions_per_tick` (at most `max_rewrites_per_tick` rewrites submitted per tick, overflow deferred, no budget = as before). Run: `scripts/remote-cargo.sh test -p turbine-kv hierarchy::tests::ladder_rewrite_budget_bounds_actions_per_tick` — expect FAIL
- [ ] Implement the fix the profile named (the budget, the hysteresis or the copy-kernel path); keep the 6b ladder tests (`ladder_under_pinned_pressure`, `ladder_l0_under_pinned_pressure`) passing unchanged
- [ ] Rerun the A/B (GPU 0): meets the S-14 target, or the item closes on the user's read of the numbers (a decisions entry records it either way)
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `perf(kv): ladder serving cost — the instrumented fix for the ORANGE dwell`

## Task 14: FP8 c1 ITL — profile, then the measured fix (S-15)

Files: `crates/turbine-kernels/src` and `kernels/rocm/src` (per the profile), `crates/turbine-server/src/metrics.rs` (a host-side fix), `.procoder/perf-log.md`, `.procoder/ask/decisions.md`
Interfaces:

- The 6a record is the miss to close: c1 ITL 0.846× (FP8-dynamic) / 0.855× (per-tensor) BF16 against 6a's ≤ 0.75× target; `llama-fp8-block` measured 8.43 ms c1 ITL (112.7 tok/s c1)
- The fix follows the profile (the S-1 `stage-share` at c1, `hip_ops` decode GEMM timings for m = 1–16): the m ≤ 16 qgemm decode tiles, graph-capture coverage of the FP8 paths, or the per-token activation quantize. No numerics change: every candidate is judged by golden c1 strict bounds and the existing tiny_model / hip_ops match tests
  Covers: S-15 AC (manual measurement)
  Depends on: Task 2, Task 3

- [ ] Profile (GPU 0): `scripts/lab-bench.sh --model llama-fp8 --concurrency 1 --quick` and `--model llama-fp8-block --concurrency 1 --quick`, plus the same-day BF16 c1 run, with the stage-share; `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops` for the decode GEMM timings m = 1–16; paste the tables into `.procoder/perf-log.md`
- [ ] Fix per the profile, test first: a host-side fix gets a failing host test first; a kernel-side change is gated by the lab timing comparison and golden c1 strict bounds; the perf log holds the before/after tables
- [ ] Landing (GPU 0): `scripts/lab-bench.sh --model llama-fp8 --concurrency 1 --quick` and `--model llama-fp8-block --concurrency 1 --quick` — ITL p50 ≤ 0.75 × the same-day BF16 c1 ITL, golden c1 PASS, the standard bench within the no-regression bound; a miss needs a user re-bound on the profile's evidence (decisions entry)
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `perf(kernels): FP8 c1 decode — the profiled fix`

## Task 15: MXFP4 prefill GEMM at hipBLASLt speed (S-16)

Files: `kernels/rocm/src/qgemm_mxfp4.hip` (the tile), `crates/turbine-kernels/src` (the registry, the timing test), `crates/turbine-kernels/tests/hip_qgemm_mxfp4.rs` (lab timing test), `.procoder/perf-log.md`, `.procoder/ask/decisions.md`
Interfaces:

- Reuse-first: evaluate CK `gemm_microscale_*` and llama.cpp's HIP MXFP4 (6a measured llama.cpp prefill ≈ ours) against the existing `turbine_hip_mxfp4` tile on the 6a shapes (3b.qkv / 3b.o / 3b.gate_up / 3b.down, m = 2048) before any tile work; record the evaluation in decisions.md
- Bound (user decision 2026-10-06 A): the MXFP4 prefill GEMM at m = 2048 ≤ 1.45 × the BF16 prefill GEMM per shape (no-regression bound; the original 1.10 × target missed: no provider within 10 %, own Prefill tile 1.0–1.44 ×); served TTFT recorded, not gated; `mxfp4` stays `experimental`
- Every change keeps prefill rows batch-invariant (`qgemm_mxfp4_prefill_rows_are_batch_invariant` stays green — the Phase 4 bitwise prefill rule) and golden within the slug tolerance
  Covers: S-16 ACs (lab timing test + served TTFT)
  Depends on: Task 3

- [x] Provider evaluation (GPU 0, under the bench lock): CK `gemm_microscale_*` and llama.cpp mmq against the own tile on the 6a shapes; record in decisions.md; a provider that wins by ≥ 10 % is registered (one file per provider, the registry entry, the conformance suite) instead of tile work
- [x] Write failing lab timing test `hip_qgemm_mxfp4 qgemm_mxfp4_prefill_within_hipblaslt_bound` (per layer shape ≤ 1.45 × the BF16 prefill GEMM time measured in the same run; first written at 1.10 ×). Run: `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_qgemm_mxfp4` — expect FAIL (tile unchanged)
- [x] Implement (provider or tile); `qgemm_mxfp4_prefill_rows_are_batch_invariant` stays green — own Prefill tile (no provider within 10 %); the timing test still FAILS (1.0–1.44×, 7 of 8 shapes over 1.10); open re-bound in decision "P5p: MXFP4 prefill GEMM — provider re-evaluation, the own Prefill tile, and the S-16 miss"
- [x] Landing (GPU 0): `scripts/lab-bench.sh --model llama8b-mxfp4 --concurrency 1 --quick` and the standard bench: TTFT p50 recorded against the same-day BF16 8B TTFT (1.25 × c1, 1.15 × c16; not gated, decision A), golden c1 unchanged; perf-log rows; timing test passes under the 1.45 × bound
- [x] Gate: `scripts/gate.sh` — expect `gate: ok`
- [x] Commit: `perf(kernels): MXFP4 prefill at hipBLASLt speed` (landed as the Prefill tile commits and `test(kernels): MXFP4 prefill timing bound re-bound to 1.45x`)

## Task 16: OLMoE BF16 back inside the 0.98× bound with margin (S-17)

Files: `crates/turbine-server/src` (per the profile; the 6b serving fixes are the candidates), `.procoder/perf-log.md`, `.procoder/ask/decisions.md`
Interfaces:

- The 6a exit watch item: OLMoE BF16 at 0.982× the phase start (602.5 vs 613.9 tok/s, the ≥ 0.98× bound holding by 0.002). Target: median ≥ 608 tok/s (0.99 × 613.9) with golden c1/c16 PASS; the profile names the fix
  Covers: S-17 AC (manual measurement)
  Depends on: Task 2, Task 3

- [x] Profile (GPU 0): three `scripts/lab-bench.sh --model olmoe` runs plus a c16 `stage-share`, compared with the 6a-exit numbers (613.9 tok/s, 24.3 ms ITL, 118 ms TTFT) — the `BENCH` and stage deltas name the stage or the serving change; the lines go into `.procoder/perf-log.md` (perf log "S-17: OLMoE BF16 margin": ITL 24.3 ms and host share 0.013 match the 6a exit and S-2; no stage regressed)
- [x] If the delta is the GREEN admission headroom cap's queueing (6b's cap at RED 0.90 working as designed), record the re-bound in a decisions entry for the user; the watch item closes there. Otherwise fix what the profile named and land it alone; the standard bench keeps OLMoE ≥ 0.98 × the S-2 baseline (no delta to fix on the merged stack, 97050b3c: no re-bound and no code change)
- [x] Landing (GPU 0): three `scripts/lab-bench.sh --model olmoe` runs — median ≥ 608 tok/s with golden c1/c16 PASS; perf-log rows (609.5 / 614.8 / 608.8, median 609.5, golden c1/c16 PASS ×3)
- [x] Gate: `scripts/gate.sh` — expect `gate: ok`
- [x] Commit: `perf(server): OLMoE BF16 margin restored` (or the recorded re-bound) (landed as `docs(perf)`: no server change was needed)

## Task 17: Slab mixing — compressed lower-tier copies actually store (S-18)

Files: `crates/turbine-kv/src/tier/mod.rs` (the drain-and-reformat in `take_slot`'s rung path), `crates/turbine-kv/src/tier/sharded.rs` (the drain bounded per tick), `crates/turbine-core/src/config/kv.rs` (`kv.cpu.slab_drains_per_tick`, validation), `crates/turbine-core/src/config/tests.rs`, `crates/turbine-kv/src/tier` tests, `crates/turbine-server/src/metrics.rs` (the reformat counters), `.procoder/perf-log.md`, `.procoder/ask/decisions.md`
Interfaces:

- `kv.cpu.slab_drains_per_tick: u32` (default 1, 0..=8; 0 = off, restoring 6b's fallback-only behavior) — how many slabs the tier may drain and reformat per tick when the rung format has no free slot (`rung_no_slot`) and pressure asks for the rung
- The drain evicts or re-stores a slab's blocks and reformats the slab to the rung's slot size; evictions, re-stores and reformats count in `turbine_kv_slab_reformat_total{tier,action}`; the 6b eviction fallback stays for a tier that cannot drain
  Covers: S-18 ACs (host test + manual A/B)
  Depends on: Task 3

- [ ] Write failing test `tier::tests::drained_slab_reformat_stores_rung_format`: two slabs full of `l0` blocks, pressure asking for a rung format, the drain stores rung-format copies with evictions/re-stores counted, and budget 0 restores the fallback-only behavior. Run: `scripts/remote-cargo.sh test -p turbine-kv tier::tests::drained_slab_reformat_stores_rung_format` — expect FAIL
- [ ] Implement the drain-and-reformat in the slab tier (`take_slot` asks the drain path before the fallback), bounded by `kv.cpu.slab_drains_per_tick`; keep the 6b tier tests (`l1_default_slab_size_stores_a_new_format_block_beside_occupied_old_format_slots`, the fallback test) passing
- [ ] Landing (GPU 0): the ladder multi-turn A/B (the Task 13 recipe) with the mixing path on — a nonzero rung-format share of stored L1 copies at the final scrape, the reformat counters recorded, the standard bench within the no-regression bound, golden1 PASS; the drain-vs-compaction choice is recorded in decisions.md
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(kv): slab drain-and-reformat — compressed copies actually store`

## Task 18: Decode bandwidth — batch-1 kernels toward peak memory bandwidth (S-19)

Files: `crates/turbine-kernels/` (batch-1 decode path: fused rmsnorm+quantize+GEMV or per-shape GEMV selection; the gemm_tune table grows a batch-1 shape), `crates/turbine-model/src/` (the decoder's batch-1 kernel choice), `kernels/rocm/` (any new fused kernel + the impl-table entry), `benches/turbine-bench/` (the bandwidth report), `.procoder/perf-log.md`, `.procoder/specs/phase-5p-serving-efficiency.md` (S-19)
Interfaces:

- `S-19`: the batch-1 decode step's effective weight-read bandwidth (weights bytes ÷ step time) rises from the measured ~500 GB/s (0.56 × peak) toward peak; reported as `host_share`-adjacent evidence and a bandwidth figure per model; outputs bit-identical or within the golden tolerance
  Covers: the measured roofline gap (S-2's c1 baseline: 77.8 tok/s ≈ 500 GB/s vs ~900 GB/s peak; host share 0.37 %, so the gap is inside kernel execution)
  Depends on: Task 3

- [ ] Profile: with the S-2 stage-share machinery plus rocprofv3, account the batch-1 decode step per kernel — GEMV vs norm/rope/attention vs inter-kernel gaps — on Llama and OLMoE; identify where the 45 % goes
- [ ] Fix the dominant cause(s) test-first: fused batch-1 kernels (e.g. rmsnorm+quantize into the GEMV), per-shape GEMV selection through the gemm_tune table, fewer passes over activations; every output change stays within the golden tolerance (compare runs bit-identical preferred)
- [ ] Landing (GPU 0): the bandwidth figure before/after per model, c1 tok/s before/after against the 77.8/126.0 baselines, golden1 PASS; the standard c16 bench within the no-regression bound
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`

## Task 19: Phase exit — full suites, multi-GPU, soak, docs

Files: `AGENTS.md` (Commands: `stage-share`, the new `lab-bench.sh` flags, `scheduler.policy: cache_aware`, `kv.partial_prefix_reuse`, `structured_output.jump_forward*`; project state: Phase 5p done), `examples/turbine.yaml` (final comments), `.procoder/perf-log.md` (phase summary row), `.procoder/specs/phase-5p-serving-efficiency.md` (tick criteria with evidence)
Interfaces:

- Consumes every earlier task; produces the phase-exit evidence
  Covers: S-11 AC `--check-config`; S-6 phase-exit lab (`lab-cluster.sh tp2-novanas`); S-13 phase-exit AC; the follow-ups' closure evidence (S-14–S-18)
  Depends on: Tasks 1–17

- [ ] Run: `cargo run -p turbine-server -- --config examples/turbine.yaml --check-config --set scheduler.policy=cache_aware --set kv.partial_prefix_reuse=false --set structured_output.jump_forward=false` — expect `config ok`; with `--set scheduler.policy=lpm` — expect exit 2 naming `default` and `cache_aware`
- [ ] Run: `scripts/gate.sh --full` — expect `gate: ok`
- [ ] Run: `scripts/lab-test.sh novanas --tier full` — expect exit 0
- [ ] Run: `scripts/bench-lock.sh scripts/lab-cluster.sh tp2-novanas` — expect exit 0 with golden c1 PASS and `kv_gpu prefix_reuse_partial_block_matches_cold` bit-exact at tp 2
- [ ] Run (GPU 0): `scripts/lab-bench.sh --model llama --golden16 --multi-turn` and `scripts/lab-bench.sh --model olmoe --golden16` — expect golden1 and golden16 PASS within the standard-bench bound
- [ ] Ask the user, then run `scripts/overload-soak.sh novanas --duration 10m` — expect `verdict.json` passing (and again with `cache_aware` if the user made it the default)
- [ ] Update `AGENTS.md`, `examples/turbine.yaml` and the perf log; tick the spec's criteria with evidence.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `docs: phase 5p serving efficiency — commands and acceptance evidence`
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
