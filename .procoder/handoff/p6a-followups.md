# Handoff: p6a-followups (lead r12 small follow-ups)

Branch `p6a-followups`, from the integration tip `aecb157`. Status: done, no GPU/lab work needed
(cpu backend only). Two commits, one per item.

## Item 1 — join the TP worker-rank threads at exit (bounded)

Commit `d2033fd fix(server): join tensor-parallel worker threads at exit, bounded`.

The gap: `startup::serve` spawns a `"turbine-rank-worker"` OS thread per local `static`-mode
worker rank (`replica.worker`, one process holding a tensor-parallel rank > 0) and never kept
its `JoinHandle`, so on a clean shutdown the process could exit while that thread was still
releasing its device resources — the exact race `ENGINE_STOP_LIMIT`'s comment already worried
about for the engine threads, just not closed for this second kind of thread. Note: `RankRuntime`
already joins its *own* internal worker threads inside `RankRuntime::shutdown` (`local` and
`static` modes, `crates/turbine-distributed/src/rank.rs`) — that part was fine. The unjoined
thread was the one `startup.rs` itself spawns to run `engine::tp::run_static_worker`.

Fix (`crates/turbine-server/src/startup.rs`):
- Collect `(rank, JoinHandle)` for every spawned `"turbine-rank-worker"` thread into
  `rank_workers`, keyed by the rank from its `Hello` (`engine::tp::StaticWorker::rank()`, a
  small new accessor in `crates/turbine-server/src/engine/tp.rs`).
- New `join_rank_workers(workers, limit)`: polls `JoinHandle::is_finished()` (never blocks on a
  stuck one), joins a finished one right away, and on timeout logs one WARN
  `event="tp_worker_join_timeout"` naming the ranks still running, then returns regardless.
- New `join_engines_and_workers(engines, workers)`: `join_engines` (unchanged, its own
  `ENGINE_STOP_LIMIT`-bounded wait), then `join_rank_workers` with whatever of that same
  `ENGINE_STOP_LIMIT` budget is left — the two joins together never exceed the one 10 s budget,
  as the brief asked ("bounded by 10 s in total"). Both clean-exit branches of `serve`'s
  `tokio::select!` (`server.await` returning `Ok(())`, and `connections_closed`) now call
  `join_engines_and_workers` instead of `join_engines`.
- Updated the module doc (shutdown order) and the comment above `runtime.shutdown_background()`
  to mention the worker-rank join.

Test: `crates/turbine-server/src/startup.rs` `tests::join_rank_workers_bounded` (cpu backend, no
GPU — plain `std::thread`s stand in for the rank threads). (a) three workers that finish after a
30 ms sleep: asserts every one's "ran" flag is set *before* `join_rank_workers` returns (racy
otherwise, since the threads finish on their own regardless of whether they're joined — the delay
makes a missing join fail this deterministically). (b) one worker stuck 5 s against an 80 ms
bound: asserts the call still returns in well under the bound, and the captured tracing output
(`tests::capture_tracing`, the `tracing_subscriber::fmt` + buffer pattern used elsewhere in the
tree, e.g. `turbine_kv::test_log`) contains `tp_worker_join_timeout` and the rank number.
`tracing-subscriber` added as a turbine-server dev-dependency for this (`Cargo.toml`).

**Mutation check (done manually, not left in the tree):** replaced `join_rank_workers`'s body
with `let _ = (workers, limit);` (a no-op) — `cargo test -p turbine-server
startup::tests::join_rank_workers_bounded` then failed at the part-(a) assertion (`worker rank 1
was not joined before join_rank_workers returned`), confirming the test catches the join being
removed. Restored the real implementation and reran green before committing.

## Item 2 — log and report the resolved rope configuration

Commit `e5c28fa feat(server): report the resolved rope configuration at startup and in status`.

- `crates/turbine-model/src/config.rs` (lead-owned, kept minimal): added `pub struct
  RopeSummary { theta, rope_type, factor, attention_factor }` and one accessor,
  `ModelArchConfig::rope_summary()`, matching on the existing `RopeScaling` enum (`default` /
  `llama3` / `yarn`; `factor` is `None` for `default`, `attention_factor` is `Some` only for
  YaRN). Added `Serialize` to the `serde` import and derived it on `RopeSummary` so it can go
  straight into the status JSON. Re-exported from `crates/turbine-model/src/lib.rs`.
- `crates/turbine-server/src/model.rs`: new `log_rope_config(&ModelArchConfig)`, called right
  after `load_model_config_with` resolves `arch` in `prepare_with` (before any further use), logs
  one INFO `event="rope_config"` with `rope_theta`, `rope_type`, `factor` and
  `attention_factor` (the last two absent when `None` — `tracing`'s `Option<T>: Value` impl).
- `crates/turbine-server/src/backend.rs`: `ModelStatus` gained a `rope: RopeSummary` field
  (nested under `model`, not top-level like `quantization`, per the brief); `ModelBackend` keeps
  `rope: RopeSummary` set once at construction (`ModelBackend::new`, `model.arch.rope_summary()`)
  and copies it into the status document in `Diagnostics::status`.
- `.procoder/contract/interfaces.md`: added a paragraph at the end of §26 (Phase 6 additions)
  naming `RopeSummary`, `rope_summary()`, the `rope_config` log event and `/turbine/v1/status`
  `model.rope`.

Tests:
- `crates/turbine-model/src/config.rs` `tests::rope_summary_reflects_scaling`: the Llama fixture
  (`llama3`, theta 500000, factor 32) via `load_model_config`, the OLMoE fixture (no
  `rope_scaling` → `default`, no factor) via `load_model_config`, and a YaRN override
  (`model.rope_scaling` = `{rope_type: yarn, factor: 4.0}`) via `load_model_config_with` — checks
  `rope_summary()`'s `attention_factor` against the same config's own resolved
  `RopeScaling::Yarn::attention_factor`.
- `crates/turbine-server/tests/tiny_server.rs` `completions_stream_and_non_stream`: extended the
  existing `/turbine/v1/status` assertions with `status["model"]["rope"]` — the tiny Llama
  fixture carries `llama3` scaling like the real checkpoint (`llama_config_json` in
  `turbine_model::testing::tiny`), so the expected values are theta 10000, `rope_type: llama3`,
  `factor: 8.0`, `attention_factor: null` (verified this against the tiny fixture's *actual*
  values by running the test locally with `kv.cpu.max_bytes` overridden down, see note below —
  my first guess, theta 1e6/`default`, was wrong and the test caught it).

## A pre-existing, unrelated local-Mac limitation

`cargo test -p turbine-server --test tiny_server` fails locally on this workstation (confirmed
with `git stash` against the unmodified `aecb157` tip too — not caused by anything in this
branch): the harness's default config leaves `kv.cpu.max_bytes` at its 64 GiB default, and this
Mac has 24 GB total RAM (`reliability.memory.host_reserve_bytes` 8 GiB default), so
`kv.cpu.max_bytes` fails `validate_host` with exit 2 before any server starts. To verify the new
`rope` status field by hand, I ran `completions_stream_and_non_stream` once with a temporary,
uncommitted `Setup { kv_extra: "  cpu:\n    max_bytes: 1MiB\n", ..Setup::default() }` swapped in
for its `TinyServer::start("")` — green, and it printed the real status document I used to fix
the test's expected values — then reverted that swap before committing. `scripts/gate.sh` runs on
`novanas`, which has plenty of RAM, so this should not show up there; worth keeping in mind if a
future run on a small-RAM host needs `turbine-server`'s test suite.

## Gate

`scripts/gate.sh --base aecb157` (full workspace: `Cargo.lock` changed, the new
`tracing-subscriber` dev-dependency): `gate: ok crates=all passed=800 failed=0`.

## Open

Nothing open; no design questions.
