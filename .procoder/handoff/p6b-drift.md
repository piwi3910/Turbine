# Handoff: p6b-drift (decision "6b: OLMoE tq4 — lossless last block in eviction order; step-time drift at startup", 2 A)

Branch `p6b-drift` from `p6b-stack` aa42093. Numbers and run ids: `.procoder/perf-log.md`, Phase 6b, "Step-time drift at startup".

## Commits

- 25fc430 `feat(engine)`: `decode_step` debug trace. Each pure decode step logs its rows, context tokens, seconds, `calm` and judged
  ratio (`DecodeStepWindow::observe` returns the ratio). Enable it with `logging.level: info,turbine_server::engine=debug`.
- 5924e3f `fix(reliability)`: judged steps age out of the drift window after `STEP_MAX_AGE` (10 s). `StepSample.at` is new, and the call
  is now `DecodeStepWindow::p95(now)`. The engine and the overload simulator pass their clock. Tests:
  `step_window::tests::judged_steps_age_out` and `controller::tests::a_drift_spike_does_not_latch_once_its_shape_stops_running` (the stuck
  RED trace through the real window and controller).
- 66994e1 `fix(reliability)`: the p95 needs `MIN_JUDGED_STEPS` (20) judged steps. Test
  `step_window::tests::one_slow_step_in_a_short_window_is_not_drift`. The spec signal table and the contract §8.3 note are amended.
- Docs: the perf log entry and this handoff.

Mutations: with the age filter off, both age tests FAIL; with the minimum off, the short-window test FAILS. Gate:
`scripts/gate.sh --base aa42093`: `gate: ok passed=834`. Soak 10 min on 66994e1 (serve run 1002071230-3f047df9): PASS 8/8, with no second of the timeline dominated by `step_time_drift`.

## Root cause (short)

- At startup, a judged bucket's first steps made a window of under 10 steps. Its "p95" is the maximum, so one slow step (rare, about 1 in
  11,500, up to 7.3×) read as RED drift.
- Above GREEN, no baseline is learned, and the one-at-a-time admissions decoded shapes never seen while calm. No step was judged any more,
  the window kept the spike, and RED latched for the whole run.

## Open: decision for the lead / user

Late-run `step_time_drift` (OLMoE `l0` YELLOW at +23–25 s in every lastblock A/B run, Llama at +29–38 s) is genuine. Decode steps run
1.5–2.4× slower while L1 → L0 promotions are in flight: every judged step at ≥ 1.5× overlapped a tier copy, against 2–3 % of normal steps.
It shows more under `l0` (16 MiB raw blocks) than under `tq4` (4.5 MiB). "Stays YELLOW" is the 10 s dwell outlasting the run, not a latch.
The controller reads the hierarchy's own copy traffic as device slowdown, and YELLOW's reclaim adds copies.

- A) Leave it. The signal is right that steps are slower. The A/B harness should then compare formats only on runs that stay GREEN, or
  note the state.
- B) Do not judge decode steps that overlapped a KV tier copy, as prefill steps already are not (the engine knows the copies in flight).
  The step still counts for throughput. Drift then measures the device, not our own transfers.
- C) B plus a perf item for the KV owner: find out why an H2D promotion on its own stream slows decode compute 1.5–2× (e.g. the stream or
  event a decode waits on, or pinned-memory DMA contention), since that cost also lands in ITL.

Recommended: C. B keeps the circuit's and the pressure controller's view of the device clean. The slowdown itself costs ITL and belongs to
the KV builder, not to `turbine-reliability`.

## Tools (novanas `scratch/p6b-drift/`)

- `drift.sh <outdir> <fmt>:<run>…`: env `MODEL=olmoe`, `BIN=<turbine-server>` (default `bin-25fc430`, the old window), `EXTRA=<--set …>`
  and `BURST=<delay>:<secs>:<procs>` (a CPU burst).
- `poll.py` polls the pressure document every 100 ms. `ana.py` summarises the `decode_step` trace, `polls.py` the state and drift
  timeline, and `corr.py` the steps against `kv_copy_stages`.
- Binaries: `bin-25fc430` (old window) and `bin-66994e1` (fixed). Results: `r1/`, `burst1/`, `forced-{old,fix}/` (thresholds forced to
  1.2/1.25/1.3; neither latched, so this does not discriminate), `fix/`, `olmoe-{old,fix}/`.
- Run them under `flock -x port18000.gate flock -x port18000.lock flock -x bench.gate flock -x bench.lock`. Replace `drift.sh` only with
  `mv` while a series is running, because bash reads a running script incrementally.
- The soak client wrapper (`SOAK_BENCH`) is in the session scratchpad as `soak-bench.sh`. It runs turbine-bench on novanas and copies the
  pressure timeline back.
