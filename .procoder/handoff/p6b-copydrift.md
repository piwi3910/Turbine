# Handoff: p6b-copydrift (decision "6b: step-time drift during KV promotions" C; "6b: OLMoE tq4 after the last-block change" A)

Branch `p6b-copydrift` from `p6b-stack` ce88483. Numbers: `.procoder/perf-log.md`, Phase 6b, "KV copies and decode steps". Not pushed.

## Done

- 05471e6 `fix(reliability)`: decode steps that overlap a KV tier copy are not judged (or learned) by the drift window.
  `StepSample.copy_overlap`, `DecodeStepWindow::kv_copy_excluded`, `kv_orchestrator::CopyMark` (in flight at launch or
  collection, or a transfer poll in between found or left one in flight), `turbine_decode_steps_unjudged_total{reason="kv_copy"}`,
  and the `decode_step` trace fields `kv_copy` / `kv_copy_excluded`. Phase 3 spec signal table and the contract (§8.3 note and the
  metrics table) are amended. Tests: `step_window::tests::steps_overlapping_a_kv_copy_are_not_judged`,
  `kv_orchestrator::tests::copy_mark_overlap`, `engine::r#loop::tests::decode_steps_during_kv_copies_are_not_judged`. Mutations
  (no exclusion; learning from overlapped calm steps; `overlapped()` always false / always true) each fail. Gate
  `scripts/gate.sh --base ce88483`: `gate: ok passed=841`.
- Hunks in `kv_orchestrator.rs` (promotion/copy path only, for the ladder builder): `CopyMark` (before `struct KvOrchestrator`), the
  field `busy_polls`, `copies_in_flight` / `copy_mark` (after `transfers_idle`), two lines in `poll`, and the test `copy_mark_overlap`.
- Perf item, part 1 (from the p6b-drift traces): a step's extra time is linear in the L1 → L0 bytes in flight with it (0.064 ms/MiB,
  r 0.99). This is SDMA queueing, not a fence. See the perf log.
- Release build and kernel build of 05471e6 on novanas: `remote/agent-p6b-copydrift/target/release/` and `kbuild/libturbine_hip.so`.

## Next (lab, when the lead says "lab free")

Harness `/home/piwi/turbine-ci/scratch/p6b-copydrift/ab.sh <abs-outdir> <model>:<fmt>:<run>…`. It is the lastblock `ab.sh` with this
branch's remote dir. It also logs `unjudged_kv_copy=` and `drift=`. Set `DEBUG=1` for the decode_step / kv_copy_stages trace and the
100 ms pressure poll, `PROF=1` to wrap the server in rocprofv3 (the server is SIGTERMed, then after 90 s the whole session is SIGKILLed),
and `EXTRA="--set …"`. Run it detached, port lock first:

`L=/home/piwi/turbine-ci; cd $L/scratch/p6b-copydrift; nohup setsid flock -x $L/port18000.gate flock -x $L/port18000.lock flock -x $L/bench.gate flock -x $L/bench.lock ./ab.sh $PWD/<dir> … > <dir>.nohup.log 2>&1 &`

1. `DEBUG=1 PROF=1 ./ab.sh $PWD/prof olmoe:l0:1`, then `python3 prof.py prof/olmoe-l0-r1.prof` and `python3 corr2.py prof/olmoe-l0-r1.server.log`.
   Confirm whether the compute stream's small copies stretch, or leave gaps, inside promotion intervals.
2. If it is SDMA queueing: one config-only probe, `DEBUG=1 EXTRA="--set kv.transfer.max_inflight_bytes=64MiB" ./ab.sh … olmoe:l0:1`.
   Compare the extra ms per step (`corr2.py`) and the promotion latency and cached ratio with the default. Then report options (shim:
   batch upload through host-mapped memory or a kernel; a smaller in-flight cap; time-sliced copy batches). Do not change kernels
   without reporting first.
3. A/B: `./ab.sh $PWD/ab olmoe:l0:1 olmoe:tq4:1 olmoe:l0:2 olmoe:tq4:2 olmoe:l0:3 olmoe:tq4:3 llama:l0:1 llama:tq4:1 llama:l0:2 llama:tq4:2 llama:l0:3 llama:tq4:3`.
   Gate: tq4 median ≥ l0 median on both models, with every run staying GREEN (check `transitions=` / `drift=` and the
   `.pressure.json`). If it holds, flip lower-tier `tq4` in `crates/turbine-core/src/support.rs` `TIER_FORMAT_REFUSALS`. The steps
   are in `p6b-lastblock.md` (evidence comment, `support::tests`, `support_startup` tests, AGENTS.md support line).
