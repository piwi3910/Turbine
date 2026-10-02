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

## Done (lab, 2026-10-02, tree 108c734)

All numbers are in the perf log, Phase 6b, "KV copies and decode steps", Part 3 and the tq4 A/B. Labbook set `phase-6b-kv-compression`,
external ids `p6b-copydrift:ab:*` and `p6b-copydrift:cap:*`. Results and scripts are on novanas under `scratch/p6b-copydrift/`
(`runall.sh`, `corr3.py`, `prof2.py`, `blit.py`, `queue.py`, `align.py`; dirs `prof`, `cap-def`, `cap-64`, `kernarg`, `ab`).

- Mechanism (rocprofv3): promotions are 1 MiB SDMA copies, one per layer, 83 µs each. Decode does no SDMA copy while serving. Its
  batch upload is a blit kernel on the compute stream. While an SDMA copy runs, every compute kernel stretches to about one copy
  (rmsnorm 6 → 80 µs, the blit 2.5 → 84 µs), and kernel occupancy drops from 0.83 to 0.45. The cost is 61–69 µs per MiB promoted.
  `HIP_FORCE_DEV_KERNARG=1` does not change it.
- 64 MiB in-flight cap (config only): no step reaches 1.5×, and promotions get faster (median 29–30 → 19–21 ms). Reuse is unchanged.
  The total time lost does not drop, tok/s is −1.6 % and later-turn TTFT p99 is worse. The default is unchanged.
- A/B: the gate holds (OLMoE 0.8625 ≥ 0.8624; Llama 0.9075 ≥ 0.9007). Lower-tier `tq4` is flipped to `supported`.

## Open: fix choice for the user (decision "6b: step-time drift during KV promotions" C, part 2)

- (a) A smaller default `kv.transfer.max_inflight_bytes` (64 MiB): it bounds the per-step tail but not the total cost.
- (b) A shim change. Decode's upload does not use the copy engine, so this means promotions without SDMA (a copy kernel reading
  pinned host memory) or smaller SDMA copies (for example 256 KiB, if the stall is per copy). Measure the ms/MiB slope.
- (c) Time-sliced copy batches: a per-step byte budget, or promotions issued beside prefill steps. Like (a), it bounds the tail.
