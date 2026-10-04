# Handoff: p6b-fp8kvbisect — the fp8-KV golden miss was the recent window on the fp8 base; fixed

Branch `p6b-fp8kvbisect`, from `p6b-stack` `5bec8eea` (the exit merge). User decision 2026-10-04,
"6b exit: llama fp8-KV golden fails deterministically", **A** (diagnose first): the window must
not apply when the base format is lossless-with-matched-scales. Confirmed, fixed, spec amended.

## Mechanism (confirmed on the cpu backend, red→green)

The window's eligibility predicate was `KvDtypeChoice::is_lossy()` (`true` for anything but
`bf16`), used at five sites; none consulted the codec's matched-scales status. So with
`kv.dtype: fp8_e4m3` and the default `kv.recent_window_blocks: 1`:

- the scheduler created every sequence's newest full block(s) — for a short prompt, its only,
  partial block — in the BF16 page class; rows are appended **unquantized** and decode reads
  them in BF16, not as `bf16(e4m3 × scale)`; p10 of the golden set is a ~50-token chat prompt,
  so its whole sequence sat on a BF16 window page. That is the deterministic per-prompt logprob
  shift (identical 32-token prefixes, p10 likely |Δ| 0.4274 vs the 0.40 bound from the 6a Task 24
  calibration, which was taken on lossless fp8-KV);
- the pool went classed (`kv_page_classes` log, tq_device's table upload, the mixed-attention
  routing of `p6b-window`'s addressing) although no conversion ever needs to run for p10 (a
  partial block never leaves L0).

Empirical anchor (novanas, cpu backend): new test
`tiny_server::recent_window_skips_lossless_fp8kv_base` failed pre-fix ("the fp8 base pool must
stay flat: no BF16 window class"), passes post-fix, and `recent_window_serves_on_cpu` (tq4)
stays green. Mutation check: reverting the `l0_page_classes` predicate back to `is_lossy()`
alone re-fails the test (snapshot → mutate → red → restore → green).

## Fix (one commit with the test and the spec amendment)

- `turbine-core/config/kv.rs`: new `KvDtypeChoice::recent_window_base()` — the window-eligible
  bases are the TurboQuant ones; `bf16` and `fp8_e4m3` are the format's reference
  representations (`fp8_e4m3` lossless with matched scales, S-1) and get no window. The window
  exists to soften a lossy base whose approximation is Turbine's own (tq4/tq2).
- Predicate swap at the five sites: `SchedulerParams::from_config` (turbine-scheduler), the
  server's `classed_tables` / `l0_page_classes` (turbine-server/model.rs), `tq_device::
reserved_bytes` (fp8 pools are flat again → no table upload, no mixed routing; the
  `tables_are_reserved…` unit test's fp8 expectations 0), and `hierarchy::window_blocks`
  (turbine-kv).
- Docs: spec S-5 window sentence and the config table amended (window = TurboQuant bases only,
  decision 2026-10-04 A); contract §26 window line likewise; stale "lossy base" comments fixed.

## Golden16

`scripts/lab-bench.sh --model llama-fp8kv --label fp8kvfix --golden16` on novanas GPU 0, from
commit `70f22e92` — **PASS restored**: golden c1 **PASS 16/16** (strict bounds) and c16 **PASS
16/16** (batched bounds); bench 838.8 tok/s (6a exit: 829.7), ITL p50 15.6 ms, TTFT p50 213 ms,
200/200 ok. The 6a Task 24 tolerance (likely 0.40 / tail 2.44) unchanged — no recalibration.
Recorded in labbook (`67de6c5f-7579-4c47-a1f1-f3dd6da70e98`); artifacts under
`target/lab-bench/fp8kvfix-llama-fp8kv/`, log `target/fp8kv/lab-bench-fp8kvfix.log` on the
workstation. Nothing left running: the serve stopped by the script itself and the locks are
released.

## Notes

- `olmoe-fp8kv` gets the same predicate change; its golden gate stays as the 6a-exit record
  (`experimental`, user decision 2026-09-30 B) — not re-run here.
- The 6a exit's 0.40/2.44 tolerance stays untouched: no recalibration happened (decision A:
  recalibrate only for legitimate accepted semantics).
- p6b-exit finding 3 (options A/B/C) is resolved by this branch (option A's bug path).
