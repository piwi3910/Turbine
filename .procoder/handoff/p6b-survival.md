# Handoff: p6b-survival (decision "6b: SURVIVAL at 20–32 multi-turn sessions on Llama-3.2-3B", A)

Branch `p6b-survival` from `p6b-stack` e02e8c2. Numbers: `.procoder/perf-log.md`, Phase 6b, "SURVIVAL at 20–32
multi-turn sessions". Commits `083b1c7` (fix) and `59e36ff` (soak start guard) each passed
`scripts/gate.sh --base e02e8c2` → `gate: ok` (804, then 805 passed).

## Mechanism (evidence: serve run 1001102511-200e23db, 32 sessions, pressure and KV documents every 100 ms)

- The forecast was right. In the 100 ms before GREEN → SURVIVAL, L0 went from 563 to 585 of 585 blocks referenced
  (0 free, 0 cached). `exhaustion_horizon` went from +∞ (the remaining ≤ 128 tokens per sequence still fit the 22
  free blocks) to 0. Admitted decodes were already logging `decode_deferred reason="kv_exhausted"`, so this was a
  real exhaustion that broke the P3 S-9 guarantee.
- The ledger was wrong. `kv_utilization` read 0.70 at that moment (104 blocks reserved, about 308 committed). A
  request's worst-case reservation leaves out the cached prefix blocks it attaches (`reserved_kv_blocks` in
  `turbine-scheduler`; P4 S-3): "other requests' reservations already paid for them, or they were cached
  unreferenced". The second case is never paid. Blocks a turn takes from the cache (its session history, about 30
  blocks a turn) are referenced but in no reservation. About 173 blocks were invisible to admission and to
  `kv_utilization`.
- Why the intermediate states were skipped: `kv_utilization` never crossed 0.70, so the only signal that saw L0 fill
  was the horizon. With 128-token outputs the horizon is close to a step function (+∞ until the free blocks cannot
  hold one more block per sequence, then under 1 s). The state machine escalates to the target level after
  `escalate_samples` (2) samples, so GREEN → SURVIVAL directly is by design (P3 S-7). It was the right reaction to a
  pool the ledger had over-committed.
- Hypotheses ruled out: promotion bursts as committed growth (the horizon reads only `max_tokens − generated` of
  running sequences; promoted blocks show up only as fewer free blocks, which was true); worst-case `max_tokens`
  reservations (128 per request, about 104 blocks in all); a too-short rate window (the rate does not matter when
  free = 0); L1 promotions not credited (they were really referenced).

## Fix (083b1c7, `fix(reliability)`)

- `turbine-reliability` `Ledger::set_held(device, pool, in_use)`: the pool's owner reports what it holds, and what
  the commits do not cover becomes `PoolUsage.held`. `available()`, `utilization()`, the `used` gauge of
  `turbine_memory_pool_bytes` and the pressure document's `used_bytes` include it. It is not journaled and not in
  the mirror digest; a TP group's leader ledger carries it, and admission and the probe take the worst rank.
- `turbine-server` engine: `sync_kv_held()` reports `pool.referenced_blocks() × block_bytes` before each submission
  (and probe), at the end of `kv_before_plan` (before the plan's queue refills) and with the stats.
- P3 spec signal table (`kv_utilization`) and the contract's `Ledger` entry amended.
- Tests: `turbine-reliability ledger::tests::held_bytes_outside_reservations_count` and
  `turbine-server engine::r#loop::tests::attached_prefix_blocks_count_in_the_ledger` (cpu backend: a repeated 100-token prompt
  attaches 6 blocks; red before the fix: "the ledger counts 2 blocks, the pool holds 7"). Mutations, all caught:
  held = in_use (double count); `available` ignores held; `used` gauge without held; engine sync a no-op.
- No overload-sim reproduction: `sim/overload.rs` has no prefix reuse and `KvSimDriver` has no gate. The
  deterministic reproduction is the cpu-backend engine test above.

## Re-measure (GPU 0, one fresh server per row)

32 / 24 / 20 / 16 sessions: 256/256, 192/192, 160/160, 128/128 ok. No SURVIVAL, no `decode_deferred`. Deepest state
RED / RED / ORANGE / GREEN, reached one state at a time on `kv_utilization`. Later-turn TTFT p50 3868 / 162 / 114 /
78 ms; p99 43.6 / 39.7 / 7.9 / 0.18 s. Before: 32 → 184/256 ok, 24 → 155/192, 20 tripped in 1 of 3.

## Soak

`scripts/overload-soak.sh novanas --duration 10m` on 59e36ff (serve run 1001112421-15fb3885,
`target/soak/novanas-20261001T112420Z`): **PASS**, all 8 checks true. ITL p99 211 ms vs 176 ms calibration; GREEN 24 s
into the cool-down; 4385 × 200, 72 × 503 `overloaded`, 2421 `queue_timeout` (passing fp8 / fp8_block soaks had
4041 / 2939). The bench client ran on novanas through an ssh `SOAK_BENCH` wrapper (nothing is built on the Mac).

The first attempt (11:00 UTC) is invalid. `lab-serve.sh` refused to start because another server answered on port
18000, and the soak ignored the refusal (`| tee … || true`): it calibrated and overloaded that foreign server for
10 minutes and failed when that server shut down (627 `shutting_down`, 67 `model_not_loaded`). Whoever ran a native
server on 18000 at about 11:00–11:18 UTC got 10 minutes of overload traffic from this soak. `59e36ff` makes a
refused start fail the soak before any load (`lab_scripts soak_refuses_a_server_it_did_not_start`, red before).

## Open (needs decisions)

1. Tail TTFT at 24–32 sessions (p99 about 40 s) is now queueing in ORANGE/RED: the live histories exceed L0 (585
   blocks), and requests wait instead of being turned away with 503. Options: A) accept (it is the designed
   ladder); B) a larger L0 or lossy L1 for this workload; C) let YELLOW/ORANGE demote the attached prefixes of
   _queued_ requests (`turbine-kv`, not this branch).
2. Requests waiting in the admission queue hold their attached prefix blocks (now counted). If queued prefixes alone
   fill L0 with nothing running, the head waits until `queue_timeout` (60 s). Not seen; options: A) leave it;
   B) release a queued request's prefix after a wait and re-attach on admission.
3. `reserved_kv_blocks` still subtracts the attached prefix, which is now correct because `held` pays for it. A
   comment there (`turbine-scheduler`) could say so.

No serve Job of this branch is left running (the soak stops its own).
