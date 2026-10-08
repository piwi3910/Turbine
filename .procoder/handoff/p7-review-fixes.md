# Handoff: p7-review-fixes (2026-10-08)

Branch `p7-review-fixes` off `p7-stack` (merged). All three findings are committed.

- Finding 1 (qwen_think): commit 9753804d. Gate `--base p7-stack` was ok (414 passed, turbine-model + turbine-server).
- Findings 2 and 3 (one commit, 8a7c62dd): `StateSlots` / `StateCharge` in `turbine-reliability/src/ledger.rs`;
  `Admission::set_state_capacity` (scheduler sets it to `max_running_requests x state` in `with_state_layout` /
  `with_gate`); the ledger `kv` capacity is the physical pool (`measure_budget` subtracts the state allocation,
  `kv_pool_config` no longer does); a request holds `n x state`; `n` beyond the allocation is refused at submit
  with `SubmitError::StateExceedsCapacity` / `ErrorCode::StateExceedsCapacity` (`state_exceeds_capacity`, 400);
  one lacking free state queues (`kv_reservation`). Tests in `crates/turbine-scheduler/tests/kv_sim.rs`:
  `state_is_charged_beside_the_kv_pool` (failed before the fix), `state_covers_every_sequence`,
  `hybrid_reservation_includes_state` (updated).

Verified on novanas before the commit: `cargo test -p turbine-scheduler -p turbine-reliability` green. A gate run on the
working tree had one failure, `server_cli sigterm_drains_then_cancels` (bad chunk size line, load flake; passes alone).

Next: the final `scripts/gate.sh --base p7-stack` on the commit could not run: novanas (192.168.10.203) stopped
answering ssh ("Connection closed by UNKNOWN port 65535", also via -J dgx-spark). Re-run it when the host is back.
Not done: a server-level test for `state_exceeds_capacity` over HTTP; the token-granular docs do not list the code.
