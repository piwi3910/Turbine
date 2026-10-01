# Handoff: p6b-queued-demote (decisions "6b: after the held-prefix ledger fix", 1 B, and "6b: queued-prefix demotion — granularity and scope", 1 A, 2 A)

Branch `p6b-queued-demote` from `p6b-stack` f9a4c91 (76af3cb before this work). Done: implemented, tested, mutation-checked, lab
measured, soak PASS. Every code commit passed `scripts/gate.sh --base 76af3cb` (last: 9d4cf27, `gate: ok`, 823 passed). Numbers:
`.procoder/perf-log.md`, Phase 6b, "Queued-prefix demotion". No serve Job of this branch is left running.

## Commits

- `c96a07a` `feat(kv)`: the mechanism. `KvHierarchy::pressure_reclaim` records `need − victims` as the queued-prefix demand at
  YELLOW/ORANGE only (only with a lower tier; without one it is `demote_to` and records none). The engine takes it after
  `kv.end_turn`, capped at `CAPACITY_BATCH`. `AdmissionGate::detach_prefixes` walks the queue behind its head, last first,
  releasing whole prefixes until the refcount-1 blocks reach the demand, and restores the full `projected_kv_blocks` in the request
  and the queue copy (`AdmissionQueue::behind_head_rev_mut`). `KvHierarchy::detach_prefix` clears the request's pending keys (the
  PENDING_WAIT trap), resets its keys and lineage, and counts the re-attach's prompt and cached tokens once. A pumped `reattach`
  request is not `admissible` until `EngineLoop::reattach_released` attaches it again (`Ready`, `Promoting` through a
  `reattaching` set, `WaitForPrefix` next turn). Metric `turbine_kv_queued_prefix_detached_blocks_total`, DEBUG
  `kv_queued_prefix_detach` (reason `queued_prefix`). Spec 6b S-8, observability and AC lines; contract paragraph.
- `8c5de54` `fix(scheduler)`: `Scheduler::reattach(id, attach, block_bytes)` commits the re-attached blocks against the
  whole-request reservation. Without it, `held` and `reserved` both counted them: the first lab run went to SURVIVAL.
- `82a47b2` `fix(server)`: on `Promoting` the engine commits `KvHierarchy::pending_blocks` (`Scheduler::commit_reattached`). The
  promotion targets are referenced from attach time on.
- `9d4cf27` `fix(reliability)`: the S-9 headroom (`within_headroom`) counts `PoolUsage::in_use` (committed + held), like
  `kv_utilization`. RED refills of whole-prompt reservations went past SURVIVAL's 0.97. This changes admission for every
  workload; it is a latent gap of the held-prefix ledger fix (083b1c7).

## Tests and mutations (15, all caught)

Tests: `turbine-kv hierarchy::tests::{queued_prefix_demand_only_at_yellow_and_orange,
detached_prefix_attaches_again_through_the_planner}`; `turbine-scheduler scheduler::tests::queued_prefix::{released_last_queued_first_up_to_the_demand,
released_request_waits_for_its_reattach}`; `turbine-reliability admission::tests::kv_headroom_counts_held_bytes`, red before the
fix; and `turbine-server engine::r#loop::tests::yellow_releases_queued_prefixes_and_they_reattach`. The engine test runs the cpu
backend with L2, and the test ticks the controller: GREEN releases nothing, YELLOW releases Q's 6 blocks, which reach L2, and Q
then yields the cold tokens.

Each of these mutations made a test fail:

- no YELLOW/ORANGE gate;
- demand ignores victims;
- no pending clear;
- re-attach counts the prompt;
- demand not reset;
- forward walk;
- head released;
- shared blocks counted;
- queue estimate not updated;
- `admissible` ignores `reattach`;
- engine never re-attaches;
- engine never releases;
- re-attach not committed;
- commit ignores the earlier commit;
- `pending_blocks` returns 0.

The engine's `commit_reattached` call on `Promoting` has no engine-level test (the scheduler and kv units cover its parts).

## Lab result (medians of 3)

- 24 sessions: p99 28.3 s → 9.6 s, RED → ORANGE.
- 32 sessions: p50 4.7 s → 0.54 s, p99 58.9 s → 19.3 s, 7–11 `queue_timeout` → 0, RED → ORANGE.
- Control (headroom fix only, throwaway commit aa9153a in a scratch worktree): 24 sessions is the same as after (the headroom
  fix does it). At 32 sessions the release adds p50 1.4–2.0 s → 0.4–0.8 s and p99 27–52 s → 18–27 s.
- Soak 10 min on 9d4cf27: PASS 8/8.
- The before rows at 32 sessions all ran on GPU 1 (the device plugin's pick; `lab-serve.sh` cannot pin a card).

Tools on novanas are in `/home/piwi/turbine-ci/remote/agent-p6b-queued-demote/runs/`: `sweep.sh`, `poll.py`, `summ.py` (the table
from the run files) and `bin/turbine-bench`. Local drivers are in the session scratchpad: `rows2.sh` and `soak-bench.sh` (the
`SOAK_BENCH` ssh wrapper that copies the `--pressure-timeline` file back).

## Open

- `lab-serve.sh`: a start whose rsync fails after it took `bench.lock` leaves its holder (`runs/serve-locks/<run>.sh`) waiting up
  to 3600 s for a Job that never comes. That blocks every GPU job. I released mine by deleting its `.state` file. The script
  should drop the holder on a failed upload.
- 32-session before rows on GPU 0, if a GPU-0-only comparison is wanted.
- The demand needs a lower tier. An L0-only server keeps the old behaviour (a release there would only recompute).
