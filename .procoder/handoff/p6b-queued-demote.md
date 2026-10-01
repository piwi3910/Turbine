# Handoff: p6b-queued-demote (decisions "6b: after the held-prefix ledger fix", 1 B, and "6b: queued-prefix demotion — granularity and scope", 1 A, 2 A)

Branch `p6b-queued-demote` from `p6b-stack` f9a4c91 (tip before this work 76af3cb). Implementation commit `c96a07a`
passed `scripts/gate.sh --base 76af3cb` → `gate: ok` (822 passed). Numbers: `.procoder/perf-log.md`, Phase 6b, "Queued-prefix
demotion".

## What landed (c96a07a)

- `turbine-kv` `hierarchy.rs`: `pressure_reclaim` records `need − victims.len()` as the queued-prefix demand, only while
  `l0_state` is YELLOW or ORANGE (max over the tick's calls; reset in `apply_reclaim`; `take_queued_prefix_demand()`). Only
  with a lower tier: without one `pressure_reclaim` is `demote_to` and records nothing (a release would only recompute).
  `detach_prefix(pool, request, &attach, alone)` releases the references, clears the request's pending keys past
  `committed` (the PENDING_WAIT trap), resets `keys`/`used`/`committed`/`lineage`/`lossy_from` and marks `detached_tokens`;
  the next `attach_prefix` counts lookups and the plan but not the prompt or cached tokens again (only the shortfall as
  recompute). Metric `turbine_kv_queued_prefix_detached_blocks_total`, `KvStats.queued_prefix_detached`, DEBUG
  `kv_queued_prefix_detach` (reason `queued_prefix`).
- `turbine-reliability` `AdmissionQueue::behind_head_rev_mut` (entries behind the head, last first; keys and order fixed).
- `turbine-scheduler`: `AdmissionGate::detach_prefixes` (refcount-1 count, whole prefixes, stops at `want`, head skipped,
  `projected_kv_blocks` back to `request_kv_blocks` in the request and the queue copy, `reattach = true`);
  `Scheduler::{detach_queued_prefixes, awaiting_reattach, reattach -> Option<PrefixAttach>}`; `admissible` is false while
  `reattach`.
- `turbine-server`: `EngineLoop::release_queued_prefixes` after `kv.end_turn` (want capped at `CAPACITY_BATCH`),
  `reattach_released` in `kv_before_plan` (`Ready` → `sched.reattach`, `Promoting` → `reattaching` set completed from
  `kv.poll`, `WaitForPrefix` → next turn), `usage.cached_tokens` from the new attach; `KvOrchestrator::{attach_again,
detach_prefix, take_queued_prefix_demand}`. Test scaffolding: `engine_full` (reliability config), `TestEngine.{pressure,
clock}`.
- Spec 6b S-8 sentence, observability line and AC line; contract paragraph "Queued-prefix demotion".

## Tests and mutations (all 12 caught)

`turbine-kv hierarchy::tests::{queued_prefix_demand_only_at_yellow_and_orange, detached_prefix_attaches_again_through_the_planner}`,
`turbine-scheduler scheduler::tests::queued_prefix::{released_last_queued_first_up_to_the_demand,
released_request_waits_for_its_reattach}`, `turbine-server engine::r#loop::tests::yellow_releases_queued_prefixes_and_they_reattach`
(cpu backend, L2, the test ticks the controller: GREEN releases nothing; YELLOW releases Q's 6 blocks, which are demoted to L2;
Q then yields the cold run's tokens). Mutations: no YELLOW/ORANGE gate; demand ignores victims; no pending clear; re-attach
counts the prompt; demand not reset; forward walk; head released; shared blocks counted; queue estimate not updated;
`admissible` ignores `reattach`; engine never re-attaches; engine never releases. Each made a test fail.

## Lab

See the perf log entry. The before rows ran from a detached worktree of 76af3cb in the scratchpad; the multi-turn tools are in
`/home/piwi/turbine-ci/remote/agent-p6b-queued-demote/runs/` on novanas (`sweep.sh`, `poll.py`, `bin/turbine-bench` copied
from the survival builder's release build).

## Open

- Design choice not in the decision: no release without a lower tier (L0-only servers keep the old behaviour).
- The `kv_utilization`-driven tail latency is still admission queueing; see the perf log for whether the release moved it.
