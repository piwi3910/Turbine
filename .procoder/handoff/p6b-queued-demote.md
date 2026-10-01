# Handoff: p6b-queued-demote (decision "6b: after the held-prefix ledger fix — tail latency and queued prefixes", 1 B, 2 A)

Branch `p6b-queued-demote` from `p6b-stack` f9a4c91. The lead paused the work during the design step. No code, no tests and no lab
runs exist yet, and this builder started no serve or lab Job. This file is the only commit.

## Where the queued prefixes live (read, verified)

- `turbine-server` `engine/loop.rs` `submit` → `attach` (`KvOrchestrator::attach` → `KvHierarchy::attach_prefix`) references the
  prefix blocks in the pool (`incref`), then `admit_submission` → `SchedRequest::attach_prefix` → `Scheduler::submit` →
  `AdmissionGate::offer`. A queued request waits in the gate's `AdmissionQueue<Waiting, AdmissionKey>`, holding
  `SchedRequest.cached_prefix` (one pool reference per block). `sync_kv_held` reports `pool.referenced_blocks()` to the ledger.
- `Scheduler::plan` pumps the gate (`gate.pump`, which reserves `q.estimate.projected_kv_blocks` = `reserved_kv_blocks`, request minus
  the attached prefix) into the scheduler's queue, then `admit` hands `cached_prefix.blocks` to seq 0's table in the same `plan`.
- `attach_prefix` registers the blocks this request will compute as `pending` in the directory (`register_pending`), and only
  `request_done` clears them. A second `attach_prefix` for the same id would see its own pending key and return `WaitForPrefix`
  until `PENDING_WAIT`, so a re-attach needs those keys cleared first.
- Reclaim: the controller's throttle plan calls `KvReclaimHandle::demote(yellow threshold)` at YELLOW, and `free_unreferenced` +
  `demote(orange threshold)` at ORANGE. `KvHierarchy::apply_reclaim` (engine `end_turn`) → `pressure_reclaim(target)`:
  `need = used − leaving − target·total`, `victims` are unreferenced blocks only (leaf-first, policy score), and at most
  `DEMOTION_INFLIGHT` (32) copies are in flight; an already-copied block (copy ahead) is freed with no copy.

## Proposed design (not implemented)

1. Demand: `pressure_reclaim` records `queued_prefix_demand = need − victims.len()` (blocks it wanted and found no unreferenced
   candidate for), only when `l0_state` is YELLOW or ORANGE, as the max over the tick's calls. The engine reads it with
   `take_queued_prefix_demand()` and caps it at `CAPACITY_BATCH` (32) per controller tick. At GREEN, RED and SURVIVAL the demand
   stays 0, so the existing rules apply.
2. Choice: `Scheduler::detach_queued_prefixes(&BlockPool, want)` walks the gate queue in reverse admission order (last to be
   admitted goes first, the head keeps its prefix). It counts only blocks with pool refcount 1 (held by that queued request alone)
   and detaches whole prefixes until the count reaches `want`; a request that holds nothing alone is skipped. On detach,
   `cached_prefix` goes to `None`, `reattach = true`, and `projected_kv_blocks` grows back to the full `request_kv_blocks` in both
   `SchedRequest.estimate` and the `Queued.estimate` copy. That copy needs an `iter_mut` (or a `rekey`) on `AdmissionQueue` in
   `turbine-reliability`. `new_prefill_tokens` / `cached_prefix_tokens` keep the claim, so ORANGE's `ExpensiveQueued` does not
   start treating it as expensive.
3. Release: the engine releases the blocks and calls a new `KvHierarchy::detach_prefix(request)`. That clears the request's
   pending keys past `committed`, sets `committed = 0` and resets `keys` / `used` / `lineage`. It also marks the `RequestKv` so the
   re-attach does not count `prompt_tokens` again. Then: metric `turbine_kv_queued_prefix_detached_blocks_total`, log event
   `kv_queued_prefix_detach` (reason `queued_prefix`), and `sync_kv_held`. `held` drops at detach, because unreferenced cached
   blocks already count as available, as for every cached block. The next `demote` tick copies the blocks to the next tier
   (they have the hit evidence from the first attach) and frees L0, or frees them at once if copy ahead already put a copy down.
4. Re-attach: a pumped request with `reattach` is not `admissible` (head-of-line, about one iteration). Before the plan, the engine
   attaches each `sched.awaiting_reattach()` request again. `Ready(a)` → `sched.reattach(id, a)`, and `ActiveRequest.cached_tokens`
   is updated. `Promoting` → a `reattaching` set, completed from the `kv.poll` loop's `None` branch. `WaitForPrefix` → retried next
   turn. Its reservation stays full size (safe if the planner then recomputes; the ledger takes max(in-use, reserved)).
   Cancellation and SURVIVAL requeue need no new path (no prefix held; `request_done(cancelled)` drops a pending promotion).

## Tests to write first (then mutation-check)

- `turbine-kv`: demand recorded only at YELLOW/ORANGE; `detach_prefix` then `attach_prefix` gives `Ready`/`Promoting`, not
  `WaitForPrefix`.
- `turbine-scheduler`: reverse-order choice with the refcount-1 count and the bound; the estimate and the `Queued` copy updated; a
  `reattach` request is not admitted until `reattach`.
- Engine (cpu backend, `engine.turn()` stepping): keep `parts.controller` in `TestEngine` and tick it with a sample
  (`at_mono_ns` from the engine clock) under `kv_utilization` thresholds of about `[0.1, 0.9, 0.95, 0.99]`. Use
  `max_running_requests` 1, the L2 tier as in `l2_round_trip_matches_cold`, and run a 100-token prompt cold. Then: R running, Q
  queued with the same prompt (6 blocks attached). At GREEN, a direct `reclaim.demote(0.1)` detaches nothing. At YELLOW, Q's 6
  blocks are detached, demoted to L2 and leave L0, and the referenced blocks drop by 6. Q is then admitted, re-attaches from L2
  (or recomputes, per the planner) and yields the cold tokens. Mutations to check: no YELLOW gate, a GREEN detach, no re-attach,
  forward order, no pending clear.

## Open questions (none blocking; recommendation first)

- Detach granularity: whole prefixes (recommended, simple re-attach) vs tail blocks only.
- Only the gate queue (recommended) vs also admitted-but-unstarted requests in the scheduler queue (they start within a few turns).
- No new config key is proposed. Lazy attach for queued requests (attach at pump only) is the alternative, and it would also
  answer 2 B, which the user did not choose.

## Next steps

Implement points 1–4 test first, run `scripts/gate.sh`, then the lab: 24 and 32 sessions before and after
(`.procoder/handoff/p6b-survival.md` commands, `kv.cpu.max_bytes=4GiB`, `kv.cpu.format=l0`, GPU 0, one fresh server per count), then
`scripts/overload-soak.sh novanas --duration 10m`, then the perf-log 6b entry.
