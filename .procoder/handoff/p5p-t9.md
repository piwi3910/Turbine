# Handoff: p5p-t9 (5p plan Task 9 — the `cache_aware` scheduling policy)

Branch `p5p-t9` from `p6b-stack` c7657649. Two commits, gate-clean:

- f20f4774 `feat(scheduler): the cache_aware scheduling policy (S-7/S-8)` (gate ok,
  crates=turbine-scheduler,turbine-server, 233 tests)
- docs commit (this one): perf-log S-7/S-8 row, plan Task 9 ticks.

Lab A/B recorded in `.procoder/perf-log.md` ("S-7/S-8 landing") and labbook (8 runs, ids in the
row); the lab run's first wrapper hit the 2 h background limit mid-`def3` and was restarted as a
single run — the orphaned native server on 18000 was SIGTERM'd before the rerun (nothing else
touched). Logs under `target/t9/`.

## The design, five lines

1. `CacheAwarePolicy` (`crates/turbine-scheduler/src/policy/cache_aware.rs`), registered after
   `default`: non-preempted requests key `tier 1, priority, arrival = ⌊arrival/window⌋ × window,
affinity = u32::MAX − cached_prefix_tokens, order = submit_no`; preempted requests, the
   victim and the chunk cap delegate to `DefaultPolicy`. Stateless, clock-free — the window
   rides the `AdmissionInfo`.
2. `AdmissionInfo` gains `prompt_tokens`, `cached_prefix_tokens` (the prefix attached before the
   gate; `Scheduler::submit/enqueue`, the SURVIVAL requeue and `preempt` all fill the fields
   through one associated `admission_info(window, …)` helper that takes the window by value —
   the call sites hold `&mut` borrows of the gate/request map) and `AdmissionKey` gains
   `affinity` between `arrival` and `order` (`default` keeps 0, keys and plans unchanged —
   `default_policy_plan_digests_match_main` untouched and passing).
3. The pressure gate runs before the policy as before: nothing in the gate or throttle changed;
   `cache_aware` orders what the gate admits.
4. `SimReport.admission_inversions` (S-8): an `AdmissionLedger` in the simulators records each
   request's first admission (arrival, priority, admitted-at, ever-preempted) and lists pairs
   (earlier, later) of equal priority, never preempted, arrival gap ≥ window, later admitted
   first. `KvSimDriver` exposes `with_policy`, `admission_inversions()` and
   `admission_span(id) → (arrival, admitted-at)`.
5. Docs: `docs/extending/scheduling-policy.md` names the policy file, the new fields, the
   bound-test command and the pitfall "the admission key is computed once at push"
   (`docs_extending` green).

## Tests and mutations

- `policy::tests::cache_aware_orders_by_window_then_prefix`: within-window affinity, across-
  window order, priority dominance, the affinity formula, preempted-keys-equal-default,
  delegation of rank/victim/chunk-cap, registry presence; `default`'s keys pinned to
  `affinity 0` in `default_policy_orders_like_main`.
- `kv_sim cache_aware_starvation_bound`: seeded (ChaCha8 seed 6) flood of 2,000 × 128-token
  prompts sharing a fully cached primer, one uncached every 50 arrivals, max_running 4, window
  1 s, pool 8192 blocks; `cache_aware` completes all 2,000 with **0 inversions** (max uncached
  wait 32.8 s), the test-only unbounded variant (arrival 0, window MAX) reports **30,055
  inversions** and 39.5 s. Both legs red-first (compile), then green.
- Mutation checks (each red, then restored from a snapshot, `diff`-verified):
  - affinity dropped (`affinity: 0`) → the order test fails.
  - window bucketing dropped (`arrival: Duration::ZERO`) → the bound test fails with exactly
    the unbounded leg's inversions. Note: replacing the bucket with the _raw_ arrival instead
    yields FIFO, which trivially respects the bound — the mutation that "drops the window"
    meaningfully is the zero-arrival one.
- `registry_conformance`, `sim::tests`, `config::tests::phase5p_keys`,
  `turbine-server` (incl. `modules.scheduling_policy` selection), `docs_extending` all green.

## The A/B (medians of 3, golden1 PASS on all eight runs)

cache_aware 0.9143 cached / 10 656 ms later p50 / 490.2 mt tok/s; default 0.9147 / 11 110 /
489.2. S-13 verdict: tok/s and first p99 pass; later p50 0.96× (target 0.85×) and mt_cached
(equal) **miss** — cache_aware is better on later p50 in all three paired runs, but the profile
ties the affinity term (one shared 2,000-word system prefix for all 64 sessions) and is bimodal
under saturation (one low-reuse run per policy: cached 0.81-0.82, ~347 tok/s vs 0.914/490).
Base legs (policy `default`, standard bench): Llama 854.1, OLMoE 615.5 tok/s — no regression
from the code itself.

**For the coordinator/user**: the plan asks whether `cache_aware` becomes the serving default.
With these numbers the recommendation is to keep `default`; the policy ships as
`scheduler.policy: cache_aware` for mixed cached/uncached mixes, which the multi-turn profile
does not produce. The S-13 A/B criterion (0.85× later p50) may simply be unmeasurable on this
profile — if the mix matters, a profile with heterogeneous prefixes would be the honest A/B.

## Deviations from the plan text

1. The plan's unbounded variant is "window ∞" (`Duration::MAX`) — the task text said
   `Duration::ZERO`, which would divide by zero in the bucketing; implemented as the plan says
   (MAX), with a `max(1 ns)` guard in the policy anyway.
2. `SimReport.admission_inversions` is filled by `Simulation::run` and exposed on
   `KvSimDriver` directly (the driver has no `SimReport` of its own; the kv_sim test reads the
   driver's accessors). No live metric was added — the S-8 AC judges the bound in the sim.
3. The kv_sim workload's pool is 8192 blocks (payload-free): at Green pressure nothing evicts
   the unique cached decode blocks each finished request leaves, and a 1024-block pool
   exhausted and stalled admission (head-of-line `admissible` break) — that sizing is part of
   the fixture, not the engine.
4. The inversion ledger's `preempted` flag covers plan-preemptions observed by the simulators;
   a request preempted and re-admitted keeps its first-admission record.

## State

`scripts/gate.sh --base c7657649` → `gate: ok` on f20f4774 (233 tests). Nothing pushed. Next:
Task 10 (`feat(scheduler,model): multi-token append for running sequences and multi-row
logits`) from this tip.
