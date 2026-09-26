# Adding a scheduling policy

A scheduling policy makes the scheduler's three free choices: the admission order of the waiting queue, the preemption victim, and the prefill chunk size. Everything else stays in the mechanism (`Scheduler::plan` in `crates/turbine-scheduler/src/scheduler.rs`): stage order (drop cancelled → decode → forks → continuing prefills → admit), block accounting, preemption eligibility and every bound. Point name `scheduling_policy`; selected by `scheduler.policy` (default `default`).

## The trait

`turbine_scheduler::policy::SchedulingPolicy: Module` (`crates/turbine-scheduler/src/policy/mod.rs`):

| Method                                          | Must do                                                                                                                                                                                                                                 |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name()` (from `Module`)                        | The configuration name, `^[a-z0-9_]{1,64}$`, unique in the registry.                                                                                                                                                                    |
| `admission_key(&AdmissionInfo)`                 | The waiting request's queue key; the smallest `AdmissionKey` is admitted first. Keys are compared whole, so keep them unique: put a counter (`submit_no`, or `push_no` for a request re-queued after preemption) in `order`.            |
| `preemption_rank(&RunningInfo)`                 | The running request's `PreemptionRank`; larger is dropped first. The mechanism lets a prefill or fork preempt only requests ranked above its own.                                                                                       |
| `pick_victim(&[(RequestId, PreemptionRank)])`   | The victim among candidates the mechanism already filtered (in admission order); `None` preempts nothing.                                                                                                                               |
| `chunk_cap(&SchedulerParams, &IterationLimits)` | The largest prefill chunk this iteration, at least 1, within `prefill_chunk_tokens` (chunked prefill) or `max_batch_tokens`, and within `limits.prefill_chunk_tokens` when set. Nothing checks it for you except the conformance suite. |

Policies are called per decision: cheap, deterministic, stateless — no clock, no randomness, no fields that change.

## Files to add

One file: `crates/turbine-scheduler/src/policy/<name>.rs` (see `crates/turbine-scheduler/src/policy/default.rs`):

```rust
//! `fifo`: arrival order, priorities ignored; the latest admission is preempted first.
use std::time::Duration;

use turbine_core::registry::Module;
use turbine_core::types::{Priority, RequestId};

use super::{AdmissionInfo, AdmissionKey, PreemptionRank, RunningInfo, SchedulingPolicy};
use crate::scheduler::{IterationLimits, SchedulerParams};

pub struct FifoPolicy;

impl Module for FifoPolicy {
    fn name(&self) -> &'static str {
        "fifo"
    }
}

impl SchedulingPolicy for FifoPolicy {
    fn admission_key(&self, req: &AdmissionInfo) -> AdmissionKey {
        AdmissionKey { tier: 0, priority: Priority(0), arrival: req.arrival, order: req.submit_no }
    }
    fn preemption_rank(&self, r: &RunningInfo) -> PreemptionRank {
        PreemptionRank(Priority(0), r.admitted.unwrap_or(u64::MAX))
    }
    fn pick_victim(&self, c: &[(RequestId, PreemptionRank)]) -> Option<RequestId> {
        c.iter().max_by_key(|(_, rank)| *rank).map(|(id, _)| *id)
    }
    fn chunk_cap(&self, p: &SchedulerParams, l: &IterationLimits) -> u32 {
        let cap = if p.chunked_prefill { p.prefill_chunk_tokens } else { p.max_batch_tokens };
        l.prefill_chunk_tokens.map_or(cap, |c| c.min(cap)).max(1)
    }
}
```

## Registry entry

In `crates/turbine-scheduler/src/policy/mod.rs`: add `mod <name>;` and `pub use <name>::<Type>;` next to `mod default;`, and append `&<Type>` to the `REGISTRY` list of `registry()`:

```rust
static REGISTRY: Registry<dyn SchedulingPolicy> =
    Registry::new("scheduling_policy", &[&DefaultPolicy, &FifoPolicy]);
```

Nothing else: the server validates `scheduler.policy` against `registry().names()` (`crates/turbine-server/src/modules.rs`) and the engine selects it by name, logging `event="module_selected" point=scheduling_policy`. Keep `default` first; it is the configuration default.

## Conformance suite

`policies_suite` (`crates/turbine-scheduler/src/policy/conformance.rs`) runs every registered policy through the deterministic simulator: `decode_never_starved` (1,000 seeded Poisson arrivals, every running sequence decodes each iteration, everything completes or is rejected), `chunk_budget` (no chunk above `prefill_chunk_tokens`, no iteration above `max_batch_tokens`, over-budget prompts rejected without chunked prefill) and `preemption_by_recompute` (16- and 128-token pages: re-prefill from position 0 over prompt + generated, no token duplicated or skipped). The simulator tests in `crates/turbine-scheduler/src/sim/mod.rs` (`sim::tests`) also iterate the registry (cancellation, overload bounds, closed-loop concurrency).

- `scripts/remote-cargo.sh test -p turbine-scheduler registry_conformance` — the suite over the registry.
- `scripts/remote-cargo.sh test -p turbine-scheduler sim::tests` — the simulator invariants over every policy.

## Lab checks

A new policy changes nothing until `scheduler.policy` names it, so landing it needs no lab run. Before recommending it for serving, measure it on novanas GPU 0 for both models: `scripts/lab-bench.sh --gpu 0 --model llama -- --set scheduler.policy=<name>` and the same with `--model olmoe` (golden c1 strict and c16 batched, then the fixed throughput bench). If you change `default` itself, both runs without `--set` must stay within 3% tok/s and 10% TTFT p50 of the last row of `.procoder/perf-log.md`, and the plan digests pinned by `default_policy_plan_digests_match_main` must not move.

## Pitfalls

- Priority is a policy input, not a mechanism rule: ignoring it (as `fifo` does) is allowed, but the ordering assertions of `sim::tests` apply only to `default` (`orders_like_default`); your own ordering needs its own unit test in your file.
- Non-unique keys make the queue order depend on insertion; always end the key with a counter.
- A preempted request is re-queued with `preempted: true` and a new `push_no`; `default` puts it ahead of every new request. If yours does not, a preempted request can wait behind new arrivals — the suite still requires it to complete.
- `chunk_cap` returning 0 or above the budgets fails `chunk_budget`; never read a clock or RNG (the digests of `default_policy_plan_digests_match_main` and the determinism checks rely on replayable runs).
- `Scheduler::with_policy` must be called before the first submission: queued requests keep the keys of the policy they were queued under.
