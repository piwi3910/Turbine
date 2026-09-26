//! The `scheduling_policy` conformance suite (Phase 2m S-13): the simulator invariants every
//! policy must keep, run over a registry — never a hand-written list — by
//! `registry_conformance::policies`, so a policy registered without passing it fails
//! `cargo test --workspace`. Test-only (the simulator's accounting pool needs host memory).
//! The ordering choices a policy is free to make (who is admitted or preempted first) are not
//! checked here; `sim::tests` pins the `default` policy's plans to main's.

use turbine_core::registry::{Registry, conformance};

use super::SchedulingPolicy;
use crate::scheduler::{BatchKind, SubmitError};
use crate::sim::workloads::{chunk_budget_runs, params, poisson_run, preemption_run};
use crate::sim::{SimReport, Simulation};

/// Runs every invariant over every policy of `reg`; `Err` lists each broken one as
/// `<policy>: <invariant>: <detail>`.
///
/// - `decode_never_starved`: over 1,000 seeded Poisson arrivals every decoding sequence that
///   was neither paused, preempted nor dropped decodes in every iteration (the simulator's
///   violation record), and every request completes or is rejected.
/// - `chunk_budget`: no prefill chunk exceeds `prefill_chunk_tokens` and no iteration
///   `max_batch_tokens`; long prompts are chunked; without chunked prefill an over-budget
///   prompt is rejected at submission and one that fits runs whole.
/// - `preemption_by_recompute` (16- and 128-token pages): on a pool too small for two requests
///   someone is preempted, re-prefills from position 0 over prompt + generated tokens, no token
///   is duplicated or skipped, and all three requests complete.
pub(crate) fn policies_suite(reg: &Registry<dyn SchedulingPolicy>) -> Result<(), Vec<String>> {
    let mut failures = Vec::new();
    if let Err(e) = conformance::check(reg) {
        failures.push(format!("registry: {e}"));
    }
    for policy in reg.iter() {
        let name = policy.name();
        let mut record = |invariant: &str, result: Result<(), String>| {
            if let Err(detail) = result {
                failures.push(format!("{name}: {invariant}: {detail}"));
            }
        };
        record(
            "decode_never_starved",
            never_starved(&poisson_run(policy, None)),
        );
        let (chunked, unchunked) = chunk_budget_runs(policy);
        record("chunk_budget", chunk_budget(&chunked, &unchunked));
        record(
            "preemption_by_recompute",
            recompute(&preemption_run(policy, 16, 12, 40, 100), 40),
        );
        record(
            "preemption_by_recompute",
            recompute(&preemption_run(policy, 128, 6, 200, 300), 200),
        );
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

fn ensure(cond: bool, detail: impl FnOnce() -> String) -> Result<(), String> {
    if cond { Ok(()) } else { Err(detail()) }
}

fn never_starved(report: &SimReport) -> Result<(), String> {
    ensure(report.violations.is_empty(), || {
        format!("{:?}", &report.violations[..report.violations.len().min(5)])
    })?;
    let done = report.completed as usize + report.rejected.len();
    ensure(done == 1000, || {
        format!("{done} of 1000 requests completed or rejected")
    })
}

fn chunk_budget(chunked: &SimReport, unchunked: &SimReport) -> Result<(), String> {
    let p = params();
    for it in &chunked.iterations {
        let mut tokens = 0;
        for item in &it.items {
            tokens += match item.kind {
                BatchKind::Prefill { len, .. } => {
                    ensure(len <= p.prefill_chunk_tokens, || {
                        format!("a chunk of {len} tokens in iteration {}", it.iteration)
                    })?;
                    len
                }
                BatchKind::Decode => 1,
            };
        }
        ensure(tokens <= p.max_batch_tokens, || {
            format!("{tokens} tokens in iteration {}", it.iteration)
        })?;
    }
    ensure(
        chunked.iterations.iter().any(|it| {
            it.items
                .iter()
                .any(|i| matches!(i.kind, BatchKind::Prefill { start, .. } if start > 0))
        }),
        || "no long prompt is chunked".into(),
    )?;
    ensure(
        unchunked.rejected == [(Simulation::request_id(0), SubmitError::PromptTooLong)],
        || {
            format!(
                "without chunked prefill the 600-token prompt is not refused: {:?}",
                unchunked.rejected
            )
        },
    )?;
    let first = unchunked.iterations.first().and_then(|it| it.items.first());
    ensure(
        first.map(|i| i.kind) == Some(BatchKind::Prefill { start: 0, len: 300 })
            && unchunked.completed == 1,
        || "without chunked prefill the 300-token prompt does not run whole".into(),
    )
}

fn recompute(report: &SimReport, prompt: u32) -> Result<(), String> {
    ensure(report.violations.is_empty(), || {
        format!("{:?}", report.violations)
    })?;
    ensure(report.completed == 3, || {
        format!("{} of 3 requests completed", report.completed)
    })?;
    let first = report
        .iterations
        .iter()
        .find_map(|it| it.preempted.first().map(|s| (it.iteration, *s)));
    let Some((at, victim)) = first else {
        return Err("the pool is too small for both, yet nobody is preempted".into());
    };
    let re_prefill: Vec<(u32, u32)> = report
        .iterations
        .iter()
        .filter(|it| it.iteration > at)
        .flat_map(|it| it.items.iter())
        .filter(|i| i.seq == victim)
        .filter_map(|i| match i.kind {
            BatchKind::Prefill { start, len } => Some((start, len)),
            BatchKind::Decode => None,
        })
        .collect();
    ensure(re_prefill.first().is_some_and(|c| c.0 == 0), || {
        format!("the victim's recompute does not start at position 0: {re_prefill:?}")
    })?;
    let recomputed: u32 = re_prefill.iter().map(|c| c.1).sum();
    ensure(recomputed > prompt, || {
        format!("the recompute covers {recomputed} tokens, not prompt + generated")
    })
}

#[cfg(test)]
mod tests {
    use turbine_core::registry::Module;

    use super::*;
    use crate::policy::{AdmissionInfo, AdmissionKey, DefaultPolicy, PreemptionRank, RunningInfo};
    use crate::scheduler::{IterationLimits, SchedulerParams};
    use turbine_core::types::RequestId;

    /// The default policy, except that its chunk cap ignores `prefill_chunk_tokens`.
    struct OversizedChunks;

    impl Module for OversizedChunks {
        fn name(&self) -> &'static str {
            "broken"
        }
    }

    impl SchedulingPolicy for OversizedChunks {
        fn admission_key(&self, req: &AdmissionInfo) -> AdmissionKey {
            DefaultPolicy.admission_key(req)
        }
        fn preemption_rank(&self, running: &RunningInfo) -> PreemptionRank {
            DefaultPolicy.preemption_rank(running)
        }
        fn pick_victim(&self, candidates: &[(RequestId, PreemptionRank)]) -> Option<RequestId> {
            DefaultPolicy.pick_victim(candidates)
        }
        fn chunk_cap(&self, params: &SchedulerParams, limits: &IterationLimits) -> u32 {
            DefaultPolicy.chunk_cap(params, limits) * 2
        }
    }

    static WITH_BROKEN: Registry<dyn SchedulingPolicy> =
        Registry::new("scheduling_policy", &[&DefaultPolicy, &OversizedChunks]);

    /// A policy whose chunks exceed the budget fails `chunk_budget` (and the other workloads'
    /// violation records, which flag every oversized chunk) under its own name; `default`
    /// passes. Breaks if the suite stops checking the chunk budget or iterates a fixed list
    /// instead of the registry.
    #[test]
    fn rejects_broken_policy() {
        let failures = policies_suite(&WITH_BROKEN).unwrap_err();
        assert!(
            failures.iter().all(|f| f.starts_with("broken: ")),
            "{failures:#?}"
        );
        assert!(
            failures
                .iter()
                .any(|f| f.starts_with("broken: chunk_budget: a chunk of 256 tokens")),
            "{failures:#?}"
        );
    }
}
