//! Plan digests (`SimReport::plan_digest`) of the simulator's seeded workloads, recorded on
//! main before the scheduling policy moved behind `SchedulingPolicy` (Phase 2m Task 2, commit
//! 8b014bf). `sim::tests::default_policy_plan_digests_match_main` holds the `default` policy to
//! them: a change here means the default policy no longer plans what main planned.
//!
//! Re-record (only for an intended change of the default plans, with the reason in the commit):
//! `scripts/remote-cargo.sh test -p turbine-scheduler sim::tests::record_plan_digests --
//! --ignored --nocapture`. A workload of several runs digests the list of its runs' digests.

/// `(workload, plan digest)`, one per seeded workload of the invariant tests.
///
/// 2026-10-05 (P5p Task 10): `decode_never_starved` and `chunk_budget_respected` re-recorded —
/// their runs schedule a multi-token append every 7th iteration (`Simulation::with_extensions`,
/// S-9), so their plans differ from main's by design; every other workload is unchanged.
pub const MAIN_DIGESTS: &[(&str, u64)] = &[
    ("ts_section7_iteration_pattern", 15134313254342144953),
    ("decode_never_starved", 10772755057896660484),
    ("chunk_budget_respected", 15676167015522452859),
    ("preemption_by_recompute", 8945296441604707762),
    (
        "preemption_by_recompute_at_128_token_pages",
        5485881804937158422,
    ),
    (
        "cancellation_frees_within_one_iteration",
        7130878329829019534,
    ),
    ("bounded_under_overload", 12179297540124653319),
    ("closed_loop_keeps_concurrency", 2592856686799339000),
    ("phase2c_lab_configs_hold_invariants", 12310692654400566413),
];
