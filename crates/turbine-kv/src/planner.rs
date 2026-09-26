//! Recompute-vs-retrieve planner (P4 S-9, TS §8 "recompute as a virtual tier").

/// Why a plan chose its cutoff (P4 §Plan decision); label of `turbine_kv_plans_total{reason}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlanReason {
    AllL0,
    RetrieveCheaper,
    RecomputeCheaper,
    L0Pressure,
    TierDegraded,
    NoMatch,
}

impl PlanReason {
    pub const ALL: [PlanReason; 6] = [
        PlanReason::AllL0,
        PlanReason::RetrieveCheaper,
        PlanReason::RecomputeCheaper,
        PlanReason::L0Pressure,
        PlanReason::TierDegraded,
        PlanReason::NoMatch,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            PlanReason::AllL0 => "all_l0",
            PlanReason::RetrieveCheaper => "retrieve_cheaper",
            PlanReason::RecomputeCheaper => "recompute_cheaper",
            PlanReason::L0Pressure => "l0_pressure",
            PlanReason::TierDegraded => "tier_degraded",
            PlanReason::NoMatch => "no_match",
        }
    }
}
