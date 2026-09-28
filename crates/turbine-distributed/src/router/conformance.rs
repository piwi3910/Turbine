//! The behaviour every registered DP router policy must show, run over the registry by
//! `registry_conformance` (the naming rules are `turbine_core::registry::conformance`). Every
//! check goes through [`route`], as the server calls it.

use turbine_core::types::{CircuitState, PressureState, ReplicaId};

use super::{DpRouteReason, ReplicaView, RouterPolicy, route};

fn view(
    i: u32,
    state: PressureState,
    circuit: CircuitState,
    outstanding: u64,
    prefix: bool,
) -> ReplicaView {
    ReplicaView {
        replica: ReplicaId(i),
        state,
        circuit,
        outstanding_tokens: outstanding,
        has_prefix: prefix,
    }
}

/// Views where an ineligible replica is the most attractive (idle, holding the prefix) and an
/// eligible one exists. Replica indices are sparse and unordered, as the server passes only the
/// ready replicas.
fn attractive_ineligible() -> Vec<Vec<ReplicaView>> {
    use CircuitState::{CircuitOpen, Draining, Healthy, Probing};
    use PressureState::{Green, Orange, Red, Survival, Yellow};
    vec![
        vec![
            view(0, Orange, Healthy, 0, true),
            view(1, Green, Healthy, 900, false),
        ],
        vec![
            view(3, Green, CircuitOpen, 0, true),
            view(1, Yellow, Healthy, 50, false),
        ],
        vec![
            view(2, Green, Draining, 0, true),
            view(5, Red, Healthy, 0, true),
            view(7, Survival, Healthy, 0, false),
            view(9, Yellow, Probing, 10_000, false),
        ],
        vec![
            view(4, Red, Healthy, 0, true),
            view(6, Green, Healthy, 1, true),
            view(8, Green, Healthy, 0, false),
        ],
    ]
}

/// Views where no replica is eligible: the policy must still pick one of them.
fn none_eligible() -> Vec<Vec<ReplicaView>> {
    use CircuitState::{CircuitOpen, Draining, Healthy};
    use PressureState::{Green, Orange, Red};
    vec![
        vec![
            view(0, Red, Healthy, 70, false),
            view(1, Red, Healthy, 20, true),
            view(2, Orange, Healthy, 40, false),
        ],
        vec![
            view(4, Green, CircuitOpen, 0, true),
            view(2, Green, Draining, 5, false),
        ],
    ]
}

/// `Err` naming the first problem. A policy, through [`route`], must:
/// - answer one replica with that replica and `only_candidate`, whatever its state;
/// - pick a replica of the slice, never with `only_candidate` among several;
/// - never pick an ineligible replica (ORANGE or worse, circuit open or draining) while an
///   eligible one exists, however idle or prefix-holding the ineligible one is;
/// - pick some replica when none is eligible (every replica is then a candidate);
/// - be deterministic and independent of the slice order;
/// - break a tie between identical replicas by the lowest replica index.
pub fn check(policy: &dyn RouterPolicy) -> Result<(), String> {
    use CircuitState::{CircuitOpen, Healthy};
    use PressureState::{Green, Red};
    let name = policy.name();

    let alone = [view(3, Red, CircuitOpen, 99, false)];
    let got = route(&alone, policy);
    if got != (ReplicaId(3), DpRouteReason::OnlyCandidate) {
        return Err(format!("{name}: one replica gave {got:?}"));
    }

    let mut all = attractive_ineligible();
    all.extend(none_eligible());
    for views in &all {
        let (replica, reason) = route(views, policy);
        let Some(chosen) = views.iter().find(|v| v.replica == replica) else {
            return Err(format!("{name}: picked {replica:?}, not in {views:?}"));
        };
        if reason == DpRouteReason::OnlyCandidate {
            return Err(format!(
                "{name}: only_candidate among {} replicas",
                views.len()
            ));
        }
        if views.iter().any(ReplicaView::eligible) && !chosen.eligible() {
            return Err(format!(
                "{name}: picked ineligible {chosen:?} while an eligible replica exists in {views:?}"
            ));
        }
        if route(views, policy) != (replica, reason) {
            return Err(format!("{name}: two calls on {views:?} disagree"));
        }
        let mut reversed = views.clone();
        reversed.reverse();
        if route(&reversed, policy).0 != replica {
            return Err(format!(
                "{name}: the pick depends on the slice order of {views:?}"
            ));
        }
    }

    for prefix in [false, true] {
        let tied: Vec<ReplicaView> = [5, 2, 7]
            .into_iter()
            .map(|i| view(i, Green, Healthy, 100, prefix))
            .collect();
        let (replica, _) = route(&tied, policy);
        if replica != ReplicaId(2) {
            return Err(format!(
                "{name}: identical replicas 5, 2, 7 (prefix {prefix}) gave {replica:?}, not the lowest"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use turbine_core::registry::Module;

    use super::*;

    /// Always the first view: ignores pressure and circuits, so the suite must refuse it.
    struct First;
    impl Module for First {
        fn name(&self) -> &'static str {
            "first"
        }
    }
    impl RouterPolicy for First {
        fn choose(&self, views: &[ReplicaView]) -> (ReplicaId, DpRouteReason) {
            (views[0].replica, DpRouteReason::LeastLoaded)
        }
    }

    #[test]
    fn conformance_rejects_broken_policy() {
        let err = check(&First).expect_err("a policy that ignores eligibility");
        assert!(err.starts_with("first: picked ineligible"), "{err}");
    }
}
