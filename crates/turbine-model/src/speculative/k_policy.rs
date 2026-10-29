//! Adaptive speculative-length (`k`) policies (P8): how many tokens the proposer drafts per
//! round. The `fixed` policy (default) uses `speculative.num_tokens` unchanged; `adaptive`
//! raises `k` after fully accepted rounds and lowers it when proposals are wasted, damped with
//! an EWMA and a dwell so it does not oscillate. The engine keeps one [`KState`] per sequence.
//!
//! Registration follows the repo convention ([`turbine_core::registry`]): one file, a
//! [`Module`] name, a compiled-in [`Registry`]; `speculative.k_policy` holds the name.
use turbine_core::config::MAX_SPECULATIVE_TOKENS;
use turbine_core::registry::{Module, Registry};

/// Per-sequence `k` controller state (the policy itself is a stateless singleton).
#[derive(Clone, Debug, PartialEq)]
pub struct KState {
    k: u32,
    /// Consecutive fully accepted rounds (all `k` proposals survived).
    all: u32,
    /// Consecutive fully rejected rounds (not even the first proposal survived).
    none: u32,
}

impl KState {
    /// A state starting at `start` proposals per round (clamped to `1..=MAX`).
    pub fn new(start: u32) -> KState {
        KState {
            k: start.clamp(1, MAX_SPECULATIVE_TOKENS),
            all: 0,
            none: 0,
        }
    }

    /// The current proposal count (before the round's ceiling is applied).
    pub fn k(&self) -> u32 {
        self.k
    }
}

/// What a policy reads to choose this round's `k`.
#[derive(Clone, Debug)]
pub struct KChoice {
    /// The sequence's controller state.
    pub state: KState,
    /// The hard ceiling this round (`MAX_SPECULATIVE_TOKENS`, the context room, …).
    pub cap: u32,
}

/// A speculative-length policy: one [`Module`] per strategy, selected by name.
pub trait KPolicy: Module {
    /// The number of proposals to draft this round (`1..=choice.cap`).
    fn choose(&self, choice: KChoice) -> u32;
    /// Feed back the round's outcome (proposals drafted, proposals the target accepted).
    fn observe(&self, state: &mut KState, proposed: u32, accepted: u32);
}

/// The `fixed` policy: `k` never changes (`speculative.num_tokens`).
pub struct FixedPolicy;

impl Module for FixedPolicy {
    fn name(&self) -> &'static str {
        "fixed"
    }
}

impl KPolicy for FixedPolicy {
    fn choose(&self, choice: KChoice) -> u32 {
        choice.state.k.clamp(1, choice.cap.max(1))
    }

    fn observe(&self, _state: &mut KState, _proposed: u32, _accepted: u32) {}
}

/// The `adaptive` policy: a deadband controller on the round outcome.
///
/// Objective is accepted tokens per unit time. A partial round (the accepted prefix is shorter
/// than `k`) is the normal, near-optimal case — the drafts past the prefix were cheap and the
/// verify was one batched forward — so it holds `k`. Only a *sustained* signal moves it: `GROW`
/// consecutive fully accepted rounds say "draft one more", `SHRINK` consecutive fully rejected
/// ones say "draft fewer". This tracks easy (boilerplate) and hard (novel) stretches but stays
/// put on a workload whose optimum is near the configured `k`, instead of oscillating.
pub struct AdaptivePolicy;

impl Module for AdaptivePolicy {
    fn name(&self) -> &'static str {
        "adaptive"
    }
}

/// Fully accepted rounds in a row that raise `k` by one. Deliberately long: a workload whose
/// optimum is near the configured `k` accepts a prefix and rejects the tail, so it never runs
/// this many fully accepted rounds and holds `k`; only genuinely easy stretches (boilerplate,
/// code, repetition) grow it.
const GROW: u32 = 6;
/// Fully rejected rounds in a row that lower `k` by one. Rejections cluster at the start of a
/// generation (and after every accepted prefix there is a tail reject), so a couple of them must
/// not drag `k` down; require a long losing streak of rounds where even the first proposal
/// failed.
const SHRINK: u32 = 8;

impl KPolicy for AdaptivePolicy {
    fn choose(&self, choice: KChoice) -> u32 {
        choice.state.k.clamp(1, choice.cap.max(1))
    }

    fn observe(&self, state: &mut KState, proposed: u32, accepted: u32) {
        if proposed == 0 {
            return;
        }
        if accepted == proposed {
            state.all += 1;
            state.none = 0;
        } else if accepted == 0 {
            state.none += 1;
            state.all = 0;
        } else {
            state.all = 0;
            state.none = 0;
        }
        if state.all >= GROW && state.k < MAX_SPECULATIVE_TOKENS {
            state.k += 1;
            state.all = 0;
        } else if state.none >= SHRINK && state.k > 1 {
            state.k -= 1;
            state.none = 0;
        }
    }
}

/// The `speculative.k_policy` registry.
pub static KPOLICIES: Registry<dyn KPolicy> =
    Registry::new("speculative_k_policy", &[&FixedPolicy, &AdaptivePolicy]);

/// The registered policy named `name`.
pub fn registry() -> &'static Registry<dyn KPolicy> {
    &KPOLICIES
}

/// The registered policy named `name`.
pub fn policy(name: &str) -> Option<&'static dyn KPolicy> {
    KPOLICIES.get(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(policy: &dyn KPolicy, state: &mut KState, cap: u32, rounds: &[(u32, u32)]) -> Vec<u32> {
        rounds
            .iter()
            .map(|&(proposed, accepted)| {
                let k = policy.choose(KChoice {
                    state: state.clone(),
                    cap,
                });
                policy.observe(state, proposed, accepted);
                k
            })
            .collect()
    }

    #[test]
    fn fixed_never_changes() {
        let mut s = KState::new(4);
        let ks = run(&FixedPolicy, &mut s, 8, &[(4, 4), (4, 0), (4, 2), (4, 4)]);
        assert_eq!(ks, [4, 4, 4, 4]);
    }

    #[test]
    fn adaptive_climbs_on_all_accepted_and_backs_off_on_rejects() {
        let mut s = KState::new(2);
        // Fully accepted rounds grow k towards the cap (damped by the dwell).
        let up = run(&AdaptivePolicy, &mut s, 8, &[(2, 2); 12]);
        assert!(up.iter().all(|&k| (1..=8).contains(&k)));
        assert!(s.k() > 2, "k grew: {}", s.k());
        assert!(s.k() <= 8);
        // Fully rejected rounds shrink k back to 1.
        let mut s = KState::new(6);
        let down = run(&AdaptivePolicy, &mut s, 8, &[(6, 0); 80]);
        assert!(down.iter().all(|&k| (1..=8).contains(&k)));
        assert_eq!(s.k(), 1);
    }

    #[test]
    fn registry_conformance() {
        assert_eq!(registry().names(), ["fixed", "adaptive"]);
        for policy in [&FixedPolicy as &dyn KPolicy, &AdaptivePolicy] {
            for start in 1..=8u32 {
                for cap in 1..=8u32 {
                    let k = policy.choose(KChoice {
                        state: KState::new(start),
                        cap,
                    });
                    assert!(
                        (1..=cap).contains(&k),
                        "{}: {k} not in 1..={cap}",
                        policy.name()
                    );
                }
            }
        }
    }

    #[test]
    fn adaptive_is_capped_by_the_round_ceiling() {
        let mut s = KState::new(8);
        let ks = run(&AdaptivePolicy, &mut s, 3, &[(8, 8); 6]);
        assert!(ks.iter().all(|&k| k <= 3), "{ks:?}");
    }
}
