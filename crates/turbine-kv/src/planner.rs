//! Recompute-vs-retrieve planner (P4 S-9, TS §8 "recompute as a virtual tier").
//!
//! For a request's matched prefix the planner picks the cutoff k minimising
//! cost(k) = Σ retrieval of blocks 0..k from their fastest tier + recompute of tokens k·B..n,
//! from measured copy estimates and the prefill-throughput EWMA; a block served lossy costs its
//! retrieval × (1 + `kv.lossy_penalty` of its codec) (P6b S-3). It is a pure function of its
//! inputs; the caller logs every plan (`kv_plan`) and counts it (`turbine_kv_plans_total`).

use turbine_core::types::PressureState;

use crate::tier::TierId;

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

/// P4 §Plan decision, verbatim fields: reuse the first `reuse_l0` L0 blocks, promote the
/// counted blocks from each slower tier, recompute the remaining prompt tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct KvPlan {
    pub reuse_l0: u32,
    pub promote: Vec<(TierId, u32)>,
    pub recompute_tokens: u32,
    pub reason: PlanReason,
}

impl KvPlan {
    /// Matched blocks the plan uses (the cutoff k).
    pub fn cutoff_blocks(&self) -> u32 {
        self.reuse_l0 + self.promote.iter().map(|(_, n)| n).sum::<u32>()
    }
}

/// Estimate of one copy path into L0 (the transfer engine's EWMAs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathCost {
    pub latency_s: f64,
    pub bandwidth_bps: f64,
}

impl PathCost {
    /// Seconds to bring one block of `block_bytes` into L0.
    pub fn block_seconds(&self, block_bytes: u64) -> f64 {
        self.latency_s + block_bytes as f64 / self.bandwidth_bps.max(1.0)
    }
}

#[derive(Clone, Debug)]
pub struct PlanInputs<'a> {
    /// Fastest tier of each matched block, in prompt order.
    pub matched: &'a [TierId],
    pub prompt_tokens: u32,
    pub block_tokens: u32,
    pub block_bytes: u64,
    /// Phase 3 prefill-throughput EWMA (tokens/s).
    pub prefill_tps: f64,
    /// `None` when the path does not exist (L1 disabled; no L2).
    pub l1_to_l0: Option<PathCost>,
    pub l2_to_l0: Option<PathCost>,
    pub l0_state: PressureState,
    pub l1_degraded: bool,
    pub l2_degraded: bool,
    /// P6b S-1: bytes each matched block's copy moves into L0 (its tier format's size, the
    /// transfer engine's measured rates are per moved byte); blocks past the slice's end move
    /// `block_bytes`.
    pub copy_bytes: &'a [u64],
    /// P6b S-3: per matched block, `Some(kv.lossy_penalty of its codec)` when it is served
    /// lossy, `None` when exact; blocks past the slice's end are exact.
    pub lossy_penalty: &'a [Option<f64>],
    /// P6b S-3: false (`x-turbine-kv-lossy: deny`) — no block from the first lossy one on is
    /// reused.
    pub allow_lossy: bool,
}

impl PlanInputs<'_> {
    fn penalty(&self, i: usize) -> Option<f64> {
        self.lossy_penalty.get(i).copied().flatten()
    }

    /// Seconds to bring matched block `i` into L0, lossy penalty included.
    fn block_retrieval(&self, i: usize) -> f64 {
        let path = match self.matched[i] {
            TierId::L0 => return 0.0,
            TierId::L1 => self.l1_to_l0,
            TierId::L2 => self.l2_to_l0,
            TierId::L3 => None,
        };
        let bytes = self.copy_bytes.get(i).copied().unwrap_or(self.block_bytes);
        let seconds = path.map_or(f64::INFINITY, |p| p.block_seconds(bytes));
        seconds * (1.0 + self.penalty(i).unwrap_or(0.0))
    }

    fn degraded(&self, t: TierId) -> bool {
        (t == TierId::L1 && self.l1_degraded) || (t == TierId::L2 && self.l2_degraded)
    }

    /// The cutoff in `0..=max_k` of least cost; ties go to the larger k (more reuse).
    fn argmin(&self, max_k: usize) -> usize {
        let mut best = 0;
        let mut best_cost = plan_cost(self, 0);
        for k in 1..=max_k {
            let c = plan_cost(self, k);
            if c <= best_cost {
                best = k;
                best_cost = c;
            }
        }
        best
    }

    fn plan(&self, k: usize, reason: PlanReason) -> KvPlan {
        let mut reuse_l0 = 0;
        let mut promote: Vec<(TierId, u32)> = Vec::new();
        for &t in &self.matched[..k] {
            if t == TierId::L0 {
                reuse_l0 += 1;
            } else if let Some(e) = promote.iter_mut().find(|(x, _)| *x == t) {
                e.1 += 1;
            } else {
                promote.push((t, 1));
            }
        }
        KvPlan {
            reuse_l0,
            promote,
            recompute_tokens: self.prompt_tokens - k as u32 * self.block_tokens,
            reason,
        }
    }
}

/// Seconds to reuse the first `k` matched blocks (retrieving each from its fastest tier) and
/// recompute the rest of the prompt. `k` must not exceed the matched blocks nor the prompt.
pub fn plan_cost(inp: &PlanInputs<'_>, k: usize) -> f64 {
    let retrieve: f64 = (0..k).map(|i| inp.block_retrieval(i)).sum();
    let recompute_tokens = inp.prompt_tokens - k as u32 * inp.block_tokens;
    retrieve + f64::from(recompute_tokens) / inp.prefill_tps.max(1.0)
}

/// Prompt tokens a reuse plan always leaves to prefill. One would give the first output token its
/// logits; two keep the warm prefill a prefill-shaped step (more than one query row), so it runs
/// the same attention and GEMM paths as the cold whole-prompt prefill and reproduces its bits: a
/// one-token step is indistinguishable from a decode step and runs decode attention and the
/// decode GEMM rows (decision 2026-09-27, "Pre-Phase-5 #1 follow-up"). It costs one block of
/// reuse for prompts one token past a block boundary.
pub const MIN_RECOMPUTE_TOKENS: u32 = 2;

/// The most leading blocks a plan may reuse: [`MIN_RECOMPUTE_TOKENS`] stay to prefill.
pub fn reuse_cap(prompt_tokens: u32, block_tokens: u32) -> usize {
    (prompt_tokens.saturating_sub(MIN_RECOMPUTE_TOKENS) / block_tokens.max(1)) as usize
}

/// Chooses the cutoff minimising [`plan_cost`]. At least [`MIN_RECOMPUTE_TOKENS`] prompt tokens
/// are always recomputed; blocks at or behind a degraded tier are never
/// retrieved; at L0 RED or above only the leading L0 blocks are reused (no promotion into L0);
/// without `allow_lossy` nothing from the first lossy block on is reused.
pub fn plan_prefix(inp: &PlanInputs<'_>) -> KvPlan {
    let exact = if inp.allow_lossy {
        inp.matched.len()
    } else {
        (0..inp.matched.len())
            .position(|i| inp.penalty(i).is_some())
            .unwrap_or(inp.matched.len())
    };
    let cap = reuse_cap(inp.prompt_tokens, inp.block_tokens)
        .min(inp.matched.len())
        .min(exact);
    let matched = &inp.matched[..cap];
    if matched.is_empty() {
        return inp.plan(0, PlanReason::NoMatch);
    }
    let l0_prefix = matched.iter().take_while(|t| **t == TierId::L0).count();
    let usable = matched
        .iter()
        .position(|t| inp.degraded(*t))
        .unwrap_or(matched.len());
    let best = inp.argmin(usable);
    if inp.l0_state >= PressureState::Red && best > l0_prefix {
        return inp.plan(inp.argmin(l0_prefix), PlanReason::L0Pressure);
    }
    if usable < matched.len() && inp.argmin(matched.len()) > usable {
        return inp.plan(best, PlanReason::TierDegraded);
    }
    let reason = if best > l0_prefix {
        PlanReason::RetrieveCheaper
    } else if best < usable {
        PlanReason::RecomputeCheaper
    } else {
        PlanReason::AllL0
    };
    inp.plan(best, reason)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// L1 at 20 µs + 20 GB/s, L2 at 50 ms + 50 MB/s, 8,000 tok/s, Llama-3.2-3B blocks.
    fn inputs(matched: &[TierId], prompt_tokens: u32) -> PlanInputs<'_> {
        PlanInputs {
            matched,
            prompt_tokens,
            block_tokens: 16,
            block_bytes: 1_835_008,
            prefill_tps: 8_000.0,
            l1_to_l0: Some(PathCost {
                latency_s: 20e-6,
                bandwidth_bps: 20e9,
            }),
            l2_to_l0: Some(PathCost {
                latency_s: 0.05,
                bandwidth_bps: 50e6,
            }),
            l0_state: PressureState::Green,
            l1_degraded: false,
            l2_degraded: false,
            copy_bytes: &[],
            lossy_penalty: &[],
            allow_lossy: true,
        }
    }

    /// Deterministic xorshift64 for the brute-force comparison.
    fn next(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    #[test]
    fn cutoff_minimises_cost() {
        use TierId::{L0, L1, L2};
        let l1 = vec![L1; 1000];
        let p = plan_prefix(&inputs(&l1, 16_002));
        assert_eq!(p.reason, PlanReason::RetrieveCheaper);
        assert_eq!((p.reuse_l0, p.promote.as_slice()), (0, &[(L1, 1000)][..]));
        assert_eq!(p.recompute_tokens, 2);
        // One token past the last block: that block is recomputed too, so the warm prefill is
        // never a one-row (decode-shaped) step.
        let p = plan_prefix(&inputs(&l1, 16_001));
        assert_eq!((p.cutoff_blocks(), p.recompute_tokens), (999, 17));

        let l2 = vec![L2; 4];
        let p = plan_prefix(&inputs(&l2, 65));
        assert_eq!(p.reason, PlanReason::RecomputeCheaper, "slow disk");
        assert_eq!((p.cutoff_blocks(), p.recompute_tokens), (0, 65));

        let mixed = [L0, L0, L1, L1];
        let mut red = inputs(&mixed, 100);
        red.l0_state = PressureState::Red;
        let p = plan_prefix(&red);
        assert_eq!(p.reason, PlanReason::L0Pressure);
        assert_eq!(
            (p.reuse_l0, p.promote.len(), p.recompute_tokens),
            (2, 0, 68)
        );
        assert_eq!(
            plan_prefix(&inputs(&mixed, 100)).reason,
            PlanReason::RetrieveCheaper,
            "the same prefix at GREEN is retrieved"
        );

        // A fully cached prompt that is an exact multiple of the block: the last block is
        // recomputed so the first output token has logits.
        let all = [L0; 4];
        let p = plan_prefix(&inputs(&all, 64));
        assert_eq!(
            (p.reuse_l0, p.recompute_tokens, p.reason),
            (3, 16, PlanReason::AllL0)
        );
        assert_eq!(plan_prefix(&inputs(&[], 40)).reason, PlanReason::NoMatch);
        assert_eq!(
            plan_prefix(&inputs(&[L0], 10)).reason,
            PlanReason::NoMatch,
            "a prompt shorter than one block"
        );

        // A degraded tier that would have been retrieved from: recompute instead.
        let mut degraded = inputs(&mixed, 100);
        degraded.l1_degraded = true;
        let p = plan_prefix(&degraded);
        assert_eq!(p.reason, PlanReason::TierDegraded);
        assert_eq!((p.reuse_l0, p.promote.len()), (2, 0));

        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        for case in 0..200 {
            let n = (next(&mut seed) % 40) as usize;
            let tiers: Vec<TierId> = (0..n)
                .map(|_| match next(&mut seed) % 3 {
                    0 => L0,
                    1 => L1,
                    _ => L2,
                })
                .collect();
            let prompt = n as u32 * 16 + 1 + (next(&mut seed) % 200) as u32;
            let mut inp = inputs(&tiers, prompt);
            inp.prefill_tps = 500.0 + (next(&mut seed) % 20_000) as f64;
            inp.l2_to_l0 = Some(PathCost {
                latency_s: (next(&mut seed) % 100) as f64 * 1e-3,
                bandwidth_bps: 1e8 + (next(&mut seed) % 50) as f64 * 1e8,
            });
            let plan = plan_prefix(&inp);
            let brute = (0..=reuse_cap(prompt, 16).min(n))
                .map(|k| plan_cost(&inp, k))
                .fold(f64::INFINITY, f64::min);
            let chosen = plan_cost(&inp, plan.cutoff_blocks() as usize);
            assert!(
                chosen <= brute + 1e-12,
                "case {case}: chose {chosen}, brute force {brute}"
            );
            assert_eq!(
                plan.recompute_tokens,
                prompt - plan.cutoff_blocks() * 16,
                "case {case}"
            );
        }
    }

    /// P6b S-3: a lossy block is retrieved only while retrieval × (1 + penalty) is cheaper than
    /// recomputing it, and never without `allow_lossy`. Breaks if the penalty is ignored or an
    /// opted-out plan reuses a lossy block.
    #[test]
    fn lossy_penalty_weighs_retrieval() {
        use TierId::{L0, L1};
        let tiers = [L0, L1, L1, L1];
        let prompt = 4 * 16 + 2;
        // L1 at 1 ms per block against 2 ms of prefill per block (16 tokens at 8,000 tok/s).
        let slow = |p: &mut PlanInputs<'_>| {
            p.l1_to_l0 = Some(PathCost {
                latency_s: 1e-3,
                bandwidth_bps: 1e18,
            });
        };
        let lossy = [None, Some(0.5), Some(0.5), Some(0.5)];
        let mut inp = inputs(&tiers, prompt);
        slow(&mut inp);
        inp.lossy_penalty = &lossy;
        let p = plan_prefix(&inp);
        assert_eq!(p.cutoff_blocks(), 4, "1.5 ms < 2 ms: retrieved");
        let heavy = [None, Some(1.5), Some(1.5), Some(1.5)];
        inp.lossy_penalty = &heavy;
        let p = plan_prefix(&inp);
        assert_eq!(
            (p.cutoff_blocks(), p.reason),
            (1, PlanReason::RecomputeCheaper),
            "2.5 ms > 2 ms: recomputed"
        );
        let exact = [None; 4];
        inp.lossy_penalty = &exact;
        assert_eq!(plan_prefix(&inp).cutoff_blocks(), 4, "exact at 1 ms");
        // An opted-out plan stops at the first lossy block, however cheap.
        inp.lossy_penalty = &lossy;
        inp.allow_lossy = false;
        assert_eq!(plan_prefix(&inp).cutoff_blocks(), 1);
        // A smaller (encoded) copy moves fewer bytes: at a quarter of the block, a slow path
        // becomes worth it again.
        let mut inp = inputs(&tiers, prompt);
        inp.l1_to_l0 = Some(PathCost {
            latency_s: 0.0,
            bandwidth_bps: 1_835_008.0 / 3e-3,
        });
        assert_eq!(plan_prefix(&inp).cutoff_blocks(), 1, "3 ms per full block");
        let quarter = [1_835_008 / 4; 4];
        inp.copy_bytes = &quarter;
        inp.lossy_penalty = &lossy;
        assert_eq!(
            plan_prefix(&inp).cutoff_blocks(),
            4,
            "0.75 ms × 1.5 per block"
        );
        // A lossy L0 block costs nothing to retrieve.
        let l0 = [L0, L0];
        let costly = [Some(100.0); 2];
        let mut inp = inputs(&l0, 34);
        inp.lossy_penalty = &costly;
        assert_eq!(plan_prefix(&inp).cutoff_blocks(), 2);
    }
}
