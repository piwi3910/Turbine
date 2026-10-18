//! Recompute-vs-retrieve planner (P4 S-9, TS §8 "recompute as a virtual tier").
//!
//! For a request's matched prefix the planner picks the cutoff k minimising
//! cost(k) = Σ retrieval of blocks 0..k from their fastest tier + recompute of tokens k·B..n,
//! from measured copy estimates and the prefill-throughput EWMA; a block served lossy costs its
//! retrieval × (1 + `kv.lossy_penalty` of its codec) (P6b S-3). A path's latency is paid once
//! per plan and shared by the plan's blocks on that path (their copies are pipelined; on a GPU
//! the latency estimate is mostly the wait behind other copies on the same copy stream), the
//! bytes once per block (decision "6b planner follow-ups", 1 A). It is a pure function of its
//! inputs; the caller logs every plan (`kv_plan`) and counts it (`turbine_kv_plans_total`).

use turbine_core::types::PressureState;

use crate::state::SnapshotRef;
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
    /// P5p S-4/S-5: tokens reused beyond the matched full blocks through a tail copy of the
    /// block at `cutoff_blocks()` (a matched full block, or the directory's partial match
    /// when the cutoff keeps every matched block).
    pub partial_tokens: u32,
    /// P7 S-12 (hybrid families): the snapshot a prefix hit restores, when the model keeps
    /// per-sequence state and the hit reached a block that carries one. `None` for
    /// attention-only families and for a hit with no snapshot at or before its cutoff.
    pub state_snapshot: Option<SnapshotRef>,
    /// P7 S-12: how many matched blocks the restored state already covers — the state recompute
    /// starts at the token `state_recompute_from × kv.block_tokens` of the prompt (the tokens
    /// after the restored snapshot are re-driven through the recurrent layers). `None` for
    /// attention-only families. A hybrid plan's `recompute_tokens` starts here, not at the
    /// cutoff (KV blocks between the snapshot and the cutoff are re-written in place).
    pub state_recompute_from: Option<u32>,
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
    /// P5p S-4: tokens the directory matched beyond the matched full blocks (the token
    /// comparison of the last block, [`crate::directory::PrefixMatch::partial`]); 0 when the
    /// comparison found nothing.
    pub partial_tokens: u32,
    /// P7 S-12: the model keeps per-sequence state (a hybrid family). A hybrid hit reuses whole
    /// blocks only — no tail copy — and restores the nearest snapshot at or before its cutoff.
    pub hybrid: bool,
    /// P7 S-12: snapshot each matched block carries, parallel to `matched` (`None` for a block
    /// without one, and for attention-only families). A hybrid hit's cutoff still counts the
    /// matched blocks; the state recompute starts at the nearest snapshot's boundary.
    pub snapshots: &'a [Option<SnapshotRef>],
}

/// The nearest snapshot at or before matched-block index `at` (the largest `i <= at` with a
/// snapshot), with its index. `None` when no matched block at or before `at` carries one.
pub fn nearest_snapshot(
    snapshots: &[Option<SnapshotRef>],
    at: usize,
) -> Option<(usize, SnapshotRef)> {
    snapshots
        .iter()
        .take(at + 1)
        .enumerate()
        .rev()
        .find_map(|(i, s)| s.map(|r| (i, r)))
}

impl PlanInputs<'_> {
    fn penalty(&self, i: usize) -> Option<f64> {
        self.lossy_penalty.get(i).copied().flatten()
    }

    /// Seconds to bring the first `k` matched blocks into L0, lossy penalties included: each
    /// path's latency once, shared by the blocks it carries (each block's share weighted by its
    /// own penalty), plus every block's bytes over its path's rate.
    fn retrieval(&self, k: usize) -> f64 {
        // Per path into L0 (L1, L2): blocks, Σ (1 + penalty), Σ bytes × (1 + penalty).
        let mut sums = [(0u32, 0.0f64, 0.0f64); 2];
        for i in 0..k {
            let slot = match self.matched[i] {
                TierId::L0 => continue,
                TierId::L1 => 0,
                TierId::L2 => 1,
                TierId::L3 => return f64::INFINITY,
            };
            let weight = 1.0 + self.penalty(i).unwrap_or(0.0);
            let bytes = self.copy_bytes.get(i).copied().unwrap_or(self.block_bytes);
            let s = &mut sums[slot];
            s.0 += 1;
            s.1 += weight;
            s.2 += bytes as f64 * weight;
        }
        let mut total = 0.0;
        for (path, (blocks, weights, bytes)) in [self.l1_to_l0, self.l2_to_l0].into_iter().zip(sums)
        {
            if blocks == 0 {
                continue;
            }
            let Some(p) = path else {
                return f64::INFINITY;
            };
            total += p.latency_s * weights / f64::from(blocks) + bytes / p.bandwidth_bps.max(1.0);
        }
        total
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

    /// The token-granular extension of a chosen whole-block cutoff (P5p S-4, S-5): reuse the
    /// first `partial_tokens` of the block at the cutoff through a tail copy, capped so
    /// `full + partial` stays within [`reuse_cap_tokens`] and within the matched coverage.
    /// Fires only when the cache reaches the end of the prompt: a partial candidate matched
    /// beyond the full blocks, or the prompt ends exactly on the cached full blocks and the
    /// block at the cutoff is itself the tail source. A prompt that continues past what the
    /// cache covers, with nothing matched beyond the full blocks, keeps the Phase 4
    /// whole-block plan.
    ///
    /// The source must be an L0-resident exact copy: a matched full block below the cutoff
    /// (its tier is L0 and it is served exact), or the directory's partial match (L0-resident
    /// and exact by construction). One would give the first output token its logits; the cap
    /// keeps the warm prefill at least [`MIN_RECOMPUTE_TOKENS`] tokens, so it stays
    /// prefill-shaped.
    fn extend_partial(&self, plan: &mut KvPlan) {
        let b = self.block_tokens.max(1);
        let covered = self.matched.len() as u32 * b + self.partial_tokens;
        if self.partial_tokens == 0 && covered != self.prompt_tokens {
            return;
        }
        let k = plan.cutoff_blocks() as usize;
        let eligible = if k < self.matched.len() {
            self.matched[k] == TierId::L0 && self.penalty(k).is_none()
        } else {
            self.partial_tokens > 0
        };
        if !eligible {
            return;
        }
        let used = k as u32 * b;
        let cap = reuse_cap_tokens(self.prompt_tokens);
        if used >= cap {
            return;
        }
        let p = (cap - used).min(b).min(covered - used);
        if p == 0 {
            return;
        }
        plan.partial_tokens = p;
        plan.recompute_tokens -= p;
    }

    /// The plan's final form: the tail-copy extension for attention-only families, or the
    /// hybrid (P7 S-12) rule — whole-block reuse only (no tail copy) and the state recompute
    /// starting at the nearest snapshot at or before the cutoff. With no such snapshot the
    /// prefix's state cannot be restored, so the state is recomputed from zero.
    fn finalize(&self, mut plan: KvPlan) -> KvPlan {
        if !self.hybrid {
            self.extend_partial(&mut plan);
            return plan;
        }
        plan.partial_tokens = 0;
        let at = plan.cutoff_blocks().saturating_sub(1) as usize;
        match nearest_snapshot(self.snapshots, at) {
            Some((i, snapshot)) => {
                plan.state_snapshot = Some(snapshot);
                plan.state_recompute_from = Some(i as u32 + 1);
                plan.recompute_tokens = self.prompt_tokens - (i as u32 + 1) * self.block_tokens;
            }
            None => {
                plan.state_recompute_from = Some(0);
                plan.recompute_tokens = self.prompt_tokens;
            }
        }
        plan
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
            partial_tokens: 0,
            state_snapshot: None,
            state_recompute_from: self.hybrid.then_some(0),
        }
    }
}

/// Seconds to reuse the first `k` matched blocks (retrieving each from its fastest tier, each
/// path's latency once) and recompute the rest of the prompt. `k` must not exceed the matched blocks nor the prompt.
pub fn plan_cost(inp: &PlanInputs<'_>, k: usize) -> f64 {
    let retrieve = inp.retrieval(k);
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

/// The most tokens a plan may reuse (whole blocks plus the partial tail): [`MIN_RECOMPUTE_TOKENS`]
/// stay to prefill (P5p S-5).
pub fn reuse_cap_tokens(prompt_tokens: u32) -> u32 {
    prompt_tokens.saturating_sub(MIN_RECOMPUTE_TOKENS)
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
        return inp.finalize(inp.plan(0, PlanReason::NoMatch));
    }
    let l0_prefix = matched.iter().take_while(|t| **t == TierId::L0).count();
    let usable = matched
        .iter()
        .position(|t| inp.degraded(*t))
        .unwrap_or(matched.len());
    let best = inp.argmin(usable);
    if inp.l0_state >= PressureState::Red && best > l0_prefix {
        return inp.finalize(inp.plan(inp.argmin(l0_prefix), PlanReason::L0Pressure));
    }
    if usable < matched.len() && inp.argmin(matched.len()) > usable {
        return inp.finalize(inp.plan(best, PlanReason::TierDegraded));
    }
    let reason = if best > l0_prefix {
        PlanReason::RetrieveCheaper
    } else if best < usable {
        PlanReason::RecomputeCheaper
    } else {
        PlanReason::AllL0
    };
    inp.finalize(inp.plan(best, reason))
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
            partial_tokens: 0,
            hybrid: false,
            snapshots: &[],
        }
    }

    fn snap(block: u32) -> SnapshotRef {
        SnapshotRef {
            block: turbine_core::types::BlockId(block),
            entry: block,
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

        // A fully cached prompt that is an exact multiple of the block: the last
        // MIN_RECOMPUTE_TOKENS are recomputed so the first output token has logits. Phase 4
        // recomputed the whole last block (16 here); P5p token-granular reuse (S-4/S-5, user
        // decision 2026-09-28) amends exactly this case to a tail copy of the block's first
        // `B − MIN_RECOMPUTE_TOKENS` tokens.
        let all = [L0; 4];
        let p = plan_prefix(&inputs(&all, 64));
        assert_eq!(
            (p.reuse_l0, p.recompute_tokens, p.partial_tokens, p.reason),
            (3, 2, 14, PlanReason::AllL0)
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

    /// Decision "6b planner follow-ups", 1 A: a path's latency (on the server mostly the shared
    /// wait behind demotions on the same copy stream) is paid once by the blocks of a plan that
    /// come over that path, since their copies are pipelined; only the bytes are paid per
    /// block. Breaks if the latency is charged once per block again (a 30 ms wait then prices
    /// every 128-token block above recomputing it and the 20-block prefix is recomputed).
    #[test]
    fn path_latency_is_charged_once_per_plan() {
        use TierId::L1;
        let bytes = 14_680_064u64;
        let l1 = vec![L1; 20];
        let mut inp = inputs(&l1, 20 * 128 + 2);
        inp.block_tokens = 128;
        inp.block_bytes = bytes;
        inp.prefill_tps = 9_000.0;
        inp.l1_to_l0 = Some(PathCost {
            latency_s: 0.03,
            bandwidth_bps: 10.45e9,
        });
        let per_block = bytes as f64 / 10.45e9;
        let want = 0.03 + 20.0 * per_block + 2.0 / 9_000.0;
        let got = plan_cost(&inp, 20);
        assert!(
            (got - want).abs() < 1e-9,
            "latency once plus 20 blocks of bytes: {got} s vs {want} s"
        );
        let p = plan_prefix(&inp);
        assert_eq!(
            (p.cutoff_blocks(), p.reason),
            (20, PlanReason::RetrieveCheaper),
            "58 ms of copies against 284 ms of prefill"
        );
        // One block still pays the whole latency: 31 ms against 14 ms of prefill.
        let one = [L1];
        let mut inp = inputs(&one, 128 + 2);
        inp.block_tokens = 128;
        inp.block_bytes = bytes;
        inp.prefill_tps = 9_000.0;
        inp.l1_to_l0 = Some(PathCost {
            latency_s: 0.03,
            bandwidth_bps: 10.45e9,
        });
        assert_eq!(plan_prefix(&inp).reason, PlanReason::RecomputeCheaper);
        // Two paths each pay their own latency once.
        let mixed = [L1, TierId::L2, L1, TierId::L2];
        let inp = inputs(&mixed, 4 * 16 + 2);
        let (l1c, l2c) = (inp.l1_to_l0.unwrap(), inp.l2_to_l0.unwrap());
        let want = l1c.latency_s
            + l2c.latency_s
            + 2.0 * 1_835_008.0 / l1c.bandwidth_bps
            + 2.0 * 1_835_008.0 / l2c.bandwidth_bps
            + 2.0 / 8_000.0;
        assert!((plan_cost(&inp, 4) - want).abs() < 1e-9);
    }

    /// P6b S-3: a lossy block is retrieved only while retrieval × (1 + penalty) is cheaper than
    /// recomputing it, and never without `allow_lossy`. Breaks if the penalty is ignored or an
    /// opted-out plan reuses a lossy block.
    #[test]
    fn lossy_penalty_weighs_retrieval() {
        use TierId::{L0, L1};
        let tiers = [L0, L1, L1, L1];
        let prompt = 4 * 16 + 2;
        // L1 at 1 ms per block (all of it bytes: a path's latency is paid once per plan)
        // against 2 ms of prefill per block (16 tokens at 8,000 tok/s).
        let slow = |p: &mut PlanInputs<'_>| {
            p.l1_to_l0 = Some(PathCost {
                latency_s: 0.0,
                bandwidth_bps: 1_835_008.0 / 1e-3,
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
    /// P5p S-4/S-5 (`token_granular_cap`): the planner's reuse cap is token-granular — a
    /// prompt one token past a block boundary recomputes [`MIN_RECOMPUTE_TOKENS`] tokens by
    /// splitting a matched block through a tail copy; without a partial candidate (and when
    /// the prompt is not an exact multiple of the block) the plan is the Phase 4 whole-block
    /// minimum. The warm prefill is never shorter than [`MIN_RECOMPUTE_TOKENS`]. Breaks if
    /// the extension reuses past [`MIN_RECOMPUTE_TOKENS`], past the matched coverage, or
    /// changes the Phase 4 plan when no partial candidate exists.
    #[test]
    fn token_granular_cap() {
        use TierId::L0;
        // A fully cached prompt: `matched` holds every full block of the prompt, L0-resident,
        // plus `partial_tokens` matched beyond them (the trailing chunk, or 0 when the
        // prompt ends on a block boundary).
        for (prompt, full, partial) in [
            (129u32, 0u32, 127u32),
            (130, 1, 0),
            (255, 1, 125),
            (256, 1, 126),
            (257, 1, 127),
        ] {
            let matched = vec![L0; (prompt / 128) as usize];
            let mut inp = inputs(&matched, prompt);
            inp.block_tokens = 128;
            inp.partial_tokens = prompt % 128;
            let p = plan_prefix(&inp);
            assert_eq!(
                (p.cutoff_blocks(), p.partial_tokens, p.recompute_tokens),
                (full, partial, prompt - full * 128 - partial),
                "prompt {prompt}"
            );
            assert!(
                p.recompute_tokens >= MIN_RECOMPUTE_TOKENS,
                "prompt {prompt}"
            );
        }

        // Without a partial candidate the plan is the whole-block minimum (the Phase 4
        // behaviour, judged on cost and on the cutoff itself).
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        for case in 0..200 {
            let n = (next(&mut seed) % 40) as usize;
            let tiers: Vec<TierId> = (0..n)
                .map(|_| match next(&mut seed) % 3 {
                    0 => TierId::L0,
                    1 => TierId::L1,
                    _ => TierId::L2,
                })
                .collect();
            let prompt = n as u32 * 16 + 1 + (next(&mut seed) % 200) as u32;
            let mut inp = inputs(&tiers, prompt);
            inp.partial_tokens = 0;
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
            assert_eq!(plan.partial_tokens, 0, "case {case}");
            assert!(plan.recompute_tokens >= MIN_RECOMPUTE_TOKENS);
        }

        // The split source must be an L0-resident exact copy: an L1-resident matched block is
        // never split (the tail copy is a device-side copy), and a lossy-served block neither.
        let mut inp = inputs(&[TierId::L1], 130);
        inp.block_tokens = 128;
        inp.partial_tokens = 2;
        let p = plan_prefix(&inp);
        assert_eq!(p.cutoff_blocks(), 1, "the L1 block is retrieved whole");
        assert_eq!(p.partial_tokens, 0, "an L1 source is never split");

        // A lossy-served L0 block is never the split source either.
        let mut inp = inputs(&[TierId::L0], 130);
        inp.block_tokens = 128;
        inp.lossy_penalty = &[Some(0.5)];
        inp.partial_tokens = 2;
        let p = plan_prefix(&inp);
        assert_eq!((p.cutoff_blocks(), p.partial_tokens), (1, 0));
    }

    /// P7 S-12 (Task 32): a hybrid family's hit ends at the last full block carrying a snapshot
    /// and reuses whole blocks only — a 3.5-block match reuses 3 blocks, restores the block-3
    /// snapshot and recomputes the half block (no tail copy). Breaks if a hybrid plan splits a
    /// block, or restores a snapshot it did not reach.
    #[test]
    fn hybrid_hit_stops_at_last_snapshot() {
        let matched = [TierId::L0; 3];
        let snaps = [Some(snap(1)), Some(snap(2)), Some(snap(3))];
        let mut inp = inputs(&matched, 3 * 32 + 16);
        inp.block_tokens = 32;
        inp.block_bytes = 16 * 32;
        inp.partial_tokens = 16; // the directory matched half of a fourth block
        inp.hybrid = true;
        inp.snapshots = &snaps;
        let p = plan_prefix(&inp);
        assert_eq!(p.cutoff_blocks(), 3, "three whole blocks are reused");
        assert_eq!(p.partial_tokens, 0, "a hybrid never splits a block");
        assert_eq!(p.state_snapshot, Some(snap(3)));
        assert_eq!(p.state_recompute_from, Some(3));
        assert_eq!(
            p.recompute_tokens, 16,
            "the state recompute starts at block 3's boundary, so only the half block remains"
        );
        // No snapshot at or before the cutoff: the state cannot be restored, so it is
        // recomputed from zero even though the KV blocks are cached.
        let mut none = inputs(&matched, 3 * 32 + 16);
        none.block_tokens = 32;
        none.hybrid = true;
        none.snapshots = &[None, None, None];
        let p = plan_prefix(&none);
        assert_eq!(p.state_snapshot, None);
        assert_eq!(p.state_recompute_from, Some(0));
        assert_eq!(p.recompute_tokens, none.prompt_tokens);
    }

    /// P7 S-12 (`nearest_snapshot`): with `kv.hybrid_snapshot_interval_blocks` 2 over a 5-block
    /// chain with snapshots at blocks 2, 4 and 5, a 3-block match restores block 2 and recomputes
    /// block 3 onward; a full 5-block match restores block 5. Breaks if the planner restores a
    /// snapshot after the match, or ignores the nearest one before it.
    #[test]
    fn hybrid_hit_uses_nearest_snapshot() {
        // Interval 2: matched blocks 1 and 3 carry no snapshot, block 2 does.
        let matched = [TierId::L0; 3];
        let snaps = [None, Some(snap(2)), None];
        assert_eq!(nearest_snapshot(&snaps, 2), Some((1, snap(2))));
        assert_eq!(nearest_snapshot(&snaps, 0), None);
        let mut inp = inputs(&matched, 3 * 32 + 4);
        inp.block_tokens = 32;
        inp.hybrid = true;
        inp.snapshots = &snaps;
        let p = plan_prefix(&inp);
        assert_eq!(
            p.cutoff_blocks(),
            3,
            "the KV blocks are reused to the match"
        );
        assert_eq!(p.state_snapshot, Some(snap(2)));
        assert_eq!(p.state_recompute_from, Some(2));
        assert_eq!(
            p.recompute_tokens,
            inp.prompt_tokens - 2 * 32,
            "block 3's tokens are recomputed"
        );

        // A prompt matching all 5 blocks restores the finished sequence's last block (5).
        let all5 = [TierId::L0; 5];
        let chain = [None, Some(snap(2)), None, Some(snap(4)), Some(snap(5))];
        let mut full = inputs(&all5, 5 * 32 + 4);
        full.block_tokens = 32;
        full.hybrid = true;
        full.snapshots = &chain;
        let p = plan_prefix(&full);
        assert_eq!(p.cutoff_blocks(), 5);
        assert_eq!(p.state_snapshot, Some(snap(5)));
        assert_eq!(p.state_recompute_from, Some(5));
        assert_eq!(p.recompute_tokens, 4);
    }
}
