//! Eviction policies (P4 S-7, TS §8; contract §24 `eviction_policy`): pluggable block scoring,
//! higher score = keep.
//!
//! A policy is one file in this directory plus one entry in [`registry`]; `kv.policy` selects
//! it by name (validated against the registry before any port is bound). Policies are pure
//! functions of the block's facts, tier estimates, the configured weights and a timestamp, so
//! they are deterministic under the simulator and benchmarkable against the `lru` baseline.
//! Which blocks may be scored at all (unreferenced, leaf-first) is the directory's call, and
//! the victim order's tie-break is [`order_victims`]'s — never the policy's.

#[cfg(test)]
pub(crate) mod conformance;
mod cost_aware;
mod lru;

pub use cost_aware::CostAwarePolicy;
pub use lru::LruPolicy;

use std::time::Duration;

use turbine_core::config::KvPolicyWeights;
use turbine_core::registry::{Module, Registry, UnknownModule};
use turbine_core::types::PressureState;

use crate::codec;
use crate::directory::{CostEstimate, KvBlock, KvPriority, Timestamp};
use crate::identity::KvKey;
use crate::tier::TierId;

/// The per-block facts a policy scores.
#[derive(Clone, Debug)]
pub struct KvBlockSummary {
    pub key: KvKey,
    pub size_bytes: u64,
    pub tokens: u32,
    pub last_access: Timestamp,
    pub decayed_hits: f64,
    pub decayed_at: Timestamp,
    pub child_count: u32,
    pub priority: KvPriority,
}

impl KvBlockSummary {
    pub fn of(b: &KvBlock) -> Self {
        KvBlockSummary {
            key: b.key,
            size_bytes: b.size_bytes,
            tokens: b.token_range.end - b.token_range.start,
            last_access: b.last_access,
            decayed_hits: b.decayed_hits,
            decayed_at: b.decayed_at,
            child_count: b.child_count,
            priority: b.priority,
        }
    }
}

/// Everything a policy needs to value one copy of a block in one tier.
#[derive(Clone, Debug)]
pub struct BlockScoreInputs {
    pub block: KvBlockSummary,
    pub tier: TierId,
    pub tier_capacity: u64,
    pub tier_pressure: PressureState,
    /// Phase 3 prefill-throughput EWMA (tokens/s).
    pub prefill_tps: f64,
    /// Seconds to bring the block back to L0 from the tier it would be demoted to (its
    /// recompute cost when it would be dropped).
    pub retrieval: CostEstimate,
    pub session_hot: bool,
    /// Token depth of the block's end (`token_range.end`).
    pub depth_tokens: u32,
}

/// `kv.policy_weights`: the tunables every policy receives (a policy may ignore them).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyWeights {
    /// Reuse boost of a block whose session is hot (0..1).
    pub session_active: f64,
    /// Half-life of a block's decayed hit count.
    pub hit_half_life: Duration,
}

impl PolicyWeights {
    pub fn from_config(w: &KvPolicyWeights) -> Self {
        PolicyWeights {
            session_active: w.session_active,
            hit_half_life: w.hit_half_life.0,
        }
    }
}

impl Default for PolicyWeights {
    fn default() -> Self {
        PolicyWeights::from_config(&KvPolicyWeights::default())
    }
}

/// An eviction policy (contract §24 `eviction_policy`). Stateless: the same inputs always give
/// the same score and action, with no clock, randomness or interior state.
pub trait EvictionPolicy: Module {
    /// Value of keeping the copy where it is; higher = keep. Finite and ≥ 0.
    fn score(&self, b: &BlockScoreInputs, w: &PolicyWeights, now: Timestamp) -> f64;

    /// What to do with one copy the hierarchy considers (P6b S-6): a victim that must leave its
    /// tier, or a block of a ladder sweep. The default is Phase 4's: demote a leaving copy to the
    /// next enabled tier at that tier's rung (never more precise than the copy), drop it from
    /// the lowest one, keep everything else. Contract: never upgrade a copy's format, never
    /// compress while the pressure controller is GREEN.
    fn action(&self, _b: &BlockScoreInputs, ctx: &LadderContext) -> EvictAction {
        ctx.phase4_action()
    }
}

/// An eviction decision (P6b S-6).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvictAction {
    /// The copy stays as it is.
    Keep,
    /// Move the copy to tier `to`, stored in codec `format`.
    Demote { to: TierId, format: &'static str },
    /// Rewrite the copy in place in the lossier codec `to` (a ladder rung).
    Compress { to: &'static str },
    /// Remove the copy from this tier (the lowest enabled one) without demoting it.
    Drop,
}

/// The compression ladder's configuration (`kv.ladder.*`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LadderLimits {
    /// The lossiest rung (`kv.ladder.max_format`, a registered lossy codec).
    pub max_format: &'static str,
    /// Tier fill above which the ladder compresses (`kv.ladder.high_water`, default 0.95).
    pub high_water: f64,
    /// GREEN headroom (`kv.ladder.low_water`, default 0.85): at YELLOW the lowest tier
    /// compresses only while its fill plus the YELLOW reclaim's demand is above it (user
    /// decision 2026-09-29, "Compress only until GREEN").
    pub low_water: f64,
}

/// The facts of one tier (and of the copy's place in it) the ladder decides on (P6b S-6). The
/// hierarchy fills it; the policy stays a pure function of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LadderContext {
    /// The tier the copy lives in.
    pub tier: TierId,
    /// That tier's used / capacity (0 ..= 1).
    pub fill: f64,
    /// The bytes the pressure controller's YELLOW reclaim wants to demote into this tier this
    /// tick, as a fraction of its capacity (0 when none).
    pub demand: f64,
    /// That tier's current rung: the codec new demotions into it take.
    pub rung: &'static str,
    /// The pressure controller's state.
    pub pressure: PressureState,
    /// The copy's current codec.
    pub format: &'static str,
    /// True when the hierarchy needs the copy's space (a victim); false in a ladder sweep.
    pub must_leave: bool,
    /// The next enabled tier down and its rung; `None` when this is the lowest enabled tier.
    pub demote_to: Option<(TierId, &'static str)>,
    /// The most precise rung among the enabled tiers below this one (`None` for the lowest):
    /// a tier compresses to a rung only once every tier below has reached it.
    pub lower_rung: Option<&'static str>,
    /// The ladder, when `kv.ladder.enabled`.
    pub ladder: Option<LadderLimits>,
}

impl LadderContext {
    /// Phase 4's decision with per-tier formats: a leaving copy is demoted at the lossier of its
    /// own format and the destination's rung, or dropped from the lowest tier; others stay.
    pub fn phase4_action(&self) -> EvictAction {
        if !self.must_leave {
            return EvictAction::Keep;
        }
        match self.demote_to {
            Some((to, rung)) => EvictAction::Demote {
                to,
                format: codec::lossier(self.format, rung),
            },
            None => EvictAction::Drop,
        }
    }

    /// True when this is the lowest enabled tier.
    pub fn lowest(&self) -> bool {
        self.demote_to.is_none()
    }
}

static REGISTRY: Registry<dyn EvictionPolicy> =
    Registry::new("eviction_policy", &[&CostAwarePolicy, &LruPolicy]);

/// Every registered eviction policy; `cost_aware` (first) is the configuration default.
pub fn registry() -> &'static Registry<dyn EvictionPolicy> {
    &REGISTRY
}

/// A registered policy with the configured weights: what the hierarchy scores with.
#[derive(Clone, Copy)]
pub struct SelectedPolicy {
    pub policy: &'static dyn EvictionPolicy,
    pub weights: PolicyWeights,
}

impl SelectedPolicy {
    pub fn name(&self) -> &'static str {
        self.policy.name()
    }

    pub fn score(&self, b: &BlockScoreInputs, now: Timestamp) -> f64 {
        self.policy.score(b, &self.weights, now)
    }
}

impl std::fmt::Debug for SelectedPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectedPolicy")
            .field("policy", &self.name())
            .field("weights", &self.weights)
            .finish()
    }
}

/// The policy `kv.policy` names, weighted by `kv.policy_weights`. No log: the server logs the
/// selection once (`registry().select`); simulators build many hierarchies.
pub fn make_policy(name: &str, w: &KvPolicyWeights) -> Result<SelectedPolicy, UnknownModule> {
    let policy = registry()
        .get(name)
        .ok_or_else(|| registry().unknown(name))?;
    Ok(SelectedPolicy {
        policy,
        weights: PolicyWeights::from_config(w),
    })
}

/// Seconds to prefill a block's tokens at its depth (attention cost grows with depth):
/// `tokens / prefill_tps × (1 + depth / 8192)`.
pub fn recompute_seconds(tokens: u32, depth_tokens: u32, prefill_tps: f64) -> f64 {
    f64::from(tokens) / prefill_tps.max(1.0) * (1.0 + f64::from(depth_tokens) / 8192.0)
}

/// Candidates scored and ordered lowest value first (the first victim first). Ties: older last
/// access first, then the deeper block first (leaves before their ancestors).
pub fn order_victims(
    policy: &SelectedPolicy,
    candidates: &[BlockScoreInputs],
    now: Timestamp,
) -> Vec<(KvKey, f64)> {
    let mut scored: Vec<(&BlockScoreInputs, f64)> = candidates
        .iter()
        .map(|c| (c, policy.score(c, now)))
        .collect();
    scored.sort_by(|(a, va), (b, vb)| {
        va.total_cmp(vb)
            .then_with(|| a.block.last_access.cmp(&b.block.last_access))
            .then_with(|| b.depth_tokens.cmp(&a.depth_tokens))
    });
    scored.into_iter().map(|(c, v)| (c.block.key, v)).collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::directory::KvDirectory;
    use crate::directory::tests::{block, l0};
    use crate::tier::KvLocation;

    /// 1 MiB, 16 tokens, 2 hits, 1 child, in a YELLOW 1 GiB L0, 10,000 tok/s, 1 ms retrieval.
    pub(crate) fn base() -> BlockScoreInputs {
        BlockScoreInputs {
            block: KvBlockSummary {
                key: KvKey([1; 16]),
                size_bytes: 1 << 20,
                tokens: 16,
                last_access: Duration::from_secs(10),
                decayed_hits: 2.0,
                decayed_at: Duration::from_secs(10),
                child_count: 1,
                priority: KvPriority(1.0),
            },
            tier: TierId::L0,
            tier_capacity: 1 << 30,
            tier_pressure: PressureState::Yellow,
            prefill_tps: 10_000.0,
            retrieval: CostEstimate { seconds: 0.001 },
            session_hot: false,
            depth_tokens: 1024,
        }
    }

    /// A one-term change to the base inputs.
    type Tweak = fn(&mut BlockScoreInputs);

    /// Scores every block the same, so only the tie-break orders them.
    struct Flat;

    impl Module for Flat {
        fn name(&self) -> &'static str {
            "flat"
        }
    }

    impl EvictionPolicy for Flat {
        fn score(&self, _b: &BlockScoreInputs, _w: &PolicyWeights, _now: Timestamp) -> f64 {
            1.0
        }
    }

    fn weights() -> PolicyWeights {
        PolicyWeights {
            session_active: 0.5,
            hit_half_life: Duration::from_secs(60),
        }
    }

    fn selected(policy: &'static dyn EvictionPolicy) -> SelectedPolicy {
        SelectedPolicy {
            policy,
            weights: weights(),
        }
    }

    #[test]
    fn cost_aware_ordering() {
        let p = selected(&CostAwarePolicy);
        assert_eq!(p.name(), "cost_aware");
        let now = Duration::from_secs(20);
        let v0 = p.score(&base(), now);
        let raise: [(&str, Tweak); 5] = [
            ("more hits", |b| b.block.decayed_hits = 20.0),
            ("a hot session", |b| b.session_hot = true),
            ("a deeper block (recompute cost)", |b| b.depth_tokens = 8192),
            ("slower prefill (recompute cost)", |b| {
                b.prefill_tps = 1_000.0
            }),
            ("high priority", |b| b.block.priority = KvPriority(2.0)),
        ];
        for (what, f) in raise {
            let mut b = base();
            f(&mut b);
            assert!(p.score(&b, now) > v0, "{what} must raise the value");
        }
        let lower: [(&str, Tweak); 4] = [
            ("a larger size", |b| b.block.size_bytes = 4 << 20),
            ("higher tier pressure", |b| {
                b.tier_pressure = PressureState::Red
            }),
            ("cheaper retrieval", |b| {
                b.retrieval = CostEstimate { seconds: 0.0001 }
            }),
            ("low priority", |b| b.block.priority = KvPriority(0.5)),
        ];
        for (what, f) in lower {
            let mut b = base();
            f(&mut b);
            assert!(p.score(&b, now) < v0, "{what} must lower the value");
        }
        assert!(
            p.score(&base(), Duration::from_secs(600)) < v0,
            "hits decay with the half-life"
        );
        let mut hot = base();
        hot.session_hot = true;
        hot.block.decayed_hits = 1e9;
        assert_eq!(
            CostAwarePolicy.reuse_probability(&hot, &weights(), now),
            1.0,
            "capped at 1"
        );
        let mut zero = base();
        zero.retrieval = CostEstimate { seconds: 0.0 };
        assert!(p.score(&zero, now) > 0.0, "retrieval is floored at 1 µs");

        // Protected blocks are never candidates: referenced ones, and parents of a child cached
        // in the same tier.
        let (k1, k2, k3) = (KvKey([1; 16]), KvKey([2; 16]), KvKey([3; 16]));
        let mut dir = KvDirectory::new(8);
        dir.insert(block(k1, None, 0, &[0; 16], &[l0(0)])).unwrap();
        dir.insert(block(k2, Some(k1), 1, &[1; 16], &[l0(1)]))
            .unwrap();
        dir.insert(block(k3, None, 0, &[3; 16], &[l0(2)])).unwrap();
        dir.get_mut(&k3).unwrap().ref_count = 1;
        let none = HashSet::new();
        let keys = |dir: &KvDirectory| -> Vec<KvKey> {
            dir.candidates(TierId::L0, &none)
                .iter()
                .map(|b| b.key)
                .collect()
        };
        assert_eq!(keys(&dir), [k2], "only the unreferenced leaf is evictable");
        // Once the child lives only in L1, the parent may leave L0.
        let l1 = KvLocation {
            tier: TierId::L1,
            slot: 0,
            format: crate::tier::L0_FORMAT,
        };
        dir.add_location(&k2, l1);
        dir.remove_location(&k2, TierId::L0);
        assert_eq!(keys(&dir), [k1]);

        // Scoring a directory block and ordering victims: lowest value first.
        let summary = KvBlockSummary::of(dir.get(&k1).unwrap());
        assert_eq!((summary.tokens, summary.child_count), (16, 1));
        let cheap = BlockScoreInputs {
            retrieval: CostEstimate { seconds: 1e-5 },
            ..base()
        };
        let mut dear = base();
        dear.block.key = KvKey([9; 16]);
        let order = order_victims(&p, &[dear.clone(), cheap.clone()], now);
        assert_eq!(
            order[0].0, cheap.block.key,
            "cheap retrieval is evicted first"
        );
        assert!(order[0].1 < order[1].1);

        // Ties: older last access first, then the deeper block (a leaf) first.
        let mut older = base();
        older.block.key = KvKey([10; 16]);
        older.block.last_access = Duration::from_secs(5);
        let mut deeper = base();
        deeper.block.key = KvKey([11; 16]);
        deeper.depth_tokens = 4096;
        let order: Vec<KvKey> = order_victims(&selected(&Flat), &[base(), deeper, older], now)
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        assert_eq!(order, [KvKey([10; 16]), KvKey([11; 16]), KvKey([1; 16])]);

        // LRU: the benchmark baseline scores recency only.
        let lru = selected(&LruPolicy);
        assert_eq!(lru.name(), "lru");
        let order = order_victims(&lru, &[base(), older_by(3)], now);
        assert_eq!(order[0].0, KvKey([12; 16]));
        assert_eq!(recompute_seconds(16, 8192, 8.0), 4.0);
    }

    /// `kv.policy` and `kv.policy_weights` select and parameterise the policy; an unregistered
    /// name is refused naming the registered ones.
    #[test]
    fn make_policy_follows_config() {
        use turbine_core::config::{HumanDuration, KvPolicyWeights};

        let now = Duration::from_secs(20);
        let w = KvPolicyWeights {
            session_active: 0.25,
            hit_half_life: HumanDuration::from_secs(30),
        };
        let lru = make_policy("lru", &w).unwrap();
        assert_eq!(lru.name(), "lru");
        assert_eq!(
            lru.score(&base(), now),
            LruPolicy.score(&base(), &weights(), now)
        );

        let cost = make_policy("cost_aware", &w).unwrap();
        assert_eq!(cost.name(), "cost_aware");
        let expected = PolicyWeights {
            session_active: 0.25,
            hit_half_life: Duration::from_secs(30),
        };
        assert_eq!(cost.weights, expected);
        let mut hot = base();
        hot.session_hot = true;
        for b in [base(), hot] {
            assert_eq!(
                cost.score(&b, now),
                CostAwarePolicy.score(&b, &expected, now)
            );
        }
        let err = make_policy("lfu", &w).unwrap_err();
        assert_eq!(err.point, "eviction_policy");
        assert_eq!(err.registered, "cost_aware, lru");
    }

    /// A YELLOW ladder context of the lowest enabled tier (L2) at 50 % fill, the copy in `l0`
    /// and leaving, the ladder up to `tq2` with high water 0.95.
    pub(crate) fn ladder_ctx() -> LadderContext {
        LadderContext {
            tier: TierId::L2,
            fill: 0.5,
            demand: 0.0,
            rung: "l0",
            pressure: PressureState::Yellow,
            format: "l0",
            must_leave: true,
            demote_to: None,
            lower_rung: None,
            ladder: Some(LadderLimits {
                max_format: "tq2",
                high_water: 0.95,
                low_water: 0.85,
            }),
        }
    }

    /// The ladder rule of `cost_aware` with pinned fills and pressure states (P6b S-6, user
    /// decisions 2026-09-29 "Start at YELLOW earlier" and "Compress only until GREEN"): from
    /// ORANGE on the lowest enabled tier compresses even with free room, at YELLOW only while
    /// `fill + demand > low_water`; at any non-GREEN state an upper tier compresses only
    /// above high water; one rung at a time (`l0` → `fp8_e4m3` → `tq4` → `tq2` → evict, bounded
    /// by `max_format`), an upper tier only once every tier below has reached the target rung;
    /// at the floor the lowest tier evicts only a leaving copy or above high water; it never
    /// upgrades a copy, and with the ladder off (or under `lru`) decides as Phase 4 did.
    /// Breaks if a rung is skipped, a copy upgraded or compressed at GREEN, if the lowest tier
    /// waits for high water at YELLOW (the pre-amendment rule), if it keeps compressing at
    /// YELLOW once its GREEN headroom is back (drifting to `tq2`), or if the floor evicts a copy
    /// from a tier with free room.
    #[test]
    fn ladder_actions() {
        let p = CostAwarePolicy;
        let b = base();
        let act = |c: LadderContext| p.action(&b, &c);
        let with = |f: fn(&mut LadderContext)| {
            let mut c = ladder_ctx();
            f(&mut c);
            c
        };

        // The lowest tier would drop: one rung down per decision, then the floor evicts.
        assert_eq!(act(ladder_ctx()), EvictAction::Compress { to: "fp8_e4m3" });
        assert_eq!(
            act(with(|c| c.format = "fp8_e4m3")),
            EvictAction::Compress { to: "tq4" }
        );
        assert_eq!(
            act(with(|c| c.format = "tq4")),
            EvictAction::Compress { to: "tq2" }
        );
        assert_eq!(act(with(|c| c.format = "tq2")), EvictAction::Drop);
        // `max_format` bounds the ladder.
        let capped = |c: &mut LadderContext| {
            c.format = "tq4";
            c.ladder = Some(LadderLimits {
                max_format: "tq4",
                high_water: 0.95,
                low_water: 0.85,
            });
        };
        assert_eq!(act(with(capped)), EvictAction::Drop);

        // Never at GREEN, never with the ladder off: Phase 4 drops the victim.
        assert_eq!(
            act(with(|c| c.pressure = PressureState::Green)),
            EvictAction::Drop
        );
        assert_eq!(act(with(|c| c.ladder = None)), EvictAction::Drop);
        for state in [
            PressureState::Yellow,
            PressureState::Orange,
            PressureState::Red,
            PressureState::Survival,
        ] {
            let mut c = ladder_ctx();
            c.pressure = state;
            assert!(matches!(act(c), EvictAction::Compress { .. }), "{state:?}");
        }

        // A sweep of the lowest tier (nothing must leave). From ORANGE on it compresses one
        // rung even with free room (before the tiers are full); at YELLOW only while its GREEN
        // headroom is short, `fill + demand > low_water` (user decision 2026-09-29, "Compress
        // only until GREEN"); at the floor it evicts only above high water, never a copy of a
        // tier with free room; GREEN never compresses.
        let sweep = |fill: f64, format: &'static str| {
            let mut c = ladder_ctx();
            c.must_leave = false;
            c.fill = fill;
            c.format = format;
            c
        };
        for state in [
            PressureState::Orange,
            PressureState::Red,
            PressureState::Survival,
        ] {
            for fill in [0.0, 0.10, 0.50, 0.90, 0.95, 0.96] {
                let mut c = sweep(fill, "l0");
                c.pressure = state;
                assert_eq!(
                    act(c),
                    EvictAction::Compress { to: "fp8_e4m3" },
                    "{state:?} at fill {fill}: the lowest tier compresses one rung"
                );
            }
        }
        // YELLOW with GREEN headroom keeps, whatever the copy's rung.
        for (fill, demand) in [
            (0.0, 0.0),
            (0.10, 0.0),
            (0.50, 0.30),
            (0.85, 0.0),
            (0.80, 0.04),
        ] {
            for format in ["l0", "fp8_e4m3", "tq4"] {
                let mut c = sweep(fill, format);
                c.demand = demand;
                assert_eq!(
                    act(c),
                    EvictAction::Keep,
                    "YELLOW at fill {fill} + demand {demand} ≤ low water keeps a {format} copy"
                );
            }
        }
        // YELLOW whose fill or demand pushes it over low water compresses one rung.
        for (fill, demand) in [
            (0.80, 0.10),
            (0.50, 0.40),
            (0.86, 0.0),
            (0.90, 0.0),
            (0.95, 0.0),
        ] {
            let mut c = sweep(fill, "l0");
            c.demand = demand;
            assert_eq!(
                act(c),
                EvictAction::Compress { to: "fp8_e4m3" },
                "YELLOW at fill {fill} + demand {demand} > low water"
            );
        }
        let mut c = sweep(0.80, "tq4");
        c.demand = 0.10;
        assert_eq!(act(c), EvictAction::Compress { to: "tq2" });
        // GREEN after YELLOW keeps, even with the demand still set.
        let mut c = sweep(0.80, "fp8_e4m3");
        c.demand = 0.10;
        c.pressure = PressureState::Green;
        assert_eq!(act(c), EvictAction::Keep, "GREEN after YELLOW keeps");
        // ORANGE with room still compresses.
        let mut c = sweep(0.10, "tq4");
        c.pressure = PressureState::Orange;
        assert_eq!(act(c), EvictAction::Compress { to: "tq2" });
        assert_eq!(
            act(sweep(0.10, "tq2")),
            EvictAction::Keep,
            "the floor with free room keeps the copy"
        );
        assert_eq!(
            act(sweep(0.95, "tq2")),
            EvictAction::Keep,
            "at high water, not above"
        );
        assert_eq!(act(sweep(0.96, "tq2")), EvictAction::Drop);
        for fill in [0.10, 0.99] {
            let mut green = sweep(fill, "l0");
            green.pressure = PressureState::Green;
            assert_eq!(act(green), EvictAction::Keep, "GREEN at fill {fill}");
        }

        // A repeated sweep at steady pressure: 16 copies of the lowest tier, starting in `l0`
        // at fill 0.92 (above low water, below high water) with a YELLOW reclaim demand of 0.02
        // each sweep; the returned actions are applied and `fill` re-derived from the copies'
        // bytes per codec after every decision. At YELLOW the tier stops once its GREEN
        // headroom is back and never reaches `tq2`; at ORANGE the same sweep does.
        let sizes = |f: &str| {
            let l0 = crate::codec::tests::layout(turbine_core::types::DType::BF16);
            crate::codec::registry()
                .get(f)
                .expect("registered codec")
                .bytes_per_block(&l0)
        };
        let steady = |pressure: PressureState| -> Vec<&'static str> {
            let mut copies: Vec<&'static str> = vec!["l0"; 16];
            let capacity = (16 * sizes("l0")) as f64 / 0.92;
            for _ in 0..50 {
                for i in 0..copies.len() {
                    let used: u64 = copies.iter().map(|f| sizes(f)).sum();
                    let mut c = sweep(used as f64 / capacity, copies[i]);
                    c.pressure = pressure;
                    c.demand = 0.02;
                    match act(c) {
                        EvictAction::Compress { to } => copies[i] = to,
                        EvictAction::Keep => {}
                        other => panic!("{pressure:?}: unexpected {other:?} in a sweep"),
                    }
                }
            }
            copies
        };
        let yellow = steady(PressureState::Yellow);
        assert!(
            !yellow.contains(&"tq2"),
            "a steady YELLOW with room never drifts to tq2: {yellow:?}"
        );
        assert!(
            yellow.contains(&"fp8_e4m3"),
            "YELLOW above low water compresses: {yellow:?}"
        );
        let used: u64 = yellow.iter().map(|f| sizes(f)).sum();
        let fill = used as f64 / ((16 * sizes("l0")) as f64 / 0.92);
        assert!(
            fill + 0.02 <= 0.85,
            "YELLOW restored GREEN headroom: {fill}"
        );
        let orange = steady(PressureState::Orange);
        assert!(
            orange.iter().all(|f| *f == "tq2"),
            "ORANGE walks the tier to tq2: {orange:?}"
        );

        // An upper tier (L1 above L2) compresses only once L2 has reached the target rung;
        // otherwise a victim is demoted as in Phase 4, at the lossier of the two formats.
        let upper = |lower: &'static str, format: &'static str, must_leave: bool| {
            let mut c = ladder_ctx();
            c.tier = TierId::L1;
            c.fill = 0.97;
            c.format = format;
            c.must_leave = must_leave;
            c.demote_to = Some((TierId::L2, lower));
            c.lower_rung = Some(lower);
            c
        };
        assert_eq!(act(upper("l0", "l0", false)), EvictAction::Keep);
        assert_eq!(
            act(upper("l0", "l0", true)),
            EvictAction::Demote {
                to: TierId::L2,
                format: "l0"
            }
        );
        assert_eq!(
            act(upper("fp8_e4m3", "l0", false)),
            EvictAction::Compress { to: "fp8_e4m3" }
        );
        // At YELLOW only the lowest tier acts with free room: an upper tier at or below high
        // water keeps its copy (or demotes a leaving one) even when L2 has reached the rung.
        for fill in [0.10, 0.95] {
            let mut c = upper("fp8_e4m3", "l0", false);
            c.fill = fill;
            assert_eq!(act(c), EvictAction::Keep, "L1 at fill {fill}");
            c.must_leave = true;
            assert_eq!(
                act(c),
                EvictAction::Demote {
                    to: TierId::L2,
                    format: "fp8_e4m3"
                },
                "leaving L1 copy at fill {fill}"
            );
        }
        assert_eq!(
            act(upper("fp8_e4m3", "fp8_e4m3", false)),
            EvictAction::Keep,
            "L1 never runs ahead of L2"
        );
        assert_eq!(
            act(upper("tq2", "tq2", false)),
            EvictAction::Keep,
            "an upper tier at the floor keeps its copy (the lowest tier evicts)"
        );
        // Never upgrade: a tq4 copy demoted into an `l0` tier stays tq4.
        let mut c = upper("l0", "tq4", true);
        c.pressure = PressureState::Green;
        assert_eq!(
            act(c),
            EvictAction::Demote {
                to: TierId::L2,
                format: "tq4"
            }
        );

        // `lru` keeps Phase 4 behaviour whatever the ladder says.
        let lru = LruPolicy;
        assert_eq!(lru.action(&b, &ladder_ctx()), EvictAction::Drop);
        assert_eq!(lru.action(&b, &sweep(0.99, "l0")), EvictAction::Keep);
        assert_eq!(
            lru.action(&b, &upper("l0", "fp8_e4m3", true)),
            EvictAction::Demote {
                to: TierId::L2,
                format: "fp8_e4m3"
            }
        );
    }

    fn older_by(secs: u64) -> BlockScoreInputs {
        let mut b = base();
        b.block.key = KvKey([12; 16]);
        b.block.last_access = Duration::from_secs(10 - secs);
        b
    }
}
