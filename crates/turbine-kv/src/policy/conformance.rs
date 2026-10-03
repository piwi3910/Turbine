//! The `eviction_policy` conformance suite (contract §24): the properties every policy must
//! keep, run over a registry — never a hand-written list — by
//! `registry_conformance::eviction_policies`, so a policy registered without passing it fails
//! `cargo test --workspace`. How a policy values blocks (which one it evicts first) is its own
//! business and has its own unit test; `policy::tests` pins `cost_aware` and `lru`.

use std::time::Duration;

use turbine_core::registry::{Registry, conformance};
use turbine_core::types::PressureState;

use super::{
    BlockScoreInputs, EvictAction, EvictionPolicy, KvBlockSummary, LadderContext, LadderLimits,
    PolicyWeights, SelectedPolicy, order_victims,
};
use crate::directory::{CostEstimate, KvPriority};
use crate::identity::KvKey;
use crate::tier::TierId;

/// Runs every property over every policy of `reg`; `Err` lists each broken one as
/// `<policy>: <property>: <detail>`.
///
/// - `finite_non_negative`: over 2,000 seeded random inputs (sizes, depths, hit counts,
///   pressures, retrieval costs, capacities, priorities, both weight extremes) every score is
///   finite and ≥ 0.
/// - `deterministic`: scoring the same inputs twice, and in reverse order, gives bit-identical
///   scores (no clock, randomness or interior state).
/// - `orders_every_candidate`: `order_victims` returns each candidate exactly once, lowest score
///   first.
/// - `ladder_contract` (P6b S-6): `action` never upgrades a copy, never compresses at GREEN or
///   with the ladder off, nor at YELLOW with GREEN headroom (`fill + demand ≤ low_water`, no
///   leaving copy), never past `kv.ladder.max_format`, demotes only to the next tier and
///   drops only from the lowest one, and only a leaving copy or above high water.
pub(crate) fn eviction_policies_suite(
    reg: &Registry<dyn EvictionPolicy>,
) -> Result<(), Vec<String>> {
    let mut failures = Vec::new();
    if let Err(e) = conformance::check(reg) {
        failures.push(format!("registry: {e}"));
    }
    let inputs = random_inputs(2_000, 0x5eed);
    let now = Duration::from_secs(3_600);
    for policy in reg.iter() {
        let name = policy.name();
        let mut record = |property: &str, result: Result<(), String>| {
            if let Err(detail) = result {
                failures.push(format!("{name}: {property}: {detail}"));
            }
        };
        for weights in [
            PolicyWeights {
                session_active: 0.0,
                hit_half_life: Duration::from_millis(1),
            },
            PolicyWeights {
                session_active: 1.0,
                hit_half_life: Duration::from_secs(86_400),
            },
        ] {
            let selected = SelectedPolicy { policy, weights };
            record("finite_non_negative", finite(&selected, &inputs, now));
            record("deterministic", deterministic(&selected, &inputs, now));
            record(
                "orders_every_candidate",
                orders_all(&selected, &inputs[..200], now),
            );
        }
        record("ladder_contract", ladder_contract(policy, &inputs[..500]));
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

fn finite(p: &SelectedPolicy, inputs: &[BlockScoreInputs], now: Duration) -> Result<(), String> {
    for (i, b) in inputs.iter().enumerate() {
        let v = p.score(b, now);
        if !v.is_finite() || v < 0.0 {
            return Err(format!("input {i} scored {v}: {b:?}"));
        }
    }
    Ok(())
}

fn deterministic(
    p: &SelectedPolicy,
    inputs: &[BlockScoreInputs],
    now: Duration,
) -> Result<(), String> {
    let forward: Vec<u64> = inputs.iter().map(|b| p.score(b, now).to_bits()).collect();
    let mut backward: Vec<u64> = inputs
        .iter()
        .rev()
        .map(|b| p.score(b, now).to_bits())
        .collect();
    backward.reverse();
    match forward.iter().zip(&backward).position(|(a, b)| a != b) {
        None => Ok(()),
        Some(i) => Err(format!("input {i} scored differently on a second call")),
    }
}

fn orders_all(
    p: &SelectedPolicy,
    inputs: &[BlockScoreInputs],
    now: Duration,
) -> Result<(), String> {
    let order = order_victims(p, inputs, now);
    if order.len() != inputs.len() {
        return Err(format!(
            "{} candidates, {} ordered",
            inputs.len(),
            order.len()
        ));
    }
    let mut keys: Vec<KvKey> = order.iter().map(|(k, _)| *k).collect();
    keys.sort_by_key(|k| k.0);
    keys.dedup();
    if keys.len() != inputs.len() {
        return Err("a candidate is missing or repeated".into());
    }
    if order.windows(2).any(|w| w[0].1 > w[1].1) {
        return Err("victims are not ordered lowest score first".into());
    }
    Ok(())
}

/// Every rung of the ladder, most precise first (the `kv_format` registry order).
const RUNGS: [&str; 4] = ["l0", "fp8_e4m3", "tq4", "tq2"];

/// `action` over seeded ladder contexts (every tier position, fill, pressure, format, rung,
/// ladder setting): a compression only to a strictly lossier rung within `max_format`, never
/// at GREEN or with the ladder off, nor at YELLOW for a copy that need not leave a tier with
/// GREEN headroom (`fill + demand ≤ low_water`, user decision 2026-09-29 "Compress only until
/// GREEN"); a demotion only to the tier below, never more precise than
/// the copy or that tier's rung; a drop only from the lowest tier, and only of a copy that must
/// leave or from a tier above high water (never evicting with free room). Deterministic.
fn ladder_contract(p: &dyn EvictionPolicy, inputs: &[BlockScoreInputs]) -> Result<(), String> {
    let mut s = 0x001a_dde7_u64;
    let mut next = move || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        s.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };
    let rank = |f: &str| RUNGS.iter().position(|r| *r == f).unwrap_or(usize::MAX);
    for (i, b) in inputs.iter().enumerate() {
        let pick = |n: u64| (n % 4) as usize;
        let lowest = next() % 3 == 0;
        let lower = RUNGS[pick(next())];
        let ctx = LadderContext {
            tier: if lowest { TierId::L2 } else { TierId::L1 },
            fill: (next() % 1_001) as f64 / 1_000.0,
            demand: (next() % 201) as f64 / 1_000.0,
            rung: RUNGS[pick(next())],
            pressure: PressureState::ALL[(next() % 5) as usize],
            format: RUNGS[pick(next())],
            must_leave: next() % 2 == 0,
            demote_to: (!lowest).then_some((TierId::L2, lower)),
            lower_rung: (!lowest).then_some(lower),
            ladder: (next() % 4 != 0).then(|| LadderLimits {
                max_format: RUNGS[1 + (next() % 3) as usize],
                high_water: 0.95,
                low_water: 0.85,
                l0_dtype: "bf16",
            }),
        };
        let a = p.action(b, &ctx);
        if a != p.action(b, &ctx) {
            return Err(format!("context {i}: two decisions differ"));
        }
        let bad = match a {
            EvictAction::Compress { to } => {
                if ctx.pressure == PressureState::Green {
                    Some("compressed at GREEN")
                } else if let Some(l) = ctx.ladder {
                    if ctx.pressure == PressureState::Yellow
                        && !ctx.must_leave
                        && ctx.fill + ctx.demand <= l.low_water
                    {
                        Some("compressed at YELLOW with GREEN headroom (fill + demand ≤ low_water)")
                    } else if rank(to) <= rank(ctx.format) {
                        Some("compressed to a rung no lossier than the copy (upgrade)")
                    } else if rank(to) > rank(l.max_format) {
                        Some("compressed past max_format")
                    } else {
                        None
                    }
                } else {
                    Some("compressed with the ladder off")
                }
            }
            EvictAction::Demote { to, format } => match ctx.demote_to {
                Some((t, rung)) if t == to => (rank(format) < rank(ctx.format)
                    || rank(format) < rank(rung))
                .then_some("demoted more precisely than the copy or the tier's rung"),
                _ => Some("demoted to a tier that is not the next one down"),
            },
            EvictAction::Drop if !ctx.lowest() => Some("dropped from an upper tier"),
            EvictAction::Drop => (!ctx.must_leave
                && ctx.ladder.is_none_or(|l| ctx.fill <= l.high_water))
            .then_some("dropped a copy that need not leave from a tier at or below high water"),
            _ => None,
        };
        if let Some(what) = bad {
            return Err(format!("context {i}: {what}: {a:?} for {ctx:?}"));
        }
    }
    Ok(())
}

/// Seeded xorshift64* inputs spanning the ranges the hierarchy produces.
fn random_inputs(n: usize, seed: u64) -> Vec<BlockScoreInputs> {
    let mut s = seed;
    let mut next = move || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        s.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };
    let pressures = [
        PressureState::Green,
        PressureState::Yellow,
        PressureState::Orange,
        PressureState::Red,
        PressureState::Survival,
    ];
    let tiers = [TierId::L0, TierId::L1, TierId::L2];
    (0..n)
        .map(|i| {
            let mut key = [0u8; 16];
            key[..8].copy_from_slice(&(i as u64).to_le_bytes());
            let tokens = 1 + (next() % 1024) as u32;
            let depth = tokens + (next() % 131_072) as u32;
            let at = Duration::from_millis(next() % 3_600_000);
            BlockScoreInputs {
                block: KvBlockSummary {
                    key: KvKey(key),
                    size_bytes: next() % (64 << 20),
                    tokens,
                    last_access: at,
                    decayed_hits: (next() % 10_000) as f64 / 10.0,
                    decayed_at: at,
                    child_count: (next() % 64) as u32,
                    priority: KvPriority([0.5, 1.0, 2.0][(next() % 3) as usize]),
                },
                tier: tiers[(next() % 3) as usize],
                tier_capacity: next() % (1 << 40),
                tier_pressure: pressures[(next() % 5) as usize],
                prefill_tps: (next() % 100_000) as f64,
                retrieval: CostEstimate {
                    seconds: (next() % 1_000_000) as f64 / 1e6,
                },
                session_hot: next() % 2 == 0,
                depth_tokens: depth,
            }
        })
        .collect()
}
