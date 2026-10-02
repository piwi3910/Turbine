# Adding an eviction policy

An eviction policy values one cached copy of a KV block in one tier; the KV hierarchy demotes or drops the lowest-valued copies first when a tier needs room (Phase 4, TS §8). Everything else stays in the mechanism (`KvHierarchy` in `crates/turbine-kv/src/hierarchy.rs` and `KvDirectory` in `crates/turbine-kv/src/directory.rs`): which blocks may leave at all (never a referenced block, leaf-first: never a parent while a child is cached in the same or a faster tier), where a victim goes (L0 → L1 → L2, or dropped below `kv.demote_min_value`), the copy-then-free order and every bound. Point name `eviction_policy`; selected by `kv.policy` (default `cost_aware`).

## The trait

`turbine_kv::policy::EvictionPolicy: Module` (`crates/turbine-kv/src/policy/mod.rs`):

| Method                   | Must do                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `name()` (from `Module`) | The configuration name, `^[a-z0-9_]{1,64}$`, unique in the registry.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `score(b, w, now)`       | The value of keeping the copy described by `BlockScoreInputs` (size, tokens, depth, decayed hits, child count, priority, tier capacity and pressure, prefill rate, retrieval cost, hot session); higher = keep. Size and retrieval cost are the copy's encoded ones (P6b, user decision 2026-09-30): `size_bytes` is the copy's bytes in its tier's format, and the retrieval is the transfer estimate for the bytes it would be stored at in the tier below — for a lossless last block (`kv.lossless_tail_blocks`), the bytes its history would be stored at, so it is ordered like its history (user decision 2026-10-02, 1 A). Finite and ≥ 0 for every input. |
| `action(b, ctx)`         | What to do with one copy the hierarchy considers (P6b compression ladder, `LadderContext`): keep, demote, compress one rung or drop. The default is Phase 4's (demote a leaving copy, drop it from the lowest tier). Never upgrade a copy, never compress at GREEN or with the ladder off, and at YELLOW never compress a copy that need not leave while `fill + demand ≤ low_water` (compress only until GREEN headroom).                                                                                                                                                                                                                                         |

`w` is `PolicyWeights` (`kv.policy_weights.session_active`, `kv.policy_weights.hit_half_life`); a policy may ignore it. Ties are broken by `order_victims` (older last access first, then the deeper block), never by the policy. Policies are called per candidate: cheap, deterministic, stateless — no clock (use `now`), no randomness, no fields that change.

## Files to add

One file: `crates/turbine-kv/src/policy/<name>.rs` (see `crates/turbine-kv/src/policy/lru.rs`):

```rust
//! `smallest_first`: keeps big blocks, evicts small ones (a toy).

use turbine_core::registry::Module;

use super::{BlockScoreInputs, EvictionPolicy, PolicyWeights};
use crate::directory::Timestamp;

pub struct SmallestFirstPolicy;

impl Module for SmallestFirstPolicy {
    fn name(&self) -> &'static str {
        "smallest_first"
    }
}

impl EvictionPolicy for SmallestFirstPolicy {
    fn score(&self, b: &BlockScoreInputs, _w: &PolicyWeights, _now: Timestamp) -> f64 {
        b.block.size_bytes as f64
    }
}
```

Run `cargo fmt --all` after adding the file.

## Registry entry

In `crates/turbine-kv/src/policy/mod.rs`: add `mod <name>;` and `pub use <name>::<Type>;` next to `mod lru;`, and append `&<Type>` to the `REGISTRY` list of `registry()`:

```rust
static REGISTRY: Registry<dyn EvictionPolicy> =
    Registry::new("eviction_policy", &[&CostAwarePolicy, &LruPolicy, &SmallestFirstPolicy]);
```

Then add the name to the pinned list in `registry_conformance::eviction_policies` (`crates/turbine-kv/src/lib.rs`). Nothing else: the server validates `kv.policy` against `registry().names()` (`crates/turbine-server/src/modules.rs`), `turbine-bench kv-sim --policy <name>` takes any registered name, and the KV document reports the chosen policy as `policy`. Keep `cost_aware` first; it is the configuration default.

## Conformance suite

`eviction_policies_suite` (`crates/turbine-kv/src/policy/conformance.rs`) runs every registered policy over 2,000 seeded random inputs with both weight extremes: `finite_non_negative` (every score finite and ≥ 0), `deterministic` (bit-identical scores on a second call and in reverse order) and `orders_every_candidate` (`order_victims` returns each candidate once, lowest first), plus `ladder_contract` over 500 seeded ladder contexts (the `action` rules above). `round_trip_under_every_policy` (in `crates/turbine-kv/src/hierarchy.rs`) also drives a demote/promote round trip through the real hierarchy with every registered policy. With `kv.ladder.enabled` the hierarchy asks `action` once per ladder tick (at most every 50 ms, at most 32 rewrites started per tick and in flight) about the unreferenced copies of each lower tier, lowest tier first and most precise format first, stops a tier's sweep at the first copy the policy keeps, and asks it again for a copy it would drop from the lowest tier; the rung new demotions take, and its step back up (only at GREEN, once the tier has been below `kv.ladder.low_water` for `reliability.pressure.deescalate_dwell`), stay in the hierarchy. `ladder_under_pinned_pressure` (`crates/turbine-scheduler/tests/kv_sim.rs`) pins that sequence for `cost_aware` against a committed pressure trace.

- `scripts/remote-cargo.sh test -p turbine-kv registry_conformance` — the suite over the registry.
- `scripts/remote-cargo.sh test -p turbine-kv hierarchy::tests::round_trip_under_every_policy` — demotion and promotion under every policy.

Compare a new policy offline before anything else: `cargo run -p turbine-bench --bin turbine-bench -- kv-sim --workload mixed --policy <name> --l0-blocks 256 --l1-blocks 1024 --seed 1 --output json` against `--policy cost_aware` (lower `simulated_prefill_seconds` is better).

## Lab checks

A new policy changes nothing until `kv.policy` names it, so landing it needs no lab run. Before recommending it for serving, run the multi-turn profile on novanas with the policy selected (`scripts/lab-serve.sh novanas scripts/lab/phase4-novanas.yaml --set kv.policy=<name>`, then `turbine-bench --profile multi-turn --sessions 16 --turns 8 --shared-prefix-words 2000 --concurrency 8 --session-hints`) and compare `cached_tokens_ratio` and later-turn TTFT p50 with `cost_aware`; the golden gate (`scripts/lab-bench.sh --gpu 0 --model llama -- --set kv.policy=<name>`) must hold, since eviction never changes outputs but a bug in the mechanism would.

## Pitfalls

- A NaN or negative score sorts unpredictably against the others; the suite rejects it, so clamp or floor your terms (as `cost_aware` floors retrieval at 1 µs).
- Do not read a clock or keep a cache in the policy: `kv-sim` and the scheduler's `kv_sim` tests replay with a fake clock and must be deterministic.
- The policy cannot protect a block by scoring it high — anything scored may still be evicted when nothing else is left — and it cannot free a protected one by scoring it low: referenced blocks and parents of cached children never reach it.
- `lru` is the benchmark baseline, not a default (TS §8 "never LRU alone").
- `kv.demote_min_value` is absolute and compares with your score's scale. Under `cost_aware` an L0 block's value falls roughly with the lower tier's compression (its retrieval is priced at the lower tier's encoded size, its memory at the L0 size), so a set threshold drops more blocks exactly when the lower tier is cheaper to fetch from (user decision 2026-09-30: meaning kept, documented). The default 0.0 drops nothing.
