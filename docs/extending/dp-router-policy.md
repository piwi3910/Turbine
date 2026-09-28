# Adding a DP router policy

A DP router policy picks the data-parallel replica one request goes to (Phase 5, S-7). Everything around the choice stays in the router (`crates/turbine-distributed/src/router/mod.rs`) and the server's replica table (`crates/turbine-server/src/replicas.rs`): which replicas are ready (only those get a `ReplicaView`), their pressure state and circuit, their outstanding tokens (what each engine published plus what is in transit to it), whether a replica holds the prompt's first KV block (the bounded affinity table), the single-replica shortcut (`only_candidate`), and the `turbine_dp_routed_total{replica,reason}` counter plus the `dp_route` log. Point name `dp_router_policy`; selected by `parallel.router` (default `prefix_affinity`; also `least_loaded`).

## The trait

`turbine_distributed::router::RouterPolicy: Module` (`crates/turbine-distributed/src/router/mod.rs`):

| Item                     | Must do                                                                                                                                                                                                                          |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name()` (from `Module`) | The configuration name, `^[a-z0-9_]{1,64}$`, unique in the registry.                                                                                                                                                             |
| `choose(views)`          | For at least two `ReplicaView`s, the chosen `ReplicaId` and its `DpRouteReason`. Picks from `candidates(views)` (the eligible replicas, or all of them when none is eligible), deterministically, ties to the lowest replica id. |

Helpers in the same module: `ReplicaView::eligible()` (below ORANGE, circuit neither `CIRCUIT_OPEN` nor `DRAINING`), `candidates(views)` and `least_loaded(iter)` (fewest outstanding tokens, ties to the lowest id). `route(views, policy)` is what the server calls: it panics on an empty slice, answers one replica with `only_candidate`, and otherwise asks the policy. `DpRouteReason` is closed (its `as_str` values are the metric's `reason` label): a policy that needs a new reason adds a variant there and to contract §15.2.

## Files to add

One file: `crates/turbine-distributed/src/router/<name>.rs` (see `crates/turbine-distributed/src/router/least_loaded.rs`):

```rust
//! `fewest_prefix_misses`: least-loaded, but a prefix holder wins ties (a toy).

use turbine_core::registry::Module;
use turbine_core::types::ReplicaId;

use super::{DpRouteReason, ReplicaView, RouterPolicy, candidates};

pub struct FewestPrefixMisses;

impl Module for FewestPrefixMisses {
    fn name(&self) -> &'static str {
        "fewest_prefix_misses"
    }
}

impl RouterPolicy for FewestPrefixMisses {
    fn choose(&self, views: &[ReplicaView]) -> (ReplicaId, DpRouteReason) {
        let v = candidates(views)
            .min_by_key(|v| (v.outstanding_tokens, !v.has_prefix, v.replica))
            .expect("route passes at least two views");
        (v.replica, DpRouteReason::LeastLoaded)
    }
}
```

Run `cargo fmt --all` after adding the file.

## Registry entry

In `crates/turbine-distributed/src/router/mod.rs`: add `pub mod <name>;` and `pub use <name>::<Type>;` next to `pub mod least_loaded;`, a `static` of the new type, and append it to `DP_ROUTER_POLICIES` (returned by `registry()`):

```rust
static DP_ROUTER_POLICIES: Registry<dyn RouterPolicy> = Registry::new(
    "dp_router_policy",
    &[&PREFIX_AFFINITY, &LEAST_LOADED, &FEWEST_PREFIX_MISSES],
);
```

Then add the name to the pinned list in `registry_conformance::dp_router_policies` (`crates/turbine-distributed/src/lib.rs`). Nothing else: the server validates `parallel.router` against `registry().names()` (`crates/turbine-server/src/modules.rs`) before any port is bound, and `crates/turbine-server/src/startup.rs` selects the policy with `turbine_distributed::router::select` (which logs `event="module_selected"`) and hands it to `ReplicaRouter`. Keep `prefix_affinity` first; it is the configuration default.

## Conformance suite

`check` (`crates/turbine-distributed/src/router/conformance.rs`) runs every registered policy through `route`: one replica (even RED with an open circuit) gives that replica with `only_candidate`; among several, the pick is one of the slice and never `only_candidate`; an ineligible replica is never picked while an eligible one exists, however idle or prefix-holding it is (ORANGE, RED, SURVIVAL, `CIRCUIT_OPEN`, `DRAINING`, with sparse unordered replica ids); some replica is picked when none is eligible; the pick is deterministic and independent of the slice order; identical replicas 5, 2, 7 give replica 2, with and without the prefix. `conformance_rejects_broken_policy` proves the suite refuses a policy that ignores eligibility. `router::tests::routing_policy` pins the decisions and reasons of the two shipped policies, and the server's `replicas::tests::routes_by_load_affinity_and_readiness` covers tokens in transit, the affinity table and replicas that are not ready.

- `scripts/remote-cargo.sh test -p turbine-distributed registry_conformance` — the suite over the registry.
- `scripts/remote-cargo.sh test -p turbine-distributed router::conformance::tests` — the suite catches a broken policy.
- `scripts/remote-cargo.sh test -p turbine-distributed router::tests::routing_policy` — the shipped policies' decisions.

## Lab checks

A policy is proven on two replicas: `scripts/lab-cluster.sh dp2-novanas` with `parallel.router: <name>` in `scripts/lab/phase5-novanas-dp2.yaml` must meet the golden gate on the replicas (`turbine-golden compare` at `--concurrency 1` and `--concurrency 16`) and keep throughput at `data_parallel_size: 2` within 3% tok/s and 10% TTFT p50 of the `prefix_affinity` run of the same scenario, under `scripts/bench-lock.sh` when the numbers are kept. A policy aimed at multi-turn traffic also compares `cached_tokens_ratio` with `turbine-bench --profile multi-turn`. Both GPUs must be free (`amd-smi monitor`, the k3s pods) and the lab rules of `AGENTS.md` apply.

## Pitfalls

- A policy that picks an ineligible replica while another is eligible defeats the pressure controller: the busy replica queues or rejects work a healthy one would take.
- `choose` runs on the request path for every request: no allocation-heavy work, no locks, no clock, no randomness (a random pick breaks prefix reuse and makes routing untestable).
- Keep no per-replica state across calls: the server passes only the replicas that are ready right now, and a remembered replica may have gone away or changed state since.
- The reason is a metric label: report `pressure_avoidance` only when a better choice was skipped for pressure, so dashboards stay meaningful.
