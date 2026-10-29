# Adding a speculative-length (k) policy

Phase 8: how many tokens the proposer drafts per round. The extension point is
`turbine_model::speculative::k_policy`: a `KPolicy` trait, the compiled-in `KPOLICIES` registry
and two implementations — `fixed` (default, `speculative.num_tokens` unchanged) and `adaptive`
(raises `k` after fully accepted rounds, lowers it when proposals are wasted). The verifier and
the acceptance maths stay in `turbine-scheduler`; a policy only chooses a count and reads the
round's outcome.

## Files to add

- `crates/turbine-model/src/speculative/k_policy.rs` — the implementation: a `Module` and a
  `KPolicy` impl, plus the `KState` the engine keeps per sequence. The skeleton is the existing
  [`k_policy.rs`](https://github.com/piwi3910/Turbine/blob/main/crates/turbine-model/src/speculative/k_policy.rs).
- `crates/turbine-model/src/speculative/mod.rs` — export the module (`pub mod k_policy;`).
- `crates/turbine-model/src/speculative/k_policy.rs` — add the policy to `KPOLICIES` (do not
  sort the list; registration order is the lookup order).
- `crates/turbine-server/src/modules.rs` — the names feed `known_module_names` (the pre-bind
  check), from `k_policy::registry()`.
- `docs/extending/README.md` — add the row to the extension-point table.

The engine owns the control loop: one `KState` per sequence, `choose` before a round and
`observe(proposed, accepted)` after it. A policy is a stateless singleton — never keep
per-sequence state inside the policy, or it leaks across requests.

## Registry entry

`crates/turbine-model/src/speculative/k_policy.rs`:

```rust
pub static KPOLICIES: Registry<dyn KPolicy> =
    Registry::new("speculative_k_policy", &[&FixedPolicy, &AdaptivePolicy]);
```

`speculative.k_policy` holds the name; an unregistered name is refused by
`Config::validate_modules` before any port is bound. `speculative.num_tokens` is the starting
`k` (and the ceiling for `fixed`); every round is additionally capped by
`MAX_SPECULATIVE_TOKENS` and the sequence's context room. `/turbine/v1/status` reports the
chosen policy; `turbine_speculative_k` is the last round's `k`.

## Conformance suite

The suite runs over the registry: each name is well-formed, `choose` stays in `1..=cap`, and
the `adaptive` policy climbs on fully accepted rounds and backs off on rejected ones.

- `cargo test -p turbine-model --lib speculative::k_policy`
- `cargo test -p turbine-model --lib k_policy::tests` (the pinned names)

## Lab checks

- `scripts/lab-test.sh novanas --tier quick` after the engine-facing change.
- `scripts/lab-bench.sh --gpu 0 --model llama -- --set speculative.method=mtp --set speculative.k_policy=adaptive`
  for golden c1/c16 and throughput, A/B'd against `k_policy=fixed` on the same workload.

## Pitfalls

- The objective is accepted tokens per unit time, not the acceptance fraction: growing `k` on a
  merely "good" round wastes drafts (the verify batch and every draft forward cost time). A
  policy that chases acceptance alone collapses to `k = 1`.
- Always damp with a dwell/hysteresis; the per-round accepted fraction is noisy and an
  undamped controller oscillates between extremes.
- `k` must never exceed the context room (`max_seq_len - accepted_len - 1`) or the verify batch
  runs past the sequence; the engine clamps, but a policy returning a huge `k` fights the cap.
- `observe` must be called for every round that ran, including fully rejected ones, or the
  controller's estimate is biased upward and `k` creeps.
