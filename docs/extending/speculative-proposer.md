# Adding a speculative proposer

Phase 8 (S-2): a speculative proposer proposes tokens a target verifies in one forward. The
extension point is `turbine_model::speculative`: a `Proposer` trait, a compiled-in
`PROPOSERS` registry and the `draft_model` implementation (`speculative.method: draft`). The
verifier and the acceptance maths stay in `turbine-scheduler`; a proposer only produces greedy
token ids for a sequence.

## Files to add

- `crates/turbine-model/src/speculative/<name>.rs` — the implementation: it owns a draft
  `ModelExecutor`, one private one-sequence `BlockPool`, its `BlockTable` and the
  `begin`/`propose`/`truncate`/`release` lifecycle. The skeleton is
  [`draft.rs`](../../crates/turbine-model/src/speculative/draft.rs).
- `crates/turbine-model/src/speculative/mod.rs` — add the factory to `PROPOSERS` (do not sort
  the list; registration order is the lookup order) and its `ProposerFactory` impl.
- `docs/extending/README.md` — add the row to the extension-point table.

A proposer owns the draft model's memory: the server charges it through the Phase 3 budget
(`DraftBudget`) and passes the prepared executor and its slot-sized pool in
`DraftProposerConfig`. Do not allocate a second pool or a second sequence slot inside the
proposer; the single-slot rule is what bounds the memory.

## Registry entry

`crates/turbine-model/src/speculative/mod.rs`:

```rust
pub static PROPOSERS: &[&dyn ProposerFactory] = &[&DraftModelFactory, &MyFactory];
```

`speculative.method` is the only config key that selects a proposer: `draft` maps to
`draft_model`, and a name with no registered factory is refused by the support matrix before
any port is bound. There is no `speculative.proposer` key; the method doubles as the registry
name. `/turbine/v1/status` reports the loaded draft under `speculative`.

## Conformance suite

`conformance(factory, cfg)` runs the contract (a `propose` before `begin` is `NotBegun`, the
slot is not shared, `truncate` keeps the prefix, `release` frees it) and the registry test pins
the names:

- `scripts/remote-cargo.sh test -p turbine-model speculative::tests::draft_model_conformance`
- `scripts/remote-cargo.sh test -p turbine-model --test conformance proposer_registry`

## Lab checks

- `scripts/lab-test.sh novanas --tier quick` after the executor-facing change.
- `scripts/lab-bench.sh --gpu 0 --model llama -- --set speculative.method=draft --set speculative.draft_model_path=/home/piwi/turbine-models/llama-3.2-1b-instruct`
  for golden c1/c16, acceptance and throughput.
- A new proposer's support row is added only after the S-11 gate (golden, serial A/B, the
  ≥ 1.3× single-stream bar, TTFT, vLLM or the no-vLLM rule and the 10-minute soak); until then
  it stays `experimental`.

## Pitfalls

- The draft KV must be truncated to the target's accepted length before every `propose`; a
  stale proposal chain silently changes the proposals while the greedy output still looks
  plausible.
- `begin` prefills the accepted prompt itself; do not reuse the target's KV or its prefix
  cache — the draft is a different model with its own layout.
- The draft executor's `kv_layout` must equal `draft.kv_layout(block_tokens)`; a mismatch is a
  startup refusal, not a runtime surprise.
- `propose` must not observe or advance the target's sampler; it returns proposals only. The
  verifier owns acceptance, the fallback draw and the emitted-token replay.
