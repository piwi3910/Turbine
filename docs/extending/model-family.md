# Adding a model family

A model family is everything that differs between decoder architectures: the Hugging Face class names it serves, its own `config.json` keys, the checkpoint's weight slots, the op requirements, the workspace, the default tool-call format and the executor. Point name `model_family`; selected by `config.json` `architectures[0]` (or a wrapper's `text_config.architectures[0]`) through `turbine_model::families::resolve`. Registered: `llama`, `olmoe`, `qwen3`, `qwen3_moe`, `mistral`, `mixtral`.

## The trait

`turbine_model::families::ModelFamily: Module` (`crates/turbine-model/src/families/mod.rs`):

| Method                                                      | Must do                                                                                                                                                                                                                                                                                                      |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `hf_architectures()`                                        | The `architectures[0]` values it serves; no other family may claim one (a double claim is refused naming both).                                                                                                                                                                                              |
| `parse_config(&Value)`                                      | Reads and checks only the family's own keys (the shared ones — layers, widths, heads, RoPE, EOS — are parsed by `crates/turbine-model/src/config.rs`); returns `FamilyConfig { moe, qk_norm, qk_norm_per_head }`. Malformed value → `invalid(..)`; a refused feature → `unsupported(key, value, supported)`. |
| `weight_slots(&ModelArchConfig)`                            | Every parameter the executor loads, checkpoint name → slot. Reuse `llama_slots` / `olmoe_slots` / `mixtral_slots` / `dense_slots` where the layout matches.                                                                                                                                                  |
| `requirements(cfg, block_tokens, opts)`                     | Every op config the forward runs; the kernel registry is built from this list at startup, so a missing entry fails the first forward.                                                                                                                                                                        |
| `workspace_bytes(cfg, limits)`                              | Device bytes of the executor's buffers (enters the memory budget).                                                                                                                                                                                                                                           |
| `default_tool_format()`                                     | The `tool_format` name used when `model.tool_call_parser` is null (e.g. `Some(HERMES)`), or `None`.                                                                                                                                                                                                          |
| `build_executor(cfg, weights, registry, mem, limits, opts)` | The executor. On the shared decoder: `DecoderExecutor::new(cfg, decoder_spec(), ..)`.                                                                                                                                                                                                                        |
| `write_tiny(dir, seed)`                                     | Writes a tiny synthetic checkpoint (ChaCha8 weights seeded by `seed`, the tiny tokenizer) — what the conformance suite runs on.                                                                                                                                                                              |
| `tp_decoder_spec()` (optional)                              | Tensor parallelism (Phase 5): `Some(decoder_spec())` when the hooks run on a rank's shard (`crates/turbine-model/src/tp.rs` splits heads, KV heads, intermediate and vocabulary); default `None` refuses tp > 1.                                                                                             |

A pre-norm decoder is a `DecoderSpec { attention, ffn }` of hooks from `crates/turbine-model/src/executor/decoder/hooks/mod.rs`: attention `PLAIN_ATTENTION`, `QK_NORM_FULL`, `QK_NORM_PER_HEAD`; FFN `SWIGLU`, `MOE`. A new layer variant is a new hook file there (`AttentionHook` or `FfnHook` in `crates/turbine-model/src/executor/decoder/mod.rs`), not a branch in the skeleton.

## Files to add

1. `crates/turbine-model/src/families/<name>.rs` — the family (see `crates/turbine-model/src/families/qwen3.rs`):

```rust
pub struct Phi3;

impl Module for Phi3 {
    fn name(&self) -> &'static str { "phi3" }
}

impl ModelFamily for Phi3 {
    fn hf_architectures(&self) -> &'static [&'static str] { &["Phi3ForCausalLM"] }
    fn parse_config(&self, text: &serde_json::Value) -> Result<FamilyConfig, ModelError> {
        // refuse what the decoder cannot run, e.g. sliding windows
        Ok(FamilyConfig::default())
    }
    fn weight_slots(&self, cfg: &ModelArchConfig) -> Vec<WeightSlot> { dense_slots(cfg, None) }
    fn requirements(&self, cfg: &ModelArchConfig, block_tokens: u32, opts: ExecutorOptions) -> Vec<OpRequirement> {
        DecoderExecutor::requirements(cfg, &decoder_spec(), block_tokens, opts)
    }
    fn workspace_bytes(&self, cfg: &ModelArchConfig, limits: ExecutorLimits) -> u64 {
        DecoderExecutor::workspace_bytes(cfg, &decoder_spec(), limits)
    }
    fn default_tool_format(&self) -> Option<&'static str> { None }
    fn write_tiny(&self, dir: &Path, seed: u64) -> TinySpec { write_tiny_family(dir, self.name(), seed) }
    fn build_executor(&self, cfg: &ModelArchConfig, weights: LoadedWeights, registry: Arc<KernelRegistry>,
                      mem: Arc<dyn DeviceMemory>, limits: ExecutorLimits, opts: ExecutorOptions)
                      -> Result<Box<dyn ModelExecutor>, ModelError> {
        Ok(Box::new(DecoderExecutor::new(cfg, decoder_spec(), weights, registry, mem, limits, opts)?))
    }
}

pub fn decoder_spec() -> DecoderSpec { DecoderSpec { attention: PLAIN_ATTENTION, ffn: SWIGLU } }
```

2. Its tiny checkpoint: a `config.json` arm in `family_config_json` and the name in `TINY_PHASE8_FAMILIES` (`crates/turbine-model/src/testing/tiny.rs`), keys copied from a real checkpoint's `config.json` (keep one under `crates/turbine-model/tests/fixtures/<slug>/`).
3. If the layer math is new (a norm, an activation, a router), the same math in the naive reference `crates/turbine-model/src/testing/naive.rs`, driven only by `ModelArchConfig` fields — never by your executor or hooks — so the suite compares two independent implementations.
4. If the chat template renders tools, a render fixture test in `crates/turbine-model/tests/family_templates.rs`.
5. Support-matrix rows in `crates/turbine-core/src/support.rs` (see Pitfalls).

## Registry entry

In `crates/turbine-model/src/families/mod.rs`: `pub mod <name>;`, `pub use <name>::<Type>;`, and `&<Type>` appended to `FAMILIES`. The list order is the order error messages list families; also add the name to the pinned list in `crates/turbine-model/src/registries.rs` (`registry_conformance::families`) and the HF name to `resolve_by_hf_name_and_refuse_unknown` in `families::tests`.

## Conformance suite

`families_suite` (`crates/turbine-model/src/conformance/families.rs`) writes each family's tiny checkpoint and runs it on the CPU provider: `hf_names`, `tiny` (its `architectures[0]` is one of the family's), `load` (every tensor maps to a slot, the executor builds), `naive` (max |Δ logit| ≤ 2e-2 against `testing::naive::forward`, argmax equal), `ragged` (a ragged batch equals single-sequence runs bit for bit), `chunked` (prefill in chunks of 3 equals whole prefill bit for bit), `pages` (16-token pages on an out-of-order table equal one 128-token page bit for bit), `fused` (`fused_ops` on vs off within 2e-2). The weight-format suite also loads every family's tiny checkpoint.

- `scripts/remote-cargo.sh test -p turbine-model registry_conformance`
- `scripts/remote-cargo.sh test -p turbine-model families::` — resolution, config parsing and slot tests.
- `scripts/remote-cargo.sh test -p turbine-model --test families --test family_templates`

## Lab checks

On the CPU backend a family is served under the `cpu` row (`experimental`); on a GPU vendor it is refused until a support-matrix row says otherwise. Before a GPU row flips to `supported`:

- `scripts/lab-test.sh novanas -- -p turbine-model --test tiny_model` — `hip_matches_cpu`, `hip_decode_graph_matches_eager` and `hip_trace_vs_cpu` on a tiny checkpoint of the family (GPU tests need `head_dim` 128, the only one the HIP attention supports).
- A golden reference from transformers (`scripts/golden/hf_reference.py`) under `tests/golden/<slug>/` with its `tolerance.json`, and `turbine-golden compare` at `--concurrency 1` and `--concurrency 16` against a lab server serving the weights.
- Existing families must not move: `scripts/lab-bench.sh --gpu 0 --model llama` and `--model olmoe` within 3% tok/s and 10% TTFT p50 of the last row of `.procoder/perf-log.md`, with a row appended.

## Pitfalls

- **Support matrix**: a family's HF name needs rows in `SUPPORT_MATRIX` (`crates/turbine-core/src/support.rs`). With none, a GPU vendor resolves to `unsupported` ("no support-matrix row") and the server exits 2 before binding; the `cpu` wildcard row serves it as `experimental`. Add a refusing `family_row("amd", "<HfName>")` until the GPU path is validated, then a fully specific `supported` row (`baseline_rows_present` checks supported rows are BF16 / BF16 KV / no speculation).
- **Decode-graph keys**: on the shared decoder, decode graphs capture embedding → LM head → `logits_reduce` and replay by `GraphKey` (sequences, reduce top-n, block-table width, feed word — `crates/turbine-model/src/executor/graphs.rs`). Everything a hook reads must live in buffers allocated once (`FfnHook::alloc`) and be uploaded before launch; a hook that reads device results on the host mid-forward must return `false` from `FfnHook::graph_capturable` for those batches, or replays run stale values. Never add a shape-dependent input that is not in the key; `graph_key_matches_main_formula` pins the key formula.
- **No family branches elsewhere**: never match on a family or architecture name outside `crates/turbine-model/src/families/` (the Phase 2m acceptance greps for it). Differences go into `FamilyConfig`, `ModelArchConfig` fields, hooks or the trait.
- `requirements` must list every op config for every `ExecutorOptions` combination you run (fused and unfused), or startup fails with "no provider supports" on the lab.
- Refuse unsupported `config.json` features in `parse_config` with `unsupported(..)` (`attention_bias: true` is refused for every family) — never silently ignore a key that changes the math.
