# Adding a model family

A model family is everything that differs between decoder architectures: the Hugging Face class names it serves, its own `config.json` keys, the checkpoint's weight slots, the op requirements, the workspace, the default tool-call format, the per-layer attention kinds and per-sequence state, the default reasoning parser and the executor. Point name `model_family`; selected by `config.json` `architectures[0]` (or a wrapper's `text_config.architectures[0]`) through `turbine_model::families::resolve`. Registered: `llama`, `olmoe`, `qwen3`, `qwen3_moe`, `mistral`, `mixtral`, `gpt_oss`, `qwen3_5`, `qwen3_5_moe`.

## The trait

`turbine_model::families::ModelFamily: Module` (`crates/turbine-model/src/families/mod.rs`):

| Method                                                      | Must do                                                                                                                                                                                                                                                                                                                      |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hf_architectures()`                                        | The `architectures[0]` values it serves; no other family may claim one (a double claim is refused naming both).                                                                                                                                                                                                              |
| `parse_config(&Value)`                                      | Reads and checks only the family's own keys (the shared ones — layers, widths, heads, RoPE, EOS — are parsed by `crates/turbine-model/src/config.rs`); returns `FamilyConfig { moe, qk_norm, qk_norm_per_head, sliding_window }`. Malformed value → `invalid(..)`; a refused feature → `unsupported(key, value, supported)`. |
| `weight_slots(&ModelArchConfig)`                            | Every parameter the executor loads, checkpoint name → slot. Reuse `llama_slots` / `olmoe_slots` / `mixtral_slots` / `dense_slots` where the layout matches.                                                                                                                                                                  |
| `requirements(cfg, block_tokens, opts)`                     | Every op config the forward runs; the kernel registry is built from this list at startup, so a missing entry fails the first forward.                                                                                                                                                                                        |
| `workspace_bytes(cfg, limits)`                              | Device bytes of the executor's buffers (enters the memory budget).                                                                                                                                                                                                                                                           |
| `default_tool_format()`                                     | The `tool_format` name used when `model.tool_call_parser` is null (e.g. `Some(HERMES)`), or `None`.                                                                                                                                                                                                                          |
| `build_executor(cfg, weights, registry, mem, limits, opts)` | The executor. On the shared decoder: `DecoderExecutor::new(cfg, decoder_spec(), ..)`.                                                                                                                                                                                                                                        |
| `write_tiny(dir, seed)`                                     | Writes a tiny synthetic checkpoint (ChaCha8 weights seeded by `seed`, the tiny tokenizer) — what the conformance suite runs on.                                                                                                                                                                                              |
| `tp_decoder_spec()` (optional)                              | Tensor parallelism (Phase 5): `Some(decoder_spec())` when the hooks run on a rank's shard (`crates/turbine-model/src/tp.rs` splits heads, KV heads, intermediate and vocabulary); default `None` refuses tp > 1.                                                                                                             |
| `chat_template_override()` (optional)                       | A tool-aware chat template `(name, Jinja source)` used when the checkpoint's own template renders neither `tools` nor an assistant turn's `tool_calls` and no `model.chat_template` is set (`ChatTemplate::resolve_for`; the mixtral family ships `mixtral_tools.jinja`). Default `None`.                                    |
| `layer_kinds(&ModelArchConfig)` (optional)                  | One `LayerKind` per layer (P7 S-1): `FullAttention`, `SlidingWindow { window }` or `LinearAttention`. Default: every layer `FullAttention`. A family whose `config.json` carries `layer_types` / `sliding_window` / `full_attention_interval` reads it (gpt-oss, Qwen3.5/3.6) — the shared config parser derives nothing, it only stores what the family returns.                                                                                                     |
| `state_layout(&ModelArchConfig)` (optional)                 | The per-sequence state the model keeps beside its paged KV (P7 S-1): `turbine_core::types::StateLayout { linear_layers, recurrent_bytes_per_layer, conv_bytes_per_layer, dtype }`. Default `StateLayout::default()` (empty). Momentum, the budget, the scheduler's worst-case reservation and the KV hierarchy's block snapshots all read it.                                                                                                                     |
| `default_reasoning_parser()` (optional)                     | The reasoning parser `model.reasoning_parser: auto` resolves to (P7 S-10): the `reasoning_parser` name (`qwen_think`, `harmony`), or `None` (default) for a family without marked reasoning.                                                                                                                        |
| `attention_bias()` (optional)                               | Whether the checkpoints carry Q/K/V/O projection biases (`attention_bias: true`): `true` only for gpt-oss. Default `false`; the shared config parser refuses `attention_bias: true` for a family that returns `false`.                                                                                             |
| `skipped_prefixes()` (optional)                             | Checkpoint tensor prefixes the executor does not load but the checkpoint legitimately ships (a vision tower, an MTP head): the loader counts them and logs one `event="weights_skipped"` per prefix instead of warning `unexpected_tensor` per tensor. Default: none.                                                    |
| `text_prefix()` (optional)                                  | The checkpoint prefix of the text model's parameters (`model`, or Qwen3.5's `model.language_model` under the multimodal wrapper); the executor builds its embedding/layer/final-norm names from it. Default `model`.                                                                                                       |

A pre-norm decoder is a `DecoderSpec { attention, ffn }` of hooks from `crates/turbine-model/src/executor/decoder/hooks/mod.rs`: attention `PLAIN_ATTENTION`, `QK_NORM_FULL`, `QK_NORM_PER_HEAD`, `ATTENTION_SINKS_WINDOW` (gpt-oss: biased projections and sinks; the per-layer window comes from `layer_kinds`); FFN `SWIGLU`, `MOE`, `MOE_GPT_OSS`. A hybrid family's linear layers are a hook pair (`GATED_DELTA_NET`, `ATTENTION_OUTPUT_GATE`) beside the full-attention hook, selected per layer from `layer_kinds`. A new layer variant is a new hook file there (`AttentionHook` or `FfnHook` in `crates/turbine-model/src/executor/decoder/mod.rs`), not a branch in the skeleton.

## Phase 7 layer kinds, state and the snapshot rule

`turbine_model::families::LayerKind` (`crates/turbine-model/src/families/mod.rs`, `#[non_exhaustive]`) tells every later stage what a layer is:

- `FullAttention` — the paged causal attention of Phases 1–2 (a KV slot per layer).
- `SlidingWindow { window }` — attention over the last `window` positions; the window reaches the paged descriptor's `window_left` field (ABI v2.17), sinks reach `sinks`.
- `LinearAttention` — a recurrent (Gated DeltaNet) layer with no paged KV: it reads and writes per-sequence state instead.

`ModelArchConfig` carries the derived lists: `layer_kinds: Vec<LayerKind>` (one per layer, from `ModelFamily::layer_kinds`), `state: StateLayout` (`ModelFamily::state_layout`), `attention_layers()` (kinds other than `LinearAttention`) and `kv_layer_index(layer) -> Option<u32>` (the KV slot of an attention layer, `None` for a linear one). `kv_layout(block_tokens).num_layers` is the attention-layer count, so a hybrid's pool sizes only the attention layers; `tp`/`ep`/`pp` derive their shards from the same count.

`turbine_core::types::StateLayout` (`crates/turbine-core/src/types.rs`) is the one type every non-model crate sees: `linear_layers`, `recurrent_bytes_per_layer`, `conv_bytes_per_layer`, `dtype: Option<StateDtype>` and `bytes_per_sequence() = linear_layers × (recurrent + conv)` (saturating; `is_empty()` for an attention-only family). The budget charges `max_running_requests × state_bytes_per_sequence` beside KV, the scheduler's worst-case reservation adds it (`AdmissionInfo.state_bytes`), and the executor owns the device buffer while `turbine_kv::state::{StateSlotPool, SnapshotPool, StateCopy}` keep the index bookkeeping.

**The hybrid snapshot rule** (P7 S-12, `kv.hybrid_snapshot_interval_blocks`, default 8 ≥ 1): a hybrid family's live state is copied at every interval-th *full* block boundary, and at the last full block of a finished sequence, into the block's snapshot (`SnapshotRef { block, entry }` in `SnapshotPool`). A prefix hit ends at the last full block with a snapshot at or before the match, restores that snapshot into the sequence's live slot and recomputes the remaining blocks — there is no partial-block state reuse. Snapshots travel with their block through the tiers (`/turbine/v1/kv` `tiers[].state_bytes`), and jump-forward is refused on a family with state (`jump_forward_hybrid`). The KV crates see only `StateLayout` — they never gain a model or GPU dependency.

**The tiny-trace fixture tool**: a new family's CPU executor is validated against a committed per-op transformers trace of its tiny checkpoint. `crates/turbine-model/examples/write_tiny.rs` writes a registered family's tiny checkpoint (`write_tiny <family> <seed> <dir>`), and `scripts/golden/tiny_trace.py --model-dir <tiny> --prompt-ids 1,2,3,4,5 --out <json>` records the per-layer hidden state, attention output, router top-k and final logits in F32 into `crates/turbine-model/tests/fixtures/tiny-<family>-trace.json` (fixture time only, `nice -n 19` under `fixture.lock`; `--prompt-file` / `--prompt-id` / `--context-tokens` / `--row` trace a real checkpoint's chosen row instead). The `tiny_model` test replays it with `DecoderExecutor::set_trace` within the Phase 1 trace bounds.

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
6. A `layer_kinds` and `state_layout` whose expectations the shared test `families::tests::layer_kinds_and_state_layout` derives independently from the tiny `config.json` (`layer_types` / `sliding_window` / `full_attention_interval`; dense and MoE families: all `FullAttention`, `StateLayout::default()`), and `default_reasoning_parser()` when the model marks its reasoning.
7. The committed per-op trace of the tiny checkpoint, `crates/turbine-model/tests/fixtures/tiny-<family>-trace.json`, generated with `crates/turbine-model/examples/write_tiny.rs` and `scripts/golden/tiny_trace.py` (see above) and replayed by the `tiny_model` reference test.

## Registry entry

In `crates/turbine-model/src/families/mod.rs`: `pub mod <name>;`, `pub use <name>::<Type>;`, and `&<Type>` appended to `FAMILIES`. The list order is the order error messages list families; also add the name to the pinned list in `crates/turbine-model/src/registries.rs` (`registry_conformance::families`) and the HF name to `resolve_by_hf_name_and_refuse_unknown` in `families::tests`.

## Conformance suite

`families_suite` (`crates/turbine-model/src/conformance/families.rs`) writes each family's tiny checkpoint and runs it on the CPU provider: `hf_names`, `tiny` (its `architectures[0]` is one of the family's), `load` (every tensor maps to a slot, the executor builds), `naive` (max |Δ logit| ≤ 2e-2 against `testing::naive::forward`, argmax equal), `ragged` (a ragged batch equals single-sequence runs bit for bit), `chunked` (prefill in chunks of 3 equals whole prefill bit for bit), `pages` (16-token pages on an out-of-order table equal one 128-token page bit for bit), `fused` (`fused_ops` on vs off within 2e-2). The weight-format suite also loads every family's tiny checkpoint. `families::tests::layer_kinds_and_state_layout` iterates the registry and checks each family's kinds and state against its raw `config.json`, so a new family's `layer_kinds` / `state_layout` are covered by registering it.

- `scripts/remote-cargo.sh test -p turbine-model registry_conformance`
- `scripts/remote-cargo.sh test -p turbine-model families::` — resolution, config parsing, slot tests, layer kinds and state layout.
- `scripts/remote-cargo.sh test -p turbine-model --test families --test family_templates`
- `scripts/remote-cargo.sh test -p turbine-model --test tiny_model gpt_oss_matches_reference` — the CPU executor against the committed per-op trace (one test per family, e.g. `qwen3_5_matches_reference`).

## Lab checks

On the CPU backend a family is served under the `cpu` row (`experimental`); on a GPU vendor it is refused until a support-matrix row says otherwise. Before a GPU row flips to `supported`:

- `scripts/lab-test.sh novanas -- -p turbine-model --test tiny_model` — `hip_matches_cpu`, `hip_decode_graph_matches_eager` and `hip_trace_vs_cpu` on a tiny checkpoint of the family (GPU tests need `head_dim` 128, the only one the HIP attention supports).
- A per-op HIP vs CPU trace on the real checkpoint, within the Phase 1 trace bounds: `TURBINE_TEST_MODEL_DIR=/models/<slug> scripts/lab-test.sh novanas -- --release -p turbine-model --test golden -- hip_trace_vs_cpu_qwen3` (one lab test per family; gpt-oss traces a decode step past its 128 window).
- A hybrid family additionally passes the engine's prefix-reuse path on the CPU backend: `scripts/remote-cargo.sh test -p turbine-server engine::r#loop` (`hybrid_prefix_reuse_matches_cold`), and its snapshots travel through the tiers (`kv_gpu hybrid_prefix_reuse_matches_cold` on the lab).
- A golden reference from transformers (`scripts/golden/hf_reference.py`) under `tests/golden/<slug>/` with its `tolerance.json`, and `turbine-golden compare` at `--concurrency 1` and `--concurrency 16` against a lab server serving the weights.
- Existing families must not move: `scripts/lab-bench.sh --gpu 0 --model llama` and `--model olmoe` within 3% tok/s and 10% TTFT p50 of the last row of `.procoder/perf-log.md`, with a row appended.

## Pitfalls

- **Support matrix**: a family's HF name needs rows in `SUPPORT_MATRIX` (`crates/turbine-core/src/support.rs`). With none, a GPU vendor resolves to `unsupported` ("no support-matrix row") and the server exits 2 before binding; the `cpu` wildcard row serves it as `experimental`. Add a refusing `family_row("amd", "<HfName>")` until the GPU path is validated, then a fully specific `supported` row (`baseline_rows_present` checks supported rows are BF16 / BF16 KV / no speculation).
- **Decode-graph keys**: on the shared decoder, decode graphs capture embedding → LM head → `logits_reduce` and replay by `GraphKey` (sequences, reduce top-n, block-table width, feed word — `crates/turbine-model/src/executor/graphs.rs`). Everything a hook reads must live in buffers allocated once (`FfnHook::alloc`) and be uploaded before launch; a hook that reads device results on the host mid-forward must return `false` from `FfnHook::graph_capturable` for those batches, or replays run stale values. Never add a shape-dependent input that is not in the key; `graph_key_matches_main_formula` pins the key formula.
- **No family branches elsewhere**: never match on a family or architecture name outside `crates/turbine-model/src/families/` (the Phase 2m acceptance greps for it). Differences go into `FamilyConfig`, `ModelArchConfig` fields, hooks or the trait.
- `requirements` must list every op config for every `ExecutorOptions` combination you run (fused and unfused), or startup fails with "no provider supports" on the lab.
- Refuse unsupported `config.json` features in `parse_config` with `unsupported(..)` (`attention_bias: true` is refused for every family) — never silently ignore a key that changes the math.
- **Layer kinds and state are load-bearing**: `layer_kinds` sizes the KV pool (`kv_layout(block_tokens).num_layers == attention_layers()`) and picks each layer's hook, and `state_layout` enters the budget and every request's worst-case reservation. A wrong kind or a state byte count off by a factor runs out of memory at the first long context on the lab, not in the suite; `families::tests::layer_kinds_and_state_layout` derives both independently from the tiny `config.json`, so keep the family's `parse_config` deriving from the raw keys, not from `ModelArchConfig`.
- **A hybrid family has no paged KV on its linear layers**: `kv_layer_index` returns `None` there. A family that reports `LinearAttention` but still writes a KV slot (or hides a linear layer as full attention) mis-sizes the pool and the snapshots; a prefix hit can only reuse full-block KV up to the last snapshot, and jump-forward stays off (`jump_forward_hybrid`).
- **`default_reasoning_parser` only names a registered parser**: `auto` resolves it, `none` is always accepted, an unregistered name exits 2 naming `model.reasoning_parser`. A family whose template marks reasoning must return the parser it actually uses or its `reasoning_content` stays empty.
- **Skipping checkpoint tensors is opt-in per prefix**: `skipped_prefixes()` makes the loader count a legitimate prefix (a vision tower, an MTP head) instead of warning per tensor; anything not listed is still an `unexpected_tensor` warning, so a new checkpoint layout must add its prefixes rather than turning warnings off.
