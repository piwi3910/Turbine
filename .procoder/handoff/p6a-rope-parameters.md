# Handoff: p6a-rope-parameters (transformers-5 `rope_parameters`)

Branch `p6a-rope-parameters`, from integration 735c7c8. Status: done, no GPU runs. The lead reruns the
W4A4 8B GSM8K after merging.

## Bug

transformers-5.x exports (`/home/piwi/turbine-models/llama-3.1-8b-instruct-mxfp4-a4/config.json`,
transformers_version 5.9.0) have no top-level `rope_theta` / `rope_scaling`, only
`rope_parameters: {rope_theta: 500000, rope_type: llama3, factor: 8, low_freq_factor: 1,
high_freq_factor: 4, original_max_position_embeddings: 8192}`. `RawConfig` didn't read that block, so
`load_model_config` served theta 10000 (`DEFAULT_ROPE_THETA`) with no llama3 scaling. That explains
the 0.54 GSM8K against vLLM's 0.735.

## Change (commit `handoff(crates/turbine-model/src/config.rs): …`)

- `RawConfig.rope_parameters` is read from the same object as every other key: the top level, or
  `text_config` for a wrapper, as `families::resolve` picks.
- `split_rope_parameters` splits the block into its `rope_theta` and the scaling part (the rest of
  the mapping, only when it names `rope_type` / `type`). The scaling part goes through the existing
  `parse_rope_scaling` with key `rope_parameters`: `default` means no scaling, llama3 and yarn are
  unchanged, and refusals name `rope_parameters.rope_type`. A per-layer-type mapping
  (`{full_attention: {…}, …}`, transformers' Case 2) is refused as `Unsupported rope_parameters`.
- Precedence: when a top-level key and `rope_parameters` both give a value, they must agree.
  Otherwise the loader refuses with `rope_theta X and rope_parameters.rope_theta Y disagree`, or
  `rope_scaling {…} and rope_parameters {…} disagree` (scaling is compared after parsing). For
  reference, transformers 5.9.0 resolves such a conflict silently: `rope_scaling or
  rope_parameters` for scaling, `rope_parameters.rope_theta` first for theta
  (`RotaryEmbeddingConfigMixin.convert_rope_params_to_dict`).
- `model.rope_scaling` still replaces the scaling part wholesale. The config's own scaling isn't
  parsed then, as before. The theta still resolves (and conflict-checks) from `config.json`.
- The rope_theta default (my choice): transformers defaults rope_theta for every model type
  (`default_theta` 10000; `MixtralConfig` overrides it to 1e6). The loader keeps that default only
  for the model types in `TRANSFORMERS_DEFAULT_THETA`: llama, mistral, mixtral (1e6, previously
  wrong at 1e4), olmoe, qwen3 and qwen3_moe, recorded from 5.9.0. It logs
  `WARN event="rope_theta_defaulted" family model_type rope_theta`. A missing or unlisted
  `model_type` is refused: `no rope_theta or rope_parameters.rope_theta, and no transformers default
  is known for model_type …`. A new family has to add its row. A `ModelFamily::default_rope_theta()`
  trait method would be the more pluggable home. I didn't make that change, to keep the diff inside
  config.rs; say if you want it.
- S-16: `rope_identity()` is built from the resolved `rope_theta` / `rope_scaling`. The test checks
  that the transformers-5 and classic layouts produce identical identities and `inv_freq`.

## Tests (`cargo test -p turbine-model --lib config::tests`, 18/18)

- `rope_parameters_transformers5_layout`: fixture `tests/fixtures/llama-3.1-8b-transformers5/config.json`
  (the W4A4 config.json verbatim minus `quantization_config`) gives theta 500000 + llama3 factor 8, with
  `inv_freq` and `rope_identity` equal to the same config moved to the classic layout. It also covers
  both layouts agreeing, `rope_type: default`, a refused `dynamic`, the override, and a `text_config` wrapper.
- `rope_parameters_conflict_is_refused`: covers the theta conflict and the scaling conflict.
- `rope_theta_default_warns_or_refuses`: checks the WARN, captured by an in-test `tracing::Subscriber`
  (no new dev-dependency); no event when theta is explicit; Mixtral defaults to 1e6; missing and
  unknown `model_type` are refused.
- Red first: all three failed before the change (theta 10000, no WARN). The mutation that ignores
  `rope_parameters` (not committed) makes `rope_parameters_transformers5_layout` and
  `rope_parameters_conflict_is_refused` fail.

## Open

- `partial_rotary_factor` (top level or inside `rope_parameters`) is still ignored. No registered
  family uses it, but a config that sets it would be served silently wrong. A refusal in the same
  place would be a 5-line follow-up.
- Rerun the W4A4 8B GSM8K (lead) after merge; expected to close the 0.54 vs 0.735 gap if RoPE was the
  whole cause.
