# Adding a weight format

A weight format says how a checkpoint's weights are stored and which dtypes the loader, the executor and the KV cache use. Point name `weight_format`; selected by `weights::detect(config.json)`: the first registered format whose `check_config` accepts the top-level `config.json`. Registered: `bf16`.

## The trait

`turbine_model::weights::WeightFormat: Module` (`crates/turbine-model/src/weights/mod.rs`):

| Method                       | Must do                                                                                                                                                                                 |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name()` (from `Module`)     | Unique name (`bf16`, `fp8`, …).                                                                                                                                                         |
| `check_config(&Value)`       | `Ok` when the top-level `config.json` declares this format (`quantization_config`, `torch_dtype` / `dtype`); else the refusal naming the key, via `unsupported(key, value, supported)`. |
| `check_tensor(&TensorEntry)` | `Ok` when a checkpoint tensor is stored in this format; else `ModelError::Unsupported` naming the tensor.                                                                               |
| `weight_dtype()`             | The dtype of the loaded parameters.                                                                                                                                                     |
| `activation_dtype()`         | The dtype of the executor's activations.                                                                                                                                                |
| `kv_dtype()`                 | The dtype of the KV cache (sizes the KV layout and the memory budget).                                                                                                                  |
| `bytes_per_param()`          | Stored bytes per parameter (the budget's weight term); equals `weight_dtype().size_bytes()` for an unpacked format.                                                                     |

## Files to add

1. `crates/turbine-model/src/weights/<name>.rs` (see `crates/turbine-model/src/weights/bf16.rs`):

```rust
pub struct Fp16;

impl Module for Fp16 {
    fn name(&self) -> &'static str { "fp16" }
}

impl WeightFormat for Fp16 {
    fn check_config(&self, top: &serde_json::Value) -> Result<(), ModelError> {
        match top.get("torch_dtype").and_then(|v| v.as_str()) {
            Some("float16") => Ok(()),
            other => Err(unsupported("torch_dtype", other.unwrap_or("<missing>"), "float16")),
        }
    }
    fn check_tensor(&self, e: &TensorEntry) -> Result<(), ModelError> {
        if e.dtype == Dtype::F16 { Ok(()) } else { Err(/* Unsupported naming e.name */ todo!()) }
    }
    fn weight_dtype(&self) -> DType { DType::F16 }
    fn activation_dtype(&self) -> DType { DType::F16 }
    fn kv_dtype(&self) -> DType { DType::F16 }
    fn bytes_per_param(&self) -> u64 { 2 }
}
```

2. A tiny checkpoint in the format: at least one registered family's `write_tiny` must be able to write it (options in `crates/turbine-model/src/testing/tiny.rs`), or the suite's `tiny` check has nothing to load.
3. Kernel support: every op config the families request carries the dtype, so the CPU reference (`crates/turbine-kernels/src/cpu/`) and each GPU provider must support it, or the kernel registry refuses the model at startup.
4. A support-matrix column value and rows (see Pitfalls).

## Registry entry

In `crates/turbine-model/src/weights/mod.rs`: `pub mod <name>;`, `pub use <name>::<Type>;`, and `&<Type>` in `WEIGHT_FORMATS` **in detection order** — `detect` returns the first format that accepts `config.json`, and on no match reports the first format's refusal, so keep `bf16` first unless yours must win over it. Update the pinned list in `crates/turbine-model/src/registries.rs` (`registry_conformance::weight_formats`).

## Conformance suite

`weights_suite` (`crates/turbine-model/src/conformance/weights.rs`): `bytes` (`bytes_per_param` equals the size of `weight_dtype`) and `tiny` (at least one registered family's tiny `config.json` declares the format, and each such checkpoint loads through `WeightLoader::load_format` with every tensor accepted and mapped to a slot). The family suite then runs those checkpoints end to end.

- `scripts/remote-cargo.sh test -p turbine-model registry_conformance`
- `scripts/remote-cargo.sh test -p turbine-model weights::` — refusal messages and detection.

## Lab checks

`scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops` for the new dtype's op configs, a golden reference of a real checkpoint in the format (`tests/golden/<slug>/`) compared at `--concurrency 1` and `--concurrency 16`, and `scripts/lab-bench.sh --gpu 0 --model llama` / `--model olmoe` unchanged within 3% tok/s and 10% TTFT p50 of the last row of `.procoder/perf-log.md` (the BF16 models must not move).

## Pitfalls

- **Support matrix**: the key the server builds is BF16 only today (`SupportKey::bf16` in `crates/turbine-core/src/support.rs`, used by `crates/turbine-server/src/support_startup.rs`). A new format needs a `WeightFormatColumn` value (or reuses one of the reserved quantized columns, which resolve to `unsupported` until their track closes), rows for it, and the server's key built from the detected format's column instead of the constant.
- The shared decoder runs weights, activations and KV in one dtype and refuses a format where they differ (`check_weight_format` in `crates/turbine-model/src/executor/decoder/mod.rs`); a mixed format needs that skeleton work first.
- Do not add `DType::BF16` literals outside `crates/turbine-model/src/weights/bf16.rs` and tests: code takes dtypes from `cfg.weight_format` (the Phase 2m acceptance counts them).
- A packed format (`bytes_per_param` ≠ the dtype size) fails `bytes` as written; extend the suite with the packing rule in the same change.
