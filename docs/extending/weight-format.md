# Adding a weight format

A weight format is a checkpoint _packaging_: how a checkpoint's linear layers are stored (BF16, FP8 with per-tensor / per-channel / block scales, INT4 groups, MXFP4), how activations are quantized before them, which tensors a layer needs and how their bytes are repacked into the layout the kernels consume (Phase 6a S-3). Point name `weight_format`; selected by `weights::detect(config.json)`: the first registered format whose `check_config` accepts the top-level `config.json`. Registered: `bf16`. Activations always run in BF16 (`weights::ACTIVATION_DTYPE`) and the KV dtype is configured (`kv.dtype`), not a property of the checkpoint.

## The trait

`turbine_model::weights::WeightFormat: Module` (`crates/turbine-model/src/weights/mod.rs`):

| Method                                 | Must do                                                                                                                                                                                                                                                                 |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name()` (from `Module`)               | Unique packaging name (`bf16`, `ct_fp8`, `awq`, …).                                                                                                                                                                                                                     |
| `check_config(&Value)`                 | `Ok` when the top-level `config.json` declares this packaging (`quantization_config.quant_method`, compressed-tensors `config_groups`, `torch_dtype` / `dtype`); else the refusal naming the key, via `unsupported(key, value, supported)`.                             |
| `check_tensor(&TensorEntry)`           | `Ok` when a checkpoint tensor is stored as this packaging stores it; else `ModelError::Unsupported` naming the tensor.                                                                                                                                                  |
| `column()`                             | The support-matrix `weight_format` column (`WeightFormatColumn` in `crates/turbine-core/src/support.rs`) the server's key uses.                                                                                                                                         |
| `scheme(&LinearSlot)` (default `Bf16`) | The `QuantScheme` of one linear layer (`cfg.linear_slots()`: every 2-D slot except the token embedding); layers the checkpoint leaves unquantized (`ignore: ["lm_head"]`) report `QuantScheme::Bf16`.                                                                   |
| `activation()` (default `None`)        | The `ActivationQuant` applied before the quantized layers (static or dynamic FP8, MXFP4 emulation).                                                                                                                                                                     |
| `slots(&WeightSlot)` (default: itself) | The checkpoint tensors the loader reads for a family slot: a quantized layer's packed data, scales, zero points or activation scale.                                                                                                                                    |
| `slot_dtype(&WeightSlot)`              | The device dtype a loaded slot is stored in (`DType::F8E4M3`, `DType::U8` for packed bytes, `F32` scales, `BF16` otherwise).                                                                                                                                            |
| `repack(&WeightSlot, bytes)`           | Rewrites a slot's checkpoint bytes into the layout the kernels consume (`crates/turbine-kernels/src/quant.rs` documents it): e.g. AWQ's interleaved nibble order to the plain one. Identity by default.                                                                 |
| `weight_bytes(&ModelArchConfig)`       | Device bytes of every parameter (the memory budget's weight term). The default sums each linear layer at `scheme(..).bytes(n, k)` (packed data, F32 scales, zero points) and everything else in BF16; override only when the packaging stores something else on device. |

## Files to add

1. `crates/turbine-model/src/weights/<name>.rs` (see `crates/turbine-model/src/weights/bf16.rs`):

```rust
pub struct CtFp8;

impl Module for CtFp8 {
    fn name(&self) -> &'static str { "ct_fp8" }
}

impl WeightFormat for CtFp8 {
    fn check_config(&self, top: &serde_json::Value) -> Result<(), ModelError> {
        // quant_method: compressed-tensors, format: float-quantized, 8-bit float weights, …
        todo!()
    }
    fn check_tensor(&self, e: &TensorEntry) -> Result<(), ModelError> { todo!() }
    fn column(&self) -> WeightFormatColumn { WeightFormatColumn::Fp8 }
    fn scheme(&self, layer: &LinearSlot) -> QuantScheme {
        if layer.name == "lm_head.weight" { QuantScheme::Bf16 } else { QuantScheme::Fp8Channel }
    }
    fn activation(&self) -> ActivationQuant { ActivationQuant::Fp8PerTokenDynamic }
    fn slots(&self, base: &WeightSlot) -> Vec<WeightSlot> { todo!() /* weight + weight_scale */ }
    fn slot_dtype(&self, slot: &WeightSlot) -> DType { todo!() }
}
```

2. A tiny checkpoint in the packaging: at least one registered family's `write_tiny` must be able to write it (`crates/turbine-model/src/testing/tiny.rs`, written in Rust, no Python), or the suite's `tiny` check has nothing to load.
3. Kernel support: a quantized layer runs through the quantized GEMM (`OpKind::QGemm`, kernel ABI v2.9), whose CPU reference (`crates/turbine-kernels/src/cpu/quant.rs`) dequantizes every scheme; each GPU provider registers implementations per scheme, or the kernel registry refuses the model at startup.
4. A support-matrix column value and rows (`crates/turbine-core/src/support.rs`): each Phase 6a column is refused naming `phase-6a-quantization` until its proof checkpoint passes the umbrella S-3 gate.

## Registry entry

In `crates/turbine-model/src/weights/mod.rs`: `pub mod <name>;`, `pub use <name>::<Type>;`, and `&<Type>` in `WEIGHT_FORMATS` **in detection order** — `detect` returns the first format that accepts `config.json`, and on no match reports the first format's refusal, so keep `bf16` first. Update the pinned list in `crates/turbine-model/src/registries.rs` (`registry_conformance::weight_formats`).

## Conformance suite

`weights_suite` (`crates/turbine-model/src/conformance/weights.rs`): `tiny` (at least one registered family's tiny `config.json` declares the format, and each such checkpoint loads through `WeightLoader::load_format` with every tensor accepted and mapped to a slot) and `bytes` (for each of those checkpoints the uploaded bytes equal `weight_bytes`, packed data and scales included). The family suite then runs those checkpoints end to end.

- `scripts/remote-cargo.sh test -p turbine-model registry_conformance`
- `scripts/remote-cargo.sh test -p turbine-model weights::` — refusal messages, detection and the BF16 description (`bf16_describes_linear_layers`).

## Lab checks

`scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops` for the scheme's quantized GEMM, a golden reference of a real checkpoint in the packaging (`tests/golden/<slug>/`, made with `scripts/golden/quant_reference.py`) compared at `--concurrency 1` and `--concurrency 16`, `scripts/lab-bench.sh --model <proof model>` against the Phase 6a targets, and `scripts/lab-bench.sh --gpu 0 --model llama` / `--model olmoe` unchanged within the no-regression bound (the BF16 models must not move).

## Pitfalls

- The decoder runs a quantized layer only once the quantized GEMM is wired for its scheme; until then `check_weight_format` in `crates/turbine-model/src/executor/decoder/mod.rs` refuses the format naming the first quantized layer.
- Fused stacks (Q/K/V, gate/up) concatenate the parts' packed rows and scales; parts with different per-tensor scales must be rescaled to a per-channel vector at load, and parts with different schemes are refused.
- `n` or `k` not a multiple of the group or block size is refused (`quant_scheme_unsupported`), as is act-order GPTQ (`gptq_act_order`).
- Do not add `DType::BF16` literals outside `crates/turbine-model/src/weights/bf16.rs` and tests: take the activation dtype from `weights::ACTIVATION_DTYPE` / `ModelArchConfig::activation_dtype`.
