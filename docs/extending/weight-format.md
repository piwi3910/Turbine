# Adding a weight format

A weight format is a checkpoint _packaging_: how a checkpoint's linear layers are stored (BF16, FP8 with per-tensor / per-channel / block scales, INT4 groups, MXFP4), how activations are quantized before them, which tensors a layer needs and how their bytes are repacked into the layout the kernels consume (Phase 6a S-3). Point name `weight_format`; selected by `weights::detect(config.json)`: the first registered format whose `check_config` accepts the top-level `config.json`, configured from it by `configure` (which layers are quantized, the weight and activation schemes). Registered: `bf16`, `ct_fp8` (compressed-tensors FP8: tensor, channel or 128 × 128 block scales; block-scaled weights stay e4m3 with BF16 activations — `turbine_hip_fp8_block` on gfx1201 — and only a layer no selected provider runs is decoded to BF16 at load, logged `fp8_block_decoded`), `hf_fp8` (`quant_method: fp8`, with or without `weight_block_size`), `awq` (AutoAWQ GEMM), `gptq` (AutoGPTQ, no act order), `ct_pack_int4` (compressed-tensors `pack-quantized` W4A16), `ct_mxfp4` (compressed-tensors `mxfp4-pack-quantized`), `quark_mxfp4` (AMD Quark `fp4`, weight-only or W4A4) and `openai_mxfp4` (`quant_method: mxfp4`). NVFP4 and ModelOpt containers are refused naming `phase-2b-nvidia`. Activations always run in BF16 (`weights::ACTIVATION_DTYPE`) and the KV dtype is configured (`kv.dtype`), not a property of the checkpoint.

## The trait

`turbine_model::weights::WeightFormat: Module` (`crates/turbine-model/src/weights/mod.rs`). The registry holds one `&'static` entry per packaging; `ModelArchConfig::weight_format` holds the configured format (`WeightFormatRef(Arc<dyn WeightFormat>)`, `.get()` borrows it).

| Method                                                                              | Must do                                                                                                                                                                                                                                                                                                                                             |
| ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name()` (from `Module`)                                                            | Unique packaging name (`bf16`, `ct_fp8`, `awq`, …).                                                                                                                                                                                                                                                                                                 |
| `check_config(&Value)`                                                              | `Ok` when the top-level `config.json` declares this packaging (`quantization_config.quant_method`, compressed-tensors `format`, `torch_dtype` / `dtype`); else the refusal naming the key, via `unsupported(key, value, supported)`. Only claims the container.                                                                                     |
| `configure(&Value)`                                                                 | The format as that `config.json` configures it (an `Arc<dyn WeightFormat>`; a format without parameters returns a fresh value of itself). Refuses a declared variant it does not serve with `quant_scheme_unsupported` naming the field.                                                                                                            |
| `describe()` (default: the name)                                                    | The configured format in words; two configured formats are equal when their descriptions are.                                                                                                                                                                                                                                                       |
| `check_tensor(&TensorEntry)`                                                        | `Ok` when a checkpoint tensor is stored as this packaging stores it (dtype, and block or group alignment); else `ModelError::Unsupported` naming the tensor.                                                                                                                                                                                        |
| `column()`                                                                          | The support-matrix `weight_format` column (`WeightFormatColumn` in `crates/turbine-core/src/support.rs`) the server's key uses.                                                                                                                                                                                                                     |
| `scheme(&LinearSlot)` (default `Bf16`)                                              | The `QuantScheme` of one linear layer (`cfg.linear_slots()`: every 2-D slot except the token embedding); layers the checkpoint leaves unquantized (its ignore list, and always `lm_head`) report `QuantScheme::Bf16`.                                                                                                                               |
| `activation()` (default `None`)                                                     | The `ActivationQuant` applied before the quantized layers (static or dynamic FP8, MXFP4 emulation).                                                                                                                                                                                                                                                 |
| `slots(&WeightSlot)` (default: itself)                                              | The checkpoint tensors the loader reads for a family slot: a quantized layer's data plus `P_scale`, `P_zeros` or `P_input_scale` slots stacked like the base slot (`P` = the base slot's stacked name).                                                                                                                                             |
| `slot_dtype(&WeightSlot)`                                                           | The device dtype a loaded slot is stored in (`DType::F8E4M3`, `DType::U8` for packed bytes, `F32` scales, `BF16` otherwise).                                                                                                                                                                                                                        |
| `repacks(&WeightSlot)` (default `false`)                                            | Whether the loader reads the slot's checkpoint tensor whole and passes it through `repack` (its checkpoint shape and dtype are then the format's to check).                                                                                                                                                                                         |
| `repack(&WeightSlot, &TensorEntry, bytes)`                                          | Rewrites a repacked slot's checkpoint bytes into exactly the slot's bytes in `slot_dtype` (`crates/turbine-kernels/src/quant.rs` documents the layouts): e.g. scales to F32, a per-tensor scale repeated per row, AWQ's nibble order to the plain one.                                                                                              |
| `companions(&WeightSlot)` / `repack_with(…, companions)` (default: none / `repack`) | Other checkpoint tensors a repack reads, passed whole in order: a block-scaled FP8 weight decoded to BF16 at load reads its block scales.                                                                                                                                                                                                           |
| `for_kernels(slots, supports)` (default `None`)                                     | The format as the selected providers serve one device's `slots` (`weights::resolve_for_providers`, called by the server before the requirements and the load): a format with a load-time fallback moves the layers `supports` refuses there — block-scaled FP8 decoded to BF16, a fused stack's parts together, logged `event="fp8_block_decoded"`. |
| `weight_bytes(&ModelArchConfig)`                                                    | Device bytes of every parameter (the memory budget's weight term). The default sums every slot of `slots` at its shape and `slot_dtype`, which is exactly what the loader uploads; override only when the packaging stores something else on device.                                                                                                |
| `write_tiny(dir, twin)` (default `Ok(false)`)                                       | Test support: rewrites a family's BF16 tiny checkpoint in `dir` into the packaging as configured (tensors and `quantization_config`) and writes into `twin` the same model in BF16 with each quantized weight's exact dequantized values.                                                                                                           |

## Files to add

1. `crates/turbine-model/src/weights/<name>.rs`. A packaging of an existing layout only parses and writes its `quantization_config`: the two FP8 packagings share `crates/turbine-model/src/weights/fp8.rs` (`Fp8Format<P>` implements the trait over an `Fp8Layout`), and each is an `Fp8Packaging` (see `crates/turbine-model/src/weights/ct_fp8.rs`):

```rust
pub struct CtFp8;

/// The registry entry (its default layout feeds the conformance suite's fixtures).
pub static CT_FP8: Fp8Format<CtFp8> = Fp8Format::ENTRY;

impl Fp8Packaging for CtFp8 {
    const NAME: &'static str = "ct_fp8";
    const DEFAULT: Fp8Layout = Fp8Layout { /* channel scales, per-token activations */ };
    fn claims(q: &Value) -> Result<(), ModelError> { /* quant_method + format */ }
    fn parse(q: &Value) -> Result<Fp8Layout, ModelError> { /* refuse other variants */ }
    fn to_json(layout: &Fp8Layout) -> Value { /* the quantization_config (fixtures) */ }
}
```

The three INT4 packagings share `crates/turbine-model/src/weights/int4.rs` and the three MXFP4 ones `crates/turbine-model/src/weights/mxfp4.rs` the same way (`Int4Format<P>` over an `Int4Packaging`; the container's tensor layouts and repacks live there, selected by `Int4Kind`), with the helpers both layouts use in `crates/turbine-model/src/weights/common.rs`. A new layout implements `WeightFormat` directly, like `crates/turbine-model/src/weights/bf16.rs`, including its `write_tiny` (Rust only, no Python).

2. Kernel support: a quantized layer runs through `quantize_act` and the quantized GEMM (`OpKind::QGemm`, kernel ABI v2.9, `crates/turbine-model/src/executor/decoder/linear.rs`), whose CPU reference (`crates/turbine-kernels/src/cpu/quant.rs`) dequantizes every scheme; each GPU provider registers implementations per scheme, or the kernel registry refuses the model at startup.
3. A support-matrix column value and rows (`crates/turbine-core/src/support.rs`): each Phase 6a column is refused naming `phase-6a-quantization` until its proof checkpoint passes the umbrella S-3 gate.

## Registry entry

In `crates/turbine-model/src/weights/mod.rs`: `pub mod <name>;` and the entry in `WEIGHT_FORMATS` **in detection order** — `detect` configures the first format that accepts `config.json`, and on no match reports the first format's refusal, so keep `bf16` first. Update the pinned list in `crates/turbine-model/src/registries.rs` (`registry_conformance::weight_formats`).

## Conformance suite

`weights_suite` (`crates/turbine-model/src/conformance/weights.rs`): `tiny` (the entry's `write_tiny` rewrites at least one registered family's tiny checkpoint into the format; each is detected as the format and loads through `WeightLoader::load_format` with every tensor accepted and mapped to a slot) and `bytes` (for each of those checkpoints the uploaded bytes equal `weight_bytes`, packed data and scales included).

- `scripts/remote-cargo.sh test -p turbine-model registry_conformance`
- `scripts/remote-cargo.sh test -p turbine-model weights::` — hand-built repack cases (`weights::int4::tests::{awq_repack_8x8, gptq_repack_8x8, ct_pack_repack_8x8}`), refusal messages, detection of every packaging variant (`detect_every_packaging`, fixtures from `testing::tiny::write_tiny_quantized`) and the BF16 description (`bf16_describes_linear_layers`).
- `scripts/remote-cargo.sh test -p turbine-model --test tiny_model quantized_matches_dequantized_bf16` — each variant on the CPU provider against the naive model over its dequantized twin, fused and unfused projections.

## Lab checks

`scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops` for the scheme's quantized GEMM, a golden reference of a real checkpoint in the packaging (`tests/golden/<slug>/`, made with `scripts/golden/quant_reference.py`) compared at `--concurrency 1` and `--concurrency 16`, `scripts/lab-bench.sh --model <proof model>` against the Phase 6a targets, and `scripts/lab-bench.sh --gpu 0 --model llama` / `--model olmoe` unchanged within the no-regression bound (the BF16 models must not move).

## Pitfalls

- `check_weight_format` in `crates/turbine-model/src/executor/decoder/mod.rs` refuses layers of different quantized schemes, a fused stack (Q/K/V, gate/up) whose parts differ in scheme, block-scaled parts off block boundaries, a quantized `lm_head`, quantized mixture-of-experts models (`quant_moe_phase7`) and tensor, expert or pipeline parallelism (until Phase 6a Task 21).
- Fused stacks concatenate the parts' packed rows and scales: per-tensor scales are stored per row so parts with different scales stay exact, and a static activation scale is the largest of the parts'.
- `n` or `k` not a multiple of the group or block size is refused (`quant_scheme_unsupported`), as is act-order GPTQ (`gptq_act_order`).
- A layer's quantization is decided from `config.json` (the ignore list), not from the tensor dtype: a checkpoint that leaves a layer in BF16 without listing it is refused by `check_tensor`.
- Do not add `DType::BF16` literals outside `crates/turbine-model/src/weights/bf16.rs` and tests: take the activation dtype from `weights::ACTIVATION_DTYPE` / `ModelArchConfig::activation_dtype`.
