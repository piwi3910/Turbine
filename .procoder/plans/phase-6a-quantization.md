# phase-6a-quantization — implementation plan

Status: draft
Spec: .procoder/specs/phase-6a-quantization.md

The user answered questions 1–20 of `.procoder/ask/decisions.md`, entry "Phase 6 spec: provisional design choices (2026-09-28)", and split Phase 6 in two (entry "Phase 6 split: 6a quantization, 6b KV compression (2026-09-28)"). This plan builds 6a: foundations (Tasks 1–4), weights (5–21), FP8 KV (22–25), YaRN (26–28), phase exit (29). `phase-6b-kv-compression` (per-tier formats, TurboQuant, the ladder) has its own plan and starts after Task 29.

## Goal

Serve FP8 (per-tensor/channel and block-scaled), INT4 AWQ/GPTQ and MXFP4 (compressed-tensors, Quark W4A4 emulated, OpenAI native loader) checkpoints of registered architectures on the R9700 (`gfx1201`), add an FP8 e4m3 L0 KV cache and static YaRN RoPE scaling for every family — each format proven against a reference, gated for quality, measured on GPU 0, and entered into the support matrix only when its gate passes — after porting the umbrella's quality-gate tooling and cleaning up the support matrix.

## Architecture

Weights: `turbine_model::weights::WeightFormat` becomes a description of a quantized linear layer (`QuantScheme`, `ActivationQuant`, extra tensor slots, a repack step), with one registry entry per checkpoint packaging; the loader builds `QuantLinear` values that the decoder hands to a new quantized GEMM op. `turbine-kernels` gains `OpKind::QGemm` / `OpKind::QuantizeAct` with a CPU reference (`cpu::quant`) and the optional kernel ABI group v2.9 (`turbine_qgemm*`, `turbine_quantize_act*`, dtype codes F8E4M3 / U8); the HIP shim registers implementations chosen by recorded provider evaluations (hipBLASLt FP8 first for `fp8`, CK `gemm_quant` for block-scaled and INT4, CK microscale / llama.cpp for MXFP4). KV: `kv.dtype: fp8_e4m3` switches the L0 layout to one byte per element with per-layer scales and an FP8-reading paged attention. YaRN is a `RopeScaling::Yarn` variant computed on the host into the existing `inv_freq` table with the attention factor folded into the attention scale; the resolved RoPE parameters and the FP8 KV scales enter the namespace key.

## Constraints

Copied verbatim from the spec (Constraints):

- Test tiers (AGENTS.md "Test tiers", decision 2026-09-27): per landing step `scripts/gate.sh`, plus `scripts/lab-test.sh novanas --tier quick` when GPU-facing code changes (`turbine-kernels`, `turbine-model`, `turbine-device`, the ABI, `kernels/rocm`), plus `scripts/lab-bench.sh --quick` when throughput or numerics can change; phase exit: `scripts/gate.sh --full`, `scripts/lab-test.sh novanas --tier full` and its two-GPU leg, `scripts/lab-bench.sh --golden16` for Llama and OLMoE BF16 plus every proof checkpoint, and `scripts/overload-soak.sh novanas --duration 10m` (asked first).
- Reuse first (AGENTS.md rule, decision "Kernel reuse policy"): every kernel of S-7 … S-10 and S-13 is preceded by a recorded evaluation in `.procoder/ask/decisions.md` of CK (`ck_tile` `gemm_quant`, microscale, FMHA FP8), hipBLASLt (FP8, scaled), llama.cpp HIP (q4/q8 and MXFP4 mat-vec and mmq, flash attention with quantized KV), vLLM / SGLang / aiter ROCm kernels and the compressed-tensors / Quark unpack paths; an own kernel only when none builds, none is correct on `gfx1201` or all are measurably slower. No Python in the build or runtime path; third-party kernel sources are pinned (CK by its existing `FetchContent` commit; llama.cpp by commit if used) and their licenses land in `kernels/rocm/third_party/LICENSES/`.
- Correctness bar: every GPU implementation has a lab test against the S-5 CPU reference; every quantized checkpoint has a golden reference; the BF16 golden tolerances, the OLMoE calibration and the Phase 4 `kv_gpu` bit-exact checks at `kv.dtype: bf16` and at the `l0` tier format do not change; lossy blocks are never served to an opted-out request.
- Nothing lossy by default: `kv.dtype` defaults to the exact behaviour; a quantized checkpoint is lossy only relative to BF16, and it is served as stored.
- Pluggability: every packaging, weight format and kernel implementation is one file (or directory) plus a registry entry; `docs/extending/weight-format.md` is rewritten for S-3; `cargo test -p turbine-model --test docs_extending` keeps them true.
- Vendor neutrality (umbrella S-5): no HIP type in a public signature of `turbine-kernels`, `turbine-tensor`, `turbine-scheduler`, `turbine-kv` or `turbine-reliability`; every ABI addition is an optional minor group specified so a CUDA library can implement it when `phase-2b-nvidia` is re-specced.
- Unsafe Rust and FFI stay in `turbine-kernels` (and the existing allowlist); `turbine-kv` stays GPU-free.
- Host copies go through pinned memory (L1 slots, the per-shard pinned bounce buffer); no pageable async copy is added (ROCm pageable-copy bug).
- Bounded everything (TS §21 rule 8): load-time staging (the existing 256 MiB staging buffer), refusal reasons from closed sets, metric labels from registered values.
- Lab: `novanas` only; perf numbers on GPU 0 only, through `scripts/lab-bench.sh` holding `scripts/bench-lock.sh`; GPU 1 for functional tests; two-GPU runs through `scripts/lab-cluster.sh --bench-lock`; every GPU test has a hard timeout; a busy GPU or lock is identified (`kubectl -n turbine-ci get pods`, `pgrep -fa`) before waiting; a card held by another workload stops the run and goes to the coordinator; weights only into `/home/piwi/turbine-models/<slug>`, only the checkpoints named in S-11, with the free disk checked first (≥ 60 GB free) and the Hugging Face token never read, printed or passed.
- Git: one commit per plan task, gate-clean; no push; nothing filed outside the repository.

From the interface contract and the work in flight (binding):

- Names in `.procoder/contract/interfaces.md` §3.8 (support matrix), §9 (kernel C ABI), §10 (`turbine-model`), §11 (`turbine-kv`), §17 (metrics) and §24 (registries) are used verbatim; this phase's additions go into a new contract section §26 "Phase 6 additions" in Task 1's commit (6b appends to it), and §9.1's planned "≥ 6" major is replaced by the optional minor v2.9 (the umbrella constraint: `TURBINE_ABI_VERSION` stays `2u`).
- Toolchain edition 2024, `rust-version = "1.97"`; `#[non_exhaustive]` on enums later phases extend (`QuantScheme`, `ActivationQuant`, `WeightFormatColumn`, `KvDtypeChoice`); config structs `#[serde(deny_unknown_fields, default)]`; metric labels from closed enums rendered with `as_str()`.
- Starting point: branch `phase-6a-quantization` (created as `phase-6-quantization`, renamed at the split) from `main` at `44e9a0a` (Phase 5 merged, Phase 5p docs only). BF16 baseline on GPU 0: Llama 855.2 tok/s (ITL p50 15.4 ms, TTFT p50 208 ms), OLMoE 613.9 (24.3, 118) — Task 4 re-measures it as the phase-start baseline.
- Builds and tests run on novanas through `scripts/remote-cargo.sh` (no local target directories); every task ends with `scripts/gate.sh` printing `gate: ok`; GPU-facing tasks add `scripts/lab-test.sh novanas --tier quick`; every task that changes serving code ends with `scripts/lab-bench.sh --quick --model llama` and `--model olmoe` (GPU 0, under `scripts/bench-lock.sh`, `LABBOOK_SET=phase-6a-quantization`) and a row in `.procoder/perf-log.md` — one change, then measure.
- Lab runs fall under the standing novanas approvals (2026-09-25, 2026-09-26) for `lab-test.sh`, `lab-serve.sh`, `lab-bench.sh`, golden and bench runs while the R9700s are free; downloads only of the checkpoints named in spec S-11 (approved by the user 2026-09-28, ≈ 41 GB, ≥ 60 GB kept free); anything else, and every soak, is asked first. Sub-agents used for file-disjoint work run in worktrees and receive these lab rules verbatim.

## Task 1: Port the umbrella tasks from `runahead/p8-umbrella`

Files: `crates/turbine-core/src/config/mod.rs` and its `quality` / `speculative` sections (umbrella Task 1), `benches/turbine-bench/src/golden/eval.rs`, `benches/turbine-bench/src/golden/mod.rs`, `benches/turbine-bench/src/bin/turbine-golden.rs`, `benches/turbine-bench/Cargo.toml`, `benches/turbine-bench/tests/golden.rs` (umbrella Task 5), `scripts/eval/make-gsm8k-200.sh`, `tests/eval/gsm8k-200.jsonl`, `tests/eval/NOTICE` (umbrella Task 6), `scripts/track-gate.sh`, `benches/turbine-bench/tests/lab_scripts.rs` (umbrella Task 8, amended gate), `.procoder/plans/phase-6-8-expansion.md` (state line), `.procoder/contract/interfaces.md` (new §26 skeleton)
Interfaces:

- ported unchanged in shape: `QualityConfig { max_accuracy_drop: f64 }` (0 ≤ value ≤ 0.1, default 0.01), `SpeculativeConfig { method: SpeculativeMethod::{None, Draft} }`; `turbine-golden eval` / `eval-compare` (umbrella spec Interfaces); `tests/eval/gsm8k-200.jsonl` (sha256 `6b7bab085cd8d5a484e59283d809525bf5e3e21edc1d3275e4b9517316775321`); `scripts/track-gate.sh <track>`
  Covers: spec S-1 (port); AC `eval_accuracy_report`, `eval_task_set_valid`, `track_gate`, `config::tests`
  Depends on: nothing in this phase

- [ ] Red: `scripts/remote-cargo.sh test -p turbine-bench --test golden eval_accuracy_report` — expect FAIL (`unrecognized subcommand 'eval'`).
- [ ] Cherry-pick in order onto `phase-6a-quantization`: `5794d4d` (config keys), `6bb5157` (eval), `844b229` (GSM8K set), `bde359f` (track gate); resolve conflicts against the Phase 2m / 5 tree by keeping main's structure and the commits' additions; apply the amended gate of the umbrella plan Task 8 with the track names `phase-6a-quantization`, `phase-6b-kv-compression`, `phase-7-model-families`, `phase-8-speculative-decoding` (6a needs no earlier track; 6b needs an `amd` row with a quantized weight column or `fp8_e4m3` KV `supported`; 7 needs a `supported` amd row with `tq4` or `tq2` KV from 6b; 8 needs a `supported` amd row for a Phase 7 family; NVIDIA rows never count; the script is renamed `scripts/track-gate.sh`).
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-bench --test golden` and `scripts/remote-cargo.sh test -p turbine-bench --test lab_scripts track_gate` and `scripts/remote-cargo.sh test -p turbine-core config::tests` — expect PASS; `shasum -a 256 tests/eval/gsm8k-200.jsonl` — expect the sha above.
- [ ] Run: `scripts/track-gate.sh phase-6a-quantization` — expect `GATE PASS phase-6a-quantization`.
- [ ] Update the umbrella plan's state line (Tasks 1, 5, 6, 8 ported on `phase-6a-quantization`) and add the contract §26 heading listing this phase's additions (filled by later tasks).
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `chore: port the phase 6-8 umbrella tasks (quality keys, eval gate, GSM8K-200, track gate)` (one commit; the cherry-picks are squashed so the branch keeps one commit per task)

## Task 2: Support-matrix cleanup and the new weight-format columns

Files: `crates/turbine-core/src/support.rs` (reasons, NVIDIA rows, `WeightFormatColumn` values, `SupportKey::for_model`), `crates/turbine-server/src/support_startup.rs` (keys from the detected format and `kv.dtype`; the `phase-8c` assert), `crates/turbine-server/tests/server_cli.rs` (`unsupported_row_exits_2_before_bind` reason), `crates/turbine-server/tests/tiny_server.rs`, `crates/turbine-model/src/config.rs`, `crates/turbine-model/src/families/mod.rs` (only the `phase-8*` strings; the `GptOssForCausalLM` unregistered-architecture examples stay until Phase 7), `AGENTS.md` (support-matrix sentence)
Interfaces:

- `const QUANT_REASON: &str = "quantized format not validated yet (track phase-6-quantization)"`, `FAMILY_REASON` naming `phase-7-model-families`, the draft row naming `phase-8-speculative-decoding`, `NVIDIA_REASON = "NVIDIA execution is deferred (phase-2b-nvidia)"`
- `WeightFormatColumn::{Fp8, Fp8Block, Mxfp4, Mxfp4A4, AwqInt4, GptqInt4}` with serde names `fp8`, `fp8_block`, `mxfp4`, `mxfp4_a4`, `awq_int4`, `gptq_int4`, each with a `*/*/*/<value>/*/none` unsupported row
- `SupportKey::for_model(vendor: &str, arch: Option<&str>, architecture: &str, weight: WeightFormatColumn, kv: KvFormatColumn, spec: SpeculativeColumn) -> SupportKey`; `SupportKey::bf16` becomes a thin wrapper used only by tests
  Covers: spec S-1; AC `baseline_rows_present`, `key_uses_detected_format`
  Depends on: Task 1

- [ ] Write failing test `support::tests::baseline_rows_present` (extend): no row reason contains `phase-8`; every `nvidia` row resolves unsupported naming `phase-2b-nvidia`; the six new columns resolve unsupported naming `phase-6a-quantization`; `validate_table` passes. Run: `scripts/remote-cargo.sh test -p turbine-core support::tests` — expect FAIL
- [ ] Write failing test `support_startup::tests::key_uses_detected_format` in `crates/turbine-server/src/support_startup.rs`: for a config with `kv.dtype` unset the key's KV column is `bf16`; the weight column comes from a `WeightFormatColumn` argument (Task 6 wires detection; until then `Bf16`). Run: `scripts/remote-cargo.sh test -p turbine-server --bin turbine-server support_startup` — expect FAIL
- [ ] Implement the reasons, the NVIDIA rows, the columns, `for_model`; update the three tests that assert `phase-8c` to `phase-7-model-families`.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-core` and `scripts/remote-cargo.sh test -p turbine-server` — expect PASS; `grep -rn "phase-8[abc]" crates` — expect no output.
- [ ] Run: `scripts/remote-cargo.sh run -p turbine-server -- --support-matrix` — expect the NVIDIA rows `unsupported` and the six new columns listed.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `fix(core): support matrix names the phase 6-8 tracks, NVIDIA rows deferred, quantized weight columns`

## Task 3: `DType::F8E4M3` and `DType::U8`

Files: `crates/turbine-core/src/types.rs` (variants, `size_bytes`, `abi_code`), `kernels/include/turbine_kernels.h` (dtype codes 16, 17 as defines inside the reserved range, no new group yet), `crates/turbine-kernels/src/ffi.rs` (code mapping), `crates/turbine-kernels/src/cpu/mod.rs` (`is_float` stays false for both), `crates/turbine-tensor/src/dtype.rs` (re-export test)
Interfaces:

- `DType::F8E4M3` (`size_bytes` 1, `abi_code` 16), `DType::U8` (1, 17); `DType::is_packed_storage(&self) -> bool` (true for `U8`)
  Covers: spec S-2
  Depends on: Task 2

- [ ] Write failing test `types::tests::quant_dtypes`: sizes, ABI codes, `Display` names `f8e4m3` / `u8`, that neither is a float for the CPU GEMM and that `KvLayout { dtype: F8E4M3, .. }.bytes_per_token()` is half the BF16 value. Run: `scripts/remote-cargo.sh test -p turbine-core types::tests::quant_dtypes` — expect FAIL
- [ ] Implement; update every exhaustive `match` on `DType` (compile errors guide it) with an explicit refusal where the dtype is not supported yet.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-core -p turbine-kernels -p turbine-tensor` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(core): F8E4M3 and U8 dtypes (ABI codes 16 and 17)`

## Task 4: Phase-start baseline and lab-bench model entries

Files: `scripts/lab-bench.sh` (`--model` values of the spec's Interfaces, each mapping to a weights directory, a golden slug and a config — the BF16 `llama` / `olmoe` keep `scripts/lab/phase2c-novanas-<model>.yaml` so baselines stay comparable, each proof model uses `scripts/lab/phase6-novanas-<model>.yaml` written by the task that proves it; `--print-model <model>`; a missing config exits 2, a missing weights directory on novanas exits 1; `BENCH` fields `weight_format=` and `kv=` from the status document's support row; the labbook `config` parameter names the config file), `benches/turbine-bench/tests/lab_scripts.rs` (model map test), `.procoder/perf-log.md` (Phase 6a section)
Interfaces:

- `scripts/lab-bench.sh --model <llama|olmoe|llama-fp8|llama-fp8-tensor|llama-fp8-block|llama-awq|llama-gptq|llama8b-mxfp4|llama8b|llama-mxfp4-a4|llama-yarn16>`; `scripts/lab-bench.sh --print-model <model>` prints `<weights> <golden slug> <config>` and exits 0 without contacting a host; unknown values exit 2 listing the valid ones
  Covers: spec S-20 (baseline)
  Depends on: Task 1

- [ ] Write failing test `lab_scripts lab_bench_model_map`: for every value `--print-model` prints the expected triple (e.g. `llama-yarn16` → `llama-3.2-3b-instruct llama-3.2-3b-instruct-yarn16 scripts/lab/phase6-novanas-llama-yarn16.yaml`), no host is contacted, and an unknown value exits 2 listing the models. Run: `scripts/remote-cargo.sh test -p turbine-bench --test lab_scripts lab_bench_model_map` — expect FAIL
- [ ] Implement the map and the helper; `bash -n` and `shellcheck scripts/lab-bench.sh` clean.
- [ ] Lab (GPU 0, bench lock): `LABBOOK_SET=phase-6a-quantization scripts/lab-bench.sh --model llama --golden16` and `--model olmoe --golden16` — expect golden1 and golden16 PASS; record both `BENCH` lines as the phase-start baseline in `.procoder/perf-log.md`.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `chore(lab): phase 6a lab-bench models and the phase-start baseline`

## Task 5: `cpu::quant` — the dequantization and rounding reference

Files: `crates/turbine-kernels/src/cpu/quant.rs` (new), `crates/turbine-kernels/src/cpu/mod.rs` (`pub mod quant`)
Interfaces:

- `pub fn fp8_e4m3_round(x: f32) -> u8` (OCP e4m3fn: RNE, saturate to ±448, NaN → 0x7F), `pub fn fp8_e4m3_value(b: u8) -> f32`
- `pub fn mxfp4_quantize_group(x: &[f32; 32], rounding: Mxfp4Rounding) -> ([u8; 16], u8)` (E2M1 codes packed low nibble first, E8M0 exponent from the group max as OCP / Quark `even`), `pub fn e2m1_value(code: u8) -> f32`
- `pub fn dequantize(scheme: &QuantSchemeDesc, data: &[u8], scales: &[f32], zeros: Option<&[u8]>, n: usize, k: usize) -> Vec<f32>` (row-major `[n, k]`); `QuantSchemeDesc` mirrors the ABI scheme codes of the spec (kernel crate cannot see `turbine-model`)
- `pub fn quantize_dequantize_activations(x: &mut [f32], rows: usize, cols: usize, mode: ActQuantDesc) -> Vec<f32>` (returns per-row or per-group scales)
  Covers: spec S-5; AC `cpu::quant::tests`
  Depends on: Task 3

- [ ] Write failing tests `cpu::quant::tests::{fp8_rounding_table, e2m1_every_code, mxfp4_group_scale, int4_group_dequant, fp8_block_dequant, act_quant_modes}` with hand-computed expectations (including ties, subnormals, ±448 saturation, NaN, all 16 E2M1 codes, an all-zero group, a group whose max is a power of two). Run: `scripts/remote-cargo.sh test -p turbine-kernels cpu::quant` — expect FAIL
- [ ] Implement.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-kernels cpu::quant` — expect PASS
- [ ] Mutation check (do not commit): make `fp8_e4m3_round` round half away from zero — expect `fp8_rounding_table` to FAIL; revert.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(kernels): CPU reference for FP8, INT4 and MXFP4 (de)quantization`

## Task 6: `WeightFormat` describes quantized linear layers (BF16 ported)

Files: `crates/turbine-model/src/weights/mod.rs` (trait, `QuantScheme`, `ActivationQuant`, `LinearSlot`, detection returning the packaging), `crates/turbine-model/src/weights/bf16.rs` (ported), `crates/turbine-model/src/config.rs` (`ModelArchConfig.weight_format` reports `column()`), `crates/turbine-model/src/conformance/weights.rs` (suite for packed formats), `crates/turbine-model/tests/registries.rs`, `crates/turbine-server/src/support_startup.rs` (weight column from detection), `docs/extending/weight-format.md` (rewritten), `crates/turbine-model/tests/docs_extending.rs` (if its checks name trait methods)
Interfaces:

- the trait, enums and `LinearSlot` exactly as the spec's Interfaces; `QuantLinear { scheme: QuantScheme, data: DeviceTensor, scales: Option<DeviceTensor>, zeros: Option<DeviceTensor>, act_scale: Option<f32> }`
- `fn detect(top: &serde_json::Value) -> Result<&'static dyn WeightFormat, ModelError>` unchanged in signature; the BF16 entry's `column()` is `WeightFormatColumn::Bf16`, `scheme()` always `Bf16`, `activation()` `None`, `slots()` returns the base slot, `repack()` the identity
- `check_weight_format` in the decoder is replaced by a per-layer check that the kernel registry has a `QGemm` (or BF16 `Gemm`) implementation for each scheme
  Covers: spec S-3 (trait), S-1 (detected column in the key); AC `registry_conformance`, `docs_extending`, `key_uses_detected_format`
  Depends on: Tasks 2, 3

- [ ] Write failing test `weights::tests::bf16_describes_linear_layers`: the BF16 entry reports column `bf16`, scheme `Bf16` for every `LinearSlot` of the tiny Llama, `activation() == None`, and `weights_suite` passes with the rewritten `bytes` check (packed bytes + scales). Run: `scripts/remote-cargo.sh test -p turbine-model weights::tests` — expect FAIL
- [ ] Implement the trait change and port `bf16.rs`, the loader and the decoder to it; server key uses `column()`.
- [ ] Rewrite `docs/extending/weight-format.md` for packagings (files, registry entry, `write_tiny_quantized`, the suite command, the lab checks: `hip_ops qgemm_matches_cpu`, golden c1/c16 against the slug's reference, lab-bench within the targets).
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-model` and `scripts/remote-cargo.sh test -p turbine-server` — expect PASS (no behaviour change for BF16)
- [ ] Lab: `scripts/lab-test.sh novanas --tier quick` — expect PASS; `scripts/lab-bench.sh --quick --model llama` — within the no-regression bound; perf-log row.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `refactor(model): weight formats describe quantized linear layers`

## Task 7: Kernel ABI v2.9 and the `QGemm` / `QuantizeAct` ops (CPU provider)

Files: `kernels/include/turbine_kernels.h` (v2.9 group: descs, trios, scheme and activation codes, op codes 17/18, FP8 KV fields on the paged desc), `crates/turbine-kernels/src/ffi.rs` (`V29Symbols`, resolution), `crates/turbine-kernels/src/ops/mod.rs` (`OpKind::{QGemm, QuantizeAct}`, configs, contexts, traits, `KernelProvider` defaults), `crates/turbine-kernels/src/registry.rs` (`OpConfig::{QGemm, QuantizeAct}`, `KernelRegistry::{qgemm, quantize_act}`), `crates/turbine-kernels/src/cpu/qgemm.rs` (new: CPU implementation over `cpu::quant`), `crates/turbine-kernels/src/shim.rs` (implementations enumeration by minor), `kernels/rocm/src/abi_minor.cpp` (still reports 8 until Task 11), `.procoder/contract/interfaces.md` §9 / §26
Interfaces:

- `QGemmConfig { n: u32, k: u32, scheme: QuantSchemeDesc, act_quant: ActQuantDesc, c_dtype: DType, group_size: u32 }`; `QGemmContext { a: TensorView, b: QuantWeightView, c: TensorView, a_scales: Option<TensorView>, prefill: bool }`; `trait QGemmKernel { fn supports(&self, cfg: &QGemmConfig) -> bool; fn implementation(&self, cfg: &QGemmConfig) -> String; fn execute(&self, ctx: &mut QGemmContext) -> Result<(), KernelError>; }`; same trio shape for `QuantizeActKernel`
- `KernelProvider::qgemm(&self) -> Option<&dyn QGemmKernel>` and `quantize_act(&self) -> Option<&dyn QuantizeActKernel>`, default `None`; the CPU provider implements both as `cpu_qgemm_ref` / `cpu_quantize_act_ref`
- C: `turbine_qgemm_desc` and `turbine_quantize_act_desc` field lists as the spec's Interfaces; `TURBINE_ABI_MINOR` constant in the header becomes 9 (a library reports its own minor)
  Covers: spec S-6; AC `ffi::tests::optional_groups_v29`, `vendor_neutral_api`
  Depends on: Task 5

- [ ] Write failing test `ffi::tests::optional_groups_v29`: a stub symbol table with minor 8 resolves no qgemm; minor 9 with every symbol resolves both trios; minor 9 missing `turbine_quantize_act_impl` resolves neither of that trio. Run: `scripts/remote-cargo.sh test -p turbine-kernels ffi::tests` — expect FAIL
- [ ] Write failing test `cpu::qgemm::tests::matches_dequantized_gemm`: for every scheme and activation mode, random seeded inputs, CPU `qgemm` equals `gemm(dequantize(W), quantize_dequantize(A))` bitwise in F32. Run: `scripts/remote-cargo.sh test -p turbine-kernels cpu::qgemm` — expect FAIL
- [ ] Implement header, FFI resolution, Rust ops, registry lookups and the CPU implementation.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-kernels` — expect PASS (including `vendor_neutral_api`)
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(kernels): ABI v2.9 quantized GEMM and activation quantization ops, CPU reference provider`

## Task 8: FP8 packagings `ct_fp8` and `hf_fp8` (host path)

Files: `crates/turbine-model/src/weights/ct_fp8.rs` (new), `crates/turbine-model/src/weights/hf_fp8.rs` (new), `crates/turbine-model/src/weights/mod.rs` (registry order: `bf16`, `ct_fp8`, `hf_fp8`, …), `crates/turbine-model/src/loader.rs` (format-driven `load`, extra slots, stack rescale, packed byte accounting), `crates/turbine-model/src/weights/fp8.rs` (new: the shared FP8 layout, `Fp8Format<P>` over an `Fp8Packaging`, its tiny writer), `crates/turbine-model/src/testing/tiny.rs` (`write_tiny_quantized`: the tiny Llama through the format's `WeightFormat::write_tiny`, plus its dequantized BF16 twin), `crates/turbine-model/src/executor/decoder/linear.rs` (new: `Linear` / `LinearView`, `take_linear`, `linear_ops`) and `crates/turbine-model/src/executor/decoder/mod.rs` (linear layers call `qgemm` for non-BF16 schemes, `quantize_act` before them), `crates/turbine-model/src/conformance/weights.rs` (the suite writes each format's fixtures), `crates/turbine-model/tests/tiny_model.rs` (test), `crates/turbine-model/src/registries.rs`, `docs/extending/weight-format.md`
Interfaces:

- detection per spec S-3 (Q5): compressed-tensors `format: float-quantized`, `weights.num_bits: 8`, `type: float`, `strategy` ∈ `tensor|channel` → `fp8`, `block` with `block_structure: [128,128]` → `fp8_block`; `input_activations` null → `None`, `strategy: tensor, dynamic: false` → `Fp8PerTensorStatic`, `strategy: token, dynamic: true` → `Fp8PerTokenDynamic`, `strategy: group, group_size: 128, dynamic: true` → `Fp8PerGroupDynamic { group: 128 }`; `ignore` list honoured (regex `re:` prefixes as compressed-tensors defines them)
- tensor slots: `<layer>.weight` (F8_E4M3), `.weight_scale`, optional `.input_scale`; a stack whose parts have different scalar scales is rescaled to per-channel (F32 vector) at load
  Covers: spec S-3, S-4, S-5 for FP8; AC `detect_every_packaging` (FP8 part), `quantized_matches_dequantized_bf16` (FP8 part)
  Depends on: Tasks 6, 7

- [ ] Write failing test `weights::tests::detect_every_packaging` (FP8 cases): tiny checkpoints of the five FP8 variants detect to `ct_fp8` / `hf_fp8` with the right column, scheme and activation; `num_bits: 4 type float` and a `[64,64]` block are refused `quant_scheme_unsupported` naming the field. Run: `scripts/remote-cargo.sh test -p turbine-model weights::tests::detect_every_packaging` — expect FAIL
- [ ] Write failing test `tiny_model quantized_matches_dequantized_bf16` (FP8 cases): logits of each FP8 tiny checkpoint on the CPU provider are within 1e-3 of the dequantized BF16 twin with activation fake-quantization. Run: `scripts/remote-cargo.sh test -p turbine-model --test tiny_model quantized_matches_dequantized_bf16` — expect FAIL
- [ ] Implement.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-model` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(model): FP8 checkpoints (compressed-tensors and quant_method fp8) load and run on the CPU provider`

## Task 9: INT4 packagings `awq`, `gptq`, `ct_pack_int4` (host path)

Files: `crates/turbine-model/src/weights/awq.rs`, `crates/turbine-model/src/weights/gptq.rs`, `crates/turbine-model/src/weights/ct_pack_int4.rs` (new), `crates/turbine-model/src/weights/mod.rs` (registry), `crates/turbine-model/src/testing/tiny.rs` (writers: AWQ interleaved order `[0,2,4,6,1,3,5,7]`, GPTQ row-packed `qweight` along k, compressed-tensors `weight_packed` / `weight_shape`), `crates/turbine-model/tests/tiny_model.rs`, `crates/turbine-model/src/weights/tests.rs` (hand-built 8 × 8 packing cases)
Interfaces:

- repack target (the layout the v2.9 `INT4_GROUP_*` schemes consume): `data` U8 `[n, k/2]`, low nibble = even k, unsigned 0..15; `scales` F32 `[n, k/group]`; `zeros` U8 `[n, k/group]` (AWQ from `qzeros`; symmetric GPTQ and compressed-tensors → scheme `INT4_GROUP_SYM` with implicit 8)
- refusals: `desc_act: true` or a `g_idx` that is not `i / group` → `gptq_act_order`; `bits` ≠ 4 or `version` ≠ `gemm` → `quant_scheme_unsupported`
  Covers: spec S-3, S-4, S-5 for INT4; AC `detect_every_packaging` (INT4), `quantized_matches_dequantized_bf16` (INT4)
  Depends on: Task 8

- [ ] Write failing tests `weights::tests::{awq_repack_8x8, gptq_repack_8x8, ct_pack_repack_8x8}` from hand-built tensors whose dequantized values are known, and extend `detect_every_packaging` with the INT4 cases and the refusals. Run: `scripts/remote-cargo.sh test -p turbine-model weights::tests` — expect FAIL
- [ ] Extend `quantized_matches_dequantized_bf16` with the three INT4 packagings. Run — expect FAIL
- [ ] Implement the three packagings and their writers.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-model` — expect PASS
- [ ] Mutation check (do not commit): swap the AWQ order table for the identity — expect `awq_repack_8x8` to FAIL; revert.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(model): AWQ, GPTQ and compressed-tensors INT4 checkpoints on the CPU provider`

## Task 10: MXFP4 packagings `ct_mxfp4`, `quark_mxfp4`, `openai_mxfp4` (host path)

Files: `crates/turbine-model/src/weights/ct_mxfp4.rs`, `crates/turbine-model/src/weights/quark_mxfp4.rs`, `crates/turbine-model/src/weights/openai_mxfp4.rs` (new), `crates/turbine-model/src/weights/mod.rs` (registry), `crates/turbine-model/src/testing/tiny.rs` (writers for the three packagings, Quark weight-only and W4A4), `crates/turbine-model/tests/tiny_model.rs`
Interfaces:

- repack target (`MXFP4` scheme): `data` U8 `[n, k/2]` E2M1 codes low nibble first, `scales` U8 `[n, k/32]` E8M0
- column: `mxfp4` for compressed-tensors `mxfp4-pack-quantized` without input activations, Quark `fp4` weights without input quantization and OpenAI `quant_method: mxfp4`; `mxfp4_a4` for Quark `fp4` weights with `input_tensors.dtype: fp4` dynamic (activation `Mxfp4Emulated`); Quark `export.pack_method: reorder` is unpacked as Quark defines it
- OpenAI native: `*.weight_blocks` / `*_blocks` `[.., k/32, 16]` and `*_scales` read for dense-shaped tiny fixtures; `modules_to_not_convert` honoured; the family (`GptOssForCausalLM`) is not registered, so only the tiny Llama fixture exercises the loader in this phase
  Covers: spec S-3, S-4, S-5, S-10 (host part); AC `detect_every_packaging` (MXFP4), `quantized_matches_dequantized_bf16` (MXFP4)
  Depends on: Task 9

- [ ] Write failing tests: `detect_every_packaging` MXFP4 cases (including a group-64 compressed-tensors config refused `quant_scheme_unsupported`) and `quantized_matches_dequantized_bf16` for the four MXFP4 tiny variants (W4A4 against the reference with MXFP4 activation fake-quantization). Run: `scripts/remote-cargo.sh test -p turbine-model weights::tests::detect_every_packaging` — expect FAIL
- [ ] Implement.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-model` — expect PASS (all nine packagings now pass `detect_every_packaging` and `registry_conformance`)
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(model): MXFP4 checkpoints (compressed-tensors, Quark, OpenAI native) on the CPU provider`

## Task 11: Download the proof checkpoints and write the reference fixture scripts

Files: `scripts/golden/dequantize_checkpoint.py` (new, uv inline deps: safetensors, torch CPU, numpy), `scripts/golden/quant_reference.py` (new: wraps `hf_reference.py` on the dequantized copy, adds activation fake-quantization hooks), `scripts/golden/hf_reference.py` (optional `--config-override <json>` for YaRN, Task 28), `benches/turbine-bench/tests/golden.rs` (`quant_fixtures_valid`), `.procoder/ask/decisions.md` (download record: repo, revision, bytes, free disk before/after)
Interfaces:

- `uv run scripts/golden/dequantize_checkpoint.py --model-dir <dir> --out <bf16-dir>`: writes BF16 `model*.safetensors` with every quantized linear decoded exactly as `cpu::quant` does (the script's decode is checked against a Rust-written tiny fixture by a committed expected-output file), copies tokenizer files, removes `quantization_config`
- `uv run scripts/golden/quant_reference.py --model-dir <dir> --prompts tests/golden/prompts.jsonl --out <reference.jsonl> [--act-quant none|fp8_token|fp8_tensor|mxfp4] [--top-logprobs 20]`
  Covers: spec S-11 (downloads, fixture tooling)
  Depends on: Task 10

- [ ] On novanas: `df -h /home/piwi` — expect ≥ 60 GB free after the planned ≈ 41 GB (4.4 + 4.4 + 3.6 + 2.3 + 2.3 + 5.8 + 2.3 + 16.1); then for each checkpoint in spec S-11 `hf download <repo> --revision <sha> --local-dir /home/piwi/turbine-models/<slug>` (the token stays on the host; nothing is printed from it); `df -h` again; record each in decisions.md. If free disk would drop below 60 GB, stop and ask the coordinator.
- [ ] Write failing test `golden quant_fixtures_valid` over the slug directories listed in the spec's Data section (present ones only until each format's task commits its fixture; the list is completed by Task 21). Run: `scripts/remote-cargo.sh test -p turbine-bench --test golden quant_fixtures_valid` — expect FAIL (no directories)
- [ ] Write the two scripts; check `dequantize_checkpoint.py` against the Rust tiny fixtures of Tasks 8–10 (`--check-tiny <dir>` compares with `cpu::quant` output dumped by `scripts/remote-cargo.sh run -p turbine-model --example dump_dequant`).
- [ ] Commit the test with its list empty-tolerant (it asserts each listed directory that exists is complete) so it passes now; each format task adds its slug.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `test(golden): quantized checkpoint reference scripts and fixture validation`

## Task 12: FP8 GEMM — provider evaluation

Files: `.procoder/ask/decisions.md` (entry "P6: FP8 GEMM — provider evaluation (kernel reuse rule)"), `kernels/rocm/tools/qgemm_eval.cpp` (new throwaway-free evaluation harness kept as a tool target: builds each candidate for gfx1201, checks against a host reference, times the Llama-3.2-3B shapes), `kernels/rocm/CMakeLists.txt` (tool target, off by default)
Interfaces:

- candidates: hipBLASLt FP8 × FP8 → BF16 with scalar scales (`SAB`) and vector scales (`SABV`) for per-token × per-channel; CK `ck_tile` `gemm_quant` `RowColQuant` / `TensorQuant` for FP8 on gfx1201; vLLM/aiter ROCm scaled-mm (whether gfx12 code paths exist); activation quantization candidates: a CK `add_rmsnorm2d_rdquant` fused norm + quant, an own elementwise kernel
- recorded per candidate: builds (y/n), correct vs `cpu::quant` (max abs err), µs per shape at M ∈ {1, 16, 128, 2048}, and the choice
  Covers: spec S-7 (evaluation), reuse rule AC
  Depends on: Task 7

- [ ] Run the harness on novanas GPU 0 under `scripts/bench-lock.sh` through `scripts/lab-test.sh`-style Job or `scripts/remote-cargo.sh` + a native run with a 20-minute timeout; if GPU 0 is busy identify the holder first.
- [ ] Write the decisions entry with the table and the pick (expected: hipBLASLt `SABV` for per-token × per-channel, scalar for per-tensor; activation quant fused into RMSNorm if CK's `rdquant` works on gfx1201, else own elementwise kernel — recorded as an own kernel with its reason).
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `docs(decisions): FP8 GEMM provider evaluation on gfx1201`

## Task 13: `fp8` on HIP — `hipblaslt_fp8` and activation quantization

Files: `kernels/rocm/src/qgemm.cpp` (new: `hipblaslt_fp8` over the existing hipBLASLt handle and tuned-table machinery), `kernels/rocm/src/quantize_act.hip` (or the CK fused-norm wrapper, per Task 12), `kernels/rocm/src/impl_table.cpp` (entries), `kernels/rocm/src/impl_exports.cpp`, `kernels/rocm/src/abi_minor.cpp` (report 9), `kernels/rocm/tuning/gemm_shapes.txt` (FP8 shapes), `crates/turbine-kernels/tests/hip_ops.rs` (`qgemm_matches_cpu`, `quantize_act_matches_cpu`), `scripts/lab-test.sh` (`qgemm_matches_cpu` timing variant into `SLOW_TESTS` if it times)
Interfaces:

- implementations `hipblaslt_fp8` (schemes `FP8_TENSOR`, `FP8_CHANNEL`; act `FP8_TENSOR`, `FP8_TOKEN`; C BF16) and `quantize_act` implementation named after the Task 12 pick
  Covers: spec S-7; AC `hip_ops qgemm_matches_cpu` (FP8), `quantize_act_matches_cpu` (FP8)
  Depends on: Task 12

- [ ] Write failing lab tests `hip_ops::qgemm_matches_cpu` (FP8 schemes, M ∈ {1, 7, 16, 128, 513}, the 3B shapes) and `hip_ops::quantize_act_matches_cpu` (bit-exact). Run: `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops qgemm_matches_cpu` — expect FAIL (no implementation)
- [ ] Implement; the decoder picks `qgemm` for FP8 layers through the registry (Task 8 wiring).
- [ ] Run the same lab command and `-- -p turbine-kernels --test hip_ops quantize_act_matches_cpu` — expect PASS; `scripts/lab-test.sh novanas --tier quick` — expect PASS
- [ ] Lab: `scripts/lab-bench.sh --quick --model llama` and `--model olmoe` — BF16 unchanged within the bound; perf-log row.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(rocm): FP8 W8A8 GEMM on hipBLASLt with per-token and per-channel scales`

## Task 14: `fp8` proof on Llama-3.2-3B (FP8-dynamic and FP8 per-tensor)

Files: `tests/golden/llama-3.2-3b-instruct-fp8-dynamic/{reference.jsonl,tolerance.json,README.md}`, `tests/golden/llama-3.2-3b-instruct-fp8/{…}`, `scripts/lab/phase6-novanas-llama-fp8.yaml`, `scripts/lab/phase6-novanas-llama-fp8-tensor.yaml`, `tests/eval/llama-3.2-3b-instruct/turbine-bf16.json`, `tests/eval/llama-3.2-3b-instruct-fp8-dynamic/{turbine.json,vllm.json,gate.json}`, `crates/turbine-core/src/support.rs` (supported rows), `.procoder/perf-log.md`, `benches/turbine-bench/tests/golden.rs` (slug list)
Interfaces:

- support rows `amd/gfx1201/LlamaForCausalLM/fp8/bf16/none` → `supported` after all gate items pass
  Covers: spec S-11 (fp8), S-17; AC S-11 per-checkpoint criterion
  Depends on: Tasks 11, 13

- [ ] Fixture (novanas, CPU is enough): `uv run scripts/golden/quant_reference.py --model-dir /home/piwi/turbine-models/llama-3.2-3b-instruct-fp8-dynamic --prompts tests/golden/prompts.jsonl --out tests/golden/llama-3.2-3b-instruct-fp8-dynamic/reference.jsonl --act-quant fp8_token`; calibrate `tolerance.json` with `uv run scripts/golden/self_spread.py` on the dequantized copy; README with repo, revision, commands. Same for the per-tensor checkpoint with `--act-quant fp8_tensor`.
- [ ] Red: `scripts/remote-cargo.sh test -p turbine-bench --test golden quant_fixtures_valid` with the two slugs added — expect PASS for the fixtures; `scripts/remote-cargo.sh run -p turbine-server -- --support-matrix` shows the `fp8` row unsupported.
- [ ] Lab (GPU 0, bench lock): `scripts/lab-bench.sh --model llama-fp8 --golden16` and `--model llama-fp8-tensor --golden16` — expect golden1/golden16 PASS; record `BENCH` lines; compare with the targets (c16 ≥ 1.10 × BF16, c1 ITL ≤ 0.75 ×).
- [ ] Reference engine: `scripts/lab-serve.sh novanas --vllm llama-3.2-3b-instruct-fp8-dynamic` then the standard bench and `turbine-golden eval` against port 18100; stop with `--stop`. If vLLM does not load it on gfx1201, record that and use the BF16 baseline with `gate.json` max drop 0.02 (Q9).
- [ ] Eval: `turbine-golden eval` for BF16 Llama and the FP8 server, `turbine-golden eval-compare` per `gate.json` — expect exit 0.
- [ ] Soak (ask the coordinator first): `scripts/overload-soak.sh novanas --duration 10m --model /home/piwi/turbine-models/llama-3.2-3b-instruct-fp8-dynamic` — expect verdict pass.
- [ ] Flip the row to `supported`; `support::tests::baseline_rows_present` updated to expect it. Run: `scripts/remote-cargo.sh test -p turbine-core support` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(core): fp8 weights supported on gfx1201 Llama (golden, bench, eval, soak recorded)`

## Task 15: `fp8_block` — evaluation, implementation, proof

Files: `.procoder/ask/decisions.md` (entry "P6: block-scaled FP8 GEMM — provider evaluation"), `kernels/rocm/src/qgemm_ck_block.cpp` (CK `ABQuantGrouped` instance for gfx1201, or the W8A16 dequant path per Q3), `kernels/rocm/CMakeLists.txt` (CK instance generation filter), `kernels/rocm/src/impl_table.cpp`, `crates/turbine-kernels/tests/hip_ops.rs` (FP8_BLOCK cases), `tests/golden/llama-3.2-3b-instruct-fp8-block/{…}`, `scripts/lab/phase6-novanas-llama-fp8-block.yaml`, `tests/eval/llama-3.2-3b-instruct-fp8-block/{…}`, `crates/turbine-core/src/support.rs`, `.procoder/perf-log.md`
Interfaces:

- implementation `ck_tile_abquant_fp8` (scheme `FP8_BLOCK`, act `FP8_GROUP128`) or, per Q3 fallback, `dequant_fp8_block_bf16` + BF16 GEMM (act `NONE`, W8A16)
  Covers: spec S-8, S-11 (fp8_block), S-17
  Depends on: Task 14

- [ ] Evaluate (GPU 0, bench lock, 20-minute timeout): CK `ABQuantGrouped` built from the pinned CK for gfx1201 (WMMA policy), correctness vs `cpu::quant`, timings; the dequant fallback candidates if it fails; write the decisions entry.
- [ ] Write failing lab test cases `qgemm_matches_cpu` for `FP8_BLOCK`. Run: `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops qgemm_matches_cpu` — expect FAIL
- [ ] Implement the chosen path; run the lab test — expect PASS; `scripts/lab-test.sh novanas --tier quick` — expect PASS
- [ ] Fixture and proof exactly as Task 14's steps with `--act-quant` per the chosen arithmetic (`fp8_group128` if W8A8, `none` if W8A16), slug `llama-3.2-3b-instruct-fp8-block`, model `llama-fp8-block`, target c16 ≥ 1.0 × BF16; flip the row only when golden, bench, eval and soak pass, else leave it `experimental` and record the finding.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(rocm): block-scaled FP8 GEMM and the fp8_block proof`

## Task 16: INT4 GEMM — provider evaluation

Files: `.procoder/ask/decisions.md` (entry "P6: INT4 group GEMM — provider evaluation"), `kernels/rocm/tools/qgemm_eval.cpp` (INT4 candidates), `kernels/rocm/cmake/fetch_llamacpp.cmake` (only if llama.cpp is a candidate that builds: pinned commit, sparse checkout of `ggml/src/ggml-cuda` HIP sources, license into `third_party/LICENSES/`)
Interfaces:

- candidates in order: CK `ck_tile` `gemm_quant` `BQuantGrouped` with `pk_int4` B (WMMA policy, gfx1201), llama.cpp HIP `mul_mat_vec_q` (decode) and `mmq` (prefill) for a Q4-group layout we can repack to, vLLM/aiter AWQ/GPTQ ROCm kernels (gfx12 code paths), own dequant-GEMV + dequant-tile BF16 WMMA
- recorded: builds, correctness vs `cpu::quant` for AWQ (zero points) and GPTQ (symmetric), µs at M ∈ {1, 4, 16, 128, 2048} on the 3B shapes, and the choice per M range
  Covers: spec S-9 (evaluation)
  Depends on: Task 10

- [ ] Run the harness (GPU 0, bench lock, 30-minute timeout); write the entry.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `docs(decisions): INT4 group GEMM provider evaluation on gfx1201`

## Task 17: INT4 on HIP

Files: `kernels/rocm/src/qgemm_int4.*` (the picked implementations: decode and prefill), `kernels/rocm/src/impl_table.cpp`, `kernels/rocm/CMakeLists.txt`, `crates/turbine-kernels/tests/hip_ops.rs` (INT4 cases)
Interfaces:

- implementations registered for `INT4_GROUP_ZP` and `INT4_GROUP_SYM`, group 128, C BF16, act `NONE`; the registry picks per M through the card profile's row thresholds (the Phase 2m `implementation_supports(spec, idx, rows)` mechanism)
  Covers: spec S-9; AC `qgemm_matches_cpu` (INT4)
  Depends on: Task 16

- [ ] Write failing lab cases `qgemm_matches_cpu` for both INT4 schemes. Run: `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops qgemm_matches_cpu` — expect FAIL
- [ ] Implement; run — expect PASS; `scripts/lab-test.sh novanas --tier quick` — expect PASS
- [ ] Lab: `scripts/lab-bench.sh --quick --model llama` — BF16 unchanged; perf-log row.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(rocm): INT4 group-quantized GEMM (AWQ and GPTQ layouts)`

## Task 18: `awq_int4` and `gptq_int4` proofs on Llama-3.2-3B

Files: `tests/golden/llama-3.2-3b-instruct-awq/{…}`, `tests/golden/llama-3.2-3b-instruct-gptq/{…}`, `scripts/lab/phase6-novanas-llama-awq.yaml`, `scripts/lab/phase6-novanas-llama-gptq.yaml`, `tests/eval/llama-3.2-3b-instruct-awq/{…}`, `tests/eval/llama-3.2-3b-instruct-gptq/{…}`, `crates/turbine-core/src/support.rs`, `.procoder/perf-log.md`, `benches/turbine-bench/tests/golden.rs`
Interfaces:

- rows `amd/gfx1201/LlamaForCausalLM/{awq_int4,gptq_int4}/bf16/none` → `supported` after their gates
  Covers: spec S-11 (INT4), S-17
  Depends on: Tasks 11, 17

- [ ] Fixtures: `quant_reference.py … --act-quant none` for both checkpoints; tolerance from `self_spread.py` on the dequantized copies; READMEs.
- [ ] Lab (GPU 0, bench lock): `scripts/lab-bench.sh --model llama-awq --golden16`, `--model llama-gptq --golden16` — expect PASS; targets c1 ITL ≤ 0.6 × BF16 and c16 ≥ 0.9 ×; vLLM-ROCm on each checkpoint where it loads (else BF16 with `gate.json` max drop 0.04, Q9); eval-compare exit 0; soak on the AWQ checkpoint (asked first).
- [ ] Flip the rows that passed; `support::tests::baseline_rows_present` updated. Run: `scripts/remote-cargo.sh test -p turbine-core support` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(core): awq_int4 and gptq_int4 on gfx1201 Llama (gates recorded)`

## Task 19: MXFP4 GEMM — evaluation and implementation (with W4A4 emulation)

Files: `.procoder/ask/decisions.md` (entry "P6: MXFP4 GEMM — provider evaluation"), `kernels/rocm/src/qgemm_mxfp4.*`, `kernels/rocm/src/quantize_act.hip` (MXFP4 quantize-dequantize mode), `kernels/rocm/src/impl_table.cpp`, `crates/turbine-kernels/tests/hip_ops.rs` (MXFP4 and `MXFP4_EMULATED` cases)
Interfaces:

- candidates: CK `gemm_quant` microscale pipeline (`gemm_microscale_*`, gfx12 policy), llama.cpp HIP `GGML_TYPE_MXFP4` mat-vec / mmq (repack-compatible layout), the INT4 path of Task 17 generalised to an E2M1 × E8M0 decode, own; the pick recorded before code
- implementation registered for scheme `MXFP4`, act `NONE` and `MXFP4_EMULATED`
  Covers: spec S-10; AC `qgemm_matches_cpu` (MXFP4), `quantize_act_matches_cpu` (MXFP4)
  Depends on: Task 17

- [ ] Evaluate (GPU 0, bench lock, 30-minute timeout) on the 3B and 8B shapes; write the entry.
- [ ] Write failing lab cases; run `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops qgemm_matches_cpu` — expect FAIL
- [ ] Implement; run — expect PASS (MXFP4 activation emulation bit-exact against `cpu::quant`); `scripts/lab-test.sh novanas --tier quick` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(rocm): MXFP4 weight-only GEMM and MXFP4 activation emulation`

## Task 20: MXFP4 proofs (compressed-tensors 8B, Quark W4A4 3B)

Files: `tests/golden/llama-3.1-8b-instruct/{…}` (BF16 baseline fixture from `hf_reference.py`), `tests/golden/llama-3.1-8b-instruct-mxfp4a16/{…}`, `tests/golden/llama-3.2-3b-mxfp4-a4/{…}`, `scripts/lab/phase6-novanas-llama8b.yaml`, `scripts/lab/phase6-novanas-llama8b-mxfp4.yaml`, `scripts/lab/phase6-novanas-llama-mxfp4-a4.yaml`, `tests/eval/<slug>/{…}` for the three, `crates/turbine-core/src/support.rs`, `.procoder/perf-log.md`, `benches/turbine-bench/tests/golden.rs`
Interfaces:

- rows `amd/gfx1201/LlamaForCausalLM/mxfp4/bf16/none` → `supported` after its gate; `…/mxfp4_a4/…` → `supported` if its gate passes, else `experimental` (decision "Phase 6 MXFP4")
  Covers: spec S-10, S-11 (MXFP4), S-17
  Depends on: Tasks 11, 19

- [ ] Fixtures: BF16 8B via `hf_reference.py` (CPU, may take hours — run as a background Job with a 6-hour timeout), MXFP4 8B via `quant_reference.py --act-quant none`, Quark 3B via `--act-quant mxfp4`; tolerances via `self_spread.py`.
- [ ] Lab (GPU 0, bench lock): `scripts/lab-bench.sh --model llama8b --golden16` (BF16 8B baseline), `--model llama8b-mxfp4 --golden16`, `--model llama-mxfp4-a4 --golden16`; targets per Interfaces; vLLM-ROCm where it loads; eval-compare per `gate.json`; soak on the 8B MXFP4 checkpoint (asked first).
- [ ] Flip the rows per their results; update `baseline_rows_present`. Run: `scripts/remote-cargo.sh test -p turbine-core support` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(core): mxfp4 (compressed-tensors) and mxfp4_a4 (Quark) on gfx1201 (gates recorded)`

## Task 21: Quantized dense layers under tensor parallelism

Files: `crates/turbine-model/src/loader.rs` (`SlotSource` sharding of data, scales and zeros per scheme; alignment check), `crates/turbine-model/src/executor/tp.rs` (quantized row/column-parallel linears), `crates/turbine-server/src/parallel.rs` or the Phase 5 plan-validation file (refusals `quant_shard_misaligned`, `quant_moe_phase7`), `crates/turbine-server/tests/tiny_server.rs` (`tp2_quantized_matches_tp1`), `scripts/lab/phase5-novanas-llama-fp8.yaml`, `scripts/lab/phase5-novanas-llama-awq.yaml`
Interfaces:

- column-parallel: `data` and per-channel `scales` split along n; per-tensor scale replicated; row-parallel: split along k, group scales and zeros split with it, requires `k_shard % group == 0` (and `% 128` for `FP8_BLOCK`) else `quant_shard_misaligned` naming the layer; any quantized expert tensor → `quant_moe_phase7`
  Covers: spec S-12; AC `tp2_quantized_matches_tp1`
  Depends on: Tasks 14, 18, 20

- [ ] Write failing test `tiny_server tp2_quantized_matches_tp1` (local and static modes over loopback tcp; `ct_fp8`, `awq`, `ct_mxfp4` tiny checkpoints; misaligned and MoE refusals). Run: `scripts/remote-cargo.sh test -p turbine-server --test tiny_server tp2_quantized` — expect FAIL
- [ ] Implement.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-server --test tiny_server` — expect PASS
- [ ] Lab (both GPUs, bench lock): `scripts/lab-cluster.sh --bench-lock tp2-novanas` with the two new configs — expect golden c1 PASS under `--batched-bounds` against each slug's reference.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(model): quantized dense layers under tensor parallelism`

## Task 22: `kv.dtype: fp8_e4m3` on the CPU provider, scales and identity

Files: `crates/turbine-core/src/config/kv.rs` (`dtype: KvDtypeChoice`), `crates/turbine-model/src/config.rs` (`kv_layout()` from the config, checkpoint `k_scale`/`v_scale` / `kv_cache_scheme`), `crates/turbine-model/src/executor/decoder/mod.rs` (paged append quantizes, attention reads with scales), `crates/turbine-kernels/src/cpu/paged.rs` and `cpu/attention.rs` (F8E4M3 pages), `crates/turbine-kernels/src/ops/mod.rs` (`AttentionPagedDesc.k_scale/v_scale`), `crates/turbine-server/src/kv_orchestrator.rs` (`KvFormat` with `KvDtype::Fp8E4m3PerTensorScale`), `crates/turbine-kv/src/identity.rs` (scales in the canonical format), `crates/turbine-model/tests/tiny_model.rs` (`fp8_kv_matches_reference`), `crates/turbine-kv/src/identity.rs` tests
Interfaces:

- `KvConfig.dtype: KvDtypeChoice::{Bf16, Fp8E4m3}` (serde `bf16`, `fp8_e4m3`); `KvFormat` canonical JSON gains `k_scales_hash` / `v_scales_hash` (BLAKE3 of the per-layer scales) when the dtype is FP8
  Covers: spec S-13 (host), S-16 (scales half); AC `fp8_kv_matches_reference`, `rope_and_scales_scope_the_namespace` (scales part)
  Depends on: Tasks 3, 6

- [ ] Write failing tests `tiny_model fp8_kv_matches_reference` (tiny Llama and OLMoE, with and without checkpoint scales) and `identity::tests::rope_and_scales_scope_the_namespace` (scales part). Run: `scripts/remote-cargo.sh test -p turbine-model --test tiny_model fp8_kv` and `scripts/remote-cargo.sh test -p turbine-kv identity` — expect FAIL
- [ ] Implement; `kv.dtype: fp8_e4m3` on a kernel library without FP8 paged attention exits 1 `kv_fp8_unavailable`.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-model -p turbine-kv -p turbine-server` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(kv): FP8 e4m3 KV cache with per-layer scales on the CPU provider`

## Task 23: FP8 paged attention — provider evaluation and HIP implementation

Files: `.procoder/ask/decisions.md` (entry "P6: FP8 paged attention — provider evaluation"), `kernels/rocm/CMakeLists.txt` (FMHA instance filter gains `fp8` if CK's instances build for gfx1201), `kernels/rocm/src/paged_attention.cpp`, `kernels/rocm/src/paged_attention_splitkv.cpp`, `kernels/rocm/src/paged_attention.hip` (FP8 append with scales; FP8-reading fallback kernel), `crates/turbine-kernels/tests/hip_ops.rs` (`paged_fp8_matches_cpu`)
Interfaces:

- `turbine_attention_paged_desc` with `dtype` F8E4M3 for pages and `k_scale`/`v_scale` (read only at minor ≥ 9); implementations named after the pick (e.g. `ck_tile_fmha_pagedkv_fp8`, `turbine_hip_fp8`)
- candidates: CK FMHA pagedkv / split-KV FP8 (`do_fp8_static_quant`, fp8 instances from `01_fmha/generate.py`) on gfx1201, the Turbine paged kernel with dequant-on-load, llama.cpp HIP flash attention (its quantized-KV path is int8 q8_0 blocks, not e4m3 pages — evaluated for adaptability), vLLM ROCm paged attention FP8 (gfx12 paths)
  Covers: spec S-13 (GPU); AC `paged_fp8_matches_cpu`
  Depends on: Task 22

- [ ] Evaluate (GPU 0, bench lock, 30-minute timeout): build, correctness vs CPU, decode/prefill µs at the Llama and OLMoE shapes; write the entry.
- [ ] Write failing lab test `hip_ops::paged_fp8_matches_cpu` (prefill and decode; block_tokens 128 and 16; every registered FP8 implementation). Run: `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops paged_fp8` — expect FAIL
- [ ] Implement; run — expect PASS; `scripts/lab-test.sh novanas --tier quick` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(rocm): FP8 KV paged attention`

## Task 24: FP8 KV lab proof and round trips

Files: `crates/turbine-server/tests/kv_gpu.rs` (`prefix_reuse_matches_cold_fp8_kv`, `nvme_round_trip_matches_cold_fp8_kv`), `tests/eval/llama-3.2-3b-instruct/turbine-fp8_e4m3.json`, `tests/eval/olmoe-1b-7b-0125-instruct/{turbine-bf16.json,turbine-fp8_e4m3.json}`, `crates/turbine-core/src/support.rs` (FP8 KV rows), `.procoder/perf-log.md`
Interfaces:

- rows `amd/gfx1201/{LlamaForCausalLM,OlmoeForCausalLM}/bf16/fp8_e4m3/none` → `supported` after the gate; FP8 KV with a quantized weight column only per combination that passed its own golden (`llama fp8 + fp8_e4m3` is run here)
  Covers: spec S-13, S-14, S-17; AC `paged_fp8` lab proof, `kv_gpu` FP8 round trips
  Depends on: Task 23

- [ ] Write failing lab tests in `kv_gpu.rs`. Run: `scripts/lab-test.sh novanas -- -p turbine-server --test kv_gpu` — expect FAIL (new tests), existing ones PASS
- [ ] Make them pass (bit-exact FP8 page bytes across L1/L2 round trips).
- [ ] Lab (GPU 0, bench lock): `scripts/lab-bench.sh --model llama --golden16 -- --set kv.dtype=fp8_e4m3` and `--model olmoe …`, and `--model llama-fp8 --golden16 -- --set kv.dtype=fp8_e4m3` — expect golden PASS under the batched bounds; L0 blocks ≥ 1.95 ×; c16 tok/s ≥ 0.95 × BF16 KV; eval-compare at 0.01 against BF16 KV — expect exit 0; soak with FP8 KV (asked first).
- [ ] Flip the passing rows; update `baseline_rows_present`. Run: `scripts/remote-cargo.sh test -p turbine-core support` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(core): FP8 KV supported on gfx1201 (golden, capacity, eval recorded)`

## Task 25: `quantization` in status, weight-format and KV metrics

Files: `crates/turbine-server/src/status.rs` or the file defining `StatusDocument` (`quantization` object), `crates/turbine-api/src/…` metrics registration for `turbine_weight_format_info`, `turbine_qgemm_calls_total`, `crates/turbine-api/tests/api.rs` (`status_reports_quantization`), `crates/turbine-model/src/weights/mod.rs` (`weight_format` log event)
Interfaces:

- `StatusDocument.quantization: QuantizationStatus { weight_format, packaging, activation, kv_dtype }` (6b adds `tier_formats` and `ladder`)
  Covers: spec S-19; AC `status_reports_quantization`
  Depends on: Task 24

- [ ] Write failing tests. Run: `scripts/remote-cargo.sh test -p turbine-api --test api status_reports_quantization` — expect FAIL
- [ ] Implement.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-api -p turbine-kv -p turbine-server` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(api): quantization status and weight-format metrics`

## Task 26: YaRN parsing, `inv_freq` and the attention factor

Files: `crates/turbine-model/src/config.rs` (`RopeScaling::Yarn`, parsing, `rope_identity()`, `max_seq_len` extension), `crates/turbine-model/src/executor/rope.rs` (`_compute_yarn_parameters` port), `crates/turbine-model/src/executor/decoder/mod.rs` (attention scale × factor²), `crates/turbine-core/src/config/mod.rs` (`model.rope_scaling` override), `crates/turbine-server/src/model.rs` (`resolve_max_seq_len` with YaRN), `crates/turbine-model/tests/fixtures/yarn_params.json` (committed transformers values for four configurations, generated once by `scripts/golden/yarn_params.py`), `scripts/golden/yarn_params.py` (new, fixture time)
Interfaces:

- `RopeScaling::Yarn { factor, original_max_position_embeddings, beta_fast, beta_slow, attention_factor, truncate }`; `fn yarn_attention_factor(factor: f64, mscale: Option<f64>, mscale_all_dim: Option<f64>) -> f64` (transformers' rule); `ModelConfig.rope_scaling: Option<serde_json::Value>` override
  Covers: spec S-15; AC `yarn_parameters_match_transformers`, `phase6_keys` (rope part)
  Depends on: Task 2

- [ ] Generate the fixture: `uv run scripts/golden/yarn_params.py --out crates/turbine-model/tests/fixtures/yarn_params.json` (transformers 4.57.1, the four configurations of the spec).
- [ ] Write failing test `config::tests::yarn_parameters_match_transformers`. Run: `scripts/remote-cargo.sh test -p turbine-model config::tests::yarn` — expect FAIL
- [ ] Implement; `dynamic` still refused naming `rope_scaling.rope_type`.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-model -p turbine-core -p turbine-server` — expect PASS
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(model): static YaRN RoPE scaling for every family`

## Task 27: RoPE configuration in the namespace key

Files: `crates/turbine-core/src/model_identity.rs` (identity carries `rope_identity`), `crates/turbine-kv/src/identity.rs` (canonical JSON field `rope`), `crates/turbine-server/src/kv_orchestrator.rs` (passes it), `crates/turbine-kv/src/identity.rs` tests
Interfaces:

- `ModelIdentity.rope: String` (canonical JSON of the resolved RoPE parameters); `namespace_key` includes it
  Covers: spec S-16 (RoPE half); AC `rope_and_scales_scope_the_namespace`
  Depends on: Tasks 22, 26

- [ ] Extend failing test `identity::tests::rope_and_scales_scope_the_namespace` with the RoPE cases. Run: `scripts/remote-cargo.sh test -p turbine-kv identity` — expect FAIL
- [ ] Implement.
- [ ] Run: `scripts/remote-cargo.sh test -p turbine-kv -p turbine-server` — expect PASS; `keys_are_stable_and_scoped` updated only where the canonical JSON gained the field (the change is documented in the test).
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `feat(kv): RoPE configuration scopes the prefix namespace`

## Task 28: YaRN proof on Llama-3.2-3B

Files: `tests/golden/llama-3.2-3b-instruct-yarn16/{reference.jsonl,tolerance.json,README.md,prompts-long.jsonl}`, `scripts/golden/hf_reference.py` (`--config-override`), `scripts/lab/phase6-novanas-llama-yarn16.yaml`, `benches/turbine-bench/tests/golden.rs`, `.procoder/perf-log.md`
Interfaces:

- the long prompt (≈ 12,000 tokens, deterministic text from the committed GSM8K questions concatenated) lives in `prompts-long.jsonl`; `turbine-golden compare --prompts` takes both files in one run
  Covers: spec S-16 (proof); AC YaRN lab golden
  Depends on: Task 27

- [ ] Fixture: `uv run scripts/golden/hf_reference.py --model-dir /home/piwi/turbine-models/llama-3.2-3b-instruct --config-override '{"rope_scaling":{"rope_type":"yarn","factor":16.0,"original_max_position_embeddings":8192,"beta_fast":32,"beta_slow":1}}' …` for both prompt files; tolerance: the Llama BF16 values unless `self_spread.py` on the override shows a larger spread (recorded in the README).
- [ ] Lab (GPU 0, bench lock): `scripts/lab-bench.sh --model llama-yarn16 --golden16` — expect PASS; `--model llama --golden16` unchanged.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `test(golden): YaRN Llama-3.2-3B reference and lab proof`

## Task 29: Phase exit (6a)

Files: `.procoder/perf-log.md` (6a summary), `AGENTS.md` (6a commands: weight formats, `kv.dtype: fp8_e4m3`, `model.rope_scaling`, lab-bench models, fixture scripts), `.procoder/contract/interfaces.md` (§26, 6a part), `.procoder/specs/phase-6a-quantization.md` (criteria ticked with evidence), `.procoder/plans/phase-6-8-expansion.md` (6a closed)
Interfaces:

- no new interface
  Covers: spec S-20 phase-exit criterion; umbrella Task 10 (track close runbook) for 6a
  Depends on: Tasks 1–28

- [ ] `scripts/gate.sh --full` — expect `gate: ok`
- [ ] `scripts/lab-test.sh novanas --tier full` and `scripts/lab-test.sh novanas --gpus 2 --features fault-injection --tier full` — expect exit 0
- [ ] `scripts/lab-bench.sh --golden16` for `llama`, `olmoe` and every proof model — expect PASS; every performance target met or its miss recorded with the user's decision.
- [ ] `scripts/overload-soak.sh novanas --duration 10m` (asked first) on BF16 Llama and each newly `supported` checkpoint — expect pass.
- [ ] `scripts/remote-cargo.sh run -p turbine-server -- --support-matrix --output json` — paste into the evidence; `scripts/track-gate.sh phase-6b-kv-compression` — expect the order check to pass.
- [ ] Report to the coordinator: every row's status, every non-passing format with its finding; 6b starts only after this report.
- [ ] Gate: `scripts/gate.sh` — expect `gate: ok`
- [ ] Commit: `docs: phase 6a exit — quantization closed`
