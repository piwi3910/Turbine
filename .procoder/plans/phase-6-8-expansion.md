# phase-6-8-expansion — implementation plan

Status: draft
Spec: .procoder/specs/phase-6-8-expansion.md

Renamed from `phase-8-expansion` and amended on 2026-09-28 with its spec (decision "Roadmap reorganisation after Phase 5 (2026-09-28)" in `.procoder/ask/decisions.md`): the tracks are now `phase-6-quantization`, `phase-7-model-families` and `phase-8-speculative-decoding`, in that order, AMD (`novanas`) only. In the code and in older docs "P8a" means phase-6-quantization, "P8c" phase-7-model-families and "P8b" phase-8-speculative-decoding; test modules named `phase8_…` keep their names.

State on main (2026-09-28): Tasks 2, 3, 4 and 7 landed in Phase 2m (S-11, from `runahead/p8-umbrella`); their text stays as the record of what was built. Tasks 1, 5, 6 and 8 were ported onto `phase-6a-quantization` by its Task 1 (2026-09-28; Task 8 as `scripts/track-gate.sh` with the track names `phase-6a-quantization`, `phase-6b-kv-compression`, `phase-7-model-families`, `phase-8-speculative-decoding` and the test `lab_scripts track_gate`). Tasks 9–12 run once per track. Since the user's split of 2026-09-28 (decision "Phase 6 split: 6a quantization, 6b KV compression"), track 1 is `phase-6a-quantization` then `phase-6b-kv-compression`: Task 9's gate applies to both specs (both written and COMPLETE on 2026-09-28), Task 10's close runbook runs once for each, and the ported `scripts/track-gate.sh` accepts both names. Track `phase-6a-quantization` closed 2026-09-30 (Task 10 run for 6a: plan Task 29, evidence in `.procoder/review-2026-09-29.md` rotation 15 and `.procoder/perf-log.md`; `scripts/track-gate.sh phase-6b-kv-compression` passes); 6b is paused until the user says go.

## Goal

Deliver the mechanisms every Phase 6–8 track shares — the support matrix with startup refusal, the `turbine-golden eval`/`eval-compare` quality gate and its committed GSM8K task set, the vendor-neutrality guard, and the scripted track-start gate and track-close procedure — so that `phase-6-quantization`, `phase-7-model-families` and `phase-8-speculative-decoding` can each be specified, gated and closed in order without re-deciding shared rules.

## Architecture

`turbine_core::support` holds the static `SUPPORT_MATRIX` table and its resolution rules (most specific row wins; `--check-config` resolves with unknown device columns); `turbine-server` prints it (`--support-matrix`), resolves it under `--check-config` (exit 2) and at startup step 3 after discovery (exit 1, WARN for experimental), puts the resolved row under `support` in `GET /turbine/v1/status` and sets `turbine_support_matrix_status{status}` through `turbine_api::support::SupportMetrics`. `turbine-bench` gains `golden::eval` behind `turbine-golden eval`/`eval-compare` plus the committed `tests/eval/gsm8k-200.jsonl`; `turbine-kernels/tests/vendor_neutral_api.rs` parses the five core crates with `syn` and rejects vendor type paths in public signatures. Track sequencing is enforced by `scripts/track-gate.sh` (previous track closed per the support matrix, track spec COMPLETE and inside the umbrella scope) and a per-track close runbook (Task 10); the track contents themselves are planned by each track's own plan.

## Constraints

From the spec (verbatim, as amended 2026-09-28):

- Rust only in the serving path; no Python at runtime (TS §21 rule 4). Python is allowed only at fixture-generation time (HF transformers dumps, reference-runtime captures, an offline-quantized fixture checkpoint), never in a build or test that `cargo test` runs. Vendor libraries are loaded at runtime through the prebuilt kernel library (_libturbine_hip.so_, CMake-built), never linked by Cargo, so the workspace still builds and every non-ignored test passes on macOS arm64 with no GPU libraries.
- `unsafe` stays in the allowlisted locations: `turbine-device`, `turbine-kernels`, `turbine-distributed/src/collective/ffi` (Phase 5), and `turbine-transport/src/rdma` once the deferred phase-10 adds it (CONFLICT C-19).
- The kernel C ABI stays vendor-neutral: every addition is an optional minor group resolved at load time (decisions "Phase 4: kernel ABI v2.5 instead of v3" and "P5 T6"), specified so a CUDA library can implement it when phase-2b is re-specced.
- Every lossy format or lossy KV transform ships with its quality gate (TS §8, S-3); nothing lossy is enabled by default unless the checkpoint itself is stored in that format.
- Bounded inputs (TS §16, §21 rule 8): speculative k ≤ 8; draft-model memory is reserved through the Phase 3 budget before speculation is enabled.
- Model weights live in `/home/piwi/turbine-models/<slug>` on `novanas`, fetched over SSH with the user's HF login there (never stored in the repo); tests read `TURBINE_TEST_MODEL_DIR` and never download.
- **Host workloads:** any lab run that needs workloads moved or memory freed on `novanas` is started only after the implementer has asked the user and the user has moved the workloads; the implementer never stops, moves or reconfigures someone else's workload. Reference-engine runs use `scripts/lab-serve.sh novanas --vllm <slug>` (vLLM-ROCm, port 18100).
- Lab host: `novanas` (192.168.10.203), 2× Radeon AI PRO R9700 (`gfx1201`, 32 GB each, native FP8 WMMA, no FP4 matrix path), ROCm 7.14.1 at `/opt/rocm/rocm`, k3s Jobs requesting `amd.com/gpu`; perf numbers on GPU 0 only. The Sparks are not used by Phases 6–8. Families that do not fit one card rely on Phase 5 TP over both cards and an in-scope quantized checkpoint (e.g. Mixtral-8x7B: ≈ 93 GB in BF16, ≈ 47 GB in FP8, so FP8 + TP 2); the track 2 spec states which per family.
- Questions the track specs must answer (inputs, not decided here): (a) the hybrids' cached checkpoints are NVFP4 (listed below), so `phase-7-model-families` needs FP8 or BF16 checkpoints of them — Qwen3.6-35B-A3B needs FP8 + TP 2 or similar to fit 2 × 32 GB; (b) `phase-6-quantization` proves each format on Llama-3.2-3B or OLMoE where such checkpoints exist and names them (e.g. RedHatAI / neuralmagic FP8, AWQ and GPTQ Llama-3.2-3B checkpoints); (c) MXFP4 checkpoints may exist only for gpt-oss, whose family arrives in track 2 — the `phase-6-quantization` spec chooses between an MXFP4 fixture checkpoint of a registered architecture quantized offline (a Python quantizer at fixture-generation time only) and proving MXFP4 in track 2 with gpt-oss-20b (the `mxfp4` row then turns `supported` when track 2 closes).
- Cached NVFP4 checkpoints (from their _config.json_, verified 2026-09-25), which defined the old Phase 8a scope and now belong to the deferred NVIDIA block: `nvidia/Qwen3.6-35B-A3B-NVFP4` (`qwen3_5_moe`, modelopt mixed precision, FP8 KV); `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` (`qwen3_5` dense hybrid, modelopt); `RadixArk/Qwen3.8-Flash-Next-NVFP4` (`qwen4_exp`, 126 GB); `YourHighnessLA/Qwen3.8-27B-DFlash2-NVFP4` (compressed-tensors `nvfp4-pack-quantized`).

From the interface contract (`.procoder/contract/interfaces.md`, binding):

- Toolchain: edition 2024, `rust-version = "1.97"`, `license = "Apache-2.0"` inherited from `[workspace.package]`; every crate but the four allowlisted ones sets `[lints.rust] unsafe_code = "forbid"`; this phase adds no `unsafe`.
- Every public enum a later phase extends is `#[non_exhaustive]`; every config struct is `#[serde(deny_unknown_fields, default)]`; every crate has one top-level `thiserror` error enum.
- Metric label values come from closed sets rendered by `as_str()`; `turbine_support_matrix_status{status}` ∈ supported, experimental, unsupported.
- Validation order (§3.2/§16.3): static `validate()` (exit 2) → device discovery → `validate_host` → P5 parallel plan (exit 2) → **P8 support-matrix resolution** (exit 1 at startup / exit 2 under `--check-config`) → kernel library/model/budget (exit 1).
- Tests: unit tests in `#[cfg(test)] mod tests`, addressed `cargo test -p <crate> <module>::tests::<name>`; integration tests `crates/<crate>/tests/<binary>.rs`, addressed `cargo test -p <crate> --test <binary> <name>`; anything needing a GPU, weights or a lab host is `#[ignore]` and runs via `scripts/lab-test.sh <host>`; non-ignored tests pass on macOS arm64 with no GPU libraries and no weights. Phase-8 additions to an existing integration-test file go into their own `mod phase8_…` block so they never collide with earlier helpers or imports (the `--test <binary> <name>` filter still matches).
- Gate after every task: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`, then `cargo test --workspace`.
- Lab rules (§21.1): no `docker run` outside the defined lab scripts; scripts never stop/restart/reconfigure non-`turbine-lab-*` workloads; any run needing workloads moved or memory freed is asked of the user first. Phases 6–8 use `novanas` only (the Spark rules apply again when phase-2b is re-specced).

## Task 1: `quality` and `speculative.method` configuration keys

Files: `crates/turbine-core/src/config/quality.rs` (new: `QualityConfig`, its validation and unit test), `crates/turbine-core/src/config/speculative.rs` (new: `SpeculativeConfig`, `SpeculativeMethod` and unit test), `crates/turbine-core/src/config/mod.rs` (declare both modules, re-export, add the two `Config` fields, call `self.quality.validate()?` in `Config::validate`).
Interfaces:

- produces `pub const DEFAULT_MAX_ACCURACY_DROP: f64 = 0.01` (in `turbine_core::config::quality`)
- produces `#[serde(deny_unknown_fields, default)] pub struct QualityConfig { pub max_accuracy_drop: f64 }` (default 0.01)
- produces `pub fn QualityConfig::validate(&self) -> Result<(), ConfigError>` (`ConfigError::Invalid { key: "quality.max_accuracy_drop", .. }` when not finite or outside 0..=0.1)
- produces `#[non_exhaustive] #[serde(rename_all = "lowercase")] pub enum SpeculativeMethod { #[default] None, Draft }` + `pub fn as_str(self) -> &'static str` (`"none"`/`"draft"`)
- produces `#[serde(deny_unknown_fields, default)] pub struct SpeculativeConfig { pub method: SpeculativeMethod }` (the phase-8-speculative-decoding plan adds `num_tokens`, `draft_model_path`, `min_acceptance`)
- produces `Config.quality: QualityConfig` (`// P8`), `Config.speculative: SpeculativeConfig` (`// P8b reserved; P8 umbrella owns method`), both after `parallel` (contract §3.2 names); re-exported as `turbine_core::config::{QualityConfig, SpeculativeConfig, SpeculativeMethod}`
- consumes P0 `ConfigError` (`key() -> Option<&str>`), `Config`, `Config::validate`

Covers: S-2 (the `speculative` column's config key), S-4 (`quality.max_accuracy_drop`); no acceptance criterion on its own (Tasks 2, 3 and 5 exercise these keys).
Depends on: phase-0 (`Config`, `ConfigError`, `serde_norway`).

- [ ] Write failing test `config::quality::tests::quality_config_validation`: the default is 0.01 and validates; 0.0, 0.05 and 0.1 validate; -0.001, 0.1001, NaN and infinity each fail with `err.key() == Some("quality.max_accuracy_drop")`.
- [ ] Write failing test `config::speculative::tests::speculative_method_parses`: YAML `method: draft` parses to `SpeculativeMethod::Draft`, the default is `None`, and both `method: eagle` and `num_tokens: 4` are rejected. Create both files with only their test modules and add `mod quality; mod speculative;` to `config/mod.rs`. Run: `cargo test -p turbine-core config::` — expect FAIL ("cannot find").
- [ ] Implement both modules with serde derives and a manual `Default` for `QualityConfig`; re-export them from `config/mod.rs`, add the two `Config` fields after `parallel`, and call `self.quality.validate()?` in `Config::validate` after the P6 distributed rules. If an earlier plan declared placeholder `QualityConfig`/`SpeculativeConfig` structs in `config/mod.rs`, delete them so these modules are the only definitions.
- [ ] Run: `cargo test -p turbine-core -- config::quality::tests::quality_config_validation config::speculative::tests::speculative_method_parses` — expect PASS; `cargo test -p turbine-core` — expect PASS (P0 `config::tests::example_config_loads` still loads because both sections default).
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(core): add quality.max_accuracy_drop and speculative.method config keys`

## Task 2: Support matrix in `turbine-core`

Files: `crates/turbine-core/src/support.rs` (new: table, resolution, refusal, table validation, unit tests), `crates/turbine-core/src/lib.rs` (add `pub mod support;`).
Interfaces:

- produces `pub const VENDORS: &[&str] = &["amd", "nvidia", "cpu"]`, `pub const WILDCARD: &str = "*"`
- produces `#[non_exhaustive] pub enum WeightFormat { Bf16, ModeloptNvfp4, ModeloptFp8, ModeloptMixed, CtNvfp4 }` + `pub const ALL: [WeightFormat; 5]`, `pub fn as_str(self) -> &'static str` (`bf16`, `modelopt_nvfp4`, `modelopt_fp8`, `modelopt_mixed`, `ct_nvfp4`)
- produces `#[non_exhaustive] pub enum KvFormatColumn { Bf16, Fp8E4m3 }` + `pub const ALL: [KvFormatColumn; 2]`, `pub fn as_str(self) -> &'static str` (`bf16`, `fp8_e4m3`)
- produces `#[non_exhaustive] pub enum SpeculativeColumn { None, Draft }` + `pub const ALL: [SpeculativeColumn; 2]`, `pub fn as_str(self) -> &'static str` (`none`, `draft`)
- produces `pub enum SupportStatus { Supported, Experimental, Unsupported { reason: Cow<'static, str> } }` + `pub fn as_str(&self) -> &'static str`, `pub fn reason(&self) -> Option<&str>`, private `fn rank(&self) -> u8` (Supported 2 > Experimental 1 > Unsupported 0)
- produces `pub struct SupportKey { pub vendor: String, pub arch: String, pub architecture: String, pub weight_format: WeightFormat, pub kv_format: KvFormatColumn, pub speculative: SpeculativeColumn }` + `impl Display` (`vendor=… arch=… architecture=… weight_format=… kv_format=… speculative=…`)
- produces `pub fn SupportKey::is_partial(&self) -> bool`, `pub fn SupportKey::blamed_config_key(&self) -> &'static str`, `pub fn SupportKey::for_check_config(cfg: &Config) -> SupportKey`, `pub fn SupportKey::for_startup(cfg: &Config, arch: &str, architecture: &str) -> SupportKey`
- produces `pub struct SupportKeyPattern { pub vendor: Option<&'static str>, pub arch: Option<&'static str>, pub architecture: Option<&'static str>, pub weight_format: Option<WeightFormat>, pub kv_format: Option<KvFormatColumn>, pub speculative: Option<SpeculativeColumn> }` (`None` = `*`) + `pub fn specificity(&self) -> u32`, `pub fn matches(&self, key: &SupportKey) -> bool`, private `fn compatible(&self, key: &SupportKey) -> bool`, `fn overlaps(&self, other: &SupportKeyPattern) -> bool`
- produces `pub struct SupportRow { pub key: SupportKeyPattern, pub status: SupportStatus }` + `pub fn view(&self) -> SupportRowView`
- produces `#[derive(Serialize)] pub struct SupportRowView { pub vendor: String, pub arch: String, pub architecture: String, pub weight_format: String, pub kv_format: String, pub speculative: String, pub status: &'static str, pub reason: Option<String> }`
- produces `pub struct SupportDecision { pub key: SupportKey, pub status: SupportStatus }` + `pub fn warning(&self) -> Option<String>` (`support matrix: {key} is experimental`), `pub fn view(&self) -> SupportRowView`
- produces `pub static SUPPORT_MATRIX: &[SupportRow]`
- produces `pub fn resolve(key: &SupportKey) -> SupportStatus`, `pub fn resolve_in(table: &[SupportRow], key: &SupportKey) -> SupportStatus`, `pub fn resolve_partial_in(table: &[SupportRow], key: &SupportKey) -> SupportStatus`
- produces `pub fn check(key: SupportKey) -> Result<SupportDecision, ConfigError>`, `pub fn check_in(table: &[SupportRow], key: SupportKey) -> Result<SupportDecision, ConfigError>`
- produces `pub fn validate_table(table: &[SupportRow]) -> Result<(), String>`
- produces `pub fn vendor_column(backend: ExecutionBackend) -> &'static str` (hip→amd, cuda→nvidia, cpu→cpu)
- produces `pub fn config_columns(cfg: &Config) -> (WeightFormat, KvFormatColumn, SpeculativeColumn)` (phase-6-quantization replaces the weight/KV derivation)
- consumes Task 1 `Config.speculative.method`, P0 `ConfigError::Invalid { key, reason }`, P1 `turbine_core::types::ExecutionBackend { Hip, Cuda, Cpu }`

Covers: S-2 — `cargo test -p turbine-core support::tests::resolution_and_refusal`, `cargo test -p turbine-core support::tests::baseline_rows_present`.
Depends on: Task 1; phase-1 (`ExecutionBackend`).

Rows are added only by track plans once their exit gate passes (Task 10); a track replaces its `unsupported` refusal rows with validated rows and keeps `validate_table` green.

Amended 2026-09-28 (a code change for the first task of the `phase-6-quantization` plan, not part of this plan's landed Task 2): the refusal reasons name the new tracks (`phase-6-quantization`, `phase-7-model-families`, `phase-8-speculative-decoding`); the four `nvidia`/`sm_121` baseline rows become `unsupported` with a reason naming the deferred `phase-2b-nvidia`; `WeightFormatColumn` gains `fp8`, `fp8_block`, `mxfp4`, `awq_int4`, `gptq_int4` with `unsupported` refusal rows; `baseline_rows_present` is updated in the same commit.

- [ ] Write failing test `support::tests::resolution_and_refusal`: on a four-row test table (amd/* unsupported "amd needs a validated arch"; amd/gfx1201 supported; nvidia/sm_121 experimental; _/_/draft unsupported), `validate_table` passes; amd/gfx1201 resolves `Supported` (most specific wins) and amd/gfx1100 `unsupported`; `check_in` for amd/gfx1100 errors with text containing `amd needs a validated arch` and `vendor=amd arch=gfx1100`; nvidia/sm_121 passes with a `warning()` containing `experimental`, amd/gfx1201 has no warning; nvidia/sm_90 resolves with reason `no support-matrix row`; on the real table a draft key (full and partial with `WILDCARD` arch/architecture) fails with `key() == Some("speculative.method")` while partial amd/bf16/none checks `Supported`; two overlapping rows of equal specificity fail `validate_table` with `equal specificity`, a vendor `intel` row fails, and `validate_table(SUPPORT_MATRIX)` passes.
- [ ] Write failing test `support::tests::baseline_rows_present`: for amd/gfx1201 and nvidia/sm_121 × `LlamaForCausalLM`/`OlmoeForCausalLM` (bf16, bf16, none) the most specific matching row has specificity 6 and resolves `Supported`; every `supported` row is bf16/bf16/none; every non-bf16 weight format on nvidia/sm_121, `fp8_e4m3` KV and `draft` resolve `unsupported`; every row's view uses only `VENDORS`/`ALL` values or `*`. Create `support.rs` with only the tests module and add `pub mod support;`. Run: `cargo test -p turbine-core support::` — expect FAIL ("cannot find").
- [ ] Implement the enums, key, pattern, row, view and decision types, and `SUPPORT_MATRIX` with exactly 11 rows: the four baseline rows (amd/gfx1201 and nvidia/sm_121 × `LlamaForCausalLM`/`OlmoeForCausalLM`, bf16, bf16, none, `Supported`); one CPU reference-provider row `(cpu, *, *, bf16, bf16, none)` `Experimental` (decision 2026-09-25, `.procoder/ask/decisions.md`); four `(*, *, *, <quantized weight format>, bf16, none)` rows `Unsupported` with reason `quantized checkpoints are not validated yet (track phase-8a-quantization)`; `(*, *, *, *, fp8_e4m3, none)` `Unsupported` with `fp8_e4m3 KV cache is not validated yet (track phase-8a-quantization)`; `(*, *, *, *, *, draft)` `Unsupported` with `draft-model speculative decoding is not validated yet (track phase-8b-speculative-decoding)`.
- [ ] Implement resolution: `resolve_in` takes the max-specificity row that `matches` exactly (no row → `Unsupported { reason: "no support-matrix row" }`); `resolve_partial_in` (used when `is_partial()`, i.e. vendor/arch/architecture is `*`) returns the best-ranked non-unsupported status among `compatible` rows (a `*` key column matches any value), else the most specific compatible row's refusal; `check_in` turns unsupported into `ConfigError::Invalid { key: blamed_config_key(), reason: "support matrix: {key} is unsupported: {reason}" }`; `blamed_config_key` returns `speculative.method` (draft) → `kv.dtype` (non-bf16 KV) → `model.path` (non-bf16 weights or an architecture no row names) → `execution.backend`; `for_check_config` uses arch `cpu` for the CPU backend and `*` otherwise with architecture `*`; `validate_table` rejects vendors outside `VENDORS`, empty or `"*"` string columns (wildcards are `None`), and overlapping row pairs with equal specificity ("rows {i} and {j} overlap with equal specificity").
- [ ] Run: `cargo test -p turbine-core -- support::tests::resolution_and_refusal support::tests::baseline_rows_present` — expect PASS.
- [ ] Mutation check (do not commit): in `resolve_in` change `max_by_key` to `min_by_key`, run `cargo test -p turbine-core support::tests::resolution_and_refusal` — expect FAIL; revert. Change the first baseline row to `Experimental`, run `cargo test -p turbine-core support::tests::baseline_rows_present` — expect FAIL; revert.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(core): add the phase 8 support matrix with most-specific-row resolution`

## Task 3: `turbine-server --support-matrix` and `--check-config` resolution

Files: `crates/turbine-server/src/support_matrix.rs` (new: table rendering and the `--check-config` check, unit test), `crates/turbine-server/src/cli.rs` (new `--support-matrix`, `--output`, `OutputFormat`; `--config` optional only with `--support-matrix`), `crates/turbine-server/src/main.rs` (dispatch `--support-matrix` before any config read; call the check in the `--check-config` branch), `crates/turbine-server/tests/server_cli.rs` (append `mod phase8_support_matrix`).
Interfaces:

- produces `#[derive(clap::ValueEnum)] pub enum OutputFormat { Text, Json }` (in `turbine_server::cli`)
- produces `Cli.config: Option<PathBuf>` (`required_unless_present = "support_matrix"`), `Cli.check_config: bool` (`conflicts_with = "support_matrix"`), `Cli.support_matrix: bool` (`conflicts_with_all = ["config", "check_config"]`), `Cli.output: OutputFormat` (default `Text`, `requires = "support_matrix"`)
- produces `pub fn support_matrix::render_matrix(output: OutputFormat) -> String` (JSON `{"rows":[SupportRowView…]}`; text: header line then one line per row, whitespace-separated columns `vendor arch architecture weight_format kv_format speculative status reason`, `-` for no reason — Task 8 parses columns 1–7)
- produces `pub fn support_matrix::check_config(cfg: &Config) -> Result<SupportDecision, ConfigError>` (= `support::check(SupportKey::for_check_config(cfg))`)
- consumes Task 2 `SUPPORT_MATRIX`, `SupportRow::view`, `SupportKey::for_check_config`, `support::check`; P0 `turbine_core::config::load`, P0 `ExitCode { Clean = 0, Startup = 1, Config = 2, .. }`

Covers: S-2 — `cargo test -p turbine-server --test server_cli support_matrix_output`.
Depends on: Task 2; phase-0 (`turbine-server` CLI, exit codes).

- [ ] Write failing test `phase8_support_matrix::support_matrix_output` in `tests/server_cli.rs`: `turbine-server --support-matrix --output json` exits 0 with a non-empty `rows` array whose rows carry all eight keys and include one `speculative: "draft"`, `status: "unsupported"` row; `--config ok.yaml --check-config` (only `model.path: /nonexistent/model`) exits 0 printing `config ok`; the same plus `speculative.method: draft` exits 2 with stderr containing `speculative.method` and `speculative=draft` and no `config ok`. Run: `cargo test -p turbine-server --test server_cli support_matrix_output` — expect FAIL ("unexpected argument '--support-matrix'").
- [ ] Write failing test `support_matrix::tests::matrix_text_lists_every_row`: the text rendering has a header starting `vendor ` plus exactly `SUPPORT_MATRIX.len()` lines, one of which splits into `amd gfx1201 LlamaForCausalLM bf16 bf16 none supported`.
- [ ] Implement the clap fields and `OutputFormat` in `cli.rs` (keep the P0 `--set` field); implement `render_matrix` with `serde_json::json!({"rows": rows})` plus a trailing newline for JSON and left-padded columns (`{:<7} {:<8} {:<18} {:<14} {:<9} {:<11} {:<12} reason`) for text.
- [ ] Wire `main.rs`: add `mod support_matrix;`; the first statement after `Cli::parse()` prints `render_matrix(cli.output)` and returns `ExitCode::Clean` when `cli.support_matrix`, otherwise takes `config_path` from `cli.config` (clap guarantees it); in the `--check-config` branch, after the P0/P5 validation and before `config ok`, a `check_config` error is printed to stderr and returns `ExitCode::Config`, using the same exit conversion the surrounding P0 branches use.
- [ ] Run: `cargo test -p turbine-server --test server_cli support_matrix_output` and `cargo test -p turbine-server support_matrix::tests::matrix_text_lists_every_row` — expect PASS; `cargo test -p turbine-server --test server_cli` — expect PASS (P0 `invalid_config_exits_2_before_bind` unaffected).
- [ ] Check by hand: `cargo run -q -p turbine-server -- --support-matrix` — expect the header plus 11 rows, the first `amd     gfx1201  LlamaForCausalLM   bf16           bf16      none        supported    -`.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(server): add --support-matrix and resolve the support row under --check-config`

## Task 4: Startup resolution, `support` status key and `turbine_support_matrix_status`

Files: `crates/turbine-api/src/support.rs` (new: `SupportMetrics` gauge family), `crates/turbine-api/src/lib.rs` (add `pub mod support;`), `crates/turbine-api/Cargo.toml` (`turbine-core` dependency if P0 did not add it — allowed edge, contract §1.1), `crates/turbine-api/tests/api.rs` (append `mod phase8_support`), `crates/turbine-server/src/support_startup.rs` (new: device arch, `config.json` architecture, startup decision, unit test), `crates/turbine-server/src/startup.rs` (call it at step 3, register the gauge, fill the status document), `crates/turbine-server/src/main.rs` (add `mod support_startup;`), the file defining `StatusDocument` in `crates/turbine-server/src/` (add the `support` field), `crates/turbine-server/Cargo.toml` (dev-deps `tempfile`, `serde_json` if absent), `crates/turbine-server/tests/tiny_server.rs` (append `mod phase8_status`).
Interfaces:

- produces `#[derive(EncodeLabelSet)] pub struct SupportStatusLabels { pub status: &'static str }` (in `turbine_api::support`; `status` ∈ supported, experimental, unsupported)
- produces `pub fn SupportMetrics::register(reg: &MetricsRegistry) -> SupportMetrics` (registers `turbine_support_matrix_status` with all three label values at 0), `pub fn SupportMetrics::set(&self, resolved: &SupportStatus)` (resolved value 1, the other two 0)
- produces `pub fn support_startup::device_arch(cfg: &Config, inventory: &DeviceInventory) -> Option<String>` (`cpu` for the CPU backend; `None` when the configured device is missing or of the other vendor)
- produces `pub fn support_startup::read_architecture(model_dir: &Path) -> Option<String>` (`architectures[0]` of `<model_dir>/config.json`; `None` when missing or malformed)
- produces `pub fn support_startup::startup_decision(cfg: &Config, arch: Option<&str>, model_dir: &Path) -> Result<Option<SupportDecision>, ConfigError>` (`Ok(None)` when arch or architecture is unknown; logs `event = "support_matrix"` at WARN for experimental)
- produces `StatusDocument.support: Option<SupportRowView>` (`#[serde(skip_serializing_if = "Option::is_none")]`, JSON key `support`)
- consumes Task 2 `support::check`, `SupportKey::for_startup`, `SupportDecision::{warning, view}`, `SupportRowView`, `SupportStatus::as_str`; P0 `MetricsRegistry::register`, `ApiState`, `router`, `Diagnostics`, `InferenceBackend`, `Readiness`, `ApiLimits`, `ApiError::not_implemented()`; P0 `turbine_device::DeviceInventory { devices: Vec<DeviceInfo { index, vendor, arch, .. }> }`; P1 `turbine_model::testing::tiny::write_tiny_llama(dir, seed)`

Covers: S-2 — `cargo test -p turbine-api --test api status_reports_support_row`; edge case "a track rolled back after release fails at next startup, never mid-request" (resolution runs only at startup).
Depends on: Tasks 1–3; phase-0 (API traits, metrics registry, discovery), phase-1 (`StatusDocument`, tiny Llama fixture, startup exit-1 path), phase-5 (parallel plan step), phase-6 (`Diagnostics::debug_faults` if present).

- [ ] Write failing test `phase8_support::status_reports_support_row` in `crates/turbine-api/tests/api.rs`: with test doubles `SupportNoInference`, `SupportStatusDiagnostics` (its status document is `{"ready": true, "support": decision.view()}`; if the P6 `Diagnostics` trait carries `debug_faults`, add it behind `#[cfg(feature = "fault-injection")]` returning `ApiError::not_implemented()`) and `SupportReady`, a CPU-backend decision from `support::check(SupportKey::for_startup(&cfg, "cpu", "LlamaForCausalLM"))` makes `GET /turbine/v1/status` report `support` = cpu/cpu/LlamaForCausalLM/bf16/bf16/none/`experimental` with a null reason, and `/metrics` has exactly three `turbine_support_matrix_status{` lines of which only `turbine_support_matrix_status{status="experimental"} 1` ends in ` 1`. Run: `cargo test -p turbine-api --test api status_reports_support_row` — expect FAIL ("unresolved import `turbine_api::support`").
- [ ] Implement `turbine_api::support` with a `prometheus_client` `Family<SupportStatusLabels, Gauge>` registered through `MetricsRegistry::register`, pre-creating all three label values at 0.
- [ ] Run: `cargo test -p turbine-api --test api status_reports_support_row` — expect PASS.
- [ ] Write failing test `phase8_status::support_row_in_status` in `crates/turbine-server/tests/tiny_server.rs`: a `turbine-server` child (killed on drop) serving a `write_tiny_llama(&model, 7)` checkpoint with `execution.backend: cpu` on a free port becomes ready within 60 s, and plain-HTTP `GET /turbine/v1/status` reports `support.vendor` `cpu`, `support.architecture` `LlamaForCausalLM`, `support.status` `experimental`, while `/metrics` contains `turbine_support_matrix_status{status="experimental"} 1` and `turbine_support_matrix_status{status="supported"} 0`. Run: `cargo test -p turbine-server --test tiny_server support_row_in_status` — expect FAIL (`left: Null`, no `support` key yet).
- [ ] Write failing test `support_startup::tests::startup_decision_cases`: with a temp `config.json` per architecture, HIP + `gfx1201` + `LlamaForCausalLM` is `Supported` with vendor `amd`; arch `None` or a directory without `config.json` gives `Ok(None)`; `MistralForCausalLM` fails with `key() == Some("model.path")` and `no support-matrix row`; `speculative.method: draft` fails with `speculative.method`; the CPU backend with `device_arch(&cpu, &DeviceInventory::default()) == Some("cpu")` is `Experimental` with a warning containing `vendor=cpu`; HIP with an empty inventory has arch `None`.
- [ ] Implement `support_startup` (`serde_json` for `config.json`, `tracing::warn!(event = "support_matrix", status = …)` for the warning) and wire startup step 3 in `startup.rs` directly after the P5 parallel plan and before `ShimLibrary::load` / choosing the CPU provider: compute `device_arch`, call `startup_decision(&cfg, arch.as_deref(), &cfg.model.path)`, on `Err` print it and take the existing P1 exit-1 path used by the model-file failure branch, register `SupportMetrics` on the process registry and `set` the resolved status, then fill `StatusDocument.support` with `support_decision.as_ref().map(|d| d.view())`.
- [ ] Run: `cargo test -p turbine-server support_startup::tests::startup_decision_cases`, `cargo test -p turbine-server --test tiny_server support_row_in_status`, `cargo test -p turbine-api --test api status_reports_support_row` — expect PASS; `cargo test -p turbine-server` — expect PASS (P1 `startup_failures_exit_1` keeps its messages: missing `config.json` and unknown devices defer to the later steps).
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(server): refuse unsupported support-matrix rows at startup and report the row in status and metrics`

## Task 5: `turbine-golden eval` and `eval-compare`

Files: `benches/turbine-bench/src/golden/eval.rs` (new: task parsing, numeric matching, run, report, compare; unit test), `benches/turbine-bench/src/golden/mod.rs` (add `pub mod eval;`), `benches/turbine-bench/src/bin/turbine-golden.rs` (add the `eval` and `eval-compare` subcommands), `benches/turbine-bench/Cargo.toml` (`reqwest` feature `json`, `turbine-core`, `thiserror`; dev-deps `axum`, `tempfile`, `tokio`), `benches/turbine-bench/tests/golden.rs` (append `mod phase8_eval`).
Interfaces:

- produces `pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(600)` (in `turbine_bench::golden::eval`)
- produces `#[serde(rename_all = "lowercase")] pub enum MatchKind { Exact, Number }`
- produces `#[serde(deny_unknown_fields)] pub struct EvalTask { pub id: String, pub prompt: Option<String>, pub messages: Option<Vec<serde_json::Value>>, pub answer: String, #[serde(rename = "match")] pub match_kind: MatchKind, pub max_tokens: u32 }` (tasks-file line `{"id","prompt"|"messages","answer","match":"exact"|"number","max_tokens"}`)
- produces `pub struct TaskResult { pub id: String, pub correct: bool, pub output: String }`
- produces `pub struct EvalReport { pub model: String, pub tasks_file: String, pub total: usize, pub correct: usize, pub accuracy: f64, pub results: Vec<TaskResult> }`
- produces `#[derive(thiserror::Error)] pub enum EvalError { Io { path: PathBuf, detail: String }, Task { path: PathBuf, line: usize, detail: String }, Request { id: String, detail: String }, Server(String) }` (`Request` displays `task {id}: {detail}`)
- produces `pub struct CompareOutcome { pub baseline_accuracy: f64, pub candidate_accuracy: f64, pub max_drop: f64, pub pass: bool }`
- produces `pub fn load_tasks(path: &Path) -> Result<Vec<EvalTask>, EvalError>`, `pub fn normalize_number(s: &str) -> Option<String>`, `pub fn is_correct(kind: MatchKind, expected: &str, output: &str) -> bool`
- produces `pub async fn run_eval(base: &str, model: Option<&str>, tasks_file: &Path, tasks: &[EvalTask]) -> Result<EvalReport, EvalError>`
- produces `pub fn compare(baseline: &EvalReport, candidate: &EvalReport, max_drop: f64) -> CompareOutcome`, `pub fn read_report(path: &Path) -> Result<EvalReport, EvalError>`
- produces CLI `turbine-golden eval --url <base> --tasks <file> [--model <name>] [--output text|json]` (exit 0 completed; 2 usage/I/O/server error naming the task, no report printed on failure) and `turbine-golden eval-compare --baseline <r.json> --candidate <r.json> [--max-drop <float>]` (prints `baseline accuracy {:.4}, candidate accuracy {:.4}, max drop {:.4}: PASS|FAIL`; exit 0 pass, 1 fail, 2 usage/I/O or `--max-drop` outside 0..=0.1)
- consumes Task 1 `QualityConfig::default().max_accuracy_drop` (default of `--max-drop`); P1 `turbine-golden` clap `Command` enum and `--output` value enum

Covers: S-3, S-4 — `cargo test -p turbine-bench --test golden eval_accuracy_report`; failure mode "eval run fails midway".
Depends on: Task 1; phase-1 (`turbine-golden`).

- [ ] Write failing test `phase8_eval::eval_accuracy_report` in `benches/turbine-bench/tests/golden.rs`: an in-process Axum mock serving `/v1/models` (`mock-model`), `/v1/completions` and `/v1/chat/completions` (asserting `temperature` 0.0 and `model` `mock-model`) answers prompt `q<i>` with answer `i*1000+7` — tasks 0..150 correct as plain, comma-grouped or trailing-period numbers, 150..200 wrong as `$`-prefixed, unit-suffixed or off-by-one, and prompt `fail` → HTTP 500; 200 tasks (even ids completions, odd ids chat, `match: number`) give `eval --output json` exit 0 with accuracy 0.75, correct 150, total 200, model `mock-model`, results 1/2 correct and 150/151 incorrect; `eval-compare` of that baseline against candidates at 0.745 and 0.73 with `--max-drop 0.01` exits 0 and 1, both printing `baseline accuracy 0.7500`; a task set whose task 120 is `fail` exits 2 with stderr containing `task t120` and empty stdout. Run: `cargo test -p turbine-bench --test golden eval_accuracy_report` — expect FAIL ("unrecognized subcommand 'eval'").
- [ ] Write failing test `golden::eval::tests::numeric_match_rules`: `1,234` and ` 1234.\n` match `1234`, `-5` matches `-5`, `2.5.` matches `2.5`; `$18`, `18 apples`, `18..` and `180` do not match `18`; exact matching trims (`yes` = `yes`) but is case-sensitive (`Yes` ≠ `yes`).
- [ ] Implement `eval.rs`: `load_tasks` rejects lines without exactly one of `prompt`/`messages`, non-numeric `number` answers and duplicate ids (error names file and line); `normalize_number` trims, strips one trailing `.`, removes `,`, then accepts only an optional `-` and digits with at most one `.`; `run_eval` uses one `reqwest::Client` with `REQUEST_TIMEOUT`, reads the model from `GET /v1/models` `data[0].id` unless given, sends tasks sequentially with `temperature: 0.0`, `stream: false` to `/v1/completions` (prompt) or `/v1/chat/completions` (messages), reads `choices[0].text` or `choices[0].message.content`, and aborts on the first failed request; `compare` passes when `candidate + 1e-9 >= baseline - max_drop`.
- [ ] Implement the `Eval(EvalArgs)` and `EvalCompare(EvalCompareArgs)` subcommands in `turbine-golden.rs` with handlers `run_eval`/`run_eval_compare` returning `std::process::ExitCode`; `eval` runs on a current-thread Tokio runtime and prints pretty JSON or `model {} tasks {}: accuracy {:.4} ({}/{})`; reuse the P1 `Output` value enum if it has `Text`/`Json`, else add it.
- [ ] Run: `cargo test -p turbine-bench --test golden eval_accuracy_report` and `cargo test -p turbine-bench golden::eval::tests::numeric_match_rules` — expect PASS; `cargo test -p turbine-bench --test golden` — expect PASS (P1 `capture_and_compare_roundtrip` unaffected).
- [ ] Mutation check (do not commit): in `compare` replace `baseline.accuracy - max_drop` with `baseline.accuracy`, run `cargo test -p turbine-bench --test golden eval_accuracy_report` — expect FAIL (`left: (Some(1), Some(1))`); revert.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(bench): add turbine-golden eval and eval-compare quality gate`

## Task 6: Committed GSM8K-200 task set

Files: `scripts/eval/make-gsm8k-200.sh` (new, mode 755: reproducible fixture generator, curl + jq, no Python), `tests/eval/gsm8k-200.jsonl` (new: 200 items), `tests/eval/NOTICE` (new: source, commit, transformation, MIT text), `benches/turbine-bench/tests/golden.rs` (append `mod phase8_eval_task_set`).
Interfaces:

- produces `tests/eval/gsm8k-200.jsonl`: ids `gsm8k-test-0000` … `gsm8k-test-0199`, one chat user message each, numeric `answer`, `"match": "number"`, `"max_tokens": 32` — used by every lossy-format gate (Task 10)
- produces `scripts/eval/make-gsm8k-200.sh [out]` (default `tests/eval/gsm8k-200.jsonl`), printing `wrote <out> (200 items, commit <sha>)`
- consumes Task 5 `load_tasks`, `MatchKind`, `normalize_number`

Covers: S-4 — `cargo test -p turbine-bench --test golden eval_task_set_valid`.
Depends on: Task 5.

- [ ] Write failing test `phase8_eval_task_set::eval_task_set_valid`: `tests/eval/gsm8k-200.jsonl` (resolved from `CARGO_MANIFEST_DIR/../../tests/eval`) has exactly 200 lines, `load_tasks` parses 200 tasks with unique ids, each `MatchKind::Number` with a numeric answer and `max_tokens > 0`, and `tests/eval/NOTICE` contains `MIT License`. Run: `cargo test -p turbine-bench --test golden eval_task_set_valid` — expect FAIL ("No such file or directory").
- [ ] Implement `scripts/eval/make-gsm8k-200.sh` (`set -euo pipefail`, temp files removed by `trap`): download `https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl` with `curl -fsSL`, take the first 200 lines and `jq -c -s` each into `{id: "gsm8k-test-" + 4-digit zero-padded index, messages: [{role: "user", content: question + "\n\nSolve the problem step by step. On the last line, write \"Answer: \" followed by the final answer as a number."}], answer: text after "#### " with `,` removed and whitespace trimmed, match: "final_number", max_tokens: 512}` (amended 2026-09-29: the first version — bare number in 32 tokens — scored 2.5 % on 3B Instruct and 8 % on 8B, measuring format compliance, not arithmetic; lead decision, provisional), assert 200 output lines, move into place and print the `wrote` line.
- [ ] Run: `scripts/eval/make-gsm8k-200.sh` from the repository root — expect `wrote tests/eval/gsm8k-200.jsonl (200 items, commit 3101c7d5072418e28b9008a6636bde82a006892c)`; `shasum -a 256 tests/eval/gsm8k-200.jsonl` — expect `6b7bab085cd8d5a484e59283d809525bf5e3e21edc1d3275e4b9517316775321` (jq 1.7); `head -c 120 tests/eval/gsm8k-200.jsonl` — expect it to start `{"id":"gsm8k-test-0000","messages":[{"role":"user","content":"Janet’s ducks lay 16 eggs per day.`.
- [ ] Write `tests/eval/NOTICE`: names `tests/eval/gsm8k-200.jsonl` as derived from the first 200 items of the GSM8K test split (`grade_school_math/data/test.jsonl`) of https://github.com/openai/grade-school-math at commit `3101c7d5072418e28b9008a6636bde82a006892c`, states the transformation (question wrapped in one user message with the answer-format instruction; answer = the number after `#### ` without thousands separators) and the regeneration script, then reproduces the full MIT License text with `Copyright (c) 2021 OpenAI`.
- [ ] Run: `cargo test -p turbine-bench --test golden eval_task_set_valid` — expect PASS.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `test(bench): commit the GSM8K-200 eval task set with its generator and NOTICE`

## Task 7: Vendor-neutral public API guard

Files: `crates/turbine-kernels/tests/vendor_neutral_api.rs` (new), `crates/turbine-kernels/Cargo.toml` (dev-dependency `syn.workspace = true`), `Cargo.toml` (workspace dependency `syn = { version = "3.0.6", features = ["full", "visit"] }`).
Interfaces:

- produces (test-local) `const CORE_CRATES: &[&str] = &["turbine-kernels", "turbine-tensor", "turbine-scheduler", "turbine-kv", "turbine-reliability"]`
- produces (test-local) `const VENDOR_WORDS: &[&str] = &["cuda", "hip", "rocm", "nccl", "rccl", "cublas", "sycl"]` (plus the word pair `level`,`zero` / `levelzero`)
- produces (test-local) `const BACKEND_ENUMS: &[&str] = &["ExecutionBackend", "CollectiveBackendKind", "ProviderKind"]` (only their variants may carry vendor names)
- produces (test-local) `fn words(ident: &str) -> Vec<String>`, `fn is_vendor_ident(ident: &str) -> bool`, `fn item_leaks(items: &[syn::Item], dir: Option<&Path>, where_: &str, leaks: &mut Vec<String>)`, `fn parse(file: &Path) -> syn::File`, `fn crate_src(name: &str) -> PathBuf`
- reads `crates/{turbine-kernels,turbine-tensor,turbine-scheduler,turbine-kv,turbine-reliability}/src` from `CARGO_MANIFEST_DIR/..`; nothing consumed by code — every track plan keeps it green (new kernels go behind the vendor-neutral traits)

Covers: S-5 — `cargo test -p turbine-kernels --test vendor_neutral_api` (tests `vendor_neutral_api` and `vendor_check_flags_leaks`). Intel re-entry needs no code: no Intel variant, discovery backend or build path is added (S-5, Out of scope).
Depends on: phase-1 (core crates exist), phase-5/phase-6 (`CollectiveBackendKind`, `ProviderKind`).

- [ ] Write failing test `vendor_check_flags_leaks`: on an inline `syn::parse_quote!` fixture, `item_leaks` reports `CudaStream` (pub fn arg), `hip::Event` (pub field), `pub use self::ffi::HipStreamRaw`, `NcclError` (trait impl), `RcclComm` (pub trait method), `level_zero::Device` (pub type) and `fixture::inner: CublasLtHandle` (nested pub mod), but not `HipRaw` (private field), `Ownership`, `Relationship` or `ExecutionBackend` (incl. `ExecutionBackend::Hip`); `is_vendor_ident` is false for `ship_date`, `Relationship`, `chip` and true for `hipblasLtHandle`, `cudaStream_t`, `RocmPath`.
- [ ] Write failing test `vendor_neutral_api`: walking each core crate from `src/lib.rs` through `pub mod` files finds no leak (message `vendor types in core public signatures:` + list). Run: `cargo test -p turbine-kernels --test vendor_neutral_api` — expect FAIL ("cannot find module or crate `syn`").
- [ ] Implement the guard: `words` splits on `_` and camel-case boundaries (keeping acronym runs together); `is_vendor_ident` matches a word equal to a vendor word or, except `hip`, starting with one, or starting with `hipblas`, or the `level`,`zero` pair; a `syn::visit::Visit` path collector flags the first vendor segment unless it follows a `BACKEND_ENUMS` segment; `item_leaks` visits signatures of pub fns, pub fields of pub structs, all enum variant fields, pub union fields, pub traits (generics, supertraits, fn signatures, assoc type bounds, consts), pub type/const/static types, trait-impl paths and self types plus pub/trait impl fn signatures, `pub use` trees, and recurses into inline and file `pub mod`s (`<name>.rs` or `<name>/mod.rs`, panicking when neither exists).
- [ ] Add `syn = { version = "3.0.6", features = ["full", "visit"] }` under `[workspace.dependencies]` in `Cargo.toml` and `syn.workspace = true` under `[dev-dependencies]` in `crates/turbine-kernels/Cargo.toml`.
- [ ] Run: `cargo test -p turbine-kernels --test vendor_neutral_api` — expect PASS. If `vendor_neutral_api` lists a leak from an earlier phase, move that type behind a vendor-neutral name (e.g. an opaque handle in the private `turbine_kernels::ffi`) — never weaken `VENDOR_WORDS`.
- [ ] Mutation check (do not commit): append `pub struct HipStream; impl DeviceBuffer { pub fn hip_stream(&self) -> HipStream { HipStream } }` to `crates/turbine-tensor/src/buffer.rs`, run `cargo test -p turbine-kernels --test vendor_neutral_api` — expect FAIL naming `turbine_tensor::buffer: HipStream`; revert.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `test(kernels): guard core public signatures against vendor-specific types`

## Task 8: Track start gate script

Files: `scripts/track-gate.sh` (new, mode 755, bash 3.2-compatible: order + spec-check + scope checks; ported from the run-ahead's `scripts/phase8-track-gate.sh` with the amended rules below), `benches/turbine-bench/tests/lab_scripts.rs` (append `mod track_gate_script`; `tempfile` dev-dep if absent), `AGENTS.md` (Commands: track entries).
Interfaces:

- produces `scripts/track-gate.sh <phase-6-quantization|phase-7-model-families|phase-8-speculative-decoding>` → `GATE PASS <track>` exit 0; one `GATE FAIL <track>: …` line per failed check plus `GATE FAIL <track>: <n> check(s) failed`, exit 1; usage exit 2
- produces env overrides `TURBINE_PROCODER_LAUNCHER` (default `$HOME/.claude/plugins/cache/procoder/procoder/3.7.0/hooks/launcher.sh`), `TURBINE_SPEC_DIR` (default `.procoder/specs`), `TURBINE_SUPPORT_MATRIX` (file holding `--support-matrix --output text`; default produced with `cargo run -q -p turbine-server`)
- produces the closed-track rule, on `amd` rows only: phase 6 closed ⇔ a `supported` `amd` row with a non-bf16 weight format or `fp8_e4m3` KV; phase 7 closed ⇔ a `supported` `amd` row whose architecture is neither `LlamaForCausalLM` nor `OlmoeForCausalLM`
- consumes Task 3 `turbine-server --support-matrix --output text` column layout, the procoder launcher `spec check <name>` (prints `… COMPLETE …`, exit 0), `.procoder/specs/<track>.md` with a `Status:` line and a `## In scope` section

Covers: S-1 (mechanism used by Tasks 9, 11, 12); test `cargo test -p turbine-bench --test lab_scripts track_gate`.
Depends on: Task 3.

- [ ] Write failing test `track_gate_script::track_gate`: with a stub launcher (COMPLETE iff the spec has the line `Status: complete`), temp specs and matrix files (baseline row; + an `amd gfx1201 LlamaForCausalLM fp8 … supported` row; + an `amd gfx1201 Qwen3ForCausalLM bf16 … supported` row; + an `nvidia sm_121 … fp8 … supported` row that must not count): an unknown track exits 2; a missing phase-6 spec exits 1 with `does not exist`; a complete in-scope phase-6 spec passes with `GATE PASS phase-6-quantization` (NVFP4/GGUF under "Out of scope" are fine), fails naming `NVFP4` when In scope lists it and fails with `not COMPLETE` and `Status line` when `Status: draft`; phase 7 fails with `phase-6-quantization has not closed` on the baseline matrix and on the nvidia-only matrix, passes once the AMD quantized row exists, and fails naming `gpt-oss` when In scope drops it; phase 8 fails with `phase-7-model-families has not closed` until the Qwen3 row exists, then passes, and fails naming `EAGLE` when In scope mentions it. Run: `cargo test -p turbine-bench --test lab_scripts track_gate` — expect FAIL (`left: Some(127)`).
- [ ] Implement `scripts/track-gate.sh` (`set -euo pipefail`, failures counted, all printed before exiting): the order check counts `supported` rows (column 7) with column 1 = `amd` and — for phase 7 — column 4 ≠ `bf16` or column 5 = `fp8_e4m3`, or — for phase 8 — column 3 ∉ {`LlamaForCausalLM`, `OlmoeForCausalLM`}, and fails with `track <prev> has not closed: …`; a missing spec fails with `<spec> does not exist; write it with /procoder:spec <track>`; a launcher `spec check` without `COMPLETE` fails with `procoder spec check is not COMPLETE: <first line>`; a missing `Status: complete` line fails with `Status line is not 'Status: complete'`; an empty `## In scope` section fails.
- [ ] Implement the scope checks, matching the `## In scope` section case-insensitively with `grep -Ei`: phase 6 — the spec names `fp8_block|mxfp4|awq_int4|gptq_int4|fp8_e4m3` somewhere, In scope needs `fp8`, `fp8_e4m3|kv\.dtype`, `mxfp4`, `awq`, `gptq`, and refuses `nvfp4`, `gguf`, `int8`; phase 7 — In scope needs `qwen3 dense|Qwen3ForCausalLM`, `qwen3 moe|Qwen3MoeForCausalLM`, `gpt-oss`, `qwen3\.5|qwen3\.6|qwen3_5`, `gated deltanet`, `mistral`, `mixtral`, `linear-attention.*(amd|gfx1201|hip|rocm)|(amd|gfx1201|hip|rocm).*linear-attention`; phase 8 — the spec names `Llama-3.2-1B-Instruct`, In scope needs `recurrent`, and refuses `(^|[^a-z])mtp([^a-z]|$)`, `eagle`, `dflash`; messages read `In scope does not cover <what> (/<regex>/)` and `In scope names <what>, outside the umbrella scope`.
- [ ] Run: `cargo test -p turbine-bench --test lab_scripts track_gate` — expect PASS; `shfmt -d scripts/track-gate.sh scripts/eval/make-gsm8k-200.sh` — expect no output; `scripts/track-gate.sh phase-7-model-families` on the real tree — expect `GATE FAIL phase-7-model-families: track phase-6-quantization has not closed: …` and exit 1.
- [ ] Append to the `## Commands` section of `AGENTS.md` these two bullets verbatim (the support-matrix bullet is already there):

```markdown
- Quality gate: `cargo run -q -p turbine-bench --bin turbine-golden -- eval --url <base> --tasks tests/eval/gsm8k-200.jsonl --output json > tests/eval/<model-slug>/<engine>.json`, then `turbine-golden eval-compare --baseline <bf16.json> --candidate <quantized.json>` (exit 1 when the drop exceeds `quality.max_accuracy_drop`, default 0.01).
- Track start (Phases 6–8): `scripts/track-gate.sh <phase-6-quantization|phase-7-model-families|phase-8-speculative-decoding>` must print `GATE PASS <track>` before a track's implementation starts; tracks close with the runbook in `.procoder/plans/phase-6-8-expansion.md` Task 10.
```

- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commit: `feat(scripts): add the phase 6–8 track start gate`

## Task 9: Gate — `phase-6-quantization` spec written and checked before track 1 starts

Files: `.procoder/specs/phase-6-quantization.md` (new, written through `/procoder:spec`).
Interfaces:

- consumes Task 8 `scripts/track-gate.sh`, the umbrella S-2 … S-6
- produces the track 1 spec that the separate `phase-6-quantization` plan implements; it cites the umbrella for S-2 … S-5 instead of restating them and reserves only `kv.dtype` (adds `fp8_e4m3`) among config keys

Covers: S-1, S-6 — acceptance criterion "Before track 1 implementation starts: `launcher.sh spec check phase-6-quantization` exits 0 reporting COMPLETE, Status `complete`, `grep -c -E "fp8_block|mxfp4|awq_int4|gptq_int4|fp8_e4m3"` non-zero while In scope names no format outside S-6".
Depends on: Tasks 1–8.

- [ ] Red: run `scripts/track-gate.sh phase-6-quantization` — expect FAIL with "does not exist; write it with /procoder:spec phase-6-quantization".
- [ ] Write the spec with `/procoder:spec phase-6-quantization` (interview the user; every open question goes to the user). Fixed inputs to carry into it: In scope = exactly S-6 (FP8 e4m3 weights per-tensor / per-channel `fp8` and block-scaled `fp8_block`; FP8 e4m3 KV as `kv.dtype: fp8_e4m3`; MXFP4 `mxfp4`, weight-only on RDNA4; INT4 `awq_int4` and `gptq_int4`, weight-only, group-wise); the spec decides the checkpoint containers per value, the Llama-3.2-3B / OLMoE checkpoints that prove each format, W8A8 vs W8A16 for FP8, the kernel provider per format after a provider evaluation (reuse-first rule), how MXFP4 is proven (offline-quantized fixture checkpoint or gpt-oss-20b in track 2; umbrella Constraints (c)), and the first code task that renames the support-matrix refusal reasons, turns the `nvidia` baseline rows `unsupported` and adds the five weight-format values (Task 2 amendment); it names `scripts/lab/phase6-quantization-novanas.yaml` on port 18000 and the exact `SUPPORT_MATRIX` rows it will turn `supported` (Task 10 procedure); NVFP4, INT8 and GGUF appear only under "Out of scope" (the gate refuses them inside "## In scope").
- [ ] Run: `"/Users/pascal/.claude/plugins/cache/procoder/procoder/3.7.0/hooks/launcher.sh" spec check phase-6-quantization` — expect `spec phase-6-quantization: COMPLETE` and exit 0.
- [ ] Run: `grep -x 'Status: complete' .procoder/specs/phase-6-quantization.md` — expect `Status: complete`; `grep -c -E "fp8_block|mxfp4|awq_int4|gptq_int4|fp8_e4m3" .procoder/specs/phase-6-quantization.md` — expect a number ≥ 1.
- [ ] Green: run `scripts/track-gate.sh phase-6-quantization` — expect `GATE PASS phase-6-quantization`; paste the four outputs into this task's todo evidence.
- [ ] Format: `prettier --write .procoder/specs/phase-6-quantization.md`; rerun the spec check — expect COMPLETE.
- [ ] Commit: `docs(spec): add the phase-6-quantization track spec`
- [ ] Hand-off: start the track with `/procoder:plan phase-6-quantization` (a separate plan; not part of this one).

## Task 10: Track close runbook (run once per track, in S-1 order)

Files: `crates/turbine-core/src/support.rs` (the track's rows flip to `supported` — the edit itself is the last task of the track's own plan; this task verifies it), `tests/eval/<model-slug>/<engine>.json` (eval reports for lossy formats, new per gated combination), `tests/bench/<track>/novanas-<model-slug>-<engine>.json` (bench reports, new).
Interfaces:

- consumes Task 3 `--support-matrix --output json`, Task 5 `turbine-golden eval`/`eval-compare`, Task 6 `tests/eval/gsm8k-200.jsonl`, P1 `turbine-golden compare`, P0/P3 `turbine-bench`
- consumes P3 `scripts/overload-soak.sh novanas [--duration <dur>] [--model <path>]` with the requested addition `--config <yaml>` (see report)
- consumes `scripts/lab-serve.sh novanas <config>` and `scripts/lab-serve.sh novanas --vllm <slug>` (vLLM-ROCm on :18100), the track spec's `scripts/lab/phase<N>-<topic>-novanas.yaml` (port 18000)
- consumes `tests/golden/<model-slug>/{reference.jsonl,tolerance.json}` (phase-1 format; for quantized checkpoints captured once from the checkpoint's reference runtime by the track plan)
- `<track>` ∈ {`phase-6-quantization`, `phase-7-model-families`, `phase-8-speculative-decoding`}; host `novanas` → `http://192.168.10.203:18000`

Covers: S-3, S-6, S-7, S-8 — acceptance criterion "Manual track close, per track in S-1 order, on `novanas`: overload-soak, golden compare, bench report, eval-compare for lossy formats, then `--support-matrix` shows the new `amd` rows `supported`".
Depends on: Tasks 2–9 (and Task 11 / Task 12 for tracks 2 and 3); the track's own plan fully closed.

- [ ] Precondition: `scripts/track-gate.sh <track>` printed `GATE PASS <track>` before the track started, and every task of the track's own plan is closed with `scripts/gate.sh --full` and `scripts/lab-test.sh novanas --tier full` green (exit 0).
- [ ] ASK THE USER FIRST when `amd.com/gpu` on `novanas` is held by another workload (standing approvals cover the Turbine lab Jobs while the cards are free); never evict someone else's workload.
- [ ] Serve the track configuration: `scripts/lab-serve.sh novanas scripts/lab/phase<N>-<topic>-novanas.yaml`; expect `curl -fsS http://192.168.10.203:18000/ready` → `{"ready":true}` and `curl -fsS http://192.168.10.203:18000/turbine/v1/status | jq .support` → the row being gated (status `experimental` while unvalidated rows are experimental in the track branch).
- [ ] Golden: `cargo run -q -p turbine-bench --bin turbine-golden -- compare --url http://192.168.10.203:18000 --reference tests/golden/<model-slug>/reference.jsonl` at `--concurrency 1` and `--concurrency 16` — expect exit 0 under the slug's `tolerance.json`.
- [ ] Quality (lossy formats only — every quantized weight format and `fp8_e4m3` KV): `cargo run -q -p turbine-bench --bin turbine-golden -- eval --url http://192.168.10.203:18000 --tasks tests/eval/gsm8k-200.jsonl --output json > tests/eval/<model-slug>/turbine-novanas.json` for the quantized row, the same against the BF16 model of the same family (or `scripts/lab-serve.sh novanas --vllm <model-slug>` on :18100 for the same checkpoint) into `tests/eval/<baseline-slug>/<engine>-novanas.json`, then `cargo run -q -p turbine-bench --bin turbine-golden -- eval-compare --baseline tests/eval/<baseline-slug>/<engine>-novanas.json --candidate tests/eval/<model-slug>/turbine-novanas.json` — expect `… : PASS` and exit 0.
- [ ] Bench: `cargo run -q -p turbine-bench --bin turbine-bench -- --url http://192.168.10.203:18000 --output json > tests/bench/<track>/novanas-<model-slug>-turbine.json` on GPU 0 — expect exit 0; where vLLM-ROCm serves the checkpoint, `scripts/lab-serve.sh novanas --vllm <model-slug>` and the same command against `http://192.168.10.203:18100` into `…-vllm.json`, then `scripts/lab-serve.sh novanas --stop`; otherwise record the Turbine report as the baseline (TS §18).
- [ ] Soak: `scripts/overload-soak.sh novanas --config scripts/lab/phase<N>-<topic>-novanas.yaml` — expect exit 0 and a passing JSON verdict under `target/soak/`.
- [ ] No regression of TS §20 first-useful-release items on AMD: with `scripts/lab/phase2-novanas-llama.yaml` and `phase2-novanas-olmoe.yaml` served in turn, `turbine-golden compare --url … --reference tests/golden/llama-3.2-3b-instruct/reference.jsonl` and `… olmoe-1b-7b-0125-instruct/reference.jsonl` — expect exit 0 each.
- [ ] Flip the rows (last task of the track's plan): the track's `amd` rows turn `supported` where all four items passed; `nvidia` rows stay `unsupported` naming `phase-2b-nvidia`; `cargo test -p turbine-core support::` — expect PASS (update `baseline_rows_present` expectations in the same commit).
- [ ] Verify: `cargo run -q -p turbine-server -- --support-matrix --output json | jq -c '.rows[] | select(.status == "supported")'` — expect the AMD baseline rows plus exactly the tracks' validated rows.
- [ ] Record: paste every command output above into the track-close todo's evidence (`/procoder:todo`); `scripts/lab-serve.sh novanas --stop`.
- [ ] Commit: `test(<track>): record the exit-gate reports`

## Task 11: Gate — `phase-7-model-families` spec written and checked before track 2 starts

Files: `.procoder/specs/phase-7-model-families.md` (new, written through `/procoder:spec`).
Interfaces:

- consumes Task 8 gate script, Task 10 executed for `phase-6-quantization` (≥ 1 `supported` `amd` quantized or `fp8_e4m3` row)
- consumes contract §10 `registry` (`ArchitectureEntry { architectures0, model_type, parse, weight_map, build_executor, kv_layout }`, `lookup(architectures0, model_type)`) and the Phase 2m families (`turbine_model::families::{Qwen3, Qwen3Moe, Mistral, Mixtral}`, CPU execution)
- produces the track 2 spec implemented by the separate `phase-7-model-families` plan

Covers: S-1, S-8 — acceptance criterion "Before track 2 implementation starts (and only after track 1 closed): spec check COMPLETE with Status `complete`, covering Qwen3 dense, Qwen3 MoE, gpt-oss-20b, the Qwen3.5/3.6 hybrids with the AMD linear-attention kernel provider decided, Mistral and Mixtral, with the checkpoint each family is served from".
Depends on: Task 8; Task 10 run for `phase-6-quantization`.

- [ ] Red: run `scripts/track-gate.sh phase-7-model-families` — expect FAIL with "does not exist" (and, until track 1 closed, "track phase-6-quantization has not closed").
- [ ] Write the spec with `/procoder:spec phase-7-model-families` (interview the user). Fixed inputs: In scope = exactly S-8 (Qwen3 dense, Qwen3 MoE, gpt-oss-20b with MXFP4 experts, attention sinks and alternating sliding-window attention, the Qwen3.5/3.6 Gated DeltaNet hybrids text-only with vision weights skipped and their recurrent/conv-state layout, Mistral and Mixtral); the AMD linear-attention kernel provider after a provider evaluation (the gate looks for `linear-attention` together with `AMD`/`gfx1201`/`hip`/`rocm` in "## In scope"); per family: the checkpoint it is served from (FP8 or BF16 for the hybrids, whose cached checkpoints are NVFP4; FP8 + TP 2 where one card does not hold it, e.g. Mixtral-8x7B and Qwen3.6-35B-A3B), its golden fixtures under `tests/golden/<model-slug>/`, and one card or Phase 5 TP; the tests that use `GptOssForCausalLM` as the example of an unregistered architecture move to another name; recurrent-state rollback for speculation stays with track 3. Carried in from Phase 6a (user decision 2026-09-30, `.procoder/ask/decisions.md` "MXFP4-A16 8B: `mxfp4` status for 6a after the p05 incremental spread"): an item that traces Turbine's CPU path against transformers op by op on prompt p05 of `tests/golden/llama-3.1-8b-instruct-mxfp4a16/` (where intermediates round to BF16; diagnostic `mxfp4_decode_vs_prefill_trace` in `crates/turbine-model/tests/golden.rs`), since the same rounding may affect other families; the `mxfp4` row flips only after it.
- [ ] Run: `"/Users/pascal/.claude/plugins/cache/procoder/procoder/3.7.0/hooks/launcher.sh" spec check phase-7-model-families` — expect `COMPLETE`, exit 0; `grep -x 'Status: complete' .procoder/specs/phase-7-model-families.md` — expect a match.
- [ ] Green: `scripts/track-gate.sh phase-7-model-families` — expect `GATE PASS phase-7-model-families`; paste the outputs into this task's todo evidence.
- [ ] Format: `prettier --write .procoder/specs/phase-7-model-families.md`; rerun the spec check — expect COMPLETE.
- [ ] Commit: `docs(spec): add the phase-7-model-families track spec`
- [ ] Hand-off: `/procoder:plan phase-7-model-families` (separate plan); close it with Task 10.

## Task 12: Gate — `phase-8-speculative-decoding` spec written and checked before track 3 starts

Files: `.procoder/specs/phase-8-speculative-decoding.md` (new, written through `/procoder:spec`).
Interfaces:

- consumes Task 8 gate script, Task 10 executed for `phase-7-model-families` (≥ 1 `supported` `amd` row of a track 2 architecture)
- consumes Task 1 `SpeculativeConfig`/`SpeculativeMethod` (the track adds `num_tokens` ≤ 8, `draft_model_path`, `min_acceptance`), Task 2 draft refusal row
- produces the track 3 spec implemented by the separate `phase-8-speculative-decoding` plan

Covers: S-1, S-7 — acceptance criterion "Before track 3 implementation starts (and only after track 2 closed): spec check COMPLETE with Status `complete`, names `Llama-3.2-1B-Instruct` as the first draft model, covers recurrent-state rollback for the hybrid targets, and has no MTP, EAGLE or DFlash proposer".
Depends on: Task 8; Task 10 run for `phase-7-model-families`.

- [ ] Red: run `scripts/track-gate.sh phase-8-speculative-decoding` — expect FAIL with "does not exist" (and, until track 2 closed, "track phase-7-model-families has not closed").
- [ ] Write the spec with `/procoder:spec phase-8-speculative-decoding` (interview the user). Fixed inputs: In scope = exactly S-7 (separate draft model sharing the target's tokenizer behind a `Proposer` trait, first pairing `meta-llama/Llama-3.2-1B-Instruct` → `meta-llama/Llama-3.2-3B-Instruct`, slug `llama-3.2-1b-instruct`; verifier in `turbine-scheduler` scoring k ≤ 8 proposals in one target forward with standard speculative rejection sampling, greedy = exact prefix match; rollback by truncating KV blocks past the last accepted position, and rollback of the hybrids' recurrent state; per-request disable below `speculative.min_acceptance`, global disable at pressure ORANGE or worse; draft memory reserved through the Phase 3 budget); config keys exactly `speculative.{method, num_tokens, draft_model_path, min_acceptance}`; track metrics with bounded labels; colocated only; MTP, EAGLE and DFlash only under "Out of scope".
- [ ] Run: `"/Users/pascal/.claude/plugins/cache/procoder/procoder/3.7.0/hooks/launcher.sh" spec check phase-8-speculative-decoding` — expect `COMPLETE`, exit 0; `grep -x 'Status: complete' .procoder/specs/phase-8-speculative-decoding.md` — expect a match; `grep -c 'Llama-3.2-1B-Instruct' .procoder/specs/phase-8-speculative-decoding.md` — expect ≥ 1.
- [ ] Green: `scripts/track-gate.sh phase-8-speculative-decoding` — expect `GATE PASS phase-8-speculative-decoding`; paste the outputs into this task's todo evidence.
- [ ] Format: `prettier --write .procoder/specs/phase-8-speculative-decoding.md`; rerun the spec check — expect COMPLETE.
- [ ] Commit: `docs(spec): add the phase-8-speculative-decoding track spec`
- [ ] Hand-off: `/procoder:plan phase-8-speculative-decoding` (separate plan); close it with Task 10; the expansion umbrella closes when Task 10 has run for all three tracks.
