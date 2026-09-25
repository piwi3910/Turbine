# phase-8-expansion — implementation plan

Status: draft
Spec: .procoder/specs/phase-8-expansion.md

## Goal

Deliver the mechanisms every Phase 8 track shares — the support matrix with startup refusal, the `turbine-golden eval`/`eval-compare` quality gate and its committed GSM8K task set, the vendor-neutrality guard, and the scripted track-start gate and track-close procedure — so that `phase-8a-quantization`, `phase-8b-speculative-decoding` and `phase-8c-model-families` can each be specified, gated and closed in order without re-deciding shared rules.

## Architecture

`turbine_core::support` holds the static `SUPPORT_MATRIX` table and its resolution rules (most specific row wins; `--check-config` resolves with unknown device columns); `turbine-server` prints it (`--support-matrix`), resolves it under `--check-config` (exit 2) and at startup step 3 after discovery (exit 1, WARN for experimental), puts the resolved row under `support` in `GET /turbine/v1/status` and sets `turbine_support_matrix_status{status}` through `turbine_api::support::SupportMetrics`. `turbine-bench` gains `golden::eval` behind `turbine-golden eval`/`eval-compare` plus the committed `tests/eval/gsm8k-200.jsonl`; `turbine-kernels/tests/vendor_neutral_api.rs` parses the five core crates with `syn` and rejects vendor type paths in public signatures. Track sequencing is enforced by `scripts/phase8-track-gate.sh` (previous track closed per the support matrix, track spec COMPLETE and inside the umbrella scope) and a per-track close runbook (Task 10); the track contents themselves are planned by each track's own plan.

## Constraints

From the spec (verbatim):

- Rust only in the serving path; no Python at runtime (TS §21 rule 4). Python is allowed only at fixture-generation time (HF transformers dumps, reference-runtime captures), never in a build or test that `cargo test` runs. Vendor libraries are loaded at runtime through the prebuilt kernel libraries (_libturbine_hip.so_, _libturbine_cuda.so_, CMake-built), never linked by Cargo, so the workspace still builds and every non-ignored test passes on macOS arm64 with no GPU libraries.
- `unsafe` stays in `turbine-device`, `turbine-kernels` and the Phase 7 transport module.
- The kernel C ABI is shared by both vendors' libraries; any additive change bumps `turbine_abi_version` for both together.
- Every lossy format or lossy KV transform ships with its quality gate (TS §8, S-3); nothing lossy is enabled by default unless the checkpoint itself is stored in that format.
- Bounded inputs (TS §16, §21 rule 8): speculative k ≤ 8; draft-model memory is reserved through the Phase 3 budget before speculation is enabled.
- Model weights live in `/home/piwi/turbine-models/<slug>` on each host, downloaded by Claude with the user's HF token at that time (never stored in the repo); tests read `TURBINE_TEST_MODEL_DIR` and never download.
- **Host workloads:** any lab run that needs production workloads (production vLLM on the Sparks, anything using the R9700 cards on `novanas`) moved or memory freed on any host is started only after the implementer has asked the user and the user has moved the workloads; the implementer never stops, moves or reconfigures production workloads itself. Reference-engine captures for a checkpoint other than the one production serves use a temporary container named `turbine-ref-*` within the free memory measured at run start, removed afterwards.
- Lab hosts: `novanas` (192.168.10.203), 2× Radeon AI PRO R9700 (`gfx1201`, 32 GB each), ROCm 7.14.1 at `/opt/rocm/rocm`, k3s Jobs requesting `amd.com/gpu`, 10 GbE; `dgx-spark` (192.168.10.246) and `dgx-spark2` (192.168.10.245), 1× GB10 (`sm_121`) each, ~121 GB unified memory, Docker, RoCE between them. Families that do not fit one device (e.g. Mixtral in BF16) rely on Phase 5 TP or Phase 7 PP/EP, or on an in-scope quantized checkpoint; the track 3 spec states which per family.
- Cached checkpoints that define the quantization scope (from their _config.json_, verified 2026-09-25): `nvidia/Qwen3.6-35B-A3B-NVFP4` (`qwen3_5_moe`, modelopt mixed precision, FP8 KV); `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` (`qwen3_5` dense hybrid, modelopt); `RadixArk/Qwen3.8-Flash-Next-NVFP4` (`qwen4_exp`, 126 GB — does not fit one Spark); `YourHighnessLA/Qwen3.8-27B-DFlash2-NVFP4` (compressed-tensors `nvfp4-pack-quantized`).

From the interface contract (`.procoder/contract/interfaces.md`, binding):

- Toolchain: edition 2024, `rust-version = "1.97"`, `license = "Apache-2.0"` inherited from `[workspace.package]`; every crate but the four allowlisted ones sets `[lints.rust] unsafe_code = "forbid"`; this phase adds no `unsafe`.
- Every public enum a later phase extends is `#[non_exhaustive]`; every config struct is `#[serde(deny_unknown_fields, default)]`; every crate has one top-level `thiserror` error enum.
- Metric label values come from closed sets rendered by `as_str()`; `turbine_support_matrix_status{status}` ∈ supported, experimental, unsupported.
- Validation order (§3.2/§16.3): static `validate()` (exit 2) → device discovery → `validate_host` → P5 parallel plan (exit 2) → **P8 support-matrix resolution** (exit 1 at startup / exit 2 under `--check-config`) → kernel library/model/budget (exit 1).
- Tests: unit tests in `#[cfg(test)] mod tests`, addressed `cargo test -p <crate> <module>::tests::<name>`; integration tests `crates/<crate>/tests/<binary>.rs`, addressed `cargo test -p <crate> --test <binary> <name>`; anything needing a GPU, weights or a lab host is `#[ignore]` and runs via `scripts/lab-test.sh <host>`; non-ignored tests pass on macOS arm64 with no GPU libraries and no weights. Phase-8 additions to an existing integration-test file go into their own `mod phase8_…` block so they never collide with earlier helpers or imports (the `--test <binary> <name>` filter still matches).
- Gate after every task: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`, then `cargo test --workspace`.
- Lab rules (§21.1): no `docker run` outside the defined lab scripts; scripts never stop/restart/reconfigure non-`turbine-lab-*` workloads; any run needing production workloads moved or memory freed is asked of the user first; Spark correctness runs proceed after the MemAvailable precondition, benchmark/soak/overload runs always ask.

## Task 1: `quality` and `speculative.method` configuration keys

Files: `crates/turbine-core/src/config/quality.rs` (new: `QualityConfig`, its validation and unit test), `crates/turbine-core/src/config/speculative.rs` (new: `SpeculativeConfig`, `SpeculativeMethod` and unit test), `crates/turbine-core/src/config/mod.rs` (declare both modules, re-export, add the two `Config` fields, call `self.quality.validate()?` in `Config::validate`)
Interfaces: produces `turbine_core::config::{QualityConfig { max_accuracy_drop: f64 }, SpeculativeConfig { method: SpeculativeMethod }, SpeculativeMethod { None, Draft }}` (`#[non_exhaustive]`, serde `"none"`/`"draft"`, `as_str()`), `turbine_core::config::quality::DEFAULT_MAX_ACCURACY_DROP = 0.01`, `QualityConfig::validate(&self) -> Result<(), ConfigError>` (`ConfigError::Invalid { key: "quality.max_accuracy_drop", .. }` outside 0..=0.1); `Config.quality`, `Config.speculative` (contract §3.2 names). The phase-8b plan adds `num_tokens`, `draft_model_path`, `min_acceptance` to `SpeculativeConfig`. Consumes P0 `ConfigError`, `Config`, `Config::validate`.
Covers: S-2 (the `speculative` column's config key), S-4 (`quality.max_accuracy_drop`); no acceptance criterion on its own (Tasks 2, 3 and 5 exercise these keys).

- [ ] Write the failing tests: create `crates/turbine-core/src/config/quality.rs` and `crates/turbine-core/src/config/speculative.rs` containing only their `#[cfg(test)] mod tests` blocks shown in the next step, and add `mod quality; mod speculative;` to `crates/turbine-core/src/config/mod.rs`. Run `cargo test -p turbine-core config::` — expect FAIL with "cannot find".
- [ ] Implement: complete `crates/turbine-core/src/config/quality.rs` exactly as:

```rust
//! `quality.*` (phase 8 umbrella, S-3/S-4): thresholds used by `turbine-golden eval-compare`.
use serde::{Deserialize, Serialize};

use super::ConfigError;

/// Default for `quality.max_accuracy_drop` (P8 §Interfaces).
pub const DEFAULT_MAX_ACCURACY_DROP: f64 = 0.01;

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct QualityConfig {
    /// Largest accuracy drop (absolute, 0..=0.1) a lossy format may show against its baseline.
    pub max_accuracy_drop: f64,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            max_accuracy_drop: DEFAULT_MAX_ACCURACY_DROP,
        }
    }
}

impl QualityConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        let v = self.max_accuracy_drop;
        if !v.is_finite() || !(0.0..=0.1).contains(&v) {
            return Err(ConfigError::Invalid {
                key: "quality.max_accuracy_drop".into(),
                reason: format!("must be between 0 and 0.1, got {v}"),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_config_validation() {
        assert_eq!(QualityConfig::default().max_accuracy_drop, 0.01);
        assert!(QualityConfig::default().validate().is_ok());
        for ok in [0.0, 0.05, 0.1] {
            assert!(
                QualityConfig {
                    max_accuracy_drop: ok
                }
                .validate()
                .is_ok(),
                "{ok}"
            );
        }
        for bad in [-0.001, 0.1001, f64::NAN, f64::INFINITY] {
            let err = QualityConfig {
                max_accuracy_drop: bad,
            }
            .validate()
            .unwrap_err();
            assert_eq!(err.key(), Some("quality.max_accuracy_drop"), "{bad}");
        }
    }
}
```

- [ ] Complete `crates/turbine-core/src/config/speculative.rs` exactly as:

```rust
//! `speculative.*`: the umbrella owns `method` only, because the support matrix (S-2) keys on it;
//! `num_tokens`, `draft_model_path` and `min_acceptance` are added by the phase-8b plan.
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum SpeculativeMethod {
    #[default]
    None,
    Draft,
}

impl SpeculativeMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            SpeculativeMethod::None => "none",
            SpeculativeMethod::Draft => "draft",
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields, default)]
pub struct SpeculativeConfig {
    pub method: SpeculativeMethod,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speculative_method_parses() {
        let cfg: SpeculativeConfig = serde_norway::from_str("method: draft").unwrap();
        assert_eq!(cfg.method, SpeculativeMethod::Draft);
        assert_eq!(SpeculativeConfig::default().method, SpeculativeMethod::None);
        assert!(serde_norway::from_str::<SpeculativeConfig>("method: eagle").is_err());
        assert!(serde_norway::from_str::<SpeculativeConfig>("num_tokens: 4").is_err());
    }
}
```

- [ ] Wire into `crates/turbine-core/src/config/mod.rs`: next to the other section modules add

```rust
mod quality;
mod speculative;
pub use quality::QualityConfig;
pub use speculative::{SpeculativeConfig, SpeculativeMethod};
```

add the fields `pub quality: QualityConfig,` (comment `// P8`) and `pub speculative: SpeculativeConfig,` (comment `// P8b reserved; P8 umbrella owns method`) to `pub struct Config` after `parallel`, and add `self.quality.validate()?;` to `Config::validate` after the P6 distributed rules. If the P0–P7 plans already declared placeholder `QualityConfig`/`SpeculativeConfig` structs in `config/mod.rs`, delete those declarations so these modules are the only definitions.

- [ ] Run `cargo test -p turbine-core -- config::quality::tests::quality_config_validation config::speculative::tests::speculative_method_parses` — expect PASS; then `cargo test -p turbine-core` — expect PASS (P0 `config::tests::example_config_loads` still loads because both sections default).
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`.
- [ ] Commit: `feat(core): add quality.max_accuracy_drop and speculative.method config keys`.

## Task 2: Support matrix in `turbine-core`

Files: `crates/turbine-core/src/support.rs` (new: table, resolution, refusal, table validation, unit tests), `crates/turbine-core/src/lib.rs` (add `pub mod support;`)
Interfaces: consumes Task 1 `Config.speculative.method`, P0 `ConfigError::Invalid`, P1 `turbine_core::types::ExecutionBackend { Hip, Cuda, Cpu }`. Produces (contract §3.8 plus the additions listed in the report):

```rust
pub const VENDORS: &[&str] = &["amd", "nvidia", "cpu"];
pub const WILDCARD: &str = "*";
pub enum WeightFormat { Bf16, ModeloptNvfp4, ModeloptFp8, ModeloptMixed, CtNvfp4 }  // as_str(), ALL
pub enum KvFormatColumn { Bf16, Fp8E4m3 }                                          // as_str(), ALL
pub enum SpeculativeColumn { None, Draft }                                         // as_str(), ALL
pub enum SupportStatus { Supported, Experimental, Unsupported { reason: Cow<'static, str> } } // as_str(), reason()
pub struct SupportKey { vendor: String, arch: String, architecture: String, weight_format, kv_format, speculative }
impl SupportKey { fn is_partial(&self) -> bool; fn blamed_config_key(&self) -> &'static str;
                  fn for_check_config(cfg: &Config) -> SupportKey; fn for_startup(cfg: &Config, arch: &str, architecture: &str) -> SupportKey }
pub struct SupportKeyPattern { vendor: Option<&'static str>, arch: Option<&'static str>, architecture: Option<&'static str>,
                               weight_format: Option<WeightFormat>, kv_format: Option<KvFormatColumn>, speculative: Option<SpeculativeColumn> }
pub struct SupportRow { key: SupportKeyPattern, status: SupportStatus }   // view() -> SupportRowView
pub struct SupportRowView { vendor, arch, architecture, weight_format, kv_format, speculative: String, status: &'static str, reason: Option<String> } // Serialize
pub struct SupportDecision { key: SupportKey, status: SupportStatus }     // warning() -> Option<String>, view() -> SupportRowView
pub static SUPPORT_MATRIX: &[SupportRow];
pub fn resolve(key: &SupportKey) -> SupportStatus;  pub fn resolve_in(table: &[SupportRow], key: &SupportKey) -> SupportStatus;
pub fn resolve_partial_in(table: &[SupportRow], key: &SupportKey) -> SupportStatus;
pub fn check(key: SupportKey) -> Result<SupportDecision, ConfigError>;  pub fn check_in(table: &[SupportRow], key: SupportKey) -> Result<SupportDecision, ConfigError>;
pub fn validate_table(table: &[SupportRow]) -> Result<(), String>;
pub fn vendor_column(backend: ExecutionBackend) -> &'static str;       // hip→amd, cuda→nvidia, cpu→cpu
pub fn config_columns(cfg: &Config) -> (WeightFormat, KvFormatColumn, SpeculativeColumn);  // phase-8a replaces the weight/KV derivation
```

Rows are added only by track plans once their exit gate passes (Task 10); a track replaces its `unsupported` refusal rows with validated rows and keeps `validate_table` green.
Covers: S-2 — `cargo test -p turbine-core support::tests::resolution_and_refusal`, `cargo test -p turbine-core support::tests::baseline_rows_present`.

- [ ] Write the failing tests: create `crates/turbine-core/src/support.rs` containing only the `#[cfg(test)] mod tests` block below (with `use super::*;`), add `pub mod support;` to `crates/turbine-core/src/lib.rs`, run `cargo test -p turbine-core support::` — expect FAIL with "cannot find".

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn key(
        vendor: &str,
        arch: &str,
        architecture: &str,
        w: WeightFormat,
        kv: KvFormatColumn,
        s: SpeculativeColumn,
    ) -> SupportKey {
        SupportKey {
            vendor: vendor.into(),
            arch: arch.into(),
            architecture: architecture.into(),
            weight_format: w,
            kv_format: kv,
            speculative: s,
        }
    }

    fn pat(
        vendor: Option<&'static str>,
        arch: Option<&'static str>,
        w: Option<WeightFormat>,
        s: Option<SpeculativeColumn>,
    ) -> SupportKeyPattern {
        SupportKeyPattern {
            vendor,
            arch,
            architecture: None,
            weight_format: w,
            kv_format: None,
            speculative: s,
        }
    }

    #[test]
    fn resolution_and_refusal() {
        use KvFormatColumn as K;
        use SpeculativeColumn as S;
        use WeightFormat as W;
        let table = [
            SupportRow {
                key: pat(Some("amd"), None, None, Some(S::None)),
                status: unsupported("amd needs a validated arch"),
            },
            SupportRow {
                key: pat(Some("amd"), Some("gfx1201"), None, Some(S::None)),
                status: SupportStatus::Supported,
            },
            SupportRow {
                key: pat(Some("nvidia"), Some("sm_121"), None, Some(S::None)),
                status: SupportStatus::Experimental,
            },
            SupportRow {
                key: pat(None, None, None, Some(S::Draft)),
                status: unsupported("speculative.method: draft not validated"),
            },
        ];
        validate_table(&table).unwrap();

        // Most specific row wins over the vendor wildcard row.
        let k = key(
            "amd",
            "gfx1201",
            "LlamaForCausalLM",
            W::Bf16,
            K::Bf16,
            S::None,
        );
        assert_eq!(resolve_in(&table, &k), SupportStatus::Supported);
        let k = key(
            "amd",
            "gfx1100",
            "LlamaForCausalLM",
            W::Bf16,
            K::Bf16,
            S::None,
        );
        assert_eq!(resolve_in(&table, &k).as_str(), "unsupported");

        // Unsupported makes validation fail with the reason and the combination.
        let k = key(
            "amd",
            "gfx1100",
            "LlamaForCausalLM",
            W::Bf16,
            K::Bf16,
            S::None,
        );
        let err = check_in(&table, k).unwrap_err().to_string();
        assert!(err.contains("amd needs a validated arch"), "{err}");
        assert!(err.contains("vendor=amd arch=gfx1100"), "{err}");

        // Experimental passes with a WARN.
        let k = key(
            "nvidia",
            "sm_121",
            "LlamaForCausalLM",
            W::Bf16,
            K::Bf16,
            S::None,
        );
        let d = check_in(&table, k).unwrap();
        assert_eq!(d.status, SupportStatus::Experimental);
        assert!(d.warning().unwrap().contains("experimental"));
        let k = key(
            "amd",
            "gfx1201",
            "LlamaForCausalLM",
            W::Bf16,
            K::Bf16,
            S::None,
        );
        assert_eq!(check_in(&table, k).unwrap().warning(), None);

        // No row at all is unsupported.
        let k = key(
            "nvidia",
            "sm_90",
            "LlamaForCausalLM",
            W::Bf16,
            K::Bf16,
            S::None,
        );
        assert_eq!(
            resolve_in(&table, &k).reason(),
            Some("no support-matrix row")
        );

        // Draft on the real table: refused, naming speculative.method, also for a partial key.
        let k = key(
            "amd",
            "gfx1201",
            "LlamaForCausalLM",
            W::Bf16,
            K::Bf16,
            S::Draft,
        );
        let err = check(k).unwrap_err();
        assert_eq!(err.key(), Some("speculative.method"));
        let k = key("amd", WILDCARD, WILDCARD, W::Bf16, K::Bf16, S::Draft);
        assert_eq!(check(k).unwrap_err().key(), Some("speculative.method"));
        let k = key("amd", WILDCARD, WILDCARD, W::Bf16, K::Bf16, S::None);
        assert_eq!(check(k).unwrap().status, SupportStatus::Supported);

        // A tie between two overlapping rows is rejected.
        let tie = [
            SupportRow {
                key: pat(Some("amd"), None, None, None),
                status: SupportStatus::Supported,
            },
            SupportRow {
                key: pat(None, Some("gfx1201"), None, None),
                status: unsupported("tie"),
            },
        ];
        assert!(
            validate_table(&tie)
                .unwrap_err()
                .contains("equal specificity")
        );
        let bad_vendor = [SupportRow {
            key: pat(Some("intel"), None, None, None),
            status: SupportStatus::Supported,
        }];
        assert!(validate_table(&bad_vendor).is_err());

        // The real table satisfies every invariant.
        validate_table(SUPPORT_MATRIX).unwrap();
    }

    #[test]
    fn baseline_rows_present() {
        use KvFormatColumn as K;
        use SpeculativeColumn as S;
        use WeightFormat as W;
        for (vendor, arch) in [("amd", "gfx1201"), ("nvidia", "sm_121")] {
            for architecture in ["LlamaForCausalLM", "OlmoeForCausalLM"] {
                let k = key(vendor, arch, architecture, W::Bf16, K::Bf16, S::None);
                let row = SUPPORT_MATRIX
                    .iter()
                    .filter(|r| r.key.matches(&k))
                    .max_by_key(|r| r.key.specificity())
                    .expect("baseline row");
                assert_eq!(
                    row.key.specificity(),
                    6,
                    "{k}: baseline rows are fully specific"
                );
                assert_eq!(resolve(&k), SupportStatus::Supported, "{k}");
            }
        }
        for r in SUPPORT_MATRIX
            .iter()
            .filter(|r| r.status == SupportStatus::Supported)
        {
            assert_eq!(r.key.weight_format, Some(W::Bf16), "{:?}", r.view());
            assert_eq!(r.key.kv_format, Some(K::Bf16), "{:?}", r.view());
            assert_eq!(r.key.speculative, Some(S::None), "{:?}", r.view());
        }
        for w in W::ALL.into_iter().filter(|w| *w != W::Bf16) {
            let k = key("nvidia", "sm_121", "LlamaForCausalLM", w, K::Bf16, S::None);
            assert_eq!(resolve(&k).as_str(), "unsupported", "{k}");
        }
        let k = key(
            "amd",
            "gfx1201",
            "LlamaForCausalLM",
            W::Bf16,
            K::Fp8E4m3,
            S::None,
        );
        assert_eq!(resolve(&k).as_str(), "unsupported");
        let k = key(
            "amd",
            "gfx1201",
            "LlamaForCausalLM",
            W::Bf16,
            K::Bf16,
            S::Draft,
        );
        assert_eq!(resolve(&k).as_str(), "unsupported");
        for r in SUPPORT_MATRIX {
            let v = r.view();
            assert!(
                VENDORS.contains(&v.vendor.as_str()) || v.vendor == WILDCARD,
                "{v:?}"
            );
            assert!(
                W::ALL.iter().any(|w| w.as_str() == v.weight_format) || v.weight_format == WILDCARD
            );
            assert!(K::ALL.iter().any(|k| k.as_str() == v.kv_format) || v.kv_format == WILDCARD);
            assert!(
                S::ALL.iter().any(|s| s.as_str() == v.speculative) || v.speculative == WILDCARD
            );
        }
    }
}
```

- [ ] Implement: put the following above the tests module in `crates/turbine-core/src/support.rs` (the table is the Phase 1–7 baseline, the CPU reference provider as `experimental`, and explicit refusal rows naming the track that may lift them):

```rust
//! Support matrix (phase 8 S-2): one declarative table of
//! `(vendor, arch, architecture, weight_format, kv_format, speculative) → status`.
//! Tracks add rows only when their exit gate (S-3) passed.
use std::borrow::Cow;
use std::fmt;

use serde::Serialize;

use crate::config::{Config, ConfigError, SpeculativeMethod};
use crate::types::ExecutionBackend;

/// Bounded `vendor` column values; `cpu` is the phase-1 CPU reference provider.
pub const VENDORS: &[&str] = &["amd", "nvidia", "cpu"];

/// Column value meaning "any" in a row and "not known yet" in a `--check-config` key.
pub const WILDCARD: &str = "*";

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum WeightFormat {
    Bf16,
    ModeloptNvfp4,
    ModeloptFp8,
    ModeloptMixed,
    CtNvfp4,
}

impl WeightFormat {
    pub const ALL: [WeightFormat; 5] = [
        WeightFormat::Bf16,
        WeightFormat::ModeloptNvfp4,
        WeightFormat::ModeloptFp8,
        WeightFormat::ModeloptMixed,
        WeightFormat::CtNvfp4,
    ];
    pub fn as_str(self) -> &'static str {
        match self {
            WeightFormat::Bf16 => "bf16",
            WeightFormat::ModeloptNvfp4 => "modelopt_nvfp4",
            WeightFormat::ModeloptFp8 => "modelopt_fp8",
            WeightFormat::ModeloptMixed => "modelopt_mixed",
            WeightFormat::CtNvfp4 => "ct_nvfp4",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum KvFormatColumn {
    Bf16,
    Fp8E4m3,
}

impl KvFormatColumn {
    pub const ALL: [KvFormatColumn; 2] = [KvFormatColumn::Bf16, KvFormatColumn::Fp8E4m3];
    pub fn as_str(self) -> &'static str {
        match self {
            KvFormatColumn::Bf16 => "bf16",
            KvFormatColumn::Fp8E4m3 => "fp8_e4m3",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SpeculativeColumn {
    None,
    Draft,
}

impl SpeculativeColumn {
    pub const ALL: [SpeculativeColumn; 2] = [SpeculativeColumn::None, SpeculativeColumn::Draft];
    pub fn as_str(self) -> &'static str {
        match self {
            SpeculativeColumn::None => "none",
            SpeculativeColumn::Draft => "draft",
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SupportStatus {
    Supported,
    Experimental,
    Unsupported { reason: Cow<'static, str> },
}

impl SupportStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            SupportStatus::Supported => "supported",
            SupportStatus::Experimental => "experimental",
            SupportStatus::Unsupported { .. } => "unsupported",
        }
    }
    pub fn reason(&self) -> Option<&str> {
        match self {
            SupportStatus::Unsupported { reason } => Some(reason),
            _ => None,
        }
    }
    /// Supported > Experimental > Unsupported (used by partial resolution).
    fn rank(&self) -> u8 {
        match self {
            SupportStatus::Supported => 2,
            SupportStatus::Experimental => 1,
            SupportStatus::Unsupported { .. } => 0,
        }
    }
}

/// The configured combination. `vendor`, `arch` and `architecture` may be [`WILDCARD`]
/// when they are not known yet (`--check-config` runs without device discovery).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SupportKey {
    pub vendor: String,
    pub arch: String,
    pub architecture: String,
    pub weight_format: WeightFormat,
    pub kv_format: KvFormatColumn,
    pub speculative: SpeculativeColumn,
}

/// One row's match pattern: `None` = `*`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SupportKeyPattern {
    pub vendor: Option<&'static str>,
    pub arch: Option<&'static str>,
    pub architecture: Option<&'static str>,
    pub weight_format: Option<WeightFormat>,
    pub kv_format: Option<KvFormatColumn>,
    pub speculative: Option<SpeculativeColumn>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SupportRow {
    pub key: SupportKeyPattern,
    pub status: SupportStatus,
}

/// JSON/text view of one row or one resolved decision (`--support-matrix`, `/turbine/v1/status`).
#[derive(Clone, PartialEq, Eq, Debug, Serialize)]
pub struct SupportRowView {
    pub vendor: String,
    pub arch: String,
    pub architecture: String,
    pub weight_format: String,
    pub kv_format: String,
    pub speculative: String,
    pub status: &'static str,
    pub reason: Option<String>,
}

const fn row(
    vendor: Option<&'static str>,
    arch: Option<&'static str>,
    architecture: Option<&'static str>,
    weight_format: Option<WeightFormat>,
    kv_format: Option<KvFormatColumn>,
    speculative: Option<SpeculativeColumn>,
    status: SupportStatus,
) -> SupportRow {
    SupportRow {
        key: SupportKeyPattern {
            vendor,
            arch,
            architecture,
            weight_format,
            kv_format,
            speculative,
        },
        status,
    }
}

const fn unsupported(reason: &'static str) -> SupportStatus {
    SupportStatus::Unsupported {
        reason: Cow::Borrowed(reason),
    }
}

const BF16: Option<WeightFormat> = Some(WeightFormat::Bf16);
const KV_BF16: Option<KvFormatColumn> = Some(KvFormatColumn::Bf16);
const NO_SPEC: Option<SpeculativeColumn> = Some(SpeculativeColumn::None);
const QUANT_REASON: &str =
    "quantized checkpoints are not validated yet (track phase-8a-quantization)";

/// The support matrix. Order is irrelevant: the most specific matching row wins, and
/// `support::tests::resolution_and_refusal` proves no two overlapping rows tie.
pub static SUPPORT_MATRIX: &[SupportRow] = &[
    // Phase 1–7 baseline (S-2).
    row(
        Some("amd"),
        Some("gfx1201"),
        Some("LlamaForCausalLM"),
        BF16,
        KV_BF16,
        NO_SPEC,
        SupportStatus::Supported,
    ),
    row(
        Some("amd"),
        Some("gfx1201"),
        Some("OlmoeForCausalLM"),
        BF16,
        KV_BF16,
        NO_SPEC,
        SupportStatus::Supported,
    ),
    row(
        Some("nvidia"),
        Some("sm_121"),
        Some("LlamaForCausalLM"),
        BF16,
        KV_BF16,
        NO_SPEC,
        SupportStatus::Supported,
    ),
    row(
        Some("nvidia"),
        Some("sm_121"),
        Some("OlmoeForCausalLM"),
        BF16,
        KV_BF16,
        NO_SPEC,
        SupportStatus::Supported,
    ),
    // CPU reference provider: tests and tiny checkpoints only.
    row(
        Some("cpu"),
        None,
        None,
        BF16,
        KV_BF16,
        NO_SPEC,
        SupportStatus::Experimental,
    ),
    // Reserved for the tracks; each track replaces its refusal with validated rows.
    row(
        None,
        None,
        None,
        Some(WeightFormat::ModeloptNvfp4),
        KV_BF16,
        NO_SPEC,
        unsupported(QUANT_REASON),
    ),
    row(
        None,
        None,
        None,
        Some(WeightFormat::ModeloptFp8),
        KV_BF16,
        NO_SPEC,
        unsupported(QUANT_REASON),
    ),
    row(
        None,
        None,
        None,
        Some(WeightFormat::ModeloptMixed),
        KV_BF16,
        NO_SPEC,
        unsupported(QUANT_REASON),
    ),
    row(
        None,
        None,
        None,
        Some(WeightFormat::CtNvfp4),
        KV_BF16,
        NO_SPEC,
        unsupported(QUANT_REASON),
    ),
    row(
        None,
        None,
        None,
        None,
        Some(KvFormatColumn::Fp8E4m3),
        NO_SPEC,
        unsupported("fp8_e4m3 KV cache is not validated yet (track phase-8a-quantization)"),
    ),
    row(
        None,
        None,
        None,
        None,
        None,
        Some(SpeculativeColumn::Draft),
        unsupported(
            "draft-model speculative decoding is not validated yet (track phase-8b-speculative-decoding)",
        ),
    ),
];

const NO_ROW: &str = "no support-matrix row";

impl SupportKeyPattern {
    /// Number of non-wildcard columns.
    pub fn specificity(&self) -> u32 {
        [
            self.vendor.is_some(),
            self.arch.is_some(),
            self.architecture.is_some(),
            self.weight_format.is_some(),
            self.kv_format.is_some(),
            self.speculative.is_some(),
        ]
        .iter()
        .filter(|b| **b)
        .count() as u32
    }

    /// Exact match: every non-wildcard column equals the key's value.
    pub fn matches(&self, key: &SupportKey) -> bool {
        str_col(self.vendor, &key.vendor, false)
            && str_col(self.arch, &key.arch, false)
            && str_col(self.architecture, &key.architecture, false)
            && self.weight_format.is_none_or(|w| w == key.weight_format)
            && self.kv_format.is_none_or(|k| k == key.kv_format)
            && self.speculative.is_none_or(|s| s == key.speculative)
    }

    /// Like [`Self::matches`], but a key column equal to [`WILDCARD`] (unknown) matches any value.
    fn compatible(&self, key: &SupportKey) -> bool {
        str_col(self.vendor, &key.vendor, true)
            && str_col(self.arch, &key.arch, true)
            && str_col(self.architecture, &key.architecture, true)
            && self.weight_format.is_none_or(|w| w == key.weight_format)
            && self.kv_format.is_none_or(|k| k == key.kv_format)
            && self.speculative.is_none_or(|s| s == key.speculative)
    }

    /// True when some concrete key matches both patterns.
    fn overlaps(&self, other: &SupportKeyPattern) -> bool {
        fn col<T: PartialEq>(a: Option<T>, b: Option<T>) -> bool {
            match (a, b) {
                (Some(x), Some(y)) => x == y,
                _ => true,
            }
        }
        col(self.vendor, other.vendor)
            && col(self.arch, other.arch)
            && col(self.architecture, other.architecture)
            && col(self.weight_format, other.weight_format)
            && col(self.kv_format, other.kv_format)
            && col(self.speculative, other.speculative)
    }
}

fn str_col(pattern: Option<&'static str>, value: &str, unknown_matches: bool) -> bool {
    match pattern {
        None => true,
        Some(p) => p == value || (unknown_matches && value == WILDCARD),
    }
}

impl SupportKey {
    /// True when vendor, arch or architecture is still unknown ([`WILDCARD`]).
    pub fn is_partial(&self) -> bool {
        [&self.vendor, &self.arch, &self.architecture]
            .iter()
            .any(|v| v.as_str() == WILDCARD)
    }

    /// The configuration key an operator changes to leave an unsupported row.
    pub fn blamed_config_key(&self) -> &'static str {
        let unknown_architecture = self.architecture != WILDCARD
            && !SUPPORT_MATRIX
                .iter()
                .any(|r| r.key.architecture == Some(self.architecture.as_str()));
        if self.speculative != SpeculativeColumn::None {
            "speculative.method"
        } else if self.kv_format != KvFormatColumn::Bf16 {
            "kv.dtype"
        } else if self.weight_format != WeightFormat::Bf16 || unknown_architecture {
            "model.path"
        } else {
            "execution.backend"
        }
    }

    /// Key for `--check-config`: no discovery, no model files, so vendor comes from
    /// `execution.backend` and arch/architecture are unknown.
    pub fn for_check_config(cfg: &Config) -> SupportKey {
        let arch = match cfg.execution.backend {
            ExecutionBackend::Cpu => "cpu",
            _ => WILDCARD,
        };
        SupportKey::for_startup(cfg, arch, WILDCARD)
    }

    /// Key at startup, after discovery (`arch` of `execution.device`, `cpu` for the CPU
    /// reference provider) and after reading `architectures[0]` from the model's `config.json`.
    pub fn for_startup(cfg: &Config, arch: &str, architecture: &str) -> SupportKey {
        let (weight_format, kv_format, speculative) = config_columns(cfg);
        SupportKey {
            vendor: vendor_column(cfg.execution.backend).to_string(),
            arch: arch.to_string(),
            architecture: architecture.to_string(),
            weight_format,
            kv_format,
            speculative,
        }
    }
}

impl fmt::Display for SupportKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "vendor={} arch={} architecture={} weight_format={} kv_format={} speculative={}",
            self.vendor,
            self.arch,
            self.architecture,
            self.weight_format.as_str(),
            self.kv_format.as_str(),
            self.speculative.as_str()
        )
    }
}

/// `execution.backend` → `vendor` column.
pub fn vendor_column(backend: ExecutionBackend) -> &'static str {
    match backend {
        ExecutionBackend::Hip => "amd",
        ExecutionBackend::Cuda => "nvidia",
        _ => "cpu",
    }
}

/// Format columns derived from the configuration. Phase-8a replaces the weight and KV
/// columns with values read from the checkpoint's quantization config and `kv.dtype`.
pub fn config_columns(cfg: &Config) -> (WeightFormat, KvFormatColumn, SpeculativeColumn) {
    let speculative = match cfg.speculative.method {
        SpeculativeMethod::Draft => SpeculativeColumn::Draft,
        _ => SpeculativeColumn::None,
    };
    (WeightFormat::Bf16, KvFormatColumn::Bf16, speculative)
}

/// Most specific row of `table` matching `key` exactly; no row → unsupported.
pub fn resolve_in(table: &[SupportRow], key: &SupportKey) -> SupportStatus {
    table
        .iter()
        .filter(|r| r.key.matches(key))
        .max_by_key(|r| r.key.specificity())
        .map(|r| r.status.clone())
        .unwrap_or(SupportStatus::Unsupported {
            reason: Cow::Borrowed(NO_ROW),
        })
}

/// Resolution for a key with unknown columns: the best status any compatible row could
/// give (unsupported only when every compatible row is unsupported, with the reason of the
/// most specific one).
pub fn resolve_partial_in(table: &[SupportRow], key: &SupportKey) -> SupportStatus {
    let compatible: Vec<&SupportRow> = table.iter().filter(|r| r.key.compatible(key)).collect();
    if let Some(best) = compatible
        .iter()
        .filter(|r| r.status.rank() > 0)
        .max_by_key(|r| r.status.rank())
    {
        return best.status.clone();
    }
    compatible
        .iter()
        .max_by_key(|r| r.key.specificity())
        .map(|r| r.status.clone())
        .unwrap_or(SupportStatus::Unsupported {
            reason: Cow::Borrowed(NO_ROW),
        })
}

/// [`resolve_in`] over [`SUPPORT_MATRIX`].
pub fn resolve(key: &SupportKey) -> SupportStatus {
    resolve_in(SUPPORT_MATRIX, key)
}

/// A resolved key and its status; `Err` from [`check_in`] when unsupported.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SupportDecision {
    pub key: SupportKey,
    pub status: SupportStatus,
}

impl SupportDecision {
    /// The WARN line an experimental row starts with.
    pub fn warning(&self) -> Option<String> {
        match self.status {
            SupportStatus::Experimental => {
                Some(format!("support matrix: {} is experimental", self.key))
            }
            _ => None,
        }
    }

    pub fn view(&self) -> SupportRowView {
        SupportRowView {
            vendor: self.key.vendor.clone(),
            arch: self.key.arch.clone(),
            architecture: self.key.architecture.clone(),
            weight_format: self.key.weight_format.as_str().to_string(),
            kv_format: self.key.kv_format.as_str().to_string(),
            speculative: self.key.speculative.as_str().to_string(),
            status: self.status.as_str(),
            reason: self.status.reason().map(str::to_string),
        }
    }
}

/// Resolve `key` against `table` (partial keys use [`resolve_partial_in`]) and refuse
/// `unsupported` as a configuration error naming the blamed key, the combination and the reason.
pub fn check_in(table: &[SupportRow], key: SupportKey) -> Result<SupportDecision, ConfigError> {
    let status = if key.is_partial() {
        resolve_partial_in(table, &key)
    } else {
        resolve_in(table, &key)
    };
    if let SupportStatus::Unsupported { reason } = &status {
        return Err(ConfigError::Invalid {
            key: key.blamed_config_key().to_string(),
            reason: format!("support matrix: {key} is unsupported: {reason}"),
        });
    }
    Ok(SupportDecision { key, status })
}

/// [`check_in`] over [`SUPPORT_MATRIX`].
pub fn check(key: SupportKey) -> Result<SupportDecision, ConfigError> {
    check_in(SUPPORT_MATRIX, key)
}

impl SupportRow {
    pub fn view(&self) -> SupportRowView {
        let s = |v: Option<&'static str>| v.unwrap_or(WILDCARD).to_string();
        SupportRowView {
            vendor: s(self.key.vendor),
            arch: s(self.key.arch),
            architecture: s(self.key.architecture),
            weight_format: s(self.key.weight_format.map(WeightFormat::as_str)),
            kv_format: s(self.key.kv_format.map(KvFormatColumn::as_str)),
            speculative: s(self.key.speculative.map(SpeculativeColumn::as_str)),
            status: self.status.as_str(),
            reason: self.status.reason().map(str::to_string),
        }
    }
}

/// Table invariants: vendors from [`VENDORS`], non-empty string columns, and no two
/// overlapping rows with equal specificity (a key matching both would be ambiguous).
pub fn validate_table(table: &[SupportRow]) -> Result<(), String> {
    for (i, r) in table.iter().enumerate() {
        if let Some(v) = r.key.vendor
            && !VENDORS.contains(&v)
        {
            return Err(format!("row {i}: vendor {v:?} not in {VENDORS:?}"));
        }
        for v in [r.key.vendor, r.key.arch, r.key.architecture]
            .into_iter()
            .flatten()
        {
            if v.is_empty() || v == WILDCARD {
                return Err(format!("row {i}: use None for a wildcard, not {v:?}"));
            }
        }
    }
    for (i, a) in table.iter().enumerate() {
        for (j, b) in table.iter().enumerate().skip(i + 1) {
            if a.key.overlaps(&b.key) && a.key.specificity() == b.key.specificity() {
                return Err(format!("rows {i} and {j} overlap with equal specificity"));
            }
        }
    }
    Ok(())
}
```

- [ ] Run `cargo test -p turbine-core -- support::tests::resolution_and_refusal support::tests::baseline_rows_present` — expect PASS.
- [ ] Mutation check (do not commit): in `resolve_in` change `max_by_key` to `min_by_key`, rerun `cargo test -p turbine-core support::tests::resolution_and_refusal` — expect FAIL with "assertion `left == right` failed"; revert. Change the first baseline row's status to `SupportStatus::Experimental`, rerun `cargo test -p turbine-core support::tests::baseline_rows_present` — expect FAIL; revert.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`.
- [ ] Commit: `feat(core): add the phase 8 support matrix with most-specific-row resolution`.

## Task 3: `turbine-server --support-matrix` and `--check-config` resolution

Files: `crates/turbine-server/src/support_matrix.rs` (new: table rendering and the `--check-config` check, unit test), `crates/turbine-server/src/cli.rs` (new `--support-matrix`, `--output`, `OutputFormat`; `--config` optional only with `--support-matrix`), `crates/turbine-server/src/main.rs` (dispatch `--support-matrix` before any config read; call the check in the `--check-config` branch), `crates/turbine-server/tests/server_cli.rs` (append `mod phase8_support_matrix`)
Interfaces: consumes Task 2 `SUPPORT_MATRIX`, `SupportRow::view`, `SupportKey::for_check_config`, `support::check`; P0 `turbine_core::config::load`, P0 `ExitCode { Clean = 0, Startup = 1, Config = 2, .. }`. Produces `turbine_server::cli::OutputFormat { Text, Json }`, `support_matrix::render_matrix(output: OutputFormat) -> String` (JSON `{"rows":[SupportRowView…]}`; text: header line then one line per row, whitespace-separated columns `vendor arch architecture weight_format kv_format speculative status reason`, `-` for no reason — Task 8's gate script parses columns 1–7), `support_matrix::check_config(cfg: &Config) -> Result<SupportDecision, ConfigError>`.
Covers: S-2 — `cargo test -p turbine-server --test server_cli support_matrix_output`.

- [ ] Write the failing test: append to `crates/turbine-server/tests/server_cli.rs`:

```rust
mod phase8_support_matrix {
    use std::process::Command;

    fn p8_server_cmd() -> Command {
        Command::new(env!("CARGO_BIN_EXE_turbine-server"))
    }

    #[test]
    fn support_matrix_output() {
        let out = p8_server_cmd()
            .args(["--support-matrix", "--output", "json"])
            .output()
            .unwrap();
        assert_eq!(
            out.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        let rows = v["rows"].as_array().expect("rows array");
        assert!(!rows.is_empty());
        for key in [
            "vendor",
            "arch",
            "architecture",
            "weight_format",
            "kv_format",
            "speculative",
            "status",
            "reason",
        ] {
            assert!(rows[0].get(key).is_some(), "row lacks {key}: {}", rows[0]);
        }
        assert!(
            rows.iter()
                .any(|r| r["speculative"] == "draft" && r["status"] == "unsupported")
        );

        let dir = tempfile::tempdir().unwrap();
        let ok = dir.path().join("ok.yaml");
        std::fs::write(&ok, "model:\n  path: /nonexistent/model\n").unwrap();
        let out = p8_server_cmd()
            .arg("--config")
            .arg(&ok)
            .arg("--check-config")
            .output()
            .unwrap();
        assert_eq!(
            out.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(String::from_utf8_lossy(&out.stdout).contains("config ok"));

        let draft = dir.path().join("draft.yaml");
        std::fs::write(
            &draft,
            "model:\n  path: /nonexistent/model\nspeculative:\n  method: draft\n",
        )
        .unwrap();
        let out = p8_server_cmd()
            .arg("--config")
            .arg(&draft)
            .arg("--check-config")
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{stderr}");
        assert!(stderr.contains("speculative.method"), "{stderr}");
        assert!(stderr.contains("speculative=draft"), "{stderr}");
        assert!(!String::from_utf8_lossy(&out.stdout).contains("config ok"));
    }
}
```

Run `cargo test -p turbine-server --test server_cli support_matrix_output` — expect FAIL with "unexpected argument '--support-matrix'".

- [ ] Implement the CLI flags in `crates/turbine-server/src/cli.rs`: keep the P0 `--set` field unchanged, change `config` to `Option<PathBuf>`, and add the rest so the struct and the new enum read:

```rust
    /// YAML configuration file.
    #[arg(long, value_name = "PATH", required_unless_present = "support_matrix")]
    pub config: Option<PathBuf>,
    /// Validate the configuration (and resolve its support-matrix row), print `config ok`, exit.
    #[arg(long, conflicts_with = "support_matrix")]
    pub check_config: bool,
    /// Print the support matrix and exit 0 without reading a configuration.
    #[arg(long, conflicts_with_all = ["config", "check_config"])]
    pub support_matrix: bool,
    /// Output format of `--support-matrix`.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text, requires = "support_matrix")]
    pub output: OutputFormat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
}
```

- [ ] Create `crates/turbine-server/src/support_matrix.rs`:

```rust
//! `--support-matrix` and the `--check-config` support-matrix resolution (phase 8 S-2).
use std::fmt::Write as _;

use turbine_core::config::{Config, ConfigError};
use turbine_core::support::{self, SUPPORT_MATRIX, SupportDecision, SupportKey};

use crate::cli::OutputFormat;

/// Body printed by `turbine-server --support-matrix`.
pub fn render_matrix(output: OutputFormat) -> String {
    let rows: Vec<_> = SUPPORT_MATRIX.iter().map(|r| r.view()).collect();
    match output {
        OutputFormat::Json => {
            let mut s = serde_json::to_string(&serde_json::json!({ "rows": rows }))
                .expect("rows serialize");
            s.push('\n');
            s
        }
        OutputFormat::Text => {
            let mut s = String::new();
            let _ = writeln!(
                s,
                "{:<7} {:<8} {:<18} {:<14} {:<9} {:<11} {:<12} reason",
                "vendor",
                "arch",
                "architecture",
                "weight_format",
                "kv_format",
                "speculative",
                "status"
            );
            for r in rows {
                let _ = writeln!(
                    s,
                    "{:<7} {:<8} {:<18} {:<14} {:<9} {:<11} {:<12} {}",
                    r.vendor,
                    r.arch,
                    r.architecture,
                    r.weight_format,
                    r.kv_format,
                    r.speculative,
                    r.status,
                    r.reason.as_deref().unwrap_or("-")
                );
            }
            s
        }
    }
}

/// `--check-config`: no discovery and no model files, so arch and architecture are unknown.
pub fn check_config(cfg: &Config) -> Result<SupportDecision, ConfigError> {
    support::check(SupportKey::for_check_config(cfg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_text_lists_every_row() {
        let text = render_matrix(OutputFormat::Text);
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].starts_with("vendor "), "{text}");
        assert_eq!(lines.len(), SUPPORT_MATRIX.len() + 1);
        assert!(
            lines
                .iter()
                .any(|l| l.split_whitespace().collect::<Vec<_>>()[..7]
                    == [
                        "amd",
                        "gfx1201",
                        "LlamaForCausalLM",
                        "bf16",
                        "bf16",
                        "none",
                        "supported"
                    ]),
            "{text}"
        );
    }
}
```

- [ ] Wire `crates/turbine-server/src/main.rs`: add `mod support_matrix;`; as the first statement after `Cli::parse()` insert

```rust
    if cli.support_matrix {
        print!("{}", support_matrix::render_matrix(cli.output));
        return ExitCode::Clean.into();
    }
    let config_path = cli.config.as_deref().expect("clap requires --config unless --support-matrix");
```

and use `config_path` where P0 used `&cli.config`. In the `--check-config` branch, after the P0/P5 validation succeeds and before printing `config ok`, insert

```rust
        if let Err(e) = support_matrix::check_config(&cfg) {
            eprintln!("{e}");
            return ExitCode::Config.into();
        }
```

(`ExitCode` is the P0 enum; `.into()` is the P0 conversion to `std::process::ExitCode` — use whatever conversion the surrounding P0 branches use.)

- [ ] Run `cargo test -p turbine-server --test server_cli support_matrix_output` and `cargo test -p turbine-server support_matrix::tests::matrix_text_lists_every_row` — expect PASS; `cargo test -p turbine-server --test server_cli` — expect PASS (P0 `invalid_config_exits_2_before_bind` unaffected).
- [ ] Check by hand: `cargo run -q -p turbine-server -- --support-matrix` prints the header plus 11 rows, the first `amd     gfx1201  LlamaForCausalLM   bf16           bf16      none        supported    -`.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`.
- [ ] Commit: `feat(server): add --support-matrix and resolve the support row under --check-config`.

## Task 4: Startup resolution, `support` status key and `turbine_support_matrix_status`

Files: `crates/turbine-api/src/support.rs` (new: `SupportMetrics` gauge family), `crates/turbine-api/src/lib.rs` (add `pub mod support;`), `crates/turbine-api/tests/api.rs` (append `mod phase8_support`), `crates/turbine-server/src/support_startup.rs` (new: device arch, `config.json` architecture, startup decision, unit test), `crates/turbine-server/src/startup.rs` (call it at step 3, register the gauge, fill the status document), `crates/turbine-server/src/main.rs` (add `mod support_startup;`), the file defining `StatusDocument` in `crates/turbine-server/src/` (add the `support` field), `crates/turbine-server/tests/tiny_server.rs` (append `mod phase8_status`)
Interfaces: consumes Task 2 `support::check`, `SupportKey::for_startup`, `SupportDecision::{warning, view}`, `SupportRowView`, `SupportStatus::as_str`; P0 `MetricsRegistry::register`, `ApiState`, `router`, `Diagnostics`, `InferenceBackend`, `Readiness`, `ApiLimits`, `ApiError::not_implemented()`; P0 `turbine_device::DeviceInventory { devices: Vec<DeviceInfo { index, vendor, arch, .. }> }`; P1 `turbine_model::testing::tiny::write_tiny_llama(dir, seed)`. Produces `turbine_api::support::{SupportMetrics::register(&MetricsRegistry) -> SupportMetrics, SupportMetrics::set(&self, &SupportStatus), SupportStatusLabels { status: &'static str }}`, `support_startup::{device_arch(cfg, inventory) -> Option<String>, read_architecture(model_dir) -> Option<String>, startup_decision(cfg, arch: Option<&str>, model_dir) -> Result<Option<SupportDecision>, ConfigError>}`, `StatusDocument.support: Option<SupportRowView>` (JSON key `support`), log event `support_matrix` (WARN) for experimental rows.
Covers: S-2 — `cargo test -p turbine-api --test api status_reports_support_row`; edge case "a track rolled back after release fails at next startup, never mid-request" (resolution runs only at startup).

- [ ] Write the failing test: append to `crates/turbine-api/tests/api.rs` (if the P6 `Diagnostics` trait carries `debug_faults`, also add to `SupportStatusDiagnostics`: `#[cfg(feature = "fault-injection")] fn debug_faults(&self, _req: turbine_api::DebugFault) -> Result<(), ApiError> { Err(ApiError::not_implemented()) }`):

```rust
mod phase8_support {
    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;
    use turbine_api::support::SupportMetrics;
    use turbine_api::{
        ApiError, ApiLimits, ApiState, BoxFuture, Diagnostics, GenerationStream, InferenceBackend,
        InferenceRequest, ModelCard, PrefetchAccepted, PrefetchRequest, Readiness, ReadyState,
        TopologyScope, router,
    };
    use turbine_core::config::Config;
    use turbine_core::support::{self, SupportKey};
    use turbine_observability::MetricsRegistry;

    struct SupportNoInference;
    impl InferenceBackend for SupportNoInference {
        fn models(&self) -> Vec<ModelCard> {
            Vec::new()
        }
        fn submit(
            &self,
            _req: InferenceRequest,
        ) -> BoxFuture<'_, Result<GenerationStream, ApiError>> {
            Box::pin(async { Err(ApiError::not_implemented()) })
        }
        fn prefetch(
            &self,
            _req: PrefetchRequest,
        ) -> BoxFuture<'_, Result<PrefetchAccepted, ApiError>> {
            Box::pin(async { Err(ApiError::not_implemented()) })
        }
    }

    struct SupportStatusDiagnostics(serde_json::Value);
    impl Diagnostics for SupportStatusDiagnostics {
        fn status(&self) -> serde_json::Value {
            self.0.clone()
        }
        fn devices(&self) -> serde_json::Value {
            serde_json::json!({"devices": [], "backends": []})
        }
        fn scheduler(&self) -> Result<serde_json::Value, ApiError> {
            Err(ApiError::not_implemented())
        }
        fn kv(&self) -> Result<serde_json::Value, ApiError> {
            Err(ApiError::not_implemented())
        }
        fn pressure(&self) -> Result<serde_json::Value, ApiError> {
            Err(ApiError::not_implemented())
        }
        fn topology(&self, _scope: TopologyScope) -> Result<serde_json::Value, ApiError> {
            Err(ApiError::not_implemented())
        }
        fn cluster(&self) -> Result<serde_json::Value, ApiError> {
            Err(ApiError::not_implemented())
        }
    }

    struct SupportReady;
    impl Readiness for SupportReady {
        fn ready(&self) -> ReadyState {
            ReadyState::Ready
        }
    }

    async fn get_text(app: &axum::Router, uri: &str) -> (StatusCode, String) {
        let resp = app
            .clone()
            .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    #[tokio::test]
    async fn status_reports_support_row() {
        // The server resolves the CPU reference provider's row exactly as startup does.
        let mut cfg = Config::default();
        cfg.execution.backend = turbine_core::types::ExecutionBackend::Cpu;
        let decision =
            support::check(SupportKey::for_startup(&cfg, "cpu", "LlamaForCausalLM")).unwrap();
        let metrics = MetricsRegistry::new();
        let gauge = SupportMetrics::register(&metrics);
        gauge.set(&decision.status);
        let status_doc = serde_json::json!({ "ready": true, "support": decision.view() });
        let app = router(ApiState {
            inference: Arc::new(SupportNoInference),
            diagnostics: Arc::new(SupportStatusDiagnostics(status_doc)),
            readiness: Arc::new(SupportReady),
            metrics: metrics.clone(),
            limits: ApiLimits::default(),
        });

        let (code, body) = get_text(&app, "/turbine/v1/status").await;
        assert_eq!(code, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        let row = &v["support"];
        assert_eq!(row["vendor"], "cpu", "{v}");
        assert_eq!(row["arch"], "cpu");
        assert_eq!(row["architecture"], "LlamaForCausalLM");
        assert_eq!(row["weight_format"], "bf16");
        assert_eq!(row["kv_format"], "bf16");
        assert_eq!(row["speculative"], "none");
        assert_eq!(row["status"], "experimental");
        assert!(row["reason"].is_null());

        let (code, text) = get_text(&app, "/metrics").await;
        assert_eq!(code, StatusCode::OK);
        let lines: Vec<&str> = text
            .lines()
            .filter(|l| l.starts_with("turbine_support_matrix_status{"))
            .collect();
        assert_eq!(lines.len(), 3, "{text}");
        let ones: Vec<&&str> = lines.iter().filter(|l| l.ends_with(" 1")).collect();
        assert_eq!(
            ones,
            [&"turbine_support_matrix_status{status=\"experimental\"} 1"],
            "{text}"
        );
    }
}
```

Run `cargo test -p turbine-api --test api status_reports_support_row` — expect FAIL with "unresolved import `turbine_api::support`".

- [ ] Implement `crates/turbine-api/src/support.rs` and add `pub mod support;` to `crates/turbine-api/src/lib.rs` (add `turbine-core` to `crates/turbine-api/Cargo.toml` `[dependencies]` if P0 did not — it is an allowed edge, contract §1.1):

```rust
//! `turbine_support_matrix_status` (phase 8 S-2): registered by `turbine-server` at startup
//! with the resolved status of the running configuration.
use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use turbine_core::support::SupportStatus;
use turbine_observability::MetricsRegistry;

/// Label set of `turbine_support_matrix_status`; `status` ∈ supported, experimental, unsupported.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct SupportStatusLabels {
    pub status: &'static str,
}

const STATUSES: [&str; 3] = ["supported", "experimental", "unsupported"];

#[derive(Clone)]
pub struct SupportMetrics {
    status: Family<SupportStatusLabels, Gauge>,
}

impl SupportMetrics {
    /// Registers `turbine_support_matrix_status` with all three label values at 0.
    pub fn register(reg: &MetricsRegistry) -> Self {
        let status = reg.register(
            "turbine_support_matrix_status",
            "Support-matrix status of the running configuration (1 for the resolved status)",
            Family::<SupportStatusLabels, Gauge>::default(),
        );
        for s in STATUSES {
            status
                .get_or_create(&SupportStatusLabels { status: s })
                .set(0);
        }
        Self { status }
    }

    /// Sets the resolved status to 1 and the other two to 0.
    pub fn set(&self, resolved: &SupportStatus) {
        for s in STATUSES {
            let v = i64::from(s == resolved.as_str());
            self.status
                .get_or_create(&SupportStatusLabels { status: s })
                .set(v);
        }
    }
}
```

- [ ] Run `cargo test -p turbine-api --test api status_reports_support_row` — expect PASS.
- [ ] Write the failing server end-to-end test: append to `crates/turbine-server/tests/tiny_server.rs` (add `tempfile` and `serde_json` to `[dev-dependencies]` of `crates/turbine-server/Cargo.toml` if absent):

```rust
mod phase8_status {
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::process::{Child, Command, Stdio};
    use std::time::{Duration, Instant};

    struct KillOnDrop(Child);
    impl Drop for KillOnDrop {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    fn free_port() -> u16 {
        TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    /// Minimal HTTP/1.1 GET (Connection: close) → (status, body).
    fn http_get(port: u16, path: &str) -> Option<(u16, String)> {
        let mut s = TcpStream::connect(("127.0.0.1", port)).ok()?;
        s.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
        write!(
            s,
            "GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        )
        .ok()?;
        let mut raw = String::new();
        s.read_to_string(&mut raw).ok()?;
        let code = raw.split_whitespace().nth(1)?.parse().ok()?;
        Some((code, raw.split_once("\r\n\r\n")?.1.to_string()))
    }

    #[test]
    fn support_row_in_status() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("tiny-llama");
        std::fs::create_dir(&model).unwrap();
        turbine_model::testing::tiny::write_tiny_llama(&model, 7);
        let port = free_port();
        let cfg = dir.path().join("cpu.yaml");
        let yaml = format!(
            "server:\n  listen: 127.0.0.1:{port}\nmodel:\n  path: {}\nexecution:\n  backend: cpu\n",
            model.display()
        );
        std::fs::write(&cfg, yaml).unwrap();
        let _server = KillOnDrop(
            Command::new(env!("CARGO_BIN_EXE_turbine-server"))
                .arg("--config")
                .arg(&cfg)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap(),
        );
        let deadline = Instant::now() + Duration::from_secs(60);
        while !matches!(http_get(port, "/ready"), Some((200, _))) {
            assert!(Instant::now() < deadline, "server not ready within 60 s");
            std::thread::sleep(Duration::from_millis(100));
        }
        let (code, body) = http_get(port, "/turbine/v1/status").unwrap();
        assert_eq!(code, 200);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["support"]["vendor"], "cpu", "{v}");
        assert_eq!(v["support"]["architecture"], "LlamaForCausalLM", "{v}");
        assert_eq!(v["support"]["status"], "experimental", "{v}");
        let (_, metrics) = http_get(port, "/metrics").unwrap();
        assert!(
            metrics.contains("turbine_support_matrix_status{status=\"experimental\"} 1"),
            "{metrics}"
        );
        assert!(
            metrics.contains("turbine_support_matrix_status{status=\"supported\"} 0"),
            "{metrics}"
        );
    }
}
```

Run `cargo test -p turbine-server --test tiny_server support_row_in_status` — expect FAIL with "assertion `left == right` failed" (no `support` key yet: `left: Null`).

- [ ] Create `crates/turbine-server/src/support_startup.rs` and add `mod support_startup;` to `main.rs`:

```rust
//! Startup support-matrix check (phase 8 S-2): runs after device discovery and the parallel
//! plan, before the kernel library, model config and weights; unsupported → exit 1.
use std::path::Path;

use turbine_core::config::{Config, ConfigError};
use turbine_core::support::{self, SupportDecision, SupportKey};
use turbine_core::types::{ExecutionBackend, Vendor};
use turbine_device::DeviceInventory;

/// `arch` of the configured execution device, `cpu` for the CPU reference provider, `None`
/// when the device is missing or of the other vendor (startup step 4 reports that, exit 1).
pub fn device_arch(cfg: &Config, inventory: &DeviceInventory) -> Option<String> {
    let vendor = match cfg.execution.backend {
        ExecutionBackend::Hip => Vendor::Amd,
        ExecutionBackend::Cuda => Vendor::Nvidia,
        _ => return Some("cpu".to_string()),
    };
    inventory
        .devices
        .iter()
        .find(|d| d.index == cfg.execution.device && d.vendor == vendor)
        .and_then(|d| d.arch.clone())
}

/// `architectures[0]` of `<model_dir>/config.json`; `None` when the file is missing or
/// malformed (startup step 5 reports that, exit 1).
pub fn read_architecture(model_dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(model_dir.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    v.get("architectures")?.get(0)?.as_str().map(str::to_string)
}

/// Startup step 3 (after discovery, before the kernel library, model config and weights):
/// `Ok(None)` when arch or architecture cannot be known yet, `Err` (exit 1) when unsupported.
pub fn startup_decision(
    cfg: &Config,
    arch: Option<&str>,
    model_dir: &Path,
) -> Result<Option<SupportDecision>, ConfigError> {
    let (Some(arch), Some(architecture)) = (arch, read_architecture(model_dir)) else {
        return Ok(None);
    };
    let decision = support::check(SupportKey::for_startup(cfg, arch, &architecture))?;
    if let Some(w) = decision.warning() {
        tracing::warn!(
            event = "support_matrix",
            status = decision.status.as_str(),
            "{w}"
        );
    }
    Ok(Some(decision))
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbine_core::config::SpeculativeMethod;
    use turbine_core::support::SupportStatus;

    fn model_dir(architecture: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let body = serde_json::json!({ "architectures": [architecture] }).to_string();
        std::fs::write(dir.path().join("config.json"), body).unwrap();
        dir
    }

    #[test]
    fn startup_decision_cases() {
        let llama = model_dir("LlamaForCausalLM");
        let mut cfg = Config::default();
        cfg.execution.backend = ExecutionBackend::Hip;
        let d = startup_decision(&cfg, Some("gfx1201"), llama.path())
            .unwrap()
            .unwrap();
        assert_eq!(d.status, SupportStatus::Supported);
        assert_eq!(d.key.vendor, "amd");

        // Unknown device or unreadable config.json: deferred to the later startup steps.
        assert_eq!(startup_decision(&cfg, None, llama.path()).unwrap(), None);
        let empty = tempfile::tempdir().unwrap();
        assert_eq!(
            startup_decision(&cfg, Some("gfx1201"), empty.path()).unwrap(),
            None
        );

        // Unregistered architecture on a validated device: no row, blamed on model.path.
        let mistral = model_dir("MistralForCausalLM");
        let err = startup_decision(&cfg, Some("gfx1201"), mistral.path()).unwrap_err();
        assert_eq!(err.key(), Some("model.path"));
        assert!(err.to_string().contains("no support-matrix row"), "{err}");

        // Draft speculation is refused naming speculative.method.
        cfg.speculative.method = SpeculativeMethod::Draft;
        let err = startup_decision(&cfg, Some("gfx1201"), llama.path()).unwrap_err();
        assert_eq!(err.key(), Some("speculative.method"));

        // CPU reference provider: experimental, with the WARN text.
        let mut cpu = Config::default();
        cpu.execution.backend = ExecutionBackend::Cpu;
        let arch = device_arch(&cpu, &DeviceInventory::default());
        assert_eq!(arch.as_deref(), Some("cpu"));
        let d = startup_decision(&cpu, arch.as_deref(), llama.path())
            .unwrap()
            .unwrap();
        assert_eq!(d.status, SupportStatus::Experimental);
        assert!(d.warning().unwrap().contains("vendor=cpu"));
        assert_eq!(device_arch(&cfg, &DeviceInventory::default()), None);
    }
}
```

- [ ] Wire startup step 3 in `crates/turbine-server/src/startup.rs`: directly after the P5 parallel plan (before `ShimLibrary::load` / the CPU provider is chosen), with `cfg`, `inventory` (P0 discovery result) and `metrics` (the process `MetricsRegistry`) in scope:

```rust
    // P8 S-2: support-matrix row (exit 1 before the kernel library, model config or weights).
    let arch = crate::support_startup::device_arch(&cfg, &inventory);
    let support_decision = match crate::support_startup::startup_decision(&cfg, arch.as_deref(), &cfg.model.path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{e}");
            return Err(ExitCode::Startup);
        }
    };
    let support_metrics = turbine_api::support::SupportMetrics::register(&metrics);
    if let Some(d) = &support_decision {
        support_metrics.set(&d.status);
    }
```

(`return Err(ExitCode::Startup)` stands for the existing P1 exit-1 path of this function — use the same statement the model-file failure branch next to it uses.) Keep `support_decision` alive until the status document is built.

- [ ] Add to `StatusDocument` (P1, turbine-server) the field

```rust
    #[serde(skip_serializing_if = "Option::is_none")]
    pub support: Option<turbine_core::support::SupportRowView>,
```

and fill it with `support: support_decision.as_ref().map(|d| d.view()),` where the document is constructed.

- [ ] Run `cargo test -p turbine-server support_startup::tests::startup_decision_cases`, `cargo test -p turbine-server --test tiny_server support_row_in_status`, `cargo test -p turbine-api --test api status_reports_support_row` — expect PASS; `cargo test -p turbine-server` — expect PASS (P1 `startup_failures_exit_1` keeps its messages: missing `config.json` and unknown devices defer to the later steps).
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`.
- [ ] Commit: `feat(server): refuse unsupported support-matrix rows at startup and report the row in status and metrics`.

## Task 5: `turbine-golden eval` and `eval-compare`

Files: `benches/turbine-bench/src/golden/eval.rs` (new: task parsing, numeric matching, run, report, compare; unit test), `benches/turbine-bench/src/golden/mod.rs` (add `pub mod eval;`), `benches/turbine-bench/src/bin/turbine-golden.rs` (add the `eval` and `eval-compare` subcommands), `benches/turbine-bench/Cargo.toml` (`reqwest` feature `json`, `turbine-core`, `thiserror`; dev-deps `axum`, `tempfile`, `tokio`), `benches/turbine-bench/tests/golden.rs` (append `mod phase8_eval`)
Interfaces: consumes Task 1 `QualityConfig::default().max_accuracy_drop`; P1 `turbine-golden` clap `Command` enum. Produces `turbine_bench::golden::eval::{EvalTask { id, prompt: Option<String>, messages: Option<Vec<Value>>, answer, match_kind: MatchKind, max_tokens: u32 }, MatchKind { Exact, Number }, TaskResult { id, correct, output }, EvalReport { model, tasks_file, total, correct, accuracy, results }, EvalError, CompareOutcome { baseline_accuracy, candidate_accuracy, max_drop, pass }, load_tasks(&Path), normalize_number(&str) -> Option<String>, is_correct(MatchKind, &str, &str) -> bool, run_eval(base, model: Option<&str>, tasks_file, tasks).await, compare(&EvalReport, &EvalReport, max_drop) -> CompareOutcome, read_report(&Path), REQUEST_TIMEOUT = 600 s}`; CLI `turbine-golden eval --url <base> --tasks <file> [--model <name>] [--output text|json]` (exit 0 completed, 2 usage/I/O/server error naming the task, no report printed on failure) and `turbine-golden eval-compare --baseline <r.json> --candidate <r.json> [--max-drop <float>]` (exit 0 pass, 1 fail, 2 usage/I/O). The tasks file line is `{"id","prompt"|"messages","answer","match":"exact"|"number","max_tokens"}`.
Covers: S-3, S-4 — `cargo test -p turbine-bench --test golden eval_accuracy_report`; failure mode "eval run fails midway".

- [ ] Write the failing test: append to `benches/turbine-bench/tests/golden.rs`:

```rust
mod phase8_eval {
    use std::path::Path;
    use std::process::Command;

    use axum::{Json, Router, http::StatusCode, routing::get, routing::post};
    use serde_json::{Value, json};

    /// Expected answer of eval mock task `i`.
    fn eval_mock_answer(i: u64) -> u64 {
        i * 1000 + 7
    }

    /// The mock's reply to a prompt `q<i>`: tasks 0..150 correct (plain, comma-grouped, trailing
    /// period), 150..200 wrong (`$` prefix, unit suffix, off by one); a prompt `fail` returns 500.
    fn eval_mock_reply(prompt: &str) -> Result<String, StatusCode> {
        if prompt == "fail" {
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
        let i: u64 = prompt
            .trim_start_matches('q')
            .parse()
            .map_err(|_| StatusCode::BAD_REQUEST)?;
        let n = eval_mock_answer(i);
        let grouped = format!("{},{:03}", n / 1000, n % 1000);
        Ok(match (i < 150, i % 3) {
            (true, 0) => n.to_string(),
            (true, 1) => grouped,
            (true, _) => format!("{n}."),
            (false, 0) => format!("${n}"),
            (false, 1) => format!("{n} apples"),
            (false, _) => (n + 1).to_string(),
        })
    }

    async fn spawn_eval_mock() -> String {
        let app = Router::new()
        .route("/v1/models", get(|| async { Json(json!({"object": "list", "data": [{"id": "mock-model"}]})) }))
        .route(
            "/v1/completions",
            post(|Json(body): Json<Value>| async move {
                assert_eq!(body["temperature"], 0.0);
                assert_eq!(body["model"], "mock-model");
                let text = eval_mock_reply(body["prompt"].as_str().unwrap_or_default())?;
                Ok::<_, StatusCode>(Json(json!({"choices": [{"index": 0, "text": text}]})))
            }),
        )
        .route(
            "/v1/chat/completions",
            post(|Json(body): Json<Value>| async move {
                let prompt = body["messages"][0]["content"].as_str().unwrap_or_default().to_string();
                let text = eval_mock_reply(&prompt)?;
                Ok::<_, StatusCode>(Json(
                    json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": text}}]}),
                ))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    /// 200 tasks, even ids as completions, odd ids as chat.
    fn write_eval_tasks(path: &Path, fail_at: Option<u64>) {
        let mut lines = String::new();
        for i in 0..200u64 {
            let q = if Some(i) == fail_at {
                "fail".to_string()
            } else {
                format!("q{i}")
            };
            let task = if i % 2 == 0 {
                json!({"id": format!("t{i}"), "prompt": q, "answer": eval_mock_answer(i).to_string(),
                   "match": "number", "max_tokens": 16})
            } else {
                json!({"id": format!("t{i}"), "messages": [{"role": "user", "content": q}],
                   "answer": eval_mock_answer(i).to_string(), "match": "number", "max_tokens": 16})
            };
            lines.push_str(&task.to_string());
            lines.push('\n');
        }
        std::fs::write(path, lines).unwrap();
    }

    fn eval_golden_cmd() -> Command {
        Command::new(env!("CARGO_BIN_EXE_turbine-golden"))
    }

    fn eval_compare(
        dir: &Path,
        baseline: &Path,
        accuracy: f64,
        correct: u64,
        max_drop: &str,
    ) -> Option<i32> {
        let mut report: Value = serde_json::from_slice(&std::fs::read(baseline).unwrap()).unwrap();
        report["accuracy"] = json!(accuracy);
        report["correct"] = json!(correct);
        let candidate = dir.join(format!("candidate-{correct}.json"));
        std::fs::write(&candidate, report.to_string()).unwrap();
        let out = eval_golden_cmd()
            .args(["eval-compare", "--baseline"])
            .arg(baseline)
            .arg("--candidate")
            .arg(&candidate)
            .args(["--max-drop", max_drop])
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("baseline accuracy 0.7500"), "{stdout}");
        out.status.code()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn eval_accuracy_report() {
        let base = spawn_eval_mock().await;
        let dir = tempfile::tempdir().unwrap();
        let tasks = dir.path().join("tasks.jsonl");
        write_eval_tasks(&tasks, None);

        let out = tokio::task::spawn_blocking({
            let (base, tasks) = (base.clone(), tasks.clone());
            move || {
                eval_golden_cmd()
                    .args(["eval", "--url", &base, "--output", "json", "--tasks"])
                    .arg(&tasks)
                    .output()
                    .unwrap()
            }
        })
        .await
        .unwrap();
        assert_eq!(
            out.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let report: Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(report["accuracy"], json!(0.75), "{report}");
        assert_eq!(report["correct"], 150);
        assert_eq!(report["total"], 200);
        assert_eq!(report["model"], "mock-model");
        let results = report["results"].as_array().unwrap();
        assert_eq!(results.len(), 200);
        assert_eq!(results[1]["id"], "t1");
        assert_eq!(results[1]["correct"], true, "comma-grouped answer accepted");
        assert_eq!(results[2]["correct"], true, "trailing period accepted");
        assert_eq!(results[150]["correct"], false, "$ prefix rejected");
        assert_eq!(results[151]["correct"], false, "unit suffix rejected");
        let baseline = dir.path().join("baseline.json");
        std::fs::write(&baseline, &out.stdout).unwrap();

        let (d, b) = (dir.path().to_path_buf(), baseline.clone());
        let codes = tokio::task::spawn_blocking(move || {
            (
                eval_compare(&d, &b, 0.745, 149, "0.01"),
                eval_compare(&d, &b, 0.73, 146, "0.01"),
            )
        })
        .await
        .unwrap();
        assert_eq!(codes, (Some(0), Some(1)));

        // A failure midway exits 2 naming the task and prints no report.
        let failing = dir.path().join("failing.jsonl");
        write_eval_tasks(&failing, Some(120));
        let out = tokio::task::spawn_blocking(move || {
            eval_golden_cmd()
                .args(["eval", "--url", &base, "--output", "json", "--tasks"])
                .arg(&failing)
                .output()
                .unwrap()
        })
        .await
        .unwrap();
        assert_eq!(out.status.code(), Some(2));
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("task t120"),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(out.stdout.is_empty());
    }
}
```

Run `cargo test -p turbine-bench --test golden eval_accuracy_report` — expect FAIL with "unrecognized subcommand 'eval'".

- [ ] Implement `benches/turbine-bench/src/golden/eval.rs` and add `pub mod eval;` to `benches/turbine-bench/src/golden/mod.rs`:

```rust
//! `turbine-golden eval` / `eval-compare` (phase 8 S-4): task-set accuracy against an
//! OpenAI-compatible endpoint, greedy, one request at a time, and the lossy-format gate.
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Per-request timeout (a 3B model answers a GSM8K item in seconds; 10 min is a hung server).
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MatchKind {
    Exact,
    Number,
}

/// One line of a tasks file: `{"id","prompt"|"messages","answer","match","max_tokens"}`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvalTask {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<serde_json::Value>>,
    pub answer: String,
    #[serde(rename = "match")]
    pub match_kind: MatchKind,
    pub max_tokens: u32,
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct TaskResult {
    pub id: String,
    pub correct: bool,
    pub output: String,
}

/// JSON report (`--output json`), committed as `tests/eval/<model-slug>/<engine>.json`.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct EvalReport {
    pub model: String,
    pub tasks_file: String,
    pub total: usize,
    pub correct: usize,
    pub accuracy: f64,
    pub results: Vec<TaskResult>,
}

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("{path}: {detail}")]
    Io { path: PathBuf, detail: String },
    #[error("{path}:{line}: {detail}")]
    Task {
        path: PathBuf,
        line: usize,
        detail: String,
    },
    #[error("task {id}: {detail}")]
    Request { id: String, detail: String },
    #[error("{0}")]
    Server(String),
}

/// Parses a tasks file; every line must be a task with exactly one of `prompt`/`messages`
/// and unique ids.
pub fn load_tasks(path: &Path) -> Result<Vec<EvalTask>, EvalError> {
    let text = std::fs::read_to_string(path).map_err(|e| EvalError::Io {
        path: path.to_path_buf(),
        detail: e.to_string(),
    })?;
    let mut tasks: Vec<EvalTask> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let bad = |detail: String| EvalError::Task {
            path: path.to_path_buf(),
            line: i + 1,
            detail,
        };
        let task: EvalTask = serde_json::from_str(line).map_err(|e| bad(e.to_string()))?;
        if task.prompt.is_some() == task.messages.is_some() {
            return Err(bad("exactly one of prompt or messages is required".into()));
        }
        if task.match_kind == MatchKind::Number && normalize_number(&task.answer).is_none() {
            return Err(bad(format!("answer {:?} is not a number", task.answer)));
        }
        if tasks.iter().any(|t| t.id == task.id) {
            return Err(bad(format!("duplicate id {}", task.id)));
        }
        tasks.push(task);
    }
    Ok(tasks)
}

/// Numeric normal form: trimmed, one trailing `.` removed, `,` separators removed, then an
/// optional `-` followed by digits with at most one `.`; anything else (units, `$`) is `None`.
pub fn normalize_number(s: &str) -> Option<String> {
    let t = s.trim();
    let t = t.strip_suffix('.').unwrap_or(t);
    let t: String = t.chars().filter(|c| *c != ',').collect();
    let digits = t.strip_prefix('-').unwrap_or(&t);
    let mut parts = digits.split('.');
    let int = parts.next()?;
    let frac = parts.next();
    if parts.next().is_some()
        || int.is_empty()
        || !int.chars().all(|c| c.is_ascii_digit())
        || frac.is_some_and(|f| f.is_empty() || !f.chars().all(|c| c.is_ascii_digit()))
    {
        return None;
    }
    Some(t)
}

pub fn is_correct(kind: MatchKind, expected: &str, output: &str) -> bool {
    match kind {
        MatchKind::Exact => output.trim() == expected.trim(),
        MatchKind::Number => match (normalize_number(expected), normalize_number(output)) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        },
    }
}

async fn served_model(client: &reqwest::Client, base: &str) -> Result<String, EvalError> {
    let v: serde_json::Value = client
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| EvalError::Server(format!("GET /v1/models: {e}")))?
        .json()
        .await
        .map_err(|e| EvalError::Server(format!("GET /v1/models: {e}")))?;
    v["data"][0]["id"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| EvalError::Server("GET /v1/models: no model listed".into()))
}

async fn complete(
    client: &reqwest::Client,
    base: &str,
    model: &str,
    task: &EvalTask,
) -> Result<String, EvalError> {
    let fail = |detail: String| EvalError::Request {
        id: task.id.clone(),
        detail,
    };
    let (path, body) = match (&task.prompt, &task.messages) {
        (Some(prompt), _) => (
            "/v1/completions",
            serde_json::json!({"model": model, "prompt": prompt, "max_tokens": task.max_tokens,
                               "temperature": 0.0, "stream": false}),
        ),
        (None, messages) => (
            "/v1/chat/completions",
            serde_json::json!({"model": model, "messages": messages, "max_tokens": task.max_tokens,
                               "temperature": 0.0, "stream": false}),
        ),
    };
    let v: serde_json::Value = client
        .post(format!("{base}{path}"))
        .json(&body)
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| fail(e.to_string()))?
        .json()
        .await
        .map_err(|e| fail(e.to_string()))?;
    let choice = &v["choices"][0];
    choice["text"]
        .as_str()
        .or_else(|| choice["message"]["content"].as_str())
        .map(str::to_string)
        .ok_or_else(|| fail(format!("no completion text in {v}")))
}

/// Runs every task in order; the first failed request aborts the run (no partial report).
pub async fn run_eval(
    base: &str,
    model: Option<&str>,
    tasks_file: &Path,
    tasks: &[EvalTask],
) -> Result<EvalReport, EvalError> {
    let base = base.trim_end_matches('/');
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|e| EvalError::Server(e.to_string()))?;
    let model = match model {
        Some(m) => m.to_string(),
        None => served_model(&client, base).await?,
    };
    let mut results = Vec::with_capacity(tasks.len());
    for task in tasks {
        let output = complete(&client, base, &model, task).await?;
        let correct = is_correct(task.match_kind, &task.answer, &output);
        results.push(TaskResult {
            id: task.id.clone(),
            correct,
            output,
        });
    }
    let correct = results.iter().filter(|r| r.correct).count();
    let total = results.len();
    Ok(EvalReport {
        model,
        tasks_file: tasks_file.display().to_string(),
        total,
        correct,
        accuracy: if total == 0 {
            0.0
        } else {
            correct as f64 / total as f64
        },
        results,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompareOutcome {
    pub baseline_accuracy: f64,
    pub candidate_accuracy: f64,
    pub max_drop: f64,
    pub pass: bool,
}

/// Gate: candidate accuracy ≥ baseline accuracy − max drop (1e-9 absorbs float rounding).
pub fn compare(baseline: &EvalReport, candidate: &EvalReport, max_drop: f64) -> CompareOutcome {
    CompareOutcome {
        baseline_accuracy: baseline.accuracy,
        candidate_accuracy: candidate.accuracy,
        max_drop,
        pass: candidate.accuracy + 1e-9 >= baseline.accuracy - max_drop,
    }
}

pub fn read_report(path: &Path) -> Result<EvalReport, EvalError> {
    let text = std::fs::read_to_string(path).map_err(|e| EvalError::Io {
        path: path.to_path_buf(),
        detail: e.to_string(),
    })?;
    serde_json::from_str(&text).map_err(|e| EvalError::Io {
        path: path.to_path_buf(),
        detail: e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_match_rules() {
        assert!(is_correct(MatchKind::Number, "1234", "1,234"));
        assert!(is_correct(MatchKind::Number, "1234", " 1234.\n"));
        assert!(is_correct(MatchKind::Number, "-5", "-5"));
        assert!(is_correct(MatchKind::Number, "2.5", "2.5."));
        assert!(!is_correct(MatchKind::Number, "18", "$18"));
        assert!(!is_correct(MatchKind::Number, "18", "18 apples"));
        assert!(!is_correct(MatchKind::Number, "18", "18.."));
        assert!(!is_correct(MatchKind::Number, "18", "180"));
        assert!(is_correct(MatchKind::Exact, "yes", " yes "));
        assert!(!is_correct(MatchKind::Exact, "yes", "Yes"));
    }
}
```

- [ ] Add the subcommands to `benches/turbine-bench/src/bin/turbine-golden.rs`: two variants on the P1 `Command` enum, their argument structs, two match arms `Command::Eval(args) => run_eval(args), Command::EvalCompare(args) => run_eval_compare(args),` and the two handlers (reuse the P1 `Output`/`--output` value enum if it has the same `Text`/`Json` variants, otherwise add this one):

```rust
// Imports (add those the P1 file lacks): use std::path::PathBuf; use std::process::ExitCode;
// use clap::ValueEnum; use turbine_bench::golden::eval;

// New subcommand variants on the P1 `Command` enum:
    /// Task-set accuracy against an OpenAI-compatible endpoint (greedy, sequential).
    Eval(EvalArgs),
    /// Gate: exit 0 when candidate accuracy ≥ baseline accuracy − max drop, else 1.
    EvalCompare(EvalCompareArgs),

// New match arms in main(): `Command::Eval(args) => run_eval(args),` and
// `Command::EvalCompare(args) => run_eval_compare(args),`. New items at module level:
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Output {
    Text,
    Json,
}

#[derive(clap::Args)]
struct EvalArgs {
    #[arg(long)]
    url: String,
    #[arg(long)]
    tasks: PathBuf,
    #[arg(long)]
    model: Option<String>,
    #[arg(long, value_enum, default_value_t = Output::Text)]
    output: Output,
}

#[derive(clap::Args)]
struct EvalCompareArgs {
    #[arg(long)]
    baseline: PathBuf,
    #[arg(long)]
    candidate: PathBuf,
    /// Defaults to quality.max_accuracy_drop's default (0.01).
    #[arg(long, default_value_t = turbine_core::config::QualityConfig::default().max_accuracy_drop)]
    max_drop: f64,
}

fn run_eval(args: EvalArgs) -> ExitCode {
    let tasks = match eval::load_tasks(&args.tasks) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("turbine-golden eval: {e}");
            return ExitCode::from(2);
        }
    };
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    match rt.block_on(eval::run_eval(
        &args.url,
        args.model.as_deref(),
        &args.tasks,
        &tasks,
    )) {
        Ok(report) => {
            match args.output {
                Output::Json => println!(
                    "{}",
                    serde_json::to_string_pretty(&report).expect("report serializes")
                ),
                Output::Text => println!(
                    "model {} tasks {}: accuracy {:.4} ({}/{})",
                    report.model, report.tasks_file, report.accuracy, report.correct, report.total
                ),
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("turbine-golden eval: {e}");
            ExitCode::from(2)
        }
    }
}

fn run_eval_compare(args: EvalCompareArgs) -> ExitCode {
    if !(0.0..=0.1).contains(&args.max_drop) {
        eprintln!("turbine-golden eval-compare: --max-drop must be between 0 and 0.1");
        return ExitCode::from(2);
    }
    let (baseline, candidate) = match (
        eval::read_report(&args.baseline),
        eval::read_report(&args.candidate),
    ) {
        (Ok(b), Ok(c)) => (b, c),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("turbine-golden eval-compare: {e}");
            return ExitCode::from(2);
        }
    };
    let o = eval::compare(&baseline, &candidate, args.max_drop);
    println!(
        "baseline accuracy {:.4}, candidate accuracy {:.4}, max drop {:.4}: {}",
        o.baseline_accuracy,
        o.candidate_accuracy,
        o.max_drop,
        if o.pass { "PASS" } else { "FAIL" }
    );
    ExitCode::from(if o.pass { 0 } else { 1 })
}
```

- [ ] Run `cargo test -p turbine-bench --test golden eval_accuracy_report` and `cargo test -p turbine-bench golden::eval::tests::numeric_match_rules` — expect PASS; `cargo test -p turbine-bench --test golden` — expect PASS (P1 `capture_and_compare_roundtrip` unaffected).
- [ ] Mutation check (do not commit): in `compare` replace `baseline.accuracy - max_drop` with `baseline.accuracy`; rerun `cargo test -p turbine-bench --test golden eval_accuracy_report` — expect FAIL with "left: (Some(1), Some(1))"; revert.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`.
- [ ] Commit: `feat(bench): add turbine-golden eval and eval-compare quality gate`.

## Task 6: Committed GSM8K-200 task set

Files: `scripts/eval/make-gsm8k-200.sh` (new: reproducible fixture generator, curl + jq, no Python), `tests/eval/gsm8k-200.jsonl` (new: 200 items), `tests/eval/NOTICE` (new: source, commit, transformation, MIT text), `benches/turbine-bench/tests/golden.rs` (append `mod phase8_eval_task_set`)
Interfaces: consumes Task 5 `load_tasks`, `MatchKind`, `normalize_number`. Produces `tests/eval/gsm8k-200.jsonl` (ids `gsm8k-test-0000` … `gsm8k-test-0199`, chat `messages`, numeric `answer`, `match: "number"`, `max_tokens: 32`), used by every lossy-format gate (Task 10).
Covers: S-4 — `cargo test -p turbine-bench --test golden eval_task_set_valid`.

- [ ] Write the failing test: append to `benches/turbine-bench/tests/golden.rs`:

```rust
mod phase8_eval_task_set {
    use std::path::Path;

    #[test]
    fn eval_task_set_valid() {
        use turbine_bench::golden::eval::{MatchKind, load_tasks, normalize_number};
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/eval");
        let path = root.join("gsm8k-200.jsonl");
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().count(), 200, "exactly 200 lines");
        let tasks = load_tasks(&path).expect("every line parses, ids unique");
        assert_eq!(tasks.len(), 200);
        let mut ids: Vec<&str> = tasks.iter().map(|t| t.id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 200, "unique ids");
        for t in &tasks {
            assert_eq!(t.match_kind, MatchKind::Number, "{}", t.id);
            assert!(
                normalize_number(&t.answer).is_some(),
                "{}: answer {:?} is not numeric",
                t.id,
                t.answer
            );
            assert!(t.max_tokens > 0, "{}", t.id);
        }
        let notice =
            std::fs::read_to_string(root.join("NOTICE")).expect("tests/eval/NOTICE exists");
        assert!(
            notice.contains("MIT License"),
            "NOTICE carries the source license"
        );
    }
}
```

Run `cargo test -p turbine-bench --test golden eval_task_set_valid` — expect FAIL with "No such file or directory".

- [ ] Create `scripts/eval/make-gsm8k-200.sh` (mode 755):

```bash
#!/usr/bin/env bash
# Regenerates tests/eval/gsm8k-200.jsonl: the first 200 items of the GSM8K test split
# (openai/grade-school-math, MIT), pinned by commit. Fixture-generation time only.
set -euo pipefail
COMMIT=3101c7d5072418e28b9008a6636bde82a006892c
URL="https://raw.githubusercontent.com/openai/grade-school-math/${COMMIT}/grade_school_math/data/test.jsonl"
OUT="${1:-tests/eval/gsm8k-200.jsonl}"
SRC="$(mktemp)"
TMP="$(mktemp)"
trap 'rm -f "$SRC" "$TMP"' EXIT
curl -fsSL -o "$SRC" "$URL"
head -n 200 "$SRC" | jq -c -s '
  to_entries[] | {
    id: ("gsm8k-test-" + ((.key | tostring) as $k | ("000" + $k)[-4:])),
    messages: [{role: "user", content: (.value.question
      + "\n\nSolve the problem. Reply with only the final answer as a number, with no units and no other text.")}],
    answer: (.value.answer | split("#### ")[1] | gsub(","; "") | gsub("^\\s+|\\s+$"; "")),
    match: "number",
    max_tokens: 32
  }' >"$TMP"
test "$(wc -l <"$TMP" | tr -d ' ')" = 200
mv "$TMP" "$OUT"
echo "wrote $OUT (200 items, commit $COMMIT)"
```

- [ ] Run `scripts/eval/make-gsm8k-200.sh` from the repository root — expect `wrote tests/eval/gsm8k-200.jsonl (200 items, commit 3101c7d5072418e28b9008a6636bde82a006892c)`; `shasum -a 256 tests/eval/gsm8k-200.jsonl` — expect `6b7bab085cd8d5a484e59283d809525bf5e3e21edc1d3275e4b9517316775321` (jq 1.7); `head -c 120 tests/eval/gsm8k-200.jsonl` — expect `{"id":"gsm8k-test-0000","messages":[{"role":"user","content":"Janet’s ducks lay 16 eggs per day.`.
- [ ] Create `tests/eval/NOTICE`:

```text
tests/eval/gsm8k-200.jsonl

Derived from the first 200 items of the GSM8K test split
(grade_school_math/data/test.jsonl) in https://github.com/openai/grade-school-math,
commit 3101c7d5072418e28b9008a6636bde82a006892c. Each question is wrapped in a single
user message with an answer-format instruction; the answer is the number after "#### "
with thousands separators removed. Regenerate with scripts/eval/make-gsm8k-200.sh.

The source is distributed under the MIT License:

MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] Run `cargo test -p turbine-bench --test golden eval_task_set_valid` — expect PASS.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`.
- [ ] Commit: `test(bench): commit the GSM8K-200 eval task set with its generator and NOTICE`.

## Task 7: Vendor-neutral public API guard

Files: `crates/turbine-kernels/tests/vendor_neutral_api.rs` (new), `crates/turbine-kernels/Cargo.toml` (dev-dependency `syn`), `Cargo.toml` (workspace dependency `syn = { version = "3.0.6", features = ["full", "visit"] }`)
Interfaces: reads the source trees `crates/{turbine-kernels,turbine-tensor,turbine-scheduler,turbine-kv,turbine-reliability}/src` from `CARGO_MANIFEST_DIR/..`; allows vendor names only as variants of `ExecutionBackend`, `CollectiveBackendKind`, `ProviderKind`. Produces nothing consumed by code; every track plan keeps it green (new kernels go behind the vendor-neutral traits).
Covers: S-5 — `cargo test -p turbine-kernels --test vendor_neutral_api` (tests `vendor_neutral_api` and `vendor_check_flags_leaks`). Intel re-entry needs no code: no Intel variant, discovery backend or build path is added (S-5, Out of scope).

- [ ] Write the test file `crates/turbine-kernels/tests/vendor_neutral_api.rs`:

```rust
//! Phase 8 S-5: no public signature of the core crates names a vendor-specific type.
//! Walks every `pub` item reachable from each crate root through `pub mod` declarations and
//! checks every type path, trait path and `pub use` path in its signature.
use std::path::{Path, PathBuf};

use syn::visit::Visit;

const CORE_CRATES: &[&str] = &[
    "turbine-kernels",
    "turbine-tensor",
    "turbine-scheduler",
    "turbine-kv",
    "turbine-reliability",
];

/// Vendor words; `level_zero` is checked as the word pair `level`, `zero`.
const VENDOR_WORDS: &[&str] = &["cuda", "hip", "rocm", "nccl", "rccl", "cublas", "sycl"];

/// Enums whose variants are allowed to carry vendor names (`ExecutionBackend::Hip`).
const BACKEND_ENUMS: &[&str] = &["ExecutionBackend", "CollectiveBackendKind", "ProviderKind"];

/// `CudaStream` → [cuda, stream]; `hip_event_t` → [hip, event, t]; `HIPBLASLt` → [hipblaslt].
fn words(ident: &str) -> Vec<String> {
    let mut out = Vec::new();
    for part in ident.split('_').filter(|p| !p.is_empty()) {
        let chars: Vec<char> = part.chars().collect();
        let mut cur = String::new();
        for (i, c) in chars.iter().enumerate() {
            let boundary = i > 0
                && c.is_uppercase()
                && (chars[i - 1].is_lowercase()
                    || chars.get(i + 1).is_some_and(|n| n.is_lowercase()))
                && !cur.chars().all(|x| x.is_uppercase());
            if boundary {
                out.push(std::mem::take(&mut cur).to_lowercase());
            }
            cur.push(*c);
        }
        out.push(cur.to_lowercase());
    }
    out
}

fn is_vendor_ident(ident: &str) -> bool {
    let w = words(ident);
    w.iter().any(|w| {
        VENDOR_WORDS
            .iter()
            .any(|v| w == v || (*v != "hip" && w.starts_with(v)) || w.starts_with("hipblas"))
    }) || w.windows(2).any(|p| p[0] == "level" && p[1] == "zero")
        || ident.to_lowercase().contains("levelzero")
}

#[derive(Default)]
struct PathCollector {
    leaks: Vec<String>,
}

impl PathCollector {
    fn check_path(&mut self, path: &syn::Path) {
        let segs: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
        for (i, seg) in segs.iter().enumerate() {
            let backend_variant = i > 0 && BACKEND_ENUMS.contains(&segs[i - 1].as_str());
            if is_vendor_ident(seg) && !backend_variant {
                self.leaks.push(segs.join("::"));
                return;
            }
        }
    }
}

impl<'ast> Visit<'ast> for PathCollector {
    fn visit_path(&mut self, path: &'ast syn::Path) {
        self.check_path(path);
        syn::visit::visit_path(self, path);
    }
}

fn is_pub(vis: &syn::Visibility) -> bool {
    matches!(vis, syn::Visibility::Public(_))
}

fn use_tree_idents(tree: &syn::UseTree, prefix: &str, out: &mut Vec<String>) {
    match tree {
        syn::UseTree::Path(p) => use_tree_idents(&p.tree, &format!("{prefix}{}::", p.ident), out),
        syn::UseTree::Name(n) => out.push(format!("{prefix}{}", n.ident)),
        syn::UseTree::Rename(r) => out.push(format!("{prefix}{} as {}", r.ident, r.rename)),
        syn::UseTree::Glob(_) => out.push(format!("{prefix}*")),
        syn::UseTree::Group(g) => g.items.iter().for_each(|t| use_tree_idents(t, prefix, out)),
    }
}

/// Leaks in `items`; `dir` is where `mod x;` files of these items live (None for inline sources).
fn item_leaks(items: &[syn::Item], dir: Option<&Path>, where_: &str, leaks: &mut Vec<String>) {
    let mut c = PathCollector::default();
    for item in items {
        match item {
            syn::Item::Fn(f) if is_pub(&f.vis) => c.visit_signature(&f.sig),
            syn::Item::Struct(s) if is_pub(&s.vis) => {
                c.visit_generics(&s.generics);
                s.fields
                    .iter()
                    .filter(|f| is_pub(&f.vis))
                    .for_each(|f| c.visit_type(&f.ty));
            }
            syn::Item::Enum(e) if is_pub(&e.vis) => {
                c.visit_generics(&e.generics);
                e.variants
                    .iter()
                    .flat_map(|v| v.fields.iter())
                    .for_each(|f| c.visit_type(&f.ty));
            }
            syn::Item::Union(u) if is_pub(&u.vis) => {
                u.fields.named.iter().for_each(|f| c.visit_type(&f.ty))
            }
            syn::Item::Trait(t) if is_pub(&t.vis) => {
                c.visit_generics(&t.generics);
                t.supertraits
                    .iter()
                    .for_each(|b| c.visit_type_param_bound(b));
                for ti in &t.items {
                    match ti {
                        syn::TraitItem::Fn(f) => c.visit_signature(&f.sig),
                        syn::TraitItem::Type(ty) => {
                            ty.bounds.iter().for_each(|b| c.visit_type_param_bound(b))
                        }
                        syn::TraitItem::Const(k) => c.visit_type(&k.ty),
                        _ => {}
                    }
                }
            }
            syn::Item::Type(t) if is_pub(&t.vis) => c.visit_type(&t.ty),
            syn::Item::Const(k) if is_pub(&k.vis) => c.visit_type(&k.ty),
            syn::Item::Static(s) if is_pub(&s.vis) => c.visit_type(&s.ty),
            syn::Item::Impl(i) => {
                let trait_impl = i.trait_.is_some();
                if let Some((path, _)) = &i.trait_ {
                    c.visit_path(path);
                    c.visit_type(&i.self_ty);
                }
                for ii in &i.items {
                    if let syn::ImplItem::Fn(f) = ii
                        && (trait_impl || is_pub(&f.vis))
                    {
                        c.visit_signature(&f.sig);
                    }
                }
            }
            syn::Item::Use(u) if is_pub(&u.vis) => {
                let mut paths = Vec::new();
                use_tree_idents(&u.tree, "", &mut paths);
                for p in paths {
                    if p.split("::")
                        .flat_map(|s| s.split(" as "))
                        .any(is_vendor_ident)
                    {
                        c.leaks.push(format!("pub use {p}"));
                    }
                }
            }
            syn::Item::Mod(m) if is_pub(&m.vis) => {
                let name = m.ident.to_string();
                let child_where = format!("{where_}::{name}");
                match (&m.content, dir) {
                    (Some((_, inner)), _) => item_leaks(
                        inner,
                        dir.map(|d| d.join(&name)).as_deref(),
                        &child_where,
                        leaks,
                    ),
                    (None, Some(d)) => {
                        let file = [d.join(format!("{name}.rs")), d.join(&name).join("mod.rs")]
                            .into_iter()
                            .find(|p| p.exists())
                            .unwrap_or_else(|| {
                                panic!(
                                    "{child_where}: no file for `pub mod {name};` in {}",
                                    d.display()
                                )
                            });
                        let parsed = parse(&file);
                        item_leaks(&parsed.items, Some(&d.join(&name)), &child_where, leaks);
                    }
                    (None, None) => {}
                }
            }
            _ => {}
        }
    }
    leaks.extend(c.leaks.into_iter().map(|l| format!("{where_}: {l}")));
}

fn parse(file: &Path) -> syn::File {
    let text = std::fs::read_to_string(file).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
    syn::parse_file(&text).unwrap_or_else(|e| panic!("{}: {e}", file.display()))
}

fn crate_src(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(name)
        .join("src")
}

#[test]
fn vendor_neutral_api() {
    let mut leaks = Vec::new();
    for name in CORE_CRATES {
        let src = crate_src(name);
        let root = parse(&src.join("lib.rs"));
        item_leaks(&root.items, Some(&src), &name.replace('-', "_"), &mut leaks);
    }
    assert!(
        leaks.is_empty(),
        "vendor types in core public signatures:\n{}",
        leaks.join("\n")
    );
}

#[test]
fn vendor_check_flags_leaks() {
    let src: syn::File = syn::parse_quote! {
        pub fn launch(s: CudaStream) {}
        pub struct Ctx { pub event: hip::Event, private: HipRaw }
        pub use self::ffi::HipStreamRaw;
        impl From<NcclError> for KernelError { fn from(e: NcclError) -> Self { todo!() } }
        pub trait Collective { fn comm(&self) -> RcclComm; }
        pub type Handle = level_zero::Device;
        pub mod inner { pub fn blas(h: CublasLtHandle) {} }
        // Allowed: neutral names, backend-enum variants, private items, words containing "hip".
        pub fn ok(b: ExecutionBackend, o: Ownership, r: Relationship) -> Option<u8> { None }
        pub const BACKEND: ExecutionBackend = ExecutionBackend::Hip;
        fn private(s: CudaStream) {}
        mod ffi { pub struct HipStreamRaw; }
    };
    let mut leaks = Vec::new();
    item_leaks(&src.items, None, "fixture", &mut leaks);
    let joined = leaks.join("\n");
    for expected in [
        "CudaStream",
        "hip::Event",
        "pub use self::ffi::HipStreamRaw",
        "NcclError",
        "RcclComm",
        "level_zero::Device",
        "fixture::inner: CublasLtHandle",
    ] {
        assert!(
            joined.contains(expected),
            "missing {expected} in:\n{joined}"
        );
    }
    for allowed in ["HipRaw", "Ownership", "Relationship", "ExecutionBackend"] {
        assert!(
            !joined.contains(allowed),
            "false positive {allowed} in:\n{joined}"
        );
    }
    assert!(
        !is_vendor_ident("ship_date")
            && !is_vendor_ident("Relationship")
            && !is_vendor_ident("chip")
    );
    assert!(
        is_vendor_ident("hipblasLtHandle")
            && is_vendor_ident("cudaStream_t")
            && is_vendor_ident("RocmPath")
    );
}
```

Run `cargo test -p turbine-kernels --test vendor_neutral_api` — expect FAIL with "cannot find module or crate `syn`".

- [ ] Add `syn = { version = "3.0.6", features = ["full", "visit"] }` under `[workspace.dependencies]` in `Cargo.toml` and `syn.workspace = true` under `[dev-dependencies]` in `crates/turbine-kernels/Cargo.toml`.
- [ ] Run `cargo test -p turbine-kernels --test vendor_neutral_api` — expect PASS. If `vendor_neutral_api` lists a leak from an earlier phase, move that type behind a vendor-neutral name (e.g. an opaque handle in `turbine_kernels::ffi`, which stays private) — never weaken `VENDOR_WORDS`.
- [ ] Mutation check (do not commit): append `pub struct HipStream; impl DeviceBuffer { pub fn hip_stream(&self) -> HipStream { HipStream } }` to `crates/turbine-tensor/src/buffer.rs`, rerun — expect FAIL with "vendor types in core public signatures:" naming `turbine_tensor::buffer: HipStream`; revert.
- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`.
- [ ] Commit: `test(kernels): guard core public signatures against vendor-specific types`.

## Task 8: Track start gate script

Files: `scripts/phase8-track-gate.sh` (new, mode 755: order + spec-check + scope checks), `benches/turbine-bench/tests/lab_scripts.rs` (append `mod phase8_track_gate_script`), `AGENTS.md` (Commands: phase 8 entries)
Interfaces: consumes Task 3 `turbine-server --support-matrix --output text` column layout, the procoder launcher `spec check <name>` (prints `… COMPLETE …`, exit 0), `.procoder/specs/<track>.md` with a `Status:` line and a `## In scope` section. Produces `scripts/phase8-track-gate.sh <phase-8a-quantization|phase-8b-speculative-decoding|phase-8c-model-families>` → `GATE PASS <track>` exit 0, `GATE FAIL <track>: …` lines exit 1, usage exit 2; env overrides `TURBINE_PROCODER_LAUNCHER`, `TURBINE_SPEC_DIR`, `TURBINE_SUPPORT_MATRIX`. Track closed ⇔ a `supported` row with a quantized weight format or `fp8_e4m3` KV (8a), with `speculative=draft` (8b).
Covers: S-1 (mechanism used by Tasks 9, 11, 12); test `cargo test -p turbine-bench --test lab_scripts phase8_track_gate`.

- [ ] Write the failing test: append to `benches/turbine-bench/tests/lab_scripts.rs` (add `tempfile` to `[dev-dependencies]` if absent):

```rust
mod phase8_track_gate_script {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    const GATE_MATRIX_HEADER: &str = "vendor  arch     architecture       weight_format  kv_format speculative status       reason\n";
    const GATE_BASELINE_ROW: &str =
        "amd     gfx1201  LlamaForCausalLM   bf16           bf16      none        supported    -\n";
    const GATE_QUANT_ROW: &str =
        "nvidia  sm_121   LlamaForCausalLM   modelopt_nvfp4 bf16      none        supported    -\n";
    const GATE_DRAFT_ROW: &str =
        "amd     gfx1201  LlamaForCausalLM   bf16           bf16      draft       supported    -\n";

    const GATE_SPEC_8A: &str = "# phase-8a-quantization\n\nStatus: complete\n\n## In scope\n\n- [S-1] modelopt NVFP4 + FP8 mixed precision.\n- [S-2] compressed-tensors nvfp4-pack-quantized.\n- [S-3] fp8_e4m3 KV cache.\n\n## Out of scope\n\n- MXFP4, GPTQ, AWQ, GGUF.\n";
    const GATE_SPEC_8B: &str = "# phase-8b-speculative-decoding\n\nStatus: complete\n\n## In scope\n\n- [S-1] Llama-3.2-1B-Instruct drafts for Llama-3.2-3B-Instruct behind a Proposer trait.\n\n## Out of scope\n\n- MTP heads, EAGLE heads, DFlash drafters.\n";
    const GATE_SPEC_8C: &str = "# phase-8c-model-families\n\nStatus: complete\n\n## In scope\n\n- [S-1] Qwen3 dense and Qwen3 MoE.\n- [S-2] Qwen3.5/3.6 hybrids with Gated DeltaNet; linear-attention kernel provider on AMD: own HIP kernels; linear-attention kernel provider on NVIDIA: FlashInfer.\n- [S-3] Mistral and Mixtral.\n\n## Out of scope\n\n- gpt-oss.\n";

    struct Gate {
        dir: tempfile::TempDir,
    }

    impl Gate {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            std::fs::create_dir(dir.path().join("specs")).unwrap();
            // Stub procoder launcher: COMPLETE iff the spec says `Status: complete`.
            let launcher = dir.path().join("launcher.sh");
            std::fs::write(
            &launcher,
            "#!/usr/bin/env bash\nf=\"$TURBINE_SPEC_DIR/$3.md\"\nif grep -qx 'Status: complete' \"$f\"; then echo \"spec $3: COMPLETE\"; else echo \"spec $3: NOT ready\"; exit 1; fi\n",
        )
        .unwrap();
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&launcher, std::fs::Permissions::from_mode(0o755)).unwrap();
            Gate { dir }
        }
        fn spec(&self, track: &str, body: &str) -> &Self {
            std::fs::write(
                self.dir.path().join("specs").join(format!("{track}.md")),
                body,
            )
            .unwrap();
            self
        }
        fn matrix(&self, rows: &[&str]) -> PathBuf {
            let p = self.dir.path().join(format!("matrix-{}.txt", rows.len()));
            std::fs::write(&p, format!("{GATE_MATRIX_HEADER}{}", rows.concat())).unwrap();
            p
        }
        fn run(&self, track: &str, matrix: &Path) -> (Option<i32>, String) {
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
            let out = Command::new("bash")
                .arg(root.join("scripts/phase8-track-gate.sh"))
                .arg(track)
                .env(
                    "TURBINE_PROCODER_LAUNCHER",
                    self.dir.path().join("launcher.sh"),
                )
                .env("TURBINE_SPEC_DIR", self.dir.path().join("specs"))
                .env("TURBINE_SUPPORT_MATRIX", matrix)
                .current_dir(&root)
                .output()
                .unwrap();
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            (out.status.code(), text)
        }
    }

    #[test]
    fn phase8_track_gate() {
        let g = Gate::new();
        let baseline = g.matrix(&[GATE_BASELINE_ROW]);
        let closed_8a = g.matrix(&[GATE_BASELINE_ROW, GATE_QUANT_ROW]);
        let closed_8b = g.matrix(&[GATE_BASELINE_ROW, GATE_QUANT_ROW, GATE_DRAFT_ROW]);

        // Usage and a missing spec.
        assert_eq!(g.run("phase-8d-multimodal", &baseline).0, Some(2));
        let (code, out) = g.run("phase-8a-quantization", &baseline);
        assert_eq!(code, Some(1), "{out}");
        assert!(out.contains("does not exist"), "{out}");

        // Track 1: complete in-scope spec passes; out-of-scope names in "Out of scope" are fine.
        g.spec("phase-8a-quantization", GATE_SPEC_8A);
        let (code, out) = g.run("phase-8a-quantization", &baseline);
        assert_eq!(code, Some(0), "{out}");
        assert!(out.contains("GATE PASS phase-8a-quantization"), "{out}");
        g.spec(
            "phase-8a-quantization",
            &GATE_SPEC_8A.replace(
                "## Out of scope",
                "- [S-4] MXFP4 weights.\n\n## Out of scope",
            ),
        );
        let (code, out) = g.run("phase-8a-quantization", &baseline);
        assert_eq!(code, Some(1), "{out}");
        assert!(out.contains("MXFP4"), "{out}");
        g.spec(
            "phase-8a-quantization",
            &GATE_SPEC_8A.replace("Status: complete", "Status: draft"),
        );
        let (code, out) = g.run("phase-8a-quantization", &baseline);
        assert_eq!(code, Some(1), "{out}");
        assert!(
            out.contains("not COMPLETE") && out.contains("Status line"),
            "{out}"
        );

        // Track 2: refused until track 1 has a supported row, then passes.
        g.spec("phase-8b-speculative-decoding", GATE_SPEC_8B);
        let (code, out) = g.run("phase-8b-speculative-decoding", &baseline);
        assert_eq!(code, Some(1), "{out}");
        assert!(
            out.contains("phase-8a-quantization has not closed"),
            "{out}"
        );
        assert_eq!(
            g.run("phase-8b-speculative-decoding", &closed_8a).0,
            Some(0)
        );
        g.spec(
            "phase-8b-speculative-decoding",
            &GATE_SPEC_8B.replace("behind a Proposer trait", "and an EAGLE head"),
        );
        let (code, out) = g.run("phase-8b-speculative-decoding", &closed_8a);
        assert_eq!(code, Some(1), "{out}");
        assert!(out.contains("EAGLE"), "{out}");

        // Track 3: refused until track 2 closed, and every S-8 family must be in scope.
        g.spec("phase-8c-model-families", GATE_SPEC_8C);
        let (code, out) = g.run("phase-8c-model-families", &closed_8a);
        assert_eq!(code, Some(1), "{out}");
        assert!(
            out.contains("phase-8b-speculative-decoding has not closed"),
            "{out}"
        );
        assert_eq!(g.run("phase-8c-model-families", &closed_8b).0, Some(0));
        g.spec(
            "phase-8c-model-families",
            &GATE_SPEC_8C.replace("Mistral and Mixtral", "Mistral"),
        );
        let (code, out) = g.run("phase-8c-model-families", &closed_8b);
        assert_eq!(code, Some(1), "{out}");
        assert!(out.contains("Mixtral"), "{out}");
    }
}
```

Run `cargo test -p turbine-bench --test lab_scripts phase8_track_gate` — expect FAIL with "left: Some(127)".

- [ ] Create `scripts/phase8-track-gate.sh` (bash 3.2-compatible: macOS `/bin/bash` runs it in the test), `chmod 755`:

```bash
#!/usr/bin/env bash
# Phase 8 track start gate (phase-8-expansion S-1): a track's implementation starts only when
#   1. the previous track closed (a `supported` support-matrix row carries its feature),
#   2. its spec passes `procoder spec check` (COMPLETE) and reads `Status: complete`,
#   3. its spec's "## In scope" section stays inside the umbrella scope (S-6, S-7, S-8).
# Usage: scripts/phase8-track-gate.sh <phase-8a-quantization|phase-8b-speculative-decoding|phase-8c-model-families>
# Env:   TURBINE_PROCODER_LAUNCHER  procoder launcher (default: the 3.7.0 plugin launcher)
#        TURBINE_SPEC_DIR           spec directory (default: .procoder/specs)
#        TURBINE_SUPPORT_MATRIX     file holding `turbine-server --support-matrix --output text`
#                                   (default: produced with `cargo run -q -p turbine-server`)
# Exit:  0 gate passed, 1 gate failed (every failure printed), 2 usage.
set -euo pipefail

track="${1:-}"
case "$track" in
phase-8a-quantization | phase-8b-speculative-decoding | phase-8c-model-families) ;;
*)
	echo "usage: $0 <phase-8a-quantization|phase-8b-speculative-decoding|phase-8c-model-families>" >&2
	exit 2
	;;
esac

launcher="${TURBINE_PROCODER_LAUNCHER:-$HOME/.claude/plugins/cache/procoder/procoder/3.7.0/hooks/launcher.sh}"
spec_dir="${TURBINE_SPEC_DIR:-.procoder/specs}"
spec="$spec_dir/$track.md"
failures=0

fail() {
	echo "GATE FAIL $track: $*"
	failures=$((failures + 1))
}

matrix_file="${TURBINE_SUPPORT_MATRIX:-}"
if [ -z "$matrix_file" ]; then
	matrix_file="$(mktemp)"
	trap 'rm -f "$matrix_file"' EXIT
	cargo run -q -p turbine-server -- --support-matrix --output text >"$matrix_file"
fi

# supported_rows <awk condition over $1..$7 = vendor arch architecture weight kv speculative status>
supported_rows() {
	awk -v cond="$1" 'NR > 1 && $7 == "supported" {
    if (cond == "quantized" && ($4 != "bf16" || $5 == "fp8_e4m3")) n++
    if (cond == "draft" && $6 == "draft") n++
  } END { print n + 0 }' "$matrix_file"
}

case "$track" in
phase-8b-speculative-decoding)
	[ "$(supported_rows quantized)" -gt 0 ] ||
		fail "track phase-8a-quantization has not closed: no supported row with a quantized weight format or fp8_e4m3 KV"
	;;
phase-8c-model-families)
	[ "$(supported_rows draft)" -gt 0 ] ||
		fail "track phase-8b-speculative-decoding has not closed: no supported row with speculative=draft"
	;;
esac

if [ ! -f "$spec" ]; then
	fail "$spec does not exist; write it with /procoder:spec $track"
else
	if check_out="$("$launcher" spec check "$track" 2>&1)" && printf '%s\n' "$check_out" | grep -q 'COMPLETE'; then
		:
	else
		fail "procoder spec check is not COMPLETE: $(printf '%s' "$check_out" | head -n 1)"
	fi
	grep -qx 'Status: complete' "$spec" || fail "Status line is not 'Status: complete'"

	in_scope="$(awk '/^## In scope/ { f = 1; next } /^## / { f = 0 } f' "$spec")"
	[ -n "$in_scope" ] || fail "no '## In scope' section"

	# need <description> <extended regex>: the In scope section must match (case-insensitive).
	need() { printf '%s\n' "$in_scope" | grep -Eqi "$2" || fail "In scope does not cover $1 (/$2/)"; }
	# refuse <description> <extended regex>: the In scope section must not match.
	refuse() { if printf '%s\n' "$in_scope" | grep -Eqi "$2"; then fail "In scope names $1, outside the umbrella scope"; fi; }

	case "$track" in
	phase-8a-quantization)
		[ "$(grep -c -E 'modelopt|compressed-tensors|fp8_e4m3' "$spec" || true)" -gt 0 ] ||
			fail "spec names none of modelopt, compressed-tensors, fp8_e4m3"
		refuse "MXFP4" 'mxfp4'
		refuse "GPTQ/AWQ/INT4" 'gptq|awq|int4'
		refuse "GGUF" 'gguf'
		refuse "FP8 block-scaled weights" 'block-scaled|block scaled'
		refuse "an NVFP4 KV cache" 'nvfp4 kv'
		;;
	phase-8b-speculative-decoding)
		grep -q 'Llama-3.2-1B-Instruct' "$spec" || fail "spec does not name Llama-3.2-1B-Instruct as the first draft model"
		refuse "MTP heads" '(^|[^a-z])mtp([^a-z]|$)'
		refuse "EAGLE" 'eagle'
		refuse "DFlash" 'dflash'
		;;
	phase-8c-model-families)
		need "Qwen3 dense" 'qwen3 dense|Qwen3ForCausalLM'
		need "Qwen3 MoE" 'qwen3 moe|Qwen3MoeForCausalLM'
		need "the Qwen3.5/3.6 hybrids" 'qwen3\.5|qwen3\.6|qwen3_5'
		need "Gated DeltaNet" 'gated deltanet'
		need "Mistral" 'mistral'
		need "Mixtral" 'mixtral'
		need "the AMD linear-attention kernel provider" 'linear-attention.*(amd|gfx1201|hip|rocm)|(amd|gfx1201|hip|rocm).*linear-attention'
		need "the NVIDIA linear-attention kernel provider" 'linear-attention.*(nvidia|sm_121|cuda)|(nvidia|sm_121|cuda).*linear-attention'
		refuse "gpt-oss" 'gpt-oss'
		;;
	esac
fi

if [ "$failures" -gt 0 ]; then
	echo "GATE FAIL $track: $failures check(s) failed"
	exit 1
fi
echo "GATE PASS $track"
```

- [ ] Run `cargo test -p turbine-bench --test lab_scripts phase8_track_gate` — expect PASS; `shfmt -d scripts/phase8-track-gate.sh scripts/eval/make-gsm8k-200.sh` — expect no output; `scripts/phase8-track-gate.sh phase-8b-speculative-decoding` on the real tree — expect `GATE FAIL phase-8b-speculative-decoding: track phase-8a-quantization has not closed: …` and exit 1.
- [ ] Append to the `## Commands` section of `AGENTS.md`:

```markdown
- Support matrix: `cargo run -q -p turbine-server -- --support-matrix [--output json]`; `--check-config` also resolves the configured row (exit 2 when unsupported).
- Quality gate: `cargo run -q -p turbine-bench --bin turbine-golden -- eval --url <base> --tasks tests/eval/gsm8k-200.jsonl --output json > tests/eval/<model-slug>/<engine>.json`, then `turbine-golden eval-compare --baseline <bf16.json> --candidate <quantized.json>` (exit 1 when the drop exceeds `quality.max_accuracy_drop`, default 0.01).
- Phase 8 track start: `scripts/phase8-track-gate.sh <phase-8a-quantization|phase-8b-speculative-decoding|phase-8c-model-families>` must print `GATE PASS <track>` before a track's implementation starts; tracks close with the runbook in `.procoder/plans/phase-8-expansion.md` Task 10.
```

- [ ] Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`.
- [ ] Commit: `feat(scripts): add the phase 8 track start gate`.

## Task 9: Gate — `phase-8a-quantization` spec written and checked before track 1 starts

Files: `.procoder/specs/phase-8a-quantization.md` (new, written through `/procoder:spec`)
Interfaces: consumes Task 8 `scripts/phase8-track-gate.sh`, the umbrella S-2 … S-6. Produces the track 1 spec that the separate `phase-8a-quantization` plan implements; it cites this spec for S-2 … S-5 instead of restating them and reserves only `kv.dtype` (adds `fp8_e4m3`) among config keys.
Covers: S-1, S-6 — acceptance criterion "Before track 1 implementation starts: `launcher.sh spec check phase-8a-quantization` exits 0 reporting COMPLETE, Status `complete`, `grep -c -E "modelopt|compressed-tensors|fp8_e4m3"` non-zero while the spec lists no format outside S-6".

- [ ] Red: run `scripts/phase8-track-gate.sh phase-8a-quantization` — expect FAIL with "does not exist; write it with /procoder:spec phase-8a-quantization".
- [ ] Write the spec with `/procoder:spec phase-8a-quantization` (interview the user; every open question goes to the user). Fixed inputs to carry into it: In scope = exactly S-6 (modelopt NVFP4 + FP8 mixed precision with per-layer resolution from `hf_quant_config.json` incl. `MIXED_PRECISION`, W4A16 NVFP4 group-16 experts and BF16 excluded modules; compressed-tensors `nvfp4-pack-quantized` with the divisor global-scale convention; FP8 e4m3 KV as `kv.dtype: fp8_e4m3`); the spec decides which Llama-3.2 / OLMoE checkpoints in those formats prove each format, the per-vendor kernel providers, and weight-only NVFP4 on RDNA4; it names its lab hosts, `scripts/lab/phase8-<track>-<host>.yaml` files on port 18000 and the exact `SUPPORT_MATRIX` rows it will turn `supported` (Task 10 procedure); MXFP4, GPTQ/AWQ/INT4, GGUF, FP8 block-scaled weights and NVFP4 KV appear only under "Out of scope" (the gate refuses them inside "## In scope").
- [ ] Run `"/Users/pascal/.claude/plugins/cache/procoder/procoder/3.7.0/hooks/launcher.sh" spec check phase-8a-quantization` — expect `spec phase-8a-quantization: COMPLETE` and exit 0.
- [ ] Run `grep -x 'Status: complete' .procoder/specs/phase-8a-quantization.md` — expect `Status: complete`; `grep -c -E "modelopt|compressed-tensors|fp8_e4m3" .procoder/specs/phase-8a-quantization.md` — expect a number ≥ 1.
- [ ] Green: run `scripts/phase8-track-gate.sh phase-8a-quantization` — expect `GATE PASS phase-8a-quantization`; paste the four outputs into this task's todo evidence.
- [ ] Format: `prettier --write .procoder/specs/phase-8a-quantization.md`; rerun the spec check — expect COMPLETE.
- [ ] Commit: `docs(spec): add the phase-8a-quantization track spec`.
- [ ] Hand-off: start the track with `/procoder:plan phase-8a-quantization` (a separate plan; not part of this one).

## Task 10: Track close runbook (run once per track, in S-1 order)

Files: `crates/turbine-core/src/support.rs` (the track's rows flip to `supported` — the edit itself is the last task of the track's own plan; this task verifies it), `tests/eval/<model-slug>/<engine>.json` (eval reports for lossy formats, new per gated combination), `tests/bench/phase8/<track>/<host>-<model-slug>-<engine>.json` (bench reports, new)
Interfaces: consumes Task 3 `--support-matrix --output json`, Task 5 `turbine-golden eval`/`eval-compare`, Task 6 `tests/eval/gsm8k-200.jsonl`, P1 `turbine-golden compare`, P0/P3 `turbine-bench`, P3 `scripts/overload-soak.sh <host> [--duration <dur>] [--model <path>]` with the requested addition `--config <yaml>` (see report), P1/P2b `scripts/lab-serve.sh <host> <config>` and `scripts/lab-serve.sh <host> --vllm <slug>` (vLLM on :18100), the track spec's `scripts/lab/phase8-<track>-<host>.yaml` (port 18000), `tests/golden/<model-slug>/{reference.jsonl,tolerance.json}` (phase-1 format; for quantized checkpoints captured once from the checkpoint's reference runtime by the track plan). `<track>` ∈ {`phase-8a-quantization`, `phase-8b-speculative-decoding`, `phase-8c-model-families`}; `<host>` ∈ {`novanas` → `http://192.168.10.203:18000`, `dgx-spark` → `http://192.168.10.246:18000`}.
Covers: S-3, S-6, S-7, S-8 — acceptance criterion "Manual track close, per track in S-1 order: overload-soak, golden compare, bench report, eval-compare for lossy formats, then `--support-matrix` shows the new rows `supported` only for the vendors where all four passed".

- [ ] Precondition: `scripts/phase8-track-gate.sh <track>` printed `GATE PASS <track>` before the track started, and every task of the track's own plan is closed with `cargo test --workspace` green on macOS and `scripts/lab-test.sh novanas` / `scripts/lab-test.sh dgx-spark` green (exit 0) for the vendors the track claims.
- [ ] ASK THE USER FIRST: every step below that runs on `novanas` uses the R9700 cards, and the soak and bench steps on the Sparks are benchmark/soak runs — ask the user to move production workloads/free memory and wait for the confirmation before each host's run. Spark correctness steps (golden compare, eval) proceed after the MemAvailable precondition that `scripts/lab-serve.sh` prints.
- [ ] Serve the track configuration per host: `scripts/lab-serve.sh <host> scripts/lab/phase8-<track>-<host>.yaml`; expect `curl -fsS http://<host-ip>:18000/ready` → `{"ready":true}` and `curl -fsS http://<host-ip>:18000/turbine/v1/status | jq .support` → the row being gated (status `experimental` while unvalidated rows are experimental in the track branch).
- [ ] Golden: `cargo run -q -p turbine-bench --bin turbine-golden -- compare --url http://<host-ip>:18000 --reference tests/golden/<model-slug>/reference.jsonl --tolerance tests/golden/<model-slug>/tolerance.json` — expect exit 0 (≥ 14 of 16 prompts with the first 32 greedy tokens identical, top-5 |Δlogprob| ≤ 0.15 nats).
- [ ] Quality (lossy formats only — every quantized weight format and `fp8_e4m3` KV): `cargo run -q -p turbine-bench --bin turbine-golden -- eval --url http://<host-ip>:18000 --tasks tests/eval/gsm8k-200.jsonl --output json > tests/eval/<model-slug>/turbine-<host>.json` for the quantized row, the same against the BF16 model of the same family (or `scripts/lab-serve.sh <host> --vllm <model-slug>` on :18100 for the same checkpoint) into `tests/eval/<baseline-slug>/<engine>-<host>.json`, then `cargo run -q -p turbine-bench --bin turbine-golden -- eval-compare --baseline tests/eval/<baseline-slug>/<engine>-<host>.json --candidate tests/eval/<model-slug>/turbine-<host>.json` — expect `… : PASS` and exit 0.
- [ ] Bench: `cargo run -q -p turbine-bench --bin turbine-bench -- --url http://<host-ip>:18000 --output json > tests/bench/phase8/<track>/<host>-<model-slug>-turbine.json` — expect exit 0; where a reference engine runs on that device, `scripts/lab-serve.sh <host> --vllm <model-slug>` and the same command against `http://<host-ip>:18100` into `…-vllm.json`, then `scripts/lab-serve.sh <host> --stop`; otherwise record the Turbine report as the baseline (TS §18).
- [ ] Soak: `scripts/overload-soak.sh <host> --config scripts/lab/phase8-<track>-<host>.yaml` — expect exit 0 and a passing JSON verdict under `target/soak/<host>-<timestamp>/`.
- [ ] No regression of TS §20 first-useful-release items on either vendor: with `scripts/lab/phase2-novanas-llama.yaml`, `phase2-novanas-olmoe.yaml`, `phase2b-spark-llama.yaml`, `phase2b-spark-olmoe.yaml` served in turn, `turbine-golden compare --url … --reference tests/golden/llama-3.2-3b-instruct/reference.jsonl` and `… olmoe-1b-7b-0125-instruct/reference.jsonl` — expect exit 0 each.
- [ ] Flip the rows (last task of the track's plan): rows turn `supported` only for vendors where all four items passed; the other vendor keeps an `unsupported` row whose reason names the failed item; `cargo test -p turbine-core support::` — expect PASS (update `baseline_rows_present` expectations in the same commit).
- [ ] Verify: `cargo run -q -p turbine-server -- --support-matrix --output json | jq -c '.rows[] | select(.status == "supported")'` — expect the four baseline rows plus exactly the track's validated rows.
- [ ] Record: paste every command output above into the track-close todo's evidence (`/procoder:todo`), commit the reports: `test(phase8): record <track> exit-gate reports`; `scripts/lab-serve.sh <host> --stop` on every host used.

## Task 11: Gate — `phase-8b-speculative-decoding` spec written and checked before track 2 starts

Files: `.procoder/specs/phase-8b-speculative-decoding.md` (new, written through `/procoder:spec`)
Interfaces: consumes Task 8 gate script, Task 10 executed for `phase-8a-quantization` (≥ 1 `supported` quantized or `fp8_e4m3` row), Task 1 `SpeculativeConfig`/`SpeculativeMethod` (the track adds `num_tokens` ≤ 8, `draft_model_path`, `min_acceptance`), Task 2 draft refusal row. Produces the track 2 spec implemented by the separate `phase-8b-speculative-decoding` plan.
Covers: S-1, S-7 — acceptance criterion "Before track 2 implementation starts (and only after track 1 closed): spec check COMPLETE with Status `complete`, names `Llama-3.2-1B-Instruct` as the first draft model and no MTP, EAGLE or DFlash proposer".

- [ ] Red: run `scripts/phase8-track-gate.sh phase-8b-speculative-decoding` — expect FAIL with "does not exist" (and, until track 1 closed, "track phase-8a-quantization has not closed").
- [ ] Write the spec with `/procoder:spec phase-8b-speculative-decoding` (interview the user). Fixed inputs: In scope = exactly S-7 (separate draft model sharing the target's tokenizer behind a `Proposer` trait, first pairing `meta-llama/Llama-3.2-1B-Instruct` → `meta-llama/Llama-3.2-3B-Instruct`, slug `llama-3.2-1b-instruct`; verifier in `turbine-scheduler` scoring k ≤ 8 proposals in one target forward with standard speculative rejection sampling, greedy = exact prefix match; rollback by truncating KV blocks past the last accepted position; per-request disable below `speculative.min_acceptance`, global disable at pressure ORANGE or worse; draft memory reserved through the Phase 3 budget); config keys exactly `speculative.{method, num_tokens, draft_model_path, min_acceptance}`; track metrics with bounded labels; colocated only (no PD/PP); MTP, EAGLE and DFlash only under "Out of scope"; linear-attention state rollback deferred to track 3.
- [ ] Run `"/Users/pascal/.claude/plugins/cache/procoder/procoder/3.7.0/hooks/launcher.sh" spec check phase-8b-speculative-decoding` — expect `COMPLETE`, exit 0; `grep -x 'Status: complete' .procoder/specs/phase-8b-speculative-decoding.md` — expect a match; `grep -c 'Llama-3.2-1B-Instruct' .procoder/specs/phase-8b-speculative-decoding.md` — expect ≥ 1.
- [ ] Green: `scripts/phase8-track-gate.sh phase-8b-speculative-decoding` — expect `GATE PASS phase-8b-speculative-decoding`; paste the outputs into this task's todo evidence.
- [ ] Format: `prettier --write .procoder/specs/phase-8b-speculative-decoding.md`; rerun the spec check — expect COMPLETE.
- [ ] Commit: `docs(spec): add the phase-8b-speculative-decoding track spec`.
- [ ] Hand-off: `/procoder:plan phase-8b-speculative-decoding` (separate plan); close it with Task 10.

## Task 12: Gate — `phase-8c-model-families` spec written and checked before track 3 starts

Files: `.procoder/specs/phase-8c-model-families.md` (new, written through `/procoder:spec`)
Interfaces: consumes Task 8 gate script, Task 10 executed for `phase-8b-speculative-decoding` (≥ 1 `supported` row with `speculative=draft`), contract §10 `registry` (`ArchitectureEntry { architectures0, model_type, parse, weight_map, build_executor, kv_layout }`, `lookup(architectures0, model_type)`), contract C-22 (TKV1 segment kinds 2/3 reserved for linear-attention state). Produces the track 3 spec implemented by the separate `phase-8c-model-families` plan.
Covers: S-1, S-8 — acceptance criterion "Before track 3 implementation starts (and only after track 2 closed): spec check COMPLETE with Status `complete`, covering Qwen3 dense, Qwen3 MoE, the Qwen3.5/3.6 hybrids with the linear-attention kernel provider decided per vendor, Mistral and Mixtral".

- [ ] Red: run `scripts/phase8-track-gate.sh phase-8c-model-families` — expect FAIL with "does not exist" (and, until track 2 closed, "track phase-8b-speculative-decoding has not closed").
- [ ] Write the spec with `/procoder:spec phase-8c-model-families` (interview the user). Fixed inputs: In scope = exactly S-8 (architecture registry keyed on `architectures[0]` and `model_type`, top level and nested `text_config`; unregistered → exit 1 naming it; Qwen3 dense, Qwen3 MoE, the Qwen3.5/3.6 Gated DeltaNet hybrids including `Qwen3.6-35B-A3B` text-only with vision weights skipped, TKV1 recurrent/conv-state segments and speculative state rollback, Mistral and Mixtral); one line per vendor naming the linear-attention kernel provider (the gate looks for `linear-attention` together with `AMD`/`gfx1201`/`hip`/`rocm` and with `NVIDIA`/`sm_121`/`cuda` in "## In scope"); per family: its golden fixtures under `tests/golden/<model-slug>/` and whether it runs on one device, Phase 5 TP, Phase 7 PP/EP or an in-scope quantized checkpoint; gpt-oss only under "Out of scope".
- [ ] Run `"/Users/pascal/.claude/plugins/cache/procoder/procoder/3.7.0/hooks/launcher.sh" spec check phase-8c-model-families` — expect `COMPLETE`, exit 0; `grep -x 'Status: complete' .procoder/specs/phase-8c-model-families.md` — expect a match.
- [ ] Green: `scripts/phase8-track-gate.sh phase-8c-model-families` — expect `GATE PASS phase-8c-model-families`; paste the outputs into this task's todo evidence.
- [ ] Format: `prettier --write .procoder/specs/phase-8c-model-families.md`; rerun the spec check — expect COMPLETE.
- [ ] Commit: `docs(spec): add the phase-8c-model-families track spec`.
- [ ] Hand-off: `/procoder:plan phase-8c-model-families` (separate plan); close it with Task 10; Phase 8 closes when Task 10 has run for all three tracks.
