//! Support matrix (phase 8 S-2, Phase 2m S-11): one declarative table of
//! `(vendor, arch, architecture, weight_format, kv_format, speculative) → status`.
//! Tracks add rows only when their exit gate (S-3) passed.
//!
//! The columns are plain names, so this crate needs neither the backend nor the family
//! registry: `vendor` is `ExecutionBackend::vendor()` of the configured backend, `arch` the
//! device architecture, `architecture` a Hugging Face class name a `ModelFamily` claims. The
//! server builds the key and refuses an `unsupported` resolution as a configuration error.
use std::borrow::Cow;
use std::fmt;

use serde::Serialize;

use crate::config::ConfigError;

/// Bounded `vendor` column values; `cpu` is the phase-1 CPU reference provider.
pub const VENDORS: &[&str] = &["amd", "nvidia", "cpu"];

/// The vendor column of a backend that runs on the host (no device): its `arch` column is the
/// same word, so a host key is complete without device discovery.
pub const HOST_VENDOR: &str = "cpu";

/// Column value meaning "any" in a row and "not known yet" in a `--check-config` key.
pub const WILDCARD: &str = "*";

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum WeightFormatColumn {
    Bf16,
    ModeloptNvfp4,
    ModeloptFp8,
    ModeloptMixed,
    CtNvfp4,
}

impl WeightFormatColumn {
    pub const ALL: [WeightFormatColumn; 5] = [
        WeightFormatColumn::Bf16,
        WeightFormatColumn::ModeloptNvfp4,
        WeightFormatColumn::ModeloptFp8,
        WeightFormatColumn::ModeloptMixed,
        WeightFormatColumn::CtNvfp4,
    ];
    pub fn as_str(self) -> &'static str {
        match self {
            WeightFormatColumn::Bf16 => "bf16",
            WeightFormatColumn::ModeloptNvfp4 => "modelopt_nvfp4",
            WeightFormatColumn::ModeloptFp8 => "modelopt_fp8",
            WeightFormatColumn::ModeloptMixed => "modelopt_mixed",
            WeightFormatColumn::CtNvfp4 => "ct_nvfp4",
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
    pub weight_format: WeightFormatColumn,
    pub kv_format: KvFormatColumn,
    pub speculative: SpeculativeColumn,
}

/// One row's match pattern: `None` = `*`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SupportKeyPattern {
    pub vendor: Option<&'static str>,
    pub arch: Option<&'static str>,
    pub architecture: Option<&'static str>,
    pub weight_format: Option<WeightFormatColumn>,
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
    weight_format: Option<WeightFormatColumn>,
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

const BF16: Option<WeightFormatColumn> = Some(WeightFormatColumn::Bf16);
const KV_BF16: Option<KvFormatColumn> = Some(KvFormatColumn::Bf16);
const NO_SPEC: Option<SpeculativeColumn> = Some(SpeculativeColumn::None);
const QUANT_REASON: &str =
    "quantized checkpoints are not validated yet (track phase-8a-quantization)";
const FAMILY_REASON: &str =
    "this model family is not validated on this vendor yet (track phase-8c-model-families)";

/// The track 3 families on a GPU vendor, BF16: refused until the track closes (Task 10 of the
/// umbrella plan flips each validated row to `supported`). The CPU reference provider serves
/// them through its `experimental` row.
const fn family_row(vendor: &'static str, architecture: &'static str) -> SupportRow {
    row(
        Some(vendor),
        None,
        Some(architecture),
        BF16,
        KV_BF16,
        NO_SPEC,
        unsupported(FAMILY_REASON),
    )
}

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
    // CPU reference provider: tests and tiny checkpoints only; `experimental` by user decision
    // (2026-09-25, `.procoder/ask/decisions.md`).
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
    family_row("amd", "Qwen3ForCausalLM"),
    family_row("amd", "Qwen3MoeForCausalLM"),
    family_row("amd", "MistralForCausalLM"),
    family_row("amd", "MixtralForCausalLM"),
    family_row("nvidia", "Qwen3ForCausalLM"),
    family_row("nvidia", "Qwen3MoeForCausalLM"),
    family_row("nvidia", "MistralForCausalLM"),
    family_row("nvidia", "MixtralForCausalLM"),
    row(
        None,
        None,
        None,
        Some(WeightFormatColumn::ModeloptNvfp4),
        KV_BF16,
        NO_SPEC,
        unsupported(QUANT_REASON),
    ),
    row(
        None,
        None,
        None,
        Some(WeightFormatColumn::ModeloptFp8),
        KV_BF16,
        NO_SPEC,
        unsupported(QUANT_REASON),
    ),
    row(
        None,
        None,
        None,
        Some(WeightFormatColumn::ModeloptMixed),
        KV_BF16,
        NO_SPEC,
        unsupported(QUANT_REASON),
    ),
    row(
        None,
        None,
        None,
        Some(WeightFormatColumn::CtNvfp4),
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
        } else if self.weight_format != WeightFormatColumn::Bf16 || unknown_architecture {
            "model.path"
        } else {
            "execution.backend"
        }
    }

    /// Key before device discovery (startup and `--check-config`): `vendor` is the configured
    /// execution backend's vendor column, `arch` is unknown ([`WILDCARD`]) except on the
    /// [`HOST_VENDOR`], and `architecture` is the model's Hugging Face architecture name, or
    /// [`WILDCARD`] when `config.json` cannot be read yet.
    pub fn before_discovery(vendor: &str, architecture: Option<&str>) -> SupportKey {
        let arch = if vendor == HOST_VENDOR {
            HOST_VENDOR
        } else {
            WILDCARD
        };
        SupportKey::bf16(vendor, arch, architecture.unwrap_or(WILDCARD))
    }

    /// Key with the BF16 weight and KV columns and no speculation — the only format columns
    /// Phase 2m serves (`speculative.method` and the quantized formats stay with Phase 8).
    pub fn bf16(vendor: &str, arch: &str, architecture: &str) -> SupportKey {
        SupportKey {
            vendor: vendor.to_string(),
            arch: arch.to_string(),
            architecture: architecture.to_string(),
            weight_format: WeightFormatColumn::Bf16,
            kv_format: KvFormatColumn::Bf16,
            speculative: SpeculativeColumn::None,
        }
    }
}

/// `<vendor>/<arch>/<architecture>/<weight_format>/<kv_format>/<speculative>`.
impl fmt::Display for SupportKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}/{}/{}/{}/{}/{}",
            self.vendor,
            self.arch,
            self.architecture,
            self.weight_format.as_str(),
            self.kv_format.as_str(),
            self.speculative.as_str()
        )
    }
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
            weight_format: s(self.key.weight_format.map(WeightFormatColumn::as_str)),
            kv_format: s(self.key.kv_format.map(KvFormatColumn::as_str)),
            speculative: s(self.key.speculative.map(SpeculativeColumn::as_str)),
            status: self.status.as_str(),
            reason: self.status.reason().map(str::to_string),
        }
    }
}

/// Parallel-mode combinations a model architecture is refused in (`unsupported`, exit 2 at
/// startup with the reason code), on every vendor. `modes` names the combination: `ep+tp` is
/// expert parallelism > 1 together with tensor parallelism > 1. Each mode on its own stays
/// supported.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ParallelRefusal {
    pub architecture: &'static str,
    pub modes: &'static str,
    pub reason: &'static str,
}

/// User decision "P5 exit: OLMoE with expert × tensor parallelism" (2026-09-28): OLMoE at
/// ep 2 × tp 2 drifts past its golden tolerance against the transformers reference (p10 likely
/// |Δ| 1.031 > 1.01, and p08) while ep 2 and tp 2 alone pass; refused until the Phase 7
/// investigation of the OLMoE tensor-parallel drift (expansion umbrella question (d)).
pub static PARALLEL_REFUSALS: &[ParallelRefusal] = &[ParallelRefusal {
    architecture: "OlmoeForCausalLM",
    modes: "ep+tp",
    reason: "olmoe_ep_tp_drift: OLMoE with expert and tensor parallelism together drifts past \
             its golden tolerance (phase 7 investigates the OLMoE tensor-parallel drift); use \
             expert_parallel_size or tensor_parallel_size alone",
}];

/// The refusal of `architecture` at tensor-parallel size `tp` and expert-parallel size `ep`,
/// if [`PARALLEL_REFUSALS`] has one.
pub fn parallel_refusal(architecture: &str, tp: u32, ep: u32) -> Option<&'static ParallelRefusal> {
    PARALLEL_REFUSALS.iter().find(|r| {
        r.architecture == architecture
            && match r.modes {
                "ep+tp" => ep > 1 && tp > 1,
                _ => false,
            }
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn key(
        vendor: &str,
        arch: &str,
        architecture: &str,
        w: WeightFormatColumn,
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
        w: Option<WeightFormatColumn>,
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
        use WeightFormatColumn as W;
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
        assert!(
            err.contains("amd/gfx1100/LlamaForCausalLM/bf16/bf16/none is unsupported"),
            "{err}"
        );

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
        use WeightFormatColumn as W;
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
        // Track 3 families: refused on both GPU vendors (any arch), experimental on the CPU
        // reference provider.
        for architecture in [
            "Qwen3ForCausalLM",
            "Qwen3MoeForCausalLM",
            "MistralForCausalLM",
            "MixtralForCausalLM",
        ] {
            for (vendor, arch) in [("amd", "gfx1201"), ("nvidia", "sm_121"), ("amd", "gfx942")] {
                let k = key(vendor, arch, architecture, W::Bf16, K::Bf16, S::None);
                let status = resolve(&k);
                assert_eq!(status.as_str(), "unsupported", "{k}");
                assert!(
                    status.reason().unwrap().contains("phase-8c-model-families"),
                    "{k}: {status:?}"
                );
            }
            let k = key("cpu", "cpu", architecture, W::Bf16, K::Bf16, S::None);
            assert_eq!(resolve(&k), SupportStatus::Experimental, "{k}");
        }
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

    /// Phase 2m: keys are built from the backend's vendor and the family's HF name; before
    /// discovery a device arch is unknown, a host backend's arch is its vendor word.
    #[test]
    fn keys_before_discovery() {
        let k = SupportKey::before_discovery("amd", Some("LlamaForCausalLM"));
        assert_eq!(k.to_string(), "amd/*/LlamaForCausalLM/bf16/bf16/none");
        assert!(k.is_partial());
        assert_eq!(check(k).unwrap().status, SupportStatus::Supported);

        let k = SupportKey::before_discovery("amd", Some("Qwen3ForCausalLM"));
        let err = check(k).unwrap_err();
        assert_eq!(err.key(), Some("execution.backend"));
        let msg = err.to_string();
        assert!(
            msg.contains("support matrix: amd/*/Qwen3ForCausalLM/bf16/bf16/none is unsupported: "),
            "{msg}"
        );

        let k = SupportKey::before_discovery(HOST_VENDOR, None);
        assert_eq!(k.to_string(), "cpu/cpu/*/bf16/bf16/none");
        assert_eq!(check(k).unwrap().status, SupportStatus::Experimental);
        let k = SupportKey::before_discovery(HOST_VENDOR, Some("Qwen3ForCausalLM"));
        assert!(!k.is_partial());
        assert_eq!(resolve(&k), SupportStatus::Experimental);

        let k = SupportKey::bf16("amd", "gfx1201", "OlmoeForCausalLM");
        assert_eq!(resolve(&k), SupportStatus::Supported);
        assert!(VENDORS.contains(&HOST_VENDOR));
    }
}
