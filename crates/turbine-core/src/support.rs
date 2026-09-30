//! Support matrix (umbrella phase-6-8-expansion S-2, Phase 2m S-11): one declarative table of
//! `(vendor, arch, architecture, weight_format, kv_format, speculative) → status`.
//! Tracks add rows only when their exit gate (S-3) passed. A vendor whose phase is deferred
//! ([`DEFERRED_VENDORS`]: NVIDIA, `phase-2b-nvidia`) is refused before any row is consulted.
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
    /// FP8 e4m3 weights with per-tensor or per-channel scales (Phase 6a).
    Fp8,
    /// FP8 e4m3 weights with 128 × 128 block scales (Phase 6a).
    Fp8Block,
    /// MXFP4 weight-only (W4A16): compressed-tensors, OpenAI native, Quark weight-only (Phase 6a).
    Mxfp4,
    /// MXFP4 weights with emulated MXFP4 activations (Quark W4A4, Phase 6a; user decision
    /// 2026-09-28, Q6).
    Mxfp4A4,
    /// INT4 AWQ, group-wise with zero points (Phase 6a).
    AwqInt4,
    /// INT4 GPTQ, group-wise (Phase 6a).
    GptqInt4,
    /// Reserved for the deferred `phase-2b-nvidia` (NVFP4 arrives with NVIDIA support).
    ModeloptNvfp4,
    ModeloptFp8,
    ModeloptMixed,
    CtNvfp4,
}

impl WeightFormatColumn {
    pub const ALL: [WeightFormatColumn; 11] = [
        WeightFormatColumn::Bf16,
        WeightFormatColumn::Fp8,
        WeightFormatColumn::Fp8Block,
        WeightFormatColumn::Mxfp4,
        WeightFormatColumn::Mxfp4A4,
        WeightFormatColumn::AwqInt4,
        WeightFormatColumn::GptqInt4,
        WeightFormatColumn::ModeloptNvfp4,
        WeightFormatColumn::ModeloptFp8,
        WeightFormatColumn::ModeloptMixed,
        WeightFormatColumn::CtNvfp4,
    ];
    /// The Phase 6a formats (each refused until its proof passes).
    pub const PHASE_6A: [WeightFormatColumn; 6] = [
        WeightFormatColumn::Fp8,
        WeightFormatColumn::Fp8Block,
        WeightFormatColumn::Mxfp4,
        WeightFormatColumn::Mxfp4A4,
        WeightFormatColumn::AwqInt4,
        WeightFormatColumn::GptqInt4,
    ];
    /// The values reserved for the deferred `phase-2b-nvidia`.
    pub const RESERVED_NVIDIA: [WeightFormatColumn; 4] = [
        WeightFormatColumn::ModeloptNvfp4,
        WeightFormatColumn::ModeloptFp8,
        WeightFormatColumn::ModeloptMixed,
        WeightFormatColumn::CtNvfp4,
    ];
    pub fn as_str(self) -> &'static str {
        match self {
            WeightFormatColumn::Bf16 => "bf16",
            WeightFormatColumn::Fp8 => "fp8",
            WeightFormatColumn::Fp8Block => "fp8_block",
            WeightFormatColumn::Mxfp4 => "mxfp4",
            WeightFormatColumn::Mxfp4A4 => "mxfp4_a4",
            WeightFormatColumn::AwqInt4 => "awq_int4",
            WeightFormatColumn::GptqInt4 => "gptq_int4",
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
    /// TurboQuant 4-bit L0 pages (P6b S-5).
    Tq4,
    /// TurboQuant 2-bit L0 pages (P6b S-5).
    Tq2,
}

impl KvFormatColumn {
    pub const ALL: [KvFormatColumn; 4] = [
        KvFormatColumn::Bf16,
        KvFormatColumn::Fp8E4m3,
        KvFormatColumn::Tq4,
        KvFormatColumn::Tq2,
    ];
    pub fn as_str(self) -> &'static str {
        match self {
            KvFormatColumn::Bf16 => "bf16",
            KvFormatColumn::Fp8E4m3 => "fp8_e4m3",
            KvFormatColumn::Tq4 => "tq4",
            KvFormatColumn::Tq2 => "tq2",
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
    "this quantized weight format is not validated yet (track phase-6a-quantization)";
const RESERVED_NVIDIA_REASON: &str = "this weight format is reserved for the deferred phase-2b-nvidia (NVFP4 arrives with NVIDIA support)";
const TQ_KV_REASON: &str =
    "TurboQuant KV pages are not validated yet (track phase-6b-kv-compression)";
const FAMILY_REASON: &str =
    "this model family is not validated on this vendor yet (track phase-7-model-families)";
/// Why every `nvidia` key is refused while Phase 2b is deferred (decision 2026-09-28).
pub const NVIDIA_REASON: &str =
    "NVIDIA execution is deferred (phase-2b-nvidia, on hold until the user lifts it)";

/// Vendors whose phase is deferred: any key of such a vendor resolves `unsupported` with the
/// reason, before the rows are consulted, so no row can make it supported by accident. The
/// table still lists the vendor's former baseline rows as `unsupported` so `--support-matrix`
/// shows them.
pub static DEFERRED_VENDORS: &[(&str, &str)] = &[("nvidia", NVIDIA_REASON)];

/// The deferral reason of `vendor`, if its phase is deferred.
pub fn deferred_vendor(vendor: &str) -> Option<&'static str> {
    DEFERRED_VENDORS
        .iter()
        .find(|(v, _)| *v == vendor)
        .map(|(_, reason)| *reason)
}

/// The Phase 7 families on the AMD GPU vendor, BF16: refused until the track closes (Task 10 of
/// the umbrella plan flips each validated row to `supported`). The CPU reference provider serves
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

/// A weight format refused on every vendor and architecture (BF16 KV, no speculation).
const fn format_row(format: WeightFormatColumn, reason: &'static str) -> SupportRow {
    row(
        None,
        None,
        None,
        Some(format),
        KV_BF16,
        NO_SPEC,
        unsupported(reason),
    )
}

/// A Phase 6a weight format on gfx1201 Llama with BF16 KV during its proof: `experimental`.
const fn gfx1201_quant_row(format: WeightFormatColumn) -> SupportRow {
    row(
        Some("amd"),
        Some("gfx1201"),
        Some("LlamaForCausalLM"),
        Some(format),
        KV_BF16,
        NO_SPEC,
        SupportStatus::Experimental,
    )
}

/// A Phase 6a weight format on the CPU reference provider: `experimental` (tests and tiny
/// checkpoints), more specific than the format's `unsupported` row.
const fn cpu_quant_row(format: WeightFormatColumn, kv: KvFormatColumn) -> SupportRow {
    row(
        Some("cpu"),
        None,
        None,
        Some(format),
        Some(kv),
        NO_SPEC,
        SupportStatus::Experimental,
    )
}

/// A former NVIDIA baseline row, refused while `phase-2b-nvidia` is deferred.
const fn nvidia_row(architecture: &'static str) -> SupportRow {
    row(
        Some("nvidia"),
        Some("sm_121"),
        Some(architecture),
        BF16,
        KV_BF16,
        NO_SPEC,
        unsupported(NVIDIA_REASON),
    )
}

/// The support matrix. Order is irrelevant: the most specific matching row wins, and
/// `support::tests::resolution_and_refusal` proves no two overlapping rows tie.
pub static SUPPORT_MATRIX: &[SupportRow] = &[
    // Phase 1–5 baseline (S-2).
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
    // Deferred with phase-2b-nvidia (decision 2026-09-28); DEFERRED_VENDORS refuses every other
    // nvidia key with the same reason.
    nvidia_row("LlamaForCausalLM"),
    nvidia_row("OlmoeForCausalLM"),
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
    // FP8 KV on gfx1201 (Phase 6a S-13, S-14). Llama: `supported` after the Task 24 proof, full
    // GSM8K at c16 (1,319 items) BF16 KV 0.7801, FP8 KV 0.7885, golden c1 / c16 16/16 against
    // tests/golden/llama-3.2-3b-instruct-fp8kv (7ad4b03). OLMoE: `experimental` for 6a (user
    // decision 2026-09-30, "OLMoE FP8 KV: golden against the emulated-KV reference misses", B):
    // full GSM8K 0.6603 vs 0.6459 (drop 0.0144, accepted as noise 2026-09-29), but golden c1 / c16
    // 13/16 against tests/golden/olmoe-1b-7b-0125-instruct-fp8kv (need 14; p03 tail 5.38). The
    // checkpoint ships no K/V scales; calibrated V scales and the golden miss are a Phase 7 item.
    row(
        Some("amd"),
        Some("gfx1201"),
        Some("LlamaForCausalLM"),
        BF16,
        Some(KvFormatColumn::Fp8E4m3),
        NO_SPEC,
        SupportStatus::Supported,
    ),
    row(
        Some("amd"),
        Some("gfx1201"),
        Some("OlmoeForCausalLM"),
        BF16,
        Some(KvFormatColumn::Fp8E4m3),
        NO_SPEC,
        SupportStatus::Experimental,
    ),
    // FP8 weights on gfx1201 Llama (Phase 6a Task 14): `supported`. The one `fp8` column covers
    // both checkpoints the proof ran. Dynamic per-token (RedHatAI FP8-dynamic): full GSM8K at c16
    // Turbine 0.7703 vs vLLM 0.7832 on the same checkpoint, a drop of 0.0129 over the 0.01 bound
    // but not significant (McNemar exact p = 0.152, 95% CI -0.0295..+0.0037), accepted as noise
    // (user decision 2026-09-29, "FP8-dynamic: accept the full-GSM8K drop as noise"); golden
    // 16/16 at c1 and c16. Static per-tensor (RedHatAI FP8): golden 16/16 at c1 and c16 against
    // the self-spread tolerance (likely 1.36, tail 2.99). The c1 ITL target stays a perf item.
    row(
        Some("amd"),
        Some("gfx1201"),
        Some("LlamaForCausalLM"),
        Some(WeightFormatColumn::Fp8),
        KV_BF16,
        NO_SPEC,
        SupportStatus::Supported,
    ),
    // fp8_block (plan Task 15) proof passed 2026-09-29 (t15-proof, novanas): weight_bytes exact
    // (3,607,615,488), no fp8_block_decoded fallback, c16 1061.44 tok/s (>= 854.7 BF16 floor,
    // 1.57x the same run's vLLM-ROCm pass), GSM8K-200 drop 0.005 (<= 0.02 gate). Labbook
    // turbine-lab-bench runs 2e31910f (turbine) / a86153ab (vllm-rocm). Closed 2026-09-30: golden
    // c1 / c16 16/16 against tests/golden/llama-3.2-3b-instruct-fp8-block (1062.4 tok/s) and the
    // 10-min overload soak PASS 8/8.
    row(
        Some("amd"),
        Some("gfx1201"),
        Some("LlamaForCausalLM"),
        Some(WeightFormatColumn::Fp8Block),
        KV_BF16,
        NO_SPEC,
        SupportStatus::Supported,
    ),
    // mxfp4 and mxfp4_a4 on gfx1201 Llama stay `experimental` for 6a (user decisions
    // 2026-09-30, A: Phase 7 items); awq_int4 and gptq_int4 below are `supported` after their gate.
    gfx1201_quant_row(WeightFormatColumn::Mxfp4),
    gfx1201_quant_row(WeightFormatColumn::Mxfp4A4),
    // awq_int4 (plan Task 18) proof passed 2026-09-29 (t18-run, novanas, commit d14402f):
    // c16 1262.6 vs BF16 842.8 tok/s (1.50x, >= 0.9x), c1 ITL p50 5.87 vs 12.37 ms (0.47x,
    // <= 0.6x), golden c1 16/16 strict and c16 16/16 batched, GSM8K-200 0.775 vs BF16 0.805
    // (drop 0.030, <= 0.04 gate; vLLM-ROCm AWQ 485.8 tok/s, 0.755). The 10-minute overload soak
    // (rotation 9, novanas, target/soak/novanas-20260929T194647Z) passed every check
    // (server_never_restarted, only_503_overload_codes, streams_complete, itl_p99_within_2x,
    // reached_orange, green_within_60s, kv_idle, reserve_held). Flip to `supported`.
    row(
        Some("amd"),
        Some("gfx1201"),
        Some("LlamaForCausalLM"),
        Some(WeightFormatColumn::AwqInt4),
        KV_BF16,
        NO_SPEC,
        SupportStatus::Supported,
    ),
    // gptq_int4 (plan Task 18) proof on kaitchup's AutoRound GPTQ checkpoint (spec S-11, user
    // decision 2026-09-30 B), passed 2026-09-30 on novanas (commit 2a53dbb): full GSM8K at c16
    // 0.7566 vs BF16 0.7801 (drop 0.0235, <= 0.04 gate), golden c1 16/16 strict and c16 16/16
    // batched against tests/golden/llama-3.2-3b-instruct-autoround-gptq, c16 1267.3 tok/s
    // (labbook run 1659f427), and the 10-minute overload soak
    // (target/soak/novanas-20260930T140754Z, calibration 7.97 req/s) passed every check,
    // reached_orange included. Flip to `supported`.
    row(
        Some("amd"),
        Some("gfx1201"),
        Some("LlamaForCausalLM"),
        Some(WeightFormatColumn::GptqInt4),
        KV_BF16,
        NO_SPEC,
        SupportStatus::Supported,
    ),
    // Quantized weights on the CPU reference provider (Phase 6a S-3): tests and tiny
    // checkpoints only, with BF16 or FP8 KV.
    cpu_quant_row(WeightFormatColumn::Fp8, KvFormatColumn::Bf16),
    cpu_quant_row(WeightFormatColumn::Fp8, KvFormatColumn::Fp8E4m3),
    cpu_quant_row(WeightFormatColumn::Fp8Block, KvFormatColumn::Bf16),
    cpu_quant_row(WeightFormatColumn::Fp8Block, KvFormatColumn::Fp8E4m3),
    cpu_quant_row(WeightFormatColumn::Mxfp4, KvFormatColumn::Bf16),
    cpu_quant_row(WeightFormatColumn::Mxfp4, KvFormatColumn::Fp8E4m3),
    cpu_quant_row(WeightFormatColumn::Mxfp4A4, KvFormatColumn::Bf16),
    cpu_quant_row(WeightFormatColumn::Mxfp4A4, KvFormatColumn::Fp8E4m3),
    cpu_quant_row(WeightFormatColumn::AwqInt4, KvFormatColumn::Bf16),
    cpu_quant_row(WeightFormatColumn::AwqInt4, KvFormatColumn::Fp8E4m3),
    cpu_quant_row(WeightFormatColumn::GptqInt4, KvFormatColumn::Bf16),
    cpu_quant_row(WeightFormatColumn::GptqInt4, KvFormatColumn::Fp8E4m3),
    // FP8 KV on the CPU reference provider (Phase 6a S-13): tests and tiny checkpoints.
    row(
        Some("cpu"),
        None,
        None,
        BF16,
        Some(KvFormatColumn::Fp8E4m3),
        NO_SPEC,
        SupportStatus::Experimental,
    ),
    // TurboQuant L0 pages on the CPU reference provider (P6b S-5, `cpu::tq_attention`): tests
    // and tiny checkpoints; a GPU provider needs the v2.10 mixed-format attention (Task 12).
    row(
        Some("cpu"),
        None,
        None,
        BF16,
        Some(KvFormatColumn::Tq4),
        NO_SPEC,
        SupportStatus::Experimental,
    ),
    row(
        Some("cpu"),
        None,
        None,
        BF16,
        Some(KvFormatColumn::Tq2),
        NO_SPEC,
        SupportStatus::Experimental,
    ),
    // Reserved for the tracks; each track replaces its refusal with validated rows.
    family_row("amd", "Qwen3ForCausalLM"),
    family_row("amd", "Qwen3MoeForCausalLM"),
    family_row("amd", "MistralForCausalLM"),
    family_row("amd", "MixtralForCausalLM"),
    // Phase 6a weight formats: each row turns `supported` for its proven combination only.
    format_row(WeightFormatColumn::Fp8, QUANT_REASON),
    format_row(WeightFormatColumn::Fp8Block, QUANT_REASON),
    format_row(WeightFormatColumn::Mxfp4, QUANT_REASON),
    format_row(WeightFormatColumn::Mxfp4A4, QUANT_REASON),
    format_row(WeightFormatColumn::AwqInt4, QUANT_REASON),
    format_row(WeightFormatColumn::GptqInt4, QUANT_REASON),
    // Reserved for the deferred phase-2b-nvidia.
    format_row(WeightFormatColumn::ModeloptNvfp4, RESERVED_NVIDIA_REASON),
    format_row(WeightFormatColumn::ModeloptFp8, RESERVED_NVIDIA_REASON),
    format_row(WeightFormatColumn::ModeloptMixed, RESERVED_NVIDIA_REASON),
    format_row(WeightFormatColumn::CtNvfp4, RESERVED_NVIDIA_REASON),
    row(
        None,
        None,
        None,
        None,
        Some(KvFormatColumn::Fp8E4m3),
        NO_SPEC,
        unsupported("fp8_e4m3 KV cache is not validated yet (track phase-6a-quantization)"),
    ),
    // TurboQuant L0 pages (P6b S-5): refused until the mixed-format attention passes its gates.
    row(
        None,
        None,
        None,
        None,
        Some(KvFormatColumn::Tq4),
        NO_SPEC,
        unsupported(TQ_KV_REASON),
    ),
    row(
        None,
        None,
        None,
        None,
        Some(KvFormatColumn::Tq2),
        NO_SPEC,
        unsupported(TQ_KV_REASON),
    ),
    row(
        None,
        None,
        None,
        None,
        None,
        Some(SpeculativeColumn::Draft),
        unsupported(
            "draft-model speculative decoding is not validated yet (track phase-8-speculative-decoding)",
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
        if deferred_vendor(&self.vendor).is_some() {
            "execution.backend"
        } else if self.speculative != SpeculativeColumn::None {
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
    /// [`HOST_VENDOR`], `architecture` is the model's Hugging Face architecture name, or
    /// [`WILDCARD`] when `config.json` cannot be read yet, and the format columns are the
    /// checkpoint's detected weight format and the configured L0 KV dtype.
    pub fn before_discovery(
        vendor: &str,
        architecture: Option<&str>,
        weight: WeightFormatColumn,
        kv: KvFormatColumn,
    ) -> SupportKey {
        let arch = if vendor == HOST_VENDOR {
            HOST_VENDOR
        } else {
            WILDCARD
        };
        SupportKey::for_model(
            vendor,
            arch,
            architecture.unwrap_or(WILDCARD),
            weight,
            kv,
            SpeculativeColumn::None,
        )
    }

    /// The key of a served model: every column spelled out.
    pub fn for_model(
        vendor: &str,
        arch: &str,
        architecture: &str,
        weight: WeightFormatColumn,
        kv: KvFormatColumn,
        speculative: SpeculativeColumn,
    ) -> SupportKey {
        SupportKey {
            vendor: vendor.to_string(),
            arch: arch.to_string(),
            architecture: architecture.to_string(),
            weight_format: weight,
            kv_format: kv,
            speculative,
        }
    }

    /// Key with the BF16 weight and KV columns and no speculation (tests and the BF16 baseline).
    pub fn bf16(vendor: &str, arch: &str, architecture: &str) -> SupportKey {
        SupportKey::for_model(
            vendor,
            arch,
            architecture,
            WeightFormatColumn::Bf16,
            KvFormatColumn::Bf16,
            SpeculativeColumn::None,
        )
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

/// Most specific row of `table` matching `key` exactly; no row → unsupported. A key of a
/// [`DEFERRED_VENDORS`] vendor is unsupported with its deferral reason whatever the rows say.
pub fn resolve_in(table: &[SupportRow], key: &SupportKey) -> SupportStatus {
    if let Some(reason) = deferred_vendor(&key.vendor) {
        return unsupported(reason);
    }
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
    if let Some(reason) = deferred_vendor(&key.vendor) {
        return unsupported(reason);
    }
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

/// The status of a lower-tier KV format (`kv.cpu.format`, `kv.nvme.format`, the ladder's
/// `kv.ladder.max_format`; P6b S-2). The support-matrix row keeps naming the L0 format, so the
/// lower-tier formats are resolved against this table instead; a format it does not list
/// (`l0`, `fp8_e4m3`) is `supported` here, its availability decided by the kernel library.
#[derive(Clone, Debug)]
pub struct TierFormatRefusal {
    pub format: &'static str,
    pub status: SupportStatus,
}

const TQ_TIER_REASON: &str =
    "TurboQuant lower-tier KV is not validated yet (track phase-6b-kv-compression)";

/// Lower-tier formats that are not `supported`: `fp8_e4m3` is `experimental` from the ABI v2.11
/// transcode (P6b Task 5) until its lab proof (Task 6); TurboQuant is refused until its codec
/// lands (P6b Tasks 7–9), then `experimental` until the S-8 gate passes.
pub static TIER_FORMAT_REFUSALS: &[TierFormatRefusal] = &[
    TierFormatRefusal {
        format: "fp8_e4m3",
        status: SupportStatus::Experimental,
    },
    TierFormatRefusal {
        format: "tq4",
        status: unsupported(TQ_TIER_REASON),
    },
    TierFormatRefusal {
        format: "tq2",
        status: unsupported(TQ_TIER_REASON),
    },
];

/// Resolves lower-tier format `format`, configured under `key`, against
/// [`TIER_FORMAT_REFUSALS`]: `unsupported` is a configuration error naming `key` (exit 2),
/// otherwise the status (`experimental` starts with a WARN).
pub fn check_tier_format(key: &str, format: &str) -> Result<SupportStatus, ConfigError> {
    let status = TIER_FORMAT_REFUSALS
        .iter()
        .find(|r| r.format == format)
        .map_or(SupportStatus::Supported, |r| r.status.clone());
    if let SupportStatus::Unsupported { reason } = &status {
        return Err(ConfigError::Invalid {
            key: key.to_string(),
            reason: format!("tier format {format} is unsupported: {reason}"),
        });
    }
    Ok(status)
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
                key: pat(Some("cpu"), Some("host-a"), None, Some(S::None)),
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
            "cpu",
            "host-a",
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
            "cpu",
            "host-b",
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
        // The AMD baseline: fully specific supported rows.
        for architecture in ["LlamaForCausalLM", "OlmoeForCausalLM"] {
            let k = key("amd", "gfx1201", architecture, W::Bf16, K::Bf16, S::None);
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
        // Every nvidia key, listed row or not, is refused naming the deferred phase-2b-nvidia.
        for architecture in [
            "LlamaForCausalLM",
            "OlmoeForCausalLM",
            "Qwen3ForCausalLM",
            "MixtralForCausalLM",
        ] {
            for w in W::ALL {
                for kv in K::ALL {
                    for spec in S::ALL {
                        for arch in ["sm_121", "sm_90", WILDCARD] {
                            let k = key("nvidia", arch, architecture, w, kv, spec);
                            let status = if k.is_partial() {
                                resolve_partial_in(SUPPORT_MATRIX, &k)
                            } else {
                                resolve(&k)
                            };
                            assert_eq!(status.as_str(), "unsupported", "{k}");
                            assert!(
                                status.reason().unwrap().contains("phase-2b-nvidia"),
                                "{k}: {status:?}"
                            );
                        }
                    }
                }
            }
        }
        let err = check(SupportKey::bf16("nvidia", "sm_121", "LlamaForCausalLM")).unwrap_err();
        assert_eq!(err.key(), Some("execution.backend"));
        // No reason names a Phase 8 run-ahead track name; every reason names a live track.
        for r in SUPPORT_MATRIX {
            if let Some(reason) = r.status.reason() {
                for stale in ["a", "b", "c"].map(|t| format!("phase-8{t}")) {
                    assert!(!reason.contains(&stale), "{:?}", r.view());
                }
            }
        }
        // Nothing but the AMD BF16 baseline, the proven gfx1201 Llama FP8 KV row (Task 24) and the
        // proven gfx1201 Llama weight rows with BF16 KV (fp8, Task 14; fp8_block, Task 15;
        // awq_int4 and gptq_int4, Task 18) are supported before the tracks add rows.
        for r in SUPPORT_MATRIX
            .iter()
            .filter(|r| r.status == SupportStatus::Supported)
        {
            assert_eq!(r.key.vendor, Some("amd"), "{:?}", r.view());
            if r.key.weight_format != Some(W::Bf16) {
                assert!(
                    matches!(
                        r.key.weight_format,
                        Some(W::Fp8 | W::Fp8Block | W::AwqInt4 | W::GptqInt4)
                    ),
                    "{:?}",
                    r.view()
                );
                assert_eq!(r.key.arch, Some("gfx1201"), "{:?}", r.view());
                assert_eq!(
                    r.key.architecture,
                    Some("LlamaForCausalLM"),
                    "{:?}",
                    r.view()
                );
                assert_eq!(r.key.kv_format, Some(K::Bf16), "{:?}", r.view());
            }
            if r.key.kv_format == Some(K::Fp8E4m3) {
                assert_eq!(r.key.arch, Some("gfx1201"), "{:?}", r.view());
                assert_eq!(
                    r.key.architecture,
                    Some("LlamaForCausalLM"),
                    "{:?}",
                    r.view()
                );
            } else {
                assert_eq!(r.key.kv_format, Some(K::Bf16), "{:?}", r.view());
            }
            assert_eq!(r.key.speculative, Some(S::None), "{:?}", r.view());
        }
        // The Phase 6a formats exist and are refused naming the track; the NVIDIA-reserved
        // formats name phase-2b-nvidia.
        let spelled: Vec<&str> = W::PHASE_6A.iter().map(|w| w.as_str()).collect();
        assert_eq!(
            spelled,
            [
                "fp8",
                "fp8_block",
                "mxfp4",
                "mxfp4_a4",
                "awq_int4",
                "gptq_int4"
            ]
        );
        for w in W::PHASE_6A {
            // Supported on gfx1201 Llama once its gate passed (fp8, Task 14; fp8_block, Task 15;
            // awq_int4, Task 18 plus the rotation 9 soak; gptq_int4, Task 18 on the AutoRound
            // checkpoint), experimental otherwise (mxfp4, mxfp4_a4); refused elsewhere.
            let k = key("amd", "gfx1201", "LlamaForCausalLM", w, K::Bf16, S::None);
            let expected = if matches!(w, W::Fp8 | W::Fp8Block | W::AwqInt4 | W::GptqInt4) {
                "supported"
            } else {
                "experimental"
            };
            assert_eq!(resolve(&k).as_str(), expected, "{k}");
            let k = key("amd", "gfx1201", "OlmoeForCausalLM", w, K::Bf16, S::None);
            let status = resolve(&k);
            assert_eq!(status.as_str(), "unsupported", "{k}");
            assert!(
                status.reason().unwrap().contains("phase-6a-quantization"),
                "{k}: {status:?}"
            );
            let err = check(k).unwrap_err();
            assert_eq!(err.key(), Some("model.path"));
        }
        for w in W::RESERVED_NVIDIA {
            let k = key("amd", "gfx1201", "LlamaForCausalLM", w, K::Bf16, S::None);
            let status = resolve(&k);
            assert!(
                status.reason().unwrap().contains("phase-2b-nvidia"),
                "{k}: {status:?}"
            );
        }
        // FP8 KV: supported on gfx1201 Llama after the Task 24 proof, experimental on gfx1201
        // OLMoE (golden miss, user decision 2026-09-30 B), refused naming the track anywhere else.
        for (architecture, expected) in [
            ("LlamaForCausalLM", "supported"),
            ("OlmoeForCausalLM", "experimental"),
        ] {
            let k = key("amd", "gfx1201", architecture, W::Bf16, K::Fp8E4m3, S::None);
            assert_eq!(resolve(&k).as_str(), expected, "{k}");
        }
        let k = key(
            "amd",
            "gfx942",
            "LlamaForCausalLM",
            W::Bf16,
            K::Fp8E4m3,
            S::None,
        );
        let status = resolve(&k);
        assert!(
            status.reason().unwrap().contains("phase-6a-quantization"),
            "{status:?}"
        );
        // TurboQuant L0 pages (P6b S-5): `experimental` on the CPU reference provider (BF16
        // weights), refused naming the track on every other vendor, blaming kv.dtype; the
        // lower-tier formats resolve through TIER_FORMAT_REFUSALS.
        for kv in [K::Tq4, K::Tq2] {
            for architecture in ["LlamaForCausalLM", "OlmoeForCausalLM"] {
                let k = key("cpu", "cpu", architecture, W::Bf16, kv, S::None);
                assert_eq!(resolve(&k).as_str(), "experimental", "{k}");
                let k = key("cpu", "cpu", architecture, W::Fp8, kv, S::None);
                assert_eq!(resolve(&k).as_str(), "unsupported", "{k}");
            }
            for (vendor, arch, architecture) in [
                ("amd", "gfx1201", "LlamaForCausalLM"),
                ("amd", "gfx1201", "OlmoeForCausalLM"),
            ] {
                let k = key(vendor, arch, architecture, W::Bf16, kv, S::None);
                let status = resolve(&k);
                assert_eq!(status.as_str(), "unsupported", "{k}");
                assert!(
                    status.reason().unwrap().contains("phase-6b-kv-compression"),
                    "{k}: {status:?}"
                );
                assert_eq!(check(k).unwrap_err().key(), Some("kv.dtype"));
            }
        }
        for format in ["tq4", "tq2"] {
            let err = check_tier_format("kv.nvme.format", format).unwrap_err();
            assert_eq!(err.key(), Some("kv.nvme.format"));
            let msg = err.to_string();
            assert!(msg.contains("phase-6b-kv-compression"), "{msg}");
            assert!(msg.contains(&format!("tier format {format}")), "{msg}");
        }
        // `fp8_e4m3` is experimental from the ABI v2.11 transcode (P6b Task 5) until its lab
        // proof (Task 6); `l0` is always supported.
        for (format, want) in [
            ("l0", SupportStatus::Supported),
            ("fp8_e4m3", SupportStatus::Experimental),
        ] {
            assert_eq!(
                check_tier_format("kv.cpu.format", format).unwrap(),
                want,
                "{format}"
            );
        }
        let k = key(
            "amd",
            "gfx1201",
            "LlamaForCausalLM",
            W::Bf16,
            K::Bf16,
            S::Draft,
        );
        let status = resolve(&k);
        assert!(
            status
                .reason()
                .unwrap()
                .contains("phase-8-speculative-decoding"),
            "{status:?}"
        );
        // Phase 7 families: refused on AMD (any arch), experimental on the CPU reference.
        for architecture in [
            "Qwen3ForCausalLM",
            "Qwen3MoeForCausalLM",
            "MistralForCausalLM",
            "MixtralForCausalLM",
        ] {
            for arch in ["gfx1201", "gfx942"] {
                let k = key("amd", arch, architecture, W::Bf16, K::Bf16, S::None);
                let status = resolve(&k);
                assert_eq!(status.as_str(), "unsupported", "{k}");
                assert!(
                    status.reason().unwrap().contains("phase-7-model-families"),
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
        use KvFormatColumn as K;
        use WeightFormatColumn as W;
        let k = SupportKey::before_discovery("amd", Some("LlamaForCausalLM"), W::Bf16, K::Bf16);
        assert_eq!(k.to_string(), "amd/*/LlamaForCausalLM/bf16/bf16/none");
        assert!(k.is_partial());
        assert_eq!(check(k).unwrap().status, SupportStatus::Supported);

        let k = SupportKey::before_discovery("amd", Some("Qwen3ForCausalLM"), W::Bf16, K::Bf16);
        let err = check(k).unwrap_err();
        assert_eq!(err.key(), Some("execution.backend"));
        let msg = err.to_string();
        assert!(
            msg.contains("support matrix: amd/*/Qwen3ForCausalLM/bf16/bf16/none is unsupported: "),
            "{msg}"
        );

        let k = SupportKey::before_discovery(HOST_VENDOR, None, W::Bf16, K::Bf16);
        assert_eq!(k.to_string(), "cpu/cpu/*/bf16/bf16/none");
        assert_eq!(check(k).unwrap().status, SupportStatus::Experimental);
        let k =
            SupportKey::before_discovery(HOST_VENDOR, Some("Qwen3ForCausalLM"), W::Bf16, K::Bf16);
        assert!(!k.is_partial());
        assert_eq!(resolve(&k), SupportStatus::Experimental);

        let k = SupportKey::bf16("amd", "gfx1201", "OlmoeForCausalLM");
        assert_eq!(resolve(&k), SupportStatus::Supported);
        // A detected quantized format and FP8 KV show in the key.
        let k = SupportKey::before_discovery("amd", Some("LlamaForCausalLM"), W::Fp8, K::Fp8E4m3);
        assert_eq!(k.to_string(), "amd/*/LlamaForCausalLM/fp8/fp8_e4m3/none");
        assert_eq!(check(k).unwrap_err().key(), Some("kv.dtype"));
        assert!(VENDORS.contains(&HOST_VENDOR));
    }
}
