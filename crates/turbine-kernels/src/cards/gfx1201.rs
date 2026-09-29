//! `gfx1201`: AMD RDNA4 (Radeon AI PRO R9700). Wave32 with WMMA matrix instructions and native
//! BF16; 64 KiB LDS per workgroup. The thresholds are the ones the HIP library tuned on the R9700:
//! the small-m MoE kernel wins up to 512 routed rows, the prefill WMMA kernels (expert weights
//! streamed into the WMMA registers, the same per-element chain) above them, 8-16 % faster than
//! the grouped WMMA kernels with OLMoE's real routing (`perf moe_prefill_timings`), and
//! Composable Kernel paged attention
//! serves pages of a multiple of 128 tokens. Paged decode prefers CK's split-KV kernel, which
//! merges the query heads of a KV head into one tile: 1.2-5x faster than `fmha_fwd_pagedkv` for
//! Llama's 24/8 heads (`hip_ops decode_attention_timings`); it refuses equal head counts
//! (OLMoE), which keep `fmha_fwd_pagedkv` and its batch-invariant numerics.

use super::{CardCapabilities, CardProfile, CardThresholds, OpPreference, RowTierSpec};
use crate::OpKind;

/// Routed rows up to which the small-m MoE kernel is preferred. For OLMoE-class shapes (hidden
/// and inter multiples of 64) both tiers run the same WMMA chain, so this bound only moves work
/// between tile shapes and never changes a row's bits (batch invariance, decision 2026-09-27);
/// with the WMMA decode kernels the small tier stays faster than the grouped tier up to at least
/// 96 tokens (`hip_batch_invariance moe_decode_tier_timings`), so 512 rows (64 decodes) stays.
const MOE_SMALL_MAX_ROWS: u32 = 512;

const RMSNORM_ORDER: &[&str] = &["ck_tile_rmsnorm2d", "turbine_hip"];
const PAGED_ORDER: &[&str] = &["ck_tile_fmha_pagedkv", "turbine_hip"];
const PAGED_DECODE_ORDER: &[&str] = &[
    "ck_tile_fmha_splitkv",
    "ck_tile_fmha_pagedkv",
    "turbine_hip",
];
const FMHA_ORDER: &[&str] = &["ck_tile_fmha_fwd"];
/// The tensor-parallel sharded RMSNorm (ABI v2.6): the Turbine kernels only (no CK instance takes
/// an external sum of squares; decision "P5 T6: sharded RMSNorm — provider evaluation").
const SHARDED_NORM_ORDER: &[&str] = &["turbine_hip"];
/// Rows up to which INT4 weights run the fused WMMA kernel (codes streamed into the matrix
/// registers); above, dequantizing to BF16 once and running hipBLASLt is faster
/// (`tools/qgemm_int4_eval` on the R9700 GPU 0, Llama-3.2-3B layer sums: 128 rows 652 µs fused
/// vs 711 dequant, 512 rows 2,387 vs 1,402; decision "P6: INT4 group GEMM — provider
/// evaluation (kernel reuse rule)").
const INT4_WMMA_MAX_ROWS: u32 = 128;
const QGEMM_SMALL_ORDER: &[&str] = &[
    "hipblaslt_fp8",
    "turbine_hip_int4_wmma",
    "turbine_hip_int4_dequant",
    "turbine_hip_mxfp4",
];

pub static GFX1201: CardProfile = CardProfile {
    name: "gfx1201",
    vendor: "amd",
    archs: &["gfx1201"],
    capabilities: CardCapabilities {
        matrix_instructions: &["wmma"],
        bf16: true,
        wave_size: 32,
        lds_bytes: 65536,
    },
    thresholds: CardThresholds {
        moe_small_max_rows: MOE_SMALL_MAX_ROWS,
        paged_page_multiple: 128,
    },
    preferences: &[
        OpPreference {
            op: OpKind::Gemm,
            order: &["hipblaslt"],
            row_tiers: &[],
        },
        OpPreference {
            op: OpKind::AttentionPrefill,
            order: FMHA_ORDER,
            row_tiers: &[],
        },
        OpPreference {
            op: OpKind::AttentionDecode,
            order: FMHA_ORDER,
            row_tiers: &[],
        },
        OpPreference {
            op: OpKind::Rmsnorm,
            order: RMSNORM_ORDER,
            row_tiers: &[],
        },
        OpPreference {
            op: OpKind::AddRmsnorm,
            order: RMSNORM_ORDER,
            row_tiers: &[],
        },
        OpPreference {
            op: OpKind::RowSumsq,
            order: SHARDED_NORM_ORDER,
            row_tiers: &[],
        },
        OpPreference {
            op: OpKind::RmsnormSharded,
            order: SHARDED_NORM_ORDER,
            row_tiers: &[],
        },
        // ABI v2.9 (Phase 6a): FP8 W8A8 on hipBLASLt and Turbine's own e4m3 activation
        // quantization (decision "P6: FP8 GEMM — provider evaluation (kernel reuse rule)");
        // MXFP4 weights and activation emulation on Turbine's own kernels at every row count
        // (decision "P6: MXFP4 GEMM — provider evaluation (kernel reuse rule)").
        OpPreference {
            op: OpKind::QGemm,
            order: QGEMM_SMALL_ORDER,
            // INT4 (decision "P6: INT4 group GEMM — provider evaluation (kernel reuse rule)"):
            // the fused kernel up to INT4_WMMA_MAX_ROWS rows, dequantize + hipBLASLt above.
            row_tiers: &[
                RowTierSpec {
                    max_rows: Some(INT4_WMMA_MAX_ROWS),
                    order: QGEMM_SMALL_ORDER,
                },
                RowTierSpec {
                    max_rows: None,
                    order: &[
                        "hipblaslt_fp8",
                        "turbine_hip_int4_dequant",
                        "turbine_hip_int4_wmma",
                        "turbine_hip_mxfp4",
                    ],
                },
            ],
        },
        OpPreference {
            op: OpKind::QuantizeAct,
            order: &["turbine_hip", "turbine_hip_mxfp4"],
            row_tiers: &[],
        },
        OpPreference {
            op: OpKind::AttentionPrefillPaged,
            order: PAGED_ORDER,
            row_tiers: &[],
        },
        OpPreference {
            op: OpKind::AttentionDecodePaged,
            order: PAGED_DECODE_ORDER,
            row_tiers: &[],
        },
        OpPreference {
            op: OpKind::MoeExperts,
            order: &[
                "turbine_hip_moe_wmma_prefill",
                "turbine_hip_moe_wmma",
                "hipblaslt_grouped",
                "hipblaslt_per_expert",
            ],
            row_tiers: &[
                RowTierSpec {
                    max_rows: Some(MOE_SMALL_MAX_ROWS),
                    order: &[
                        "turbine_hip_moe_small_m",
                        "turbine_hip_moe_wmma_prefill",
                        "turbine_hip_moe_wmma",
                        "hipblaslt_grouped",
                        "hipblaslt_per_expert",
                    ],
                },
                RowTierSpec {
                    max_rows: None,
                    order: &[
                        "turbine_hip_moe_wmma_prefill",
                        "turbine_hip_moe_wmma",
                        "hipblaslt_grouped",
                        "hipblaslt_per_expert",
                    ],
                },
            ],
        },
    ],
};
