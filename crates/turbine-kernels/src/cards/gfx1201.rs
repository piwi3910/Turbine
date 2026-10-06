//! `gfx1201`: AMD RDNA4 (Radeon AI PRO R9700). Wave32 with WMMA matrix instructions and native
//! BF16; 64 KiB LDS per workgroup. The thresholds are the ones the HIP library tuned on the R9700:
//! the small-m MoE kernel wins up to 512 routed rows, the grouped WMMA kernel up to 11,264 rows
//! and the prefill WMMA kernels (expert weights streamed into the WMMA registers, the same
//! per-element chain) above them, with OLMoE's real routing (`perf moe_prefill_timings`; see
//! `MOE_WMMA_MAX_ROWS`), and Composable Kernel paged attention
//! serves pages of a multiple of 128 tokens. Paged decode prefers the own b1 kernel (decision
//! "5p decode-BW floor", user decision B): grouped query heads only, grid (sequences, KV heads,
//! splits) with the append and the split combine fused into the launch; from 4 sequences that
//! entry runs CK's split-KV kernel itself (faster there, `hip_ops decode_attention_timings`),
//! which merges the query heads of a KV head into one tile (1.2-5x faster
//! than `fmha_fwd_pagedkv` for Llama's 24/8 heads) and refuses equal head counts (OLMoE), which
//! keep `fmha_fwd_pagedkv` and its batch-invariant numerics.

use super::{CardCapabilities, CardProfile, CardThresholds, OpPreference, RowTierSpec};
use crate::OpKind;

/// Routed rows up to which the small-m MoE kernel is preferred. For OLMoE-class shapes (hidden
/// and inter multiples of 64) both tiers run the same WMMA chain, so this bound only moves work
/// between tile shapes and never changes a row's bits (batch invariance, decision 2026-09-27);
/// with the WMMA decode kernels the small tier stays faster than the grouped tier up to at least
/// 96 tokens (`hip_batch_invariance moe_decode_tier_timings`), so 512 rows (64 decodes) stays.
const MOE_SMALL_MAX_ROWS: u32 = 512;

/// Routed rows up to which, above [`MOE_SMALL_MAX_ROWS`], the grouped `turbine_hip_moe_wmma`
/// kernel is preferred over `turbine_hip_moe_wmma_prefill` (user decision 2026-10-06 O1: the
/// prefill kernel's k step dropped 128 → 64 for the K-split slice order, 200d074d, so it now
/// wins only on large calls). Both run the same WMMA chain, so this bound never changes a row's
/// bits. Measured with OLMoE's real routing (`perf moe_prefill_timings`, GPU 0, layers 0/5/10/15
/// summed, prefill ÷ wmma): 1.17 at 512 rows, 1.13 at 2,048, 1.11 at 6,144, 1.05 at 8,192, 1.02
/// at 10,240, 1.006 at 11,264, 0.97 at 12,288, 0.88 at 16,384 — the crossover lies between
/// 11,264 and 12,288 rows (1,408 and 1,536 prefill tokens).
const MOE_WMMA_MAX_ROWS: u32 = 11_264;

const MOE_PREFILL_ORDER: &[&str] = &[
    "turbine_hip_moe_wmma_prefill",
    "turbine_hip_moe_wmma",
    "hipblaslt_grouped",
    "hipblaslt_per_expert",
];

const GEMM_ORDER: &[&str] = &["hipblaslt"];
/// Rows up to which `gemm` prefers the single-row GEMV `vllm_wvsplitk` (it computes one row;
/// the library entry runs hipBLASLt for prefill steps, so only decode rows change class).
const GEMV_MAX_ROWS: u32 = 1;

const RMSNORM_ORDER: &[&str] = &["ck_tile_rmsnorm2d", "turbine_hip"];
const PAGED_ORDER: &[&str] = &["ck_tile_fmha_pagedkv", "turbine_hip"];
const PAGED_DECODE_ORDER: &[&str] = &[
    "turbine_hip_b1_decode",
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
/// Decode rows up to which FP8 W8A8 weights (FP8_TENSOR / FP8_CHANNEL) run Turbine's own WMMA
/// kernel (`turbine_hip_fp8_wmma`, `kMaxRows` in `qgemm_fp8_wmma.hip`): at batch 1 hipBLASLt's
/// best FP8 solutions read the weights at 350-460 GB/s (decision "5p Task 14: FP8 c1 decode —
/// provider evaluation (kernel reuse rule)"). The entry runs hipBLASLt itself for prefill steps
/// and larger calls, so the tier only moves decode steps of up to this many rows.
const FP8_WMMA_MAX_ROWS: u32 = 16;
const QGEMM_DECODE_ORDER: &[&str] = &[
    "turbine_hip_fp8_wmma",
    "hipblaslt_fp8",
    "turbine_hip_int4_wmma",
    "turbine_hip_int4_dequant",
    "turbine_hip_mxfp4",
    "turbine_hip_fp8_block",
];
const QGEMM_SMALL_ORDER: &[&str] = &[
    "hipblaslt_fp8",
    "turbine_hip_int4_wmma",
    "turbine_hip_int4_dequant",
    "turbine_hip_mxfp4",
    "turbine_hip_fp8_block",
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
        // Single-row calls of dense BF16 decoders (`GemmConfig::gemv`) on vLLM's wvSplitK GEMV
        // (user decision "5p fused batch-1 GEMV", B staged, stage A: the fastest correct m = 1
        // GEMV on every Llama-3.2-3B decode shape); every other call on hipBLASLt.
        OpPreference {
            op: OpKind::Gemm,
            order: GEMM_ORDER,
            row_tiers: &[
                RowTierSpec {
                    max_rows: Some(GEMV_MAX_ROWS),
                    order: &["vllm_wvsplitk", "hipblaslt"],
                },
                RowTierSpec {
                    max_rows: None,
                    order: GEMM_ORDER,
                },
            ],
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
            // FP8 W8A8: the own small-m kernel up to FP8_WMMA_MAX_ROWS rows (5p Task 14).
            row_tiers: &[
                RowTierSpec {
                    max_rows: Some(FP8_WMMA_MAX_ROWS),
                    order: QGEMM_DECODE_ORDER,
                },
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
                        "turbine_hip_fp8_block",
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
            order: MOE_PREFILL_ORDER,
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
                    max_rows: Some(MOE_WMMA_MAX_ROWS),
                    order: &[
                        "turbine_hip_moe_wmma",
                        "turbine_hip_moe_wmma_prefill",
                        "hipblaslt_grouped",
                        "hipblaslt_per_expert",
                    ],
                },
                RowTierSpec {
                    max_rows: None,
                    order: MOE_PREFILL_ORDER,
                },
            ],
        },
    ],
};
