//! `gfx1201`: AMD RDNA4 (Radeon AI PRO R9700). Wave32 with WMMA matrix instructions and native
//! BF16; 64 KiB LDS per workgroup. The thresholds are the ones the HIP library tuned on the R9700:
//! the small-m MoE kernel wins up to 512 routed rows, and Composable Kernel paged attention
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
                "turbine_hip_moe_wmma",
                "hipblaslt_grouped",
                "hipblaslt_per_expert",
            ],
            row_tiers: &[
                RowTierSpec {
                    max_rows: Some(MOE_SMALL_MAX_ROWS),
                    order: &[
                        "turbine_hip_moe_small_m",
                        "turbine_hip_moe_wmma",
                        "hipblaslt_grouped",
                        "hipblaslt_per_expert",
                    ],
                },
                RowTierSpec {
                    max_rows: None,
                    order: &[
                        "turbine_hip_moe_wmma",
                        "hipblaslt_grouped",
                        "hipblaslt_per_expert",
                    ],
                },
            ],
        },
    ],
};
