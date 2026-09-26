//! The decoder hooks (Phase 2m S-3): one file per attention or FFN variant, each a stateless
//! `&'static` value a family names in its [`super::DecoderSpec`].
//!
//! - [`PLAIN_ATTENTION`]: nothing between the Q/K/V projections and RoPE (Llama).
//! - [`QK_NORM_FULL`]: RMSNorm over the full Q and K projections (OLMoE).
//! - [`QK_NORM_PER_HEAD`]: RMSNorm over each Q and K head (Qwen3, Qwen3-MoE); keeps Q/K/V
//!   unfused.
//! - [`SWIGLU`]: the dense `down(silu(gate(x)) · up(x))` MLP (Llama).
//! - [`MOE`]: the sparse mixture of SwiGLU experts behind a softmax top-k router (OLMoE).

mod moe;
mod plain_attention;
mod qk_norm;
mod swiglu;

pub use moe::{MOE, Moe};
pub use plain_attention::{PLAIN_ATTENTION, PlainAttention};
pub use qk_norm::{QK_NORM_FULL, QK_NORM_PER_HEAD, QkNormFull, QkNormPerHead};
pub use swiglu::{SWIGLU, SwiGlu};
