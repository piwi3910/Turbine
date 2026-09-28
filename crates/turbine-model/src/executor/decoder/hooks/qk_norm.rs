//! Q/K RMSNorm over the full projections (OLMoE): before RoPE, each new row's whole Q
//! projection (`heads·head_dim` elements) is RMS-normalised with `self_attn.q_norm.weight` and
//! its K projection (`kv_heads·head_dim`) with `self_attn.k_norm.weight`, in place (each row is
//! read whole before it is written, so the row-strided fused layout works too).
//!
//! Tensor parallelism (P5 S-6): a rank holds only its heads' slice of each row, so the norm
//! runs sharded — `row_sumsq` of the rank's Q and K slices into one F32 buffer, one all-reduce
//! of those partial sums across the group, then `rmsnorm_sharded` of each slice with the full
//! row's mean square and the rank's slice of the weight.
//!
//! Q/K RMSNorm per head (Qwen3): [`QkNormPerHead`], below.
use turbine_core::types::DType;
use turbine_kernels::{
    OpConfig, RmsnormShardedConfig, RmsnormShardedContext, RowSumsqConfig, RowSumsqContext,
};
use turbine_tensor::{Tensor, TensorView};

use crate::ModelError;
use crate::executor::decoder::{AttentionHook, DecoderDims, HookWeights, LayerRun, invalid};
use crate::loader::LoadedWeights;

/// Parameter indices of [`HookWeights`].
const Q_NORM: usize = 0;
const K_NORM: usize = 1;

/// RMSNorm over the full Q and K projections.
pub struct QkNormFull;

/// The [`QkNormFull`] hook.
pub const QK_NORM_FULL: &dyn AttentionHook = &QkNormFull;

impl AttentionHook for QkNormFull {
    fn name(&self) -> &'static str {
        "qk_norm_full"
    }

    /// One device: RMSNorm over `q_dim` and `kv_dim`. A tensor-parallel rank: `row_sumsq` and
    /// `rmsnorm_sharded` over its slices.
    fn requirements(&self, d: &DecoderDims) -> Vec<OpConfig> {
        match d.tp {
            None => vec![
                OpConfig::Rmsnorm(d.norm(d.q_dim)),
                OpConfig::Rmsnorm(d.norm(d.kv_dim)),
            ],
            Some(_) => {
                let (q_sum, q_norm) = sharded(d, false);
                let (k_sum, k_norm) = sharded(d, true);
                vec![
                    OpConfig::RowSumsq(q_sum),
                    OpConfig::RowSumsq(k_sum),
                    OpConfig::RmsnormSharded(q_norm),
                    OpConfig::RmsnormSharded(k_norm),
                ]
            }
        }
    }

    fn fuses_qkv(&self) -> bool {
        true
    }

    fn load_layer(
        &self,
        _d: &DecoderDims,
        prefix: &str,
        weights: &mut LoadedWeights,
    ) -> Result<HookWeights, ModelError> {
        Ok(HookWeights::tensors(vec![
            weights.take(&format!("{prefix}.self_attn.q_norm.weight"))?,
            weights.take(&format!("{prefix}.self_attn.k_norm.weight"))?,
        ]))
    }

    fn after_projections(
        &self,
        run: &LayerRun<'_>,
        w: &HookWeights,
        q: &TensorView<'_>,
        k: &TensorView<'_>,
    ) -> Result<(), ModelError> {
        if let Some(sumsq) = run.sumsq() {
            return sharded_norms(run, w, q, k, sumsq);
        }
        run.rmsnorm(q.clone(), &w.0[Q_NORM], q.clone())?;
        run.trace("q_norm", q)?;
        run.rmsnorm(k.clone(), &w.0[K_NORM], k.clone())?;
        run.trace("k_norm", k)
    }
}

/// The `row_sumsq` and `rmsnorm_sharded` configs of the rank's Q slice (`k` false: normalised
/// over the model's Q width) or K slice (`k`: over the K norm's width,
/// [`crate::executor::decoder::TpDims`]).
fn sharded(d: &DecoderDims, k: bool) -> (RowSumsqConfig, RmsnormShardedConfig) {
    let tp = d.tp.expect("sharded norms only on a tensor-parallel rank");
    let (dim, full) = if k {
        (d.kv_dim, tp.k_norm_dim)
    } else {
        (d.q_dim, tp.q_norm_dim)
    };
    (
        RowSumsqConfig {
            dim: dim as u32,
            dtype: d.act,
        },
        RmsnormShardedConfig {
            dim: dim as u32,
            full_dim: full as u32,
            dtype: d.act,
        },
    )
}

/// The tensor-parallel Q/K norm: partial sums of squares of both slices into `sumsq` (Q rows
/// first), one F32 all-reduce, then each slice normalised in place.
fn sharded_norms(
    run: &LayerRun<'_>,
    w: &HookWeights,
    q: &TensorView<'_>,
    k: &TensorView<'_>,
    sumsq: &Tensor,
) -> Result<(), ModelError> {
    let (d, t) = (run.dims, run.tokens);
    let sums = |i: usize| TensorView::contiguous(sumsq.storage.whole(), i * t, &[t], DType::F32);
    for (i, x) in [(0, q), (1, k)] {
        let (cfg, _) = sharded(d, i == 1);
        run.op(OpConfig::RowSumsq(cfg), || {
            run.registry
                .row_sumsq(&cfg)
                .row_sumsq(&mut RowSumsqContext {
                    x: x.clone(),
                    sumsq: sums(i),
                })
        })?;
    }
    run.all_reduce(&TensorView::contiguous(
        sumsq.storage.whole(),
        0,
        &[2 * t],
        DType::F32,
    ))?;
    for (i, x, weight, name) in [
        (0, q, &w.0[Q_NORM], "q_norm"),
        (1, k, &w.0[K_NORM], "k_norm"),
    ] {
        let (_, cfg) = sharded(d, i == 1);
        run.op(OpConfig::RmsnormSharded(cfg), || {
            run.registry
                .rmsnorm_sharded(&cfg)
                .rmsnorm_sharded(&mut RmsnormShardedContext {
                    x: x.clone(),
                    sumsq: sums(i),
                    weight: weight.view(),
                    out: x.clone(),
                    full_dim: cfg.full_dim,
                    eps: d.eps,
                })
        })?;
        run.trace(name, x)?;
    }
    Ok(())
}

/// RMSNorm over each Q and K head (Qwen3, Qwen3-MoE): before RoPE, every `head_dim` elements
/// of each new row's Q projection are RMS-normalised with `self_attn.q_norm.weight` and every
/// `head_dim` elements of its K projection with `self_attn.k_norm.weight` (both `[head_dim]`),
/// in place, as `[tokens · heads, head_dim]` rows. The norm op takes one row stride, and in the
/// fused Q/K/V layout a token's heads are strided by the fused width, so this hook keeps the
/// projections dense ([`AttentionHook::fuses_qkv`] false: one GEMM per projection); rows never
/// mix tokens, so ragged batches stay independent.
pub struct QkNormPerHead;

/// The [`QkNormPerHead`] hook.
pub const QK_NORM_PER_HEAD: &dyn AttentionHook = &QkNormPerHead;

/// A dense `[tokens, heads · head_dim]` projection view as `[tokens · heads, head_dim]` rows.
fn head_rows<'a>(view: &TensorView<'a>, head_dim: usize) -> Result<TensorView<'a>, ModelError> {
    let (tokens, cols) = match view.shape.as_slice() {
        &[tokens, cols] => (tokens, cols),
        other => {
            return Err(invalid(format!(
                "per-head Q/K norm needs a [tokens, heads·head_dim] view, got {other:?}"
            )));
        }
    };
    if view.strides.as_slice() != [cols, 1] || !cols.is_multiple_of(head_dim) {
        return Err(invalid(format!(
            "per-head Q/K norm needs a dense projection of whole heads of {head_dim}, got shape \
             {:?} strides {:?}",
            view.shape.as_slice(),
            view.strides.as_slice()
        )));
    }
    Ok(TensorView::contiguous(
        view.slice,
        0,
        &[tokens * cols / head_dim, head_dim],
        view.dtype,
    ))
}

impl AttentionHook for QkNormPerHead {
    fn name(&self) -> &'static str {
        "qk_norm_per_head"
    }

    fn requirements(&self, d: &DecoderDims) -> Vec<OpConfig> {
        vec![OpConfig::Rmsnorm(d.norm(d.head_dim))]
    }

    fn fuses_qkv(&self) -> bool {
        false
    }

    fn load_layer(
        &self,
        d: &DecoderDims,
        prefix: &str,
        weights: &mut LoadedWeights,
    ) -> Result<HookWeights, ModelError> {
        let mut take = |proj: &str| {
            d.take_weight(
                weights,
                &format!("{prefix}.self_attn.{proj}.weight"),
                &[d.head_dim],
            )
        };
        Ok(HookWeights::tensors(vec![take("q_norm")?, take("k_norm")?]))
    }

    fn after_projections(
        &self,
        run: &LayerRun<'_>,
        w: &HookWeights,
        q: &TensorView<'_>,
        k: &TensorView<'_>,
    ) -> Result<(), ModelError> {
        let head_dim = run.dims.head_dim;
        let (qr, kr) = (head_rows(q, head_dim)?, head_rows(k, head_dim)?);
        run.rmsnorm(qr.clone(), &w.0[Q_NORM], qr)?;
        run.trace("q_norm", q)?;
        run.rmsnorm(kr.clone(), &w.0[K_NORM], kr)?;
        run.trace("k_norm", k)
    }
}
