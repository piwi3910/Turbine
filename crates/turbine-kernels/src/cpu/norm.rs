//! `rmsnorm`, the fused `add_rmsnorm` and the tensor-parallel sharded RMSNorm (`row_sumsq`,
//! `rmsnorm_sharded`) on the cpu-reference provider (Hugging Face LlamaRMSNorm rounding).
use turbine_core::types::DType;
use turbine_tensor::TensorView;

use super::{
    CpuReference, expect_rank, expect_shape, invalid, is_float, load, math, round_to, store,
};
use crate::KernelError;
use crate::ops::{
    AddRmsnormConfig, AddRmsnormContext, AddRmsnormKernel, NormConfig, NormContext, NormKernel,
    RmsnormShardedConfig, RmsnormShardedContext, RowSumsqConfig, RowSumsqContext,
    ShardedNormKernel,
};

impl NormKernel for CpuReference {
    fn supports(&self, cfg: &NormConfig) -> bool {
        cfg.dim > 0 && is_float(cfg.dtype)
    }

    fn implementation(&self, _cfg: &NormConfig) -> String {
        "cpu_rmsnorm".into()
    }

    fn execute(&self, ctx: &mut NormContext<'_>) -> Result<(), KernelError> {
        expect_rank("x", &ctx.x, 2)?;
        let (rows, dim) = (ctx.x.shape[0], ctx.x.shape[1]);
        expect_shape("out", &ctx.out, &[rows, dim])?;
        expect_shape("weight", &ctx.weight, &[dim])?;
        let (x_dt, out_dt) = (ctx.x.dtype, ctx.out.dtype);
        let x = load(&ctx.x)?;
        let w = load(&ctx.weight)?;
        let out = math::rmsnorm(
            &x,
            &w,
            dim,
            ctx.eps,
            |v| round_to(x_dt, v),
            |v| round_to(out_dt, v),
        );
        store(&ctx.out, &out)
    }
}

impl AddRmsnormKernel for CpuReference {
    fn supports(&self, cfg: &AddRmsnormConfig) -> bool {
        cfg.dim > 0 && is_float(cfg.dtype)
    }

    fn implementation(&self, _cfg: &AddRmsnormConfig) -> String {
        "cpu_add_rmsnorm".into()
    }

    /// Exactly `add` (sum in f32, rounded to the residual's dtype) followed by `rmsnorm` of the
    /// rounded sum, so the fused path is bitwise equal to the separate ops.
    fn execute(&self, ctx: &mut AddRmsnormContext<'_>) -> Result<(), KernelError> {
        expect_rank("residual", &ctx.residual, 2)?;
        let (rows, dim) = (ctx.residual.shape[0], ctx.residual.shape[1]);
        expect_shape("x", &ctx.x, &[rows, dim])?;
        expect_shape("out", &ctx.out, &[rows, dim])?;
        expect_shape("weight", &ctx.weight, &[dim])?;
        let (res_dt, out_dt) = (ctx.residual.dtype, ctx.out.dtype);
        let residual = load(&ctx.residual)?;
        let x = load(&ctx.x)?;
        let sum: Vec<f32> = residual
            .iter()
            .zip(&x)
            .map(|(r, v)| round_to(res_dt, r + v))
            .collect();
        store(&ctx.residual, &sum)?;
        let w = load(&ctx.weight)?;
        let out = math::rmsnorm(
            &sum,
            &w,
            dim,
            ctx.eps,
            |v| round_to(res_dt, v),
            |v| round_to(out_dt, v),
        );
        store(&ctx.out, &out)
    }
}

/// Checks that `v` is an F32 view of shape `[rows]` (the per-row sums of squares).
fn expect_sumsq(v: &TensorView<'_>, rows: usize) -> Result<(), KernelError> {
    if v.dtype != DType::F32 {
        return Err(invalid(format!(
            "sumsq must be f32, is {}",
            v.dtype.as_str()
        )));
    }
    expect_shape("sumsq", v, &[rows])
}

/// Exact f32 numerics: `row_sumsq` sums each row sequentially (the order of `rmsnorm`), and
/// `rmsnorm_sharded` is `rmsnorm`'s scaling step with the given sum over `full_dim`, so one
/// shard with its own sum is bitwise `rmsnorm`.
impl ShardedNormKernel for CpuReference {
    fn supports_row_sumsq(&self, cfg: &RowSumsqConfig) -> bool {
        cfg.dim > 0 && is_float(cfg.dtype)
    }

    fn supports_rmsnorm_sharded(&self, cfg: &RmsnormShardedConfig) -> bool {
        cfg.dim > 0 && cfg.full_dim >= cfg.dim && is_float(cfg.dtype)
    }

    fn implementation_row_sumsq(&self, _cfg: &RowSumsqConfig) -> String {
        "cpu_row_sumsq".into()
    }

    fn implementation_rmsnorm_sharded(&self, _cfg: &RmsnormShardedConfig) -> String {
        "cpu_rmsnorm_sharded".into()
    }

    fn row_sumsq(&self, ctx: &mut RowSumsqContext<'_>) -> Result<(), KernelError> {
        expect_rank("x", &ctx.x, 2)?;
        let (rows, dim) = (ctx.x.shape[0], ctx.x.shape[1]);
        expect_sumsq(&ctx.sumsq, rows)?;
        if dim == 0 {
            return Err(invalid("row_sumsq of rows of 0 elements".into()));
        }
        let x = load(&ctx.x)?;
        store(&ctx.sumsq, &math::row_sumsq(&x, dim))
    }

    fn rmsnorm_sharded(&self, ctx: &mut RmsnormShardedContext<'_>) -> Result<(), KernelError> {
        expect_rank("x", &ctx.x, 2)?;
        let (rows, dim) = (ctx.x.shape[0], ctx.x.shape[1]);
        expect_shape("out", &ctx.out, &[rows, dim])?;
        expect_shape("weight", &ctx.weight, &[dim])?;
        expect_sumsq(&ctx.sumsq, rows)?;
        let full_dim = ctx.full_dim as usize;
        if dim == 0 || full_dim < dim {
            return Err(invalid(format!(
                "rmsnorm_sharded of a {dim}-wide slice of rows {full_dim} wide"
            )));
        }
        let (x_dt, out_dt) = (ctx.x.dtype, ctx.out.dtype);
        let x = load(&ctx.x)?;
        let w = load(&ctx.weight)?;
        let sumsq = load(&ctx.sumsq)?;
        let out = math::rmsnorm_from_sumsq(
            &x,
            &w,
            dim,
            &sumsq,
            full_dim,
            ctx.eps,
            |v| round_to(x_dt, v),
            |v| round_to(out_dt, v),
        );
        store(&ctx.out, &out)
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::test_util::*;
    use crate::cpu::*;

    #[test]
    fn rmsnorm_rounds_like_hf_llama() {
        let mem = host();
        let x = tensor(&mem, &[1, 2], DType::BF16, &[3.0, 4.0]);
        let w = tensor(&mem, &[2], DType::BF16, &[1.0, 0.5]);
        let out = Tensor::empty(&mem, &[1, 2], DType::BF16).expect("alloc");
        CpuReference
            .norm()
            .expect("norm family")
            .execute(&mut NormContext {
                x: x.view(),
                weight: w.view(),
                out: out.view(),
                eps: 0.0,
            })
            .expect("rmsnorm");
        // rms = sqrt(12.5); x/rms rounded to bf16, then · w rounded to bf16.
        let r = 1.0 / 12.5f32.sqrt();
        let want = [
            round_to(DType::BF16, round_to(DType::BF16, 3.0 * r)),
            round_to(DType::BF16, round_to(DType::BF16, 4.0 * r) * 0.5),
        ];
        assert_eq!(load(&out.view()).expect("load"), want);
    }

    /// Columns `[start, start + cols)` of every row of the `[rows, full]` tensor `t`, as a
    /// row-strided `[rows, cols]` view (one tensor-parallel rank's slice).
    fn slice_view(t: &Tensor, start: usize, cols: usize) -> TensorView<'_> {
        let (rows, full) = (t.shape[0], t.shape[1]);
        let es = t.dtype.size_bytes();
        TensorView {
            slice: t
                .storage
                .whole()
                .sub(start * es, ((rows - 1) * full + cols) * es),
            shape: (&[rows, cols][..]).into(),
            strides: (&[full, 1][..]).into(),
            dtype: t.dtype,
        }
    }

    /// Phase 5 Task 6 (ABI v2.6): splitting 4 rows of 256 into two 128-wide slices (as two
    /// tensor-parallel ranks hold them), summing the slices' `row_sumsq` (the FP32 all-reduce)
    /// and applying `rmsnorm_sharded` to each slice with its half of the weight gives the full
    /// `rmsnorm`: within 1e-6 in F32 (only the summation order of the squares differs), and for
    /// BF16 inputs within one BF16 rounding step of `x·inv_rms` times the weight. One slice
    /// spanning the whole row with its own sum is bitwise `rmsnorm`. Breaks if `rmsnorm_sharded`
    /// divides by the slice width instead of `full_dim`, ignores the given sum, or rounds
    /// anywhere else than rmsnorm.
    #[test]
    fn sharded_norm_matches_full() {
        let mem = HostMemory::new(DeviceId(0), 1 << 24) as Arc<dyn DeviceMemory>;
        let cpu = cpu_reference_provider();
        let sharded = cpu.sharded_norm().expect("sharded norm family");
        let (rows, full, half) = (4usize, 256usize, 128usize);
        let eps = 1e-5;
        for dtype in [DType::F32, DType::BF16] {
            let x_values: Vec<f32> = seeded(7, rows * full).iter().map(|v| v * 3.0).collect();
            let w_values: Vec<f32> = seeded(8, full).iter().map(|v| 1.0 + v / 2.0).collect();
            let x = tensor(&mem, &[rows, full], dtype, &x_values);
            let w = tensor(&mem, &[full], dtype, &w_values);

            let want = Tensor::empty(&mem, &[rows, full], dtype).expect("out");
            cpu.norm()
                .expect("norm family")
                .execute(&mut NormContext {
                    x: x.view(),
                    weight: w.view(),
                    out: want.view(),
                    eps,
                })
                .expect("rmsnorm");
            let want = load(&want.view()).expect("load");

            let sum_cfg = RowSumsqConfig {
                dim: half as u32,
                dtype,
            };
            let norm_cfg = RmsnormShardedConfig {
                dim: half as u32,
                full_dim: full as u32,
                dtype,
            };
            assert!(sharded.supports_row_sumsq(&sum_cfg));
            assert!(sharded.supports_rmsnorm_sharded(&norm_cfg));
            assert!(!sharded.supports_rmsnorm_sharded(&RmsnormShardedConfig {
                full_dim: 64,
                ..norm_cfg
            }));
            assert_eq!(sharded.implementation_row_sumsq(&sum_cfg), "cpu_row_sumsq");
            assert_eq!(
                sharded.implementation_rmsnorm_sharded(&norm_cfg),
                "cpu_rmsnorm_sharded"
            );

            // Each rank's partial sums, then the all-reduce (an f32 sum).
            let mut total = vec![0f32; rows];
            for rank in 0..2 {
                let partial = Tensor::empty(&mem, &[rows], DType::F32).expect("sumsq");
                sharded
                    .row_sumsq(&mut RowSumsqContext {
                        x: slice_view(&x, rank * half, half),
                        sumsq: partial.view(),
                    })
                    .expect("row_sumsq");
                for (t, p) in total.iter_mut().zip(load(&partial.view()).expect("load")) {
                    *t += p;
                }
            }
            let sumsq = tensor(&mem, &[rows], DType::F32, &total);
            let got = Tensor::empty(&mem, &[rows, full], dtype).expect("out");
            for rank in 0..2 {
                let w_slice = tensor(
                    &mem,
                    &[half],
                    dtype,
                    &w_values[rank * half..(rank + 1) * half],
                );
                sharded
                    .rmsnorm_sharded(&mut RmsnormShardedContext {
                        x: slice_view(&x, rank * half, half),
                        sumsq: sumsq.view(),
                        weight: w_slice.view(),
                        out: slice_view(&got, rank * half, half),
                        full_dim: full as u32,
                        eps,
                    })
                    .expect("rmsnorm_sharded");
            }
            let got = load(&got.view()).expect("load");
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                let d = (g - w).abs();
                let bound = match dtype {
                    DType::F32 => 1e-6,
                    // A rounding flip of x·inv_rms (|x·inv_rms| < 2 here, so one BF16 step is
                    // at most 2^-7) times |γ| ≤ 1.5, plus one step of the output's rounding.
                    _ => 2f32.powi(-7) * (1.5 + w.abs()),
                };
                assert!(
                    d <= bound,
                    "{} element {i}: sharded {g} vs full {w}",
                    dtype.as_str()
                );
            }

            // One slice spanning the whole row with its own sum: bitwise rmsnorm.
            let own = Tensor::empty(&mem, &[rows], DType::F32).expect("sumsq");
            sharded
                .row_sumsq(&mut RowSumsqContext {
                    x: x.view(),
                    sumsq: own.view(),
                })
                .expect("row_sumsq");
            let whole = Tensor::empty(&mem, &[rows, full], dtype).expect("out");
            sharded
                .rmsnorm_sharded(&mut RmsnormShardedContext {
                    x: x.view(),
                    sumsq: own.view(),
                    weight: w.view(),
                    out: whole.view(),
                    full_dim: full as u32,
                    eps,
                })
                .expect("rmsnorm_sharded");
            let whole = load(&whole.view()).expect("load");
            assert!(
                whole
                    .iter()
                    .zip(&want)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{}: one full-width shard is bitwise rmsnorm",
                dtype.as_str()
            );
        }
    }

    /// The fused op leaves `residual` and `out` bitwise equal to `add` into the residual followed
    /// by `rmsnorm` of it. Breaks if the fused op normalises the unrounded sum, skips the in-place
    /// residual update or rounds anywhere else than the separate ops do.
    #[test]
    fn add_rmsnorm_equals_add_then_rmsnorm() {
        let mem = HostMemory::new(DeviceId(0), 1 << 24) as Arc<dyn DeviceMemory>;
        let cpu = cpu_reference_provider();
        let fused = cpu.add_rmsnorm().expect("add_rmsnorm family");
        let dtype = DType::BF16;
        let eps = 1e-5;
        let bytes = |t: &Tensor| t.storage.whole().read_bytes().expect("read");
        for rows in [1usize, 7, 64] {
            for dim in [64usize, 3072] {
                let cfg = AddRmsnormConfig {
                    dtype,
                    dim: dim as u32,
                };
                assert!(fused.supports(&cfg));
                assert_eq!(fused.implementation(&cfg), "cpu_add_rmsnorm");
                let seed = (rows * 10_000 + dim) as u64;
                let start: Vec<f32> = seeded(seed, rows * dim).iter().map(|v| v * 4.0).collect();
                let x = tensor(&mem, &[rows, dim], dtype, &seeded(seed + 1, rows * dim));
                let weight: Vec<f32> = seeded(seed + 2, dim)
                    .iter()
                    .map(|v| 1.0 + v / 2.0)
                    .collect();
                let weight = tensor(&mem, &[dim], dtype, &weight);

                let residual_a = tensor(&mem, &[rows, dim], dtype, &start);
                let out_a = Tensor::empty(&mem, &[rows, dim], dtype).expect("out");
                cpu.elementwise()
                    .expect("add family")
                    .execute(&mut ElementwiseContext {
                        a: residual_a.view(),
                        b: x.view(),
                        out: residual_a.view(),
                    })
                    .expect("add");
                cpu.norm()
                    .expect("norm family")
                    .execute(&mut NormContext {
                        x: residual_a.view(),
                        weight: weight.view(),
                        out: out_a.view(),
                        eps,
                    })
                    .expect("rmsnorm");

                let residual_b = tensor(&mem, &[rows, dim], dtype, &start);
                let out_b = Tensor::empty(&mem, &[rows, dim], dtype).expect("out");
                fused
                    .execute(&mut AddRmsnormContext {
                        residual: residual_b.view(),
                        x: x.view(),
                        weight: weight.view(),
                        out: out_b.view(),
                        eps,
                    })
                    .expect("add_rmsnorm");

                assert_ne!(
                    bytes(&residual_b),
                    bytes(&tensor(&mem, &[rows, dim], dtype, &start)),
                    "the residual is updated in place"
                );
                assert_eq!(
                    bytes(&residual_b),
                    bytes(&residual_a),
                    "residual {rows}x{dim}"
                );
                assert_eq!(bytes(&out_b), bytes(&out_a), "out {rows}x{dim}");
            }
        }
    }
}
