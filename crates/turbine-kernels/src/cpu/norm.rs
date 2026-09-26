//! `rmsnorm` and the fused `add_rmsnorm` on the cpu-reference provider (Hugging Face
//! LlamaRMSNorm rounding).
use super::{CpuReference, expect_rank, expect_shape, is_float, load, math, round_to, store};
use crate::KernelError;
use crate::ops::{
    AddRmsnormConfig, AddRmsnormContext, AddRmsnormKernel, NormConfig, NormContext, NormKernel,
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
