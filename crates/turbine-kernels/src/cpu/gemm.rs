//! `gemm` on the cpu-reference provider: f32 accumulation, rounded to `c`'s dtype.
use super::{CpuReference, expect_rank, expect_shape, is_float, load, math, store};
use crate::KernelError;
use crate::ops::{GemmConfig, GemmContext, GemmKernel};

impl GemmKernel for CpuReference {
    fn supports(&self, cfg: &GemmConfig) -> bool {
        is_float(cfg.a_dtype) && is_float(cfg.b_dtype) && is_float(cfg.c_dtype)
    }

    fn implementation(&self, _cfg: &GemmConfig) -> String {
        "cpu_gemm_f32acc".into()
    }

    fn execute(&self, ctx: &mut GemmContext<'_>) -> Result<(), KernelError> {
        expect_rank("a", &ctx.a, 2)?;
        expect_rank("c", &ctx.c, 2)?;
        let (m, k) = (ctx.a.shape[0], ctx.a.shape[1]);
        let n = ctx.c.shape[1];
        expect_shape("c", &ctx.c, &[m, n])?;
        expect_shape("b", &ctx.b, &if ctx.trans_b { [n, k] } else { [k, n] })?;
        let a = load(&ctx.a)?;
        let b = load(&ctx.b)?;
        let c_old = if ctx.beta != 0.0 {
            load(&ctx.c)?
        } else {
            Vec::new()
        };
        let shape = math::GemmShape {
            m,
            n,
            k,
            trans_b: ctx.trans_b,
        };
        let c = math::gemm(&a, &b, &c_old, &shape, ctx.alpha, ctx.beta);
        store(&ctx.c, &c)
    }
}
