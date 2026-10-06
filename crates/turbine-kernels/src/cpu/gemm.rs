//! `gemm` on the cpu-reference provider: f32 accumulation, rounded to `c`'s dtype.
use turbine_tensor::TensorView;

use super::{CpuReference, expect_rank, expect_shape, is_float, load, math, round_to, store};
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
        gemm_on(ctx, &a, m, n, k)
    }

    /// The card GEMV tier's fusion (kernel ABI v2.14) on the reference: single-row calls of a
    /// `gemv` config in decode steps, so host tests run the executor's fused SwiGLU path.
    fn fuses_silu_mul(&self, cfg: &GemmConfig, rows: usize, prefill: bool) -> bool {
        cfg.gemv && rows == 1 && !prefill && self.supports(cfg)
    }

    /// The operand as the `silu_mul` op stores it (rounded to `a`'s dtype), then [`Self::execute`]'s
    /// product: bit for bit the unfused pair.
    fn execute_silu_mul(
        &self,
        ctx: &mut GemmContext<'_>,
        up: TensorView<'_>,
    ) -> Result<(), KernelError> {
        expect_rank("a", &ctx.a, 2)?;
        expect_rank("c", &ctx.c, 2)?;
        let (m, k) = (ctx.a.shape[0], ctx.a.shape[1]);
        let n = ctx.c.shape[1];
        expect_shape("up", &up, &[m, k])?;
        expect_shape("c", &ctx.c, &[m, n])?;
        expect_shape("b", &ctx.b, &if ctx.trans_b { [n, k] } else { [k, n] })?;
        let dt = ctx.a.dtype;
        let gate = load(&ctx.a)?;
        let up = load(&up)?;
        let a: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| round_to(dt, round_to(dt, math::silu(g)) * u))
            .collect();
        gemm_on(ctx, &a, m, n, k)
    }
}

/// `c = alpha · a · b(ᵀ) + beta · c` for the loaded operand `a` (`[m, k]`).
fn gemm_on(
    ctx: &GemmContext<'_>,
    a: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), KernelError> {
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
    let c = math::gemm(a, &b, &c_old, &shape, ctx.alpha, ctx.beta);
    store(&ctx.c, &c)
}

#[cfg(test)]
mod tests {
    use crate::cpu::test_util::*;
    use crate::cpu::*;

    /// The fused SwiGLU input (gate and up columns of one `[1, 2·k]` row) gives the bits of
    /// `silu_mul` into a buffer and the GEMM on it, and only single-row decode calls of a `gemv`
    /// config take it. Breaks if the fused operand skips a rounding (SiLU or the product).
    #[test]
    fn silu_mul_input_matches_silu_mul_then_gemm() {
        let mem = host();
        let (n, k) = (6, 16);
        let gate_up = tensor(&mem, &[1, 2 * k], DType::BF16, &seeded(3, 2 * k));
        let w = tensor(&mem, &[n, k], DType::BF16, &seeded(4, n * k));
        let view = |off| TensorView::contiguous(gate_up.storage.whole(), off, &[1, k], DType::BF16);
        let act = Tensor::empty(&mem, &[1, k], DType::BF16).expect("alloc");
        CpuReference
            .activation()
            .expect("activation family")
            .execute(&mut ActivationContext {
                gate: view(0),
                up: view(k),
                out: act.view(),
            })
            .expect("silu_mul");
        let gemm = CpuReference.gemm().expect("gemm family");
        let ctx = |a, c| GemmContext {
            a,
            b: w.view(),
            c,
            trans_b: true,
            alpha: 1.0,
            beta: 0.0,
            prefill: false,
        };
        let want = Tensor::empty(&mem, &[1, n], DType::BF16).expect("alloc");
        gemm.execute(&mut ctx(act.view(), want.view()))
            .expect("gemm");
        let got = Tensor::empty(&mem, &[1, n], DType::BF16).expect("alloc");
        gemm.execute_silu_mul(&mut ctx(view(0), got.view()), view(k))
            .expect("fused");
        assert_eq!(
            load(&got.view()).expect("load"),
            load(&want.view()).expect("load")
        );

        let cfg = GemmConfig {
            n: n as u64,
            k: k as u64,
            trans_b: true,
            a_dtype: DType::BF16,
            b_dtype: DType::BF16,
            c_dtype: DType::BF16,
            gemv: true,
        };
        assert!(gemm.fuses_silu_mul(&cfg, 1, false));
        assert!(!gemm.fuses_silu_mul(&cfg, 1, true));
        assert!(!gemm.fuses_silu_mul(&cfg, 2, false));
        assert!(!gemm.fuses_silu_mul(&GemmConfig { gemv: false, ..cfg }, 1, false));
    }
}
