//! `gemm` on the cpu-reference provider: f32 accumulation, rounded to `c`'s dtype.
use turbine_tensor::TensorView;

use super::{CpuReference, expect_rank, expect_shape, is_float, load, math, round_to, store};
use crate::KernelError;
use crate::ops::{GemmConfig, GemmContext, GemmFused, GemmFusion, GemmKernel};

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

    /// Every fused form (kernel ABI v2.14 / v2.15) on the same calls as
    /// [`Self::fuses_silu_mul`], the RMSNorm input not with the SwiGLU one.
    fn fuses(&self, cfg: &GemmConfig, rows: usize, prefill: bool, fusion: GemmFusion) -> bool {
        !(fusion.silu_mul && fusion.rmsnorm)
            && fusion != GemmFusion::default()
            && self.fuses_silu_mul(cfg, rows, prefill)
    }

    /// The unfused ops' numbers: the operand as `silu_mul` or `rmsnorm` stores it, the product
    /// rounded to `c`'s dtype, then the residual added and rounded again (the `add` op).
    fn execute_fused(
        &self,
        ctx: &mut GemmContext<'_>,
        f: GemmFused<'_>,
    ) -> Result<(), KernelError> {
        expect_rank("a", &ctx.a, 2)?;
        expect_rank("c", &ctx.c, 2)?;
        let (m, k) = (ctx.a.shape[0], ctx.a.shape[1]);
        let n = ctx.c.shape[1];
        expect_shape("c", &ctx.c, &[m, n])?;
        expect_shape("b", &ctx.b, &if ctx.trans_b { [n, k] } else { [k, n] })?;
        let dt = ctx.a.dtype;
        // Read before the product overwrites it (the residual may be `c` itself).
        let residual = match &f.residual {
            Some(r) => {
                expect_shape("residual", r, &[m, n])?;
                Some(load(r)?)
            }
            None => None,
        };
        let a = match (&f.up, &f.norm) {
            (Some(_), Some(_)) => {
                return Err(KernelError::Unsupported {
                    message: "the SwiGLU and RMSNorm GEMM inputs do not combine".into(),
                });
            }
            (Some(up), None) => {
                expect_shape("up", up, &[m, k])?;
                let (gate, up) = (load(&ctx.a)?, load(up)?);
                gate.iter()
                    .zip(&up)
                    .map(|(&g, &u)| round_to(dt, round_to(dt, math::silu(g)) * u))
                    .collect()
            }
            (None, Some((w, eps))) => {
                expect_shape("norm weight", w, &[k])?;
                let round = |v| round_to(dt, v);
                math::rmsnorm(&load(&ctx.a)?, &load(w)?, k, *eps, round, round)
            }
            (None, None) => load(&ctx.a)?,
        };
        gemm_on(ctx, &a, m, n, k)?;
        if let Some(r) = residual {
            let out_dt = ctx.c.dtype;
            let c: Vec<f32> = load(&ctx.c)?
                .iter()
                .zip(&r)
                .map(|(&c, &r)| round_to(out_dt, c + r))
                .collect();
            store(&ctx.c, &c)?;
        }
        Ok(())
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
    use crate::ops::{GemmFused, GemmFusion};

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

    /// The v2.15 forms on the reference: the RMSNorm input gives the bits of `rmsnorm` into a
    /// buffer and the GEMM on it, and the residual epilogue written in place over the residual
    /// the bits of the GEMM then `add`; the SwiGLU and RMSNorm inputs do not combine. Breaks if
    /// the reference's fused path drifts from its unfused ops (host decode runs take it).
    #[test]
    fn norm_input_and_residual_epilogue_match_the_unfused_ops() {
        let mem = host();
        let (n, k) = (6, 16);
        let x = tensor(&mem, &[1, k], DType::BF16, &seeded(5, k));
        let g = tensor(&mem, &[k], DType::BF16, &seeded(6, k));
        let w = tensor(&mem, &[n, k], DType::BF16, &seeded(7, n * k));
        let res = seeded(8, n);
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
        let h = Tensor::empty(&mem, &[1, k], DType::BF16).expect("alloc");
        CpuReference
            .norm()
            .expect("norm family")
            .execute(&mut NormContext {
                x: x.view(),
                weight: g.view(),
                out: h.view(),
                eps: 1e-5,
            })
            .expect("rmsnorm");
        let proj = Tensor::empty(&mem, &[1, n], DType::BF16).expect("alloc");
        gemm.execute(&mut ctx(h.view(), proj.view())).expect("gemm");
        let want = tensor(&mem, &[1, n], DType::BF16, &res);
        CpuReference
            .elementwise()
            .expect("elementwise family")
            .execute(&mut ElementwiseContext {
                a: want.view(),
                b: proj.view(),
                out: want.view(),
            })
            .expect("add");
        let got = tensor(&mem, &[1, n], DType::BF16, &res);
        gemm.execute_fused(
            &mut ctx(x.view(), got.view()),
            GemmFused {
                norm: Some((g.view(), 1e-5)),
                residual: Some(got.view()),
                ..GemmFused::default()
            },
        )
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
        let norm_res = GemmFusion {
            rmsnorm: true,
            residual: true,
            ..GemmFusion::default()
        };
        assert!(gemm.fuses(&cfg, 1, false, norm_res));
        assert!(!gemm.fuses(&cfg, 1, true, norm_res));
        assert!(!gemm.fuses(&cfg, 2, false, norm_res));
        let both = GemmFusion {
            silu_mul: true,
            ..norm_res
        };
        assert!(!gemm.fuses(&cfg, 1, false, both));
        assert!(!gemm.fuses(&cfg, 1, false, GemmFusion::default()));
    }
}
