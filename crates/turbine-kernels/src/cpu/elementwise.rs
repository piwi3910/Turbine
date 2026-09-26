//! Elementwise `add` on the cpu-reference provider.
use super::{CpuReference, invalid, is_float, load, store};
use crate::KernelError;
use crate::ops::{ElementwiseConfig, ElementwiseContext, ElementwiseKernel};

impl ElementwiseKernel for CpuReference {
    fn supports(&self, cfg: &ElementwiseConfig) -> bool {
        is_float(cfg.dtype)
    }

    fn implementation(&self, _cfg: &ElementwiseConfig) -> String {
        "cpu_add".into()
    }

    fn execute(&self, ctx: &mut ElementwiseContext<'_>) -> Result<(), KernelError> {
        let n = ctx.out.numel();
        if ctx.a.numel() != n || ctx.b.numel() != n {
            return Err(invalid(format!(
                "add of {} and {} elements into {n}",
                ctx.a.numel(),
                ctx.b.numel()
            )));
        }
        let a = load(&ctx.a)?;
        let b = load(&ctx.b)?;
        let out: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        store(&ctx.out, &out)
    }
}
