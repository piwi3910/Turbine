//! `silu_mul` on the cpu-reference provider (SiLU rounded before the multiply).
use super::{CpuReference, expect_rank, expect_shape, is_float, load, math, round_to, store};
use crate::KernelError;
use crate::ops::{ActivationConfig, ActivationContext, ActivationKernel};

impl ActivationKernel for CpuReference {
    fn supports(&self, cfg: &ActivationConfig) -> bool {
        is_float(cfg.dtype)
    }

    fn implementation(&self, _cfg: &ActivationConfig) -> String {
        "cpu_silu_mul".into()
    }

    fn execute(&self, ctx: &mut ActivationContext<'_>) -> Result<(), KernelError> {
        expect_rank("gate", &ctx.gate, 2)?;
        let shape = ctx.gate.shape.clone();
        expect_shape("up", &ctx.up, &shape)?;
        expect_shape("out", &ctx.out, &shape)?;
        let dt = ctx.out.dtype;
        let gate = load(&ctx.gate)?;
        let up = load(&ctx.up)?;
        let out: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| round_to(dt, math::silu(g)) * u)
            .collect();
        store(&ctx.out, &out)
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::test_util::*;
    use crate::cpu::*;

    #[test]
    fn silu_mul_embedding_and_add() {
        let mem = host();
        let gate = tensor(&mem, &[1, 2], DType::F32, &[0.0, 1.0]);
        let up = tensor(&mem, &[1, 2], DType::F32, &[3.0, 2.0]);
        let out = Tensor::empty(&mem, &[1, 2], DType::F32).expect("alloc");
        CpuReference
            .activation()
            .expect("activation family")
            .execute(&mut ActivationContext {
                gate: gate.view(),
                up: up.view(),
                out: out.view(),
            })
            .expect("silu_mul");
        assert_close(
            &load(&out.view()).expect("load"),
            &[0.0, 2.0 * sigmoid(1.0)],
            1e-6,
        );

        // Table rows 0..2 of a shard starting at id 10; ids outside the shard give zero rows.
        let table = tensor(&mem, &[2, 2], DType::BF16, &[1.0, 2.0, 3.0, 4.0]);
        let ids = i32_tensor(&mem, &[11, 9, 10, 12]);
        let emb = Tensor::empty(&mem, &[4, 2], DType::BF16).expect("alloc");
        CpuReference
            .embedding()
            .expect("embedding family")
            .execute(&mut EmbeddingContext {
                ids: ids.view(),
                table: table.view(),
                out: emb.view(),
                vocab_offset: 10,
            })
            .expect("embedding");
        assert_eq!(
            load(&emb.view()).expect("load"),
            [3.0, 4.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0]
        );

        // 1 + 2^-9 is not representable in bf16: the sum rounds to 1.
        let a = tensor(&mem, &[2], DType::F32, &[1.0, 2.0]);
        let b = tensor(&mem, &[2], DType::F32, &[2f32.powi(-9), 0.5]);
        let sum = Tensor::empty(&mem, &[2], DType::BF16).expect("alloc");
        CpuReference
            .elementwise()
            .expect("elementwise family")
            .execute(&mut ElementwiseContext {
                a: a.view(),
                b: b.view(),
                out: sum.view(),
            })
            .expect("add");
        assert_eq!(load(&sum.view()).expect("load"), [1.0, 2.5]);
    }
}
