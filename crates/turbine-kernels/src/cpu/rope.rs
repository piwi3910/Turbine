//! Half-split rotary embedding on the cpu-reference provider.
use super::{
    CpuReference, expect_rank, expect_shape, is_float, load, load_i32, math, round_to, store,
};
use crate::KernelError;
use crate::ops::{RopeConfig, RopeContext, RopeKernel};

impl RopeKernel for CpuReference {
    fn supports(&self, cfg: &RopeConfig) -> bool {
        cfg.rotary_dim > 0
            && cfg.rotary_dim <= cfg.head_dim
            && cfg.rotary_dim.is_multiple_of(2)
            && is_float(cfg.dtype)
    }

    fn implementation(&self, _cfg: &RopeConfig) -> String {
        "cpu_rope_half_split".into()
    }

    fn execute(&self, ctx: &mut RopeContext<'_>) -> Result<(), KernelError> {
        if !RopeKernel::supports(self, &ctx.cfg) {
            return Err(KernelError::Unsupported {
                message: format!("cpu-reference rope {}", ctx.cfg),
            });
        }
        let d = ctx.cfg.head_dim as usize;
        let rotary_dim = ctx.cfg.rotary_dim as usize;
        let (hq, hkv) = (ctx.cfg.num_q_heads as usize, ctx.cfg.num_kv_heads as usize);
        expect_rank("positions", &ctx.positions, 1)?;
        let tokens = ctx.positions.shape[0];
        expect_shape("q", &ctx.q, &[tokens, hq, d])?;
        expect_shape("k", &ctx.k, &[tokens, hkv, d])?;
        expect_shape("inv_freq", &ctx.inv_freq, &[rotary_dim / 2])?;
        let positions = load_i32(&ctx.positions)?;
        let inv_freq = load(&ctx.inv_freq)?;
        for (x_view, heads) in [(&ctx.q, hq), (&ctx.k, hkv)] {
            let dt = x_view.dtype;
            let mut x = load(x_view)?;
            math::rope(&mut x, &positions, &inv_freq, heads, d, rotary_dim, |v| {
                round_to(dt, v)
            });
            store(x_view, &x)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::test_util::*;
    use crate::cpu::*;

    #[test]
    fn rope_half_split_rotates_q_and_k() {
        let mem = host();
        let cfg = RopeConfig {
            num_q_heads: 1,
            num_kv_heads: 1,
            head_dim: 4,
            rotary_dim: 2,
            dtype: DType::F32,
        };
        // Position 0 is the identity; position 1 with inv_freq π/2 maps (x1, x2) to (−x2, x1).
        let q = tensor(
            &mem,
            &[2, 1, 4],
            DType::F32,
            &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        );
        let k = tensor(
            &mem,
            &[2, 1, 4],
            DType::F32,
            &[5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0],
        );
        let positions = i32_tensor(&mem, &[0, 1]);
        let inv_freq = tensor(&mem, &[1], DType::F32, &[std::f32::consts::FRAC_PI_2]);
        CpuReference
            .rope()
            .expect("rope family")
            .execute(&mut RopeContext {
                cfg,
                q: q.view(),
                k: k.view(),
                positions: positions.view(),
                inv_freq: inv_freq.view(),
            })
            .expect("rope");
        assert_close(
            &load(&q.view()).expect("q"),
            &[1.0, 2.0, 3.0, 4.0, -2.0, 1.0, 3.0, 4.0],
            1e-6,
        );
        assert_close(
            &load(&k.view()).expect("k"),
            &[5.0, 6.0, 7.0, 8.0, -6.0, 5.0, 7.0, 8.0],
            1e-6,
        );
    }
}
