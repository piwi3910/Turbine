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

    fn attn_factor_supported(&self) -> bool {
        true
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
        if !(ctx.attn_factor.is_finite() && ctx.attn_factor > 0.0) {
            return Err(KernelError::InvalidArgument {
                message: format!("rope attn_factor {} is not finite and > 0", ctx.attn_factor),
            });
        }
        let positions = load_i32(&ctx.positions)?;
        let inv_freq = load(&ctx.inv_freq)?;
        for (x_view, heads) in [(&ctx.q, hq), (&ctx.k, hkv)] {
            let dt = x_view.dtype;
            let mut x = load(x_view)?;
            math::rope(
                &mut x,
                &positions,
                &inv_freq,
                heads,
                d,
                rotary_dim,
                ctx.attn_factor,
                |v| round_to(dt, v),
            );
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
                attn_factor: 1.0,
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

    /// Phase 6a Task 28a (spec S-15): with an attention factor `m`, cos and sin are multiplied
    /// by `m` in f32 and only then rounded to BF16, as transformers does (`cos() · m`, then
    /// `.to(bfloat16)`). Rotating the unit pair `(1, 0)` returns `(c, s)`, so the op's cos and
    /// sin are read back directly. The angle is chosen where rounding first (`round(round(cos) ·
    /// m)`) gives a different BF16 value, and a second pair checks a full rotation. Breaks if the
    /// factor is applied after rounding, folded elsewhere (not applied) or ignored.
    #[test]
    fn rope_attn_factor_scales_before_rounding() {
        let mem = host();
        let m = 1.277_258_9_f32;
        let bf = |v: f32| round_to(DType::BF16, v);
        // An angle whose cos and sin both round differently when rounded before scaling.
        let differs = |f: f32| {
            bf(f.cos() * m) != bf(bf(f.cos()) * m) && bf(f.sin() * m) != bf(bf(f.sin()) * m)
        };
        let freq = (1..2000)
            .map(|n| n as f32 * 1e-3)
            .find(|&f| differs(f))
            .expect("an angle where the rounding order matters");
        let cfg = RopeConfig {
            num_q_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            rotary_dim: 2,
            dtype: DType::BF16,
        };
        let (x1, x2) = (bf(1.75), bf(-0.625));
        let q = tensor(&mem, &[1, 1, 2], DType::BF16, &[1.0, 0.0]);
        let k = tensor(&mem, &[1, 1, 2], DType::BF16, &[x1, x2]);
        let positions = i32_tensor(&mem, &[1]);
        let inv_freq = tensor(&mem, &[1], DType::F32, &[freq]);
        CpuReference
            .rope()
            .expect("rope family")
            .execute(&mut RopeContext {
                cfg,
                q: q.view(),
                k: k.view(),
                positions: positions.view(),
                inv_freq: inv_freq.view(),
                attn_factor: m,
            })
            .expect("rope");
        let (c, s) = (bf(freq.cos() * m), bf(freq.sin() * m));
        assert_eq!(
            load(&q.view()).expect("q"),
            [c, s],
            "cos·m, sin·m at {freq}"
        );
        assert_ne!(c, bf(bf(freq.cos()) * m));
        assert_ne!(s, bf(bf(freq.sin()) * m));
        assert_eq!(
            load(&k.view()).expect("k"),
            [bf(bf(x1 * c) + bf(-x2 * s)), bf(bf(x2 * c) + bf(x1 * s))]
        );
        assert!(CpuReference.rope().expect("rope").attn_factor_supported());
    }
}
