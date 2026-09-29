//! Attention on the cpu-reference provider: the contiguous kinds here, the paged kinds through
//! `paged.rs`.
use super::{
    CpuReference, expect_rank, expect_shape, invalid, is_float, load, math, paged, round_to, store,
};
use turbine_core::types::DType;

use crate::KernelError;
use crate::ops::{AttentionConfig, AttentionContext, AttentionKernel, PagedAttentionContext};

impl AttentionKernel for CpuReference {
    fn supports(&self, cfg: &AttentionConfig) -> bool {
        cfg.num_kv_heads > 0
            && cfg.num_q_heads.is_multiple_of(cfg.num_kv_heads)
            && cfg.head_dim > 0
            && (is_float(cfg.dtype)
                || cfg.kind.is_paged()
                    && (cfg.dtype == DType::F8E4M3 || cfg.dtype.tq_record_bytes().is_some()))
            && if cfg.kind.is_paged() {
                cfg.block_tokens.is_some_and(|b| b > 0)
            } else {
                cfg.block_tokens.is_none()
            }
    }

    fn implementation(&self, cfg: &AttentionConfig) -> String {
        if cfg.dtype.tq_record_bytes().is_some() {
            "cpu_attention_paged_tqkv_f64acc".into()
        } else if cfg.dtype == DType::F8E4M3 {
            "cpu_attention_paged_fp8kv_f32acc".into()
        } else if cfg.kind.is_paged() {
            "cpu_attention_paged_f32acc".into()
        } else {
            "cpu_attention_f32acc".into()
        }
    }

    fn execute(&self, ctx: &mut AttentionContext<'_>) -> Result<(), KernelError> {
        if ctx.cfg.kind.is_paged() || !AttentionKernel::supports(self, &ctx.cfg) {
            return Err(KernelError::Unsupported {
                message: format!("cpu-reference attention {}", ctx.cfg),
            });
        }
        let hq = ctx.cfg.num_q_heads as usize;
        let hkv = ctx.cfg.num_kv_heads as usize;
        let d = ctx.cfg.head_dim as usize;
        expect_rank("q", &ctx.q, 3)?;
        let q_len = ctx.q.shape[0];
        let q_start = ctx.q_start as usize;
        let kv_len = q_start + q_len;
        expect_shape("q", &ctx.q, &[q_len, hq, d])?;
        expect_shape("out", &ctx.out, &[q_len, hq, d])?;
        for (name, cache) in [("k_cache", &ctx.k_cache), ("v_cache", &ctx.v_cache)] {
            expect_rank(name, cache, 3)?;
            expect_shape(name, cache, &[cache.shape[0], hkv, d])?;
            if cache.shape[0] < kv_len {
                return Err(invalid(format!(
                    "{name} holds {} rows but q_start {q_start} + q_len {q_len} are needed",
                    cache.shape[0]
                )));
            }
        }
        let q = load(&ctx.q)?;
        let k = load(&ctx.k_cache.rows(0, kv_len))?;
        let v = load(&ctx.v_cache.rows(0, kv_len))?;
        let shape = math::AttnShape {
            q_len,
            q_start,
            hq,
            hkv,
            d,
            causal: ctx.cfg.causal,
        };
        let dt = ctx.cfg.dtype;
        store(
            &ctx.out,
            &math::attention(&q, &k, &v, &shape, ctx.scale, |p| round_to(dt, p)),
        )
    }

    fn execute_paged(&self, ctx: &mut PagedAttentionContext<'_>) -> Result<(), KernelError> {
        if !ctx.cfg.kind.is_paged() || !AttentionKernel::supports(self, &ctx.cfg) {
            return Err(KernelError::Unsupported {
                message: format!("cpu-reference paged attention {}", ctx.cfg),
            });
        }
        paged::attention(ctx)
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::test_util::*;
    use crate::cpu::*;

    #[test]
    fn gemm_and_causal_gqa_attention_reference() {
        let mem = host();
        let cpu = cpu_reference_provider();
        assert_eq!(cpu.id().0, "cpu-reference");

        // [[1,2,3],[-1,0.5,2]] · [[1,0,1],[2,1,0]]ᵀ, BF16 in (exact), F32 out.
        let a = tensor(&mem, &[2, 3], DType::BF16, &[1.0, 2.0, 3.0, -1.0, 0.5, 2.0]);
        let b = tensor(&mem, &[2, 3], DType::BF16, &[1.0, 0.0, 1.0, 2.0, 1.0, 0.0]);
        let c = Tensor::empty(&mem, &[2, 2], DType::F32).expect("alloc");
        let gemm_cfg = GemmConfig {
            n: 2,
            k: 3,
            trans_b: true,
            a_dtype: DType::BF16,
            b_dtype: DType::BF16,
            c_dtype: DType::F32,
        };
        let gemm = cpu.gemm().expect("gemm family");
        assert!(gemm.supports(&gemm_cfg));
        assert_eq!(gemm.implementation(&gemm_cfg), "cpu_gemm_f32acc");
        gemm.execute(&mut GemmContext {
            a: a.view(),
            b: b.view(),
            c: c.view(),
            trans_b: true,
            alpha: 1.0,
            beta: 0.0,
            prefill: false,
        })
        .expect("gemm");
        assert_eq!(load(&c.view()).expect("load"), [4.0, 4.0, 1.0, -1.5]);

        // 2 query heads sharing 1 KV head, head_dim 2; keys one-hot, v = [[10,0],[0,20]].
        let cfg = AttentionConfig {
            kind: AttentionKind::Prefill,
            num_q_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            dtype: DType::F32,
            block_tokens: None,
            causal: true,
        };
        // token 0: both heads [1,0]; token 1: head 0 [1,0], head 1 [0,1].
        let q = tensor(
            &mem,
            &[2, 2, 2],
            DType::F32,
            &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        );
        let k = tensor(&mem, &[2, 1, 2], DType::F32, &[1.0, 0.0, 0.0, 1.0]);
        let v = tensor(&mem, &[2, 1, 2], DType::F32, &[10.0, 0.0, 0.0, 20.0]);
        let out = Tensor::empty(&mem, &[2, 2, 2], DType::F32).expect("alloc");
        let attn = cpu.attention().expect("attention family");
        assert!(attn.supports(&cfg));
        assert_eq!(attn.implementation(&cfg), "cpu_attention_f32acc");
        attn.execute(&mut AttentionContext {
            cfg,
            q: q.view(),
            k_cache: k.view(),
            v_cache: v.view(),
            out: out.view(),
            q_start: 0,
            scale: 1.0,
        })
        .expect("attention");
        let s = sigmoid(1.0);
        assert_close(
            &load(&out.view()).expect("load"),
            &[
                10.0,
                0.0,
                10.0,
                0.0,
                10.0 * s,
                20.0 * (1.0 - s),
                10.0 * (1.0 - s),
                20.0 * s,
            ],
            1e-5,
        );
    }

    /// BF16 attention follows CK FMHA (and PyTorch's CPU flash attention): the unnormalised
    /// `exp(s − max)` feeds the row sum in f32, is rounded to BF16 for P·V (accumulated in f32),
    /// and the output is divided by the f32 sum. Breaks if P stays f32, if the sum is taken over
    /// the rounded values, or if P is normalised before rounding.
    #[test]
    fn bf16_attention_rounds_unnormalised_probabilities_like_ck() {
        let mem = host();
        let run = |dtype: DType| {
            let cfg = AttentionConfig {
                kind: AttentionKind::Decode,
                num_q_heads: 1,
                num_kv_heads: 1,
                head_dim: 1,
                dtype,
                block_tokens: None,
                causal: true,
            };
            // Scores [0, −0.25] (all inputs exact in BF16); v = [0, 1].
            let q = tensor(&mem, &[1, 1, 1], dtype, &[-0.25]);
            let k = tensor(&mem, &[2, 1, 1], dtype, &[0.0, 1.0]);
            let v = tensor(&mem, &[2, 1, 1], dtype, &[0.0, 1.0]);
            let out = Tensor::empty(&mem, &[1, 1, 1], DType::F32).expect("alloc");
            CpuReference
                .attention()
                .expect("attention family")
                .execute(&mut AttentionContext {
                    cfg,
                    q: q.view(),
                    k_cache: k.view(),
                    v_cache: v.view(),
                    out: out.view(),
                    q_start: 1,
                    scale: 1.0,
                })
                .expect("attention");
            load(&out.view()).expect("load")[0]
        };
        let p = (-0.25f32).exp();
        let p_bf16 = bf16::from_f32(p).to_f32();
        assert_ne!(p_bf16, p, "the probability must not be exact in BF16");
        assert_eq!(run(DType::BF16), p_bf16 / (1.0 + p));
        assert_eq!(run(DType::F32), p / (1.0 + p));
    }

    #[test]
    fn decode_attends_the_cache_prefix_only() {
        let mem = host();
        let cfg = AttentionConfig {
            kind: AttentionKind::Decode,
            num_q_heads: 1,
            num_kv_heads: 1,
            head_dim: 1,
            dtype: DType::F32,
            block_tokens: None,
            causal: true,
        };
        // Capacity 4, rows [0, 2) valid; row 3 holds garbage that must not be read.
        let k = tensor(&mem, &[4, 1, 1], DType::F32, &[0.0, 0.0, 0.0, 1e9]);
        let v = tensor(&mem, &[4, 1, 1], DType::F32, &[2.0, 6.0, 0.0, 1e9]);
        let q = tensor(&mem, &[1, 1, 1], DType::F32, &[1.0]);
        let out = Tensor::empty(&mem, &[1, 1, 1], DType::F32).expect("alloc");
        CpuReference
            .attention()
            .expect("attention family")
            .execute(&mut AttentionContext {
                cfg,
                q: q.view(),
                k_cache: k.view(),
                v_cache: v.view(),
                out: out.view(),
                q_start: 1,
                scale: 1.0,
            })
            .expect("attention");
        assert_eq!(load(&out.view()).expect("load"), [4.0]);

        // Too few cache rows for q_start + q_len is an argument error, not a panic.
        let short = tensor(&mem, &[1, 1, 1], DType::F32, &[0.0]);
        let err = CpuReference
            .attention()
            .expect("attention family")
            .execute(&mut AttentionContext {
                cfg,
                q: q.view(),
                k_cache: short.view(),
                v_cache: short.view(),
                out: out.view(),
                q_start: 1,
                scale: 1.0,
            })
            .expect_err("short cache");
        assert!(err.to_string().contains("k_cache holds 1 rows"), "{err}");
    }
}
