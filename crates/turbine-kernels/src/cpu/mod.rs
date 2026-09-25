//! The `cpu-reference` provider (P1 S-6): pure Rust, always available, and the numerical
//! reference every other provider is tested against.
//!
//! Numerics: every op reads its views (honouring strides) into logical row-major f32 arrays,
//! computes with f32 accumulation sequential over the reduced dimension, and rounds to the
//! output view's dtype at the op boundary, plus the intermediate roundings where the Hugging
//! Face BF16 forward materialises a BF16 tensor (RMSNorm, RoPE, SiLU·up). Results are written
//! back with a read-modify-write of the view's byte range, so bytes between strided rows (other
//! KV cache rows, other heads) are preserved.
//!
//! Works on any `DeviceMemory` through `DeviceSlice::read_bytes`/`write_bytes`; with the host
//! backend that is a plain memory copy.
use std::sync::Arc;

use half::{bf16, f16};
use turbine_core::types::DType;
use turbine_tensor::TensorView;

use crate::KernelError;
use crate::ops::{
    ActivationConfig, ActivationContext, ActivationKernel, AttentionConfig, AttentionContext,
    AttentionKernel, ElementwiseConfig, ElementwiseContext, ElementwiseKernel, EmbeddingConfig,
    EmbeddingContext, EmbeddingKernel, GemmConfig, GemmContext, GemmKernel, KernelProvider,
    NormConfig, NormContext, NormKernel, ProviderId, RopeConfig, RopeContext, RopeKernel,
};

mod math;

/// The `cpu-reference` provider; stateless, implements every op family.
pub struct CpuReference;

/// The `cpu-reference` provider as a registry entry.
pub fn cpu_reference_provider() -> Arc<dyn KernelProvider> {
    Arc::new(CpuReference)
}

/// `v` rounded to `dtype` precision (round-to-nearest-even) and widened back to f32: the value
/// a tensor of that dtype would hold. Identity for F32 and the integer dtypes.
pub fn round_to(dtype: DType, v: f32) -> f32 {
    match dtype {
        DType::BF16 => bf16::from_f32(v).to_f32(),
        DType::F16 => f16::from_f32(v).to_f32(),
        _ => v,
    }
}

/// The floating-point element encodings the reference computes on.
#[derive(Clone, Copy)]
enum FloatCodec {
    Bf16,
    F16,
    F32,
}

impl FloatCodec {
    fn of(dtype: DType) -> Option<FloatCodec> {
        match dtype {
            DType::BF16 => Some(FloatCodec::Bf16),
            DType::F16 => Some(FloatCodec::F16),
            DType::F32 => Some(FloatCodec::F32),
            _ => None,
        }
    }

    fn require(dtype: DType) -> Result<FloatCodec, KernelError> {
        FloatCodec::of(dtype).ok_or_else(|| KernelError::Unsupported {
            message: format!(
                "cpu-reference computes on bf16/f16/f32 tensors, not {}",
                dtype.as_str()
            ),
        })
    }

    fn decode(self, b: &[u8]) -> f32 {
        match self {
            FloatCodec::Bf16 => bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32(),
            FloatCodec::F16 => f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32(),
            FloatCodec::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        }
    }

    fn encode(self, v: f32, out: &mut [u8]) {
        match self {
            FloatCodec::Bf16 => {
                out[..2].copy_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes())
            }
            FloatCodec::F16 => out[..2].copy_from_slice(&f16::from_f32(v).to_bits().to_le_bytes()),
            FloatCodec::F32 => out[..4].copy_from_slice(&v.to_le_bytes()),
        }
    }
}

fn is_float(dtype: DType) -> bool {
    FloatCodec::of(dtype).is_some()
}

fn invalid(message: String) -> KernelError {
    KernelError::InvalidArgument { message }
}

/// Byte offsets of every logical element of `v` in row-major order, after checking that the
/// view's strides stay inside its slice.
fn element_offsets(v: &TensorView<'_>) -> Result<Vec<usize>, KernelError> {
    if v.strides.len() != v.shape.len() {
        return Err(invalid(format!(
            "view has shape {:?} but strides {:?}",
            v.shape.as_slice(),
            v.strides.as_slice()
        )));
    }
    let es = v.dtype.size_bytes();
    let numel = v.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }
    let last: usize = v
        .shape
        .iter()
        .zip(&v.strides)
        .map(|(&d, &s)| (d - 1) * s)
        .sum();
    if (last + 1) * es > v.slice.len() {
        return Err(invalid(format!(
            "view of shape {:?} strides {:?} ({}) addresses {} bytes but its slice holds {}",
            v.shape.as_slice(),
            v.strides.as_slice(),
            v.dtype.as_str(),
            (last + 1) * es,
            v.slice.len()
        )));
    }
    let rank = v.shape.len();
    let mut offsets = Vec::with_capacity(numel);
    let mut idx = vec![0usize; rank];
    let mut elem = 0usize;
    for _ in 0..numel {
        offsets.push(elem * es);
        // Odometer increment of the multi-index, keeping `elem` = Σ idx·stride.
        for dim in (0..rank).rev() {
            idx[dim] += 1;
            elem += v.strides[dim];
            if idx[dim] < v.shape[dim] {
                break;
            }
            elem -= v.shape[dim] * v.strides[dim];
            idx[dim] = 0;
        }
    }
    Ok(offsets)
}

/// Reads a bf16/f16/f32 view as logical row-major f32 values.
pub(crate) fn load(v: &TensorView<'_>) -> Result<Vec<f32>, KernelError> {
    let codec = FloatCodec::require(v.dtype)?;
    let offsets = element_offsets(v)?;
    let bytes = v.slice.read_bytes()?;
    Ok(offsets.iter().map(|&o| codec.decode(&bytes[o..])).collect())
}

/// Reads an I32 view exactly.
pub(crate) fn load_i32(v: &TensorView<'_>) -> Result<Vec<i32>, KernelError> {
    if v.dtype != DType::I32 {
        return Err(invalid(format!(
            "expected an i32 view, got {}",
            v.dtype.as_str()
        )));
    }
    let offsets = element_offsets(v)?;
    let bytes = v.slice.read_bytes()?;
    Ok(offsets
        .iter()
        .map(|&o| i32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]))
        .collect())
}

/// Writes logical row-major `values` into a bf16/f16/f32 view, rounding to its dtype. Bytes of
/// the slice that the view does not address are preserved.
pub(crate) fn store(v: &TensorView<'_>, values: &[f32]) -> Result<(), KernelError> {
    let codec = FloatCodec::require(v.dtype)?;
    let offsets = element_offsets(v)?;
    if values.len() != offsets.len() {
        return Err(invalid(format!(
            "{} values for a view of {} elements",
            values.len(),
            offsets.len()
        )));
    }
    let mut bytes = v.slice.read_bytes()?;
    for (&o, &value) in offsets.iter().zip(values) {
        codec.encode(value, &mut bytes[o..]);
    }
    v.slice.write_bytes(&bytes)?;
    Ok(())
}

fn expect_shape(name: &str, v: &TensorView<'_>, want: &[usize]) -> Result<(), KernelError> {
    if v.shape.as_slice() == want {
        Ok(())
    } else {
        Err(invalid(format!(
            "{name} has shape {:?}, expected {want:?}",
            v.shape.as_slice()
        )))
    }
}

fn expect_rank(name: &str, v: &TensorView<'_>, rank: usize) -> Result<(), KernelError> {
    if v.shape.len() == rank {
        Ok(())
    } else {
        Err(invalid(format!(
            "{name} has shape {:?}, expected rank {rank}",
            v.shape.as_slice()
        )))
    }
}

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

impl AttentionKernel for CpuReference {
    fn supports(&self, cfg: &AttentionConfig) -> bool {
        cfg.num_kv_heads > 0
            && cfg.num_q_heads.is_multiple_of(cfg.num_kv_heads)
            && cfg.head_dim > 0
            && is_float(cfg.dtype)
            && cfg.block_tokens.is_none()
    }

    fn implementation(&self, _cfg: &AttentionConfig) -> String {
        "cpu_attention_f32acc".into()
    }

    fn execute(&self, ctx: &mut AttentionContext<'_>) -> Result<(), KernelError> {
        if !AttentionKernel::supports(self, &ctx.cfg) {
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
        store(&ctx.out, &math::attention(&q, &k, &v, &shape, ctx.scale))
    }
}

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

impl EmbeddingKernel for CpuReference {
    fn supports(&self, cfg: &EmbeddingConfig) -> bool {
        is_float(cfg.dtype)
    }

    fn implementation(&self, _cfg: &EmbeddingConfig) -> String {
        "cpu_embedding".into()
    }

    fn execute(&self, ctx: &mut EmbeddingContext<'_>) -> Result<(), KernelError> {
        expect_rank("ids", &ctx.ids, 1)?;
        expect_rank("table", &ctx.table, 2)?;
        let tokens = ctx.ids.shape[0];
        let (vocab_rows, hidden) = (ctx.table.shape[0], ctx.table.shape[1]);
        expect_shape("out", &ctx.out, &[tokens, hidden])?;
        let ids = load_i32(&ctx.ids)?;
        let table = load(&ctx.table)?;
        let mut out = vec![0f32; tokens * hidden];
        for (dst, &id) in out.chunks_exact_mut(hidden.max(1)).zip(&ids) {
            let row = i64::from(id) - ctx.vocab_offset;
            if let Ok(row) = usize::try_from(row)
                && row < vocab_rows
            {
                dst.copy_from_slice(&table[row * hidden..(row + 1) * hidden]);
            }
        }
        store(&ctx.out, &out)
    }
}

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

impl KernelProvider for CpuReference {
    fn id(&self) -> ProviderId {
        ProviderId("cpu-reference")
    }
    fn gemm(&self) -> Option<&dyn GemmKernel> {
        Some(self)
    }
    fn attention(&self) -> Option<&dyn AttentionKernel> {
        Some(self)
    }
    fn norm(&self) -> Option<&dyn NormKernel> {
        Some(self)
    }
    fn rope(&self) -> Option<&dyn RopeKernel> {
        Some(self)
    }
    fn activation(&self) -> Option<&dyn ActivationKernel> {
        Some(self)
    }
    fn embedding(&self) -> Option<&dyn EmbeddingKernel> {
        Some(self)
    }
    fn elementwise(&self) -> Option<&dyn ElementwiseKernel> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use turbine_core::types::{DType, DeviceId};
    use turbine_tensor::host::HostMemory;
    use turbine_tensor::{DeviceBuffer, DeviceMemory, Tensor, TensorView};

    use super::*;
    use crate::ops::{AttentionConfig, AttentionContext, AttentionKind, GemmConfig, GemmContext};

    fn host() -> Arc<dyn DeviceMemory> {
        HostMemory::new(DeviceId(0), 1 << 20)
    }

    /// A contiguous tensor holding `values` rounded to `dtype`.
    fn tensor(
        mem: &Arc<dyn DeviceMemory>,
        shape: &[usize],
        dtype: DType,
        values: &[f32],
    ) -> Tensor {
        let t = Tensor::empty(mem, shape, dtype).expect("alloc");
        store(&t.view(), values).expect("store");
        t
    }

    fn i32_tensor(mem: &Arc<dyn DeviceMemory>, values: &[i32]) -> Tensor {
        let t = Tensor::empty(mem, &[values.len()], DType::I32).expect("alloc");
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        t.storage.whole().write_bytes(&bytes).expect("write");
        t
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn assert_close(got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len(), "got {got:?}, want {want:?}");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= tol,
                "element {i}: got {g}, want {w} (all: {got:?})"
            );
        }
    }

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

    #[test]
    fn strided_store_preserves_the_bytes_between_rows() {
        let mem = host();
        let buf = DeviceBuffer::alloc(&mem, 4 * 4 * 4).expect("alloc");
        let sentinel = vec![0xAB; buf.len()];
        buf.whole().write_bytes(&sentinel).expect("fill");
        // [4, 2] F32 view with a row stride of 4 elements: columns 2..4 of each row are padding.
        let view = TensorView {
            slice: buf.whole(),
            shape: (&[4usize, 2][..]).into(),
            strides: (&[4usize, 1][..]).into(),
            dtype: DType::F32,
        };
        let rows = view.rows(1, 2);
        store(&rows, &[1.0, 2.0, 3.0, 4.0]).expect("store");
        assert_eq!(load(&rows).expect("load"), [1.0, 2.0, 3.0, 4.0]);
        let bytes = buf.whole().read_bytes().expect("read");
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            let (row, col) = (i / 4, i % 4);
            if (1..3).contains(&row) && col < 2 {
                continue;
            }
            assert_eq!(chunk, [0xAB; 4], "row {row} col {col} was overwritten");
        }
    }

    #[test]
    fn bad_shapes_and_dtypes_are_errors() {
        let mem = host();
        let a = tensor(&mem, &[2, 3], DType::F32, &[0.0; 6]);
        let c = Tensor::empty(&mem, &[2, 2], DType::F32).expect("alloc");
        let err = CpuReference
            .gemm()
            .expect("gemm family")
            .execute(&mut GemmContext {
                a: a.view(),
                b: a.view(),
                c: c.view(),
                trans_b: false,
                alpha: 1.0,
                beta: 0.0,
            })
            .expect_err("b must be [k, n]");
        assert!(matches!(err, KernelError::InvalidArgument { .. }), "{err}");
        assert_eq!(
            err.to_string(),
            "invalid argument: b has shape [2, 3], expected [3, 2]"
        );

        let ints = i32_tensor(&mem, &[1, 2]);
        assert!(matches!(
            load(&ints.view()),
            Err(KernelError::Unsupported { .. })
        ));
        assert!(!ElementwiseKernel::supports(
            &CpuReference,
            &ElementwiseConfig { dtype: DType::I64 }
        ));
        assert!(!AttentionKernel::supports(
            &CpuReference,
            &AttentionConfig {
                kind: AttentionKind::Prefill,
                num_q_heads: 3,
                num_kv_heads: 2,
                head_dim: 2,
                dtype: DType::BF16,
                block_tokens: None,
                causal: true,
            }
        ));
    }

    #[test]
    fn round_to_matches_the_dtype_precision() {
        let v = 1.0 + 2f32.powi(-10);
        assert_eq!(round_to(DType::BF16, v), 1.0);
        assert_eq!(round_to(DType::F16, v), v);
        assert_eq!(round_to(DType::F32, 0.1), 0.1);
    }
}
