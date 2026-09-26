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
    ActivationConfig, ActivationContext, ActivationKernel, AddRmsnormConfig, AddRmsnormContext,
    AddRmsnormKernel, AttentionConfig, AttentionContext, AttentionKernel, ElementwiseConfig,
    ElementwiseContext, ElementwiseKernel, EmbeddingConfig, EmbeddingContext, EmbeddingKernel,
    GemmConfig, GemmContext, GemmKernel, KernelProvider, KvCopyKernel, LogitsReduceConfig,
    LogitsReduceContext, LogitsReduceKernel, MoeKernel, NormConfig, NormContext, NormKernel,
    PagedAttentionContext, ProviderId, RopeConfig, RopeContext, RopeKernel,
};

mod math;
mod moe;
mod paged;

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

/// Writes logical row-major `values` into an I32 view. Bytes of the slice that the view does not
/// address are preserved.
pub(crate) fn store_i32(v: &TensorView<'_>, values: &[i32]) -> Result<(), KernelError> {
    if v.dtype != DType::I32 {
        return Err(invalid(format!(
            "expected an i32 view, got {}",
            v.dtype.as_str()
        )));
    }
    let offsets = element_offsets(v)?;
    if values.len() != offsets.len() {
        return Err(invalid(format!(
            "{} values for a view of {} elements",
            values.len(),
            offsets.len()
        )));
    }
    let mut bytes = v.slice.read_bytes()?;
    for (&o, value) in offsets.iter().zip(values) {
        bytes[o..o + 4].copy_from_slice(&value.to_le_bytes());
    }
    v.slice.write_bytes(&bytes)?;
    Ok(())
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
            && if cfg.kind.is_paged() {
                cfg.block_tokens.is_some_and(|b| b > 0)
            } else {
                cfg.block_tokens.is_none()
            }
    }

    fn implementation(&self, cfg: &AttentionConfig) -> String {
        if cfg.kind.is_paged() {
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

/// Checks that `v` is a dense `dtype` view of rank `1 + cols.is_some()` with at least `rows` rows
/// (and exactly `cols` columns), returning its first `rows` rows.
fn leading_rows<'a>(
    name: &str,
    v: &TensorView<'a>,
    rows: usize,
    cols: Option<usize>,
    dtype: DType,
) -> Result<TensorView<'a>, KernelError> {
    let rank = 1 + usize::from(cols.is_some());
    let dense = turbine_tensor::tensor::contiguous_strides(&v.shape);
    if v.dtype != dtype
        || v.shape.len() != rank
        || v.shape[0] < rows
        || cols.is_some_and(|c| v.shape[1] != c)
        || v.strides != dense
    {
        return Err(invalid(format!(
            "{name} must be a dense {} view of at least {rows} rows{}, has {} shape {:?} strides {:?}",
            dtype.as_str(),
            cols.map_or(String::new(), |c| format!(" of {c} columns")),
            v.dtype.as_str(),
            v.shape.as_slice(),
            v.strides.as_slice()
        )));
    }
    Ok(v.rows(0, rows))
}

impl LogitsReduceKernel for CpuReference {
    fn supports(&self, cfg: &LogitsReduceConfig) -> bool {
        cfg.vocab > 0 && cfg.top_n <= LogitsReduceConfig::MAX_TOP_N && cfg.top_n <= cfg.vocab
    }

    fn implementation(&self, _cfg: &LogitsReduceConfig) -> String {
        "cpu_logits_reduce".into()
    }

    fn execute(&self, ctx: &mut LogitsReduceContext<'_>) -> Result<(), KernelError> {
        let rows = ctx.rows as usize;
        expect_rank("logits", &ctx.logits, 2)?;
        expect_rank("top_ids", &ctx.top_ids, 2)?;
        let vocab = ctx.logits.shape[1];
        let top_n = ctx.top_ids.shape[1];
        let cfg = LogitsReduceConfig {
            vocab: u32::try_from(vocab).map_err(|_| invalid(format!("vocab {vocab}")))?,
            top_n: u32::try_from(top_n).map_err(|_| invalid(format!("top_n {top_n}")))?,
        };
        if !LogitsReduceKernel::supports(self, &cfg) {
            return Err(KernelError::Unsupported {
                message: format!("cpu-reference logits_reduce {cfg}"),
            });
        }
        if ctx.logits.dtype != DType::F32 || ctx.logits.shape[0] < rows {
            return Err(invalid(format!(
                "logits must be an f32 view of at least {rows} rows, has {} shape {:?}",
                ctx.logits.dtype.as_str(),
                ctx.logits.shape.as_slice()
            )));
        }
        let f32_rows = |name, v| leading_rows(name, v, rows, None, DType::F32);
        let temperature = load(&f32_rows("temperature", &ctx.temperature)?)?;
        let uniform = load(&f32_rows("uniform", &ctx.uniform)?)?;
        let top_p = load(&f32_rows("top_p", &ctx.top_p)?)?;
        let mode = load_i32(&leading_rows("mode", &ctx.mode, rows, None, DType::I32)?)?;
        let top_ids_view = leading_rows("top_ids", &ctx.top_ids, rows, Some(top_n), DType::I32)?;
        let top_values_view =
            leading_rows("top_values", &ctx.top_values, rows, Some(top_n), DType::F32)?;
        let lse_view = f32_rows("lse", &ctx.lse)?;
        let sampled_view = leading_rows("sampled", &ctx.sampled, rows, None, DType::I32)?;
        let sampled_logit_view = f32_rows("sampled_logit", &ctx.sampled_logit)?;
        let logits = load(&ctx.logits.rows(0, rows))?;

        let mut top_ids = Vec::with_capacity(rows * top_n);
        let mut top_values = Vec::with_capacity(rows * top_n);
        let mut lse = Vec::with_capacity(rows);
        let mut sampled = Vec::with_capacity(rows);
        let mut sampled_logit = Vec::with_capacity(rows);
        for (r, row) in logits.chunks_exact(vocab).enumerate() {
            for (id, value) in math::top_n(row, top_n) {
                top_ids.push(id as i32);
                top_values.push(value);
            }
            lse.push(math::log_sum_exp(row));
            match mode[r] {
                0 => {
                    sampled.push(-1);
                    sampled_logit.push(f32::NAN);
                }
                1 => {
                    let id = if top_p[r] < 1.0 {
                        math::nucleus(row, temperature[r], top_p[r], uniform[r])
                    } else {
                        math::categorical(row, temperature[r], uniform[r])
                    };
                    sampled.push(id as i32);
                    sampled_logit.push(row[id as usize]);
                }
                other => {
                    return Err(invalid(format!("mode[{r}] = {other}, expected 0 or 1")));
                }
            }
        }
        store_i32(&top_ids_view, &top_ids)?;
        store(&top_values_view, &top_values)?;
        store(&lse_view, &lse)?;
        store_i32(&sampled_view, &sampled)?;
        store(&sampled_logit_view, &sampled_logit)
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
    fn kv_copy(&self) -> Option<&dyn KvCopyKernel> {
        Some(self)
    }
    fn moe(&self) -> Option<&dyn MoeKernel> {
        Some(self)
    }
    fn add_rmsnorm(&self) -> Option<&dyn AddRmsnormKernel> {
        Some(self)
    }
    fn logits_reduce(&self) -> Option<&dyn LogitsReduceKernel> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use turbine_core::types::{DType, DeviceId};
    use turbine_tensor::host::HostMemory;
    use turbine_tensor::{DeviceBuffer, DeviceMemory, Tensor, TensorView};

    use turbine_core::types::BlockId;

    use super::*;
    use crate::ops::{
        AddRmsnormConfig, AddRmsnormContext, AttentionConfig, AttentionContext, AttentionKind,
        GemmConfig, GemmContext, KvCopyConfig, KvCopyContext, LogitsReduceConfig,
        LogitsReduceContext, MoeExpertsConfig, MoeExpertsContext, MoeRouteConfig, MoeRouteContext,
        PagedAttentionContext,
    };

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

    /// Deterministic values in [-1, 1) (64-bit LCG), so the test needs no RNG crate.
    fn seeded(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn i32_tensor_2d(mem: &Arc<dyn DeviceMemory>, rows: usize, values: &[i32]) -> Tensor {
        let t = Tensor::empty(mem, &[rows, values.len() / rows], DType::I32).expect("alloc");
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        t.storage.whole().write_bytes(&bytes).expect("write");
        t
    }

    #[test]
    fn paged_attention_equals_contiguous_and_moe_route_ties() {
        let mem = HostMemory::new(DeviceId(0), 1 << 22) as Arc<dyn DeviceMemory>;
        let cpu = cpu_reference_provider();
        let (hq, hkv, d, bt) = (4usize, 2usize, 8usize, 16usize);
        let dtype = DType::BF16;
        let q_lens = [5usize, 1, 3];
        let kv_lens = [20usize, 9, 3];
        // Shuffled block table over an 8-block pool; unused entries are -1 and must not be read.
        let table = [5, 2, 7, -1, 0, -1];
        let num_blocks = 8;
        let paged_cfg = AttentionConfig {
            kind: AttentionKind::PrefillPaged,
            num_q_heads: hq as u32,
            num_kv_heads: hkv as u32,
            head_dim: d as u32,
            dtype,
            block_tokens: Some(bt as u32),
            causal: true,
        };
        let attn = cpu.attention().expect("attention family");
        assert!(attn.supports(&paged_cfg));
        assert_eq!(
            attn.implementation(&paged_cfg),
            "cpu_attention_paged_f32acc"
        );

        // Every sequence's full K/V history and its new queries.
        let full_k: Vec<Vec<f32>> = (0..3)
            .map(|s| seeded(10 + s as u64, kv_lens[s] * hkv * d))
            .collect();
        let full_v: Vec<Vec<f32>> = (0..3)
            .map(|s| seeded(20 + s as u64, kv_lens[s] * hkv * d))
            .collect();
        let qs: Vec<Vec<f32>> = (0..3)
            .map(|s| seeded(30 + s as u64, q_lens[s] * hq * d))
            .collect();

        // The pool of this layer holds the tokens before this step (kv_len − q_len of them).
        let pool = Tensor::empty(&mem, &[num_blocks, 2, bt, hkv, d], dtype).expect("pool");
        let mut pool_values = vec![0f32; num_blocks * 2 * bt * hkv * d];
        let token = hkv * d;
        for s in 0..3 {
            for p in 0..kv_lens[s] - q_lens[s] {
                let block = table[s * 2 + p / bt] as usize;
                let k_at = ((block * 2) * bt + p % bt) * token;
                let v_at = ((block * 2 + 1) * bt + p % bt) * token;
                pool_values[k_at..k_at + token]
                    .copy_from_slice(&full_k[s][p * token..(p + 1) * token]);
                pool_values[v_at..v_at + token]
                    .copy_from_slice(&full_v[s][p * token..(p + 1) * token]);
            }
        }
        store(&pool.view(), &pool_values).expect("fill pool");

        let total_q: usize = q_lens.iter().sum();
        let mut q_all = Vec::new();
        let (mut k_new, mut v_new) = (Vec::new(), Vec::new());
        for s in 0..3 {
            q_all.extend_from_slice(&qs[s]);
            let first_new = (kv_lens[s] - q_lens[s]) * token;
            k_new.extend_from_slice(&full_k[s][first_new..]);
            v_new.extend_from_slice(&full_v[s][first_new..]);
        }
        let q = tensor(&mem, &[total_q, hq, d], dtype, &q_all);
        let kn = tensor(&mem, &[total_q, hkv, d], dtype, &k_new);
        let vn = tensor(&mem, &[total_q, hkv, d], dtype, &v_new);
        let out = Tensor::empty(&mem, &[total_q, hq, d], dtype).expect("out");
        let block_table = i32_tensor_2d(&mem, 3, &table);
        let q_indptr = i32_tensor(&mem, &[0, 5, 6, 9]);
        let kv_lens_t = i32_tensor(&mem, &[20, 9, 3]);
        attn.execute_paged(&mut PagedAttentionContext {
            cfg: paged_cfg,
            q: q.view(),
            k_new: kn.view(),
            v_new: vn.view(),
            out: out.view(),
            kv_layer: pool.view(),
            block_table: block_table.view(),
            q_indptr: q_indptr.view(),
            kv_lens: kv_lens_t.view(),
            max_q_len: 5,
            max_kv_len: 20,
            max_blocks_per_seq: 2,
            scale: 1.0 / (d as f32).sqrt(),
        })
        .expect("paged attention");
        let paged_out = load(&out.view()).expect("load");

        // Per sequence, the contiguous op over the same history gives the same rows.
        let contiguous_cfg = AttentionConfig {
            kind: AttentionKind::Prefill,
            block_tokens: None,
            ..paged_cfg
        };
        let mut row = 0;
        for s in 0..3 {
            let k = tensor(&mem, &[kv_lens[s], hkv, d], dtype, &full_k[s]);
            let v = tensor(&mem, &[kv_lens[s], hkv, d], dtype, &full_v[s]);
            let qv = tensor(&mem, &[q_lens[s], hq, d], dtype, &qs[s]);
            let o = Tensor::empty(&mem, &[q_lens[s], hq, d], dtype).expect("out");
            attn.execute(&mut AttentionContext {
                cfg: contiguous_cfg,
                q: qv.view(),
                k_cache: k.view(),
                v_cache: v.view(),
                out: o.view(),
                q_start: (kv_lens[s] - q_lens[s]) as u32,
                scale: 1.0 / (d as f32).sqrt(),
            })
            .expect("contiguous attention");
            let want = load(&o.view()).expect("load");
            let n = q_lens[s] * hq * d;
            assert_eq!(&paged_out[row..row + n], want.as_slice(), "sequence {s}");
            row += n;
        }
        // The new K/V rows were appended into their page slots: sequence 0, token 17 → block 2,
        // slot 1; sequence 2, token 0 → block 0, slot 0.
        let pool_after = load(&pool.view()).expect("pool");
        let k_slot = |block: usize, slot: usize| {
            let at = (block * 2 * bt + slot) * token;
            pool_after[at..at + token].to_vec()
        };
        let v_slot = |block: usize, slot: usize| {
            let at = ((block * 2 + 1) * bt + slot) * token;
            pool_after[at..at + token].to_vec()
        };
        let rounded = |v: &[f32]| v.iter().map(|&x| round_to(dtype, x)).collect::<Vec<_>>();
        assert_eq!(k_slot(2, 1), rounded(&full_k[0][17 * token..18 * token]));
        assert_eq!(v_slot(2, 1), rounded(&full_v[0][17 * token..18 * token]));
        assert_eq!(k_slot(0, 0), rounded(&full_k[2][..token]));

        // moe_route: logits [1, 1, 0, 0] top-2 → experts 0 then 1 with unrenormalised softmax
        // weights; [0, 0, 0, 2] → expert 3, then the lowest id of the tied rest (0).
        let route_cfg = MoeRouteConfig {
            num_experts: 4,
            top_k: 2,
            renormalize: false,
        };
        let moe = cpu.moe().expect("moe family");
        assert!(moe.supports_route(&route_cfg));
        assert_eq!(moe.implementation_route(&route_cfg), "cpu_moe_route");
        let logits = tensor(
            &mem,
            &[2, 4],
            DType::F32,
            &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        );
        let topk_ids = Tensor::empty(&mem, &[2, 2], DType::I32).expect("ids");
        let topk_weights = Tensor::empty(&mem, &[2, 2], DType::F32).expect("weights");
        let sorted_rows = Tensor::empty(&mem, &[4], DType::I32).expect("rows");
        let expert_offsets = Tensor::empty(&mem, &[5], DType::I32).expect("offsets");
        moe.route(&mut MoeRouteContext {
            cfg: route_cfg,
            router_logits: logits.view(),
            topk_ids: topk_ids.view(),
            topk_weights: topk_weights.view(),
            sorted_rows: sorted_rows.view(),
            expert_offsets: expert_offsets.view(),
        })
        .expect("route");
        assert_eq!(load_i32(&topk_ids.view()).expect("ids"), [0, 1, 3, 0]);
        let e = std::f32::consts::E;
        let (e2, sum1) = (e * e, 2.0 * e + 2.0);
        assert_close(
            &load(&topk_weights.view()).expect("weights"),
            &[e / sum1, e / sum1, e2 / (e2 + 3.0), 1.0 / (e2 + 3.0)],
            1e-6,
        );
        // Rows token·top_k + slot grouped by expert: e0 {0, 3}, e1 {1}, e2 {}, e3 {2}.
        assert_eq!(load_i32(&sorted_rows.view()).expect("rows"), [0, 3, 1, 2]);
        assert_eq!(
            load_i32(&expert_offsets.view()).expect("offsets"),
            [0, 2, 3, 3, 4]
        );

        // copy_blocks duplicates block 3 into block 7 on every layer and nothing else.
        let (layers, blocks, block_bytes) = (3usize, 8usize, 24usize);
        let buf = DeviceBuffer::alloc(&mem, layers * blocks * block_bytes).expect("pool");
        let before: Vec<u8> = (0..buf.len()).map(|i| (i % 251) as u8).collect();
        buf.whole().write_bytes(&before).expect("fill");
        let copy_cfg = KvCopyConfig {
            num_layers: layers as u32,
            block_bytes: block_bytes as u64,
        };
        let kv_copy = cpu.kv_copy().expect("kv_copy family");
        assert!(kv_copy.supports(&copy_cfg));
        assert_eq!(kv_copy.implementation(&copy_cfg), "cpu_copy_blocks");
        kv_copy
            .execute(&mut KvCopyContext {
                pool: buf.whole(),
                layer_stride_bytes: (blocks * block_bytes) as u64,
                block_bytes: block_bytes as u64,
                num_layers: layers as u32,
                pairs: &[(BlockId(3), BlockId(7))],
            })
            .expect("copy_blocks");
        let after = buf.whole().read_bytes().expect("read");
        let mut want = before.clone();
        for l in 0..layers {
            let src = (l * blocks + 3) * block_bytes;
            let dst = (l * blocks + 7) * block_bytes;
            want.copy_within(src..src + block_bytes, dst);
        }
        assert_eq!(after, want);
        assert_ne!(after, before);
    }

    #[test]
    fn moe_experts_accumulate_weighted_expert_outputs() {
        let mem = host();
        let moe = CpuReference.moe().expect("moe family");
        let (h, inter, experts, top_k) = (3usize, 2usize, 3usize, 2usize);
        // Two tokens: token 0 → experts (2, 0), token 1 → experts (0, 1); weights per slot.
        let sorted = [1, 2, 3, 0]; // e0 {1, 2}, e1 {3}, e2 {0}
        let offsets = [0, 2, 3, 4];
        let weights = [0.5f32, 0.25, 0.75, 0.125];
        let x_vals = [1.0f32, -2.0, 0.5, 0.25, 1.0, -1.0];
        let wg = seeded(1, experts * inter * h);
        let wu = seeded(2, experts * inter * h);
        let wd = seeded(3, experts * h * inter);
        let cfg = MoeExpertsConfig {
            hidden: h as u32,
            inter: inter as u32,
            num_experts: experts as u32,
            top_k: top_k as u32,
            expert_begin: 0,
            expert_end: experts as u32,
            dtype: DType::F32,
        };
        assert!(moe.supports_experts(&cfg));
        assert_eq!(moe.implementation_experts(&cfg), "cpu_moe_experts_f32acc");
        let x = tensor(&mem, &[2, h], DType::F32, &x_vals);
        let out = tensor(&mem, &[2, h], DType::F32, &[10.0; 6]);
        let w_gate = tensor(&mem, &[experts, inter, h], DType::F32, &wg);
        let w_up = tensor(&mem, &[experts, inter, h], DType::F32, &wu);
        let w_down = tensor(&mem, &[experts, h, inter], DType::F32, &wd);
        let sorted_rows = i32_tensor(&mem, &sorted);
        let expert_offsets = i32_tensor(&mem, &offsets);
        let topk_weights = tensor(&mem, &[2, top_k], DType::F32, &weights);
        moe.experts(&mut MoeExpertsContext {
            cfg,
            x: x.view(),
            w_gate: w_gate.view(),
            w_up: w_up.view(),
            w_down: w_down.view(),
            sorted_rows: sorted_rows.view(),
            expert_offsets: expert_offsets.view(),
            topk_weights: topk_weights.view(),
            host_expert_offsets: &offsets,
            out: out.view(),
            workspace: None,
        })
        .expect("experts");

        // Naive: out[t] = 10 + Σ over experts in ascending id of w · expert_e(x[t]).
        let expert = |e: usize, xt: &[f32]| -> Vec<f32> {
            let act: Vec<f32> = (0..inter)
                .map(|j| {
                    let dot = |w: &[f32]| (0..h).map(|c| xt[c] * w[(e * inter + j) * h + c]).sum();
                    let (g, u): (f32, f32) = (dot(&wg), dot(&wu));
                    g / (1.0 + (-g).exp()) * u
                })
                .collect();
            (0..h)
                .map(|c| {
                    (0..inter)
                        .map(|j| act[j] * wd[(e * h + c) * inter + j])
                        .sum()
                })
                .collect()
        };
        let routes = [[(2usize, 0.5f32), (0, 0.25)], [(0, 0.75), (1, 0.125)]];
        let mut want = Vec::new();
        for (t, route) in routes.iter().enumerate() {
            let xt = &x_vals[t * h..(t + 1) * h];
            let mut acc = [10.0f32; 3];
            let mut by_expert = route.to_vec();
            by_expert.sort_by_key(|&(e, _)| e);
            for (e, w) in by_expert {
                for (a, y) in acc.iter_mut().zip(expert(e, xt)) {
                    *a += w * y;
                }
            }
            want.extend_from_slice(&acc);
        }
        assert_close(&load(&out.view()).expect("out"), &want, 1e-5);

        // The host copy of the offsets must agree with the device offsets.
        let err = moe
            .experts(&mut MoeExpertsContext {
                cfg,
                x: x.view(),
                w_gate: w_gate.view(),
                w_up: w_up.view(),
                w_down: w_down.view(),
                sorted_rows: sorted_rows.view(),
                expert_offsets: expert_offsets.view(),
                topk_weights: topk_weights.view(),
                host_expert_offsets: &[0, 1, 3, 4],
                out: out.view(),
                workspace: None,
            })
            .expect_err("mismatched host offsets");
        assert!(matches!(err, KernelError::InvalidArgument { .. }), "{err}");

        // The reference reads the offsets from `expert_offsets` itself: it never needs the host
        // copy, and an empty one gives the same result as the full one.
        for rows in [0, 8, 512, 513, 1 << 20] {
            assert!(!moe.needs_host_offsets(&cfg, rows), "{rows} routed rows");
        }
        assert_eq!(cfg.routed_rows(3), 3 * top_k);
        let with_host = load(&out.view()).expect("out");
        let again = tensor(&mem, &[2, h], DType::F32, &[10.0; 6]);
        moe.experts(&mut MoeExpertsContext {
            cfg,
            x: x.view(),
            w_gate: w_gate.view(),
            w_up: w_up.view(),
            w_down: w_down.view(),
            sorted_rows: sorted_rows.view(),
            expert_offsets: expert_offsets.view(),
            topk_weights: topk_weights.view(),
            host_expert_offsets: &[],
            out: again.view(),
            workspace: None,
        })
        .expect("experts without host offsets");
        assert_eq!(load(&again.view()).expect("out"), with_host);
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

    /// The host sampler's candidate order: descending, ties to the lower id, NaN last (a full
    /// sort, independent of the provider's partial selection).
    fn reference_top(row: &[f32], n: usize) -> Vec<(u32, f32)> {
        let mut pairs: Vec<(u32, f32)> = (0..row.len() as u32).zip(row.iter().copied()).collect();
        pairs.sort_by(|a, b| match (a.1.is_nan(), b.1.is_nan()) {
            (true, true) => a.0.cmp(&b.0),
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)),
        });
        pairs.truncate(n);
        pairs
    }

    /// Log-sum-exp over the non-NaN values, entirely in f64.
    fn reference_lse(row: &[f32]) -> f64 {
        let finite = row.iter().filter(|v| !v.is_nan()).map(|&v| f64::from(v));
        let max = finite.clone().fold(f64::NEG_INFINITY, f64::max);
        max + finite.map(|v| (v - max).exp()).sum::<f64>().ln()
    }

    /// The host sampler's draw with `top_k` −1 and `top_p` 1 (candidates in id order): scaled
    /// logits `v · (1/T)`, weights `exp(scaled − max)` summed sequentially in f64, and the first id
    /// whose cumulative weight exceeds `u · total`.
    fn reference_draw(row: &[f32], temperature: f32, u: f32) -> u32 {
        let inv_t = 1.0 / temperature;
        let scaled: Vec<f32> = row.iter().map(|&v| v * inv_t).collect();
        let max = scaled
            .iter()
            .copied()
            .filter(|v| !v.is_nan())
            .fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f64> = scaled
            .iter()
            .map(|&s| {
                if s.is_nan() {
                    0.0
                } else {
                    f64::from(s - max).exp()
                }
            })
            .collect();
        let total: f64 = weights.iter().sum();
        let target = f64::from(u) * total;
        let mut cum = 0.0;
        for (id, w) in weights.iter().enumerate() {
            cum += w;
            if target < cum {
                return id as u32;
            }
        }
        weights.iter().rposition(|&w| w > 0.0).unwrap_or(0) as u32
    }

    /// On seeded rows of both model vocabularies (tied maxima, a tied run inside the top-20, NaN
    /// and −∞ entries, a padded row stride) the reference `logits_reduce` returns the sampler's
    /// top-20 in order with the raw values, its log-sum-exp within 1e-6 (relative) and, in
    /// categorical mode at temperature 0.7 with ChaCha8 uniforms, the sampler's inverse-CDF token;
    /// rows with `top_p` < 1 (a broad row at 0.9 and 0.5, a peaked one whose nucleus ends inside
    /// the tied run) return the sampler's top-p token. Breaks if ties go to the higher id, NaN
    /// ranks first, the lse ignores part of the row, the draw is taken in another order or at
    /// another temperature, or the nucleus is cut at another mass or drawn in id order.
    #[test]
    fn logits_reduce_matches_sampler() {
        use rand_chacha::ChaCha8Rng;
        use rand_chacha::rand_core::{RngCore, SeedableRng};

        let mem = HostMemory::new(DeviceId(0), 1 << 24) as Arc<dyn DeviceMemory>;
        let reduce = CpuReference.logits_reduce().expect("logits_reduce family");
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        let mut uniform = || (rng.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
        let (rows, top_n, pad) = (7usize, 20usize, 5usize);
        // Row 0 reduces only; rows 1..7 also draw at temperature 0.7 (row 6 at 0.4), rows 4..7
        // from a nucleus.
        let modes = [0, 1, 1, 1, 1, 1, 1];
        let temperatures = [0.0f32, 0.7, 0.7, 0.7, 0.7, 0.7, 0.4];
        let top_ps = [1.0f32, 1.0, 1.0, 1.0, 0.9, 0.5, 0.8];
        let mut draws_below_the_top = 0;
        let mut nucleus_draws_below_the_top = 0;
        for vocab in [50_304usize, 128_256] {
            let cfg = LogitsReduceConfig {
                vocab: vocab as u32,
                top_n: top_n as u32,
            };
            assert!(reduce.supports(&cfg));
            assert_eq!(reduce.implementation(&cfg), "cpu_logits_reduce");
            assert!(!reduce.supports(&LogitsReduceConfig {
                vocab: 100,
                top_n: 65
            }));

            let mut logits = Vec::with_capacity(rows);
            for r in 0..rows {
                // Row 6's bulk is narrow (±2), the others' ±8.
                let spread = if r == 6 { 2.0 } else { 8.0 };
                let mut row: Vec<f32> = seeded((vocab + r) as u64, vocab)
                    .iter()
                    .map(|v| v * spread)
                    .collect();
                // Three tied maxima, a tied run just below them, NaN and −∞ entries.
                for id in [7, 1000 + r, vocab - 3] {
                    row[id] = 9.0;
                }
                for value in &mut row[100..110] {
                    *value = 8.5;
                }
                row[3] = f32::NAN;
                row[vocab / 2] = f32::NAN;
                row[11] = f32::NEG_INFINITY;
                if r == 6 {
                    // Peaked: the three maxima and the tied run hold nearly all the mass at
                    // T = 0.4, so the 0.8 nucleus ends inside the run (at its eighth id).
                    for value in &mut row[100..110] {
                        *value = 8.9;
                    }
                }
                logits.push(row);
            }
            // Logits rows padded to a stride of vocab + pad elements; the padding must never
            // be read (it would win every ranking).
            let padded = Tensor::empty(&mem, &[rows, vocab + pad], DType::F32).expect("logits");
            let flat: Vec<f32> = logits
                .iter()
                .flat_map(|row| row.iter().copied().chain([1e30; 5]))
                .collect();
            store(&padded.view(), &flat).expect("fill");
            let logits_view = TensorView {
                slice: padded.storage.whole(),
                shape: (&[rows, vocab][..]).into(),
                strides: (&[vocab + pad, 1][..]).into(),
                dtype: DType::F32,
            };
            let uniforms: Vec<f32> = (0..rows).map(|_| uniform()).collect();
            // Outputs hold one spare row, which the op must leave alone.
            let spare = rows + 1;
            let temperature = tensor(&mem, &[rows], DType::F32, &temperatures);
            let uniform_t = tensor(&mem, &[rows], DType::F32, &uniforms);
            let top_p = tensor(&mem, &[rows], DType::F32, &top_ps);
            let mode = i32_tensor(&mem, &modes);
            let top_ids = i32_tensor_2d(&mem, spare, &vec![-7; spare * top_n]);
            let top_values = Tensor::empty(&mem, &[spare, top_n], DType::F32).expect("values");
            let lse = Tensor::empty(&mem, &[spare], DType::F32).expect("lse");
            let sampled = i32_tensor(&mem, &vec![-7; spare]);
            let sampled_logit = Tensor::empty(&mem, &[spare], DType::F32).expect("logit");
            reduce
                .execute(&mut LogitsReduceContext {
                    logits: logits_view,
                    temperature: temperature.view(),
                    uniform: uniform_t.view(),
                    top_p: top_p.view(),
                    mode: mode.view(),
                    top_ids: top_ids.view(),
                    top_values: top_values.view(),
                    lse: lse.view(),
                    sampled: sampled.view(),
                    sampled_logit: sampled_logit.view(),
                    rows: rows as u32,
                })
                .expect("logits_reduce");

            let got_ids = load_i32(&top_ids.view()).expect("ids");
            let got_values = load(&top_values.view()).expect("values");
            let got_lse = load(&lse.view()).expect("lse");
            let got_sampled = load_i32(&sampled.view()).expect("sampled");
            let got_logit = load(&sampled_logit.view()).expect("logit");
            for (r, row) in logits.iter().enumerate() {
                let want = reference_top(row, top_n);
                let mut maxima = vec![7, 1000 + r as u32, vocab as u32 - 3];
                maxima.sort_unstable();
                let tied: Vec<u32> = want[..3].iter().map(|c| c.0).collect();
                assert_eq!(tied, maxima, "tied maxima rank by id");
                let ids = &got_ids[r * top_n..(r + 1) * top_n];
                let values = &got_values[r * top_n..(r + 1) * top_n];
                assert_eq!(
                    ids,
                    want.iter().map(|c| c.0 as i32).collect::<Vec<_>>(),
                    "vocab {vocab} row {r}"
                );
                for (got, want) in values.iter().zip(&want) {
                    assert_eq!(got.to_bits(), want.1.to_bits(), "vocab {vocab} row {r}");
                }
                let want_lse = reference_lse(row);
                assert!(
                    (f64::from(got_lse[r]) - want_lse).abs() <= 1e-6 * want_lse.abs().max(1.0),
                    "vocab {vocab} row {r}: lse {} vs {want_lse}",
                    got_lse[r]
                );
                if modes[r] == 0 {
                    assert_eq!(got_sampled[r], -1);
                    assert!(got_logit[r].is_nan());
                } else if top_ps[r] < 1.0 {
                    let want = reference_nucleus(row, temperatures[r], top_ps[r], uniforms[r]);
                    assert_eq!(got_sampled[r], want as i32, "vocab {vocab} row {r} (top_p)");
                    assert_eq!(got_logit[r].to_bits(), row[want as usize].to_bits());
                    if !maxima.contains(&want) {
                        nucleus_draws_below_the_top += 1;
                    }
                } else {
                    let want = reference_draw(row, temperatures[r], uniforms[r]);
                    assert_eq!(got_sampled[r], want as i32, "vocab {vocab} row {r}");
                    assert_eq!(got_logit[r].to_bits(), row[want as usize].to_bits());
                    if want != maxima[0] {
                        draws_below_the_top += 1;
                    }
                }
            }
            assert_eq!(&got_ids[rows * top_n..], [-7; 20], "spare row written");
            assert_eq!(got_sampled[rows], -7, "spare row written");
        }
        assert!(
            draws_below_the_top > 0,
            "every draw returned the argmax: the categorical path is not exercised"
        );
        assert!(
            nucleus_draws_below_the_top > 0,
            "every nucleus draw returned a maximum: the top-p path is not exercised"
        );
    }

    /// The host sampler's seeded top-p draw (`top_k` −1, `top_p` < 1), written out: all ids
    /// sorted descending (ties by id, NaN last), f64 weights summed in that order, the shortest
    /// prefix reaching `top_p · total`, then `u` × the prefix's sum scanned in the same order.
    fn reference_nucleus(row: &[f32], temperature: f32, top_p: f32, u: f32) -> u32 {
        let inv_t = 1.0 / temperature;
        let sorted = reference_top(row, row.len());
        let max = sorted
            .iter()
            .map(|c| c.1 * inv_t)
            .filter(|v| !v.is_nan())
            .fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f64> = sorted
            .iter()
            .map(|c| {
                let s = c.1 * inv_t;
                if s.is_nan() {
                    0.0
                } else {
                    f64::from(s - max).exp()
                }
            })
            .collect();
        let target = f64::from(top_p) * weights.iter().sum::<f64>();
        let mut cum = 0.0;
        let keep = weights
            .iter()
            .position(|w| {
                cum += w;
                cum >= target
            })
            .map_or(weights.len(), |i| i + 1);
        let target = f64::from(u) * weights[..keep].iter().sum::<f64>();
        let mut cum = 0.0;
        let i = weights[..keep]
            .iter()
            .position(|w| {
                cum += w;
                target < cum
            })
            .expect("u < 1 falls inside the nucleus");
        sorted[i].0
    }

    #[test]
    fn round_to_matches_the_dtype_precision() {
        let v = 1.0 + 2f32.powi(-10);
        assert_eq!(round_to(DType::BF16, v), 1.0);
        assert_eq!(round_to(DType::F16, v), v);
        assert_eq!(round_to(DType::F32, 0.1), 0.1);
    }
}
