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
    ActivationKernel, AddRmsnormKernel, AttentionKernel, ElementwiseKernel, EmbeddingKernel,
    GemmKernel, KernelProvider, KvCopyKernel, LogitsReduceKernel, MoeKernel, NormKernel,
    ProviderId, QGemmKernel, QuantizeActKernel, RopeKernel, ShardedNormKernel,
};

// One file per op family (Phase 2m S-5); `math`, `paged` and `topk` hold the shared numerics.
mod activation;
mod attention;
mod elementwise;
mod embedding;
mod gemm;
mod kv_copy;
mod logits_reduce;
mod math;
mod moe;
mod norm;
mod paged;
mod qgemm;
pub mod quant;
mod rope;
mod topk;

pub use topk::torch_topk;

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
    // Bytes the view addresses, (Σ (d − 1)·stride + 1)·es; None when that overflows usize.
    let span = v
        .shape
        .iter()
        .zip(&v.strides)
        .try_fold(0usize, |acc, (&d, &s)| {
            acc.checked_add((d - 1).checked_mul(s)?)
        })
        .and_then(|last| last.checked_add(1)?.checked_mul(es));
    if span.is_none_or(|bytes| bytes > v.slice.len()) {
        return Err(invalid(format!(
            "view of shape {:?} strides {:?} ({}) addresses {} bytes but its slice holds {}",
            v.shape.as_slice(),
            v.strides.as_slice(),
            v.dtype.as_str(),
            span.map_or_else(|| "more than usize::MAX".to_string(), |b| b.to_string()),
            v.slice.len()
        )));
    }
    let rank = v.shape.len();
    let mut offsets = Vec::with_capacity(numel);
    let mut idx = vec![0usize; rank];
    let mut elem = 0usize;
    for _ in 0..numel {
        offsets.push(elem * es);
        // Odometer increment of the multi-index, keeping `elem` = Σ idx·stride. The carry step
        // may pass through values above the checked span (a size-1 dimension with a huge stride),
        // so it wraps; every pushed `elem` is within the span.
        for dim in (0..rank).rev() {
            idx[dim] += 1;
            elem = elem.wrapping_add(v.strides[dim]);
            if idx[dim] < v.shape[dim] {
                break;
            }
            elem = elem.wrapping_sub(v.shape[dim].wrapping_mul(v.strides[dim]));
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

/// Reads a one-byte view (U8 or F8E4M3) as its raw bytes, in logical row-major order.
pub(crate) fn load_bytes(v: &TensorView<'_>) -> Result<Vec<u8>, KernelError> {
    if v.dtype.size_bytes() != 1 {
        return Err(invalid(format!(
            "expected a one-byte view, got {}",
            v.dtype.as_str()
        )));
    }
    let offsets = element_offsets(v)?;
    let bytes = v.slice.read_bytes()?;
    Ok(offsets.iter().map(|&o| bytes[o]).collect())
}

/// Writes raw bytes into a one-byte view (U8 or F8E4M3) in logical row-major order; bytes of
/// the slice the view does not address are preserved.
pub(crate) fn store_bytes(v: &TensorView<'_>, values: &[u8]) -> Result<(), KernelError> {
    if v.dtype.size_bytes() != 1 {
        return Err(invalid(format!(
            "expected a one-byte view, got {}",
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
    for (&o, &b) in offsets.iter().zip(values) {
        bytes[o] = b;
    }
    v.slice.write_bytes(&bytes)?;
    Ok(())
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
    fn sharded_norm(&self) -> Option<&dyn ShardedNormKernel> {
        Some(self)
    }
    fn qgemm(&self) -> Option<&dyn QGemmKernel> {
        Some(self)
    }
    fn quantize_act(&self) -> Option<&dyn QuantizeActKernel> {
        Some(self)
    }
}

/// Test helpers and imports shared by the op-family test modules.
#[cfg(test)]
mod test_util {
    pub(super) use std::sync::Arc;

    pub(super) use turbine_core::types::{BlockId, DType, DeviceId};
    pub(super) use turbine_tensor::host::HostMemory;
    pub(super) use turbine_tensor::{DeviceBuffer, DeviceMemory, Tensor, TensorView};

    pub(super) use crate::ops::*;

    use super::store;

    pub(super) fn host() -> Arc<dyn DeviceMemory> {
        HostMemory::new(DeviceId(0), 1 << 20)
    }

    /// A contiguous tensor holding `values` rounded to `dtype`.
    pub(super) fn tensor(
        mem: &Arc<dyn DeviceMemory>,
        shape: &[usize],
        dtype: DType,
        values: &[f32],
    ) -> Tensor {
        let t = Tensor::empty(mem, shape, dtype).expect("alloc");
        store(&t.view(), values).expect("store");
        t
    }

    pub(super) fn i32_tensor(mem: &Arc<dyn DeviceMemory>, values: &[i32]) -> Tensor {
        let t = Tensor::empty(mem, &[values.len()], DType::I32).expect("alloc");
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        t.storage.whole().write_bytes(&bytes).expect("write");
        t
    }

    pub(super) fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    pub(super) fn assert_close(got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len(), "got {got:?}, want {want:?}");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= tol,
                "element {i}: got {g}, want {w} (all: {got:?})"
            );
        }
    }

    /// Deterministic values in [-1, 1) (64-bit LCG), so the test needs no RNG crate.
    pub(super) fn seeded(seed: u64, n: usize) -> Vec<f32> {
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

    pub(super) fn i32_tensor_2d(
        mem: &Arc<dyn DeviceMemory>,
        rows: usize,
        values: &[i32],
    ) -> Tensor {
        let t = Tensor::empty(mem, &[rows, values.len() / rows], DType::I32).expect("alloc");
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        t.storage.whole().write_bytes(&bytes).expect("write");
        t
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::test_util::*;
    use crate::cpu::*;

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

    /// Catches: `(d - 1) * stride` summed without overflow checks, so a huge stride wrapped
    /// the bounds check (a debug-build panic, an out-of-range index in release).
    #[test]
    fn overflowing_strides_are_an_error_and_unit_dims_ignore_their_stride() {
        let mem = host();
        let buf = DeviceBuffer::alloc(&mem, 4 * 4).expect("alloc");
        let view = |shape: &[usize], strides: &[usize]| TensorView {
            slice: buf.whole(),
            shape: shape.into(),
            strides: strides.into(),
            dtype: DType::F32,
        };
        let err = element_offsets(&view(&[2, 2], &[usize::MAX, 1])).unwrap_err();
        assert!(
            matches!(err, KernelError::InvalidArgument { .. }),
            "got {err:?}"
        );
        let err = element_offsets(&view(&[2], &[usize::MAX / 4 + 1])).unwrap_err();
        assert!(
            matches!(err, KernelError::InvalidArgument { .. }),
            "got {err:?}"
        );
        // A size-1 dimension never advances by its stride, however large.
        assert_eq!(
            element_offsets(&view(&[1, 4], &[usize::MAX, 1])).expect("offsets"),
            [0, 4, 8, 12]
        );
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
                prefill: false,
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
