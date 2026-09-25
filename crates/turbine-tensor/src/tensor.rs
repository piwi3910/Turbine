//! TS §6 tensors: an owned `Tensor` and a borrowed `TensorView`. Shapes and strides are in
//! elements, row-major.
use std::sync::Arc;

use smallvec::SmallVec;
use turbine_core::types::{DType, DeviceId};

use crate::buffer::{DeviceBuffer, DeviceMemory, DeviceSlice, MemoryError};

/// One owned device allocation plus shape, strides (elements) and dtype.
#[derive(Debug)]
pub struct Tensor {
    pub storage: DeviceBuffer,
    pub shape: SmallVec<[usize; 4]>,
    pub strides: SmallVec<[usize; 4]>,
    pub dtype: DType,
    pub device: DeviceId,
}

impl Tensor {
    /// Allocates a contiguous row-major tensor (contents unspecified).
    pub fn empty(
        mem: &Arc<dyn DeviceMemory>,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor, MemoryError> {
        let bytes = shape
            .iter()
            .try_fold(dtype.size_bytes(), |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| {
                MemoryError::InvalidArgument(format!(
                    "tensor of shape {shape:?} ({}) overflows the address space",
                    dtype.as_str()
                ))
            })?;
        let storage = DeviceBuffer::alloc(mem, bytes)?;
        Ok(Tensor {
            device: storage.device(),
            storage,
            shape: shape.into(),
            strides: contiguous_strides(shape),
            dtype,
        })
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            slice: self.storage.whole(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
        }
    }
}

/// A borrowed view: the byte range it covers, shape/strides (elements) and dtype. `slice`
/// starts at element `[0, 0, ...]` and spans exactly the bytes the view can address.
#[derive(Clone, Debug)]
pub struct TensorView<'a> {
    pub slice: DeviceSlice<'a>,
    pub shape: SmallVec<[usize; 4]>,
    pub strides: SmallVec<[usize; 4]>,
    pub dtype: DType,
}

impl<'a> TensorView<'a> {
    /// Contiguous view of `shape` starting `offset_elems` elements into `slice`. Panics when
    /// the view does not fit inside `slice` (a caller bug).
    pub fn contiguous(
        slice: DeviceSlice<'a>,
        offset_elems: usize,
        shape: &[usize],
        dtype: DType,
    ) -> TensorView<'a> {
        let es = dtype.size_bytes();
        let numel: usize = shape.iter().product();
        TensorView {
            slice: slice.sub(offset_elems * es, numel * es),
            shape: shape.into(),
            strides: contiguous_strides(shape),
            dtype,
        }
    }

    /// Rows `[start, start + count)` along dimension 0, keeping the strides. The slice covers
    /// `(count − 1) · row_stride + row_width` bytes, so a strided parent's trailing padding is
    /// never addressed. Panics when the rows are out of range.
    pub fn rows(&self, start: usize, count: usize) -> TensorView<'a> {
        assert!(!self.shape.is_empty(), "rows() on a 0-dimensional view");
        assert!(
            start
                .checked_add(count)
                .is_some_and(|end| end <= self.shape[0]),
            "rows {start}+{count} outside dimension 0 of size {}",
            self.shape[0]
        );
        let es = self.dtype.size_bytes();
        let row_stride_bytes = self.strides[0] * es;
        let len = if count == 0 {
            0
        } else {
            (count - 1) * row_stride_bytes + self.row_width_bytes()
        };
        let mut shape = self.shape.clone();
        shape[0] = count;
        TensorView {
            slice: self.slice.sub(start * row_stride_bytes, len),
            shape,
            strides: self.strides.clone(),
            dtype: self.dtype,
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Bytes spanned by one row (index 0 along dimension 0): the last addressable element of
    /// the trailing dimensions plus one.
    fn row_width_bytes(&self) -> usize {
        let inner = &self.shape[1..];
        if inner.contains(&0) {
            return 0;
        }
        let last: usize = inner
            .iter()
            .zip(&self.strides[1..])
            .map(|(&d, &s)| (d - 1) * s)
            .sum();
        (last + 1) * self.dtype.size_bytes()
    }
}

/// Row-major strides (elements) for `shape`.
pub fn contiguous_strides(shape: &[usize]) -> SmallVec<[usize; 4]> {
    let mut strides: SmallVec<[usize; 4]> = SmallVec::from_elem(1, shape.len());
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::HostMemory;

    fn host() -> Arc<dyn DeviceMemory> {
        HostMemory::new(DeviceId(0), 1 << 20)
    }

    #[test]
    fn empty_allocates_contiguous_storage() {
        let mem = host();
        let t = Tensor::empty(&mem, &[3, 4, 5], DType::BF16).expect("alloc");
        assert_eq!(t.storage.len(), 3 * 4 * 5 * 2);
        assert_eq!(t.strides.as_slice(), &[20, 5, 1]);
        assert_eq!(t.numel(), 60);
        assert_eq!(t.device, DeviceId(0));
        let v = t.view();
        assert_eq!(v.slice.len(), 120);
        assert_eq!(v.numel(), 60);
        assert!(matches!(
            Tensor::empty(&mem, &[usize::MAX, 2], DType::F32),
            Err(MemoryError::InvalidArgument(_))
        ));
        assert!(matches!(
            Tensor::empty(&mem, &[1 << 20], DType::F32),
            Err(MemoryError::OutOfMemory { .. })
        ));
    }

    #[test]
    fn contiguous_and_rows_address_the_right_bytes() {
        let mem = host();
        let buf = DeviceBuffer::alloc(&mem, 64).expect("alloc");
        let bytes: Vec<u8> = (0..64).collect();
        buf.whole().write_bytes(&bytes).expect("write");

        // A [3, 4] I32 view (48 bytes) starting at element 2 (byte 8).
        let v = TensorView::contiguous(buf.whole(), 2, &[3, 4], DType::I32);
        assert_eq!(v.slice.ptr(), buf.ptr().offset(8));
        assert_eq!(v.slice.len(), 48);
        let r = v.rows(1, 2);
        assert_eq!(r.shape.as_slice(), &[2, 4]);
        assert_eq!(r.slice.ptr(), buf.ptr().offset(8 + 16));
        assert_eq!(r.slice.read_bytes().expect("read"), bytes[24..56]);
        assert_eq!(v.rows(3, 0).slice.len(), 0);
    }

    #[test]
    fn strided_rows_skip_trailing_padding() {
        let mem = host();
        let buf = DeviceBuffer::alloc(&mem, 64).expect("alloc");
        // [4, 3] F16 rows with a row stride of 8 elements (16 bytes): row width 6 bytes.
        let v = TensorView {
            slice: buf.whole(),
            shape: SmallVec::from_slice(&[4, 3]),
            strides: SmallVec::from_slice(&[8, 1]),
            dtype: DType::F16,
        };
        let r = v.rows(2, 2);
        assert_eq!(r.slice.ptr(), buf.ptr().offset(32));
        assert_eq!(r.slice.len(), 16 + 6);
        let last = v.rows(3, 1);
        assert_eq!(last.slice.len(), 6);
    }

    #[test]
    #[should_panic(expected = "outside dimension 0")]
    fn rows_out_of_range_panics() {
        let mem = host();
        let t = Tensor::empty(&mem, &[2, 2], DType::F32).expect("alloc");
        let _ = t.view().rows(1, 2);
    }

    #[test]
    fn dropping_a_tensor_frees_its_storage() {
        let mem = host();
        let t = Tensor::empty(&mem, &[256], DType::F32).expect("alloc");
        assert_eq!(mem.mem_info().expect("info").free_bytes, (1 << 20) - 1024);
        drop(t);
        assert_eq!(mem.mem_info().expect("info").free_bytes, 1 << 20);
    }
}
