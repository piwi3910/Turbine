//! Device pointers, the safe `DeviceMemory` backend trait, and the owning `DeviceBuffer` /
//! borrowed `DeviceSlice` handles over it.
//!
//! Ownership rules (TS §21 rule 10): every device pointer is allocated by a backend and owned by
//! exactly one `DeviceBuffer`, which frees it on `Drop`. A `DeviceBuffer` holds an `Arc` of its
//! `DeviceMemory` (the backend context), so the context outlives every allocation made from it.
use std::fmt;
use std::sync::Arc;

use turbine_core::types::DeviceId;

use crate::host::HostMemory;

/// Opaque device address. Never dereferenced outside the `unsafe`-allowed crates.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DevicePtr(u64);

impl DevicePtr {
    /// The null address; no backend ever returns it from `alloc`.
    pub const NULL: DevicePtr = DevicePtr(0);

    pub fn from_addr(addr: u64) -> DevicePtr {
        DevicePtr(addr)
    }

    pub fn addr(self) -> u64 {
        self.0
    }

    /// The address `bytes` past this one. Panics on address overflow (a caller bug).
    pub fn offset(self, bytes: u64) -> DevicePtr {
        DevicePtr(
            self.0
                .checked_add(bytes)
                .expect("device pointer offset overflows the address space"),
        )
    }
}

/// Device memory counters as reported by the backend.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MemInfo {
    pub free_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("out of device memory ({requested} bytes)")]
    OutOfMemory { requested: u64 },
    /// `sticky` marks errors after which the device context is unusable.
    #[error("device error: {message}")]
    Device { message: String, sticky: bool },
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
}

/// A handle to a backend compute stream. The native handle is 0 until kernel ABI v4 (Phase 5).
/// Holds its owner so the stream cannot outlive the context that created it.
#[derive(Clone)]
pub struct StreamRef {
    native: u64,
    device: DeviceId,
    // Held only to keep the owning context alive while the stream handle exists.
    _owner: Arc<dyn DeviceMemory>,
}

impl StreamRef {
    pub fn new(native: u64, device: DeviceId, owner: Arc<dyn DeviceMemory>) -> StreamRef {
        StreamRef {
            native,
            device,
            _owner: owner,
        }
    }

    pub fn native_handle(&self) -> u64 {
        self.native
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }
}

impl fmt::Debug for StreamRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamRef")
            .field("native", &self.native)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

/// A device-memory backend. Implemented by `turbine_kernels::ShimContext` (GPU) and
/// `host::HostMemory` (CPU reference and tests). Safe by construction: pointers are opaque
/// addresses, and every access goes through the backend, which validates ranges.
pub trait DeviceMemory: Send + Sync {
    fn device(&self) -> DeviceId;
    fn alloc(&self, bytes: usize) -> Result<DevicePtr, MemoryError>;
    /// Releases an allocation returned by `alloc`. Called only by `DeviceBuffer::drop`.
    fn free(&self, ptr: DevicePtr);
    /// Copies `src` to `dst`; enqueued on the compute stream, the caller synchronizes.
    fn copy_h2d(&self, dst: DevicePtr, src: &[u8]) -> Result<(), MemoryError>;
    fn copy_d2h(&self, dst: &mut [u8], src: DevicePtr) -> Result<(), MemoryError>;
    fn copy_d2d(&self, dst: DevicePtr, src: DevicePtr, bytes: usize) -> Result<(), MemoryError>;
    fn synchronize(&self) -> Result<(), MemoryError>;
    fn mem_info(&self) -> Result<MemInfo, MemoryError>;
    fn compute_stream(&self) -> StreamRef;
    /// The host backend, when this is one; lets CPU kernels reach the bytes safely.
    fn as_host(&self) -> Option<&HostMemory> {
        None
    }
}

/// Owns exactly one device allocation; freed on `Drop`. Holds an `Arc` of its memory
/// (the backend context), so it cannot outlive it. Not `Clone`.
pub struct DeviceBuffer {
    ptr: DevicePtr,
    len: usize,
    device: DeviceId,
    mem: Arc<dyn DeviceMemory>,
}

impl fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl DeviceBuffer {
    pub fn alloc(mem: &Arc<dyn DeviceMemory>, bytes: usize) -> Result<DeviceBuffer, MemoryError> {
        let ptr = mem.alloc(bytes)?;
        Ok(DeviceBuffer {
            ptr,
            len: bytes,
            device: mem.device(),
            mem: Arc::clone(mem),
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }

    pub fn ptr(&self) -> DevicePtr {
        self.ptr
    }

    pub fn memory(&self) -> &Arc<dyn DeviceMemory> {
        &self.mem
    }

    /// Bytes `[offset, offset + len)`. Panics when the range leaves the buffer (a caller bug).
    pub fn slice(&self, offset: usize, len: usize) -> DeviceSlice<'_> {
        assert!(
            in_bounds(offset, len, self.len),
            "slice {offset}+{len} outside buffer of {} bytes",
            self.len
        );
        DeviceSlice {
            buf: self,
            offset,
            len,
        }
    }

    pub fn whole(&self) -> DeviceSlice<'_> {
        self.slice(0, self.len)
    }

    /// Enqueued on the context stream; the caller synchronizes before relying on the data.
    pub fn copy_from_host(&mut self, offset: usize, src: &[u8]) -> Result<(), MemoryError> {
        self.check(offset, src.len())?;
        self.mem.copy_h2d(self.ptr.offset(offset as u64), src)
    }

    pub fn copy_to_host(&self, offset: usize, dst: &mut [u8]) -> Result<(), MemoryError> {
        self.check(offset, dst.len())?;
        self.mem.copy_d2h(dst, self.ptr.offset(offset as u64))
    }

    fn check(&self, offset: usize, len: usize) -> Result<(), MemoryError> {
        if in_bounds(offset, len, self.len) {
            Ok(())
        } else {
            Err(MemoryError::InvalidArgument(format!(
                "range {offset}+{len} outside buffer of {} bytes",
                self.len
            )))
        }
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        self.mem.free(self.ptr);
    }
}

/// A borrowed byte range of one `DeviceBuffer`.
#[derive(Clone, Copy)]
pub struct DeviceSlice<'a> {
    buf: &'a DeviceBuffer,
    offset: usize,
    len: usize,
}

impl fmt::Debug for DeviceSlice<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceSlice")
            .field("ptr", &self.ptr())
            .field("len", &self.len)
            .field("device", &self.buf.device)
            .finish()
    }
}

impl<'a> DeviceSlice<'a> {
    pub fn ptr(&self) -> DevicePtr {
        self.buf.ptr.offset(self.offset as u64)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn device(&self) -> DeviceId {
        self.buf.device
    }

    pub fn memory(&self) -> &'a Arc<dyn DeviceMemory> {
        &self.buf.mem
    }

    /// Bytes `[offset, offset + len)` of this slice. Panics when the range leaves the slice.
    pub fn sub(&self, offset: usize, len: usize) -> DeviceSlice<'a> {
        assert!(
            in_bounds(offset, len, self.len),
            "sub-slice {offset}+{len} outside slice of {} bytes",
            self.len
        );
        DeviceSlice {
            buf: self.buf,
            offset: self.offset + offset,
            len,
        }
    }

    /// Blocking read of the whole slice (synchronizes the context before and after).
    pub fn read_bytes(&self) -> Result<Vec<u8>, MemoryError> {
        let mem = &self.buf.mem;
        mem.synchronize()?;
        let mut out = vec![0u8; self.len];
        mem.copy_d2h(&mut out, self.ptr())?;
        mem.synchronize()?;
        Ok(out)
    }

    /// Blocking write of `src` at the start of the slice (synchronizes the context first).
    pub fn write_bytes(&self, src: &[u8]) -> Result<(), MemoryError> {
        if src.len() > self.len {
            return Err(MemoryError::InvalidArgument(format!(
                "{} bytes into a {}-byte slice",
                src.len(),
                self.len
            )));
        }
        let mem = &self.buf.mem;
        mem.synchronize()?;
        mem.copy_h2d(self.ptr(), src)?;
        mem.synchronize()
    }
}

fn in_bounds(offset: usize, len: usize, size: usize) -> bool {
    offset.checked_add(len).is_some_and(|end| end <= size)
}
