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
    // Keeps the owning context alive while the stream handle exists.
    owner: Arc<dyn DeviceMemory>,
}

impl StreamRef {
    pub fn new(native: u64, device: DeviceId, owner: Arc<dyn DeviceMemory>) -> StreamRef {
        StreamRef {
            native,
            device,
            owner,
        }
    }

    pub fn native_handle(&self) -> u64 {
        self.native
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// The backend context that owns this stream (for scratch buffers and synchronization
    /// ordered on it, e.g. a collective barrier).
    pub fn memory(&self) -> &Arc<dyn DeviceMemory> {
        &self.owner
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

    /// Allocates `bytes` of host staging memory (page-locked on a GPU backend, kernel ABI v2.3)
    /// for the asynchronous copies [`DeviceMemory::copy_h2d_staged`] and
    /// [`DeviceMemory::copy_d2h_staged`]. `Unsupported` when the backend has none; callers then
    /// use the synchronous `copy_h2d`/`copy_d2h`. Owned by exactly one [`HostStaging`].
    fn staging_alloc(&self, bytes: usize) -> Result<StagingId, MemoryError> {
        let _ = bytes;
        Err(no_staging())
    }

    /// Releases a staging buffer once the staged copies still using it are done. Called only by
    /// `HostStaging::drop`.
    fn staging_free(&self, id: StagingId) {
        let _ = id;
    }

    /// Writes `src` at `offset` of staging buffer `id`, first waiting for the staged copies that
    /// use this buffer (and only those) to complete.
    fn staging_write(&self, id: StagingId, offset: usize, src: &[u8]) -> Result<(), MemoryError> {
        let _ = (id, offset, src);
        Err(no_staging())
    }

    /// Reads `dst.len()` bytes at `offset` of staging buffer `id`, first waiting for the staged
    /// copies that use this buffer (and only those) to complete.
    fn staging_read(
        &self,
        id: StagingId,
        offset: usize,
        dst: &mut [u8],
    ) -> Result<(), MemoryError> {
        let _ = (id, offset, dst);
        Err(no_staging())
    }

    /// Enqueues the copy of `bytes` bytes at `offset` of staging buffer `id` to `dst` on the
    /// compute stream and returns at once: it waits neither for earlier work on the stream nor
    /// for the copy.
    fn copy_h2d_staged(
        &self,
        dst: DevicePtr,
        id: StagingId,
        offset: usize,
        bytes: usize,
    ) -> Result<(), MemoryError> {
        let _ = (dst, id, offset, bytes);
        Err(no_staging())
    }

    /// Enqueues the copy of the `bytes` bytes at `src` to `offset` of staging buffer `id` on the
    /// compute stream and returns at once; `staging_read` of that buffer waits for it.
    fn copy_d2h_staged(
        &self,
        id: StagingId,
        offset: usize,
        src: DevicePtr,
        bytes: usize,
    ) -> Result<(), MemoryError> {
        let _ = (id, offset, src, bytes);
        Err(no_staging())
    }
}

fn no_staging() -> MemoryError {
    MemoryError::Unsupported("host staging needs kernel ABI v2.3".into())
}

/// A staging buffer of one backend ([`DeviceMemory::staging_alloc`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StagingId(pub u64);

/// Owns one host staging buffer (kernel ABI v2.3), freed on `Drop`: host bytes the device copies
/// to and from asynchronously. Every host access first waits for the staged copies that use this
/// buffer, so a copy is never overwritten or read before it completes, while copies of other
/// buffers and the rest of the stream keep running. Holds an `Arc` of its memory, like
/// `DeviceBuffer`, so the backend context outlives it.
pub struct HostStaging {
    id: StagingId,
    len: usize,
    mem: Arc<dyn DeviceMemory>,
}

impl fmt::Debug for HostStaging {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostStaging")
            .field("id", &self.id)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl HostStaging {
    /// `Unsupported` when `mem` has no staging (see [`DeviceMemory::staging_alloc`]).
    pub fn alloc(mem: &Arc<dyn DeviceMemory>, bytes: usize) -> Result<HostStaging, MemoryError> {
        let id = mem.staging_alloc(bytes)?;
        Ok(HostStaging {
            id,
            len: bytes,
            mem: Arc::clone(mem),
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Writes `src` at `offset` once this buffer's staged copies are done.
    pub fn write(&self, offset: usize, src: &[u8]) -> Result<(), MemoryError> {
        self.check(offset, src.len())?;
        self.mem.staging_write(self.id, offset, src)
    }

    /// Reads `dst.len()` bytes at `offset` once this buffer's staged copies are done.
    pub fn read(&self, offset: usize, dst: &mut [u8]) -> Result<(), MemoryError> {
        self.check(offset, dst.len())?;
        self.mem.staging_read(self.id, offset, dst)
    }

    /// Enqueues the copy of `dst.len()` bytes at `offset` into `dst` (asynchronous).
    pub fn upload(&self, offset: usize, dst: DeviceSlice<'_>) -> Result<(), MemoryError> {
        self.check(offset, dst.len())?;
        self.same_memory(dst.memory())?;
        self.mem
            .copy_h2d_staged(dst.ptr(), self.id, offset, dst.len())
    }

    /// Enqueues the copy of `src` into the bytes at `offset` (asynchronous).
    pub fn download(&self, offset: usize, src: DeviceSlice<'_>) -> Result<(), MemoryError> {
        self.check(offset, src.len())?;
        self.same_memory(src.memory())?;
        self.mem
            .copy_d2h_staged(self.id, offset, src.ptr(), src.len())
    }

    fn check(&self, offset: usize, len: usize) -> Result<(), MemoryError> {
        if in_bounds(offset, len, self.len) {
            Ok(())
        } else {
            Err(MemoryError::InvalidArgument(format!(
                "range {offset}+{len} outside staging buffer of {} bytes",
                self.len
            )))
        }
    }

    fn same_memory(&self, other: &Arc<dyn DeviceMemory>) -> Result<(), MemoryError> {
        if std::ptr::addr_eq(Arc::as_ptr(other), Arc::as_ptr(&self.mem)) {
            Ok(())
        } else {
            Err(MemoryError::InvalidArgument(
                "staged copy between different device memories".into(),
            ))
        }
    }
}

impl Drop for HostStaging {
    fn drop(&mut self) {
        self.mem.staging_free(self.id);
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
