//! Page-locked host memory and asynchronous copies (P4 S-5/S-6, kernel C ABI v3, CONFLICT C-6).
//! `turbine_kernels::ShimContext` implements these on GPUs; `host::HostPinned` is the host-only
//! allocator for tests and GPU-free builds.

use std::sync::Arc;

use crate::buffer::{DevicePtr, MemoryError};

/// Allocates page-locked host buffers.
pub trait PinnedMemory: Send + Sync {
    fn alloc_pinned(&self, bytes: usize) -> Result<PinnedBuffer, MemoryError>;
}

/// The allocator side of a [`PinnedBuffer`]: reaches its bytes and frees them.
pub trait PinnedOwner: Send + Sync {
    /// Calls `f` exactly once with the whole buffer `id`. Callers never do this while a
    /// [`CopyTicket`] writing or reading that buffer is outstanding.
    fn with_bytes_dyn(&self, id: u64, f: &mut dyn FnMut(&mut [u8]));
    /// Frees buffer `id`; called once, by the owning [`PinnedBuffer`]'s Drop.
    fn free_pinned(&self, id: u64);
}

/// One page-locked host allocation, freed on Drop. Not `Clone`: exactly one owner.
pub struct PinnedBuffer {
    id: u64,
    len: usize,
    owner: Arc<dyn PinnedOwner>,
}

impl PinnedBuffer {
    /// Wraps allocation `id` of `owner`; the buffer frees it when dropped.
    pub fn new(id: u64, len: usize, owner: Arc<dyn PinnedOwner>) -> Self {
        PinnedBuffer { id, len, owner }
    }

    /// The allocator's id, the `buffer_id` of [`CopyTarget::Pinned`].
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn with_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        self.with_bytes_mut(|b| f(b))
    }

    pub fn with_bytes_mut<R>(&self, f: impl FnOnce(&mut [u8]) -> R) -> R {
        let mut f = Some(f);
        let mut out = None;
        self.owner.with_bytes_dyn(self.id, &mut |b: &mut [u8]| {
            if let Some(f) = f.take() {
                out = Some(f(b));
            }
        });
        out.expect("PinnedOwner::with_bytes_dyn calls its closure exactly once")
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        self.owner.free_pinned(self.id);
    }
}

impl std::fmt::Debug for PinnedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedBuffer")
            .field("id", &self.id)
            .field("len", &self.len)
            .finish()
    }
}

/// One end of an asynchronous copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyTarget {
    Device(DevicePtr),
    Pinned { buffer_id: u64, offset: usize },
}

pub type CopySource = CopyTarget;

/// Completion handle of one enqueued copy (an event recorded on the copy stream).
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct CopyTicket {
    pub id: u64,
    pub bytes: usize,
}

/// The dedicated copy stream of one device (P4 S-6).
pub trait CopyEngine: Send + Sync {
    fn copy_async(
        &self,
        dst: CopyTarget,
        src: CopySource,
        bytes: usize,
    ) -> Result<CopyTicket, MemoryError>;
    /// `Ok(true)` once the ticket's event has signalled.
    fn poll(&self, t: &CopyTicket) -> Result<bool, MemoryError>;
    fn wait(&self, t: &CopyTicket) -> Result<(), MemoryError>;
    /// A ticket that signals once the work already enqueued on the device's compute stream has
    /// completed (an event recorded on it): what a caller waits on before reusing a buffer a
    /// compute-stream kernel (a KV transcode, P6b S-1) may still be reading. `Unsupported` for
    /// an engine without compute-stream events.
    fn fence_compute(&self) -> Result<CopyTicket, MemoryError> {
        Err(MemoryError::Unsupported(
            "this copy engine cannot fence the compute stream".into(),
        ))
    }
}
