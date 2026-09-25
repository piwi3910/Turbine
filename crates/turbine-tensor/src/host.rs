//! Host-memory backend: "device" allocations are plain heap buffers addressed by synthetic
//! device pointers. Used by the CPU reference provider and by tests.
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, RwLock, Weak};

use turbine_core::types::DeviceId;

use crate::buffer::{DeviceMemory, DevicePtr, MemInfo, MemoryError, StreamRef};

/// Synthetic addresses start above zero so `DevicePtr::NULL` is never a valid allocation.
const BASE_ADDR: u64 = 0x1000_0000;
/// Allocation alignment; consecutive allocations are also separated by one guard page so an
/// address one past the end of an allocation never resolves to the next one.
const ALIGN: u64 = 4096;

/// One host-backed allocation.
type Block = Arc<RwLock<Box<[u8]>>>;

/// A bounded host-memory "device". `capacity` bounds the total allocated bytes.
///
/// Locking: the allocation map lock is never held while a block lock is held by a caller, so
/// `with_slice` / `with_slice_mut` closures may access other allocations. A closure must not
/// access the allocation it is already borrowing mutably (that would deadlock).
pub struct HostMemory {
    device: DeviceId,
    capacity: u64,
    state: Mutex<HostState>,
    blocks: RwLock<BTreeMap<u64, Block>>,
    self_ref: Weak<HostMemory>,
}

struct HostState {
    next_addr: u64,
    used: u64,
}

impl HostMemory {
    /// `mem_info` reports `capacity - used` as free.
    pub fn new(device: DeviceId, capacity: u64) -> Arc<HostMemory> {
        Arc::new_cyclic(|weak| HostMemory {
            device,
            capacity,
            state: Mutex::new(HostState {
                next_addr: BASE_ADDR,
                used: 0,
            }),
            blocks: RwLock::new(BTreeMap::new()),
            self_ref: weak.clone(),
        })
    }

    /// The allocation containing `[p, p + len)` and the offset of `p` inside it.
    fn find(&self, p: DevicePtr, len: usize) -> Result<(Block, usize), MemoryError> {
        let blocks = self.blocks.read().expect("host memory map lock");
        let (base, block) = blocks.range(..=p.addr()).next_back().ok_or_else(|| {
            MemoryError::InvalidArgument(format!("pointer {:#x} is not allocated", p.addr()))
        })?;
        let offset = usize::try_from(p.addr() - base).map_err(|_| {
            MemoryError::InvalidArgument(format!("pointer {:#x} is not allocated", p.addr()))
        })?;
        let size = block.read().expect("host block lock").len();
        if offset.checked_add(len).is_none_or(|end| end > size) {
            return Err(MemoryError::InvalidArgument(format!(
                "range {:#x}+{len} outside allocation {base:#x} of {size} bytes",
                p.addr()
            )));
        }
        Ok((Arc::clone(block), offset))
    }

    /// Calls `f` with the `len` bytes at `p`. Panics when the range is not allocated
    /// (a programming error in a CPU kernel).
    pub fn with_slice<R>(&self, p: DevicePtr, len: usize, f: impl FnOnce(&[u8]) -> R) -> R {
        let (block, off) = self
            .find(p, len)
            .unwrap_or_else(|e| panic!("host memory: {e}"));
        let guard = block.read().expect("host block lock");
        f(&guard[off..off + len])
    }

    /// Calls `f` with the `len` bytes at `p`, mutably. Panics when the range is not allocated.
    pub fn with_slice_mut<R>(&self, p: DevicePtr, len: usize, f: impl FnOnce(&mut [u8]) -> R) -> R {
        let (block, off) = self
            .find(p, len)
            .unwrap_or_else(|e| panic!("host memory: {e}"));
        let mut guard = block.write().expect("host block lock");
        f(&mut guard[off..off + len])
    }
}

impl DeviceMemory for HostMemory {
    fn device(&self) -> DeviceId {
        self.device
    }

    fn alloc(&self, bytes: usize) -> Result<DevicePtr, MemoryError> {
        let requested = bytes as u64;
        let mut st = self.state.lock().expect("host memory state lock");
        if st
            .used
            .checked_add(requested)
            .is_none_or(|total| total > self.capacity)
        {
            return Err(MemoryError::OutOfMemory { requested });
        }
        let span = requested.div_ceil(ALIGN).max(1) * ALIGN + ALIGN;
        let addr = st.next_addr;
        st.next_addr = addr
            .checked_add(span)
            .ok_or(MemoryError::OutOfMemory { requested })?;
        st.used += requested;
        // Insert while still holding the state lock so `free` never sees `used` without the block.
        self.blocks.write().expect("host memory map lock").insert(
            addr,
            Arc::new(RwLock::new(vec![0u8; bytes].into_boxed_slice())),
        );
        Ok(DevicePtr::from_addr(addr))
    }

    fn free(&self, ptr: DevicePtr) {
        let mut st = self.state.lock().expect("host memory state lock");
        let removed = self
            .blocks
            .write()
            .expect("host memory map lock")
            .remove(&ptr.addr());
        if let Some(block) = removed {
            st.used -= block.read().expect("host block lock").len() as u64;
        }
    }

    fn copy_h2d(&self, dst: DevicePtr, src: &[u8]) -> Result<(), MemoryError> {
        let (block, off) = self.find(dst, src.len())?;
        block.write().expect("host block lock")[off..off + src.len()].copy_from_slice(src);
        Ok(())
    }

    fn copy_d2h(&self, dst: &mut [u8], src: DevicePtr) -> Result<(), MemoryError> {
        let (block, off) = self.find(src, dst.len())?;
        dst.copy_from_slice(&block.read().expect("host block lock")[off..off + dst.len()]);
        Ok(())
    }

    fn copy_d2d(&self, dst: DevicePtr, src: DevicePtr, bytes: usize) -> Result<(), MemoryError> {
        // Staged through a temporary so overlapping ranges in one allocation copy correctly.
        let mut tmp = vec![0u8; bytes];
        self.copy_d2h(&mut tmp, src)?;
        self.copy_h2d(dst, &tmp)
    }

    fn synchronize(&self) -> Result<(), MemoryError> {
        Ok(())
    }

    fn mem_info(&self) -> Result<MemInfo, MemoryError> {
        let used = self.state.lock().expect("host memory state lock").used;
        Ok(MemInfo {
            free_bytes: self.capacity - used,
            total_bytes: self.capacity,
        })
    }

    fn compute_stream(&self) -> StreamRef {
        let owner: Arc<dyn DeviceMemory> = self
            .self_ref
            .upgrade()
            .expect("HostMemory is only constructed inside an Arc");
        StreamRef::new(0, self.device, owner)
    }

    fn as_host(&self) -> Option<&HostMemory> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::DeviceBuffer;

    #[test]
    fn alloc_copy_free_round_trip() {
        let host = HostMemory::new(DeviceId(0), 1 << 20);
        let mem: Arc<dyn DeviceMemory> = host;
        let mut buf = DeviceBuffer::alloc(&mem, 16).expect("alloc");
        buf.copy_from_host(4, &[1, 2, 3]).expect("h2d");
        let mut out = [0u8; 3];
        buf.copy_to_host(4, &mut out).expect("d2h");
        assert_eq!(out, [1, 2, 3]);
        assert_eq!(mem.mem_info().expect("info").free_bytes, (1 << 20) - 16);
        drop(buf);
        assert_eq!(mem.mem_info().expect("info").free_bytes, 1 << 20);
        assert!(matches!(
            DeviceBuffer::alloc(&mem, 2 << 20),
            Err(MemoryError::OutOfMemory { requested }) if requested == 2 << 20
        ));
    }

    #[test]
    fn addresses_are_aligned_non_null_and_disjoint() {
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(3), 1 << 20);
        let a = DeviceBuffer::alloc(&mem, 4096).expect("a");
        let b = DeviceBuffer::alloc(&mem, 1).expect("b");
        let empty = DeviceBuffer::alloc(&mem, 0).expect("empty");
        for buf in [&a, &b, &empty] {
            assert_ne!(buf.ptr(), DevicePtr::NULL);
            assert_eq!(buf.ptr().addr() % ALIGN, 0);
            assert_eq!(buf.device(), DeviceId(3));
        }
        assert!(b.ptr().addr() > a.ptr().addr() + 4096);
        // One past the end of `a` is not addressable, and does not alias `b`.
        assert!(matches!(
            mem.copy_h2d(a.ptr().offset(4096), &[9]),
            Err(MemoryError::InvalidArgument(_))
        ));
        assert!(matches!(
            mem.copy_h2d(DevicePtr::NULL, &[9]),
            Err(MemoryError::InvalidArgument(_))
        ));
    }

    #[test]
    fn out_of_range_copies_are_rejected() {
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 20);
        let mut buf = DeviceBuffer::alloc(&mem, 8).expect("alloc");
        assert!(matches!(
            buf.copy_from_host(6, &[1, 2, 3]),
            Err(MemoryError::InvalidArgument(_))
        ));
        let mut out = [0u8; 4];
        assert!(matches!(
            buf.copy_to_host(usize::MAX, &mut out),
            Err(MemoryError::InvalidArgument(_))
        ));
    }

    #[test]
    fn device_to_device_and_host_access() {
        let host = HostMemory::new(DeviceId(0), 1 << 20);
        let mem: Arc<dyn DeviceMemory> = host.clone();
        let mut src = DeviceBuffer::alloc(&mem, 4).expect("src");
        let dst = DeviceBuffer::alloc(&mem, 4).expect("dst");
        src.copy_from_host(0, &[5, 6, 7, 8]).expect("h2d");
        mem.copy_d2d(dst.ptr(), src.ptr().offset(1), 3)
            .expect("d2d");
        assert_eq!(dst.whole().read_bytes().expect("read"), [6, 7, 8, 0]);

        let host_view = mem.as_host().expect("host backend");
        host_view.with_slice_mut(dst.ptr().offset(3), 1, |b| b[0] = 42);
        let sum: u32 = host.with_slice(dst.ptr(), 4, |b| b.iter().map(|&x| u32::from(x)).sum());
        assert_eq!(sum, 6 + 7 + 8 + 42);

        dst.slice(1, 2).write_bytes(&[1, 1]).expect("write");
        assert_eq!(dst.whole().read_bytes().expect("read"), [6, 1, 1, 42]);
        assert!(matches!(
            dst.slice(1, 2).write_bytes(&[1, 1, 1]),
            Err(MemoryError::InvalidArgument(_))
        ));
    }

    #[test]
    fn stream_keeps_its_context_alive() {
        let host = HostMemory::new(DeviceId(1), 1 << 10);
        let stream = host.compute_stream();
        let weak = Arc::downgrade(&host);
        drop(host);
        assert!(weak.upgrade().is_some(), "stream must hold its context");
        assert_eq!(stream.native_handle(), 0);
        assert_eq!(stream.device(), DeviceId(1));
        drop(stream);
        assert!(weak.upgrade().is_none());
    }
}
