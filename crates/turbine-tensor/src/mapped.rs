//! Host memory mapped into every device of the process, and the one-shot collective steps that
//! exchange data through it (kernel C ABI v2.7): the device side of the `hostmem` collective
//! backend (`turbine-distributed`), for devices without a peer-to-peer path.
//! `turbine_kernels::ShimContext` implements [`MappedCollectives`] when its kernel library
//! exports the group; every other backend answers `None` from
//! [`crate::DeviceMemory::mapped_collectives`].

use std::sync::Arc;
use std::time::Duration;

use crate::buffer::{DevicePtr, MemoryError};
use crate::dtype::DType;

/// One mapped allocation, seen from the host. Implemented by the backend that allocated it;
/// the memory is freed when the last [`MappedRegion`] handle drops.
pub trait MappedHost: Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// An opaque host address identifying the allocation to the backend that maps it into a
    /// device ([`MappedCollectives::mapped_device_addr`]); never dereferenced by callers.
    fn host_addr(&self) -> u64;
    /// The `u32` at byte `offset` (4-aligned, in bounds), read with acquire ordering.
    fn load_u32(&self, offset: usize) -> u32;
    /// Stores `value` at byte `offset` (4-aligned, in bounds) with release ordering.
    fn store_u32(&self, offset: usize, value: u32);
}

/// A shared handle to one mapped allocation (host memory every device of the process can
/// address). Cloning shares it; the memory is freed when the last handle drops.
#[derive(Clone)]
pub struct MappedRegion(Arc<dyn MappedHost>);

impl MappedRegion {
    pub fn new(host: Arc<dyn MappedHost>) -> MappedRegion {
        MappedRegion(host)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.len() == 0
    }

    pub fn host_addr(&self) -> u64 {
        self.0.host_addr()
    }

    /// Panics when `offset` is misaligned or out of bounds (a caller bug).
    pub fn load_u32(&self, offset: usize) -> u32 {
        self.check(offset);
        self.0.load_u32(offset)
    }

    /// Panics when `offset` is misaligned or out of bounds (a caller bug).
    pub fn store_u32(&self, offset: usize, value: u32) {
        self.check(offset);
        self.0.store_u32(offset, value)
    }

    fn check(&self, offset: usize) {
        assert!(
            offset.is_multiple_of(4) && offset.saturating_add(4) <= self.len(),
            "mapped word at {offset} outside a region of {} bytes",
            self.len()
        );
    }
}

impl std::fmt::Debug for MappedRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedRegion")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

/// What one step computes (kernel ABI `TURBINE_MAPPED_*`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MappedKind {
    AllReduce,
    AllGather,
    ReduceScatter,
    Broadcast { root: u32 },
}

impl MappedKind {
    /// The `TURBINE_MAPPED_*` code.
    pub fn abi_code(self) -> i32 {
        match self {
            MappedKind::AllReduce => 0,
            MappedKind::AllGather => 1,
            MappedKind::ReduceScatter => 2,
            MappedKind::Broadcast { .. } => 3,
        }
    }

    /// The kind whose `TURBINE_MAPPED_*` code is `code` (a broadcast with root 0).
    pub fn from_abi_code(code: u32) -> Option<MappedKind> {
        Some(match code {
            0 => MappedKind::AllReduce,
            1 => MappedKind::AllGather,
            2 => MappedKind::ReduceScatter,
            3 => MappedKind::Broadcast { root: 0 },
            _ => return None,
        })
    }
}

/// The element-wise reduction of a reducing step (`TURBINE_REDUCE_*`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MappedReduce {
    Sum,
    Max,
}

/// `abort_word` reasons (bits 24..31 of the word; bits 16..23 hold the kind of the step that
/// timed out, bits 0..15 the rank that stored it).
pub const MAPPED_ABORT_TIMEOUT: u32 = 1;
pub const MAPPED_ABORT_HOST: u32 = 2;

/// One collective step of one rank over a mapped region (the kernel ABI's
/// `turbine_mapped_collective_desc`; its header comment is the contract). Every address is this
/// device's.
#[derive(Clone, Copy, Debug)]
pub struct MappedStep {
    pub kind: MappedKind,
    pub reduce: MappedReduce,
    /// BF16 or F32 for the reductions; ignored by the byte-wise kinds.
    pub dtype: DType,
    pub rank: u32,
    pub world: u32,
    pub send: DevicePtr,
    pub recv: DevicePtr,
    /// Bytes of one rank's part.
    pub bytes: u64,
    /// Reduce-scatter: bytes between the ranks' parts of `send`.
    pub send_stride: u64,
    /// All-gather: bytes between the ranks' parts of `recv`.
    pub recv_stride: u64,
    /// `2 × world` slots of `slot_bytes`.
    pub slots: DevicePtr,
    pub slot_bytes: u64,
    /// `world × max_blocks` flag words.
    pub flags: DevicePtr,
    pub max_blocks: u32,
    pub abort_word: DevicePtr,
    /// ≥ 1, strictly increasing, equal on every rank for one step.
    pub seq: u64,
    /// How long one wait for a peer may spin before the step gives up.
    pub timeout: Duration,
}

/// Mapped host memory and the one-shot collective steps over it, on one device (kernel ABI
/// v2.7). Steps are enqueued on the device's compute stream in order with its other work.
pub trait MappedCollectives: Send + Sync {
    /// `bytes` of zeroed page-locked host memory that every device of this process can map,
    /// coherent between the host and the devices.
    fn alloc_mapped(&self, bytes: usize) -> Result<MappedRegion, MemoryError>;
    /// The address of `region` on this device.
    fn mapped_device_addr(&self, region: &MappedRegion) -> Result<DevicePtr, MemoryError>;
    /// Whether this device runs `step` (pointer fields ignored).
    fn mapped_step_supported(&self, step: &MappedStep) -> bool;
    /// Enqueues `step` on the compute stream; returns without waiting for it or for the peers.
    fn enqueue_mapped_step(&self, step: &MappedStep) -> Result<(), MemoryError>;
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    struct Words(Mutex<Vec<u32>>);

    impl MappedHost for Words {
        fn len(&self) -> usize {
            self.0.lock().unwrap().len() * 4
        }
        fn host_addr(&self) -> u64 {
            1
        }
        fn load_u32(&self, offset: usize) -> u32 {
            self.0.lock().unwrap()[offset / 4]
        }
        fn store_u32(&self, offset: usize, value: u32) {
            self.0.lock().unwrap()[offset / 4] = value;
        }
    }

    #[test]
    fn region_words_are_bounds_checked() {
        let r = MappedRegion::new(Arc::new(Words(Mutex::new(vec![0; 4]))));
        assert_eq!(r.len(), 16);
        r.store_u32(12, 7);
        assert_eq!(r.clone().load_u32(12), 7);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| r.load_u32(14))).is_err());
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| r.load_u32(16))).is_err());
    }

    #[test]
    fn kind_codes_round_trip() {
        for code in 0..4 {
            let kind = MappedKind::from_abi_code(code).expect("known");
            assert_eq!(kind.abi_code(), code as i32);
        }
        assert_eq!(MappedKind::from_abi_code(4), None);
    }
}
