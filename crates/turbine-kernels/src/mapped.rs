//! Kernel C ABI v2.7 (Phase 5, decision "P5: small-message all-reduce latency on novanas") and
//! its v2.8 device-sequenced step (P5 Task 32, tensor-parallel decode graphs):
//! host memory mapped into every device of the process and the one-shot collective steps over
//! it, exposed as `turbine_tensor::MappedCollectives` on `ShimContext` (through
//! `DeviceMemory::mapped_collectives`). A library without the group answers `None` there, and
//! the `hostmem` collective backend is unavailable on it.
//!
//! Ownership rules (TS §21 rule 10):
//! - a mapped allocation comes from `turbine_host_alloc_mapped` on one context and is owned by
//!   one `ShimMapped`, shared through `MappedRegion` handles and freed exactly once, in
//!   `ShimMapped::drop`, through that context (the `ShimMapped` holds an `Arc` of it, so the
//!   context outlives the allocation). The caller drops its last handle only after every step
//!   that uses the region has completed on every device (the `hostmem` backend synchronizes each
//!   rank's stream before it lets go of the region);
//! - the host reaches the bytes only through 4-byte atomic loads and stores of words inside the
//!   allocation (the flag and abort words), never as a slice, because the devices write the
//!   memory concurrently;
//! - a device address of the region (`turbine_host_mapped_device_ptr`) is only carried as a
//!   `DevicePtr` into step descriptors; it is valid while the allocation lives.
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use turbine_tensor::{
    DevicePtr, MappedCollectives, MappedDma, MappedHost, MappedKind, MappedReduce, MappedRegion,
    MappedStep, MemoryError,
};

use crate::ffi::{MappedCollectiveDesc, MappedDmaDesc, MappedFns};
use crate::shim::ShimContext;

/// One `turbine_host_alloc_mapped` allocation, freed on Drop through its context.
struct ShimMapped {
    host: *mut u8,
    len: usize,
    ctx: Arc<ShimContext>,
    free: unsafe extern "C" fn(*mut crate::ffi::TurbineCtx, *mut c_void) -> i32,
}

// SAFETY: `host` is page-locked host memory owned by this value; the Rust side touches it only
// through atomic 4-byte loads and stores (see the module's ownership rules), which are sound
// from any thread, and the shim's free may be called from any thread (contract §9.4).
unsafe impl Send for ShimMapped {}
// SAFETY: see `Send`: every access through `&self` is an atomic word access.
unsafe impl Sync for ShimMapped {}

impl ShimMapped {
    fn word(&self, offset: usize) -> &AtomicU32 {
        assert!(
            offset.is_multiple_of(4) && offset + 4 <= self.len,
            "mapped word at {offset} outside {} bytes",
            self.len
        );
        // SAFETY: `host .. host + len` is a live allocation of this value (freed only in Drop,
        // which needs `&mut self`), the offset is 4-aligned and in bounds (checked above; the
        // allocation itself is page-aligned), and every access to these bytes from the host goes
        // through `AtomicU32`, so no non-atomic access races with it. Devices write the word with
        // system-scope atomic stores of the same width.
        unsafe { AtomicU32::from_ptr(self.host.add(offset).cast::<u32>()) }
    }
}

impl MappedHost for ShimMapped {
    fn len(&self) -> usize {
        self.len
    }
    fn host_addr(&self) -> u64 {
        self.host as u64
    }
    fn load_u32(&self, offset: usize) -> u32 {
        self.word(offset).load(Ordering::Acquire)
    }
    fn store_u32(&self, offset: usize, value: u32) {
        self.word(offset).store(value, Ordering::Release)
    }
}

impl Drop for ShimMapped {
    fn drop(&mut self) {
        // SAFETY: `host` came from `turbine_host_alloc_mapped` on `ctx`'s library and is freed
        // exactly once, here; the context is alive (held by `self.ctx`), and by the module's
        // ownership rules no enqueued step uses the region any more.
        let code = unsafe { (self.free)(self.ctx.raw_ctx(), self.host.cast::<c_void>()) };
        if code != 0 {
            tracing::error!(event = "kernel_mapped_free_failed", code);
        }
    }
}

impl ShimContext {
    /// True when the library exports the ABI v2.7 group (host-mapped memory and the one-shot
    /// collective steps), so `DeviceMemory::mapped_collectives` answers `Some`.
    pub fn has_mapped_collectives(&self) -> bool {
        self.library().syms().v21.mapped.is_some()
    }

    fn mapped_fns(&self) -> Result<MappedFns, MemoryError> {
        self.library().syms().v21.mapped.ok_or_else(|| {
            MemoryError::Unsupported(format!(
                "{} does not export the kernel ABI v2.7 host-mapped group (minor {})",
                self.library().path().display(),
                self.library().abi_minor()
            ))
        })
    }

    fn mapped_check(&self, code: i32) -> Result<(), MemoryError> {
        crate::ffi::check(code, self.library().syms(), self.raw_ctx()).map_err(MemoryError::from)
    }
}

/// The descriptor of `step`; pointer fields carry this device's addresses.
fn descriptor(step: &MappedStep) -> Result<MappedCollectiveDesc, MemoryError> {
    let i64_of = |what: &str, v: u64| {
        i64::try_from(v).map_err(|_| MemoryError::InvalidArgument(format!("{what} {v} overflows")))
    };
    let timeout_ns = i64::try_from(step.timeout.as_nanos()).unwrap_or(i64::MAX);
    let root = match step.kind {
        MappedKind::Broadcast { root } => root,
        _ => 0,
    };
    let as_i32 = |what: &str, v: u32| {
        i32::try_from(v).map_err(|_| MemoryError::InvalidArgument(format!("{what} {v} overflows")))
    };
    Ok(MappedCollectiveDesc {
        send: step.send.addr() as *const c_void,
        recv: step.recv.addr() as *mut c_void,
        bytes: i64_of("bytes", step.bytes)?,
        send_stride: i64_of("send_stride", step.send_stride)?,
        recv_stride: i64_of("recv_stride", step.recv_stride)?,
        slots: step.slots.addr() as *mut c_void,
        slot_bytes: i64_of("slot_bytes", step.slot_bytes)?,
        flags: step.flags.addr() as *mut u64,
        abort_word: step.abort_word.addr() as *mut u32,
        seq: step.seq,
        timeout_ns: timeout_ns.max(1),
        kind: step.kind.abi_code(),
        reduce_op: match step.reduce {
            MappedReduce::Sum => 0,
            MappedReduce::Max => 1,
        },
        // The byte-wise kinds ignore it; a reduction of another type is refused by the library.
        dtype: step.dtype.abi_code(),
        rank: as_i32("rank", step.rank)?,
        world: as_i32("world", step.world)?,
        root: as_i32("root", root)?,
        max_blocks: as_i32("max_blocks", step.max_blocks)?,
    })
}

impl MappedCollectives for ShimContext {
    fn alloc_mapped(&self, bytes: usize) -> Result<MappedRegion, MemoryError> {
        let fns = self.mapped_fns()?;
        let mut host: *mut c_void = std::ptr::null_mut();
        // SAFETY: `host` is a live out-pointer on this stack frame; on success the shim stores a
        // zeroed page-locked allocation of `bytes` bytes, owned from here by one `ShimMapped`.
        let code = unsafe { (fns.alloc)(self.raw_ctx(), bytes, &mut host) };
        match self.mapped_check(code) {
            Ok(()) => {}
            Err(MemoryError::OutOfMemory { .. }) => {
                return Err(MemoryError::OutOfMemory {
                    requested: bytes as u64,
                });
            }
            Err(e) => return Err(e),
        }
        if host.is_null() {
            return Err(MemoryError::Device {
                message: "turbine_host_alloc_mapped succeeded but returned NULL".into(),
                sticky: false,
            });
        }
        Ok(MappedRegion::new(Arc::new(ShimMapped {
            host: host.cast::<u8>(),
            len: bytes,
            ctx: self.owning_arc(),
            free: fns.free,
        })))
    }

    fn mapped_device_addr(&self, region: &MappedRegion) -> Result<DevicePtr, MemoryError> {
        let fns = self.mapped_fns()?;
        let mut dev: *mut c_void = std::ptr::null_mut();
        // SAFETY: `region.host_addr()` is the host address of a live mapped allocation (the
        // region handle keeps it alive for this call); the shim only looks it up and writes the
        // device address into the live out-pointer `dev`.
        let code = unsafe {
            (fns.device_ptr)(self.raw_ctx(), region.host_addr() as *mut c_void, &mut dev)
        };
        self.mapped_check(code)?;
        Ok(DevicePtr::from_addr(dev as u64))
    }

    fn mapped_step_supported(&self, step: &MappedStep) -> bool {
        let (Ok(fns), Ok(desc)) = (self.mapped_fns(), descriptor(step)) else {
            return false;
        };
        // SAFETY: `_supported` reads only the descriptor's scalar fields (header: pointer fields
        // ignored, no context); `desc` is a live local.
        unsafe { (fns.collective.supported)(&desc) == 1 }
    }

    fn enqueue_mapped_step(&self, step: &MappedStep) -> Result<(), MemoryError> {
        let fns = self.mapped_fns()?;
        let desc = descriptor(step)?;
        // SAFETY: `desc` is a live local the shim reads during the call only. Its device
        // pointers are the caller's: `send`/`recv` are device memory of this context that the
        // caller keeps alive until the step completes on the compute stream, and
        // `slots`/`flags`/`abort_word` are this context's addresses of a mapped region the caller
        // keeps alive until then as well (see the module's ownership rules).
        let code = unsafe { (fns.collective.run)(self.raw_ctx(), &desc) };
        self.mapped_check(code)
    }

    fn mapped_dseq_supported(&self) -> bool {
        self.library().syms().v21.mapped_dseq.is_some()
    }

    fn enqueue_mapped_step_dseq(
        &self,
        step: &MappedStep,
        seq_counter: DevicePtr,
    ) -> Result<(), MemoryError> {
        let run = self.library().syms().v21.mapped_dseq.ok_or_else(|| {
            MemoryError::Unsupported(format!(
                "{} does not export the kernel ABI v2.8 device-sequenced collective step \
                 (minor {})",
                self.library().path().display(),
                self.library().abi_minor()
            ))
        })?;
        let mut desc = descriptor(step)?;
        // Ignored by the library, but a valid descriptor has seq >= 1.
        desc.seq = 1;
        // SAFETY: as in `enqueue_mapped_step`; `seq_counter` is 16 bytes of this context's
        // device memory that the caller keeps alive (and uses for this channel only) until the
        // step completes, and every graph that captured it is destroyed (hostmem's rules).
        let code = unsafe { run(self.raw_ctx(), &desc, seq_counter.addr() as *mut u64) };
        self.mapped_check(code)
    }

    fn mapped_capturing(&self) -> bool {
        self.is_capturing()
    }

    fn mapped_dma_supported(&self) -> bool {
        self.library().syms().v21.mapped_dma.is_some()
    }

    fn enqueue_mapped_all_reduce_dma(
        &self,
        step: &MappedStep,
        dma: &MappedDma,
    ) -> Result<(), MemoryError> {
        let fns = self.library().syms().v21.mapped_dma.ok_or_else(|| {
            MemoryError::Unsupported(format!(
                "{} does not export the kernel ABI v2.8 copy-engine all-reduce (minor {})",
                self.library().path().display(),
                self.library().abi_minor()
            ))
        })?;
        let desc = descriptor(step)?;
        let (copy, event) = self.collective_copy_resources()?;
        let chunk_bytes = i64::try_from(dma.chunk_bytes).map_err(|_| {
            MemoryError::InvalidArgument(format!("chunk_bytes {} overflows", dma.chunk_bytes))
        })?;
        let m = MappedDmaDesc {
            copy,
            event,
            scratch: dma.scratch.addr() as *mut c_void,
            chunk_bytes,
            seq_counter: dma.seq_counter.addr() as *mut u64,
            flags: i32::from(dma.peer_read),
        };
        // SAFETY: `desc` and `m` are live locals read during the call only. `copy` and `event`
        // are this context's (owned by its pinned state until the context is destroyed, and
        // destroying the stream waits for its copies); the device pointers follow
        // `enqueue_mapped_step`'s rules, and `scratch` / `seq_counter` are this context's device
        // memory the caller keeps alive until the step completes on the compute stream.
        let code = unsafe { (fns.run)(self.raw_ctx(), &desc, &m) };
        self.mapped_check(code)
    }

    fn alloc_dma_region(&self, bytes: usize) -> Result<MappedRegion, MemoryError> {
        let (fns, mapped) = match (self.library().syms().v21.mapped_dma, self.mapped_fns()) {
            (Some(fns), Ok(mapped)) => (fns, mapped),
            _ => {
                return Err(MemoryError::Unsupported(
                    "copy-engine slots (kernel ABI v2.8)".into(),
                ));
            }
        };
        let mut host: *mut c_void = std::ptr::null_mut();
        // SAFETY: `host` is a live out-pointer; on success the shim stores a page-locked
        // allocation of `bytes` bytes, owned from here by one `ShimMapped` (freed once through
        // `turbine_host_free_mapped`, as the header says).
        let code = unsafe { (fns.alloc)(self.raw_ctx(), bytes, &mut host) };
        self.mapped_check(code)?;
        if host.is_null() {
            return Err(MemoryError::Device {
                message: "turbine_host_alloc_dma succeeded but returned NULL".into(),
                sticky: false,
            });
        }
        Ok(MappedRegion::new(Arc::new(ShimMapped {
            host: host.cast::<u8>(),
            len: bytes,
            ctx: self.owning_arc(),
            free: mapped.free,
        })))
    }
}
