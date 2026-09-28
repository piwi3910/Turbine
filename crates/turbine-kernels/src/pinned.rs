//! Kernel C ABI v2.3 + v2.5 (P4 S-5/S-6, CONFLICT C-6; provisional decision "Phase 4: kernel
//! ABI v2.5 instead of v3"): page-locked host memory, the per-device copy stream and
//! copy-completion events, exposed as `turbine_tensor::{PinnedMemory, PinnedOwner, CopyEngine}`
//! on `ShimContext`. No HIP or CUDA runtime is bound outside the shim. A library without both
//! optional groups answers `MemoryError::Unsupported` (the server then runs L0 only, with a
//! WARN); `ShimContext::has_copy_engine` says which it is.
//!
//! Ownership rules (TS §21 rule 10):
//! - a pinned host buffer comes from `turbine_host_alloc_pinned` on one context and is owned by
//!   exactly one `PinnedBuffer`, whose Drop frees it through that context
//!   (`PinnedOwner::free_pinned`); the buffer holds an `Arc` of the context, so the context
//!   outlives every buffer;
//! - a buffer's bytes are reached only through `with_bytes_dyn`, one caller at a time (a
//!   per-buffer lock), and never while a `CopyTicket` reading or writing it is outstanding
//!   (the `PinnedOwner` contract: callers poll or wait the ticket first);
//! - the copy stream and the events of outstanding tickets are owned by the context and
//!   destroyed in `ShimContext::drop`, before `turbine_ctx_destroy` (destroying the copy stream
//!   waits for its copies, so no copy outlives the buffers it reads or writes).
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, MutexGuard};

use turbine_tensor::{
    CopyEngine, CopySource, CopyTarget, CopyTicket, MemoryError, PinnedBuffer, PinnedMemory,
    PinnedOwner,
};

use crate::KernelError;
use crate::ffi::{
    self, COPY_D2D, COPY_D2H, COPY_H2D, CopyFns, StagingFns, TurbineCtx, TurbineEvent,
    TurbineStream,
};
use crate::shim::ShimContext;

/// A copy stream (`turbine_stream*`) of one context, destroyed on Drop through that context.
struct ShimStream {
    raw: *mut TurbineStream,
    ctx: *mut TurbineCtx,
    destroy: unsafe extern "C" fn(*mut TurbineCtx, *mut TurbineStream) -> i32,
}

// SAFETY: the stream pointer is never dereferenced on the Rust side; it is only passed to the
// shim, whose entry points may be called from any thread (contract §9.4).
unsafe impl Send for ShimStream {}

impl Drop for ShimStream {
    fn drop(&mut self) {
        // SAFETY: `raw` came from `turbine_copy_stream_create` on `ctx` and is destroyed exactly
        // once, here (the shim waits for its copies first). Streams live only inside their
        // context's `PinnedState`, which `ShimContext::drop` empties before
        // `turbine_ctx_destroy`, so `ctx` is still alive.
        let code = unsafe { (self.destroy)(self.ctx, self.raw) };
        if code != 0 {
            tracing::error!(event = "kernel_copy_stream_destroy_failed", code);
        }
    }
}

/// A copy-completion event (`turbine_event*`) of one context, destroyed on Drop through it.
struct ShimEvent {
    raw: *mut TurbineEvent,
    ctx: *mut TurbineCtx,
    destroy: unsafe extern "C" fn(*mut TurbineCtx, *mut TurbineEvent) -> i32,
}

// SAFETY: as for `ShimStream`: only passed to thread-safe shim entry points.
unsafe impl Send for ShimEvent {}

impl Drop for ShimEvent {
    fn drop(&mut self) {
        // SAFETY: `raw` came from `turbine_event_create` on `ctx` and is destroyed exactly once,
        // here; events live only inside their context's `PinnedState` or on the stack of a call
        // on that context (see `ShimStream`). The runtime keeps a recorded event's resources
        // until its work completes, so destroying it early never cancels a wait.
        let code = unsafe { (self.destroy)(self.ctx, self.raw) };
        if code != 0 {
            tracing::error!(event = "kernel_event_destroy_failed", code);
        }
    }
}

/// One registered pinned allocation.
struct PinnedAlloc {
    addr: usize,
    len: usize,
    /// Serialises `with_bytes_dyn` on this buffer.
    access: Arc<Mutex<()>>,
}

#[derive(Default)]
struct PinnedInner {
    next_buffer: u64,
    buffers: HashMap<u64, PinnedAlloc>,
    stream: Option<ShimStream>,
    next_ticket: u64,
    /// Events of tickets not yet seen complete.
    tickets: HashMap<u64, ShimEvent>,
    /// The hostmem copy-engine all-reduce's own copy stream and fence event (ABI v2.8), apart
    /// from the KV copy stream so tier copies never delay a collective.
    collective: Option<(ShimStream, ShimEvent)>,
    /// The ring of side-stream marks (tensor-parallel prefill overlap, P5 Task 32).
    side_events: Vec<ShimEvent>,
    side_next: u64,
}

/// The Phase 4 pinned-memory state of one `ShimContext`.
#[derive(Default)]
pub(crate) struct PinnedState {
    inner: Mutex<PinnedInner>,
}

impl PinnedState {
    // Every update is a single insert or remove, so a lock poisoned by a panicking caller holds
    // consistent state and is safe to reuse.
    fn lock(&self) -> MutexGuard<'_, PinnedInner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Destroys the events of outstanding tickets and the copy stream (which waits for its
    /// copies); `ShimContext::drop` calls it before destroying the context.
    pub(crate) fn release(&self) {
        let mut s = self.lock();
        s.tickets.clear();
        s.stream = None;
        s.side_events.clear();
        s.collective = None;
    }
}

/// A resolved end of a copy: its address and whether it is device memory.
struct End {
    addr: usize,
    device: bool,
}

impl ShimContext {
    /// Both optional groups the Phase 4 copy engine needs: v2.3 (pinned memory, events) and v2.5
    /// (copy streams, asynchronous copies, event query and stream waits).
    pub fn has_copy_engine(&self) -> bool {
        self.copy_fns().is_ok()
    }

    fn copy_fns(&self) -> Result<(StagingFns, CopyFns), MemoryError> {
        let syms = &self.library().syms().v21;
        match (syms.staging, syms.copies) {
            (Some(staging), Some(copies)) => Ok((staging, copies)),
            _ => Err(MemoryError::Unsupported(format!(
                "{} does not export the kernel ABI v2.3 and v2.5 pinned-memory and copy-stream \
                 functions (minor {})",
                self.library().path().display(),
                syms.minor
            ))),
        }
    }

    fn resolve_end(
        s: &PinnedInner,
        what: &str,
        end: CopyTarget,
        bytes: usize,
    ) -> Result<End, MemoryError> {
        match end {
            CopyTarget::Device(p) => Ok(End {
                addr: p.addr() as usize,
                device: true,
            }),
            CopyTarget::Pinned { buffer_id, offset } => {
                let b = s.buffers.get(&buffer_id).ok_or_else(|| {
                    MemoryError::InvalidArgument(format!(
                        "{what}: pinned buffer {buffer_id} is not allocated on this context"
                    ))
                })?;
                if offset.checked_add(bytes).is_none_or(|end| end > b.len) {
                    return Err(MemoryError::InvalidArgument(format!(
                        "{what}: {bytes} bytes at offset {offset} exceed pinned buffer \
                         {buffer_id} of {} bytes",
                        b.len
                    )));
                }
                Ok(End {
                    addr: b.addr + offset,
                    device: false,
                })
            }
        }
    }

    /// The context's copy stream, created on first use.
    fn copy_stream(
        &self,
        copies: &CopyFns,
        s: &mut PinnedInner,
    ) -> Result<*mut TurbineStream, KernelError> {
        if let Some(stream) = &s.stream {
            return Ok(stream.raw);
        }
        let mut raw: *mut TurbineStream = std::ptr::null_mut();
        // SAFETY: `raw` is a live out-pointer; on success the stream belongs to this context
        // and is owned by the `ShimStream` stored below, destroyed before the context.
        let code = unsafe { (copies.stream_create)(self.raw_ctx(), &mut raw) };
        self.check_code(code)?;
        s.stream = Some(ShimStream {
            raw,
            ctx: self.raw_ctx(),
            destroy: copies.stream_destroy,
        });
        Ok(raw)
    }

    /// A new event of this context, owned by the returned `ShimEvent`.
    fn new_event(&self, staging: &StagingFns) -> Result<ShimEvent, KernelError> {
        let mut raw: *mut TurbineEvent = std::ptr::null_mut();
        // SAFETY: `raw` is a live out-pointer; the event is owned by the `ShimEvent` returned.
        let code = unsafe { (staging.event_create)(self.raw_ctx(), &mut raw) };
        self.check_code(code)?;
        Ok(ShimEvent {
            raw,
            ctx: self.raw_ctx(),
            destroy: staging.event_destroy,
        })
    }

    /// Makes work enqueued on `stream` from now on wait for everything already enqueued on the
    /// compute stream (an event recorded there, waited on by `stream`).
    fn order_after_compute(
        &self,
        staging: &StagingFns,
        copies: &CopyFns,
        stream: *mut TurbineStream,
    ) -> Result<(), KernelError> {
        let fence = self.new_event(staging)?;
        // SAFETY: `fence` is a live event of this context; a null stream is the compute stream.
        let code =
            unsafe { (staging.event_record)(self.raw_ctx(), fence.raw, std::ptr::null_mut()) };
        self.check_code(code)?;
        // SAFETY: `stream` is this context's live copy stream and `fence` was recorded above.
        let code = unsafe { (copies.stream_wait_event)(self.raw_ctx(), stream, fence.raw) };
        self.check_code(code)
    }

    fn check_code(&self, code: i32) -> Result<(), KernelError> {
        ffi::check(code, self.library().syms(), self.raw_ctx())
    }

    /// The copy stream and event of the hostmem copy-engine all-reduce (ABI v2.8), created on
    /// first use and owned by this context (destroyed with its other copy resources).
    pub(crate) fn collective_copy_resources(
        &self,
    ) -> Result<(*mut TurbineStream, *mut TurbineEvent), MemoryError> {
        let (staging, copies) = self.copy_fns()?;
        let mut s = self.pinned_state().lock();
        if let Some((stream, event)) = &s.collective {
            return Ok((stream.raw, event.raw));
        }
        let mut raw: *mut TurbineStream = std::ptr::null_mut();
        // SAFETY: `raw` is a live out-pointer; on success the stream belongs to this context
        // and is owned by the `ShimStream` stored below, destroyed before the context.
        let code = unsafe { (copies.stream_create)(self.raw_ctx(), &mut raw) };
        self.check_code(code)?;
        let stream = ShimStream {
            raw,
            ctx: self.raw_ctx(),
            destroy: copies.stream_destroy,
        };
        let event = self.new_event(&staging)?;
        let out = (stream.raw, event.raw);
        s.collective = Some((stream, event));
        Ok(out)
    }

    /// Makes work enqueued on the collective side stream ([`Self::collective_copy_resources`])
    /// from now on wait for everything already enqueued on the compute stream.
    pub(crate) fn side_after_compute(&self) -> Result<(), MemoryError> {
        let (staging, copies) = self.copy_fns()?;
        let (stream, _) = self.collective_copy_resources()?;
        Ok(self.order_after_compute(&staging, &copies, stream)?)
    }

    /// Records an event after everything enqueued on the side stream so far and returns its
    /// mark for [`Self::compute_after_mark`]. The events come from a ring of
    /// [`SIDE_MARKS`]: a mark older than that waits for a later point of the side stream (safe,
    /// only later).
    pub(crate) fn side_mark(&self) -> Result<u64, MemoryError> {
        let (staging, _) = self.copy_fns()?;
        let (stream, _) = self.collective_copy_resources()?;
        let mut s = self.pinned_state().lock();
        if s.side_events.len() < SIDE_MARKS {
            let e = self.new_event(&staging)?;
            s.side_events.push(e);
        }
        let mark = s.side_next;
        s.side_next += 1;
        let slot = (mark % SIDE_MARKS as u64) as usize;
        let slot = slot.min(s.side_events.len() - 1);
        // SAFETY: the event and the stream are this context's live objects (owned by its
        // pinned state until the context is destroyed).
        let code =
            unsafe { (staging.event_record)(self.raw_ctx(), s.side_events[slot].raw, stream) };
        self.check_code(code)?;
        Ok(mark)
    }

    /// Makes work enqueued on the compute stream from now on wait for the side stream's work up
    /// to `mark` ([`Self::side_mark`]).
    pub(crate) fn compute_after_mark(&self, mark: u64) -> Result<(), MemoryError> {
        let (_, copies) = self.copy_fns()?;
        let s = self.pinned_state().lock();
        let slot = ((mark % SIDE_MARKS as u64) as usize).min(s.side_events.len().saturating_sub(1));
        let Some(event) = s.side_events.get(slot) else {
            return Err(MemoryError::InvalidArgument(format!("no side mark {mark}")));
        };
        // SAFETY: a null stream is the compute stream; the event is this context's and was
        // recorded by `side_mark`.
        let code =
            unsafe { (copies.stream_wait_event)(self.raw_ctx(), std::ptr::null_mut(), event.raw) };
        Ok(self.check_code(code)?)
    }
}

/// Events in the side-stream mark ring ([`ShimContext::side_mark`]).
const SIDE_MARKS: usize = 16;

impl PinnedMemory for ShimContext {
    fn alloc_pinned(&self, bytes: usize) -> Result<PinnedBuffer, MemoryError> {
        let (staging, _) = self.copy_fns()?;
        let mut out: *mut c_void = std::ptr::null_mut();
        // SAFETY: `out` is a live out-pointer; on success the page-locked allocation is owned by
        // the `PinnedBuffer` returned below and freed once through `free_pinned`.
        let code = unsafe { (staging.host_alloc)(self.raw_ctx(), bytes, &mut out) };
        match self.check_code(code) {
            Ok(()) => {}
            Err(KernelError::OutOfMemory { .. }) => {
                return Err(MemoryError::OutOfMemory {
                    requested: bytes as u64,
                });
            }
            Err(e) => return Err(e.into()),
        }
        if out.is_null() {
            return Err(MemoryError::Device {
                message: "turbine_host_alloc_pinned succeeded but returned NULL".into(),
                sticky: false,
            });
        }
        let id = {
            let mut s = self.pinned_state().lock();
            s.next_buffer += 1;
            let id = s.next_buffer;
            s.buffers.insert(
                id,
                PinnedAlloc {
                    addr: out as usize,
                    len: bytes,
                    access: Arc::new(Mutex::new(())),
                },
            );
            id
        };
        let owner: Arc<dyn PinnedOwner> = self.owning_arc();
        Ok(PinnedBuffer::new(id, bytes, owner))
    }
}

impl PinnedOwner for ShimContext {
    fn with_bytes_dyn(&self, id: u64, f: &mut dyn FnMut(&mut [u8])) {
        let (addr, len, access) = {
            let s = self.pinned_state().lock();
            let b = s
                .buffers
                .get(&id)
                .unwrap_or_else(|| panic!("pinned buffer {id} is not live on this context"));
            (b.addr, b.len, Arc::clone(&b.access))
        };
        let _guard = access.lock().unwrap_or_else(|p| p.into_inner());
        // SAFETY: `addr..addr + len` is the live pinned allocation `id`: it stays allocated until
        // its `PinnedBuffer` is dropped, which cannot happen during this call (the caller
        // borrows it). The per-buffer lock held above makes this the only slice of it, and the
        // `PinnedOwner` contract rules out an outstanding copy on it.
        let bytes = unsafe { std::slice::from_raw_parts_mut(addr as *mut u8, len) };
        f(bytes);
    }

    fn free_pinned(&self, id: u64) {
        let Some(b) = self.pinned_state().lock().buffers.remove(&id) else {
            return;
        };
        let Ok((staging, _)) = self.copy_fns() else {
            return;
        };
        // SAFETY: `b.addr` came from `turbine_host_alloc_pinned` on this context and is freed
        // exactly once, here (removed from the registry above); its `PinnedBuffer` is dropping.
        let code = unsafe { (staging.host_free)(self.raw_ctx(), b.addr as *mut c_void) };
        if let Err(e) = self.check_code(code) {
            tracing::error!(event = "kernel_pinned_free_failed", error = %e);
        }
    }
}

impl CopyEngine for ShimContext {
    fn copy_async(
        &self,
        dst: CopyTarget,
        src: CopySource,
        bytes: usize,
    ) -> Result<CopyTicket, MemoryError> {
        let (staging, copies) = self.copy_fns()?;
        let mut s = self.pinned_state().lock();
        let d = Self::resolve_end(&s, "copy destination", dst, bytes)?;
        let from = Self::resolve_end(&s, "copy source", src, bytes)?;
        let kind = match (d.device, from.device) {
            (true, false) => COPY_H2D,
            (false, true) => COPY_D2H,
            (true, true) => COPY_D2D,
            (false, false) => {
                return Err(MemoryError::Unsupported(
                    "pinned-to-pinned copies are host memcpy, not copy-stream work".into(),
                ));
            }
        };
        let stream = self.copy_stream(&copies, &mut s)?;
        if from.device {
            // A device source may still be written by work already on the compute stream (the
            // forward pass that filled a KV block): the copy waits for it.
            self.order_after_compute(&staging, &copies, stream)?;
        }
        // The event exists before the copy is enqueued, so a failure to create it leaves no
        // copy in flight without a ticket.
        let event = self.new_event(&staging)?;
        // SAFETY: both ends were resolved above: a device pointer of the caller's allocation or
        // `bytes` inside a live pinned buffer of this context. The copy only enqueues; the
        // buffers are not freed or touched by the host until the ticket's event signals (the
        // `PinnedOwner` contract; `PinnedBuffer`s are dropped only after their tickets complete).
        let code = unsafe {
            (copies.memcpy_async)(
                self.raw_ctx(),
                stream,
                d.addr as *mut c_void,
                from.addr as *const c_void,
                bytes,
                kind,
            )
        };
        self.check_code(code)?;
        // SAFETY: `event` and `stream` are live objects of this context. A failure here is a
        // device error that leaves the context unusable (the copy stays enqueued without a
        // ticket and the caller's next device call fails too).
        let code = unsafe { (staging.event_record)(self.raw_ctx(), event.raw, stream) };
        self.check_code(code)?;
        s.next_ticket += 1;
        let id = s.next_ticket;
        s.tickets.insert(id, event);
        Ok(CopyTicket { id, bytes })
    }

    fn poll(&self, t: &CopyTicket) -> Result<bool, MemoryError> {
        let (_, copies) = self.copy_fns()?;
        let mut s = self.pinned_state().lock();
        let Some(event) = s.tickets.get(&t.id) else {
            return finished_or_unknown(&s, t);
        };
        // SAFETY: `event` is a live event of this context, recorded after the ticket's copy.
        let code = unsafe { (copies.event_query)(self.raw_ctx(), event.raw) };
        match code {
            1 => {
                s.tickets.remove(&t.id);
                Ok(true)
            }
            0 => Ok(false),
            err => Err(self.check_code(err).err().map_or_else(
                || MemoryError::Device {
                    message: format!("turbine_event_query returned {err}"),
                    sticky: false,
                },
                MemoryError::from,
            )),
        }
    }

    fn wait(&self, t: &CopyTicket) -> Result<(), MemoryError> {
        let (staging, _) = self.copy_fns()?;
        // Taken out of the table first, so other copies proceed while this one is awaited.
        let event = {
            let mut s = self.pinned_state().lock();
            match s.tickets.remove(&t.id) {
                Some(e) => e,
                None => return finished_or_unknown(&s, t).map(|_| ()),
            }
        };
        // SAFETY: `event` is a live event of this context; it is destroyed when dropped below.
        let code = unsafe { (staging.event_synchronize)(self.raw_ctx(), event.raw) };
        self.check_code(code)?;
        Ok(())
    }
}

/// A ticket without an event completed earlier; an id never issued is a caller error.
fn finished_or_unknown(s: &PinnedInner, t: &CopyTicket) -> Result<bool, MemoryError> {
    if t.id >= 1 && t.id <= s.next_ticket {
        Ok(true)
    } else {
        Err(MemoryError::InvalidArgument(format!(
            "copy ticket {} was not issued by this context",
            t.id
        )))
    }
}
