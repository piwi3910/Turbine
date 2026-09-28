//! The startup host-link probe (P5 S-13) over a device's pinned host memory and copy stream
//! (kernel ABI v2.3 + v2.5): times bounded host↔device copies so the topology graph can carry
//! measured bandwidth ([`turbine_device::topology::apply_link_probe`]).

use std::sync::Arc;
use std::time::Instant;

use turbine_core::types::DeviceId;
use turbine_device::topology::{LinkBandwidth, LinkProbe};
use turbine_tensor::{CopyEngine, CopyTarget, DeviceBuffer, DeviceMemory, PinnedMemory};

/// Timed repetitions per direction; the median is reported.
const REPEATS: usize = 3;

/// One device's memory, copy stream and pinned host allocator.
pub struct ProbeTarget {
    pub memory: Arc<dyn DeviceMemory>,
    pub copies: Arc<dyn CopyEngine>,
    pub pinned: Arc<dyn PinnedMemory>,
}

/// Measures one device's host link: one untimed copy each way, then [`REPEATS`] timed copies of
/// `bytes` each way (enqueue to completion); the median of each direction, in GB/s.
pub fn measure_host_link(t: &ProbeTarget, bytes: usize) -> Result<LinkBandwidth, String> {
    let host = t
        .pinned
        .alloc_pinned(bytes)
        .map_err(|e| format!("pinned host buffer of {bytes} B: {e}"))?;
    let dev = DeviceBuffer::alloc(&t.memory, bytes)
        .map_err(|e| format!("device buffer of {bytes} B: {e}"))?;
    let h = || CopyTarget::Pinned {
        buffer_id: host.id(),
        offset: 0,
    };
    let d = || CopyTarget::Device(dev.ptr());
    let time = |dst: CopyTarget, src: CopyTarget| -> Result<f64, String> {
        let started = Instant::now();
        let ticket = t
            .copies
            .copy_async(dst, src, bytes)
            .map_err(|e| format!("copy of {bytes} B: {e}"))?;
        t.copies
            .wait(&ticket)
            .map_err(|e| format!("copy wait: {e}"))?;
        Ok(started.elapsed().as_secs_f64())
    };
    time(d(), h())?;
    time(h(), d())?;
    let run = |dst: &dyn Fn() -> CopyTarget, src: &dyn Fn() -> CopyTarget| {
        let mut secs = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            secs.push(time(dst(), src())?);
        }
        secs.sort_by(f64::total_cmp);
        let s = secs[REPEATS / 2].max(1e-9);
        Ok::<f64, String>(bytes as f64 / s / 1e9)
    };
    let h2d_gbps = run(&d, &h)?;
    let d2h_gbps = run(&h, &d)?;
    Ok(LinkBandwidth { h2d_gbps, d2h_gbps })
}

/// A [`LinkProbe`] over targets the caller opens per device (the server's contexts, or a lab
/// test's).
pub struct CopyLinkProbe<F: FnMut(DeviceId) -> Result<ProbeTarget, String>> {
    open: F,
}

impl<F: FnMut(DeviceId) -> Result<ProbeTarget, String>> CopyLinkProbe<F> {
    pub fn new(open: F) -> Self {
        CopyLinkProbe { open }
    }
}

impl<F: FnMut(DeviceId) -> Result<ProbeTarget, String>> LinkProbe for CopyLinkProbe<F> {
    fn host_link(&mut self, device: DeviceId, bytes: u64) -> Result<LinkBandwidth, String> {
        let target = (self.open)(device)?;
        let bytes = usize::try_from(bytes).map_err(|e| e.to_string())?;
        measure_host_link(&target, bytes)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    use turbine_tensor::host::HostMemory;
    use turbine_tensor::{CopyTicket, MemoryError, PinnedBuffer, PinnedOwner};

    use super::*;

    /// A host "device" with a synchronous copy stream between its pinned buffers and its
    /// memory; counts the bytes each way.
    struct HostCtx {
        mem: Arc<dyn DeviceMemory>,
        bufs: Arc<Bufs>,
        h2d: AtomicU64,
        d2h: AtomicU64,
    }

    #[derive(Default)]
    struct Bufs {
        map: Mutex<HashMap<u64, Box<[u8]>>>,
        next: AtomicU64,
    }

    impl PinnedOwner for Bufs {
        fn with_bytes_dyn(&self, id: u64, f: &mut dyn FnMut(&mut [u8])) {
            f(self.map.lock().unwrap().get_mut(&id).expect("live buffer"));
        }

        fn free_pinned(&self, id: u64) {
            self.map.lock().unwrap().remove(&id);
        }
    }

    impl PinnedMemory for HostCtx {
        fn alloc_pinned(&self, bytes: usize) -> Result<PinnedBuffer, MemoryError> {
            let id = self.bufs.next.fetch_add(1, Ordering::Relaxed);
            let buf = vec![0u8; bytes].into_boxed_slice();
            self.bufs.map.lock().unwrap().insert(id, buf);
            Ok(PinnedBuffer::new(id, bytes, Arc::clone(&self.bufs) as _))
        }
    }

    impl CopyEngine for HostCtx {
        fn copy_async(
            &self,
            dst: CopyTarget,
            src: CopyTarget,
            bytes: usize,
        ) -> Result<CopyTicket, MemoryError> {
            let mut map = self.bufs.map.lock().unwrap();
            match (dst, src) {
                (CopyTarget::Pinned { buffer_id, offset }, CopyTarget::Device(ptr)) => {
                    let buf = map.get_mut(&buffer_id).expect("pinned buffer");
                    self.mem.copy_d2h(&mut buf[offset..offset + bytes], ptr)?;
                    self.d2h.fetch_add(bytes as u64, Ordering::Relaxed);
                }
                (CopyTarget::Device(ptr), CopyTarget::Pinned { buffer_id, offset }) => {
                    let buf = map.get(&buffer_id).expect("pinned buffer");
                    self.mem.copy_h2d(ptr, &buf[offset..offset + bytes])?;
                    self.h2d.fetch_add(bytes as u64, Ordering::Relaxed);
                }
                _ => return Err(MemoryError::InvalidArgument("unsupported copy".into())),
            }
            Ok(CopyTicket { id: 0, bytes })
        }

        fn poll(&self, _t: &CopyTicket) -> Result<bool, MemoryError> {
            Ok(true)
        }

        fn wait(&self, _t: &CopyTicket) -> Result<(), MemoryError> {
            Ok(())
        }
    }

    /// P5 S-13: the probe copies `bytes` each way once untimed and three times timed per
    /// direction and reports positive GB/s; a device the opener refuses is an `Err` naming it,
    /// and an allocation failure is an `Err` naming the buffer. Breaks if the probe skips a
    /// direction, copies more than its bound, or panics on a failure.
    #[test]
    fn host_probe_measures_both_directions() {
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 64 << 20);
        let ctx = Arc::new(HostCtx {
            mem: Arc::clone(&mem),
            bufs: Arc::default(),
            h2d: AtomicU64::new(0),
            d2h: AtomicU64::new(0),
        });
        let c = Arc::clone(&ctx);
        let mut probe = CopyLinkProbe::new(move |d: DeviceId| {
            if d.0 != 0 {
                return Err(format!("no context on device {}", d.0));
            }
            Ok(ProbeTarget {
                memory: Arc::clone(&c.mem),
                copies: Arc::clone(&c) as _,
                pinned: Arc::clone(&c) as _,
            })
        });
        let bytes = 1u64 << 20;
        let bw = probe.host_link(DeviceId(0), bytes).expect("probe");
        assert!(bw.h2d_gbps > 0.0 && bw.d2h_gbps > 0.0, "{bw:?}");
        assert_eq!(
            ctx.h2d.load(Ordering::Relaxed),
            4 * bytes,
            "1 warm-up + 3 timed"
        );
        assert_eq!(ctx.d2h.load(Ordering::Relaxed), 4 * bytes);
        assert!(
            ctx.bufs.map.lock().unwrap().is_empty(),
            "pinned buffer freed"
        );

        let e = probe.host_link(DeviceId(1), bytes).unwrap_err();
        assert!(e.contains("device 1"), "{e}");
        // Larger than the device's capacity: the device buffer allocation fails.
        let e = probe.host_link(DeviceId(0), 128 << 20).unwrap_err();
        assert!(e.contains("device buffer"), "{e}");
    }
}
