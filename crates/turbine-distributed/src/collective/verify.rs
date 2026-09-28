//! `parallel.collective.verify` (P5 Task 32 corruption diagnosis; default off): a diagnosis and
//! canary mode that checks every collective whose output must be identical on every rank —
//! all-reduce, all-gather, broadcast — after it completes. Each rank synchronizes its stream,
//! reads the output back (through the device's pinned bounce buffer), hashes it (FNV-1a, 64
//! bits) and all-gathers the hashes over the same communicator; ranks that disagree fail the
//! step with reason code `collective_corrupt` (`CollectiveError::Backend`, so the step's
//! failure path and the circuit run), counted in `turbine_collective_errors_total{kind=corrupt}`
//! and logged as a WARN `event=collective_corrupt` with every rank's hash. Reduce-scatter,
//! send / recv and barriers pass through (their outputs differ by rank). Calls issued while the
//! stream is captured into a graph are not checked (a read-back cannot be captured).
//!
//! Cost: a stream synchronize, a device-to-host read of the output and one 8-byte all-gather
//! per checked call, so it is for diagnosis runs, not for serving throughput.
use std::sync::Arc;

use turbine_tensor::{DType, DeviceBuffer, DeviceSlice, StreamRef};

use super::{
    Collective, CollectiveError, CollectiveErrorKind, CollectiveInit, CollectiveLibrary,
    CollectiveMetrics, ReduceOp,
};

/// FNV-1a over `bytes`.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// A library whose communicators are [`VerifyingCollective`]s over `inner`'s.
pub struct VerifyingLibrary {
    inner: Arc<dyn CollectiveLibrary>,
}

impl VerifyingLibrary {
    pub fn wrap(inner: Arc<dyn CollectiveLibrary>) -> Arc<dyn CollectiveLibrary> {
        Arc::new(VerifyingLibrary { inner })
    }
}

impl CollectiveLibrary for VerifyingLibrary {
    fn backend(&self) -> &'static str {
        self.inner.backend()
    }
    fn version(&self) -> Option<String> {
        self.inner.version()
    }
    fn unique_id(&self) -> Result<[u8; super::UNIQUE_ID_BYTES], CollectiveError> {
        self.inner.unique_id()
    }
    fn open(self: Arc<Self>, init: CollectiveInit) -> Result<Arc<dyn Collective>, CollectiveError> {
        let metrics = init.metrics.clone();
        let inner = Arc::clone(&self.inner).open(init)?;
        Ok(Arc::new(VerifyingCollective { inner, metrics }))
    }
}

/// One rank's communicator with its outputs checked across ranks (module docs).
pub struct VerifyingCollective {
    inner: Arc<dyn Collective>,
    metrics: Option<CollectiveMetrics>,
}

impl VerifyingCollective {
    /// A checking communicator over `inner` (tests; the server wraps the library).
    pub fn new(inner: Arc<dyn Collective>, metrics: Option<CollectiveMetrics>) -> Self {
        VerifyingCollective { inner, metrics }
    }

    /// Checks that `out` holds the same bytes on every rank (after `op` completed on `stream`).
    fn check(
        &self,
        op: &'static str,
        out: &DeviceSlice,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        let mem = stream.memory();
        if mem
            .mapped_collectives()
            .is_some_and(|m| m.mapped_capturing())
        {
            return Ok(());
        }
        let io = |what: &str, e: &dyn std::fmt::Display| CollectiveError::Backend {
            code: -1,
            message: format!("collective verify: {what}: {e}"),
        };
        let bytes = out.read_bytes().map_err(|e| io("reading the output", &e))?;
        let hash = fnv1a(&bytes);
        let world = self.inner.world_size();
        let mine = DeviceBuffer::alloc(mem, 8).map_err(|e| io("allocating", &e))?;
        let all = DeviceBuffer::alloc(mem, 8 * world).map_err(|e| io("allocating", &e))?;
        mine.whole()
            .write_bytes(&hash.to_le_bytes())
            .map_err(|e| io("writing the hash", &e))?;
        self.inner
            .all_gather(&mine.whole(), &mut all.whole(), stream)?;
        mem.synchronize().map_err(|e| io("synchronizing", &e))?;
        let hashes: Vec<u64> = all
            .whole()
            .read_bytes()
            .map_err(|e| io("reading the hashes", &e))?
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes")))
            .collect();
        if hashes.iter().all(|&h| h == hashes[0]) {
            return Ok(());
        }
        let backend = self.inner.backend();
        if let Some(m) = &self.metrics {
            m.error(backend, CollectiveErrorKind::Corrupt);
        }
        tracing::warn!(
            event = "collective_corrupt",
            op,
            backend,
            rank = self.inner.rank(),
            bytes = bytes.len(),
            hashes = ?hashes,
            "the ranks hold different outputs of one collective"
        );
        Err(CollectiveError::Backend {
            code: -1,
            message: format!(
                "collective_corrupt: {op} of {} bytes differs across ranks (hashes {hashes:x?})",
                bytes.len()
            ),
        })
    }
}

impl Collective for VerifyingCollective {
    fn backend(&self) -> &'static str {
        self.inner.backend()
    }
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn world_size(&self) -> usize {
        self.inner.world_size()
    }
    fn all_reduce(
        &self,
        buf: &mut DeviceSlice,
        dtype: DType,
        op: ReduceOp,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        self.inner.all_reduce(buf, dtype, op, stream)?;
        self.check("all_reduce", buf, stream)
    }
    fn all_gather(
        &self,
        send: &DeviceSlice,
        recv: &mut DeviceSlice,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        self.inner.all_gather(send, recv, stream)?;
        self.check("all_gather", recv, stream)
    }
    fn reduce_scatter(
        &self,
        send: &DeviceSlice,
        recv: &mut DeviceSlice,
        dtype: DType,
        op: ReduceOp,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        self.inner.reduce_scatter(send, recv, dtype, op, stream)
    }
    fn broadcast(
        &self,
        buf: &mut DeviceSlice,
        root: usize,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        self.inner.broadcast(buf, root, stream)?;
        self.check("broadcast", buf, stream)
    }
    fn barrier(&self, stream: &StreamRef) -> Result<(), CollectiveError> {
        self.inner.barrier(stream)
    }
    fn send(
        &self,
        buf: &DeviceSlice,
        peer: usize,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        self.inner.send(buf, peer, stream)
    }
    fn recv(
        &self,
        buf: &mut DeviceSlice,
        peer: usize,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        self.inner.recv(buf, peer, stream)
    }
    fn step_begin(&self) {
        self.inner.step_begin();
    }
    fn step_end(&self) -> Result<(), CollectiveError> {
        self.inner.step_end()
    }
    fn abort(&self) {
        self.inner.abort();
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use turbine_tensor::host::HostMemory;
    use turbine_tensor::{DeviceId, DeviceMemory};

    use super::*;
    use crate::collective::HostCollective;

    /// A communicator that flips one output byte on one rank after every all-reduce: the
    /// injected corruption.
    struct Corrupting {
        inner: Arc<dyn Collective>,
        rank: usize,
    }

    impl Collective for Corrupting {
        fn backend(&self) -> &'static str {
            self.inner.backend()
        }
        fn rank(&self) -> usize {
            self.inner.rank()
        }
        fn world_size(&self) -> usize {
            self.inner.world_size()
        }
        fn all_reduce(
            &self,
            buf: &mut DeviceSlice,
            dtype: DType,
            op: ReduceOp,
            stream: &StreamRef,
        ) -> Result<(), CollectiveError> {
            self.inner.all_reduce(buf, dtype, op, stream)?;
            if self.inner.rank() == self.rank {
                let mut bytes = buf.read_bytes().expect("read");
                bytes[0] ^= 1;
                buf.write_bytes(&bytes).expect("write");
            }
            Ok(())
        }
        fn all_gather(
            &self,
            send: &DeviceSlice,
            recv: &mut DeviceSlice,
            stream: &StreamRef,
        ) -> Result<(), CollectiveError> {
            self.inner.all_gather(send, recv, stream)
        }
        fn reduce_scatter(
            &self,
            send: &DeviceSlice,
            recv: &mut DeviceSlice,
            dtype: DType,
            op: ReduceOp,
            stream: &StreamRef,
        ) -> Result<(), CollectiveError> {
            self.inner.reduce_scatter(send, recv, dtype, op, stream)
        }
        fn broadcast(
            &self,
            buf: &mut DeviceSlice,
            root: usize,
            stream: &StreamRef,
        ) -> Result<(), CollectiveError> {
            self.inner.broadcast(buf, root, stream)
        }
        fn barrier(&self, stream: &StreamRef) -> Result<(), CollectiveError> {
            self.inner.barrier(stream)
        }
        fn send(
            &self,
            buf: &DeviceSlice,
            peer: usize,
            stream: &StreamRef,
        ) -> Result<(), CollectiveError> {
            self.inner.send(buf, peer, stream)
        }
        fn recv(
            &self,
            buf: &mut DeviceSlice,
            peer: usize,
            stream: &StreamRef,
        ) -> Result<(), CollectiveError> {
            self.inner.recv(buf, peer, stream)
        }
        fn abort(&self) {
            self.inner.abort();
        }
    }

    /// Two ranks on the host backend: a clean all-reduce, all-gather and broadcast pass the
    /// check; with one byte flipped on rank 1 after the all-reduce, both ranks fail the call with
    /// `collective_corrupt` and the corrupt-error counter moves. Breaks if the check compares
    /// the wrong buffer, hashes only one rank or lets the mismatch through.
    #[test]
    fn verify_detects_a_corrupt_rank() {
        let reg = turbine_observability::MetricsRegistry::new();
        let metrics = CollectiveMetrics::register(&reg);
        for corrupt in [false, true] {
            let group = HostCollective::group(2, Duration::from_secs(20));
            let results: Vec<Result<(), CollectiveError>> = std::thread::scope(|s| {
                let handles: Vec<_> = group
                    .into_iter()
                    .enumerate()
                    .map(|(r, c)| {
                        let metrics = metrics.clone();
                        s.spawn(move || {
                            let inner: Arc<dyn Collective> = if corrupt {
                                Arc::new(Corrupting {
                                    inner: Arc::new(c),
                                    rank: 1,
                                })
                            } else {
                                Arc::new(c)
                            };
                            let comm = VerifyingCollective::new(inner, Some(metrics));
                            let mem: Arc<dyn DeviceMemory> =
                                HostMemory::new(DeviceId(r as u32), 1 << 20);
                            let stream = mem.compute_stream();
                            let buf = DeviceBuffer::alloc(&mem, 64).expect("alloc");
                            buf.whole().write_bytes(&[r as u8 + 1; 64]).expect("write");
                            comm.all_reduce(&mut buf.whole(), DType::F32, ReduceOp::Sum, &stream)?;
                            let out = DeviceBuffer::alloc(&mem, 128).expect("alloc");
                            comm.all_gather(&buf.whole(), &mut out.whole(), &stream)?;
                            comm.broadcast(&mut buf.whole(), 0, &stream)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("rank"))
                    .collect()
            });
            for (r, res) in results.iter().enumerate() {
                match (corrupt, res) {
                    (false, Ok(())) => {}
                    (true, Err(CollectiveError::Backend { message, .. }))
                        if message.starts_with("collective_corrupt: all_reduce") => {}
                    other => panic!("rank {r}, corrupt {corrupt}: {other:?}"),
                }
            }
        }
        let text = reg.render().expect("renders");
        assert!(text.contains("kind=\"corrupt\""), "{text}");
    }
}
