//! Host reference backend: ranks are threads of one process, buffers live in any
//! [`DeviceMemory`] reachable through `DeviceSlice::read_bytes` / `write_bytes` (normally
//! `turbine_tensor::host::HostMemory`), and every reduction runs in rank order 0..n on the last
//! arriving rank — so results are deterministic and bit-identical on every rank.
//!
//! Each operation is one rendezvous generation: every rank deposits its bytes, the last arrival
//! computes the result and releases the others, and the generation closes when every rank has
//! taken its part. Waits are bounded by the op timeout; the first rank to time out (or an
//! explicit [`Collective::abort`]) aborts the group, after which every call on every rank
//! returns [`CollectiveError::RemoteAbort`] naming that rank.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use half::bf16;
use turbine_core::registry::Module;
use turbine_core::types::Vendor;
use turbine_tensor::{DType, DeviceSlice, MemoryError, StreamRef};

use super::{
    Collective, CollectiveBackend, CollectiveError, CollectiveInit, CollectiveLibrary, ReduceOp,
    UNIQUE_ID_BYTES,
};

/// What the last arriving rank computes from every rank's deposit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum OpKind {
    AllReduce(ReduceOp),
    AllGather,
    ReduceScatter(ReduceOp),
    Broadcast(usize),
    Barrier,
}

impl OpKind {
    fn name(self) -> &'static str {
        match self {
            OpKind::AllReduce(_) => "all_reduce",
            OpKind::AllGather => "all_gather",
            OpKind::ReduceScatter(_) => "reduce_scatter",
            OpKind::Broadcast(_) => "broadcast",
            OpKind::Barrier => "barrier",
        }
    }
}

struct Deposit {
    op: OpKind,
    /// Element type of a reduction; `None` for the byte-wise gather, broadcast and barrier.
    dtype: Option<DType>,
    bytes: Vec<u8>,
    /// Bytes this rank expects back.
    recv_len: usize,
}

enum Phase {
    /// Ranks are depositing.
    Collecting,
    /// The result is ready; ranks are taking their part.
    Releasing(Result<Arc<Vec<u8>>, ()>),
}

struct State {
    generation: u64,
    phase: Phase,
    deposits: Vec<Option<Deposit>>,
    arrived: usize,
    departed: usize,
    /// The rank whose timeout or abort poisoned the group.
    aborted_by: Option<usize>,
}

struct Shared {
    world: usize,
    op_timeout: Duration,
    state: Mutex<State>,
    cv: Condvar,
}

impl Shared {
    // The state is updated only in whole steps under the lock, so a panicking rank thread
    // (a test failure) leaves it consistent enough for the others to fail cleanly.
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }
}

/// One rank of a host communicator. Create a whole group with [`HostCollective::group`].
pub struct HostCollective {
    rank: usize,
    shared: Arc<Shared>,
}

impl Shared {
    fn new(world: usize, op_timeout: Duration) -> Arc<Shared> {
        assert!(world > 0, "a collective group needs at least one rank");
        Arc::new(Shared {
            world,
            op_timeout,
            state: Mutex::new(State {
                generation: 0,
                phase: Phase::Collecting,
                deposits: (0..world).map(|_| None).collect(),
                arrived: 0,
                departed: 0,
                aborted_by: None,
            }),
            cv: Condvar::new(),
        })
    }
}

/// The `host` module of the `collective_backend` registry: ranks are threads of this process
/// that open the same unique id. No library, no device vendor (host memory only).
pub struct HostBackend;

impl Module for HostBackend {
    fn name(&self) -> &'static str {
        "host"
    }
}

impl CollectiveBackend for HostBackend {
    fn vendors(&self) -> &'static [Vendor] {
        &[]
    }
    fn load(
        &self,
        _explicit: Option<&Path>,
    ) -> Result<Arc<dyn CollectiveLibrary>, CollectiveError> {
        Ok(Arc::new(HostLibrary))
    }
}

/// Groups being opened, by unique id: the first rank to open an id creates its rendezvous and
/// the entry leaves the table once every rank has joined. Process-local.
static JOINING: Mutex<Vec<Joining>> = Mutex::new(Vec::new());
/// A group being opened: its id, its rendezvous and which ranks have joined.
type Joining = ([u8; UNIQUE_ID_BYTES], Arc<Shared>, Vec<bool>);
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

struct HostLibrary;

impl CollectiveLibrary for HostLibrary {
    fn backend(&self) -> &'static str {
        "host"
    }
    fn version(&self) -> Option<String> {
        None
    }
    /// Unique within this process (a counter plus the process id and the time).
    fn unique_id(&self) -> Result<[u8; UNIQUE_ID_BYTES], CollectiveError> {
        let mut id = [0u8; UNIQUE_ID_BYTES];
        let n = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        id[..8].copy_from_slice(&n.to_le_bytes());
        id[8..12].copy_from_slice(&std::process::id().to_le_bytes());
        id[12..20].copy_from_slice(&t.to_le_bytes());
        Ok(id)
    }
    /// Joins (or starts) the group of `init.unique_id`. Ranks wait for each other only inside
    /// the first operation, which the op timeout bounds.
    fn open(self: Arc<Self>, init: CollectiveInit) -> Result<Arc<dyn Collective>, CollectiveError> {
        if init.world == 0 || init.rank >= init.world {
            return Err(CollectiveError::ShapeMismatch);
        }
        let mut joining = JOINING.lock().unwrap_or_else(|p| p.into_inner());
        let at = match joining.iter().position(|(id, _, _)| *id == init.unique_id) {
            Some(at) => at,
            None => {
                joining.push((
                    init.unique_id,
                    Shared::new(init.world, init.op_timeout),
                    vec![false; init.world],
                ));
                joining.len() - 1
            }
        };
        let (_, shared, joined) = &mut joining[at];
        if shared.world != init.world || joined[init.rank] {
            return Err(CollectiveError::Backend {
                code: -1,
                message: format!(
                    "host group: rank {} of {} does not fit a group of {} (or joined twice)",
                    init.rank, init.world, shared.world
                ),
            });
        }
        joined[init.rank] = true;
        let rank = HostCollective {
            rank: init.rank,
            shared: Arc::clone(shared),
        };
        if joined.iter().all(|j| *j) {
            joining.swap_remove(at);
        }
        Ok(Arc::new(rank))
    }
}

impl HostCollective {
    /// `world` ranks sharing one rendezvous.
    pub fn group(world: usize, op_timeout: Duration) -> Vec<HostCollective> {
        let shared = Shared::new(world, op_timeout);
        (0..world)
            .map(|rank| HostCollective {
                rank,
                shared: Arc::clone(&shared),
            })
            .collect()
    }

    /// Deposits `bytes` for `op`, waits for every rank, and returns this rank's result bytes.
    fn exchange(
        &self,
        op: OpKind,
        dtype: Option<DType>,
        bytes: Vec<u8>,
        recv_len: usize,
    ) -> Result<Vec<u8>, CollectiveError> {
        let sh = &*self.shared;
        let deadline = Instant::now() + sh.op_timeout;
        let mut st = sh.lock();

        // Wait for the previous generation to close.
        loop {
            if let Some(rank) = st.aborted_by {
                return Err(CollectiveError::RemoteAbort { rank });
            }
            if matches!(st.phase, Phase::Collecting) && st.deposits[self.rank].is_none() {
                break;
            }
            st = self.wait(st, deadline, op)?;
        }

        let generation = st.generation;
        st.deposits[self.rank] = Some(Deposit {
            op,
            dtype,
            bytes,
            recv_len,
        });
        st.arrived += 1;
        if st.arrived == sh.world {
            let deposits: Vec<Deposit> = st.deposits.iter_mut().filter_map(Option::take).collect();
            st.phase = Phase::Releasing(compute(&deposits, sh.world).map(Arc::new));
            sh.cv.notify_all();
        }

        let result = loop {
            if let Some(rank) = st.aborted_by {
                return Err(CollectiveError::RemoteAbort { rank });
            }
            if st.generation == generation
                && let Phase::Releasing(r) = &st.phase
            {
                break r.clone();
            }
            st = self.wait(st, deadline, op)?;
        };

        st.departed += 1;
        if st.departed == sh.world {
            st.generation += 1;
            st.phase = Phase::Collecting;
            st.arrived = 0;
            st.departed = 0;
            sh.cv.notify_all();
        }
        drop(st);

        let full = result.map_err(|()| CollectiveError::ShapeMismatch)?;
        Ok(match op {
            OpKind::ReduceScatter(_) => {
                full[self.rank * recv_len..(self.rank + 1) * recv_len].to_vec()
            }
            _ => full.to_vec(),
        })
    }

    /// One bounded wait; on expiry aborts the group and reports the timeout.
    fn wait<'a>(
        &self,
        st: MutexGuard<'a, State>,
        deadline: Instant,
        op: OpKind,
    ) -> Result<MutexGuard<'a, State>, CollectiveError> {
        let sh = &*self.shared;
        let now = Instant::now();
        if now >= deadline {
            let mut st = st;
            st.aborted_by.get_or_insert(self.rank);
            sh.cv.notify_all();
            tracing::warn!(
                event = "collective_timeout",
                backend = "host",
                op = op.name(),
                rank = self.rank,
                after_ms = sh.op_timeout.as_millis() as u64,
                "collective timed out; communicator aborted"
            );
            return Err(CollectiveError::Timeout {
                op: op.name(),
                after: sh.op_timeout,
            });
        }
        let (st, _) = sh
            .cv
            .wait_timeout(st, deadline - now)
            .unwrap_or_else(|p| p.into_inner());
        Ok(st)
    }

    /// A reduction over whole BF16 or FP32 elements.
    fn check_elems(dtype: DType, len: usize) -> Result<(), CollectiveError> {
        if matches!(dtype, DType::BF16 | DType::F32) && len.is_multiple_of(dtype.size_bytes()) {
            Ok(())
        } else {
            Err(CollectiveError::ShapeMismatch)
        }
    }
}

/// Every rank's deposit agrees on op, dtype and sizes, or the whole generation is a mismatch.
fn compute(deposits: &[Deposit], world: usize) -> Result<Vec<u8>, ()> {
    let first = &deposits[0];
    let agree = deposits.iter().all(|d| {
        d.op == first.op
            && d.dtype == first.dtype
            && d.bytes.len() == first.bytes.len()
            && d.recv_len == first.recv_len
    });
    if !agree {
        return Err(());
    }
    let inputs: Vec<&[u8]> = deposits.iter().map(|d| d.bytes.as_slice()).collect();
    match first.op {
        OpKind::AllReduce(op) => Ok(reduce(first.dtype.ok_or(())?, &inputs, op)),
        OpKind::ReduceScatter(op) => {
            if first.bytes.len() != first.recv_len * world {
                return Err(());
            }
            Ok(reduce(first.dtype.ok_or(())?, &inputs, op))
        }
        OpKind::AllGather => {
            if first.recv_len != first.bytes.len() * world {
                return Err(());
            }
            Ok(inputs.concat())
        }
        OpKind::Broadcast(root) => inputs.get(root).map(|b| b.to_vec()).ok_or(()),
        OpKind::Barrier => Ok(Vec::new()),
    }
}

/// Element-wise reduction in rank order 0..n; BF16 accumulates in FP32 and rounds once.
fn reduce(dtype: DType, inputs: &[&[u8]], op: ReduceOp) -> Vec<u8> {
    let decode = |b: &[u8]| -> Vec<f32> {
        match dtype {
            DType::BF16 => b
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            _ => b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        }
    };
    let mut acc = decode(inputs[0]);
    for input in &inputs[1..] {
        for (a, x) in acc.iter_mut().zip(decode(input)) {
            *a = match op {
                ReduceOp::Sum => *a + x,
                ReduceOp::Max => {
                    if x > *a {
                        x
                    } else {
                        *a
                    }
                }
            };
        }
    }
    match dtype {
        DType::BF16 => acc
            .iter()
            .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
            .collect(),
        _ => acc.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

fn memory(e: MemoryError) -> CollectiveError {
    CollectiveError::Backend {
        code: -1,
        message: format!("host memory: {e}"),
    }
}

impl Collective for HostCollective {
    fn backend(&self) -> &'static str {
        "host"
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.shared.world
    }

    fn all_reduce(
        &self,
        buf: &mut DeviceSlice,
        dtype: DType,
        op: ReduceOp,
        _stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        Self::check_elems(dtype, buf.len())?;
        let bytes = buf.read_bytes().map_err(memory)?;
        let out = self.exchange(OpKind::AllReduce(op), Some(dtype), bytes, buf.len())?;
        buf.write_bytes(&out).map_err(memory)
    }

    fn all_gather(
        &self,
        send: &DeviceSlice,
        recv: &mut DeviceSlice,
        _stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        let bytes = send.read_bytes().map_err(memory)?;
        let out = self.exchange(OpKind::AllGather, None, bytes, recv.len())?;
        recv.write_bytes(&out).map_err(memory)
    }

    fn reduce_scatter(
        &self,
        send: &DeviceSlice,
        recv: &mut DeviceSlice,
        dtype: DType,
        op: ReduceOp,
        _stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        Self::check_elems(dtype, recv.len())?;
        let bytes = send.read_bytes().map_err(memory)?;
        let out = self.exchange(OpKind::ReduceScatter(op), Some(dtype), bytes, recv.len())?;
        recv.write_bytes(&out).map_err(memory)
    }

    fn broadcast(
        &self,
        buf: &mut DeviceSlice,
        root: usize,
        _stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        if root >= self.shared.world {
            return Err(CollectiveError::ShapeMismatch);
        }
        let bytes = buf.read_bytes().map_err(memory)?;
        let out = self.exchange(OpKind::Broadcast(root), None, bytes, buf.len())?;
        buf.write_bytes(&out).map_err(memory)
    }

    fn barrier(&self, _stream: &StreamRef) -> Result<(), CollectiveError> {
        self.exchange(OpKind::Barrier, None, Vec::new(), 0)
            .map(drop)
    }

    fn abort(&self) {
        let mut st = self.shared.lock();
        st.aborted_by.get_or_insert(self.rank);
        self.shared.cv.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use half::bf16;
    use turbine_tensor::host::HostMemory;
    use turbine_tensor::{DType, DeviceBuffer, DeviceId, DeviceMemory};

    use super::*;
    use crate::collective::{Collective, CollectiveError, ReduceOp};

    /// splitmix64: deterministic test values without a RNG dependency.
    fn values(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                // Uniform in [-8, 8) with a fractional part, so sums round.
                ((z >> 40) as f32 / (1u64 << 24) as f32) * 16.0 - 8.0
            })
            .collect()
    }

    fn encode(dtype: DType, v: &[f32]) -> Vec<u8> {
        match dtype {
            DType::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            DType::BF16 => v
                .iter()
                .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
                .collect(),
            other => panic!("unsupported {other:?}"),
        }
    }

    fn decode(dtype: DType, b: &[u8]) -> Vec<f32> {
        match dtype {
            DType::F32 => b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            DType::BF16 => b
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            other => panic!("unsupported {other:?}"),
        }
    }

    /// Naive single-threaded reduction in rank order 0..n; BF16 accumulates in FP32 and
    /// rounds once.
    fn reference_reduce(dtype: DType, inputs: &[Vec<u8>], op: ReduceOp) -> Vec<u8> {
        let decoded: Vec<Vec<f32>> = inputs.iter().map(|b| decode(dtype, b)).collect();
        let mut acc = decoded[0].clone();
        for r in &decoded[1..] {
            for (a, x) in acc.iter_mut().zip(r) {
                *a = match op {
                    ReduceOp::Sum => *a + *x,
                    ReduceOp::Max => {
                        if *x > *a {
                            *x
                        } else {
                            *a
                        }
                    }
                };
            }
        }
        encode(dtype, &acc)
    }

    struct RankResult {
        sum: Vec<u8>,
        max: Vec<u8>,
        gathered: Vec<u8>,
        scattered: Vec<u8>,
        broadcast: Vec<u8>,
    }

    fn run_rank(
        c: &HostCollective,
        dtype: DType,
        elems: usize,
        input: &[u8],
        scatter_input: &[u8],
    ) -> RankResult {
        let world = c.world_size();
        let esize = dtype.size_bytes();
        let mem: Arc<dyn DeviceMemory> =
            HostMemory::new(DeviceId(c.rank() as u32), 1 << 30) as Arc<dyn DeviceMemory>;
        let stream = mem.compute_stream();
        let n = elems * esize;

        let buf = DeviceBuffer::alloc(&mem, n).expect("alloc");
        let mut s = buf.whole();
        s.write_bytes(input).expect("write");
        c.all_reduce(&mut s, dtype, ReduceOp::Sum, &stream)
            .expect("all_reduce sum");
        let sum = s.read_bytes().expect("read");

        s.write_bytes(input).expect("write");
        c.all_reduce(&mut s, dtype, ReduceOp::Max, &stream)
            .expect("all_reduce max");
        let max = s.read_bytes().expect("read");

        let send = DeviceBuffer::alloc(&mem, n).expect("alloc");
        send.whole().write_bytes(input).expect("write");
        let recv = DeviceBuffer::alloc(&mem, n * world).expect("alloc");
        let mut r = recv.whole();
        c.all_gather(&send.whole(), &mut r, &stream)
            .expect("all_gather");
        let gathered = r.read_bytes().expect("read");

        let ssend = DeviceBuffer::alloc(&mem, n * world).expect("alloc");
        ssend.whole().write_bytes(scatter_input).expect("write");
        let srecv = DeviceBuffer::alloc(&mem, n).expect("alloc");
        let mut sr = srecv.whole();
        c.reduce_scatter(&ssend.whole(), &mut sr, dtype, ReduceOp::Sum, &stream)
            .expect("reduce_scatter");
        let scattered = sr.read_bytes().expect("read");

        s.write_bytes(input).expect("write");
        c.broadcast(&mut s, world - 1, &stream).expect("broadcast");
        let broadcast = s.read_bytes().expect("read");

        c.barrier(&stream).expect("barrier");
        RankResult {
            sum,
            max,
            gathered,
            scattered,
            broadcast,
        }
    }

    #[test]
    fn ops_match_reference() {
        for world in [1usize, 2, 3, 4, 8] {
            for dtype in [DType::F32, DType::BF16] {
                for elems in [1usize, 7, 4099] {
                    let ctx = format!("world {world} {dtype:?} {elems} elements");
                    let inputs: Vec<Vec<u8>> = (0..world)
                        .map(|r| encode(dtype, &values(r as u64 * 7919 + elems as u64, elems)))
                        .collect();
                    let scatter_inputs: Vec<Vec<u8>> = (0..world)
                        .map(|r| {
                            encode(
                                dtype,
                                &values(r as u64 * 104_729 + 3 + elems as u64, elems * world),
                            )
                        })
                        .collect();
                    let group = HostCollective::group(world, Duration::from_secs(20));
                    let results: Vec<RankResult> = std::thread::scope(|scope| {
                        let handles: Vec<_> = group
                            .iter()
                            .enumerate()
                            .map(|(r, c)| {
                                let (i, si) = (&inputs[r], &scatter_inputs[r]);
                                scope.spawn(move || run_rank(c, dtype, elems, i, si))
                            })
                            .collect();
                        handles
                            .into_iter()
                            .map(|h| h.join().expect("rank thread"))
                            .collect()
                    });

                    let sum = reference_reduce(dtype, &inputs, ReduceOp::Sum);
                    let max = reference_reduce(dtype, &inputs, ReduceOp::Max);
                    let gathered: Vec<u8> = inputs.concat();
                    let scattered = reference_reduce(dtype, &scatter_inputs, ReduceOp::Sum);
                    let chunk = elems * dtype.size_bytes();
                    for (r, res) in results.iter().enumerate() {
                        assert_eq!(res.sum, sum, "{ctx} rank {r} all_reduce sum");
                        assert_eq!(res.max, max, "{ctx} rank {r} all_reduce max");
                        assert_eq!(res.gathered, gathered, "{ctx} rank {r} all_gather");
                        assert_eq!(
                            res.scattered,
                            scattered[r * chunk..(r + 1) * chunk],
                            "{ctx} rank {r} reduce_scatter"
                        );
                        assert_eq!(res.broadcast, inputs[world - 1], "{ctx} rank {r} broadcast");
                    }
                }
            }
        }
    }

    #[test]
    fn op_timeout_aborts() {
        let group = HostCollective::group(2, Duration::from_millis(500));
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 20);
        let stream = mem.compute_stream();
        let buf = DeviceBuffer::alloc(&mem, 16).expect("alloc");
        let mut s = buf.whole();

        let started = Instant::now();
        let err = group[0]
            .all_reduce(&mut s, DType::F32, ReduceOp::Sum, &stream)
            .expect_err("rank 1 never enters");
        assert!(started.elapsed() < Duration::from_millis(1500), "{err}");
        assert!(
            matches!(err, CollectiveError::Timeout { op: "all_reduce", after } if after == Duration::from_millis(500)),
            "{err:?}"
        );
        for c in &group {
            let later = c.barrier(&stream).expect_err("group is aborted");
            assert!(
                matches!(later, CollectiveError::RemoteAbort { rank: 0 }),
                "{later:?}"
            );
        }
    }

    #[test]
    fn shape_mismatch_fails_every_rank() {
        let group = HostCollective::group(2, Duration::from_secs(5));
        let errors: Vec<CollectiveError> = std::thread::scope(|scope| {
            let handles: Vec<_> = group
                .iter()
                .enumerate()
                .map(|(r, c)| {
                    scope.spawn(move || {
                        let mem: Arc<dyn DeviceMemory> =
                            HostMemory::new(DeviceId(r as u32), 1 << 20);
                        let stream = mem.compute_stream();
                        let buf = DeviceBuffer::alloc(&mem, 4 * (r + 1)).expect("alloc");
                        let mut s = buf.whole();
                        c.all_reduce(&mut s, DType::F32, ReduceOp::Sum, &stream)
                            .expect_err("lengths differ")
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rank thread"))
                .collect()
        });
        for e in errors {
            assert!(matches!(e, CollectiveError::ShapeMismatch), "{e:?}");
        }
        assert_eq!(group[0].backend(), "host");
    }
}
