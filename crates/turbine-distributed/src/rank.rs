//! Rank runtime (P5 S-5): the leader broadcasts each step's [`StepPlan`] to its worker ranks,
//! which execute it in lockstep; the leader executes its own rank itself.
//!
//! Two modes. `local`: one thread per worker rank in this process, fed through a bounded
//! channel; at most `depth` (`parallel.plan_queue_depth`) plans are outstanding per worker
//! (queued or executing), so a stalled worker blocks the leader instead of growing a queue.
//! `static`: one process per rank; workers connect to the leader over a registered rank
//! transport (`parallel.ranks.transport`, [`crate::transport`]; `tcp` in Phase 5), the leader checks
//! each `Hello` (protocol, world size, rank, model/config fingerprints, vendor, architecture)
//! and answers `Welcome { unique_id }` once every rank joined, or `Reject { reason }`. Frames are
//! a u32 little-endian length plus a postcard body, at most 16 MiB. A closed leader socket makes
//! a worker abort its communicator and return [`RankError::Closed`]; `Shutdown` travels both ways.

use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, Read, Write};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use turbine_core::types::{BlockId, ModelFingerprint, SeqId, Vendor};

use crate::collective::{Collective, CollectiveError};
use crate::transport::{RankStream, Transport};

/// Static-mode protocol version carried in `Hello`.
pub const PROTOCOL_VERSION: u16 = 1;
/// Largest frame body accepted or sent (16 MiB).
pub const MAX_FRAME_BYTES: usize = 16 << 20;

/// One engine step, identical on every rank.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct StepPlan {
    pub step: u64,
    pub sequences: Vec<StepSeq>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct StepSeq {
    pub seq_id: SeqId,
    pub tokens: Vec<u32>,
    pub positions: Vec<u32>,
    /// Logical block ids allocated by the leader; every rank uses the same ids.
    pub block_table: Vec<BlockId>,
    pub is_prefill: bool,
}

/// Runs one rank's share of a step (implemented in `turbine-server` over `turbine-model`).
pub trait StepExecutor: Send {
    fn execute(&mut self, plan: &StepPlan) -> Result<StepOutput, ExecError>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct StepOutput {
    /// Leader only: `rows × vocab` logits.
    pub logits: Option<Vec<f32>>,
    pub rows: usize,
    pub vocab: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum ExecError {
    #[error(transparent)]
    Collective(#[from] CollectiveError),
    #[error("executor: {0}")]
    Executor(String),
    #[error("sticky device error: {0}")]
    DeviceFatal(String),
}

/// 128 opaque bytes (`ncclUniqueId`), serialised as a byte string.
mod unique_id_bytes {
    use super::*;

    pub fn serialize<S: Serializer>(id: &[u8; 128], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(id)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 128], D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = [u8; 128];
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("128 bytes")
            }
            fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<[u8; 128], E> {
                v.try_into()
                    .map_err(|_| E::invalid_length(v.len(), &"128 bytes"))
            }
            fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<[u8; 128], A::Error> {
                let mut out = [0u8; 128];
                for (i, slot) in out.iter_mut().enumerate() {
                    *slot = seq
                        .next_element()?
                        .ok_or_else(|| de::Error::invalid_length(i, &"128 bytes"))?;
                }
                Ok(out)
            }
        }
        d.deserialize_bytes(V)
    }
}

/// Static-mode protocol v1 messages.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum RankMessage {
    Hello {
        protocol: u16,
        rank: u32,
        world_size: u32,
        model_fingerprint: ModelFingerprint,
        config_fingerprint: [u8; 32],
        device_vendor: Vendor,
        device_arch: String,
    },
    Welcome {
        #[serde(with = "unique_id_bytes")]
        unique_id: [u8; 128],
    },
    Reject {
        reason: String,
    },
    StepPlan(StepPlan),
    Shutdown {
        reason: String,
    },
}

/// Writes one frame: u32 LE body length, then the postcard body (≤ [`MAX_FRAME_BYTES`]).
pub fn write_frame(w: &mut impl Write, m: &RankMessage) -> io::Result<()> {
    let body = postcard::to_stdvec(m).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    if body.len() > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("frame of {} bytes exceeds {MAX_FRAME_BYTES}", body.len()),
        ));
    }
    w.write_all(&(body.len() as u32).to_le_bytes())?;
    w.write_all(&body)?;
    w.flush()
}

/// Reads one frame; a length above [`MAX_FRAME_BYTES`] is `InvalidData` before any body read.
pub fn read_frame(r: &mut impl Read) -> io::Result<RankMessage> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len)?;
    let len = u32::from_le_bytes(len) as usize;
    if len > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame of {len} bytes exceeds {MAX_FRAME_BYTES}"),
        ));
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body)?;
    postcard::from_bytes(&body).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// What the leader requires of every joining rank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HelloExpect {
    pub model_fingerprint: ModelFingerprint,
    pub config_fingerprint: [u8; 32],
    pub device_vendor: Vendor,
    pub device_arch: String,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum RankError {
    #[error("ranks {missing:?} did not join before the init timeout")]
    Timeout { missing: Vec<u32> },
    #[error("rejected: {0}")]
    Rejected(String),
    #[error("rank {rank} closed its connection")]
    Closed { rank: u32 },
    #[error("i/o: {0}")]
    Io(String),
    #[error("rank {rank} failed: {detail}")]
    Executor { rank: u32, detail: String },
}

fn io_err(e: io::Error) -> RankError {
    RankError::Io(e.to_string())
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// First failure seen by any rank; later steps report it.
#[derive(Default)]
struct Failure(Mutex<Option<RankError>>);

impl Failure {
    fn set(&self, e: RankError) {
        lock(&self.0).get_or_insert(e);
    }
    fn get(&self) -> Option<RankError> {
        lock(&self.0).clone()
    }
}

/// Outstanding plans of one local worker.
struct Slots {
    outstanding: Mutex<usize>,
    cv: Condvar,
    alive: AtomicBool,
}

struct LocalWorker {
    rank: u32,
    tx: Option<SyncSender<StepPlan>>,
    slots: Arc<Slots>,
    handle: Option<JoinHandle<()>>,
}

struct StaticLink {
    rank: u32,
    tx: Option<SyncSender<RankMessage>>,
    /// A clone of the connection, kept to close it (both directions) on shutdown or drop.
    control: Box<dyn RankStream>,
    writer: Option<JoinHandle<()>>,
}

enum Mode {
    Local(Vec<LocalWorker>),
    Static {
        links: Vec<StaticLink>,
        closing: Arc<AtomicBool>,
    },
}

/// The leader's side of one TP group: broadcasts step plans to the worker ranks.
pub struct RankRuntime {
    mode: Mode,
    depth: usize,
    failure: Arc<Failure>,
}

fn local_worker_loop(
    rank: u32,
    mut exec: Box<dyn StepExecutor>,
    rx: Receiver<StepPlan>,
    slots: Arc<Slots>,
    failure: Arc<Failure>,
) {
    for plan in rx {
        let result = exec.execute(&plan);
        *lock(&slots.outstanding) -= 1;
        slots.cv.notify_all();
        if let Err(e) = result {
            tracing::warn!(event = "rank_failed", rank, error = %e, "worker rank failed");
            failure.set(RankError::Executor {
                rank,
                detail: e.to_string(),
            });
            break;
        }
    }
    slots.alive.store(false, Ordering::Release);
    slots.cv.notify_all();
}

/// Checks one `Hello` against the leader's expectations.
fn check_hello(
    msg: &RankMessage,
    expect: &HelloExpect,
    world: usize,
    joined: &BTreeMap<u32, Box<dyn RankStream>>,
) -> Result<u32, String> {
    let RankMessage::Hello {
        protocol,
        rank,
        world_size,
        model_fingerprint,
        config_fingerprint,
        device_vendor,
        device_arch,
    } = msg
    else {
        return Err("expected Hello".into());
    };
    if *protocol != PROTOCOL_VERSION {
        return Err(format!(
            "protocol {protocol} differs from the leader's {PROTOCOL_VERSION}"
        ));
    }
    if *world_size as usize != world {
        return Err(format!(
            "world_size {world_size} differs from the leader's {world}"
        ));
    }
    if *rank == 0 || *rank as usize >= world {
        return Err(format!("rank {rank} is outside 1..{}", world - 1));
    }
    if joined.contains_key(rank) {
        return Err(format!("rank {rank} already joined"));
    }
    if *model_fingerprint != expect.model_fingerprint {
        return Err("model_fingerprint differs from the leader's".into());
    }
    if *config_fingerprint != expect.config_fingerprint {
        return Err("config_fingerprint differs from the leader's".into());
    }
    if *device_vendor != expect.device_vendor {
        return Err(format!(
            "device_vendor {} differs from the leader's {}",
            device_vendor.as_str(),
            expect.device_vendor.as_str()
        ));
    }
    if *device_arch != expect.device_arch {
        return Err(format!(
            "device_arch {device_arch} differs from the leader's {}",
            expect.device_arch
        ));
    }
    Ok(*rank)
}

impl RankRuntime {
    /// Local mode: `executors[i]` runs rank `i + 1` on its own thread; at most `depth` plans
    /// are outstanding per worker.
    pub fn local(executors: Vec<Box<dyn StepExecutor>>, depth: usize) -> Self {
        let depth = depth.max(1);
        let failure = Arc::new(Failure::default());
        let workers = executors
            .into_iter()
            .enumerate()
            .map(|(i, exec)| {
                let rank = i as u32 + 1;
                let (tx, rx) = sync_channel(depth);
                let slots = Arc::new(Slots {
                    outstanding: Mutex::new(0),
                    cv: Condvar::new(),
                    alive: AtomicBool::new(true),
                });
                let (s, f) = (Arc::clone(&slots), Arc::clone(&failure));
                let handle = std::thread::Builder::new()
                    .name(format!("turbine-rank-{rank}"))
                    .spawn(move || local_worker_loop(rank, exec, rx, s, f))
                    .expect("spawn a rank thread");
                LocalWorker {
                    rank,
                    tx: Some(tx),
                    slots,
                    handle: Some(handle),
                }
            })
            .collect();
        RankRuntime {
            mode: Mode::Local(workers),
            depth,
            failure,
        }
    }

    /// Static mode, rank 0: accepts ranks `1..world` on `listen` over `transport` until all
    /// joined (then `Welcome { unique_id }` to each) or `init_timeout` passed (`Timeout {
    /// missing }`, and `Shutdown` to the ranks that did join). Bad `Hello`s get `Reject { reason }`.
    pub fn static_leader(
        transport: &dyn Transport,
        listen: SocketAddr,
        expect: HelloExpect,
        world: usize,
        init_timeout: Duration,
        unique_id: [u8; 128],
        depth: usize,
    ) -> Result<Self, RankError> {
        let listener = transport.listen(listen).map_err(io_err)?;
        let deadline = Instant::now() + init_timeout;
        let mut joined: BTreeMap<u32, Box<dyn RankStream>> = BTreeMap::new();
        while joined.len() + 1 < world {
            let now = Instant::now();
            if now >= deadline {
                let missing: Vec<u32> = (1..world as u32)
                    .filter(|r| !joined.contains_key(r))
                    .collect();
                let reason = format!("ranks {missing:?} did not join within {init_timeout:?}");
                for stream in joined.values_mut() {
                    let _ = write_frame(
                        stream,
                        &RankMessage::Shutdown {
                            reason: reason.clone(),
                        },
                    );
                }
                tracing::warn!(
                    event = "rank_join_timeout",
                    ?missing,
                    "static rank join timed out"
                );
                return Err(RankError::Timeout { missing });
            }
            let mut stream = match listener.accept(deadline - now) {
                Ok(Some(s)) => s,
                Ok(None) => continue,
                Err(e) => return Err(io_err(e)),
            };
            let left = deadline.saturating_duration_since(Instant::now());
            if stream
                .set_read_timeout(Some(left.max(Duration::from_millis(1))))
                .is_err()
            {
                continue;
            }
            let Ok(msg) = read_frame(&mut stream) else {
                continue;
            };
            match check_hello(&msg, &expect, world, &joined) {
                Ok(rank) => {
                    tracing::info!(event = "rank_joined", rank, "rank joined");
                    joined.insert(rank, stream);
                }
                Err(reason) => {
                    tracing::warn!(event = "rank_rejected", %reason, "rank rejected");
                    let _ = write_frame(&mut stream, &RankMessage::Reject { reason });
                }
            }
        }
        let depth = depth.max(1);
        let failure = Arc::new(Failure::default());
        let closing = Arc::new(AtomicBool::new(false));
        let mut links = Vec::with_capacity(joined.len());
        for (rank, mut stream) in joined {
            stream.set_read_timeout(None).map_err(io_err)?;
            write_frame(&mut stream, &RankMessage::Welcome { unique_id }).map_err(io_err)?;
            let control = stream.try_clone().map_err(io_err)?;
            let mut reader = stream.try_clone().map_err(io_err)?;
            let (tx, rx) = sync_channel::<RankMessage>(depth);
            let (f, c) = (Arc::clone(&failure), Arc::clone(&closing));
            let writer = std::thread::Builder::new()
                .name(format!("turbine-rank-{rank}-tx"))
                .spawn(move || {
                    for msg in rx {
                        if write_frame(&mut stream, &msg).is_err() {
                            if !c.load(Ordering::Acquire) {
                                f.set(RankError::Closed { rank });
                            }
                            break;
                        }
                    }
                })
                .map_err(io_err)?;
            let (f, c) = (Arc::clone(&failure), Arc::clone(&closing));
            std::thread::Builder::new()
                .name(format!("turbine-rank-{rank}-rx"))
                .spawn(move || match read_frame(&mut reader) {
                    Ok(RankMessage::Shutdown { reason }) => f.set(RankError::Executor {
                        rank,
                        detail: reason,
                    }),
                    _ if c.load(Ordering::Acquire) => {}
                    _ => f.set(RankError::Closed { rank }),
                })
                .map_err(io_err)?;
            links.push(StaticLink {
                rank,
                tx: Some(tx),
                control,
                writer: Some(writer),
            });
        }
        tracing::info!(event = "ranks_ready", world, "every static rank joined");
        Ok(RankRuntime {
            mode: Mode::Static { links, closing },
            depth,
            failure,
        })
    }

    /// Static mode, ranks 1..: connects to `leader` over `transport` (retrying with 50 ms
    /// doubling to 1 s backoff until `init_timeout`), sends `hello` and waits for `Welcome` or
    /// `Reject`.
    pub fn static_worker(
        transport: &dyn Transport,
        leader: SocketAddr,
        hello: RankMessage,
        init_timeout: Duration,
    ) -> Result<WorkerLink, RankError> {
        let RankMessage::Hello { rank, .. } = hello else {
            return Err(RankError::Io("static_worker needs a Hello message".into()));
        };
        let deadline = Instant::now() + init_timeout;
        let mut backoff = Duration::from_millis(50);
        let mut stream = loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(RankError::Timeout { missing: vec![0] });
            }
            match transport.connect(leader, left) {
                Ok(s) => break s,
                Err(_) => {
                    std::thread::sleep(
                        backoff.min(deadline.saturating_duration_since(Instant::now())),
                    );
                    backoff = (backoff * 2).min(Duration::from_secs(1));
                }
            }
        };
        write_frame(&mut stream, &hello).map_err(io_err)?;
        let left = deadline.saturating_duration_since(Instant::now());
        stream
            .set_read_timeout(Some(left.max(Duration::from_millis(1))))
            .map_err(io_err)?;
        match read_frame(&mut stream) {
            Ok(RankMessage::Welcome { unique_id }) => {
                stream.set_read_timeout(None).map_err(io_err)?;
                Ok(WorkerLink {
                    stream,
                    rank,
                    unique_id,
                })
            }
            Ok(RankMessage::Reject { reason } | RankMessage::Shutdown { reason }) => {
                Err(RankError::Rejected(reason))
            }
            Ok(other) => Err(RankError::Io(format!(
                "unexpected handshake message {other:?}"
            ))),
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                Err(RankError::Timeout { missing: vec![0] })
            }
            Err(_) => Err(RankError::Closed { rank: 0 }),
        }
    }

    /// Hands `plan` to every worker rank. Blocks while a worker has `depth` plans outstanding;
    /// returns the first failure any rank reported.
    pub fn step(&mut self, plan: StepPlan) -> Result<(), RankError> {
        if let Some(e) = self.failure.get() {
            return Err(e);
        }
        let depth = self.depth;
        match &mut self.mode {
            Mode::Local(workers) => {
                for w in workers.iter() {
                    let mut outstanding = lock(&w.slots.outstanding);
                    while *outstanding >= depth && w.slots.alive.load(Ordering::Acquire) {
                        outstanding = w
                            .slots
                            .cv
                            .wait(outstanding)
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                    }
                    if !w.slots.alive.load(Ordering::Acquire) {
                        drop(outstanding);
                        return Err(self
                            .failure
                            .get()
                            .unwrap_or(RankError::Closed { rank: w.rank }));
                    }
                    *outstanding += 1;
                    drop(outstanding);
                    let sent = w.tx.as_ref().map(|tx| tx.send(plan.clone()));
                    if !matches!(sent, Some(Ok(()))) {
                        return Err(self
                            .failure
                            .get()
                            .unwrap_or(RankError::Closed { rank: w.rank }));
                    }
                }
            }
            Mode::Static { links, .. } => {
                for l in links.iter() {
                    let sent =
                        l.tx.as_ref()
                            .is_some_and(|tx| tx.send(RankMessage::StepPlan(plan.clone())).is_ok());
                    if !sent {
                        return Err(self
                            .failure
                            .get()
                            .unwrap_or(RankError::Closed { rank: l.rank }));
                    }
                }
            }
        }
        Ok(())
    }

    /// Clean shutdown: static workers receive `Shutdown { reason }`; local workers finish their
    /// queued plans and exit.
    pub fn shutdown(&mut self, reason: &str) {
        match &mut self.mode {
            Mode::Local(workers) => {
                for w in workers.iter_mut() {
                    w.tx = None;
                }
                for w in workers.iter_mut() {
                    if let Some(h) = w.handle.take() {
                        let _ = h.join();
                    }
                }
            }
            Mode::Static { links, closing } => {
                closing.store(true, Ordering::Release);
                for l in links.iter_mut() {
                    if let Some(tx) = l.tx.take() {
                        let _ = tx.send(RankMessage::Shutdown {
                            reason: reason.to_string(),
                        });
                    }
                }
                for l in links.iter_mut() {
                    if let Some(h) = l.writer.take() {
                        let _ = h.join();
                    }
                    let _ = l.control.shutdown();
                }
            }
        }
        tracing::info!(event = "ranks_shutdown", reason, "rank runtime shut down");
    }
}

impl Drop for RankRuntime {
    /// Abrupt: local workers stop after their queued plans; static sockets are closed without
    /// `Shutdown`, so workers see the leader as lost.
    fn drop(&mut self) {
        match &mut self.mode {
            Mode::Local(workers) => {
                for w in workers.iter_mut() {
                    w.tx = None;
                }
            }
            Mode::Static { links, closing } => {
                closing.store(true, Ordering::Release);
                for l in links.iter_mut() {
                    l.tx = None;
                    let _ = l.control.shutdown();
                }
            }
        }
    }
}

/// A static-mode worker's connection to its leader, after `Welcome`.
pub struct WorkerLink {
    stream: Box<dyn RankStream>,
    rank: u32,
    unique_id: [u8; 128],
}

impl WorkerLink {
    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// The communicator unique id from `Welcome`.
    pub fn unique_id(&self) -> [u8; 128] {
        self.unique_id
    }

    /// Executes every `StepPlan` the leader sends until `Shutdown` (`Ok`). A lost leader aborts
    /// `comm` and returns `Closed { rank: 0 }`; a failed step sends `Shutdown` to the leader,
    /// aborts `comm` and returns `Executor`.
    pub fn run(
        mut self,
        exec: &mut dyn StepExecutor,
        comm: &dyn Collective,
    ) -> Result<(), RankError> {
        loop {
            match read_frame(&mut self.stream) {
                Ok(RankMessage::StepPlan(plan)) => {
                    if let Err(e) = exec.execute(&plan) {
                        let detail = e.to_string();
                        let _ = write_frame(
                            &mut self.stream,
                            &RankMessage::Shutdown {
                                reason: format!("rank {}: {detail}", self.rank),
                            },
                        );
                        comm.abort();
                        return Err(RankError::Executor {
                            rank: self.rank,
                            detail,
                        });
                    }
                }
                Ok(RankMessage::Shutdown { reason }) => {
                    tracing::info!(event = "rank_shutdown", rank = self.rank, %reason, "leader shut down");
                    return Ok(());
                }
                Ok(other) => {
                    comm.abort();
                    return Err(RankError::Io(format!("unexpected message {other:?}")));
                }
                Err(_) => {
                    tracing::warn!(
                        event = "leader_lost",
                        rank = self.rank,
                        "leader connection closed"
                    );
                    comm.abort();
                    return Err(RankError::Closed { rank: 0 });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::{SocketAddr, TcpListener};
    use std::sync::mpsc;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{Duration, Instant};

    use turbine_core::types::{BlockId, ModelFingerprint, SeqId, Vendor};
    use turbine_tensor::host::HostMemory;
    use turbine_tensor::{DeviceBuffer, DeviceMemory};

    use super::*;
    use crate::collective::{Collective, CollectiveError, HostCollective, ReduceOp};

    /// The `tcp` rank transport, as `parallel.ranks.transport: tcp` selects it.
    fn tcp() -> &'static dyn Transport {
        crate::transport::registry()
            .get("tcp")
            .expect("tcp is registered")
    }

    fn free_addr() -> SocketAddr {
        let l = TcpListener::bind("127.0.0.1:0").expect("bind");
        l.local_addr().expect("addr")
    }

    fn expect() -> HelloExpect {
        HelloExpect {
            model_fingerprint: ModelFingerprint([7; 32]),
            config_fingerprint: [9; 32],
            device_vendor: Vendor::Amd,
            device_arch: "gfx1201".into(),
        }
    }

    fn hello(rank: u32, world: u32) -> RankMessage {
        let e = expect();
        RankMessage::Hello {
            protocol: PROTOCOL_VERSION,
            rank,
            world_size: world,
            model_fingerprint: e.model_fingerprint,
            config_fingerprint: e.config_fingerprint,
            device_vendor: e.device_vendor,
            device_arch: e.device_arch,
        }
    }

    fn step_plan(step: u64) -> StepPlan {
        StepPlan {
            step,
            sequences: vec![StepSeq {
                seq_id: SeqId(step),
                tokens: vec![1, 2, 3],
                positions: vec![0, 1, 2],
                block_table: vec![BlockId(4)],
                is_prefill: true,
            }],
        }
    }

    fn unique_id() -> [u8; 128] {
        std::array::from_fn(|i| (i * 3) as u8)
    }

    #[test]
    fn frames_round_trip_and_are_bounded() {
        let msgs = [
            hello(1, 2),
            RankMessage::Welcome {
                unique_id: unique_id(),
            },
            RankMessage::Reject {
                reason: "model_fingerprint differs".into(),
            },
            RankMessage::StepPlan(step_plan(3)),
            RankMessage::Shutdown {
                reason: "bye".into(),
            },
        ];
        let mut wire = Vec::new();
        for m in &msgs {
            write_frame(&mut wire, m).expect("write");
        }
        let mut r = wire.as_slice();
        for m in &msgs {
            assert_eq!(&read_frame(&mut r).expect("read"), m);
        }
        // A length prefix above 16 MiB is refused before reading the body.
        let mut big = ((MAX_FRAME_BYTES + 1) as u32).to_le_bytes().to_vec();
        big.extend_from_slice(&[0; 8]);
        let e = read_frame(&mut big.as_slice()).expect_err("oversized");
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn static_protocol_handshake() {
        // Three workers join a world of four; all get the same id.
        let addr = free_addr();
        let leader = thread::spawn(move || {
            RankRuntime::static_leader(
                tcp(),
                addr,
                expect(),
                4,
                Duration::from_secs(5),
                unique_id(),
                2,
            )
        });
        let workers: Vec<_> = (1..4)
            .map(|rank| {
                thread::spawn(move || {
                    RankRuntime::static_worker(tcp(), addr, hello(rank, 4), Duration::from_secs(5))
                })
            })
            .collect();
        let mut leader = leader.join().unwrap().expect("all ranks joined");
        for w in workers {
            let link = w.join().unwrap().expect("welcome");
            assert_eq!(link.unique_id(), unique_id());
        }
        leader.shutdown("test done");

        // A world of three where rank 2 never joins correctly.
        let addr = free_addr();
        let started = Instant::now();
        let leader = thread::spawn(move || {
            RankRuntime::static_leader(
                tcp(),
                addr,
                expect(),
                3,
                Duration::from_secs(1),
                unique_id(),
                2,
            )
        });
        let good = thread::spawn(move || {
            RankRuntime::static_worker(tcp(), addr, hello(1, 3), Duration::from_secs(3))
        });
        let reject = |msg: RankMessage| match RankRuntime::static_worker(
            tcp(),
            addr,
            msg,
            Duration::from_secs(3),
        ) {
            Err(RankError::Rejected(reason)) => reason,
            other => panic!("expected a rejection, got {:?}", other.map(|l| l.rank())),
        };
        thread::sleep(Duration::from_millis(100));
        let mut other_model = hello(2, 3);
        if let RankMessage::Hello {
            model_fingerprint, ..
        } = &mut other_model
        {
            *model_fingerprint = ModelFingerprint([8; 32]);
        }
        assert!(reject(other_model).contains("model_fingerprint"));
        let mut nvidia = hello(2, 3);
        if let RankMessage::Hello { device_vendor, .. } = &mut nvidia {
            *device_vendor = Vendor::Nvidia;
        }
        assert!(reject(nvidia).contains("device_vendor"));
        // Rank 1 is already taken by the good worker.
        thread::sleep(Duration::from_millis(100));
        assert!(reject(hello(1, 3)).contains("rank 1"));

        match leader.join().unwrap() {
            Err(RankError::Timeout { missing }) => assert_eq!(missing, vec![2]),
            other => panic!("expected Timeout, got {:?}", other.err()),
        }
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "{:?}",
            started.elapsed()
        );
        // The joined worker is told why and does not get a Welcome.
        assert!(good.join().unwrap().is_err());
    }

    /// Executes nothing; counts plans.
    struct Count(Arc<std::sync::atomic::AtomicU64>);
    impl StepExecutor for Count {
        fn execute(&mut self, plan: &StepPlan) -> Result<StepOutput, ExecError> {
            self.0.store(plan.step, std::sync::atomic::Ordering::SeqCst);
            Ok(StepOutput {
                logits: None,
                rows: 0,
                vocab: 0,
            })
        }
    }

    #[test]
    fn leader_loss_aborts_workers() {
        let addr = free_addr();
        let mut comms = HostCollective::group(3, Duration::from_secs(5)).into_iter();
        let _leader_comm = comms.next().expect("rank 0");
        let leader = thread::spawn(move || {
            RankRuntime::static_leader(
                tcp(),
                addr,
                expect(),
                3,
                Duration::from_secs(5),
                unique_id(),
                2,
            )
        });
        let (done_tx, done_rx) = mpsc::channel();
        let mut shared = Vec::new();
        for (rank, comm) in (1..3).zip(comms) {
            let comm: Arc<dyn Collective> = Arc::new(comm);
            shared.push(Arc::clone(&comm));
            let done = done_tx.clone();
            thread::spawn(move || {
                let link =
                    RankRuntime::static_worker(tcp(), addr, hello(rank, 3), Duration::from_secs(5))
                        .expect("welcome");
                let steps = Arc::new(std::sync::atomic::AtomicU64::new(0));
                let mut exec = Count(Arc::clone(&steps));
                let result = link.run(&mut exec, comm.as_ref());
                done.send((
                    rank,
                    result,
                    steps.load(std::sync::atomic::Ordering::SeqCst),
                ))
                .unwrap();
            });
        }
        let mut leader = leader.join().unwrap().expect("joined");
        leader.step(step_plan(1)).expect("step 1");
        leader.step(step_plan(2)).expect("step 2");
        thread::sleep(Duration::from_millis(100));
        let dropped = Instant::now();
        drop(leader);
        for _ in 0..2 {
            let (rank, result, steps) = done_rx
                .recv_timeout(Duration::from_secs(2))
                .expect("worker loop ends within 2 s");
            assert!(
                matches!(result, Err(RankError::Closed { rank: 0 })),
                "rank {rank}: {result:?}"
            );
            assert_eq!(steps, 2, "rank {rank} executed both plans");
        }
        assert!(dropped.elapsed() < Duration::from_secs(2));
        // Every worker aborted its communicator.
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(turbine_core::types::DeviceId(0), 64);
        let stream = mem.compute_stream();
        let buf = DeviceBuffer::alloc(&mem, 4).expect("alloc");
        for comm in shared {
            let err = comm
                .all_reduce(
                    &mut buf.whole(),
                    turbine_tensor::DType::F32,
                    ReduceOp::Sum,
                    &stream,
                )
                .expect_err("aborted");
            assert!(
                matches!(err, CollectiveError::RemoteAbort { .. }),
                "{err:?}"
            );
        }
    }

    /// Blocks every `execute` on a barrier shared with the test.
    struct Gate(Arc<Barrier>);
    impl StepExecutor for Gate {
        fn execute(&mut self, _plan: &StepPlan) -> Result<StepOutput, ExecError> {
            self.0.wait();
            Ok(StepOutput {
                logits: None,
                rows: 0,
                vocab: 0,
            })
        }
    }

    #[test]
    fn plan_queue_bounded() {
        let gate = Arc::new(Barrier::new(2));
        let mut rt = RankRuntime::local(vec![Box::new(Gate(Arc::clone(&gate)))], 2);
        let (tx, rx) = mpsc::channel();
        let leader = thread::spawn(move || {
            for step in 1..=3 {
                rt.step(step_plan(step)).expect("step");
                tx.send(step).unwrap();
            }
            rt
        });
        assert_eq!(rx.recv_timeout(Duration::from_secs(1)), Ok(1));
        assert_eq!(rx.recv_timeout(Duration::from_secs(1)), Ok(2));
        // Plans 1 and 2 are outstanding (1 executing, blocked): the third step must wait.
        assert!(
            rx.recv_timeout(Duration::from_millis(200)).is_err(),
            "the third step did not block"
        );
        gate.wait(); // release plan 1
        assert_eq!(rx.recv_timeout(Duration::from_secs(1)), Ok(3));
        gate.wait(); // plan 2
        gate.wait(); // plan 3
        let mut rt = leader.join().unwrap();
        rt.shutdown("test done");
    }

    /// Fails on its second plan.
    struct FailSecond(u32);
    impl StepExecutor for FailSecond {
        fn execute(&mut self, _plan: &StepPlan) -> Result<StepOutput, ExecError> {
            self.0 += 1;
            if self.0 == 2 {
                return Err(ExecError::Executor("boom".into()));
            }
            Ok(StepOutput {
                logits: None,
                rows: 0,
                vocab: 0,
            })
        }
    }

    #[test]
    fn local_worker_failure_surfaces_on_step() {
        let mut rt = RankRuntime::local(vec![Box::new(FailSecond(0))], 2);
        rt.step(step_plan(1)).expect("step 1");
        rt.step(step_plan(2)).expect("step 2 is queued");
        let deadline = Instant::now() + Duration::from_secs(2);
        let err = loop {
            match rt.step(step_plan(3)) {
                Err(e) => break e,
                Ok(()) if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
                Ok(()) => panic!("worker failure never surfaced"),
            }
        };
        assert!(
            matches!(&err, RankError::Executor { rank: 1, detail } if detail.contains("boom")),
            "{err:?}"
        );
    }
}
