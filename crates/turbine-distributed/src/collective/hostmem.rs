//! `hostmem`: collectives through page-locked host memory mapped into every rank's device, for
//! GPUs without a peer-to-peer path (decision "P5: small-message all-reduce latency on
//! novanas": the two R9700s of novanas have no GPU↔GPU link, and RCCL's staged path costs
//! ~21 µs per small all-reduce and ~4.3 GB/s bus bandwidth at 16 MiB).
//!
//! Ranks are threads of one process (`parallel.ranks.mode: local`): the first rank to open a
//! group allocates one mapped region (kernel ABI v2.7 `turbine_host_alloc_mapped`) and every
//! rank addresses it from its own device. Each operation is one kernel per rank and step
//! (`turbine_mapped_collective`): the rank writes its contribution into its slot, publishes a
//! sequence-numbered flag, waits for its peers' flags and combines their contributions in rank
//! order — no host thread on the path, and every rank computes the bits of the `host`
//! reference backend. Messages larger than a slot are split into several steps.
//!
//! Layout of the region: `world × MAX_BLOCKS` flag words (u64), the abort word (u32), then from
//! the next 4 KiB boundary `2 × world` slots of `slot_bytes` (two per rank, alternating by the
//! step's sequence number).
//!
//! Bounds: a kernel waits for a peer at most `op_timeout` and then stores the timeout into the
//! abort word, so a missing peer never holds a stream much longer than `op_timeout`; every
//! later step of every rank ends at once, and every call (and `step_end`) reports the abort:
//! `Timeout { op }` on the rank whose kernel timed out, `RemoteAbort { rank }` elsewhere.
//! [`Collective::abort`] writes the abort word from the host, which releases spinning peers.
//!
//! Routing: the kernels win for small messages and lose to RCCL's copy path for large ones
//! (the latency table under "hostmem against rccl" in `docs/extending/collective-backend.md`), so a call whose message
//! (nccl-tests bytes) exceeds `CollectiveInit::route_max_bytes` — `auto`: the measured per-op
//! crossover [`auto_max_bytes`] — goes to an RCCL communicator the group opens beside its own
//! (`above_threshold`); the rest stay here (`below_threshold`); every choice is counted in
//! `turbine_collective_route_total{op,backend,reason}`. The choice depends only on the op and
//! its size, so every rank routes a call the same way. On a kernel library without the v2.7
//! group the communicator is RCCL's alone (`op_unsupported`, logged once). Without a loadable
//! RCCL (default search) hostmem keeps every message. Point-to-point (`send` / `recv`, pipeline
//! stages) stays on hostmem at every size: RCCL's `ncclSend` / `ncclRecv` need a peer path
//! between the devices, which the group does not know to exist (novanas has none; RCCL fails
//! with "unhandled system error"), so a call above the threshold is counted with reason
//! `no_peer_access` (logged once) instead of going to the delegate.
//!
//! Sequencing: on a kernel library with ABI v2.8 each rank's group channel keeps its step
//! counter in device memory (`turbine_mapped_collective_dseq`), so the steps of a call can be
//! captured into a graph and every replay runs with fresh sequence numbers (tensor-parallel
//! decode graphs, P5 Task 32); the arithmetic is the same. On a v2.7 library the counter is the
//! host's, and a step refuses stream capture. [`set_device_sequencing`] turns the device counter
//! off for groups opened afterwards (an A/B switch for `turbine-collbench`).
//!
//! Copy-engine all-reduce (P5 Task 32, off by default): with
//! `parallel.collective.hostmem_dma_min_bytes` ([`set_dma_min_bytes`]) an all-reduce of at
//! least that many bytes — outside a graph capture, on a kernel library with ABI v2.8 — moves
//! its bytes with the copy engines (~2× the kernels' mapped-memory rate on novanas) through
//! slots of its own (`2 × world ×` [`DMA_SLOT_BYTES`], allocated on first use) in pipelined
//! chunks, and reduces on the device in rank order: bit for bit the one-shot kernels' result,
//! counted as reason `copy_engine`. It takes precedence over the RCCL delegate.
//!
//! In `static` rank mode (one process per rank) there is no shared allocation: `open` answers
//! `Unavailable`, the planner refuses `hostmem` there and `auto` takes `rccl`.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use turbine_core::registry::Module;
use turbine_core::types::Vendor;
use turbine_tensor::{
    DType, DeviceBuffer, DeviceMemory, DevicePtr, DeviceSlice, MAPPED_ABORT_HOST,
    MAPPED_ABORT_TIMEOUT, MappedCollectives, MappedDma, MappedKind, MappedReduce, MappedRegion,
    MappedStep, StreamRef,
};

use turbine_core::config::ParallelConfig;

use super::{
    Collective, CollectiveBackend, CollectiveError, CollectiveInit, CollectiveLibrary,
    CollectiveMetrics, CollectiveOp, ReduceOp, RouteReason, UNIQUE_ID_BYTES,
};

/// The registered name and `backend` label.
const NAME: &str = "hostmem";
/// Bytes of one rank's slot: the largest step (a reduce-scatter step holds `world` parts). A
/// tensor-parallel prefill chunk of 2,048 tokens × 3,072 BF16 (12 MiB) fits one step.
pub const SLOT_BYTES: u64 = 32 << 20;
/// Flag words per rank: the most blocks one step runs (the R9700 has 64 compute units, so every
/// block of a step is resident at once and no block waits behind a spinning one).
pub const MAX_BLOCKS: u32 = 64;
/// `TURBINE_MAPPED_MAX_WORLD`.
pub const MAX_WORLD: usize = 8;
/// Whether groups opened from now on sequence their steps on the device when the kernel library
/// can (ABI v2.8); on by default.
static DEVICE_SEQ: AtomicBool = AtomicBool::new(true);

/// Turns device-sequenced steps on or off for the groups opened afterwards (every rank of a group
/// must open under the same setting). Off: host sequence numbers, which a graph cannot replay.
pub fn set_device_sequencing(on: bool) {
    DEVICE_SEQ.store(on, Ordering::Relaxed);
}

/// `parallel.collective.hostmem_dma_min_bytes` for the libraries loaded from now on (0: off).
static DMA_MIN_BYTES: AtomicU64 = AtomicU64::new(0);

/// Sets the copy-engine threshold (`parallel.collective.hostmem_dma_min_bytes`; `None`: off) of
/// the hostmem libraries loaded afterwards: an all-reduce of at least this many bytes (outside a
/// graph capture) runs on the copy-engine path, kernel ABI v2.8 `turbine_mapped_all_reduce_dma`,
/// instead of the one-shot steps or the RCCL delegate (reason `copy_engine`). Every rank of a
/// group must load under the same setting (one process in `local` mode).
pub fn set_dma_min_bytes(min: Option<u64>) {
    DMA_MIN_BYTES.store(
        min.unwrap_or(0).max(u64::from(min.is_some())),
        Ordering::Relaxed,
    );
}

/// How the copy-engine path brings the peers' chunks in (groups opened afterwards): `true`
/// (default) the reduction kernel reads them from the mapped slots while the copy engine
/// writes the next chunk out (the two directions of the link at once); `false` the copy engine
/// copies them into device scratch first. An A/B switch for `turbine-collbench`.
static DMA_PEER_READ: AtomicBool = AtomicBool::new(true);

/// Sets [`DMA_PEER_READ`] for the groups opened afterwards.
pub fn set_dma_peer_read(on: bool) {
    DMA_PEER_READ.store(on, Ordering::Relaxed);
}

/// Bytes of one rank's copy-engine slot: larger all-reduces run as several calls.
pub const DMA_SLOT_BYTES: u64 = 16 << 20;
/// The copy-engine path's pipeline chunk: the message over [`DMA_TARGET_CHUNKS`] chunks,
/// within [`DMA_MIN_CHUNK`] ..= [`DMA_MAX_CHUNK`] bytes (so at most 64 chunks per call).
pub const DMA_MIN_CHUNK: u64 = 64 << 10;
pub const DMA_MAX_CHUNK: u64 = 2 << 20;
pub const DMA_TARGET_CHUNKS: u64 = 8;

/// The chunk bytes of a copy-engine call over `n` bytes (a multiple of 16, the same on every
/// rank).
pub fn dma_chunk_bytes(n: u64) -> u64 {
    n.div_ceil(DMA_TARGET_CHUNKS)
        .next_multiple_of(16)
        .clamp(DMA_MIN_CHUNK, DMA_MAX_CHUNK)
}

/// Identifies a hostmem group id (bytes 20..28), next to the creating process id.
const ID_MARKER: &[u8; 8] = b"hostmem\0";

/// The largest message (nccl-tests bytes: the all-reduce / broadcast / send buffer, the
/// all-gather receive buffer, the reduce-scatter send buffer) `op` keeps on hostmem when
/// `parallel.collective.hostmem_max_bytes` is `auto`: the crossover measured against RCCL on
/// novanas (2-GPU, GPU0 Gen5 x8 + GPU1 Gen4 x8, BF16, back-to-back ops, `scripts/lab-cluster.sh
/// collbench-hostmem-novanas`, 2026-09-28). Larger messages go to RCCL — except send / recv,
/// which stay on hostmem (reason `no_peer_access`, module docs).
pub fn auto_max_bytes(op: CollectiveOp) -> u64 {
    match op {
        CollectiveOp::AllReduce => 128 << 10,
        CollectiveOp::AllGather | CollectiveOp::ReduceScatter => 256 << 10,
        CollectiveOp::Broadcast | CollectiveOp::Send | CollectiveOp::Recv => 32 << 10,
        CollectiveOp::Barrier => u64::MAX,
    }
}

/// Offsets of one group's region.
#[derive(Clone, Copy, Debug)]
struct Layout {
    abort: usize,
    slots: usize,
    slot_bytes: u64,
    total: usize,
}

impl Layout {
    fn new(world: usize, slot_bytes: u64) -> Layout {
        let abort = world * MAX_BLOCKS as usize * 8;
        let slots = (abort + 4).next_multiple_of(4096);
        Layout {
            abort,
            slots,
            slot_bytes,
            total: slots + 2 * world * slot_bytes as usize,
        }
    }
}

/// The `hostmem` module of the `collective_backend` registry. Its kernels come from the kernel
/// library of each rank's device context (ABI v2.7); its library is RCCL's, the delegate of
/// large messages (`parallel.rccl_library`).
pub struct HostmemBackend;

impl Module for HostmemBackend {
    fn name(&self) -> &'static str {
        NAME
    }
}

impl CollectiveBackend for HostmemBackend {
    fn vendors(&self) -> &'static [Vendor] {
        &[Vendor::Amd]
    }
    fn configured_library<'a>(&self, cfg: &'a ParallelConfig) -> Option<&'a Path> {
        cfg.rccl_library.as_deref()
    }
    /// Loads RCCL, the delegate of large messages: an explicit library that fails is fatal;
    /// without one, a failed default search leaves hostmem with every message (WARN).
    fn load(&self, explicit: Option<&Path>) -> Result<Arc<dyn CollectiveLibrary>, CollectiveError> {
        let delegate = match super::RCCL.load(explicit) {
            Ok(lib) => Some(lib),
            Err(e) if explicit.is_some() => return Err(e),
            Err(e) => {
                tracing::warn!(
                    event = "collective_route_unavailable",
                    backend = NAME,
                    delegate = "rccl",
                    error = %e,
                    "RCCL did not load: hostmem keeps every message, whatever its size"
                );
                None
            }
        };
        let dma_min = match DMA_MIN_BYTES.load(Ordering::Relaxed) {
            0 => None,
            n => Some(n),
        };
        Ok(Arc::new(HostmemLibrary {
            slot_bytes: SLOT_BYTES,
            dma_min,
            delegate,
        }))
    }
    fn one_process_only(&self) -> bool {
        true
    }
}

/// A group being opened or open: its region and which ranks joined.
struct Group {
    world: usize,
    layout: Layout,
    region: MappedRegion,
    joined: Mutex<Vec<bool>>,
    /// Point-to-point regions by (lower rank, higher rank), allocated on first use.
    pairs: Mutex<std::collections::HashMap<(usize, usize), MappedRegion>>,
    /// The copy-engine slots (`2 × world × DMA_SLOT_BYTES`), allocated on first use.
    dma: Mutex<Option<MappedRegion>>,
}

/// Open groups by unique id (process-local); an entry dies with its last rank.
type Groups = Vec<([u8; UNIQUE_ID_BYTES], Weak<Group>)>;
static GROUPS: Mutex<Groups> = Mutex::new(Vec::new());
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

struct HostmemLibrary {
    slot_bytes: u64,
    /// All-reduces of at least this many bytes take the copy-engine path (`None`: never).
    dma_min: Option<u64>,
    /// The library of large messages (RCCL), if it loaded.
    delegate: Option<Arc<dyn CollectiveLibrary>>,
}

/// Delegate group ids by hostmem id (process-local): the first rank to open makes it, and the
/// entry leaves once every rank has taken it.
type DelegateIds = Vec<([u8; UNIQUE_ID_BYTES], [u8; UNIQUE_ID_BYTES], usize)>;
static DELEGATE_IDS: Mutex<DelegateIds> = Mutex::new(Vec::new());

impl HostmemLibrary {
    /// Opens rank `init.rank` of the delegate group paired with `init.unique_id`.
    fn open_delegate(
        &self,
        delegate: &Arc<dyn CollectiveLibrary>,
        init: &CollectiveInit,
    ) -> Result<Arc<dyn Collective>, CollectiveError> {
        let id = {
            let mut ids = DELEGATE_IDS.lock().unwrap_or_else(|p| p.into_inner());
            let at = match ids.iter().position(|(h, _, _)| *h == init.unique_id) {
                Some(at) => at,
                None => {
                    ids.push((init.unique_id, delegate.unique_id()?, 0));
                    ids.len() - 1
                }
            };
            ids[at].2 += 1;
            let id = ids[at].1;
            if ids[at].2 == init.world {
                ids.swap_remove(at);
            }
            id
        };
        let delegated = CollectiveInit {
            rank: init.rank,
            world: init.world,
            unique_id: id,
            init_timeout: init.init_timeout,
            op_timeout: init.op_timeout,
            clock: Arc::clone(&init.clock),
            metrics: init.metrics.clone(),
            memory: init.memory.clone(),
            route_max_bytes: None,
        };
        // The delegate bounds its own init (RCCL: the init watchdog, and a helper thread given
        // up after the init timeout plus a grace; `ffi::NcclApi::open`).
        Arc::clone(delegate).open(delegated)
    }
}

impl CollectiveLibrary for HostmemLibrary {
    fn backend(&self) -> &'static str {
        NAME
    }
    fn version(&self) -> Option<String> {
        None
    }
    /// Unique within this process: a counter, the process id, the time and the marker.
    fn unique_id(&self) -> Result<[u8; UNIQUE_ID_BYTES], CollectiveError> {
        let mut id = [0u8; UNIQUE_ID_BYTES];
        let n = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        id[..8].copy_from_slice(&n.to_le_bytes());
        id[8..12].copy_from_slice(&std::process::id().to_le_bytes());
        id[12..20].copy_from_slice(&t.to_le_bytes());
        id[20..28].copy_from_slice(ID_MARKER);
        Ok(id)
    }
    /// Joins (or starts) the group of `init.unique_id` on `init.memory`. Ranks meet only inside
    /// the first operation, whose kernel waits at most `op_timeout`.
    fn open(self: Arc<Self>, init: CollectiveInit) -> Result<Arc<dyn Collective>, CollectiveError> {
        if init.world == 0 || init.world > MAX_WORLD || init.rank >= init.world {
            return Err(CollectiveError::ShapeMismatch);
        }
        let unavailable = |detail: String| CollectiveError::Unavailable {
            library: NAME.into(),
            detail,
        };
        if init.unique_id[20..28] != *ID_MARKER
            || init.unique_id[8..12] != std::process::id().to_le_bytes()
        {
            return Err(unavailable(
                "every rank of a hostmem group must be a thread of the process that made its \
                 id (parallel.ranks.mode local); separate rank processes share no mapped memory"
                    .into(),
            ));
        }
        let Some(memory) = init.memory.clone() else {
            return Err(unavailable(
                "hostmem needs each rank's device context (CollectiveInit::memory)".into(),
            ));
        };
        let Some(mc) = memory.mapped_collectives() else {
            let detail = format!(
                "the kernel library of device {} does not export the ABI v2.7 host-mapped group",
                memory.device().0
            );
            let Some(delegate) = &self.delegate else {
                return Err(unavailable(detail));
            };
            tracing::warn!(
                event = "collective_route",
                backend = NAME,
                delegate = delegate.backend(),
                reason = RouteReason::OpUnsupported.as_str(),
                rank = init.rank,
                "{detail}: the communicator is the delegate's for every call"
            );
            return self.open_delegate(delegate, &init);
        };
        let group = join(&init, self.slot_bytes, mc)?;
        let base = mc
            .mapped_device_addr(&group.region)
            .map_err(|e| CollectiveError::Backend {
                code: -1,
                message: format!("hostmem: mapping the region into device: {e}"),
            })?;
        // A threshold no message can exceed needs no delegate communicator.
        let delegate = match &self.delegate {
            Some(d) if init.route_max_bytes != Some(u64::MAX) => {
                Some(self.open_delegate(d, &init)?)
            }
            _ => None,
        };
        // The group channel's device step counter (v2.8): two zeroed words.
        let device_seq = if DEVICE_SEQ.load(Ordering::Relaxed) && mc.mapped_dseq_supported() {
            let counter = DeviceBuffer::alloc(&memory, 16)
                .and_then(|b| b.whole().write_bytes(&[0u8; 16]).map(|()| b))
                .map_err(|e| CollectiveError::Backend {
                    code: -1,
                    message: format!("hostmem: the device step counter: {e}"),
                })?;
            Some(counter)
        } else {
            None
        };
        // The copy-engine path needs the device counter and the v2.8 copy-engine step.
        let dma_min = self
            .dma_min
            .filter(|_| device_seq.is_some() && mc.mapped_dma_supported());
        tracing::info!(
            event = "collective_init",
            backend = NAME,
            rank = init.rank,
            world = init.world,
            device_sequenced = device_seq.is_some(),
            dma_min_bytes = ?dma_min,
            region_bytes = group.layout.total,
            delegate = delegate.as_ref().map_or("none", |d| d.backend()),
            route_max_bytes = ?init.route_max_bytes,
            "communicator ready"
        );
        Ok(Arc::new(HostmemCollective {
            rank: init.rank,
            world: init.world,
            group,
            memory,
            base,
            seq: AtomicU64::new(1),
            device_seq,
            op_timeout: init.op_timeout,
            metrics: init.metrics,
            reported: AtomicBool::new(false),
            pairs: Mutex::new(std::collections::HashMap::new()),
            delegate,
            p2p_kept_logged: AtomicBool::new(false),
            graph_refusal_logged: AtomicBool::new(false),
            dma_min,
            dma: Mutex::new(None),
            dma_peer_read: DMA_PEER_READ.load(Ordering::Relaxed),
            route_max: init.route_max_bytes,
        }))
    }
}

/// The group of `init.unique_id`, allocating its region through `mc` if this rank is first.
fn join(
    init: &CollectiveInit,
    slot_bytes: u64,
    mc: &dyn MappedCollectives,
) -> Result<Arc<Group>, CollectiveError> {
    let mut groups = GROUPS.lock().unwrap_or_else(|p| p.into_inner());
    groups.retain(|(_, g)| g.strong_count() > 0);
    let existing = groups
        .iter()
        .find(|(id, _)| *id == init.unique_id)
        .and_then(|(_, g)| g.upgrade());
    let group = match existing {
        Some(g) => g,
        None => {
            let layout = Layout::new(init.world, slot_bytes);
            let region = mc
                .alloc_mapped(layout.total)
                .map_err(|e| CollectiveError::Backend {
                    code: -1,
                    message: format!("hostmem: allocating {} mapped bytes: {e}", layout.total),
                })?;
            let g = Arc::new(Group {
                world: init.world,
                layout,
                region,
                joined: Mutex::new(vec![false; init.world]),
                pairs: Mutex::new(std::collections::HashMap::new()),
                dma: Mutex::new(None),
            });
            groups.push((init.unique_id, Arc::downgrade(&g)));
            g
        }
    };
    let mut joined = group.joined.lock().unwrap_or_else(|p| p.into_inner());
    if group.world != init.world || joined[init.rank] {
        return Err(CollectiveError::Backend {
            code: -1,
            message: format!(
                "hostmem group: rank {} of {} does not fit a group of {} (or joined twice)",
                init.rank, init.world, group.world
            ),
        });
    }
    joined[init.rank] = true;
    drop(joined);
    Ok(group)
}

/// One rank of a hostmem group. Operations enqueue on the device's compute stream and return
/// at once; a rank's device work never waits longer than about `op_timeout` for a peer.
pub struct HostmemCollective {
    rank: usize,
    world: usize,
    group: Arc<Group>,
    memory: Arc<dyn DeviceMemory>,
    /// The region's address on this rank's device.
    base: DevicePtr,
    /// The next step's sequence number (the same on every rank); with `device_seq` only a count
    /// of the steps this host enqueued (graph replays advance the device counter alone).
    seq: AtomicU64,
    /// The group channel's device step counter (kernel ABI v2.8), when its steps are sequenced
    /// on the device.
    device_seq: Option<DeviceBuffer>,
    op_timeout: Duration,
    metrics: Option<CollectiveMetrics>,
    /// The abort was logged and counted.
    reported: AtomicBool,
    /// This rank's pair channels by peer.
    pairs: Mutex<std::collections::HashMap<usize, Arc<Pair>>>,
    /// The RCCL communicator of large messages, if RCCL loaded.
    delegate: Option<Arc<dyn Collective>>,
    /// A point-to-point call above the threshold kept on hostmem was logged.
    p2p_kept_logged: AtomicBool,
    /// A delegate call refused under graph capture was logged.
    graph_refusal_logged: AtomicBool,
    /// The copy-engine threshold (`None`: off; also off without device sequencing or the v2.8
    /// copy-engine step).
    dma_min: Option<u64>,
    /// This rank's copy-engine state, made on the first copy-engine call.
    dma: Mutex<Option<DmaState>>,
    /// The copy-engine path's reduction reads the peers' slots directly ([`DMA_PEER_READ`]).
    dma_peer_read: bool,
    /// The operator's threshold; `None` is [`auto_max_bytes`].
    route_max: Option<u64>,
}

impl std::fmt::Debug for HostmemCollective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostmemCollective")
            .field("rank", &self.rank)
            .field("world", &self.world)
            .finish_non_exhaustive()
    }
}

/// The kinds a step can be, by the ABI code stored in the abort word.
fn op_of_code(code: u32) -> &'static str {
    match MappedKind::from_abi_code(code) {
        Some(MappedKind::AllReduce) => CollectiveOp::AllReduce.as_str(),
        Some(MappedKind::AllGather) => CollectiveOp::AllGather.as_str(),
        Some(MappedKind::ReduceScatter) => CollectiveOp::ReduceScatter.as_str(),
        Some(MappedKind::Broadcast { .. }) => CollectiveOp::Broadcast.as_str(),
        None => "step",
    }
}

fn reduce_of(op: ReduceOp) -> MappedReduce {
    match op {
        ReduceOp::Sum => MappedReduce::Sum,
        ReduceOp::Max => MappedReduce::Max,
    }
}

/// Byte `offset` of `slice`'s address.
fn at(slice: &DeviceSlice, offset: u64) -> DevicePtr {
    slice.ptr().offset(offset)
}

/// The flags, slots and sequence counter one call's steps run over: the group's own, or a
/// pair channel of two ranks (point-to-point). The abort word is always the group's.
struct Chan<'a> {
    /// This rank's index in the channel.
    rank: u32,
    world: u32,
    slots: DevicePtr,
    slot_bytes: u64,
    flags: DevicePtr,
    seq: &'a AtomicU64,
    /// The device step counter, when the channel's steps are device-sequenced.
    counter: Option<DevicePtr>,
}

/// A pair channel of this rank: a 2-rank exchange region (allocated by whichever rank of the
/// pair first needs it) mapped into this device, with this side's step counter.
struct Pair {
    _region: MappedRegion,
    base: DevicePtr,
    seq: AtomicU64,
}

/// One rank's copy-engine state: the group's copy-engine slots mapped into this device, the
/// device scratch the peers' chunks land in, and the count of copy-engine calls (the slot
/// parity, equal on every rank).
struct DmaState {
    _region: MappedRegion,
    slots: DevicePtr,
    scratch: DeviceBuffer,
    calls: u64,
}

/// Bytes of one rank's point-to-point slot (a pair region holds 2 × 2 of them).
pub const P2P_SLOT_BYTES: u64 = 8 << 20;

/// Offset of a pair region's slots (after 2 × `MAX_BLOCKS` flag words).
const PAIR_SLOTS: u64 = 4096;

/// What one call enqueues: its kind and per-step geometry.
struct Plan {
    kind: MappedKind,
    reduce: MappedReduce,
    dtype: DType,
    send: DevicePtr,
    recv: DevicePtr,
    /// Bytes of one rank's part over the whole call.
    part: u64,
    send_stride: u64,
    recv_stride: u64,
    /// The most part bytes one step carries.
    chunk: u64,
}

impl HostmemCollective {
    fn mapped(&self) -> &dyn MappedCollectives {
        self.memory
            .mapped_collectives()
            .expect("open checked that the device exports the host-mapped group")
    }

    /// `Err` once the group is aborted (by a kernel timeout on any rank or `abort`).
    fn check(&self) -> Result<(), CollectiveError> {
        let word = self.group.region.load_u32(self.group.layout.abort);
        if word == 0 {
            return Ok(());
        }
        let (reason, kind, who) = (word >> 24, (word >> 16) & 0xff, (word & 0xffff) as usize);
        let err = if reason == MAPPED_ABORT_TIMEOUT && who == self.rank {
            CollectiveError::Timeout {
                op: op_of_code(kind),
                after: self.op_timeout,
            }
        } else {
            CollectiveError::RemoteAbort { rank: who }
        };
        if !self.reported.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                event = "collective_aborted",
                backend = NAME,
                rank = self.rank,
                reason = if reason == MAPPED_ABORT_TIMEOUT { "timeout" } else { "abort" },
                by_rank = who,
                error = %err,
                "hostmem group aborted"
            );
            if let (Some(m), Some(kind)) = (&self.metrics, err.kind()) {
                m.error(NAME, kind);
            }
        }
        Err(err)
    }

    fn stream_ok(&self, stream: &StreamRef) -> Result<(), CollectiveError> {
        if stream.device() == self.memory.device() {
            Ok(())
        } else {
            Err(CollectiveError::Backend {
                code: -1,
                message: format!(
                    "hostmem rank {} runs on device {}, not on the stream's device {}",
                    self.rank,
                    self.memory.device().0,
                    stream.device().0
                ),
            })
        }
    }

    /// Enqueues every step of `plan`, then records the call's metrics.
    fn run(
        &self,
        op: CollectiveOp,
        metric_bytes: usize,
        stream: &StreamRef,
        plan: Plan,
        chan: Chan<'_>,
    ) -> Result<(), CollectiveError> {
        self.stream_ok(stream)?;
        self.check()?;
        let started = Instant::now();
        let layout = self.group.layout;
        let mut off = 0u64;
        while off < plan.part {
            let n = plan.chunk.min(plan.part - off);
            let seq = chan.seq.fetch_add(1, Ordering::Relaxed);
            let step = MappedStep {
                kind: plan.kind,
                reduce: plan.reduce,
                dtype: plan.dtype,
                rank: chan.rank,
                world: chan.world,
                send: plan.send.offset(off),
                recv: plan.recv.offset(off),
                bytes: n,
                send_stride: plan.send_stride,
                recv_stride: plan.recv_stride,
                slots: chan.slots,
                slot_bytes: chan.slot_bytes,
                flags: chan.flags,
                max_blocks: MAX_BLOCKS,
                abort_word: self.base.offset(layout.abort as u64),
                seq,
                timeout: self.op_timeout,
            };
            let enqueued = match chan.counter {
                Some(counter) => self.mapped().enqueue_mapped_step_dseq(&step, counter),
                None => self.mapped().enqueue_mapped_step(&step),
            };
            if let Err(e) = enqueued {
                // The peers expect this step: fail the group rather than let them time out.
                self.abort();
                let err = CollectiveError::Backend {
                    code: -1,
                    message: format!("hostmem {} step {seq}: {e}", op.as_str()),
                };
                if let Some(m) = &self.metrics {
                    m.error(NAME, super::CollectiveErrorKind::Backend);
                }
                return Err(err);
            }
            off += n;
        }
        if let Some(m) = &self.metrics {
            m.observe(
                op,
                NAME,
                metric_bytes as u64,
                started.elapsed().as_secs_f64(),
            );
        }
        Ok(())
    }

    /// The delegate when a call of `op` over `bytes` (nccl-tests bytes) goes there, `None`
    /// when hostmem runs it; counts the choice.
    /// While the stream is captured into a graph, a call the delegate would run is refused
    /// ([`RouteReason::GraphCaptureRefused`]): RCCL operations replayed from a graph gave wrong
    /// results on novanas.
    fn route(
        &self,
        op: CollectiveOp,
        bytes: usize,
    ) -> Result<Option<&dyn Collective>, CollectiveError> {
        let Some(delegate) = self.delegate.as_deref() else {
            return Ok(None);
        };
        let max = self.route_max.unwrap_or_else(|| auto_max_bytes(op));
        if bytes as u64 > max && self.mapped().mapped_capturing() {
            return Err(super::refuse_captured(
                op,
                NAME,
                self.metrics.as_ref(),
                &self.graph_refusal_logged,
            ));
        }
        let (to, backend, reason) = if bytes as u64 > max {
            (
                Some(delegate),
                delegate.backend(),
                RouteReason::AboveThreshold,
            )
        } else {
            (None, NAME, RouteReason::BelowThreshold)
        };
        if let Some(m) = &self.metrics {
            m.route(op, backend, reason);
        }
        tracing::trace!(
            event = "collective_route",
            op = op.as_str(),
            bytes,
            backend,
            reason = reason.as_str(),
            "collective routed"
        );
        Ok(to)
    }

    /// Counts where a point-to-point call of `op` over `bytes` runs: always here. Above the
    /// threshold, with a delegate loaded, the reason is `no_peer_access` (logged once): the
    /// delegate's send / recv need a peer path between the devices, which the group has no
    /// knowledge of (and novanas lacks).
    fn route_p2p(&self, op: CollectiveOp, bytes: usize) {
        let max = self.route_max.unwrap_or_else(|| auto_max_bytes(op));
        let reason = if self.delegate.is_some() && bytes as u64 > max {
            if !self.p2p_kept_logged.swap(true, Ordering::Relaxed) {
                tracing::info!(
                    event = "collective_route",
                    op = op.as_str(),
                    bytes,
                    backend = NAME,
                    reason = RouteReason::NoPeerAccess.as_str(),
                    rank = self.rank,
                    "point-to-point above the threshold stays on hostmem: the delegate's \
                     send / recv need a peer path between the devices"
                );
            }
            RouteReason::NoPeerAccess
        } else {
            RouteReason::BelowThreshold
        };
        if let Some(m) = &self.metrics {
            m.route(op, NAME, reason);
        }
    }

    /// Whether an all-reduce of `bytes` takes the copy-engine path: at least the threshold, and
    /// not while the stream is captured (the path is not capturable). Counted when it does.
    fn route_dma(&self, bytes: usize) -> bool {
        let dma = self.dma_min.is_some_and(|min| bytes as u64 >= min)
            && !self.mapped().mapped_capturing();
        if dma && let Some(m) = &self.metrics {
            m.route(CollectiveOp::AllReduce, NAME, RouteReason::CopyEngine);
        }
        dma
    }

    /// The copy-engine all-reduce of `buf` in place (module docs): calls of at most
    /// [`DMA_SLOT_BYTES`], each over [`dma_chunk_bytes`] chunks.
    fn all_reduce_dma(
        &self,
        buf: &mut DeviceSlice,
        dtype: DType,
        op: ReduceOp,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        self.stream_ok(stream)?;
        self.check()?;
        let started = Instant::now();
        let counter = self
            .device_seq
            .as_ref()
            .map(|b| b.whole().ptr())
            .expect("the copy-engine path needs the device step counter (open)");
        let backend = |what: &str, e: &dyn std::fmt::Display| CollectiveError::Backend {
            code: -1,
            message: format!("hostmem copy engine: {what}: {e}"),
        };
        let mut guard = self.dma.lock().unwrap_or_else(|p| p.into_inner());
        if guard.is_none() {
            let region = {
                let mut shared = self.group.dma.lock().unwrap_or_else(|p| p.into_inner());
                match &*shared {
                    Some(r) => r.clone(),
                    None => {
                        let bytes = (2 * self.world as u64 * DMA_SLOT_BYTES) as usize;
                        let mc = self.mapped();
                        let r = if self.dma_peer_read {
                            mc.alloc_mapped(bytes)
                        } else {
                            mc.alloc_dma_region(bytes)
                        }
                        .map_err(|e| backend("allocating the slots", &e))?;
                        *shared = Some(r.clone());
                        r
                    }
                }
            };
            // Kernels read mapped slots at their device address; the copy engines reach
            // the portable ones at their host address.
            let slots = if self.dma_peer_read {
                self.mapped()
                    .mapped_device_addr(&region)
                    .map_err(|e| backend("mapping the slots", &e))?
            } else {
                DevicePtr::from_addr(region.host_addr())
            };
            let scratch = DeviceBuffer::alloc(
                &self.memory,
                (self.world.max(2) - 1) * DMA_MAX_CHUNK as usize,
            )
            .map_err(|e| backend("the scratch buffer", &e))?;
            *guard = Some(DmaState {
                _region: region,
                slots,
                scratch,
                calls: 0,
            });
        }
        let state = guard.as_mut().expect("made above");
        let layout = self.group.layout;
        let total = buf.len() as u64;
        let mut off = 0u64;
        while off < total {
            let n = DMA_SLOT_BYTES.min(total - off);
            state.calls += 1;
            let step = MappedStep {
                kind: MappedKind::AllReduce,
                reduce: reduce_of(op),
                dtype,
                rank: self.rank as u32,
                world: self.world as u32,
                send: at(buf, off),
                recv: at(buf, off),
                bytes: n,
                send_stride: 0,
                recv_stride: 0,
                slots: state.slots,
                slot_bytes: DMA_SLOT_BYTES,
                flags: self.base,
                max_blocks: MAX_BLOCKS,
                abort_word: self.base.offset(layout.abort as u64),
                seq: state.calls,
                timeout: self.op_timeout,
            };
            let dma = MappedDma {
                scratch: state.scratch.whole().ptr(),
                chunk_bytes: dma_chunk_bytes(n),
                seq_counter: counter,
                peer_read: self.dma_peer_read,
            };
            if let Err(e) = self.mapped().enqueue_mapped_all_reduce_dma(&step, &dma) {
                // The peers expect this call: fail the group rather than let them time out.
                self.abort();
                if let Some(m) = &self.metrics {
                    m.error(NAME, super::CollectiveErrorKind::Backend);
                }
                return Err(backend(&format!("call {}", state.calls), &e));
            }
            off += n;
        }
        if let Some(m) = &self.metrics {
            m.observe(
                CollectiveOp::AllReduce,
                NAME,
                total,
                started.elapsed().as_secs_f64(),
            );
        }
        Ok(())
    }

    /// The group's own channel.
    fn group_chan(&self) -> Chan<'_> {
        let layout = self.group.layout;
        Chan {
            rank: self.rank as u32,
            world: self.world as u32,
            slots: self.base.offset(layout.slots as u64),
            slot_bytes: layout.slot_bytes,
            flags: self.base,
            seq: &self.seq,
            counter: self.device_seq.as_ref().map(|b| b.whole().ptr()),
        }
    }

    /// The pair channel to `peer`, allocating the pair's region if this rank is first.
    fn pair(&self, peer: usize) -> Result<Arc<Pair>, CollectiveError> {
        let mut mine = self.pairs.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(p) = mine.get(&peer) {
            return Ok(Arc::clone(p));
        }
        let slot = self.p2p_slot_bytes();
        let key = (self.rank.min(peer), self.rank.max(peer));
        let region = {
            let mut shared = self.group.pairs.lock().unwrap_or_else(|p| p.into_inner());
            match shared.get(&key) {
                Some(r) => r.clone(),
                None => {
                    let bytes = (PAIR_SLOTS + 4 * slot) as usize;
                    let r = self.mapped().alloc_mapped(bytes).map_err(|e| {
                        CollectiveError::Backend {
                            code: -1,
                            message: format!("hostmem: allocating a {bytes}-byte pair region: {e}"),
                        }
                    })?;
                    shared.insert(key, r.clone());
                    r
                }
            }
        };
        let base =
            self.mapped()
                .mapped_device_addr(&region)
                .map_err(|e| CollectiveError::Backend {
                    code: -1,
                    message: format!("hostmem: mapping a pair region: {e}"),
                })?;
        let p = Arc::new(Pair {
            _region: region,
            base,
            seq: AtomicU64::new(1),
        });
        mine.insert(peer, Arc::clone(&p));
        Ok(p)
    }

    /// A point-to-point slot: at most `P2P_SLOT_BYTES`, never more than the group's slot.
    fn p2p_slot_bytes(&self) -> u64 {
        self.group.layout.slot_bytes.min(P2P_SLOT_BYTES)
    }

    /// One point-to-point transfer from rank `from` over the pair channel to `peer`: a
    /// broadcast of the 2-rank channel rooted at the sender, in place in `buf`.
    fn p2p(
        &self,
        op: CollectiveOp,
        buf: &DeviceSlice,
        peer: usize,
        from: usize,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        if peer >= self.world || peer == self.rank {
            return Err(CollectiveError::ShapeMismatch);
        }
        self.check()?;
        let pair = self.pair(peer)?;
        // Channel ranks: the lower global rank is 0.
        let local = |r: usize| u32::from(r > self.rank.min(peer));
        let slot = self.p2p_slot_bytes();
        let chan = Chan {
            rank: local(self.rank),
            world: 2,
            slots: pair.base.offset(PAIR_SLOTS),
            slot_bytes: slot,
            flags: pair.base,
            seq: &pair.seq,
            // Point-to-point (pipeline stages) is never captured.
            counter: None,
        };
        let plan = Plan {
            kind: MappedKind::Broadcast { root: local(from) },
            reduce: MappedReduce::Sum,
            dtype: DType::BF16,
            send: at(buf, 0),
            recv: at(buf, 0),
            part: buf.len() as u64,
            send_stride: 0,
            recv_stride: 0,
            chunk: slot,
        };
        self.run(op, buf.len(), stream, plan, chan)
    }

    /// `dtype` is BF16 or FP32 and `len` whole elements of it.
    fn reduction(dtype: DType, len: usize) -> Result<(), CollectiveError> {
        if matches!(dtype, DType::BF16 | DType::F32) && len.is_multiple_of(dtype.size_bytes()) {
            Ok(())
        } else {
            Err(CollectiveError::ShapeMismatch)
        }
    }
}

impl Collective for HostmemCollective {
    fn backend(&self) -> &'static str {
        NAME
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world
    }

    fn all_reduce(
        &self,
        buf: &mut DeviceSlice,
        dtype: DType,
        op: ReduceOp,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        Self::reduction(dtype, buf.len())?;
        if self.route_dma(buf.len()) {
            return self.all_reduce_dma(buf, dtype, op, stream);
        }
        if let Some(d) = self.route(CollectiveOp::AllReduce, buf.len())? {
            return d.all_reduce(buf, dtype, op, stream);
        }
        let plan = Plan {
            kind: MappedKind::AllReduce,
            reduce: reduce_of(op),
            dtype,
            send: at(buf, 0),
            recv: at(buf, 0),
            part: buf.len() as u64,
            send_stride: 0,
            recv_stride: 0,
            chunk: self.group.layout.slot_bytes,
        };
        self.run(
            CollectiveOp::AllReduce,
            buf.len(),
            stream,
            plan,
            self.group_chan(),
        )
    }

    fn all_gather(
        &self,
        send: &DeviceSlice,
        recv: &mut DeviceSlice,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        let n = send.len();
        if recv.len() != n * self.world {
            return Err(CollectiveError::ShapeMismatch);
        }
        if let Some(d) = self.route(CollectiveOp::AllGather, recv.len())? {
            return d.all_gather(send, recv, stream);
        }
        let plan = Plan {
            kind: MappedKind::AllGather,
            reduce: MappedReduce::Sum,
            dtype: DType::BF16,
            send: at(send, 0),
            recv: at(recv, 0),
            part: n as u64,
            send_stride: 0,
            recv_stride: n as u64,
            chunk: self.group.layout.slot_bytes,
        };
        self.run(CollectiveOp::AllGather, n, stream, plan, self.group_chan())
    }

    fn reduce_scatter(
        &self,
        send: &DeviceSlice,
        recv: &mut DeviceSlice,
        dtype: DType,
        op: ReduceOp,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        Self::reduction(dtype, recv.len())?;
        let n = recv.len();
        if send.len() != n * self.world {
            return Err(CollectiveError::ShapeMismatch);
        }
        if let Some(d) = self.route(CollectiveOp::ReduceScatter, send.len())? {
            return d.reduce_scatter(send, recv, dtype, op, stream);
        }
        // A step's slot holds `world` parts, each rounded up to 16 bytes.
        let chunk = (self.group.layout.slot_bytes / self.world as u64) / 16 * 16;
        let plan = Plan {
            kind: MappedKind::ReduceScatter,
            reduce: reduce_of(op),
            dtype,
            send: at(send, 0),
            recv: at(recv, 0),
            part: n as u64,
            send_stride: n as u64,
            recv_stride: 0,
            chunk,
        };
        self.run(
            CollectiveOp::ReduceScatter,
            send.len(),
            stream,
            plan,
            self.group_chan(),
        )
    }

    fn broadcast(
        &self,
        buf: &mut DeviceSlice,
        root: usize,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        if root >= self.world {
            return Err(CollectiveError::ShapeMismatch);
        }
        if let Some(d) = self.route(CollectiveOp::Broadcast, buf.len())? {
            return d.broadcast(buf, root, stream);
        }
        let plan = Plan {
            kind: MappedKind::Broadcast { root: root as u32 },
            reduce: MappedReduce::Sum,
            dtype: DType::BF16,
            send: at(buf, 0),
            recv: at(buf, 0),
            part: buf.len() as u64,
            send_stride: 0,
            recv_stride: 0,
            chunk: self.group.layout.slot_bytes,
        };
        self.run(
            CollectiveOp::Broadcast,
            buf.len(),
            stream,
            plan,
            self.group_chan(),
        )
    }

    /// An all-reduce of one FP32 on `stream`, then a synchronize of the device: returns once
    /// every rank has arrived, or with the abort (a missing peer: after about `op_timeout`).
    fn barrier(&self, stream: &StreamRef) -> Result<(), CollectiveError> {
        self.stream_ok(stream)?;
        self.check()?;
        let scratch =
            DeviceBuffer::alloc(&self.memory, 4).map_err(|e| CollectiveError::Backend {
                code: -1,
                message: format!("barrier scratch allocation: {e}"),
            })?;
        let s = scratch.whole();
        let plan = Plan {
            kind: MappedKind::AllReduce,
            reduce: MappedReduce::Sum,
            dtype: DType::F32,
            send: at(&s, 0),
            recv: at(&s, 0),
            part: 4,
            send_stride: 0,
            recv_stride: 0,
            chunk: 4,
        };
        self.run(CollectiveOp::Barrier, 4, stream, plan, self.group_chan())?;
        self.memory
            .synchronize()
            .map_err(|e| CollectiveError::Backend {
                code: -1,
                message: format!("barrier synchronize: {e}"),
            })?;
        drop(scratch);
        self.check()
    }

    /// A broadcast from this rank over the pair channel to `peer`: the kernel waits (bounded by
    /// `op_timeout`) until the peer's matching `recv` has arrived. Never the delegate's
    /// ([`HostmemCollective::route_p2p`]).
    fn send(
        &self,
        buf: &DeviceSlice,
        peer: usize,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        self.route_p2p(CollectiveOp::Send, buf.len());
        self.p2p(CollectiveOp::Send, buf, peer, self.rank, stream)
    }

    fn recv(
        &self,
        buf: &mut DeviceSlice,
        peer: usize,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        self.route_p2p(CollectiveOp::Recv, buf.len());
        self.p2p(CollectiveOp::Recv, buf, peer, peer, stream)
    }

    /// Nothing to arm for the kernels (each bounds its own waits by `op_timeout`); arms the
    /// delegate's step deadline.
    fn step_begin(&self) {
        if let Some(d) = &self.delegate {
            d.step_begin();
        }
    }

    /// `Err` when a step of this or another rank timed out, the group was aborted, or the
    /// delegate's step failed.
    fn step_end(&self) -> Result<(), CollectiveError> {
        let own = self.check();
        let delegated = self.delegate.as_ref().map_or(Ok(()), |d| d.step_end());
        own.and(delegated)
    }

    /// Stores the host abort into the abort word (unless a reason is already there): every
    /// waiting and later step of every rank ends, and every later call fails.
    fn abort(&self) {
        if let Some(d) = &self.delegate {
            d.abort();
        }
        let offset = self.group.layout.abort;
        if self.group.region.load_u32(offset) == 0 {
            self.group
                .region
                .store_u32(offset, (MAPPED_ABORT_HOST << 24) | self.rank as u32);
        }
    }
}

impl Drop for HostmemCollective {
    /// Waits for this rank's enqueued steps (bounded by `op_timeout` each) before the region
    /// can be freed with the group's last rank.
    fn drop(&mut self) {
        if let Err(e) = self.memory.synchronize() {
            tracing::warn!(
                event = "collective_destroy_failed",
                backend = NAME,
                rank = self.rank,
                error = %e,
                "synchronize before releasing the hostmem region failed"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use half::bf16;
    use turbine_core::clock::SystemClock;
    use turbine_kernels::test_support::{stub_mapped_context, stub_mapped_context_minor};
    use turbine_tensor::host::HostMemory;
    use turbine_tensor::{DType, DeviceBuffer, DeviceId, DeviceMemory};

    use super::*;
    use crate::collective::HostCollective;

    /// splitmix64 values in [-8, 8) with a fractional part, so sums round.
    fn values(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z >> 40) as f32 / (1u64 << 24) as f32) * 16.0 - 8.0
            })
            .collect()
    }

    fn encode(dtype: DType, v: &[f32]) -> Vec<u8> {
        match dtype {
            DType::BF16 => v
                .iter()
                .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
                .collect(),
            _ => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }

    fn init(
        rank: usize,
        world: usize,
        id: [u8; UNIQUE_ID_BYTES],
        timeout: Duration,
        memory: Option<Arc<dyn DeviceMemory>>,
    ) -> CollectiveInit {
        CollectiveInit {
            rank,
            world,
            unique_id: id,
            init_timeout: timeout,
            op_timeout: timeout,
            clock: Arc::new(SystemClock::new()),
            metrics: None,
            memory,
            route_max_bytes: None,
        }
    }

    /// hostmem without a delegate (the build host's RCCL is never loaded by these tests).
    fn plain() -> Arc<dyn CollectiveLibrary> {
        Arc::new(HostmemLibrary {
            slot_bytes: SLOT_BYTES,
            dma_min: None,
            delegate: None,
        })
    }

    /// One rank's results of every op, in order: all-reduce sum, all-reduce max, all-gather,
    /// reduce-scatter, broadcast from the last rank.
    type Outs = Vec<Vec<u8>>;

    fn ops(
        comm: &dyn Collective,
        mem: &Arc<dyn DeviceMemory>,
        dtype: DType,
        input: &[u8],
        scatter_input: &[u8],
    ) -> Outs {
        let world = comm.world_size();
        let stream = mem.compute_stream();
        let n = input.len();
        let buf = DeviceBuffer::alloc(mem, n).expect("alloc");
        let mut s = buf.whole();
        let mut out = Vec::new();
        for op in [ReduceOp::Sum, ReduceOp::Max] {
            s.write_bytes(input).expect("write");
            comm.all_reduce(&mut s, dtype, op, &stream)
                .expect("all_reduce");
            mem.synchronize().expect("sync");
            out.push(s.read_bytes().expect("read"));
        }
        let recv = DeviceBuffer::alloc(mem, n * world).expect("alloc");
        s.write_bytes(input).expect("write");
        comm.all_gather(&s, &mut recv.whole(), &stream)
            .expect("all_gather");
        mem.synchronize().expect("sync");
        out.push(recv.whole().read_bytes().expect("read"));
        let ssend = DeviceBuffer::alloc(mem, n * world).expect("alloc");
        ssend.whole().write_bytes(scatter_input).expect("write");
        let srecv = DeviceBuffer::alloc(mem, n).expect("alloc");
        comm.reduce_scatter(
            &ssend.whole(),
            &mut srecv.whole(),
            dtype,
            ReduceOp::Sum,
            &stream,
        )
        .expect("reduce_scatter");
        mem.synchronize().expect("sync");
        out.push(srecv.whole().read_bytes().expect("read"));
        s.write_bytes(input).expect("write");
        comm.broadcast(&mut s, world - 1, &stream)
            .expect("broadcast");
        mem.synchronize().expect("sync");
        out.push(s.read_bytes().expect("read"));
        if world > 1 {
            // Around the ring: even ranks send first, odd ranks receive first.
            let rank = comm.rank();
            let (next, prev) = ((rank + 1) % world, (rank + world - 1) % world);
            let got = DeviceBuffer::alloc(mem, n).expect("alloc");
            s.write_bytes(input).expect("write");
            if rank.is_multiple_of(2) {
                comm.send(&s, next, &stream).expect("send");
                comm.recv(&mut got.whole(), prev, &stream).expect("recv");
            } else {
                comm.recv(&mut got.whole(), prev, &stream).expect("recv");
                comm.send(&s, next, &stream).expect("send");
            }
            mem.synchronize().expect("sync");
            out.push(got.whole().read_bytes().expect("read"));
        }
        comm.barrier(&stream).expect("barrier");
        comm.step_end().expect("healthy");
        out
    }

    /// Every rank's outputs through `lib` (hostmem on stub device contexts of kernel ABI minor
    /// `minor`: 7 host-sequenced, 8 device-sequenced) or, with `None`, through the host
    /// reference backend.
    fn run_group(
        minor: u32,
        lib: Option<Arc<dyn CollectiveLibrary>>,
        world: usize,
        dtype: DType,
        inputs: &[Vec<u8>],
        scatter_inputs: &[Vec<u8>],
    ) -> Vec<Outs> {
        let host = HostCollective::group(world, Duration::from_secs(20));
        let id = lib.as_ref().map(|l| l.unique_id().expect("id"));
        std::thread::scope(|scope| {
            let handles: Vec<_> = host
                .into_iter()
                .enumerate()
                .map(|(r, h)| {
                    let (lib, i, si) = (lib.clone(), &inputs[r], &scatter_inputs[r]);
                    scope.spawn(move || match lib {
                        Some(lib) => {
                            let ctx = stub_mapped_context_minor(r as u32, minor);
                            let mem: Arc<dyn DeviceMemory> = ctx;
                            let comm = lib
                                .open(init(
                                    r,
                                    world,
                                    id.expect("id"),
                                    Duration::from_secs(20),
                                    Some(Arc::clone(&mem)),
                                ))
                                .expect("open");
                            ops(comm.as_ref(), &mem, dtype, i, si)
                        }
                        None => {
                            let mem: Arc<dyn DeviceMemory> =
                                HostMemory::new(DeviceId(r as u32), 1 << 30);
                            ops(&h, &mem, dtype, i, si)
                        }
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rank thread"))
                .collect()
        })
    }

    /// Bit for bit the host backend's results on every rank, for world 1–3, BF16 and FP32, odd
    /// sizes, and messages larger than a slot (a 4 KiB slot splits them into several steps, so
    /// both slot parities and the sequence numbers are exercised). Breaks if the reduction
    /// order, the chunking, the strides of all-gather / reduce-scatter or the broadcast root is
    /// wrong. Both sequencings: host sequence numbers (a v2.7 library) and the device step counter
    /// (v2.8) — a counter that does not advance per step would reuse a slot parity and a tag —
    /// and the copy-engine all-reduce (v2.8, every all-reduce of at least 16 bytes, among the
    /// one-shot steps of the other ops and the barrier on the same counter; 300,001 elements
    /// span several chunks, and 4,500,007 F32 elements several copy-engine calls).
    #[test]
    fn matches_host_backend() {
        for (minor, slot_bytes, dma_min) in [
            (8, SLOT_BYTES, None),
            (8, 4096, None),
            (7, 4096, None),
            (8, SLOT_BYTES, Some(16)),
        ] {
            let lib: Arc<dyn CollectiveLibrary> = Arc::new(HostmemLibrary {
                slot_bytes,
                dma_min,
                delegate: None,
            });
            let sizes: &[usize] = if dma_min.is_some() {
                &[1, 7, 4099, 300_001, 4_500_007]
            } else {
                &[1, 7, 4099]
            };
            for world in [1usize, 2, 3] {
                for dtype in [DType::F32, DType::BF16] {
                    for &elems in sizes {
                        if elems > 1_000_000 && (dtype == DType::BF16 || world != 2) {
                            continue;
                        }
                        let ctx = format!(
                            "v2.{minor} slot {slot_bytes} dma {dma_min:?} world {world} \
                             {dtype:?} {elems}"
                        );
                        let inputs: Vec<Vec<u8>> = (0..world)
                            .map(|r| encode(dtype, &values(r as u64 * 7919 + elems as u64, elems)))
                            .collect();
                        let scatter: Vec<Vec<u8>> = (0..world)
                            .map(|r| encode(dtype, &values(r as u64 * 104_729 + 3, elems * world)))
                            .collect();
                        let want = run_group(minor, None, world, dtype, &inputs, &scatter);
                        let got = run_group(
                            minor,
                            Some(Arc::clone(&lib)),
                            world,
                            dtype,
                            &inputs,
                            &scatter,
                        );
                        for (r, (g, w)) in got.iter().zip(&want).enumerate() {
                            assert_eq!(g, w, "{ctx} rank {r}");
                        }
                    }
                }
            }
        }
    }

    /// A peer that never arrives: the step gives up after the op timeout (not a hang), the
    /// rank reports `Timeout` naming the op from `step_end` and every later call, and the group
    /// stays aborted. Breaks if a wait is unbounded or the timeout is not surfaced.
    #[test]
    fn missing_peer_times_out() {
        let lib = plain();
        let id = lib.unique_id().expect("id");
        let mem: Arc<dyn DeviceMemory> = stub_mapped_context(0);
        let comm = Arc::clone(&lib)
            .open(init(
                0,
                2,
                id,
                Duration::from_millis(300),
                Some(Arc::clone(&mem)),
            ))
            .expect("open");
        let stream = mem.compute_stream();
        let buf = DeviceBuffer::alloc(&mem, 64).expect("alloc");
        let started = Instant::now();
        comm.step_begin();
        comm.all_reduce(&mut buf.whole(), DType::F32, ReduceOp::Sum, &stream)
            .expect("enqueued");
        mem.synchronize().expect("the step ends by itself");
        let waited = started.elapsed();
        assert!(
            waited >= Duration::from_millis(300) && waited < Duration::from_secs(3),
            "{waited:?}"
        );
        let err = comm.step_end().expect_err("timed out");
        assert!(
            matches!(err, CollectiveError::Timeout { op: "all_reduce", after } if after == Duration::from_millis(300)),
            "{err:?}"
        );
        let later = comm
            .all_gather(&buf.whole(), &mut buf.whole(), &stream)
            .expect_err("aborted");
        assert!(
            matches!(
                later,
                CollectiveError::ShapeMismatch | CollectiveError::Timeout { .. }
            ),
            "{later:?}"
        );
        assert!(matches!(
            comm.barrier(&stream),
            Err(CollectiveError::Timeout { .. })
        ));
    }

    /// `abort` on one rank releases a peer spinning in a step long before its op timeout, and
    /// the peer reports `RemoteAbort` naming the aborting rank.
    #[test]
    fn abort_releases_a_spinning_peer() {
        let lib = plain();
        let id = lib.unique_id().expect("id");
        let mems: Vec<Arc<dyn DeviceMemory>> = (0..2)
            .map(|r| stub_mapped_context(r) as Arc<dyn DeviceMemory>)
            .collect();
        let comms: Vec<Arc<dyn Collective>> = (0..2)
            .map(|r| {
                Arc::clone(&lib)
                    .open(init(
                        r,
                        2,
                        id,
                        Duration::from_secs(30),
                        Some(Arc::clone(&mems[r])),
                    ))
                    .expect("open")
            })
            .collect();
        let started = Instant::now();
        std::thread::scope(|s| {
            let (comm, mem) = (&comms[1], &mems[1]);
            let waiter = s.spawn(move || {
                let buf = DeviceBuffer::alloc(mem, 16).expect("alloc");
                // The stub runs the step on this thread: it spins until rank 0's abort.
                comm.all_reduce(
                    &mut buf.whole(),
                    DType::F32,
                    ReduceOp::Sum,
                    &mem.compute_stream(),
                )
                .expect("enqueued");
                comm.step_end()
            });
            std::thread::sleep(Duration::from_millis(100));
            comms[0].abort();
            let err = waiter.join().expect("rank 1").expect_err("aborted");
            assert!(
                matches!(err, CollectiveError::RemoteAbort { rank: 0 }),
                "{err:?}"
            );
        });
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(matches!(
            comms[0].step_end(),
            Err(CollectiveError::RemoteAbort { rank: 0 })
        ));
    }

    /// Routing with the host backend as the delegate: calls above the threshold go to the
    /// delegate, the others stay on hostmem, the results are the host backend's either way, and
    /// `turbine_collective_route_total` counts each choice. Breaks if the threshold is compared
    /// on another size, a rank routes differently, or the counts are not kept.
    #[test]
    fn routes_above_the_threshold_to_the_delegate() {
        let delegate = crate::collective::HostBackend.load(None).expect("host");
        let lib: Arc<dyn CollectiveLibrary> = Arc::new(HostmemLibrary {
            slot_bytes: SLOT_BYTES,
            dma_min: None,
            delegate: Some(delegate),
        });
        let id = lib.unique_id().expect("id");
        let reg = turbine_observability::MetricsRegistry::new();
        let metrics = CollectiveMetrics::register(&reg);
        let outs: Vec<Vec<Vec<u8>>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..2usize)
                .map(|r| {
                    let (lib, metrics) = (Arc::clone(&lib), metrics.clone());
                    s.spawn(move || {
                        let mem: Arc<dyn DeviceMemory> = stub_mapped_context(r as u32);
                        let comm = lib
                            .open(CollectiveInit {
                                metrics: Some(metrics),
                                route_max_bytes: Some(64),
                                ..init(r, 2, id, Duration::from_secs(20), Some(Arc::clone(&mem)))
                            })
                            .expect("open");
                        assert_eq!(comm.backend(), "hostmem");
                        let stream = mem.compute_stream();
                        [16usize, 4096]
                            .iter()
                            .map(|&n| {
                                let v = encode(DType::F32, &values(r as u64 + n as u64, n / 4));
                                let buf = DeviceBuffer::alloc(&mem, n).expect("alloc");
                                buf.whole().write_bytes(&v).expect("write");
                                comm.all_reduce(
                                    &mut buf.whole(),
                                    DType::F32,
                                    ReduceOp::Sum,
                                    &stream,
                                )
                                .expect("all_reduce");
                                mem.synchronize().expect("sync");
                                comm.step_end().expect("healthy");
                                buf.whole().read_bytes().expect("read")
                            })
                            .collect()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rank"))
                .collect()
        });
        for (k, n) in [16usize, 4096].iter().enumerate() {
            let inputs: Vec<Vec<u8>> = (0..2)
                .map(|r| encode(DType::F32, &values(r as u64 + *n as u64, n / 4)))
                .collect();
            let f = |b: &[u8]| -> Vec<f32> {
                b.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            };
            let sum: Vec<f32> = f(&inputs[0])
                .iter()
                .zip(f(&inputs[1]))
                .map(|(a, b)| a + b)
                .collect();
            let want = encode(DType::F32, &sum);
            for (r, o) in outs.iter().enumerate() {
                assert_eq!(o[k], want, "{n} bytes rank {r}");
            }
        }
        let text = reg.render().expect("renders");
        for (backend, reason) in [("hostmem", "below_threshold"), ("host", "above_threshold")] {
            let line = format!(
                "turbine_collective_route_total{{op=\"all_reduce\",backend=\"{backend}\",reason=\"{reason}\"}} 2"
            );
            assert!(text.contains(&line), "{line}\n{text}");
        }
        assert_eq!(auto_max_bytes(CollectiveOp::AllReduce), 128 << 10);
        assert_eq!(auto_max_bytes(CollectiveOp::Barrier), u64::MAX);
    }

    /// Point-to-point above the threshold stays on hostmem even with a delegate loaded (the
    /// delegate's send / recv need a peer path; RCCL fails without one on novanas): a 4 KiB
    /// send / recv pair against a 64-byte threshold arrives bit for bit over hostmem's pair
    /// channel, and `turbine_collective_route_total` counts it on `hostmem` with reason
    /// `no_peer_access`, never on the delegate. Breaks if send / recv go back to the size
    /// routing.
    #[test]
    fn point_to_point_stays_on_hostmem_above_the_threshold() {
        let delegate = crate::collective::HostBackend.load(None).expect("host");
        let lib: Arc<dyn CollectiveLibrary> = Arc::new(HostmemLibrary {
            slot_bytes: SLOT_BYTES,
            dma_min: None,
            delegate: Some(delegate),
        });
        let id = lib.unique_id().expect("id");
        let reg = turbine_observability::MetricsRegistry::new();
        let metrics = CollectiveMetrics::register(&reg);
        let sent = encode(DType::F32, &values(99, 1024));
        let got: Vec<Vec<u8>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..2usize)
                .map(|r| {
                    let (lib, metrics, sent) = (Arc::clone(&lib), metrics.clone(), &sent);
                    s.spawn(move || {
                        let mem: Arc<dyn DeviceMemory> = stub_mapped_context(r as u32);
                        let comm = lib
                            .open(CollectiveInit {
                                metrics: Some(metrics),
                                route_max_bytes: Some(64),
                                ..init(r, 2, id, Duration::from_secs(20), Some(Arc::clone(&mem)))
                            })
                            .expect("open");
                        let stream = mem.compute_stream();
                        let buf = DeviceBuffer::alloc(&mem, sent.len()).expect("alloc");
                        if r == 0 {
                            buf.whole().write_bytes(sent).expect("write");
                            comm.send(&buf.whole(), 1, &stream).expect("send");
                        } else {
                            comm.recv(&mut buf.whole(), 0, &stream).expect("recv");
                        }
                        mem.synchronize().expect("sync");
                        comm.step_end().expect("healthy");
                        buf.whole().read_bytes().expect("read")
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rank"))
                .collect()
        });
        assert_eq!(got[1], sent, "the receiver holds the sender's bytes");
        let text = reg.render().expect("renders");
        for op in ["send", "recv"] {
            let line = format!(
                "turbine_collective_route_total{{op=\"{op}\",backend=\"hostmem\",reason=\"no_peer_access\"}} 1"
            );
            assert!(text.contains(&line), "{line}\n{text}");
            let delegated = format!("op=\"{op}\",backend=\"host\"");
            assert!(!text.contains(&delegated), "{delegated}\n{text}");
        }
    }

    /// While the stream is captured into a graph, a call the delegate would run is refused
    /// (`Unavailable`, reason `graph_rccl_delegate_unsupported`, counted in
    /// `turbine_collective_route_total`) without enqueuing anything or aborting the group; after
    /// the capture ends the same call routes to the delegate again. Breaks if an RCCL call can
    /// be captured, or the refusal aborts the group.
    #[test]
    fn delegate_calls_are_refused_under_graph_capture() {
        let delegate = crate::collective::HostBackend.load(None).expect("host");
        let lib: Arc<dyn CollectiveLibrary> = Arc::new(HostmemLibrary {
            slot_bytes: SLOT_BYTES,
            dma_min: None,
            delegate: Some(delegate),
        });
        let id = lib.unique_id().expect("id");
        let reg = turbine_observability::MetricsRegistry::new();
        let metrics = CollectiveMetrics::register(&reg);
        let ctx = stub_mapped_context(0);
        let mem: Arc<dyn DeviceMemory> = ctx.clone();
        let comm = lib
            .open(CollectiveInit {
                metrics: Some(metrics),
                route_max_bytes: Some(64),
                ..init(0, 1, id, Duration::from_secs(5), Some(Arc::clone(&mem)))
            })
            .expect("open");
        let stream = mem.compute_stream();
        let buf = DeviceBuffer::alloc(&mem, 4096).expect("alloc");
        ctx.graph_begin().expect("begin capture");
        let err = comm
            .all_reduce(&mut buf.whole(), DType::F32, ReduceOp::Sum, &stream)
            .expect_err("refused under capture");
        drop(ctx.graph_end());
        assert!(
            matches!(&err, CollectiveError::Unavailable { detail, .. }
                if detail.contains("graph_rccl_delegate_unsupported")),
            "{err:?}"
        );
        comm.step_end().expect("the group is not aborted");
        comm.all_reduce(&mut buf.whole(), DType::F32, ReduceOp::Sum, &stream)
            .expect("routed to the delegate once the capture ended");
        let text = reg.render().expect("renders");
        let line = "turbine_collective_route_total{op=\"all_reduce\",backend=\"hostmem\",\
                    reason=\"graph_rccl_delegate_unsupported\"} 1";
        assert!(text.contains(line), "{line}\n{text}");
    }

    /// A delegate whose init waits for a peer that never opens: the rank's open fails with
    /// `Timeout { op: "comm_init" }` within the init timeout (here the host backend's delegate
    /// init returns at once, so the bound comes from the delegate's own first-op wait: the
    /// group opens; a lone rank's routed call then times out instead of hanging).
    #[test]
    fn delegate_with_a_missing_peer_is_bounded() {
        let delegate = crate::collective::HostBackend.load(None).expect("host");
        let lib: Arc<dyn CollectiveLibrary> = Arc::new(HostmemLibrary {
            slot_bytes: SLOT_BYTES,
            dma_min: None,
            delegate: Some(delegate),
        });
        let id = lib.unique_id().expect("id");
        let mem: Arc<dyn DeviceMemory> = stub_mapped_context(0);
        let started = Instant::now();
        let comm = lib
            .open(CollectiveInit {
                route_max_bytes: Some(0),
                ..init(0, 2, id, Duration::from_millis(300), Some(Arc::clone(&mem)))
            })
            .expect("the host delegate opens without its peer");
        let buf = DeviceBuffer::alloc(&mem, 64).expect("alloc");
        let err = comm
            .all_reduce(
                &mut buf.whole(),
                DType::F32,
                ReduceOp::Sum,
                &mem.compute_stream(),
            )
            .expect_err("routed to the delegate, whose peer never comes");
        assert!(matches!(err, CollectiveError::Timeout { .. }), "{err:?}");
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "{:?}",
            started.elapsed()
        );
    }

    /// Unavailable (never a panic) without a device context, on a device whose library lacks
    /// the v2.7 group, and for an id made by another process (static rank mode).
    #[test]
    fn unavailable_cases() {
        let lib = plain();
        let id = lib.unique_id().expect("id");
        let t = Duration::from_secs(1);
        let unavailable = |r: Result<Arc<dyn Collective>, CollectiveError>, want: &str| match r {
            Err(CollectiveError::Unavailable { library, detail }) => {
                assert_eq!(library, "hostmem");
                assert!(detail.contains(want), "{detail}");
            }
            other => panic!("expected Unavailable, got {:?}", other.map(|_| ())),
        };
        unavailable(
            Arc::clone(&lib).open(init(0, 2, id, t, None)),
            "device context",
        );
        let host: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 20);
        unavailable(
            Arc::clone(&lib).open(init(0, 2, id, t, Some(host))),
            "ABI v2.7",
        );
        let mut foreign = id;
        foreign[8..12].copy_from_slice(&(std::process::id() ^ 1).to_le_bytes());
        unavailable(
            Arc::clone(&lib).open(init(0, 2, foreign, t, Some(stub_mapped_context(0)))),
            "parallel.ranks.mode local",
        );
        let err = HostmemBackend
            .load(Some(Path::new("/nonexistent/libx.so")))
            .map(|_| ())
            .expect_err("an explicit delegate library that does not exist");
        assert!(
            matches!(&err, CollectiveError::Unavailable { library, .. } if library == "/nonexistent/libx.so"),
            "{err:?}"
        );
    }
}
