//! Block copies between local tiers (P4 S-6): the copy paths and their metric labels, and the
//! transfer engine bounding in-flight bytes over a pluggable [`TransferBackend`] — copy streams
//! and events on GPUs (`turbine-server`), tier get/put on an I/O pool, virtual time in the
//! simulator.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use turbine_core::clock::Clock;
use turbine_core::types::RequestId;

use crate::identity::KvKey;
use crate::planner::PathCost;
use crate::tier::{KvTier, TierBlockMut, TierBlockRef, TierError, TierId, TierSlot};

/// One local copy direction; label of `turbine_kv_transfer_*{path}` (CONFLICT C-9: local tier
/// copies only). `L0ToL2` / `L2ToL0` exist only on unified-memory devices, where L1 is disabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransferPath {
    L0ToL1,
    L1ToL0,
    L1ToL2,
    L2ToL1,
    L0ToL2,
    L2ToL0,
}

impl TransferPath {
    pub const ALL: [TransferPath; 6] = [
        TransferPath::L0ToL1,
        TransferPath::L1ToL0,
        TransferPath::L1ToL2,
        TransferPath::L2ToL1,
        TransferPath::L0ToL2,
        TransferPath::L2ToL0,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            TransferPath::L0ToL1 => "l0_to_l1",
            TransferPath::L1ToL0 => "l1_to_l0",
            TransferPath::L1ToL2 => "l1_to_l2",
            TransferPath::L2ToL1 => "l2_to_l1",
            TransferPath::L0ToL2 => "l0_to_l2",
            TransferPath::L2ToL0 => "l2_to_l0",
        }
    }

    /// The path copying a block from `from` to `to`; `None` for a non-local or same-tier pair.
    pub fn between(from: TierId, to: TierId) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|p| p.from() == from && p.to() == to)
    }

    pub fn from(self) -> TierId {
        match self {
            TransferPath::L0ToL1 | TransferPath::L0ToL2 => TierId::L0,
            TransferPath::L1ToL0 | TransferPath::L1ToL2 => TierId::L1,
            TransferPath::L2ToL1 | TransferPath::L2ToL0 => TierId::L2,
        }
    }

    pub fn to(self) -> TierId {
        match self {
            TransferPath::L1ToL0 | TransferPath::L2ToL0 => TierId::L0,
            TransferPath::L0ToL1 | TransferPath::L2ToL1 => TierId::L1,
            TransferPath::L1ToL2 | TransferPath::L0ToL2 => TierId::L2,
        }
    }

    /// Conservative estimate used until calibration, or when it fails (P4 Failure modes):
    /// 8 GB/s for L0 ↔ L1, 1 GB/s for every other path.
    pub fn fallback(self) -> PathCost {
        match self {
            TransferPath::L0ToL1 | TransferPath::L1ToL0 => PathCost {
                latency_s: 20e-6,
                bandwidth_bps: 8e9,
            },
            _ => PathCost {
                latency_s: 100e-6,
                bandwidth_bps: 1e9,
            },
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// Why a block is copied (every copy carries a reason, TS §21 rule 7).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransferPurpose {
    Demote,
    Promote,
    Prefetch,
}

impl TransferPurpose {
    pub fn as_str(self) -> &'static str {
        match self {
            TransferPurpose::Demote => "demote",
            TransferPurpose::Promote => "promote",
            TransferPurpose::Prefetch => "prefetch",
        }
    }
}

/// The formats at both ends of a copy (P6b S-1): the source copy is decoded from `from` and the
/// destination copy encoded in `to`, both `kv_format` codec names (`l0` for an L0 block, whose
/// pages hold the L0 format). `from_bytes` / `to_bytes` are the two copies' sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransferCodec {
    pub from: &'static str,
    pub from_bytes: u64,
    pub to: &'static str,
    pub to_bytes: u64,
}

impl TransferCodec {
    /// A copy at the L0 format on both ends, `bytes` each.
    pub fn l0(bytes: u64) -> Self {
        TransferCodec {
            from: crate::tier::L0_FORMAT,
            from_bytes: bytes,
            to: crate::tier::L0_FORMAT,
            to_bytes: bytes,
        }
    }

    /// The bytes move unchanged (no decode, no encode).
    pub fn is_identity(&self) -> bool {
        self.from == self.to
    }
}

/// One block copy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransferRequest {
    pub path: TransferPath,
    pub key: KvKey,
    /// Bytes counted against `max_inflight_bytes`.
    pub bytes: u64,
    /// The request the copy serves; `None` for demotions and cache-only prefetches.
    pub owner: Option<RequestId>,
    pub purpose: TransferPurpose,
    /// Backend slot numbers (L0 `BlockId`, or `slab << 32 | slot` in L1/L2).
    pub src_slot: u64,
    pub dst_slot: u64,
    /// Formats of the source and destination copies (P6b S-1).
    pub codec: TransferCodec,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransferTicket {
    pub id: u64,
    pub req: TransferRequest,
}

/// Moves the bytes of a ticket: copy streams and events on GPUs (`turbine-server`), tier
/// get/put on an I/O pool, or virtual time in the simulator ([`SimTransferBackend`]).
pub trait TransferBackend {
    /// Starts the copy; an error completes the ticket with that error at once.
    fn start(&mut self, t: &TransferTicket) -> Result<(), TierError>;
    /// `Ok(Some(destination slot))` once the copy has completed, `Ok(None)` while it runs.
    fn poll(&mut self, t: &TransferTicket) -> Result<Option<TierSlot>, TierError>;
    /// How long the copy `poll` just completed took, when the backend measured it itself (a
    /// copy some of whose parts run elsewhere, P5 Task 30); `None`: the time from its start to
    /// the `poll` that saw it complete.
    fn took(&mut self, t: &TransferTicket) -> Option<Duration> {
        let _ = t;
        None
    }
}

#[derive(Debug)]
pub struct TransferCompletion {
    pub ticket: TransferTicket,
    /// Copy duration (start to observed completion) and destination slot.
    pub result: Result<(Duration, TierSlot), TierError>,
    /// The owner was cancelled while the copy was in flight: its bytes belong to the cache only,
    /// never to the cancelled sequence (P4 S-12).
    pub owner_cancelled: bool,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TransferError {
    #[error("transfer queue full ({0} queued copies)")]
    QueueFull(usize),
}

/// EWMA weight of one completed copy.
const ALPHA: f64 = 0.2;

/// Bounded copy scheduler (P4 S-6): a FIFO queue bounded by `max_queued`, in-flight bytes
/// bounded by `max_inflight_bytes`, per-path latency and bandwidth EWMAs, and cancellation by
/// owner. Driven synchronously by `pump` at iteration boundaries; decisions read the injected
/// clock only.
pub struct TransferEngine {
    max_inflight_bytes: u64,
    max_queued: usize,
    clock: Arc<dyn Clock>,
    next_id: u64,
    queued: VecDeque<TransferTicket>,
    /// In-flight tickets with their start time.
    inflight: Vec<(TransferTicket, Duration)>,
    inflight_bytes: u64,
    peak_inflight_bytes: u64,
    /// Owners cancelled while one of their copies is still in flight.
    cancelled: HashSet<RequestId>,
    /// Queued or in-flight copies per key.
    busy: HashMap<KvKey, u32>,
    estimates: [PathCost; 6],
}

impl TransferEngine {
    pub fn new(max_inflight_bytes: u64, max_queued: usize, clock: Arc<dyn Clock>) -> Self {
        TransferEngine {
            max_inflight_bytes,
            max_queued,
            clock,
            next_id: 1,
            queued: VecDeque::with_capacity(max_queued.min(1024)),
            inflight: Vec::new(),
            inflight_bytes: 0,
            peak_inflight_bytes: 0,
            cancelled: HashSet::new(),
            busy: HashMap::new(),
            estimates: TransferPath::ALL.map(TransferPath::fallback),
        }
    }

    /// Queues a copy; it starts on a later `pump` in FIFO order.
    pub fn submit(&mut self, r: TransferRequest) -> Result<TransferTicket, TransferError> {
        if self.queued.len() >= self.max_queued {
            return Err(TransferError::QueueFull(self.max_queued));
        }
        let ticket = TransferTicket {
            id: self.next_id,
            req: r,
        };
        self.next_id += 1;
        *self.busy.entry(ticket.req.key).or_insert(0) += 1;
        self.queued.push_back(ticket.clone());
        Ok(ticket)
    }

    /// Polls in-flight copies (completions update the path estimates), then starts queued copies
    /// while in-flight bytes stay within the bound; a copy larger than the bound starts alone.
    pub fn pump(&mut self, b: &mut dyn TransferBackend) -> Vec<TransferCompletion> {
        let now = self.clock.now_mono();
        let mut out = Vec::new();
        let mut running = Vec::with_capacity(self.inflight.len());
        for (ticket, started) in std::mem::take(&mut self.inflight) {
            let result = match b.poll(&ticket) {
                Ok(None) => {
                    running.push((ticket, started));
                    continue;
                }
                Ok(Some(slot)) => Ok((
                    b.took(&ticket)
                        .unwrap_or_else(|| now.saturating_sub(started)),
                    slot,
                )),
                Err(e) => Err(e),
            };
            self.inflight_bytes -= ticket.req.bytes;
            if let Ok((took, _)) = &result {
                self.observe(ticket.req.path, ticket.req.bytes, *took);
            }
            let owner_cancelled = ticket
                .req
                .owner
                .is_some_and(|o| self.cancelled.contains(&o));
            self.finish(ticket, result, owner_cancelled, &mut out);
        }
        self.inflight = running;
        let inflight = &self.inflight;
        self.cancelled
            .retain(|o| inflight.iter().any(|(t, _)| t.req.owner == Some(*o)));

        while let Some(front) = self.queued.front() {
            let fits = self.inflight.is_empty()
                || self.inflight_bytes + front.req.bytes <= self.max_inflight_bytes;
            if !fits {
                break;
            }
            let ticket = self.queued.pop_front().expect("front exists");
            match b.start(&ticket) {
                Ok(()) => {
                    self.inflight_bytes += ticket.req.bytes;
                    self.peak_inflight_bytes = self.peak_inflight_bytes.max(self.inflight_bytes);
                    self.inflight.push((ticket, now));
                }
                Err(e) => self.finish(ticket, Err(e), false, &mut out),
            }
        }
        out
    }

    /// Drops the queued copies of `owner` (returned) and marks its in-flight ones, whose
    /// completions then carry `owner_cancelled`.
    pub fn cancel_owner(&mut self, owner: RequestId) -> Vec<TransferTicket> {
        let (dropped, kept): (Vec<_>, Vec<_>) = self
            .queued
            .drain(..)
            .partition(|t| t.req.owner == Some(owner));
        self.queued = kept.into();
        for t in &dropped {
            self.unbusy(&t.req.key);
        }
        if self
            .inflight
            .iter()
            .any(|(t, _)| t.req.owner == Some(owner))
        {
            self.cancelled.insert(owner);
        }
        dropped
    }

    /// Replaces a path estimate (startup calibration).
    pub fn seed(&mut self, p: TransferPath, c: PathCost) {
        self.estimates[p.index()] = c;
    }

    pub fn estimate(&self, p: TransferPath) -> PathCost {
        self.estimates[p.index()]
    }

    pub fn inflight_bytes(&self) -> u64 {
        self.inflight_bytes
    }

    pub fn max_inflight_bytes(&self) -> u64 {
        self.max_inflight_bytes
    }

    pub fn peak_inflight_bytes(&self) -> u64 {
        self.peak_inflight_bytes
    }

    pub fn queued(&self) -> usize {
        self.queued.len()
    }

    /// Whether a copy of `key` is queued or in flight.
    pub fn is_busy_with(&self, key: &KvKey) -> bool {
        self.busy.contains_key(key)
    }

    pub fn is_idle(&self) -> bool {
        self.queued.is_empty() && self.inflight.is_empty()
    }

    /// Folds one completed copy into its path's EWMAs. A single copy cannot separate latency
    /// from bandwidth, so bandwidth is the effective rate (bytes over the whole duration, which
    /// errs slow) and latency is what the duration leaves beyond the current bandwidth estimate.
    fn observe(&mut self, path: TransferPath, bytes: u64, took: Duration) {
        let secs = took.as_secs_f64().max(1e-9);
        let e = &mut self.estimates[path.index()];
        let latency = (secs - bytes as f64 / e.bandwidth_bps.max(1.0)).clamp(0.0, secs);
        e.bandwidth_bps = (1.0 - ALPHA) * e.bandwidth_bps + ALPHA * (bytes as f64 / secs);
        e.latency_s = (1.0 - ALPHA) * e.latency_s + ALPHA * latency;
    }

    fn finish(
        &mut self,
        ticket: TransferTicket,
        result: Result<(Duration, TierSlot), TierError>,
        owner_cancelled: bool,
        out: &mut Vec<TransferCompletion>,
    ) {
        self.unbusy(&ticket.req.key);
        out.push(TransferCompletion {
            ticket,
            result,
            owner_cancelled,
        });
    }

    fn unbusy(&mut self, key: &KvKey) {
        if let Some(n) = self.busy.get_mut(key) {
            *n -= 1;
            if *n == 0 {
                self.busy.remove(key);
            }
        }
    }
}

/// Virtual-time backend for tests and `turbine-bench kv-sim`: a copy completes once
/// latency + bytes / bandwidth of its path has passed on the clock, then moves the block between
/// the host tiers with get/put. L0, and any host tier not given to the simulator, holds no
/// bytes: a copy out of it reads zeros and a copy into it lands in `dst_slot`.
pub struct SimTransferBackend {
    clock: Arc<dyn Clock>,
    costs: [PathCost; 6],
    l1: Option<Arc<dyn KvTier>>,
    l2: Option<Arc<dyn KvTier>>,
    scratch: Vec<u8>,
    /// False when built with `block_bytes` 0: the tiers are payload-free and copies move no
    /// bytes (a payload-free `MemTier` accounts the logical size `put_as` names).
    payload: bool,
    due: HashMap<u64, Duration>,
    fail_next: u32,
}

impl SimTransferBackend {
    pub fn new(
        clock: Arc<dyn Clock>,
        l1: Option<Arc<dyn KvTier>>,
        l2: Option<Arc<dyn KvTier>>,
        block_bytes: usize,
    ) -> Self {
        SimTransferBackend {
            clock,
            costs: TransferPath::ALL.map(TransferPath::fallback),
            l1,
            l2,
            scratch: Vec::with_capacity(block_bytes),
            payload: block_bytes > 0,
            due: HashMap::new(),
            fail_next: 0,
        }
    }

    pub fn set_cost(&mut self, p: TransferPath, c: PathCost) {
        self.costs[p.index()] = c;
    }

    /// The next `n` completions fail with a copy error.
    pub fn fail_next(&mut self, n: u32) {
        self.fail_next = n;
    }

    fn tier(&self, t: TierId) -> Option<&Arc<dyn KvTier>> {
        match t {
            TierId::L1 => self.l1.as_ref(),
            TierId::L2 => self.l2.as_ref(),
            _ => None,
        }
    }

    /// Reads the source copy (`codec.from_bytes`) and stores the destination copy
    /// (`codec.to_bytes`, labelled `codec.to`). The simulator does not transcode: when the
    /// sizes differ, the destination holds the source bytes cut or zero-padded to its size.
    fn move_block(&self, req: &TransferRequest, buf: &mut Vec<u8>) -> Result<TierSlot, TierError> {
        let (src, dst) = (self.tier(req.path.from()), self.tier(req.path.to()));
        if src.is_none() && dst.is_none() {
            return Ok(TierSlot(req.dst_slot));
        }
        let c = req.codec;
        let (from_len, to_len) = if self.payload {
            (c.from_bytes as usize, c.to_bytes as usize)
        } else {
            (0, 0)
        };
        buf.clear();
        buf.resize(from_len.max(to_len), 0);
        if let Some(src) = src {
            src.get(&req.key, TierBlockMut::Host(&mut buf[..from_len]))?;
            buf[from_len..].fill(0);
        }
        match dst {
            Some(dst) => dst.put_as(
                req.key,
                c.to,
                c.to_bytes,
                TierBlockRef::Host(&buf[..to_len]),
            ),
            None => Ok(TierSlot(req.dst_slot)),
        }
    }
}

impl TransferBackend for SimTransferBackend {
    fn start(&mut self, t: &TransferTicket) -> Result<(), TierError> {
        let secs = self.costs[t.req.path.index()].block_seconds(t.req.bytes);
        let due = self.clock.now_mono() + Duration::from_secs_f64(secs.max(0.0));
        self.due.insert(t.id, due);
        Ok(())
    }

    fn poll(&mut self, t: &TransferTicket) -> Result<Option<TierSlot>, TierError> {
        let due = *self.due.get(&t.id).ok_or(TierError::Missing)?;
        if self.clock.now_mono() < due {
            return Ok(None);
        }
        self.due.remove(&t.id);
        if self.fail_next > 0 {
            self.fail_next -= 1;
            return Err(TierError::Io("injected copy error".into()));
        }
        let mut buf = std::mem::take(&mut self.scratch);
        let result = self.move_block(&t.req, &mut buf);
        self.scratch = buf;
        result.map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tier::MemTier;
    use turbine_core::clock::FakeClock;

    #[test]
    fn paths_round_trip_their_endpoints() {
        for p in TransferPath::ALL {
            assert_eq!(TransferPath::between(p.from(), p.to()), Some(p));
            assert_eq!(
                p.as_str(),
                format!("{}_to_{}", p.from().as_str(), p.to().as_str())
            );
        }
        assert_eq!(TransferPath::between(TierId::L0, TierId::L0), None);
        assert_eq!(TransferPath::between(TierId::L3, TierId::L0), None);
    }

    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;

    fn fake() -> (Arc<FakeClock>, Arc<dyn Clock>) {
        let fake = Arc::new(FakeClock::new(Duration::ZERO));
        let clock: Arc<dyn Clock> = fake.clone();
        (fake, clock)
    }

    fn request(
        path: TransferPath,
        key: u8,
        bytes: u64,
        owner: Option<RequestId>,
    ) -> TransferRequest {
        TransferRequest {
            path,
            key: KvKey([key; 16]),
            bytes,
            owner,
            purpose: TransferPurpose::Promote,
            src_slot: u64::from(key),
            dst_slot: u64::from(key),
            // `bytes` only paces the copy; the tiers of these tests hold 4 KiB blocks.
            codec: TransferCodec::l0(4096),
        }
    }

    #[test]
    fn inflight_bounded() {
        let (fake, clock) = fake();
        let mut engine = TransferEngine::new(GIB, 4096, clock.clone());
        let mut backend = SimTransferBackend::new(clock, None, None, 0);
        let total = 10 * GIB / (4 * MIB);
        assert_eq!(total, 2560);
        for i in 0..total {
            let path = if i % 2 == 0 {
                TransferPath::L1ToL0
            } else {
                TransferPath::L2ToL0
            };
            engine
                .submit(request(path, (i % 251) as u8, 4 * MIB, None))
                .unwrap();
        }
        let mut completed = 0;
        let mut rounds = 0;
        while !engine.is_idle() {
            for c in engine.pump(&mut backend) {
                let (took, slot) = c.result.expect("every simulated copy succeeds");
                assert!(took > Duration::ZERO);
                assert_eq!(slot, TierSlot(c.ticket.req.dst_slot), "L0 slot is dst_slot");
                assert!(!c.owner_cancelled);
                completed += 1;
            }
            assert!(
                engine.inflight_bytes() <= GIB,
                "in-flight {} exceeds 1 GiB",
                engine.inflight_bytes()
            );
            fake.advance(Duration::from_micros(200));
            rounds += 1;
            assert!(rounds < 1_000_000, "transfers never drain");
        }
        assert_eq!(completed, total);
        assert_eq!(engine.inflight_bytes(), 0);
        assert!(engine.peak_inflight_bytes() <= GIB);
        assert!(
            engine.peak_inflight_bytes() > GIB / 2,
            "the bound is used, not starved: peak {}",
            engine.peak_inflight_bytes()
        );
        let l1 = engine.estimate(TransferPath::L1ToL0);
        assert!(l1.bandwidth_bps > 0.0 && l1.latency_s >= 0.0);
        assert_ne!(
            l1,
            TransferPath::L1ToL0.fallback(),
            "completions update the estimate"
        );
    }

    #[test]
    fn queue_bound_cancellation_and_failures() {
        let (fake, clock) = fake();
        let owner = RequestId::new_v4();
        let other = RequestId::new_v4();
        let mut engine = TransferEngine::new(8 * MIB, 3, clock.clone());
        let l1: Arc<dyn KvTier> = Arc::new(MemTier::new(TierId::L1, 16 * MIB, clock.clone()));
        let l2: Arc<dyn KvTier> = Arc::new(MemTier::new(TierId::L2, 16 * MIB, clock.clone()));
        let mut backend = SimTransferBackend::new(clock, Some(l1.clone()), Some(l2.clone()), 4096);
        backend.set_cost(
            TransferPath::L1ToL2,
            PathCost {
                latency_s: 1e-3,
                bandwidth_bps: 1e9,
            },
        );
        let block: Vec<u8> = (0..4096u32).map(|i| i as u8).collect();
        l1.put(KvKey([1; 16]), TierBlockRef::Host(&block)).unwrap();

        // Bounded queue.
        engine
            .submit(request(TransferPath::L1ToL2, 1, 4096, Some(owner)))
            .unwrap();
        engine
            .submit(request(TransferPath::L1ToL0, 1, 4 * MIB, Some(owner)))
            .unwrap();
        engine
            .submit(request(TransferPath::L1ToL0, 3, 4 * MIB, Some(other)))
            .unwrap();
        assert_eq!(
            engine.submit(request(TransferPath::L1ToL0, 4, 1, None)),
            Err(TransferError::QueueFull(3))
        );
        assert!(engine.is_busy_with(&KvKey([1; 16])), "queued keys are busy");
        assert!(!engine.is_busy_with(&KvKey([2; 16])));

        // First pump starts 4 KiB + 4 MiB; the third copy would exceed 8 MiB.
        assert!(engine.pump(&mut backend).is_empty());
        assert_eq!(engine.inflight_bytes(), 4 * MIB + 4096);
        // Cancelling the owner drops nothing queued (both of its copies are in flight) but marks them.
        assert!(engine.cancel_owner(owner).is_empty());
        let queued = engine.cancel_owner(other);
        assert_eq!(queued.len(), 1, "the queued copy of `other` is dropped");
        assert!(!engine.is_busy_with(&KvKey([3; 16])));

        fake.advance(Duration::from_millis(10));
        let done = engine.pump(&mut backend);
        assert_eq!(done.len(), 2);
        assert!(done.iter().all(|c| c.owner_cancelled && c.result.is_ok()));
        let mut out = vec![0u8; 4096];
        l2.get(&KvKey([1; 16]), TierBlockMut::Host(&mut out))
            .unwrap();
        assert_eq!(
            out, block,
            "an in-flight copy of a cancelled owner lands in the cache"
        );
        assert!(engine.is_idle() && !engine.is_busy_with(&KvKey([1; 16])));

        // A failed copy completes with its error, frees its bytes and leaves the estimate alone.
        let before = engine.estimate(TransferPath::L1ToL0);
        backend.fail_next(1);
        engine
            .submit(request(TransferPath::L1ToL0, 5, MIB, None))
            .unwrap();
        engine.pump(&mut backend);
        fake.advance(Duration::from_millis(10));
        let done = engine.pump(&mut backend);
        assert!(matches!(done[0].result, Err(TierError::Io(_))));
        assert!(!done[0].owner_cancelled);
        assert_eq!(engine.estimate(TransferPath::L1ToL0), before);
        assert!(engine.is_idle() && engine.inflight_bytes() == 0);

        // A copy larger than the whole bound still runs when nothing else is in flight.
        engine
            .submit(request(TransferPath::L1ToL0, 6, 64 * MIB, None))
            .unwrap();
        engine.pump(&mut backend);
        assert_eq!(engine.inflight_bytes(), 64 * MIB);

        // Seeding replaces an estimate (startup calibration).
        let seeded = PathCost {
            latency_s: 5e-6,
            bandwidth_bps: 25e9,
        };
        engine.seed(TransferPath::L0ToL1, seeded);
        assert_eq!(engine.estimate(TransferPath::L0ToL1), seeded);
    }
}
