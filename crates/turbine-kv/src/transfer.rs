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
    /// A compression-ladder rewrite (P6b S-6): the copy in `path.from()` is re-encoded from
    /// `codec.from` to `codec.to` and stored back in the same tier under the same key. `path`
    /// is that tier's path to L0, the route a device transcode stages the block through; the
    /// rewrite is not a copy between tiers, so it feeds neither the path's estimate nor its
    /// transfer metrics.
    Compress,
}

impl TransferPurpose {
    pub fn as_str(self) -> &'static str {
        match self {
            TransferPurpose::Demote => "demote",
            TransferPurpose::Promote => "promote",
            TransferPurpose::Prefetch => "prefetch",
            TransferPurpose::Compress => "compress",
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
    /// How long the copy `poll` just completed took, as far as the backend knows it (a copy
    /// some of whose parts run elsewhere, P5 Task 30; a copy that ends on a GPU copy stream,
    /// whose completion is seen only at a poll). `None`: the engine bounds it itself, from the
    /// last poll that saw it running to the poll that saw it complete.
    fn took(&mut self, t: &TransferTicket) -> Option<CopyTime> {
        let _ = t;
        None
    }
}

/// How long a completed copy took ([`TransferBackend::took`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyTime {
    /// Measured from its start to its completion.
    Exact(Duration),
    /// Completed somewhere in `at_least..=at_most` after its start: the copy was seen running
    /// `at_least` in and done only at a poll `at_most` in. A poll-bounded copy says the path is
    /// not slower than `at_most` and not faster than `at_least`, nothing in between, so the
    /// estimate learns only when it falls outside (decision "6b Task 6", point 3: copy-stream
    /// copies polled once per iteration had priced a 1.4 ms block at ~25–100 ms).
    Within {
        at_least: Duration,
        at_most: Duration,
    },
}

impl CopyTime {
    /// The duration reported for the copy (metrics): the measured one, or the poll bound.
    pub fn reported(self) -> Duration {
        match self {
            CopyTime::Exact(d) => d,
            CopyTime::Within { at_most, .. } => at_most,
        }
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
    /// In-flight tickets with their start time and the last poll that saw them running.
    inflight: Vec<(TransferTicket, Duration, Option<Duration>)>,
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
        for (ticket, started, seen_running) in std::mem::take(&mut self.inflight) {
            let last_running = seen_running.unwrap_or(started);
            let result = match b.poll(&ticket) {
                Ok(None) => {
                    running.push((ticket, started, Some(now)));
                    continue;
                }
                Ok(Some(slot)) => Ok((
                    b.took(&ticket).unwrap_or(CopyTime::Within {
                        at_least: last_running.saturating_sub(started),
                        at_most: now.saturating_sub(started),
                    }),
                    slot,
                )),
                Err(e) => Err(e),
            };
            self.inflight_bytes -= ticket.req.bytes;
            if let Ok((took, _)) = &result
                && ticket.req.purpose != TransferPurpose::Compress
            {
                // Estimates are rates per encoded byte, the bytes the copy moved (the smaller
                // end's, `req.bytes`): the planner and the eviction score price a copy at its
                // encoded size (P6b S-3, decision A), so the compression saving counts once.
                let sample = self.observe(ticket.req.path, ticket.req.bytes, *took);
                let e = self.estimates[ticket.req.path.index()];
                tracing::debug!(
                    event = "kv_copy_timed",
                    path = ticket.req.path.as_str(),
                    purpose = ticket.req.purpose.as_str(),
                    bytes = ticket.req.bytes,
                    took_s = took.reported().as_secs_f64(),
                    exact = matches!(took, CopyTime::Exact(_)),
                    sample_s = sample.as_secs_f64(),
                    est_latency_s = e.latency_s,
                    est_bandwidth_bps = e.bandwidth_bps,
                );
            }
            let result = result.map(|(took, slot)| (took.reported(), slot));
            let owner_cancelled = ticket
                .req
                .owner
                .is_some_and(|o| self.cancelled.contains(&o));
            self.finish(ticket, result, owner_cancelled, &mut out);
        }
        self.inflight = running;
        let inflight = &self.inflight;
        self.cancelled
            .retain(|o| inflight.iter().any(|(t, _, _)| t.req.owner == Some(*o)));

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
                    self.inflight.push((ticket, now, None));
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
            .any(|(t, _, _)| t.req.owner == Some(owner))
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

    /// Folds one completed copy of `bytes` encoded bytes into its path's EWMAs, keeping the
    /// estimate `latency + bytes / bandwidth` linear in the moved bytes (P6b S-3: one path
    /// carries L0-format and compressed copies of very different sizes). A single copy cannot
    /// separate latency from bandwidth, so the bandwidth sample is the rate over the duration
    /// the current latency estimate leaves, taken only when that is at least half the copy
    /// (a latency-bound copy says little about the rate), and latency is what the duration
    /// leaves beyond the current bandwidth estimate.
    ///
    /// A poll-bounded copy ([`CopyTime::Within`]) is folded in as the current estimate clamped
    /// to its bounds: inside them it changes nothing, outside it moves the estimate to the
    /// nearer bound. Returns the duration folded in.
    fn observe(&mut self, path: TransferPath, bytes: u64, took: CopyTime) -> Duration {
        let took = match took {
            CopyTime::Exact(d) => d,
            CopyTime::Within { at_least, at_most } => {
                let predicted =
                    Duration::from_secs_f64(self.estimates[path.index()].block_seconds(bytes));
                predicted.min(at_most).max(at_least)
            }
        };
        let secs = took.as_secs_f64().max(1e-9);
        let e = &mut self.estimates[path.index()];
        let moving = secs - e.latency_s.clamp(0.0, secs);
        if moving >= secs / 2.0 {
            e.bandwidth_bps = (1.0 - ALPHA) * e.bandwidth_bps + ALPHA * (bytes as f64 / moving);
        }
        let latency = (secs - bytes as f64 / e.bandwidth_bps.max(1.0)).clamp(0.0, secs);
        e.latency_s = (1.0 - ALPHA) * e.latency_s + ALPHA * latency;
        took
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
/// the host tiers with get/put, and reports that modelled time as its duration (`took`), not
/// the time until the poll that sees it done. L0, and any host tier not given to the simulator, holds no
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
    /// Per ticket: when the copy completes and how long it takes.
    due: HashMap<u64, (Duration, Duration)>,
    /// Durations of copies `poll` just completed, until `took` reads them.
    took: HashMap<u64, Duration>,
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
            took: HashMap::new(),
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
        let src = self.tier(req.path.from());
        // A ladder rewrite stores the re-encoded copy back in its own tier.
        let dst = if req.purpose == TransferPurpose::Compress {
            src
        } else {
            self.tier(req.path.to())
        };
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
        let took = Duration::from_secs_f64(secs.max(0.0));
        self.due.insert(t.id, (self.clock.now_mono() + took, took));
        Ok(())
    }

    fn poll(&mut self, t: &TransferTicket) -> Result<Option<TierSlot>, TierError> {
        let (due, took) = *self.due.get(&t.id).ok_or(TierError::Missing)?;
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
        if result.is_ok() {
            self.took.insert(t.id, took);
        }
        result.map(Some)
    }

    fn took(&mut self, t: &TransferTicket) -> Option<CopyTime> {
        self.took.remove(&t.id).map(CopyTime::Exact)
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

    /// P6b S-3 (decision "6b Task 4: planner copy bytes", A): the planner prices a copy at the
    /// block's encoded bytes (`PlanInputs.copy_bytes`), so estimates are rates per encoded
    /// (moved) byte. Breaks if completions of compressed copies are observed at the decoded
    /// size, which would count the compression saving twice.
    #[test]
    fn estimates_are_per_encoded_byte() {
        let (fake, clock) = fake();
        let mut engine = TransferEngine::new(GIB, 16, clock.clone());
        let mut backend = SimTransferBackend::new(clock, None, None, 0);
        let cost = PathCost {
            latency_s: 0.0,
            bandwidth_bps: 1e9,
        };
        backend.set_cost(TransferPath::L1ToL0, cost);
        // A tq4-sized copy (1 MB moved) decoded into a 4 MB L0 block.
        let encoded = 1_000_000;
        for i in 0..64u8 {
            let mut req = request(TransferPath::L1ToL0, i, encoded, None);
            req.codec = TransferCodec {
                from: "tq4",
                from_bytes: encoded,
                to: crate::tier::L0_FORMAT,
                to_bytes: 4 * encoded,
            };
            engine.submit(req).unwrap();
            engine.pump(&mut backend);
            fake.advance(Duration::from_secs_f64(cost.block_seconds(encoded)));
            assert_eq!(engine.pump(&mut backend).len(), 1);
        }
        let est = engine.estimate(TransferPath::L1ToL0);
        let (want, got) = (cost.block_seconds(encoded), est.block_seconds(encoded));
        assert!(
            (got - want).abs() < want * 0.05,
            "the estimate prices the encoded bytes at the moved rate: {got} s vs {want} s ({est:?})"
        );
    }

    /// Copies of mixed encoded sizes (L0-format and compressed blocks on one path) keep the
    /// estimate linear in the bytes: latency is not folded into the per-byte rate, so small,
    /// latency-bound copies do not make a full-size block look slower than it is. Breaks if the
    /// bandwidth sample includes the latency (a full block then prices ~50 % slow here).
    #[test]
    fn mixed_copy_sizes_keep_latency_out_of_the_rate() {
        let (fake, clock) = fake();
        let mut engine = TransferEngine::new(GIB, 16, clock.clone());
        let mut backend = SimTransferBackend::new(clock, None, None, 0);
        let path = TransferPath::L2ToL0;
        let cost = path.fallback();
        assert_eq!(
            engine.estimate(path),
            cost,
            "estimates start at the fallback"
        );
        let (full, small) = (1_835_008u64, 131_072u64);
        for i in 0..64u8 {
            let bytes = if i % 4 == 0 { full } else { small };
            engine.submit(request(path, i, bytes, None)).unwrap();
            engine.pump(&mut backend);
            fake.advance(Duration::from_secs_f64(cost.block_seconds(bytes)));
            assert_eq!(engine.pump(&mut backend).len(), 1);
        }
        let est = engine.estimate(path);
        for bytes in [full, small] {
            let (want, got) = (cost.block_seconds(bytes), est.block_seconds(bytes));
            assert!(
                (got - want).abs() < want * 0.05,
                "{bytes} B: {got} s vs {want} s ({est:?})"
            );
        }
    }

    /// The simulator's copies are timed by the simulator, not by the poll that sees them done:
    /// polled at 1 ms iteration boundaries, a compressed L2 → L0 copy (0.39 ms) looked like 1 ms
    /// and an fp8 one (1.02 ms) like 2 ms, which dragged the path's per-byte rate down until a
    /// full block priced above recomputing it (kv_sim `ladder_under_pinned_pressure`). Breaks if
    /// `SimTransferBackend` stops reporting the copy time it models.
    #[test]
    fn simulated_copies_are_timed_by_the_simulator() {
        let (fake, clock) = fake();
        let mut engine = TransferEngine::new(GIB, 16, clock.clone());
        let mut backend = SimTransferBackend::new(clock, None, None, 0);
        let path = TransferPath::L2ToL0;
        let cost = path.fallback();
        let sizes = [1_835_008u64, 917_728, 286_720];
        for i in 0..96u8 {
            let bytes = sizes[i as usize % sizes.len()];
            engine.submit(request(path, i, bytes, None)).unwrap();
            engine.pump(&mut backend);
            let mut done = 0;
            while done == 0 {
                fake.advance(Duration::from_millis(1));
                done = engine.pump(&mut backend).len();
            }
        }
        let est = engine.estimate(path);
        for bytes in sizes {
            let (want, got) = (cost.block_seconds(bytes), est.block_seconds(bytes));
            assert!(
                (got - want).abs() < want * 0.05,
                "{bytes} B: {got} s vs {want} s ({est:?})"
            );
        }
    }

    /// A backend that cannot time its own copies (a copy that ends on a GPU copy stream: the
    /// ABI has no event timestamps), so the engine sees each one done only at the next poll.
    struct PollTimed(SimTransferBackend);

    impl TransferBackend for PollTimed {
        fn start(&mut self, t: &TransferTicket) -> Result<(), TierError> {
            self.0.start(t)
        }
        fn poll(&mut self, t: &TransferTicket) -> Result<Option<TierSlot>, TierError> {
            let r = self.0.poll(t);
            self.0.took(t);
            r
        }
    }

    /// The server's stressed multi-turn run (decision "6b Task 6", point 3): L1 → L0 copies of
    /// 14.7 MB take 1.4 ms (10.45 GB/s calibrated) but end on the copy stream, which is polled
    /// once per iteration (25–200 ms at c32), so each was observed at ~25 ms; the latency
    /// estimate climbed to 63 ms and every L1 block priced above recomputing its 128 tokens. A
    /// copy seen done at a poll only bounds its duration from above: it says the estimate is
    /// not too fast, nothing more. Breaks if a poll-bounded completion is folded in as the copy's
    /// duration.
    #[test]
    fn poll_bounded_copies_do_not_drag_the_estimate() {
        use crate::planner::{PlanInputs, PlanReason, plan_prefix};
        use turbine_core::types::PressureState;

        let (fake, clock) = fake();
        let mut engine = TransferEngine::new(GIB, 16, clock.clone());
        let path = TransferPath::L1ToL0;
        let calibrated = PathCost {
            latency_s: 100e-6,
            bandwidth_bps: 10.45e9,
        };
        engine.seed(path, calibrated);
        let mut sim = SimTransferBackend::new(clock, None, None, 0);
        sim.set_cost(path, calibrated);
        let mut backend = PollTimed(sim);
        let bytes = 14_680_064u64;
        let iteration = Duration::from_millis(25);
        for i in 0..64u8 {
            engine.submit(request(path, i, bytes, None)).unwrap();
            engine.pump(&mut backend);
            fake.advance(iteration);
            assert_eq!(engine.pump(&mut backend).len(), 1, "done by the next poll");
        }
        let est = engine.estimate(path);
        let (want, got) = (calibrated.block_seconds(bytes), est.block_seconds(bytes));
        assert!(
            (got - want).abs() < want * 0.05,
            "poll-bounded copies keep the calibrated {want} s per block: {got} s ({est:?})"
        );
        // The planner then retrieves an L1 block rather than recomputing its 128 tokens at the
        // run's 9,000 tok/s.
        let matched = [TierId::L0, TierId::L1];
        let plan = plan_prefix(&PlanInputs {
            matched: &matched,
            prompt_tokens: 2 * 128 + 2,
            block_tokens: 128,
            block_bytes: bytes,
            prefill_tps: 9_000.0,
            l1_to_l0: Some(est),
            l2_to_l0: None,
            l0_state: PressureState::Green,
            l1_degraded: false,
            l2_degraded: false,
            copy_bytes: &[],
            lossy_penalty: &[],
            allow_lossy: true,
        });
        assert_eq!(
            (plan.cutoff_blocks(), plan.reason),
            (2, PlanReason::RetrieveCheaper)
        );

        // A copy still running at a poll did take at least that long: copies that stay in flight
        // for three iterations make the path slower.
        let mut slow = SimTransferBackend::new(fake_clock(&fake), None, None, 0);
        slow.set_cost(
            path,
            PathCost {
                latency_s: 0.06,
                bandwidth_bps: 10.45e9,
            },
        );
        let mut backend = PollTimed(slow);
        for i in 0..32u8 {
            engine.submit(request(path, i, bytes, None)).unwrap();
            engine.pump(&mut backend);
            let mut done = 0;
            while done == 0 {
                fake.advance(iteration);
                done = engine.pump(&mut backend).len();
            }
        }
        let got = engine.estimate(path).block_seconds(bytes);
        assert!(
            got > 0.045,
            "copies seen in flight at 50 ms price at least that: {got} s"
        );
    }

    fn fake_clock(fake: &Arc<FakeClock>) -> Arc<dyn Clock> {
        fake.clone()
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
