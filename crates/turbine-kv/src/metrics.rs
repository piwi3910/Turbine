//! KV metric families (P2 `turbine_kv_blocks{tier,state}`, tier label `l0` per CONFLICT C-4;
//! P4 S-14). Every label value comes from a closed enum rendered with `as_str()`. Counters are
//! registered without the `_total` suffix; prometheus-client appends it.

use std::sync::atomic::AtomicU64;

use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{Histogram, exponential_buckets};
use turbine_observability::{GaugeFamilyShare, GaugeShare, MetricsRegistry};

use crate::directory::PrefixMatch;
use crate::planner::PlanReason;
use crate::pool::BlockPool;
use crate::tier::TierId;
use crate::transfer::TransferPath;

/// The Phase 2 tier label of the GPU pool (CONFLICT C-4); equals `TierId::L0.as_str()`.
pub const TIER_L0: &str = "l0";

type Labels1 = [(&'static str, &'static str); 1];
type Labels2 = [(&'static str, &'static str); 2];
type Labels4 = [(&'static str, &'static str); 4];

/// Why a copy left a tier (or its rung). `turbine_kv_evictions_total{reason}` uses the first
/// five; `turbine_kv_drops_total{reason}` (a block gone from every tier) adds `below_min_value`
/// and `no_room`. The compression ladder (P6b S-6) adds `compressed` (a copy rewritten one rung
/// down) and `ladder_floor` (evicted at `kv.ladder.max_format`), [`EvictReason::LADDER`]:
/// pre-registered on `turbine_kv_evictions_total` (both) and `turbine_kv_drops_total`
/// (`ladder_floor`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EvictReason {
    Capacity,
    Pressure,
    SessionExpired,
    Checksum,
    TierDegraded,
    BelowMinValue,
    NoRoom,
    Compressed,
    LadderFloor,
    /// An L0-only partial entry reclaimed (P5p S-5; it demotes never, so a reclaim is its
    /// only way out).
    PartialL0Only,
    /// A copy evicted from an L1 slab being drained for a rung's slot size (P5p S-18).
    SlabDrain,
}

impl EvictReason {
    /// Label values of `turbine_kv_evictions_total{reason}`.
    pub const EVICTION: [EvictReason; 7] = [
        EvictReason::Capacity,
        EvictReason::Pressure,
        EvictReason::SessionExpired,
        EvictReason::Checksum,
        EvictReason::TierDegraded,
        EvictReason::PartialL0Only,
        EvictReason::SlabDrain,
    ];
    /// Label values of `turbine_kv_drops_total{reason}`.
    pub const ALL: [EvictReason; 9] = [
        EvictReason::Capacity,
        EvictReason::Pressure,
        EvictReason::SessionExpired,
        EvictReason::Checksum,
        EvictReason::TierDegraded,
        EvictReason::BelowMinValue,
        EvictReason::NoRoom,
        EvictReason::PartialL0Only,
        EvictReason::SlabDrain,
    ];
    /// The compression ladder's reasons (P6b S-6).
    pub const LADDER: [EvictReason; 2] = [EvictReason::Compressed, EvictReason::LadderFloor];

    pub fn as_str(self) -> &'static str {
        match self {
            EvictReason::Capacity => "capacity",
            EvictReason::Pressure => "pressure",
            EvictReason::SessionExpired => "session_expired",
            EvictReason::Checksum => "checksum",
            EvictReason::TierDegraded => "tier_degraded",
            EvictReason::BelowMinValue => "below_min_value",
            EvictReason::NoRoom => "no_room",
            EvictReason::Compressed => "compressed",
            EvictReason::LadderFloor => "ladder_floor",
            EvictReason::PartialL0Only => "partial_l0_only",
            EvictReason::SlabDrain => "slab_drain",
        }
    }
}

/// How a lookup's token-granular comparison of the prompt beyond its matched full blocks
/// ended (P5p S-4, S-5): label of `turbine_kv_partial_reuse_total{outcome}`, one per lookup
/// that reached the comparison.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PartialOutcome {
    /// The comparison matched a child and the plan reuses (some of) its tokens through a
    /// tail copy.
    Reused,
    /// No child of the last matched key shares at least one token.
    NoCandidate,
    /// A child would match but holds no L0 copy (its copies are all in L1/L2); a full block
    /// below L0 is reused whole, never partially.
    NotResident,
    /// The comparison matched, but the plan's cutoff or the `MIN_RECOMPUTE_TOKENS` cap left
    /// the matched tokens unused.
    PlannerCutoff,
    /// The matching children hold lossy rung-class pages or are of lossy lineage: their KV
    /// is never a bit-exact source, so no partial match is returned.
    LossySource,
    /// `kv.partial_prefix_reuse` is `false`.
    Disabled,
}

impl PartialOutcome {
    pub const ALL: [PartialOutcome; 6] = [
        PartialOutcome::Reused,
        PartialOutcome::NoCandidate,
        PartialOutcome::NotResident,
        PartialOutcome::PlannerCutoff,
        PartialOutcome::LossySource,
        PartialOutcome::Disabled,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            PartialOutcome::Reused => "reused",
            PartialOutcome::NoCandidate => "no_candidate",
            PartialOutcome::NotResident => "not_resident",
            PartialOutcome::PlannerCutoff => "planner_cutoff",
            PartialOutcome::LossySource => "lossy_source",
            PartialOutcome::Disabled => "disabled",
        }
    }
}

/// Why the compression ladder acted (`turbine_kv_ladder_actions_total{reason}`, P6b S-6): a
/// copy rewritten one rung down by the ladder's sweep of a tier under pressure
/// (`fill_high_water`) or instead of being dropped from the lowest tier (`would_drop`), a
/// demotion stored at the tier's ladder rung instead of its configured format
/// (`new_demotion`), the tier's rung for new demotions stepping back up after the dwell
/// (`rung_step_up`), a copy evicted at the lossiest rung (`floor_evict`), a tier's rewrites
/// backed off after one found no room for its new format (`no_room_backoff`, once per back-off;
/// user decision "6b Task 16: ladder proof results — four open points", 3 A), and a demotion
/// stored at the tier's own format because no slot of the rung's size was free
/// (`rung_no_slot`; user decision "6b: ladder-on regression — demotions into a tier whose rung
/// changed", A).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LadderReason {
    FillHighWater,
    WouldDrop,
    NewDemotion,
    RungStepUp,
    FloorEvict,
    NoRoomBackoff,
    RungNoSlot,
    /// A recent-window exit: an out-of-window BF16 block recompressed into the L0 base
    /// format (S-5, `p6b-t17`).
    RecentWindow,
}

impl LadderReason {
    pub const ALL: [LadderReason; 8] = [
        LadderReason::FillHighWater,
        LadderReason::WouldDrop,
        LadderReason::NewDemotion,
        LadderReason::RungStepUp,
        LadderReason::FloorEvict,
        LadderReason::NoRoomBackoff,
        LadderReason::RungNoSlot,
        LadderReason::RecentWindow,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            LadderReason::FillHighWater => "fill_high_water",
            LadderReason::WouldDrop => "would_drop",
            LadderReason::NewDemotion => "new_demotion",
            LadderReason::RungStepUp => "rung_step_up",
            LadderReason::FloorEvict => "floor_evict",
            LadderReason::NoRoomBackoff => "no_room_backoff",
            LadderReason::RungNoSlot => "rung_no_slot",
            LadderReason::RecentWindow => "recent_window",
        }
    }
}

/// The `to` label of a `floor_evict` ladder action: the copy left the tier.
pub const LADDER_EVICT: &str = "evict";

/// What the compression ladder spent time on (`turbine_kv_ladder_action_seconds_total{tier,action}`,
/// P5p S-14): the host time of a tier's ladder sweep (`sweep`), the time a rewrite waited in the
/// transfer queue before it started (`queue`), and a rewrite's run from its start to its
/// completion as the transfer backend saw it (`rewrite`: an L1/L2 rewrite's copies, transcode
/// and store; an L0 rewrite completes once its device work is enqueued).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LadderTime {
    Sweep,
    Queue,
    Rewrite,
}

impl LadderTime {
    pub const ALL: [LadderTime; 3] = [LadderTime::Sweep, LadderTime::Queue, LadderTime::Rewrite];

    pub fn as_str(self) -> &'static str {
        match self {
            LadderTime::Sweep => "sweep",
            LadderTime::Queue => "queue",
            LadderTime::Rewrite => "rewrite",
        }
    }
}

/// What a slab drain did (`turbine_kv_slab_reformat_total{tier,action}`, P5p S-18): a copy of a
/// draining slab evicted (`drain_evict`: another tier held the block, or nowhere could take
/// it), a copy re-stored one tier down (`drain_restore`: the tier held the block's only copy),
/// and a drained slab reformatted to the rung's slot size (`reformat`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SlabAction {
    DrainEvict,
    DrainRestore,
    Reformat,
}

impl SlabAction {
    pub const ALL: [SlabAction; 3] = [
        SlabAction::DrainEvict,
        SlabAction::DrainRestore,
        SlabAction::Reformat,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            SlabAction::DrainEvict => "drain_evict",
            SlabAction::DrainRestore => "drain_restore",
            SlabAction::Reformat => "reformat",
        }
    }
}

/// Outcome of one prefetched block (`turbine_kv_prefetch_total{outcome}`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrefetchOutcome {
    Used,
    Wasted,
    Cancelled,
    Rejected,
}

impl PrefetchOutcome {
    pub const ALL: [PrefetchOutcome; 4] = [
        PrefetchOutcome::Used,
        PrefetchOutcome::Wasted,
        PrefetchOutcome::Cancelled,
        PrefetchOutcome::Rejected,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            PrefetchOutcome::Used => "used",
            PrefetchOutcome::Wasted => "wasted",
            PrefetchOutcome::Cancelled => "cancelled",
            PrefetchOutcome::Rejected => "rejected",
        }
    }
}

/// Shared handles to every KV family; cloning shares the underlying values.
#[derive(Clone)]
pub struct KvMetrics {
    /// Gauges are this pool's share: data-parallel replicas add up ([`KvMetrics::another_replica`]).
    pub blocks: GaugeFamilyShare<Labels2>,
    pub bytes: GaugeFamilyShare<Labels2>,
    pub lookups: Family<Labels1, Counter>,
    pub prefix_cached_tokens: Counter,
    /// P6b S-3: prompt tokens served from lossy cached blocks.
    pub lossy_cached_tokens: Counter,
    /// P6b S-3: lookups a lossy-reuse opt-out cut at a block only a lossy copy held.
    pub lossy_denied: Counter,
    pub prompt_tokens: Counter,
    pub promotions: Family<Labels2, Counter>,
    pub demotions: Family<Labels2, Counter>,
    pub evictions: Family<Labels2, Counter>,
    pub drops: Family<Labels1, Counter>,
    pub recompute_tokens: Family<Labels1, Counter>,
    pub plans: Family<Labels1, Counter>,
    pub transfer_seconds: Family<Labels1, Histogram>,
    pub transfer_bytes: Family<Labels1, Counter>,
    pub transfer_bandwidth: Family<Labels1, Gauge<f64, AtomicU64>>,
    pub prefetch: Family<Labels1, Counter>,
    pub sessions: GaugeShare,
    pub tier_degraded: Family<Labels1, Gauge>,
    pub storage_queue_depth: GaugeShare,
    pub storage_latency: Histogram,
    /// `turbine_kv_ladder_rung{tier}`: rung index (registration order of the `kv_format`
    /// codecs: 0 `l0`, 1 `fp8_e4m3`, 2 `tq4`, 3 `tq2`) new demotions into the tier take.
    pub ladder_rung: Family<Labels1, Gauge>,
    /// `turbine_kv_ladder_actions_total{tier,from,to,reason}` (P6b S-6).
    pub ladder_actions: Family<Labels4, Counter>,
    /// `turbine_kv_ladder_action_seconds_total{tier,action}` (P5p S-14): seconds the ladder
    /// spent per tier on each [`LadderTime`].
    pub ladder_action_seconds: Family<Labels2, Counter<f64, AtomicU64>>,
    /// `turbine_kv_slab_reformat_total{tier,action}` (P5p S-18): [`SlabAction`]s per tier.
    pub slab_reformat: Family<Labels2, Counter>,
    /// `turbine_kv_copy_ahead_total{to}` (P6b S-8, decision "6b: shared-prefix eval never
    /// demotes the prefix to L1", A): copies of a shared parent into a lower tier made while
    /// its L0 copy stays.
    pub copy_ahead: Family<Labels1, Counter>,
    /// `turbine_kv_queued_prefix_detached_blocks_total` (decision "6b: queued-prefix demotion —
    /// granularity and scope"): blocks requests in the admission queue released for pressure
    /// reclaim at YELLOW/ORANGE, counting only those no other request held.
    pub queued_prefix_detached_blocks: Counter,
    /// `turbine_kv_partial_reuse_total{outcome}` (P5p S-4/S-5): how each lookup that reached
    /// the token-granular comparison ended.
    pub partial_reuse: Family<Labels1, Counter>,
    /// `turbine_kv_partial_reused_tokens_total` (P5p S-5): prompt tokens reused through tail
    /// copies.
    pub partial_reused_tokens: Counter,
    /// `turbine_kv_partial_entries` (P5p S-5): partial entries currently in the directory.
    pub partial_entries: Gauge,
}

/// 10 µs .. ~42 s in ×4 steps: covers pinned copies of one block through slow NVMe reads.
fn latency_histogram() -> Histogram {
    Histogram::new(exponential_buckets(1e-5, 4.0, 12))
}

impl KvMetrics {
    /// Families not attached to any registry (unit tests and the offline simulator).
    pub fn unregistered() -> Self {
        KvMetrics {
            blocks: GaugeFamilyShare::new(Family::default()),
            bytes: GaugeFamilyShare::new(Family::default()),
            lookups: Family::default(),
            prefix_cached_tokens: Counter::default(),
            lossy_cached_tokens: Counter::default(),
            lossy_denied: Counter::default(),
            prompt_tokens: Counter::default(),
            promotions: Family::default(),
            demotions: Family::default(),
            evictions: Family::default(),
            drops: Family::default(),
            recompute_tokens: Family::default(),
            plans: Family::default(),
            transfer_seconds: Family::new_with_constructor(latency_histogram),
            transfer_bytes: Family::default(),
            transfer_bandwidth: Family::default(),
            prefetch: Family::default(),
            sessions: GaugeShare::default(),
            tier_degraded: Family::default(),
            storage_queue_depth: GaugeShare::default(),
            storage_latency: latency_histogram(),
            ladder_rung: Family::default(),
            ladder_actions: Family::default(),
            ladder_action_seconds: Family::default(),
            slab_reformat: Family::default(),
            copy_ahead: Family::default(),
            queued_prefix_detached_blocks: Counter::default(),
            partial_reuse: Family::default(),
            partial_reused_tokens: Counter::default(),
            partial_entries: Gauge::default(),
        }
    }

    /// Registers every family on `reg` and touches every documented label value.
    pub fn register(reg: &MetricsRegistry) -> Self {
        let m = Self::unregistered();
        reg.register(
            "turbine_kv_blocks",
            "KV blocks per tier and state",
            m.blocks.family().clone(),
        );
        reg.register(
            "turbine_kv_bytes",
            "KV bytes per tier (capacity, used)",
            m.bytes.family().clone(),
        );
        reg.register(
            "turbine_kv_lookups",
            "Prompt blocks looked up, by the fastest tier holding them or miss",
            m.lookups.clone(),
        );
        reg.register(
            "turbine_kv_prefix_cached_tokens",
            "Prompt tokens served from cached prefixes",
            m.prefix_cached_tokens.clone(),
        );
        reg.register(
            "turbine_kv_lossy_cached_tokens",
            "Prompt tokens served from lossy cached KV blocks",
            m.lossy_cached_tokens.clone(),
        );
        reg.register(
            "turbine_kv_lossy_denied",
            "Prefix lookups cut at a lossy block by a lossy-reuse opt-out",
            m.lossy_denied.clone(),
        );
        reg.register(
            "turbine_kv_prompt_tokens",
            "Prompt tokens looked up in the KV directory",
            m.prompt_tokens.clone(),
        );
        reg.register(
            "turbine_kv_promotions",
            "KV block copies into a faster tier",
            m.promotions.clone(),
        );
        reg.register(
            "turbine_kv_demotions",
            "KV block copies into a slower tier",
            m.demotions.clone(),
        );
        reg.register(
            "turbine_kv_evictions",
            "KV block copies removed from a tier",
            m.evictions.clone(),
        );
        reg.register(
            "turbine_kv_drops",
            "KV blocks removed from every tier",
            m.drops.clone(),
        );
        reg.register(
            "turbine_kv_recompute_tokens",
            "Prompt tokens recomputed, by plan reason",
            m.recompute_tokens.clone(),
        );
        reg.register(
            "turbine_kv_plans",
            "Recompute-vs-retrieve plans, by reason",
            m.plans.clone(),
        );
        reg.register(
            "turbine_kv_transfer_seconds",
            "Duration of one local KV tier copy",
            m.transfer_seconds.clone(),
        );
        reg.register(
            "turbine_kv_transfer_bytes",
            "Bytes copied between local KV tiers",
            m.transfer_bytes.clone(),
        );
        reg.register(
            "turbine_kv_transfer_bandwidth_bytes_per_second",
            "EWMA bandwidth of each local KV copy path",
            m.transfer_bandwidth.clone(),
        );
        reg.register(
            "turbine_kv_prefetch",
            "Prefetched KV blocks, by outcome",
            m.prefetch.clone(),
        );
        reg.register(
            "turbine_kv_sessions",
            "Sessions in the KV session table",
            m.sessions.gauge().clone(),
        );
        reg.register(
            "turbine_kv_tier_degraded",
            "1 while a KV tier is degraded",
            m.tier_degraded.clone(),
        );
        reg.register(
            "turbine_storage_queue_depth",
            "In-flight L2 NVMe I/O operations",
            m.storage_queue_depth.gauge().clone(),
        );
        reg.register(
            "turbine_storage_latency_seconds",
            "Latency of one L2 NVMe I/O operation",
            m.storage_latency.clone(),
        );
        reg.register(
            "turbine_kv_ladder_rung",
            "Compression ladder rung new demotions into a tier take (0 l0, 1 fp8_e4m3, 2 tq4, 3 tq2)",
            m.ladder_rung.clone(),
        );
        reg.register(
            "turbine_kv_ladder_actions",
            "Compression ladder actions, by tier, formats and reason",
            m.ladder_actions.clone(),
        );
        reg.register(
            "turbine_kv_ladder_action_seconds",
            "Seconds the compression ladder spent, by tier and action (sweep, queue, rewrite)",
            m.ladder_action_seconds.clone(),
        );
        reg.register(
            "turbine_kv_slab_reformat",
            "Slab drains for a ladder rung's slot size: copies evicted, re-stored, slabs reformatted",
            m.slab_reformat.clone(),
        );
        reg.register(
            "turbine_kv_copy_ahead",
            "Shared-prefix KV blocks copied into a slower tier while their L0 copy stays",
            m.copy_ahead.clone(),
        );
        reg.register(
            "turbine_kv_queued_prefix_detached_blocks",
            "KV blocks queued requests released for pressure reclaim (reason queued_prefix)",
            m.queued_prefix_detached_blocks.clone(),
        );
        reg.register(
            "turbine_kv_partial_reuse",
            "Token-granular prefix comparisons by how they ended",
            m.partial_reuse.clone(),
        );
        reg.register(
            "turbine_kv_partial_reused_tokens",
            "Prompt tokens reused through copy-on-write tail copies",
            m.partial_reused_tokens.clone(),
        );
        reg.register(
            "turbine_kv_partial_entries",
            "Partial entries currently in the KV directory",
            m.partial_entries.clone(),
        );
        m.init_labels();
        m
    }

    /// The same families for another data-parallel replica's pool: counters and histograms are
    /// shared, the occupancy gauges get a fresh share (the gauges report the sum over replicas).
    pub fn another_replica(&self) -> KvMetrics {
        KvMetrics {
            blocks: self.blocks.another(),
            bytes: self.bytes.another(),
            sessions: self.sessions.another(),
            storage_queue_depth: self.storage_queue_depth.another(),
            ..self.clone()
        }
    }

    /// Touches every documented label combination so each family renders from startup.
    pub fn init_labels(&self) {
        for t in TierId::LOCAL {
            for s in ["used", "free"] {
                self.blocks.touch(&[("tier", t.as_str()), ("state", s)]);
            }
            for k in ["capacity", "used"] {
                self.bytes.touch(&[("tier", t.as_str()), ("kind", k)]);
            }
            let _ = self.lookups.get_or_create(&[("result", t.as_str())]);
            let _ = self.tier_degraded.get_or_create(&[("tier", t.as_str())]);
            for r in EvictReason::EVICTION.into_iter().chain(EvictReason::LADDER) {
                let _ = self
                    .evictions
                    .get_or_create(&[("tier", t.as_str()), ("reason", r.as_str())]);
            }
            let _ = self.ladder_rung.get_or_create(&[("tier", t.as_str())]);
            for a in LadderTime::ALL {
                let _ = self
                    .ladder_action_seconds
                    .get_or_create(&[("tier", t.as_str()), ("action", a.as_str())]);
            }
        }
        let _ = self
            .drops
            .get_or_create(&[("reason", EvictReason::LadderFloor.as_str())]);
        // The ladder's one-rung moves of L1 and L2 (P6b S-6): each adjacent pair of the
        // `kv_format` codecs down (sweep, would-drop) and up (dwell), and the floor eviction.
        let rungs: Vec<&'static str> = crate::codec::registry().iter().map(|c| c.name()).collect();
        for t in [TierId::L1, TierId::L2] {
            for pair in rungs.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                for (from, to, r) in [
                    (a, b, LadderReason::FillHighWater),
                    (a, b, LadderReason::WouldDrop),
                    (b, a, LadderReason::RungStepUp),
                ] {
                    let _ = self.ladder_actions.get_or_create(&[
                        ("tier", t.as_str()),
                        ("from", from),
                        ("to", to),
                        ("reason", r.as_str()),
                    ]);
                }
            }
            if let Some(last) = rungs.last() {
                let _ = self.ladder_actions.get_or_create(&[
                    ("tier", t.as_str()),
                    ("from", last),
                    ("to", LADDER_EVICT),
                    ("reason", LadderReason::FloorEvict.as_str()),
                ]);
            }
        }
        for t in [TierId::L1, TierId::L2] {
            let _ = self.copy_ahead.get_or_create(&[("to", t.as_str())]);
            for a in SlabAction::ALL {
                let _ = self
                    .slab_reformat
                    .get_or_create(&[("tier", t.as_str()), ("action", a.as_str())]);
            }
        }
        let _ = self.lookups.get_or_create(&[("result", "miss")]);
        for p in TransferPath::ALL {
            let (from, to) = (p.from().as_str(), p.to().as_str());
            let family = if p.to() > p.from() {
                &self.demotions
            } else {
                &self.promotions
            };
            let _ = family.get_or_create(&[("from", from), ("to", to)]);
            let _ = self.transfer_seconds.get_or_create(&[("path", p.as_str())]);
            let _ = self.transfer_bytes.get_or_create(&[("path", p.as_str())]);
            let _ = self
                .transfer_bandwidth
                .get_or_create(&[("path", p.as_str())]);
        }
        for r in EvictReason::ALL {
            let _ = self.drops.get_or_create(&[("reason", r.as_str())]);
        }
        for r in PlanReason::ALL {
            let _ = self.plans.get_or_create(&[("reason", r.as_str())]);
            let _ = self
                .recompute_tokens
                .get_or_create(&[("reason", r.as_str())]);
        }
        for o in PartialOutcome::ALL {
            let _ = self.partial_reuse.get_or_create(&[("outcome", o.as_str())]);
        }
        for o in PrefetchOutcome::ALL {
            let _ = self.prefetch.get_or_create(&[("outcome", o.as_str())]);
        }
    }

    fn lookup_label(result: Option<TierId>) -> &'static str {
        result.map_or("miss", TierId::as_str)
    }

    /// One lookup of one block: the fastest tier holding it, or `None` for a miss.
    pub fn lookup(&self, result: Option<TierId>) {
        self.lookups
            .get_or_create(&[("result", Self::lookup_label(result))])
            .inc();
    }

    /// Counts every full prompt block of a lookup: matched blocks by tier, the rest as misses.
    pub fn record_lookup(&self, m: &PrefixMatch) {
        for b in &m.blocks {
            self.lookup(Some(b.tier));
        }
        for _ in m.blocks.len()..m.keys.len() {
            self.lookup(None);
        }
    }

    /// One lookup that reached the token-granular comparison ended with `o`; when it reused,
    /// `tokens` prompt tokens went through a tail copy (P5p S-5).
    pub fn partial_outcome(&self, o: PartialOutcome, tokens: u32) {
        self.partial_reuse
            .get_or_create(&[("outcome", o.as_str())])
            .inc();
        if tokens > 0 {
            self.partial_reused_tokens.inc_by(u64::from(tokens));
        }
    }

    /// Current `turbine_kv_lookups_total{result}` value.
    pub fn lookups_value(&self, result: Option<TierId>) -> u64 {
        self.lookups
            .get_or_create(&[("result", Self::lookup_label(result))])
            .get()
    }

    pub fn eviction(&self, tier: TierId, reason: EvictReason) {
        self.evictions
            .get_or_create(&[("tier", tier.as_str()), ("reason", reason.as_str())])
            .inc();
    }

    /// Current `turbine_kv_evictions_total{tier,reason}` value.
    pub fn evictions_value(&self, tier: TierId, reason: EvictReason) -> u64 {
        self.evictions
            .get_or_create(&[("tier", tier.as_str()), ("reason", reason.as_str())])
            .get()
    }

    pub fn drop_block(&self, reason: EvictReason) {
        self.drops
            .get_or_create(&[("reason", reason.as_str())])
            .inc();
    }

    pub fn demotion(&self, from: TierId, to: TierId) {
        self.demotions
            .get_or_create(&[("from", from.as_str()), ("to", to.as_str())])
            .inc();
    }

    /// One copy-ahead copy into `to` landed (the L0 copy stays).
    pub fn copy_ahead(&self, to: TierId) {
        self.copy_ahead.get_or_create(&[("to", to.as_str())]).inc();
    }

    /// Current `turbine_kv_copy_ahead_total{to}` value.
    pub fn copy_ahead_value(&self, to: TierId) -> u64 {
        self.copy_ahead.get_or_create(&[("to", to.as_str())]).get()
    }

    pub fn promotion(&self, from: TierId, to: TierId) {
        self.promotions
            .get_or_create(&[("from", from.as_str()), ("to", to.as_str())])
            .inc();
    }

    /// One plan and the prompt tokens it recomputes.
    pub fn plan(&self, reason: PlanReason, recompute_tokens: u32) {
        self.plans
            .get_or_create(&[("reason", reason.as_str())])
            .inc();
        self.recompute_tokens
            .get_or_create(&[("reason", reason.as_str())])
            .inc_by(u64::from(recompute_tokens));
    }

    pub fn prefetch_outcome(&self, o: PrefetchOutcome) {
        self.prefetch
            .get_or_create(&[("outcome", o.as_str())])
            .inc();
    }

    /// One completed copy on `path`, with the path's bandwidth EWMA after it.
    pub fn transfer(&self, path: TransferPath, bytes: u64, seconds: f64, bandwidth_ewma: f64) {
        let label = [("path", path.as_str())];
        self.transfer_seconds.get_or_create(&label).observe(seconds);
        self.transfer_bytes.get_or_create(&label).inc_by(bytes);
        self.transfer_bandwidth
            .get_or_create(&label)
            .set(bandwidth_ewma);
    }

    /// The rung new demotions into `tier` take (a registered codec name).
    pub fn set_ladder_rung(&self, tier: TierId, rung: &str) {
        let index = crate::codec::rung_index(rung).unwrap_or(0);
        self.ladder_rung
            .get_or_create(&[("tier", tier.as_str())])
            .set(i64::try_from(index).unwrap_or(0));
    }

    /// One ladder action on a copy of `tier` (or its rung) from codec `from` to codec `to`
    /// ([`LADDER_EVICT`] for a floor eviction).
    pub fn ladder_action(&self, tier: TierId, from: &str, to: &str, reason: LadderReason) {
        // Label values must be `'static`: codec names come from the registry.
        let name = |f: &str| -> &'static str {
            crate::codec::registry().get(f).map_or(
                if f == LADDER_EVICT {
                    LADDER_EVICT
                } else {
                    "unknown"
                },
                |c| c.name(),
            )
        };
        self.ladder_actions
            .get_or_create(&[
                ("tier", tier.as_str()),
                ("from", name(from)),
                ("to", name(to)),
                ("reason", reason.as_str()),
            ])
            .inc();
    }

    /// `n` slab-drain `action`s on `tier` (P5p S-18).
    pub fn slab_action(&self, tier: TierId, action: SlabAction, n: u64) {
        self.slab_reformat
            .get_or_create(&[("tier", tier.as_str()), ("action", action.as_str())])
            .inc_by(n);
    }

    /// Current `turbine_kv_slab_reformat_total` value of one series.
    pub fn slab_action_value(&self, tier: TierId, action: SlabAction) -> u64 {
        self.slab_reformat
            .get_or_create(&[("tier", tier.as_str()), ("action", action.as_str())])
            .get()
    }

    /// `seconds` the ladder spent on `action` for `tier` (P5p S-14).
    pub fn ladder_seconds(&self, tier: TierId, action: LadderTime, seconds: f64) {
        if seconds > 0.0 {
            self.ladder_action_seconds
                .get_or_create(&[("tier", tier.as_str()), ("action", action.as_str())])
                .inc_by(seconds);
        }
    }

    /// Current `turbine_kv_ladder_action_seconds_total` value of one series.
    pub fn ladder_seconds_value(&self, tier: TierId, action: LadderTime) -> f64 {
        self.ladder_action_seconds
            .get_or_create(&[("tier", tier.as_str()), ("action", action.as_str())])
            .get()
    }

    /// Current `turbine_kv_ladder_actions_total` value of one series.
    pub fn ladder_actions_value(
        &self,
        tier: TierId,
        from: &'static str,
        to: &'static str,
        reason: LadderReason,
    ) -> u64 {
        self.ladder_actions
            .get_or_create(&[
                ("tier", tier.as_str()),
                ("from", from),
                ("to", to),
                ("reason", reason.as_str()),
            ])
            .get()
    }

    pub fn set_degraded(&self, tier: TierId, degraded: bool) {
        self.tier_degraded
            .get_or_create(&[("tier", tier.as_str())])
            .set(i64::from(degraded));
    }

    /// Publish the L0 pool's current occupancy (Phase 2; the engine calls it every iteration).
    pub fn record(&self, pool: &BlockPool) {
        self.blocks.set(
            &[("tier", TIER_L0), ("state", "used")],
            i64::from(pool.used_blocks()),
        );
        self.blocks.set(
            &[("tier", TIER_L0), ("state", "free")],
            i64::from(pool.free_blocks()),
        );
    }

    /// Capacity and usage of one tier (a disabled tier reports zeros).
    pub fn tier_usage(
        &self,
        tier: TierId,
        capacity_bytes: u64,
        used_bytes: u64,
        used_blocks: u64,
        free_blocks: u64,
    ) {
        let t = tier.as_str();
        let gauge = |v: u64| i64::try_from(v).unwrap_or(i64::MAX);
        self.bytes
            .set(&[("tier", t), ("kind", "capacity")], gauge(capacity_bytes));
        self.bytes
            .set(&[("tier", t), ("kind", "used")], gauge(used_bytes));
        self.blocks
            .set(&[("tier", t), ("state", "used")], gauge(used_blocks));
        self.blocks
            .set(&[("tier", t), ("state", "free")], gauge(free_blocks));
    }
}

/// The startup INFO line (`event="kv_pool"`) naming the block size and count.
pub fn log_pool_startup(pool: &BlockPool) {
    let layout = pool.layout();
    tracing::info!(
        event = "kv_pool",
        tier = TIER_L0,
        dtype = layout.dtype.as_str(),
        block_tokens = layout.block_tokens,
        block_bytes = layout.block_bytes(),
        num_blocks = pool.total_blocks(),
        "KV block pool allocated"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every P4 family renders at startup with only its documented label values.
    #[test]
    fn families_render_with_closed_labels() {
        let reg = MetricsRegistry::new();
        let m = KvMetrics::register(&reg);
        m.tier_usage(TierId::L1, 1 << 30, 1 << 20, 3, 5);
        m.plan(PlanReason::RecomputeCheaper, 65);
        m.transfer(TransferPath::L1ToL0, 4096, 0.001, 4.0e6);
        let text = reg.render().unwrap();
        for family in [
            "turbine_kv_blocks{",
            "turbine_kv_bytes{",
            "turbine_kv_lookups_total{",
            "turbine_kv_prefix_cached_tokens_total",
            "turbine_kv_lossy_cached_tokens_total 0",
            "turbine_kv_lossy_denied_total 0",
            "turbine_kv_prompt_tokens_total",
            "turbine_kv_promotions_total{",
            "turbine_kv_demotions_total{",
            "turbine_kv_evictions_total{",
            "turbine_kv_drops_total{",
            "turbine_kv_recompute_tokens_total{",
            "turbine_kv_plans_total{",
            "turbine_kv_transfer_seconds_bucket{",
            "turbine_kv_transfer_bytes_total{",
            "turbine_kv_transfer_bandwidth_bytes_per_second{",
            "turbine_kv_prefetch_total{",
            "turbine_kv_sessions ",
            "turbine_kv_tier_degraded{",
            "turbine_storage_queue_depth ",
            "turbine_storage_latency_seconds_bucket{",
        ] {
            assert!(text.contains(family), "{family} missing from:\n{text}");
        }
        assert!(text.contains(r#"turbine_kv_bytes{tier="l1",kind="capacity"} 1073741824"#));
        assert!(
            text.contains(r#"turbine_kv_recompute_tokens_total{reason="recompute_cheaper"} 65"#)
        );
        assert!(text.contains(r#"turbine_kv_demotions_total{from="l0",to="l2"} 0"#));
        assert!(text.contains(r#"turbine_kv_promotions_total{from="l2",to="l0"} 0"#));
        assert!(!text.contains(r#"tier="l3""#), "no L3 before Phase 6");
        assert_eq!(TierId::L0.as_str(), TIER_L0);
        assert!(!text.contains(r#"turbine_kv_evictions_total{tier="l0",reason="no_room"}"#));
    }

    /// The ladder's reason codes render as the spec's label values and stay out of the Phase 4
    /// constant sets (`init_labels` pre-registers them beside those, [`ladder_families`]).
    #[test]
    fn ladder_reasons() {
        let labels: Vec<&str> = EvictReason::LADDER.iter().map(|r| r.as_str()).collect();
        assert_eq!(labels, ["compressed", "ladder_floor"]);
        for r in EvictReason::LADDER {
            assert!(!EvictReason::EVICTION.contains(&r) && !EvictReason::ALL.contains(&r));
        }
    }

    /// P6b S-6: the ladder families render from startup (the rung per tier at 0, the eviction
    /// and drop reasons pre-registered) and count with the spec's label values only.
    #[test]
    fn ladder_families() {
        let reg = MetricsRegistry::new();
        let m = KvMetrics::register(&reg);
        m.set_ladder_rung(TierId::L2, "tq4");
        m.ladder_action(TierId::L2, "l0", "fp8_e4m3", LadderReason::FillHighWater);
        m.ladder_action(TierId::L2, "tq2", LADDER_EVICT, LadderReason::FloorEvict);
        let text = reg.render().unwrap();
        for line in [
            r#"turbine_kv_ladder_rung{tier="l1"} 0"#,
            r#"turbine_kv_ladder_rung{tier="l2"} 2"#,
            r#"turbine_kv_evictions_total{tier="l1",reason="compressed"} 0"#,
            r#"turbine_kv_evictions_total{tier="l2",reason="ladder_floor"} 0"#,
            r#"turbine_kv_drops_total{reason="ladder_floor"} 0"#,
            r#"turbine_kv_ladder_actions_total{tier="l2",from="l0",to="fp8_e4m3",reason="fill_high_water"} 1"#,
            r#"turbine_kv_ladder_actions_total{tier="l2",from="tq2",to="evict",reason="floor_evict"} 1"#,
        ] {
            assert!(text.contains(line), "{line} missing from:\n{text}");
        }
        let labels: Vec<&str> = LadderReason::ALL.iter().map(|r| r.as_str()).collect();
        assert_eq!(
            labels,
            [
                "fill_high_water",
                "would_drop",
                "new_demotion",
                "rung_step_up",
                "floor_evict",
                "no_room_backoff",
                "rung_no_slot",
                "recent_window"
            ]
        );
    }
}
