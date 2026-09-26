//! Decode graphs (P2c S-10): a decode-only iteration's device work — embedding through the LM
//! head and, when used, `logits_reduce` — captured once per decode shape into a graph through
//! the kernel ABI v2.1 graph functions and replayed on later iterations of that shape, one
//! launch instead of a few hundred.
//!
//! A graph records the device pointers its ops were captured with, so everything it reads lives
//! in buffers the executor allocated once — activations, the batch metadata (token ids,
//! positions, `q_indptr`, `kv_lens`, block tables), the logits and the reduction's inputs — and
//! each step's values are uploaded into them before the launch. The launch shape must not
//! change between steps of one graph: the key ([`GraphKey`]) holds the sequence count, the
//! reduction's candidate count and the block-table width, which graph mode rounds up to a power
//! of two ([`table_width`]) with the paged decode's `max_kv_len` set to the tokens that width
//! covers, so a sequence growing into a new block reuses the graph until the width doubles. The
//! KV pool is the caller's: the graphs are dropped when a forward names another pool.
//!
//! Capture happens on the second iteration seen with a key, so the first runs eagerly (and a
//! GEMM shape is never tuned under capture). The cache holds at most
//! min(_scheduler.max_running_requests_, 64) graphs, least recently used evicted. Any capture
//! failure — an op error, a runtime call the capture refuses, a library without graph functions —
//! turns graphs off for the executor's lifetime after one WARN (`decode_graphs_unavailable`),
//! and the iteration runs eagerly; no request fails.
use std::sync::Arc;

use turbine_kernels::{GraphHandle, KernelError, ShimContext};
use turbine_tensor::KvPoolView;

use super::batch::{BatchLimits, HostBatch, Packed};
use crate::ModelError;

/// Stream capture and replay on one device context (the kernel ABI v2.1 graph functions).
pub trait GraphBackend: Send + Sync {
    type Graph: Send;
    /// Starts capturing the compute stream: until [`GraphBackend::end`] only op calls may be
    /// issued.
    fn begin(&self) -> Result<(), KernelError>;
    /// Stops the capture and instantiates it; after an error the context is usable again.
    fn end(&self) -> Result<Self::Graph, KernelError>;
    /// Enqueues a captured graph on the compute stream.
    fn launch(&self, g: &Self::Graph) -> Result<(), KernelError>;
}

impl GraphBackend for ShimContext {
    type Graph = GraphHandle;
    fn begin(&self) -> Result<(), KernelError> {
        self.graph_begin()
    }
    fn end(&self) -> Result<GraphHandle, KernelError> {
        self.graph_end()
    }
    fn launch(&self, g: &GraphHandle) -> Result<(), KernelError> {
        self.graph_launch(g)
    }
}

/// The most graphs one cache holds (spec S-10).
pub const MAX_GRAPHS: usize = 64;

/// The cache bound for a scheduler of `max_running_requests`: min(that, [`MAX_GRAPHS`]), at
/// least 1.
pub fn capacity_for(max_running_requests: u32) -> usize {
    (max_running_requests as usize).clamp(1, MAX_GRAPHS)
}

/// Block-table columns graph mode packs for a batch whose longest sequence needs `blocks`
/// blocks: the next power of two, at most `max_blocks` (the executor's full table width).
pub fn table_width(blocks: u32, max_blocks: u32) -> u32 {
    blocks.max(1).next_power_of_two().min(max_blocks.max(1))
}

/// The graph key of the batch `pack` just wrote into `host` as `p`, when it can run as a decode
/// graph: every sequence decodes one token, `reduce_top_n` says the rows are all reduced or
/// all full ([`super::logits::LogitsHead::graph_top_n`]) and `feeds` (the step's
/// [`super::TokenFeed`] runs, P2c overlap scheduling) are none or one run over every token (its
/// word is part of the key; other feed shapes run eagerly). The block table is then widened to
/// [`table_width`] columns and `max_kv_len` to the tokens they cover (at most the executor's
/// `max_positions`), so every step of the key has the same launch shape. `None` leaves `p`
/// as packed.
pub(super) fn decode_key(
    p: &mut Packed,
    host: &mut HostBatch,
    limits: &BatchLimits,
    reduce_top_n: u8,
    feeds: &[(usize, usize, usize)],
) -> Option<GraphKey> {
    if !p.is_decode() {
        return None;
    }
    let feed_word = match feeds {
        [] => None,
        [(0, word, len)] if *len == p.total_q => Some(*word as u64),
        _ => return None,
    };
    let width = table_width(p.max_blocks_per_seq, limits.max_blocks_per_seq());
    let covered = u64::from(width) * u64::from(limits.layout.block_tokens);
    let max_kv_len = covered.min(u64::from(limits.max_positions)) as u32;
    host.widen(p, width, max_kv_len);
    Some(GraphKey {
        seqs: p.num_seqs as u32,
        reduce_top_n,
        table_width: width,
        feed_word,
    })
}

/// What fixes a decode graph's launch shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GraphKey {
    /// Sequences (one token each).
    pub seqs: u32,
    /// Candidates per row of the device reduction; 0 when no row is reduced (every row is
    /// reduced otherwise: a batch mixing both runs eagerly).
    pub reduce_top_n: u8,
    /// Block-table columns ([`table_width`]).
    pub table_width: u32,
    /// The word of the previous step's logits buffer the step's tokens are fed from on the
    /// device, when they are (P2c overlap scheduling): the feed's embedding reads it.
    pub feed_word: Option<u64>,
}

/// What an iteration does with its device work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphStep {
    /// Launch every op (graphs off, a mixed iteration, or the first time a key is seen).
    Eager,
    /// Capture the ops into a graph for the key, then launch it.
    Capture,
    /// Launch the key's captured graph.
    Replay,
}

/// Graph outcomes so far (`turbine_decode_graph_total{outcome}`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphCounters {
    pub captured: u64,
    pub replayed: u64,
    pub evicted: u64,
    pub capture_failed: u64,
}

impl GraphCounters {
    /// The counts since `earlier` (a previous reading of the same counters).
    pub fn since(&self, earlier: &GraphCounters) -> GraphCounters {
        GraphCounters {
            captured: self.captured.saturating_sub(earlier.captured),
            replayed: self.replayed.saturating_sub(earlier.replayed),
            evicted: self.evicted.saturating_sub(earlier.evicted),
            capture_failed: self.capture_failed.saturating_sub(earlier.capture_failed),
        }
    }
}

struct Entry<G> {
    key: GraphKey,
    graph: G,
    last_used: u64,
}

/// The bounded graph cache: which step an iteration takes, the captured graphs by key (least
/// recently used evicted at `capacity`) and the outcome counters.
pub struct GraphCache<G> {
    capacity: usize,
    entries: Vec<Entry<G>>,
    /// Keys seen once and not captured yet, oldest first; at most `4 · capacity`.
    seen: Vec<GraphKey>,
    clock: u64,
    disabled: bool,
    counters: GraphCounters,
}

impl<G> GraphCache<G> {
    /// A cache of at most `capacity` graphs (clamped to `1..=`[`MAX_GRAPHS`]).
    pub fn new(capacity: usize) -> GraphCache<G> {
        let capacity = capacity.clamp(1, MAX_GRAPHS);
        GraphCache {
            capacity,
            entries: Vec::with_capacity(capacity),
            seen: Vec::new(),
            clock: 0,
            disabled: false,
            counters: GraphCounters::default(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Captured graphs held.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The step of an iteration with launch shape `key`: `Eager` for a mixed iteration, while
    /// disabled and the first time `key` is seen; `Capture` the second time; `Replay` once a
    /// graph for it is held (counted as replayed).
    pub fn plan(&mut self, key: GraphKey, decode_only: bool) -> GraphStep {
        if self.disabled || !decode_only {
            return GraphStep::Eager;
        }
        self.clock += 1;
        if let Some(e) = self.entries.iter_mut().find(|e| e.key == key) {
            e.last_used = self.clock;
            self.counters.replayed += 1;
            return GraphStep::Replay;
        }
        if let Some(i) = self.seen.iter().position(|k| *k == key) {
            self.seen.remove(i);
            return GraphStep::Capture;
        }
        if self.seen.len() >= 4 * self.capacity {
            self.seen.remove(0);
        }
        self.seen.push(key);
        GraphStep::Eager
    }

    /// Holds `graph` for `key` (counted as captured), evicting the least recently used graph
    /// when the cache is full.
    pub fn insert(&mut self, key: GraphKey, graph: G) {
        self.clock += 1;
        if let Some(e) = self.entries.iter_mut().find(|e| e.key == key) {
            e.graph = graph;
            e.last_used = self.clock;
            self.counters.captured += 1;
            return;
        }
        if self.entries.len() >= self.capacity
            && let Some(lru) = (0..self.entries.len()).min_by_key(|&i| self.entries[i].last_used)
        {
            self.entries.swap_remove(lru);
            self.counters.evicted += 1;
        }
        self.entries.push(Entry {
            key,
            graph,
            last_used: self.clock,
        });
        self.counters.captured += 1;
    }

    /// The graph held for `key`.
    pub fn graph(&self, key: GraphKey) -> Option<&G> {
        self.entries.iter().find(|e| e.key == key).map(|e| &e.graph)
    }

    /// Counts a failed capture and turns graphs off for good ([`GraphCache::disable`]).
    pub fn capture_failed(&mut self) {
        self.counters.capture_failed += 1;
        self.disable();
    }

    /// Every later iteration runs eagerly; the held graphs are dropped (not counted as evicted).
    pub fn disable(&mut self) {
        self.disabled = true;
        self.entries.clear();
        self.seen.clear();
    }

    pub fn is_disabled(&self) -> bool {
        self.disabled
    }

    /// Drops every graph (counted as evicted) and forgets the keys seen, e.g. when the buffers
    /// they were captured with change.
    pub fn clear(&mut self) {
        self.counters.evicted += self.entries.len() as u64;
        self.entries.clear();
        self.seen.clear();
    }

    pub fn counters(&self) -> GraphCounters {
        self.counters
    }
}

/// Identity of a KV pool as a graph records it: its storage address and extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolId {
    addr: u64,
    bytes: usize,
    num_blocks: u32,
    layer_stride_bytes: u64,
}

impl PoolId {
    pub fn of(kv: &KvPoolView<'_>) -> PoolId {
        PoolId {
            addr: kv.storage.ptr().addr(),
            bytes: kv.storage.len(),
            num_blocks: kv.num_blocks,
            layer_stride_bytes: kv.layer_stride_bytes,
        }
    }
}

/// An executor's decode graphs: the backend that captures and launches them and the cache
/// ([`GraphCache`]). Given to an executor with [`super::ModelExecutor::set_decode_graphs`].
pub struct DecodeGraphs<G = GraphHandle> {
    backend: Arc<dyn GraphBackend<Graph = G>>,
    cache: GraphCache<G>,
    /// The pool the held graphs were captured on.
    pool: Option<PoolId>,
}

impl<G> std::fmt::Debug for DecodeGraphs<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodeGraphs")
            .field("capacity", &self.cache.capacity())
            .field("graphs", &self.cache.len())
            .field("disabled", &self.cache.is_disabled())
            .field("counters", &self.cache.counters())
            .finish()
    }
}

impl<G: Send> DecodeGraphs<G> {
    /// Graphs on `backend`, at most `capacity` of them ([`capacity_for`]).
    pub fn new(backend: Arc<dyn GraphBackend<Graph = G>>, capacity: usize) -> DecodeGraphs<G> {
        DecodeGraphs {
            backend,
            cache: GraphCache::new(capacity),
            pool: None,
        }
    }

    pub fn counters(&self) -> GraphCounters {
        self.cache.counters()
    }

    /// False once a capture failed: every iteration runs eagerly.
    pub fn is_enabled(&self) -> bool {
        !self.cache.is_disabled()
    }

    /// Runs an iteration's device work `enqueue` (every op after the metadata upload, before the
    /// logits copy): eagerly when `key` is `None` (not a graph-mode iteration) or the cache says
    /// so, else through the key's graph — captured now, or replayed. A failed capture turns
    /// graphs off (one WARN `decode_graphs_unavailable`) and runs `enqueue` eagerly; a replay's
    /// launch error is returned. `pool` is the forward's KV pool: graphs captured on another are
    /// dropped first.
    pub fn run(
        &mut self,
        key: Option<GraphKey>,
        pool: PoolId,
        enqueue: impl Fn() -> Result<(), ModelError>,
    ) -> Result<(), ModelError> {
        let Some(key) = key.filter(|_| self.is_enabled()) else {
            return enqueue();
        };
        if self.pool != Some(pool) {
            self.cache.clear();
            self.pool = Some(pool);
        }
        match self.cache.plan(key, true) {
            GraphStep::Eager => enqueue(),
            GraphStep::Replay => self.launch(key),
            GraphStep::Capture => {
                if let Err(e) = self.backend.begin() {
                    self.failed(key, &e);
                    return enqueue();
                }
                let captured = enqueue();
                // The capture always ends, even after an op error, so the stream is usable.
                let ended = self.backend.end();
                match (captured, ended) {
                    (Ok(()), Ok(graph)) => {
                        self.cache.insert(key, graph);
                        self.launch(key)
                    }
                    (Err(e), _) => {
                        self.failed(key, &e);
                        enqueue()
                    }
                    (Ok(()), Err(e)) => {
                        self.failed(key, &e);
                        enqueue()
                    }
                }
            }
        }
    }

    fn launch(&self, key: GraphKey) -> Result<(), ModelError> {
        let graph = self
            .cache
            .graph(key)
            .expect("a replayed or just captured key has its graph");
        Ok(self.backend.launch(graph)?)
    }

    fn failed(&mut self, key: GraphKey, error: &dyn std::fmt::Display) {
        self.cache.capture_failed();
        tracing::warn!(
            event = "decode_graphs_unavailable",
            seqs = key.seqs,
            error = %error,
            "decode graph capture failed; decode iterations run eagerly from now on"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    fn key(seqs: u32) -> GraphKey {
        GraphKey {
            seqs,
            reduce_top_n: 0,
            table_width: 1,
            feed_word: None,
        }
    }

    /// Capture on the second occurrence of a key and replay afterwards; mixed iterations always
    /// eager; LRU eviction at the bound; a failed capture counts once and leaves every later
    /// iteration eager. Breaks if the cache grows past its bound, captures a mixed iteration or
    /// retries after a failure.
    #[test]
    fn cache_bounded_and_fallback() {
        let mut c: GraphCache<u32> = GraphCache::new(3);
        assert_eq!(c.plan(key(4), true), GraphStep::Eager);
        assert_eq!(c.plan(key(4), false), GraphStep::Eager, "mixed");
        assert_eq!(c.plan(key(4), true), GraphStep::Capture);
        c.insert(key(4), 40);
        assert_eq!(c.plan(key(4), true), GraphStep::Replay);
        assert_eq!(c.graph(key(4)), Some(&40));
        assert_eq!(c.plan(key(4), false), GraphStep::Eager, "mixed");
        // Keys differing only in the reduction or the table width are other shapes.
        let wide = GraphKey {
            table_width: 2,
            ..key(4)
        };
        assert_eq!(c.plan(wide, true), GraphStep::Eager);
        let fed = GraphKey {
            feed_word: Some(4 * 263),
            ..key(4)
        };
        assert_eq!(
            c.plan(fed, true),
            GraphStep::Eager,
            "another feed is another shape"
        );

        // Sizes 4, 1 and 2 fill the cache of 3; replaying 4 leaves 1 the least recently used,
        // so capturing a fourth size (3) evicts it.
        for s in [1, 2] {
            assert_eq!(c.plan(key(s), true), GraphStep::Eager);
            assert_eq!(c.plan(key(s), true), GraphStep::Capture);
            c.insert(key(s), s * 10);
        }
        assert_eq!(c.len(), 3);
        assert_eq!(c.plan(key(4), true), GraphStep::Replay);
        assert_eq!(c.plan(key(3), true), GraphStep::Eager);
        assert_eq!(c.plan(key(3), true), GraphStep::Capture);
        c.insert(key(3), 30);
        assert_eq!(c.len(), 3, "bounded");
        assert_eq!(
            c.graph(key(1)),
            None,
            "the least recently used size is evicted"
        );
        assert!(c.graph(key(2)).is_some() && c.graph(key(4)).is_some());
        assert_eq!(
            c.counters(),
            GraphCounters {
                captured: 4,
                replayed: 2,
                evicted: 1,
                capture_failed: 0,
            }
        );

        // A failed capture: counted once, graphs off for good.
        assert_eq!(c.plan(key(5), true), GraphStep::Eager);
        assert_eq!(c.plan(key(5), true), GraphStep::Capture);
        c.capture_failed();
        for s in [2, 4, 5, 5] {
            assert_eq!(c.plan(key(s), true), GraphStep::Eager);
        }
        assert!(c.is_empty());
        assert_eq!(c.counters().capture_failed, 1);
        assert_eq!(c.counters().replayed, 2);

        // The bound is min(max_running_requests, 64); the table width a power of two.
        assert_eq!(capacity_for(16), 16);
        assert_eq!(capacity_for(256), MAX_GRAPHS);
        assert_eq!(capacity_for(0), 1);
        assert_eq!(GraphCache::<u32>::new(1000).capacity(), MAX_GRAPHS);
        assert_eq!(
            [0, 1, 3, 4, 5, 9].map(|b| table_width(b, 64)),
            [1, 1, 4, 4, 8, 16]
        );
        assert_eq!(table_width(9, 12), 12, "capped at the full width");
    }

    /// A mock backend: `begin`/`end`/`launch` are logged; `end` fails when `fail_end`.
    #[derive(Default)]
    struct Mock {
        log: Mutex<Vec<String>>,
        fail_end: bool,
        capturing: Mutex<bool>,
    }

    impl GraphBackend for Mock {
        type Graph = u32;
        fn begin(&self) -> Result<(), KernelError> {
            *self.capturing.lock().unwrap() = true;
            self.log.lock().unwrap().push("begin".into());
            Ok(())
        }
        fn end(&self) -> Result<u32, KernelError> {
            *self.capturing.lock().unwrap() = false;
            self.log.lock().unwrap().push("end".into());
            if self.fail_end {
                Err(KernelError::InvalidArgument {
                    message: "not capturable".into(),
                })
            } else {
                Ok(7)
            }
        }
        fn launch(&self, g: &u32) -> Result<(), KernelError> {
            self.log.lock().unwrap().push(format!("launch {g}"));
            Ok(())
        }
    }

    fn pool(addr: u64) -> PoolId {
        PoolId {
            addr,
            bytes: 1024,
            num_blocks: 4,
            layer_stride_bytes: 256,
        }
    }

    /// The driver runs the work eagerly, then captures and launches it, then only launches; a
    /// failed capture re-runs the work eagerly and never captures again; another pool drops
    /// the graphs. Breaks if an iteration's work is skipped or run twice.
    #[test]
    fn decode_graphs_drive_the_backend() {
        let mock = Arc::new(Mock::default());
        let mut g = DecodeGraphs::new(Arc::clone(&mock) as Arc<dyn GraphBackend<Graph = u32>>, 4);
        let log = |m: &Mock| std::mem::take(&mut *m.log.lock().unwrap());
        let work = |m: &Arc<Mock>| {
            let m = Arc::clone(m);
            move || {
                let tag = if *m.capturing.lock().unwrap() {
                    "op (captured)"
                } else {
                    "op"
                };
                m.log.lock().unwrap().push(tag.into());
                Ok(())
            }
        };
        let k = Some(key(2));
        g.run(k, pool(1), work(&mock)).unwrap();
        assert_eq!(log(&mock), ["op"]);
        g.run(k, pool(1), work(&mock)).unwrap();
        assert_eq!(log(&mock), ["begin", "op (captured)", "end", "launch 7"]);
        g.run(k, pool(1), work(&mock)).unwrap();
        assert_eq!(log(&mock), ["launch 7"]);
        g.run(None, pool(1), work(&mock)).unwrap();
        assert_eq!(log(&mock), ["op"], "not a graph-mode iteration");
        // Another pool: the graph is dropped, the key starts over.
        g.run(k, pool(2), work(&mock)).unwrap();
        assert_eq!(log(&mock), ["op"]);
        assert_eq!(
            g.counters(),
            GraphCounters {
                captured: 1,
                replayed: 1,
                evicted: 1,
                capture_failed: 0,
            }
        );

        let failing = Arc::new(Mock {
            fail_end: true,
            ..Mock::default()
        });
        let mut g = DecodeGraphs::new(
            Arc::clone(&failing) as Arc<dyn GraphBackend<Graph = u32>>,
            4,
        );
        g.run(k, pool(1), work(&failing)).unwrap();
        g.run(k, pool(1), work(&failing)).unwrap();
        assert_eq!(log(&failing), ["op", "begin", "op (captured)", "end", "op"]);
        assert!(!g.is_enabled());
        g.run(k, pool(1), work(&failing)).unwrap();
        assert_eq!(log(&failing), ["op"]);
        assert_eq!(g.counters().capture_failed, 1);
        assert_eq!(g.counters().captured, 0);
    }
}
