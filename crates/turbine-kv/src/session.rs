//! Sessions and prefetch bookkeeping (P4 S-10): a bounded state machine over a timestamp. The
//! hierarchy turns the emitted `SessionAction`s into demotions and prefetches.
//!
//! A session is keyed by its `prompt_cache_key` plus the salt of its first request, so a
//! different salt starts a different session (no cross-salt sharing). While hot (a request in
//! flight, idle < `hot_ttl`, or within an `x-turbine-session-resume-within` window) its blocks
//! get the `session_active` boost; idle sessions demote L0→L1 after `hot_ttl` and L1→L2 after
//! `warm_ttl`, and their metadata is dropped after `max_idle`.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use turbine_core::config::{KvPrefetchConfig, KvSessionConfig};
use turbine_core::request::SessionHints;
use turbine_core::types::PressureState;

use crate::directory::SessionId;
use crate::identity::KvKey;

/// Parsed and validated session hints of one request (the API layer validates them).
pub type SessionHintsParsed = SessionHints;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Warmth {
    /// Blocks keep the `session_active` boost and stay in L0.
    Hot,
    /// Idle past `hot_ttl`: blocks demoted L0 → L1 (L0 → L2 on unified devices).
    Warm,
    /// Idle past `warm_ttl`: blocks demoted L1 → L2.
    Cold,
}

#[derive(Clone, Debug)]
pub struct Session {
    pub id: SessionId,
    /// Key of the last full block of the latest finished turn.
    pub tail: Option<KvKey>,
    /// Full-block keys of the latest finished turn, in order.
    pub blocks: Vec<KvKey>,
    pub last_activity: Duration,
    /// EWMA (α = 0.3) of the idle gaps between turns.
    pub gap_ewma: Option<Duration>,
    /// End of an `x-turbine-session-resume-within` boost window.
    pub resume_until: Option<Duration>,
    pub inflight: u32,
    pub end_requested: bool,
    pub warmth: Warmth,
    /// A predicted-resume prefetch was already issued for the current idle gap.
    pub prefetched: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionAction {
    DemoteFromL0(Vec<KvKey>),
    DemoteFromL1(Vec<KvKey>),
    Prefetch(SessionId, Vec<KvKey>),
    Dropped(SessionId),
}

/// Bounds and timings of the table: `kv.session.{max_sessions, hot_ttl, warm_ttl, max_idle}`
/// and `kv.prefetch.lead_time`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionLimits {
    pub max_sessions: u32,
    pub hot_ttl: Duration,
    pub warm_ttl: Duration,
    pub max_idle: Duration,
    pub lead_time: Duration,
}

/// Weight of the newest gap in the inter-turn gap EWMA.
const GAP_ALPHA: f64 = 0.3;

pub struct SessionTable {
    limits: SessionLimits,
    sessions: HashMap<SessionId, Session>,
}

impl SessionTable {
    /// A table bounded and timed by `kv.session.*`, predicting resumes `kv.prefetch.lead_time`
    /// ahead.
    pub fn new(cfg: KvSessionConfig, prefetch: &KvPrefetchConfig) -> Self {
        Self::with_limits(SessionLimits {
            max_sessions: cfg.max_sessions,
            hot_ttl: cfg.hot_ttl.0,
            warm_ttl: cfg.warm_ttl.0,
            max_idle: cfg.max_idle.0,
            lead_time: prefetch.lead_time.0,
        })
    }

    pub fn with_limits(limits: SessionLimits) -> Self {
        SessionTable {
            limits,
            sessions: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    pub fn limits(&self) -> SessionLimits {
        self.limits
    }

    /// `kv.session.max_sessions`.
    pub fn max(&self) -> u32 {
        self.limits.max_sessions
    }

    pub fn get(&self, id: &SessionId) -> Option<&Session> {
        self.sessions.get(id)
    }

    /// The session of a `prompt_cache_key` (for a prefetch by `session_id`): the most recently
    /// active entry under any salt.
    pub fn find_by_key(&self, key: &str) -> Option<&Session> {
        self.sessions
            .values()
            .filter(|s| s.id.key == key)
            .max_by(|a, b| {
                a.last_activity
                    .cmp(&b.last_activity)
                    .then_with(|| b.id.cmp(&a.id))
            })
    }

    /// A request of the session arrived. Creates the entry, dropping the least recently active
    /// one (idle entries first) at the bound; updates the gap EWMA between turns.
    pub fn begin(&mut self, id: SessionId, hints: &SessionHints, now: Duration) {
        if !self.sessions.contains_key(&id)
            && self.sessions.len() >= self.limits.max_sessions as usize
        {
            let victim = self
                .sessions
                .values()
                .min_by(|a, b| {
                    (a.inflight > 0, a.last_activity, &a.id).cmp(&(
                        b.inflight > 0,
                        b.last_activity,
                        &b.id,
                    ))
                })
                .map(|s| s.id.clone());
            if let Some(victim) = victim {
                self.sessions.remove(&victim);
            }
        }
        let s = self.sessions.entry(id.clone()).or_insert_with(|| Session {
            id,
            tail: None,
            blocks: Vec::new(),
            last_activity: now,
            gap_ewma: None,
            resume_until: None,
            inflight: 0,
            end_requested: false,
            warmth: Warmth::Hot,
            prefetched: false,
        });
        if s.inflight == 0 && s.tail.is_some() {
            let gap = now.saturating_sub(s.last_activity).as_secs_f64();
            let ewma = match s.gap_ewma {
                None => gap,
                Some(g) => (1.0 - GAP_ALPHA) * g.as_secs_f64() + GAP_ALPHA * gap,
            };
            s.gap_ewma = Some(Duration::from_secs_f64(ewma));
        }
        s.inflight += 1;
        s.last_activity = now;
        s.warmth = Warmth::Hot;
        s.prefetched = false;
        if let Some(secs) = hints.resume_within_secs {
            s.resume_until = Some(now + Duration::from_secs(u64::from(secs)));
        }
        if hints.end {
            s.end_requested = true;
            s.resume_until = None;
        }
    }

    /// A request of the session finished with the given full-block keys. Returns true when the
    /// entry was dropped (`x-turbine-session-end` and no other request in flight).
    pub fn finish(&mut self, id: &SessionId, blocks: Vec<KvKey>, now: Duration) -> bool {
        let Some(s) = self.sessions.get_mut(id) else {
            return false;
        };
        s.inflight = s.inflight.saturating_sub(1);
        s.last_activity = now;
        if let Some(tail) = blocks.last() {
            s.tail = Some(*tail);
            s.blocks = blocks;
        }
        if s.end_requested && s.inflight == 0 {
            self.sessions.remove(id);
            return true;
        }
        false
    }

    /// Whether blocks of `id` get the `session_active` boost at `now`.
    pub fn is_hot(&self, id: &SessionId, now: Duration) -> bool {
        self.sessions.get(id).is_some_and(|s| {
            !s.end_requested
                && (s.inflight > 0
                    || now.saturating_sub(s.last_activity) < self.limits.hot_ttl
                    || s.resume_until.is_some_and(|u| now < u))
        })
    }

    /// Time-driven transitions: TTL demotions, metadata expiry after `max_idle`, and one
    /// predicted-resume prefetch per idle gap at last activity + gap EWMA − lead time, only at
    /// GREEN or YELLOW. Sessions are visited in `SessionId` order so simulations are
    /// deterministic.
    pub fn sweep(&mut self, now: Duration, pressure: PressureState) -> Vec<SessionAction> {
        let mut out = Vec::new();
        let mut expired = Vec::new();
        let mut ids: Vec<SessionId> = self.sessions.keys().cloned().collect();
        ids.sort();
        for id in ids {
            let Some(s) = self.sessions.get_mut(&id) else {
                continue;
            };
            if s.inflight > 0 {
                continue;
            }
            let idle = now.saturating_sub(s.last_activity);
            if idle >= self.limits.max_idle {
                expired.push(id);
                continue;
            }
            let boosted = s.resume_until.is_some_and(|u| now < u);
            if s.warmth == Warmth::Hot && idle >= self.limits.hot_ttl && !boosted {
                s.warmth = Warmth::Warm;
                out.push(SessionAction::DemoteFromL0(s.blocks.clone()));
            }
            if s.warmth == Warmth::Warm && idle >= self.limits.warm_ttl && !boosted {
                s.warmth = Warmth::Cold;
                out.push(SessionAction::DemoteFromL1(s.blocks.clone()));
            }
            if let Some(gap) = s.gap_ewma {
                let due = (s.last_activity + gap).saturating_sub(self.limits.lead_time);
                if s.warmth != Warmth::Hot
                    && !s.prefetched
                    && now >= due
                    && pressure <= PressureState::Yellow
                {
                    s.prefetched = true;
                    out.push(SessionAction::Prefetch(s.id.clone(), s.blocks.clone()));
                }
            }
        }
        for id in expired {
            self.sessions.remove(&id);
            out.push(SessionAction::Dropped(id));
        }
        out
    }
}

/// Prefetched blocks not yet used by a request, and the outcome counters of
/// `turbine_kv_prefetch_total{outcome}`: each issued block counts once.
#[derive(Debug, Default)]
pub struct PrefetchTracker {
    outstanding: HashSet<KvKey>,
    pub used: u64,
    pub wasted: u64,
    pub cancelled: u64,
    pub rejected: u64,
}

impl PrefetchTracker {
    pub fn issued(&mut self, key: KvKey) {
        self.outstanding.insert(key);
    }

    /// A request attached the block: `used`, once.
    pub fn attached(&mut self, key: &KvKey) -> bool {
        let hit = self.outstanding.remove(key);
        self.used += u64::from(hit);
        hit
    }

    /// The block left L0 before any request used it: `wasted`, once.
    pub fn evicted(&mut self, key: &KvKey) -> bool {
        let hit = self.outstanding.remove(key);
        self.wasted += u64::from(hit);
        hit
    }

    /// The prefetch was cancelled before completing: `cancelled`, once.
    pub fn cancelled(&mut self, key: &KvKey) -> bool {
        let hit = self.outstanding.remove(key);
        self.cancelled += u64::from(hit);
        hit
    }

    /// `n` blocks were refused because the prefetch queue was full.
    pub fn reject(&mut self, n: u64) {
        self.rejected += n;
    }

    pub fn outstanding(&self) -> usize {
        self.outstanding.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: fn(u64) -> Duration = Duration::from_secs;

    /// hot 60 s / warm 600 s / idle 1 h / lead 2 s.
    fn table(max: u32) -> SessionTable {
        SessionTable::with_limits(SessionLimits {
            max_sessions: max,
            hot_ttl: S(60),
            warm_ttl: S(600),
            max_idle: S(3600),
            lead_time: S(2),
        })
    }

    fn sid(k: &str) -> SessionId {
        SessionId {
            key: k.into(),
            salt: String::new(),
        }
    }

    fn hints(resume: Option<u32>, end: bool) -> SessionHints {
        SessionHints {
            session_id: "x".into(),
            resume_within_secs: resume,
            end,
        }
    }

    fn prefetches(actions: &[SessionAction]) -> usize {
        actions
            .iter()
            .filter(|a| matches!(a, SessionAction::Prefetch(..)))
            .count()
    }

    /// `kv.session.*` and `kv.prefetch.lead_time` become the table's limits.
    #[test]
    fn new_takes_limits_from_config() {
        use turbine_core::config::{HumanDuration, KvPrefetchConfig, KvSessionConfig};

        let cfg = KvSessionConfig {
            max_sessions: 7,
            hot_ttl: HumanDuration::from_secs(30),
            warm_ttl: HumanDuration::from_secs(300),
            max_idle: HumanDuration::from_secs(900),
        };
        let prefetch = KvPrefetchConfig {
            lead_time: HumanDuration::from_millis(1500),
            max_queue: 16,
        };
        let t = SessionTable::new(cfg, &prefetch);
        assert_eq!(t.max(), 7);
        assert_eq!(
            t.limits(),
            SessionLimits {
                max_sessions: 7,
                hot_ttl: S(30),
                warm_ttl: S(300),
                max_idle: S(900),
                lead_time: Duration::from_millis(1500),
            }
        );
        assert!(t.is_empty());
    }

    #[test]
    fn lifecycle_and_prefetch() {
        let keys = vec![KvKey([1; 16]), KvKey([2; 16])];
        let mut t = table(3);
        assert_eq!(t.max(), 3);
        t.begin(sid("a"), &hints(None, false), S(0));
        assert!(!t.finish(&sid("a"), keys.clone(), S(1)));
        assert_eq!(t.get(&sid("a")).unwrap().tail, Some(keys[1]));
        assert!(t.is_hot(&sid("a"), S(30)));
        assert!(t.sweep(S(30), PressureState::Green).is_empty());
        assert_eq!(
            t.sweep(S(61), PressureState::Green),
            [SessionAction::DemoteFromL0(keys.clone())],
            "idle past hot_ttl: L0 -> L1 (L2 on unified devices)"
        );
        assert!(!t.is_hot(&sid("a"), S(61)));
        assert_eq!(
            t.sweep(S(601), PressureState::Green),
            [SessionAction::DemoteFromL1(keys.clone())],
            "idle past warm_ttl: L1 -> L2"
        );
        assert!(
            t.sweep(S(700), PressureState::Green).is_empty(),
            "once each"
        );

        // Turns after gaps of 800 s and 200 s: gap EWMA 0.7·800 + 0.3·200 s.
        t.begin(sid("a"), &hints(None, false), S(801));
        assert!(!t.finish(&sid("a"), keys.clone(), S(802)));
        assert_eq!(t.get(&sid("a")).unwrap().gap_ewma, Some(S(800)));
        t.begin(sid("a"), &hints(None, false), S(1002));
        t.finish(&sid("a"), keys.clone(), S(1003));
        let gap = t.get(&sid("a")).unwrap().gap_ewma.unwrap();
        assert_eq!(gap, Duration::from_secs_f64(0.7 * 800.0 + 0.3 * 200.0));
        // Predicted resume at last activity + gap EWMA − lead time, only at GREEN/YELLOW.
        let due = S(1003) + gap - S(2);
        assert_eq!(prefetches(&t.sweep(S(1064), PressureState::Green)), 0);
        assert_eq!(prefetches(&t.sweep(due - S(1), PressureState::Green)), 0);
        assert_eq!(
            prefetches(&t.sweep(due, PressureState::Orange)),
            0,
            "no predicted prefetch at ORANGE"
        );
        assert!(
            t.sweep(due, PressureState::Yellow)
                .contains(&SessionAction::Prefetch(sid("a"), keys.clone()))
        );
        assert_eq!(
            prefetches(&t.sweep(due + S(1), PressureState::Green)),
            0,
            "once per idle gap"
        );
        assert_eq!(t.find_by_key("a").map(|s| &s.id), Some(&sid("a")));
        assert!(t.find_by_key("zz").is_none());

        // resume-within 600 keeps the boost 600 s, past hot_ttl, and suppresses the demotion.
        t.begin(sid("b"), &hints(Some(600), false), S(2000));
        t.finish(&sid("b"), keys.clone(), S(2000));
        assert!(t.is_hot(&sid("b"), S(2599)));
        assert!(
            !t.sweep(S(2599), PressureState::Green)
                .contains(&SessionAction::DemoteFromL0(keys.clone()))
        );
        assert!(!t.is_hot(&sid("b"), S(2601)));

        // session-end releases the boost at once; the entry goes after the last in-flight request.
        t.begin(sid("c"), &hints(None, false), S(3000));
        t.begin(sid("c"), &hints(None, true), S(3000));
        assert!(!t.is_hot(&sid("c"), S(3000)), "end releases the boost");
        assert!(
            !t.finish(&sid("c"), keys.clone(), S(3001)),
            "another request is still in flight"
        );
        assert!(t.finish(&sid("c"), keys.clone(), S(3002)));
        assert!(t.get(&sid("c")).is_none());

        // Bounded: the least recently active entry is dropped.
        t.begin(sid("d"), &hints(None, false), S(4000));
        t.finish(&sid("d"), vec![], S(4000));
        t.begin(sid("e"), &hints(None, false), S(4001));
        t.finish(&sid("e"), vec![], S(4001));
        assert_eq!(t.len(), 3);
        assert!(
            t.get(&sid("a")).is_none(),
            "the oldest session metadata is dropped"
        );

        // Expiry after max_idle.
        let swept = t.sweep(S(4001) + S(3600), PressureState::Green);
        assert!(
            swept.contains(&SessionAction::Dropped(sid("e"))),
            "{swept:?}"
        );
        assert!(t.is_empty());

        // Prefetch accounting: each prefetched block counts once, used or wasted.
        let mut p = PrefetchTracker::default();
        p.issued(keys[0]);
        p.issued(keys[1]);
        p.issued(KvKey([3; 16]));
        assert_eq!(p.outstanding(), 3);
        assert!(p.attached(&keys[0]));
        assert!(!p.attached(&keys[0]), "counted once");
        assert!(p.evicted(&keys[1]));
        assert!(!p.evicted(&keys[0]), "a used block is never counted wasted");
        assert!(p.cancelled(&KvKey([3; 16])));
        p.reject(2);
        assert_eq!((p.used, p.wasted, p.cancelled, p.rejected), (1, 1, 1, 2));
        assert_eq!(p.outstanding(), 0);
    }

    #[test]
    fn sessions_are_salt_scoped_and_idle_entries_go_first() {
        let mut t = table(2);
        let salted = SessionId {
            key: "a".into(),
            salt: "s".into(),
        };
        t.begin(sid("a"), &hints(None, false), S(0));
        t.begin(salted.clone(), &hints(None, false), S(5));
        t.finish(&salted, vec![KvKey([9; 16])], S(6));
        assert_eq!(
            t.len(),
            2,
            "the same key under another salt is another session"
        );
        assert_eq!(
            t.find_by_key("a").map(|s| &s.id),
            Some(&salted),
            "a prefetch by session id uses the most recently active entry"
        );
        // "a" is older but still has a request in flight: the idle entry is dropped instead.
        t.begin(sid("b"), &hints(None, false), S(10));
        assert!(t.get(&sid("a")).is_some());
        assert!(t.get(&salted).is_none());
        assert_eq!(t.len(), 2);
    }
}
