//! Per-request timers on the engine clock (P2 S-7, S-8): the total request deadline
//! (`server.request_timeout`) and the slow-client timer (`server.slow_client_timeout`, running
//! while a request is paused on a full output channel). The queue wait limit
//! (`reliability.admission.queue_timeout`, C-1) is enforced by `Scheduler::plan` through the P3
//! admission gate, which drops a request that waited
//! too long with `CancelReason::QueueTimeout`; the engine plans at least every few milliseconds
//! while any request exists, so that check runs while a request is queued.
//!
//! The engine calls [`Deadlines::expired`] at every iteration boundary and cancels what it
//! returns; the scheduler then frees the blocks before the next batch is chosen. Time comes only
//! from the injected [`Clock`] (`SystemClock` in the server, `FakeClock` in tests).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use turbine_core::clock::Clock;
use turbine_core::config::ServerConfig;
use turbine_core::types::RequestId;
use turbine_scheduler::CancelReason;

/// Timers of one tracked request.
#[derive(Clone, Copy, Debug)]
struct Timers {
    /// Monotonic time at which the request times out; `None` once reported.
    deadline: Option<Duration>,
    /// Monotonic time the request was last paused (or its slow-client timer re-armed).
    paused_since: Option<Duration>,
}

/// The configured limits the timers run to.
#[derive(Clone, Copy, Debug)]
pub struct Timeouts {
    /// `server.request_timeout`.
    pub request: Duration,
    /// `server.slow_client_timeout`.
    pub slow_client: Duration,
}

impl Timeouts {
    pub fn from_config(server: &ServerConfig) -> Timeouts {
        Timeouts {
            request: server.request_timeout.0,
            slow_client: server.slow_client_timeout.0,
        }
    }
}

pub(crate) struct Deadlines {
    clock: Arc<dyn Clock>,
    request_timeout: Duration,
    slow_client_timeout: Duration,
    requests: HashMap<RequestId, Timers>,
}

impl Deadlines {
    pub fn new(clock: Arc<dyn Clock>, timeouts: Timeouts) -> Deadlines {
        Deadlines {
            clock,
            request_timeout: timeouts.request,
            slow_client_timeout: timeouts.slow_client,
            requests: HashMap::new(),
        }
    }

    /// Starts request `id`'s deadline: now + `server.request_timeout`, or `deadline_ms` (a
    /// monotonic millisecond time on the same clock; `u64::MAX` means none) when earlier.
    pub fn track(&mut self, id: RequestId, deadline_ms: u64) {
        let mut deadline = self.clock.now_mono().saturating_add(self.request_timeout);
        if deadline_ms != u64::MAX {
            deadline = deadline.min(Duration::from_millis(deadline_ms));
        }
        self.requests.insert(
            id,
            Timers {
                deadline: Some(deadline),
                paused_since: None,
            },
        );
    }

    /// Request `id`'s output channel is full: start its slow-client timer unless it runs.
    pub fn paused(&mut self, id: RequestId) {
        let now = self.clock.now_mono();
        if let Some(t) = self.requests.get_mut(&id) {
            t.paused_since.get_or_insert(now);
        }
    }

    /// Request `id`'s output channel drained: stop its slow-client timer.
    pub fn resumed(&mut self, id: RequestId) {
        if let Some(t) = self.requests.get_mut(&id) {
            t.paused_since = None;
        }
    }

    /// Request `id` is gone from the engine.
    pub fn forget(&mut self, id: RequestId) {
        self.requests.remove(&id);
    }

    /// Timers that fired since the last call, in no particular order. A request deadline is
    /// reported once; a slow-client timer is re-armed from now, so a request that stays paused
    /// after its cancellation (its final events still unread) is reported again one
    /// `server.slow_client_timeout` later. A request with both due reports `RequestTimeout`.
    pub fn expired(&mut self) -> Vec<(RequestId, CancelReason)> {
        let now = self.clock.now_mono();
        let mut out = Vec::new();
        for (&id, t) in &mut self.requests {
            if t.deadline.is_some_and(|d| now >= d) {
                t.deadline = None;
                out.push((id, CancelReason::RequestTimeout));
            } else if t
                .paused_since
                .is_some_and(|p| now.saturating_sub(p) >= self.slow_client_timeout)
            {
                t.paused_since = Some(now);
                out.push((id, CancelReason::SlowClient));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use turbine_core::clock::FakeClock;

    use super::*;

    const SECOND: Duration = Duration::from_secs(1);

    fn setup(request_timeout: Duration, slow_client_timeout: Duration) -> (FakeClock, Deadlines) {
        let clock = FakeClock::new(Duration::from_secs(100));
        let d = Deadlines::new(
            Arc::new(clock.clone()),
            Timeouts {
                request: request_timeout,
                slow_client: slow_client_timeout,
            },
        );
        (clock, d)
    }

    #[test]
    fn request_deadline_fires_once_at_the_timeout() {
        let (clock, mut d) = setup(10 * SECOND, 30 * SECOND);
        let (a, b) = (RequestId::new_v4(), RequestId::new_v4());
        d.track(a, u64::MAX);
        clock.advance(4 * SECOND);
        d.track(b, u64::MAX);
        clock.advance(6 * SECOND - Duration::from_millis(1));
        assert!(d.expired().is_empty());
        clock.advance(Duration::from_millis(1));
        assert_eq!(d.expired(), vec![(a, CancelReason::RequestTimeout)]);
        assert!(d.expired().is_empty(), "a deadline is reported once");
        clock.advance(4 * SECOND);
        assert_eq!(d.expired(), vec![(b, CancelReason::RequestTimeout)]);
    }

    #[test]
    fn an_earlier_explicit_deadline_wins() {
        let (clock, mut d) = setup(10 * SECOND, 30 * SECOND);
        let id = RequestId::new_v4();
        // now = 100 s; an explicit deadline at 102 s beats 100 s + 10 s.
        d.track(id, 102_000);
        clock.advance(2 * SECOND);
        assert_eq!(d.expired(), vec![(id, CancelReason::RequestTimeout)]);
        // A later explicit deadline does not extend server.request_timeout.
        let late = RequestId::new_v4();
        d.track(late, 1_000_000);
        clock.advance(10 * SECOND);
        assert_eq!(d.expired(), vec![(late, CancelReason::RequestTimeout)]);
    }

    #[test]
    fn slow_client_fires_only_while_paused() {
        let (clock, mut d) = setup(600 * SECOND, SECOND);
        let id = RequestId::new_v4();
        d.track(id, u64::MAX);
        clock.advance(5 * SECOND);
        assert!(d.expired().is_empty(), "not paused: no slow-client timer");

        d.paused(id);
        clock.advance(SECOND / 2);
        // A second pause while paused does not restart the timer.
        d.paused(id);
        clock.advance(SECOND / 4);
        d.resumed(id);
        clock.advance(2 * SECOND);
        assert!(d.expired().is_empty(), "resumed before the timeout");

        d.paused(id);
        clock.advance(SECOND - Duration::from_millis(1));
        assert!(d.expired().is_empty());
        clock.advance(Duration::from_millis(1));
        assert_eq!(d.expired(), vec![(id, CancelReason::SlowClient)]);
        // Still paused: re-armed, fires again one timeout later.
        clock.advance(SECOND / 2);
        assert!(d.expired().is_empty());
        clock.advance(SECOND / 2);
        assert_eq!(d.expired(), vec![(id, CancelReason::SlowClient)]);
    }

    #[test]
    fn request_timeout_wins_over_slow_client() {
        let (clock, mut d) = setup(2 * SECOND, SECOND);
        let id = RequestId::new_v4();
        d.track(id, u64::MAX);
        clock.advance(SECOND);
        d.paused(id);
        clock.advance(SECOND);
        assert_eq!(d.expired(), vec![(id, CancelReason::RequestTimeout)]);
        clock.advance(SECOND);
        assert_eq!(d.expired(), vec![(id, CancelReason::SlowClient)]);
    }

    #[test]
    fn forgotten_and_untracked_requests_never_fire() {
        let (clock, mut d) = setup(SECOND, SECOND);
        let (a, stranger) = (RequestId::new_v4(), RequestId::new_v4());
        d.track(a, u64::MAX);
        d.paused(a);
        d.paused(stranger);
        d.resumed(stranger);
        d.forget(a);
        clock.advance(10 * SECOND);
        assert!(d.expired().is_empty());
    }
}
