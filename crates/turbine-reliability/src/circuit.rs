//! Circuit breaker `HEALTHY → DEGRADED → CIRCUIT_OPEN → DRAINING → PROBING → HEALTHY`
//! (P3 S-12, the transition table). Time is passed in explicitly (`now`, monotonic).

use std::collections::VecDeque;
use std::time::Duration;

use serde::Serialize;
use turbine_core::config::CircuitConfig;
use turbine_core::types::CircuitState;

use crate::metrics::{CircuitTransitionLabels, ReliabilityMetrics, StateLabel};

/// A probe succeeds only within this multiple of the latency baseline (P3 transition table).
pub const PROBE_LATENCY_LIMIT: f64 = 2.0;

/// Closed set of circuit transition reasons (metric label `reason`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CircuitReason {
    LatencyDrift,
    ThermalThrottle,
    OomRecovered,
    TelemetryStale,
    RepeatedOom,
    RecoveryFailed,
    DeviceError,
    ProbeFailed,
    DeviceFatal,
    ControllerFailed,
    NoTrigger,
    DrainStarted,
    DrainComplete,
    ProbesSucceeded,
}

impl CircuitReason {
    pub const ALL: [CircuitReason; 14] = [
        CircuitReason::LatencyDrift,
        CircuitReason::ThermalThrottle,
        CircuitReason::OomRecovered,
        CircuitReason::TelemetryStale,
        CircuitReason::RepeatedOom,
        CircuitReason::RecoveryFailed,
        CircuitReason::DeviceError,
        CircuitReason::ProbeFailed,
        CircuitReason::DeviceFatal,
        CircuitReason::ControllerFailed,
        CircuitReason::NoTrigger,
        CircuitReason::DrainStarted,
        CircuitReason::DrainComplete,
        CircuitReason::ProbesSucceeded,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            CircuitReason::LatencyDrift => "latency_drift",
            CircuitReason::ThermalThrottle => "thermal_throttle",
            CircuitReason::OomRecovered => "oom_recovered",
            CircuitReason::TelemetryStale => "telemetry_stale",
            CircuitReason::RepeatedOom => "repeated_oom",
            CircuitReason::RecoveryFailed => "recovery_failed",
            CircuitReason::DeviceError => "device_error",
            CircuitReason::ProbeFailed => "probe_failed",
            CircuitReason::DeviceFatal => "device_fatal",
            CircuitReason::ControllerFailed => "controller_failed",
            CircuitReason::NoTrigger => "no_trigger",
            CircuitReason::DrainStarted => "drain_started",
            CircuitReason::DrainComplete => "drain_complete",
            CircuitReason::ProbesSucceeded => "probes_succeeded",
        }
    }
}

/// Inputs to the breaker.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CircuitEvent {
    /// An engine iteration ran (drift and thermal triggers need one within `window`).
    Iteration,
    /// Windowed p95 step time / baseline.
    LatencyDrift {
        ratio: f64,
    },
    ThermalThrottle,
    TelemetryStale,
    OomRecovered,
    RecoveryFailed,
    /// Non-OOM kernel/device error; `sticky` = context-corrupting (drain, then exit 3).
    DeviceError {
        sticky: bool,
    },
    /// The controller task panicked (fatal).
    ControllerFailed,
    /// Probe latency / baseline.
    ProbeSucceeded {
        latency_ratio: f64,
    },
    ProbeFailed,
    /// Controller tick with the number of running sequences.
    Tick {
        running: u32,
    },
}

/// `(from, to, reason)`.
pub type CircuitTransition = (CircuitState, CircuitState, CircuitReason);

pub struct CircuitBreaker {
    cfg: CircuitConfig,
    state: CircuitState,
    since: Duration,
    /// `now` of the latest event.
    last_now: Duration,
    /// Last DEGRADED trigger (DEGRADED → HEALTHY after `window` without one).
    last_trigger: Duration,
    last_iteration: Option<Duration>,
    /// OOM recoveries within `window` (sliding).
    ooms: VecDeque<Duration>,
    opened_at: Duration,
    probes_ok: u32,
    fatal: bool,
    last_reason: Option<CircuitReason>,
    metrics: ReliabilityMetrics,
}

impl std::fmt::Debug for CircuitBreaker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CircuitBreaker")
            .field("state", &self.state)
            .field("since", &self.since)
            .field("fatal", &self.fatal)
            .field("last_reason", &self.last_reason)
            .finish_non_exhaustive()
    }
}

impl CircuitBreaker {
    pub fn new(cfg: &CircuitConfig, metrics: ReliabilityMetrics, now: Duration) -> Self {
        let b = CircuitBreaker {
            cfg: cfg.clone(),
            state: CircuitState::Healthy,
            since: now,
            last_now: now,
            last_trigger: now,
            last_iteration: None,
            ooms: VecDeque::new(),
            opened_at: now,
            probes_ok: 0,
            fatal: false,
            last_reason: None,
            metrics,
        };
        b.publish();
        b
    }

    pub fn state(&self) -> CircuitState {
        self.state
    }

    /// Monotonic time of the last transition (or construction).
    pub fn since(&self) -> Duration {
        self.since
    }

    pub fn last_reason(&self) -> Option<CircuitReason> {
        self.last_reason
    }

    /// A sticky device error or controller failure: drain, then exit the process with code 3.
    pub fn is_fatal(&self) -> bool {
        self.fatal
    }

    /// Remaining cooldown in whole seconds, at least 1 (`Retry-After` of `circuit_open`).
    pub fn retry_after_secs(&self) -> u64 {
        let end = self.opened_at + self.cfg.cooldown.0;
        let remaining = end.saturating_sub(self.last_now);
        let secs = remaining.as_secs() + u64::from(remaining.subsec_nanos() > 0);
        secs.max(1)
    }

    /// Running sequences still active `drain_timeout` after opening fail (`circuit_open`).
    pub fn drain_expired(&self) -> bool {
        self.state == CircuitState::Draining
            && self.last_now.saturating_sub(self.opened_at) >= self.cfg.drain_timeout.0
    }

    /// True for PROBING → HEALTHY: reset the latency baseline then.
    pub fn baseline_reset_due(t: &CircuitTransition) -> bool {
        t.0 == CircuitState::Probing && t.1 == CircuitState::Healthy
    }

    fn worked_recently(&self, now: Duration) -> bool {
        self.last_iteration
            .is_some_and(|t| now.saturating_sub(t) <= self.cfg.window.0)
    }

    pub fn on_event(&mut self, ev: CircuitEvent, now: Duration) -> Option<CircuitTransition> {
        self.last_now = now;
        // DRAINING and CIRCUIT_OPEN are already open; only a fatal error changes them.
        let can_open = matches!(
            self.state,
            CircuitState::Healthy | CircuitState::Degraded | CircuitState::Probing
        );
        match ev {
            CircuitEvent::Iteration => {
                self.last_iteration = Some(now);
                None
            }
            // Drift and thermal are ignored while idle (no iteration in the window).
            CircuitEvent::LatencyDrift { .. } | CircuitEvent::ThermalThrottle
                if !self.worked_recently(now) =>
            {
                None
            }
            CircuitEvent::LatencyDrift { ratio } => {
                if ratio >= self.cfg.latency_drift_open && can_open {
                    self.open(CircuitReason::LatencyDrift, now)
                } else if ratio >= self.cfg.latency_drift_degraded {
                    self.degrade(CircuitReason::LatencyDrift, now)
                } else {
                    None
                }
            }
            CircuitEvent::ThermalThrottle => self.degrade(CircuitReason::ThermalThrottle, now),
            CircuitEvent::TelemetryStale => self.degrade(CircuitReason::TelemetryStale, now),
            CircuitEvent::OomRecovered => {
                self.ooms.push_back(now);
                while self
                    .ooms
                    .front()
                    .is_some_and(|t| now.saturating_sub(*t) > self.cfg.window.0)
                {
                    self.ooms.pop_front();
                }
                let repeated = self.ooms.len() >= self.cfg.oom_recoveries_to_open as usize;
                if repeated && can_open {
                    self.open(CircuitReason::RepeatedOom, now)
                } else {
                    self.degrade(CircuitReason::OomRecovered, now)
                }
            }
            CircuitEvent::RecoveryFailed if can_open => {
                self.open(CircuitReason::RecoveryFailed, now)
            }
            CircuitEvent::DeviceError { sticky: false } if can_open => {
                self.open(CircuitReason::DeviceError, now)
            }
            CircuitEvent::RecoveryFailed | CircuitEvent::DeviceError { sticky: false } => None,
            CircuitEvent::DeviceError { sticky: true } | CircuitEvent::ControllerFailed => {
                self.fatal = true;
                let reason = if ev == CircuitEvent::ControllerFailed {
                    CircuitReason::ControllerFailed
                } else {
                    CircuitReason::DeviceFatal
                };
                if can_open {
                    self.open(reason, now)
                } else {
                    // Already open or draining: keep draining, now towards exit 3.
                    self.last_reason = Some(reason);
                    None
                }
            }
            CircuitEvent::ProbeFailed if self.state == CircuitState::Probing => {
                self.open(CircuitReason::ProbeFailed, now)
            }
            CircuitEvent::ProbeSucceeded { latency_ratio }
                if self.state == CircuitState::Probing =>
            {
                // NaN (no measurement) counts as a failed probe.
                let within = latency_ratio.is_finite() && latency_ratio <= PROBE_LATENCY_LIMIT;
                if !within {
                    return self.open(CircuitReason::ProbeFailed, now);
                }
                self.probes_ok += 1;
                if self.probes_ok < self.cfg.probe_successes {
                    return None;
                }
                self.ooms.clear();
                self.last_trigger = now;
                self.go(CircuitState::Healthy, CircuitReason::ProbesSucceeded, now)
            }
            CircuitEvent::ProbeFailed | CircuitEvent::ProbeSucceeded { .. } => None,
            CircuitEvent::Tick { running } => match self.state {
                CircuitState::CircuitOpen => {
                    self.go(CircuitState::Draining, CircuitReason::DrainStarted, now)
                }
                CircuitState::Draining
                    if !self.fatal
                        && running == 0
                        && now.saturating_sub(self.opened_at) >= self.cfg.cooldown.0 =>
                {
                    self.probes_ok = 0;
                    self.go(CircuitState::Probing, CircuitReason::DrainComplete, now)
                }
                CircuitState::Degraded
                    if now.saturating_sub(self.last_trigger) >= self.cfg.window.0 =>
                {
                    self.go(CircuitState::Healthy, CircuitReason::NoTrigger, now)
                }
                _ => None,
            },
        }
    }

    fn degrade(&mut self, reason: CircuitReason, now: Duration) -> Option<CircuitTransition> {
        self.last_trigger = now;
        if self.state == CircuitState::Healthy {
            self.go(CircuitState::Degraded, reason, now)
        } else {
            None
        }
    }

    fn open(&mut self, reason: CircuitReason, now: Duration) -> Option<CircuitTransition> {
        self.opened_at = now;
        self.go(CircuitState::CircuitOpen, reason, now)
    }

    fn go(
        &mut self,
        to: CircuitState,
        reason: CircuitReason,
        now: Duration,
    ) -> Option<CircuitTransition> {
        let from = self.state;
        self.state = to;
        self.since = now;
        self.last_reason = Some(reason);
        self.metrics
            .circuit_transitions
            .get_or_create(&CircuitTransitionLabels {
                from: from.as_str(),
                to: to.as_str(),
                reason: reason.as_str(),
            })
            .inc();
        self.publish();
        tracing::warn!(
            event = "circuit_transition",
            from = from.as_str(),
            to = to.as_str(),
            reason = reason.as_str(),
            fatal = self.fatal,
        );
        Some((from, to, reason))
    }

    fn publish(&self) {
        for s in CircuitState::ALL {
            self.metrics
                .circuit_state
                .get_or_create(&StateLabel { state: s.as_str() })
                .set(i64::from(s == self.state));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use CircuitEvent as E;
    use CircuitReason as R;
    use CircuitState::*;

    fn s(n: u64) -> Duration {
        Duration::from_secs(n)
    }
    fn breaker() -> CircuitBreaker {
        CircuitBreaker::new(
            &CircuitConfig::default(),
            ReliabilityMetrics::unregistered(),
            Duration::ZERO,
        )
    }
    /// Open with `ev`, then walk OPEN → DRAINING → PROBING.
    fn to_probing(b: &mut CircuitBreaker, at: u64) {
        b.on_event(E::Tick { running: 0 }, s(at));
        assert_eq!(b.state(), Draining);
        assert_eq!(
            b.on_event(E::Tick { running: 0 }, s(at + 29)),
            None,
            "cooldown not elapsed"
        );
        assert_eq!(
            b.on_event(E::Tick { running: 0 }, s(at + 30)),
            Some((Draining, Probing, R::DrainComplete))
        );
    }

    #[test]
    fn transition_table() {
        // Drift and thermal triggers are ignored without an iteration in the window.
        let mut b = breaker();
        assert_eq!(b.on_event(E::LatencyDrift { ratio: 5.0 }, s(1)), None);
        assert_eq!(b.on_event(E::ThermalThrottle, s(1)), None);
        b.on_event(E::Iteration, s(2));
        assert_eq!(b.on_event(E::LatencyDrift { ratio: 1.9 }, s(2)), None);
        assert_eq!(
            b.on_event(E::LatencyDrift { ratio: 2.0 }, s(3)),
            Some((Healthy, Degraded, R::LatencyDrift))
        );
        // DEGRADED → HEALTHY after `window` with no trigger.
        assert_eq!(b.on_event(E::Tick { running: 1 }, s(62)), None);
        assert_eq!(
            b.on_event(E::Tick { running: 1 }, s(63)),
            Some((Degraded, Healthy, R::NoTrigger))
        );
        // Other HEALTHY → DEGRADED triggers.
        for (ev, reason) in [
            (E::ThermalThrottle, R::ThermalThrottle),
            (E::OomRecovered, R::OomRecovered),
            (E::TelemetryStale, R::TelemetryStale),
        ] {
            let mut b = breaker();
            b.on_event(E::Iteration, s(1));
            assert_eq!(b.on_event(ev, s(1)), Some((Healthy, Degraded, reason)));
        }
        // any → CIRCUIT_OPEN.
        let mut b = breaker();
        b.on_event(E::OomRecovered, s(1));
        b.on_event(E::OomRecovered, s(2));
        assert_eq!(
            b.on_event(E::OomRecovered, s(3)),
            Some((Degraded, CircuitOpen, R::RepeatedOom))
        );
        let mut b = breaker();
        b.on_event(E::OomRecovered, s(1));
        b.on_event(E::OomRecovered, s(2));
        assert_eq!(
            b.on_event(E::OomRecovered, s(70)),
            None,
            "recoveries outside the window do not count"
        );
        let mut b = breaker();
        b.on_event(E::Iteration, s(1));
        assert_eq!(
            b.on_event(E::LatencyDrift { ratio: 4.0 }, s(1)),
            Some((Healthy, CircuitOpen, R::LatencyDrift))
        );
        assert_eq!(
            breaker().on_event(E::RecoveryFailed, s(1)),
            Some((Healthy, CircuitOpen, R::RecoveryFailed))
        );
        assert_eq!(
            breaker().on_event(E::DeviceError { sticky: false }, s(1)),
            Some((Healthy, CircuitOpen, R::DeviceError))
        );
        // CIRCUIT_OPEN → DRAINING immediately (next tick); retry-after is the remaining cooldown.
        let mut b = breaker();
        b.on_event(E::DeviceError { sticky: false }, s(10));
        assert_eq!(b.retry_after_secs(), 30);
        assert_eq!(
            b.on_event(E::Tick { running: 3 }, s(10)),
            Some((CircuitOpen, Draining, R::DrainStarted))
        );
        // DRAINING → PROBING needs both: running finished and cooldown elapsed.
        assert_eq!(b.on_event(E::Tick { running: 0 }, s(39)), None);
        assert_eq!(b.retry_after_secs(), 1);
        assert_eq!(b.on_event(E::Tick { running: 2 }, s(40)), None);
        assert!(!b.drain_expired());
        assert_eq!(b.on_event(E::Tick { running: 2 }, s(130)), None);
        assert!(
            b.drain_expired(),
            "drain_timeout reached: running sequences fail with circuit_open"
        );
        assert_eq!(
            b.on_event(E::Tick { running: 0 }, s(131)),
            Some((Draining, Probing, R::DrainComplete))
        );
        // PROBING → HEALTHY after probe_successes consecutive good probes (≤ 2 × baseline).
        assert_eq!(
            b.on_event(E::ProbeSucceeded { latency_ratio: 1.1 }, s(132)),
            None
        );
        assert_eq!(
            b.on_event(E::ProbeSucceeded { latency_ratio: 1.9 }, s(133)),
            None
        );
        let back = b.on_event(E::ProbeSucceeded { latency_ratio: 1.0 }, s(134));
        assert_eq!(back, Some((Probing, Healthy, R::ProbesSucceeded)));
        assert!(CircuitBreaker::baseline_reset_due(&back.unwrap()));
        // PROBING → CIRCUIT_OPEN on a failed or too-slow probe.
        let mut b = breaker();
        b.on_event(E::RecoveryFailed, s(0));
        to_probing(&mut b, 0);
        assert_eq!(
            b.on_event(E::ProbeFailed, s(31)),
            Some((Probing, CircuitOpen, R::ProbeFailed))
        );
        let mut b = breaker();
        b.on_event(E::RecoveryFailed, s(0));
        to_probing(&mut b, 0);
        assert_eq!(
            b.on_event(E::ProbeSucceeded { latency_ratio: 2.5 }, s(31)),
            Some((Probing, CircuitOpen, R::ProbeFailed))
        );
        // Sticky device error: CIRCUIT_OPEN (device_fatal), drains, never probes.
        let mut b = breaker();
        assert_eq!(
            b.on_event(E::DeviceError { sticky: true }, s(1)),
            Some((Healthy, CircuitOpen, R::DeviceFatal))
        );
        assert!(b.is_fatal());
        b.on_event(E::Tick { running: 0 }, s(1));
        assert_eq!(b.on_event(E::Tick { running: 0 }, s(100)), None);
        assert_eq!(b.state(), Draining);
        // Controller panic is fatal too.
        let mut b = breaker();
        assert_eq!(
            b.on_event(E::ControllerFailed, s(1)),
            Some((Healthy, CircuitOpen, R::ControllerFailed))
        );
        assert!(b.is_fatal());
        // A sticky error while already draining keeps draining, now fatal.
        let mut b = breaker();
        b.on_event(E::DeviceError { sticky: false }, s(1));
        b.on_event(E::Tick { running: 1 }, s(1));
        assert_eq!(b.on_event(E::DeviceError { sticky: true }, s(2)), None);
        assert!(b.is_fatal() && b.state() == Draining);
        assert_eq!(b.last_reason(), Some(R::DeviceFatal));

        // Each transition is counted and the state gauge follows.
        let metrics = ReliabilityMetrics::unregistered();
        let mut b = CircuitBreaker::new(&CircuitConfig::default(), metrics.clone(), s(0));
        b.on_event(E::TelemetryStale, s(1));
        let counted = metrics
            .circuit_transitions
            .get_or_create(&CircuitTransitionLabels {
                from: "HEALTHY",
                to: "DEGRADED",
                reason: "telemetry_stale",
            })
            .get();
        assert_eq!(counted, 1);
        let gauge = |state| {
            metrics
                .circuit_state
                .get_or_create(&StateLabel { state })
                .get()
        };
        assert_eq!((gauge("HEALTHY"), gauge("DEGRADED")), (0, 1));
        let names: Vec<&str> = CircuitReason::ALL.iter().map(|r| r.as_str()).collect();
        let mut unique = names.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), names.len(), "reason codes are distinct");
    }
}
