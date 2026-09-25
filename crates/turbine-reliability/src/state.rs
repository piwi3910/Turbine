//! Pressure state machine with hysteresis (P3 S-7): `GREEN → YELLOW → ORANGE → RED →
//! SURVIVAL`, escalating after `escalate_samples` consecutive samples (allocation failure:
//! at once) and de-escalating one level per `deescalate_dwell` of all-clear signals.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use turbine_core::clock::Clock;
use turbine_core::config::ReliabilityConfig;
use turbine_core::types::PressureState;

use crate::metrics::{ReliabilityMetrics, StateLabel, TransitionLabels};
use crate::signals::{PressureSignal, SignalThresholds, SignalValue};

/// Transitions kept for the pressure document.
pub const TRANSITION_HISTORY: usize = 32;

/// One explained state change.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transition {
    /// Monotonic time.
    pub at: Duration,
    pub at_wall: SystemTime,
    pub from: PressureState,
    pub to: PressureState,
    /// The dominant signal that caused the change.
    pub signal: PressureSignal,
    pub value: f64,
    /// The threshold crossed (entering `to`, or the exit threshold of `from` when descending).
    pub threshold: f64,
}

/// Inputs from outside the signal table: the circuit's pressure floor (DEGRADED → YELLOW,
/// with the signal that explains it) and whether the emergency reserve is held (the state
/// may not drop below RED otherwise).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gates {
    pub floor: PressureState,
    pub floor_signal: PressureSignal,
    pub reserve_held: bool,
}

impl Default for Gates {
    fn default() -> Self {
        Gates {
            floor: PressureState::Green,
            floor_signal: PressureSignal::TelemetryStale,
            reserve_held: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MachineConfig {
    /// false: the state stays GREEN.
    pub enabled: bool,
    pub escalate_samples: u32,
    pub deescalate_dwell: Duration,
    pub exit_margin: f64,
}

impl MachineConfig {
    pub fn from_config(cfg: &ReliabilityConfig) -> Self {
        MachineConfig {
            enabled: cfg.enabled,
            escalate_samples: cfg.pressure.escalate_samples,
            deescalate_dwell: cfg.pressure.deescalate_dwell.0,
            exit_margin: cfg.pressure.exit_margin,
        }
    }
}

pub struct PressureMachine {
    cfg: MachineConfig,
    thresholds: BTreeMap<PressureSignal, SignalThresholds>,
    state: PressureState,
    since_wall: SystemTime,
    /// Consecutive samples whose target was above the current state.
    up_count: u32,
    /// Minimum target seen across those samples.
    up_level: PressureState,
    /// Start of the current all-clear run (restarts at each de-escalation step).
    clear_since: Option<Duration>,
    history: VecDeque<Transition>,
    metrics: ReliabilityMetrics,
    clock: Arc<dyn Clock>,
}

impl std::fmt::Debug for PressureMachine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PressureMachine")
            .field("cfg", &self.cfg)
            .field("state", &self.state)
            .field("up_count", &self.up_count)
            .field("clear_since", &self.clear_since)
            .finish_non_exhaustive()
    }
}

impl PressureMachine {
    pub fn new(
        cfg: MachineConfig,
        thresholds: BTreeMap<PressureSignal, SignalThresholds>,
        metrics: ReliabilityMetrics,
        clock: Arc<dyn Clock>,
    ) -> Self {
        let m = PressureMachine {
            cfg,
            thresholds,
            state: PressureState::Green,
            since_wall: clock.now_wall(),
            up_count: 0,
            up_level: PressureState::Green,
            clear_since: None,
            history: VecDeque::with_capacity(TRANSITION_HISTORY),
            metrics,
            clock,
        };
        m.publish_state();
        m
    }

    pub fn state(&self) -> PressureState {
        self.state
    }

    /// Wall time of the last transition (or of construction).
    pub fn since_wall(&self) -> SystemTime {
        self.since_wall
    }

    /// Oldest first; at most [`TRANSITION_HISTORY`] entries.
    pub fn history(&self) -> impl DoubleEndedIterator<Item = &Transition> + ExactSizeIterator {
        self.history.iter()
    }

    /// Evaluate one telemetry tick; stale signals are ignored.
    pub fn evaluate(&mut self, signals: &[SignalValue], gates: Gates) -> Option<Transition> {
        if !self.cfg.enabled {
            return None;
        }
        let now = self.clock.now_mono();
        let live: Vec<&SignalValue> = signals.iter().filter(|s| !s.stale).collect();
        // Highest level wins; ties go to the earlier signal of the table.
        let dominant = live
            .iter()
            .copied()
            .max_by(|a, b| a.level.cmp(&b.level).then(b.signal.cmp(&a.signal)));
        let target = dominant
            .map_or(PressureState::Green, |d| d.level)
            .max(gates.floor);

        if let Some(s) = live.iter().find(|s| {
            s.signal == PressureSignal::AllocationFailure && s.level == PressureState::Survival
        }) && self.state < PressureState::Survival
        {
            let threshold = self
                .threshold_of(s.signal, PressureState::Survival)
                .unwrap_or(s.value);
            return Some(self.transition(
                PressureState::Survival,
                s.signal,
                s.value,
                threshold,
                now,
            ));
        }

        if target > self.state {
            self.clear_since = None;
            self.up_level = if self.up_count == 0 {
                target
            } else {
                self.up_level.min(target)
            };
            self.up_count += 1;
            if self.up_count < self.cfg.escalate_samples {
                return None;
            }
            let to = self.up_level;
            let (signal, value, threshold) = match dominant {
                Some(d) if d.level >= to => (
                    d.signal,
                    d.value,
                    self.threshold_of(d.signal, to).unwrap_or(d.value),
                ),
                // Only the floor gate asks for `to`: a gate is a boolean input, 1 = active.
                _ => (gates.floor_signal, 1.0, 1.0),
            };
            return Some(self.transition(to, signal, value, threshold, now));
        }
        self.up_count = 0;

        let at_floor = self.state <= gates.floor;
        let reserve_blocks = self.state == PressureState::Red && !gates.reserve_held;
        let all_clear = live.iter().all(|s| {
            self.thresholds
                .get(&s.signal)
                .is_none_or(|t| t.below_exit(s.value, self.state, self.cfg.exit_margin))
        });
        if self.state == PressureState::Green || at_floor || reserve_blocks || !all_clear {
            self.clear_since = None;
            return None;
        }
        let since = *self.clear_since.get_or_insert(now);
        if now.saturating_sub(since) < self.cfg.deescalate_dwell {
            return None;
        }
        let from = self.state;
        let (signal, value, threshold) = match dominant {
            Some(d) => {
                let exit = self
                    .thresholds
                    .get(&d.signal)
                    .and_then(|t| t.exit_threshold(from, self.cfg.exit_margin));
                (d.signal, d.value, exit.unwrap_or(d.value))
            }
            None => (gates.floor_signal, 0.0, 0.0),
        };
        let t = self.transition(from.lower(), signal, value, threshold, now);
        self.clear_since = Some(now);
        Some(t)
    }

    fn threshold_of(&self, signal: PressureSignal, level: PressureState) -> Option<f64> {
        self.thresholds
            .get(&signal)
            .and_then(|t| t.threshold(level))
    }

    fn transition(
        &mut self,
        to: PressureState,
        signal: PressureSignal,
        value: f64,
        threshold: f64,
        now: Duration,
    ) -> Transition {
        let t = Transition {
            at: now,
            at_wall: self.clock.now_wall(),
            from: self.state,
            to,
            signal,
            value,
            threshold,
        };
        self.state = to;
        self.since_wall = t.at_wall;
        self.up_count = 0;
        if self.history.len() == TRANSITION_HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(t);
        self.metrics
            .pressure_transitions
            .get_or_create(&TransitionLabels {
                from: t.from.as_str(),
                to: t.to.as_str(),
                signal: signal.as_str(),
            })
            .inc();
        self.publish_state();
        tracing::info!(
            event = "pressure_transition",
            from = t.from.as_str(),
            to = t.to.as_str(),
            signal = signal.as_str(),
            value = t.value,
            threshold = t.threshold,
        );
        t
    }

    fn publish_state(&self) {
        for s in PressureState::ALL {
            self.metrics
                .pressure_state
                .get_or_create(&StateLabel { state: s.as_str() })
                .set(i64::from(s == self.state));
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::signals::default_thresholds;
    use tracing::field::{Field, Visit};
    use tracing_subscriber::layer::{Context, SubscriberExt};
    use turbine_core::clock::FakeClock;

    pub(crate) const TICK: Duration = Duration::from_millis(100);

    pub(crate) fn machine(
        escalate: u32,
        dwell: Duration,
    ) -> (PressureMachine, FakeClock, ReliabilityMetrics) {
        let clock = FakeClock::new(Duration::ZERO);
        let metrics = ReliabilityMetrics::unregistered();
        let cfg = MachineConfig {
            enabled: true,
            escalate_samples: escalate,
            deescalate_dwell: dwell,
            exit_margin: 0.05,
        };
        (
            PressureMachine::new(
                cfg,
                default_thresholds(),
                metrics.clone(),
                Arc::new(clock.clone()),
            ),
            clock,
            metrics,
        )
    }

    pub(crate) fn sig(signal: PressureSignal, value: f64) -> SignalValue {
        SignalValue {
            signal,
            value,
            level: default_thresholds()[&signal].level(value),
            stale: false,
        }
    }

    /// Steps the fake clock by one tick and evaluates `signals`.
    pub(crate) fn tick(
        m: &mut PressureMachine,
        clock: &FakeClock,
        signals: &[SignalValue],
        gates: Gates,
    ) -> Option<Transition> {
        clock.advance(TICK);
        m.evaluate(signals, gates)
    }

    #[test]
    fn hysteresis_holds() {
        let (mut m, clock, _) = machine(2, Duration::from_secs(10));
        let mut transitions = Vec::new();
        // 60 s of kv_utilization alternating 0.83 / 0.81 every 500 ms around the 0.82 ORANGE threshold.
        for i in 0..600 {
            let v = if (i / 5) % 2 == 0 { 0.83 } else { 0.81 };
            transitions.extend(tick(
                &mut m,
                &clock,
                &[sig(PressureSignal::KvUtilization, v)],
                Gates::default(),
            ));
        }
        assert_eq!(
            transitions.len(),
            1,
            "exactly one transition: {transitions:?}"
        );
        assert_eq!(
            (transitions[0].from, transitions[0].to),
            (PressureState::Green, PressureState::Orange)
        );
        // Below 0.779 (0.82 − 5 %) the state holds for the whole dwell, then drops exactly one level.
        let start = clock.now_mono() + TICK;
        let mut down = None;
        for _ in 0..101 {
            if let Some(t) = tick(
                &mut m,
                &clock,
                &[sig(PressureSignal::KvUtilization, 0.77)],
                Gates::default(),
            ) {
                down = Some(t);
                break;
            }
        }
        let down = down.expect("de-escalates after the dwell");
        assert_eq!(
            (down.from, down.to),
            (PressureState::Orange, PressureState::Yellow)
        );
        assert_eq!(down.at - start, Duration::from_secs(10));
        // 0.78 is below the threshold but inside the exit margin: no de-escalation from ORANGE.
        let (mut m2, clock2, _) = machine(1, Duration::from_secs(10));
        tick(
            &mut m2,
            &clock2,
            &[sig(PressureSignal::KvUtilization, 0.83)],
            Gates::default(),
        );
        for _ in 0..300 {
            assert!(
                tick(
                    &mut m2,
                    &clock2,
                    &[sig(PressureSignal::KvUtilization, 0.78)],
                    Gates::default()
                )
                .is_none()
            );
        }
        assert_eq!(m2.state(), PressureState::Orange);
    }

    #[test]
    fn escalation_and_stepwise_deescalation() {
        let dwell = Duration::from_secs(10);
        // Allocation failure: GREEN → SURVIVAL in one sample.
        let (mut m, clock, _) = machine(2, dwell);
        let t = tick(
            &mut m,
            &clock,
            &[sig(PressureSignal::AllocationFailure, 1.0)],
            Gates::default(),
        )
        .expect("immediate");
        assert_eq!(
            (t.from, t.to),
            (PressureState::Green, PressureState::Survival)
        );
        // Stepwise descent, one level per dwell.
        let clear = [
            sig(PressureSignal::AllocationFailure, 0.0),
            sig(PressureSignal::KvUtilization, 0.1),
        ];
        let start = clock.now_mono();
        let mut steps = Vec::new();
        for _ in 0..450 {
            if let Some(t) = tick(&mut m, &clock, &clear, Gates::default()) {
                steps.push((t.from, t.to, t.at - start));
            }
        }
        let s = Duration::from_secs;
        assert_eq!(
            steps,
            vec![
                (PressureState::Survival, PressureState::Red, s(10) + TICK),
                (PressureState::Red, PressureState::Orange, s(20) + TICK),
                (PressureState::Orange, PressureState::Yellow, s(30) + TICK),
                (PressureState::Yellow, PressureState::Green, s(40) + TICK),
            ]
        );
        // Other signals need `escalate_samples` consecutive samples.
        let (mut m, clock, _) = machine(3, dwell);
        let red = [sig(PressureSignal::KvUtilization, 0.91)];
        assert!(tick(&mut m, &clock, &red, Gates::default()).is_none());
        assert!(tick(&mut m, &clock, &red, Gates::default()).is_none());
        assert!(
            tick(
                &mut m,
                &clock,
                &[sig(PressureSignal::KvUtilization, 0.5)],
                Gates::default()
            )
            .is_none(),
            "a GREEN sample resets the count"
        );
        assert!(tick(&mut m, &clock, &red, Gates::default()).is_none());
        assert!(tick(&mut m, &clock, &red, Gates::default()).is_none());
        assert_eq!(
            tick(&mut m, &clock, &red, Gates::default()).map(|t| t.to),
            Some(PressureState::Red)
        );
    }

    type Fields = BTreeMap<String, String>;
    thread_local! {
        static CAPTURED: std::cell::RefCell<Option<Vec<Fields>>> = const { std::cell::RefCell::new(None) };
    }
    struct FieldVisitor<'a>(&'a mut Fields);
    impl Visit for FieldVisitor<'_> {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.0.insert(
                field.name().to_string(),
                format!("{value:?}").trim_matches('"').to_string(),
            );
        }
        fn record_f64(&mut self, field: &Field, value: f64) {
            self.0.insert(field.name().to_string(), value.to_string());
        }
        fn record_str(&mut self, field: &Field, value: &str) {
            self.0.insert(field.name().to_string(), value.to_string());
        }
    }
    struct ThreadCapture;
    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for ThreadCapture {
        fn on_event(&self, event: &tracing::Event<'_>, _: Context<'_, S>) {
            CAPTURED.with(|c| {
                if let Some(events) = c.borrow_mut().as_mut() {
                    let mut fields = Fields::new();
                    event.record(&mut FieldVisitor(&mut fields));
                    events.push(fields);
                }
            });
        }
    }

    /// Runs `f` with every tracing event of this thread captured. A process-wide subscriber is installed once
    /// (scoped dispatchers race with callsite-interest caching when tests run in parallel).
    pub(crate) fn capture_events<R>(f: impl FnOnce() -> R) -> (R, Vec<Fields>) {
        static INSTALL: std::sync::Once = std::sync::Once::new();
        INSTALL.call_once(|| {
            let _ = tracing::subscriber::set_global_default(
                tracing_subscriber::registry().with(ThreadCapture),
            );
        });
        // Callsites first hit by a parallel test before the install cached "never"; recompute them.
        tracing::callsite::rebuild_interest_cache();
        CAPTURED.with(|c| *c.borrow_mut() = Some(Vec::new()));
        let r = f();
        let events = CAPTURED.with(|c| c.borrow_mut().take()).unwrap_or_default();
        (r, events)
    }

    #[test]
    fn transitions_explained() {
        let ((m, metrics, n), events) = capture_events(|| {
            let (mut m, clock, metrics) = machine(1, TICK);
            let mut n = 0;
            for i in 0..400 {
                let v = if (i / 10) % 2 == 0 { 0.91 } else { 0.10 };
                n += tick(
                    &mut m,
                    &clock,
                    &[sig(PressureSignal::KvUtilization, v)],
                    Gates::default(),
                )
                .is_some() as usize;
            }
            (m, metrics, n)
        });
        assert!(
            n > TRANSITION_HISTORY,
            "the scenario produces more than 32 transitions ({n})"
        );
        assert_eq!(
            m.history().count(),
            TRANSITION_HISTORY,
            "history is capped at 32"
        );
        let counted: u64 = [
            (PressureState::Green, PressureState::Red),
            (PressureState::Red, PressureState::Orange),
            (PressureState::Orange, PressureState::Yellow),
            (PressureState::Yellow, PressureState::Green),
            (PressureState::Orange, PressureState::Red),
            (PressureState::Yellow, PressureState::Red),
        ]
        .iter()
        .map(|(f, t)| {
            metrics
                .pressure_transitions
                .get_or_create(&TransitionLabels {
                    from: f.as_str(),
                    to: t.as_str(),
                    signal: "kv_utilization",
                })
                .get()
        })
        .sum();
        assert_eq!(
            counted as usize, n,
            "one turbine_pressure_transitions_total increment per transition"
        );
        let logs: Vec<_> = events
            .iter()
            .filter(|f| f.get("event").map(String::as_str) == Some("pressure_transition"))
            .cloned()
            .collect();
        assert_eq!(
            logs.len(),
            n,
            "one pressure_transition log event per transition"
        );
        for (log, t) in logs.iter().rev().zip(m.history().rev()) {
            assert_eq!(log["signal"], t.signal.as_str());
            assert_eq!(log["from"], t.from.as_str());
            assert_eq!(log["to"], t.to.as_str());
            assert_eq!(log["value"].parse::<f64>().unwrap(), t.value);
            assert_eq!(log["threshold"].parse::<f64>().unwrap(), t.threshold);
        }
        for t in m.history() {
            assert!(
                t.threshold > 0.0,
                "every transition names the threshold it crossed: {t:?}"
            );
        }
    }

    #[test]
    fn gates_and_disabled() {
        // The circuit floor raises the state with its signal and holds it there.
        let (mut m, clock, _) = machine(1, TICK);
        let floor = Gates {
            floor: PressureState::Yellow,
            floor_signal: PressureSignal::TelemetryStale,
            reserve_held: true,
        };
        let calm = [sig(PressureSignal::KvUtilization, 0.1)];
        let t = tick(&mut m, &clock, &calm, floor).expect("floor raises the state");
        assert_eq!(
            (t.to, t.signal, t.value, t.threshold),
            (
                PressureState::Yellow,
                PressureSignal::TelemetryStale,
                1.0,
                1.0
            )
        );
        for _ in 0..50 {
            assert!(
                tick(&mut m, &clock, &calm, floor).is_none(),
                "never below the floor"
            );
        }
        tick(&mut m, &clock, &calm, Gates::default());
        assert_eq!(
            tick(&mut m, &clock, &calm, Gates::default()).map(|t| t.to),
            Some(PressureState::Green),
            "descends once the floor lifts"
        );
        // Stale signals are ignored.
        let mut stale = sig(PressureSignal::KvUtilization, 0.99);
        stale.stale = true;
        assert!(tick(&mut m, &clock, &[stale], Gates::default()).is_none());

        // enabled: false keeps GREEN whatever the signals say.
        let clock = FakeClock::new(Duration::ZERO);
        let metrics = ReliabilityMetrics::unregistered();
        let cfg = MachineConfig::from_config(&turbine_core::config::ReliabilityConfig {
            enabled: false,
            ..Default::default()
        });
        assert_eq!(cfg.escalate_samples, 2);
        let mut off = PressureMachine::new(
            cfg,
            default_thresholds(),
            metrics.clone(),
            Arc::new(clock.clone()),
        );
        for _ in 0..10 {
            assert!(
                tick(
                    &mut off,
                    &clock,
                    &[sig(PressureSignal::AllocationFailure, 1.0)],
                    Gates::default()
                )
                .is_none()
            );
        }
        assert_eq!(off.state(), PressureState::Green);
        let green = metrics
            .pressure_state
            .get_or_create(&StateLabel { state: "GREEN" })
            .get();
        assert_eq!(green, 1);
    }
}
