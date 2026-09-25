//! Pressure signals and their documented default thresholds (P3 S-6, contract §8.3).

use std::collections::BTreeMap;
use std::time::Duration;

use serde::Serialize;
use turbine_core::config::PressureConfig;
use turbine_core::telemetry::{SourceStatus, TelemetrySample};
use turbine_core::types::{DeviceId, MemoryKind, PressureState};

pub use turbine_core::types::PressureSignal;

/// One signal's thresholds for YELLOW, ORANGE, RED, SURVIVAL.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SignalThresholds {
    /// `None` = the signal never reaches that level.
    pub levels: [Option<f64>; 4],
    pub lower_is_worse: bool,
}

impl SignalThresholds {
    fn reaches(&self, value: f64, threshold: f64) -> bool {
        if self.lower_is_worse {
            value <= threshold
        } else {
            value >= threshold
        }
    }

    /// Highest level whose threshold `value` reaches (GREEN below YELLOW; NaN is GREEN).
    pub fn level(&self, value: f64) -> PressureState {
        let mut level = PressureState::Green;
        for (i, t) in (1u8..).zip(self.levels) {
            if let Some(t) = t
                && self.reaches(value, t)
            {
                level = PressureState::from_level(i);
            }
        }
        level
    }

    /// The threshold that enters `level` (`None` for GREEN or an unused level).
    pub fn threshold(&self, level: PressureState) -> Option<f64> {
        match level.as_u8() {
            0 => None,
            l => self.levels[usize::from(l) - 1],
        }
    }

    /// True when `value` is clear of `level`'s threshold by `margin` (relative to the
    /// threshold), or the signal has no threshold at that level.
    pub fn below_exit(&self, value: f64, level: PressureState, margin: f64) -> bool {
        match self.exit_threshold(level, margin) {
            None => true,
            Some(t) if self.lower_is_worse => value > t,
            Some(t) => value < t,
        }
    }

    /// The value `level` must be left by to count toward de-escalation.
    pub fn exit_threshold(&self, level: PressureState, margin: f64) -> Option<f64> {
        self.threshold(level).map(|t| {
            if self.lower_is_worse {
                t * (1.0 + margin)
            } else {
                t * (1.0 - margin)
            }
        })
    }
}

/// The P3 signal table, verbatim.
pub fn default_thresholds() -> BTreeMap<PressureSignal, SignalThresholds> {
    use PressureSignal::*;
    let up = |levels: [Option<f64>; 4]| SignalThresholds {
        levels,
        lower_is_worse: false,
    };
    let down = |levels: [Option<f64>; 4]| SignalThresholds {
        levels,
        lower_is_worse: true,
    };
    BTreeMap::from([
        (
            KvUtilization,
            up([Some(0.70), Some(0.82), Some(0.90), Some(0.97)]),
        ),
        (
            DeviceMemory,
            up([Some(0.85), Some(0.90), Some(0.95), Some(0.98)]),
        ),
        (
            HostAvailable,
            down([Some(4.0), Some(2.0), Some(1.0), Some(0.5)]),
        ),
        (
            PsiMemorySomeAvg10,
            up([Some(5.0), Some(10.0), Some(25.0), Some(50.0)]),
        ),
        (
            SwapInRate,
            up([Some(1.0), Some(100.0), Some(1000.0), Some(10000.0)]),
        ),
        (
            ExhaustionHorizon,
            down([Some(60.0), Some(20.0), Some(5.0), Some(1.0)]),
        ),
        (QueueFill, up([Some(0.50), Some(0.80), Some(0.95), None])),
        (StepTimeDrift, up([Some(1.5), Some(2.0), Some(3.0), None])),
        (Thermal, up([Some(1.0), Some(2.0), None, None])),
        (TelemetryStale, up([Some(1.0), None, None, None])),
        (AllocationFailure, up([None, None, None, Some(1.0)])),
    ])
}

/// Defaults overlaid with the `reliability.pressure.thresholds` overrides (validated at startup).
pub fn effective_thresholds(cfg: &PressureConfig) -> BTreeMap<PressureSignal, SignalThresholds> {
    let mut t = default_thresholds();
    for (signal, levels) in &cfg.thresholds {
        t.insert(
            *signal,
            SignalThresholds {
                levels: *levels,
                lower_is_worse: signal.lower_is_worse(),
            },
        );
    }
    t
}

/// Signals computed for a device of this memory kind: `device_memory` only on dedicated devices.
pub fn active_signals(kind: MemoryKind) -> Vec<PressureSignal> {
    PressureSignal::ALL
        .into_iter()
        .filter(|s| kind == MemoryKind::Dedicated || *s != PressureSignal::DeviceMemory)
        .collect()
}

/// One evaluated signal (a `signals` entry of the pressure document).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SignalValue {
    #[serde(rename = "name")]
    pub signal: PressureSignal,
    pub value: f64,
    pub level: PressureState,
    pub stale: bool,
}

/// Everything the evaluator needs besides the telemetry sample.
#[derive(Clone, Debug)]
pub struct SignalInputs<'a> {
    pub sample: &'a TelemetrySample,
    pub host_reserve_bytes: u64,
    /// (device, memory kind, budgeted bytes): `device_memory` = used / budget on dedicated devices.
    pub devices: &'a [(DeviceId, MemoryKind, u64)],
    pub exhaustion_horizon_seconds: f64,
    /// Windowed p95 decode step time / GREEN baseline; `None` until a baseline exists.
    pub step_time_drift: Option<f64>,
    /// A device OOM within the last `deescalate_dwell`.
    pub allocation_failure_recent: bool,
}

/// A telemetry source tracked for staleness: the host `/proc` reader or one device.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Source {
    Host,
    Device(DeviceId),
}

/// Converts samples into signal values; keeps the state rates and staleness need.
#[derive(Debug)]
pub struct SignalEvaluator {
    thresholds: BTreeMap<PressureSignal, SignalThresholds>,
    interval: Duration,
    stale_after: Duration,
    last_pswpin: Option<(u64, Duration)>,
    stale_since: BTreeMap<Source, Duration>,
}

impl SignalEvaluator {
    pub fn new(
        thresholds: BTreeMap<PressureSignal, SignalThresholds>,
        interval: Duration,
        stale_after: Duration,
    ) -> Self {
        SignalEvaluator {
            thresholds,
            interval,
            stale_after,
            last_pswpin: None,
            stale_since: BTreeMap::new(),
        }
    }

    pub fn thresholds(&self) -> &BTreeMap<PressureSignal, SignalThresholds> {
        &self.thresholds
    }

    fn value(&self, signal: PressureSignal, value: f64, stale: bool) -> SignalValue {
        let level = self
            .thresholds
            .get(&signal)
            .map_or(PressureState::Green, |t| t.level(value));
        SignalValue {
            signal,
            value,
            level,
            stale,
        }
    }

    /// True once `source` has been stale for at least `stale_after`.
    fn stale_too_long(&mut self, source: Source, status: SourceStatus, now: Duration) -> bool {
        if status == SourceStatus::Stale {
            let since = *self.stale_since.entry(source).or_insert(now);
            now.saturating_sub(since) >= self.stale_after
        } else {
            self.stale_since.remove(&source);
            false
        }
    }

    /// Signals of `Unavailable` sources are omitted; those of `Stale` sources are flagged.
    pub fn evaluate(&mut self, inp: &SignalInputs<'_>, now: Duration) -> Vec<SignalValue> {
        use PressureSignal::*;
        let s = inp.sample;
        let mut out = vec![
            self.value(KvUtilization, s.ledger.kv_utilization, false),
            self.value(QueueFill, s.ledger.queue_fill, false),
            self.value(ExhaustionHorizon, inp.exhaustion_horizon_seconds, false),
        ];
        if let Some(drift) = inp.step_time_drift {
            out.push(self.value(StepTimeDrift, drift, false));
        }
        let oom = if inp.allocation_failure_recent {
            1.0
        } else {
            0.0
        };
        out.push(self.value(AllocationFailure, oom, false));

        let mut any_stale_long = self.stale_too_long(Source::Host, s.host.status, now);
        if s.host.status != SourceStatus::Unavailable {
            let stale = s.host.status == SourceStatus::Stale;
            if let Some(avail) = s.host.mem_available_bytes
                && inp.host_reserve_bytes > 0
            {
                let multiple = avail as f64 / inp.host_reserve_bytes as f64;
                out.push(self.value(HostAvailable, multiple, stale));
            }
            if let Some(psi) = s.host.psi_memory_some_avg10 {
                out.push(self.value(PsiMemorySomeAvg10, psi, stale));
            }
            if let Some(pswpin) = s.host.pswpin_total {
                // A tick gap above 10 × interval (suspend, clock jump) discards this tick's rate.
                if let Some((prev, at)) = self.last_pswpin
                    && now > at
                    && now - at <= self.interval * 10
                {
                    let rate = pswpin.saturating_sub(prev) as f64 / (now - at).as_secs_f64();
                    out.push(self.value(SwapInRate, rate, stale));
                }
                self.last_pswpin = Some((pswpin, now));
            }
        }

        // Worst value over devices for the per-device signals.
        let mut device_memory: Option<(f64, bool)> = None;
        let mut thermal: Option<(f64, bool)> = None;
        for d in &s.devices {
            any_stale_long |= self.stale_too_long(Source::Device(d.device), d.status, now);
            if d.status == SourceStatus::Unavailable {
                continue;
            }
            let stale = d.status == SourceStatus::Stale;
            let budget = inp.devices.iter().find(|(id, _, _)| *id == d.device);
            if let (Some((_, MemoryKind::Dedicated, budget)), Some(used)) =
                (budget, d.memory_used_bytes)
                && *budget > 0
            {
                let v = used as f64 / *budget as f64;
                if device_memory.is_none_or(|(m, _)| v > m) {
                    device_memory = Some((v, stale));
                }
            }
            if d.temperature_c.is_some() || d.throttle.thermal {
                let t = if d.throttle.thermal {
                    2.0
                } else {
                    match (d.temperature_c, d.slowdown_temperature_c) {
                        (Some(temp), Some(slow)) if temp >= slow - 5.0 => 1.0,
                        _ => 0.0,
                    }
                };
                if thermal.is_none_or(|(m, _)| t > m) {
                    thermal = Some((t, stale));
                }
            }
        }
        if let Some((v, stale)) = device_memory {
            out.push(self.value(DeviceMemory, v, stale));
        }
        if let Some((v, stale)) = thermal {
            out.push(self.value(Thermal, v, stale));
        }
        let stale_value = if any_stale_long { 1.0 } else { 0.0 };
        out.push(self.value(TelemetryStale, stale_value, false));
        out
    }
}

#[cfg(test)]
mod tests {
    use turbine_core::telemetry::{DeviceSample, HostSample, LedgerSample, ThrottleReasons};

    use super::*;

    const GIB: u64 = 1 << 30;

    fn find(values: &[SignalValue], s: PressureSignal) -> Option<SignalValue> {
        values.iter().copied().find(|v| v.signal == s)
    }

    #[test]
    fn levels_follow_the_table() {
        let t = default_thresholds();
        assert_eq!(t.len(), PressureSignal::ALL.len());
        let kv = &t[&PressureSignal::KvUtilization];
        assert_eq!(kv.level(0.69), PressureState::Green);
        assert_eq!(kv.level(0.70), PressureState::Yellow);
        assert_eq!(kv.level(0.83), PressureState::Orange);
        assert_eq!(kv.level(0.97), PressureState::Survival);
        assert_eq!(kv.level(f64::NAN), PressureState::Green);
        let host = &t[&PressureSignal::HostAvailable];
        assert_eq!(host.level(5.0), PressureState::Green);
        assert_eq!(host.level(1.5), PressureState::Orange);
        assert_eq!(host.level(0.4), PressureState::Survival);
        let orange_exit = kv.exit_threshold(PressureState::Orange, 0.05).unwrap();
        assert!((orange_exit - 0.779).abs() < 1e-12);
        assert!(kv.below_exit(0.77, PressureState::Orange, 0.05));
        assert!(!kv.below_exit(0.78, PressureState::Orange, 0.05));
        assert!(host.below_exit(2.2, PressureState::Orange, 0.05));
        assert!(!host.below_exit(2.05, PressureState::Orange, 0.05));
        let queue = &t[&PressureSignal::QueueFill];
        assert_eq!(queue.level(1.0), PressureState::Red, "never SURVIVAL");
        assert!(queue.below_exit(1.0, PressureState::Survival, 0.05));
        assert_eq!(
            t[&PressureSignal::AllocationFailure].level(1.0),
            PressureState::Survival
        );
        assert_eq!(
            t[&PressureSignal::ExhaustionHorizon].level(f64::INFINITY),
            PressureState::Green
        );

        let mut cfg = PressureConfig::default();
        cfg.thresholds.insert(
            PressureSignal::KvUtilization,
            [Some(0.6), Some(0.8), Some(0.9), None],
        );
        let eff = effective_thresholds(&cfg);
        assert_eq!(
            eff[&PressureSignal::KvUtilization].level(0.99),
            PressureState::Red
        );
        assert_eq!(
            eff[&PressureSignal::DeviceMemory],
            t[&PressureSignal::DeviceMemory]
        );
    }

    #[test]
    fn evaluator_rates_staleness_and_devices() {
        let interval = Duration::from_millis(100);
        let mut ev = SignalEvaluator::new(default_thresholds(), interval, Duration::from_secs(5));
        let devices = [(DeviceId(0), MemoryKind::Dedicated, 10 * GIB)];
        let mut sample = TelemetrySample {
            at_mono_ns: 0,
            host: HostSample {
                mem_available_bytes: Some(12 * GIB),
                pswpin_total: Some(1000),
                psi_memory_some_avg10: Some(12.0),
                ..HostSample::default()
            },
            devices: vec![DeviceSample {
                memory_used_bytes: Some(9 * GIB + GIB / 5),
                temperature_c: Some(86.0),
                slowdown_temperature_c: Some(90.0),
                ..DeviceSample::empty(DeviceId(0), SourceStatus::Ok)
            }],
            ledger: LedgerSample {
                kv_utilization: 0.5,
                queue_fill: 0.9,
            },
            storage: None,
        };
        let inputs = |s: &TelemetrySample| {
            let s = s.clone();
            move |ev: &mut SignalEvaluator, now: Duration| {
                ev.evaluate(
                    &SignalInputs {
                        sample: &s,
                        host_reserve_bytes: 8 * GIB,
                        devices: &devices,
                        exhaustion_horizon_seconds: f64::INFINITY,
                        step_time_drift: None,
                        allocation_failure_recent: false,
                    },
                    now,
                )
            }
        };
        let first = inputs(&sample)(&mut ev, Duration::ZERO);
        assert!(
            find(&first, PressureSignal::SwapInRate).is_none(),
            "no rate without a delta"
        );
        let host = find(&first, PressureSignal::HostAvailable).unwrap();
        assert_eq!((host.value, host.level), (1.5, PressureState::Orange));
        assert_eq!(
            find(&first, PressureSignal::PsiMemorySomeAvg10)
                .unwrap()
                .level,
            PressureState::Orange
        );
        assert_eq!(
            find(&first, PressureSignal::QueueFill).unwrap().level,
            PressureState::Orange
        );
        let mem = find(&first, PressureSignal::DeviceMemory).unwrap();
        assert!((mem.value - 0.92).abs() < 1e-9 && mem.level == PressureState::Orange);
        assert_eq!(find(&first, PressureSignal::Thermal).unwrap().value, 1.0);
        assert!(find(&first, PressureSignal::StepTimeDrift).is_none());

        // 200 pages in 100 ms = 2000 pages/s (RED).
        sample.host.pswpin_total = Some(1200);
        let second = inputs(&sample)(&mut ev, interval);
        let swap = find(&second, PressureSignal::SwapInRate).unwrap();
        assert!((swap.value - 2000.0).abs() < 1e-6 && swap.level == PressureState::Red);
        // A gap over 10 × interval discards the rate.
        sample.host.pswpin_total = Some(5000);
        let gap = inputs(&sample)(&mut ev, interval + interval * 11);
        assert!(find(&gap, PressureSignal::SwapInRate).is_none());

        // Thermal throttle reason = 2; a stale device is flagged, then telemetry_stale after 5 s.
        sample.devices[0].throttle = ThrottleReasons {
            thermal: true,
            ..ThrottleReasons::default()
        };
        sample.devices[0].status = SourceStatus::Stale;
        let t0 = Duration::from_secs(10);
        let stale = inputs(&sample)(&mut ev, t0);
        let thermal = find(&stale, PressureSignal::Thermal).unwrap();
        assert_eq!((thermal.value, thermal.stale), (2.0, true));
        assert_eq!(
            find(&stale, PressureSignal::TelemetryStale).unwrap().value,
            0.0
        );
        let later = inputs(&sample)(&mut ev, t0 + Duration::from_secs(5));
        let ts = find(&later, PressureSignal::TelemetryStale).unwrap();
        assert_eq!((ts.value, ts.level), (1.0, PressureState::Yellow));

        // An unavailable source contributes no signals (and never counts as stale).
        sample.devices[0].status = SourceStatus::Unavailable;
        sample.host.status = SourceStatus::Unavailable;
        let none = inputs(&sample)(&mut ev, t0 + Duration::from_secs(6));
        for s in [
            PressureSignal::DeviceMemory,
            PressureSignal::Thermal,
            PressureSignal::HostAvailable,
            PressureSignal::PsiMemorySomeAvg10,
            PressureSignal::SwapInRate,
        ] {
            assert!(
                find(&none, s).is_none(),
                "{s:?} emitted for an unavailable source"
            );
        }
        assert_eq!(
            find(&none, PressureSignal::TelemetryStale).unwrap().value,
            0.0
        );
    }
}
