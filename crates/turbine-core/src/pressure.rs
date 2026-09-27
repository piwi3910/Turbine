//! Pressure and circuit vocabulary (contract §3.4, P3 S-6/S-7/S-12), re-exported from
//! `turbine_core::types`. Lives in core because `reliability.pressure.thresholds.<signal>`
//! keys are parsed here and every later phase reads the states.

use serde::{Deserialize, Serialize};

/// TS §9 pressure state. `as_u8()` is 0..=4 for gauges; serialised `"GREEN"`..`"SURVIVAL"`
/// everywhere (CONFLICT C-5). Ordered by severity.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PressureState {
    Green,
    Yellow,
    Orange,
    Red,
    Survival,
}

impl PressureState {
    pub const ALL: [PressureState; 5] = [
        PressureState::Green,
        PressureState::Yellow,
        PressureState::Orange,
        PressureState::Red,
        PressureState::Survival,
    ];

    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Metric label / log value.
    pub fn as_str(self) -> &'static str {
        match self {
            PressureState::Green => "GREEN",
            PressureState::Yellow => "YELLOW",
            PressureState::Orange => "ORANGE",
            PressureState::Red => "RED",
            PressureState::Survival => "SURVIVAL",
        }
    }

    /// Level 0..=4 → state; values above 4 saturate at SURVIVAL.
    pub fn from_level(level: u8) -> PressureState {
        PressureState::ALL[usize::from(level.min(4))]
    }

    /// One level lower, saturating at GREEN.
    pub fn lower(self) -> PressureState {
        PressureState::from_level(self.as_u8().saturating_sub(1))
    }
}

/// TS §9 circuit-breaker state; serialised `"HEALTHY"`, `"CIRCUIT_OPEN"`, … (CONFLICT C-5).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CircuitState {
    Healthy,
    Degraded,
    CircuitOpen,
    Draining,
    Probing,
}

impl CircuitState {
    pub const ALL: [CircuitState; 5] = [
        CircuitState::Healthy,
        CircuitState::Degraded,
        CircuitState::CircuitOpen,
        CircuitState::Draining,
        CircuitState::Probing,
    ];

    /// Metric label / log value.
    pub fn as_str(self) -> &'static str {
        match self {
            CircuitState::Healthy => "HEALTHY",
            CircuitState::Degraded => "DEGRADED",
            CircuitState::CircuitOpen => "CIRCUIT_OPEN",
            CircuitState::Draining => "DRAINING",
            CircuitState::Probing => "PROBING",
        }
    }

    /// `/ready` answers 503 `circuit_open` in these states (P3 S-12).
    pub fn blocks_readiness(self) -> bool {
        matches!(
            self,
            CircuitState::CircuitOpen | CircuitState::Draining | CircuitState::Probing
        )
    }
}

/// P3 S-6 pressure signals (the signal table); snake_case in config keys, metrics and JSON.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PressureSignal {
    KvUtilization,
    DeviceMemory,
    HostAvailable,
    PsiMemorySomeAvg10,
    SwapInRate,
    ExhaustionHorizon,
    QueueFill,
    StepTimeDrift,
    Thermal,
    TelemetryStale,
    AllocationFailure,
}

impl PressureSignal {
    pub const ALL: [PressureSignal; 11] = [
        PressureSignal::KvUtilization,
        PressureSignal::DeviceMemory,
        PressureSignal::HostAvailable,
        PressureSignal::PsiMemorySomeAvg10,
        PressureSignal::SwapInRate,
        PressureSignal::ExhaustionHorizon,
        PressureSignal::QueueFill,
        PressureSignal::StepTimeDrift,
        PressureSignal::Thermal,
        PressureSignal::TelemetryStale,
        PressureSignal::AllocationFailure,
    ];

    /// Metric label / config key / JSON name.
    pub fn as_str(self) -> &'static str {
        match self {
            PressureSignal::KvUtilization => "kv_utilization",
            PressureSignal::DeviceMemory => "device_memory",
            PressureSignal::HostAvailable => "host_available",
            PressureSignal::PsiMemorySomeAvg10 => "psi_memory_some_avg10",
            PressureSignal::SwapInRate => "swap_in_rate",
            PressureSignal::ExhaustionHorizon => "exhaustion_horizon",
            PressureSignal::QueueFill => "queue_fill",
            PressureSignal::StepTimeDrift => "step_time_drift",
            PressureSignal::Thermal => "thermal",
            PressureSignal::TelemetryStale => "telemetry_stale",
            PressureSignal::AllocationFailure => "allocation_failure",
        }
    }

    /// "Lower is worse" signals take descending thresholds (P3 signal table).
    pub fn lower_is_worse(self) -> bool {
        matches!(
            self,
            PressureSignal::HostAvailable | PressureSignal::ExhaustionHorizon
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_helpers_and_serde_names() {
        let names: Vec<&str> = PressureState::ALL.iter().map(|s| s.as_str()).collect();
        assert_eq!(names, ["GREEN", "YELLOW", "ORANGE", "RED", "SURVIVAL"]);
        for (i, s) in PressureState::ALL.iter().enumerate() {
            assert_eq!(usize::from(s.as_u8()), i);
            assert_eq!(PressureState::from_level(s.as_u8()), *s);
            assert_eq!(
                serde_json::to_string(s).unwrap(),
                format!("\"{}\"", s.as_str())
            );
        }
        assert_eq!(PressureState::from_level(9), PressureState::Survival);
        assert_eq!(PressureState::Green.lower(), PressureState::Green);
        assert_eq!(PressureState::Red.lower(), PressureState::Orange);
        assert!(PressureState::Orange > PressureState::Yellow);
        for c in CircuitState::ALL {
            assert_eq!(
                serde_json::to_string(&c).unwrap(),
                format!("\"{}\"", c.as_str())
            );
        }
        let blocking: Vec<CircuitState> = CircuitState::ALL
            .into_iter()
            .filter(|c| c.blocks_readiness())
            .collect();
        assert_eq!(
            blocking,
            [
                CircuitState::CircuitOpen,
                CircuitState::Draining,
                CircuitState::Probing
            ]
        );
        for s in PressureSignal::ALL {
            assert_eq!(
                serde_json::to_string(&s).unwrap(),
                format!("\"{}\"", s.as_str())
            );
        }
        let lower: Vec<PressureSignal> = PressureSignal::ALL
            .into_iter()
            .filter(|s| s.lower_is_worse())
            .collect();
        assert_eq!(
            lower,
            [
                PressureSignal::HostAvailable,
                PressureSignal::ExhaustionHorizon
            ]
        );
    }
}
