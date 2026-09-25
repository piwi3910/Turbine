//! Memory budget, reservation ledger, pressure control, admission, recovery and circuit
//! breaking (P3). No GPU, FFI or `unsafe`: every time-dependent component takes a
//! [`Clock`] so tests drive time deterministically (P3 S-1).

pub mod budget;
pub mod metrics;
pub mod signals;

pub use turbine_core::clock::Clock;
pub use turbine_core::types::{CircuitState, PressureState};
