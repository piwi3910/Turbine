//! `turbine-bench`: streaming load generator for any OpenAI-compatible endpoint (TS §18).

pub mod args;
pub mod client;
pub mod golden;
pub mod kv_sim;
pub mod multi_turn;
pub mod open_loop;
pub mod prompt;
pub mod report;

pub use args::{BenchArgs, EndpointArg, OutputFormat, Profile, ThinkTime};
pub use client::{BenchError, run};
pub use open_loop::{Breakdown, RangeArg};
pub use report::{Percentiles, Report};
