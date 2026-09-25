//! `turbine-bench`: streaming load generator for any OpenAI-compatible endpoint (TS §18).

pub mod args;
pub mod client;
pub mod golden;
pub mod prompt;
pub mod report;

pub use args::{BenchArgs, EndpointArg, OutputFormat};
pub use client::{BenchError, run};
pub use report::{Percentiles, Report};
