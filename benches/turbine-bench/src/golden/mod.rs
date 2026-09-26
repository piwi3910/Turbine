//! `turbine-golden`: compare an OpenAI-compatible endpoint with golden fixtures, or capture
//! fixtures from one (P1 S-11).

pub mod client;
pub mod compare;
pub mod fixture;

pub use client::{Endpoint, Generation, capture, compare};
pub use compare::{CompareReport, MissingTopK, PromptVerdict, compare_prompt, judge};
pub use fixture::{
    GoldenError, LogprobBounds, PromptKind, PromptRecord, ReferenceRecord, Tolerance,
};
