//! Axum router for the OpenAI-compatible API and the `/turbine/v1` diagnostics (contract §14).
//! The API talks to the engine only through the traits in [`backend`].

pub mod backend;
pub mod error;
pub mod openai;
mod routes;

pub use backend::{
    ApiLimits, ApiState, BoxFuture, Diagnostics, GenerationStream, InferenceBackend,
    InferenceRequest, ModelCard, NotReadyReason, Readiness, ReadyState,
};
pub use error::{ApiError, ErrorType};
pub use routes::router;
pub use turbine_core::request::ErrorCode;
