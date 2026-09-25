//! Traits through which the API reaches the engine and diagnostics (implemented by turbine-server).

use std::sync::Arc;

use serde::Serialize;
use turbine_observability::MetricsRegistry;

use crate::error::ApiError;

/// `GET /v1/models` entry (empty list in Phase 0).
#[derive(Clone, Debug, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    pub max_model_len: u32,
}

/// Inference entry point. Phase 0 exposes the model list only; P1 adds `submit`.
pub trait InferenceBackend: Send + Sync {
    fn models(&self) -> Vec<ModelCard>;
}

/// `/turbine/v1/*` documents. Each returns the JSON document or `ApiError::not_implemented()`.
pub trait Diagnostics: Send + Sync {
    fn status(&self) -> serde_json::Value;
    fn devices(&self) -> serde_json::Value;
    fn scheduler(&self) -> Result<serde_json::Value, ApiError>;
    fn kv(&self) -> Result<serde_json::Value, ApiError>;
    fn pressure(&self) -> Result<serde_json::Value, ApiError>;
}

/// Source of the `/ready` answer.
pub trait Readiness: Send + Sync {
    fn ready(&self) -> ReadyState;
}

/// `/ready`: 200 `{"ready":true}` or 503 `{"ready":false,"reason":"<r>"}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadyState {
    Ready,
    NotReady { reason: NotReadyReason },
}

/// `/ready` 503 reasons. Later phases add variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum NotReadyReason {
    NoModelLoaded,
}

impl NotReadyReason {
    /// The `reason` string in the `/ready` body.
    pub fn as_str(self) -> &'static str {
        match self {
            NotReadyReason::NoModelLoaded => "no_model_loaded",
        }
    }
}

/// Request limits enforced by the router.
#[derive(Clone, Copy, Debug)]
pub struct ApiLimits {
    /// `server.max_request_bytes`; larger bodies get 413 `request_too_large`.
    pub max_request_bytes: usize,
}

/// Everything the router needs; cloned per request by Axum.
#[derive(Clone)]
pub struct ApiState {
    pub inference: Arc<dyn InferenceBackend>,
    pub diagnostics: Arc<dyn Diagnostics>,
    pub readiness: Arc<dyn Readiness>,
    pub metrics: MetricsRegistry,
    pub limits: ApiLimits,
}
