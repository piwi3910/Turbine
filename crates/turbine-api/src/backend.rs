//! Traits through which the API reaches the engine and diagnostics (implemented by turbine-server).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::Serialize;
use turbine_core::request::{Endpoint, ErrorCode, GenerationEvent};
use turbine_core::types::{CircuitState, RequestId};
use turbine_observability::MetricsRegistry;

use crate::error::ApiError;
use crate::kv::{PrefetchAccepted, PrefetchRequest, TurbineHeaders};
use crate::openai::request::OpenAiRequest;

/// A boxed, sendable future (keeps [`InferenceBackend`] object-safe).
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Per-request event stream from the engine (bounded: 64 events in Phase 1, 256 in Phase 2). Dropping the
/// receiver is the cancellation signal: the engine stops generating when its send fails.
pub type GenerationStream = tokio::sync::mpsc::Receiver<GenerationEvent>;

/// A validated OpenAI request handed to the engine.
#[derive(Clone, Debug)]
pub struct InferenceRequest {
    /// Created by the API; rendered as `cmpl-<uuid>` / `chatcmpl-<uuid>` in responses.
    pub id: RequestId,
    pub endpoint: Endpoint,
    pub body: OpenAiRequest,
    /// `x-request-id` of the HTTP request: a log field, never a metric label.
    pub http_request_id: String,
    /// Validated `x-turbine-*` headers (Phase 4); the session id is `body.prompt_cache_key`.
    pub hints: TurbineHeaders,
}

/// `GET /v1/models` entry (empty list in Phase 0).
#[derive(Clone, Debug, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    pub max_model_len: u32,
}

/// Inference entry point: the model list and request submission.
pub trait InferenceBackend: Send + Sync {
    fn models(&self) -> Vec<ModelCard>;

    /// Start a generation. An `Err` is returned before any event (templating, context length,
    /// busy slot) and becomes a plain HTTP error even for streaming requests. The default is
    /// the no-model answer, 503 `model_not_loaded`.
    fn submit(&self, req: InferenceRequest) -> BoxFuture<'_, Result<GenerationStream, ApiError>> {
        drop(req);
        Box::pin(async { Err(ApiError::model_not_loaded()) })
    }

    /// Display text of one token for logprob entries; the default is the id form `token_id:<id>`.
    fn token_text(&self, token_id: u32) -> String {
        format!("token_id:{token_id}")
    }

    /// Count a request the API rejected before `submit`
    /// (`turbine_requests_total{outcome="rejected"}`). The default records nothing.
    fn record_rejection(&self, endpoint: Endpoint, code: ErrorCode) {
        let _ = (endpoint, code);
    }

    /// `POST /turbine/v1/kv/prefetch` (Phase 4): queue promotions of a session's or a
    /// prompt's cached blocks. The default, for a build without the KV hierarchy, is 501
    /// `not_implemented`.
    fn prefetch(&self, req: PrefetchRequest) -> BoxFuture<'_, Result<PrefetchAccepted, ApiError>> {
        drop(req);
        Box::pin(async { Err(ApiError::not_implemented()) })
    }
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
    /// Startup is loading weights or warming up.
    LoadingModel,
    ModelLoadFailed,
    DeviceError,
    /// SIGINT/SIGTERM received: draining for up to `server.shutdown_grace` (Phase 2).
    ShuttingDown,
    /// The circuit breaker is CIRCUIT_OPEN, DRAINING or PROBING (Phase 3).
    CircuitOpen,
}

impl NotReadyReason {
    /// The `reason` string in the `/ready` body.
    pub fn as_str(self) -> &'static str {
        match self {
            NotReadyReason::NoModelLoaded => "no_model_loaded",
            NotReadyReason::LoadingModel => "loading_model",
            NotReadyReason::ModelLoadFailed => "model_load_failed",
            NotReadyReason::DeviceError => "device_error",
            NotReadyReason::ShuttingDown => "shutting_down",
            NotReadyReason::CircuitOpen => "circuit_open",
        }
    }
}

/// `/ready` behind the circuit breaker (P3 S-12): not ready with `circuit_open` while the
/// circuit blocks readiness (CIRCUIT_OPEN, DRAINING, PROBING); a `base` that is already not
/// ready keeps its own reason.
pub fn readiness_for_circuit(circuit: CircuitState, base: ReadyState) -> ReadyState {
    match base {
        ReadyState::Ready if circuit.blocks_readiness() => ReadyState::NotReady {
            reason: NotReadyReason::CircuitOpen,
        },
        other => other,
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
