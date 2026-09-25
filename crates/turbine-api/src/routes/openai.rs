//! Health, readiness, metrics and the OpenAI routes (Phase 0: no model, so 503).

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::extract::rejection::BytesRejection;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::json;
use turbine_observability::OPENMETRICS_CONTENT_TYPE;

use crate::backend::{ApiState, ReadyState};
use crate::error::ApiError;

pub(super) async fn health() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

pub(super) async fn ready(State(state): State<ApiState>) -> Response {
    match state.readiness.ready() {
        ReadyState::Ready => (StatusCode::OK, Json(json!({"ready": true}))).into_response(),
        ReadyState::NotReady { reason } => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"ready": false, "reason": reason.as_str()})),
        )
            .into_response(),
    }
}

pub(super) async fn metrics(State(state): State<ApiState>) -> Response {
    match state.metrics.render() {
        Ok(text) => ([(header::CONTENT_TYPE, OPENMETRICS_CONTENT_TYPE)], text).into_response(),
        Err(e) => {
            tracing::error!(error = %e, "metrics rendering failed");
            ApiError::internal(e.to_string()).into_response()
        }
    }
}

pub(super) async fn models(State(state): State<ApiState>) -> Json<serde_json::Value> {
    Json(json!({"object": "list", "data": state.inference.models()}))
}

/// Read (and bound) the body before answering, so oversized requests get 413, not 503.
fn check_body(state: &ApiState, body: Result<Bytes, BytesRejection>) -> Result<(), ApiError> {
    match body {
        Err(rejection) if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE => {
            Err(ApiError::request_too_large(state.limits.max_request_bytes))
        }
        // Phase 0 does not parse inference bodies (Phase 2): any other body outcome is irrelevant.
        _ => Ok(()),
    }
}

pub(super) async fn chat_completions(
    State(state): State<ApiState>,
    body: Result<Bytes, BytesRejection>,
) -> Result<Response, ApiError> {
    check_body(&state, body)?;
    Err(ApiError::model_not_loaded())
}

pub(super) async fn completions(
    State(state): State<ApiState>,
    body: Result<Bytes, BytesRejection>,
) -> Result<Response, ApiError> {
    check_body(&state, body)?;
    Err(ApiError::model_not_loaded())
}
