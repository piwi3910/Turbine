//! `/turbine/v1/*` diagnostics.

use std::collections::HashMap;

use axum::Json;
use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};

use crate::backend::{ApiState, TopologyScope};
use crate::error::ApiError;
use crate::kv::{PrefetchAccepted, PrefetchRequest};

pub(super) async fn status(State(state): State<ApiState>) -> Json<serde_json::Value> {
    Json(state.diagnostics.status())
}

pub(super) async fn devices(State(state): State<ApiState>) -> Json<serde_json::Value> {
    Json(state.diagnostics.devices())
}

pub(super) async fn kv(State(state): State<ApiState>) -> Result<Json<serde_json::Value>, ApiError> {
    state.diagnostics.kv().map(Json)
}

/// `POST /turbine/v1/kv/prefetch` (Phase 4): 202 with the queued and resident block counts;
/// 404 `session_not_found`, 429 `prefetch_queue_full`, 409 `pressure_too_high`.
pub(super) async fn kv_prefetch(
    State(state): State<ApiState>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Result<(StatusCode, Json<PrefetchAccepted>), ApiError> {
    let bytes = match body {
        Ok(bytes) => bytes,
        Err(rejection) if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE => {
            return Err(ApiError::request_too_large(state.limits.max_request_bytes));
        }
        Err(rejection) => return Err(ApiError::invalid_request(rejection.body_text())),
    };
    let req = PrefetchRequest::parse(&bytes, &headers)?;
    let accepted = state.inference.prefetch(req).await?;
    Ok((StatusCode::ACCEPTED, Json(accepted)))
}

pub(super) async fn pressure(
    State(state): State<ApiState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    state.diagnostics.pressure().map(Json)
}

pub(super) async fn scheduler(
    State(state): State<ApiState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    state.diagnostics.scheduler().map(Json)
}

/// `?scope=node` (the default); any other scope is 400 `unsupported_parameter` until phase 6.
pub(super) async fn topology(
    State(state): State<ApiState>,
    Query(query): Query<HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let scope = match query.get("scope").map(String::as_str) {
        None | Some("node") => TopologyScope::Node,
        Some(_) => return Err(ApiError::unsupported_parameter("scope")),
    };
    state.diagnostics.topology(scope).map(Json)
}
