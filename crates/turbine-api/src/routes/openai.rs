//! Health, readiness, metrics and the OpenAI routes.

use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{Extension, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::json;
use turbine_core::request::Endpoint;
use turbine_core::types::RequestId;
use turbine_observability::OPENMETRICS_CONTENT_TYPE;
use turbine_observability::http::RequestIdExt;

use crate::backend::{ApiState, InferenceRequest, ReadyState};
use crate::error::ApiError;
use crate::kv::parse_turbine_headers;
use crate::openai::request::OpenAiRequest;
use crate::openai::response::{ResponseContext, collect};
use crate::openai::stream::sse;

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

pub(super) async fn chat_completions(
    State(state): State<ApiState>,
    request_id: Option<Extension<RequestIdExt>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Result<Response, ApiError> {
    generate(state, Endpoint::ChatCompletions, request_id, headers, body).await
}

pub(super) async fn completions(
    State(state): State<ApiState>,
    request_id: Option<Extension<RequestIdExt>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Result<Response, ApiError> {
    generate(state, Endpoint::Completions, request_id, headers, body).await
}

/// Shared completions/chat handler. Every error before `submit` returns a plain HTTP error and
/// is reported through `record_rejection`; a `submit` error is a plain HTTP error even for
/// `stream: true` (the engine counts those itself).
async fn generate(
    state: ApiState,
    endpoint: Endpoint,
    request_id: Option<Extension<RequestIdExt>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Result<Response, ApiError> {
    let created = unix_seconds();
    let reject = |e: ApiError| {
        state.inference.record_rejection(endpoint, e.code);
        e
    };
    let (body, model) = admit(&state, endpoint, body).map_err(reject)?;
    let hints = parse_turbine_headers(&headers, body.prompt_cache_key.is_some()).map_err(reject)?;

    let id = RequestId::new_v4();
    let stream = body.stream == Some(true);
    let ctx = ResponseContext {
        id: ResponseContext::response_id(endpoint, id),
        endpoint,
        created,
        model,
        logprobs: body.logprobs_n(endpoint).is_some(),
        token_ids_as_text: body.return_tokens_as_token_ids == Some(true),
        include_usage: body.include_usage(),
        choices: body.n(),
        backend: state.inference.clone(),
    };
    let http_request_id = request_id
        .map(|Extension(RequestIdExt(r))| r)
        .unwrap_or_default();
    let events = state
        .inference
        .submit(InferenceRequest {
            id,
            endpoint,
            body,
            http_request_id,
            hints,
        })
        .await?;
    if stream {
        Ok(sse(events, ctx).into_response())
    } else {
        Ok(Json(collect(events, ctx).await?).into_response())
    }
}

/// Body limit, model presence, JSON parsing, field validation and the `model` check, in that
/// order. Returns the parsed body and the requested model id.
fn admit(
    state: &ApiState,
    endpoint: Endpoint,
    body: Result<Bytes, BytesRejection>,
) -> Result<(OpenAiRequest, String), ApiError> {
    let bytes = match body {
        Ok(bytes) => bytes,
        Err(rejection) if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE => {
            return Err(ApiError::request_too_large(state.limits.max_request_bytes));
        }
        Err(rejection) => return Err(ApiError::invalid_request(rejection.body_text())),
    };
    let served = state.inference.models();
    if served.is_empty() {
        return Err(ApiError::model_not_loaded());
    }
    let body = OpenAiRequest::from_slice(&bytes)?;
    body.validate(endpoint)?;
    let model = body.model.clone().unwrap_or_default();
    if !served.iter().any(|card| card.id == model) {
        return Err(ApiError::model_not_found(&model));
    }
    Ok((body, model))
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}
