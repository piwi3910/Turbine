//! Route table (TS §13) and middleware stack.

mod diagnostics;
mod openai;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::http::{Method, Uri};
use axum::routing::{get, post};
use turbine_observability::http::{HttpMetrics, http_metrics_layer, request_id_layer};

use crate::backend::ApiState;
use crate::error::ApiError;

/// Every V1 route, the body limit, the 404/405 fallbacks, and the request-id + metrics layers.
pub fn router(state: ApiState) -> Router {
    let http_metrics = HttpMetrics::register(&state.metrics);
    let limit = state.limits.max_request_bytes;
    Router::new()
        .route("/health", get(openai::health))
        .route("/ready", get(openai::ready))
        .route("/metrics", get(openai::metrics))
        .route("/v1/models", get(openai::models))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/completions", post(openai::completions))
        .route("/turbine/v1/status", get(diagnostics::status))
        .route("/turbine/v1/devices", get(diagnostics::devices))
        .route("/turbine/v1/kv", get(diagnostics::kv))
        .route("/turbine/v1/kv/prefetch", post(diagnostics::kv_prefetch))
        .route("/turbine/v1/pressure", get(diagnostics::pressure))
        .route("/turbine/v1/scheduler", get(diagnostics::scheduler))
        .route("/turbine/v1/topology", get(diagnostics::topology))
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed)
        .layer(DefaultBodyLimit::max(limit))
        .layer(http_metrics_layer(http_metrics))
        .layer(request_id_layer())
        .with_state(state)
}

async fn not_found(uri: Uri) -> ApiError {
    ApiError::not_found(uri.path())
}

async fn method_not_allowed(method: Method, uri: Uri) -> ApiError {
    ApiError::method_not_allowed(method.as_str(), uri.path())
}
