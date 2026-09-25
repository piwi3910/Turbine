//! `/turbine/v1/*` diagnostics.

use axum::Json;
use axum::extract::State;

use crate::backend::ApiState;
use crate::error::ApiError;

pub(super) async fn status(State(state): State<ApiState>) -> Json<serde_json::Value> {
    Json(state.diagnostics.status())
}

pub(super) async fn devices(State(state): State<ApiState>) -> Json<serde_json::Value> {
    Json(state.diagnostics.devices())
}

pub(super) async fn kv(State(state): State<ApiState>) -> Result<Json<serde_json::Value>, ApiError> {
    state.diagnostics.kv().map(Json)
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
