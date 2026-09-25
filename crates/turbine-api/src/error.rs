//! OpenAI-shape errors: `{"error":{"message","type","code"}}`.

use axum::Json;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use turbine_core::request::ErrorCode;

/// OpenAI error `type` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ErrorType {
    InvalidRequestError,
    RateLimitError,
    ServiceUnavailable,
    NotImplemented,
    NotFound,
    ServerError,
    Timeout,
}

/// An HTTP error rendered as the OpenAI body `{"error":{"message","type","code"}}`,
/// with an optional `Retry-After` header (seconds).
#[derive(Clone, Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub kind: ErrorType,
    pub code: ErrorCode,
    pub message: String,
    pub retry_after: Option<u64>,
}

impl ApiError {
    /// An error with no `Retry-After`.
    pub fn new(
        status: StatusCode,
        kind: ErrorType,
        code: ErrorCode,
        message: impl Into<String>,
    ) -> Self {
        ApiError {
            status,
            kind,
            code,
            message: message.into(),
            retry_after: None,
        }
    }

    /// 503 `service_unavailable`/`model_not_loaded`: inference with no model.
    pub fn model_not_loaded() -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            ErrorType::ServiceUnavailable,
            ErrorCode::ModelNotLoaded,
            "no model is loaded",
        )
    }

    /// 501 `not_implemented`: a diagnostic this build does not serve yet.
    pub fn not_implemented() -> Self {
        Self::new(
            StatusCode::NOT_IMPLEMENTED,
            ErrorType::NotImplemented,
            ErrorCode::NotImplemented,
            "this diagnostic is not implemented in this build",
        )
    }

    /// 404 `not_found`: no route matches `path`.
    pub fn not_found(path: &str) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            ErrorType::NotFound,
            ErrorCode::NotFound,
            format!("no route for {path}"),
        )
    }

    /// 405 `invalid_request_error`/`method_not_allowed`: the route exists, the method does not.
    pub fn method_not_allowed(method: &str, path: &str) -> Self {
        Self::new(
            StatusCode::METHOD_NOT_ALLOWED,
            ErrorType::InvalidRequestError,
            ErrorCode::MethodNotAllowed,
            format!("method {method} is not allowed on {path}"),
        )
    }

    /// 413 `invalid_request_error`/`request_too_large`: body over `limit` bytes.
    pub fn request_too_large(limit: usize) -> Self {
        Self::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            ErrorType::InvalidRequestError,
            ErrorCode::RequestTooLarge,
            format!("request body exceeds server.max_request_bytes ({limit} bytes)"),
        )
    }

    /// 500 `server_error`/`internal_error`.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            ErrorType::ServerError,
            ErrorCode::InternalError,
            message,
        )
    }

    /// 404 `invalid_request_error`/`model_not_found`: `model` is not the served model.
    pub fn model_not_found(model: &str) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            ErrorType::InvalidRequestError,
            ErrorCode::ModelNotFound,
            format!("the model `{model}` does not exist"),
        )
    }

    /// 400 `invalid_request_error`/`unsupported_parameter`: a known OpenAI field at a value this
    /// build does not support; `field` names it.
    pub fn unsupported_parameter(field: &str) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            ErrorType::InvalidRequestError,
            ErrorCode::UnsupportedParameter,
            format!("unsupported parameter: {field}"),
        )
    }

    /// 400 `invalid_request_error`/`context_length_exceeded`: prompt + `max_tokens` over the context.
    pub fn context_length_exceeded(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            ErrorType::InvalidRequestError,
            ErrorCode::ContextLengthExceeded,
            message,
        )
    }

    /// 429 `rate_limit_error`/`engine_busy` with `retry-after: 1`: the Phase 1 generation slot is taken.
    pub fn engine_busy() -> Self {
        Self::from_code(
            ErrorCode::EngineBusy,
            "a generation is already running; retry later",
        )
    }

    /// 400 `invalid_request_error`/`template_error`: the chat template failed to render.
    pub fn template_error(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            ErrorType::InvalidRequestError,
            ErrorCode::TemplateError,
            message,
        )
    }

    /// 400 `invalid_request_error`/`invalid_request`: malformed body, missing or out-of-range field.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            ErrorType::InvalidRequestError,
            ErrorCode::InvalidRequest,
            message,
        )
    }

    /// Status, `type` and `retry-after` for a code reported by the engine (contract §14.3 table).
    pub fn from_code(code: ErrorCode, message: impl Into<String>) -> Self {
        let (status, kind) = match code {
            ErrorCode::ModelNotLoaded => (
                StatusCode::SERVICE_UNAVAILABLE,
                ErrorType::ServiceUnavailable,
            ),
            ErrorCode::NotImplemented => (StatusCode::NOT_IMPLEMENTED, ErrorType::NotImplemented),
            ErrorCode::NotFound => (StatusCode::NOT_FOUND, ErrorType::NotFound),
            ErrorCode::RequestTooLarge => (
                StatusCode::PAYLOAD_TOO_LARGE,
                ErrorType::InvalidRequestError,
            ),
            ErrorCode::MethodNotAllowed => (
                StatusCode::METHOD_NOT_ALLOWED,
                ErrorType::InvalidRequestError,
            ),
            ErrorCode::ModelNotFound => (StatusCode::NOT_FOUND, ErrorType::InvalidRequestError),
            ErrorCode::UnsupportedParameter
            | ErrorCode::ContextLengthExceeded
            | ErrorCode::TemplateError
            | ErrorCode::InvalidRequest => {
                (StatusCode::BAD_REQUEST, ErrorType::InvalidRequestError)
            }
            ErrorCode::EngineBusy => (StatusCode::TOO_MANY_REQUESTS, ErrorType::RateLimitError),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, ErrorType::ServerError),
        };
        let mut e = Self::new(status, kind, code, message);
        if code == ErrorCode::EngineBusy {
            e.retry_after = Some(1);
        }
        e
    }

    /// The `{"error":{"message","type","code"}}` document (also the mid-stream SSE error event).
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "error": {"message": self.message, "type": self.kind, "code": self.code}
        })
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut resp = (self.status, Json(self.to_json())).into_response();
        if let Some(secs) = self.retry_after {
            resp.headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from(secs));
        }
        resp
    }
}
