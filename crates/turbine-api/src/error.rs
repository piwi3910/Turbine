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
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    error: ErrorFields<'a>,
}

#[derive(Serialize)]
struct ErrorFields<'a> {
    message: &'a str,
    #[serde(rename = "type")]
    kind: ErrorType,
    code: ErrorCode,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ErrorBody {
            error: ErrorFields {
                message: &self.message,
                kind: self.kind,
                code: self.code,
            },
        };
        let mut resp = (self.status, Json(body)).into_response();
        if let Some(secs) = self.retry_after {
            resp.headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from(secs));
        }
        resp
    }
}
