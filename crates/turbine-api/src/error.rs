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

    /// 429 `rate_limit_error`/`queue_full` with `retry-after: 1`: the waiting queue is full
    /// (Phase 2 `scheduler.max_queued_requests`, from Phase 3 `reliability.admission.max_queue`).
    pub fn queue_full() -> Self {
        Self::from_code(
            ErrorCode::QueueFull,
            "the request queue is full; retry later",
        )
    }

    /// 503 `service_unavailable`/`queue_timeout`: waited longer than
    /// `reliability.admission.queue_timeout` (CONFLICT C-1).
    pub fn queue_timeout() -> Self {
        Self::from_code(
            ErrorCode::QueueTimeout,
            "the request waited longer than reliability.admission.queue_timeout before it could start",
        )
    }

    /// An admission or recovery error (P3 reject table, contract §14.3): status and `type`
    /// from the code — `context_exceeds_kv_capacity` 400 `invalid_request_error`, `queue_full`
    /// 429 `rate_limit_error`, `queue_timeout` / `overloaded` / `circuit_open` 503
    /// `service_unavailable`, `resource_exhausted` 503 `server_error` — with `Retry-After` set to
    /// `retry_after` seconds (at least 1) when given and absent otherwise.
    pub fn overload(code: ErrorCode, retry_after: Option<u64>) -> Self {
        let message = match code {
            ErrorCode::ContextExceedsKvCapacity => {
                "the request's KV cache at completion exceeds the KV pool capacity"
            }
            ErrorCode::QueueFull => "the admission queue is full; retry later",
            ErrorCode::QueueTimeout => {
                "the request waited longer than reliability.admission.queue_timeout before it could start"
            }
            ErrorCode::Overloaded => "the server is overloaded; retry later",
            ErrorCode::CircuitOpen => {
                "the device circuit breaker is open; retry after the cooldown"
            }
            ErrorCode::ResourceExhausted => {
                "device memory was exhausted and recovery retries failed"
            }
            _ => "the request could not be admitted",
        };
        let mut e = Self::from_code(code, message);
        e.retry_after = retry_after.map(|s| s.max(1));
        e
    }

    /// 400 `invalid_request_error`/`context_exceeds_kv_capacity`: the request's KV at completion
    /// needs more blocks than the pool holds (CONFLICT C-2).
    pub fn context_exceeds_kv_capacity(message: impl Into<String>) -> Self {
        Self::from_code(ErrorCode::ContextExceedsKvCapacity, message)
    }

    /// 400 `invalid_request_error`/`invalid_json_schema`: the schema or tool grammar does not
    /// compile within bounds; `message` names the llguidance error or the bound.
    pub fn invalid_json_schema(message: impl Into<String>) -> Self {
        Self::from_code(ErrorCode::InvalidJsonSchema, message)
    }

    /// 400 `invalid_request_error`/`tools_not_supported`: `tools` on a model with no tool-call parser.
    pub fn tools_not_supported(model: &str) -> Self {
        Self::from_code(
            ErrorCode::ToolsNotSupported,
            format!("the model `{model}` does not support tools (model.tool_call_parser is none)"),
        )
    }

    /// 400 `invalid_request_error`/`unknown_tool`: `tool_choice` names a function not in `tools`.
    pub fn unknown_tool(name: &str) -> Self {
        Self::from_code(
            ErrorCode::UnknownTool,
            format!("tool_choice names the function `{name}`, which is not in tools"),
        )
    }

    /// 503 `service_unavailable`/`shutting_down`: the server is draining for shutdown.
    pub fn shutting_down() -> Self {
        Self::from_code(
            ErrorCode::ShuttingDown,
            "the server is shutting down and accepts no new requests",
        )
    }

    /// 504 `timeout`/`request_timeout`: the request ran past `server.request_timeout`.
    pub fn request_timeout() -> Self {
        Self::from_code(
            ErrorCode::RequestTimeout,
            "the request exceeded server.request_timeout",
        )
    }

    /// 400 `invalid_request_error`/`invalid_session_id`: `prompt_cache_key` is not 1-128 visible
    /// ASCII characters.
    pub fn invalid_session_id() -> Self {
        Self::from_code(
            ErrorCode::InvalidSessionId,
            "prompt_cache_key must be 1 to 128 visible ASCII characters",
        )
    }

    /// 400 `invalid_request_error`/`invalid_session_hint`: a bad `x-turbine-session-*` header,
    /// or one without `prompt_cache_key`.
    pub fn invalid_session_hint() -> Self {
        Self::from_code(
            ErrorCode::InvalidSessionHint,
            "x-turbine-session-resume-within must be an integer 1..86400 and \
             x-turbine-session-end true or false; both need prompt_cache_key",
        )
    }

    /// 400 `invalid_request_error`/`invalid_cache_salt`: `x-turbine-cache-salt` is not 1-128
    /// visible ASCII characters.
    pub fn invalid_cache_salt() -> Self {
        Self::from_code(
            ErrorCode::InvalidCacheSalt,
            "x-turbine-cache-salt must be 1 to 128 visible ASCII characters",
        )
    }

    /// 404 `not_found`/`session_not_found`: a prefetch names no known session.
    pub fn session_not_found() -> Self {
        Self::from_code(
            ErrorCode::SessionNotFound,
            "no KV session has this prompt_cache_key",
        )
    }

    /// 429 `rate_limit_error`/`prefetch_queue_full`: the prefetch queue is at
    /// `kv.prefetch.max_queue`.
    pub fn prefetch_queue_full() -> Self {
        Self::from_code(
            ErrorCode::PrefetchQueueFull,
            "the KV prefetch queue is full (kv.prefetch.max_queue)",
        )
    }

    /// 409 `invalid_request_error`/`pressure_too_high`: no prefetch at ORANGE pressure or above.
    pub fn pressure_too_high() -> Self {
        Self::from_code(
            ErrorCode::PressureTooHigh,
            "KV memory pressure is ORANGE or above; prefetch refused",
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
            | ErrorCode::InvalidRequest
            | ErrorCode::ContextExceedsKvCapacity
            | ErrorCode::InvalidJsonSchema
            | ErrorCode::ToolsNotSupported
            | ErrorCode::UnknownTool
            | ErrorCode::InvalidSessionId
            | ErrorCode::InvalidSessionHint
            | ErrorCode::InvalidCacheSalt => {
                (StatusCode::BAD_REQUEST, ErrorType::InvalidRequestError)
            }
            ErrorCode::SessionNotFound => (StatusCode::NOT_FOUND, ErrorType::NotFound),
            ErrorCode::PrefetchQueueFull => {
                (StatusCode::TOO_MANY_REQUESTS, ErrorType::RateLimitError)
            }
            ErrorCode::PressureTooHigh => (StatusCode::CONFLICT, ErrorType::InvalidRequestError),
            ErrorCode::EngineBusy | ErrorCode::QueueFull => {
                (StatusCode::TOO_MANY_REQUESTS, ErrorType::RateLimitError)
            }
            ErrorCode::QueueTimeout
            | ErrorCode::ShuttingDown
            | ErrorCode::Overloaded
            | ErrorCode::CircuitOpen => (
                StatusCode::SERVICE_UNAVAILABLE,
                ErrorType::ServiceUnavailable,
            ),
            // P3: 503 for a non-streaming request; the mid-stream event keeps `server_error`.
            ErrorCode::ResourceExhausted => {
                (StatusCode::SERVICE_UNAVAILABLE, ErrorType::ServerError)
            }
            ErrorCode::RequestTimeout => (StatusCode::GATEWAY_TIMEOUT, ErrorType::Timeout),
            // `internal_error`, and `slow_client` (stream only, `server_error`).
            _ => (StatusCode::INTERNAL_SERVER_ERROR, ErrorType::ServerError),
        };
        let mut e = Self::new(status, kind, code, message);
        if matches!(code, ErrorCode::EngineBusy | ErrorCode::QueueFull) {
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
