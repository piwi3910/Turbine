//! Request vocabulary shared by the API and the engine (contract §3.5, §14.3).
//! Phase 0 carries only the endpoint and the error codes its routes return.

use serde::{Deserialize, Serialize};

/// The OpenAI endpoint a request arrived on.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Endpoint {
    Completions,
    ChatCompletions,
}

impl Endpoint {
    /// Route template, also the metric label value.
    pub fn as_str(self) -> &'static str {
        match self {
            Endpoint::Completions => "/v1/completions",
            Endpoint::ChatCompletions => "/v1/chat/completions",
        }
    }
}

/// OpenAI-shape error `code` values (contract §14.3). Later phases add variants.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ErrorCode {
    ModelNotLoaded,
    NotImplemented,
    NotFound,
    RequestTooLarge,
    MethodNotAllowed,
    InternalError,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::ModelNotLoaded => "model_not_loaded",
            ErrorCode::NotImplemented => "not_implemented",
            ErrorCode::NotFound => "not_found",
            ErrorCode::RequestTooLarge => "request_too_large",
            ErrorCode::MethodNotAllowed => "method_not_allowed",
            ErrorCode::InternalError => "internal_error",
        }
    }
}
