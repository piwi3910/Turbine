//! Request vocabulary shared by the API and the engine (contract §3.5, §14.3).
//! Phase 1 adds sampling, stop conditions, the generation request and its event stream.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::types::RequestId;

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
    ModelNotFound,
    UnsupportedParameter,
    ContextLengthExceeded,
    EngineBusy,
    TemplateError,
    InvalidRequest,
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
            ErrorCode::ModelNotFound => "model_not_found",
            ErrorCode::UnsupportedParameter => "unsupported_parameter",
            ErrorCode::ContextLengthExceeded => "context_length_exceeded",
            ErrorCode::EngineBusy => "engine_busy",
            ErrorCode::TemplateError => "template_error",
            ErrorCode::InvalidRequest => "invalid_request",
        }
    }
}

/// Sampling parameters of one request (P1 S-9; Phase 2 adds penalties, bias, min_tokens).
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub seed: Option<u64>,
    /// Alternatives reported per generated token (0..=20); `None` = no logprobs.
    pub logprobs: Option<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1,
            seed: None,
            logprobs: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct StopConditions {
    pub eos_token_ids: SmallVec<[u32; 4]>,
    pub stop_strings: Vec<String>,
    pub max_tokens: u32,
    pub ignore_eos: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FinishReason {
    Stop,
    Length,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// Built by `turbine-server` after templating and tokenization.
#[derive(Clone, Debug)]
pub struct GenerationRequest {
    pub id: RequestId,
    pub endpoint: Endpoint,
    /// `x-request-id` of the HTTP request: a log field, never a metric label.
    pub http_request_id: String,
    pub prompt_tokens: Vec<u32>,
    pub sampling: SamplingParams,
    pub stop: StopConditions,
}

/// Set by the HTTP side (client disconnect), polled by the engine between steps.
#[derive(Clone, Default, Debug)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Per-request event stream from the engine to the API.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum GenerationEvent {
    /// First event of a choice (the chat stream renders it as the `delta.role` chunk).
    Started {
        choice: u32,
    },
    /// One generated token; `text` is empty while bytes are held (partial UTF-8 or a possible stop-string prefix).
    Token {
        choice: u32,
        text: String,
        token_id: u32,
        logprob: Option<f32>,
        top_logprobs: Vec<(u32, f32)>,
    },
    Finished {
        choice: u32,
        reason: FinishReason,
        usage: Option<Usage>,
    },
    Error {
        code: ErrorCode,
        message: String,
    },
}
