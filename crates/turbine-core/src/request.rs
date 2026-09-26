//! Request vocabulary shared by the API and the engine (contract §3.5, §14.3).
//! Phase 1 adds sampling, stop conditions, the generation request and its event stream.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::types::{Priority, RequestId};

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
    // Phase 2
    QueueFull,
    QueueTimeout,
    ContextExceedsKvCapacity,
    InvalidJsonSchema,
    ToolsNotSupported,
    UnknownTool,
    ShuttingDown,
    RequestTimeout,
    SlowClient,
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
            ErrorCode::QueueFull => "queue_full",
            ErrorCode::QueueTimeout => "queue_timeout",
            ErrorCode::ContextExceedsKvCapacity => "context_exceeds_kv_capacity",
            ErrorCode::InvalidJsonSchema => "invalid_json_schema",
            ErrorCode::ToolsNotSupported => "tools_not_supported",
            ErrorCode::UnknownTool => "unknown_tool",
            ErrorCode::ShuttingDown => "shutting_down",
            ErrorCode::RequestTimeout => "request_timeout",
            ErrorCode::SlowClient => "slow_client",
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
    /// OpenAI presence penalty (Phase 2); 0 = off.
    pub presence_penalty: f32,
    /// OpenAI frequency penalty (Phase 2); 0 = off.
    pub frequency_penalty: f32,
    /// Multiplicative repetition penalty (Phase 2, vLLM extension); 1.0 = off.
    pub repetition_penalty: f32,
    /// `(token id, bias)` pairs added to the logits before sampling (Phase 2).
    pub logit_bias: Vec<(u32, f32)>,
    /// EOS and stop token ids are suppressed until this many tokens were generated (Phase 2).
    pub min_tokens: u32,
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
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            logit_bias: Vec::new(),
            min_tokens: 0,
            logprobs: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct StopConditions {
    pub eos_token_ids: SmallVec<[u32; 4]>,
    pub stop_strings: Vec<String>,
    /// Extra token ids that end the request like EOS (Phase 2).
    pub stop_token_ids: Vec<u32>,
    pub max_tokens: u32,
    pub ignore_eos: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FinishReason {
    Stop,
    Length,
    /// The output was parsed into tool calls (Phase 2).
    ToolCalls,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
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
    /// Number of choices (Phase 2); each choice is one forked sequence.
    pub n: u32,
    pub sampling: SamplingParams,
    pub stop: StopConditions,
    /// Scheduling priority (Phase 2): lower is served first.
    pub priority: Priority,
    /// Completions `echo` (Phase 2): the prompt text prefixes the output.
    pub echo: bool,
    /// Constrained decoding (Phase 2 `response_format` / tool grammar).
    pub constraint: Option<ConstraintSpec>,
    /// Monotonic deadline in milliseconds on the engine clock (`server.request_timeout`).
    pub deadline_ms: u64,
}

/// What constrains a request's output (P2 S-17, S-18).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ConstraintSpec {
    /// `response_format: {"type": "json_object"}`.
    JsonObject,
    /// `response_format: {"type": "json_schema"}` with its schema.
    JsonSchema { schema: serde_json::Value },
    /// `tool_choice` `required` or named: an llguidance grammar built from the tool schemas.
    ToolCall { grammar_source: String },
}

/// One parsed tool call of a choice (P2 S-18).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallOut {
    pub index: u32,
    /// `call_` followed by 24 alphanumerics.
    pub id: String,
    pub name: String,
    /// The call's arguments as a JSON string.
    pub arguments: String,
}

/// Resources a request needs, computed before it is queued (P3 §Data field names; Phase 2
/// fills the token fields and `projected_kv_blocks`).
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct ResourceEstimate {
    pub prompt_tokens: u32,
    /// Tokens served from a cached prefix; 0 until Phase 4.
    pub cached_prefix_tokens: u32,
    pub new_prefill_tokens: u32,
    pub max_output_tokens: u32,
    /// KV blocks the request holds at completion.
    pub projected_kv_blocks: u32,
    pub workspace_bytes: u64,
    pub est_prefill_seconds: f64,
    pub est_decode_seconds: f64,
    /// True once a cost model filled the time estimates (Phase 3).
    pub estimated: bool,
}

impl ResourceEstimate {
    /// Phase 2 estimate: `projected_kv_blocks = ceil((prompt + max_output) / block_tokens)`.
    /// `block_tokens` must be at least 1 (config validation guarantees it).
    pub fn for_request(prompt: u32, max_output: u32, block_tokens: u32) -> ResourceEstimate {
        let tokens = u64::from(prompt) + u64::from(max_output);
        let blocks = tokens.div_ceil(u64::from(block_tokens.max(1)));
        ResourceEstimate {
            prompt_tokens: prompt,
            cached_prefix_tokens: 0,
            new_prefill_tokens: prompt,
            max_output_tokens: max_output,
            projected_kv_blocks: u32::try_from(blocks).unwrap_or(u32::MAX),
            workspace_bytes: 0,
            est_prefill_seconds: 0.0,
            est_decode_seconds: 0.0,
            estimated: false,
        }
    }
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
    /// The choice's output parsed into tool calls (Phase 2); followed by `Finished`.
    ToolCalls {
        choice: u32,
        calls: Vec<ToolCallOut>,
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::clock::{Clock, FakeClock};
    use crate::types::{Priority, PriorityClass};

    #[test]
    fn resource_estimate_blocks() {
        // 100-token prompt, max_tokens 60, 16-token blocks → ceil(160 / 16) = 10 blocks.
        let e = ResourceEstimate::for_request(100, 60, 16);
        assert_eq!(e.projected_kv_blocks, 10);
        assert_eq!(e.prompt_tokens, 100);
        assert_eq!(e.new_prefill_tokens, 100);
        assert_eq!(e.cached_prefix_tokens, 0);
        assert_eq!(e.max_output_tokens, 60);
        // One token past a block boundary needs one more block.
        assert_eq!(
            ResourceEstimate::for_request(100, 61, 16).projected_kv_blocks,
            11
        );

        // Priority classes (CONFLICT C-10): lower is served first.
        assert_eq!(Priority(-1).class(), PriorityClass::High);
        assert_eq!(Priority::default().class(), PriorityClass::Normal);
        assert_eq!(Priority(3).class(), PriorityClass::Low);

        let start = Duration::from_secs(7);
        let clock = FakeClock::new(start);
        clock.advance(Duration::from_millis(250));
        assert_eq!(clock.now_mono(), start + Duration::from_millis(250));
        clock.set(Duration::from_secs(1));
        assert_eq!(clock.now_mono(), Duration::from_secs(1));

        // Phase 2 wire strings.
        assert_eq!(FinishReason::ToolCalls.as_str(), "tool_calls");
        assert_eq!(
            ErrorCode::ContextExceedsKvCapacity.as_str(),
            "context_exceeds_kv_capacity"
        );
        assert_eq!(ErrorCode::QueueFull.as_str(), "queue_full");
        assert_eq!(
            serde_json::to_value(ErrorCode::InvalidJsonSchema).unwrap(),
            "invalid_json_schema"
        );
        assert_eq!(SamplingParams::default().repetition_penalty, 1.0);
    }
}
