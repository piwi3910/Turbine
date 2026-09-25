//! `POST /v1/completions` and `/v1/chat/completions` request bodies (P1 S-10) and their validation.
//!
//! Unknown fields are ignored. Known OpenAI fields this phase does not support are accepted at
//! their default value and rejected otherwise with 400 `unsupported_parameter` naming the field
//! (contract §14.2). Supported fields out of range are 400 `invalid_request`.

use serde::Deserialize;
use serde_json::{Map, Value, json};
use turbine_core::request::Endpoint;

use crate::error::ApiError;

/// Largest `logprobs` (completions) / `top_logprobs` (chat) value.
pub const MAX_LOGPROBS: u32 = 20;
/// Most stop strings per request.
pub const MAX_STOP_STRINGS: usize = 4;

/// A completions or chat request body. Fields after `chat_template_kwargs` are the known-but-
/// unsupported Phase 1 fields; they are kept as raw JSON so any value can be judged.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct OpenAiRequest {
    pub model: Option<String>,
    pub prompt: Option<PromptInput>,
    pub messages: Option<Vec<ChatMessageIn>>,
    pub max_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub seed: Option<u64>,
    pub stop: Option<StopInput>,
    pub stream: Option<bool>,
    pub stream_options: Option<StreamOptions>,
    pub logprobs: Option<LogprobsField>,
    pub top_logprobs: Option<u32>,
    pub echo: Option<bool>,
    pub n: Option<u32>,
    pub ignore_eos: Option<bool>,
    pub return_tokens_as_token_ids: Option<bool>,
    pub chat_template_kwargs: Option<Map<String, Value>>,
    pub tools: Option<Value>,
    pub tool_choice: Option<Value>,
    pub response_format: Option<Value>,
    pub logit_bias: Option<Value>,
    pub presence_penalty: Option<Value>,
    pub frequency_penalty: Option<Value>,
    pub repetition_penalty: Option<Value>,
    pub best_of: Option<Value>,
    pub suffix: Option<Value>,
    pub parallel_tool_calls: Option<Value>,
}

/// Completions `prompt`: text, or pre-tokenized ids.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum PromptInput {
    Text(String),
    Tokens(Vec<u32>),
}

/// One chat message. `tool_calls` / `tool_call_id` are Phase 2 and rejected when present.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct ChatMessageIn {
    pub role: String,
    pub content: Option<MessageContent>,
    pub tool_calls: Option<Value>,
    pub tool_call_id: Option<Value>,
}

/// Chat `content`: a string, or an array of parts.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// One content part; only `{"type":"text","text":…}` is supported (image/audio are rejected).
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: Option<String>,
}

/// `stop`: one string or a list of at most [`MAX_STOP_STRINGS`].
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum StopInput {
    One(String),
    Many(Vec<String>),
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct StreamOptions {
    pub include_usage: Option<bool>,
}

/// `logprobs`: a bool on chat, an integer `0..=20` on completions.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum LogprobsField {
    Bool(bool),
    Int(u32),
}

impl OpenAiRequest {
    /// Parse a JSON body; malformed JSON or a mistyped known field → 400 `invalid_request`.
    pub fn from_slice(body: &[u8]) -> Result<Self, ApiError> {
        serde_json::from_slice(body)
            .map_err(|e| ApiError::invalid_request(format!("invalid request body: {e}")))
    }

    /// Phase 1 rules for `endpoint`; the first violation wins. Does not check `model` against the
    /// served name (the handler does, 404 `model_not_found`) nor the context length (the engine).
    pub fn validate(&self, endpoint: Endpoint) -> Result<(), ApiError> {
        self.check_unsupported()?;
        if self.model.as_deref().is_none_or(str::is_empty) {
            return Err(ApiError::invalid_request("model is required"));
        }
        match endpoint {
            Endpoint::ChatCompletions => self.check_messages()?,
            _ => {
                if self.prompt.is_none() {
                    return Err(ApiError::invalid_request(
                        "prompt is required on /v1/completions",
                    ));
                }
            }
        }
        self.check_ranges()?;
        self.check_logprobs(endpoint)
    }

    fn check_unsupported(&self) -> Result<(), ApiError> {
        let non_default = [
            ("tools", self.tools.as_ref().is_some_and(non_empty_array)),
            (
                "tool_choice",
                self.tool_choice
                    .as_ref()
                    .is_some_and(|v| !v.is_null() && v != "none" && v != "auto"),
            ),
            (
                "response_format",
                self.response_format
                    .as_ref()
                    .is_some_and(|v| !v.is_null() && *v != json!({"type": "text"})),
            ),
            ("n", self.n.is_some_and(|n| n > 1)),
            (
                "logit_bias",
                self.logit_bias.as_ref().is_some_and(non_empty_object),
            ),
            (
                "presence_penalty",
                !is_default_number(&self.presence_penalty, 0.0),
            ),
            (
                "frequency_penalty",
                !is_default_number(&self.frequency_penalty, 0.0),
            ),
            (
                "repetition_penalty",
                !is_default_number(&self.repetition_penalty, 1.0),
            ),
            ("best_of", !is_default_number(&self.best_of, 1.0)),
            (
                "suffix",
                self.suffix
                    .as_ref()
                    .is_some_and(|v| !v.is_null() && v != ""),
            ),
            ("echo", self.echo == Some(true)),
            (
                "parallel_tool_calls",
                self.parallel_tool_calls
                    .as_ref()
                    .is_some_and(|v| !v.is_null() && *v != Value::Bool(true)),
            ),
        ];
        match non_default.iter().find(|(_, set)| *set) {
            Some((field, _)) => Err(ApiError::unsupported_parameter(field)),
            None => Ok(()),
        }
    }

    fn check_messages(&self) -> Result<(), ApiError> {
        let messages = match &self.messages {
            Some(m) if !m.is_empty() => m,
            _ => {
                return Err(ApiError::invalid_request(
                    "messages is required on /v1/chat/completions and must not be empty",
                ));
            }
        };
        for (i, m) in messages.iter().enumerate() {
            if !matches!(m.role.as_str(), "system" | "user" | "assistant") {
                return Err(ApiError::unsupported_parameter(&format!(
                    "messages.role (messages[{i}] has role `{}`; supported: system, user, assistant)",
                    m.role
                )));
            }
            if m.tool_calls.as_ref().is_some_and(|v| !v.is_null()) {
                return Err(ApiError::unsupported_parameter(&format!(
                    "messages[{i}].tool_calls"
                )));
            }
            if m.tool_call_id.as_ref().is_some_and(|v| !v.is_null()) {
                return Err(ApiError::unsupported_parameter(&format!(
                    "messages[{i}].tool_call_id"
                )));
            }
            match &m.content {
                None => {
                    return Err(ApiError::invalid_request(format!(
                        "messages[{i}].content is required"
                    )));
                }
                Some(MessageContent::Text(_)) => {}
                Some(MessageContent::Parts(parts)) => {
                    for (j, p) in parts.iter().enumerate() {
                        if p.kind != "text" {
                            return Err(ApiError::unsupported_parameter(&format!(
                                "messages[{i}].content[{j}] of type {}",
                                p.kind
                            )));
                        }
                        if p.text.is_none() {
                            return Err(ApiError::invalid_request(format!(
                                "messages[{i}].content[{j}].text is required"
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn check_ranges(&self) -> Result<(), ApiError> {
        if let Some(t) = self.temperature
            && !(0.0..=2.0).contains(&t)
        {
            return Err(ApiError::invalid_request(format!(
                "temperature must be in [0, 2], got {t}"
            )));
        }
        if let Some(p) = self.top_p
            && !(p > 0.0 && p <= 1.0)
        {
            return Err(ApiError::invalid_request(format!(
                "top_p must be in (0, 1], got {p}"
            )));
        }
        if let Some(k) = self.top_k
            && k != -1
            && k < 1
        {
            return Err(ApiError::invalid_request(format!(
                "top_k must be -1 or >= 1, got {k}"
            )));
        }
        if let Some(StopInput::Many(v)) = &self.stop
            && v.len() > MAX_STOP_STRINGS
        {
            return Err(ApiError::invalid_request(format!(
                "stop accepts at most {MAX_STOP_STRINGS} strings, got {}",
                v.len()
            )));
        }
        if self.n == Some(0) {
            return Err(ApiError::invalid_request("n must be >= 1"));
        }
        Ok(())
    }

    fn check_logprobs(&self, endpoint: Endpoint) -> Result<(), ApiError> {
        let chat = matches!(endpoint, Endpoint::ChatCompletions);
        match (&self.logprobs, chat) {
            (Some(LogprobsField::Int(_)), true) => {
                return Err(ApiError::invalid_request(
                    "logprobs must be a boolean on /v1/chat/completions (use top_logprobs for alternatives)",
                ));
            }
            (Some(LogprobsField::Bool(_)), false) => {
                return Err(ApiError::invalid_request(format!(
                    "logprobs must be an integer in [0, {MAX_LOGPROBS}] on /v1/completions"
                )));
            }
            (Some(LogprobsField::Int(n)), false) if *n > MAX_LOGPROBS => {
                return Err(ApiError::invalid_request(format!(
                    "logprobs must be an integer in [0, {MAX_LOGPROBS}], got {n}"
                )));
            }
            _ => {}
        }
        if let Some(n) = self.top_logprobs {
            if !chat {
                return Err(ApiError::invalid_request(
                    "top_logprobs is only accepted on /v1/chat/completions; use logprobs",
                ));
            }
            if n > MAX_LOGPROBS {
                return Err(ApiError::invalid_request(format!(
                    "top_logprobs must be in [0, {MAX_LOGPROBS}], got {n}"
                )));
            }
            if n > 0 && self.logprobs != Some(LogprobsField::Bool(true)) {
                return Err(ApiError::invalid_request(
                    "top_logprobs requires logprobs: true",
                ));
            }
        }
        Ok(())
    }

    /// Messages as chat-template input: `{"role","content"}`, text parts joined by `\n`.
    pub fn messages_json(&self) -> Vec<Value> {
        self.messages
            .iter()
            .flatten()
            .map(|m| {
                let content = match &m.content {
                    Some(MessageContent::Text(t)) => t.clone(),
                    Some(MessageContent::Parts(parts)) => parts
                        .iter()
                        .filter_map(|p| p.text.as_deref())
                        .collect::<Vec<_>>()
                        .join("\n"),
                    None => String::new(),
                };
                json!({"role": m.role, "content": content})
            })
            .collect()
    }

    /// `stop` as a list (empty when absent).
    pub fn stop_strings(&self) -> Vec<String> {
        match &self.stop {
            None => Vec::new(),
            Some(StopInput::One(s)) => vec![s.clone()],
            Some(StopInput::Many(v)) => v.clone(),
        }
    }

    /// `max_completion_tokens`, else `max_tokens`; `None` means the remaining context.
    pub fn max_tokens(&self) -> Option<u32> {
        self.max_completion_tokens.or(self.max_tokens)
    }

    /// `stream_options.include_usage` (default false).
    pub fn include_usage(&self) -> bool {
        self.stream_options
            .as_ref()
            .and_then(|o| o.include_usage)
            .unwrap_or(false)
    }

    /// Alternatives per generated token (`SamplingParams::logprobs`); `None` when no logprobs
    /// were requested. Chat: `top_logprobs` (default 0) when `logprobs: true`; completions: the
    /// integer `logprobs`. Call after [`validate`](Self::validate).
    pub fn logprobs_n(&self, endpoint: Endpoint) -> Option<u32> {
        match (&self.logprobs, endpoint) {
            (Some(LogprobsField::Bool(true)), Endpoint::ChatCompletions) => {
                Some(self.top_logprobs.unwrap_or(0).min(MAX_LOGPROBS))
            }
            (Some(LogprobsField::Int(n)), Endpoint::Completions) => Some((*n).min(MAX_LOGPROBS)),
            _ => None,
        }
    }
}

/// Absent, null, or the number `default`.
fn is_default_number(v: &Option<Value>, default: f64) -> bool {
    match v {
        None | Some(Value::Null) => true,
        Some(v) => v.as_f64() == Some(default),
    }
}

/// Anything but null or an empty array.
fn non_empty_array(v: &Value) -> bool {
    !v.is_null() && v.as_array().is_none_or(|a| !a.is_empty())
}

/// Anything but null or an empty object.
fn non_empty_object(v: &Value) -> bool {
    !v.is_null() && v.as_object().is_none_or(|o| !o.is_empty())
}
