//! `POST /v1/completions` and `/v1/chat/completions` request bodies (P1 S-10, P2 S-10/S-17/S-18)
//! and their validation.
//!
//! Unknown fields are ignored. Known OpenAI fields this build does not support are accepted at
//! their default value and rejected otherwise with 400 `unsupported_parameter` naming the field
//! (contract §14.2). Supported fields that are malformed, missing or out of range are 400
//! `invalid_request`; a `tool_choice` naming a function not in `tools` is 400 `unknown_tool`.
//! Checks that need the model (vocabulary size for `logit_bias`, the tool-call parser, schema
//! compilation, context and KV capacity) are the backend's.

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use turbine_core::request::Endpoint;

use crate::error::ApiError;

/// Largest `logprobs` (completions) / `top_logprobs` (chat) value.
pub const MAX_LOGPROBS: u32 = 20;
/// Most stop strings per request.
pub const MAX_STOP_STRINGS: usize = 4;
/// `presence_penalty` / `frequency_penalty` range (OpenAI).
const PENALTY_RANGE: std::ops::RangeInclusive<f32> = -2.0..=2.0;
/// `logit_bias` value range (OpenAI).
const LOGIT_BIAS_RANGE: std::ops::RangeInclusive<f32> = -100.0..=100.0;

/// A completions or chat request body. `best_of` and `suffix` are the known-but-unsupported
/// fields; they are kept as raw JSON so any value can be judged.
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
    /// Completions only (P2): the prompt text is prepended to each choice.
    pub echo: Option<bool>,
    /// Choices per request (P2: ≥ 1, forked after prefill).
    pub n: Option<u32>,
    pub ignore_eos: Option<bool>,
    pub return_tokens_as_token_ids: Option<bool>,
    pub chat_template_kwargs: Option<Map<String, Value>>,
    /// `-2..=2`, default 0 (P2).
    pub presence_penalty: Option<f32>,
    /// `-2..=2`, default 0 (P2).
    pub frequency_penalty: Option<f32>,
    /// `> 0`, default 1 (P2, vLLM extension).
    pub repetition_penalty: Option<f32>,
    /// Token id (decimal string) → bias in `-100..=100` (P2); ids are checked against the
    /// vocabulary by the backend. Use [`logit_bias`](Self::logit_bias).
    pub logit_bias: Option<BTreeMap<String, f32>>,
    /// EOS and stop token ids are suppressed until this many tokens were generated (P2).
    pub min_tokens: Option<u32>,
    /// Token ids that end a choice (P2, vLLM extension).
    pub stop_token_ids: Option<Vec<u32>>,
    /// Scheduling priority, lower first (P2, vLLM extension; CONFLICT C-10).
    pub priority: Option<i32>,
    /// End-user id: recorded on the request span only (P2).
    pub user: Option<String>,
    /// `text` (default), `json_object` or `json_schema` (P2 S-17).
    pub response_format: Option<ResponseFormat>,
    /// Chat only (P2 S-18).
    pub tools: Option<Vec<ToolDefinition>>,
    /// Chat only; use [`tool_choice_mode`](Self::tool_choice_mode).
    pub tool_choice: Option<ToolChoiceIn>,
    /// Chat only, default true; use [`parallel_tool_calls_enabled`](Self::parallel_tool_calls_enabled).
    pub parallel_tool_calls: Option<bool>,
    pub best_of: Option<Value>,
    pub suffix: Option<Value>,
}

/// Completions `prompt`: text, or pre-tokenized ids.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum PromptInput {
    Text(String),
    Tokens(Vec<u32>),
}

/// One chat message: roles `system`, `user`, `assistant` (optionally with `tool_calls`) and
/// `tool` (with `tool_call_id`).
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct ChatMessageIn {
    pub role: String,
    pub content: Option<MessageContent>,
    /// Assistant messages only: the calls the model made earlier in the conversation.
    pub tool_calls: Option<Vec<ToolCallIn>>,
    /// `tool` messages only: the id of the call this message answers.
    pub tool_call_id: Option<String>,
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

/// `response_format` (S-17); an unknown `type` is a malformed body (400 `invalid_request`).
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: JsonSchemaFormat },
}

/// `response_format.json_schema`. `strict` is accepted and does not change enforcement.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct JsonSchemaFormat {
    pub name: String,
    pub description: Option<String>,
    /// Required: a JSON Schema object (or boolean schema); compiled by the backend.
    pub schema: Option<Value>,
    pub strict: Option<bool>,
}

/// One `tools` entry: `{"type":"function","function":{name, description, parameters}}`.
/// Serializes back to the same shape for the chat template.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDefinition,
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The arguments' JSON Schema (an object when present).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// `tool_choice`: `"none"`, `"auto"`, `"required"` or `{"type":"function","function":{"name"}}`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum ToolChoiceIn {
    Mode(String),
    Function(NamedToolChoice),
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct NamedToolChoice {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionName,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct FunctionName {
    pub name: String,
}

/// The effective tool choice after defaults: `auto` when `tools` is non-empty and `tool_choice`
/// is absent, `none` when there are no tools.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolChoiceMode {
    None,
    Auto,
    Required,
    Named(String),
}

/// One `tool_calls` entry of an assistant message.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct ToolCallIn {
    pub id: String,
    /// `"function"` when present.
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub function: FunctionCallIn,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct FunctionCallIn {
    pub name: String,
    /// The arguments as a JSON string (OpenAI shape).
    pub arguments: String,
}

impl OpenAiRequest {
    /// Parse a JSON body; malformed JSON or a mistyped known field → 400 `invalid_request`.
    pub fn from_slice(body: &[u8]) -> Result<Self, ApiError> {
        serde_json::from_slice(body)
            .map_err(|e| ApiError::invalid_request(format!("invalid request body: {e}")))
    }

    /// The rules for `endpoint`; the first violation wins. Does not check `model` against the
    /// served name (the handler does, 404 `model_not_found`) nor what needs the model (the backend).
    pub fn validate(&self, endpoint: Endpoint) -> Result<(), ApiError> {
        let chat = matches!(endpoint, Endpoint::ChatCompletions);
        self.check_unsupported(chat)?;
        if self.model.as_deref().is_none_or(str::is_empty) {
            return Err(ApiError::invalid_request("model is required"));
        }
        if chat {
            self.check_messages()?;
        } else if self.prompt.is_none() {
            return Err(ApiError::invalid_request(
                "prompt is required on /v1/completions",
            ));
        }
        self.check_ranges()?;
        self.check_logprobs(chat)?;
        self.check_response_format()?;
        self.check_tools()
    }

    fn check_unsupported(&self, chat: bool) -> Result<(), ApiError> {
        let non_default = [
            ("best_of", !is_default_number(&self.best_of, 1.0)),
            (
                "suffix",
                self.suffix
                    .as_ref()
                    .is_some_and(|v| !v.is_null() && v != ""),
            ),
            // `echo` is a completions field; chat has no prompt text to echo.
            (
                "echo (on /v1/chat/completions)",
                chat && self.echo == Some(true),
            ),
            // Tools are a chat feature (S-18).
            (
                "tools (on /v1/completions)",
                !chat && self.tools.as_ref().is_some_and(|t| !t.is_empty()),
            ),
            (
                "tool_choice (on /v1/completions)",
                !chat && self.tool_choice.as_ref().is_some_and(|c| !c.is_none()),
            ),
            (
                "parallel_tool_calls (on /v1/completions)",
                !chat && self.parallel_tool_calls == Some(false),
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
            let role = m.role.as_str();
            if !matches!(role, "system" | "user" | "assistant" | "tool") {
                return Err(ApiError::unsupported_parameter(&format!(
                    "messages.role (messages[{i}] has role `{role}`; supported: system, user, \
                     assistant, tool)"
                )));
            }
            let has_calls = m.tool_calls.as_ref().is_some_and(|c| !c.is_empty());
            if let Some(calls) = &m.tool_calls {
                if role != "assistant" {
                    return Err(ApiError::invalid_request(format!(
                        "messages[{i}].tool_calls is only allowed on assistant messages"
                    )));
                }
                check_tool_calls_in(i, calls)?;
            }
            match (role, &m.tool_call_id) {
                ("tool", None) => {
                    return Err(ApiError::invalid_request(format!(
                        "messages[{i}].tool_call_id is required on tool messages"
                    )));
                }
                ("tool", Some(id)) if id.is_empty() => {
                    return Err(ApiError::invalid_request(format!(
                        "messages[{i}].tool_call_id must not be empty"
                    )));
                }
                ("tool", Some(_)) | (_, None) => {}
                (_, Some(_)) => {
                    return Err(ApiError::invalid_request(format!(
                        "messages[{i}].tool_call_id is only allowed on tool messages"
                    )));
                }
            }
            match &m.content {
                // An assistant message that carries tool calls may omit its content.
                None if has_calls => {}
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
        for (field, value) in [
            ("presence_penalty", self.presence_penalty),
            ("frequency_penalty", self.frequency_penalty),
        ] {
            if let Some(v) = value
                && !PENALTY_RANGE.contains(&v)
            {
                return Err(ApiError::invalid_request(format!(
                    "{field} must be in [-2, 2], got {v}"
                )));
            }
        }
        if let Some(r) = self.repetition_penalty
            && !(r > 0.0 && r.is_finite())
        {
            return Err(ApiError::invalid_request(format!(
                "repetition_penalty must be > 0, got {r}"
            )));
        }
        for (key, bias) in self.logit_bias.iter().flatten() {
            if key.parse::<u32>().is_err() {
                return Err(ApiError::invalid_request(format!(
                    "logit_bias keys must be token ids, got `{key}`"
                )));
            }
            if !LOGIT_BIAS_RANGE.contains(bias) {
                return Err(ApiError::invalid_request(format!(
                    "logit_bias values must be in [-100, 100], got {bias} for token {key}"
                )));
            }
        }
        if let (Some(min), Some(max)) = (self.min_tokens, self.max_tokens())
            && min > max
        {
            return Err(ApiError::invalid_request(format!(
                "min_tokens ({min}) must not exceed max_tokens ({max})"
            )));
        }
        Ok(())
    }

    fn check_logprobs(&self, chat: bool) -> Result<(), ApiError> {
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

    fn check_response_format(&self) -> Result<(), ApiError> {
        if let Some(ResponseFormat::JsonSchema { json_schema }) = &self.response_format {
            if json_schema.name.is_empty() {
                return Err(ApiError::invalid_request(
                    "response_format.json_schema.name must not be empty",
                ));
            }
            match &json_schema.schema {
                Some(Value::Object(_) | Value::Bool(_)) => {}
                Some(_) => {
                    return Err(ApiError::invalid_request(
                        "response_format.json_schema.schema must be a JSON Schema object",
                    ));
                }
                None => {
                    return Err(ApiError::invalid_request(
                        "response_format.json_schema.schema is required",
                    ));
                }
            }
        }
        Ok(())
    }

    fn check_tools(&self) -> Result<(), ApiError> {
        let tools = self.tools.as_deref().unwrap_or_default();
        let mut names = HashSet::new();
        for (i, tool) in tools.iter().enumerate() {
            if tool.kind != "function" {
                return Err(ApiError::unsupported_parameter(&format!(
                    "tools[{i}].type `{}` (supported: function)",
                    tool.kind
                )));
            }
            let f = &tool.function;
            if f.name.is_empty() {
                return Err(ApiError::invalid_request(format!(
                    "tools[{i}].function.name must not be empty"
                )));
            }
            if !names.insert(f.name.as_str()) {
                return Err(ApiError::invalid_request(format!(
                    "tools[{i}].function.name `{}` is defined twice",
                    f.name
                )));
            }
            if f.parameters.as_ref().is_some_and(|p| !p.is_object()) {
                return Err(ApiError::invalid_request(format!(
                    "tools[{i}].function.parameters must be a JSON Schema object"
                )));
            }
        }
        match &self.tool_choice {
            None => {}
            Some(ToolChoiceIn::Mode(mode)) => match mode.as_str() {
                "none" | "auto" => {}
                "required" if tools.is_empty() => {
                    return Err(ApiError::invalid_request(
                        "tool_choice `required` needs a non-empty tools list",
                    ));
                }
                "required" => {}
                other => {
                    return Err(ApiError::invalid_request(format!(
                        "tool_choice must be none, auto, required or a named function, got `{other}`"
                    )));
                }
            },
            Some(ToolChoiceIn::Function(named)) => {
                if named.kind != "function" {
                    return Err(ApiError::invalid_request(format!(
                        "tool_choice.type must be function, got `{}`",
                        named.kind
                    )));
                }
                if !names.contains(named.function.name.as_str()) {
                    return Err(ApiError::unknown_tool(&named.function.name));
                }
            }
        }
        let constrained_output = self
            .response_format
            .as_ref()
            .is_some_and(|f| *f != ResponseFormat::Text);
        if constrained_output && self.tool_choice_mode() != ToolChoiceMode::None {
            return Err(ApiError::unsupported_parameter(
                "response_format other than text together with tool_choice other than none",
            ));
        }
        Ok(())
    }

    /// Messages as chat-template input: `{"role","content"}` with text parts joined by `\n`;
    /// assistant `tool_calls` (their `arguments` parsed to JSON when they parse, as the
    /// Llama-3.x template expects) and the `tool_call_id` of tool messages are carried over.
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
                let mut out = json!({"role": m.role, "content": content});
                if let Some(calls) = &m.tool_calls {
                    let calls: Vec<Value> = calls
                        .iter()
                        .map(|c| {
                            let arguments = serde_json::from_str(&c.function.arguments)
                                .unwrap_or_else(|_| Value::String(c.function.arguments.clone()));
                            json!({
                                "id": c.id,
                                "type": "function",
                                "function": {"name": c.function.name, "arguments": arguments},
                            })
                        })
                        .collect();
                    out["tool_calls"] = Value::Array(calls);
                }
                if let Some(id) = &m.tool_call_id {
                    out["tool_call_id"] = Value::String(id.clone());
                }
                out
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

    /// Choices requested (default 1).
    pub fn n(&self) -> u32 {
        self.n.unwrap_or(1)
    }

    /// `logit_bias` as `(token id, bias)` sorted by id (`SamplingParams::logit_bias`). Call after
    /// [`validate`](Self::validate); keys that are not token ids are skipped.
    pub fn logit_bias(&self) -> Vec<(u32, f32)> {
        let mut bias: Vec<(u32, f32)> = self
            .logit_bias
            .iter()
            .flatten()
            .filter_map(|(k, &v)| k.parse().ok().map(|id| (id, v)))
            .collect();
        bias.sort_by_key(|&(id, _)| id);
        bias
    }

    /// The `tools` as JSON for the chat template and the tool grammar (empty when absent).
    pub fn tools_json(&self) -> Vec<Value> {
        self.tools
            .iter()
            .flatten()
            .map(|t| serde_json::to_value(t).unwrap_or(Value::Null))
            .collect()
    }

    /// The effective `tool_choice`: `none` without tools, `auto` when `tools` is non-empty and
    /// `tool_choice` is absent. Call after [`validate`](Self::validate).
    pub fn tool_choice_mode(&self) -> ToolChoiceMode {
        if self.tools.as_ref().is_none_or(Vec::is_empty) {
            return ToolChoiceMode::None;
        }
        match &self.tool_choice {
            None => ToolChoiceMode::Auto,
            Some(ToolChoiceIn::Function(named)) => {
                ToolChoiceMode::Named(named.function.name.clone())
            }
            Some(ToolChoiceIn::Mode(mode)) => match mode.as_str() {
                "none" => ToolChoiceMode::None,
                "required" => ToolChoiceMode::Required,
                _ => ToolChoiceMode::Auto,
            },
        }
    }

    /// `parallel_tool_calls` (default true).
    pub fn parallel_tool_calls_enabled(&self) -> bool {
        self.parallel_tool_calls.unwrap_or(true)
    }
}

impl ToolChoiceIn {
    /// `"none"`.
    fn is_none(&self) -> bool {
        matches!(self, ToolChoiceIn::Mode(m) if m == "none")
    }
}

/// Each assistant `tool_calls` entry needs an id, type `function` and a function name.
fn check_tool_calls_in(i: usize, calls: &[ToolCallIn]) -> Result<(), ApiError> {
    for (j, call) in calls.iter().enumerate() {
        if call.id.is_empty() {
            return Err(ApiError::invalid_request(format!(
                "messages[{i}].tool_calls[{j}].id must not be empty"
            )));
        }
        if call.kind.as_deref().is_some_and(|k| k != "function") {
            return Err(ApiError::invalid_request(format!(
                "messages[{i}].tool_calls[{j}].type must be function"
            )));
        }
        if call.function.name.is_empty() {
            return Err(ApiError::invalid_request(format!(
                "messages[{i}].tool_calls[{j}].function.name must not be empty"
            )));
        }
    }
    Ok(())
}

/// Absent, null, or the number `default`.
fn is_default_number(v: &Option<Value>, default: f64) -> bool {
    match v {
        None | Some(Value::Null) => true,
        Some(v) => v.as_f64() == Some(default),
    }
}
