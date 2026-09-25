//! Non-streaming OpenAI responses (P1 S-10): the per-request rendering context shared with the
//! SSE stream, logprob rendering, and [`collect`], which folds a generation into one body.

use std::sync::Arc;

use serde_json::{Map, Value, json};
use turbine_core::request::{Endpoint, FinishReason, GenerationEvent, Usage};
use turbine_core::types::RequestId;

use crate::backend::{GenerationStream, InferenceBackend};
use crate::error::ApiError;

/// Message of the internal error raised when the engine drops the stream without `Finished`.
pub(crate) const NO_FINISH: &str = "generation ended without a finish event";

/// Everything needed to render one request's response or chunks.
pub(crate) struct ResponseContext {
    /// `cmpl-<uuid>` or `chatcmpl-<uuid>`.
    pub id: String,
    pub endpoint: Endpoint,
    /// Unix seconds at request start.
    pub created: u64,
    /// The `model` of the request (equal to the served model id).
    pub model: String,
    /// Logprobs were requested (`logprobs` on completions, `logprobs: true` on chat).
    pub logprobs: bool,
    /// `return_tokens_as_token_ids`: token strings are `token_id:<id>`.
    pub token_ids_as_text: bool,
    /// `stream_options.include_usage`.
    pub include_usage: bool,
    pub backend: Arc<dyn InferenceBackend>,
}

impl ResponseContext {
    /// The response id for `request` on `endpoint`.
    pub fn response_id(endpoint: Endpoint, request: RequestId) -> String {
        let prefix = if is_chat(endpoint) {
            "chatcmpl"
        } else {
            "cmpl"
        };
        format!("{prefix}-{}", request.0)
    }

    pub fn chat(&self) -> bool {
        is_chat(self.endpoint)
    }

    /// The `object` of a full response (`streaming = false`) or of a stream chunk.
    pub fn object(&self, streaming: bool) -> &'static str {
        match (self.chat(), streaming) {
            (true, false) => "chat.completion",
            (true, true) => "chat.completion.chunk",
            (false, _) => "text_completion",
        }
    }

    /// Display string of one token in logprob entries.
    fn token_string(&self, token_id: u32) -> String {
        if self.token_ids_as_text {
            format!("token_id:{token_id}")
        } else {
            self.backend.token_text(token_id)
        }
    }

    /// The `logprobs` value of a choice: `null` when not requested, else the endpoint's shape.
    /// Completions: `{tokens, token_logprobs, top_logprobs, text_offset}`; chat:
    /// `{"content":[{token, logprob, bytes, top_logprobs:[{token, logprob, bytes}]}]}`.
    pub fn render_logprobs(&self, entries: &[TokenLogprob]) -> Value {
        if !self.logprobs {
            return Value::Null;
        }
        if self.chat() {
            let content: Vec<Value> = entries
                .iter()
                .map(|e| {
                    let top: Vec<Value> = e
                        .top
                        .iter()
                        .map(|&(id, lp)| self.chat_token(id, Some(lp)))
                        .collect();
                    let mut entry = self.chat_token(e.token_id, e.logprob);
                    entry["top_logprobs"] = Value::Array(top);
                    entry
                })
                .collect();
            json!({ "content": content })
        } else {
            let tokens: Vec<String> = entries
                .iter()
                .map(|e| self.token_string(e.token_id))
                .collect();
            let token_logprobs: Vec<Value> =
                entries.iter().map(|e| logprob_value(e.logprob)).collect();
            let top_logprobs: Vec<Value> = entries
                .iter()
                .map(|e| {
                    let map: Map<String, Value> = e
                        .top
                        .iter()
                        .map(|&(id, lp)| (self.token_string(id), logprob_value(Some(lp))))
                        .collect();
                    Value::Object(map)
                })
                .collect();
            let text_offset: Vec<usize> = entries.iter().map(|e| e.text_offset).collect();
            json!({
                "tokens": tokens,
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
                "text_offset": text_offset,
            })
        }
    }

    fn chat_token(&self, token_id: u32, logprob: Option<f32>) -> Value {
        let token = self.token_string(token_id);
        let bytes = token.as_bytes().to_vec();
        json!({"token": token, "logprob": logprob_value(logprob), "bytes": bytes})
    }
}

fn is_chat(endpoint: Endpoint) -> bool {
    matches!(endpoint, Endpoint::ChatCompletions)
}

fn logprob_value(logprob: Option<f32>) -> Value {
    logprob.map_or(Value::Null, |lp| json!(lp))
}

/// One generated token's logprob data, with the character offset of its text in the output.
pub(crate) struct TokenLogprob {
    pub token_id: u32,
    pub logprob: Option<f32>,
    pub top: Vec<(u32, f32)>,
    pub text_offset: usize,
}

/// The OpenAI `usage` object.
pub(crate) fn usage_json(usage: Usage) -> Value {
    json!({
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.prompt_tokens + usage.completion_tokens,
    })
}

/// Usage reported by the engine, else the tokens counted by the API (prompt unknown: 0).
pub(crate) fn usage_or_counted(usage: Option<Usage>, counted: u32) -> Usage {
    usage.unwrap_or(Usage {
        prompt_tokens: 0,
        completion_tokens: counted,
    })
}

/// Generated text and logprob entries accumulated from `Token` events.
#[derive(Default)]
pub(crate) struct TokenAccumulator {
    /// Characters of generated text so far (the next token's `text_offset`).
    pub chars: usize,
    pub tokens: u32,
}

impl TokenAccumulator {
    /// Account one token; returns its logprob entry when logprobs were requested.
    pub fn token(
        &mut self,
        ctx: &ResponseContext,
        text: &str,
        token_id: u32,
        logprob: Option<f32>,
        top: Vec<(u32, f32)>,
    ) -> Option<TokenLogprob> {
        let entry = ctx.logprobs.then_some(TokenLogprob {
            token_id,
            logprob,
            top,
            text_offset: self.chars,
        });
        self.chars += text.chars().count();
        self.tokens += 1;
        entry
    }
}

/// Fold a generation into the non-streaming response body. An `Error` event (or a stream that
/// ends without `Finished`) becomes the matching HTTP error.
pub(crate) async fn collect(
    mut stream: GenerationStream,
    ctx: ResponseContext,
) -> Result<Value, ApiError> {
    let mut acc = TokenAccumulator::default();
    let mut text = String::new();
    let mut entries = Vec::new();
    while let Some(event) = stream.recv().await {
        match event {
            GenerationEvent::Token {
                text: t,
                token_id,
                logprob,
                top_logprobs,
                ..
            } => {
                entries.extend(acc.token(&ctx, &t, token_id, logprob, top_logprobs));
                text.push_str(&t);
            }
            GenerationEvent::Finished { reason, usage, .. } => {
                let usage = usage_or_counted(usage, acc.tokens);
                return Ok(full_body(&ctx, text, &entries, reason, usage));
            }
            GenerationEvent::Error { code, message } => {
                return Err(ApiError::from_code(code, message));
            }
            // `Started` carries nothing for a full body; later-phase events are not produced
            // for Phase 1 requests.
            _ => {}
        }
    }
    Err(ApiError::internal(NO_FINISH))
}

fn full_body(
    ctx: &ResponseContext,
    text: String,
    entries: &[TokenLogprob],
    reason: FinishReason,
    usage: Usage,
) -> Value {
    let logprobs = ctx.render_logprobs(entries);
    let choice = if ctx.chat() {
        json!({
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "logprobs": logprobs,
            "finish_reason": reason.as_str(),
        })
    } else {
        json!({
            "index": 0,
            "text": text,
            "logprobs": logprobs,
            "finish_reason": reason.as_str(),
        })
    };
    json!({
        "id": ctx.id,
        "object": ctx.object(false),
        "created": ctx.created,
        "model": ctx.model,
        "choices": [choice],
        "usage": usage_json(usage),
    })
}
