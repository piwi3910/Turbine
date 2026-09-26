//! Non-streaming OpenAI responses (P1 S-10, P2 S-10/S-18): the per-request rendering context
//! shared with the SSE stream, logprob and tool-call rendering, per-choice accounting, and
//! [`collect`], which folds a generation into one body with one choice per index.

use std::sync::Arc;

use serde_json::{Map, Value, json};
use turbine_core::request::{Endpoint, FinishReason, GenerationEvent, ToolCallOut, Usage};
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
    /// Choices requested (`n`, ≥ 1); the response is complete when every one has finished.
    pub choices: u32,
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

/// A tool call as OpenAI renders it in `message.tool_calls` (the streamed `delta.tool_calls`
/// entry adds `index`).
pub(crate) fn tool_call_json(call: &ToolCallOut) -> Value {
    json!({
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": call.arguments},
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

/// Per-choice accounting shared by the full body and the SSE stream.
#[derive(Default)]
pub(crate) struct ChoiceState {
    pub acc: TokenAccumulator,
    /// Logprob entries not rendered yet (the stream drains them into its next chunk).
    pub entries: Vec<TokenLogprob>,
    /// Set by the choice's `Finished` event.
    pub finish: Option<FinishReason>,
    /// The engine's usage for this choice, when reported.
    pub usage: Option<Usage>,
}

impl ChoiceState {
    /// One state per requested choice.
    pub fn for_request(ctx: &ResponseContext) -> Vec<ChoiceState> {
        (0..ctx.choices.max(1))
            .map(|_| ChoiceState::default())
            .collect()
    }
}

/// The index of `choice` in a request with `len` choices, or the internal error an engine that
/// names a choice it was not asked for deserves.
pub(crate) fn choice_index(choice: u32, len: usize) -> Result<usize, ApiError> {
    usize::try_from(choice)
        .ok()
        .filter(|&i| i < len)
        .ok_or_else(|| {
            ApiError::internal(format!(
                "generation event for choice {choice} of a request with {len} choices"
            ))
        })
}

/// Usage of the whole request: the prompt once (the largest count reported) and the completion
/// tokens of every choice (the engine's count, else the tokens the API saw).
pub(crate) fn total_usage<'a>(choices: impl IntoIterator<Item = &'a ChoiceState>) -> Usage {
    let mut total = Usage::default();
    for c in choices {
        let (prompt, completion) = match c.usage {
            Some(u) => (u.prompt_tokens, u.completion_tokens),
            None => (0, c.acc.tokens),
        };
        total.prompt_tokens = total.prompt_tokens.max(prompt);
        total.completion_tokens += completion;
    }
    total
}

/// One choice of a non-streaming response.
#[derive(Default)]
struct CollectedChoice {
    state: ChoiceState,
    text: String,
    tool_calls: Vec<ToolCallOut>,
}

/// Fold a generation into the non-streaming response body; complete once every choice has
/// finished. An `Error` event (or a stream that ends before every choice finished) becomes the
/// matching HTTP error.
pub(crate) async fn collect(
    mut stream: GenerationStream,
    ctx: ResponseContext,
) -> Result<Value, ApiError> {
    let mut choices: Vec<CollectedChoice> = ChoiceState::for_request(&ctx)
        .into_iter()
        .map(|state| CollectedChoice {
            state,
            ..CollectedChoice::default()
        })
        .collect();
    let n = choices.len();
    while let Some(event) = stream.recv().await {
        match event {
            GenerationEvent::Token {
                choice,
                text,
                token_id,
                logprob,
                top_logprobs,
            } => {
                let c = &mut choices[choice_index(choice, n)?];
                let entry = c
                    .state
                    .acc
                    .token(&ctx, &text, token_id, logprob, top_logprobs);
                c.state.entries.extend(entry);
                c.text.push_str(&text);
            }
            GenerationEvent::ToolCalls { choice, calls } => {
                choices[choice_index(choice, n)?].tool_calls.extend(calls);
            }
            GenerationEvent::Finished {
                choice,
                reason,
                usage,
            } => {
                let c = &mut choices[choice_index(choice, n)?];
                c.state.finish = Some(reason);
                c.state.usage = usage;
                if choices.iter().all(|c| c.state.finish.is_some()) {
                    return Ok(full_body(&ctx, &mut choices));
                }
            }
            GenerationEvent::Error { code, message } => {
                return Err(ApiError::from_code(code, message));
            }
            // `Started` carries nothing for a full body.
            _ => {}
        }
    }
    Err(ApiError::internal(NO_FINISH))
}

fn full_body(ctx: &ResponseContext, choices: &mut [CollectedChoice]) -> Value {
    let rendered: Vec<Value> = choices
        .iter_mut()
        .zip(0u32..)
        .map(|(c, index)| {
            let logprobs = ctx.render_logprobs(&c.state.entries);
            let finish_reason = c.state.finish.map_or(Value::Null, |r| r.as_str().into());
            if ctx.chat() {
                let mut message = json!({"role": "assistant", "content": c.text});
                if !c.tool_calls.is_empty() {
                    c.tool_calls.sort_by_key(|call| call.index);
                    if c.text.is_empty() {
                        message["content"] = Value::Null;
                    }
                    message["tool_calls"] = c.tool_calls.iter().map(tool_call_json).collect();
                }
                json!({
                    "index": index,
                    "message": message,
                    "logprobs": logprobs,
                    "finish_reason": finish_reason,
                })
            } else {
                json!({
                    "index": index,
                    "text": c.text,
                    "logprobs": logprobs,
                    "finish_reason": finish_reason,
                })
            }
        })
        .collect();
    json!({
        "id": ctx.id,
        "object": ctx.object(false),
        "created": ctx.created,
        "model": ctx.model,
        "choices": rendered,
        "usage": usage_json(total_usage(choices.iter().map(|c| &c.state))),
    })
}
