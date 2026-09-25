//! SSE streaming (P1 S-10): maps the engine's [`GenerationEvent`]s to OpenAI chunks.
//!
//! Order: chat role chunk (from `Started`), one chunk per non-empty token text, a finish chunk
//! with `finish_reason`, a `{"choices":[],"usage":…}` chunk when `include_usage`, then
//! `data: [DONE]`. A mid-stream `Error` event (or an engine that drops the stream without
//! `Finished`) sends `data: {"error":{…}}` followed by `data: [DONE]` (CONFLICT C-3).
//!
//! The receiver lives inside the response body stream: a client disconnect drops the body, the
//! receiver with it, and the engine sees its next send fail (the cancellation signal).

use std::convert::Infallible;

use axum::response::sse::{Event, Sse};
use futures_util::stream::{self, Stream, StreamExt};
use serde_json::{Value, json};
use turbine_core::request::{FinishReason, GenerationEvent, Usage};

use crate::backend::GenerationStream;
use crate::error::ApiError;
use crate::openai::response::{
    NO_FINISH, ResponseContext, TokenAccumulator, TokenLogprob, usage_json, usage_or_counted,
};

/// The SSE response for one generation.
pub(crate) fn sse(
    stream: GenerationStream,
    ctx: ResponseContext,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let state = StreamState {
        rx: stream,
        ctx,
        acc: TokenAccumulator::default(),
        pending: Vec::new(),
        done: false,
    };
    let events = stream::unfold(state, |mut st| async move {
        if st.done {
            return None;
        }
        let event = st.rx.recv().await;
        let out = st.on_event(event);
        Some((out, st))
    })
    .flat_map(|events| stream::iter(events.into_iter().map(Ok)));
    Sse::new(events)
}

struct StreamState {
    rx: GenerationStream,
    ctx: ResponseContext,
    acc: TokenAccumulator,
    /// Logprob entries of tokens whose text is still held; they ride on the next chunk.
    pending: Vec<TokenLogprob>,
    /// `[DONE]` has been produced; the stream ends (and drops the receiver).
    done: bool,
}

impl StreamState {
    /// The SSE events for one engine event (`None`: the engine dropped its sender).
    fn on_event(&mut self, event: Option<GenerationEvent>) -> Vec<Event> {
        match event {
            Some(GenerationEvent::Started { .. }) if self.ctx.chat() => {
                vec![self.chunk(json!({"role": "assistant", "content": ""}), None)]
            }
            Some(GenerationEvent::Token {
                text,
                token_id,
                logprob,
                top_logprobs,
                ..
            }) => {
                let entry = self
                    .acc
                    .token(&self.ctx, &text, token_id, logprob, top_logprobs);
                self.pending.extend(entry);
                if text.is_empty() {
                    Vec::new()
                } else {
                    vec![self.text_chunk(text)]
                }
            }
            Some(GenerationEvent::Finished { reason, usage, .. }) => {
                let usage = usage_or_counted(usage, self.acc.tokens);
                self.finish(reason, usage)
            }
            Some(GenerationEvent::Error { code, message }) => {
                self.fail(ApiError::from_code(code, message))
            }
            None => self.fail(ApiError::internal(NO_FINISH)),
            // Completions have no role chunk; later-phase events are not produced for Phase 1.
            Some(_) => Vec::new(),
        }
    }

    fn text_chunk(&mut self, text: String) -> Event {
        let content = if self.ctx.chat() {
            json!({ "content": text })
        } else {
            Value::String(text)
        };
        self.chunk(content, None)
    }

    fn finish(&mut self, reason: FinishReason, usage: Usage) -> Vec<Event> {
        let empty = if self.ctx.chat() {
            json!({})
        } else {
            Value::String(String::new())
        };
        let mut out = vec![self.chunk(empty, Some(reason))];
        if self.ctx.include_usage {
            out.push(data(&json!({
                "id": self.ctx.id,
                "object": self.ctx.object(true),
                "created": self.ctx.created,
                "model": self.ctx.model,
                "choices": [],
                "usage": usage_json(usage),
            })));
        }
        out.push(done_event());
        self.done = true;
        out
    }

    fn fail(&mut self, error: ApiError) -> Vec<Event> {
        tracing::warn!(
            response_id = %self.ctx.id,
            code = error.code.as_str(),
            message = %error.message,
            "generation failed mid-stream"
        );
        self.done = true;
        vec![data(&error.to_json()), done_event()]
    }

    /// One chunk: `content` is the chat `delta` or the completion `text`; pending logprob
    /// entries are attached and drained.
    fn chunk(&mut self, content: Value, finish: Option<FinishReason>) -> Event {
        let logprobs = if self.pending.is_empty() {
            Value::Null
        } else {
            let entries = std::mem::take(&mut self.pending);
            self.ctx.render_logprobs(&entries)
        };
        let finish_reason = finish.map_or(Value::Null, |r| Value::from(r.as_str()));
        let key = if self.ctx.chat() { "delta" } else { "text" };
        data(&json!({
            "id": self.ctx.id,
            "object": self.ctx.object(true),
            "created": self.ctx.created,
            "model": self.ctx.model,
            "choices": [{
                "index": 0,
                key: content,
                "logprobs": logprobs,
                "finish_reason": finish_reason,
            }],
        }))
    }
}

fn data(value: &Value) -> Event {
    Event::default().data(value.to_string())
}

fn done_event() -> Event {
    Event::default().data("[DONE]")
}
