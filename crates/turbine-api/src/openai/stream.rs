//! SSE streaming (P1 S-10, P2 S-10/S-18): maps the engine's [`GenerationEvent`]s to OpenAI chunks.
//!
//! Every chunk carries one choice, named by its `index`; the choices of an `n` > 1 request
//! interleave. Per choice: the chat role chunk (from `Started`), one chunk per non-empty token
//! text, one chunk per tool call (`delta.tool_calls` with `index`, `id`, `type`, `function.name`
//! and the complete `function.arguments`), and a finish chunk with `finish_reason`. Once every
//! choice has finished: a `{"choices":[],"usage":…}` chunk when `include_usage` (the prompt once,
//! the completion tokens of every choice), then `data: [DONE]`.
//!
//! A mid-stream `Error` event — including the server-side cancellations `request_timeout`,
//! `slow_client` and `shutting_down` — or an engine that drops the stream before every choice
//! finished sends `data: {"error":{…}}` followed by `data: [DONE]` (CONFLICT C-3).
//!
//! The receiver lives inside the response body stream: a client disconnect drops the body, the
//! receiver with it, and the engine sees its next send fail (the cancellation signal).

use std::convert::Infallible;

use axum::response::sse::{Event, Sse};
use futures_util::stream::{self, Stream, StreamExt};
use serde_json::{Value, json};
use turbine_core::request::{FinishReason, GenerationEvent, ToolCallOut, Usage};

use crate::backend::GenerationStream;
use crate::error::ApiError;
use crate::openai::response::{
    ChoiceState, NO_FINISH, ResponseContext, choice_index, tool_call_json, total_usage, usage_json,
};

/// The SSE response for one generation.
pub(crate) fn sse(
    stream: GenerationStream,
    ctx: ResponseContext,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let state = StreamState {
        rx: stream,
        choices: ChoiceState::for_request(&ctx),
        ctx,
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
    /// Per choice; `entries` holds the logprob entries of tokens whose text is still held, which
    /// ride on that choice's next chunk.
    choices: Vec<ChoiceState>,
    /// `[DONE]` has been produced; the stream ends (and drops the receiver).
    done: bool,
}

impl StreamState {
    /// The SSE events for one engine event (`None`: the engine dropped its sender).
    fn on_event(&mut self, event: Option<GenerationEvent>) -> Vec<Event> {
        let result = match event {
            Some(GenerationEvent::Started { choice }) => self.started(choice),
            Some(GenerationEvent::Token {
                choice,
                text,
                token_id,
                logprob,
                top_logprobs,
            }) => self.token(choice, text, token_id, logprob, top_logprobs),
            Some(GenerationEvent::ToolCalls { choice, calls }) => self.tool_calls(choice, calls),
            Some(GenerationEvent::Finished {
                choice,
                reason,
                usage,
            }) => self.finished(choice, reason, usage),
            Some(GenerationEvent::Error { code, message }) => {
                Err(ApiError::from_code(code, message))
            }
            None => Err(ApiError::internal(NO_FINISH)),
            // Later-phase events carry nothing a Phase 2 client renders.
            Some(_) => Ok(Vec::new()),
        };
        result.unwrap_or_else(|e| self.fail(e))
    }

    fn index(&self, choice: u32) -> Result<usize, ApiError> {
        choice_index(choice, self.choices.len())
    }

    /// Chat: the role chunk. Completions have none.
    fn started(&mut self, choice: u32) -> Result<Vec<Event>, ApiError> {
        let i = self.index(choice)?;
        if !self.ctx.chat() {
            return Ok(Vec::new());
        }
        Ok(vec![self.chunk(
            i,
            json!({"role": "assistant", "content": ""}),
            None,
        )])
    }

    fn token(
        &mut self,
        choice: u32,
        text: String,
        token_id: u32,
        logprob: Option<f32>,
        top: Vec<(u32, f32)>,
    ) -> Result<Vec<Event>, ApiError> {
        let i = self.index(choice)?;
        let c = &mut self.choices[i];
        let entry = c.acc.token(&self.ctx, &text, token_id, logprob, top);
        c.entries.extend(entry);
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let content = if self.ctx.chat() {
            json!({ "content": text })
        } else {
            Value::String(text)
        };
        Ok(vec![self.chunk(i, content, None)])
    }

    /// Chat: one chunk per call, in call order. Completions have no tool calls.
    fn tool_calls(
        &mut self,
        choice: u32,
        mut calls: Vec<ToolCallOut>,
    ) -> Result<Vec<Event>, ApiError> {
        let i = self.index(choice)?;
        if !self.ctx.chat() {
            return Ok(Vec::new());
        }
        calls.sort_by_key(|call| call.index);
        Ok(calls
            .iter()
            .map(|call| {
                let mut entry = tool_call_json(call);
                entry["index"] = json!(call.index);
                self.chunk(i, json!({ "tool_calls": [entry] }), None)
            })
            .collect())
    }

    fn finished(
        &mut self,
        choice: u32,
        reason: FinishReason,
        usage: Option<Usage>,
    ) -> Result<Vec<Event>, ApiError> {
        let i = self.index(choice)?;
        if self.choices[i].finish.is_some() {
            return Err(ApiError::internal(format!(
                "choice {choice} finished twice"
            )));
        }
        self.choices[i].finish = Some(reason);
        self.choices[i].usage = usage;
        let empty = if self.ctx.chat() {
            json!({})
        } else {
            Value::String(String::new())
        };
        let mut out = vec![self.chunk(i, empty, Some(reason))];
        if self.choices.iter().all(|c| c.finish.is_some()) {
            if self.ctx.include_usage {
                out.push(data(&json!({
                    "id": self.ctx.id,
                    "object": self.ctx.object(true),
                    "created": self.ctx.created,
                    "model": self.ctx.model,
                    "choices": [],
                    "usage": usage_json(total_usage(&self.choices)),
                })));
            }
            out.push(done_event());
            self.done = true;
        }
        Ok(out)
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

    /// One chunk for choice `i`: `content` is the chat `delta` or the completion `text`; the
    /// choice's pending logprob entries are attached and drained.
    fn chunk(&mut self, i: usize, content: Value, finish: Option<FinishReason>) -> Event {
        let entries = std::mem::take(&mut self.choices[i].entries);
        let logprobs = if entries.is_empty() {
            Value::Null
        } else {
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
                "index": i,
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
