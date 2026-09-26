//! [`ModelBackend`] answers the API (`InferenceBackend`, `Readiness`, `Diagnostics`): it renders
//! and tokenizes a request, builds its [`GenerationRequest`] and forwards it to the engine thread
//! (`crate::engine`), whose scheduler decides admission (P2 §Scheduling rules, submission
//! checks): `context_length_exceeded`, `context_exceeds_kv_capacity`, `queue_full` (429,
//! `retry-after: 1`) and `shutting_down` (503) are plain HTTP errors returned before any event.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};
use turbine_api::openai::request::{OpenAiRequest, PromptInput, ResponseFormat, ToolChoiceMode};
use turbine_api::{
    ApiError, BoxFuture, Diagnostics, GenerationStream, InferenceBackend, InferenceRequest,
    ModelCard, NotReadyReason, Readiness, ReadyState,
};
use turbine_core::request::{
    Endpoint, ErrorCode, GenerationRequest, SamplingParams, StopConditions,
};
use turbine_core::types::Priority;
use turbine_device::DeviceInventory;
use turbine_kv::blocks_for_tokens;
use turbine_model::{ChatTemplate, Tokenizer};
use turbine_scheduler::SubmitError;

use crate::engine::{EVENT_CHANNEL_CAPACITY, EngineCommand, EngineHandle, EngineShared, Fatal};
use crate::metrics::{Outcome, ServerMetrics};
use crate::model::PreparedModel;

const STATE_LOADING: u8 = 0;
const STATE_READY: u8 = 1;
const STATE_LOAD_FAILED: u8 = 2;
const STATE_DEVICE_ERROR: u8 = 3;
/// SIGINT/SIGTERM received (P2 S-13): `/ready` and new requests answer 503 `shutting_down`.
const STATE_SHUTTING_DOWN: u8 = 4;

/// Sampling defaults from `generation_config.json` (request fields override them).
#[derive(Clone, Copy, Debug)]
struct SamplingDefaults {
    temperature: f32,
    top_p: f32,
    top_k: i32,
}

/// `model` object of `GET /turbine/v1/status`.
#[derive(Serialize)]
struct ModelStatus<'a> {
    served_name: &'a str,
    architecture: &'a str,
    weight_bytes: u64,
    load_seconds: Option<f64>,
}

/// `GET /turbine/v1/status` document.
#[derive(Serialize)]
struct StatusDocument<'a> {
    version: &'static str,
    uptime_seconds: u64,
    ready: bool,
    device_count: u64,
    model: ModelStatus<'a>,
}

/// Set once the model is loaded and warmed up.
struct Loaded {
    engine: EngineHandle,
    shared: Arc<EngineShared>,
    load_seconds: f64,
    weight_bytes: u64,
    created: u64,
}

/// The server's `InferenceBackend`, `Readiness` and `Diagnostics`.
pub struct ModelBackend {
    state: AtomicU8,
    loaded: OnceLock<Loaded>,
    served_name: String,
    architecture: &'static str,
    expected_weight_bytes: u64,
    max_seq_len: u32,
    vocab_size: u32,
    /// `kv.block_tokens` and the pool's block count, for the `context_exceeds_kv_capacity`
    /// message.
    block_tokens: u32,
    pool_blocks: u32,
    /// `scheduler.max_batch_tokens`, for the message of a prompt refused without chunked
    /// prefill.
    max_batch_tokens: u32,
    eos_token_ids: smallvec::SmallVec<[u32; 4]>,
    defaults: SamplingDefaults,
    tokenizer: Arc<Tokenizer>,
    template: Arc<ChatTemplate>,
    metrics: ServerMetrics,
    started: Instant,
    device_count: u64,
    devices: Value,
}

impl ModelBackend {
    /// A backend in the `loading_model` state for `model`.
    pub fn new(
        model: &PreparedModel,
        inventory: &DeviceInventory,
        metrics: ServerMetrics,
    ) -> ModelBackend {
        let eos_token_ids = if model.arch.eos_token_ids.is_empty() {
            model.generation.eos_token_ids.clone()
        } else {
            model.arch.eos_token_ids.clone()
        };
        ModelBackend {
            state: AtomicU8::new(STATE_LOADING),
            loaded: OnceLock::new(),
            served_name: model.served_name.clone(),
            architecture: model.arch.architecture.as_str(),
            expected_weight_bytes: model.budget.weights,
            max_seq_len: model.max_seq_len,
            vocab_size: model.arch.vocab_size,
            block_tokens: model.block_tokens,
            pool_blocks: model.pool.num_blocks,
            max_batch_tokens: model.scheduler.max_batch_tokens,
            eos_token_ids,
            defaults: SamplingDefaults {
                temperature: model.generation.temperature.unwrap_or(1.0),
                top_p: model.generation.top_p.unwrap_or(1.0),
                top_k: model.generation.top_k.unwrap_or(-1),
            },
            tokenizer: Arc::clone(&model.tokenizer),
            template: Arc::clone(&model.template),
            metrics,
            started: Instant::now(),
            device_count: inventory.devices.len() as u64,
            devices: serde_json::to_value(inventory).unwrap_or(Value::Null),
        }
    }

    /// Weights loaded and warmed up: `/ready` turns 200 and requests are accepted.
    pub fn set_ready(
        &self,
        engine: EngineHandle,
        shared: Arc<EngineShared>,
        load_seconds: f64,
        weight_bytes: u64,
    ) {
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        if self
            .loaded
            .set(Loaded {
                engine,
                shared,
                load_seconds,
                weight_bytes,
                created,
            })
            .is_ok()
        {
            // A shutdown that began during the load keeps `shutting_down`.
            let _ = self.state.compare_exchange(
                STATE_LOADING,
                STATE_READY,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
    }

    /// `/ready` 503 with the reason of `fatal` until the process exits.
    pub fn set_failed(&self, fatal: &Fatal) {
        let state = match fatal {
            Fatal::LoadFailed(_) => STATE_LOAD_FAILED,
            Fatal::DeviceError(_) => STATE_DEVICE_ERROR,
        };
        self.state.store(state, Ordering::Release);
    }

    /// Shutdown begins: `/ready` turns 503 `shutting_down` and new requests are refused with
    /// 503 `shutting_down`, while the engine keeps serving what it holds. A failure state
    /// stays.
    pub fn begin_shutdown(&self) {
        let _ = self
            .state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |s| {
                matches!(s, STATE_LOADING | STATE_READY).then_some(STATE_SHUTTING_DOWN)
            });
    }

    /// No request is queued or running in the engine (true before it runs).
    pub fn engine_idle(&self) -> bool {
        self.loaded
            .get()
            .and_then(|l| l.shared.docs())
            .is_none_or(|d| {
                let s = &d.scheduler;
                s.waiting + s.prefilling + s.decoding + s.paused == 0
            })
    }

    /// The engine thread has stopped (its command receiver is gone), or never started.
    pub fn engine_stopped(&self) -> bool {
        self.loaded
            .get()
            .is_none_or(|l| l.engine.submit_tx.is_closed())
    }

    /// Asks the engine to cancel what is left and stop (the server is exiting). Never waits.
    pub fn stop_engine(&self) {
        if let Some(loaded) = self.loaded.get() {
            let _ = loaded.engine.submit_tx.try_send(EngineCommand::Shutdown);
        }
    }

    fn is_ready(&self) -> bool {
        self.state.load(Ordering::Acquire) == STATE_READY
    }

    /// Ready, or draining for shutdown: the model is still listed, so a new request reaches
    /// `start` and is refused with `shutting_down` rather than `model_not_loaded`.
    fn is_serving(&self) -> bool {
        matches!(
            self.state.load(Ordering::Acquire),
            STATE_READY | STATE_SHUTTING_DOWN
        )
    }

    fn reject(&self, endpoint: Endpoint, e: ApiError) -> ApiError {
        self.metrics.request(endpoint, Outcome::Rejected);
        tracing::info!(
            event = "request_rejected",
            endpoint = endpoint.as_str(),
            reason = e.code.as_str(),
            message = %e.message,
            "request rejected"
        );
        e
    }

    /// Prompt token ids: the rendered chat template (its text carries the BOS token), the
    /// tokenized prompt string (`add_special_tokens`), or the given ids.
    fn prompt_tokens(&self, req: &InferenceRequest) -> Result<Vec<u32>, ApiError> {
        let tokens = match req.endpoint {
            Endpoint::ChatCompletions => {
                let empty = serde_json::Map::new();
                let kwargs = req.body.chat_template_kwargs.as_ref().unwrap_or(&empty);
                let text = self
                    .template
                    .render(&req.body.messages_json(), None, true, kwargs)
                    .map_err(|e| ApiError::template_error(e.to_string()))?;
                self.tokenizer
                    .encode(&text, false)
                    .map_err(|e| ApiError::invalid_request(e.to_string()))?
            }
            _ => match &req.body.prompt {
                Some(PromptInput::Text(text)) => self
                    .tokenizer
                    .encode(text, true)
                    .map_err(|e| ApiError::invalid_request(e.to_string()))?,
                Some(PromptInput::Tokens(ids)) => {
                    if let Some(bad) = ids.iter().find(|&&id| id >= self.vocab_size) {
                        return Err(ApiError::invalid_request(format!(
                            "prompt token id {bad} is outside the vocabulary (size {})",
                            self.vocab_size
                        )));
                    }
                    ids.clone()
                }
                None => return Err(ApiError::invalid_request("prompt is required")),
            },
        };
        if tokens.is_empty() {
            return Err(ApiError::invalid_request(
                "the prompt tokenizes to 0 tokens",
            ));
        }
        Ok(tokens)
    }

    /// Builds the generation request: prompt tokens, `max_tokens` (default: the rest of the
    /// context), sampling defaults and priority. The size checks are the scheduler's.
    fn build(&self, req: &InferenceRequest) -> Result<GenerationRequest, ApiError> {
        if let Some(field) = unserved_phase2_field(&req.body) {
            return Err(ApiError::unsupported_parameter(field));
        }
        let prompt_tokens = self.prompt_tokens(req)?;
        let prompt_len = u32::try_from(prompt_tokens.len()).unwrap_or(u32::MAX);
        let body = &req.body;
        let max_tokens = body
            .max_tokens()
            .unwrap_or_else(|| self.max_seq_len.saturating_sub(prompt_len));
        Ok(GenerationRequest {
            id: req.id,
            n: 1,
            priority: Priority(body.priority.unwrap_or(0)),
            echo: false,
            constraint: None,
            deadline_ms: u64::MAX,
            endpoint: req.endpoint,
            http_request_id: req.http_request_id.clone(),
            prompt_tokens,
            sampling: SamplingParams {
                temperature: body.temperature.unwrap_or(self.defaults.temperature),
                top_p: body.top_p.unwrap_or(self.defaults.top_p),
                top_k: body.top_k.unwrap_or(self.defaults.top_k),
                seed: body.seed,
                logprobs: body.logprobs_n(req.endpoint),
                ..SamplingParams::default()
            },
            stop: StopConditions {
                eos_token_ids: self.eos_token_ids.clone(),
                stop_strings: body.stop_strings(),
                max_tokens,
                ignore_eos: body.ignore_eos == Some(true),
                ..StopConditions::default()
            },
        })
    }

    /// The HTTP error for a submission the scheduler refused.
    fn submit_error(&self, e: SubmitError, prompt_len: u32, max_tokens: u32) -> ApiError {
        match e {
            SubmitError::QueueFull => ApiError::queue_full(),
            SubmitError::ShuttingDown => ApiError::shutting_down(),
            SubmitError::ContextLengthExceeded => ApiError::context_length_exceeded(format!(
                "this model's maximum context length is {} tokens; the request has {prompt_len} \
                 prompt tokens and asks for {max_tokens} completion tokens",
                self.max_seq_len
            )),
            SubmitError::ContextExceedsKvCapacity => {
                let tokens = prompt_len.saturating_add(max_tokens);
                ApiError::context_exceeds_kv_capacity(format!(
                    "the request needs {} KV blocks of {} tokens at completion ({tokens} tokens); \
                     the KV pool holds {} blocks",
                    blocks_for_tokens(tokens, self.block_tokens),
                    self.block_tokens,
                    self.pool_blocks
                ))
            }
            SubmitError::PromptTooLong => ApiError::context_length_exceeded(format!(
                "the prompt has {prompt_len} tokens; without scheduler.chunked_prefill a prompt \
                 must fit scheduler.max_batch_tokens ({})",
                self.max_batch_tokens
            )),
        }
    }

    /// Validates the request and hands it to the engine; returns its event stream once the
    /// scheduler has queued it.
    async fn start(&self, req: InferenceRequest) -> Result<GenerationStream, ApiError> {
        let endpoint = req.endpoint;
        if self.state.load(Ordering::Acquire) == STATE_SHUTTING_DOWN {
            return Err(self.reject(endpoint, ApiError::shutting_down()));
        }
        let Some(loaded) = self.loaded.get().filter(|_| self.is_ready()) else {
            return Err(self.reject(endpoint, ApiError::model_not_loaded()));
        };
        let request = self.build(&req).map_err(|e| self.reject(endpoint, e))?;
        let prompt_len = u32::try_from(request.prompt_tokens.len()).unwrap_or(u32::MAX);
        let max_tokens = request.stop.max_tokens;
        let (events, stream) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
        let (ack, admitted) = oneshot::channel();
        // Waits for room in the bounded command channel, which the engine drains every turn.
        let sent = loaded
            .engine
            .submit_tx
            .send(EngineCommand::Submit(Box::new(request), events, ack))
            .await;
        let engine_gone = || {
            self.metrics.request(endpoint, Outcome::Failed);
            ApiError::internal("the engine thread has stopped")
        };
        if sent.is_err() {
            return Err(engine_gone());
        }
        match admitted.await {
            Ok(Ok(())) => Ok(stream),
            Ok(Err(e)) => Err(self.reject(endpoint, self.submit_error(e, prompt_len, max_tokens))),
            Err(_) => Err(engine_gone()),
        }
    }
}

impl InferenceBackend for ModelBackend {
    fn models(&self) -> Vec<ModelCard> {
        match self.loaded.get().filter(|_| self.is_serving()) {
            Some(loaded) => vec![ModelCard {
                id: self.served_name.clone(),
                object: "model".into(),
                created: loaded.created,
                owned_by: "turbine".into(),
                max_model_len: self.max_seq_len,
            }],
            None => Vec::new(),
        }
    }

    fn submit(&self, req: InferenceRequest) -> BoxFuture<'_, Result<GenerationStream, ApiError>> {
        Box::pin(self.start(req))
    }

    fn token_text(&self, token_id: u32) -> String {
        self.tokenizer
            .decode(&[token_id], false)
            .unwrap_or_else(|_| format!("token_id:{token_id}"))
    }

    fn record_rejection(&self, endpoint: Endpoint, code: ErrorCode) {
        let _ = code;
        self.metrics.request(endpoint, Outcome::Rejected);
    }
}

impl Readiness for ModelBackend {
    fn ready(&self) -> ReadyState {
        let reason = match self.state.load(Ordering::Acquire) {
            STATE_READY => return ReadyState::Ready,
            STATE_LOADING => NotReadyReason::LoadingModel,
            STATE_LOAD_FAILED => NotReadyReason::ModelLoadFailed,
            STATE_SHUTTING_DOWN => NotReadyReason::ShuttingDown,
            _ => NotReadyReason::DeviceError,
        };
        ReadyState::NotReady { reason }
    }
}

impl Diagnostics for ModelBackend {
    fn status(&self) -> Value {
        let loaded = self.loaded.get();
        serde_json::to_value(StatusDocument {
            version: env!("CARGO_PKG_VERSION"),
            uptime_seconds: self.started.elapsed().as_secs(),
            ready: self.is_ready(),
            device_count: self.device_count,
            model: ModelStatus {
                served_name: &self.served_name,
                architecture: self.architecture,
                weight_bytes: loaded.map_or(self.expected_weight_bytes, |l| l.weight_bytes),
                load_seconds: loaded.map(|l| l.load_seconds),
            },
        })
        .unwrap_or(Value::Null)
    }
    fn devices(&self) -> Value {
        self.devices.clone()
    }
    /// `Scheduler::snapshot` as of the engine's last step; 503 until the engine runs.
    fn scheduler(&self) -> Result<Value, ApiError> {
        let docs = self.engine_docs()?;
        serde_json::to_value(docs.scheduler).map_err(|e| ApiError::internal(e.to_string()))
    }
    /// `KvDocument::from_pool` as of the engine's last step; 503 until the engine runs.
    fn kv(&self) -> Result<Value, ApiError> {
        let docs = self.engine_docs()?;
        serde_json::to_value(docs.kv).map_err(|e| ApiError::internal(e.to_string()))
    }
    fn pressure(&self) -> Result<Value, ApiError> {
        Err(ApiError::not_implemented())
    }
}

impl ModelBackend {
    fn engine_docs(&self) -> Result<crate::engine::EngineDocs, ApiError> {
        self.loaded
            .get()
            .and_then(|l| l.shared.docs())
            .ok_or_else(ApiError::model_not_loaded)
    }
}

/// The first Phase 2 request field the API accepts but the engine does not honour yet, at a
/// non-default value. Such requests are refused with 400 `unsupported_parameter` instead of
/// having the field silently ignored (TS §21 rule 2); `priority` (served by the scheduler),
/// `user` and `parallel_tool_calls` pass. Plan Task 17 serves the rest and removes this check.
fn unserved_phase2_field(body: &OpenAiRequest) -> Option<&'static str> {
    let unserved = [
        ("n", body.n() > 1),
        (
            "presence_penalty",
            body.presence_penalty.is_some_and(|p| p != 0.0),
        ),
        (
            "frequency_penalty",
            body.frequency_penalty.is_some_and(|p| p != 0.0),
        ),
        (
            "repetition_penalty",
            body.repetition_penalty.is_some_and(|p| p != 1.0),
        ),
        ("logit_bias", !body.logit_bias().is_empty()),
        ("min_tokens", body.min_tokens.is_some_and(|m| m > 0)),
        (
            "stop_token_ids",
            body.stop_token_ids.as_ref().is_some_and(|t| !t.is_empty()),
        ),
        ("echo", body.echo == Some(true)),
        (
            "response_format",
            body.response_format
                .as_ref()
                .is_some_and(|f| *f != ResponseFormat::Text),
        ),
        ("tools", body.tool_choice_mode() != ToolChoiceMode::None),
        (
            "messages.tool_calls / tool messages",
            body.messages
                .iter()
                .flatten()
                .any(|m| m.role == "tool" || m.tool_calls.as_ref().is_some_and(|c| !c.is_empty())),
        ),
    ];
    unserved
        .into_iter()
        .find_map(|(field, set)| set.then_some(field))
}
