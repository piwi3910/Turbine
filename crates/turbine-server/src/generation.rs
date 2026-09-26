//! The Phase 1 engine: [`ModelBackend`] answers the API (`InferenceBackend`, `Readiness`,
//! `Diagnostics`), and one generation thread owns the executor and runs one request at a time.
//!
//! The single slot is an `AtomicBool` claimed in `submit` (a second request gets 429
//! `engine_busy`) and released by the generation thread when the request ends or is cancelled.
//! The request's event channel holds at most 64 events; a closed receiver (client disconnect)
//! is the cancellation signal.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
use std::sync::{Arc, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::Value;
use tokio::sync::mpsc;
use turbine_api::openai::request::{OpenAiRequest, PromptInput, ResponseFormat, ToolChoiceMode};
use turbine_api::{
    ApiError, BoxFuture, Diagnostics, GenerationStream, InferenceBackend, InferenceRequest,
    ModelCard, NotReadyReason, Readiness, ReadyState,
};
use turbine_core::request::{
    CancelFlag, Endpoint, ErrorCode, GenerationEvent, GenerationRequest, SamplingParams,
    StopConditions,
};
use turbine_device::DeviceInventory;
use turbine_model::executor::ModelExecutor;
use turbine_model::{ChatTemplate, GenerateOptions, ModelMetrics, Tokenizer, generate};

use crate::metrics::{Outcome, ServerMetrics, TokenKind};
use crate::model::PreparedModel;

/// Events buffered per request between the generation thread and the HTTP response (P1 bound).
pub const EVENT_CHANNEL_CAPACITY: usize = 64;
/// Consecutive failed requests after which the server reports `device_error` and exits 1 (C-25).
pub const MAX_CONSECUTIVE_FAILURES: u32 = 3;

const STATE_LOADING: u8 = 0;
const STATE_READY: u8 = 1;
const STATE_LOAD_FAILED: u8 = 2;
const STATE_DEVICE_ERROR: u8 = 3;

/// Why the engine asks the server to exit 1.
#[derive(Debug)]
pub enum Fatal {
    /// Weight load or warm-up failed after the listener bound.
    LoadFailed(String),
    /// `MAX_CONSECUTIVE_FAILURES` requests in a row failed.
    DeviceError(String),
}

/// One accepted request on its way to the generation thread.
pub struct Job {
    pub request: GenerationRequest,
    pub events: mpsc::Sender<GenerationEvent>,
    /// When `submit` accepted it: the start of TTFT and end-to-end latency.
    pub arrived: Instant,
}

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
    jobs: SyncSender<Job>,
    load_seconds: f64,
    weight_bytes: u64,
    created: u64,
}

/// The server's `InferenceBackend`, `Readiness` and `Diagnostics`.
pub struct ModelBackend {
    state: AtomicU8,
    slot: Arc<AtomicBool>,
    loaded: OnceLock<Loaded>,
    served_name: String,
    architecture: &'static str,
    expected_weight_bytes: u64,
    max_seq_len: u32,
    vocab_size: u32,
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
            slot: Arc::new(AtomicBool::new(false)),
            loaded: OnceLock::new(),
            served_name: model.served_name.clone(),
            architecture: model.arch.architecture.as_str(),
            expected_weight_bytes: model.budget.weights,
            max_seq_len: model.max_seq_len,
            vocab_size: model.arch.vocab_size,
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

    /// The slot flag the generation thread releases.
    pub fn slot(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.slot)
    }

    pub fn metrics(&self) -> &ServerMetrics {
        &self.metrics
    }

    /// Weights loaded and warm-up done: `/ready` turns 200 and requests are accepted.
    pub fn set_ready(&self, jobs: SyncSender<Job>, load_seconds: f64, weight_bytes: u64) {
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        if self
            .loaded
            .set(Loaded {
                jobs,
                load_seconds,
                weight_bytes,
                created,
            })
            .is_ok()
        {
            self.state.store(STATE_READY, Ordering::Release);
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

    fn is_ready(&self) -> bool {
        self.state.load(Ordering::Acquire) == STATE_READY
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

    /// Builds the generation request: prompt tokens, context-length check, sampling defaults.
    fn build(&self, req: &InferenceRequest) -> Result<GenerationRequest, ApiError> {
        if let Some(field) = unserved_phase2_field(&req.body) {
            return Err(ApiError::unsupported_parameter(field));
        }
        let prompt_tokens = self.prompt_tokens(req)?;
        let prompt_len = u32::try_from(prompt_tokens.len()).unwrap_or(u32::MAX);
        let remaining = self.max_seq_len.saturating_sub(prompt_len);
        let max_tokens = req.body.max_tokens().unwrap_or(remaining);
        if prompt_len > self.max_seq_len || max_tokens > remaining {
            return Err(ApiError::context_length_exceeded(format!(
                "this model's maximum context length is {} tokens; the request has {prompt_len} \
                 prompt tokens and asks for {max_tokens} completion tokens",
                self.max_seq_len
            )));
        }
        let body = &req.body;
        Ok(GenerationRequest {
            id: req.id,
            n: 1,
            priority: turbine_core::types::Priority::default(),
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

    /// Validates, claims the slot and hands the request to the generation thread.
    fn start(&self, req: InferenceRequest) -> Result<GenerationStream, ApiError> {
        let endpoint = req.endpoint;
        let Some(loaded) = self.loaded.get().filter(|_| self.is_ready()) else {
            return Err(self.reject(endpoint, ApiError::model_not_loaded()));
        };
        let request = self.build(&req).map_err(|e| self.reject(endpoint, e))?;
        if self
            .slot
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(self.reject(endpoint, ApiError::engine_busy()));
        }
        let (tx, rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
        let job = Job {
            request,
            events: tx,
            arrived: Instant::now(),
        };
        match loaded.jobs.try_send(job) {
            Ok(()) => Ok(rx),
            Err(TrySendError::Full(_)) => {
                // The thread has not yet taken the previous job although it released the slot:
                // impossible by construction (it releases after dequeuing), but never block here.
                self.slot.store(false, Ordering::Release);
                Err(self.reject(endpoint, ApiError::engine_busy()))
            }
            Err(TrySendError::Disconnected(_)) => {
                self.slot.store(false, Ordering::Release);
                self.metrics.request(endpoint, Outcome::Failed);
                Err(ApiError::internal("the generation thread has stopped"))
            }
        }
    }
}

impl InferenceBackend for ModelBackend {
    fn models(&self) -> Vec<ModelCard> {
        match self.loaded.get().filter(|_| self.is_ready()) {
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
        Box::pin(async move { self.start(req) })
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
    fn scheduler(&self) -> Result<Value, ApiError> {
        Err(ApiError::not_implemented())
    }
    fn kv(&self) -> Result<Value, ApiError> {
        Err(ApiError::not_implemented())
    }
    fn pressure(&self) -> Result<Value, ApiError> {
        Err(ApiError::not_implemented())
    }
}

/// What the generation thread needs besides its executor.
pub struct Engine {
    pub tokenizer: Arc<Tokenizer>,
    pub max_seq_len: u32,
    pub slot: Arc<AtomicBool>,
    pub metrics: ServerMetrics,
    pub model_metrics: ModelMetrics,
}

impl Engine {
    /// Serves jobs until the sender is dropped. Returns early with the failure message once
    /// `MAX_CONSECUTIVE_FAILURES` requests in a row failed.
    pub fn run(&self, exec: &mut dyn ModelExecutor, jobs: &Receiver<Job>) -> Result<(), String> {
        let mut consecutive_failures = 0u32;
        while let Ok(job) = jobs.recv() {
            let (outcome, detail) = self.serve(exec, job);
            match outcome {
                Outcome::Failed => {
                    consecutive_failures += 1;
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                        return Err(format!(
                            "{consecutive_failures} consecutive requests failed; last error: \
                             {detail}"
                        ));
                    }
                }
                Outcome::Ok => consecutive_failures = 0,
                Outcome::Cancelled | Outcome::Rejected => {}
            }
        }
        Ok(())
    }

    /// Runs one request to its end; returns its outcome and, for a failure, the error message.
    /// The request is accounted and the slot released *before* its final event is sent, so a
    /// client that has read the whole response can start the next request at once and sees
    /// its metrics.
    fn serve(&self, exec: &mut dyn ModelExecutor, job: Job) -> (Outcome, String) {
        let Job {
            request,
            events,
            arrived,
        } = job;
        let cancel = CancelFlag::default();
        self.metrics
            .add_tokens(TokenKind::Prompt, request.prompt_tokens.len() as u64);
        let mut outcome = None;
        let mut detail = String::new();
        let mut generated = 0u64;
        let mut last_token: Option<Instant> = None;
        let opts = GenerateOptions {
            max_seq_len: self.max_seq_len,
            metrics: Some(&self.model_metrics),
        };
        for event in generate(exec, Arc::clone(&self.tokenizer), &request, &cancel, opts) {
            match &event {
                GenerationEvent::Token { .. } => {
                    let now = Instant::now();
                    match last_token {
                        None => self.metrics.ttft.observe((now - arrived).as_secs_f64()),
                        Some(prev) => self.metrics.itl.observe((now - prev).as_secs_f64()),
                    }
                    last_token = Some(now);
                    generated += 1;
                }
                GenerationEvent::Finished { .. } => {
                    outcome = Some(Outcome::Ok);
                    self.finish(&request, arrived, Outcome::Ok, generated, &detail);
                }
                GenerationEvent::Error { message, .. } => {
                    outcome = Some(Outcome::Failed);
                    detail.clone_from(message);
                    self.finish(&request, arrived, Outcome::Failed, generated, &detail);
                }
                _ => {}
            }
            if events.blocking_send(event).is_err() {
                cancel.cancel();
                break;
            }
        }
        if let Some(outcome) = outcome {
            return (outcome, detail);
        }
        let outcome = if cancel.is_cancelled() {
            Outcome::Cancelled
        } else {
            detail = "generation ended without a finish event".into();
            Outcome::Failed
        };
        self.finish(&request, arrived, outcome, generated, &detail);
        (outcome, detail)
    }

    /// Accounts a finished request (tokens, latency, outcome, log) and frees the slot.
    fn finish(
        &self,
        request: &GenerationRequest,
        arrived: Instant,
        outcome: Outcome,
        generated: u64,
        detail: &str,
    ) {
        self.metrics.add_tokens(TokenKind::Generated, generated);
        if outcome != Outcome::Cancelled {
            self.metrics.e2e.observe(arrived.elapsed().as_secs_f64());
        }
        self.metrics.request(request.endpoint, outcome);
        tracing::info!(
            event = "request_finished",
            request_id = %request.http_request_id,
            endpoint = request.endpoint.as_str(),
            outcome = outcome.as_str(),
            prompt_tokens = request.prompt_tokens.len(),
            generated_tokens = generated,
            e2e_seconds = arrived.elapsed().as_secs_f64(),
            "request finished"
        );
        if outcome == Outcome::Failed {
            tracing::error!(request_id = %request.http_request_id, error = %detail, "request failed");
        }
        self.slot.store(false, Ordering::Release);
    }
}

/// The first Phase 2 request field the API accepts but this single-slot engine does not honour
/// yet, at a non-default value. Such requests are refused with 400 `unsupported_parameter`
/// instead of having the field silently ignored (TS §21 rule 2); `priority`, `user` and
/// `parallel_tool_calls` change nothing for one slot and pass. The Phase 2 engine (plan Tasks
/// 15 and 17) serves them all and removes this check.
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

#[cfg(test)]
mod tests {
    use std::sync::mpsc::sync_channel;

    use turbine_core::types::{KvLayout, ModelShape, RequestId};
    use turbine_kernels::KernelError;
    use turbine_model::ModelError;
    use turbine_model::executor::{BatchInput, Logits};
    use turbine_model::testing::TempDir;
    use turbine_model::testing::tiny::write_tiny_llama;
    use turbine_observability::MetricsRegistry;

    use super::*;

    /// An executor whose every forward fails like a device error.
    struct FailingExecutor {
        shape: ModelShape,
        kv: KvLayout,
    }

    impl ModelExecutor for FailingExecutor {
        fn shape(&self) -> &ModelShape {
            &self.shape
        }
        fn kv_layout(&self) -> &KvLayout {
            &self.kv
        }
        fn forward(&mut self, _batch: &BatchInput<'_>) -> Result<Logits, ModelError> {
            Err(ModelError::Kernel(KernelError::Device {
                message: "hipErrorOutOfMemory: injected".into(),
            }))
        }
    }

    fn job(events: mpsc::Sender<GenerationEvent>) -> Job {
        Job {
            request: GenerationRequest {
                id: RequestId::new_v4(),
                n: 1,
                priority: turbine_core::types::Priority::default(),
                echo: false,
                constraint: None,
                deadline_ms: u64::MAX,
                endpoint: Endpoint::Completions,
                http_request_id: "test".into(),
                prompt_tokens: vec![1, 2, 3],
                sampling: SamplingParams::default(),
                stop: StopConditions {
                    max_tokens: 4,
                    ..StopConditions::default()
                },
            },
            events,
            arrived: Instant::now(),
        }
    }

    #[test]
    fn three_consecutive_failures_stop_the_engine() {
        let dir = TempDir::new("turbine-server-engine");
        let spec = write_tiny_llama(dir.path(), 7);
        let tokenizer = Arc::new(Tokenizer::from_file(&dir.path().join("tokenizer.json")).unwrap());
        let reg = MetricsRegistry::new();
        let slot = Arc::new(AtomicBool::new(false));
        let engine = Engine {
            tokenizer,
            max_seq_len: 64,
            slot: Arc::clone(&slot),
            metrics: ServerMetrics::register(&reg),
            model_metrics: ModelMetrics::register(&reg),
        };
        let mut exec = FailingExecutor {
            shape: spec.config.shape(),
            kv: spec.config.kv_layout(1),
        };
        let (jobs_tx, jobs_rx) = sync_channel(4);
        let mut receivers = Vec::new();
        for _ in 0..4 {
            let (tx, rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
            jobs_tx.send(job(tx)).unwrap();
            receivers.push(rx);
        }
        slot.store(true, Ordering::Release);

        let err = engine.run(&mut exec, &jobs_rx).unwrap_err();
        assert!(err.contains("3 consecutive requests failed"), "{err}");
        assert!(err.contains("hipErrorOutOfMemory"), "{err}");
        assert!(!slot.load(Ordering::Acquire), "the slot is released");
        // The first three requests each saw an error event; the fourth was never started.
        for rx in &mut receivers[..3] {
            let mut saw_error = false;
            while let Ok(event) = rx.try_recv() {
                saw_error |= matches!(
                    event,
                    GenerationEvent::Error {
                        code: ErrorCode::InternalError,
                        ..
                    }
                );
            }
            assert!(saw_error);
        }
        assert!(receivers[3].try_recv().is_err());
        let text = reg.render().unwrap();
        assert!(
            text.contains(
                r#"turbine_requests_total{endpoint="/v1/completions",outcome="failed"} 3"#
            ),
            "{text}"
        );
    }
}
