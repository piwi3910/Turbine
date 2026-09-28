//! [`ModelBackend`] answers the API (`InferenceBackend`, `Readiness`, `Diagnostics`): it renders
//! and tokenizes a request, builds its [`GenerationRequest`] and forwards it to the engine thread
//! (`crate::engine`), whose scheduler checks it (P2 §Scheduling rules: `context_length_exceeded`,
//! `context_exceeds_kv_capacity`, `shutting_down`) and whose admission gate decides it (P3 reject
//! table: `queue_full` 429, `overloaded` / `circuit_open` 503, with `Retry-After`) — plain HTTP
//! errors returned before any event.
//!
//! `/ready` follows the circuit breaker (503 `circuit_open` while it is CIRCUIT_OPEN, DRAINING
//! or PROBING, and after a fatal circuit until the process exits); `/turbine/v1/pressure` is the
//! pressure controller's document and `/turbine/v1/status` names its state and circuit.
//!
//! Before queueing, a `response_format` or a `required`/named `tool_choice` is compiled into one
//! matcher per choice off the engine thread (`engine::grammar`); a grammar that fails, is too
//! large or too slow is 400 `invalid_json_schema`. `tools` on a model without a tool-call parser
//! is 400 `tools_not_supported`; `tool_choice: "none"` renders the template without tools.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};
use turbine_api::kv::PrefetchTarget;
use turbine_api::openai::request::{OpenAiRequest, PromptInput, ResponseFormat, ToolChoiceMode};
use turbine_api::{
    ApiError, BoxFuture, Diagnostics, GenerationStream, InferenceBackend, InferenceRequest,
    ModelCard, NotReadyReason, PrefetchAccepted, PrefetchRequest, Readiness, ReadyState,
    TopologyScope, readiness_for_circuit,
};
use turbine_core::request::{
    ConstraintSpec, Endpoint, ErrorCode, GenerationRequest, SamplingParams, SessionHints,
    StopConditions,
};
use turbine_core::support::SupportRowView;
use turbine_core::types::{CircuitState, PressureState, Priority};
use turbine_device::DeviceInventory;
use turbine_device::topology::TopologyGraph;
use turbine_kernels::Selection;
use turbine_kv::blocks_for_tokens;
use turbine_kv::hierarchy::PrefetchError;
use turbine_model::{ChatTemplate, Tokenizer, ToolChoice};
use turbine_reliability::budget::PoolKind;
use turbine_reliability::controller::ControllerHandle;
use turbine_scheduler::{SchedulerMetrics, SubmitError};

use crate::engine::grammar::GrammarService;
use crate::engine::{
    EVENT_CHANNEL_CAPACITY, EngineCommand, EngineHandle, EngineMetrics, EngineShared, Fatal,
    Submission, ToolOutput, ToolParser,
};
use crate::kv_orchestrator::{PrefetchRefused, PrefetchTargetOwned};
use crate::metrics::{Outcome, ServerMetrics};
use crate::model::PreparedModel;
use crate::modules::ModuleChoices;
use crate::reliability::api_error_for;

const STATE_LOADING: u8 = 0;
const STATE_READY: u8 = 1;
const STATE_LOAD_FAILED: u8 = 2;
/// The circuit is fatal (P3 S-12): `/ready` answers `circuit_open` until the process exits 3.
const STATE_DEVICE_FATAL: u8 = 3;
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
    /// The module picked at each extension point (Phase 2m).
    modules: &'a ModuleChoices,
    /// Per op config of the kernel registry, in requirement order: who serves it and why.
    kernels: &'a [KernelChoiceView],
    /// The support-matrix row resolved at startup (Phase 2m S-11).
    #[serde(skip_serializing_if = "Option::is_none")]
    support: Option<&'a SupportRowView>,
    /// P3 S-13: the pressure state and the circuit state (GREEN / HEALTHY before the load).
    pressure_state: PressureState,
    circuit_state: CircuitState,
}

/// One entry of `kernels` in `GET /turbine/v1/status`: a `KernelRegistry` selection.
#[derive(Clone, Debug, Serialize)]
pub struct KernelChoiceView {
    pub op: String,
    pub config: String,
    pub provider: String,
    /// The implementation that runs (for a tiered op, its first tier's).
    pub implementation: String,
    /// The implementation's family (`hipblaslt`, `ck`, `turbine_hip`), or the provider for a
    /// provider that chooses internally.
    pub impl_provider: String,
    pub reason: String,
    /// `profile_preferred`, `profile_fallback`, `library_order` or `provider_internal`.
    pub reason_code: String,
    /// Routed-row tiers of a tiered op (`moe_experts`), empty otherwise.
    pub tiers: Vec<KernelTierView>,
}

/// One routed-row tier of a [`KernelChoiceView`]: up to `max_rows` rows (`null` = no bound).
#[derive(Clone, Debug, Serialize)]
pub struct KernelTierView {
    pub max_rows: Option<u32>,
    pub implementation: String,
}

impl From<&Selection> for KernelChoiceView {
    fn from(s: &Selection) -> KernelChoiceView {
        KernelChoiceView {
            op: s.op.as_str().to_string(),
            config: s.config.clone(),
            provider: s.provider.0.to_string(),
            implementation: s.implementation.clone(),
            impl_provider: s.impl_provider.clone(),
            reason: s.reason.clone(),
            reason_code: s.reason_code.to_string(),
            tiers: s
                .tiers
                .iter()
                .map(|(max_rows, implementation)| KernelTierView {
                    max_rows: *max_rows,
                    implementation: implementation.clone(),
                })
                .collect(),
        }
    }
}

/// Set once the model is loaded and warmed up.
struct Loaded {
    engine: EngineHandle,
    shared: Arc<EngineShared>,
    /// The pressure controller's snapshot: circuit, state and the pressure document.
    controller: ControllerHandle,
    load_seconds: f64,
    weight_bytes: u64,
    created: u64,
}

/// The server's `InferenceBackend`, `Readiness` and `Diagnostics`.
pub struct ModelBackend {
    state: AtomicU8,
    loaded: OnceLock<Loaded>,
    served_name: String,
    architecture: String,
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
    /// Compiles `response_format` and tool grammars before queueing.
    grammars: GrammarService,
    /// The format of `model.tool_call_parser`; `None` when it resolved to `none`.
    tool_parser: Option<ToolParser>,
    metrics: ServerMetrics,
    /// `turbine_admission_total` for `invalid_json_schema` (refused before the scheduler).
    admission: SchedulerMetrics,
    started: Instant,
    device_count: u64,
    devices: Value,
    modules: ModuleChoices,
    kernels: Vec<KernelChoiceView>,
    /// The support-matrix row resolved at startup (`support` of the status document).
    support: Option<SupportRowView>,
    /// The node topology graph captured at startup (`GET /turbine/v1/topology`, P5 S-1).
    topology: Option<Value>,
}

impl ModelBackend {
    /// A backend in the `loading_model` state for `model`.
    pub fn new(
        model: &PreparedModel,
        inventory: &DeviceInventory,
        metrics: &EngineMetrics,
    ) -> ModelBackend {
        let eos_token_ids = crate::model::eos_token_ids(&model.arch, &model.generation)
            .into_iter()
            .collect();
        let tool_parser = model.tool_format.as_ref().map(|bound| ToolParser {
            format: Arc::clone(bound),
            parser: Arc::from(bound.format.parser()),
            label: bound.format.name(),
        });
        ModelBackend {
            state: AtomicU8::new(STATE_LOADING),
            loaded: OnceLock::new(),
            served_name: model.served_name.clone(),
            architecture: model.arch.hf_architecture.clone(),
            expected_weight_bytes: model.budget.pool(PoolKind::Weights),
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
            grammars: GrammarService::new(
                Arc::clone(&model.grammar),
                &model.structured_output,
                metrics.model.clone(),
            ),
            tool_parser,
            metrics: metrics.server.clone(),
            admission: metrics.scheduler.clone(),
            started: Instant::now(),
            device_count: inventory.devices.len() as u64,
            devices: serde_json::to_value(inventory).unwrap_or(Value::Null),
            modules: model.modules.clone(),
            kernels: model
                .registry
                .selections()
                .iter()
                .map(KernelChoiceView::from)
                .collect(),
            support: None,
            topology: None,
        }
    }

    /// Serves `graph` at `GET /turbine/v1/topology` (captured once at startup).
    pub fn with_topology(mut self, graph: &TopologyGraph) -> ModelBackend {
        self.topology = serde_json::to_value(graph).ok();
        self
    }

    /// Reports `support` (the support-matrix row resolved at startup) in the status document.
    pub fn with_support(mut self, support: SupportRowView) -> ModelBackend {
        self.support = Some(support);
        self
    }

    /// Weights loaded and warmed up: `/ready` turns 200 and requests are accepted.
    pub fn set_ready(
        &self,
        engine: EngineHandle,
        shared: Arc<EngineShared>,
        controller: ControllerHandle,
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
                controller,
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
            Fatal::DeviceFatal(_) => STATE_DEVICE_FATAL,
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

    /// Prompt token ids: the rendered chat template (its text carries the BOS token; `tools`
    /// rendered when given), the tokenized prompt string (`add_special_tokens`), or the given
    /// ids.
    fn prompt_tokens(
        &self,
        req: &InferenceRequest,
        tools: Option<&[Value]>,
    ) -> Result<Vec<u32>, ApiError> {
        let tokens = match req.endpoint {
            Endpoint::ChatCompletions => {
                let empty = serde_json::Map::new();
                let kwargs = req.body.chat_template_kwargs.as_ref().unwrap_or(&empty);
                let text = self
                    .template
                    .render(&req.body.messages_json(), tools, true, kwargs)
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

    /// Builds the submission: prompt tokens (tools rendered unless `tool_choice` is `none`),
    /// `max_tokens` (default: the rest of the context), sampling defaults and the Phase 2
    /// fields, the constraint to compile and the tool-call handling. The size checks are the
    /// scheduler's.
    fn build(&self, req: &InferenceRequest) -> Result<Submission, ApiError> {
        let body = &req.body;
        let mode = body.tool_choice_mode();
        let tools = body.tools_json();
        let parser = match (&self.tool_parser, tools.is_empty()) {
            (_, true) => None,
            (Some(p), false) => Some(p.clone()),
            (None, false) => return Err(ApiError::tools_not_supported(&self.served_name)),
        };
        let render_tools = (mode != ToolChoiceMode::None).then_some(tools.as_slice());
        let prompt_tokens = self.prompt_tokens(req, render_tools)?;
        let prompt_len = u32::try_from(prompt_tokens.len()).unwrap_or(u32::MAX);
        let logit_bias = body.logit_bias();
        let stop_token_ids = body.stop_token_ids.clone().unwrap_or_default();
        for (field, id) in logit_bias
            .iter()
            .map(|&(id, _)| ("logit_bias", id))
            .chain(stop_token_ids.iter().map(|&id| ("stop_token_ids", id)))
        {
            if id >= self.vocab_size {
                return Err(ApiError::invalid_request(format!(
                    "{field}: token id {id} is outside the vocabulary (size {})",
                    self.vocab_size
                )));
            }
        }
        let tool_choice = match &mode {
            ToolChoiceMode::None => ToolChoice::None,
            ToolChoiceMode::Auto => ToolChoice::Auto,
            ToolChoiceMode::Required => ToolChoice::Required,
            ToolChoiceMode::Named(name) => ToolChoice::Named(name.clone()),
        };
        let (constraint, tool_output) = match (&tool_choice, parser) {
            (ToolChoice::Auto | ToolChoice::Required | ToolChoice::Named(_), Some(p)) => {
                // `auto` is held to free text or schema-valid calls; it is still held and
                // parsed only when it opens like a call.
                let spec = p
                    .format
                    .format
                    .grammar(&tools, &tool_choice, body.parallel_tool_calls_enabled())
                    .map_err(|e| ApiError::invalid_json_schema(e.to_string()))?;
                let output = match tool_choice {
                    ToolChoice::Auto => ToolOutput::Auto(p),
                    _ => ToolOutput::Constrained(p),
                };
                (Some(spec), output)
            }
            _ => (
                response_constraint(body.response_format.as_ref()),
                ToolOutput::None,
            ),
        };
        let echo = req.endpoint == Endpoint::Completions && body.echo == Some(true);
        let echo_text = if echo {
            Some(match &body.prompt {
                Some(PromptInput::Text(text)) => text.clone(),
                _ => self
                    .tokenizer
                    .decode(&prompt_tokens, true)
                    .map_err(|e| ApiError::invalid_request(e.to_string()))?,
            })
        } else {
            None
        };
        let max_tokens = body
            .max_tokens()
            .unwrap_or_else(|| self.max_seq_len.saturating_sub(prompt_len));
        let request = GenerationRequest {
            id: req.id,
            n: body.n().max(1),
            priority: Priority(body.priority.unwrap_or(0)),
            echo,
            constraint,
            deadline_ms: u64::MAX,
            session: body
                .prompt_cache_key
                .clone()
                .map(|session_id| SessionHints {
                    session_id,
                    resume_within_secs: req.hints.session_resume_within,
                    end: req.hints.session_end,
                }),
            cache_salt: req.hints.cache_salt.clone(),
            endpoint: req.endpoint,
            http_request_id: req.http_request_id.clone(),
            prompt_tokens,
            sampling: SamplingParams {
                temperature: body.temperature.unwrap_or(self.defaults.temperature),
                top_p: body.top_p.unwrap_or(self.defaults.top_p),
                top_k: body.top_k.unwrap_or(self.defaults.top_k),
                seed: body.seed,
                presence_penalty: body.presence_penalty.unwrap_or(0.0),
                frequency_penalty: body.frequency_penalty.unwrap_or(0.0),
                repetition_penalty: body.repetition_penalty.unwrap_or(1.0),
                logit_bias,
                min_tokens: body.min_tokens.unwrap_or(0),
                logprobs: body.logprobs_n(req.endpoint),
            },
            stop: StopConditions {
                eos_token_ids: self.eos_token_ids.clone(),
                stop_strings: body.stop_strings(),
                stop_token_ids,
                max_tokens,
                ignore_eos: body.ignore_eos == Some(true),
            },
        };
        Ok(Submission {
            request,
            matchers: Vec::new(),
            tools: tool_output,
            echo_text,
        })
    }

    /// Compiles the submission's constraint into one matcher per choice (off the engine
    /// thread, bounded); a failure is 400 `invalid_json_schema`, counted as an admission
    /// rejection.
    async fn compile(&self, submission: &mut Submission) -> Result<(), ApiError> {
        let Some(spec) = &submission.request.constraint else {
            return Ok(());
        };
        match self.grammars.compile(spec, submission.request.n).await {
            Ok(matchers) => {
                submission.matchers = matchers;
                Ok(())
            }
            Err(e) => {
                if e.code == ErrorCode::InvalidJsonSchema {
                    self.admission.record_rejection(e.code.as_str());
                }
                Err(e)
            }
        }
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
            // P3 admission gate: the reject table.
            SubmitError::Rejected {
                reason,
                retry_after_secs,
            } => api_error_for(reason, retry_after_secs),
            _ => ApiError::internal(format!("submission refused: {e}")),
        }
    }

    /// `POST /turbine/v1/kv/prefetch` (P4 S-10): a prompt or messages body is tokenized like
    /// the OpenAI routes (the chat template with the generation prompt), then the engine queues
    /// the promotions of its cached blocks.
    async fn prefetch_blocks(&self, req: PrefetchRequest) -> Result<PrefetchAccepted, ApiError> {
        let Some(loaded) = self.loaded.get().filter(|_| self.is_ready()) else {
            return Err(ApiError::model_not_loaded());
        };
        let cache_salt = req.cache_salt.unwrap_or_default();
        let target = match req.target {
            PrefetchTarget::Session { session_id } => PrefetchTargetOwned::Session(session_id),
            PrefetchTarget::Prompt { prompt } => PrefetchTargetOwned::Tokens {
                prompt: self
                    .tokenizer
                    .encode(&prompt, true)
                    .map_err(|e| ApiError::invalid_request(e.to_string()))?,
                cache_salt,
            },
            PrefetchTarget::Messages { messages } => {
                let body = OpenAiRequest {
                    messages: Some(messages),
                    ..OpenAiRequest::default()
                };
                let text = self
                    .template
                    .render(&body.messages_json(), None, true, &serde_json::Map::new())
                    .map_err(|e| ApiError::template_error(e.to_string()))?;
                PrefetchTargetOwned::Tokens {
                    prompt: self
                        .tokenizer
                        .encode(&text, false)
                        .map_err(|e| ApiError::invalid_request(e.to_string()))?,
                    cache_salt,
                }
            }
        };
        match loaded.engine.kv.prefetch(target).await {
            Ok(a) => Ok(PrefetchAccepted {
                blocks_queued: a.blocks_queued,
                blocks_resident: a.blocks_resident,
            }),
            Err(PrefetchRefused::Kv(PrefetchError::SessionNotFound)) => {
                Err(ApiError::session_not_found())
            }
            Err(PrefetchRefused::Kv(PrefetchError::QueueFull)) => {
                Err(ApiError::prefetch_queue_full())
            }
            Err(PrefetchRefused::Kv(PrefetchError::PressureTooHigh)) => {
                Err(ApiError::pressure_too_high())
            }
            Err(PrefetchRefused::EngineGone) => {
                Err(ApiError::internal("the engine thread has stopped"))
            }
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
        let mut submission = self.build(&req).map_err(|e| self.reject(endpoint, e))?;
        self.compile(&mut submission)
            .await
            .map_err(|e| self.reject(endpoint, e))?;
        let prompt_len = u32::try_from(submission.request.prompt_tokens.len()).unwrap_or(u32::MAX);
        let max_tokens = submission.request.stop.max_tokens;
        let (events, stream) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
        let (ack, admitted) = oneshot::channel();
        // Waits for room in the bounded command channel, which the engine drains every turn.
        let sent = loaded
            .engine
            .submit_tx
            .send(EngineCommand::Submit(Box::new(submission), events, ack))
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

    fn prefetch(&self, req: PrefetchRequest) -> BoxFuture<'_, Result<PrefetchAccepted, ApiError>> {
        Box::pin(self.prefetch_blocks(req))
    }
}

impl Readiness for ModelBackend {
    fn ready(&self) -> ReadyState {
        let reason = match self.state.load(Ordering::Acquire) {
            STATE_READY => {
                let circuit = self
                    .loaded
                    .get()
                    .map_or(CircuitState::Healthy, |l| l.controller.circuit());
                return readiness_for_circuit(circuit, ReadyState::Ready);
            }
            STATE_LOADING => NotReadyReason::LoadingModel,
            STATE_LOAD_FAILED => NotReadyReason::ModelLoadFailed,
            STATE_SHUTTING_DOWN => NotReadyReason::ShuttingDown,
            _ => NotReadyReason::CircuitOpen,
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
                architecture: &self.architecture,
                weight_bytes: loaded.map_or(self.expected_weight_bytes, |l| l.weight_bytes),
                load_seconds: loaded.map(|l| l.load_seconds),
            },
            modules: &self.modules,
            kernels: &self.kernels,
            support: self.support.as_ref(),
            pressure_state: loaded.map_or(PressureState::Green, |l| l.controller.state()),
            circuit_state: loaded.map_or(CircuitState::Healthy, |l| l.controller.circuit()),
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
    /// The KV hierarchy's document (P4 §Data) as of the engine's last step; 503 until the
    /// engine runs.
    fn kv(&self) -> Result<Value, ApiError> {
        let docs = self.engine_docs()?;
        serde_json::to_value(docs.kv).map_err(|e| ApiError::internal(e.to_string()))
    }
    /// The pressure controller's document (P3 §Data); 503 until the engine runs.
    fn pressure(&self) -> Result<Value, ApiError> {
        let loaded = self.loaded.get().ok_or_else(ApiError::model_not_loaded)?;
        serde_json::to_value(loaded.controller.document())
            .map_err(|e| ApiError::internal(e.to_string()))
    }
    /// The graph captured at startup; only the node scope exists before Phase 6.
    fn topology(&self, scope: TopologyScope) -> Result<Value, ApiError> {
        match (scope, &self.topology) {
            (TopologyScope::Node, Some(graph)) => Ok(graph.clone()),
            _ => Err(ApiError::not_implemented()),
        }
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

/// The constraint of a `response_format` (`text` or none: unconstrained).
fn response_constraint(format: Option<&ResponseFormat>) -> Option<ConstraintSpec> {
    match format? {
        ResponseFormat::Text => None,
        ResponseFormat::JsonObject => Some(ConstraintSpec::JsonObject),
        ResponseFormat::JsonSchema { json_schema } => Some(ConstraintSpec::JsonSchema {
            // The API requires `schema`; a missing one constrains to any JSON value.
            schema: json_schema.schema.clone().unwrap_or(Value::Bool(true)),
        }),
    }
}
