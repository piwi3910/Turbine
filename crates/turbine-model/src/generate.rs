//! The Phase 1 single-request generation loop (P1 S-9): prefill, then one decode step per
//! token, host sampling, stop conditions and cancellation between steps. Phase 2's engine loop
//! replaces it.
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use turbine_core::request::{
    CancelFlag, ErrorCode, FinishReason, GenerationEvent, GenerationRequest, StopConditions, Usage,
};

use crate::executor::{ModelExecutor, SequenceKv};
use crate::metrics::{ForwardPhase, ModelMetrics};
use crate::sampler::Sampler;
use crate::tokenizer::{IncrementalDetokenizer, Tokenizer};

/// Per-call settings of [`generate`] that are not part of the request.
#[derive(Clone, Copy, Debug)]
pub struct GenerateOptions<'a> {
    /// The executor's context limit: generation stops with `length` when prompt + generated
    /// tokens reach it.
    pub max_seq_len: u32,
    /// Where forward durations go (`turbine_forward_seconds{phase}`); `None` records nothing.
    pub metrics: Option<&'a ModelMetrics>,
}

/// Runs one request to completion as an iterator of events: `Started`, one `Token` per
/// generated token (EOS included; `text` is empty while held back), then `Finished` — or
/// `Error { code: InternalError }` when a forward pass fails. When `cancel` fires the stream
/// ends without `Finished`; it is checked before every forward pass.
///
/// The sequence's K/V lives in `kv` (a single-sequence pool; its length bounds the context
/// together with `opts.max_seq_len`). Each `next` runs at most one forward pass, so dropping the
/// iterator stops the work.
pub fn generate<'a>(
    exec: &'a mut dyn ModelExecutor,
    kv: &'a mut SequenceKv,
    tokenizer: Arc<Tokenizer>,
    req: &'a GenerationRequest,
    cancel: &'a CancelFlag,
    opts: GenerateOptions<'a>,
) -> Generation<'a> {
    Generation {
        exec,
        kv,
        req,
        cancel,
        opts,
        sampler: Sampler::new(&req.sampling, &req.prompt_tokens, &end_ids(&req.stop)),
        detok: IncrementalDetokenizer::new(tokenizer),
        held: String::new(),
        generated: 0,
        last_token: None,
        pending: VecDeque::from([GenerationEvent::Started { choice: 0 }]),
        state: State::Running,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    Running,
    /// Finished, failed or cancelled: only queued events remain.
    Done,
}

/// The ids `min_tokens` holds back: EOS (unless ignored) and the request's stop token ids.
fn end_ids(stop: &StopConditions) -> Vec<u32> {
    let eos = if stop.ignore_eos {
        &[][..]
    } else {
        &stop.eos_token_ids[..]
    };
    eos.iter().chain(&stop.stop_token_ids).copied().collect()
}

/// The event stream of one request; see [`generate`].
pub struct Generation<'a> {
    exec: &'a mut dyn ModelExecutor,
    kv: &'a mut SequenceKv,
    req: &'a GenerationRequest,
    cancel: &'a CancelFlag,
    opts: GenerateOptions<'a>,
    sampler: Sampler,
    detok: IncrementalDetokenizer,
    /// Decoded text not yet streamed because it may begin a stop string.
    held: String,
    generated: u32,
    /// The token to feed to the next decode step (`None` before the prefill).
    last_token: Option<u32>,
    pending: VecDeque<GenerationEvent>,
    state: State,
}

impl std::fmt::Debug for Generation<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Generation")
            .field("request", &self.req.id)
            .field("generated", &self.generated)
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

impl Iterator for Generation<'_> {
    type Item = GenerationEvent;

    fn next(&mut self) -> Option<GenerationEvent> {
        if let Some(event) = self.pending.pop_front() {
            return Some(event);
        }
        if self.state == State::Done {
            return None;
        }
        self.step();
        self.pending.pop_front()
    }
}

impl Generation<'_> {
    /// One forward pass and one sampled token, queuing their events.
    fn step(&mut self) {
        let prompt_len = self.req.prompt_tokens.len() as u32;
        if self.req.stop.max_tokens == 0 {
            self.finish(FinishReason::Length);
            return;
        }
        if self.cancel.is_cancelled() {
            tracing::debug!(request_id = ?self.req.id, generated = self.generated, "generation cancelled");
            self.state = State::Done;
            return;
        }
        // Prefill feeds the whole prompt from position 0; each decode step feeds the last
        // sampled token at the next position.
        let (phase, tokens, positions): (_, &[u32], Vec<u32>) = match &self.last_token {
            None => (
                ForwardPhase::Prefill,
                &self.req.prompt_tokens,
                (0..prompt_len).collect(),
            ),
            Some(token) => (
                ForwardPhase::Decode,
                std::slice::from_ref(token),
                vec![prompt_len + self.generated - 1],
            ),
        };
        let started = Instant::now();
        let result = self.kv.forward(self.exec, tokens, &positions);
        if let Some(m) = self.opts.metrics {
            m.observe_forward(phase, started.elapsed().as_secs_f64());
        }
        let mut logits = match result {
            Ok(logits) => logits,
            Err(e) => {
                tracing::error!(request_id = ?self.req.id, phase = phase.as_str(), error = %e, "forward pass failed");
                self.pending.push_back(GenerationEvent::Error {
                    code: ErrorCode::InternalError,
                    message: format!("{} forward pass failed: {e}", phase.as_str()),
                });
                self.state = State::Done;
                return;
            }
        };
        let vocab = logits.vocab;
        let sampled = self.sampler.sample(&mut logits.data[..vocab], None);
        let token = sampled.token;
        self.sampler.observe(token);
        self.generated += 1;
        self.last_token = Some(token);

        let stop = &self.req.stop;
        let is_eos = !stop.ignore_eos && stop.eos_token_ids.contains(&token);
        if !is_eos && let Some(chunk) = self.detok.push(token) {
            self.held.push_str(&chunk);
        }
        let length = self.generated >= stop.max_tokens
            || prompt_len + self.generated >= self.opts.max_seq_len;
        let (text, finish) = if let Some(cut) = find_stop(&self.held, &stop.stop_strings) {
            // The stop string and everything after it are dropped.
            let text = self.held[..cut].to_string();
            self.held.clear();
            (text, Some(FinishReason::Stop))
        } else if is_eos || length {
            // Last token: release everything still held, including bytes the detokenizer kept
            // back, unless that completes a stop string.
            let rest = self.drain_all();
            match find_stop(&rest, &stop.stop_strings) {
                Some(cut) => (rest[..cut].to_string(), Some(FinishReason::Stop)),
                None if is_eos => (rest, Some(FinishReason::Stop)),
                None => (rest, Some(FinishReason::Length)),
            }
        } else {
            // Stream all but a suffix that may begin a stop string.
            let emit = self.held.len() - held_suffix_len(&self.held, &stop.stop_strings);
            let text: String = self.held.drain(..emit).collect();
            (text, None)
        };
        self.pending.push_back(GenerationEvent::Token {
            choice: 0,
            text,
            token_id: token,
            logprob: sampled.logprob,
            top_logprobs: sampled.top_logprobs,
        });
        if let Some(reason) = finish {
            self.finish(reason);
        }
    }

    /// Held text plus whatever the detokenizer still holds.
    fn drain_all(&mut self) -> String {
        let mut rest = std::mem::take(&mut self.held);
        if let Some(tail) = self.detok.flush() {
            rest.push_str(&tail);
        }
        rest
    }

    fn finish(&mut self, reason: FinishReason) {
        let usage = Usage {
            prompt_tokens: self.req.prompt_tokens.len() as u32,
            completion_tokens: self.generated,
        };
        tracing::debug!(
            request_id = ?self.req.id,
            finish_reason = reason.as_str(),
            prompt_tokens = usage.prompt_tokens,
            completion_tokens = usage.completion_tokens,
            "generation finished"
        );
        self.pending.push_back(GenerationEvent::Finished {
            choice: 0,
            reason,
            usage: Some(usage),
        });
        self.state = State::Done;
    }
}

/// Byte offset of the earliest stop-string occurrence in `text`.
fn find_stop(text: &str, stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()))
        .min()
}

/// Length of the longest suffix of `text` that is a proper prefix of some stop string: those
/// bytes stay held until the next token decides.
fn held_suffix_len(text: &str, stops: &[String]) -> usize {
    let mut longest = 0;
    for stop in stops {
        for (i, _) in text.char_indices() {
            let suffix = &text[i..];
            if suffix.len() <= longest {
                break;
            }
            if suffix.len() < stop.len() && stop.starts_with(suffix) {
                longest = suffix.len();
                break;
            }
        }
    }
    longest
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use turbine_core::request::{
        CancelFlag, Endpoint, ErrorCode, FinishReason, GenerationEvent, GenerationRequest,
        SamplingParams, StopConditions, Usage,
    };
    use turbine_core::types::{DType, DeviceId, KvLayout, ModelShape, RequestId};
    use turbine_kernels::{KernelMetrics, KernelRegistry, cpu_reference_provider};
    use turbine_observability::MetricsRegistry;
    use turbine_tensor::DeviceMemory;
    use turbine_tensor::host::HostMemory;

    use super::*;
    use turbine_core::types::BlockId;
    use turbine_tensor::KvPoolView;

    use crate::executor::{
        BatchInput, ExecutorOptions, LlamaExecutor, Logits, ModelExecutor, SequenceKv,
    };
    use crate::metrics::ModelMetrics;
    use crate::sampler::argmax;
    use crate::testing::TempDir;
    use crate::testing::tiny::{TINY_EOS, TINY_VOCAB, TinySpec, write_tiny_llama};
    use crate::{MAX_STAGING_BYTES, ModelError, SafetensorsIndex, Tokenizer, WeightLoader};

    /// Emits one-hot logits: the `i`-th forward call makes `script[i]` the argmax.
    struct Scripted {
        shape: ModelShape,
        kv: KvLayout,
        script: Vec<u32>,
        /// `(first position, token count)` of every forward call.
        calls: Vec<(u32, usize)>,
        fail_at: Option<usize>,
        cancel_after: Option<(usize, CancelFlag)>,
        next_expected: u32,
    }

    impl Scripted {
        fn new(script: &[u32]) -> Scripted {
            Scripted {
                shape: ModelShape {
                    architecture: "Scripted".into(),
                    num_layers: 1,
                    hidden: 8,
                    num_attention_heads: 1,
                    num_kv_heads: 1,
                    head_dim: 8,
                    intermediate: 8,
                    vocab: TINY_VOCAB,
                    num_experts: 0,
                    experts_per_token: 0,
                    tied_embeddings: true,
                    weight_bytes: 0,
                    max_position_embeddings: 512,
                },
                kv: KvLayout {
                    num_layers: 1,
                    num_kv_heads: 1,
                    head_dim: 8,
                    dtype: DType::BF16,
                    block_tokens: 16,
                },
                script: script.to_vec(),
                calls: Vec::new(),
                fail_at: None,
                cancel_after: None,
                next_expected: 0,
            }
        }
    }

    impl ModelExecutor for Scripted {
        fn shape(&self) -> &ModelShape {
            &self.shape
        }
        fn kv_layout(&self) -> &KvLayout {
            &self.kv
        }
        fn forward(&mut self, batch: &BatchInput<'_>) -> Result<Logits, ModelError> {
            let step = self.calls.len();
            let start = batch.positions[0];
            assert_eq!(batch.tokens.len(), batch.positions.len());
            assert_eq!(start, self.next_expected, "positions are consecutive");
            self.next_expected = start + batch.tokens.len() as u32;
            self.calls.push((start, batch.tokens.len()));
            if self.fail_at == Some(step) {
                return Err(ModelError::MissingTensor("scripted failure".into()));
            }
            if let Some((after, flag)) = &self.cancel_after
                && step + 1 == *after
            {
                flag.cancel();
            }
            let mut data = vec![-10.0f32; TINY_VOCAB as usize];
            data[self.script[step] as usize] = 10.0;
            Ok(Logits {
                rows: 1,
                vocab: TINY_VOCAB as usize,
                data,
            })
        }
        fn copy_blocks(
            &mut self,
            _kv: &KvPoolView<'_>,
            _src: &[BlockId],
            _dst: &[BlockId],
        ) -> Result<(), ModelError> {
            unreachable!("generate never forks blocks")
        }
    }

    fn host_mem() -> Arc<dyn DeviceMemory> {
        HostMemory::new(DeviceId(0), 1 << 30)
    }

    fn tiny() -> (TempDir, TinySpec, Arc<Tokenizer>) {
        let dir = TempDir::new("turbine-generate");
        let spec = write_tiny_llama(dir.path(), 7);
        let tokenizer =
            Tokenizer::from_file(&spec.dir.join("tokenizer.json")).expect("tiny tokenizer");
        (dir, spec, Arc::new(tokenizer))
    }

    fn request(
        prompt: &[u32],
        sampling: SamplingParams,
        stop: StopConditions,
    ) -> GenerationRequest {
        GenerationRequest {
            id: RequestId::new_v4(),
            n: 1,
            priority: turbine_core::types::Priority::default(),
            echo: false,
            constraint: None,
            deadline_ms: u64::MAX,
            endpoint: Endpoint::Completions,
            http_request_id: "test".into(),
            prompt_tokens: prompt.to_vec(),
            sampling,
            stop,
        }
    }

    fn greedy() -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        }
    }

    fn stops(max_tokens: u32, stop_strings: &[&str], ignore_eos: bool) -> StopConditions {
        StopConditions {
            eos_token_ids: TINY_EOS.iter().copied().collect(),
            stop_strings: stop_strings.iter().map(|s| s.to_string()).collect(),
            max_tokens,
            ignore_eos,
            ..StopConditions::default()
        }
    }

    /// Runs one scripted generation; returns the events and the executor's forward calls.
    fn run(
        tokenizer: &Arc<Tokenizer>,
        exec: &mut Scripted,
        req: &GenerationRequest,
        max_seq_len: u32,
        metrics: Option<&ModelMetrics>,
    ) -> Vec<GenerationEvent> {
        let cancel = exec
            .cancel_after
            .as_ref()
            .map(|(_, f)| f.clone())
            .unwrap_or_default();
        let opts = GenerateOptions {
            max_seq_len,
            metrics,
        };
        let mut kv = SequenceKv::new(&host_mem(), exec.kv, max_seq_len).expect("kv");
        generate(exec, &mut kv, Arc::clone(tokenizer), req, &cancel, opts).collect()
    }

    /// Token ids, concatenated text, per-token texts and the finish of an event list.
    fn summary(
        events: &[GenerationEvent],
    ) -> (
        Vec<u32>,
        String,
        Vec<String>,
        Option<FinishReason>,
        Option<Usage>,
    ) {
        assert_eq!(
            events.first(),
            Some(&GenerationEvent::Started { choice: 0 })
        );
        let mut ids = Vec::new();
        let mut text = String::new();
        let mut texts = Vec::new();
        let mut finish = None;
        let mut usage = None;
        for e in &events[1..] {
            match e {
                GenerationEvent::Token {
                    text: t, token_id, ..
                } => {
                    assert!(finish.is_none(), "no token after Finished");
                    ids.push(*token_id);
                    text.push_str(t);
                    texts.push(t.clone());
                }
                GenerationEvent::Finished {
                    reason, usage: u, ..
                } => {
                    finish = Some(*reason);
                    usage = *u;
                }
                other => panic!("unexpected event {other:?}"),
            }
        }
        (ids, text, texts, finish, usage)
    }

    const PROMPT: [u32; 3] = [256, 72, 105]; // <|begin_of_text|> "Hi"
    const H: u32 = b'h' as u32;
    const I: u32 = b'i' as u32;
    const X: u32 = b'x' as u32;
    const Y: u32 = b'y' as u32;
    const A: u32 = b'a' as u32;
    const B: u32 = b'b' as u32;

    #[test]
    fn stop_conditions() {
        let (_dir, _spec, tok) = tiny();

        // Each id of the EOS list stops; the EOS token is streamed with empty text.
        for eos in TINY_EOS {
            let mut exec = Scripted::new(&[H, I, eos, X, X]);
            let req = request(&PROMPT, greedy(), stops(100, &[], false));
            let (ids, text, _, finish, usage) = summary(&run(&tok, &mut exec, &req, 512, None));
            assert_eq!(ids, vec![H, I, eos], "EOS {eos}");
            assert_eq!(text, "hi");
            assert_eq!(finish, Some(FinishReason::Stop), "EOS {eos}");
            assert_eq!(
                usage,
                Some(Usage {
                    prompt_tokens: 3,
                    completion_tokens: 3
                })
            );
            // prefill of the prompt, then one decode per token except after the last
            assert_eq!(exec.calls, vec![(0, 3), (3, 1), (4, 1)]);
        }

        // A stop string spanning two tokens: excluded, and the held "a" never streams.
        let mut exec = Scripted::new(&[X, A, B, Y, Y]);
        let req = request(&PROMPT, greedy(), stops(100, &["ab"], false));
        let (ids, text, texts, finish, _) = summary(&run(&tok, &mut exec, &req, 512, None));
        assert_eq!(ids, vec![X, A, B]);
        assert_eq!(texts, vec!["x".to_string(), String::new(), String::new()]);
        assert_eq!(text, "x");
        assert_eq!(finish, Some(FinishReason::Stop));

        // A held prefix that turns out not to be a stop is released.
        let mut exec = Scripted::new(&[A, Y, A, 257]);
        let req = request(&PROMPT, greedy(), stops(100, &["ab"], false));
        let (_, text, texts, finish, _) = summary(&run(&tok, &mut exec, &req, 512, None));
        assert_eq!(
            texts,
            vec![String::new(), "ay".into(), String::new(), "a".into()]
        );
        assert_eq!(text, "aya", "the held text is flushed at the finish");
        assert_eq!(finish, Some(FinishReason::Stop));

        // max_tokens
        let mut exec = Scripted::new(&[X; 10]);
        let req = request(&PROMPT, greedy(), stops(5, &[], false));
        let (ids, text, _, finish, _) = summary(&run(&tok, &mut exec, &req, 512, None));
        assert_eq!(ids.len(), 5);
        assert_eq!(text, "xxxxx");
        assert_eq!(finish, Some(FinishReason::Length));
        assert_eq!(exec.calls.len(), 5, "no forward after the last token");

        // prompt + generated reaches max_seq_len
        let mut exec = Scripted::new(&[X; 10]);
        let req = request(&PROMPT, greedy(), stops(100, &[], false));
        let (ids, _, _, finish, _) = summary(&run(&tok, &mut exec, &req, 7, None));
        assert_eq!(ids.len(), 4, "3 prompt + 4 generated = max_seq_len 7");
        assert_eq!(finish, Some(FinishReason::Length));
        assert!(exec.calls.iter().all(|&(p, n)| p + n as u32 <= 7));

        // ignore_eos continues past 257
        let mut exec = Scripted::new(&[H, 257, I, 260, X, X]);
        let req = request(&PROMPT, greedy(), stops(5, &[], true));
        let (ids, text, _, finish, _) = summary(&run(&tok, &mut exec, &req, 512, None));
        assert_eq!(ids, vec![H, 257, I, 260, X]);
        assert_eq!(text, "hix");
        assert_eq!(finish, Some(FinishReason::Length));

        // max_tokens 0: length without a forward
        let mut exec = Scripted::new(&[X]);
        let req = request(&PROMPT, greedy(), stops(0, &[], false));
        let events = run(&tok, &mut exec, &req, 512, None);
        let (ids, _, _, finish, usage) = summary(&events);
        assert!(ids.is_empty());
        assert_eq!(finish, Some(FinishReason::Length));
        assert_eq!(usage.map(|u| u.completion_tokens), Some(0));
        assert!(exec.calls.is_empty(), "max_tokens 0 runs no forward");
    }

    #[test]
    fn logprobs_are_reported_when_requested() {
        let (_dir, _spec, tok) = tiny();
        let mut exec = Scripted::new(&[H, I, 257]);
        let sampling = SamplingParams {
            temperature: 0.0,
            logprobs: Some(2),
            ..SamplingParams::default()
        };
        let req = request(&PROMPT, sampling, stops(10, &[], false));
        let events = run(&tok, &mut exec, &req, 512, None);
        let GenerationEvent::Token {
            token_id,
            logprob,
            top_logprobs,
            ..
        } = &events[1]
        else {
            panic!("{events:?}")
        };
        assert_eq!(*token_id, H);
        assert!(logprob.expect("logprob requested") > -1e-3);
        assert_eq!(top_logprobs.len(), 2);
        assert_eq!(top_logprobs[0].0, H);

        let mut exec = Scripted::new(&[H, 257]);
        let req = request(&PROMPT, greedy(), stops(10, &[], false));
        let events = run(&tok, &mut exec, &req, 512, None);
        let GenerationEvent::Token {
            logprob,
            top_logprobs,
            ..
        } = &events[1]
        else {
            panic!("{events:?}")
        };
        assert_eq!(*logprob, None);
        assert!(top_logprobs.is_empty());
    }

    #[test]
    fn cancellation_ends_the_stream_without_finished() {
        let (_dir, _spec, tok) = tiny();
        let flag = CancelFlag::default();
        let mut exec = Scripted::new(&[X; 20]);
        // the flag fires during the 3rd forward; its token is still streamed
        exec.cancel_after = Some((3, flag));
        let req = request(&PROMPT, greedy(), stops(20, &[], false));
        let events = run(&tok, &mut exec, &req, 512, None);
        let tokens = events
            .iter()
            .filter(|e| matches!(e, GenerationEvent::Token { .. }))
            .count();
        assert_eq!(tokens, 3);
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, GenerationEvent::Finished { .. })),
            "{events:?}"
        );
        assert_eq!(exec.calls.len(), 3, "no forward once cancelled");
    }

    #[test]
    fn forward_failure_is_an_internal_error() {
        let (_dir, _spec, tok) = tiny();
        let mut exec = Scripted::new(&[X; 5]);
        exec.fail_at = Some(2);
        let req = request(&PROMPT, greedy(), stops(5, &[], false));
        let events = run(&tok, &mut exec, &req, 512, None);
        assert_eq!(events.len(), 4, "{events:?}");
        let GenerationEvent::Error { code, message } = &events[3] else {
            panic!("{events:?}")
        };
        assert_eq!(*code, ErrorCode::InternalError);
        assert!(message.contains("scripted failure"), "{message}");
    }

    #[test]
    fn forward_durations_are_observed_by_phase() {
        let (_dir, _spec, tok) = tiny();
        let reg = MetricsRegistry::new();
        let metrics = ModelMetrics::register(&reg);
        let mut exec = Scripted::new(&[X; 4]);
        let req = request(&PROMPT, greedy(), stops(4, &[], false));
        run(&tok, &mut exec, &req, 512, Some(&metrics));
        let text = reg.render().expect("render");
        assert!(
            text.contains("turbine_forward_seconds_count{phase=\"prefill\"} 1"),
            "{text}"
        );
        assert!(
            text.contains("turbine_forward_seconds_count{phase=\"decode\"} 3"),
            "{text}"
        );
    }

    fn tiny_executor(spec: &TinySpec) -> (LlamaExecutor, SequenceKv) {
        let cfg = &spec.config;
        let mem = host_mem();
        let index = SafetensorsIndex::open(&spec.dir).expect("open tiny index");
        let weights = WeightLoader::load(&index, &crate::llama_slots(cfg), &mem, MAX_STAGING_BYTES)
            .expect("load");
        let provider = cpu_reference_provider();
        let order = [provider.id()];
        let registry = KernelRegistry::build(
            vec![provider],
            &order,
            &LlamaExecutor::requirements(cfg, 16, ExecutorOptions::default()),
            &KernelMetrics::register(&MetricsRegistry::new()),
        )
        .expect("every op has a provider");
        let kv = SequenceKv::new(&mem, cfg.kv_layout(16), 64).expect("kv");
        let exec = LlamaExecutor::new(
            cfg,
            weights,
            Arc::new(registry),
            mem,
            16,
            64,
            1,
            ExecutorOptions::default(),
        )
        .expect("executor");
        (exec, kv)
    }

    fn tokens_of(events: &[GenerationEvent]) -> Vec<u32> {
        events
            .iter()
            .filter_map(|e| match e {
                GenerationEvent::Token { token_id, .. } => Some(*token_id),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn seeded_sampling_is_deterministic() {
        let (_dir, spec, tok) = tiny();
        let (mut exec, mut kv) = tiny_executor(&spec);
        let prompt: Vec<u32> = std::iter::once(256)
            .chain("The tiny model".bytes().map(u32::from))
            .collect();
        let cancel = CancelFlag::default();
        let mut sample = |sampling: SamplingParams| -> Vec<u32> {
            let req = request(&prompt, sampling, stops(16, &[], true));
            let opts = GenerateOptions {
                max_seq_len: 64,
                metrics: None,
            };
            let events: Vec<GenerationEvent> =
                generate(&mut exec, &mut kv, Arc::clone(&tok), &req, &cancel, opts).collect();
            assert!(
                matches!(
                    events.last(),
                    Some(GenerationEvent::Finished {
                        reason: FinishReason::Length,
                        ..
                    })
                ),
                "{events:?}"
            );
            tokens_of(&events)
        };
        let seeded = |seed: u64| SamplingParams {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 50,
            seed: Some(seed),
            logprobs: None,
            ..SamplingParams::default()
        };
        let a = sample(seeded(7));
        let b = sample(seeded(7));
        let c = sample(seeded(8));
        assert_eq!(a.len(), 16);
        assert_eq!(a, b, "same seed, same tokens");
        assert_ne!(a, c, "seed 8 differs from seed 7");

        let greedy_tokens = sample(SamplingParams {
            temperature: 0.0,
            top_p: 0.1,
            top_k: 3,
            seed: None,
            logprobs: None,
            ..SamplingParams::default()
        });

        // The argmax sequence, driven by hand.
        let (mut exec, mut kv) = tiny_executor(&spec);
        let mut expected = Vec::new();
        let positions: Vec<u32> = (0..prompt.len() as u32).collect();
        let mut logits = kv.forward(&mut exec, &prompt, &positions).expect("prefill");
        for step in 0..16u32 {
            let t = argmax(logits.row(0));
            expected.push(t);
            if step == 15 {
                break;
            }
            let pos = prompt.len() as u32 + step;
            logits = kv.forward(&mut exec, &[t], &[pos]).expect("decode");
        }
        assert_eq!(greedy_tokens, expected, "temperature 0 is argmax");
    }
}
