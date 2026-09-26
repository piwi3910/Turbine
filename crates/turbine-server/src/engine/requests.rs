//! Per-request host state of the engine (P2 S-2, S-6, S-7, S-10, S-17, S-18): the request, its
//! output channel and the events that did not fit it, and per choice the token history, sampler
//! (seeded with the request seed plus the choice index), detokenizer, stop state,
//! constrained-decoding matcher and tool-call buffer. Everything here lives on the engine thread
//! and survives a preemption: a re-prefill replays the history from
//! [`ActiveRequest::token_at`] and continues the same sampler and matcher, so no token is sent
//! twice and the matcher is never replayed.
//!
//! Tool calls (P2 §Structured output and tool-call rules): with a `required` or named
//! `tool_choice` the whole output is held and parsed at the finish; with `auto` it streams
//! unless the model's tool format says it opens like a call
//! ([`turbine_model::formats::ToolFormat::opens_like_call`]), which holds the rest. A
//! parsed output becomes one `ToolCalls` event and `finish_reason: "tool_calls"`; an output that
//! does not parse is returned as content with its finish reason unchanged
//! (`tool_call_parse_failed`).

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use smallvec::SmallVec;
use tokio::sync::mpsc::{self, error::TrySendError};
use turbine_core::request::{
    ErrorCode, FinishReason, GenerationEvent, GenerationRequest, SamplingParams, StopConditions,
    Usage,
};
use turbine_core::types::SeqId;
use turbine_model::formats::{BoundToolFormat, Opening};
use turbine_model::{
    IncrementalDetokenizer, ModelError, ModelMetrics, SampledToken, Sampler, TokenMask,
    TokenMatcher, Tokenizer, ToolCallOutcome, ToolCallParser, ToolParse, step_mask,
};

/// How a request's output is turned into tool calls.
#[derive(Clone, Default)]
pub(crate) enum ToolOutput {
    /// Plain content (no tools, or `tool_choice: "none"`).
    #[default]
    None,
    /// `tool_choice: "auto"`: constrained to text or calls; held and parsed when it starts
    /// like a call.
    Auto(ToolParser),
    /// `required` or a named function: constrained by the tool grammar, always parsed.
    Constrained(ToolParser),
}

/// The model's tool-call format (`model.tool_call_parser`) and its parser.
#[derive(Clone)]
pub(crate) struct ToolParser {
    /// The format bound to the tokenizer: decides whether an `auto` output opens like a call.
    pub format: Arc<BoundToolFormat>,
    pub parser: Arc<dyn ToolCallParser>,
    /// The `parser` label of `turbine_tool_calls_total` (the format's name).
    pub label: &'static str,
}

/// What the HTTP side hands the engine for one request.
pub(crate) struct Submission {
    pub request: GenerationRequest,
    /// One compiled matcher per choice when `request.constraint` is set (compiled off the
    /// engine thread before queueing); empty otherwise.
    pub matchers: Vec<Box<dyn TokenMatcher>>,
    pub tools: ToolOutput,
    /// Completions `echo`: the prompt text each choice's output starts with.
    pub echo_text: Option<String>,
}

impl From<GenerationRequest> for Submission {
    /// An unconstrained request without tools or echo.
    fn from(request: GenerationRequest) -> Submission {
        Submission {
            request,
            matchers: Vec::new(),
            tools: ToolOutput::None,
            echo_text: None,
        }
    }
}

/// The events one sampled token produces and whether it ended its choice.
#[derive(Debug)]
pub(crate) struct Step {
    /// A `Token` event, then a `ToolCalls` event when the finished output parsed as calls.
    pub events: SmallVec<[GenerationEvent; 2]>,
    pub finish: Option<FinishReason>,
    /// Time of the step after sampling: detokenisation, the stop-string search, tool-call
    /// holding and parsing, and building the events (the engine's `detokenize` stage).
    pub detokenize: Duration,
}

/// Where an event went.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Delivery {
    Sent,
    /// Held on the host behind a full channel; `now_full` when this event found the channel
    /// full (the request must pause), false when it queued behind earlier held events.
    Held {
        now_full: bool,
    },
    /// The client dropped the stream.
    Closed,
}

/// Result of retrying the held events.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Flush {
    /// Nothing is held any more.
    Drained,
    /// The channel is full again.
    Held,
    Closed,
}

/// One accepted request on the engine thread.
pub(crate) struct ActiveRequest {
    pub request: GenerationRequest,
    events: mpsc::Sender<GenerationEvent>,
    /// When the engine received it: the start of TTFT and end-to-end latency.
    pub arrived: Instant,
    pub choices: SmallVec<[Choice; 1]>,
    tools: ToolOutput,
    /// Scratch bitmask for the constrained choices' allowed tokens.
    mask: Option<TokenMask>,
    /// Events the full channel did not take, oldest first. While any are held every new event
    /// queues behind them, so the client sees the stream in order.
    held: VecDeque<GenerationEvent>,
    /// Accounted as finished, failed, rejected or cancelled: only held events remain to deliver.
    /// A done request whose client never reads again keeps its (at most a few) held events
    /// until the client goes away; it holds no KV.
    pub done: bool,
}

impl ActiveRequest {
    /// `seqs[i]` is choice `i`; choice `i` gets `submission.matchers[i]` when constrained.
    pub fn new(
        submission: Submission,
        events: mpsc::Sender<GenerationEvent>,
        seqs: &[SeqId],
        tokenizer: &Arc<Tokenizer>,
    ) -> ActiveRequest {
        let Submission {
            request,
            matchers,
            tools,
            echo_text,
        } = submission;
        let end_ids = end_ids(&request.stop);
        let mut matchers = matchers.into_iter();
        let choices = seqs
            .iter()
            .zip(0u32..)
            .map(|(&seq, index)| Choice {
                index,
                seq,
                sampler: Sampler::new(
                    &choice_params(&request.sampling, index),
                    &request.prompt_tokens,
                    &end_ids,
                ),
                detok: IncrementalDetokenizer::new(Arc::clone(tokenizer)),
                held_text: String::new(),
                matcher: matchers.next(),
                echo: echo_text.clone(),
                tool_text: match tools {
                    ToolOutput::None => ToolText::Streaming,
                    ToolOutput::Auto(_) => ToolText::Undecided(String::new()),
                    ToolOutput::Constrained(_) => ToolText::Holding(String::new()),
                },
                generated: Vec::new(),
                finish: None,
                last_token_at: None,
            })
            .collect();
        ActiveRequest {
            request,
            events,
            arrived: Instant::now(),
            choices,
            tools,
            mask: None,
            held: VecDeque::new(),
            done: false,
        }
    }

    pub fn prompt_len(&self) -> u32 {
        u32::try_from(self.request.prompt_tokens.len()).unwrap_or(u32::MAX)
    }

    /// Token at `position` of choice `choice`'s sequence: the prompt, then what it generated.
    /// `None` past the generated tokens.
    pub fn token_at(&self, choice: usize, position: u32) -> Option<u32> {
        let p = position as usize;
        let prompt = &self.request.prompt_tokens;
        match p.checked_sub(prompt.len()) {
            None => prompt.get(p).copied(),
            Some(g) => self.choices.get(choice)?.generated.get(g).copied(),
        }
    }

    /// Every choice reached its finish reason.
    pub fn all_finished(&self) -> bool {
        self.choices.iter().all(|c| c.finish.is_some())
    }

    /// Generated tokens over all choices.
    pub fn generated_tokens(&self) -> u64 {
        self.choices.iter().map(|c| c.generated.len() as u64).sum()
    }

    /// Sequences of choices that have not finished.
    pub fn live_seqs(&self) -> impl Iterator<Item = SeqId> + '_ {
        self.choices
            .iter()
            .filter(|c| c.finish.is_none())
            .map(|c| c.seq)
    }

    /// Choices after the first that have not sampled a token yet: they fork from choice 0's
    /// shared prefill (`n` > 1), whose last logits row gives each its first token.
    pub fn forking_choices(&self) -> Vec<usize> {
        self.choices
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(_, c)| c.generated.is_empty() && c.finish.is_none())
            .map(|(i, _)| i)
            .collect()
    }

    pub fn is_closed(&self) -> bool {
        self.events.is_closed()
    }

    pub fn has_held(&self) -> bool {
        !self.held.is_empty()
    }

    /// Sends `event` without waiting; a full channel holds it on the host.
    pub fn emit(&mut self, event: GenerationEvent) -> Delivery {
        if self.events.is_closed() {
            return Delivery::Closed;
        }
        if !self.held.is_empty() {
            self.held.push_back(event);
            return Delivery::Held { now_full: false };
        }
        match self.events.try_send(event) {
            Ok(()) => Delivery::Sent,
            Err(TrySendError::Full(event)) => {
                self.held.push_back(event);
                Delivery::Held { now_full: true }
            }
            Err(TrySendError::Closed(_)) => Delivery::Closed,
        }
    }

    /// Retries the held events in order.
    pub fn flush(&mut self) -> Flush {
        while let Some(event) = self.held.pop_front() {
            match self.events.try_send(event) {
                Ok(()) => {}
                Err(TrySendError::Full(event)) => {
                    self.held.push_front(event);
                    return Flush::Held;
                }
                Err(TrySendError::Closed(_)) => {
                    self.held.clear();
                    return Flush::Closed;
                }
            }
        }
        Flush::Drained
    }

    /// The error event that ends the request's stream.
    pub fn error_event(code: ErrorCode, message: impl Into<String>) -> GenerationEvent {
        GenerationEvent::Error {
            code,
            message: message.into(),
        }
    }

    /// The `Finished` event of choice `choice` with `reason`.
    pub fn finished_event(&self, choice: usize, reason: FinishReason) -> GenerationEvent {
        let c = &self.choices[choice];
        GenerationEvent::Finished {
            choice: c.index,
            reason,
            usage: Some(Usage {
                prompt_tokens: self.prompt_len(),
                completion_tokens: c.generated.len() as u32,
            }),
        }
    }

    /// Samples the next token of choice `choice` from its logits row (adjusted in place) —
    /// through the choice's matcher mask when constrained — and applies the stop conditions,
    /// `echo` and the tool-call rules. A matcher failure (llguidance step limit, a stuck
    /// constraint) is [`ModelError::Constraint`] for this request alone.
    pub fn step(
        &mut self,
        choice: usize,
        logits: &mut [f32],
        max_seq_len: u32,
        metrics: Option<&ModelMetrics>,
    ) -> Result<Step, ModelError> {
        let stop = &self.request.stop;
        let c = &mut self.choices[choice];
        let mask = match c.matcher.as_mut() {
            Some(matcher) => {
                let started = Instant::now();
                let mask = self
                    .mask
                    .get_or_insert_with(|| TokenMask::new_none(logits.len()));
                step_mask(matcher.as_mut(), &stop.eos_token_ids, mask)?;
                if let Some(m) = metrics {
                    m.observe_token_mask(started.elapsed().as_secs_f64());
                }
                Some(&*mask)
            }
            None => None,
        };
        let sampled = c.sampler.sample(logits, mask);
        self.step_sampled(choice, sampled, max_seq_len, metrics)
    }

    /// [`ActiveRequest::step`] for a token already drawn by choice `choice`'s own sampler
    /// (an unconstrained choice sampled on a worker thread, see
    /// [`Choice::unconstrained_sampler`]): the matcher commit, the stop conditions, `echo` and
    /// the tool-call rules.
    pub fn step_sampled(
        &mut self,
        choice: usize,
        sampled: SampledToken,
        max_seq_len: u32,
        metrics: Option<&ModelMetrics>,
    ) -> Result<Step, ModelError> {
        let prompt_len = self.prompt_len();
        let stop = &self.request.stop;
        let c = &mut self.choices[choice];
        let token = sampled.token;
        if let Some(matcher) = c.matcher.as_mut() {
            matcher.commit(token)?;
        }
        c.sampler.observe(token);
        c.generated.push(token);
        let generated = c.generated.len() as u32;
        let detokenize_started = Instant::now();

        // EOS (unless ignored) and `stop_token_ids` end the choice; their text is not output.
        let is_end = (!stop.ignore_eos && stop.eos_token_ids.contains(&token))
            || stop.stop_token_ids.contains(&token);
        if !is_end && let Some(chunk) = c.detok.push(token) {
            c.held_text.push_str(&chunk);
        }
        let length = generated >= stop.max_tokens || prompt_len + generated >= max_seq_len;
        let (text, mut finish) = if let Some(cut) = find_stop(&c.held_text, &stop.stop_strings) {
            // The stop string and everything after it are dropped.
            let text = c.held_text[..cut].to_string();
            c.held_text.clear();
            (text, Some(FinishReason::Stop))
        } else if is_end || length {
            // Last token: release everything still held, including bytes the detokenizer kept
            // back, unless that completes a stop string.
            let rest = c.drain_all();
            match find_stop(&rest, &stop.stop_strings) {
                Some(cut) => (rest[..cut].to_string(), Some(FinishReason::Stop)),
                None if is_end => (rest, Some(FinishReason::Stop)),
                None => (rest, Some(FinishReason::Length)),
            }
        } else {
            // Stream all but a suffix that may begin a stop string.
            let emit = c.held_text.len() - held_suffix_len(&c.held_text, &stop.stop_strings);
            let text: String = c.held_text.drain(..emit).collect();
            (text, None)
        };

        // Tool calls: hold what may be a call, parse it at the finish.
        let parser = match &self.tools {
            ToolOutput::None => None,
            ToolOutput::Auto(p) | ToolOutput::Constrained(p) => Some(p),
        };
        let first_token = c.generated.first().copied();
        let mut text = c.tool_text.push(text, |pending| match parser {
            Some(p) => p.format.opens_like_call(pending, first_token),
            None => Opening::Content,
        });
        let mut calls = None;
        if finish.is_some() {
            let (held, release) = c.tool_text.finish();
            text.push_str(&release);
            if let (Some(held), Some(p)) = (held, parser) {
                match p.parser.parse(&held) {
                    ToolParse::Calls(parsed) => {
                        calls = Some(parsed);
                        finish = Some(FinishReason::ToolCalls);
                        if let Some(m) = metrics {
                            m.record_tool_call(p.label, ToolCallOutcome::Parsed);
                        }
                    }
                    ToolParse::Content(raw) => {
                        if let Some(m) = metrics {
                            m.record_tool_call(p.label, ToolCallOutcome::ParseFailed);
                        }
                        tracing::info!(
                            event = "tool_call_parse_failed",
                            request_id = %self.request.http_request_id,
                            choice = c.index,
                            parser = p.label,
                            reason = "tool_call_parse_failed",
                            "the output is returned as content"
                        );
                        text.push_str(&raw);
                    }
                }
            }
        }
        if let Some(echo) = c.echo.take() {
            text.insert_str(0, &echo);
        }
        c.finish = finish;
        let mut events = SmallVec::new();
        events.push(GenerationEvent::Token {
            choice: c.index,
            text,
            token_id: token,
            logprob: sampled.logprob,
            top_logprobs: sampled.top_logprobs,
        });
        if let Some(calls) = calls {
            events.push(GenerationEvent::ToolCalls {
                choice: c.index,
                calls,
            });
        }
        Ok(Step {
            events,
            finish,
            detokenize: detokenize_started.elapsed(),
        })
    }
}

/// Where a choice's output text goes (see the module comment on tool calls).
#[derive(Debug)]
enum ToolText {
    /// Straight to the client.
    Streaming,
    /// `auto` before the output shows whether it opens like a call: held until the format
    /// decides.
    Undecided(String),
    /// Held for the parser.
    Holding(String),
}

impl ToolText {
    /// Takes the next text chunk; returns what may be streamed now. `opening` decides an
    /// undecided `auto` output from its pending text.
    fn push(&mut self, text: String, opening: impl FnOnce(&str) -> Opening) -> String {
        match self {
            ToolText::Streaming => text,
            ToolText::Holding(held) => {
                held.push_str(&text);
                String::new()
            }
            ToolText::Undecided(pending) => {
                pending.push_str(&text);
                match opening(pending) {
                    Opening::Undecided => String::new(),
                    Opening::Call => {
                        *self = ToolText::Holding(std::mem::take(pending));
                        String::new()
                    }
                    Opening::Content => {
                        let out = std::mem::take(pending);
                        *self = ToolText::Streaming;
                        out
                    }
                }
            }
        }
    }

    /// At the finish: the output held for the parser, if any, and text to release as
    /// content (whitespace an `auto` output never got past).
    fn finish(&mut self) -> (Option<String>, String) {
        match std::mem::replace(self, ToolText::Streaming) {
            ToolText::Holding(held) => (Some(held), String::new()),
            ToolText::Undecided(pending) => (None, pending),
            ToolText::Streaming => (None, String::new()),
        }
    }
}

/// One choice (`n` > 1 has several) and its sequence.
pub(crate) struct Choice {
    pub index: u32,
    pub seq: SeqId,
    sampler: Sampler,
    detok: IncrementalDetokenizer,
    /// Decoded text not yet streamed because it may begin a stop string.
    held_text: String,
    /// The constraint's matcher (`response_format` or the tool grammar), advanced by every
    /// sampled token.
    matcher: Option<Box<dyn TokenMatcher>>,
    /// `echo` text not yet sent (it prefixes the choice's first token event).
    echo: Option<String>,
    tool_text: ToolText,
    /// Generated tokens in order, EOS included.
    pub generated: Vec<u32>,
    pub finish: Option<FinishReason>,
    /// When the previous token was sampled (inter-token latency).
    pub last_token_at: Option<Instant>,
}

impl Choice {
    /// The sampler of a live choice without a constraint: its next token needs no matcher mask,
    /// so it can be drawn away from the request (`turbine_model::sample_rows`).
    pub fn unconstrained_sampler(&mut self) -> Option<&mut Sampler> {
        (self.matcher.is_none() && self.finish.is_none()).then_some(&mut self.sampler)
    }

    /// The choice's sampler, constrained or not.
    pub fn sampler_mut(&mut self) -> &mut Sampler {
        &mut self.sampler
    }

    /// The choice samples under a constraint (`response_format` or the tool grammar): its
    /// steps carry a token mask.
    pub fn is_constrained(&self) -> bool {
        self.matcher.is_some()
    }

    /// The request carries a `seed` (its draws must stay reproducible).
    pub fn is_seeded(&self) -> bool {
        self.sampler.is_seeded()
    }

    /// Held text plus whatever the detokenizer still holds.
    fn drain_all(&mut self) -> String {
        let mut rest = std::mem::take(&mut self.held_text);
        if let Some(tail) = self.detok.flush() {
            rest.push_str(&tail);
        }
        rest
    }
}

/// Choice `index`'s sampling parameters: the request's, with the seed offset by the index so
/// that the choices of a seeded `n` > 1 request differ and each is reproducible.
fn choice_params(params: &SamplingParams, index: u32) -> SamplingParams {
    SamplingParams {
        seed: params.seed.map(|s| s.wrapping_add(u64::from(index))),
        ..params.clone()
    }
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
    use turbine_core::request::{Endpoint, SamplingParams};
    use turbine_core::types::{Priority, RequestId};
    use turbine_model::testing::TempDir;
    use turbine_model::testing::tiny::write_tiny_llama;

    use super::*;

    fn request(prompt: &[u32], max_tokens: u32, stop_strings: &[&str]) -> GenerationRequest {
        GenerationRequest {
            id: RequestId::new_v4(),
            n: 1,
            priority: Priority::default(),
            echo: false,
            constraint: None,
            deadline_ms: u64::MAX,
            endpoint: Endpoint::Completions,
            http_request_id: "t".into(),
            prompt_tokens: prompt.to_vec(),
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            stop: StopConditions {
                max_tokens,
                stop_strings: stop_strings.iter().map(|s| (*s).to_string()).collect(),
                ..StopConditions::default()
            },
        }
    }

    /// One-hot logits favouring `token`.
    fn row(token: u32, vocab: usize) -> Vec<f32> {
        let mut r = vec![-10.0; vocab];
        r[token as usize] = 10.0;
        r
    }

    /// One sampled token of choice 0: its `Token` event and finish reason.
    fn step(
        r: &mut ActiveRequest,
        mut logits: Vec<f32>,
        max_seq_len: u32,
    ) -> (GenerationEvent, Option<FinishReason>) {
        let s = r.step(0, &mut logits, max_seq_len, None).expect("step");
        (s.events[0].clone(), s.finish)
    }

    fn tiny_tokenizer(name: &str) -> (TempDir, Arc<Tokenizer>) {
        let dir = TempDir::new(name);
        let spec = write_tiny_llama(dir.path(), 7);
        let tokenizer = Arc::new(Tokenizer::from_file(&spec.dir.join("tokenizer.json")).unwrap());
        (dir, tokenizer)
    }

    fn token_text(event: &GenerationEvent) -> &str {
        match event {
            GenerationEvent::Token { text, .. } => text,
            other => panic!("not a token event: {other:?}"),
        }
    }

    /// Feeds `text` byte by byte (tiny tokenizer: byte `b` is token `b`), then `last`; returns
    /// the streamed text and the final step.
    fn feed(r: &mut ActiveRequest, vocab: usize, text: &str, last: u32) -> (String, Step) {
        let mut streamed = String::new();
        for b in text.bytes() {
            let s = r
                .step(0, &mut row(u32::from(b), vocab), 512, None)
                .expect("step");
            assert_eq!(s.finish, None, "{text:?} ended early");
            streamed.push_str(token_text(&s.events[0]));
        }
        let s = r.step(0, &mut row(last, vocab), 512, None).expect("step");
        streamed.push_str(token_text(&s.events[0]));
        (streamed, s)
    }

    #[test]
    fn stop_token_ids_echo_and_choice_seeds() {
        let (_dir, tokenizer) = tiny_tokenizer("turbine-engine-requests-p2");
        let vocab = tokenizer.vocab_size() as usize;

        // A stop token id ends the choice like EOS; its text is not output.
        let mut req = request(&[256], 10, &[]);
        req.stop.stop_token_ids = vec![98];
        let (tx, _rx) = mpsc::channel(8);
        let mut r = ActiveRequest::new(req.into(), tx, &[SeqId(1)], &tokenizer);
        let (ev, finish) = step(&mut r, row(97, vocab), 512);
        assert_eq!((token_text(&ev), finish), ("a", None));
        let (ev, finish) = step(&mut r, row(98, vocab), 512);
        assert_eq!((token_text(&ev), finish), ("", Some(FinishReason::Stop)));

        // `echo` prefixes the first token event only.
        let (tx, _rx) = mpsc::channel(8);
        let mut r = ActiveRequest::new(
            Submission {
                echo_text: Some("Hi".into()),
                ..request(&[256, 72, 105], 10, &[]).into()
            },
            tx,
            &[SeqId(1)],
            &tokenizer,
        );
        assert_eq!(token_text(&step(&mut r, row(97, vocab), 512).0), "Hia");
        assert_eq!(token_text(&step(&mut r, row(97, vocab), 512).0), "a");

        // Seeded choices draw from the request seed plus their index: choice 0 matches a
        // single-choice request with the same seed, choice 1 draws differently.
        let mut req = request(&[256], 64, &[]);
        req.n = 2;
        req.sampling.temperature = 1.0;
        req.sampling.seed = Some(5);
        let single = {
            let mut one = req.clone();
            one.n = 1;
            one
        };
        let flat = vec![0.0f32; vocab];
        let (tx, _rx) = mpsc::channel(256);
        let mut two = ActiveRequest::new(req.into(), tx, &[SeqId(1), SeqId(2)], &tokenizer);
        let (tx, _rx1) = mpsc::channel(256);
        let mut one = ActiveRequest::new(single.into(), tx, &[SeqId(3)], &tokenizer);
        for _ in 0..20 {
            for c in 0..2 {
                two.step(c, &mut flat.clone(), 512, None).unwrap();
            }
            one.step(0, &mut flat.clone(), 512, None).unwrap();
        }
        assert_eq!(two.choices[0].generated, one.choices[0].generated);
        assert_ne!(two.choices[0].generated, two.choices[1].generated);
        assert_eq!(two.forking_choices(), Vec::<usize>::new());
    }

    fn tool_request(tools: ToolOutput, tokenizer: &Arc<Tokenizer>) -> ActiveRequest {
        let mut req = request(&[256], 64, &[]);
        req.endpoint = Endpoint::ChatCompletions;
        req.stop.eos_token_ids = SmallVec::from_slice(&[260]);
        let (tx, rx) = mpsc::channel(256);
        // The receiver is dropped: these tests only look at the returned events.
        drop(rx);
        ActiveRequest::new(
            Submission {
                tools,
                ..req.into()
            },
            tx,
            &[SeqId(1)],
            tokenizer,
        )
    }

    fn parser(tokenizer: &Tokenizer) -> ToolParser {
        let format = turbine_model::formats::registry()
            .get("llama3_json")
            .expect("registered");
        ToolParser {
            format: Arc::new(turbine_model::formats::bind(format, tokenizer).expect("bind")),
            parser: Arc::new(turbine_model::Llama3JsonParser::seeded(1)),
            label: format.name(),
        }
    }

    #[test]
    fn tool_calls_held_and_parsed() {
        let (_dir, tokenizer) = tiny_tokenizer("turbine-engine-requests-tools");
        let vocab = tokenizer.vocab_size() as usize;
        let call = r#"{"name": "f", "parameters": {"x": 1}}"#;

        // auto: a call is held, parsed at EOS and reported as tool calls.
        let mut r = tool_request(ToolOutput::Auto(parser(&tokenizer)), &tokenizer);
        let (streamed, last) = feed(&mut r, vocab, &format!("  {call}"), 260);
        assert_eq!(streamed, "");
        assert_eq!(last.finish, Some(FinishReason::ToolCalls));
        match &last.events[..] {
            [
                GenerationEvent::Token { text, .. },
                GenerationEvent::ToolCalls { choice: 0, calls },
            ] => {
                assert!(text.is_empty());
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "f");
                assert_eq!(calls[0].arguments, r#"{"x":1}"#);
            }
            other => panic!("{other:?}"),
        }

        // auto: the format's call-opening special token (262, no text of its own) opens a
        // call too.
        let mut r = tool_request(ToolOutput::Auto(parser(&tokenizer)), &tokenizer);
        step(&mut r, row(262, vocab), 512);
        let (streamed, last) = feed(&mut r, vocab, call, 260);
        assert_eq!(streamed, "");
        assert_eq!(last.finish, Some(FinishReason::ToolCalls));

        // auto: plain text streams; JSON that is not a call comes back as content.
        let mut r = tool_request(ToolOutput::Auto(parser(&tokenizer)), &tokenizer);
        let (streamed, last) = feed(&mut r, vocab, " hi", 260);
        assert_eq!(streamed, " hi");
        assert_eq!(
            (last.finish, last.events.len()),
            (Some(FinishReason::Stop), 1)
        );
        let mut r = tool_request(ToolOutput::Auto(parser(&tokenizer)), &tokenizer);
        let (streamed, last) = feed(&mut r, vocab, r#"{"a": 1}"#, 260);
        assert_eq!(streamed, r#"{"a": 1}"#);
        assert_eq!(
            (last.finish, last.events.len()),
            (Some(FinishReason::Stop), 1)
        );

        // Constrained: everything is held; an output cut by max_tokens is returned as content
        // with finish `length`.
        let mut r = tool_request(ToolOutput::Constrained(parser(&tokenizer)), &tokenizer);
        r.request.stop.max_tokens = 6;
        let (streamed, last) = feed(&mut r, vocab, "{\"nam", u32::from(b'e'));
        assert_eq!(streamed, "{\"name");
        assert_eq!(last.finish, Some(FinishReason::Length));
        // Without tools nothing is held.
        let mut r = tool_request(ToolOutput::None, &tokenizer);
        let (streamed, last) = feed(&mut r, vocab, call, 260);
        assert_eq!(streamed, call);
        assert_eq!(last.finish, Some(FinishReason::Stop));
    }

    /// Allows only `ids`; `fail` makes the mask computation fail.
    struct Only {
        ids: Vec<u32>,
        fail: bool,
    }

    impl TokenMatcher for Only {
        fn allowed(&mut self, mask: &mut TokenMask) -> Result<(), ModelError> {
            if self.fail {
                return Err(ModelError::Constraint("step limit exceeded".into()));
            }
            mask.clear();
            for &id in &self.ids {
                mask.allow(id);
            }
            Ok(())
        }
        fn commit(&mut self, token: u32) -> Result<(), ModelError> {
            assert!(self.ids.contains(&token), "committed {token}");
            Ok(())
        }
        fn accepts_eos(&self) -> bool {
            false
        }
    }

    #[test]
    fn matcher_masks_each_choice_and_fails_alone() {
        let (_dir, tokenizer) = tiny_tokenizer("turbine-engine-requests-matcher");
        let vocab = tokenizer.vocab_size() as usize;
        let mut req = request(&[256], 10, &[]);
        req.n = 2;
        let (tx, _rx) = mpsc::channel(8);
        let mut r = ActiveRequest::new(
            Submission {
                matchers: vec![
                    Box::new(Only {
                        ids: vec![120],
                        fail: false,
                    }),
                    Box::new(Only {
                        ids: vec![121],
                        fail: true,
                    }),
                ],
                ..req.into()
            },
            tx,
            &[SeqId(1), SeqId(2)],
            &tokenizer,
        );
        // Greedy would pick 97; the mask allows only 120.
        let s = r.step(0, &mut row(97, vocab), 512, None).unwrap();
        assert!(matches!(
            s.events[0],
            GenerationEvent::Token { token_id: 120, .. }
        ));
        let err = r.step(1, &mut row(97, vocab), 512, None).unwrap_err();
        assert!(matches!(err, ModelError::Constraint(ref m) if m.contains("step limit")));
    }

    #[test]
    fn history_stop_strings_and_held_events() {
        let dir = TempDir::new("turbine-engine-requests");
        let spec = write_tiny_llama(dir.path(), 7);
        let tokenizer = Arc::new(Tokenizer::from_file(&spec.dir.join("tokenizer.json")).unwrap());
        let vocab = tokenizer.vocab_size() as usize;
        // Byte tokens: "a" = 97, "b" = 98, "c" = 99; stop at "bc".
        let (tx, mut rx) = mpsc::channel(2);
        let mut r = ActiveRequest::new(
            request(&[256, 97], 10, &["bc"]).into(),
            tx,
            &[SeqId(1)],
            &tokenizer,
        );

        let (ev, finish) = step(&mut r, row(97, vocab), 512);
        assert_eq!(finish, None);
        assert!(matches!(&ev, GenerationEvent::Token { text, token_id: 97, .. } if text == "a"));
        // "b" may begin the stop string: held back.
        let (ev, finish) = step(&mut r, row(98, vocab), 512);
        assert_eq!(finish, None);
        assert!(matches!(&ev, GenerationEvent::Token { text, .. } if text.is_empty()));
        let (ev, finish) = step(&mut r, row(99, vocab), 512);
        assert_eq!(finish, Some(FinishReason::Stop));
        assert!(matches!(&ev, GenerationEvent::Token { text, .. } if text.is_empty()));
        assert!(r.all_finished());
        assert_eq!(r.live_seqs().count(), 0);

        // Prompt then generated tokens, by position.
        assert_eq!(
            (0..5).map(|p| r.token_at(0, p)).collect::<Vec<_>>(),
            [Some(256), Some(97), Some(97), Some(98), Some(99)]
        );
        assert_eq!(r.token_at(0, 5), None);
        assert_eq!(r.generated_tokens(), 3);

        // Channel of 2: the third event is held, the fourth queues behind it, in order.
        let started = |choice| GenerationEvent::Started { choice };
        assert_eq!(r.emit(started(0)), Delivery::Sent);
        assert_eq!(r.emit(started(1)), Delivery::Sent);
        assert_eq!(r.emit(started(2)), Delivery::Held { now_full: true });
        assert_eq!(r.emit(started(3)), Delivery::Held { now_full: false });
        assert!(r.has_held());
        assert_eq!(r.flush(), Flush::Held);
        assert_eq!(rx.try_recv().unwrap(), started(0));
        assert_eq!(r.flush(), Flush::Held);
        assert_eq!(rx.try_recv().unwrap(), started(1));
        assert_eq!(rx.try_recv().unwrap(), started(2));
        assert_eq!(r.flush(), Flush::Drained);
        assert_eq!(rx.try_recv().unwrap(), started(3));
        assert!(!r.has_held());

        let finished = r.finished_event(0, FinishReason::Stop);
        assert_eq!(
            finished,
            GenerationEvent::Finished {
                choice: 0,
                reason: FinishReason::Stop,
                usage: Some(Usage {
                    prompt_tokens: 2,
                    completion_tokens: 3
                })
            }
        );
        drop(rx);
        assert!(r.is_closed());
        assert_eq!(r.emit(finished), Delivery::Closed);
    }

    #[test]
    fn length_at_max_tokens_or_context() {
        let dir = TempDir::new("turbine-engine-requests-len");
        let spec = write_tiny_llama(dir.path(), 7);
        let tokenizer = Arc::new(Tokenizer::from_file(&spec.dir.join("tokenizer.json")).unwrap());
        let vocab = tokenizer.vocab_size() as usize;
        let (tx, _rx) = mpsc::channel(8);
        let mut r = ActiveRequest::new(request(&[256], 2, &[]).into(), tx, &[SeqId(1)], &tokenizer);
        assert_eq!(step(&mut r, row(97, vocab), 512).1, None);
        assert_eq!(
            step(&mut r, row(97, vocab), 512).1,
            Some(FinishReason::Length)
        );
        // The context limit ends a choice before max_tokens: prompt 1 + 1 generated = 2.
        let (tx, _rx) = mpsc::channel(8);
        let mut r =
            ActiveRequest::new(request(&[256], 10, &[]).into(), tx, &[SeqId(2)], &tokenizer);
        assert_eq!(
            step(&mut r, row(97, vocab), 2).1,
            Some(FinishReason::Length)
        );
    }
}
