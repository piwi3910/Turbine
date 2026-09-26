//! Per-request host state of the engine (P2 S-2, S-6, S-7): the request, its output channel and
//! the events that did not fit it, and per choice the token history, sampler, detokenizer and
//! stop state. Everything here lives on the engine thread and survives a preemption: a
//! re-prefill replays the history from [`ActiveRequest::token_at`] and continues the same
//! sampler, so no token is sent twice.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use smallvec::SmallVec;
use tokio::sync::mpsc::{self, error::TrySendError};
use turbine_core::request::{
    ErrorCode, FinishReason, GenerationEvent, GenerationRequest, StopConditions, Usage,
};
use turbine_core::types::SeqId;
use turbine_model::{IncrementalDetokenizer, Sampler, Tokenizer};

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
    /// Events the full channel did not take, oldest first. While any are held every new event
    /// queues behind them, so the client sees the stream in order.
    held: VecDeque<GenerationEvent>,
    /// Accounted as finished, failed, rejected or cancelled: only held events remain to deliver.
    /// A done request whose client never reads again keeps its (at most a few) held events
    /// until the client goes away; it holds no KV.
    pub done: bool,
}

impl ActiveRequest {
    /// `seqs[i]` is choice `i`.
    pub fn new(
        request: GenerationRequest,
        events: mpsc::Sender<GenerationEvent>,
        seqs: &[SeqId],
        tokenizer: &Arc<Tokenizer>,
    ) -> ActiveRequest {
        let end_ids = end_ids(&request.stop);
        let choices = seqs
            .iter()
            .zip(0u32..)
            .map(|(&seq, index)| Choice {
                index,
                seq,
                sampler: Sampler::new(&request.sampling, &request.prompt_tokens, &end_ids),
                detok: IncrementalDetokenizer::new(Arc::clone(tokenizer)),
                held_text: String::new(),
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

    /// Samples the next token of choice `choice` from its logits row (adjusted in place) and
    /// applies the stop conditions: returns the `Token` event and, when this token ends the
    /// choice, its finish reason (also recorded on the choice).
    pub fn step(
        &mut self,
        choice: usize,
        logits: &mut [f32],
        max_seq_len: u32,
    ) -> (GenerationEvent, Option<FinishReason>) {
        let prompt_len = self.prompt_len();
        let with_logprobs = self.request.sampling.logprobs.is_some();
        let stop = &self.request.stop;
        let c = &mut self.choices[choice];
        let sampled = c.sampler.sample(logits, None);
        let token = sampled.token;
        c.sampler.observe(token);
        c.generated.push(token);
        let generated = c.generated.len() as u32;

        let is_eos = !stop.ignore_eos && stop.eos_token_ids.contains(&token);
        if !is_eos && let Some(chunk) = c.detok.push(token) {
            c.held_text.push_str(&chunk);
        }
        let length = generated >= stop.max_tokens || prompt_len + generated >= max_seq_len;
        let (text, finish) = if let Some(cut) = find_stop(&c.held_text, &stop.stop_strings) {
            // The stop string and everything after it are dropped.
            let text = c.held_text[..cut].to_string();
            c.held_text.clear();
            (text, Some(FinishReason::Stop))
        } else if is_eos || length {
            // Last token: release everything still held, including bytes the detokenizer kept
            // back, unless that completes a stop string.
            let rest = c.drain_all();
            match find_stop(&rest, &stop.stop_strings) {
                Some(cut) => (rest[..cut].to_string(), Some(FinishReason::Stop)),
                None if is_eos => (rest, Some(FinishReason::Stop)),
                None => (rest, Some(FinishReason::Length)),
            }
        } else {
            // Stream all but a suffix that may begin a stop string.
            let emit = c.held_text.len() - held_suffix_len(&c.held_text, &stop.stop_strings);
            let text: String = c.held_text.drain(..emit).collect();
            (text, None)
        };
        c.finish = finish;
        let event = GenerationEvent::Token {
            choice: c.index,
            text,
            token_id: token,
            logprob: with_logprobs.then_some(sampled.logprob),
            top_logprobs: sampled.top_logprobs,
        };
        (event, finish)
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
    /// Generated tokens in order, EOS included.
    pub generated: Vec<u32>,
    pub finish: Option<FinishReason>,
    /// When the previous token was sampled (inter-token latency).
    pub last_token_at: Option<Instant>,
}

impl Choice {
    /// Held text plus whatever the detokenizer still holds.
    fn drain_all(&mut self) -> String {
        let mut rest = std::mem::take(&mut self.held_text);
        if let Some(tail) = self.detok.flush() {
            rest.push_str(&tail);
        }
        rest
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

    #[test]
    fn history_stop_strings_and_held_events() {
        let dir = TempDir::new("turbine-engine-requests");
        let spec = write_tiny_llama(dir.path(), 7);
        let tokenizer = Arc::new(Tokenizer::from_file(&spec.dir.join("tokenizer.json")).unwrap());
        let vocab = tokenizer.vocab_size() as usize;
        // Byte tokens: "a" = 97, "b" = 98, "c" = 99; stop at "bc".
        let (tx, mut rx) = mpsc::channel(2);
        let mut r = ActiveRequest::new(
            request(&[256, 97], 10, &["bc"]),
            tx,
            &[SeqId(1)],
            &tokenizer,
        );

        let (ev, finish) = r.step(0, &mut row(97, vocab), 512);
        assert_eq!(finish, None);
        assert!(matches!(&ev, GenerationEvent::Token { text, token_id: 97, .. } if text == "a"));
        // "b" may begin the stop string: held back.
        let (ev, finish) = r.step(0, &mut row(98, vocab), 512);
        assert_eq!(finish, None);
        assert!(matches!(&ev, GenerationEvent::Token { text, .. } if text.is_empty()));
        let (ev, finish) = r.step(0, &mut row(99, vocab), 512);
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
        let mut r = ActiveRequest::new(request(&[256], 2, &[]), tx, &[SeqId(1)], &tokenizer);
        assert_eq!(r.step(0, &mut row(97, vocab), 512).1, None);
        assert_eq!(
            r.step(0, &mut row(97, vocab), 512).1,
            Some(FinishReason::Length)
        );
        // The context limit ends a choice before max_tokens: prompt 1 + 1 generated = 2.
        let (tx, _rx) = mpsc::channel(8);
        let mut r = ActiveRequest::new(request(&[256], 10, &[]), tx, &[SeqId(2)], &tokenizer);
        assert_eq!(
            r.step(0, &mut row(97, vocab), 2).1,
            Some(FinishReason::Length)
        );
    }
}
