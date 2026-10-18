//! The `harmony` reasoning parser (gpt-oss; Phase 7 S-9/S-10): the analysis channel is the
//! reasoning, the final channel the content, and a commentary-channel tool call is withheld
//! for the tool parser.
//!
//! The output is a sequence of messages, each `<|channel|><header><|message|><body>`, ended
//! by `<|end|>` (another message may follow), `<|return|>` (the generation ends) or
//! `<|call|>` (a tool call ends). Between messages come `<|start|>assistant` openers, which
//! the parser skips. Only the final channel reaches the content output; the withheld call
//! text is released at the finish as content, where the tool parser picks it up. Markers
//! split across pushes are held back until decided; the state machine is deterministic over
//! any split because it only ever decides on complete markers in a pending buffer.

use turbine_core::registry::Module;

use crate::formats::harmony as markers;
use crate::reasoning::{
    ReasoningDelta, ReasoningParser, ReasoningSample, ReasoningStream, held_prefix_len,
};

/// The `harmony` parser.
pub struct Harmony;

impl Module for Harmony {
    fn name(&self) -> &'static str {
        "harmony"
    }
}

impl ReasoningParser for Harmony {
    fn start(&self, _prompt_tail: &str) -> Box<dyn ReasoningStream> {
        Box::new(HarmonyStream {
            mode: Mode::Preamble,
            pending: String::new(),
            call: String::new(),
        })
    }

    fn sample(&self) -> ReasoningSample {
        ReasoningSample {
            prompt_tail: "<|start|>assistant",
            output: "<|channel|>analysis<|message|>reason it out<|end|><|start|>assistant<|channel|>final<|message|>The answer.<|return|>",
            reasoning: "reason it out",
            content: "The answer.",
        }
    }
}

/// What the stream is between.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Between messages: `<|start|>assistant` openers and stray text until the next
    /// `<|channel|>`.
    Preamble,
    /// A header is accumulating; dispatched at `<|message|>`.
    Header,
    /// The analysis channel: reasoning until `<|end|>`.
    Reasoning,
    /// The final channel (or a commentary message without a recipient): content until
    /// `<|end|>` or `<|return|>`.
    Content,
    /// A commentary call: its text is withheld until `<|call|>` (or the finish).
    Call,
    /// After `<|return|>`: everything else is dropped.
    Done,
}

/// One choice's Harmony channel stream.
struct HarmonyStream {
    mode: Mode,
    /// Text not decided yet.
    pending: String,
    /// The call form accumulated so far for a withheld tool call (`<|channel|>` + header +
    /// `<|message|>`, extended by the body and `<|call|>` when the call completes); released
    /// as content for the tool parser. Empty outside [`Mode::Call`].
    call: String,
}

impl ReasoningStream for HarmonyStream {
    fn push(&mut self, text: &str) -> ReasoningDelta {
        self.pending.push_str(text);
        let mut delta = ReasoningDelta::default();
        loop {
            match self.mode {
                Mode::Preamble => {
                    // `<|start|>assistant` openers; a channel marker dispatches to Header.
                    if let Some(after) = self.pending.strip_prefix(markers::START) {
                        self.pending = after.to_owned();
                    } else if let Some(after) = self.pending.strip_prefix(markers::ASSISTANT) {
                        self.pending = after.to_owned();
                    } else if markers::START.starts_with(&self.pending)
                        || markers::ASSISTANT.starts_with(&self.pending)
                    {
                        // A split opener: wait.
                        return delta;
                    } else if self.pending.starts_with(markers::CHANNEL) {
                        self.pending.drain(..markers::CHANNEL.len());
                        self.mode = Mode::Header;
                    } else if markers::CHANNEL.starts_with(&self.pending) {
                        // A split channel marker: wait.
                        return delta;
                    } else {
                        // Stray text before any channel (a malformed stream): it is content,
                        // minus a suffix that may still begin a marker.
                        let keep = held_prefix_len(&self.pending, markers::CHANNEL)
                            .max(held_prefix_len(&self.pending, markers::START));
                        let emit = self.pending.len() - keep;
                        delta.content.extend(self.pending.drain(..emit));
                        if keep > 0 {
                            return delta;
                        }
                        // All emitted; loop in case a complete marker now leads the buffer.
                    }
                }
                Mode::Header => {
                    let Some(at) = self.pending.find(markers::MESSAGE) else {
                        // The header is never emitted: it is held until it completes (or the
                        // finish releases it raw).
                        return delta;
                    };
                    let raw_header = self.pending[..at].to_owned();
                    let header = raw_header.trim();
                    let next = if header.starts_with("analysis") {
                        Mode::Reasoning
                    } else if header.starts_with("final") {
                        Mode::Content
                    } else if header.starts_with(markers::COMMENTARY)
                        && header.contains(markers::TO_FUNCTIONS)
                    {
                        // A tool call: the whole form is withheld for the tool parser.
                        self.call.push_str(markers::CHANNEL);
                        self.call.push_str(&raw_header);
                        self.call.push_str(markers::MESSAGE);
                        Mode::Call
                    } else {
                        // A commentary message without a recipient is a preamble to the user.
                        Mode::Content
                    };
                    self.pending.drain(..at + markers::MESSAGE.len());
                    self.mode = next;
                }
                Mode::Reasoning => {
                    let Some(at) = self.pending.find(markers::END) else {
                        let keep = held_prefix_len(&self.pending, markers::END);
                        let emit = self.pending.len() - keep;
                        delta.reasoning.extend(self.pending.drain(..emit));
                        return delta;
                    };
                    delta.reasoning.push_str(&self.pending[..at]);
                    self.pending.drain(..at + markers::END.len());
                    self.mode = Mode::Preamble;
                }
                Mode::Content => {
                    let end = self
                        .pending
                        .find(markers::END)
                        .into_iter()
                        .chain(self.pending.find(markers::RETURN))
                        .min();
                    let Some(at) = end else {
                        let keep = held_prefix_len(&self.pending, markers::END)
                            .max(held_prefix_len(&self.pending, markers::RETURN));
                        let emit = self.pending.len() - keep;
                        delta.content.extend(self.pending.drain(..emit));
                        return delta;
                    };
                    delta.content.push_str(&self.pending[..at]);
                    let done = self.pending[at..].starts_with(markers::RETURN);
                    let marker_len = if done {
                        markers::RETURN.len()
                    } else {
                        markers::END.len()
                    };
                    self.pending.drain(..at + marker_len);
                    self.mode = if done { Mode::Done } else { Mode::Preamble };
                }
                Mode::Call => {
                    let Some(at) = self.pending.find(markers::CALL) else {
                        // The call text stays withheld until it completes (or the finish
                        // releases it as content for the tool parser).
                        return delta;
                    };
                    // The call completed: release the whole form as content, where the tool
                    // parser picks it up.
                    let mut released = std::mem::take(&mut self.call);
                    released.extend(self.pending.drain(..at + markers::CALL.len()));
                    delta.content.push_str(&released);
                    self.mode = Mode::Preamble;
                }
                Mode::Done => {
                    self.pending.clear();
                    return delta;
                }
            }
        }
    }

    fn finish(&mut self) -> ReasoningDelta {
        let rest = std::mem::take(&mut self.pending);
        match self.mode {
            Mode::Reasoning => ReasoningDelta {
                reasoning: rest,
                content: String::new(),
            },
            // An unfinished call releases the form it accumulated so far (the tool parser
            // returns it as content, as any output that does not parse).
            Mode::Call => {
                self.call.push_str(&rest);
                ReasoningDelta {
                    reasoning: String::new(),
                    content: std::mem::take(&mut self.call),
                }
            }
            Mode::Preamble | Mode::Header | Mode::Content => ReasoningDelta {
                reasoning: String::new(),
                content: rest,
            },
            Mode::Done => ReasoningDelta::default(),
        }
    }
}
