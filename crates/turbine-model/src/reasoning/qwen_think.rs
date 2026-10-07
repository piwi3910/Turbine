//! `qwen_think`: reasoning between `<think>` and `</think>` (Qwen3, Qwen3-MoE).
//!
//! The stream starts in reasoning when the prompt's generation prompt left a `<think>` open
//! (the Qwen3 templates with `enable_thinking`); otherwise an output that opens with `<think>`
//! (after optional whitespace) enters reasoning there. Everything up to `</think>` is
//! reasoning, everything after it content; a `<think>` never closed before the output ends is
//! all reasoning. Markers split across pushes are held back until decided; the markers
//! themselves are never output.

use turbine_core::registry::Module;

use super::{ReasoningDelta, ReasoningParser, ReasoningSample, ReasoningStream, held_prefix_len};

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

/// The `qwen_think` parser.
pub struct QwenThink;

impl Module for QwenThink {
    fn name(&self) -> &'static str {
        "qwen_think"
    }
}

impl ReasoningParser for QwenThink {
    fn start(&self, prompt_tail: &str) -> Box<dyn ReasoningStream> {
        let open = match (prompt_tail.rfind(OPEN), prompt_tail.rfind(CLOSE)) {
            (Some(o), Some(c)) => o > c,
            (Some(_), None) => true,
            (None, _) => false,
        };
        Box::new(QwenThinkStream {
            mode: if open { Mode::Reasoning } else { Mode::Opening },
            pending: String::new(),
        })
    }

    fn sample(&self) -> ReasoningSample {
        ReasoningSample {
            prompt_tail: "<|im_start|>assistant\n<think>\n",
            output: "plan</think>\n\nanswer",
            reasoning: "plan",
            content: "\n\nanswer",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Nothing but whitespace yet: the output may still open with `<think>`.
    Opening,
    Reasoning,
    Content,
}

struct QwenThinkStream {
    mode: Mode,
    /// Text not decided yet.
    pending: String,
}

impl ReasoningStream for QwenThinkStream {
    fn push(&mut self, text: &str) -> ReasoningDelta {
        self.pending.push_str(text);
        let mut delta = ReasoningDelta::default();
        loop {
            match self.mode {
                Mode::Opening => {
                    let rest = self.pending.trim_start();
                    if let Some(after) = rest.strip_prefix(OPEN) {
                        self.pending = after.to_string();
                        self.mode = Mode::Reasoning;
                    } else if OPEN.starts_with(rest) {
                        // Whitespace or a partial `<think>`: wait.
                        return delta;
                    } else {
                        self.mode = Mode::Content;
                    }
                }
                Mode::Reasoning => {
                    if let Some(at) = self.pending.find(CLOSE) {
                        delta.reasoning.push_str(&self.pending[..at]);
                        self.pending.drain(..at + CLOSE.len());
                        self.mode = Mode::Content;
                    } else {
                        let keep = held_prefix_len(&self.pending, CLOSE);
                        let emit = self.pending.len() - keep;
                        delta.reasoning.extend(self.pending.drain(..emit));
                        return delta;
                    }
                }
                Mode::Content => {
                    delta.content.push_str(&std::mem::take(&mut self.pending));
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
            Mode::Opening | Mode::Content => ReasoningDelta {
                reasoning: String::new(),
                content: rest,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(tail: &str, pieces: &[&str]) -> ReasoningDelta {
        let mut stream = QwenThink.start(tail);
        let mut out = ReasoningDelta::default();
        for piece in pieces {
            out.append(stream.push(piece));
        }
        out.append(stream.finish());
        out
    }

    fn split(reasoning: &str, content: &str) -> ReasoningDelta {
        ReasoningDelta {
            reasoning: reasoning.into(),
            content: content.into(),
        }
    }

    /// The prompt decides the start: an open `<think>` starts in reasoning, a closed one
    /// (`enable_thinking: false`) or none in content. Breaks if the tail is ignored.
    #[test]
    fn prompt_tail_decides_the_start() {
        assert_eq!(run("assistant\n<think>\n", &["a b"]), split("a b", ""));
        assert_eq!(
            run("assistant\n<think>\n\n</think>\n\n", &["a b"]),
            split("", "a b")
        );
        assert_eq!(run("assistant\n", &["a b"]), split("", "a b"));
    }

    /// An output that opens `<think>` itself (after whitespace, possibly split) is reasoning
    /// up to `</think>`; a `<think>` later in content is content. Breaks if the opening marker
    /// leaks or a late one is taken as reasoning.
    #[test]
    fn output_may_open_the_reasoning() {
        assert_eq!(
            run("assistant\n", &["\n<th", "ink>x</th", "ink>y"]),
            split("x", "y")
        );
        assert_eq!(
            run("assistant\n", &["hi <think>x</think>"]),
            split("", "hi <think>x</think>")
        );
    }

    /// A reasoning never closed before the output ends is all reasoning, a partial close
    /// marker included. Breaks if the held-back suffix is lost at the finish.
    #[test]
    fn unclosed_think_is_all_reasoning() {
        assert_eq!(run("<think>\n", &["abc</thi"]), split("abc</thi", ""));
        let mut stream = QwenThink.start("<think>\n");
        assert_eq!(stream.push("abc</"), split("abc", ""));
        assert_eq!(stream.push("think>d"), split("", "d"));
    }
}
