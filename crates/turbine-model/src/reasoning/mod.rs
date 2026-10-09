//! Reasoning parsers (Phase 7 S-10, extension point `reasoning_parser`): how a model's output
//! splits into reasoning (`reasoning_content` in the OpenAI responses) and content. One file per
//! parser and one entry in [`registry`]; the server resolves `model.reasoning_parser` (`auto`:
//! the family's [`crate::ModelFamily::default_reasoning_parser`]) here and starts one
//! [`ReasoningStream`] per choice from the tail of the rendered prompt.

use turbine_core::registry::{Module, Registry};

pub mod harmony;
pub mod qwen_think;

pub use harmony::Harmony;
pub use qwen_think::QwenThink;

/// One way a model marks its reasoning.
pub trait ReasoningParser: Module {
    /// A stream for one choice; `prompt_tail` is the end of the rendered prompt (the generation
    /// prompt may already open the reasoning, e.g. `<think>\n`).
    fn start(&self, prompt_tail: &str) -> Box<dyn ReasoningStream>;
    /// A known output and its split, for the conformance suite
    /// ([`crate::conformance::reasoning_suite`]).
    fn sample(&self) -> ReasoningSample;
}

impl std::fmt::Debug for dyn ReasoningParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// One choice's output, split as it streams. Text that may still turn out to be a marker is
/// held back until the next push decides it, or until [`ReasoningStream::finish`].
pub trait ReasoningStream: Send {
    /// Takes the next decoded text; returns what is decided now.
    fn push(&mut self, text: &str) -> ReasoningDelta;
    /// The end of the output: releases everything held.
    fn finish(&mut self) -> ReasoningDelta;
}

/// The reasoning and content one push decided (either may be empty).
#[derive(Default, PartialEq, Debug)]
pub struct ReasoningDelta {
    pub reasoning: String,
    pub content: String,
}

impl ReasoningDelta {
    /// Appends `other` to this delta.
    pub fn append(&mut self, other: ReasoningDelta) {
        self.reasoning.push_str(&other.reasoning);
        self.content.push_str(&other.content);
    }
}

/// A parser's known split: `output` after a prompt ending in `prompt_tail` is `reasoning` plus
/// `content`. The conformance suite feeds `output` whole and split at every char boundary and
/// expects the same split each time.
pub struct ReasoningSample {
    pub prompt_tail: &'static str,
    pub output: &'static str,
    pub reasoning: &'static str,
    pub content: &'static str,
}

static REASONING_PARSERS: Registry<dyn ReasoningParser> =
    Registry::new("reasoning_parser", &[&QwenThink, &Harmony]);

/// Every reasoning parser, in registration order.
pub fn registry() -> &'static Registry<dyn ReasoningParser> {
    &REASONING_PARSERS
}

/// The longest suffix of `text` (at a char boundary) that is a proper prefix of `marker`: the
/// bytes a stream holds back because the next push may complete the marker.
pub(crate) fn held_prefix_len(text: &str, marker: &str) -> usize {
    let max = text.len().min(marker.len().saturating_sub(1));
    (1..=max)
        .rev()
        .find(|&k| {
            let start = text.len() - k;
            text.is_char_boundary(start) && marker.starts_with(&text[start..])
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod registry_conformance {
    use crate::conformance::reasoning::suite as reasoning_suite;

    /// Every registered parser passes `reasoning_suite` (its sample split whole, per char and at
    /// every boundary). Breaks if a parser loses or duplicates text across pushes, or leaks a
    /// marker into the output.
    #[test]
    fn every_parser_passes_the_suite() {
        if let Err(failures) = reasoning_suite(super::registry()) {
            let lines: Vec<String> = failures.iter().map(ToString::to_string).collect();
            panic!(
                "{} conformance failures:\n{}",
                lines.len(),
                lines.join("\n")
            );
        }
    }

    /// The registered parsers in order. Breaks if one is dropped, renamed or registered twice.
    #[test]
    fn names() {
        let reg = super::registry();
        assert_eq!(reg.point(), "reasoning_parser");
        assert_eq!(reg.names(), ["qwen_think", "harmony"]);
    }

    /// The held-back suffix is the longest proper marker prefix, on char boundaries.
    #[test]
    fn held_prefix() {
        use super::held_prefix_len;
        assert_eq!(held_prefix_len("abc</thi", "</think>"), 5);
        assert_eq!(held_prefix_len("abc<", "</think>"), 1);
        assert_eq!(held_prefix_len("abc", "</think>"), 0);
        assert_eq!(held_prefix_len("</think>", "</think>"), 0);
        assert_eq!(held_prefix_len("é", "</think>"), 0);
    }
}
