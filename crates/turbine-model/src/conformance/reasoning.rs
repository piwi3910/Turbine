//! The `reasoning_parser` suite: every registered parser splits its own sample
//! ([`ReasoningParser::sample`]) into the sample's reasoning and content, whether the output
//! arrives whole, one char per push, or in two pieces split at any char boundary.

use turbine_core::registry::Registry;

use super::{ConformanceFailure, Report, ensure};
use crate::reasoning::{ReasoningDelta, ReasoningParser};

/// Runs the output `pieces` through a stream started on `tail`, finish included.
fn split(parser: &dyn ReasoningParser, tail: &str, pieces: &[&str]) -> ReasoningDelta {
    let mut stream = parser.start(tail);
    let mut out = ReasoningDelta::default();
    for piece in pieces {
        out.append(stream.push(piece));
    }
    out.append(stream.finish());
    out
}

/// Runs the suite over every parser of `reg`.
pub fn suite(reg: &Registry<dyn ReasoningParser>) -> Result<(), Vec<ConformanceFailure>> {
    let mut report = Report::new(reg);
    for parser in reg.iter() {
        let name = parser.name();
        let s = parser.sample();
        let want = ReasoningDelta {
            reasoning: s.reasoning.to_string(),
            content: s.content.to_string(),
        };
        let check = |pieces: &[&str], how: &str| {
            let got = split(parser, s.prompt_tail, pieces);
            ensure(got == want, || {
                format!("{how}: got {got:?}, want {want:?} (pieces {pieces:?})")
            })
        };
        report.check(name, "whole", || check(&[s.output], "whole"));
        report.check(name, "per_char", || {
            let chars: Vec<String> = s.output.chars().map(String::from).collect();
            let pieces: Vec<&str> = chars.iter().map(String::as_str).collect();
            check(&pieces, "one char per push")
        });
        report.check(name, "two_pieces", || {
            for (at, _) in s.output.char_indices().skip(1) {
                let (a, b) = s.output.split_at(at);
                check(&[a, b], "two pieces")?;
            }
            Ok(())
        });
        report.check(name, "empty_pushes", || {
            check(&["", s.output, ""], "empty pushes around the output")
        });
    }
    report.finish()
}
