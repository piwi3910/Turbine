# Adding a reasoning parser

A reasoning parser is how one model's output splits into the tokens it thought with (`reasoning_content` in the OpenAI response) and the tokens it answered with (`content`), as the text streams. Point name `reasoning_parser`; selected by `model.reasoning_parser` (`auto` = the family's `default_reasoning_parser()`, `none` = no split); the server starts one stream per choice (one per `n`) from the last bytes of the rendered prompt and feeds it every decoded text piece. Registered: `qwen_think`, `harmony`.

## The trait

`turbine_model::reasoning::ReasoningParser: Module` (`crates/turbine-model/src/reasoning/mod.rs`):

| Method                    | Must do                                                                                                                                                                                                                                                                          |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name()` (from `Module`)  | The configuration name, `^[a-z0-9_]{1,64}$`; also the `parser` label of the `reasoning_parser` module entry in `/turbine/v1/status`.                                                                                                                                            |
| `start(&str)`             | A fresh `Box<dyn ReasoningStream>` for one choice. `prompt_tail` is the end of the rendered prompt: it decides the opening mode (e.g. a generation prompt already ending in an unclosed ` thinking\n` starts in reasoning). Give every choice its own stream — state is never shared. |
| `sample()`                | A known `ReasoningSample { prompt_tail, output, reasoning, content }` the conformance suite feeds whole, one char per push and in two pieces at every char boundary, expecting the same split each time.                                                                            |

The stream (`ReasoningStream: Send`, returned by `start`):

| Method                    | Must do                                                                                                                                                                                                                                                                    |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `push(&mut str) -> ReasoningDelta` | Take the next decoded text (may be empty) and return what it decided now. Text that may still turn out to be part of a marker is *held back* until the next push or `finish` (a proper prefix of a marker at a char boundary).                                    |
| `finish() -> ReasoningDelta`       | Release everything still held: a marker never closed before EOS / `max_tokens` stays whatever the parser's open mode was (e.g. all reasoning).                                                                                                                     |

`ReasoningDelta { reasoning: String, content: String }` (`Default`, `PartialEq`; either side may be empty). For a tool-call format, keep the tool-call text out of `content` (the tool parser reads the whole output separately); a channel that is neither reasoning nor content is withheld.

## Files to add

1. `crates/turbine-model/src/reasoning/<name>.rs` — the parser and its stream (see `crates/turbine-model/src/reasoning/qwen_think.rs`; Harmony's channel state machine is `crates/turbine-model/src/reasoning/harmony.rs`):

```rust
//! `bracket_think`: text in `[think]…[/think]` is reasoning.
use turbine_core::registry::Module;

use crate::reasoning::{ReasoningDelta, ReasoningParser, ReasoningSample, ReasoningStream, held_prefix_len};

pub const BRACKET_THINK: &str = "bracket_think";

pub struct BracketThink;

impl Module for BracketThink {
    fn name(&self) -> &'static str { BRACKET_THINK }
}

impl ReasoningParser for BracketThink {
    fn start(&self, prompt_tail: &str) -> Box<dyn ReasoningStream> { Box::new(Stream::new(prompt_tail)) }
    fn sample(&self) -> ReasoningSample {
        ReasoningSample {
            prompt_tail: "assistant\n",
            output: "[think]plan[/think]answer",
            reasoning: "plan",
            content: "answer",
        }
    }
}
```

2. If the family should use it by default, return its name from that family's `default_reasoning_parser()` (`crates/turbine-model/src/families/<name>.rs`).
3. If its marker arrives split across pushes, hold the longest proper marker prefix at a char boundary with `held_prefix_len(text, marker)` (used by both existing parsers) so no half marker leaks into `content` or `reasoning`.

## Registry entry

In `crates/turbine-model/src/reasoning/mod.rs`: `pub mod <name>;`, `pub use <name>::<Type>;`, and `&<Type>` appended to the `REASONING_PARSERS` list (`registry()`). Registration order is the order the registry's `names` test pins — update `reasoning::registry_conformance::names` with yours. The server resolves `model.reasoning_parser` against this registry (`crates/turbine-server/src/modules.rs`, `resolve_reasoning_parser`); an unregistered name exits 2 before any port is bound and no server change is needed.

## Conformance suite

`reasoning_suite` (`crates/turbine-model/src/conformance/reasoning.rs`) runs over the registry: every parser splits its own `sample()` into the sample's `reasoning` and `content` when the output arrives `whole`, one char per push (`per_char`), in two pieces at every char boundary (`two_pieces`) and with empty pushes around it (`empty_pushes`) — it fails if a parser loses or duplicates text across pushes or leaks a marker into the output. `held_prefix` unit-tests the hold-back helper.

- `scripts/remote-cargo.sh test -p turbine-model reasoning::registry_conformance` — the suite over the registry plus the pinned names.
- `scripts/remote-cargo.sh test -p turbine-model reasoning::` — the per-parser tests in your file.
- `scripts/remote-cargo.sh test -p turbine-model --test conformance` — the suite still rejects broken modules.
- `scripts/remote-cargo.sh test -p turbine-server --test tiny_server reasoning_content_follows_enable_thinking` — `reasoning_content` end to end on the tiny server (non-streamed field, streamed deltas, `modules.reasoning_parser`, the metric).

## Lab checks

A parser a served model does not use changes nothing on the GPU path. When it becomes a served family's parser: serve it (`scripts/lab-serve.sh novanas scripts/lab/phase7-novanas-<model>.yaml --set model.reasoning_parser=<name>`), then `TURBINE_LAB_URL=http://192.168.10.203:18000 scripts/remote-cargo.sh test -p turbine-server --test lab_openai harmony_channels -- --ignored` (non-streamed `reasoning_content` and `content`, the streamed deltas concatenating to the same, `n: 2` carrying independent streams, and `reasoning_effort` reaching the template), and stop the serve Job with `scripts/lab-serve.sh novanas --stop`.

## Pitfalls

- **Hold back, never trim**: text that may still complete a marker must be kept and re-emitted once the next push decides it — dropping it silently loses tokens. `held_prefix_len` finds the longest suffix that is a proper prefix of the marker at a char boundary; never split a multi-byte character.
- **Never lose text either way**: every pushed byte ends up in exactly one of `reasoning` or `content` (or is withheld for the tool parser), by `finish()` at the latest. The suite's per-char and two-piece passes catch a delta that drops or duplicates a boundary.
- **`prompt_tail` decides the opening mode**: the generation prompt may already open the reasoning (`…assistant\n thinking\n` with thinking on, `…\n\n</think>\n\n` with it off), so the same output can split differently. Do not look past the tail you were given.
- **A marker never closed** is not an error: whatever the open mode was holds through `finish()` (Harmony's analysis channel and Qwen's thinking both stay reasoning to EOS).
- **Independent streams for `n > 1`**: the parser must be stateless across choices — the server builds one stream per choice from `start`; a shared `static` or interior mutability leaks one choice's mode into another.
- **Keep the tool-call text out of `content`**: Harmony's commentary channel (a tool call) is withheld so the tool parser sees the raw output; folding it into `content` would double-render the call as prose.
- **One parser, one marker family**: the split is the model's own convention; a parser that guesses conservatively (holds back too much) stalls streaming until `finish`, one that guesses eagerly leaks markers.
