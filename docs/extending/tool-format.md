# Adding a tool-call format

A tool format is how a model writes tool calls: the special tokens it needs, the llguidance grammar that constrains `tool_choice` `auto` / `required` / named output, the parser that turns a finished output into `tool_calls`, and the `auto`-mode test of whether an output opens like a call. Point name `tool_format`; selected by `model.tool_call_parser`, else the family's `default_tool_format()`; bound to the served tokenizer once at startup (`formats::bind`). Registered: `llama3_json`, `hermes`, `mistral`, `harmony`, `qwen3_xml`.

## The trait

`turbine_model::formats::ToolFormat: Module` (`crates/turbine-model/src/formats/mod.rs`):

| Method                                             | Must do                                                                                                                                                                                                                                                                                                                        |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `name()` (from `Module`)                           | The configuration name, also the `parser` label of `turbine_tool_calls_total`.                                                                                                                                                                                                                                                 |
| `special_tokens()`                                 | The format's special tokens by text; `required: true` refuses the format at startup on a tokenizer lacking one. Prefer optional tokens that the parser also accepts as plain text; the grammar references them as bare marker atoms (llguidance special tokens, `envelope::lark_envelope`), never inside quoted literals.      |
| `render_tools(&[Value])`                           | Default `None`: the model's chat template renders `tools`. Only override for a template that cannot.                                                                                                                                                                                                                           |
| `grammar(tools, &ToolChoice, parallel)`            | A `ConstraintSpec::ToolCall { grammar_source }` (Lark for llguidance). `auto` = free text not starting like a call, or the calls; `required` = one or more calls (`parallel`); named = exactly that function. `none`, no tools or an unknown name → `ModelError::Constraint` naming the field (use `envelope::allowed_tools`). |
| `parser()`                                         | A fresh `Box<dyn ToolCallParser>` whose `parse(text)` returns `ToolParse::Calls` for well-formed calls and `ToolParse::Content` (unchanged text) for anything else. Call ids come from `envelope::CallIds`.                                                                                                                    |
| `opens_like_call(text, first_token, &BoundTokens)` | `auto` streaming: `Opening::Call` (hold for the parser), `Content` (stream it) or `Undecided` (e.g. whitespace or a prefix of the opener so far).                                                                                                                                                                              |
| `sample_call()`                                    | One call as the model writes it: `get_weather` with `{"location": "Oslo", "unit": "celsius"}` (the suite's `fixture_tools`).                                                                                                                                                                                                   |

## Formats that ship with the tool family

- **`hermes`** (`hermes` / `hermes2` templates): `<tool_call>{"name": …, "arguments": {…}}</tool_call>` for Qwen3 and Qwen3-MoE.
- **`mistral`**: `[TOOL_CALLS] [{…}]` for Mistral and Mixtral.
- **`llama3_json`**: the Llama 3.1/3.2 python-tag JSON form.
- **`harmony`** (gpt-oss, Phase 7 S-9): the OpenAI harmony channel syntax. A call is a commentary-channel message addressed to a function — `<|start|>assistant<|channel|>commentary to=functions.<name> <|constrain|>json<|message|>{…}<|call|>`; the analysis channel's text is reasoning (the `harmony` reasoning parser reads it, see [reasoning-parser.md](reasoning-parser.md)) and the final channel is content. `special_tokens` requires `<|start|>`, `<|channel|>`, `<|message|>`, `<|constrain|>`, `<|call|>`, `<|end|>` and `<|return|>`; `grammar` constrains `required` / named calls to the commentary-channel call form (the analysis channel is left free for reasoning). A tool call written in the analysis channel is still parsed as a call.
- **`qwen3_xml`** (Qwen3.5/3.6, Phase 7 S-18): the XML tool form their chat template renders — `<tool_call>\n<function=<name>\n<parameter=<key>\n<value>\n…</parameter>\n</function>\n</tool_call>` (`TOOL_CALL_OPEN` / `TOOL_CALL_CLOSE` / `FUNCTION_OPEN` / `FUNCTION_CLOSE` / `PARAMETER_OPEN` / `PARAMETER_CLOSE`; the markers bind as plain text, the format declares no required special tokens). A family uses it by default through `default_tool_format()` (`qwen3_xml` for Qwen3.5/3.6).

Adding one of these to a family is only `default_tool_format()` in its module — the format itself is already registered.

## Files to add

`crates/turbine-model/src/formats/<name>.rs` (see `crates/turbine-model/src/formats/hermes.rs`; enveloped formats reuse `crates/turbine-model/src/formats/envelope.rs`):

```rust
pub const PYTHONIC: &str = "pythonic";

pub struct Pythonic;

impl Module for Pythonic {
    fn name(&self) -> &'static str { PYTHONIC }
}

impl ToolFormat for Pythonic {
    fn special_tokens(&self) -> &'static [SpecialToken] { &[] }
    fn grammar(&self, tools: &[Value], choice: &ToolChoice, parallel: bool) -> Result<ConstraintSpec, ModelError> {
        let allowed = allowed_tools(tools, choice)?;
        // build `start:` for auto / required / named plus one rule per allowed tool
        Ok(ConstraintSpec::ToolCall { grammar_source: todo!() })
    }
    fn parser(&self) -> Box<dyn ToolCallParser> { Box::new(PythonicParser::new()) }
    fn opens_like_call(&self, text: &str, _first: Option<u32>, _t: &BoundTokens) -> Opening {
        let s = text.trim_start();
        if s.starts_with('[') { Opening::Call } else if s.is_empty() { Opening::Undecided } else { Opening::Content }
    }
    fn sample_call(&self) -> &'static str { r#"[get_weather(location="Oslo", unit="celsius")]"# }
}
```

If a family should use it by default, return its name from that family's `default_tool_format()` (`crates/turbine-model/src/families/<name>.rs`).

## Registry entry

In `crates/turbine-model/src/formats/mod.rs`: `pub mod <name>;`, `pub use <name>::{<Type>, <Type>Parser};`, and `&<Type>` appended to `TOOL_FORMATS`. Add the name to the pinned list in `crates/turbine-model/src/registries.rs` (`registry_conformance::tool_formats`). The server validates `model.tool_call_parser` against the registry (`crates/turbine-server/src/modules.rs`) — no server change.

## Conformance suite

`formats_suite` (`crates/turbine-model/src/conformance/formats.rs`), over the registry on the tiny tokenizer: `special_tokens` (non-empty texts, distinct, binds to the tiny tokenizer), `render`, `grammar` (compiles `auto` parallel, `required` single and parallel, named `get_weather`; refuses `none`, no tools, an unknown name), `parse` (the sample call → the `get_weather` call; plain text stays content), `round_trip` (the `required`, `auto` and named matchers accept the sample call token by token; `required` refuses plain text), `opening`.

- `scripts/remote-cargo.sh test -p turbine-model registry_conformance`
- `scripts/remote-cargo.sh test -p turbine-model formats::` — the per-format parser tests in your file and `crates/turbine-model/src/formats/tests.rs`.
- `scripts/remote-cargo.sh test -p turbine-server --test server_cli --test tiny_server` — `model.tool_call_parser` validation and `tool_choice_modes` end to end on the tiny server.

## Lab checks

A format no served model uses changes nothing on the GPU path. When it becomes the format of a served GPU family: serve it (`scripts/lab-serve.sh novanas <config> --set model.tool_call_parser=<name>`) and run `scripts/lab-test.sh novanas -- -p turbine-server --test lab_openai` (`tools_and_json_schema` against the served model), then stop the serve Job with `scripts/lab-serve.sh novanas --stop`.

## Pitfalls

- The suite adds every registered format's `special_tokens()` to the fixture tokenizer before building its grammar compiler, so a grammar can reference its markers as special-token atoms. The server still refuses the format on a real tokenizer lacking a `required: true` token; mark tokens optional and let the parser also accept their plain text, as `hermes` does.
- The parser must never lose text: anything that is not exactly a well-formed call returns `ToolParse::Content` with the input unchanged (the server then streams it as content).
- The grammar and the parser must agree: `round_trip` feeds `sample_call()` through each compiled grammar token by token; whitespace the grammar forbids but the model writes makes constrained requests fail on the lab even when parsing works.
- Never put format-specific strings (`<|python_tag|>`, `[TOOL_CALLS]`) in `crates/turbine-server`; everything the server needs goes through `BoundToolFormat`.
- `turbine_tool_calls_total{parser}` is labelled with the name: keep it a short fixed word.
- **Harmony's markers are all required**: a real gpt-oss tokenizer carries them, but a checkpoint whose tokenizer lacks one refuses the format at startup. Its channel state must survive markers arriving split across tokens — the parser withholds a partial marker rather than emitting it (the `harmony_round_trip` test feeds the output one char at a time).
- **`qwen3_xml`'s markers bind as plain text**: unlike harmony they are not special tokens, so a tokenizer need not carry them and the parser matches them in the decoded string; a stray `</parameter>` the model writes mid-value must not truncate the call. Its grammar is the template's form, not JSON, so a constrained `required` call is XML and the parser must accept exactly that.
