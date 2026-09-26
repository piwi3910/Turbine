//! The P8 S-8 families' chat templates, as their `tokenizer_config.json` ship them
//! (`tests/fixtures/<slug>/`), rendered by `ChatTemplate` the way transformers'
//! `apply_chat_template` renders them: Qwen3 (ChatML with `<think>` handling, Hermes-style
//! `<tools>` and `<tool_call>` blocks), Mixtral (`[INST]`, strict role alternation) and
//! Mistral v0.3 (`[AVAILABLE_TOOLS]`). The expected strings follow the templates' own
//! control flow; each case says which branch it covers.
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};
use turbine_model::ChatTemplate;

fn fixture(slug: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(slug)
        .join("tokenizer_config.json")
}

fn render(
    template: &ChatTemplate,
    messages: &[Value],
    tools: Option<&[Value]>,
    kwargs: Value,
) -> String {
    let kwargs: Map<String, Value> = kwargs.as_object().cloned().unwrap_or_default();
    template
        .render(messages, tools, true, &kwargs)
        .expect("render")
}

fn weather_tool() -> Value {
    json!({"type": "function", "function": {
        "name": "get_weather",
        "description": "Weather in a city.",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
    }})
}

#[test]
fn qwen3_template() {
    let t = ChatTemplate::load(&fixture("qwen3-0.6b")).expect("Qwen3 template loads");
    assert!(t.renders_tools());
    let chat = [
        json!({"role": "system", "content": "You are helpful."}),
        json!({"role": "user", "content": "Hi"}),
    ];
    // System + user, generation prompt (thinking left to the model).
    let base = "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n\
                <|im_start|>assistant\n";
    assert_eq!(render(&t, &chat, None, json!({})), base);
    // `enable_thinking: false` pre-fills an empty reasoning block.
    assert_eq!(
        render(&t, &chat, None, json!({"enable_thinking": false})),
        format!("{base}<think>\n\n</think>\n\n")
    );

    // Tools go into the system turn, one `tojson` line each.
    let with_tools = render(&t, &chat, Some(&[weather_tool()]), json!({}));
    let want = concat!(
        "<|im_start|>system\nYou are helpful.\n\n# Tools\n\nYou may call one or more functions ",
        "to assist with the user query.\n\nYou are provided with function signatures within ",
        "<tools></tools> XML tags:\n<tools>\n",
        r#"{"type": "function", "function": {"name": "get_weather", "description": "Weather in a city.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}}"#,
        "\n</tools>\n\nFor each function call, return a json object with function name and ",
        "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": ",
        "<function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n",
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    );
    assert_eq!(with_tools, want);

    // A tool round trip: the assistant's call (after the last user query, no reasoning, not
    // the last message: no `<think>` block) and the tool's response as a user turn.
    let round_trip = [
        json!({"role": "user", "content": "Weather?"}),
        json!({"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "get_weather", "arguments": {"location": "Paris"}}}
        ]}),
        json!({"role": "tool", "content": "sunny"}),
    ];
    assert_eq!(
        render(&t, &round_trip, Some(&[weather_tool()]), json!({}))
            .split_once("<|im_start|>user\nWeather?")
            .expect("user turn")
            .1,
        concat!(
            "<|im_end|>\n<|im_start|>assistant\n<tool_call>\n",
            "{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n",
            "</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nsunny\n",
            "</tool_response><|im_end|>\n<|im_start|>assistant\n"
        )
    );

    // Earlier assistant turns keep only their answer: the `<think>` block is split off.
    let history = [
        json!({"role": "user", "content": "1+1?"}),
        json!({"role": "assistant", "content": "<think>\nadd\n</think>\n\n2"}),
        json!({"role": "user", "content": "2+2?"}),
    ];
    assert_eq!(
        render(&t, &history, None, json!({})),
        "<|im_start|>user\n1+1?<|im_end|>\n<|im_start|>assistant\n2<|im_end|>\n\
         <|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn mixtral_template() {
    let t = ChatTemplate::load(&fixture("mixtral-8x7b-instruct-v0.1")).expect("loads");
    assert!(!t.renders_tools());
    let chat = [
        json!({"role": "system", "content": "Be brief."}),
        json!({"role": "user", "content": "Hi"}),
        json!({"role": "assistant", "content": "Hello!"}),
        json!({"role": "user", "content": "Bye"}),
    ];
    assert_eq!(
        render(&t, &chat, None, json!({})),
        "<s> [INST] Be brief.\n\nHi [/INST] Hello!</s> [INST] Bye [/INST]"
    );
    // Roles must alternate: the template raises.
    let err = t
        .render(
            &[
                json!({"role": "user", "content": "a"}),
                json!({"role": "user", "content": "b"}),
            ],
            None,
            true,
            &Map::new(),
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("roles must alternate"), "{err}");
}

#[test]
fn mistral_v03_template_with_tools() {
    let t = ChatTemplate::load(&fixture("mistral-7b-instruct-v0.3")).expect("loads");
    assert!(t.renders_tools());
    let tools = [json!({"type": "function", "function": {
        "name": "get_time", "description": "Current time."
    }})];
    let chat = [json!({"role": "user", "content": "What time is it?"})];
    assert_eq!(
        render(&t, &chat, Some(&tools), json!({})),
        concat!(
            "<s>[AVAILABLE_TOOLS] [",
            r#"{"type": "function", "function": {"name": "get_time", "description": "Current time."}}"#,
            "][/AVAILABLE_TOOLS][INST] What time is it?[/INST]"
        )
    );
    // Without tools, and a system message folded into the last user turn.
    let chat = [
        json!({"role": "system", "content": "Be brief."}),
        json!({"role": "user", "content": "Hi"}),
    ];
    assert_eq!(
        render(&t, &chat, None, json!({})),
        "<s>[INST] Be brief.\n\nHi[/INST]"
    );
}
