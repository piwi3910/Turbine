# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "transformers==4.57.1",
#     "jinja2==3.1.6",
# ]
# ///
"""Chat-template render fixture generator for Turbine (P1 S-4, P2 S-18).

Fixture generation only: never part of the build or the serving path.

Renders three conversations with the tokenizer's chat template through
`AutoTokenizer.apply_chat_template(..., tokenize=False,
add_generation_prompt=True, date_string="26 Jul 2024")` and writes, as
`expected_renders.json`:

    {"transformers_version", "kwargs",
     "system_user": {"messages", "text", "ids"},
     "user_only": {"messages", "text", "ids"},
     "tools": {"messages", "tools", "text", "ids"}}

`tools` renders a `get_weather` tool (required `location` string, optional
`unit` enum), a user message, an assistant `tool_calls` message and a `tool`
result, with the tool list passed as `apply_chat_template(..., tools=...)`.

`ids` is `tokenizer.encode(text, add_special_tokens=False)` (the template emits
BOS itself). The Rust tests `chat_template::tests::renders_target_template` and
`chat_template::tests::renders_llama_tools` assert Turbine renders and
tokenizes the conversations identically.

Usage:
    uv run scripts/golden/render_fixture.py <tokenizer-dir> [--out <file>]

`<tokenizer-dir>` holds `tokenizer.json` and `tokenizer_config.json` (and
`chat_template.jinja` when the checkpoint ships one); `--out` defaults to
`<tokenizer-dir>/expected_renders.json`. Written to a temp file and renamed;
exit 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

KWARGS = {"date_string": "26 Jul 2024"}

CONVERSATIONS = {
    "system_user": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Réponds en français si on te le demande.",
        },
        {"role": "user", "content": "What is the capital of Belgium? 🇧🇪"},
    ],
    "user_only": [{"role": "user", "content": "  Hello 世界!  "}],
    "tools": [
        {"role": "user", "content": "What is the weather in Paris?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_abcdefghijklmnopqrstuvwx",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "Paris"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_abcdefghijklmnopqrstuvwx",
            "content": '{"temperature": 21}',
        },
    ],
}

# The tool list rendered with the conversation of the same name (OpenAI tool objects).
TOOLS = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
}


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # usage errors are failures too: exit 1
        self.print_usage(sys.stderr)
        print(f"render_fixture.py: error: {message}", file=sys.stderr)
        sys.exit(1)


def run(tokenizer_dir: Path, out: Path) -> None:
    import transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    fixture: dict = {
        "transformers_version": transformers.__version__,
        "kwargs": KWARGS,
    }
    for name, messages in CONVERSATIONS.items():
        tools = TOOLS.get(name)
        text = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **KWARGS,
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        entry: dict = {"messages": messages}
        if tools is not None:
            entry["tools"] = tools
        entry.update({"text": text, "ids": ids})
        fixture[name] = entry

    tmp = out.with_name(out.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(fixture, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    for name in CONVERSATIONS:
        print(f"{name}: {len(fixture[name]['ids'])} tokens", file=sys.stderr)


def main(argv: list[str]) -> int:
    p = _Parser(
        description="Generate the chat-template render fixture with transformers."
    )
    p.add_argument("tokenizer_dir", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    out = args.out or args.tokenizer_dir / "expected_renders.json"
    try:
        run(args.tokenizer_dir, out)
    except Exception as e:  # noqa: BLE001 -- any failure: exit 1, no partial file
        print(f"render_fixture.py: error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
