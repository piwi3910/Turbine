# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "transformers==4.57.1",
#     "jinja2==3.1.6",
# ]
# ///
"""Writes the prompt file of the YaRN golden (Phase 6a S-16, plan Task 28).

The output is the 16 golden prompts (`tests/golden/prompts.jsonl`, copied byte for byte) plus
one long chat prompt, id `p17-long`: the committed GSM8K questions
(`tests/eval/gsm8k-200.jsonl`, the question text without its answer instruction, which starts
at "\n\nSolve the problem") numbered and concatenated in file order until the rendered prompt holds at least `--target-tokens` tokens
(default 12,000), followed by a request to repeat question 1 word for word — an answer that
needs attention from the last position back to the start of the context, beyond the 8,192
positions the factor-16 YaRN override treats as the original context.

`turbine-golden compare` reads `prompts.jsonl` beside the reference by default, so this one file
serves both the 16 short prompts and the long one. Deterministic for a given tokenizer: the
tokenizer is read (never downloaded) only to count tokens.

usage: uv run scripts/golden/yarn_long_prompt.py --model-dir <llama-3.2-3b-instruct dir> \
    --out tests/golden/llama-3.2-3b-instruct-yarn16/prompts.jsonl
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

# Every GSM8K task ends with an answer instruction after this marker (its wording changed once;
# the question before it did not).
INSTRUCTION = "\n\nSolve the problem"
ASK = (
    "\n\nThe text above is a numbered list of math questions. Do not solve any of them. "
    "Repeat question 1 word for word."
)
KWARGS = {"date_string": "26 Jul 2024"}


def main() -> None:
    """Writes the prompt file named by --out."""
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--target-tokens", type=int, default=12000)
    p.add_argument("--short", type=Path, default=Path("tests/golden/prompts.jsonl"))
    p.add_argument("--questions", type=Path, default=Path("tests/eval/gsm8k-200.jsonl"))
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    questions = []
    for line in args.questions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            content = json.loads(line)["messages"][0]["content"]
            assert INSTRUCTION in content, content
            questions.append(content.split(INSTRUCTION)[0].strip())

    def render(n: int) -> tuple[list[dict], int]:
        body = "\n\n".join(
            f"Question {i + 1}: {q}" for i, q in enumerate(questions[:n])
        )
        messages = [{"role": "user", "content": body + ASK}]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **KWARGS
        )
        return messages, len(tok(text, add_special_tokens=False)["input_ids"])

    n, count = 1, 0
    while True:
        messages, count = render(n)
        if count >= args.target_tokens or n == len(questions):
            break
        n += 1
    if count < args.target_tokens:
        raise SystemExit(f"all {n} questions give only {count} tokens")
    long = {
        "id": "p17-long",
        "kind": "chat",
        "messages": messages,
        "max_tokens": 32,
        "chat_template_kwargs": KWARGS,
    }
    short = args.short.read_text(encoding="utf-8")
    assert short.endswith("\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        short + json.dumps(long, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"p17-long: {n} questions, {count} prompt tokens")


if __name__ == "__main__":
    main()
