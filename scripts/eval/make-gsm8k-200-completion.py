#!/usr/bin/env python3
"""Converts the GSM8K test-split JSONL (with full reference solutions) into the
completion-form eval-fixture format: tests/eval/gsm8k-200-completion.jsonl.

Invoked by scripts/eval/make-gsm8k-200-completion.sh; not run directly. Builds a fixed
few-shot chain-of-thought completion prompt (no chat template) for the first 200 test
items, so a base checkpoint with no chat template can be evaluated the same way the
chat-form tests/eval/gsm8k-200.jsonl evaluates an Instruct checkpoint.

Few-shot exemplars: items 200..207 of the same source file (8 shots), chosen because
they are outside the 200 scored items and, unlike tests/eval/gsm8k-full.jsonl, this
source file carries each exemplar's full worked solution (gsm8k-full.jsonl keeps only
the final number). Every exemplar and every scored item uses the same
"Question: ... / Answer: <steps> / Answer: <number>" shape.
"""

import json
import re
import sys

CALC_RE = re.compile(r"<<[^>]*>>")
INSTRUCTION = (
    'Solve the problem step by step. On the last line, write "Answer: " '
    "followed by the final answer as a number."
)
NUM_SHOTS = 8
SHOT_START = 200


def clean_solution(raw_answer: str) -> tuple[str, str]:
    """Splits a raw GSM8K `answer` field into (steps, final_number), calculator
    annotations (`<<...>>`) stripped from the steps."""
    steps_part, _, tail = raw_answer.partition("#### ")
    steps = CALC_RE.sub("", steps_part).strip()
    number = tail.strip().replace(",", "")
    return steps, number


def render_shot(question: str, steps: str, number: str) -> str:
    return f"Question: {question}\n\n{INSTRUCTION}\n{steps}\nAnswer: {number}"


def render_query(question: str) -> str:
    return f"Question: {question}\n\n{INSTRUCTION}\n"


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    shots = []
    for row in rows[SHOT_START : SHOT_START + NUM_SHOTS]:
        steps, number = clean_solution(row["answer"])
        shots.append(render_shot(row["question"], steps, number))
    preamble = "\n\n".join(shots)

    with open(out, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows[:200]):
            _, number = clean_solution(row["answer"])
            prompt = preamble + "\n\n" + render_query(row["question"])
            record = {
                "id": f"gsm8k-test-{i:04d}-completion",
                "prompt": prompt,
                "answer": number,
                "match": "final_number",
                "max_tokens": 512,
                "stop": ["\n\nQuestion:"],
            }
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")

    print(
        f"wrote {out} (200 items, {NUM_SHOTS}-shot from items {SHOT_START}..{SHOT_START + NUM_SHOTS - 1})"
    )


if __name__ == "__main__":
    main()
