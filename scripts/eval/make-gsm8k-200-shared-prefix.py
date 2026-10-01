#!/usr/bin/env python3
"""Builds tests/eval/gsm8k-200-shared-prefix.jsonl from tests/eval/gsm8k-200.jsonl.

Invoked by scripts/eval/make-gsm8k-200-shared-prefix.sh. The 200 items, their user message
(question plus the chain-of-thought instruction), answer, matcher and max_tokens are
copied unchanged; each item gets one extra leading `system` message, identical in every
record: a fixed preamble plus PREFIX_WORDS pseudo-words. The words are the bench's seeded
synthetic prompt (benches/turbine-bench/src/prompt.rs: built-in 1,000-word list, SplitMix64
seeded with PREFIX_SEED + index 0), ported here byte for byte; the Rust test
`eval_task_set_shared_prefix_valid` regenerates the text with that Rust code and compares,
so the two cannot drift. The text is deterministic, license-free and needs no network.
"""

import json
import sys

PREFIX_SEED = 20261001
PREFIX_WORDS = 2000
PREAMBLE = (
    "Background notes for a tutoring session. They are unrelated to the math "
    "problems that follow; ignore them when solving.\n\n"
)

ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "t"]
NUCLEI = ["a", "e", "i", "o", "u", "ai", "ea", "io", "ou", "ue"]
CODAS = ["", "n", "r", "s", "t", "l", "m", "x", "nd", "st"]
MASK = (1 << 64) - 1


def word(i: int) -> str:
    i %= 1000
    return ONSETS[i // 100] + NUCLEI[(i // 10) % 10] + CODAS[i % 10]


def prompt(seed: int, index: int, words: int) -> str:
    state = (seed + index) & MASK
    out = []
    for _ in range(words):
        state = (state + 0x9E3779B97F4A7C15) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        z ^= z >> 31
        out.append(word(z % 1000))
    return " ".join(out)


def main(src: str, dst: str) -> None:
    system = PREAMBLE + prompt(PREFIX_SEED, 0, PREFIX_WORDS)
    lines = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            task = json.loads(line)
            assert [m["role"] for m in task["messages"]] == ["user"]
            task["messages"] = [{"role": "system", "content": system}] + task["messages"]
            lines.append(json.dumps(task, ensure_ascii=False, separators=(",", ":")))
    assert len(lines) == 200
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
