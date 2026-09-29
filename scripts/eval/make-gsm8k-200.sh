#!/usr/bin/env bash
# Regenerates tests/eval/gsm8k-200.jsonl: the first 200 items of the GSM8K test split
# (openai/grade-school-math, MIT), pinned by commit. Fixture-generation time only.
set -euo pipefail
COMMIT=3101c7d5072418e28b9008a6636bde82a006892c
URL="https://raw.githubusercontent.com/openai/grade-school-math/${COMMIT}/grade_school_math/data/test.jsonl"
OUT="${1:-tests/eval/gsm8k-200.jsonl}"
SRC="$(mktemp)"
TMP="$(mktemp)"
trap 'rm -f "$SRC" "$TMP"' EXIT
curl -fsSL -o "$SRC" "$URL"
head -n 200 "$SRC" | jq -c -s '
  to_entries[] | {
    id: ("gsm8k-test-" + ((.key | tostring) as $k | ("000" + $k)[-4:])),
    messages: [{role: "user", content: (.value.question
      + "\n\nSolve the problem step by step. On the last line, write \"Answer: \" followed by the final answer as a number.")}],
    answer: (.value.answer | split("#### ")[1] | gsub(","; "") | gsub("^\\s+|\\s+$"; "")),
    match: "final_number",
    max_tokens: 512
  }' >"$TMP"
test "$(wc -l <"$TMP" | tr -d ' ')" = 200
mv "$TMP" "$OUT"
echo "wrote $OUT (200 items, commit $COMMIT)"
