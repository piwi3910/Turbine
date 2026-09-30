#!/usr/bin/env bash
# Regenerates tests/eval/gsm8k-200-completion.jsonl: the completion-form (no chat
# template) counterpart of tests/eval/gsm8k-200.jsonl, for base checkpoints that have
# no chat template. Same source and pin as make-gsm8k-200.sh (openai/grade-school-math,
# MIT), fetched once here so make-gsm8k-200-completion.py has each item's full worked
# solution to build few-shot exemplars from (make-gsm8k-200.sh's jq filter keeps only
# the final number). Fixture-generation time only.
set -euo pipefail
COMMIT=3101c7d5072418e28b9008a6636bde82a006892c
URL="https://raw.githubusercontent.com/openai/grade-school-math/${COMMIT}/grade_school_math/data/test.jsonl"
OUT="${1:-tests/eval/gsm8k-200-completion.jsonl}"
SRC="$(mktemp)"
TMP="$(mktemp)"
trap 'rm -f "$SRC" "$TMP"' EXIT
curl -fsSL -o "$SRC" "$URL"
python3 "$(dirname "$0")/make-gsm8k-200-completion.py" "$SRC" "$TMP"
test "$(wc -l <"$TMP" | tr -d ' ')" = 200
mv "$TMP" "$OUT"
echo "wrote $OUT (200 items, commit $COMMIT)"
