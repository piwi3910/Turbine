#!/usr/bin/env bash
# Regenerates tests/eval/gsm8k-full.jsonl: the full GSM8K test split (1,319 items,
# openai/gsm8k, config main, split test, MIT), pinned by dataset revision. The same
# id scheme, user-message wrapper, answer extraction and match/max_tokens as
# tests/eval/gsm8k-200.jsonl (scripts/eval/make-gsm8k-200.sh) — its first 200 items
# are byte-identical to that file. Fixture-generation time only; needs `hf` and `uv`
# (run on novanas: the Hugging Face token lives there, and nothing is uploaded).
set -euo pipefail
REVISION=740312add88f781978c0658806c59bc2815b9866
DATASET_DIR="${TURBINE_GSM8K_DATASET_DIR:-/home/piwi/turbine-ci/datasets/gsm8k}"
OUT="${1:-tests/eval/gsm8k-full.jsonl}"
PARQUET="$DATASET_DIR/main/test-00000-of-00001.parquet"

if [ ! -f "$PARQUET" ]; then
  mkdir -p "$DATASET_DIR"
  hf download openai/gsm8k --repo-type dataset --revision "$REVISION" \
    --local-dir "$DATASET_DIR" --include "main/test-00000-of-00001.parquet"
fi

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

uv run --with pyarrow python3 "$(dirname "$0")/make-gsm8k-full.py" "$PARQUET" "$TMP"

test "$(wc -l <"$TMP" | tr -d ' ')" = 1319
mv "$TMP" "$OUT"
echo "wrote $OUT (1319 items, dataset revision $REVISION)"
