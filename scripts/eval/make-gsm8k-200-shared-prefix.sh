#!/usr/bin/env bash
# Regenerates tests/eval/gsm8k-200-shared-prefix.jsonl: the committed gsm8k-200.jsonl with one
# identical ~2000-word system message in front of every item (the lossy-KV gate variant, user
# decision 2026-10-01 "6b Task 6" point 1 A). Offline: reads only tests/eval/gsm8k-200.jsonl.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${2:-$HERE/../../tests/eval/gsm8k-200.jsonl}"
OUT="${1:-$HERE/../../tests/eval/gsm8k-200-shared-prefix.jsonl}"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
python3 "$HERE/make-gsm8k-200-shared-prefix.py" "$SRC" "$TMP"
test "$(wc -l <"$TMP" | tr -d ' ')" = 200
chmod 644 "$TMP"
mv "$TMP" "$OUT"
echo "wrote $OUT (200 items)"
