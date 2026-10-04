#!/usr/bin/env python3
"""Converts the GSM8K test-split parquet file into the eval-fixture JSONL format.

Invoked by scripts/eval/make-gsm8k-full.sh; not run directly. Same id scheme,
user-message wrapper, answer extraction and match/max_tokens fields as
scripts/eval/make-gsm8k-200.sh's jq filter, so that the first 200 lines are
byte-identical to the committed tests/eval/gsm8k-200.jsonl.
"""

import json
import sys

import pyarrow.parquet as pq


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    rows = pq.read_table(src).to_pylist()

    with open(out, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            question = row["question"]
            answer = row["answer"].split("#### ")[1].replace(",", "").strip()
            record = {
                "id": f"gsm8k-test-{i:04d}",
                "messages": [
                    {
                        "role": "user",
                        "content": question
                        + "\n\nSolve the problem step by step. On the last line, "
                        'write "Answer: " followed by the final answer as a number.',
                    }
                ],
                "answer": answer,
                "match": "final_number",
                "max_tokens": 512,
            }
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")

    print(f"wrote {out} ({len(rows)} items)")


if __name__ == "__main__":
    main()
