#!/usr/bin/env bash
# Phase 7 Task 39c: make the gpt-oss-20b golden reference date-independent by regenerating it
# with the date-pinned Harmony template (tests/golden/gpt-oss-20b/chat_template.jinja,
# `strftime_now("%Y-%m-%d")` fixed to 2026-10-16) — the same file `model.chat_template` names in
# scripts/lab/phase7-novanas-gpt-oss-20b.yaml, so reference and candidate render identically.
#
# The records are not all regenerated (a full 16-prompt CPU run is ~4-6 h): a record is reused
# from the committed reference when its committed prompt_token_ids equal the pinned render —
# true for the 9 completion-kind records p01-p09 (the template is never consulted for them) and
# for p14-p16 (already rendered at the pinned date) — and regenerated otherwise (p10-p13, whose
# committed render used 2026-10-14). p01 is also regenerated (11 min) as an in-environment
# determinism proof: it must reproduce the committed p01 record bit-for-bit.
#
# Detached on novanas: setsid nohup bash .procoder/handoff/p7_t39c_ref.sh </dev/null &
# Under fixture.lock, nice 19 on cores 12-15, 4 threads. Queue entry name: p7-ref-gpt-oss.
# Last log line: "p7-t39c-ref: done rc=<rc>".
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
UV=/home/piwi/.local/bin/uv
SRC=/home/piwi/turbine-ci/scratch/p7-t39c/src
M=/home/piwi/turbine-models/gpt-oss-20b-bf16
OUT=/home/piwi/turbine-ci/scratch/p7-t39c/out
LOCK=/home/piwi/turbine-ci/fixture.lock
LOG=$OUT/p7-t39c-ref.log
mkdir -p "$OUT"
if [ "${1:-}" = --locked ]; then
	cd "$SRC" || exit 1
	if [ ! -f "$M/config.json" ]; then
		echo "p7-t39c-ref: $M missing; run the Task 27 fixture first"
		exit 1
	fi
	cat >"$OUT/prep.py" <<'PY'
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["transformers==4.57.1", "jinja2==3.1.6"]
# ///
"""Classify every golden record: cache it when its committed render equals the pinned render."""
import json
import sys
from pathlib import Path

SLUG = Path("tests/golden/gpt-oss-20b")
OUT = Path("/home/piwi/turbine-ci/scratch/p7-t39c/out")
# The committed render used 2026-10-14 for these four, so they must differ and be regenerated.
REGEN_DATE = {"p10", "p11", "p12", "p13"}
# Regenerated although it matches, as an in-environment determinism proof (see post.py).
REGEN_PROOF = {"p01"}

committed = {
    rec["id"]: rec
    for rec in (
        json.loads(line)
        for line in (SLUG / "reference.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
}
prompts = [
    json.loads(line)
    for line in (SLUG / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
tok = __import__("transformers").AutoTokenizer.from_pretrained(
    "/home/piwi/turbine-models/gpt-oss-20b-bf16"
)
tok.chat_template = (SLUG / "chat_template.jinja").read_text(encoding="utf-8")

seed = []
failed = False
for rec in prompts:
    rid = rec["id"]
    if rec["kind"] == "chat":
        text = tok.apply_chat_template(
            rec["messages"],
            tokenize=False,
            add_generation_prompt=True,
            **(rec.get("chat_template_kwargs") or {}),
        )
        ids = [int(i) for i in tok(text, add_special_tokens=False)["input_ids"]]
        pinned = "2026-10-16" in text
    else:
        ids = [int(i) for i in tok(rec["prompt"], add_special_tokens=True)["input_ids"]]
        pinned = None
    prior = committed.get(rid)
    old = [int(i) for i in prior["prompt_token_ids"]] if prior else None
    same = old == ids
    action = "regenerate" if rid in REGEN_DATE else "cache"
    if rid in REGEN_PROOF:
        action = "regenerate (determinism proof)"
    if rid in REGEN_DATE and same:
        print(f"{rid}: ERROR the committed render already equals the pinned render")
        failed = True
    if rid not in REGEN_DATE and not same:
        print(f"{rid}: ERROR a record meant to be cached does not match the pinned render")
        failed = True
    if rid not in REGEN_DATE and rid not in REGEN_PROOF and same:
        seed.append(prior)
    print(
        f"{rid}: kind={rec['kind']} pinned_date={pinned} committed_ids_equal_pinned={same} "
        f"action={action}"
    )
    if rec["kind"] == "chat" and pinned is not True:
        print(f"{rid}: ERROR pinned render does not carry 2026-10-16")
        failed = True
    if rid in REGEN_DATE and old is not None and len(old) == len(ids):
        diff = [i for i, (a, b) in enumerate(zip(old, ids)) if a != b]
        print(f"{rid}: committed vs pinned prompt ids differ at {len(diff)} index(es): {diff}")
if failed:
    sys.exit(1)
(OUT / "reference.jsonl.tmp").write_text(
    "".join(
        json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in seed
    ),
    encoding="utf-8",
)
print("seed:", ",".join(r["id"] for r in seed))
print("to regenerate:", ",".join(sorted(REGEN_DATE | REGEN_PROOF)))
PY
	cat >"$OUT/post.py" <<'PY'
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = []
# ///
"""p01 must reproduce the committed record bit-for-bit: the records cached above rely on it."""
import json
from pathlib import Path

SLUG = Path("tests/golden/gpt-oss-20b")
OUT = Path("/home/piwi/turbine-ci/scratch/p7-t39c/out")


def load(path):
    return {
        rec["id"]: rec
        for rec in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


old = load(SLUG / "reference.jsonl")["p01"]
new = load(OUT / "reference.jsonl")["p01"]
ok = old["prompt_token_ids"] == new["prompt_token_ids"] and old["tokens"] == new["tokens"]
worst = 0.0
for pa, pb in zip(old["top_logprobs"], new["top_logprobs"]):
    da = {int(i): float(v) for i, v in pa}
    db = {int(i): float(v) for i, v in pb}
    ok = ok and set(da) == set(db)
    worst = max([worst] + [abs(da[k] - db[k]) for k in da if k in db])
print(f"p01: tokens_equal={old['tokens'] == new['tokens']} top_k_ids_equal={ok} "
      f"max_abs_logprob_diff={worst:.3e}")
print("p01-MATCH" if ok and worst == 0.0 else "p01-DIFFER")
PY
	echo "p7-t39c-ref: $(date '+%F %T') fixture.lock held; classify records"
	env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
		timeout 1h "$UV" run --with protobuf --with sentencepiece "$OUT/prep.py" \
		>"$OUT/prep.log" 2>&1 </dev/null
	rc=$?
	cat "$OUT/prep.log"
	if [ "$rc" -ne 0 ]; then
		echo "p7-t39c-ref: classification failed rc=$rc"
		exit "$rc"
	fi
	echo "p7-t39c-ref: $(date '+%F %T') generating reference (resume: p01, p10-p13)"
	env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
		timeout 6h "$UV" run --with protobuf --with sentencepiece scripts/golden/hf_reference.py \
		--model-dir "$M" --prompts tests/golden/gpt-oss-20b/prompts.jsonl --out "$OUT/reference.jsonl" \
		--model-name openai/gpt-oss-20b \
		--chat-template tests/golden/gpt-oss-20b/chat_template.jinja >"$OUT/reference.log" 2>&1 </dev/null
	rc=$?
	tail -3 "$OUT/reference.log"
	if [ "$rc" -ne 0 ]; then
		echo "p7-t39c-ref: generation failed rc=$rc"
		exit "$rc"
	fi
	env OMP_NUM_THREADS=1 "$UV" run "$OUT/post.py" >"$OUT/post.log" 2>&1 </dev/null
	cat "$OUT/post.log"
	grep -q "p01-MATCH" "$OUT/post.log" || echo "p7-t39c-ref: WARNING p01 not bit-identical (see post.log)"
	exit 0
fi
{
	echo "p7-t39c-ref: $(date '+%F %T') waiting for fixture.lock"
	flock "$LOCK" nice -n 19 taskset -c 12-15 env FIXTURE_JOB=p7-ref-gpt-oss bash "$0" --locked
	echo "p7-t39c-ref: done rc=$?"
} >>"$LOG" 2>&1
