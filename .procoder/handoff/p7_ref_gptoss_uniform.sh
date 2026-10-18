#!/usr/bin/env bash
# Phase 7 exit cleanup: regenerate the gpt-oss-20b golden reference uniformly (all 16 prompts,
# one session, the date-pinned Harmony template) so `captured` stamps are uniform instead of the
# mixed p01/p10-13 = 2026-10-18, p02-09 = 2026-10-14, p14-16 = 2026-10-16 set from Task 27/39c.
# Deterministic: T39c showed p01 regenerates bit-for-bit, so the committed and regenerated values
# are identical and the golden c1/c16 verdicts are unchanged.
#
# Detached on novanas: setsid nohup bash /home/piwi/turbine-ci/scratch/p7-t40/p7_ref_gptoss_uniform.sh </dev/null &
# Under fixture.lock, nice 19 on cores 12-15, 4 threads. Queue entry: p7-ref-gpt-oss-uniform.
# Last log line: "p7-ref-gpt-oss-uniform: done rc=<rc>".
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
UV=/home/piwi/.local/bin/uv
SRC=/home/piwi/turbine-ci/scratch/p7-t40/src
M=/home/piwi/turbine-models/gpt-oss-20b
BF=/home/piwi/turbine-models/gpt-oss-20b-bf16
OUT=/home/piwi/turbine-ci/scratch/p7-t40/out
LOCK=/home/piwi/turbine-ci/fixture.lock
LOG=$OUT/p7-ref-gpt-oss-uniform.log
mkdir -p "$OUT"
if [ "${1:-}" = --locked ]; then
	cd "$SRC" || exit 1
	if [ ! -f "$BF/config.json" ]; then
		echo "p7-ref-gpt-oss-uniform: dequantizing $M -> $BF"
		env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$UV" run scripts/golden/dequantize_checkpoint.py \
			--model-dir "$M" --out "$BF" >"$OUT/dequant.log" 2>&1 || { echo "dequant failed rc=$?"; exit 1; }
	fi
	echo "p7-ref-gpt-oss-uniform: generating the uniform 16-prompt reference (pinned template)"
	env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
		timeout 8h "$UV" run --with protobuf --with sentencepiece scripts/golden/hf_reference.py \
		--model-dir "$BF" --prompts tests/golden/gpt-oss-20b/prompts.jsonl --out "$OUT/reference.jsonl" \
		--model-name openai/gpt-oss-20b \
		--chat-template tests/golden/gpt-oss-20b/chat_template.jinja >"$OUT/reference.log" 2>&1
	rc=$?
	tail -3 "$OUT/reference.log"
	echo "p7-ref-gpt-oss-uniform: generation done rc=$rc"
	rm -rf "$BF"
	exit "$rc"
fi
{
	echo "p7-ref-gpt-oss-uniform: $(date '+%F %T') waiting for fixture.lock"
	flock "$LOCK" nice -n 19 taskset -c 12-15 env FIXTURE_JOB=p7-ref-gpt-oss-uniform bash "$0" --locked
	echo "p7-ref-gpt-oss-uniform: done rc=$?"
} >>"$LOG" 2>&1