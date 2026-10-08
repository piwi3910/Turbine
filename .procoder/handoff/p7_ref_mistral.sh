#!/usr/bin/env bash
# Phase 7 Task 20: transformers reference for Mistral-7B-Instruct-v0.3 (CPU, novanas, detached:
# setsid nohup bash p7_ref_mistral.sh </dev/null &). Under fixture.lock, nice 19 on cores 12-15, 4 threads.
# Last log line: "p7-ref-mistral: done rc=<rc>".
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
UV=/home/piwi/.local/bin/uv
SRC=/home/piwi/turbine-ci/scratch/p7-t20/src
M=/home/piwi/turbine-models/mistral-7b-instruct-v0.3
OUT=/home/piwi/turbine-ci/scratch/p7-t20/out
LOCK=/home/piwi/turbine-ci/fixture.lock
LOG=$OUT/p7-ref-mistral.log
mkdir -p "$OUT"
if [ "${1:-}" = --locked ]; then
	cd "$SRC" || exit 1
	echo "p7-ref-mistral: $(date '+%F %T') fixture.lock held"
	env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
		timeout 4h "$UV" run --with protobuf --with sentencepiece scripts/golden/hf_reference.py --model-dir "$M" \
		--prompts tests/golden/prompts.jsonl --out "$OUT/reference.jsonl" \
		--model-name mistralai/Mistral-7B-Instruct-v0.3 >"$OUT/reference.log" 2>&1 </dev/null
	rc=$?
	tail -5 "$OUT/reference.log"
	exit $rc
fi
{
	echo "p7-ref-mistral: $(date '+%F %T') waiting for fixture.lock"
	flock "$LOCK" nice -n 19 taskset -c 12-15 env FIXTURE_JOB=p7-ref-mistral bash "$0" --locked
	echo "p7-ref-mistral: done rc=$?"
} >>"$LOG" 2>&1
