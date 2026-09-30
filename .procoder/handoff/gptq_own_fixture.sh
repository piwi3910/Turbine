#!/usr/bin/env bash
# Phase 6a Task 18: golden reference + self-spread for our own GPTQ checkpoint
# (llm-compressor W4A16 sym g128 damp 0.01, no act-order, compressed-tensors pack-quantized ->
# Turbine ct_pack_int4 -> gptq_int4). CPU only, on novanas, detached
# (setsid nohup bash gptq_own_fixture.sh </dev/null &), under fixture.lock for its whole length,
# nice 19 on cores 12-15 with 4 threads, like t14_spread.sh / fixtures6.sh (never bench.lock; the
# GPU drivers SIGSTOP it through 'scripts/golden/').
#
# ALL EIGHT self-spread variants in one run (lesson of the MXFP4 p05 decision 2026-09-30: judging
# only a partial set of variants missed a knife-edge decode position that a later variant caught).
#
# 1. quant_reference.py --act-quant none (weight-only W4A16) against the own checkpoint, keeping
#    the dequantized BF16 copy.
# 2. self_spread.py over all eight variants on that dequantized copy against the reference just
#    written.
# 3. Copies reference.jsonl / spread.json into tests/golden/llama-3.2-3b-instruct-gptq-own/.
# Last log line: "gptq-own-fixture: done rc=<rc>".
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
UV=/home/piwi/.local/bin/uv
SRC=/home/piwi/turbine-ci/remote/agent-p6a-gptq-numerics/src
M=/home/piwi/turbine-models/llama-3.2-3b-instruct-gptq-own
OUT=/home/piwi/turbine-ci/scratch/p6a-gptq-own/fixture
FIXDIR="$SRC/tests/golden/llama-3.2-3b-instruct-gptq-own"
ALL="bf16-sdpa-incremental,bf16-sdpa-full,bf16-eager-incremental,bf16-eager-full,fp32-sdpa-incremental,fp32-sdpa-full,fp32-eager-incremental,fp32-eager-full"
LOCK=/home/piwi/turbine-ci/fixture.lock
mkdir -p "$OUT" "$FIXDIR"
LOG=$OUT/gptq-own-fixture.log

run() { # <timeout> <log> <uv run args...>
	local t=$1 log=$2
	shift 2
	env OMP_NUM_THREADS=4 \
		MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
		timeout "$t" "$UV" run "$@" >"$log" 2>&1 </dev/null
}

main() {
	cd "$SRC" || return 1
	echo "gptq-own-fixture: $(date '+%F %T') fixture.lock held"
	local avail
	avail=$(df --output=avail -B1G /dev/shm | tail -1 | tr -d ' ')
	echo "gptq-own-fixture: /dev/shm ${avail} GiB free"
	if [ "$avail" -lt 12 ]; then
		echo "gptq-own-fixture: less than 12 GiB free in /dev/shm; not started"
		return 3
	fi
	WORK=$(mktemp -d /dev/shm/turbine-gptq-own-XXXXXX)
	trap 'rm -rf "$WORK"' EXIT
	echo "gptq-own-fixture: $(date '+%F %T') reference (work $WORK)"
	run 2h "$OUT/reference.log" scripts/golden/quant_reference.py --model-dir "$M" \
		--prompts tests/golden/prompts.jsonl --out "$OUT/reference.jsonl" --act-quant none \
		--work-dir "$WORK" --keep-dequantized \
		--model-name turbine/Llama-3.2-3B-Instruct-GPTQ-own
	local rc=$?
	echo "gptq-own-fixture: reference rc=$rc"
	tail -5 "$OUT/reference.log"
	[ "$rc" = 0 ] || return "$rc"
	grep -m1 '"rotary"\|"rope"' "$OUT/reference.log" || true
	local bf16
	bf16=$(find "$WORK" -mindepth 1 -maxdepth 3 -name bf16 -type d | head -1)
	[ -n "$bf16" ] || {
		echo "gptq-own-fixture: no dequantized copy under $WORK"
		return 1
	}
	echo "gptq-own-fixture: $(date '+%F %T') self_spread (all 8) on $bf16"
	run 20h "$OUT/spread.log" scripts/golden/self_spread.py "$bf16" "$OUT/reference.jsonl" \
		"$OUT/spread.json" "$ALL"
	rc=$?
	echo "gptq-own-fixture: self_spread rc=$rc"
	tail -20 "$OUT/spread.log"
	[ "$rc" = 0 ] || return "$rc"
	cp "$OUT/reference.jsonl" "$FIXDIR/reference.jsonl"
	cp "$OUT/spread.json" "$FIXDIR/spread.json"
	echo "gptq-own-fixture: wrote $FIXDIR/reference.jsonl and spread.json"
	return 0
}

# The whole script runs as one fixture job: the waiter's command line names fixture.lock and
# FIXTURE_JOB=gptq-own-fixture, so scripts/lab/fixture-order.sh can rank it (fixture.queue).
if [ "${1:-}" = --locked ]; then
	main
	exit $?
fi
{
	echo "gptq-own-fixture: $(date '+%F %T') waiting for fixture.lock"
	flock "$LOCK" nice -n 19 taskset -c 12-15 env FIXTURE_JOB=gptq-own-fixture bash "$0" --locked
	echo "gptq-own-fixture: done rc=$?"
} >>"$LOG" 2>&1
