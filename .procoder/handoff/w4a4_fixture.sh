#!/usr/bin/env bash
# w4a4_fixture.sh — Phase 6a Task 20, W4A4 8B (mxfp4_a4) golden reference + self-spread.
# Runs ON novanas, self-contained, under fixture.lock (queued after fp8-block-self-spread /
# llama-3.1-8b per /home/piwi/turbine-ci/fixture.queue). CPU only, nice 19, taskset 12-15, 4
# threads, never bench.lock. Uses the p6a-mxfp4-golden worktree's scripts/golden/ (rope_parameters
# fix in dequantize_checkpoint.py: the raw checkpoint's transformers-5 rope_parameters block is
# now split into classic rope_theta/rope_scaling before the BF16 copy's config.json is written,
# so transformers 4.57.1 sees theta 500000 + llama3 scaling instead of silently defaulting to
# 10000/none — .procoder/handoff/p6a-w4a4-numerics.md's root cause).
# Last line always "w4a4-fixture: done rc=<rc>" (0 only when both reference and spread ran or
# were already complete).
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
LOCK=/home/piwi/turbine-ci/fixture.lock
UV=/home/piwi/.local/bin/uv
SRC=/home/piwi/turbine-ci/remote/agent-p6a-mxfp4-golden/src
MODEL=/home/piwi/turbine-models/llama-3.1-8b-instruct-mxfp4-a4
NAME=amd/Llama-3.1-8B-Instruct-MXFP4-W4A4-MLCAL-C1000-GPTQ
OUT=/home/piwi/turbine-ci/scratch/w4a4-fixture
mkdir -p "$OUT"
cd "$SRC" || {
	echo "w4a4-fixture: done rc=1 (no such dir $SRC)"
	exit 1
}

say() { echo "w4a4-fixture: $(date '+%F %T') $*"; }
finish() {
	say "done rc=$1"
	exit "$1"
}

ref_ok() { [[ -s "$1" ]] && [[ $(wc -l <"$1") -eq $(wc -l <"$2") ]]; }

REF=$OUT/llama-3.1-8b-instruct-mxfp4-a4.reference.jsonl
SPREAD=$OUT/llama-3.1-8b-instruct-mxfp4-a4.spread.json
V8="bf16-sdpa-full,bf16-eager-full,fp32-sdpa-full,fp32-eager-full"
WORK=$OUT/dequant-bf16
BF16=$WORK/bf16

if ref_ok "$REF" tests/golden/prompts.jsonl; then
	say "reference already complete, not re-run"
else
	rm -rf "$WORK"
	mkdir -p "$WORK"
	say "reference start (work $WORK, log $OUT/reference.log)"
	flock "$LOCK" nice -n 19 taskset -c 12-15 \
		env FIXTURE_JOB=w4a4-mxfp4-a4-reference OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
		CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
		timeout 24h "$UV" run scripts/golden/quant_reference.py \
		--model-dir "$MODEL" --prompts tests/golden/prompts.jsonl --out "$REF" \
		--act-quant mxfp4 --model-name "$NAME" --work-dir "$WORK" --keep-dequantized \
		>"$OUT/reference.log" 2>&1 </dev/null
	rc=$?
	say "reference rc=$rc"
	tail -20 "$OUT/reference.log" | cut -c1-220
	grep -m1 "rotary " "$OUT/reference.log" | tee "$OUT/rotary.txt"
	if [[ $rc -ne 0 ]] || ! ref_ok "$REF" tests/golden/prompts.jsonl; then
		finish 1
	fi
	found=$(find "$WORK" -mindepth 2 -maxdepth 2 -name bf16 -type d | head -1)
	if [[ -z "$found" ]]; then
		say "reference ok but no dequantized copy found under $WORK"
		finish 1
	fi
	BF16=$found
fi

if [[ ! -d "$BF16" ]]; then
	found=$(find "$WORK" -mindepth 2 -maxdepth 2 -name bf16 -type d 2>/dev/null | head -1)
	if [[ -n "$found" ]]; then
		BF16=$found
	else
		say "no dequantized copy on disk; re-dequantizing for the spread"
		rm -rf "$WORK"
		mkdir -p "$WORK"
		flock "$LOCK" nice -n 19 taskset -c 12-15 \
			env FIXTURE_JOB=w4a4-mxfp4-a4-dequantize OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
			timeout 4h "$UV" run scripts/golden/dequantize_checkpoint.py \
			--model-dir "$MODEL" --out "$BF16" \
			>"$OUT/dequant.log" 2>&1 </dev/null
		rc=$?
		say "dequantize rc=$rc"
		[[ $rc -ne 0 ]] && finish 1
	fi
fi

say "spread start (bf16 $BF16, log $OUT/spread.log)"
flock "$LOCK" nice -n 19 taskset -c 12-15 \
	env FIXTURE_JOB=w4a4-mxfp4-a4-spread OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
	CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
	timeout 24h "$UV" run scripts/golden/self_spread.py "$BF16" "$REF" "$SPREAD" "$V8" \
	--act-quant mxfp4 \
	>"$OUT/spread.log" 2>&1 </dev/null
rc=$?
say "spread rc=$rc"
tail -20 "$OUT/spread.log" | cut -c1-220

rm -rf "$WORK"
finish "$rc"
