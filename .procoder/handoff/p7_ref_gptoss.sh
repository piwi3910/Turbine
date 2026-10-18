#!/usr/bin/env bash
# Phase 7 Task 27: transformers reference for gpt-oss-20b on the dequantized MXFP4 -> BF16
# weights (CPU, novanas, detached: setsid nohup bash p7_ref_gptoss.sh </dev/null &).
# Under fixture.lock, nice 19 on cores 12-15, 4 threads. Stage 1 dequantizes the openai_mxfp4
# checkpoint to /home/piwi/turbine-models/gpt-oss-20b-bf16 (~40 GB) when that copy is absent;
# stage 2 runs hf_reference.py with reasoning_effort low (tests/golden/gpt-oss-20b/prompts.jsonl).
# Last log line: "p7-ref-gpt-oss: done rc=<rc>".
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
UV=/home/piwi/.local/bin/uv
SRC=/home/piwi/turbine-ci/scratch/p7-t27/src
Q=/home/piwi/turbine-models/gpt-oss-20b
M=/home/piwi/turbine-models/gpt-oss-20b-bf16
OUT=/home/piwi/turbine-ci/scratch/p7-t27/out
LOCK=/home/piwi/turbine-ci/fixture.lock
LOG=$OUT/p7-ref-gpt-oss.log
mkdir -p "$OUT"
if [ "${1:-}" = --locked ]; then
	cd "$SRC" || exit 1
	rc=0
	if [ ! -f "$M/config.json" ]; then
		echo "p7-ref-gpt-oss: $(date '+%F %T') fixture.lock held; dequantizing $Q -> $M"
		env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
			timeout 6h "$UV" run scripts/golden/dequantize_checkpoint.py --model-dir "$Q" --out "$M" \
			--verify-transformers 2 >"$OUT/dequantize.log" 2>&1 </dev/null
		rc=$?
		if [ "$rc" -ne 0 ]; then
			echo "p7-ref-gpt-oss: dequantize failed rc=$rc"
			tail -5 "$OUT/dequantize.log"
			exit "$rc"
		fi
		# hf_reference records the Hub revision from this metadata (the dequant copy has no .cache).
		mkdir -p "$M/.cache/huggingface/download"
		cp "$Q/.cache/huggingface/download/config.json.metadata" "$M/.cache/huggingface/download/" 2>/dev/null || true
	else
		echo "p7-ref-gpt-oss: $(date '+%F %T') fixture.lock held; $M exists, dequantize skipped"
	fi
	echo "p7-ref-gpt-oss: $(date '+%F %T') generating reference"
	env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
		timeout 6h "$UV" run --with protobuf --with sentencepiece scripts/golden/hf_reference.py \
		--model-dir "$M" --prompts tests/golden/gpt-oss-20b/prompts.jsonl --out "$OUT/reference.jsonl" \
		--model-name openai/gpt-oss-20b >"$OUT/reference.log" 2>&1 </dev/null
	rc=$?
	tail -5 "$OUT/reference.log"
	exit "$rc"
fi
{
	echo "p7-ref-gpt-oss: $(date '+%F %T') waiting for fixture.lock"
	flock "$LOCK" nice -n 19 taskset -c 12-15 env FIXTURE_JOB=p7-ref-gpt-oss bash "$0" --locked
	echo "p7-ref-gpt-oss: done rc=$?"
} >>"$LOG" 2>&1
