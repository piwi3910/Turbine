#!/usr/bin/env bash
# Our own GPTQ checkpoint (Phase 6a Task 18 re-judge, user decision 2026-09-30 "C then A", option A):
# llm-compressor GPTQ W4A16 sym g128 damp 0.01 no act-order, 512 ultrachat_200k samples x 2048 tokens,
# from /home/piwi/turbine-models/llama-3.2-3b-instruct into .../llama-3.2-3b-instruct-gptq-own
# (compressed-tensors pack-quantized -> Turbine ct_pack_int4 -> gptq_int4 row). Recipe and provenance:
# scripts/eval/gptq_calibrate.py.
#
# Runs entirely on novanas, detached (setsid nohup … </dev/null &):
#   1. setup, CPU only, no lock (nice 19, cores 12-15, GPU hidden): the calibration dataset shard at its
#      pinned revision (hf download; the token stays on the host), then the uv environment
#      (`gptq_calibrate.py --check`: torch ROCm wheel, llm-compressor, ...) under $SCR.
#   2. waits for the lead's go-file /home/piwi/turbine-ci/gpu-queue/gptq-calib.go (24 h bound), then
#      port18000.lock -> bench.gate -> bench.lock (the one-GPU-job PSU rule), CPU fixture jobs paused,
#      ONE GPU (ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0), cores 0-11, 4 h timeout.
# Log $SCR/calib.log; last line "gptq-calib: done rc=<rc>". The checkpoint's provenance is
# <out>/turbine_calibration.json. `--setup-only` stops after step 1 ("gptq-calib-setup: done rc=").
set -uo pipefail

REMOTE=/home/piwi/turbine-ci/remote/agent-p6a-gptq-numerics
SRC=$REMOTE/src
CI=/home/piwi/turbine-ci
SCR=$CI/scratch/gptq-own
MODEL=/home/piwi/turbine-models/llama-3.2-3b-instruct
OUTDIR=/home/piwi/turbine-models/llama-3.2-3b-instruct-gptq-own
DSDIR=$CI/datasets/ultrachat_200k
DSREV=8049631c405ae6576f93f445c6b8166f76f5505a
DSFILE=data/train_sft-00000-of-00003-a3ecf92756993583.parquet
GOFILE=$CI/gpu-queue/gptq-calib.go
SETUP_ONLY=0
[ "${1:-}" = --setup-only ] && SETUP_ONLY=1

export UV_CACHE_DIR=$SCR/uv-cache
export UV_PYTHON_INSTALL_DIR=$SCR/uv-python
UV=/home/piwi/.local/bin/uv
HF=/home/piwi/.local/bin/hf

mkdir -p "$SCR"
cd "$SRC" || exit 1

{
	echo "== $(date -u +%FT%TZ) gptq_calib_run.sh starting, pid $$"
	echo "== setup: dataset $DSFILE @ $DSREV"
	if [ ! -s "$DSDIR/$DSFILE" ]; then
		# hf reads the host's own login itself; nothing here reads or passes the token.
		nice -n 19 "$HF" download HuggingFaceH4/ultrachat_200k --repo-type dataset \
			--revision "$DSREV" --include "$DSFILE" --local-dir "$DSDIR"
	fi
	if [ ! -s "$DSDIR/$DSFILE" ]; then
		echo "gptq-calib: done rc=1 (dataset download)"
		exit 1
	fi
	echo "== $(date -u +%FT%TZ) setup: uv environment (CPU only, GPU hidden)"
	HF_HUB_OFFLINE=1 HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' nice -n 19 taskset -c 12-15 \
		"$UV" run scripts/eval/gptq_calibrate.py --check
	src=$?
	echo "setup rc=$src"
	if [ "$src" != 0 ]; then
		echo "gptq-calib: done rc=1 (setup)"
		exit 1
	fi
	if [ "$SETUP_ONLY" = 1 ]; then
		echo "gptq-calib-setup: done rc=0"
		exit 0
	fi

	echo "== $(date -u +%FT%TZ) waiting for go-file $GOFILE"
	waited=0
	while [ ! -e "$GOFILE" ]; do
		if [ "$waited" -ge 86400 ]; then
			echo "gptq-calib: done rc=1 (no go-file after 24 h)"
			exit 1
		fi
		sleep 60
		waited=$((waited + 60))
	done
	echo "== $(date -u +%FT%TZ) go-file present; waiting for port18000 lock"
	flock -x 200
	echo "== $(date -u +%FT%TZ) waiting for bench.gate"
	flock -x 201
	echo "== $(date -u +%FT%TZ) waiting for bench.lock"
	flock -x 202
	echo "== $(date -u +%FT%TZ) locks held, calibrating on GPU 0"
	pkill -STOP -u piwi -f 'scripts/[g]olden/' 2>/dev/null || true
	HF_HUB_OFFLINE=1 ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 timeout 14400 taskset -c 0-11 \
		"$UV" run scripts/eval/gptq_calibrate.py --model-dir "$MODEL" --dataset-file "$DSDIR/$DSFILE" \
		--out "$OUTDIR" --samples 512 --seq-len 2048 --seed 42 --damp 0.01
	rc=$?
	pkill -CONT -u piwi -f 'scripts/[g]olden/' 2>/dev/null || true
	echo "== $(date -u +%FT%TZ) calibration rc=$rc"
	rm -rf "$OUTDIR.tmp"
	if [ "$rc" = 0 ]; then
		python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["quantization_config"], indent=1))' \
			"$OUTDIR/config.json"
		du -sh "$OUTDIR"
	fi
	echo "gptq-calib: done rc=$rc"
} >>"$SCR/calib.log" 2>&1 200>"$CI/port18000.lock" 201>"$CI/bench.gate" 202>"$CI/bench.lock"
