#!/usr/bin/env bash
# Our own GPTQ checkpoint, retry after a SECOND ROCm Triton fix (Phase 6a Task 18 re-judge, user
# decision 2026-09-30 "C then A", option A; rotation 12 fix for the gptq-calib2 rc=1 failure below).
# llm-compressor GPTQ W4A16 sym g128 damp 0.01 no act-order, 512 ultrachat_200k samples x 2048 tokens,
# from /home/piwi/turbine-models/llama-3.2-3b-instruct into .../llama-3.2-3b-instruct-gptq-own
# (compressed-tensors pack-quantized -> Turbine ct_pack_int4 -> gptq_int4 row). Recipe and provenance:
# scripts/eval/gptq_calibrate.py.
#
# gptq-calib.go (calib.log, rotation 11) hit "Implicit conversion of CUDA __nv_fdiv_rn device
# function has been dropped" in llm-compressor's fused Triton GPTQ block-update kernel; fixed by
# LLMCOMPRESSOR_DISABLE_GPTQ_TRITON=1 (gptq_calibrate.py's --gptq-backend auto, already the default
# used by gptq-calib2). gptq-calib2.go (calib2.log, rotation 11) ran through every layer's GPTQ block
# update (METRIC lines present) and then hit a SECOND, independent ROCm Triton kernel at save time:
# "Implicit conversion of CUDA __nv_rintf device function has been dropped" from
# compressed_tensors.quantization.lifecycle.forward_helpers._quantize_triton, reached from
# model.save_pretrained(save_compressed=True) -> ModelCompressor.compress_model -> compress_module ->
# PackedQuantizationCompressor.compress -> quantize (its own ImplBackend dispatch, priority 0 ahead
# of the torch fallback, gated by `triton_req()`, not by an env var). Fix (rotation 12, no supported
# switch exists for this one, so gptq_calibrate.py sets the module flag directly): when
# --gptq-backend resolves to torch (the ROCm default), it now also sets
# compressed_tensors.utils.triton.HAS_TRITON = False before the model is touched, which every
# `triton_req()`-gated callsite (this one, llm-compressor's MSE observer, nvfp4's cast_to_fp4) reads
# fresh on each call — one patch instead of chasing a third Triton kernel next time. Recorded in
# turbine_calibration.json / --check as "ct_triton": "disabled"|"enabled", next to "gptq_backend".
# Verified with a CPU-only smoke of PackedQuantizationCompressor.compress() on a tiny synthetic
# weight (ct_pack_smoke.py, scratchpad, not committed): confirmed `_quantize_triton_req(...)` is
# False and compress() returns a finite, correctly-shaped int32-packed weight. compress_model's
# remaining Triton call sites in this environment (compressed-tensors 0.19.0, llmcompressor 0.14.0)
# were grepped and are all `triton_req()`-gated, so this one patch covers the whole save path.
#
# Runs entirely on novanas, detached (setsid nohup … </dev/null &):
#   1. setup, CPU only, no lock (nice 19, cores 12-15, GPU hidden): dataset shard and uv environment
#      are already in place from the first attempt ($SCR); re-checked, not re-downloaded.
#   2. waits for the lead's go-file /home/piwi/turbine-ci/gpu-queue/gptq-calib3.go (24 h bound), then
#      port18000.lock -> bench.gate -> bench.lock (the one-GPU-job PSU rule), CPU fixture jobs paused,
#      ONE GPU (ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0), cores 0-11, 4 h timeout.
# Log $SCR/calib3.log; last line "gptq-calib3: done rc=<rc>". The checkpoint's provenance is
# <out>/turbine_calibration.json. `--setup-only` stops after step 1 ("gptq-calib3-setup: done rc=").
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
GOFILE=$CI/gpu-queue/gptq-calib3.go
SETUP_ONLY=0
[ "${1:-}" = --setup-only ] && SETUP_ONLY=1

export UV_CACHE_DIR=$SCR/uv-cache
export UV_PYTHON_INSTALL_DIR=$SCR/uv-python
UV=/home/piwi/.local/bin/uv
HF=/home/piwi/.local/bin/hf

mkdir -p "$SCR"
cd "$SRC" || exit 1

{
	echo "== $(date -u +%FT%TZ) gptq_calib_run3.sh starting, pid $$"
	echo "== setup: dataset $DSFILE @ $DSREV"
	if [ ! -s "$DSDIR/$DSFILE" ]; then
		# hf reads the host's own login itself; nothing here reads or passes the token.
		nice -n 19 "$HF" download HuggingFaceH4/ultrachat_200k --repo-type dataset \
			--revision "$DSREV" --include "$DSFILE" --local-dir "$DSDIR"
	fi
	if [ ! -s "$DSDIR/$DSFILE" ]; then
		echo "gptq-calib3: done rc=1 (dataset download)"
		exit 1
	fi
	echo "== $(date -u +%FT%TZ) setup: uv environment (CPU only, GPU hidden), gptq-backend check"
	HF_HUB_OFFLINE=1 HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' nice -n 19 taskset -c 12-15 \
		"$UV" run scripts/eval/gptq_calibrate.py --check --gptq-backend auto
	src=$?
	echo "setup rc=$src"
	if [ "$src" != 0 ]; then
		echo "gptq-calib3: done rc=1 (setup)"
		exit 1
	fi
	if [ "$SETUP_ONLY" = 1 ]; then
		echo "gptq-calib3-setup: done rc=0"
		exit 0
	fi

	echo "== $(date -u +%FT%TZ) waiting for go-file $GOFILE"
	waited=0
	while [ ! -e "$GOFILE" ]; do
		if [ "$waited" -ge 86400 ]; then
			echo "gptq-calib3: done rc=1 (no go-file after 24 h)"
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
	echo "== $(date -u +%FT%TZ) locks held, calibrating on GPU 0 (gptq-backend auto -> torch_eager + ct_triton disabled on ROCm)"
	pkill -STOP -u piwi -f 'scripts/[g]olden/' 2>/dev/null || true
	HF_HUB_OFFLINE=1 ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 timeout 14400 taskset -c 0-11 \
		"$UV" run scripts/eval/gptq_calibrate.py --model-dir "$MODEL" --dataset-file "$DSDIR/$DSFILE" \
		--out "$OUTDIR" --samples 512 --seq-len 2048 --seed 42 --damp 0.01 --gptq-backend auto
	rc=$?
	pkill -CONT -u piwi -f 'scripts/[g]olden/' 2>/dev/null || true
	echo "== $(date -u +%FT%TZ) calibration rc=$rc"
	rm -rf "$OUTDIR.tmp"
	if [ "$rc" = 0 ]; then
		python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["quantization_config"], indent=1))' \
			"$OUTDIR/config.json"
		python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("gptq_backend:", d["gptq_backend"], "ct_triton:", d.get("ct_triton"))' \
			"$OUTDIR/turbine_calibration.json"
		du -sh "$OUTDIR"
	fi
	echo "gptq-calib3: done rc=$rc"
} >>"$SCR/calib3.log" 2>&1 200>"$CI/port18000.lock" 201>"$CI/bench.gate" 202>"$CI/bench.lock"
