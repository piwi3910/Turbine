#!/usr/bin/env bash
# Phase 7 Task 20: transformers self-spread of the Mistral-7B reference (detached: setsid nohup bash p7_spread_mistral.sh).
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
UV=/home/piwi/.local/bin/uv
SRC=/home/piwi/turbine-ci/scratch/p7-t20/src
M=/home/piwi/turbine-models/mistral-7b-instruct-v0.3
OUT=/home/piwi/turbine-ci/scratch/p7-t20/out
LOG=$OUT/p7-spread-mistral.log
if [ "${1:-}" = --locked ]; then
	cd "$SRC/scripts/golden" || exit 1
	echo "p7-spread-mistral: $(date "+%F %T") fixture.lock held"
	env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
		"$UV" run self_spread.py "$M" "$OUT/reference.jsonl" "$OUT/self_spread.json" >"$OUT/spread.log" 2>&1 </dev/null
	exit $?
fi
{
	echo "p7-spread-mistral: $(date "+%F %T") waiting for fixture.lock"
	flock /home/piwi/turbine-ci/fixture.lock nice -n 19 taskset -c 12-15 env FIXTURE_JOB=p7-spread-mistral bash "$0" --locked
	echo "p7-spread-mistral: done rc=$?"
} >>"$LOG" 2>&1
