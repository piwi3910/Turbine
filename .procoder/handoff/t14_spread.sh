#!/usr/bin/env bash
# Phase 6a Task 14: the FP8 per-tensor golden reference and its self-spread (CPU only). Runs ON
# novanas, detached (setsid nohup bash t14_spread.sh), under fixture.lock for its whole length, nice 19 on cores 12-15 with 4 threads
# like fixtures-r5.sh (never bench.lock; the GPU drivers SIGSTOP it through 'scripts/golden/').
#
# 1. Regenerates the reference with the current quant_reference.py (fused-max static input scales,
#    4729281): the committed reference was captured 2026-09-28 22:22Z, before that change, and its
#    engine string cannot tell per-part from fused-max. The new one goes to $OUT/reference.jsonl;
#    reference-diff.txt says whether it differs from the old one. When it differs, the fixture dir's
#    reference.jsonl is replaced (the old one kept as reference.pre-t14.jsonl).
# 2. self_spread.py --act-quant fp8_tensor over all eight variants on the dequantized copy the
#    reference left in /dev/shm, against the fixture dir's reference.jsonl, written to
#    $F/spread.json (the path fixtures-r5.sh step 4 checks, so it then skips it).
# Last log line: "t14-spread: done rc=<rc>".
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
UV=/home/piwi/.local/bin/uv
SRC=/home/piwi/turbine-ci/remote/agent-p6a-fp8-t14/src
F=/home/piwi/turbine-ci/remote/agent-adb021480bf19bfbe/fixtures/llama-3.2-3b-instruct-fp8
M=/home/piwi/turbine-models/llama-3.2-3b-instruct-fp8
OUT=/home/piwi/turbine-ci/scratch/p6a-fp8-t14/spread
ALL="bf16-sdpa-incremental,bf16-sdpa-full,bf16-eager-incremental,bf16-eager-full,fp32-sdpa-incremental,fp32-sdpa-full,fp32-eager-incremental,fp32-eager-full"
LOCK=/home/piwi/turbine-ci/fixture.lock
mkdir -p "$OUT"
LOG=$OUT/t14-spread.log

run() { # <timeout> <log> <uv run args...>
	local t=$1 log=$2
	shift 2
	env OMP_NUM_THREADS=4 \
		MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
		timeout "$t" "$UV" run "$@" >"$log" 2>&1 </dev/null
}

main() {
	cd "$SRC" || return 1
	echo "t14-spread: $(date '+%F %T') fixture.lock held"
	if python3 -c "import json,sys; d=json.load(open('$F/spread.json')); sys.exit(0 if all(v in d for v in '$ALL'.split(',')) else 1)" 2>/dev/null; then
		echo "t14-spread: $F/spread.json already has every variant"
		return 0
	fi
	local avail
	avail=$(df --output=avail -B1G /dev/shm | tail -1 | tr -d ' ')
	echo "t14-spread: /dev/shm ${avail} GiB free"
	if [ "$avail" -lt 12 ]; then
		echo "t14-spread: less than 12 GiB free in /dev/shm; not started"
		return 3
	fi
	WORK=$(mktemp -d /dev/shm/turbine-t14-fp8-XXXXXX)
	trap 'rm -rf "$WORK"' EXIT
	echo "t14-spread: $(date '+%F %T') reference (work $WORK)"
	run 4h "$OUT/reference.log" scripts/golden/quant_reference.py --model-dir "$M" \
		--prompts tests/golden/prompts.jsonl --out "$OUT/reference.jsonl" --act-quant fp8_tensor \
		--work-dir "$WORK" --keep-dequantized --model-name RedHatAI/Llama-3.2-3B-Instruct-FP8
	local rc=$?
	echo "t14-spread: reference rc=$rc"
	tail -3 "$OUT/reference.log"
	[ "$rc" = 0 ] || return "$rc"
	local bf16
	bf16=$(find "$WORK" -mindepth 1 -maxdepth 3 -name bf16 -type d | head -1)
	[ -n "$bf16" ] || {
		echo "t14-spread: no dequantized copy under $WORK"
		return 1
	}
	python3 - "$F/reference.jsonl" "$OUT/reference.jsonl" >"$OUT/reference-diff.txt" <<'EOF'
import json, sys
a = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
b = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
same_tokens = all(x["tokens"] == y["tokens"] for x, y in zip(a, b)) and len(a) == len(b)
worst = 0.0
for x, y in zip(a, b):
    for rx, ry in zip(x["top_logprobs"], y["top_logprobs"]):
        dx = {int(t): v for t, v in rx}
        for t, v in ry:
            if int(t) in dx:
                worst = max(worst, abs(dx[int(t)] - v))
print(f"same_tokens={same_tokens} max_abs_logprob_diff={worst:.6f}")
print("DIFFERENT" if not same_tokens or worst > 1e-4 else "SAME")
EOF
	cat "$OUT/reference-diff.txt"
	if grep -q DIFFERENT "$OUT/reference-diff.txt"; then
		[ -e "$F/reference.pre-t14.jsonl" ] || cp -p "$F/reference.jsonl" "$F/reference.pre-t14.jsonl"
		cp "$OUT/reference.jsonl" "$F/reference.jsonl"
		echo "t14-spread: fixture reference replaced (old kept as reference.pre-t14.jsonl)"
	fi
	echo "t14-spread: $(date '+%F %T') self_spread on $bf16"
	run 20h "$OUT/spread.log" scripts/golden/self_spread.py "$bf16" "$F/reference.jsonl" \
		"$OUT/spread.json" "$ALL" --act-quant fp8_tensor
	rc=$?
	echo "t14-spread: self_spread rc=$rc"
	tail -10 "$OUT/spread.log"
	if [ "$rc" = 0 ] && [ -s "$OUT/spread.json" ]; then
		cp "$OUT/spread.json" "$F/spread.json.t14tmp" && mv "$F/spread.json.t14tmp" "$F/spread.json"
		echo "t14-spread: wrote $F/spread.json"
	fi
	return "$rc"
}

# The whole script runs as one fixture job: the waiter's command line names fixture.lock and
# FIXTURE_JOB=t14:fp8-tensor-spread, so scripts/lab/fixture-order.sh can rank it (fixture.queue).
if [ "${1:-}" = --locked ]; then
	main
	exit $?
fi
{
	echo "t14-spread: $(date '+%F %T') waiting for fixture.lock"
	flock "$LOCK" nice -n 19 taskset -c 12-15 env FIXTURE_JOB=t14:fp8-tensor-spread bash "$0" --locked
	echo "t14-spread: done rc=$?"
} >>"$LOG" 2>&1
