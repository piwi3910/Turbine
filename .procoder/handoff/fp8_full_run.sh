#!/usr/bin/env bash
# Phase 6a Task 14 FP8-dynamic full-GSM8K accuracy driver: Turbine vs vLLM-ROCm serving the same
# FP8-dynamic checkpoint (RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic) on gfx1201, both at
# concurrency 16, judged over the full 1,319-item tests/eval/gsm8k-full.jsonl (user decision
# 2026-09-29: "FP8-dynamic (Task 14): accuracy on full GSM8K against vLLM, c1 ITL as a perf item" —
# this driver covers accuracy only; c1 ITL is a separate, later perf item).
#
# Runs entirely on novanas, detached (setsid nohup from the caller), so it survives the caller's
# disconnect. Builds the kernel library itself (nice 19, cores 12-15, no lock — CPU work), then
# takes port18000.lock -> bench.gate -> bench.lock (the one-GPU-job PSU rule) ONCE and holds all
# three for the whole script (both legs), pausing CPU fixture jobs (SIGSTOP on 'scripts/golden/')
# for the duration:
#   turbine : serves FP8-dynamic natively (phase6-novanas-llama-fp8.yaml, GPU 0, cores 0-11),
#             `turbine-golden eval --concurrency 16` over gsm8k-full.jsonl, server always killed
#   vllm    : vLLM-ROCm on the same checkpoint (scripts/lab/novanas-vllm-job.yaml as k3s Job
#             turbine-lab-vllm-fp8full-<ts>, port 18100), same eval; a refusal is recorded
#             (vllm.log, status REFUSED) and the run still finishes so the gate has both numbers
#             it can get; the Job is always deleted
# Then: `turbine-golden eval-compare --baseline vllm.json --candidate turbine.json --max-drop 0.01`
# (the max_drop of tests/eval/llama-3.2-3b-instruct-fp8-dynamic/gate.json), output appended to this
# log. Last line: "fp8-full: done rc=<rc> turbine=<rc> vllm=<rc>".
set -uo pipefail

REMOTE=/home/piwi/turbine-ci/remote/agent-p6a-fp8-full
SRC=$REMOTE/src
KB=$REMOTE/kbuild
KLIB=$KB/libturbine_hip.so
BIN=$REMOTE/target/release
CI=/home/piwi/turbine-ci
BENCH_GATE=$CI/bench.gate
BENCH_LOCK=$CI/bench.lock
PORT_LOCK=$CI/port18000.lock
NS=turbine-ci
OUT=$CI/scratch/p6a-fp8-full/run
URL=http://127.0.0.1:18000
VURL=http://127.0.0.1:18100
PIDF=/tmp/fp8full-server.pid
GATE_MAX_DROP=0.01
export KUBECTL_KUBERC=false

mkdir -p "$OUT"
cd "$SRC" || exit 1

kube() { kubectl "$@" 2> >(grep -v 'permission denied' >&2); }

stop_server() {
	if [ -f "$PIDF" ]; then
		local p
		p=$(cat "$PIDF")
		if [ "$(cat /proc/"$p"/comm 2>/dev/null)" = turbine-server ]; then
			kill "$p" 2>/dev/null || true
			for _ in 1 2 3 4 5 6 7 8 9 10; do
				[ -d "/proc/$p" ] || break
				sleep 1
			done
			kill -9 "$p" 2>/dev/null || true
		fi
		rm -f "$PIDF"
	fi
}

resume_fixtures() { pkill -CONT -u piwi -f 'scripts/[g]olden/' 2>/dev/null || true; }

run_turbine() {
	local name=turbine o=$OUT
	if [ -s "$o/turbine.done" ]; then
		echo "pass $name: already done ($(cat "$o/turbine.done")); skipping"
		return 0
	fi
	if ss -ltnH 'sport = :18000' | grep -q .; then
		echo "pass $name: port 18000 already served; skipping"
		return 1
	fi
	rm -f "$PIDF"
	setsid bash -c "echo \$\$ > $PIDF; cd '$SRC' && \
    LD_LIBRARY_PATH=/opt/rocm/rocm/lib:/opt/rocm/rocm/lib/rocm_sysdeps/lib \
    TURBINE_AMD_SMI_LIBRARY=/opt/rocm/rocm/lib/libamd_smi.so.26.5.0 ROCR_VISIBLE_DEVICES=0 \
    exec taskset -c 0-11 '$BIN/turbine-server' --config scripts/lab/phase6-novanas-llama-fp8.yaml \
      --set model.path=/home/piwi/turbine-models/llama-3.2-3b-instruct-fp8-dynamic \
      --set execution.kernel_library='$KLIB'" \
		>"$o/turbine-server.log" 2>&1 </dev/null &
	local ready=0 code
	for _ in $(seq 1 150); do
		code=$(curl -s -o /dev/null -w '%{http_code}' $URL/ready)
		if [ "$code" = 200 ]; then
			ready=1
			break
		fi
		sleep 2
	done
	if [ "$ready" != 1 ]; then
		echo "pass $name: SERVER NOT READY"
		tail -n 60 "$o/turbine-server.log"
		stop_server
		return 1
	fi
	echo "== $(date -u +%FT%TZ) pass $name: server ready, running full-GSM8K eval (c16, ~6h timeout)"
	timeout 21600 taskset -c 0-11 "$BIN/turbine-golden" eval --url $URL --concurrency 16 \
		--tasks tests/eval/gsm8k-full.jsonl --output json >"$o/turbine.json" 2>"$o/turbine.err"
	local rc=$?
	echo "== $(date -u +%FT%TZ) pass $name: eval rc=$rc"
	stop_server
	[ "$rc" = 0 ] && date -u +%FT%TZ >"$o/turbine.done"
	return "$rc"
}

vllm_job_gone() {
	local j=$1
	for _ in $(seq 1 60); do
		[ -z "$(kube -n $NS get pods -l "job-name=$j" -o name 2>/dev/null)" ] && return 0
		sleep 5
	done
	return 1
}

run_vllm() {
	local name=vllm o=$OUT slug=llama-3.2-3b-instruct-fp8-dynamic
	if [ -s "$o/vllm.done" ]; then
		echo "pass $name: already done ($(cat "$o/vllm.done")); skipping"
		return 0
	fi
	if ss -ltnH 'sport = :18100' | grep -q .; then
		echo "pass $name: port 18100 already served; skipping"
		return 1
	fi
	local run job
	run=fp8full-$(date -u +%m%d%H%M%S)
	job=turbine-lab-vllm-$run
	echo "pass $name: job $job"
	sed -e "s/__RUN_ID__/$run/g" -e "s/__SLUG__/$slug/g" \
		-e "s|__SERVED_NAME__|RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic|g" \
		-e "s/__MAX_MODEL_LEN__/32768/g" -e "s/__GPUS__/1/g" -e '/# __EXTRA_ARGS__$/d' \
		scripts/lab/novanas-vllm-job.yaml >"$o/vllm-job.yaml"
	if ! kube apply -f "$o/vllm-job.yaml"; then
		echo "pass $name: kubectl apply failed"
		return 1
	fi
	local ready=0 code failed t=0
	while [ $t -lt 2400 ]; do
		code=$(curl -s -o /dev/null -w '%{http_code}' $VURL/v1/models)
		if [ "$code" = 200 ]; then
			ready=1
			break
		fi
		failed=$(kube -n $NS get job "$job" -o jsonpath='{.status.failed}' 2>/dev/null)
		[ -n "$failed" ] && [ "$failed" != 0 ] && break
		sleep 10
		t=$((t + 10))
	done
	local rc=0
	if [ "$ready" != 1 ]; then
		kube -n $NS logs "job/$job" >"$o/vllm.log" 2>&1
		kube -n $NS describe pods -l "job-name=$job" >"$o/vllm-pod.txt" 2>&1
		echo "pass $name: REFUSED or not ready after ${t}s (failed=$failed); log $o/vllm.log"
		tail -n 30 "$o/vllm.log"
		echo REFUSED >"$o/vllm.status"
		rc=1
	else
		echo "== $(date -u +%FT%TZ) pass $name: vLLM ready after ${t}s, running full-GSM8K eval (c16, ~6h timeout)"
		timeout 21600 taskset -c 0-11 "$BIN/turbine-golden" eval --url $VURL --concurrency 16 \
			--tasks tests/eval/gsm8k-full.jsonl --output json >"$o/vllm.json" 2>"$o/vllm.err"
		rc=$?
		echo "== $(date -u +%FT%TZ) pass $name: eval rc=$rc"
		kube -n $NS logs "job/$job" >"$o/vllm.log" 2>&1
		echo SERVED >"$o/vllm.status"
	fi
	kube -n $NS delete job "$job" --wait=true >/dev/null 2>&1
	vllm_job_gone "$job" || echo "pass $name: pod of $job still present after 300 s"
	[ "$rc" = 0 ] && date -u +%FT%TZ >"$o/vllm.done"
	return "$rc"
}

{
	echo "== $(date -u +%FT%TZ) fp8_full_run.sh starting, pid $$"
	echo "== $(date -u +%FT%TZ) kernel library build (nice 19, cores 12-15, no lock)"
	nice -n 19 taskset -c 12-15 cmake -S kernels/rocm -B "$KB" -G Ninja -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_HIP_COMPILER=/opt/rocm/rocm/bin/hipcc -DGPU_TARGETS=gfx1201 >/dev/null &&
		nice -n 19 taskset -c 12-15 cmake --build "$KB" -j 4 >"$OUT/kbuild.log" 2>&1
	krc=$?
	echo "kernel build rc=$krc"
	if [ "$krc" != 0 ] || [ ! -x "$BIN/turbine-server" ] || [ ! -x "$BIN/turbine-golden" ]; then
		echo "fp8-full: done rc=1 turbine=- vllm=- (build)"
		exit 1
	fi

	echo "== $(date -u +%FT%TZ) waiting for port18000 lock"
	flock -x 200
	echo "== $(date -u +%FT%TZ) waiting for bench.gate"
	flock -x 201
	echo "== $(date -u +%FT%TZ) waiting for bench.lock"
	flock -x 202
	echo "== $(date -u +%FT%TZ) locks held, starting (held for the whole script)"
	pkill -STOP -u piwi -f 'scripts/[g]olden/' 2>/dev/null || true

	run_turbine
	rc_t=$?
	run_vllm
	rc_v=$?

	resume_fixtures

	crc=2
	if [ -s "$OUT/turbine.json" ] && [ -s "$OUT/vllm.json" ]; then
		echo "== $(date -u +%FT%TZ) eval-compare (max-drop $GATE_MAX_DROP)"
		"$BIN/turbine-golden" eval-compare --baseline "$OUT/vllm.json" --candidate "$OUT/turbine.json" \
			--max-drop "$GATE_MAX_DROP"
		crc=$?
		echo "eval-compare rc=$crc"
	else
		echo "== $(date -u +%FT%TZ) eval-compare skipped: turbine.json or vllm.json missing/empty"
	fi

	rc=$((rc_t | rc_v | crc))
	echo "fp8-full: done rc=$rc turbine=$rc_t vllm=$rc_v"
} 200>"$PORT_LOCK" 201>"$BENCH_GATE" 202>"$BENCH_LOCK"
# Everything above is on stdout/stderr; the caller (setsid nohup bash fp8_full_run.sh
# >run.log 2>&1 </dev/null &) is what actually creates the log file.
