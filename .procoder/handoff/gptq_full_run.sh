#!/usr/bin/env bash
# Full-GSM8K (1319 items) at concurrency 16 on a Turbine `gptq_int4` checkpoint (Phase 6a Task 18
# GPTQ re-judge, p6a-gptq-numerics), optionally followed by vLLM-ROCm on the same checkpoint.
# One driver for every GPTQ candidate, parameterised by label and model directory:
#
#   gptq_full_run.sh --name <label> --model-dir <dir> --served-name <id> \
#       [--expect-packaging gptq|ct_pack_int4] [--vllm] [--vllm-slug <dir under turbine-models>]
#
#   gptq-full       shuyuej/Llama-3.2-3B-Instruct-GPTQ (first run, 2026-09-29: 0.7369)
#   gptq-autoround  kaitchup/Llama-3.2-3B-Instruct-AutoRoundGPTQ-4bit (user decision C: early data point)
#   gptq-own        our llm-compressor GPTQ checkpoint, ct_pack_int4 (user decision A: decides the row), --vllm
#
# Runs entirely on novanas, detached (setsid nohup … </dev/null &), so it survives the caller's
# disconnect: kernel library build (no lock, nice 19, cores 12-15), then, once the lead's go-file
# /home/piwi/turbine-ci/gpu-queue/<label>.go exists (24 h bound), port18000.lock -> bench.gate ->
# bench.lock (the one-GPU-job PSU rule), held for the Turbine pass and the vLLM pass, CPU fixture
# jobs paused, GPU 0, cores 0-11, config scripts/lab/phase6-novanas-llama-gptq.yaml with model.path
# and model.served_name overridden. The vLLM pass (k3s Job from scripts/lab/novanas-vllm-job.yaml,
# port 18100 under port18100.lock) records a refusal (vllm.status REFUSED, vllm.log) and still ends.
#
# Output: $REMOTE/<label>/ — turbine-full.json (+ .err, server.log, status.json, turbine-paired.json),
# vllm-full.json (+ vllm.err, vllm.log, vllm.status, vllm-paired.json). Each eval is judged against
# the BF16 full run (tests/eval/llama-3.2-3b-instruct/turbine-bf16-full.json, 0.7801, c16) by
# scripts/eval/paired_compare.py (drop vs 0.04, McNemar, 95 % CI). Log $REMOTE/<label>/run.log; last
# lines "<label>: done rc=<rc>" and, with --vllm, "<label>-vllm: done rc=<rc>".
set -uo pipefail

NAME=gptq-full
MODEL_DIR=/home/piwi/turbine-models/llama-3.2-3b-instruct-gptq
SERVED=shuyuej/Llama-3.2-3B-Instruct-GPTQ
EXPECT_PACKAGING=gptq
VLLM=0
VLLM_SLUG=
while [ $# -gt 0 ]; do
	case $1 in
	--name)
		NAME=$2
		shift 2
		;;
	--model-dir)
		MODEL_DIR=$2
		shift 2
		;;
	--served-name)
		SERVED=$2
		shift 2
		;;
	--expect-packaging)
		EXPECT_PACKAGING=$2
		shift 2
		;;
	--vllm)
		VLLM=1
		shift
		;;
	--vllm-slug)
		VLLM_SLUG=$2
		shift 2
		;;
	*)
		echo "gptq_full_run.sh: unknown argument $1" >&2
		exit 2
		;;
	esac
done
[ -n "$VLLM_SLUG" ] || VLLM_SLUG=$(basename "$MODEL_DIR")

REMOTE=/home/piwi/turbine-ci/remote/agent-p6a-gptq-numerics
SRC=$REMOTE/src
KB=$REMOTE/kbuild
KLIB=$KB/libturbine_hip.so
BIN=$REMOTE/target/release
CI=/home/piwi/turbine-ci
BENCH_GATE=$CI/bench.gate
BENCH_LOCK=$CI/bench.lock
PORT_LOCK=$CI/port18000.lock
VPORT_LOCK=$CI/port18100.lock
NS=turbine-ci
OUT=$REMOTE/$NAME
URL=http://127.0.0.1:18000
VURL=http://127.0.0.1:18100
PIDF=/tmp/$NAME-server.pid
GOFILE=$CI/gpu-queue/$NAME.go
BASELINE=tests/eval/llama-3.2-3b-instruct/turbine-bf16-full.json
# An accuracy eval, not a reliability test: a step-time drift of 4x (the default
# reliability.circuit.latency_drift_open) opened the circuit 2 min into gptq-autoround's c16 eval
# (2026-09-29T21:48Z, host load 12-13 from a CPU fixture plus a uv/torch install; gptq-full ran
# the same config for 2 h with no transition) and aborted the whole run on one 503. Drift >= 2x
# still DEGRADES the circuit and is logged (counted below after the eval); only a 100x step
# time opens it. Device errors, OOM and every other circuit trigger are unchanged.
DRIFT_OPEN=100
MAX_DROP=0.04
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

judge() { # <candidate.json> <paired.json>
	python3 scripts/eval/paired_compare.py "$BASELINE" "$1" --max-drop $MAX_DROP
	python3 scripts/eval/paired_compare.py "$BASELINE" "$1" --max-drop $MAX_DROP --json >"$2"
}

run_turbine() {
	local out=$OUT/turbine-full.json
	if [ ! -s "$MODEL_DIR/config.json" ]; then
		echo "pass turbine: no checkpoint at $MODEL_DIR"
		return 1
	fi
	if ss -ltnH 'sport = :18000' | grep -q .; then
		echo "pass turbine: port 18000 already served; not starting"
		return 1
	fi
	rm -f "$PIDF"
	setsid bash -c "echo \$\$ > $PIDF; cd '$SRC' && \
    LD_LIBRARY_PATH=/opt/rocm/rocm/lib:/opt/rocm/rocm/lib/rocm_sysdeps/lib \
    TURBINE_AMD_SMI_LIBRARY=/opt/rocm/rocm/lib/libamd_smi.so.26.5.0 ROCR_VISIBLE_DEVICES=0 \
    exec taskset -c 0-11 '$BIN/turbine-server' --config scripts/lab/phase6-novanas-llama-gptq.yaml \
      --set model.path='$MODEL_DIR' --set model.served_name='$SERVED' \
      --set reliability.circuit.latency_drift_open=$DRIFT_OPEN \
      --set execution.kernel_library='$KLIB'" \
		>"$OUT/server.log" 2>&1 </dev/null &
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
		echo "pass turbine: SERVER NOT READY"
		tail -n 60 "$OUT/server.log"
		stop_server
		return 1
	fi
	curl -s $URL/turbine/v1/status >"$OUT/status.json"
	local pk wf
	pk=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["quantization"]["packaging"])' "$OUT/status.json" 2>/dev/null)
	wf=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["support"]["weight_format"])' "$OUT/status.json" 2>/dev/null)
	echo "pass turbine: loaded packaging=$pk weight_format=$wf (expected $EXPECT_PACKAGING / gptq_int4)"
	if [ "$pk" != "$EXPECT_PACKAGING" ] || [ "$wf" != gptq_int4 ]; then
		echo "pass turbine: unexpected packaging or support row; not evaluating"
		stop_server
		return 1
	fi
	echo "== $(date -u +%FT%TZ) pass turbine: server ready, full-GSM8K eval at c16 (6 h timeout)"
	rm -f "$out" "${out%.json}.err"
	timeout 21600 taskset -c 0-11 "$BIN/turbine-golden" eval --url $URL \
		--tasks tests/eval/gsm8k-full.jsonl --concurrency 16 --output json \
		>"$out" 2>"${out%.json}.err"
	local rc=$?
	echo "== $(date -u +%FT%TZ) pass turbine: eval rc=$rc"
	curl -s $URL/turbine/v1/status >"$OUT/status-end.json"
	stop_server
	echo "pass turbine: circuit transitions during the run: $(grep -c 'circuit_transition' "$OUT/server.log")"
	sed 's/\x1b\[[0-9;]*m//g' "$OUT/server.log" | grep 'circuit_transition' | head -n 20
	[ "$rc" = 0 ] && judge "$out" "$OUT/turbine-paired.json"
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
	local out=$OUT/vllm-full.json
	if ss -ltnH 'sport = :18100' | grep -q .; then
		echo "pass vllm: port 18100 already served; not starting"
		return 1
	fi
	local run job
	run=$NAME-$(date -u +%m%d%H%M%S)
	job=turbine-lab-vllm-$run
	echo "pass vllm: job $job (slug $VLLM_SLUG)"
	sed -e "s/__RUN_ID__/$run/g" -e "s/__SLUG__/$VLLM_SLUG/g" -e "s|__SERVED_NAME__|$SERVED|g" \
		-e "s/__MAX_MODEL_LEN__/32768/g" -e "s/__GPUS__/1/g" -e '/# __EXTRA_ARGS__$/d' \
		scripts/lab/novanas-vllm-job.yaml >"$OUT/vllm-job.yaml"
	if ! kube apply -f "$OUT/vllm-job.yaml"; then
		echo "pass vllm: kubectl apply failed"
		return 1
	fi
	local ready=0 code failed t=0 rc=0
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
	if [ "$ready" != 1 ]; then
		kube -n $NS logs "job/$job" >"$OUT/vllm.log" 2>&1
		kube -n $NS describe pods -l "job-name=$job" >"$OUT/vllm-pod.txt" 2>&1
		echo "pass vllm: REFUSED or not ready after ${t}s (failed=$failed); log $OUT/vllm.log"
		tail -n 30 "$OUT/vllm.log"
		echo REFUSED >"$OUT/vllm.status"
		rc=1
	else
		echo "== $(date -u +%FT%TZ) pass vllm: ready after ${t}s, full-GSM8K eval at c16 (6 h timeout)"
		rm -f "$out"
		timeout 21600 taskset -c 0-11 "$BIN/turbine-golden" eval --url $VURL --concurrency 16 \
			--tasks tests/eval/gsm8k-full.jsonl --output json >"$out" 2>"$OUT/vllm.err"
		rc=$?
		echo "== $(date -u +%FT%TZ) pass vllm: eval rc=$rc"
		kube -n $NS logs "job/$job" >"$OUT/vllm.log" 2>&1
		echo SERVED >"$OUT/vllm.status"
	fi
	kube -n $NS delete job "$job" --wait=true >/dev/null 2>&1
	vllm_job_gone "$job" || echo "pass vllm: pod of $job still present after 300 s"
	[ "$rc" = 0 ] && judge "$out" "$OUT/vllm-paired.json"
	return "$rc"
}

{
	echo "== $(date -u +%FT%TZ) gptq_full_run.sh $NAME starting, pid $$ (model $MODEL_DIR, served $SERVED, vllm=$VLLM)"
	echo "== kernel library build (serialized on $KB.lock: drivers share the build dir)"
	(
		flock -x 9
		nice -n 19 taskset -c 12-15 cmake -S kernels/rocm -B "$KB" -G Ninja -DCMAKE_BUILD_TYPE=Release \
			-DCMAKE_HIP_COMPILER=/opt/rocm/rocm/bin/hipcc -DGPU_TARGETS=gfx1201 >/dev/null &&
			nice -n 19 taskset -c 12-15 cmake --build "$KB" -j 4 >"$OUT/kbuild.log" 2>&1
	) 9>"$KB.lock"
	krc=$?
	echo "kernel build rc=$krc"
	if [ "$krc" != 0 ] || [ ! -x "$BIN/turbine-server" ] || [ ! -x "$BIN/turbine-golden" ]; then
		echo "$NAME: done rc=1 (build)"
		[ "$VLLM" = 1 ] && echo "$NAME-vllm: done rc=1 (build)"
		exit 1
	fi
	# GPU queue order (lead-owned go-files): wait for ours before touching any lock, bounded 24 h.
	echo "== $(date -u +%FT%TZ) waiting for go-file $GOFILE"
	waited=0
	while [ ! -e "$GOFILE" ]; do
		if [ "$waited" -ge 86400 ]; then
			echo "$NAME: done rc=1 (no go-file after 24 h)"
			[ "$VLLM" = 1 ] && echo "$NAME-vllm: done rc=1 (no go-file after 24 h)"
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
	echo "== $(date -u +%FT%TZ) locks held"
	pkill -STOP -u piwi -f 'scripts/[g]olden/' 2>/dev/null || true
	run_turbine
	rc=$?
	echo "$NAME: done rc=$rc"
	if [ "$VLLM" = 1 ]; then
		echo "== $(date -u +%FT%TZ) waiting for port18100 lock"
		flock -x 203
		run_vllm
		vrc=$?
		echo "$NAME-vllm: done rc=$vrc"
	fi
	pkill -CONT -u piwi -f 'scripts/[g]olden/' 2>/dev/null || true
} >>"$OUT/run.log" 2>&1 200>"$PORT_LOCK" 201>"$BENCH_GATE" 202>"$BENCH_LOCK" 203>"$VPORT_LOCK"
