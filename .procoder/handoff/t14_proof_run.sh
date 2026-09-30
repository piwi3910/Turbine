#!/usr/bin/env bash
# Phase 6a Task 14 FP8 proof driver. Runs entirely on novanas, detached (setsid nohup), so it
# survives the caller's disconnect. Builds the kernel library (no lock, nice 19, cores 12-15),
# then four passes one after another, each under port lock -> bench.gate -> bench.lock (the
# one-GPU-job PSU rule) with CPU fixture jobs paused (SIGSTOP on 'scripts/golden/'), GPU 0 for
# Turbine (ROCR_VISIBLE_DEVICES=0), server and clients on cores 0-11, its own server always killed:
#   bf16       : bench c16 (200 req) and c1 (the same-run baseline), GSM8K-200 eval at c16
#   fp8        : FP8-dynamic (phase6-novanas-llama-fp8.yaml): golden c1 + c16, bench c16 + c1,
#                GSM8K-200 eval at c16
#   fp8-tensor : FP8 per-tensor (phase6-novanas-llama-fp8-tensor.yaml): golden c1 + c16 (against the
#                tolerance.json in the tree when the pass runs), a c1 capture for re-judging offline
#                once the per-tensor spread lands, bench c16 + c1
#   vllm       : vLLM-ROCm on the FP8-dynamic checkpoint (scripts/lab/novanas-vllm-job.yaml as k3s Job
#                turbine-lab-vllm-t14-<ts>, port 18100 under port18100.lock): bench c16 + c1, GSM8K-200
#                eval at c16; a refusal is recorded (vllm.log, status REFUSED) and the pass ends
# The measurement commands are lab-bench.sh's (client on novanas against the loopback URL).
# Results: $OUT/<pass>/{golden1.txt,golden16.txt,capture.jsonl,bench.json,bench-c1.json,quality.json,
# metrics.txt,status.json,server.log|vllm.log}; one "T14 <pass> ..." summary line per pass.
# Last log line: "t14_proof_run: done rc=<rc>".
set -uo pipefail

REMOTE=/home/piwi/turbine-ci/remote/agent-p6a-fp8-t14
SRC=$REMOTE/src
KB=$REMOTE/kbuild
KLIB=$KB/libturbine_hip.so
BIN=$REMOTE/target/release
CI=/home/piwi/turbine-ci
BENCH_GATE=$CI/bench.gate
BENCH_LOCK=$CI/bench.lock
PORT_LOCK=$CI/port18000.lock
PORT_LOCK_VLLM=$CI/port18100.lock
OUT=$CI/scratch/p6a-fp8-t14/run
URL=http://127.0.0.1:18000
VURL=http://127.0.0.1:18100
PIDF=/tmp/t14fp8-server.pid
NS=turbine-ci
export KUBECTL_KUBERC=false

mkdir -p "$OUT"
cd "$SRC" || exit 1

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

cleanup_pass() {
	stop_server
	resume_fixtures
}

# summary <pass> : one line with the numbers a collector needs
summary() {
	python3 - "$1" "$OUT/$1" <<'EOF'
import json, os, sys
name, o = sys.argv[1], sys.argv[2]
def load(p):
    try:
        return json.load(open(os.path.join(o, p)))
    except Exception:
        return {}
def last(p):
    try:
        lines = [l for l in open(os.path.join(o, p)).read().splitlines() if l.strip()]
        return lines[-1].split()[0] if lines else "-"
    except Exception:
        return "-"
b, c1, q = load("bench.json"), load("bench-c1.json"), load("quality.json")
p50 = lambda d, k: (d.get(k) or {}).get("p50")
print(f"T14 {name} tok/s={b.get('output_token_throughput')} ok={b.get('requests_ok')} "
      f"ttft_p50={p50(b, 'ttft_ms')} itl_p50={p50(b, 'itl_ms')} c1_itl_p50={p50(c1, 'itl_ms')} "
      f"c1_tok/s={c1.get('output_token_throughput')} golden1={last('golden1.txt')} "
      f"golden16={last('golden16.txt')} gsm8k={q.get('accuracy')} gsm8k_c={q.get('concurrency')}")
EOF
}

# measure <o> <url> <quality 0|1> : bench c16, bench c1, optional GSM8K-200 at c16
measure() {
	local o=$1 url=$2 quality=$3 rc=0
	timeout 1800 taskset -c 0-11 "$BIN/turbine-bench" --url "$url" --concurrency 16 --requests 200 \
		--prompt-words 512 --max-tokens 256 --ignore-eos --output json >"$o/bench.json" 2>"$o/bench.err"
	echo "bench c16 rc=$?"
	timeout 900 taskset -c 0-11 "$BIN/turbine-bench" --url "$url" --concurrency 1 --requests 10 \
		--max-tokens 128 --ignore-eos --output json >"$o/bench-c1.json" 2>"$o/bench-c1.err"
	echo "bench c1 rc=$?"
	if [ "$quality" = 1 ]; then
		timeout 7200 taskset -c 0-11 "$BIN/turbine-golden" eval --url "$url" --concurrency 16 \
			--tasks tests/eval/gsm8k-200.jsonl --output json >"$o/quality.json" 2>"$o/quality.err"
		rc=$?
		echo "gsm8k-200 c16 rc=$rc"
	fi
	return "$rc"
}

take_locks() {
	echo "== $(date -u +%FT%TZ) pass $1: waiting for port lock"
	flock -x 200
	echo "== $(date -u +%FT%TZ) pass $1: waiting for bench.gate"
	flock -x 201
	echo "== $(date -u +%FT%TZ) pass $1: waiting for bench.lock"
	flock -x 202
	echo "== $(date -u +%FT%TZ) pass $1: locks held, starting"
}

# run_pass <name> <config> <weights slug> <golden slug|-> <quality 0|1>
run_pass() {
	local name=$1 cfg=$2 slug=$3 gslug=$4 quality=$5
	local o=$OUT/$name
	if [ -s "$o/done" ]; then
		echo "pass $name: already done ($(cat "$o/done")); skipping"
		return 0
	fi
	mkdir -p "$o"
	take_locks "$name"
	if ss -ltnH 'sport = :18000' | grep -q .; then
		echo "pass $name: port 18000 already served; skipping"
		return 1
	fi
	pkill -STOP -u piwi -f 'scripts/[g]olden/' 2>/dev/null || true
	rm -f "$PIDF"
	setsid bash -c "echo \$\$ > $PIDF; cd '$SRC' && \
    LD_LIBRARY_PATH=/opt/rocm/rocm/lib:/opt/rocm/rocm/lib/rocm_sysdeps/lib \
    TURBINE_AMD_SMI_LIBRARY=/opt/rocm/rocm/lib/libamd_smi.so.26.5.0 ROCR_VISIBLE_DEVICES=0 \
    exec taskset -c 0-11 '$BIN/turbine-server' --config $cfg \
      --set model.path=/home/piwi/turbine-models/$slug \
      --set execution.kernel_library='$KLIB'" \
		>"$o/server.log" 2>&1 </dev/null &
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
		tail -n 40 "$o/server.log"
		cleanup_pass
		return 1
	fi
	echo "== $(date -u +%FT%TZ) pass $name: server ready"
	local rc=0 r
	if [ "$gslug" != - ]; then
		timeout 1800 "$BIN/turbine-golden" compare --url $URL \
			--reference "tests/golden/$gslug/reference.jsonl" --concurrency 1 >"$o/golden1.txt" 2>&1
		r=$?
		echo "pass $name: golden c1 rc=$r $(tail -n 1 "$o/golden1.txt")"
		timeout 1800 "$BIN/turbine-golden" compare --url $URL \
			--reference "tests/golden/$gslug/reference.jsonl" --concurrency 16 >"$o/golden16.txt" 2>&1
		r=$?
		echo "pass $name: golden c16 rc=$r $(tail -n 1 "$o/golden16.txt")"
		if [ "$name" = fp8-tensor ]; then
			timeout 1800 "$BIN/turbine-golden" capture --url $URL --prompts tests/golden/prompts.jsonl \
				--out "$o/capture.jsonl" >"$o/capture.txt" 2>&1
			echo "pass $name: capture rc=$?"
			cp -p "tests/golden/$gslug/tolerance.json" "$o/tolerance-used.json"
			sha1sum "tests/golden/$gslug/reference.jsonl" >"$o/reference-used.sha1"
		fi
	fi
	curl -s $URL/metrics >"$o/metrics-before.txt"
	measure "$o" $URL "$quality"
	rc=$?
	curl -s $URL/metrics >"$o/metrics.txt"
	curl -s $URL/turbine/v1/status >"$o/status.json"
	cleanup_pass
	summary "$name"
	[ "$rc" = 0 ] && date -u +%FT%TZ >"$o/done"
	echo "== $(date -u +%FT%TZ) pass $name: done rc=$rc"
	return "$rc"
}

kube() { kubectl "$@" 2> >(grep -v 'permission denied' >&2); }

vllm_job_gone() {
	local j=$1
	for _ in $(seq 1 60); do
		[ -z "$(kube -n $NS get pods -l "job-name=$j" -o name 2>/dev/null)" ] && return 0
		sleep 5
	done
	return 1
}

run_vllm() {
	local name=vllm slug=llama-3.2-3b-instruct-fp8-dynamic
	local o=$OUT/$name
	if [ -s "$o/done" ]; then
		echo "pass $name: already done ($(cat "$o/done")); skipping"
		return 0
	fi
	mkdir -p "$o"
	take_locks "$name"
	if ss -ltnH 'sport = :18100' | grep -q .; then
		echo "pass $name: port 18100 already served; skipping"
		return 1
	fi
	pkill -STOP -u piwi -f 'scripts/[g]olden/' 2>/dev/null || true
	local run job
	run=t14-$(date -u +%m%d%H%M%S)
	job=turbine-lab-vllm-$run
	echo "pass $name: job $job"
	sed -e "s/__RUN_ID__/$run/g" -e "s/__SLUG__/$slug/g" \
		-e "s|__SERVED_NAME__|RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic|g" \
		-e "s/__MAX_MODEL_LEN__/32768/g" -e "s/__GPUS__/1/g" -e '/# __EXTRA_ARGS__$/d' \
		scripts/lab/novanas-vllm-job.yaml >"$o/job.yaml"
	if ! kube apply -f "$o/job.yaml"; then
		echo "pass $name: kubectl apply failed"
		resume_fixtures
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
		kube -n $NS describe pods -l "job-name=$job" >"$o/pod.txt" 2>&1
		echo "pass $name: REFUSED or not ready after ${t}s (failed=$failed); log $o/vllm.log"
		tail -n 30 "$o/vllm.log"
		echo REFUSED >"$o/status"
		rc=1
	else
		echo "== $(date -u +%FT%TZ) pass $name: vLLM ready after ${t}s"
		(sleep 90 && { /opt/rocm/rocm/bin/amd-smi monitor 2>&1 || amd-smi monitor 2>&1; } | head -8 >"$o/amd-smi.txt") &
		measure "$o" $VURL 1
		rc=$?
		kube -n $NS logs "job/$job" >"$o/vllm.log" 2>&1
		echo SERVED >"$o/status"
	fi
	kube -n $NS delete job "$job" --wait=true >/dev/null 2>&1
	vllm_job_gone "$job" || echo "pass $name: pod of $job still present after 300 s"
	resume_fixtures
	summary "$name"
	# A refusal is a result (gate falls back to BF16): the pass is done either way.
	date -u +%FT%TZ >"$o/done"
	echo "== $(date -u +%FT%TZ) pass $name: done rc=$rc"
	return 0
}

{
	echo "== $(date -u +%FT%TZ) t14_proof_run.sh starting, pid $$, commit $(cat "$OUT/commit" 2>/dev/null)"
	echo "== kernel library build"
	nice -n 19 taskset -c 12-15 cmake -S kernels/rocm -B "$KB" -G Ninja -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_HIP_COMPILER=/opt/rocm/rocm/bin/hipcc -DGPU_TARGETS=gfx1201 >/dev/null &&
		nice -n 19 taskset -c 12-15 cmake --build "$KB" -j 4 >"$OUT/kbuild.log" 2>&1
	krc=$?
	echo "kernel build rc=$krc"
	if [ "$krc" != 0 ] || [ ! -x "$BIN/turbine-server" ] || [ ! -x "$BIN/turbine-golden" ] ||
		[ ! -x "$BIN/turbine-bench" ]; then
		echo "t14_proof_run: done rc=1 (build)"
		exit 1
	fi
	run_pass bf16 scripts/lab/phase2c-novanas-llama.yaml llama-3.2-3b-instruct - 1 \
		200>"$PORT_LOCK" 201>"$BENCH_GATE" 202>"$BENCH_LOCK"
	r1=$?
	run_pass fp8 scripts/lab/phase6-novanas-llama-fp8.yaml llama-3.2-3b-instruct-fp8-dynamic \
		llama-3.2-3b-instruct-fp8-dynamic 1 \
		200>"$PORT_LOCK" 201>"$BENCH_GATE" 202>"$BENCH_LOCK"
	r2=$?
	run_pass fp8-tensor scripts/lab/phase6-novanas-llama-fp8-tensor.yaml llama-3.2-3b-instruct-fp8 \
		llama-3.2-3b-instruct-fp8 0 \
		200>"$PORT_LOCK" 201>"$BENCH_GATE" 202>"$BENCH_LOCK"
	r3=$?
	run_vllm 200>"$PORT_LOCK_VLLM" 201>"$BENCH_GATE" 202>"$BENCH_LOCK"
	r4=$?
	echo "ALLDONE rc: bf16=$r1 fp8=$r2 fp8-tensor=$r3 vllm=$r4 (vllm status $(cat "$OUT/vllm/status" 2>/dev/null))"
	echo "t14_proof_run: done rc=$((r1 | r2 | r3 | r4))"
} >>"$OUT/run.log" 2>&1
