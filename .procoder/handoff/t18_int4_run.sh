#!/usr/bin/env bash
# Phase 6a Task 18 INT4 proof driver. Runs entirely on novanas, detached (setsid nohup), so it
# survives the caller's disconnect. Builds the kernel library (no lock, nice 19, cores 12-15),
# then three passes one after another, each under port18000.lock -> bench.gate -> bench.lock
# (the one-GPU-job PSU rule) with CPU fixture jobs paused, GPU 0, cores 0-11:
#   bf16 : golden c1, bench c16 (200 req) and c1 — the same-run baseline for the INT4 targets
#   awq  : golden c1 + c16, bench c16 (200 req) and c1
#   gptq : golden c1 + c16, bench c16 (200 req) and c1, GSM8K-200 eval
# The measurement commands are lab-bench.sh's (client on novanas against the loopback URL).
# Results: $OUT/<pass>/{golden1.txt,golden16.txt,bench.json,bench-c1.json,quality.json,
# metrics.txt,status.json,server.log}. Last log line: "t18_int4_run: done rc=<rc>".
set -uo pipefail

REMOTE=/home/piwi/turbine-ci/remote/agent-abee0e2542a317f3c
SRC=$REMOTE/src
KB=$REMOTE/kbuild
KLIB=$KB/libturbine_hip.so
BIN=$REMOTE/target/release
BENCH_GATE=/home/piwi/turbine-ci/bench.gate
BENCH_LOCK=/home/piwi/turbine-ci/bench.lock
PORT_LOCK=/home/piwi/turbine-ci/port18000.lock
OUT=/home/piwi/turbine-ci/scratch/p6a-int4/t18-run
URL=http://127.0.0.1:18000
PIDF=/tmp/t18int4-server.pid

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
		fi
		rm -f "$PIDF"
	fi
}

cleanup_pass() {
	stop_server
	pkill -CONT -u piwi -f 'scripts/[g]olden/' 2>/dev/null || true
}

# run_pass <name> <config> <weights slug> <golden slug|-> <golden16 0|1> <quality 0|1>
run_pass() {
	local name=$1 cfg=$2 slug=$3 gslug=$4 g16=$5 quality=$6
	local o=$OUT/$name
	mkdir -p "$o"
	echo "== $(date -u +%FT%TZ) pass $name: waiting for port18000 lock"
	flock -x 200
	echo "== $(date -u +%FT%TZ) pass $name: waiting for bench.gate"
	flock -x 201
	echo "== $(date -u +%FT%TZ) pass $name: waiting for bench.lock"
	flock -x 202
	echo "== $(date -u +%FT%TZ) pass $name: locks held, starting"
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
	local rc=0
	if [ "$gslug" != - ]; then
		timeout 1800 "$BIN/turbine-golden" compare --url $URL \
			--reference "tests/golden/$gslug/reference.jsonl" --concurrency 1 >"$o/golden1.txt" 2>&1
		echo "pass $name: golden c1 rc=$? $(tail -n 1 "$o/golden1.txt")"
		if [ "$g16" = 1 ]; then
			timeout 1800 "$BIN/turbine-golden" compare --url $URL \
				--reference "tests/golden/$gslug/reference.jsonl" --concurrency 16 >"$o/golden16.txt" 2>&1
			echo "pass $name: golden c16 rc=$? $(tail -n 1 "$o/golden16.txt")"
		fi
	fi
	curl -s $URL/metrics >"$o/metrics-before.txt"
	timeout 1800 taskset -c 0-11 "$BIN/turbine-bench" --url $URL --concurrency 16 --requests 200 \
		--prompt-words 512 --max-tokens 256 --ignore-eos --output json >"$o/bench.json" 2>"$o/bench.err"
	echo "pass $name: bench c16 rc=$?"
	timeout 900 taskset -c 0-11 "$BIN/turbine-bench" --url $URL --concurrency 1 --requests 10 \
		--max-tokens 128 --ignore-eos --output json >"$o/bench-c1.json" 2>"$o/bench-c1.err"
	echo "pass $name: bench c1 rc=$?"
	curl -s $URL/metrics >"$o/metrics.txt"
	curl -s $URL/turbine/v1/status >"$o/status.json"
	if [ "$quality" = 1 ]; then
		timeout 7200 taskset -c 0-11 "$BIN/turbine-golden" eval --url $URL \
			--tasks tests/eval/gsm8k-200.jsonl --output json >"$o/quality.json" 2>"$o/quality.err"
		rc=$?
		echo "pass $name: gsm8k-200 rc=$rc"
	fi
	cleanup_pass
	echo "== $(date -u +%FT%TZ) pass $name: done"
	return "$rc"
}

{
	echo "== $(date -u +%FT%TZ) t18_int4_run.sh starting, pid $$, commit $(cat "$OUT/commit" 2>/dev/null)"
	echo "== kernel library build"
	nice -n 19 taskset -c 12-15 cmake -S kernels/rocm -B "$KB" -G Ninja -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_HIP_COMPILER=/opt/rocm/rocm/bin/hipcc -DGPU_TARGETS=gfx1201 >/dev/null &&
		nice -n 19 taskset -c 12-15 cmake --build "$KB" -j 4 >"$OUT/kbuild.log" 2>&1
	krc=$?
	echo "kernel build rc=$krc"
	if [ "$krc" != 0 ] || [ ! -x "$BIN/turbine-server" ]; then
		echo "t18_int4_run: done rc=1 (build)"
		exit 1
	fi
	run_pass bf16 scripts/lab/phase2c-novanas-llama.yaml llama-3.2-3b-instruct llama-3.2-3b-instruct 0 0 \
		200>"$PORT_LOCK" 201>"$BENCH_GATE" 202>"$BENCH_LOCK"
	r1=$?
	run_pass awq scripts/lab/phase6-novanas-llama-awq.yaml llama-3.2-3b-instruct-awq llama-3.2-3b-instruct-awq 1 0 \
		200>"$PORT_LOCK" 201>"$BENCH_GATE" 202>"$BENCH_LOCK"
	r2=$?
	run_pass gptq scripts/lab/phase6-novanas-llama-gptq.yaml llama-3.2-3b-instruct-gptq llama-3.2-3b-instruct-gptq 1 1 \
		200>"$PORT_LOCK" 201>"$BENCH_GATE" 202>"$BENCH_LOCK"
	r3=$?
	echo "ALLDONE rc: bf16=$r1 awq=$r2 gptq=$r3"
	echo "t18_int4_run: done rc=$((r1 | r2 | r3))"
} >"$OUT/run.log" 2>&1
