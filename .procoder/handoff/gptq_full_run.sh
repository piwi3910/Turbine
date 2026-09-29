#!/usr/bin/env bash
# Full-GSM8K (1319 items) at concurrency 16 on Turbine GPTQ INT4 (Phase 6a Task 18 numerics
# investigation, p6a-gptq-numerics). Built from agent-a67eec8abc8f117eb/gsm8k_full_run_r5.sh.
# Runs entirely on novanas, detached (setsid nohup), so it survives the caller's disconnect:
# kernel library build (no lock, nice 19, cores 12-15), then one pass under port18000.lock ->
# bench.gate -> bench.lock (the one-GPU-job PSU rule), CPU fixture jobs paused, GPU 0, cores 0-11.
# Pair: tests/eval/llama-3.2-3b-instruct/turbine-bf16-full.json (0.7801, concurrency 16).
# Output: $REMOTE/gptq-full/turbine-full.json (+ .err, server.log). Last log line:
# "gptq-full: done rc=<rc>".
set -uo pipefail

REMOTE=/home/piwi/turbine-ci/remote/agent-p6a-gptq-numerics
SRC=$REMOTE/src
KB=$REMOTE/kbuild
KLIB=$KB/libturbine_hip.so
BIN=$REMOTE/target/release
BENCH_GATE=/home/piwi/turbine-ci/bench.gate
BENCH_LOCK=/home/piwi/turbine-ci/bench.lock
PORT_LOCK=/home/piwi/turbine-ci/port18000.lock
OUT=$REMOTE/gptq-full
URL=http://127.0.0.1:18000
PIDF=/tmp/gptqfull-server.pid

mkdir -p "$OUT"
cd "$SRC" || exit 1

cleanup_pass() {
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
	pkill -CONT -u piwi -f 'scripts/[g]olden/' 2>/dev/null || true
}

run_pass() {
	local out=$OUT/turbine-full.json
	echo "== $(date -u +%FT%TZ) waiting for port18000 lock"
	flock -x 200
	echo "== $(date -u +%FT%TZ) waiting for bench.gate"
	flock -x 201
	echo "== $(date -u +%FT%TZ) waiting for bench.lock"
	flock -x 202
	echo "== $(date -u +%FT%TZ) locks held, starting"
	if ss -ltnH 'sport = :18000' | grep -q .; then
		echo "port 18000 already served; not starting"
		return 1
	fi
	pkill -STOP -u piwi -f 'scripts/[g]olden/' 2>/dev/null || true
	rm -f "$PIDF"
	setsid bash -c "echo \$\$ > $PIDF; cd '$SRC' && \
    LD_LIBRARY_PATH=/opt/rocm/rocm/lib:/opt/rocm/rocm/lib/rocm_sysdeps/lib \
    TURBINE_AMD_SMI_LIBRARY=/opt/rocm/rocm/lib/libamd_smi.so.26.5.0 ROCR_VISIBLE_DEVICES=0 \
    exec taskset -c 0-11 '$BIN/turbine-server' --config scripts/lab/phase6-novanas-llama-gptq.yaml \
      --set model.path=/home/piwi/turbine-models/llama-3.2-3b-instruct-gptq \
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
		echo "SERVER NOT READY"
		tail -n 60 "$OUT/server.log"
		cleanup_pass
		return 1
	fi
	echo "== $(date -u +%FT%TZ) server ready, full-GSM8K eval at c16 (6 h timeout)"
	rm -f "$out" "${out%.json}.err"
	timeout 21600 taskset -c 0-11 "$BIN/turbine-golden" eval --url $URL \
		--tasks tests/eval/gsm8k-full.jsonl --concurrency 16 --output json \
		>"$out" 2>"${out%.json}.err"
	local rc=$?
	echo "== $(date -u +%FT%TZ) eval rc=$rc"
	curl -s $URL/turbine/v1/status >"$OUT/status.json"
	cleanup_pass
	return "$rc"
}

{
	echo "== $(date -u +%FT%TZ) gptq_full_run.sh starting, pid $$"
	echo "== kernel library build"
	nice -n 19 taskset -c 12-15 cmake -S kernels/rocm -B "$KB" -G Ninja -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_HIP_COMPILER=/opt/rocm/rocm/bin/hipcc -DGPU_TARGETS=gfx1201 >/dev/null &&
		nice -n 19 taskset -c 12-15 cmake --build "$KB" -j 4 >"$OUT/kbuild.log" 2>&1
	krc=$?
	echo "kernel build rc=$krc"
	if [ "$krc" != 0 ] || [ ! -x "$BIN/turbine-server" ] || [ ! -x "$BIN/turbine-golden" ]; then
		echo "gptq-full: done rc=1 (build)"
		exit 1
	fi
	run_pass 200>"$PORT_LOCK" 201>"$BENCH_GATE" 202>"$BENCH_LOCK"
	rc=$?
	echo "gptq-full: done rc=$rc"
} >>"$OUT/run.log" 2>&1
