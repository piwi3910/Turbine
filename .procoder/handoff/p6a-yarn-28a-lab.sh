#!/usr/bin/env bash
# P6a Task 28a lab run, detached on novanas (YaRN attention factor on cos/sin, kernel ABI v2.10).
# Builds (CPU: nice 19, cores 12-15, 4 jobs) the kernel library, the release server / bench /
# golden and the golden test binary from this workspace's src, then under port18000.lock ->
# bench.gate -> bench.lock on GPU 0 (CPU fixture jobs paused):
#   1. turbine-model --test golden yarn_teacher_forced_vs_reference for p16,p17-long (hip + cpu)
#   2. serve llama-yarn16 (phase6 yarn16 config): golden c1 + c16 vs the yarn16 reference, c16 bench
#   3. serve llama (phase2c config): golden c1 + c16 vs the llama reference, c16 bench
# Last line of $R/t28a-lab.log: `t28a-lab: done rc=<rc>`.
set -uo pipefail
R=/home/piwi/turbine-ci/remote/agent-p6a-yarn-t28a
O=$R/t28a-lab
BENCH_GATE=/home/piwi/turbine-ci/bench.gate
BENCH_LOCK=/home/piwi/turbine-ci/bench.lock
PORT_LOCK=/home/piwi/turbine-ci/port18000.lock
MODEL=/home/piwi/turbine-models/llama-3.2-3b-instruct
LIBS=/opt/rocm/rocm/lib:/opt/rocm/rocm/lib/rocm_sysdeps/lib
URL=http://127.0.0.1:18000
PIDF=$O/server.pid
ts() { date -u +%FT%TZ; }
cargo_() { CARGO_TARGET_DIR=$R/target CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=4 nice -n 19 taskset -c 12-15 "$HOME/.cargo/bin/cargo" +1.97 "$@"; }

stop_server() {
	[ -f "$PIDF" ] || return 0
	local p
	p=$(cat "$PIDF")
	if [ "$(cat /proc/$p/comm 2>/dev/null)" = turbine-server ]; then
		kill "$p"
		for _ in $(seq 1 20); do
			[ -d /proc/$p ] || break
			sleep 1
		done
	fi
	rm -f "$PIDF"
}

# serve <name> <config>: starts this run's server on GPU 0 and waits for /ready (≤ 10 min).
serve() {
	local name=$1 cfg=$2
	if ss -ltnH 'sport = :18000' | grep -q .; then
		echo "t28a-lab: port 18000 already served; not starting $name"
		return 1
	fi
	rm -f "$PIDF"
	(cd "$R/src" && setsid bash -c 'echo $$ >"$0"; exec "$@"' "$PIDF" \
		env LD_LIBRARY_PATH=$LIBS TURBINE_AMD_SMI_LIBRARY=/opt/rocm/rocm/lib/libamd_smi.so.26.5.0 ROCR_VISIBLE_DEVICES=0 \
		taskset -c 0-11 "$R/target/release/turbine-server" --config "$cfg" \
		--set model.path=$MODEL --set execution.kernel_library=$R/kbuild/libturbine_hip.so \
		>"$O/server-$name.log" 2>&1 </dev/null &)
	for _ in $(seq 1 20); do
		[ -s "$PIDF" ] && break
		sleep 0.5
	done
	for _ in $(seq 1 300); do
		[ "$(curl -s -o /dev/null -w '%{http_code}' $URL/ready)" = 200 ] && return 0
		[ -d /proc/$(cat "$PIDF") ] || {
			echo "t28a-lab: $name server exited"
			tail -20 "$O/server-$name.log"
			return 1
		}
		sleep 2
	done
	echo "t28a-lab: $name not ready"
	tail -20 "$O/server-$name.log"
	stop_server
	return 1
}

# measure <name> <golden slug>: golden c1 / c16 and the c16 throughput bench (64 requests).
measure() {
	local name=$1 slug=$2 rc=0
	(cd "$R/src" && taskset -c 0-11 "$R/target/release/turbine-golden" compare --url $URL \
		--reference tests/golden/$slug/reference.jsonl --concurrency 1) >"$O/$name-golden1.txt" 2>&1 || rc=1
	(cd "$R/src" && taskset -c 0-11 "$R/target/release/turbine-golden" compare --url $URL \
		--reference tests/golden/$slug/reference.jsonl --concurrency 16) >"$O/$name-golden16.txt" 2>&1 || rc=1
	taskset -c 0-11 "$R/target/release/turbine-bench" --url $URL --concurrency 16 --requests 64 \
		--prompt-words 512 --max-tokens 256 --ignore-eos --output json >"$O/$name-bench.json" 2>"$O/$name-bench.err" || rc=1
	curl -s $URL/turbine/v1/status >"$O/$name-status.json"
	echo "t28a-lab: $(ts) $name golden1: $(tail -1 "$O/$name-golden1.txt")"
	echo "t28a-lab: $(ts) $name golden16: $(tail -1 "$O/$name-golden16.txt")"
	python3 -c "import json;d=json.load(open('$O/$name-bench.json'));print('t28a-lab: $name bench tok/s', d.get('output_token_throughput'), 'ok', d.get('requests_ok'))" || true
	return $rc
}

main() {
	mkdir -p "$O"
	echo "t28a-lab: $(ts) start pid $$ (src $(cat $R/src/.t28a-commit 2>/dev/null || echo ?))"
	cd "$R/src" || return 1
	echo "t28a-lab: $(ts) kernel library build"
	if [ ! -f "$R/kbuild/build.ninja" ]; then
		nice -n 19 taskset -c 12-15 cmake -S kernels/rocm -B "$R/kbuild" -G Ninja -DCMAKE_BUILD_TYPE=Release \
			-DCMAKE_HIP_COMPILER=/opt/rocm/rocm/bin/hipcc -DGPU_TARGETS=gfx1201 >"$O/cmake.log" 2>&1 || {
			tail -30 "$O/cmake.log"
			return 1
		}
	fi
	nice -n 19 taskset -c 12-15 cmake --build "$R/kbuild" -j 4 >"$O/kbuild.log" 2>&1 || {
		tail -40 "$O/kbuild.log"
		return 1
	}
	echo "t28a-lab: $(ts) release binaries"
	cargo_ build --release -p turbine-server -p turbine-bench >"$O/cargo-build.log" 2>&1 || {
		tail -40 "$O/cargo-build.log"
		return 1
	}
	local bin
	bin=$(cargo_ test --release -p turbine-model --test golden --no-run --message-format=json 2>>"$O/cargo-test-build.err" |
		grep -o '"executable":"[^"]*golden-[^"]*"' | tail -1 | cut -d'"' -f4)
	[ -n "$bin" ] && [ -x "$bin" ] || {
		echo "t28a-lab: golden test build failed"
		tail -40 "$O/cargo-test-build.err"
		return 1
	}
	echo "t28a-lab: $(ts) built; waiting for go-file"
	local GO=/home/piwi/turbine-ci/gpu-queue/t28a-lab.go waited=0
	while [ ! -e "$GO" ]; do
		[ $((waited % 1800)) -eq 0 ] && echo "t28a-lab: $(ts) waiting for go-file"
		sleep 60
		waited=$((waited + 60))
		[ "$waited" -ge 86400 ] && {
			echo "t28a-lab: $(ts) go-file timeout after 24h"
			return 1
		}
	done
	echo "t28a-lab: $(ts) go-file present; waiting port18000.lock"
	exec 200>"$PORT_LOCK" 201>"$BENCH_GATE" 202>"$BENCH_LOCK"
	flock -x 200
	echo "t28a-lab: $(ts) waiting bench.gate"
	flock -x 201
	echo "t28a-lab: $(ts) waiting bench.lock"
	flock -x 202
	echo "t28a-lab: $(ts) locks held"
	trap "stop_server; pkill -CONT -u piwi -f 'scripts/[g]olden/' 2>/dev/null || true" EXIT
	pkill -STOP -u piwi -f 'scripts/[g]olden/' 2>/dev/null || true
	local rc=0
	echo "t28a-lab: $(ts) 1. teacher-forced p16,p17-long"
	(cd "$R/src/crates/turbine-model" && LD_LIBRARY_PATH=$LIBS ROCR_VISIBLE_DEVICES=0 TURBINE_TEST_BACKEND=hip \
		TURBINE_KERNEL_LIBRARY=$R/kbuild/libturbine_hip.so TURBINE_TEST_MODEL_DIR=$MODEL TURBINE_GOLDEN_YARN=p16,p17-long \
		timeout 4h taskset -c 0-11 "$bin" --ignored --exact yarn_teacher_forced_vs_reference --nocapture) >"$O/tf.out" 2>&1 || rc=1
	grep -E '^YaRN .* max' "$O/tf.out" | sed 's/^/t28a-lab: /'
	echo "t28a-lab: $(ts) 2. llama-yarn16"
	if serve yarn16 "$R/src/scripts/lab/phase6-novanas-llama-yarn16.yaml"; then
		measure yarn16 llama-3.2-3b-instruct-yarn16 || rc=1
		stop_server
	else rc=1; fi
	echo "t28a-lab: $(ts) 3. llama (BF16, unchanged)"
	if serve llama "$R/src/scripts/lab/phase2c-novanas-llama.yaml"; then
		measure llama llama-3.2-3b-instruct || rc=1
		stop_server
	else rc=1; fi
	return $rc
}

main >>"$R/t28a-lab.log" 2>&1
rc=$?
echo "t28a-lab: done rc=$rc" >>"$R/t28a-lab.log"
