#!/usr/bin/env bash
# fp8block-bench: Task 15 GPU job A, detached on novanas (rotation 11, lead brief r11-fp8block).
# The equivalent of `scripts/lab-bench.sh --model llama-fp8-block --golden16 --c1`, run as a
# native detached script (not lab-bench.sh itself, which foreground-blocks the caller for the
# whole measurement): builds (CPU: nice 19, cores 12-15, 4 jobs) the kernel library and the
# release server/bench/golden from this workspace's src, waits for the go-file
# fp8block-bench.go, then under port18000.lock -> bench.gate -> bench.lock on GPU 0 (CPU fixture
# jobs paused): serves the fp8_block config, runs golden c1, golden c16, the c16 throughput bench
# and the c1 single-request latency run, and prints a BENCH-equivalent line.
# Last line of $O/fp8block-bench.log: `fp8block-bench: done rc=<rc>`.
set -uo pipefail
R=/home/piwi/turbine-ci/remote/agent-a4baec4689b995379
O=$R/fp8block-bench
GO=/home/piwi/turbine-ci/gpu-queue/fp8block-bench.go
BENCH_GATE=/home/piwi/turbine-ci/bench.gate
BENCH_LOCK=/home/piwi/turbine-ci/bench.lock
PORT_LOCK=/home/piwi/turbine-ci/port18000.lock
MODEL=/home/piwi/turbine-models/llama-3.2-3b-instruct-fp8-block
SLUG=llama-3.2-3b-instruct-fp8-block
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

# serve: starts the fp8_block server on GPU 0 and waits for /ready (<= 10 min). The latency-drift
# circuit threshold is relaxed for this single-request-heavy measurement run (golden at c1 and
# the c1 latency run both serialize one request at a time, which can look like a latency drift
# to the default threshold of 4.0); circuit transitions are grepped into the log regardless.
serve() {
	if ss -ltnH 'sport = :18000' | grep -q .; then
		echo "fp8block-bench: port 18000 already served; not starting"
		return 1
	fi
	rm -f "$PIDF"
	(cd "$R/src" && setsid bash -c 'echo $$ >"$0"; exec "$@"' "$PIDF" \
		env LD_LIBRARY_PATH=$LIBS TURBINE_AMD_SMI_LIBRARY=/opt/rocm/rocm/lib/libamd_smi.so.26.5.0 ROCR_VISIBLE_DEVICES=0 \
		taskset -c 0-11 "$R/target/release/turbine-server" --config "$R/src/scripts/lab/phase6-novanas-llama-fp8-block.yaml" \
		--set model.path=$MODEL --set execution.kernel_library=$R/kbuild/libturbine_hip.so \
		--set reliability.circuit.latency_drift_open=100 \
		>"$O/server.log" 2>&1 </dev/null &)
	for _ in $(seq 1 20); do
		[ -s "$PIDF" ] && break
		sleep 0.5
	done
	for _ in $(seq 1 300); do
		[ "$(curl -s -o /dev/null -w '%{http_code}' $URL/ready)" = 200 ] && return 0
		[ -d /proc/$(cat "$PIDF") ] || {
			echo "fp8block-bench: server exited"
			tail -30 "$O/server.log"
			return 1
		}
		sleep 2
	done
	echo "fp8block-bench: not ready"
	tail -30 "$O/server.log"
	stop_server
	return 1
}

# measure: golden c1 / c16, the c16 throughput bench (200 requests) and the c1 single-request
# latency run (10 requests, 128 tokens, --ignore-eos), reported the way lab-bench.sh does
# (itl_c1_p50 / tok/s_c1).
measure() {
	local rc=0
	(cd "$R/src" && taskset -c 0-11 "$R/target/release/turbine-golden" compare --url $URL \
		--reference tests/golden/$SLUG/reference.jsonl --concurrency 1) >"$O/golden1.txt" 2>&1 || rc=1
	(cd "$R/src" && taskset -c 0-11 "$R/target/release/turbine-golden" compare --url $URL \
		--reference tests/golden/$SLUG/reference.jsonl --concurrency 16) >"$O/golden16.txt" 2>&1 || rc=1
	taskset -c 0-11 "$R/target/release/turbine-bench" --url $URL --concurrency 16 --requests 200 \
		--prompt-words 512 --max-tokens 256 --ignore-eos --output json >"$O/bench.json" 2>"$O/bench.err" || rc=1
	taskset -c 0-11 "$R/target/release/turbine-bench" --url $URL --concurrency 1 --requests 10 \
		--max-tokens 128 --ignore-eos --output json >"$O/bench-c1.json" 2>"$O/bench-c1.err" || rc=1
	curl -s $URL/turbine/v1/status >"$O/status.json"
	echo "fp8block-bench: $(ts) golden1: $(tail -1 "$O/golden1.txt")"
	echo "fp8block-bench: $(ts) golden16: $(tail -1 "$O/golden16.txt")"
	python3 -c "
import json
b = json.load(open('$O/bench.json'))
c1 = json.load(open('$O/bench-c1.json'))
print('fp8block-bench: bench tok/s', b.get('output_token_throughput'), 'ok', b.get('requests_ok'))
itl = (c1.get('itl_ms') or {}).get('p50')
print('fp8block-bench: c1 itl_ms.p50', itl, 'tok/s_c1', c1.get('output_token_throughput'), 'ok', c1.get('requests_ok'))
" 2>&1 || true
	grep -iE '"event":"circuit_transition"|circuit.*(HEALTHY|DEGRADED|CIRCUIT_OPEN|DRAINING|PROBING)' "$O/server.log" | sed 's/^/fp8block-bench: circuit: /' || true
	return $rc
}

main() {
	mkdir -p "$O"
	echo "fp8block-bench: $(ts) start pid $$"
	cd "$R/src" || return 1
	echo "fp8block-bench: $(ts) kernel library build"
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
	echo "fp8block-bench: $(ts) release binaries"
	cargo_ build --release -p turbine-server -p turbine-bench >"$O/cargo-build.log" 2>&1 || {
		tail -40 "$O/cargo-build.log"
		return 1
	}
	echo "fp8block-bench: $(ts) built; waiting for go-file $GO"
	local waited=0
	while [ ! -e "$GO" ]; do
		[ $((waited % 1800)) -eq 0 ] && echo "fp8block-bench: $(ts) waiting for go-file"
		sleep 60
		waited=$((waited + 60))
		[ "$waited" -ge 86400 ] && {
			echo "fp8block-bench: $(ts) go-file timeout after 24h"
			return 1
		}
	done
	echo "fp8block-bench: $(ts) go-file present; waiting port18000.lock"
	exec 200>"$PORT_LOCK" 201>"$BENCH_GATE" 202>"$BENCH_LOCK"
	flock -x 200
	echo "fp8block-bench: $(ts) waiting bench.gate"
	flock -x 201
	echo "fp8block-bench: $(ts) waiting bench.lock"
	flock -x 202
	echo "fp8block-bench: $(ts) locks held"
	trap "stop_server; pkill -CONT -u piwi -f 'scripts/[g]olden/' 2>/dev/null || true" EXIT
	pkill -STOP -u piwi -f 'scripts/[g]olden/' 2>/dev/null || true
	local rc=0
	if serve; then
		measure || rc=1
		stop_server
	else
		rc=1
	fi
	local commit
	commit=$(git -C "$R/src" rev-parse --short HEAD 2>/dev/null || echo unknown)
	echo "fp8block-bench: BENCH r11-fp8block llama-fp8-block commit=$commit gpu=0 rc=$rc (see golden1.txt golden16.txt bench.json bench-c1.json status.json under $O)"
	return $rc
}

main >>"$O/fp8block-bench.log" 2>&1
rc=$?
mkdir -p "$O"
echo "fp8block-bench: done rc=$rc" >>"$O/fp8block-bench.log"
