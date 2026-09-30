#!/usr/bin/env bash
# W4A4 exit-segfault repro (p6a-w4a4-numerics): start/serve/SIGTERM cycles of the W4A4 8B server
# (RoPE-patched model dir), core dumps on, rocgdb backtrace of any core.
#   OLD = the frozen q4n binaries of the 0.740 run (engine thread not joined at exit)
#   NEW = this branch's release turbine-server + kernel library (engine threads joined)
# Waits for the go-file gpu-queue/w4a4-segv.go, then port18000 -> bench.gate -> bench.lock.
# Log: $W/w4a4segv.log, last line `w4a4-segv: done rc=<rc>`; per cycle `cycle <set> <i> rc=<rc>`
# (139 = SIGSEGV); backtraces in $W/bt-<set>-<i>.txt. Cores: private anon memory only, at most
# 30 GiB, deleted after the backtrace.
set -uo pipefail
W=/home/piwi/turbine-ci/scratch/w4a4-segv
CI=/home/piwi/turbine-ci
OLD=/home/piwi/turbine-ci/remote/agent-a4784842b25c93376/q4n/bin
R=/home/piwi/turbine-ci/remote/agent-p6a-w4a4-numerics
MODEL=/home/piwi/turbine-ci/scratch/w4a4-numerics/model
CFG=/home/piwi/turbine-ci/scratch/w4a4-numerics/llama8b-a4.yaml
NAME=amd/Llama-3.1-8B-Instruct-MXFP4-W4A4-MLCAL-C1000-GPTQ
CYCLES=${CYCLES:-8}
GO=$CI/gpu-queue/w4a4-segv.go
LOG=$W/w4a4segv.log
say() { echo "$(date -u +%FT%TZ) w4a4-segv: $*"; }
SPID=
cleanup() {
	pkill -CONT -u piwi -f 'scripts/[g]olden/' 2>/dev/null || true
	if [ -n "$SPID" ] && [ -d "/proc/$SPID" ]; then
		kill "$SPID" 2>/dev/null
		for _ in $(seq 1 20); do
			[ -d "/proc/$SPID" ] || break
			sleep 1
		done
		kill -9 "$SPID" 2>/dev/null
	fi
}
trap cleanup EXIT
cycle() { # <set> <bindir> <libpath> <i>
	local set=$1 bin=$2 lib=$3 i=$4 run=$W/run-$1-$4
	mkdir -p "$run"
	(cd "$run" && ulimit -c 31457280 && echo 0x1 >/proc/self/coredump_filter &&
		LD_LIBRARY_PATH=/opt/rocm/rocm/lib:/opt/rocm/rocm/lib/rocm_sysdeps/lib \
			TURBINE_AMD_SMI_LIBRARY=/opt/rocm/rocm/lib/libamd_smi.so.26.5.0 ROCR_VISIBLE_DEVICES=0 \
			exec taskset -c 0-11 "$bin/turbine-server" --config "$CFG" \
			--set model.path="$MODEL" --set execution.kernel_library="$lib") \
		>"$run/server.log" 2>&1 </dev/null &
	SPID=$!
	local ready=0 code
	for _ in $(seq 1 180); do
		code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:18000/ready)
		[ "$code" = 200 ] && {
			ready=1
			break
		}
		[ -d "/proc/$SPID" ] || break
		sleep 2
	done
	if [ "$ready" != 1 ]; then
		say "cycle $set $i SERVER NOT READY"
		tail -n 20 "$run/server.log"
		cleanup
		wait "$SPID"
		SPID=
		return 90
	fi
	for q in "What is 17 times 23?" "Name three primes above 100."; do
		curl -s -m 120 http://127.0.0.1:18000/v1/chat/completions -H 'Content-Type: application/json' \
			-d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"max_tokens\":64}" >/dev/null
	done
	kill -TERM "$SPID"
	local waited=0
	while [ -d "/proc/$SPID" ] && [ $waited -lt 60 ]; do
		sleep 1
		waited=$((waited + 1))
	done
	[ -d "/proc/$SPID" ] && {
		say "cycle $set $i did not exit in 60 s; SIGKILL"
		kill -9 "$SPID"
	}
	wait "$SPID"
	local rc=$?
	say "cycle $set $i rc=$rc exit_after_sigterm_s<=$waited stopped=$(grep -c engine_stopped "$run/server.log") complete=$(grep -c 'shutdown complete' "$run/server.log")"
	SPID=
	local core
	core=$(find "$run" -maxdepth 1 -name "core*" -type f | head -1)
	if [ -n "$core" ]; then
		say "cycle $set $i core $(du -h "$core" | cut -f1)"
		timeout 600 /opt/rocm/rocm/bin/rocgdb -batch -ex 'info threads' -ex 'thread apply all bt 30' \
			"$bin/turbine-server" "$core" >"$W/bt-$set-$i.txt" 2>&1
		grep -n -A12 'received signal\|SIGSEGV\|^#0' "$W/bt-$set-$i.txt" | head -60
		rm -f "$core"
	fi
	return 0
}
run() {
	say "starting, pid $$"
	cat /proc/sys/kernel/core_pattern
	for f in "$OLD/turbine-server" "$OLD/libturbine_hip.so" "$R/target/release/turbine-server" "$R/kbuild/libturbine_hip.so"; do
		[ -f "$f" ] || {
			say "missing $f"
			say "done rc=2"
			return 2
		}
	done
	mkdir -p "$W/new"
	cp "$R/target/release/turbine-server" "$R/kbuild/libturbine_hip.so" "$W/new/"
	say "waiting for go-file $GO"
	local n=0
	while [ ! -e "$GO" ]; do
		sleep 60
		n=$((n + 1))
		[ $n -gt 1440 ] && {
			say "no go-file in 24 h"
			say "done rc=3"
			return 3
		}
	done
	say "waiting for port18000 lock"
	flock -x 200
	say "waiting for bench.gate"
	flock -x 201
	say "waiting for bench.lock"
	flock -x 202
	say "locks held"
	pkill -STOP -u piwi -f 'scripts/[g]olden/' 2>/dev/null || true
	local i
	for i in $(seq 1 "$CYCLES"); do cycle old "$OLD" "$OLD/libturbine_hip.so" "$i"; done
	for i in $(seq 1 "$CYCLES"); do cycle new "$W/new" "$W/new/libturbine_hip.so" "$i"; done
	pkill -CONT -u piwi -f 'scripts/[g]olden/' 2>/dev/null || true
	say "summary old: $(grep -c 'cycle old .* rc=139' "$LOG") segv of $CYCLES; new: $(grep -c 'cycle new .* rc=139' "$LOG") segv of $CYCLES"
	say "done rc=0"
	return 0
}
mkdir -p "$W"
run >>"$LOG" 2>&1 200>"$CI/port18000.lock" 201>"$CI/bench.gate" 202>"$CI/bench.lock"
