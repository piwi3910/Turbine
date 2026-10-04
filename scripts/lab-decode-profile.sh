#!/usr/bin/env bash
# lab-decode-profile.sh — per-kernel profile of a batch-1 decode step on novanas GPU 0
# (phase-5p Task 18 / S-19).
#
#   scripts/lab-decode-profile.sh [--model llama|olmoe] [--label L] [--attach-ms <ms>]
#                                 [--requests <n>] [--max-tokens <n>] [--dry-run]
#
# 1. Builds the release server and the HIP kernel library (cached, like lab-bench.sh).
# 2. Takes the host's port18000 lock, then the exclusive benchmark lock (bench.gate →
#    bench.lock), serves the model's phase2c config natively on GPU 0.
# 3. Starts the fixed single-request load (turbine-bench --concurrency 1 --ignore-eos) and,
#    once decode steps are running, attaches rocprofv3 to the server for --attach-ms of kernel
#    trace. The c1 load keeps every step batch-1, so the trace is pure decode steps.
# 4. Stops its own server (pid file, bounded kill), releases both locks on every exit path,
#    copies the trace and the scrape results to target/decodebw/<label>-<model>/ and prints one
#    PROFILE line. The step split (per-kernel time and inter-kernel gaps) is computed from the
#    CSV by scripts/decode-profile-report.py on the workstation.
set -uo pipefail
orig_args=("$@")

host="${TURBINE_REMOTE_HOST:-piwi@192.168.10.203}"
model=llama
label="decodeprof"
attach_ms=3000
requests=24
max_tokens=256
dry=0
usage() {
	echo "usage: scripts/lab-decode-profile.sh [--model llama|olmoe] [--label L] [--attach-ms <ms>] [--requests <n>] [--max-tokens <n>] [--dry-run]" >&2
	exit 2
}
while [[ $# -gt 0 ]]; do
	case "$1" in
	--model)
		model="$2"
		shift 2
		;;
	--label)
		label="$2"
		shift 2
		;;
	--attach-ms)
		[[ "$2" =~ ^[0-9]+$ ]] || usage
		attach_ms="$2"
		shift 2
		;;
	--requests)
		[[ "$2" =~ ^[0-9]+$ ]] || usage
		requests="$2"
		shift 2
		;;
	--max-tokens)
		[[ "$2" =~ ^[0-9]+$ ]] || usage
		max_tokens="$2"
		shift 2
		;;
	--dry-run)
		dry=1
		shift
		;;
	--) shift break ;;
	*) usage ;;
	esac
done
case "$model" in
llama)
	slug=llama-3.2-3b-instruct
	cfg=scripts/lab/phase2c-novanas-llama.yaml
	;;
olmoe)
	slug=olmoe-1b-7b-0125-instruct
	cfg=scripts/lab/phase2c-novanas-olmoe.yaml
	;;
*)
	echo "lab-decode-profile: unknown model $model" >&2
	usage
	;;
esac

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root" || exit 1
name="$(basename "$root")"
remote="/home/piwi/turbine-ci/remote/$name"
outdir="$root/target/decodebw/$label-$model"
mkdir -p "$outdir"
url=http://192.168.10.203:18000
pidf=/tmp/decode-profile-server.pid
profdir="$remote/decodeprof"

# Only this run's server is stopped (pid file, verified turbine-server, bounded kill); the
# CPU fixture jobs pause from the lock hold to the end, as in lab-bench.sh.
pause_fixtures="pkill -STOP -u piwi -f 'scripts/[g]olden/'"
resume_fixtures="pkill -CONT -u piwi -f 'scripts/[g]olden/'"
stop_server="if [ -f $pidf ]; then p=\$(cat $pidf); \
  if [ \"\$(cat /proc/\$p/comm 2>/dev/null)\" = turbine-server ]; then kill \$p; \
    for i in 1 2 3 4 5 6 7 8 9 10 15 20; do [ -d /proc/\$p ] || break; sleep 1; done; \
    [ -d /proc/\$p ] && kill -9 \$p; fi; rm -f $pidf; fi"
cleanup() {
	if [[ $dry -eq 0 ]]; then
		ssh -o BatchMode=yes "$host" "$stop_server" >/dev/null 2>&1
		ssh -o BatchMode=yes "$host" "pkill -f 'rocprofv3 --attach' >/dev/null 2>&1" || true
		ssh -o BatchMode=yes "$host" "$resume_fixtures" >/dev/null 2>&1 || true
	fi
	if [[ -n "${release_locks:-}" ]]; then
		release_locks
	fi
}
trap cleanup EXIT

# The port lock wraps the whole run (the bench lock is taken inside it).
if [[ $dry -eq 0 && -z "${DECODE_PROFILE_PORT_LOCKED:-}" ]]; then
	DECODE_PROFILE_PORT_LOCKED=1 exec scripts/bench-lock.sh --host "$host" --name port18000 \
		"$root/scripts/lab-decode-profile.sh" "${orig_args[@]}"
fi

ssh() {
	if [[ $dry -eq 1 ]]; then
		echo "+ ssh $*"
	else
		command ssh "$@"
	fi
}

# Build (server + kernels), like lab-bench.sh. Builds never take the bench lock.
if [[ $dry -eq 1 ]]; then
	echo "+ scripts/remote-cargo.sh build -q --release -p turbine-server -p turbine-bench"
	echo "+ ssh $host 'cmake -S kernels/rocm -B $remote/kbuild ... && cmake --build $remote/kbuild'"
else
	if ! scripts/remote-cargo.sh build -q --release -p turbine-server -p turbine-bench \
		>"$outdir/build-cargo.log" 2>&1; then
		echo "PROFILE $label $model BUILD FAILED (see $outdir/build-cargo.log)"
		exit 1
	fi
	if ! ssh -o BatchMode=yes "$host" "cd '$remote/src' && nice -n 19 taskset -c 12-15 \
      cmake -S kernels/rocm -B '$remote/kbuild' -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_HIP_COMPILER=/opt/rocm/rocm/bin/hipcc -DGPU_TARGETS=gfx1201 >/dev/null &&
      nice -n 19 taskset -c 12-15 cmake --build '$remote/kbuild' -j 4" \
		>"$outdir/build-kernels.log" 2>&1; then
		echo "PROFILE $label $model BUILD FAILED (see $outdir/build-kernels.log)"
		exit 1
	fi
fi

# The exclusive bench lock (gate first), held by a remote flock fed by a fifo, as lab-bench.sh.
if [[ $dry -eq 1 ]]; then
	echo "+ ssh $host 'flock -x bench.gate flock -x bench.lock (the exclusive bench lock)'"
else
	lockdir="$(mktemp -d)"
	mkfifo "$lockdir/in" "$lockdir/out"
	ssh -o BatchMode=yes "$host" \
		"flock -x /home/piwi/turbine-ci/bench.gate flock -x /home/piwi/turbine-ci/bench.lock sh -c 'echo locked; cat >/dev/null'" \
		<"$lockdir/in" >"$lockdir/out" &
	exec 7>"$lockdir/in"
	echo "lab-decode-profile: waiting for the exclusive bench lock" >&2
	read -r lock_state <"$lockdir/out"
	if [[ "$lock_state" != locked ]]; then
		echo "lab-decode-profile: could not take the bench lock on $host" >&2
		exit 1
	fi
	release_locks() {
		exec 7>&- 2>/dev/null
		wait $! 2>/dev/null
		rm -rf "$lockdir"
	}
	ssh -o BatchMode=yes "$host" "$pause_fixtures" >/dev/null 2>&1 || true
fi

# Only this run's server is stopped (pid file, verified turbine-server, bounded kill); the
# CPU fixture jobs pause from the lock hold to the end, as in lab-bench.sh.
if [[ $dry -eq 0 ]] && ssh -o BatchMode=yes "$host" "ss -ltnH 'sport = :18000' | grep -q ."; then
	echo "lab-decode-profile: port 18000 on $host is already served; stop that server first" >&2
	exit 1
fi
ssh -o BatchMode=yes "$host" "rm -rf $profdir; mkdir -p $profdir"
if [[ $dry -eq 1 ]]; then
	echo "+ ssh $host 'setsid ... turbine-server --config $cfg ...'"
else
	ssh -o BatchMode=yes "$host" "setsid bash -c \"echo \\\$\\\$ > $pidf; cd '$remote/src' && \
    LD_LIBRARY_PATH=/opt/rocm/rocm/lib:/opt/rocm/rocm/lib/rocm_sysdeps/lib \
    TURBINE_AMD_SMI_LIBRARY=/opt/rocm/rocm/lib/libamd_smi.so.26.5.0 ROCR_VISIBLE_DEVICES=0 \
    ROCP_TOOL_ATTACH=1 \
    exec taskset -c 0-11 '$remote/target/release/turbine-server' --config $cfg \
      --set model.path=/home/piwi/turbine-models/$slug \
      --set execution.kernel_library='$remote/kbuild/libturbine_hip.so'\" \
    > /tmp/decode-profile-server.log 2>&1 < /dev/null &"
	ready=0
	for _ in $(seq 1 150); do
		if [[ "$(curl -s -o /dev/null -w '%{http_code}' "$url/ready")" == 200 ]]; then
			ready=1
			break
		fi
		sleep 2
	done
	if [[ $ready -ne 1 ]]; then
		ssh -o BatchMode=yes "$host" "tail -30 /tmp/decode-profile-server.log" >"$outdir/server.log" 2>&1
		echo "PROFILE $label $model SERVER NOT READY (see $outdir/server.log)"
		exit 1
	fi
	owner=$(ssh -o BatchMode=yes "$host" "ss -ltnpH 'sport = :18000' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2")
	mine=$(ssh -o BatchMode=yes "$host" "cat $pidf 2>/dev/null")
	if [[ -z "$owner" || "$owner" != "$mine" ]]; then
		echo "PROFILE $label $model PORT TAKEN (served by pid ${owner:-unknown}, not this run's ${mine:-none})"
		exit 1
	fi
fi

# The single-request load runs on novanas (the workstation has no Linux turbine-bench).
if [[ $dry -eq 1 ]]; then
	echo "+ ssh $host 'turbine-bench --concurrency 1 --requests $requests --max-tokens $max_tokens --ignore-eos --output json'"
	echo "+ ssh $host 'rocprofv3 --attach <pid> --attach-duration-msec $attach_ms --kernel-trace --memory-copy-trace --hip-runtime-trace -f csv'"
else
	# The module/kernel choices of this server, for the record.
	curl -s "$url/turbine/v1/status" >"$outdir/status.json"
	curl -s "$url/metrics" >"$outdir/metrics-before.txt"
	ssh -o BatchMode=yes "$host" "taskset -c 0-11 '$remote/target/release/turbine-bench' \
    --url http://127.0.0.1:18000 --concurrency 1 --requests $requests \
    --max-tokens $max_tokens --ignore-eos --output json" \
		>"$outdir/bench-c1.json" 2>"$outdir/bench-c1.err" &
	bench_pid=$!
	# Past the prefill and a few steps, then attach for the window.
	sleep 12
	server_pid=$(ssh -o BatchMode=yes "$host" "cat $pidf")
	ssh -o BatchMode=yes "$host" "cd $profdir && \
    LD_LIBRARY_PATH=/opt/rocm/rocm/lib /opt/rocm/rocm/bin/rocprofv3 \
      --attach $server_pid --attach-duration-msec $attach_ms \
      --kernel-trace --memory-copy-trace --hip-runtime-trace -f csv" \
		>"$outdir/rocprof.log" 2>&1
	wait "$bench_pid"
	bench_rc=$?
	curl -s "$url/metrics" >"$outdir/metrics.txt"
	scp -o BatchMode=yes -q "$host:$profdir/*.csv" "$outdir/" 2>/dev/null
	tok=$(python3 -c "import json;d=json.load(open('$outdir/bench-c1.json'));print(d.get('output_token_throughput',0))" 2>/dev/null)
	echo "PROFILE $label $model commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown) attach_ms=$attach_ms tok/s_c1=$tok bench_rc=$bench_rc out=$outdir"
fi
