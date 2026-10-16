#!/usr/bin/env bash
# golden-engine-spread.sh — capture every golden slug under engine variants that differ only in
# summation order, for scripts/golden/engine_spread.py (decision "5p golden tolerance vs engine
# summation order", 2026-10-06 B).
#
#   scripts/lab/golden-engine-spread.sh [--dry-run] <label> [<slug>...]
#
# From the workstation: builds the release turbine-server / turbine-golden (scripts/remote-cargo.sh)
# and the HIP kernel library on novanas as scripts/lab-bench.sh does, then starts this script on
# the host with --on-host, detached (log and captures under
# /home/piwi/turbine-ci/scratch/golden-engine-spread/<label>/), and returns. Fetch the results
# with: rsync -a piwi@192.168.10.203:/home/piwi/turbine-ci/scratch/golden-engine-spread/<label>/ <dir>/
#
# On the host, for each slug (default: every slug below) it serves natively on GPU 0
# (ROCR_VISIBLE_DEVICES=0, cores 0-11, loopback port 18000) three times and records
# `turbine-golden capture` files (free-running greedy, top-20 logprobs):
#   default server:              c1-cold.jsonl (first pass, nothing cached), c1-warm.jsonl (the
#                                same prompts again: prefix reuse), par-00..15.jsonl (16 captures
#                                at once, each over the prompts rotated by its index, so every
#                                prompt decodes in a batch: batched GEMMs and, from 4 sequences,
#                                CK split-KV decode attention instead of the b1 kernel)
#   execution.gemm_autotune=false: c1-heuristic.jsonl (hipBLASLt heuristic algorithms)
#   execution.fused_ops=false:     c1-unfused.jsonl (separate QKV / gate-up GEMMs and norms)
# The whole run holds port18000.lock then bench.lock exclusively (gates first; the order of
# lab-bench.sh under bench-lock.sh --name port18000, so the two never deadlock) and pauses the CPU
# fixture jobs (pkill -STOP 'scripts/golden/'), like lab-bench.sh. Last log line:
# "golden-engine-spread: done".
set -uo pipefail

# slug | weights dir | lab config | prompts (default tests/golden/prompts.jsonl)
SLUGS="
llama-3.2-3b-instruct|llama-3.2-3b-instruct|scripts/lab/phase2c-novanas-llama.yaml|
olmoe-1b-7b-0125-instruct|olmoe-1b-7b-0125-instruct|scripts/lab/phase2c-novanas-olmoe.yaml|
llama-3.2-3b-instruct-fp8-dynamic|llama-3.2-3b-instruct-fp8-dynamic|scripts/lab/phase6-novanas-llama-fp8.yaml|
llama-3.2-3b-instruct-fp8|llama-3.2-3b-instruct-fp8|scripts/lab/phase6-novanas-llama-fp8-tensor.yaml|
llama-3.2-3b-instruct-fp8-block|llama-3.2-3b-instruct-fp8-block|scripts/lab/phase6-novanas-llama-fp8-block.yaml|
llama-3.2-3b-instruct-awq|llama-3.2-3b-instruct-awq|scripts/lab/phase6-novanas-llama-awq.yaml|
llama-3.2-3b-instruct-gptq|llama-3.2-3b-instruct-gptq|scripts/lab/phase6-novanas-llama-gptq.yaml|
llama-3.2-3b-instruct-gptq-own|llama-3.2-3b-instruct-gptq-own|scripts/lab/phase6-novanas-llama-gptq-own.yaml|
llama-3.2-3b-instruct-autoround-gptq|llama-3.2-3b-instruct-autoround-gptq|scripts/lab/phase6-novanas-llama-gptq-autoround.yaml|
llama-3.2-3b-instruct-fp8kv|llama-3.2-3b-instruct|scripts/lab/phase6-novanas-llama-fp8kv.yaml|
olmoe-1b-7b-0125-instruct-fp8kv|olmoe-1b-7b-0125-instruct|scripts/lab/phase6-novanas-olmoe-fp8kv.yaml|
llama-3.2-3b-instruct-yarn16|llama-3.2-3b-instruct|scripts/lab/phase6-novanas-llama-yarn16.yaml|tests/golden/llama-3.2-3b-instruct-yarn16/prompts.jsonl
llama-3.1-8b-instruct|llama-3.1-8b-instruct|scripts/lab/phase6-novanas-llama8b.yaml|
llama-3.1-8b-instruct-mxfp4a16|llama-3.1-8b-instruct-mxfp4a16|scripts/lab/phase6-novanas-llama8b-mxfp4.yaml|
gpt-oss-20b|gpt-oss-20b|scripts/lab/phase7-novanas-gpt-oss-20b.yaml|tests/golden/gpt-oss-20b/prompts.jsonl
"
host="${TURBINE_REMOTE_HOST:-piwi@192.168.10.203}"
ci=/home/piwi/turbine-ci
scratch=$ci/scratch/golden-engine-spread
PAR=16

usage() {
	echo "usage: scripts/lab/golden-engine-spread.sh [--dry-run] <label> [<slug>...]" >&2
	exit 2
}

on_host() { # <remote root> <label> [slugs]
	local remote=$1 label=$2
	shift 2
	local out=$scratch/$label
	local url=http://127.0.0.1:18000
	local golden="$remote/target/release/turbine-golden"
	pidf=/tmp/golden-engine-spread-server.pid # global: the EXIT trap reads it
	cd "$remote/src" || exit 1
	echo "golden-engine-spread: $(date '+%F %T') locks held, commit $(cat "$out/commit" 2>/dev/null)"
	pkill -STOP -u piwi -f 'scripts/[g]olden/' || true
	trap 'pkill -CONT -u piwi -f "scripts/[g]olden/" || true; stop_server' EXIT
	stop_server() {
		[[ -f $pidf ]] || return 0
		local p
		p=$(cat "$pidf")
		if [[ "$(cat /proc/"$p"/comm 2>/dev/null)" == turbine-server ]]; then
			kill "$p"
			for _ in $(seq 1 20); do
				[[ -d /proc/$p ]] || break
				sleep 1
			done
		fi
		rm -f $pidf
	}
	serve() { # <dir> <weights> <cfg> [--set ...]
		local dir=$1 weights=$2 cfg=$3
		shift 3
		if ss -ltnH 'sport = :18000' | grep -q .; then
			echo "golden-engine-spread: port 18000 is served by someone else; stopping" >&2
			return 1
		fi
		LD_LIBRARY_PATH=/opt/rocm/rocm/lib:/opt/rocm/rocm/lib/rocm_sysdeps/lib \
			TURBINE_AMD_SMI_LIBRARY=/opt/rocm/rocm/lib/libamd_smi.so.26.5.0 ROCR_VISIBLE_DEVICES=0 \
			setsid taskset -c 0-11 "$remote/target/release/turbine-server" --config "$cfg" \
			--set model.path=/home/piwi/turbine-models/"$weights" \
			--set execution.kernel_library="$remote/kbuild/libturbine_hip.so" \
			--set server.listen=127.0.0.1:18000 "$@" >"$dir/server.log" 2>&1 </dev/null &
		echo $! >$pidf
		for _ in $(seq 1 300); do
			[[ "$(curl -s -o /dev/null -w '%{http_code}' $url/ready)" == 200 ]] && return 0
			[[ -d /proc/$(cat "$pidf") ]] || break
			sleep 2
		done
		echo "golden-engine-spread: server not ready ($dir/server.log)" >&2
		stop_server
		return 1
	}
	capture() { # <prompts> <out>
		taskset -c 0-11 "$golden" capture --url $url --prompts "$1" --out "$2" >"$2.log" 2>&1
	}
	local want=" $* " slug weights cfg prompts dir n i
	while IFS='|' read -r slug weights cfg prompts; do
		[[ -n "$slug" ]] || continue
		[[ "$want" == "  " || "$want" == *" $slug "* ]] || continue
		prompts=${prompts:-tests/golden/prompts.jsonl}
		dir=$out/$slug
		mkdir -p "$dir/rot"
		if [[ ! -f /home/piwi/turbine-models/$weights/config.json ]]; then
			echo "golden-engine-spread: $slug: no weights, skipped"
			continue
		fi
		n=$(grep -c . "$prompts")
		for i in $(seq 0 $((PAR - 1))); do
			local k=$((i % n))
			{
				tail -n +$((k + 1)) "$prompts"
				head -n $k "$prompts"
			} | grep . >"$dir/rot/$i.jsonl"
		done
		echo "golden-engine-spread: $(date '+%T') $slug default"
		if serve "$dir" "$weights" "$cfg"; then
			capture "$prompts" "$dir/c1-cold.jsonl"
			capture "$prompts" "$dir/c1-warm.jsonl"
			local pids=()
			for i in $(seq 0 $((PAR - 1))); do
				capture "$dir/rot/$i.jsonl" "$dir/$(printf 'par-%02d' "$i").jsonl" &
				pids+=($!)
			done
			wait "${pids[@]}" # not a bare wait: the server is a child of this shell too
			curl -s $url/turbine/v1/status >"$dir/status.json"
			stop_server
		fi
		echo "golden-engine-spread: $(date '+%T') $slug gemm_autotune=false"
		if serve "$dir" "$weights" "$cfg" --set execution.gemm_autotune=false; then
			capture "$prompts" "$dir/c1-heuristic.jsonl"
			stop_server
		fi
		echo "golden-engine-spread: $(date '+%T') $slug fused_ops=false"
		if serve "$dir" "$weights" "$cfg" --set execution.fused_ops=false; then
			capture "$prompts" "$dir/c1-unfused.jsonl"
			stop_server
		fi
		echo "golden-engine-spread: $slug captures: $(find "$dir" -maxdepth 1 -name "*.jsonl" | wc -l)"
	done <<<"$SLUGS"
	echo "golden-engine-spread: done"
}

if [[ "${1:-}" == "--on-host" ]]; then
	shift
	on_host "$@"
	exit $?
fi
dry=0
if [[ "${1:-}" == "--dry-run" ]]; then
	dry=1
	shift
fi
[[ $# -ge 1 ]] || usage
label=$1
shift
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root" || exit 1
remote="$ci/remote/$(basename "$root")"
out=$scratch/$label
locks="flock -x $ci/port18000.gate flock -x $ci/port18000.lock flock -x $ci/bench.gate flock -x $ci/bench.lock"
start="mkdir -p $out && echo $(git rev-parse --short HEAD) > $out/commit && cd $remote/src && \
setsid nohup $locks bash scripts/lab/golden-engine-spread.sh --on-host $remote $label $* \
  > $out/run.log 2>&1 < /dev/null &"
build_kernels="cd '$remote/src' && nice -n 19 taskset -c 12-15 cmake -S kernels/rocm -B '$remote/kbuild' -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER=/opt/rocm/rocm/bin/hipcc -DGPU_TARGETS=gfx1201 >/dev/null && \
  nice -n 19 taskset -c 12-15 cmake --build '$remote/kbuild' -j 4 >/dev/null"
if [[ $dry -eq 1 ]]; then
	echo "+ scripts/remote-cargo.sh build -q --release -p turbine-server -p turbine-bench"
	echo "+ ssh $host \"$build_kernels\""
	echo "+ ssh $host \"$start\""
	exit 0
fi
scripts/remote-cargo.sh build -q --release -p turbine-server -p turbine-bench || exit 1
ssh -o BatchMode=yes "$host" "$build_kernels" || exit 1
ssh -o BatchMode=yes "$host" "$start" || exit 1
echo "golden-engine-spread: started on $host; log $out/run.log"
