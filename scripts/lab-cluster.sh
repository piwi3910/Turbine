#!/usr/bin/env bash
# Run one Phase 5 multi-GPU lab scenario on novanas as a k3s Job holding both R9700s.
#
#   scripts/lab-cluster.sh [--dry-run] <collbench-novanas|collbench-sweep-novanas|
#                                       collbench-hostmem-novanas|tp2-novanas|dp2-novanas|
#                                       ep2-novanas>
#   scripts/lab-cluster.sh [--dry-run] --stop <run-id>
#
# Scenarios (P5 S-9; everything runs inside the Job on loopback — no Service, no host port):
#   collbench-novanas  turbine-collbench --backend rccl --devices 0,1 --op all --max-bytes 1GiB;
#                      PASS when every row is correct and all-reduce busbw > 0 at 268,435,456 B.
#   collbench-sweep-novanas  small-message all-reduce latency under RCCL settings (diagnostics).
#   collbench-hostmem-novanas  turbine-collbench --op all, 8 B .. 1 GiB, BF16, for hostmem's
#                      kernels at every size (--route-max-bytes 1TiB), hostmem routed at its
#                      measured crossover (hostmem-auto) and rccl, each per op (op +
#                      synchronize, median) and --pipelined (back to back, mean): every row
#                      printed as `hm <hostmem|hostmem-auto|rccl> <per-op|pipelined> <op>
#                      <bytes> <time_us> <busbw_gbps>`; PASS when every row of every run is
#                      correct (each rank checks its result bit for bit against the host
#                      reference backend, so both ranks hold the same bits). Keep the numbers
#                      only from a run under scripts/bench-lock.sh.
#   tp2-novanas        Llama-3.2-3B-Instruct: first a tp 1 baseline on device 0 on the standard
#                      throughput workload (512-word prompts, 256 tokens with --ignore-eos,
#                      concurrency 16, 200 requests, after a 16-request warm-up); then tp 2 in
#                      local mode (golden at concurrency 1 — strict bounds — and 16 — batched
#                      bounds, the TP accuracy bound; turbine-bench --concurrency 4 --requests 64
#                      must report requests_ok 64; the standard workload), then OLMoE-1B-7B at
#                      tp 2 in local mode (golden c1 and c16), then Llama in static mode (ranks
#                      0 and 1, leader 127.0.0.1:18100; golden c1 and c16). Prints one
#                      `tp-bench <run> tok/s=… ttft_p50_ms=… itl_p50_ms=…` line per bench run.
#                      2-GPU numbers are "2-GPU (GPU0 Gen5 x8 + GPU1 Gen4 x8)".
#   dp2-novanas        Llama-3.2-3B-Instruct at tp 1 on the standard throughput workload
#                      (512-word prompts, 256 tokens with --ignore-eos, 200 requests, after a
#                      16-request warm-up): first a dp 1 baseline on device 0 at concurrency 16,
#                      then dp 2 (one replica per R9700): golden at concurrency 1 and 16, the
#                      bench at concurrency 32 (the load that fills both cards) and at 16 (the
#                      same load as dp 1), each requests_ok 200, and turbine_dp_routed_total
#                      non-zero for replica 0 and replica 1.
#   ep2-novanas        OLMoE-1B-7B (scripts/lab/phase5-novanas-ep2.yaml): first an ep 1 baseline
#                      on device 0 on the standard throughput workload (512-word prompts, 256
#                      tokens with --ignore-eos, concurrency 16, 200 requests, after a
#                      16-request warm-up); then ep 2 with tp 1 over rccl (golden at
#                      concurrency 1 and 16, turbine_expert_rank_tokens_total non-zero for rank
#                      0 and rank 1, the standard workload), ep 2 with tp 1 over hostmem (golden
#                      c1, the standard workload) and ep 2 with tp 2 (golden c1). Prints one
#                      `ep-bench <run> tok/s=… ttft_p50_ms=… itl_p50_ms=…` line per bench run and
#                      the scheduler document's `expert` section. 2-GPU numbers are "2-GPU
#                      (GPU0 Gen5 x8 + GPU1 Gen4 x8)".
#
# Like scripts/lab-test.sh, the tree is uploaded to /home/piwi/turbine-ci/runs/<run id>/src and
# one template (scripts/lab/novanas-cluster-job.yaml) runs twice: first GPU-less as
# turbine-lab-cluster-build-<run id> (kernels and the release binaries into the cached slot
# cluster-0), then as turbine-lab-cluster-<run id> with `amd.com/gpu: 2`, which runs
# `scripts/lab-cluster.sh --in-job <scenario>` from that slot. A GPU Job still Pending with
# "Insufficient amd.com/gpu" after 60 s is deleted and the script exits 1 with
# `lab-cluster: amd.com/gpu unavailable on novanas` — another workload holds a card; ask the
# user, never evict it. Whatever happens, the script deletes only the Jobs of its own run
# (label selector turbine-lab=true,turbine-lab-run=<run id>) and its upload.
# --dry-run prints every command that would contact the host (and the rendered Jobs).
#
# The last line is `lab-cluster: <scenario> PASS` (exit 0) or `lab-cluster: <scenario> FAIL
# <reason>` (exit 1); usage errors exit 2.
set -euo pipefail

SCENARIOS="collbench-novanas|collbench-sweep-novanas|collbench-hostmem-novanas|tp2-novanas|dp2-novanas|ep2-novanas"

usage() {
	echo "usage: scripts/lab-cluster.sh [--dry-run] <${SCENARIOS}>" >&2
	echo "       scripts/lab-cluster.sh [--dry-run] --stop <run-id>" >&2
	exit 2
}

valid_scenario() {
	[[ "$1" =~ ^(collbench-novanas|collbench-sweep-novanas|collbench-hostmem-novanas|tp2-novanas|dp2-novanas|ep2-novanas)$ ]]
}

# ---------------------------------------------------------------------------------------------
# Inside the Job: `scripts/lab-cluster.sh --in-job <scenario>`, run from the slot's source tree
# with the release binaries in $CARGO_TARGET_DIR/release and the weights at /models.
# ---------------------------------------------------------------------------------------------

HTTP=127.0.0.1:18000
URL="http://${HTTP}"
SERVER_PIDS=()
# The fixed throughput workload of scripts/lab-bench.sh and scripts/lab-perf.sh (concurrency
# and request count are added per run).
STANDARD_BENCH=(--prompt-words 512 --max-tokens 256 --ignore-eos)

job_fail() {
	echo "lab-cluster: ${SCENARIO} FAIL $1"
	stop_servers
	exit 1
}

stop_servers() {
	local pid
	for pid in ${SERVER_PIDS[@]+"${SERVER_PIDS[@]}"}; do
		kill -TERM "$pid" 2>/dev/null || true
	done
	for pid in ${SERVER_PIDS[@]+"${SERVER_PIDS[@]}"}; do
		wait "$pid" 2>/dev/null || true
	done
	SERVER_PIDS=()
}

# start_server <log> <config> [--set k=v]...: turbine-server in the background.
start_server() {
	local log="$1" config="$2"
	shift 2
	"${BIN}/turbine-server" --config "$config" "$@" >"$log" 2>&1 &
	SERVER_PIDS+=($!)
	echo "lab-step: turbine-server --config ${config} $* (pid $!, log ${log})"
}

# wait_ready <url> <log>: /ready 200 within 30 min (cold weight loads), else fail with the log.
wait_ready() {
	local url="$1" log="$2" waited
	for ((waited = 0; waited < 1800; waited += 2)); do
		if [[ "$(curl -s -o /dev/null -w '%{http_code}' "${url}/ready")" == 200 ]]; then
			echo "lab-step: ${url}/ready 200 after ${waited} s"
			return 0
		fi
		if ! kill -0 "${SERVER_PIDS[-1]}" 2>/dev/null; then
			tail -n 40 "$log"
			job_fail "turbine-server exited before ${url}/ready answered 200"
		fi
		sleep 2
	done
	tail -n 40 "$log"
	job_fail "${url}/ready not 200 within 30 min"
}

# golden <slug> <label> [compare args]...: turbine-golden compare against the committed reference.
golden() {
	local slug="$1" label="$2"
	shift 2
	echo "lab-step: golden ${label} $*"
	"${BIN}/turbine-golden" compare --url "$URL" \
		--reference "tests/golden/${slug}/reference.jsonl" "$@" ||
		job_fail "golden ${label} outside tolerance"
}

# bench_ok <requests> <json> [bench args]...: turbine-bench must report requests_ok <requests>.
bench_ok() {
	local want="$1" out="$2"
	shift 2
	echo "lab-step: turbine-bench $*"
	"${BIN}/turbine-bench" --url "$URL" "$@" --output json >"$out" ||
		job_fail "turbine-bench failed"
	cat "$out"
	jq -e --argjson want "$want" '.requests_ok == $want' "$out" >/dev/null ||
		job_fail "turbine-bench requests_ok is not $want"
}

scenario_collbench() {
	local out="${WORK}/collbench.json"
	echo "lab-step: turbine-collbench --backend rccl --devices 0,1 --op all --max-bytes 1GiB"
	"${BIN}/turbine-collbench" --backend rccl --devices 0,1 --op all --max-bytes 1GiB \
		--output json >"$out" || {
		cat "$out"
		job_fail "turbine-collbench exited non-zero"
	}
	cat "$out"
	jq -e '([.ops[].rows[]] | length > 0 and all(.correct))
		and ([.ops[] | select(.op == "all_reduce") | .rows[] | select(.bytes == 268435456)
			| .busbw_gbps] | length == 1 and .[0] > 0)' "$out" >/dev/null ||
		job_fail "a row is incorrect or all-reduce busbw is 0 at 268435456 B"
}

# collbench-sweep-novanas: small-message all-reduce latency under RCCL settings (diagnostics; always
# PASS unless a run fails). For each setting, BF16 all-reduce 8 B .. 16 MiB twice: op + synchronize
# per iteration (median) and back to back with one synchronize (mean per op). Every row is printed
# as `sweep <setting> <per-op|pipelined> <bytes> <time_us>`; the default setting also runs once with
# NCCL_DEBUG=INFO so the log shows the transport, algorithm and protocol RCCL picks.
scenario_collbench_sweep() {
	local settings=(
		"default"
		"NCCL_PROTO=LL"
		"NCCL_PROTO=LL128"
		"NCCL_PROTO=Simple"
		"NCCL_ALGO=Tree"
		"NCCL_ALGO=Ring"
		"RCCL_MSCCL_ENABLE=0"
		"RCCL_MSCCLPP_ENABLE=0"
		"NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1"
		"NCCL_SHM_USE_CUDA_MEMCPY=1"
		"HSA_ENABLE_SDMA=0"
	)
	echo "lab-step: NCCL_DEBUG=INFO turbine-collbench --op all_reduce --max-bytes 8"
	NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,ENV "${BIN}/turbine-collbench" \
		--backend rccl --devices 0,1 --op all_reduce --max-bytes 8 --iters 5 --warmup 1 \
		--output json >/dev/null || job_fail "NCCL_DEBUG run failed"
	local s mode out
	for s in "${settings[@]}"; do
		for mode in per-op pipelined; do
			out="${WORK}/sweep.json"
			local flags=(--backend rccl --devices 0,1 --op all_reduce --max-bytes 16MiB --iters 50
				--warmup 10 --output json)
			[[ $mode == pipelined ]] && flags+=(--pipelined)
			echo "lab-step: [$s] turbine-collbench ${flags[*]}"
			if [[ $s == default ]]; then
				"${BIN}/turbine-collbench" "${flags[@]}" >"$out" || job_fail "sweep [$s] $mode failed"
			else
				env $s "${BIN}/turbine-collbench" "${flags[@]}" >"$out" ||
					job_fail "sweep [$s] $mode failed"
			fi
			jq -e '[.ops[].rows[].correct] | all' "$out" >/dev/null ||
				job_fail "sweep [$s] $mode: a row is incorrect"
			jq -r --arg s "$s" --arg m "$mode" \
				'.ops[].rows[] | "sweep \($s | gsub(" "; "+")) \($m) \(.bytes) \(.time_us)"' "$out"
		done
	done
}

# tp_prefix_check <label>: is prefix reuse bit-exact on the running server? One greedy request
# with a ~600-token prompt (whole 128-token blocks cached) under cache salt A (cold), again under
# A (its prefix blocks reused) and under salt B (cold again); prints whether the top-5 logprobs
# of the reused and the repeated cold run equal the first bit for bit. Informational only.
tp_prefix_check() {
	local label="$1" prompt i
	prompt="$(for ((i = 0; i < 120; i++)); do printf 'The engine counts step %d of the plan. ' "$i"; done)"
	local body
	body="$(jq -n --arg p "$prompt" '{model: "meta-llama/Llama-3.2-3B-Instruct", prompt: $p,
		max_tokens: 16, temperature: 0, logprobs: 5}')"
	for i in a1:A a2:A b1:B; do
		curl -s -H 'content-type: application/json' -H "x-turbine-cache-salt: ${i#*:}" \
			-d "$body" "${URL}/v1/completions" >"${WORK}/prefix-${label}-${i%%:*}.json"
	done
	local cold reused again
	cold="$(jq -S -c '.choices[0].logprobs' "${WORK}/prefix-${label}-a1.json")"
	reused="$(jq -S -c '.choices[0].logprobs' "${WORK}/prefix-${label}-a2.json")"
	again="$(jq -S -c '.choices[0].logprobs' "${WORK}/prefix-${label}-b1.json")"
	echo "tp-prefix ${label} cached_tokens=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' \
		"${WORK}/prefix-${label}-a2.json") reuse_bitexact=$([[ "$cold" == "$reused" ]] && echo yes || echo no) cold_repeat_bitexact=$([[ "$cold" == "$again" ]] && echo yes || echo no)"
}

# collbench-hostmem-novanas: hostmem against rccl, every op, 8 B .. 1 GiB, per op and pipelined.
scenario_collbench_hostmem() {
	local run backend mode out
	# hostmem = its kernels at every size; hostmem-auto = routed at the measured crossover.
	for run in hostmem hostmem-auto rccl; do
		backend="${run%-auto}"
		for mode in per-op pipelined; do
			out="${WORK}/hm-${run}-${mode}.json"
			local flags=(--backend "$backend" --devices 0,1 --op all --max-bytes 1GiB --output json)
			[[ $run == hostmem ]] && flags+=(--route-max-bytes 1TiB)
			[[ $mode == pipelined ]] && flags+=(--pipelined)
			echo "lab-step: turbine-collbench ${flags[*]}"
			"${BIN}/turbine-collbench" "${flags[@]}" >"$out" || {
				cat "$out"
				job_fail "turbine-collbench ${run} ${mode} exited non-zero"
			}
			jq -e '([.ops[].rows[]] | length > 0 and all(.correct))' "$out" >/dev/null ||
				job_fail "${run} ${mode}: a row differs from the host reference"
			jq -r --arg b "$run" --arg m "$mode" \
				'.ops[] | .op as $op | .rows[] | "hm \($b) \($m) \($op) \(.bytes) \(.time_us) \(.busbw_gbps)"' "$out"
		done
	done
}

scenario_tp2() {
	local llama=scripts/lab/phase5-novanas-llama.yaml olmoe=scripts/lab/phase5-novanas-olmoe.yaml
	# The tp 1 baseline: the same configuration with one rank on device 0.
	start_server "${WORK}/tp1.log" "$llama" --set parallel.tensor_parallel_size=1 \
		--set "parallel.devices=[0]"
	wait_ready "$URL" "${WORK}/tp1.log"
	tp_prefix_check tp1
	bench_ok 16 "${WORK}/tp1-warmup.json" "${STANDARD_BENCH[@]}" --concurrency 16 --requests 16
	bench_ok 200 "${WORK}/tp1-c16-bench.json" "${STANDARD_BENCH[@]}" --concurrency 16 \
		--requests 200
	stop_servers

	start_server "${WORK}/tp2-local.log" "$llama"
	wait_ready "$URL" "${WORK}/tp2-local.log"
	tp_prefix_check tp2-local
	golden llama-3.2-3b-instruct "llama tp2 local c1"
	golden llama-3.2-3b-instruct "llama tp2 local c16" --concurrency 16
	bench_ok 64 "${WORK}/tp2-bench.json" --concurrency 4 --requests 64
	bench_ok 16 "${WORK}/tp2-warmup.json" "${STANDARD_BENCH[@]}" --concurrency 16 --requests 16
	bench_ok 200 "${WORK}/tp2-c16-bench.json" "${STANDARD_BENCH[@]}" --concurrency 16 \
		--requests 200
	local f
	for f in tp1-c16 tp2-c16 tp2; do
		jq -r --arg f "$f" '"tp-bench \($f) tok/s=\(.output_token_throughput) ttft_p50_ms=\(.ttft_ms.p50) itl_p50_ms=\(.itl_ms.p50) requests_ok=\(.requests_ok)"' \
			"${WORK}/${f}-bench.json"
	done
	stop_servers

	start_server "${WORK}/tp2-olmoe.log" "$olmoe"
	wait_ready "$URL" "${WORK}/tp2-olmoe.log"
	golden olmoe-1b-7b-0125-instruct "olmoe tp2 local c1"
	golden olmoe-1b-7b-0125-instruct "olmoe tp2 local c16" --concurrency 16
	stop_servers

	# Static mode: one process per rank; rank 1 joins the leader on 127.0.0.1:18100 and serves
	# only /health, /ready and /metrics on its own port.
	local static=(--set parallel.ranks.mode=static --set parallel.ranks.leader=127.0.0.1:18100)
	start_server "${WORK}/tp2-static-rank1.log" "$llama" "${static[@]}" \
		--set parallel.ranks.rank=1 --set "parallel.ranks.local_devices=[1]" \
		--set server.listen=127.0.0.1:18001
	start_server "${WORK}/tp2-static-rank0.log" "$llama" "${static[@]}" \
		--set parallel.ranks.rank=0 --set "parallel.ranks.local_devices=[0]"
	wait_ready "$URL" "${WORK}/tp2-static-rank0.log"
	tp_prefix_check tp2-static
	golden llama-3.2-3b-instruct "llama tp2 static c1"
	golden llama-3.2-3b-instruct "llama tp2 static c16" --concurrency 16
	stop_servers
}

scenario_dp2() {
	local dp2=scripts/lab/phase5-novanas-dp2.yaml
	# The dp 1 baseline: the same configuration with one replica on device 0.
	start_server "${WORK}/dp1.log" "$dp2" --set parallel.data_parallel_size=1 \
		--set "parallel.devices=[0]"
	wait_ready "$URL" "${WORK}/dp1.log"
	bench_ok 16 "${WORK}/dp1-warmup.json" "${STANDARD_BENCH[@]}" --concurrency 16 --requests 16
	bench_ok 200 "${WORK}/dp1-c16-bench.json" "${STANDARD_BENCH[@]}" --concurrency 16 \
		--requests 200
	stop_servers

	start_server "${WORK}/dp2.log" "$dp2"
	wait_ready "$URL" "${WORK}/dp2.log"
	golden llama-3.2-3b-instruct "llama dp2 c1"
	golden llama-3.2-3b-instruct "llama dp2 c16" --concurrency 16
	bench_ok 32 "${WORK}/dp2-warmup.json" "${STANDARD_BENCH[@]}" --concurrency 32 --requests 32
	bench_ok 200 "${WORK}/dp2-c32-bench.json" "${STANDARD_BENCH[@]}" --concurrency 32 \
		--requests 200
	bench_ok 200 "${WORK}/dp2-c16-bench.json" "${STANDARD_BENCH[@]}" --concurrency 16 \
		--requests 200
	local f
	for f in dp1-c16 dp2-c32 dp2-c16; do
		jq -r --arg f "$f" '"dp-bench \($f) tok/s=\(.output_token_throughput) ttft_p50_ms=\(.ttft_ms.p50) itl_p50_ms=\(.itl_ms.p50)"' \
			"${WORK}/${f}-bench.json"
	done
	curl -s "${URL}/metrics" >"${WORK}/dp2-metrics.txt" || job_fail "GET /metrics failed"
	grep '^turbine_dp_routed_total' "${WORK}/dp2-metrics.txt" || true
	local replica total
	for replica in 0 1; do
		total="$(awk -v r="replica=\"$replica\"" '/^turbine_dp_routed_total\{/ && index($0, r) { s += $NF } END { print s + 0 }' \
			"${WORK}/dp2-metrics.txt")"
		awk -v t="$total" 'BEGIN { exit !(t > 0) }' ||
			job_fail "turbine_dp_routed_total is 0 for replica $replica"
	done
	stop_servers
}

# ep_counts <label>: both EP ranks computed token-expert assignments
# (turbine_expert_rank_tokens_total{rank} > 0 for rank 0 and rank 1); prints the series, the
# imbalance and the scheduler document's `expert` section.
ep_counts() {
	local label="$1" rank total
	curl -s "${URL}/metrics" >"${WORK}/${label}-metrics.txt" || job_fail "GET /metrics failed"
	grep -E '^turbine_expert_(rank_tokens_total|imbalance_ratio)' "${WORK}/${label}-metrics.txt" || true
	curl -s "${URL}/turbine/v1/scheduler" | jq -c '.["0"].expert' || true
	curl -s "${URL}/turbine/v1/status" | jq -c '.parallel | {tp, ep, backend, plan_reasons, experts: .groups[0].experts}' || true
	for rank in 0 1; do
		total="$(awk -v r="rank=\"$rank\"" '/^turbine_expert_rank_tokens_total\{/ && index($0, r) { s += $NF } END { print s + 0 }' \
			"${WORK}/${label}-metrics.txt")"
		echo "ep-counts ${label} rank=${rank} tokens=${total}"
		awk -v t="$total" 'BEGIN { exit !(t > 0) }' ||
			job_fail "turbine_expert_rank_tokens_total is 0 for rank ${rank} (${label})"
	done
}

scenario_ep2() {
	local ep2=scripts/lab/phase5-novanas-ep2.yaml slug=olmoe-1b-7b-0125-instruct
	# The ep 1 baseline: the same configuration with one device.
	start_server "${WORK}/ep1.log" "$ep2" --set parallel.expert_parallel_size=1 \
		--set "parallel.devices=[0]"
	wait_ready "$URL" "${WORK}/ep1.log"
	bench_ok 16 "${WORK}/ep1-warmup.json" "${STANDARD_BENCH[@]}" --concurrency 16 --requests 16
	bench_ok 200 "${WORK}/ep1-c16-bench.json" "${STANDARD_BENCH[@]}" --concurrency 16 \
		--requests 200
	stop_servers

	start_server "${WORK}/ep2.log" "$ep2"
	wait_ready "$URL" "${WORK}/ep2.log"
	golden "$slug" "olmoe ep2 c1"
	golden "$slug" "olmoe ep2 c16" --concurrency 16
	ep_counts ep2
	bench_ok 16 "${WORK}/ep2-warmup.json" "${STANDARD_BENCH[@]}" --concurrency 16 --requests 16
	bench_ok 200 "${WORK}/ep2-c16-bench.json" "${STANDARD_BENCH[@]}" --concurrency 16 \
		--requests 200
	ep_counts ep2-after-bench
	stop_servers

	start_server "${WORK}/ep2-hostmem.log" "$ep2" --set parallel.collective_backend=hostmem
	wait_ready "$URL" "${WORK}/ep2-hostmem.log"
	golden "$slug" "olmoe ep2 hostmem c1"
	bench_ok 16 "${WORK}/ep2-hostmem-warmup.json" "${STANDARD_BENCH[@]}" --concurrency 16 \
		--requests 16
	bench_ok 200 "${WORK}/ep2-hostmem-c16-bench.json" "${STANDARD_BENCH[@]}" --concurrency 16 \
		--requests 200
	ep_counts ep2-hostmem
	stop_servers

	start_server "${WORK}/ep2-tp2.log" "$ep2" --set parallel.tensor_parallel_size=2
	wait_ready "$URL" "${WORK}/ep2-tp2.log"
	golden "$slug" "olmoe ep2 tp2 c1"
	ep_counts ep2-tp2
	stop_servers

	local f
	for f in ep1-c16 ep2-c16 ep2-hostmem-c16; do
		jq -r --arg f "$f" '"ep-bench \($f) tok/s=\(.output_token_throughput) ttft_p50_ms=\(.ttft_ms.p50) itl_p50_ms=\(.itl_ms.p50) requests_ok=\(.requests_ok)"' \
			"${WORK}/${f}-bench.json"
	done
}

in_job() {
	SCENARIO="$1"
	valid_scenario "$SCENARIO" || usage
	: "${CARGO_TARGET_DIR:?the Job sets CARGO_TARGET_DIR}"
	BIN="${CARGO_TARGET_DIR}/release"
	WORK="$(mktemp -d)"
	trap 'stop_servers' EXIT
	case "$SCENARIO" in
	collbench-novanas) scenario_collbench ;;
	collbench-sweep-novanas) scenario_collbench_sweep ;;
	collbench-hostmem-novanas) scenario_collbench_hostmem ;;
	tp2-novanas) scenario_tp2 ;;
	dp2-novanas) scenario_dp2 ;;
	ep2-novanas) scenario_ep2 ;;
	esac
	echo "lab-cluster: ${SCENARIO} PASS"
}

if [[ "${1:-}" == --in-job ]]; then
	[[ $# -eq 2 ]] || usage
	in_job "$2"
	exit 0
fi

# ---------------------------------------------------------------------------------------------
# On the workstation: upload, build Job, GPU Job, cleanup.
# ---------------------------------------------------------------------------------------------

DRY_RUN=0
if [[ "${1:-}" == --dry-run ]]; then
	DRY_RUN=1
	shift
fi
MODE=run
STOP_RUN=""
SCENARIO=""
case "${1:-}" in
--stop)
	[[ $# -eq 2 && "$2" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ ]] || usage
	MODE=stop
	STOP_RUN="$2"
	;;
*)
	if [[ $# -ne 1 ]] || ! valid_scenario "$1"; then
		usage
	fi
	SCENARIO="$1"
	;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE=piwi@192.168.10.203
CI_ROOT=/home/piwi/turbine-ci
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10)
NS=turbine-ci
# Short, unique and DNS-1123 safe: UTC time plus 30 random bits.
RUN_ID="$(date -u +%m%d%H%M%S)-$(printf '%08x' $(((RANDOM << 15) | RANDOM)))"
[[ $MODE == stop ]] && RUN_ID="$STOP_RUN"
JOB="turbine-lab-cluster-${RUN_ID}"
BUILD_JOB="turbine-lab-cluster-build-${RUN_ID}"
RUN_DIR="${CI_ROOT}/runs/${RUN_ID}"
# Every Job of this run, and nothing else.
SELECTOR="turbine-lab=true,turbine-lab-run=${RUN_ID}"
# How long a GPU pod may stay unschedulable before the run gives up.
UNSCHEDULABLE_LIMIT=60

say() {
	echo "lab-cluster: novanas: $*"
}

# The final line on failure.
fail() {
	echo "lab-cluster: ${SCENARIO:-stop} FAIL $1"
	exit 1
}

# Runs a local command, or prints it under --dry-run.
run() {
	if [[ $DRY_RUN -eq 1 ]]; then
		echo "+ $*"
	else
		"$@"
	fi
}

# Arguments are one remote shell command, expanded locally on purpose.
# shellcheck disable=SC2029
remote() {
	if [[ $DRY_RUN -eq 1 ]]; then
		echo "+ ssh ${SSH_OPTS[*]} ${REMOTE} '$*'"
	else
		ssh "${SSH_OPTS[@]}" "$REMOTE" "$@"
	fi
}

# Like remote, with stdin passed through (shown under --dry-run).
# shellcheck disable=SC2029
remote_stdin() {
	if [[ $DRY_RUN -eq 1 ]]; then
		echo "+ ssh ${SSH_OPTS[*]} ${REMOTE} '$*' <<'EOF'"
		cat
		echo "EOF"
	else
		ssh "${SSH_OPTS[@]}" "$REMOTE" "$@" 2> >(grep -v -e 'permission denied' >&2)
	fi
}

# kubectl on novanas; the same warning filter as scripts/lab-test.sh.
kube() {
	if [[ $DRY_RUN -eq 1 ]]; then
		remote "export KUBECTL_KUBERC=false; kubectl $*"
	else
		remote "export KUBECTL_KUBERC=false; kubectl $*" 2> >(grep -v -e 'permission denied' >&2)
	fi
}

# Deletes this run's Jobs (by its run label) and its upload, nothing else.
cleanup() {
	if [[ $DRY_RUN -eq 1 ]]; then
		kube "-n ${NS} delete job -l ${SELECTOR} --ignore-not-found"
	else
		kube "-n ${NS} delete job -l ${SELECTOR} --ignore-not-found" >/dev/null || true
	fi
	remote "rm -rf ${RUN_DIR}" || true
}

on_interrupt() {
	trap - INT TERM
	echo "lab-cluster: novanas: interrupted; deleting the jobs of run ${RUN_ID}" >&2
	cleanup
	exit 130
}

# render_job build|run: the template as the GPU-less build Job or the two-GPU scenario Job.
render_job() {
	local name="$JOB" gpus=2 role=cluster
	if [[ $1 == build ]]; then
		name="$BUILD_JOB"
		gpus=0
		role=cluster-build
	fi
	sed -e "s/__JOB__/${name}/g" -e "s/__ROLE__/${role}/g" -e "s/__RUN_ID__/${RUN_ID}/g" \
		-e "s/__GPUS__/${gpus}/g" -e "s/__PHASE__/$1/g" -e "s/__SCENARIO__/${SCENARIO}/g" \
		"${REPO_ROOT}/scripts/lab/novanas-cluster-job.yaml"
}

# wait_for_pod <job> <gpus>: until its pod runs; a GPU pod Pending on amd.com/gpu is given up.
wait_for_pod() {
	local job="$1" gpus="$2"
	if [[ $DRY_RUN -eq 1 ]]; then
		echo "+ wait until pod job-name=${job} runs (Insufficient amd.com/gpu limit ${UNSCHEDULABLE_LIMIT} s)"
		return
	fi
	local waited=0 phase="" unschedulable=""
	while :; do
		phase="$(kube "-n ${NS} get pods -l job-name=${job} -o jsonpath='{.items[*].status.phase}'" || true)"
		[[ "$phase" == Running || "$phase" == Succeeded || "$phase" == Failed ]] && return 0
		unschedulable="$(kube "-n ${NS} get pods -l job-name=${job} -o jsonpath='{.items[*].status.conditions[?(@.reason==\"Unschedulable\")].message}'" || true)"
		if [[ $gpus -gt 0 && "$unschedulable" == *"amd.com/gpu"* && $waited -ge $UNSCHEDULABLE_LIMIT ]]; then
			echo "lab-cluster: novanas: ${unschedulable}" >&2
			cleanup
			echo "lab-cluster: amd.com/gpu unavailable on novanas"
			exit 1
		fi
		if [[ $waited -ge 1800 ]]; then
			cleanup
			fail "pod of ${job} did not start within 30 min (phase: ${phase:-none})"
		fi
		sleep 5
		waited=$((waited + 5))
	done
}

# wait_for_result <job>: 0 when the Job succeeded, 1 otherwise.
wait_for_result() {
	local job="$1"
	if [[ $DRY_RUN -eq 1 ]]; then
		echo "+ wait until job ${job} succeeds or fails"
		return 0
	fi
	local succeeded="" failed=""
	while :; do
		succeeded="$(kube "-n ${NS} get job ${job} -o jsonpath='{.status.succeeded}'" || true)"
		failed="$(kube "-n ${NS} get job ${job} -o jsonpath='{.status.failed}'" || true)"
		[[ "$succeeded" == 1 || -n "$failed" ]] && break
		sleep 5
	done
	[[ "$succeeded" == 1 ]]
}

# apply_and_follow build|run <job> <gpus>: applies the rendered Job, streams its log, waits.
apply_and_follow() {
	local phase="$1" job="$2" gpus="$3"
	say "applying scripts/lab/novanas-cluster-job.yaml as ${job} (${gpus} GPU(s))"
	render_job "$phase" | remote_stdin "export KUBECTL_KUBERC=false; kubectl apply -f -" ||
		{
			cleanup
			fail "kubectl apply of ${job} failed"
		}
	wait_for_pod "$job" "$gpus"
	say "streaming pod log of ${job}"
	kube "-n ${NS} logs -f job/${job}" || echo "lab-cluster: novanas: log stream of ${job} ended with an error" >&2
	wait_for_result "$job"
}

run_scenario() {
	say "run ${RUN_ID}: job ${JOB}, scenario ${SCENARIO}, 2 GPU(s)"
	remote "mkdir -p ${RUN_DIR}/src ${CI_ROOT}/cache/slots && find ${CI_ROOT}/runs -mindepth 1 -maxdepth 1 -mmin +1440 -exec rm -rf {} +" ||
		fail "ssh to ${REMOTE} failed"
	[[ $DRY_RUN -eq 1 ]] || trap on_interrupt INT TERM
	say "syncing working tree to ${RUN_DIR}/src"
	run rsync -rlpcz --delete --exclude target/ --exclude .git/ --exclude .claude/ \
		-e "ssh ${SSH_OPTS[*]}" "${REPO_ROOT}/" "${REMOTE}:${RUN_DIR}/src/" ||
		{
			cleanup
			fail "rsync to ${REMOTE}:${RUN_DIR}/src failed"
		}
	remote "command -v kubectl >/dev/null" || fail "kubectl is not available on novanas"
	kube "create namespace ${NS} --dry-run=client -o yaml | kubectl apply -f - >/dev/null" ||
		fail "cannot create namespace ${NS}"

	# Compile without a GPU claim, then hold both R9700s only for the scenario itself.
	if ! apply_and_follow build "$BUILD_JOB" 0; then
		cleanup
		fail "build job ${BUILD_JOB} failed"
	fi
	local ok=0
	apply_and_follow run "$JOB" 2 && ok=1
	say "deleting the jobs of run ${RUN_ID}"
	cleanup
	[[ $DRY_RUN -eq 1 ]] && say "dry run: nothing contacted"
	if [[ $ok -eq 1 ]]; then
		echo "lab-cluster: ${SCENARIO} PASS"
	else
		fail "job ${JOB} failed (see the log above)"
	fi
}

stop_run() {
	say "deleting the jobs of run ${RUN_ID} in namespace ${NS}"
	kube "-n ${NS} delete job -l ${SELECTOR} --ignore-not-found --wait=true" ||
		fail "cannot delete the jobs of run ${RUN_ID}"
	remote "rm -rf ${RUN_DIR}" || fail "cannot remove ${RUN_DIR}"
	if [[ $DRY_RUN -eq 1 ]]; then say "dry run: nothing contacted"; else say "stopped"; fi
}

case "$MODE" in
run) run_scenario ;;
stop) stop_run ;;
esac
