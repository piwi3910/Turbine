#!/usr/bin/env bash
# Run one Phase 5 multi-GPU lab scenario on novanas as a k3s Job holding both R9700s.
#
#   scripts/lab-cluster.sh [--dry-run] [--bench-lock] <collbench-novanas|collbench-sweep-novanas|
#                                       collbench-hostmem-novanas|tp2-novanas|dp2-novanas|
#                                       ep2-novanas|pp2-novanas>
#   scripts/lab-cluster.sh [--dry-run] --stop <run-id>
#
# Scenarios (P5 S-9; everything runs inside the Job on loopback — no Service, no host port):
#   collbench-novanas  turbine-collbench --backend rccl --devices 0,1 --op all --max-bytes 1GiB;
#                      PASS when every row is correct and all-reduce busbw > 0 at 268,435,456 B.
#   collbench-sweep-novanas  small-message all-reduce latency under RCCL settings (diagnostics).
#   collbench-hostmem-novanas  turbine-collbench --op all, 8 B .. 1 GiB, BF16, for hostmem's
#                      kernels at every size (--route-max-bytes 1TiB), hostmem routed at its
#                      measured crossover (hostmem-auto), hostmem's copy-engine all-reduce
#                      from 64 KiB (hostmem-dma, P5 Task 32) and rccl, each per op (op +
#                      synchronize, median) and --pipelined (back to back, mean): every row
#                      printed as `hm <hostmem|hostmem-auto|hostmem-dma|rccl> <per-op|pipelined> <op>
#                      <bytes> <time_us> <busbw_gbps>`; PASS when every row of every run is
#                      correct (each rank checks its result bit for bit against the host
#                      reference backend, so both ranks hold the same bits). Keep the numbers
#                      only from a run under scripts/bench-lock.sh.
#   tp2-novanas        Llama-3.2-3B-Instruct: first a tp 1 baseline on device 0 on the standard
#                      throughput workload (512-word prompts, 256 tokens with --ignore-eos,
#                      concurrency 16, 200 requests, after a 16-request warm-up); then tp 2 in
#                      local mode (golden at concurrency 1 — strict bounds — and 16 — batched
#                      bounds, the TP accuracy bound; turbine-bench --concurrency 4 --requests 64
#                      must report requests_ok 64; the standard workload; the standard workload
#                      again with parallel.collective_backend=rccl), then OLMoE-1B-7B at
#                      tp 2 in local mode (golden c1 and c16; Task 29: an OLMoE tp 1 capture on
#                      device 0 first, the tp 2 run also compared with it — informational — and
#                      p10 position by position), then Llama in static mode (ranks
#                      0 and 1, leader 127.0.0.1:18100; golden c1 and c16), then the KV-tier
#                      legs (Task 30) on a small L0 with the step-time-drift thresholds raised:
#                      tp 1 on device 0, tp 2 local and tp 2 static (every rank its own tiers)
#                      each run the Phase 4 multi-turn load (8 sessions × 4 turns, 400 shared
#                      words, session hints; cached_tokens_ratio > 0), 48 one-off prompts and
#                      the multi-turn load replayed (cached_tokens_ratio > 0, 0 bytes in flight
#                      after idle); every leg must promote and reuse in the replay, and static
#                      must reach 0.75 × local on L1 -> L0 promotions and on replay cached
#                      tokens; golden c1 against the one-GPU capture
#                      on the tiered static group. Prints one `tp-bench <run> tok/s=…
#                      ttft_p50_ms=… itl_p50_ms=…` line per bench run and `tp-multiturn …`,
#                      `tp-tiers …` lines.
#                      2-GPU numbers are "2-GPU (GPU0 Gen5 x8 + GPU1 Gen4 x8)". Every tp 2
#                      golden takes the TP rule (user decision "P5: tensor-parallel accuracy gate
#                      against the one-GPU capture"): batched bounds at c1 and c16 against a
#                      one-GPU capture of the same model taken in the run and against the
#                      committed reference; the one-GPU leg itself is gated strictly against
#                      the committed reference. For OLMoE the capture leg is informational
#                      (`info olmoe_tp_capture`) and the committed reference is its gate. A
#                      violation fails the scenario at its end, after every leg ran.
#   dp2-novanas       Llama-3.2-3B-Instruct at tp 1 on the standard throughput workload
#                      (512-word prompts, 256 tokens with --ignore-eos, 200 requests, after a
#                      16-request warm-up): first a dp 1 baseline on device 0 at concurrency 16,
#                      then dp 2 (one replica per R9700): golden at concurrency 1 and 16, the
#                      bench at concurrency 32 (the load that fills both cards) and at 16 (the
#                      same load as dp 1), each requests_ok 200, and turbine_dp_routed_total
#                      non-zero for replica 0 and replica 1.
#   ep2-novanas        OLMoE-1B-7B (scripts/lab/phase5-novanas-ep2.yaml): first an ep 1 baseline
#                      on device 0 (golden c1 against the committed transformers reference, a
#                      capture of its greedy outputs, the teacher-forced `turbine-golden
#                      positions` report of p14, the standard throughput workload: 512-word
#                      prompts, 256 tokens with --ignore-eos, concurrency 16, 200 requests,
#                      after a 16-request warm-up); then ep 2 with tp 1 over rccl, ep 2 over
#                      hostmem and ep 2 with tp 2. The accuracy gate of every EP run is the ep 1
#                      capture of the same run: strict bounds at concurrency 1, batched at 16
#                      (user decision "P5: OLMoE golden tolerance under expert parallelism");
#                      the transformers reference is compared for information only (and p14's
#                      positions report printed). turbine_expert_rank_tokens_total must be
#                      non-zero for rank 0 and rank 1. The ep 2 × tp 2 leg includes tensor
#                      parallelism and takes the TP rule (as tp2-novanas; its capture leg is
#                      informational for OLMoE). A gate violation fails
#                      the scenario at its end, after every run. Prints one
#                      `ep-bench <run> tok/s=… ttft_p50_ms=… itl_p50_ms=…` line per bench run and
#                      the scheduler document's `expert` section. 2-GPU numbers are
#                      "2-GPU (GPU0 Gen5 x8 + GPU1 Gen4 x8)".
#   pp2-novanas        Llama-3.2-3B-Instruct (scripts/lab/phase5-novanas-pp2.yaml): first a pp 1
#                      baseline on device 0 (a capture of its greedy outputs, golden c1 against
#                      the committed reference for information, the standard workload at
#                      concurrency 16); then pp 2 (2 stages, 2 micro-batches, local mode):
#                      golden at concurrency 1 (strict) and 16 (batched) against the pp 1
#                      capture — the gate — and against the committed reference for information,
#                      the standard workload (512-word prompts, 256 tokens with --ignore-eos,
#                      200 requests, after a 32-request warm-up) at concurrency 16 and 32; then
#                      tp 2 (phase5-novanas-llama.yaml) and dp 2 (phase5-novanas-dp2.yaml) on the
#                      same workload at 16 and 32. Prints each server's parallel plan (backend,
#                      stages, reasons), the pipeline section and metrics, and one `pp-bench <run>
#                      tok/s=… ttft_p50_ms=… itl_p50_ms=…` line per bench run. A golden
#                      violation fails the scenario at its end. 2-GPU numbers are "2-GPU (GPU0
#                      Gen5 x8 + GPU1 Gen4 x8)".
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
# --bench-lock takes the exclusive benchmark lock (the one scripts/bench-lock.sh takes) after the
# build Job and holds it only while the two-GPU Job runs: use it for every run whose numbers are
# kept, instead of wrapping the whole script in scripts/bench-lock.sh (which would hold the lock
# through the build).
#
# The last line is `lab-cluster: <scenario> PASS` (exit 0) or `lab-cluster: <scenario> FAIL
# <reason>` (exit 1); usage errors exit 2.
set -euo pipefail

SCENARIOS="collbench-novanas|collbench-sweep-novanas|collbench-hostmem-novanas|tp2-novanas|dp2-novanas|ep2-novanas|pp2-novanas"

usage() {
	echo "usage: scripts/lab-cluster.sh [--dry-run] [--bench-lock] <${SCENARIOS}>" >&2
	echo "       scripts/lab-cluster.sh [--dry-run] --stop <run-id>" >&2
	exit 2
}

valid_scenario() {
	[[ "$1" =~ ^(collbench-novanas|collbench-sweep-novanas|collbench-hostmem-novanas|tp2-novanas|dp2-novanas|ep2-novanas|pp2-novanas)$ ]]
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
	# The server logs live in the Job's scratch directory: show their ends before it goes.
	local f
	for f in "${WORK:-/nonexistent}"/*.log; do
		[[ -f $f ]] || continue
		echo "lab-info: warnings, errors and circuit or pressure changes in ${f##*/}"
		sed 's/\x1b\[[0-9;]*m//g' "$f" |
			grep -E ' (WARN|ERROR) |circuit_(transition|state)|pressure_transition' |
			tail -n 20 | sed "s/^/  ${f##*/}: /" || true
		echo "lab-info: last lines of ${f##*/}"
		tail -n 40 "$f" | sed "s/^/  ${f##*/}: /"
	done
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
	# hostmem = its kernels at every size; hostmem-auto = routed at the measured crossover;
	# hostmem-dma = the copy-engine all-reduce (P5 Task 32) from 64 KiB, all-reduce only;
	# hostmem-dma-copyin = the same with the peers' chunks copied in first (A/B, not by default).
	# An uncommitted scripts/lab/collbench-hostmem.local narrows a diagnostic run: a line
	# `runs=<run>...` picks the runs, `max_bytes=<size>` caps the sizes, `devices=<list>` picks
	# the devices (e.g. `0` for a one-rank run of the copy-engine path: its copies out alone).
	local runs=(hostmem hostmem-auto hostmem-dma rccl) max=1GiB devices=0,1 line
	if [[ -f scripts/lab/collbench-hostmem.local ]]; then
		while read -r line; do
			case "$line" in
			runs=*) read -r -a runs <<<"${line#runs=}" ;;
			max_bytes=*) max="${line#max_bytes=}" ;;
			devices=*) devices="${line#devices=}" ;;
			esac
		done <scripts/lab/collbench-hostmem.local
		echo "lab-info: collbench-hostmem narrowed: runs=${runs[*]} max_bytes=${max} devices=${devices}"
	fi
	for run in "${runs[@]}"; do
		backend="${run%-auto}"
		backend="${backend%-copyin}"
		backend="${backend%-dma}"
		for mode in per-op pipelined; do
			out="${WORK}/hm-${run}-${mode}.json"
			local flags=(--backend "$backend" --devices "$devices" --op all --max-bytes "$max" --output json)
			[[ $run == hostmem ]] && flags+=(--route-max-bytes 1TiB)
			[[ $run == hostmem-dma* ]] && flags=(--backend hostmem --devices "$devices" --op all_reduce
				--min-bytes 64KiB --max-bytes "$max" --output json --route-max-bytes 1TiB
				--hostmem-dma-min-bytes 64KiB)
			[[ $run == hostmem-dma-copyin ]] && flags+=(--hostmem-dma-copy-in)
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

# collective_report <label>: the running server's collective backend and its per-call routes.
collective_report() {
	local label="$1"
	echo "tp-collective ${label} backend=$(curl -s "${URL}/turbine/v1/status" | jq -r '.parallel.backend // "?"')"
	curl -s "${URL}/metrics" | grep '^turbine_collective_route_total' | sed "s/^/tp-collective ${label} /" || true
}

scenario_tp2() {
	local llama=scripts/lab/phase5-novanas-llama.yaml olmoe=scripts/lab/phase5-novanas-olmoe.yaml
	# The tp 1 baseline: the same configuration with one rank on device 0.
	start_server "${WORK}/tp1.log" "$llama" --set parallel.tensor_parallel_size=1 \
		--set "parallel.devices=[0]"
	wait_ready "$URL" "${WORK}/tp1.log"
	tp_prefix_check tp1
	# The one-GPU capture every Llama tp 2 run is gated against (spec amendment 2026-09-28: a
	# multi-GPU run is judged against a one-GPU capture of the same commit and lab run; the
	# committed transformers reference is information only).
	capture_one_gpu llama-3.2-3b-instruct "${WORK}/llama-tp1-capture.jsonl"
	bench_ok 16 "${WORK}/tp1-warmup.json" "${STANDARD_BENCH[@]}" --concurrency 16 --requests 16
	bench_ok 200 "${WORK}/tp1-c16-bench.json" "${STANDARD_BENCH[@]}" --concurrency 16 \
		--requests 200
	stop_servers

	start_server "${WORK}/tp2-local.log" "$llama"
	wait_ready "$URL" "${WORK}/tp2-local.log"
	tp_prefix_check tp2-local
	gate_vs_capture llama-3.2-3b-instruct "${WORK}/llama-tp1-capture.jsonl" "llama tp2 local"
	bench_ok 64 "${WORK}/tp2-bench.json" --concurrency 4 --requests 64
	bench_ok 16 "${WORK}/tp2-warmup.json" "${STANDARD_BENCH[@]}" --concurrency 16 --requests 16
	bench_ok 200 "${WORK}/tp2-c16-bench.json" "${STANDARD_BENCH[@]}" --concurrency 16 \
		--requests 200
	local f
	for f in tp1-c16 tp2-c16 tp2; do
		jq -r --arg f "$f" '"tp-bench \($f) tok/s=\(.output_token_throughput) ttft_p50_ms=\(.ttft_ms.p50) itl_p50_ms=\(.itl_ms.p50) requests_ok=\(.requests_ok)"' \
			"${WORK}/${f}-bench.json"
	done
	collective_report tp2-local
	stop_servers
	tp2_variant_leg "$llama"

	if [[ $TP2_SKIP_REST -eq 1 ]]; then
		echo "lab-info: tp2 variant: skip-rest (the rccl and static legs do not run)"
	else
		# The same tp 2 run over RCCL alone (the default `auto` picks hostmem in local mode), so the
		# collective's share of a throughput change is visible (plan Task 29).
		start_server "${WORK}/tp2-rccl.log" "$llama" --set parallel.collective_backend=rccl
		wait_ready "$URL" "${WORK}/tp2-rccl.log"
		bench_ok 16 "${WORK}/tp2-rccl-warmup.json" "${STANDARD_BENCH[@]}" --concurrency 16 \
			--requests 16
		bench_ok 200 "${WORK}/tp2-rccl-c16-bench.json" "${STANDARD_BENCH[@]}" --concurrency 16 \
			--requests 200
		jq -r '"tp-bench tp2-rccl-c16 tok/s=\(.output_token_throughput) ttft_p50_ms=\(.ttft_ms.p50) itl_p50_ms=\(.itl_ms.p50) requests_ok=\(.requests_ok)"' \
			"${WORK}/tp2-rccl-c16-bench.json"
		stop_servers
	fi

	# OLMoE: a one-GPU capture first (tp 1 on device 0), the reference the tp 2 run is also
	# compared with, position by position on p10 (plan Task 29: OLMoE tp 2 p10).
	local slug=olmoe-1b-7b-0125-instruct
	start_server "${WORK}/tp1-olmoe.log" "$olmoe" --set parallel.tensor_parallel_size=1 \
		--set "parallel.devices=[0]"
	wait_ready "$URL" "${WORK}/tp1-olmoe.log"
	capture_one_gpu "$slug" "${WORK}/olmoe-tp1-capture.jsonl"
	positions_p "$slug" olmoe-tp1 p10
	stop_servers

	start_server "${WORK}/tp2-olmoe.log" "$olmoe" ${TP2_VARIANT[@]+"${TP2_VARIANT[@]}"}
	wait_ready "$URL" "${WORK}/tp2-olmoe.log"
	gate_vs_capture "$slug" "${WORK}/olmoe-tp1-capture.jsonl" "olmoe tp2 local"
	positions_p "$slug" olmoe-tp2 p10
	echo "lab-step: turbine-golden positions olmoe-tp2-vs-1gpu p10"
	"${BIN}/turbine-golden" positions --url "$URL" --reference "${WORK}/olmoe-tp1-capture.jsonl" \
		--tolerance "tests/golden/${slug}/tolerance.json" --prompt-id p10 |
		sed "s/^/positions olmoe-tp2-vs-1gpu /" ||
		echo "lab-info: positions olmoe-tp2-vs-1gpu failed"
	collective_report tp2-olmoe
	stop_servers

	if [[ $TP2_SKIP_REST -eq 1 ]]; then
		if [[ ${#GATE_FAILED[@]} -gt 0 ]]; then
			job_fail "outside tolerance: ${GATE_FAILED[*]}"
		fi
		return 0
	fi
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
	gate_vs_capture llama-3.2-3b-instruct "${WORK}/llama-tp1-capture.jsonl" "llama tp2 static"
	collective_report tp2-static
	stop_servers
	tiers_legs "$llama" "${static[@]}"
	if [[ ${#GATE_FAILED[@]} -gt 0 ]]; then
		job_fail "outside tolerance: ${GATE_FAILED[*]}"
	fi
}

# The KV-tier settings of the tiers legs (plan Task 30): a small L0 (1 GiB per tp 2 rank, 2 GiB
# at tp 1: 146 blocks either way, as a 73-block tp 1 L0 reads the flood as SURVIVAL;
# model.max_seq_len 4096, one sequence of which needs 32; 16 running requests), L1 2 GiB (one
# 1 GiB slab per tp 2 rank), L2 4 GiB under ${WORK}/kv-<leg>, and the step-time-drift thresholds of the pressure
# controller and of the circuit breaker raised (pressure 100/200/300, circuit 100/200): tp 2
# prefill under the flood below drifts past the RED threshold (3.0), where Phase 4 plans no
# promotion (`l0_pressure`), and past the circuit's 4.0 without the throttle. The standard legs
# keep the defaults.
tiers_settings() {
	# tp 1 holds whole blocks (14 MiB), a tp 2 rank half of each: the same 146 blocks per tier.
	local gpu=1GiB
	[[ $1 == tp1 ]] && gpu=2GiB
	TIERS_SETTINGS=(--set "kv.gpu.max_bytes=${gpu}" --set model.max_seq_len=4096
		--set scheduler.max_running_requests=16
		--set kv.cpu.enabled=true --set kv.cpu.max_bytes=2GiB --set kv.nvme.enabled=true
		--set "kv.nvme.path=${WORK}/kv-$1" --set kv.nvme.max_bytes=4GiB
		--set "reliability.pressure.thresholds.step_time_drift=[100.0, 200.0, 300.0, null]"
		--set reliability.circuit.latency_drift_degraded=100.0
		--set reliability.circuit.latency_drift_open=200.0)
	echo "lab-info: tiers leg $1: step-time-drift thresholds raised (pressure 100/200/300, circuit 100/200)"
}

# kv_counter <series>: the leader's value of one Prometheus series (0 when absent).
kv_counter() {
	curl -s "${URL}/metrics" | awk -v s="$1" '$1 == s { v = $2 } END { print v + 0 }'
}

# kv_report <label>: the leader's KV document (tiers, hit rate, transfers) and plan, lookup,
# eviction and drop counters, for the record.
kv_report() {
	curl -s "${URL}/turbine/v1/kv" | jq -c . | cut -c1-3000 | sed "s/^/tp-kvdoc $1 /" || true
	curl -s "${URL}/metrics" |
		grep -E '^turbine_kv_(plans|lookups|evictions|drops|recompute_tokens)_total' |
		sed "s/^/tp-kv $1 /" || true
}

# kv_cached_tokens: the leader's cached prompt tokens (the KV document's hit-rate window).
kv_cached_tokens() {
	curl -s "${URL}/turbine/v1/kv" | jq -r '.hit_rate.cached_tokens // 0'
}

# tiers_workload <leg>: the tiers legs' workload on the running server (plan Task 30):
# the Phase 4 multi-turn run (8 sessions × 4 turns, 400 shared words, session hints, 4 at once),
# 48 one-off prompts that push its blocks out of the 146-block L0 (capacity demotion copied them
# to L1), then the multi-turn run replayed (same seed, same prompts: its prefixes come back
# from L1, or L2), and after a short idle nothing may stay in flight. Prints one
# `tp-tiers <leg> …` line; the parity check compares the legs' lines.
tiers_workload() {
	local leg="$1"
	local out="${WORK}/tiers-${leg}-multiturn.json" replay="${WORK}/tiers-${leg}-replay.json"
	bench_ok 32 "$out" --profile multi-turn --sessions 8 --turns 4 --shared-prefix-words 400 \
		--session-hints --concurrency 4
	jq -r --arg l "$leg" '"tp-multiturn \($l) cached_tokens_ratio=\(.cached_tokens_ratio) ttft_first_p50_ms=\(.ttft_ms_first_turn.p50 // "?") ttft_later_p50_ms=\(.ttft_ms_later_turns.p50 // "?") requests_ok=\(.requests_ok)"' \
		"$out"
	local series=(
		'turbine_kv_demotions_total{from="l0",to="l1"}' 'turbine_kv_demotions_total{from="l1",to="l2"}'
		'turbine_kv_promotions_total{from="l1",to="l0"}' 'turbine_kv_promotions_total{from="l2",to="l1"}'
		'turbine_kv_promotions_total{from="l2",to="l0"}'
	)
	local before=() i
	for i in "${!series[@]}"; do before[i]="$(kv_counter "${series[i]}")"; done
	kv_report "${leg}-before-flood"
	bench_ok 48 "${WORK}/tiers-${leg}-flood.json" --prompt-words 512 --max-tokens 64 \
		--concurrency 4 --requests 48 --seed 7
	kv_report "${leg}-after-flood"
	local cached_before
	cached_before="$(kv_cached_tokens)"
	bench_ok 32 "$replay" --profile multi-turn --sessions 8 --turns 4 --shared-prefix-words 400 \
		--session-hints --concurrency 4
	local cached_replay=$(($(kv_cached_tokens) - cached_before))
	sleep 3
	kv_report "${leg}-after-replay"
	local inflight
	inflight="$(curl -s "${URL}/turbine/v1/kv" | jq -r '.transfers.inflight_bytes // -1')"
	local delta=()
	for i in "${!series[@]}"; do delta[i]=$(($(kv_counter "${series[i]}") - before[i])); done
	local promoted=$((delta[2] + delta[3] + delta[4]))
	echo "tp-tiers ${leg} demoted_l0_l1=${delta[0]} demoted_l1_l2=${delta[1]} promoted_l1_l0=${delta[2]} promoted_l2_l1=${delta[3]} promoted_l2_l0=${delta[4]} promoted=${promoted} replay_cached_tokens=${cached_replay} replay_cached_ratio=$(jq -r '.cached_tokens_ratio' "$replay") inflight_after_idle=${inflight}"
	jq -e '(.cached_tokens_ratio // 0) > 0' "$out" >/dev/null ||
		GATE_FAILED+=("tiers ${leg}: first multi-turn run cached_tokens_ratio not > 0")
	jq -e '(.cached_tokens_ratio // 0) > 0' "$replay" >/dev/null ||
		GATE_FAILED+=("tiers ${leg}: replay cached_tokens_ratio not > 0")
	[[ $inflight == 0 ]] || GATE_FAILED+=("tiers ${leg}: ${inflight} bytes still in flight after idle")
	TIERS_PROMOTED[$leg]=$promoted
	TIERS_PROMOTED_L1[$leg]=${delta[2]}
	TIERS_REPLAY_CACHED[$leg]=$cached_replay
}

declare -A TIERS_PROMOTED=() TIERS_PROMOTED_L1=() TIERS_REPLAY_CACHED=()

# tiers_legs <config> <static args>...: the tiers workload on tp 1 (device 0), tp 2 local and
# tp 2 static (plan Task 30 and its follow-up): every leg must promote (> 0) and reuse in the
# replay (> 0), else the scenario does not exercise the tiers; static must reach 0.75 × local on
# L1 -> L0 promotions and on replay cached tokens (one-sided, user decision "Task 30 parity":
# how capacity demotion and the flood's allocations interleave with the 4 sessions moves both
# by more than a block between runs and modes — run 0928111413-39b46c20: tp 1 27 promotions and
# 22,400 replay tokens, local 42 and 28,032, static 35 and 35,072); then golden c1 against the
# one-GPU capture on the tiered static group.
tiers_legs() {
	local config="$1"
	shift
	tiers_settings tp1
	start_server "${WORK}/tiers-tp1.log" "$config" "${TIERS_SETTINGS[@]}" \
		--set parallel.tensor_parallel_size=1 --set "parallel.devices=[0]"
	wait_ready "$URL" "${WORK}/tiers-tp1.log"
	tiers_workload tp1
	stop_servers

	tiers_settings local
	start_server "${WORK}/tiers-local.log" "$config" "${TIERS_SETTINGS[@]}"
	wait_ready "$URL" "${WORK}/tiers-local.log"
	tiers_workload local
	stop_servers

	tiers_settings static
	start_server "${WORK}/tiers-static-rank1.log" "$config" "$@" "${TIERS_SETTINGS[@]}" \
		--set parallel.ranks.rank=1 --set "parallel.ranks.local_devices=[1]" \
		--set server.listen=127.0.0.1:18001
	start_server "${WORK}/tiers-static-rank0.log" "$config" "$@" "${TIERS_SETTINGS[@]}" \
		--set parallel.ranks.rank=0 --set "parallel.ranks.local_devices=[0]"
	wait_ready "$URL" "${WORK}/tiers-static-rank0.log"
	tiers_workload static
	# A TP leg: batched bounds against the one-GPU capture (user decision, follow-up (a)).
	echo "lab-step: golden llama tp2 static tiers c1 vs 1 GPU (batched bounds)"
	"${BIN}/turbine-golden" compare --url "$URL" \
		--reference "${WORK}/llama-tp1-capture.jsonl" \
		--tolerance tests/golden/llama-3.2-3b-instruct/tolerance.json \
		--prompts tests/golden/prompts.jsonl --concurrency 1 --batched-bounds ||
		GATE_FAILED+=("golden llama tp2 static tiers c1 vs 1 GPU")
	stop_servers

	local leg
	for leg in tp1 local static; do
		((TIERS_PROMOTED[$leg] > 0)) ||
			GATE_FAILED+=("tiers ${leg}: no promotion (the scenario does not exercise the tiers)")
		((TIERS_REPLAY_CACHED[$leg] > 0)) ||
			GATE_FAILED+=("tiers ${leg}: no reuse in the replay")
	done
	local sp=${TIERS_PROMOTED_L1[static]} lp=${TIERS_PROMOTED_L1[local]}
	local sc=${TIERS_REPLAY_CACHED[static]} lc=${TIERS_REPLAY_CACHED[local]}
	echo "tp-tiers parity static/local promoted_l1_l0=${sp}/${lp} replay_cached_tokens=${sc}/${lc} floor=0.75"
	((4 * sp >= 3 * lp)) ||
		GATE_FAILED+=("tiers static promotes ${sp} blocks L1 -> L0, below 0.75 x local's ${lp}")
	((4 * sc >= 3 * lc)) ||
		GATE_FAILED+=("tiers static reuses ${sc} tokens in the replay, below 0.75 x local's ${lc}")
}

# The tp 2 A/B variant (P5 Task 32): an uncommitted scripts/lab/tp2-variant.local in the uploaded
# tree, one `set <dotted.key>=<value>` per line (and optionally `skip-rest`), adds a leg after
# tp2-local with those settings — golden c1 / c16 against the one-GPU capture, the standard
# workload, a `tp-bench tp2-variant-c16 …` line — applies them to the OLMoE tp 2 leg, and with
# skip-rest leaves out the rccl and static legs. Without the file nothing changes.
TP2_VARIANT=()
TP2_SKIP_REST=0
read_tp2_variant() {
	local f=scripts/lab/tp2-variant.local line
	[[ -f $f ]] || return 0
	while read -r line; do
		case "$line" in
		set\ *) TP2_VARIANT+=(--set "${line#set }") ;;
		skip-rest) TP2_SKIP_REST=1 ;;
		esac
	done <"$f"
	echo "lab-info: tp2 variant: ${TP2_VARIANT[*]:-none} skip_rest=${TP2_SKIP_REST}"
}

# tp2_variant_leg <config>: the variant leg (above), when a variant is set.
tp2_variant_leg() {
	[[ ${#TP2_VARIANT[@]} -gt 0 ]] || return 0
	local config="$1"
	start_server "${WORK}/tp2-variant.log" "$config" "${TP2_VARIANT[@]}"
	wait_ready "$URL" "${WORK}/tp2-variant.log"
	gate_vs_capture llama-3.2-3b-instruct "${WORK}/llama-tp1-capture.jsonl" "llama tp2 variant"
	bench_ok 16 "${WORK}/tp2-variant-warmup.json" "${STANDARD_BENCH[@]}" --concurrency 16 \
		--requests 16
	bench_ok 200 "${WORK}/tp2-variant-c16-bench.json" "${STANDARD_BENCH[@]}" --concurrency 16 \
		--requests 200
	jq -r '"tp-bench tp2-variant-c16 tok/s=\(.output_token_throughput) ttft_p50_ms=\(.ttft_ms.p50) itl_p50_ms=\(.itl_ms.p50) requests_ok=\(.requests_ok)"' \
		"${WORK}/tp2-variant-c16-bench.json"
	curl -s "${URL}/metrics" | grep -E '^turbine_decode_graph' | sed 's/^/tp-graphs tp2-variant /' || true
	collective_report tp2-variant
	grep -h -E 'decode_graphs|collective_init' "${WORK}/tp2-variant.log" | head -n 6 |
		sed 's/^/lab-info: /' || true
	stop_servers
}

# capture_one_gpu <slug> <out>: the one-GPU server's golden c1 against the committed reference
# (the one-GPU gate) and a capture of its greedy outputs, the reference of the multi-GPU runs.
capture_one_gpu() {
	local slug="$1" out="$2"
	golden_gate "$slug" "${slug} 1 GPU c1"
	echo "lab-step: turbine-golden capture (${slug} 1 GPU)"
	"${BIN}/turbine-golden" capture --url "$URL" --prompts tests/golden/prompts.jsonl \
		--out "$out" || job_fail "turbine-golden capture failed"
}

# gate_vs_capture <slug> <capture> <label>: the tensor-parallel gate (user decision "P5:
# tensor-parallel accuracy gate against the one-GPU capture", A): against the one-GPU capture
# with the batched bounds at c1 and c16, and against the committed transformers reference with
# the batched bounds at c1 and c16 too (follow-up (a)). A violation is recorded in GATE_FAILED;
# the scenario goes on and fails at its end. For OLMoE (follow-up "HF only for OLMoE TP") the
# capture leg is informational: a violation prints `info olmoe_tp_capture …` and does not fail.
gate_vs_capture() {
	local slug="$1" capture="$2" label="$3" c
	local tol=(--tolerance "tests/golden/${slug}/tolerance.json" --prompts tests/golden/prompts.jsonl)
	for c in 1 16; do
		echo "lab-step: golden ${label} c${c} vs 1 GPU (batched bounds)"
		if ! "${BIN}/turbine-golden" compare --url "$URL" --reference "$capture" "${tol[@]}" \
			--concurrency "$c" --batched-bounds; then
			if [[ $slug == olmoe-* ]]; then
				echo "info olmoe_tp_capture golden ${label} c${c} vs 1 GPU outside the batched bounds"
			else
				GATE_FAILED+=("golden ${label} c${c} vs 1 GPU")
			fi
		fi
		golden_gate "$slug" "${label} c${c} vs HF" --concurrency "$c" --batched-bounds
	done
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

# golden_gate <slug> <label> [compare args]...: like golden, but a violation is recorded in
# GATE_FAILED and the scenario goes on (its later checks and benches still run); the scenario
# fails at its end. golden_info: the same, never failing (diagnostics).
GATE_FAILED=()
golden_gate() {
	local slug="$1" label="$2"
	shift 2
	echo "lab-step: golden ${label} $*"
	"${BIN}/turbine-golden" compare --url "$URL" \
		--reference "tests/golden/${slug}/reference.jsonl" "$@" || GATE_FAILED+=("golden ${label}")
}
golden_info() {
	local label="$1"
	shift
	echo "lab-step: golden (informational) ${label} $*"
	"${BIN}/turbine-golden" compare --url "$URL" "$@" || echo "lab-info: golden ${label} outside tolerance"
}

# positions_p <slug> <label> <prompt id>: turbine-golden positions, teacher-forced on the committed
# reference (every position has the reference's history), printed with a prefix per line.
positions_p() {
	local slug="$1" label="$2" id="$3"
	echo "lab-step: turbine-golden positions ${label} ${id}"
	"${BIN}/turbine-golden" positions --url "$URL" --reference "tests/golden/${slug}/reference.jsonl" \
		--prompt-id "$id" | sed "s/^/positions ${label} /" || echo "lab-info: positions ${label} failed"
}

scenario_ep2() {
	local ep2=scripts/lab/phase5-novanas-ep2.yaml slug=olmoe-1b-7b-0125-instruct
	local tol=(--tolerance "tests/golden/${slug}/tolerance.json" --prompts tests/golden/prompts.jsonl)
	# The multi-GPU accuracy gate (user decision "P5: OLMoE golden tolerance under expert
	# parallelism", A then C): a strict compare at c1 and a batched one at c16 against the one-GPU
	# capture taken in this run; the committed transformers reference is information only.
	# golden_vs_one_gpu <label> [known_fail <task>]: with known_fail a violation is printed as
	# `known_fail <task> golden <label> …` and does not fail the scenario (it is tracked there).
	golden_vs_one_gpu() {
		local label="$1" known="${3:-}" c
		echo "lab-step: golden ${label} vs 1 GPU c1 and c16"
		for c in 1 16; do
			if ! "${BIN}/turbine-golden" compare --url "$URL" \
				--reference "${WORK}/ep1-capture.jsonl" "${tol[@]}" --concurrency "$c"; then
				if [[ -n $known ]]; then
					echo "known_fail ${known} golden ${label} c${c} vs 1 GPU"
				else
					GATE_FAILED+=("golden ${label} c${c} vs 1 GPU")
				fi
			fi
		done
	}
	# The ep 1 baseline: the same configuration with one device. Its golden c1 and a capture of
	# its greedy outputs, the one-device reference the EP runs are also compared with.
	start_server "${WORK}/ep1.log" "$ep2" --set parallel.expert_parallel_size=1 \
		--set "parallel.devices=[0]"
	wait_ready "$URL" "${WORK}/ep1.log"
	golden_info "olmoe ep1 c1" --reference "tests/golden/${slug}/reference.jsonl"
	echo "lab-step: turbine-golden capture (ep1)"
	"${BIN}/turbine-golden" capture --url "$URL" --prompts tests/golden/prompts.jsonl \
		--out "${WORK}/ep1-capture.jsonl" || job_fail "turbine-golden capture failed"
	positions_p "$slug" ep1 p14
	bench_ok 16 "${WORK}/ep1-warmup.json" "${STANDARD_BENCH[@]}" --concurrency 16 --requests 16
	bench_ok 200 "${WORK}/ep1-c16-bench.json" "${STANDARD_BENCH[@]}" --concurrency 16 \
		--requests 200
	stop_servers

	start_server "${WORK}/ep2.log" "$ep2"
	wait_ready "$URL" "${WORK}/ep2.log"
	golden_vs_one_gpu "olmoe ep2"
	golden_info "olmoe ep2 c1 vs HF" --reference "tests/golden/${slug}/reference.jsonl"
	golden_info "olmoe ep2 c16 vs HF" --reference "tests/golden/${slug}/reference.jsonl" --concurrency 16
	positions_p "$slug" ep2 p14
	ep_counts ep2
	bench_ok 16 "${WORK}/ep2-warmup.json" "${STANDARD_BENCH[@]}" --concurrency 16 --requests 16
	bench_ok 200 "${WORK}/ep2-c16-bench.json" "${STANDARD_BENCH[@]}" --concurrency 16 \
		--requests 200
	ep_counts ep2-after-bench
	stop_servers

	start_server "${WORK}/ep2-hostmem.log" "$ep2" --set parallel.collective_backend=hostmem
	wait_ready "$URL" "${WORK}/ep2-hostmem.log"
	golden_vs_one_gpu "olmoe ep2 hostmem"
	golden_info "olmoe ep2 hostmem c1 vs HF" --reference "tests/golden/${slug}/reference.jsonl"
	bench_ok 16 "${WORK}/ep2-hostmem-warmup.json" "${STANDARD_BENCH[@]}" --concurrency 16 \
		--requests 16
	bench_ok 200 "${WORK}/ep2-hostmem-c16-bench.json" "${STANDARD_BENCH[@]}" --concurrency 16 \
		--requests 200
	ep_counts ep2-hostmem
	stop_servers

	start_server "${WORK}/ep2-tp2.log" "$ep2" --set parallel.tensor_parallel_size=2
	wait_ready "$URL" "${WORK}/ep2-tp2.log"
	# ep 2 Ã tp 2 includes tensor parallelism: the TP rule (batched bounds against the ep 1
	# capture, the golden rule against the transformers reference).
	gate_vs_capture "$slug" "${WORK}/ep1-capture.jsonl" "olmoe ep2 tp2"
	ep_counts ep2-tp2
	stop_servers

	local f
	for f in ep1-c16 ep2-c16 ep2-hostmem-c16; do
		jq -r --arg f "$f" '"ep-bench \($f) tok/s=\(.output_token_throughput) ttft_p50_ms=\(.ttft_ms.p50) itl_p50_ms=\(.itl_ms.p50) requests_ok=\(.requests_ok)"' \
			"${WORK}/${f}-bench.json"
	done
	if [[ ${#GATE_FAILED[@]} -gt 0 ]]; then
		job_fail "outside tolerance: ${GATE_FAILED[*]}"
	fi
}

# pp_parallel <label>: the running server's parallel plan (backend, pp, stages, reasons) and
# pipeline section (stage busy ratios, micro-batches) plus the pipeline metrics.
pp_parallel() {
	local label="$1"
	echo "pp-parallel ${label} $(curl -s "${URL}/turbine/v1/status" | jq -c '.parallel | {tp, dp, pp, backend, plan_reasons, stages: .groups[0].stages}')"
	echo "pp-pipeline ${label} $(curl -s "${URL}/turbine/v1/scheduler" | jq -c '.["0"].pipeline')"
	curl -s "${URL}/metrics" | grep -E '^turbine_pipeline_(bubble_ratio|stage_duration_seconds_(sum|count))' |
		sed "s/^/pp-metrics ${label} /" || true
}

scenario_pp2() {
	local pp2=scripts/lab/phase5-novanas-pp2.yaml slug=llama-3.2-3b-instruct f
	local tol=(--tolerance "tests/golden/${slug}/tolerance.json" --prompts tests/golden/prompts.jsonl)
	# The pp 1 baseline: the same configuration on device 0 alone. A capture of its greedy
	# outputs is the reference pp 2 is gated against (user decision 2026-09-28, "A then C":
	# multi-GPU runs against a one-GPU capture of the same model and commit, same run).
	start_server "${WORK}/pp1.log" "$pp2" --set parallel.pipeline_parallel_size=1 \
		--set "parallel.devices=[0]"
	wait_ready "$URL" "${WORK}/pp1.log"
	golden_info "llama pp1 c1" --reference "tests/golden/${slug}/reference.jsonl"
	echo "lab-step: turbine-golden capture (pp1)"
	"${BIN}/turbine-golden" capture --url "$URL" --prompts tests/golden/prompts.jsonl \
		--out "${WORK}/pp1-capture.jsonl" || job_fail "turbine-golden capture failed"
	bench_ok 16 "${WORK}/pp1-warmup.json" "${STANDARD_BENCH[@]}" --concurrency 16 --requests 16
	bench_ok 200 "${WORK}/pp1-c16-bench.json" "${STANDARD_BENCH[@]}" --concurrency 16 \
		--requests 200
	stop_servers

	start_server "${WORK}/pp2.log" "$pp2"
	wait_ready "$URL" "${WORK}/pp2.log"
	grep -E 'topology_link|pipeline_stages|pp_pool_agreed|pp_pipeline_ready|decode_graphs_unavailable' \
		"${WORK}/pp2.log" | tail -n 12 || true
	pp_parallel pp2-start
	echo "lab-step: golden llama pp2 c1 vs pp1 (the gate)"
	"${BIN}/turbine-golden" compare --url "$URL" --reference "${WORK}/pp1-capture.jsonl" \
		"${tol[@]}" || GATE_FAILED+=("golden llama pp2 c1 vs pp1")
	echo "lab-step: golden llama pp2 c16 vs pp1 (the gate)"
	"${BIN}/turbine-golden" compare --url "$URL" --reference "${WORK}/pp1-capture.jsonl" \
		"${tol[@]}" --concurrency 16 || GATE_FAILED+=("golden llama pp2 c16 vs pp1")
	if [[ ${#GATE_FAILED[@]} -gt 0 ]]; then
		sed 's/\x1b\[[0-9;]*m//g' "${WORK}/pp2.log" | grep -iE 'error|fail|abort' | head -n 30 || true
		job_fail "outside tolerance: ${GATE_FAILED[*]}"
	fi
	golden_info "llama pp2 c1" --reference "tests/golden/${slug}/reference.jsonl"
	golden_info "llama pp2 c16" --reference "tests/golden/${slug}/reference.jsonl" \
		--concurrency 16
	bench_ok 32 "${WORK}/pp2-warmup.json" "${STANDARD_BENCH[@]}" --concurrency 32 --requests 32
	bench_ok 200 "${WORK}/pp2-c16-bench.json" "${STANDARD_BENCH[@]}" --concurrency 16 \
		--requests 200
	bench_ok 200 "${WORK}/pp2-c32-bench.json" "${STANDARD_BENCH[@]}" --concurrency 32 \
		--requests 200
	pp_parallel pp2-after-bench
	stop_servers

	# tp 2 and dp 2 on the same workload, in the same Job.
	start_server "${WORK}/tp2.log" scripts/lab/phase5-novanas-llama.yaml
	wait_ready "$URL" "${WORK}/tp2.log"
	bench_ok 32 "${WORK}/tp2-warmup.json" "${STANDARD_BENCH[@]}" --concurrency 32 --requests 32
	bench_ok 200 "${WORK}/tp2-c16-bench.json" "${STANDARD_BENCH[@]}" --concurrency 16 \
		--requests 200
	bench_ok 200 "${WORK}/tp2-c32-bench.json" "${STANDARD_BENCH[@]}" --concurrency 32 \
		--requests 200
	collective_report tp2
	stop_servers

	start_server "${WORK}/dp2.log" scripts/lab/phase5-novanas-dp2.yaml
	wait_ready "$URL" "${WORK}/dp2.log"
	bench_ok 32 "${WORK}/dp2-warmup.json" "${STANDARD_BENCH[@]}" --concurrency 32 --requests 32
	bench_ok 200 "${WORK}/dp2-c16-bench.json" "${STANDARD_BENCH[@]}" --concurrency 16 \
		--requests 200
	bench_ok 200 "${WORK}/dp2-c32-bench.json" "${STANDARD_BENCH[@]}" --concurrency 32 \
		--requests 200
	echo "pp-parallel dp2 $(curl -s "${URL}/turbine/v1/status" | jq -c '.parallel | {tp, dp, backend, plan_reasons}')"
	stop_servers

	for f in pp1-c16 pp2-c16 pp2-c32 tp2-c16 tp2-c32 dp2-c16 dp2-c32; do
		jq -r --arg f "$f" '"pp-bench \($f) tok/s=\(.output_token_throughput) ttft_p50_ms=\(.ttft_ms.p50) ttft_p99_ms=\(.ttft_ms.p99) itl_p50_ms=\(.itl_ms.p50) itl_p99_ms=\(.itl_ms.p99) requests_ok=\(.requests_ok)"' \
			"${WORK}/${f}-bench.json"
	done
	if [[ ${#GATE_FAILED[@]} -gt 0 ]]; then
		job_fail "outside tolerance: ${GATE_FAILED[*]}"
	fi
}

in_job() {
	SCENARIO="$1"
	valid_scenario "$SCENARIO" || usage
	: "${CARGO_TARGET_DIR:?the Job sets CARGO_TARGET_DIR}"
	BIN="${CARGO_TARGET_DIR}/release"
	WORK="$(mktemp -d)"
	trap 'stop_servers' EXIT
	read_tp2_variant
	case "$SCENARIO" in
	collbench-novanas) scenario_collbench ;;
	collbench-sweep-novanas) scenario_collbench_sweep ;;
	collbench-hostmem-novanas) scenario_collbench_hostmem ;;
	tp2-novanas) scenario_tp2 ;;
	dp2-novanas) scenario_dp2 ;;
	ep2-novanas) scenario_ep2 ;;
	pp2-novanas) scenario_pp2 ;;
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
BENCH_LOCK=0
while [[ "${1:-}" == --dry-run || "${1:-}" == --bench-lock ]]; do
	[[ $1 == --dry-run ]] && DRY_RUN=1
	[[ $1 == --bench-lock ]] && BENCH_LOCK=1
	shift
done
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
# How long it may wait behind other Turbine lab Jobs (our own queue) before giving up.
OWN_QUEUE_LIMIT="${TURBINE_LAB_QUEUE_LIMIT:-10800}"

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
		# Waiting behind another Turbine lab Job (label turbine-lab=true) is a queue, not a
		# failure: name the holders every 5 min and wait up to OWN_QUEUE_LIMIT. Anything else
		# holding amd.com/gpu gives up after UNSCHEDULABLE_LIMIT (never evicted).
		if [[ $gpus -gt 0 && "$unschedulable" == *"amd.com/gpu"* ]]; then
			local ours=""
			ours="$(kube "-n ${NS} get pods -l turbine-lab=true --field-selector=status.phase=Running -o jsonpath='{range .items[*]}{.metadata.labels.job-name} {end}'" || true)"
			ours="${ours//${job}/}"
			if [[ -n "${ours// /}" && $waited -lt $OWN_QUEUE_LIMIT ]]; then
				((waited % 300 == 0)) && say "queued behind our lab Job(s):${ours} (waited ${waited} s)"
				sleep 5
				waited=$((waited + 5))
				continue
			fi
		fi
		if [[ $gpus -gt 0 && "$unschedulable" == *"amd.com/gpu"* && $waited -ge $UNSCHEDULABLE_LIMIT ]]; then
			echo "lab-cluster: novanas: ${unschedulable}" >&2
			cleanup
			echo "lab-cluster: amd.com/gpu unavailable on novanas"
			exit 1
		fi
		# Unschedulable for another reason (a node taint such as disk pressure): print the
		# scheduler's own message rather than waiting out the 30 min.
		if [[ -n "$unschedulable" && "$unschedulable" != *"amd.com/gpu"* && $waited -ge $UNSCHEDULABLE_LIMIT ]]; then
			cleanup
			fail "pod of ${job} unschedulable: ${unschedulable}"
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

# take_bench_lock / release_bench_lock: the exclusive flock on ${CI_ROOT}/bench.lock that
# scripts/bench-lock.sh takes (the same file and mode), held from before the GPU Job is applied
# until it ends, so builds never hold it.
LOCK_DIR=""
LOCK_PID=""
take_bench_lock() {
	if [[ $DRY_RUN -eq 1 ]]; then
		echo "+ ssh ${SSH_OPTS[*]} ${REMOTE} 'flock -x ${CI_ROOT}/bench.lock …' (held for the GPU Job)"
		return
	fi
	LOCK_DIR="$(mktemp -d)"
	mkfifo "${LOCK_DIR}/in" "${LOCK_DIR}/out"
	# shellcheck disable=SC2029 # CI_ROOT is expanded here on purpose
	ssh "${SSH_OPTS[@]}" "$REMOTE" \
		"flock -x ${CI_ROOT}/bench.lock sh -c 'echo locked; cat >/dev/null'" \
		<"${LOCK_DIR}/in" >"${LOCK_DIR}/out" &
	LOCK_PID=$!
	exec 7>"${LOCK_DIR}/in"
	say "waiting for the benchmark lock"
	local state=""
	read -r state <"${LOCK_DIR}/out" || true
	if [[ "$state" != locked ]]; then
		cleanup
		fail "could not take the benchmark lock"
	fi
	say "holding the benchmark lock"
}
release_bench_lock() {
	[[ $DRY_RUN -eq 1 || -z $LOCK_PID ]] && return 0
	exec 7>&-
	wait "$LOCK_PID" || true
	rm -rf "$LOCK_DIR"
	LOCK_PID=""
	say "released the benchmark lock"
}

# wait_for_free_gpus: until no GPU-holding Turbine lab Job (roles test, cluster, serve) is running
# or pending, naming them every 5 min, up to OWN_QUEUE_LIMIT; our builds (GPU-less) don't count.
wait_for_free_gpus() {
	if [[ $DRY_RUN -eq 1 ]]; then
		echo "+ wait until no turbine-lab test/cluster/serve pod is running or pending"
		return
	fi
	local waited=0 holders=""
	while :; do
		holders="$(kube "-n ${NS} get pods -l 'turbine-lab=true,turbine-lab-role in (test,cluster,serve)' --field-selector=status.phase!=Succeeded,status.phase!=Failed -o jsonpath='{range .items[*]}{.metadata.labels.job-name} {end}'" || true)"
		[[ -z "${holders// /}" ]] && return 0
		if [[ $waited -ge $OWN_QUEUE_LIMIT ]]; then
			cleanup
			fail "the GPUs stayed held by our lab Job(s) ${holders} for ${waited} s"
		fi
		((waited % 300 == 0)) && say "waiting for the GPUs (held by our lab Job(s): ${holders}) before taking the benchmark lock"
		sleep 5
		waited=$((waited + 5))
	done
}

run_scenario() {
	say "run ${RUN_ID}: job ${JOB}, scenario ${SCENARIO}, 2 GPU(s)"
	remote "mkdir -p ${RUN_DIR}/src ${CI_ROOT}/cache/slots && find ${CI_ROOT}/runs -mindepth 1 -maxdepth 1 -mmin +1440 -exec rm -rf {} +" ||
		fail "ssh to ${REMOTE} failed"
	[[ $DRY_RUN -eq 1 ]] || trap on_interrupt INT TERM
	# Stale remote/agent-*/target caches of removed worktrees (scripts/lab-prune.sh; never fails).
	local prune=(--host "$REMOTE")
	[[ $DRY_RUN -eq 1 ]] && prune+=(--dry-run)
	"${REPO_ROOT}/scripts/lab-prune.sh" "${prune[@]}" || true
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
	# --bench-lock: the exclusive benchmark lock covers the GPU Job only (build first, unlocked),
	# and is taken only once no other GPU-holding lab Job of ours is running or pending, so a
	# queued run never blocks every build's shared lock while it waits for the cards.
	if [[ $BENCH_LOCK -eq 1 ]]; then
		wait_for_free_gpus
		take_bench_lock
	fi
	apply_and_follow run "$JOB" 2 && ok=1
	[[ $BENCH_LOCK -eq 1 ]] && release_bench_lock
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
