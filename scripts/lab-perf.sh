#!/usr/bin/env bash
# Same-day throughput comparison of Turbine and vLLM-ROCm on one novanas R9700 (Phase 2c, S-13).
#
#   scripts/lab-perf.sh [--dry-run] novanas <llama|olmoe> [--runs <n>] [--skip-vllm]
#                       [--config <yaml>] [--set <dotted.key>=<value>]...
#
# For each engine, one after the other on the same card: serve it (Turbine:
# `scripts/lab-serve.sh novanas <config> [--set ...]` with the model's Phase 2c lab config
# scripts/lab/phase2c-novanas-<model>.yaml, or --config; vLLM: `scripts/lab-serve.sh novanas
# --vllm <slug>`, the pinned Phase 2 image), warm it up with 16 requests of the baseline shape,
# run the baseline workload
#   turbine-bench --concurrency 16 --requests 200 --prompt-words 512 --max-tokens 256 --ignore-eos
# --runs times (default 3), then stop the serve Job this script started (by its run id, never
# anyone else's). --skip-vllm measures Turbine only and compares with the recorded Phase 2 vLLM
# numbers (Llama-3.2-3B 738, OLMoE-1B-7B 535 output tok/s). The benchmark runs on this
# workstation with the release turbine-bench, built first.
#
# Writes target/lab-perf/<run-id>/{turbine-<i>.json,vllm-<i>.json,summary.json} (plus the
# warm-up reports and the serve logs), prints every run's numbers and the medians
# (output_token_throughput, itl_ms.p50, ttft_ms.p50, computed with jq), and ends with
#   lab-perf: <model> turbine=<tok/s> vllm=<tok/s> ratio=<r> target=<553|401> verdict=<PASS|FAIL>
# PASS: Turbine's median is >= 0.75 x vLLM's median and >= the target, and no Turbine request
# failed.
#
# --dry-run prints the serve (as lab-serve.sh --dry-run renders it), warm-up, benchmark and stop
# commands instead of running them; nothing is contacted or written.
# Exit codes: 0 PASS, 1 FAIL or a serve/benchmark failure, 2 usage error.
set -euo pipefail

usage() {
	echo "usage: scripts/lab-perf.sh [--dry-run] novanas <llama|olmoe> [--runs <n>] [--skip-vllm] [--config <yaml>] [--set <dotted.key>=<value>]..." >&2
	exit 2
}

DRY_RUN=0
if [[ "${1:-}" == --dry-run ]]; then
	DRY_RUN=1
	shift
fi
[[ $# -ge 2 ]] || usage
HOST="$1"
case "$HOST" in
novanas) ADDR=192.168.10.203 ;;
*) usage ;;
esac
MODEL="$2"
# The vLLM slug, the S-15 target (75% of the recorded vLLM number) and the recorded Phase 2 vLLM
# medians (output tok/s, ITL p50 ms, TTFT p50 ms).
case "$MODEL" in
llama)
	SLUG=llama-3.2-3b-instruct
	TARGET=553
	RECORDED_TOKS=738
	RECORDED_VLLM='{"output_token_throughput":738,"itl_p50_ms":17,"ttft_p50_ms":338}'
	;;
olmoe)
	SLUG=olmoe-1b-7b-0125-instruct
	TARGET=401
	RECORDED_TOKS=535
	RECORDED_VLLM='{"output_token_throughput":535,"itl_p50_ms":27,"ttft_p50_ms":201}'
	;;
*) usage ;;
esac
shift 2

RUNS=3
SKIP_VLLM=0
CONFIG=""
SETS=()
while [[ $# -gt 0 ]]; do
	case "$1" in
	--runs)
		[[ $# -ge 2 && "$2" =~ ^[1-9][0-9]?$ ]] || usage
		RUNS="$2"
		shift 2
		;;
	--skip-vllm)
		SKIP_VLLM=1
		shift
		;;
	--config)
		[[ $# -ge 2 ]] || usage
		CONFIG="$2"
		shift 2
		;;
	--set)
		[[ $# -ge 2 ]] || usage
		# The same rule as lab-serve.sh, checked before anything starts.
		if [[ ! "$2" =~ ^[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)*=[A-Za-z0-9_.:/@+,-]*$ ]]; then
			echo "lab-perf: --set expects <dotted.key>=<value> with a plain-word value (letters, digits, _ . : / @ + , -), got: $2" >&2
			exit 2
		fi
		SETS+=(--set "$2")
		shift 2
		;;
	*) usage ;;
	esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "$CONFIG" ]]; then
	if [[ ! -f "$CONFIG" || ! -r "$CONFIG" ]]; then
		echo "lab-perf: config file not found or unreadable: ${CONFIG}" >&2
		exit 2
	fi
	# Paths stay as given when run from the repository root (they are printed as typed).
	[[ "$CONFIG" == /* || "$PWD" == "$REPO_ROOT" ]] || CONFIG="${PWD}/${CONFIG}"
fi
cd "$REPO_ROOT"
[[ -n "$CONFIG" ]] || CONFIG="scripts/lab/phase2c-novanas-${MODEL}.yaml"

[[ $DRY_RUN -eq 1 ]] || command -v jq >/dev/null || {
	echo "lab-perf: jq is required on this workstation" >&2
	exit 2
}

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
BENCH="${CARGO_TARGET_DIR:-target}/release/turbine-bench"
VLLM_IMAGE="$(sed -n 's/^ *image: *//p' scripts/lab/novanas-vllm-job.yaml | head -n 1)"
OUT="target/lab-perf/${RUN_ID}"
# The serve logs (parsed for the run id); a dry run keeps them in a temporary directory.
LOG_DIR="$OUT"
DRY=()
TMP_DIR=""
if [[ $DRY_RUN -eq 1 ]]; then
	DRY=(--dry-run)
	TMP_DIR="$(mktemp -d)"
	LOG_DIR="$TMP_DIR"
fi

# The serve Job this script started and has not stopped: the log of the lab-serve.sh start in
# progress, then the run id it announced.
SERVE_LOG=""
SERVE_RUN=""

say() {
	echo "lab-perf: $*"
}

serve_run_id() {
	sed -n 's/^lab-serve: [a-z0-9-]*: run \([0-9]\{10\}-[0-9a-f]\{8\}\): job .*/\1/p' "$1" | head -n 1
}

stop_serve() {
	if [[ -z "$SERVE_RUN" && -n "$SERVE_LOG" && -f "$SERVE_LOG" ]]; then
		SERVE_RUN="$(serve_run_id "$SERVE_LOG")"
	fi
	SERVE_LOG=""
	[[ -n "$SERVE_RUN" ]] || return 0
	echo "+ scripts/lab-serve.sh ${HOST} --stop ${SERVE_RUN}"
	local run="$SERVE_RUN"
	SERVE_RUN=""
	"${REPO_ROOT}/scripts/lab-serve.sh" ${DRY[@]+"${DRY[@]}"} "$HOST" --stop "$run"
}

on_exit() {
	local status=$?
	trap - EXIT
	if [[ -n "$SERVE_RUN" || -n "$SERVE_LOG" ]]; then
		echo "lab-perf: stopping the serve Job this run started" >&2
		stop_serve || echo "lab-perf: stopping the serve Job failed; run scripts/lab-serve.sh ${HOST} --stop <run-id>" >&2
	fi
	[[ -z "$TMP_DIR" ]] || rm -rf "$TMP_DIR"
	exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT TERM

fail() {
	echo "lab-perf: $1" >&2
	exit 1
}

# serve <engine> <lab-serve.sh arguments after the host>...
serve() {
	local engine="$1"
	shift
	SERVE_LOG="${LOG_DIR}/serve-${engine}.log"
	echo "+ scripts/lab-serve.sh ${HOST} $*"
	local status=0
	"${REPO_ROOT}/scripts/lab-serve.sh" ${DRY[@]+"${DRY[@]}"} "$HOST" "$@" | tee "$SERVE_LOG" || status=$?
	SERVE_RUN="$(serve_run_id "$SERVE_LOG")"
	SERVE_LOG=""
	[[ $status -eq 0 ]] || fail "${engine}: scripts/lab-serve.sh failed (exit ${status}); see ${OUT}/serve-${engine}.log"
	[[ -n "$SERVE_RUN" ]] || fail "${engine}: scripts/lab-serve.sh did not announce its run id"
}

# bench <engine> <port> <requests> <report file>
bench() {
	local engine="$1" port="$2" requests="$3" report="$4"
	local cmd=("$BENCH" --url "http://${ADDR}:${port}" --concurrency 16 --requests "$requests"
		--prompt-words 512 --max-tokens 256 --ignore-eos --output json)
	echo "+ ${cmd[*]} > ${report}"
	[[ $DRY_RUN -eq 0 ]] || return 0
	"${cmd[@]}" >"$report" || fail "${engine}: turbine-bench failed (exit $?) against port ${port}"
}

# One line per report: the numbers the verdict and the summary use.
report_line() {
	jq -r '"output_token_throughput=\(.output_token_throughput * 10 | round / 10) tok/s itl_p50=\(.itl_ms.p50 * 10 | round / 10) ms ttft_p50=\(.ttft_ms.p50 | round) ms requests_ok=\(.requests_ok) requests_failed=\(.requests_failed)"' "$1"
}

# measure <engine> <port> <lab-serve.sh arguments after the host>...
measure() {
	local engine="$1" port="$2"
	shift 2
	serve "$engine" "$@"
	say "${engine}: warm-up (16 requests)"
	bench "$engine" "$port" 16 "${OUT}/${engine}-warmup.json"
	local i
	for ((i = 1; i <= RUNS; i++)); do
		say "${engine}: run ${i}/${RUNS}"
		bench "$engine" "$port" 200 "${OUT}/${engine}-${i}.json"
		[[ $DRY_RUN -eq 1 ]] || say "${engine} run ${i}: $(report_line "${OUT}/${engine}-${i}.json")"
	done
	stop_serve
}

# The runs of one engine and their medians.
engine_summary() {
	local engine="$1" files=() i
	for ((i = 1; i <= RUNS; i++)); do files+=("${OUT}/${engine}-${i}.json"); done
	jq -s 'def median: sort | if length % 2 == 1 then .[(length - 1) / 2] else (.[length / 2 - 1] + .[length / 2]) / 2 end;
		{runs: ., median_output_token_throughput: (map(.output_token_throughput) | median),
		 median_itl_p50_ms: (map(.itl_ms.p50) | median), median_ttft_p50_ms: (map(.ttft_ms.p50) | median)}' "${files[@]}"
}

say "run ${RUN_ID}: ${MODEL} on ${HOST}, ${RUNS} run(s) per engine$([[ $SKIP_VLLM -eq 0 ]] || echo ", vLLM skipped (recorded numbers)")"
echo "+ cargo build --release -p turbine-bench --bin turbine-bench"
if [[ $DRY_RUN -eq 0 ]]; then
	cargo build --release -p turbine-bench --bin turbine-bench || fail "building turbine-bench failed"
	mkdir -p "$OUT"
fi

measure turbine 18000 "$CONFIG" ${SETS[@]+"${SETS[@]}"}
[[ $SKIP_VLLM -eq 1 ]] || measure vllm 18100 --vllm "$SLUG"

if [[ $DRY_RUN -eq 1 ]]; then
	if [[ $SKIP_VLLM -eq 1 ]]; then vllm_word="${RECORDED_TOKS}(recorded)"; else vllm_word="<median>"; fi
	say "dry run: would write ${OUT}/summary.json and print"
	say "dry run: lab-perf: ${MODEL} turbine=<median> vllm=${vllm_word} ratio=<r> target=${TARGET} verdict=<PASS|FAIL>"
	say "dry run: nothing contacted"
	exit 0
fi

TURBINE_JSON="$(engine_summary turbine)"
if [[ $SKIP_VLLM -eq 1 ]]; then
	VLLM_JSON="$(jq '{recorded: true, runs: [], median_output_token_throughput: .output_token_throughput,
		median_itl_p50_ms: .itl_p50_ms, median_ttft_p50_ms: .ttft_p50_ms}' <<<"$RECORDED_VLLM")"
else
	VLLM_JSON="$(engine_summary vllm)"
fi
jq -n --arg model "$SLUG" --arg date "$(date -u +%Y-%m-%d)" \
	--arg card "Radeon AI PRO R9700 (gfx1201, 32 GB), ${HOST}" --arg rocm 7.14.1 \
	--arg vllm_image "$VLLM_IMAGE" --arg config "$CONFIG" --arg sets "${SETS[*]:-}" \
	--arg workload "turbine-bench --concurrency 16 --requests 200 --prompt-words 512 --max-tokens 256 --ignore-eos (warm-up: 16 requests; ${RUNS} run(s))" \
	--argjson turbine "$TURBINE_JSON" --argjson vllm "$VLLM_JSON" --argjson target "$TARGET" '
	($turbine.median_output_token_throughput / $vllm.median_output_token_throughput) as $ratio
	| ([$turbine.runs[].requests_failed] | add) as $failed
	| {model: $model, date: $date, card: $card, rocm: $rocm, vllm_image: $vllm_image,
	   turbine_config: $config, turbine_set: $sets, workload: $workload,
	   turbine: $turbine, vllm: $vllm, ratio: $ratio, target: $target,
	   verdict: (if $ratio >= 0.75 and $turbine.median_output_token_throughput >= $target and $failed == 0
	             then "PASS" else "FAIL" end)}' >"${OUT}/summary.json"

say "medians over ${RUNS} run(s) (summary: ${OUT}/summary.json):"
jq -r '("turbine", "vllm") as $e | .[$e]
	| "lab-perf:   \($e): output_token_throughput=\(.median_output_token_throughput * 10 | round / 10) tok/s itl_p50=\(.median_itl_p50_ms * 10 | round / 10) ms ttft_p50=\(.median_ttft_p50_ms | round) ms\(if .recorded then " (recorded Phase 2 numbers)" else "" end)"' \
	"${OUT}/summary.json"
FAILED="$(jq '[.turbine.runs[].requests_failed] | add' "${OUT}/summary.json")"
[[ "$FAILED" == 0 ]] || say "turbine: ${FAILED} request(s) failed"
LINE="$(jq -r --arg model "$MODEL" '"lab-perf: \($model) turbine=\(.turbine.median_output_token_throughput * 10 | round / 10) vllm=\(if .vllm.recorded then "\(.vllm.median_output_token_throughput)(recorded)" else "\(.vllm.median_output_token_throughput * 10 | round / 10)" end) ratio=\(.ratio * 100 | round / 100) target=\(.target) verdict=\(.verdict)"' "${OUT}/summary.json")"
echo "$LINE"
[[ "$LINE" == *verdict=PASS ]] || exit 1
