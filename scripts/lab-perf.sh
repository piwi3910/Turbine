#!/usr/bin/env bash
# Same-day throughput comparison of Turbine and vLLM-ROCm on the same novanas R9700(s) (Phase 2c
# S-13; the Phase 7 S-13 placements, user decision 2026-10-08 B).
#
#   scripts/lab-perf.sh [--dry-run] novanas <model> [--placement tp1|tp2|ep2] [--eval] [--runs <n>]
#                       [--skip-vllm] [--config <yaml>] [--set <dotted.key>=<value>]...
#
# <model>: llama|olmoe (Phase 2c, one card), or a Phase 7 family: one card (placement tp1, the
# default) qwen3-4b|mistral-7b|gpt-oss-20b|qwen3.5-9b with scripts/lab/phase7-novanas-<model>.yaml;
# two cards (placement tp2 or ep2, required) qwen3moe|mixtral|qwen36 with
# scripts/lab/phase7-novanas-<model>-<placement>.yaml, served with `--gpus 2 --set
# server.listen=0.0.0.0:18000` (the config listens on the cluster pod's loopback), then vLLM-ROCm
# on the same two cards with `--gpus 2 --vllm-arg --tensor-parallel-size --vllm-arg 2` (plus
# `--vllm-arg --enable-expert-parallel` for ep2).
#
# For each engine, one after the other on the same card(s): serve it (Turbine:
# `scripts/lab-serve.sh novanas <config> [--set ...]` with the model's lab config, or --config;
# vLLM: `scripts/lab-serve.sh novanas --vllm <slug>`, the pinned image), warm it up with 16
# requests of the baseline shape, run the baseline workload
#   turbine-bench --concurrency 16 --requests 200 --prompt-words 512 --max-tokens 256 --ignore-eos
# --runs times (default 3), then stop the serve Job this script started (by its run id, never
# anyone else's). --eval (Phase 7) adds, after vLLM's benches, the GSM8K set against vLLM
# (`turbine-golden eval --tasks tests/eval/gsm8k-200.jsonl --concurrency 16`) into
# vllm-gsm8k.json: the baseline committed as tests/eval/<slug>/vllm[-<placement>].json.
# --skip-vllm measures Turbine only and compares with the recorded Phase 2 vLLM numbers
# (Llama-3.2-3B 738, OLMoE-1B-7B 535 output tok/s); a Phase 7 model has none, so it is a usage
# error there. The benchmark runs on this workstation with the release turbine-bench (and
# turbine-golden with --eval), built first.
#
# Writes target/lab-perf/<run-id>/{turbine-<i>.json,vllm-<i>.json,summary.json} (plus the
# warm-up reports, vllm-gsm8k.json and the serve logs), prints every run's numbers and the
# medians (output_token_throughput, itl_ms.p50, ttft_ms.p50, computed with jq), and ends with
#   lab-perf: <model> turbine=<tok/s> vllm=<tok/s> ratio=<r> target=<553|401> verdict=<PASS|FAIL>
# PASS: Turbine's median is >= 0.75 x vLLM's median and >= the target, and no Turbine request
# failed. A Phase 7 model's line is
#   lab-perf: <model> placement=<p> turbine=<tok/s> vllm=<tok/s> ratio=<r> target=0.75x verdict=<PASS|FAIL>
# (the ratio alone), or `… vllm=unavailable ratio=none target=0.75x verdict=NO_VLLM` (exit 1) when
# vLLM-ROCm fails to serve it: the S-13 no-vLLM rule (try a newer rocm/vllm tag once, then gate
# without vLLM).
#
# --dry-run prints the serve (as lab-serve.sh --dry-run renders it), warm-up, benchmark, eval and
# stop commands instead of running them; nothing is contacted or written.
# Exit codes: 0 PASS, 1 FAIL, NO_VLLM or a serve/benchmark failure, 2 usage error.
set -euo pipefail

usage() {
	echo "usage: scripts/lab-perf.sh [--dry-run] novanas <llama|olmoe|qwen3-4b|mistral-7b|gpt-oss-20b|qwen3.5-9b|qwen3moe|mixtral|qwen36> [--placement tp1|tp2|ep2] [--eval] [--reasoning-effort low|medium|high] [--runs <n>] [--skip-vllm] [--config <yaml>] [--set <dotted.key>=<value>]..." >&2
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
# medians (output tok/s, ITL p50 ms, TTFT p50 ms). A Phase 7 model (PHASE7=1) has no recorded
# number and no absolute target: its verdict is the ratio alone; CARDS is how many R9700s it uses.
PHASE7=0
CARDS=1
TARGET=""
RECORDED_TOKS=""
RECORDED_VLLM=""
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
qwen3-4b) SLUG=qwen3-4b-instruct-2507 ;;
mistral-7b) SLUG=mistral-7b-instruct-v0.3 ;;
gpt-oss-20b) SLUG=gpt-oss-20b ;;
qwen3.5-9b) SLUG=qwen3.5-9b ;;
qwen3moe) SLUG=qwen3-30b-a3b-instruct-2507-fp8 ;;
mixtral) SLUG=mixtral-8x7b-instruct-v0.1-fp8 ;;
qwen36) SLUG=qwen3.6-35b-a3b-fp8 ;;
*) usage ;;
esac
case "$MODEL" in
llama | olmoe) ;;
qwen3moe | mixtral | qwen36)
	PHASE7=1
	CARDS=2
	;;
*) PHASE7=1 ;;
esac
shift 2

RUNS=3
SKIP_VLLM=0
CONFIG=""
SETS=()
PLACEMENT=""
EVAL=0
REASONING_EFFORT=""
while [[ $# -gt 0 ]]; do
	case "$1" in
	--placement)
		[[ $# -ge 2 && "$2" =~ ^(tp1|tp2|ep2)$ ]] || usage
		PLACEMENT="$2"
		shift 2
		;;
	--eval)
		EVAL=1
		shift
		;;
	--reasoning-effort)
		# A reasoning model (gpt-oss Harmony) reasons past --max-tokens and a content-only
		# client sees an empty stream without it; `low` keeps it reaching the final channel.
		# Sent to both engines (turbine-bench and turbine-golden eval).
		[[ $# -ge 2 && "$2" =~ ^(low|medium|high)$ ]] || usage
		REASONING_EFFORT="$2"
		shift 2
		;;
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

# The placement rules, checked before anything starts.
usage_error() {
	echo "lab-perf: $1" >&2
	exit 2
}
if [[ $PHASE7 -eq 0 ]]; then
	[[ -z "$PLACEMENT" ]] || usage_error "--placement is for the Phase 7 models; ${MODEL} runs on one card"
	[[ $EVAL -eq 0 ]] || usage_error "--eval is for the Phase 7 models"
else
	[[ $SKIP_VLLM -eq 0 ]] ||
		usage_error "--skip-vllm: ${MODEL} has no recorded vLLM number (Phase 7 measures vLLM-ROCm on the same cards)"
	if [[ $CARDS -eq 2 ]]; then
		[[ "$PLACEMENT" == tp2 || "$PLACEMENT" == ep2 ]] ||
			usage_error "--placement tp2 or ep2 is required for ${MODEL} (both R9700s)"
	else
		[[ -z "$PLACEMENT" || "$PLACEMENT" == tp1 ]] ||
			usage_error "--placement ${PLACEMENT}: ${MODEL} runs on one card (tp1)"
		PLACEMENT=tp1
	fi
fi

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
if [[ -z "$CONFIG" ]]; then
	if [[ $PHASE7 -eq 0 ]]; then
		CONFIG="scripts/lab/phase2c-novanas-${MODEL}.yaml"
	elif [[ $CARDS -eq 2 ]]; then
		CONFIG="scripts/lab/phase7-novanas-${MODEL}-${PLACEMENT}.yaml"
	else
		CONFIG="scripts/lab/phase7-novanas-${MODEL}.yaml"
	fi
fi
# Each engine's serve arguments after the config / slug: both cards for a two-card placement (the
# Turbine config listens on the cluster pod's loopback, so the host-network serve Job overrides it).
TURBINE_SERVE=()
VLLM_SERVE=()
if [[ $CARDS -eq 2 ]]; then
	TURBINE_SERVE=(--gpus 2 --set server.listen=0.0.0.0:18000)
	VLLM_SERVE=(--gpus 2 --vllm-arg --tensor-parallel-size --vllm-arg 2)
	[[ "$PLACEMENT" != ep2 ]] || VLLM_SERVE+=(--vllm-arg --enable-expert-parallel)
fi

[[ $DRY_RUN -eq 1 ]] || command -v jq >/dev/null || {
	echo "lab-perf: jq is required on this workstation" >&2
	exit 2
}

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
BENCH="${CARGO_TARGET_DIR:-target}/release/turbine-bench"
GOLDEN="${CARGO_TARGET_DIR:-target}/release/turbine-golden"
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

# serve <engine> <lab-serve.sh arguments after the host>...: returns 3 instead of failing when
# vLLM-ROCm cannot serve a Phase 7 model (the S-13 no-vLLM rule).
serve() {
	local engine="$1"
	shift
	SERVE_LOG="${LOG_DIR}/serve-${engine}.log"
	echo "+ scripts/lab-serve.sh ${HOST} $*"
	local status=0
	"${REPO_ROOT}/scripts/lab-serve.sh" ${DRY[@]+"${DRY[@]}"} "$HOST" "$@" | tee "$SERVE_LOG" || status=$?
	SERVE_RUN="$(serve_run_id "$SERVE_LOG")"
	SERVE_LOG=""
	if [[ $status -ne 0 && "$engine" == vllm && $PHASE7 -eq 1 ]]; then
		say "vllm: scripts/lab-serve.sh failed (exit ${status}); see ${OUT}/serve-${engine}.log"
		stop_serve || true
		return 3
	fi
	[[ $status -eq 0 ]] || fail "${engine}: scripts/lab-serve.sh failed (exit ${status}); see ${OUT}/serve-${engine}.log"
	[[ -n "$SERVE_RUN" ]] || fail "${engine}: scripts/lab-serve.sh did not announce its run id"
}

# bench <engine> <port> <requests> <report file>
bench() {
	local engine="$1" port="$2" requests="$3" report="$4"
	local cmd=("$BENCH" --url "http://${ADDR}:${port}" --concurrency 16 --requests "$requests"
		--prompt-words 512 --max-tokens 256 --ignore-eos --output json)
	[[ -z "$REASONING_EFFORT" ]] || cmd+=(--reasoning-effort "$REASONING_EFFORT")
	echo "+ ${cmd[*]} > ${report}"
	[[ $DRY_RUN -eq 0 ]] || return 0
	"${cmd[@]}" >"$report" || fail "${engine}: turbine-bench failed (exit $?) against port ${port}"
}

# vllm_eval: the GSM8K set against the vLLM server (--eval), the baseline of the S-13 GSM8K gate.
vllm_eval() {
	local report="${OUT}/vllm-gsm8k.json"
	local cmd=("$GOLDEN" eval --url "http://${ADDR}:18100" --tasks tests/eval/gsm8k-200.jsonl
		--concurrency 16 --output json)
	[[ -z "$REASONING_EFFORT" ]] || cmd+=(--reasoning-effort "$REASONING_EFFORT")
	say "vllm: GSM8K (--eval)"
	echo "+ ${cmd[*]} > ${report}"
	[[ $DRY_RUN -eq 0 ]] || return 0
	"${cmd[@]}" >"$report" || fail "vllm: turbine-golden eval failed (exit $?)"
	say "vllm: GSM8K accuracy $(jq -r '.accuracy' "$report") (${report})"
}

# One line per report: the numbers the verdict and the summary use.
report_line() {
	jq -r '"output_token_throughput=\(.output_token_throughput * 10 | round / 10) tok/s itl_p50=\(.itl_ms.p50 * 10 | round / 10) ms ttft_p50=\(.ttft_ms.p50 | round) ms requests_ok=\(.requests_ok) requests_failed=\(.requests_failed)"' "$1"
}

# measure <engine> <port> <lab-serve.sh arguments after the host>...: returns 3 when vLLM cannot
# serve a Phase 7 model.
measure() {
	local engine="$1" port="$2"
	shift 2
	serve "$engine" "$@" || return $?
	say "${engine}: warm-up (16 requests)"
	bench "$engine" "$port" 16 "${OUT}/${engine}-warmup.json"
	local i
	for ((i = 1; i <= RUNS; i++)); do
		say "${engine}: run ${i}/${RUNS}"
		bench "$engine" "$port" 200 "${OUT}/${engine}-${i}.json"
		[[ $DRY_RUN -eq 1 ]] || say "${engine} run ${i}: $(report_line "${OUT}/${engine}-${i}.json")"
	done
	[[ "$engine" != vllm || $EVAL -eq 0 ]] || vllm_eval
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

# The line prefix and the target word: Phase 7 lines name the placement and the ratio target.
LINE_HEAD="lab-perf: ${MODEL}"
TARGET_WORD="$TARGET"
if [[ $PHASE7 -eq 1 ]]; then
	LINE_HEAD="lab-perf: ${MODEL} placement=${PLACEMENT}"
	TARGET_WORD=0.75x
fi

say "run ${RUN_ID}: ${MODEL} on ${HOST}$([[ $PHASE7 -eq 0 ]] || echo " (${PLACEMENT}, ${CARDS} card(s))"), ${RUNS} run(s) per engine$([[ $SKIP_VLLM -eq 0 ]] || echo ", vLLM skipped (recorded numbers)")"
BUILD=(cargo build --release -p turbine-bench --bin turbine-bench)
[[ $EVAL -eq 0 ]] || BUILD+=(--bin turbine-golden)
echo "+ ${BUILD[*]}"
if [[ $DRY_RUN -eq 0 ]]; then
	"${BUILD[@]}" || fail "building turbine-bench failed"
	mkdir -p "$OUT"
fi

measure turbine 18000 "$CONFIG" ${TURBINE_SERVE[@]+"${TURBINE_SERVE[@]}"} ${SETS[@]+"${SETS[@]}"}
VLLM_OK=1
if [[ $SKIP_VLLM -eq 0 ]]; then
	status=0
	measure vllm 18100 --vllm "$SLUG" ${VLLM_SERVE[@]+"${VLLM_SERVE[@]}"} || status=$?
	[[ $status -eq 0 ]] || VLLM_OK=0
fi

if [[ $DRY_RUN -eq 1 ]]; then
	if [[ $SKIP_VLLM -eq 1 ]]; then vllm_word="${RECORDED_TOKS}(recorded)"; else vllm_word="<median>"; fi
	say "dry run: would write ${OUT}/summary.json and print"
	say "dry run: ${LINE_HEAD} turbine=<median> vllm=${vllm_word} ratio=<r> target=${TARGET_WORD} verdict=<PASS|FAIL>"
	say "dry run: nothing contacted"
	exit 0
fi

TURBINE_JSON="$(engine_summary turbine)"
if [[ $VLLM_OK -eq 0 ]]; then
	# The S-13 no-vLLM rule: Turbine's numbers are recorded, there is no ratio verdict.
	jq -n --arg model "$SLUG" --arg placement "$PLACEMENT" --arg config "$CONFIG" \
		--arg vllm_image "$VLLM_IMAGE" --argjson turbine "$TURBINE_JSON" \
		'{model: $model, placement: $placement, turbine_config: $config, vllm_image: $vllm_image,
		  turbine: $turbine, vllm: "unavailable", verdict: "NO_VLLM"}' >"${OUT}/summary.json"
	echo "${LINE_HEAD} turbine=$(jq -r '.turbine.median_output_token_throughput * 10 | round / 10' "${OUT}/summary.json") vllm=unavailable ratio=none target=${TARGET_WORD} verdict=NO_VLLM"
	exit 1
fi
if [[ $SKIP_VLLM -eq 1 ]]; then
	VLLM_JSON="$(jq '{recorded: true, runs: [], median_output_token_throughput: .output_token_throughput,
		median_itl_p50_ms: .itl_p50_ms, median_ttft_p50_ms: .ttft_p50_ms}' <<<"$RECORDED_VLLM")"
else
	VLLM_JSON="$(engine_summary vllm)"
fi
# A Phase 7 model has no absolute target (null): the ratio alone decides.
jq -n --arg model "$SLUG" --arg date "$(date -u +%Y-%m-%d)" --arg placement "$PLACEMENT" \
	--arg card "Radeon AI PRO R9700 (gfx1201, 32 GB) x${CARDS}, ${HOST}" --arg rocm 7.14.1 \
	--arg vllm_image "$VLLM_IMAGE" --arg config "$CONFIG" --arg sets "${SETS[*]:-}" \
	--arg workload "turbine-bench --concurrency 16 --requests 200 --prompt-words 512 --max-tokens 256 --ignore-eos (warm-up: 16 requests; ${RUNS} run(s))" \
	--argjson turbine "$TURBINE_JSON" --argjson vllm "$VLLM_JSON" --argjson target "${TARGET:-null}" '
	($turbine.median_output_token_throughput / $vllm.median_output_token_throughput) as $ratio
	| ([$turbine.runs[].requests_failed] | add) as $failed
	| {model: $model, date: $date, placement: $placement, card: $card, rocm: $rocm,
	   vllm_image: $vllm_image, turbine_config: $config, turbine_set: $sets, workload: $workload,
	   turbine: $turbine, vllm: $vllm, ratio: $ratio, target: $target,
	   verdict: (if $ratio >= 0.75 and ($target == null or $turbine.median_output_token_throughput >= $target)
	                and $failed == 0
	             then "PASS" else "FAIL" end)}' >"${OUT}/summary.json"

say "medians over ${RUNS} run(s) (summary: ${OUT}/summary.json):"
jq -r '("turbine", "vllm") as $e | .[$e]
	| "lab-perf:   \($e): output_token_throughput=\(.median_output_token_throughput * 10 | round / 10) tok/s itl_p50=\(.median_itl_p50_ms * 10 | round / 10) ms ttft_p50=\(.median_ttft_p50_ms | round) ms\(if .recorded then " (recorded Phase 2 numbers)" else "" end)"' \
	"${OUT}/summary.json"
FAILED="$(jq '[.turbine.runs[].requests_failed] | add' "${OUT}/summary.json")"
[[ "$FAILED" == 0 ]] || say "turbine: ${FAILED} request(s) failed"
LINE="$(jq -r --arg head "$LINE_HEAD" --arg target "$TARGET_WORD" '"\($head) turbine=\(.turbine.median_output_token_throughput * 10 | round / 10) vllm=\(if .vllm.recorded then "\(.vllm.median_output_token_throughput)(recorded)" else "\(.vllm.median_output_token_throughput * 10 | round / 10)" end) ratio=\(.ratio * 100 | round / 100) target=\($target) verdict=\(.verdict)"' "${OUT}/summary.json")"
echo "$LINE"
[[ "$LINE" == *verdict=PASS ]] || exit 1
