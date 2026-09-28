#!/usr/bin/env bash
# Run turbine-server on a lab host for manual golden and benchmark runs.
#
#   scripts/lab-serve.sh [--dry-run] novanas <config.yaml> [--set <dotted.key>=<value>]...
#   scripts/lab-serve.sh [--dry-run] novanas --vllm <slug> [--gpus 1|2] [--vllm-arg <arg>]...
#   scripts/lab-serve.sh [--dry-run] novanas --stop [<run-id>]
#
# novanas: uploads the tree and <config.yaml> to /home/piwi/turbine-ci/runs/<run id> and applies
#   the k3s Job template scripts/lab/novanas-serve-job.yaml as turbine-lab-serve-<run id>
#   (namespace turbine-ci, one R9700, host network), which syncs them into the cached release
#   workspace slot, builds libturbine_hip.so and the release turbine-server incrementally and
#   runs it with that config. Each --set pair is appended as `--set <dotted.key>=<value>` to the
#   turbine-server command line, so sweeps need no config edits; since the pairs are rendered
#   into the Job's shell script, values are plain words (letters, digits and _ . : / @ + , -).
#   The script refuses to start while something already answers on
#   port 18000, streams the Job log until http://192.168.10.203:18000/ready answers 200, then
#   exits 0 and leaves the server running. --stop deletes the serve Jobs (label
#   turbine-lab-role=serve: the Turbine server and the vLLM baseline; one of each at a time, the
#   ports are fixed) and nothing else; --stop <run-id> deletes only the serve Job of that run
#   (the id a start prints as `run <run-id>: job ...`); an interrupted or failed start deletes
#   its own Job.
# novanas --vllm <slug>: the Phase 2 baseline. Applies scripts/lab/novanas-vllm-job.yaml as
#   turbine-lab-vllm-<run id> (upstream rocm/vllm at a pinned tag, one R9700, host network)
#   serving /home/piwi/turbine-models/<slug> read-only on port 18100 under the model id
#   Turbine's lab config uses; <slug> is llama-3.2-3b-instruct or olmoe-1b-7b-0125-instruct.
#   Nothing is uploaded. Refuses to start while something answers on port 18100, streams the pod
#   log until http://192.168.10.203:18100/v1/models answers 200 and exits 0 leaving vLLM running;
#   when vLLM fails to start or serve (e.g. on gfx1201) it prints the pod log and exits 1.
#   --gpus 2 gives the Job both R9700s and each --vllm-arg is appended to `vllm serve` in
#   order (plain words: letters, digits and _ . : = / -), for the Phase 5 two-GPU baselines
#   (e.g. --vllm-arg --tensor-parallel-size --vllm-arg 2).
# --dry-run prints every command that would contact the host instead of running it.
# TURBINE_LAB_SERVE_TIMEOUT: seconds to wait for /ready once the pod runs (default 3600; a cold
#   run compiles Composable Kernel and the release server).
# Exit codes: 0 ready (or stopped), 2 usage error, otherwise non-zero with a message naming the
# failed step.
set -euo pipefail

usage() {
	echo "usage: scripts/lab-serve.sh [--dry-run] novanas <config.yaml> [--set <dotted.key>=<value>]..." >&2
	echo "       scripts/lab-serve.sh [--dry-run] novanas --vllm <slug> [--gpus 1|2] [--vllm-arg <arg>]..." >&2
	echo "       scripts/lab-serve.sh [--dry-run] novanas --stop [<run-id>]" >&2
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

MODE=start
CONFIG=""
SLUG=""
STOP_RUN=""
# Rendered into the serve Job's turbine-server command line (scripts/lab/novanas-serve-job.yaml).
SERVER_ARGS=""
VLLM_GPUS=1
VLLM_ARGS=()
case "$2" in
--stop)
	[[ $# -le 3 ]] || usage
	MODE=stop
	if [[ $# -eq 3 ]]; then
		STOP_RUN="$3"
		if [[ ! "$STOP_RUN" =~ ^[0-9]{10}-[0-9a-f]{8}$ ]]; then
			echo "lab-serve: ${HOST}: not a run id: ${STOP_RUN}" >&2
			usage
		fi
	fi
	;;
--vllm)
	[[ $# -ge 3 ]] || usage
	MODE=vllm
	SLUG="$3"
	shift 3
	while [[ $# -gt 0 ]]; do
		case "$1" in
		--gpus)
			[[ $# -ge 2 ]] || usage
			if [[ "$2" != 1 && "$2" != 2 ]]; then
				echo "lab-serve: ${HOST}: --gpus is 1 or 2 (the R9700s of novanas), got: $2" >&2
				exit 2
			fi
			VLLM_GPUS="$2"
			;;
		--vllm-arg)
			[[ $# -ge 2 ]] || usage
			if [[ ! "$2" =~ ^[A-Za-z0-9_.:=/-]+$ ]]; then
				echo "lab-serve: ${HOST}: --vllm-arg expects a plain word (letters, digits, _ . : = / -), got: $2" >&2
				exit 2
			fi
			VLLM_ARGS+=("$2")
			;;
		*) usage ;;
		esac
		shift 2
	done
	# The model id each slug is served under (as in scripts/lab/phase2-novanas-*.yaml) and
	# Turbine's default max_seq_len for it: min(32768, max_position_embeddings).
	case "$SLUG" in
	llama-3.2-3b-instruct)
		SERVED_NAME=meta-llama/Llama-3.2-3B-Instruct
		MAX_MODEL_LEN=32768
		;;
	olmoe-1b-7b-0125-instruct)
		SERVED_NAME=allenai/OLMoE-1B-7B-0125-Instruct
		MAX_MODEL_LEN=4096
		;;
	*)
		echo "lab-serve: ${HOST}: unknown model slug for --vllm: ${SLUG} (llama-3.2-3b-instruct or olmoe-1b-7b-0125-instruct)" >&2
		usage
		;;
	esac
	;;
-*) usage ;;
*)
	CONFIG="$2"
	shift 2
	while [[ $# -gt 0 ]]; do
		[[ "$1" == --set && $# -ge 2 ]] || usage
		if [[ ! "$2" =~ ^[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)*=[A-Za-z0-9_.:/@+,-]*$ ]]; then
			echo "lab-serve: ${HOST}: --set expects <dotted.key>=<value> with a plain-word value (letters, digits, _ . : / @ + , -), got: $2" >&2
			exit 2
		fi
		SERVER_ARGS+=" --set $2"
		shift 2
	done
	if [[ ! -f "$CONFIG" || ! -r "$CONFIG" ]]; then
		echo "lab-serve: ${HOST}: config file not found or unreadable: ${CONFIG}" >&2
		exit 2
	fi
	;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="piwi@${ADDR}"
CI_ROOT=/home/piwi/turbine-ci
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10)
NS=turbine-ci
# Short, unique and DNS-1123 safe: UTC time plus 30 random bits.
RUN_ID="$(date -u +%m%d%H%M%S)-$(printf '%08x' $(((RANDOM << 15) | RANDOM)))"
JOB="turbine-lab-serve-${RUN_ID}"
RUN_DIR="${CI_ROOT}/runs/${RUN_ID}"
URL="http://${ADDR}:18000"
READY_PATH=/ready
if [[ "$MODE" == vllm ]]; then
	JOB="turbine-lab-vllm-${RUN_ID}"
	URL="http://${ADDR}:18100"
	READY_PATH=/v1/models
fi
TIMEOUT="${TURBINE_LAB_SERVE_TIMEOUT:-3600}"
[[ "$TIMEOUT" =~ ^[0-9]+$ ]] || {
	echo "lab-serve: TURBINE_LAB_SERVE_TIMEOUT must be a number of seconds, got ${TIMEOUT}" >&2
	exit 2
}

fail() {
	echo "lab-serve: ${HOST}: $1" >&2
	exit "${2:-1}"
}

say() {
	echo "lab-serve: ${HOST}: $*"
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
		# Same kubectl warning filter as kube.
		ssh "${SSH_OPTS[@]}" "$REMOTE" "$@" 2> >(grep -v -e 'permission denied' >&2)
	fi
}

kube() {
	# Same filtering as scripts/lab-test.sh: kubectl on novanas warns about unreadable config
	# files for user piwi; KUBECTL_KUBERC=false silences the kuberc one, the rest are dropped.
	if [[ $DRY_RUN -eq 1 ]]; then
		remote "export KUBECTL_KUBERC=false; kubectl $*"
	else
		remote "export KUBECTL_KUBERC=false; kubectl $*" 2> >(grep -v -e 'permission denied' >&2)
	fi
}

stop() {
	local selector=turbine-lab-role=serve
	[[ -z "$STOP_RUN" ]] || selector+=",turbine-lab-run=${STOP_RUN}"
	say "deleting the serve jobs (label ${selector}) in namespace ${NS}"
	kube "-n ${NS} delete job -l ${selector} --ignore-not-found --wait=true" ||
		fail "cannot delete the serve jobs"
	if [[ $DRY_RUN -eq 1 ]]; then say "dry run: nothing contacted"; else say "stopped"; fi
}

# The step the Job's setup script reported as failed (`lab-step failed: <name>`), if any.
failed_step() {
	kube "-n ${NS} logs job/${JOB} --tail=400" 2>/dev/null |
		sed -n 's/^lab-step failed: //p' | tail -n 1 || true
}

# Deletes this run's Job and upload, nothing else.
cleanup() {
	kube "-n ${NS} delete job ${JOB} --ignore-not-found" >/dev/null || true
	[[ "$MODE" == vllm ]] || remote "rm -rf ${RUN_DIR}" || true
}

# vLLM prints no lab steps: its failure record is the end of the pod log.
print_pod_log() {
	echo "lab-serve: ${HOST}: pod log of ${JOB} (last 200 lines):" >&2
	kube "-n ${NS} logs job/${JOB} --tail=200" >&2 || true
}

# Why a Job ended before serving: the failed lab step (Turbine), or for vLLM a pointer to the
# pod log, printed first.
ended_reason() {
	if [[ "$MODE" == vllm ]]; then
		print_pod_log
		echo "vLLM (exited before serving; pod log above)"
	else
		local step
		step="$(failed_step)"
		echo "${step:-turbine-server (exited before /ready; see the log above)}"
	fi
}

on_interrupt() {
	trap - INT TERM
	echo "lab-serve: ${HOST}: interrupted; deleting job ${JOB}" >&2
	cleanup
	exit 130
}

LOG_PID=""
stop_log_stream() {
	if [[ -n "$LOG_PID" ]]; then
		kill "$LOG_PID" 2>/dev/null || true
		wait "$LOG_PID" 2>/dev/null || true
		LOG_PID=""
	fi
}
trap stop_log_stream EXIT

# Streams the Job log in the background; the stream is stopped once the server is ready.
# shellcheck disable=SC2029
stream_log() {
	local cmd="export KUBECTL_KUBERC=false; kubectl -n ${NS} logs -f job/${JOB} 2>&1 | grep --line-buffered -v 'permission denied'"
	if [[ $DRY_RUN -eq 1 ]]; then
		echo "+ ssh ${SSH_OPTS[*]} ${REMOTE} '${cmd}'"
		return
	fi
	(exec ssh -n "${SSH_OPTS[@]}" "$REMOTE" "$cmd") &
	LOG_PID=$!
}

wait_for_pod() {
	if [[ $DRY_RUN -eq 1 ]]; then
		echo "+ wait until pod job-name=${JOB} is Running (unschedulable limit 120 s)"
		return
	fi
	local waited=0 phase="" unschedulable=""
	while :; do
		phase="$(kube "-n ${NS} get pods -l job-name=${JOB} -o jsonpath='{.items[*].status.phase}'" || true)"
		[[ "$phase" == Running ]] && return 0
		if [[ "$phase" == Succeeded || "$phase" == Failed ]]; then
			# The reason is read from the Job's log, so before the Job is deleted.
			local reason
			if [[ "$MODE" == vllm ]]; then
				reason="$(ended_reason)"
				cleanup
				fail "job ${JOB} ended (${phase}) before serving: ${reason}; job deleted"
			fi
			reason="$(failed_step)"
			cleanup
			fail "job ${JOB} ended (${phase}) before turbine-server started${reason:+; failed step: ${reason}}; job deleted"
		fi
		if [[ "$MODE" == vllm && $waited -ge 300 ]]; then
			# An image that cannot be pulled keeps the pod Pending forever.
			local pull
			pull="$(kube "-n ${NS} get pods -l job-name=${JOB} -o jsonpath='{.items[*].status.containerStatuses[*].state.waiting.reason}'" || true)"
			if [[ "$pull" == *ErrImagePull* || "$pull" == *ImagePullBackOff* || "$pull" == *InvalidImageName* ]]; then
				print_pod_log
				cleanup
				fail "cannot pull the vLLM image (${pull}) for 300 s; job ${JOB} deleted"
			fi
		fi
		unschedulable="$(kube "-n ${NS} get pods -l job-name=${JOB} -o jsonpath='{.items[*].status.conditions[?(@.reason==\"Unschedulable\")].message}'" || true)"
		if [[ -n "$unschedulable" && $waited -ge 120 ]]; then
			# The scheduler's own reason (a busy amd.com/gpu, or a node taint such as disk
			# pressure), never a guess.
			cleanup
			if [[ "$unschedulable" == *"amd.com/gpu"* ]]; then
				fail "pod unschedulable for 120 s: ${unschedulable} — amd.com/gpu is held; identify the holder, ask the user to free an R9700"
			fi
			fail "pod unschedulable for 120 s: ${unschedulable}"
		fi
		if [[ $waited -ge 1800 ]]; then
			cleanup
			fail "pod did not start within 30 min (phase: ${phase:-none})"
		fi
		sleep 5
		waited=$((waited + 5))
	done
}

wait_for_ready() {
	local probe=(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "${URL}${READY_PATH}")
	if [[ $DRY_RUN -eq 1 ]]; then
		run "${probe[@]}"
		echo "+ repeat every 5 s until 200 (limit ${TIMEOUT} s); stop when job ${JOB} fails"
		return
	fi
	local waited=0 code="" failed="" step=""
	while :; do
		code="$("${probe[@]}" || true)"
		if [[ "$code" == 200 ]]; then
			return 0
		fi
		failed="$(kube "-n ${NS} get job ${JOB} -o jsonpath='{.status.failed}'" || true)"
		if [[ -n "$failed" && "$failed" != 0 ]]; then
			stop_log_stream
			step="$(ended_reason)"
			cleanup
			fail "job ${JOB} failed at step: ${step}; job deleted"
		fi
		if [[ $waited -ge $TIMEOUT ]]; then
			stop_log_stream
			[[ "$MODE" != vllm ]] || print_pod_log
			cleanup
			fail "${URL}${READY_PATH} did not answer 200 within ${TIMEOUT} s (last status: ${code:-none}); job ${JOB} deleted"
		fi
		sleep 5
		waited=$((waited + 5))
	done
}

# Refuses to start while something already answers on the port this run serves.
require_free_port() {
	local probe=(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "${URL}${READY_PATH}")
	if [[ $DRY_RUN -eq 1 ]]; then
		run "${probe[@]}"
		echo "+ refuse to start unless that fails to connect (000)"
	else
		local code
		code="$("${probe[@]}" || true)"
		[[ "$code" == 000 || -z "$code" ]] ||
			fail "${URL} already answers (HTTP ${code}); stop that server first (scripts/lab-serve.sh ${HOST} --stop)"
	fi
}

start() {
	say "run ${RUN_ID}: job ${JOB}"
	require_free_port

	remote "mkdir -p ${RUN_DIR}/src ${CI_ROOT}/cache/slots && find ${CI_ROOT}/runs -mindepth 1 -maxdepth 1 -mmin +1440 -exec rm -rf {} +" ||
		fail "ssh to ${REMOTE} failed"
	[[ $DRY_RUN -eq 1 ]] || trap on_interrupt INT TERM

	say "syncing working tree to ${RUN_DIR}/src"
	run rsync -rlpcz --delete --exclude target/ --exclude .git/ --exclude .claude/ \
		-e "ssh ${SSH_OPTS[*]}" "${REPO_ROOT}/" "${REMOTE}:${RUN_DIR}/src/" ||
		fail "rsync to ${REMOTE}:${RUN_DIR}/src failed"

	say "copying ${CONFIG}"
	run rsync -az -e "ssh ${SSH_OPTS[*]}" "$CONFIG" "${REMOTE}:${RUN_DIR}/config.yaml" ||
		fail "copying ${CONFIG} to ${REMOTE}:${RUN_DIR}/config.yaml failed"

	remote "command -v kubectl >/dev/null" || fail "kubectl is not available on ${HOST}"
	kube "create namespace ${NS} --dry-run=client -o yaml | kubectl apply -f - >/dev/null" ||
		fail "cannot create namespace ${NS}"
	say "applying scripts/lab/novanas-serve-job.yaml as ${JOB}"
	sed -e "s/__RUN_ID__/${RUN_ID}/g" -e "s|__SERVER_ARGS__|${SERVER_ARGS}|g" \
		"${REPO_ROOT}/scripts/lab/novanas-serve-job.yaml" |
		remote_stdin "export KUBECTL_KUBERC=false; kubectl apply -f -" || fail "kubectl apply failed"

	say "waiting for the pod"
	wait_for_pod
	say "streaming pod log; waiting for ${URL}/ready"
	stream_log
	wait_for_ready
	stop_log_stream
	trap - INT TERM
	if [[ $DRY_RUN -eq 1 ]]; then
		say "dry run: nothing contacted"
	else
		say "ready at ${URL} (stop with: scripts/lab-serve.sh ${HOST} --stop)"
	fi
}

start_vllm() {
	say "run ${RUN_ID}: job ${JOB} (vLLM-ROCm serving ${SLUG} as ${SERVED_NAME}, ${VLLM_GPUS} GPU(s)${VLLM_ARGS[*]+, ${VLLM_ARGS[*]}})"
	require_free_port
	remote "test -d /home/piwi/turbine-models/${SLUG}" ||
		fail "weights /home/piwi/turbine-models/${SLUG} are not provisioned (or ssh to ${REMOTE} failed)"
	remote "command -v kubectl >/dev/null" || fail "kubectl is not available on ${HOST}"
	kube "create namespace ${NS} --dry-run=client -o yaml | kubectl apply -f - >/dev/null" ||
		fail "cannot create namespace ${NS}"
	[[ $DRY_RUN -eq 1 ]] || trap on_interrupt INT TERM
	say "applying scripts/lab/novanas-vllm-job.yaml as ${JOB}"
	local extra="" a
	for a in "${VLLM_ARGS[@]+"${VLLM_ARGS[@]}"}"; do
		extra+="            - \"${a}\"\n"
	done
	sed -e "s/__RUN_ID__/${RUN_ID}/g" -e "s/__SLUG__/${SLUG}/g" \
		-e "s|__SERVED_NAME__|${SERVED_NAME}|g" -e "s/__MAX_MODEL_LEN__/${MAX_MODEL_LEN}/g" \
		-e "s/__GPUS__/${VLLM_GPUS}/g" \
		"${REPO_ROOT}/scripts/lab/novanas-vllm-job.yaml" |
		awk -v extra="$extra" '/# __EXTRA_ARGS__$/ { printf "%s", extra; next } { print }' |
		remote_stdin "export KUBECTL_KUBERC=false; kubectl apply -f -" || fail "kubectl apply failed"

	say "waiting for the pod"
	wait_for_pod
	say "streaming pod log; waiting for ${URL}${READY_PATH}"
	stream_log
	wait_for_ready
	stop_log_stream
	trap - INT TERM
	if [[ $DRY_RUN -eq 1 ]]; then
		say "dry run: nothing contacted"
	else
		say "vLLM ready at ${URL} (stop with: scripts/lab-serve.sh ${HOST} --stop)"
	fi
}

case "$MODE" in
start) start ;;
vllm) start_vllm ;;
stop) stop ;;
esac
