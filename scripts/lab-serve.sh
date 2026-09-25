#!/usr/bin/env bash
# Run turbine-server on a lab host for manual golden and benchmark runs.
#
#   scripts/lab-serve.sh [--dry-run] novanas <config.yaml>
#   scripts/lab-serve.sh [--dry-run] novanas --stop
#
# novanas: rsyncs the tree to /home/piwi/turbine-ci/src, copies <config.yaml> to
#   /home/piwi/turbine-ci/serve/config.yaml and applies the k3s Job
#   scripts/lab/novanas-serve-job.yaml (namespace turbine-ci, one R9700, host network), which
#   builds libturbine_hip.so and the release turbine-server and runs it with that config. The
#   script streams the Job log until http://192.168.10.203:18000/ready answers 200, then exits 0
#   and leaves the server running. --stop deletes that Job and nothing else.
# --dry-run prints every command that would contact the host instead of running it.
# TURBINE_LAB_SERVE_TIMEOUT: seconds to wait for /ready once the pod runs (default 3600; a cold
#   run compiles Composable Kernel and the release server).
# Exit codes: 0 ready (or stopped), 2 usage error, otherwise non-zero with a message naming the
# failed step.
set -euo pipefail

usage() {
	echo "usage: scripts/lab-serve.sh [--dry-run] novanas <config.yaml>" >&2
	echo "       scripts/lab-serve.sh [--dry-run] novanas --stop" >&2
	exit 2
}

DRY_RUN=0
if [[ "${1:-}" == --dry-run ]]; then
	DRY_RUN=1
	shift
fi
[[ $# -eq 2 ]] || usage
HOST="$1"
case "$HOST" in
novanas) ADDR=192.168.10.203 ;;
*) usage ;;
esac

MODE=start
CONFIG=""
case "$2" in
--stop) MODE=stop ;;
-*) usage ;;
*)
	CONFIG="$2"
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
JOB=turbine-lab-serve
URL="http://${ADDR}:18000"
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
	say "deleting job ${JOB} in namespace ${NS}"
	kube "-n ${NS} delete job ${JOB} --ignore-not-found --wait=true" ||
		fail "cannot delete job ${JOB}"
	if [[ $DRY_RUN -eq 1 ]]; then say "dry run: nothing contacted"; else say "stopped"; fi
}

# The step the Job's setup script reported as failed (`lab-step failed: <name>`), if any.
failed_step() {
	kube "-n ${NS} logs job/${JOB} --tail=400" 2>/dev/null |
		sed -n 's/^lab-step failed: //p' | tail -n 1 || true
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
		phase="$(kube "-n ${NS} get pods -l job-name=${JOB} -o jsonpath='{.items[0].status.phase}'" || true)"
		[[ "$phase" == Running ]] && return 0
		if [[ "$phase" == Succeeded || "$phase" == Failed ]]; then
			local step
			step="$(failed_step)"
			fail "job ${JOB} ended (${phase}) before turbine-server started${step:+; failed step: ${step}}"
		fi
		unschedulable="$(kube "-n ${NS} get pods -l job-name=${JOB} -o jsonpath='{.items[0].status.conditions[?(@.reason==\"Unschedulable\")].message}'" || true)"
		if [[ -n "$unschedulable" && $waited -ge 120 ]]; then
			kube "-n ${NS} delete job ${JOB} --ignore-not-found >/dev/null" || true
			fail "pod unschedulable for 120 s (${unschedulable}); amd.com/gpu is held by another workload — ask the user to free an R9700"
		fi
		if [[ $waited -ge 1800 ]]; then
			kube "-n ${NS} delete job ${JOB} --ignore-not-found >/dev/null" || true
			fail "pod did not start within 30 min (phase: ${phase:-none})"
		fi
		sleep 5
		waited=$((waited + 5))
	done
}

wait_for_ready() {
	local probe=(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "${URL}/ready")
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
			step="$(failed_step)"
			[[ -n "$step" ]] || step="turbine-server (exited before /ready; see the log above)"
			fail "job ${JOB} failed at step: ${step}"
		fi
		if [[ $waited -ge $TIMEOUT ]]; then
			stop_log_stream
			kube "-n ${NS} delete job ${JOB} --ignore-not-found >/dev/null" || true
			fail "${URL}/ready did not answer 200 within ${TIMEOUT} s (last status: ${code:-none}); job ${JOB} deleted"
		fi
		sleep 5
		waited=$((waited + 5))
	done
}

start() {
	say "preparing ${CI_ROOT}"
	remote "mkdir -p ${CI_ROOT}/src ${CI_ROOT}/target ${CI_ROOT}/cargo-registry ${CI_ROOT}/serve" ||
		fail "ssh to ${REMOTE} failed"

	say "syncing working tree"
	run rsync -az --delete --exclude target/ --exclude .git/ \
		-e "ssh ${SSH_OPTS[*]}" "${REPO_ROOT}/" "${REMOTE}:${CI_ROOT}/src/" ||
		fail "rsync to ${REMOTE}:${CI_ROOT}/src failed"

	say "copying ${CONFIG}"
	run rsync -az -e "ssh ${SSH_OPTS[*]}" "$CONFIG" "${REMOTE}:${CI_ROOT}/serve/config.yaml" ||
		fail "copying ${CONFIG} to ${REMOTE}:${CI_ROOT}/serve/config.yaml failed"

	remote "command -v kubectl >/dev/null" || fail "kubectl is not available on ${HOST}"
	kube "create namespace ${NS} --dry-run=client -o yaml | kubectl apply -f - >/dev/null" ||
		fail "cannot create namespace ${NS}"
	# Only our own Job is replaced (a previous server or an aborted start).
	kube "-n ${NS} delete job ${JOB} --ignore-not-found --wait=true >/dev/null" ||
		fail "cannot delete the previous ${JOB} job"
	say "applying scripts/lab/novanas-serve-job.yaml"
	kube "apply -f ${CI_ROOT}/src/scripts/lab/novanas-serve-job.yaml" || fail "kubectl apply failed"

	say "waiting for the pod"
	wait_for_pod
	say "streaming pod log; waiting for ${URL}/ready"
	stream_log
	wait_for_ready
	stop_log_stream
	if [[ $DRY_RUN -eq 1 ]]; then
		say "dry run: nothing contacted"
	else
		say "ready at ${URL} (stop with: scripts/lab-serve.sh ${HOST} --stop)"
	fi
}

case "$MODE" in
start) start ;;
stop) stop ;;
esac
