#!/usr/bin/env bash
# Run the full Turbine test suite, including #[ignore] GPU tests, on one lab host.
#
#   scripts/lab-test.sh <dgx-spark|dgx-spark2|novanas>
#
# dgx-spark / dgx-spark2: `docker run --rm --gpus all` of rust:1.97-trixie (container
#   turbine-lab-test only; production vLLM containers are never touched).
# novanas: k3s Job scripts/lab/novanas-test-job.yaml in namespace turbine-ci (amd.com/gpu: 2).
# Exit code: the test command's exit code; non-zero with a message naming the failed step.
set -euo pipefail

usage() {
	echo "usage: scripts/lab-test.sh <dgx-spark|dgx-spark2|novanas>" >&2
	exit 2
}

[[ $# -eq 1 ]] || usage
HOST="$1"
case "$HOST" in
dgx-spark) ADDR=192.168.10.246 ;;
dgx-spark2) ADDR=192.168.10.245 ;;
novanas) ADDR=192.168.10.203 ;;
*) usage ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="piwi@${ADDR}"
CI_ROOT=/home/piwi/turbine-ci
IMAGE=rust:1.97-trixie
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10)
NS=turbine-ci
JOB=turbine-lab-test

fail() {
	echo "lab-test: ${HOST}: $1" >&2
	exit "${2:-1}"
}

# Arguments are one remote shell command, expanded locally on purpose.
# shellcheck disable=SC2029
remote() {
	ssh "${SSH_OPTS[@]}" "$REMOTE" "$@"
}

echo "lab-test: ${HOST}: preparing ${CI_ROOT}"
remote "mkdir -p ${CI_ROOT}/src ${CI_ROOT}/target ${CI_ROOT}/cargo-registry" ||
	fail "ssh to ${REMOTE} failed"

echo "lab-test: ${HOST}: syncing working tree"
rsync -az --delete --exclude 'target/' --exclude '.git/' \
	-e "ssh ${SSH_OPTS[*]}" "${REPO_ROOT}/" "${REMOTE}:${CI_ROOT}/src/" ||
	fail "rsync to ${REMOTE}:${CI_ROOT}/src failed"

run_spark() {
	remote "command -v docker >/dev/null" || fail "docker is not available on ${HOST}"
	# Only our own container name is ever removed (a leftover from an aborted run).
	remote "docker rm -f ${JOB} >/dev/null 2>&1 || true"
	echo "lab-test: ${HOST}: docker run ${IMAGE} cargo test --workspace -- --include-ignored"
	set +e
	remote "docker run --rm --gpus all --name ${JOB} --memory 32g \
    -v ${CI_ROOT}/src:/src -v ${CI_ROOT}/target:/target -v turbine-cargo:/usr/local/cargo/registry \
    -e CARGO_TARGET_DIR=/target -e TURBINE_EXPECT_NVIDIA=1 -e TURBINE_EXPECT_NVIDIA_MEMORY=unified \
    -w /src ${IMAGE} cargo test --workspace -- --include-ignored --show-output"
	local code=$?
	set -e
	case "$code" in
	0) echo "lab-test: ${HOST}: PASS" ;;
	125 | 126 | 127) fail "docker run failed (exit ${code})" "$code" ;;
	255) fail "ssh connection lost during the test run" 255 ;;
	*) fail "tests failed (exit ${code})" "$code" ;;
	esac
}

kube() {
	# kubectl on novanas warns about unreadable config files for user piwi. KUBECTL_KUBERC=false
	# silences the kuberc warning (it lacks a trailing newline, so a following error would share
	# its line); the remaining k3s config.yaml warnings are whole lines and only those are dropped.
	remote "export KUBECTL_KUBERC=false; kubectl $*" 2> >(grep -v -e 'permission denied' >&2)
}

run_novanas() {
	remote "command -v kubectl >/dev/null" || fail "kubectl is not available on ${HOST}"
	kube "create namespace ${NS} --dry-run=client -o yaml | kubectl apply -f - >/dev/null" ||
		fail "cannot create namespace ${NS}"
	kube "-n ${NS} delete job ${JOB} --ignore-not-found --wait=true >/dev/null" ||
		fail "cannot delete the previous ${JOB} job"
	kube "apply -f ${CI_ROOT}/src/scripts/lab/novanas-test-job.yaml" || fail "kubectl apply failed"

	# Wait for the pod to start (or finish); give up when it stays unschedulable.
	local waited=0 phase="" unschedulable=""
	while :; do
		phase="$(kube "-n ${NS} get pods -l job-name=${JOB} -o jsonpath='{.items[0].status.phase}'" || true)"
		[[ "$phase" == Running || "$phase" == Succeeded || "$phase" == Failed ]] && break
		unschedulable="$(kube "-n ${NS} get pods -l job-name=${JOB} -o jsonpath='{.items[0].status.conditions[?(@.reason==\"Unschedulable\")].message}'" || true)"
		if [[ -n "$unschedulable" && $waited -ge 120 ]]; then
			kube "-n ${NS} delete job ${JOB} --ignore-not-found >/dev/null" || true
			fail "pod unschedulable for 120 s (${unschedulable}); amd.com/gpu is held by another workload — ask the user to free the GPUs"
		fi
		if [[ $waited -ge 1800 ]]; then
			kube "-n ${NS} delete job ${JOB} --ignore-not-found >/dev/null" || true
			fail "pod did not start within 30 min (phase: ${phase:-none})"
		fi
		sleep 5
		waited=$((waited + 5))
	done

	echo "lab-test: ${HOST}: streaming pod log"
	kube "-n ${NS} logs -f job/${JOB}" || echo "lab-test: ${HOST}: log stream ended with an error" >&2

	# The Job has activeDeadlineSeconds: 1800, so it always reaches a terminal state.
	local succeeded="" failed=""
	while :; do
		succeeded="$(kube "-n ${NS} get job ${JOB} -o jsonpath='{.status.succeeded}'" || true)"
		failed="$(kube "-n ${NS} get job ${JOB} -o jsonpath='{.status.failed}'" || true)"
		[[ "$succeeded" == 1 || -n "$failed" ]] && break
		sleep 5
	done
	if [[ "$succeeded" == 1 ]]; then
		echo "lab-test: ${HOST}: PASS"
		return 0
	fi
	local code
	code="$(kube "-n ${NS} get pods -l job-name=${JOB} -o jsonpath='{.items[0].status.containerStatuses[0].state.terminated.exitCode}'" || true)"
	[[ "$code" =~ ^[0-9]+$ && "$code" -ne 0 ]] || code=1
	fail "tests failed (exit ${code})" "$code"
}

case "$HOST" in
dgx-spark | dgx-spark2) run_spark ;;
novanas) run_novanas ;;
esac
