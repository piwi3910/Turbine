#!/usr/bin/env bash
# Run the Turbine test suite, including #[ignore] GPU tests, on one lab host.
#
#   scripts/lab-test.sh [--dry-run] <dgx-spark|dgx-spark2|novanas> [--gpus 1|2] [--with-hf-reference]
#                       [--features <list>] [--tier quick|perf|full] [-- <cargo test arguments>]
#   scripts/lab-test.sh [--dry-run] novanas --stop <run-id>
#
# Runs `cargo test --no-fail-fast <selection> -- --include-ignored --show-output`, so one run
# reports every failing test target. <selection> defaults to --workspace; anything after `--`
# replaces it (e.g. `-- -p turbine-kernels --test hip_ops`), and a second `--` passes the rest
# to the test harness after ours (e.g. `-- -p turbine-model --test golden -- logits_match`).
# The default run skips hf_reference_matches_cpu (the Hugging Face transformers CPU reference:
# the golden references are committed, it only matters when regenerating them);
# --with-hf-reference runs it and installs uv for it. --features <list> is passed to that cargo test
# (e.g. `--features fault-injection` for the P3 fault tests, `tests/fault.rs`).
# --tier (default full, unchanged behaviour): `quick` skips the slow perf/timing tests listed in
#   SLOW_TESTS below (a single data-driven list — see there for how it was measured) via libtest
#   `--skip` filters, for a fast per-landing-step GPU pass; `perf` runs only those tests (the
#   reverse selection, for a focused perf pass); `full` runs everything, unchanged.
#
# dgx-spark / dgx-spark2: `docker run --rm --gpus all` of rust:1.97-trixie (container
#   turbine-lab-test only; production vLLM containers are never touched).
# novanas: the k3s Job template scripts/lab/novanas-test-job.yaml in namespace turbine-ci, named
#   turbine-lab-test-<run id> so several runs can go side by side. --gpus (default 1) is the
#   R9700 count the Job requests and the AMD device count the inventory test expects; only the
#   Phase 0 inventory check needs 2. The tree and the test command (NUL-separated argv) are
#   uploaded to /home/piwi/turbine-ci/runs/<run id>; the Job syncs the tree into a cached
#   workspace slot (see the template for the cache layout and the per-slot target dirs).
#   The same template first runs GPU-less as turbine-lab-build-<run id>, which compiles the
#   kernels and the test binaries (`--no-run`) into a slot, so the GPU Job holds its R9700 only
#   for the tests themselves.
#   Ctrl-C or a failed start deletes this run's Jobs and nothing else; `--stop <run-id>` does the
#   same for a run started elsewhere.
# --dry-run prints every command that would contact the host (and the rendered Job) instead of
#   running it.
# Exit code: the test command's exit code; 2 for usage errors; otherwise non-zero with a message
# naming the failed step.
set -euo pipefail

usage() {
	echo "usage: scripts/lab-test.sh [--dry-run] <dgx-spark|dgx-spark2|novanas> [--gpus 1|2] [--with-hf-reference] [--features <list>] [--tier quick|perf|full] [-- <cargo test args>]" >&2
	echo "       scripts/lab-test.sh [--dry-run] novanas --stop <run-id>" >&2
	exit 2
}

# The turbine-model perf/timing tests and the turbine-kernels hip_ops timing/exhaustive-match
# tests: > ~60 s each on novanas (nextest's "has been running for over 60 seconds" warning in
# scripts/lab-test.sh full runs, e.g. target/lab-perf and lab-test logs from 2026-09-26/27).
# One list, used both ways by --tier: skipped for `quick`, the only ones run for `perf`.
SLOW_TESTS=(
	# crates/turbine-model/tests/perf.rs, tests/host_step.rs
	serving_mix
	forward_profile
	host_step_costs
	# crates/turbine-kernels/tests/hip_ops.rs
	decode_forward_timing
	decode_op_timings
	fused_projection_timings
	prefill_op_timings
	every_implementation_matches_cpu
	implementations_enumerated
	gemm_matches_cpu
	norm_rope_silu_embedding_add_match_cpu
	paged_prefill_ck_128_matches_cpu
	paged_and_moe_ops
	moe_experts_small_m_matches_cpu
	moe_experts_grouped_matches_cpu
	logits_reduce_matches_cpu
	host_staging_does_not_wait_for_the_stream
	prefill_shapes_match_cpu
)

DRY_RUN=0
if [[ "${1:-}" == --dry-run ]]; then
	DRY_RUN=1
	shift
fi
[[ $# -ge 1 ]] || usage
HOST="$1"
shift
case "$HOST" in
dgx-spark) ADDR=192.168.10.246 ;;
dgx-spark2) ADDR=192.168.10.245 ;;
novanas) ADDR=192.168.10.203 ;;
*) usage ;;
esac

MODE=run
GPUS=1
GPUS_SET=0
HF_REFERENCE=0
FEATURES=""
TIER=full
STOP_RUN=""
CARGO_ARGS=()
while [[ $# -gt 0 ]]; do
	case "$1" in
	--gpus)
		[[ $# -ge 2 && "$2" =~ ^[12]$ ]] || usage
		GPUS="$2"
		GPUS_SET=1
		shift 2
		;;
	--with-hf-reference)
		HF_REFERENCE=1
		shift
		;;
	--features)
		[[ $# -ge 2 && "$2" =~ ^[A-Za-z0-9_/,-]+$ ]] || usage
		FEATURES="$2"
		shift 2
		;;
	--tier)
		[[ $# -ge 2 && "$2" =~ ^(quick|perf|full)$ ]] || usage
		TIER="$2"
		shift 2
		;;
	--stop)
		[[ $# -eq 2 && "$2" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ ]] || usage
		MODE=stop
		STOP_RUN="$2"
		shift 2
		;;
	--)
		shift
		CARGO_ARGS=("$@")
		break
		;;
	*) usage ;;
	esac
done
if [[ "$HOST" != novanas && ($MODE == stop || $GPUS_SET -eq 1) ]]; then
	usage
fi
if [[ $MODE == stop && ($GPUS_SET -eq 1 || $HF_REFERENCE -eq 1 || -n $FEATURES || $TIER != full || ${#CARGO_ARGS[@]} -gt 0) ]]; then
	usage
fi

# The test command, one argv: cargo test --no-fail-fast <selection> -- <harness args> [<extra>].
TEST_CMD=(cargo test --no-fail-fast)
build_test_command() {
	local select=() extra=() seen=0 a t
	for a in ${CARGO_ARGS[@]+"${CARGO_ARGS[@]}"}; do
		if [[ $seen -eq 0 && "$a" == -- ]]; then
			seen=1
		elif [[ $seen -eq 0 ]]; then
			select+=("$a")
		else
			extra+=("$a")
		fi
	done
	[[ ${#select[@]} -gt 0 ]] || select=(--workspace)
	[[ -z $FEATURES ]] || select+=(--features "$FEATURES")
	TEST_CMD+=("${select[@]}" -- --include-ignored --show-output)
	# The golden references are committed; the Hugging Face transformers CPU reference is only
	# needed to regenerate them.
	[[ $HF_REFERENCE -eq 1 ]] || TEST_CMD+=(--skip hf_reference_matches_cpu)
	# quick: skip the slow perf/timing tests (SLOW_TESTS above); perf: run only those (the same
	# list as bare filter args, which libtest ORs by substring); full: neither, unchanged.
	if [[ $TIER == quick ]]; then
		for t in "${SLOW_TESTS[@]}"; do
			TEST_CMD+=(--skip "$t")
		done
	elif [[ $TIER == perf ]]; then
		TEST_CMD+=("${SLOW_TESTS[@]}")
	fi
	TEST_CMD+=(${extra[@]+"${extra[@]}"})
}
build_test_command

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="piwi@${ADDR}"
CI_ROOT=/home/piwi/turbine-ci
IMAGE=rust:1.97-trixie
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10)
NS=turbine-ci
# Short, unique and DNS-1123 safe: UTC time plus 30 random bits.
RUN_ID="$(date -u +%m%d%H%M%S)-$(printf '%08x' $(((RANDOM << 15) | RANDOM)))"
[[ $MODE == stop ]] && RUN_ID="$STOP_RUN"
JOB="turbine-lab-test-${RUN_ID}"
# The GPU-less build Job that compiles into a slot before the GPU Job runs.
BUILD_JOB="turbine-lab-build-${RUN_ID}"
RUN_DIR="${CI_ROOT}/runs/${RUN_ID}"

fail() {
	echo "lab-test: ${HOST}: $1" >&2
	exit "${2:-1}"
}

say() {
	echo "lab-test: ${HOST}: $*"
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
		tr '\0' '\n'
		echo "EOF"
	else
		# Same kubectl warning filter as kube.
		ssh "${SSH_OPTS[@]}" "$REMOTE" "$@" 2> >(grep -v -e 'permission denied' >&2)
	fi
}

kube() {
	# kubectl on novanas warns about unreadable config files for user piwi. KUBECTL_KUBERC=false
	# silences the kuberc warning (it lacks a trailing newline, so a following error would share
	# its line); the remaining k3s config.yaml warnings are whole lines and only those are dropped.
	if [[ $DRY_RUN -eq 1 ]]; then
		remote "export KUBECTL_KUBERC=false; kubectl $*"
	else
		remote "export KUBECTL_KUBERC=false; kubectl $*" 2> >(grep -v -e 'permission denied' >&2)
	fi
}

upload_tree() {
	say "syncing working tree to ${1}"
	run rsync -rlpcz --delete --exclude target/ --exclude .git/ --exclude .claude/ \
		-e "ssh ${SSH_OPTS[*]}" "${REPO_ROOT}/" "${REMOTE}:${1}/" ||
		fail "rsync to ${REMOTE}:${1} failed"
}

run_spark() {
	say "preparing ${CI_ROOT}"
	remote "mkdir -p ${CI_ROOT}/src ${CI_ROOT}/target ${CI_ROOT}/cargo-registry" ||
		fail "ssh to ${REMOTE} failed"
	upload_tree "${CI_ROOT}/src"
	remote "command -v docker >/dev/null" || fail "docker is not available on ${HOST}"
	# Only our own container name is ever removed (a leftover from an aborted run).
	remote "docker rm -f turbine-lab-test >/dev/null 2>&1 || true"
	local cmd
	cmd="$(printf '%q ' "${TEST_CMD[@]}")"
	say "docker run ${IMAGE} ${cmd}"
	set +e
	remote "docker run --rm --gpus all --name turbine-lab-test --memory 32g \
    -v ${CI_ROOT}/src:/src -v ${CI_ROOT}/target:/target -v turbine-cargo:/usr/local/cargo/registry \
    -e CARGO_TARGET_DIR=/target -e TURBINE_EXPECT_NVIDIA=1 -e TURBINE_EXPECT_NVIDIA_MEMORY=unified \
    -w /src ${IMAGE} ${cmd}"
	local code=$?
	set -e
	if [[ $DRY_RUN -eq 1 ]]; then
		say "dry run: nothing contacted"
		return
	fi
	case "$code" in
	0) say "PASS" ;;
	125 | 126 | 127) fail "docker run failed (exit ${code})" "$code" ;;
	255) fail "ssh connection lost during the test run" 255 ;;
	*) fail "tests failed (exit ${code})" "$code" ;;
	esac
}

# Deletes this run's Jobs and upload, nothing else.
cleanup_novanas() {
	kube "-n ${NS} delete job ${BUILD_JOB} ${JOB} --ignore-not-found" >/dev/null || true
	remote "rm -rf ${RUN_DIR}" || true
}

on_interrupt() {
	trap - INT TERM
	echo "lab-test: ${HOST}: interrupted; deleting jobs ${BUILD_JOB} ${JOB}" >&2
	cleanup_novanas
	exit 130
}

# render_job build|test: the template as the GPU-less build Job or the GPU test Job.
render_job() {
	local gpus="$GPUS" rename=()
	if [[ $1 == build ]]; then
		gpus=0
		rename=(-e "s/turbine-lab-test-__RUN_ID__/${BUILD_JOB}/" -e "s/turbine-lab-role: test/turbine-lab-role: build/")
	fi
	sed "${rename[@]+"${rename[@]}"}" -e "s/__RUN_ID__/${RUN_ID}/g" -e "s/__GPUS__/${gpus}/g" \
		-e "s/__PHASE__/$1/g" "${REPO_ROOT}/scripts/lab/novanas-test-job.yaml"
}

# wait_for_pod <job> <gpus>: until its pod runs; an unschedulable GPU pod is given up after 120 s.
wait_for_pod() {
	local job="$1" gpus="$2"
	if [[ $DRY_RUN -eq 1 ]]; then
		echo "+ wait until pod job-name=${job} runs (unschedulable limit 120 s)"
		return
	fi
	local waited=0 phase="" unschedulable=""
	while :; do
		phase="$(kube "-n ${NS} get pods -l job-name=${job} -o jsonpath='{.items[*].status.phase}'" || true)"
		[[ "$phase" == Running || "$phase" == Succeeded || "$phase" == Failed ]] && return 0
		unschedulable="$(kube "-n ${NS} get pods -l job-name=${job} -o jsonpath='{.items[*].status.conditions[?(@.reason==\"Unschedulable\")].message}'" || true)"
		if [[ -n "$unschedulable" && $waited -ge 120 ]]; then
			cleanup_novanas
			fail "pod unschedulable for 120 s (${unschedulable}); ${gpus} amd.com/gpu is not free — another workload holds it; ask the user"
		fi
		if [[ $waited -ge 1800 ]]; then
			cleanup_novanas
			fail "pod did not start within 30 min (phase: ${phase:-none})"
		fi
		sleep 5
		waited=$((waited + 5))
	done
}

# wait_for_result <job> <what>: 0 when the Job succeeded; otherwise fails with its exit code.
wait_for_result() {
	local job="$1" what="$2"
	if [[ $DRY_RUN -eq 1 ]]; then
		echo "+ wait until job ${job} succeeds or fails; a failure exits with its exit code"
		return
	fi
	# The Job has activeDeadlineSeconds (5400), so it always reaches a terminal state.
	local succeeded="" failed=""
	while :; do
		succeeded="$(kube "-n ${NS} get job ${job} -o jsonpath='{.status.succeeded}'" || true)"
		failed="$(kube "-n ${NS} get job ${job} -o jsonpath='{.status.failed}'" || true)"
		[[ "$succeeded" == 1 || -n "$failed" ]] && break
		sleep 5
	done
	# Keep the complete pod log before anything can delete the Job: the live stream above can
	# drop mid-run, and the Job (with its log) goes away on cleanup.
	local keep="${REPO_ROOT}/target/lab-test/${RUN_ID}"
	mkdir -p "$keep" &&
		kube "-n ${NS} logs job/${job}" >"${keep}/${what}.log" 2>&1 &&
		say "full ${what} log kept at ${keep}/${what}.log"
	[[ "$succeeded" == 1 ]] && return 0
	local code
	code="$(kube "-n ${NS} get pods -l job-name=${job} -o jsonpath='{.items[*].status.containerStatuses[0].state.terminated.exitCode}'" || true)"
	[[ "$code" =~ ^[0-9]+$ && "$code" -ne 0 ]] || code=1
	[[ "$what" == build ]] && remote "rm -rf ${RUN_DIR}" || true
	fail "${what} failed (exit ${code}, job ${job})" "$code"
}

# apply_and_follow build|test <job> <gpus>: applies the rendered Job, streams its log, waits.
apply_and_follow() {
	local phase="$1" job="$2" gpus="$3" what=tests
	[[ $phase == build ]] && what=build
	say "applying scripts/lab/novanas-test-job.yaml as ${job} (${gpus} GPU(s))"
	render_job "$phase" | remote_stdin "export KUBECTL_KUBERC=false; kubectl apply -f -" ||
		fail "kubectl apply failed"
	wait_for_pod "$job" "$gpus"
	say "streaming pod log of ${job}"
	kube "-n ${NS} logs -f job/${job}" || echo "lab-test: ${HOST}: log stream ended with an error" >&2
	wait_for_result "$job" "$what"
}

run_novanas() {
	say "run ${RUN_ID}: job ${JOB}, ${GPUS} GPU(s)"
	remote "mkdir -p ${RUN_DIR}/src ${CI_ROOT}/cache/slots && find ${CI_ROOT}/runs -mindepth 1 -maxdepth 1 -mmin +1440 -exec rm -rf {} +" ||
		fail "ssh to ${REMOTE} failed"
	[[ $DRY_RUN -eq 1 ]] || trap on_interrupt INT TERM
	upload_tree "${RUN_DIR}/src"
	say "test command: ${TEST_CMD[*]}"
	local cmd_file="cat > ${RUN_DIR}/test-command"
	[[ $HF_REFERENCE -eq 1 ]] && cmd_file+=" && touch ${RUN_DIR}/hf-reference"
	printf '%s\0' "${TEST_CMD[@]}" | remote_stdin "$cmd_file" ||
		fail "writing the test command failed"

	remote "command -v kubectl >/dev/null" || fail "kubectl is not available on ${HOST}"
	kube "create namespace ${NS} --dry-run=client -o yaml | kubectl apply -f - >/dev/null" ||
		fail "cannot create namespace ${NS}"
	# Compile without a GPU claim, then hold the R9700(s) only while the tests run.
	apply_and_follow build "$BUILD_JOB" 0
	apply_and_follow test "$JOB" "$GPUS"
	if [[ $DRY_RUN -eq 1 ]]; then
		say "dry run: nothing contacted"
	else
		say "PASS (job ${JOB})"
	fi
}

stop_novanas() {
	say "deleting jobs ${BUILD_JOB} ${JOB} in namespace ${NS}"
	kube "-n ${NS} delete job ${BUILD_JOB} ${JOB} --ignore-not-found --wait=true" || fail "cannot delete jobs ${BUILD_JOB} ${JOB}"
	remote "rm -rf ${RUN_DIR}" || fail "cannot remove ${RUN_DIR}"
	if [[ $DRY_RUN -eq 1 ]]; then say "dry run: nothing contacted"; else say "stopped"; fi
}

case "$HOST:$MODE" in
novanas:run) run_novanas ;;
novanas:stop) stop_novanas ;;
*) run_spark ;;
esac
