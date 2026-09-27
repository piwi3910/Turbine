#!/usr/bin/env bash
# gate.sh — the commit gate: fast locally, then clippy + a diff-scoped test run on novanas.
#
#   scripts/gate.sh [--full] [--base <rev>]
#
# 1. `cargo fmt --all --check`, locally (no build, no network).
# 2. On novanas via scripts/remote-cargo.sh: `cargo clippy --workspace --all-targets -- -D
#    warnings`. Clippy always runs over the whole workspace — with a warm cache on novanas it is
#    fast enough that scoping it would not save meaningful time, and it catches cross-crate
#    lint fallout a scoped run would miss.
# 3. Tests: the crates whose files changed between --base (default: the merge-base of HEAD and
#    main) and HEAD, plus every crate that (transitively, including dev-dependencies) depends on
#    one of them — computed from `cargo metadata --no-deps`, so a change to a leaf crate like
#    turbine-tensor also re-tests turbine-kernels, turbine-model, turbine-server, and so on. A
#    change to a workspace-wide file (root Cargo.toml/Cargo.lock, rust-toolchain*, .cargo/**)
#    or `--full` runs the whole workspace instead. Runs with `cargo nextest` if it is installed
#    on novanas (this script installs it there once via plain ssh if missing, and falls back to
#    `cargo test` if that install fails). Only host-side tests run here: GPU tests stay
#    #[ignore]d (scripts/lab-test.sh runs those).
#
# Prints one line: "gate: ok|FAIL crates=<list> passed=N failed=M". Exit code is the failure
# (0 on a clean gate, matching the last failing step's exit code otherwise).
set -euo pipefail

usage() {
	echo "usage: scripts/gate.sh [--full] [--base <rev>]" >&2
	exit 2
}

FULL=0
BASE_ARG=""
while [[ $# -gt 0 ]]; do
	case "$1" in
	--full)
		FULL=1
		shift
		;;
	--base)
		[[ $# -ge 2 ]] || usage
		BASE_ARG="$2"
		shift 2
		;;
	*) usage ;;
	esac
done

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

HOST="${TURBINE_REMOTE_HOST:-piwi@192.168.10.203}"
TOOLCHAIN="${TURBINE_REMOTE_TOOLCHAIN:-1.97}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10)

say() { echo "gate: $*" >&2; }

# Prints the final one-line summary and exits with the gate's verdict.
finish() {
	local status="$1" crates="$2" passed="$3" failed="$4" code="$5"
	echo "gate: ${status} crates=${crates} passed=${passed} failed=${failed}"
	exit "$code"
}

# 1. fmt check, locally.
say "cargo fmt --all --check"
if ! cargo fmt --all --check; then
	finish FAIL "-" 0 0 1
fi

# 2. Which crates does the diff touch, plus their reverse dependencies?
CRATES_DISPLAY="all"
declare -a TEST_SELECT=(--workspace)
if [[ $FULL -eq 0 ]]; then
	BASE="${BASE_ARG:-$(git merge-base HEAD main)}"
	say "diff scope: ${BASE}...HEAD"
	SELECTION="$(
		python3 - "$BASE" "$ROOT" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

base, root = sys.argv[1], Path(sys.argv[2])

diff = subprocess.run(
    ["git", "diff", "--name-only", f"{base}...HEAD"],
    cwd=root, capture_output=True, text=True, check=True,
).stdout.splitlines()

meta = json.loads(subprocess.run(
    ["cargo", "metadata", "--no-deps", "--format-version", "1"],
    cwd=root, capture_output=True, text=True, check=True,
).stdout)

# name -> "relative/dir/" (with a trailing slash, so prefix matching cannot cross a boundary
# like crates/turbine-kv vs crates/turbine-kvx).
members = {}
for pkg in meta["packages"]:
    reldir = Path(pkg["manifest_path"]).parent.relative_to(root).as_posix()
    members[pkg["name"]] = reldir + "/"

# Reverse-dependency graph over workspace members only, normal and dev edges alike: a dev-only
# edge (e.g. turbine-api's tests use turbine-reliability) still means that crate's tests can
# break when the dependency changes.
rdeps = {name: set() for name in members}
for pkg in meta["packages"]:
    if pkg["name"] not in members:
        continue
    for dep in pkg["dependencies"]:
        if dep["name"] in members:
            rdeps[dep["name"]].add(pkg["name"])

# Paths outside crates/*/ and benches/turbine-bench/ that still map to a crate's tests.
EXTRA_MAP = [
    ("kernels/", "turbine-kernels"),
    ("tests/golden/", "turbine-model"),
    ("scripts/lab/", "turbine-bench"),
    ("examples/", "turbine-bench"),
]
WHOLE_FILES = {"Cargo.toml", "Cargo.lock"}
WHOLE_PREFIXES = (".cargo/", "rust-toolchain")

direct = set()
for f in diff:
    if f in WHOLE_FILES or f.startswith(WHOLE_PREFIXES):
        print("ALL")
        sys.exit(0)
    hit = False
    for name, reldir in members.items():
        if f.startswith(reldir):
            direct.add(name)
            hit = True
            break
    if hit:
        continue
    for prefix, name in EXTRA_MAP:
        if f.startswith(prefix):
            direct.add(name)
            break
    # Anything else (docs, .procoder/**, AGENTS.md, other scripts, ...) has no Rust-test effect.

closure = set(direct)
frontier = list(direct)
while frontier:
    n = frontier.pop()
    for dep in rdeps.get(n, ()):
        if dep not in closure:
            closure.add(dep)
            frontier.append(dep)

print(" ".join(sorted(closure)))
PY
	)"
	if [[ "$SELECTION" == "ALL" ]]; then
		CRATES_DISPLAY="all"
		TEST_SELECT=(--workspace)
	elif [[ -z "$SELECTION" ]]; then
		CRATES_DISPLAY="none"
		TEST_SELECT=()
	else
		CRATES_DISPLAY="$(echo "$SELECTION" | tr ' ' ,)"
		TEST_SELECT=()
		for c in $SELECTION; do
			TEST_SELECT+=(-p "$c")
		done
	fi
fi
say "crates: ${CRATES_DISPLAY}"

# 2b. clippy, always whole-workspace.
say "cargo clippy --workspace --all-targets -- -D warnings (on ${HOST})"
if ! scripts/remote-cargo.sh clippy --workspace --all-targets -- -D warnings; then
	finish FAIL "$CRATES_DISPLAY" 0 0 1
fi

# 3. Nothing to test (a diff that touched no Rust-affecting file).
if [[ $FULL -eq 0 && "$CRATES_DISPLAY" == "none" ]]; then
	finish ok "$CRATES_DISPLAY" 0 0 0
fi

# cargo-nextest, installed once on novanas; falls back to cargo test if that install fails.
NEXTEST=0
say "checking cargo-nextest on ${HOST}"
if ssh "${SSH_OPTS[@]}" "$HOST" "test -x \$HOME/.cargo/bin/cargo-nextest" 2>/dev/null; then
	NEXTEST=1
else
	say "cargo-nextest not found on ${HOST}; installing once"
	# TOOLCHAIN is meant to expand locally (it names our toolchain); HOME must expand remotely.
	# shellcheck disable=SC2029
	if ssh "${SSH_OPTS[@]}" "$HOST" "\$HOME/.cargo/bin/cargo +${TOOLCHAIN} install cargo-nextest --locked"; then
		NEXTEST=1
	else
		say "cargo-nextest install failed; falling back to cargo test"
		NEXTEST=0
	fi
fi

LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT
RC=0
if [[ $NEXTEST -eq 1 ]]; then
	say "cargo nextest run ${TEST_SELECT[*]:-} (on ${HOST})"
	scripts/remote-cargo.sh nextest run "${TEST_SELECT[@]}" 2>&1 | tee "$LOG" || RC=1
else
	say "cargo test --no-fail-fast ${TEST_SELECT[*]:-} (on ${HOST})"
	scripts/remote-cargo.sh test --no-fail-fast "${TEST_SELECT[@]}" 2>&1 | tee "$LOG" || RC=1
fi

PASSED="$(grep -oE '[0-9]+ passed' "$LOG" | awk '{s+=$1} END{print s+0}')"
FAILED="$(grep -oE '[0-9]+ failed' "$LOG" | awk '{s+=$1} END{print s+0}')"

if [[ $RC -ne 0 || $FAILED -ne 0 ]]; then
	finish FAIL "$CRATES_DISPLAY" "$PASSED" "$FAILED" 1
fi
finish ok "$CRATES_DISPLAY" "$PASSED" "$FAILED" 0
