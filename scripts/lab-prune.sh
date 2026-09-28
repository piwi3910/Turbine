#!/usr/bin/env bash
# lab-prune.sh — free novanas disk by deleting stale remote-cargo build caches (user decision
# 2026-09-28, "Lab disk: pruning stale remote build caches on novanas").
#
#   scripts/lab-prune.sh [--dry-run | --report] [--host <ssh-host>]
#
# Deletes the target/ of every /home/piwi/turbine-ci/remote/agent-*/ workspace that
#   - has no matching local worktree under <main checkout>/.claude/worktrees/ (nor is this
#     checkout),
#   - has had no write anywhere in the workspace for TURBINE_PRUNE_IDLE_HOURS (default 12), and
#   - is not in use (no process whose command line or working directory is inside it).
# Nothing else is ever touched: not src/, not the per-slot caches under turbine-ci/cache/, not
# runs/, not any other directory. Every removal is logged with its bytes, then the total.
# Concurrent prunes skip (a non-blocking flock). A host that cannot be reached is a warning, never
# a failure: scripts/remote-cargo.sh, lab-test.sh and lab-cluster.sh call this before each run.
#
# --dry-run prints the keep list and the remote command without contacting a host; --report
# contacts the host and prints what would be removed without removing anything.
set -uo pipefail

host="${TURBINE_REMOTE_HOST:-piwi@192.168.10.203}"
hours="${TURBINE_PRUNE_IDLE_HOURS:-12}"
base=/home/piwi/turbine-ci/remote
mode=remove
while [[ $# -gt 0 ]]; do
	case "$1" in
	--dry-run) mode=dry-run ;;
	--report) mode=report ;;
	--host)
		host="${2:?--host needs a value}"
		shift
		;;
	*)
		echo "usage: scripts/lab-prune.sh [--dry-run | --report] [--host <ssh-host>]" >&2
		exit 2
		;;
	esac
	shift
done
[[ "$hours" =~ ^[0-9]+$ && "$hours" -ge 1 ]] || {
	echo "lab-prune: TURBINE_PRUNE_IDLE_HOURS must be a whole number of hours >= 1" >&2
	exit 2
}

# The local worktrees: this checkout plus every directory under the main checkout's
# .claude/worktrees/ (the remote workspace of a checkout is named after its directory).
# Without git (e.g. a synced copy of the tree) the worktrees cannot be listed, and an
# incomplete keep list could delete a live cache: then nothing is pruned (a dry run still prints).
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if root="$(git -C "$here" rev-parse --show-toplevel 2>/dev/null)" &&
	common="$(git -C "$root" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"; then
	main="$(dirname "$common")"
else
	root="$here" main=""
	if [[ $mode != dry-run ]]; then
		echo "lab-prune: warning: not a git checkout, local worktrees unknown; nothing pruned" >&2
		exit 0
	fi
fi
keep=("$(basename "$root")")
if [[ -n "$main" ]]; then
	for d in "$main"/.claude/worktrees/*/; do
		[[ -d "$d" ]] && keep+=("$(basename "$d")")
	done
fi

# shellcheck disable=SC2016 # expanded on the host, not here
remote_script='
set -u
mode=$1 hours=$2 base=$3
shift 3
[[ -d $base ]] || exit 0
exec 9>"$base/.prune.lock"
flock -n 9 || { echo "lab-prune: another prune is running; skipped"; exit 0; }
declare -A keep=()
for k in "$@"; do keep[$k]=1; done
busy() {
	pgrep -f -- "$1/" >/dev/null 2>&1 && return 0
	local p c
	for p in /proc/[0-9]*/cwd; do
		c=$(readlink "$p" 2>/dev/null) || continue
		[[ $c == "$1" || $c == "$1"/* ]] && return 0
	done
	return 1
}
n=0 freed=0
for d in "$base"/agent-*; do
	name=${d##*/} t=$d/target
	[[ -d $t && ! -L $t && ! -L $d ]] || continue
	[[ -n ${keep[$name]:-} ]] && continue
	if [[ -n $(find "$d" -mmin -$((hours * 60)) -print -quit 2>/dev/null) ]]; then
		echo "lab-prune: keep $name (written within ${hours} h)"
		continue
	fi
	if busy "$d"; then
		echo "lab-prune: keep $name (in use)"
		continue
	fi
	bytes=$(du -sb "$t" 2>/dev/null | cut -f1)
	bytes=${bytes:-0}
	if [[ $mode == report ]]; then
		echo "lab-prune: would remove $t ($bytes bytes)"
	else
		rm -rf -- "$t" || { echo "lab-prune: could not remove $t"; continue; }
		echo "lab-prune: removed $t ($bytes bytes)"
	fi
	n=$((n + 1)) freed=$((freed + bytes))
done
verb=removed
[[ $mode == report ]] && verb="would remove"
echo "lab-prune: $verb $n stale target dir(s), $freed bytes (idle >= ${hours} h, no local worktree)"
'

if [[ $mode == dry-run ]]; then
	echo "+ lab-prune: keep ${keep[*]}"
	echo "+ ssh -o BatchMode=yes -o ConnectTimeout=10 ${host} bash -s -- remove ${hours} ${base} <keep> <<'PRUNE'"
	printf '%s\n' "$remote_script"
	echo "PRUNE"
	exit 0
fi

if ! printf '%s\n' "$remote_script" |
	ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" bash -s -- "$mode" "$hours" "$base" "${keep[@]}"; then
	echo "lab-prune: warning: prune on ${host} failed; continuing" >&2
fi
exit 0
