#!/usr/bin/env bash
# bench-lock.sh — run a command while holding the build host's benchmark lock exclusively.
#
#   scripts/bench-lock.sh [--host <ssh-host>] [--shared] <command...>
#
# Takes an exclusive flock on <host>:/home/piwi/turbine-ci/bench.lock (waiting for running
# scripts/remote-cargo.sh builds, which hold it shared, to finish), runs <command> locally, then
# releases the lock. While it is held no new remote-cargo build starts, so a benchmark against a
# lab server measures the engine, not the CPU contention of parallel builds. Exits with the
# command's exit code. --shared takes the lock shared instead: wrap scripts/lab-test.sh runs in it
# so their in-Job builds also wait while a benchmark holds the lock. Environment: TURBINE_REMOTE_HOST (default piwi@192.168.10.203, novanas).
set -euo pipefail

host="${TURBINE_REMOTE_HOST:-piwi@192.168.10.203}"
if [[ "${1:-}" == "--host" ]]; then
	host="$2"
	shift 2
fi
mode="-x"
if [[ "${1:-}" == "--shared" ]]; then
	mode="-s"
	shift
fi
if [[ $# -eq 0 ]]; then
	echo "usage: scripts/bench-lock.sh [--host <ssh-host>] [--shared] <command...>" >&2
	exit 2
fi

dir="$(mktemp -d)"
trap 'rm -rf "$dir"' EXIT
mkfifo "$dir/in" "$dir/out"
# The remote side prints "locked" once it holds the lock and keeps it until its stdin closes.
ssh -o BatchMode=yes "$host" \
	"flock $mode /home/piwi/turbine-ci/bench.lock sh -c 'echo locked; cat >/dev/null'" \
	<"$dir/in" >"$dir/out" &
lock_pid=$!
exec 3>"$dir/in"
echo "bench-lock: waiting for $host benchmark lock" >&2
read -r state <"$dir/out"
if [[ "$state" != "locked" ]]; then
	echo "bench-lock: could not take the lock on $host" >&2
	exit 1
fi
echo "bench-lock: holding $host benchmark lock" >&2

rc=0
"$@" || rc=$?
exec 3>&-
wait "$lock_pid" || true
echo "bench-lock: released" >&2
exit "$rc"
