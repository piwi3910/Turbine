#!/usr/bin/env bash
# fixture-pause.sh — run a command while the build host's CPU fixture jobs are paused.
#
#   scripts/fixture-pause.sh [--host <ssh-host>] <command...>
#
# Stops (SIGSTOP) every process of piwi on <host> whose command line names `scripts/golden/`
# (the transformers reference, spread and dequantization jobs of the Phase 6 fixtures), runs
# <command> locally, then continues them (SIGCONT) — also when <command> fails or is
# interrupted. A running fixture job touches pages the host swapped out, and the swap-in makes
# a lab server's pressure controller reject requests (SURVIVAL on `swap_in_rate`), so serving
# benchmarks and evals pause them (scripts/lab-bench.sh does). Exits with the command's exit
# code. Environment: TURBINE_REMOTE_HOST (default piwi@192.168.10.203, novanas).
set -uo pipefail

host="${TURBINE_REMOTE_HOST:-piwi@192.168.10.203}"
if [[ "${1:-}" == "--host" ]]; then
	host="$2"
	shift 2
fi
if [[ $# -eq 0 ]]; then
	echo "usage: scripts/fixture-pause.sh [--host <ssh-host>] <command...>" >&2
	exit 2
fi

resume() {
	ssh -o BatchMode=yes "$host" "pkill -CONT -u piwi -f 'scripts/[g]olden/'" >/dev/null 2>&1 || true
}
trap resume EXIT INT TERM
paused=$(ssh -o BatchMode=yes "$host" "pgrep -u piwi -f 'scripts/[g]olden/' | wc -l; pkill -STOP -u piwi -f 'scripts/[g]olden/'" 2>/dev/null | head -1)
echo "fixture-pause: paused ${paused:-0} fixture process(es) on $host" >&2
rc=0
"$@" || rc=$?
resume
trap - EXIT INT TERM
echo "fixture-pause: resumed" >&2
exit "$rc"
