#!/usr/bin/env bash
# fixture-order.sh — run ON novanas: gives the CPU fixture queue (`fixture.lock`) an explicit order.
#
#   nohup scripts/lab/fixture-order.sh [<queue-file>] >>/home/piwi/turbine-ci/fixture-order.log 2>&1 &
#
# flock hands a freed lock to any waiter. This loop keeps every waiting
# `flock …/fixture.lock …` process stopped (SIGSTOP) except the best-ranked one, so that one
# takes the lock next. Rank = the first line of <queue-file> (default
# /home/piwi/turbine-ci/fixture.queue; one extended regex per line, `#` comments, re-read every
# pass) that matches the waiter's command line; unmatched waiters come after every listed one,
# oldest first. The lock holder (a flock process with a child) and its job are never touched.
# fixture-pause.sh continues every fixture process after a bench; the next pass (2 s) stops the
# lower-ranked waiters again. A waiter wrapped in `timeout … flock` keeps its timeout running
# while stopped. Logs one line per change of the running waiter. Stop it with
# `pkill -f scripts/lab/fixture-order.sh` — that also continues every waiter it stopped.
set -uo pipefail

queue="${1:-/home/piwi/turbine-ci/fixture.queue}"
lock=/home/piwi/turbine-ci/fixture.lock

waiters() {
	# "<pid> <cmdline>" for every flock process on fixture.lock with no child (not holding it).
	local pid cmd
	for pid in $(pgrep -u "$(id -u)" -x flock); do
		cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null) || continue
		[[ "$cmd" == *"$lock"* ]] || continue
		pgrep -P "$pid" >/dev/null && continue
		echo "$pid $cmd"
	done
}

rank() {
	local cmd="$1" n=0 pat
	while IFS= read -r pat; do
		[[ -z "$pat" || "$pat" == \#* ]] && continue
		if [[ "$cmd" =~ $pat ]]; then
			echo "$n"
			return
		fi
		n=$((n + 1))
	done <"$queue"
	echo 9999
}

release_all() {
	local pid _
	while read -r pid _; do kill -CONT "$pid" 2>/dev/null; done < <(waiters)
	echo "fixture-order: $(date '+%F %T') stopped; all waiters continued"
	exit 0
}
trap release_all INT TERM

echo "fixture-order: $(date '+%F %T') started (queue $queue)"
last=""
while true; do
	best="" best_rank=99999 all=()
	while read -r pid cmd; do
		[[ -n "$pid" ]] || continue
		all+=("$pid")
		r=$(rank "$cmd")
		# pgrep lists pids ascending, so on a tie the oldest (lowest pid) wins.
		if ((r < best_rank)); then
			best=$pid best_rank=$r
		fi
	done < <(waiters)
	for pid in "${all[@]}"; do
		if [[ "$pid" == "$best" ]]; then
			kill -CONT "$pid" 2>/dev/null
		else
			kill -STOP "$pid" 2>/dev/null
		fi
	done
	if [[ "$best" != "$last" ]]; then
		[[ -n "$best" ]] && echo "fixture-order: $(date '+%F %T') next pid $best (rank $best_rank) of ${#all[@]}: $(tr '\0' ' ' <"/proc/$best/cmdline" 2>/dev/null | cut -c1-240)"
		last=$best
	fi
	sleep 2
done
