#!/usr/bin/env bash
# remote-cargo.sh — run cargo for this checkout on a lab build host instead of the local machine.
#
#   scripts/remote-cargo.sh [--host <ssh-host>] <cargo args...>
#   scripts/remote-cargo.sh [--host <ssh-host>] --clean
#
# Syncs the sources of the current checkout (git worktree root; no target/, .git/ or .claude/) to
# <host>:/home/piwi/turbine-ci/remote/<checkout name>/src and runs `cargo +<toolchain> <args>` there
# with CARGO_TARGET_DIR=<...>/target, so build artifacts live on the build host's disk. Changed files
# get a fresh mtime on the host (checksum comparison, no time preservation), so cargo never mistakes
# an edited file for an already-built one. Exits with cargo's exit code. `--clean` deletes this
# checkout's remote directory.
#
# Environment: TURBINE_REMOTE_HOST (default piwi@192.168.10.203, novanas), TURBINE_REMOTE_TOOLCHAIN
# (default 1.97), TURBINE_REMOTE_JOBS (cargo build jobs, default 6: several checkouts share 16 cores).
# Only builds and host-side tests run here: GPU tests stay #[ignore]d and run through lab-test.sh.
set -euo pipefail

host="${TURBINE_REMOTE_HOST:-piwi@192.168.10.203}"
toolchain="${TURBINE_REMOTE_TOOLCHAIN:-1.97}"
jobs="${TURBINE_REMOTE_JOBS:-6}"
if [[ "${1:-}" == "--host" ]]; then
	host="$2"
	shift 2
fi
if [[ $# -eq 0 ]]; then
	echo "usage: scripts/remote-cargo.sh [--host <ssh-host>] <cargo args...> | --clean" >&2
	exit 2
fi

root="$(git rev-parse --show-toplevel)"
name="$(basename "$root")"
remote="/home/piwi/turbine-ci/remote/$name"

if [[ "$1" == "--clean" ]]; then
	ssh -o BatchMode=yes "$host" "rm -rf '$remote'"
	echo "remote-cargo: removed $host:$remote"
	exit 0
fi

ssh -o BatchMode=yes "$host" "mkdir -p '$remote/src' '$remote/target'"
rsync -rlpc --delete \
	--exclude /target --exclude /.git --exclude /.claude --exclude node_modules \
	"$root/" "$host:$remote/src/"

args=""
for a in "$@"; do
	args+=" $(printf '%q' "$a")"
done
exec ssh -o BatchMode=yes "$host" \
	"cd '$remote/src' && CARGO_TARGET_DIR='$remote/target' CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=$jobs \
   CARGO_TERM_COLOR=never nice -n 5 \$HOME/.cargo/bin/cargo +$toolchain$args"
