#!/usr/bin/env bash
# fetch-checkpoint.sh — one pinned Phase 7 checkpoint download on novanas (P7 S-17).
#
#   scripts/fetch-checkpoint.sh [--dry-run] novanas <slug>
#
# Reads the slug's row of scripts/lab/phase7-checkpoints.tsv (slug, repo, revision, include
# globs, GB), refuses (exit 1, `fetch: not enough disk`) when the free space under
# /home/piwi/turbine-models is below the row's GB + 20, else runs on novanas
#   hf download <repo> --revision <rev> --local-dir /home/piwi/turbine-models/<slug> [--include <glob>]…
# and prints `fetch: <slug> <rev> ok`. The download uses the user's cached `hf auth login` on
# novanas: this script never reads, prints or passes a token. Ask the user before each download
# (AGENTS.md lab rules) and run it detached (it takes minutes).
#
# --dry-run prints the remote command and contacts nothing. Exit: 0 ok, 1 disk or download
# failure, 2 usage (unknown host or slug).
set -euo pipefail

usage() {
	echo "usage: scripts/fetch-checkpoint.sh [--dry-run] novanas <slug>" >&2
	exit 2
}

dry_run=0
if [[ ${1:-} == --dry-run ]]; then
	dry_run=1
	shift
fi
[[ $# -eq 2 ]] || usage
[[ $1 == novanas ]] || usage
slug=$2

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
table="$here/lab/phase7-checkpoints.tsv"
row="$(awk -F'\t' -v s="$slug" '$1 !~ /^#/ && $1 == s { print; exit }' "$table")"
if [[ -z $row ]]; then
	echo "fetch: unknown slug '$slug' (rows of scripts/lab/phase7-checkpoints.tsv:" \
		"$(awk -F'\t' '$1 !~ /^#/ { printf "%s ", $1 }' "$table"))" >&2
	exit 2
fi
IFS=$'\t' read -r _ repo rev include gb <<<"$row"
[[ $rev =~ ^[0-9a-f]{40}$ && $gb =~ ^[0-9]+$ ]] || {
	echo "fetch: malformed row for '$slug' in $table" >&2
	exit 2
}

host="piwi@192.168.10.203"
models=/home/piwi/turbine-models
dest="$models/$slug"
need_kib=$(((gb + 20) * 1024 * 1024))

download="hf download $repo --revision $rev --local-dir $dest"
if [[ $include != '*' ]]; then
	IFS=, read -r -a globs <<<"$include"
	for g in "${globs[@]}"; do
		if [[ $g == *[*?[]* ]]; then
			download+=" --include '$g'"
		else
			download+=" --include $g"
		fi
	done
fi

# The remote command: a disk check, then the download (non-interactive ssh: add the usual
# per-user install locations of `hf` to PATH).
remote="set -euo pipefail
export PATH=\"\$HOME/.local/bin:\$PATH\"
avail=\$(df --output=avail $models | tail -n 1 | tr -d ' ')
if (( avail < $need_kib )); then
	echo \"fetch: not enough disk (\$((avail / 1024 / 1024)) GiB free under $models, need $((gb + 20)))\" >&2
	exit 1
fi
mkdir -p $dest
$download"

if ((dry_run)); then
	echo "ssh $host bash -s <<'EOF'"
	echo "$remote"
	echo "EOF"
	exit 0
fi

if ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" bash -s <<<"$remote"; then
	echo "fetch: $slug $rev ok"
else
	status=$?
	echo "fetch: $slug $rev FAIL (exit $status)" >&2
	exit 1
fi
