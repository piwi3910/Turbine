#!/usr/bin/env bash
# Phase 3 overload soak (P3 S-19): calibrate -> overload -> cool-down against a freshly started
# turbine-server on a lab host, then a pass/fail verdict.
#
#   scripts/overload-soak.sh <novanas|dgx-spark|dgx-spark2> [--duration <dur>] [--model <path>]
#
# 0. precondition: prints the host, the GPU and the memory it will claim and refuses to start
#    (exit 1, `precondition`) unless an R9700 on novanas has < 1 GiB VRAM in use (amdgpu sysfs
#    mem_info_vram_used, read over SSH), or a Spark's MemAvailable exceeds the 24 GiB container
#    cap plus reliability.memory.host_reserve_bytes (8 GiB). Freeing a host is the user's
#    decision, never this script's.
# 1. start: novanas runs scripts/lab-serve.sh novanas scripts/lab/phase3-novanas-soak.yaml (a
#    k3s Job with one R9700, rust:1.97-trixie, ROCm mounted read-only, the weights read-only at
#    /models; --model <host path under /home/piwi/turbine-models> becomes model.path) and waits
#    at most 10 min for /ready. The Sparks need the Phase 2b NVIDIA serve path (docker run
#    --gpus all --memory 24g of the Phase 2b lab image); until it exists this step fails there.
# 2. calibrate (2 min): closed loop at --concurrency 4: baseline ITL p99 and request rate R.
# 3. overload (--duration, default 10m; 4h for the pre-exit run): open loop --rate 4R,
#    --prompt-words-range 64..6000, --max-tokens-range 16..1024, --concurrency 1024, with the
#    pressure timeline.
# 4. cool-down (5 min): no load; /turbine/v1/pressure sampled every second.
# 5. verdict: the server never restarted (uptime covers the whole run) and still answers;
#    5xx are only 503 with codes overloaded / queue_timeout / circuit_open; no incomplete
#    stream; overload ITL p99 <= 2 x the calibration p99; the timeline reached ORANGE or worse;
#    GREEN + HEALTHY within 60 s of the cool-down start; the KV pool back to 0 used / 0
#    reserved and the emergency reserve held at the end. Prints verdict.json; exit 0 on pass,
#    1 on fail. A trap always stops the serve Job it started (and nothing else).
#
# Outputs: target/soak/<host>-<timestamp>/{calibrate.json,overload.json,timeline.jsonl,
# cooldown.jsonl,verdict.json,serve.log}.
# Environment: SOAK_BENCH (a turbine-bench binary; default `cargo run --release` of it),
# SOAK_CALIBRATE (default 2m), SOAK_COOLDOWN_SECONDS (default 300), SOAK_SOURCE_ONLY=1 (define
# the functions and return: the tests call soak_precondition).
# Exit codes: 0 pass, 1 fail or a failed step (named), 2 usage.
set -euo pipefail

GIB=$((1 << 30))
# novanas: the VRAM another workload may hold on the R9700 the Job gets.
NOVANAS_VRAM_LIMIT=$GIB
# Sparks: the soak container's memory cap plus reliability.memory.host_reserve_bytes.
SPARK_CONTAINER_CAP=$((24 * GIB))
HOST_RESERVE=$((8 * GIB))

usage() {
	echo "usage: scripts/overload-soak.sh <novanas|dgx-spark|dgx-spark2> [--duration <dur>] [--model <path>]" >&2
	exit 2
}

# soak_precondition <host> <bytes>: novanas <bytes> = VRAM in use on the least-used R9700;
# a Spark <bytes> = MemAvailable. Returns 1 printing `precondition: ...` when the host is busy.
soak_precondition() {
	local host="$1" bytes="$2"
	[[ "$bytes" =~ ^[0-9]+$ ]] || {
		echo "precondition: ${host}: not a byte count: ${bytes}" >&2
		return 1
	}
	case "$host" in
	novanas)
		if ((bytes >= NOVANAS_VRAM_LIMIT)); then
			echo "precondition: novanas: every R9700 holds >= 1 GiB VRAM (least used: ${bytes} bytes); free one first (the user's decision)" >&2
			return 1
		fi
		;;
	dgx-spark | dgx-spark2)
		local need=$((SPARK_CONTAINER_CAP + HOST_RESERVE))
		if ((bytes <= need)); then
			echo "precondition: ${host}: MemAvailable ${bytes} bytes <= 24 GiB container cap + 8 GiB host reserve (${need} bytes); ask the user before freeing memory" >&2
			return 1
		fi
		;;
	*)
		echo "precondition: unknown host ${host}" >&2
		return 1
		;;
	esac
	echo "precondition: ${host}: ok (${bytes} bytes)"
}

if [[ "${SOAK_SOURCE_ONLY:-0}" == 1 ]]; then
	# shellcheck disable=SC2317 # `exit` runs only when the file is executed, not sourced.
	return 0 2>/dev/null || exit 0
fi

[[ $# -ge 1 ]] || usage
HOST="$1"
shift
case "$HOST" in
novanas) ADDR=192.168.10.203 ;;
dgx-spark) ADDR=192.168.10.246 ;;
dgx-spark2) ADDR=192.168.10.245 ;;
*) usage ;;
esac
DURATION=10m
MODEL=/home/piwi/turbine-models/llama-3.2-3b-instruct
while [[ $# -gt 0 ]]; do
	case "$1" in
	--duration)
		[[ $# -ge 2 && "$2" =~ ^[0-9]+(ms|s|m|h)$ ]] || usage
		DURATION="$2"
		shift 2
		;;
	--model)
		[[ $# -ge 2 && "$2" =~ ^/home/piwi/turbine-models/[A-Za-z0-9._-]+/?$ ]] || usage
		MODEL="${2%/}"
		shift 2
		;;
	*) usage ;;
	esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${REPO_ROOT}/target/soak/${HOST}-$(date -u +%Y%m%dT%H%M%SZ)"
URL="http://${ADDR}:18000"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 "piwi@${ADDR}")
CALIBRATE="${SOAK_CALIBRATE:-2m}"
COOLDOWN_SECONDS="${SOAK_COOLDOWN_SECONDS:-300}"
STEP=setup
RUN_ID=""

fail() {
	echo "overload-soak: ${HOST}: step ${STEP} failed: $1" >&2
	exit 1
}

step() {
	STEP="$1"
	echo "overload-soak: ${HOST}: step ${STEP}"
}

cleanup() {
	if [[ -n "$RUN_ID" ]]; then
		echo "overload-soak: ${HOST}: stopping serve run ${RUN_ID}" >&2
		"${REPO_ROOT}/scripts/lab-serve.sh" "$HOST" --stop "$RUN_ID" >&2 || true
	fi
}
trap cleanup EXIT

bench() {
	if [[ -n "${SOAK_BENCH:-}" ]]; then
		"$SOAK_BENCH" "$@"
	else
		(cd "$REPO_ROOT" && cargo run --release -q -p turbine-bench --bin turbine-bench -- "$@")
	fi
}

for tool in ssh curl python3; do
	command -v "$tool" >/dev/null || fail "missing dependency: ${tool}"
done
mkdir -p "$OUT"

step precondition
case "$HOST" in
novanas)
	# shellcheck disable=SC2016 # expanded by the remote shell
	used=$("${SSH[@]}" 'min=; for f in /sys/class/drm/card*/device/mem_info_vram_used; do v=$(cat "$f"); if [ -z "$min" ] || [ "$v" -lt "$min" ]; then min=$v; fi; done; echo "${min:-}"') ||
		fail "cannot read VRAM use over SSH"
	[[ -n "$used" ]] || fail "no amdgpu VRAM counters on ${HOST}"
	echo "overload-soak: novanas: GPU: one R9700 (k3s amd.com/gpu: 1); claims the reliability budget of its free VRAM; least-used R9700 holds ${used} bytes"
	soak_precondition novanas "$used" || exit 1
	;;
*)
	avail=$("${SSH[@]}" "awk '/^MemAvailable:/ {print \$2 * 1024}' /proc/meminfo") ||
		fail "cannot read MemAvailable over SSH"
	echo "overload-soak: ${HOST}: GPU: GB10 (unified); claims at most the 24 GiB container cap; MemAvailable ${avail} bytes"
	soak_precondition "$HOST" "$avail" || exit 1
	;;
esac

step start
serve_rc=0
[[ "$HOST" == novanas ]] ||
	fail "the Spark serve path (Phase 2b docker image) is not available in this tree"
TURBINE_LAB_SERVE_TIMEOUT=600 "${REPO_ROOT}/scripts/lab-serve.sh" novanas \
	"${REPO_ROOT}/scripts/lab/phase3-novanas-soak.yaml" \
	--set "model.path=/models/$(basename "$MODEL")" 2>&1 | tee "$OUT/serve.log" || serve_rc=$?
RUN_ID=$(sed -n 's/^lab-serve: novanas: run \([0-9a-f-]*\): .*/\1/p' "$OUT/serve.log" | head -n 1)
# A refused start (e.g. another server already answers on the port) must not go on to measure
# whatever answers there: that is someone else's server.
[[ "$serve_rc" -eq 0 ]] || fail "lab-serve did not start the soak's server (exit ${serve_rc}; see $OUT/serve.log)"
curl -fsS -m 5 "$URL/ready" >/dev/null || fail "turbine-server did not become ready within 10 min (see $OUT/serve.log)"
STARTED=$(date +%s)
MODEL_ID=$(curl -fsS -m 5 "$URL/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])') ||
	fail "cannot read the served model id"

step calibrate
bench --url "$URL" --model "$MODEL_ID" --concurrency 4 --duration "$CALIBRATE" \
	--prompt-words-range 64..6000 --max-tokens-range 16..1024 --output json >"$OUT/calibrate.json" ||
	fail "calibration run failed"
RATE=$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(4 * r["requests_ok"] / r["wall_seconds"])' "$OUT/calibrate.json") ||
	fail "calibration report has no request rate"
echo "overload-soak: ${HOST}: calibration: 4R = ${RATE} req/s"

step overload
bench --url "$URL" --model "$MODEL_ID" --rate "$RATE" --duration "$DURATION" \
	--prompt-words-range 64..6000 --max-tokens-range 16..1024 --concurrency 1024 \
	--pressure-timeline "$OUT/timeline.jsonl" --output json >"$OUT/overload.json" ||
	fail "overload run failed"

step cool-down
: >"$OUT/cooldown.jsonl"
COOL_START=$(date +%s)
for ((i = 0; i < COOLDOWN_SECONDS; i++)); do
	t=$(($(date +%s) - COOL_START))
	doc=$(curl -fsS -m 2 "$URL/turbine/v1/pressure" || echo '{"error":"unreachable"}')
	printf '{"t":%s,"doc":%s}\n' "$t" "$doc" >>"$OUT/cooldown.jsonl"
	sleep 1
done

step verdict
ELAPSED=$(($(date +%s) - STARTED))
STATUS=$(curl -fsS -m 5 "$URL/turbine/v1/status" || echo '{}')
python3 - "$OUT" "$ELAPSED" "$STATUS" <<'PY'
import json, sys
out, elapsed, status = sys.argv[1], int(sys.argv[2]), json.loads(sys.argv[3])
cal = json.load(open(f"{out}/calibrate.json"))
over = json.load(open(f"{out}/overload.json"))
timeline = [json.loads(l) for l in open(f"{out}/timeline.jsonl") if l.strip()]
cool = [json.loads(l) for l in open(f"{out}/cooldown.jsonl") if l.strip()]
order = ["GREEN", "YELLOW", "ORANGE", "RED", "SURVIVAL"]
allowed = {"overloaded", "queue_timeout", "circuit_open", "queue_full"}
checks = {}
checks["server_never_restarted"] = status.get("uptime_seconds", -1) >= elapsed
checks["only_503_overload_codes"] = all(
    s == "503" for s in over.get("by_status", {}) if int(s) >= 500
) and set(over.get("by_error_code", {})) <= allowed
checks["streams_complete"] = over.get("streams_incomplete", 1) == 0
p99_cal, p99_over = cal["itl_ms"]["p99"], over["itl_ms"]["p99"]
checks["itl_p99_within_2x"] = p99_over <= 2 * p99_cal
checks["reached_orange"] = any(
    order.index(t["state"]) >= 2 for t in timeline if t.get("state") in order
)
green = [
    c["t"] for c in cool
    if c["doc"].get("state") == "GREEN" and c["doc"].get("circuit", {}).get("state") == "HEALTHY"
]
checks["green_within_60s"] = bool(green) and green[0] <= 60
last = cool[-1]["doc"] if cool else {}
memory = (last.get("memory") or [{}])[0]
kv = next((p for p in memory.get("pools", []) if p.get("name") == "kv"), {})
checks["kv_idle"] = kv.get("used_bytes") == 0 and kv.get("reserved_bytes") == 0
checks["reserve_held"] = memory.get("emergency_reserve_held") is True
verdict = {
    "pass": all(checks.values()),
    "checks": checks,
    "calibration_itl_p99_ms": p99_cal,
    "overload_itl_p99_ms": p99_over,
    "green_after_cooldown_s": green[0] if green else None,
    "by_status": over.get("by_status"),
    "by_error_code": over.get("by_error_code"),
    "client_dropped": over.get("client_dropped"),
}
json.dump(verdict, open(f"{out}/verdict.json", "w"), indent=2)
print(json.dumps(verdict, indent=2))
sys.exit(0 if verdict["pass"] else 1)
PY
