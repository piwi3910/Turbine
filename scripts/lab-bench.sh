#!/usr/bin/env bash
# lab-bench.sh — fast land-and-measure step on novanas without a k8s Job.
#
#   scripts/lab-bench.sh [--gpu 0] [--model <model>] [--label L] [--with-tests] [--skip-tests]
#                         [--golden16] [--c1] [--quality] [--quick] [-- <--set k=v>...]
#   scripts/lab-bench.sh --print-model <model>
#
# <model>: llama|olmoe|llama-fp8|llama-fp8-tensor|llama-fp8-block|llama-awq|llama-gptq|llama8b-mxfp4|llama8b|llama-mxfp4-a4|llama-yarn16|llama-fp8kv|olmoe-fp8kv
# (Phase 6a: one value per proof checkpoint; each maps to a weights directory under
# /home/piwi/turbine-models/, a golden slug under tests/golden/ and a lab config).
# --print-model prints "<weights> <golden slug> <config>" and exits without contacting a host.
#
# 1. On novanas: the release turbine-server build (scripts/remote-cargo.sh, cached) plus the HIP
#    kernel library (CMake, cached); workspace tests only with --with-tests (off by default:
#    a bench-only landing step should not pay for a full test build; --skip-tests is accepted as
#    a no-op alias for that same default, kept for compatibility with older call sites).
# 2. Starts the server directly on the host, pinned to GPU 0 with ROCR_VISIBLE_DEVICES, so every
#    step runs on the same card. GPU 0 is the only card with a throughput-comparable PCIe link;
#    --gpu only accepts 0 and refuses anything else with a clear message, rather than silently
#    producing numbers that are not comparable with earlier runs.
# 3. Under the exclusive benchmark lock: golden at concurrency 1 (the gate; always), golden at 16
#    only with --golden16 (opt-in: it roughly doubles the golden time for a number this step does
#    not gate on), and the fixed throughput bench (16 concurrent, 512-word prompts, 256 tokens;
#    200 requests, or 64 with --quick for a faster read during iteration); with --c1 also the
#    single-request latency run (10 requests of 128 tokens, --ignore-eos), reported as itl_c1_p50
#    and tok/s_c1 (Phase 6a targets stated in c1 ITL). With --quality, after the measurement and
#    outside the benchmark lock (the port stays this run's), the accuracy task set
#    (turbine-golden, tests/eval/gsm8k-200.jsonl, greedy, one request at a time) against the
#    served model: quality.json in the results (commit it as tests/eval/<slug>/turbine.json) and
#    gsm8k=<accuracy> on the BENCH line (the Phase 6a S-17 quality gate).
# 4. Stops the server and prints one "BENCH ..." summary line (quick=1 added when --quick was
#    used; golden16=SKIP when --golden16 was not); exits 1 if tests, golden c1 or the bench fail.
#    Results land in target/lab-bench/<label>/ on the workstation.
set -uo pipefail
orig_args=("$@")

host="${TURBINE_REMOTE_HOST:-piwi@192.168.10.203}"
gpu=0
model=llama
label="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
run_tests=0
run_golden16=0
run_c1=0
run_quality=0
quick=0
print_model=0
usage() {
	echo "usage: scripts/lab-bench.sh [--gpu 0] [--model <model>] [--label L] [--with-tests] [--skip-tests] [--golden16] [--c1] [--quality] [--quick] [-- --set k=v ...] | --print-model <model>" >&2
	echo "models: llama olmoe llama-fp8 llama-fp8-tensor llama-fp8-block llama-awq llama-gptq llama8b-mxfp4 llama8b llama-mxfp4-a4 llama-yarn16 llama-fp8kv olmoe-fp8kv" >&2
	exit 2
}
while [[ $# -gt 0 ]]; do
	case "$1" in
	--gpu)
		gpu="$2"
		shift 2
		;;
	--model)
		model="$2"
		shift 2
		;;
	--label)
		label="$2"
		shift 2
		;;
	--with-tests)
		run_tests=1
		shift
		;;
	--skip-tests)
		# No-op alias: tests are already off by default. Kept for older call sites.
		run_tests=0
		shift
		;;
	--golden16)
		run_golden16=1
		shift
		;;
	--c1)
		run_c1=1
		shift
		;;
	--quality)
		run_quality=1
		shift
		;;
	--quick)
		quick=1
		shift
		;;
	--print-model)
		[[ $# -ge 2 ]] || usage
		model="$2"
		print_model=1
		shift 2
		;;
	--)
		shift
		break
		;;
	*) usage ;;
	esac
done
# GPU 0 is the only card perf numbers are comparable across: GPU 1's PCIe link is fixed at a
# different (lower) bandwidth, so a throughput run there would silently mislead. lab-bench.sh is
# a throughput/perf tool end to end, so this refuses rather than warns.
if [[ "$gpu" != 0 ]]; then
	echo "lab-bench: --gpu $gpu refused: throughput and golden numbers must come from GPU 0 (GPU 1's PCIe link is not comparable); functional-only lab-test.sh runs may use GPU 1" >&2
	exit 2
fi
# model → weights directory (under /home/piwi/turbine-models), golden slug (under tests/golden)
# and lab config. The BF16 models keep their phase2c configs so baselines stay comparable; each
# Phase 6a checkpoint gets scripts/lab/phase6-novanas-<model>.yaml in the task that proves it.
golden_slug=""
case "$model" in
llama)
	slug=llama-3.2-3b-instruct
	cfg=scripts/lab/phase2c-novanas-llama.yaml
	;;
olmoe)
	slug=olmoe-1b-7b-0125-instruct
	cfg=scripts/lab/phase2c-novanas-olmoe.yaml
	;;
llama-fp8) slug=llama-3.2-3b-instruct-fp8-dynamic ;;
llama-fp8-tensor) slug=llama-3.2-3b-instruct-fp8 ;;
llama-fp8-block) slug=llama-3.2-3b-instruct-fp8-block ;;
llama-awq) slug=llama-3.2-3b-instruct-awq ;;
llama-gptq) slug=llama-3.2-3b-instruct-gptq ;;
llama8b-mxfp4) slug=llama-3.1-8b-instruct-mxfp4a16 ;;
llama8b) slug=llama-3.1-8b-instruct ;;
llama-mxfp4-a4) slug=llama-3.2-3b-mxfp4-a4 ;;
llama-yarn16)
	slug=llama-3.2-3b-instruct
	golden_slug=llama-3.2-3b-instruct-yarn16
	;;
# FP8 KV (Phase 6a S-13): the BF16 checkpoints, `kv.dtype: fp8_e4m3` in their configs, judged
# against a reference with FP8 KV emulated.
llama-fp8kv)
	slug=llama-3.2-3b-instruct
	golden_slug=llama-3.2-3b-instruct-fp8kv
	;;
olmoe-fp8kv)
	slug=olmoe-1b-7b-0125-instruct
	golden_slug=olmoe-1b-7b-0125-instruct-fp8kv
	;;
*)
	echo "lab-bench: unknown model $model" >&2
	usage
	;;
esac
[[ -n "$golden_slug" ]] || golden_slug="$slug"
[[ -n "${cfg:-}" ]] || cfg="scripts/lab/phase6-novanas-$model.yaml"
if [[ $print_model -eq 1 ]]; then
	echo "$slug $golden_slug $cfg"
	exit 0
fi
requests=200
[[ $quick -eq 1 ]] && requests=64

# Path-based, like lab-test.sh/lab-serve.sh's REPO_ROOT: matches this checkout even when it has
# no .git (e.g. run from a remote-cargo.sh-synced tree, which excludes .git).
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root" || exit 1
if [[ ! -f "$cfg" ]]; then
	echo "lab-bench: $cfg does not exist yet (written by the Phase 6a task that proves $model)" >&2
	exit 2
fi
name="$(basename "$root")"
remote="/home/piwi/turbine-ci/remote/$name"
out="$root/target/lab-bench/$label-$model"
mkdir -p "$out"
commit="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
url=http://192.168.10.203:18000

# The whole serve-and-measure run holds the host's `port18000` lock (scripts/bench-lock.sh
# --name), so a second lab-bench waits instead of racing this one for the port.
if [[ -z "${LAB_BENCH_PORT_LOCKED:-}" ]]; then
	LAB_BENCH_PORT_LOCKED=1 exec scripts/bench-lock.sh --host "$host" --name port18000 \
		"$root/scripts/lab-bench.sh" "${orig_args[@]}"
fi
# 1. build (server + kernels) on novanas, tests alongside only with --with-tests
(
	scripts/remote-cargo.sh build -q --release -p turbine-server -p turbine-bench &&
		ssh -o BatchMode=yes "$host" "cd '$remote/src' && flock -s /home/piwi/turbine-ci/bench.lock \
      cmake -S kernels/rocm -B '$remote/kbuild' -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_HIP_COMPILER=/opt/rocm/rocm/bin/hipcc -DGPU_TARGETS=gfx1201 >/dev/null &&
      flock -s /home/piwi/turbine-ci/bench.lock cmake --build '$remote/kbuild' >/dev/null"
) >"$out/build.log" 2>&1 &
build_pid=$!
if [[ $run_tests -eq 1 ]]; then
	scripts/remote-cargo.sh test --workspace --no-fail-fast >"$out/tests.log" 2>&1 &
	test_pid=$!
fi
wait "$build_pid"
build_rc=$?
tests="skipped"
if [[ $run_tests -eq 1 ]]; then
	wait "$test_pid"
	tests=$(grep -E '^test result' "$out/tests.log" | awk '{p+=$4;f+=$6} END {print p"/"f}')
fi
if [[ $build_rc -ne 0 ]]; then
	echo "BENCH $label $model commit=$commit BUILD FAILED (see $out/build.log)"
	exit 1
fi
if ! ssh -o BatchMode=yes "$host" "test -f /home/piwi/turbine-models/$slug/config.json"; then
	echo "lab-bench: /home/piwi/turbine-models/$slug is missing on novanas (download it first)" >&2
	exit 1
fi

# 2. serve natively, pinned to one card
# CPU fixture jobs (scripts/golden/*) are paused from here to the end (scripts/fixture-pause.sh
# explains why: their swap-in pushes the server's pressure controller into SURVIVAL).
# From here to the end the run also holds bench.lock exclusively (the gate first, as
# scripts/bench-lock.sh does): GPU 0 serves this run only, so kernel evaluations (bench-lock),
# lab-test and lab-serve Jobs (bench-lock --shared) and builds (flock -s) wait for it.
lockdir="$(mktemp -d)"
mkfifo "$lockdir/in" "$lockdir/out"
ssh -o BatchMode=yes "$host" \
	"flock -x /home/piwi/turbine-ci/bench.gate flock -x /home/piwi/turbine-ci/bench.lock sh -c 'echo locked; cat >/dev/null'" \
	<"$lockdir/in" >"$lockdir/out" &
exec 7>"$lockdir/in"
echo "lab-bench: waiting for the exclusive bench lock" >&2
read -r lock_state <"$lockdir/out"
if [[ "$lock_state" != locked ]]; then
	echo "lab-bench: could not take the bench lock on $host" >&2
	exit 1
fi
ssh -o BatchMode=yes "$host" "pkill -STOP -u piwi -f 'scripts/[g]olden/'" >/dev/null 2>&1 || true
trap 'ssh -o BatchMode=yes "$host" "pkill -CONT -u piwi -f '"'"'scripts/[g]olden/'"'"'" >/dev/null 2>&1 || true; exec 7>&-; rm -rf "$lockdir"' EXIT
# Only this run's own server is ever stopped: its pid is kept in $pidf and checked to still be
# a turbine-server before the kill (a blanket pkill would also end other agents' test servers
# running as piwi). A port already served is refused, not taken over.
pidf=/tmp/lab-bench-server.pid
# Server and client run on cores 0-11: CPU fixture jobs are pinned to 12-15 (AGENTS.md lab rules).
bench_cpus=${TURBINE_LAB_BENCH_CPUS:-0-11}
stop_server="if [ -f $pidf ]; then p=\$(cat $pidf); \
  if [ \"\$(cat /proc/\$p/comm 2>/dev/null)\" = turbine-server ]; then kill \$p; \
    for i in 1 2 3 4 5 6 7 8 9 10; do [ -d /proc/\$p ] || break; sleep 1; done; fi; \
  rm -f $pidf; fi"
# Another run's server (live pid file) or anything else on the port: wait for it to go, up
# to TURBINE_LAB_BENCH_PORT_WAIT seconds (default 3 h, naming the holder every 10 min), never
# kill it (a leftover lab-bench server is stopped deliberately: kill "$(cat $pidf)" on the host).
port_wait=${TURBINE_LAB_BENCH_PORT_WAIT:-10800}
waited=0
while ssh -o BatchMode=yes "$host" "ss -ltnH 'sport = :18000' | grep -q ."; do
	holder=$(ssh -o BatchMode=yes "$host" "cat $pidf 2>/dev/null || echo none")
	if ((waited >= port_wait)); then
		echo "lab-bench: port 18000 on $host still served after ${waited}s (pid file $pidf: $holder); stop that server first" >&2
		exit 1
	fi
	((waited % 600 == 0)) && echo "lab-bench: port 18000 on $host is served (pid file $pidf: $holder); waiting (${waited}s of ${port_wait}s)" >&2
	sleep 60
	waited=$((waited + 60))
done
ssh -o BatchMode=yes "$host" "rm -f $pidf"
ssh -o BatchMode=yes "$host" "setsid bash -c \"echo \\\$\\\$ > $pidf; cd '$remote/src' && \
    LD_LIBRARY_PATH=/opt/rocm/rocm/lib:/opt/rocm/rocm/lib/rocm_sysdeps/lib \
    TURBINE_AMD_SMI_LIBRARY=/opt/rocm/rocm/lib/libamd_smi.so.26.5.0 ROCR_VISIBLE_DEVICES=$gpu \
    exec taskset -c $bench_cpus '$remote/target/release/turbine-server' --config $cfg \
      --set model.path=/home/piwi/turbine-models/$slug \
      --set execution.kernel_library='$remote/kbuild/libturbine_hip.so' $*\" \
    > /tmp/lab-bench-server.log 2>&1 < /dev/null &"
ready=0
for _ in $(seq 1 150); do
	if [[ "$(curl -s -o /dev/null -w '%{http_code}' "$url/ready")" == 200 ]]; then
		ready=1
		break
	fi
	sleep 2
done
if [[ $ready -ne 1 ]]; then
	ssh -o BatchMode=yes "$host" "tail -30 /tmp/lab-bench-server.log; $stop_server" >"$out/server.log" 2>&1
	echo "BENCH $label $model commit=$commit SERVER NOT READY (see $out/server.log)"
	exit 1
fi
# The server answering must be this run's (a lab-serve Job on the host network could have taken
# the port meanwhile): its pid owns the listening socket.
owner=$(ssh -o BatchMode=yes "$host" "ss -ltnpH 'sport = :18000' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2")
mine=$(ssh -o BatchMode=yes "$host" "cat $pidf 2>/dev/null")
if [[ -z "$owner" || "$owner" != "$mine" ]]; then
	ssh -o BatchMode=yes "$host" "$stop_server" >/dev/null 2>&1
	echo "BENCH $label $model commit=$commit PORT TAKEN (port 18000 served by pid ${owner:-unknown}, not this run's ${mine:-none})"
	exit 1
fi

# 3. measure under the exclusive lock
# The golden check runs on the build host with the freshly built Linux binary (current rules).
# The throughput client runs on the workstation when it has a release turbine-bench (rows stay
# comparable with earlier ones); otherwise — the rule is to build only on novanas — it runs on
# the build host against the loopback URL, and the BENCH line says client=novanas.
if [[ -x "$root/target/release/turbine-bench" ]]; then
	client=local
	bench_cmd="'$root/target/release/turbine-bench' --url $url"
else
	client=novanas
	bench_cmd="ssh -o BatchMode=yes $host taskset -c $bench_cpus '$remote/target/release/turbine-bench' --url http://127.0.0.1:18000"
fi
golden="ssh -o BatchMode=yes $host cd '$remote/src' \\&\\& '$remote/target/release/turbine-golden'"
ref="tests/golden/$golden_slug/reference.jsonl"
golden16_cmd=""
[[ $run_golden16 -eq 1 ]] && golden16_cmd="$golden compare --url http://127.0.0.1:18000 --reference $ref --concurrency 16 > '$out/golden16.txt' 2>&1"
# --c1: the single-request decode latency (the Phase 1 baseline workload), for targets stated as
# c1 ITL (Phase 6a: FP8 <= 0.75x, INT4 / MXFP4 <= 0.6x the BF16 run).
c1_cmd=""
rm -f "$out/bench-c1.json"
[[ $run_c1 -eq 1 ]] && c1_cmd="$bench_cmd --concurrency 1 --requests 10 --max-tokens 128 --ignore-eos --output json > '$out/bench-c1.json' 2> '$out/bench-c1.err'"
sh -c "
  $golden compare --url http://127.0.0.1:18000 --reference $ref --concurrency 1 > '$out/golden1.txt' 2>&1
  $golden16_cmd
  curl -s '$url/metrics' > '$out/metrics-before.txt'
  $bench_cmd --concurrency 16 --requests $requests --prompt-words 512 --max-tokens 256 \
    --ignore-eos --output json > '$out/bench.json' 2> '$out/bench.err'
  $c1_cmd
" 2>/dev/null
curl -s "$url/metrics" >"$out/metrics.txt"
curl -s "$url/turbine/v1/status" >"$out/status.json"
rm -f "$out/quality.json"
if [[ $run_quality -eq 1 ]]; then
	ssh -o BatchMode=yes "$host" "cd '$remote/src' && taskset -c $bench_cpus \
    '$remote/target/release/turbine-golden' eval --url http://127.0.0.1:18000 \
      --tasks tests/eval/gsm8k-200.jsonl --output json" >"$out/quality.json" 2>"$out/quality.err" ||
		echo "lab-bench: quality run failed (see $out/quality.err)" >&2
fi
ssh -o BatchMode=yes "$host" "cp /tmp/lab-bench-server.log /tmp/lab-bench-server.last.log; $stop_server"

# 4. summary
python3 - "$out" "$label" "$model" "$commit" "$gpu" "$tests" "$run_golden16" "$quick" "$client" <<'EOF'
import json, re, sys
out, label, model, commit, gpu, tests, run_golden16, quick, client = sys.argv[1:]
g1 = open(f"{out}/golden1.txt").read().strip().splitlines()[-1][:5]
if run_golden16 == "1":
    g16 = open(f"{out}/golden16.txt").read().strip().splitlines()[-1][:5]
else:
    g16 = "SKIP"
quick_field = (" quick=1" if quick == "1" else "") + f" client={client}"
try:
    d = json.load(open(f"{out}/bench.json"))
except Exception as e:
    print(f"BENCH {label} {model} commit={commit} gpu={gpu} tests={tests} golden1={g1}{quick_field} BENCH FAILED ({e})")
    sys.exit(1)
# The resolved support row names the served weight format and L0 KV format (Phase 6a).
try:
    support = json.load(open(f"{out}/status.json")).get("support") or {}
except Exception:
    support = {}
formats = f" weight_format={support.get('weight_format', '?')} kv={support.get('kv_format', '?')}"
# Forward-time averages over the throughput run only: subtract the scrape taken after the golden
# runs, whose batch-1 steps would otherwise pull the decode average down.
def read(path):
    try:
        return open(path).read()
    except OSError:
        return ""
after, before = read(f"{out}/metrics.txt"), read(f"{out}/metrics-before.txt")
def val(m, kind, phase):
    r = re.search(rf'turbine_forward_seconds_{kind}{{phase="{phase}"}} ([0-9.e+-]+)', m)
    return float(r.group(1)) if r else 0.0
def avg(phase):
    s = val(after, "sum", phase) - val(before, "sum", phase)
    c = val(after, "count", phase) - val(before, "count", phase)
    return 1000 * s / c if c > 0 else float("nan")
values = {"tok_s": round(d["output_token_throughput"], 1), "itl_p50_ms": round(d["itl_ms"]["p50"], 1),
          "ttft_p50_ms": round(d["ttft_ms"]["p50"]), "ttft_p99_ms": round(d["ttft_ms"]["p99"]),
          "itl_p99_ms": round(d["itl_ms"]["p99"], 1), "decode_fwd_ms": round(avg("decode"), 1),
          "requests_ok": d["requests_ok"], "requests_failed": d["requests_failed"],
          "golden_c1": g1.startswith("PASS")}
if g16 != "SKIP":
    values["golden_c16"] = g16.startswith("PASS")
c1_field = ""
try:
    c1 = json.load(open(f"{out}/bench-c1.json"))
    values["itl_c1_p50_ms"] = round(c1["itl_ms"]["p50"], 2)
    values["tok_s_c1"] = round(c1["output_token_throughput"], 1)
    c1_field = f" itl_c1_p50={c1['itl_ms']['p50']:.2f} tok/s_c1={c1['output_token_throughput']:.1f}"
except OSError:
    pass
except Exception as e:
    c1_field = f" c1=FAILED({e})"
try:
    q = json.load(open(f"{out}/quality.json"))
    values["gsm8k_accuracy"] = round(q["accuracy"], 4)
    c1_field += f" gsm8k={q['accuracy']:.3f}"
except OSError:
    pass
except Exception as e:
    c1_field += f" gsm8k=FAILED({e})"
json.dump({k: v for k, v in values.items() if v == v}, open(f"{out}/labbook-values.json", "w"))
print(f"BENCH {label} {model} commit={commit} gpu={gpu} tests={tests} golden1={g1} golden16={g16}{quick_field}{formats} "
      f"ok={d['requests_ok']} failed={d['requests_failed']} tok/s={d['output_token_throughput']:.1f} "
      f"itl_p50={d['itl_ms']['p50']:.1f} ttft_p50={d['ttft_ms']['p50']:.0f} decode_fwd_ms={avg('decode'):.1f}{c1_field}")
bad = (tests != "skipped" and not tests.endswith("/0")) or g1 != "PASS:" or d["requests_failed"] != 0
if model == "olmoe":
    bad = (tests != "skipped" and not tests.endswith("/0")) or d["requests_failed"] != 0
sys.exit(1 if bad else 0)
EOF
rc=$?

# 5. record the run in labbook (https://labbook.kw.watteel.lab) when the uploader and its token are
# installed; LABBOOK_UPLOAD=0 skips it, LABBOOK_SET=<slug> adds the run to a set. An upload failure
# only warns: the local files under $out stay the record of truth.
uploader="$HOME/.local/bin/labbook-submit.mjs"
if [[ $rc -eq 0 && "${LABBOOK_UPLOAD:-1}" != 0 && -f "$uploader" && -f "$HOME/.config/labbook/token" && -f "$out/labbook-values.json" ]]; then
	set_args=()
	[[ -n "${LABBOOK_SET:-}" ]] && set_args=(--set "$LABBOOK_SET")
	attach=()
	for f in bench.json bench-c1.json quality.json golden1.txt golden16.txt metrics.txt status.json; do
		[[ -s "$out/$f" ]] && attach+=(--attach "$out/$f")
	done
	branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
	config_param="$(printf '%s ' "$(basename "$cfg" .yaml)" "$@" | sed 's/--set //g; s/ *$//')"
	if LABBOOK_URL="${LABBOOK_URL:-https://labbook.kw.watteel.lab}" NODE_EXTRA_CA_CERTS="$HOME/.labbook/cluster-ca.crt" \
		node "$uploader" run --type turbine-lab-bench --external-id "lab-bench:$label-$model:$commit" \
		--param model="$slug" --param gpu="R9700 GPU$gpu" --param config="$config_param" \
		--param engine=turbine --param commit="$commit" --param label="$label" \
		--values-json "$out/labbook-values.json" --commit "$commit" --branch "$branch" \
		${set_args[@]+"${set_args[@]}"} ${attach[@]+"${attach[@]}"} >"$out/labbook.log" 2>&1; then
		echo "lab-bench: recorded in labbook ($(tail -1 "$out/labbook.log"))"
	else
		echo "lab-bench: labbook upload failed (see $out/labbook.log); results kept in $out" >&2
	fi
fi
exit $rc
