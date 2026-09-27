#!/usr/bin/env bash
# lab-bench.sh — fast land-and-measure step on novanas without a k8s Job.
#
#   scripts/lab-bench.sh [--gpu 0] [--model llama|olmoe] [--label L] [--with-tests] [--skip-tests]
#                         [--golden16] [--quick] [-- <--set k=v>...]
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
#    200 requests, or 64 with --quick for a faster read during iteration).
# 4. Stops the server and prints one "BENCH ..." summary line (quick=1 added when --quick was
#    used; golden16=SKIP when --golden16 was not); exits 1 if tests, golden c1 or the bench fail.
#    Results land in target/lab-bench/<label>/ on the workstation.
set -uo pipefail

host="${TURBINE_REMOTE_HOST:-piwi@192.168.10.203}"
gpu=0
model=llama
label="$(git rev-parse --short HEAD)"
run_tests=0
run_golden16=0
quick=0
usage() {
	echo "usage: scripts/lab-bench.sh [--gpu 0] [--model llama|olmoe] [--label L] [--with-tests] [--skip-tests] [--golden16] [--quick] [-- --set k=v ...]" >&2
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
	--quick)
		quick=1
		shift
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
case "$model" in
llama)
	slug=llama-3.2-3b-instruct
	cfg=scripts/lab/phase2c-novanas-llama.yaml
	;;
olmoe)
	slug=olmoe-1b-7b-0125-instruct
	cfg=scripts/lab/phase2c-novanas-olmoe.yaml
	;;
*)
	echo "lab-bench: unknown model $model" >&2
	exit 2
	;;
esac
requests=200
[[ $quick -eq 1 ]] && requests=64

root="$(git rev-parse --show-toplevel)"
cd "$root" || exit 1
name="$(basename "$root")"
remote="/home/piwi/turbine-ci/remote/$name"
out="$root/target/lab-bench/$label-$model"
mkdir -p "$out"
commit="$(git rev-parse --short HEAD)"
url=http://192.168.10.203:18000

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

# 2. serve natively, pinned to one card
ssh -o BatchMode=yes "$host" "pkill -u piwi -x turbine-server; sleep 1; \
  setsid bash -c \"cd '$remote/src' && \
    LD_LIBRARY_PATH=/opt/rocm/rocm/lib:/opt/rocm/rocm/lib/rocm_sysdeps/lib \
    TURBINE_AMD_SMI_LIBRARY=/opt/rocm/rocm/lib/libamd_smi.so.26.5.0 ROCR_VISIBLE_DEVICES=$gpu \
    exec '$remote/target/release/turbine-server' --config $cfg \
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
	ssh -o BatchMode=yes "$host" "tail -30 /tmp/lab-bench-server.log; pkill -u piwi -x turbine-server" >"$out/server.log" 2>&1
	echo "BENCH $label $model commit=$commit SERVER NOT READY (see $out/server.log)"
	exit 1
fi

# 3. measure under the exclusive lock
bench_bin="$root/target/release/turbine-bench"
# The golden check runs on the build host with the freshly built Linux binary (current rules);
# the throughput client stays on the workstation so rows stay comparable with earlier ones.
golden="ssh -o BatchMode=yes $host cd '$remote/src' \\&\\& '$remote/target/release/turbine-golden'"
ref="tests/golden/$slug/reference.jsonl"
golden16_cmd=""
[[ $run_golden16 -eq 1 ]] && golden16_cmd="$golden compare --url http://127.0.0.1:18000 --reference $ref --concurrency 16 > '$out/golden16.txt' 2>&1"
scripts/bench-lock.sh sh -c "
  $golden compare --url http://127.0.0.1:18000 --reference $ref --concurrency 1 > '$out/golden1.txt' 2>&1
  $golden16_cmd
  '$bench_bin' --url $url --concurrency 16 --requests $requests --prompt-words 512 --max-tokens 256 \
    --ignore-eos --output json > '$out/bench.json' 2> '$out/bench.err'
" 2>/dev/null
curl -s "$url/metrics" >"$out/metrics.txt"
curl -s "$url/turbine/v1/status" >"$out/status.json"
ssh -o BatchMode=yes "$host" "cp /tmp/lab-bench-server.log /tmp/lab-bench-server.last.log; pkill -u piwi -x turbine-server"

# 4. summary
python3 - "$out" "$label" "$model" "$commit" "$gpu" "$tests" "$run_golden16" "$quick" <<'EOF'
import json, re, sys
out, label, model, commit, gpu, tests, run_golden16, quick = sys.argv[1:]
g1 = open(f"{out}/golden1.txt").read().strip().splitlines()[-1][:5]
if run_golden16 == "1":
    g16 = open(f"{out}/golden16.txt").read().strip().splitlines()[-1][:5]
else:
    g16 = "SKIP"
quick_field = " quick=1" if quick == "1" else ""
try:
    d = json.load(open(f"{out}/bench.json"))
except Exception as e:
    print(f"BENCH {label} {model} commit={commit} gpu={gpu} tests={tests} golden1={g1}{quick_field} BENCH FAILED ({e})")
    sys.exit(1)
m = open(f"{out}/metrics.txt").read()
def avg(phase):
    s = re.search(rf'turbine_forward_seconds_sum{{phase="{phase}"}} ([0-9.e+-]+)', m)
    c = re.search(rf'turbine_forward_seconds_count{{phase="{phase}"}} ([0-9.e+-]+)', m)
    return 1000 * float(s.group(1)) / float(c.group(1)) if s and c and float(c.group(1)) else float("nan")
print(f"BENCH {label} {model} commit={commit} gpu={gpu} tests={tests} golden1={g1} golden16={g16}{quick_field} "
      f"ok={d['requests_ok']} failed={d['requests_failed']} tok/s={d['output_token_throughput']:.1f} "
      f"itl_p50={d['itl_ms']['p50']:.1f} ttft_p50={d['ttft_ms']['p50']:.0f} decode_fwd_ms={avg('decode'):.1f}")
bad = (tests != "skipped" and not tests.endswith("/0")) or g1 != "PASS:" or d["requests_failed"] != 0
if model == "olmoe":
    bad = (tests != "skipped" and not tests.endswith("/0")) or d["requests_failed"] != 0
sys.exit(1 if bad else 0)
EOF
