#!/usr/bin/env bash
# lab-bench.sh — fast land-and-measure step on novanas without a k8s Job.
#
#   scripts/lab-bench.sh [--gpu N] [--model llama|olmoe] [--label L] [--skip-tests] [-- <--set k=v>...]
#
# 1. In parallel on novanas: the release turbine-server build (scripts/remote-cargo.sh, cached)
#    plus the HIP kernel library (CMake, cached), and the workspace tests.
# 2. Starts the server directly on the host, pinned to one R9700 with ROCR_VISIBLE_DEVICES
#    (default GPU 0), so every step runs on the same card.
# 3. Under the exclusive benchmark lock: golden at concurrency 1 (the gate), golden at 16
#    (reported), and the fixed throughput bench (16 concurrent, 200 requests, 512-word prompts,
#    256 tokens).
# 4. Stops the server and prints one "BENCH ..." summary line; exits 1 if tests, golden c1 or the
#    bench fail. Results land in target/lab-bench/<label>/ on the workstation.
set -uo pipefail

host="${TURBINE_REMOTE_HOST:-piwi@192.168.10.203}"
gpu=0
model=llama
label="$(git rev-parse --short HEAD)"
run_tests=1
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
	--skip-tests)
		run_tests=0
		shift
		;;
	--)
		shift
		break
		;;
	*)
		echo "usage: scripts/lab-bench.sh [--gpu N] [--model llama|olmoe] [--label L] [--skip-tests] [-- --set k=v ...]" >&2
		exit 2
		;;
	esac
done
case "$model" in
llama)
	slug=llama-3.2-3b-instruct
	cfg=scripts/lab/phase2-novanas-llama.yaml
	;;
olmoe)
	slug=olmoe-1b-7b-0125-instruct
	cfg=scripts/lab/phase2-novanas-olmoe.yaml
	;;
*)
	echo "lab-bench: unknown model $model" >&2
	exit 2
	;;
esac

root="$(git rev-parse --show-toplevel)"
cd "$root" || exit 1
name="$(basename "$root")"
remote="/home/piwi/turbine-ci/remote/$name"
out="$root/target/lab-bench/$label-$model"
mkdir -p "$out"
commit="$(git rev-parse --short HEAD)"
url=http://192.168.10.203:18000

# 1. build (server + kernels) and tests in parallel on novanas
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
scripts/bench-lock.sh sh -c "
  $golden compare --url http://127.0.0.1:18000 --reference $ref --concurrency 1 > '$out/golden1.txt' 2>&1
  $golden compare --url http://127.0.0.1:18000 --reference $ref --concurrency 16 > '$out/golden16.txt' 2>&1
  '$bench_bin' --url $url --concurrency 16 --requests 200 --prompt-words 512 --max-tokens 256 \
    --ignore-eos --output json > '$out/bench.json' 2> '$out/bench.err'
" 2>/dev/null
curl -s "$url/metrics" >"$out/metrics.txt"
ssh -o BatchMode=yes "$host" "cp /tmp/lab-bench-server.log /tmp/lab-bench-server.last.log; pkill -u piwi -x turbine-server"

# 4. summary
python3 - "$out" "$label" "$model" "$commit" "$gpu" "$tests" <<'EOF'
import json, re, sys
out, label, model, commit, gpu, tests = sys.argv[1:]
g1 = open(f"{out}/golden1.txt").read().strip().splitlines()[-1][:5]
g16 = open(f"{out}/golden16.txt").read().strip().splitlines()[-1][:5]
try:
    d = json.load(open(f"{out}/bench.json"))
except Exception as e:
    print(f"BENCH {label} {model} commit={commit} gpu={gpu} tests={tests} golden1={g1} BENCH FAILED ({e})")
    sys.exit(1)
m = open(f"{out}/metrics.txt").read()
def avg(phase):
    s = re.search(rf'turbine_forward_seconds_sum{{phase="{phase}"}} ([0-9.e+-]+)', m)
    c = re.search(rf'turbine_forward_seconds_count{{phase="{phase}"}} ([0-9.e+-]+)', m)
    return 1000 * float(s.group(1)) / float(c.group(1)) if s and c and float(c.group(1)) else float("nan")
print(f"BENCH {label} {model} commit={commit} gpu={gpu} tests={tests} golden1={g1} golden16={g16} "
      f"ok={d['requests_ok']} failed={d['requests_failed']} tok/s={d['output_token_throughput']:.1f} "
      f"itl_p50={d['itl_ms']['p50']:.1f} ttft_p50={d['ttft_ms']['p50']:.0f} decode_fwd_ms={avg('decode'):.1f}")
bad = (tests != "skipped" and not tests.endswith("/0")) or g1 != "PASS:" or d["requests_failed"] != 0
if model == "olmoe":
    bad = (tests != "skipped" and not tests.endswith("/0")) or d["requests_failed"] != 0
sys.exit(1 if bad else 0)
EOF
