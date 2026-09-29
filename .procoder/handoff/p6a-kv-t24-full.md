# Handoff: p6a-kv-t24-full (plan Task 24, full-GSM8K FP8 KV gate, Llama + OLMoE)

Written 2026-09-29 ~09:10 +04 by builder a67eec8abc8f117eb, stopped on the coordinator's order while
its run waits in the novanas GPU queue. Branch `p6a-kv-t24-full` = `p6a-kv-t24` + `phase-6a-quantization`
(d466c0b, which carries `tests/eval/gsm8k-full.jsonl`, 1,319 items).

## The detached run on novanas

- Script: `/home/piwi/turbine-ci/remote/agent-a67eec8abc8f117eb/gsm8k_full_run.sh` (bash pid 260700,
  started 08:40 +04 with `setsid nohup`, session leader, survives any ssh disconnect).
- Log: `/home/piwi/turbine-ci/remote/agent-a67eec8abc8f117eb/gsm8k_full_run.log`
  (server log of the current pass: `/tmp/gsm8kfull-server.log`, pid file `/tmp/gsm8kfull-server.pid`).
- Binaries: release `turbine-server` / `turbine-golden` built from this branch in
  `/home/piwi/turbine-ci/remote/agent-a67eec8abc8f117eb/target/release/`, kernel library
  `/home/piwi/turbine-ci/remote/agent-a67eec8abc8f117eb/kbuild/libturbine_hip.so`.
- Passes, strictly one after another: Llama BF16 KV (`scripts/lab/phase2c-novanas-llama.yaml`),
  Llama FP8 KV (`scripts/lab/phase6-novanas-llama-fp8kv.yaml`), OLMoE BF16 KV
  (`scripts/lab/phase2c-novanas-olmoe.yaml`), OLMoE FP8 KV (`scripts/lab/phase6-novanas-olmoe-fp8kv.yaml`).
  Each pass takes `port18000.lock`, then `bench.gate` + `bench.lock` exclusively (fd-based `flock`,
  released when the pass returns), pauses the `scripts/golden/` fixture jobs (SIGSTOP, SIGCONT after),
  serves natively on GPU 0 (cores 0-11), runs
  `turbine-golden eval --url http://127.0.0.1:18000 --tasks tests/eval/gsm8k-full.jsonl --output json`
  under `timeout 21600`, then stops only its own server.
- Outputs (in the remote source tree `/home/piwi/turbine-ci/remote/agent-a67eec8abc8f117eb/src/`):
  - `tests/eval/llama-3.2-3b-instruct/turbine-bf16-full.json`
  - `tests/eval/llama-3.2-3b-instruct/turbine-fp8_e4m3-full.json`
  - `tests/eval/olmoe-1b-7b-0125-instruct/turbine-bf16-full.json`
  - `tests/eval/olmoe-1b-7b-0125-instruct/turbine-fp8_e4m3-full.json`
  - each with a `.err` beside it (the eval's stderr).
- State at handoff: queued (`pass llama-bf16: waiting for bench.gate`) behind another agent's exclusive
  `bench.lock` session (pid 251034, since 08:39, a lab-test Job on GPU 1) and two earlier exclusive
  waiters (pids 194608, 243975). Nothing of ours is on the GPU yet.

**Do not** run `scripts/remote-cargo.sh`, `lab-bench.sh` or `gate.sh` from worktree
`agent-a67eec8abc8f117eb` until the run is done: its `rsync --delete` would wipe the four output files
(not in the local tree) and rebuild the binaries mid-run. Do not remove that worktree either:
`lab-prune.sh` may then delete its remote `target/` (the binaries later passes exec).

## How to tell it is done

`tail -3 /home/piwi/turbine-ci/remote/agent-a67eec8abc8f117eb/gsm8k_full_run.log` ends with
`ALLDONE rc: llama-bf16=<rc> llama-fp8kv=<rc> olmoe-bf16=<rc> olmoe-fp8kv=<rc>` (and
`ps -p 260700` finds nothing). rc 0 = report written; 2 = eval failed (see the `.err`); 124 = the 6 h
timeout hit (report missing or partial); 1 = `SERVER NOT READY` (the log holds the server tail). Each
pass logs `pass <name>: locks held, starting`, `server ready`, and `eval rc=` with UTC times.

## Collect and judge

```sh
d=/home/piwi/turbine-ci/remote/agent-a67eec8abc8f117eb/src/tests/eval
for s in llama-3.2-3b-instruct olmoe-1b-7b-0125-instruct; do
  scp "piwi@192.168.10.203:$d/$s/turbine-{bf16,fp8_e4m3}-full.json" tests/eval/$s/
done
# eval-compare runs on novanas (nothing is built on the Mac):
ssh piwi@192.168.10.203 'cd /home/piwi/turbine-ci/remote/agent-a67eec8abc8f117eb/src && \
  for s in llama-3.2-3b-instruct olmoe-1b-7b-0125-instruct; do \
    ../target/release/turbine-golden eval-compare --baseline tests/eval/$s/turbine-bf16-full.json \
      --candidate tests/eval/$s/turbine-fp8_e4m3-full.json --max-drop 0.01; echo "$s exit=$?"; done'
```

Exit 0 = PASS, 1 = drop > 0.01, 2 = I/O. Commit the four JSONs as
`test(eval): full GSM8K (1,319 items) for the FP8 KV gate — Llama and OLMoE`. Reference points
(GSM8K-200): Llama 0.805 → 0.790, OLMoE 0.655 → 0.615. The first 200 lines of `gsm8k-full.jsonl`
are the GSM8K-200 set, so those items can be compared per item with the 200-run JSONs as a sanity check.
Do not flip `support.rs` rows or loosen any bound: report the verdicts to the coordinator first.

## If OLMoE misses 0.01 (decision 185ccee): look for a numerics cause first

What this builder already read (no run yet):

- `crates/turbine-model/src/kv_scales.rs`: `KvCache::fp8_from_checkpoint` reads
  `model.layers.{l}.self_attn.{k,v}_scale` (or `{k,v}_proj.output_scale`), all-or-nothing, else **all
  1.0**. Both lab configs say the checkpoints store none, so Llama and OLMoE both run with every scale
  1.0 — the "wrong layer's scale" hypothesis cannot change any value unless a checkpoint does carry
  scales. Confirm first: list the tensor names of `/home/piwi/turbine-models/olmoe-1b-7b-0125-instruct`
  for `_scale` (and check `/turbine/v1/status` `quantization` / the startup log of an FP8 KV serve).
- `crates/turbine-model/src/executor/decoder/mod.rs` ~1800: `model_layer = pp.layers.start + i`
  selects the scales; `batch::kv_layer(kv, i)` selects the pages. Without pipeline parallelism both are
  `i`, consistent.

Steps, one hypothesis at a time (procoder:debug):

1. Scale values: as above. If scales are all 1.0, the per-layer-scale path is ruled out for this
   checkpoint; say so in the report.
2. Saturation / range: with scale 1.0, e4m3 clips at ±448 and has 3 mantissa bits. Check OLMoE's
   post-RoPE K and V magnitudes (OLMoE has QK-norm, so K should be small; V is not normalised): trace
   one golden prompt with `DecoderExecutor::set_trace` (`golden hip_trace_vs_cpu_3b`-style, release build,
   `TURBINE_GOLDEN_TRACE=<ids>`) and record per-layer max |k_rope| / |v|. Any value > 448, or many
   subnormals (< 2^-6), points at the missing calibration rather than a bug.
3. Turbine vs the emulated-FP8-KV transformers reference: the FP8 KV golden fixture job
   (`/home/piwi/turbine-ci/remote/agent-ad2d9c8c7cc380c42/fp8kv_fixtures_t24.log`, outputs
   `/home/piwi/turbine-ci/golden-work/fp8kv/<slug>/{reference.jsonl,self_spread.json}`, made with
   `scripts/golden/quant_reference.py --kv-quant fp8_e4m3` and `self_spread.py --kv-quant fp8_e4m3`)
   gives the reference. Serve `phase6-novanas-olmoe-fp8kv.yaml` and run
   `turbine-golden compare --reference <that reference.jsonl> --concurrency 1` plus
   `turbine-golden positions --prompt-id <worst id>`: if Turbine tracks the emulated reference within
   its self-spread, Turbine implements FP8 KV correctly and the drop is the format's (no bug); if it
   diverges where BF16 KV does not, bisect by layer with the trace (K write vs FP8 read in the paged
   attention kernel, prefill vs decode path, CK FP8 instance vs Turbine fallback).
4. Report the finding (bug found + fixing commit with a test that catches it, or "format drop, Turbine
   matches the emulated reference") to the coordinator before any `support.rs` or tolerance change;
   `support.rs` changes go as a `handoff(crates/turbine-core/src/support.rs)` commit.
