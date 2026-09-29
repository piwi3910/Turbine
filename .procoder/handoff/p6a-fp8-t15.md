# Handoff: p6a-fp8-t15 (plan Task 15, `fp8_block` kernel + FP8 layout in the loader)

Written 2026-09-29 by the Task 15 builder. Branch `p6a-fp8-t15` (from `p6a-fp8` 7174d4c), worktree
`.claude/worktrees/agent-a4baec4689b995379`. Not pushed. The proof (fixtures, bench, eval, soak) is the
next agent's.

## Done (committed)

- ca0bad2 `feat(rocm)`: own kernel `turbine_hip_fp8_block` (W8A16; fused WMMA for decode m <= 64,
  dequantize-to-BF16 + tuned hipBLASLt GEMM above / in prefill), plus the `qgemm_eval --block 1` timing.
- Handoffs, one per shared file: db9aea0 impl_table.cpp, 9cf6b63 turbine_hip.hpp, f8f3bbd CMakeLists.txt,
  82a6cd6 cards/gfx1201.rs, c3493ad hip_qgemm.rs, 6f99811 weights/mod.rs, 983c670 weights/fp8.rs
  (+ ct_fp8/hf_fp8 constructors, docs/extending/weight-format.md, the mod.rs test expectations),
  43f4c8d turbine-server model.rs, 7b6c977 tiny_model.rs.
- Test judgement: the old builder's widened bound (2e-2 + |c|/128) is gone. The dequantize path is
  judged against the CPU BF16 gemm of `bf16(e4m3(q)·s)` (what it multiplies), the fused path against
  the CPU qgemm (exact q·s), both with the Phase 1 BF16 tolerance. All 80 cases pass
  (max |Δ| 1.56e-2 = one BF16 ulp at |c| ≥ 2).
- Loader: `fp8_block` keeps e4m3 + F32 block scales; `WeightFormat::for_kernels` /
  `weights::resolve_for_providers` (called in `prepare_with` before the requirements) decodes to BF16
  only the stacks no selected provider runs (`kernel_unsupported`) or whose TP shard cuts a block
  (`shard_misaligned`), logged `event="fp8_block_decoded"`. Activations stay BF16 for block
  (W8A16, the user's option B); the checkpoint's group-128 activation scheme is not applied, so its
  golden reference must be made with `--act-quant none` (the dequant path multiplies
  bf16(q·s) exactly as `dequantize_checkpoint.py` would; the fused decode path uses exact q·s).

## Results

- `scripts/gate.sh`: ok, 774 passed (crates all).
- `scripts/lab-test.sh novanas --tier quick`: PASS (job turbine-lab-test-0929043937-1be4f7b8),
  including `qgemm_fp8_block_matches_cpu`, `qgemm_fp8_block_rows_do_not_depend_on_m`,
  `fp8_block_decode_fallback_per_stack`, `tp2_quantized_matches_tp1_on_host`.
- Weight bytes, computed from the checkpoint headers (the FP8 layout the loader uploads: e4m3
  linears, F32 block scales, BF16 embedding/norms, tied head skipped): **3,607,615,488 B** for
  `llama-3.2-3b-instruct-fp8-block` vs **6,425,499,648 B** BF16 = 0.56× (the 788 MB BF16 embedding
  keeps it above 0.5; the linear layers are exactly half).

## Pending: the served number (detached, queued on port18000 + bench.lock)

- Launched `scripts/bench-lock.sh --name port18000 scripts/bench-lock.sh scratchpad/t15_serve.sh`
  (nohup). It serves `scratchpad/t15-fp8-block.yaml` (copy of the old worktree's
  `phase6-novanas-llama-fp8-block.yaml`) with `lab-serve.sh`, greps the log, and always stops its own
  serve Job.
- Watch: `/private/tmp/claude-501/-Users-pascal-Development-Turbine/7482a1b1-2407-47bb-91f2-d826e25c21af/scratchpad/t15_weight_bytes.txt`
  (last line `t15-serve: done rc=<rc> run=<id>`); full log `…/scratchpad/t15_serve.txt`.
- Judge: `rc=0`; the `weight_format` event's `weight_bytes` should be 3,607,615,488 (± nothing — it is
  exact), `layers=bf16:1,fp8_block_128x128:196` (or similar: every decoder linear FP8, lm_head BF16),
  `activation=none`, and **no** `fp8_block_decoded` line. A `fp8_block_decoded` line or ~6.4 GB means the
  provider refused a 3B shape — a bug to fix before the proof.
- If rc≠0, read `t15_serve.txt`; make sure `scripts/lab-serve.sh novanas --stop <run>` ran.

## Left for the proof agent

Fixture (`--act-quant none`), golden c1/c16, `lab-bench --model llama-fp8-block`, eval, soak, row flip,
decisions/perf-log entries — plan Task 15 steps 4–6. The untracked golden/config files are in the old
worktree `agent-adb021480bf19bfbe`.

## Proof agent (2026-09-29, lead rotation 5)

Merged `phase-6a-quantization` (a0ba309). Added `scripts/lab/phase6-novanas-llama-fp8-block.yaml` (the
lab-bench config for `--model llama-fp8-block`; copy of the phase2c Llama config). The Mac-side
`t15_serve.sh` (killed in the ssh outage) is replaced by one novanas-side detached driver that does
the served-bytes check, the GSM8K-200 eval and the bench in one serve, then the vLLM baseline:

- Script: `novanas:/home/piwi/turbine-ci/remote/agent-a4baec4689b995379/t15p/t15_proof_run.sh`
  (Mac copy: session scratchpad `scratchpad/t15p/t15_proof_run.sh`), started 2026-09-29 09:24Z, pid 112710.
  Builds `kbuild/libturbine_hip.so` (nice 19, cores 12-15), then pass A (Turbine, native, GPU 0) and
  pass B (vLLM k3s Job, port 18100), each under port18000.lock → bench.gate → bench.lock, fixtures
  paused. Queued behind the full-GSM8K FP8 KV requeue.
- Log: `…/t15p/t15_proof_run.log`, last line `t15-proof: done rc=<rc> turbine=<rc> vllm=<rc>`, with
  two `SUMMARY` lines (tok/s, TTFT/ITL p50, c1 ITL p50, GSM8K). Outputs in `…/t15p/`:
  `load_events.txt`, `server.log`, `status.json`, `turbine-quality.json`, `eval-compare.txt`,
  `turbine-bench{,-c1}.json`, `vllm-bench{,-c1}.json`, `vllm-quality.json`, `vllm-pod.log`.

How to judge:

1. `load_events.txt`: the `weight_format` event's `weight_bytes` = **3,607,615,488** exactly, every
   decoder linear FP8 (`fp8_block_128x128`), lm_head/embedding BF16, `activation=none`, and **no**
   `fp8_block_decoded` line. ~6.4 GB or a decoded line = the kernel refused a 3B shape: a bug in the
   lead-owned loader/registry → `handoff(<file>)` commit before the proof counts.
2. `eval-compare.txt`: exit 0 against `tests/eval/llama-3.2-3b-instruct/turbine-bf16.json` (0.805)
   with max drop 0.02 (Q9). Copy `turbine-quality.json` to
   `tests/eval/llama-3.2-3b-instruct-fp8-block/turbine.json` and `vllm-quality.json` to `vllm.json`.
3. Bench: c16 tok/s ≥ 1.0 × BF16 (854.7, perf-log phase-start baseline); Turbine ≥ 0.9 × vLLM where
   vLLM serves it. vLLM ran as a k3s Job (the GPU is k3s's pick, maybe GPU 1: its numbers are
   indicative only if the pod landed on GPU 1). No BF16 c1 in this run.
4. Still open after it: the golden fixture (`--act-quant none`, rank 4 in
   `/home/piwi/turbine-ci/fixture.queue`), then `lab-bench --model llama-fp8-block --golden16 --c1`
   (Mac-driven, background), tolerance from the self-spread, labbook, soak (ask the lead), row flip
   as a `handoff(support.rs)` commit, perf-log row.

## Verdict (2026-09-29, proof collector, lead rotation 7)

`t15_proof_run.log`: `t15-proof: done rc=0 turbine=0 vllm=0`. Collected the pass directory by scp
from `novanas:/home/piwi/turbine-ci/remote/agent-a4baec4689b995379/t15p/` into this session's
scratchpad (`t15p_collected/t15p/`). All four pass criteria hold:

| Criterion | Required | Observed | Verdict |
|---|---|---|---|
| `weight_bytes` | 3,607,615,488 | 3,607,615,488 (`load_events.txt` `weight_format` event) | PASS (exact) |
| `fp8_block_decoded` | absent | absent (grepped `load_events.txt`) | PASS |
| c16 tok/s | ≥ 854.7 | 1061.44 | PASS (1.24× BF16 floor, 1.57× this run's own vLLM pass) |
| GSM8K drop vs pair | ≤ 0.02 | baseline (BF16 turbine) 0.805 → candidate (fp8-block turbine) 0.800, drop 0.005 (`eval-compare.txt`: PASS) | PASS |

Other load-event facts: `layers=fp8_block:196`, `activation=none`, `packaging=ct_fp8`, KV `bf16`.
Support row logged at startup: `amd/gfx1201/LlamaForCausalLM/fp8_block/bf16/none` = `experimental`
(WARN, as expected pre-flip).

**BENCH-equivalent line** (not a `lab-bench.sh` run — the driver is `t15_proof_run.sh` — but the
same fields):

```
BENCH t15-proof llama-fp8-block commit=d76d123 gpu=0 tests=skip golden1=SKIP golden16=SKIP \
  tok_s=1061.44 tok_s_c1=114.68 itl_p50_ms=11.46 itl_c1_p50_ms=8.42 ttft_p50_ms=227.24 \
  gsm8k=0.800 (bf16 ref 0.805, drop 0.005) \
  vllm_tok_s=677.79 vllm_tok_s_c1=75.94 vllm_gsm8k=0.805 (vLLM GPU picked by k3s, may not be GPU 0) \
  verdict=PASS
```

**Labbook**: test type `turbine-lab-bench`, set `phase-6a-quantization`.
- Turbine run `2e31910f-20a4-4706-8ae7-e89b5f582a61` (`t15-proof:llama-fp8-block:turbine:d76d123`),
  model `llama-3.2-3b-instruct-fp8-block`, status `pass`.
- vLLM-ROCm run `a86153ab-ff32-49a9-82c3-64fe64434f8c`
  (`t15-proof:llama-fp8-block:vllm-rocm:d76d123`), same model key, status `pass`; added as the
  `vllm-rocm-0-23-gpu0` baseline's member for `model=llama-3.2-3b-instruct-fp8-block` (matching the
  AWQ precedent). Turbine vs this baseline: tok_s ratio 1.566, itl_p50 ratio 0.624, ttft_p50 ratio
  0.794 — target `≥ 75% of vLLM-ROCm 0.23.0 (GPU 0)` reads `on`. `golden_c1`/`golden_c16` left
  unset on both runs (fixture not run yet); `golden_summary` on the Turbine row notes why.

**Open questions / next steps** (unchanged from the previous rotation's list, still open):
1. Golden fixture with `--act-quant none` — rank 4 in `/home/piwi/turbine-ci/fixture.queue`, not
   yet run. Until it lands, this proof's PASS does not cover golden-level numerics, only the eval
   accuracy gate.
2. `lab-bench --model llama-fp8-block --golden16 --c1` once the fixture reference exists.
3. Tolerance bounds from the self-spread method (as OLMoE's was calibrated) — not yet computed for
   this checkpoint; the eval-compare 0.02 drop bound was used instead (Phase 6a per-format default
   for FP8).
4. Soak (`scripts/overload-soak.sh novanas --duration 10m` at minimum) — needs the lead's go-ahead
   per the lab rules.
5. perf-log row for this proof — not yet added; the labbook runs above are the durable record in
   the meantime.

**Row-flip commit**: prepared separately (see below), not merged into this branch's history — the
lead reviews and applies it.
