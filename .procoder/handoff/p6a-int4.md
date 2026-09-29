# Handoff: p6a-int4 (plan Tasks 16–18)

## State 2026-09-29 ~13:40 (T18 builder, rotation 5)

Branch tip after `phase-6a-quantization` (a0ba309) merged in: d14402f + this handoff. Gate ok 777 at d14402f.

Done:
- d14402f `test(golden)`: AWQ and GPTQ tolerances from the self-spreads (OLMoE method), GPTQ fixture dir
  (`tests/golden/llama-3.2-3b-instruct-gptq/`), both READMEs with repo / revision / commands / spread tables,
  `tests/eval/llama-3.2-3b-instruct-gptq/gate.json` (vLLM refusal recorded). `golden.rs` already listed both
  slugs in `QUANT_SLUGS`; `quant_fixtures_valid` passes (in the gate).
  - AWQ spread: every variant 16/16, max likely 0.1169 (bf16 sdpa full, p11), tail 0.3557 (bf16 eager incr,
    p08) → 0.12 / 0.36, inside the BF16 Llama bounds, so tolerance.json keeps 0.15 / 0.55 (batched 0.25 / 0.75).
  - GPTQ spread: every variant 16/16, likely 0.1358 (p10), tail 0.9092 (p04, BF16 eager only; others ≤ 0.33)
    → likely 0.15, tail 0.91, batched likely 0.25, batched tail 0.91.
  - Choice to confirm (lead): bounds = max(spread, BF16 Llama bounds) — a quantized slug never tighter than its
    BF16 model. The pure OLMoE rule would give AWQ 0.12 / 0.36.
  - Spread JSONs (not committed): novanas `/home/piwi/turbine-ci/scratch/p6a-int4/fixtures/*-spread*.json`.

Running detached on novanas (started 09:35Z, pid 133875; script copy `.procoder/handoff/t18_int4_run.sh`,
on the host `/home/piwi/turbine-ci/scratch/p6a-int4/t18_int4_run.sh`): kernel build, then three passes each
under port18000.lock → bench.gate → bench.lock, GPU 0, fixtures paused, binaries of d14402f
(`agent-abee0e2542a317f3c/target/release`): `bf16` (golden c1, bench c16 200 req, c1), `awq` (golden c1 + c16,
bench c16, c1), `gptq` (golden c1 + c16, bench c16, c1, GSM8K-200). Queued behind the full-GSM8K requeue.
- Log: `/home/piwi/turbine-ci/scratch/p6a-int4/t18-run/run.log`, last line `t18_int4_run: done rc=<rc>`.
- Results: `…/t18-run/<bf16|awq|gptq>/{golden1.txt,golden16.txt,bench.json,bench-c1.json,quality.json,server.log,status.json}`.

How to judge (plan Task 18):
- golden: last line of golden1/golden16 PASS for awq and gptq (c16 judged with the batched bounds).
- throughput: awq/gptq `bench.json` `output_token_throughput` ≥ 0.9 × bf16's; c1 ITL (`bench-c1.json` itl p50)
  ≤ 0.6 × bf16's. Earlier quick AWQ run: 1290.6 vs 875.7 tok/s, ITL 5.87 vs 12.28 ms.
- GPTQ accuracy: copy `gptq/quality.json` to `tests/eval/llama-3.2-3b-instruct-gptq/turbine.json`, then
  `turbine-golden eval-compare --baseline tests/eval/llama-3.2-3b-instruct/turbine-bf16.json --candidate
  tests/eval/llama-3.2-3b-instruct-gptq/turbine.json --max-drop 0.04` (BF16 0.805 → pass ≥ 0.765). AWQ accuracy
  already in (0.775 vs vLLM 0.755 / BF16 0.805: pass).
- vLLM-ROCm GPTQ: refused (transposed qzeros in TritonW4A16LinearKernel), in gate.json; not retried.
  vLLM AWQ baseline: 485.8 tok/s, GSM8K 0.755.

Remaining after the run: commit the GPTQ eval result + perf-log rows; soak on AWQ (ask the lead first);
`handoff(crates/turbine-core/src/support.rs)` commit flipping the passing rows (+ `baseline_rows_present`);
labbook upload (`LABBOOK_SET=phase-6a-quantization`, vLLM AWQ as baseline).

## Earlier handoff (08:00)


Written 2026-09-29 ~08:00 by the new 6a lead (the builder was killed without a handoff).
Worktree `.claude/worktrees/agent-abee0e2542a317f3c`, tip 2116221 — all committed INT4 code is on
`phase-6a-quantization`. Read AGENTS.md and plan Task 18 first.

## Done (on phase-6a-quantization)

- Task 16: 3c99e40 (harness + fused WMMA candidate), 13db1cd / 1a16ec6 (decisions entries).
- Task 17: f574496 (INT4 group GEMM, AWQ + GPTQ layouts), 44d0d8a (prefill rows invariant), handoffs b8e07a5,
  7c1508f, b63db86, 946d8b3, 9ca195d, f31cc24 (fused kernel up to 128 rows, dequant path above).
- Task 18 measurements so far (GPU 0, `--quick`, commit 3b54bf6): AWQ 1290.6 tok/s vs BF16 875.7 (1.47×),
  c1 ITL 5.87 ms vs 12.28 (0.48×, target ≤ 0.6×), golden c1 PASS (BF16 bounds), GSM8K-200 (CoT) 0.775 vs BF16 0.805.
  vLLM-ROCm on the AWQ checkpoint: 485.8 tok/s, GSM8K 0.755.

## Uncommitted (results, look complete for AWQ)

- `tests/golden/llama-3.2-3b-instruct-awq/{reference.jsonl,tolerance.json}` — README missing; tolerance is the BF16
  one until the AWQ self-spread finishes (below).
- `tests/eval/llama-3.2-3b-instruct-awq/{turbine.json,vllm.json,gate.json}`, `tests/eval/llama-3.2-3b-instruct/turbine-bf16.json`.
- `scripts/lab/phase6-novanas-llama-{awq,gptq}.yaml`.

## Running on novanas (let finish)

- `scratchpad/int4/fixtures6.sh` (remote, `/home/piwi/turbine-ci/scratch/p6a-int4/fixtures6.log`, fixture.lock):
  AWQ dequantize → AWQ self-spread (remaining variants, `…-awq-spread-b.json`) → GPTQ reference → GPTQ
  self-spread. Outputs in `/home/piwi/turbine-ci/scratch/p6a-int4/fixtures/`.
- `scratchpad/int4/vllm_session.sh llama-3.2-3b-instruct-gptq gptq all` (local pid 40836; log
  `scratchpad/int4/vllm-session-gptq.log`): vLLM GPTQ serve + bench + GSM8K, queued on port18100 then bench.lock.

## Exact next steps (one fresh builder, sonnet is not enough for tolerance work — default model)

1. When fixtures6 finishes: tolerance.json for AWQ and GPTQ from the spreads (OLMoE method,
   `tests/golden/olmoe-1b-7b-0125-instruct/README.md`); READMEs with repo, revision, commands; GPTQ fixture dir.
2. `golden.rs` slug list + `quant_fixtures_valid` (remote-cargo).
3. Lab: `lab-bench.sh --model llama-awq --golden16 --c1` and `--model llama-gptq --golden16 --c1` (full 200 req);
   GPTQ eval + eval-compare; soak on AWQ (ask the coordinator first).
4. Flip passing rows in `support.rs` — that is a lead-owned file: send it as a `handoff(crates/turbine-core/src/support.rs)` commit.
5. Upload BENCH/eval results to labbook with vLLM per format as baseline.

## Lab traps

- ONE GPU job at a time (PSU); bench.lock / port locks; fixture jobs one at a time on fixture.lock, paused by
  `fixture-pause.sh` during benches.
- The first GSM8K runs (`gsm8k=0.040/0.025`) used the old bare-number set — void; only the CoT results count.
- novanas crashed twice this morning; re-check fixture logs for their `rc=` lines.
