# Handoff: p6a-int4 (plan Tasks 16–18)

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
