# Handoff: p6a-int4 (plan Tasks 16–18)

## State 2026-09-29 ~15:30 (T18 builder, lead rotation 7/8)

T18 run (`t18-run/run.log`, commit d14402f): `ALLDONE rc: bf16=0 awq=0 gptq=1`.

AWQ verdict: PASS (full 200 req, GPU 0, same-run BF16 pair):
- c16: 1262.6 vs BF16 842.8 tok/s = 1.50× (≥ 0.9×). c1 ITL p50: 5.87 vs 12.37 ms = 0.47× (≤ 0.6×).
- golden c1 16/16 strict, c16 16/16 batched. GSM8K-200 0.775 vs BF16 0.805 (−0.030, bound 0.04);
  vLLM-ROCm AWQ 485.8/494.2 tok/s, 0.755 → Turbine 2.55×, +0.020.
- Labbook (set phase-6a-quantization): `lab-bench:t18-llama-awq:d14402f` pass,
  `lab-bench:t18-llama:d14402f` (BF16 pair) auto-status fail. The cause is the tok_s relMin 0.97 against the
  previous comparable run, `p6a-fp8-pinned-llama:56720ea` (876.5 tok/s, a 64-request `--quick` run): 842.8 is
  0.962×. golden_c16 has no bound, so it plays no part. Full 200-request BF16 runs sit at 839–853 (t24c 852.7).
  The status is unchanged; the lead decides.

GPTQ start-up failure fixed: 44c30eb `handoff(crates/turbine-model/src/safetensors.rs)`. When
`<dir>/config.json` has `tie_word_embeddings: true`, a listed-but-absent `lm_head.weight` is dropped from
the index (WARN `event="index_entry_absent"`). Every other absence, and an untied or config-less
checkpoint, still errors (test `safetensors::tests::tied_lm_head_listed_but_absent`, red then green).
Lead review needed.

GPTQ-only requeue, started 13:18Z (pid 609880; got the locks at once) detached on novanas (script copy `.procoder/handoff/t18_gptq_run.sh`, on the host
`/home/piwi/turbine-ci/scratch/p6a-int4/t18_gptq_run.sh`; binaries of 44c30eb in
`agent-abee0e2542a317f3c/target/release`): kernel build, then the gptq pass (golden c1 + c16, bench c16
200 req, c1, GSM8K-200) under port18000.lock → bench.gate → bench.lock.
- Log `/home/piwi/turbine-ci/scratch/p6a-int4/t18-run/run-gptq.log`, last line `t18_gptq_run: done rc=<rc>`.
- Done marker `t18-run/gptq/done` (written only on rc=0). Results `t18-run/gptq/{golden1,golden16}.txt`,
  `bench.json`, `bench-c1.json`, `quality.json`, `server.log` (grep `index_entry_absent` to confirm the
  fix fired).
- Golden already in at 13:2xZ: c1 16/16 PASS strict (tail 0.91), c16 16/16 PASS batched.
- Judge against the bf16 pass in `t18-run/bf16/` (842.8 tok/s, c1 ITL 12.37 ms):
  - golden1 and golden16 last lines PASS.
  - bench.json `output_token_throughput` ≥ 758.5 (0.9×).
  - bench-c1.json `itl_ms.p50` ≤ 7.42 ms (0.6×).
  - GSM8K: copy `quality.json` to `tests/eval/llama-3.2-3b-instruct-gptq/turbine.json`, then run
    `turbine-golden eval-compare --baseline tests/eval/llama-3.2-3b-instruct/turbine-bf16.json
    --candidate … --max-drop 0.04` (pass ≥ 0.765). vLLM refuses GPTQ, so there is no vLLM pair.

Remaining after the GPTQ run: commit the GPTQ eval + perf-log rows, labbook GPTQ run, soak on AWQ (ask the
lead first), `handoff(crates/turbine-core/src/support.rs)` flipping the passing awq/gptq rows.

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
