# Handoff: p6a-mxfp4 (plan Tasks 19–20)

Written 2026-09-29 ~08:00 by the new 6a lead (builder killed without a handoff).
Worktree `.claude/worktrees/agent-a4784842b25c93376`, tip 7ca32e3 (= 2116221 + the lab-serve `--vllm` slug for
the AMD 8B W4A4 checkpoint, identical to integration's 62ab8b7). Read AGENTS.md and plan Task 20 first.

## Done (on phase-6a-quantization)

- Task 19: c1aea74 (MXFP4 weight-only GEMM + MXFP4 activation emulation), handoffs 75a02d6, 41803d1, a3a7db9,
  4495714; 5babb2a (evaluation entry, accepted by the user).
- 2116221: Quark `kv_cache_quant_config` ignored with WARN `kv_cache_quant_ignored` (verified by the lead:
  `weights/quark_mxfp4.rs` + a detect test in `weights/mod.rs`); lab-bench model `llama8b-mxfp4-a4`.

## User decisions in force (2026-09-29, final)

- W4A4 golden + chat: `amd/Llama-3.1-8B-Instruct-MXFP4-W4A4-MLCAL-C1000-GPTQ` @ 00b0d018 in
  `/home/piwi/turbine-models/llama-3.1-8b-instruct-mxfp4-a4`; KV recipe ignored; reference built the same way
  (activation fake-quant, BF16 KV); accuracy baseline BF16 Llama-3.1-8B-Instruct.
- The base 3B W4A4 checkpoint (`matmelis/…`) is judged against BF16 Llama-3.2-3B **base**, reported as a side result.

## Uncommitted (in progress)

- `benches/turbine-bench/tests/golden.rs`: the W4A4 slug switched from the 3B base to the 8B Instruct checkpoint.
  Probably should *add* the 8B entry and keep/drop the 3B one per whether a 3B golden fixture is kept (the 3B has
  no chat template; the README in `tests/golden/llama-3.2-3b-mxfp4-a4/` + `chat_template.jinja` date from the
  overnight "borrow the template" pick, now superseded — decide: completion-only side result or delete).
- Fixtures: `tests/golden/llama-3.1-8b-instruct/{reference.jsonl,tolerance.json,README.md}` (BF16 8B, complete),
  `tests/golden/llama-3.1-8b-instruct-mxfp4a16/README.md` and `…-mxfp4-a4/README.md` (no reference yet).
- Evals: `tests/eval/llama-3.1-8b-instruct/turbine-bf16.json` (0.89) and
  `tests/eval/llama-3.1-8b-instruct-mxfp4a16/turbine-bf16.json` (0.835) — **the second is misnamed**: it is the
  MXFP4-A16 server's result (`scratchpad/mxfp4/ev2/llama8b-mxfp4.json`); rename to `turbine.json`.
  `gate.json` (BF16 baseline, max drop 0.04; vLLM refuses the checkpoint on gfx1201). **0.89 → 0.835 is a 0.055 drop:
  the MXFP4-A16 eval gate FAILS as it stands.** Report to the lead before changing anything.
- Lab configs `scripts/lab/phase6-novanas-llama{8b,8b-mxfp4,8b-mxfp4-a4,-mxfp4-a4,3b-base}.yaml`.
- Bench so far: BF16 8B 401.4 tok/s, ITL 31.9 ms, c1 ITL 27.9 ms, golden1 PASS, golden16 FAIL (look at why:
  8B BF16 tolerance is the 3B one, perhaps too tight — needs its self-spread).

## Running (let finish)

- `scratchpad/mxfp4/queue3.sh` (local pid 8681): W4A4 8B GSM8K eval (`mxeval.sh`, waiting for port18000) → vLLM
  8B-a4 attempt on port18100 → `lab-bench --golden16 --c1` for llama8b, llama8b-mxfp4, llama8b-mxfp4-a4 (the
  latter two need their golden fixtures first, so their golden lines will fail until those land).
- Remote `fixq.sh` (`/home/piwi/turbine-ci/remote/agent-a4784842b25c93376/scratch/fixtures/out/fixq3.log`,
  fixture.lock): MXFP4-A16 8B `quant_reference.py --act-quant none`, then (check the script) the W4A4 8B reference
  `--act-quant mxfp4` and spreads.

## Exact next steps (fresh builder, default model)

1. Wait for fixq; calibrate tolerances (self_spread on the dequantized copies, `--act-quant mxfp4` for W4A4); BF16 8B
   self-spread too (golden16 fail).
2. Rename the misnamed eval; W4A4 8B eval-compare vs BF16 8B; 3B base side result vs BF16 3B base.
3. Full `lab-bench --golden16 --c1` for the three 8B models; labbook upload; rows via a `handoff(support.rs)` commit.
4. Escalate the MXFP4-A16 GSM8K drop (0.055 > 0.04) to the lead with the per-item diff.

## Lab traps

- ONE GPU job at a time; 8B runs take ~2× the 3B time — use `timeout`s from queue3.sh.
- A queue3 bench was killed by the 07:00 crash ("Connection reset by peer" in `bench-llama8b-mxfp4.log`).
