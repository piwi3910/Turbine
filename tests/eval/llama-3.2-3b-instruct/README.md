# Llama-3.2-3B-Instruct GSM8K reports

`turbine-golden eval` reports over `tests/eval/gsm8k-200.jsonl` / `gsm8k-full.jsonl` (1,319 items). The
BF16 baseline of every full-set comparison is `turbine-bf16-full.json` (unsloth @ `006f5dcd…`, 1029/1319 =
0.7801, concurrency 16).

## GPTQ INT4 re-judge (Phase 6a Task 18, user decisions 2026-09-30 "B", then "C then A")

The published `shuyuej/Llama-3.2-3B-Instruct-GPTQ` (damp 0.1, unknown calibration) scored 972/1319 = 0.7369 on
the full set at c16: drop 0.0432 over the 0.04 4-bit bound (lost 122 / gained 65, exact McNemar p = 3.7e-5,
95 % CI of the drop [+0.023, +0.063]). Two better checkpoints follow, each served on gfx1201 at c16 over
`gsm8k-full.jsonl` by `.procoder/handoff/gptq_full_run.sh` (one driver, `--name` / `--model-dir`) and judged
by `scripts/eval/paired_compare.py turbine-bf16-full.json <candidate> --max-drop 0.04` (drop, lost/gained,
McNemar with continuity correction and exact, 95 % CI of the paired drop):

- `turbine-gptq-autoround-full.json` — **AutoRound, an early data point only; it does not decide the
  `gptq_int4` row.** `kaitchup/Llama-3.2-3B-Instruct-AutoRoundGPTQ-4bit` @
  `e11f15d2291d8c343a4de84d6bb16ebf7c871dfc` (auto-round 0.4.5, 500 iters, 512 samples × 2048, sym, group 128,
  `desc_act` false, GPTQ v1 packing), served through Turbine's `gptq` packaging.
- `turbine-gptq-own-full.json` / `vllm-gptq-own-full.json` — **our own GPTQ checkpoint, which decides the
  row.** Made by `scripts/eval/gptq_calibrate.py` (llm-compressor `GPTQModifier`, W4A16: 4-bit int, symmetric,
  group 128, dampening 0.01, no act order, `lm_head` kept BF16) from the BF16 unsloth checkpoint above, with
  512 conversations of `HuggingFaceH4/ultrachat_200k` @ `8049631c405ae6576f93f445c6b8166f76f5505a` (split
  `train_sft`, shard `train_sft-00000-of-00003`, shuffle seed 42, chat template, ≤ 2048 tokens), on one R9700.
  Written as compressed-tensors `pack-quantized` (Turbine `ct_pack_int4` → `gptq_int4` row); the exact
  package versions (torch + HIP, llm-compressor, compressed-tensors, transformers) are recorded in the
  checkpoint's `turbine_calibration.json` and copied below when the run lands. The vLLM-ROCm run serves the
  same checkpoint (or records its refusal).
