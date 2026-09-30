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

- `turbine-gptq-autoround-full.json` / `turbine-gptq-autoround-paired.json` — **AutoRound, an early data
  point only; it does not decide the `gptq_int4` row.** `kaitchup/Llama-3.2-3B-Instruct-AutoRoundGPTQ-4bit` @
  `e11f15d2291d8c343a4de84d6bb16ebf7c871dfc` (auto-round 0.4.5, 500 iters, 512 samples × 2048, sym, group 128,
  `desc_act` false, GPTQ v1 packing), served through Turbine's `gptq` packaging: 998/1319 = 0.7566, drop
  0.0235 ≤ 0.04 PASS (lost 95 / gained 64, McNemar exact p 0.0171, 95 % CI of the drop [+0.0048, +0.0422]).
- `turbine-gptq-own-full.json` / `vllm-gptq-own-full.json` (+ `*-paired.json`) — **our own GPTQ checkpoint,
  which decides the row.** Made by `scripts/eval/gptq_calibrate.py` (llm-compressor `GPTQModifier`, W4A16:
  4-bit int, symmetric, group 128, dampening 0.01, no act order, `lm_head` kept BF16) from the BF16 unsloth
  checkpoint above, with 512 conversations of `HuggingFaceH4/ultrachat_200k` @
  `8049631c405ae6576f93f445c6b8166f76f5505a` (split `train_sft`, shard `train_sft-00000-of-00003`, shuffle
  seed 42, chat template, ≤ 2048 tokens), on one R9700. Written as compressed-tensors `pack-quantized`
  (Turbine `ct_pack_int4` → `gptq_int4` row). Calibration ran the eager-torch GPTQ path
  (`LLMCOMPRESSOR_DISABLE_GPTQ_TRITON=1`) with compressed-tensors' compress-time quantize also forced off
  Triton (`compressed_tensors.utils.triton.HAS_TRITON = False`, monkeypatched in the script): ROCm Triton
  3.7.1's AMD backend cannot lower two independent CUDA-only kernels reached by this recipe (llm-compressor's
  fused GPTQ block update and compressed-tensors' pack-quantize), so both run the reference eager-torch math
  instead — same algorithm, no change to the quantization result. Package versions recorded in the
  checkpoint's `turbine_calibration.json`: torch 2.13.0+rocm7.2 (HIP 7.2.53211), llmcompressor 0.14.0,
  compressed-tensors 0.19.0, transformers 5.17.0, accelerate 1.15.0, datasets 5.0.1, safetensors 0.8.0,
  python 3.12.14; calibration took 1099.6 s.

  Turbine on the own checkpoint: 988/1319 = 0.7491, drop 0.0311 ≤ 0.04 PASS (lost 106 / gained 65, McNemar
  p 0.00222, exact p 0.00213, 95 % CI of the drop [+0.0117, +0.0504]). vLLM-ROCm on the same checkpoint:
  988/1319 = 0.7491 (identical accuracy; lost 105 / gained 64, McNemar exact p 0.002) — Turbine vs vLLM drop
  0.0000. No circuit transitions in either run.
