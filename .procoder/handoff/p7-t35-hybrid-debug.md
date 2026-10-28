# Qwen3.5 / Qwen3.6 hybrid golden: divergence handoff (2026-10-28)

## State

- `Qwen/Qwen3.6-35B-A3B-FP8` is served on `novanas` at `http://192.168.10.203:18000`
  (`scripts/lab-serve.sh novanas scripts/lab/phase7-novanas-qwen36-tp2.yaml --gpus 2
  --set server.listen=0.0.0.0:18000`); `/ready` 200, TP2, `fp8_block`, support `experimental`,
  speculative `none`. Chat with `chat_template_kwargs:{"enable_thinking":false}` answers.
- Golden c1 against the committed transformers 5.9.0 reference: **1/16 prompts** (Qwen3.6),
  9/16 (Qwen3.5). The gate is open; both rows stay `experimental`.
- Speculative decoding on these families is refused: only the Llama BF16 → BF16 draft pairing
  is a support-matrix row, `speculative_parallel_unsupported` refuses a draft on tp > 1, and no
  same-family small hybrid draft is provisioned.

## Evidence gathered

- The chat template renders the reference ids exactly (logged for p16: 23 ids equal to
  `tests/golden/qwen3.5-9b/reference.jsonl`), and the completions endpoint given the same
  decoded text produces the same answer as the chat endpoint. Render/tokenization is not the
  cause.
- HIP vs CPU (`hip_trace_vs_cpu_qwen3_5`, release, Qwen3.5-9B, p01) passes: both Rust providers
  agree; the divergence is against transformers, not HIP-vs-CPU.
- The divergence is not a near-tie: first-divergence margins are 0.73 (p09), 2.22 (p11),
  0.96 (p12), 1.03 (p14), 6.21 (p16) nats; only p06 (0.015) and p15 (0.0009) are near-ties.
- Qwen3.6 p01 (raw completion prompt, 14 ids): the engine's first tokens are
  `[248046, 198, 248045, 248046, …]` (`<|im_end|>\n<|im_start|>…`, logprob of 248046 = −2.08),
  while the reference starts `[52290, 49847, 11, …]` (52290 = −0.62). The engine emits chat
  scaffolding for a raw-text completion, so the model state/logits are wrong, not the sampler
  comparison (`missing_top_k` is null and the top-5 diff reads 0.0 because the reference's
  candidates are absent).
- `hip_trace_vs_cpu_qwen3_5` previously panicked on the dense Qwen3.5 checkpoint because the
  MoE-only `shared_expert`/`experts` layout check was unconditional; fixed in `0f5b44a0`
  (gated on `family == qwen3_5_moe`).

## Next steps (highest signal first)

1. Dump transformers per-layer hidden states for Qwen3.6 p01 (hooks in
   `modeling_qwen3_5_moe.py`) and compare against the engine's per-layer trace on the same ids
   (extend `hip_trace_vs_cpu_qwen3_5` to emit the CPU-reference trace for the Qwen3.6 p01).
   The first divergent op should be the GDN/attention gate/norm/partial-rotary or the MoE
   router/shared expert.
2. Check the fp8_block load against the checkpoint for the GDN linears and the shared expert
   (the layout assertions cover schemes, not values); compare one layer's BF16 dequantized
   weights against `dequantize_checkpoint.py`.
3. Re-run the Qwen3.6 reference/golden after the fix; then the Task 35 gates (GSM8K, tp2/ep2,
   multi-GPU bounds) before any row flip. Spec decoding for a hybrid needs a provisioned
   same-family draft and TP/EP support, which is separate scope.
