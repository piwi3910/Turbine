# gpt-oss-20b golden fixture

Phase 7 Task 27 (spec S-8, S-13), fixture half. The GSM8K pair, throughput and the
`openai_mxfp4` support rows are the gate half, run against the committed reference after Task 26
merges.

## Reference

- Checkpoint: `openai/gpt-oss-20b` at revision `6cee5e81ee83917806bbde320786a8fb61efebee`
  (`/home/piwi/turbine-models/gpt-oss-20b` on `novanas`; `quant_method: mxfp4`,
  `modules_to_not_convert`: `model.layers.*.self_attn`, `mlp.router`, `embed_tokens`, `lm_head`).
- `reference.jsonl`: transformers 4.57.1, BF16 on CPU, FP32 LM head on the BF16 final-norm output
  (`scripts/golden/hf_reference.py`, engine string
  `transformers-4.57.1-bf16-cpu-fp32-logits`), 16 prompts × 32 tokens, greedy, `ignore_eos`. The
  weights it reads are the checkpoint dequantized to BF16
  (`/home/piwi/turbine-models/gpt-oss-20b-bf16`, 39 GB), written by
  `scripts/golden/dequantize_checkpoint.py` extended for the checkpoint's fused-expert storage:
  the experts live as `…mlp.experts.<proj>_blocks` U8 `[E, n, k/32, 16]` with `<proj>_scales`
  `[E, n, k/32]` (the same rule as `turbine-model/src/weights/mxfp4.rs`
  `fused_expert_tensor`), decoded with the script's standard MXFP4 decode (E2M1, low nibble
  first, E8M0 ldexp, F32 math rounded to BF16) and written as the fused BF16 parameter
  `<proj>` `[E, k, n]` that plain transformers' `GptOssExperts` loads — the layout
  transformers' own dequantize (`convert_moe_packed_tensors`, ending in `transpose(1, 2)`)
  produces. The decode is bit-exact against transformers' own: the generator ran with
  `--verify-transformers 2` (first 2 and last fused expert tensors, gate_up and down of the
  first and last layer, bit-equal; `turbine_dequant.json` beside the BF16 copy records the
  layers). Attention, router, embedding and `lm_head` are BF16 in the checkpoint and are copied
  as stored.
- `reasoning_effort: low` in the system message of both runs (decision Harmony A): the fixture's
  `prompts.jsonl` is the shared `tests/golden/prompts.jsonl` with `reasoning_effort: "low"`
  added to every record's `chat_template_kwargs` — `hf_reference.py` passes them to
  `apply_chat_template` and `turbine-golden` replays the same kwargs through the chat route, so
  reference and candidate tokenize identically. The `date_string` kwarg of the shared file is
  kept (unused by the gpt-oss template).
- Tolerances and engine captures are not part of the fixture half; the gate half re-derives the
  bounds from the engine spread (Qwen3-MoE method) before `lab-bench --golden16`.

## Fixture provenance

- novanas CPU, one fixture job at a time under `fixture.lock` (queue entry `p7-ref-gpt-oss`),
  `nice -n 19`, cores 12–15, 4 threads; guarded script
  `.procoder/handoff/p7_ref_gptoss.sh`, scratch `/home/piwi/turbine-ci/scratch/p7-t27`.
- Dequantize + transformers cross-check: 2026-10-14 18:34:36–18:37:21 UTC (2 m 45 s; the first
  two attempts failed on the fused-expert support the extension adds — root causes in git
  history).
- Reference: 2026-10-14, 18:40:04 UTC start, 16 prompts, ~2 min/prompt.
