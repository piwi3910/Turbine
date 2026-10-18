# Qwen3.6-35B-A3B-FP8 golden reference — generated on CPU (Task 35, 2026-10-17)

`Qwen/Qwen3.6-35B-A3B-FP8` at revision `95a723d08a9490559dae23d0cff1d9466213d989` is a block-FP8
(`fp8_block`, `weight_block_size` `[128, 128]`) hybrid MoE: Gated DeltaNet (`linear_attn.*`) and
full-attention layers with 256 routed experts. The reference was generated from the dequantized BF16
copy (`scripts/golden/dequantize_checkpoint.py`), thinking off, exactly as the Qwen3.5-9B one
(`tests/golden/qwen3.5-9b/README.md`).

## The run

`hf_reference.py` on CPU is slow for this checkpoint — the Gated DeltaNet torch fallback has no fused
kernel on CPU, ~30 min per short prompt and hours for the 2,091-token `p09` — so the 16-prompt run
took ~8 h on `novanas` (MemAvailable 116 GB > the 90 GB rule; `cpuhost` had only 17 GB free). It ran
under `fixture.lock` and is resumable (`hf_reference.py --out` seeds from a partial `.tmp`), which is
how it completed after an earlier 11/16 checkpoint.

`reference.jsonl` — 16 records, engine `transformers-5.9.0-bf16-cpu-fp32-logits`, model
`Qwen/Qwen3.6-35B-A3B-FP8`.

## Why the row stays open

The same family numerics that keep the Qwen3.5-9B row open apply here — the Gated DeltaNet recurrence
amplifies BF16 rounding, so the standard golden prefix criterion cannot be met (see
`tests/golden/qwen3.5-9b/README.md` and decisions.md "P7 hybrid golden gates"). The reference is
committed so a future recurrence-aware gate or a GPU-transformers re-run can use it.

## What is here

- `prompts.jsonl` — the shared 16 prompts with `chat_template_kwargs: {"enable_thinking": false}`.
- `reference.jsonl` — the committed reference (above).
- `tolerance.json` — the seeded floor (Llama's quantized bounds); a recurrence-aware derivation is
  still open.
- `gate.json` (`tests/eval/qwen3.6-35b-a3b-fp8/`) — `max_drop` 0.02 (FP8), ready for the GSM8K pair.

Recorded as `- P7 open: Qwen3_5MoeForConditionalGeneration/fp8_block/bf16/{tp2,ep2}` in
`.procoder/perf-log.md`; the `SUPPORT_MATRIX` row stays `experimental`.
