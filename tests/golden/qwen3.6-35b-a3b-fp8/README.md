# Qwen3.6-35B-A3B-FP8 golden reference — not generated (Task 35, 2026-10-17)

`Qwen/Qwen3.6-35B-A3B-FP8` at revision `95a723d08a9490559dae23d0cff1d9466213d989` is a
block-FP8 (`fp8_block`, `weight_block_size` `[128, 128]`) hybrid MoE: Gated DeltaNet
(`linear_attn.*`) and full-attention layers with 256 routed experts. The reference would be
generated from the dequantized BF16 copy, thinking off, exactly as the Qwen3.5-9B one
(`tests/golden/qwen3.5-9b/README.md`).

## Why it is missing

The CPU transformers reference is not practical for this checkpoint. The dequantized BF16 copy
is produced in ~3 min (`scripts/golden/dequantize_checkpoint.py --model-dir
/home/piwi/turbine-models/qwen3.6-35b-a3b-fp8 --out …-bf16`, 71.9 GB, 18 shards), but
`hf_reference.py` on CPU runs at ~30 min per short prompt — the Gated DeltaNet torch fallback has
no fused kernel on CPU — and the 2,091-token `p09` alone takes hours, so a 16-prompt reference is
~8–12 h on `novanas`. `novanas` satisfies the 90 GB MemAvailable rule (116 GB at start), and
`cpuhost` (TrueNAS) has only 17 GB free, so the fallback host is not available either.

The same family numerics that keep the Qwen3.5-9B row open (the Gated DeltaNet recurrence
amplifies BF16 rounding; see `tests/golden/qwen3.5-9b/README.md`) apply here, so the row cannot
close regardless of the reference.

## What is here

- `prompts.jsonl` — the shared 16 prompts with `chat_template_kwargs: {"enable_thinking": false}`
  on every record (ready for a future `hf_reference.py` run).
- `tolerance.json` — the seeded floor (Llama's quantized bounds); no reference to derive from yet.
- `gate.json` (`tests/eval/qwen3.6-35b-a3b-fp8/`) — `max_drop` 0.02 (FP8), ready for the GSM8K
  pair.

Recorded as `- P7 open: Qwen3_5MoeForConditionalGeneration/fp8_block/bf16/{tp2,ep2}` in
`.procoder/perf-log.md`; the `SUPPORT_MATRIX` row stays `experimental`.
