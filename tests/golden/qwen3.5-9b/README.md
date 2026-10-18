# Qwen3.5-9B golden reference

`Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (BF16), run through
Hugging Face transformers 5.9.0 on CPU with the LM head in FP32 on the BF16 final-norm output
(the Llama reference's convention, user decision 2026-09-26). The family is a hybrid: every
`full_attention_interval`-th layer is full attention, the rest are Gated DeltaNet
(`linear_attn.*`), whose `A_log` and `norm.weight` are stored F32 and cast to the model dtype.

## Fixture

- `prompts.jsonl` — the shared 16 prompts with `chat_template_kwargs: {"enable_thinking": false}`
  on every record, so the reference render (`hf_reference.py`) and the replayed render
  (`turbine-golden compare/capture/eval --chat-template-kwargs '{"enable_thinking":false}'`) both
  run thinking off.
- `reference.jsonl` — 16 prompts × 32 tokens, engine `transformers-5.9.0-bf16-cpu-fp32-logits`.
- `tolerance.json` — the seeded floor (Llama's quantized bounds). **Not** the derived bounds: the
  gate is open (below), and the engine spread's proposed bounds (8.59 / 11.30) would make it
  vacuous.
- `engine_spread.json` — `scripts/golden/engine_spread.py` over the
  `scripts/lab/golden-engine-spread.sh p7t35 qwen3.5-9b` captures (commit `c1fdfce4`, R9700 0).

## Gate — open (Task 35, 2026-10-17)

The golden gate **fails**: only 9 of 16 prompts (p01–p05, p07, p08, p10, p13) keep a 32-token
identical prefix, against `min_prompts_passing` 16. `engine_spread.py` reports
`vs ref c1-cold prefix 11/16`, likely 1.78 / tail 5.29, engine noise likely max 8.43 / tail max
11.36 — **OVER CAP**.

Where the tokens agree the engine's top candidates match the reference within ~0.5 nats (p10
position 0: token 760 −0.030 vs −0.033, 42373 −3.688 vs −3.609), so the mismatch is not a gross
bug. The other 7 prompts flip a near-tie (p06 at position 9, p09 at 0, p11 at 1,
p12/p14/p15/p16 at 0–2), and the flip is config-independent — the default server, the
`gemm_autotune=false` heuristic and the `fused_ops=false` captures all flip the same prompts. The
Gated DeltaNet recurrence amplifies BF16 rounding over the sequence, so the difference compounds
and no meaningful tolerance closes it.

This is a family correctness question (Turbine's Gated DeltaNet vs transformers' torch fallback on
the real checkpoint), recorded as `- P7 open: Qwen3_5ForConditionalGeneration/bf16/bf16/tp1` in
`.procoder/perf-log.md`; the `SUPPORT_MATRIX` row stays `experimental`.

## Regenerating

```
uv run --with torch==2.9.0 --with transformers==5.9.0 --with safetensors==0.6.2 \
  scripts/golden/hf_reference.py --model-dir /home/piwi/turbine-models/qwen3.5-9b \
  --prompts tests/golden/qwen3.5-9b/prompts.jsonl --out tests/golden/qwen3.5-9b/reference.jsonl \
  --model-name Qwen/Qwen3.5-9B
```

On `novanas` under `fixture.lock` (`nice -n 19 taskset -c 12-15`, 4 threads): the 2,091-token
`p09` dominates (~18 min), the whole run ~1 h.
