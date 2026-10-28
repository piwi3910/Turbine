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

## Update (2026-10-28)

- The Qwen3.5 model is correct: an engine CPU trace vs transformers 5.9.0 hooks on p16 matches
  to BF16 rounding at every layer (embed 0.009290 = 0.009290, layer-0 resid 0.079740 = 0.079740,
  final norm 2.008 vs 2.012), and HIP matches CPU. The golden failures were **not** the model.
- They were hybrid prefix reuse running on a stale recurrent state: the snapshot buffer was
  sized one entry per L0 block (15.2 GB for a 6 GiB pool), failed to allocate, and reuse stayed
  on without state. Fixed by sizing entries per `hybrid_snapshot_interval_blocks` (103 MB);
  Qwen3.5 golden is now **14/16** (p03 identical 32/32 but 0.639 vs the 0.56 tail bound, p09 a
  0.025-margin near-tie).
- A TP hybrid cannot copy state yet (`hybrid_state_copies` refuses), so TP prefix reuse was also
  reusing KV without state; the TP engine now disables prefix sharing for hybrid starts. Qwen3.6
  is still 1/16 with reuse off, so its remaining failure is the FP8/model path (or the 35B
  reference), not reuse. Next: run the same per-layer diff on Qwen3.6 against a dequantized BF16
  reference (`dequantize_checkpoint.py`) or the FP8 transformers model, starting at p01.
- Tool: `scripts/golden/hf_layer_trace.py --model-dir <dir> --ids '<json list>'` prints the
  transformers per-layer summaries the engine's `TURBINE_TRACE_DUMP` (now printed by
  `hip_trace_vs_cpu_qwen3_5`'s first step) gives.

## Qwen3.6 per-layer result (2026-10-28)

Reference: the checkpoint dequantized to BF16 by `dequantize_checkpoint.py`
(`/home/piwi/turbine-models/qwen3.6-35b-a3b-bf16-deq`, 71.9 GB) and traced with
`hf_layer_trace.py` on p01's 14 ids. Engine: `hip_trace_vs_cpu_qwen3_5` CPU-reference dump on
the FP8 checkpoint.

| layer | engine resid_mlp rms | reference hidden rms |
| --- | --- | --- |
| embed | 0.009609 | 0.009609 |
| 0 | 0.018262 | 0.018250 |
| 20 | 0.071035 | 0.071173 |
| 30 | 0.143793 | 0.144053 |
| 35 | 0.208469 | 0.208566 |
| 38 | 0.310419 | 0.311611 |

The drift is gradual and small (≤1 % through layer 20, ~0.4 % at 38), not a structural bug; the
engine's own HIP-vs-CPU trace agrees. The committed golden reference comes from transformers on
the **FP8** checkpoint, so the next authoritative comparison is an FP8 transformers trace (CPU
fp8 may be unsupported) or an engine-vs-reference top-5 logits comparison on p01. The BF16-deq
copy and its trace are for diagnosis only and are not the golden reference.

## Root cause of the Qwen3.6 golden gap (2026-10-28)

The Qwen3.6 TP2 path is **known to diverge from the one-device reference**: `qwen3_5_moe_tp2_trace_leg`
documents "the one-device greedy agreement is 1/3, and `resid_attn` exceeds the bound from layer 15
on", and the test was relaxed to (a) rank-vs-rank bitwise agreement, (b) structure/counts, (c) only
the first full-attention layer held to the trace bound. The Gated DeltaNet value-head sharding (32
v_heads → 16 per rank, with the 16 k_heads' repeat-interleave mapping into the `[q|k|v]` concat)
is the suspect: a wrong q/k slice for a rank's v-head group changes the recurrence and drifts the
residual stream, while the LM head and logits path are contiguous and correct by inspection.

So the 1/16 is the TP sharding accuracy gap, not FP8 dequant, prefix reuse (tested off, unchanged)
or device sampling (tested off, unchanged: same token 279 at -2.867 vs the reference's 52250 at
-0.62). Fix order: the GDN `in_proj_qkv`/conv shard mapping in `tp.rs` (`SlotSource` three-part
slice), then re-run the tp2 trace leg expecting greedy agreement ≫ 1/3, then the Qwen3.6 golden.
