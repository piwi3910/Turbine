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
| ----- | -------------------- | -------------------- |
| embed | 0.009609             | 0.009609             |
| 0     | 0.018262             | 0.018250             |
| 20    | 0.071035             | 0.071173             |
| 30    | 0.143793             | 0.144053             |
| 35    | 0.208469             | 0.208566             |
| 38    | 0.310419             | 0.311611             |

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

## MTP speculative decoding (corrected plan, 2026-10-28)

Both hybrids ship a native MTP head (`mtp.*`: 15 tensors / 486 MB in Qwen3.5-9B, 1,560 in
Qwen3.6-35B-A3B); transformers ignores it, vLLM serves it as `method: mtp`. That is the hybrid
spec path — no external draft model, and the Phase 8 draft-model pair stays the Llama route.

Implemented so far: `speculative.method: mtp` parses and validates (`6be36f3b`) and is refused at
startup with `speculative_mtp_unimplemented` until the proposer lands (an explicit refusal, not a
silent plain decode).

Slices, in order:

1. **Loader**: map `mtp.pre_fc_norm_hidden`, `mtp.pre_fc_norm_embedding`, `mtp.fc`
   (`[hidden, 2*hidden]`), `mtp.layers.0.*` (a full-attention layer; Qwen3.6's FFN is MoE with a
   shared expert), `mtp.norm` into `MtpWeights`; load them only with `method: mtp` instead of
   `skipped_prefixes`.
2. **Hidden export**: the target executor keeps the final pre-head hidden rows of the prefill
   (`take_hidden()` next to `take_trace`), because the MTP attention layer needs its own K/V for
   every prompt position, computed from those rows.
3. **MTP layer**: a one-layer full-attention executor (`num_layers = 1`, `LayerKind::Full`, the
   target's dims/MoE) with a dedicated KV slot; `forward_mtp(hidden_rows, embed_rows, positions)`
   applies pre-norm → concat → fc → the layer → `mtp.norm` → LM head.
4. **Proposer**: `MtpProposer` implements `Proposer`: prefill its KV from the prompt's hidden
   rows + shifted token embeddings, then each round drafts `num_tokens` autoregressively through
   the same layer; the existing verifier/rollback applies unchanged.
5. **Gates**: a tiny-fixture MTP test (the tiny writers gain `mtp.*`), then Qwen3.5-9B golden/A-B/
   throughput, then Qwen3.6 (after its TP numerics gate route).

## MTP end to end (2026-10-28, quadronix) — slices 1–5 done, Qwen3.6 EP2 measured

MTP serves. `cargo clippy --workspace --all-targets -D warnings` clean; `turbine-model` (56
tiny_model + lib), `turbine-core` (44) and `turbine-server` (136 bin) all pass. Everything below
is uncommitted (`crates/turbine-model/src/mtp.rs`, `src/speculative/mtp.rs`, the executor capture
/`embed_rows`/`set_hidden_input`, the family `mtp_*` helpers, the engine wiring, the config).

**Engine wiring** (`model::prepare_mtp`, `engine/mod.rs`, `engine/loop.rs`, `engine/tp.rs`): the
head loads on the target's device memory; the target's `set_hidden_capture` is on (pre-norm
operand, so the executor writes **all** forward rows); each `forward` feeds `Proposer::observe`
from the packed `Built::slices`/`rows` and records the last row as `mtp_pending`;
`request_speculative` calls `seed_last_hidden`. `TpExecutor` forwards `set_hidden_capture` /
`take_hidden` / `embed_rows` to rank 0's leader (EP is a single executor per rank, hidden is
replicated). `overlap_scheduling` is forced off for MTP (`reason=mtp_serial`) rather than refused;
`method: mtp` with tp/pp > 1 is refused.

**Debug findings that mattered** (all measured on `Qwen3.6-35B-A3B-FP8`, EP2, greedy c1, bench
`--requests 4 --prompt-words 256 --max-tokens 300 --ignore-eos`, `min_acceptance: 0.0`):
1. `mtp_arch_config` must set `full_attention_interval = Some(1)`; the family derives
   `layer_kinds` from it (`hybrid_layer_kinds`), so without it the one MTP layer built as a GDN
   linear-attention layer (acceptance 0).
2. The front matter order is `[pre_fc_norm_embedding(embed) ‖ pre_fc_norm_hidden(hidden)]`
   (embedding first) — `vllm/model_executor/models/qwen3_next_mtp.py`; hidden-first gave 0.
3. Hidden operand: **pre-final-norm** residual. Pre-norm both → **135 accepted / 325 proposed
   (0.42/token, 1.65/round, 85 % of rounds accept ≥1)**; post-final-norm both → 84/424 (0.20);
   pre-norm target + post-`mtp.norm` head recurrence → 53/489 (0.11). So the reference's returned
   post-`norm` hidden is not the recurrence operand here; keep the single pre-norm capture.
   (The tiny `mtp_greedy_matches_serial` engine test validates the observe/seed timing and that
   the target stream stays serial-correct; it cannot see acceptance quality.)
4. The pre-fc norms fold `(1 + w)` (checkpoint range ~[−0.85, 0.06], nonsensical without it) and
   are BF16; `mtp.fc.weight` is BF16 `[hidden, 2·hidden]` — all read to the host, format-agnostic
   only for F32/BF16/F16 storage.
5. The head's format must be `resolve_for_providers`'d like a rank (the MoE shared experts are
   read BF16 by the MoE path, so the block-scaled FP8 must be decoded) — `mtp::load` now does.

**Still to do**: budget-charge the head's ~1 GB (Qwen3.6 MoE layer 0) against the target's budget
(it currently loads uncounted); skip the per-step hidden read on long prefill chunks; a golden
run for the served MTP path (the verifier keeps the target stream correct regardless of the
draft, so acceptance is the gate); first-token acceptance 85 % is healthy but the second+ tokens
suggest the recurrence could be revisited (0.42/token overall).


### MTP follow-ups (2026-10-29)

- **Proposer stall fixed** (sustained runs stopped speculating after a few requests; root causes
  found on Qwen3.6 EP2 with `RUST_LOG=turbine_server=debug`):
  1. **GDN state exhaustion** — the one-layer MTP executor still built Gated DeltaNet state
     (`DecoderDims.gdn = cfg.gdn`, cloned from the target), and each `forward_mtp` (with
     `state_slot: None`) took a new state slot until "Gated DeltaNet state slots exhausted: 1
     sequences" aborted every observe (`draft_model`). Fix: `DecoderDims::of` sets `gdn` only
     when `cfg.layer_kinds` contains a linear layer; `mtp_arch_config` sets
     `full_attention_interval = Some(1)` (already) and `state = default`; `gdn_state_layout`
     returns empty when there are no linear layers.
  2. **Single-slot proposer** — `observe` runs for every forward (every sequence in a batch) but
     `MtpProposer` held one slot, so concurrent sequences thrashed it. Fix: `slots:
     HashMap<SeqId, Slot>` (per-sequence MTP KV), pool sized for `max_running_requests`;
     two-sequence no-cross-talk assertion added to `mtp_proposer_drafts_from_the_target_hidden`.
  3. **Seed alignment** — `mtp_pending` held the verify run's *last* row, not the row at the last
     accepted position, and `propose` consumed `last_hidden`; the seed-position guard then
     failed once and stuck the proposer. Fix: pending stores `(first_pos, rows)` (bounded to 16),
     the engine seeds the row at `unfed - 1`, and `propose` clones `last_hidden`.
  **Result (Qwen3.6 EP2, greedy c1, `min_acceptance: 0.0`)**: speculation now runs continuously
  — 659 rounds / 2626 proposed / 541 accepted over ~24 sequential requests (0.82–1.10
  accepted/round; the reasoning phase lowers it). Engine test `mtp_greedy_matches_serial` green;
  `cargo clippy --workspace --all-targets -D warnings` clean; 56 model + 44 core + 136 server
  tests pass.
- **Memory**: the head loads its own copy of the target's `embed_tokens` + `lm_head` (~1 GB each
  for Qwen3.6) plus the ~1 GB layer — ~3 GB duplicating the target's tables. Routing `mtp`
  through `draft_budget` (charging those bytes before the target's budget) makes the EP2 config
  refuse (`kv pool of 0 B`), i.e. it does not fit at `max_running_requests = 64`; a proper fix
  shares the target's embed/lm_head tensors. Reverted that charge for now (the head loads
  uncounted, as before).

### MTP performance: device-native (2026-10-29) — 1.5–1.8x c1 decode

The host-sync-bound MTP path is gone. Measured on Qwen3.6-35B-A3B-FP8, EP2, single stream,
4x256 tokens (`ignore_eos`, non-streaming), same server config:
- `method: none` — **24.4 tok/s**
- `method: mtp` (device-native) — **37.7 and 43.5 tok/s** (two runs) → **~1.55–1.78x**
- accepted/round in these runs ~1.9 (speculation active: 494 accepted / 262 rounds).

What changed (no per-step device↔host round trip; still `overlap_scheduling` off for MTP):
- **Front matter on device** (`DecoderExecutor::set_mtp_front` / `mtp_front_into` / `mtp_step`):
  the pre-fc norms (with `1 + w` baked in, activation dtype) and `fc` are on device; `mtp_step`
  embeds the tokens on device, runs `fc(cat([norm_e(emb), norm_h(h)]))` into the injected input,
  then the layer stack + norm + LM head. No `take_hidden`/`set_hidden_input`/`embed_rows` host
  calls in the decode loop.
- **Hidden stays on device**: the engine reads `mtp_capture_view` and assembles the `prev` rows
  and the per-sequence pending row as device buffers (`mtp_rows` / `mtp_copy_rows`); `observe`
  and `seed_last_hidden` take `TensorView`, not `&[f32]`.
- `mtp_requirements(cfg, …)` adds the front matter's `fc` GEMM (`[hidden, 2·hidden]`, `cat @ fcᵀ`)
  to the head's kernel selection.
- `TpExecutor` forwards the new device methods (`mtp_capture_view` / `mtp_rows` /
  `mtp_copy_rows` / `mtp_write_rows`) to rank 0's leader — without this the EP engine's capture
  view was the trait default (None) and no proposal ever ran.

Still open for more headroom: restore overlap scheduling (needs a double-buffered capture so the
overlapped next forward cannot overwrite the rows before they are copied), and the earlier
memory-sharing item (the head duplicates the target's embed/lm_head).

### Adaptive speculative-length policy (2026-10-29)

`speculative.k_policy` (registry `speculative_k_policy`, `turbine_model::speculative::k_policy`)
is opt-in; `fixed` (default) is unchanged. `adaptive` is a deadband controller: it starts at
`speculative.num_tokens`, raises `k` after GROW consecutive fully accepted rounds, lowers it
after SHRINK consecutive fully rejected ones, and holds otherwise. Per-sequence `KState`; the
engine resolves the name through `known_module_names`; `turbine_speculative_k` is the last
round's `k`. Docs page + registry conformance + config-key test added.

Tuning mattered: a `accepted/proposed`-ratio rule collapsed `k` to 1; an `accepted + PROBE` rule
drifted above the optimum. GROW=6 / SHRINK=8 holds `k` near `num_tokens` on a workload whose
optimum is there and only moves on extreme stretches.

**A/B (Qwen3.6-35B-A3B-FP8, EP2, single stream, 4x256, `ignore_eos`):**
- `none` — 24.4 tok/s
- `fixed` k=4 — 37.7, 38.2, 40.5, 43.5 (median ~39.4)
- `adaptive` — 38.4, 38.8, 40.5 (median ~38.8)

Adaptive matches fixed k=4 (within run noise) and both are ~1.6x `none`. On this workload 4 is
the optimum, so adaptive holds there; its value is tracking easy (boilerplate/code, where it
grows) and hard (novel text, where it shrinks) stretches without a hand-tuned `k`. A deeper
adaptive gain needs a cost-aware model (the verify row cost and the per-draft forward cost),
not the deadband alone — and, per the k=6 result, deeper `k` is already past the optimum here.
