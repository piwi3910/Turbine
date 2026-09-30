# Handoff: p6b-t8 (Phase 6b Task 8, TurboQuant GPU transcode)

Branch `p6b-t8` from 0bae895 (the Task 5 kernel commit merged with `p6b-stack`). Step 1 of the task is done; steps 2
to 4 are NOT started.

**Checkpoint: stopped on an ABI question (below), per the task rule "no new ABI fields unless unavoidable — if
unavoidable, stop and report first".**

## Done

- Provider evaluation, decisions entry "P6b: TurboQuant transcode — provider evaluation (kernel reuse rule)": vLLM's
  TurboQuant is Triton / FlyDSL (Python) and a different codec (no seeded signs, no QJL, uniform V, FP16 norm); SGLang
  has only a generic CUDA FWHT; CK has no Hadamard; llama.cpp `ggml-cuda/fwht.cu` (MIT, HIP-built) has the codec's
  butterfly order but scales before the butterflies. Pick: own kernel `turbine_hip_tq` in `kv_transcode_tq.hip`,
  reusing llama.cpp's wave32 shuffle-then-register FWHT pattern with the scale moved after.

## The ABI question

The descriptor carries `seed` only. The codec needs, per (layer, KV head): K and V signs (integer SplitMix64 of the
seed — the GPU can regenerate them bit-exactly), the four codebooks (plan: "codebooks and seeds passed through
`turbine_kv_transcode_desc`") and the 128 × 128 QJL projection `S`, which `qjl.rs` generates with F64 `ln` / `cos` /
`sin` (platform libm) and documents as "the GPU codec receives the same table from the host, never regenerates it".
The header comment of the v2.11 group already says the TurboQuant "descriptor fields arrive with them". So a field is
needed unless the GPU regenerates `S` (ocml vs glibc F64 transcendentals: not guaranteed equal; a 1-ulp F64 difference
flips the F32 of `S` with probability ≈ 2⁻²⁸ per element, ≈ 3.7 M elements at the Llama shape) and hard-codes the
codebooks.

- A) Append `const turbine_tq_params *tq_params` to `turbine_kv_transcode_desc` (read only for TQ4 / TQ2), with
  `turbine_tq_params { uint64_t seed; const float *codebooks[4] /*device, 2/4/8/16 entries*/; const float *tables
/*device F32 [layers][num_kv_heads][k_signs 128 | v_signs 128 | S 128×128]*/ }` — the same struct Task 12 appends to
  `turbine_attention_paged_desc` (spec Interfaces line). Rust: `KvTranscodeContext.tq: Option<…>` of device slices;
  the server uploads the tables once at startup (≈ 15 MiB Llama, ≈ 17 MiB OLMoE). Recommended: matches the codec doc,
  the plan's "codebooks through the descriptor" and Task 12; bit-exact by construction.
- B) No new field: signs from `seed` on the GPU, codebooks as kernel constants, `S` regenerated on the GPU in F64 (and
  cached per seed in the library). Not bit-exact by construction, contradicts `qjl.rs`, F64 is slow on RDNA4.
- C) As A, but `tables` holds only `S` (signs regenerated from `seed` on the GPU): 1.5 % smaller, one more place the
  sign derivation lives.

## Left

Steps 2 to 4 of the task once the question is answered: failing `hip_ops::kv_transcode_matches_cpu` tq4 / tq2 cases
(Llama, OLMoE, odd shape; decode bit-exact, encode ties counted and bounded), the kernel, lab runs, timing, mutation
check. With A or C, the header, `ffi::KvTranscodeDesc`, `KvTranscodeContext`, the shim provider's validation and the
contract §9.1 / §26 change too (files beyond this task's list; the lead assigns them).
