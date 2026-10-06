# Handoff: 5p fused batch-1 GEMV, stage A done → stage B (silu·mul fusion)

Decision: `.procoder/ask/decisions.md` "5p fused batch-1 GEMV: provider evaluation and proposal", user decision
B, staged. Branch `p5p-gemv-a` (from `p6b-stack`), one commit. Numbers: `.procoder/perf-log.md` "5p GEMV stage A".

## What landed (stage A)

- `kernels/rocm/src/gemm_wvsplitk.hip`: vLLM's `wvSplitK_hf_sml_` (skinny_gemms.cu @ b267fe4, Apache-2.0, license in
  `kernels/rocm/third_party/LICENSES/vllm-LICENSE`; header lists what was taken and the one change: an `out_t`
  template parameter + `turbine_store<out_t>` so the LM head writes F32). Turbine launcher after vLLM's gfx1x choice
  (YTILE 1 / UNRL 4 when n ≤ 4 · CUs, else YTILE 2 / UNRL 2; 16 waves × 32 lanes; one block per CU).
- Library: second `gemm` implementation `vllm_wvsplitk` (provider `vllm`, index 1; library order unchanged, so the
  v2 `turbine_gemm` default stays hipBLASLt). `supports`: hipBLASLt's supports, m = 1, BF16 a/b, BF16 or F32 c,
  trans_b, k % 8 = 0, k ≤ 32768 (x staged in 64 KiB LDS), n even, ldb % 8 = 0, n·ldb < 2^32. `run`: prefill steps
  (`ctx->gemm_prefill`), alpha ≠ 1 / beta ≠ 0, unaligned a/b or a non-wave32 device → `turbine_gemm` (hipBLASLt).
  `turbine_ctx::cu_count` is read lazily. No ABI change.
- Rust: `GemmConfig::gemv` (not rendered), set by `DecoderDims::gemv()` = no MoE && no TP && no quantized
  checkpoint. The shim probes a config without `gemv` with ≥ 2 rows (`gemm_probe(cfg, rows)`), so the GEMV never binds
  to OLMoE / TP / quantized GEMMs; `execute` passes the call's m as the tier rows. gfx1201 `gemm` row tiers:
  `≤ 1: [vllm_wvsplitk, hipblaslt]`, open: `[hipblaslt]`.
- Tests: `hip_ops gemv_matches_cpu` (5 Llama shapes vs CPU, 3-run bit determinism, prefill step bit-equal to
  hipBLASLt, selection only for one row of a `gemv` config), `gemv_timings` (graph-timed, asserts layer sum ≤ 1.02 ×
  hipBLASLt), a one-row case in `every_implementation_matches_cpu`, `implementations_enumerated` table,
  `cards gfx1201_carries_every_threshold`, `decoder gemv_only_for_dense_unquantized_one_device`, tiny_model
  `hip_v23_library_matches_cpu` (v2.3 compares against the open tier).

## Results

Kernel layer 327.9 vs 331.8 µs (graph-timed; eager timing flips the order — always time GEMVs in a graph).
Served Llama c1: 87.0 tok/s, ITL p50 11.3 ms (from 11.81, 0.957×); c16 853.2 (no change); golden c1/c16 PASS;
OLMoE quick 622.2 (no change). Lab: hip_ops gemv/every_implementation/bf16_gemm, tiny_model `hip_*` (incl.
`hip_matches_cpu`, decode graph), kv_gpu (17/17) passed; gate clean.

## Known, not from this change

- `hip_ops implementations_enumerated` fails on `p6b-stack` already: its `attention_decode_paged` table and
  `tests/lab/kernel-choices-llama.json` predate `turbine_hip_b1_decode` (served Llama decode is b1; recording says
  `ck_tile_fmha_splitkv`). The gemm part passes.
- novanas had no DNS (github.com) on 2026-10-06: a fresh checkout's first lab-bench build cannot fetch CK. Seed
  `remote/<checkout>/kbuild/_deps/composable_kernel-src` from another workspace (same CK commit, has the
  `.turbine-ck-<commit>` stamp) with `cp -a`.

## Stage B (next): silu·mul fused into the gate_up GEMV

- Own kernel family on `wvSplitK`'s persistent structure (decision B: own kernels, one fusion at a time, each measured;
  stop if not faster). GLU epilogue: pair rows `[gate; up]` (gate row r and up row r + inter in the fused
  `[2·inter, k]` weight) in one wave's YTILE, write `round_bf16(silu(round_bf16(g))) · round_bf16(u)` exactly as
  `elementwise.hip`'s `silu_mul` does from the two BF16 GEMV outputs, so the result is bit-equal to GEMV + `silu_mul`
  (test: bitwise vs `vllm_wvsplitk` then `silu_mul` on the same inputs). Saves the 2.9 µs kernel + ~1.6 µs boundary
  per layer (~0.12 ms/step).
- Needs a new op or an epilogue field (a kernel ABI minor group: header, `V21Symbols`, stub variant,
  `abi_header_neutral`), the decoder's swiglu hook (`hooks/swiglu.rs`) calling it only where the gate_up GEMM binds
  the GEMV tier (`gemv` configs, decode steps — a prefill step must keep GEMM + `silu_mul`), and the CPU reference.
- Then stage C: residual-add epilogue on o_proj / down + RMSNorm prologue on qkv / gate_up / LM head.
