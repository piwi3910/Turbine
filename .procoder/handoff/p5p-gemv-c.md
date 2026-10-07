# Handoff: 5p fused batch-1 GEMV, stage C (residual epilogue + RMSNorm input): DONE, measured faster

Branch `p5p-gemv-c` (from `p6b-stack` abaa8a6d), not merged, not pushed. Decision: `.procoder/ask/decisions.md`
"5p fused batch-1 GEMV: provider evaluation and proposal" (B, staged; stage C "As built" note). Numbers:
`.procoder/perf-log.md` "5p GEMV stage C".

## Commits

- `feat(kernels)`: kernel ABI v2.15 (`turbine_gemm_desc.c_residual`, `a_norm_weight`, `a_norm_eps`), `vllm_wvsplitk`
  PRO/RES template (`gemm_wvsplitk.hip`), `GemmFusion` / `GemmFused` / `GemmKernel::fuses` / `execute_fused`, shim
  probe, CPU reference, lab tests `hip_ops gemv_residual_matches_gemv_then_add`, `gemv_rmsnorm_matches_reference`,
  `gemv_norm_residual_timings`; contract §9 v2.15, `docs/extending/kernel-implementation.md`.
- `perf(model)`: decoder `fuses_norm_residual(rows)` + `FfnHook::fuses_norm_residual` (SwiGLU), `layer(.., fused)`,
  `tiny_model fused_ops_match_unfused` expectation (host decode folds on dense Llama, still bitwise).

## Results

Kernel −3.1 µs/layer. Served one session on GPU 0 vs base abaa8a6d: Llama c1 ITL 11.238 → 11.138 ms (88.3–88.4 tok/s,
0.902× of 640 GB/s), c16 856.0 vs 856.4 (unchanged), OLMoE quick 628.4 vs 626.0 (unchanged), golden c1/c16 PASS. Lab:
hip_ops `gemv_` all pass, tiny_model 41/41 GPU, `scripts/gate.sh --base abaa8a6d` ok (573 passed).

## Notes

- The norm prologue needs k ≤ LDS staging − 32 elements (64 KB LDS is full; wave partials live in its tail).
- Not built: the LM head's final-norm prologue (0.003 ms estimated) and a gate_up GLU epilogue.
- Known, not from this change: `hip_ops implementations_enumerated` fails on `p6b-stack` (stage A handoff).
