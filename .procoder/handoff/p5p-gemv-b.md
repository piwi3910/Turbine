# Handoff: 5p fused batch-1 GEMV, stage B done → stage C (residual add + RMSNorm fusion)

Decision: `.procoder/ask/decisions.md` "5p fused batch-1 GEMV: provider evaluation and proposal", user decision
B, staged (stop at any stage that does not measure faster). Branch `p5p-gemv-b` (from `p6b-stack` 97a70b34), one
commit. Numbers: `.procoder/perf-log.md` "5p GEMV stage B". Stage A handoff: `.procoder/handoff/p5p-gemv-a.md`.

## What landed (stage B)

- Kernel ABI v2.14 (no new symbol): `turbine_gemm_desc` gains `int64_t a_up_offset; int32_t a_silu_mul;` (96 → 112
  bytes). With `a_silu_mul` 1 the left operand is `silu_mul(gate, up)`, gate row r at `a + r·lda`, up at
  `+ a_up_offset` elements, bit for bit `turbine_silu_mul` then the GEMM. hipBLASLt (`gemm.cpp supported`) refuses it.
- `gemm_wvsplitk.hip`: template `GLU` + kernel argument `UpOff`; the LDS staging loop stores
  `turbine_silu_mul_chunk` (bf16.hpp rounding, `g / (1 + expf(-g))`, round, × u, round — elementwise.hip's formula)
  instead of copying `a`. `supports` (form): the plain shapes, alpha 1 / beta 0, offset % 8 = 0, no overlap, 32-bit.
  Fallback (prefill option, unaligned, non-wave32): `turbine_silu_mul` in place over the gate columns, then hipBLASLt
  (documented in the header: the gate columns may be overwritten).
- Rust: `GemmKernel::fuses_silu_mul(cfg, rows, prefill)` / `execute_silu_mul(ctx, up)` (defaults: false /
  unsupported). Shim: minor ≥ 14, not prefill, bound tier's implementation `impl_supports` the probe
  (`gemm_silu_mul_probe`: lda 2k, offset k). CPU reference: `gemv` configs, one row, decode — so host runs cover the
  fused path. Decoder: `LayerRun::linear_silu_mul` → `gemm_silu_mul` (BF16 weight, tracing off, so op-by-op traces
  keep their `act` row); `hooks/swiglu.rs` calls it and skips `silu_mul` when it returns true.
- Tests: `cpu::gemm silu_mul_input_matches_silu_mul_then_gemm`, `abi_header_neutral
header_declares_the_v214_gemm_silu_mul_input` (all minor asserts → 14u), lab `hip_ops
gemv_silu_mul_matches_unfused` (bitwise, 3 runs, prefill fallback == silu_mul + hipBLASLt), `gemv_silu_mul_timings`.
- Docs: contract §9 v2.14 paragraph, `docs/extending/kernel-implementation.md` single-row bullet.

## Results

Kernel: down 87.3 (silu_mul + GEMV) → 85.2 µs (−2.1/layer; the prologue costs ~1.5 µs over the plain 83.7, less than
the 3.6 µs kernel + boundary it removes). Served, interleaved A/B in one session on GPU 0: Llama c1 ITL 11.30 → 11.23 ms
(87.1 → 87.6 tok/s), c16 853.9 (no change), golden c1/c16 PASS, OLMoE quick 627.7 (no change).

## Notes for stage C

- Lab: `p6b-stack` is checked out in another worktree (`agent-p6b-t2`); a same-session base needs
  `git checkout --detach <sha>`. CK seeding for a fresh lab-bench workspace as in stage A (copy
  `remote/<other>/kbuild/_deps/composable_kernel-src`, stamp `.turbine-ck-cd95740…`).
- The prologue cost (~1.5 µs on down, k = 8192) is the risk for stage C's norm prologue: the RMSNorm prologue on
  qkv / gate_up (k = 3072) must also reduce the sum of squares per block before the first weight load. Measure it in
  `gemv_*_timings` before wiring the decoder. Residual-add epilogues (o_proj, down) were free in the prototype.
- The fused norm cannot be bitwise vs CK rmsnorm (sum-of-squares order): that is a numerics-class change — report the
  ulp test result and get a decision before landing it (decision text, "Numerics"). The residual epilogue can be
  bitwise (round as `add_rmsnorm` rounds).
- Same ABI pattern: a new minor (v2.15) with fields on `turbine_gemm_desc` (e.g. a residual pointer, norm weight and
  eps), read only at minor ≥ 15; the shim's probe + `fuses_*` gate; the decoder hook in
  `executor/decoder/mod.rs` (`add_rmsnorm` sites) and the CPU reference.
- Known, not from this change: `hip_ops implementations_enumerated` fails on `p6b-stack` (stage A handoff).
