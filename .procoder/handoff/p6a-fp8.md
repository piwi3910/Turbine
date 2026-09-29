# Handoff: p6a-fp8 (plan Tasks 14–15, the `fp8_block` kernel)

Written 2026-09-29 ~08:00 by the new 6a lead after the builder was killed mid-work (no handoff of its own).
Worktree `.claude/worktrees/agent-adb021480bf19bfbe`, branch tip 2116221 (= integration's parent; every
committed FP8 change is already on `phase-6a-quantization`). Read AGENTS.md and plan Tasks 14–15 first.

## Done (on phase-6a-quantization)

- Tasks 12–13: ac11ec6 (evaluation), 7bdcef0 (`hipblaslt_fp8` W8A8 + `quantize_act`), 757c4ee (pinned FP8
  decode solutions, row-invariant FP8 prefill), c93e0a0 (staged prefill leaves single-query rows to decode).
- 1c85bfd: `fp8_block` decoded to BF16 at load (now only the logged fallback, user decision 2026-09-29).
- 0cd27b3: decisions entries for FP8 GEMM, prefill row invariance, block-scaled FP8 evaluation.
- Task 14 interim perf (quick, GPU 0): Llama FP8 tensor 965.1 tok/s = 1.13× BF16, ITL 14.0 ms. Golden c1 FAILED
  against the BF16 bounds → fixed by the calibrated fake-quant tolerance (user decision 2026-09-29).

## Uncommitted in this worktree (work in progress, NOT finished)

1. Own kernel `turbine_hip_fp8_block` (user decision "Write own kernel in 6a"):
   `kernels/rocm/src/qgemm_fp8_block.{hip,hpp}`, `qgemm_fp8_block_kernels.hpp` (new, ~500 lines);
   shared-file hunks in `impl_table.cpp` (registry entry), `turbine_hip.hpp` (`Fp8BlockScratch` in `turbine_ctx`),
   `CMakeLists.txt` (source), `cards/gfx1201.rs` (in the qgemm orders), `tools/qgemm_eval.cpp`, and
   `crates/turbine-kernels/tests/hip_qgemm.rs` (+227: `qgemm_fp8_block_matches_cpu`,
   `qgemm_fp8_block_rows_do_not_depend_on_m`).
   Design: W8A16. m ≤ 64 → fused WMMA kernel (e4m3 decoded in registers, row-invariant). Larger m / prefill →
   dequantize to BF16 (`bf16(e4m3(q)·s)`) into a per-context staging buffer (≤ 256 MiB, column-chunked beyond)
   and hipBLASLt BF16 GEMM with `TURBINE_OPTION_GEMM_PREFILL` in prefill. FP8 weights stay on the device.
2. Lab result (09-29 07:29, `scratchpad/fp8/labtest_block.txt`): rows test PASS; matches_cpu FAIL on the dequant
   path, m=128 prefill n=k=3072: |Δ| 0.0117 (hip −0.527 vs cpu −0.539). Cause: BF16 rounding of the dequantized
   weight vs the CPU reference's exact `q·s`. The builder then widened the test tolerance
   (`assert_close_fp8_block`: bf16_close or |Δ| ≤ 2e-2 + |c|/128; patch script `scratchpad/fp8/tol_patch.py`,
   already applied in the worktree) and re-queued `lab-test -- -p turbine-kernels --test hip_qgemm` on bench.lock
   (local pid 72411, output `scratchpad/fp8/labtest_block2.txt`).
   **Judge this before accepting it:** a better test compares the dequant path with a CPU reference that
   rounds the dequantized weight to BF16 first (what the golden reference does), keeping the tight bound, and
   keeps the widened bound only if that is impossible. Don't loosen tolerances to pass.
3. Golden fixtures (untracked): `tests/golden/llama-3.2-3b-instruct-fp8{,-dynamic}/` — reference.jsonl + README,
   `tolerance.json` still the BF16 values (provisional). Lab configs `scripts/lab/phase6-novanas-llama-fp8{,-tensor,-block}.yaml`.

## Running on novanas (let finish)

- `scratchpad/fp8/fixture_chain2.sh` (local pid 76333): under fixture.lock, FP8-dynamic `self_spread.py --act-quant
  fp8_token` running now (output `/home/piwi/turbine-ci/remote/agent-adb021480bf19bfbe/fixtures/<slug>/spread.json`),
  then the fp8-block reference (`--act-quant none`). A second, duplicate fp8-block reference ssh (local pid 34584)
  also waits on fixture.lock — harmless (serialized) but one of them can be killed.
- The fp8 tensor fixture: `reference.jsonl` (02:41) exists; its `spread.log` (06:29) is from a run the 07:00 crash
  may have cut — check it has a result, else re-run `self_spread.py … --act-quant fp8_tensor`.

## Exact next steps

1. Task 15 kernel (fresh builder, default model): finish the matches_cpu judgement above; `lab-test --tier quick`;
   send the shared-file hunks (impl_table.cpp, turbine_hip.hpp, CMakeLists.txt, gfx1201.rs, hip_qgemm.rs) to the
   lead as `handoff(<file>)` commits; kernel files in a `feat(rocm)` commit. The lead restores the FP8 layout in the
   loader for `fp8_block` (decode fallback per unsupported shape, `event="fp8_block_decoded"`).
2. Task 14 fixtures: copy spread results into `tolerance.json` per the OLMoE method; `quant_fixtures_valid`.
3. Task 14 proofs: `lab-bench.sh --model llama-fp8 --golden16 --c1` and `--model llama-fp8-tensor …`; vLLM via
   `scratchpad/fp8/vllm_fp8.sh`; eval + eval-compare; soak (ask the coordinator); flip rows.

## Lab traps

- ONE GPU job at a time (PSU). Everything GPU goes through bench.lock; serve sessions also through port18000/18100.
- Lab scripts rsync the working tree, uncommitted files included.
- novanas crashed 06:23 and 07:00 — fixture jobs die silently; check logs for an `rc=` line.
- Nothing is built on the Mac: `scripts/remote-cargo.sh`.
