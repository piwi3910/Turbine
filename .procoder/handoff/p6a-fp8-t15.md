# Handoff: p6a-fp8-t15 (plan Task 15, `fp8_block` kernel + FP8 layout in the loader)

Written 2026-09-29 by the Task 15 builder. Branch `p6a-fp8-t15` (from `p6a-fp8` 7174d4c), worktree
`.claude/worktrees/agent-a4baec4689b995379`. Not pushed. The proof (fixtures, bench, eval, soak) is the
next agent's.

## Done (committed)

- ca0bad2 `feat(rocm)`: own kernel `turbine_hip_fp8_block` (W8A16; fused WMMA for decode m <= 64,
  dequantize-to-BF16 + tuned hipBLASLt GEMM above / in prefill), plus the `qgemm_eval --block 1` timing.
- Handoffs, one per shared file: db9aea0 impl_table.cpp, 9cf6b63 turbine_hip.hpp, f8f3bbd CMakeLists.txt,
  82a6cd6 cards/gfx1201.rs, c3493ad hip_qgemm.rs, 6f99811 weights/mod.rs, 983c670 weights/fp8.rs
  (+ ct_fp8/hf_fp8 constructors, docs/extending/weight-format.md, the mod.rs test expectations),
  43f4c8d turbine-server model.rs, 7b6c977 tiny_model.rs.
- Test judgement: the old builder's widened bound (2e-2 + |c|/128) is gone. The dequantize path is
  judged against the CPU BF16 gemm of `bf16(e4m3(q)·s)` (what it multiplies), the fused path against
  the CPU qgemm (exact q·s), both with the Phase 1 BF16 tolerance. All 80 cases pass
  (max |Δ| 1.56e-2 = one BF16 ulp at |c| ≥ 2).
- Loader: `fp8_block` keeps e4m3 + F32 block scales; `WeightFormat::for_kernels` /
  `weights::resolve_for_providers` (called in `prepare_with` before the requirements) decodes to BF16
  only the stacks no selected provider runs (`kernel_unsupported`) or whose TP shard cuts a block
  (`shard_misaligned`), logged `event="fp8_block_decoded"`. Activations stay BF16 for block
  (W8A16, the user's option B); the checkpoint's group-128 activation scheme is not applied, so its
  golden reference must be made with `--act-quant none` (the dequant path multiplies
  bf16(q·s) exactly as `dequantize_checkpoint.py` would; the fused decode path uses exact q·s).

## Results

- `scripts/gate.sh`: ok, 774 passed (crates all).
- `scripts/lab-test.sh novanas --tier quick`: PASS (job turbine-lab-test-0929043937-1be4f7b8),
  including `qgemm_fp8_block_matches_cpu`, `qgemm_fp8_block_rows_do_not_depend_on_m`,
  `fp8_block_decode_fallback_per_stack`, `tp2_quantized_matches_tp1_on_host`.
- Weight bytes, computed from the checkpoint headers (the FP8 layout the loader uploads: e4m3
  linears, F32 block scales, BF16 embedding/norms, tied head skipped): **3,607,615,488 B** for
  `llama-3.2-3b-instruct-fp8-block` vs **6,425,499,648 B** BF16 = 0.56× (the 788 MB BF16 embedding
  keeps it above 0.5; the linear layers are exactly half).

## Pending: the served number (detached, queued on port18000 + bench.lock)

- Launched `scripts/bench-lock.sh --name port18000 scripts/bench-lock.sh scratchpad/t15_serve.sh`
  (nohup). It serves `scratchpad/t15-fp8-block.yaml` (copy of the old worktree's
  `phase6-novanas-llama-fp8-block.yaml`) with `lab-serve.sh`, greps the log, and always stops its own
  serve Job.
- Watch: `/private/tmp/claude-501/-Users-pascal-Development-Turbine/7482a1b1-2407-47bb-91f2-d826e25c21af/scratchpad/t15_weight_bytes.txt`
  (last line `t15-serve: done rc=<rc> run=<id>`); full log `…/scratchpad/t15_serve.txt`.
- Judge: `rc=0`; the `weight_format` event's `weight_bytes` should be 3,607,615,488 (± nothing — it is
  exact), `layers=bf16:1,fp8_block_128x128:196` (or similar: every decoder linear FP8, lm_head BF16),
  `activation=none`, and **no** `fp8_block_decoded` line. A `fp8_block_decoded` line or ~6.4 GB means the
  provider refused a 3B shape — a bug to fix before the proof.
- If rc≠0, read `t15_serve.txt`; make sure `scripts/lab-serve.sh novanas --stop <run>` ran.

## Left for the proof agent

Fixture (`--act-quant none`), golden c1/c16, `lab-bench --model llama-fp8-block`, eval, soak, row flip,
decisions/perf-log entries — plan Task 15 steps 4–6. The untracked golden/config files are in the old
worktree `agent-adb021480bf19bfbe`.
