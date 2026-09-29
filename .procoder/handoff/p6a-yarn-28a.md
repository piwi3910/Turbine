# Handoff: p6a-yarn-t28a (plan Task 28a, YaRN attention factor on cos/sin, kernel ABI v2.10)

Written 2026-09-29 by the 28a builder at rotation (coordinator request). Worktree
`.claude/worktrees/agent-p6a-yarn-t28a`, branch `p6a-yarn-t28a`, off d877f55. `git status` first.

## Done (commits)

- fd201ab `handoff(turbine_kernels.h, ffi.rs, ops)`: `turbine_rope_desc.attn_factor` (trailing float),
  `TURBINE_ABI_MINOR 10u`; `RopeContext.attn_factor: f32`; `RopeKernel::attn_factor_supported()`
  (default false; cpu-reference true; shim = `ShimLibrary::rope_attn_factor()` = minor ≥ 10);
  `V21Symbols.rope_attn_factor`; the shim refuses a factor ≠ 1 below minor 10 with
  `rope_attn_factor_unavailable` before calling the library; CPU `math::rope` and the HIP
  `rope_kernel` / `rope_token_kernel` compute `round(cos_f32 · m)` (the v2.3 build ignores the field);
  `abi_minor.cpp` comment → 10; stub `TURBINE_STUB_GFX942_V210` (V29 stub now reports 9) with hooks
  `stub_rope_calls` / `stub_rope_last_attn_factor`; `shim::tests` helpers made `pub(crate)`.
  Tests: `cpu::rope::tests::rope_attn_factor_scales_before_rounding`, `ffi::tests::optional_groups_v210_rope`,
  `abi_header_neutral header_declares_the_v210_rope_attn_factor` (other header tests now expect 10u),
  hip_ops: `rope_case` runs at 1.0 and the yarn16 factor, `prefill_shapes_match_cpu` adds two YaRN
  cases (17 tokens @ 9000, 300 tokens @ 11,900, both kernels) and asserts HIP `attn_factor_supported`.
- fcdd146 `handoff(turbine-model config/decoder)`: `ModelArchConfig::rope_attention_factor()`;
  `attention_scale()` = head_dim^-0.5 always; `DecoderDims.rope_attn_factor` passed to the rope op;
  naive/trace references scale cos/sin; `executor::rope::tests::yarn_cos_sin_times_factor_match_transformers`
  on the new fixture `crates/turbine-model/tests/fixtures/yarn_rope.json` (`scripts/golden/yarn_params.py
  --rope-out`, transformers 4.57.1 / torch 2.9.0 run on the Mac; `--out` regenerated identically, not
  committed); `decoder::tests::yarn_softmax_scale_is_head_dim_rsqrt`; server
  `check_rope_attn_factor` in `model.rs` (exit 1, `event="kernel_capability"`,
  `reason=rope_attn_factor_unavailable`) + `model::tests::rope_attn_factor_unavailable_refuses_old_rope`;
  golden.rs teacher-forced test drops UnfoldedRope (cpu is now transformers' placement); yarn16 README
  (A/B + tolerance rule), `yarn_self_spread.py` doc.
- a61c453 contract §26 v2.10 row.

Evidence: `scripts/remote-cargo.sh test -p turbine-kernels -p turbine-model -p turbine-server` rc=0,
every new test ok. **`scripts/gate.sh` NOT run yet** (commits are ungated; clippy may flag something).

## Running detached (do not duplicate)

- Transformers no-fold spread (tolerance input), CPU fixture job: `/home/piwi/turbine-ci/scratch/t28a/yarn_nofold_spread.sh`,
  log `/home/piwi/turbine-ci/scratch/t28a/yarn-nofold-spread.log` (last line `t28a-spread: done rc=<rc> short=… long=…`),
  outputs `…/t28a/yarn-nofold-short.json` (p01–p16, variants bf16-sdpa-incremental,bf16-sdpa-full,bf16-eager-incremental,fp32-sdpa-incremental)
  and `…/t28a/yarn-nofold-long.json` (p17-long, bf16-sdpa-incremental), per-part logs `yarn-nofold-{short,long}.log`.
  Queued under fixture.lock; queue entry `t28a:yarn-nofold` added to `/home/piwi/turbine-ci/fixture.queue`
  after `t14:fp8-tensor` (backup `…/t28a/fixture.queue.bak`). Judge: per prompt max |Δ| likely/tail over
  the variants; tolerance.json = max(spread, Llama BF16 bounds 0.15 / 0.55, batched 0.25 / 0.75).

## Exact next steps

1. `scripts/gate.sh` (background) → fix clippy/fmt, amend or add a fix commit.
2. `scripts/lab-test.sh novanas --tier quick` (background): hip_ops rope cases with the factor.
3. GPU proof: a ready-made detached script is at `.procoder/handoff/p6a-yarn-28a-lab.sh` (copy to
   `/home/piwi/turbine-ci/remote/agent-p6a-yarn-t28a/`, sync src first with `scripts/remote-cargo.sh build`,
   start with `setsid nohup bash t28a_lab.sh >/dev/null 2>&1 </dev/null &`). It builds kernels + release bins,
   then port18000.lock → bench.gate → bench.lock on GPU 0: teacher-forced p16,p17-long (hip + cpu), serve
   yarn16 golden c1/c16 + bench, serve llama golden c1/c16 + bench. Log `…/t28a-lab.log`, last line
   `t28a-lab: done rc=<rc>`, artifacts in `…/t28a-lab/`. Judge: p16 hip tail ≈ 0.40 (was 1.38), cpu ≈ 0.395;
   yarn16 golden PASS; llama golden unchanged PASS. Not yet run (never started).
   Then the formal `scripts/lab-bench.sh --model llama-yarn16 --golden16` and `--model llama --golden16`, perf-log row.
4. When the spread lands: tolerance.json per the rule, README spread line, commit
   `feat(kernels): YaRN attention factor on cos/sin (ABI v2.10)` per plan (or a follow-up test commit).

## Open questions

- Plan names `backends/hip.rs` for the refusal; it lives in `shim.rs` (`ShimProvider::execute`) plus the
  server startup check; hip.rs only logs `rope_attn_factor`. Plan's `ffi::tests::optional_groups_v210_rope`
  name kept (test in ffi.rs using shim test helpers).
- 6b groups written as v2.10 must renumber to v2.11 on rebase (decision); untouched here.
