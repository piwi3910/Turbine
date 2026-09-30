# Handoff: p6a-yarn-t28a (plan Task 28a, YaRN attention factor on cos/sin, kernel ABI v2.10)

Written 2026-09-29 by the 28a builder at rotation (coordinator request). Worktree
`.claude/worktrees/agent-p6a-yarn-t28a`, branch `p6a-yarn-t28a`, off d877f55. `git status` first.

Updated 2026-09-29 ~22:05 by rotation 9 builder: merged integration tip `eba77a1` (p6a-mxfp4,
clean, no conflicts). `scripts/gate.sh` was attempted 3× (21:54, 22:01, 22:04 +04) and each time
`ssh` to novanas died mid-clippy/mid-lab-prune (`Connection closed by UNKNOWN port 65535`), never
a real test/lint failure — matches the documented novanas ssh instability under concurrent-session
load. Per rules-r5.md ("don't retry in a loop"), stopped after 3 attempts; **gate still not
clean-passed**. Added the `t28a-lab.go` go-file wait to `p6a-yarn-28a-lab.sh` (rotation-9 GPU
queue-order rule) before it takes `port18000.lock`. Did not touch lab-test.sh or the GPU (per
brief, asked the lead for a slot instead).

Updated 2026-09-30 ~00:30 by the same builder: `scripts/remote-cargo.sh build --release -p
turbine-server -p turbine-bench -p turbine-model --tests` synced src and built clean (rc=0, ~42
min cold). Copied the updated `p6a-yarn-28a-lab.sh` to
`/home/piwi/turbine-ci/remote/agent-p6a-yarn-t28a/t28a_lab.sh` and started it detached
(`setsid nohup bash t28a_lab.sh`). It waited on `t28a-lab.go` (created by the lead ~21:20 +04) and
**ran to completion, rc=0, all PASS**: teacher-forced p16 max |Δ| likely/tail cpu (0.0667,
0.3945) hip (0.0968, 0.2922), p17-long hip (0.0229, 0.4038); `llama-yarn16` golden c1/c16 17/17
prompts passing (need 15), tok/s 854.7; `llama` (BF16, unchanged) golden c1/c16 16/16 (need 14),
tok/s 871.9. Log `/home/piwi/turbine-ci/remote/agent-p6a-yarn-t28a/t28a-lab.log`, artifacts under
`.../t28a-lab/`.

The `t28a:yarn-nofold` spread fixture (queued under `fixture.lock`) also finished: `.../t28a/
yarn-nofold-spread.log` last line `t28a-spread: done rc=0 short=0 long=0`. Computed max |Δ| across
every prompt and variant in `yarn-nofold-{short,long}.json`: likely 0.1864316463470459, tail
0.8377771377563477. Set `tests/golden/llama-3.2-3b-instruct-yarn16/tolerance.json` per the rule
(max(spread, Llama BF16 0.15/0.55, batched 0.25/0.75), rounded up to 3 decimals): strict likely
0.187, strict tail 0.838, batched likely 0.25 (unchanged — spread doesn't exceed it), batched tail
0.838. Every number above (teacher-forced and golden) already passes the new, tighter-than-nothing
bounds. README updated with the final numbers. Committed as `fdfc953` (go-file wait + status) and
a follow-up tolerance/README commit (see `git log`).

**Both the GPU proof and the CPU spread are done — nothing is running detached anymore.**

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

## Running detached

Nothing. Both the GPU proof (`t28a-lab.sh`, rc=0) and the CPU spread fixture (`t28a:yarn-nofold`,
rc=0) finished 2026-09-30 ~00:xx and their results are folded into `tolerance.json` / the README
(see the update above). The go-file (`t28a-lab.go`) and the fixture.queue entry can be cleaned up
by whoever owns queue hygiene; not done here to avoid touching shared novanas state beyond scope.

## Exact next steps

1. `scripts/gate.sh` — attempted 3x by the rotation-9 builder, all 3 failed on an ssh transport
   drop to novanas (not a code failure); **still needs a clean run**. Retry when novanas ssh load
   is lower; if clippy/fmt then flags something real, fix and amend/add a commit.
2. `scripts/lab-test.sh novanas --tier quick` (ask the lead for a GPU slot; not run yet): hip_ops
   rope cases with the factor. The GPU proof above already exercised the HIP rope kernel with the
   factor end-to-end (teacher-forced + served golden), so this is a lint-style backstop, not a
   discovery step.
3. GPU proof: **done**, see the 2026-09-30 update above and the README's "GPU proof" paragraph.
   The formal `scripts/lab-bench.sh --model llama-yarn16 --golden16` and `--model llama --golden16`
   perf-log rows are still outstanding (the ad-hoc lab script's numbers are a proxy, not the
   official perf-log entry) — needs a GPU slot from the lead.
4. Spread landed and tolerance.json/README updated (see above). Still open: the plan's final
   commit `feat(kernels): YaRN attention factor on cos/sin (ABI v2.10)` — the three `fd201ab` /
   `fcdd146` / `a61c453` commits plus this rotation's `fdfc953` and the tolerance/README commit
   are all `handoff(...)`-prefixed per the ownership rule (lead owns turbine-model
   config/decoder/loader, the kernel header, ffi.rs, ops, `.procoder/`); the lead should fold or
   re-tag them into the plan's canonical commit when merging this branch.

## Open questions

- Plan names `backends/hip.rs` for the refusal; it lives in `shim.rs` (`ShimProvider::execute`) plus the
  server startup check; hip.rs only logs `rope_attn_factor`. Plan's `ffi::tests::optional_groups_v210_rope`
  name kept (test in ffi.rs using shim test helpers).
- 6b groups written as v2.10 must renumber to v2.11 on rebase (decision); untouched here.
