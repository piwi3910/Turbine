# p6a-review-fixes — the 12 verified 6a review findings (lead brief r13)

Branch `p6a-review-fixes` from integration 5c5f69b. Every fix is test-first: the new tests were run
red against the old code on novanas (`scripts/remote-cargo.sh`), where they failed (or, where they
name a new function, did not compile), then green with the fix.

## Commits

| Commit                                                                                                                                                    | Findings           | Tests                                                                                                                                                                                                                                                                                                      |
| --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| db0555c fix(kernels): shim qgemm operand layout; FP8 act groups ≠ 128 refused                                                                             | C1, C2             | `turbine-kernels` `shim::tests::qgemm_operands_checked_against_scheme_layout`, `shim::tests::fp8_act_group_other_than_128_is_refused`                                                                                                                                                                      |
| 229b708 fix(rocm): staged FP8 prefill / FP8 decode kernel refuse GQA group > 8                                                                            | P1                 | lab `hip_ops paged_fp8_group16_avoids_the_decode_kernel` (#[ignore]d, GPU)                                                                                                                                                                                                                                 |
| 407ba40 fix(server): leader-stopped rank exits through the bounded join                                                                                   | C6                 | `turbine-server` `startup::tests::rank_shutdown_joins_the_worker`                                                                                                                                                                                                                                          |
| 630c5a3 fix(model): FP8 scale shapes and values checked at load; act-quant scratch sized with checked arithmetic                                          | C5, P4, P5         | `weights::fp8::tests::repack_checks_the_scale_shape`, `weights::fp8::tests::scales_must_be_finite_and_positive`, `executor::decoder::tests::act_quant_scratch_size_does_not_wrap`                                                                                                                          |
| 4aa3561 fix(model): checkpoint metadata validation (ct output/KV schemes, MXFP4 packed dtype + loader byte length, INT4 check-only dims, ct weight_g_idx) | C7, C8, C3, C4, C9 | `weights::tests::ct_packagings_refuse_output_and_kv_schemes`, `weights::mxfp4::tests::packed_data_must_be_u8`, `weights::mxfp4::tests::loader_refuses_a_wider_tensor_of_the_slot_shape`, `weights::int4::tests::gptq_repack_8x8` (extended), `weights::int4::tests::ct_pack_checks_weight_shape_and_g_idx` |

The lead ran the commits: this session's cwd was outside the repository, so the procoder hook
formatted with the wrong rustfmt edition (see "Notes").

## What changed, per finding

- C1: `shim.rs` `check_qgemm` validates every operand before a pointer reaches the library:
  dims vs config, a/c dtypes vs config, b dense `[n, data_bytes(1, k)]` of the scheme's dtype
  (F8E4M3 / U8, k even when packed), b_scales dense with exactly `scale_count(n, k)` elements
  (F32; U8 E8M0 for MXFP4), b_zeros required and sized for `Int4GroupZp`, a_scales required for
  FP8 modes with at least `scale_count(m, k)` F32 elements.
- C2: `abi_act_mode` refuses `Fp8Group { group != 128 }` in `supports` and `execute` of both
  qgemm and quantize_act.
- P1: `path_serves(CkPagedkvFp8Staged)` needs group ≤ `kPagedFp8DecodeMaxGroup`, so the router
  falls back to `turbine_hip_fp8`; `launch_paged_decode_fp8` returns `TURBINE_E_UNSUPPORTED`
  above that group (the last line of defence).
- C6: `rank_shutdown` stops the engine and runs `join_engines_and_workers` before `Clean`.
- C7/C8: `common::ct_check_kv_scheme` (moved from ct_fp8) and `common::ct_check_output_activations`
  are called by ct_fp8, ct_pack_int4 and ct_mxfp4 (`quant_scheme_unsupported`).
- C5: `Fp8Layout::repack` checks the scale tensor's shape: block `== [n/bn, k/bk]`, channel
  `[n]` or `[n, 1]`, per-tensor (weight or input scale) all-ones (`[]`, `[1]`); the BF16 decode
  fallback checks its block-scale shape too.
- P4: `scale_values` refuses an empty scale tensor and any value that is not finite and > 0
  (weight and input scales, check-only slots included). Zero is refused: the three lab FP8
  checkpoints (fp8, fp8-dynamic, fp8-block; 947k scale values) hold no zero, negative or
  non-finite scale (scanned on novanas). `largest_f32` also refuses empty / non-positive.
- P5: `act_quant_bytes` computes in u64 with checked ops and saturates to `u64::MAX` (the budget
  refuses it); `alloc_act_quant` errors when `tokens × k` overflows.
- C3: `Mxfp4Layout::check_tensor`: data under its own name (`weight_packed`, `weight_blocks`)
  must be U8; the dtype rule stays only for Quark's `weight`. The loader now refuses an
  as-stored upload whose byte length is not the slot's (shape × slot dtype), which caught a
  Quark BF16 `[n, k/2]` `weight` spilling over the fused gate/up stack (red on old code).
- C4/C9: INT4 check-only slots carry the layer's full dims as shape `[0, n, k]`; `g_idx` /
  `weight_g_idx` must have k entries and be the identity `i / group` (`gptq_act_order`);
  symmetric `qzeros` must be `[k/group, n/8]`; `weight_shape` must equal `[n, k]`. compressed-
  tensors `weight_g_idx` is an optional check slot: new `WeightFormat::optional` (default false),
  and the loader skips an absent optional check-only slot. None of the lab ct INT4 checkpoints
  has `weight_g_idx` (scanned).

## Real-checkpoint proof (lead request)

`WeightLoader::plan` is the loader's validation half, now split out of `load_part` without
changing its behaviour: tensors present (an absent optional check-only slot skipped), dtypes,
shapes, as-stored byte lengths, stacks. The new example
`crates/turbine-model/examples/check_checkpoint.rs` runs `load_model_config` (detect + parse),
`check_supported_weights`, `WeightLoader::plan` and every repacked slot's `repack_with`
(scales, zeros, check-only g_idx / qzeros / weight_shape, packed data). It uploads nothing and
uses no GPU. Run on novanas via `scripts/remote-cargo.sh run -p turbine-model --example
check_checkpoint -- /home/piwi/turbine-models/<dir>...`, CPU only, without bench.lock. All 14
lab checkpoints pass:

| Checkpoint                           | Result                                                                                     |
| ------------------------------------ | ------------------------------------------------------------------------------------------ |
| llama-3.1-8b-instruct                | ok bf16, 291 slots                                                                         |
| llama-3.1-8b-instruct-mxfp4a16       | ok ct_mxfp4, 515 slots                                                                     |
| llama-3.1-8b-instruct-mxfp4-a4       | ok quark_mxfp4, 515 slots                                                                  |
| llama-3.2-3b                         | ok bf16, 254 slots                                                                         |
| llama-3.2-3b-instruct                | ok bf16, 254 slots                                                                         |
| llama-3.2-3b-instruct-autoround-gptq | ok gptq, 842 slots, 450 repacked, 392 check-only                                           |
| llama-3.2-3b-instruct-awq            | ok awq, 646 slots, 646 repacked                                                            |
| llama-3.2-3b-instruct-fp8            | ok ct_fp8, 646 slots, 392 repacked                                                         |
| llama-3.2-3b-instruct-fp8-block      | ok ct_fp8, 450 slots, 196 repacked                                                         |
| llama-3.2-3b-instruct-fp8-dynamic    | ok ct_fp8, 450 slots, 196 repacked                                                         |
| llama-3.2-3b-instruct-gptq           | ok gptq, 842 slots, 450 repacked, 392 check-only                                           |
| llama-3.2-3b-instruct-gptq-own       | ok ct_pack_int4, 842 slots, 646 planned (196 absent optional weight_g_idx), 196 check-only |
| llama-3.2-3b-mxfp4-a4                | ok quark_mxfp4, 450 slots                                                                  |
| olmoe-1b-7b-0125-instruct            | ok bf16, 3219 slots                                                                        |

## Lead-owned files touched (brief r13 authorizes them for these fixes)

`crates/turbine-kernels/src/shim.rs`, `kernels/rocm/src/paged_attention.cpp`,
`kernels/rocm/src/paged_attention.hip`, `crates/turbine-model/src/loader.rs`,
`crates/turbine-model/src/weights/{mod,common,fp8,ct_fp8,ct_pack_int4,ct_mxfp4,int4,mxfp4}.rs`,
`crates/turbine-model/src/executor/decoder/{mod,linear}.rs`, and the example
`crates/turbine-model/examples/int4_layer_dump.rs` (follows the new check-slot shape). The
C ABI header was not changed.

## Gate and lab

- Gate: `scripts/gate.sh --base 5c5f69b` → `gate: ok crates=turbine-distributed,turbine-kernels,turbine-model,turbine-server passed=489 failed=0`. It ran after 4aa3561 and again with the `WeightLoader::plan` split and the example added; both runs ok.
- Lab: `scripts/lab-test.sh novanas --tier quick` → PASS (job turbine-lab-test-0930081659-3e5b5dfa), including `hip_ops paged_fp8_group16_avoids_the_decode_kernel` and `paged_fp8_matches_cpu`.

## Notes

- A local `target/debug` (about 500 MB of `.rmeta`, i.e. a cargo check) appeared in this
  worktree twice: at 11:59, during this builder's first commit attempt, and at 12:05, when the
  lead ran the first commits. Both times match the procoder commit hook, which probably runs a
  local cargo check. The builder ran no cargo except fmt. Both copies were deleted.
- Not changed, as the lead dismissed them: pi P2 (YaRN max_positions) and pi P3 (odd k).
