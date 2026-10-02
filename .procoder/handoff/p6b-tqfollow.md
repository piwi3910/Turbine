# Handoff: p6b-tqfollow (decision "6b Task 13: TurboQuant in L0 — gate results, prefill profile", 2 B, 3 C, 4 A)

Branch `p6b-tqfollow` from `p6b-stack` 01f2c44. **Checkpoint: steps 1–4 done, ready to merge.** No lab Job or server of this
branch is left running. Numbers: `.procoder/perf-log.md`, Phase 6b, "TurboQuant prefill follow-ups"; the decisions entry "P6b:
TurboQuant transcode — provider evaluation" has the encode paragraph.

## Commits

- 8cc1a3f `feat(support)`: `kv.dtype: tq2` is refused on every backend: exit 2 naming `kv.dtype` with the reason code
  `kv_tq2_l0_refused` (`turbine_core::support::TQ2_L0_REASON`), before any port is bound, under `--check-config` too. The CPU and
  gfx1201 `tq2` L0 rows are gone, and the wildcard `tq2` row carries the new reason. `tq2` stays a lower-tier format and ladder
  rung. Tests: `server_cli kv_dtype_tq2_exits_2_before_bind`, `support::tests::resolution_and_refusal`,
  `support_startup::tests::*`. AGENTS.md support line, spec S-5 and `docs/extending/kv-format.md` are updated.
- b4e9f46 `feat(kv)`: over TurboQuant L0 pages, `KvOrchestrator` sets an attach's `lossy_tokens` to its `cached_tokens` on the first
  attach, on `attach_again` and on promotion completions. `turbine_kv_lossy_cached_tokens_total` gets the rest once per request: the
  `reattaching` set skips the second count, and `request_done` clears it. `crates/turbine-kv` is untouched. Test: `tiny_server
tq_l0_reuse_counts_lossy_cached_tokens` (bf16 control). Spec S-3 is amended.
- 497ddd1 `perf(rocm)`: `run_mixed_staged` runs the single-query `turbine_hip_mixed` pass only when
  `may_have_single_query_row(d)` holds, i.e. not one sequence, and not every sequence at `max_q_len > 1`. New lab test `hip_ops
paged_mixed_staged_skips_single_pass`: prefill rows are bit-identical with and without a decode row riding along, for both staged
  implementations. `paged_mixed_case` now draws history, q, k and v from per-tensor streams in sequence order.
- ad28f9a, f2faa53, 0a26bd4, 8e39d7d `perf(rocm)`: the TurboQuant encode (`tq_device.hpp encode_chunk`, transcode and mixed append).
  - It uses a midpoint binary search and packs bytes from codes.
  - f2faa53 tried a double-F32 norm. It was slower, and 0a26bd4 replaced it with an F64 FMA norm (bit-identical: an F32 square is
    exact in F64).
  - Loads are 16 bytes when aligned and stores are 32-bit words. `tq_pages` gains a tiny / subnormal vector.
- Docs commit: perf log, decisions paragraph, this handoff.

## Evidence

- `scripts/gate.sh` passed after each step: 905 passed, then 906 (crates all); the kernel-only commits `--base HEAD` ok.
- Host mutations (remote-cargo):
  - tq2 row back to experimental: the `server_cli` test goes RED.
  - L0 lossy flag off: `tq_l0_reuse` goes RED.
  - Both trees were restored.
- Lab, all on the k3s card, every run below green:
  - Job `1002035705-3402e5ea`: step 3, `paged_mixed*` + `kv_transcode_matches_cpu`.
  - Job `1002044309-3809aecb`: 4a.
  - Job `1002050201-37ad3203`: final.
- GPU mutations, all RED:
  - 4a, binary search skipping the upper half: `1002044522-0a82ff79`.
  - FMA norm rounded to F32: `1002053535-3456837a`.
  - Transcode load halves swapped: `1002053619-1501d855`.
  - Append load halves swapped: `1002053659-2f3d8fb2`.
- Lab `--tier quick` (`1002061609-01bb554a`): everything passed except the known flake `turbine-server
engine::tp::tests::static_tiers_flood_then_resume`, the same `[0,0,0,0]` vs `[32,…]` signature as in `p6b-t9.md`. It passed 3 of 3
  reruns on novanas (remote-cargo).
- Results:

  | Measure                                              | Before    | After         | BF16  |
  | ---------------------------------------------------- | --------- | ------------- | ----- |
  | `tq4` encode, 32 Llama blocks                        | 22,278 µs | 7,162 µs      |       |
  | `tq2` encode, 32 Llama blocks                        | 12,662 µs | 5,985 µs      |       |
  | Llama `tq4` c1 TTFT, 2,000 words (pass skip / total) | 227.2 ms  | 215.7 / 204.3 | 196.8 |
  | OLMoE `tq4` c1 TTFT, 1,500 words                     | 114.3 ms  | 101.0 ms      | 94.1  |
  | lab-bench `--quick` Llama `tq4`, c16 TTFT p50        | 281 ms    | 248 ms        |       |
  | lab-bench `--quick` Llama `tq4`, tok/s               | 834.9     | 866.6         |       |
  - Golden output was byte-identical across the three lab-bench runs, and still FAILs 0/16 (tq4 L0 stays experimental).

## Not done / next

1. In mixed c16 batches the single-query pass still launches a grid over every query row. A grid over sequences, with each
   workgroup taking the row of a `q_len` 1 sequence, would drop its prefill-row cost. This is device-side indexing in `mixed_prep`,
   `mixed_attn` and `mixed_combine`.
2. Encode: the F64 norm is about 2 of the 7.2 ms. The other lever is the 41 KB of `EncodeLds` per workgroup: BF16 `xs` would
   halve it and raise occupancy.
3. The lower-tier `tq4` demotion rate and CPU-fallback count were not remeasured: 46 of 428 demotions fell back at 0.70 ms a block,
   and it is now 0.22. That needs a multi-turn run.
4. The local experiment branch `p6b-tqfollow-exp` (TQX variant switch) is not for merge and can be deleted. Remote scratch:
   `/home/piwi/turbine-ci/remote/agent-p6b-tqfollow/ttft.sh`, which bounds its server with SIGTERM then SIGKILL after 30 s on
   every exit.
