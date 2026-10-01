# Handoff: p6b-t6 (Phase 6b Task 6: per-tier FP8 lab proof, and the fixes it needed)

Branch `p6b-t6` from `p6b-stack` a4b80fc, merged with `p6b-stack` 3d60c68 (decode-graph SIGSEGV fix).

## Findings that change the t5 "open" items

1. **No lookup bug.** The lookup already prefers the fastest lossy candidate (`directory::lookup`); host test
   `hierarchy::tests::lookup_prefers_the_promoted_lossy_l0_copy` passes on the unchanged lookup and reuses the
   promoted L0 copies (mutation: prefer the slowest tier, FAIL). The "recompute after prefetch" in
   `nvme_round_trip_fp8_tier` had two causes, neither in the lookup:
   - the test prefetched while L0 was still draining (87 % used, 70 % threshold): the pressure reclaim demoted the
     freshly promoted blocks again (they are the lowest-value cached blocks) before the request arrived. The test now
     waits for L0 to settle (`wait_l0_settled`, also used by the two lossless NVMe tests);
   - **an idle engine never answered a prefetch**: the engine parks on its submit channel, the prefetch travels on the
     KV command channel, so `POST /turbine/v1/kv/prefetch` on an idle server waited for the next request (the t5
     300 s read-timeout "flake", 3 of 3 once the test stopped keeping the engine busy). Fixed: `KvHandle` holds a weak
     engine sender and sends `EngineCommand::Wake` after queueing (`tiny_server::prefetch_on_an_idle_engine_answers`,
     red before: 10 s read timeout).
   Also fixed in the prefetch path: a lossy prefetch was tracked under the exact key (`used` never counted) and a
   second prefetch re-promoted blocks already in L0 under their lossy key (`blocks_resident` now counts them).
2. **Staging in the ledger.** `kv_orchestrator::transcode_staging_bytes(kv, layout)` (DEMOTION_INFLIGHT slots of the
   largest encoded block among the enabled tiers' lossy, device-servable formats; 0 for `l0` tiers, FP8 pages or no
   tier) is added to the workspace requirement in `model::prepare` (`reliability_for_workspace(.., workspace +
   staging)`, single-shard only, like `enable_device_transcode`) and to the Workspace reservation in
   `post_load_budget` (`PreparedModel.transcode_staging_bytes`). The default `workspace_bytes` of 1 GiB already covers
   the ~235 MiB for Llama FP8; a budget too small for it refuses at startup (exit 1, `workspace=` names the raised
   pool). Tests: `kv_orchestrator::tests::transcode_staging_is_sized_from_the_tier_formats`, `tiny_server::
   transcode_staging_is_in_the_workspace_pool` (mutation: staging not added, FAIL).
3. `GET /turbine/v1/kv` tiers gain `formats: {<codec>: {blocks, bytes}}` (spec interface, S-2; `rung` and `lossy`
   remain for the ladder task). `blocks_total`/`blocks_used` still count L0-format blocks (bytes / L0 block bytes), so
   an fp8 L1 shows the same 292 "blocks" as `l0`; capacity in own blocks comes from `formats`.
   Test `hierarchy::tests::document_lists_copies_per_codec` (mutation: price at L0 size, FAIL).

## Task 6

- Lab test `kv_gpu::lossy_tier_reuse` (fp8 L1 from BF16 L0; filler sessions push A to L1): it passed on its first run,
  so there was no red state; it needed no product fix. Mutation: `attach_prefix` ignoring `x-turbine-kv-lossy: deny`
  FAILS it (job `turbine-lab-test-0930224817-1c27c8c9`, "a denied request reused lossy"). Result: lossy cached 256 of
  447 prompt tokens, worst first-8 |delta logprob| 0.067, 1.00 within 0.5, deny run cached 0 and bit-equal to cold.
  `nvme_round_trip_fp8_tier` now asserts a reused prefix (cached 384, lossy 256).
- Lab jobs: kv_gpu full file PASS 10/10 `turbine-lab-test-0930221914-1bee1067`; `--tier quick` PASS
  `turbine-lab-test-0930222243-109d3440` (tiny_model included); lab-bench golden1/golden16 PASS, labbook run
  7683e112 (set `phase-6b-kv-compression`, created here). Numbers: `.procoder/perf-log.md`, section "Phase 6b".
- Config example: `scripts/lab/phase6-novanas-llama.yaml` (+ `kv_gpu::phase6b_tier_lab_config_loads`).
- `tests/eval/llama-3.2-3b-instruct/turbine-l1-fp8.json`: 162/200 at c16; compared with `turbine-bf16-c16.json` (the
  spec names `turbine-bf16.json`, measured at c1; comparisons must share the concurrency).

## Open

- The GSM8K eval does not exercise lossy reuse (short prompts, no shared full block). Option: a long-shared-prefix
  variant of the task set served twice around an L1 round trip. Needs a decision.
- fp8 L1 capacity is 1.78x with the lossless tail share seen in multi-turn (12 %), below the kv_sim AC's 1.9x (which
  has no tail blocks); the lab AC for TurboQuant says nothing about tails. Decide whether 1.9x applies to real runs.
- In the stressed multi-turn run the planner picks `recompute_cheaper` for most L1 hits of lossy blocks (155 plans),
  so lossy reuse is small (3 blocks). Penalty/rate tuning is not attempted here.
- Prefetched blocks are not protected from the pressure reclaim that follows; a prefetch into a draining L0 is wasted.
- Incident: a failed `lab-serve.sh` start in my first tier script ran a global `--stop`, deleting another builder's
  serve Job (run 0930230116-347031c7) at about 22:01. The script now stops only its own run id.
