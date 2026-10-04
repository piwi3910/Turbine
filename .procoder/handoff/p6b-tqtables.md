# Handoff: p6b-tqtables (server upload of the TurboQuant tables)

Branch `p6b-tqtables` from `p6b-stack` 65d8c4c. **Checkpoint: done, ready to merge into `p6b-stack`.**

## What was built

- `crates/turbine-server/src/tq_device.rs`: host tables from the codec generators (`kv_tq::layer_params`, namespace
  seed), one upload per consumer in the `turbine_tq_params` layout, `reserved_bytes` (workspace pool, before the KV pool
  is sized), `install` (executor). Nothing is built or allocated without a TurboQuant format.
- Transcode: `DeviceTranscode` holds the upload and calls `execute_with_tables` for `tq4`/`tq2`; `serves` is false for a
  TurboQuant codec without tables (host codec fallback, WARN `kv_transcode_tq_tables_failed`).
- Decoder: `ModelExecutor::set_tq_device_tables` (new trait method, default ignores; `DecoderExecutor` forwards), called in
  `model::load` before the warm-up. Uploaded on the cpu backend too (the cpu provider ignores the device copy).
- Support: `TIER_FORMAT_REFUSALS` tq4/tq2 `experimental`; rows `amd/gfx1201/{Llama,Olmoe}/bf16/{tq4,tq2}/none`
  `experimental`; `kv_format_availability` refuses `kv.dtype tq*` on a GPU only when the loaded library lacks v2.11.
- Sizes: 14,909,440 B Llama (28x8; log `tq_tables bytes=14909440`), 17,039,480 B OLMoE (16x16) per upload; `kv.dtype tq4`
  and a tq tier together cannot both need the transcode upload (a tier below TurboQuant pages stores them as they are).

## Evidence

- Gate ok (884 passed) on the feature and test commits. Mutations (snapshot/restore): orchestrator passes `None` tables ->
  `tq_transcode_receives_the_codec_tables` FAILS; layer index 0 for every layer -> `uploaded_tables_equal_the_codec`
  and `host_tables_are_the_l0_cache_tables` FAIL; attention upload not reserved -> `tq_tables_are_in_the_workspace_pool`
  FAILS (35,046,780 vs 34,780,420).
- Lab: `hip_tq_transcode_with_uploaded_tables_equals_the_host_codec` (job `turbine-lab-test-1001102029-20155825`): through
  `CopyStreamBackend` + HIP, L2 bytes and promoted pages equal the host codec for tq4/tq2 at 28x8 and 16x16.
  `tq_kv_serves_on_the_device` (`kv.dtype=tq4`: 4,128,768 B blocks, 2080 blocks, tables uploaded, answers) and
  `lossy_tier_reuse_tq4` (`kv.cpu.format=tq4`, device transcode, lossy cached 256) pass (job
  `turbine-lab-test-1001102511-14a7eca6`; fp8 `lossy_tier_reuse` still green, worst 0.067).

## For Task 9 / 13

- tq4 L1 reuse worst |dlogprob| is 2.473 on the lab prompt (fp8: 0.067) with 0.86 of positions within 0.5. The transcode
  is bit-exact to the host codec, so this is the codec's loss; `lossy_tier_reuse_tq4` has no bound until Task 9 sets one.
- `kv.dtype tq4` and a tq tier share the seed (pool namespace); a tq tier under TurboQuant pages is refused as before.
