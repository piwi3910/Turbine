# Handoff: p6b-d2h (decision "6b Task 6" follow-up 4, A: why pinned D2H calibrates slow)

Branch `p6b-d2h` from `p6b-stack` dc71f14. Numbers: `.procoder/perf-log.md`, Phase 6b, "Pinned D2H". Commit 6be446b
passed `scripts/gate.sh --base dc71f14` (`gate: ok`, 772 passed).

## Root cause

`ShimContext::copy_async` (`crates/turbine-kernels/src/pinned.rs`) fenced the compute stream for every device-source
copy (event create + record on the compute stream + `hipStreamWaitEvent` on the copy stream) and recorded a completion
event for each copy. `stream_copies` (`crates/turbine-server/src/kv_orchestrator.rs`) issued one copy per layer segment
(28 × 512 KiB per Llama block). Each fence costs ~30 µs on the SDMA queue, so D2H ran at about half the H2D rate. H2D
copies have no fence, which explains the asymmetry. Calibration also timed the first copies on a fresh copy stream,
D2H first. Those copies run at ~60 % speed.

## Fix (6be446b)

- `turbine-tensor`: `CopyOp` and `CopyEngine::copy_async_batch` (default: `copy_async` per op, and on an error it waits
  for the copies already enqueued). The contract entry for `CopyEngine` is updated.
- `turbine-kernels`: `ShimContext::enqueue` resolves every end before it enqueues anything, fences once, enqueues all
  ops and records one event, giving one ticket of the batch's bytes. `copy_async` is now a batch of one.
- `turbine-server` (minimal, copy-backend part only): `stream_copies` sends one batch per block per shard, and
  `calibrate_l1` runs one untimed pass each way before timing.

## Tests

- `turbine-kernels shim::tests::a_copy_batch_fences_and_signals_once` (stub): one stream wait, one ticket, data at the
  right offsets, a bad end refuses the batch with nothing enqueued. Red with the default impl (4 tickets). Mutation
  "fence per op" fails it.
- `turbine-server kv_orchestrator::tests::block_copies_are_one_batch_and_calibration_is_warm`: calibration does 4 ×
  blocks batches (fails at 32 without the warm-up, 0 with per-segment copies), and an L0 → L1 → L0 round trip does 2
  batches. Both mutations fail it.
- Lab `turbine-kernels --test lab pinned_block_batches_d2h_keeps_up_with_h2d`: batched D2H ≥ 0.75 × H2D on the same card.
  Measured 1.02 (9.42 / 9.22). Per segment the D2H rate was 5.21.

## Measured

`kv_calibration` on GPU 0: 4.16 / 10.63 GB/s (before) → 11.57 / 11.52 GB/s (after). `lab-bench --quick` at 6be446b:
golden c1 PASS, 863.7 tok/s, ITL p50 15.4 ms (labbook 71350afa).

## Open

1. **SDMA ceiling ~12.5 GB/s each way** on a Gen5 x8 link (~26 GB/s usable). With `HSA_ENABLE_SDMA=0`, blit kernels
   reach 18.5 GB/s H2D on one large copy, but they are slower than SDMA at segment sizes (and D2H on large copies drops
   to 8 GB/s). Options: A) leave it; B) a gather/scatter copy kernel on the copy path for whole blocks (it competes with
   compute); C) a contiguous per-block KV layout (changes the pool, large).
2. **Link facts differ from the memory notes.** sysfs shows both GPUs behind Gen5 x8 root ports (`00:01.0`, `00:01.1`,
   32 GT/s x8). GPU 0 is not on x16, and GPU 1 is not on Gen4 x8. The `host_link_probe` test doc says GPU1 is Gen4 x8.
   Nothing was changed. The lead/user should confirm before they update the notes.
3. The 1.6 GB/s figure from the 6b planner runs was not reproduced. It is consistent with the same per-segment fence at
   smaller (FP8, 256 KiB) segments plus the cold first pass. Re-check `kv_calibration` on the next FP8 serve.
4. `copy_async` of the staged path (`staged_copies` / transcode staging) is one copy per call and was already single
   segment. It is unchanged.

The scratch dir on novanas was removed, and no server or Job of this branch is left running.
