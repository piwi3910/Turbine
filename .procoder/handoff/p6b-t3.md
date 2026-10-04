# Handoff: p6b-t3 (Phase 6b Task 3, host path)

Branch `p6b-t3` from `df47c83` (p6b stack). WIP 86a4872 of the killed builder was reviewed and squashed into
the feature commit (its L1 per-size slabs and `MemTier::put_as` / `KvTier::put_as` kept; its L2 was completed).

## Done

- `feat(kv): lower tiers store blocks in their configured format` (1a8c3e8):
  - `KvLocation.format: &'static str` (`tier::L0_FORMAT` = `l0`; `Deserialize` dropped, nothing deserialized it).
  - `HierarchyConfig.{l1_format, l2_format, lossless_tail_blocks}` from `kv.cpu.format`, `kv.nvme.format`,
    `kv.lossless_tail_blocks` (unknown codec → `UnknownModule`).
  - Lossless tail: at `request_done` the last N committed full blocks of the sequence are flagged (`tail`
    set, entries leave with the directory entry in `forget`); the sequence's earlier blocks are unflagged
    (the flag follows the latest finished sequence holding the block). Demotion format: tier format, `l0`
    for tail blocks, never more precise than the source copy (`codec::lossier`); promotion decodes to `l0`.
  - `TransferRequest.codec: TransferCodec { from, from_bytes, to, to_bytes }`; `bytes` = the smaller side.
    Demotions in flight track their destination bytes; `make_room` counts in the tier format's block size,
    a tail block takes `demotion_units` of them.
  - L1: slabs of one slot size each (sizes from the WIP). L2: slab header v2 (codec name 16 B at 60, seed at
    76), per-codec free lists, `used_bytes` = stored bytes, an unused slab (`taken == 0`) is rewritten for
    another codec once all slab files exist (startup calibration leaves l0 slabs behind).
  - Server: `HostCodec` (layout + seed = first 8 B of the unsalted namespace key) on the I/O threads:
    `IoOp::{Write, Read, Move}` run `encode_cpu` / `decode_cpu`; the L1 copy stream refuses a
    non-identity codec (needs Task 5); `check_tier_formats` at startup: codec must support the layout,
    and a non-`l0` tier format is refused (exit 1) with tensor/pipeline shards or `static` remote mode.
  - `docs/extending/kv-format.md`: "How the tiers store a codec's blocks".
- Tests: `kv_sim per_tier_formats` (new; formats per location, tail at l0 through L1 and L2, per-copy
  accounting, promotion back at l0), `kv_orchestrator::tests::sync_l2_stores_the_tier_format` (fp8 host
  round trip equals the codec's), `tier::tests::nvme_slabs_per_codec`; L2 header test now expects v2.
  `transfer::tests` request helper: codec sized to the 4 KiB test blocks (`bytes` only paces copies).
  `remote-cargo test -p turbine-kv -p turbine-scheduler -p turbine-server`: all pass, including
  `engine::r#loop::tests::l2_round_trip_matches_cold`.
- Test-first note: `per_tier_formats` was written with the implementation (the previous builder left no
  test); its failing state is shown by the mutation check below.
- Mutation check (not committed): tail ignored in `copy_codec` → `per_tier_formats` FAILS
  ("L2 copy of block 80..96 (tail true): left tq4, right l0").

## Open

- `support_startup::kv_format_availability` still refuses any non-`l0` tier format (exit 1,
  `kv_transcode_unavailable`) on every backend, so the host path runs only in tests. Lifting it for the
  cpu backend (Sync device, L2 only) is a lead decision.
- FP8 L0 + TurboQuant tier: `HostCodec` passes no L0 scales (TQ round trips in the page's scaled domain);
  wire the real scales when Task 5/9 needs them.
- L2 fragmentation: a slab in use by one codec is not shared; a demotion that finds no slab fails the copy
  (the block stays where it was).
