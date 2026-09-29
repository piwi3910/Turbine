# Adding a KV format (codec)

A KV codec decides how one KV block is stored in a lower tier (Phase 6b S-1): it turns one L0 block — per layer `[2, block_tokens, kv_heads, head_dim]` elements of the L0 dtype (`kv.dtype`), K before V — into the bytes of one tier slot and back. The tiers, the directory, the transfers and the ladder stay in the mechanism (`KvHierarchy` in `crates/turbine-kv/src/hierarchy.rs`); a codec only encodes and decodes. Point name `kv_format`; selected by `kv.cpu.format` and `kv.nvme.format` (default `l0`, the L0 bytes unchanged), and walked rung by rung by the compression ladder up to `kv.ladder.max_format`.

## The trait

`turbine_kv::codec::KvCodec: Module` (`crates/turbine-kv/src/codec/mod.rs`):

| Method                      | Must do                                                                                                                                                                                     |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name()` (from `Module`)    | The configuration name, `^[a-z0-9_]{1,64}$`, unique in the registry.                                                                                                                        |
| `lossy(l0)`                 | True when `decode(encode(x))` may differ from `x` for an L0 of this layout (`fp8_e4m3` is lossless below an FP8 L0, lossy below a BF16 one).                                                |
| `abi_code()`                | The `TURBINE_KVFMT_*` code of the ABI v2.10 `turbine_kv_transcode` op that runs this codec on the GPU.                                                                                      |
| `nmse_bound()`              | The documented bound on ‖x − x̂‖² / ‖x‖² over a block (0 for a lossless codec); the suite holds the codec to it.                                                                             |
| `default_lossy_penalty()`   | The planner penalty of a block in this codec when `kv.lossy_penalty` does not override it (its retrieval cost is multiplied by `1 + penalty`); 0 for a lossless codec.                      |
| `supports(l0)`              | Whether the codec can store blocks of this L0 layout (default: BF16 or FP8 pages); e.g. TurboQuant needs `head_dim` 128.                                                                    |
| `bytes_per_block(l0)`       | Bytes of one slot; never more than the L0 block, and never more than the previous codec in the registry.                                                                                    |
| `encode_cpu` / `decode_cpu` | The CPU reference: deterministic, refusing a buffer of the wrong size. The GPU transcode must decode bit-exactly like it; `turbine-kv` stays GPU-free, so the codec itself has no GPU code. |

`CodecParams` carries the rotation seed (the first 8 bytes of the namespace key) and the per-layer K / V scales of an FP8 L0.

## Files to add

One file (or directory) under `crates/turbine-kv/src/codec/` (see `crates/turbine-kv/src/codec/l0.rs`, `crates/turbine-kv/src/codec/fp8_e4m3.rs` and the `crates/turbine-kv/src/codec/turboquant/` directory):

```rust
//! `half`: keeps every other token (a toy; lossy).

use turbine_core::registry::Module;
use turbine_core::types::KvLayout;

use super::{CodecError, CodecParams, KvCodec};

pub struct HalfCodec;

impl Module for HalfCodec {
    fn name(&self) -> &'static str {
        "half"
    }
}

impl KvCodec for HalfCodec {
    fn lossy(&self, _l0: &KvLayout) -> bool { true }
    fn abi_code(&self) -> u8 { 4 }
    fn nmse_bound(&self) -> f64 { 1.0 }
    fn default_lossy_penalty(&self) -> f64 { 0.8 }
    fn bytes_per_block(&self, l0: &KvLayout) -> u64 { l0.block_bytes() / 2 }
    fn encode_cpu(&self, src: &[u8], l0: &KvLayout, dst: &mut [u8], p: &CodecParams)
        -> Result<(), CodecError> { todo!() }
    fn decode_cpu(&self, src: &[u8], l0: &KvLayout, dst: &mut [u8], p: &CodecParams)
        -> Result<(), CodecError> { todo!() }
}
```

A codec that runs on the GPU also needs its `TURBINE_KVFMT_*` code and encode/decode in the kernel library's `kv_transcode` op (kernels follow `docs/extending/kernel-implementation.md`, after the reuse evaluation of the AGENTS.md rule). Run `cargo fmt --all` after adding the files.

## Registry entry

In `crates/turbine-kv/src/codec/mod.rs`: add `mod <name>;` and `pub use <name>::<Type>;`, and insert `&<Type>` into the `REGISTRY` list at its place in the lossiness order — registration order is the ladder's rung order (`rung_index`, `next_rung`), `l0` stays first, and slot sizes must not grow along it:

```rust
static REGISTRY: Registry<dyn KvCodec> = Registry::new(
    "kv_format",
    &[&L0Codec, &Fp8E4m3Codec, &Tq4Codec, &Tq2Codec],
);
```

Then pin the name in `codec::tests::registry_lists_codecs`. Nothing else: the configuration takes codec names as plain module names, the server validates `kv.cpu.format`, `kv.nvme.format`, `kv.ladder.max_format` and the `kv.lossy_penalty` entries against `registry().names()` (`crates/turbine-server/src/modules.rs`), takes the tier ordering from the registration order (`codec::tier_rung`, checked at startup by `support_startup::tests::tier_formats_and_availability`) and a codec's planner penalty from `default_lossy_penalty` unless `kv.lossy_penalty.<name>` overrides it. A new format starts refused or `experimental` in `TIER_FORMAT_REFUSALS` (`crates/turbine-core/src/support.rs`) until its quality gate passes.

## How the tiers store a codec's blocks

Every directory location carries its format (`KvLocation.format`, `l0` in L0). A demotion into L1 or L2 writes the tier's codec (`HierarchyConfig.l1_format` / `l2_format`), except the last `kv.lossless_tail_blocks` full blocks of the latest finished sequence holding them, which keep `l0`, and never a format more precise than the source copy; a promotion decodes back to the L0 format. Each copy request names both ends (`TransferRequest.codec`, a `TransferCodec`), and the tiers size slots by the codec: L1 slabs hold slots of one size each, L2 slab files (header version 2) name their slots' codec and rotation seed. On the host path (the I/O threads of `CopyStreamBackend` in `crates/turbine-server/src/kv_orchestrator.rs`) L0 ↔ L2 and L1 ↔ L2 copies run `encode_cpu` / `decode_cpu`; the L0 ↔ L1 copy stream moves `l0` blocks only until the v2.10 GPU transcode. `scripts/remote-cargo.sh test -p turbine-scheduler --test kv_sim per_tier_formats` pins this.

## Conformance suite

`kv_codecs_suite` (`crates/turbine-kv/src/codec/conformance.rs`) runs every registered codec over BF16 and FP8 L0 layouts with seeded Gaussian and outlier-heavy blocks: `fits_slot`, `size_order` (`l0` first, slots never grow), `round_trip` (bit-exact for a lossless codec, within `nmse_bound` for a lossy one), `deterministic` and `sizes_checked`.

- `scripts/remote-cargo.sh test -p turbine-kv registry_conformance` — the suite over the registry.
- `scripts/remote-cargo.sh test -p turbine-kv codec::tests` — the registry order and the per-codec unit tests.
- `scripts/remote-cargo.sh test -p turbine-kv codec::tests::tier_ordering` — the tier ordering and the penalty defaults.
- `scripts/remote-cargo.sh test -p turbine-server support_startup::tests::tier_formats_and_availability` — the startup checks: registry names, ordering (exit 2 naming both keys), refusals and the v2.10 gating.
- `scripts/remote-cargo.sh test -p turbine-core config::tests::phase6_keys` — the keys' syntax, defaults and ranges.
- `scripts/remote-cargo.sh test -p turbine-model --test docs_extending` — this page stays true to the tree.

## Lab checks

Progressive gating (Phase 6b): until plan Task 5 lands the ABI v2.10 KV transcode, a lower tier not stored at the L0 format (`kv.cpu.format` / `kv.nvme.format` other than `l0`, or `fp8_e4m3` below a BF16 L0) and `kv.ladder.enabled: true` exit 1 before binding with `kv_transcode_unavailable`; until Task 12 lands the v2.10 mixed-format paged attention, `kv.dtype: tq4|tq2` and the ladder with `kv.ladder.l0: true` (the default) exit 1 with `kv_tq_unavailable`. `--check-config` does not load a kernel library, so it still answers `config ok` for them. The check is `support_startup::kv_format_availability` (`crates/turbine-server/src/support_startup.rs`); those tasks replace it with the loaded library's v2.10 check.

A codec changes nothing until a tier names it, so landing its CPU reference needs no lab run. Its GPU transcode needs the lab test of the transcode op against the CPU codec (`scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops`, the `kv_transcode_matches_cpu` case: decode bit-exact, encode equal except at documented ties), and before its `TIER_FORMAT_REFUSALS` entry turns `supported` the quality gate of Phase 6b S-8 on GPU 0: `scripts/lab-bench.sh --model llama --golden16 -- --set kv.cpu.format=<name>` (and `--model olmoe`) with golden c1 and c16 under the batched bounds, plus `turbine-golden eval-compare` against the BF16 KV report.

## Pitfalls

- Tier ordering: a tier may not be more precise than the tier above it (`kv.nvme.format` against `kv.cpu.format`, `kv.cpu.format` against `kv.dtype`); the configuration refuses it, so place a new codec correctly in the lossiness order.
- Lineage keys: a lossy copy is filed under `lossy_key(key, format, seed)` and blocks computed over it chain from that key; a codec never decides reuse, and a request that opts out of lossy reuse must never see its blocks.
- Bit-exact decode: the GPU decode is tested equal to `decode_cpu` bit for bit, so the CPU reference defines the format — rounding, packing and padding included. Keep it deterministic (no clock, no thread-count-dependent reductions).
- `lossy(l0)` depends on the L0 dtype: a codec that equals the L0 format (FP8 below an FP8 L0) is lossless and keeps the exact lineage.
