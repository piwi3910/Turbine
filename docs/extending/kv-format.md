# Adding a KV format (codec)

A KV codec decides how one KV block is stored in a lower tier (Phase 6b S-1): it turns one L0 block — per layer `[2, block_tokens, kv_heads, head_dim]` elements of the L0 dtype (`kv.dtype`), K before V — into the bytes of one tier slot and back. The tiers, the directory, the transfers and the ladder stay in the mechanism (`KvHierarchy` in `crates/turbine-kv/src/hierarchy.rs`); a codec only encodes and decodes. Point name `kv_format`; selected by `kv.cpu.format` and `kv.nvme.format` (default `l0`, the L0 bytes unchanged), and walked rung by rung by the compression ladder up to `kv.ladder.max_format`.

## The trait

`turbine_kv::codec::KvCodec: Module` (`crates/turbine-kv/src/codec/mod.rs`):

| Method                      | Must do                                                                                                                                                                                     |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name()` (from `Module`)    | The configuration name, `^[a-z0-9_]{1,64}$`, unique in the registry.                                                                                                                        |
| `lossy(l0)`                 | True when `decode(encode(x))` may differ from `x` for an L0 of this layout (`fp8_e4m3` is lossless below an FP8 L0, lossy below a BF16 one).                                                |
| `abi_code()`                | The `TURBINE_KVFMT_*` code of the ABI v2.11 `turbine_kv_transcode` op that runs this codec on the GPU.                                                                                      |
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

Every directory location carries its format (`KvLocation.format`, `l0` in L0). A demotion into L1 or L2 writes the tier's codec (`HierarchyConfig.l1_format` / `l2_format`), except the last `kv.lossless_tail_blocks` full blocks of the latest finished sequence holding them, which keep `l0`, and never a format more precise than the source copy; a promotion decodes back to the L0 format. Each copy request names both ends (`TransferRequest.codec`, a `TransferCodec`), and the tiers size slots by the codec: L1 slabs hold slots of one size each, L2 slab files (header version 2) name their slots' codec and rotation seed. Where the kernel library serves the codec (`DeviceTranscode` in `crates/turbine-server/src/kv_orchestrator.rs`: the ABI v2.11 `kv_transcode`, `fp8_e4m3` over BF16 pages today), a demotion encodes the block into a device staging slot (`DEMOTION_INFLIGHT` slots of the largest encoded block, allocated at startup only when a tier format needs them) and only its small bytes cross the host link through the pinned copies into L1 or L2, and a promotion copies the small bytes up and decodes them into the L0 pages; any other copy (another codec, no free slot, a library without the group) takes the host path, where the I/O threads run `encode_cpu` / `decode_cpu`. L1 and L2 store the same bytes either way (`scripts/remote-cargo.sh test -p turbine-server kv_orchestrator::tests::device_transcode_matches_the_host_codec_through_l1_and_l2`). `scripts/remote-cargo.sh test -p turbine-scheduler --test kv_sim per_tier_formats` pins this.

## Conformance suite

`kv_codecs_suite` (`crates/turbine-kv/src/codec/conformance.rs`) runs every registered codec over BF16 and FP8 L0 layouts with seeded Gaussian and outlier-heavy blocks: `fits_slot`, `size_order` (`l0` first, slots never grow), `round_trip` (bit-exact for a lossless codec, within `nmse_bound` for a lossy one), `deterministic` and `sizes_checked`.

- `scripts/remote-cargo.sh test -p turbine-kv registry_conformance` — the suite over the registry.
- `scripts/remote-cargo.sh test -p turbine-kv codec::tests` — the registry order and the per-codec unit tests.
- `scripts/remote-cargo.sh test -p turbine-kv codec::tests::tier_ordering` — the tier ordering and the penalty defaults.
- `scripts/remote-cargo.sh test -p turbine-server support_startup::tests::tier_formats_and_availability` — the startup checks: registry names, ordering (exit 2 naming both keys), refusals and the v2.11 gating.
- `scripts/remote-cargo.sh test -p turbine-core config::tests::phase6_keys` — the keys' syntax, defaults and ranges.
- `scripts/remote-cargo.sh test -p turbine-model --test docs_extending` — this page stays true to the tree.

## Lab checks

TurboQuant in L0 (Phase 6b S-5, CPU backend today): `kv.dtype: tq4|tq2` makes the L0 pages TurboQuant records (`DType::Tq4` / `Tq2`, ABI codes 18 / 19; 144 / 80 bytes per token and KV head, sized by `DType::tq_record_bytes`). The paged attention reads each block by its format code (`PagedAttentionContext.block_formats`, one byte per block-table entry: `KV_FMT_BF16` 0, `KV_FMT_FP8_E4M3` 1, `KV_FMT_TQ4` 2, `KV_FMT_TQ2` 3; empty = every block in the page dtype) and TurboQuant blocks in the rotated domain with the layer's `TqParams` (`PagedAttentionContext.tq`, built by `crates/turbine-server/src/kv_tq.rs` from the rotation seed); the paged append encodes new rows through the `TqEncodeFn` the caller passes, so `turbine-kernels` holds no copy of the encoder. With `kv.ladder.l0` the L0 pool grows a page class per lossy rung (`l0_page_classes` in `crates/turbine-server/src/model.rs`) beside the base format, addressed per class through the v2.11 descriptors; a new L0 format needs a format code, a record size and a branch in `cpu::tq_attention` (and in the v2.11 HIP attention). `scripts/remote-cargo.sh test -p turbine-model --test tiny_model tq_kv_matches_reference` pins it.

Progressive gating (Phase 6b): a lower tier not stored at the L0 format (`kv.cpu.format` / `kv.nvme.format` other than `l0`, or `fp8_e4m3` below a BF16 L0) exits 1 before binding with `kv_transcode_unavailable` when the loaded kernel library lacks the ABI v2.11 KV transcode (the cpu reference provider has it); `kv.ladder.enabled: true` stays refused with the same code until the ladder's rewrites run on the device transcode; `kv.dtype: tq4|tq2` on a GPU backend exits 1 with `kv_tq_unavailable` when the loaded library lacks the same v2.11 group (the mixed-format paged attention, plan Task 12), and the ladder with `kv.ladder.l0: true` (the default) is refused with `kv_tq_unavailable` for the same reason when the loaded library lacks the same group (its rung classes are read by the v2.11 mixed-format paged attention; plan Task 18 lifted the unconditional refusal once per-class block addressing landed). `--check-config` does not load a kernel library, so it still answers `config ok` for them. The check is `support_startup::kv_format_availability` (`crates/turbine-server/src/support_startup.rs`), called before the library is loaded (`None`) and again once the model's provider is prepared (`Some(ShimLibrary::kv_transcode())`). `fp8_e4m3` (plan Task 6) and `tq4` (Task 9 and the multi-turn A/B) are `supported` as lower-tier formats, and `tq2` is `experimental` in `TIER_FORMAT_REFUSALS` (it fails the Task 9 eval) and `amd/gfx1201` Llama / OLMoE with BF16 weights and `kv.dtype: tq4` is `experimental` (the S-8 gate of Task 13 failed); `kv.dtype: tq2` is refused on every backend with exit 2 and `kv_tq2_l0_refused` (`TQ2_L0_REASON` in `turbine_core::support`, user decision 2026-10-02), while `tq2` stays a lower-tier format.

TurboQuant tables (plan Task 8): a GPU library never regenerates the rotation signs or the codebooks. `crates/turbine-server/src/tq_device.rs` builds them once on the host from the codec's generators (`kv_tq::layer_params`, under the namespace seed), uploads them in the layout of `turbine_tq_params` (per layer and KV head: K signs then V signs; 229 KB for Llama-3.2-3B, 262 KB for OLMoE-1B-7B) and hands them to the TurboQuant `kv_transcode` calls of the copy backend (`execute_with_tables`, one upload for `kv.cpu.format` / `kv.nvme.format` of `tq4|tq2` under BF16 pages) and to the decoder with `ModelExecutor::set_tq_device_tables` (per-layer slices, one upload for `kv.dtype: tq4|tq2`) before the first forward or graph capture. `tq_device::reserved_bytes` counts them in the workspace pool before the KV pool is sized (a budget that cannot hold them exits 1); no TurboQuant format configured means no build and no allocation.

A codec changes nothing until a tier names it, so landing its CPU reference needs no lab run. Its GPU transcode needs the lab test of the transcode op against the CPU codec (`scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops`, the `kv_transcode_matches_cpu` case: decode bit-exact, encode equal except at documented ties), and before its `TIER_FORMAT_REFUSALS` entry turns `supported` the quality gate of Phase 6b S-8 on GPU 0: `scripts/lab-bench.sh --model llama --golden16 -- --set kv.cpu.format=<name>` (and `--model olmoe`) with golden c1 and c16 under the batched bounds, plus `turbine-golden eval-compare` against the BF16 KV report.

## Pitfalls

- Tier ordering: a tier may not be more precise than the tier above it (`kv.nvme.format` against `kv.cpu.format`, `kv.cpu.format` against `kv.dtype`); the configuration refuses it, so place a new codec correctly in the lossiness order.
- Lineage keys: a lossy copy is filed under `lossy_key(key, format, seed)` and blocks computed over it chain from that key; a codec never decides reuse, and a request that opts out of lossy reuse must never see its blocks.
- Bit-exact decode: the GPU decode is tested equal to `decode_cpu` bit for bit, so the CPU reference defines the format — rounding, packing and padding included. Keep it deterministic (no clock, no thread-count-dependent reductions).
- `lossy(l0)` depends on the L0 dtype: a codec that equals the L0 format (FP8 below an FP8 L0) is lossless and keeps the exact lineage.
