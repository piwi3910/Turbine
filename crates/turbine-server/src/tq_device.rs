//! TurboQuant tables in device memory (P6b Task 8; user decisions "6b Task 8: TurboQuant tables
//! through the transcode descriptor" A and "6b Task 12: how per-layer TurboQuant tables reach
//! the paged-attention call" A).
//!
//! A GPU kernel library never regenerates the rotation signs or the codebooks: the server builds
//! them once on the host, from the CPU codec's own generators
//! ([`crate::kv_tq::layer_params`], under the seed of the pool's KV namespace), and uploads them
//! to the device once per rank, in the layout of `turbine_tq_params` (per layer and KV head the
//! K signs then the V signs; Llama-3.2-3B: 28 · 8 · 256 F32, 229 KB). Two consumers read that
//! upload:
//!
//! - the KV transcode of a `kv.cpu.format` / `kv.nvme.format` of `tq4` / `tq2` below BF16 pages
//!   (`CopyStreamBackend::enable_device_transcode`, `KvTranscodeKernel::execute_with_tables`),
//!   uploaded in [`upload`] with the seed and layout of the host codec;
//! - the paged attention over `kv.dtype: tq4 | tq2` pages (`DecoderExecutor::
//!   set_tq_device_tables`, [`install`]), uploaded from the model's `KvCache` tables before the
//!   first forward, so decode graphs capture the final pointers.
//!
//! Without a TurboQuant format nothing is built or allocated. The bytes are counted in the
//! workspace pool before the KV pool is sized ([`reserved_bytes`]), like the transcode's staging
//! slots, so a budget that cannot hold them refuses at startup.

use std::sync::Arc;

use turbine_core::config::KvConfig;
use turbine_core::types::{DType, KvLayout};
use turbine_kernels::KvTranscodeTables;
use turbine_kv::codec::turboquant::codebook::TQ_DIM;
use turbine_model::ModelError;
use turbine_model::executor::{ModelExecutor, TqDeviceTables};
use turbine_model::kv_scales::TqKv;
use turbine_tensor::DeviceMemory;

use crate::kv_tq;
use crate::model::{StartupError, model_error};

/// Whether `format` is a TurboQuant tier / transcode format.
pub fn is_tq(format: &str) -> bool {
    matches!(format, "tq4" | "tq2")
}

/// Codebook centroids uploaded next to the tables: 1 to 4 bits.
const CODEBOOK_ELEMS: u64 = 2 + 4 + 8 + 16;

/// The host tables of every layer of `layout` under `seed`: exactly what the CPU codec and the
/// CPU attention read (the same generators as [`crate::kv_tq::kv_cache`]).
pub fn host_tables(seed: u64, layout: &KvLayout) -> TqKv {
    let layers: Vec<_> = (0..layout.num_layers)
        .map(|l| kv_tq::layer_params(seed, l, layout.num_kv_heads))
        .collect();
    TqKv {
        seed,
        layers: Arc::from(layers),
        encode: kv_tq::encode,
    }
}

/// Uploads the tables of `layout` under `seed` to `mem` (the transcode's copy).
pub fn upload(
    mem: &Arc<dyn DeviceMemory>,
    seed: u64,
    layout: &KvLayout,
) -> Result<TqDeviceTables, ModelError> {
    TqDeviceTables::upload(mem, &host_tables(seed, layout))
}

/// The whole-model view of an upload, as `KvTranscodeKernel::execute_with_tables` takes it
/// (every layer; the attention takes one layer's slice, [`TqDeviceTables::layer`]).
pub fn transcode_view(t: &TqDeviceTables) -> KvTranscodeTables<'_> {
    KvTranscodeTables {
        codebooks: std::array::from_fn(|i| t.codebooks[i].view()),
        tables: t.tables.view(),
    }
}

/// Device bytes of one upload for pages of `layout`: `[layers][kv heads][2·d]` F32 plus
/// the four codebooks.
pub fn table_bytes(layout: &KvLayout) -> u64 {
    let per_head = KvTranscodeTables::head_elems(layout.head_dim) as u64;
    4 * (layout.num_layers as u64 * layout.num_kv_heads as u64 * per_head + CODEBOOK_ELEMS)
}

/// Whether the device transcode of the enabled tiers needs the tables: a `tq4` / `tq2` tier below
/// BF16 pages of head dimension 128 (what `turbine_hip_tq` runs; any other combination keeps
/// the host codec).
pub fn transcode_needs_tables(cfg: &KvConfig, layout: &KvLayout) -> bool {
    needs_tables(&crate::kv_orchestrator::tier_formats(cfg), layout)
}

/// [`transcode_needs_tables`] for the tier formats `formats`.
pub fn needs_tables(formats: &[&str], layout: &KvLayout) -> bool {
    layout.dtype == DType::BF16
        && layout.head_dim as usize == TQ_DIM
        && formats.iter().any(|f| is_tq(f))
}

/// Device bytes the TurboQuant tables take on this rank, 0 when no TurboQuant format is
/// configured: one upload for the transcode ([`transcode_needs_tables`]) and one for the
/// attention over TurboQuant pages (uploaded for every provider: the CPU provider reads the host
/// tables and ignores the device copy). The memory budget counts them in the workspace pool.
pub fn reserved_bytes(cfg: &KvConfig, layout: &KvLayout) -> u64 {
    let transcode = transcode_needs_tables(cfg, layout);
    // The attention's upload: any classed pool — the recent window's BF16 class (a lossy L0
    // base format with the window on, the default) or the ladder's rung classes
    // (`kv.ladder.l0`) — plus TurboQuant L0 pages. The v2.11 mixed-format paged attention
    // descriptor carries the tables whenever a block table mixes formats, TurboQuant pages
    // among them or not.
    let classed = (layout.dtype != DType::BF16 && cfg.recent_window_blocks > 0)
        || (cfg.ladder.enabled && cfg.ladder.l0);
    let attention = classed || layout.dtype.tq_record_bytes().is_some();
    (u64::from(transcode) + u64::from(attention)) * table_bytes(layout)
}

/// Gives an executor over TurboQuant pages (`tq`: the model's host tables) the device copy
/// (`DecoderExecutor::set_tq_device_tables`), before its first forward. Nothing for BF16 / FP8
/// pages.
pub fn install(
    tq: Option<&TqKv>,
    mem: &Arc<dyn DeviceMemory>,
    executor: &mut dyn ModelExecutor,
) -> Result<(), StartupError> {
    let Some(tq) = tq else {
        return Ok(());
    };
    let tables =
        TqDeviceTables::upload(mem, tq).map_err(|e| model_error("TurboQuant tables", e))?;
    let bytes = tables.tables.storage.len();
    executor
        .set_tq_device_tables(tables)
        .map_err(|e| model_error("TurboQuant tables", e))?;
    tracing::info!(
        event = "tq_tables",
        bytes,
        "TurboQuant tables are in device memory for the paged attention"
    );
    Ok(())
}

/// The bytes `turbine_tq_params.tables` must hold per layer, straight from the codec's
/// generators (not through [`host_tables`]): per KV head the K signs then the V signs. The
/// reference the upload tests compare against.
#[cfg(test)]
pub(crate) fn codec_table_bytes(seed: u64, layers: u32, heads: u32) -> Vec<Vec<u8>> {
    use turbine_kv::codec::turboquant::hadamard::{SignKind, rademacher};
    (0..layers)
        .map(|l| {
            let mut bytes = Vec::new();
            for h in 0..heads {
                for v in rademacher(seed, l, h, SignKind::K, TQ_DIM)
                    .iter()
                    .chain(&rademacher(seed, l, h, SignKind::V, TQ_DIM))
                {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
            }
            bytes
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use turbine_core::config::ModuleName;
    use turbine_core::types::DeviceId;
    use turbine_kv::codec::turboquant::codebook::codebook;
    use turbine_tensor::host::HostMemory;

    use super::*;

    fn layout(dtype: DType) -> KvLayout {
        KvLayout {
            num_layers: 3,
            num_kv_heads: 2,
            head_dim: 128,
            dtype,
            block_tokens: 16,
        }
    }

    fn host_mem() -> Arc<dyn DeviceMemory> {
        HostMemory::new(DeviceId(u32::MAX), 1 << 30)
    }

    /// The tables on the device equal the CPU codec's byte for byte: every layer's slice (the
    /// attention's view), the whole upload (the transcode's), the four codebooks and the byte
    /// count the budget reserves; another seed gives other bytes. Breaks if a layer or head
    /// index, the K / V sign order, `S`'s layout, the seed or a codebook drifts from the codec,
    /// or the budget estimate from the upload.
    #[test]
    fn uploaded_tables_equal_the_codec() {
        let mem = host_mem();
        let l = layout(DType::BF16);
        let seed = 0x5eed_u64;
        let t = upload(&mem, seed, &l).expect("upload");
        let want = codec_table_bytes(seed, l.num_layers, l.num_kv_heads);

        let read = |v: &turbine_tensor::TensorView<'_>| v.slice.read_bytes().expect("read back");
        for (layer, want) in want.iter().enumerate() {
            let view = t.layer(layer, 128).expect("layer slice");
            assert_eq!(&read(&view.tables), want, "layer {layer}");
        }
        assert!(t.layer(3, 128).is_none());
        let whole = transcode_view(&t);
        assert_eq!(read(&whole.tables), want.concat(), "whole upload");
        for bits in 1..=4_u32 {
            let bytes: Vec<u8> = codebook(bits)
                .iter()
                .flat_map(|c| c.to_le_bytes())
                .collect();
            assert_eq!(
                read(&whole.codebooks[bits as usize - 1]),
                bytes,
                "{bits}-bit codebook"
            );
        }
        let uploaded = t.tables.storage.len() as u64
            + t.codebooks
                .iter()
                .map(|c| c.storage.len() as u64)
                .sum::<u64>();
        assert_eq!(
            uploaded,
            table_bytes(&l),
            "the budget reserves what is uploaded"
        );

        let other = upload(&mem, seed + 1, &l).expect("upload");
        assert_ne!(read(&transcode_view(&other).tables), want.concat());
    }

    /// The tables of the pool's KV namespace are the ones the CPU L0 pages carry: the server's
    /// host tables for the seed of the layout are `kv_tq::kv_cache`'s. Breaks if the transcode
    /// and the attention could disagree about the rotation.
    #[test]
    fn host_tables_are_the_l0_cache_tables() {
        use turbine_core::types::ModelIdentity;
        let identity = ModelIdentity::from_bytes(b"cfg", b"index");
        let l = layout(DType::Tq4);
        let cache = kv_tq::kv_cache(&identity, l).expect("tq cache");
        let from_cache = cache.tq.expect("tables");
        let ours = host_tables(from_cache.seed, &l);
        assert_eq!(ours, from_cache);
    }

    /// The tables are reserved where a format or the mixed block table needs them, and per
    /// use: a `tq4` / `tq2` tier under BF16 pages (the transcode), TurboQuant pages (the
    /// attention), both (two uploads); a classed pool too — FP8 pages with the recent window's
    /// BF16 class, or the L0 ladder's rung classes — because the v2.11 mixed descriptor
    /// carries the tables whether or not a block can be TurboQuant; nothing for BF16 pages
    /// with `l0` tiers, the window off and the ladder off, nor for a head dimension the
    /// kernels do not run. Breaks if a configuration without any of those pays for tables, or
    /// one with them starts without the bytes counted.
    #[test]
    fn tables_are_reserved_only_where_a_turboquant_format_needs_them() {
        let name = |s: &str| ModuleName::new(s).unwrap();
        let mut kv = KvConfig::default();
        kv.cpu.enabled = true;
        kv.nvme.enabled = true;
        let bf16 = layout(DType::BF16);
        let one = table_bytes(&bf16);
        assert_eq!(reserved_bytes(&kv, &bf16), 0, "l0 tiers");
        kv.nvme.format = name("fp8_e4m3");
        assert_eq!(reserved_bytes(&kv, &bf16), 0, "fp8 tier");
        kv.nvme.format = name("tq4");
        assert_eq!(reserved_bytes(&kv, &bf16), one, "tq4 tier");
        kv.cpu.format = name("tq2");
        assert_eq!(reserved_bytes(&kv, &bf16), one, "one upload for both tiers");
        kv.nvme.enabled = false;
        kv.cpu.enabled = false;
        assert_eq!(reserved_bytes(&kv, &bf16), 0, "disabled tiers");
        kv.cpu.enabled = true;
        let narrow = KvLayout {
            head_dim: 64,
            ..bf16
        };
        assert_eq!(reserved_bytes(&kv, &narrow), 0, "head_dim 64");
        assert_eq!(
            reserved_bytes(&kv, &layout(DType::F8E4M3)),
            one,
            "fp8 pages carry the window class's tables"
        );

        let none = KvConfig::default();
        assert_eq!(reserved_bytes(&none, &layout(DType::Tq4)), one, "tq4 pages");
        assert_eq!(reserved_bytes(&none, &layout(DType::Tq2)), one, "tq2 pages");
        // FP8 pages carry the descriptor's tables for the window's BF16 class (never read:
        // no TurboQuant block can exist in such a pool).
        assert_eq!(reserved_bytes(&none, &layout(DType::F8E4M3)), one);
        // TurboQuant pages with a tier below them: the tier stores them as they are, so the
        // transcode needs nothing; only the attention's upload counts.
        assert_eq!(reserved_bytes(&kv, &layout(DType::Tq4)), one);
    }

    /// The L0 ladder's mixed-format paged attention carries the tables whenever the pool grows
    /// page classes (`kv.ladder.l0`), even over BF16 pages whose rung classes hold no
    /// TurboQuant page yet (`max_format: fp8_e4m3`): the v2.11 mixed descriptor requires them.
    /// Breaks if a ladder-on server starts without the bytes counted (the budget then refuses,
    /// or the executor's warm-up forward fails on the missing `TqPaged::device`).
    #[test]
    fn tables_are_reserved_when_the_l0_ladder_is_on() {
        let name = |s: &str| ModuleName::new(s).unwrap();
        let mut kv = KvConfig::default();
        kv.ladder.enabled = true;
        kv.ladder.l0 = true;
        kv.ladder.max_format = name("fp8_e4m3");
        let bf16 = layout(DType::BF16);
        assert_eq!(reserved_bytes(&kv, &bf16), table_bytes(&bf16), "l0 ladder");
        let mut off = kv.clone();
        off.ladder.enabled = false;
        assert_eq!(reserved_bytes(&off, &bf16), 0, "ladder off");
    }

    /// The sizes at the served models: 229 KB for Llama-3.2-3B (28 layers x 8 KV heads) and
    /// 262 KB for OLMoE-1B-7B (16 x 16), the signs and codebooks only (MSE-only K, no QJL `S`).
    #[test]
    fn table_sizes_at_the_served_models() {
        let shape = |layers, heads| KvLayout {
            num_layers: layers,
            num_kv_heads: heads,
            head_dim: 128,
            dtype: DType::BF16,
            block_tokens: 128,
        };
        assert_eq!(table_bytes(&shape(28, 8)), 4 * (224 * 256 + 30));
        assert_eq!(table_bytes(&shape(16, 16)), 4 * (256 * 256 + 30));
        assert_eq!(table_bytes(&shape(28, 8)), 229_496);
        assert_eq!(table_bytes(&shape(16, 16)), 262_264);
    }
}
