//! TurboQuant L0 pages (P6b S-5, `kv.dtype: tq4 | tq2`) on the CPU backend: the tables the
//! paged attention reads TurboQuant blocks with and the codec it encodes appended rows through.
//!
//! The rotation seed is the tier codecs' ([`HostCodec::of`]): the first 8 bytes of the unsalted
//! namespace key of the L0 format, so an L0 record and a lower-tier copy of it decode alike.
//! Every table is derived from `turbine-kv`'s TurboQuant codec (rotation signs, QJL projection,
//! Lloyd–Max codebooks) and its `encode_record`, so L0 records are byte for byte the records
//! `encode_cpu` writes.

use std::sync::Arc;

use turbine_core::types::{DType, KvLayout, ModelIdentity};
use turbine_kernels::{KV_FMT_TQ2, KV_FMT_TQ4, TqHeadTables, TqParams};
use turbine_kv::codec::turboquant::codebook::{TQ_DIM, codebook};
use turbine_kv::codec::turboquant::hadamard::{SignKind, rademacher};
use turbine_kv::codec::turboquant::{Tq2Codec, Tq4Codec, TqWidths, encode_record, qjl};
use turbine_model::kv_scales::{KvCache, TqKv};

use crate::kv_orchestrator::{HostCodec, kv_format};

/// The rotation seed of TurboQuant L0 pages of `layout` for the model `identity` (its
/// rope-scoped identity, as the KV orchestrator keys blocks with).
pub fn seed(identity: &ModelIdentity, layout: KvLayout) -> u64 {
    HostCodec::of(identity, &kv_format(layout)).params.seed
}

/// The TurboQuant tables of model layer `layer` (`num_kv_heads` heads) under `seed`.
pub fn layer_params(seed: u64, layer: u32, num_kv_heads: u32) -> TqParams {
    TqParams {
        heads: (0..num_kv_heads)
            .map(|h| TqHeadTables {
                k_signs: rademacher(seed, layer, h, SignKind::K, TQ_DIM),
                v_signs: rademacher(seed, layer, h, SignKind::V, TQ_DIM),
                qjl: qjl::projection(seed, layer, h, TQ_DIM),
            })
            .collect(),
        codebooks: [codebook(1), codebook(2), codebook(3), codebook(4)].map(<[f32]>::to_vec),
    }
}

/// The widths of TurboQuant block format `fmt`.
fn widths(fmt: u8) -> Option<TqWidths> {
    match fmt {
        KV_FMT_TQ4 => Some(Tq4Codec::WIDTHS),
        KV_FMT_TQ2 => Some(Tq2Codec::WIDTHS),
        _ => None,
    }
}

/// The paged append's codec ([`turbine_kernels::TqEncodeFn`]): `turbine-kv`'s record encoder.
/// The kernel only calls it for TurboQuant blocks (a record of another format is left as is).
pub fn encode(fmt: u8, k: &[f32], v: &[f32], head: &TqHeadTables, record: &mut [u8]) {
    if let Some(w) = widths(fmt) {
        encode_record(w, k, v, &head.k_signs, &head.v_signs, &head.qjl, record);
    }
}

/// The L0 KV cache of TurboQuant pages of `layout` (dtype [`DType::Tq4`] or [`DType::Tq2`]):
/// the tables of every layer and [`encode`]. `Err` names why the layout cannot hold them.
pub fn kv_cache(identity: &ModelIdentity, layout: KvLayout) -> Result<KvCache, String> {
    if layout.dtype.tq_record_bytes().is_none() {
        return Err(format!(
            "{} is not a TurboQuant KV dtype",
            layout.dtype.as_str()
        ));
    }
    if layout.head_dim as usize != TQ_DIM {
        return Err(format!(
            "TurboQuant KV pages need head_dim {TQ_DIM}, the model has {}",
            layout.head_dim
        ));
    }
    let seed = seed(identity, layout);
    let layers: Vec<TqParams> = (0..layout.num_layers)
        .map(|l| layer_params(seed, l, layout.num_kv_heads))
        .collect();
    Ok(KvCache::turboquant(
        layout.dtype,
        TqKv {
            seed,
            layers: Arc::from(layers),
            encode,
        },
    ))
}

/// The L0 dtype of a TurboQuant `kv.dtype` spelling.
pub fn dtype(name: &str) -> Option<DType> {
    match name {
        "tq4" => Some(DType::Tq4),
        "tq2" => Some(DType::Tq2),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use turbine_kv::codec::turboquant::decode_record;

    use super::*;

    fn layout(dtype: DType) -> KvLayout {
        KvLayout {
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 128,
            dtype,
            block_tokens: 16,
        }
    }

    /// The L0 tables and codec are the tier codec's under the namespace seed: a record the
    /// paged append would write decodes with `turbine-kv`'s `decode_record` under the same
    /// seed, layer and head, and each layer's tables differ. Breaks if the L0 seed, a layer's
    /// or a head's tables, or the encoder drift from the tier codec.
    #[test]
    fn tables_match_the_tier_codec() {
        let identity = ModelIdentity::from_bytes(b"tq config", b"tq index");
        for (dtype, fmt, w) in [
            (DType::Tq4, KV_FMT_TQ4, Tq4Codec::WIDTHS),
            (DType::Tq2, KV_FMT_TQ2, Tq2Codec::WIDTHS),
        ] {
            let cache = kv_cache(&identity, layout(dtype)).expect("tq cache");
            let tq = cache.tq.as_ref().expect("tables");
            assert_eq!(tq.seed, seed(&identity, layout(dtype)));
            assert_eq!(tq.layers.len(), 2);
            assert_ne!(tq.layers[0], tq.layers[1]);
            let k: Vec<f32> = (0..TQ_DIM)
                .map(|i| ((i * 7 % 13) as f32 - 6.0) / 4.0)
                .collect();
            let v: Vec<f32> = (0..TQ_DIM)
                .map(|i| ((i * 5 % 11) as f32 - 5.0) / 3.0)
                .collect();
            let p = cache.tq_paged(1).expect("layer 1");
            let mut record = vec![0xAA_u8; w.record_bytes()];
            (p.encode)(fmt, &k, &v, &p.params.heads[1], &mut record);
            let direct = decode_record(w, &record, tq.seed, 1, 1);
            let mut again = vec![0u8; w.record_bytes()];
            encode_record(
                w,
                &k,
                &v,
                &p.params.heads[1].k_signs,
                &p.params.heads[1].v_signs,
                &p.params.heads[1].qjl,
                &mut again,
            );
            assert_eq!(record, again);
            // Relative squared error of V: the codec's MSE bound (4 bits ≈ 0.01, 2 bits ≈
            // 0.12); tables of another seed, layer or head decode to about 2.
            let sq = |x: &[f32]| x.iter().map(|a| a * a).sum::<f32>();
            let diff: Vec<f32> = direct.1.iter().zip(&v).map(|(a, b)| a - b).collect();
            let rel = sq(&diff) / sq(&v);
            let bound = if dtype == DType::Tq4 { 0.05 } else { 0.3 };
            assert!(
                rel < bound,
                "{dtype:?}: V decodes far from its input ({rel})"
            );
        }
        // TurboQuant L0 formats have their own namespaces, hence seeds.
        assert_ne!(
            seed(&identity, layout(DType::Tq4)),
            seed(&identity, layout(DType::Tq2))
        );
        let err = kv_cache(
            &identity,
            KvLayout {
                head_dim: 64,
                ..layout(DType::Tq4)
            },
        )
        .unwrap_err();
        assert!(err.contains("head_dim 128"), "{err}");
    }
}
