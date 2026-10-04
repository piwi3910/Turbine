//! The `kv_format` conformance suite (contract §24): the contract every KV codec keeps, run over
//! a registry — never a hand-written list — by `registry_conformance::kv_codecs`, so a codec
//! registered without passing it fails `cargo test --workspace`.

use turbine_core::registry::{Registry, conformance};
use turbine_core::types::{DType, KvLayout};

use super::tests::{block, layout, nmse};
use super::{CodecParams, KvCodec};

/// Runs every property over every codec of `reg`; `Err` lists each broken one as
/// `<codec>: <property>: <detail>`.
///
/// - `fits_slot`: `bytes_per_block` is > 0 and never larger than the L0 block.
/// - `size_order`: along the registration (lossiness) order slot sizes never grow; the first
///   codec is the lossless `l0`.
/// - `round_trip`: over BF16 and FP8 L0 layouts, seeded Gaussian and outlier-heavy blocks,
///   `decode(encode(x))` is `x` bit for bit for a lossless codec and within
///   [`KvCodec::nmse_bound`] for a lossy one.
/// - `deterministic`: encoding the same block twice gives the same bytes.
/// - `sizes_checked`: a slot buffer one byte short is refused.
pub(crate) fn kv_codecs_suite(reg: &Registry<dyn KvCodec>) -> Result<(), Vec<String>> {
    let mut failures = Vec::new();
    if let Err(e) = conformance::check(reg) {
        failures.push(format!("registry: {e}"));
    }
    let params = [
        CodecParams::default(),
        CodecParams {
            seed: 0x0123_4567_89ab_cdef,
            k_scales: vec![0.05, 0.5],
            v_scales: vec![0.1, 2.0],
        },
    ];
    let layouts = [layout(DType::BF16), layout(DType::F8E4M3)];
    let mut prev: Option<(&'static str, u64)> = None;
    for (i, codec) in reg.iter().enumerate() {
        let name = codec.name();
        let mut record = |property: &str, result: Result<(), String>| {
            if let Err(detail) = result {
                failures.push(format!("{name}: {property}: {detail}"));
            }
        };
        if i == 0 {
            record(
                "size_order",
                (name == "l0")
                    .then_some(())
                    .ok_or_else(|| "the first codec must be the lossless `l0`".to_string()),
            );
        }
        let bf16 = codec.bytes_per_block(&layouts[0]);
        if let Some((p, size)) = prev {
            record(
                "size_order",
                (bf16 <= size)
                    .then_some(())
                    .ok_or_else(|| format!("{bf16} bytes after `{p}`'s {size}")),
            );
        }
        prev = Some((name, bf16));
        for l0 in &layouts {
            record("fits_slot", fits(codec, l0));
            for p in &params {
                for (seed, outliers) in [(1u64, false), (2, true)] {
                    record("round_trip", round_trip(codec, l0, p, seed, outliers));
                }
                record("deterministic", deterministic(codec, l0, p));
            }
            record("sizes_checked", sizes_checked(codec, l0));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

fn fits(c: &dyn KvCodec, l0: &KvLayout) -> Result<(), String> {
    let n = c.bytes_per_block(l0);
    if n == 0 || n > l0.block_bytes() {
        return Err(format!("{n} bytes for an L0 block of {}", l0.block_bytes()));
    }
    Ok(())
}

fn encode(c: &dyn KvCodec, l0: &KvLayout, src: &[u8], p: &CodecParams) -> Result<Vec<u8>, String> {
    let mut enc = vec![0u8; c.bytes_per_block(l0) as usize];
    c.encode_cpu(src, l0, &mut enc, p)
        .map_err(|e| e.to_string())?;
    Ok(enc)
}

fn round_trip(
    c: &dyn KvCodec,
    l0: &KvLayout,
    p: &CodecParams,
    seed: u64,
    outliers: bool,
) -> Result<(), String> {
    let src = block(l0, seed, outliers, p);
    let enc = encode(c, l0, &src, p)?;
    let mut dec = vec![0u8; l0.block_bytes() as usize];
    c.decode_cpu(&enc, l0, &mut dec, p)
        .map_err(|e| e.to_string())?;
    let what = format!("{:?} L0, outliers {outliers}", l0.dtype);
    if !c.lossy(l0) {
        if dec != src {
            return Err(format!("{what}: a lossless codec changed the block"));
        }
        return Ok(());
    }
    let e = nmse(&src, &dec, l0, p);
    if e.is_nan() || e > c.nmse_bound() {
        return Err(format!(
            "{what}: nmse {e} above the bound {}",
            c.nmse_bound()
        ));
    }
    Ok(())
}

fn deterministic(c: &dyn KvCodec, l0: &KvLayout, p: &CodecParams) -> Result<(), String> {
    let src = block(l0, 3, true, p);
    if encode(c, l0, &src, p)? != encode(c, l0, &src, p)? {
        return Err("two encodings of one block differ".into());
    }
    Ok(())
}

fn sizes_checked(c: &dyn KvCodec, l0: &KvLayout) -> Result<(), String> {
    let p = CodecParams::default();
    let src = vec![0u8; l0.block_bytes() as usize];
    let mut short = vec![0u8; c.bytes_per_block(l0) as usize - 1];
    if c.encode_cpu(&src, l0, &mut short, &p).is_ok() {
        return Err("encode into a short slot succeeded".into());
    }
    let mut dst = vec![0u8; l0.block_bytes() as usize];
    if c.decode_cpu(&short, l0, &mut dst, &p).is_ok() {
        return Err("decode of a short slot succeeded".into());
    }
    Ok(())
}
