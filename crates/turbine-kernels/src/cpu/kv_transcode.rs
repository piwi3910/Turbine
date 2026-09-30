//! `kv_transcode` on the cpu-reference provider (ABI v2.11, Phase 6b S-1): block by block, the
//! codec function table the caller passes in ([`crate::ops::KvCodecFns`], in practice
//! `turbine_kv::codec`'s `encode_cpu` / `decode_cpu`) runs over the pages and the coded slots.
//! The reference a GPU transcode is compared against; this crate holds no codec of its own.
use turbine_core::types::DType;

use super::{CpuReference, invalid, load};
use crate::KernelError;
use crate::ops::{
    KvTranscodeConfig, KvTranscodeContext, KvTranscodeDirection, KvTranscodeKernel,
    KvTranscodeTables,
};

impl KvTranscodeKernel for CpuReference {
    fn supports(&self, cfg: &KvTranscodeConfig) -> bool {
        cfg.direction().is_some()
            && matches!(cfg.page_dtype, DType::BF16 | DType::F8E4M3)
            && cfg.layers > 0
            && cfg.half_layer_elems() > 0
    }

    fn implementation(&self, _cfg: &KvTranscodeConfig) -> String {
        "cpu_kv_transcode_ref".into()
    }

    /// The codec table the context carries is the whole codec: device tables add nothing.
    fn execute_with_tables(
        &self,
        ctx: &mut KvTranscodeContext<'_>,
        _tables: Option<&KvTranscodeTables<'_>>,
    ) -> Result<(), KernelError> {
        self.execute(ctx)
    }

    fn execute(&self, ctx: &mut KvTranscodeContext<'_>) -> Result<(), KernelError> {
        let cfg = ctx.cfg;
        let Some(direction) = cfg.direction().filter(|_| self.supports(&cfg)) else {
            return Err(invalid(format!("kv_transcode: unsupported config {cfg}")));
        };
        let layers = cfg.layers as usize;
        let page_bytes = cfg.page_bytes();
        let slot = ctx.coded_block_bytes;
        let num_blocks = ctx.num_blocks();
        if slot == 0
            || ctx.coded.len() != num_blocks * slot
            || ctx.pages.len() != num_blocks * layers
        {
            return Err(invalid(format!(
                "kv_transcode: {} pages and {} coded bytes do not make whole blocks of {layers} \
                 layers and {slot}-byte slots",
                ctx.pages.len(),
                ctx.coded.len()
            )));
        }
        let scales = |v: &Option<turbine_tensor::TensorView<'_>>| -> Result<Vec<f32>, KernelError> {
            v.as_ref().map_or(Ok(Vec::new()), load)
        };
        let (k_scales, v_scales) = (scales(&ctx.k_scales)?, scales(&ctx.v_scales)?);
        let fail = |e: String| {
            invalid(format!(
                "kv_transcode ({}): {e}",
                cfg.codec().map_or("l0", |c| c.as_str())
            ))
        };
        for b in 0..num_blocks {
            let pages = &ctx.pages[b * layers..(b + 1) * layers];
            let coded = ctx.coded.sub(b * slot, slot);
            match direction {
                KvTranscodeDirection::Encode => {
                    let mut block = Vec::with_capacity(layers * page_bytes);
                    for p in pages {
                        block.extend(p.read_bytes()?);
                    }
                    let mut out = vec![0u8; slot];
                    ctx.codecs
                        .encode(&cfg, ctx.seed, (&k_scales, &v_scales), &block, &mut out)
                        .map_err(fail)?;
                    coded.write_bytes(&out)?;
                }
                KvTranscodeDirection::Decode => {
                    let mut block = vec![0u8; layers * page_bytes];
                    ctx.codecs
                        .decode(
                            &cfg,
                            ctx.seed,
                            (&k_scales, &v_scales),
                            &coded.read_bytes()?,
                            &mut block,
                        )
                        .map_err(fail)?;
                    for (p, bytes) in pages.iter().zip(block.chunks(page_bytes)) {
                        p.write_bytes(bytes)?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use turbine_tensor::host::HostMemory;
    use turbine_tensor::{DeviceBuffer, DeviceMemory};

    use super::*;
    use crate::ops::{KvCodecFns, KvTranscodeFormat};

    /// A "codec" that stores a block reversed, prefixed by the seed's low byte: enough to see
    /// that the provider moves the right bytes to the right place.
    struct Reverse;

    impl KvCodecFns for Reverse {
        fn encode(
            &self,
            _cfg: &KvTranscodeConfig,
            seed: u64,
            _scales: (&[f32], &[f32]),
            block: &[u8],
            slot: &mut [u8],
        ) -> Result<(), String> {
            slot[0] = seed as u8;
            for (d, s) in slot[1..].iter_mut().zip(block.iter().rev()) {
                *d = *s;
            }
            Ok(())
        }

        fn decode(
            &self,
            _cfg: &KvTranscodeConfig,
            seed: u64,
            _scales: (&[f32], &[f32]),
            slot: &[u8],
            block: &mut [u8],
        ) -> Result<(), String> {
            if slot[0] != seed as u8 {
                return Err("seed mismatch".into());
            }
            for (d, s) in block.iter_mut().rev().zip(&slot[1..]) {
                *d = *s;
            }
            Ok(())
        }
    }

    fn cfg() -> KvTranscodeConfig {
        KvTranscodeConfig {
            src_format: KvTranscodeFormat::L0,
            dst_format: KvTranscodeFormat::Fp8E4m3,
            page_dtype: DType::BF16,
            head_dim: 4,
            num_kv_heads: 1,
            block_tokens: 2,
            layers: 2,
        }
    }

    /// Two blocks of two layers encode into their slots (block `b` into slot `b`) and decode
    /// back into other pages bit for bit; a config with both sides coded, or pages of the wrong
    /// size, is refused. Breaks if the provider mixes up blocks, layers or slots.
    #[test]
    fn encodes_and_decodes_block_by_block() {
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(turbine_core::types::DeviceId(0), 1 << 16);
        let cfg = cfg();
        let page = cfg.page_bytes();
        let slot = 1 + 2 * page;
        let src: Vec<DeviceBuffer> = (0..4)
            .map(|i| {
                let mut b = DeviceBuffer::alloc(&mem, page).unwrap();
                b.copy_from_host(0, &vec![i as u8 + 1; page]).unwrap();
                b
            })
            .collect();
        let dst: Vec<DeviceBuffer> = (0..4)
            .map(|_| DeviceBuffer::alloc(&mem, page).unwrap())
            .collect();
        let coded = DeviceBuffer::alloc(&mem, 2 * slot).unwrap();
        let provider = crate::cpu_reference_provider();
        let kernel = provider.kv_transcode().expect("cpu kv_transcode");
        let pages: Vec<_> = src.iter().map(DeviceBuffer::whole).collect();
        kernel
            .execute(&mut KvTranscodeContext {
                cfg,
                pages: &pages,
                coded: coded.whole(),
                coded_block_bytes: slot,
                seed: 7,
                k_scales: None,
                v_scales: None,
                codecs: &Reverse,
            })
            .expect("encode");
        let mut first = vec![0u8; slot];
        coded.copy_to_host(0, &mut first).unwrap();
        assert_eq!(first[0], 7);
        // Block 0 is layers 0 and 1 (filled 1 and 2), reversed as one block.
        assert_eq!(first[1], 2);
        assert_eq!(first[slot - 1], 1);

        let decode = KvTranscodeConfig {
            src_format: KvTranscodeFormat::Fp8E4m3,
            dst_format: KvTranscodeFormat::L0,
            ..cfg
        };
        let pages: Vec<_> = dst.iter().map(DeviceBuffer::whole).collect();
        kernel
            .execute(&mut KvTranscodeContext {
                cfg: decode,
                pages: &pages,
                coded: coded.whole(),
                coded_block_bytes: slot,
                seed: 7,
                k_scales: None,
                v_scales: None,
                codecs: &Reverse,
            })
            .expect("decode");
        for (i, d) in dst.iter().enumerate() {
            let mut got = vec![0u8; page];
            d.copy_to_host(0, &mut got).unwrap();
            assert_eq!(got, vec![i as u8 + 1; page], "page {i}");
        }

        let both = KvTranscodeConfig {
            src_format: KvTranscodeFormat::Fp8E4m3,
            dst_format: KvTranscodeFormat::Tq4,
            ..cfg
        };
        assert!(!kernel.supports(&both));
        let err = kernel
            .execute(&mut KvTranscodeContext {
                cfg,
                pages: &pages[..3],
                coded: coded.whole(),
                coded_block_bytes: slot,
                seed: 7,
                k_scales: None,
                v_scales: None,
                codecs: &Reverse,
            })
            .expect_err("three pages for two blocks of two layers");
        assert!(err.to_string().contains("whole blocks"), "{err}");
    }
}
