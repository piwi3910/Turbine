//! The KV block fork (`copy_blocks`) on the cpu-reference provider: block by block, in pair
//! order, every layer.
use super::{CpuReference, invalid};
use crate::KernelError;
use crate::ops::{KvCopyConfig, KvCopyContext, KvCopyKernel};

fn byte_offset(name: &str, v: u64) -> Result<usize, KernelError> {
    usize::try_from(v).map_err(|_| invalid(format!("{name} = {v} does not fit usize")))
}

impl KvCopyKernel for CpuReference {
    fn supports(&self, cfg: &KvCopyConfig) -> bool {
        cfg.num_layers > 0 && cfg.block_bytes > 0
    }

    fn implementation(&self, _cfg: &KvCopyConfig) -> String {
        "cpu_copy_blocks".into()
    }

    fn execute(&self, ctx: &mut KvCopyContext<'_>) -> Result<(), KernelError> {
        let block_bytes = byte_offset("block_bytes", ctx.block_bytes)?;
        let layer_stride = byte_offset("layer_stride_bytes", ctx.layer_stride_bytes)?;
        let layers = ctx.num_layers as usize;
        if block_bytes == 0 || layer_stride < block_bytes {
            return Err(invalid(format!(
                "block_bytes {block_bytes} must be positive and at most layer_stride_bytes {layer_stride}"
            )));
        }
        let blocks_per_layer = layer_stride / block_bytes;
        let extent = (layers.max(1) - 1)
            .checked_mul(layer_stride)
            .and_then(|e| e.checked_add(blocks_per_layer * block_bytes));
        if layers > 0 && extent.is_none_or(|e| e > ctx.pool.len()) {
            return Err(invalid(format!(
                "{layers} layers of {layer_stride} bytes exceed the pool of {} bytes",
                ctx.pool.len()
            )));
        }
        // With classes, a pair copies one class's page bytes at the class offsets (the caller
        // refuses pairs whose blocks differ in class).
        let flat = |b: u32, layer: usize| layer * layer_stride + b as usize * block_bytes;
        let classed = |b: u32, fmt: u8, layer: usize| -> Result<(usize, usize), KernelError> {
            let Some(c) = ctx.classes else {
                return Ok((flat(b, layer), block_bytes));
            };
            let off = c.page_offset(b, fmt).ok_or_else(|| {
                invalid(format!(
                    "block id {b} is outside the pool's {} ids",
                    c.num_blocks
                ))
            })?;
            let len = c.page_bytes(fmt).expect("page_offset resolved the format") as usize;
            Ok((layer * layer_stride + off as usize, len))
        };
        let fmt_of = |i: usize| -> Result<u8, KernelError> {
            match ctx.classes {
                None => Ok(0),
                Some(c) => {
                    let fmt = ctx.pair_fmts.get(i).copied().unwrap_or(c.base_code());
                    let src_ok = c.page_offset(ctx.pairs[i].0.0, fmt).is_some();
                    let dst_ok = c.page_offset(ctx.pairs[i].1.0, fmt).is_some();
                    if !src_ok || !dst_ok {
                        return Err(invalid(format!(
                            "copy pair {i}: a block is outside the pool's {} ids",
                            c.num_blocks
                        )));
                    }
                    Ok(fmt)
                }
            }
        };
        for layer in 0..layers {
            for (i, &(src, dst)) in ctx.pairs.iter().enumerate() {
                let fmt = fmt_of(i)?;
                let (sat, slen) = classed(src.0, fmt, layer)?;
                let (dat, dlen) = classed(dst.0, fmt, layer)?;
                let len = slen.min(dlen);
                let bytes = ctx.pool.sub(sat, len).read_bytes()?;
                ctx.pool.sub(dat, len).write_bytes(&bytes)?;
            }
        }
        Ok(())
    }
}
