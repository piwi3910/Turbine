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
        if layers > 0
            && (layers - 1) * layer_stride + blocks_per_layer * block_bytes > ctx.pool.len()
        {
            return Err(invalid(format!(
                "{layers} layers of {layer_stride} bytes exceed the pool of {} bytes",
                ctx.pool.len()
            )));
        }
        for &(src, dst) in ctx.pairs {
            for b in [src.0, dst.0] {
                if b as usize >= blocks_per_layer {
                    return Err(invalid(format!(
                        "block id {b} is outside the {blocks_per_layer} blocks of a layer"
                    )));
                }
            }
        }
        for layer in 0..layers {
            for &(src, dst) in ctx.pairs {
                let at = |b: u32| layer * layer_stride + b as usize * block_bytes;
                let bytes = ctx.pool.sub(at(src.0), block_bytes).read_bytes()?;
                ctx.pool.sub(at(dst.0), block_bytes).write_bytes(&bytes)?;
            }
        }
        Ok(())
    }
}
