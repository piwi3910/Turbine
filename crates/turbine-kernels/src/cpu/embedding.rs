//! Embedding gather on the cpu-reference provider.
use super::{CpuReference, expect_rank, expect_shape, is_float, load, load_i32, store};
use crate::KernelError;
use crate::ops::{EmbeddingConfig, EmbeddingContext, EmbeddingKernel};

impl EmbeddingKernel for CpuReference {
    fn supports(&self, cfg: &EmbeddingConfig) -> bool {
        is_float(cfg.dtype)
    }

    fn implementation(&self, _cfg: &EmbeddingConfig) -> String {
        "cpu_embedding".into()
    }

    fn execute(&self, ctx: &mut EmbeddingContext<'_>) -> Result<(), KernelError> {
        expect_rank("ids", &ctx.ids, 1)?;
        expect_rank("table", &ctx.table, 2)?;
        let tokens = ctx.ids.shape[0];
        let (vocab_rows, hidden) = (ctx.table.shape[0], ctx.table.shape[1]);
        expect_shape("out", &ctx.out, &[tokens, hidden])?;
        let ids = load_i32(&ctx.ids)?;
        let table = load(&ctx.table)?;
        let mut out = vec![0f32; tokens * hidden];
        for (dst, &id) in out.chunks_exact_mut(hidden.max(1)).zip(&ids) {
            let row = i64::from(id) - ctx.vocab_offset;
            if let Ok(row) = usize::try_from(row)
                && row < vocab_rows
            {
                dst.copy_from_slice(&table[row * hidden..(row + 1) * hidden]);
            }
        }
        store(&ctx.out, &out)
    }
}
