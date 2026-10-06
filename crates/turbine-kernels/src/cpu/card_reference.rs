//! The cpu-reference provider in the numerics class a card's kernels run (user decision
//! 2026-10-06 "5p b1 attention breaks `tiny_model hip_matches_cpu`", A): paged decode attention
//! runs in two numerics classes on a card whose decode-attention order puts the b1 kernel first
//! (gfx1201), and a test that holds the card's logits to the reference within float rounding
//! needs the reference in the class the library ran.
//!
//! The kernel library (`kernels/rocm/src/paged_attention.cpp`) binds [`B1_DECODE_IMPL`] for
//! grouped-head BF16 decode and hands a batch of `kTurbineB1CkSeqs` sequences or more to CK
//! `fmha_fwd_splitkv` when CK serves the page size. The two classes differ in one rounding the
//! reference can name:
//!
//! - CK (pagedkv, splitkv): the probabilities P are rounded to BF16 before the P·V GEMM, which is
//!   what [`CpuReference`] does for every call;
//! - b1 (`paged_attention_b1.hip`): P stays in F32 through P·V (its scores and P·V accumulate in
//!   F32 like the reference's; the online-softmax split order differs, at F32 rounding).
//!
//! [`CardReference`] is [`CpuReference`] for every op except paged decode attention, which keeps
//! P in F32 exactly where [`b1_runs`] says the library runs b1, on a card that prefers it. It
//! models the class, not the kernel's split structure, so it keeps tracking b1 while the
//! kernel's internals change. The constants below mirror the library's dispatch; changing it
//! there means changing them here.
use std::sync::Arc;

use turbine_core::types::DType;

use super::CpuReference;
use super::paged::{self, PRounding};
use crate::KernelError;
use crate::cards::CardProfile;
use crate::ops::{
    ActivationKernel, AddRmsnormKernel, AttentionConfig, AttentionContext, AttentionKernel,
    AttentionKind, ElementwiseKernel, EmbeddingKernel, GemmKernel, KernelProvider, KvCopyKernel,
    KvTranscodeKernel, LogitsReduceKernel, MoeKernel, NormKernel, OpKind, PagedAttentionContext,
    ProviderId, QGemmKernel, QuantizeActKernel, RopeKernel, ShardedNormKernel,
};

/// The b1 decode kernel's implementation name in a card's decode-attention order.
pub const B1_DECODE_IMPL: &str = "turbine_hip_b1_decode";
/// `kTurbineB1MaxGroup` (the library's header): the b1 kernel's largest query-head group.
pub const B1_MAX_GROUP: u32 = 8;
/// `kTurbineB1CkSeqs` (the library's header): the decode batch from which the b1 entry runs CK
/// split-KV instead, when CK serves the page size.
pub const B1_CK_SEQS: usize = 10;
/// `kCkPagedkvPage` (paged_attention.cpp): CK serves page sizes that are multiples of this.
pub const CK_PAGE_MULTIPLE: u32 = 128;

/// Whether the library's bound b1 entry runs the b1 kernel for a paged call of `cfg` over
/// `num_seqs` sequences: decode of grouped BF16 heads (at most [`B1_MAX_GROUP`] per KV head),
/// unless the batch reaches [`B1_CK_SEQS`] and CK serves the page size.
pub fn b1_runs(cfg: &AttentionConfig, num_seqs: usize) -> bool {
    let grouped = cfg.num_kv_heads > 0
        && cfg.num_q_heads > cfg.num_kv_heads
        && cfg.num_q_heads / cfg.num_kv_heads <= B1_MAX_GROUP;
    let b1_serves = cfg.kind == AttentionKind::DecodePaged && cfg.dtype == DType::BF16 && grouped;
    let ck_takes = num_seqs >= B1_CK_SEQS
        && cfg
            .block_tokens
            .is_some_and(|b| b.is_multiple_of(CK_PAGE_MULTIPLE));
    b1_serves && !ck_takes
}

/// [`CpuReference`] with paged decode attention in the class a card's kernels run (module doc).
pub struct CardReference {
    /// The card's decode-attention order puts [`B1_DECODE_IMPL`] first.
    b1_first: bool,
}

impl CardReference {
    /// The reference for `card` (`None`: no card, so [`CpuReference`]'s classes everywhere).
    pub fn new(card: Option<&CardProfile>) -> CardReference {
        let b1_first = card
            .and_then(|c| c.preference(OpKind::AttentionDecodePaged))
            .is_some_and(|p| p.order.first() == Some(&B1_DECODE_IMPL));
        CardReference { b1_first }
    }
}

/// [`CardReference`] as a registry entry. Its id is `cpu-reference`: it is that provider, in
/// the card's decode attention class.
pub fn card_reference_provider(card: Option<&CardProfile>) -> Arc<dyn KernelProvider> {
    Arc::new(CardReference::new(card))
}

static CPU: CpuReference = CpuReference;

impl AttentionKernel for CardReference {
    fn supports(&self, cfg: &AttentionConfig) -> bool {
        AttentionKernel::supports(&CPU, cfg)
    }

    fn implementation(&self, cfg: &AttentionConfig) -> String {
        AttentionKernel::implementation(&CPU, cfg)
    }

    fn execute(&self, ctx: &mut AttentionContext<'_>) -> Result<(), KernelError> {
        AttentionKernel::execute(&CPU, ctx)
    }

    fn execute_paged(&self, ctx: &mut PagedAttentionContext<'_>) -> Result<(), KernelError> {
        let num_seqs = ctx.kv_lens.shape.first().copied().unwrap_or(0);
        if self.b1_first
            && b1_runs(&ctx.cfg, num_seqs)
            && ctx.block_formats.is_none()
            && AttentionKernel::supports(&CPU, &ctx.cfg)
        {
            paged::attention_with(ctx, PRounding::F32)
        } else {
            AttentionKernel::execute_paged(&CPU, ctx)
        }
    }
}

impl KernelProvider for CardReference {
    fn id(&self) -> ProviderId {
        CPU.id()
    }
    fn gemm(&self) -> Option<&dyn GemmKernel> {
        CPU.gemm()
    }
    fn attention(&self) -> Option<&dyn AttentionKernel> {
        Some(self)
    }
    fn norm(&self) -> Option<&dyn NormKernel> {
        CPU.norm()
    }
    fn rope(&self) -> Option<&dyn RopeKernel> {
        CPU.rope()
    }
    fn activation(&self) -> Option<&dyn ActivationKernel> {
        CPU.activation()
    }
    fn embedding(&self) -> Option<&dyn EmbeddingKernel> {
        CPU.embedding()
    }
    fn elementwise(&self) -> Option<&dyn ElementwiseKernel> {
        CPU.elementwise()
    }
    fn kv_copy(&self) -> Option<&dyn KvCopyKernel> {
        CPU.kv_copy()
    }
    fn moe(&self) -> Option<&dyn MoeKernel> {
        CPU.moe()
    }
    fn add_rmsnorm(&self) -> Option<&dyn AddRmsnormKernel> {
        CPU.add_rmsnorm()
    }
    fn logits_reduce(&self) -> Option<&dyn LogitsReduceKernel> {
        CPU.logits_reduce()
    }
    fn sharded_norm(&self) -> Option<&dyn ShardedNormKernel> {
        CPU.sharded_norm()
    }
    fn qgemm(&self) -> Option<&dyn QGemmKernel> {
        CPU.qgemm()
    }
    fn quantize_act(&self) -> Option<&dyn QuantizeActKernel> {
        CPU.quantize_act()
    }
    fn kv_transcode(&self) -> Option<&dyn KvTranscodeKernel> {
        CPU.kv_transcode()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::store;
    use crate::cpu::test_util::*;

    fn cfg(
        kind: AttentionKind,
        heads: (u32, u32),
        dtype: DType,
        block_tokens: u32,
    ) -> AttentionConfig {
        AttentionConfig {
            kind,
            num_q_heads: heads.0,
            num_kv_heads: heads.1,
            head_dim: 128,
            dtype,
            block_tokens: Some(block_tokens),
            causal: true,
        }
    }

    /// Catches: the reference drifting from paged_attention.cpp's dispatch — the b1 class at a
    /// batch CK takes (or the reverse), b1 for equal heads, groups above 8, FP8 pages or prefill.
    #[test]
    fn b1_runs_follows_the_library_dispatch() {
        use AttentionKind::{DecodePaged, PrefillPaged};
        let llama = (24, 8);
        for seqs in 1..B1_CK_SEQS {
            assert!(
                b1_runs(&cfg(DecodePaged, llama, DType::BF16, 128), seqs),
                "{seqs}"
            );
        }
        assert!(!b1_runs(
            &cfg(DecodePaged, llama, DType::BF16, 128),
            B1_CK_SEQS
        ));
        assert!(!b1_runs(&cfg(DecodePaged, llama, DType::BF16, 256), 16));
        // CK does not serve 16-token pages: b1 runs at every batch.
        assert!(b1_runs(&cfg(DecodePaged, llama, DType::BF16, 16), 64));
        // Groups 2..=8 run b1; equal heads (OLMoE) and groups above 8 do not.
        assert!(b1_runs(&cfg(DecodePaged, (4, 2), DType::BF16, 128), 1));
        assert!(b1_runs(&cfg(DecodePaged, (64, 8), DType::BF16, 128), 1));
        assert!(!b1_runs(&cfg(DecodePaged, (16, 16), DType::BF16, 128), 1));
        assert!(!b1_runs(&cfg(DecodePaged, (72, 8), DType::BF16, 128), 1));
        assert!(!b1_runs(&cfg(DecodePaged, llama, DType::F8E4M3, 128), 1));
        assert!(!b1_runs(&cfg(PrefillPaged, llama, DType::BF16, 128), 1));
    }

    /// One grouped BF16 decode step over `kv_lens` (one query per sequence, sequence `s` in
    /// blocks `2s, 2s + 1` of 128 tokens) on `provider`; returns the output rows.
    fn decode(provider: &dyn KernelProvider, kv_lens: &[usize]) -> Vec<f32> {
        let mem = HostMemory::new(DeviceId(0), 1 << 26) as Arc<dyn DeviceMemory>;
        let (hq, hkv, d, bt) = (4usize, 2usize, 128usize, 128usize);
        let seqs = kv_lens.len();
        let num_blocks = 2 * seqs;
        let dtype = DType::BF16;
        let pool = Tensor::empty(&mem, &[num_blocks, 2, bt, hkv, d], dtype).expect("pool");
        store(&pool.view(), &seeded(5, num_blocks * 2 * bt * hkv * d)).expect("fill pool");
        let q = tensor(&mem, &[seqs, hq, d], dtype, &seeded(6, seqs * hq * d));
        let kn = tensor(&mem, &[seqs, hkv, d], dtype, &seeded(7, seqs * hkv * d));
        let vn = tensor(&mem, &[seqs, hkv, d], dtype, &seeded(8, seqs * hkv * d));
        let out = Tensor::empty(&mem, &[seqs, hq, d], dtype).expect("out");
        let table: Vec<i32> = (0..num_blocks as i32).collect();
        let block_table = i32_tensor_2d(&mem, seqs, &table);
        let indptr: Vec<i32> = (0..=seqs as i32).collect();
        let q_indptr = i32_tensor(&mem, &indptr);
        let lens: Vec<i32> = kv_lens.iter().map(|&l| l as i32).collect();
        let kv_lens_t = i32_tensor(&mem, &lens);
        provider
            .attention()
            .expect("attention family")
            .execute_paged(&mut PagedAttentionContext {
                cfg: AttentionConfig {
                    kind: AttentionKind::DecodePaged,
                    num_q_heads: hq as u32,
                    num_kv_heads: hkv as u32,
                    head_dim: d as u32,
                    dtype,
                    block_tokens: Some(bt as u32),
                    causal: true,
                },
                q: q.view(),
                k_new: kn.view(),
                v_new: vn.view(),
                out: out.view(),
                kv_layer: pool.view(),
                block_table: block_table.view(),
                q_indptr: q_indptr.view(),
                kv_lens: kv_lens_t.view(),
                max_q_len: 1,
                max_kv_len: *kv_lens.iter().max().unwrap() as u32,
                max_blocks_per_seq: 2,
                scale: 1.0 / (d as f32).sqrt(),
                k_scale: 1.0,
                v_scale: 1.0,
                block_formats: None,
                classes: None,
                tq: None,
                rope: None,
            })
            .expect("decode");
        crate::cpu::load(&out.view()).expect("load out")
    }

    /// Catches: the provider not routing b1-class calls to F32 P (it would equal the CK-class
    /// reference bitwise) or routing CK-class batches there (it would differ from it).
    #[test]
    fn decode_keeps_p_in_f32_where_b1_runs() {
        let ck = CpuReference;
        let b1 = CardReference::new(Some(&crate::cards::GFX1201));
        // Without a card (or on a card whose decode order does not start with b1) the reference
        // is CpuReference's class everywhere.
        let none = CardReference::new(None);
        assert_eq!(decode(&none, &[200]), decode(&ck, &[200]));
        // One sequence: b1 on the HIP side; the classes differ, by at most a few BF16 ulps of
        // the output (P's BF16 rounding is ≤ 2^-9 relative per probability).
        let lens = [200];
        let (got, want_ck) = (decode(&b1, &lens), decode(&ck, &lens));
        assert_ne!(got, want_ck, "b1 class equals the CK class bitwise");
        assert_close(&got, &want_ck, 2e-2);
        // B1_CK_SEQS sequences: CK on the HIP side; both providers agree bitwise.
        let lens: Vec<usize> = [200, 37, 129, 256]
            .into_iter()
            .cycle()
            .take(B1_CK_SEQS)
            .collect();
        assert_eq!(decode(&b1, &lens), decode(&ck, &lens));
        // One below the edge: still b1 on the HIP side.
        let lens = &lens[..B1_CK_SEQS - 1];
        assert_ne!(decode(&b1, lens), decode(&ck, lens));
    }
}
