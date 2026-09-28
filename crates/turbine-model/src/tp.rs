//! Tensor-parallel execution of one model across a group of ranks (P5 S-6, contract §10).
//!
//! Each rank holds only its shard, following the `turbine_distributed::tp` rules: Q/K/V
//! column-parallel by heads (KV heads replicated when the group is larger than the KV head
//! count and a multiple of it), O row-parallel, gate/up column-parallel along the intermediate
//! dimension and down row-parallel (every MoE expert likewise, the router replicated), the
//! embedding and the LM head vocab-parallel (tied embeddings: one shard serves both), norms
//! replicated, and a Q/K norm over the full projections ([`crate::executor::decoder::QK_NORM_FULL`])
//! sliced to the rank's heads. The rank's executor is the ordinary [`DecoderExecutor`] over the
//! rank's dimensions ([`rank_config`]) with the collectives of [`TpContext`] inserted:
//!
//! - after the embedding: all-reduce (BF16) of the embedded rows (each id is embedded by the one
//!   rank whose vocabulary shard holds it; the others write zeros);
//! - after the O projection and after the FFN's down projection (dense and MoE): all-reduce
//!   (BF16) of the partial sums before the residual add;
//! - the full-projection Q/K norm: `row_sumsq` of the rank's slices, one all-reduce (F32) of the
//!   partial sums of both, then `rmsnorm_sharded`;
//! - the LM head: each rank computes its vocabulary shard's F32 logits, an all-gather collects
//!   every rank's (rank-major `[world][rows][shard]`), and device-to-device copies reorder them
//!   into the row-major `[rows, vocab]` rows the logits head reduces or reads, leaving the
//!   padding of an uneven last shard behind.
//!
//! Every rank therefore ends each forward with the same full logits rows; the leader samples.
//! A forward's collectives are bracketed by [`Collective::step_begin`] / `step_end`.
//!
//! Numerics against one device: the BF16 all-reduces round each rank's partial sum before
//! adding them (one device rounds once, after an F32 accumulation over the whole reduction
//! dimension), the sharded norm adds per-rank F32 partial sums of squares, and every GEMM runs
//! at the rank's (narrower) shape. No overlapped launches (a launch enqueues its collectives
//! before it returns). Decode graphs capture the collectives too when the server gives the rank
//! graphs (`parallel.tp_decode_graphs`, P5 Task 32): the `hostmem` backend's steps then read
//! their sequence numbers from a device counter (kernel ABI v2.8), so a replay is bitwise the
//! eager step; the host backend cannot be captured (it has no graph backend either).

use std::sync::Arc;

use turbine_core::types::KvLayout;
use turbine_distributed::collective::Collective;
use turbine_distributed::tp::{kv_head_range, vocab_shard};
use turbine_kernels::{KernelProvider, KernelRegistry, OpConfig, OpRequirement};
use turbine_tensor::{DeviceMemory, StreamRef};

pub use turbine_distributed::tp::ShardSpec;

use crate::ModelError;
use crate::config::{ModelArchConfig, unsupported};
use crate::executor::{
    DecoderExecutor, DecoderSpec, ExecutorLimits, ExecutorOptions, ModelExecutor,
};
use crate::loader::{LM_HEAD, LoadedWeights, SlotSource, WeightSlot};

/// The embedding's checkpoint name (vocabulary-sharded with the LM head).
const EMBED: &str = "model.embed_tokens.weight";

/// One rank of a tensor-parallel group, as its executor sees it (contract §10): its position,
/// the group's communicator and the stream the collectives are ordered on (the rank's compute
/// stream).
#[derive(Clone)]
pub struct TpContext {
    pub rank: u32,
    pub world: u32,
    pub collective: Arc<dyn Collective>,
    pub stream: StreamRef,
}

impl TpContext {
    pub fn shard(&self) -> ShardSpec {
        ShardSpec {
            rank: self.rank,
            world: self.world,
        }
    }
}

impl std::fmt::Debug for TpContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TpContext")
            .field("rank", &self.rank)
            .field("world", &self.world)
            .field("backend", &self.collective.backend())
            .field("stream", &self.stream)
            .finish()
    }
}

fn refuse(cfg: &ModelArchConfig, s: ShardSpec, why: String) -> ModelError {
    unsupported(
        "parallel.tensor_parallel_size",
        format!("{} ({} {why})", s.world, cfg.family.0.name()),
        "a size dividing the attention heads, the KV heads (or a multiple of them), the \
         intermediate size and every expert's, for a family with tensor-parallel hooks (llama, \
         olmoe)",
    )
}

/// The family's decoder hooks for tensor-parallel ranks, or why it has none.
fn tp_spec(cfg: &ModelArchConfig, s: ShardSpec) -> Result<DecoderSpec, ModelError> {
    cfg.family
        .0
        .tp_decoder_spec()
        .ok_or_else(|| refuse(cfg, s, "has no tensor-parallel hooks".into()))
}

/// Refuses a group `s` cannot split `cfg` for (reason code in the message): a family without
/// tensor-parallel hooks, or heads, KV heads, intermediate or expert widths the rules cannot
/// split evenly. `world == 1` always passes.
pub fn check(cfg: &ModelArchConfig, s: ShardSpec) -> Result<(), ModelError> {
    if s.world == 0 || s.rank >= s.world {
        return Err(refuse(cfg, s, format!("has no rank {}", s.rank)));
    }
    if s.world == 1 {
        return Ok(());
    }
    tp_spec(cfg, s)?;
    let (heads, kv) = (cfg.num_attention_heads, cfg.num_kv_heads);
    if !heads.is_multiple_of(s.world) {
        return Err(refuse(cfg, s, format!("{heads} attention heads")));
    }
    let kv_ok = if s.world <= kv {
        kv.is_multiple_of(s.world)
    } else {
        kv > 0 && s.world.is_multiple_of(kv)
    };
    if !kv_ok {
        return Err(refuse(cfg, s, format!("{kv} KV heads")));
    }
    match cfg.moe {
        Some(m) if !m.expert_intermediate.is_multiple_of(s.world) => Err(refuse(
            cfg,
            s,
            format!("expert intermediate size {}", m.expert_intermediate),
        )),
        None if !cfg.intermediate.is_multiple_of(s.world) => Err(refuse(
            cfg,
            s,
            format!("intermediate size {}", cfg.intermediate),
        )),
        _ => Ok(()),
    }
}

/// `cfg` as rank `s` runs it: its attention heads, its KV heads (one when they are replicated),
/// its intermediate columns and every expert's; the vocabulary, hidden size and everything else
/// stay the model's (the vocabulary shard is the executor's, [`vocab_shard`]).
pub fn rank_config(cfg: &ModelArchConfig, s: ShardSpec) -> Result<ModelArchConfig, ModelError> {
    check(cfg, s)?;
    let mut rank = cfg.clone();
    rank.num_attention_heads = cfg.num_attention_heads / s.world;
    rank.num_kv_heads = kv_head_range(cfg.num_kv_heads, s).len() as u32;
    if cfg.intermediate.is_multiple_of(s.world) {
        rank.intermediate = cfg.intermediate / s.world;
    }
    if let Some(m) = rank.moe.as_mut() {
        m.expert_intermediate /= s.world;
    }
    Ok(rank)
}

/// The KV layout of rank `s`'s pool: its KV heads only. Every rank's pool has the leader's block
/// count and is indexed by the leader's block ids.
pub fn kv_layout(
    cfg: &ModelArchConfig,
    s: ShardSpec,
    block_tokens: u32,
) -> Result<KvLayout, ModelError> {
    Ok(rank_config(cfg, s)?.kv_layout(block_tokens))
}

/// Rank `s`'s weight slots: the family's slots of [`rank_config`] (so the fused Q/K/V and
/// gate/up stacks and the expert stacks are laid out at the rank's widths), each sharded tensor
/// reading only its part of the checkpoint tensor ([`SlotSource`]): K/V rows (and a full K
/// norm's elements) at the rank's KV heads, every other split at `rank × part`; the embedding
/// and an untied LM head as `[padded_rows, hidden]` vocabulary shards (padding rows zero).
/// Replicated tensors (norms, the router) are whole. Load them with [`crate::WeightLoader`].
pub fn weight_slots(cfg: &ModelArchConfig, s: ShardSpec) -> Result<Vec<WeightSlot>, ModelError> {
    let rank = rank_config(cfg, s)?;
    let family = cfg.family.0;
    let full = family.weight_slots(cfg);
    let mut slots = family.weight_slots(&rank);
    let mismatch = |what: String| {
        ModelError::Kernel(turbine_kernels::KernelError::InvalidArgument {
            message: format!("tensor-parallel slots of {}: {what}", family.name()),
        })
    };
    if slots.len() != full.len() {
        return Err(mismatch(format!(
            "{} slots at rank width, {} unsharded",
            slots.len(),
            full.len()
        )));
    }
    let head_dim = cfg.head_dim as usize;
    let kv_start = kv_head_range(cfg.num_kv_heads, s).start as usize * head_dim;
    let (v_offset, v_rows, v_padded) = vocab_shard(cfg.vocab_size, s);
    for (slot, whole) in slots.iter_mut().zip(&full) {
        if slot.name != whole.name {
            return Err(mismatch(format!("{} against {}", slot.name, whole.name)));
        }
        if slot.name == EMBED || slot.name == LM_HEAD {
            slot.shape = vec![v_padded as usize, cfg.hidden as usize];
            slot.source = Some(SlotSource {
                shape: whole.shape.clone(),
                axis: 0,
                start: v_offset as usize,
                len: v_rows as usize,
            });
            continue;
        }
        if slot.shape == whole.shape {
            continue;
        }
        let differ: Vec<usize> = (0..slot.shape.len().min(whole.shape.len()))
            .filter(|&d| slot.shape[d] != whole.shape[d])
            .collect();
        let [axis] = differ[..] else {
            return Err(mismatch(format!(
                "{} is {:?} at rank width, {:?} unsharded",
                slot.name, slot.shape, whole.shape
            )));
        };
        let len = slot.shape[axis];
        let kv_part = ["k_proj.weight", "v_proj.weight", "k_norm.weight"]
            .iter()
            .any(|suffix| slot.name.ends_with(suffix));
        let start = if kv_part {
            kv_start
        } else {
            s.rank as usize * len
        };
        slot.source = Some(SlotSource {
            shape: whole.shape.clone(),
            axis,
            start,
            len,
        });
    }
    Ok(slots)
}

/// Every op config rank `s`'s forward runs over KV blocks of `block_tokens` tokens with `opts`
/// (the registry of that rank is built from it).
pub fn requirements(
    cfg: &ModelArchConfig,
    s: ShardSpec,
    block_tokens: u32,
    opts: ExecutorOptions,
) -> Result<Vec<OpRequirement>, ModelError> {
    let spec = tp_spec(cfg, s)?;
    DecoderExecutor::requirements_for(cfg, &spec, block_tokens, opts, Some(s))
}

/// [`requirements`] without the optional ops none of `providers` serves (as
/// [`crate::executor::available_requirements`]).
pub fn available_requirements(
    cfg: &ModelArchConfig,
    s: ShardSpec,
    block_tokens: u32,
    opts: ExecutorOptions,
    providers: &[Arc<dyn KernelProvider>],
) -> Result<Vec<OpRequirement>, ModelError> {
    Ok(requirements(cfg, s, block_tokens, opts)?
        .into_iter()
        .filter(|r| {
            !matches!(r.spec, OpConfig::AddRmsnorm(_))
                || providers.iter().any(|p| r.spec.supported_by(p.as_ref()))
        })
        .collect())
}

/// Device bytes of rank `s`'s executor buffers for batches within `limits`.
pub fn workspace_bytes(
    cfg: &ModelArchConfig,
    s: ShardSpec,
    limits: ExecutorLimits,
) -> Result<u64, ModelError> {
    let spec = tp_spec(cfg, s)?;
    DecoderExecutor::workspace_bytes_for(cfg, &spec, limits, Some(s))
}

/// Rank `tp.rank`'s executor over its shard `weights` (loaded from [`weight_slots`]); `registry`
/// must have been built from [`requirements`] of the same shard, block size and `opts`, and
/// `mem` must be the device memory `tp.stream` belongs to. Every rank of the group must be fed
/// the same batches in the same order; each returns the full logits.
#[allow(clippy::too_many_arguments)]
pub fn build_executor(
    cfg: &ModelArchConfig,
    weights: LoadedWeights,
    registry: Arc<KernelRegistry>,
    mem: Arc<dyn DeviceMemory>,
    limits: ExecutorLimits,
    opts: ExecutorOptions,
    tp: TpContext,
) -> Result<Box<dyn ModelExecutor>, ModelError> {
    let spec = tp_spec(cfg, tp.shard())?;
    Ok(Box::new(DecoderExecutor::new_tp(
        cfg,
        spec,
        weights,
        registry,
        mem,
        limits,
        opts,
        Some(tp),
    )?))
}

#[cfg(test)]
mod tests {
    use turbine_core::types::{DType, DeviceId};
    use turbine_tensor::host::HostMemory;

    use super::*;
    use crate::loader::{gate_up_proj_name, qkv_proj_name};
    use crate::testing::TempDir;
    use crate::testing::tiny::{write_tiny_llama, write_tiny_olmoe};
    use crate::{MAX_STAGING_BYTES, SafetensorsIndex, WeightLoader, stacked_experts_name};

    /// BF16 elements of checkpoint tensor `name`, as raw little-endian `u16`s.
    fn tensor(dir: &std::path::Path, name: &str) -> (Vec<usize>, Vec<u16>) {
        let index = SafetensorsIndex::open(dir).unwrap();
        let e = index.get(name).unwrap();
        let file = std::fs::read(&e.file).unwrap();
        let bytes = &file[e.range.start as usize..e.range.end as usize];
        (
            e.shape.clone(),
            bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect(),
        )
    }

    fn loaded(w: &LoadedWeights, name: &str) -> (Vec<usize>, Vec<u16>) {
        let t = &w.tensors[name];
        assert_eq!(t.dtype, DType::BF16);
        let mut raw = vec![0u8; t.numel() * 2];
        t.storage.copy_to_host(0, &mut raw).unwrap();
        (
            t.shape.to_vec(),
            raw.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect(),
        )
    }

    /// Rows `rows` of a row-major `[_, cols]` matrix.
    fn rows_of(m: &[u16], cols: usize, rows: std::ops::Range<usize>) -> Vec<u16> {
        m[rows.start * cols..rows.end * cols].to_vec()
    }

    /// Columns `c` of every row of a row-major `[rows, cols]` matrix.
    fn cols_of(m: &[u16], cols: usize, c: std::ops::Range<usize>) -> Vec<u16> {
        m.chunks_exact(cols)
            .flat_map(|row| row[c.clone()].to_vec())
            .collect()
    }

    fn load_rank(dir: &std::path::Path, cfg: &ModelArchConfig, s: ShardSpec) -> LoadedWeights {
        let index = SafetensorsIndex::open(dir).unwrap();
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
        let slots = weight_slots(cfg, s).unwrap();
        // A 64-byte staging buffer splits the per-row runs of the column blocks, too.
        let small = WeightLoader::load(&index, &slots, &mem, 64).unwrap();
        let w = WeightLoader::load(&index, &slots, &mem, MAX_STAGING_BYTES).unwrap();
        assert_eq!(small.weight_bytes, w.weight_bytes);
        for (name, t) in &w.tensors {
            assert_eq!(loaded(&small, name), loaded(&w, name), "{name}");
            let _ = t;
        }
        w
    }

    /// The tiny Llama (4 heads of 16, 2 KV heads, intermediate 128, vocabulary 263, tied) at
    /// tp 2 and tp 4: each rank loads exactly its rows and columns — Q rows of its heads, K/V rows
    /// of its KV head (the same head on ranks 0 and 1 at tp 4: replicated), O and down columns,
    /// gate/up rows, the vocabulary rows `[132·r, …)` with the last shard's padding row zero —
    /// the norms whole, and reads only those bytes. Breaks if any split takes another rank's
    /// slice, if the fused stacks are not rebuilt per rank, or if the loader reads the whole
    /// tensor.
    #[test]
    fn sharded_llama_loads_only_its_slices() {
        let tmp = TempDir::new("tp-slots-llama");
        let spec = write_tiny_llama(tmp.path(), 5);
        let cfg = &spec.config;
        let dir = spec.dir.as_path();
        let (h, hd) = (64usize, 16usize);
        let (_, q) = tensor(dir, "model.layers.1.self_attn.q_proj.weight");
        let (_, k) = tensor(dir, "model.layers.1.self_attn.k_proj.weight");
        let (_, v) = tensor(dir, "model.layers.1.self_attn.v_proj.weight");
        let (_, o) = tensor(dir, "model.layers.1.self_attn.o_proj.weight");
        let (_, gate) = tensor(dir, "model.layers.0.mlp.gate_proj.weight");
        let (_, up) = tensor(dir, "model.layers.0.mlp.up_proj.weight");
        let (_, down) = tensor(dir, "model.layers.0.mlp.down_proj.weight");
        let (_, embed) = tensor(dir, EMBED);
        let (_, norm) = tensor(dir, "model.norm.weight");
        let total: u64 = cfg.shape().weight_bytes;
        for world in [2u32, 4] {
            let mut sum = 0;
            for rank in 0..world {
                let s = ShardSpec { rank, world };
                let w = load_rank(dir, cfg, s);
                let r = rank as usize;
                let (heads, inter) = (4 / world as usize, 128 / world as usize);
                let kv_head = if world == 2 { r } else { r / 2 };
                let q_rows = rows_of(&q, h, r * heads * hd..(r + 1) * heads * hd);
                let kv = |m: &[u16]| rows_of(m, h, kv_head * hd..(kv_head + 1) * hd);
                let (shape, qkv) = loaded(&w, &qkv_proj_name(1));
                assert_eq!(shape, [heads * hd + 2 * hd, h], "{s:?}");
                assert_eq!(qkv, [q_rows, kv(&k), kv(&v)].concat(), "{s:?} qkv");
                let (shape, got) = loaded(&w, "model.layers.1.self_attn.o_proj.weight");
                assert_eq!(shape, [h, heads * hd]);
                assert_eq!(
                    got,
                    cols_of(
                        &o,
                        heads * hd * world as usize,
                        r * heads * hd..(r + 1) * heads * hd
                    )
                );
                let (shape, got) = loaded(&w, &gate_up_proj_name(0));
                assert_eq!(shape, [2 * inter, h]);
                let part = r * inter..(r + 1) * inter;
                assert_eq!(
                    got,
                    [
                        rows_of(&gate, h, part.clone()),
                        rows_of(&up, h, part.clone())
                    ]
                    .concat()
                );
                let (shape, got) = loaded(&w, "model.layers.0.mlp.down_proj.weight");
                assert_eq!(shape, [h, inter]);
                assert_eq!(got, cols_of(&down, 128, part));
                // Vocabulary: shards of ceil(263 / world) rows; the last one padded with zeros.
                let (offset, rows, padded) = vocab_shard(263, s);
                let (shape, got) = loaded(&w, EMBED);
                assert_eq!(shape, [padded as usize, h]);
                let (offset, rows, padded) = (offset as usize, rows as usize, padded as usize);
                assert_eq!(got[..rows * h], rows_of(&embed, h, offset..offset + rows));
                assert!(
                    got[rows * h..padded * h].iter().all(|&x| x == 0),
                    "{s:?} padding"
                );
                assert_eq!(
                    loaded(&w, "model.norm.weight").1,
                    norm,
                    "norms are replicated"
                );
                assert!(
                    !w.tensors.contains_key(LM_HEAD),
                    "tied: one shard serves both"
                );
                sum += w.weight_bytes;
                assert_eq!(
                    w.weight_bytes,
                    w.tensors
                        .values()
                        .map(|t| t.numel() as u64 * 2)
                        .sum::<u64>()
                        - ((padded - rows) * h * 2) as u64,
                    "{s:?}: bytes read are the shard's, padding excluded"
                );
            }
            if world == 2 {
                // Every sharded byte is read once across the group; replicated ones per rank.
                // Per layer the two norms, plus the final norm.
                let replicated: u64 = (2 * 2 * h + h) as u64 * 2;
                assert_eq!(sum, total + replicated * (u64::from(world) - 1));
            }
        }
        // A size the rules cannot split is refused before anything is loaded.
        let err = weight_slots(cfg, ShardSpec { rank: 0, world: 3 })
            .unwrap_err()
            .to_string();
        assert!(err.contains("parallel.tensor_parallel_size = 3"), "{err}");
    }

    /// The tiny OLMoE (4 heads = 4 KV heads, full-projection Q/K norms, 8 experts of 32, router
    /// replicated, untied) at tp 2: each expert's gate/up rows and down columns at the rank's
    /// intermediate slice in the stacked layout, the Q/K norm weights sliced to the rank's heads,
    /// the router whole, the untied LM head a vocabulary shard.
    #[test]
    fn sharded_olmoe_slices_experts_and_norms() {
        let tmp = TempDir::new("tp-slots-olmoe");
        let spec = write_tiny_olmoe(tmp.path(), 5);
        let cfg = &spec.config;
        let dir = spec.dir.as_path();
        let h = 64usize;
        let s = ShardSpec { rank: 1, world: 2 };
        let w = load_rank(dir, cfg, s);
        let (shape, gate) = loaded(&w, &stacked_experts_name(0, "gate_proj"));
        assert_eq!(shape, [8, 16, h]);
        let (shape, down) = loaded(&w, &stacked_experts_name(1, "down_proj"));
        assert_eq!(shape, [8, h, 16]);
        for e in 0..8 {
            let (_, g) = tensor(
                dir,
                &format!("model.layers.0.mlp.experts.{e}.gate_proj.weight"),
            );
            assert_eq!(
                gate[e * 16 * h..(e + 1) * 16 * h],
                rows_of(&g, h, 16..32),
                "expert {e}"
            );
            let (_, d) = tensor(
                dir,
                &format!("model.layers.1.mlp.experts.{e}.down_proj.weight"),
            );
            assert_eq!(
                down[e * h * 16..(e + 1) * h * 16],
                cols_of(&d, 32, 16..32),
                "expert {e}"
            );
        }
        let (_, qn) = tensor(dir, "model.layers.0.self_attn.q_norm.weight");
        assert_eq!(
            loaded(&w, "model.layers.0.self_attn.q_norm.weight").1,
            qn[32..64]
        );
        let (_, router) = tensor(dir, "model.layers.0.mlp.gate.weight");
        assert_eq!(loaded(&w, "model.layers.0.mlp.gate.weight").1, router);
        let (shape, head) = loaded(&w, LM_HEAD);
        assert_eq!(shape, [132, h]);
        let (_, full) = tensor(dir, LM_HEAD);
        assert_eq!(head[..131 * h], rows_of(&full, h, 132..263));
    }
}
