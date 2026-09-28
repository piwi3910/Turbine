//! Expert-parallel execution of a mixture-of-experts model across a group of ranks (P5 S-11).
//!
//! Every rank of an EP group holds the routed experts `turbine_distributed::expert` places on it
//! ([`ExpertPlacement`]: contiguous by default, 64 experts over 2 ranks = 32 each) and nothing of
//! the others. Everything else is either replicated (attention, router, norms, embedding and LM
//! head, [`EpAttention::Replicated`], tp = 1) or tensor-parallel over the same ranks
//! ([`EpAttention::TensorParallel`], tp = ep: attention, embedding and LM head sharded by
//! [`crate::tp`]'s rules, every local expert whole). The rank's executor is the ordinary
//! [`DecoderExecutor`] with the MoE hook restricted to the rank's experts:
//!
//! - every rank holds every token of the step (the hidden rows are identical on all ranks:
//!   trivially with replicated attention, after the O projection's all-reduce otherwise), so
//!   the router runs identically on every rank and its choices are exact;
//! - dispatch is local: `moe_experts` runs over the rank's experts only, one call per run of
//!   consecutive expert ids ([`EpDims::runs`]; one call for a contiguous placement) through the
//!   local expert range `[expert_begin, expert_end)` every provider takes (kernel ABI v2), with
//!   the weights of that run; the rows routed to remote experts are outside the range's part of
//!   the sorted list, so they are neither gathered, multiplied nor scattered;
//! - combine is one all-reduce of the per-rank outputs (BF16): each rank's output is the
//!   provider's weighted sum over its experts (ascending expert order, from zero), and the
//!   backends sum the ranks' values in FP32 in rank order and round to BF16 once. With
//!   tp = ep it is the all-reduce tensor parallelism already runs after the FFN (the partial
//!   sums of the ranks' experts instead of their intermediate columns); there is no all-to-all.
//!
//! Numerics against one device: a token whose selected experts all live on one rank gets one
//! device's bits (the other ranks add exact zeros); otherwise its experts' weighted outputs are
//! summed per rank and the rank sums added once, where one device rounds after every expert.
//!
//! Token counts: after every collected step the rank reads the step's per-layer expert offsets
//! back (one `[layers, experts + 1]` I32 copy, after the step's synchronisation) and adds each
//! expert's routed rows (token · expert pairs) to [`ExpertTokenCounts`], per expert per layer
//! and per rank of the placement. Every rank counts the same (the router is replicated); the
//! server reads one of them.
//!
//! Not captured into decode graphs, and no overlapped launches (as tensor parallelism).

use std::sync::{Arc, Mutex};

use turbine_core::types::KvLayout;
use turbine_distributed::collective::Collective;
use turbine_kernels::{KernelProvider, KernelRegistry, OpConfig, OpRequirement};
use turbine_tensor::{DeviceMemory, StreamRef};

pub use turbine_distributed::expert::{EP_KEY, ExpertPlacement, PLACEMENT_KEY};

use crate::ModelError;
use crate::config::{ModelArchConfig, unsupported};
use crate::executor::{
    DecoderDims, DecoderExecutor, DecoderSpec, ExecutorLimits, ExecutorOptions, ModelExecutor,
    TpDims,
};
use crate::loader::{LoadedWeights, StackPlace, WeightSlot, stacked_experts_name};
use crate::tp::{self, ShardSpec, TpContext};

/// The reason code of an EP group over a model without routed experts.
pub const EP_MOE_ONLY: &str = "ep_moe_only";

/// How an EP group runs everything but the experts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpAttention {
    /// tp = 1: attention, router, norms, embedding and LM head whole on every rank.
    Replicated,
    /// tp = ep: attention, embedding and LM head tensor-parallel over the EP ranks.
    TensorParallel,
}

/// One rank's place in an EP group: its rank, the group size and how attention runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpShard {
    pub rank: u32,
    pub world: u32,
    pub attention: EpAttention,
}

impl EpShard {
    /// The rank's tensor-parallel shard when attention is sharded (tp = ep), else `None`.
    pub fn tp_shard(&self) -> Option<ShardSpec> {
        (self.attention == EpAttention::TensorParallel).then_some(ShardSpec {
            rank: self.rank,
            world: self.world,
        })
    }
}

/// Routed rows (token · expert pairs) per expert since the executor started, recorded by an EP
/// rank after every collected step. `turbine_expert_rank_tokens_total{rank}` is
/// [`ExpertCountsSnapshot::per_rank`].
#[derive(Debug, Default)]
pub struct ExpertTokenCounts(Mutex<ExpertCountsSnapshot>);

/// What [`ExpertTokenCounts::snapshot`] returns.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExpertCountsSnapshot {
    /// Collected steps counted.
    pub steps: u64,
    /// Rows routed to the experts each rank holds, summed over the MoE layers; index = rank.
    pub per_rank: Vec<u64>,
    /// Rows routed to each expert id, summed over the MoE layers; index = expert.
    pub per_expert: Vec<u64>,
    /// `(layer, rows per expert)` per MoE layer, in layer order.
    pub per_layer: Vec<(u32, Vec<u64>)>,
}

impl ExpertCountsSnapshot {
    /// The `n` (layer, expert) pairs with the most routed rows, most first (ties: lower layer,
    /// then lower expert).
    pub fn top_experts(&self, n: usize) -> Vec<(u32, u32, u64)> {
        let mut all: Vec<(u32, u32, u64)> = self
            .per_layer
            .iter()
            .flat_map(|(l, rows)| {
                rows.iter()
                    .enumerate()
                    .map(move |(e, &r)| (*l, e as u32, r))
            })
            .collect();
        all.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
        all.truncate(n);
        all
    }
}

impl ExpertTokenCounts {
    pub fn snapshot(&self) -> ExpertCountsSnapshot {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Adds one step: `offsets[i]` are MoE layer `placement.layers[i]`'s `[experts + 1]` expert
    /// offsets (the sorted routed rows of expert `e` are `offsets[e]..offsets[e + 1]`).
    pub(crate) fn record(&self, placement: &ExpertPlacement, offsets: &[&[i32]]) {
        let mut s = self.0.lock().unwrap_or_else(|p| p.into_inner());
        if s.per_layer.len() != placement.layers.len() {
            let experts = placement.layers.first().map_or(0, |(_, r)| r.len());
            s.per_rank = vec![0; placement.ranks as usize];
            s.per_expert = vec![0; experts];
            s.per_layer = placement
                .layers
                .iter()
                .map(|(l, _)| (*l, vec![0; experts]))
                .collect();
        }
        let s = &mut *s;
        s.steps += 1;
        for ((layer_ranks, (_, layer_rows)), off) in placement
            .layers
            .iter()
            .map(|(_, r)| r)
            .zip(s.per_layer.iter_mut())
            .zip(offsets)
        {
            for (e, &rank) in layer_ranks.iter().enumerate() {
                let rows = u64::try_from(off[e + 1] - off[e]).unwrap_or(0);
                layer_rows[e] += rows;
                s.per_expert[e] += rows;
                s.per_rank[rank as usize] += rows;
            }
        }
    }
}

/// One rank of an expert-parallel group, as its executor sees it: its position, the placement
/// of every MoE layer's experts, the group's communicator (the combine; unused at tp = ep, where
/// the tensor-parallel communicator over the same ranks combines) and the stream the
/// collectives are ordered on (the rank's compute stream), plus the token counts the rank
/// records (keep a clone of the `Arc` to read them).
#[derive(Clone)]
pub struct EpContext {
    pub rank: u32,
    pub world: u32,
    pub placement: Arc<ExpertPlacement>,
    pub collective: Arc<dyn Collective>,
    pub stream: StreamRef,
    pub counts: Arc<ExpertTokenCounts>,
}

impl std::fmt::Debug for EpContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EpContext")
            .field("rank", &self.rank)
            .field("world", &self.world)
            .field("backend", &self.collective.backend())
            .field("stream", &self.stream)
            .finish()
    }
}

/// What an EP rank's decoder dims add ([`crate::executor::DecoderDims::ep`]): its rank, the
/// runs of consecutive expert ids it holds per decoder layer, the placement and the counts.
#[derive(Debug)]
pub struct EpDims {
    pub rank: u32,
    pub world: u32,
    /// Per decoder layer: the rank's experts as runs `[begin, end)` of global expert ids,
    /// ascending (one run for a contiguous placement).
    pub runs: Vec<Vec<(u32, u32)>>,
    pub placement: Arc<ExpertPlacement>,
    pub counts: Arc<ExpertTokenCounts>,
}

impl EpDims {
    /// Experts the rank holds in `layer`.
    pub fn local_experts(&self, layer: usize) -> u32 {
        self.runs[layer].iter().map(|(b, e)| e - b).sum()
    }
}

/// Runs of consecutive ids of ascending `ids`.
fn runs_of(ids: &[u32]) -> Vec<(u32, u32)> {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for &e in ids {
        match runs.last_mut() {
            Some((_, end)) if *end == e => *end += 1,
            _ => runs.push((e, e + 1)),
        }
    }
    runs
}

/// The MoE layers of `cfg` (every decoder layer of a model with routed experts, none
/// otherwise): the layers a placement covers.
pub fn moe_layers(cfg: &ModelArchConfig) -> Vec<u32> {
    if cfg.moe.is_some() {
        (0..cfg.num_layers).collect()
    } else {
        Vec::new()
    }
}

fn refuse(cfg: &ModelArchConfig, s: EpShard, why: impl std::fmt::Display) -> ModelError {
    unsupported(
        EP_KEY,
        format!("{} ({} {why})", s.world, cfg.family.0.name()),
        "a size dividing the routed experts of a mixture-of-experts family with a placement \
         covering every MoE layer (olmoe), tp = 1 or tp = ep",
    )
}

/// The family's decoder hooks for EP ranks: its tensor-parallel hooks, whose FFN must be the
/// mixture of experts.
fn ep_spec(cfg: &ModelArchConfig, s: EpShard) -> Result<DecoderSpec, ModelError> {
    let spec = cfg.family.0.tp_decoder_spec();
    match spec {
        Some(spec) if cfg.moe.is_some() && spec.ffn.name() == "moe" => Ok(spec),
        _ => Err(refuse(
            cfg,
            s,
            format!("{EP_MOE_ONLY}: the model has no routed experts to place"),
        )),
    }
}

/// Refuses an EP group `s` cannot run `cfg` with `placement` (reason in the message): a model
/// without routed experts ([`EP_MOE_ONLY`]), a rank outside the group, a placement of another
/// group size or not covering exactly the model's MoE layers and experts, or (tp = ep)
/// attention the tensor-parallel rules cannot split.
pub fn check(
    cfg: &ModelArchConfig,
    s: EpShard,
    placement: &ExpertPlacement,
) -> Result<(), ModelError> {
    ep_spec(cfg, s)?;
    if s.world == 0 || s.rank >= s.world {
        return Err(refuse(cfg, s, format!("has no rank {}", s.rank)));
    }
    if placement.ranks != s.world {
        return Err(refuse(
            cfg,
            s,
            format!("with a placement over {} ranks", placement.ranks),
        ));
    }
    let experts = cfg.moe.map_or(0, |m| m.num_experts) as usize;
    let layers: Vec<u32> = placement.layers.iter().map(|(l, _)| *l).collect();
    if layers != moe_layers(cfg) || placement.layers.iter().any(|(_, r)| r.len() != experts) {
        return Err(unsupported(
            PLACEMENT_KEY,
            format!("{} MoE layers", layers.len()),
            &format!(
                "every one of the model's {} MoE layers with all {experts} experts",
                cfg.num_layers
            ),
        ));
    }
    if let Some(t) = s.tp_shard() {
        tp::check(cfg, t)?;
    }
    Ok(())
}

/// `cfg` as rank `s` runs it: [`tp::rank_config`] of its shard at tp = ep with every expert
/// whole, `cfg` itself at tp = 1 (the expert count stays the model's: the router and the
/// offsets cover every expert).
pub fn rank_config(cfg: &ModelArchConfig, s: EpShard) -> Result<ModelArchConfig, ModelError> {
    match s.tp_shard() {
        Some(t) => {
            let mut rank = tp::rank_config(cfg, t)?;
            rank.moe = cfg.moe;
            Ok(rank)
        }
        None => Ok(cfg.clone()),
    }
}

/// The decoder dims of rank `s` (its tensor-parallel widths at tp = ep, every expert whole)
/// with [`EpDims`] from `placement` and `counts`.
pub(crate) fn rank_dims(
    cfg: &ModelArchConfig,
    s: EpShard,
    placement: &Arc<ExpertPlacement>,
    counts: &Arc<ExpertTokenCounts>,
) -> Result<DecoderDims, ModelError> {
    check(cfg, s, placement)?;
    let mut d = DecoderDims::for_shard(cfg, s.tp_shard())?;
    d.moe = cfg.moe;
    let runs = (0..cfg.num_layers)
        .map(|l| runs_of(&placement.local_experts(l, s.rank)))
        .collect();
    d.ep = Some(Arc::new(EpDims {
        rank: s.rank,
        world: s.world,
        runs,
        placement: Arc::clone(placement),
        counts: Arc::clone(counts),
    }));
    Ok(d)
}

/// The KV layout of rank `s`'s pool: the model's at tp = 1 (KV is replicated: every rank runs
/// the whole attention), the tensor-parallel rank's KV heads at tp = ep. Every rank's pool has
/// the leader's block count and is indexed by the leader's block ids.
pub fn kv_layout(
    cfg: &ModelArchConfig,
    s: EpShard,
    block_tokens: u32,
) -> Result<KvLayout, ModelError> {
    match s.tp_shard() {
        Some(t) => tp::kv_layout(cfg, t, block_tokens),
        None => Ok(cfg.kv_layout(block_tokens)),
    }
}

/// Rank `s`'s weight slots: everything but the experts as one device's (tp = 1) or as the
/// tensor-parallel rank's shard (tp = ep), plus every expert `placement` puts on the rank,
/// whole, stacked in ascending expert order into `[local experts, …]` stacks of the
/// [`stacked_experts_name`] parameters. Other ranks' experts are not read; load the slots with
/// [`crate::WeightLoader::load_part`] and the family's whole slot list so those experts are skipped
/// quietly.
pub fn weight_slots(
    cfg: &ModelArchConfig,
    s: EpShard,
    placement: &ExpertPlacement,
) -> Result<Vec<WeightSlot>, ModelError> {
    check(cfg, s, placement)?;
    let family = cfg.family.0;
    let base = match s.tp_shard() {
        Some(t) => tp::weight_slots(cfg, t)?,
        None => family.weight_slots(cfg),
    };
    let stacks: Vec<(u32, String)> = (0..cfg.num_layers)
        .flat_map(|l| {
            ["gate_proj", "up_proj", "down_proj"].map(|p| (l, stacked_experts_name(l, p)))
        })
        .collect();
    // The layer of a slot that is an expert of a layer's stack.
    let expert_layer = |slot: &WeightSlot| {
        let place = slot.stack.as_ref()?;
        stacks
            .iter()
            .find(|(_, name)| *name == place.name)
            .map(|(l, _)| *l)
    };
    let mut slots: Vec<WeightSlot> = base
        .into_iter()
        .filter(|slot| expert_layer(slot).is_none())
        .collect();
    // The experts come whole from the unsharded slots, restacked at their local index.
    for slot in family.weight_slots(cfg) {
        let (Some(layer), Some(place)) = (expert_layer(&slot), slot.stack.clone()) else {
            continue;
        };
        let numel: usize = slot.shape.iter().product();
        let expert = (place.offset / numel) as u32;
        let local = placement.local_experts(layer, s.rank);
        let Some(index) = local.iter().position(|&e| e == expert) else {
            continue;
        };
        let mut shape = place.shape;
        shape[0] = local.len();
        let stack = StackPlace {
            name: place.name,
            shape,
            offset: index * numel,
        };
        slots.push(WeightSlot {
            stack: Some(stack),
            ..slot
        });
    }
    Ok(slots)
}

/// Every op config rank `s`'s forward runs over KV blocks of `block_tokens` tokens with `opts`
/// (the registry of that rank is built from it): one `moe_experts` config per distinct run of
/// the rank's experts.
pub fn requirements(
    cfg: &ModelArchConfig,
    s: EpShard,
    placement: &Arc<ExpertPlacement>,
    block_tokens: u32,
    opts: ExecutorOptions,
) -> Result<Vec<OpRequirement>, ModelError> {
    let spec = ep_spec(cfg, s)?;
    let d = rank_dims(cfg, s, placement, &Arc::default())?;
    Ok(DecoderExecutor::requirements_of(
        &rank_config(cfg, s)?,
        &d,
        &spec,
        block_tokens,
        opts,
    ))
}

/// [`requirements`] without the optional ops none of `providers` serves (as
/// [`crate::executor::available_requirements`]).
pub fn available_requirements(
    cfg: &ModelArchConfig,
    s: EpShard,
    placement: &Arc<ExpertPlacement>,
    block_tokens: u32,
    opts: ExecutorOptions,
    providers: &[Arc<dyn KernelProvider>],
) -> Result<Vec<OpRequirement>, ModelError> {
    Ok(requirements(cfg, s, placement, block_tokens, opts)?
        .into_iter()
        .filter(|r| {
            !matches!(r.spec, OpConfig::AddRmsnorm(_))
                || providers.iter().any(|p| r.spec.supported_by(p.as_ref()))
        })
        .collect())
}

/// Device bytes of rank `s`'s executor buffers for batches within `limits`: one device's (or
/// the tensor-parallel rank's) with the per-layer expert offsets the counts read back.
pub fn workspace_bytes(
    cfg: &ModelArchConfig,
    s: EpShard,
    placement: &Arc<ExpertPlacement>,
    limits: ExecutorLimits,
) -> Result<u64, ModelError> {
    let spec = ep_spec(cfg, s)?;
    let d = rank_dims(cfg, s, placement, &Arc::default())?;
    let tp_bytes = match d.tp {
        Some(TpDims { world, .. }) => {
            4 * (2 * u64::from(limits.max_batch_tokens)
                + (1 + u64::from(world)) * u64::from(limits.max_seqs) * d.vocab_rows as u64)
        }
        None => 0,
    };
    Ok(DecoderExecutor::workspace_of(&rank_config(cfg, s)?, &d, &spec, limits) + tp_bytes)
}

/// Rank `ep.rank`'s executor over its `weights` (loaded from [`weight_slots`]); `registry` must
/// have been built from [`requirements`] of the same rank, placement, block size and `opts`,
/// and `mem` must be the device memory `ep.stream` belongs to. `tp` is `None` for replicated
/// attention (tp = 1) or rank `ep.rank` of the tensor-parallel group over the same ranks
/// (tp = ep). Every rank of the group must be fed the same batches in the same order; each
/// returns the full logits.
#[allow(clippy::too_many_arguments)]
pub fn build_executor(
    cfg: &ModelArchConfig,
    weights: LoadedWeights,
    registry: Arc<KernelRegistry>,
    mem: Arc<dyn DeviceMemory>,
    limits: ExecutorLimits,
    opts: ExecutorOptions,
    ep: EpContext,
    tp: Option<TpContext>,
) -> Result<Box<dyn ModelExecutor>, ModelError> {
    let s = EpShard {
        rank: ep.rank,
        world: ep.world,
        attention: if tp.is_some() {
            EpAttention::TensorParallel
        } else {
            EpAttention::Replicated
        },
    };
    let spec = ep_spec(cfg, s)?;
    Ok(Box::new(DecoderExecutor::new_parallel(
        cfg,
        spec,
        weights,
        registry,
        mem,
        limits,
        opts,
        tp,
        Some(ep),
    )?))
}

#[cfg(test)]
mod tests {
    use turbine_core::types::{DType, DeviceId};
    use turbine_tensor::host::HostMemory;

    use super::*;
    use crate::testing::TempDir;
    use crate::testing::tiny::write_tiny_olmoe;
    use crate::{MAX_STAGING_BYTES, SafetensorsIndex, WeightLoader};

    fn load(dir: &std::path::Path, slots: &[WeightSlot]) -> LoadedWeights {
        let index = SafetensorsIndex::open(dir).unwrap();
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
        WeightLoader::load(&index, slots, &mem, MAX_STAGING_BYTES).unwrap()
    }

    fn raw(w: &LoadedWeights, name: &str) -> (Vec<usize>, Vec<u8>) {
        let t = &w.tensors[name];
        assert_eq!(t.dtype, DType::BF16);
        let mut bytes = vec![0u8; t.numel() * 2];
        t.storage.copy_to_host(0, &mut bytes).unwrap();
        (t.shape.to_vec(), bytes)
    }

    /// The tiny OLMoE (8 experts, 2 layers) at EP 2 with a non-contiguous placement (layer 0:
    /// experts 1, 2, 5 and 6 on rank 1; layer 1: the odd experts): each rank's expert stacks
    /// hold exactly its experts, whole, in ascending order (`[4, 32, 64]`), the stacks of one
    /// device's load at those indices; everything else is one device's, and the rank reads
    /// only its experts' bytes (the two ranks together read one device's expert bytes once).
    /// At tp = ep the attention is the tensor-parallel rank's and the experts stay whole.
    /// Breaks if a rank loads another rank's expert, stacks them out of order or reads them.
    #[test]
    fn rank_loads_only_its_experts() {
        let tmp = TempDir::new("ep-slots");
        let spec = write_tiny_olmoe(tmp.path(), 5);
        let cfg = &spec.config;
        let dir = spec.dir.as_path();
        let placement = ExpertPlacement::parse(
            "0: [0, 1, 1, 0, 0, 1, 1, 0]\n1: [0, 1, 0, 1, 0, 1, 0, 1]\n",
            8,
            &moe_layers(cfg),
            2,
        )
        .unwrap();
        let one = load(dir, &cfg.family.0.weight_slots(cfg));
        let one_experts: u64 = one
            .tensors
            .iter()
            .filter(|(n, _)| n.contains("experts"))
            .map(|(_, t)| t.numel() as u64 * 2)
            .sum();
        let mut expert_bytes = 0;
        for attention in [EpAttention::Replicated, EpAttention::TensorParallel] {
            for rank in 0..2 {
                let s = EpShard {
                    rank,
                    world: 2,
                    attention,
                };
                let w = load(dir, &weight_slots(cfg, s, &placement).unwrap());
                let mut rank_experts = 0;
                for layer in 0..2 {
                    let local = placement.local_experts(layer, rank);
                    for proj in ["gate_proj", "up_proj", "down_proj"] {
                        let name = stacked_experts_name(layer, proj);
                        let (shape, got) = raw(&w, &name);
                        let (full_shape, full) = raw(&one, &name);
                        assert_eq!(shape[0], 4);
                        assert_eq!(shape[1..], full_shape[1..], "{name}: experts stay whole");
                        let per = full.len() / 8;
                        let want: Vec<u8> = local
                            .iter()
                            .flat_map(|&e| full[e as usize * per..(e as usize + 1) * per].to_vec())
                            .collect();
                        assert!(got == want, "{s:?} {name}: the rank's experts in order");
                        rank_experts += got.len() as u64;
                    }
                }
                let router = "model.layers.1.mlp.gate.weight";
                assert_eq!(
                    raw(&w, router),
                    raw(&one, router),
                    "the router is replicated"
                );
                let o = "model.layers.0.self_attn.o_proj.weight";
                let (shape, _) = raw(&w, o);
                if attention == EpAttention::Replicated {
                    expert_bytes += rank_experts;
                }
                let heads = if attention == EpAttention::Replicated {
                    64
                } else {
                    32
                };
                assert_eq!(shape, [64, heads], "{s:?}: attention replicated or sharded");
                if attention == EpAttention::Replicated {
                    let theirs = one_experts - rank_experts;
                    assert_eq!(
                        w.weight_bytes,
                        one.weight_bytes - theirs,
                        "{s:?} reads no other rank's expert"
                    );
                }
            }
        }
        assert_eq!(
            expert_bytes, one_experts,
            "every expert on exactly one rank"
        );
    }

    /// The token counts of a step: per layer, expert and the rank of the placement holding it.
    #[test]
    fn counts_follow_the_placement() {
        let placement =
            ExpertPlacement::parse("0: [0, 1, 1, 0]\n3: [1, 0, 0, 1]\n", 4, &[0, 3], 2).unwrap();
        let counts = ExpertTokenCounts::default();
        // Layer 0 routes 1, 2, 0, 3 rows to experts 0..4; layer 3 routes 2, 2, 2, 0.
        counts.record(&placement, &[&[0, 1, 3, 3, 6], &[0, 2, 4, 6, 6]]);
        counts.record(&placement, &[&[0, 1, 3, 3, 6], &[0, 2, 4, 6, 6]]);
        let s = counts.snapshot();
        assert_eq!(s.steps, 2);
        assert_eq!(s.per_layer, [(0, vec![2, 4, 0, 6]), (3, vec![4, 4, 4, 0])]);
        assert_eq!(s.per_expert, [6, 8, 4, 6]);
        // Rank 0: layer 0 experts 0, 3 (2 + 6), layer 3 experts 1, 2 (4 + 4).
        assert_eq!(s.per_rank, [16, 8]);
        assert_eq!(s.top_experts(2), [(0, 3, 6), (0, 1, 4)]);
    }

    #[test]
    fn runs_of_consecutive_ids() {
        assert_eq!(runs_of(&[0, 1, 2, 3]), [(0, 4)]);
        assert_eq!(runs_of(&[1, 2, 5, 6, 7]), [(1, 3), (5, 8)]);
        assert_eq!(runs_of(&[1, 3]), [(1, 2), (3, 4)]);
        assert!(runs_of(&[]).is_empty());
    }
}
