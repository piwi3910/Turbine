//! Pipeline-parallel stages of one model (P5 S-10, contract §10): each stage holds a contiguous
//! range of decoder layers on its own device and hands the hidden state to the next stage through
//! the group's point-to-point [`Collective::send`] / [`Collective::recv`].
//!
//! A stage is the ordinary [`DecoderExecutor`] over [`stage_config`] (the model with only the
//! stage's layers, so its KV layout, shape and batch limits are the stage's), whose layers load
//! the checkpoint's layers `layers.start..layers.end` ([`weight_slots`]). The first stage holds
//! the embedding, the last the final norm and the LM head (a tied embedding is loaded on both).
//! One forward of a batch on a group of `stages` stages:
//!
//! - stage 0 embeds the batch's tokens, runs its layers and sends the residual stream's rows
//!   (BF16 `[tokens, hidden]`, the activation dtype) to stage 1;
//! - every later stage first receives those rows from the stage before it, normalises them with
//!   its first layer's input norm and runs its layers; a middle stage sends its rows on;
//! - the last stage runs the final norm, the LM head and the device logits reduction, and returns
//!   the logits, as one device does. A non-last stage's `forward` / `collect` returns an empty
//!   [`crate::executor::Logits`] (0 rows): its output went to the next stage.
//!
//! Every stage is fed the same [`crate::executor::BatchInput`] (tokens, positions, sequence
//! slices, and its own KV pool, laid out as [`kv_layout`] of its layers and indexed by the
//! leader's block ids), in the same order. A stage's layer `i` is layer `layers.start + i` of the
//! model, so its pool holds the K/V of its layers only.
//!
//! Numerics: pipeline parallelism does not change arithmetic. The hand-off moves the residual
//! stream's bytes; the next stage's input norm of them is the op one device fuses with the
//! residual add (`add_rmsnorm`), which rounds the sum to the activation dtype and normalises
//! that, so the logits are bitwise one device's.
//!
//! Not captured into decode graphs, and no overlapped launches (as tensor parallelism: the
//! hand-off runs inside the launch); a forward's point-to-point calls are bracketed by
//! [`Collective::step_begin`] / `step_end`.

use std::ops::Range;
use std::sync::Arc;

use turbine_core::types::KvLayout;
use turbine_distributed::collective::Collective;
use turbine_kernels::{KernelProvider, KernelRegistry, OpConfig, OpRequirement};
use turbine_tensor::{DeviceMemory, StreamRef};

pub use turbine_distributed::pipeline::{PP_KEY, SPLIT_KEY, StageSpec};

use crate::ModelError;
use crate::config::{ModelArchConfig, unsupported};
use crate::executor::{
    DecoderExecutor, DecoderSpec, ExecutorLimits, ExecutorOptions, ModelExecutor,
};
use crate::loader::{LM_HEAD, LoadedWeights, WeightSlot};

/// The embedding's checkpoint name (on the first stage, and the last when tied).
const EMBED: &str = "model.embed_tokens.weight";
/// The final norm's checkpoint name (on the last stage).
const FINAL_NORM: &str = "model.norm.weight";

/// One stage of a pipeline-parallel group, as its executor sees it (contract §10): its position,
/// the model layers it runs, the group's communicator (rank = stage) and the stream the
/// hand-off is ordered on (the stage's compute stream).
#[derive(Clone)]
pub struct PpContext {
    pub stage: u32,
    pub stages: u32,
    pub layers: Range<u32>,
    pub collective: Arc<dyn Collective>,
    pub stream: StreamRef,
}

impl PpContext {
    /// Embeds the tokens (no stage before it).
    pub fn is_first(&self) -> bool {
        self.stage == 0
    }

    /// Returns the logits (no stage after it).
    pub fn is_last(&self) -> bool {
        self.stage + 1 == self.stages
    }
}

impl std::fmt::Debug for PpContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PpContext")
            .field("stage", &self.stage)
            .field("stages", &self.stages)
            .field("layers", &self.layers)
            .field("backend", &self.collective.backend())
            .field("stream", &self.stream)
            .finish()
    }
}

fn refuse(key: &str, value: String, supported: &str) -> ModelError {
    unsupported(key, value, supported)
}

/// The family's decoder hooks for pipeline stages, or why it has none.
fn stage_spec(cfg: &ModelArchConfig) -> Result<DecoderSpec, ModelError> {
    cfg.family.0.tp_decoder_spec().ok_or_else(|| {
        refuse(
            PP_KEY,
            format!(
                "> 1 ({} has no decoder hooks for pipeline stages)",
                cfg.family.0.name()
            ),
            "a family on the shared decoder skeleton (llama, olmoe)",
        )
    })
}

/// Refuses a layer range that is empty or reaches past the model's layers.
fn check_layers(cfg: &ModelArchConfig, layers: &Range<u32>) -> Result<(), ModelError> {
    if layers.is_empty() || layers.end > cfg.num_layers {
        return Err(refuse(
            SPLIT_KEY,
            format!("layers {layers:?}"),
            &format!(
                "a non-empty range of the model's {} layers per stage",
                cfg.num_layers
            ),
        ));
    }
    Ok(())
}

/// Refuses a stage the model cannot run as: a family without decoder hooks, an empty range or
/// one past the model's layers, the embedding anywhere but on the stage starting at layer 0, or
/// the final norm and LM head anywhere but on the stage ending at the last layer.
pub fn check_stage(cfg: &ModelArchConfig, stage: &StageSpec) -> Result<(), ModelError> {
    stage_spec(cfg)?;
    check_layers(cfg, &stage.layers)?;
    let first = stage.layers.start == 0;
    let last = stage.layers.end == cfg.num_layers;
    if stage.embedding != first || stage.lm_head != last {
        return Err(refuse(
            SPLIT_KEY,
            format!(
                "stage {} over layers {:?} with embedding {} and LM head {}",
                stage.stage, stage.layers, stage.embedding, stage.lm_head
            ),
            "the embedding on the stage starting at layer 0 and the final norm and LM head on \
             the stage ending at the last layer",
        ));
    }
    Ok(())
}

/// Refuses a pipeline plan the model cannot run: no stage, more stages than layers, stages out
/// of order, or layer ranges that do not cover every layer exactly once in order (each stage is
/// also checked by [`check_stage`]). One stage over every layer is one device.
pub fn check(cfg: &ModelArchConfig, stages: &[StageSpec]) -> Result<(), ModelError> {
    let n = stages.len() as u32;
    if n == 0 || n > cfg.num_layers {
        return Err(refuse(
            PP_KEY,
            n.to_string(),
            &format!("1..={} stages (the model's layer count)", cfg.num_layers),
        ));
    }
    let mut next = 0;
    for (i, s) in stages.iter().enumerate() {
        if s.stage != i as u32 || s.layers.start != next {
            return Err(refuse(
                SPLIT_KEY,
                format!(
                    "stage {} over layers {:?} at position {i}",
                    s.stage, s.layers
                ),
                "contiguous layer ranges in stage order covering every layer once",
            ));
        }
        check_stage(cfg, s)?;
        next = s.layers.end;
    }
    if next != cfg.num_layers {
        return Err(refuse(
            SPLIT_KEY,
            format!("stages covering layers 0..{next}"),
            &format!("stages covering all {} layers", cfg.num_layers),
        ));
    }
    Ok(())
}

/// `cfg` as a stage over `layers` runs it: `num_layers` is the stage's layer count (its KV
/// layout, shape and budget terms are the stage's); everything else stays the model's.
pub fn stage_config(
    cfg: &ModelArchConfig,
    layers: &Range<u32>,
) -> Result<ModelArchConfig, ModelError> {
    check_layers(cfg, layers)?;
    let mut stage = cfg.clone();
    stage.num_layers = layers.len() as u32;
    Ok(stage)
}

/// The KV layout of the stage over `layers`: its layers only. Every stage's pool has the
/// leader's block count and is indexed by the leader's block ids.
pub fn kv_layout(
    cfg: &ModelArchConfig,
    layers: &Range<u32>,
    block_tokens: u32,
) -> Result<KvLayout, ModelError> {
    Ok(stage_config(cfg, layers)?.kv_layout(block_tokens))
}

/// The model layer a checkpoint tensor (or stack) name belongs to, `None` outside the layers.
fn layer_of(name: &str) -> Option<u32> {
    name.strip_prefix("model.layers.")?
        .split('.')
        .next()?
        .parse()
        .ok()
}

/// The stage's weight slots: the family's slots of the stage's layers (fused and stacked as on
/// one device), the embedding on the first stage and on a tied last stage, the final norm and an
/// untied LM head on the last. Other stages' tensors are not read; load the slots with
/// [`crate::WeightLoader::load_part`] and the family's whole slot list so those tensors are skipped
/// quietly (counted, not warned as `unexpected_tensor`).
pub fn weight_slots(
    cfg: &ModelArchConfig,
    stage: &StageSpec,
) -> Result<Vec<WeightSlot>, ModelError> {
    check_stage(cfg, stage)?;
    let embed = stage.embedding || (stage.lm_head && cfg.tie_word_embeddings);
    Ok(cfg
        .family
        .0
        .weight_slots(cfg)
        .into_iter()
        .filter(|slot| match layer_of(&slot.name) {
            Some(l) => stage.layers.contains(&l),
            None if slot.name == EMBED => embed,
            None if slot.name == FINAL_NORM || slot.name == LM_HEAD => stage.lm_head,
            None => true,
        })
        .collect())
}

/// Every op config the stage over `layers` runs over KV blocks of `block_tokens` tokens with
/// `opts` (the registry of that stage is built from it): one device's list over the stage's
/// layers.
pub fn requirements(
    cfg: &ModelArchConfig,
    layers: &Range<u32>,
    block_tokens: u32,
    opts: ExecutorOptions,
) -> Result<Vec<OpRequirement>, ModelError> {
    let spec = stage_spec(cfg)?;
    Ok(DecoderExecutor::requirements(
        &stage_config(cfg, layers)?,
        &spec,
        block_tokens,
        opts,
    ))
}

/// [`requirements`] without the optional ops none of `providers` serves (as
/// [`crate::executor::available_requirements`]).
pub fn available_requirements(
    cfg: &ModelArchConfig,
    layers: &Range<u32>,
    block_tokens: u32,
    opts: ExecutorOptions,
    providers: &[Arc<dyn KernelProvider>],
) -> Result<Vec<OpRequirement>, ModelError> {
    Ok(requirements(cfg, layers, block_tokens, opts)?
        .into_iter()
        .filter(|r| {
            !matches!(r.spec, OpConfig::AddRmsnorm(_))
                || providers.iter().any(|p| r.spec.supported_by(p.as_ref()))
        })
        .collect())
}

/// Device bytes of the stage's executor buffers for batches within `limits`: one device's over
/// the stage's layers, the logits rows only on the last stage (a non-last stage keeps one).
pub fn workspace_bytes(
    cfg: &ModelArchConfig,
    stage: &StageSpec,
    limits: ExecutorLimits,
) -> Result<u64, ModelError> {
    check_stage(cfg, stage)?;
    let spec = stage_spec(cfg)?;
    Ok(DecoderExecutor::stage_workspace_bytes(
        &stage_config(cfg, &stage.layers)?,
        &spec,
        limits,
        stage.lm_head,
    ))
}

/// Stage `pp.stage`'s executor over its `weights` (loaded from [`weight_slots`] of the same
/// stage); `registry` must have been built from [`requirements`] of its layers, block size and
/// `opts`, and `mem` must be the device memory `pp.stream` belongs to; `pp.collective` is rank
/// `pp.stage` of a group of `pp.stages`. Every stage must be fed the same batches in the same
/// order; the last returns the logits, the others empty logits (0 rows).
#[allow(clippy::too_many_arguments)]
pub fn build_executor(
    cfg: &ModelArchConfig,
    weights: LoadedWeights,
    registry: Arc<KernelRegistry>,
    mem: Arc<dyn DeviceMemory>,
    limits: ExecutorLimits,
    opts: ExecutorOptions,
    pp: PpContext,
) -> Result<Box<dyn ModelExecutor>, ModelError> {
    let spec = stage_spec(cfg)?;
    Ok(Box::new(DecoderExecutor::new_stage(
        cfg, spec, weights, registry, mem, limits, opts, pp,
    )?))
}

#[cfg(test)]
mod tests {
    use turbine_core::types::DeviceId;
    use turbine_tensor::host::HostMemory;

    use super::*;
    use crate::testing::TempDir;
    use crate::testing::tiny::{write_tiny_llama, write_tiny_olmoe};
    use crate::{MAX_STAGING_BYTES, SafetensorsIndex, WeightLoader};

    fn stage(stage: u32, layers: Range<u32>, n: u32) -> StageSpec {
        StageSpec {
            stage,
            device: DeviceId(stage),
            embedding: layers.start == 0,
            lm_head: layers.end == n,
            layers,
        }
    }

    /// The error's configuration key.
    fn key(e: ModelError) -> String {
        match e {
            ModelError::Unsupported { field, .. } => field,
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    /// A pipeline plan that drops, repeats or reorders a layer, has an empty stage, a stage
    /// past the last layer, more stages than layers, the embedding or the LM head on the wrong
    /// stage, or runs a family without decoder hooks is refused before anything is loaded,
    /// naming `parallel.pipeline.layer_split` (or the stage count's key); the 2-stage split of
    /// the tiny 2-layer Llama and one stage over both layers are accepted. Breaks if a bad
    /// split reaches the loader.
    #[test]
    fn pp_refuses_bad_split() {
        let tmp = TempDir::new("pp-split");
        let spec = write_tiny_llama(tmp.path(), 3);
        let cfg = &spec.config;
        assert_eq!(cfg.num_layers, 2);
        check(cfg, &[stage(0, 0..1, 2), stage(1, 1..2, 2)]).expect("2 stages");
        check(cfg, &[stage(0, 0..2, 2)]).expect("one stage");

        let bad: Vec<(&str, Vec<StageSpec>, &str)> = vec![
            ("no stage", vec![], PP_KEY),
            (
                "3 stages for 2 layers",
                vec![stage(0, 0..1, 2), stage(1, 1..2, 2), stage(2, 2..2, 2)],
                PP_KEY,
            ),
            ("layer 1 dropped", vec![stage(0, 0..1, 2)], SPLIT_KEY),
            (
                "layer 0 twice",
                vec![stage(0, 0..1, 2), stage(1, 0..2, 2)],
                SPLIT_KEY,
            ),
            (
                "out of order",
                vec![stage(0, 1..2, 2), stage(1, 0..1, 2)],
                SPLIT_KEY,
            ),
            (
                "empty stage",
                vec![stage(0, 0..0, 2), stage(1, 0..2, 2)],
                SPLIT_KEY,
            ),
            (
                "past the last layer",
                vec![stage(0, 0..1, 3), stage(1, 1..3, 3)],
                SPLIT_KEY,
            ),
            (
                "stage numbers",
                vec![stage(0, 0..1, 2), stage(2, 1..2, 2)],
                SPLIT_KEY,
            ),
        ];
        for (what, stages, want) in bad {
            let err = check(cfg, &stages).expect_err(what);
            assert_eq!(key(err), want, "{what}");
        }
        let mut no_head = stage(1, 1..2, 2);
        no_head.lm_head = false;
        let mut embeds = stage(1, 1..2, 2);
        embeds.embedding = true;
        for s in [no_head, embeds] {
            assert_eq!(key(check_stage(cfg, &s).unwrap_err()), SPLIT_KEY, "{s:?}");
            assert_eq!(key(weight_slots(cfg, &s).unwrap_err()), SPLIT_KEY, "{s:?}");
        }
        let err = stage_config(cfg, &(1..3)).unwrap_err();
        assert_eq!(key(err), SPLIT_KEY);
    }

    fn names(slots: &[WeightSlot]) -> Vec<String> {
        let mut n: Vec<String> = slots.iter().map(|s| s.name.clone()).collect();
        n.sort();
        n
    }

    /// Each stage of the tiny Llama (tied) and the tiny OLMoE (untied, stacked experts) loads
    /// exactly its layers' tensors: stage 0 layer 0 and the embedding, stage 1 layer 1, the
    /// final norm and the LM head (the embedding again when tied); together the two stages read
    /// every one-device tensor, the tied embedding twice, and nothing else; a stage's KV layout
    /// has its layers only. Breaks if a stage loads another stage's layer or misses the tied
    /// embedding.
    #[test]
    fn stage_loads_only_its_layers() {
        let tmp = TempDir::new("pp-slots");
        for spec in [
            write_tiny_llama(&tmp.path().join("llama"), 3),
            write_tiny_olmoe(&tmp.path().join("olmoe"), 3),
        ] {
            let cfg = &spec.config;
            let one = cfg.family.0.weight_slots(cfg);
            let stages = [stage(0, 0..1, 2), stage(1, 1..2, 2)];
            let slots: Vec<Vec<WeightSlot>> = stages
                .iter()
                .map(|s| weight_slots(cfg, s).expect("slots"))
                .collect();
            for (s, slots) in stages.iter().zip(&slots) {
                for slot in slots {
                    match layer_of(&slot.name) {
                        Some(l) => assert!(s.layers.contains(&l), "{s:?}: {}", slot.name),
                        None if slot.name == EMBED => {
                            assert!(s.embedding || cfg.tie_word_embeddings, "{s:?}")
                        }
                        None => assert!(s.lm_head, "{s:?}: {}", slot.name),
                    }
                }
                let layout = kv_layout(cfg, &s.layers, 128).expect("layout");
                assert_eq!(layout.num_layers, 1);
                assert_eq!(
                    layout.block_bytes() * 2,
                    cfg.kv_layout(128).block_bytes(),
                    "{s:?}"
                );
            }
            let mut both: Vec<WeightSlot> = slots.concat();
            if cfg.tie_word_embeddings {
                let at = both.iter().rposition(|s| s.name == EMBED).expect("tied");
                both.remove(at);
            } else {
                assert!(!names(&slots[1]).contains(&EMBED.to_string()));
                assert!(names(&slots[1]).contains(&LM_HEAD.to_string()));
            }
            assert_eq!(names(&both), names(&one), "{}", cfg.family.0.name());
            // The loader reads a stage's slots (stacks included) from the checkpoint.
            let index = SafetensorsIndex::open(&spec.dir).unwrap();
            let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
            let w = WeightLoader::load(&index, &slots[1], &mem, MAX_STAGING_BYTES).unwrap();
            assert!(
                w.tensors.keys().all(|n| layer_of(n) != Some(0)),
                "{:?}",
                w.tensors.keys()
            );
            assert!(w.tensors.contains_key(FINAL_NORM));
            assert!(
                !w.unexpected.is_empty(),
                "plain load warns stage 0's tensors"
            );
            // Told the whole model's slots, the loader skips stage 0's tensors quietly.
            let part = WeightLoader::load_part(
                cfg.weight_format.0,
                &index,
                &slots[1],
                &one,
                &mem,
                MAX_STAGING_BYTES,
            )
            .unwrap();
            assert!(part.unexpected.is_empty(), "{:?}", part.unexpected);
            assert_eq!(part.elsewhere, w.unexpected.len());
            assert_eq!(part.weight_bytes, w.weight_bytes);
        }
    }
}
