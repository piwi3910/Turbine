//! The FP8 e4m3 checkpoint layout shared by the `ct_fp8` (compressed-tensors) and `hf_fp8`
//! (`quant_method: fp8`) packagings (Phase 6a S-3, S-4): [`Fp8Format`] over an
//! [`Fp8Packaging`], which only parses and writes its `quantization_config`.
//!
//! A quantized linear layer `X` (checkpoint `X.weight`, F8_E4M3 `[n, k]`) comes with its weight
//! scale (`X.weight_scale`, or `X.weight_scale_inv` for block-scaled `quant_method: fp8`; F32,
//! BF16 or F16) and, for static activations, `X.input_scale` (one value). The loader stores,
//! under the layer's parameter name `P` (the stacked name of a fused projection, else
//! `X.weight`):
//!
//! - `P`: the e4m3 bytes, `[n, k]` (stacked by rows like the BF16 weights; `[E, n, k]` for a
//!   stack of experts);
//! - `P_scale`: F32 scales — `[n]` per output row (per-tensor scales are expanded to one per row,
//!   so the parts of a fused projection keep their own scale exactly: spec edge case "a fused
//!   stack whose parts carry different per-tensor FP8 scales"), or `[n / 128, k / 128]` per
//!   block;
//! - `P_input_scale` (static activations): F32 `[n]`, the part's value repeated per row; the
//!   decoder quantizes a fused projection's input with the largest of its parts' values (vLLM's
//!   rule).
//!
//! `lm_head` is never quantized (Phase 6a: the LM head stays BF16), nor are embeddings, norms
//! or the modules the packaging's ignore list names.
use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;

use turbine_core::registry::Module;
use turbine_core::support::WeightFormatColumn;
use turbine_core::types::DType;
use turbine_kernels::QGemmConfig;
use turbine_kernels::cpu::quant::{dequantize, fp8_e4m3_round, fp8_e4m3_value};
use turbine_kernels::quant::{ActQuantDesc, QuantSchemeDesc};

use super::common::{
    Follows, Owned, bf16_bytes, bf16_values, f32_bytes, matches_ignore, module_of, pow2_at_least,
    quantization_config, read_owned, scheme_unsupported, shard_like, shard_misaligned,
    write_fixture,
};
use super::{ActivationQuant, LinearSlot, QuantScheme, WeightFormat};
use crate::ModelError;
use crate::config::unsupported;
use crate::loader::{LM_HEAD, StackPlace, WeightSlot};
use crate::safetensors::{Dtype, TensorEntry};

/// How the weights of a quantized layer are scaled.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fp8Weights {
    Tensor,
    Channel,
    Block { n: u32, k: u32 },
}

/// The block shape Phase 6a serves (the Qwen3-FP8 / DeepSeek style).
pub const FP8_BLOCK: u32 = 128;

/// The FP8 e4m3 maximum magnitude (OCP e4m3fn).
const FP8_MAX: f32 = 448.0;

/// One FP8 packaging's parsed configuration.
#[derive(Clone, PartialEq, Debug)]
pub struct Fp8Layout {
    pub weights: Fp8Weights,
    pub act: ActivationQuant,
    /// Module names left unquantized (exact names, bare final components, or `re:` regular
    /// expressions).
    pub ignore: Vec<String>,
    /// The weight-scale tensor suffix (`weight_scale` or `weight_scale_inv`).
    pub scale_suffix: &'static str,
    /// Block-scaled layers (checkpoint weight names) decoded to BF16 at load because no
    /// selected provider runs them ([`Fp8Layout::fallback_layers`]); empty as parsed.
    pub decoded: Vec<String>,
}

/// A checkpoint container of FP8 weights: its registry name, how its `quantization_config`
/// is recognised and parsed, and (test support) written.
pub trait Fp8Packaging: Send + Sync + 'static {
    const NAME: &'static str;
    /// The layout of the registry entry (used only by the conformance suite's fixtures).
    const DEFAULT: Fp8Layout;
    /// `Ok` when `q` (a non-null `quantization_config`) is this container's FP8; else the
    /// refusal naming the key.
    fn claims(q: &serde_json::Value) -> Result<(), ModelError>;
    /// The layout `q` declares; refuses a variant Phase 6a does not serve
    /// (`quant_scheme_unsupported` naming the field).
    fn parse(q: &serde_json::Value) -> Result<Fp8Layout, ModelError>;
    /// The `quantization_config` declaring `layout` (test support).
    fn to_json(layout: &Fp8Layout) -> serde_json::Value;
}

impl Fp8Layout {
    /// Whether the linear layer with parameter (checkpoint) name `name` is quantized: a 2-D
    /// `….weight` other than the token embedding and `lm_head`, not in `ignore`.
    pub fn quantizes(&self, name: &str, shape: &[usize]) -> bool {
        let Some(module) = module_of(name) else {
            return false;
        };
        shape.len() == 2
            && !name.ends_with("embed_tokens.weight")
            && name != LM_HEAD
            && !self.ignore.iter().any(|pat| matches_ignore(pat, module))
    }

    /// Whether block-scaled layer `name` is decoded to BF16 at load and served by the BF16
    /// GEMM: the fallback for a layer no selected provider runs (user decision 2026-09-29,
    /// "Write own kernel in 6a": `turbine_hip_fp8_block` serves the rest in FP8).
    pub fn decodes(&self, name: &str) -> bool {
        matches!(self.weights, Fp8Weights::Block { .. }) && self.decoded.iter().any(|d| d == name)
    }

    /// Activations of block-scaled layers stay BF16 (W8A16: `turbine_hip_fp8_block` and the
    /// decode fallback, decision "P6: block-scaled FP8 GEMM"); else the checkpoint's scheme.
    pub fn activation(&self) -> ActivationQuant {
        match self.weights {
            Fp8Weights::Block { .. } => ActivationQuant::None,
            Fp8Weights::Tensor | Fp8Weights::Channel => self.act,
        }
    }

    /// The block-scaled layers of `slots` (the slots one device loads) to decode to BF16 at
    /// load, each stack with its `[n, k]` and reason: a layer whose stack (the parts of a fused
    /// projection together) `supports` refuses at the stack's or a part's `[n, k]`
    /// (`kernel_unsupported`), or whose tensor-parallel shard cuts a scale block
    /// (`shard_misaligned`). Whole stacks, in slot order; empty for other weights.
    pub fn fallback_layers(
        &self,
        slots: &[WeightSlot],
        supports: &dyn Fn(&QGemmConfig) -> bool,
    ) -> Vec<(Vec<String>, [usize; 2], &'static str)> {
        let Fp8Weights::Block { n: bn, k: bk } = self.weights else {
            return Vec::new();
        };
        let qgemm = |n: usize, k: usize| QGemmConfig {
            n: n as u32,
            k: k as u32,
            scheme: QuantSchemeDesc::Fp8Block {
                block_n: bn,
                block_k: bk,
            },
            act_quant: ActQuantDesc::None,
            a_dtype: super::ACTIVATION_DTYPE,
            c_dtype: super::ACTIVATION_DTYPE,
        };
        // Per stack (or lone layer): its key, parts, [n, k] and first refusal.
        type Group = (String, Vec<String>, [usize; 2], Option<&'static str>);
        let mut groups: Vec<Group> = Vec::new();
        for slot in slots {
            if !self.quantizes(&slot.name, &slot.shape) || self.decodes(&slot.name) {
                continue;
            }
            let (n, k) = (slot.shape[0], slot.shape[1]);
            let (key, whole) = match &slot.stack {
                Some(p) if p.shape.len() == 2 => (p.name.clone(), [p.shape[0], p.shape[1]]),
                _ => (slot.name.clone(), [n, k]),
            };
            let reason = if !self.derive(slot).1 {
                Some("shard_misaligned")
            } else if !(supports(&qgemm(n, k)) && supports(&qgemm(whole[0], whole[1]))) {
                Some("kernel_unsupported")
            } else {
                None
            };
            match groups.iter_mut().find(|g| g.0 == key) {
                Some(g) => {
                    g.1.push(slot.name.clone());
                    g.3 = g.3.or(reason);
                }
                None => groups.push((key, vec![slot.name.clone()], whole, reason)),
            }
        }
        groups
            .into_iter()
            .filter_map(|(_, parts, shape, reason)| reason.map(|r| (parts, shape, r)))
            .collect()
    }

    pub fn scheme(&self, l: &LinearSlot) -> QuantScheme {
        if !self.quantizes(&l.name, &[l.n as usize, l.k as usize]) || self.decodes(&l.name) {
            return QuantScheme::Bf16;
        }
        match self.weights {
            // Per-tensor scales are stored per row (exact): one scheme for fused projections.
            Fp8Weights::Tensor | Fp8Weights::Channel => QuantScheme::Fp8Channel,
            Fp8Weights::Block { n, k } => QuantScheme::Fp8Block { n, k },
        }
    }

    /// The column: `fp8` for tensor or channel scales, `fp8_block` for block scales.
    pub fn column(&self) -> WeightFormatColumn {
        match self.weights {
            Fp8Weights::Tensor | Fp8Weights::Channel => WeightFormatColumn::Fp8,
            Fp8Weights::Block { .. } => WeightFormatColumn::Fp8Block,
        }
    }

    /// The scale shape of `rows` rows of `k` columns.
    fn scale_shape(&self, rows: usize, k: usize) -> Vec<usize> {
        match self.weights {
            Fp8Weights::Tensor | Fp8Weights::Channel => vec![rows],
            Fp8Weights::Block { n: bn, k: bk } => {
                vec![rows.div_ceil(bn as usize), k.div_ceil(bk as usize)]
            }
        }
    }

    /// The slots of family slot `base`: itself (the e4m3 bytes when quantized), its scale and,
    /// with static activations, its input scale — stacked like `base` (by rows of a 2-D stack,
    /// or per entry of a stack of experts).
    pub fn slots(&self, base: &WeightSlot) -> Vec<WeightSlot> {
        self.derive(base).0
    }

    /// Refuses a tensor-parallel shard that cuts a scale block.
    pub fn check_shard(&self, base: &WeightSlot) -> Result<(), ModelError> {
        match (&base.source, self.derive(base).1) {
            (Some(src), false) => Err(shard_misaligned(
                &base.name,
                src,
                "the 128 × 128 scale block",
            )),
            _ => Ok(()),
        }
    }

    /// [`Fp8Layout::slots`], sharded like `base`, and whether the shard is aligned.
    fn derive(&self, base: &WeightSlot) -> (Vec<WeightSlot>, bool) {
        let mut out = vec![base.clone()];
        if !self.quantizes(&base.name, &base.shape) {
            return (out, true);
        }
        let module = module_of(&base.name).expect("quantizes() checked the suffix");
        if self.decodes(&base.name) {
            // The weight is decoded into its BF16 slot (sharded like any BF16 weight); its
            // scales (and a static input scale) are only read, as companions or checked.
            out.push(check_slot(&format!("{module}.{}", self.scale_suffix)));
            if self.act == ActivationQuant::Fp8PerTensorStatic {
                out.push(check_slot(&format!("{module}.input_scale")));
            }
            return (out, true);
        }
        let (n, k) = (base.shape[0], base.shape[1]);
        // The stack's leading dimensions (none for a 2-D stack), its rows and `base`'s first
        // row in the stack's rows flattened.
        let (key, lead, rows, row0) = match &base.stack {
            Some(p) => {
                let d = p.shape.len();
                (
                    p.name.clone(),
                    p.shape[..d - 2].to_vec(),
                    p.shape[d - 2],
                    p.offset / k,
                )
            }
            None => (base.name.clone(), Vec::new(), n, 0),
        };
        let stacked = |per_rows: Vec<usize>| {
            let mut shape = lead.clone();
            shape.extend(per_rows);
            shape
        };
        // An entry of a stack of experts starts at a whole matrix; rows of a 2-D stack start at
        // a block boundary (checked by `check_tensor` for block scales).
        let (entry, row_in) = (row0 / rows, row0 % rows);
        let scale_offset = |per_row: usize, per_matrix: usize| {
            entry * per_matrix
                + match self.weights {
                    Fp8Weights::Tensor | Fp8Weights::Channel => row_in * per_row,
                    Fp8Weights::Block { n: bn, .. } => (row_in / bn as usize) * per_row,
                }
        };
        let matrix = self.scale_shape(rows, k);
        let per_row = matrix.get(1).copied().unwrap_or(1);
        out.push(WeightSlot {
            name: format!("{module}.{}", self.scale_suffix),
            shape: self.scale_shape(n, k),
            stack: Some(StackPlace {
                name: format!("{key}_scale"),
                shape: stacked(matrix.clone()),
                offset: scale_offset(per_row, matrix.iter().product()),
            }),
            source: None,
        });
        if self.act == ActivationQuant::Fp8PerTensorStatic {
            out.push(WeightSlot {
                name: format!("{module}.input_scale"),
                shape: vec![n],
                stack: Some(StackPlace {
                    name: format!("{key}_input_scale"),
                    shape: stacked(vec![rows]),
                    offset: entry * rows + row_in,
                }),
                source: None,
            });
        }
        let mut aligned = true;
        if let Some(src) = &base.source {
            let (full_n, full_k) = (src.shape[0], src.shape[1]);
            let (rows_per, cols) = match self.weights {
                Fp8Weights::Tensor | Fp8Weights::Channel => (1, None),
                Fp8Weights::Block { n: bn, k: bk } => (bn as usize, Some(bk as usize)),
            };
            let scale_full = self.scale_shape(full_n, full_k);
            aligned &= shard_like(
                &mut out[1],
                src,
                scale_full,
                Follows {
                    rows: rows_per,
                    cols,
                },
            );
            if let Some(input) = out.get_mut(2) {
                aligned &= shard_like(
                    input,
                    src,
                    vec![full_n],
                    Follows {
                        rows: 1,
                        cols: None,
                    },
                );
            }
        }
        (out, aligned)
    }

    pub fn slot_dtype(&self, slot: &WeightSlot) -> DType {
        if self.decodes(&slot.name) && self.quantizes(&slot.name, &slot.shape) {
            super::Bf16::DTYPE
        } else if self.repacks(slot) {
            DType::F32
        } else if self.quantizes(&slot.name, &slot.shape) {
            DType::F8E4M3
        } else {
            super::Bf16::DTYPE
        }
    }

    fn is_scale(&self, name: &str) -> bool {
        name.ends_with(&format!(".{}", self.scale_suffix)) || name.ends_with(".input_scale")
    }

    /// The scale slots are converted to F32 (and expanded per row); a decoded block-scaled
    /// weight is converted to BF16.
    pub fn repacks(&self, slot: &WeightSlot) -> bool {
        self.is_scale(&slot.name)
            || (self.decodes(&slot.name) && self.quantizes(&slot.name, &slot.shape))
    }

    /// A decoded weight reads its block scales.
    pub fn companions(&self, slot: &WeightSlot) -> Vec<String> {
        match module_of(&slot.name) {
            Some(module) if self.decodes(&slot.name) && self.quantizes(&slot.name, &slot.shape) => {
                vec![format!("{module}.{}", self.scale_suffix)]
            }
            _ => Vec::new(),
        }
    }

    /// [`Fp8Layout::repack`], and the BF16 decode of a block-scaled weight (`companions`: its
    /// scales); check-only scale slots are validated and store nothing.
    pub fn repack_with(
        &self,
        slot: &WeightSlot,
        entry: &TensorEntry,
        bytes: Vec<u8>,
        companions: &[(&TensorEntry, Vec<u8>)],
    ) -> Result<Vec<u8>, ModelError> {
        if slot.shape == [0] {
            scale_values(entry, &bytes)?;
            return Ok(Vec::new());
        }
        let Fp8Weights::Block { n: bn, k: bk } = self.weights else {
            return self.repack(slot, entry, bytes);
        };
        if !(self.decodes(&slot.name) && self.quantizes(&slot.name, &slot.shape)) {
            return self.repack(slot, entry, bytes);
        }
        let [(scale_entry, scale_bytes)] = companions else {
            return Err(ModelError::MissingTensor(format!(
                "the block scales of {}",
                entry.name
            )));
        };
        let (n, k) = (slot.shape[0], slot.shape[1]);
        let scales = scale_values(scale_entry, scale_bytes)?;
        let scheme = QuantSchemeDesc::Fp8Block {
            block_n: bn,
            block_k: bk,
        };
        if bytes.len() != n * k || scales.len() != scheme.scale_count(n, k) {
            return Err(ModelError::Safetensors {
                file: entry.file.clone(),
                tensor: entry.name.clone(),
                rule: format!(
                    "{} bytes and {} block scales for [{n}, {k}]",
                    bytes.len(),
                    scales.len()
                ),
            });
        }
        Ok(bf16_bytes(&dequantize(scheme, &bytes, &scales, None, n, k)))
    }

    pub fn check_tensor(&self, entry: &TensorEntry) -> Result<(), ModelError> {
        let ok = if self.is_scale(&entry.name) {
            matches!(entry.dtype, Dtype::F32 | Dtype::BF16 | Dtype::F16)
        } else if self.quantizes(&entry.name, &entry.shape) {
            entry.dtype == Dtype::F8_E4M3
        } else {
            entry.dtype == Dtype::BF16
        };
        if !ok {
            return Err(ModelError::Unsupported {
                field: "tensor dtype".to_string(),
                value: format!("{} ({})", entry.dtype, entry.name),
                supported: "F8_E4M3 quantized weights, F32/BF16/F16 scales, BF16 otherwise"
                    .to_string(),
            });
        }
        if let Fp8Weights::Block { n: bn, k: bk } = self.weights
            && entry.dtype == Dtype::F8_E4M3
            && !(entry.shape[0].is_multiple_of(bn as usize)
                && entry.shape[1].is_multiple_of(bk as usize))
        {
            return Err(unsupported(
                "weight shape",
                format!("{:?} ({})", entry.shape, entry.name),
                &format!("quant_scheme_unsupported: multiples of the [{bn}, {bk}] block"),
            ));
        }
        Ok(())
    }

    /// Scales to F32; a per-tensor value (weight or input scale) repeated for each of the
    /// slot's rows.
    pub fn repack(
        &self,
        slot: &WeightSlot,
        entry: &TensorEntry,
        bytes: Vec<u8>,
    ) -> Result<Vec<u8>, ModelError> {
        let values = scale_values(entry, &bytes)?;
        let want: usize = slot.shape.iter().product();
        let per_tensor = slot.name.ends_with(".input_scale") || self.weights == Fp8Weights::Tensor;
        let out: Vec<f32> = if per_tensor && values.len() == 1 {
            vec![values[0]; want]
        } else if values.len() == want && !per_tensor {
            values
        } else {
            return Err(ModelError::Safetensors {
                file: entry.file.clone(),
                tensor: entry.name.clone(),
                rule: format!(
                    "{} scales for a slot of shape {:?} ({:?} weights)",
                    values.len(),
                    slot.shape,
                    self.weights
                ),
            });
        };
        Ok(out.iter().flat_map(|v| v.to_le_bytes()).collect())
    }
}

/// A slot only validated: no elements, not stored.
fn check_slot(name: &str) -> WeightSlot {
    WeightSlot {
        name: name.to_string(),
        shape: vec![0],
        stack: None,
        source: None,
    }
}

/// A scale tensor's values as F32.
fn scale_values(entry: &TensorEntry, bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    Ok(match entry.dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        other => {
            return Err(unsupported(
                "scale dtype",
                format!("{other} ({})", entry.name),
                "F32, BF16, F16",
            ));
        }
    })
}

/// A weight format over the FP8 layout of packaging `P`: the registry entry holds
/// [`Fp8Packaging::DEFAULT`], [`WeightFormat::configure`] the checkpoint's layout.
pub struct Fp8Format<P> {
    pub layout: Fp8Layout,
    packaging: PhantomData<fn() -> P>,
}

impl<P: Fp8Packaging> Fp8Format<P> {
    /// The registry entry.
    pub const ENTRY: Fp8Format<P> = Fp8Format::with(P::DEFAULT);

    pub const fn with(layout: Fp8Layout) -> Fp8Format<P> {
        Fp8Format {
            layout,
            packaging: PhantomData,
        }
    }
}

impl<P: Fp8Packaging> Module for Fp8Format<P> {
    fn name(&self) -> &'static str {
        P::NAME
    }
}

impl<P: Fp8Packaging> WeightFormat for Fp8Format<P> {
    fn check_config(&self, top: &serde_json::Value) -> Result<(), ModelError> {
        match quantization_config(top) {
            Some(q) => P::claims(q),
            None => Err(unsupported("quantization_config", "null", P::NAME)),
        }
    }

    fn configure(&self, top: &serde_json::Value) -> Result<Arc<dyn WeightFormat>, ModelError> {
        let q = quantization_config(top).expect("check_config accepted a quantization_config");
        Ok(Arc::new(Fp8Format::<P>::with(P::parse(q)?)))
    }

    fn describe(&self) -> String {
        format!("{} {:?}", P::NAME, self.layout)
    }

    fn check_tensor(&self, entry: &TensorEntry) -> Result<(), ModelError> {
        self.layout.check_tensor(entry)
    }

    fn column(&self) -> WeightFormatColumn {
        self.layout.column()
    }

    fn scheme(&self, layer: &LinearSlot) -> QuantScheme {
        self.layout.scheme(layer)
    }

    fn activation(&self) -> ActivationQuant {
        self.layout.activation()
    }

    fn for_kernels(
        &self,
        slots: &[WeightSlot],
        supports: &dyn Fn(&QGemmConfig) -> bool,
    ) -> Option<Arc<dyn WeightFormat>> {
        let fallback = self.layout.fallback_layers(slots, supports);
        if fallback.is_empty() {
            return None;
        }
        let mut layout = self.layout.clone();
        for (parts, [n, k], reason) in fallback {
            tracing::warn!(
                event = "fp8_block_decoded",
                reason,
                layers = %parts.join(","),
                n,
                k,
                "block-scaled FP8 layer decoded to BF16 at load: no selected provider runs it in FP8"
            );
            layout.decoded.extend(parts);
        }
        Some(Arc::new(Fp8Format::<P>::with(layout)))
    }

    fn slots(&self, base: &WeightSlot) -> Vec<WeightSlot> {
        self.layout.slots(base)
    }

    fn check_shard(&self, base: &WeightSlot) -> Result<(), ModelError> {
        self.layout.check_shard(base)
    }

    fn slot_dtype(&self, slot: &WeightSlot) -> DType {
        self.layout.slot_dtype(slot)
    }

    fn repacks(&self, slot: &WeightSlot) -> bool {
        self.layout.repacks(slot)
    }

    fn companions(&self, slot: &WeightSlot) -> Vec<String> {
        self.layout.companions(slot)
    }

    fn repack_with(
        &self,
        slot: &WeightSlot,
        entry: &TensorEntry,
        bytes: Vec<u8>,
        companions: &[(&TensorEntry, Vec<u8>)],
    ) -> Result<Vec<u8>, ModelError> {
        self.layout.repack_with(slot, entry, bytes, companions)
    }

    fn repack(
        &self,
        slot: &WeightSlot,
        entry: &TensorEntry,
        bytes: Vec<u8>,
    ) -> Result<Vec<u8>, ModelError> {
        self.layout.repack(slot, entry, bytes)
    }

    fn write_tiny(&self, dir: &Path, twin: Option<&Path>) -> Result<bool, ModelError> {
        write_tiny_fp8(&self.layout, &P::to_json(&self.layout), dir, twin)?;
        Ok(true)
    }
}

/// Refuses a block shape other than 128 × 128.
pub fn check_block(field: &str, block: (u32, u32)) -> Result<Fp8Weights, ModelError> {
    if block == (FP8_BLOCK, FP8_BLOCK) {
        Ok(Fp8Weights::Block {
            n: FP8_BLOCK,
            k: FP8_BLOCK,
        })
    } else {
        Err(scheme_unsupported(
            field,
            format!("[{}, {}]", block.0, block.1),
            "[128, 128]",
        ))
    }
}

// ------------------------------------------------------------------------ tiny fixtures

/// The static activation scale the tiny writer gives layer `module` (BF16-exact, different per
/// projection so a fused projection's largest-part rule is exercised).
fn tiny_input_scale(module: &str) -> f32 {
    const BY_PROJ: [(&str, f32); 7] = [
        ("q_proj", 0.031_25),
        ("k_proj", 0.062_5),
        ("v_proj", 0.093_75),
        ("o_proj", 0.046_875),
        ("gate_proj", 0.062_5),
        ("up_proj", 0.031_25),
        ("down_proj", 0.078_125),
    ];
    BY_PROJ
        .iter()
        .find(|(p, _)| module.ends_with(p))
        .map_or(0.062_5, |(_, s)| *s)
}

/// One quantized tiny weight: its e4m3 codes, its scales in the checkpoint's shape and the
/// dequantized values.
fn quantize_tiny(
    layout: &Fp8Layout,
    w: &[f32],
    n: usize,
    k: usize,
) -> (Vec<u8>, Vec<usize>, Vec<f32>, Vec<f32>) {
    let (bn, bk, scale_shape) = match layout.weights {
        Fp8Weights::Tensor => (n, k, vec![1]),
        Fp8Weights::Channel => (1, k, vec![n, 1]),
        Fp8Weights::Block { n: bn, k: bk } => (
            bn as usize,
            bk as usize,
            vec![n.div_ceil(bn as usize), k.div_ceil(bk as usize)],
        ),
    };
    let (rb, cb) = (n.div_ceil(bn), k.div_ceil(bk));
    let mut scales = vec![0f32; rb * cb];
    for (i, s) in scales.iter_mut().enumerate() {
        let (r0, c0) = ((i / cb) * bn, (i % cb) * bk);
        let mut amax = 0f32;
        for r in r0..(r0 + bn).min(n) {
            for c in c0..(c0 + bk).min(k) {
                amax = amax.max(w[r * k + c].abs());
            }
        }
        *s = pow2_at_least(amax / FP8_MAX);
    }
    let mut codes = vec![0u8; n * k];
    let mut deq = vec![0f32; n * k];
    for r in 0..n {
        for c in 0..k {
            let s = scales[(r / bn) * cb + c / bk];
            codes[r * k + c] = fp8_e4m3_round(w[r * k + c] / s);
            deq[r * k + c] = fp8_e4m3_value(codes[r * k + c]) * s;
        }
    }
    (codes, scale_shape, scales, deq)
}

/// [`WeightFormat::write_tiny`] of an FP8 layout: each BF16 weight the layout quantizes becomes
/// e4m3 codes with power-of-two F32 scales (per tensor, per row or per block) plus, for static
/// activations, an F32 `input_scale` of one value; `config.json` gains `quantization_config`.
/// The twin holds the BF16 checkpoint with each quantized weight replaced by its dequantized
/// values (exact in BF16).
fn write_tiny_fp8(
    layout: &Fp8Layout,
    config: &serde_json::Value,
    dir: &Path,
    twin: Option<&Path>,
) -> Result<(), ModelError> {
    let mut quantized: Vec<Owned> = Vec::new();
    let mut dequantized: HashMap<String, Vec<u8>> = HashMap::new();
    for (name, dtype, shape, data) in read_owned(&dir.join("model.safetensors"))? {
        if dtype != Dtype::BF16 || !layout.quantizes(&name, &shape) {
            quantized.push((name, dtype, shape, data));
            continue;
        }
        let w = bf16_values(&data);
        let (n, k) = (shape[0], shape[1]);
        let (codes, scale_shape, scales, deq) = quantize_tiny(layout, &w, n, k);
        let module = module_of(&name).expect("quantizes() checked the suffix");
        quantized.push((
            format!("{module}.{}", layout.scale_suffix),
            Dtype::F32,
            scale_shape,
            f32_bytes(&scales),
        ));
        if layout.act == ActivationQuant::Fp8PerTensorStatic {
            quantized.push((
                format!("{module}.input_scale"),
                Dtype::F32,
                vec![1],
                f32_bytes(&[tiny_input_scale(module)]),
            ));
        }
        dequantized.insert(name.clone(), bf16_bytes(&deq));
        quantized.push((name, Dtype::F8_E4M3, shape, codes));
    }
    write_fixture(dir, twin, &quantized, dequantized, config)
}
