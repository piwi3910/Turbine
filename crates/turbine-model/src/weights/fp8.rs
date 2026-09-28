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
use turbine_kernels::cpu::quant::{fp8_e4m3_round, fp8_e4m3_value};

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

/// `X` of a parameter named `X.weight`.
fn module_of(name: &str) -> Option<&str> {
    name.strip_suffix(".weight")
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

    pub fn scheme(&self, l: &LinearSlot) -> QuantScheme {
        if !self.quantizes(&l.name, &[l.n as usize, l.k as usize]) {
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
        let mut out = vec![base.clone()];
        if !self.quantizes(&base.name, &base.shape) {
            return out;
        }
        let module = module_of(&base.name).expect("quantizes() checked the suffix");
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
        out
    }

    pub fn slot_dtype(&self, slot: &WeightSlot) -> DType {
        if self.repacks(slot) {
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

    /// The scale slots are converted to F32 (and expanded per row).
    pub fn repacks(&self, slot: &WeightSlot) -> bool {
        self.is_scale(&slot.name)
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

/// The non-null `quantization_config` of `top`.
pub fn quantization_config(top: &serde_json::Value) -> Option<&serde_json::Value> {
    top.get("quantization_config").filter(|q| !q.is_null())
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
        self.layout.act
    }

    fn slots(&self, base: &WeightSlot) -> Vec<WeightSlot> {
        self.layout.slots(base)
    }

    fn slot_dtype(&self, slot: &WeightSlot) -> DType {
        self.layout.slot_dtype(slot)
    }

    fn repacks(&self, slot: &WeightSlot) -> bool {
        self.layout.repacks(slot)
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

/// compressed-tensors / HF `ignore` entries: an exact module name, a `re:` regular expression
/// over the module name (Python `re.match`: anchored at the start, at the end only with `$`),
/// or a bare final component (`lm_head`) matching any module ending in it.
pub fn matches_ignore(pattern: &str, module: &str) -> bool {
    if let Some(re) = pattern.strip_prefix("re:") {
        return regex_lite_match(re, module);
    }
    module == pattern || module.ends_with(&format!(".{pattern}"))
}

/// The regular expressions compressed-tensors ignore lists use in practice: `.*` wildcards
/// around literal text (`\.` a literal dot, a bare `.` any one character), matched from the
/// start, to the end only when it ends in `$`.
fn regex_lite_match(re: &str, text: &str) -> bool {
    let (re, anchored) = match re.strip_suffix('$') {
        Some(r) => (r, true),
        None => (re, false),
    };
    // Tokens: literal characters (`None` for "any one character") between `.*` wildcards.
    let parts: Vec<Vec<Option<char>>> = re
        .split(".*")
        .map(|p| {
            let mut out = Vec::new();
            let mut chars = p.chars();
            while let Some(c) = chars.next() {
                match c {
                    '\\' => out.push(chars.next()),
                    '.' => out.push(None),
                    c => out.push(Some(c)),
                }
            }
            out
        })
        .collect();
    let text: Vec<char> = text.chars().collect();
    let at = |pos: usize, part: &[Option<char>]| {
        pos + part.len() <= text.len()
            && part
                .iter()
                .zip(&text[pos..])
                .all(|(p, t)| p.is_none_or(|c| c == *t))
    };
    let last = parts.len() - 1;
    if !at(0, &parts[0]) {
        return false;
    }
    let mut pos = parts[0].len();
    if last == 0 {
        return !anchored || pos == text.len();
    }
    for part in &parts[1..last] {
        match (pos..=text.len()).find(|&p| at(p, part)) {
            Some(p) => pos = p + part.len(),
            None => return false,
        }
    }
    let tail = &parts[last];
    if anchored {
        text.len() >= pos + tail.len() && at(text.len() - tail.len(), tail)
    } else {
        (pos..=text.len()).any(|p| at(p, tail))
    }
}

/// A JSON value's `[a, b]` array of non-negative integers.
pub fn pair(v: &serde_json::Value) -> Option<(u32, u32)> {
    let a = v.as_array()?;
    if a.len() != 2 {
        return None;
    }
    Some((
        u32::try_from(a[0].as_u64()?).ok()?,
        u32::try_from(a[1].as_u64()?).ok()?,
    ))
}

/// The strings of the list `obj[key]` (non-strings skipped; absent or null: empty).
pub fn string_list(obj: &serde_json::Value, key: &str) -> Vec<String> {
    obj.get(key)
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
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

/// The refusal of a declared variant outside Phase 6a (spec S-3, reason
/// `quant_scheme_unsupported`).
pub fn scheme_unsupported(field: &str, value: impl Into<String>, supported: &str) -> ModelError {
    unsupported(
        field,
        value,
        &format!("quant_scheme_unsupported: {supported}"),
    )
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

/// The smallest power of two `s` with `amax / s ≤ 448` (1 for an all-zero group): e4m3 codes
/// times `s` are exact in BF16.
fn pow2_scale(amax: f32) -> f32 {
    if amax == 0.0 {
        1.0
    } else {
        (amax / FP8_MAX).log2().ceil().exp2()
    }
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
        *s = pow2_scale(amax);
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

type Owned = (String, Dtype, Vec<usize>, Vec<u8>);

fn io_error(path: &Path, detail: impl ToString) -> ModelError {
    ModelError::Io {
        path: path.to_path_buf(),
        detail: detail.to_string(),
    }
}

fn read_owned(path: &Path) -> Result<Vec<Owned>, ModelError> {
    let bytes = std::fs::read(path).map_err(|e| io_error(path, e))?;
    let st = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| io_error(path, e))?;
    let mut out: Vec<Owned> = st
        .tensors()
        .into_iter()
        .map(|(n, t)| (n, t.dtype(), t.shape().to_vec(), t.data().to_vec()))
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

fn write_owned(path: &Path, tensors: &[Owned]) -> Result<(), ModelError> {
    let views = tensors
        .iter()
        .map(|(n, d, s, b)| {
            safetensors::tensor::TensorView::new(*d, s.clone(), b)
                .map(|v| (n.clone(), v))
                .map_err(|e| io_error(path, e))
        })
        .collect::<Result<Vec<_>, _>>()?;
    safetensors::serialize_to_file(views, None, path).map_err(|e| io_error(path, e))
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
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
    let path = dir.join("model.safetensors");
    let tensors = read_owned(&path)?;
    let mut quantized: Vec<Owned> = Vec::new();
    let mut dequantized: HashMap<String, Vec<u8>> = HashMap::new();
    for (name, dtype, shape, data) in tensors {
        if dtype != Dtype::BF16 || !layout.quantizes(&name, &shape) {
            quantized.push((name, dtype, shape, data));
            continue;
        }
        let w: Vec<f32> = data
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect();
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
        dequantized.insert(
            name.clone(),
            deq.iter()
                .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                .collect(),
        );
        quantized.push((name, Dtype::F8_E4M3, shape, codes));
    }
    if let Some(twin) = twin {
        super::copy_checkpoint(dir, twin)?;
        let twin_path = twin.join("model.safetensors");
        let mut bf16 = read_owned(&twin_path)?;
        for (name, _, _, data) in &mut bf16 {
            if let Some(d) = dequantized.remove(name.as_str()) {
                *data = d;
            }
        }
        write_owned(&twin_path, &bf16)?;
    }
    write_owned(&path, &quantized)?;
    let config_path = dir.join("config.json");
    let text = std::fs::read(&config_path).map_err(|e| io_error(&config_path, e))?;
    let mut top: serde_json::Value =
        serde_json::from_slice(&text).map_err(|e| io_error(&config_path, e))?;
    top["quantization_config"] = config.clone();
    let text = serde_json::to_string_pretty(&top).expect("serialize JSON");
    std::fs::write(&config_path, text).map_err(|e| io_error(&config_path, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ignore_patterns() {
        assert!(matches_ignore("lm_head", "lm_head"));
        assert!(matches_ignore("lm_head", "model.lm_head"));
        assert!(!matches_ignore("lm_head", "model.layers.0.mlp.down_proj"));
        assert!(matches_ignore("re:.*lm_head", "lm_head"));
        assert!(matches_ignore(
            "re:model.layers.0.*",
            "model.layers.0.self_attn.q_proj"
        ));
        assert!(!matches_ignore(
            "re:model.layers.1.*",
            "model.layers.0.self_attn.q_proj"
        ));
        // `re.match` anchors at the start only; `$` anchors the end.
        assert!(matches_ignore(
            r"re:.*mlp\.gate$",
            "model.layers.3.mlp.gate"
        ));
        assert!(!matches_ignore(
            r"re:.*mlp\.gate$",
            "model.layers.3.mlp.gate_proj"
        ));
        assert!(matches_ignore(
            r"re:.*mlp\.gate",
            "model.layers.3.mlp.gate_proj"
        ));
        assert!(!matches_ignore(r"re:mlp\.gate", "model.layers.3.mlp.gate"));
        // A bare `.` is any character.
        assert!(matches_ignore(
            "re:model.layers.1.self_attn",
            "model_layers_1_self_attn"
        ));
        assert!(!matches_ignore(r"re:model\.layers", "model_layers"));
    }
}
