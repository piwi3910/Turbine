//! The group-wise INT4 layout shared by the `awq`, `gptq` and `ct_pack_int4` packagings (Phase
//! 6a S-3, S-4): [`Int4Format`] over an [`Int4Packaging`], which parses and writes its
//! `quantization_config` and names its container ([`Int4Kind`]); this file holds the tensor
//! layouts of all three and the repack into the one layout the kernels consume
//! (`crates/turbine-kernels/src/quant.rs`).
//!
//! A quantized linear layer `X` (`n` output rows, `k` inputs, groups of `group` inputs) is
//! loaded, under the layer's parameter name `P` (the stacked name of a fused projection, else
//! `X.weight`), as:
//!
//! - `P`: U8 `[n, k/2]`, the 4-bit codes 0..15 of each row, the even column in the low nibble;
//! - `P_scale`: F32 `[n, k/group]`;
//! - `P_zeros`: U8 `[n, k/group]` zero points 0..15 (`Int4GroupZp`; symmetric schemes have the
//!   implicit zero point 8).
//!
//! Value = `(code − zero) × scale`. The checkpoints:
//!
//! - AutoAWQ GEMM (`awq`): `X.qweight` I32 `[k, n/8]` packed along `n` in AWQ's order (nibble
//!   `i` holds column `[0, 2, 4, 6, 1, 3, 5, 7][i]` of its eight), `X.qzeros` I32
//!   `[k/group, n/8]` packed the same way, `X.scales` F16 `[k/group, n]`.
//! - AutoGPTQ (`gptq`): `X.qweight` I32 `[k/8, n]` packed along `k` (nibble `i` = input
//!   `8r + i`), `X.qzeros` I32 `[k/group, n/8]` packed along `n` holding `zero − 1`
//!   (`checkpoint_format: gptq`; `gptq_v2` holds `zero`), `X.scales` F16 `[k/group, n]`,
//!   `X.g_idx` I32 `[k]` (must be `i / group`: act order is refused, `gptq_act_order`).
//!   Symmetric checkpoints (`sym: true`) must hold zero 8 everywhere and load without zeros.
//! - compressed-tensors `pack-quantized` (`ct_pack_int4`): `X.weight_packed` I32 `[n, k/8]`
//!   packed along `k`, codes `value + 8`; `X.weight_scale` `[n, k/group]`; `X.weight_shape`
//!   `[2]`; symmetric only.
//!
//! Tensors the packaging leaves unquantized may be F16 (AWQ and GPTQ checkpoints often are):
//! they are converted to BF16 at load.
use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;

use turbine_core::registry::Module;
use turbine_core::support::WeightFormatColumn;
use turbine_core::types::DType;

use super::common::{
    Owned, bf16_bytes, bf16_values, matches_ignore, module_of, pow2_at_least, quantization_config,
    read_owned, scheme_unsupported, write_fixture,
};
use super::{LinearSlot, QuantScheme, WeightFormat};
use crate::ModelError;
use crate::config::unsupported;
use crate::loader::{LM_HEAD, StackPlace, WeightSlot};
use crate::safetensors::{Dtype, TensorEntry};

/// The group sizes Phase 6a serves (the proof checkpoints use 128).
pub const INT4_GROUPS: [u32; 3] = [32, 64, 128];

/// AutoAWQ's order: nibble `i` of a packed word holds column `AWQ_ORDER[i]` of its eight.
const AWQ_ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
/// Sequential packing: nibble `i` holds element `i`.
const SEQ_ORDER: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

/// The container of an INT4 packaging.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Int4Kind {
    Awq,
    Gptq,
    CtPack,
}

/// One INT4 packaging's parsed configuration.
#[derive(Clone, PartialEq, Debug)]
pub struct Int4Layout {
    pub kind: Int4Kind,
    pub group: u32,
    /// Explicit zero points (`Int4GroupZp`); else symmetric (zero point 8).
    pub zero_points: bool,
    /// What GPTQ's stored zeros lack: 1 for `checkpoint_format: gptq`, 0 otherwise.
    pub zeros_offset: u8,
    /// Module names left unquantized (as [`matches_ignore`] reads them).
    pub ignore: Vec<String>,
}

/// A checkpoint container of INT4 weights: its registry name and column, how its
/// `quantization_config` is recognised and parsed, and (test support) written.
pub trait Int4Packaging: Send + Sync + 'static {
    const NAME: &'static str;
    const COLUMN: WeightFormatColumn;
    /// The layout of the registry entry (group 32, so every registered family's tiny
    /// checkpoint can be written in it; used only by the conformance suite's fixtures).
    const DEFAULT: Int4Layout;
    /// `Ok` when `q` (a non-null `quantization_config`) is this container; else the refusal.
    fn claims(q: &serde_json::Value) -> Result<(), ModelError>;
    /// The layout `q` declares; refuses a variant Phase 6a does not serve.
    fn parse(q: &serde_json::Value) -> Result<Int4Layout, ModelError>;
    /// The `quantization_config` declaring `layout` (test support).
    fn to_json(layout: &Int4Layout) -> serde_json::Value;
}

/// Refuses a group size outside [`INT4_GROUPS`].
pub fn check_group(field: &str, group: Option<u64>) -> Result<u32, ModelError> {
    match group.and_then(|g| u32::try_from(g).ok()) {
        Some(g) if INT4_GROUPS.contains(&g) => Ok(g),
        other => Err(scheme_unsupported(
            field,
            format!("{other:?}"),
            "group size 32, 64 or 128",
        )),
    }
}

/// The checkpoint tensors of a quantized layer.
struct Names {
    data: &'static str,
    scales: &'static str,
    zeros: Option<&'static str>,
    /// Tensors only validated (not stored).
    checks: &'static [&'static str],
}

impl Int4Kind {
    fn names(self) -> Names {
        match self {
            Int4Kind::Awq => Names {
                data: "qweight",
                scales: "scales",
                zeros: Some("qzeros"),
                checks: &[],
            },
            Int4Kind::Gptq => Names {
                data: "qweight",
                scales: "scales",
                zeros: Some("qzeros"),
                checks: &["g_idx"],
            },
            Int4Kind::CtPack => Names {
                data: "weight_packed",
                scales: "weight_scale",
                zeros: None,
                checks: &["weight_shape"],
            },
        }
    }
}

/// What a slot of an INT4 checkpoint is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Part {
    Data,
    Scales,
    Zeros,
    /// Validated only (GPTQ `g_idx`, symmetric GPTQ `qzeros`, compressed-tensors
    /// `weight_shape`).
    Check,
    /// A tensor the packaging leaves unquantized (BF16, or F16 converted to BF16).
    Plain,
}

impl Int4Layout {
    /// Whether the linear layer with parameter name `name` is quantized: a 2-D `….weight`
    /// other than the token embedding and `lm_head`, not in `ignore`.
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
        if self.quantizes(&l.name, &[l.n as usize, l.k as usize]) {
            QuantScheme::Int4Group {
                group: self.group,
                zero_points: self.zero_points,
            }
        } else {
            QuantScheme::Bf16
        }
    }

    /// What slot `name` is (by its checkpoint suffix).
    fn part(&self, name: &str) -> Part {
        let names = self.kind.names();
        let suffix = name.rsplit('.').next().unwrap_or("");
        if !name.contains('.') {
            Part::Plain
        } else if suffix == names.data {
            Part::Data
        } else if suffix == names.scales {
            Part::Scales
        } else if Some(suffix) == names.zeros {
            if self.zero_points {
                Part::Zeros
            } else {
                Part::Check
            }
        } else if names.checks.contains(&suffix) {
            Part::Check
        } else {
            Part::Plain
        }
    }

    /// The slots of family slot `base`: for a quantized layer its data, scales, zero points
    /// and check-only tensors, stacked like `base`; else `base` (converted from F16 when so
    /// stored).
    pub fn slots(&self, base: &WeightSlot) -> Vec<WeightSlot> {
        if !self.quantizes(&base.name, &base.shape) {
            return vec![base.clone()];
        }
        let module = module_of(&base.name).expect("quantizes() checked the suffix");
        let names = self.kind.names();
        let (n, k) = (base.shape[0], base.shape[1]);
        let groups = k.div_ceil(self.group as usize);
        // The stack's leading dimensions and rows, and `base`'s first row among them.
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
        let per_row = |suffix: &str, stack: String, cols: usize| {
            let mut shape = lead.clone();
            shape.extend([rows, cols]);
            WeightSlot {
                name: format!("{module}.{suffix}"),
                shape: vec![n, cols],
                stack: Some(StackPlace {
                    name: stack,
                    shape,
                    offset: row0 * cols,
                }),
                source: None,
            }
        };
        let mut out = vec![
            per_row(names.data, key.clone(), k / 2),
            per_row(names.scales, format!("{key}_scale"), groups),
        ];
        if let Some(zeros) = names.zeros {
            out.push(if self.zero_points {
                per_row(zeros, format!("{key}_zeros"), groups)
            } else {
                check_slot(module, zeros)
            });
        }
        out.extend(names.checks.iter().map(|c| check_slot(module, c)));
        out
    }

    pub fn slot_dtype(&self, slot: &WeightSlot) -> DType {
        match self.part(&slot.name) {
            Part::Data | Part::Zeros | Part::Check => DType::U8,
            Part::Scales => DType::F32,
            Part::Plain => super::Bf16::DTYPE,
        }
    }

    pub fn check_tensor(&self, entry: &TensorEntry) -> Result<(), ModelError> {
        let (ok, supported) = match self.part(&entry.name) {
            Part::Data | Part::Zeros => (entry.dtype == Dtype::I32, "I32"),
            Part::Scales => (
                matches!(entry.dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32),
                "F16, BF16 or F32",
            ),
            Part::Check => (matches!(entry.dtype, Dtype::I32 | Dtype::I64), "I32 or I64"),
            Part::Plain if self.kind == Int4Kind::CtPack => (entry.dtype == Dtype::BF16, "BF16"),
            Part::Plain => (
                matches!(entry.dtype, Dtype::BF16 | Dtype::F16),
                "BF16 or F16",
            ),
        };
        if !ok {
            return Err(ModelError::Unsupported {
                field: "tensor dtype".to_string(),
                value: format!("{} ({})", entry.dtype, entry.name),
                supported: supported.to_string(),
            });
        }
        if self.part(&entry.name) == Part::Data {
            let (n, k) = self.data_dims(&entry.shape).ok_or_else(|| {
                unsupported(
                    "tensor shape",
                    format!("{:?} ({})", entry.shape, entry.name),
                    "a 2-D packed INT4 tensor",
                )
            })?;
            if !k.is_multiple_of(self.group as usize) || !n.is_multiple_of(8) {
                return Err(scheme_unsupported(
                    "weight shape",
                    format!("[{n}, {k}] ({})", entry.name),
                    &format!("n a multiple of 8, k of the group size {}", self.group),
                ));
            }
        }
        Ok(())
    }

    /// `(n, k)` of a packed data tensor's shape.
    fn data_dims(&self, shape: &[usize]) -> Option<(usize, usize)> {
        let &[a, b] = shape else { return None };
        Some(match self.kind {
            Int4Kind::Awq => (b * 8, a),
            Int4Kind::Gptq => (b, a * 8),
            Int4Kind::CtPack => (a, b * 8),
        })
    }

    /// Every quantized slot goes through [`Int4Layout::repack`], and so do the unquantized ones
    /// of AWQ and GPTQ (converted from F16 when so stored); compressed-tensors checkpoints are
    /// BF16 and copy those as stored.
    pub fn repacks(&self, slot: &WeightSlot) -> bool {
        self.part(&slot.name) != Part::Plain || self.kind != Int4Kind::CtPack
    }

    pub fn repack(
        &self,
        slot: &WeightSlot,
        entry: &TensorEntry,
        bytes: Vec<u8>,
    ) -> Result<Vec<u8>, ModelError> {
        let bad = |rule: String| ModelError::Safetensors {
            file: entry.file.clone(),
            tensor: entry.name.clone(),
            rule,
        };
        let words = || -> Vec<u32> {
            bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };
        match self.part(&slot.name) {
            Part::Plain => {
                if entry.shape != slot.shape {
                    return Err(bad(format!("shape {:?} != {:?}", entry.shape, slot.shape)));
                }
                Ok(match entry.dtype {
                    Dtype::F16 => bf16_bytes(
                        &bytes
                            .chunks_exact(2)
                            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                            .collect::<Vec<_>>(),
                    ),
                    _ => bytes,
                })
            }
            Part::Data => {
                let (n, k) = (slot.shape[0], slot.shape[1] * 2);
                if self.data_dims(&entry.shape) != Some((n, k)) {
                    return Err(bad(format!(
                        "shape {:?} does not pack [{n}, {k}]",
                        entry.shape
                    )));
                }
                let codes = match self.kind {
                    Int4Kind::Awq => transpose(&unpack_cols(&words(), k, n / 8, AWQ_ORDER), k, n),
                    Int4Kind::Gptq => transpose(&unpack_rows(&words(), k / 8, n), k, n),
                    Int4Kind::CtPack => unpack_cols(&words(), n, k / 8, SEQ_ORDER),
                };
                Ok(codes.chunks_exact(2).map(|p| p[0] | (p[1] << 4)).collect())
            }
            Part::Scales => {
                let (n, groups) = (slot.shape[0], slot.shape[1]);
                let values = float_values(entry, &bytes)?;
                let transposed = self.kind != Int4Kind::CtPack;
                let want = if transposed { [groups, n] } else { [n, groups] };
                if entry.shape != want {
                    return Err(bad(format!("shape {:?} != {want:?}", entry.shape)));
                }
                let values = if transposed {
                    transpose(&values, groups, n)
                } else {
                    values
                };
                Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect())
            }
            Part::Zeros => {
                let (n, groups) = (slot.shape[0], slot.shape[1]);
                if entry.shape != [groups, n / 8] {
                    return Err(bad(format!(
                        "shape {:?} != [{groups}, {}]",
                        entry.shape,
                        n / 8
                    )));
                }
                let zeros = self.zeros_of(&words(), groups, n);
                if let Some(z) = zeros.iter().find(|&&z| z > 15) {
                    return Err(bad(format!("zero point {z} outside 0..=15")));
                }
                Ok(transpose(&zeros, groups, n))
            }
            Part::Check => self.check(entry, &bytes).map(|()| Vec::new()),
        }
    }

    /// The zero points `[groups, n]` of a packed `qzeros` tensor, offset restored.
    fn zeros_of(&self, words: &[u32], groups: usize, n: usize) -> Vec<u8> {
        let order = if self.kind == Int4Kind::Awq {
            AWQ_ORDER
        } else {
            SEQ_ORDER
        };
        unpack_cols(words, groups, n / 8, order)
            .into_iter()
            .map(|z| z + self.zeros_offset)
            .collect()
    }

    /// Validates a check-only tensor.
    fn check(&self, entry: &TensorEntry, bytes: &[u8]) -> Result<(), ModelError> {
        let ints: Vec<i64> = match entry.dtype {
            Dtype::I64 => bytes
                .chunks_exact(8)
                .map(|b| i64::from_le_bytes(b.try_into().expect("8 bytes")))
                .collect(),
            _ => bytes
                .chunks_exact(4)
                .map(|b| i64::from(i32::from_le_bytes(b.try_into().expect("4 bytes"))))
                .collect(),
        };
        let suffix = entry.name.rsplit('.').next().unwrap_or("");
        match suffix {
            "g_idx" => {
                let g = i64::from(self.group);
                if let Some((i, v)) = (0i64..).zip(&ints).find(|(i, v)| **v != i / g) {
                    return Err(unsupported(
                        "g_idx",
                        format!("{} [{i}] = {v}", entry.name),
                        &format!("gptq_act_order: input i in group i / {g} (no act order)"),
                    ));
                }
                Ok(())
            }
            "qzeros" => {
                // Symmetric GPTQ: every zero point must be the implicit 8.
                let words: Vec<u32> = ints.iter().map(|&w| w as u32).collect();
                let zeros = self.zeros_of(&words, words.len(), 8);
                match zeros.iter().find(|&&z| z != 8) {
                    None => Ok(()),
                    Some(z) => Err(scheme_unsupported(
                        "sym",
                        format!("true with zero point {z} ({})", entry.name),
                        "zero point 8 for symmetric weights",
                    )),
                }
            }
            "weight_shape" if ints.len() == 2 && ints.iter().all(|&d| d > 0) => Ok(()),
            _ => Err(ModelError::Safetensors {
                file: entry.file.clone(),
                tensor: entry.name.clone(),
                rule: format!("unexpected values {ints:?}"),
            }),
        }
    }
}

/// A slot only validated: no elements, not stored.
fn check_slot(module: &str, suffix: &str) -> WeightSlot {
    WeightSlot {
        name: format!("{module}.{suffix}"),
        shape: vec![0],
        stack: None,
        source: None,
    }
}

/// F16 / BF16 / F32 values as F32.
fn float_values(entry: &TensorEntry, bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    Ok(match entry.dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        Dtype::BF16 => bf16_values(bytes),
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

/// `rows × (words_per_row · 8)` codes of words packed along the columns: nibble `i` of word
/// `w` holds column `8w + order[i]`.
fn unpack_cols(words: &[u32], rows: usize, words_per_row: usize, order: [usize; 8]) -> Vec<u8> {
    let cols = words_per_row * 8;
    let mut out = vec![0u8; rows * cols];
    for r in 0..rows {
        for w in 0..words_per_row {
            let word = words[r * words_per_row + w];
            for (i, &c) in order.iter().enumerate() {
                out[r * cols + 8 * w + c] = ((word >> (4 * i)) & 0xf) as u8;
            }
        }
    }
    out
}

/// `(word_rows · 8) × cols` codes of words packed along the rows: nibble `i` of word
/// `(r, c)` holds row `8r + i`.
fn unpack_rows(words: &[u32], word_rows: usize, cols: usize) -> Vec<u8> {
    let mut out = vec![0u8; word_rows * 8 * cols];
    for r in 0..word_rows {
        for c in 0..cols {
            let word = words[r * cols + c];
            for i in 0..8 {
                out[(8 * r + i) * cols + c] = ((word >> (4 * i)) & 0xf) as u8;
            }
        }
    }
    out
}

/// The words of `rows × cols` codes packed along the columns in `order` (inverse of
/// [`unpack_cols`]).
fn pack_cols(codes: &[u8], rows: usize, cols: usize, order: [usize; 8]) -> Vec<u32> {
    let mut out = vec![0u32; rows * cols / 8];
    for r in 0..rows {
        for w in 0..cols / 8 {
            for (i, &c) in order.iter().enumerate() {
                out[r * cols / 8 + w] |= u32::from(codes[r * cols + 8 * w + c]) << (4 * i);
            }
        }
    }
    out
}

/// The words of `rows × cols` codes packed along the rows (inverse of [`unpack_rows`]).
fn pack_rows(codes: &[u8], rows: usize, cols: usize) -> Vec<u32> {
    let mut out = vec![0u32; rows / 8 * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[(r / 8) * cols + c] |= u32::from(codes[r * cols + c]) << (4 * (r % 8));
        }
    }
    out
}

/// The `cols × rows` transpose of a row-major `rows × cols` matrix.
fn transpose<T: Copy + Default>(m: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut out = vec![T::default(); m.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = m[r * cols + c];
        }
    }
    out
}

/// A weight format over the INT4 layout of packaging `P`.
pub struct Int4Format<P> {
    pub layout: Int4Layout,
    packaging: PhantomData<fn() -> P>,
}

impl<P: Int4Packaging> Int4Format<P> {
    /// The registry entry.
    pub const ENTRY: Int4Format<P> = Int4Format::with(P::DEFAULT);

    pub const fn with(layout: Int4Layout) -> Int4Format<P> {
        Int4Format {
            layout,
            packaging: PhantomData,
        }
    }
}

impl<P: Int4Packaging> Module for Int4Format<P> {
    fn name(&self) -> &'static str {
        P::NAME
    }
}

impl<P: Int4Packaging> WeightFormat for Int4Format<P> {
    fn check_config(&self, top: &serde_json::Value) -> Result<(), ModelError> {
        match quantization_config(top) {
            Some(q) => P::claims(q),
            None => Err(unsupported("quantization_config", "null", P::NAME)),
        }
    }

    fn configure(&self, top: &serde_json::Value) -> Result<Arc<dyn WeightFormat>, ModelError> {
        let q = quantization_config(top).expect("check_config accepted a quantization_config");
        Ok(Arc::new(Int4Format::<P>::with(P::parse(q)?)))
    }

    fn describe(&self) -> String {
        format!("{} {:?}", P::NAME, self.layout)
    }

    fn check_tensor(&self, entry: &TensorEntry) -> Result<(), ModelError> {
        self.layout.check_tensor(entry)
    }

    fn column(&self) -> WeightFormatColumn {
        P::COLUMN
    }

    fn scheme(&self, layer: &LinearSlot) -> QuantScheme {
        self.layout.scheme(layer)
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
        write_tiny_int4(&self.layout, &P::to_json(&self.layout), dir, twin)?;
        Ok(true)
    }
}

// ------------------------------------------------------------------------ tiny fixtures

/// One quantized tiny weight `[n, k]`: codes `[n, k]`, zero points and scales `[n, groups]`
/// (power-of-two scales, so the dequantized values are exact in BF16) and the dequantized
/// values.
fn quantize_tiny(
    layout: &Int4Layout,
    w: &[f32],
    n: usize,
    k: usize,
) -> (Vec<u8>, Vec<u8>, Vec<f32>, Vec<f32>) {
    let g = layout.group as usize;
    let groups = k / g;
    let (mut codes, mut zeros, mut scales) = (
        vec![0u8; n * k],
        vec![8u8; n * groups],
        vec![0f32; n * groups],
    );
    for r in 0..n {
        for gi in 0..groups {
            let vals = &w[r * k + gi * g..r * k + (gi + 1) * g];
            let (lo, hi) = vals
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            let (s, z) = if layout.zero_points {
                let s = pow2_at_least((hi - lo) / 15.0);
                (s, (-lo / s).round().clamp(0.0, 15.0) as u8)
            } else {
                (pow2_at_least(lo.abs().max(hi.abs()) / 7.0), 8)
            };
            scales[r * groups + gi] = s;
            zeros[r * groups + gi] = z;
            for (c, v) in vals.iter().enumerate() {
                codes[r * k + gi * g + c] = ((v / s).round() + f32::from(z)).clamp(0.0, 15.0) as u8;
            }
        }
    }
    let deq = (0..n * k)
        .map(|i| {
            let gi = (i / k) * groups + (i % k) / g;
            (f32::from(codes[i]) - f32::from(zeros[gi])) * scales[gi]
        })
        .collect();
    (codes, zeros, scales, deq)
}

fn i32_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect()
}

/// [`WeightFormat::write_tiny`] of an INT4 layout: each BF16 weight the layout quantizes
/// becomes the packaging's tensors (AWQ and GPTQ with F16 scales, and every tensor they leave
/// unquantized stored as F16, as those checkpoints are; compressed-tensors with BF16 scales);
/// `config.json` gains `quantization_config`. The twin holds the model as loaded: dequantized
/// weights, and F16-stored tensors as their BF16 conversion.
fn write_tiny_int4(
    layout: &Int4Layout,
    config: &serde_json::Value,
    dir: &Path,
    twin: Option<&Path>,
) -> Result<(), ModelError> {
    let names = layout.kind.names();
    let f16_checkpoint = layout.kind != Int4Kind::CtPack;
    let mut out: Vec<Owned> = Vec::new();
    let mut loaded: HashMap<String, Vec<u8>> = HashMap::new();
    for (name, dtype, shape, data) in read_owned(&dir.join("model.safetensors"))? {
        if dtype != Dtype::BF16 {
            out.push((name, dtype, shape, data));
            continue;
        }
        let w = bf16_values(&data);
        if !layout.quantizes(&name, &shape) {
            if f16_checkpoint {
                let f16 = f16_bytes(&w);
                let back: Vec<f32> = f16
                    .chunks_exact(2)
                    .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect();
                loaded.insert(name.clone(), bf16_bytes(&back));
                out.push((name, Dtype::F16, shape, f16));
            } else {
                out.push((name, dtype, shape, data));
            }
            continue;
        }
        let (n, k) = (shape[0], shape[1]);
        let groups = k / layout.group as usize;
        let (codes, zeros, scales, deq) = quantize_tiny(layout, &w, n, k);
        let module = module_of(&name).expect("quantizes() checked the suffix");
        let t = |suffix: &str, dtype: Dtype, shape: Vec<usize>, bytes: Vec<u8>| -> Owned {
            (format!("{module}.{suffix}"), dtype, shape, bytes)
        };
        let stored_zeros: Vec<u8> = zeros.iter().map(|z| z - layout.zeros_offset).collect();
        match layout.kind {
            Int4Kind::Awq => {
                let packed = pack_cols(&transpose(&codes, n, k), k, n, AWQ_ORDER);
                out.push(t(
                    names.data,
                    Dtype::I32,
                    vec![k, n / 8],
                    i32_bytes(&packed),
                ));
                let z = pack_cols(&transpose(&stored_zeros, n, groups), groups, n, AWQ_ORDER);
                out.push(t("qzeros", Dtype::I32, vec![groups, n / 8], i32_bytes(&z)));
                let s = transpose(&scales, n, groups);
                out.push(t(names.scales, Dtype::F16, vec![groups, n], f16_bytes(&s)));
            }
            Int4Kind::Gptq => {
                let packed = pack_rows(&transpose(&codes, n, k), k, n);
                out.push(t(
                    names.data,
                    Dtype::I32,
                    vec![k / 8, n],
                    i32_bytes(&packed),
                ));
                let z = pack_cols(&transpose(&stored_zeros, n, groups), groups, n, SEQ_ORDER);
                out.push(t("qzeros", Dtype::I32, vec![groups, n / 8], i32_bytes(&z)));
                let s = transpose(&scales, n, groups);
                out.push(t(names.scales, Dtype::F16, vec![groups, n], f16_bytes(&s)));
                let g_idx: Vec<u32> = (0..k).map(|i| (i / layout.group as usize) as u32).collect();
                out.push(t("g_idx", Dtype::I32, vec![k], i32_bytes(&g_idx)));
            }
            Int4Kind::CtPack => {
                let packed = pack_cols(&codes, n, k, SEQ_ORDER);
                out.push(t(
                    names.data,
                    Dtype::I32,
                    vec![n, k / 8],
                    i32_bytes(&packed),
                ));
                out.push(t(
                    names.scales,
                    Dtype::BF16,
                    vec![n, groups],
                    bf16_bytes(&scales),
                ));
                let dims: Vec<u8> = [n as i64, k as i64]
                    .iter()
                    .flat_map(|d| d.to_le_bytes())
                    .collect();
                out.push(t("weight_shape", Dtype::I64, vec![2], dims));
            }
        }
        loaded.insert(name, bf16_bytes(&deq));
    }
    write_fixture(dir, twin, &out, loaded, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A layout of `kind` with groups of 8 (hand-built cases only).
    fn layout(kind: Int4Kind, zero_points: bool, zeros_offset: u8) -> Int4Layout {
        Int4Layout {
            kind,
            group: 8,
            zero_points,
            zeros_offset,
            ignore: Vec::new(),
        }
    }

    fn entry(name: &str, dtype: Dtype, shape: Vec<usize>) -> TensorEntry {
        TensorEntry {
            name: name.to_string(),
            dtype,
            shape,
            file: "model.safetensors".into(),
            range: 0..0,
        }
    }

    fn slot(name: &str, shape: Vec<usize>) -> WeightSlot {
        WeightSlot {
            name: name.to_string(),
            shape,
            stack: None,
            source: None,
        }
    }

    /// The 8 × 8 codes `[n][k]` of the hand-built cases: code(r, c) = (r + 3c) mod 16.
    fn codes() -> Vec<u8> {
        (0..64)
            .map(|i| ((i / 8 + 3 * (i % 8)) % 16) as u8)
            .collect()
    }

    /// The loaded bytes `[n, k/2]` of [`codes`]: even column low.
    fn loaded() -> Vec<u8> {
        codes()
            .chunks_exact(2)
            .map(|p| p[0] | (p[1] << 4))
            .collect()
    }

    fn words(ws: &[u32]) -> Vec<u8> {
        i32_bytes(ws)
    }

    /// AWQ: `qweight [k=8, n/8=1]`, word of input row c holds output column `AWQ_ORDER[i]` in
    /// nibble `i`; `qzeros [1, 1]` likewise. Breaks with the identity order.
    #[test]
    fn awq_repack_8x8() {
        // AutoAWQ's order, spelled out (not the constant under test).
        const ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
        let l = layout(Int4Kind::Awq, true, 0);
        let c = codes();
        // Word for input column kc: nibble i = code(n = ORDER[i], kc).
        let qweight: Vec<u32> = (0..8)
            .map(|kc| {
                (0..8).fold(0u32, |w, i| {
                    w | (u32::from(c[ORDER[i] * 8 + kc]) << (4 * i))
                })
            })
            .collect();
        let data = l
            .repack(
                &slot("m.qweight", vec![8, 4]),
                &entry("m.qweight", Dtype::I32, vec![8, 1]),
                words(&qweight),
            )
            .unwrap();
        assert_eq!(data, loaded());
        // Zero points per output row n: z(n) = n + 1, packed like the weights.
        let qzeros = (0..8).fold(0u32, |w, i| w | ((ORDER[i] as u32 + 1) << (4 * i)));
        let zeros = l
            .repack(
                &slot("m.qzeros", vec![8, 1]),
                &entry("m.qzeros", Dtype::I32, vec![1, 1]),
                words(&[qzeros]),
            )
            .unwrap();
        assert_eq!(zeros, (1..=8).collect::<Vec<u8>>());
    }

    /// GPTQ: `qweight [k/8=1, n=8]`, word of output column n holds input `i` in nibble `i`;
    /// `qzeros` sequential along n holding `zero − 1`; `g_idx` must be `i / group`.
    #[test]
    fn gptq_repack_8x8() {
        let l = layout(Int4Kind::Gptq, true, 1);
        let c = codes();
        let qweight: Vec<u32> = (0..8)
            .map(|n| (0..8).fold(0u32, |w, i| w | (u32::from(c[n * 8 + i]) << (4 * i))))
            .collect();
        let data = l
            .repack(
                &slot("m.qweight", vec![8, 4]),
                &entry("m.qweight", Dtype::I32, vec![1, 8]),
                words(&qweight),
            )
            .unwrap();
        assert_eq!(data, loaded());
        let qzeros = (0..8u32).fold(0u32, |w, i| w | (i << (4 * i)));
        let zeros = l
            .repack(
                &slot("m.qzeros", vec![8, 1]),
                &entry("m.qzeros", Dtype::I32, vec![1, 1]),
                words(&[qzeros]),
            )
            .unwrap();
        assert_eq!(zeros, (1..=8).collect::<Vec<u8>>());
        // g_idx: identity passes, act order refused.
        let g = |v: &[u32]| {
            l.repack(
                &slot("m.g_idx", vec![0]),
                &entry("m.g_idx", Dtype::I32, vec![v.len()]),
                words(v),
            )
        };
        assert_eq!(g(&[0; 8]).unwrap(), Vec::<u8>::new());
        let err = g(&[0, 0, 0, 0, 0, 0, 0, 1]).unwrap_err().to_string();
        assert!(err.contains("gptq_act_order"), "{err}");
        // Symmetric: every stored zero must restore to 8.
        let sym = layout(Int4Kind::Gptq, false, 1);
        let q = |z: u32| {
            sym.repack(
                &slot("m.qzeros", vec![0]),
                &entry("m.qzeros", Dtype::I32, vec![1, 1]),
                words(&[(0..8).fold(0u32, |w, i| w | (z << (4 * i)))]),
            )
        };
        assert!(q(7).is_ok());
        let err = q(6).unwrap_err().to_string();
        assert!(err.contains("quant_scheme_unsupported"), "{err}");
    }

    /// compressed-tensors: `weight_packed [n=8, k/8=1]`, nibble `i` = input `i`, codes as
    /// stored (value + 8); scales `[n, groups]` untransposed.
    #[test]
    fn ct_pack_repack_8x8() {
        let l = layout(Int4Kind::CtPack, false, 0);
        let c = codes();
        let packed: Vec<u32> = (0..8)
            .map(|n| (0..8).fold(0u32, |w, i| w | (u32::from(c[n * 8 + i]) << (4 * i))))
            .collect();
        let data = l
            .repack(
                &slot("m.weight_packed", vec![8, 4]),
                &entry("m.weight_packed", Dtype::I32, vec![8, 1]),
                words(&packed),
            )
            .unwrap();
        assert_eq!(data, loaded());
        let scales: Vec<f32> = (0..8).map(|n| 0.5 * (n + 1) as f32).collect();
        let s = l
            .repack(
                &slot("m.weight_scale", vec![8, 1]),
                &entry("m.weight_scale", Dtype::BF16, vec![8, 1]),
                bf16_bytes(&scales),
            )
            .unwrap();
        assert_eq!(
            s,
            scales
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>()
        );
    }

    /// The fixture packers invert the unpackers.
    #[test]
    fn pack_round_trips() {
        let c: Vec<u8> = (0..16 * 24).map(|i| (i * 7 % 16) as u8).collect();
        for order in [AWQ_ORDER, SEQ_ORDER] {
            assert_eq!(unpack_cols(&pack_cols(&c, 16, 24, order), 16, 3, order), c);
        }
        assert_eq!(unpack_rows(&pack_rows(&c, 16, 24), 2, 24), c);
    }
}
