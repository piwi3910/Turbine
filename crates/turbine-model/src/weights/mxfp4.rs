//! The OCP MXFP4 layout shared by the `ct_mxfp4`, `quark_mxfp4` and `openai_mxfp4` packagings
//! (Phase 6a S-3, S-4, S-10): [`Mxfp4Format`] over an [`Mxfp4Packaging`], which parses and
//! writes its `quantization_config` and names its container ([`Mxfp4Kind`]).
//!
//! A quantized linear layer `X` (`n` output rows, `k` inputs, `k` a multiple of 32) is loaded,
//! under the layer's parameter name `P` (the stacked name of a fused projection, else
//! `X.weight`), as the layout the kernels consume (`crates/turbine-kernels/src/quant.rs`):
//!
//! - `P`: U8 `[n, k/2]`, the E2M1 codes of each row, the even column in the low nibble;
//! - `P_scale`: U8 `[n, k/32]`, one E8M0 exponent per 32 inputs.
//!
//! Value = `e2m1(code) × 2^(e − 127)`. The checkpoints already store that layout:
//! compressed-tensors `mxfp4-pack-quantized` as `X.weight_packed` / `X.weight_scale`, AMD Quark
//! `fp4` as `X.weight` / `X.weight_scale` (its `pack_method: reorder` does not reorder fp4), and
//! OpenAI's `quant_method: mxfp4` as `X.weight_blocks` `[n, k/32, 16]` / `X.weight_scales`
//! `[n, k/32]` (the same bytes, grouped by block).
use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;

use turbine_core::registry::Module;
use turbine_core::support::WeightFormatColumn;
use turbine_core::types::DType;
use turbine_kernels::cpu::quant::{e2m1_value, e8m0_value, mxfp4_quantize_group};
use turbine_kernels::quant::MX_BLOCK;

use super::common::{
    Follows, Owned, bf16_bytes, bf16_values, matches_ignore, module_of, quantization_config,
    read_owned, scheme_unsupported, shard_like, shard_misaligned, write_fixture,
};
use super::{ActivationQuant, LinearSlot, QuantScheme, WeightFormat};
use crate::ModelError;
use crate::config::unsupported;
use crate::loader::{LM_HEAD, StackPlace, WeightSlot};
use crate::safetensors::{Dtype, TensorEntry};

/// The container of an MXFP4 packaging.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mxfp4Kind {
    CompressedTensors,
    Quark,
    OpenAi,
}

impl Mxfp4Kind {
    /// The data and scale tensor suffixes.
    fn names(self) -> (&'static str, &'static str) {
        match self {
            Mxfp4Kind::CompressedTensors => ("weight_packed", "weight_scale"),
            Mxfp4Kind::Quark => ("weight", "weight_scale"),
            Mxfp4Kind::OpenAi => ("weight_blocks", "weight_scales"),
        }
    }
}

/// One MXFP4 packaging's parsed configuration.
#[derive(Clone, PartialEq, Debug)]
pub struct Mxfp4Layout {
    pub kind: Mxfp4Kind,
    /// `None` (weight-only, column `mxfp4`) or `Mxfp4Emulated` (Quark W4A4, `mxfp4_a4`).
    pub act: ActivationQuant,
    /// Module names left unquantized (as [`matches_ignore`] reads them).
    pub ignore: Vec<String>,
}

/// A checkpoint container of MXFP4 weights.
pub trait Mxfp4Packaging: Send + Sync + 'static {
    const NAME: &'static str;
    /// The layout of the registry entry (used only by the conformance suite's fixtures).
    const DEFAULT: Mxfp4Layout;
    fn claims(q: &serde_json::Value) -> Result<(), ModelError>;
    fn parse(q: &serde_json::Value) -> Result<Mxfp4Layout, ModelError>;
    /// The `quantization_config` declaring `layout` (test support).
    fn to_json(layout: &Mxfp4Layout) -> serde_json::Value;
}

/// An ignore entry written as a glob (`model.layers.*.self_attn`, Quark `*lm_head`) as a
/// [`matches_ignore`] pattern: `*` any text, matched from the start.
pub fn glob_ignore(glob: &str) -> String {
    if glob.contains('*') {
        format!("re:{}", glob.replace('.', "\\.").replace('*', ".*"))
    } else {
        glob.to_string()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Part {
    Data,
    Scales,
    Plain,
}

impl Mxfp4Layout {
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
            QuantScheme::Mxfp4
        } else {
            QuantScheme::Bf16
        }
    }

    pub fn column(&self) -> WeightFormatColumn {
        if self.act == ActivationQuant::Mxfp4Emulated {
            WeightFormatColumn::Mxfp4A4
        } else {
            WeightFormatColumn::Mxfp4
        }
    }

    /// What slot `slot` is: the quantized data (for Quark named like the family slot), its
    /// scales, or a BF16 tensor.
    fn part(&self, slot: &WeightSlot) -> Part {
        let (data, scales) = self.kind.names();
        let suffix = slot.name.rsplit('.').next().unwrap_or("");
        if suffix == scales && slot.name.contains('.') {
            Part::Scales
        } else if suffix == data
            && (self.kind != Mxfp4Kind::Quark || self.quantizes(&slot.name, &slot.shape))
        {
            // Quark's data slot has the family slot's name (`X.weight`) and is 2-D like it:
            // a quantized layer's `X.weight` is always its data slot.
            Part::Data
        } else {
            Part::Plain
        }
    }

    /// The slots of family slot `base`: for a quantized layer its data and scales, stacked like
    /// `base`; else `base`.
    pub fn slots(&self, base: &WeightSlot) -> Vec<WeightSlot> {
        self.derive(base).0
    }

    /// Refuses a tensor-parallel shard that cuts a 32-input block.
    pub fn check_shard(&self, base: &WeightSlot) -> Result<(), ModelError> {
        match (&base.source, self.derive(base).1) {
            (Some(src), false) => Err(shard_misaligned(&base.name, src, "the 32-input block")),
            _ => Ok(()),
        }
    }

    /// [`Mxfp4Layout::slots`], sharded like `base`, and whether the shard is aligned.
    fn derive(&self, base: &WeightSlot) -> (Vec<WeightSlot>, bool) {
        if !self.quantizes(&base.name, &base.shape) {
            return (vec![base.clone()], true);
        }
        let module = module_of(&base.name).expect("quantizes() checked the suffix");
        let (data, scales) = self.kind.names();
        let (n, k) = (base.shape[0], base.shape[1]);
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
            per_row(data, key.clone(), k / 2),
            per_row(scales, format!("{key}_scale"), k.div_ceil(MX_BLOCK)),
        ];
        let mut aligned = true;
        if let Some(src) = &base.source {
            let (full_n, full_k) = (src.shape[0], src.shape[1]);
            for (slot, cols) in out.iter_mut().zip([2, MX_BLOCK]) {
                aligned &= shard_like(
                    slot,
                    src,
                    vec![full_n, full_k / cols],
                    Follows {
                        rows: 1,
                        cols: Some(cols),
                    },
                );
            }
        }
        (out, aligned)
    }

    pub fn slot_dtype(&self, slot: &WeightSlot) -> DType {
        match self.part(slot) {
            Part::Data | Part::Scales => DType::U8,
            Part::Plain => super::Bf16::DTYPE,
        }
    }

    pub fn check_tensor(&self, entry: &TensorEntry) -> Result<(), ModelError> {
        let (data, scales) = self.kind.names();
        let suffix = entry.name.rsplit('.').next().unwrap_or("");
        // The data is packed U8 whenever its name is its own (compressed-tensors
        // `weight_packed`, OpenAI `weight_blocks`); only Quark's data shares the name `weight`
        // with an unquantized BF16 layer, so there the dtype tells them apart (review r13 C3).
        let packed =
            suffix == scales || (suffix == data && (data != "weight" || entry.dtype == Dtype::U8));
        let ok = if packed {
            entry.dtype == Dtype::U8
        } else {
            entry.dtype == Dtype::BF16
        };
        if !ok {
            return Err(ModelError::Unsupported {
                field: "tensor dtype".to_string(),
                value: format!("{} ({})", entry.dtype, entry.name),
                supported: "U8 MXFP4 data and E8M0 scales, BF16 otherwise".to_string(),
            });
        }
        if suffix == data && packed {
            let k = match entry.shape.as_slice() {
                [_, half] if self.kind != Mxfp4Kind::OpenAi => half * 2,
                [_, blocks, 16] if self.kind == Mxfp4Kind::OpenAi => blocks * MX_BLOCK,
                other => {
                    return Err(unsupported(
                        "tensor shape",
                        format!("{other:?} ({})", entry.name),
                        "packed MXFP4 data",
                    ));
                }
            };
            if !k.is_multiple_of(MX_BLOCK) {
                return Err(scheme_unsupported(
                    "weight shape",
                    format!("k = {k} ({})", entry.name),
                    "k a multiple of 32",
                ));
            }
        }
        Ok(())
    }

    /// OpenAI's `[n, k/32, 16]` blocks are the loaded bytes regrouped: validated and passed on.
    pub fn repacks(&self, slot: &WeightSlot) -> bool {
        self.kind == Mxfp4Kind::OpenAi && self.part(slot) == Part::Data
    }

    pub fn repack(
        &self,
        slot: &WeightSlot,
        entry: &TensorEntry,
        bytes: Vec<u8>,
    ) -> Result<Vec<u8>, ModelError> {
        let (n, half) = (slot.shape[0], slot.shape[1]);
        if entry.shape != [n, half * 2 / MX_BLOCK, 16] {
            return Err(ModelError::Safetensors {
                file: entry.file.clone(),
                tensor: entry.name.clone(),
                rule: format!(
                    "shape {:?} != [{n}, {}, 16]",
                    entry.shape,
                    half * 2 / MX_BLOCK
                ),
            });
        }
        Ok(bytes)
    }
}

/// A weight format over the MXFP4 layout of packaging `P`.
pub struct Mxfp4Format<P> {
    pub layout: Mxfp4Layout,
    packaging: PhantomData<fn() -> P>,
}

impl<P: Mxfp4Packaging> Mxfp4Format<P> {
    /// The registry entry.
    pub const ENTRY: Mxfp4Format<P> = Mxfp4Format::with(P::DEFAULT);

    pub const fn with(layout: Mxfp4Layout) -> Mxfp4Format<P> {
        Mxfp4Format {
            layout,
            packaging: PhantomData,
        }
    }
}

impl<P: Mxfp4Packaging> Module for Mxfp4Format<P> {
    fn name(&self) -> &'static str {
        P::NAME
    }
}

impl<P: Mxfp4Packaging> WeightFormat for Mxfp4Format<P> {
    fn check_config(&self, top: &serde_json::Value) -> Result<(), ModelError> {
        match quantization_config(top) {
            Some(q) => P::claims(q),
            None => Err(unsupported("quantization_config", "null", P::NAME)),
        }
    }

    fn configure(&self, top: &serde_json::Value) -> Result<Arc<dyn WeightFormat>, ModelError> {
        let q = quantization_config(top).expect("check_config accepted a quantization_config");
        Ok(Arc::new(Mxfp4Format::<P>::with(P::parse(q)?)))
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

    fn check_shard(&self, base: &WeightSlot) -> Result<(), ModelError> {
        self.layout.check_shard(base)
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
        write_tiny_mxfp4(&self.layout, &P::to_json(&self.layout), dir, twin)?;
        Ok(true)
    }
}

/// [`WeightFormat::write_tiny`] of an MXFP4 layout: each BF16 weight the layout quantizes
/// becomes E2M1 codes and E8M0 exponents (Quark's `even` rule, exact in BF16 when decoded) in
/// the packaging's tensors; `config.json` gains `quantization_config`.
fn write_tiny_mxfp4(
    layout: &Mxfp4Layout,
    config: &serde_json::Value,
    dir: &Path,
    twin: Option<&Path>,
) -> Result<(), ModelError> {
    let (data_suffix, scale_suffix) = layout.kind.names();
    let mut out: Vec<Owned> = Vec::new();
    let mut dequantized: HashMap<String, Vec<u8>> = HashMap::new();
    for (name, dtype, shape, data) in read_owned(&dir.join("model.safetensors"))? {
        if dtype != Dtype::BF16 || !layout.quantizes(&name, &shape) {
            out.push((name, dtype, shape, data));
            continue;
        }
        let w = bf16_values(&data);
        let (n, k) = (shape[0], shape[1]);
        if !k.is_multiple_of(MX_BLOCK) {
            return Err(scheme_unsupported(
                "weight shape",
                format!("k = {k} ({name})"),
                "k a multiple of 32",
            ));
        }
        let blocks = k / MX_BLOCK;
        let (mut packed, mut scales, mut deq) = (
            Vec::with_capacity(n * k / 2),
            Vec::with_capacity(n * blocks),
            Vec::with_capacity(n * k),
        );
        for group in w.chunks_exact(MX_BLOCK) {
            let (codes, e) = mxfp4_quantize_group(group.try_into().expect("32 values"));
            let s = e8m0_value(e);
            for b in codes {
                deq.push(e2m1_value(b & 0xf) * s);
                deq.push(e2m1_value(b >> 4) * s);
            }
            packed.extend(codes);
            scales.push(e);
        }
        let module = module_of(&name).expect("quantizes() checked the suffix");
        let data_shape = if layout.kind == Mxfp4Kind::OpenAi {
            vec![n, blocks, 16]
        } else {
            vec![n, k / 2]
        };
        out.push((
            format!("{module}.{data_suffix}"),
            Dtype::U8,
            data_shape,
            packed,
        ));
        out.push((
            format!("{module}.{scale_suffix}"),
            Dtype::U8,
            vec![n, blocks],
            scales,
        ));
        dequantized.insert(name, bf16_bytes(&deq));
    }
    write_fixture(dir, twin, &out, dequantized, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::SafetensorsIndex;
    use crate::weights::common::write_owned;
    use crate::{WeightLoader, families};

    fn layout(kind: Mxfp4Kind) -> Mxfp4Layout {
        Mxfp4Layout {
            kind,
            act: ActivationQuant::None,
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

    /// Review r13 C3: MXFP4 data whose tensor name is its own (compressed-tensors
    /// `weight_packed`, OpenAI `weight_blocks`) must be U8 — a BF16 one is refused, not taken
    /// for an unquantized layer — while Quark's `weight` keeps the dtype rule (U8 packed data,
    /// BF16 an unquantized layer). Breaks if BF16 packed data passes `check_tensor`.
    #[test]
    fn packed_data_must_be_u8() {
        let ct = layout(Mxfp4Kind::CompressedTensors);
        let name = "model.layers.0.mlp.gate_proj.weight_packed";
        assert!(
            ct.check_tensor(&entry(name, Dtype::U8, vec![256, 64]))
                .is_ok()
        );
        let err = ct
            .check_tensor(&entry(name, Dtype::BF16, vec![256, 64]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("tensor dtype"), "{err}");
        let oa = layout(Mxfp4Kind::OpenAi);
        let blocks = "model.layers.0.mlp.gate_proj.weight_blocks";
        assert!(
            oa.check_tensor(&entry(blocks, Dtype::U8, vec![256, 4, 16]))
                .is_ok()
        );
        assert!(
            oa.check_tensor(&entry(blocks, Dtype::BF16, vec![256, 4, 16]))
                .is_err()
        );
        let quark = layout(Mxfp4Kind::Quark);
        let w = "model.layers.0.mlp.gate_proj.weight";
        assert!(
            quark
                .check_tensor(&entry(w, Dtype::U8, vec![256, 64]))
                .is_ok()
        );
        assert!(
            quark
                .check_tensor(&entry(w, Dtype::BF16, vec![256, 128]))
                .is_ok()
        );
    }

    /// Review r13 C3 (loader): a Quark layer whose `weight` is BF16 but shaped like packed data
    /// (`[n, k/2]`) passes the dtype rule (it looks unquantized) and the shape check; the
    /// loader refuses it by its byte length instead of copying twice the slot's bytes over the
    /// next part of the fused gate/up stack. Breaks if an as-stored upload is not checked
    /// against its slot's byte length.
    #[test]
    fn loader_refuses_a_wider_tensor_of_the_slot_shape() {
        let tmp = crate::testing::TempDir::new("mxfp4-wide-data");
        let q = serde_json::json!({
            "quant_method": "quark",
            "global_quant_config": {
                "weight": {"dtype": "fp4", "qscheme": "per_group", "group_size": 32,
                           "scale_format": "e8m0", "is_dynamic": false,
                           "round_method": "half_even", "scale_calculation_mode": "even"},
                "input_tensors": null, "output_tensors": null, "bias": null},
            "exclude": ["lm_head"],
            "export": {"weight_format": "real_quantized", "pack_method": "reorder"},
            "algo_config": [], "layer_quant_config": {}, "layer_type_quant_config": {},
            "kv_cache_quant_config": {},
        });
        let fixture = crate::testing::tiny::write_tiny_quantized(tmp.path(), 3, &q, 128, 128);
        let cfg = &fixture.quantized.config;
        let file = fixture.quantized.dir.join("model.safetensors");
        let target = "model.layers.0.mlp.gate_proj.weight";
        let mut tensors = read_owned(&file).expect("read");
        let t = tensors
            .iter_mut()
            .find(|t| t.0 == target)
            .expect("the gate projection");
        assert_eq!(t.1, Dtype::U8);
        t.1 = Dtype::BF16;
        t.3 = vec![0; t.3.len() * 2];
        write_owned(&file, &tensors).expect("write");
        let index = SafetensorsIndex::open(&fixture.quantized.dir).expect("index");
        let format = cfg.weight_format.get();
        let slots = families::llama_slots(cfg);
        let mem: Arc<dyn turbine_tensor::DeviceMemory> =
            turbine_tensor::host::HostMemory::new(turbine_core::types::DeviceId(0), 1 << 30);
        let err = WeightLoader::load_format(format, &index, &slots, &mem, 1 << 20)
            .expect_err("a BF16 tensor in a U8 slot")
            .to_string();
        assert!(err.contains(target) && err.contains("bytes"), "{err}");
    }

    #[test]
    fn glob_ignores() {
        let p = glob_ignore("model.layers.*.self_attn");
        assert!(matches_ignore(&p, "model.layers.3.self_attn.q_proj"));
        assert!(!matches_ignore(&p, "model.layers.3.mlp.down_proj"));
        assert!(matches_ignore(&glob_ignore("*lm_head"), "lm_head"));
        assert_eq!(glob_ignore("lm_head"), "lm_head");
    }
}
