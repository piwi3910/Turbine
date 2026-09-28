//! Weight formats (Phase 2m S-10, amended by Phase 6a S-3; contract §24 `weight_format`): one
//! entry per checkpoint *packaging* (BF16, compressed-tensors FP8, AutoAWQ, …), describing how
//! each linear layer is quantized ([`QuantScheme`]), how activations are quantized before it
//! ([`ActivationQuant`]), which tensors a layer needs and how they are repacked into the layout
//! the kernels consume. Activations always run in BF16 and the KV dtype is configured
//! (`kv.dtype`), not a property of the checkpoint. One file per packaging and one entry in
//! [`registry`]; [`detect`] picks the packaging of a `config.json` and configures it from the
//! `quantization_config` ([`WeightFormat::configure`]).

use std::path::Path;
use std::sync::Arc;

use turbine_core::registry::{Module, Registry};
use turbine_core::support::WeightFormatColumn;
use turbine_core::types::DType;
use turbine_kernels::quant::{ActQuantDesc, QuantSchemeDesc};

use crate::ModelError;
use crate::config::ModelArchConfig;
use crate::loader::WeightSlot;
use crate::safetensors::TensorEntry;

pub mod bf16;
pub mod ct_fp8;
mod fp8;
pub mod hf_fp8;

pub use bf16::Bf16;

/// How one linear layer's weight is stored after loading (Phase 6a S-3).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum QuantScheme {
    /// Unquantized BF16 (also every layer a checkpoint leaves unquantized, e.g. `lm_head`).
    Bf16,
    Fp8Tensor,
    Fp8Channel,
    Fp8Block {
        n: u32,
        k: u32,
    },
    Int4Group {
        group: u32,
        zero_points: bool,
    },
    Mxfp4,
}

impl QuantScheme {
    /// The kernel-side scheme; `None` for BF16 (the plain GEMM runs it).
    pub fn kernel(self) -> Option<QuantSchemeDesc> {
        Some(match self {
            QuantScheme::Bf16 => return None,
            QuantScheme::Fp8Tensor => QuantSchemeDesc::Fp8Tensor,
            QuantScheme::Fp8Channel => QuantSchemeDesc::Fp8Channel,
            QuantScheme::Fp8Block { n, k } => QuantSchemeDesc::Fp8Block {
                block_n: n,
                block_k: k,
            },
            QuantScheme::Int4Group {
                group,
                zero_points: true,
            } => QuantSchemeDesc::Int4GroupZp { group },
            QuantScheme::Int4Group {
                group,
                zero_points: false,
            } => QuantSchemeDesc::Int4GroupSym { group },
            QuantScheme::Mxfp4 => QuantSchemeDesc::Mxfp4,
        })
    }

    /// Device bytes of an `n × k` layer in this scheme: packed data, F32 scales (E8M0 bytes for
    /// MXFP4) and zero points.
    pub fn bytes(self, n: u64, k: u64) -> u64 {
        let Some(desc) = self.kernel() else {
            return n * k * DType::BF16.size_bytes() as u64;
        };
        let (nu, ku) = (n as usize, k as usize);
        let data = desc.data_bytes(nu, ku) as u64;
        let scales = desc.scale_count(nu, ku) as u64;
        match desc {
            QuantSchemeDesc::Mxfp4 => data + scales,
            QuantSchemeDesc::Int4GroupZp { .. } => data + 4 * scales + scales,
            _ => data + 4 * scales,
        }
    }
}

/// How activations are quantized before a quantized linear layer (Phase 6a S-3).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum ActivationQuant {
    /// BF16 activations (weight-only schemes, and every BF16 layer).
    None,
    /// FP8 with the checkpoint's static per-tensor `input_scale`.
    Fp8PerTensorStatic,
    Fp8PerTokenDynamic,
    Fp8PerGroupDynamic {
        group: u32,
    },
    /// MXFP4 quantize-dequantize (Quark W4A4, emulated on RDNA4).
    Mxfp4Emulated,
}

impl ActivationQuant {
    /// The kernel-side mode (the static FP8 scale, `input_scale`, travels with each call).
    pub fn kernel(self) -> ActQuantDesc {
        match self {
            ActivationQuant::None => ActQuantDesc::None,
            ActivationQuant::Fp8PerTensorStatic => ActQuantDesc::Fp8Tensor,
            ActivationQuant::Fp8PerTokenDynamic => ActQuantDesc::Fp8Token,
            ActivationQuant::Fp8PerGroupDynamic { group } => ActQuantDesc::Fp8Group { group },
            ActivationQuant::Mxfp4Emulated => ActQuantDesc::Mxfp4Emulated,
        }
    }
}

/// A linear layer of the model: the checkpoint tensor name of its weight (`….weight`) and its
/// `n` output rows × `k` input columns.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LinearSlot {
    pub name: String,
    pub n: u32,
    pub k: u32,
}

/// The dtype every executor activation uses, whatever the weight format (Phase 6a: quantized
/// formats dequantize or quantize at the GEMM; `model.dtype` is `bf16`).
pub const ACTIVATION_DTYPE: DType = Bf16::DTYPE;

/// A checkpoint packaging: the `config.json` keys that declare it, the tensors it accepts, the
/// support-matrix column it serves, and how each linear layer is stored and fed.
pub trait WeightFormat: Module {
    /// `Ok` when the top-level `config.json` declares this format; else the refusal naming the
    /// offending key.
    fn check_config(&self, top: &serde_json::Value) -> Result<(), ModelError>;
    /// The format as the `config.json` `top` (which [`WeightFormat::check_config`] accepted)
    /// configures it: which layers are quantized, the weight and activation schemes. Refuses a
    /// declared variant this packaging does not serve (`quant_scheme_unsupported` naming the
    /// field). A format without parameters returns itself.
    fn configure(&self, top: &serde_json::Value) -> Result<Arc<dyn WeightFormat>, ModelError>;
    /// The configured format in words (its name, plus the parameters for a configured format):
    /// two formats are equal when their descriptions are.
    fn describe(&self) -> String {
        self.name().to_string()
    }
    /// `Ok` when a checkpoint tensor is stored in this format; else the refusal naming it.
    fn check_tensor(&self, entry: &TensorEntry) -> Result<(), ModelError>;
    /// The support-matrix `weight_format` column of checkpoints in this packaging.
    fn column(&self) -> WeightFormatColumn;
    /// The scheme of one linear layer (`Bf16` for layers the checkpoint leaves unquantized).
    fn scheme(&self, _layer: &LinearSlot) -> QuantScheme {
        QuantScheme::Bf16
    }
    /// How activations are quantized before the quantized layers.
    fn activation(&self) -> ActivationQuant {
        ActivationQuant::None
    }
    /// The checkpoint tensors the loader reads for `base` (a family's slot): for a quantized
    /// linear layer its packed data plus scales, zero points or activation scales; else `base`.
    fn slots(&self, base: &WeightSlot) -> Vec<WeightSlot> {
        vec![base.clone()]
    }
    /// The dtype a loaded slot is stored in on the device.
    fn slot_dtype(&self, _slot: &WeightSlot) -> DType {
        Bf16::DTYPE
    }
    /// Whether the loader passes `slot`'s checkpoint bytes through [`WeightFormat::repack`]
    /// (read whole, rewritten, then uploaded) instead of copying them as they are. The shape of a
    /// repacked slot is the loaded layout; its checkpoint tensor may differ in shape and dtype.
    fn repacks(&self, _slot: &WeightSlot) -> bool {
        false
    }
    /// Rewrites the checkpoint tensor `entry`'s bytes into the layout of `slot` the kernels
    /// consume (e.g. AWQ's nibble order, scales to F32); the result must be exactly the slot's
    /// bytes in [`WeightFormat::slot_dtype`]. Only called when [`WeightFormat::repacks`].
    fn repack(
        &self,
        _slot: &WeightSlot,
        _entry: &TensorEntry,
        bytes: Vec<u8>,
    ) -> Result<Vec<u8>, ModelError> {
        Ok(bytes)
    }
    /// Device bytes of every parameter `cfg` loads: every slot of [`WeightFormat::slots`] at
    /// its shape and [`WeightFormat::slot_dtype`] (a quantized layer's packed data, scales, zero
    /// points and activation scales; embeddings and norms in BF16). The memory budget's weight
    /// term; exactly the bytes the loader uploads.
    fn weight_bytes(&self, cfg: &ModelArchConfig) -> u64 {
        cfg.family
            .0
            .weight_slots(cfg)
            .iter()
            .flat_map(|s| self.slots(s))
            .map(|s| {
                s.shape.iter().map(|&d| d as u64).product::<u64>()
                    * self.slot_dtype(&s).size_bytes() as u64
            })
            .sum()
    }
    /// Test support (Phase 6a fixtures): rewrites the BF16 tiny checkpoint in `dir` (written by
    /// a family's `write_tiny`) into this packaging as configured — every tensor it quantizes,
    /// and `config.json`'s `quantization_config` — and, when `twin` is given, writes there the
    /// same model in BF16 whose weights are the exact dequantized values (the reference the
    /// quantized checkpoint is compared with). `Ok(false)`: the format has no tiny writer.
    fn write_tiny(&self, dir: &Path, twin: Option<&Path>) -> Result<bool, ModelError> {
        let _ = (dir, twin);
        Ok(false)
    }
}

/// A configured weight format as a value of `ModelArchConfig`: equal by
/// [`WeightFormat::describe`], printed as its name.
#[derive(Clone)]
pub struct WeightFormatRef(pub Arc<dyn WeightFormat>);

impl WeightFormatRef {
    /// The format.
    pub fn get(&self) -> &dyn WeightFormat {
        self.0.as_ref()
    }
}

impl PartialEq for WeightFormatRef {
    fn eq(&self, other: &WeightFormatRef) -> bool {
        self.0.describe() == other.0.describe()
    }
}

impl Eq for WeightFormatRef {}

impl std::fmt::Debug for dyn WeightFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::fmt::Debug for WeightFormatRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0.name())
    }
}

static WEIGHT_FORMATS: Registry<dyn WeightFormat> =
    Registry::new("weight_format", &[&Bf16, &ct_fp8::CT_FP8, &hf_fp8::HF_FP8]);

/// Every weight format, in detection order.
pub fn registry() -> &'static Registry<dyn WeightFormat> {
    &WEIGHT_FORMATS
}

/// Test support: copies every file of the checkpoint directory `from` into `to` (created).
pub(crate) fn copy_checkpoint(from: &Path, to: &Path) -> Result<(), ModelError> {
    let io = |path: &Path, e: std::io::Error| ModelError::Io {
        path: path.to_path_buf(),
        detail: e.to_string(),
    };
    std::fs::create_dir_all(to).map_err(|e| io(to, e))?;
    for entry in std::fs::read_dir(from).map_err(|e| io(from, e))? {
        let path = entry.map_err(|e| io(from, e))?.path();
        if path.is_file() {
            let dest = to.join(path.file_name().expect("a file has a name"));
            std::fs::copy(&path, &dest).map_err(|e| io(&dest, e))?;
        }
    }
    Ok(())
}

/// The format of a top-level `config.json`: the first registered format whose
/// [`WeightFormat::check_config`] accepts it, configured from it
/// ([`WeightFormat::configure`], whose refusal is returned), else the first format's refusal.
pub fn detect(top: &serde_json::Value) -> Result<Arc<dyn WeightFormat>, ModelError> {
    let mut first_err = None;
    for format in registry().iter() {
        match format.check_config(top) {
            Ok(()) => return format.configure(top),
            Err(e) => {
                first_err.get_or_insert(e);
            }
        }
    }
    Err(first_err.expect("the weight-format registry is not empty"))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::json;

    use super::*;
    use crate::safetensors::{Dtype, TensorEntry};

    fn entry(dtype: Dtype) -> TensorEntry {
        TensorEntry {
            name: "model.layers.0.mlp.down_proj.weight".to_string(),
            dtype,
            shape: vec![4, 4],
            file: PathBuf::from("model.safetensors"),
            range: 0..32,
        }
    }

    /// The refusals main's `load_model_config` and `check_supported_weights` produced, kept as
    /// literals so the move cannot change a message.
    #[test]
    fn bf16_refusals_match_main() {
        let format = registry().get("bf16").expect("bf16 is registered");
        let err = |top: serde_json::Value| format.check_config(&top).unwrap_err().to_string();
        assert_eq!(
            err(json!({"quantization_config": {"quant_method": "fp8"}})),
            r#"unsupported quantization_config = {"quant_method":"fp8"}; supported: none"#
        );
        assert_eq!(
            err(json!({"torch_dtype": "float16"})),
            "unsupported torch_dtype = float16; supported: bfloat16"
        );
        assert_eq!(
            err(json!({"dtype": "float32"})),
            "unsupported dtype = float32; supported: bfloat16"
        );
        // A null quantization_config is no quantization.
        format
            .check_config(&json!({"quantization_config": null}))
            .unwrap();
        assert_eq!(
            format
                .check_tensor(&entry(Dtype::F16))
                .unwrap_err()
                .to_string(),
            "unsupported tensor dtype = F16 (model.layers.0.mlp.down_proj.weight); supported: BF16"
        );
        format.check_tensor(&entry(Dtype::BF16)).unwrap();
        // The same refusal through `detect`: the first format's error.
        assert_eq!(
            detect(&json!({"torch_dtype": "float16"}))
                .unwrap_err()
                .to_string(),
            "unsupported torch_dtype = float16; supported: bfloat16"
        );
    }

    #[test]
    fn detect_picks_bf16() {
        for top in [
            json!({}),
            json!({"torch_dtype": "bfloat16"}),
            json!({"dtype": "bfloat16"}),
        ] {
            let format = detect(&top).unwrap();
            assert_eq!(format.name(), "bf16", "{top}");
            assert_eq!(format.column(), WeightFormatColumn::Bf16);
        }
        let a = WeightFormatRef(detect(&json!({})).unwrap());
        assert_eq!(a, WeightFormatRef(Arc::new(Bf16)));
        assert_eq!(format!("{a:?}"), "bf16");
    }

    /// A compressed-tensors FP8 `quantization_config` of weight strategy `strategy` and input
    /// activations `input` (`null`: weight-only).
    fn ct(strategy: &str, block: serde_json::Value, input: serde_json::Value) -> serde_json::Value {
        json!({
            "config_groups": {"group_0": {
                "input_activations": input,
                "targets": ["Linear"],
                "weights": {"num_bits": 8, "type": "float", "strategy": strategy,
                            "dynamic": false, "symmetric": true, "block_structure": block},
            }},
            "format": "float-quantized",
            "ignore": ["lm_head"],
            "kv_cache_scheme": null,
            "quant_method": "compressed-tensors",
        })
    }

    fn act(strategy: &str, dynamic: bool, group: serde_json::Value) -> serde_json::Value {
        json!({"num_bits": 8, "type": "float", "strategy": strategy, "dynamic": dynamic,
               "group_size": group, "symmetric": true})
    }

    /// The field and supported text of a `quant_scheme_unsupported` refusal.
    fn refused(q: serde_json::Value) -> (String, String) {
        match detect(&json!({ "quantization_config": q })) {
            Err(ModelError::Unsupported {
                field, supported, ..
            }) => (field, supported),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    /// Phase 6a S-3 (FP8 part; INT4 and MXFP4 join with Tasks 9 and 10): tiny checkpoints
    /// written in each FP8 variant are detected as the right packaging and column, with the
    /// right scheme on a projection, `BF16` on the LM head, and the right activation mode;
    /// unsupported variants are refused `quant_scheme_unsupported` naming the field. Breaks if
    /// a container maps to the wrong column, scheme or activation, or a refusal loses its
    /// reason.
    #[test]
    fn detect_every_packaging() {
        use crate::testing::tiny::write_tiny_quantized;
        let tmp = crate::testing::TempDir::new("turbine-detect-packaging");
        let block = json!([128, 128]);
        let cases = [
            (
                ct("tensor", json!(null), act("tensor", false, json!(null))),
                "ct_fp8",
                WeightFormatColumn::Fp8,
                QuantScheme::Fp8Channel,
                ActivationQuant::Fp8PerTensorStatic,
            ),
            (
                ct("channel", json!(null), act("token", true, json!(null))),
                "ct_fp8",
                WeightFormatColumn::Fp8,
                QuantScheme::Fp8Channel,
                ActivationQuant::Fp8PerTokenDynamic,
            ),
            (
                ct("channel", json!(null), json!(null)),
                "ct_fp8",
                WeightFormatColumn::Fp8,
                QuantScheme::Fp8Channel,
                ActivationQuant::None,
            ),
            (
                ct("block", block.clone(), act("group", true, json!(128))),
                "ct_fp8",
                WeightFormatColumn::Fp8Block,
                QuantScheme::Fp8Block { n: 128, k: 128 },
                ActivationQuant::Fp8PerGroupDynamic { group: 128 },
            ),
            (
                json!({"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
                       "weight_block_size": [128, 128]}),
                "hf_fp8",
                WeightFormatColumn::Fp8Block,
                QuantScheme::Fp8Block { n: 128, k: 128 },
                ActivationQuant::Fp8PerGroupDynamic { group: 128 },
            ),
            (
                json!({"quant_method": "fp8", "activation_scheme": "static"}),
                "hf_fp8",
                WeightFormatColumn::Fp8,
                QuantScheme::Fp8Channel,
                ActivationQuant::Fp8PerTensorStatic,
            ),
        ];
        for (i, (q, name, column, scheme, activation)) in cases.into_iter().enumerate() {
            let fixture = write_tiny_quantized(&tmp.path().join(i.to_string()), 3, &q, 128, 128);
            let cfg = &fixture.quantized.config;
            let format = cfg.weight_format.get();
            assert_eq!(
                (format.name(), format.column(), format.activation()),
                (name, column, activation),
                "{q}"
            );
            let linear = cfg.linear_slots();
            let q_proj = linear
                .iter()
                .find(|l| l.name.ends_with("layers.0.self_attn.q_proj.weight"))
                .expect("q_proj");
            assert_eq!(format.scheme(q_proj), scheme, "{q}");
            let lm_head = crate::weights::LinearSlot {
                name: crate::loader::LM_HEAD.into(),
                n: cfg.vocab_size,
                k: cfg.hidden,
            };
            assert_eq!(format.scheme(&lm_head), QuantScheme::Bf16, "{q}");
            // The twin is plain BF16.
            assert_eq!(fixture.twin.config.weight_format.get().name(), "bf16");
            // Configured formats compare by their parameters.
            assert_ne!(cfg.weight_format, WeightFormatRef(Arc::new(Bf16)), "{q}");
        }

        let mut four_bit = ct("channel", json!(null), json!(null));
        four_bit["config_groups"]["group_0"]["weights"]["num_bits"] = json!(4);
        let (field, supported) = refused(four_bit);
        assert_eq!(field, "weights");
        assert!(
            supported.starts_with("quant_scheme_unsupported"),
            "{supported}"
        );
        let (field, supported) = refused(ct("block", json!([64, 64]), json!(null)));
        assert_eq!(field, "weights.block_structure");
        assert!(
            supported.starts_with("quant_scheme_unsupported"),
            "{supported}"
        );
        let (field, supported) = refused(
            json!({"quant_method": "fp8", "activation_scheme": "dynamic",
                   "weight_block_size": [64, 64]}),
        );
        assert_eq!(field, "weight_block_size");
        assert!(
            supported.starts_with("quant_scheme_unsupported"),
            "{supported}"
        );
        let (field, _) = refused(ct("channel", json!(null), act("token", false, json!(null))));
        assert_eq!(field, "input_activations");
    }

    /// Phase 6a S-3: the BF16 entry describes every linear layer of a tiny Llama as unquantized
    /// BF16 with BF16 activations, its slots are the family's, and its weight bytes are two per
    /// parameter; the scheme byte counts include scales and zero points. Breaks if a default
    /// quantizes a BF16 layer or the byte accounting drops the scale tensors.
    #[test]
    fn bf16_describes_linear_layers() {
        let tmp = crate::testing::TempDir::new("turbine-bf16-linear");
        let family = crate::families::registry().get("llama").expect("llama");
        let spec = family.write_tiny(tmp.path(), 3);
        let cfg = &spec.config;
        let format = registry().get("bf16").unwrap();
        let linear = cfg.linear_slots();
        assert!(!linear.is_empty());
        // q/k/v/o and gate/up/down per layer, plus lm_head when untied.
        let per_layer = 7;
        let head = usize::from(!cfg.tie_word_embeddings);
        assert_eq!(linear.len(), cfg.num_layers as usize * per_layer + head);
        for l in &linear {
            assert_eq!(format.scheme(l), QuantScheme::Bf16, "{}", l.name);
            assert!(!l.name.contains("embed_tokens"), "{}", l.name);
        }
        assert_eq!(format.activation(), ActivationQuant::None);
        for s in family.weight_slots(cfg) {
            assert_eq!(format.slots(&s), vec![s.clone()]);
            assert_eq!(format.slot_dtype(&s), DType::BF16);
        }
        let params: u64 = family
            .weight_slots(cfg)
            .iter()
            .map(|s| s.shape.iter().map(|&d| d as u64).product::<u64>())
            .sum();
        assert_eq!(format.weight_bytes(cfg), 2 * params);
        assert_eq!(cfg.shape().weight_bytes, 2 * params);

        // Scheme sizes for a 256 × 512 layer.
        assert_eq!(QuantScheme::Bf16.bytes(256, 512), 262_144);
        assert_eq!(QuantScheme::Fp8Tensor.bytes(256, 512), 131_072 + 4);
        assert_eq!(QuantScheme::Fp8Channel.bytes(256, 512), 131_072 + 4 * 256);
        assert_eq!(
            QuantScheme::Fp8Block { n: 128, k: 128 }.bytes(256, 512),
            131_072 + 4 * 8
        );
        assert_eq!(
            QuantScheme::Int4Group {
                group: 128,
                zero_points: true
            }
            .bytes(256, 512),
            65_536 + 5 * 1024
        );
        assert_eq!(
            QuantScheme::Int4Group {
                group: 128,
                zero_points: false
            }
            .bytes(256, 512),
            65_536 + 4 * 1024
        );
        assert_eq!(QuantScheme::Mxfp4.bytes(256, 512), 65_536 + 4096);
        assert_eq!(QuantScheme::Bf16.kernel(), None);
        assert_eq!(
            ActivationQuant::Fp8PerTensorStatic.kernel(),
            ActQuantDesc::Fp8Tensor
        );
    }
}
