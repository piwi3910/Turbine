//! Linear layers of the decoder (Phase 6a S-5): a weight is BF16 (the plain GEMM) or quantized
//! as the weight format describes ([`crate::weights::QuantScheme`]): its packed data, scales,
//! zero points and static activation scale, run by `quantize_act` (when activations are
//! quantized) then `qgemm`. The executor dispatches on [`LinearView::q`], so a checkpoint that
//! leaves some layers in BF16 (its ignore list) runs those through the GEMM.
//!
//! Loaded layout ([`crate::weights::WeightFormat::slots`]): parameter `P` holds the data,
//! `P_scale` the scales (one row of scales per weight row, or per block of `block_n` rows),
//! `P_zeros` the zero points (INT4 AWQ) and `P_input_scale` the static FP8 activation scale per
//! weight row (the layer quantizes its input with the largest of them: a fused projection's
//! parts may differ).
use turbine_core::types::DType;
use turbine_kernels::quant::{ActQuantDesc, QuantSchemeDesc};
use turbine_kernels::{OpConfig, QGemmConfig, QuantizeActConfig};
use turbine_tensor::{Tensor, TensorView};

use super::{DecoderDims, invalid};
use crate::ModelError;
use crate::config::ModelArchConfig;
use crate::loader::LoadedWeights;
use crate::weights::{ActivationQuant, QuantScheme};

/// The one quantization of a decoder's quantized linear layers (mixed schemes are refused).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LinearQuant {
    pub scheme: QuantScheme,
    pub act: ActivationQuant,
}

impl LinearQuant {
    /// The quantization of `cfg`'s decoder layers (every linear layer but the LM head), and
    /// whether some of them stay BF16; `Err` names two layers of different schemes.
    pub fn of(cfg: &ModelArchConfig) -> Result<(Option<LinearQuant>, bool), String> {
        let format = cfg.weight_format.get();
        let mut quant: Option<(QuantScheme, String)> = None;
        let mut bf16 = false;
        for l in cfg.linear_slots() {
            if l.name == crate::loader::LM_HEAD {
                continue;
            }
            match (format.scheme(&l), &quant) {
                (QuantScheme::Bf16, _) => bf16 = true,
                (s, None) => quant = Some((s, l.name)),
                (s, Some((first, name))) if s != *first => {
                    return Err(format!("{} is {s:?} but {name} is {first:?}", l.name));
                }
                _ => {}
            }
        }
        Ok((
            quant.map(|(scheme, _)| LinearQuant {
                scheme,
                act: format.activation(),
            }),
            bf16,
        ))
    }

    /// The kernel scheme.
    pub fn kernel(self) -> QuantSchemeDesc {
        self.scheme.kernel().expect("a quantized scheme")
    }

    /// The dtype `qgemm` reads activations in: e4m3 after FP8 quantization, else the
    /// activation dtype (weight-only, or MXFP4 quantize-dequantized in place).
    fn a_dtype(self, act: DType) -> DType {
        if self.act.kernel().is_fp8() {
            DType::F8E4M3
        } else {
            act
        }
    }
}

/// A linear layer's parameters: the weight `[n, k]` (BF16, or the scheme's packed data) and,
/// when quantized, its scales.
pub struct Linear {
    pub w: Tensor,
    pub q: Option<QuantParts>,
}

/// A quantized layer's scales, zero points and static activation scale.
pub struct QuantParts {
    pub quant: LinearQuant,
    pub scales: Tensor,
    pub zeros: Option<Tensor>,
    /// The checkpoint's static FP8 activation scale (the largest of the layer's parts); 1.0
    /// when activations are not statically quantized.
    pub input_scale: f32,
}

/// A borrowed [`Linear`], or rows of one.
#[derive(Clone)]
pub struct LinearView<'a> {
    pub w: TensorView<'a>,
    pub q: Option<QuantView<'a>>,
}

/// A borrowed [`QuantParts`].
#[derive(Clone)]
pub struct QuantView<'a> {
    pub quant: LinearQuant,
    pub scales: TensorView<'a>,
    pub zeros: Option<TensorView<'a>>,
    pub input_scale: f32,
}

impl Linear {
    pub fn view(&self) -> LinearView<'_> {
        LinearView {
            w: self.w.view(),
            q: self.q.as_ref().map(|q| QuantView {
                quant: q.quant,
                scales: q.scales.view(),
                zeros: q.zeros.as_ref().map(Tensor::view),
                input_scale: q.input_scale,
            }),
        }
    }
}

impl<'a> LinearView<'a> {
    /// Weight rows `[start, start + count)` with their scales and zero points (a block-scaled
    /// weight is sliced at block boundaries: the loader refuses stacked parts that are not).
    pub fn rows(&self, start: usize, count: usize) -> LinearView<'a> {
        LinearView {
            w: self.w.rows(start, count),
            q: self.q.as_ref().map(|q| {
                let per = match q.quant.scheme {
                    QuantScheme::Fp8Block { n, .. } => n as usize,
                    _ => 1,
                };
                let slice = |t: &TensorView<'a>| match q.quant.scheme {
                    QuantScheme::Fp8Tensor => t.clone(),
                    _ => t.rows(start / per, count.div_ceil(per)),
                };
                QuantView {
                    quant: q.quant,
                    scales: slice(&q.scales),
                    zeros: q.zeros.as_ref().map(slice),
                    input_scale: q.input_scale,
                }
            }),
        }
    }
}

impl<'a> From<TensorView<'a>> for LinearView<'a> {
    fn from(w: TensorView<'a>) -> LinearView<'a> {
        LinearView { w, q: None }
    }
}

impl DecoderDims {
    /// The `qgemm` of a quantized `[n, k]` layer.
    pub fn qgemm(&self, n: usize, k: usize, q: LinearQuant) -> QGemmConfig {
        QGemmConfig {
            n: n as u32,
            k: k as u32,
            scheme: q.kernel(),
            act_quant: q.act.kernel(),
            a_dtype: q.a_dtype(self.act),
            c_dtype: self.act,
        }
    }

    /// The `quantize_act` before a quantized layer of `k` inputs; `None` for weight-only
    /// schemes.
    pub fn quantize_act(&self, k: usize, q: LinearQuant) -> Option<QuantizeActConfig> {
        let mode = q.act.kernel();
        (mode != ActQuantDesc::None).then(|| QuantizeActConfig {
            cols: k as u32,
            mode,
            x_dtype: self.act,
            out_dtype: q.a_dtype(self.act),
        })
    }

    /// The op configs a decoder linear layer `[n, k]` may run: the GEMM when some layers are
    /// BF16 (or none is quantized), and `quantize_act` + `qgemm` when some are quantized.
    pub fn linear_ops(&self, n: usize, k: usize) -> Vec<OpConfig> {
        let mut ops = Vec::with_capacity(3);
        if self.quant.is_none() || self.bf16_linears {
            ops.push(OpConfig::Gemm(self.gemm(n, k, self.act)));
        }
        if let Some(q) = self.quant {
            ops.extend(self.quantize_act(k, q).map(OpConfig::QuantizeAct));
            ops.push(OpConfig::QGemm(self.qgemm(n, k, q)));
        }
        ops
    }

    /// The widest input (`k`) of a quantized decoder layer: the activation-quantization
    /// scratch holds one `[tokens, k]` matrix and its scales.
    fn widest_input(&self) -> usize {
        self.hidden.max(self.q_dim).max(self.inter)
    }

    /// Device bytes of the activation-quantization scratch for `tokens` rows (none for
    /// BF16 or weight-only layers).
    pub fn act_quant_bytes(&self, tokens: usize) -> u64 {
        let Some(q) = self.quant else { return 0 };
        let Some(cfg) = self.quantize_act(self.widest_input(), q) else {
            return 0;
        };
        let k = self.widest_input();
        (tokens * k * cfg.out_dtype.size_bytes()
            + cfg.mode.scale_count(tokens, k) * DType::F32.size_bytes()) as u64
    }

    /// The activation-quantization scratch for `tokens` rows: the quantized matrix and its F32
    /// scales, flat; `None` without activation quantization.
    pub fn alloc_act_quant(
        &self,
        tokens: usize,
        mem: &std::sync::Arc<dyn turbine_tensor::DeviceMemory>,
    ) -> Result<Option<(Tensor, Tensor)>, ModelError> {
        let Some(q) = self.quant else { return Ok(None) };
        let k = self.widest_input();
        let Some(cfg) = self.quantize_act(k, q) else {
            return Ok(None);
        };
        Ok(Some((
            Tensor::empty(mem, &[tokens * k], cfg.out_dtype)?,
            Tensor::empty(mem, &[cfg.mode.scale_count(tokens, k).max(1)], DType::F32)?,
        )))
    }

    /// Linear layer `name` taken from `weights`: a BF16 `shape` matrix (or stack), or the
    /// quantized data with its `_scale`, optional `_zeros` and (static activations)
    /// `_input_scale` parameters.
    pub fn take_linear(
        &self,
        weights: &mut LoadedWeights,
        name: &str,
        shape: &[usize],
    ) -> Result<Linear, ModelError> {
        let w = weights.take(name)?;
        if w.shape.as_slice() != shape {
            return Err(invalid(format!(
                "{name} is {:?}, expected {shape:?}",
                w.shape.as_slice()
            )));
        }
        if w.dtype == self.act {
            return Ok(Linear { w, q: None });
        }
        let Some(quant) = self.quant else {
            return Err(invalid(format!(
                "{name} is {}, but the weight format quantizes no layer",
                w.dtype.as_str()
            )));
        };
        let scales = weights.take(&format!("{name}_scale"))?;
        let zeros = weights.tensors.remove(&format!("{name}_zeros"));
        let input_scale = match weights.tensors.remove(&format!("{name}_input_scale")) {
            Some(t) => largest_f32(&t)?,
            None if quant.act == ActivationQuant::Fp8PerTensorStatic => {
                return Err(ModelError::MissingTensor(format!("{name}_input_scale")));
            }
            None => 1.0,
        };
        Ok(Linear {
            w,
            q: Some(QuantParts {
                quant,
                scales,
                zeros,
                input_scale,
            }),
        })
    }
}

/// The largest value of an F32 tensor (read back once at load).
fn largest_f32(t: &Tensor) -> Result<f32, ModelError> {
    let mut bytes = vec![0u8; t.shape.iter().product::<usize>() * 4];
    t.storage.copy_to_host(0, &mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .fold(f32::MIN, f32::max))
}
