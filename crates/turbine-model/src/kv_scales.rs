//! The L0 KV cache element format of a model (Phase 6a S-13, `kv.dtype`): BF16 (the exact
//! default), or FP8 e4m3 pages with one K and one V scale per layer (user decision 2026-09-28,
//! Q10: the checkpoint's `k_scale` / `v_scale` tensors when present, else 1.0). A K element is
//! stored as `e4m3(k / k_scale)` and read back as `e4m3 · k_scale`, likewise V.

use std::os::unix::fs::FileExt;
use std::sync::Arc;

use half::{bf16, f16};
use turbine_core::types::DType;

use crate::ModelError;
use crate::safetensors::{Dtype, SafetensorsIndex, TensorEntry, io_err, open_regular};

/// How the paged KV pool stores K and V ([`crate::ModelArchConfig::kv_cache`]).
#[derive(Clone, PartialEq, Debug)]
pub struct KvCache {
    /// Element type of the pool pages: the activation dtype (BF16) or [`DType::F8E4M3`].
    pub dtype: DType,
    /// Per-layer K dequantization scales (FP8 only; empty for BF16).
    pub k_scales: Arc<[f32]>,
    /// Per-layer V dequantization scales (FP8 only; empty for BF16).
    pub v_scales: Arc<[f32]>,
}

impl KvCache {
    /// BF16 pages: the exact, default format.
    pub fn bf16() -> KvCache {
        KvCache {
            dtype: DType::BF16,
            k_scales: Arc::from(Vec::new()),
            v_scales: Arc::from(Vec::new()),
        }
    }

    /// FP8 e4m3 pages with these per-layer scales.
    pub fn fp8_e4m3(k_scales: Vec<f32>, v_scales: Vec<f32>) -> KvCache {
        KvCache {
            dtype: DType::F8E4M3,
            k_scales: Arc::from(k_scales),
            v_scales: Arc::from(v_scales),
        }
    }

    /// True when the pages are quantized (FP8).
    pub fn is_fp8(&self) -> bool {
        self.dtype == DType::F8E4M3
    }

    /// Layer `layer`'s K scale (model layer index; 1.0 when none is stored).
    pub fn k_scale(&self, layer: usize) -> f32 {
        self.k_scales.get(layer).copied().unwrap_or(1.0)
    }

    /// Layer `layer`'s V scale (model layer index; 1.0 when none is stored).
    pub fn v_scale(&self, layer: usize) -> f32 {
        self.v_scales.get(layer).copied().unwrap_or(1.0)
    }

    /// FP8 e4m3 pages with the checkpoint's per-layer scales (Q10): every layer's K and V
    /// scale ([`scale_names`]) when the checkpoint stores them, all 1.0 when it stores none.
    /// Errors name the tensor: scales for only some layers or halves, or a scale that is not
    /// one finite positive F32 / BF16 / F16 element.
    pub fn fp8_from_checkpoint(
        index: &SafetensorsIndex,
        num_layers: u32,
    ) -> Result<KvCache, ModelError> {
        let mut found: Vec<Option<f32>> = Vec::with_capacity(2 * num_layers as usize);
        let mut first_missing = None;
        for half in ["k", "v"] {
            for layer in 0..num_layers {
                let names = scale_names(layer, half);
                match names.iter().find_map(|n| index.get(n)) {
                    Some(entry) => found.push(Some(read_scale(entry)?)),
                    None => {
                        first_missing.get_or_insert_with(|| names[0].clone());
                        found.push(None);
                    }
                }
            }
        }
        let present = found.iter().filter(|s| s.is_some()).count();
        if present == 0 {
            let ones = vec![1.0; num_layers as usize];
            return Ok(KvCache::fp8_e4m3(ones.clone(), ones));
        }
        if let Some(missing) = first_missing {
            return Err(ModelError::MissingTensor(format!(
                "{missing} (the checkpoint stores {present} of the {} per-layer KV scales)",
                found.len()
            )));
        }
        let scales: Vec<f32> = found.into_iter().flatten().collect();
        let (k, v) = scales.split_at(num_layers as usize);
        Ok(KvCache::fp8_e4m3(k.to_vec(), v.to_vec()))
    }
}

impl Default for KvCache {
    fn default() -> Self {
        KvCache::bf16()
    }
}

/// Checkpoint names of layer `layer`'s K (`half` = `k`) or V scale, in lookup order: the
/// compressed-tensors / transformers spelling (`self_attn.k_scale`), then the older
/// `k_proj.output_scale` (vLLM reads both).
fn scale_names(layer: u32, half: &str) -> [String; 2] {
    let p = format!("model.layers.{layer}.self_attn");
    [
        format!("{p}.{half}_scale"),
        format!("{p}.{half}_proj.output_scale"),
    ]
}

/// One scale scalar (shape `[]` or `[1]`), finite and positive.
fn read_scale(entry: &TensorEntry) -> Result<f32, ModelError> {
    let rule = |rule: String| ModelError::Safetensors {
        file: entry.file.clone(),
        tensor: entry.name.clone(),
        rule,
    };
    if entry.shape.len() > 1 || entry.shape.iter().product::<usize>() != 1 {
        return Err(rule(format!(
            "a KV scale must be one element, has shape {:?}",
            entry.shape
        )));
    }
    let mut buf = [0u8; 4];
    let n = entry.byte_len() as usize;
    if n > buf.len() {
        return Err(rule(format!("a KV scale of {n} bytes")));
    }
    let file = open_regular(&entry.file)?;
    file.read_exact_at(&mut buf[..n], entry.range.start)
        .map_err(|e| io_err(&entry.file, e))?;
    let value = match entry.dtype {
        Dtype::F32 => f32::from_le_bytes(buf),
        Dtype::BF16 => bf16::from_le_bytes([buf[0], buf[1]]).to_f32(),
        Dtype::F16 => f16::from_le_bytes([buf[0], buf[1]]).to_f32(),
        other => {
            return Err(rule(format!(
                "a KV scale must be F32, BF16 or F16, is {other:?}"
            )));
        }
    };
    if !(value.is_finite() && value > 0.0) {
        return Err(rule(format!(
            "a KV scale must be finite and positive, is {value}"
        )));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TempDir;

    type Raw<'a> = (&'a str, Dtype, Vec<usize>, Vec<u8>);

    fn checkpoint(dir: &std::path::Path, tensors: &[Raw<'_>]) -> SafetensorsIndex {
        let views: Vec<(String, ::safetensors::tensor::TensorView<'_>)> = tensors
            .iter()
            .map(|(n, d, s, b)| {
                let view = ::safetensors::tensor::TensorView::new(*d, s.clone(), b).expect("view");
                (n.to_string(), view)
            })
            .collect();
        let bytes = ::safetensors::serialize(views, None).expect("serialize");
        std::fs::write(dir.join("model.safetensors"), bytes).expect("write");
        SafetensorsIndex::open(dir).expect("index")
    }

    fn f32s(v: f32) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }

    /// Scales from either checkpoint spelling and dtype, all-or-nothing, finite and positive.
    /// Breaks if a missing layer silently falls back to 1.0 or a bad scale is accepted.
    #[test]
    fn checkpoint_scales_are_all_or_nothing() {
        let tmp = TempDir::new("kv-scales");
        let w: Raw<'_> = (
            "model.embed_tokens.weight",
            Dtype::BF16,
            vec![2],
            vec![0u8; 4],
        );

        let none = checkpoint(tmp.path(), std::slice::from_ref(&w));
        let none = KvCache::fp8_from_checkpoint(&none, 2).expect("none");
        assert_eq!(&none.k_scales[..], &[1.0, 1.0]);
        assert_eq!(&none.v_scales[..], &[1.0, 1.0]);
        assert_eq!(none.dtype, DType::F8E4M3);
        assert_eq!((none.k_scale(1), none.v_scale(5)), (1.0, 1.0));

        let bf = bf16::from_f32(0.25).to_le_bytes().to_vec();
        let hf = f16::from_f32(3.0).to_le_bytes().to_vec();
        let both = checkpoint(
            tmp.path(),
            &[
                w.clone(),
                (
                    "model.layers.0.self_attn.k_scale",
                    Dtype::F32,
                    vec![],
                    f32s(0.5),
                ),
                (
                    "model.layers.1.self_attn.k_proj.output_scale",
                    Dtype::F32,
                    vec![1],
                    f32s(2.0),
                ),
                ("model.layers.0.self_attn.v_scale", Dtype::BF16, vec![], bf),
                ("model.layers.1.self_attn.v_scale", Dtype::F16, vec![1], hf),
            ],
        );
        let read = KvCache::fp8_from_checkpoint(&both, 2).expect("scales");
        assert_eq!(&read.k_scales[..], &[0.5, 2.0]);
        assert_eq!(&read.v_scales[..], &[0.25, 3.0]);

        let partial = checkpoint(
            tmp.path(),
            &[
                w.clone(),
                (
                    "model.layers.0.self_attn.k_scale",
                    Dtype::F32,
                    vec![],
                    f32s(0.5),
                ),
            ],
        );
        let err = KvCache::fp8_from_checkpoint(&partial, 2)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("model.layers.1.self_attn.k_scale") && err.contains("1 of the 4"),
            "{err}"
        );

        for (bad, shape) in [
            (0.0f32, vec![]),
            (f32::NAN, vec![]),
            (-1.0, vec![]),
            (1.0, vec![2]),
        ] {
            let bytes = if shape.is_empty() {
                f32s(bad)
            } else {
                [f32s(bad), f32s(bad)].concat()
            };
            let name = "model.layers.0.self_attn.k_scale";
            let ck = checkpoint(tmp.path(), &[w.clone(), (name, Dtype::F32, shape, bytes)]);
            let err = KvCache::fp8_from_checkpoint(&ck, 1)
                .unwrap_err()
                .to_string();
            assert!(err.contains(name), "{err}");
        }
    }
}
