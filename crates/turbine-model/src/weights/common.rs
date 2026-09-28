//! Helpers shared by the quantized packagings: `quantization_config` parsing (ignore lists,
//! refusals), module names and the tiny-fixture safetensors I/O.
use std::collections::HashMap;
use std::path::Path;

use crate::ModelError;
use crate::config::unsupported;
use crate::safetensors::Dtype;

/// `X` of a parameter named `X.weight`.
pub fn module_of(name: &str) -> Option<&str> {
    name.strip_suffix(".weight")
}

/// The non-null `quantization_config` of `top`.
pub fn quantization_config(top: &serde_json::Value) -> Option<&serde_json::Value> {
    top.get("quantization_config").filter(|q| !q.is_null())
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
pub fn regex_lite_match(re: &str, text: &str) -> bool {
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

/// The refusal of a declared variant outside Phase 6a (spec S-3, reason
/// `quant_scheme_unsupported`).
pub fn scheme_unsupported(field: &str, value: impl Into<String>, supported: &str) -> ModelError {
    unsupported(
        field,
        value,
        &format!("quant_scheme_unsupported: {supported}"),
    )
}

/// The smallest power of two `s ≥ x` (1 for `x == 0`): a tiny fixture's scale, so that codes
/// times `s` are exact in BF16.
pub fn pow2_at_least(x: f32) -> f32 {
    if x == 0.0 {
        1.0
    } else {
        x.log2().ceil().exp2()
    }
}

pub type Owned = (String, Dtype, Vec<usize>, Vec<u8>);

pub fn io_error(path: &Path, detail: impl ToString) -> ModelError {
    ModelError::Io {
        path: path.to_path_buf(),
        detail: detail.to_string(),
    }
}

pub fn read_owned(path: &Path) -> Result<Vec<Owned>, ModelError> {
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

pub fn write_owned(path: &Path, tensors: &[Owned]) -> Result<(), ModelError> {
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

pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Writes a quantized tiny fixture: `tensors` replace `dir/model.safetensors`, `config` becomes
/// `config.json`'s `quantization_config`, and (with `twin`) the BF16 checkpoint as it was, its
/// tensors named in `dequantized` replaced by those BF16 bytes, is copied to `twin` first.
pub fn write_fixture(
    dir: &Path,
    twin: Option<&Path>,
    tensors: &[Owned],
    mut dequantized: HashMap<String, Vec<u8>>,
    config: &serde_json::Value,
) -> Result<(), ModelError> {
    if let Some(twin) = twin {
        super::copy_checkpoint(dir, twin)?;
        let twin_path = twin.join("model.safetensors");
        let mut bf16 = read_owned(&twin_path)?;
        for (name, dtype, _, data) in &mut bf16 {
            if let Some(d) = dequantized.remove(name.as_str()) {
                *dtype = Dtype::BF16;
                *data = d;
            }
        }
        write_owned(&twin_path, &bf16)?;
    }
    write_owned(&dir.join("model.safetensors"), tensors)?;
    let config_path = dir.join("config.json");
    let text = std::fs::read(&config_path).map_err(|e| io_error(&config_path, e))?;
    let mut top: serde_json::Value =
        serde_json::from_slice(&text).map_err(|e| io_error(&config_path, e))?;
    top["quantization_config"] = config.clone();
    let text = serde_json::to_string_pretty(&top).expect("serialize JSON");
    std::fs::write(&config_path, text).map_err(|e| io_error(&config_path, e))
}

/// BF16 bytes of `values`.
pub fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
        .collect()
}

/// The values of BF16 bytes.
pub fn bf16_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect()
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
