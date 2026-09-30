//! OpenAI's native MXFP4 checkpoints (`openai_mxfp4`, Phase 6a S-3, S-10, column `mxfp4`):
//! `quant_method: mxfp4`, `modules_to_not_convert` naming modules left in BF16 (globs such as
//! `model.layers.*.self_attn`). Tensors: `X.weight_blocks` U8 `[n, k/32, 16]` (32 E2M1 codes
//! per block, low nibble first) and `X.weight_scales` U8 `[n, k/32]` ([`super::mxfp4`]). Its
//! family (`GptOssForCausalLM`, whose experts are stored this way) arrives with Phase 7; in
//! this phase only the dense tiny fixtures exercise the loader, and the support-matrix row
//! stays `unsupported`.

use serde_json::{Value, json};

use super::ActivationQuant;
use super::common::string_list;
use super::mxfp4::{Mxfp4Format, Mxfp4Kind, Mxfp4Layout, Mxfp4Packaging, glob_ignore};
use crate::ModelError;
use crate::config::unsupported;

/// The OpenAI MXFP4 container.
pub struct OpenAiMxfp4;

/// The registry entry.
pub static OPENAI_MXFP4: Mxfp4Format<OpenAiMxfp4> = Mxfp4Format::ENTRY;

impl Mxfp4Packaging for OpenAiMxfp4 {
    const NAME: &'static str = "openai_mxfp4";
    const DEFAULT: Mxfp4Layout = Mxfp4Layout {
        kind: Mxfp4Kind::OpenAi,
        act: ActivationQuant::None,
        ignore: Vec::new(),
    };

    fn claims(q: &Value) -> Result<(), ModelError> {
        let method = q.get("quant_method").and_then(Value::as_str).unwrap_or("");
        if method == "mxfp4" {
            Ok(())
        } else {
            Err(unsupported(
                "quantization_config.quant_method",
                method,
                "mxfp4",
            ))
        }
    }

    fn parse(q: &Value) -> Result<Mxfp4Layout, ModelError> {
        Ok(Mxfp4Layout {
            kind: Mxfp4Kind::OpenAi,
            act: ActivationQuant::None,
            ignore: string_list(q, "modules_to_not_convert")
                .iter()
                .map(|g| glob_ignore(g))
                .collect(),
        })
    }

    fn to_json(layout: &Mxfp4Layout) -> Value {
        let globs: Vec<String> = layout
            .ignore
            .iter()
            .map(|p| match p.strip_prefix("re:") {
                Some(re) => re.replace(".*", "*").replace("\\.", "."),
                None => p.clone(),
            })
            .collect();
        json!({"quant_method": "mxfp4", "modules_to_not_convert": globs})
    }
}
