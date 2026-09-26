//! `min_tokens`: EOS and stop ids are set to −∞ while fewer than `min_tokens` tokens were
//! generated.
use turbine_core::registry::Module;

use crate::sampling::{LogitsProcessor, ProcessorParams, ProcessorState, Touched};

pub struct MinTokens;

impl Module for MinTokens {
    fn name(&self) -> &'static str {
        "min_tokens"
    }
}

impl LogitsProcessor for MinTokens {
    fn applies(&self, p: &ProcessorParams, s: &ProcessorState<'_>) -> bool {
        (s.step as u64) < u64::from(p.min_tokens)
    }

    fn device_capable(&self) -> bool {
        false
    }

    fn needs_full_row(&self) -> bool {
        true
    }

    fn apply(
        &self,
        logits: &mut [f32],
        touched: &mut Touched,
        _: &ProcessorParams,
        s: &ProcessorState<'_>,
    ) {
        for &id in s.eos_token_ids {
            if let Some(i) = touched.touch(logits, id) {
                logits[i] = f32::NEG_INFINITY;
            }
        }
    }
}
