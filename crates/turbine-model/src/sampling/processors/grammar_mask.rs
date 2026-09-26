//! `grammar_mask`: the constraint's token mask sets every disallowed id to −∞ (last in the
//! chain, so the mask wins over `logit_bias`). It records no originals: a disallowed id is never
//! sampled.
use turbine_core::registry::Module;

use crate::sampling::{LogitsProcessor, ProcessorParams, ProcessorState, Touched};

pub struct GrammarMask;

impl Module for GrammarMask {
    fn name(&self) -> &'static str {
        "grammar_mask"
    }
}

impl LogitsProcessor for GrammarMask {
    fn applies(&self, _: &ProcessorParams, s: &ProcessorState<'_>) -> bool {
        s.mask.is_some()
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
        _: &mut Touched,
        _: &ProcessorParams,
        s: &ProcessorState<'_>,
    ) {
        if let Some(mask) = s.mask {
            mask.apply(logits);
        }
    }
}
