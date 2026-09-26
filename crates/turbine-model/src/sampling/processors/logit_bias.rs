//! `logit_bias`: adds each request `(id, bias)` to the row (ids past the row are ignored).
use turbine_core::registry::Module;

use crate::sampling::{LogitsProcessor, ProcessorParams, ProcessorState, Touched};

pub struct LogitBias;

impl Module for LogitBias {
    fn name(&self) -> &'static str {
        "logit_bias"
    }
}

impl LogitsProcessor for LogitBias {
    fn applies(&self, p: &ProcessorParams, _: &ProcessorState<'_>) -> bool {
        !p.logit_bias.is_empty()
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
        p: &ProcessorParams,
        _: &ProcessorState<'_>,
    ) {
        for &(id, bias) in &p.logit_bias {
            if let Some(i) = touched.touch(logits, id) {
                logits[i] += bias;
            }
        }
    }
}
