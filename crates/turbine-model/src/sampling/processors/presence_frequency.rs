//! `presence_frequency_penalty` (OpenAI): every generated id loses
//! `frequency · count + presence`.
use turbine_core::registry::Module;

use crate::sampling::{LogitsProcessor, ProcessorParams, ProcessorState, Touched};

pub struct PresenceFrequencyPenalty;

impl Module for PresenceFrequencyPenalty {
    fn name(&self) -> &'static str {
        "presence_frequency_penalty"
    }
}

impl LogitsProcessor for PresenceFrequencyPenalty {
    fn applies(&self, p: &ProcessorParams, _: &ProcessorState<'_>) -> bool {
        p.presence_penalty != 0.0 || p.frequency_penalty != 0.0
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
        s: &ProcessorState<'_>,
    ) {
        for (&id, &count) in s.counts {
            if let Some(i) = touched.touch(logits, id) {
                logits[i] -= p.frequency_penalty * count as f32 + p.presence_penalty;
            }
        }
    }
}
