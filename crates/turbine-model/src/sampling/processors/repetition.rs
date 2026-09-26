//! `repetition_penalty` (HF): every id seen in the prompt or the output so far is divided by the
//! penalty when positive and multiplied by it otherwise.
use turbine_core::registry::Module;

use crate::sampling::{LogitsProcessor, ProcessorParams, ProcessorState, Touched};

pub struct RepetitionPenalty;

impl Module for RepetitionPenalty {
    fn name(&self) -> &'static str {
        "repetition_penalty"
    }
}

impl LogitsProcessor for RepetitionPenalty {
    fn applies(&self, p: &ProcessorParams, _: &ProcessorState<'_>) -> bool {
        p.repetition_penalty != 1.0
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
        let penalty = p.repetition_penalty;
        let prompt = s.prompt_tokens;
        let seen = prompt.iter().copied().chain(
            s.counts
                .keys()
                .copied()
                .filter(|id| prompt.binary_search(id).is_err()),
        );
        for id in seen {
            if let Some(i) = touched.touch(logits, id) {
                let v = logits[i];
                logits[i] = if v > 0.0 { v / penalty } else { v * penalty };
            }
        }
    }
}
