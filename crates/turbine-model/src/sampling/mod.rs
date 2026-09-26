//! Sampling (Phase 2m S-8, contract §24 `logits_processor`): the logits processors — one file
//! each under `processors/` — applied as an ordered chain, and the [`sampler`] that runs the
//! chain, then the sampling steps (greedy, or temperature → top-k → top-p and one draw).
//!
//! The chain order is the registry order: `logit_bias` → `repetition_penalty` →
//! `presence_frequency_penalty` → `min_tokens` → `grammar_mask`. Each processor says whether it
//! [`applies`](LogitsProcessor::applies) to a step, whether it can run on the device and
//! whether it needs the whole vocabulary row; the device fast path (P2c S-4) is open to a step
//! exactly when every processor that applies to it is device-capable
//! ([`ProcessorChain::device_eligible`]). Processors draw no random numbers: every draw stays in
//! the [`sampler::Sampler`], in its fixed order, so seeded streams are unchanged.
//!
//! Adding a processor: one file under `processors/`, one entry in [`registry`] at its place in
//! the chain, and `registry_conformance::logits_processors` must pass.
use std::collections::HashMap;

use turbine_core::registry::{Module, Registry};
use turbine_core::request::SamplingParams;

use crate::structured::TokenMask;

pub mod sampler;

/// One module per processor.
pub mod processors {
    mod grammar_mask;
    mod logit_bias;
    mod min_tokens;
    mod presence_frequency;
    mod repetition;

    pub use grammar_mask::GrammarMask;
    pub use logit_bias::LogitBias;
    pub use min_tokens::MinTokens;
    pub use presence_frequency::PresenceFrequencyPenalty;
    pub use repetition::RepetitionPenalty;
}

/// The request fields the processors read, fixed for the request's lifetime (taken from its
/// [`SamplingParams`] once, when its sampler is built).
#[derive(Clone, Debug)]
pub struct ProcessorParams {
    /// `(token id, bias)` pairs added to the logits; ids past the row are ignored.
    pub logit_bias: Vec<(u32, f32)>,
    /// HF repetition penalty over prompt and generated ids; 1.0 = off.
    pub repetition_penalty: f32,
    /// OpenAI presence penalty over generated ids; 0 = off.
    pub presence_penalty: f32,
    /// OpenAI frequency penalty over generated ids; 0 = off.
    pub frequency_penalty: f32,
    /// EOS and stop ids are banned until this many tokens were generated.
    pub min_tokens: u32,
}

impl ProcessorParams {
    pub fn new(params: &SamplingParams) -> ProcessorParams {
        ProcessorParams {
            logit_bias: params.logit_bias.clone(),
            repetition_penalty: params.repetition_penalty,
            presence_penalty: params.presence_penalty,
            frequency_penalty: params.frequency_penalty,
            min_tokens: params.min_tokens,
        }
    }
}

/// Neutral: no processor applies (repetition penalty 1.0, not 0).
impl Default for ProcessorParams {
    fn default() -> ProcessorParams {
        ProcessorParams::new(&SamplingParams::default())
    }
}

/// What a processor sees of the request at one step.
#[derive(Clone, Copy, Debug)]
pub struct ProcessorState<'a> {
    /// Distinct prompt ids, sorted (kept only while the repetition penalty is active).
    pub prompt_tokens: &'a [u32],
    /// Occurrences of each generated id.
    pub counts: &'a HashMap<u32, u32>,
    /// The step's index: the number of tokens generated before it.
    pub step: usize,
    /// EOS and stop ids, sorted and distinct.
    pub eos_token_ids: &'a [u32],
    /// The constraint's token mask of this step, when the choice is constrained. For the
    /// eligibility question only its presence matters.
    pub mask: Option<&'a TokenMask>,
}

/// The raw value of every id the chain changed this step (the logprob "originals": reported
/// logprobs are those of the row before any processor).
#[derive(Clone, Debug, Default)]
pub struct Touched {
    originals: HashMap<u32, f32>,
}

impl Touched {
    /// Records the current value of `id` unless an earlier processor already did, and returns
    /// its index in `logits`; `None` for an id past the row.
    pub fn touch(&mut self, logits: &[f32], id: u32) -> Option<usize> {
        let i = id as usize;
        let v = *logits.get(i)?;
        self.originals.entry(id).or_insert(v);
        Some(i)
    }

    /// Forgets every id (at the start of each step).
    pub fn clear(&mut self) {
        self.originals.clear();
    }

    pub fn originals(&self) -> &HashMap<u32, f32> {
        &self.originals
    }
}

/// One adjustment of the logits row before the sampling steps.
pub trait LogitsProcessor: Module {
    /// Whether this processor changes the row of this step.
    fn applies(&self, p: &ProcessorParams, s: &ProcessorState<'_>) -> bool;
    /// Whether it can run on the device, so a step it applies to may still be reduced there.
    fn device_capable(&self) -> bool;
    /// Whether it reads or writes the whole vocabulary row.
    fn needs_full_row(&self) -> bool;
    /// Adjusts `logits` in place, recording each id it changes in `touched` first (a mask that
    /// only removes ids need not).
    fn apply(
        &self,
        logits: &mut [f32],
        touched: &mut Touched,
        p: &ProcessorParams,
        s: &ProcessorState<'_>,
    );
}

static PROCESSORS: Registry<dyn LogitsProcessor> = Registry::new(
    "logits_processor",
    &[
        &processors::LogitBias,
        &processors::RepetitionPenalty,
        &processors::PresenceFrequencyPenalty,
        &processors::MinTokens,
        &processors::GrammarMask,
    ],
);

/// The logits processors, in chain order.
pub fn registry() -> &'static Registry<dyn LogitsProcessor> {
    &PROCESSORS
}

/// The processors of a registry run in its order.
#[derive(Clone, Copy)]
pub struct ProcessorChain {
    reg: &'static Registry<dyn LogitsProcessor>,
}

impl std::fmt::Debug for ProcessorChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ProcessorChain")
            .field(&self.reg.names())
            .finish()
    }
}

impl ProcessorChain {
    /// The chain of [`registry`].
    pub fn standard() -> ProcessorChain {
        ProcessorChain::new(registry())
    }

    /// The chain of another registry (tests, conformance suites).
    pub fn new(reg: &'static Registry<dyn LogitsProcessor>) -> ProcessorChain {
        ProcessorChain { reg }
    }

    /// Runs every processor that applies, in registry order.
    pub fn apply(
        &self,
        logits: &mut [f32],
        touched: &mut Touched,
        p: &ProcessorParams,
        s: &ProcessorState<'_>,
    ) {
        for m in self.reg.iter() {
            if m.applies(p, s) {
                m.apply(logits, touched, p, s);
            }
        }
    }

    /// Whether the step may be reduced on the device as far as the processors go: every
    /// processor that applies to it is device-capable (none applying included).
    pub fn device_eligible(&self, p: &ProcessorParams, s: &ProcessorState<'_>) -> bool {
        self.reg
            .iter()
            .all(|m| m.device_capable() || !m.applies(p, s))
    }
}

#[cfg(test)]
mod tests;
