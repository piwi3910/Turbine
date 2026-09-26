use std::collections::HashMap;

use turbine_core::registry::{Module, Registry};
use turbine_core::request::SamplingParams;

use super::sampler::{DEVICE_MAX_TOP_LOGPROBS, Sampler};
use super::*;
use crate::executor::logits::MAX_TOP_N;
use crate::structured::TokenMask;

/// Main's device fast-path eligibility (`Sampler::device_request_ahead` before Phase 2m) plus
/// the engine's rule that a constrained choice keeps its whole row: the hand-written
/// conjunction the derived eligibility must reproduce.
fn main_eligible(p: &SamplingParams, step: usize, constrained: bool) -> bool {
    let top_k = usize::try_from(p.top_k).ok().filter(|&k| k > 0);
    let top_logprobs = p.logprobs.map_or(0, |n| n as usize);
    let eligible = p.logit_bias.is_empty()
        && p.presence_penalty == 0.0
        && p.frequency_penalty == 0.0
        && p.repetition_penalty == 1.0
        && step as u64 >= u64::from(p.min_tokens)
        && top_logprobs <= DEVICE_MAX_TOP_LOGPROBS;
    let greedy = p.temperature <= 0.0;
    eligible && !(!greedy && top_k.is_some_and(|k| k > MAX_TOP_N)) && !constrained
}

fn sorted_distinct(ids: &[u32]) -> Vec<u32> {
    let mut ids = ids.to_vec();
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Main's `Sampler::adjust`, copied verbatim over explicit inputs: returns the raw value of
/// every id it changed before the mask.
fn main_adjust(
    logits: &mut [f32],
    p: &SamplingParams,
    prompt: &[u32],
    generated: &[u32],
    eos: &[u32],
    mask: Option<&TokenMask>,
) -> HashMap<u32, f32> {
    let prompt = if p.repetition_penalty != 1.0 {
        sorted_distinct(prompt)
    } else {
        Vec::new()
    };
    let eos = sorted_distinct(eos);
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for &t in generated {
        *counts.entry(t).or_insert(0) += 1;
    }
    let mut originals = HashMap::new();
    let mut touch = |logits: &[f32], id: u32| -> Option<usize> {
        let i = id as usize;
        let v = *logits.get(i)?;
        originals.entry(id).or_insert(v);
        Some(i)
    };
    for &(id, bias) in &p.logit_bias {
        if let Some(i) = touch(logits, id) {
            logits[i] += bias;
        }
    }
    if p.repetition_penalty != 1.0 {
        let r = p.repetition_penalty;
        let seen = prompt.iter().copied().chain(
            counts
                .keys()
                .copied()
                .filter(|id| prompt.binary_search(id).is_err()),
        );
        for id in seen {
            if let Some(i) = touch(logits, id) {
                let v = logits[i];
                logits[i] = if v > 0.0 { v / r } else { v * r };
            }
        }
    }
    if p.presence_penalty != 0.0 || p.frequency_penalty != 0.0 {
        for (&id, &count) in &counts {
            if let Some(i) = touch(logits, id) {
                logits[i] -= p.frequency_penalty * count as f32 + p.presence_penalty;
            }
        }
    }
    if (generated.len() as u64) < u64::from(p.min_tokens) {
        for &id in &eos {
            if let Some(i) = touch(logits, id) {
                logits[i] = f32::NEG_INFINITY;
            }
        }
    }
    if let Some(mask) = mask {
        mask.apply(logits);
    }
    originals
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The chain in registry order gives bit-identical logits and the same raw originals as main's
/// `adjust`, over a row where every processor applies and overlaps (the biased id 5 is also a
/// prompt id and an EOS id, the mask cuts ids ≥ 10 including a generated one), with and
/// without the mask, before and after `min_tokens`; and `Sampler::sample` leaves the same row
/// behind. Breaks if a processor moves in the chain, a processor records a value after another
/// changed it, or the sampler hands the chain the wrong step, history or prompt.
#[test]
fn chain_order_is_mains() {
    let p = SamplingParams {
        temperature: 0.0,
        logit_bias: vec![(5, 3.0), (40, 1.0), (12, -0.5)],
        repetition_penalty: 1.3,
        presence_penalty: 0.5,
        frequency_penalty: 0.2,
        min_tokens: 4,
        seed: Some(1),
        ..SamplingParams::default()
    };
    let prompt = [2u32, 5, 9, 2, 12];
    let eos = [5u32, 11, 3, 5];
    let mut mask = TokenMask::new_none(16);
    for id in 0..10 {
        mask.allow(id);
    }
    let row: Vec<f32> = (0..16).map(|i| (i as f32 * 0.77).sin() * 3.0).collect();
    let chain = ProcessorChain::standard();
    let params = ProcessorParams::new(&p);
    let mut checked = 0;
    for generated in [vec![4u32], vec![4, 12, 4, 7, 0]] {
        for m in [Some(&mask), None] {
            let mut want = row.clone();
            let originals = main_adjust(&mut want, &p, &prompt, &generated, &eos, m);
            assert!(originals.len() >= 5, "every processor touched ids");

            let mut counts = HashMap::new();
            for &t in &generated {
                *counts.entry(t).or_insert(0) += 1;
            }
            let prompt_sorted = sorted_distinct(&prompt);
            let eos_sorted = sorted_distinct(&eos);
            let state = ProcessorState {
                prompt_tokens: &prompt_sorted,
                counts: &counts,
                step: generated.len(),
                eos_token_ids: &eos_sorted,
                mask: m,
            };
            let mut got = row.clone();
            let mut touched = Touched::default();
            chain.apply(&mut got, &mut touched, &params, &state);
            assert_eq!(
                bits(&got),
                bits(&want),
                "{generated:?} mask {}",
                m.is_some()
            );
            assert_eq!(touched.originals(), &originals);

            let mut s = Sampler::new(&p, &prompt, &eos);
            for &t in &generated {
                s.observe(t);
            }
            let mut sampled = row.clone();
            s.sample(&mut sampled, m);
            assert_eq!(bits(&sampled), bits(&want), "sampler {generated:?}");
            checked += 1;
        }
    }
    assert_eq!(checked, 4);
}

/// Phase 2m S-8: the derived device fast-path eligibility equals main's hand-written rule for
/// every combination of the request fields that enter it, at the steps before and after
/// `min_tokens` (reached by observing, or by asking one step ahead), for seeded and unseeded,
/// constrained and unconstrained choices; an ineligible step draws nothing.
#[test]
fn device_eligibility_matches_main() {
    let mut n = 0;
    let mut eligible = 0;
    for lb in [vec![], vec![(3u32, 1.0f32)]] {
        for pres in [0.0, 0.5] {
            for freq in [0.0, 0.2] {
                for rep in [1.0, 1.3] {
                    for min_tokens in [0, 3] {
                        for logprobs in [None, Some(0), Some(5), Some(21)] {
                            for top_k in [-1, 20, 100] {
                                for temperature in [0.0, 0.8] {
                                    for seed in [Some(1), None] {
                                        for constrained in [false, true] {
                                            for (observed, ahead) in [(0, 0), (5, 0), (4, 1)] {
                                                let p = SamplingParams {
                                                    temperature,
                                                    top_p: 0.9,
                                                    top_k,
                                                    seed,
                                                    presence_penalty: pres,
                                                    frequency_penalty: freq,
                                                    repetition_penalty: rep,
                                                    logit_bias: lb.clone(),
                                                    min_tokens,
                                                    logprobs,
                                                };
                                                let mut s = Sampler::new(&p, &[1, 2], &[7]);
                                                for t in 0..observed {
                                                    s.observe(t);
                                                }
                                                let before = s.state();
                                                let got = s
                                                    .device_request_ahead(ahead, constrained)
                                                    .is_some();
                                                let want = main_eligible(
                                                    &p,
                                                    observed as usize + ahead,
                                                    constrained,
                                                );
                                                assert_eq!(
                                                    got, want,
                                                    "{p:?} observed {observed} ahead {ahead} \
                                                     constrained {constrained}"
                                                );
                                                if !got {
                                                    assert_eq!(s.state(), before, "{p:?}");
                                                }
                                                n += 1;
                                                eligible += usize::from(got);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(n, 2 * 2 * 2 * 2 * 2 * 4 * 3 * 2 * 2 * 2 * 3);
    assert!(eligible > 0 && eligible < n);
}

/// A processor that runs on the device (none is registered today).
struct DeviceBias;

impl Module for DeviceBias {
    fn name(&self) -> &'static str {
        "device_bias"
    }
}

impl LogitsProcessor for DeviceBias {
    fn applies(&self, p: &ProcessorParams, _: &ProcessorState<'_>) -> bool {
        !p.logit_bias.is_empty()
    }
    fn device_capable(&self) -> bool {
        true
    }
    fn needs_full_row(&self) -> bool {
        false
    }
    fn apply(&self, _: &mut [f32], _: &mut Touched, _: &ProcessorParams, _: &ProcessorState<'_>) {}
}

static WITH_DEVICE: Registry<dyn LogitsProcessor> =
    Registry::new("logits_processor", &[&DeviceBias, &processors::MinTokens]);

/// Eligibility is derived from the chain: an applying device-capable processor keeps the step
/// on the device, an applying host-only one does not, and a processor that does not apply
/// never counts. Breaks if eligibility is hand-written again instead of read from the chain.
#[test]
fn eligibility_is_derived_from_the_chain() {
    let chain = ProcessorChain::new(&WITH_DEVICE);
    let counts = HashMap::new();
    let state = |step| ProcessorState {
        prompt_tokens: &[],
        counts: &counts,
        step,
        eos_token_ids: &[2],
        mask: None,
    };
    let biased = ProcessorParams::new(&SamplingParams {
        logit_bias: vec![(1, 1.0)],
        min_tokens: 2,
        ..SamplingParams::default()
    });
    assert!(
        chain.device_eligible(&biased, &state(2)),
        "device bias only"
    );
    assert!(!chain.device_eligible(&biased, &state(1)), "min_tokens");
    assert!(ProcessorChain::standard().device_eligible(&ProcessorParams::default(), &state(0)));
    assert!(!ProcessorChain::standard().device_eligible(&biased, &state(2)));
}

/// `Touched` keeps the first raw value of each id and ignores ids past the row.
#[test]
fn touched_keeps_the_first_raw_value() {
    let mut logits = vec![1.0f32, 2.0];
    let mut t = Touched::default();
    assert_eq!(t.touch(&logits, 1), Some(1));
    logits[1] = 5.0;
    assert_eq!(t.touch(&logits, 1), Some(1));
    assert_eq!(t.touch(&logits, 2), None);
    assert_eq!(t.originals(), &HashMap::from([(1, 2.0)]));
    t.clear();
    assert!(t.originals().is_empty());
}
