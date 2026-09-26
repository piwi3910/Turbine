//! Conformance tests over every registry of this crate (Phase 2m S-1, S-13): each registry
//! passes `turbine_core::registry::conformance::check`, and each registered module passes its
//! extension point's shared suite.

#[cfg(test)]
mod registry_conformance {
    use std::collections::HashMap;

    use turbine_core::registry::conformance::{self, check};
    use turbine_core::request::SamplingParams;

    use crate::sampling::{self, ProcessorChain, ProcessorParams, ProcessorState, Touched};
    use crate::structured::TokenMask;

    /// The logits-processor registry is well formed, lists the chain in main's order, and every
    /// registered processor passes the shared suite: it does not apply to a neutral request (so
    /// the chain leaves such a row and the touched record alone), it applies when its request
    /// field is set, and one that cannot run on the device keeps an applying step on the host.
    /// Breaks if a processor is registered twice, out of order, or applies to requests that do
    /// not ask for it.
    #[test]
    fn logits_processors() {
        let reg = sampling::registry();
        conformance::check(reg).expect("logits_processor registry");
        assert_eq!(reg.point(), "logits_processor");
        assert_eq!(
            reg.names(),
            [
                "logit_bias",
                "repetition_penalty",
                "presence_frequency_penalty",
                "min_tokens",
                "grammar_mask"
            ]
        );
        let neutral = ProcessorParams::new(&SamplingParams::default());
        let counts: HashMap<u32, u32> = [(4, 2)].into_iter().collect();
        let quiet = ProcessorState {
            prompt_tokens: &[1, 2],
            counts: &counts,
            step: 3,
            eos_token_ids: &[5],
            mask: None,
        };
        let busy_params = ProcessorParams::new(&SamplingParams {
            logit_bias: vec![(3, 1.0)],
            repetition_penalty: 1.2,
            presence_penalty: 0.5,
            frequency_penalty: 0.5,
            min_tokens: 8,
            ..SamplingParams::default()
        });
        let mask = TokenMask::new_all(8);
        let busy = ProcessorState {
            mask: Some(&mask),
            ..quiet
        };
        let chain = ProcessorChain::standard();
        for m in reg.iter() {
            let name = m.name();
            assert!(
                !m.applies(&neutral, &quiet),
                "{name} applies to a neutral request"
            );
            assert!(m.applies(&busy_params, &busy), "{name} never applies");
            if !m.device_capable() {
                assert!(!chain.device_eligible(&busy_params, &busy), "{name}");
            }
            // As registered today: every processor needs the whole row on the host.
            assert!(!m.device_capable() && m.needs_full_row(), "{name}");
        }
        let row: Vec<f32> = (0..8).map(|i| i as f32 - 3.5).collect();
        let mut logits = row.clone();
        let mut touched = Touched::default();
        chain.apply(&mut logits, &mut touched, &neutral, &quiet);
        assert_eq!(logits, row);
        assert!(touched.originals().is_empty());
        assert!(chain.device_eligible(&neutral, &quiet));
    }

    #[test]
    fn families() {
        let reg = crate::families::registry();
        check(reg).unwrap();
        // Every family serves at least one HF name, and no two families claim the same one.
        let mut seen = std::collections::HashMap::new();
        for family in reg.iter() {
            assert!(!family.hf_architectures().is_empty(), "{}", family.name());
            for hf in family.hf_architectures() {
                if let Some(other) = seen.insert(*hf, family.name()) {
                    panic!("{hf} is claimed by {other} and {}", family.name());
                }
            }
        }
    }

    #[test]
    fn tool_formats() {
        let reg = crate::formats::registry();
        check(reg).unwrap();
        // Every special token a format names is non-empty and named once.
        for format in reg.iter() {
            let texts: Vec<&str> = format.special_tokens().iter().map(|t| t.text).collect();
            for (i, text) in texts.iter().enumerate() {
                assert!(!text.is_empty(), "{}", format.name());
                assert!(
                    !texts[..i].contains(text),
                    "{}: {text} twice",
                    format.name()
                );
            }
        }
    }

    #[test]
    fn weight_formats() {
        check(crate::weights::registry()).unwrap();
    }
}
