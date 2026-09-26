//! Conformance tests over every registry of this crate (Phase 2m S-1, S-13): each registry
//! passes `turbine_core::registry::conformance::check`, and each registered module passes its
//! extension point's shared suite ([`crate::conformance`]), run over the registry itself — a
//! module added to a registry is checked without touching these tests.

#[cfg(test)]
mod registry_conformance {
    use turbine_core::registry::Registry;

    use crate::conformance::{
        ConformanceFailure, families_suite, formats_suite, processors_suite, weights_suite,
    };

    /// Panics listing every failure of a suite.
    fn passes(result: Result<(), Vec<ConformanceFailure>>) {
        if let Err(failures) = result {
            let lines: Vec<String> = failures.iter().map(ToString::to_string).collect();
            panic!(
                "{} conformance failures:\n{}",
                lines.len(),
                lines.join("\n")
            );
        }
    }

    /// The names every registry lists, in registration order (the chain order for the logits
    /// processors, the error-message order for the families).
    fn names_are<T: ?Sized + turbine_core::registry::Module>(
        reg: &Registry<T>,
        point: &str,
        names: &[&str],
    ) {
        assert_eq!(reg.point(), point);
        assert_eq!(reg.names(), names);
        for name in names {
            assert!(reg.get(name).is_some(), "{point}: {name}");
        }
    }

    /// The logits-processor chain in main's order, and every processor passes
    /// `processors_suite`. Breaks if a processor is registered twice, out of order, applies to
    /// requests that do not ask for it, or claims the device without matching it.
    #[test]
    fn logits_processors() {
        let reg = crate::sampling::registry();
        names_are(
            reg,
            "logits_processor",
            &[
                "logit_bias",
                "repetition_penalty",
                "presence_frequency_penalty",
                "min_tokens",
                "grammar_mask",
            ],
        );
        // As registered today: every processor needs the whole row on the host.
        assert!(
            reg.iter()
                .all(|m| !m.device_capable() && m.needs_full_row())
        );
        passes(processors_suite(reg));
    }

    /// Every family passes `families_suite` (tiny checkpoint vs the naive decoder, batching,
    /// chunking, paging and fusion equivalences on the CPU provider).
    #[test]
    fn families() {
        let reg = crate::families::registry();
        names_are(
            reg,
            "model_family",
            &["llama", "olmoe", "qwen3", "qwen3_moe", "mistral", "mixtral"],
        );
        passes(families_suite(reg));
    }

    /// Every tool format passes `formats_suite` (render, grammar, parse, round trip through the
    /// constrained matcher on the tiny tokenizer, opening).
    #[test]
    fn tool_formats() {
        let reg = crate::formats::registry();
        names_are(reg, "tool_format", &["llama3_json", "hermes", "mistral"]);
        passes(formats_suite(reg));
    }

    /// Every weight format passes `weights_suite`.
    #[test]
    fn weight_formats() {
        let reg = crate::weights::registry();
        names_are(reg, "weight_format", &["bf16"]);
        passes(weights_suite(reg));
    }
}
