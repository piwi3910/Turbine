//! Conformance over every registry of this crate (Phase 2m S-1, S-13): each registry passes
//! `turbine_core::registry::conformance::check`, and each registered module passes its
//! extension point's shared suite, run over the registry itself — a module added to a registry
//! is checked without touching these tests:
//!
//! - `execution_backend`: [`crate::backends::conformance::backends_suite`];
//! - `card_profile`: [`crate::cards::conformance::cards_suite`];
//! - kernel implementations (what a kernel library enumerates through ABI v2.4): every
//!   implementation of every op against the `cpu-reference` provider under that op's
//!   tolerance, on the lab — `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops
//!   -- every_implementation_matches_cpu` — since it needs the device. It enumerates the
//!   library, so a new implementation is covered without a test change.

#[cfg(test)]
mod registry_conformance {
    use crate::backends::conformance::backends_suite;
    use crate::cards::conformance::cards_suite;

    /// Every backend passes `backends_suite`; `cpu` and `hip` are registered, in that order.
    #[test]
    fn backends() {
        let reg = crate::backends::registry();
        assert_eq!(reg.point(), "execution_backend");
        assert_eq!(reg.names(), ["cpu", "hip"]);
        if let Err(failures) = backends_suite(reg) {
            panic!("backend conformance failures:\n{}", failures.join("\n"));
        }
    }

    /// Every card profile passes `cards_suite`; `gfx1201` is registered.
    #[test]
    fn card_profiles() {
        let reg = crate::cards::registry();
        assert_eq!(reg.point(), "card_profile");
        assert_eq!(reg.names(), ["gfx1201"]);
        if let Err(failures) = cards_suite(reg) {
            panic!(
                "card profile conformance failures:\n{}",
                failures.join("\n")
            );
        }
    }
}
