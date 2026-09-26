//! Conformance suites (Phase 2m S-13, contract §24): one shared suite per extension point of
//! this crate, each run over a [`Registry`] — never over a hand-written list — so a module
//! registered without passing its suite fails `cargo test --workspace`
//! (`registries::registry_conformance`, and `tests/conformance.rs` with deliberately broken
//! toy modules).
//!
//! | Extension point    | Suite               | Checks per registered module                                                                                   |
//! | ------------------ | ------------------- | -------------------------------------------------------------------------------------------------------------- |
//! | `model_family`     | [`families_suite`]  | tiny checkpoint on the CPU provider vs the naive decoder; ragged batch, chunked prefill, page size, fusion       |
//! | `tool_format`      | [`formats_suite`]   | special tokens, render, grammar for `auto` / `required` / named, parse of its sample call, round trip, opening |
//! | `weight_format`    | [`weights_suite`]   | bytes per parameter, a registered family's tiny checkpoint loads in the format                                 |
//! | `logits_processor` | [`processors_suite`] | neutral requests untouched, applies when asked, host == device (`logits_reduce` CPU reference) when device-capable |
//!
//! The suites of the other crates' extension points follow the same shape:
//! `turbine_scheduler::policy::conformance::policies_suite` (simulator invariants),
//! `turbine_kernels::backends::conformance::backends_suite`,
//! `turbine_kernels::cards::conformance::cards_suite`; kernel implementations are checked
//! against the CPU reference on the lab (`hip_ops every_implementation_matches_cpu`).
//!
//! Every suite starts with [`turbine_core::registry::conformance::check`] (unique, well-formed
//! names that lookup returns) and collects every failure instead of stopping at the first; a
//! check that panics is reported as a failure of that check.

use std::panic::{AssertUnwindSafe, catch_unwind};

use turbine_core::registry::{Module, Registry};

mod families;
mod formats;
mod processors;
mod weights;

pub use families::families_suite;
pub use formats::{feed_text, fixture_tools, formats_suite};
pub use processors::processors_suite;
pub use weights::weights_suite;

/// One broken check of one module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConformanceFailure {
    /// The extension point (`model_family`, `tool_format`, …).
    pub point: &'static str,
    /// The module's registered name (`registry` for a failure of the registry itself).
    pub module: &'static str,
    /// The suite's check (`load`, `parse`, `device`, …).
    pub check: &'static str,
    pub detail: String,
}

impl std::fmt::Display for ConformanceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {}: {}: {}",
            self.point, self.module, self.check, self.detail
        )
    }
}

/// The failures of one suite run.
pub(crate) struct Report {
    point: &'static str,
    failures: Vec<ConformanceFailure>,
}

impl Report {
    /// A report for `reg`, starting with the registry's own checks.
    pub(crate) fn new<T: ?Sized + Module>(reg: &Registry<T>) -> Report {
        let mut report = Report {
            point: reg.point(),
            failures: Vec::new(),
        };
        report.check("registry", "registry", || {
            turbine_core::registry::conformance::check(reg)
        });
        report
    }

    /// Runs `f` as check `check` of `module`; an `Err` or a panic is recorded as a failure.
    /// Returns whether the check passed.
    pub(crate) fn check(
        &mut self,
        module: &'static str,
        check: &'static str,
        f: impl FnOnce() -> Result<(), String>,
    ) -> bool {
        let detail = match catch_unwind(AssertUnwindSafe(f)) {
            Ok(Ok(())) => return true,
            Ok(Err(detail)) => detail,
            Err(panic) => {
                let message = panic
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "non-string panic".into());
                format!("panicked: {message}")
            }
        };
        self.failures.push(ConformanceFailure {
            point: self.point,
            module,
            check,
            detail,
        });
        false
    }

    pub(crate) fn finish(self) -> Result<(), Vec<ConformanceFailure>> {
        if self.failures.is_empty() {
            Ok(())
        } else {
            Err(self.failures)
        }
    }
}

/// `Err(detail)` unless `cond`.
pub(crate) fn ensure(cond: bool, detail: impl FnOnce() -> String) -> Result<(), String> {
    if cond { Ok(()) } else { Err(detail()) }
}

/// Greedy argmax, ties to the lower id.
pub(crate) fn argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as u32
}

/// The largest |a − b| over two rows of equal length.
pub(crate) fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "row lengths");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Whether two rows are equal bit for bit.
pub(crate) fn bitwise_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct A;
    impl Module for A {
        fn name(&self) -> &'static str {
            "a"
        }
    }
    static DUP: Registry<A> = Registry::new("toy", &[&A, &A]);

    /// A report records `Err` results and panics under the check's name, and a registry with
    /// a duplicate name fails the `registry` check. Breaks if a panic escapes the report or a
    /// failure is dropped.
    #[test]
    fn report_collects_errors_and_panics() {
        let mut r = Report::new(&DUP);
        assert!(r.check("a", "ok", || Ok(())));
        assert!(!r.check("a", "err", || Err("bad".into())));
        assert!(!r.check("a", "panic", || panic!("boom {}", 1)));
        let failures = r.finish().unwrap_err();
        let got: Vec<(&str, &str)> = failures.iter().map(|f| (f.module, f.check)).collect();
        assert_eq!(
            got,
            [("registry", "registry"), ("a", "err"), ("a", "panic")]
        );
        assert!(failures[2].detail.contains("boom 1"), "{}", failures[2]);
        assert!(failures.iter().all(|f| f.point == "toy"));
    }
}
