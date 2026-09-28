//! The `card_profile` conformance suite (Phase 2m S-13): run over a registry — never a
//! hand-written list — by `registries::registry_conformance::card_profiles`, so a profile
//! registered without passing it fails `cargo test --workspace`.

use std::path::Path;

use turbine_core::registry::{Registry, conformance};

use super::CardProfile;
use crate::OpKind;

/// The file listing the architectures a vendor's kernel library is built for: `amd` →
/// `kernels/rocm/cmake/card_profiles.cmake` (`TURBINE_PROFILE_ARCHS`).
fn build_list_file(vendor: &str) -> Option<&'static str> {
    match vendor {
        "amd" => Some("kernels/rocm/cmake/card_profiles.cmake"),
        _ => None,
    }
}

/// The `TURBINE_PROFILE_ARCHS` list of the CMake file `rel` (relative to the repository root),
/// comments skipped.
pub fn cmake_profile_archs(rel: &str) -> Result<Vec<String>, String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{rel}: {e}"))?;
    let body = text
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join(" ");
    let key = "set(TURBINE_PROFILE_ARCHS";
    let start = body
        .find(key)
        .ok_or_else(|| format!("{rel}: no {key} …)"))?;
    let rest = &body[start + key.len()..];
    let end = rest
        .find(')')
        .ok_or_else(|| format!("{rel}: no closing parenthesis"))?;
    Ok(rest[..end].split_whitespace().map(str::to_string).collect())
}

/// Runs every check over every profile of `reg`; `Err` lists each broken check as
/// `<profile>: <check>: <detail>`.
///
/// Per profile: `vendor` (a registered execution backend's vendor), `archs` (at least one, none
/// listed by another profile of `reg`, each in the vendor kernel library's build list —
/// `kernels/rocm/cmake/card_profiles.cmake` for `amd`), `thresholds` (every threshold and
/// capability number non-zero), `preferences` (each op at most once, every order non-empty
/// with distinct names), `tiers` (row-tier bounds non-zero and ascending, exactly one open tier
/// and it comes last, the `moe_experts` first tier bounded by `moe_small_max_rows` — the bound
/// the kernel library's own default uses).
pub fn cards_suite(reg: &Registry<CardProfile>) -> Result<(), Vec<String>> {
    let mut failures = Vec::new();
    if let Err(e) = conformance::check(reg) {
        failures.push(format!("registry: {e}"));
    }
    let vendors: Vec<&str> = crate::backends::registry()
        .iter()
        .map(|b| b.vendor())
        .collect();
    let mut claimed: Vec<(&str, &str)> = Vec::new();
    for p in reg.iter() {
        let mut fail = |check: &str, detail: String| {
            failures.push(format!("{}: {check}: {detail}", p.name));
        };
        if !vendors.contains(&p.vendor) {
            fail(
                "vendor",
                format!("{} is not a registered backend's vendor", p.vendor),
            );
        }
        if p.archs.is_empty() {
            fail("archs", "lists no architecture".into());
        }
        let build_list = match build_list_file(p.vendor) {
            Some(file) => cmake_profile_archs(file).map(|list| (file, list)),
            None => Err(format!(
                "no kernel-library build list for vendor {}",
                p.vendor
            )),
        };
        // A missing or unreadable build list is one failure of the profile, not one per arch.
        if let Err(e) = &build_list {
            fail("archs", e.clone());
        }
        for arch in p.archs {
            if let Some((_, other)) = claimed.iter().find(|(a, _)| a == arch) {
                fail("archs", format!("{arch} is also listed by {other}"));
            }
            claimed.push((arch, p.name));
            if let Ok((file, list)) = &build_list
                && !list.iter().any(|a| a == arch)
            {
                fail("archs", format!("{arch} is not built: missing from {file}"));
            }
        }
        let t = &p.thresholds;
        let c = &p.capabilities;
        for (what, value) in [
            ("moe_small_max_rows", t.moe_small_max_rows),
            ("paged_page_multiple", t.paged_page_multiple),
            ("wave_size", c.wave_size),
            ("lds_bytes", c.lds_bytes),
        ] {
            if value == 0 {
                fail("thresholds", format!("{what} is 0"));
            }
        }
        for (i, pref) in p.preferences.iter().enumerate() {
            let op = pref.op;
            if p.preferences[..i].iter().any(|q| q.op == op) {
                fail("preferences", format!("{op} is listed twice"));
            }
            let mut orders = vec![pref.order];
            orders.extend(pref.row_tiers.iter().map(|t| t.order));
            for order in orders {
                let repeated = order
                    .iter()
                    .enumerate()
                    .any(|(j, n)| n.is_empty() || order[..j].contains(n));
                if order.is_empty() || repeated {
                    fail(
                        "preferences",
                        format!("{op}: empty order or empty / repeated name in {order:?}"),
                    );
                }
            }
            if pref.row_tiers.is_empty() {
                continue;
            }
            let bounds: Vec<Option<u32>> = pref.row_tiers.iter().map(|t| t.max_rows).collect();
            let open = bounds.iter().filter(|b| b.is_none()).count();
            let bounded: Vec<u32> = bounds.iter().flatten().copied().collect();
            let ascending = bounded.windows(2).all(|w| w[0] < w[1]);
            if open != 1 || bounds.last() != Some(&None) || !ascending || bounded.contains(&0) {
                fail(
                    "tiers",
                    format!("{op}: bounds {bounds:?} must ascend from > 0 to one open tier"),
                );
            }
            if op == OpKind::MoeExperts && bounds.first() != Some(&Some(t.moe_small_max_rows)) {
                fail(
                    "tiers",
                    format!(
                        "moe_experts first tier {:?} is not moe_small_max_rows {}",
                        bounds.first(),
                        t.moe_small_max_rows
                    ),
                );
            }
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::{CardCapabilities, CardThresholds, GFX1201, OpPreference, RowTierSpec};

    /// `gfx1201` again under another name, its `moe_experts` tiers out of order and an
    /// architecture no CMake list builds.
    static BROKEN: CardProfile = CardProfile {
        name: "broken",
        vendor: "amd",
        archs: &["gfx9999"],
        capabilities: CardCapabilities {
            matrix_instructions: &["wmma"],
            bf16: true,
            wave_size: 32,
            lds_bytes: 65536,
        },
        thresholds: CardThresholds {
            moe_small_max_rows: 512,
            paged_page_multiple: 128,
        },
        preferences: &[OpPreference {
            op: OpKind::MoeExperts,
            order: &["turbine_hip_moe_wmma"],
            row_tiers: &[
                RowTierSpec {
                    max_rows: None,
                    order: &["turbine_hip_moe_wmma"],
                },
                RowTierSpec {
                    max_rows: Some(512),
                    order: &["turbine_hip_moe_small_m"],
                },
            ],
        }],
    };

    static WITH_BROKEN: Registry<CardProfile> = Registry::new("card_profile", &[&GFX1201, &BROKEN]);

    /// A profile whose architecture CMake does not build and whose open tier comes first fails
    /// `archs` and `tiers` under its own name; `gfx1201` passes. Breaks if the suite stops
    /// reading the CMake list or checking tier order, or iterates a fixed list.
    #[test]
    fn rejects_broken_profile() {
        let failures = cards_suite(&WITH_BROKEN).unwrap_err();
        let checks: Vec<&str> = failures
            .iter()
            .map(|f| f.split(": ").take(2).last().unwrap())
            .collect();
        assert!(
            failures.iter().all(|f| f.starts_with("broken: ")),
            "{failures:#?}"
        );
        assert_eq!(checks, ["archs", "tiers", "tiers"], "{failures:#?}");
    }

    /// A vendor with no kernel-library build list, two architectures, nothing else wrong.
    static NO_BUILD_LIST: CardProfile = CardProfile {
        name: "no_build_list",
        vendor: "cpu",
        archs: &["cpu_a", "cpu_b"],
        capabilities: CardCapabilities {
            matrix_instructions: &[],
            bf16: true,
            wave_size: 1,
            lds_bytes: 1,
        },
        thresholds: CardThresholds {
            moe_small_max_rows: 1,
            paged_page_multiple: 1,
        },
        preferences: &[],
    };

    static WITH_NO_BUILD_LIST: Registry<CardProfile> =
        Registry::new("card_profile", &[&NO_BUILD_LIST]);

    /// The missing build list is one failure per profile, not one per architecture
    /// (Scout 8f4b1230).
    #[test]
    fn missing_build_list_reported_once() {
        let failures = cards_suite(&WITH_NO_BUILD_LIST).unwrap_err();
        assert_eq!(
            failures,
            ["no_build_list: archs: no kernel-library build list for vendor cpu"],
            "{failures:#?}"
        );
    }
}
