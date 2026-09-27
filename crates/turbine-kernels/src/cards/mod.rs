//! Card profiles (Phase 2m S-6, contract §24): one declarative Rust profile per card family —
//! capabilities, tuned thresholds and the preferred implementation order per op — in the
//! `card_profile` registry. The HIP backend picks the profile of the opened device
//! (`execution.card_profile: auto`) or the configured one; the kernel registry's selection
//! (Task 11) and the library's defaults (`turbine_ctx_set_profile`, Task 10) read it. Every
//! architecture a profile lists is built by `kernels/rocm/CMakeLists.txt`
//! (`kernels/rocm/cmake/card_profiles.cmake`, kept equal by a test).

use turbine_core::registry::{Module, Registry, UnknownModule};
use turbine_device::DeviceInfo;

use crate::OpKind;

pub mod conformance;
mod gfx1201;

pub use gfx1201::GFX1201;

/// A card family's declarative profile.
#[derive(Debug)]
pub struct CardProfile {
    pub name: &'static str,
    /// The support-matrix vendor column (`amd`).
    pub vendor: &'static str,
    /// Device architectures (`DeviceInfo::arch`) the profile describes.
    pub archs: &'static [&'static str],
    pub capabilities: CardCapabilities,
    pub thresholds: CardThresholds,
    /// Preferred implementation order per op; an op without an entry keeps library order.
    pub preferences: &'static [OpPreference],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CardCapabilities {
    /// Matrix instruction families (`wmma`, `mfma`, …).
    pub matrix_instructions: &'static [&'static str],
    pub bf16: bool,
    pub wave_size: u32,
    /// Local data share (shared memory) per workgroup, bytes.
    pub lds_bytes: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CardThresholds {
    /// Routed rows up to which `moe_experts` prefers the small-m kernel (the first row tier).
    pub moe_small_max_rows: u32,
    /// KV page sizes (`kv.block_tokens`) the preferred paged-attention path serves are
    /// multiples of this.
    pub paged_page_multiple: u32,
}

/// The preferred implementation order of one op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpPreference {
    pub op: OpKind,
    pub order: &'static [&'static str],
    /// Row-count tiers (`moe_experts`), each with its own order; empty = `order` for any rows.
    pub row_tiers: &'static [RowTierSpec],
}

/// One row-count tier: up to `max_rows` routed rows (`None` = no upper bound).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowTierSpec {
    pub max_rows: Option<u32>,
    pub order: &'static [&'static str],
}

impl CardProfile {
    /// The preference of `op`, if the profile has one.
    pub fn preference(&self, op: OpKind) -> Option<&OpPreference> {
        self.preferences.iter().find(|p| p.op == op)
    }
}

impl Module for CardProfile {
    fn name(&self) -> &'static str {
        self.name
    }
}

/// Why no card profile could be chosen.
#[derive(Debug, thiserror::Error)]
pub enum CardError {
    #[error("no card profile for device architecture {arch} (profiles: {profiles})")]
    NoProfile { arch: String, profiles: String },
    #[error(transparent)]
    Unknown(#[from] UnknownModule),
}

static CARD_PROFILES: Registry<CardProfile> = Registry::new("card_profile", &[&GFX1201]);

/// The registered card profiles: `gfx1201`.
pub fn registry() -> &'static Registry<CardProfile> {
    &CARD_PROFILES
}

/// The profile that lists `device.arch`.
pub fn profile_for(device: &DeviceInfo) -> Result<&'static CardProfile, CardError> {
    let arch = device.arch.as_deref().unwrap_or("unknown");
    registry()
        .iter()
        .find(|p| p.archs.contains(&arch))
        .ok_or_else(|| CardError::NoProfile {
            arch: arch.to_string(),
            profiles: registry().names().join(", "),
        })
}

/// `execution.card_profile` on `device`: `auto` → [`profile_for`] (reason `discovered arch
/// <arch>`), a name → that profile (reason `execution.card_profile`), even for a device of
/// another architecture (for experiments). Logs `module_selected` (point `card_profile`).
pub fn select(configured: &str, device: &DeviceInfo) -> Result<&'static CardProfile, CardError> {
    if configured == "auto" {
        let profile = profile_for(device)?;
        let arch = device.arch.as_deref().unwrap_or("unknown");
        return Ok(registry().select(profile.name, &format!("discovered arch {arch}"))?);
    }
    Ok(registry().select(configured, "execution.card_profile")?)
}

/// The architectures every registered profile lists, in registration order, each once.
pub fn profile_archs() -> Vec<&'static str> {
    let mut archs: Vec<&'static str> = Vec::new();
    for arch in registry().iter().flat_map(|p| p.archs.iter().copied()) {
        if !archs.contains(&arch) {
            archs.push(arch);
        }
    }
    archs
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use turbine_core::types::{DeviceId, MemoryKind, Vendor};
    use turbine_device::DeviceMemoryInfo;

    use super::*;

    fn amd_device(arch: &str) -> DeviceInfo {
        DeviceInfo {
            index: DeviceId(0),
            vendor: Vendor::Amd,
            vendor_index: 0,
            name: "test device".into(),
            uuid: None,
            pci_bus_id: None,
            arch: Some(arch.into()),
            driver_version: None,
            memory: DeviceMemoryInfo {
                kind: MemoryKind::Dedicated,
                total_bytes: 1 << 30,
                shared_with_host: false,
            },
        }
    }

    #[test]
    fn gfx1201_carries_every_threshold() {
        let p = &GFX1201;
        assert_eq!(p.name, "gfx1201");
        assert_eq!(p.vendor, "amd");
        assert_eq!(p.archs, ["gfx1201"]);
        assert_eq!(
            p.capabilities,
            CardCapabilities {
                matrix_instructions: &["wmma"],
                bf16: true,
                wave_size: 32,
                lds_bytes: 65536,
            }
        );
        assert_eq!(
            p.thresholds,
            CardThresholds {
                moe_small_max_rows: 512,
                paged_page_multiple: 128,
            }
        );
        let order = |op| p.preference(op).map(|pref| pref.order);
        assert_eq!(order(OpKind::Gemm), Some(&["hipblaslt"][..]));
        for op in [OpKind::AttentionPrefill, OpKind::AttentionDecode] {
            assert_eq!(order(op), Some(&["ck_tile_fmha_fwd"][..]), "{op}");
        }
        for op in [OpKind::Rmsnorm, OpKind::AddRmsnorm] {
            assert_eq!(
                order(op),
                Some(&["ck_tile_rmsnorm2d", "turbine_hip"][..]),
                "{op}"
            );
        }
        for op in [OpKind::AttentionPrefillPaged, OpKind::AttentionDecodePaged] {
            assert_eq!(
                order(op),
                Some(&["ck_tile_fmha_pagedkv", "turbine_hip"][..]),
                "{op}"
            );
        }
        let moe = p.preference(OpKind::MoeExperts).expect("moe_experts tiers");
        assert_eq!(
            moe.row_tiers,
            [
                RowTierSpec {
                    max_rows: Some(512),
                    order: &[
                        "turbine_hip_moe_small_m",
                        "turbine_hip_moe_wmma",
                        "hipblaslt_grouped",
                        "hipblaslt_per_expert",
                    ],
                },
                RowTierSpec {
                    max_rows: None,
                    order: &[
                        "turbine_hip_moe_wmma",
                        "hipblaslt_grouped",
                        "hipblaslt_per_expert",
                    ],
                },
            ]
        );
        assert_eq!(
            moe.row_tiers[0].max_rows,
            Some(p.thresholds.moe_small_max_rows)
        );
        for op in [
            OpKind::Rope,
            OpKind::SiluMul,
            OpKind::Embedding,
            OpKind::Add,
            OpKind::CopyBlocks,
            OpKind::MoeRoute,
            OpKind::LogitsReduce,
        ] {
            assert!(p.preference(op).is_none(), "{op} keeps library order");
        }
    }

    #[test]
    fn profile_for_device_and_refusal() {
        assert_eq!(profile_for(&amd_device("gfx1201")).unwrap().name, "gfx1201");
        let err = profile_for(&amd_device("gfx942")).unwrap_err();
        assert!(matches!(err, CardError::NoProfile { .. }));
        assert_eq!(
            err.to_string(),
            "no card profile for device architecture gfx942 (profiles: gfx1201)"
        );
        assert_eq!(
            select("auto", &amd_device("gfx1201")).unwrap().name,
            "gfx1201"
        );
        assert!(select("auto", &amd_device("gfx942")).is_err());
        // A configured profile is forced onto a device of another architecture.
        assert_eq!(
            select("gfx1201", &amd_device("gfx942")).unwrap().name,
            "gfx1201"
        );
        let unknown = select("gfx942", &amd_device("gfx1201")).unwrap_err();
        assert_eq!(
            unknown.to_string(),
            "card_profile: `gfx942` is not registered (registered: gfx1201)"
        );
    }

    /// `kernels/rocm/cmake/card_profiles.cmake` lists exactly the architectures of the
    /// registered profiles, so CMake builds every architecture a profile describes.
    #[test]
    fn cmake_lists_every_profile_arch() {
        let cmake = conformance::cmake_profile_archs("kernels/rocm/cmake/card_profiles.cmake")
            .expect("card_profiles.cmake");
        assert_eq!(cmake, profile_archs());
    }

    /// Every tuned GEMM table (`kernels/rocm/tuning/<arch>/gemm.tsv`, compiled into the HIP
    /// library by `kernels/rocm/cmake/gemm_table.py`) belongs to a registered profile's
    /// architecture, and its rows are well formed: positive n, k and m_max, trans_b 0 or 1, a
    /// `bf16` / `f32` output, a solution index and name (or -1 and `heuristic`), a mode
    /// (`invariant` or `speed`), m_max strictly ascending per shape.
    /// Breaks if a table is added for an architecture no profile describes (it would never be
    /// built) or a hand edit leaves a row the build refuses.
    #[test]
    fn tuned_gemm_tables_are_card_data() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/rocm/tuning");
        let archs = profile_archs();
        for entry in std::fs::read_dir(&dir).expect("kernels/rocm/tuning") {
            let entry = entry.expect("tuning entry");
            if !entry.file_type().expect("file type").is_dir() {
                continue;
            }
            let arch = entry.file_name().to_string_lossy().into_owned();
            assert!(
                archs.contains(&arch.as_str()),
                "kernels/rocm/tuning/{arch}: no card profile lists {arch} ({archs:?})"
            );
            let Ok(text) = std::fs::read_to_string(entry.path().join("gemm.tsv")) else {
                continue;
            };
            let mut last: Option<((i64, i64, i64, String), i64)> = None;
            for (i, line) in text.lines().enumerate() {
                if line.trim().is_empty() || line.starts_with('#') {
                    continue;
                }
                let at = format!("{arch}/gemm.tsv:{}", i + 1);
                let c: Vec<&str> = line.split('\t').collect();
                assert!(c.len() >= 8, "{at}: {} columns", c.len());
                assert!(
                    matches!(c[7], "invariant" | "speed"),
                    "{at}: mode {} (invariant or speed)",
                    c[7]
                );
                let num = |j: usize| {
                    c[j].parse::<i64>()
                        .unwrap_or_else(|e| panic!("{at}: column {j}: {e}"))
                };
                let (n, k, trans_b, m_max, index) = (num(0), num(1), num(2), num(4), num(5));
                assert_eq!(
                    c[6] == "heuristic",
                    index == -1,
                    "{at}: index -1 goes with `heuristic`"
                );
                assert!(index >= -1, "{at}: solution index");
                assert!(n > 0 && k > 0 && m_max > 0, "{at}: n, k, m_max positive");
                assert!(trans_b <= 1, "{at}: trans_b");
                assert!(matches!(c[3], "bf16" | "f32"), "{at}: c_dtype {}", c[3]);
                assert!(
                    !c[6].is_empty()
                        && c[6]
                            .chars()
                            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_'),
                    "{at}: solution name"
                );
                let shape = (n, k, trans_b, c[3].to_string());
                if let Some((prev, prev_m)) = &last
                    && *prev == shape
                {
                    assert!(
                        m_max > *prev_m,
                        "{at}: m_max not ascending within the shape"
                    );
                }
                last = Some((shape, m_max));
            }
        }
    }
}
