//! Shared vocabulary types (contract §3.4).

use serde::{Deserialize, Serialize};

/// Global device index: the Phase 0 inventory index, stable for the process lifetime.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DeviceId(pub u32);

/// GPU vendor. Serialized as `"nvidia"` / `"amd"`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Vendor {
    Nvidia,
    Amd,
}

impl Vendor {
    /// Metric label / log value.
    pub fn as_str(self) -> &'static str {
        match self {
            Vendor::Nvidia => "nvidia",
            Vendor::Amd => "amd",
        }
    }
}

/// How a device's memory relates to host memory. Serialized as `"dedicated"` / `"unified"`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MemoryKind {
    Dedicated,
    Unified,
}
