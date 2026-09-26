//! Block copies between local tiers (P4 S-6): the copy paths and their metric labels.

use crate::tier::TierId;

/// One local copy direction; label of `turbine_kv_transfer_*{path}` (CONFLICT C-9: local tier
/// copies only). `L0ToL2` / `L2ToL0` exist only on unified-memory devices, where L1 is disabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransferPath {
    L0ToL1,
    L1ToL0,
    L1ToL2,
    L2ToL1,
    L0ToL2,
    L2ToL0,
}

impl TransferPath {
    pub const ALL: [TransferPath; 6] = [
        TransferPath::L0ToL1,
        TransferPath::L1ToL0,
        TransferPath::L1ToL2,
        TransferPath::L2ToL1,
        TransferPath::L0ToL2,
        TransferPath::L2ToL0,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            TransferPath::L0ToL1 => "l0_to_l1",
            TransferPath::L1ToL0 => "l1_to_l0",
            TransferPath::L1ToL2 => "l1_to_l2",
            TransferPath::L2ToL1 => "l2_to_l1",
            TransferPath::L0ToL2 => "l0_to_l2",
            TransferPath::L2ToL0 => "l2_to_l0",
        }
    }

    /// The path copying a block from `from` to `to`; `None` for a non-local or same-tier pair.
    pub fn between(from: TierId, to: TierId) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|p| p.from() == from && p.to() == to)
    }

    pub fn from(self) -> TierId {
        match self {
            TransferPath::L0ToL1 | TransferPath::L0ToL2 => TierId::L0,
            TransferPath::L1ToL0 | TransferPath::L1ToL2 => TierId::L1,
            TransferPath::L2ToL1 | TransferPath::L2ToL0 => TierId::L2,
        }
    }

    pub fn to(self) -> TierId {
        match self {
            TransferPath::L1ToL0 | TransferPath::L2ToL0 => TierId::L0,
            TransferPath::L0ToL1 | TransferPath::L2ToL1 => TierId::L1,
            TransferPath::L1ToL2 | TransferPath::L0ToL2 => TierId::L2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_round_trip_their_endpoints() {
        for p in TransferPath::ALL {
            assert_eq!(TransferPath::between(p.from(), p.to()), Some(p));
            assert_eq!(
                p.as_str(),
                format!("{}_to_{}", p.from().as_str(), p.to().as_str())
            );
        }
        assert_eq!(TransferPath::between(TierId::L0, TierId::L0), None);
        assert_eq!(TransferPath::between(TierId::L3, TierId::L0), None);
    }
}
