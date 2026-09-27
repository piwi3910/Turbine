//! The L0 `KvReclaimer` (P3 S-10). The Phase 2 pool keeps no unreferenced cached blocks — a
//! block returns to the free list as soon as its last holder releases it — and has no lower
//! tier to demote to, so there is nothing to reclaim; phase-4 replaces this with the prefix
//! cache's reclaimer.

use turbine_reliability::throttle::KvReclaimer;

/// Reclaims nothing: every L0 block in use belongs to a live sequence.
#[derive(Clone, Copy, Debug, Default)]
pub struct L0Reclaimer;

impl KvReclaimer for L0Reclaimer {
    fn demote(&self, _target_utilization: f64) -> u64 {
        0
    }

    fn free_unreferenced(&self, _target_utilization: f64) -> u64 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_to_reclaim() {
        let r = L0Reclaimer;
        assert_eq!(r.demote(0.7), 0);
        assert_eq!(r.free_unreferenced(0.0), 0);
        assert_eq!(r.free_optional(), 0);
    }
}
