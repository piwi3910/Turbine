//! Multi-GPU building blocks (P5): the backend-neutral [`collective::Collective`] trait, its
//! backends as modules of the `collective_backend` registry (a deterministic host reference
//! backend, and one runtime-loaded NCCL-API binding serving `rccl` and `nccl`), the static-mode
//! rank link as modules of the `rank_transport` registry ([`transport`]: `tcp`), and the
//! data-parallel router.
#![deny(unsafe_code)]

pub mod collective;
pub mod plan;
pub mod rank;
pub mod router;
pub mod tp;
pub mod transport;

#[cfg(test)]
mod registry_conformance {
    use turbine_core::registry::conformance;

    #[test]
    fn collective_backends() {
        let reg = crate::collective::registry();
        conformance::check(reg).unwrap();
        assert_eq!(reg.names(), ["host", "rccl", "nccl"]);
        for backend in reg.iter() {
            crate::collective::conformance::check(backend).unwrap();
        }
    }

    #[test]
    fn rank_transports() {
        let reg = crate::transport::registry();
        conformance::check(reg).unwrap();
        assert_eq!(reg.names(), ["tcp"]);
        for transport in reg.iter() {
            crate::transport::conformance::check(transport).unwrap();
        }
    }
}
