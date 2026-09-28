//! Multi-GPU building blocks (P5): the backend-neutral [`collective::Collective`] trait, its
//! backends as modules of the `collective_backend` registry (a deterministic host reference
//! backend, one runtime-loaded NCCL-API binding serving `rccl` and `nccl`, and `hostmem`, the
//! host-mapped one-shot collectives of the kernel library for GPUs without peer access).
#![deny(unsafe_code)]

pub mod collective;
pub mod plan;
pub mod rank;
pub mod router;
pub mod tp;

#[cfg(test)]
mod registry_conformance {
    use turbine_core::registry::conformance;

    #[test]
    fn collective_backends() {
        let reg = crate::collective::registry();
        conformance::check(reg).unwrap();
        assert_eq!(reg.names(), ["host", "hostmem", "rccl", "nccl"]);
        for backend in reg.iter() {
            crate::collective::conformance::check(backend).unwrap();
        }
    }
}
