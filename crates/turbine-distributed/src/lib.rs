//! Multi-GPU building blocks (P5): the backend-neutral [`collective::Collective`] trait, its
//! backends as modules of the `collective_backend` registry (a deterministic host reference
//! backend first).

pub mod collective;

#[cfg(test)]
mod registry_conformance {
    use turbine_core::registry::conformance;

    #[test]
    fn collective_backends() {
        let reg = crate::collective::registry();
        conformance::check(reg).unwrap();
        assert_eq!(reg.names(), ["host"]);
        for backend in reg.iter() {
            crate::collective::conformance::check(backend).unwrap();
        }
    }
}
