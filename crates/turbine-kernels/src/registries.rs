//! The shared conformance checks (Phase 2m S-1, `turbine_core::registry::conformance`) over
//! every registry of this crate.

#[cfg(test)]
mod registry_conformance {
    use turbine_core::registry::conformance::check;

    #[test]
    fn backends() {
        let reg = crate::backends::registry();
        check(reg).unwrap();
        assert_eq!(reg.names(), ["cpu", "hip"]);
    }
}
