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

    #[test]
    fn card_profiles() {
        let reg = crate::cards::registry();
        check(reg).unwrap();
        assert_eq!(reg.names(), ["gfx1201"]);
        // Every profile names a registered backend's vendor and at least one architecture, and
        // no architecture is claimed by two profiles.
        let vendors: Vec<&str> = crate::backends::registry()
            .iter()
            .map(|b| b.vendor())
            .collect();
        let mut archs: Vec<&str> = Vec::new();
        for p in reg.iter() {
            assert!(
                vendors.contains(&p.vendor),
                "{}: vendor {}",
                p.name,
                p.vendor
            );
            assert!(!p.archs.is_empty(), "{} lists no architecture", p.name);
            for arch in p.archs {
                assert!(!archs.contains(arch), "{arch} is in two profiles");
                archs.push(arch);
            }
        }
    }
}
