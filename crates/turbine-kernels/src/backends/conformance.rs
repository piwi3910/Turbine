//! The `execution_backend` conformance suite (Phase 2m S-13): run over a registry — never a
//! hand-written list — by `registries::registry_conformance::backends`, so a backend registered
//! without passing it fails `cargo test --workspace`.

use std::path::PathBuf;

use turbine_core::registry::{Registry, conformance};
use turbine_core::support::{HOST_VENDOR, VENDORS};
use turbine_core::types::DeviceId;
use turbine_device::DeviceInventory;

use super::{BackendRequest, ExecutionBackend};

/// Runs every check over every backend of `reg`; `Err` lists each broken check as
/// `<backend>: <check>: <detail>`.
///
/// Registry: the shared name checks, and at least one backend runs on the host (vendor `cpu`)
/// so the host test suites have one. Per backend: `vendor` (a support-matrix vendor, so the
/// matrix decides where it may serve), `sticky` (its sticky device-error prefixes are non-empty
/// and distinct), `notes` (no note without selections), `host` (a host backend opens with an
/// empty device inventory: at least one kernel provider, `order` naming exactly them, no
/// device, card or kernel-library context).
pub fn backends_suite(reg: &Registry<dyn ExecutionBackend>) -> Result<(), Vec<String>> {
    let mut failures = Vec::new();
    if let Err(e) = conformance::check(reg) {
        failures.push(format!("registry: {e}"));
    }
    if !reg.iter().any(|b| b.vendor() == HOST_VENDOR) {
        failures.push(format!(
            "registry: no backend of vendor {HOST_VENDOR} (the host)"
        ));
    }
    let meminfo = meminfo_fixture();
    for backend in reg.iter() {
        let name = backend.name();
        let mut fail =
            |check: &str, detail: String| failures.push(format!("{name}: {check}: {detail}"));
        if !VENDORS.contains(&backend.vendor()) {
            fail(
                "vendor",
                format!(
                    "{} is not a support-matrix vendor ({})",
                    backend.vendor(),
                    VENDORS.join(", ")
                ),
            );
        }
        let prefixes = backend.sticky_error_prefixes();
        for (i, p) in prefixes.iter().enumerate() {
            if p.is_empty() || prefixes[..i].contains(p) {
                fail("sticky", format!("empty or repeated prefix {p:?}"));
            }
        }
        let notes = backend.selection_notes(None, &[]);
        if !notes.is_empty() {
            fail("notes", format!("notes without selections: {notes:?}"));
        }
        if backend.vendor() != HOST_VENDOR {
            continue;
        }
        let Some(meminfo) = meminfo.as_ref() else {
            fail("host", "cannot write the meminfo fixture".into());
            continue;
        };
        let inventory = DeviceInventory {
            devices: Vec::new(),
            backends: Vec::new(),
        };
        let opened = backend.open(&BackendRequest {
            device: DeviceId(0),
            kernel_library: None,
            inventory: &inventory,
            meminfo,
            card_profile: "auto",
        });
        match opened {
            Err(e) => fail("host", format!("does not open on the host: {e}")),
            Ok(o) => {
                let ids: Vec<_> = o.providers.iter().map(|p| p.id()).collect();
                if ids.is_empty() || ids != o.order {
                    fail("host", format!("providers {ids:?}, order {:?}", o.order));
                }
                if o.device.is_some() || o.card.is_some() || o.context.is_some() {
                    fail(
                        "host",
                        "a host backend reports a device, card or context".into(),
                    );
                }
            }
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

/// A `/proc/meminfo` fixture of 1 GiB available.
fn meminfo_fixture() -> Option<PathBuf> {
    let dir = std::env::temp_dir().join(format!(
        "turbine-kernels-backends-conformance-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join("meminfo");
    std::fs::write(
        &path,
        "MemTotal:       2097152 kB\nMemAvailable:   1048576 kB\n",
    )
    .ok()?;
    Some(path)
}

#[cfg(test)]
mod tests {
    use turbine_core::registry::Module;

    use super::*;
    use crate::backends::{BackendError, CpuBackend, OpenedBackend};

    /// Claims the host but cannot open there, and repeats a sticky prefix.
    struct Broken;

    impl Module for Broken {
        fn name(&self) -> &'static str {
            "broken"
        }
    }

    impl ExecutionBackend for Broken {
        fn vendor(&self) -> &'static str {
            HOST_VENDOR
        }
        fn open(&self, _: &BackendRequest<'_>) -> Result<OpenedBackend, BackendError> {
            Err(BackendError::Startup("nothing here".into()))
        }
        fn sticky_error_prefixes(&self) -> &'static [&'static str] {
            &["x", "x"]
        }
    }

    static BROKEN: Registry<dyn ExecutionBackend> =
        Registry::new("execution_backend", &[&CpuBackend, &Broken]);

    /// A host backend that does not open, with a repeated sticky prefix, fails exactly those
    /// two checks under its own name; `cpu` passes. Breaks if the suite stops opening host
    /// backends or iterates a fixed list.
    #[test]
    fn rejects_broken_backend() {
        let failures = backends_suite(&BROKEN).unwrap_err();
        assert_eq!(failures.len(), 2, "{failures:#?}");
        assert!(failures[0].starts_with("broken: sticky:"), "{failures:#?}");
        assert!(failures[1].starts_with("broken: host:"), "{failures:#?}");
    }
}
