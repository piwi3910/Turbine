//! The registry convention every extension point follows (Phase 2m S-1, contract §24): a trait
//! extending [`Module`], one file (or directory) per implementation, and a compiled-in `static`
//! [`Registry`] listing them. Lookup is by name, enumeration keeps registration order (for
//! diagnostics), and every selection logs `event="module_selected"` with the extension point,
//! the name and the reason once at startup. Nothing is loaded at run time.
//!
//! ```
//! use turbine_core::registry::{Module, Registry};
//!
//! trait Greeter: Module {}
//! struct Hello;
//! impl Module for Hello {
//!     fn name(&self) -> &'static str {
//!         "hello"
//!     }
//! }
//! impl Greeter for Hello {}
//!
//! static GREETERS: Registry<dyn Greeter> = Registry::new("greeter", &[&Hello]);
//! assert_eq!(GREETERS.names(), ["hello"]);
//! assert!(GREETERS.get("hello").is_some());
//! ```

use std::sync::Once;

/// A named implementation of an extension point. Names match `^[a-z0-9_]{1,64}$`.
pub trait Module: Send + Sync + 'static {
    fn name(&self) -> &'static str;
}

/// A compiled-in list of the modules of one extension point (`point`), in registration order.
pub struct Registry<T: ?Sized + Module + 'static> {
    point: &'static str,
    modules: &'static [&'static T],
    /// Guards the one-time `registry_duplicate` warnings of [`Registry::select`].
    duplicates_warned: Once,
}

/// A configured or requested name no module of the extension point carries.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{point}: `{name}` is not registered (registered: {registered})")]
pub struct UnknownModule {
    pub point: &'static str,
    pub name: String,
    /// The registered names joined with `, `, in registration order.
    pub registered: String,
}

impl<T: ?Sized + Module + 'static> Registry<T> {
    pub const fn new(point: &'static str, modules: &'static [&'static T]) -> Registry<T> {
        Registry {
            point,
            modules,
            duplicates_warned: Once::new(),
        }
    }

    /// The extension point's name (`model_family`, `execution_backend`, …).
    pub fn point(&self) -> &'static str {
        self.point
    }

    /// Every module in registration order.
    pub fn iter(&self) -> impl Iterator<Item = &'static T> + '_ {
        self.modules.iter().copied()
    }

    /// Every module's name in registration order (duplicates included).
    pub fn names(&self) -> Vec<&'static str> {
        self.iter().map(Module::name).collect()
    }

    /// The module named `name`; the first registration wins when two carry the same name.
    pub fn get(&self, name: &str) -> Option<&'static T> {
        self.iter().find(|m| m.name() == name)
    }

    /// Names carried by more than one module, each once, in order of first registration.
    pub fn duplicate_names(&self) -> Vec<&'static str> {
        let names = self.names();
        let mut out: Vec<&'static str> = Vec::new();
        for (i, name) in names.iter().enumerate() {
            if names[..i].contains(name) && !out.contains(name) {
                out.push(name);
            }
        }
        out
    }

    /// The module named `name`, logging `event="module_selected"` with `reason`. On its first
    /// call it also logs one `WARN event="registry_duplicate"` per duplicate name.
    pub fn select(&self, name: &str, reason: &str) -> Result<&'static T, UnknownModule> {
        self.duplicates_warned.call_once(|| {
            for dup in self.duplicate_names() {
                tracing::warn!(
                    event = "registry_duplicate",
                    point = self.point,
                    name = dup,
                    "two modules share this name; the first registration wins"
                );
            }
        });
        match self.get(name) {
            Some(module) => {
                log_selected(self.point, name, reason);
                Ok(module)
            }
            None => Err(self.unknown(name)),
        }
    }

    /// The [`UnknownModule`] error for `name` at this extension point.
    pub fn unknown(&self, name: &str) -> UnknownModule {
        UnknownModule {
            point: self.point,
            name: name.to_string(),
            registered: self.names().join(", "),
        }
    }
}

/// Logs one selection: `INFO event="module_selected" point name reason`.
pub fn log_selected(point: &'static str, name: &str, reason: &str) {
    tracing::info!(
        event = "module_selected",
        point,
        name,
        reason,
        "module selected"
    );
}

/// True when `name` matches `^[a-z0-9_]{1,64}$`, the form of every module name.
pub fn valid_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}

/// The checks every registry's conformance test runs.
pub mod conformance {
    use super::{Module, Registry, valid_name};

    /// `Err` naming the first problem: an empty registry, a duplicate name, a name that does
    /// not match `^[a-z0-9_]{1,64}$`, or a name `get` does not return.
    pub fn check<T: ?Sized + Module>(reg: &Registry<T>) -> Result<(), String> {
        let point = reg.point();
        if reg.iter().next().is_none() {
            return Err(format!("{point}: the registry is empty"));
        }
        if let Some(dup) = reg.duplicate_names().first() {
            return Err(format!("{point}: `{dup}` is registered more than once"));
        }
        for module in reg.iter() {
            let name = module.name();
            if !valid_name(name) {
                return Err(format!(
                    "{point}: `{name}` does not match ^[a-z0-9_]{{1,64}}$"
                ));
            }
            let found = reg.get(name).map(|m| std::ptr::addr_eq(m, module));
            if found != Some(true) {
                return Err(format!("{point}: get(`{name}`) does not return its module"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    use super::*;

    trait Toy: Module {
        fn id(&self) -> u32;
    }

    struct ToyModule(&'static str, u32);

    impl Module for ToyModule {
        fn name(&self) -> &'static str {
            self.0
        }
    }

    impl Toy for ToyModule {
        fn id(&self) -> u32 {
            self.1
        }
    }

    static A1: ToyModule = ToyModule("a", 1);
    static B: ToyModule = ToyModule("b", 2);
    static A2: ToyModule = ToyModule("a", 3);

    static WITH_DUP: Registry<dyn Toy> = Registry::new("toy", &[&A1, &B, &A2]);
    static CLEAN: Registry<dyn Toy> = Registry::new("toy", &[&A1, &B]);

    #[test]
    fn lookup_enumeration_and_duplicates() {
        assert_eq!(WITH_DUP.point(), "toy");
        assert_eq!(WITH_DUP.names(), ["a", "b", "a"]);
        assert_eq!(
            WITH_DUP.iter().map(|m| m.id()).collect::<Vec<_>>(),
            [1, 2, 3]
        );
        assert_eq!(WITH_DUP.get("a").map(|m| m.id()), Some(1));
        assert_eq!(WITH_DUP.get("b").map(|m| m.id()), Some(2));
        assert!(WITH_DUP.get("z").is_none());
        assert_eq!(WITH_DUP.duplicate_names(), ["a"]);
        let err = conformance::check(&WITH_DUP).unwrap_err();
        assert!(err.contains("`a`"), "{err}");

        assert!(CLEAN.duplicate_names().is_empty());
        assert_eq!(conformance::check(&CLEAN), Ok(()));

        static EMPTY: Registry<dyn Toy> = Registry::new("toy", &[]);
        assert!(conformance::check(&EMPTY).is_err());
        static BAD: ToyModule = ToyModule("Bad-Name", 4);
        static BAD_NAME: Registry<dyn Toy> = Registry::new("toy", &[&BAD]);
        let err = conformance::check(&BAD_NAME).unwrap_err();
        assert!(err.contains("Bad-Name"), "{err}");

        assert!(valid_name("llama3_json"));
        assert!(valid_name(&"x".repeat(64)));
        assert!(!valid_name(&"x".repeat(65)));
        assert!(!valid_name(""));
        assert!(!valid_name("Hip!"));
    }

    /// A `MakeWriter` target collecting every formatted log line.
    #[derive(Clone, Default)]
    struct Captured(Arc<Mutex<Vec<u8>>>);

    impl Write for Captured {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("capture lock").extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn select_logs_module_selected() {
        let captured = Captured::default();
        let writer = captured.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .with_max_level(tracing::Level::INFO)
            .finish();
        let (picked, missing) = tracing::subscriber::with_default(subscriber, || {
            // Another test thread may have cached these callsites as disabled.
            tracing::callsite::rebuild_interest_cache();
            (
                CLEAN.select("b", "configured"),
                CLEAN.select("z", "configured"),
            )
        });
        assert_eq!(picked.map(|m| m.id()), Ok(2));
        assert_eq!(
            missing.map(|m| m.id()),
            Err(UnknownModule {
                point: "toy",
                name: "z".into(),
                registered: "a, b".into(),
            })
        );
        assert_eq!(
            UnknownModule {
                point: "toy",
                name: "z".into(),
                registered: "a, b".into(),
            }
            .to_string(),
            "toy: `z` is not registered (registered: a, b)"
        );
        let text = String::from_utf8(captured.0.lock().unwrap().clone()).unwrap();
        let lines: Vec<&str> = text
            .lines()
            .filter(|l| l.contains("event=\"module_selected\""))
            .collect();
        assert_eq!(lines.len(), 1, "{text}");
        let line = lines[0];
        for part in ["point=\"toy\"", "name=\"b\"", "reason=\"configured\""] {
            assert!(line.contains(part), "{part} missing in {line}");
        }
    }

    #[test]
    fn select_warns_once_per_duplicate() {
        static DUP_ONCE: Registry<dyn Toy> = Registry::new("toy_dup", &[&A1, &A2]);
        let captured = Captured::default();
        let writer = captured.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .with_max_level(tracing::Level::INFO)
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            tracing::callsite::rebuild_interest_cache();
            assert_eq!(DUP_ONCE.select("a", "test").map(|m| m.id()), Ok(1));
            assert_eq!(DUP_ONCE.select("a", "test").map(|m| m.id()), Ok(1));
        });
        let text = String::from_utf8(captured.0.lock().unwrap().clone()).unwrap();
        let warnings = text
            .lines()
            .filter(|l| l.contains("event=\"registry_duplicate\""))
            .count();
        assert_eq!(warnings, 1, "{text}");
    }
}
