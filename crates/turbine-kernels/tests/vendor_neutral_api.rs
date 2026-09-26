//! Phase 8 S-5: no public signature of the core crates names a vendor-specific type.
//! Walks every `pub` item reachable from each crate root through `pub mod` declarations and
//! checks every type path, trait path and `pub use` path in its signature.
use std::path::{Path, PathBuf};

use syn::visit::Visit;

const CORE_CRATES: &[&str] = &[
    "turbine-kernels",
    "turbine-tensor",
    "turbine-scheduler",
    "turbine-kv",
    "turbine-reliability",
];

/// Vendor words; `level_zero` is checked as the word pair `level`, `zero`.
const VENDOR_WORDS: &[&str] = &["cuda", "hip", "rocm", "nccl", "rccl", "cublas", "sycl"];

/// Enums whose variants are allowed to carry vendor names (`CollectiveBackendKind::Rccl`). The
/// Phase 2m `ExecutionBackend` is a trait with one module per backend (below), no longer an enum.
const BACKEND_ENUMS: &[&str] = &["CollectiveBackendKind", "ProviderKind"];

/// Registries whose child modules are the vendor-specific implementations by design (Phase 2m,
/// contract §24): `turbine_kernels::backends::hip` is reached only through the neutral
/// `ExecutionBackend` trait and `backends::registry()`, so its own items and their `pub use`
/// re-exports in the registry module are not core signatures.
const VENDOR_MODULE_REGISTRIES: &[&str] = &["turbine_kernels::backends"];

/// `CudaStream` → [cuda, stream]; `hip_event_t` → [hip, event, t]; `HIPBLASLt` → [hipblaslt].
fn words(ident: &str) -> Vec<String> {
    let mut out = Vec::new();
    for part in ident.split('_').filter(|p| !p.is_empty()) {
        let chars: Vec<char> = part.chars().collect();
        let mut cur = String::new();
        for (i, c) in chars.iter().enumerate() {
            let boundary = i > 0
                && c.is_uppercase()
                && (chars[i - 1].is_lowercase()
                    || chars.get(i + 1).is_some_and(|n| n.is_lowercase()))
                && !cur.chars().all(|x| x.is_uppercase());
            if boundary {
                out.push(std::mem::take(&mut cur).to_lowercase());
            }
            cur.push(*c);
        }
        out.push(cur.to_lowercase());
    }
    out
}

fn is_vendor_ident(ident: &str) -> bool {
    let w = words(ident);
    w.iter().any(|w| {
        VENDOR_WORDS
            .iter()
            .any(|v| w == v || (*v != "hip" && w.starts_with(v)) || w.starts_with("hipblas"))
    }) || w.windows(2).any(|p| p[0] == "level" && p[1] == "zero")
        || ident.to_lowercase().contains("levelzero")
}

#[derive(Default)]
struct PathCollector {
    leaks: Vec<String>,
}

impl PathCollector {
    fn check_path(&mut self, path: &syn::Path) {
        let segs: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
        for (i, seg) in segs.iter().enumerate() {
            let backend_variant = i > 0 && BACKEND_ENUMS.contains(&segs[i - 1].as_str());
            if is_vendor_ident(seg) && !backend_variant {
                self.leaks.push(segs.join("::"));
                return;
            }
        }
    }
}

impl<'ast> Visit<'ast> for PathCollector {
    fn visit_path(&mut self, path: &'ast syn::Path) {
        self.check_path(path);
        syn::visit::visit_path(self, path);
    }
}

fn is_pub(vis: &syn::Visibility) -> bool {
    matches!(vis, syn::Visibility::Public(_))
}

fn use_tree_idents(tree: &syn::UseTree, prefix: &str, out: &mut Vec<String>) {
    match tree {
        syn::UseTree::Path(p) => use_tree_idents(&p.tree, &format!("{prefix}{}::", p.ident), out),
        syn::UseTree::Name(n) => out.push(format!("{prefix}{}", n.ident)),
        syn::UseTree::Rename(r) => out.push(format!("{prefix}{} as {}", r.ident, r.rename)),
        syn::UseTree::Glob(_) => out.push(format!("{prefix}*")),
        syn::UseTree::Group(g) => g.items.iter().for_each(|t| use_tree_idents(t, prefix, out)),
    }
}

/// Leaks in `items`; `dir` is where `mod x;` files of these items live (None for inline sources).
fn item_leaks(items: &[syn::Item], dir: Option<&Path>, where_: &str, leaks: &mut Vec<String>) {
    let mut c = PathCollector::default();
    // The implementation modules of a vendor-module registry (see VENDOR_MODULE_REGISTRIES).
    let vendor_modules: Vec<String> = if VENDOR_MODULE_REGISTRIES.contains(&where_) {
        items
            .iter()
            .filter_map(|i| match i {
                syn::Item::Mod(m) => Some(m.ident.to_string()),
                _ => None,
            })
            .collect()
    } else {
        Vec::new()
    };
    for item in items {
        match item {
            syn::Item::Mod(m) if vendor_modules.contains(&m.ident.to_string()) => {}
            syn::Item::Use(u)
                if is_pub(&u.vis)
                    && matches!(&u.tree, syn::UseTree::Path(p)
                        if vendor_modules.contains(&p.ident.to_string())) => {}
            syn::Item::Fn(f) if is_pub(&f.vis) => c.visit_signature(&f.sig),
            syn::Item::Struct(s) if is_pub(&s.vis) => {
                c.visit_generics(&s.generics);
                s.fields
                    .iter()
                    .filter(|f| is_pub(&f.vis))
                    .for_each(|f| c.visit_type(&f.ty));
            }
            syn::Item::Enum(e) if is_pub(&e.vis) => {
                c.visit_generics(&e.generics);
                e.variants
                    .iter()
                    .flat_map(|v| v.fields.iter())
                    .for_each(|f| c.visit_type(&f.ty));
            }
            syn::Item::Union(u) if is_pub(&u.vis) => {
                u.fields.named.iter().for_each(|f| c.visit_type(&f.ty))
            }
            syn::Item::Trait(t) if is_pub(&t.vis) => {
                c.visit_generics(&t.generics);
                t.supertraits
                    .iter()
                    .for_each(|b| c.visit_type_param_bound(b));
                for ti in &t.items {
                    match ti {
                        syn::TraitItem::Fn(f) => c.visit_signature(&f.sig),
                        syn::TraitItem::Type(ty) => {
                            ty.bounds.iter().for_each(|b| c.visit_type_param_bound(b))
                        }
                        syn::TraitItem::Const(k) => c.visit_type(&k.ty),
                        _ => {}
                    }
                }
            }
            syn::Item::Type(t) if is_pub(&t.vis) => c.visit_type(&t.ty),
            syn::Item::Const(k) if is_pub(&k.vis) => c.visit_type(&k.ty),
            syn::Item::Static(s) if is_pub(&s.vis) => c.visit_type(&s.ty),
            syn::Item::Impl(i) => {
                let trait_impl = i.trait_.is_some();
                if let Some((path, _)) = &i.trait_ {
                    c.visit_path(path);
                    c.visit_type(&i.self_ty);
                }
                for ii in &i.items {
                    if let syn::ImplItem::Fn(f) = ii
                        && (trait_impl || is_pub(&f.vis))
                    {
                        c.visit_signature(&f.sig);
                    }
                }
            }
            syn::Item::Use(u) if is_pub(&u.vis) => {
                let mut paths = Vec::new();
                use_tree_idents(&u.tree, "", &mut paths);
                for p in paths {
                    if p.split("::")
                        .flat_map(|s| s.split(" as "))
                        .any(is_vendor_ident)
                    {
                        c.leaks.push(format!("pub use {p}"));
                    }
                }
            }
            syn::Item::Mod(m) if is_pub(&m.vis) => {
                let name = m.ident.to_string();
                let child_where = format!("{where_}::{name}");
                match (&m.content, dir) {
                    (Some((_, inner)), _) => item_leaks(
                        inner,
                        dir.map(|d| d.join(&name)).as_deref(),
                        &child_where,
                        leaks,
                    ),
                    (None, Some(d)) => {
                        let file = [d.join(format!("{name}.rs")), d.join(&name).join("mod.rs")]
                            .into_iter()
                            .find(|p| p.exists())
                            .unwrap_or_else(|| {
                                panic!(
                                    "{child_where}: no file for `pub mod {name};` in {}",
                                    d.display()
                                )
                            });
                        let parsed = parse(&file);
                        item_leaks(&parsed.items, Some(&d.join(&name)), &child_where, leaks);
                    }
                    (None, None) => {}
                }
            }
            _ => {}
        }
    }
    leaks.extend(c.leaks.into_iter().map(|l| format!("{where_}: {l}")));
}

fn parse(file: &Path) -> syn::File {
    let text = std::fs::read_to_string(file).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
    syn::parse_file(&text).unwrap_or_else(|e| panic!("{}: {e}", file.display()))
}

fn crate_src(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(name)
        .join("src")
}

/// Core crates that must exist now; `turbine-reliability` is created by Phase 3 and is checked
/// from the moment it exists.
const EXISTING_CORE_CRATES: &[&str] = &[
    "turbine-kernels",
    "turbine-tensor",
    "turbine-scheduler",
    "turbine-kv",
];

#[test]
fn vendor_types_in_core_signatures() {
    let mut leaks = Vec::new();
    for name in CORE_CRATES {
        let src = crate_src(name);
        if !src.parent().is_some_and(Path::exists) {
            assert!(
                !EXISTING_CORE_CRATES.contains(name),
                "{name} is missing at {}",
                src.display()
            );
            continue;
        }
        let root = parse(&src.join("lib.rs"));
        item_leaks(&root.items, Some(&src), &name.replace('-', "_"), &mut leaks);
    }
    assert!(
        leaks.is_empty(),
        "vendor types in core public signatures:\n{}",
        leaks.join("\n")
    );
}

#[test]
fn vendor_check_flags_leaks() {
    let src: syn::File = syn::parse_quote! {
        pub fn launch(s: CudaStream) {}
        pub struct Ctx { pub event: hip::Event, private: HipRaw }
        pub use self::ffi::HipStreamRaw;
        impl From<NcclError> for KernelError { fn from(e: NcclError) -> Self { todo!() } }
        pub trait Collective { fn comm(&self) -> RcclComm; }
        pub type Handle = level_zero::Device;
        pub mod inner { pub fn blas(h: CublasLtHandle) {} }
        // Allowed: neutral names, backend-enum variants, private items, words containing "hip".
        pub fn ok(b: &dyn ExecutionBackend, o: Ownership, r: Relationship) -> Option<u8> { None }
        pub const BACKEND: CollectiveBackendKind = CollectiveBackendKind::Rccl;
        fn private(s: CudaStream) {}
        mod ffi { pub struct HipStreamRaw; }
        // A vendor module outside a vendor-module registry is checked like any other.
        pub mod backends {
            pub mod hip { pub struct HipBackend; impl ExecutionBackend for HipBackend {} }
            pub use hip::HipBackend;
        }
    };
    let mut leaks = Vec::new();
    item_leaks(&src.items, None, "fixture", &mut leaks);
    let joined = leaks.join("\n");
    assert!(
        joined.contains("fixture::backends: pub use hip::HipBackend")
            && joined.contains("fixture::backends::hip: HipBackend"),
        "{joined}"
    );

    // The implementation modules of `turbine_kernels::backends` are vendor-specific by design;
    // the registry module's own signatures are still checked.
    let kernels: syn::File = syn::parse_quote! {
        pub mod backends {
            pub mod hip { pub struct HipBackend; impl ExecutionBackend for HipBackend {} }
            pub use hip::HipBackend;
            pub fn leak(s: HipStream) {}
        }
    };
    let mut registry_leaks = Vec::new();
    item_leaks(&kernels.items, None, "turbine_kernels", &mut registry_leaks);
    assert_eq!(
        registry_leaks,
        ["turbine_kernels::backends: HipStream"],
        "{registry_leaks:?}"
    );
    for expected in [
        "CudaStream",
        "hip::Event",
        "pub use self::ffi::HipStreamRaw",
        "NcclError",
        "RcclComm",
        "level_zero::Device",
        "fixture::inner: CublasLtHandle",
    ] {
        assert!(
            joined.contains(expected),
            "missing {expected} in:\n{joined}"
        );
    }
    for allowed in ["HipRaw", "Ownership", "Relationship", "ExecutionBackend"] {
        assert!(
            !joined.contains(allowed),
            "false positive {allowed} in:\n{joined}"
        );
    }
    assert!(
        !is_vendor_ident("ship_date")
            && !is_vendor_ident("Relationship")
            && !is_vendor_ident("chip")
    );
    assert!(
        is_vendor_ident("hipblasLtHandle")
            && is_vendor_ident("cudaStream_t")
            && is_vendor_ident("RocmPath")
    );
}
