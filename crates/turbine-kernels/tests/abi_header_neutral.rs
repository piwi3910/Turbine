//! The kernel C ABI header names no vendor, declares the entry-point trio of every op the
//! registry binds, and carries the ABI version the Rust side expects (P1 AC S-1/S-7, contract §9)
//! and the additive minor revision v2.1 (P2c AC S-5).
use std::path::Path;

use turbine_kernels::TURBINE_KERNELS_ABI_VERSION;
use turbine_kernels::ops::OpKind;

/// Identifier prefixes that name a vendor or vendor runtime (contract §9.2), lowercase.
const VENDOR_PREFIXES: &[&str] = &["hip", "cuda", "rocm", "nv"];

fn header() -> String {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/include/turbine_kernels.h");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Removes `/* … */` and `// …` comments (C comments do not nest).
fn strip_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(c) = rest.chars().next() {
        if let Some(r) = rest.strip_prefix("/*") {
            out.push(' ');
            rest = r.find("*/").map_or("", |e| &r[e + 2..]);
        } else if let Some(r) = rest.strip_prefix("//") {
            rest = r.find('\n').map_or("", |e| &r[e..]);
        } else {
            out.push(c);
            rest = &rest[c.len_utf8()..];
        }
    }
    out
}

/// C identifiers in `code`: maximal `[A-Za-z_][A-Za-z0-9_]*` runs.
fn identifiers(code: &str) -> Vec<&str> {
    code.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|w| w.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_'))
        .collect()
}

#[test]
fn header_has_no_vendor_identifiers() {
    let code = strip_comments(&header());
    let vendor: Vec<&str> = identifiers(&code)
        .into_iter()
        .filter(|id| {
            let lower = id.to_ascii_lowercase();
            VENDOR_PREFIXES.iter().any(|v| lower.starts_with(v))
        })
        .collect();
    assert!(
        vendor.is_empty(),
        "vendor identifiers in turbine_kernels.h: {vendor:?}"
    );
}

#[test]
fn header_declares_every_registry_op() {
    let code = strip_comments(&header());
    let missing: Vec<String> = OpKind::ALL
        .iter()
        .flat_map(|op| {
            ["", "_supported", "_impl"].map(|suffix| format!("turbine_{}{suffix}(", op.as_str()))
        })
        .filter(|decl| !code.contains(decl.as_str()))
        .collect();
    assert!(missing.is_empty(), "turbine_kernels.h lacks {missing:?}");
    for v2 in [
        "turbine_ctx_get_info(",
        "turbine_ctx_info;",
        "turbine_attention_paged_desc;",
        "turbine_copy_blocks_desc;",
        "turbine_moe_route_desc;",
        "turbine_moe_experts_desc;",
    ] {
        assert!(code.contains(v2), "turbine_kernels.h lacks the v2 {v2}");
    }
}

#[test]
fn header_abi_version_matches_rust_constant() {
    let code = strip_comments(&header());
    let defines: Vec<&str> = code
        .lines()
        .filter_map(|l| l.trim().strip_prefix("#define TURBINE_ABI_VERSION"))
        .map(str::trim)
        .collect();
    assert_eq!(
        defines.len(),
        1,
        "expected exactly one TURBINE_ABI_VERSION define"
    );
    let value: u32 = defines[0]
        .strip_suffix('u')
        .unwrap_or_else(|| {
            panic!(
                "TURBINE_ABI_VERSION {} is not an unsigned literal",
                defines[0]
            )
        })
        .parse()
        .unwrap_or_else(|e| panic!("TURBINE_ABI_VERSION {}: {e}", defines[0]));
    assert_eq!(value, TURBINE_KERNELS_ABI_VERSION);
    assert_eq!(TURBINE_KERNELS_ABI_VERSION, 2, "Phase 2 ships ABI v2");
}

/// The single `#define <name> <value>` of the header, with its value trimmed.
fn define(code: &str, name: &str) -> String {
    let values: Vec<&str> = code
        .lines()
        .filter_map(|l| l.trim().strip_prefix("#define "))
        .filter_map(|l| l.strip_prefix(name))
        .filter(|rest| rest.starts_with(char::is_whitespace))
        .map(str::trim)
        .collect();
    assert_eq!(values.len(), 1, "expected exactly one {name} define");
    values[0].to_string()
}

#[test]
fn header_declares_the_v21_minor_revision() {
    let code = strip_comments(&header());
    assert_eq!(
        define(&code, "TURBINE_ABI_VERSION"),
        "2u",
        "v2.1 keeps major 2"
    );
    assert_eq!(define(&code, "TURBINE_ABI_MINOR"), "1u");
    assert_eq!(define(&code, "TURBINE_OPTION_GEMM_AUTOTUNE"), "1");
    assert_eq!(define(&code, "TURBINE_OPTION_GEMM_TUNED_SHAPES"), "2");
    // Declarations compared with whitespace collapsed, so line wrapping does not matter.
    let flat = code.split_whitespace().collect::<Vec<_>>().join(" ");
    for decl in [
        "uint32_t turbine_abi_minor(void);",
        "int32_t turbine_ctx_set_option(turbine_ctx *ctx, int32_t option, int64_t value);",
        "int32_t turbine_ctx_get_option(turbine_ctx *ctx, int32_t option, int64_t *out);",
        "} turbine_add_rmsnorm_desc;",
        // The nucleus mass is the descriptor's last member (added after the Task 9 fields).
        "float *sampled_logit; const float *top_p; } turbine_logits_reduce_desc;",
        "int32_t turbine_add_rmsnorm(turbine_ctx *ctx, const turbine_add_rmsnorm_desc *d);",
        "int32_t turbine_add_rmsnorm_supported(const turbine_add_rmsnorm_desc *d);",
        "const char *turbine_add_rmsnorm_impl(const turbine_add_rmsnorm_desc *d);",
        "int32_t turbine_logits_reduce(turbine_ctx *ctx, const turbine_logits_reduce_desc *d);",
        "int32_t turbine_logits_reduce_supported(const turbine_logits_reduce_desc *d);",
        "const char *turbine_logits_reduce_impl(const turbine_logits_reduce_desc *d);",
        "typedef struct turbine_graph turbine_graph;",
        "int32_t turbine_graph_begin(turbine_ctx *ctx);",
        "int32_t turbine_graph_end(turbine_ctx *ctx, turbine_graph **out);",
        "int32_t turbine_graph_launch(turbine_ctx *ctx, turbine_graph *g);",
        "int32_t turbine_graph_destroy(turbine_ctx *ctx, turbine_graph *g);",
    ] {
        assert!(
            flat.contains(decl),
            "turbine_kernels.h lacks the v2.1 {decl}"
        );
    }
    for op in [OpKind::AddRmsnorm, OpKind::LogitsReduce] {
        assert!(
            OpKind::ALL.contains(&op),
            "{op} is a registry op, so header_declares_every_registry_op checks its trio"
        );
    }
}
