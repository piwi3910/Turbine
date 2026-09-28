//! The kernel C ABI header names no vendor, declares the entry-point trio of every op the
//! registry binds, and carries the ABI version the Rust side expects (P1 AC S-1/S-7, contract §9)
//! and the additive minor revisions v2.1 (P2c AC S-5), v2.2 (the `moe_route` BF16-logits
//! flag), v2.4 (Phase 2m: implementation enumeration and the card profile), v2.5 (Phase 4:
//! copy streams and asynchronous copies) and v2.6 (Phase 5: native stream handles and the
//! sharded RMSNorm ops).
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
        // v2.3
        "typedef struct turbine_event turbine_event;",
        "int32_t turbine_host_alloc_pinned(turbine_ctx *ctx, size_t bytes, void **out);",
        "int32_t turbine_host_free_pinned(turbine_ctx *ctx, void *ptr);",
        "int32_t turbine_event_create(turbine_ctx *ctx, turbine_event **out);",
        "typedef struct turbine_stream turbine_stream;",
        "int32_t turbine_event_record(turbine_ctx *ctx, turbine_event *e, turbine_stream *s);",
        "int32_t turbine_event_synchronize(turbine_ctx *ctx, turbine_event *e);",
        "int32_t turbine_event_destroy(turbine_ctx *ctx, turbine_event *e);",
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

/// v2.2 turns `turbine_moe_route_desc`'s `renormalize` into `flags` (same position and type, 1
/// still renormalises) and adds the BF16-logits bit. Breaks if the member moves or a bit changes
/// value (an older library would then misread a descriptor instead of rejecting it).
#[test]
fn header_declares_the_v22_moe_route_flags() {
    let code = strip_comments(&header());
    assert_eq!(define(&code, "TURBINE_MOE_ROUTE_RENORMALIZE"), "1");
    assert_eq!(define(&code, "TURBINE_MOE_ROUTE_BF16_LOGITS"), "2");
    let flat = code.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(
        flat.contains(
            "typedef struct turbine_moe_route_desc { const float *router_logits; \
             int32_t num_tokens, num_experts, top_k, flags; int32_t *topk_ids;"
        ),
        "turbine_moe_route_desc changed shape"
    );
}

/// v2.4 (Phase 2m S-5/S-6): the minor becomes 4; the 15 op codes are the order of
/// `OpKind::ALL` (`OpKind::abi_code`); the five optional functions and the two structs are
/// declared. Breaks if an op code moves (a library would run another op's implementation) or a
/// struct member moves.
#[test]
fn header_declares_the_v24_minor_revision() {
    let code = strip_comments(&header());
    assert_eq!(
        define(&code, "TURBINE_ABI_VERSION"),
        "2u",
        "v2.4 keeps major 2"
    );
    // v2.5 (Phase 4) and v2.6 (Phase 5) raised the minor; the v2.4 group is unchanged.
    assert_eq!(define(&code, "TURBINE_ABI_MINOR"), "6u");
    for (i, op) in OpKind::ALL.iter().enumerate() {
        let name = format!("TURBINE_OP_{}", op.as_str().to_ascii_uppercase());
        assert_eq!(define(&code, &name), i.to_string(), "{name}");
        assert_eq!(op.abi_code(), i as i32, "{op}");
    }
    assert_eq!(define(&code, "TURBINE_IMPL_NEEDS_HOST_OFFSETS"), "1u");
    let flat = code.split_whitespace().collect::<Vec<_>>().join(" ");
    for decl in [
        "typedef struct turbine_impl_entry { const char *name; const char *provider; \
         uint32_t flags; } turbine_impl_entry;",
        "typedef struct turbine_card_profile { uint32_t struct_bytes; const char *arch; \
         int32_t wave_size; int32_t lds_bytes; int64_t moe_small_max_rows; \
         int32_t paged_page_multiple; } turbine_card_profile;",
        "int32_t turbine_impl_count(int32_t op);",
        "int32_t turbine_impl_info(int32_t op, int32_t index, turbine_impl_entry *out);",
        "int32_t turbine_impl_supports(int32_t op, int32_t index, const void *desc);",
        "int32_t turbine_impl_run(turbine_ctx *ctx, int32_t op, int32_t index, const void *desc);",
        "int32_t turbine_ctx_set_profile(turbine_ctx *ctx, const turbine_card_profile *p);",
    ] {
        assert!(
            flat.contains(decl),
            "turbine_kernels.h lacks the v2.4 {decl}"
        );
    }
}

/// v2.5 (Phase 4 S-5/S-6, provisional decision "Phase 4: kernel ABI v2.5 instead of v3"): the
/// minor becomes 5 and the copy-stream group is declared with the Phase 4 names; the major stays
/// 2 (additive). Breaks if a direction code moves (a library would copy the wrong way) or a
/// signature changes.
#[test]
fn header_declares_the_v25_copy_streams() {
    let code = strip_comments(&header());
    assert_eq!(
        define(&code, "TURBINE_ABI_VERSION"),
        "2u",
        "v2.5 keeps major 2"
    );
    // v2.6 (Phase 5) raised the minor; the v2.5 group is unchanged.
    assert_eq!(define(&code, "TURBINE_ABI_MINOR"), "6u");
    assert_eq!(define(&code, "TURBINE_COPY_H2D"), "0");
    assert_eq!(define(&code, "TURBINE_COPY_D2H"), "1");
    assert_eq!(define(&code, "TURBINE_COPY_D2D"), "2");
    let flat = code.split_whitespace().collect::<Vec<_>>().join(" ");
    for decl in [
        "int32_t turbine_copy_stream_create(turbine_ctx *ctx, turbine_stream **out);",
        "int32_t turbine_copy_stream_destroy(turbine_ctx *ctx, turbine_stream *s);",
        "int32_t turbine_memcpy_async(turbine_ctx *ctx, turbine_stream *s, void *dst, \
         const void *src, size_t bytes, int32_t kind);",
        "int32_t turbine_event_query(turbine_ctx *ctx, turbine_event *e);",
        "int32_t turbine_stream_wait_event(turbine_ctx *ctx, turbine_stream *s, turbine_event *e);",
    ] {
        assert!(
            flat.contains(decl),
            "turbine_kernels.h lacks the v2.5 {decl}"
        );
    }
}

/// v2.6 (Phase 5 Task 6, decision "P5 T6", answer B): the minor becomes 6, the native stream
/// handle and the two sharded RMSNorm trios are declared with their descriptors (the member
/// order of contract §9.3), and the two new op codes follow `OpKind::ALL`; the major stays 2.
/// Breaks if a descriptor member moves (a library would read another field) or the op codes
/// are not appended.
#[test]
fn header_declares_the_v26_tensor_parallel_group() {
    let code = strip_comments(&header());
    assert_eq!(
        define(&code, "TURBINE_ABI_VERSION"),
        "2u",
        "v2.6 keeps major 2"
    );
    assert_eq!(define(&code, "TURBINE_ABI_MINOR"), "6u");
    assert_eq!(define(&code, "TURBINE_OP_ROW_SUMSQ"), "15");
    assert_eq!(define(&code, "TURBINE_OP_RMSNORM_SHARDED"), "16");
    for op in [OpKind::RowSumsq, OpKind::RmsnormSharded] {
        assert!(
            OpKind::ALL.contains(&op),
            "{op} is a registry op, so header_declares_every_registry_op checks its trio"
        );
        assert_eq!(op.abi_minor(), 6, "{op}");
    }
    let flat = code.split_whitespace().collect::<Vec<_>>().join(" ");
    for decl in [
        "int32_t turbine_stream_native_handle(turbine_ctx *ctx, turbine_stream *s, void **out);",
        "typedef struct turbine_row_sumsq_desc { const void *x; float *sumsq; \
         int64_t rows, dim, x_stride_row; int32_t dtype; } turbine_row_sumsq_desc;",
        "typedef struct turbine_rmsnorm_sharded_desc { const void *x; const void *weight; \
         const float *sumsq; void *out; int64_t rows, dim, full_dim, x_stride_row, \
         out_stride_row; float eps; int32_t dtype; } turbine_rmsnorm_sharded_desc;",
        "int32_t turbine_row_sumsq(turbine_ctx *ctx, const turbine_row_sumsq_desc *d);",
        "int32_t turbine_rmsnorm_sharded(turbine_ctx *ctx, const turbine_rmsnorm_sharded_desc *d);",
    ] {
        assert!(
            flat.contains(decl),
            "turbine_kernels.h lacks the v2.6 {decl}"
        );
    }
}
