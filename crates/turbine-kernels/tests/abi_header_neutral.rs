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
    // v2.5 (Phase 4), v2.6, v2.7 and v2.8 (Phase 5) and v2.9 (Phase 6a) raised the minor; the
    // v2.4 group is unchanged.
    assert_eq!(define(&code, "TURBINE_ABI_MINOR"), "11u");
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
    // v2.6, v2.7 and v2.8 (Phase 5) raised the minor; the v2.5 group is unchanged.
    assert_eq!(define(&code, "TURBINE_ABI_MINOR"), "11u");
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
    // v2.7 and v2.8 raised the minor; the v2.6 group is unchanged.
    assert_eq!(define(&code, "TURBINE_ABI_MINOR"), "11u");
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

/// v2.7 (Phase 5, decision "P5: small-message all-reduce latency on novanas"): the minor
/// becomes 7 and the host-mapped group is declared: the three memory functions, the
/// `mapped_collective` trio and its descriptor (the member order `ffi::MappedCollectiveDesc`
/// mirrors), the kinds, reductions and abort reasons. It adds no registry op code (the trio is
/// driven by the `hostmem` collective backend, not the kernel registry). Breaks if a member or a
/// code moves (the library would read another field or run another kind).
#[test]
fn header_declares_the_v27_host_mapped_group() {
    let code = strip_comments(&header());
    assert_eq!(
        define(&code, "TURBINE_ABI_VERSION"),
        "2u",
        "v2.7 keeps major 2"
    );
    // v2.8 raised the minor; the v2.7 group is unchanged.
    assert_eq!(define(&code, "TURBINE_ABI_MINOR"), "11u");
    for (name, value) in [
        ("TURBINE_MAPPED_ALL_REDUCE", "0"),
        ("TURBINE_MAPPED_ALL_GATHER", "1"),
        ("TURBINE_MAPPED_REDUCE_SCATTER", "2"),
        ("TURBINE_MAPPED_BROADCAST", "3"),
        ("TURBINE_REDUCE_SUM", "0"),
        ("TURBINE_REDUCE_MAX", "1"),
        ("TURBINE_MAPPED_MAX_WORLD", "8"),
        ("TURBINE_MAPPED_MAX_BLOCKS", "1024"),
        ("TURBINE_MAPPED_ABORT_TIMEOUT", "1u"),
        ("TURBINE_MAPPED_ABORT_HOST", "2u"),
    ] {
        assert_eq!(define(&code, name), value, "{name}");
    }
    assert_eq!(
        OpKind::ALL.iter().filter(|op| op.abi_minor() <= 7).count(),
        17,
        "v2.7 adds no op code: TURBINE_OP_RMSNORM_SHARDED stays the last before v2.9"
    );
    let flat = code.split_whitespace().collect::<Vec<_>>().join(" ");
    for decl in [
        "typedef struct turbine_mapped_collective_desc { const void *send; void *recv; \
         int64_t bytes, send_stride, recv_stride; void *slots; int64_t slot_bytes; \
         uint64_t *flags; uint32_t *abort_word; uint64_t seq; int64_t timeout_ns; \
         int32_t kind, reduce_op, dtype; int32_t rank, world, root; int32_t max_blocks; \
         } turbine_mapped_collective_desc;",
        "int32_t turbine_host_alloc_mapped(turbine_ctx *ctx, size_t bytes, void **out);",
        "int32_t turbine_host_mapped_device_ptr(turbine_ctx *ctx, void *host, void **out);",
        "int32_t turbine_host_free_mapped(turbine_ctx *ctx, void *host);",
        "int32_t turbine_mapped_collective(turbine_ctx *ctx, \
         const turbine_mapped_collective_desc *d);",
        "int32_t turbine_mapped_collective_supported(const turbine_mapped_collective_desc *d);",
        "const char * turbine_mapped_collective_impl(const turbine_mapped_collective_desc *d);",
    ] {
        assert!(
            flat.contains(decl),
            "turbine_kernels.h lacks the v2.7 {decl}"
        );
    }
}

/// v2.8 (P5 Task 32, tensor-parallel decode graphs): the minor becomes 8 and the
/// device-sequenced step is declared with the v2.7 descriptor and a device counter pointer. No
/// registry op code. Breaks if the declaration drifts from `ffi::MappedDseqFn`.
#[test]
fn header_declares_the_v28_device_sequenced_step() {
    let code = strip_comments(&header());
    assert_eq!(
        define(&code, "TURBINE_ABI_VERSION"),
        "2u",
        "v2.8 keeps major 2"
    );
    assert_eq!(define(&code, "TURBINE_ABI_MINOR"), "11u");
    assert_eq!(
        OpKind::ALL.iter().filter(|op| op.abi_minor() <= 8).count(),
        17,
        "v2.8 adds no op code"
    );
    let flat = code.split_whitespace().collect::<Vec<_>>().join(" ");
    let decl = "int32_t turbine_mapped_collective_dseq(turbine_ctx *ctx, \
                const turbine_mapped_collective_desc *d, uint64_t *seq_counter);";
    assert!(
        flat.contains(decl),
        "turbine_kernels.h lacks the v2.8 {decl}"
    );
}

/// v2.9 (Phase 6a Task 7): the minor becomes 9; the quantized GEMM and activation quantization
/// trios are declared with op codes 17 and 18, the scheme and activation codes equal the Rust
/// `abi_code`s, the FP8 dtype codes are 16 and 17, and the paged-attention descriptor gains the
/// trailing `k_scale` / `v_scale`. Breaks if a code drifts between the header and
/// `turbine_kernels::quant` (a library would dequantize with the wrong layout).
#[test]
fn header_declares_the_v29_quantization_group() {
    use turbine_core::types::DType;
    use turbine_kernels::quant::{ActQuantDesc, QuantSchemeDesc};
    let code = strip_comments(&header());
    assert_eq!(define(&code, "TURBINE_ABI_MINOR"), "11u");
    assert_eq!(define(&code, "TURBINE_OP_QGEMM"), "17");
    assert_eq!(define(&code, "TURBINE_OP_QUANTIZE_ACT"), "18");
    assert_eq!(OpKind::QGemm.abi_code(), 17);
    assert_eq!(OpKind::QuantizeAct.abi_code(), 18);
    assert_eq!(
        define(&code, "TURBINE_DTYPE_F8E4M3"),
        DType::F8E4M3.abi_code().to_string()
    );
    assert_eq!(
        define(&code, "TURBINE_DTYPE_U8"),
        DType::U8.abi_code().to_string()
    );
    for (name, scheme) in [
        ("TURBINE_QSCHEME_FP8_TENSOR", QuantSchemeDesc::Fp8Tensor),
        ("TURBINE_QSCHEME_FP8_CHANNEL", QuantSchemeDesc::Fp8Channel),
        (
            "TURBINE_QSCHEME_FP8_BLOCK",
            QuantSchemeDesc::Fp8Block {
                block_n: 128,
                block_k: 128,
            },
        ),
        (
            "TURBINE_QSCHEME_INT4_GROUP_ZP",
            QuantSchemeDesc::Int4GroupZp { group: 128 },
        ),
        (
            "TURBINE_QSCHEME_INT4_GROUP_SYM",
            QuantSchemeDesc::Int4GroupSym { group: 128 },
        ),
        ("TURBINE_QSCHEME_MXFP4", QuantSchemeDesc::Mxfp4),
    ] {
        assert_eq!(define(&code, name), scheme.abi_code().to_string(), "{name}");
    }
    for (name, mode) in [
        ("TURBINE_ACTQ_NONE", ActQuantDesc::None),
        ("TURBINE_ACTQ_FP8_TENSOR", ActQuantDesc::Fp8Tensor),
        ("TURBINE_ACTQ_FP8_TOKEN", ActQuantDesc::Fp8Token),
        (
            "TURBINE_ACTQ_FP8_GROUP128",
            ActQuantDesc::Fp8Group { group: 128 },
        ),
        ("TURBINE_ACTQ_MXFP4_EMULATED", ActQuantDesc::Mxfp4Emulated),
    ] {
        assert_eq!(define(&code, name), mode.abi_code().to_string(), "{name}");
    }
    let flat = code.split_whitespace().collect::<Vec<_>>().join(" ");
    for decl in [
        "int32_t turbine_qgemm(turbine_ctx *ctx, const turbine_qgemm_desc *d);",
        "int32_t turbine_qgemm_supported(const turbine_qgemm_desc *d);",
        "const char *turbine_qgemm_impl(const turbine_qgemm_desc *d);",
        "int32_t turbine_quantize_act(turbine_ctx *ctx, const turbine_quantize_act_desc *d);",
        "int32_t turbine_quantize_act_supported(const turbine_quantize_act_desc *d);",
        "const char *turbine_quantize_act_impl(const turbine_quantize_act_desc *d);",
        // v2.11 appends block_formats and tq_params after the v2.9 scales.
        "int32_t causal, dtype; float k_scale, v_scale; const uint8_t *block_formats;",
    ] {
        assert!(
            flat.contains(decl),
            "turbine_kernels.h lacks the v2.9 {decl}"
        );
    }
}

/// v2.10 (Phase 6a Task 28a): the minor becomes 10 and `turbine_rope_desc` gains the trailing
/// `float attn_factor` (YaRN's attention factor on cos/sin); no op code and no symbol are added.
/// Breaks if the field moves (a library would read another field as the factor) or a v2.10 op
/// code appears.
#[test]
fn header_declares_the_v210_rope_attn_factor() {
    let code = strip_comments(&header());
    assert_eq!(define(&code, "TURBINE_ABI_MINOR"), "11u");
    assert_eq!(
        OpKind::ALL.iter().filter(|op| op.abi_minor() == 10).count(),
        0,
        "v2.10 adds no op code"
    );
    let flat = code.split_whitespace().collect::<Vec<_>>().join(" ");
    let decl = "int32_t style; int32_t dtype; float attn_factor; } turbine_rope_desc;";
    assert!(
        flat.contains(decl),
        "turbine_kernels.h lacks the v2.10 {decl}"
    );
}

/// v2.11 (Phase 6b Task 5): the minor becomes 11; the KV transcode trio is declared with op code
/// 19, the format codes equal `KvTranscodeFormat::abi_code` (and the 6b block format codes of
/// `KV_FMT_*`), the direction codes are 0 and 1, and the descriptor keeps its field order, ending
/// with the TurboQuant tables `tq_params` (Task 8) whose struct keeps its own.
/// Breaks if a code drifts between the header and `turbine_kernels::ops` (a library would
/// encode with the wrong codec) or the descriptor fields move.
#[test]
fn header_declares_the_v211_kv_transcode_group() {
    use turbine_kernels::{
        KV_FMT_BF16, KV_FMT_FP8_E4M3, KV_FMT_TQ2, KV_FMT_TQ4, KvTranscodeFormat,
    };
    let code = strip_comments(&header());
    assert_eq!(define(&code, "TURBINE_ABI_MINOR"), "11u");
    assert_eq!(define(&code, "TURBINE_OP_KV_TRANSCODE"), "19");
    assert_eq!(OpKind::KvTranscode.abi_code(), 19);
    assert_eq!(OpKind::KvTranscode.abi_minor(), 11);
    assert_eq!(OpKind::ALL.len(), 20);
    for (name, format, block_format) in [
        ("TURBINE_KVFMT_L0", KvTranscodeFormat::L0, KV_FMT_BF16),
        (
            "TURBINE_KVFMT_FP8_E4M3",
            KvTranscodeFormat::Fp8E4m3,
            KV_FMT_FP8_E4M3,
        ),
        ("TURBINE_KVFMT_TQ4", KvTranscodeFormat::Tq4, KV_FMT_TQ4),
        ("TURBINE_KVFMT_TQ2", KvTranscodeFormat::Tq2, KV_FMT_TQ2),
    ] {
        assert_eq!(define(&code, name), format.abi_code().to_string(), "{name}");
        assert_eq!(i32::from(block_format), format.abi_code(), "{name}");
    }
    assert_eq!(define(&code, "TURBINE_KV_ENCODE"), "0");
    assert_eq!(define(&code, "TURBINE_KV_DECODE"), "1");
    let flat = code.split_whitespace().collect::<Vec<_>>().join(" ");
    for decl in [
        "void *const *pages; const float *k_scales; const float *v_scales; void *coded; \
         int64_t coded_block_bytes; uint64_t seed; int32_t num_blocks, layers, block_tokens, \
         num_kv_heads, head_dim; int32_t page_dtype; int32_t format; int32_t direction; \
         const turbine_tq_params *tq_params; } turbine_kv_transcode_desc;",
        "typedef struct turbine_tq_params { uint64_t seed; const float *codebooks[4]; \
         const float *tables; } turbine_tq_params;",
        "int32_t turbine_kv_transcode(turbine_ctx *ctx, const turbine_kv_transcode_desc *d);",
        "int32_t turbine_kv_transcode_supported(const turbine_kv_transcode_desc *d);",
        "const char *turbine_kv_transcode_impl(const turbine_kv_transcode_desc *d);",
    ] {
        let decl = decl.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(
            flat.contains(&decl),
            "turbine_kernels.h lacks the v2.11 {decl}"
        );
    }
}

/// Kernel ABI v2.11 host-to-device copy kernel (P6b, decision "6b: KV promotions slow decode —
/// which fix" A): `turbine_copy_seg` keeps its field order (dst, src, bytes: the layout of
/// `ffi::CopySeg`) and `turbine_memcpy_h2d_kernel` its signature. Breaks if a field moves (a
/// library would copy from the destination) or the declaration drifts from the Rust binding.
#[test]
fn header_declares_the_v211_copy_kernel() {
    let flat = strip_comments(&header())
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    for decl in [
        "typedef struct turbine_copy_seg { void *dst; const void *src; size_t bytes; } \
         turbine_copy_seg;",
        "int32_t turbine_memcpy_h2d_kernel(turbine_ctx *ctx, turbine_stream *s, const \
         turbine_copy_seg *segs, int32_t count, int32_t workgroups);",
    ] {
        let decl = decl.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(
            flat.contains(&decl),
            "turbine_kernels.h lacks the v2.11 {decl}"
        );
    }
}

/// Kernel ABI v2.11 mixed-format paged attention (P6b Task 12): the paged attention descriptor
/// ends with the device `block_formats` table and the layer's `tq_params` (after the v2.9 FP8
/// scales), and the TurboQuant page dtypes keep the codes of `DType::{Tq4, Tq2}`. Breaks if the
/// fields move (a library would read a block format as a pointer) or a dtype code drifts.
#[test]
fn header_declares_the_v211_mixed_paged_attention_fields() {
    use turbine_core::types::DType;
    let code = strip_comments(&header());
    assert_eq!(
        define(&code, "TURBINE_DTYPE_TQ4"),
        DType::Tq4.abi_code().to_string()
    );
    assert_eq!(
        define(&code, "TURBINE_DTYPE_TQ2"),
        DType::Tq2.abi_code().to_string()
    );
    let flat = code.split_whitespace().collect::<Vec<_>>().join(" ");
    let decl = "float scale; int32_t causal, dtype; float k_scale, v_scale; \
                const uint8_t *block_formats; const struct turbine_tq_params *tq_params; \
                } turbine_attention_paged_desc;";
    let decl = decl.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(
        flat.contains(&decl),
        "turbine_kernels.h lacks the v2.11 paged attention fields {decl}"
    );
}
