//! Phase 2m S-5: the `cpu-reference` provider keeps one file per op family, so adding or
//! changing a family's reference touches its file only. `src/cpu/mod.rs` holds the provider
//! struct, its `KernelProvider` impl and the shared helpers, and no family impl.
use std::path::Path;

/// `(KernelProvider accessor, file under src/cpu, family trait)`: the paged attention kinds are
/// part of `attention` (their math lives in `paged.rs`), `add_rmsnorm` sits with `norm`.
const FAMILIES: &[(&str, &str, &str)] = &[
    ("gemm", "gemm.rs", "GemmKernel"),
    ("attention", "attention.rs", "AttentionKernel"),
    ("norm", "norm.rs", "NormKernel"),
    ("add_rmsnorm", "norm.rs", "AddRmsnormKernel"),
    ("rope", "rope.rs", "RopeKernel"),
    ("activation", "activation.rs", "ActivationKernel"),
    ("embedding", "embedding.rs", "EmbeddingKernel"),
    ("elementwise", "elementwise.rs", "ElementwiseKernel"),
    ("kv_copy", "kv_copy.rs", "KvCopyKernel"),
    ("moe", "moe.rs", "MoeKernel"),
    ("logits_reduce", "logits_reduce.rs", "LogitsReduceKernel"),
];

fn read(dir: &Path, file: &str) -> String {
    let path = dir.join(file);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Breaks if a family's impl moves back into `mod.rs` or into another family's file, or a
/// family loses its file.
#[test]
fn every_op_family_has_its_file() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/cpu");
    for (accessor, file, family) in FAMILIES {
        let text = read(&dir, file);
        let block = format!("impl {family} for CpuReference");
        assert!(
            text.contains(&block),
            "src/cpu/{file} lacks `{block}` (family `{accessor}`)"
        );
        for (_, other, _) in FAMILIES.iter().filter(|(_, f, _)| f != file) {
            assert!(
                !read(&dir, other).contains(&block),
                "`{block}` is in src/cpu/{other}, not {file}"
            );
        }
    }
    assert!(
        read(&dir, "paged.rs").contains("pub(super) fn attention("),
        "src/cpu/paged.rs holds the paged attention kinds"
    );
    let module = read(&dir, "mod.rs");
    for line in module.lines() {
        assert!(
            !(line.starts_with("impl ") && line.contains("Kernel for CpuReference")),
            "src/cpu/mod.rs still holds `{line}`"
        );
    }
    assert!(module.contains("impl KernelProvider for CpuReference"));
}
