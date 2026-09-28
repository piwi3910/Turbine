//! `unsafe` stays inside the allow-listed crates, every `unsafe` block or impl carries a
//! preceding `// SAFETY:` comment, and every other crate forbids `unsafe_code` in its manifest
//! (P1 AC S-1, contract §1.3).
use std::path::{Path, PathBuf};

/// Source trees (a directory, or a module given without `.rs`) that may contain `unsafe`
/// (contract §1.3, Phase 1 and Phase 5 rows; CONFLICT C-19).
const ALLOWED_SOURCES: &[&str] = &[
    "crates/turbine-device/src",
    "crates/turbine-kernels/src",
    "crates/turbine-distributed/src/collective/ffi",
];
/// Crate directories whose manifests allow `unsafe_code`.
const UNSAFE_CRATES: &[&str] = &["turbine-device", "turbine-kernels"];
/// Crates that deny `unsafe_code` in the manifest and at the crate root, and allow it on one
/// module listed in `ALLOWED_SOURCES`.
const DENY_CRATES: &[&str] = &["turbine-distributed"];
/// Top-level directories holding one crate per subdirectory.
const CRATE_ROOTS: &[&str] = &["crates", "benches"];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repository root resolves")
}

/// Every crate directory under `crates/` and `benches/`, sorted for stable messages.
fn crate_dirs(root: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    for top in CRATE_ROOTS {
        let dir = root.join(top);
        let entries =
            std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
        for entry in entries {
            let path = entry.expect("directory entry").path();
            if path.join("Cargo.toml").is_file() {
                dirs.push(path);
            }
        }
    }
    dirs.sort();
    dirs
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|x| x == "rs") {
            out.push(path);
        }
    }
}

fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Returns `src` with every comment, string literal and char literal replaced by spaces, keeping
/// newlines, so line numbers and code positions survive and only real code tokens remain.
fn mask_non_code(src: &str) -> String {
    let chars: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let blank = |c: char| if c == '\n' { '\n' } else { ' ' };
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        let prev_is_ident = i > 0 && is_ident_char(chars[i - 1]);
        if c == '/' && next == Some('/') {
            while i < chars.len() && chars[i] != '\n' {
                out.push(' ');
                i += 1;
            }
        } else if c == '/' && next == Some('*') {
            // Block comments nest in Rust.
            let mut depth = 0usize;
            while i < chars.len() {
                if chars[i] == '/' && chars.get(i + 1) == Some(&'*') {
                    depth += 1;
                    out.push_str("  ");
                    i += 2;
                } else if chars[i] == '*' && chars.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    out.push_str("  ");
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    out.push(blank(chars[i]));
                    i += 1;
                }
            }
        } else if !prev_is_ident && (c == 'r' || (c == 'b' && next == Some('r'))) && {
            // Raw string: r"…", r#"…"#, br"…".
            let start = if c == 'b' { i + 2 } else { i + 1 };
            let hashes = chars[start.min(chars.len())..]
                .iter()
                .take_while(|&&h| h == '#')
                .count();
            chars.get(start + hashes) == Some(&'"')
        } {
            let start = if c == 'b' { i + 2 } else { i + 1 };
            let hashes = chars[start..].iter().take_while(|&&h| h == '#').count();
            let body = start + hashes + 1;
            for _ in i..body {
                out.push(' ');
            }
            i = body;
            while i < chars.len() {
                if chars[i] == '"' && chars[i + 1..].iter().take(hashes).all(|&h| h == '#') {
                    let end = (i + 1 + hashes).min(chars.len());
                    for _ in i..end {
                        out.push(' ');
                    }
                    i = end;
                    break;
                }
                out.push(blank(chars[i]));
                i += 1;
            }
        } else if c == '"' {
            out.push(' ');
            i += 1;
            while i < chars.len() {
                match chars[i] {
                    '\\' => {
                        out.push(' ');
                        if let Some(&e) = chars.get(i + 1) {
                            out.push(blank(e));
                        }
                        i += 2;
                    }
                    '"' => {
                        out.push(' ');
                        i += 1;
                        break;
                    }
                    other => {
                        out.push(blank(other));
                        i += 1;
                    }
                }
            }
        } else if c == '\'' {
            // Char literal ('x', '\n', '\u{..}') or lifetime ('a): only literals are masked.
            if next == Some('\\') {
                let close = chars[i + 2..]
                    .iter()
                    .position(|&x| x == '\'')
                    .map_or(chars.len(), |p| i + 2 + p + 1);
                for _ in i..close {
                    out.push(' ');
                }
                i = close;
            } else if chars.get(i + 2) == Some(&'\'') {
                out.push_str("   ");
                i += 3;
            } else {
                out.push(c);
                i += 1;
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

/// Byte offsets of every word-bounded `unsafe` in a masked code line.
fn unsafe_positions(code: &str) -> Vec<usize> {
    code.match_indices("unsafe")
        .filter(|&(i, w)| {
            let before = code[..i].chars().next_back();
            let after = code[i + w.len()..].chars().next();
            !before.is_some_and(is_ident_char) && !after.is_some_and(is_ident_char)
        })
        .map(|(i, _)| i)
        .collect()
}

/// `unsafe` at `pos` opens a block (`unsafe {`) or an impl (`unsafe impl`).
fn is_block_or_impl(code: &str, pos: usize) -> bool {
    let rest = code[pos + "unsafe".len()..].trim_start();
    rest.starts_with('{')
        || rest
            .strip_prefix("impl")
            .is_some_and(|r| !r.starts_with(is_ident_char))
}

/// True when a `// SAFETY:` comment precedes line `idx`, walking up over comment lines,
/// attribute lines and earlier lines of the same statement (a rustfmt-wrapped
/// `let x =\n    unsafe { … }` keeps its comment above the `let`). A code line ending in `;`, `{`,
/// `}` or `,` — or a blank line — ends the walk: the comment must then sit directly above.
fn preceded_by_safety(raw: &[&str], masked: &[&str], idx: usize) -> bool {
    for i in (0..idx).rev() {
        let t = raw[i].trim_start();
        if t.starts_with("// SAFETY:") {
            return true;
        }
        if t.starts_with("//") || t.starts_with("#[") || t.starts_with("#![") {
            continue;
        }
        let code = masked[i].trim_end();
        if code.trim().is_empty() || code.ends_with([';', '{', '}', ',']) {
            return false;
        }
    }
    false
}

/// Violations (`<path>:<line>: <what>`) in one source file, `rel` relative to the repository.
fn scan_source(rel: &str, text: &str) -> Vec<String> {
    let allowed = ALLOWED_SOURCES
        .iter()
        .any(|a| rel.starts_with(&format!("{a}/")) || rel == format!("{a}.rs"));
    let masked_text = mask_non_code(text);
    let masked: Vec<&str> = masked_text.lines().collect();
    let raw: Vec<&str> = text.lines().collect();
    let mut violations = Vec::new();
    for (i, code) in masked.iter().enumerate() {
        let positions = unsafe_positions(code);
        if positions.is_empty() {
            continue;
        }
        if !allowed {
            violations.push(format!(
                "{rel}:{}: `unsafe` outside {ALLOWED_SOURCES:?}",
                i + 1
            ));
            continue;
        }
        if positions.iter().any(|&p| is_block_or_impl(code, p))
            && !preceded_by_safety(&raw, &masked, i)
        {
            violations.push(format!(
                "{rel}:{}: `unsafe` block or impl without a preceding `// SAFETY:` comment",
                i + 1
            ));
        }
    }
    violations
}

#[test]
fn unsafe_only_in_allowed_crates_with_safety_comments() {
    let root = repo_root();
    let mut files = Vec::new();
    for dir in crate_dirs(&root) {
        rust_files(&dir.join("src"), &mut files);
    }
    files.sort();
    assert!(
        files
            .iter()
            .any(|f| f.ends_with("crates/turbine-kernels/src/lib.rs")),
        "scan found no turbine-kernels sources under {}",
        root.display()
    );
    let mut violations = Vec::new();
    for file in &files {
        let rel = file
            .strip_prefix(&root)
            .expect("source under the repository root")
            .to_string_lossy()
            .replace('\\', "/");
        let text = std::fs::read_to_string(file)
            .unwrap_or_else(|e| panic!("read {}: {e}", file.display()));
        violations.extend(scan_source(&rel, &text));
    }
    assert!(violations.is_empty(), "{}", violations.join("\n"));
}

#[test]
fn other_crates_forbid_unsafe_code() {
    let root = repo_root();
    let mut missing = Vec::new();
    for dir in crate_dirs(&root) {
        let name = dir
            .file_name()
            .expect("crate directory name")
            .to_string_lossy()
            .into_owned();
        if UNSAFE_CRATES.contains(&name.as_str()) {
            continue;
        }
        let manifest = dir.join("Cargo.toml");
        let text = std::fs::read_to_string(&manifest)
            .unwrap_or_else(|e| panic!("read {}: {e}", manifest.display()));
        let level = if DENY_CRATES.contains(&name.as_str()) {
            let lib = dir.join("src/lib.rs");
            let root = std::fs::read_to_string(&lib)
                .unwrap_or_else(|e| panic!("read {}: {e}", lib.display()));
            assert!(
                root.lines().any(|l| l.trim() == "#![deny(unsafe_code)]"),
                "{} lacks `#![deny(unsafe_code)]`",
                lib.display()
            );
            "deny"
        } else {
            "forbid"
        };
        if !text
            .lines()
            .any(|l| l.trim() == format!(r#"unsafe_code = "{level}""#))
        {
            missing.push(
                manifest
                    .strip_prefix(&root)
                    .unwrap_or(&manifest)
                    .display()
                    .to_string(),
            );
        }
    }
    assert!(
        missing.is_empty(),
        "manifests without `unsafe_code = \"forbid\"` (or \"deny\" for {DENY_CRATES:?}): {missing:?}"
    );
}

/// Every allow-listed location exists (a directory or `<module>.rs`), so a moved module cannot
/// leave a stale entry that silently allows `unsafe` wherever the path is recreated.
#[test]
fn allowlisted_sources_exist() {
    let root = repo_root();
    for a in ALLOWED_SOURCES {
        let dir = root.join(a);
        let file = root.join(format!("{a}.rs"));
        assert!(
            dir.is_dir() || file.is_file(),
            "allow-listed `unsafe` location {a} does not exist"
        );
    }
}

/// The scanner itself: masked comments/strings never count, and both failure kinds name the line.
#[test]
fn scanner_flags_leaks_and_missing_safety_comments() {
    let leak = "fn f() {\n    // unsafe in a comment is fine\n    let s = \"unsafe { }\";\n    let x = unsafe { g() };\n}\n";
    assert_eq!(
        scan_source("crates/turbine-tensor/src/x.rs", leak),
        vec![format!(
            "crates/turbine-tensor/src/x.rs:4: `unsafe` outside {ALLOWED_SOURCES:?}"
        )]
    );

    let missing = "fn f() {\n    let a = 1;\n    let x = unsafe { g() };\n}\n";
    assert_eq!(
        scan_source("crates/turbine-kernels/src/x.rs", missing),
        vec![
            "crates/turbine-kernels/src/x.rs:3: `unsafe` block or impl without a preceding `// SAFETY:` comment"
                .to_string()
        ]
    );

    // A comment above an earlier, finished statement does not cover the block.
    let stale = "// SAFETY: covers only the next line.\nlet a = unsafe { g() };\nlet b =\n    unsafe { h() };\n";
    assert_eq!(
        scan_source("crates/turbine-kernels/src/x.rs", stale),
        vec![
            "crates/turbine-kernels/src/x.rs:4: `unsafe` block or impl without a preceding `// SAFETY:` comment"
                .to_string()
        ]
    );

    // Only the collective::ffi module of turbine-distributed is allow-listed.
    let block = "// SAFETY: the library outlives the pointer.\nlet x = unsafe { g() };\n";
    assert!(scan_source("crates/turbine-distributed/src/collective/ffi.rs", block).is_empty());
    assert!(
        scan_source(
            "crates/turbine-distributed/src/collective/ffi/table.rs",
            block
        )
        .is_empty()
    );
    for other in [
        "crates/turbine-distributed/src/collective/host.rs",
        "crates/turbine-distributed/src/collective/ffi_extra.rs",
        "crates/turbine-distributed/src/lib.rs",
    ] {
        assert_eq!(
            scan_source(other, block),
            vec![format!("{other}:2: `unsafe` outside {ALLOWED_SOURCES:?}")]
        );
    }

    let ok = "// SAFETY: the pointer is owned by this context.\n#[allow(clippy::x)]\nunsafe impl Send for C {}\n// SAFETY: see above.\nlet x = unsafe { g() };\n// SAFETY: wrapped statement.\nlet y =\n    unsafe { g() } == 0;\ntype F = unsafe extern \"C\" fn();\nlet r = r#\"unsafe {\"#; let c = '\"'; fn l<'a>() {}\n";
    assert!(scan_source("crates/turbine-kernels/src/x.rs", ok).is_empty());
}

/// P5 Task 32 corruption guard: host copies reach the library's `turbine_memcpy_h2d` /
/// `turbine_memcpy_d2h` only from `shim.rs`, three times each — through the pinned bounce buffer,
/// through a pinned staging buffer, and the counted, debug-asserted pageable fallback of a
/// library without pinned memory. A new call site (a pageable copy creeping back) fails here and
/// must be reviewed: device-to-host copies into untouched pageable memory returned wrong bytes on
/// ROCm 7.14.1 with two GPUs in one process.
#[test]
fn host_copies_reach_the_library_only_through_the_audited_sites() {
    let src = repo_root().join("crates/turbine-kernels/src");
    let mut sites = Vec::new();
    for entry in std::fs::read_dir(&src).expect("src dir") {
        let path = entry.expect("entry").path();
        if path.extension().is_some_and(|e| e == "rs") {
            let text = std::fs::read_to_string(&path).expect("read");
            for (i, line) in text.lines().enumerate() {
                if line.contains("syms.memcpy_h2d") || line.contains("syms.memcpy_d2h") {
                    sites.push(format!(
                        "{}:{}",
                        path.file_name().unwrap().to_string_lossy(),
                        i + 1
                    ));
                }
            }
        }
    }
    assert_eq!(
        sites.iter().filter(|s| s.starts_with("shim.rs:")).count(),
        6,
        "memcpy_h2d/d2h call sites: {sites:?}"
    );
    assert_eq!(sites.len(), 6, "memcpy_h2d/d2h call sites: {sites:?}");
}
