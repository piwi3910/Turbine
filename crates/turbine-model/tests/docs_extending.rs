//! Phase 2m S-12: `docs/extending/` has one how-to per extension point, and every page stays
//! true to the tree. Each page has the four sections a module author follows (files to add, the
//! registry entry, the conformance suite, the lab checks); every backticked path it names under
//! `crates/`, `kernels/` or `docs/` exists (a `<name>` placeholder segment matches any entry of
//! that directory with the same prefix and suffix); every backticked `cargo test` (or
//! `scripts/remote-cargo.sh test`) command names an existing package, test target and a test
//! filter whose every `::` segment is a test function or module in the tree. The `Registry
//! entry` section names at least one existing file and the `Conformance suite` section at least
//! one test command, so a page cannot drop either.

use std::path::{Path, PathBuf};

/// The nine extension points, one page each.
const PAGES: [&str; 9] = [
    "model-family",
    "tool-format",
    "weight-format",
    "kernel-implementation",
    "card-family",
    "backend",
    "logits-processor",
    "scheduling-policy",
    "eviction-policy",
];

const SECTIONS: [&str; 4] = [
    "## Files to add",
    "## Registry entry",
    "## Conformance suite",
    "## Lab checks",
];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The inline code spans of `text`, outside fenced code blocks.
fn code_spans(text: &str) -> Vec<String> {
    let mut spans = Vec::new();
    let mut fenced = false;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            fenced = !fenced;
            continue;
        }
        if fenced {
            continue;
        }
        let mut parts = line.split('`');
        parts.next();
        while let (Some(span), Some(_)) = (parts.next(), parts.next()) {
            spans.push(span.to_string());
        }
    }
    spans
}

/// The text of section `heading` (up to the next `## `).
fn section<'a>(text: &'a str, heading: &str) -> &'a str {
    let start = text
        .find(&format!("\n{heading}\n"))
        .map(|i| i + heading.len() + 2)
        .unwrap_or(text.len());
    let rest = &text[start..];
    let end = rest.find("\n## ").unwrap_or(rest.len());
    &rest[..end]
}

fn is_tree_path(span: &str) -> bool {
    ["crates/", "kernels/", "docs/"]
        .iter()
        .any(|p| span.starts_with(p))
        && !span.contains(char::is_whitespace)
}

/// Whether `path` (relative to the root, `<…>` segments as wildcards) exists.
fn path_exists(path: &str) -> bool {
    let mut candidates = vec![root()];
    for segment in path.trim_end_matches('/').split('/') {
        let mut next = Vec::new();
        for dir in &candidates {
            match (segment.find('<'), segment.find('>')) {
                (Some(open), Some(close)) if open < close => {
                    let (prefix, suffix) = (&segment[..open], &segment[close + 1..]);
                    let Ok(entries) = std::fs::read_dir(dir) else {
                        continue;
                    };
                    for entry in entries.flatten() {
                        let name = entry.file_name().to_string_lossy().into_owned();
                        if name.starts_with(prefix) && name.ends_with(suffix) {
                            next.push(entry.path());
                        }
                    }
                }
                _ => {
                    let p = dir.join(segment);
                    if p.exists() {
                        next.push(p);
                    }
                }
            }
        }
        candidates = next;
    }
    !candidates.is_empty()
}

/// Every `.rs` file under `dir`.
fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            rust_files(&p, out);
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
}

/// Whether some file of `files` defines `fn <name>` or `mod <name>`, or is `<name>.rs`.
fn names_test_item(files: &[(PathBuf, String)], name: &str) -> bool {
    files.iter().any(|(path, text)| {
        path.file_stem().is_some_and(|s| s == name)
            || text.contains(&format!("fn {name}("))
            || text.contains(&format!("fn {name}<"))
            || text.contains(&format!("mod {name} "))
            || text.contains(&format!("mod {name};"))
    })
}

/// Checks one test command span; `Err` says what does not exist.
fn check_test_command(span: &str) -> Result<(), String> {
    let args: Vec<&str> = span.split_whitespace().collect();
    let start = args
        .windows(2)
        .position(|w| (w[0] == "cargo" || w[0].ends_with("remote-cargo.sh")) && w[1] == "test")
        .ok_or("not a test command")?
        + 2;
    let mut package = None;
    let mut targets = Vec::new();
    let mut filters = Vec::new();
    let mut i = start;
    while i < args.len() {
        match args[i] {
            "-p" | "--package" => {
                package = args.get(i + 1).copied();
                i += 1;
            }
            "--test" => {
                targets.push(args.get(i + 1).copied().ok_or("--test without a name")?);
                i += 1;
            }
            flag if flag.starts_with('-') => {}
            filter => filters.push(filter),
        }
        i += 1;
    }
    let crate_dir = match package {
        Some(p) => {
            let dir = root().join("crates").join(p);
            if !dir.join("Cargo.toml").exists() {
                return Err(format!("package {p} is not crates/{p}"));
            }
            dir
        }
        None => root().join("crates"),
    };
    for t in &targets {
        if !crate_dir.join("tests").join(format!("{t}.rs")).exists() {
            return Err(format!(
                "test target {t} is not {}/tests/{t}.rs",
                crate_dir.display()
            ));
        }
    }
    let mut paths = Vec::new();
    rust_files(&crate_dir, &mut paths);
    let files: Vec<(PathBuf, String)> = paths
        .into_iter()
        .filter_map(|p| std::fs::read_to_string(&p).ok().map(|t| (p, t)))
        .collect();
    for filter in filters {
        for segment in filter.split("::").filter(|s| !s.is_empty()) {
            if !names_test_item(&files, segment) {
                return Err(format!("test filter {filter}: no fn or mod `{segment}`"));
            }
        }
    }
    Ok(())
}

fn is_test_command(span: &str) -> bool {
    span.contains("cargo test ") || span.contains("remote-cargo.sh test ")
}

/// Breaks if a page is missing, loses one of the four sections, or names a file, package, test
/// target or test that does not exist in the tree.
#[test]
fn docs_extending_pages_exist() {
    let dir = root().join("docs/extending");
    let index = std::fs::read_to_string(dir.join("README.md")).expect("docs/extending/README.md");
    let mut failures = Vec::new();
    for page in PAGES {
        let file = format!("{page}.md");
        if !index.contains(&file) {
            failures.push(format!("README.md does not link {file}"));
        }
        let Ok(text) = std::fs::read_to_string(dir.join(&file)) else {
            failures.push(format!("{file}: missing"));
            continue;
        };
        for heading in SECTIONS {
            if !text.contains(&format!("\n{heading}\n")) {
                failures.push(format!("{file}: no section `{heading}`"));
            }
        }
        for span in code_spans(&text) {
            if is_tree_path(&span) && !path_exists(&span) {
                failures.push(format!("{file}: `{span}` does not exist"));
            }
            if is_test_command(&span)
                && let Err(e) = check_test_command(&span)
            {
                failures.push(format!("{file}: `{span}`: {e}"));
            }
        }
        let registry = code_spans(section(&text, "## Registry entry"));
        if !registry.iter().any(|s| is_tree_path(s)) {
            failures.push(format!("{file}: `## Registry entry` names no file"));
        }
        let suite = code_spans(section(&text, "## Conformance suite"));
        if !suite.iter().any(|s| is_test_command(s)) {
            failures.push(format!(
                "{file}: `## Conformance suite` names no test command"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The checker itself: a placeholder matches, a missing file or test does not.
#[test]
fn docs_checker_rejects_what_does_not_exist() {
    assert!(path_exists("crates/turbine-scheduler/src/policy/<name>.rs"));
    assert!(path_exists("crates/turbine-model/src/sampling/mod.rs"));
    assert!(!path_exists(
        "crates/turbine-scheduler/src/policy/<name>.py"
    ));
    assert!(!path_exists("crates/turbine-model/src/no_such_file.rs"));
    check_test_command("cargo test -p turbine-scheduler registry_conformance").unwrap();
    check_test_command("scripts/remote-cargo.sh test -p turbine-model --test docs_extending")
        .unwrap();
    assert!(check_test_command("cargo test -p turbine-scheduler no_such_test_anywhere").is_err());
    assert!(check_test_command("cargo test -p turbine-model --test no_such_target").is_err());
    assert!(check_test_command("cargo test -p no-such-crate x").is_err());
    let spans = code_spans("a `x` b `y`\n```\n`z`\n```\n");
    assert_eq!(spans, ["x", "y"]);
}
