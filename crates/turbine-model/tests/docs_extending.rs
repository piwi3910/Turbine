//! Phase 2m S-12: `docs/extending/` has one how-to per extension point, and every page stays
//! true to the tree. Each page has the four sections a module author follows (files to add, the
//! registry entry, the conformance suite, the lab checks); every backticked path it names under
//! `crates/`, `kernels/` or `docs/` exists (a `<name>` placeholder segment matches any entry of
//! that directory with the same prefix and suffix); every backticked `cargo test` (or
//! `scripts/remote-cargo.sh test`) command names an existing package, test target and a test
//! filter whose every `::` segment is a test function or module in the tree. The `Registry
//! entry` section names at least one existing file and the `Conformance suite` section at least
//! one test command, so a page cannot drop either.
//!
//! `docs/support-matrix.md` is the human view of `turbine_core::support::SUPPORT_MATRIX`; the
//! `docs_support_matrix_*` tests render the table, the gfx1201 resolution, the tier formats and
//! the deferred / parallel refusals from code and require the page to carry exactly those.

use std::path::{Path, PathBuf};

use turbine_core::support::{
    self, DEFERRED_VENDORS, KvFormatColumn, PARALLEL_REFUSALS, SUPPORT_MATRIX, SpeculativeColumn,
    SupportKey, TIER_FORMAT_REFUSALS, WeightFormatColumn,
};

/// The thirteen extension points, one page each.
const PAGES: [&str; 13] = [
    "model-family",
    "tool-format",
    "weight-format",
    "kernel-implementation",
    "card-family",
    "backend",
    "logits-processor",
    "scheduling-policy",
    "eviction-policy",
    "kv-format",
    "collective-backend",
    "rank-transport",
    "dp-router-policy",
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
                package = Some(args.get(i + 1).copied().ok_or("-p without a package")?);
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
    // `-p` with no package name is refused, like `--test` without a name (Scout 6c7fc8ed).
    assert!(check_test_command("cargo test -p").is_err());
    let spans = code_spans("a `x` b `y`\n```\n`z`\n```\n");
    assert_eq!(spans, ["x", "y"]);
}

/// The text of `docs/support-matrix.md`.
fn support_matrix_doc() -> String {
    std::fs::read_to_string(root().join("docs/support-matrix.md")).expect("docs/support-matrix.md")
}

/// The text from `heading` up to the next `## `, `### ` or end of file (the existing `section`
/// helper stops at `## ` only).
fn table_under<'a>(text: &'a str, heading: &str) -> &'a str {
    let start = text
        .find(heading)
        .unwrap_or_else(|| panic!("{heading} not found"))
        + heading.len();
    let rest = &text[start..];
    let end = ["\n### ", "\n## "]
        .iter()
        .filter_map(|s| rest.find(s))
        .min()
        .unwrap_or(rest.len());
    &rest[..end]
}

/// The stable part of a refusal reason this page must name: the `phase-*` track it names, else
/// the `reason_code:` prefix.
fn reason_anchor(reason: &str) -> Option<String> {
    if let Some(i) = reason.find("phase-") {
        let rest = &reason[i..];
        let end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '-'))
            .unwrap_or(rest.len());
        return Some(rest[..end].trim_end_matches('-').to_string());
    }
    let code = reason.split(':').next().unwrap_or("");
    if !code.is_empty()
        && code
            .chars()
            .all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit())
    {
        return Some(code.to_string());
    }
    None
}

/// The line with runs of whitespace collapsed to one space: the formatter pads table cells to
/// align the columns, so the comparison ignores that padding.
fn squash(line: &str) -> String {
    line.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Fails unless the doc's table rows (the lines starting with `` | ` ``) are exactly `expected`,
/// in order, each doc line starting with the expected prefix (the page may append prose notes).
fn expect_table(failures: &mut Vec<String>, what: &str, doc_rows: &[&str], expected: &[String]) {
    if doc_rows.len() != expected.len() {
        failures.push(format!(
            "{what}: the page has {} table rows, the code says {}\nthe code's table:\n{}",
            doc_rows.len(),
            expected.len(),
            expected.join("\n")
        ));
        return;
    }
    for (i, expected) in expected.iter().enumerate() {
        if !squash(doc_rows[i]).starts_with(&squash(expected)) {
            failures.push(format!(
                "{what} row {i}: expected {expected:?}, found {:?}",
                doc_rows[i]
            ));
        }
    }
}

fn table_rows(body: &str) -> Vec<&str> {
    body.lines().filter(|l| l.starts_with("| `")).collect()
}

/// docs/support-matrix.md carries every `SUPPORT_MATRIX` row (key, status and, for a refusal,
/// the reason anchor) and no contradiction: a status changed in code, a row added or dropped,
/// or an edited status in the page turns this red.
#[test]
fn docs_support_matrix_rows_match_code() {
    let doc = support_matrix_doc();
    let body = section(&doc, "## Every row of the table");
    let expected: Vec<String> = SUPPORT_MATRIX
        .iter()
        .map(|r| {
            let v = r.view();
            format!(
                "| `{}/{}/{}/{}/{}/{}` | {}",
                v.vendor,
                v.arch,
                v.architecture,
                v.weight_format,
                v.kv_format,
                v.speculative,
                v.status
            )
        })
        .collect();
    let mut failures = Vec::new();
    let doc_rows = table_rows(body);
    if doc_rows.len() != expected.len() {
        failures.push(format!(
            "row table: the page has {} rows, SUPPORT_MATRIX has {}\nexpected:\n{}",
            doc_rows.len(),
            expected.len(),
            expected.join("\n")
        ));
        return;
    }
    for (i, (row, expected)) in SUPPORT_MATRIX.iter().zip(expected.iter()).enumerate() {
        let line = squash(doc_rows[i]);
        if !line.starts_with(&squash(expected)) {
            failures.push(format!("row {i}: expected {expected:?}, found {line:?}"));
            continue;
        }
        if let Some(reason) = row.status.reason()
            && let Some(anchor) = reason_anchor(reason)
            && !line.contains(&anchor)
        {
            failures.push(format!(
                "row {i}: the reason anchor `{anchor}` is missing: {line}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The gfx1201 resolution tables are `support::resolve` over every weight × KV combination, per
/// architecture: an edited status or a resolution change in the rows shows up here.
#[test]
fn docs_support_matrix_gfx1201_tables_match_resolution() {
    let doc = support_matrix_doc();
    let mut failures = Vec::new();
    for architecture in ["LlamaForCausalLM", "OlmoeForCausalLM"] {
        let doc_rows = table_rows(table_under(&doc, &format!("### {architecture}")));
        let expected: Vec<String> = WeightFormatColumn::ALL
            .iter()
            .map(|w| {
                let mut line = format!("| `{}`", w.as_str());
                for kv in KvFormatColumn::ALL {
                    let key = SupportKey::for_model(
                        "amd",
                        "gfx1201",
                        architecture,
                        *w,
                        kv,
                        SpeculativeColumn::None,
                    );
                    line.push_str(&format!(" | {}", support::resolve(&key).as_str()));
                }
                line.push_str(" |");
                line
            })
            .collect();
        expect_table(&mut failures, architecture, &doc_rows, &expected);
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The lower-tier format table is `TIER_FORMAT_REFUSALS` plus the formats that are `supported`
/// by default (`l0` and the KV columns proven as lower tiers).
#[test]
fn docs_support_matrix_tier_formats_match_code() {
    let doc = support_matrix_doc();
    let doc_rows = table_rows(section(&doc, "## Lower-tier KV formats"));
    let mut expected: Vec<String> = Vec::new();
    for format in ["l0", "fp8_e4m3", "tq4"] {
        let status = support::check_tier_format("kv.cpu.format", format).unwrap();
        expected.push(format!("| `{format}` | {}", status.as_str()));
    }
    for refusal in TIER_FORMAT_REFUSALS {
        expected.push(format!(
            "| `{}` | {}",
            refusal.format,
            refusal.status.as_str()
        ));
    }
    let mut failures = Vec::new();
    expect_table(&mut failures, "tier formats", &doc_rows, &expected);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The deferred-vendor and parallel-refusal sections name every entry of `DEFERRED_VENDORS` and
/// `PARALLEL_REFUSALS` with its status and reason code.
#[test]
fn docs_support_matrix_deferred_and_parallel_match_code() {
    let doc = support_matrix_doc();
    let mut failures = Vec::new();
    let deferred = section(&doc, "## Deferred vendors");
    for (vendor, reason) in DEFERRED_VENDORS {
        let needle = format!("| `{vendor}` |");
        let line = deferred
            .lines()
            .find(|l| squash(l).contains(&needle))
            .unwrap_or_else(|| {
                failures.push(format!("deferred vendor `{vendor}`: no table row"));
                ""
            });
        if !line.contains("unsupported") {
            failures.push(format!(
                "deferred vendor `{vendor}`: status not `unsupported`: {line:?}"
            ));
        }
        if let Some(anchor) = reason_anchor(reason)
            && !line.contains(&anchor)
        {
            failures.push(format!(
                "deferred vendor `{vendor}`: reason anchor `{anchor}` missing: {line:?}"
            ));
        }
    }
    let parallel = section(&doc, "## Parallel-mode refusals");
    for refusal in PARALLEL_REFUSALS {
        let code = refusal.reason.split(':').next().unwrap_or("");
        let line = parallel
            .lines()
            .find(|l| {
                let line = squash(l);
                line.contains(refusal.architecture) && line.contains(refusal.modes)
            })
            .unwrap_or_else(|| {
                failures.push(format!(
                    "parallel refusal {} {}: no table row",
                    refusal.architecture, refusal.modes
                ));
                ""
            });
        if !code.is_empty() && !line.contains(code) {
            failures.push(format!(
                "parallel refusal {} {}: reason code `{code}` missing",
                refusal.architecture, refusal.modes
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The checker itself: the reason anchor picks the track or the code, not prose.
#[test]
fn docs_support_matrix_reason_anchors() {
    assert_eq!(
        reason_anchor(
            "this quantized weight format is not validated yet (track phase-6a-quantization)"
        )
        .as_deref(),
        Some("phase-6a-quantization")
    );
    assert_eq!(
        reason_anchor("kv_tq2_l0_refused: TurboQuant 2-bit L0 pages fail the quality gate")
            .as_deref(),
        Some("kv_tq2_l0_refused")
    );
    assert_eq!(reason_anchor("no reason at all"), None);
    expect_table(
        &mut Vec::new(),
        "x",
        &["| `a` | supported | note |"],
        &["| `a` | supported".to_string()],
    );
    let mut failures = Vec::new();
    expect_table(
        &mut failures,
        "x",
        &["| `a` | experimental | note |"],
        &["| `a` | supported".to_string()],
    );
    assert!(failures.len() == 1 && failures[0].contains("expected"));
}
