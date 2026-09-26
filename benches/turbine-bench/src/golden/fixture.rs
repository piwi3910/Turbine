//! Golden fixture formats (P1 §Data, contract §21.4): prompts, reference records, tolerance.

use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Errors of the golden tooling. The variant decides the process exit code.
#[derive(Debug, thiserror::Error)]
pub enum GoldenError {
    /// Bad flags or malformed fixture files: exit 2.
    #[error("{0}")]
    Usage(String),
    /// Local file I/O failed: exit 2.
    #[error("{0}")]
    Io(String),
    /// The endpoint was unreachable or answered with an error or an unusable body: exit 1.
    #[error("{0}")]
    Endpoint(String),
}

impl GoldenError {
    pub fn exit_code(&self) -> u8 {
        match self {
            GoldenError::Usage(_) | GoldenError::Io(_) => 2,
            GoldenError::Endpoint(_) => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptKind {
    /// Replayed against `/v1/completions` with `prompt`.
    Completion,
    /// Replayed against `/v1/chat/completions` with `messages`.
    Chat,
}

/// One line of `tests/golden/prompts.jsonl`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromptRecord {
    pub id: String,
    pub kind: PromptKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<Value>>,
    pub max_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<Map<String, Value>>,
}

/// One line of `tests/golden/<slug>/reference.jsonl`. `top_logprobs[pos]` holds
/// `[token_id, logprob]` pairs, highest logprob first.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReferenceRecord {
    pub id: String,
    pub engine: String,
    pub model: String,
    pub captured: String,
    pub prompt_token_ids: Vec<u32>,
    pub tokens: Vec<u32>,
    pub top_logprobs: Vec<Vec<(u32, f32)>>,
}

/// `tests/golden/<slug>/tolerance.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tolerance {
    /// Greedy tokens that must match from position 0 (capped at the reference length).
    pub min_identical_prefix: usize,
    /// Prompts that must pass the prefix rule.
    pub min_prompts_passing: usize,
    /// Reference top-k entries whose logprobs are compared.
    pub top_k: usize,
    /// Max |Δ logprob| (nats) before the first divergence for reference top-k candidates whose
    /// reference logprob is above `likely_logprob_floor`.
    pub max_abs_logprob_diff_likely: f32,
    /// Max |Δ logprob| (nats) before the first divergence for the other (tail) reference top-k
    /// candidates, whose logprob is at or below `likely_logprob_floor`.
    pub max_abs_logprob_diff_tail: f32,
    /// Reference logprob (nats) splitting likely from tail candidates; a candidate exactly at
    /// the floor is a tail candidate.
    pub likely_logprob_floor: f32,
    /// A divergence is excused when the reference top-1/top-2 margin there is below this (nats).
    pub margin_nats: f32,
    /// `max_abs_logprob_diff_likely` for runs with more than one prompt in flight (batch
    /// composition changes GEMM rounding); absent → the strict bound applies there too.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_abs_logprob_diff_likely_batched: Option<f32>,
    /// `max_abs_logprob_diff_tail` for runs with more than one prompt in flight; absent → the
    /// strict bound applies there too.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_abs_logprob_diff_tail_batched: Option<f32>,
}

/// The logprob bounds one `compare` run is judged by (see [`Tolerance::logprob_bounds`]).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct LogprobBounds {
    /// A `*_batched` bound from the tolerance file applies (concurrency > 1 and the key present).
    pub batched: bool,
    pub max_abs_logprob_diff_likely: f32,
    pub max_abs_logprob_diff_tail: f32,
}

impl Tolerance {
    /// The bounds for a run with `concurrency` prompts in flight: the strict ones at 1 (or 0);
    /// above 1 each tier takes its `*_batched` bound, falling back to the strict one when the
    /// key is absent. The token rule (`min_identical_prefix`, `min_prompts_passing`,
    /// `margin_nats`) is the same at every concurrency.
    pub fn logprob_bounds(&self, concurrency: usize) -> LogprobBounds {
        let (likely, tail) = if concurrency > 1 {
            (
                self.max_abs_logprob_diff_likely_batched,
                self.max_abs_logprob_diff_tail_batched,
            )
        } else {
            (None, None)
        };
        LogprobBounds {
            batched: likely.is_some() || tail.is_some(),
            max_abs_logprob_diff_likely: likely.unwrap_or(self.max_abs_logprob_diff_likely),
            max_abs_logprob_diff_tail: tail.unwrap_or(self.max_abs_logprob_diff_tail),
        }
    }

    /// This tolerance with the strict bounds replaced by [`Self::logprob_bounds`] for
    /// `concurrency`, the form [`super::compare_prompt`] judges a prompt with.
    pub fn at_concurrency(&self, concurrency: usize) -> Tolerance {
        let bounds = self.logprob_bounds(concurrency);
        Tolerance {
            max_abs_logprob_diff_likely: bounds.max_abs_logprob_diff_likely,
            max_abs_logprob_diff_tail: bounds.max_abs_logprob_diff_tail,
            ..self.clone()
        }
    }
}

/// Read a JSONL file (blank lines skipped); errors name the file and line.
pub fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>, GoldenError> {
    let text = fs::read_to_string(path)
        .map_err(|e| GoldenError::Io(format!("cannot read {}: {e}", path.display())))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let rec = serde_json::from_str(line)
            .map_err(|e| GoldenError::Usage(format!("{}:{}: {e}", path.display(), i + 1)))?;
        out.push(rec);
    }
    if out.is_empty() {
        return Err(GoldenError::Usage(format!(
            "{} holds no records",
            path.display()
        )));
    }
    Ok(out)
}

/// Read and validate a prompts file: unique ids, `prompt` for completions, `messages` for chat.
pub fn read_prompts(path: &Path) -> Result<Vec<PromptRecord>, GoldenError> {
    let prompts: Vec<PromptRecord> = read_jsonl(path)?;
    let mut seen = HashSet::new();
    for p in &prompts {
        if !seen.insert(p.id.as_str()) {
            return Err(GoldenError::Usage(format!(
                "{}: duplicate prompt id {}",
                path.display(),
                p.id
            )));
        }
        let ok = match p.kind {
            PromptKind::Completion => p.prompt.is_some() && p.messages.is_none(),
            PromptKind::Chat => p.messages.is_some() && p.prompt.is_none(),
        };
        if !ok {
            return Err(GoldenError::Usage(format!(
                "{}: prompt {} of kind {:?} needs exactly one of \"prompt\" (completion) or \"messages\" (chat)",
                path.display(),
                p.id,
                p.kind
            )));
        }
    }
    Ok(prompts)
}

/// Read a tolerance file (unknown keys are refused).
pub fn read_tolerance(path: &Path) -> Result<Tolerance, GoldenError> {
    let text = fs::read_to_string(path)
        .map_err(|e| GoldenError::Io(format!("cannot read {}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| GoldenError::Usage(format!("{}: {e}", path.display())))
}

/// Write one JSON object per line to `<path>.tmp`, then rename it over `path`; on any error the
/// temp file is removed and `path` is left untouched.
pub fn write_jsonl_atomic<T: Serialize>(path: &Path, records: &[T]) -> Result<(), GoldenError> {
    let tmp = tmp_path(path);
    let result = (|| {
        let mut f = fs::File::create(&tmp)?;
        for r in records {
            let line = serde_json::to_string(r).map_err(std::io::Error::other)?;
            f.write_all(line.as_bytes())?;
            f.write_all(b"\n")?;
        }
        f.sync_all()?;
        fs::rename(&tmp, path)
    })();
    result.map_err(|e| {
        let _ = fs::remove_file(&tmp);
        GoldenError::Io(format!("cannot write {}: {e}", path.display()))
    })
}

/// `<path>.tmp`.
pub fn tmp_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    path.with_file_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "turbine-golden-fixture-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn fixture_formats_roundtrip() {
        let dir = temp_dir("roundtrip");
        let prompts = dir.join("prompts.jsonl");
        fs::write(
            &prompts,
            concat!(
                r#"{"id":"p01","kind":"completion","prompt":"Hello","max_tokens":32,"chat_template_kwargs":{"date_string":"26 Jul 2024"}}"#,
                "\n\n",
                r#"{"id":"p02","kind":"chat","messages":[{"role":"user","content":"Hi"}],"max_tokens":8}"#,
                "\n"
            ),
        )
        .unwrap();
        let p = read_prompts(&prompts).unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].kind, PromptKind::Completion);
        assert_eq!(p[0].prompt.as_deref(), Some("Hello"));
        assert_eq!(
            p[0].chat_template_kwargs.as_ref().unwrap()["date_string"],
            "26 Jul 2024"
        );
        assert_eq!(p[1].kind, PromptKind::Chat);
        assert_eq!(p[1].messages.as_ref().unwrap().len(), 1);

        let reference = ReferenceRecord {
            id: "p01".into(),
            engine: "transformers-5.17.0-bf16-cpu".into(),
            model: "meta-llama/Llama-3.2-3B-Instruct".into(),
            captured: "2026-09-25T12:00:00Z".into(),
            prompt_token_ids: vec![128000, 9906],
            tokens: vec![11, 1917],
            top_logprobs: vec![vec![(11, -0.25), (13, -1.5)], vec![(1917, -0.5)]],
        };
        let out = dir.join("reference.jsonl");
        write_jsonl_atomic(&out, std::slice::from_ref(&reference)).unwrap();
        assert!(!tmp_path(&out).exists());
        let line = fs::read_to_string(&out).unwrap();
        assert!(
            line.contains(r#""top_logprobs":[[[11,-0.25],[13,-1.5]],[[1917,-0.5]]]"#),
            "{line}"
        );
        let back: Vec<ReferenceRecord> = read_jsonl(&out).unwrap();
        assert_eq!(back, vec![reference]);

        let tol = dir.join("tolerance.json");
        fs::write(
            &tol,
            r#"{"min_identical_prefix":32,"min_prompts_passing":14,"top_k":5,"max_abs_logprob_diff_likely":0.15,"max_abs_logprob_diff_tail":0.55,"likely_logprob_floor":-2.0,"margin_nats":0.5}"#,
        )
        .unwrap();
        let t = read_tolerance(&tol).unwrap();
        assert_eq!(t.min_identical_prefix, 32);
        assert_eq!(t.min_prompts_passing, 14);
        assert_eq!(t.top_k, 5);
        assert_eq!(t.max_abs_logprob_diff_likely, 0.15);
        assert_eq!(t.max_abs_logprob_diff_tail, 0.55);
        assert_eq!(t.likely_logprob_floor, -2.0);
        assert_eq!(t.max_abs_logprob_diff_likely_batched, None);
        assert_eq!(t.max_abs_logprob_diff_tail_batched, None);
        assert_eq!(t.margin_nats, 0.5);
        fs::remove_dir_all(&dir).unwrap();
    }

    /// Breaks if the batched bounds apply at concurrency 1, or if one absent batched key drops
    /// the other tier's batched bound or its strict fallback.
    #[test]
    fn batched_bounds_fall_back_per_tier() {
        let strict: Tolerance = serde_json::from_str(
            r#"{"min_identical_prefix":32,"min_prompts_passing":14,"top_k":5,"max_abs_logprob_diff_likely":0.15,"max_abs_logprob_diff_tail":0.55,"likely_logprob_floor":-2.0,"margin_nats":0.5}"#,
        )
        .unwrap();
        let both = Tolerance {
            max_abs_logprob_diff_likely_batched: Some(0.25),
            max_abs_logprob_diff_tail_batched: Some(0.75),
            ..strict.clone()
        };
        let likely_only = Tolerance {
            max_abs_logprob_diff_likely_batched: Some(0.25),
            ..strict.clone()
        };
        let b = |likely, tail, batched| LogprobBounds {
            batched,
            max_abs_logprob_diff_likely: likely,
            max_abs_logprob_diff_tail: tail,
        };
        for c in [0, 1] {
            assert_eq!(
                both.logprob_bounds(c),
                b(0.15, 0.55, false),
                "concurrency {c}"
            );
        }
        assert_eq!(both.logprob_bounds(16), b(0.25, 0.75, true));
        assert_eq!(likely_only.logprob_bounds(2), b(0.25, 0.55, true));
        assert_eq!(strict.logprob_bounds(16), b(0.15, 0.55, false));
        let t = both.at_concurrency(16);
        assert_eq!(
            (t.max_abs_logprob_diff_likely, t.max_abs_logprob_diff_tail),
            (0.25, 0.75)
        );
        assert_eq!(both.at_concurrency(1), both);
    }

    #[test]
    fn malformed_fixtures_are_usage_errors() {
        let dir = temp_dir("malformed");
        let cases = [
            (
                "dup.jsonl",
                concat!(
                    r#"{"id":"p01","kind":"completion","prompt":"a","max_tokens":1}"#,
                    "\n",
                    r#"{"id":"p01","kind":"completion","prompt":"b","max_tokens":1}"#
                ),
                "duplicate prompt id p01",
            ),
            (
                "chat-without-messages.jsonl",
                r#"{"id":"p02","kind":"chat","prompt":"a","max_tokens":1}"#,
                "needs exactly one of",
            ),
            (
                "bad-kind.jsonl",
                r#"{"id":"p03","kind":"embedding","prompt":"a","max_tokens":1}"#,
                "bad-kind.jsonl:1",
            ),
            ("empty.jsonl", "\n", "holds no records"),
        ];
        for (name, body, needle) in cases {
            let path = dir.join(name);
            fs::write(&path, body).unwrap();
            let err = read_prompts(&path).unwrap_err();
            assert_eq!(err.exit_code(), 2, "{name}: {err}");
            assert!(err.to_string().contains(needle), "{name}: {err}");
        }
        let tol = dir.join("tolerance.json");
        fs::write(&tol, r#"{"min_identical_prefix":32,"unknown":1}"#).unwrap();
        assert_eq!(read_tolerance(&tol).unwrap_err().exit_code(), 2);
        let missing = read_prompts(&dir.join("absent.jsonl")).unwrap_err();
        assert!(matches!(missing, GoldenError::Io(_)), "{missing}");
        fs::remove_dir_all(&dir).unwrap();
    }
}
