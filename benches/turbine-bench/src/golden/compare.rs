//! Judging a candidate generation against a reference record (P1 S-11, tolerance.json).

use serde::Serialize;

use super::fixture::{ReferenceRecord, Tolerance};

/// A reference top-k id absent from the candidate's top list at a compared position.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct MissingTopK {
    pub position: usize,
    pub token_id: u32,
}

/// Per-prompt result of [`compare_prompt`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PromptVerdict {
    pub id: String,
    /// Greedy positions in the reference.
    pub reference_len: usize,
    /// Leading positions where the candidate chose the reference token.
    pub identical_prefix: usize,
    /// First position where the candidate differs from (or stops before) the reference.
    pub first_divergence: Option<usize>,
    /// Reference top-1 minus top-2 logprob (nats) at `first_divergence`.
    pub margin_at_divergence: Option<f32>,
    /// Max |Δ logprob| over the reference top-k at every position before the divergence.
    pub max_abs_logprob_diff: f32,
    /// First reference top-k id the candidate's top list did not contain, if any.
    pub missing_top_k: Option<MissingTopK>,
    /// No missing id and `max_abs_logprob_diff` ≤ the tolerance.
    pub logprob_within_bound: bool,
    /// The prefix rule (or a small-margin excuse) holds and the logprob bound holds.
    pub passed: bool,
}

/// Top-1 minus top-2 logprob of one reference row, whatever its order.
fn margin(row: &[(u32, f32)]) -> Option<f32> {
    let mut lps: Vec<f32> = row.iter().map(|e| e.1).collect();
    lps.sort_by(|a, b| b.total_cmp(a));
    (lps.len() >= 2).then(|| lps[0] - lps[1])
}

/// The `k` highest-logprob entries of one reference row.
fn top_k(row: &[(u32, f32)], k: usize) -> Vec<(u32, f32)> {
    let mut sorted = row.to_vec();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    sorted.truncate(k);
    sorted
}

/// Compare a candidate's greedy tokens and top logprobs with one reference record.
///
/// A candidate that stops early diverges at its length; a reference top-k id missing from the
/// candidate's top list at a compared position violates the logprob bound.
pub fn compare_prompt(
    reference: &ReferenceRecord,
    got_tokens: &[u32],
    got_top: &[Vec<(u32, f32)>],
    tol: &Tolerance,
) -> PromptVerdict {
    let n = reference.tokens.len();
    let identical_prefix = reference
        .tokens
        .iter()
        .zip(got_tokens)
        .take_while(|(r, g)| r == g)
        .count();
    let first_divergence = (identical_prefix < n).then_some(identical_prefix);
    let margin_at_divergence = first_divergence
        .and_then(|d| reference.top_logprobs.get(d))
        .and_then(|row| margin(row));

    let mut max_abs_logprob_diff = 0.0f32;
    let mut missing_top_k = None;
    for pos in 0..first_divergence.unwrap_or(n) {
        let Some(ref_row) = reference.top_logprobs.get(pos) else {
            break;
        };
        let got_row = got_top.get(pos).map(Vec::as_slice).unwrap_or(&[]);
        for (id, lp) in top_k(ref_row, tol.top_k) {
            match got_row.iter().find(|e| e.0 == id) {
                Some(&(_, got_lp)) => {
                    max_abs_logprob_diff = max_abs_logprob_diff.max((got_lp - lp).abs())
                }
                None => {
                    missing_top_k.get_or_insert(MissingTopK {
                        position: pos,
                        token_id: id,
                    });
                }
            }
        }
    }
    let logprob_within_bound =
        missing_top_k.is_none() && max_abs_logprob_diff <= tol.max_abs_logprob_diff;
    let prefix_ok = identical_prefix >= tol.min_identical_prefix.min(n);
    let excused = margin_at_divergence.is_some_and(|m| m < tol.margin_nats);

    PromptVerdict {
        id: reference.id.clone(),
        reference_len: n,
        identical_prefix,
        first_divergence,
        margin_at_divergence,
        max_abs_logprob_diff,
        missing_top_k,
        logprob_within_bound,
        passed: (prefix_ok || excused) && logprob_within_bound,
    }
}

/// The whole set holds: the logprob bound on every prompt and enough prompts passing.
pub fn judge(verdicts: &[PromptVerdict], tol: &Tolerance) -> bool {
    !verdicts.is_empty()
        && verdicts.iter().all(|v| v.logprob_within_bound)
        && verdicts.iter().filter(|v| v.passed).count() >= tol.min_prompts_passing
}

/// The `compare` report (`--output json` serialises it as is).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CompareReport {
    pub prompts: Vec<PromptVerdict>,
    pub prompts_passing: usize,
    pub prompts_total: usize,
    pub passed: bool,
    pub tolerance: Tolerance,
}

impl CompareReport {
    pub fn new(prompts: Vec<PromptVerdict>, tolerance: &Tolerance) -> Self {
        Self {
            prompts_passing: prompts.iter().filter(|v| v.passed).count(),
            prompts_total: prompts.len(),
            passed: judge(&prompts, tolerance),
            tolerance: tolerance.clone(),
            prompts,
        }
    }

    /// One line per prompt, then the verdict.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        for v in &self.prompts {
            let divergence = v
                .first_divergence
                .map_or_else(|| "none".to_string(), |d| d.to_string());
            let margin = v
                .margin_at_divergence
                .map_or_else(|| "-".to_string(), |m| format!("{m:.3}"));
            out.push_str(&format!(
                "{} {} identical_prefix={}/{} first_divergence={divergence} margin={margin} max_abs_logprob_diff={:.4}",
                if v.passed { "PASS" } else { "FAIL" },
                v.id,
                v.identical_prefix,
                v.reference_len,
                v.max_abs_logprob_diff,
            ));
            if let Some(m) = v.missing_top_k {
                out.push_str(&format!(
                    " missing_top_k=token {} at position {}",
                    m.token_id, m.position
                ));
            }
            out.push('\n');
        }
        let bound_held = self.prompts.iter().all(|v| v.logprob_within_bound);
        out.push_str(&format!(
            "{}: {}/{} prompts passing (need {}); |Δ logprob| over top-{} ≤ {} on every prompt: {}\n",
            if self.passed { "PASS" } else { "FAIL" },
            self.prompts_passing,
            self.prompts_total,
            self.tolerance.min_prompts_passing,
            self.tolerance.top_k,
            self.tolerance.max_abs_logprob_diff,
            if bound_held { "yes" } else { "no" },
        ));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tol() -> Tolerance {
        Tolerance {
            min_identical_prefix: 4,
            min_prompts_passing: 1,
            top_k: 2,
            max_abs_logprob_diff: 0.15,
            margin_nats: 0.5,
        }
    }

    /// Four positions; token 10+i, runner-up 20+i with the given margins.
    fn reference(margins: [f32; 4]) -> ReferenceRecord {
        ReferenceRecord {
            id: "p".into(),
            engine: "e".into(),
            model: "m".into(),
            captured: "2026-09-25T00:00:00Z".into(),
            prompt_token_ids: vec![1],
            tokens: (0..4).map(|i| 10 + i).collect(),
            top_logprobs: margins
                .iter()
                .enumerate()
                .map(|(i, m)| vec![(10 + i as u32, -0.5), (20 + i as u32, -0.5 - m), (99, -9.0)])
                .collect(),
        }
    }

    #[test]
    fn early_stop_diverges_at_its_length() {
        let r = reference([1.0, 1.0, 0.2, 1.0]);
        let v = compare_prompt(&r, &r.tokens[..2], &r.top_logprobs[..2], &tol());
        assert_eq!(v.identical_prefix, 2);
        assert_eq!(v.first_divergence, Some(2));
        assert!((v.margin_at_divergence.unwrap() - 0.2).abs() < 1e-6);
        assert!(v.passed, "small margin excuses the early stop: {v:?}");
    }

    #[test]
    fn missing_top_k_id_violates_the_bound() {
        let r = reference([1.0; 4]);
        let mut top = r.top_logprobs.clone();
        top[1].retain(|e| e.0 != 21);
        let v = compare_prompt(&r, &r.tokens, &top, &tol());
        assert_eq!(
            v.missing_top_k,
            Some(MissingTopK {
                position: 1,
                token_id: 21
            })
        );
        assert!(!v.logprob_within_bound && !v.passed);
        assert!(!judge(&[v], &tol()));
        // An id outside the reference top-k (99 is third) may be missing.
        let mut top = r.top_logprobs.clone();
        top[1].retain(|e| e.0 != 99);
        assert!(compare_prompt(&r, &r.tokens, &top, &tol()).passed);
    }

    #[test]
    fn logprobs_after_divergence_are_not_compared() {
        let r = reference([1.0; 4]);
        let mut tokens = r.tokens.clone();
        tokens[2] = 22;
        let mut top = r.top_logprobs.clone();
        top[3][0].1 = -5.0;
        let v = compare_prompt(&r, &tokens, &top, &tol());
        assert_eq!(v.max_abs_logprob_diff, 0.0);
        assert!(v.logprob_within_bound);
        assert!(!v.passed, "margin 1.0 is not excused");
        // The prefix requirement is capped at the reference length.
        let t = Tolerance {
            min_identical_prefix: 32,
            ..tol()
        };
        assert!(compare_prompt(&r, &r.tokens, &r.top_logprobs, &t).passed);
    }

    #[test]
    fn judge_counts_passing_prompts() {
        let r = reference([1.0; 4]);
        let pass = compare_prompt(&r, &r.tokens, &r.top_logprobs, &tol());
        let mut tokens = r.tokens.clone();
        tokens[0] = 0;
        let fail = compare_prompt(&r, &tokens, &r.top_logprobs, &tol());
        let two = Tolerance {
            min_prompts_passing: 2,
            ..tol()
        };
        assert!(judge(&[pass.clone(), fail.clone()], &tol()));
        assert!(!judge(&[pass, fail], &two));
        assert!(!judge(&[], &tol()));
    }
}
