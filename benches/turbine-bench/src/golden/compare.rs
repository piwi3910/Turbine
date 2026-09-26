//! Judging a candidate generation against a reference record (P1 S-11, tolerance.json).

use serde::Serialize;

use super::fixture::{LogprobBounds, ReferenceRecord, Tolerance};

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
    /// Max |Δ logprob| before the divergence over the reference top-k candidates above the
    /// tolerance's `likely_logprob_floor`.
    pub max_abs_logprob_diff_likely: f32,
    /// Max |Δ logprob| before the divergence over the other (tail) reference top-k candidates.
    pub max_abs_logprob_diff_tail: f32,
    /// First reference top-k id the candidate's top list did not contain, if any.
    pub missing_top_k: Option<MissingTopK>,
    /// No missing id and both tiers within their tolerance bound.
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
/// candidate's top list at a compared position violates the logprob bound. Each reference
/// top-k candidate is judged by its reference logprob: above `likely_logprob_floor` against
/// `max_abs_logprob_diff_likely`, at or below it against `max_abs_logprob_diff_tail`.
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

    let mut max_abs_logprob_diff_likely = 0.0f32;
    let mut max_abs_logprob_diff_tail = 0.0f32;
    let mut missing_top_k = None;
    for pos in 0..first_divergence.unwrap_or(n) {
        let Some(ref_row) = reference.top_logprobs.get(pos) else {
            break;
        };
        let got_row = got_top.get(pos).map(Vec::as_slice).unwrap_or(&[]);
        for (id, lp) in top_k(ref_row, tol.top_k) {
            match got_row.iter().find(|e| e.0 == id) {
                Some(&(_, got_lp)) => {
                    let tier = if lp > tol.likely_logprob_floor {
                        &mut max_abs_logprob_diff_likely
                    } else {
                        &mut max_abs_logprob_diff_tail
                    };
                    *tier = tier.max((got_lp - lp).abs());
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
    let logprob_within_bound = missing_top_k.is_none()
        && max_abs_logprob_diff_likely <= tol.max_abs_logprob_diff_likely
        && max_abs_logprob_diff_tail <= tol.max_abs_logprob_diff_tail;
    let prefix_ok = identical_prefix >= tol.min_identical_prefix.min(n);
    let excused = margin_at_divergence.is_some_and(|m| m < tol.margin_nats);

    PromptVerdict {
        id: reference.id.clone(),
        reference_len: n,
        identical_prefix,
        first_divergence,
        margin_at_divergence,
        max_abs_logprob_diff_likely,
        max_abs_logprob_diff_tail,
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
    /// The tolerance file as read (batched keys included when present).
    pub tolerance: Tolerance,
    /// Prompts in flight at once during the run.
    pub concurrency: usize,
    /// The logprob bounds the prompts were judged by: strict at concurrency 1, the tolerance's
    /// batched bounds above it (`batched` says whether one applied).
    pub logprob_bounds: LogprobBounds,
}

impl CompareReport {
    /// `prompts` must have been judged with `tolerance.at_concurrency(concurrency)`.
    pub fn new(prompts: Vec<PromptVerdict>, tolerance: &Tolerance, concurrency: usize) -> Self {
        Self {
            prompts_passing: prompts.iter().filter(|v| v.passed).count(),
            prompts_total: prompts.len(),
            passed: judge(&prompts, tolerance),
            tolerance: tolerance.clone(),
            concurrency,
            logprob_bounds: tolerance.logprob_bounds(concurrency),
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
                "{} {} identical_prefix={}/{} first_divergence={divergence} margin={margin} max_abs_logprob_diff_likely={:.4} max_abs_logprob_diff_tail={:.4}",
                if v.passed { "PASS" } else { "FAIL" },
                v.id,
                v.identical_prefix,
                v.reference_len,
                v.max_abs_logprob_diff_likely,
                v.max_abs_logprob_diff_tail,
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
            "{}: {}/{} prompts passing (need {}); {} bounds (concurrency {}): |Δ logprob| over top-{} ≤ {} (reference logprob > {}) and ≤ {} (tail) on every prompt: {}\n",
            if self.passed { "PASS" } else { "FAIL" },
            self.prompts_passing,
            self.prompts_total,
            self.tolerance.min_prompts_passing,
            if self.logprob_bounds.batched {
                "batched"
            } else {
                "strict"
            },
            self.concurrency,
            self.tolerance.top_k,
            self.logprob_bounds.max_abs_logprob_diff_likely,
            self.tolerance.likely_logprob_floor,
            self.logprob_bounds.max_abs_logprob_diff_tail,
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
            max_abs_logprob_diff_likely: 0.15,
            max_abs_logprob_diff_tail: 0.55,
            likely_logprob_floor: -2.0,
            margin_nats: 0.5,
            max_abs_logprob_diff_likely_batched: None,
            max_abs_logprob_diff_tail_batched: None,
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
        assert_eq!(v.max_abs_logprob_diff_likely, 0.0);
        assert_eq!(v.max_abs_logprob_diff_tail, 0.0);
        assert!(v.logprob_within_bound);
        assert!(!v.passed, "margin 1.0 is not excused");
        // The prefix requirement is capped at the reference length.
        let t = Tolerance {
            min_identical_prefix: 32,
            ..tol()
        };
        assert!(compare_prompt(&r, &r.tokens, &r.top_logprobs, &t).passed);
    }

    /// The bound a top-k candidate gets depends on its reference logprob: above the floor (−2)
    /// it is likely (0.15), at the floor exactly or below it is tail (0.55). Breaks if the
    /// tier is chosen by the candidate's logprob, if −2 counts as likely, or if one bound
    /// applies to both tiers.
    #[test]
    fn logprob_bound_has_a_likely_and_a_tail_tier() {
        let t = tol();
        // One position: top-1 token 10, runner-up 20 at `ref_lp`; the candidate moves the
        // runner-up by `delta`.
        let check = |ref_lp: f32, delta: f32| {
            let r = ReferenceRecord {
                tokens: vec![10],
                top_logprobs: vec![vec![(10, -0.1), (20, ref_lp)]],
                ..reference([1.0; 4])
            };
            let mut top = r.top_logprobs.clone();
            top[0][1].1 += delta;
            compare_prompt(&r, &r.tokens, &top, &t)
        };
        let close = |a: f32, b: f32| (a - b).abs() < 1e-5;

        // Just above the floor: the likely bound.
        let v = check(-1.99, 0.1);
        assert!(v.logprob_within_bound && v.passed, "{v:?}");
        assert!(close(v.max_abs_logprob_diff_likely, 0.1), "{v:?}");
        assert_eq!(v.max_abs_logprob_diff_tail, 0.0);
        let v = check(-1.99, 0.5);
        assert!(!v.logprob_within_bound && !v.passed, "{v:?}");

        // Exactly at the floor and below it: the tail bound.
        for ref_lp in [-2.0, -2.01] {
            let v = check(ref_lp, 0.5);
            assert!(v.logprob_within_bound && v.passed, "{ref_lp}: {v:?}");
            assert!(close(v.max_abs_logprob_diff_tail, 0.5), "{v:?}");
            assert_eq!(v.max_abs_logprob_diff_likely, 0.0);
            let v = check(ref_lp, 0.6);
            assert!(!v.logprob_within_bound && !v.passed, "{ref_lp}: {v:?}");
        }
        // A candidate dropping below the floor keeps its reference (likely) tier.
        let v = check(-1.99, -0.5);
        assert!(!v.logprob_within_bound, "{v:?}");
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
