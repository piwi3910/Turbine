//! `turbine-golden positions`: one reference prompt judged position by position — which
//! reference top-k candidate is how far from the reference at every position, and which bound
//! it meets (P5, decision "P5: OLMoE golden tolerance under expert parallelism", step A).
//!
//! Two sources for the candidate rows: a live endpoint **teacher-forced** on the reference's own
//! tokens (position `p` is one greedy step on the prompt ids plus the reference's first `p`
//! tokens, so every position has the reference's history whatever the engine would have
//! generated), or a captured free-running record (only its positions before its first
//! divergence share the reference's history).

use serde::Serialize;

use super::client::{Endpoint, Generation, parse_completion};
use super::fixture::{GoldenError, ReferenceRecord, Tolerance};

/// One reference top-k candidate at one position.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CandidateRow {
    /// 1-based rank in the reference's top-k.
    pub rank: usize,
    pub token_id: u32,
    pub reference_logprob: f32,
    /// `None`: missing from the candidate's top list (a violation by itself).
    pub candidate_logprob: Option<f32>,
    pub abs_diff: Option<f32>,
    /// `likely` (reference logprob above the floor) or `tail`.
    pub tier: &'static str,
    pub bound: f32,
    pub exceeds: bool,
}

/// One position of the prompt.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PositionRow {
    pub position: usize,
    pub reference_token: u32,
    pub candidate_token: Option<u32>,
    /// Reference top-1 minus top-2 logprob (a near-tie below the tolerance's `margin_nats`).
    pub reference_margin: Option<f32>,
    pub candidates: Vec<CandidateRow>,
}

impl PositionRow {
    /// The largest |Δ| of one tier at this position.
    pub fn max_diff(&self, tier: &str) -> f32 {
        self.candidates
            .iter()
            .filter(|c| c.tier == tier)
            .filter_map(|c| c.abs_diff)
            .fold(0.0, f32::max)
    }
}

fn sorted_top(row: &[(u32, f32)], k: usize) -> Vec<(u32, f32)> {
    let mut sorted = row.to_vec();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    sorted.truncate(k);
    sorted
}

/// Judges `rows[p]` (the candidate's top list at position `p`, its greedy token
/// `tokens[p]`) against the reference's position `p`, for every `p` in `positions`, with the
/// strict bounds (concurrency 1) of `tol`.
pub fn judge_positions(
    reference: &ReferenceRecord,
    tokens: &[Option<u32>],
    rows: &[Vec<(u32, f32)>],
    positions: std::ops::Range<usize>,
    tol: &Tolerance,
) -> Vec<PositionRow> {
    let bounds = tol.logprob_bounds(1);
    positions
        .filter_map(|p| {
            let ref_row = reference.top_logprobs.get(p)?;
            let got = rows.get(p).map(Vec::as_slice).unwrap_or(&[]);
            let top = sorted_top(ref_row, tol.top_k);
            let margin = (top.len() >= 2).then(|| top[0].1 - top[1].1);
            let candidates = top
                .iter()
                .enumerate()
                .map(|(i, &(id, lp))| {
                    let likely = lp > tol.likely_logprob_floor;
                    let bound = if likely {
                        bounds.max_abs_logprob_diff_likely
                    } else {
                        bounds.max_abs_logprob_diff_tail
                    };
                    let got_lp = got.iter().find(|e| e.0 == id).map(|e| e.1);
                    let abs_diff = got_lp.map(|g| (g - lp).abs());
                    CandidateRow {
                        rank: i + 1,
                        token_id: id,
                        reference_logprob: lp,
                        candidate_logprob: got_lp,
                        abs_diff,
                        tier: if likely { "likely" } else { "tail" },
                        bound,
                        exceeds: abs_diff.is_none_or(|d| d > bound),
                    }
                })
                .collect();
            Some(PositionRow {
                position: p,
                reference_token: *reference.tokens.get(p)?,
                candidate_token: tokens.get(p).copied().flatten(),
                reference_margin: margin,
                candidates,
            })
        })
        .collect()
}

/// A captured free-running record against the reference, over the positions before its first
/// divergence (after it the histories differ).
pub fn positions_of_capture(
    reference: &ReferenceRecord,
    capture: &ReferenceRecord,
    tol: &Tolerance,
) -> Vec<PositionRow> {
    let shared = reference
        .tokens
        .iter()
        .zip(&capture.tokens)
        .take_while(|(r, c)| r == c)
        .count();
    // The divergence position itself still has the reference's history.
    let end = (shared + 1).min(reference.tokens.len());
    let tokens: Vec<Option<u32>> = capture.tokens.iter().map(|&t| Some(t)).collect();
    judge_positions(reference, &tokens, &capture.top_logprobs, 0..end, tol)
}

/// The endpoint teacher-forced on the reference: position `p` is one greedy completion step on
/// the reference's `prompt_token_ids` followed by its first `p` tokens (the completions route,
/// token-id prompt, as-is — no template, no BOS added).
pub async fn positions_teacher_forced(
    endpoint: &Endpoint,
    model: &str,
    reference: &ReferenceRecord,
    tol: &Tolerance,
) -> Result<Vec<PositionRow>, GoldenError> {
    if reference.prompt_token_ids.is_empty() {
        return Err(GoldenError::Usage(format!(
            "{}: the record has no prompt_token_ids to teacher-force on (a turbine-golden \
             capture of an engine that does not return them); use a transformers reference, or \
             --candidate with the capture",
            reference.id
        )));
    }
    let top = tol.top_k.max(5) as u32;
    let n = reference.tokens.len();
    let (mut tokens, mut rows) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for p in 0..n {
        let mut ids = reference.prompt_token_ids.clone();
        ids.extend_from_slice(&reference.tokens[..p]);
        let g: Generation = endpoint.step(model, &ids, top).await?;
        tokens.push(g.tokens.first().copied());
        rows.push(g.top_logprobs.into_iter().next().unwrap_or_default());
    }
    Ok(judge_positions(reference, &tokens, &rows, 0..n, tol))
}

impl Endpoint {
    /// One greedy step on a token-id prompt through `/v1/completions`, with `top` logprobs.
    pub async fn step(
        &self,
        model: &str,
        ids: &[u32],
        top: u32,
    ) -> Result<Generation, GoldenError> {
        let body = serde_json::json!({
            "model": model,
            "prompt": ids,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": top,
            "return_tokens_as_token_ids": true,
            "ignore_eos": true,
            "stream": false,
        });
        let v = self.post_json("/v1/completions", &body).await?;
        parse_completion(&v, top as usize)
            .map_err(|e| GoldenError::Endpoint(format!("POST /v1/completions (forced step): {e}")))
    }
}

/// Human-readable rows: one line per position, candidates over their bound marked `EXCEEDS`,
/// then the worst |Δ| per tier and where.
pub fn to_text(prompt_id: &str, source: &str, rows: &[PositionRow]) -> String {
    let mut out = format!("positions {prompt_id} ({source})\n");
    let mut worst = [("likely", 0.0f32, None), ("tail", 0.0f32, None)];
    for r in rows {
        let flag = if r.candidates.iter().any(|c| c.exceeds) {
            " EXCEEDS"
        } else {
            ""
        };
        out.push_str(&format!(
            "pos {:>3} ref={} got={} margin={}{flag}\n",
            r.position,
            r.reference_token,
            r.candidate_token
                .map_or_else(|| "-".to_string(), |t| t.to_string()),
            r.reference_margin
                .map_or_else(|| "-".to_string(), |m| format!("{m:.3}")),
        ));
        for c in &r.candidates {
            out.push_str(&format!(
                "    #{} id={} ref={:.4} got={} |d|={} {}<={}{}\n",
                c.rank,
                c.token_id,
                c.reference_logprob,
                c.candidate_logprob
                    .map_or_else(|| "missing".to_string(), |g| format!("{g:.4}")),
                c.abs_diff
                    .map_or_else(|| "-".to_string(), |d| format!("{d:.4}")),
                c.tier,
                c.bound,
                if c.exceeds { " EXCEEDS" } else { "" }
            ));
        }
        for w in &mut worst {
            let d = r.max_diff(w.0);
            if d > w.1 {
                *w = (w.0, d, Some(r.position));
            }
        }
    }
    for (tier, d, pos) in worst {
        out.push_str(&format!(
            "worst {tier} |d|={d:.4} at {}\n",
            pos.map_or_else(|| "-".to_string(), |p| p.to_string())
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tol() -> Tolerance {
        serde_json::from_value(serde_json::json!({
            "min_identical_prefix": 32, "min_prompts_passing": 14, "top_k": 3,
            "max_abs_logprob_diff_likely": 1.0, "max_abs_logprob_diff_tail": 1.5,
            "likely_logprob_floor": -2.0, "margin_nats": 0.5
        }))
        .unwrap()
    }

    fn reference() -> ReferenceRecord {
        serde_json::from_value(serde_json::json!({
            "id": "p", "engine": "hf", "model": "m", "captured": "now",
            "prompt_token_ids": [1, 2], "tokens": [10, 11, 12],
            "top_logprobs": [
                [[10, -0.1], [20, -2.5], [30, -3.0]],
                [[11, -0.2], [21, -1.9], [31, -4.0]],
                [[12, -0.3], [22, -0.4], [32, -5.0]]
            ]
        }))
        .unwrap()
    }

    /// Each reference top-k candidate is tiered by its reference logprob and judged by its
    /// tier's strict bound; a missing candidate exceeds; the capture is judged only up to and
    /// including its first divergence. Breaks if the tiering, the bound or the range is wrong.
    #[test]
    fn rows_tier_and_bound_every_candidate() {
        let (r, t) = (reference(), tol());
        let rows = judge_positions(
            &r,
            &[Some(10), Some(11), Some(22)],
            &[
                vec![(10, -0.2), (20, -2.0), (30, -3.0)],
                vec![(11, -0.2), (21, -3.0)],
                vec![(22, -0.2), (12, -0.5), (32, -5.1)],
            ],
            0..3,
            &t,
        );
        assert_eq!(rows.len(), 3);
        let c = &rows[0].candidates;
        assert_eq!((c[0].tier, c[1].tier), ("likely", "tail"));
        assert!(!c[1].exceeds, "tail |0.5| <= 1.5");
        let c = &rows[1].candidates;
        assert_eq!(c[1].tier, "likely");
        assert!(c[1].exceeds, "likely |1.1| > 1.0");
        assert!(c[2].exceeds && c[2].candidate_logprob.is_none(), "missing");
        assert!((rows[2].reference_margin.unwrap() - 0.1).abs() < 1e-6);
        assert!((rows[1].max_diff("likely") - 1.1).abs() < 1e-6);

        let capture = ReferenceRecord {
            tokens: vec![10, 99, 12],
            ..reference()
        };
        let rows = positions_of_capture(&r, &capture, &t);
        assert_eq!(rows.len(), 2, "positions 0 and the divergence at 1");
        let text = to_text("p", "capture", &rows);
        assert!(text.contains("worst likely"), "{text}");
    }
}
