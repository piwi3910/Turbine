//! Phase 8 S-3: the speculative verifier. Pure acceptance maths over the target's per-row
//! post-processing distributions; the engine supplies the rows (from the sampler) and the
//! draw (from the choice's sampler RNG), and this module decides how many proposals the target
//! accepted and which token it emits next.
//!
//! Greedy targets accept the longest exact prefix against the target's argmax. Sampled targets
//! use the standard speculative rejection scheme with a point-mass proposal: accept the
//! proposal `t` with probability `p(t)`, and on rejection draw from `p` restricted to the ids
//! other than `t` (the engine renormalises through [`Draw::draw`]). The emitted stream keeps
//! the target's distribution.
use thiserror::Error;

/// The target's rows: one per verified position, each already post-processed by the sampler.
pub trait Rows {
    /// Rows available (the proposals plus the row that supplies the next token).
    fn len(&self) -> usize;

    /// No rows (a verifier call never succeeds with none; the pair keeps `len` idiomatic).
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The greedy token of `row` (its argmax after post-processing).
    fn greedy(&self, row: usize) -> u32;

    /// The post-processing probabilities of `row` (the greedy path never asks).
    fn probabilities(&self, row: usize) -> &[f32];
}

/// The verifier's draws: a uniform for the acceptance test and a weighted draw excluding a
/// token for the rejection fallback. The engine implements both over the choice's sampler RNG.
pub trait Draw {
    fn uniform(&mut self) -> f64;

    /// One draw over `weights`, with `exclude`'s mass removed and the rest renormalised.
    fn draw(&mut self, weights: &[f32], exclude: Option<u32>) -> u32;
}

/// How many proposals were accepted and the target's next token after them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Verdict {
    pub accepted: u32,
    pub next: u32,
}

/// A malformed verification input (a row is missing, or its distribution does not cover the
/// proposed id).
#[derive(Debug, Error, PartialEq, Eq)]
pub enum VerifyError {
    #[error("the target rows ({len}) do not cover the proposals ({proposals})")]
    MissingRow { len: usize, proposals: usize },
    #[error("the target row does not carry a probability for token {0}")]
    MissingProbability(u32),
}

/// Greedy: accept the longest exact prefix; at the first disagreement emit the target's token.
pub fn verify_greedy(rows: &impl Rows, proposals: &[u32]) -> Result<Verdict, VerifyError> {
    if proposals.len() >= rows.len() {
        return Err(VerifyError::MissingRow {
            len: rows.len(),
            proposals: proposals.len(),
        });
    }
    for (row, &proposal) in proposals.iter().enumerate() {
        let target = rows.greedy(row);
        if proposal != target {
            return Ok(Verdict {
                accepted: row as u32,
                next: target,
            });
        }
    }
    Ok(Verdict {
        accepted: proposals.len() as u32,
        next: rows.greedy(proposals.len()),
    })
}

/// Sampling: accept `t` with probability `p(t)`; on rejection draw from `p` without `t`.
pub fn verify_sampling(
    rows: &impl Rows,
    proposals: &[u32],
    draw: &mut impl Draw,
) -> Result<Verdict, VerifyError> {
    if proposals.len() >= rows.len() {
        return Err(VerifyError::MissingRow {
            len: rows.len(),
            proposals: proposals.len(),
        });
    }
    for (row, &proposal) in proposals.iter().enumerate() {
        let probabilities = rows.probabilities(row);
        let accept = probabilities
            .get(proposal as usize)
            .copied()
            .ok_or(VerifyError::MissingProbability(proposal))?;
        if draw.uniform() < f64::from(accept) {
            continue;
        }
        let next = draw.draw(probabilities, Some(proposal));
        return Ok(Verdict {
            accepted: row as u32,
            next,
        });
    }
    Ok(Verdict {
        accepted: proposals.len() as u32,
        next: draw.draw(rows.probabilities(proposals.len()), None),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedRows {
        greedy: Vec<u32>,
        probabilities: Vec<Vec<f32>>,
    }

    impl Rows for FixedRows {
        fn len(&self) -> usize {
            self.greedy.len()
        }

        fn greedy(&self, row: usize) -> u32 {
            self.greedy[row]
        }

        fn probabilities(&self, row: usize) -> &[f32] {
            &self.probabilities[row]
        }
    }

    /// A 64-bit LCG; deterministic and independent of any crate feature.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    impl Draw for Lcg {
        fn uniform(&mut self) -> f64 {
            self.next()
        }

        fn draw(&mut self, weights: &[f32], exclude: Option<u32>) -> u32 {
            let total: f64 = weights
                .iter()
                .enumerate()
                .filter(|(i, _)| Some(*i as u32) != exclude)
                .map(|(_, w)| f64::from(*w))
                .sum();
            let u = self.next() * total;
            let mut cum = 0.0;
            let mut last = 0;
            for (i, &w) in weights.iter().enumerate() {
                if Some(i as u32) == exclude || w <= 0.0 {
                    continue;
                }
                cum += f64::from(w);
                last = i as u32;
                if u < cum {
                    return i as u32;
                }
            }
            last
        }
    }

    #[test]
    fn greedy_prefix() {
        let rows = FixedRows {
            greedy: vec![7, 8, 9, 10],
            probabilities: vec![Vec::new(); 4],
        };
        assert_eq!(
            verify_greedy(&rows, &[7, 8, 9]).unwrap(),
            Verdict {
                accepted: 3,
                next: 10
            }
        );
        assert_eq!(
            verify_greedy(&rows, &[7, 1, 9]).unwrap(),
            Verdict {
                accepted: 1,
                next: 8
            }
        );
        assert_eq!(
            verify_greedy(&rows, &[1]).unwrap(),
            Verdict {
                accepted: 0,
                next: 7
            }
        );
        assert_eq!(
            verify_greedy(&rows, &[]).unwrap(),
            Verdict {
                accepted: 0,
                next: 7
            }
        );
        assert!(matches!(
            verify_greedy(&rows, &[7; 4]),
            Err(VerifyError::MissingRow { .. })
        ));
    }

    /// The accepted proposal and the rejection fallback reproduce the target's distribution:
    /// with a point-mass draft at token 0, the first emitted token is 0 with `p(0)` and
    /// otherwise drawn from `p` without 0.
    #[test]
    fn sampling_is_exact() {
        let probabilities = [0.6, 0.3, 0.1];
        let mut counts = [0u64; 3];
        let rounds = 1_000_000u64;
        for i in 0..rounds {
            let rows = FixedRows {
                greedy: vec![0; 2],
                probabilities: vec![probabilities.to_vec(), probabilities.to_vec()],
            };
            let mut draw = Lcg(i.wrapping_mul(2_654_435_761).wrapping_add(9));
            let verdict = verify_sampling(&rows, &[0], &mut draw).unwrap();
            let first = if verdict.accepted == 1 {
                0
            } else {
                verdict.next
            };
            counts[first as usize] += 1;
        }
        for (got, want) in counts.iter().zip(probabilities) {
            let got = *got as f64 / rounds as f64;
            assert!(
                (got - f64::from(want)).abs() < 0.01,
                "{counts:?} vs {probabilities:?}"
            );
        }
    }

    #[test]
    fn zero_probability_is_never_accepted() {
        let rows = FixedRows {
            greedy: vec![0; 2],
            probabilities: vec![vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]],
        };
        let mut draw = Lcg(7);
        assert_eq!(
            verify_sampling(&rows, &[1], &mut draw).unwrap(),
            Verdict {
                accepted: 0,
                next: 0
            }
        );
    }

    #[test]
    fn all_accepted_draws_the_next_token() {
        let rows = FixedRows {
            greedy: vec![0; 3],
            probabilities: vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        };
        let mut draw = Lcg(3);
        assert_eq!(
            verify_sampling(&rows, &[0, 1], &mut draw).unwrap(),
            Verdict {
                accepted: 2,
                next: 2
            }
        );
    }
}
