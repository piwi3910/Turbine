//! Expert parallelism on one node (P5 S-11): which rank of an EP group holds each routed expert
//! of each MoE layer.
//!
//! The placement is static for the process lifetime: contiguous by default (with 64 experts over
//! 2 ranks, experts 0–31 on rank 0 and 32–63 on rank 1) or read from a YAML file
//! `{layer: [rank per expert]}` covering every expert of every MoE layer. Every rank holds at
//! least one expert of every MoE layer, so every rank takes part in every layer's combine.

use std::collections::BTreeMap;
use std::path::Path;

use crate::plan::PlanError;

/// The configuration key of the EP size.
pub const EP_KEY: &str = "parallel.expert_parallel_size";
/// The configuration key of the placement.
pub const PLACEMENT_KEY: &str = "parallel.expert.placement";

/// Rank per expert, per MoE layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertPlacement {
    /// The EP group size.
    pub ranks: u32,
    /// `(layer index, rank of each expert)`, in layer order, one entry per MoE layer.
    pub layers: Vec<(u32, Vec<u32>)>,
}

fn err(key: &str, reason: impl Into<String>) -> PlanError {
    PlanError {
        key: key.to_string(),
        reason: reason.into(),
    }
}

fn check_size(num_experts: u32, ep: u32) -> Result<(), PlanError> {
    if ep == 0 || !num_experts.is_multiple_of(ep) {
        return Err(err(
            EP_KEY,
            format!("{ep} does not divide the {num_experts} routed experts"),
        ));
    }
    Ok(())
}

impl ExpertPlacement {
    /// Experts `r × n/ep .. (r + 1) × n/ep` on rank `r`, for every layer of `moe_layers`.
    pub fn contiguous(num_experts: u32, moe_layers: &[u32], ep: u32) -> Result<Self, PlanError> {
        check_size(num_experts, ep)?;
        let per = num_experts / ep;
        let ranks: Vec<u32> = (0..num_experts).map(|e| e / per).collect();
        Ok(ExpertPlacement {
            ranks: ep,
            layers: moe_layers.iter().map(|&l| (l, ranks.clone())).collect(),
        })
    }

    /// A placement file's text: YAML `{layer: [rank per expert]}`. Refused (naming
    /// `parallel.expert.placement`) when a MoE layer or an expert is missing, a layer is not a
    /// MoE layer, a rank is outside the group or holds no expert of a layer.
    pub fn parse(
        text: &str,
        num_experts: u32,
        moe_layers: &[u32],
        ep: u32,
    ) -> Result<Self, PlanError> {
        check_size(num_experts, ep)?;
        let map: BTreeMap<u32, Vec<u32>> = serde_norway::from_str(text)
            .map_err(|e| err(PLACEMENT_KEY, format!("not a {{layer: [rank]}} map: {e}")))?;
        if let Some(extra) = map.keys().find(|l| !moe_layers.contains(l)) {
            return Err(err(
                PLACEMENT_KEY,
                format!("layer {extra} is not a MoE layer of the model"),
            ));
        }
        let mut layers = Vec::with_capacity(moe_layers.len());
        for &layer in moe_layers {
            let ranks = map
                .get(&layer)
                .ok_or_else(|| err(PLACEMENT_KEY, format!("layer {layer} is missing")))?;
            if ranks.len() != num_experts as usize {
                let missing = ranks.len();
                return Err(err(
                    PLACEMENT_KEY,
                    format!(
                        "layer {layer} places {missing} experts; expert {missing} of \
                         {num_experts} is missing"
                    ),
                ));
            }
            if let Some((e, r)) = ranks.iter().enumerate().find(|&(_, &r)| r >= ep) {
                return Err(err(
                    PLACEMENT_KEY,
                    format!("layer {layer} expert {e} is on rank {r}, outside 0..{ep}"),
                ));
            }
            if let Some(idle) = (0..ep).find(|r| !ranks.contains(r)) {
                return Err(err(
                    PLACEMENT_KEY,
                    format!("rank {idle} holds no expert of layer {layer}"),
                ));
            }
            layers.push((layer, ranks.clone()));
        }
        Ok(ExpertPlacement { ranks: ep, layers })
    }

    /// [`ExpertPlacement::parse`] of the file at `path`.
    pub fn from_file(
        path: &Path,
        num_experts: u32,
        moe_layers: &[u32],
        ep: u32,
    ) -> Result<Self, PlanError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| err(PLACEMENT_KEY, format!("{}: {e}", path.display())))?;
        Self::parse(&text, num_experts, moe_layers, ep)
    }

    /// The ranks of `layer`'s experts; `None` for a layer without experts.
    pub fn layer(&self, layer: u32) -> Option<&[u32]> {
        self.layers
            .iter()
            .find(|(l, _)| *l == layer)
            .map(|(_, r)| r.as_slice())
    }

    /// The global ids of the experts `rank` holds in `layer`, ascending.
    pub fn local_experts(&self, layer: u32, rank: u32) -> Vec<u32> {
        self.layer(layer)
            .map(|ranks| {
                (0..ranks.len() as u32)
                    .filter(|&e| ranks[e as usize] == rank)
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// OLMoE's 16 MoE layers.
    fn olmoe_layers() -> Vec<u32> {
        (0..16).collect()
    }

    fn yaml(layers: &[u32], ranks: impl Fn(u32, u32) -> u32, experts: u32) -> String {
        layers
            .iter()
            .map(|&l| {
                let r: Vec<String> = (0..experts).map(|e| ranks(l, e).to_string()).collect();
                format!("{l}: [{}]\n", r.join(", "))
            })
            .collect()
    }

    /// Contiguous placement of OLMoE's 64 experts over 2 ranks gives 32 each; a file missing
    /// expert 17 of layer 3, one naming rank 2 at ep 2 and one leaving a rank idle are refused
    /// naming `parallel.expert.placement`; ep 3 with 64 experts is refused naming
    /// `parallel.expert_parallel_size`. Breaks if an invalid map loads.
    #[test]
    fn placement_map_valid() {
        let layers = olmoe_layers();
        let p = ExpertPlacement::contiguous(64, &layers, 2).unwrap();
        for &l in &layers {
            assert_eq!(p.local_experts(l, 0), (0..32).collect::<Vec<_>>());
            assert_eq!(p.local_experts(l, 1), (32..64).collect::<Vec<_>>());
        }
        assert!(p.local_experts(99, 0).is_empty());

        // A valid file: interleaved experts.
        let text = yaml(&layers, |_, e| e % 2, 64);
        let f = ExpertPlacement::parse(&text, 64, &layers, 2).unwrap();
        assert_eq!(f.local_experts(5, 1)[..3], [1, 3, 5]);

        // Expert 17 of layer 3 missing (layer 3 lists only 17 experts).
        let mut short = yaml(&layers, |_, e| e / 32, 64);
        short = short
            .lines()
            .map(|line| {
                if line.starts_with("3:") {
                    let r: Vec<String> = (0..17).map(|e| (e % 2).to_string()).collect();
                    format!("3: [{}]\n", r.join(", "))
                } else {
                    format!("{line}\n")
                }
            })
            .collect();
        let e = ExpertPlacement::parse(&short, 64, &layers, 2).unwrap_err();
        assert_eq!(e.key, PLACEMENT_KEY);
        assert!(e.reason.contains("expert 17"), "{e}");

        let rank2 = yaml(
            &layers,
            |l, e| if l == 4 && e == 9 { 2 } else { e / 32 },
            64,
        );
        let e = ExpertPlacement::parse(&rank2, 64, &layers, 2).unwrap_err();
        assert_eq!(e.key, PLACEMENT_KEY);
        assert!(e.reason.contains("rank 2"), "{e}");

        let idle = yaml(&layers, |_, _| 0, 64);
        let e = ExpertPlacement::parse(&idle, 64, &layers, 2).unwrap_err();
        assert!(e.reason.contains("rank 1 holds no expert"), "{e}");

        let missing_layer = yaml(&layers[1..], |_, e| e / 32, 64);
        let e = ExpertPlacement::parse(&missing_layer, 64, &layers, 2).unwrap_err();
        assert!(e.reason.contains("layer 0 is missing"), "{e}");

        let e = ExpertPlacement::contiguous(64, &layers, 3).unwrap_err();
        assert_eq!(e.key, EP_KEY);
        let e = ExpertPlacement::parse(&text, 64, &layers, 3).unwrap_err();
        assert_eq!(e.key, EP_KEY);
    }
}
