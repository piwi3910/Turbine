//! Pipeline parallelism on one node (P5 S-10): splitting a model's decoder layers into contiguous
//! stages and placing the stages on devices.
//!
//! [`partition`] balances the stages by estimated cost — each layer's weight bytes (the decode
//! step reads them all) and its per-token FLOPs (prefill), each normalised by the model total and
//! summed, so neither unit dominates; a MoE layer's FLOPs count its active experts only, its
//! bytes every expert. The embedding lives on the first stage and the final norm and LM head on
//! the last, whose cost is added to that stage (a tied embedding is loaded on both). An explicit
//! `parallel.pipeline.layer_split` replaces the balance after [`validate_split`].
//!
//! [`place_stages`] puts the stages on the group's devices. Every GPU↔GPU byte on a board without
//! peer access crosses host memory, and the last stage moves the most host-side bytes (the logits
//! or their device reduction every step, plus the KV tier copies of its layers), so the stages
//! are ordered by host traffic and the busiest goes on the device with the fastest measured host
//! link; without measurements the stages keep device order.

use std::fmt;
use std::ops::Range;

use turbine_core::types::DeviceId;

use crate::plan::PlanError;

/// The configuration key of an explicit split.
pub const SPLIT_KEY: &str = "parallel.pipeline.layer_split";
/// The configuration key of the stage count.
pub const PP_KEY: &str = "parallel.pipeline_parallel_size";

/// One decoder layer's (or the head's) estimated cost.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LayerCost {
    /// Bytes of weights resident for the layer (every expert of a MoE layer).
    pub weight_bytes: u64,
    /// Multiply-accumulate FLOPs per token (the active experts of a MoE layer).
    pub flops_per_token: u64,
}

/// A model's costs as the partitioner sees them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PipelineCosts {
    /// One entry per decoder layer, in order.
    pub layers: Vec<LayerCost>,
    /// Added to the first stage (the embedding lookup).
    pub first: LayerCost,
    /// Added to the last stage (the final norm and the LM head).
    pub last: LayerCost,
}

impl PipelineCosts {
    /// The costs of a model as its configuration describes it (P5 S-10): per decoder layer the
    /// attention projections (Q, K, V, O) plus the MLP — three `hidden × intermediate`
    /// projections, or on a mixture-of-experts model the router and every expert's three for
    /// the weight bytes but only the `experts_per_token` active experts' for the FLOPs — and on
    /// the last stage the final norm and the LM head (`vocab × hidden`). The embedding lookup
    /// reads one row per token: no cost on the first stage. Bytes count two per parameter
    /// (BF16); a constant factor cancels in the normalisation.
    pub fn from_shape(m: &turbine_core::types::ModelShape) -> PipelineCosts {
        let h = u64::from(m.hidden);
        let q = u64::from(m.num_attention_heads) * u64::from(m.head_dim);
        let kv = u64::from(m.num_kv_heads) * u64::from(m.head_dim);
        let i = u64::from(m.intermediate);
        let attention = 2 * h * q + 2 * h * kv;
        let (resident, active) = if m.num_experts > 0 {
            let e = u64::from(m.num_experts);
            let top = u64::from(m.experts_per_token.max(1));
            let expert = 3 * h * i;
            (
                attention + h * e + e * expert,
                attention + h * e + top * expert,
            )
        } else {
            (attention + 3 * h * i, attention + 3 * h * i)
        };
        let layer = LayerCost {
            weight_bytes: 2 * resident,
            flops_per_token: 2 * active,
        };
        let head = u64::from(m.vocab) * h;
        PipelineCosts {
            layers: vec![layer; m.num_layers as usize],
            first: LayerCost::default(),
            last: LayerCost {
                weight_bytes: 2 * head,
                flops_per_token: 2 * head,
            },
        }
    }

    /// Normalised costs: per layer, then the first- and last-stage extras.
    fn normalised(&self) -> (Vec<f64>, f64, f64) {
        let all = self.layers.iter().chain([&self.first, &self.last]).copied();
        let (w, f) = all.fold((0u64, 0u64), |(w, f), c| {
            (w + c.weight_bytes, f + c.flops_per_token)
        });
        let norm = |c: &LayerCost| {
            let part = |x: u64, total: u64| {
                if total == 0 {
                    0.0
                } else {
                    x as f64 / total as f64
                }
            };
            part(c.weight_bytes, w) + part(c.flops_per_token, f)
        };
        (
            self.layers.iter().map(norm).collect(),
            norm(&self.first),
            norm(&self.last),
        )
    }

    /// Each stage's normalised cost for the given ranges.
    pub fn stage_costs(&self, ranges: &[Range<u32>]) -> Vec<f64> {
        let (layers, first, last) = self.normalised();
        let n = ranges.len();
        ranges
            .iter()
            .enumerate()
            .map(|(s, r)| {
                let mut c: f64 = layers[r.start as usize..r.end as usize].iter().sum();
                if s == 0 {
                    c += first;
                }
                if s + 1 == n {
                    c += last;
                }
                c
            })
            .collect()
    }
}

fn err(key: &str, reason: impl Into<String>) -> PlanError {
    PlanError {
        key: key.to_string(),
        reason: reason.into(),
    }
}

/// Checks an explicit split against the layer and stage counts.
pub fn validate_split(split: &[u32], layers: u32, stages: u32) -> Result<(), PlanError> {
    if split.len() != stages as usize {
        return Err(err(
            SPLIT_KEY,
            format!(
                "{} entries for {stages} pipeline stages (one per stage)",
                split.len()
            ),
        ));
    }
    if let Some(s) = split.iter().position(|&n| n == 0) {
        return Err(err(SPLIT_KEY, format!("stage {s} has no layers")));
    }
    let sum: u64 = split.iter().map(|&n| u64::from(n)).sum();
    if sum != u64::from(layers) {
        return Err(err(
            SPLIT_KEY,
            format!("the split covers {sum} layers; the model has {layers}"),
        ));
    }
    Ok(())
}

/// Contiguous layer ranges, one per stage, in stage order: `split` when given (validated), else
/// the ranges minimising the costliest stage (ties: the earliest cut). Every stage gets at least
/// one layer.
pub fn partition(
    costs: &PipelineCosts,
    stages: u32,
    split: Option<&[u32]>,
) -> Result<Vec<Range<u32>>, PlanError> {
    let layers = costs.layers.len() as u32;
    if stages == 0 || stages > layers {
        return Err(err(
            PP_KEY,
            format!("{stages} stages for a model of {layers} layers (1..={layers})"),
        ));
    }
    if let Some(split) = split {
        validate_split(split, layers, stages)?;
        let mut start = 0;
        return Ok(split
            .iter()
            .map(|&n| {
                let r = start..start + n;
                start += n;
                r
            })
            .collect());
    }
    let (c, first, last) = costs.normalised();
    let n = c.len();
    let k = stages as usize;
    // prefix[i] = cost of layers 0..i
    let mut prefix = vec![0.0f64; n + 1];
    for i in 0..n {
        prefix[i + 1] = prefix[i] + c[i];
    }
    let seg = |a: usize, b: usize, s: usize| {
        let mut x = prefix[b] - prefix[a];
        if s == 0 {
            x += first;
        }
        if s + 1 == k {
            x += last;
        }
        x
    };
    // best[s][i]: the smallest possible max stage cost placing layers 0..i into stages 0..=s,
    // stage s ending at i; cut[s][i]: where stage s starts.
    let mut best = vec![vec![f64::INFINITY; n + 1]; k];
    let mut cut = vec![vec![0usize; n + 1]; k];
    for (i, b) in best[0].iter_mut().enumerate().skip(1) {
        *b = seg(0, i, 0);
    }
    for s in 1..k {
        for i in (s + 1)..=n {
            for j in s..i {
                let v = best[s - 1][j].max(seg(j, i, s));
                if v < best[s][i] - 1e-12 {
                    best[s][i] = v;
                    cut[s][i] = j;
                }
            }
        }
    }
    let mut ends = vec![n; k];
    for s in (1..k).rev() {
        ends[s - 1] = cut[s][ends[s]];
    }
    let mut start = 0usize;
    Ok(ends
        .into_iter()
        .map(|e| {
            let r = start as u32..e as u32;
            start = e;
            r
        })
        .collect())
}

/// One pipeline stage as planned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageSpec {
    pub stage: u32,
    pub device: DeviceId,
    pub layers: Range<u32>,
    /// Holds the embedding (the first stage).
    pub embedding: bool,
    /// Holds the final norm and the LM head (the last stage).
    pub lm_head: bool,
}

/// Why the stages sit where they are (logged and echoed in the plan).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StageReason {
    /// The busiest stage (by host traffic) is on this device, the fastest measured host link.
    HostTraffic(DeviceId),
    /// No host link was measured: stages follow device order.
    NominalLinks,
}

impl fmt::Display for StageReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StageReason::HostTraffic(d) => write!(f, "pp_stage_host_traffic:{}", d.0),
            StageReason::NominalLinks => f.write_str("pp_stage_device_order"),
        }
    }
}

/// Places `ranges` (stage order) on `devices` (one per stage). `host_link_gbps` gives a device's
/// measured host-link bandwidth (the slower of its two directions), `None` when unmeasured.
/// Host traffic ranks the last stage first, then the other stages by layer count (their KV tier
/// copies), later stages first on ties; devices rank by bandwidth, then by index.
pub fn place_stages(
    ranges: &[Range<u32>],
    devices: &[DeviceId],
    host_link_gbps: &dyn Fn(DeviceId) -> Option<f64>,
) -> Result<(Vec<StageSpec>, StageReason), PlanError> {
    if ranges.len() != devices.len() {
        return Err(err(
            PP_KEY,
            format!(
                "{} stages need {} devices, the plan has {}",
                ranges.len(),
                ranges.len(),
                devices.len()
            ),
        ));
    }
    let n = ranges.len();
    let spec = |stage: usize, device: DeviceId| StageSpec {
        stage: stage as u32,
        device,
        layers: ranges[stage].clone(),
        embedding: stage == 0,
        lm_head: stage + 1 == n,
    };
    let measured: Vec<Option<f64>> = devices.iter().map(|&d| host_link_gbps(d)).collect();
    if measured.iter().all(Option::is_none) {
        let stages = devices
            .iter()
            .enumerate()
            .map(|(s, &d)| spec(s, d))
            .collect();
        return Ok((stages, StageReason::NominalLinks));
    }
    let mut by_link: Vec<usize> = (0..n).collect();
    by_link.sort_by(|&a, &b| {
        let (x, y) = (measured[a].unwrap_or(0.0), measured[b].unwrap_or(0.0));
        y.total_cmp(&x).then(devices[a].0.cmp(&devices[b].0))
    });
    let mut by_traffic: Vec<usize> = (0..n).collect();
    by_traffic.sort_by(|&a, &b| {
        let key = |s: usize| (s + 1 == n, ranges[s].len(), s);
        key(b).cmp(&key(a))
    });
    let mut placed = vec![DeviceId(0); n];
    for (&stage, &dev) in by_traffic.iter().zip(&by_link) {
        placed[stage] = devices[dev];
    }
    let stages = placed
        .iter()
        .enumerate()
        .map(|(s, &d)| spec(s, d))
        .collect();
    Ok((stages, StageReason::HostTraffic(devices[by_link[0]])))
}

#[cfg(test)]
mod tests {
    use turbine_core::types::ModelShape;

    use super::*;

    /// Llama-3.2-3B-Instruct: 28 layers, hidden 3072, intermediate 8192, 24 heads / 8 KV heads of
    /// 128, vocabulary 128,256, tied embeddings, BF16.
    fn llama() -> PipelineCosts {
        let (h, i, q, kv, v) = (3072u64, 8192u64, 24 * 128u64, 8 * 128u64, 128_256u64);
        let params = h * q + 2 * h * kv + q * h + 3 * h * i;
        let layer = LayerCost {
            weight_bytes: 2 * params,
            flops_per_token: 2 * params,
        };
        PipelineCosts {
            layers: vec![layer; 28],
            first: LayerCost::default(),
            last: LayerCost {
                weight_bytes: 2 * v * h,
                flops_per_token: 2 * v * h,
            },
        }
    }

    /// OLMoE-1B-7B: 16 MoE layers, hidden 2048, 16 = 16 heads of 128, 64 experts of intermediate
    /// 1024 with top-8 routing, vocabulary 50,304, untied embeddings, BF16.
    fn olmoe() -> PipelineCosts {
        let (h, e, top, ei, v) = (2048u64, 64u64, 8u64, 1024u64, 50_304u64);
        let attn = 4 * h * h;
        let expert = 3 * h * ei;
        let layer = LayerCost {
            weight_bytes: 2 * (attn + h * e + e * expert),
            flops_per_token: 2 * (attn + h * e + top * expert),
        };
        PipelineCosts {
            layers: vec![layer; 16],
            first: LayerCost::default(),
            last: LayerCost {
                weight_bytes: 2 * v * h,
                flops_per_token: 2 * v * h,
            },
        }
    }

    fn assert_cover(ranges: &[Range<u32>], layers: u32, what: &str) {
        let mut next = 0;
        for (s, r) in ranges.iter().enumerate() {
            assert_eq!(
                r.start, next,
                "{what} stage {s} starts where the last ended"
            );
            assert!(r.end > r.start, "{what} stage {s} has a layer");
            next = r.end;
        }
        assert_eq!(next, layers, "{what} covers every layer");
    }

    /// Contiguous ranges covering every layer once, with max/min stage cost ≤ 1.25 for Llama at
    /// 2, 4 and 7 stages and OLMoE at 2 and 4 (the LM head costed on the last stage). Breaks if a
    /// layer is dropped, duplicated or out of order, or the balance ignores the head.
    #[test]
    fn partition_balances_cost() {
        for (name, costs, stages) in [
            ("llama", llama(), &[2u32, 4, 7][..]),
            ("olmoe", olmoe(), &[2, 4][..]),
        ] {
            let layers = costs.layers.len() as u32;
            for &k in stages {
                let what = format!("{name} pp {k}");
                let ranges = partition(&costs, k, None).expect(&what);
                assert_eq!(ranges.len(), k as usize, "{what}");
                assert_cover(&ranges, layers, &what);
                let c = costs.stage_costs(&ranges);
                let (max, min) = c
                    .iter()
                    .fold((0f64, f64::INFINITY), |(a, b), &x| (a.max(x), b.min(x)));
                println!(
                    "partition_balances_cost {what}: {ranges:?} max/min {:.3}",
                    max / min
                );
                assert!(max / min <= 1.25, "{what}: {ranges:?} costs {c:?}");
            }
        }
        // The head weighs on the last stage: two Llama stages give it fewer layers.
        let two = partition(&llama(), 2, None).unwrap();
        assert!(two[1].len() < two[0].len(), "{two:?}");
        // One stage holds everything; more stages than layers is refused.
        assert_eq!(partition(&llama(), 1, None).unwrap(), vec![0..28]);
        let e = partition(&llama(), 29, None).unwrap_err();
        assert_eq!(e.key, PP_KEY);
    }

    /// `[14, 14]` is taken as given for 28 layers; `[14, 13]`, `[28, 0]` and a 3-entry list with
    /// two stages are refused naming `parallel.pipeline.layer_split`.
    #[test]
    fn explicit_split_validated() {
        let costs = llama();
        assert_eq!(
            partition(&costs, 2, Some(&[14, 14])).unwrap(),
            vec![0..14, 14..28]
        );
        assert_eq!(
            partition(&costs, 2, Some(&[20, 8])).unwrap(),
            vec![0..20, 20..28]
        );
        for bad in [&[14u32, 13][..], &[28, 0], &[10, 10, 8]] {
            let e = partition(&costs, 2, Some(bad)).unwrap_err();
            assert_eq!(e.key, SPLIT_KEY, "{bad:?}: {e}");
        }
    }

    /// novanas: GPU0 (Gen5 x8) measures faster than GPU1 (Gen4 x8), so the last stage goes on
    /// GPU0 with `pp_stage_host_traffic:0`; swapping the measurements swaps the placement;
    /// without measurements the stages keep device order.
    #[test]
    fn last_stage_on_fastest_link() {
        let ranges = vec![0..14, 14..28];
        let devices = [DeviceId(0), DeviceId(1)];
        let fast0 = |d: DeviceId| Some(if d.0 == 0 { 24.0 } else { 12.0 });
        let (stages, reason) = place_stages(&ranges, &devices, &fast0).unwrap();
        assert_eq!(stages[1].device, DeviceId(0), "{stages:?}");
        assert_eq!(stages[0].device, DeviceId(1));
        assert!(stages[0].embedding && !stages[0].lm_head);
        assert!(stages[1].lm_head && !stages[1].embedding);
        assert_eq!(reason.to_string(), "pp_stage_host_traffic:0");

        let fast1 = |d: DeviceId| Some(if d.0 == 1 { 24.0 } else { 12.0 });
        let (stages, reason) = place_stages(&ranges, &devices, &fast1).unwrap();
        assert_eq!(stages[1].device, DeviceId(1));
        assert_eq!(stages[0].device, DeviceId(0));
        assert_eq!(reason, StageReason::HostTraffic(DeviceId(1)));

        let (stages, reason) = place_stages(&ranges, &devices, &|_| None).unwrap();
        assert_eq!(stages.iter().map(|s| s.device).collect::<Vec<_>>(), devices);
        assert_eq!(reason, StageReason::NominalLinks);

        assert!(place_stages(&ranges, &devices[..1], &fast0).is_err());
    }

    fn shape(name: &str, layers: u32, experts: (u32, u32), tied: bool) -> ModelShape {
        let llama = name == "llama";
        ModelShape {
            architecture: name.into(),
            num_layers: layers,
            hidden: if llama { 3072 } else { 2048 },
            num_attention_heads: if llama { 24 } else { 16 },
            num_kv_heads: if llama { 8 } else { 16 },
            head_dim: 128,
            intermediate: if llama { 8192 } else { 1024 },
            vocab: if llama { 128_256 } else { 50_304 },
            num_experts: experts.0,
            experts_per_token: experts.1,
            tied_embeddings: tied,
            weight_bytes: 0,
            max_position_embeddings: 4096,
        }
    }

    /// The costs read from a model shape are the hand-written ones of Llama-3.2-3B and
    /// OLMoE-1B-7B above (a MoE layer's bytes count every expert, its FLOPs the active ones),
    /// and the balanced split of Llama over 2 stages gives the head's stage fewer layers.
    /// Breaks if MoE layers are costed by every expert's FLOPs or the head is dropped.
    #[test]
    fn costs_from_the_model_shape() {
        assert_eq!(
            PipelineCosts::from_shape(&shape("llama", 28, (0, 0), true)),
            llama()
        );
        assert_eq!(
            PipelineCosts::from_shape(&shape("olmoe", 16, (64, 8), false)),
            olmoe()
        );
        let ranges = partition(
            &PipelineCosts::from_shape(&shape("llama", 28, (0, 0), true)),
            2,
            None,
        )
        .unwrap();
        assert_eq!(ranges[0].end, ranges[1].start);
        assert!(ranges[1].len() < ranges[0].len(), "{ranges:?}");
    }
}
