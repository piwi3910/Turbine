//! The `logits_processor` suite: a processor leaves a neutral request alone, applies when its
//! request field is set, and — when it claims it can run on the device — gives the device
//! path's result: a step it applies to is then reduced on the device from the raw row
//! (`logits_reduce`, simulated here by the `cpu-reference` provider's kernel), so applying it
//! on the host must not change what that reduction picks.

use std::collections::HashMap;
use std::sync::Arc;

use turbine_core::registry::Registry;
use turbine_core::request::SamplingParams;
use turbine_core::types::{DType, DeviceId};
use turbine_kernels::{LogitsReduceConfig, LogitsReduceContext, cpu_reference_provider};
use turbine_tensor::host::HostMemory;
use turbine_tensor::{DeviceMemory, Tensor};

use super::{ConformanceFailure, Report, ensure};
use crate::sampling::{LogitsProcessor, ProcessorParams, ProcessorState, Touched};
use crate::structured::TokenMask;

/// Vocabulary of the suite's rows.
const VOCAB: usize = 64;
/// Candidates compared between the host and the device path.
const TOP_N: usize = 5;

/// Runs every check over every processor of `reg`; `Err` lists each broken check.
///
/// Per processor: `neutral` (it does not apply to a request that sets no processor field),
/// `applies` (it applies when every processor field is set and a mask is present), `device`
/// (only when [`LogitsProcessor::device_capable`]: over seeded rows, the host path — the
/// processor applied, then the top candidates — equals the `logits_reduce` reference's top
/// candidates of the raw row, the greedy token included).
pub fn processors_suite(
    reg: &Registry<dyn LogitsProcessor>,
) -> Result<(), Vec<ConformanceFailure>> {
    let mut report = Report::new(reg);
    let counts: HashMap<u32, u32> = [(4, 2)].into_iter().collect();
    let quiet = ProcessorState {
        prompt_tokens: &[1, 2],
        counts: &counts,
        step: 3,
        eos_token_ids: &[5],
        mask: None,
    };
    let neutral = ProcessorParams::new(&SamplingParams::default());
    let busy_params = ProcessorParams::new(&SamplingParams {
        logit_bias: vec![(3, 1.0)],
        repetition_penalty: 1.2,
        presence_penalty: 0.5,
        frequency_penalty: 0.5,
        min_tokens: 8,
        ..SamplingParams::default()
    });
    let mask = TokenMask::new_all(VOCAB);
    let busy = ProcessorState {
        mask: Some(&mask),
        ..quiet
    };
    for m in reg.iter() {
        let name = m.name();
        report.check(name, "neutral", || {
            ensure(!m.applies(&neutral, &quiet), || {
                "applies to a request that asks for no processor".into()
            })
        });
        report.check(name, "applies", || {
            ensure(m.applies(&busy_params, &busy), || {
                "does not apply when every processor field is set".into()
            })
        });
        if m.device_capable() {
            report.check(name, "device", || {
                for seed in 0..4u64 {
                    let raw = seeded_row(seed);
                    let mut host = raw.clone();
                    let mut touched = Touched::default();
                    m.apply(&mut host, &mut touched, &busy_params, &busy);
                    let host_top = top_ids(&host);
                    let device_top = device_top_ids(&raw)?;
                    ensure(host_top == device_top, || {
                        format!("row {seed}: host top-{TOP_N} {host_top:?}, device {device_top:?}")
                    })?;
                }
                Ok(())
            });
        }
    }
    report.finish()
}

/// A row of distinct values in [−8, 8) from a xorshift stream seeded by `seed`.
fn seeded_row(seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut row: Vec<f32> = (0..VOCAB)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // A distinct fraction per id keeps every value unique.
            ((state >> 40) % 16) as f32 - 8.0 + i as f32 / (2 * VOCAB) as f32
        })
        .collect();
    row.rotate_left((seed as usize) % VOCAB);
    row
}

/// The `TOP_N` largest ids of `row`, largest first, ties to the lower id, NaN never ranked.
fn top_ids(row: &[f32]) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..row.len() as u32)
        .filter(|&i| !row[i as usize].is_nan())
        .collect();
    ids.sort_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]).then(a.cmp(&b)));
    ids.truncate(TOP_N);
    ids
}

/// The top candidates the device's `logits_reduce` returns for `row` (greedy: no draw), as
/// the `cpu-reference` provider computes them.
fn device_top_ids(row: &[f32]) -> Result<Vec<u32>, String> {
    let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 20);
    let provider = cpu_reference_provider();
    let kernel = provider
        .logits_reduce()
        .ok_or("the cpu-reference provider has no logits_reduce")?;
    let tensor =
        |shape: &[usize], dtype| Tensor::empty(&mem, shape, dtype).map_err(|e| e.to_string());
    let mut logits = tensor(&[1, VOCAB], DType::F32)?;
    let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
    logits
        .storage
        .copy_from_host(0, &bytes)
        .map_err(|e| e.to_string())?;
    let mut scalars = Vec::new();
    for value in [0.0f32, 0.0, 1.0] {
        let mut t = tensor(&[1], DType::F32)?;
        t.storage
            .copy_from_host(0, &value.to_le_bytes())
            .map_err(|e| e.to_string())?;
        scalars.push(t);
    }
    let mut mode = tensor(&[1], DType::I32)?;
    mode.storage
        .copy_from_host(0, &0i32.to_le_bytes())
        .map_err(|e| e.to_string())?;
    let top_ids = tensor(&[1, TOP_N], DType::I32)?;
    let top_values = tensor(&[1, TOP_N], DType::F32)?;
    let lse = tensor(&[1], DType::F32)?;
    let sampled = tensor(&[1], DType::I32)?;
    let sampled_logit = tensor(&[1], DType::F32)?;
    let cfg = LogitsReduceConfig {
        vocab: VOCAB as u32,
        top_n: TOP_N as u32,
    };
    ensure(kernel.supports(&cfg), || format!("logits_reduce {cfg}"))?;
    kernel
        .execute(&mut LogitsReduceContext {
            logits: logits.view(),
            temperature: scalars[0].view(),
            uniform: scalars[1].view(),
            top_p: scalars[2].view(),
            mode: mode.view(),
            top_ids: top_ids.view(),
            top_values: top_values.view(),
            lse: lse.view(),
            sampled: sampled.view(),
            sampled_logit: sampled_logit.view(),
            rows: 1,
        })
        .map_err(|e| e.to_string())?;
    let mut out = vec![0u8; 4 * TOP_N];
    top_ids
        .storage
        .copy_to_host(0, &mut out)
        .map_err(|e| e.to_string())?;
    Ok(out
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
        .collect())
}
