//! The `model_family` suite: every registered family's tiny checkpoint
//! ([`ModelFamily::write_tiny`]) runs on the `cpu-reference` provider through the family's own
//! weight slots and executor, and must match the naive decoder ([`crate::testing::naive`],
//! which reads the checkpoint under its own names) and give the same logits however the work
//! is batched, chunked or paged.

use std::collections::HashMap;
use std::sync::Arc;

use turbine_core::registry::Registry;
use turbine_core::types::{BlockId, DeviceId, KvLayout, SeqId};
use turbine_kernels::{KernelMetrics, KernelRegistry, cpu_reference_provider};
use turbine_observability::MetricsRegistry;
use turbine_tensor::host::HostMemory;
use turbine_tensor::{DeviceBuffer, DeviceMemory, KvPoolView};

use super::{ConformanceFailure, Report, argmax, bitwise_equal, ensure, max_abs_diff};
use crate::executor::{
    self, BatchInput, ExecutorOptions, Logits, ModelExecutor, SeqSlice, build_executor,
};
use crate::families::{FamilyRef, ModelFamily};
use crate::testing::TempDir;
use crate::testing::naive::Naive;
use crate::testing::tiny::TinySpec;
use crate::{MAX_STAGING_BYTES, SafetensorsIndex, WeightLoader};

const SEED: u64 = 11;
/// Prompt tokens of the naive, chunked, paged and fusion runs.
const PROMPT_LEN: usize = 20;
/// Greedy decode steps after the prefill.
const DECODE_STEPS: usize = 4;
const MAX_SEQ_LEN: u32 = 64;
const MAX_SEQS: u32 = 4;
/// The serving default page, and the small page the page check compares it with.
const PAGE: u32 = 128;
const SMALL_PAGE: u32 = 16;
/// Chunk size of the chunked-prefill check.
const CHUNK: usize = 3;
/// Bound of the executor against the naive decoder, and of fused against unfused ops.
const MAX_LOGIT_DIFF: f32 = 2e-2;

/// Runs every check over every family of `reg`; `Err` lists each broken check.
///
/// Per family: `hf_names` (at least one Hugging Face name, none claimed by another family of
/// `reg`), `tiny` (it writes its tiny checkpoint, whose `architectures[0]` it claims), `load`
/// (its weight slots cover every checkpoint tensor and its executor builds on the CPU
/// provider; the family's other checks need it), `naive` (a prefill and greedy decode within
/// 2e-2 of the naive decoder, argmax equal), `ragged` (two sequences batched give each the
/// logits of its own run, bit for bit), `chunked` (a prefill in chunks of 3 equals the whole
/// prefill, bit for bit), `pages` (16-token pages equal 128-token pages, bit for bit), `fused`
/// (`fused_ops` off within 2e-2 of on).
pub fn families_suite(reg: &Registry<dyn ModelFamily>) -> Result<(), Vec<ConformanceFailure>> {
    let mut report = Report::new(reg);
    let mut claimed: HashMap<&str, &str> = HashMap::new();
    for family in reg.iter() {
        let name = family.name();
        report.check(name, "hf_names", || {
            ensure(!family.hf_architectures().is_empty(), || {
                "serves no Hugging Face architecture".into()
            })?;
            for hf in family.hf_architectures() {
                if let Some(other) = claimed.insert(hf, name) {
                    return Err(format!("{hf} is also claimed by {other}"));
                }
            }
            Ok(())
        });
    }

    let tmp = TempDir::new("turbine-conformance-families");
    let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
    for family in reg.iter() {
        let name = family.name();
        let mut spec = None;
        let written = report.check(name, "tiny", || {
            let s = family.write_tiny(&tmp.path().join(name), SEED);
            ensure(
                family
                    .hf_architectures()
                    .contains(&s.config.hf_architecture.as_str()),
                || {
                    format!(
                        "the tiny checkpoint says {}, which the family does not claim",
                        s.config.hf_architecture
                    )
                },
            )?;
            spec = Some(s);
            Ok(())
        });
        let Some(mut spec) = spec.filter(|_| written) else {
            continue;
        };
        // Run the family under test even when `reg` is not the crate's registry (which
        // `load_model_config` resolved the checkpoint through).
        spec.config.family = FamilyRef(family);

        let mut base = None;
        let loaded = report.check(name, "load", || {
            base = Some(cpu_model(&spec, &mem, PAGE, ExecutorOptions::default())?);
            Ok(())
        });
        let Some(mut base) = base.filter(|_| loaded) else {
            continue;
        };
        report.check(name, "naive", || naive_check(&spec, base.as_mut(), &mem));
        report.check(name, "ragged", || ragged_check(&spec, base.as_mut(), &mem));
        report.check(name, "chunked", || {
            chunked_check(&spec, base.as_mut(), &mem)
        });
        report.check(name, "pages", || {
            let mut small = cpu_model(&spec, &mem, SMALL_PAGE, ExecutorOptions::default())?;
            // 128-token pages: one block; 16-token pages: blocks out of order in a pool.
            let big = greedy_rows(&spec, base.as_mut(), &mem, 2, &[BlockId(1)])?;
            let table = [BlockId(5), BlockId(2), BlockId(7), BlockId(0)];
            let small_rows = greedy_rows(&spec, small.as_mut(), &mem, 8, &table)?;
            for (step, (a, b)) in big.iter().zip(&small_rows).enumerate() {
                ensure(bitwise_equal(a, b), || {
                    format!(
                        "step {step}: 16-token pages differ from 128-token pages by {}",
                        max_abs_diff(a, b)
                    )
                })?;
            }
            Ok(())
        });
        report.check(name, "fused", || {
            let mut unfused = cpu_model(&spec, &mem, PAGE, ExecutorOptions::from_fused_ops(false))?;
            let fused = greedy_rows(&spec, base.as_mut(), &mem, 1, &[BlockId(0)])?;
            let plain = greedy_rows(&spec, unfused.as_mut(), &mem, 1, &[BlockId(0)])?;
            for (step, (a, b)) in fused.iter().zip(&plain).enumerate() {
                let diff = max_abs_diff(a, b);
                ensure(diff <= MAX_LOGIT_DIFF, || {
                    format!("step {step}: fused ops differ from unfused by {diff}")
                })?;
            }
            Ok(())
        });
    }
    report.finish()
}

/// The family's executor (`spec.config.family`) on the CPU provider over pages of
/// `block_tokens`, after loading the checkpoint through the family's weight slots; every
/// checkpoint tensor must map to a slot.
fn cpu_model(
    spec: &TinySpec,
    mem: &Arc<dyn DeviceMemory>,
    block_tokens: u32,
    opts: ExecutorOptions,
) -> Result<Box<dyn ModelExecutor>, String> {
    let cfg = &spec.config;
    let index = SafetensorsIndex::open(&spec.dir).map_err(|e| e.to_string())?;
    let slots = cfg.family.0.weight_slots(cfg);
    let weights = WeightLoader::load_format(
        cfg.weight_format.get(),
        &index,
        &slots,
        mem,
        MAX_STAGING_BYTES,
    )
    .map_err(|e| e.to_string())?;
    ensure(
        weights.unexpected.is_empty() && weights.ignored.is_empty(),
        || {
            format!(
                "checkpoint tensors without a weight slot: {:?}, ignored: {:?}",
                weights.unexpected, weights.ignored
            )
        },
    )?;
    let provider = cpu_reference_provider();
    let order = [provider.id()];
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    let reqs =
        executor::available_requirements(cfg, block_tokens, opts, std::slice::from_ref(&provider));
    let registry = KernelRegistry::build(vec![provider], &order, &reqs, &metrics, None)
        .map_err(|e| e.to_string())?;
    build_executor(
        cfg,
        weights,
        Arc::new(registry),
        Arc::clone(mem),
        block_tokens,
        MAX_SEQ_LEN,
        MAX_SEQS,
        opts,
    )
    .map_err(|e| e.to_string())
}

/// A deterministic prompt of `PROMPT_LEN` ids below `vocab`.
fn prompt(vocab: u32, salt: u32) -> Vec<u32> {
    (0..PROMPT_LEN as u32)
        .map(|i| (i * 37 + 11 + salt) % vocab)
        .collect()
}

/// A KV pool of `blocks` blocks of `layout`.
struct Pool {
    storage: DeviceBuffer,
    layout: KvLayout,
    blocks: u32,
}

impl Pool {
    fn new(mem: &Arc<dyn DeviceMemory>, layout: KvLayout, blocks: u32) -> Result<Pool, String> {
        let bytes = layout.block_bytes() * u64::from(blocks);
        let storage = DeviceBuffer::alloc(mem, bytes as usize).map_err(|e| e.to_string())?;
        Ok(Pool {
            storage,
            layout,
            blocks,
        })
    }

    fn view(&self) -> KvPoolView<'_> {
        KvPoolView {
            storage: &self.storage,
            layout: self.layout,
            num_blocks: self.blocks,
            layer_stride_bytes: self.layout.block_bytes() / u64::from(self.layout.num_layers)
                * u64::from(self.blocks),
        }
    }
}

/// One forward of `seqs`: `(new tokens, start position, block table)` each.
fn forward(
    exec: &mut dyn ModelExecutor,
    pool: &Pool,
    seqs: &[(&[u32], u32, &[BlockId])],
) -> Result<Logits, String> {
    let mut tokens = Vec::new();
    let mut positions = Vec::new();
    let mut slices = Vec::new();
    for (i, &(new, start, table)) in seqs.iter().enumerate() {
        slices.push(SeqSlice {
            seq: SeqId(i as u64 + 1),
            q_start: tokens.len() as u32,
            q_len: new.len() as u32,
            kv_len: start + new.len() as u32,
            block_table: table,
            reduce: None,
        });
        tokens.extend_from_slice(new);
        positions.extend(start..start + new.len() as u32);
    }
    exec.forward(&BatchInput {
        tokens: &tokens,
        positions: &positions,
        seqs: &slices,
        kv: &pool.view(),
    })
    .map_err(|e| e.to_string())
}

/// The rows of a whole-prompt prefill and `DECODE_STEPS` greedy decode steps of one sequence
/// on `table` in a pool of `blocks` blocks.
fn greedy_rows(
    spec: &TinySpec,
    exec: &mut dyn ModelExecutor,
    mem: &Arc<dyn DeviceMemory>,
    blocks: u32,
    table: &[BlockId],
) -> Result<Vec<Vec<f32>>, String> {
    let pool = Pool::new(mem, *exec.kv_layout(), blocks)?;
    let tokens = prompt(spec.vocab, 0);
    let mut rows = vec![
        forward(exec, &pool, &[(&tokens, 0, table)])?
            .row(0)
            .to_vec(),
    ];
    let start = tokens.len() as u32;
    for len in start..start + DECODE_STEPS as u32 {
        let next = argmax(rows.last().expect("a row"));
        rows.push(
            forward(exec, &pool, &[(&[next], len, table)])?
                .row(0)
                .to_vec(),
        );
    }
    Ok(rows)
}

/// The prefill and each greedy decode step within [`MAX_LOGIT_DIFF`] of the naive decoder,
/// with the same argmax.
fn naive_check(
    spec: &TinySpec,
    exec: &mut dyn ModelExecutor,
    mem: &Arc<dyn DeviceMemory>,
) -> Result<(), String> {
    let naive = Naive::load(&spec.dir, &spec.config);
    let rows = greedy_rows(spec, exec, mem, 1, &[BlockId(0)])?;
    let mut tokens = prompt(spec.vocab, 0);
    for (step, row) in rows.iter().enumerate() {
        let want = naive.last_logits(&tokens);
        let diff = max_abs_diff(row, &want);
        ensure(diff <= MAX_LOGIT_DIFF, || {
            format!("step {step}: max |Δ logit| {diff} against the naive decoder")
        })?;
        ensure(argmax(row) == argmax(&want), || {
            format!(
                "step {step}: argmax {} but the naive decoder's is {}",
                argmax(row),
                argmax(&want)
            )
        })?;
        tokens.push(argmax(row));
    }
    Ok(())
}

/// Two prompts of different lengths prefilled together, then decoded together, give each the
/// logits of its own single-sequence run, bit for bit.
fn ragged_check(
    spec: &TinySpec,
    exec: &mut dyn ModelExecutor,
    mem: &Arc<dyn DeviceMemory>,
) -> Result<(), String> {
    let pool = Pool::new(mem, *exec.kv_layout(), 4)?;
    let a = prompt(spec.vocab, 0);
    let b = prompt(spec.vocab, 5)[..13].to_vec();
    let (ta, tb) = ([BlockId(2)], [BlockId(0)]);
    let batch = forward(exec, &pool, &[(&a, 0, &ta), (&b, 0, &tb)])?;
    let (na, nb) = (argmax(batch.row(0)), argmax(batch.row(1)));
    let decode = forward(
        exec,
        &pool,
        &[(&[na], a.len() as u32, &ta), (&[nb], b.len() as u32, &tb)],
    )?;
    let table = [BlockId(3)];
    for (i, (p, next)) in [(&a, na), (&b, nb)].into_iter().enumerate() {
        let single = forward(exec, &pool, &[(p, 0, &table)])?;
        ensure(bitwise_equal(batch.row(i), single.row(0)), || {
            format!(
                "sequence {i} prefill: batched differs from single by {}",
                max_abs_diff(batch.row(i), single.row(0))
            )
        })?;
        let single = forward(exec, &pool, &[(&[next], p.len() as u32, &table)])?;
        ensure(bitwise_equal(decode.row(i), single.row(0)), || {
            format!(
                "sequence {i} decode: batched differs from single by {}",
                max_abs_diff(decode.row(i), single.row(0))
            )
        })?;
    }
    Ok(())
}

/// A prefill in chunks of [`CHUNK`] tokens ends with the whole prefill's row, and the decode
/// step after it matches too, bit for bit.
fn chunked_check(
    spec: &TinySpec,
    exec: &mut dyn ModelExecutor,
    mem: &Arc<dyn DeviceMemory>,
) -> Result<(), String> {
    let pool = Pool::new(mem, *exec.kv_layout(), 2)?;
    let tokens = prompt(spec.vocab, 0);
    let (whole_table, chunk_table) = ([BlockId(0)], [BlockId(1)]);
    let whole = forward(exec, &pool, &[(&tokens, 0, &whole_table)])?
        .row(0)
        .to_vec();
    let mut last = Vec::new();
    for (i, chunk) in tokens.chunks(CHUNK).enumerate() {
        let start = (i * CHUNK) as u32;
        last = forward(exec, &pool, &[(chunk, start, &chunk_table)])?
            .row(0)
            .to_vec();
    }
    ensure(bitwise_equal(&whole, &last), || {
        format!(
            "prefill in chunks of {CHUNK} differs from the whole prefill by {}",
            max_abs_diff(&whole, &last)
        )
    })?;
    let next = argmax(&whole);
    let len = tokens.len() as u32;
    let a = forward(exec, &pool, &[(&[next], len, &whole_table)])?;
    let b = forward(exec, &pool, &[(&[next], len, &chunk_table)])?;
    ensure(bitwise_equal(a.row(0), b.row(0)), || {
        format!(
            "the decode after a chunked prefill differs by {}",
            max_abs_diff(a.row(0), b.row(0))
        )
    })
}
