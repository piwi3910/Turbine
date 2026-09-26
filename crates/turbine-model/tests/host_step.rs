//! Host-side cost of one engine decode iteration, without a GPU (`#[ignore]`d: a measurement,
//! not a check to run on every build). Per batch row: the sampler over a full Llama-3
//! vocabulary row, for a seeded and an unseeded request (one after another, and unseeded
//! through [`sample_rows`]), and the incremental detokenizer step
//! with the real tokenizer; per batch the logits byte → f32 conversion the executor does after
//! its device read. Prints median and minimum milliseconds per iteration.
//!
//! `cargo test --release -p turbine-model --test host_step -- --ignored --nocapture`, or on the
//! lab host `scripts/lab-test.sh novanas -- --release -p turbine-model --test host_step`.
//! `turbine-bench` sends no sampling parameters, so the server samples at temperature 1,
//! top_p 1, without top_k, logprobs or seed: the `unseeded` columns of each table's first row
//! (the engine samples a decode batch with `sample_rows`).
use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use turbine_core::request::SamplingParams;
use turbine_model::{IncrementalDetokenizer, SampleJob, Sampler, Tokenizer, sample_rows};

const VOCAB: usize = 128_256;
const ITERATIONS: usize = 40;

/// Llama-like logits: a broad bulk around 0 and a few confident candidates.
fn logits_row(rng: &mut ChaCha8Rng) -> Vec<f32> {
    let mut u = || (rng.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
    let mut row: Vec<f32> = (0..VOCAB)
        .map(|_| (u() + u() + u() + u() - 2.0) * 3.0)
        .collect();
    for k in 0..8 {
        let id = (u() * VOCAB as f32) as usize % VOCAB;
        row[id] = 18.0 - k as f32;
    }
    row
}

/// "median (min)" in milliseconds: on a busy host the minimum is the closer estimate of the
/// uncontended cost.
fn stat(samples: &mut [f64]) -> String {
    samples.sort_by(f64::total_cmp);
    format!(
        "{:8.3} ({:7.3})",
        samples[samples.len() / 2] * 1e3,
        samples[0] * 1e3
    )
}

fn samplers(batch: usize, temperature: f32, top_p: f32, seeded: bool) -> Vec<Sampler> {
    (0..batch)
        .map(|i| {
            let p = SamplingParams {
                temperature,
                top_p,
                seed: seeded.then_some(i as u64),
                ..SamplingParams::default()
            };
            Sampler::new(&p, &[], &[128_001, 128_008, 128_009])
        })
        .collect()
}

/// Samples every row with its sampler, one after another; returns the seconds taken.
fn serial(samplers: &mut [Sampler], mut logits: Vec<f32>) -> f64 {
    let t = Instant::now();
    for (s, row) in samplers.iter_mut().zip(logits.chunks_exact_mut(VOCAB)) {
        let token = s.sample(row, None).token;
        s.observe(token);
    }
    t.elapsed().as_secs_f64()
}

/// Samples every row through [`sample_rows`]; returns the seconds taken.
fn batched(samplers: &mut [Sampler], mut logits: Vec<f32>) -> f64 {
    let t = Instant::now();
    let jobs: Vec<SampleJob<'_>> = samplers
        .iter_mut()
        .zip(logits.chunks_exact_mut(VOCAB))
        .map(|(sampler, logits)| SampleJob {
            sampler,
            logits,
            mask: None,
        })
        .collect();
    let tokens: Vec<u32> = sample_rows(jobs).iter().map(|s| s.token).collect();
    for (s, token) in samplers.iter_mut().zip(tokens) {
        s.observe(token);
    }
    t.elapsed().as_secs_f64()
}

#[test]
#[ignore = "measurement: run with --release -- --ignored --nocapture"]
fn host_step_costs() {
    let tok_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/llama-3.2-3b-instruct/tokenizer.json");
    let tokenizer = Arc::new(Tokenizer::from_file(&tok_path).expect("fixture tokenizer"));
    let text = "The quick brown fox jumps over the lazy dog. Turbine serves Llama models on \
                Radeon cards with a paged KV cache, continuous batching and chunked prefill; \
                naïve café 日本語テキスト 👩‍👩‍👧 and some code: fn main() { println!(\"hi\"); }\n";
    let text_ids = tokenizer.encode(&text.repeat(8), false).expect("encode");
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let rows: Vec<Vec<f32>> = (0..8).map(|_| logits_row(&mut rng)).collect();

    println!(
        "vocab {VOCAB}, {ITERATIONS} iterations, {} threads available; ms per iteration: median (min)",
        std::thread::available_parallelism().map_or(1, |n| n.get())
    );
    println!(
        "{:>5} {:>4} {:>5} | {:>18} | {:>18} | {:>18} | {:>18} | {:>18}",
        "batch",
        "T",
        "top_p",
        "bytes->f32",
        "seeded serial",
        "unseeded serial",
        "unseeded rows",
        "detokenizer"
    );
    for batch in [16usize, 64] {
        let flat: Vec<f32> = (0..batch)
            .flat_map(|i| rows[i % rows.len()].clone())
            .collect();
        let bytes: Vec<u8> = flat.iter().flat_map(|v| v.to_le_bytes()).collect();
        for (temperature, top_p) in [(1.0f32, 1.0f32), (0.0, 1.0), (0.7, 0.9)] {
            let mut seeded = samplers(batch, temperature, top_p, true);
            let mut unseeded = samplers(batch, temperature, top_p, false);
            let mut rows_unseeded = samplers(batch, temperature, top_p, false);
            let mut detoks: Vec<IncrementalDetokenizer> = (0..batch)
                .map(|_| IncrementalDetokenizer::new(Arc::clone(&tokenizer)))
                .collect();
            let (mut convert, mut t_seeded, mut t_unseeded, mut t_rows, mut t_detok) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for it in 0..ITERATIONS {
                let t = Instant::now();
                let data: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                black_box(&data);
                convert.push(t.elapsed().as_secs_f64());
                t_seeded.push(serial(&mut seeded, flat.clone()));
                t_unseeded.push(serial(&mut unseeded, flat.clone()));
                t_rows.push(batched(&mut rows_unseeded, flat.clone()));
                let t = Instant::now();
                for d in &mut detoks {
                    black_box(d.push(text_ids[it % text_ids.len()]));
                }
                t_detok.push(t.elapsed().as_secs_f64());
            }
            println!(
                "{batch:>5} {temperature:>4} {top_p:>5} | {} | {} | {} | {} | {}",
                stat(&mut convert),
                stat(&mut t_seeded),
                stat(&mut t_unseeded),
                stat(&mut t_rows),
                stat(&mut t_detok)
            );
        }
    }
}
