//! Lab only, two GPUs (P5 Task 32 / Task 34): Llama-3.2-3B at tp 2 over `hostmem` (its RCCL
//! delegate above the thresholds, as served) on both R9700s. One sequence of N prompt tokens is
//! prefilled serially (twice: the determinism baseline) and with the prefill overlap (split at
//! [`overlap_split_row`]: a KV block boundary nearest the middle, Task 34), each into a fresh KV
//! pool; the test prints, per N, the logits' max |Δ| against the first serial run and, for the
//! first layers whose K/V differ, the differing token range, and fails unless the split run is
//! bitwise the serial one (logits and every layer's K/V). Before Task 34 the middle-row split
//! (mid-block for N = 385, 386, 401) differed from layer 2 on. Run by `scripts/lab-test.sh novanas --gpus 2 -- -p turbine-model --test
//! tp_overlap_lab -- --nocapture`.
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use turbine_core::clock::SystemClock;
use turbine_core::types::{BlockId, KvLayout, SeqId};
use turbine_distributed::collective::{self, Collective, CollectiveInit};
use turbine_kernels::test_support::{open_context_nth, require_backend, require_env_dir};
use turbine_kernels::{KernelMetrics, KernelRegistry, ShimContext, shim_provider};
use turbine_model::config::ModelArchConfig;
use turbine_model::executor::decoder::overlap_split_row;
use turbine_model::executor::{
    BatchInput, DecoderExecutor, ExecutorLimits, ExecutorOptions, ModelExecutor, SeqSlice,
};
use turbine_model::tp;
use turbine_model::{MAX_STAGING_BYTES, SafetensorsIndex, WeightLoader, load_model_config};
use turbine_observability::MetricsRegistry;
use turbine_tensor::{DeviceBuffer, DeviceMemory, KvPoolView};

const BLOCK_TOKENS: u32 = 128;
const MAX_TOKENS: u32 = 2048;
const POOL_BLOCKS: u32 = 8;

fn rank_decoder(
    cfg: &ModelArchConfig,
    dir: &Path,
    ctx: &Arc<ShimContext>,
    s: tp::ShardSpec,
    collective: Arc<dyn Collective>,
) -> DecoderExecutor {
    let opts = ExecutorOptions::default();
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let index = SafetensorsIndex::open(dir).expect("open safetensors");
    let slots = tp::weight_slots(cfg, s).expect("shard slots");
    let weights = WeightLoader::load(&index, &slots, &mem, MAX_STAGING_BYTES).expect("weights");
    let provider = shim_provider(ctx.clone());
    let order = [provider.id()];
    let reqs =
        tp::available_requirements(cfg, s, BLOCK_TOKENS, opts, std::slice::from_ref(&provider))
            .expect("rank requirements");
    let card = provider.card_profile();
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    let registry = Arc::new(
        KernelRegistry::build(vec![provider], &order, &reqs, &metrics, card)
            .expect("every op has a provider"),
    );
    let spec = cfg
        .family
        .0
        .tp_decoder_spec()
        .expect("tensor-parallel hooks");
    let stream = mem.compute_stream();
    DecoderExecutor::new_tp(
        cfg,
        spec,
        weights,
        registry,
        mem,
        ExecutorLimits {
            block_tokens: BLOCK_TOKENS,
            max_batch_tokens: MAX_TOKENS,
            max_seqs: 4,
        },
        opts,
        Some(tp::TpContext {
            rank: s.rank,
            world: s.world,
            collective,
            stream,
        }),
    )
    .expect("rank executor")
}

/// One prefill of `n` tokens into a fresh (zeroed) pool: the last row's logits and the pool.
fn prefill(
    exec: &mut DecoderExecutor,
    mem: &Arc<dyn DeviceMemory>,
    layout: &KvLayout,
    vocab: u32,
    n: u32,
) -> (Vec<f32>, Vec<u8>) {
    let bytes = (layout.block_bytes() * u64::from(POOL_BLOCKS)) as usize;
    let storage = DeviceBuffer::alloc(mem, bytes).expect("pool");
    storage
        .whole()
        .write_bytes(&vec![0u8; bytes])
        .expect("zero");
    let kv = KvPoolView {
        storage: &storage,
        layout: *layout,
        num_blocks: POOL_BLOCKS,
        layer_stride_bytes: layout.block_bytes() / u64::from(layout.num_layers)
            * u64::from(POOL_BLOCKS),
    };
    let tokens: Vec<u32> = (0..n).map(|i| (i * 37 + 11) % vocab).collect();
    let positions: Vec<u32> = (0..n).collect();
    let table: Vec<BlockId> = (0..POOL_BLOCKS).map(BlockId).collect();
    let seqs = [SeqSlice {
        seq: SeqId(1),
        q_start: 0,
        q_len: n,
        kv_len: n,
        block_table: &table,
        block_formats: &[],
        reduce: None,
    }];
    let logits = exec
        .forward(&BatchInput {
            tokens: &tokens,
            positions: &positions,
            seqs: &seqs,
            kv: &kv,
        })
        .expect("prefill");
    let row = logits.row(0).to_vec();
    (row, storage.whole().read_bytes().expect("read pool"))
}

/// Per layer, K and V: the tokens (< n) whose bytes differ between `a` and `b`, as
/// `(layer, "K"|"V", first, last, count)`.
fn kv_diff(
    layout: &KvLayout,
    n: u32,
    a: &[u8],
    b: &[u8],
) -> Vec<(u32, &'static str, u32, u32, u32)> {
    let tok = (layout.num_kv_heads * layout.head_dim) as usize * layout.dtype.size_bytes();
    let bt = layout.block_tokens as usize;
    let layer_stride =
        (layout.block_bytes() / u64::from(layout.num_layers) * u64::from(POOL_BLOCKS)) as usize;
    let mut out = Vec::new();
    for l in 0..layout.num_layers {
        for (kvi, name) in [(0usize, "K"), (1, "V")] {
            let mut diff: Vec<u32> = Vec::new();
            for t in 0..n as usize {
                let (blk, i) = (t / bt, t % bt);
                let off = l as usize * layer_stride + ((blk * 2 + kvi) * bt + i) * tok;
                if a[off..off + tok] != b[off..off + tok] {
                    diff.push(t as u32);
                }
            }
            if let (Some(&f), Some(&last)) = (diff.first(), diff.last()) {
                out.push((l, name, f, last, diff.len() as u32));
            }
        }
    }
    out
}

#[test]
#[ignore = "needs two HIP devices, libturbine_hip.so, RCCL and TURBINE_TEST_MODEL_DIR (lab-test --gpus 2)"]
fn hostmem_tp2_3b_prefill_split_diagnosis() {
    if !require_backend("hip") {
        return;
    }
    let dir = require_env_dir("TURBINE_TEST_MODEL_DIR");
    let cfg = load_model_config(&dir).expect("config.json");
    let lib = collective::registry()
        .get("hostmem")
        .expect("registered")
        .load(None)
        .expect("load hostmem with its RCCL delegate");
    let id = lib.unique_id().expect("id");
    const SIZES: [u32; 7] = [256, 384, 385, 386, 401, 512, 640];
    let reports: Vec<Vec<String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2u32)
            .map(|rank| {
                let (lib, cfg, dir) = (Arc::clone(&lib), &cfg, &dir);
                scope.spawn(move || {
                    let ctx = open_context_nth("hip", rank as usize);
                    let mem: Arc<dyn DeviceMemory> = ctx.clone();
                    let collective = lib
                        .open(CollectiveInit {
                            rank: rank as usize,
                            world: 2,
                            unique_id: id,
                            init_timeout: Duration::from_secs(120),
                            op_timeout: Duration::from_secs(120),
                            clock: Arc::new(SystemClock::new()),
                            metrics: None,
                            memory: Some(Arc::clone(&mem)),
                            route_max_bytes: None,
                        })
                        .expect("open hostmem");
                    let s = tp::ShardSpec { rank, world: 2 };
                    let layout = tp::kv_layout(cfg, s, BLOCK_TOKENS).expect("layout");
                    let mut exec = rank_decoder(cfg, dir, &ctx, s, collective);
                    let mut lines = Vec::new();
                    for n in SIZES {
                        exec.set_prefill_overlap(None);
                        let serial = prefill(&mut exec, &mem, &layout, cfg.vocab_size, n);
                        let again = prefill(&mut exec, &mem, &layout, cfg.vocab_size, n);
                        assert!(exec.set_prefill_overlap(Some(2)), "overlap on");
                        let split = prefill(&mut exec, &mem, &layout, cfg.vocab_size, n);
                        let dmax = |a: &[f32], b: &[f32]| {
                            a.iter()
                                .zip(b)
                                .map(|(x, y)| (x - y).abs())
                                .fold(0f32, f32::max)
                        };
                        let row = overlap_split_row(
                            &[SeqSlice {
                                seq: SeqId(1),
                                q_start: 0,
                                q_len: n,
                                kv_len: n,
                                block_table: &[],
                block_formats: &[],
                                reduce: None,
                            }],
                            n as usize,
                            BLOCK_TOKENS as usize,
                        );
                        let exact = dmax(&serial.0, &split.0) == 0.0
                            && kv_diff(&layout, n, &serial.1, &split.1).is_empty();
                        lines.push(format!(
                            "{} rank {rank} n {n} split_row {row:?}: serial-vs-serial logits max|d| {} \
                             kv_diff {:?}; split-vs-serial logits max|d| {} first kv diffs {:?}",
                            if exact { "EXACT" } else { "DIFFERS" },
                            dmax(&serial.0, &again.0),
                            kv_diff(&layout, n, &serial.1, &again.1)
                                .into_iter()
                                .take(2)
                                .collect::<Vec<_>>(),
                            dmax(&serial.0, &split.0),
                            kv_diff(&layout, n, &serial.1, &split.1)
                                .into_iter()
                                .take(6)
                                .collect::<Vec<_>>(),
                        ));
                    }
                    lines
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank"))
            .collect()
    });
    for line in reports.iter().flatten() {
        println!("tp-split-diag {line}");
    }
    let differs: Vec<&String> = reports
        .iter()
        .flatten()
        .filter(|l| l.starts_with("DIFFERS"))
        .collect();
    assert!(
        differs.is_empty(),
        "the block-aligned overlap split differs from the serial prefill: {differs:?}"
    );
}
