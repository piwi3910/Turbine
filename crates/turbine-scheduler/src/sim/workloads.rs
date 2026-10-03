//! Seeded simulator workloads shared by the simulator's invariant tests (`sim::tests`), the
//! plan digests and the scheduling-policy conformance suite
//! (`crate::policy::conformance::policies_suite`). Test-only.

use std::sync::Arc;
use std::time::Duration;

use turbine_core::types::{DType, DeviceId, KvLayout, Priority, SeqId};
use turbine_kv::{BlockPool, BlockPoolConfig};
use turbine_tensor::DeviceMemory;
use turbine_tensor::host::HostMemory;

use super::{ArrivalProcess, CostModel, SimArrival, SimExecutor, SimReport, Simulation};
use crate::policy::SchedulingPolicy;
use crate::scheduler::SchedulerParams;

/// Accounting-only pool of 16-token blocks: zero bytes per block.
pub(crate) fn pool(n: u32) -> BlockPool {
    pool_of(n, 16)
}

/// Accounting-only pool of `n` blocks of `block_tokens` tokens.
pub(crate) fn pool_of(n: u32, block_tokens: u32) -> BlockPool {
    let layout = KvLayout {
        num_layers: 0,
        num_kv_heads: 0,
        head_dim: 0,
        dtype: DType::BF16,
        block_tokens,
    };
    let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 0);
    BlockPool::new(
        BlockPoolConfig {
            layout,
            num_blocks: n,
        },
        mem,
    )
    .unwrap()
}

pub(crate) fn params() -> SchedulerParams {
    SchedulerParams {
        max_running_requests: 32,
        max_batch_tokens: 512,
        prefill_chunk_tokens: 128,
        max_queued_requests: 1024,
        chunked_prefill: true,
        block_tokens: 16,
        free_watermark: 0.01,
        max_seq_len: 4096,
        queue_timeout: Duration::from_secs(3600),
        recent_window: None,
    }
}

pub(crate) fn secs(s: f64) -> Duration {
    Duration::from_secs_f64(s)
}

pub(crate) fn per_seq_cost() -> SimExecutor {
    SimExecutor {
        cost: CostModel {
            per_prefill_token_s: 0.0,
            per_decode_step_s: 0.0,
            per_seq_s: 1.0,
        },
    }
}

pub(crate) fn realistic_cost() -> SimExecutor {
    SimExecutor {
        cost: CostModel {
            per_prefill_token_s: 0.000_2,
            per_decode_step_s: 0.02,
            per_seq_s: 0.000_5,
        },
    }
}

pub(crate) type Policy = &'static dyn SchedulingPolicy;

/// A simulation of the real scheduler under `policy`.
pub(crate) fn simulation(
    policy: Policy,
    p: SchedulerParams,
    pool: BlockPool,
    exec: SimExecutor,
    arrivals: ArrivalProcess,
) -> Simulation {
    Simulation::new(p, pool, exec, arrivals).with_policy(policy)
}

/// 1,000 seeded Poisson arrivals; `paused` is paused for iterations 200..260.
pub(crate) fn poisson_run(policy: Policy, paused: Option<SeqId>) -> SimReport {
    let arrivals = ArrivalProcess::poisson(4.0, 42).with_limit(1000);
    let mut sim = simulation(policy, params(), pool(1024), realistic_cost(), arrivals);
    if let Some(seq) = paused {
        sim.pause_at(200, seq);
        sim.resume_at(260, seq);
    }
    sim.run(secs(100_000.0))
}

/// The Poisson run with chunked prefill, and two scripted prompts without it (600 tokens:
/// over the budget; 300: fits).
pub(crate) fn chunk_budget_runs(policy: Policy) -> (SimReport, SimReport) {
    let chunked = poisson_run(policy, None);
    let p = SchedulerParams {
        chunked_prefill: false,
        ..params()
    };
    let arrivals = vec![
        SimArrival::new(secs(0.0), 600, 4),
        SimArrival::new(secs(0.0), 300, 4),
    ];
    let mut sim = simulation(
        policy,
        p,
        pool(256),
        realistic_cost(),
        ArrivalProcess::scripted(arrivals),
    );
    (chunked, sim.run(secs(100.0)))
}

/// Two requests of `prompt` + `max_new` tokens on a pool of `pool_blocks` blocks of
/// `block_tokens`, too small for both to finish together. R1 has the lower priority (larger
/// value) and is the only victim; R2 arrives later and waits for a free running slot.
pub(crate) fn preemption_run(
    policy: Policy,
    block_tokens: u32,
    pool_blocks: u32,
    prompt: u32,
    max_new: u32,
) -> SimReport {
    let mut r0 = SimArrival::new(secs(0.0), prompt, max_new);
    r0.priority = Priority(0);
    let mut r1 = SimArrival::new(secs(0.0), prompt, max_new);
    r1.priority = Priority(1);
    let r2 = SimArrival::new(secs(0.5), 20, 5);
    let p = SchedulerParams {
        max_running_requests: 2,
        max_batch_tokens: 256,
        prefill_chunk_tokens: 64,
        block_tokens,
        ..params()
    };
    let mut sim = simulation(
        policy,
        p,
        pool_of(pool_blocks, block_tokens),
        realistic_cost(),
        ArrivalProcess::scripted(vec![r0, r1, r2]),
    );
    sim.run(secs(1000.0))
}
