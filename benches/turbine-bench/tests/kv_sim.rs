//! `turbine-bench kv-sim` policy comparison (P4 S-7, S-16).

use turbine_bench::kv_sim::{KvSimArgs, SimOutput, Workload, run};

fn sim(workload: Workload, policy: &'static str) -> f64 {
    let args = KvSimArgs {
        workload,
        policy,
        // 128-token blocks: 32,768 tokens (3.8 GB of Llama-3.2-3B KV) in L0, 4x that in L1.
        l0_blocks: 256,
        l1_blocks: 1024,
        l2_blocks: 0,
        seed: 1,
        output: SimOutput::Json,
    };
    let r = run(&args);
    assert_eq!(r, run(&args), "equal seeds give equal reports");
    eprintln!("{}", r.to_text());
    r.simulated_prefill_seconds
}

#[test]
fn cost_aware_beats_lru() {
    for (w, bound) in [
        (Workload::MultiTurn, 0.9),
        (Workload::Mixed, 0.9),
        (Workload::SharedSystem, 1.0),
    ] {
        let (ca, lru) = (sim(w, "cost_aware"), sim(w, "lru"));
        assert!(
            ca <= bound * lru,
            "{w:?}: cost_aware {ca:.3}s vs lru {lru:.3}s (bound {bound})"
        );
    }
}
