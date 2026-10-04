# Handoff: p6b-crash — the slab-run "release of unreferenced KV block" transient

Branch `p6b-crash`, from `p6b-stack` 7a67569. Root cause of the original transient **not
found**; two real reference-discipline defects on the release paths found and fixed instead
(each red first, mutation-checked; see the perf log 6b, "Release-path audit after the
slab-run crash transient"). Tree clean, gate run per commit, no lab Job, server or lock of
this branch left.

## What is known about the transient

- One crash: `engine thread panicked: release of unreferenced KV block BlockId(146)`, fatal
  exit 3, first 1 GiB-slab mt3 run (bench 153 ok/39 failed), binaries built at 4c62e82
  (pre-t17-merge). Never reproduced in the 6 later full runs.
- Evidence trail is gone: the crashed run's server log (`/tmp/slabs-server-*.log` on
  novanas) was overwritten by the reruns, and it carried no DEBUG events; the assertion is
  `BlockPool::release` (`crates/turbine-kv/src/pool.rs`), so only "some release path freed a
  block whose refcount was already 0" is known, not which.
- Static audit of every `pool.release` site (hierarchy, pool, scheduler, engine loop, sim):
  the demote/promote/attach/detach/finish paths balance their references, victims exclude
  demoting/busy keys (`eligible`), and `remove_copy` guards refcounts. Two defects found and
  fixed (below). The surviving suspicion is a rare interleaving in the promotion/detach/
  cancel machinery; nothing in the audit pins it.

## Commits

- `fix(kv)` + `test(server)`: the two fixes and their tests, plus the standing stress.
  - Promotion completing after a sibling's failed copy: the failure resolves the pending
    entry, so the sibling's completion found no pending entry and never released its target
    block — a permanent page leak per copy failure. `on_copy_done` (Promote) now releases
    the block when the owner has no pending entry and is not cancelled. Test
    `hierarchy::tests::a_promotion_completing_after_a_failed_sibling_releases_its_block`
    (red: 3 blocks leaked; mutation: drop the new `None => pool.release` arm → red).
  - L0 rewrite completion (`Compress` from L0, S-7): completed unconditionally. When the
    block's copy gained a holder (attach) or a recompute published a new L0 page while the
    rewrite ran, it filed the rewritten page and dropped the referenced source's location —
    `evict_cached` no-ops on a referenced page, so the source page strands outside the
    directory (class pages `allocate` can never reclaim), or a live page gets released (the
    transient's shape). Now the completion proceeds only when the directory's L0 location is
    still the submitted source page AND that page is unreferenced; otherwise only the
    rewritten page goes back. Tests
    `an_l0_rewrite_over_an_attached_copy_keeps_the_old_page_serving` (mutation: drop the
    refcount guard → attach served the rewrite's page) and
    `a_rewrite_completions_old_page_forgets_only_itself`. Dormant today (the recent window
    defaults 0, `kv.ladder.l0` refused at startup) — live the moment either switches on.
  - The rewrite's old page now also forgets its `l0_keys` mapping (`sync_l0_refs` mirrors
    refcounts through `l0_keys`; a stale entry let the recompressed copy inherit a new
    holder's refcount). **Its dedicated mutation check is inconclusive**: with the stale
    entry present, two `l0_keys` entries map one key, and HashMap order decides which write
    wins — the deterministic repro needs forcing iteration order. The line mirrors what
    `remove_copy` does and is kept; next builder may want a `BTreeMap` for `l0_keys` (also
    gives the sweeps a deterministic order) and re-pin the check.
  - `engine::r#loop::tests::multi_turn_release_paths_stress_without_an_unreferenced_release`
    (turbine-server, cpu backend, tiny Llama at head_dim 128 so tq4 exists): 12 seeds × 6
    sessions × 6 turns over one shared prefix, YELLOW from 1 % utilisation (queued prefixes
    release and attach again every turn), ladder on over L2, one stream dropped mid-run;
    asserts the engine never panics and L0 ends empty. 9 s. No repro of the transient.
- `docs(perf)`: the perf-log 6b entry above and this handoff.

## Open

1. The transient itself: root cause unknown. The stress covers the engine shape; if it ever
   fires, the pool assertion's message plus a DEBUG-enabled server log
   (`kv_plan`, `kv_queued_prefix_detach`, `kv_copy_timed`, `kv_transfer_failed`) should pin
   the path. A lab ladder-on mt3 ×3 watch is worthwhile at the next lab visit, and before
   `kv.recent_window_blocks` defaults on.
2. The window/rewrite flow on a class pool showed a pool-accounting inconsistency in my
   probes (total/free/cached did not sum over base+class pages after one plain request with
   the recent window on, sim). I could not decide whether that is real drift or the class
   accounting semantics (`total_blocks`, `cached_unreferenced` mix base and class counts);
   probes were cut off by the timebox. Worth one focused pass before the window defaults on:
   assert in a sim test that `used_blocks()` and per-class `used_pages` reconcile with
   refcounts after each step.
3. Carried from p6b-t17 (not mine, still open): bisect `lossy_tier_reuse{,_tq4}` between
   the last green quick tier and d04b4bc.
