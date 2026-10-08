# Handoff: p7-review-fixes (paused 2026-10-08 on user request)

Branch `p7-review-fixes` off `p7-stack`. No gate run yet (`scripts/gate.sh --base p7-stack`, in the background).

## Finding 1 — qwen_think start mode: DONE (gate pending)

Commit 9753804d: `QwenThink::start` looks only at the text after the last `<|im_start|>` in the tail (the whole
tail when there is none). Regression test `marker_in_an_earlier_message_is_ignored` (it also re-checks the sample
case) is in `crates/turbine-model/src/reasoning/qwen_think.rs`. Not run on novanas yet; rustfmt check is clean.

## Finding 2 — admission vs the physical KV pool: NOT STARTED (design read done)

Facts:

- The ledger's `kv` capacity is the budget's `kv` pool: KV plus the state allocation.
- `kv_pool_config` (`crates/turbine-server/src/model.rs` ~1143) subtracts `max_running × state` before it sizes
  the block pool.
- `Admission::reservation_bytes` / `reserve_kv` (`crates/turbine-reliability/src/admission.rs`) reserve KV plus
  state as one `kv` reservation. `Reservation::state_bytes` is only a mark.

Rejected: a new `PoolKind::State`. TP mirror ledgers journal and replay ops per pool (`engine/tp.rs`,
`LedgerReplica`), so a new pool would ripple into TP replication.

Planned smallest change:

- `turbine-reliability/src/ledger.rs`: add `StateSlots { capacity, used: Mutex<u64> }` and a `StateCharge`
  guard whose Drop releases it. `Reservation.state_bytes: u64` becomes `state: Option<StateCharge>`, and
  `state_bytes()` returns the charge's bytes. `Reservation::bytes()` then counts KV only.
- `Admission::with_state_capacity(bytes)`: the KV need (blocks × block_bytes) is checked against the ledger and
  the state need against `StateSlots`. A state need above capacity is a Reject (`ContextExceedsKvCapacity`, or a
  new reason); one above what is available is a Queue (`KvReservation`). `reserve_kv` takes the KV reservation
  (group path: `try_reserve_group`, KV only) and attaches the state charge on rank 0.
- The ledger's `kv` capacity becomes the physical pool bytes: reduce the budget's `kv` pool by the state
  allocation where the ledger is built (server). `kv_pool_config` must then not subtract it a second time.
- Update `hybrid_reservation_includes_state` (kv_sim): `kv` holds 4 MiB, state 3 MiB.
- Regression test (kv_sim): 64 MiB budget, 16 running, 3 MiB/seq → a pool of 16 pages. Of three 8-page
  requests, the third is not admitted. Families with an empty StateLayout get state capacity 0 and behave as before.

## Finding 3 — state for n > 1: NOT STARTED

`scheduler.rs` `submit` / `submit_probe` set `r.estimate.state_bytes = self.state_bytes`. Change it to
`state_bytes × r.seqs.len()`.

Proposed rule (a design choice for the lead to confirm before building):

- State slots = `max_running_requests` sequences, so the allocation stays `max_running × state`.
- A request whose `n × state` exceeds the whole state allocation is refused at submit with a reason code (only
  for families with state). Otherwise it queues until state is available.
- Why per sequence: admitted requests are ≤ max_running, but sequences are not (the API only checks n ≥ 1).

Regression test: 3 MiB/seq, n=4 → 12 MiB reserved.
