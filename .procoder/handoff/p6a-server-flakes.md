# Handoff: p6a-server-flakes (tiny_server failures under load)

Branch `p6a-server-flakes` from integration `eba77a1`. Host-only work, no GPU.

## What failed and why

Seen on the tip and on two unrelated branches, 3 remote runs each (`scratchpad/r9-tip-flaky.txt`,
`r9-rope-flaky.txt`): the gate's tests run at nice 19 on cores 12-15, the cores the CPU fixture jobs
also use (load average about 22 on 16 cores).

### slow_client_paused_then_cancelled: a real server bug (fixed)

The stream ended with `internal_error` instead of `slow_client`. The lead's hypothesis is **confirmed**
by the deterministic engine test `engine::r#loop::tests::slow_client_closed_after_cancel_ends_with_slow_client`
(fake clock, manual turns). Before the fix it was red: the client got `Started` and three tokens, then a
bare close. The sequence:

1. The slow stream pauses on a full channel, which starts the slow-client timer.
2. One `slow_client_timeout` later, `expire_deadlines` cancels the request. `on_dropped` accounts it as
   `slow_client` and queues the `slow_client` error event behind the full channel, and `Deadlines::expired`
   restarts the timer.
3. The client has still not drained the channel one timeout later, because on a loaded host the test's
   `other` request plus its polling takes more than about 3 s. `expire_deadlines` then sees a `done`
   request and calls `forget(id)`. That drops the held events, including the error event, and the sender.
4. The API stream gets `None` with no finish (`turbine-api` `openai/stream.rs`,
   `ApiError::internal(NO_FINISH)`) and reports `internal_error` (with the fixed message from 38703b9).
   The metrics were already right: `turbine_requests_cancelled_total{reason="slow_client"}` = 1.

Fix (`crates/turbine-server/src/engine/requests.rs`, `loop.rs`):
- `ActiveRequest` now reserves one slot of its output channel at admission, using `try_reserve_owned`.
- The new `ActiveRequest::close_with(event)` drops the held events and sends `event` through that slot.
- When `expire_deadlines` closes a done request for `SlowClient`, it calls
  `close_with(error_event(SlowClient, …))` before `forget`.
- A client that reads on now sees what the channel buffered, then `slow_client`, then `[DONE]`.
- The same applies to a request that finished `Ok` and whose `Finished` event stays unread for a whole
  timeout. Its stream is truncated and now says `slow_client` instead of `internal_error`. Its accounting
  stays `ok`, because `account` runs once.
- The channel's usable capacity drops by one: 255 of `EVENT_CHANNEL_CAPACITY` 256.
- Tests: the new loop test, and `requests::tests::history_stop_strings_and_held_events`, which now also
  covers `close_with` on a full channel.
- Mutation check: with no reserved slot, both tests fail.

### phase2_metrics_and_reasons: client read timeout (timing only)

The panic at `tiny_server.rs:597:45` is `.expect("read response head")` in one of the six request threads.
The join at 2464 re-raises it.

These are non-streaming requests, so the response head arrives only when the request ends. The sixth
request waits in the admission queue for KV until one of the five 450-token requests finishes, then
generates its own 450 tokens. In a debug build on four starved cores that takes more than `request()`'s
30 s socket read timeout, so `read_line` returns `WouldBlock` or `TimedOut`.

The server did nothing wrong, and the test already set `queue_timeout: 10m`.

### queue_full_429: queue_timeout (timing only)

A queued request waited behind the held 1,500-token stream (20 logprobs per token) for longer than the
default `reliability.admission.queue_timeout`, so it got 503 `queue_timeout`, which is correct server
behaviour. Once the queue timeout is raised, the next limit would be the same 30 s client read timeout.

## Test changes (`crates/turbine-server/tests/tiny_server.rs`)

- `request_within(…, limit)` is `request()` with a caller-chosen read timeout. `request()` keeps 30 s.
- `QUEUED_RESPONSE_LIMIT` is 10 min. It only bounds a hung server; no assertion measures the wait.
- `queue_full_429`: sets `queue_timeout: 10m`, and the five clients use `QUEUED_RESPONSE_LIMIT`.
- `phase2_metrics_and_reasons`: the six concurrent clients use `QUEUED_RESPONSE_LIMIT`.
- `slow_client_paused_then_cancelled`: no change. The fix makes its error code independent of timing.
- No assertion was removed or loosened, and no sleep was added.

## Not investigated

`server_cli sigterm_drains_then_cancels` failed once in the lead's runs; it is out of this brief's scope.

## Proposal (not implemented): gate tests vs fixture cores

`remote-cargo.sh` pins builds and tests to cores 12-15 at nice 19, the same cores the CPU fixture jobs use
(YaRN self-spread python at about 180 %). A gate run that shares those cores with fixtures turns
wall-clock bounds into lottery tickets, and 4 cores for the parallel tiny_server binary (each test a
server process plus clients) is already tight.

Options, in order of preference:
1. While fixtures run, give the gate cores the bench does not use. Only lab-bench and the GPU drivers need
   0-11, and only while they hold `bench.lock`. Take 8-11 (for example) when `bench.lock` is free, else
   12-15.
2. Pin the fixture jobs to 14-15 and let the gate have 12-13 plus its own nice level.
3. Run tiny_server with `--test-threads 2` in the gate.

Whichever is chosen, the tests should keep their bounds generous, as done here.

## Verification

- `remote-cargo test -p turbine-server --bin turbine-server engine::`: 44 passed. The new test was red
  before the fix, and the mutation (no reserved slot) fails both tests.
- `remote-cargo test -p turbine-server --test tiny_server -- queue_full_429 phase2_metrics_and_reasons
  slow_client_paused_then_cancelled request_and_queue_timeouts`: 4 passed in 22.9 s. That run did not have
  the lead's heavy load, so it does not show robustness under load.
- Gate: `scripts/gate.sh --base eba77a1`. It is running detached from the Mac with nohup; its log,
  `scratchpad/flakes-gate.log`, ends in `gate: …` and `rc=<n>`.
  - The first attempt started about 23:50 +04 and had checked about 146 dependency crates after 80 min.
    novanas cores 12-15 were starved: roughly one crate every 1-2 minutes.
  - That attempt's remote clippy still holds the target-dir lock, and the detached rerun waits on it.
  - **Gate result not yet known when this handoff was written.**
