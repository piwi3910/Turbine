# Reliability: pressure, admission and circuit breaker

Turbine's reliability layer decides, before memory runs out, how much new work to accept and what to give up to keep the engine running. It has three parts: a pressure controller that sets a state from memory and host signals, admission control that reserves KV for each request, and a circuit breaker that stops work after device errors and probes the device back into service.

Every automatic decision emits a reason code, a metric and a log event, so an operator can see what the server did and why.

## Memory budget

At startup Turbine computes the device budget and reserves part of it. The budget is what remains of device memory after the weights, the workspace, the runtime overhead and the emergency reserve.

| Key                                      | Default | Meaning                                                             |
| ---------------------------------------- | ------- | ------------------------------------------------------------------- |
| `reliability.emergency_vram_reserve`     | `2GiB`  | Held back from normal scheduling. Released only in SURVIVAL.        |
| `reliability.memory.host_reserve_bytes`  | `8GiB`  | Host memory kept out of the L1 KV tier.                             |
| `reliability.memory.device_budget_bytes` | not set | An explicit device budget. Unset, the budget comes from the device. |

The budget also reserves workspace and runtime overhead (`reliability.memory.workspace_bytes` and `runtime_overhead_bytes`). The L0 pool is what remains. `kv.gpu.max_bytes` caps it further.

## Pressure states

The pressure controller moves through five states:

```text
GREEN -> YELLOW -> ORANGE -> RED -> SURVIVAL
```

It escalates after `reliability.pressure.escalate_samples` (2 by default) consecutive samples above the current state's target. An allocation failure escalates it at once. It de-escalates one level after `reliability.pressure.deescalate_dwell` (10 s by default) of all-clear signals. The exit threshold of each state sits below its entry threshold, so the state does not flap around a single value.

Each state changes behaviour as follows.

| State    | What changes                                                                                                                                                                                             |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GREEN    | Normal operation.                                                                                                                                                                                        |
| YELLOW   | With the ladder enabled, the lowest KV tier compresses until its fill is back at `kv.ladder.low_water`.                                                                                                  |
| ORANGE   | Prefills larger than `reliability.admission.large_prefill_tokens` count as expensive. Prefetch requests are refused with 409 `pressure_too_high`.                                                        |
| RED      | The emergency reserve stays held. It is excluded from normal scheduling.                                                                                                                                 |
| SURVIVAL | Admission rejects new work with 503 `overloaded`. The emergency reserve is released so that admitted work can finish. How admitted work is kept live is set by `reliability.recovery.survival_liveness`. |

The current state, the signals behind it and its recent transitions are in `GET /turbine/v1/pressure` and `GET /turbine/v1/status`. The metrics are `turbine_pressure_state`, `turbine_pressure_transitions`, `turbine_pressure_signal` and `turbine_pressure_exhaustion_horizon_seconds`.

### SURVIVAL liveness

A server in SURVIVAL must still finish the work it has admitted. `reliability.recovery.survival_liveness` chooses how KV is released to do that:

- `requeue_unstarted` (default): requests that have not started are returned to the queue, and running requests keep their KV.
- `continue_prefills`: in-flight prefills continue, and the released KV comes from elsewhere.

## Admission

Every request reserves its worst-case KV footprint before it runs: the prompt plus `max_tokens`, in blocks. Admission returns one of three decisions:

- **Admit**: the reservation fits, and the request runs.
- **Queue** with a reason: the reservation does not fit now, and the request waits.
- **Reject** with a reason: the request cannot be admitted in the current state.

The waiting queue is bounded:

| Key                                          | Default | Effect                                                                                                                    |
| -------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------- |
| `reliability.admission.max_queue`            | `256`   | A full queue answers 429 `queue_full`.                                                                                    |
| `reliability.admission.queue_timeout`        | `30s`   | A request that waits longer answers 503 `queue_timeout`.                                                                  |
| `reliability.admission.large_prefill_tokens` | `2048`  | Prefills above this size count as expensive at ORANGE and above. Must be at least `kv.block_tokens`.                      |
| `reliability.admission.max_bypass`           | `8`     | Smaller requests may overtake a queue head that does not fit. This bounds how many may do so, so the head is not starved. |

`scheduler.max_running_requests` (64 in the example) caps the number of running requests, and `scheduler.max_queued_requests` bounds the channel between HTTP and the engine.

Admission counters are `turbine_admission_decisions`, `turbine_admission_queue_depth` and `turbine_admission_queue_wait_seconds`.

## Circuit breaker

The circuit breaker protects the device from work it cannot complete:

```text
HEALTHY -> DEGRADED -> CIRCUIT_OPEN -> DRAINING -> PROBING -> HEALTHY
```

- **DEGRADED** is entered on latency drift: the windowed p95 step time rises above the latency baseline. It returns to HEALTHY after a window with no trigger. Thermal throttling and stale telemetry also register here.
- **CIRCUIT_OPEN** is entered after `reliability.circuit.oom_recoveries_to_open` (3 by default) out-of-memory recoveries within the window, after a non-sticky device error, or after a failed collective. While open, new requests get 503 `circuit_open` with a `Retry-After` equal to the remaining cooldown, and `/ready` answers 503 `circuit_open`.
- **DRAINING** starts when the cooldown (`reliability.circuit.cooldown`, 30 s by default) has passed. Running requests get `reliability.circuit.drain_timeout` to finish. Those still running after it fail with `circuit_open`.
- **PROBING** runs internal probe work. `reliability.circuit.probe_successes` (3 by default) successful probes return the circuit to HEALTHY. A probe whose latency exceeds the baseline by too much, or that fails, returns it to CIRCUIT_OPEN.

Some faults are not recoverable by draining. A sticky device error (a fault that corrupts the context) or a controller failure drains the server and then exits with code 3. A supervisor should restart the process. A repeated kernel argument rejection, which is a bug that every request hits, also opens the circuit. A single rejected request fails only its own iteration.

The state is `turbine_circuit_state`, the transitions are `turbine_circuit_transitions` (labelled with the reason), and request failures are counted by reason in `turbine_request_failures`.

## Recovery from allocation failures

When an allocation fails, Turbine tries recovery before it fails the request. `reliability.recovery.max_retries` (3 by default, at most 10) bounds the retries, and `reliability.recovery.backoff` (50 ms) spaces them. A successful recovery is counted in `turbine_recoveries` and `turbine_recovery_retries`; an unsuccessful one is counted in `turbine_allocation_failures`.

## Shutdown

On SIGINT or SIGTERM the server stops accepting work and answers 503 `shutting_down` on `/ready` and on new requests. Running requests get `server.shutdown_grace` (30 s in the example) to finish. The rest are cancelled, their KV is released, and the process exits with code 0. Cancellation releases resources promptly; this is tested.

## Fault injection

A build with the `fault-injection` feature can inject faults to test these paths:

- `reliability.fault_injection.alloc_fail_every`: fail every Nth pool reservation.
- `reliability.fault_injection.oom_at_iteration`: raise a device out-of-memory error at an iteration.
- `reliability.fault_injection.kernel_error_at_iteration` and `kernel_error_sticky`: raise a kernel error, optionally sticky.
- Telemetry overrides for temperature and delay.

A default build refuses the `reliability.fault_injection` section with exit code 2. Do not enable fault injection in production configurations.

## Exit codes

| Code | Reliability meaning                                                                                                           |
| ---- | ----------------------------------------------------------------------------------------------------------------------------- |
| 1    | The memory budget cannot hold the model, or a device-side resource is unusable at startup.                                    |
| 3    | A fatal circuit: a sticky device error, a pressure-controller failure, or an engine panic. A supervisor restarts the process. |

## Verifying the behaviour

Host tests run without a GPU:

```sh
cargo test -p turbine-reliability                  # budget, ledger, pressure, admission, throttle, circuit
cargo test -p turbine-scheduler --test overload_sim  # deterministic 10x overload, OOM recovery, cancellation
cargo test -p turbine-server --features fault-injection --test fault   # fault injection
```

The overload simulation is deterministic and needs no GPU. The SURVIVAL liveness regressions are named tests in `overload_sim`, and a sweep over seeds prints the outcome for each. Overload soak runs on a GPU host and checks behaviour over minutes and hours. See [Testing and quality gates](../testing.md).
