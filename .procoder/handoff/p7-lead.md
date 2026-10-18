# Phase 7 lead handoff (2026-10-16)

This file hands the Phase 7 (model families) lead role to a new agent. Read it top to bottom before
acting. It assumes you have read `AGENTS.md` at the repository root.

## Where things are

- **Integration branch:** `p7-stack`, checked out in worktree
  `/Volumes/DATA/Development/Turbine/.claude/worktrees/agent-p6b-t2`. The tip is `6cbb3ebb`, and
  `scripts/gate.sh --base main` there is clean: `passed=1085 failed=0`.
- **`main`** is at `6a0d91f5`, pushed to `origin` (`c3bf4acf..6a0d91f5`, 2026-10-18). The user amended
  the rules that day: the lead merges `p7-stack` into `main` itself **and pushes** — never stops to
  ask for either. Put no attribution lines in commits.
- **Plan:** `.procoder/plans/phase-7-model-families.md`.
- **Spec:** `.procoder/specs/phase-7-model-families.md`.
- **Decisions:** `.procoder/ask/decisions.md`. Read the `P7 …` entries before re-opening any
  question.
- **Perf log:** `.procoder/perf-log.md`, the `P7 …` sections. Each closed or open support row has a
  `- P7 closed:` / `- P7 open:` line there, and `support::tests::phase7_rows` enforces them.
- **Per-task handoffs:** `.procoder/handoff/p7-t*.md`.
- **Phase is paused.** The user said "pause" on 2026-10-14. Nothing of ours is running on any host.
  Start work only when the user says to continue.

## Done (merged into `p7-stack`, gated)

Plan Tasks 1–26, 29, 30 and 36, plus fixes:

- the sharded memory-budget fix: quantized slots were counted as BF16;
- the Harmony `tool_choice: "required"` fix (`6cbb3ebb`): an EOS id that the grammar forces as a
  literal (gpt-oss `<|call|>`) no longer empties the mask;
- lab-script test updates.

Family outcomes:

| Family / checkpoint                    | Result                                                                                                                                                                                                                                                                                                            |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Qwen3-4B, Mistral-7B                   | `supported`                                                                                                                                                                                                                                                                                                       |
| Qwen3-30B-A3B FP8 (Qwen3-MoE, Task 18) | `supported` on gfx1201 at tp2 and ep2. tp2 is 400.8 tok/s against vLLM's 91.6 (4.37×). Golden, GSM8K and gates PASS.                                                                                                                                                                                              |
| Mixtral-8x7B FP8 (Task 22)             | Both placements stay `experimental` (decision A, "P7 Mixtral ep2 gate"). tp2 passes every leg: 3.17× vLLM, GSM8K 0.60 against vLLM's 0.59. ep2 passes everything except golden c16, which fails the `missing_top_k` clause: an EP-specific near-tie flip in 6 of 16 captures, with every \|Δ\| inside the bounds. |
| OLMoE FP8 KV (Task 36)                 | `supported`                                                                                                                                                                                                                                                                                                       |
| gpt-oss-20b (Task 26)                  | Runs on HIP, including Harmony channels and required tool calls. The golden trace passes; the router "bug" turned out to be a wrong bias vector in a repro.                                                                                                                                                       |

## Remaining work, in order

### 1. Task 27: gpt-oss fixtures and gate

**Fixture half (branch `p7-t27-fixture`, WIP commit `b168a9b5`).** The branch holds:

- the fused-expert MXFP4 support in `scripts/golden/dequantize_checkpoint.py`;
- `tests/golden/gpt-oss-20b/{prompts.jsonl,tolerance.json,README.md}`;
- the reference job script `.procoder/handoff/p7_ref_gptoss.sh`.

The BF16 copy already exists on novanas at `/home/piwi/turbine-models/gpt-oss-20b-bf16`.

The reference run hit its 6 h timeout after 13 of 16 prompts. The partial output is in
`/home/piwi/turbine-ci/scratch/p7-t27/out/reference.jsonl.tmp`. `hf_reference.py` has no resume.

Your options:

1. Add per-prompt resume to `hf_reference.py`. This is the same pattern Task 22 added to
   `self_spread.py` in commit `bd026dc6`: checkpoint after each prompt, then skip ids already
   present.
2. Rerun with a longer timeout.

Prompt p09 alone takes about 40 min, since it is 1943 tokens on CPU.

Once the reference is in, check that the tolerance follows the plan text, then commit the fixture.

**Gate half (after the fixture lands).** Follow plan Task 27 steps:

- `scripts/lab-bench.sh --model gpt-oss-20b --golden16`;
- GSM8K against vLLM-ROCm or BF16 transformers, with `gate.json`;
- one logged run with `reasoning_effort: high`;
- the `GptOssForCausalLM` mxfp4 and `openai_mxfp4` support rows;
- the perf-log section "P7 gpt-oss gate".

### 2. Task 28: MXFP4 p05 trace and the `mxfp4_a4` gate

See the plan.

### 3. Hybrids, Tasks 31–35 (Qwen3.5 and Qwen3.6)

The user ordered these last. **Ask the user before starting them.** Their kernels (GDN and
causal-conv, ABI v2.18, Task 30) are already merged.

### 4. Task 37: L0 `tq4` re-gate

This must run after Task 35, per the plan.

### 5. Task 38 (docs, contract, AGENTS.md), then Task 39 (phase exit)

Task 39 is the exit tier:

- `gate.sh --full`;
- `lab-test --tier full` and the two-GPU leg;
- `lab-bench --golden16`;
- the 10-minute soak;
- the merge to `main`.

### Open follow-ups (decisions or perf, not blockers)

- **Mixtral ep2 gate semantics.** Choose one, and ask the user:
  - an ep-only refusal mode plus placement-keyed closing;
  - a `missing_top_k` margin excuse under the multi-GPU bounds;
  - more investigation.
- **Mixtral's deferred fourth tolerance variant** (`bf16-eager-full`). Its 13/16 checkpoint sits on
  cpuhost, which is now storage-only. The 93 GB model breaks the fit rule on novanas too, so treat
  it as dropped unless the user says otherwise.
- **Quantized experts cannot capture decode graphs.** Every quantized MoE implementation carries
  `TURBINE_IMPL_NEEDS_HOST_OFFSETS`. This is a perf follow-up.
- **gpt-oss attention biases go through a per-row `turbine_add`.** The ABI has no GEMM bias. This is
  a perf item that needs an ABI decision.
- **Optional:** the instrumented LUN-stall soak (`target/p7-soak.sh` in the integration worktree)
  against the `sync=disabled` zvol. Its telemetry sampler has a quoting bug, so fix that first.

## Infrastructure: read before any lab run

- **novanas** (192.168.10.203) is the only compute host. Reach it through `ProxyJump dgx-spark`.
  - It has two R9700s, runs k3s with jobs in namespace `turbine-ci`, and has 125 GB RAM.
  - `piwi` has passwordless sudo.
  - It powers off at random. Don't debug that; just resume.
  - Its NIC `enp6s0` is now MTU 9000, persisted in `/etc/network/interfaces`. The whole 10G lab
    network runs jumbo frames.
- **`/home/piwi/turbine-ci/remote` and `/home/piwi/turbine-ci/cache/slots`** are bind mounts of an
  ext4 on `/dev/sda`. `/dev/sda` is an iSCSI LUN from the TrueNAS, zvol
  `Pool0/novanas-turbine/novanas-turbine`, now set to `sync=disabled` by the user.
- **iSCSI initiator timeouts** were raised to noop 30/90 s and abort 60 s.
- **After a novanas reboot,** the iSCSI session may not log in. Recover it like this:
  1. Stop our lab Jobs.
  2. Run `sudo iscsiadm -m node -T iqn.2005-10.org.freenas.ctl:novanas-turbine -p 192.168.10.253:3260 --login`.
  3. If needed, run `echo running | sudo tee /sys/block/sda/device/state`.
  4. If the fs went read-only, run `sudo e2fsck -fy /dev/sda` with the paths unmounted.
  5. Run `sudo mount -a`.
  6. Do a 4 KB write and read-back probe under `turbine-ci/remote`.
  7. Clean anything a Job wrote into the bare directories while the LUN was down. Use a bind view of
     `/`; files there are shadowed under the mount.
- **The TrueNAS (cpuhost, 192.168.10.253) is storage only.** The user's words were "truenas is for
  storage, period, nothing else". Run no compute, fixture jobs or references on it, ever.
- **Fixture jobs** (golden references and self-spreads) run on novanas under `fixture.lock`, at
  `nice -n 19 taskset -c 12-15` with 4 threads.
  - Never load a model that does not fit RAM with real headroom. Defer it, or use a quantized
    variant instead (user rule, 2026-10-14).
  - Give every long job a resume or checkpoint.
- **`scripts/lab-test.sh` can exit 0 without running anything** (FailedScheduling, lock queue, ssh
  drop). Check the log's `test result:` line.
- **A stale `index.lock`** under `.git/worktrees/<wt>/` has blocked merges once. If no git process
  holds it, remove it.

## Standing rules: verbatim intent from the user

- Never read, print or pass the HF token. Never run anything on dgx-spark or dgx-spark2.
- Never evict other workloads. Stop only our own runs, by run id. Never use broad `pkill`.
- Push and merge yourself; never stop for that (user amendment 2026-10-18). Put no attribution lines in commits.
- No cargo on the Mac; `cargo fmt` is the only exception. Use `scripts/gate.sh` and
  `scripts/remote-cargo.sh` on novanas.
- Never block in the foreground on long remote runs. Run them in the background.
- **Every design decision goes to the user.**
  1. Record it in `.procoder/ask/decisions.md`: a heading, options A/B/C, then the decision line.
  2. Ask with the structured question tool.
  3. Never decide alone.
- Answer questions plainly.
- Keep about five builders concurrent; six hit the rate limit.
  - One job per builder, each with a handoff file.
  - Recycle a builder at about 150k tokens.
  - Use specialist agent types: rust-engineer, cpp-pro, debugger, test-automator.
- Merge each builder's branch into `p7-stack`, then run `scripts/gate.sh --base main` from the
  integration worktree. Put the `cd` inside background commands. Rerun known load flakes alone:
  - `sigterm_drains_then_cancels`
  - `iteration_stage_breakdown_serial`
  - `json_schema_jump_forward_matches`
  - `openai_phase2_fields`
  - `pinned_copy_kernel_is_bit_exact`
  - `split_events_and_multibyte_characters`
- Don't idle while one builder runs; fill free slots with work the plan allows in parallel. When
  the user asks for status, say what is running now.
- Standing approvals:
  - Phase 7 checkpoint downloads.
  - Phase 7 lab runs: `lab-test`, `lab-serve`, `lab-bench`, `lab-cluster`, vLLM serve.
- **Formatting hook.** The procoder commit hook blocks unformatted files. Write the formatter
  output to `procoder format F > F.formatted` in its own Bash call, move it over `F`, and commit
  in a separate call.
