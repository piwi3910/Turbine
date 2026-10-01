# Handoff: p6b-eval-prefix (shared-prefix eval variant for lossy-KV gates)

Branch `p6b-eval-prefix` from `p6b-stack` fdd3ad3. User decisions 2026-10-01 (decision "6b Task 6: FP8 lower-tier proof — three open points"): 1 A
(every lossy-KV gate runs a long-shared-prefix variant of the eval), 2 A (the 1.9 x / 3.5 x / 6 x capacity targets are the codec's, held by
`kv_sim`; real runs report the measured ratio with the lossless-tail share).

## What landed

- `tests/eval/gsm8k-200-shared-prefix.jsonl`: the 200 items of `gsm8k-200.jsonl` (same ids, user message, answer, `final_number` matcher,
  `max_tokens` 512) each behind one identical `system` message: a fixed preamble ("ignore these notes") plus 2000 pseudo-words. 1.97 MB.
  - Why generated words: deterministic, license-free, offline, no copyright question; they are the bench's seeded synthetic prompt
    (`turbine_bench::prompt::prompt(20261001, 0, 2000)`). Cost: gibberish tokenizes into more tokens than prose (an estimated 3 per word, so
    about 6k tokens, about 47 blocks of `kv.block_tokens` 128) and is out of distribution; the BF16 baseline sees the same text, so the gate
    compares like with like. **The real token count is unmeasured here: read `prompt_tokens` of the first report and adjust the L0 size below.**
  - Placement: the system message is the first turn, so the chat template renders a fixed header, the fixed system text and an end-of-turn
    token, then the per-item user turn. Everything before the user turn is token-identical across records (full blocks of it are shared);
    only the partial last block of that run and the user turn are computed per item. Llama's template adds a fixed date line before the system
    text (no `date_string` is passed), so it is identical too.
- Generator: `scripts/eval/make-gsm8k-200-shared-prefix.sh` (offline, reads `gsm8k-200.jsonl`) calling `make-gsm8k-200-shared-prefix.py`, which
  ports the bench's SplitMix64 word generator. Notice paragraph in `tests/eval/NOTICE`.
- Validator: `cargo test -p turbine-bench --test golden eval_task_set_shared_prefix_valid`: every record's system message is byte-identical and
  equals the preamble plus the Rust `prompt::prompt(20261001, 0, 2000)` (so the Python port and the committed file cannot drift); items equal
  `gsm8k-200.jsonl`'s. Mutation checked: editing one record's preamble fails it naming the item.
- Runner (`turbine-golden eval`), each with tests in `benches/turbine-bench` (`eval.rs` unit tests, `tests/golden.rs` end-to-end over a mock):
  - report records per item `prompt_tokens`, `cached_tokens`, `lossy_cached_tokens` (from `usage`), and `filler_requests` / `filler_words`;
    text output prints the token shares;
  - `--filler-requests <n>` / `--filler-words <n>` (default 0 / 2000): the first item runs alone (publishes the prefix), then `n` unrelated
    one-token random-word chat requests one after another (they fill L0, so the prefix is demoted to the lower tier), then the other 199 items
    at `--concurrency`;
  - `--min-cached-ratio <f>` and `--min-lossy-cached-ratio <f>`: exit 1 (report still printed) when the share of the items' prompt tokens
    served from cache / from lossy blocks is lower, or when the server sent no usage. A gate that reused no lossy block cannot pass;
  - `eval-compare` exits 2 when baseline and candidate used different fillers (as it does for different concurrency).
- Spec S-8, its capacity lines and the lab ACs, and plan Tasks 6, 9, 13, 16, 18 name the variant and the `-sp` report suffix; `procoder spec
check` and `plan check` both COMPLETE.

## Gate-run recipe (not run: no lab runs here)

Report names: `tests/eval/<slug>/turbine-bf16-sp.json` (baseline), `turbine-l1-fp8-sp.json`, `turbine-l1-tq4-sp.json`, `turbine-l1-tq2-sp.json`,
`turbine-l0-tq4-sp.json`, `turbine-ladder-sp.json`, ... Same model, same concurrency (16, like `turbine-bf16-c16.json`), same fillers on both
sides. Everything below is provisional until the first run shows the real token counts; record the final numbers in the perf log.

1. Serve with `scripts/lab-serve.sh novanas scripts/lab/phase6-novanas-llama.yaml` and a small L0 so a handful of fillers push the cached
   prefix out: `--set kv.gpu.max_bytes=4GiB` (Llama 3B: 114,688 B per token, 14.7 MB per 128-token block, so about 290 blocks; the prefix is
   about 47 blocks, 16 running requests need about 100 more, a filler is about 47 blocks), `--set kv.cpu.max_bytes=4GiB`, and the format under
   test: baseline `--set kv.cpu.format=l0`, candidate `--set kv.cpu.format=fp8_e4m3` (or `tq4`, `tq2`). Prefix sharing stays on (default); do
   not send a cache salt (the eval sends none). Stop the serve Job by its run id afterwards.
2. For each server (baseline then candidate; GPU 0, under the bench lock like the other numbers):
   `turbine-golden eval --url http://192.168.10.203:18000 --tasks tests/eval/gsm8k-200-shared-prefix.jsonl --concurrency 16
--filler-requests 32 --filler-words 2000 --output json > tests/eval/<slug>/turbine-<name>-sp.json` (32, not 8: see "Measured by p6b-copyahead" below)
   - candidate only, add `--min-lossy-cached-ratio 0.5` (about 0.8 expected when the prefix is about 90 % of the prompt and all 199 items hit);
   - baseline only, add `--min-cached-ratio 0.5` (the exact prefix is reused from L1 or L0; it shows the baseline has the same reuse shape);
   - exit 1 from a guard means no usable report: do not commit it. If the lossy ratio is 0, the prefix was never demoted (raise
     `--filler-requests`, shrink `kv.gpu.max_bytes`) or it was dropped (`kv.cpu.max_bytes` too small for the fillers: the tier is sized in
     blocks of its own format, so size it for the fillers plus the prefix at the BF16 baseline).
3. `turbine-golden eval-compare --baseline tests/eval/<slug>/turbine-bf16-sp.json --candidate tests/eval/<slug>/turbine-l1-fp8-sp.json
--max-drop 0.01` (0.04 is not the gate for 8-bit KV); exit 0 is the gate. Take the first run's `prompt_tokens` to confirm the 47-block
   estimate and note the lossy share in the perf log.
4. L0 TurboQuant (`kv.dtype=tq4` / `tq2`, Task 13): every block is lossy from the start, so no demotion is needed and no lossy-ratio guard
   applies; run the same command (the same `--filler-requests 8`, harmless here, so the pair matches) against the `turbine-bf16-sp.json`
   baseline and `eval-compare` it.
5. Ladder (Tasks 16, 18): use the ladder config's small tiers, the same filler flags, and compare with a BF16-KV run on the same variant.

## Open / for the next builder

- Whether the first item plus fillers really demotes the prefix to L1 depends on the pressure reclaim (70 % L0 threshold) and `cost_aware`
  ranking (the older, hit-once prefix blocks versus the fillers' unreferenced blocks); the guards fail loudly if it does not. If no filler
  count works, the fallback is a server-side hook (not in this task's ownership).
- `server_cli::sigterm_drains_then_cancels` failed once in the full-workspace gate under load 14 on novanas ("bad chunk size line") and passed
  alone; unrelated to this change.

## Measured by p6b-planner2 (2026-10-01; `.procoder/handoff/p6b-planner2.md`)

- The shared system message renders to 2,944 tokens (23 blocks of 128), an item's prompt to about 3,150; a 2000-word filler is about 23 blocks
  too, so the 47-block estimate above is twice the real size.
- The runner now sends the first two items before the fillers (`FILLER_HEAD`, so the prefix has a hit). Even so, with `kv.gpu.max_bytes=4GiB`
  (292 blocks) and 8 or 16 fillers the prefix never leaves L0: capacity demotion walks leaf-first over blocks with reuse evidence only, and the
  head items' own blocks (children of the prefix, no evidence) keep it from ever being a leaf; allocation drops the fillers instead. Raising
  `--filler-requests` or shrinking L0 does not change that. The recipe needs one of the options in `p6b-planner2.md` before it can gate.

## Measured by p6b-copyahead (2026-10-01; `.procoder/handoff/p6b-copyahead.md`)

- With `--filler-requests 32` (and `kv.gpu.max_bytes=4GiB`, `kv.cpu.max_bytes=4GiB`, c16) the prefix leaves L0 before the other 198
  items arrive: allocation reclaims the head items' own blocks, then the prefix (copied ahead into L1 at GREEN since a8c3b7c) is freed
  from L0 with no further copy and promoted back once. FP8 L1: lossy cached ratio 0.927. The base commit e02e8c2 gets there too with 32
  fillers (through the YELLOW reclaim), so 8 or 16 fillers were simply too few.
- Llama gate pair committed: `tests/eval/llama-3.2-3b-instruct/turbine-bf16-sp.json` 0.775 and `turbine-l1-fp8-sp.json` 0.775,
  `eval-compare --max-drop 0.01` PASS. Use the same 32 fillers for every later `-sp` pair (`eval-compare` refuses mixed fillers), so new
  pairs compare against this baseline.
- The eval client: nothing is built on the Mac; build `turbine-golden` with `scripts/remote-cargo.sh build --release -p turbine-bench
--bin turbine-golden` and run it over ssh from the remote workspace against `http://127.0.0.1:18000`.
- 2 of 4 fp8 serve runs got 503 at filler 2–3, with L0 RED and only 5–7 of 292 blocks used (the early pressure trip under
  investigation elsewhere). Rerun such a run rather than reading it as a quality result.
