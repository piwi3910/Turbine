# Handoff: p6b-window (the BF16 recent window works on the GPU)

Branch `p6b-window`, from `p6b-stack` 7a67569. Closes Task 17's open item (user decision
"6b Task 17: the BF16 recent window needs per-class block addressing", **A**): the executor
resolves every block id through its page class, so the window serves and its default is 1.

## The ABI delta (v2.11, purely additive — no minor bump, existing descriptors unchanged)

- Header (`kernels/include/turbine_kernels.h`): new `typedef struct turbine_kv_page_class
  { int32_t fmt; int32_t per_layer_bytes }`; `turbine_attention_paged_desc` and
  `turbine_copy_blocks_desc` each gain trailing `const turbine_kv_page_class *page_classes`
  plus `int32_t num_page_classes, base_blocks, slab_stride, slab_base_blocks` (the copy
  descriptor also `const uint8_t *pair_formats`, one `TURBINE_KVFMT_*` per pair — the class of
  BOTH blocks; a conversion is not a byte copy). All read only by a library of minor ≥ 11.
- Addressing: `page_classes` NULL = the flat layout, byte-identical to before. Set:
  `num_blocks` is the whole id space, `kv_layer` stays the layer's
  `base_blocks × base_page_bytes` region, and block `b` sits at `b × base_page_bytes` below
  `base_blocks`, else at `(b−base)/slab_stride × slab_base_blocks × base_page_bytes +
  (b−base)%slab_stride × page_classes[c].per_layer_bytes`, `c` the class whose `fmt` equals
  the block's `block_formats` byte (copy: `pair_formats[i]`). The scalars are pool constants,
  so a decode graph bakes them; `page_classes` is host memory read during the call only (the
  HIP kernels take the values by value as a `Pages` struct, `pages_of(d)`).

## What landed (one commit, `feat(model): per-class block addressing; the BF16 recent window works on GPU`)

- `turbine-tensor`: `KvPageClass` / `KvPageClasses` (the resolution, plus per-id
  `class_codes`) next to `KvPoolView`; the view gains `classes: Option<KvPageClasses>` and
  `KvPoolView::flat` (`num_blocks` is now the whole id space; `layer_stride_bytes` stays one
  base region). Unit tests on `page_offset`.
- `turbine-kv/pool.rs`: the pool keeps `class_codes` per id (updated as slabs are carved and
  returned) and `view()` reports the classes. **Bug fix found by the tiny-server test**:
  `class_code` mapped `"bf16"` to the base code, so on a tq4-based pool the window's BF16
  class was tagged `TQ4` — it is now code 0 (`KV_FMT_BF16`) whatever the base; only `"l0"`
  follows `dtype`. `BlockPool::fork` allocates a forked tail in the tail's own class (the
  executor refuses a `copy_blocks` pair across classes).
- `turbine-kernels`: `PagedAttentionContext.classes`, `KvCopyContext.{classes, pair_fmts}`;
  the CPU paged attention routes a classed pool through `mixed()` with per-class page
  slices (`kv_layer` of a classed pool is the layer's region as U8, and the classes are
  checked against the codecs' page bytes); the CPU `copy_blocks` copies a pair's class page at
  the class offsets; the shim packs the new fields (and requires minor ≥ 11 for a classed
  call); the HIP mixed kernels resolve `Pages` in `where_of` (flat pools keep the old
  fit-in-slot skip), `paged_attention.cpp`'s `mixed()` routes classed descriptors to the mixed
  implementations, `copy_blocks.cpp` resolves per-class offsets and bounds every page against
  the layer region. ABI-header tests updated (`header_declares_the_v211_per_class_page_fields`).
- `turbine-model`: `check_pool` sizes the base region by `base_blocks`; `kv_layer` of a
  classed pool is the byte region; `copy_blocks` derives the pair classes from the view's
  `class_codes` and refuses cross-class pairs; the decoder passes `kv.classes` into every
  paged attention.
- Window default: `kv.recent_window_blocks` default **1** (config, spec S-5 and the config
  table amended; the contract §26 already said 1). New tests: cpu
  `tiny_server::recent_window_serves_on_cpu` (the window grows the BF16 class, serves a
  300-token completion, and a second request's ticks drive the exit conversions — the metric
  is `turbine_kv_ladder_actions_total{tier="l0",from="bf16",to=…,reason="recent_window"}`),
  cpu kernels `classed_pages_append_and_read_at_their_class_offsets` (mutation-checked: pool
  bytes at the class offsets, the flat offset asserted untouched, classes checked against the
  codec) and `classed_copy_moves_a_class_page`, lab `hip_ops::paged_mixed_classed_matches_cpu`
  (tq4 base + two BF16 class pages over a 24-token prefill with a decode row riding along,
  HIP vs CPU).

## Results

- Host: turbine-tensor/-kv/-kernels lib, kv_sim 11/11, turbine-core, turbine-model,
  turbine-server bin (engine) and tiny_server (tq + window) all green; `gate.sh` ok (see the
  commit).
- Lab: see `.procoder/perf-log.md` 6b "The BF16 recent window on the GPU" — the
  `paged_mixed_classed_matches_cpu` run and `lab-bench --quick --model llama --
  --set kv.dtype=tq4` golden c1 (Task 13's tq4 golden was 0/16; the window should improve it).

## Notes for the next builder

- Task 18 owns golden c16 under tq4+window and the eval gate; this task's golden is c1 only.
- `kv.ladder.l0`'s startup refusal (Task 16) still stands; with the addressing in place the
  L0 ladder rungs would now serve — revisit as its own decision.
- The prefill staging path on the GPU decodes class pages through the same `where_of`
  (staging gather), so a BF16 window block stages like any mixed block; not separately
  benchmarked.
- `turbine-kv`'s `class_codes` are authoritative for `copy_blocks`; a block's code follows its
  slab's current owner, so codes change when a slab returns to base — never hold a code
  across a pool mutation.
