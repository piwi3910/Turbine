//! Host-side preparation of a ragged batch (P2 S-9): validation of a [`BatchInput`] against the
//! executor's limits and the pool, and packing of token ids, positions, `q_indptr`, `kv_lens`
//! and the block tables into one I32 device buffer allocated once, filled by one host-to-device
//! copy per forward (P2c: through a double-buffered host staging buffer when the memory has
//! one, so the copy never waits for the stream). Also the block fork shared by the executors
//! ([`copy_blocks`]) and the Phase 1 single-sequence KV ([`SequenceKv`]).
use std::sync::Arc;

use turbine_core::types::{BlockId, DType, KvLayout, SeqId};
use turbine_kernels::{KernelError, KernelRegistry, KvCopyConfig, KvCopyContext};
use turbine_tensor::{
    DeviceBuffer, DeviceMemory, HostStaging, KvPoolView, MemoryError, TensorView,
};

use super::{BatchInput, Logits, ModelExecutor, SeqSlice};
use crate::ModelError;

fn invalid(message: String) -> ModelError {
    ModelError::Kernel(KernelError::InvalidArgument { message })
}

const I32: usize = 4;

/// Bytes of one block within one layer: `[2, block_tokens, kv_heads, head_dim]`, or one
/// TurboQuant record per (token, KV head) (P6b S-5).
pub(crate) fn layer_block_bytes(layout: &KvLayout) -> u64 {
    layout.layer_block_bytes()
}

/// Checks that `kv` has the executor's layout and that its storage holds every layer.
pub(crate) fn check_pool(kv: &KvPoolView<'_>, layout: &KvLayout) -> Result<(), ModelError> {
    if kv.layout != *layout {
        return Err(invalid(format!(
            "KV pool layout {:?} differs from the executor's {layout:?}",
            kv.layout
        )));
    }
    // The base region sizes each layer; `num_blocks` is the whole id space (with page
    // classes, more than the base blocks).
    let base_blocks = kv.classes.as_ref().map_or(kv.num_blocks, |c| c.base_blocks);
    let per_layer = u64::from(base_blocks) * layer_block_bytes(layout);
    if kv.num_blocks == 0 || kv.layer_stride_bytes < per_layer {
        return Err(invalid(format!(
            "KV pool of {} blocks needs a layer stride of at least {per_layer} bytes, has {}",
            base_blocks, kv.layer_stride_bytes
        )));
    }
    let needed = u64::from(layout.num_layers.saturating_sub(1)) * kv.layer_stride_bytes + per_layer;
    if (kv.storage.len() as u64) < needed {
        return Err(invalid(format!(
            "KV pool storage holds {} bytes, {} layers need {needed}",
            kv.storage.len(),
            layout.num_layers
        )));
    }
    Ok(())
}

/// Layer `layer` of the pool as the dense `[num_blocks, 2, block_tokens, kv_heads, head_dim]`
/// tensor the paged attention op takes; TurboQuant pages (P6b S-5), whose records are not
/// elements, as `[num_blocks, layer_block_bytes]` bytes of their dtype. With page classes
/// (P6b S-5/S-7), the layer's whole region as U8 bytes `[layer_stride_bytes]` — a consumer
/// resolves every block id through the view's [`KvPageClasses`]. The pool must have passed
/// [`check_pool`].
pub(crate) fn kv_layer<'a>(kv: &KvPoolView<'a>, layer: usize) -> TensorView<'a> {
    let l = &kv.layout;
    if let Some(_c) = kv.classes {
        let bytes = kv.layer_stride_bytes as usize;
        return TensorView::contiguous(
            kv.storage.slice(layer * bytes, bytes),
            0,
            &[bytes],
            DType::U8,
        );
    }
    let block = layer_block_bytes(l) as usize;
    let bytes = kv.num_blocks as usize * block;
    let slice = kv
        .storage
        .slice(layer * kv.layer_stride_bytes as usize, bytes);
    if l.dtype.tq_record_bytes().is_some() {
        return TensorView::contiguous(slice, 0, &[kv.num_blocks as usize, block], l.dtype);
    }
    TensorView::contiguous(
        slice,
        0,
        &[
            kv.num_blocks as usize,
            2,
            l.block_tokens as usize,
            l.num_kv_heads as usize,
            l.head_dim as usize,
        ],
        l.dtype,
    )
}

/// Forks blocks across every layer of `kv` through the registry's `copy_blocks` op: block
/// `src[i]` is copied to `dst[i]`, pairs in order. An empty list does nothing.
pub fn copy_blocks(
    registry: &KernelRegistry,
    layout: &KvLayout,
    kv: &KvPoolView<'_>,
    src: &[BlockId],
    dst: &[BlockId],
) -> Result<(), ModelError> {
    if src.len() != dst.len() {
        return Err(invalid(format!(
            "copy_blocks: {} source blocks but {} destinations",
            src.len(),
            dst.len()
        )));
    }
    if src.is_empty() {
        return Ok(());
    }
    check_pool(kv, layout)?;
    if let Some(bad) = src.iter().chain(dst).find(|b| b.0 >= kv.num_blocks) {
        return Err(invalid(format!(
            "copy_blocks: block id {} is outside the pool of {} blocks",
            bad.0, kv.num_blocks
        )));
    }
    let pairs: Vec<(BlockId, BlockId)> = src.iter().copied().zip(dst.iter().copied()).collect();
    let cfg = copy_config(layout);
    // With page classes, a pair copies one class's page: both blocks must share the class (the
    // fork allocates the tail destination in the source's class), and the codes come from the
    // view's per-id table.
    let fmt = |b: BlockId| -> u8 {
        kv.classes
            .as_ref()
            .and_then(|c| c.class_codes.get(b.0 as usize).copied())
            .unwrap_or_else(|| kv.classes.as_ref().map_or(0, |c| c.base_code()))
    };
    let pair_fmts: Vec<u8> = src.iter().map(|&b| fmt(b)).collect();
    if kv.classes.is_some() {
        for (i, (&s, &d)) in src.iter().zip(dst.iter()).enumerate() {
            if fmt(s) != fmt(d) {
                return Err(invalid(format!(
                    "copy_blocks: pair {i} copies between page classes ({a} and {b}); a \
                     conversion is not a byte copy",
                    a = fmt(s),
                    b = fmt(d)
                )));
            }
        }
    }
    registry.kv_copy(&cfg).execute(&mut KvCopyContext {
        pool: kv.storage.whole(),
        layer_stride_bytes: kv.layer_stride_bytes,
        block_bytes: cfg.block_bytes,
        num_layers: layout.num_layers,
        pairs: &pairs,
        classes: kv.classes,
        pair_fmts: &pair_fmts,
    })?;
    Ok(())
}

/// The `copy_blocks` config of `layout`.
pub(crate) fn copy_config(layout: &KvLayout) -> KvCopyConfig {
    KvCopyConfig {
        num_layers: layout.num_layers,
        block_bytes: layer_block_bytes(layout),
    }
}

/// What the executor accepts per forward.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BatchLimits {
    pub vocab: u32,
    pub max_batch_tokens: u32,
    pub max_seqs: u32,
    /// Positions must stay below this (`max_position_embeddings`).
    pub max_positions: u32,
    pub layout: KvLayout,
}

impl BatchLimits {
    /// Block-table entries one sequence may need.
    pub fn max_blocks_per_seq(&self) -> u32 {
        self.max_positions.div_ceil(self.layout.block_tokens)
    }
}

/// Per-batch facts the forward pass needs after [`HostBatch::pack`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Packed {
    pub total_q: usize,
    pub num_seqs: usize,
    pub max_q_len: u32,
    pub max_kv_len: u32,
    /// Columns of the packed block table (the longest sequence's block count).
    pub max_blocks_per_seq: u32,
    /// Row in the token batch of each sequence's last token, in `seqs` order.
    pub last_rows: Vec<usize>,
}

impl Packed {
    /// Every sequence decodes one token (the decode-specialised paged attention applies).
    pub fn is_decode(&self) -> bool {
        self.max_q_len == 1
    }
}

/// Little-endian I32 bytes of every array a forward uploads, in upload order.
#[derive(Default)]
pub(crate) struct HostBatch {
    pub ids: Vec<u8>,
    pub positions: Vec<u8>,
    pub q_indptr: Vec<u8>,
    pub kv_lens: Vec<u8>,
    pub block_table: Vec<u8>,
    /// One `TURBINE_KVFMT_*` byte per block-table entry (v2.11); packed like the table.
    pub block_formats: Vec<u8>,
}

fn push_i32(out: &mut Vec<u8>, v: u32) {
    // Every value was checked below `max_positions` or the vocabulary, both ≤ i32::MAX.
    out.extend_from_slice(&(v as i32).to_le_bytes());
}

impl HostBatch {
    /// Validates `batch` against `limits` and packs its arrays. The block table is
    /// `[num_seqs, max_blocks_per_seq]`; entries past a sequence's needed blocks are block 0
    /// (never read).
    pub fn pack(
        &mut self,
        batch: &BatchInput<'_>,
        limits: &BatchLimits,
    ) -> Result<Packed, ModelError> {
        let t = batch.tokens.len();
        if t == 0 {
            return Err(invalid("empty batch: no tokens".into()));
        }
        if batch.positions.len() != t {
            return Err(invalid(format!(
                "{t} tokens but {} positions",
                batch.positions.len()
            )));
        }
        if t > limits.max_batch_tokens as usize {
            return Err(invalid(format!(
                "{t} tokens exceed max_batch_tokens {}",
                limits.max_batch_tokens
            )));
        }
        let n = batch.seqs.len();
        if n == 0 {
            return Err(invalid("batch has tokens but no sequences".into()));
        }
        if n > limits.max_seqs as usize {
            return Err(invalid(format!(
                "{n} sequences exceed max_seqs {}",
                limits.max_seqs
            )));
        }
        if let Some(&bad) = batch.tokens.iter().find(|&&id| id >= limits.vocab) {
            return Err(invalid(format!(
                "token id {bad} outside the vocabulary of {}",
                limits.vocab
            )));
        }
        check_pool(batch.kv, &limits.layout)?;
        let mut ids: Vec<SeqId> = batch.seqs.iter().map(|s| s.seq).collect();
        ids.sort_unstable();
        if let Some(w) = ids.windows(2).find(|w| w[0] == w[1]) {
            return Err(invalid(format!(
                "sequence {} appears twice in one batch",
                w[0].0
            )));
        }

        let bt = limits.layout.block_tokens;
        let mut next_row = 0u32;
        let mut packed = Packed {
            total_q: t,
            num_seqs: n,
            max_q_len: 0,
            max_kv_len: 0,
            max_blocks_per_seq: 0,
            last_rows: Vec::with_capacity(n),
        };
        for s in batch.seqs {
            self.check_seq(s, next_row, batch, limits)?;
            next_row += s.q_len;
            packed.max_q_len = packed.max_q_len.max(s.q_len);
            packed.max_kv_len = packed.max_kv_len.max(s.kv_len);
            packed.max_blocks_per_seq = packed.max_blocks_per_seq.max(s.kv_len.div_ceil(bt));
            packed.last_rows.push(next_row as usize - 1);
        }
        if next_row as usize != t {
            return Err(invalid(format!(
                "sequences cover {next_row} of the batch's {t} tokens"
            )));
        }

        self.ids.clear();
        self.positions.clear();
        self.q_indptr.clear();
        self.kv_lens.clear();
        self.block_table.clear();
        self.block_formats.clear();
        for (&id, &pos) in batch.tokens.iter().zip(batch.positions) {
            push_i32(&mut self.ids, id);
            push_i32(&mut self.positions, pos);
        }
        push_i32(&mut self.q_indptr, 0);
        let width = packed.max_blocks_per_seq as usize;
        for s in batch.seqs {
            push_i32(&mut self.q_indptr, s.q_start + s.q_len);
            push_i32(&mut self.kv_lens, s.kv_len);
            let needed = s.kv_len.div_ceil(bt) as usize;
            for b in &s.block_table[..needed] {
                push_i32(&mut self.block_table, b.0);
            }
            for _ in needed..width {
                push_i32(&mut self.block_table, 0);
            }
            // An empty table says every block holds the pool's base format (code of the
            // pool's dtype; BF16 = 0); padding entries are never read.
            if s.block_formats.is_empty() {
                self.block_formats
                    .resize(self.block_formats.len() + needed, 0);
            } else {
                self.block_formats
                    .extend_from_slice(&s.block_formats[..needed]);
            }
            self.block_formats
                .resize(self.block_formats.len() + (width - needed), 0);
        }
        Ok(packed)
    }

    /// Re-lays the block table `pack` wrote for `p` out `width` (≥ `p.max_blocks_per_seq`)
    /// columns wide, padding with block 0 (never read), and sets `p.max_kv_len` to
    /// `max_kv_len` (≥ every sequence's `kv_len`): decode graph mode, whose launch shape must
    /// not change while sequences grow ([`super::graphs::table_width`]). Attention reads
    /// each sequence's own `kv_len` from the device, so the result is the same.
    pub fn widen(&mut self, p: &mut Packed, width: u32, max_kv_len: u32) {
        debug_assert!(width >= p.max_blocks_per_seq && max_kv_len >= p.max_kv_len);
        let (old, new) = (p.max_blocks_per_seq as usize * I32, width as usize * I32);
        if new != old {
            let mut table = Vec::with_capacity(p.num_seqs * new);
            for s in 0..p.num_seqs {
                table.extend_from_slice(&self.block_table[s * old..(s + 1) * old]);
                table.resize((s + 1) * new, 0);
            }
            self.block_table = table;
            let (bold, bnew) = (p.max_blocks_per_seq as usize, width as usize);
            let mut formats = Vec::with_capacity(p.num_seqs * bnew);
            for s in 0..p.num_seqs {
                formats.extend_from_slice(&self.block_formats[s * bold..(s + 1) * bold]);
                formats.resize((s + 1) * bnew, 0);
            }
            self.block_formats = formats;
        }
        p.max_blocks_per_seq = width;
        p.max_kv_len = max_kv_len;
    }

    /// One sequence: it starts at `row`, its positions continue its cached prefix, and its
    /// block table covers `kv_len` tokens with blocks of the pool.
    fn check_seq(
        &self,
        s: &SeqSlice<'_>,
        row: u32,
        batch: &BatchInput<'_>,
        limits: &BatchLimits,
    ) -> Result<(), ModelError> {
        let id = s.seq.0;
        if s.q_start != row {
            return Err(invalid(format!(
                "sequence {id} starts at token {} but the previous sequence ends at {row}",
                s.q_start
            )));
        }
        if s.q_len == 0 || s.q_len > s.kv_len {
            return Err(invalid(format!(
                "sequence {id}: q_len {} must be in 1..=kv_len {}",
                s.q_len, s.kv_len
            )));
        }
        let end = u64::from(s.q_start) + u64::from(s.q_len);
        if end > batch.tokens.len() as u64 {
            return Err(invalid(format!(
                "sequence {id}: tokens {}..{end} exceed the batch's {}",
                s.q_start,
                batch.tokens.len()
            )));
        }
        if s.kv_len > limits.max_positions {
            return Err(invalid(format!(
                "sequence {id}: kv_len {} exceeds max_position_embeddings {}",
                s.kv_len, limits.max_positions
            )));
        }
        let first = s.kv_len - s.q_len;
        let rows = s.q_start as usize..end as usize;
        if let Some((i, &p)) = batch.positions[rows]
            .iter()
            .enumerate()
            .find(|&(i, &p)| p != first + i as u32)
        {
            return Err(invalid(format!(
                "sequence {id}: token {i} sits at position {p}, expected {} (positions must be \
                 consecutive from kv_len − q_len = {first})",
                first + i as u32
            )));
        }
        let needed = s.kv_len.div_ceil(limits.layout.block_tokens) as usize;
        if s.block_table.len() < needed {
            return Err(invalid(format!(
                "sequence {id}: kv_len {} needs {needed} blocks of {} tokens, block table has {}",
                s.kv_len,
                limits.layout.block_tokens,
                s.block_table.len()
            )));
        }
        if let Some(b) = s.block_table[..needed]
            .iter()
            .find(|b| b.0 >= batch.kv.num_blocks)
        {
            return Err(invalid(format!(
                "sequence {id}: block id {} is outside the pool of {} blocks",
                b.0, batch.kv.num_blocks
            )));
        }
        Ok(())
    }
}

/// Two host staging buffers used in turn, so the buffer a forward fills was last used two
/// forwards ago: with at most two forwards in flight (the executor's overlap bound) its copy
/// has completed and writing it never waits. `None` when the memory has no staging (kernel ABI
/// before v2.3): copies are then synchronous.
pub(crate) struct StagingPair {
    bufs: [HostStaging; 2],
    next: usize,
}

impl StagingPair {
    /// Two buffers of `bytes`; `Ok(None)` when `mem` has no staging.
    pub fn alloc(
        mem: &Arc<dyn DeviceMemory>,
        bytes: usize,
    ) -> Result<Option<StagingPair>, ModelError> {
        let first = match HostStaging::alloc(mem, bytes) {
            Ok(b) => b,
            Err(MemoryError::Unsupported(_)) => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        Ok(Some(StagingPair {
            bufs: [first, HostStaging::alloc(mem, bytes)?],
            next: 0,
        }))
    }

    /// The index of the buffer for the next forward, grown to at least `bytes` (twice its size or
    /// `bytes`, at most `cap`; a replaced buffer is freed once its copies complete).
    pub fn next(
        &mut self,
        mem: &Arc<dyn DeviceMemory>,
        bytes: usize,
        cap: usize,
    ) -> Result<usize, ModelError> {
        let i = self.next;
        self.next ^= 1;
        let len = self.bufs[i].len();
        if len < bytes {
            self.bufs[i] = HostStaging::alloc(mem, bytes.max(2 * len).min(cap.max(bytes)))?;
        }
        Ok(i)
    }

    pub fn buf(&self, i: usize) -> &HostStaging {
        &self.bufs[i]
    }
}

/// Device copy of the packed arrays in one I32 buffer sized once for the executor's limits.
/// A forward's arrays sit back to back from element 0: ids and positions (`t` each),
/// `q_indptr` (`n + 1`), `kv_lens` (`n`), the block table (`n × max_blocks_per_seq`).
pub(crate) struct DeviceBatch {
    buf: DeviceBuffer,
    staging: Option<StagingPair>,
    /// The packed bytes of the synchronous path.
    scratch: Vec<u8>,
    /// Byte offset of the `block_formats` table: the I32 arrays' full-length region.
    formats_at: usize,
}

impl DeviceBatch {
    /// I32 bytes of the metadata buffer: ids and positions per token; `q_indptr`, `kv_lens` and
    /// a full-length block table per sequence, plus the extra `q_indptr` entry.
    pub fn bytes(limits: &BatchLimits) -> u64 {
        let t = u64::from(limits.max_batch_tokens);
        let n = u64::from(limits.max_seqs);
        (I32 as u64) * (2 * t + 2 * n + 1 + n * u64::from(limits.max_blocks_per_seq()))
    }

    /// Bytes of the whole buffer: the I32 arrays plus the `block_formats` table (`n ×
    /// max_blocks_per_seq` bytes, v2.11), which sits behind the I32 region at
    /// [`DeviceBatch::bytes`].
    pub fn total_bytes(limits: &BatchLimits) -> u64 {
        Self::bytes(limits) + u64::from(limits.max_seqs) * u64::from(limits.max_blocks_per_seq())
    }

    /// The device buffer and, when `mem` has staging, two host staging buffers of the same size.
    pub fn alloc(
        mem: &Arc<dyn DeviceMemory>,
        limits: &BatchLimits,
    ) -> Result<DeviceBatch, ModelError> {
        let bytes = Self::bytes(limits) as usize;
        Ok(DeviceBatch {
            buf: DeviceBuffer::alloc(mem, Self::total_bytes(limits) as usize)?,
            staging: StagingPair::alloc(mem, Self::total_bytes(limits) as usize)?,
            scratch: Vec::new(),
            formats_at: bytes,
        })
    }

    /// The uploads never wait for the stream (staging present).
    pub fn is_async(&self) -> bool {
        self.staging.is_some()
    }

    /// Enqueues the one host-to-device copy of `host`: asynchronous through staging, else a
    /// synchronous copy.
    pub fn upload(&mut self, host: &HostBatch) -> Result<(), ModelError> {
        let parts = [
            &host.ids,
            &host.positions,
            &host.q_indptr,
            &host.kv_lens,
            &host.block_table,
        ];
        let mem = Arc::clone(self.buf.memory());
        let formats_at = self.formats_at;
        match self.staging.as_mut() {
            Some(pair) => {
                let i = pair.next(&mem, formats_at + host.block_formats.len(), self.buf.len())?;
                let staging = pair.buf(i);
                let mut at = 0;
                for part in parts {
                    staging.write(at, part)?;
                    at += part.len();
                }
                staging.write(formats_at, &host.block_formats)?;
                staging.upload(0, self.buf.slice(0, formats_at + host.block_formats.len()))?;
            }
            None => {
                self.scratch.clear();
                for part in parts {
                    self.scratch.extend_from_slice(part);
                }
                self.scratch.resize(formats_at, 0);
                self.scratch.extend_from_slice(&host.block_formats);
                self.buf.copy_from_host(0, &self.scratch)?;
            }
        }
        Ok(())
    }

    fn view(&self, offset: usize, shape: &[usize]) -> TensorView<'_> {
        TensorView::contiguous(self.buf.whole(), offset, shape, DType::I32)
    }

    pub fn ids_view(&self, p: &Packed) -> TensorView<'_> {
        self.view(0, &[p.total_q])
    }

    pub fn positions_view(&self, p: &Packed) -> TensorView<'_> {
        self.view(p.total_q, &[p.total_q])
    }

    pub fn q_indptr_view(&self, p: &Packed) -> TensorView<'_> {
        self.view(2 * p.total_q, &[p.num_seqs + 1])
    }

    pub fn kv_lens_view(&self, p: &Packed) -> TensorView<'_> {
        self.view(2 * p.total_q + p.num_seqs + 1, &[p.num_seqs])
    }

    pub fn block_table_view(&self, p: &Packed) -> TensorView<'_> {
        self.view(
            2 * p.total_q + 2 * p.num_seqs + 1,
            &[p.num_seqs, p.max_blocks_per_seq as usize],
        )
    }

    /// The v2.11 `block_formats` table behind the I32 region (`[num_seqs, max_blocks]` U8).
    pub fn block_formats_view(&self, p: &Packed) -> TensorView<'_> {
        TensorView::contiguous(
            self.buf.whole(),
            self.formats_at,
            &[p.num_seqs, p.max_blocks_per_seq as usize],
            DType::U8,
        )
    }
}

/// The Phase 1 single-request KV on the paged executor: a private pool of
/// `⌈max_seq_len / block_tokens⌉` blocks holding one sequence, whose block table is simply
/// `0, 1, 2, …`. [`SequenceKv::forward`] keeps the Phase 1 batch rules: consecutive positions
/// starting at or before the cached length (starting at 0 begins a new sequence), all below
/// `max_seq_len`.
pub struct SequenceKv {
    storage: DeviceBuffer,
    layout: KvLayout,
    num_blocks: u32,
    table: Vec<BlockId>,
    max_seq_len: u32,
    /// Positions `[0, cached_len)` hold valid K/V.
    cached_len: u32,
}

impl std::fmt::Debug for SequenceKv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SequenceKv")
            .field("num_blocks", &self.num_blocks)
            .field("max_seq_len", &self.max_seq_len)
            .field("cached_len", &self.cached_len)
            .finish_non_exhaustive()
    }
}

impl SequenceKv {
    /// Device bytes of the pool for `max_seq_len` tokens (the budget's KV reservation).
    pub fn bytes(layout: &KvLayout, max_seq_len: u32) -> u64 {
        u64::from(max_seq_len.div_ceil(layout.block_tokens.max(1)))
            .saturating_mul(layout.block_bytes())
    }

    /// Allocates the pool for `max_seq_len` tokens of `layout` on `mem`.
    pub fn new(
        mem: &Arc<dyn DeviceMemory>,
        layout: KvLayout,
        max_seq_len: u32,
    ) -> Result<SequenceKv, ModelError> {
        if max_seq_len == 0 || layout.block_tokens == 0 {
            return Err(invalid(format!(
                "max_seq_len {max_seq_len} and block_tokens {} must be positive",
                layout.block_tokens
            )));
        }
        let num_blocks = max_seq_len.div_ceil(layout.block_tokens);
        let storage = DeviceBuffer::alloc(mem, Self::bytes(&layout, max_seq_len) as usize)?;
        Ok(SequenceKv {
            storage,
            layout,
            num_blocks,
            table: (0..num_blocks).map(BlockId).collect(),
            max_seq_len,
            cached_len: 0,
        })
    }

    pub fn max_seq_len(&self) -> u32 {
        self.max_seq_len
    }

    /// The pool as the executor reads it.
    pub fn view(&self) -> KvPoolView<'_> {
        KvPoolView::flat(&self.storage, self.layout, self.num_blocks)
    }

    /// Runs one Phase 1 step of the sequence on `exec`: `tokens[i]` at `positions[i]`.
    pub fn forward(
        &mut self,
        exec: &mut dyn ModelExecutor,
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<Logits, ModelError> {
        let t = tokens.len();
        if t == 0 {
            return Err(invalid("empty batch: no tokens".into()));
        }
        if positions.len() != t {
            return Err(invalid(format!(
                "{t} tokens but {} positions",
                positions.len()
            )));
        }
        if let Some(i) = (1..t).find(|&i| positions[i] != positions[i - 1].wrapping_add(1)) {
            return Err(invalid(format!(
                "positions must be consecutive: {} follows {}",
                positions[i],
                positions[i - 1]
            )));
        }
        let p0 = positions[0];
        if p0 > self.cached_len {
            return Err(invalid(format!(
                "batch starts at position {p0} but only {} positions are cached",
                self.cached_len
            )));
        }
        let kv_len = u64::from(p0) + t as u64;
        if kv_len > u64::from(self.max_seq_len) {
            return Err(invalid(format!(
                "position {} is not below max_seq_len {}",
                kv_len - 1,
                self.max_seq_len
            )));
        }
        let kv_len = kv_len as u32;
        // Positions from p0 on are rewritten by this step; until it completes only the prefix
        // before p0 is valid.
        self.cached_len = p0;
        let view = self.view();
        let seqs = [SeqSlice {
            seq: SeqId(0),
            q_start: 0,
            q_len: t as u32,
            kv_len,
            block_table: &self.table,
            block_formats: &[],
            reduce: None,
        }];
        let logits = exec.forward(&BatchInput {
            tokens,
            positions,
            seqs: &seqs,
            kv: &view,
        })?;
        self.cached_len = kv_len;
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use turbine_core::types::DeviceId;
    use turbine_tensor::host::HostMemory;

    use super::*;

    fn layout() -> KvLayout {
        KvLayout {
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 4,
            dtype: DType::BF16,
            block_tokens: 4,
        }
    }

    fn limits() -> BatchLimits {
        BatchLimits {
            vocab: 100,
            max_batch_tokens: 16,
            max_seqs: 4,
            max_positions: 64,
            layout: layout(),
        }
    }

    fn i32s(bytes: &[u8]) -> Vec<i32> {
        bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn pool(mem: &Arc<dyn DeviceMemory>, blocks: u32) -> DeviceBuffer {
        let bytes =
            u64::from(layout().num_layers) * u64::from(blocks) * layer_block_bytes(&layout());
        DeviceBuffer::alloc(mem, bytes as usize).expect("pool")
    }

    fn view(storage: &DeviceBuffer, blocks: u32) -> KvPoolView<'_> {
        KvPoolView::flat(storage, layout(), blocks)
    }

    fn message(r: Result<Packed, ModelError>) -> String {
        match r {
            Err(ModelError::Kernel(KernelError::InvalidArgument { message })) => message,
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn packs_a_ragged_batch() {
        let mem = HostMemory::new(DeviceId(0), 1 << 20) as Arc<dyn DeviceMemory>;
        let storage = pool(&mem, 8);
        let kv = view(&storage, 8);
        // A 1-token decode at position 9 (10 tokens, 3 blocks), then a 5-token first prefill
        // chunk (5 tokens, 2 blocks; its table has a spare third entry).
        let a = [BlockId(5), BlockId(2), BlockId(7)];
        let b = [BlockId(0), BlockId(3), BlockId(6)];
        let seqs = [
            SeqSlice {
                seq: SeqId(10),
                q_start: 0,
                q_len: 1,
                kv_len: 10,
                block_table: &a,
                block_formats: &[],
                reduce: None,
            },
            SeqSlice {
                seq: SeqId(11),
                q_start: 1,
                q_len: 5,
                kv_len: 5,
                block_table: &b,
                block_formats: &[],
                reduce: None,
            },
        ];
        let batch = BatchInput {
            tokens: &[7, 1, 2, 3, 4, 5],
            positions: &[9, 0, 1, 2, 3, 4],
            seqs: &seqs,
            kv: &kv,
        };
        let mut host = HostBatch::default();
        let packed = host.pack(&batch, &limits()).expect("pack");
        assert_eq!(
            packed,
            Packed {
                total_q: 6,
                num_seqs: 2,
                max_q_len: 5,
                max_kv_len: 10,
                max_blocks_per_seq: 3,
                last_rows: vec![0, 5],
            }
        );
        assert!(!packed.is_decode());
        assert_eq!(i32s(&host.ids), [7, 1, 2, 3, 4, 5]);
        assert_eq!(i32s(&host.positions), [9, 0, 1, 2, 3, 4]);
        assert_eq!(i32s(&host.q_indptr), [0, 1, 6]);
        assert_eq!(i32s(&host.kv_lens), [10, 5]);
        // Sequence 11 needs 2 blocks; its third column is padding (block 0).
        assert_eq!(i32s(&host.block_table), [5, 2, 7, 0, 3, 0]);

        // One upload, through staging (host memory has it) or synchronously (it has not): the
        // views find every array where the packing put it.
        let bare: Arc<dyn DeviceMemory> = Arc::new(NoStaging(mem.clone()));
        for m in [&mem, &bare] {
            let mut dev = DeviceBatch::alloc(m, &limits()).expect("alloc");
            assert_eq!(dev.is_async(), Arc::ptr_eq(m, &mem));
            for _ in 0..3 {
                dev.upload(&host).expect("upload");
            }
            let read = |v: TensorView<'_>| i32s(&v.slice.read_bytes().expect("read"));
            assert_eq!(read(dev.ids_view(&packed)), [7, 1, 2, 3, 4, 5]);
            assert_eq!(read(dev.positions_view(&packed)), [9, 0, 1, 2, 3, 4]);
            assert_eq!(read(dev.q_indptr_view(&packed)), [0, 1, 6]);
            assert_eq!(read(dev.kv_lens_view(&packed)), [10, 5]);
            let table = dev.block_table_view(&packed);
            assert_eq!(table.shape.as_slice(), [2, 3]);
            assert_eq!(read(table), [5, 2, 7, 0, 3, 0]);
        }

        // Decode graph mode widens the table to 4 columns and the attention bound to 16.
        let mut wide = packed.clone();
        host.widen(&mut wide, 4, 16);
        assert_eq!((wide.max_blocks_per_seq, wide.max_kv_len), (4, 16));
        assert_eq!(i32s(&host.block_table), [5, 2, 7, 0, 0, 3, 0, 0]);
        assert_eq!(
            i32s(&host.kv_lens),
            [10, 5],
            "only the table and bounds change"
        );
    }

    /// `HostMemory` without its staging: the synchronous path of a kernel library before ABI
    /// v2.3.
    struct NoStaging(Arc<dyn DeviceMemory>);

    impl DeviceMemory for NoStaging {
        fn device(&self) -> DeviceId {
            self.0.device()
        }
        fn alloc(&self, bytes: usize) -> Result<turbine_tensor::DevicePtr, MemoryError> {
            self.0.alloc(bytes)
        }
        fn free(&self, ptr: turbine_tensor::DevicePtr) {
            self.0.free(ptr);
        }
        fn copy_h2d(&self, dst: turbine_tensor::DevicePtr, src: &[u8]) -> Result<(), MemoryError> {
            self.0.copy_h2d(dst, src)
        }
        fn copy_d2h(
            &self,
            dst: &mut [u8],
            src: turbine_tensor::DevicePtr,
        ) -> Result<(), MemoryError> {
            self.0.copy_d2h(dst, src)
        }
        fn copy_d2d(
            &self,
            dst: turbine_tensor::DevicePtr,
            src: turbine_tensor::DevicePtr,
            bytes: usize,
        ) -> Result<(), MemoryError> {
            self.0.copy_d2d(dst, src, bytes)
        }
        fn synchronize(&self) -> Result<(), MemoryError> {
            self.0.synchronize()
        }
        fn mem_info(&self) -> Result<turbine_tensor::MemInfo, MemoryError> {
            self.0.mem_info()
        }
        fn compute_stream(&self) -> turbine_tensor::StreamRef {
            self.0.compute_stream()
        }
    }

    #[test]
    fn rejects_malformed_batches() {
        let mem = HostMemory::new(DeviceId(0), 1 << 20) as Arc<dyn DeviceMemory>;
        let storage = pool(&mem, 4);
        let kv = view(&storage, 4);
        let table = [BlockId(0), BlockId(1)];
        #[allow(clippy::too_many_arguments)]
        fn seq<'a>(
            seq: u64,
            q_start: u32,
            q_len: u32,
            kv_len: u32,
            block_table: &'a [BlockId],
            block_formats: &'a [u8],
        ) -> SeqSlice<'a> {
            SeqSlice {
                seq: SeqId(seq),
                q_start,
                q_len,
                kv_len,
                block_table,
                block_formats,
                reduce: None,
            }
        }
        let mut host = HostBatch::default();
        let mut pack = |tokens: &[u32], positions: &[u32], seqs: &[SeqSlice<'_>]| {
            message(host.pack(
                &BatchInput {
                    tokens,
                    positions,
                    seqs,
                    kv: &kv,
                },
                &limits(),
            ))
        };
        let one = [seq(1, 0, 2, 2, &table, &[])];
        assert!(pack(&[], &[], &one).contains("empty"));
        assert!(pack(&[1, 2], &[0], &one).contains("positions"));
        assert!(pack(&[1, 2], &[0, 1], &[]).contains("no sequences"));
        assert!(pack(&[1, 100], &[0, 1], &one).contains("vocabulary"));
        assert!(pack(&[1, 2], &[0, 2], &one).contains("position 2, expected 1"));
        let short = [seq(1, 0, 1, 1, &table, &[])];
        assert!(pack(&[1, 2], &[0, 1], &short).contains("cover 1 of the batch's 2"));
        let twice = [seq(1, 0, 1, 1, &table, &[]), seq(1, 1, 1, 1, &table, &[])];
        assert!(pack(&[1, 2], &[0, 0], &twice).contains("twice"));
        let gap = [seq(1, 0, 1, 1, &table, &[]), seq(2, 2, 1, 1, &table, &[])];
        assert!(pack(&[1, 2, 3], &[0, 0, 0], &gap).contains("starts at token 2"));
        let zero = [seq(1, 0, 0, 1, &table, &[])];
        assert!(pack(&[1], &[0], &zero).contains("q_len 0"));
        // 9 tokens need 3 blocks of 4; the table has 2.
        let long = [seq(1, 0, 1, 9, &table, &[])];
        assert!(pack(&[1], &[8], &long).contains("needs 3 blocks"));
        let outside = [BlockId(4)];
        assert!(pack(&[1], &[0], &[seq(1, 0, 1, 1, &outside, &[])]).contains("outside the pool"));
        assert!(pack(&[1], &[64], &[seq(1, 0, 1, 65, &table, &[])]).contains("max_position"));
        let many: Vec<u32> = vec![1; 17];
        assert!(pack(&many, &many, &one).contains("max_batch_tokens"));

        // A pool whose layout differs from the executor's is refused.
        let other = KvPoolView {
            layout: KvLayout {
                block_tokens: 8,
                ..layout()
            },
            ..kv
        };
        let err = message(host.pack(
            &BatchInput {
                tokens: &[1],
                positions: &[0],
                seqs: &[seq(1, 0, 1, 1, &table, &[])],
                kv: &other,
            },
            &limits(),
        ));
        assert!(err.contains("layout"), "{err}");
    }
}
