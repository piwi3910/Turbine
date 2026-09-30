//! Weight loader (P1 S-2): maps checkpoint tensor names to the executor's parameter slots,
//! validates every slot (present, stored in the weight format, expected shape) before
//! allocating anything, then uploads
//! each tensor to the device through one bounded, reused host staging buffer filled with
//! positioned reads (`read_exact_at`; no `mmap`, so this crate stays free of `unsafe`).
//! Checkpoint tensors no slot names are reported by name (`unexpected`), and a tied model's
//! shipped `lm_head.weight` is `ignored`. A slot may name a place in a stacked parameter
//! ([`StackPlace`]): its bytes are uploaded straight into that parameter at the place's offset,
//! so stacking (OLMoE's per-expert weights) and row concatenation (the fused Q/K/V and gate/up
//! projections, Phase 2c) need no device-to-device copy, which the HIP shim lacks before kernel
//! ABI v3.
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::Arc;

use turbine_kernels::KernelError;
use turbine_tensor::{DeviceMemory, Tensor};

use crate::ModelError;
use crate::safetensors::{SafetensorsIndex, TensorEntry, io_err, open_regular};
use crate::weights::{Bf16, WeightFormat};

/// Upper bound of the host staging buffer (P1 bounded resources).
pub const MAX_STAGING_BYTES: usize = 256 << 20;
/// The untied output projection; ignored when the model ties it to the embedding.
pub const LM_HEAD: &str = "lm_head.weight";

/// One executor parameter: checkpoint tensor name and expected shape (row-major, elements),
/// and, for a slice of a stacked parameter, where it lands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightSlot {
    pub name: String,
    pub shape: Vec<usize>,
    /// `Some`: the tensor is a contiguous part of the stacked parameter `stack.name` and is
    /// loaded only as part of it (`LoadedWeights` holds the stack, not the slot's own name).
    pub stack: Option<StackPlace>,
    /// `Some`: the slot is one tensor-parallel rank's shard of a larger checkpoint tensor
    /// ([`crate::tp::weight_slots`]); only its bytes are read. `None`: the whole tensor, of
    /// `shape`.
    pub source: Option<SlotSource>,
}

/// The part of a checkpoint tensor of `shape` a sharded [`WeightSlot`] loads: elements
/// `[start, start + len)` along `axis` (0: whole rows, one contiguous byte range; 1: a column
/// block of a 2-D tensor, one positioned read per row), every other dimension whole. The slot's
/// own `shape` is the checkpoint's with `len` along `axis`, except that along axis 0 it may be
/// longer: the rows past `len` are zeros (the last vocabulary shard's padding).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotSource {
    pub shape: Vec<usize>,
    pub axis: usize,
    pub start: usize,
    pub len: usize,
}

impl SlotSource {
    /// Whether `slot_shape` can hold this part (see [`SlotSource`]).
    fn fits(&self, slot_shape: &[usize]) -> bool {
        let dims = self.shape.len();
        (self.axis == 0 && dims >= 1 || self.axis == 1 && dims == 2)
            && slot_shape.len() == dims
            && self.start + self.len <= self.shape[self.axis]
            && (0..dims).all(|d| {
                if d != self.axis {
                    slot_shape[d] == self.shape[d]
                } else if d == 0 {
                    slot_shape[d] >= self.len
                } else {
                    slot_shape[d] == self.len
                }
            })
    }

    /// The `(file offset, bytes)` runs to read, relative to the tensor's first byte, in
    /// destination order.
    fn runs(&self, es: usize) -> Vec<(u64, usize)> {
        let inner: usize = self.shape[1..].iter().product();
        match self.axis {
            0 => vec![((self.start * inner * es) as u64, self.len * inner * es)],
            _ => (0..self.shape[0])
                .map(|r| (((r * inner + self.start) * es) as u64, self.len * es))
                .collect(),
        }
    }
}

/// Elements `[offset, offset + slot numel)` of the stacked parameter `name` of `shape`: an entry
/// along its first axis (OLMoE's `[experts, rows, cols]` expert stacks) or a block of rows (the
/// fused `[q; k; v]` and `[gate; up]` projections). The slots naming one stack must tile it
/// exactly (every element covered once), and each slot's trailing dimensions must be the stack's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StackPlace {
    pub name: String,
    pub shape: Vec<usize>,
    /// First element of the slot in the stack (row-major).
    pub offset: usize,
}

/// The fused `[q_dim + 2·kv_dim, hidden]` Q/K/V projection of layer `layer` (rows Q, then K,
/// then V), as [`crate::families::llama_slots`] and
/// [`crate::families::olmoe_slots`] load it.
pub fn qkv_proj_name(layer: u32) -> String {
    format!("model.layers.{layer}.self_attn.qkv_proj.weight")
}

/// The fused `[2·intermediate, hidden]` gate/up projection of Llama layer `layer` (gate rows,
/// then up rows), as [`crate::families::llama_slots`] loads it.
pub fn gate_up_proj_name(layer: u32) -> String {
    format!("model.layers.{layer}.mlp.gate_up_proj.weight")
}

/// Slots for the `[rows, cols]` checkpoint tensors `parts` concatenated along their rows, in
/// order, into the stacked parameter `stack` of `[Σ rows, cols]`.
pub(crate) fn row_concat(stack: String, parts: &[(String, usize)], cols: usize) -> Vec<WeightSlot> {
    let total: usize = parts.iter().map(|(_, rows)| rows).sum();
    let mut offset = 0;
    parts
        .iter()
        .map(|(name, rows)| {
            let slot = WeightSlot {
                name: name.clone(),
                shape: vec![*rows, cols],
                stack: Some(StackPlace {
                    name: stack.clone(),
                    shape: vec![total, cols],
                    offset,
                }),
                source: None,
            };
            offset += rows * cols;
            slot
        })
        .collect()
}

/// Layer `i`'s Q, K and V projections as rows of its fused [`qkv_proj_name`] parameter.
pub(crate) fn qkv_slots(i: u32, q: usize, kv: usize, hidden: usize) -> Vec<WeightSlot> {
    let p = format!("model.layers.{i}.self_attn");
    row_concat(
        qkv_proj_name(i),
        &[
            (format!("{p}.q_proj.weight"), q),
            (format!("{p}.k_proj.weight"), kv),
            (format!("{p}.v_proj.weight"), kv),
        ],
        hidden,
    )
}

/// The stacked `[experts, rows, cols]` parameter of layer `layer`'s expert projection `proj`
/// (`gate_proj`, `up_proj`, `down_proj`), as [`crate::families::olmoe_slots`] loads it.
pub fn stacked_experts_name(layer: u32, proj: &str) -> String {
    format!("model.layers.{layer}.mlp.experts.{proj}.weight")
}

/// The loaded parameters by checkpoint name (a stacked parameter by its [`StackPlace`] name),
/// plus what the checkpoint held beyond them.
#[derive(Debug)]
pub struct LoadedWeights {
    pub tensors: HashMap<String, Tensor>,
    /// Bytes uploaded to the device.
    pub weight_bytes: u64,
    /// Checkpoint tensors no slot names (logged `event="unexpected_tensor"`), sorted.
    pub unexpected: Vec<String>,
    /// Checkpoint tensors deliberately skipped: `lm_head.weight` of a tied model.
    pub ignored: Vec<String>,
    /// Checkpoint tensors another pipeline stage or expert-parallel rank loads
    /// ([`WeightLoader::load_part`]): counted, logged at debug level only.
    pub elsewhere: usize,
}

impl LoadedWeights {
    /// Removes and returns one parameter.
    pub fn take(&mut self, name: &str) -> Result<Tensor, ModelError> {
        self.tensors
            .remove(name)
            .ok_or_else(|| ModelError::MissingTensor(name.to_string()))
    }
}

/// Uploads checkpoint tensors into device tensors.
#[derive(Debug)]
pub struct WeightLoader;

impl WeightLoader {
    /// [`WeightLoader::load_format`] of a BF16 checkpoint.
    pub fn load(
        index: &SafetensorsIndex,
        slots: &[WeightSlot],
        mem: &Arc<dyn DeviceMemory>,
        staging_bytes: usize,
    ) -> Result<LoadedWeights, ModelError> {
        WeightLoader::load_format(&Bf16, index, slots, mem, staging_bytes)
    }

    /// Validates every slot against `index` (present, stored in `format`
    /// ([`WeightFormat::check_tensor`]), expected shape) and every stack
    /// (each index named once, slot shapes matching), then allocates each tensor on `mem` and
    /// uploads it through one host staging buffer of `min(staging_bytes, MAX_STAGING_BYTES)`
    /// bytes (at least 1, at most the largest tensor); a stacked slot's bytes go straight to its
    /// offset in the stacked tensor (host-to-device copies only). Nothing is allocated when
    /// validation fails. The staging buffer is reused only after the previous copy completed
    /// (`synchronize`), since device copies are enqueued. Parameters are allocated as
    /// [`WeightFormat::slot_dtype`].
    pub fn load_format(
        format: &dyn WeightFormat,
        index: &SafetensorsIndex,
        slots: &[WeightSlot],
        mem: &Arc<dyn DeviceMemory>,
        staging_bytes: usize,
    ) -> Result<LoadedWeights, ModelError> {
        WeightLoader::load_part(format, index, slots, &[], mem, staging_bytes)
    }

    /// [`WeightLoader::load_format`] of one part of a model split by pipeline stages or expert
    /// ranks: `whole` is the unsplit model's slot list (the family's
    /// `weight_slots`); a checkpoint tensor it names but `slots` does not belongs to another
    /// stage or rank and is skipped quietly (counted in [`LoadedWeights::elsewhere`], one
    /// `debug` line) instead of warning `unexpected_tensor`. Tensors neither list names still
    /// warn. An empty `whole` is [`WeightLoader::load_format`].
    ///
    /// Both lists are the family's slots: each is expanded by [`WeightFormat::slots`] (a
    /// quantized layer's scales, zero points and activation scales). A slot the format
    /// [`WeightFormat::repacks`] is read whole, rewritten by [`WeightFormat::repack`] and
    /// uploaded (its checkpoint tensor's shape is the format's to check), or only validated
    /// when it has no elements (a check-only slot); every other slot is copied as stored.
    pub fn load_part(
        format: &dyn WeightFormat,
        index: &SafetensorsIndex,
        slots: &[WeightSlot],
        whole: &[WeightSlot],
        mem: &Arc<dyn DeviceMemory>,
        staging_bytes: usize,
    ) -> Result<LoadedWeights, ModelError> {
        for base in slots.iter().filter(|s| s.source.is_some()) {
            format.check_shard(base)?;
        }
        let slots: Vec<WeightSlot> = slots.iter().flat_map(|s| format.slots(s)).collect();
        let whole: Vec<WeightSlot> = whole.iter().flat_map(|s| format.slots(s)).collect();
        let (slots, whole) = (slots.as_slice(), whole.as_slice());
        let mut planned: Vec<(&WeightSlot, &TensorEntry)> = Vec::with_capacity(slots.len());
        for slot in slots {
            let empty = slot.shape.iter().product::<usize>() == 0;
            let entry = match index.get(&slot.name) {
                Some(entry) => entry,
                None if empty && format.optional(slot) => continue,
                None => return Err(ModelError::MissingTensor(slot.name.clone())),
            };
            format.check_tensor(entry)?;
            if format.repacks(slot) {
                for name in format.companions(slot) {
                    if index.get(&name).is_none() {
                        return Err(ModelError::MissingTensor(name));
                    }
                }
                planned.push((slot, entry));
                continue;
            }
            let expected = slot.source.as_ref().map_or(&slot.shape, |src| &src.shape);
            if entry.shape != *expected {
                return Err(ModelError::Safetensors {
                    file: entry.file.clone(),
                    tensor: entry.name.clone(),
                    rule: format!("shape {:?} != expected {:?}", entry.shape, expected),
                });
            }
            // Copied as stored: its bytes must be exactly the slot's, else a tensor of the
            // right shape in a wider dtype would spill into the next slot of a stack (review
            // r13 C3).
            let dtype = format.slot_dtype(slot);
            let want = expected.iter().product::<usize>() as u64 * dtype.size_bytes() as u64;
            if entry.byte_len() != want {
                return Err(ModelError::Safetensors {
                    file: entry.file.clone(),
                    tensor: entry.name.clone(),
                    rule: format!(
                        "{} {} bytes for a {} slot of shape {:?} ({want} bytes)",
                        entry.dtype,
                        entry.byte_len(),
                        dtype.as_str(),
                        expected
                    ),
                });
            }
            if let Some(src) = &slot.source
                && !src.fits(&slot.shape)
            {
                return Err(ModelError::Safetensors {
                    file: entry.file.clone(),
                    tensor: entry.name.clone(),
                    rule: format!("shard {src:?} does not fit slot shape {:?}", slot.shape),
                });
            }
            planned.push((slot, entry));
        }
        check_stacks(slots)?;

        let wanted: HashSet<&str> = slots.iter().map(|s| s.name.as_str()).collect();
        let others: HashSet<&str> = whole
            .iter()
            .map(|s| s.name.as_str())
            .filter(|n| !wanted.contains(n))
            .collect();
        let (mut unexpected, mut ignored, mut elsewhere) = (Vec::new(), Vec::new(), 0usize);
        for entry in index.entries() {
            if wanted.contains(entry.name.as_str()) {
                continue;
            }
            if others.contains(entry.name.as_str()) {
                elsewhere += 1;
            } else if entry.name == LM_HEAD {
                tracing::warn!(
                    event = "ignored_tensor",
                    tensor = %entry.name,
                    file = %entry.file.display(),
                    "tied model: checkpoint lm_head.weight is ignored"
                );
                ignored.push(entry.name.clone());
            } else {
                tracing::warn!(
                    event = "unexpected_tensor",
                    tensor = %entry.name,
                    file = %entry.file.display(),
                    "checkpoint tensor has no parameter slot"
                );
                unexpected.push(entry.name.clone());
            }
        }
        if elsewhere > 0 {
            tracing::debug!(
                event = "tensors_elsewhere",
                tensors = elsewhere,
                "checkpoint tensors of other pipeline stages or expert ranks are not read here"
            );
        }

        let largest = planned
            .iter()
            .filter(|(slot, _)| !format.repacks(slot))
            .map(|(slot, e)| match slot.source {
                None => e.byte_len(),
                Some(_) => {
                    (slot.shape.iter().product::<usize>() * format.slot_dtype(slot).size_bytes())
                        as u64
                }
            })
            .max()
            .unwrap_or(0);
        let staging_len = staging_bytes
            .min(MAX_STAGING_BYTES)
            .min(usize::try_from(largest).unwrap_or(usize::MAX))
            .max(1);
        let mut staging = vec![0u8; staging_len];
        let mut files: HashMap<PathBuf, File> = HashMap::new();
        let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(planned.len());
        let mut weight_bytes = 0u64;
        for (slot, entry) in planned {
            let dtype = format.slot_dtype(slot);
            // The destination tensor and this slot's byte offset in it.
            let (key, shape, base) = match &slot.stack {
                Some(place) => (&place.name, &place.shape, place.offset * dtype.size_bytes()),
                None => (&slot.name, &slot.shape, 0),
            };
            let file = match files.get(&entry.file) {
                Some(f) => f,
                None => {
                    let f = open_regular(&entry.file)?;
                    files.entry(entry.file.clone()).or_insert(f)
                }
            };
            // A check-only slot (no elements): the format validates the tensor (e.g. GPTQ's
            // `g_idx`), nothing is stored.
            if format.repacks(slot) && slot.shape.iter().product::<usize>() == 0 {
                let mut raw = vec![0u8; entry.byte_len() as usize];
                file.read_exact_at(&mut raw, entry.range.start)
                    .map_err(|e| io_err(&entry.file, e))?;
                let companions = read_companions(format, slot, index, &mut files)?;
                let companions: Vec<(&TensorEntry, Vec<u8>)> =
                    companions.iter().map(|(e, b)| (*e, b.clone())).collect();
                if !format
                    .repack_with(slot, entry, raw, &companions)?
                    .is_empty()
                {
                    return Err(ModelError::Safetensors {
                        file: entry.file.clone(),
                        tensor: entry.name.clone(),
                        rule: "a check-only slot repacked to bytes".to_string(),
                    });
                }
                continue;
            }
            if !tensors.contains_key(key) {
                tensors.insert(key.clone(), Tensor::empty(mem, shape, dtype)?);
            }
            if format.repacks(slot) {
                let companions = read_companions(format, slot, index, &mut files)?;
                let companions: Vec<(&TensorEntry, Vec<u8>)> =
                    companions.iter().map(|(e, b)| (*e, b.clone())).collect();
                let file = files.get(&entry.file).expect("opened above");
                let mut raw = vec![0u8; entry.byte_len() as usize];
                file.read_exact_at(&mut raw, entry.range.start)
                    .map_err(|e| io_err(&entry.file, e))?;
                // A sharded slot: the whole tensor is repacked into the loaded layout (the
                // source's shape), then the rank's part is taken from it.
                let bytes = match &slot.source {
                    None => format.repack_with(slot, entry, raw, &companions)?,
                    Some(src) => {
                        let whole = WeightSlot {
                            shape: src.shape.clone(),
                            source: None,
                            ..slot.clone()
                        };
                        let all = format.repack_with(&whole, entry, raw, &companions)?;
                        let mut part = take_part(&all, src, dtype.size_bytes());
                        // The last vocabulary shard's padding rows are zeros.
                        let want = slot.shape.iter().product::<usize>() * dtype.size_bytes();
                        if src.axis == 0 && part.len() < want {
                            part.resize(want, 0);
                        }
                        part
                    }
                };
                let want = slot.shape.iter().product::<usize>() * dtype.size_bytes();
                if bytes.len() != want {
                    return Err(ModelError::Safetensors {
                        file: entry.file.clone(),
                        tensor: entry.name.clone(),
                        rule: format!(
                            "repacked to {} bytes, slot {:?} {} needs {want}",
                            bytes.len(),
                            slot.shape,
                            dtype.as_str()
                        ),
                    });
                }
                let tensor = tensors.get_mut(key).expect("inserted above");
                tensor.storage.copy_from_host(base, &bytes)?;
                mem.synchronize()?;
                weight_bytes += bytes.len() as u64;
                continue;
            }
            let tensor = tensors.get_mut(key).expect("inserted above");
            if let Some(src) = &slot.source {
                weight_bytes += upload_shard(
                    file,
                    entry,
                    src,
                    &mut staging,
                    tensor,
                    base,
                    (slot.shape.iter().product::<usize>() * dtype.size_bytes()) as u64,
                    mem.as_ref(),
                )?;
                continue;
            }
            let mut done = 0u64;
            while done < entry.byte_len() {
                let n = (entry.byte_len() - done).min(staging_len as u64) as usize;
                let chunk = &mut staging[..n];
                file.read_exact_at(chunk, entry.range.start + done)
                    .map_err(|e| io_err(&entry.file, e))?;
                tensor.storage.copy_from_host(base + done as usize, chunk)?;
                mem.synchronize()?;
                done += n as u64;
            }
            weight_bytes += entry.byte_len();
        }
        tracing::debug!(
            event = "weights_loaded",
            tensors = tensors.len(),
            weight_bytes,
            staging_bytes = staging_len,
        );
        Ok(LoadedWeights {
            tensors,
            weight_bytes,
            unexpected,
            ignored,
            elsewhere,
        })
    }
}

/// The companion tensors of `slot` ([`WeightFormat::companions`]), each read whole.
fn read_companions<'i>(
    format: &dyn WeightFormat,
    slot: &WeightSlot,
    index: &'i SafetensorsIndex,
    files: &mut HashMap<PathBuf, File>,
) -> Result<Vec<(&'i TensorEntry, Vec<u8>)>, ModelError> {
    format
        .companions(slot)
        .iter()
        .map(|name| {
            let entry = index
                .get(name)
                .ok_or_else(|| ModelError::MissingTensor(name.clone()))?;
            if !files.contains_key(&entry.file) {
                let f = open_regular(&entry.file)?;
                files.insert(entry.file.clone(), f);
            }
            let file = &files[&entry.file];
            let mut raw = vec![0u8; entry.byte_len() as usize];
            file.read_exact_at(&mut raw, entry.range.start)
                .map_err(|e| io_err(&entry.file, e))?;
            Ok((entry, raw))
        })
        .collect()
}

/// The part `src` of a whole row-major tensor `all` of `src.shape` (elements of `es` bytes):
/// rows `start..start + len` along axis 0, or that column block of every row along axis 1.
fn take_part(all: &[u8], src: &SlotSource, es: usize) -> Vec<u8> {
    let row: usize = src.shape[1..].iter().product::<usize>() * es;
    match src.axis {
        0 => all[src.start * row..(src.start + src.len) * row].to_vec(),
        _ => all
            .chunks_exact(row)
            .flat_map(|r| &r[src.start * es..(src.start + src.len) * es])
            .copied()
            .collect(),
    }
}

/// Uploads the part `src` of checkpoint tensor `entry` to `tensor` at byte `base`, then zeros up
/// to `dest_bytes` (the padding rows), through `staging`: the runs of [`SlotSource::runs`] are
/// read with positioned reads into consecutive staging bytes, each full staging buffer uploaded
/// once (so a column block of many short rows is one upload per buffer, not per row). Returns
/// the checkpoint bytes read.
#[allow(clippy::too_many_arguments)]
fn upload_shard(
    file: &File,
    entry: &TensorEntry,
    src: &SlotSource,
    staging: &mut [u8],
    tensor: &mut Tensor,
    base: usize,
    dest_bytes: u64,
    mem: &dyn DeviceMemory,
) -> Result<u64, ModelError> {
    let numel = entry.shape.iter().product::<usize>().max(1) as u64;
    let es = (entry.byte_len() / numel) as usize;
    let mut filled = 0usize;
    let mut written = 0usize;
    let mut read = 0u64;
    let mut flush = |staging: &[u8], filled: &mut usize, written: &mut usize| {
        if *filled == 0 {
            return Ok::<(), ModelError>(());
        }
        tensor
            .storage
            .copy_from_host(base + *written, &staging[..*filled])?;
        mem.synchronize()?;
        *written += *filled;
        *filled = 0;
        Ok(())
    };
    for (offset, len) in src.runs(es) {
        let mut done = 0usize;
        while done < len {
            if filled == staging.len() {
                flush(staging, &mut filled, &mut written)?;
            }
            let n = (len - done).min(staging.len() - filled);
            file.read_exact_at(
                &mut staging[filled..filled + n],
                entry.range.start + offset + done as u64,
            )
            .map_err(|e| io_err(&entry.file, e))?;
            filled += n;
            done += n;
            read += n as u64;
        }
    }
    flush(staging, &mut filled, &mut written)?;
    // Padding rows: zeros, a staging buffer at a time.
    let pad = dest_bytes as usize - written;
    if pad > 0 {
        staging.fill(0);
        let mut done = 0usize;
        while done < pad {
            filled = (pad - done).min(staging.len());
            done += filled;
            flush(staging, &mut filled, &mut written)?;
        }
    }
    Ok(read)
}

/// Every stack is tiled exactly: each slot's trailing dimensions are the stack's, and the slots
/// naming it cover its elements once, without gap or overlap.
fn check_stacks(slots: &[WeightSlot]) -> Result<(), ModelError> {
    let bad = |message: String| ModelError::Kernel(KernelError::InvalidArgument { message });
    // Per stack: its shape and the `(offset, len, slot)` element ranges naming it.
    type Parts<'a> = Vec<(usize, usize, &'a str)>;
    let mut stacks: HashMap<&str, (&[usize], Parts<'_>)> = HashMap::new();
    for slot in slots {
        let Some(place) = &slot.stack else { continue };
        let (shape, parts) = stacks
            .entry(place.name.as_str())
            .or_insert_with(|| (place.shape.as_slice(), Vec::new()));
        // An entry along the stack's first axis, or a block of its rows.
        let entry = shape.get(1..) == Some(slot.shape.as_slice());
        let rows = slot.shape.len() == shape.len() && slot.shape.get(1..) == shape.get(1..);
        if place.shape.as_slice() != *shape || !(entry || rows) {
            return Err(bad(format!(
                "{} {:?} does not fit stack {} {:?}",
                slot.name, slot.shape, place.name, place.shape
            )));
        }
        parts.push((
            place.offset,
            slot.shape.iter().product(),
            slot.name.as_str(),
        ));
    }
    for (name, (shape, mut parts)) in stacks {
        parts.sort_unstable();
        let mut next = 0usize;
        for (offset, len, slot) in parts {
            if offset != next {
                return Err(bad(if offset < next {
                    format!("{slot} overlaps another slot of stack {name}")
                } else {
                    format!("stack {name} has no slot for elements {next}..{offset}")
                }));
            }
            next = offset + len;
        }
        let numel: usize = shape.iter().product();
        if next != numel {
            return Err(bad(if next > numel {
                format!("stack {name} {shape:?} is overrun by its slots ({next} elements)")
            } else {
                format!("stack {name} has no slot for elements {next}..{numel}")
            }));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use turbine_core::types::{DType, DeviceId};
    use turbine_tensor::host::HostMemory;
    use turbine_tensor::{DevicePtr, MemInfo, MemoryError, StreamRef};

    use super::*;
    use crate::config::load_model_config;
    use crate::families::{llama_slots, olmoe_slots};
    use crate::testing::TempDir;
    use crate::testing::tiny::{
        TinyOptions, write_tiny_llama, write_tiny_llama_with, write_tiny_olmoe,
    };

    /// Host memory that counts allocations, to prove a failed validation allocates nothing, and
    /// has no device-to-device copy (as the HIP shim under kernel ABI v2), to prove loading
    /// never needs one.
    struct CountingMemory {
        inner: Arc<HostMemory>,
        allocs: AtomicUsize,
    }

    impl DeviceMemory for CountingMemory {
        fn device(&self) -> DeviceId {
            self.inner.device()
        }
        fn alloc(&self, bytes: usize) -> Result<DevicePtr, MemoryError> {
            self.allocs.fetch_add(1, Ordering::SeqCst);
            self.inner.alloc(bytes)
        }
        fn free(&self, ptr: DevicePtr) {
            self.inner.free(ptr)
        }
        fn copy_h2d(&self, dst: DevicePtr, src: &[u8]) -> Result<(), MemoryError> {
            self.inner.copy_h2d(dst, src)
        }
        fn copy_d2h(&self, dst: &mut [u8], src: DevicePtr) -> Result<(), MemoryError> {
            self.inner.copy_d2h(dst, src)
        }
        fn copy_d2d(&self, _: DevicePtr, _: DevicePtr, _: usize) -> Result<(), MemoryError> {
            Err(MemoryError::Unsupported(
                "device-to-device copy needs kernel ABI v3".into(),
            ))
        }
        fn synchronize(&self) -> Result<(), MemoryError> {
            self.inner.synchronize()
        }
        fn mem_info(&self) -> Result<MemInfo, MemoryError> {
            self.inner.mem_info()
        }
        fn compute_stream(&self) -> StreamRef {
            self.inner.compute_stream()
        }
        fn as_host(&self) -> Option<&HostMemory> {
            Some(&self.inner)
        }
    }

    fn counting() -> (Arc<CountingMemory>, Arc<dyn DeviceMemory>) {
        let mem = Arc::new(CountingMemory {
            inner: HostMemory::new(DeviceId(0), 1 << 30),
            allocs: AtomicUsize::new(0),
        });
        let dyn_mem: Arc<dyn DeviceMemory> = mem.clone();
        (mem, dyn_mem)
    }

    fn load(dir: &TempDir, staging: usize) -> Result<LoadedWeights, ModelError> {
        let cfg = load_model_config(dir.path())?;
        let index = SafetensorsIndex::open(dir.path())?;
        let (_, mem) = counting();
        WeightLoader::load(&index, &llama_slots(&cfg), &mem, staging)
    }

    /// The device bytes `slot` was loaded into: its own tensor, or its place in its stack.
    fn slot_bytes_loaded(weights: &LoadedWeights, slot: &WeightSlot) -> Vec<u8> {
        let (tensor, offset) = match &slot.stack {
            Some(place) => {
                let stack = &weights.tensors[&place.name];
                assert_eq!(
                    stack.shape.as_slice(),
                    place.shape.as_slice(),
                    "{}",
                    place.name
                );
                (stack, place.offset * DType::BF16.size_bytes())
            }
            None => {
                let tensor = &weights.tensors[&slot.name];
                assert_eq!(
                    tensor.shape.as_slice(),
                    slot.shape.as_slice(),
                    "{}",
                    slot.name
                );
                (tensor, 0)
            }
        };
        assert_eq!(tensor.dtype, DType::BF16);
        let mut got = vec![0u8; slot.shape.iter().product::<usize>() * DType::BF16.size_bytes()];
        tensor.storage.copy_to_host(offset, &mut got).unwrap();
        got
    }

    /// Asserts every slot is loaded with its shape and exactly the checkpoint's bytes: the norms,
    /// embedding, O and down projections (and an untied LM head) as their own tensors, Q/K/V and
    /// gate/up as rows of each layer's two fused parameters.
    fn assert_matches_file(dir: &TempDir, weights: &LoadedWeights) {
        let cfg = load_model_config(dir.path()).unwrap();
        let index = SafetensorsIndex::open(dir.path()).unwrap();
        let slots = llama_slots(&cfg);
        // Per layer five projections become two fused parameters.
        let layers = cfg.num_layers as usize;
        assert_eq!(weights.tensors.len(), slots.len() - 5 * layers + 2 * layers);
        let file = std::fs::read(dir.path().join("model.safetensors")).unwrap();
        let mut total = 0u64;
        for slot in &slots {
            let entry = index.get(&slot.name).unwrap();
            let want = &file[entry.range.start as usize..entry.range.end as usize];
            let got = slot_bytes_loaded(weights, slot);
            assert!(got == want, "{} bytes differ from the file", slot.name);
            total += entry.byte_len();
        }
        assert_eq!(weights.weight_bytes, total);
    }

    #[test]
    fn tensor_mapping() {
        // The default tiny checkpoint: every slot filled, nothing unexpected or ignored.
        let dir = TempDir::new("load-tiny");
        let spec = write_tiny_llama(dir.path(), 3);
        let slots = llama_slots(&spec.config);
        assert_eq!(slots.len(), 1 + 2 * 9 + 1, "tied: no lm_head slot");
        assert!(slots.iter().all(|s| s.name != LM_HEAD));
        let weights = load(&dir, MAX_STAGING_BYTES).unwrap();
        assert_matches_file(&dir, &weights);
        assert!(weights.unexpected.is_empty() && weights.ignored.is_empty());
        // Q/K/V (64 + 32 + 32 rows) and gate/up (128 + 128 rows) are one parameter each; the
        // separate projections are not kept.
        assert_eq!(
            weights.tensors[&qkv_proj_name(1)].shape.as_slice(),
            &[128, 64]
        );
        assert_eq!(
            weights.tensors[&gate_up_proj_name(0)].shape.as_slice(),
            &[256, 64]
        );
        for gone in ["self_attn.q_proj", "self_attn.v_proj", "mlp.up_proj"] {
            assert!(
                !weights
                    .tensors
                    .contains_key(&format!("model.layers.0.{gone}.weight"))
            );
        }
        let place = |name: &str| slots.iter().find(|s| s.name == name).unwrap().stack.clone();
        assert_eq!(
            place("model.layers.1.self_attn.v_proj.weight"),
            Some(StackPlace {
                name: qkv_proj_name(1),
                shape: vec![128, 64],
                offset: 96 * 64,
            })
        );
        assert_eq!(
            place("model.layers.0.mlp.up_proj.weight").map(|p| (p.name, p.offset)),
            Some((gate_up_proj_name(0), 128 * 64))
        );
        let mut weights = weights;
        let norm = weights.take("model.norm.weight").unwrap();
        assert_eq!(norm.shape.as_slice(), &[64]);
        assert!(matches!(
            weights.take("model.norm.weight"),
            Err(ModelError::MissingTensor(n)) if n == "model.norm.weight"
        ));

        // A 64-byte staging buffer loads identical bytes.
        let small = load(&dir, 64).unwrap();
        assert_matches_file(&dir, &small);

        // A missing tensor is refused, naming it, before any tensor is allocated.
        let missing = TempDir::new("load-missing");
        let omit = "model.layers.1.mlp.down_proj.weight".to_string();
        let opts = TinyOptions {
            omit: vec![omit.clone()],
            ..TinyOptions::default()
        };
        let spec = write_tiny_llama_with(missing.path(), 3, &opts);
        let index = SafetensorsIndex::open(missing.path()).unwrap();
        let (counter, mem) = counting();
        let err = WeightLoader::load(&index, &llama_slots(&spec.config), &mem, 1 << 20)
            .expect_err("missing tensor must be refused");
        assert_eq!(err.to_string(), format!("missing tensor {omit}"));
        assert_eq!(counter.allocs.load(Ordering::SeqCst), 0);

        // An extra tensor is reported by name.
        let extra = TempDir::new("load-extra");
        let opts = TinyOptions {
            extra: vec!["model.extra.weight".to_string()],
            ..TinyOptions::default()
        };
        write_tiny_llama_with(extra.path(), 3, &opts);
        let weights = load(&extra, 1 << 20).unwrap();
        assert_eq!(weights.unexpected, vec!["model.extra.weight".to_string()]);
        assert!(weights.ignored.is_empty());
        assert!(!weights.tensors.contains_key("model.extra.weight"));

        // Tied with a shipped lm_head: ignored, not loaded.
        let shipped = TempDir::new("load-shipped");
        let opts = TinyOptions {
            ship_lm_head: true,
            ..TinyOptions::default()
        };
        write_tiny_llama_with(shipped.path(), 3, &opts);
        let weights = load(&shipped, 1 << 20).unwrap();
        assert_eq!(weights.ignored, vec![LM_HEAD.to_string()]);
        assert!(weights.unexpected.is_empty());
        assert!(!weights.tensors.contains_key(LM_HEAD));

        // Untied: lm_head.weight is a slot and is loaded.
        let untied = TempDir::new("load-untied");
        let opts = TinyOptions {
            tied: false,
            ..TinyOptions::default()
        };
        let spec = write_tiny_llama_with(untied.path(), 3, &opts);
        assert!(!spec.config.tie_word_embeddings);
        let weights = load(&untied, 1 << 20).unwrap();
        assert_matches_file(&untied, &weights);
        assert_eq!(weights.tensors[LM_HEAD].shape.as_slice(), &[263, 64]);
        assert!(weights.ignored.is_empty());
    }

    #[test]
    fn olmoe_slot_mapping() {
        // OLMoE-1B-7B: 3 top-level tensors + per layer 9 attention/norm/router tensors and
        // 64 experts × 3 projections = 3219, the tensor count of its safetensors index.
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/olmoe-1b-7b-0125-instruct");
        let cfg = load_model_config(&fixture).unwrap();
        let slots = olmoe_slots(&cfg);
        assert_eq!(slots.len(), 3219);
        let shape = |name: &str| {
            slots
                .iter()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("no slot {name}"))
                .shape
                .clone()
        };
        assert_eq!(shape("model.layers.15.self_attn.q_norm.weight"), [2048]);
        assert_eq!(shape("model.layers.15.self_attn.k_norm.weight"), [2048]);
        assert_eq!(shape("model.layers.0.mlp.gate.weight"), [64, 2048]);
        assert_eq!(
            shape("model.layers.3.mlp.experts.63.gate_proj.weight"),
            [1024, 2048]
        );
        assert_eq!(
            shape("model.layers.3.mlp.experts.0.up_proj.weight"),
            [1024, 2048]
        );
        assert_eq!(
            shape("model.layers.3.mlp.experts.7.down_proj.weight"),
            [2048, 1024]
        );
        assert_eq!(shape(LM_HEAD), [50_304, 2048]);
        assert!(!slots.iter().any(|s| s.name.contains("mlp.gate_proj")));
        // Every expert projection is one entry of its layer's stacked parameter.
        let place = |name: &str| slots.iter().find(|s| s.name == name).unwrap().stack.clone();
        assert_eq!(
            place("model.layers.3.mlp.experts.7.down_proj.weight"),
            Some(StackPlace {
                name: "model.layers.3.mlp.experts.down_proj.weight".into(),
                shape: vec![64, 2048, 1024],
                offset: 7 * 2048 * 1024,
            })
        );
        assert_eq!(
            place("model.layers.0.mlp.experts.63.gate_proj.weight").map(|p| (p.shape, p.offset)),
            Some((vec![64, 1024, 2048], 63 * 1024 * 2048))
        );
        assert_eq!(place("model.layers.0.mlp.gate.weight"), None);
        // Q/K/V are rows of the layer's fused projection (OLMoE-1B-7B: 16 heads of 128, MHA).
        assert_eq!(
            place("model.layers.2.self_attn.k_proj.weight"),
            Some(StackPlace {
                name: qkv_proj_name(2),
                shape: vec![3 * 2048, 2048],
                offset: 2048 * 2048,
            })
        );
        assert_eq!(
            slots.iter().filter(|s| s.stack.is_some()).count(),
            16 * (64 * 3 + 3)
        );

        // The tiny OLMoE (2 layers, 8 experts) loads every slot with its exact bytes, each expert
        // at its offset in the layer's stacked parameter (also through a 64-byte staging buffer,
        // smaller than one expert); nothing unexpected, no device-to-device copy.
        let dir = TempDir::new("load-tiny-olmoe");
        let spec = write_tiny_olmoe(dir.path(), 4);
        let index = SafetensorsIndex::open(dir.path()).unwrap();
        let (_, mem) = counting();
        let slots = olmoe_slots(&spec.config);
        let file = std::fs::read(dir.path().join("model.safetensors")).unwrap();
        for staging in [1 << 20, 64] {
            let weights = WeightLoader::load(&index, &slots, &mem, staging).unwrap();
            assert!(weights.unexpected.is_empty() && weights.ignored.is_empty());
            let plain = slots.iter().filter(|s| s.stack.is_none()).count();
            assert_eq!(
                weights.tensors.len(),
                plain + 2 * (3 + 1),
                "one stack per layer and expert proj, plus the fused Q/K/V"
            );
            assert!(
                !weights
                    .tensors
                    .contains_key("model.layers.0.mlp.experts.0.up_proj.weight")
            );
            let stacked = &weights.tensors[&stacked_experts_name(1, "down_proj")];
            assert_eq!(stacked.shape.as_slice(), &[8, 64, 32]);
            for slot in &slots {
                let entry = index.get(&slot.name).unwrap();
                let got = slot_bytes_loaded(&weights, slot);
                assert!(
                    got == file[entry.range.start as usize..entry.range.end as usize],
                    "{} (staging {staging})",
                    slot.name
                );
            }
            assert_eq!(weights.weight_bytes, spec.config.shape().weight_bytes);
        }

        // A stack with elements named twice, or left unnamed, or a slot that does not fit its
        // stack's shape, is refused before any allocation.
        let (counter, fresh) = counting();
        let expert_stack = |s: &WeightSlot| {
            s.stack
                .as_ref()
                .is_some_and(|p| p.name == stacked_experts_name(0, "gate_proj"))
        };
        let mut twice = slots.clone();
        let i = twice.iter().position(expert_stack).unwrap();
        twice[i + 3].stack.as_mut().unwrap().offset = 0;
        let err = WeightLoader::load(&index, &twice, &fresh, 1 << 20).expect_err("offset twice");
        assert!(
            err.to_string().contains(&format!(
                "overlaps another slot of stack {}",
                stacked_experts_name(0, "gate_proj")
            )),
            "{err}"
        );
        let mut gap = slots.clone();
        gap.remove(i);
        let err = WeightLoader::load(&index, &gap, &fresh, 1 << 20).expect_err("unfilled stack");
        assert!(
            err.to_string().contains(&format!(
                "stack {} has no slot for elements 0..{}",
                stacked_experts_name(0, "gate_proj"),
                32 * 64
            )),
            "{err}"
        );
        let mut tail = slots.clone();
        let v = tail
            .iter()
            .position(|s| s.name == "model.layers.1.self_attn.v_proj.weight")
            .unwrap();
        tail.remove(v);
        let err = WeightLoader::load(&index, &tail, &fresh, 1 << 20).expect_err("no v rows");
        assert!(
            err.to_string().contains(&format!(
                "stack {} has no slot for elements",
                qkv_proj_name(1)
            )),
            "{err}"
        );
        let mut misfit = slots.clone();
        misfit[v].stack.as_mut().unwrap().shape = vec![64, 64];
        let err = WeightLoader::load(&index, &misfit, &fresh, 1 << 20).expect_err("misfit");
        assert!(err.to_string().contains("does not fit stack"), "{err}");
        assert_eq!(counter.allocs.load(Ordering::SeqCst), 0);

        // The Llama slot set does not fit an OLMoE checkpoint: it is refused, naming a tensor.
        let err = WeightLoader::load(&index, &llama_slots(&spec.config), &mem, 1 << 20)
            .expect_err("llama slots on olmoe");
        assert_eq!(
            err.to_string(),
            "missing tensor model.layers.0.mlp.gate_proj.weight"
        );
    }

    #[test]
    fn rejects_wrong_shape_before_allocating() {
        let dir = TempDir::new("load-shape");
        let spec = write_tiny_llama(dir.path(), 3);
        let index = SafetensorsIndex::open(dir.path()).unwrap();
        let mut slots = llama_slots(&spec.config);
        slots[1].shape = vec![65];
        let (counter, mem) = counting();
        let err = WeightLoader::load(&index, &slots, &mem, 1 << 20).expect_err("bad shape");
        let text = err.to_string();
        assert!(
            text.contains("model.layers.0.input_layernorm.weight")
                && text.contains("shape [64] != expected [65]"),
            "{text}"
        );
        assert_eq!(counter.allocs.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn never_opens_pickle() {
        let dir = TempDir::new("load-pickle");
        let bin = dir.path().join("pytorch_model.bin");
        std::fs::write(&bin, b"\x80\x02pickle").unwrap();
        // Unreadable: any attempt to open it would fail with a permission error instead.
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o000)).unwrap();
        let err = SafetensorsIndex::open(dir.path()).expect_err("pickle-only directory");
        let text = err.to_string();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(
            matches!(err, ModelError::Pickle { ref path } if *path == bin),
            "{text}"
        );
        assert!(text.contains(&bin.display().to_string()), "{text}");
        assert!(text.contains("pickle formats are not supported"), "{text}");
    }
}
