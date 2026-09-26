//! Weight loader (P1 S-2): maps checkpoint tensor names to the executor's parameter slots,
//! validates every slot (present, BF16, expected shape) before allocating anything, then uploads
//! each tensor to the device through one bounded, reused host staging buffer filled with
//! positioned reads (`read_exact_at`; no `mmap`, so this crate stays free of `unsafe`).
//! Checkpoint tensors no slot names are reported by name (`unexpected`), and a tied model's
//! shipped `lm_head.weight` is `ignored`. A slot may name a place in a stacked parameter
//! ([`StackPlace`]): its bytes are uploaded straight into that parameter at the place's offset,
//! so stacking (OLMoE's per-expert weights) needs no device-to-device copy, which the HIP shim
//! lacks before kernel ABI v3.
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::Arc;

use turbine_core::types::DType;
use turbine_kernels::KernelError;
use turbine_tensor::{DeviceMemory, Tensor};

use crate::ModelError;
use crate::config::{Architecture, ModelArchConfig};
use crate::safetensors::{Dtype, SafetensorsIndex, TensorEntry, io_err, open_regular};

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
    /// `Some`: the tensor is entry `index` of the stacked parameter `stack.name` and is loaded
    /// only as part of it (`LoadedWeights` holds the stack, not the slot's own name).
    pub stack: Option<StackPlace>,
}

/// Entry `index` along the first axis of the stacked parameter `name` of `shape`
/// (`[count, slot shape…]`). Every index in `0..count` must be named by exactly one slot of the
/// same slot shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StackPlace {
    pub name: String,
    pub shape: Vec<usize>,
    pub index: usize,
}

/// The stacked `[experts, rows, cols]` parameter of layer `layer`'s expert projection `proj`
/// (`gate_proj`, `up_proj`, `down_proj`), as [`olmoe_slots`] loads it.
pub fn stacked_experts_name(layer: u32, proj: &str) -> String {
    format!("model.layers.{layer}.mlp.experts.{proj}.weight")
}

/// Every parameter slot of a Llama model, in load order: embedding, per layer the attention
/// and MLP weights with their norms, the final norm, and `lm_head.weight` only when untied.
pub fn llama_slots(cfg: &ModelArchConfig) -> Vec<WeightSlot> {
    let hidden = cfg.hidden as usize;
    let q = cfg.num_attention_heads as usize * cfg.head_dim as usize;
    let kv = cfg.num_kv_heads as usize * cfg.head_dim as usize;
    let inter = cfg.intermediate as usize;
    let vocab = cfg.vocab_size as usize;
    let slot = |name: String, shape: Vec<usize>| WeightSlot {
        name,
        shape,
        stack: None,
    };

    let mut slots = vec![slot(
        "model.embed_tokens.weight".into(),
        vec![vocab, hidden],
    )];
    for i in 0..cfg.num_layers {
        let p = format!("model.layers.{i}");
        slots.extend([
            slot(format!("{p}.input_layernorm.weight"), vec![hidden]),
            slot(format!("{p}.self_attn.q_proj.weight"), vec![q, hidden]),
            slot(format!("{p}.self_attn.k_proj.weight"), vec![kv, hidden]),
            slot(format!("{p}.self_attn.v_proj.weight"), vec![kv, hidden]),
            slot(format!("{p}.self_attn.o_proj.weight"), vec![hidden, q]),
            slot(format!("{p}.post_attention_layernorm.weight"), vec![hidden]),
            slot(format!("{p}.mlp.gate_proj.weight"), vec![inter, hidden]),
            slot(format!("{p}.mlp.up_proj.weight"), vec![inter, hidden]),
            slot(format!("{p}.mlp.down_proj.weight"), vec![hidden, inter]),
        ]);
    }
    slots.push(slot("model.norm.weight".into(), vec![hidden]));
    if !cfg.tie_word_embeddings {
        slots.push(slot(LM_HEAD.into(), vec![vocab, hidden]));
    }
    slots
}

/// Every parameter slot of an OLMoE model, in load order: embedding, per layer the attention
/// weights with the Q/K norms (over the full `heads·head_dim` and `kv_heads·head_dim`
/// projections), the router `mlp.gate.weight` `[experts, hidden]`, every expert's SwiGLU
/// weights, the two layer norms, then the final norm and `lm_head.weight` only when untied.
/// Each expert projection is entry `e` of the layer's stacked `[experts, rows, cols]` parameter
/// [`stacked_experts_name`] (the `moe_experts` layout). A dense config (`moe: None`) has no
/// experts and yields only the non-MLP slots.
pub fn olmoe_slots(cfg: &ModelArchConfig) -> Vec<WeightSlot> {
    let hidden = cfg.hidden as usize;
    let q = cfg.num_attention_heads as usize * cfg.head_dim as usize;
    let kv = cfg.num_kv_heads as usize * cfg.head_dim as usize;
    let vocab = cfg.vocab_size as usize;
    let (experts, inter) = cfg.moe.map_or((0, 0), |m| {
        (m.num_experts as usize, m.expert_intermediate as usize)
    });
    let slot = |name: String, shape: Vec<usize>| WeightSlot {
        name,
        shape,
        stack: None,
    };

    let mut slots = vec![slot(
        "model.embed_tokens.weight".into(),
        vec![vocab, hidden],
    )];
    for i in 0..cfg.num_layers {
        let p = format!("model.layers.{i}");
        slots.extend([
            slot(format!("{p}.input_layernorm.weight"), vec![hidden]),
            slot(format!("{p}.self_attn.q_proj.weight"), vec![q, hidden]),
            slot(format!("{p}.self_attn.k_proj.weight"), vec![kv, hidden]),
            slot(format!("{p}.self_attn.v_proj.weight"), vec![kv, hidden]),
            slot(format!("{p}.self_attn.o_proj.weight"), vec![hidden, q]),
            slot(format!("{p}.self_attn.q_norm.weight"), vec![q]),
            slot(format!("{p}.self_attn.k_norm.weight"), vec![kv]),
            slot(format!("{p}.post_attention_layernorm.weight"), vec![hidden]),
            slot(format!("{p}.mlp.gate.weight"), vec![experts, hidden]),
        ]);
        let expert = |e: usize, proj: &str, shape: Vec<usize>| {
            let stack = StackPlace {
                name: stacked_experts_name(i, proj),
                shape: [vec![experts], shape.clone()].concat(),
                index: e,
            };
            WeightSlot {
                name: format!("{p}.mlp.experts.{e}.{proj}.weight"),
                shape,
                stack: Some(stack),
            }
        };
        for e in 0..experts {
            slots.extend([
                expert(e, "gate_proj", vec![inter, hidden]),
                expert(e, "up_proj", vec![inter, hidden]),
                expert(e, "down_proj", vec![hidden, inter]),
            ]);
        }
    }
    slots.push(slot("model.norm.weight".into(), vec![hidden]));
    if !cfg.tie_word_embeddings {
        slots.push(slot(LM_HEAD.into(), vec![vocab, hidden]));
    }
    slots
}

/// The parameter slots of `cfg`'s architecture.
pub(crate) fn weight_slots(cfg: &ModelArchConfig) -> Vec<WeightSlot> {
    match cfg.architecture {
        Architecture::Llama => llama_slots(cfg),
        Architecture::Olmoe => olmoe_slots(cfg),
    }
}

/// Phase 1 loads BF16 weights only; anything else is refused naming the tensor.
pub(crate) fn require_bf16(entry: &TensorEntry) -> Result<(), ModelError> {
    if entry.dtype == Dtype::BF16 {
        Ok(())
    } else {
        Err(ModelError::Unsupported {
            field: "tensor dtype".to_string(),
            value: format!("{} ({})", entry.dtype, entry.name),
            supported: "BF16".to_string(),
        })
    }
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
    /// Validates every slot against `index` (present, BF16, expected shape) and every stack
    /// (each index named once, slot shapes matching), then allocates each tensor on `mem` and
    /// uploads it through one host staging buffer of `min(staging_bytes, MAX_STAGING_BYTES)`
    /// bytes (at least 1, at most the largest tensor); a stacked slot's bytes go straight to its
    /// offset in the stacked tensor (host-to-device copies only). Nothing is allocated when
    /// validation fails. The staging buffer is reused only after the previous copy completed
    /// (`synchronize`), since device copies are enqueued.
    pub fn load(
        index: &SafetensorsIndex,
        slots: &[WeightSlot],
        mem: &Arc<dyn DeviceMemory>,
        staging_bytes: usize,
    ) -> Result<LoadedWeights, ModelError> {
        let mut planned: Vec<(&WeightSlot, &TensorEntry)> = Vec::with_capacity(slots.len());
        for slot in slots {
            let entry = index
                .get(&slot.name)
                .ok_or_else(|| ModelError::MissingTensor(slot.name.clone()))?;
            require_bf16(entry)?;
            if entry.shape != slot.shape {
                return Err(ModelError::Safetensors {
                    file: entry.file.clone(),
                    tensor: entry.name.clone(),
                    rule: format!("shape {:?} != expected {:?}", entry.shape, slot.shape),
                });
            }
            planned.push((slot, entry));
        }
        check_stacks(slots)?;

        let wanted: HashSet<&str> = slots.iter().map(|s| s.name.as_str()).collect();
        let (mut unexpected, mut ignored) = (Vec::new(), Vec::new());
        for entry in index.entries() {
            if wanted.contains(entry.name.as_str()) {
                continue;
            }
            if entry.name == LM_HEAD {
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

        let largest = planned.iter().map(|(_, e)| e.byte_len()).max().unwrap_or(0);
        let staging_len = staging_bytes
            .min(MAX_STAGING_BYTES)
            .min(usize::try_from(largest).unwrap_or(usize::MAX))
            .max(1);
        let mut staging = vec![0u8; staging_len];
        let mut files: HashMap<PathBuf, File> = HashMap::new();
        let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(planned.len());
        let mut weight_bytes = 0u64;
        for (slot, entry) in planned {
            // The destination tensor and this slot's byte offset in it.
            let (key, shape, base) = match &slot.stack {
                Some(place) => (&place.name, &place.shape, place.index * slot_bytes(slot)),
                None => (&slot.name, &slot.shape, 0),
            };
            if !tensors.contains_key(key) {
                tensors.insert(key.clone(), Tensor::empty(mem, shape, DType::BF16)?);
            }
            let tensor = tensors.get_mut(key).expect("inserted above");
            let file = match files.get(&entry.file) {
                Some(f) => f,
                None => {
                    let f = open_regular(&entry.file)?;
                    files.entry(entry.file.clone()).or_insert(f)
                }
            };
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
        })
    }
}

/// BF16 bytes of one slot's tensor.
fn slot_bytes(slot: &WeightSlot) -> usize {
    slot.shape.iter().product::<usize>() * DType::BF16.size_bytes()
}

/// Every stack is filled exactly: each index in `0..count` named by one slot whose shape is the
/// stack's shape without its first axis.
fn check_stacks(slots: &[WeightSlot]) -> Result<(), ModelError> {
    let bad = |message: String| ModelError::Kernel(KernelError::InvalidArgument { message });
    let mut stacks: HashMap<&str, (&[usize], Vec<bool>)> = HashMap::new();
    for slot in slots {
        let Some(place) = &slot.stack else { continue };
        let count = place.shape.first().copied().unwrap_or(0);
        let (shape, filled) = stacks
            .entry(place.name.as_str())
            .or_insert_with(|| (place.shape.as_slice(), vec![false; count]));
        if place.shape.as_slice() != *shape || shape.get(1..) != Some(slot.shape.as_slice()) {
            return Err(bad(format!(
                "{} {:?} does not fit stack {} {:?}",
                slot.name, slot.shape, place.name, place.shape
            )));
        }
        match filled.get_mut(place.index) {
            Some(f) if !*f => *f = true,
            _ => {
                return Err(bad(format!(
                    "{} names index {} of stack {} twice or out of range",
                    slot.name, place.index, place.name
                )));
            }
        }
    }
    for (name, (_, filled)) in stacks {
        if let Some(missing) = filled.iter().position(|f| !f) {
            return Err(bad(format!("stack {name} has no slot for index {missing}")));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use turbine_core::types::DeviceId;
    use turbine_tensor::host::HostMemory;
    use turbine_tensor::{DevicePtr, MemInfo, MemoryError, StreamRef};

    use super::*;
    use crate::config::load_model_config;
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

    /// Asserts every slot is loaded with its shape and exactly the checkpoint's bytes.
    fn assert_matches_file(dir: &TempDir, weights: &LoadedWeights) {
        let cfg = load_model_config(dir.path()).unwrap();
        let index = SafetensorsIndex::open(dir.path()).unwrap();
        let slots = llama_slots(&cfg);
        assert_eq!(weights.tensors.len(), slots.len());
        let file = std::fs::read(dir.path().join("model.safetensors")).unwrap();
        let mut total = 0u64;
        for slot in &slots {
            let tensor = &weights.tensors[&slot.name];
            assert_eq!(
                tensor.shape.as_slice(),
                slot.shape.as_slice(),
                "{}",
                slot.name
            );
            assert_eq!(tensor.dtype, DType::BF16);
            let entry = index.get(&slot.name).unwrap();
            let mut got = vec![0u8; tensor.storage.len()];
            tensor.storage.copy_to_host(0, &mut got).unwrap();
            let want = &file[entry.range.start as usize..entry.range.end as usize];
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
                index: 7,
            })
        );
        assert_eq!(
            place("model.layers.0.mlp.experts.63.gate_proj.weight").map(|p| (p.shape, p.index)),
            Some((vec![64, 1024, 2048], 63))
        );
        assert_eq!(place("model.layers.0.mlp.gate.weight"), None);
        assert_eq!(
            slots.iter().filter(|s| s.stack.is_some()).count(),
            16 * 64 * 3
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
                plain + 2 * 3,
                "one stack per layer and proj"
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
                let len = entry.byte_len() as usize;
                let (tensor, offset) = match &slot.stack {
                    Some(place) => (&weights.tensors[&place.name], place.index * len),
                    None => (&weights.tensors[&slot.name], 0),
                };
                let mut got = vec![0u8; len];
                tensor.storage.copy_to_host(offset, &mut got).unwrap();
                assert!(
                    got == file[entry.range.start as usize..entry.range.end as usize],
                    "{} (staging {staging})",
                    slot.name
                );
            }
            assert_eq!(weights.weight_bytes, spec.config.shape().weight_bytes);
        }

        // A stack with an index named twice, or one left unnamed, is refused before any
        // allocation.
        let (counter, fresh) = counting();
        let mut twice = slots.clone();
        let i = twice.iter().position(|s| s.stack.is_some()).unwrap();
        twice[i + 3].stack.as_mut().unwrap().index = 0;
        let err = WeightLoader::load(&index, &twice, &fresh, 1 << 20).expect_err("index twice");
        assert!(err.to_string().contains("twice or out of range"), "{err}");
        let mut gap = slots.clone();
        gap.remove(i);
        let err = WeightLoader::load(&index, &gap, &fresh, 1 << 20).expect_err("unfilled stack");
        assert!(
            err.to_string().contains(&format!(
                "stack {} has no slot for index 0",
                stacked_experts_name(0, "gate_proj")
            )),
            "{err}"
        );
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
