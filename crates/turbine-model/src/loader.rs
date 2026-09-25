//! Weight loader (P1 S-2): maps checkpoint tensor names to the executor's parameter slots,
//! validates every slot (present, BF16, expected shape) before allocating anything, then uploads
//! each tensor to the device through one bounded, reused host staging buffer filled with
//! positioned reads (`read_exact_at`; no `mmap`, so this crate stays free of `unsafe`).
//! Checkpoint tensors no slot names are reported by name (`unexpected`), and a tied model's
//! shipped `lm_head.weight` is `ignored`.
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::Arc;

use turbine_core::types::DType;
use turbine_tensor::{DeviceMemory, Tensor};

use crate::ModelError;
use crate::config::ModelArchConfig;
use crate::safetensors::{Dtype, SafetensorsIndex, TensorEntry, io_err, open_regular};

/// Upper bound of the host staging buffer (P1 bounded resources).
pub const MAX_STAGING_BYTES: usize = 256 << 20;
/// The untied output projection; ignored when the model ties it to the embedding.
pub const LM_HEAD: &str = "lm_head.weight";

/// One executor parameter: checkpoint tensor name and expected shape (row-major, elements).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightSlot {
    pub name: String,
    pub shape: Vec<usize>,
}

/// Every parameter slot of a Llama model, in load order: embedding, per layer the attention
/// and MLP weights with their norms, the final norm, and `lm_head.weight` only when untied.
pub fn llama_slots(cfg: &ModelArchConfig) -> Vec<WeightSlot> {
    let hidden = cfg.hidden as usize;
    let q = cfg.num_attention_heads as usize * cfg.head_dim as usize;
    let kv = cfg.num_kv_heads as usize * cfg.head_dim as usize;
    let inter = cfg.intermediate as usize;
    let vocab = cfg.vocab_size as usize;
    let slot = |name: String, shape: Vec<usize>| WeightSlot { name, shape };

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

/// The loaded parameters by checkpoint name, plus what the checkpoint held beyond them.
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
    /// Validates every slot against `index` (present, BF16, expected shape), then allocates each
    /// tensor on `mem` and uploads it through one host staging buffer of
    /// `min(staging_bytes, MAX_STAGING_BYTES)` bytes (at least 1, at most the largest tensor).
    /// Nothing is allocated when validation fails. The staging buffer is reused only after the
    /// previous copy completed (`synchronize`), since device copies are enqueued.
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
        let mut tensors = HashMap::with_capacity(planned.len());
        let mut weight_bytes = 0u64;
        for (slot, entry) in planned {
            let mut tensor = Tensor::empty(mem, &slot.shape, DType::BF16)?;
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
                tensor.storage.copy_from_host(done as usize, chunk)?;
                mem.synchronize()?;
                done += n as u64;
            }
            weight_bytes += entry.byte_len();
            tensors.insert(slot.name.clone(), tensor);
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
    use crate::testing::tiny::{TinyOptions, write_tiny_llama, write_tiny_llama_with};

    /// Host memory that counts allocations, to prove a failed validation allocates nothing.
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
        fn copy_d2d(
            &self,
            dst: DevicePtr,
            src: DevicePtr,
            bytes: usize,
        ) -> Result<(), MemoryError> {
            self.inner.copy_d2d(dst, src, bytes)
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
