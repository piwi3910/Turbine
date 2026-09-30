//! KV identity (P4 S-1): namespace keys and 128-bit block keys forming a BLAKE3 hash chain.
//!
//! A block key covers the namespace (model identity with its RoPE parameters, KV format, block
//! size, cache salt), the parent block's key and the block's token ids, so equal keys mean equal
//! prefixes up to a hash collision; the directory still compares stored tokens, so a collision is
//! a miss.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use turbine_core::types::{KvDtype, KvLayout, ModelIdentity};

/// Block key (TS §8 name): first 128 bits of BLAKE3(namespace ‖ parent ‖ tokens as LE u32).
/// Displayed as 32 lowercase hex characters.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct KvKey(pub [u8; 16]);

/// Phase 6 name for the same key (CONFLICT C-17).
pub type BlockKey = KvKey;

impl fmt::Display for KvKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for KvKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KvKey({self})")
    }
}

/// BLAKE3 of the canonical namespace JSON (P4 §Data).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NamespaceKey(pub [u8; 32]);

/// BLAKE3 of the per-layer FP8 KV scales (Phase 6a S-16): K's and V's, each over the scales as
/// little-endian f32 in layer order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct KvScaleHashes {
    pub k: [u8; 32],
    pub v: [u8; 32],
}

impl KvScaleHashes {
    /// The hashes of per-layer scales `k` and `v`.
    pub fn of(k: &[f32], v: &[f32]) -> KvScaleHashes {
        let hash = |scales: &[f32]| {
            let mut h = blake3::Hasher::new();
            for s in scales {
                h.update(&s.to_le_bytes());
            }
            *h.finalize().as_bytes()
        };
        KvScaleHashes {
            k: hash(k),
            v: hash(v),
        }
    }
}

/// Stored KV format: element type plus the per-token layout (which carries `block_tokens`).
///
/// Under tensor parallelism (Phase 5 §Data) every rank's pool holds its own KV heads and `layout`
/// is one rank's; a tier copy of a block is then the `shards` rank shards concatenated in rank
/// order, [`KvFormat::block_bytes`] bytes in all. `shards` is 1 without tensor parallelism.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct KvFormat {
    pub dtype: KvDtype,
    pub layout: KvLayout,
    /// Rank shards of one block (the tensor-parallel size; 1 without it).
    pub shards: u32,
    /// The per-layer scales of quantized pages (FP8, Phase 6a S-16); `None` for BF16.
    pub scales: Option<KvScaleHashes>,
}

impl KvFormat {
    /// One block's format on one device (no tensor parallelism).
    pub fn single(dtype: KvDtype, layout: KvLayout) -> Self {
        KvFormat {
            dtype,
            layout,
            shards: 1,
            scales: None,
        }
    }

    /// Bytes of one logical block: every rank shard of it.
    pub fn block_bytes(&self) -> u64 {
        self.layout.block_bytes() * u64::from(self.shards.max(1))
    }
}

/// Canonical namespace JSON. Field order is the sorted key order and must never change:
/// serde_json writes fields in declaration order, with no whitespace.
#[derive(Serialize)]
struct CanonicalNamespace<'a> {
    block_tokens: u32,
    cache_salt: &'a str,
    kv_format: CanonicalFormat,
    model_config_hash: String,
    /// BLAKE3 of the resolved RoPE parameters (Phase 6a S-16, plan Task 27): a
    /// `model.rope_scaling` override leaves `model_config_hash` unchanged.
    rope_hash: String,
    weights_index_hash: String,
}

#[derive(Serialize)]
struct CanonicalFormat {
    dtype: &'static str,
    head_dim: u32,
    /// Only for quantized pages (FP8), like `v_scales_hash`, so BF16 keys never change.
    #[serde(skip_serializing_if = "Option::is_none")]
    k_scales_hash: Option<String>,
    num_kv_heads: u32,
    num_layers: u32,
    /// Only with tensor parallelism (> 1), so single-device keys never change: a tier copy of a
    /// tp = n block is n rank shards and must never be read by a process of another tp size.
    #[serde(skip_serializing_if = "Option::is_none")]
    shards: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    v_scales_hash: Option<String>,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Namespace key of one model, KV format and cache salt (`""` when the request sent none, which
/// is the one global namespace every unsalted request shares).
pub fn namespace_key(id: &ModelIdentity, fmt: &KvFormat, cache_salt: &str) -> NamespaceKey {
    let canonical = CanonicalNamespace {
        block_tokens: fmt.layout.block_tokens,
        cache_salt,
        kv_format: CanonicalFormat {
            dtype: fmt.dtype.as_str(),
            head_dim: fmt.layout.head_dim,
            k_scales_hash: fmt.scales.map(|s| hex(&s.k)),
            num_kv_heads: fmt.layout.num_kv_heads,
            num_layers: fmt.layout.num_layers,
            shards: (fmt.shards > 1).then_some(fmt.shards),
            v_scales_hash: fmt.scales.map(|s| hex(&s.v)),
        },
        model_config_hash: hex(&id.config_hash),
        rope_hash: hex(&id.rope_hash),
        weights_index_hash: hex(&id.weights_index_hash),
    };
    // Serialising a struct of strings and integers into a Vec cannot fail.
    let json = serde_json::to_vec(&canonical).expect("canonical namespace JSON serialises");
    NamespaceKey(*blake3::hash(&json).as_bytes())
}

/// The parent key hashed for a prompt's first block.
pub const ROOT_PARENT: KvKey = KvKey([0; 16]);

/// Key of one full block: first 16 bytes of BLAKE3(ns ‖ parent-or-`ROOT_PARENT` ‖ tokens LE u32).
pub fn block_key(ns: &NamespaceKey, parent: Option<&KvKey>, tokens: &[u32]) -> KvKey {
    let mut h = blake3::Hasher::new();
    h.update(&ns.0);
    h.update(&parent.unwrap_or(&ROOT_PARENT).0);
    for t in tokens {
        h.update(&t.to_le_bytes());
    }
    let mut out = [0u8; 16];
    out.copy_from_slice(&h.finalize().as_bytes()[..16]);
    KvKey(out)
}

/// Key of a lossy copy of block `key` stored in codec `format` (P6b S-3): the first 16 bytes of
/// BLAKE3("lossy" ‖ key ‖ format ‖ seed LE). A lossy copy promoted into L0 is filed under it, and
/// a block computed over a lossy prefix chains from it, so exact and lossy lineages never alias.
pub fn lossy_key(key: KvKey, format: &str, seed: u64) -> KvKey {
    let mut h = blake3::Hasher::new();
    h.update(b"lossy");
    h.update(&key.0);
    h.update(format.as_bytes());
    h.update(&seed.to_le_bytes());
    let mut out = [0u8; 16];
    out.copy_from_slice(&h.finalize().as_bytes()[..16]);
    KvKey(out)
}

impl NamespaceKey {
    /// The tier codecs' rotation seed of this namespace (also the [`lossy_key`] seed): its
    /// first 8 bytes, little-endian.
    pub fn seed(&self) -> u64 {
        let mut seed = [0u8; 8];
        seed.copy_from_slice(&self.0[..8]);
        u64::from_le_bytes(seed)
    }
}

/// Computes a block's key from its parent and tokens. Production uses [`Blake3Hasher`]; tests
/// substitute a colliding hasher to prove a collision is a miss.
pub trait KeyHasher {
    fn key(&self, parent: Option<&KvKey>, tokens: &[u32]) -> KvKey;
}

/// [`block_key`] under one namespace.
pub struct Blake3Hasher(pub NamespaceKey);

impl KeyHasher for Blake3Hasher {
    fn key(&self, parent: Option<&KvKey>, tokens: &[u32]) -> KvKey {
        block_key(&self.0, parent, tokens)
    }
}

/// Keys of every full block of `tokens`, each chained to the previous one; a trailing partial
/// block has no key (only full blocks are shareable).
pub fn prefix_keys(hasher: &dyn KeyHasher, tokens: &[u32], block_tokens: u32) -> Vec<KvKey> {
    let mut keys: Vec<KvKey> = Vec::with_capacity(tokens.len() / block_tokens.max(1) as usize);
    for chunk in tokens.chunks_exact(block_tokens as usize) {
        let k = hasher.key(keys.last(), chunk);
        keys.push(k);
    }
    keys
}

/// Namespace keys memoised per distinct cache salt. Bounded: the memo is cleared once it holds
/// [`NamespaceCache::MAX_SALTS`] salts (recomputing a key costs one small hash).
pub struct NamespaceCache {
    id: ModelIdentity,
    fmt: KvFormat,
    memo: HashMap<String, NamespaceKey>,
}

impl NamespaceCache {
    pub const MAX_SALTS: usize = 4096;

    pub fn new(id: ModelIdentity, fmt: KvFormat) -> Self {
        NamespaceCache {
            id,
            fmt,
            memo: HashMap::new(),
        }
    }

    pub fn get(&mut self, cache_salt: &str) -> NamespaceKey {
        if let Some(k) = self.memo.get(cache_salt) {
            return *k;
        }
        if self.memo.len() >= Self::MAX_SALTS {
            self.memo.clear();
        }
        let k = namespace_key(&self.id, &self.fmt, cache_salt);
        self.memo.insert(cache_salt.to_owned(), k);
        k
    }

    pub fn format(&self) -> KvFormat {
        self.fmt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbine_core::types::{DType, KvDtype, KvLayout, ModelIdentity};

    fn llama_format(block_tokens: u32) -> KvFormat {
        KvFormat::single(
            KvDtype::Bf16,
            KvLayout {
                num_layers: 28,
                num_kv_heads: 8,
                head_dim: 128,
                dtype: DType::BF16,
                block_tokens,
            },
        )
    }

    fn model(config_byte: u8) -> ModelIdentity {
        ModelIdentity {
            config_hash: [config_byte; 32],
            weights_index_hash: [7; 32],
            rope_hash: [0; 32],
        }
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    // Golden values: a change here means every cached prefix silently changes identity.
    // Phase 6a Task 27 changed them once, on purpose: the canonical namespace JSON gained
    // `rope_hash` (the resolved RoPE parameters; here the all-zero test hash), so blocks cached
    // under one RoPE configuration are never reused under another. Before: namespace
    // 7e1498499592da3bafd9f9da2af720288b56e58647a30c92555d8992abe96933, keys
    // 048a689576719fe25e37707f36c80bcb, 1a566e6449ac3bd7d1fccfda42604a3d.
    const GOLDEN_NS: &str = "ed23b1a59bb18212b164ef794fa9cf9d322aab219e06c339fae5ed8039daf370";
    const GOLDEN_K0: &str = "c702bd85238874d09bfd14994b61c7dd";
    const GOLDEN_K1: &str = "3160290f63b886a8ec86883ec3e382cf";

    /// P6b S-3: a lossy key differs from its block's key and from every other format's or
    /// seed's, and is stable. Breaks if lossy copies could alias the exact block.
    #[test]
    fn lossy_keys_never_alias() {
        let ns = namespace_key(&model(1), &llama_format(16), "");
        let k = block_key(&ns, None, &[1, 2, 3]);
        let tq4 = lossy_key(k, "tq4", ns.seed());
        assert_ne!(tq4, k);
        assert_eq!(tq4, lossy_key(k, "tq4", ns.seed()));
        assert_ne!(tq4, lossy_key(k, "tq2", ns.seed()));
        assert_ne!(tq4, lossy_key(k, "tq4", ns.seed() ^ 1));
        assert_ne!(tq4, lossy_key(KvKey([1; 16]), "tq4", ns.seed()));
        assert_eq!(ns.seed(), u64::from_le_bytes(ns.0[..8].try_into().unwrap()));
    }

    #[test]
    fn keys_are_stable_and_scoped() {
        let tokens: Vec<u32> = (100..132).collect();
        let ns = namespace_key(&model(1), &llama_format(16), "");
        let h = Blake3Hasher(ns);
        let keys = prefix_keys(&h, &tokens, 16);
        assert_eq!(keys.len(), 2);
        assert_eq!(hex(&ns.0), GOLDEN_NS);
        assert_eq!(keys[0].to_string(), GOLDEN_K0);
        assert_eq!(keys[1].to_string(), GOLDEN_K1);
        assert_eq!(keys[1], block_key(&ns, Some(&keys[0]), &tokens[16..32]));
        assert_eq!(keys[0], block_key(&ns, None, &tokens[..16]));

        let mut changed = tokens.clone();
        changed[20] += 1;
        let changed_keys = prefix_keys(&h, &changed, 16);
        assert_eq!(changed_keys[0], keys[0], "block 0 is unaffected");
        assert_ne!(changed_keys[1], keys[1], "one token changes the key");
        assert_ne!(
            block_key(&ns, Some(&KvKey([9; 16])), &tokens[16..32]),
            keys[1],
            "parent changes the key"
        );
        let other_model = Blake3Hasher(namespace_key(&model(2), &llama_format(16), ""));
        assert_ne!(
            prefix_keys(&other_model, &tokens, 16)[0],
            keys[0],
            "model config hash changes the key"
        );
        let other_block = Blake3Hasher(namespace_key(&model(1), &llama_format(32), ""));
        assert_ne!(
            prefix_keys(&other_block, &tokens, 32)[0],
            keys[0],
            "block size changes the key"
        );
        let salted = Blake3Hasher(namespace_key(&model(1), &llama_format(16), "a"));
        assert_ne!(
            prefix_keys(&salted, &tokens, 16)[0],
            keys[0],
            "cache salt changes the key"
        );

        let mut cache = NamespaceCache::new(model(1), llama_format(16));
        assert_eq!(
            cache.get(""),
            ns,
            "no salt header = the one global namespace"
        );
        assert_eq!(
            cache.get("a"),
            namespace_key(&model(1), &llama_format(16), "a")
        );
        assert_eq!(cache.format(), llama_format(16));
        assert!(
            prefix_keys(&h, &tokens[..15], 16).is_empty(),
            "a partial block has no key"
        );
        assert_eq!(prefix_keys(&h, &tokens[..31], 16).len(), 1);
    }

    #[test]
    fn shard_count_scopes_the_namespace_only_under_tp() {
        let one = llama_format(16);
        // tp = 1 adds no `shards` field, so its namespace is the golden one.
        assert_eq!(hex(&namespace_key(&model(1), &one, "").0), GOLDEN_NS);
        assert_eq!(one.block_bytes(), one.layout.block_bytes());
        let two = KvFormat { shards: 2, ..one };
        let four = KvFormat { shards: 4, ..one };
        let ns1 = namespace_key(&model(1), &one, "");
        let ns2 = namespace_key(&model(1), &two, "");
        let ns4 = namespace_key(&model(1), &four, "");
        assert_ne!(ns2, ns1, "a tp = 2 blob is never read by a tp = 1 process");
        assert_ne!(ns4, ns2, "nor by another tp size");
        assert_eq!(two.block_bytes(), 2 * one.layout.block_bytes());
        let tokens: Vec<u32> = (100..116).collect();
        assert_ne!(
            block_key(&ns2, None, &tokens),
            block_key(&ns1, None, &tokens),
            "the shard count changes every block key"
        );
    }

    fn fp8_format(k: &[f32], v: &[f32]) -> KvFormat {
        let bf16 = llama_format(16);
        KvFormat {
            dtype: KvDtype::Fp8E4m3PerTensorScale,
            layout: KvLayout {
                dtype: DType::F8E4M3,
                ..bf16.layout
            },
            scales: Some(KvScaleHashes::of(k, v)),
            ..bf16
        }
    }

    /// Phase 6a S-16: the FP8 KV scales and the resolved RoPE parameters enter the namespace
    /// key. Identical scales give one namespace; a scale change in any layer of K or of V, or K
    /// and V swapped, gives another; a BF16 format keeps the golden key. Two identities that
    /// differ only in RoPE parameters give different keys, identical ones the same key. Breaks
    /// if blocks written under one set of scales or one RoPE configuration could be read under
    /// another.
    #[test]
    fn rope_and_scales_scope_the_namespace() {
        let bf16 = llama_format(16);
        assert_eq!(bf16.scales, None);
        assert_eq!(hex(&namespace_key(&model(1), &bf16, "").0), GOLDEN_NS);
        let ones = vec![1.0f32; 28];
        let ns = |f: &KvFormat| namespace_key(&model(1), f, "");
        let base = fp8_format(&ones, &ones);
        assert_eq!(
            ns(&base),
            ns(&fp8_format(&ones, &ones)),
            "same scales, same key"
        );
        assert_ne!(ns(&base), ns(&bf16), "FP8 pages are never read as BF16");
        assert_eq!(
            base.block_bytes() * 2,
            bf16.block_bytes(),
            "FP8 halves a block"
        );
        let mut k = ones.clone();
        k[27] = 0.5;
        assert_ne!(
            ns(&base),
            ns(&fp8_format(&k, &ones)),
            "one K scale changes the key"
        );
        assert_ne!(
            ns(&base),
            ns(&fp8_format(&ones, &k)),
            "one V scale changes the key"
        );
        let half = vec![0.5f32; 28];
        assert_ne!(
            ns(&fp8_format(&half, &ones)),
            ns(&fp8_format(&ones, &half)),
            "K and V scales are not interchangeable"
        );
        let tokens: Vec<u32> = (100..116).collect();
        assert_ne!(
            block_key(&ns(&base), None, &tokens),
            block_key(&ns(&fp8_format(&k, &ones)), None, &tokens),
            "the scales change every block key"
        );

        // RoPE (plan Task 27): the resolved RoPE parameters (`ModelArchConfig::rope_identity`)
        // scope the namespace, so a `model.rope_scaling` override — which leaves config.json,
        // hence `config_hash`, unchanged — never reuses blocks cached without it.
        let llama3 = r#"{"rotary_dim":128,"scaling":{"factor":32.0,"high_freq_factor":4.0,"low_freq_factor":1.0,"original_max_position_embeddings":8192,"type":"llama3"},"theta":500000.0}"#;
        let yarn16 = r#"{"rotary_dim":128,"scaling":{"attention_factor":1.2772588722239782,"beta_fast":32.0,"beta_slow":1.0,"factor":16.0,"original_max_position_embeddings":8192,"truncate":true,"type":"yarn"},"theta":500000.0}"#;
        let yarn8 = yarn16.replace(r#""factor":16.0"#, r#""factor":8.0"#);
        let with = |rope: &str| namespace_key(&model(1).with_rope(rope), &bf16, "");
        assert_eq!(with(llama3), with(llama3), "same RoPE, same key");
        assert_ne!(
            with(llama3),
            with(yarn16),
            "YaRN blocks are never read without it"
        );
        assert_ne!(with(yarn16), with(&yarn8), "nor under another factor");
        assert_ne!(
            with(yarn16),
            namespace_key(&model(1), &bf16, ""),
            "a resolved RoPE differs from none"
        );
        assert_ne!(
            block_key(&with(llama3), None, &tokens),
            block_key(&with(yarn16), None, &tokens),
            "RoPE changes every block key"
        );
        assert_eq!(
            model(1).with_rope(yarn16).rope_hash,
            *blake3::hash(yarn16.as_bytes()).as_bytes()
        );
    }
}
