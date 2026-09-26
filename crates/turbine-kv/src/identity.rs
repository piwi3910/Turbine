//! KV identity (P4 S-1): namespace keys and 128-bit block keys forming a BLAKE3 hash chain.
//!
//! A block key covers the namespace (model identity, KV format, block size, cache salt), the
//! parent block's key and the block's token ids, so equal keys mean equal prefixes up to a hash
//! collision; the directory still compares stored tokens, so a collision is a miss.

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

/// Stored KV format: element type plus the per-token layout (which carries `block_tokens`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct KvFormat {
    pub dtype: KvDtype,
    pub layout: KvLayout,
}

/// Canonical namespace JSON. Field order is the sorted key order and must never change:
/// serde_json writes fields in declaration order, with no whitespace.
#[derive(Serialize)]
struct CanonicalNamespace<'a> {
    block_tokens: u32,
    cache_salt: &'a str,
    kv_format: CanonicalFormat,
    model_config_hash: String,
    weights_index_hash: String,
}

#[derive(Serialize)]
struct CanonicalFormat {
    dtype: &'static str,
    head_dim: u32,
    num_kv_heads: u32,
    num_layers: u32,
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
            num_kv_heads: fmt.layout.num_kv_heads,
            num_layers: fmt.layout.num_layers,
        },
        model_config_hash: hex(&id.config_hash),
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
        KvFormat {
            dtype: KvDtype::Bf16,
            layout: KvLayout {
                num_layers: 28,
                num_kv_heads: 8,
                head_dim: 128,
                dtype: DType::BF16,
                block_tokens,
            },
        }
    }

    fn model(config_byte: u8) -> ModelIdentity {
        ModelIdentity {
            config_hash: [config_byte; 32],
            weights_index_hash: [7; 32],
        }
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    // Golden values: a change here means every cached prefix silently changes identity.
    const GOLDEN_NS: &str = "7e1498499592da3bafd9f9da2af720288b56e58647a30c92555d8992abe96933";
    const GOLDEN_K0: &str = "048a689576719fe25e37707f36c80bcb";
    const GOLDEN_K1: &str = "1a566e6449ac3bd7d1fccfda42604a3d";

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
}
