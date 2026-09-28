//! Model identity and KV element format (contract §3.4): the inputs of the Phase 4 KV namespace
//! key (P4 S-1) and the fingerprint that tags every cached block (CONFLICT C-17).
//! Re-exported from `turbine_core::types`.

use serde::{Deserialize, Serialize};

/// What the KV cache of a model depends on: BLAKE3 of the model's `config.json` bytes and of its
/// safetensors index (or single-file header), and of the resolved RoPE parameters (P6a S-16).
/// Two checkpoints share KV only if all three match.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct ModelIdentity {
    pub config_hash: [u8; 32],
    pub weights_index_hash: [u8; 32],
    /// BLAKE3 of the canonical JSON of the resolved RoPE parameters
    /// (`turbine_model::ModelArchConfig::rope_identity`, set by [`ModelIdentity::with_rope`]):
    /// a `model.rope_scaling` override changes the rotary table without changing
    /// `config.json`, so `config_hash` alone would let blocks cached under one RoPE
    /// configuration be reused under another. [`ModelIdentity::from_bytes`] leaves it the hash
    /// of the empty string (no RoPE resolved).
    pub rope_hash: [u8; 32],
}

impl ModelIdentity {
    /// The identity of a checkpoint from the bytes of its `config.json` and of its safetensors
    /// index (`model.safetensors.index.json`, or a single file's header); no RoPE resolved yet.
    pub fn from_bytes(config_json: &[u8], weights_index: &[u8]) -> ModelIdentity {
        ModelIdentity {
            config_hash: *blake3::hash(config_json).as_bytes(),
            weights_index_hash: *blake3::hash(weights_index).as_bytes(),
            rope_hash: *blake3::hash(b"").as_bytes(),
        }
    }

    /// This identity under the resolved RoPE parameters `rope_identity` (canonical JSON).
    pub fn with_rope(self, rope_identity: &str) -> ModelIdentity {
        ModelIdentity {
            rope_hash: *blake3::hash(rope_identity.as_bytes()).as_bytes(),
            ..self
        }
    }

    /// BLAKE3(config_hash ‖ weights_index_hash).
    pub fn fingerprint(&self) -> ModelFingerprint {
        let mut h = blake3::Hasher::new();
        h.update(&self.config_hash);
        h.update(&self.weights_index_hash);
        ModelFingerprint(*h.finalize().as_bytes())
    }
}

/// One value naming a model's weights and configuration (TS §8 `ModelId`, CONFLICT C-17).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct ModelFingerprint(pub [u8; 32]);

/// Element format of stored KV. Only `Bf16` is produced before Phase 8a; the codes are the
/// Phase 7 TKV1 `kv_format` wire codes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum KvDtype {
    Bf16,
    Fp16,
    Fp8E4m3PerTensorScale,
    Fp8E4m3PerBlockScale,
}

impl KvDtype {
    /// Log / canonical-JSON spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            KvDtype::Bf16 => "bf16",
            KvDtype::Fp16 => "fp16",
            KvDtype::Fp8E4m3PerTensorScale => "fp8_e4m3_per_tensor_scale",
            KvDtype::Fp8E4m3PerBlockScale => "fp8_e4m3_per_block_scale",
        }
    }
    /// TKV1 `kv_format` code.
    pub fn wire_code(self) -> u16 {
        match self {
            KvDtype::Bf16 => 0,
            KvDtype::Fp16 => 1,
            KvDtype::Fp8E4m3PerTensorScale => 2,
            KvDtype::Fp8E4m3PerBlockScale => 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_depends_on_both_hashes() {
        let a = ModelIdentity {
            config_hash: [1; 32],
            weights_index_hash: [7; 32],
            rope_hash: [0; 32],
        };
        let b = ModelIdentity {
            weights_index_hash: [8; 32],
            ..a
        };
        let c = ModelIdentity {
            config_hash: [2; 32],
            ..a
        };
        assert_eq!(a.fingerprint(), a.fingerprint());
        assert_ne!(a.fingerprint(), b.fingerprint());
        assert_ne!(a.fingerprint(), c.fingerprint());
        let mut concat = [1u8; 64];
        concat[32..].fill(7);
        assert_eq!(a.fingerprint().0, *blake3::hash(&concat).as_bytes());
        let from = ModelIdentity::from_bytes(b"{}", b"index");
        assert_eq!(from.config_hash, *blake3::hash(b"{}").as_bytes());
        assert_eq!(from.weights_index_hash, *blake3::hash(b"index").as_bytes());
        assert_ne!(from, ModelIdentity::from_bytes(b"{} ", b"index"));
        assert_eq!(from.rope_hash, *blake3::hash(b"").as_bytes());
        let roped = from.with_rope(r#"{"theta":10000.0}"#);
        assert_eq!(
            roped.rope_hash,
            *blake3::hash(br#"{"theta":10000.0}"#).as_bytes()
        );
        assert_eq!(
            (roped.config_hash, roped.weights_index_hash),
            (from.config_hash, from.weights_index_hash)
        );
        assert_eq!(
            [KvDtype::Bf16, KvDtype::Fp8E4m3PerBlockScale].map(|d| (d.as_str(), d.wire_code())),
            [("bf16", 0), ("fp8_e4m3_per_block_scale", 3)]
        );
    }
}
