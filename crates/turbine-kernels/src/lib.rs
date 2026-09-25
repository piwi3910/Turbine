//! Kernel capability traits (one per op family), the kernel registry, the `cpu-reference`
//! provider and the runtime-loaded bindings to the vendor-neutral kernel C ABI
//! (`kernels/include/turbine_kernels.h`). With `turbine-device`, the only crate allowed to contain
//! `unsafe`; every block carries a `// SAFETY:` comment (checked by `tests/unsafe_isolation.rs`).
use std::path::PathBuf;

use turbine_tensor::MemoryError;

pub mod ops;

pub use ops::{
    ActivationConfig, ActivationContext, ActivationKernel, AttentionConfig, AttentionContext,
    AttentionKernel, AttentionKind, ElementwiseConfig, ElementwiseContext, ElementwiseKernel,
    EmbeddingConfig, EmbeddingContext, EmbeddingKernel, GemmConfig, GemmContext, GemmKernel,
    KernelProvider, NormConfig, NormContext, NormKernel, OpKind, ProviderId, RopeConfig,
    RopeContext, RopeKernel,
};

/// The kernel C ABI version this crate speaks; must equal `turbine_abi_version()` of the loaded
/// shim library and `TURBINE_ABI_VERSION` in the header exactly (contract §9.1).
pub const TURBINE_KERNELS_ABI_VERSION: u32 = 1;

/// Device error names after which the context is corrupted and every later call fails
/// (P3 "sticky" errors). Device messages start with the runtime's own error name (contract §9.2).
const STICKY_DEVICE_ERRORS: &[&str] = &[
    "hipErrorIllegalAddress",
    "hipErrorLaunchFailure",
    "hipErrorAssert",
];

/// Every failure of a kernel provider, the shim library or kernel selection (contract §7.1).
/// Codes −1…−5 of the C ABI map to the first five variants.
#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error("invalid argument: {message}")]
    InvalidArgument { message: String },
    #[error("unsupported configuration: {message}")]
    Unsupported { message: String },
    #[error("out of device memory: {message}")]
    OutOfMemory { message: String },
    /// The message starts with the device runtime's error name.
    #[error("{message}")]
    Device { message: String },
    #[error("kernel library error: {message}")]
    Library { message: String },
    #[error("cannot load {}: {detail}", path.display())]
    Load { path: PathBuf, detail: String },
    #[error("kernel ABI version mismatch: library {found}, expected {expected}")]
    AbiMismatch { expected: u32, found: u32 },
    #[error("kernel library backend {found}, configured {expected}")]
    BackendMismatch { expected: String, found: String },
    #[error("device arch {device_arch} not in library build archs {build_archs}")]
    ArchMismatch {
        device_arch: String,
        build_archs: String,
    },
    #[error("no kernel provider supports {op} {config}")]
    NoProvider { op: OpKind, config: String },
}

impl KernelError {
    /// A device error that leaves the context unusable (illegal address, launch failure, device
    /// assert).
    pub fn is_sticky(&self) -> bool {
        matches!(self, KernelError::Device { message }
            if STICKY_DEVICE_ERRORS.iter().any(|name| message.starts_with(name)))
    }

    pub fn is_oom(&self) -> bool {
        matches!(self, KernelError::OutOfMemory { .. })
    }
}

impl From<KernelError> for MemoryError {
    /// Out-of-memory keeps its kind but not a byte count (`KernelError` carries none, so
    /// `requested` is 0); allocation paths that know the size build `MemoryError` directly.
    fn from(e: KernelError) -> MemoryError {
        match e {
            KernelError::OutOfMemory { .. } => MemoryError::OutOfMemory { requested: 0 },
            KernelError::InvalidArgument { message } => MemoryError::InvalidArgument(message),
            KernelError::Unsupported { message } => MemoryError::Unsupported(message),
            other => MemoryError::Device {
                sticky: other.is_sticky(),
                message: other.to_string(),
            },
        }
    }
}

impl From<MemoryError> for KernelError {
    fn from(e: MemoryError) -> KernelError {
        match e {
            MemoryError::OutOfMemory { requested } => KernelError::OutOfMemory {
                message: format!("{requested} bytes requested"),
            },
            MemoryError::InvalidArgument(message) => KernelError::InvalidArgument { message },
            MemoryError::Unsupported(message) => KernelError::Unsupported { message },
            MemoryError::Device { message, .. } => KernelError::Device { message },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_follow_the_contract() {
        let cases = [
            (
                KernelError::AbiMismatch {
                    expected: 1,
                    found: 999,
                },
                "kernel ABI version mismatch: library 999, expected 1",
            ),
            (
                KernelError::ArchMismatch {
                    device_arch: "gfx1201".into(),
                    build_archs: "gfx942".into(),
                },
                "device arch gfx1201 not in library build archs gfx942",
            ),
            (
                KernelError::Load {
                    path: PathBuf::from("/nonexistent/libturbine_hip.so"),
                    detail: "no such file".into(),
                },
                "cannot load /nonexistent/libturbine_hip.so: no such file",
            ),
            (
                KernelError::BackendMismatch {
                    expected: "hip".into(),
                    found: "cuda".into(),
                },
                "kernel library backend cuda, configured hip",
            ),
            (
                KernelError::NoProvider {
                    op: OpKind::AttentionPrefill,
                    config: "head_dim=128 kv_heads=8".into(),
                },
                "no kernel provider supports attention_prefill head_dim=128 kv_heads=8",
            ),
            (
                KernelError::Device {
                    message: "hipErrorLaunchFailure: unspecified launch failure".into(),
                },
                "hipErrorLaunchFailure: unspecified launch failure",
            ),
        ];
        for (err, want) in cases {
            assert_eq!(err.to_string(), want);
        }
    }

    #[test]
    fn sticky_and_oom_classification() {
        let device = |m: &str| KernelError::Device { message: m.into() };
        assert!(device("hipErrorIllegalAddress: an illegal memory access").is_sticky());
        assert!(device("hipErrorAssert: device-side assert").is_sticky());
        assert!(!device("hipErrorNotReady: pending").is_sticky());
        assert!(
            !KernelError::Library {
                message: "hipErrorIllegalAddress".into()
            }
            .is_sticky()
        );
        assert!(
            KernelError::OutOfMemory {
                message: "4096 bytes".into()
            }
            .is_oom()
        );
        assert!(!device("hipErrorOutOfMemory").is_oom());
    }

    #[test]
    fn memory_error_round_trip_keeps_kind_and_stickiness() {
        let mem: MemoryError = KernelError::Device {
            message: "hipErrorIllegalAddress: fault".into(),
        }
        .into();
        assert!(
            matches!(mem, MemoryError::Device { sticky: true, ref message }
            if message == "hipErrorIllegalAddress: fault")
        );
        let back: KernelError = mem.into();
        assert!(back.is_sticky());

        let oom: KernelError = MemoryError::OutOfMemory { requested: 64 }.into();
        assert!(oom.is_oom());
        assert_eq!(oom.to_string(), "out of device memory: 64 bytes requested");
        assert!(matches!(
            MemoryError::from(oom),
            MemoryError::OutOfMemory { .. }
        ));
        assert!(matches!(
            MemoryError::from(KernelError::Unsupported {
                message: "d2d".into()
            }),
            MemoryError::Unsupported(m) if m == "d2d"
        ));
    }
}
