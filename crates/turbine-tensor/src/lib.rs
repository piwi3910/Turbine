//! Device-memory handles and tensors (TS §6). Backends implement the safe `DeviceMemory` trait;
//! this crate never touches raw device memory itself.
pub mod buffer;
pub mod dtype;
pub mod host;
pub mod kv_view;
pub mod mapped;
pub mod pinned;
pub mod tensor;

pub use buffer::{
    DeviceBuffer, DeviceMemory, DevicePtr, DeviceSlice, HostStaging, MemInfo, MemoryError,
    StagingId, StreamRef,
};
pub use dtype::DType;
pub use kv_view::KvPoolView;
pub use mapped::{
    MAPPED_ABORT_HOST, MAPPED_ABORT_TIMEOUT, MappedCollectives, MappedDma, MappedHost, MappedKind,
    MappedReduce, MappedRegion, MappedStep,
};
pub use pinned::{
    CopyEngine, CopySource, CopyTarget, CopyTicket, PinnedBuffer, PinnedMemory, PinnedOwner,
};
pub use tensor::{Tensor, TensorView};
pub use turbine_core::types::DeviceId;
