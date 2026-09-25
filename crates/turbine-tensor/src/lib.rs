//! Device-memory handles and tensors (TS §6). Backends implement the safe `DeviceMemory` trait;
//! this crate never touches raw device memory itself.
pub mod buffer;
pub mod dtype;
pub mod host;
pub mod tensor;

pub use buffer::{
    DeviceBuffer, DeviceMemory, DevicePtr, DeviceSlice, MemInfo, MemoryError, StreamRef,
};
pub use dtype::DType;
pub use tensor::{Tensor, TensorView};
pub use turbine_core::types::DeviceId;
