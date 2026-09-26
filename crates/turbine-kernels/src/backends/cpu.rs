//! `cpu`: the `cpu-reference` provider on host memory. The capacity is host `MemAvailable`,
//! else the configured device's total memory, else unbounded (logged).

use std::sync::Arc;

use turbine_core::registry::Module;
use turbine_core::types::MemoryKind;
use turbine_device::host_mem_available;
use turbine_tensor::DeviceMemory;
use turbine_tensor::host::HostMemory;

use super::{BackendError, BackendRequest, ExecutionBackend, OpenedBackend};
use crate::{ProviderId, cpu_reference_provider};

pub struct CpuBackend;

impl Module for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }
}

impl ExecutionBackend for CpuBackend {
    fn vendor(&self) -> &'static str {
        "cpu"
    }

    fn open(&self, req: &BackendRequest<'_>) -> Result<OpenedBackend, BackendError> {
        let device_total = req
            .inventory
            .devices
            .iter()
            .find(|d| d.index == req.device)
            .map(|d| d.memory.total_bytes);
        let capacity = host_mem_available(req.meminfo)
            .or(device_total)
            .unwrap_or_else(|| {
                tracing::warn!(
                    event = "memory_budget",
                    "host MemAvailable and device memory unknown; the cpu backend is unbounded"
                );
                u64::MAX
            });
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(req.device, capacity);
        tracing::info!(backend = "cpu", capacity, "kernel provider: cpu-reference");
        Ok(OpenedBackend {
            mem,
            providers: vec![cpu_reference_provider()],
            order: vec![ProviderId("cpu-reference")],
            memory_kind: MemoryKind::Dedicated,
            context: None,
            graphs: None,
            device: None,
            card: None,
        })
    }
}
