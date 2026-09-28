//! Startup step 3 (P5 S-4, contract §16.3): the parallel plan, computed after device discovery
//! and the topology capture and before the kernel provider loads or the port binds, so an
//! impossible plan exits 2 with the key to change.
//!
//! - `execution.backend: cpu`: every replica runs on `execution.device` on the host
//!   ([`plan_execution_device`], no vendor); tensor parallelism is refused.
//! - GPU backends: `ParallelConfig::validate_devices` against the inventory, then either the
//!   single-GPU default (tp 1, dp 1, `devices: auto` → `execution.device`, as before Phase 5) or
//!   the topology-driven planner, which needs the model's head counts (read from `config.json`
//!   here, before the kernel provider) and a per-device budget for `tp: auto`.
//!
//! The plan is published as `turbine_parallel_info{tp,dp,backend,mode} 1` and as the `parallel`
//! object of `GET /turbine/v1/status`.

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use serde_json::{Value, json};
use turbine_core::config::{Config, DeviceSelection, SizeOrAuto};
use turbine_core::types::{DeviceId, Vendor};
use turbine_device::DeviceInventory;
use turbine_device::topology::TopologyGraph;
use turbine_distributed::plan::{ParallelPlan, plan, plan_execution_device};
use turbine_model::load_model_config;
use turbine_observability::MetricsRegistry;

/// Why startup cannot use the configured plan.
#[derive(Debug, PartialEq, Eq)]
pub enum PlanFailure {
    /// Impossible or unsupported plan: exit 2, before bind.
    Config(String),
    /// The planner needs the model's shape and `config.json` cannot be read: exit 1.
    Startup(String),
}

/// The device vendor of the registered execution backend `name` (`None` for the host `cpu`
/// backend or an unknown name, which `Config::validate_modules` already refused).
fn backend_vendor(name: &str) -> Option<Vendor> {
    let vendor = turbine_kernels::backends::registry().get(name)?.vendor();
    [Vendor::Amd, Vendor::Nvidia]
        .into_iter()
        .find(|v| v.as_str() == vendor)
}

/// What the configured collective backend serves, for `ParallelConfig::validate_devices`.
fn collective_vendors(name: &str) -> Option<&'static [Vendor]> {
    turbine_distributed::collective::registry()
        .get(name)
        .map(|b| b.vendors())
}

/// Computes the plan (module comment).
pub fn plan_for(
    config: &Config,
    inventory: &DeviceInventory,
    topology: &TopologyGraph,
) -> Result<ParallelPlan, PlanFailure> {
    let p = &config.parallel;
    let exec = &config.execution;
    let host = topology.node.hostname.as_str();
    let plan_error = |e: turbine_distributed::plan::PlanError| PlanFailure::Config(e.to_string());
    let Some(vendor) = backend_vendor(exec.backend.as_str()) else {
        return plan_execution_device(p, exec.device, None, host).map_err(plan_error);
    };
    let inv: Vec<(DeviceId, Vendor, Option<String>)> = inventory
        .devices
        .iter()
        .map(|d| (d.index, d.vendor, d.arch.clone()))
        .collect();
    p.validate_devices(&inv, collective_vendors(p.collective_backend.as_str()))
        .map_err(|e| PlanFailure::Config(e.to_string()))?;
    let single_default = p.tensor_parallel_size == SizeOrAuto::Size(1)
        && p.data_parallel_size == SizeOrAuto::Size(1)
        && p.devices == DeviceSelection::Auto;
    if single_default {
        return plan_execution_device(p, exec.device, Some(vendor), host).map_err(plan_error);
    }
    let shape = load_model_config(&config.model.path)
        .map_err(|e| PlanFailure::Startup(format!("model config: {e}")))?
        .shape();
    // Phase 3's device budget replaces this: total memory less the emergency reserve.
    let reserve = config.reliability.emergency_vram_reserve.0;
    let budget = |id: DeviceId| {
        inventory
            .devices
            .iter()
            .find(|d| d.index == id)
            .map_or(0, |d| d.memory.total_bytes.saturating_sub(reserve))
    };
    plan(inventory, topology, p, &shape, &budget).map_err(plan_error)
}

/// Refuses a plan this build cannot execute yet (exit 2, like any other unusable
/// configuration): tensor-parallel model execution and data-parallel replicas are not wired
/// into the engine yet.
pub fn check_executable(plan: &ParallelPlan) -> Result<(), PlanFailure> {
    if plan.tp > 1 {
        return Err(PlanFailure::Config(format!(
            "parallel.tensor_parallel_size: {} ranks planned; tensor-parallel model execution is \
             not available in this build",
            plan.tp
        )));
    }
    if plan.dp > 1 {
        return Err(PlanFailure::Config(format!(
            "parallel.data_parallel_size: {} replicas planned; data-parallel replicas are not \
             available in this build",
            plan.dp
        )));
    }
    Ok(())
}

/// The device of the single engine: rank 0 of replica 0.
pub fn engine_device(plan: &ParallelPlan) -> DeviceId {
    plan.groups[0].ranks[0].device
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct ParallelInfoLabels {
    tp: String,
    dp: String,
    backend: &'static str,
    mode: String,
}

/// `turbine_parallel_info{tp,dp,backend,mode}`, always 1.
pub fn register_info(reg: &MetricsRegistry, plan: &ParallelPlan) {
    let info = reg.register(
        "turbine_parallel_info",
        "The process's parallel plan (always 1)",
        Family::<ParallelInfoLabels, Gauge>::default(),
    );
    info.get_or_create(&ParallelInfoLabels {
        tp: plan.tp.to_string(),
        dp: plan.dp.to_string(),
        backend: plan.backend,
        mode: mode_str(plan),
    })
    .set(1);
}

/// `parallel.ranks.mode` as the configuration spells it (`local`, `static`).
fn mode_str(plan: &ParallelPlan) -> String {
    match serde_json::to_value(plan.mode) {
        Ok(Value::String(s)) => s,
        _ => format!("{:?}", plan.mode).to_lowercase(),
    }
}

/// The `parallel` object of `GET /turbine/v1/status`.
pub fn status(plan: &ParallelPlan) -> Value {
    json!({
        "tp": plan.tp,
        "dp": plan.dp,
        "backend": plan.backend,
        "mode": mode_str(plan),
        "groups": plan.groups.iter().map(|g| json!({
            "replica": g.replica.0,
            "ranks": g.ranks.iter().map(|r| json!({
                "rank": r.rank,
                "device": r.device.0,
                "host": r.host,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "plan_reasons": plan.reasons.iter().map(ToString::to_string).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use turbine_core::types::ReplicaId;
    use turbine_distributed::plan::{PlanReason, RankSlot, ReplicaGroup};

    use super::*;

    fn plan(tp: u32, dp: u32) -> ParallelPlan {
        ParallelPlan {
            tp,
            dp,
            backend: "host",
            mode: turbine_core::config::RankMode::Local,
            vendor: Some(Vendor::Amd),
            excluded_devices: Vec::new(),
            groups: vec![ReplicaGroup {
                replica: ReplicaId(0),
                ranks: vec![RankSlot {
                    rank: 0,
                    device: DeviceId(1),
                    host: "novanas".into(),
                }],
            }],
            reasons: vec![PlanReason::ExecutionDevice],
        }
    }

    /// The status object has the contract's keys, and the info gauge renders its labels.
    #[test]
    fn status_and_info_gauge() {
        let p = plan(1, 1);
        assert_eq!(
            status(&p),
            json!({"tp": 1, "dp": 1, "backend": "host", "mode": "local",
                   "groups": [{"replica": 0, "ranks": [{"rank": 0, "device": 1, "host": "novanas"}]}],
                   "plan_reasons": ["execution_device"]})
        );
        assert_eq!(engine_device(&p), DeviceId(1));
        let reg = MetricsRegistry::new();
        register_info(&reg, &p);
        let text = reg.render().unwrap();
        assert!(
            text.contains(r#"turbine_parallel_info{tp="1",dp="1",backend="host",mode="local"} 1"#),
            "{text}"
        );
        assert!(check_executable(&p).is_ok());
        for (tp, dp, key) in [
            (2, 1, "parallel.tensor_parallel_size"),
            (1, 2, "parallel.data_parallel_size"),
        ] {
            match check_executable(&plan(tp, dp)) {
                Err(PlanFailure::Config(m)) => assert!(m.starts_with(key), "{m}"),
                other => panic!("{tp}x{dp}: {other:?}"),
            }
        }
    }
}
