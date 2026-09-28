//! `parallel` section (Phase 5): tensor/data-parallel sizes, device selection, collective
//! backend and timeouts, and the rank bootstrap. `validate` holds the static rules (exit 2
//! before device discovery); `validate_devices` the rules that need the device inventory.
//! Model-dependent rules (head divisibility, `auto` sizing) live in the P5 planner.

use std::collections::BTreeSet;
use std::fmt;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{ConfigError, HumanDuration, ModuleName, invalid};
use crate::types::{DeviceId, Vendor};

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct ParallelConfig {
    pub tensor_parallel_size: SizeOrAuto,
    pub data_parallel_size: SizeOrAuto,
    pub devices: DeviceSelection,
    /// A registered collective backend (extension point `collective_backend`: `rccl`, `nccl`,
    /// `host`; checked by `Config::validate_modules`) or `auto`, the backend serving the plan's
    /// vendor (`host` when tensor_parallel_size is 1).
    pub collective_backend: ModuleName,
    /// Explicit `libnccl.so.2`; failure to load it is fatal (exit 1).
    pub nccl_library: Option<PathBuf>,
    /// Explicit `librccl.so.1`; failure to load it is fatal (exit 1).
    pub rccl_library: Option<PathBuf>,
    /// Lets DP replicas share one device (testing); TP ranks never share.
    pub allow_device_sharing: bool,
    /// Leader → worker plan channel bound.
    pub plan_queue_depth: u32,
    /// A registered DP router policy (extension point `dp_router_policy`: `prefix_affinity`,
    /// `least_loaded`; checked by `Config::validate_modules`).
    pub router: ModuleName,
    pub collective: CollectiveTimeouts,
    pub ranks: RanksConfig,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        ParallelConfig {
            tensor_parallel_size: SizeOrAuto::Size(1),
            data_parallel_size: SizeOrAuto::Size(1),
            devices: DeviceSelection::Auto,
            collective_backend: ModuleName::fixed("auto"),
            nccl_library: None,
            rccl_library: None,
            allow_device_sharing: false,
            plan_queue_depth: 2,
            router: ModuleName::fixed("prefix_affinity"),
            collective: CollectiveTimeouts::default(),
            ranks: RanksConfig::default(),
        }
    }
}

/// `parallel.collective`: communicator init and per-step collective bounds.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct CollectiveTimeouts {
    pub init_timeout: HumanDuration,
    pub op_timeout: HumanDuration,
}

impl Default for CollectiveTimeouts {
    fn default() -> Self {
        CollectiveTimeouts {
            init_timeout: HumanDuration::from_secs(120),
            op_timeout: HumanDuration::from_secs(30),
        }
    }
}

/// `parallel.ranks`: how the ranks of one TP group find each other.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct RanksConfig {
    pub mode: RankMode,
    /// This process's rank (`static` only).
    pub rank: u32,
    /// Rank 0 listens here, the others connect (`static` only; required there).
    pub leader: Option<SocketAddr>,
    /// The device this rank process drives (`static` only; exactly one in Phase 5).
    pub local_devices: Vec<DeviceId>,
    /// A registered rank transport (extension point `rank_transport`: `tcp`; checked by
    /// `Config::validate_modules`) carrying the `static` bootstrap and step plans.
    pub transport: ModuleName,
}

impl Default for RanksConfig {
    fn default() -> Self {
        RanksConfig {
            mode: RankMode::Local,
            rank: 0,
            leader: None,
            local_devices: vec![DeviceId(0)],
            transport: ModuleName::fixed("tcp"),
        }
    }
}

/// A YAML integer or `auto`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SizeOrAuto {
    Auto,
    Size(u32),
}

impl SizeOrAuto {
    /// The explicit size, `None` for `auto`.
    pub fn fixed(self) -> Option<u32> {
        match self {
            SizeOrAuto::Auto => None,
            SizeOrAuto::Size(n) => Some(n),
        }
    }
}

impl Serialize for SizeOrAuto {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            SizeOrAuto::Auto => s.serialize_str("auto"),
            SizeOrAuto::Size(n) => s.serialize_u32(*n),
        }
    }
}

impl<'de> Deserialize<'de> for SizeOrAuto {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl Visitor<'_> for V {
            type Value = SizeOrAuto;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a non-negative integer or `auto`")
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<SizeOrAuto, E> {
                u32::try_from(v)
                    .map(SizeOrAuto::Size)
                    .map_err(|_| E::custom(format!("{v} is too large")))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<SizeOrAuto, E> {
                u64::try_from(v)
                    .map_err(|_| E::custom(format!("{v} is negative")))
                    .and_then(|v| self.visit_u64(v))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<SizeOrAuto, E> {
                if v == "auto" {
                    Ok(SizeOrAuto::Auto)
                } else {
                    Err(E::custom(format!(
                        "expected an integer or `auto`, got {v:?}"
                    )))
                }
            }
        }
        d.deserialize_any(V)
    }
}

/// `parallel.devices`: `auto` or a list of global device indices.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DeviceSelection {
    Auto,
    List(Vec<DeviceId>),
}

impl Serialize for DeviceSelection {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            DeviceSelection::Auto => s.serialize_str("auto"),
            DeviceSelection::List(list) => list.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for DeviceSelection {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = DeviceSelection;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("`auto` or a list of device indices")
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<DeviceSelection, E> {
                if v == "auto" {
                    Ok(DeviceSelection::Auto)
                } else {
                    Err(E::custom(format!(
                        "expected `auto` or a list of device indices, got {v:?}"
                    )))
                }
            }
            fn visit_seq<A: de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<DeviceSelection, A::Error> {
                let mut list = Vec::new();
                while let Some(id) = seq.next_element::<DeviceId>()? {
                    list.push(id);
                }
                Ok(DeviceSelection::List(list))
            }
        }
        d.deserialize_any(V)
    }
}

/// `parallel.ranks.mode`.
#[derive(Deserialize, Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RankMode {
    /// One thread per local device in one process.
    Local,
    /// One process per rank, joined over TCP to the leader.
    Static,
}

fn check_duration(
    key: &str,
    value: HumanDuration,
    min: Duration,
    max: Duration,
    range: &str,
) -> Result<(), ConfigError> {
    if value.0 < min || value.0 > max {
        return Err(invalid(
            key,
            format!("must be between {range}, got {value}"),
        ));
    }
    Ok(())
}

impl ParallelConfig {
    /// Static rules (exit 2 before device discovery).
    pub fn validate(&self) -> Result<(), ConfigError> {
        if let SizeOrAuto::Size(n) = self.tensor_parallel_size
            && (!(1..=8).contains(&n) || !n.is_power_of_two())
        {
            return Err(invalid(
                "parallel.tensor_parallel_size",
                format!("must be `auto` or a power of two between 1 and 8, got {n}"),
            ));
        }
        if let SizeOrAuto::Size(n) = self.data_parallel_size
            && !(1..=64).contains(&n)
        {
            return Err(invalid(
                "parallel.data_parallel_size",
                format!("must be `auto` or between 1 and 64, got {n}"),
            ));
        }
        if let DeviceSelection::List(list) = &self.devices {
            if list.is_empty() {
                return Err(invalid("parallel.devices", "must not be an empty list"));
            }
            let distinct = list.iter().collect::<BTreeSet<_>>().len();
            if !self.allow_device_sharing && distinct != list.len() {
                return Err(invalid(
                    "parallel.devices",
                    "lists a device twice; only DP replicas may share a device, with \
                     parallel.allow_device_sharing: true",
                ));
            }
            if let Some(tp) = self.tensor_parallel_size.fixed()
                && (distinct as u32) < tp
            {
                return Err(invalid(
                    "parallel.devices",
                    format!(
                        "a TP group of {tp} needs {tp} distinct devices, the list has {distinct}"
                    ),
                ));
            }
        }
        if !(1..=16).contains(&self.plan_queue_depth) {
            return Err(invalid(
                "parallel.plan_queue_depth",
                format!("must be between 1 and 16, got {}", self.plan_queue_depth),
            ));
        }
        check_duration(
            "parallel.collective.init_timeout",
            self.collective.init_timeout,
            Duration::from_secs(1),
            Duration::from_secs(30 * 60),
            "1s and 30m",
        )?;
        check_duration(
            "parallel.collective.op_timeout",
            self.collective.op_timeout,
            Duration::from_millis(100),
            Duration::from_secs(10 * 60),
            "100ms and 10m",
        )?;
        if self.ranks.mode == RankMode::Static {
            let tp = match self.tensor_parallel_size.fixed() {
                Some(tp) if tp > 1 => tp,
                _ => {
                    return Err(invalid(
                        "parallel.ranks.mode",
                        "static requires an explicit parallel.tensor_parallel_size greater than 1",
                    ));
                }
            };
            if self.data_parallel_size != SizeOrAuto::Size(1) {
                return Err(invalid(
                    "parallel.ranks.mode",
                    "static requires parallel.data_parallel_size: 1",
                ));
            }
            if self.ranks.leader.is_none() {
                return Err(invalid(
                    "parallel.ranks.leader",
                    "is required when parallel.ranks.mode is static",
                ));
            }
            if self.ranks.rank >= tp {
                return Err(invalid(
                    "parallel.ranks.rank",
                    format!("must be below the world size {tp}, got {}", self.ranks.rank),
                ));
            }
            if self.ranks.local_devices.len() != 1 {
                return Err(invalid(
                    "parallel.ranks.local_devices",
                    format!(
                        "must list exactly one device per rank process in static mode, got {}",
                        self.ranks.local_devices.len()
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Rules that need the device inventory: `(index, vendor, architecture)` per device.
    /// `backend_vendors` is what the configured collective backend serves: `None` for `auto`,
    /// `Some(&[])` for a host-memory backend, else the device vendors of the registered module.
    pub fn validate_devices(
        &self,
        inv: &[(DeviceId, Vendor, Option<String>)],
        backend_vendors: Option<&[Vendor]>,
    ) -> Result<(), ConfigError> {
        let vendor_of = |id: DeviceId| inv.iter().find(|(d, _, _)| *d == id).map(|(_, v, _)| *v);
        let tp = self.tensor_parallel_size.fixed();
        let selected: Vec<Vendor> = match &self.devices {
            DeviceSelection::Auto => inv.iter().map(|(_, v, _)| *v).collect(),
            DeviceSelection::List(list) => {
                let mut vendors = Vec::with_capacity(list.len());
                for id in list {
                    match vendor_of(*id) {
                        Some(v) => vendors.push(v),
                        None => {
                            return Err(invalid(
                                "parallel.devices",
                                format!("device {} is not in the inventory", id.0),
                            ));
                        }
                    }
                }
                if vendors.iter().any(|v| *v != vendors[0]) {
                    return Err(invalid(
                        "parallel.devices",
                        "mixes vendors; a plan uses one vendor (vendor-mixed plans arrive with \
                         phase 7)",
                    ));
                }
                if let (Some(tp), Some(dp)) = (tp, self.data_parallel_size.fixed())
                    && !self.allow_device_sharing
                    && list.len() as u32 != tp * dp
                {
                    return Err(invalid(
                        "parallel.devices",
                        format!(
                            "lists {} devices, tensor_parallel_size × data_parallel_size = {}",
                            list.len(),
                            tp * dp
                        ),
                    ));
                }
                vendors
            }
        };
        // An explicit backend must serve the selected devices: a host-memory backend (no device
        // vendors) only a plan without tensor parallelism, a device backend only its vendors.
        let Some(serves) = backend_vendors else {
            return Ok(());
        };
        let name = self.collective_backend.as_str();
        let key = "parallel.collective_backend";
        if serves.is_empty() {
            if !selected.is_empty() && tp != Some(1) {
                return Err(invalid(
                    key,
                    format!(
                        "`{name}` is a host-memory reference backend for tensor_parallel_size \
                         1 only; GPUs with tensor parallelism need a device backend"
                    ),
                ));
            }
            return Ok(());
        }
        if let Some(v) = selected.iter().find(|v| !serves.contains(v)) {
            return Err(invalid(
                key,
                format!("`{name}` cannot drive {v:?} devices; use a backend for them or auto"),
            ));
        }
        Ok(())
    }
}
