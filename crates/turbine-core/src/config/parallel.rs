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

use super::{ByteSize, ConfigError, HumanDuration, ModuleName, invalid};
use crate::types::{DeviceId, Vendor};

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct ParallelConfig {
    pub tensor_parallel_size: SizeOrAuto,
    pub data_parallel_size: SizeOrAuto,
    pub devices: DeviceSelection,
    /// A registered collective backend (extension point `collective_backend`: `host`,
    /// `hostmem`, `rccl`, `nccl`; checked by `Config::validate_modules`) or `auto`, the first
    /// registered backend serving the plan's vendor and rank mode (`host` when
    /// tensor_parallel_size is 1; `hostmem` for AMD in `local` mode, `rccl` in `static` mode).
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
    /// Pipeline stages (P5 S-10): contiguous layer ranges, one device each; > 1 needs
    /// tensor_parallel_size 1, expert_parallel_size 1 and `local` ranks (S-12).
    pub pipeline_parallel_size: SizeOrAuto,
    /// Expert-parallel ranks (P5 S-11): routed experts split over the group; > 1 needs a MoE
    /// model (the planner), tensor_parallel_size 1 or equal to it, no pipeline, `local` ranks.
    pub expert_parallel_size: SizeOrAuto,
    pub pipeline: PipelineConfig,
    pub expert: ExpertConfig,
    pub topology: TopologyConfig,
}

/// `parallel.pipeline`.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct PipelineConfig {
    /// Layers per stage, in stage order; `None` balances by cost. Its sum is checked against
    /// the model by the planner.
    pub layer_split: Option<Vec<u32>>,
    /// Micro-batches in flight; `auto` = the stage count.
    pub micro_batches: SizeOrAuto,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            layer_split: None,
            micro_batches: SizeOrAuto::Auto,
        }
    }
}

/// `parallel.expert`.
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields, default)]
pub struct ExpertConfig {
    pub placement: ExpertPlacementChoice,
}

/// `parallel.expert.placement`: `contiguous`, or the path of a YAML `{layer: [rank per expert]}`
/// file.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ExpertPlacementChoice {
    #[default]
    Contiguous,
    File(PathBuf),
}

impl Serialize for ExpertPlacementChoice {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            ExpertPlacementChoice::Contiguous => s.serialize_str("contiguous"),
            ExpertPlacementChoice::File(p) => p.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for ExpertPlacementChoice {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Ok(match s.as_str() {
            "contiguous" => ExpertPlacementChoice::Contiguous,
            "" => return Err(de::Error::custom("expected `contiguous` or a file path")),
            _ => ExpertPlacementChoice::File(PathBuf::from(s)),
        })
    }
}

/// `parallel.topology`.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct TopologyConfig {
    /// Measure each GPU's host link at startup (P5 S-13); `false` keeps nominal edges.
    pub measure_links: bool,
}

impl Default for TopologyConfig {
    fn default() -> Self {
        TopologyConfig {
            measure_links: true,
        }
    }
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
            pipeline_parallel_size: SizeOrAuto::Size(1),
            expert_parallel_size: SizeOrAuto::Size(1),
            pipeline: PipelineConfig::default(),
            expert: ExpertConfig::default(),
            topology: TopologyConfig::default(),
        }
    }
}

/// `parallel.collective`: communicator init and per-step collective bounds, and the size
/// threshold of the `hostmem` backend.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct CollectiveTimeouts {
    pub init_timeout: HumanDuration,
    pub op_timeout: HumanDuration,
    /// The largest message (nccl-tests bytes) the `hostmem` backend keeps on its own kernels;
    /// larger ones go to RCCL. `auto` (default): the measured per-op crossover.
    pub hostmem_max_bytes: ByteSizeOrAuto,
}

impl Default for CollectiveTimeouts {
    fn default() -> Self {
        CollectiveTimeouts {
            init_timeout: HumanDuration::from_secs(120),
            op_timeout: HumanDuration::from_secs(30),
            hostmem_max_bytes: ByteSizeOrAuto::Auto,
        }
    }
}

/// A byte size or `auto`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ByteSizeOrAuto {
    Auto,
    Bytes(ByteSize),
}

impl ByteSizeOrAuto {
    /// The explicit size in bytes, `None` for `auto`.
    pub fn fixed(self) -> Option<u64> {
        match self {
            ByteSizeOrAuto::Auto => None,
            ByteSizeOrAuto::Bytes(b) => Some(b.0),
        }
    }
}

impl Serialize for ByteSizeOrAuto {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            ByteSizeOrAuto::Auto => s.serialize_str("auto"),
            ByteSizeOrAuto::Bytes(b) => b.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for ByteSizeOrAuto {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl Visitor<'_> for V {
            type Value = ByteSizeOrAuto;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a byte size (e.g. \"256KiB\") or `auto`")
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<ByteSizeOrAuto, E> {
                Ok(ByteSizeOrAuto::Bytes(ByteSize(v)))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<ByteSizeOrAuto, E> {
                u64::try_from(v)
                    .map(|v| ByteSizeOrAuto::Bytes(ByteSize(v)))
                    .map_err(|_| E::custom(format!("invalid byte size {v}: must be non-negative")))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<ByteSizeOrAuto, E> {
                if v == "auto" {
                    Ok(ByteSizeOrAuto::Auto)
                } else {
                    v.parse().map(ByteSizeOrAuto::Bytes).map_err(E::custom)
                }
            }
        }
        d.deserialize_any(V)
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
    /// Devices one model instance spans: pipeline stages × max(tensor, expert) ranks; `None`
    /// while a size is `auto`.
    pub fn group_size(&self) -> Option<u32> {
        let tp = self.tensor_parallel_size.fixed()?;
        let ep = self.expert_parallel_size.fixed()?;
        let pp = self.pipeline_parallel_size.fixed()?;
        Some(pp * tp.max(ep))
    }

    /// The pipeline and expert sizes and the combinations Phase 5 supports (S-12): tp × dp;
    /// pp × dp with tp = ep = 1; ep × dp with tp ∈ {1, ep}; pp and ep in `local` ranks only.
    /// Anything else is refused naming the key and `combination_unsupported:<modes>`.
    fn validate_modes(&self) -> Result<(), ConfigError> {
        const PP: &str = "parallel.pipeline_parallel_size";
        const EP: &str = "parallel.expert_parallel_size";
        if let SizeOrAuto::Size(n) = self.pipeline_parallel_size
            && !(1..=64).contains(&n)
        {
            return Err(invalid(
                PP,
                format!("must be `auto` or between 1 and 64, got {n}"),
            ));
        }
        if let SizeOrAuto::Size(n) = self.expert_parallel_size
            && !(1..=64).contains(&n)
        {
            return Err(invalid(
                EP,
                format!("must be `auto` or between 1 and 64, got {n}"),
            ));
        }
        let tp = self.tensor_parallel_size.fixed();
        let pp = self.pipeline_parallel_size.fixed();
        let ep = self.expert_parallel_size.fixed();
        let unsupported = |key: &str, modes: &str, why: &str| {
            Err(invalid(
                key,
                format!("combination_unsupported:{modes}: {why}"),
            ))
        };
        if let Some(pp) = pp.filter(|&p| p > 1) {
            if tp != Some(1) {
                return unsupported(
                    PP,
                    "pp+tp",
                    "pipeline stages run one device each in Phase 5; set \
                     parallel.tensor_parallel_size: 1",
                );
            }
            if ep != Some(1) {
                return unsupported(
                    PP,
                    "pp+ep",
                    "pipeline and expert parallelism do not combine in Phase 5",
                );
            }
            if self.ranks.mode != RankMode::Local {
                return unsupported(PP, "pp+static", "pipeline stages run in `local` ranks only");
            }
            if let Some(split) = &self.pipeline.layer_split
                && split.len() != pp as usize
            {
                return Err(invalid(
                    "parallel.pipeline.layer_split",
                    format!("{} entries for {pp} pipeline stages", split.len()),
                ));
            }
            if let SizeOrAuto::Size(m) = self.pipeline.micro_batches
                && !(1..=4 * pp).contains(&m)
            {
                return Err(invalid(
                    "parallel.pipeline.micro_batches",
                    format!("must be `auto` or between 1 and {}, got {m}", 4 * pp),
                ));
            }
        }
        if let Some(split) = &self.pipeline.layer_split
            && split.contains(&0)
        {
            return Err(invalid(
                "parallel.pipeline.layer_split",
                "every stage needs at least one layer",
            ));
        }
        if let Some(ep) = ep.filter(|&e| e > 1) {
            if tp.is_some_and(|tp| tp != 1 && tp != ep) {
                return unsupported(
                    EP,
                    "ep+tp",
                    "expert parallelism runs with tensor_parallel_size 1 or equal to it",
                );
            }
            if self.ranks.mode != RankMode::Local {
                return unsupported(EP, "ep+static", "expert ranks run in `local` ranks only");
            }
        }
        Ok(())
    }

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
        self.validate_modes()?;
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
            if let Some(group) = self.group_size()
                && (distinct as u32) < group
            {
                return Err(invalid(
                    "parallel.devices",
                    format!(
                        "a model group of {group} devices (pipeline stages × tensor or expert \
                         ranks) needs {group} distinct devices, the list has {distinct}"
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
                if let (Some(group), Some(dp)) =
                    (self.group_size(), self.data_parallel_size.fixed())
                    && !self.allow_device_sharing
                    && list.len() as u32 != group * dp
                {
                    return Err(invalid(
                        "parallel.devices",
                        format!(
                            "lists {} devices; the plan needs {} (pipeline stages × tensor or \
                             expert ranks × data_parallel_size)",
                            list.len(),
                            group * dp
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
