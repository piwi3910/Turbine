//! The module chosen at each extension point (Phase 2m S-1, contract §24): the names the
//! configuration is checked against before any port is bound ([`known_module_names`]) and the
//! choices `/turbine/v1/status` reports as `modules` ([`ModuleChoices`]).
//!
//! Every lane of Phase 2m moves one field of [`known_module_names`] from the fixed list below to
//! its registry's `names()`.

use std::sync::{LazyLock, OnceLock};

use serde::Serialize;
use turbine_core::config::ModuleNames;

/// Tool-call formats (`tool_format`): the names of `turbine_model::formats::registry`.
pub static TOOL_FORMATS: LazyLock<Vec<&'static str>> =
    LazyLock::new(|| turbine_model::formats::registry().names());
/// Execution backends (`execution_backend`): the names of `turbine_kernels::backends::registry()`.
pub fn backends() -> &'static [&'static str] {
    static NAMES: OnceLock<Vec<&'static str>> = OnceLock::new();
    NAMES.get_or_init(|| turbine_kernels::backends::registry().names())
}
/// Card profiles (`card_profile`): the names of `turbine_kernels::cards::registry()`.
pub fn card_profiles() -> &'static [&'static str] {
    static NAMES: OnceLock<Vec<&'static str>> = OnceLock::new();
    NAMES.get_or_init(|| turbine_kernels::cards::registry().names())
}
/// Scheduling policies (`scheduling_policy`): the names of `turbine_scheduler::policy::registry`.
pub static SCHEDULING_POLICIES: LazyLock<Vec<&'static str>> =
    LazyLock::new(|| turbine_scheduler::policy::registry().names());

/// Eviction policies (`eviction_policy`): the names of `turbine_kv::policy::registry`.
pub static EVICTION_POLICIES: LazyLock<Vec<&'static str>> =
    LazyLock::new(|| turbine_kv::policy::registry().names());

/// Collective backends (`collective_backend`, Phase 5): the names of
/// `turbine_distributed::collective::registry()`.
pub static COLLECTIVE_BACKENDS: LazyLock<Vec<&'static str>> =
    LazyLock::new(|| turbine_distributed::collective::registry().names());

/// Rank transports (`rank_transport`, Phase 5): the names of
/// `turbine_distributed::transport::registry()`.
pub static RANK_TRANSPORTS: LazyLock<Vec<&'static str>> =
    LazyLock::new(|| turbine_distributed::transport::registry().names());

/// DP router policies (`dp_router_policy`, Phase 5): the names of
/// `turbine_distributed::router::registry()`.
pub static ROUTER_POLICIES: LazyLock<Vec<&'static str>> =
    LazyLock::new(|| turbine_distributed::router::registry().names());

/// The registered module names, per configuration key, for `Config::validate_modules`.
pub fn known_module_names() -> ModuleNames<'static> {
    ModuleNames {
        tool_formats: &TOOL_FORMATS,
        backends: backends(),
        card_profiles: card_profiles(),
        scheduling_policies: &SCHEDULING_POLICIES,
        eviction_policies: &EVICTION_POLICIES,
        collective_backends: &COLLECTIVE_BACKENDS,
        rank_transports: &RANK_TRANSPORTS,
        router_policies: &ROUTER_POLICIES,
    }
}

/// `modules` of `GET /turbine/v1/status`: the module picked at each extension point.
#[derive(Clone, Debug, Serialize)]
pub struct ModuleChoices {
    pub family: String,
    /// `None` when tool calling is off.
    pub tool_format: Option<String>,
    pub weight_format: String,
    pub backend: String,
    /// `None` on a backend without card profiles (`cpu`).
    pub card_profile: Option<String>,
    pub scheduling_policy: String,
    /// `kv.policy` (Phase 4).
    pub eviction_policy: String,
}

impl ModuleChoices {
    /// Logs `event="module_selected"` for every extension point whose registry does not log
    /// its own selection, once at startup. `scheduling_policy` is logged by
    /// `turbine_scheduler::policy::registry().select` when the engine starts, `execution_backend`
    /// by `turbine_kernels::backends::registry().select`, `card_profile` by
    /// `turbine_kernels::cards::registry().select` when the backend has one, and
    /// `eviction_policy` by `turbine_kv::policy::registry().select` when the KV hierarchy
    /// starts.
    pub fn log(&self) {
        use turbine_core::registry::log_selected;
        log_selected("model_family", &self.family, "config.json architectures");
        log_selected(
            "tool_format",
            self.tool_format.as_deref().unwrap_or("none"),
            "model.tool_call_parser, else the family default",
        );
        log_selected(
            "weight_format",
            &self.weight_format,
            "config.json dtype and quantization_config",
        );
        if self.card_profile.is_none() {
            log_selected("card_profile", "none", "the backend has no card profiles");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbine_core::registry::valid_name;

    #[test]
    fn known_names_are_module_names() {
        let known = known_module_names();
        for names in [
            known.tool_formats,
            known.backends,
            known.card_profiles,
            known.scheduling_policies,
            known.eviction_policies,
            known.collective_backends,
            known.rank_transports,
            known.router_policies,
        ] {
            assert!(!names.is_empty());
            for name in names {
                assert!(valid_name(name), "{name}");
            }
        }
        assert!(known.scheduling_policies.contains(&"default"));
        assert!(known.eviction_policies.contains(&"cost_aware"));
        assert!(known.rank_transports.contains(&"tcp"));
        assert!(known.router_policies.contains(&"prefix_affinity"));
    }
}
