//! Per-op provider and implementation selection at startup (P1 S-6, Phase 2m S-5): for every
//! distinct op config the model needs, the first provider in the configured order that supports
//! it wins. A provider that enumerates its implementations (kernel ABI v2.4) gets one chosen from
//! the card profile's preference order (per routed-row tier for `moe_experts`) and is bound to
//! it, so every call runs exactly that implementation; any other provider keeps its own choice
//! (`reason_code` `provider_internal`). Every choice is logged (`event="kernel_selected"`) with
//! its reason and reason code and exported as
//! `turbine_kernel_provider_selected{op,provider,impl} 1`; a config no provider supports fails
//! startup, never first use.
use std::collections::HashMap;
use std::sync::Arc;

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use turbine_observability::MetricsRegistry;

use crate::KernelError;
use crate::cards::CardProfile;
use crate::ops::{
    ActivationConfig, ActivationKernel, AddRmsnormConfig, AddRmsnormKernel, AttentionConfig,
    AttentionKernel, ElementwiseConfig, ElementwiseKernel, EmbeddingConfig, EmbeddingKernel,
    GemmConfig, GemmKernel, ImplChoice, ImplInfo, KernelProvider, KvCopyConfig, KvCopyKernel,
    KvTranscodeConfig, KvTranscodeKernel, LogitsReduceConfig, LogitsReduceKernel, MoeExpertsConfig,
    MoeKernel, MoeRouteConfig, NormConfig, NormKernel, OpKind, ProviderId, QGemmConfig,
    QGemmKernel, QuantizeActConfig, QuantizeActKernel, RmsnormShardedConfig, RopeConfig,
    RopeKernel, RowSumsqConfig, RowTier, ShardedNormKernel,
};

/// `reason_code` of a selection: the card profile's first listed implementation (that the
/// library has) supports the config.
pub const PROFILE_PREFERRED: &str = "profile_preferred";
/// `reason_code`: a later implementation of the profile's order (or one it does not list) runs,
/// because the preferred ones are missing from the library or refuse the config.
pub const PROFILE_FALLBACK: &str = "profile_fallback";
/// `reason_code`: the card profile lists no order for the op; the first supporting implementation
/// in library order runs.
pub const LIBRARY_ORDER: &str = "library_order";
/// `reason_code`: the provider does not enumerate implementations (the CPU reference, a kernel
/// library of ABI minor 3 or earlier) or there is no card profile; it chooses per call itself.
pub const PROVIDER_INTERNAL: &str = "provider_internal";

/// Labels of `turbine_kernel_provider_selected`.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct SelectedLabels {
    pub op: String,
    pub provider: String,
    pub r#impl: String,
}

/// The kernel crate's metrics (contract §17).
#[derive(Clone, Debug)]
pub struct KernelMetrics {
    /// `turbine_kernel_provider_selected{op,provider,impl}` = 1 per chosen triple.
    pub provider_selected: Family<SelectedLabels, Gauge>,
}

impl KernelMetrics {
    pub fn register(reg: &MetricsRegistry) -> KernelMetrics {
        KernelMetrics {
            provider_selected: reg.register(
                "turbine_kernel_provider_selected",
                "Kernel provider and implementation selected per op; 1 per chosen triple",
                Family::default(),
            ),
        }
    }
}

/// The typed config of one op; the variant fixes the op family (attention's `kind` picks
/// prefill or decode, contiguous or paged).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OpConfig {
    Gemm(GemmConfig),
    Attention(AttentionConfig),
    Rmsnorm(NormConfig),
    Rope(RopeConfig),
    SiluMul(ActivationConfig),
    Embedding(EmbeddingConfig),
    Add(ElementwiseConfig),
    CopyBlocks(KvCopyConfig),
    MoeRoute(MoeRouteConfig),
    MoeExperts(MoeExpertsConfig),
    /// ABI v2.1; a shim library without the symbols has no provider for it.
    AddRmsnorm(AddRmsnormConfig),
    /// ABI v2.1; a shim library without the symbols has no provider for it.
    LogitsReduce(LogitsReduceConfig),
    /// ABI v2.6; a shim library without the group has no provider for it.
    RowSumsq(RowSumsqConfig),
    /// ABI v2.6; a shim library without the group has no provider for it.
    RmsnormSharded(RmsnormShardedConfig),
    /// ABI v2.9; a shim library without the group has no provider for it.
    QGemm(QGemmConfig),
    /// ABI v2.9; a shim library without the group has no provider for it.
    QuantizeAct(QuantizeActConfig),
    /// ABI v2.11; a shim library without the group has no provider for it.
    KvTranscode(KvTranscodeConfig),
}

impl OpConfig {
    pub fn op(&self) -> OpKind {
        match self {
            OpConfig::Gemm(_) => OpKind::Gemm,
            OpConfig::Attention(cfg) => cfg.op(),
            OpConfig::Rmsnorm(_) => OpKind::Rmsnorm,
            OpConfig::Rope(_) => OpKind::Rope,
            OpConfig::SiluMul(_) => OpKind::SiluMul,
            OpConfig::Embedding(_) => OpKind::Embedding,
            OpConfig::Add(_) => OpKind::Add,
            OpConfig::CopyBlocks(_) => OpKind::CopyBlocks,
            OpConfig::MoeRoute(_) => OpKind::MoeRoute,
            OpConfig::MoeExperts(_) => OpKind::MoeExperts,
            OpConfig::AddRmsnorm(_) => OpKind::AddRmsnorm,
            OpConfig::LogitsReduce(_) => OpKind::LogitsReduce,
            OpConfig::RowSumsq(_) => OpKind::RowSumsq,
            OpConfig::RmsnormSharded(_) => OpKind::RmsnormSharded,
            OpConfig::QGemm(_) => OpKind::QGemm,
            OpConfig::QuantizeAct(_) => OpKind::QuantizeAct,
            OpConfig::KvTranscode(_) => OpKind::KvTranscode,
        }
    }

    /// The config's log and failure-message form, e.g. `head_dim=128 kv_heads=8 dtype=bf16 …`.
    pub fn render(&self) -> String {
        match self {
            OpConfig::Gemm(cfg) => cfg.to_string(),
            OpConfig::Attention(cfg) => cfg.to_string(),
            OpConfig::Rmsnorm(cfg) => cfg.to_string(),
            OpConfig::Rope(cfg) => cfg.to_string(),
            OpConfig::SiluMul(cfg) => cfg.to_string(),
            OpConfig::Embedding(cfg) => cfg.to_string(),
            OpConfig::Add(cfg) => cfg.to_string(),
            OpConfig::CopyBlocks(cfg) => cfg.to_string(),
            OpConfig::MoeRoute(cfg) => cfg.to_string(),
            OpConfig::MoeExperts(cfg) => cfg.to_string(),
            OpConfig::AddRmsnorm(cfg) => cfg.to_string(),
            OpConfig::LogitsReduce(cfg) => cfg.to_string(),
            OpConfig::RowSumsq(cfg) => cfg.to_string(),
            OpConfig::RmsnormSharded(cfg) => cfg.to_string(),
            OpConfig::QGemm(cfg) => cfg.to_string(),
            OpConfig::QuantizeAct(cfg) => cfg.to_string(),
            OpConfig::KvTranscode(cfg) => cfg.to_string(),
        }
    }

    /// The implementation name `provider` would run for this config, or `None` when it lacks
    /// the op family or its `supports()` is false: the provider's own choice (for a shim
    /// library, `turbine_<op>_impl`).
    pub fn probe(&self, provider: &dyn KernelProvider) -> Option<String> {
        match self {
            OpConfig::Gemm(cfg) => provider
                .gemm()
                .filter(|k| k.supports(cfg))
                .map(|k| k.implementation(cfg)),
            OpConfig::Attention(cfg) => provider
                .attention()
                .filter(|k| k.supports(cfg))
                .map(|k| k.implementation(cfg)),
            OpConfig::Rmsnorm(cfg) => provider
                .norm()
                .filter(|k| k.supports(cfg))
                .map(|k| k.implementation(cfg)),
            OpConfig::Rope(cfg) => provider
                .rope()
                .filter(|k| k.supports(cfg))
                .map(|k| k.implementation(cfg)),
            OpConfig::SiluMul(cfg) => provider
                .activation()
                .filter(|k| k.supports(cfg))
                .map(|k| k.implementation(cfg)),
            OpConfig::Embedding(cfg) => provider
                .embedding()
                .filter(|k| k.supports(cfg))
                .map(|k| k.implementation(cfg)),
            OpConfig::Add(cfg) => provider
                .elementwise()
                .filter(|k| k.supports(cfg))
                .map(|k| k.implementation(cfg)),
            OpConfig::CopyBlocks(cfg) => provider
                .kv_copy()
                .filter(|k| k.supports(cfg))
                .map(|k| k.implementation(cfg)),
            OpConfig::MoeRoute(cfg) => provider
                .moe()
                .filter(|k| k.supports_route(cfg))
                .map(|k| k.implementation_route(cfg)),
            OpConfig::MoeExperts(cfg) => provider
                .moe()
                .filter(|k| k.supports_experts(cfg))
                .map(|k| k.implementation_experts(cfg)),
            OpConfig::AddRmsnorm(cfg) => provider
                .add_rmsnorm()
                .filter(|k| k.supports(cfg))
                .map(|k| k.implementation(cfg)),
            OpConfig::LogitsReduce(cfg) => provider
                .logits_reduce()
                .filter(|k| k.supports(cfg))
                .map(|k| k.implementation(cfg)),
            OpConfig::RowSumsq(cfg) => provider
                .sharded_norm()
                .filter(|k| k.supports_row_sumsq(cfg))
                .map(|k| k.implementation_row_sumsq(cfg)),
            OpConfig::RmsnormSharded(cfg) => provider
                .sharded_norm()
                .filter(|k| k.supports_rmsnorm_sharded(cfg))
                .map(|k| k.implementation_rmsnorm_sharded(cfg)),
            OpConfig::QGemm(cfg) => provider
                .qgemm()
                .filter(|k| k.supports(cfg))
                .map(|k| k.implementation(cfg)),
            OpConfig::QuantizeAct(cfg) => provider
                .quantize_act()
                .filter(|k| k.supports(cfg))
                .map(|k| k.implementation(cfg)),
            OpConfig::KvTranscode(cfg) => provider
                .kv_transcode()
                .filter(|k| k.supports(cfg))
                .map(|k| k.implementation(cfg)),
        }
    }

    /// True when `provider` has the op family and supports this config. Callers use it before
    /// `KernelRegistry::build` to decide whether an optional op (the v2.1 `add_rmsnorm` and
    /// `logits_reduce`) joins the requirements or the model falls back to the v2 ops.
    pub fn supported_by(&self, provider: &dyn KernelProvider) -> bool {
        self.probe(provider).is_some()
    }
}

/// One op the model needs at startup, with its rendered config.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct OpRequirement {
    pub op: OpKind,
    pub config: String,
    pub spec: OpConfig,
}

impl From<OpConfig> for OpRequirement {
    fn from(spec: OpConfig) -> OpRequirement {
        OpRequirement {
            op: spec.op(),
            config: spec.render(),
            spec,
        }
    }
}

/// One selection as logged: which provider and implementation serve `op` at `config`, and why.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Selection {
    pub op: OpKind,
    pub config: String,
    pub provider: ProviderId,
    /// The implementation that runs (the first tier's for a tiered op).
    pub implementation: String,
    pub reason: String,
    /// The implementation's family (`hipblaslt`, `ck`, `turbine_hip`); the provider id for a
    /// provider that chooses internally.
    pub impl_provider: String,
    /// [`PROFILE_PREFERRED`], [`PROFILE_FALLBACK`], [`LIBRARY_ORDER`] or [`PROVIDER_INTERNAL`].
    pub reason_code: &'static str,
    /// Per routed-row tier `(max_rows, implementation)` of a tiered op (`moe_experts`); empty
    /// otherwise.
    pub tiers: Vec<(Option<u32>, String)>,
}

/// The implementation(s) chosen on one provider.
struct Chosen {
    choice: ImplChoice,
    implementation: String,
    impl_provider: String,
    reason_code: &'static str,
    reason: String,
    tiers: Vec<(Option<u32>, String)>,
}

/// One candidate of a preference order: an implementation of the library, or a listed name the
/// library lacks.
enum Candidate<'a> {
    Present(&'a ImplInfo),
    Missing(&'a str),
}

/// `impls` in the order `order` lists them (names the library lacks kept as missing), then the
/// implementations `order` does not list, in library order.
fn ordered<'a>(order: &'a [&'a str], impls: &'a [ImplInfo]) -> Vec<Candidate<'a>> {
    let mut out: Vec<Candidate<'a>> = order
        .iter()
        .map(|name| match impls.iter().find(|i| i.name == *name) {
            Some(info) => Candidate::Present(info),
            None => Candidate::Missing(name),
        })
        .collect();
    out.extend(
        impls
            .iter()
            .filter(|i| !order.contains(&i.name.as_str()))
            .map(Candidate::Present),
    );
    out
}

/// The first candidate of `order` (then library order) `provider` supports `spec` with, and the
/// candidates refused before it (`<name> (not in library)` / `<name> (unsupported)`). `Err`
/// lists every refusal.
fn first_supporting<'a>(
    provider: &dyn KernelProvider,
    spec: &OpConfig,
    order: &'a [&'a str],
    impls: &'a [ImplInfo],
    rows: Option<u32>,
) -> Result<(&'a ImplInfo, Vec<String>), Vec<String>> {
    let mut refused = Vec::new();
    for candidate in ordered(order, impls) {
        match candidate {
            Candidate::Missing(name) => refused.push(format!("{name} (not in library)")),
            Candidate::Present(info) => {
                if provider.implementation_supports(spec, info.index, rows) {
                    return Ok((info, refused));
                }
                refused.push(format!("{} (unsupported)", info.name));
            }
        }
    }
    Err(refused)
}

/// Chooses among the implementations `provider` enumerates for `spec` by `card`'s preference:
/// one per routed-row tier when the profile has row tiers for the op, else one. `Err` names
/// every implementation and its refusal.
fn choose(
    provider: &dyn KernelProvider,
    spec: &OpConfig,
    impls: &[ImplInfo],
    card: &CardProfile,
) -> Result<Chosen, String> {
    let pref = card.preference(spec.op());
    let row_tiers = pref.map_or(&[][..], |p| p.row_tiers);
    if row_tiers.is_empty() {
        let order = pref.map_or(&[][..], |p| p.order);
        let (info, refused) = first_supporting(provider, spec, order, impls, None)
            .map_err(|refused| refused.join(", "))?;
        let (reason_code, reason) = if order.is_empty() {
            (
                LIBRARY_ORDER,
                format!(
                    "card profile {} lists no order for {}: first supporting implementation in \
                     library order",
                    card.name,
                    spec.op()
                ),
            )
        } else if refused.is_empty() {
            (
                PROFILE_PREFERRED,
                format!("card profile {} prefers {}", card.name, info.name),
            )
        } else {
            (
                PROFILE_FALLBACK,
                format!(
                    "card profile {} order; refused: {}",
                    card.name,
                    refused.join(", ")
                ),
            )
        };
        return Ok(Chosen {
            choice: ImplChoice::Single(info.index),
            implementation: info.name.clone(),
            impl_provider: info.provider.clone(),
            reason_code,
            reason,
            tiers: Vec::new(),
        });
    }
    // One implementation per tier, probed at the tier's bound (the open tier at 4 × the largest
    // bound).
    let open_rows = row_tiers
        .iter()
        .filter_map(|t| t.max_rows)
        .max()
        .map(|m| m.saturating_mul(4));
    let mut tiers = Vec::with_capacity(row_tiers.len());
    let mut picked: Vec<&ImplInfo> = Vec::with_capacity(row_tiers.len());
    let mut notes = Vec::new();
    for tier in row_tiers {
        let rows = tier.max_rows.or(open_rows);
        let bound = tier
            .max_rows
            .map_or_else(|| "open tier".to_string(), |m| format!("<= {m} rows"));
        let (info, refused) = first_supporting(provider, spec, tier.order, impls, rows)
            .map_err(|refused| format!("{bound}: {}", refused.join(", ")))?;
        if !refused.is_empty() {
            notes.push(format!("{bound}: refused {}", refused.join(", ")));
        }
        tiers.push(RowTier {
            max_rows: tier.max_rows,
            index: info.index,
        });
        picked.push(info);
    }
    let named: Vec<(Option<u32>, String)> = row_tiers
        .iter()
        .zip(&picked)
        .map(|(t, info)| (t.max_rows, info.name.clone()))
        .collect();
    let listing: Vec<String> = named
        .iter()
        .map(|(max, name)| match max {
            Some(m) => format!("<= {m} rows {name}"),
            None => format!("above {name}"),
        })
        .collect();
    let (reason_code, reason) = if notes.is_empty() {
        (
            PROFILE_PREFERRED,
            format!(
                "card profile {} row tiers: {}",
                card.name,
                listing.join(", ")
            ),
        )
    } else {
        (
            PROFILE_FALLBACK,
            format!(
                "card profile {} row tiers: {}; {}",
                card.name,
                listing.join(", "),
                notes.join("; ")
            ),
        )
    };
    Ok(Chosen {
        choice: ImplChoice::ByRows(tiers),
        implementation: picked[0].name.clone(),
        impl_provider: picked[0].provider.clone(),
        reason_code,
        reason,
        tiers: named,
    })
}

/// The startup-resolved map from op config to the provider that runs it (bound to the chosen
/// implementation when the provider enumerates them).
pub struct KernelRegistry {
    chosen: HashMap<OpConfig, Arc<dyn KernelProvider>>,
    selections: Vec<Selection>,
}

impl KernelRegistry {
    /// Picks, per distinct requirement, the first provider in `order` that supports it: with a
    /// `card`, a provider that enumerates its implementations through
    /// [`KernelProvider::implementation_supports`] in the profile's preference order, bound to
    /// the choice with [`KernelProvider::bind`]; any other provider (and every provider without
    /// a card) through its family's `supports()`, keeping its own choice. Logs op, config,
    /// provider, implementation, reason and reason code and sets the selection gauge. Returns
    /// [`KernelError::NoProvider`] (with each enumerated implementation's refusal) for the first
    /// requirement nothing supports. An id in `order` with no registered provider is skipped.
    pub fn build(
        providers: Vec<Arc<dyn KernelProvider>>,
        order: &[ProviderId],
        reqs: &[OpRequirement],
        metrics: &KernelMetrics,
        card: Option<&CardProfile>,
    ) -> Result<KernelRegistry, KernelError> {
        let mut chosen: HashMap<OpConfig, Arc<dyn KernelProvider>> = HashMap::new();
        let mut selections = Vec::new();
        for req in reqs {
            if chosen.contains_key(&req.spec) {
                continue;
            }
            let mut unsupported: Vec<&'static str> = Vec::new();
            let mut refusals: Vec<String> = Vec::new();
            let mut picked = None;
            for id in order {
                let Some(provider) = providers.iter().find(|p| p.id() == *id) else {
                    continue;
                };
                let impls = card.map_or_else(Vec::new, |_| provider.implementations(req.op));
                if let (Some(card), false) = (card, impls.is_empty()) {
                    match choose(provider.as_ref(), &req.spec, &impls, card) {
                        Ok(c) => {
                            let run = provider
                                .bind(&req.spec, &c.choice)
                                .unwrap_or_else(|| Arc::clone(provider));
                            picked = Some((provider.id(), run, c));
                            break;
                        }
                        Err(detail) => {
                            unsupported.push(id.0);
                            refusals.push(format!("{}: {detail}", id.0));
                        }
                    }
                    continue;
                }
                match req.spec.probe(provider.as_ref()) {
                    Some(implementation) => {
                        let c = Chosen {
                            choice: ImplChoice::Single(0),
                            implementation,
                            impl_provider: id.0.to_string(),
                            reason_code: PROVIDER_INTERNAL,
                            reason: String::new(),
                            tiers: Vec::new(),
                        };
                        picked = Some((provider.id(), Arc::clone(provider), c));
                        break;
                    }
                    None => unsupported.push(id.0),
                }
            }
            let Some((provider_id, run, mut c)) = picked else {
                return Err(KernelError::NoProvider {
                    op: req.op,
                    config: req.config.clone(),
                    detail: refusals.join("; "),
                });
            };
            let order_note = if unsupported.is_empty() {
                "first provider in order supports config".to_string()
            } else {
                format!(
                    "first provider in order supporting config; unsupported by: {}",
                    unsupported.join(", ")
                )
            };
            c.reason = if c.reason.is_empty() {
                order_note
            } else {
                format!("{}; {order_note}", c.reason)
            };
            let tiers = c
                .tiers
                .iter()
                .map(|(max, name)| match max {
                    Some(m) => format!("<={m}:{name}"),
                    None => format!("open:{name}"),
                })
                .collect::<Vec<_>>()
                .join(",");
            tracing::info!(
                event = "kernel_selected",
                op = req.op.as_str(),
                config = %req.config,
                provider = provider_id.0,
                "impl" = %c.implementation,
                impl_provider = %c.impl_provider,
                reason_code = c.reason_code,
                tiers = %tiers,
                reason = %c.reason,
                "kernel provider selected"
            );
            metrics
                .provider_selected
                .get_or_create(&SelectedLabels {
                    op: req.op.as_str().to_string(),
                    provider: provider_id.0.to_string(),
                    r#impl: c.implementation.clone(),
                })
                .set(1);
            selections.push(Selection {
                op: req.op,
                config: req.config.clone(),
                provider: provider_id,
                implementation: c.implementation,
                reason: c.reason,
                impl_provider: c.impl_provider,
                reason_code: c.reason_code,
                tiers: c.tiers,
            });
            chosen.insert(req.spec, run);
        }
        Ok(KernelRegistry { chosen, selections })
    }

    /// True when `spec` was among the startup requirements and a provider was selected for it:
    /// how an executor learns whether an optional op (the v2.1 `add_rmsnorm`) it listed survived
    /// [`OpConfig::supported_by`] filtering, before calling its accessor.
    pub fn is_selected(&self, spec: &OpConfig) -> bool {
        self.chosen.contains_key(spec)
    }

    /// Every selection made at startup, in requirement order.
    pub fn selections(&self) -> &[Selection] {
        &self.selections
    }

    /// The provider selected for `spec`. Panics when `spec` was not among the startup
    /// requirements: the executor derives its requirement list from the same code path that
    /// calls the accessors, so a miss is a programming error.
    fn provider(&self, spec: OpConfig) -> &dyn KernelProvider {
        match self.chosen.get(&spec) {
            Some(provider) => provider.as_ref(),
            None => panic!(
                "kernel not selected at startup: {} {}",
                spec.op(),
                spec.render()
            ),
        }
    }

    pub fn gemm(&self, cfg: &GemmConfig) -> &dyn GemmKernel {
        self.provider(OpConfig::Gemm(*cfg))
            .gemm()
            .expect("the selected provider implements gemm")
    }

    pub fn attention(&self, cfg: &AttentionConfig) -> &dyn AttentionKernel {
        self.provider(OpConfig::Attention(*cfg))
            .attention()
            .expect("the selected provider implements attention")
    }

    pub fn norm(&self, cfg: &NormConfig) -> &dyn NormKernel {
        self.provider(OpConfig::Rmsnorm(*cfg))
            .norm()
            .expect("the selected provider implements rmsnorm")
    }

    pub fn rope(&self, cfg: &RopeConfig) -> &dyn RopeKernel {
        self.provider(OpConfig::Rope(*cfg))
            .rope()
            .expect("the selected provider implements rope")
    }

    pub fn activation(&self, cfg: &ActivationConfig) -> &dyn ActivationKernel {
        self.provider(OpConfig::SiluMul(*cfg))
            .activation()
            .expect("the selected provider implements silu_mul")
    }

    pub fn embedding(&self, cfg: &EmbeddingConfig) -> &dyn EmbeddingKernel {
        self.provider(OpConfig::Embedding(*cfg))
            .embedding()
            .expect("the selected provider implements embedding")
    }

    pub fn elementwise(&self, cfg: &ElementwiseConfig) -> &dyn ElementwiseKernel {
        self.provider(OpConfig::Add(*cfg))
            .elementwise()
            .expect("the selected provider implements add")
    }

    pub fn kv_copy(&self, cfg: &KvCopyConfig) -> &dyn KvCopyKernel {
        self.provider(OpConfig::CopyBlocks(*cfg))
            .kv_copy()
            .expect("the selected provider implements copy_blocks")
    }

    /// The MoE kernel selected for routing at `cfg`.
    pub fn moe_route(&self, cfg: &MoeRouteConfig) -> &dyn MoeKernel {
        self.provider(OpConfig::MoeRoute(*cfg))
            .moe()
            .expect("the selected provider implements moe_route")
    }

    /// The MoE kernel selected for expert compute at `cfg`.
    pub fn moe_experts(&self, cfg: &MoeExpertsConfig) -> &dyn MoeKernel {
        self.provider(OpConfig::MoeExperts(*cfg))
            .moe()
            .expect("the selected provider implements moe_experts")
    }

    /// The fused residual-add + RMSNorm kernel selected for `cfg` (ABI v2.1).
    pub fn add_rmsnorm(&self, cfg: &AddRmsnormConfig) -> &dyn AddRmsnormKernel {
        self.provider(OpConfig::AddRmsnorm(*cfg))
            .add_rmsnorm()
            .expect("the selected provider implements add_rmsnorm")
    }

    /// The logits reduction kernel selected for `cfg` (ABI v2.1).
    pub fn logits_reduce(&self, cfg: &LogitsReduceConfig) -> &dyn LogitsReduceKernel {
        self.provider(OpConfig::LogitsReduce(*cfg))
            .logits_reduce()
            .expect("the selected provider implements logits_reduce")
    }

    /// The sharded RMSNorm kernel selected for `row_sumsq` at `cfg` (ABI v2.6).
    pub fn row_sumsq(&self, cfg: &RowSumsqConfig) -> &dyn ShardedNormKernel {
        self.provider(OpConfig::RowSumsq(*cfg))
            .sharded_norm()
            .expect("the selected provider implements row_sumsq")
    }

    /// The sharded RMSNorm kernel selected for `rmsnorm_sharded` at `cfg` (ABI v2.6).
    pub fn rmsnorm_sharded(&self, cfg: &RmsnormShardedConfig) -> &dyn ShardedNormKernel {
        self.provider(OpConfig::RmsnormSharded(*cfg))
            .sharded_norm()
            .expect("the selected provider implements rmsnorm_sharded")
    }

    /// The quantized GEMM kernel selected for `cfg` (ABI v2.9).
    pub fn qgemm(&self, cfg: &QGemmConfig) -> &dyn QGemmKernel {
        self.provider(OpConfig::QGemm(*cfg))
            .qgemm()
            .expect("the selected provider implements qgemm")
    }

    /// The activation quantization kernel selected for `cfg` (ABI v2.9).
    pub fn quantize_act(&self, cfg: &QuantizeActConfig) -> &dyn QuantizeActKernel {
        self.provider(OpConfig::QuantizeAct(*cfg))
            .quantize_act()
            .expect("the selected provider implements quantize_act")
    }

    /// The KV transcode kernel selected for `cfg` (ABI v2.11).
    pub fn kv_transcode(&self, cfg: &KvTranscodeConfig) -> &dyn KvTranscodeKernel {
        self.provider(OpConfig::KvTranscode(*cfg))
            .kv_transcode()
            .expect("the selected provider implements kv_transcode")
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::sync::Mutex;

    use turbine_core::types::DType;

    use std::path::Path;

    use turbine_core::types::{DeviceId, MemoryKind, Vendor};
    use turbine_device::{DeviceInfo, DeviceMemoryInfo};
    use turbine_tensor::host::HostMemory;
    use turbine_tensor::{DeviceMemory, Tensor};

    use super::*;
    use crate::cards::{CardCapabilities, CardThresholds, OpPreference, RowTierSpec};
    use crate::ops::{
        AttentionContext, AttentionKind, MoeExpertsContext, MoeRouteContext, PagedAttentionContext,
    };
    use crate::{ShimLibrary, shim_provider};

    /// A provider implementing only attention, supporting the configs `accepts` admits.
    struct FakeProvider {
        id: &'static str,
        accepts: fn(&AttentionConfig) -> bool,
    }

    impl AttentionKernel for FakeProvider {
        fn supports(&self, cfg: &AttentionConfig) -> bool {
            (self.accepts)(cfg)
        }
        fn implementation(&self, _cfg: &AttentionConfig) -> String {
            format!("{}_fmha", self.id)
        }
        fn execute(&self, _ctx: &mut AttentionContext<'_>) -> Result<(), KernelError> {
            Ok(())
        }
        fn execute_paged(&self, _ctx: &mut PagedAttentionContext<'_>) -> Result<(), KernelError> {
            Ok(())
        }
    }

    impl KernelProvider for FakeProvider {
        fn id(&self) -> ProviderId {
            ProviderId(self.id)
        }
        fn gemm(&self) -> Option<&dyn GemmKernel> {
            None
        }
        fn attention(&self) -> Option<&dyn AttentionKernel> {
            Some(self)
        }
        fn norm(&self) -> Option<&dyn NormKernel> {
            None
        }
        fn rope(&self) -> Option<&dyn RopeKernel> {
            None
        }
        fn activation(&self) -> Option<&dyn ActivationKernel> {
            None
        }
        fn embedding(&self) -> Option<&dyn EmbeddingKernel> {
            None
        }
        fn elementwise(&self) -> Option<&dyn ElementwiseKernel> {
            None
        }
        fn kv_copy(&self) -> Option<&dyn KvCopyKernel> {
            None
        }
        fn moe(&self) -> Option<&dyn MoeKernel> {
            None
        }
    }

    /// `first`: head_dim 64 only.
    fn first() -> Arc<dyn KernelProvider> {
        Arc::new(FakeProvider {
            id: "first",
            accepts: |cfg| cfg.head_dim == 64,
        })
    }

    /// `second`: head_dim 128 with 8 KV heads.
    fn second() -> Arc<dyn KernelProvider> {
        Arc::new(FakeProvider {
            id: "second",
            accepts: |cfg| cfg.head_dim == 128 && cfg.num_kv_heads == 8,
        })
    }

    fn prefill_128_8() -> AttentionConfig {
        AttentionConfig {
            kind: AttentionKind::Prefill,
            num_q_heads: 24,
            num_kv_heads: 8,
            head_dim: 128,
            dtype: DType::BF16,
            block_tokens: None,
            causal: true,
        }
    }

    /// A `MakeWriter` target collecting every formatted log line.
    #[derive(Clone, Default)]
    struct Captured(Arc<Mutex<Vec<u8>>>);

    impl Write for Captured {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("capture lock").extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn selection_order_and_reason() {
        let reg = MetricsRegistry::new();
        let metrics = KernelMetrics::register(&reg);
        let captured = Captured::default();
        let writer = captured.clone();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(move || writer.clone())
            .with_max_level(tracing::Level::INFO)
            .finish();
        let cfg = prefill_128_8();
        let registry = tracing::subscriber::with_default(subscriber, || {
            // Another test thread may have cached this callsite as disabled (no subscriber
            // there); recompute interest now that the capture subscriber is registered.
            tracing::callsite::rebuild_interest_cache();
            KernelRegistry::build(
                vec![first(), second()],
                &[ProviderId("first"), ProviderId("second")],
                &[OpConfig::Attention(cfg).into()],
                &metrics,
                None,
            )
        })
        .expect("second provider supports the config");

        let reason = "first provider in order supporting config; unsupported by: first";
        assert_eq!(
            registry.selections(),
            [Selection {
                op: OpKind::AttentionPrefill,
                config: "head_dim=128 kv_heads=8 dtype=bf16 q_heads=24 causal=1".into(),
                provider: ProviderId("second"),
                implementation: "second_fmha".into(),
                reason: reason.into(),
                impl_provider: "second".into(),
                reason_code: PROVIDER_INTERNAL,
                tiers: Vec::new(),
            }]
        );
        // The accessor hands out the selected provider's kernel.
        assert_eq!(registry.attention(&cfg).implementation(&cfg), "second_fmha");

        let log =
            String::from_utf8(captured.0.lock().expect("capture lock").clone()).expect("utf8 log");
        for needle in [
            "\"event\":\"kernel_selected\"",
            "\"op\":\"attention_prefill\"",
            "\"config\":\"head_dim=128 kv_heads=8 dtype=bf16 q_heads=24 causal=1\"",
            "\"provider\":\"second\"",
            "\"impl\":\"second_fmha\"",
            &format!("\"reason\":\"{reason}\""),
        ] {
            assert!(log.contains(needle), "log lacks {needle}: {log}");
        }

        let text = reg.render().expect("render");
        assert!(
            text.contains(
                "turbine_kernel_provider_selected{op=\"attention_prefill\",provider=\"second\",impl=\"second_fmha\"} 1"
            ),
            "{text}"
        );
    }

    #[test]
    fn no_provider_is_startup_error() {
        let metrics = KernelMetrics::register(&MetricsRegistry::new());
        let result = KernelRegistry::build(
            vec![first()],
            &[ProviderId("first")],
            &[OpConfig::Attention(prefill_128_8()).into()],
            &metrics,
            None,
        );
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("an unsupported op must fail registry construction"),
        };
        assert!(matches!(
            err,
            KernelError::NoProvider {
                op: OpKind::AttentionPrefill,
                ..
            }
        ));
        assert_eq!(
            err.to_string(),
            "no kernel provider supports attention_prefill head_dim=128 kv_heads=8 dtype=bf16 q_heads=24 causal=1"
        );
    }

    #[test]
    fn phase2_ops_select_and_skip_providers_without_the_family() {
        let metrics = KernelMetrics::register(&MetricsRegistry::new());
        let paged = AttentionConfig {
            kind: AttentionKind::DecodePaged,
            block_tokens: Some(16),
            ..prefill_128_8()
        };
        let copy = KvCopyConfig {
            num_layers: 28,
            block_bytes: 65_536,
        };
        let route = MoeRouteConfig {
            num_experts: 64,
            top_k: 8,
            renormalize: false,
            bf16_logits: true,
        };
        let experts = MoeExpertsConfig {
            hidden: 2048,
            inter: 1024,
            num_experts: 64,
            top_k: 8,
            expert_begin: 0,
            expert_end: 64,
            dtype: DType::BF16,
        };
        let reqs: Vec<OpRequirement> = [
            OpConfig::Attention(paged),
            OpConfig::CopyBlocks(copy),
            OpConfig::MoeRoute(route),
            OpConfig::MoeExperts(experts),
        ]
        .into_iter()
        .map(OpRequirement::from)
        .collect();
        // `first` has neither kv_copy nor moe and rejects head_dim 128.
        let registry = KernelRegistry::build(
            vec![first(), crate::cpu_reference_provider()],
            &[ProviderId("first"), ProviderId("cpu-reference")],
            &reqs,
            &metrics,
            None,
        )
        .expect("cpu-reference supports every phase 2 op");
        let picked: Vec<(OpKind, &str, &str)> = registry
            .selections()
            .iter()
            .map(|s| (s.op, s.provider.0, s.implementation.as_str()))
            .collect();
        assert_eq!(
            picked,
            [
                (
                    OpKind::AttentionDecodePaged,
                    "cpu-reference",
                    "cpu_attention_paged_f32acc"
                ),
                (OpKind::CopyBlocks, "cpu-reference", "cpu_copy_blocks"),
                (OpKind::MoeRoute, "cpu-reference", "cpu_moe_route"),
                (
                    OpKind::MoeExperts,
                    "cpu-reference",
                    "cpu_moe_experts_f32acc"
                ),
            ]
        );
        assert!(registry.selections().iter().all(
            |s| s.reason == "first provider in order supporting config; unsupported by: first"
        ));
        assert_eq!(
            registry.selections()[1].config,
            "num_layers=28 block_bytes=65536"
        );
        assert_eq!(
            registry.kv_copy(&copy).implementation(&copy),
            "cpu_copy_blocks"
        );
        assert_eq!(
            registry.moe_route(&route).implementation_route(&route),
            "cpu_moe_route"
        );
        assert_eq!(
            registry
                .moe_experts(&experts)
                .implementation_experts(&experts),
            "cpu_moe_experts_f32acc"
        );
        assert_eq!(
            registry.attention(&paged).implementation(&paged),
            "cpu_attention_paged_f32acc"
        );
    }

    /// A provider without the v2.1 families (the trait defaults) is skipped for them and the
    /// next provider in order serves them; with no other provider they are a startup error that
    /// callers avoid by checking `OpConfig::supported_by` first.
    #[test]
    fn v21_ops_skip_providers_without_them() {
        let metrics = KernelMetrics::register(&MetricsRegistry::new());
        let fused = AddRmsnormConfig {
            dtype: DType::BF16,
            dim: 3072,
        };
        let reduce = LogitsReduceConfig {
            vocab: 128_256,
            top_n: 20,
        };
        let reqs: Vec<OpRequirement> =
            [OpConfig::AddRmsnorm(fused), OpConfig::LogitsReduce(reduce)]
                .into_iter()
                .map(OpRequirement::from)
                .collect();
        let only_first = first();
        for spec in [reqs[0].spec, reqs[1].spec] {
            assert!(!spec.supported_by(only_first.as_ref()));
            assert!(spec.supported_by(crate::cpu_reference_provider().as_ref()));
        }
        let registry = KernelRegistry::build(
            vec![first(), crate::cpu_reference_provider()],
            &[ProviderId("first"), ProviderId("cpu-reference")],
            &reqs,
            &metrics,
            None,
        )
        .expect("cpu-reference supports both v2.1 ops");
        let picked: Vec<(OpKind, &str, &str, &str)> = registry
            .selections()
            .iter()
            .map(|s| {
                (
                    s.op,
                    s.provider.0,
                    s.implementation.as_str(),
                    s.config.as_str(),
                )
            })
            .collect();
        assert_eq!(
            picked,
            [
                (
                    OpKind::AddRmsnorm,
                    "cpu-reference",
                    "cpu_add_rmsnorm",
                    "dim=3072 dtype=bf16"
                ),
                (
                    OpKind::LogitsReduce,
                    "cpu-reference",
                    "cpu_logits_reduce",
                    "vocab=128256 top_n=20"
                ),
            ]
        );
        assert_eq!(
            registry.add_rmsnorm(&fused).implementation(&fused),
            "cpu_add_rmsnorm"
        );
        assert!(registry.is_selected(&OpConfig::AddRmsnorm(fused)));
        let other = AddRmsnormConfig {
            dtype: DType::BF16,
            dim: 2048,
        };
        assert!(!registry.is_selected(&OpConfig::AddRmsnorm(other)));
        assert_eq!(
            registry.logits_reduce(&reduce).implementation(&reduce),
            "cpu_logits_reduce"
        );

        let err = match KernelRegistry::build(
            vec![first()],
            &[ProviderId("first")],
            &reqs[..1],
            &metrics,
            None,
        ) {
            Err(e) => e,
            Ok(_) => panic!("a provider without add_rmsnorm cannot serve it"),
        };
        assert_eq!(
            err.to_string(),
            "no kernel provider supports add_rmsnorm dim=3072 dtype=bf16"
        );
    }

    /// ABI v2.6: the sharded RMSNorm ops are an optional family like the v2.1 ops: a provider
    /// without it is skipped, the cpu-reference serves both, and each op has its own accessor
    /// (`row_sumsq`, `rmsnorm_sharded`) returning the one `ShardedNormKernel`. Breaks if the ops
    /// are not selectable through the registry or render another config form.
    #[test]
    fn v26_sharded_norm_ops_select() {
        let metrics = KernelMetrics::register(&MetricsRegistry::new());
        let sumsq = RowSumsqConfig {
            dim: 1024,
            dtype: DType::BF16,
        };
        let sharded = RmsnormShardedConfig {
            dim: 1024,
            full_dim: 2048,
            dtype: DType::BF16,
        };
        let reqs: Vec<OpRequirement> =
            [OpConfig::RowSumsq(sumsq), OpConfig::RmsnormSharded(sharded)]
                .into_iter()
                .map(OpRequirement::from)
                .collect();
        for req in &reqs {
            assert!(!req.spec.supported_by(first().as_ref()));
        }
        let registry = KernelRegistry::build(
            vec![first(), crate::cpu_reference_provider()],
            &[ProviderId("first"), ProviderId("cpu-reference")],
            &reqs,
            &metrics,
            None,
        )
        .expect("cpu-reference supports both v2.6 ops");
        let picked: Vec<(OpKind, &str, &str, &str)> = registry
            .selections()
            .iter()
            .map(|s| {
                (
                    s.op,
                    s.provider.0,
                    s.implementation.as_str(),
                    s.config.as_str(),
                )
            })
            .collect();
        assert_eq!(
            picked,
            [
                (
                    OpKind::RowSumsq,
                    "cpu-reference",
                    "cpu_row_sumsq",
                    "dim=1024 dtype=bf16"
                ),
                (
                    OpKind::RmsnormSharded,
                    "cpu-reference",
                    "cpu_rmsnorm_sharded",
                    "dim=1024 full_dim=2048 dtype=bf16"
                ),
            ]
        );
        assert_eq!(
            registry.row_sumsq(&sumsq).implementation_row_sumsq(&sumsq),
            "cpu_row_sumsq"
        );
        assert_eq!(
            registry
                .rmsnorm_sharded(&sharded)
                .implementation_rmsnorm_sharded(&sharded),
            "cpu_rmsnorm_sharded"
        );
    }

    #[test]
    fn duplicate_requirements_select_once() {
        let metrics = KernelMetrics::register(&MetricsRegistry::new());
        let req: OpRequirement = OpConfig::Attention(prefill_128_8()).into();
        let registry = KernelRegistry::build(
            vec![second()],
            &[ProviderId("second")],
            &[req.clone(), req],
            &metrics,
            None,
        )
        .expect("second supports the config");
        assert_eq!(registry.selections().len(), 1);
        assert_eq!(
            registry.selections()[0].reason,
            "first provider in order supports config"
        );
    }

    #[test]
    #[should_panic(
        expected = "kernel not selected at startup: attention_prefill head_dim=64 kv_heads=8"
    )]
    fn accessor_panics_on_unselected_config() {
        let metrics = KernelMetrics::register(&MetricsRegistry::new());
        let registry = KernelRegistry::build(
            vec![second()],
            &[ProviderId("second")],
            &[OpConfig::Attention(prefill_128_8()).into()],
            &metrics,
            None,
        )
        .expect("second supports the config");
        let other = AttentionConfig {
            head_dim: 64,
            ..prefill_128_8()
        };
        registry.attention(&other);
    }

    const TOY_CAPABILITIES: CardCapabilities = CardCapabilities {
        matrix_instructions: &[],
        bf16: true,
        wave_size: 32,
        lds_bytes: 65536,
    };
    const TOY_THRESHOLDS: CardThresholds = CardThresholds {
        moe_small_max_rows: 8,
        paged_page_multiple: 16,
    };

    /// A toy profile preferring `stub_b`, then `stub_a` for `rmsnorm`.
    static PREFER_B: CardProfile = CardProfile {
        name: "toy_b",
        vendor: "amd",
        archs: &["gfx942"],
        capabilities: TOY_CAPABILITIES,
        thresholds: TOY_THRESHOLDS,
        preferences: &[OpPreference {
            op: OpKind::Rmsnorm,
            order: &["stub_b", "stub_a"],
            row_tiers: &[],
        }],
    };

    /// A toy profile whose preferred `rmsnorm` implementation the library does not have.
    static PREFER_MISSING: CardProfile = CardProfile {
        name: "toy_missing",
        vendor: "amd",
        archs: &["gfx942"],
        capabilities: TOY_CAPABILITIES,
        thresholds: TOY_THRESHOLDS,
        preferences: &[OpPreference {
            op: OpKind::Rmsnorm,
            order: &["stub_x", "stub_b"],
            row_tiers: &[],
        }],
    };

    /// A toy profile listing no order for any op.
    static LISTS_NOTHING: CardProfile = CardProfile {
        name: "toy_none",
        vendor: "amd",
        archs: &["gfx942"],
        capabilities: TOY_CAPABILITIES,
        thresholds: TOY_THRESHOLDS,
        preferences: &[],
    };

    fn gfx942_device() -> DeviceInfo {
        DeviceInfo {
            index: DeviceId(0),
            vendor: Vendor::Amd,
            vendor_index: 0,
            name: "stub device".into(),
            uuid: None,
            pci_bus_id: None,
            arch: Some("gfx942".into()),
            driver_version: None,
            memory: DeviceMemoryInfo {
                kind: MemoryKind::Dedicated,
                total_bytes: 1 << 30,
                shared_with_host: false,
            },
        }
    }

    /// The shim provider of a stub library (`hip`, built for gfx942).
    fn stub_provider(path: &str) -> Arc<dyn KernelProvider> {
        let lib = ShimLibrary::load(Path::new(path), "hip").expect("stub library");
        shim_provider(lib.create_context(&gfx942_device()).expect("context"))
    }

    fn norm(dim: u64) -> OpRequirement {
        OpConfig::Rmsnorm(NormConfig {
            dim,
            dtype: DType::BF16,
        })
        .into()
    }

    /// The (implementation, impl_provider, reason_code) of each selection.
    fn picks(registry: &KernelRegistry) -> Vec<(String, String, &'static str)> {
        registry
            .selections()
            .iter()
            .map(|s| {
                (
                    s.implementation.clone(),
                    s.impl_provider.clone(),
                    s.reason_code,
                )
            })
            .collect()
    }

    /// Phase 2m S-5: with an enumerating library (the V24 stub: `stub_a` supports every
    /// `rmsnorm`, `stub_b` refuses dim 4096) the registry follows the card profile's order:
    /// `stub_b` where it supports the config (`profile_preferred`), else `stub_a`
    /// (`profile_fallback`, naming the refusal), a listed name the library lacks is skipped
    /// (`profile_fallback`), and a profile listing nothing takes library order
    /// (`library_order`). A config no implementation supports fails naming each refusal.
    #[test]
    fn profile_order_picks_and_falls_back() {
        let metrics = KernelMetrics::register(&MetricsRegistry::new());
        let provider = stub_provider(env!("TURBINE_STUB_GFX942_V24"));
        let order = [provider.id()];
        let build = |card: &CardProfile, reqs: &[OpRequirement]| {
            KernelRegistry::build(
                vec![Arc::clone(&provider)],
                &order,
                reqs,
                &metrics,
                Some(card),
            )
        };
        let registry = build(&PREFER_B, &[norm(2048), norm(4096)]).expect("stub rmsnorm");
        assert_eq!(
            picks(&registry),
            [
                ("stub_b".into(), "stub_alt".into(), PROFILE_PREFERRED),
                ("stub_a".into(), "stub".into(), PROFILE_FALLBACK),
            ]
        );
        assert_eq!(
            registry.selections()[1].reason,
            "card profile toy_b order; refused: stub_b (unsupported); first provider in order \
             supports config"
        );
        // The accessor hands out the provider bound to the choice.
        let cfg = NormConfig {
            dim: 2048,
            dtype: DType::BF16,
        };
        assert_eq!(registry.norm(&cfg).implementation(&cfg), "stub_b");

        let registry = build(&PREFER_MISSING, &[norm(2048)]).expect("stub rmsnorm");
        assert_eq!(
            picks(&registry),
            [("stub_b".into(), "stub_alt".into(), PROFILE_FALLBACK)]
        );
        assert!(
            registry.selections()[0]
                .reason
                .contains("stub_x (not in library)"),
            "{}",
            registry.selections()[0].reason
        );

        let registry = build(&LISTS_NOTHING, &[norm(4096)]).expect("stub rmsnorm");
        assert_eq!(
            picks(&registry),
            [("stub_a".into(), "stub".into(), LIBRARY_ORDER)]
        );

        let gemm: OpRequirement = OpConfig::Gemm(GemmConfig {
            n: 64,
            k: 64,
            trans_b: true,
            a_dtype: DType::BF16,
            b_dtype: DType::BF16,
            c_dtype: DType::BF16,
        })
        .into();
        let err = match build(&PREFER_B, &[gemm]) {
            Err(e) => e,
            Ok(_) => panic!("the stub supports no gemm"),
        };
        assert_eq!(
            err.to_string(),
            "no kernel provider supports gemm n=64 k=64 trans_b=1 a_dtype=bf16 b_dtype=bf16 \
             c_dtype=bf16 (hip: stub_gemm (unsupported))"
        );
    }

    /// A kernel library without the v2.4 group (the V21 stub, minor 3) keeps main's selection
    /// with or without a card profile (`provider_internal`, the next provider in order when it
    /// refuses), and so does an enumerating library without a card profile.
    #[test]
    fn v23_library_uses_library_internal_choice() {
        let metrics = KernelMetrics::register(&MetricsRegistry::new());
        for (path, card) in [
            (
                env!("TURBINE_STUB_GFX942_V21"),
                Some(&crate::cards::GFX1201),
            ),
            (env!("TURBINE_STUB_GFX942_V21"), None),
            (env!("TURBINE_STUB_GFX942_V24"), None),
        ] {
            let stub = stub_provider(path);
            let order = [stub.id(), ProviderId("cpu-reference")];
            let registry = KernelRegistry::build(
                vec![stub, crate::cpu_reference_provider()],
                &order,
                &[norm(2048)],
                &metrics,
                card,
            )
            .expect("cpu-reference serves rmsnorm");
            assert_eq!(
                registry.selections(),
                [Selection {
                    op: OpKind::Rmsnorm,
                    config: "dim=2048 dtype=bf16".into(),
                    provider: ProviderId("cpu-reference"),
                    implementation: "cpu_rmsnorm".into(),
                    reason: "first provider in order supporting config; unsupported by: hip".into(),
                    impl_provider: "cpu-reference".into(),
                    reason_code: PROVIDER_INTERNAL,
                    tiers: Vec::new(),
                }],
                "{path} card {:?}",
                card.map(|c| c.name)
            );
        }
    }

    /// A fake enumerating MoE provider: two `moe_experts` implementations, `fast` (device
    /// offsets) and `host` (reads host offsets); a bound copy records the index each
    /// `experts` call runs.
    struct FakeMoe {
        choice: Option<ImplChoice>,
        runs: Arc<Mutex<Vec<u32>>>,
    }

    impl FakeMoe {
        fn impls() -> Vec<ImplInfo> {
            [("fast", false), ("host", true)]
                .into_iter()
                .enumerate()
                .map(|(i, (name, host))| ImplInfo {
                    index: i as u32,
                    name: name.into(),
                    provider: "fake".into(),
                    needs_host_offsets: host,
                })
                .collect()
        }

        fn index(&self, rows: usize) -> usize {
            self.choice.as_ref().map_or(0, |c| c.index_for(rows)) as usize
        }
    }

    impl MoeKernel for FakeMoe {
        fn supports_route(&self, _cfg: &MoeRouteConfig) -> bool {
            false
        }
        fn supports_experts(&self, _cfg: &MoeExpertsConfig) -> bool {
            true
        }
        fn implementation_route(&self, _cfg: &MoeRouteConfig) -> String {
            String::new()
        }
        fn implementation_experts(&self, _cfg: &MoeExpertsConfig) -> String {
            Self::impls()[self.index(1)].name.clone()
        }
        fn route(&self, _ctx: &mut MoeRouteContext<'_>) -> Result<(), KernelError> {
            Ok(())
        }
        fn experts(&self, ctx: &mut MoeExpertsContext<'_>) -> Result<(), KernelError> {
            let rows = ctx.cfg.routed_rows(ctx.x.shape[0]);
            self.runs
                .lock()
                .expect("runs")
                .push(self.index(rows) as u32);
            Ok(())
        }
        fn needs_host_offsets(&self, _cfg: &MoeExpertsConfig, routed_rows: usize) -> bool {
            Self::impls()[self.index(routed_rows)].needs_host_offsets
        }
    }

    impl KernelProvider for FakeMoe {
        fn id(&self) -> ProviderId {
            ProviderId("fake")
        }
        fn gemm(&self) -> Option<&dyn GemmKernel> {
            None
        }
        fn attention(&self) -> Option<&dyn AttentionKernel> {
            None
        }
        fn norm(&self) -> Option<&dyn NormKernel> {
            None
        }
        fn rope(&self) -> Option<&dyn RopeKernel> {
            None
        }
        fn activation(&self) -> Option<&dyn ActivationKernel> {
            None
        }
        fn embedding(&self) -> Option<&dyn EmbeddingKernel> {
            None
        }
        fn elementwise(&self) -> Option<&dyn ElementwiseKernel> {
            None
        }
        fn kv_copy(&self) -> Option<&dyn KvCopyKernel> {
            None
        }
        fn moe(&self) -> Option<&dyn MoeKernel> {
            Some(self)
        }
        fn implementations(&self, op: OpKind) -> Vec<ImplInfo> {
            if op == OpKind::MoeExperts {
                Self::impls()
            } else {
                Vec::new()
            }
        }
        fn implementation_supports(
            &self,
            _spec: &OpConfig,
            _index: u32,
            _rows: Option<u32>,
        ) -> bool {
            true
        }
        fn bind(&self, _spec: &OpConfig, choice: &ImplChoice) -> Option<Arc<dyn KernelProvider>> {
            Some(Arc::new(FakeMoe {
                choice: Some(choice.clone()),
                runs: Arc::clone(&self.runs),
            }))
        }
    }

    /// A toy profile with `moe_experts` row tiers: up to 8 routed rows `fast` (then `host`),
    /// above only `host`.
    static MOE_TIERS: CardProfile = CardProfile {
        name: "toy_moe",
        vendor: "amd",
        archs: &["gfx942"],
        capabilities: TOY_CAPABILITIES,
        thresholds: TOY_THRESHOLDS,
        preferences: &[OpPreference {
            op: OpKind::MoeExperts,
            order: &["host"],
            row_tiers: &[
                RowTierSpec {
                    max_rows: Some(8),
                    order: &["fast", "host"],
                },
                RowTierSpec {
                    max_rows: None,
                    order: &["host"],
                },
            ],
        }],
    };

    /// Phase 2m S-5: a tiered op resolves one implementation per routed-row tier into
    /// `ImplChoice::ByRows`; the bound provider runs index 0 for 8 routed rows and 1 for 9, and
    /// `needs_host_offsets` follows the flag of the index a call runs. Breaks if the tier bound
    /// is off by one or the per-call choice ignores the rows.
    #[test]
    fn row_tiers_bind_per_call() {
        let metrics = KernelMetrics::register(&MetricsRegistry::new());
        let runs = Arc::new(Mutex::new(Vec::new()));
        let fake: Arc<dyn KernelProvider> = Arc::new(FakeMoe {
            choice: None,
            runs: Arc::clone(&runs),
        });
        let cfg = MoeExpertsConfig {
            hidden: 8,
            inter: 8,
            num_experts: 2,
            top_k: 1,
            expert_begin: 0,
            expert_end: 2,
            dtype: DType::BF16,
        };
        let registry = KernelRegistry::build(
            vec![fake],
            &[ProviderId("fake")],
            &[OpConfig::MoeExperts(cfg).into()],
            &metrics,
            Some(&MOE_TIERS),
        )
        .expect("fake moe_experts");
        let s = &registry.selections()[0];
        assert_eq!(
            (s.implementation.as_str(), s.reason_code),
            ("fast", PROFILE_PREFERRED)
        );
        assert_eq!(
            s.tiers,
            [(Some(8), "fast".to_string()), (None, "host".to_string())]
        );
        let moe = registry.moe_experts(&cfg);
        assert!(!moe.needs_host_offsets(&cfg, 8));
        assert!(moe.needs_host_offsets(&cfg, 9));

        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 20);
        let t = |shape: &[usize], dtype| Tensor::empty(&mem, shape, dtype).expect("tensor");
        let (w_gate, w_up, w_down) = (
            t(&[2, 8, 8], DType::BF16),
            t(&[2, 8, 8], DType::BF16),
            t(&[2, 8, 8], DType::BF16),
        );
        let offsets = t(&[3], DType::I32);
        for tokens in [8usize, 9] {
            let (x, out) = (t(&[tokens, 8], DType::BF16), t(&[tokens, 8], DType::BF16));
            let (rows, weights) = (t(&[tokens], DType::I32), t(&[tokens, 1], DType::F32));
            moe.experts(&mut MoeExpertsContext {
                cfg,
                x: x.view(),
                w_gate: w_gate.view(),
                w_up: w_up.view(),
                w_down: w_down.view(),
                sorted_rows: rows.view(),
                expert_offsets: offsets.view(),
                topk_weights: weights.view(),
                host_expert_offsets: &[],
                out: out.view(),
                workspace: None,
            })
            .expect("fake experts");
        }
        assert_eq!(*runs.lock().expect("runs"), [0, 1]);
        assert_eq!(
            ImplChoice::ByRows(vec![
                RowTier {
                    max_rows: Some(8),
                    index: 0
                },
                RowTier {
                    max_rows: None,
                    index: 1
                },
            ])
            .index_for(9),
            1
        );
    }
}
