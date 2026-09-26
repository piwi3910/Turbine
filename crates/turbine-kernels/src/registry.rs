//! Per-op provider selection at startup (P1 S-6): for every distinct op config the model needs,
//! the first provider in the configured order whose family trait `supports()` it wins. Every
//! choice is logged (`event="kernel_selected"`) with its reason and exported as
//! `turbine_kernel_provider_selected{op,provider,impl} 1`; a config no provider supports fails
//! startup, never first use.
use std::collections::HashMap;
use std::sync::Arc;

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use turbine_observability::MetricsRegistry;

use crate::KernelError;
use crate::ops::{
    ActivationConfig, ActivationKernel, AddRmsnormConfig, AddRmsnormKernel, AttentionConfig,
    AttentionKernel, ElementwiseConfig, ElementwiseKernel, EmbeddingConfig, EmbeddingKernel,
    GemmConfig, GemmKernel, KernelProvider, KvCopyConfig, KvCopyKernel, LogitsReduceConfig,
    LogitsReduceKernel, MoeExpertsConfig, MoeKernel, MoeRouteConfig, NormConfig, NormKernel,
    OpKind, ProviderId, RopeConfig, RopeKernel,
};

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
        }
    }

    /// The implementation name `provider` would run for this config, or `None` when it lacks
    /// the op family or its `supports()` is false.
    fn probe(&self, provider: &dyn KernelProvider) -> Option<String> {
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
    pub implementation: String,
    pub reason: String,
}

/// The startup-resolved map from op config to the provider that runs it.
pub struct KernelRegistry {
    chosen: HashMap<OpConfig, Arc<dyn KernelProvider>>,
    selections: Vec<Selection>,
}

impl KernelRegistry {
    /// Picks, per distinct requirement, the first provider in `order` whose `supports()` is true;
    /// logs op, config, provider, implementation and reason and sets the selection gauge.
    /// Returns [`KernelError::NoProvider`] for the first requirement nothing supports. An id in
    /// `order` with no registered provider is skipped.
    pub fn build(
        providers: Vec<Arc<dyn KernelProvider>>,
        order: &[ProviderId],
        reqs: &[OpRequirement],
        metrics: &KernelMetrics,
    ) -> Result<KernelRegistry, KernelError> {
        let mut chosen: HashMap<OpConfig, Arc<dyn KernelProvider>> = HashMap::new();
        let mut selections = Vec::new();
        for req in reqs {
            if chosen.contains_key(&req.spec) {
                continue;
            }
            let mut unsupported: Vec<&'static str> = Vec::new();
            let mut picked = None;
            for id in order {
                let Some(provider) = providers.iter().find(|p| p.id() == *id) else {
                    continue;
                };
                match req.spec.probe(provider.as_ref()) {
                    Some(implementation) => {
                        picked = Some((Arc::clone(provider), implementation));
                        break;
                    }
                    None => unsupported.push(id.0),
                }
            }
            let Some((provider, implementation)) = picked else {
                return Err(KernelError::NoProvider {
                    op: req.op,
                    config: req.config.clone(),
                });
            };
            let reason = if unsupported.is_empty() {
                "first provider in order supports config".to_string()
            } else {
                format!(
                    "first provider in order supporting config; unsupported by: {}",
                    unsupported.join(", ")
                )
            };
            let provider_id = provider.id();
            tracing::info!(
                event = "kernel_selected",
                op = req.op.as_str(),
                config = %req.config,
                provider = provider_id.0,
                "impl" = %implementation,
                reason = %reason,
                "kernel provider selected"
            );
            metrics
                .provider_selected
                .get_or_create(&SelectedLabels {
                    op: req.op.as_str().to_string(),
                    provider: provider_id.0.to_string(),
                    r#impl: implementation.clone(),
                })
                .set(1);
            selections.push(Selection {
                op: req.op,
                config: req.config.clone(),
                provider: provider_id,
                implementation,
                reason,
            });
            chosen.insert(req.spec, provider);
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
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::sync::Mutex;

    use turbine_core::types::DType;

    use super::*;
    use crate::ops::{AttentionContext, AttentionKind, PagedAttentionContext};

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
        ) {
            Err(e) => e,
            Ok(_) => panic!("a provider without add_rmsnorm cannot serve it"),
        };
        assert_eq!(
            err.to_string(),
            "no kernel provider supports add_rmsnorm dim=3072 dtype=bf16"
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
        )
        .expect("second supports the config");
        let other = AttentionConfig {
            head_dim: 64,
            ..prefill_128_8()
        };
        registry.attention(&other);
    }
}
