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
    ActivationConfig, ActivationKernel, AttentionConfig, AttentionKernel, ElementwiseConfig,
    ElementwiseKernel, EmbeddingConfig, EmbeddingKernel, GemmConfig, GemmKernel, KernelProvider,
    NormConfig, NormKernel, OpKind, ProviderId, RopeConfig, RopeKernel,
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
/// prefill or decode).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OpConfig {
    Gemm(GemmConfig),
    Attention(AttentionConfig),
    Rmsnorm(NormConfig),
    Rope(RopeConfig),
    SiluMul(ActivationConfig),
    Embedding(EmbeddingConfig),
    Add(ElementwiseConfig),
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
        }
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
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::sync::Mutex;

    use turbine_core::types::DType;

    use super::*;
    use crate::ops::{AttentionContext, AttentionKind};

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
