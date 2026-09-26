//! Phase 2m S-13: the conformance suites run over a registry, not a hand-written list, so a
//! module registered without passing its suite fails. Each suite is run here over a test-only
//! registry holding one good module and one deliberately broken toy module: the suite must
//! name exactly the broken module and the check it breaks, and pass over the real registries.
use std::path::Path;
use std::sync::Arc;

use serde_json::Value;
use turbine_core::registry::{Module, Registry};
use turbine_kernels::{KernelRegistry, OpRequirement};
use turbine_model::config::ModelArchConfig;
use turbine_model::conformance::{
    ConformanceFailure, families_suite, formats_suite, processors_suite, weights_suite,
};
use turbine_model::executor::{ExecutorLimits, ExecutorOptions, ModelExecutor};
use turbine_model::families::{self, FamilyConfig, Llama, Mistral, ModelFamily};
use turbine_model::formats::{
    self, BoundTokens, ConstraintSpec, Llama3Json, Opening, SpecialToken, ToolFormat,
};
use turbine_model::sampling::processors::LogitBias;
use turbine_model::sampling::{self, LogitsProcessor, ProcessorParams, ProcessorState, Touched};
use turbine_model::testing::tiny::TinySpec;
use turbine_model::{LoadedWeights, ModelError, ToolCallParser, ToolChoice, ToolParse, WeightSlot};
use turbine_tensor::DeviceMemory;

/// The failures' `(module, check)` pairs.
fn pairs(failures: &[ConformanceFailure]) -> Vec<(&'static str, &'static str)> {
    failures.iter().map(|f| (f.module, f.check)).collect()
}

// ---- tool formats ------------------------------------------------------------------------

/// A parser that never finds a call.
struct ContentOnly;

impl ToolCallParser for ContentOnly {
    fn parse(&self, text: &str) -> ToolParse {
        ToolParse::Content(text.to_string())
    }
}

/// `llama3_json` whose parser returns content for its own sample call.
struct BrokenFormat;

impl Module for BrokenFormat {
    fn name(&self) -> &'static str {
        "broken"
    }
}

impl ToolFormat for BrokenFormat {
    fn special_tokens(&self) -> &'static [SpecialToken] {
        Llama3Json.special_tokens()
    }
    fn grammar(
        &self,
        tools: &[Value],
        choice: &ToolChoice,
        parallel: bool,
    ) -> Result<ConstraintSpec, ModelError> {
        Llama3Json.grammar(tools, choice, parallel)
    }
    fn parser(&self) -> Box<dyn ToolCallParser> {
        Box::new(ContentOnly)
    }
    fn opens_like_call(&self, text: &str, first: Option<u32>, tokens: &BoundTokens) -> Opening {
        Llama3Json.opens_like_call(text, first, tokens)
    }
    fn sample_call(&self) -> &'static str {
        Llama3Json.sample_call()
    }
}

static BROKEN_FORMATS: Registry<dyn ToolFormat> =
    Registry::new("tool_format", &[&Llama3Json, &BrokenFormat]);

// ---- model families ----------------------------------------------------------------------

/// Mistral (an untied checkpoint) whose weight slots drop `lm_head.weight`.
struct DropsLmHead;

impl Module for DropsLmHead {
    fn name(&self) -> &'static str {
        "broken"
    }
}

impl ModelFamily for DropsLmHead {
    fn hf_architectures(&self) -> &'static [&'static str] {
        Mistral.hf_architectures()
    }
    fn parse_config(&self, text: &Value) -> Result<FamilyConfig, ModelError> {
        Mistral.parse_config(text)
    }
    fn weight_slots(&self, cfg: &ModelArchConfig) -> Vec<WeightSlot> {
        let mut slots = Mistral.weight_slots(cfg);
        slots.retain(|s| s.name != "lm_head.weight");
        slots
    }
    fn requirements(
        &self,
        cfg: &ModelArchConfig,
        block_tokens: u32,
        opts: ExecutorOptions,
    ) -> Vec<OpRequirement> {
        Mistral.requirements(cfg, block_tokens, opts)
    }
    fn workspace_bytes(&self, cfg: &ModelArchConfig, limits: ExecutorLimits) -> u64 {
        Mistral.workspace_bytes(cfg, limits)
    }
    fn default_tool_format(&self) -> Option<&'static str> {
        Mistral.default_tool_format()
    }
    fn build_executor(
        &self,
        cfg: &ModelArchConfig,
        weights: LoadedWeights,
        registry: Arc<KernelRegistry>,
        mem: Arc<dyn DeviceMemory>,
        limits: ExecutorLimits,
        opts: ExecutorOptions,
    ) -> Result<Box<dyn ModelExecutor>, ModelError> {
        Mistral.build_executor(cfg, weights, registry, mem, limits, opts)
    }
    fn write_tiny(&self, dir: &Path, seed: u64) -> TinySpec {
        Mistral.write_tiny(dir, seed)
    }
}

static BROKEN_FAMILIES: Registry<dyn ModelFamily> =
    Registry::new("model_family", &[&Llama, &DropsLmHead]);

// ---- logits processors -------------------------------------------------------------------

/// Claims to run on the device, but moves the row's maximum to id 0 (a device reduction of
/// the raw row cannot know that).
struct BrokenProcessor;

impl Module for BrokenProcessor {
    fn name(&self) -> &'static str {
        "broken"
    }
}

impl LogitsProcessor for BrokenProcessor {
    fn applies(&self, p: &ProcessorParams, _: &ProcessorState<'_>) -> bool {
        !p.logit_bias.is_empty()
    }
    fn device_capable(&self) -> bool {
        true
    }
    fn needs_full_row(&self) -> bool {
        false
    }
    fn apply(
        &self,
        logits: &mut [f32],
        touched: &mut Touched,
        _: &ProcessorParams,
        _: &ProcessorState<'_>,
    ) {
        if let Some(i) = touched.touch(logits, 0) {
            logits[i] += 100.0;
        }
    }
}

static BROKEN_PROCESSORS: Registry<dyn LogitsProcessor> =
    Registry::new("logits_processor", &[&LogitBias, &BrokenProcessor]);

/// A suite fed a registry with a broken module reports exactly that module and the check it
/// breaks (the parser, the loaded weights, the device path), and every suite passes over the
/// real registry. Breaks if a suite iterates a fixed list instead of the registry it is given,
/// or stops checking the property the broken module violates.
#[test]
fn conformance_rejects_broken_module() {
    let failures = formats_suite(&BROKEN_FORMATS).expect_err("the broken format is rejected");
    assert_eq!(pairs(&failures), [("broken", "parse")], "{failures:#?}");
    assert!(failures.iter().all(|f| f.point == "tool_format"));

    let failures = families_suite(&BROKEN_FAMILIES).expect_err("the broken family is rejected");
    assert_eq!(pairs(&failures), [("broken", "load")], "{failures:#?}");
    assert!(
        failures[0].detail.contains("lm_head"),
        "{}",
        failures[0].detail
    );

    let failures =
        processors_suite(&BROKEN_PROCESSORS).expect_err("the broken processor is rejected");
    assert_eq!(pairs(&failures), [("broken", "device")], "{failures:#?}");

    formats_suite(formats::registry()).expect("every registered tool format conforms");
    families_suite(families::registry()).expect("every registered family conforms");
    processors_suite(sampling::registry()).expect("every registered processor conforms");
    weights_suite(turbine_model::weights::registry()).expect("every weight format conforms");
}
