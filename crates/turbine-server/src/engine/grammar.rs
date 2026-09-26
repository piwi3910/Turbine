//! Grammar compilation for constrained requests (P2 S-17, S-18, §Constraints): a
//! `response_format` schema or a `tool_choice` grammar is compiled into one matcher per choice
//! *before* the request is queued, never on the engine thread. Compilation runs on Tokio's
//! blocking pool behind a semaphore of [`COMPILE_PERMITS`] permits, and the whole wait (permit
//! plus compilation) is bounded by `structured_output.compile_timeout`. Every failure — a
//! grammar llguidance rejects, an unsupported keyword, a source over
//! `structured_output.max_schema_bytes`, or the timeout — is 400 `invalid_json_schema` naming
//! the cause.
//!
//! A compilation that times out keeps its permit until the blocking task returns, so at most
//! [`COMPILE_PERMITS`] compilations ever run at once.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Semaphore;
use turbine_api::ApiError;
use turbine_core::config::StructuredOutputConfig;
use turbine_core::request::ConstraintSpec;
use turbine_model::{GrammarCompiler, GrammarLimits, ModelMetrics, TokenMatcher, constraint_kind};

/// Grammar compilations that may run at once on the blocking pool.
pub const COMPILE_PERMITS: usize = 4;

/// Compiles request constraints off the engine thread (shared by every request).
pub struct GrammarService {
    compiler: Arc<GrammarCompiler>,
    permits: Arc<Semaphore>,
    limits: GrammarLimits,
    timeout: Duration,
    /// `turbine_grammar_compile_seconds{kind}`.
    metrics: Option<ModelMetrics>,
}

impl GrammarService {
    /// A service over `compiler` (built once at startup) with the `structured_output` bounds;
    /// compile durations go to `metrics`.
    pub fn new(
        compiler: Arc<GrammarCompiler>,
        config: &StructuredOutputConfig,
        metrics: ModelMetrics,
    ) -> GrammarService {
        let mut service = GrammarService::with_bounds(
            compiler,
            GrammarLimits {
                max_schema_bytes: usize::try_from(config.max_schema_bytes.0).unwrap_or(usize::MAX),
            },
            config.compile_timeout.0,
        );
        service.metrics = Some(metrics);
        service
    }

    fn with_bounds(
        compiler: Arc<GrammarCompiler>,
        limits: GrammarLimits,
        timeout: Duration,
    ) -> GrammarService {
        GrammarService {
            compiler,
            permits: Arc::new(Semaphore::new(COMPILE_PERMITS)),
            limits,
            timeout,
            metrics: None,
        }
    }

    /// Compiles `spec` into `choices` independent matchers (each choice of `n` > 1 keeps its
    /// own position). Errors are 400 `invalid_json_schema` naming the llguidance error or the
    /// bound that was exceeded.
    pub async fn compile(
        &self,
        spec: &ConstraintSpec,
        choices: u32,
    ) -> Result<Vec<Box<dyn TokenMatcher>>, ApiError> {
        let permits = Arc::clone(&self.permits);
        let compiler = Arc::clone(&self.compiler);
        let limits = self.limits;
        let owned = spec.clone();
        let metrics = self.metrics.clone();
        let work = async move {
            // The semaphore is never closed, so acquiring only waits.
            let permit = permits
                .acquire_owned()
                .await
                .map_err(|e| ApiError::internal(format!("grammar compilation: {e}")))?;
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                let started = std::time::Instant::now();
                let compiled = (0..choices.max(1))
                    .map(|_| compiler.compile(&owned, &limits))
                    .collect::<Result<Vec<_>, _>>();
                if let Some(m) = &metrics {
                    m.observe_grammar_compile(
                        constraint_kind(&owned),
                        started.elapsed().as_secs_f64(),
                    );
                }
                compiled
            })
            .await
            .map_err(|e| ApiError::internal(format!("grammar compilation task: {e}")))?
            .map_err(|e| ApiError::invalid_json_schema(e.to_string()))
        };
        match tokio::time::timeout(self.timeout, work).await {
            Ok(result) => result,
            Err(_) => Err(ApiError::invalid_json_schema(format!(
                "grammar compilation did not finish within structured_output.compile_timeout \
                 ({} ms)",
                self.timeout.as_millis()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use turbine_core::request::ErrorCode;
    use turbine_model::testing::TempDir;
    use turbine_model::testing::tiny::{TINY_EOS, write_tiny_llama};
    use turbine_model::{TokenMask, Tokenizer, ToolChoice, step_mask, tool_call_grammar};

    use super::*;

    fn compiler(dir: &TempDir) -> Arc<GrammarCompiler> {
        let spec = write_tiny_llama(dir.path(), 3);
        let tokenizer = Tokenizer::from_file(&spec.dir.join("tokenizer.json")).expect("tokenizer");
        Arc::new(GrammarCompiler::new(&tokenizer, &TINY_EOS).expect("token trie"))
    }

    fn service(dir: &TempDir, max_schema_bytes: usize, timeout: Duration) -> GrammarService {
        GrammarService::with_bounds(compiler(dir), GrammarLimits { max_schema_bytes }, timeout)
    }

    /// Commits the bytes of `text` (tiny tokenizer: byte `b` is token `b`), checking each one
    /// is allowed first.
    fn feed(matcher: &mut dyn TokenMatcher, text: &str) {
        let mut mask = TokenMask::new_none(263);
        for b in text.bytes() {
            step_mask(matcher, &TINY_EOS, &mut mask).expect("mask");
            assert!(mask.is_allowed(u32::from(b)), "{:?} not allowed", b as char);
            matcher.commit(u32::from(b)).expect("commit");
        }
    }

    #[tokio::test]
    async fn compiles_one_matcher_per_choice() {
        let dir = TempDir::new("turbine-grammar-choices");
        let grammars = service(&dir, 64 * 1024, Duration::from_secs(30));
        let mut matchers = grammars
            .compile(&ConstraintSpec::JsonObject, 2)
            .await
            .expect("json_object compiles");
        assert_eq!(matchers.len(), 2);
        // Advancing one choice leaves the other at the start.
        feed(matchers[0].as_mut(), "{}");
        assert!(matchers[0].accepts_eos());
        assert!(!matchers[1].accepts_eos());
        feed(matchers[1].as_mut(), "{\"a\":1}");
        assert!(matchers[1].accepts_eos());

        // A tool-call grammar compiles through the same path.
        let tools = [serde_json::json!({
            "type": "function",
            "function": {"name": "f", "parameters": {"type": "object"}}
        })];
        let spec = tool_call_grammar(&tools, &ToolChoice::Named("f".into()), true).expect("spec");
        let mut matchers = grammars.compile(&spec, 1).await.expect("tool grammar");
        feed(
            matchers[0].as_mut(),
            "{\"name\": \"f\", \"parameters\": {}}",
        );
        assert!(matchers[0].accepts_eos());
    }

    #[tokio::test]
    async fn bad_schemas_are_invalid_json_schema() {
        let dir = TempDir::new("turbine-grammar-bad");
        let grammars = service(&dir, 1024, Duration::from_secs(30));
        let unsupported = ConstraintSpec::JsonSchema {
            schema: serde_json::json!({"type": "array", "uniqueItems": true}),
        };
        let e = grammars
            .compile(&unsupported, 1)
            .await
            .err()
            .expect("rejected");
        assert_eq!(
            (e.status.as_u16(), e.code),
            (400, ErrorCode::InvalidJsonSchema)
        );
        assert!(e.message.contains("uniqueItems"), "{}", e.message);

        let big = ConstraintSpec::JsonSchema {
            schema: serde_json::json!({"type": "string", "description": "x".repeat(2048)}),
        };
        let e = grammars.compile(&big, 1).await.err().expect("rejected");
        assert_eq!(e.code, ErrorCode::InvalidJsonSchema);
        assert!(e.message.contains("max_schema_bytes"), "{}", e.message);
    }

    #[tokio::test]
    async fn permits_bound_compilations_and_the_timeout_covers_the_wait() {
        let dir = TempDir::new("turbine-grammar-timeout");
        let grammars = service(&dir, 64 * 1024, Duration::from_millis(100));
        assert_eq!(grammars.permits.available_permits(), COMPILE_PERMITS);
        // Every permit is taken: the next compilation waits and runs into the timeout.
        let held = Arc::clone(&grammars.permits)
            .acquire_many_owned(COMPILE_PERMITS as u32)
            .await
            .expect("permits");
        let e = grammars
            .compile(&ConstraintSpec::JsonObject, 1)
            .await
            .err()
            .expect("timed out");
        assert_eq!(e.code, ErrorCode::InvalidJsonSchema);
        assert!(e.message.contains("compile_timeout"), "{}", e.message);
        drop(held);
        assert!(
            grammars
                .compile(&ConstraintSpec::JsonObject, 1)
                .await
                .is_ok()
        );
        assert_eq!(grammars.permits.available_permits(), COMPILE_PERMITS);
    }
}
