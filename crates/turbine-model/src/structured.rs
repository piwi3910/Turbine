//! Constrained decoding (P2 S-17, S-18): the allowed-token bitmask, the matcher trait the engine
//! drives each decode step, and the llguidance-backed [`GrammarCompiler`].
//!
//! Per step the engine calls [`step_mask`] (the matcher fills the mask; EOS stays disallowed
//! until the matcher accepts), passes the mask to `Sampler::sample`, then commits the sampled
//! token to the matcher. Compilation is bounded ([`GrammarLimits`]) and runs off the engine
//! thread; llguidance's per-step parser limits stay at their defaults, and a step that exceeds
//! them fails with [`ModelError::Constraint`] for that request only.
use std::sync::Arc;
use std::time::Instant;

use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::SimpleVob;
use llguidance::{Matcher, ParserFactory};
use toktrie_hf_tokenizers::{ByteTokenizer, ByteTokenizerEnv};
use turbine_core::request::ConstraintSpec;

use crate::ModelError;
use crate::metrics::ModelMetrics;
use crate::tokenizer::Tokenizer;

/// A bitset over the vocabulary: bit `id` set = token `id` may be sampled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenMask {
    words: Vec<u32>,
    vocab: usize,
}

impl TokenMask {
    /// Every id of `vocab` allowed.
    pub fn new_all(vocab: usize) -> TokenMask {
        let mut mask = TokenMask {
            words: vec![u32::MAX; vocab.div_ceil(32)],
            vocab,
        };
        mask.clear_tail();
        mask
    }

    /// No id allowed.
    pub fn new_none(vocab: usize) -> TokenMask {
        TokenMask {
            words: vec![0; vocab.div_ceil(32)],
            vocab,
        }
    }

    /// Ids `0..vocab` this mask covers.
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Disallows every id.
    pub fn clear(&mut self) {
        self.words.fill(0);
    }

    /// Allows `id`; ids outside the vocabulary are ignored.
    pub fn allow(&mut self, id: u32) {
        if (id as usize) < self.vocab {
            self.words[id as usize / 32] |= 1 << (id % 32);
        }
    }

    /// Disallows `id`; ids outside the vocabulary are ignored.
    pub fn disallow(&mut self, id: u32) {
        if (id as usize) < self.vocab {
            self.words[id as usize / 32] &= !(1 << (id % 32));
        }
    }

    /// Whether `id` may be sampled (ids outside the vocabulary never may).
    pub fn is_allowed(&self, id: u32) -> bool {
        (id as usize) < self.vocab && self.words[id as usize / 32] & (1 << (id % 32)) != 0
    }

    /// Number of allowed ids.
    pub fn count_allowed(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Sets every disallowed logit to −∞; logits past the mask's vocabulary (padding rows of the
    /// LM head) are disallowed too.
    pub fn apply(&self, logits: &mut [f32]) {
        for (chunk, &word) in logits.chunks_mut(32).zip(&self.words) {
            if word == u32::MAX {
                continue;
            }
            for (bit, v) in chunk.iter_mut().enumerate() {
                if word & (1 << bit) == 0 {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
        let covered = self.words.len() * 32;
        if logits.len() > covered {
            logits[covered..].fill(f32::NEG_INFINITY);
        }
    }

    /// Replaces the bits with llguidance's mask (same bit layout: 32-bit words, LSB first).
    fn copy_from(&mut self, vob: &SimpleVob) {
        self.clear();
        for (dst, &src) in self.words.iter_mut().zip(vob.as_slice()) {
            *dst = src;
        }
        self.clear_tail();
    }

    /// Clears the bits past `vocab` in the last word.
    fn clear_tail(&mut self) {
        let rem = self.vocab % 32;
        if rem != 0
            && let Some(last) = self.words.last_mut()
        {
            *last &= (1u32 << rem) - 1;
        }
    }
}

/// One request's constrained-decoding state, driven by the engine thread once per decode step.
pub trait TokenMatcher: Send {
    /// Overwrites `mask` with the ids the constraint allows next.
    fn allowed(&mut self, mask: &mut TokenMask) -> Result<(), ModelError>;
    /// Advances past the sampled `token`.
    fn commit(&mut self, token: u32) -> Result<(), ModelError>;
    /// Whether the output so far is complete, so EOS may end it.
    fn accepts_eos(&self) -> bool;
}

/// Fills `mask` for the next step of `matcher`: its allowed ids, with every id of
/// `eos_token_ids` disallowed until the matcher accepts. A mask that allows nothing is an error
/// (the constraint is stuck).
pub fn step_mask(
    matcher: &mut dyn TokenMatcher,
    eos_token_ids: &[u32],
    mask: &mut TokenMask,
) -> Result<(), ModelError> {
    matcher.allowed(mask)?;
    if !matcher.accepts_eos() {
        for &id in eos_token_ids {
            mask.disallow(id);
        }
    }
    if mask.count_allowed() == 0 {
        return Err(ModelError::Constraint(
            "the constraint allows no token".to_string(),
        ));
    }
    Ok(())
}

/// Bounds on what a request may ask to compile (TS §21 rule 8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrammarLimits {
    /// `structured_output.max_schema_bytes`: the compact JSON of a schema, or the grammar source.
    pub max_schema_bytes: usize,
}

/// The `kind` label of `turbine_grammar_compile_seconds` for `spec`.
pub fn constraint_kind(spec: &ConstraintSpec) -> &'static str {
    match spec {
        ConstraintSpec::JsonObject => "json_object",
        ConstraintSpec::JsonSchema { .. } => "json_schema",
        ConstraintSpec::ToolCall { .. } => "tool_call",
        _ => "other",
    }
}

/// The longest run of whitespace a constrained JSON output may put between two tokens of JSON
/// syntax (outside strings): room for a newline and an indentation four levels deep, never
/// unbounded, so a bounded schema still yields bounded output.
pub const JSON_MAX_WHITESPACE: usize = 16;

/// The llguidance JSON options (`x-guidance`) of every JSON constraint (`json_object`,
/// `json_schema` and tool-call parameters): the separators `,` and `:` with between any two
/// JSON tokens at most [`JSON_MAX_WHITESPACE`] characters of JSON whitespace. The model writes
/// JSON the way it learned it (`{"name": "John"}`, pretty-printed or compact); forcing compact
/// JSON pushes it to put the spacing it wants inside strings instead (`{"name":": "}`).
pub fn json_options() -> serde_json::Value {
    serde_json::json!({
        "item_separator": ",",
        "key_separator": ":",
        "whitespace_flexible": false,
        "whitespace_pattern": format!("[\\x20\\x0A\\x0D\\x09]{{1,{JSON_MAX_WHITESPACE}}}")
    })
}

/// `schema` with [`json_options`], replacing any caller-supplied `x-guidance` options.
pub(crate) fn with_json_options(schema: &serde_json::Value) -> serde_json::Value {
    let mut schema = match schema {
        serde_json::Value::Bool(true) => serde_json::json!({}),
        other => other.clone(),
    };
    if let Some(obj) = schema.as_object_mut() {
        obj.insert("x-guidance".to_string(), json_options());
    }
    schema
}

/// Compiles [`ConstraintSpec`]s into matchers over one tokenizer. The token trie is built once
/// (at startup) and shared by every compiled matcher; `compile` may run on any thread.
pub struct GrammarCompiler {
    factory: Arc<ParserFactory>,
    eos_token_ids: Vec<u32>,
    metrics: Option<ModelMetrics>,
}

impl GrammarCompiler {
    /// Builds the llguidance token trie from `tokenizer` (the same `tokenizers` 0.21 crate the
    /// model layer loads); `eos_token_ids` are the ids that end a constrained output.
    pub fn new(
        tokenizer: &Tokenizer,
        eos_token_ids: &[u32],
    ) -> Result<GrammarCompiler, ModelError> {
        let vocab = tokenizer.vocab_size();
        if eos_token_ids.is_empty() {
            return Err(ModelError::Constraint(
                "token trie: no EOS token id".to_string(),
            ));
        }
        if let Some(bad) = eos_token_ids.iter().find(|&&id| id >= vocab) {
            return Err(ModelError::Constraint(format!(
                "token trie: EOS id {bad} is outside the vocabulary of {vocab}"
            )));
        }
        let mut byte_tokenizer = ByteTokenizer::from_tokenizer(tokenizer.inner().clone())
            .map_err(|e| ModelError::Constraint(format!("token trie: {e}")))?;
        byte_tokenizer.set_eos_tokens(eos_token_ids);
        let env = ByteTokenizerEnv::new(byte_tokenizer, None)
            .map_err(|e| ModelError::Constraint(format!("token trie: {e}")))?
            .to_env();
        let mut factory = ParserFactory::new_simple(&env)
            .map_err(|e| ModelError::Constraint(format!("token trie: {e}")))?;
        // Errors come back as values; nothing goes to stderr.
        factory.quiet();
        Ok(GrammarCompiler {
            factory: Arc::new(factory),
            eos_token_ids: eos_token_ids.to_vec(),
            metrics: None,
        })
    }

    /// Records compile and mask durations in `metrics` from now on.
    pub fn with_metrics(mut self, metrics: ModelMetrics) -> GrammarCompiler {
        self.metrics = Some(metrics);
        self
    }

    /// Compiles `spec` into a fresh matcher. A source over `limits`, a keyword llguidance does
    /// not support (named in the message) or any other grammar error is
    /// [`ModelError::Constraint`].
    pub fn compile(
        &self,
        spec: &ConstraintSpec,
        limits: &GrammarLimits,
    ) -> Result<Box<dyn TokenMatcher>, ModelError> {
        let started = Instant::now();
        let kind = constraint_kind(spec);
        let check_size = |bytes: usize| {
            if bytes > limits.max_schema_bytes {
                Err(ModelError::Constraint(format!(
                    "{kind}: {bytes} bytes exceed structured_output.max_schema_bytes ({})",
                    limits.max_schema_bytes
                )))
            } else {
                Ok(())
            }
        };
        let grammar = match spec {
            ConstraintSpec::JsonObject => TopLevelGrammar::from_json_schema(with_json_options(
                &serde_json::json!({"type": "object"}),
            )),
            ConstraintSpec::JsonSchema { schema } => {
                let bytes = serde_json::to_vec(schema)
                    .map_err(|e| ModelError::Constraint(format!("{kind}: {e}")))?
                    .len();
                check_size(bytes)?;
                TopLevelGrammar::from_json_schema(with_json_options(schema))
            }
            ConstraintSpec::ToolCall { grammar_source } => {
                check_size(grammar_source.len())?;
                TopLevelGrammar::from_lark(grammar_source.clone())
            }
            _ => {
                return Err(ModelError::Constraint(format!(
                    "unsupported constraint {spec:?}"
                )));
            }
        };
        let parser = self
            .factory
            .create_parser(grammar)
            .map_err(|e| ModelError::Constraint(format!("{kind}: {e}")))?;
        let mut matcher = Matcher::new(Ok(parser));
        let accepting = matcher
            .is_accepting()
            .map_err(|e| ModelError::Constraint(format!("{kind}: {e}")))?;
        if let Some(metrics) = &self.metrics {
            metrics.observe_grammar_compile(kind, started.elapsed().as_secs_f64());
        }
        Ok(Box::new(LlguidanceMatcher {
            matcher,
            eos_token_ids: self.eos_token_ids.clone(),
            accepting,
            metrics: self.metrics.clone(),
        }))
    }
}

/// A [`TokenMatcher`] over one `llguidance::Matcher`.
struct LlguidanceMatcher {
    matcher: Matcher,
    eos_token_ids: Vec<u32>,
    /// `Matcher::is_accepting` after the last commit (it needs `&mut`).
    accepting: bool,
    metrics: Option<ModelMetrics>,
}

fn constraint_error(e: impl std::fmt::Display) -> ModelError {
    ModelError::Constraint(e.to_string())
}

impl TokenMatcher for LlguidanceMatcher {
    fn allowed(&mut self, mask: &mut TokenMask) -> Result<(), ModelError> {
        let started = Instant::now();
        let vob = self
            .matcher
            .compute_mask_or_eos()
            .map_err(constraint_error)?;
        mask.copy_from(&vob);
        // llguidance marks EOS in its mask when the output may end; mirror that onto every
        // EOS id of the model.
        let eos_allowed = self
            .eos_token_ids
            .iter()
            .any(|&id| (id as usize) < vob.len() && vob.is_allowed(id));
        for &id in &self.eos_token_ids {
            if eos_allowed {
                mask.allow(id);
            } else {
                mask.disallow(id);
            }
        }
        if let Some(metrics) = &self.metrics {
            metrics.observe_token_mask(started.elapsed().as_secs_f64());
        }
        Ok(())
    }

    fn commit(&mut self, token: u32) -> Result<(), ModelError> {
        if self.eos_token_ids.contains(&token) {
            return if self.accepting {
                Ok(())
            } else {
                Err(ModelError::Constraint(format!(
                    "EOS {token} before the constraint is complete"
                )))
            };
        }
        self.matcher
            .consume_token(token)
            .map_err(constraint_error)?;
        self.accepting = self.matcher.is_accepting().map_err(constraint_error)?;
        Ok(())
    }

    fn accepts_eos(&self) -> bool {
        self.accepting
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::PathBuf;

    use rand_chacha::ChaCha8Rng;
    use rand_core::{RngCore, SeedableRng};
    use turbine_core::request::{ConstraintSpec, SamplingParams};

    use super::*;
    use crate::ModelError;
    use crate::sampler::Sampler;
    use crate::tokenizer::Tokenizer;

    /// Allows exactly `ids`; accepts EOS once `accept_after` tokens were committed.
    struct MockMatcher {
        ids: Vec<u32>,
        committed: usize,
        accept_after: usize,
    }

    impl TokenMatcher for MockMatcher {
        fn allowed(&mut self, mask: &mut TokenMask) -> Result<(), ModelError> {
            mask.clear();
            for &id in &self.ids {
                mask.allow(id);
            }
            Ok(())
        }
        fn commit(&mut self, token: u32) -> Result<(), ModelError> {
            assert!(self.ids.contains(&token), "committed disallowed {token}");
            self.committed += 1;
            Ok(())
        }
        fn accepts_eos(&self) -> bool {
            self.committed >= self.accept_after
        }
    }

    #[test]
    fn token_mask_bits_and_padding() {
        let mut mask = TokenMask::new_none(40);
        mask.allow(0);
        mask.allow(33);
        mask.allow(40); // outside the vocabulary: ignored
        assert_eq!(mask.count_allowed(), 2);
        assert!(mask.is_allowed(33) && !mask.is_allowed(34) && !mask.is_allowed(40));
        // Logits past the mask's vocabulary (LM-head padding) are disallowed too.
        let mut logits = vec![1.0f32; 70];
        mask.apply(&mut logits);
        let finite: Vec<usize> = (0..70).filter(|&i| logits[i].is_finite()).collect();
        assert_eq!(finite, vec![0, 33]);
        let all = TokenMask::new_all(40);
        assert_eq!(all.count_allowed(), 40);
        mask.disallow(33);
        assert_eq!(mask.count_allowed(), 1);
    }

    #[test]
    fn compiler_is_shared_across_threads() {
        // compile() runs on Tokio's blocking pool while the engine thread owns the matchers.
        fn shared<T: Send + Sync>() {}
        shared::<GrammarCompiler>();
    }

    #[test]
    fn mask_applied_before_sampling() {
        const VOCAB: usize = 16;
        const EOS: u32 = 9;
        let mut raw: Vec<f32> = (0..VOCAB).map(|i| i as f32 * 0.1).collect();
        raw[EOS as usize] = 5.0; // EOS is the favourite once allowed
        raw[3] = 1.0;
        let bias = vec![(3u32, 100.0f32)];

        // Greedy: 5 until the matcher accepts, then EOS; never the biased id 3.
        let greedy = SamplingParams {
            temperature: 0.0,
            logit_bias: bias.clone(),
            seed: Some(1),
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(&greedy, &[], &[EOS]);
        let mut matcher = MockMatcher {
            ids: vec![5, 9],
            committed: 0,
            accept_after: 3,
        };
        let mut mask = TokenMask::new_none(VOCAB);
        for step in 0..6 {
            step_mask(&mut matcher, &[EOS], &mut mask).expect("mask");
            let s = sampler.sample(&mut raw.clone(), Some(&mask));
            let expected = if matcher.accepts_eos() { EOS } else { 5 };
            assert_eq!(s.token, expected, "greedy step {step}");
            matcher.commit(s.token).expect("commit");
            sampler.observe(s.token);
        }

        // Seeded sampling: 200 draws stay inside {5, 9}; 9 only once the matcher accepts.
        let sampled = SamplingParams {
            temperature: 1.0,
            logit_bias: bias,
            seed: Some(7),
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(&sampled, &[], &[EOS]);
        let mut matcher = MockMatcher {
            ids: vec![5, 9],
            committed: 0,
            accept_after: 50,
        };
        let mut eos_after_accept = 0;
        for step in 0..200 {
            step_mask(&mut matcher, &[EOS], &mut mask).expect("mask");
            let accepting = matcher.accepts_eos();
            let s = sampler.sample(&mut raw.clone(), Some(&mask));
            assert!(s.token == 5 || s.token == EOS, "step {step}: {}", s.token);
            if s.token == EOS {
                assert!(
                    accepting,
                    "EOS sampled at step {step} before the matcher accepts"
                );
                eos_after_accept += 1;
            }
            matcher.commit(s.token).expect("commit");
            sampler.observe(s.token);
        }
        assert!(eos_after_accept > 0, "EOS is reachable once accepted");
    }

    fn llama_tokenizer() -> Tokenizer {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/llama-3.2-3b-instruct/tokenizer.json");
        Tokenizer::from_file(&path).expect("committed Llama tokenizer")
    }

    /// Runs `text` (tokenized by `tokenizer`, special tokens such as `<|python_tag|>` included)
    /// through `matcher`, checking each token against the step mask first. `Ok(accepting)` when
    /// every token was allowed; `Err` names the first disallowed token.
    pub(crate) fn feed_text(
        matcher: &mut dyn TokenMatcher,
        tokenizer: &Tokenizer,
        eos: &[u32],
        text: &str,
    ) -> Result<bool, String> {
        let ids = tokenizer.encode(text, false).expect("encode");
        let mut mask = TokenMask::new_none(tokenizer.vocab_size() as usize);
        for (i, &id) in ids.iter().enumerate() {
            step_mask(matcher, eos, &mut mask).map_err(|e| format!("token {i}: {e}"))?;
            if !mask.is_allowed(id) {
                let done = tokenizer.decode(&ids[..i], false).expect("decode");
                let piece = tokenizer.decode(&[id], false).expect("decode");
                return Err(format!("token {i} {piece:?} disallowed after {done:?}"));
            }
            matcher.commit(id).map_err(|e| format!("token {i}: {e}"))?;
        }
        Ok(matcher.accepts_eos())
    }

    pub(crate) fn llama_compiler() -> (Tokenizer, GrammarCompiler) {
        let tokenizer = llama_tokenizer();
        let compiler = GrammarCompiler::new(&tokenizer, &LLAMA_EOS).expect("token trie");
        (tokenizer, compiler)
    }

    pub(crate) const LLAMA_EOS: [u32; 3] = [128_001, 128_008, 128_009];

    /// The longest run of whitespace outside JSON string literals in `text`.
    pub(crate) fn longest_whitespace_outside_strings(text: &str) -> usize {
        let (mut in_string, mut escaped) = (false, false);
        let (mut run, mut longest) = (0, 0);
        for c in text.chars() {
            if in_string {
                match (escaped, c) {
                    (true, _) => escaped = false,
                    (false, '\\') => escaped = true,
                    (false, '"') => in_string = false,
                    _ => {}
                }
            } else if c.is_whitespace() {
                run += 1;
                longest = longest.max(run);
                continue;
            } else if c == '"' {
                in_string = true;
            }
            run = 0;
        }
        longest
    }

    #[test]
    fn json_whitespace_natural_and_bounded() {
        let (tokenizer, compiler) = llama_compiler();
        let limits = GrammarLimits {
            max_schema_bytes: 64 * 1024,
        };
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"}
            },
            "required": ["name", "age", "city"],
            "additionalProperties": false
        });
        let run = |spec: &ConstraintSpec, text: &str| {
            let mut matcher = compiler.compile(spec, &limits).expect("compile");
            feed_text(matcher.as_mut(), &tokenizer, &LLAMA_EOS, text)
        };
        let person = ConstraintSpec::JsonSchema { schema };
        let pad = " ".repeat(JSON_MAX_WHITESPACE);
        // What Llama writes (`": "`, `", "`), compact JSON, pretty-printed JSON and a gap of
        // exactly the bound are all complete outputs.
        for text in [
            r#"{"name": "John", "age": 30, "city": "Paris"}"#.to_string(),
            r#"{"name":"John","age":30,"city":"Paris"}"#.to_string(),
            "{\n  \"name\": \"John\",\n  \"age\": 30,\n  \"city\": \"Paris\"\n}".to_string(),
            format!(r#"{{"name":{pad}"John", "age": 30, "city": "Paris"}}"#),
        ] {
            assert_eq!(run(&person, &text), Ok(true), "{text:?}");
            assert!(longest_whitespace_outside_strings(&text) <= JSON_MAX_WHITESPACE);
        }
        // One whitespace character past the bound, and a string where the schema wants an
        // integer, are refused.
        let over = format!(r#"{{"name": {pad}"John", "age": 30, "city": "Paris"}}"#);
        assert!(run(&person, &over).is_err(), "{over:?} accepted");
        let wrong_type = r#"{"name": "John", "age": "30", "city": "Paris"}"#;
        assert!(run(&person, wrong_type).is_err(), "{wrong_type:?} accepted");
        // json_object takes the same natural whitespace.
        assert_eq!(
            run(
                &ConstraintSpec::JsonObject,
                r#"{"a": [1, 2], "b": {"c": true}}"#
            ),
            Ok(true)
        );
    }

    #[test]
    fn json_schema_matcher_on_llama_tokenizer() {
        const EOS: [u32; 3] = [128_001, 128_008, 128_009];
        let tokenizer = llama_tokenizer();
        let compiler = GrammarCompiler::new(&tokenizer, &EOS).expect("token trie");
        let limits = GrammarLimits {
            max_schema_bytes: 64 * 1024,
        };
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": false
        });
        let validator = jsonschema::validator_for(&schema).expect("valid schema");
        let vocab = tokenizer.vocab_size() as usize;
        for seed in 0..4u64 {
            let mut matcher = compiler
                .compile(
                    &ConstraintSpec::JsonSchema {
                        schema: schema.clone(),
                    },
                    &limits,
                )
                .expect("compile");
            let greedy = SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            };
            let mut sampler = Sampler::new(&greedy, &[], &EOS);
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut mask = TokenMask::new_none(vocab);
            let mut out = Vec::new();
            let mut ended = false;
            for _ in 0..256 {
                step_mask(matcher.as_mut(), &EOS, &mut mask).expect("mask");
                let mut logits: Vec<f32> = (0..vocab)
                    .map(|_| (rng.next_u32() >> 8) as f32 / (1u32 << 24) as f32)
                    .collect();
                let token = sampler.sample(&mut logits, Some(&mask)).token;
                if EOS.contains(&token) {
                    assert!(matcher.accepts_eos(), "EOS only when accepting");
                    ended = true;
                    break;
                }
                matcher.commit(token).expect("commit");
                sampler.observe(token);
                out.push(token);
            }
            assert!(
                ended,
                "seed {seed}: the grammar reached EOS within 256 tokens"
            );
            let text = tokenizer.decode(&out, false).expect("decode");
            let value: serde_json::Value =
                serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
            assert!(validator.is_valid(&value), "seed {seed}: {text}");
            assert!(
                longest_whitespace_outside_strings(&text) <= JSON_MAX_WHITESPACE,
                "whitespace over the bound: {text:?}"
            );
        }

        // An unsupported keyword is named; a schema over the byte bound is rejected.
        let unsupported = serde_json::json!({"type": "array", "uniqueItems": true});
        let err = compiler
            .compile(
                &ConstraintSpec::JsonSchema {
                    schema: unsupported,
                },
                &limits,
            )
            .err()
            .expect("uniqueItems is not supported");
        assert!(
            matches!(&err, ModelError::Constraint(m) if m.contains("uniqueItems")),
            "{err}"
        );
        let err = compiler
            .compile(
                &ConstraintSpec::JsonSchema { schema },
                &GrammarLimits {
                    max_schema_bytes: 16,
                },
            )
            .err()
            .expect("over the byte bound");
        assert!(
            matches!(&err, ModelError::Constraint(m) if m.contains("max_schema_bytes")),
            "{err}"
        );
    }
}
