//! Hugging Face `tokenizer.json` wrapper and streaming (incremental) detokenization (P1 S-4).
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::ModelError;

/// A loaded `tokenizer.json` (wraps `tokenizers::Tokenizer`).
#[derive(Clone)]
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    path: PathBuf,
}

impl fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tokenizer")
            .field("path", &self.path)
            .field("vocab_size", &self.vocab_size())
            .finish()
    }
}

impl Tokenizer {
    /// Loads a Hugging Face `tokenizer.json`.
    pub fn from_file(path: &Path) -> Result<Tokenizer, ModelError> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| ModelError::Io {
            path: path.to_path_buf(),
            detail: format!("cannot load tokenizer: {e}"),
        })?;
        Ok(Tokenizer {
            inner,
            path: path.to_path_buf(),
        })
    }

    /// Encodes `text`; `add_special_tokens` applies the tokenizer's post-processor (BOS for Llama-3).
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, ModelError> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| self.error("encode", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decodes `ids` to text (invalid UTF-8 byte sequences become U+FFFD).
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, ModelError> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| self.error("decode", e))
    }

    /// Vocabulary size including added tokens.
    pub fn vocab_size(&self) -> u32 {
        u32::try_from(self.inner.get_vocab_size(true)).unwrap_or(u32::MAX)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Adds `tokens` as special added tokens (new ids past the vocabulary): the conformance
    /// suite carries a format's declared markers this way, so its grammars can reference them
    /// as llguidance special-token atoms on the fixture tokenizer.
    pub fn add_special_tokens(&mut self, tokens: &[&str]) -> Result<(), ModelError> {
        let added: Vec<tokenizers::AddedToken> = tokens
            .iter()
            .map(|&text| tokenizers::AddedToken {
                content: text.to_string(),
                special: true,
                ..Default::default()
            })
            .collect();
        let _count = self.inner.add_tokens(&added);
        Ok(())
    }

    /// The wrapped `tokenizers::Tokenizer` (Phase 2 builds the llguidance token trie from it).
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }

    fn error(&self, op: &str, e: tokenizers::Error) -> ModelError {
        ModelError::Io {
            path: self.path.clone(),
            detail: format!("{op} failed: {e}"),
        }
    }
}

/// Turns a stream of generated token ids into text chunks without ever emitting a partial UTF-8
/// sequence: the window `ids[prefix_offset..]` is re-decoded on each push and only the text past
/// the previously emitted part is returned, and only once it no longer ends in U+FFFD. Special
/// ids declared through [`emitting_specials`](IncrementalDetokenizer::emitting_specials) also
/// emit their text (P7 decision "Harmony marker visibility").
pub struct IncrementalDetokenizer {
    tokenizer: Arc<Tokenizer>,
    /// Special token ids whose text is emitted (the bound tool format's markers, P7 decision
    /// "Harmony marker visibility").
    keep: Vec<u32>,
    ids: Vec<u32>,
    prefix_offset: usize,
    read_offset: usize,
}

impl IncrementalDetokenizer {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        IncrementalDetokenizer {
            tokenizer,
            keep: Vec::new(),
            ids: Vec::new(),
            prefix_offset: 0,
            read_offset: 0,
        }
    }

    /// Declares the special ids whose text the stream emits (the bound tool format's declared
    /// markers). Ids that already decode to text are ignored; ids not in the vocabulary too.
    /// The window decodes stay free of the markers because each one closes the segment before
    /// it ([`IncrementalDetokenizer::flush`]).
    pub fn emitting_specials(mut self, ids: impl IntoIterator<Item = u32>) -> Self {
        for id in ids {
            let skipped = self
                .tokenizer
                .decode(&[id], true)
                .is_ok_and(|text| text.is_empty());
            if skipped && !self.keep.contains(&id) {
                self.keep.push(id);
            }
        }
        self
    }

    /// Adds one token; returns the text it completes, or `None` while the pending bytes are an
    /// incomplete UTF-8 sequence (or the token decodes to nothing, e.g. a skipped special token
    /// that is not in the emit-list).
    pub fn push(&mut self, token: u32) -> Option<String> {
        self.ids.push(token);
        if self.keep.contains(&token) {
            // A declared marker: its own text is emitted on its own. Close the segment before
            // it first, in order (the flush is lossy on an incomplete UTF-8 sequence, as at
            // the end of generation), then emit the marker text.
            let held = self.flush();
            self.prefix_offset = self.ids.len();
            self.read_offset = self.ids.len();
            let text = self.tokenizer.id_to_token(token)?;
            return match held {
                Some(held) => Some(format!("{held}{text}")),
                None => Some(text),
            };
        }
        let prefix_text = self.decode(self.prefix_offset, self.read_offset)?;
        let new_text = self.decode(self.prefix_offset, self.ids.len())?;
        if new_text.len() <= prefix_text.len() || new_text.ends_with('\u{FFFD}') {
            return None;
        }
        let chunk = new_text.get(prefix_text.len()..)?.to_string();
        self.prefix_offset = self.read_offset;
        self.read_offset = self.ids.len();
        Some(chunk)
    }

    /// At the end of generation: any text still held back (lossy: an unfinished sequence
    /// becomes U+FFFD).
    pub fn flush(&mut self) -> Option<String> {
        if self.read_offset == self.ids.len() {
            // Every token's text was already emitted: nothing to decode.
            return None;
        }
        let prefix_text = self.decode(self.prefix_offset, self.read_offset)?;
        let new_text = self.decode(self.prefix_offset, self.ids.len())?;
        self.prefix_offset = self.ids.len();
        self.read_offset = self.ids.len();
        let rest = new_text.get(prefix_text.len()..)?;
        if rest.is_empty() {
            None
        } else {
            Some(rest.to_string())
        }
    }

    fn decode(&self, start: usize, end: usize) -> Option<String> {
        self.tokenizer.decode(&self.ids[start..end], true).ok()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use super::*;

    fn fixture_tokenizer() -> Tokenizer {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/llama-3.2-3b-instruct/tokenizer.json");
        Tokenizer::from_file(&path).expect("fixture tokenizer.json loads")
    }

    #[test]
    fn incremental_detokenize_utf8() {
        let tokenizer = Arc::new(fixture_tokenizer());
        let text = "Hello 世界 👩\u{200d}👩\u{200d}👧\u{200d}👦 🇧🇪 naïve café 日本語テキスト";
        let ids = tokenizer.encode(text, false).expect("encode");
        let full = tokenizer.decode(&ids, true).expect("decode");
        assert_eq!(full, text, "Llama-3 byte-level BPE round-trips");

        let mut detok = IncrementalDetokenizer::new(Arc::clone(&tokenizer));
        let mut streamed = String::new();
        let mut held = 0usize;
        for &id in &ids {
            match detok.push(id) {
                Some(chunk) => {
                    assert!(!chunk.is_empty(), "push never returns an empty chunk");
                    assert!(
                        !chunk.contains('\u{FFFD}'),
                        "chunk {chunk:?} contains U+FFFD after token {id}"
                    );
                    streamed.push_str(&chunk);
                }
                None => held += 1,
            }
        }
        if let Some(rest) = detok.flush() {
            assert!(!rest.contains('\u{FFFD}'));
            streamed.push_str(&rest);
        }
        assert_eq!(streamed, full);
        assert!(
            held > 0,
            "expected at least one token to end inside a UTF-8 sequence"
        );
    }

    #[test]
    fn special_tokens_and_lookups() {
        let tokenizer = fixture_tokenizer();
        assert_eq!(tokenizer.vocab_size(), 128_256);
        assert_eq!(tokenizer.token_to_id("<|begin_of_text|>"), Some(128_000));
        assert_eq!(tokenizer.token_to_id("<|eot_id|>"), Some(128_009));
        assert_eq!(
            tokenizer.id_to_token(128_009).as_deref(),
            Some("<|eot_id|>")
        );
        assert_eq!(tokenizer.token_to_id("no-such-token-xyz"), None);
        let with_bos = tokenizer.encode("Hi", true).expect("encode");
        assert_eq!(with_bos.first(), Some(&128_000));
        let without = tokenizer.encode("Hi", false).expect("encode");
        assert_eq!(&with_bos[1..], &without[..]);
        assert_eq!(tokenizer.decode(&with_bos, true).expect("decode"), "Hi");
        assert_eq!(
            tokenizer.decode(&with_bos, false).expect("decode"),
            "<|begin_of_text|>Hi"
        );
        assert_eq!(tokenizer.inner().get_vocab_size(true), 128_256);
    }

    #[test]
    fn special_token_is_skipped_in_stream() {
        let tokenizer = Arc::new(fixture_tokenizer());
        let mut ids = tokenizer.encode("A", false).expect("encode");
        ids.push(128_009);
        ids.extend(tokenizer.encode(" B", false).expect("encode"));
        let mut detok = IncrementalDetokenizer::new(Arc::clone(&tokenizer));
        let out: String = ids.iter().filter_map(|&id| detok.push(id)).collect();
        assert_eq!(out, "A B");
        assert_eq!(detok.flush(), None);
    }

    /// A declared marker id emits its text mid-stream, in order with the text around it.
    #[test]
    fn declared_marker_is_emitted_in_stream() {
        let tokenizer = Arc::new(fixture_tokenizer());
        let eot = tokenizer.token_to_id("<|eot_id|>").expect("eot id");
        let ids = tokenizer.encode("A<|eot_id|>B", false).expect("encode");
        let mut detok =
            IncrementalDetokenizer::new(Arc::clone(&tokenizer)).emitting_specials([eot]);
        let out: String = ids.iter().filter_map(|&id| detok.push(id)).collect();
        assert_eq!(out, "A<|eot_id|>B");
        assert_eq!(detok.flush(), None);
    }

    /// `emitting_specials` only keeps ids that actually skip: a token that already decodes
    /// to text is left to the normal window path.
    #[test]
    fn emitting_specials_ignores_text_ids() {
        let tokenizer = Arc::new(fixture_tokenizer());
        let plain = tokenizer.encode("A", false).expect("encode")[0];
        let detok =
            IncrementalDetokenizer::new(Arc::clone(&tokenizer)).emitting_specials([plain, 128_009]);
        assert_eq!(detok.keep, vec![128_009]);
    }

    /// With every pushed token already emitted, `flush` is a no-op: `None`, and it leaves the
    /// window alone instead of re-decoding it (Scout 806bcf1e).
    #[test]
    fn flush_with_nothing_pending_is_a_no_op() {
        let tokenizer = Arc::new(fixture_tokenizer());
        let ids = tokenizer.encode("Hello world", false).expect("encode");
        let mut detok = IncrementalDetokenizer::new(Arc::clone(&tokenizer));
        let out: String = ids.iter().filter_map(|&id| detok.push(id)).collect();
        assert_eq!(out, "Hello world");
        assert_eq!(detok.read_offset, ids.len(), "everything was emitted");
        let window = (detok.prefix_offset, detok.read_offset);
        assert_eq!(detok.flush(), None);
        assert_eq!((detok.prefix_offset, detok.read_offset), window);
    }

    #[test]
    fn missing_file_is_io_error() {
        let path = PathBuf::from("/nonexistent/turbine/tokenizer.json");
        match Tokenizer::from_file(&path) {
            Err(ModelError::Io { path: p, .. }) => assert_eq!(p, path),
            other => panic!("expected Io error, got {other:?}"),
        }
    }
}
