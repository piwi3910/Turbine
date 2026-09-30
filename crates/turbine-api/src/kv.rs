//! Phase 4 request vocabulary of the KV hierarchy (P4 §Session hints, §HTTP routes): the
//! `x-turbine-*` request headers, `prompt_cache_key` validation and the
//! `POST /turbine/v1/kv/prefetch` body.

use axum::http::HeaderMap;
use serde::{Deserialize, Serialize};

use turbine_core::request::RequestKvPolicy;

use crate::error::ApiError;
use crate::openai::request::ChatMessageIn;

pub const SESSION_RESUME_WITHIN: &str = "x-turbine-session-resume-within";
pub const SESSION_END: &str = "x-turbine-session-end";
pub const CACHE_SALT: &str = "x-turbine-cache-salt";
pub const TARGET_REPLICA: &str = "x-turbine-target-replica";
/// `allow` or `deny` (P6b S-3): whether the request may reuse lossy cached KV.
pub const KV_LOSSY: &str = "x-turbine-kv-lossy";

/// Longest `prompt_cache_key` and cache salt, in characters.
const MAX_ID_CHARS: usize = 128;
/// Longest `x-turbine-session-resume-within`, in seconds (one day).
const MAX_RESUME_WITHIN_SECS: u32 = 86_400;

/// The `x-turbine-*` headers of one request (contract §14.1).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TurbineHeaders {
    /// `x-turbine-session-resume-within`: keep the session boost this many seconds.
    pub session_resume_within: Option<u32>,
    /// `x-turbine-session-end: true`: drop the session after this response.
    pub session_end: bool,
    /// `x-turbine-cache-salt`: only requests with the same salt share prefixes.
    pub cache_salt: Option<String>,
    /// `x-turbine-target-replica` (Phase 6 fault injection); carried, not interpreted here.
    pub target_replica: Option<String>,
    /// `x-turbine-kv-lossy: allow|deny` (P6b S-3); `None` takes `kv.lossy_reuse`.
    pub kv_policy: Option<RequestKvPolicy>,
}

/// 1–128 visible ASCII characters (`!`..=`~`).
fn visible_ascii(s: &str) -> bool {
    (1..=MAX_ID_CHARS).contains(&s.len()) && s.bytes().all(|b| b.is_ascii_graphic())
}

/// A `prompt_cache_key` (session id): 1–128 visible ASCII characters, else 400
/// `invalid_session_id`.
pub fn validate_session_id(id: &str) -> Result<(), ApiError> {
    if visible_ascii(id) {
        Ok(())
    } else {
        Err(ApiError::invalid_session_id())
    }
}

/// The value of header `name` as text; a non-UTF-8 value is `bad`.
fn header_text<'a>(
    h: &'a HeaderMap,
    name: &str,
    bad: fn() -> ApiError,
) -> Result<Option<&'a str>, ApiError> {
    h.get(name)
        .map(|v| v.to_str().map_err(|_| bad()))
        .transpose()
}

/// The validated cache salt of `x-turbine-cache-salt`, if present.
pub fn parse_cache_salt(h: &HeaderMap) -> Result<Option<String>, ApiError> {
    match header_text(h, CACHE_SALT, ApiError::invalid_cache_salt)? {
        Some(salt) if visible_ascii(salt) => Ok(Some(salt.to_string())),
        Some(_) => Err(ApiError::invalid_cache_salt()),
        None => Ok(None),
    }
}

/// `x-turbine-kv-lossy`: `allow` or `deny`, else 400 `invalid_request`.
pub fn parse_kv_lossy(h: &HeaderMap) -> Result<Option<RequestKvPolicy>, ApiError> {
    let bad = || ApiError::invalid_request("x-turbine-kv-lossy must be allow or deny");
    match header_text(h, KV_LOSSY, bad)? {
        None => Ok(None),
        Some("allow") => Ok(Some(RequestKvPolicy { allow_lossy: true })),
        Some("deny") => Ok(Some(RequestKvPolicy { allow_lossy: false })),
        Some(_) => Err(bad()),
    }
}

/// Parses and validates the `x-turbine-*` headers (P4 §Session hints): resume-within an integer
/// 1..=86400, session-end `true` or `false`, the salt 1–128 visible ASCII characters; either
/// session header without a `prompt_cache_key` is `invalid_session_hint`.
pub fn parse_turbine_headers(
    h: &HeaderMap,
    has_prompt_cache_key: bool,
) -> Result<TurbineHeaders, ApiError> {
    let resume = header_text(h, SESSION_RESUME_WITHIN, ApiError::invalid_session_hint)?;
    let session_resume_within = resume
        .map(|v| {
            v.parse::<u32>()
                .ok()
                .filter(|s| (1..=MAX_RESUME_WITHIN_SECS).contains(s))
                .ok_or_else(ApiError::invalid_session_hint)
        })
        .transpose()?;
    let session_end = match header_text(h, SESSION_END, ApiError::invalid_session_hint)? {
        None | Some("false") => false,
        Some("true") => true,
        Some(_) => return Err(ApiError::invalid_session_hint()),
    };
    let session_header = resume.is_some() || h.contains_key(SESSION_END);
    if session_header && !has_prompt_cache_key {
        return Err(ApiError::invalid_session_hint());
    }
    let target_replica = h
        .get(TARGET_REPLICA)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    Ok(TurbineHeaders {
        session_resume_within,
        session_end,
        cache_salt: parse_cache_salt(h)?,
        target_replica,
        kv_policy: parse_kv_lossy(h)?,
    })
}

/// What a prefetch names: a session (`prompt_cache_key`) or a prompt, tokenized like the
/// OpenAI routes.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum PrefetchTarget {
    Session { session_id: String },
    Prompt { prompt: String },
    Messages { messages: Vec<ChatMessageIn> },
}

/// Body of `POST /turbine/v1/kv/prefetch` plus the `x-turbine-cache-salt` it arrived with.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefetchRequest {
    pub target: PrefetchTarget,
    pub cache_salt: Option<String>,
}

impl PrefetchRequest {
    /// Parses the body (`{"session_id"}`, `{"prompt"}` or `{"messages"}`) and the salt header;
    /// anything else is 400 `invalid_request`, a bad session id `invalid_session_id`.
    pub fn parse(body: &[u8], headers: &HeaderMap) -> Result<PrefetchRequest, ApiError> {
        let target: PrefetchTarget = serde_json::from_slice(body).map_err(|_| {
            ApiError::invalid_request(
                "the prefetch body must be {\"session_id\": …}, {\"prompt\": …} or \
                 {\"messages\": […]}",
            )
        })?;
        if let PrefetchTarget::Session { session_id } = &target {
            validate_session_id(session_id)?;
        }
        Ok(PrefetchRequest {
            target,
            cache_salt: parse_cache_salt(headers)?,
        })
    }
}

/// `202` body of a prefetch: blocks queued for promotion and blocks already in L0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct PrefetchAccepted {
    pub blocks_queued: u32,
    pub blocks_resident: u32,
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderValue;

    use super::*;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(*k, HeaderValue::from_str(v).unwrap());
        }
        h
    }

    fn code(r: Result<TurbineHeaders, ApiError>) -> &'static str {
        r.expect_err("rejected").code.as_str()
    }

    #[test]
    fn header_rules() {
        assert_eq!(
            parse_turbine_headers(&HeaderMap::new(), false).unwrap(),
            TurbineHeaders::default()
        );
        let ok = parse_turbine_headers(
            &headers(&[
                (SESSION_RESUME_WITHIN, "86400"),
                (SESSION_END, "true"),
                (CACHE_SALT, "tenant-7"),
            ]),
            true,
        )
        .unwrap();
        assert_eq!(ok.session_resume_within, Some(86_400));
        assert!(ok.session_end);
        assert_eq!(ok.cache_salt.as_deref(), Some("tenant-7"));
        for bad in ["0", "86401", "-1", "1.5", "soon"] {
            let h = headers(&[(SESSION_RESUME_WITHIN, bad)]);
            assert_eq!(
                code(parse_turbine_headers(&h, true)),
                "invalid_session_hint"
            );
        }
        let h = headers(&[(SESSION_END, "yes")]);
        assert_eq!(
            code(parse_turbine_headers(&h, true)),
            "invalid_session_hint"
        );
        let h = headers(&[(SESSION_END, "false")]);
        assert_eq!(
            code(parse_turbine_headers(&h, false)),
            "invalid_session_hint"
        );
        for bad in ["", "has space", &"s".repeat(129)] {
            let h = headers(&[(CACHE_SALT, bad)]);
            assert_eq!(code(parse_turbine_headers(&h, false)), "invalid_cache_salt");
        }
        assert!(validate_session_id(&"k".repeat(128)).is_ok());
        for bad in ["", "tab\tkey", "é", &"k".repeat(129)] {
            assert_eq!(
                validate_session_id(bad).unwrap_err().code.as_str(),
                "invalid_session_id"
            );
        }
    }

    /// P6b S-3: `x-turbine-kv-lossy` is `allow` or `deny`, absent is `None`, anything else a
    /// 400. Breaks if a bad value is silently taken as either.
    #[test]
    fn kv_lossy_header() {
        let policy = |v: &str| parse_turbine_headers(&headers(&[(KV_LOSSY, v)]), false);
        assert_eq!(
            policy("deny").unwrap().kv_policy,
            Some(RequestKvPolicy { allow_lossy: false })
        );
        assert_eq!(
            policy("allow").unwrap().kv_policy,
            Some(RequestKvPolicy { allow_lossy: true })
        );
        for bad in ["", "Deny", "no", "true"] {
            assert_eq!(code(policy(bad)), "invalid_request", "{bad:?}");
        }
    }

    #[test]
    fn prefetch_bodies() {
        let h = headers(&[(CACHE_SALT, "a")]);
        let r = PrefetchRequest::parse(br#"{"session_id":"s1"}"#, &h).unwrap();
        assert_eq!(
            r.target,
            PrefetchTarget::Session {
                session_id: "s1".into()
            }
        );
        assert_eq!(r.cache_salt.as_deref(), Some("a"));
        let r = PrefetchRequest::parse(br#"{"prompt":"hello"}"#, &HeaderMap::new()).unwrap();
        assert_eq!(
            r.target,
            PrefetchTarget::Prompt {
                prompt: "hello".into()
            }
        );
        let r = PrefetchRequest::parse(
            br#"{"messages":[{"role":"user","content":"hi"}]}"#,
            &HeaderMap::new(),
        )
        .unwrap();
        assert!(matches!(r.target, PrefetchTarget::Messages { messages } if messages.len() == 1));
        let err = PrefetchRequest::parse(br#"{"other":1}"#, &HeaderMap::new()).unwrap_err();
        assert_eq!(err.code.as_str(), "invalid_request");
        let err = PrefetchRequest::parse(br#"{"session_id":""}"#, &HeaderMap::new()).unwrap_err();
        assert_eq!(err.code.as_str(), "invalid_session_id");
    }
}
