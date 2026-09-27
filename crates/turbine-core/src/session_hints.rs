//! Session hints of one request (P4 §Session hints; contract §3.5). Re-exported from
//! `turbine_core::request`. The API layer validates them; the KV session table consumes them.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionHints {
    /// The OpenAI `prompt_cache_key` body field: 1–128 visible ASCII characters.
    pub session_id: String,
    /// `x-turbine-session-resume-within`: keep the session boost this many seconds (1..86400).
    pub resume_within_secs: Option<u32>,
    /// `x-turbine-session-end: true`: drop the session after this response.
    pub end: bool,
}
