//! Command line (contract §19).

use clap::{Parser, ValueEnum};

/// Which OpenAI route to load: `/v1/chat/completions` or `/v1/completions`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum EndpointArg {
    Chat,
    Completions,
}

/// Report rendering on stdout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
}

/// `turbine-bench` flags (contract §19).
#[derive(Clone, Debug, Parser)]
#[command(
    name = "turbine-bench",
    version,
    about = "Streaming load generator for OpenAI-compatible endpoints"
)]
pub struct BenchArgs {
    /// Base URL, e.g. http://127.0.0.1:8000 (http only).
    #[arg(long)]
    pub url: String,
    /// Model id; default: the first id from GET <url>/v1/models.
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long, value_enum, default_value_t = EndpointArg::Chat)]
    pub endpoint: EndpointArg,
    /// Requests in flight at once.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
    pub concurrency: u32,
    /// Total requests.
    #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u32).range(1..))]
    pub requests: u32,
    #[arg(long, default_value_t = 256)]
    pub prompt_words: u32,
    #[arg(long, default_value_t = 128)]
    pub max_tokens: u32,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Adds "ignore_eos": true (vLLM / SGLang fixed-length outputs).
    #[arg(long)]
    pub ignore_eos: bool,
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub output: OutputFormat,
}
