//! Command line (contract §16.1).

use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use turbine_core::config::Override;

#[derive(Debug, Parser)]
#[command(
    name = "turbine-server",
    version,
    about = "Turbine LLM inference server"
)]
pub struct Cli {
    /// YAML configuration file.
    #[arg(long, value_name = "PATH", required_unless_present = "support_matrix")]
    pub config: Option<PathBuf>,
    /// Override one key after the file is read, e.g. `--set kv.cpu.enabled=false`. Repeatable.
    #[arg(long = "set", value_name = "KEY=VALUE")]
    pub set: Vec<Override>,
    /// Validate the configuration and resolve its support-matrix row, print it and `config ok`
    /// and exit (no discovery, no bind).
    #[arg(long, conflicts_with = "support_matrix")]
    pub check_config: bool,
    /// Print the support matrix and exit 0 without reading a configuration.
    #[arg(long, conflicts_with_all = ["config", "check_config"])]
    pub support_matrix: bool,
    /// Output format of `--support-matrix`.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text, requires = "support_matrix")]
    pub output: OutputFormat,
}

/// `--output` of `--support-matrix`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
}
