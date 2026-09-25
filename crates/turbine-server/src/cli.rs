//! Command line (contract §16.1).

use std::path::PathBuf;

use clap::Parser;
use turbine_core::config::Override;

#[derive(Debug, Parser)]
#[command(
    name = "turbine-server",
    version,
    about = "Turbine LLM inference server"
)]
pub struct Cli {
    /// YAML configuration file.
    #[arg(long, value_name = "PATH")]
    pub config: PathBuf,
    /// Override one key after the file is read, e.g. `--set kv.cpu.enabled=false`. Repeatable.
    #[arg(long = "set", value_name = "KEY=VALUE")]
    pub set: Vec<Override>,
    /// Validate the configuration, print `config ok` and exit (no discovery, no bind).
    #[arg(long)]
    pub check_config: bool,
}
