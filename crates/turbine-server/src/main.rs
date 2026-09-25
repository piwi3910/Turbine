//! `turbine-server --config <path> [--set <dotted.key>=<yaml value>]... [--check-config]`

mod cli;
mod exit;
mod startup;

use clap::Parser;

fn main() -> std::process::ExitCode {
    // clap prints usage errors and exits 2 itself.
    let cli = cli::Cli::parse();
    startup::run(cli).into()
}
