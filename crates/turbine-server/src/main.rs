//! `turbine-server --config <path> [--set <dotted.key>=<yaml value>]... [--check-config]`
//! `turbine-server --support-matrix [--output text|json]`

mod backend;
mod cli;
mod engine;
mod exit;
mod metrics;
mod model;
mod modules;
mod startup;
mod support_matrix;
mod support_startup;

use clap::Parser;

fn main() -> std::process::ExitCode {
    // clap prints usage errors and exits 2 itself.
    let cli = cli::Cli::parse();
    if cli.support_matrix {
        print!("{}", support_matrix::render_matrix(cli.output));
        return exit::ExitCode::Clean.into();
    }
    startup::run(cli).into()
}
