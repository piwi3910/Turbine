//! `turbine-bench --url <base-url> [...]` — exit 0 when ≥ 1 request succeeded, 1 when all
//! failed, 2 on usage errors. `turbine-bench kv-sim [...]` runs the offline KV policy simulator
//! (exit 0, or 2 on usage errors).

use std::process::ExitCode;

use clap::Parser;
use turbine_bench::{BenchArgs, OutputFormat, run};

fn main() -> ExitCode {
    if std::env::args().nth(1).as_deref() == Some("kv-sim") {
        return kv_sim();
    }
    let args = BenchArgs::parse();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("turbine-bench: cannot start the async runtime: {e}");
            return ExitCode::from(1);
        }
    };
    let report = match runtime.block_on(run(&args)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("turbine-bench: {e}");
            return ExitCode::from(e.exit_code());
        }
    };
    match args.output {
        OutputFormat::Json => println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("report serializes")
        ),
        OutputFormat::Text => print!("{}", report.to_text()),
    }
    if report.requests_ok > 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

/// `turbine-bench kv-sim …`: the subcommand name stands in for the binary name.
fn kv_sim() -> ExitCode {
    use turbine_bench::kv_sim::{KvSimArgs, SimOutput, run};
    let args = KvSimArgs::parse_from(std::env::args().skip(1));
    let report = run(&args);
    match args.output {
        SimOutput::Json => println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("report serializes")
        ),
        SimOutput::Text => print!("{}", report.to_text()),
    }
    ExitCode::SUCCESS
}
