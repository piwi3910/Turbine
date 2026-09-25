//! `turbine-bench --url <base-url> [...]` — exit 0 when ≥ 1 request succeeded, 1 when all
//! failed, 2 on usage errors.

use std::process::ExitCode;

use clap::Parser;
use turbine_bench::{BenchArgs, OutputFormat, run};

fn main() -> ExitCode {
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
