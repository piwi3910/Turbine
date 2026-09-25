//! `turbine-golden compare|capture` (P1 S-11): exit 0 when the tolerance holds (or the capture
//! was written), 1 when it is violated or the endpoint fails, 2 on usage or I/O errors.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use turbine_bench::golden::fixture::{
    read_jsonl, read_prompts, read_tolerance, write_jsonl_atomic,
};
use turbine_bench::golden::{Endpoint, GoldenError, ReferenceRecord, capture, compare};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Debug, Parser)]
#[command(
    name = "turbine-golden",
    version,
    about = "Compare an OpenAI-compatible endpoint with golden fixtures, or capture fixtures from one"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Replay every reference prompt greedily and judge it against tolerance.json.
    Compare {
        /// Base URL, e.g. http://127.0.0.1:8000 (http only).
        #[arg(long)]
        url: String,
        /// reference.jsonl to compare against.
        #[arg(long)]
        reference: PathBuf,
        /// Prompts by id; default: prompts.jsonl beside the reference, else one directory up.
        #[arg(long)]
        prompts: Option<PathBuf>,
        /// Model id; default: the first id from GET <url>/v1/models.
        #[arg(long)]
        model: Option<String>,
        /// Default: tolerance.json beside the reference.
        #[arg(long)]
        tolerance: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        output: OutputFormat,
    },
    /// Record reference.jsonl from any OpenAI-compatible endpoint.
    Capture {
        #[arg(long)]
        url: String,
        #[arg(long)]
        prompts: PathBuf,
        /// Written as <out>.tmp, then renamed; nothing is left behind on failure.
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        model: Option<String>,
        #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u32).range(2..=20))]
        top_logprobs: u32,
    },
}

/// The reference's directory (`.` for a bare file name).
fn reference_dir(reference: &Path) -> &Path {
    reference
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

/// `prompts.jsonl` beside the reference, else in its parent directory (the committed layout
/// `tests/golden/{prompts.jsonl,<slug>/reference.jsonl}`).
fn default_prompts(reference: &Path) -> Result<PathBuf, GoldenError> {
    let dir = reference_dir(reference);
    let beside = dir.join("prompts.jsonl");
    let above = dir.join("..").join("prompts.jsonl");
    [&beside, &above]
        .into_iter()
        .find(|p| p.is_file())
        .cloned()
        .ok_or_else(|| {
            GoldenError::Usage(format!(
                "no prompts file at {} or {}; pass --prompts",
                beside.display(),
                above.display()
            ))
        })
}

async fn run(cli: Cli) -> Result<ExitCode, GoldenError> {
    match cli.command {
        Command::Compare {
            url,
            reference,
            prompts,
            model,
            tolerance,
            output,
        } => {
            let endpoint = Endpoint::new(&url)?;
            let references: Vec<ReferenceRecord> = read_jsonl(&reference)?;
            let prompts = match prompts {
                Some(p) => p,
                None => default_prompts(&reference)?,
            };
            let prompts = read_prompts(&prompts)?;
            let tolerance =
                tolerance.unwrap_or_else(|| reference_dir(&reference).join("tolerance.json"));
            let tol = read_tolerance(&tolerance)?;
            let model = endpoint.model(model.as_deref()).await?;
            let report = compare(&endpoint, &model, &references, &prompts, &tol).await?;
            match output {
                OutputFormat::Json => println!(
                    "{}",
                    serde_json::to_string_pretty(&report).expect("report serializes")
                ),
                OutputFormat::Text => print!("{}", report.to_text()),
            }
            Ok(if report.passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            })
        }
        Command::Capture {
            url,
            prompts,
            out,
            model,
            top_logprobs,
        } => {
            let endpoint = Endpoint::new(&url)?;
            let prompts = read_prompts(&prompts)?;
            let model = endpoint.model(model.as_deref()).await?;
            let records = capture(&endpoint, &model, &prompts, top_logprobs).await?;
            write_jsonl_atomic(&out, &records)?;
            eprintln!(
                "turbine-golden: captured {} prompts into {}",
                records.len(),
                out.display()
            );
            Ok(ExitCode::SUCCESS)
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("turbine-golden: cannot start the async runtime: {e}");
            return ExitCode::from(1);
        }
    };
    match runtime.block_on(run(cli)) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("turbine-golden: {e}");
            ExitCode::from(e.exit_code())
        }
    }
}
