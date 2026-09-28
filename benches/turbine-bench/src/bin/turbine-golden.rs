//! `turbine-golden compare|capture|positions` (P1 S-11; `positions`: P5): exit 0 when the tolerance holds (or the capture
//! was written), 1 when it is violated or the endpoint fails, 2 on usage or I/O errors.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use turbine_bench::golden::fixture::{
    read_jsonl, read_prompts, read_tolerance, write_jsonl_atomic,
};
use turbine_bench::golden::{Endpoint, GoldenError, ReferenceRecord, capture, compare, positions};

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
        /// Reference prompts in flight at once; results are still reported in prompt order.
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
        concurrency: u32,
        /// Judge with the tolerance's batched logprob bounds at every concurrency (the
        /// tensor-parallel gate against a one-GPU capture, P5).
        #[arg(long)]
        batched_bounds: bool,
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
    /// One reference prompt judged position by position: every reference top-k candidate's
    /// |Δ logprob|, tier and strict bound. With --url the endpoint is teacher-forced on the
    /// reference's tokens (every position has the reference's history); with --candidate a
    /// captured record is judged up to its first divergence. Always exits 0 on success.
    Positions {
        #[arg(long)]
        reference: PathBuf,
        /// The prompt id, e.g. p14.
        #[arg(long)]
        prompt_id: String,
        /// Teacher-force this endpoint (completions route, token-id prompts).
        #[arg(
            long,
            conflicts_with = "candidate",
            required_unless_present = "candidate"
        )]
        url: Option<String>,
        /// A captured reference.jsonl to judge instead of an endpoint.
        #[arg(long)]
        candidate: Option<PathBuf>,
        #[arg(long)]
        model: Option<String>,
        /// Default: tolerance.json beside the reference.
        #[arg(long)]
        tolerance: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        output: OutputFormat,
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
            concurrency,
            batched_bounds,
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
            let tol = if batched_bounds {
                tol.batched_everywhere()
            } else {
                tol
            };
            let model = endpoint.model(model.as_deref()).await?;
            let mut report = compare(
                &endpoint,
                &model,
                &references,
                &prompts,
                &tol,
                concurrency as usize,
            )
            .await?;
            if batched_bounds {
                report.logprob_bounds.batched = true;
            }
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
        Command::Positions {
            reference,
            prompt_id,
            url,
            candidate,
            model,
            tolerance,
            output,
        } => {
            let references: Vec<ReferenceRecord> = read_jsonl(&reference)?;
            let find = |records: &[ReferenceRecord], what: &Path| {
                records
                    .iter()
                    .find(|r| r.id == prompt_id)
                    .cloned()
                    .ok_or_else(|| {
                        GoldenError::Usage(format!("no prompt {prompt_id} in {}", what.display()))
                    })
            };
            let record = find(&references, &reference)?;
            let tolerance =
                tolerance.unwrap_or_else(|| reference_dir(&reference).join("tolerance.json"));
            let tol = read_tolerance(&tolerance)?;
            let (rows, source) = match (url, candidate) {
                (Some(url), _) => {
                    let endpoint = Endpoint::new(&url)?;
                    let model = endpoint.model(model.as_deref()).await?;
                    let rows =
                        positions::positions_teacher_forced(&endpoint, &model, &record, &tol)
                            .await?;
                    (rows, format!("teacher-forced {url}"))
                }
                (None, Some(path)) => {
                    let captured: Vec<ReferenceRecord> = read_jsonl(&path)?;
                    let cand = find(&captured, &path)?;
                    (
                        positions::positions_of_capture(&record, &cand, &tol),
                        format!("capture {}", path.display()),
                    )
                }
                (None, None) => {
                    return Err(GoldenError::Usage("pass --url or --candidate".into()));
                }
            };
            match output {
                OutputFormat::Json => println!(
                    "{}",
                    serde_json::to_string_pretty(&rows).expect("rows serialize")
                ),
                OutputFormat::Text => print!("{}", positions::to_text(&prompt_id, &source, &rows)),
            }
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
