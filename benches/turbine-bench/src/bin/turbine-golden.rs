//! `turbine-golden compare|capture|positions` (P1 S-11; `positions`: P5): exit 0 when the tolerance holds (or the capture
//! was written), 1 when it is violated or the endpoint fails, 2 on usage or I/O errors.
//! `turbine-golden eval|eval-compare` (P8 S-4): task-set accuracy and the lossy-format gate.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use turbine_bench::golden::fixture::{
    read_jsonl, read_prompts, read_tolerance, write_jsonl_atomic,
};
use turbine_bench::golden::{
    Endpoint, GoldenError, ReferenceRecord, capture, compare, eval, positions,
};
use turbine_core::config::QualityConfig;

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
    /// Task-set accuracy against an OpenAI-compatible endpoint (greedy, sequential) (P8 S-4).
    Eval {
        /// Base URL, e.g. http://127.0.0.1:8000.
        #[arg(long)]
        url: String,
        /// Tasks file (JSONL), e.g. tests/eval/gsm8k-200.jsonl.
        #[arg(long)]
        tasks: PathBuf,
        /// Model id; default: the first id from GET <url>/v1/models.
        #[arg(long)]
        model: Option<String>,
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        output: OutputFormat,
        /// Requests in flight at once; results are still reported in task-file order. Both
        /// sides of a comparison must use the same value.
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=256))]
        concurrency: u32,
        /// After the first two items (the second hits the prefix the first published), send
        /// this many unrelated filler requests (one token each), one after another, before the
        /// other items: they fill L0 so the first items' shared
        /// prefix is demoted to a lossy lower KV tier and the other items reuse it from there
        /// (lossy-KV gates on `tests/eval/gsm8k-200-shared-prefix.jsonl`). Both sides of a
        /// comparison must use the same value.
        #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..=1000))]
        filler_requests: u32,
        /// Words in each filler prompt (about the shared prefix's size fills L0 fastest).
        #[arg(long, default_value_t = 2000, value_parser = clap::value_parser!(u32).range(1..=20000))]
        filler_words: u32,
        /// Exit 1 (the report is still printed) when fewer than this share of the items'
        /// prompt tokens were served from cached KV (`usage.prompt_tokens_details.cached_tokens`).
        #[arg(long)]
        min_cached_ratio: Option<f64>,
        /// Exit 1 (the report is still printed) when fewer than this share of the items'
        /// prompt tokens were served from lossy KV blocks
        /// (`usage.prompt_tokens_details.lossy_cached_tokens`): a lossy-KV gate that reused no
        /// lossy block measured nothing.
        #[arg(long)]
        min_lossy_cached_ratio: Option<f64>,
    },
    /// Quality gate: exit 0 when candidate accuracy ≥ baseline accuracy − max drop, else 1.
    EvalCompare {
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long)]
        candidate: PathBuf,
        /// Default: the `quality.max_accuracy_drop` default (0.01); 0..=0.1.
        #[arg(long, default_value_t = QualityConfig::default().max_accuracy_drop)]
        max_drop: f64,
    },
}

/// `eval`: exit 0 with the report on a completed run; 2 (no report) on any I/O, task-file or
/// endpoint failure, naming the failing task.
async fn run_eval(
    url: &str,
    tasks_path: &Path,
    model: Option<&str>,
    output: OutputFormat,
    concurrency: u32,
    fillers: eval::Fillers,
    guards: (Option<f64>, Option<f64>),
) -> ExitCode {
    if guards
        .0
        .into_iter()
        .chain(guards.1)
        .any(|r| !(0.0..=1.0).contains(&r))
    {
        eprintln!("turbine-golden eval: reuse ratios must be between 0 and 1");
        return ExitCode::from(2);
    }
    let report = match eval::load_tasks(tasks_path) {
        Ok(tasks) => eval::run_eval(url, model, tasks_path, &tasks, concurrency, fillers).await,
        Err(e) => Err(e),
    };
    match report {
        Ok(report) => {
            match output {
                OutputFormat::Json => println!(
                    "{}",
                    serde_json::to_string_pretty(&report).expect("report serializes")
                ),
                OutputFormat::Text => {
                    println!(
                        "model {} tasks {}: accuracy {:.4} ({}/{}) at concurrency {}",
                        report.model,
                        report.tasks_file,
                        report.accuracy,
                        report.correct,
                        report.total,
                        report.concurrency
                    );
                    if let Some(t) = report.token_totals() {
                        println!(
                            "prompt tokens {}: cached {} ({:.3}), lossy cached {} ({:.3})",
                            t.prompt,
                            t.cached,
                            t.cached_ratio(),
                            t.lossy_cached,
                            t.lossy_cached_ratio()
                        );
                    }
                }
            }
            if let Err(why) = eval::check_reuse(&report, guards.0, guards.1) {
                eprintln!("turbine-golden eval: {why}");
                return ExitCode::from(1);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("turbine-golden eval: {e}");
            ExitCode::from(2)
        }
    }
}

/// `eval-compare`: exit 0 on pass, 1 on fail, 2 on usage or I/O errors.
fn run_eval_compare(baseline: &Path, candidate: &Path, max_drop: f64) -> ExitCode {
    if !(0.0..=0.1).contains(&max_drop) {
        eprintln!("turbine-golden eval-compare: --max-drop must be between 0 and 0.1");
        return ExitCode::from(2);
    }
    let (baseline, candidate) = match (eval::read_report(baseline), eval::read_report(candidate)) {
        (Ok(b), Ok(c)) => (b, c),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("turbine-golden eval-compare: {e}");
            return ExitCode::from(2);
        }
    };
    if baseline.concurrency != candidate.concurrency {
        eprintln!(
            "turbine-golden eval-compare: baseline was measured at concurrency {} but candidate at concurrency {}; a comparison must use one concurrency on both sides",
            baseline.concurrency, candidate.concurrency
        );
        return ExitCode::from(2);
    }
    if (baseline.filler_requests, baseline.filler_words)
        != (candidate.filler_requests, candidate.filler_words)
    {
        eprintln!(
            "turbine-golden eval-compare: baseline was measured with {} filler requests of {} words but candidate with {} of {}; a comparison must use the same fillers on both sides",
            baseline.filler_requests,
            baseline.filler_words,
            candidate.filler_requests,
            candidate.filler_words
        );
        return ExitCode::from(2);
    }
    let o = eval::compare(&baseline, &candidate, max_drop);
    println!(
        "baseline accuracy {:.4}, candidate accuracy {:.4}, max drop {:.4} (concurrency {}): {}",
        o.baseline_accuracy,
        o.candidate_accuracy,
        o.max_drop,
        baseline.concurrency,
        if o.pass { "PASS" } else { "FAIL" }
    );
    ExitCode::from(if o.pass { 0 } else { 1 })
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
        Command::Eval {
            url,
            tasks,
            model,
            output,
            concurrency,
            filler_requests,
            filler_words,
            min_cached_ratio,
            min_lossy_cached_ratio,
        } => Ok(run_eval(
            &url,
            &tasks,
            model.as_deref(),
            output,
            concurrency,
            eval::Fillers {
                requests: filler_requests,
                words: filler_words,
            },
            (min_cached_ratio, min_lossy_cached_ratio),
        )
        .await),
        Command::EvalCompare {
            baseline,
            candidate,
            max_drop,
        } => Ok(run_eval_compare(&baseline, &candidate, max_drop)),
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
