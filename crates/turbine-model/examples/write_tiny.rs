//! Writes a registered family's tiny synthetic checkpoint (the one its conformance suite runs on
//! the CPU provider) into a directory: fixture tooling, e.g. for `scripts/golden/tiny_trace.py`.
//!
//! usage: cargo run -p turbine-model --example write_tiny -- <family> <seed> <dir> [window]
//!
//! `<family>` is a registry name (`llama`, `olmoe`, `gpt_oss`, …); `[window]` (gpt-oss only) is
//! the sliding window of its sliding-attention layers (default 128, the family's tiny
//! checkpoint).

use std::path::Path;
use std::process::ExitCode;

use turbine_model::families;
use turbine_model::testing::tiny::write_tiny_gpt_oss;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [family, seed, dir, rest @ ..] = args.as_slice() else {
        eprintln!("usage: write_tiny <family> <seed> <dir> [window]");
        return ExitCode::from(2);
    };
    let Ok(seed) = seed.parse::<u64>() else {
        eprintln!("write_tiny: seed must be an unsigned integer, got {seed}");
        return ExitCode::from(2);
    };
    let Some(found) = families::registry().get(family) else {
        let names: Vec<&str> = families::registry().iter().map(|f| f.name()).collect();
        eprintln!(
            "write_tiny: unknown family {family} (registered: {})",
            names.join(", ")
        );
        return ExitCode::from(2);
    };
    let dir = Path::new(dir);
    let spec = match (family.as_str(), rest) {
        ("gpt_oss", [window]) => match window.parse::<u32>() {
            Ok(w) if w > 0 => write_tiny_gpt_oss(dir, seed, w),
            _ => {
                eprintln!("write_tiny: window must be a positive integer, got {window}");
                return ExitCode::from(2);
            }
        },
        (_, []) => found.write_tiny(dir, seed),
        _ => {
            eprintln!("write_tiny: only gpt_oss takes a window");
            return ExitCode::from(2);
        }
    };
    println!("write_tiny: {family} seed {seed} -> {}", spec.dir.display());
    ExitCode::SUCCESS
}
