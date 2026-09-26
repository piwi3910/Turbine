//! `--support-matrix` (Phase 2m S-11, from the Phase 8 run-ahead): every row of the support
//! matrix as text or JSON, without reading a configuration.
use std::fmt::Write as _;

use turbine_core::support::{SUPPORT_MATRIX, SupportDecision};

use crate::cli::OutputFormat;

/// Body printed by `turbine-server --support-matrix`.
pub fn render_matrix(output: OutputFormat) -> String {
    let rows: Vec<_> = SUPPORT_MATRIX.iter().map(|r| r.view()).collect();
    match output {
        OutputFormat::Json => {
            let mut s = serde_json::to_string(&serde_json::json!({ "rows": rows }))
                .expect("rows serialize");
            s.push('\n');
            s
        }
        OutputFormat::Text => {
            let mut s = String::new();
            let _ = writeln!(
                s,
                "{:<7} {:<8} {:<18} {:<14} {:<9} {:<11} {:<12} reason",
                "vendor",
                "arch",
                "architecture",
                "weight_format",
                "kv_format",
                "speculative",
                "status"
            );
            for r in rows {
                let _ = writeln!(
                    s,
                    "{:<7} {:<8} {:<18} {:<14} {:<9} {:<11} {:<12} {}",
                    r.vendor,
                    r.arch,
                    r.architecture,
                    r.weight_format,
                    r.kv_format,
                    r.speculative,
                    r.status,
                    r.reason.as_deref().unwrap_or("-")
                );
            }
            s
        }
    }
}

/// The `--check-config` line for a resolved decision: `support: <status> (<row>)`.
pub fn check_config_line(decision: &SupportDecision) -> String {
    format!("support: {} ({})", decision.status.as_str(), decision.key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_text_lists_every_row() {
        let text = render_matrix(OutputFormat::Text);
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].starts_with("vendor "), "{text}");
        assert_eq!(lines.len(), SUPPORT_MATRIX.len() + 1);
        assert!(
            lines
                .iter()
                .any(|l| l.split_whitespace().collect::<Vec<_>>()[..7]
                    == [
                        "amd",
                        "gfx1201",
                        "LlamaForCausalLM",
                        "bf16",
                        "bf16",
                        "none",
                        "supported"
                    ]),
            "{text}"
        );
    }
}
