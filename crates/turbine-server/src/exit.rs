//! Process exit codes (contract §16.2).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitCode {
    /// Clean shutdown after SIGINT/SIGTERM, or `--check-config` success.
    Clean = 0,
    /// Startup failure after config validation (bind, explicit GPU library).
    Startup = 1,
    /// Invalid configuration or CLI usage.
    Config = 2,
}

impl From<ExitCode> for std::process::ExitCode {
    fn from(code: ExitCode) -> Self {
        std::process::ExitCode::from(code as u8)
    }
}
