//! Parsers for the host `/proc` files read on the fast tick (P3 S-5). They are pure functions
//! over file contents, so tests run on macOS; where `/proc` is absent the reader reports the
//! host source `unavailable`.

use std::path::PathBuf;

/// The host files the fast tick reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcFile {
    Meminfo,
    Vmstat,
    PressureMemory,
}

impl ProcFile {
    /// Path below the `/proc` root.
    pub fn relative_path(self) -> &'static str {
        match self {
            ProcFile::Meminfo => "meminfo",
            ProcFile::Vmstat => "vmstat",
            ProcFile::PressureMemory => "pressure/memory",
        }
    }
}

/// Source of `/proc` contents: the real filesystem, or fixtures and fakes in tests.
pub trait ProcSource: Send {
    fn read(&self, file: ProcFile) -> std::io::Result<String>;
}

/// Reads `<root>/meminfo`, `<root>/vmstat` and `<root>/pressure/memory`.
#[derive(Clone, Debug)]
pub struct FsProc {
    pub root: PathBuf,
}

impl Default for FsProc {
    /// The host `/proc`.
    fn default() -> Self {
        FsProc {
            root: PathBuf::from("/proc"),
        }
    }
}

impl ProcSource for FsProc {
    fn read(&self, file: ProcFile) -> std::io::Result<String> {
        std::fs::read_to_string(self.root.join(file.relative_path()))
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ParseError {
    #[error("{file}: missing field {field}")]
    Missing {
        file: &'static str,
        field: &'static str,
    },
    #[error("{file}: malformed line {line:?}")]
    Malformed { file: &'static str, line: String },
}

/// `/proc/meminfo` fields the sampler uses, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemInfo {
    /// `MemTotal` (P4: `kv.cpu.max_bytes` is checked against it at startup).
    pub mem_total_bytes: Option<u64>,
    pub mem_available_bytes: u64,
    /// Absent on kernels built without swap support.
    pub swap_total_bytes: Option<u64>,
    pub swap_free_bytes: Option<u64>,
}

/// Cumulative page counters from `/proc/vmstat`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VmStat {
    pub pswpin: u64,
    pub pswpout: u64,
}

/// `/proc/pressure/memory` 10-second averages (percent of wall time).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PsiMemory {
    pub some_avg10: f64,
    /// The `full` line is absent on some older kernels.
    pub full_avg10: Option<f64>,
}

/// `/proc/meminfo`: values carry a `kB` unit that means KiB (`MemAvailable:   44950556 kB`).
pub fn parse_meminfo(s: &str) -> Result<MemInfo, ParseError> {
    let field = |name: &str| -> Result<Option<u64>, ParseError> {
        let Some(line) = s.lines().find(|l| l.split(':').next() == Some(name)) else {
            return Ok(None);
        };
        let malformed = || ParseError::Malformed {
            file: "meminfo",
            line: line.to_string(),
        };
        let mut parts = line.split_whitespace().skip(1);
        let value: u64 = parts
            .next()
            .ok_or_else(malformed)?
            .parse()
            .map_err(|_| malformed())?;
        match parts.next() {
            Some("kB") => value.checked_mul(1024).map(Some).ok_or_else(malformed),
            None => Ok(Some(value)),
            Some(_) => Err(malformed()),
        }
    };
    Ok(MemInfo {
        mem_total_bytes: field("MemTotal")?,
        mem_available_bytes: field("MemAvailable")?.ok_or(ParseError::Missing {
            file: "meminfo",
            field: "MemAvailable",
        })?,
        swap_total_bytes: field("SwapTotal")?,
        swap_free_bytes: field("SwapFree")?,
    })
}

/// `/proc/vmstat`: `pswpin 1200` (pages, cumulative since boot).
pub fn parse_vmstat(s: &str) -> Result<VmStat, ParseError> {
    let field = |name: &'static str| -> Result<u64, ParseError> {
        let line = s
            .lines()
            .find(|l| l.split_whitespace().next() == Some(name))
            .ok_or(ParseError::Missing {
                file: "vmstat",
                field: name,
            })?;
        line.split_whitespace()
            .nth(1)
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| ParseError::Malformed {
                file: "vmstat",
                line: line.to_string(),
            })
    };
    Ok(VmStat {
        pswpin: field("pswpin")?,
        pswpout: field("pswpout")?,
    })
}

/// `/proc/pressure/memory`: `some avg10=12.50 avg60=3.10 avg300=0.84 total=38490211`.
pub fn parse_psi(s: &str) -> Result<PsiMemory, ParseError> {
    let avg10 = |kind: &str| -> Result<Option<f64>, ParseError> {
        let Some(line) = s
            .lines()
            .find(|l| l.split_whitespace().next() == Some(kind))
        else {
            return Ok(None);
        };
        line.split_whitespace()
            .find_map(|kv| kv.strip_prefix("avg10="))
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite())
            .map(Some)
            .ok_or_else(|| ParseError::Malformed {
                file: "pressure/memory",
                line: line.to_string(),
            })
    };
    Ok(PsiMemory {
        some_avg10: avg10("some")?.ok_or(ParseError::Missing {
            file: "pressure/memory",
            field: "some",
        })?,
        full_avg10: avg10("full")?,
    })
}

/// The "Available" column (1024-byte blocks) of `df -Pk` output, in bytes (P4: `kv.nvme.max_bytes`
/// against free disk at startup).
pub fn parse_df_available(output: &str) -> Option<u64> {
    let line = output.lines().nth(1)?;
    let kib: u64 = line.split_whitespace().nth(3)?.parse().ok()?;
    kib.checked_mul(1024)
}

/// Free bytes of the filesystem holding `path` (its nearest existing ancestor when `path` does
/// not exist yet), from `df -Pk` (POSIX output: Linux and macOS alike, no new dependency).
pub fn disk_free_bytes(path: &std::path::Path) -> std::io::Result<u64> {
    let mut probe: PathBuf = path.to_path_buf();
    while !probe.exists() {
        if !probe.pop() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no existing ancestor of {}", path.display()),
            ));
        }
    }
    let out = std::process::Command::new("df")
        .arg("-Pk")
        .arg(&probe)
        .output()?;
    if !out.status.success() {
        return Err(std::io::Error::other(format!(
            "df -Pk {}: {}",
            probe.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    parse_df_available(&String::from_utf8_lossy(&out.stdout)).ok_or_else(|| {
        std::io::Error::other(format!("df -Pk {}: unexpected output", probe.display()))
    })
}
