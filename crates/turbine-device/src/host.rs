//! Host memory figures (`/proc/meminfo`), read by the cpu backend and the unified-memory budget.

use std::path::Path;

/// Host `MemAvailable` in bytes from a `/proc/meminfo`-format file (`MemAvailable: <n> kB`);
/// `None` when the file or the line is missing or malformed.
pub fn host_mem_available(meminfo_path: &Path) -> Option<u64> {
    let text = std::fs::read_to_string(meminfo_path).ok()?;
    let line = text.lines().find_map(|l| l.strip_prefix("MemAvailable:"))?;
    let mut parts = line.split_whitespace();
    let kib: u64 = parts.next()?.parse().ok()?;
    match parts.next() {
        Some("kB") => kib.checked_mul(1024),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn mem_available_from_meminfo() {
        let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/proc");
        // 95 043 880 kB × 1024.
        assert_eq!(
            host_mem_available(&fixtures.join("meminfo-gb10")),
            Some(97_324_933_120)
        );
        assert_eq!(host_mem_available(&fixtures.join("absent")), None);
    }
}
