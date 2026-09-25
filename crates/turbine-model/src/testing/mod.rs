//! Test utilities shared by the unit and integration tests of `turbine-model` and of the crates
//! above it (`turbine-server`, `turbine-bench`): temporary directories and, in [`tiny`], the tiny
//! synthetic checkpoint.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub mod tiny;

static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

/// A uniquely named directory under `std::env::temp_dir()`, removed with its contents on `Drop`.
#[derive(Debug)]
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    /// Creates `<temp>/<prefix>-<pid>-<nanos>-<counter>`. Panics when the directory cannot be
    /// created (a test-only utility).
    pub fn new(prefix: &str) -> TempDir {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let n = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("{prefix}-{}-{nanos}-{n}", std::process::id()));
        std::fs::create_dir_all(&path)
            .unwrap_or_else(|e| panic!("create temp dir {}: {e}", path.display()));
        TempDir { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_dir_is_unique_and_removed() {
        let a = TempDir::new("turbine-tmp");
        let b = TempDir::new("turbine-tmp");
        assert_ne!(a.path(), b.path());
        std::fs::write(a.path().join("x"), b"x").expect("write");
        let kept = a.path().to_path_buf();
        drop(a);
        assert!(!kept.exists());
        assert!(b.path().is_dir());
    }
}
