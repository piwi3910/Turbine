//! Compiles the NCCL-API stub library that `collective::ffi` tests load, with the host C
//! compiler (no GPU toolchain; nothing here links RCCL or NCCL). Variants land in
//! `$OUT_DIR/<variant>/<file name>` and the directory is exported to the crate as
//! `TURBINE_NCCL_STUB_DIR`:
//! `rccl/librccl.so.1`, `nccl/libnccl.so.2` (version 23004), `low/librccl.so.1` (version 21800)
//! `nogroupend/libnccl.so.2` (built without `ncclGroupEnd`) and `inithang/librccl.so.1`
//! (communicator init that never completes).
use std::path::{Path, PathBuf};

/// `NCCL_VERSION_CODE` the stubs report (2.30.4, the ROCm 7.14.1 RCCL), at or above both
/// recorded minimums.
const STUB_VERSION: u32 = 23_004;
/// Below both minimums (2.18.0).
const LOW_VERSION: u32 = 21_800;

fn build_stub(out_dir: &Path, variant: &str, file: &str, version: u32, defines: &[&str]) {
    let manifest = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"),
    );
    let source = manifest.join("tests/stub/nccl_stub.c");
    let dir = out_dir.join(variant);
    std::fs::create_dir_all(&dir)
        .unwrap_or_else(|e| panic!("cannot create {}: {e}", dir.display()));
    let output = dir.join(file);
    let compiler = cc::Build::new().cargo_metadata(false).get_compiler();
    let mut cmd = compiler.to_command();
    cmd.args(["-std=c11", "-shared", "-fPIC", "-O0"])
        .arg(format!("-DSTUB_VERSION={version}"));
    for d in defines {
        cmd.arg(format!("-D{d}"));
    }
    cmd.arg("-o").arg(&output).arg(&source);
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("cannot run the host C compiler {:?}: {e}", compiler.path()));
    assert!(
        status.success(),
        "building NCCL stub {variant} failed: {cmd:?}"
    );
}

fn main() {
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    build_stub(&out_dir, "rccl", "librccl.so.1", STUB_VERSION, &[]);
    build_stub(&out_dir, "nccl", "libnccl.so.2", STUB_VERSION, &[]);
    build_stub(&out_dir, "low", "librccl.so.1", LOW_VERSION, &[]);
    build_stub(
        &out_dir,
        "nogroupend",
        "libnccl.so.2",
        STUB_VERSION,
        &["STUB_OMIT_GROUP_END"],
    );
    build_stub(
        &out_dir,
        "inithang",
        "librccl.so.1",
        STUB_VERSION,
        &["STUB_INIT_NEVER_COMPLETES"],
    );
    println!(
        "cargo:rustc-env=TURBINE_NCCL_STUB_DIR={}",
        out_dir.display()
    );
    println!("cargo:rustc-env=TURBINE_NCCL_STUB_VERSION={STUB_VERSION}");
    println!("cargo:rerun-if-changed=tests/stub/nccl_stub.c");
}
