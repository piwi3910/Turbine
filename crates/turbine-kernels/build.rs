//! Compiles the stub shim libraries that `shim::tests` load, with the host C compiler (no GPU
//! toolchain). Nothing here links a GPU library: the real shim is loaded at run time.
//!
//! Each variant's path is exported to the crate as a compile-time environment variable:
//! `TURBINE_STUB_ABI999` (ABI version 999) and `TURBINE_STUB_GFX942` (ABI 1, backend `hip`,
//! build archs `gfx942`).
use std::path::{Path, PathBuf};

fn build_stub(out_dir: &Path, name: &str, abi: u32, backend: &str, archs: &str) -> PathBuf {
    let manifest = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"),
    );
    let include = manifest.join("../../kernels/include");
    let source = manifest.join("stub/stub_shim.c");
    let output = out_dir.join(format!("lib{name}.so"));
    let compiler = cc::Build::new().cargo_metadata(false).get_compiler();
    let mut cmd = compiler.to_command();
    cmd.args(["-std=c11", "-shared", "-fPIC", "-O0"])
        .arg(format!("-I{}", include.display()))
        .arg(format!("-DSTUB_ABI={abi}u"))
        .arg(format!("-DSTUB_BACKEND=\"{backend}\""))
        .arg(format!("-DSTUB_ARCHS=\"{archs}\""))
        .arg("-o")
        .arg(&output)
        .arg(&source);
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("cannot run the host C compiler {:?}: {e}", compiler.path()));
    assert!(
        status.success(),
        "building stub shim {name} failed: {cmd:?}"
    );
    output
}

fn main() {
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let abi999 = build_stub(&out_dir, "turbine_stub_abi999", 999, "hip", "gfx1201");
    let gfx942 = build_stub(&out_dir, "turbine_stub_gfx942", 1, "hip", "gfx942");
    println!("cargo:rustc-env=TURBINE_STUB_ABI999={}", abi999.display());
    println!("cargo:rustc-env=TURBINE_STUB_GFX942={}", gfx942.display());
    println!("cargo:rerun-if-changed=stub/stub_shim.c");
    println!("cargo:rerun-if-changed=../../kernels/include/turbine_kernels.h");
}
