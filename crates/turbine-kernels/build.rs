//! Compiles the stub shim libraries that `shim::tests` load, with the host C compiler (no GPU
//! toolchain). Nothing here links a GPU library: the real shim is loaded at run time.
//!
//! Each variant's path is exported to the crate as a compile-time environment variable:
//! `TURBINE_STUB_ABI999` (ABI version 999), `TURBINE_STUB_GFX942` (ABI 2.0, backend `hip`, build
//! archs `gfx942`), `TURBINE_STUB_GFX942_V21` (the same with the optional ABI v2.1 and v2.3
//! symbols, compiled with `-DTURBINE_STUB_V21`; it reports minor 3), `TURBINE_STUB_GFX942_V24`
//! (also the v2.4 implementation group, `-DTURBINE_STUB_V24`; minor 4),
//! `TURBINE_STUB_GFX942_V25` (also the v2.5 copy streams, `-DTURBINE_STUB_V25`; minor 5),
//! `TURBINE_STUB_GFX942_V26` (also the v2.6 native stream handles and sharded RMSNorm trios,
//! `-DTURBINE_STUB_V26`; minor 6), `TURBINE_STUB_GFX942_V27` (also the v2.7 host-mapped
//! memory and collectives, `-DTURBINE_STUB_V27`; minor 7), `TURBINE_STUB_GFX942_V28` (also
//! the v2.8 device-sequenced collective step, `-DTURBINE_STUB_V28`; minor 8) and
//! `TURBINE_STUB_GFX942_V29` (also the v2.9 quantized GEMM and activation quantization trios,
//! `-DTURBINE_STUB_V29`; minor 9).
use std::path::{Path, PathBuf};

fn build_stub(
    out_dir: &Path,
    name: &str,
    abi: u32,
    backend: &str,
    archs: &str,
    defines: &[&str],
) -> PathBuf {
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
        .args(defines.iter().map(|d| format!("-D{d}")))
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
    let abi999 = build_stub(&out_dir, "turbine_stub_abi999", 999, "hip", "gfx1201", &[]);
    let gfx942 = build_stub(&out_dir, "turbine_stub_gfx942", 2, "hip", "gfx942", &[]);
    let gfx942_v21 = build_stub(
        &out_dir,
        "turbine_stub_gfx942_v21",
        2,
        "hip",
        "gfx942",
        &["TURBINE_STUB_V21"],
    );
    let gfx942_v24 = build_stub(
        &out_dir,
        "turbine_stub_gfx942_v24",
        2,
        "hip",
        "gfx942",
        &["TURBINE_STUB_V21", "TURBINE_STUB_V24"],
    );
    let gfx942_v25 = build_stub(
        &out_dir,
        "turbine_stub_gfx942_v25",
        2,
        "hip",
        "gfx942",
        &["TURBINE_STUB_V21", "TURBINE_STUB_V24", "TURBINE_STUB_V25"],
    );
    let gfx942_v26 = build_stub(
        &out_dir,
        "turbine_stub_gfx942_v26",
        2,
        "hip",
        "gfx942",
        &[
            "TURBINE_STUB_V21",
            "TURBINE_STUB_V24",
            "TURBINE_STUB_V25",
            "TURBINE_STUB_V26",
        ],
    );
    let gfx942_v27 = build_stub(
        &out_dir,
        "turbine_stub_gfx942_v27",
        2,
        "hip",
        "gfx942",
        &[
            "TURBINE_STUB_V21",
            "TURBINE_STUB_V24",
            "TURBINE_STUB_V25",
            "TURBINE_STUB_V26",
            "TURBINE_STUB_V27",
        ],
    );
    let gfx942_v28 = build_stub(
        &out_dir,
        "turbine_stub_gfx942_v28",
        2,
        "hip",
        "gfx942",
        &[
            "TURBINE_STUB_V21",
            "TURBINE_STUB_V24",
            "TURBINE_STUB_V25",
            "TURBINE_STUB_V26",
            "TURBINE_STUB_V27",
            "TURBINE_STUB_V28",
        ],
    );
    let gfx942_v29 = build_stub(
        &out_dir,
        "turbine_stub_gfx942_v29",
        2,
        "hip",
        "gfx942",
        &[
            "TURBINE_STUB_V21",
            "TURBINE_STUB_V24",
            "TURBINE_STUB_V25",
            "TURBINE_STUB_V26",
            "TURBINE_STUB_V27",
            "TURBINE_STUB_V28",
            "TURBINE_STUB_V29",
        ],
    );
    println!(
        "cargo:rustc-env=TURBINE_STUB_GFX942_V29={}",
        gfx942_v29.display()
    );
    println!("cargo:rustc-env=TURBINE_STUB_ABI999={}", abi999.display());
    println!(
        "cargo:rustc-env=TURBINE_STUB_GFX942_V28={}",
        gfx942_v28.display()
    );
    println!(
        "cargo:rustc-env=TURBINE_STUB_GFX942_V27={}",
        gfx942_v27.display()
    );
    println!(
        "cargo:rustc-env=TURBINE_STUB_GFX942_V26={}",
        gfx942_v26.display()
    );
    println!(
        "cargo:rustc-env=TURBINE_STUB_GFX942_V25={}",
        gfx942_v25.display()
    );
    println!("cargo:rustc-env=TURBINE_STUB_GFX942={}", gfx942.display());
    println!(
        "cargo:rustc-env=TURBINE_STUB_GFX942_V21={}",
        gfx942_v21.display()
    );
    println!(
        "cargo:rustc-env=TURBINE_STUB_GFX942_V24={}",
        gfx942_v24.display()
    );
    println!("cargo:rerun-if-changed=stub/stub_shim.c");
    println!("cargo:rerun-if-changed=../../kernels/include/turbine_kernels.h");
}
