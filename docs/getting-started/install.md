# Install and build

This page covers what Turbine needs, how to build the Rust workspace, and how to build the HIP kernel library that serves models on AMD GPUs. The host build needs no GPU, no ROCm and no model weights.

## Requirements

| Item                                           | Needed for            | Notes                                                                                                     |
| ---------------------------------------------- | --------------------- | --------------------------------------------------------------------------------------------------------- |
| Rust 1.97 or newer, edition 2024               | every build           | Install with [rustup](https://rustup.rs). The workspace sets `rust-version`.                              |
| Git                                            | cloning               |                                                                                                           |
| A Linux host with ROCm and a supported AMD GPU | serving models        | Primary target: AMD Radeon AI PRO R9700 (gfx1201).                                                        |
| CMake and a HIP compiler (`hipcc`)             | `libturbine_hip.so`   | Build the kernel library on the ROCm host. The minimum CMake version is in `kernels/rocm/CMakeLists.txt`. |
| Network access during the kernel build         | Composable Kernel     | CMake fetches Composable Kernel at a pinned commit.                                                       |
| Model weights                                  | serving and GPU tests | Hugging Face checkpoints in safetensors format; see [quickstart](quickstart.md).                          |

## Build the Rust workspace

Clone the repository and build:

```sh
git clone https://github.com/piwi3910/Turbine.git
cd Turbine
cargo build --workspace
```

Release builds are needed for serving and benchmarking:

```sh
cargo build --release -p turbine-server
cargo build --release -p turbine-bench --bin turbine-bench --bin turbine-golden
```

The workspace has these binaries:

- `turbine-server`: the server. Its flags are `--config`, `--set`, `--check-config`, `--support-matrix` and `--output`.
- `turbine-bench`: a load generator, and the offline KV policy simulator (`kv-sim`).
- `turbine-golden`: golden compare, golden capture, GSM8K evaluation and comparison.

## Host checks

These run on any Linux or macOS machine with Rust installed:

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

The CPU backend serves tiny checkpoints for development. It is experimental, so the server logs a warning when it is selected.

Confirm that no vendor GPU crate is in the dependency tree. The output of the following command should be empty:

```sh
cargo tree --workspace | grep -Ei 'hip|rocm|cuda' || echo "no vendor GPU crate"
```

## Build the HIP kernel library

Run this on a ROCm host with `hipcc` installed. The kernel library is built from `kernels/rocm`:

```sh
cmake -S kernels/rocm -B build/rocm \
  -DCMAKE_HIP_COMPILER="$(command -v hipcc)" \
  -DGPU_TARGETS=gfx1201
cmake --build build/rocm
```

The result is `build/rocm/libturbine_hip.so`. Two notes:

- `GPU_TARGETS` selects the device architecture. The kernels are tuned for `gfx1201`, which is the R9700. Other architectures are not in the support matrix.
- If ROCm is not in the default location, set `TURBINE_ROCM_PATH` to the ROCm root before running CMake.

The library uses the ABI declared in `kernels/include/turbine_kernels.h`. The server checks the ABI version at load time, and optional symbols (for example the ABI v2.9 quantized GEMM or the v2.11 KV transcode) are probed. A library that lacks a symbol that the configuration needs either degrades with a warning or refuses to start with a reason code, depending on the feature.

## Locate the kernel library at run time

The server finds `libturbine_hip.so` in this order:

1. `execution.kernel_library` in the configuration, when it is not `null`.
2. The `TURBINE_KERNEL_LIBRARY` environment variable.
3. `libturbine_hip.so` beside the `turbine-server` executable.
4. The dynamic loader's search path.

An explicitly configured library that fails to load exits with code 1.

## Check the configuration

Without a GPU, two commands check that the installation works:

```sh
cargo run -p turbine-server -- --config examples/turbine.yaml --check-config
cargo run -p turbine-server -- --support-matrix
```

The first validates the example configuration and prints the support row it resolves to, then `config ok`. The second prints every row of the support matrix. Both exit with code 0 on success and code 2 on an invalid configuration.

## Next

- [Quickstart](quickstart.md): serve a model on the R9700.
- [Configuration reference](configuration.md): every configuration section.
