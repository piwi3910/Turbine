# License and notices

Turbine is licensed under the Apache License, Version 2.0. The full text is in [LICENSE](https://github.com/piwi3910/Turbine/blob/main/LICENSE) in the repository.

Copyright 2026 The Turbine Authors.

## Third-party code

The HIP kernel library (`kernels/rocm`) contains code ported or adapted from the projects below. Each such source file names the upstream file, the commit it was taken from, what was taken and what was changed. The full licence texts are in `kernels/rocm/third_party/LICENSES/`.

| Project                                                                                                    | Licence            | Where it appears                                                                                                                               |
| ---------------------------------------------------------------------------------------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Composable Kernel (ROCm/rocm-libraries, projects/composablekernel), Copyright Advanced Micro Devices, Inc. | MIT                | Fetched at build time at a pinned commit (`kernels/rocm/CMakeLists.txt`) and compiled into `libturbine_hip.so`, for the fused attention paths. |
| vLLM (github.com/vllm-project/vllm), copyright the vLLM contributors                                       | Apache License 2.0 | Ported device code, for example the wvSplitK skinny GEMM (`kernels/rocm/src/gemm_wvsplitk.hip`).                                               |
| llama.cpp and ggml (github.com/ggml-org/llama.cpp), copyright the ggml authors                             | MIT                | Ported device code, for example the gated delta net recurrence (`kernels/rocm/src/gdn.hip`).                                                   |
| AITER (github.com/ROCm/aiter), Copyright Advanced Micro Devices, Inc.                                      | MIT                | Ported device code, for example the causal conv1d prefill (`kernels/rocm/src/conv1d.hip`).                                                     |

## Test data

`tests/eval/` contains items derived from GSM8K (openai/grade-school-math, MIT License). The exact sources and revisions are in `tests/eval/NOTICE`.

## Where to read more

- [NOTICE](https://github.com/piwi3910/Turbine/blob/main/NOTICE): the complete attribution file.
- [LICENSE](https://github.com/piwi3910/Turbine/blob/main/LICENSE): the Apache License 2.0.
- [Contributing](contributing.md): how contributed code is licensed, and how ported code is recorded.
