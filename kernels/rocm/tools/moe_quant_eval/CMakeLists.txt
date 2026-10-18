cmake_minimum_required(VERSION 3.21)
project(moe_quant_eval LANGUAGES CXX HIP)
# P7 Task 14 harness: links a built libturbine_hip.so (TURBINE_LIB) and hipBLASLt.
# Build (ROCm host; CMake refuses the hipcc wrapper as the HIP compiler, use ROCm's clang++):
#   cmake -S kernels/rocm/tools/moe_quant_eval -B <build> -G Ninja -DGPU_TARGETS=gfx1201 \
#     -DCMAKE_HIP_COMPILER=$(hipconfig -l)/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm/rocm \
#     -DTURBINE_SRC=<repo> -DTURBINE_LIB=<kernel build>/libturbine_hip.so && cmake --build <build>
# Run (GPU 0, under scripts/bench-lock.sh): LD_LIBRARY_PATH=<kernel build> <build>/moe_quant_eval
#   [--shapes qwen3_30b_a3b,mixtral_8x7b,gpt_oss_20b] [--tokens 1,4,16,256,2048] [--cands <names>] [--check-only 1]
set(TURBINE_SRC "" CACHE PATH "Turbine source root")
set(TURBINE_LIB "" CACHE FILEPATH "libturbine_hip.so")
find_package(hip REQUIRED)
find_package(hipblaslt REQUIRED)
add_executable(moe_quant_eval moe_quant_eval.cpp)
set_source_files_properties(moe_quant_eval.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(moe_quant_eval PROPERTIES HIP_ARCHITECTURES "${GPU_TARGETS}" CXX_STANDARD 20 HIP_STANDARD 20)
target_include_directories(moe_quant_eval PRIVATE "${TURBINE_SRC}/kernels/include")
target_link_libraries(moe_quant_eval PRIVATE hip::host roc::hipblaslt "${TURBINE_LIB}")
