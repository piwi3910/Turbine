# libturbine_hip.so: the Turbine kernel C ABI (kernels/include/turbine_kernels.h)
# on ROCm. GEMM through hipBLASLt, attention through Composable Kernel ck_tile
# FMHA (contiguous and, for 128-multiple pages, paged), RMSNorm through ck_tile
# rmsnorm2d where an instance exists, MoE expert GEMMs through hipBLASLt, and
# Turbine HIP kernels for the rest (16-token paged attention, MoE routing,
# elementwise ops). Loaded at run time by turbine-kernels.
#
#   cmake -S kernels/rocm -B <build> -DCMAKE_HIP_COMPILER=/opt/rocm/rocm/bin/hipcc \
#         -DGPU_TARGETS=gfx1201 && cmake --build <build>
#
# TURBINE_ROCM_PATH (cache variable or environment) overrides /opt/rocm/rocm.
cmake_minimum_required(VERSION 3.25)

set(TURBINE_ROCM_PATH "/opt/rocm/rocm" CACHE PATH "ROCm installation root")
if(DEFINED ENV{TURBINE_ROCM_PATH})
  set(TURBINE_ROCM_PATH "$ENV{TURBINE_ROCM_PATH}" CACHE PATH "ROCm installation root" FORCE)
endif()
message(STATUS "ROCm root: ${TURBINE_ROCM_PATH}")

set(GPU_TARGETS "gfx1201" CACHE STRING "GPU architectures to compile for")
if(NOT GPU_TARGETS STREQUAL "gfx1201")
  message(FATAL_ERROR "libturbine_hip.so is built and validated for gfx1201 only (GPU_TARGETS=${GPU_TARGETS})")
endif()
set(CMAKE_HIP_ARCHITECTURES "${GPU_TARGETS}")

# CMake's HIP language drives ROCm's clang directly and rejects the hipcc
# wrapper; accept the documented -DCMAKE_HIP_COMPILER=<rocm>/bin/hipcc and use
# the clang it wraps.
if(CMAKE_HIP_COMPILER MATCHES "hipcc$")
  set(CMAKE_HIP_COMPILER "${TURBINE_ROCM_PATH}/lib/llvm/bin/clang++" CACHE FILEPATH "HIP compiler" FORCE)
endif()
if(NOT CMAKE_HIP_COMPILER)
  set(CMAKE_HIP_COMPILER "${TURBINE_ROCM_PATH}/lib/llvm/bin/clang++")
endif()
# ROCm's clang derives the HIP runtime location from its real path, which on a
# versioned install (…/core-7.14/lib/llvm) is not the ROCm root: name it.
set(CMAKE_HIP_COMPILER_ROCM_ROOT "${TURBINE_ROCM_PATH}")
string(PREPEND CMAKE_HIP_FLAGS "--rocm-path=${TURBINE_ROCM_PATH} ")

project(turbine_hip LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

list(PREPEND CMAKE_PREFIX_PATH "${TURBINE_ROCM_PATH}")
find_package(hip CONFIG REQUIRED)
find_package(hipblaslt CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# ---- Composable Kernel: fetched (not built) at the pinned commit ----
# Tag therock-7.14.1; identical to the installed ${TURBINE_ROCM_PATH}/include/ck_tile.
set(TURBINE_CK_REPOSITORY "https://github.com/ROCm/rocm-libraries.git")
set(TURBINE_CK_COMMIT "cd9574023093742434e8c992d13b89ab9a6c1cf8")
include(FetchContent)
FetchContent_Declare(composable_kernel
  DOWNLOAD_COMMAND
    ${CMAKE_COMMAND}
    -DREPOSITORY=${TURBINE_CK_REPOSITORY}
    -DCOMMIT=${TURBINE_CK_COMMIT}
    -DSUBDIR=projects/composablekernel
    -DDEST=<SOURCE_DIR>
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_ck.cmake
  # No CMakeLists.txt there: CK is fetched for its headers and generators only.
  SOURCE_SUBDIR turbine-fetch-only)
FetchContent_MakeAvailable(composable_kernel)
set(CK_ROOT "${composable_kernel_SOURCE_DIR}/projects/composablekernel")
set(CK_FMHA_DIR "${CK_ROOT}/example/ck_tile/01_fmha")
if(NOT EXISTS "${CK_FMHA_DIR}/generate.py")
  message(FATAL_ERROR "Composable Kernel checkout at ${CK_ROOT} lacks ${CK_FMHA_DIR}/generate.py")
endif()

# ---- CK FMHA instances: head_dim 128, BF16, no bias/lse/dropout/sink ----
# fwd: contiguous prefill/decode (ABI v1 attention). pagedkv_prefill: group-mode
# paged attention for pages of a multiple of 128 tokens (ABI v2 paged attention);
# one --filter pattern per --api entry, in order.
set(FMHA_GEN_DIR "${CMAKE_BINARY_DIR}/fmha")
file(MAKE_DIRECTORY "${FMHA_GEN_DIR}")
set(FMHA_GEN_ARGS
  "${CK_FMHA_DIR}/generate.py"
  --targets ${GPU_TARGETS}
  --api fwd,pagedkv_prefill
  --filter "*d128_bf16*nlogits_nbias*nlse_ndropout_nskip_nqscale_ntrload_nsink*,*d128_bf16_group*nlogits_nbias*nlse_nskip_nsquant_pagedkv_nsink*")
file(REMOVE "${FMHA_GEN_DIR}/blobs.txt")
execute_process(
  COMMAND "${Python3_EXECUTABLE}" ${FMHA_GEN_ARGS} --list_blobs "${FMHA_GEN_DIR}/blobs.txt"
  RESULT_VARIABLE fmha_list_rc)
if(NOT fmha_list_rc EQUAL 0)
  message(FATAL_ERROR "CK FMHA generate.py --list_blobs failed (${fmha_list_rc})")
endif()
file(STRINGS "${FMHA_GEN_DIR}/blobs.txt" FMHA_BLOBS)
list(LENGTH FMHA_BLOBS fmha_blob_count)
if(fmha_blob_count LESS 2)
  message(FATAL_ERROR "CK FMHA generator listed no gfx1201 d128 bf16 instances")
endif()
list(FILTER FMHA_BLOBS INCLUDE REGEX "fmha_fwd_pagedkv_d128_bf16_group" OUTPUT_VARIABLE fmha_pagedkv_blobs)
if(NOT fmha_pagedkv_blobs)
  message(FATAL_ERROR "CK FMHA generator listed no gfx1201 d128 bf16 group-mode pagedkv instance")
endif()
message(STATUS "CK FMHA: ${fmha_blob_count} generated sources")
file(GLOB_RECURSE FMHA_CODEGEN_SCRIPTS "${CK_FMHA_DIR}/codegen/*.py")
add_custom_command(
  OUTPUT ${FMHA_BLOBS}
  COMMAND "${Python3_EXECUTABLE}" ${FMHA_GEN_ARGS} --output_dir "${FMHA_GEN_DIR}"
  DEPENDS "${CK_FMHA_DIR}/generate.py" ${FMHA_CODEGEN_SCRIPTS}
  COMMENT "Generating CK FMHA forward and pagedkv instances for ${GPU_TARGETS}")

# ---- the shared library ----
set(TURBINE_SOURCES
  src/context.cpp
  src/memory.cpp
  src/gemm.cpp
  src/attention.cpp
  src/rmsnorm.cpp
  src/elementwise.hip
  src/paged_attention.cpp
  src/paged_attention.hip
  src/copy_blocks.cpp
  src/moe.cpp
  src/moe.hip)
set(CK_SOURCES src/attention.cpp src/paged_attention.cpp src/rmsnorm.cpp ${FMHA_BLOBS})

add_library(turbine_hip SHARED ${TURBINE_SOURCES} ${FMHA_BLOBS})
set_source_files_properties(${TURBINE_SOURCES} ${FMHA_BLOBS} PROPERTIES LANGUAGE HIP)
set_target_properties(turbine_hip PROPERTIES
  HIP_ARCHITECTURES "${GPU_TARGETS}"
  CXX_VISIBILITY_PRESET hidden
  HIP_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  LINKER_LANGUAGE HIP)
target_include_directories(turbine_hip PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../include"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CK_ROOT}/include"
  "${CK_FMHA_DIR}"
  "${FMHA_GEN_DIR}")
string(REPLACE ";" "," TURBINE_BUILD_ARCHS "${GPU_TARGETS}")
target_compile_definitions(turbine_hip PRIVATE
  TURBINE_CK_COMMIT="${TURBINE_CK_COMMIT}"
  TURBINE_BUILD_ARCHS="${TURBINE_BUILD_ARCHS}")
# Composable Kernel's own flags for its RDNA FMHA build (01_fmha/CMakeLists.txt);
# the BF16 conversion mode (round to nearest even, canonical NaN) also applies to
# the rmsnorm2d instance, whose CK default would truncate.
set_source_files_properties(${CK_SOURCES} PROPERTIES COMPILE_OPTIONS
  "-DCK_TILE_FMHA_FWD_FAST_EXP2=1;-fgpu-flush-denormals-to-zero;-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=5;-Wno-undefined-func-template;-Wno-float-equal;-Wno-unknown-warning-option")
target_link_libraries(turbine_hip PRIVATE hip::host roc::hipblaslt)

# Export exactly the ABI: everything else stays local even if a dependency's
# header forces default visibility.
set(TURBINE_EXPORTS "${CMAKE_BINARY_DIR}/turbine_hip.exports")
file(WRITE "${TURBINE_EXPORTS}" "{\n  global: turbine_*;\n  local: *;\n};\n")
target_link_options(turbine_hip PRIVATE
  "LINKER:--version-script=${TURBINE_EXPORTS}"
  "LINKER:--no-undefined")
set_property(TARGET turbine_hip APPEND PROPERTY LINK_DEPENDS "${TURBINE_EXPORTS}")
