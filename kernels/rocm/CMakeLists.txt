# libturbine_hip.so: the Turbine kernel C ABI (kernels/include/turbine_kernels.h)
# on ROCm. GEMM through hipBLASLt, attention through Composable Kernel ck_tile
# FMHA (contiguous and, for 128-multiple pages, paged; split-KV for paged decode
# of grouped query heads), RMSNorm through ck_tile
# rmsnorm2d where an instance exists, MoE expert GEMMs through hipBLASLt, and
# Turbine HIP kernels for the rest (16-token paged attention, MoE routing,
# elementwise ops). Loaded at run time by turbine-kernels. The same build also
# produces libturbine_hip_v23.so: the same kernels without the optional ABI
# v2.4 implementation group (minor 3), for the v2.3 fallback test.
#
#   cmake -S kernels/rocm -B <build> -DCMAKE_HIP_COMPILER=/opt/rocm/rocm/bin/hipcc \
#         -DGPU_TARGETS=gfx1201 && cmake --build <build>
#
# GPU_TARGETS defaults to every architecture of cmake/card_profiles.cmake (the card profiles of
# crates/turbine-kernels/src/cards/) and must name only those.
#
# TURBINE_ROCM_PATH (cache variable or environment) overrides /opt/rocm/rocm.
cmake_minimum_required(VERSION 3.25)

set(TURBINE_ROCM_PATH "/opt/rocm/rocm" CACHE PATH "ROCm installation root")
if(DEFINED ENV{TURBINE_ROCM_PATH})
  set(TURBINE_ROCM_PATH "$ENV{TURBINE_ROCM_PATH}" CACHE PATH "ROCm installation root" FORCE)
endif()
message(STATUS "ROCm root: ${TURBINE_ROCM_PATH}")

# The architectures the card profiles describe (TURBINE_PROFILE_ARCHS): the default build list,
# and the only ones a GPU_TARGETS entry may name.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/card_profiles.cmake)
set(GPU_TARGETS "${TURBINE_PROFILE_ARCHS}" CACHE STRING "GPU architectures to compile for")
foreach(arch IN LISTS GPU_TARGETS)
  if(NOT arch IN_LIST TURBINE_PROFILE_ARCHS)
    message(FATAL_ERROR "GPU_TARGETS ${arch} has no card profile; libturbine_hip.so builds the \
card-profile architectures only (${TURBINE_PROFILE_ARCHS}, kernels/rocm/cmake/card_profiles.cmake)")
  endif()
endforeach()
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
# paged attention for pages of a multiple of 128 tokens (ABI v2 paged attention).
# fwd_splitkv: group-mode split-KV paged decode without a mask (the
# ck_tile_fmha_splitkv decode implementation) and its combine kernels
# (filter "<combine>@<splitkv>"). One --filter pattern per --api entry, in order.
set(FMHA_GEN_DIR "${CMAKE_BINARY_DIR}/fmha")
file(MAKE_DIRECTORY "${FMHA_GEN_DIR}")
set(FMHA_GEN_ARGS
  "${CK_FMHA_DIR}/generate.py"
  --targets ${GPU_TARGETS}
  --api fwd,pagedkv_prefill,fwd_splitkv
  --filter "*d128_bf16*nlogits_nbias*nlse_ndropout_nskip_nqscale_ntrload_nsink*,*d128_bf16_group*nlogits_nbias*nlse_nskip_nsquant_pagedkv_nsink*,*d128_bf16_group*_nsquant*@*d128_bf16_group*nlogits_nbias_nmask_lse_nsquant_pagedkv_nsink*")
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
set(fmha_pagedkv_blobs ${FMHA_BLOBS})
list(FILTER fmha_pagedkv_blobs INCLUDE REGEX "fmha_fwd_pagedkv_d128_bf16_group")
if(NOT fmha_pagedkv_blobs)
  message(FATAL_ERROR "CK FMHA generator listed no gfx1201 d128 bf16 group-mode pagedkv instance")
endif()
set(fmha_splitkv_blobs ${FMHA_BLOBS})
list(FILTER fmha_splitkv_blobs INCLUDE REGEX "fmha_fwd_splitkv_d128_bf16_group.*_pagedkv_")
set(fmha_combine_blobs ${FMHA_BLOBS})
list(FILTER fmha_combine_blobs INCLUDE REGEX "fmha_fwd_splitkv_combine_d128_bf16_group")
if(NOT fmha_splitkv_blobs OR NOT fmha_combine_blobs)
  message(FATAL_ERROR "CK FMHA generator listed no gfx1201 d128 bf16 group-mode split-KV paged instance or combine kernel")
endif()
message(STATUS "CK FMHA: ${fmha_blob_count} generated sources")
file(GLOB_RECURSE FMHA_CODEGEN_SCRIPTS "${CK_FMHA_DIR}/codegen/*.py")
add_custom_command(
  OUTPUT ${FMHA_BLOBS}
  COMMAND "${Python3_EXECUTABLE}" ${FMHA_GEN_ARGS} --output_dir "${FMHA_GEN_DIR}"
  DEPENDS "${CK_FMHA_DIR}/generate.py" ${FMHA_CODEGEN_SCRIPTS}
  COMMENT "Generating CK FMHA forward and pagedkv instances for ${GPU_TARGETS}")

# ---- tuned GEMM tables: kernels/rocm/tuning/<arch>/gemm.tsv, compiled in ----
# One row per GEMM shape and m bucket: the hipBLASLt solution the tuner
# (turbine_gemm_tune below) measured fastest on that card. Editing a table re-runs
# the configure step.
set(TURBINE_TUNING_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tuning" CACHE PATH
  "Directory of the tuned GEMM tables (<arch>/gemm.tsv); an empty directory builds a library without them")
set(GEMM_TABLE_INC "${CMAKE_BINARY_DIR}/gemm_tuned_table.inc")
set(gemm_table_sources "${CMAKE_CURRENT_SOURCE_DIR}/cmake/gemm_table.py")
foreach(arch IN LISTS GPU_TARGETS)
  if(EXISTS "${TURBINE_TUNING_DIR}/${arch}/gemm.tsv")
    list(APPEND gemm_table_sources "${TURBINE_TUNING_DIR}/${arch}/gemm.tsv")
  endif()
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${gemm_table_sources})
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/gemm_table.py"
    --tuning-dir "${TURBINE_TUNING_DIR}" --archs "${GPU_TARGETS}" --out "${GEMM_TABLE_INC}"
  RESULT_VARIABLE gemm_table_rc)
if(NOT gemm_table_rc EQUAL 0)
  message(FATAL_ERROR "cmake/gemm_table.py failed (${gemm_table_rc}): a tuned GEMM table is malformed")
endif()

# ---- the shared libraries ----
# Every kernel source is compiled once into an OBJECT library. libturbine_hip.so
# links it with the ABI v2.4 exports (impl_exports.cpp) and abi_minor.cpp
# (minor 7: the v2.5 copy streams and the v2.6 stream handle, copy_stream.cpp,
# the v2.6 sharded RMSNorm, sharded_norm.hip, and the v2.7 host-mapped memory
# and collectives, hostmem.hip, are in every build);
# libturbine_hip_v23.so links the same objects with abi_minor.cpp compiled
# with TURBINE_V23_BUILD and without the v2.4 exports: a real v2.3 library
# (minor 3, no implementation group) for the fallback test
# (turbine-model tiny_model hip_v23_library_matches_cpu).
set(TURBINE_SOURCES
  src/context.cpp
  src/memory.cpp
  src/copy_stream.cpp
  src/graph.cpp
  src/gemm.cpp
  src/gemm_table.cpp
  src/attention.cpp
  src/rmsnorm.cpp
  src/elementwise.hip
  src/paged_attention.cpp
  src/paged_attention_splitkv.cpp
  src/paged_attention.hip
  src/copy_blocks.cpp
  src/moe.cpp
  src/moe.hip
  src/moe_small_m.hip
  src/moe_grouped.hip
  src/logits_reduce.hip
  src/sharded_norm.hip
  src/hostmem.hip
  src/impl_table.cpp)
set(CK_SOURCES src/attention.cpp src/paged_attention.cpp src/paged_attention_splitkv.cpp
  src/rmsnorm.cpp ${FMHA_BLOBS})
set(TURBINE_LIBRARY_SOURCES src/impl_exports.cpp src/abi_minor.cpp)
set_source_files_properties(${TURBINE_SOURCES} ${FMHA_BLOBS} ${TURBINE_LIBRARY_SOURCES}
  PROPERTIES LANGUAGE HIP)
string(REPLACE ";" "," TURBINE_BUILD_ARCHS "${GPU_TARGETS}")
# Composable Kernel's own flags for its RDNA FMHA build (01_fmha/CMakeLists.txt);
# the BF16 conversion mode (round to nearest even, canonical NaN) also applies to
# the rmsnorm2d instance, whose CK default would truncate.
set_source_files_properties(${CK_SOURCES} PROPERTIES COMPILE_OPTIONS
  "-DCK_TILE_FMHA_FWD_FAST_EXP2=1;-fgpu-flush-denormals-to-zero;-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=5;-Wno-undefined-func-template;-Wno-float-equal;-Wno-unknown-warning-option")

# Compile settings shared by the object library and the two shared libraries.
function(turbine_hip_settings target)
  set_target_properties(${target} PROPERTIES
    HIP_ARCHITECTURES "${GPU_TARGETS}"
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    HIP_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
  target_include_directories(${target} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CK_ROOT}/include"
    "${CK_FMHA_DIR}"
    "${FMHA_GEN_DIR}"
    "${CMAKE_BINARY_DIR}")
  target_compile_definitions(${target} PRIVATE
    TURBINE_CK_COMMIT="${TURBINE_CK_COMMIT}"
    TURBINE_BUILD_ARCHS="${TURBINE_BUILD_ARCHS}")
  target_link_libraries(${target} PRIVATE hip::host roc::hipblaslt)
endfunction()

add_library(turbine_hip_objects OBJECT ${TURBINE_SOURCES} ${FMHA_BLOBS})
turbine_hip_settings(turbine_hip_objects)

# Export exactly the ABI: everything else stays local even if a dependency's
# header forces default visibility.
set(TURBINE_EXPORTS "${CMAKE_BINARY_DIR}/turbine_hip.exports")
file(WRITE "${TURBINE_EXPORTS}" "{\n  global: turbine_*;\n  local: *;\n};\n")

# One shared library of the kernel objects plus extra sources.
function(turbine_hip_library target)
  add_library(${target} SHARED $<TARGET_OBJECTS:turbine_hip_objects> ${ARGN})
  turbine_hip_settings(${target})
  set_target_properties(${target} PROPERTIES LINKER_LANGUAGE HIP)
  target_link_options(${target} PRIVATE
    "LINKER:--version-script=${TURBINE_EXPORTS}"
    "LINKER:--no-undefined")
  set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS "${TURBINE_EXPORTS}")
endfunction()

turbine_hip_library(turbine_hip ${TURBINE_LIBRARY_SOURCES})

# abi_minor.cpp is compiled once more for the v2.3 build, with its define.
turbine_hip_library(turbine_hip_v23 src/abi_minor.cpp)
target_compile_definitions(turbine_hip_v23 PRIVATE TURBINE_V23_BUILD)

# The GEMM tuner: times every hipBLASLt solution for the shapes of
# tuning/gemm_shapes.txt on the current device and writes tuning/<arch>/gemm.tsv
# (docs/extending/card-family.md, "Tuned GEMM table"). A lab tool, not loaded by
# the server.
add_executable(turbine_gemm_tune tools/gemm_tune.cpp)
set_source_files_properties(tools/gemm_tune.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(turbine_gemm_tune PROPERTIES HIP_ARCHITECTURES "${GPU_TARGETS}")
target_include_directories(turbine_gemm_tune PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../include"
  "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(turbine_gemm_tune PRIVATE hip::host roc::hipblaslt)
