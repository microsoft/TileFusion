# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
# MIT License.
# --------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(gemm_bench LANGUAGES C CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      "${PROJECT_SOURCE_DIR}/../../../cmake")
set(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/../../../3rd-party")

include(generic)

include_directories("${PROJECT_SOURCE_DIR}/../../../include")
include_directories("${PROJECT_SOURCE_DIR}/../../utils/cpp")
include_directories("${THIRD_PARTY_DIR}/cutlass/include")

add_executable(bench_fused_two_gemms bench.cu)
target_link_libraries(bench_fused_two_gemms ${CUDA_CUBLAS_LIBRARIES})
