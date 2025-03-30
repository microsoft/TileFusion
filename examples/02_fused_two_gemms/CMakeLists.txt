# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
# MIT License.
# --------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(fused_two_consecutive_gemms LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../../cmake")
include(generic)

list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

set(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/../../3rd-party")
include_directories(${THIRD_PARTY_DIR}/cutlass/include)
include_directories("${PROJECT_SOURCE_DIR}/../../include")
include_directories(${TORCH_INCLUDE_DIRS})

add_executable(fused_two_consecutive_gemms fused_two_consecutive_gemms.cu)
target_link_libraries(fused_two_consecutive_gemms ${CUDA_CUBLAS_LIBRARIES}
                      ${TORCH_LIBRARIES})
