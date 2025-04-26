# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
# MIT License.
# --------------------------------------------------------------------------

set(TARGET "tilefusion")

file(GLOB_RECURSE SOURCES "kernels/*.cu" "kernels/*.cc" "*.cc")

# Create the target first before setting properties
cuda_add_library(${TARGET} SHARED ${SOURCES})

# Define path for NVCC
add_compile_definitions(NVCC_PATH="${CMAKE_CUDA_COMPILER}")

# Now set properties on the existing target
set_target_properties(
  ${TARGET}
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
             CUDA_EXTENSIONS OFF
             CUDA_RESOLVE_DEVICE_SYMBOLS ON
             CUDA_SEPARABLE_COMPILATION ON)

execute_process(
  COMMAND python -c "import torch; print(torch.__version__)"
  OUTPUT_VARIABLE TORCH_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if(TORCH_VERSION VERSION_LESS "2.7.0")
  message(STATUS "PyTorch version ${TORCH_VERSION} detected, using old ABI")
  # Refer to this issue for more context:
  # https://github.com/pytorch/pytorch/issues/13541
  target_compile_definitions(${TARGET} PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
else()
  message(STATUS "PyTorch version ${TORCH_VERSION} detected, using new ABI")
endif()

target_compile_options(
  ${TARGET}
  PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
         -Werror,-Wall
         -rdc=true
         -std=c++20
         -fconcepts
         -fpermissive
         --use_fast_math
         --generate-line-info
         >)
target_compile_features(${TARGET} PUBLIC cxx_std_20 cuda_std_20)

# Link with CUDA libraries
find_package(CUDAToolkit REQUIRED)
target_link_libraries(${TARGET} "${TORCH_LIBRARIES}" CUDA::cudart
                      CUDA::cuda_driver)
