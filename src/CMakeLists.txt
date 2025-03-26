# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
# MIT License.
# --------------------------------------------------------------------------

set(TARGET "tilefusion")

file(GLOB_RECURSE SOURCES "kernels/*.cu" "*.cc")

cuda_add_library(${TARGET} SHARED ${SOURCES})

# Refer to this issue for more context:
# https://github.com/pytorch/pytorch/issues/13541
target_compile_definitions(${TARGET} PUBLIC _GLIBCXX_USE_CXX11_ABI=0)

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
target_link_libraries(${TARGET} "${TORCH_LIBRARIES}")

if(USE_CUDA)
  set_target_properties(${TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
