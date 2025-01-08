# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
# MIT License.
# --------------------------------------------------------------------------

# CMake 3.25 is required for CUDA 20.
cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(tilefusion LANGUAGES C CXX CUDA)

# Prohibit in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source build are not supported")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

option(WITH_TESTING "Build with CTests" ON)
if(WITH_TESTING)
  enable_testing()
endif()

option(ENABLE_DEBUG "Enable debug mode" OFF)

# this is to be compatible with the latest glog. DO NOT remove it.
add_compile_definitions(GLOG_USE_GLOG_EXPORT)

include(generic)
include(dependencies)

include_directories(include)
add_subdirectory(src)
add_subdirectory(tests/cpp)

# FIXME(ying): temporarily disable examples
option(BUILD_EXAMPLES "Build TileFusion with examples" OFF)
if(BUILD_EXAMPLES)
  set(EXAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/examples/cpp")
  file(GLOB SUBDIRS "${EXAMPLES_DIR}/*")

  foreach(SUBDIR ${SUBDIRS})
    if(IS_DIRECTORY ${SUBDIR})
      add_subdirectory(${SUBDIR})
    endif()
  endforeach()
endif()
