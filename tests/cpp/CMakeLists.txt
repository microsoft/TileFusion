if(WITH_TESTING)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  file(
    GLOB_RECURSE UNIT_TESTS
    LIST_DIRECTORIES FALSE
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "test_*.cu" "test_*.cc")

  foreach(FILE_PATH ${UNIT_TESTS})
    string(REGEX REPLACE "(.+/)?([^.]+)\\.(cu|cc)$" "\\2" TEST_NAME
                         ${FILE_PATH})

    if("${TEST_NAME}" STREQUAL "test_gemm")
      continue() # the unittest for gemm requires extra dependencies
    endif()

    if("${TEST_NAME}" STREQUAL "test_jit")
      continue() # the unittest for jit requires extra dependencies
    endif()

    cuda_test(${TEST_NAME} SRCS "${CMAKE_CURRENT_SOURCE_DIR}/${FILE_PATH}")
  endforeach()

  cuda_test(test_gemm SRCS "${CMAKE_CURRENT_SOURCE_DIR}/cell/test_gemm.cu"
            "${PROJECT_SOURCE_DIR}/src/cuda_utils.cc" DEPS
            ${CUDA_CUBLAS_LIBRARIES})

  cuda_test(
    test_jit
    SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/jit/test_jit.cc"
    "${PROJECT_SOURCE_DIR}/src/jit/compiler.cc"
    "${PROJECT_SOURCE_DIR}/src/cuda_info.cc"
    DEPS
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIBRARIES}
    glog::glog
    cuda)

endif()
