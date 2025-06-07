// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

namespace benchmarks {

template <int a, int b>
inline constexpr int CeilDiv = (a + b - 1) / b;  // for compile-time values

#if defined(__CUDA_ARCH__)
  #define HOST_DEVICE __forceinline__ __host__ __device__
  #define DEVICE __forceinline__ __device__
  #define HOST __forceinline__ __host__
#else
  #define HOST_DEVICE inline
  #define DEVICE inline
  #define HOST inline
#endif

const char* cublasGetErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "unknown error";
}

inline void __cublasCheck(const cublasStatus_t err, const char* file,
                          int line) {
  if (err != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "%s(%d): Cublas error: %s.\n", file, line,
            cublasGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
#define CublasCheck(call) __cublasCheck(call, __FILE__, __LINE__)

inline void __cudaCheck(const cudaError err, const char* file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s(%d): CUDA error: %s.\n", file, line,
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
#define CudaCheck(call) __cudaCheck(call, __FILE__, __LINE__)

}  // namespace benchmarks
