// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "config.hpp"

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

namespace tilefusion {
template <int a, int b>
inline constexpr int CeilDiv = (a + b - 1) / b;  // for compile-time values

const char* cublasGetErrorString(cublasStatus_t status);

inline void __cudaCheck(const cudaError err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s(%d): CUDA error: %s.\n", file, line,
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(call) __cudaCheck(call, __FILE__, __LINE__)

inline void __cublasCheck(const cublasStatus_t err, const char* file,
                          int line) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s(%d): Cublas error: %s.\n", file, line,
                cublasGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
#define CUBLAS_CHECK(call) __cublasCheck(call, __FILE__, __LINE__)

#define CUDA_DRIVER_CHECK(call)                                                \
    do {                                                                       \
        CUresult result = call;                                                \
        if (result != CUDA_SUCCESS) {                                          \
            const char* error_string;                                          \
            cuGetErrorString(result, &error_string);                           \
            std::stringstream err;                                             \
            err << "CUDA error: " << error_string << " (" << result << ") at " \
                << __FILE__ << ":" << __LINE__;                                \
            LOG(ERROR) << err.str();                                           \
            throw std::runtime_error(err.str());                               \
        }                                                                      \
    } while (0)

}  // namespace tilefusion
