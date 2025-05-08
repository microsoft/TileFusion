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

// Runtime version of CeilDiv
inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

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

inline void check_gpu_memory() {
    size_t free_byte;
    size_t total_byte;
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;
    printf("GPU memory usage: used = %f MB, free = %f MB, total = %f MB\n",
           used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0,
           total_db / 1024.0 / 1024.0);
}

}  // namespace tilefusion
