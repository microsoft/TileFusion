// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "util/cuda_timer.hpp"

#include <cublas_v2.h>

namespace benchmarks {
// In this implementation, A and C is laid out in row-major, B, is laid out
// in column-major: C[m, n] = A[m, k] @ B[k, n]
void cublas_hgemm(int64_t kM, int64_t kN, int64_t kK,  // problem shape
                  const __half* A, const __half* B, __half* C, float* time,
                  int64_t iters = 20, int64_t warm_up = 5) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    __half alf = static_cast<__half>(1.);
    __half bet = static_cast<__half>(0.);

    if (iters) {  // measure time
        for (int i = 0; i < warm_up; ++i) {
            cublasHgemm(handle, CUBLAS_OP_T /* transb*/, CUBLAS_OP_N, kN, kM,
                        kK, &alf, B, kK, A, kK, &bet, C, kN);
        }
        cudaDeviceSynchronize();

        tilefusion::CudaTimer timer;
        timer.start();
        for (int i = 0; i < iters; ++i) {
            cublasHgemm(handle, CUBLAS_OP_T /* transb*/, CUBLAS_OP_N, kN, kM,
                        kK, &alf, B, kK, A, kK, &bet, C, kN);
        }
        cudaDeviceSynchronize();
        time[0] = timer.stop() / iters;
    } else {
        // C = A @ B, but in cuBLAS, matrix is by default laid out in
        // column-major, therefore we compute:
        // C^T = B^T @ A^T [n, m] = [n, k] @ [k, m]
        cublasHgemm(handle, CUBLAS_OP_T /* transb*/, CUBLAS_OP_N, kN, kM, kK,
                    &alf, B, kK, A, kK, &bet, C, kN);
        cudaDeviceSynchronize();
    }

    cublasDestroy(handle);
}
}  // namespace benchmarks
