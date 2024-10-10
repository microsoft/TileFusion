// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cublas_gemm.cuh"

#include <torch/script.h>

void gemm_op(int64_t m, int64_t n, int64_t k, const torch::Tensor& A,
             const torch::Tensor& B, torch::Tensor& C, torch::Tensor& time,
             int64_t iters = 20, int64_t warm_up = 5) {
    using namespace benchmarks;
    using DType = __half;

    auto* dA = reinterpret_cast<const DType*>(A.data_ptr());
    auto* dB = reinterpret_cast<const DType*>(B.data_ptr());
    auto* dC = reinterpret_cast<DType*>(C.data_ptr());
    auto* time_data = reinterpret_cast<float*>(time.data_ptr());

    cublas_hgemm(m, n, k, dA, dB, dC, time_data, iters, warm_up);
}

TORCH_LIBRARY(cublas_gemm, t) { t.def("gemm", &gemm_op); };
