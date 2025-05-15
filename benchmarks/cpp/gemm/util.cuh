// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "util/cuda_timer.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cfloat>

using namespace benchmarks;
using namespace tilefusion;

template <const int kM, const int kN, const int kK>
using GemmShape = TileShape<kM, kN, kK>;

float rand_float(float a = 1e-4, float b = 5e-3) {
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

namespace {
bool check_results_impl(const __half* values1, const __half* values2,
                        int numel) {
    bool passed = true;
    const float epsilon = 1e-3;

    double total_diff = 0.;
    double max_abs_diff = FLT_MIN;
    double diff = 0.;

#ifdef DEBUG
    int cut_off = 128;
    printf("\nground truth:\n");
    for (int i = 0; i < cut_off; ++i) {
        printf("%.5f, ", __half2float(values1[i]));
        if (i && (i + 1) % 16 == 0) printf("\n");
    }
    printf("\ncomputed values:\n");
    for (int i = 0; i < cut_off; ++i) {
        printf("%.5f, ", __half2float(values2[i]));
        if (i && (i + 1) % 16 == 0) printf("\n");
    }
#endif

    for (int i = 0; i < numel; ++i) {
        float v1 = __half2float(values1[i]);
        float v2 = __half2float(values2[i]);

        diff = fabs(v1 - v2);
        max_abs_diff = max_abs_diff < diff ? diff : max_abs_diff;
        total_diff += diff;

#ifdef DEBUG
        if (diff > epsilon) {
            printf("the %d-th value differs (%.4f): %.4f vs. %.4f\n", i, diff,
                   v1, v2);
        }
#endif
    }

    double avg_diff = total_diff / numel;
    if (avg_diff > epsilon) passed = false;

    return passed;
}
}  // namespace

template <typename T>
bool check_results(const T* values1_, const cutlass::half_t* values2,
                   int numel);

template <>
bool check_results(const cutlass::half_t* values1_,
                   const cutlass::half_t* values2_, int numel) {
    const __half* values1 = reinterpret_cast<const __half*>(values1_);
    const __half* values2 = reinterpret_cast<const __half*>(values2_);
    return check_results_impl(values1, values2, numel);
}

template <>
bool check_results(const __half* values1, const cutlass::half_t* values2_,
                   int numel) {
    const __half* values2 = reinterpret_cast<const __half*>(values2_);
    return check_results_impl(values1, values2, numel);
}

template <>
bool check_results(const float* values1, const cutlass::half_t* values2_,
                   int numel) {
    const __half* values2 = reinterpret_cast<const __half*>(values2_);
    __half* hvalues1 = (__half*)malloc(numel * sizeof(__half));
    for (int i = 0; i < numel; ++i) {
        hvalues1[i] = __float2half(values1[i]);
    }
    return check_results_impl(hvalues1, values2, numel);
}

float cublas_hgemm(int64_t kM, int64_t kN, int64_t kK, const __half* A,
                   const __half* B, __half* C, bool timeit = false,
                   int warm_up = 5, int iters = 20) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    __half alf = static_cast<__half>(1.);
    __half bet = static_cast<__half>(0.);

    float elapsed = 0.;

    if (timeit) {
        for (int i = 0; i < warm_up; ++i) {
            cublasHgemm(handle, CUBLAS_OP_T /* transb*/, CUBLAS_OP_N, kN, kM,
                        kK, &alf, B, kK, A, kK, &bet, C, kN);
        }
        cudaDeviceSynchronize();

        CudaTimer timer;
        timer.start();
        for (int i = 0; i < iters; ++i) {
            cublasHgemm(handle, CUBLAS_OP_T /* transb*/, CUBLAS_OP_N, kN, kM,
                        kK, &alf, B, kK, A, kK, &bet, C, kN);
        }
        cudaDeviceSynchronize();
        elapsed = timer.stop() / iters;
    } else {
        cublasHgemm(handle, CUBLAS_OP_T /* transb*/, CUBLAS_OP_N, kN, kM, kK,
                    &alf, B, kK, A, kK, &bet, C, kN);
    }
    cudaDeviceSynchronize();

    cublasDestroy(handle);
    return elapsed;
}
