// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_utils.hpp"

#include <torch/script.h>

namespace tilefusion::kernels {

template <typename InType,
          typename AccType,                                      //
          typename OutType,                                      //
          typename GIteratorQ, typename SharedQ, typename RegQ,  //
          typename SharedQLoader, typename RegQLoader,           //
          typename GIteratorK, typename SharedK, typename RegK,  //
          typename SharedKLoader, typename RegKLoader,           //
          typename GIteratorV, typename SharedV, typename RegV,  //
          typename SharedVLoader, typename RegVLoader,           //
          typename RegAcc, typename RegAccCast, typename GlobalO, typename RegO,
          typename RegOCast, typename OStorer, typename ConvertAcc,
          typename ConvertO, typename RegVec, typename CopyVec, typename RowMax,
          typename RowSum, typename BroadcastSub, typename BroadcastMul,
          typename BroadcastDiv, typename BlockExp, typename BlockAdd,
          typename VecMax, typename VecAdd, typename VecSub, typename VecMul,
          typename VecExp>
__global__ void ke_flash_attention(const InType* dQ, const InType* dK,
                                   const InType* dV, InType* dO, int kM, int kN,
                                   int kK, int kP, int kTM, int kTN, int kTK,
                                   int kTP);

extern "C" {
__attribute__((visibility("default"))) void flash_attention(
    torch::Tensor& Q, torch::Tensor& K, torch::Tensor& V, torch::Tensor& O,
    int64_t m, int64_t n, int64_t k, int64_t p);
}

}  // namespace tilefusion::kernels
