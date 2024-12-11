// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_utils.cuh"
#include "cutlass/copy.cuh"
#include "cutlass/traits_base.cuh"

#include <cute/tensor.hpp>

namespace benchmarks {
namespace cutlass_wrapper {
using namespace cute;

template <typename InType, typename AccType>
__global__ void fa_kernel(const InType* dQ, const InType* dK, const InType* dV,
                          InType* dO) {
    constexpr float softmax_scale = 1.250000e-01f;
}

}  // namespace cutlass_wrapper
}  // namespace benchmarks