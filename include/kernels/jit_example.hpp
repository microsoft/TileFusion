// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "kernel_registry.hpp"

#include <ATen/ATen.h>
#include <torch/script.h>

namespace tilefusion::kernels {

/**
 * Example function demonstrating the use of JIT compilation for CUDA kernels.
 * This function takes two input tensors and an output tensor, and performs
 * element-wise addition using a JIT-compiled CUDA kernel.
 *
 * @param input_a First input tensor
 * @param input_b Second input tensor
 * @param output Output tensor that will contain the result of input_a + input_b
 */
TILEFUSION_EXPORT void jit_add(at::Tensor input_a, at::Tensor input_b,
                               at::Tensor output);

}  // namespace tilefusion::kernels
