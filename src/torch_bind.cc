// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kernel_registry.hpp"
#include "kernels/scatter_nd.hpp"

#include <ATen/ATen.h>
#include <torch/script.h>

namespace tilefusion {

// Register kernel
REGISTER_KERNEL(
    scatter_nd,
    "scatter_nd(Tensor data, Tensor(a!) updates, Tensor indices) -> ()",
    kernels::scatter_nd);

// PyTorch registration
TORCH_LIBRARY(tilefusion, m) {
    KernelRegistry::instance().register_with_torch(m);
}

TORCH_LIBRARY_IMPL(tilefusion, CUDA, m) {
    KernelRegistry::instance().register_implementations(m);
}

}  // namespace tilefusion
