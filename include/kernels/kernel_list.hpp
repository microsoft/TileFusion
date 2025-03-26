// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "kernel_registry.hpp"
#include "kernels/mod.hpp"

namespace tilefusion {
namespace kernels {

// Macro for kernel registration
#define REGISTER_TILEFUSION_KERNEL(name, schema, func)                        \
    namespace {                                                               \
    [[maybe_unused]] static const auto name##_registered = []() {             \
        tilefusion::KernelRegistry::instance().register_kernel(#name, schema, \
                                                               func);         \
        return true;                                                          \
    }();                                                                      \
    }

// All kernel registrations
#define REGISTER_ALL_KERNELS()                                               \
    REGISTER_TILEFUSION_KERNEL(                                              \
        scatter_nd,                                                          \
        "scatter_nd(Tensor data, Tensor(a!) updates, Tensor indices) -> ()", \
        tilefusion::kernels::scatter_nd);                                    \
                                                                             \
    REGISTER_TILEFUSION_KERNEL(                                              \
        flash_attention,                                                     \
        "flash_attention(Tensor Q, Tensor K, Tensor V, Tensor(a!) O, "       \
        "int m, int n, int k, int p) -> ()",                                 \
        tilefusion::kernels::flash_attention)

// Add new kernels by extending REGISTER_ALL_KERNELS macro

}  // namespace kernels
}  // namespace tilefusion
