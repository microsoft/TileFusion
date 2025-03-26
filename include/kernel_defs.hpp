#pragma once
#include "kernel_registry.hpp"

namespace tilefusion {

// Define all kernels
DEFINE_KERNEL(
    scatter_nd,
    "scatter_nd(Tensor(a!) data, Tensor updates, Tensor indices) -> ()");

DEFINE_KERNEL(flash_attention,
              R"DOC(flash_attention(
        Tensor(a!) Q,
        Tensor K, Tensor V, Tensor O,
        int m, int n, int k, int p) -> ()
    )DOC");

// Array of all kernel entries
static const KernelRegEntry* all_kernels[] = {&scatter_nd_kernel,
                                              &flash_attention_kernel};

}  // namespace tilefusion
