// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kernel_registry.hpp"
#include "kernels/flash_attn.hpp"
#include "kernels/scatter_nd.hpp"

#include <ATen/ATen.h>
#include <torch/script.h>

namespace tilefusion {

// Register scatter_nd kernel
REGISTER_KERNEL(
    scatter_nd,
    "scatter_nd(Tensor data, Tensor(a!) updates, Tensor indices) -> ()",
    tilefusion::kernels::scatter_nd);

// Register flash attention kernel
REGISTER_KERNEL(flash_attention,
                "flash_attention(Tensor Q, Tensor K, Tensor V, Tensor(a!) O, "
                "int m, int n, int k, int p) -> ()",
                tilefusion::kernels::flash_attention);

TORCH_LIBRARY_IMPL(tilefusion, CUDA, m) {
    const auto& registry = KernelRegistry::instance();
    for (const auto& [name, info] : registry.getAllKernels()) {
        if (name == "scatter_nd") {
            using Func =
                void (*)(at::Tensor&, const at::Tensor&, const at::Tensor&);
            auto func = reinterpret_cast<Func>(info.fn_ptr);
            m.impl(name.c_str(), func);
        } else if (name == "flash_attention") {
            using Func =
                void (*)(at::Tensor&, at::Tensor, at::Tensor, at::Tensor,
                         int64_t, int64_t, int64_t, int64_t);
            auto func = reinterpret_cast<Func>(info.fn_ptr);
            m.impl(name.c_str(), func);
        }
    }
}

TORCH_LIBRARY(tilefusion, m) {
    const auto& registry = KernelRegistry::instance();
    for (const auto& [name, info] : registry.getAllKernels()) {
        m.def(info.schema.c_str());
    }
}

}  // namespace tilefusion
