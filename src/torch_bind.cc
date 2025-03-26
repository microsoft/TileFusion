// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kernel_defs.hpp"
#include "kernel_registry.hpp"
#include "kernels/mod.hpp"

namespace tilefusion {
using namespace tilefusion::kernels;

namespace {

static const std::vector<KernelRegEntry> kernel_registry = []() {
    std::vector<KernelRegEntry> registry;
    for (const auto* kernel : all_kernels) {
        registry.push_back(*kernel);
    }
    return registry;
}();

std::vector<KernelInfo> get_kernel_registry() {
    std::vector<KernelInfo> kernels;
    for (const auto& entry : kernel_registry) {
        kernels.push_back(
            KernelInfo{entry.name, entry.sig,
                       torch::CppFunction(
                           reinterpret_cast<void (*)(void)>(entry.fn_ptr))});
    }
    return kernels;
}

void register_kernels(torch::Library& m, bool is_impl) {
    for (auto&& k : get_kernel_registry()) {
        if (is_impl) {
            m.impl(k.name, std::move(k.fn));
        } else {
            m.def(k.sig);
        }
    }
}
}  // anonymous namespace

TORCH_LIBRARY_IMPL(tilefusion, CUDA, m) { register_kernels(m, true); }

TORCH_LIBRARY(tilefusion, m) { register_kernels(m, false); }
}  // namespace tilefusion
