// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kernels/kernel_list.hpp"

namespace tilefusion {

TORCH_LIBRARY(tilefusion, m) {
    KernelRegistry::instance().register_with_torch(m);
}

TORCH_LIBRARY_IMPL(tilefusion, CUDA, m) {
    KernelRegistry::instance().register_implementations(m);
}

}  // namespace tilefusion
