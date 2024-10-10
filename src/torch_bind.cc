// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kernels/mod.hpp"

#include <torch/script.h>

namespace tilefusion {
using namespace tilefusion::kernels;

TORCH_LIBRARY(tilefusion, t) {
    t.def("scatter_nd", &custom_scatter_op);
    t.def("flash_attention_fwd", &custom_flash_attention_op);
};

}  // namespace tilefusion
