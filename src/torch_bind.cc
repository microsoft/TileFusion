// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kernels/mod.hpp"

namespace tilefusion {
using namespace tilefusion::kernels;

TORCH_LIBRARY_IMPL(tilefusion, CUDA, m) {
    m.impl("scatter_nd", scatter_op);
    m.impl("flash_attention_fwd", flash_attention_op);
};

TORCH_LIBRARY(tilefusion, m) {
    m.def("scatter_nd(Tensor(a!) data, Tensor updates, Tensor indices) -> ()");
    m.def(
        R"DOC(flash_attention_fwd(
            Tensor(a!) Q,
            Tensor K, Tensor V, Tensor O,
            int m, int n, int k, int p) -> ()
    )DOC");
}
}  // namespace tilefusion
