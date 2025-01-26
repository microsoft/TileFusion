// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cell/sync.hpp"

using namespace tilefusion::cell;

template <typename Element, typename Global, typename Shared, typename Loader,
          const int kRepeat>
__global__ void copy_g2s(const Element* data, Loader& loader) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);

    Global g_tile(data);
    Shared s_tile(buf);

    for (int i = 0; i < kRepeat; ++i) {
        loader(g_tile, s_tile);
        __copy_async();
        __syncthreads();
    }

#if defined DEBUG
    if (threadIdx.x == 0) {
        for (int i = 0; i < kRows * kCols; ++i) {
            printf("%.0f, ", __half2float(buf[i]));

            if ((i + 1) % 16 == 0) printf("\n");
        }
    }
#endif
}
