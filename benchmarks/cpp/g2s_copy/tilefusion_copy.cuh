// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cell/copy/sync.hpp"

using namespace tilefusion::cell;

template <typename Element, typename Global, typename Shared, typename Loader,
          typename Storer, const int kRepeat>
__global__ void g2s_data_transfer(const Element* src_ptr, Element* dst_ptr,
                                  Loader& loader, Storer& storer) {
  extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
  auto* buf = reinterpret_cast<Element*>(buf_);

  Global src(src_ptr);
  Shared inter(buf);
  Global dst(dst_ptr);  // global memory tile

  for (int i = 0; i < kRepeat; ++i) {
    loader(src, inter);
    copy::__copy_async();
    __syncthreads();

    storer(inter, dst);
    __syncthreads();
  }
}
