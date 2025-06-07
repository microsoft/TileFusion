// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace tilefusion::cell::copy {
enum class WarpReuse {
  // data are evenly partitioned to be loaded by warps.
  kCont = 0,          // all warps continuously load data, no reuse
  kRowReuseCont = 1,  // Row-wise even reuse, warps in the same row
                      // repeatedly load the same data
  kColReuseCont = 2   // Column-wise even reuse, warps in the same column
                      // repeatedly load the same data
};
}  // namespace tilefusion::cell::copy
