// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "config.hpp"

namespace tilefusion::cell::copy {
enum class CopyInst {
    kLoadMat = 0,   // ldmatrix for loading data from shared memory to register.
    kStoreMat = 1,  // stmatrix for storing data from register to shared memory.
    kLoadShared32 = 2,  // ldsm32 for loading 32-bit data from shared memory.
    kLoadShared128 = 3  // ldsm128 for loading 128-bit data from shared memory.
};

enum class WarpReuse {
    // data are evenly partitioned to be loaded by warps.
    kCont = 0,          // all warps continuously load data, no reuse
    kRowReuseCont = 1,  // Row-wise even reuse, warps in the same row
                        // repeatedly load the same data
    kColReuseCont = 2   // Column-wise even reuse, warps in the same column
                        // repeatedly load the same data
};
}  // namespace tilefusion::cell::copy
