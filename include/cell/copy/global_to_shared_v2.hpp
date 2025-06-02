// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "cell/copy/mod.hpp"
#include "types/mod.hpp"

namespace tilefusion::cell::copy {
using namespace atom;
namespace tl = tile_layout;

/// @brief The thread-block level API that cooperatively transfers a data tile
///        from global memory to shared memory by all the threads within a
///        thread block.
template <typename Shared_, typename WarpLayout_>
struct GlobalToSharedLoaderV2 {
  using Shared = Shared_;
  using DType = Shared::DType;
  using WarpLayout = WarpLayout_;

  // NOTE: The WarpShape calculated here is for the warp reuse mode `kCont`.
  // If you use a different mode, update the WarpShape accordingly.
  static_assert((Shared::kRows % WarpLayout ::kRows == 0) &&
                    (Shared::kCols % WarpLayout::kCols == 0),
                "The shape of SharedTile must be divisible by the shape of "
                "WarpLayout.");

  template <typename Global>
  DEVICE void operator()(const Global& src, Shared& dst) {}
};

template <typename Shared_, typename WarpLayout_>
struct SharedToGlobalStorerV2 {
  using Shared = Shared_;
  using DType = Shared::DType;
  using WarpLayout = WarpLayout_;

  static const WarpReuse kMode = WarpReuse::kCont;  // warp reuse mode

  template <typename Global>
  DEVICE void operator()(const Shared& src_, Global& dst_) {}
};

}  // namespace tilefusion::cell::copy
