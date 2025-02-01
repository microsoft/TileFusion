// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_utils.hpp"
#include "layout.hpp"

#include <cassert>

namespace tilefusion::cell {
namespace tl = tile_layout;

/**
 * @brief A swizzle functor.
 * A Swizzle can handle 2^B * 2^S * 2^M elements.
 */
template <const int kB, const int kM, const int kS>
struct Swizzle {
    static constexpr int Bbits = kB;
    static constexpr int Mbits = kM;
    static constexpr int Sbits = kS;
    /**
     * @brief Apply the swizzle to an index.
     *
     * @param idx The index in a swizzle chunk of 2^B * 2^S * 2^M elements.
     * @return The swizzled index.
     */
    HOST_DEVICE int operator()(int idx) const {
        // | Bbits | Sbits | Mbits |
        // Mbits as mask for the lower bits.

        int bs = idx >> Mbits;
        // (b, s) as a 2d coordinate.
        int y = bs & ((1 << Sbits) - 1);
        int x = bs >> Sbits;

        int swizzled_y = x ^ y;

        // Use swizzled_y instead of y and build swizzled idx.
        return (x << (Mbits + Sbits)) | (swizzled_y << Mbits) |
               (idx & ((1 << Mbits) - 1));
    }
};

/**
 * @brief `SwizzledLayout` accepts a `Layout` and parameters for the swizzle
 *        function. The `Layout` defines a function that maps a 2-D coordinate
 *        to a 1-D index, while the swizzle function defines a permutation
 *        function that permutes the 1-D indices. In summary, `SwizzledLayout`
 *        implements a composed function such that, given a 2-D coordinate `x`,
 *        the layout function translates it into an intermediate 1-D index
 *        `x'`. The specified swizzle function then permutes `x'` into a new
 *        1-D index `y`, which is finally interpreted as a 2-D coordinate in
 *        the 2-D space defined by `Layout`.
 *        NOTE: Elements in the 1-D swizzle space can be larger than the
 *              elements in the 2-D space defined by `Layout`.
 *
 * @tparam Layout The `Layout` defines a function that maps a 2-D coordinate
 *                  to a 1-D index.
 * @tparam kB The number of bits for B.
 * @tparam kM The number of bits for M.
 * @tparam kS The number of bits for S.
 * @tparam kType The type of the Layout.
 */
template <typename Layout, const int kB, const int kM, const int kS,
          const tl::Layout kType = Layout::kType>
struct SwizzledLayout;

template <typename Layout_, const int kB, const int kM, const int kS>
struct SwizzledLayout<Layout_, kB, kM, kS, tl::Layout::kRowMajor> {
    static constexpr int Bbits = kB;
    static constexpr int Mbits = kM;
    static constexpr int Sbits = kS;

    using Layout = Layout_;
    using Swizzle = Swizzle<Bbits, Mbits, Sbits>;

    static_assert(
        (1 << (Bbits + Mbits + Sbits)) >= Layout::kNumel,
        "The number of elements in the swizzled space must be greater than or "
        "equal to the number of elements in the layout space.");
    static_assert((1 << (Mbits + Sbits)) % Layout::kCols == 0,
                  "The number of columns in the swizzle space must be a "
                  "multiple of the number of columns of the given Layout.");

    /**
     * @brief Apply the swizzle function.
     *
     * @param x the row index.
     * @param y the col index.
     */
    HOST_DEVICE int operator()(int x, int y) const {
        int swizzled_idx = swizzle_(layout_(x, y));

        return layout_(swizzled_idx / Layout::kCols,
                       swizzled_idx % Layout::kCols);
    }

  private:
    Swizzle swizzle_;
    Layout layout_;
};

template <typename Layout_, const int kB, const int kM, const int kS>
struct SwizzledLayout<Layout_, kB, kM, kS, tl::Layout::kColMajor> {
    static constexpr int Bbits = kB;
    static constexpr int Mbits = kM;
    static constexpr int Sbits = kS;

    using Layout = Layout_;
    using Swizzle = Swizzle<Bbits, Mbits, Sbits>;

    static_assert(
        (1 << (Bbits + Mbits + Sbits)) >= Layout::kNumel,
        "The number of elements in the swizzled space must be greater than or "
        "equal to the number of elements in the layout space.");
    static_assert((1 << (Mbits + Sbits)) % Layout::kCols == 0,
                  "The number of columns in the swizzle space must be a "
                  "multiple of the number of columns of the given Layout.");

    /**
     * @brief Apply the swizzle function.
     *
     * @param x the row index.
     * @param y the col index.
     */
    HOST_DEVICE int operator()(int x, int y) const {
        int swizzled_idx = swizzle_(layout_(x, y));

        return layout_(swizzled_idx % Layout::kRows,
                       swizzled_idx / Layout::kRows);
    }

  private:
    Swizzle swizzle_;
    Layout layout_;
};
}  // namespace tilefusion::cell
