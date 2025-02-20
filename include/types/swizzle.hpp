// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_utils.hpp"
#include "types/layout.hpp"

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
     * @brief Applies the swizzle function to permute a 1-D index.
     *
     * @param idx The 1-D index within the swizzle space of 2^B * 2^S * 2^M
     *            elements.
     * @return The permuted (swizzled) index.
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
 * @brief `SwizzledLayout` combines a `Layout` with a swizzle function.
 *        The `Layout` maps a 2-D coordinate to a 1-D index, while the swizzle
 *        function permutes these 1-D indices. Essentially, `SwizzledLayout`
 *        implements a composed function where a given 2-D coordinate is
 *        first translated into an intermediate 1-D index by the layout
 *        function. The swizzle function then permutes this intermediate 1-D
 *        index into a new 1-D index, which is finally interpreted again as a
 *        2-D coordinate in the space defined by `Layout`.
 * @tparam Layout The `Layout` that defines a function mapping a 2-D coordinate
 *                to a 1-D index.
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

    static_assert(Layout::kRows == (1 << Bbits),
                  "The number of rows in the layout should be 2^B.");
    static_assert(Layout::kCols == (1 << (Mbits + Sbits)),
                  "The number of columns in the layout should be 2^S * 2^M.");

    /**
     * @brief Compose the swizzle function with the layout function.
     *
     * @param x The row index, with a total of 2^B rows.
     * @param y The column index, with a total of 2^S * 2^M columns.
     * @return The swizzled index after applying the layout function.
     */
    HOST_DEVICE auto operator()(int x, int y) const {
        int idx = (x << (Mbits + Sbits)) | y;

        int swizzled_idx = swizzle_(idx);
        int swizzled_x = swizzled_idx >> (Mbits + Sbits);
        int swizzled_y = swizzled_idx & ((1 << (Mbits + Sbits)) - 1);
        return layout_(swizzled_x, swizzled_y);
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

    static_assert(Layout::kRows == (1 << (Mbits + Sbits)),
                  "The number of rows in the layout should be 2^S * 2^M.");
    static_assert(Layout::kCols == (1 << Bbits),
                  "The number of columns in the layout should be 2^B.");

    /**
     * @brief Compose the swizzle function with the layout function.
     *
     * @param x The row index, with a total of 2^B rows.
     * @param y The column index, with a total of 2^S * 2^M columns.
     * @return The swizzled index after applying the layout function.
     */
    HOST_DEVICE auto operator()(int x, int y) const {
        int idx = (y << (Bbits + Mbits)) | x;

        int swizzled_idx = swizzle_(idx);
        int swizzled_y = swizzled_idx >> (Mbits + Sbits);
        int swizzled_x = swizzled_idx & ((1 << (Mbits + Sbits)) - 1);
        return layout_(swizzled_x, swizzled_y);
    }

  private:
    Swizzle swizzle_;
    Layout layout_;
};

}  // namespace tilefusion::cell
