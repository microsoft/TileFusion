// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_utils.hpp"

#include <cassert>

namespace tilefusion::cell {
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
    HOST_DEVICE int apply(int idx) const {
        // | Bbits | Sbits | Mbits |
        // Mbits as mask for the lower bits.

        assert(idx < (1 << (Bbits + Mbits + Sbits)));

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
 * @brief Swizzle Layout.
 *
 * @tparam Layout_ The layout to swizzle.
 * @tparam kB The number of bits for B.
 * @tparam kM The number of bits for M.
 * @tparam kS The number of bits for S.
 */
template <typename Layout_, const int kB = 3, const int kM = 3,
          const int kS = 3>
struct SwizzleLayout {
    static constexpr int Bbits = kB;
    static constexpr int Mbits = kM;
    static constexpr int Sbits = kS;

    using Layout = Layout_;
    using Swizzle = Swizzle<Bbits, Mbits, Sbits>;

    /**
     * @brief Apply the swizzle in a layout.
     *
     * @param x Row dimension index, with a total of 2^B rows.
     * @param y Column dimension index, with a total of 2^S * 2^M columns.
     */
    HOST_DEVICE auto operator()(int x, int y) const {
        int idx = (x << (Mbits + Sbits)) | y;

        assert(idx < (1 << (Bbits + Mbits + Sbits)));

        int swizzled_idx = swizzle_.apply(idx);
        int swizzled_x = swizzled_idx >> (Mbits + Sbits);
        int swizzled_y = swizzled_idx & ((1 << (Mbits + Sbits)) - 1);
        return Layout{}(swizzled_x, swizzled_y);
    }

  private:
    Swizzle swizzle_;
};

}  // namespace tilefusion::cell