#pragma once

#include "cuda_utils.hpp"

namespace tilefusion::cell {
/**
 * @brief A swizzle functor.
 */
template <const int kB, const int kM, const int kS>
struct Swizzle {
    static constexpr int Bbits = kB;
    static constexpr int Mbits = kM;
    static constexpr int Sbits = kS;

    DEVICE int32 swizzle(int32 idx) const {
        // | Bbits | Sbits | Mbits |
        // Mbits as mask for the lower bits.
        int32 bs = idx >> Mbits;
        // (b, s) as a 2d coordinate.
        int32 y = bs & ((1 << Sbits) - 1);
        int32 x = bs >> Sbits;

        int32 swizzled_y = x ^ y;

        // Use swizzled_y instead of y and build swizzled idx.
        return (x << (Mbits + Sbits)) | (swizzled_y << Mbits) |
               (idx & ((1 << Mbits) - 1));
    }
};
}  // namespace tilefusion::cell
