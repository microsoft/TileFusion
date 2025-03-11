#pragma once

#include "cuda_utils.hpp"

namespace tilefusion::cell {

template <typename DType, const int kNums>
struct Packing;

template <>
struct Packing<__half, 2> {
    static constexpr int kDateBytes = 2;
    static constexpr int kPackedBytes = 4;
    using PackedType = int;
};

template <>
struct Packing<float, 2> {
    static constexpr int kDateBytes = 4;
    static constexpr int kPackedBytes = 4;
    using PackedType = int2;
};

}  // namespace tilefusion::cell
