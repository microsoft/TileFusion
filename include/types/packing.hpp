#pragma once

#include "cuda_utils.hpp"

#include <cutlass/numeric_types.h>

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
struct Packing<cutlass::half_t, 2> {
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
