#pragma once

#include "cuda_utils.hpp"

namespace tilefusion::cell {

template <typename DType>
struct Packing;

template <>
struct Packing<__half> {
    using PackedType = int;
};

template <>
struct Packing<float> {
    using PackedType = int2;
};

}  // namespace tilefusion::cell