// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace tilefusion::cell::copy {

/**
 * @brief Vectorize a data type.
 *
 * @tparam Element Data type.
 * @tparam kVecNums Number of vecotrized elements.
 */
template <typename Element, const int kVecNums>
struct Vectorize {
    using UnVecType = Element;
    using VecType = Element;
    static constexpr int vectorize_nums = kVecNums;

    /**
     * @brief Copy data from unvectorized to vectorized.
     *
     * @param src Source data.
     * @param dst Destination data.
     */
    DEVICE void operator()(const UnVecType* src, UnVecType* dst) {
        const VecType* src_vec = reinterpret_cast<const VecType*>(src);
        VecType* dst_vec = reinterpret_cast<VecType*>(dst);
        *dst_vec = *src_vec;
    }
};

template <>
struct Vectorize<__half, 2> {
    using UnVecType = __half;
    using VecType = __half2;
    static constexpr int vectorize_nums = 2;
    static constexpr int vectorize_bits = 32;

    DEVICE void operator()(const __half* src, __half* dst) {
        const __half2* src_vec = reinterpret_cast<const __half2*>(src);
        __half2* dst_vec = reinterpret_cast<__half2*>(dst);
        *dst_vec = *src_vec;
    }
};

template <>
struct Vectorize<float, 2> {
    using UnVecType = float;
    using VecType = float2;
    static constexpr int vectorize_nums = 2;
    static constexpr int vectorize_bits = 64;

    DEVICE void operator()(const float* src, float* dst) {
        const float2* src_vec = reinterpret_cast<const float2*>(src);
        float2* dst_vec = reinterpret_cast<float2*>(dst);
        *dst_vec = *src_vec;
    }
};

}  // namespace tilefusion::cell::copy
