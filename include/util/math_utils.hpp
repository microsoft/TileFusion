// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_utils.hpp"

namespace tilefusion {

/// @brief Helper function to check if a number is power of 2 at compile time
template <int kN>
consteval bool is_power_of_2() {
    return kN > 0 && (kN & (kN - 1)) == 0;
}

/// @brief Helper function to count trailing zeros
template <int kN>
HOST_DEVICE constexpr int count_trailing_zeros() {
    static_assert(is_power_of_2<kN>(), "kN must be a power of 2");
    int count = 0;
    int temp = kN;
    while ((temp & 1) == 0) {
        temp >>= 1;
        count++;
    }
    return count;
}

/// @brief Helper function to compute division for power of 2
template <int kN>
HOST_DEVICE constexpr int div_pow2(int x) {
    static_assert(is_power_of_2<kN>(), "kN must be a power of 2");
    return x >> count_trailing_zeros<kN>();
}

/// @brief Helper function to compute modulo for power of 2
template <int kN>
HOST_DEVICE constexpr int mod_pow2(int x) {
    static_assert(is_power_of_2<kN>(), "kN must be a power of 2");
    return x & (kN - 1);
}

/// @brief Helper function to compute division and modulo for any number
template <int kN>
HOST_DEVICE constexpr int div_any(int x) {
    return x / kN;
}

/// @brief Helper function to compute modulo for any number
template <int kN>
HOST_DEVICE constexpr int mod_any(int x) {
    return x % kN;
}

/// @brief Select appropriate division/modulo functions based on whether n is
///        power of 2
template <bool kIsPow2, int kN>
struct DivModSelector;

template <int kN>
struct DivModSelector<false, kN> {
    static HOST_DEVICE constexpr int div(int x) { return div_any<kN>(x); }

    static HOST_DEVICE constexpr int mod(int x) { return mod_any<kN>(x); }
};

template <int kN>
struct DivModSelector<true, kN> {
    static HOST_DEVICE constexpr int div(int x) { return div_pow2<kN>(x); }

    static HOST_DEVICE constexpr int mod(int x) { return mod_pow2<kN>(x); }
};
}  // namespace tilefusion
