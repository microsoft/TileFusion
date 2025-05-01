// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "traits/base.hpp"

namespace tilefusion::jit {
template <typename DType>
static constexpr const char* get_type_string() {
    if constexpr (std::is_same_v<DType, float>) {
        return "float";
    } else if constexpr (std::is_same_v<DType, double>) {
        return "double";
    } else if constexpr (std::is_same_v<DType, int>) {
        return "int";
    } else if constexpr (std::is_same_v<DType, __half>) {
        return "__half";
    } else if constexpr (std::is_same_v<DType, __bfloat16>) {
        return "__bfloat16";
    } else {
        // Makes the assertion dependent on the template parameter
        // Only triggers when an unsupported type is actually used.
        static_assert(sizeof(DType) == 0, "Unsupported data type");
    }
}
}  // namespace tilefusion::jit
