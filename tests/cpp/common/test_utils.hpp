// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "config.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

namespace tilefusion::testing {

template <typename T>
void assert_equal(const T* v1, const T* v2, int64_t numel, float epsilon);

float rand_float(float min = 0.f, float max = 1.f);

}  // namespace tilefusion::testing
