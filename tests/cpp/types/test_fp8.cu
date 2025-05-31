// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/test_utils.hpp"
#include "types/base.hpp"
#include "types/mod.hpp"

using namespace tilefusion;

/// @brief Device kernel for testing FP8 operations
__global__ void fp8_conversion_kernel(const float* input, void* output_e4m3,
                                      void* output_e5m2, float* result_e4m3,
                                      float* result_e5m2, int size) {
#ifdef CUDA_FP8_AVAILABLE
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        __nv_fp8_e4m3* e4m3_output = static_cast<__nv_fp8_e4m3*>(output_e4m3);
        __nv_fp8_e5m2* e5m2_output = static_cast<__nv_fp8_e5m2*>(output_e5m2);

        // Convert float to FP8
        e4m3_output[idx] = from_float<__nv_fp8_e4m3>(input[idx]);
        e5m2_output[idx] = from_float<__nv_fp8_e5m2>(input[idx]);

        // Convert back to float
        result_e4m3[idx] = to_float(e4m3_output[idx]);
        result_e5m2[idx] = to_float(e5m2_output[idx]);
    }
#endif
}

namespace tilefusion::testing {

#ifdef CUDA_FP8_AVAILABLE

/// @brief Test basic FP8 construction and conversion
TEST(TestFP8, test_fp8_construction) {
    // Test with values that are exactly representable in FP8
    {
        // Test simple powers of 2 and small integers
        // usually exactly representable
        __nv_fp8_e4m3 e4m3_val = from_float<__nv_fp8_e4m3>(2.0f);
        float e4m3_back = to_float(e4m3_val);
        printf("E4M3 (2.0): %f\n", e4m3_back);
        EXPECT_EQ(e4m3_back, 2.0f);  // Should be exact

        __nv_fp8_e5m2 e5m2_val(4.0f);
        float e5m2_back = static_cast<float>(e5m2_val);
        printf("E5M2 (4.0): %f\n", e5m2_back);
        EXPECT_EQ(e5m2_back, 4.0f);  // Should be exact
    }

    // Test edge cases
    {
        __nv_fp8_e4m3 e4m3_zero(0.0f);
        __nv_fp8_e5m2 e5m2_zero(0.0f);
        EXPECT_EQ(static_cast<float>(e4m3_zero), 0.0f);
        EXPECT_EQ(static_cast<float>(e5m2_zero), 0.0f);

        __nv_fp8_e4m3 e4m3_one(1.0f);
        __nv_fp8_e5m2 e5m2_one(1.0f);
        EXPECT_EQ(static_cast<float>(e4m3_one), 1.0f);
        EXPECT_EQ(static_cast<float>(e5m2_one), 1.0f);
    }
}

/// @brief Test FP8 precision characteristics and ranges
TEST(TestFP8, test_fp8_precision_characteristics) {
    {  // Test small values in the precise range
        float test_val = 0.5f;
        __nv_fp8_e4m3 e4m3_val(test_val);
        __nv_fp8_e5m2 e5m2_val(test_val);

        printf("Small value (0.5): E4M3=%f, E5M2=%f\n", to_float(e4m3_val),
               to_float(e5m2_val));

        // Use relative tolerance
        EXPECT_NEAR(to_float(e4m3_val), test_val, test_val * 0.1f);
        EXPECT_NEAR(to_float(e5m2_val), test_val, test_val * 0.1f);
    }

    {  // Test medium values (where precision loss starts)
        float test_val = 3.0f;  // Use a value more likely to be representable
        __nv_fp8_e4m3 e4m3_val(test_val);
        __nv_fp8_e5m2 e5m2_val(test_val);

        printf("Medium value (3.0): E4M3=%f, E5M2=%f\n", to_float(e4m3_val),
               to_float(e5m2_val));

        // Use relative tolerance
        EXPECT_NEAR(to_float(e4m3_val), test_val, test_val * 0.15f);
        EXPECT_NEAR(to_float(e5m2_val), test_val, test_val * 0.25f);
    }

    {  // Test larger values (significant quantization)
        float test_val = 9.0f;
        __nv_fp8_e4m3 e4m3_val = from_float<__nv_fp8_e4m3>(test_val);
        __nv_fp8_e5m2 e5m2_val = from_float<__nv_fp8_e5m2>(test_val);

        printf("Large value (8.0): E4M3=%f, E5M2=%f\n", to_float(e4m3_val),
               to_float(e5m2_val));

        // Much larger relative tolerance for larger values
        EXPECT_NEAR(to_float(e4m3_val), test_val, test_val * 0.25f);
        EXPECT_NEAR(to_float(e5m2_val), test_val, test_val * 0.5f);
    }
}

/// @brief Test that conversion functions work without crashing
TEST(TestFP8, test_fp8_conversion_safety) {
    // Test a diverse range of values to ensure no crashes
    std::vector<float> test_values = {
        // Small values
        0.0f, 0.0625f, 0.125f, 0.1875f, 0.25f, 0.375f, 0.5f, 0.625f, 0.75f,
        0.875f,
        // Around 1.0
        1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
        // Small integers and fractions
        2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
        // Medium values
        4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
        // Larger values (testing FP8 range limits)
        8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 14.0f, 16.0f, 20.0f, 24.0f, 28.0f};

    for (float val : test_values) {
        // Just test that conversion works without crashing
        __nv_fp8_e4m3 e4m3_val = from_float<__nv_fp8_e4m3>(val);
        __nv_fp8_e5m2 e5m2_val = from_float<__nv_fp8_e5m2>(val);

        float e4m3_back = to_float(e4m3_val);
        float e5m2_back = to_float(e5m2_val);

        // Basic sanity check - result should be finite
        EXPECT_TRUE(std::isfinite(e4m3_back))
            << "E4M3 conversion of " << val << " produced non-finite result";
        EXPECT_TRUE(std::isfinite(e5m2_back))
            << "E5M2 conversion of " << val << " produced non-finite result";

        printf("Value %f -> E4M3: %f, E5M2: %f\n", val, e4m3_back, e5m2_back);
    }
}

/// @brief Test TileFusion utility functions
TEST(TestFP8, test_fp8_utility_functions) {
    float original = 1.5f;

    // Test E4M3 conversions
    __nv_fp8_e4m3 e4m3_val = from_float<__nv_fp8_e4m3>(original);
    float e4m3_result = to_float(e4m3_val);
    EXPECT_NEAR(e4m3_result, original, 0.01f);

    // Test E5M2 conversions
    __nv_fp8_e5m2 e5m2_val = from_float<__nv_fp8_e5m2>(original);
    float e5m2_result = to_float(e5m2_val);
    EXPECT_NEAR(e5m2_result, original, 0.01f);
}

/// @brief Test FP8 arithmetic operations (through float conversion)
TEST(TestFP8, test_fp8_arithmetic) {
    __nv_fp8_e4m3 a(3.0f);
    __nv_fp8_e5m2 b(2.0f);

    // Convert to float for computation
    float a_float = to_float(a);
    float b_float = to_float(b);

    // Perform computation in float
    float sum = a_float + b_float;
    float product = a_float * b_float;

    // Convert back to FP8
    __nv_fp8_e4m3 sum_e4m3(sum);
    __nv_fp8_e5m2 product_e5m2(product);

    EXPECT_NEAR(static_cast<float>(sum_e4m3), 5.0f, 0.1f);
    EXPECT_NEAR(static_cast<float>(product_e5m2), 6.0f, 0.1f);
}

/// @brief Test FP8 type traits
TEST(TestFP8, test_fp8_traits) {
    // Test that FP8 types satisfy BaseType concept
    static_assert(BaseType<__nv_fp8_e4m3>);
    static_assert(BaseType<__nv_fp8_e5m2>);

    // Test that FP8 types satisfy Fp8Type concept
    static_assert(Fp8Type<__nv_fp8_e4m3>);
    static_assert(Fp8Type<__nv_fp8_e5m2>);

    // Test that other types don't satisfy Fp8Type concept
    static_assert(!Fp8Type<float>);
    static_assert(!Fp8Type<__half>);
    static_assert(!Fp8Type<__bfloat16>);
}

/// @brief Test FP8 operations on device
TEST(TestFP8, test_fp8_device_operations) {
    const int size = 64;
    const int bytes = size * sizeof(float);

    std::vector<float> h_input(size);
    std::vector<float> h_output_e4m3(size);
    std::vector<float> h_output_e5m2(size);

    // Initialize input with better test values for FP8
    std::vector<float> good_fp8_values = {
        // Small precise values
        0.0f, 0.125f, 0.25f, 0.375f, 0.5f, 0.625f, 0.75f, 0.875f,
        // Around 1.0
        1.0f, 1.25f, 1.5f, 1.75f,
        // Small integers and key fractions
        2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f, 5.0f, 6.0f, 7.0f, 8.0f,
        // Larger values within FP8 range
        9.0f, 10.0f, 12.0f, 14.0f, 16.0f};

    for (int i = 0; i < size; ++i) {
        h_input[i] = good_fp8_values[i % good_fp8_values.size()];
    }

    // Device memory
    float *d_input, *d_result_e4m3, *d_result_e5m2;
    __nv_fp8_e4m3* d_fp8_e4m3;
    __nv_fp8_e5m2* d_fp8_e5m2;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_result_e4m3, bytes);
    cudaMalloc(&d_result_e5m2, bytes);
    cudaMalloc(&d_fp8_e4m3, size * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_fp8_e5m2, size * sizeof(__nv_fp8_e5m2));

    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    fp8_conversion_kernel<<<grid, block>>>(
        d_input, static_cast<void*>(d_fp8_e4m3), static_cast<void*>(d_fp8_e5m2),
        d_result_e4m3, d_result_e5m2, size);

    // Copy results back
    cudaMemcpy(h_output_e4m3.data(), d_result_e4m3, bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_e5m2.data(), d_result_e5m2, bytes,
               cudaMemcpyDeviceToHost);

    // Verify results with appropriate tolerances
    for (int i = 0; i < size; ++i) {
        float input_val = h_input[i];
        float e4m3_result = h_output_e4m3[i];
        float e5m2_result = h_output_e5m2[i];

        // Use relative tolerance that scales with the input value
        float e4m3_tolerance = std::max(0.1f, input_val * 0.2f);
        float e5m2_tolerance = std::max(0.1f, input_val * 0.3f);

        EXPECT_NEAR(e4m3_result, input_val, e4m3_tolerance)
            << "E4M3 mismatch at index " << i << " (input=" << input_val << ")";
        EXPECT_NEAR(e5m2_result, input_val, e5m2_tolerance)
            << "E5M2 mismatch at index " << i << " (input=" << input_val << ")";
    }

    cudaFree(d_input);
    cudaFree(d_result_e4m3);
    cudaFree(d_result_e5m2);
    cudaFree(d_fp8_e4m3);
    cudaFree(d_fp8_e5m2);
}

#else  // !CUDA_FP8_AVAILABLE

/// @brief Test that runs when FP8 is not available
TEST(TestFP8, test_fp8_not_available) {
    // This test ensures that the build works even when FP8 is not available
    GTEST_SKIP() << "FP8 support not available - requires Ada Lovelace or "
                    "Hopper GPU with CUDA 11.8+";
}

#endif  // CUDA_FP8_AVAILABLE

/// @brief Test hardware detection (runs regardless of FP8 support)
TEST(TestFP8, test_hardware_detection) {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    EXPECT_EQ(err, cudaSuccess);

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    EXPECT_EQ(err, cudaSuccess);

    int major = prop.major;
    int minor = prop.minor;
    int compute_capability = major * 10 + minor;

    LOG(INFO) << "GPU: " << prop.name;
    LOG(INFO) << "Compute Capability: " << major << "." << minor;

    if (compute_capability >= 89) {
        LOG(INFO) << "FP8 hardware support: YES";
        if (major == 8 && minor == 9) {
            LOG(INFO) << "Architecture: Ada Lovelace";
        } else if (major >= 9) {
            LOG(INFO) << "Architecture: Hopper";
        }
    } else {
        LOG(INFO)
            << "FP8 hardware support: NO (requires compute capability 8.9+)";
    }

#ifdef CUDA_FP8_AVAILABLE
    LOG(INFO) << "FP8 compile-time support: YES";
#else
    LOG(INFO) << "FP8 compile-time support: NO";
#endif

    // Test always passes - this is just for information
    EXPECT_TRUE(true);
}

}  // namespace tilefusion::testing
