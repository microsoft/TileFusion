// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/test_utils.hpp"
#include "types/base.hpp"
#include "types/mod.hpp"

/// @brief Device kernel for testing FP8 operations (must be at global scope)
__global__ void fp8_conversion_kernel(const float* input, void* output_e4m3,
                                      void* output_e5m2, float* result_e4m3,
                                      float* result_e5m2, int size) {
#ifdef CUDA_FP8_AVAILABLE
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        __nv_fp8_e4m3* e4m3_output = static_cast<__nv_fp8_e4m3*>(output_e4m3);
        __nv_fp8_e5m2* e5m2_output = static_cast<__nv_fp8_e5m2*>(output_e5m2);

        // Convert float to FP8
        e4m3_output[idx] = __nv_fp8_e4m3(input[idx]);
        e5m2_output[idx] = __nv_fp8_e5m2(input[idx]);

        // Convert back to float
        result_e4m3[idx] = static_cast<float>(e4m3_output[idx]);
        result_e5m2[idx] = static_cast<float>(e5m2_output[idx]);
    }
#endif
}

namespace tilefusion::testing {

#ifdef CUDA_FP8_AVAILABLE

/// @brief Test basic FP8 construction and conversion
TEST(TestFP8, test_fp8_construction) {
    // Test FP8 E4M3 construction
    __nv_fp8_e4m3 e4m3_val(3.14f);
    float e4m3_back = static_cast<float>(e4m3_val);

    // Test FP8 E5M2 construction
    __nv_fp8_e5m2 e5m2_val(2.71f);
    float e5m2_back = static_cast<float>(e5m2_val);

    // Check that we can round-trip (with some precision loss expected)
    EXPECT_NEAR(e4m3_back, 3.14f, 0.1f);  // E4M3 has limited precision
    EXPECT_NEAR(e5m2_back, 2.71f, 0.1f);  // E5M2 has different precision
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
    float a_float = static_cast<float>(a);
    float b_float = static_cast<float>(b);

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

/// @brief Test precision and range characteristics
TEST(TestFP8, test_fp8_precision_ranges) {
    // Test small values
    float small_val = 0.125f;
    __nv_fp8_e4m3 e4m3_small(small_val);
    __nv_fp8_e5m2 e5m2_small(small_val);

    EXPECT_NEAR(static_cast<float>(e4m3_small), small_val, 0.01f);
    EXPECT_NEAR(static_cast<float>(e5m2_small), small_val, 0.01f);

    // Test larger values (within FP8 range)
    float large_val = 8.0f;
    __nv_fp8_e4m3 e4m3_large(large_val);
    __nv_fp8_e5m2 e5m2_large(large_val);

    EXPECT_NEAR(static_cast<float>(e4m3_large), large_val, 0.5f);
    EXPECT_NEAR(static_cast<float>(e5m2_large), large_val, 0.5f);
}

/// @brief Test FP8 operations on device
TEST(TestFP8, test_fp8_device_operations) {
    const int size = 1024;
    const int bytes = size * sizeof(float);

    // Host data
    std::vector<float> h_input(size);
    std::vector<float> h_output_e4m3(size);
    std::vector<float> h_output_e5m2(size);

    // Initialize input with test values
    for (int i = 0; i < size; ++i) {
        h_input[i] = static_cast<float>(i % 100) * 0.1f;  // 0.0 to 9.9
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

    // Verify results
    for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(h_output_e4m3[i], h_input[i], 0.5f)
            << "E4M3 mismatch at index " << i;
        EXPECT_NEAR(h_output_e5m2[i], h_input[i], 0.5f)
            << "E5M2 mismatch at index " << i;
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

    // Check compute capability
    int major = prop.major;
    int minor = prop.minor;
    int compute_capability = major * 10 + minor;

    // Log information (will show in test output)
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
