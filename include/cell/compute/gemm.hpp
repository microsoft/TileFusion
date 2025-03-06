// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_utils.hpp"
#include "traits/base.hpp"
#include "types/layout.hpp"
#include "types/tile_shape.hpp"

namespace tilefusion::cell::compute {
namespace tl = tile_layout;

namespace detail {

template <typename InType, typename AccType>
struct TiledMma;

template <>
struct TiledMma<__half, float> {
    DEVICE void operator()(const __half* ra, const __half* rb, float* rc) {
        const uint32_t* A = reinterpret_cast<const uint32_t*>(ra);
        const uint32_t* B = reinterpret_cast<const uint32_t*>(rb);
        float* C = static_cast<float*>(rc);

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,  %1,  %2,  %3},"
            "{%4,  %5,  %6,  %7},"
            "{%8,  %9},"
            "{%10, %11, %12, %13};\n"
            : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[2]),
              "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,  %1,  %2,  %3},"
            "{%4,  %5,  %6,  %7},"
            "{%8,  %9},"
            "{%10, %11, %12, %13};\n"
            : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[1]), "r"(B[3]),
              "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
    }
};

template <>
struct TiledMma<cutlass::half_t, float> {
    DEVICE void operator()(const cutlass::half_t* ra, const cutlass::half_t* rb,
                           float* rc) {
        const uint32_t* A = reinterpret_cast<const uint32_t*>(ra);
        const uint32_t* B = reinterpret_cast<const uint32_t*>(rb);
        float* C = static_cast<float*>(rc);

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,  %1,  %2,  %3},"
            "{%4,  %5,  %6,  %7},"
            "{%8,  %9},"
            "{%10, %11, %12, %13};\n"
            : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[2]),
              "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,  %1,  %2,  %3},"
            "{%4,  %5,  %6,  %7},"
            "{%8,  %9},"
            "{%10, %11, %12, %13};\n"
            : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[1]), "r"(B[3]),
              "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
    }
};

template <>
struct TiledMma<__half, __half> {
    DEVICE void operator()(const __half* ra, const __half* rb, __half* rc) {
        const uint32_t* A = reinterpret_cast<const uint32_t*>(ra);
        const uint32_t* B = reinterpret_cast<const uint32_t*>(rb);
        uint32_t* C = reinterpret_cast<uint32_t*>(rc);

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
            "{%0, %1}, "
            "{%2, %3, %4, %5}, "
            "{%6, %7}, "
            "{%8, %9};"
            // D matrix
            : "=r"(C[0]), "=r"(C[1])
            // A matrix
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
              // B matrix
              "r"(B[0]), "r"(B[2]),
              // Accumulator
              "r"(C[0]), "r"(C[1]));

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
            "{%0, %1}, "
            "{%2, %3, %4, %5}, "
            "{%6, %7}, "
            "{%8, %9};"
            // D matrix
            : "=r"(C[2]), "=r"(C[3])
            // A matrix
            : "r"(A[4]), "r"(A[5]), "r"(A[6]), "r"(A[7]),
              // B matrix
              "r"(B[1]), "r"(B[3]),
              // Accumulator
              "r"(C[2]), "r"(C[3]));
    }
};

template <>
struct TiledMma<cutlass::half_t, cutlass::half_t> {
    DEVICE void operator()(const cutlass::half_t* ra, const cutlass::half_t* rb,
                           cutlass::half_t* rc) {
        const uint32_t* A = reinterpret_cast<const uint32_t*>(ra);
        const uint32_t* B = reinterpret_cast<const uint32_t*>(rb);
        uint32_t* C = reinterpret_cast<uint32_t*>(rc);

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
            "{%0, %1}, "
            "{%2, %3, %4, %5}, "
            "{%6, %7}, "
            "{%8, %9};"
            // D matrix
            : "=r"(C[0]), "=r"(C[1])
            // A matrix
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
              // B matrix
              "r"(B[0]), "r"(B[2]),
              // Accumulator
              "r"(C[0]), "r"(C[1]));

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
            "{%0, %1}, "
            "{%2, %3, %4, %5}, "
            "{%6, %7}, "
            "{%8, %9};"
            // D matrix
            : "=r"(C[2]), "=r"(C[3])
            // A matrix
            : "r"(A[4]), "r"(A[5]), "r"(A[6]), "r"(A[7]),
              // B matrix
              "r"(B[1]), "r"(B[3]),
              // Accumulator
              "r"(C[2]), "r"(C[3]));
    }
};

/// @brief: Functor to warp wmma PTX instruction. See the below document for
///         various choices and detailed parameters of the wmma PTX instruction.
///         https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma
template <typename RegTileA, typename RegTileB, typename RegTileC>
struct Gemm {
    using InTypeA = typename RegTileA::DType::DType;
    using InTypeB = typename RegTileB::DType::DType;
    using OutType = typename RegTileC::DType::DType;

    using BaseShape = traits::BaseTileShape<InTypeA>;

    static_assert(std::is_same_v<InTypeA, cutlass::half_t> ||
                      std::is_same_v<InTypeA, __half>,
                  "This GEMM implementation supports only half-precision as "
                  "the input element type.");
    static_assert(std::is_same_v<OutType, float> ||
                      std::is_same_v<OutType, __half> ||
                      std::is_same_v<OutType, cutlass::half_t>,
                  "The output type must be float or half.");
    static_assert(std::is_same_v<InTypeA, InTypeB>,
                  "Mismatched data type for operand A and B.");
    static_assert(RegTileB::kRows == RegTileA::kCols,
                  "Mismatched k-dimension for operand A and B.");

    static constexpr int kMs = RegTileA::kRows;
    static constexpr int kNs = RegTileB::kCols;
    static constexpr int kKs = RegTileA::kCols;
    static_assert(kMs && kNs && kKs, "Invalid tile shapes for GEMM.");

    DEVICE void operator()(const RegTileA& a, const RegTileB& b, RegTileC& c) {
        for (int i = 0; i < kMs; ++i) {
            for (int j = 0; j < kNs; ++j) {
#pragma unroll
                for (int k = 0; k < kKs; ++k) {
                    tile_wmma(a(i, k).data(), b(k, j).data(),
                              c(i, j).mutable_data());
                }
            }
        }
    }

  private:
    TiledMma<InTypeA, OutType> tile_wmma;
};

}  // namespace detail

template <typename RegTileA, typename RegTileB, typename RegTileC>
DEVICE void gemm(const RegTileA& a, const RegTileB& b, RegTileC& c) {
    detail::Gemm<RegTileA, RegTileB, RegTileC> gemm;
    gemm(a, b, c);
}

}  // namespace tilefusion::cell::compute

