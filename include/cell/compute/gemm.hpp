// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "types/layout.hpp"
#include "types/tile_shape.hpp"

namespace tilefusion::cell::compute {
namespace tl = tile_layout;

using MMA_ATOM_16x16x16 = TileShape<16, 16, 16>;

template <typename InTypeA, typename InTypeB, typename AccType,
          typename AtomicShape>
struct MmaAtom;

template <>
struct MmaAtom<__half, __half, float, MMA_ATOM_16x16x16> {
    struct BaseTile {
        static constexpr int kRows = 16;
        static constexpr int kCols = 16;
        static constexpr int kNumel = 256;
    };
    using BaseTileA = BaseTile;
    using BaseTileB = BaseTile;
    using BaseTileC = BaseTile;

    DEVICE void operator()(const __half* ra, const __half* rb, float* rc) {
        const uint32_t* A = reinterpret_cast<const uint32_t*>(ra);
        const uint32_t* B = reinterpret_cast<const uint32_t*>(rb);
        float* C = static_cast<float*>(rc);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
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
#else
        assert(false &&
               "This GEMM implementation requires SM80 (Ampere) or later "
               "architecture");

#endif
    }
};

template <>
struct MmaAtom<__half, __half, __half, MMA_ATOM_16x16x16> {
    struct BaseTile {
        static constexpr int kRows = 16;
        static constexpr int kCols = 16;
        static constexpr int kNumel = 256;
    };
    using BaseTileA = BaseTile;
    using BaseTileB = BaseTile;
    using BaseTileC = BaseTile;

    DEVICE void operator()(const __half* ra, const __half* rb, __half* rc) {
        const uint32_t* A = reinterpret_cast<const uint32_t*>(ra);
        const uint32_t* B = reinterpret_cast<const uint32_t*>(rb);
        uint32_t* C = reinterpret_cast<uint32_t*>(rc);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
            "{%0, %1}, "
            "{%2, %3, %4, %5}, "
            "{%6, %7}, "
            "{%8, %9};"
            : "=r"(C[0]), "=r"(C[1])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[2]),
              "r"(C[0]), "r"(C[1]));

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
            "{%0, %1}, "
            "{%2, %3, %4, %5}, "
            "{%6, %7}, "
            "{%8, %9};"
            : "=r"(C[2]), "=r"(C[3])
            : "r"(A[4]), "r"(A[5]), "r"(A[6]), "r"(A[7]), "r"(B[1]), "r"(B[3]),
              "r"(C[2]), "r"(C[3]));
#else
        assert(false &&
               "This GEMM implementation requires SM80 (Ampere) or later "
               "architecture");
#endif
    }
};

template <>
struct MmaAtom<__bfloat16, __bfloat16, float, MMA_ATOM_16x16x16> {
    struct BaseTile {
        static constexpr int kRows = 16;
        static constexpr int kCols = 16;
        static constexpr int kNumel = 256;
    };
    using BaseTileA = BaseTile;
    using BaseTileB = BaseTile;
    using BaseTileC = BaseTile;

    DEVICE void operator()(const __bfloat16* ra, const __bfloat16* rb,
                           float* rc) {
        const uint32_t* A = reinterpret_cast<const uint32_t*>(ra);
        const uint32_t* B = reinterpret_cast<const uint32_t*>(rb);
        float* C = static_cast<float*>(rc);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,  %1,  %2,  %3},"
            "{%4,  %5,  %6,  %7},"
            "{%8,  %9},"
            "{%10, %11, %12, %13};\n"
            : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[2]),
              "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,  %1,  %2,  %3},"
            "{%4,  %5,  %6,  %7},"
            "{%8,  %9},"
            "{%10, %11, %12, %13};\n"
            : "+f"(C[4]), "+f"(C[5]), "+f"(C[6]), "+f"(C[7])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[1]), "r"(B[3]),
              "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
#else
        assert(false &&
               "This GEMM implementation requires SM80 (Ampere) or later "
               "architecture");
#endif
    }
};

/// @brief: Functor to warp wmma PTX instruction. See the below document for
///         various choices and detailed parameters of the wmma PTX instruction.
///         https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma
template <typename RegTileA, typename RegTileB, typename RegTileC,
          typename AtomicShape>
struct Gemm {
    using InTypeA = typename RegTileA::DType::DType;
    using InTypeB = typename RegTileB::DType::DType;
    using OutType = typename RegTileC::DType::DType;

    static_assert(std::is_same_v<InTypeA, __half> ||
                      std::is_same_v<InTypeA, __bfloat16>,
                  "This GEMM implementation supports only half-precision as "
                  "the input element type.");
    static_assert(std::is_same_v<OutType, float> ||
                      std::is_same_v<OutType, __half>,
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
                    mma(a(i, k).data(), b(k, j).data(), c(i, j).mutable_data());
                }
            }
        }
    }

  private:
    using MmaAtom = MmaAtom<InTypeA, InTypeB, OutType, AtomicShape>;
    MmaAtom mma;
};

template <typename RegTileA, typename RegTileB, typename RegTileC,
          typename AtomicShape = MMA_ATOM_16x16x16,
          typename GemmOp = Gemm<RegTileA, RegTileB, RegTileC, AtomicShape>>
DEVICE void gemm(const RegTileA& a, const RegTileB& b, RegTileC& c) {
    GemmOp gemm;
    gemm(a, b, c);
}

}  // namespace tilefusion::cell::compute
