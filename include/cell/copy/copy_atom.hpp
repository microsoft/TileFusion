// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/**
 * @file copy_atom.hpp
 * @brief This file contains lightweight wrappers for hardware-accelerated copy
 *        instructions.
 */
#pragma once

#include "traits/base.hpp"
#include "types/layout.hpp"

#include <cute/tensor.hpp>

namespace tilefusion::cell::copy::atom {
namespace tl = tile_layout;
using namespace cute;

namespace {
template <const int kBytes>
DEVICE void ld_global_st_shared(uint32_t dst, void const* src) {
    static_assert(kBytes == 4 || kBytes == 8 || kBytes == 16);

#if (__CUDA_ARCH__ >= 900)
    // SM90, hopper
    assert(false && "Not implemented yet.");
#elif (__CUDA_ARCH__ >= 800)
    // SM80, SM86, ampere
    // TODO(ying): add a wrapper to allow choosing between different caching
    // policies (e.g. "cache all levels").
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst),
                 "l"(src), "n"(kBytes));
#else
    // SM75, turing
    unsigned tmp[kBytes / 4];
    if constexpr (kBytes == 16) {
        asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(tmp[0]), "=r"(tmp[1]), "=r"(tmp[2]), "=r"(tmp[3])
                     : "l"(src));
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(dst),
                     "r"(tmp[0]), "r"(tmp[1]), "r"(tmp[2]), "r"(tmp[3]));
    } else if constexpr (kBytes == 8) {
        asm volatile("ld.global.v2.b32 {%0, %1}, [%2];\n"
                     : "=r"(tmp[0]), "=r"(tmp[1])
                     : "l"(src));
        asm volatile("st.shared.v2.b32 [%0], {%1, %2};\n" ::"r"(dst),
                     "r"(tmp[0]), "r"(tmp[1]));
    } else if constexpr (kBytes == 4) {
        asm volatile("ld.global.b32 %0, [%1];\n" : "=r"(tmp[0]) : "l"(src));
        asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(dst), "r"(tmp[0]));
    }
#endif
}

/// ld.shared
template <const int kBytes>
DEVICE void ld_shared(void* dst, uint32_t src);

/// ld.shared - 16b
template <>
DEVICE void ld_shared<2>(void* dst, uint32_t src) {
    asm volatile("ld.shared.u16 %0, [%1];\n"
                 : "=h"(*reinterpret_cast<uint16_t*>(dst))
                 : "r"(src));
}

/// ld.shared - 32b
template <>
DEVICE void ld_shared<4>(void* dst, uint32_t src) {
    asm volatile("ld.shared.u32 %0, [%1];\n"
                 : "=r"(*reinterpret_cast<uint32_t*>(dst))
                 : "r"(src));
}

/// ld.shared - 64b
template <>
DEVICE void ld_shared<8>(void* dst, uint32_t src) {
    uint2* dst_u64 = reinterpret_cast<uint2*>(dst);
    asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];\n"
                 : "=r"(dst_u64->x), "=r"(dst_u64->y)
                 : "r"(src));
}

/// ld.shared - 128b
template <>
DEVICE void ld_shared<16>(void* dst, uint32_t src) {
    uint4* dst_u128 = reinterpret_cast<uint4*>(dst);
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(dst_u128->x), "=r"(dst_u128->y), "=r"(dst_u128->z),
                   "=r"(dst_u128->w)
                 : "r"(src));
}

/// st.shared
template <int kBytes>
DEVICE void st_shared(uint32_t dst, void const* src);

/// st.shared - 16b
template <>
DEVICE void st_shared<2>(uint32_t dst, void const* src) {
    asm volatile("st.shared.u16 [%0], %1;\n"
                 :
                 : "r"(dst), "h"(*reinterpret_cast<uint16_t const*>(src)));
}

/// st.shared - 32b
template <>
DEVICE void st_shared<4>(uint32_t dst, void const* src) {
    asm volatile("st.shared.u32 [%0], %1;\n"
                 :
                 : "r"(dst), "r"(*reinterpret_cast<uint32_t const*>(src)));
}

/// st.shared - 64b
template <>
DEVICE void st_shared<8>(uint32_t dst, void const* src) {
    uint2 const* dst_u64 = reinterpret_cast<uint2 const*>(src);
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};\n"
                 :
                 : "r"(dst), "r"(dst_u64->x), "r"(dst_u64->y));
}

/// st.shared - 128b
template <>
DEVICE void st_shared<16>(uint32_t dst, void const* src) {
    uint4 const* dst_u128 = reinterpret_cast<uint4 const*>(src);
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
                 :
                 : "r"(dst), "r"(dst_u128->x), "r"(dst_u128->y),
                   "r"(dst_u128->z), "r"(dst_u128->w));
}

/// st.global
template <int kBytes>
DEVICE void st_global(void* dst, const void* src);

template <>
DEVICE void st_global<16>(void* dst, const void* src) {
    uint4 const* dst_u128 = reinterpret_cast<uint4 const*>(src);
    asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};\n"
                 :
                 : "l"(dst), "r"(dst_u128->x), "r"(dst_u128->y),
                   "r"(dst_u128->z), "r"(dst_u128->w));
}

template <int kBytes>
DEVICE void ld_shared_st_global(void* dst, uint32_t src);

template <>
DEVICE void ld_shared_st_global<16>(void* dst, uint32_t src) {
    unsigned tmp[4];
    ld_shared<16>(tmp, src);
    st_global<16>(dst, tmp);
}
}  // namespace

template <typename Element>
    requires std::is_same_v<Element, __half> ||
             std::is_same_v<Element, cutlass::half_t>
struct LoadMatBase {
    using DType = Element;
    using ThreadLayout = tile_layout::ColMajor<16, 2>;

    static constexpr int kAccessInBits = 128;  // 128 bits
    static constexpr int kElmentBits = sizeof(DType) * 8;
    static constexpr int kNumPerAccess = kAccessInBits / kElmentBits;

    /// @brief Returns the lane row of the current thread within a warp.
    //         For ldmatrix, threads in a warp are arranged in a 16x2
    //         column-major layout:
    //
    //         |  | 0 |  1|
    //         |--|---|---|
    //         |0 | 0 | 16|
    //         |1 | 2 | 17|
    //         |2 | 4 | 18|
    //         |  |...|...|
    //         |15| 15| 31|
    /// For example, if threadIdx.x is 43, its lane_row is 8 and lane_col is 0.

    /// @brief Returns the lane row of the current thread within a warp.
    DEVICE int lane_row_id() {
        int lane_id = threadIdx.x % WARP_SIZE;
        return lane_id % tl::num_rows<ThreadLayout>;
    }

    /// @brief returns the lane col of the current thread within a warp.
    DEVICE int lane_col_id() {
        int lane_id = threadIdx.x % WARP_SIZE;
        return lane_id / tl::num_rows<ThreadLayout>;
    }

    /// @brief a thin wrapper for executing ldmatrix instruction to load a
    ///        `16x16` tile to register.
    DEVICE void ldmatrix(const DType* src, DType* dst) {
        uint32_t* reg = reinterpret_cast<uint32_t*>(dst);
        uint32_t smem_addr =
            static_cast<uint32_t>(__cvta_generic_to_shared(src));

        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
            : "r"(smem_addr));
    }
};

template <typename Shared, const tl::Layout kType>
struct StoreMatBase;

/// TODO(haruhi): try to reduece reusable codes.
template <typename Shared>
struct StoreMatBase<Shared, tl::Layout::kRowMajor> {
    using DType = Shared::DType;

    // the thread layout for wmma's output tile.
    using ThreadLayout = tile_layout::RowMajor<8, 4>;

    static constexpr int kThreadRows = tl::num_rows<ThreadLayout>;
    static constexpr int kThreadCols = tl::num_cols<ThreadLayout>;

    // in the output of a wmma tile, each thread stores four segments in 2x2
    // layout, and each fragment contains 2 elements regardless of the data
    // type
    static constexpr int kSegRows = 2;
    static constexpr int kSegCols = 2;

    // the number of elements per segment, vectorized instruction are used to
    // access `kElemPerSeg` elements.
    static constexpr int kElemPerSeg = 2;

    static constexpr int kAccessInBits = kElemPerSeg * int(sizeof(DType) * 8);
    typename tl::SharedLayoutWrapper<Shared, kAccessInBits>::Layout in_tile_;

    DEVICE int lane_row_id() {
        return (threadIdx.x % WARP_SIZE) / tl::num_cols<ThreadLayout>;
    }

    DEVICE int lane_col_id() {
        return (threadIdx.x % WARP_SIZE) % tl::num_cols<ThreadLayout>;
    }
};

/// TODO(haruhi): try to reduece reusable codes.
template <typename Shared>
struct StoreMatBase<Shared, tl::Layout::kColMajor> {
    using DType = Shared::DType;

    // the thread layout for wmma's output tile.
    using ThreadLayout = tile_layout::ColMajor<4, 8>;

    static constexpr int kThreadRows = tl::num_rows<ThreadLayout>;
    static constexpr int kThreadCols = tl::num_cols<ThreadLayout>;

    // in the output of a wmma tile, each thread stores four segments in 2x2
    // layout, and each fragment contains 2 elements regardless of the data
    // type
    static constexpr int kSegRows = 2;
    static constexpr int kSegCols = 2;

    // the number of elements per segment, vectorized instruction are used to
    // access `kElemPerSeg` elements.
    static constexpr int kElemPerSeg = 2;

    static constexpr int kAccessInBits = kElemPerSeg * int(sizeof(DType) * 8);
    typename tl::SharedLayoutWrapper<Shared, kAccessInBits>::Layout in_tile_;

    DEVICE int lane_row_id() {
        return (threadIdx.x % WARP_SIZE) % tl::num_rows<ThreadLayout>;
    }

    DEVICE int lane_col_id() {
        return (threadIdx.x % WARP_SIZE) / tl::num_rows<ThreadLayout>;
    }
};

template <typename Global, typename Shared, typename BaseShape,
          const tl::Layout kType = Shared::kType>
struct GlobalToSharedBaseTileLoader;

/// @brief Load a BaseTile from global memory to shared memory.
template <typename Global, typename Shared, typename BaseShape_>
struct GlobalToSharedBaseTileLoader<Global, Shared, BaseShape_,
                                    tl::Layout::kColMajor> {
    using DType = Shared::DType;

    // The macro kernel breaks down the entire copy operation into iterations
    // over 16x16 BaseTiles. To transfer a single BaseTile, threads in a warp
    // are arranged in a 16x2 row-major layout. Each thread uses 128-bit data in
    // a single access.
    using ThreadLayout = tile_layout::RowMajor<2, 16>;
    static constexpr int kThreadsPerRow = tl::num_rows<ThreadLayout>;
    static constexpr int kThreadsPerCol = tl::num_cols<ThreadLayout>;

    static constexpr int kNumPerAccess =
        traits::AccessBase<DType>::kNumPerAccess;

    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kRowStride = kThreadsPerRow * kNumPerAccess;
    static constexpr int kExecCount = BaseShape::kRows / kRowStride;

    // create CuTe's compatible column-major layout for the global memory.
    using BaseTileGlobalLayout =
        cute::Layout<Shape<Int<BaseShape::kRows>, Int<BaseShape::kCols>>,
                     Stride<_1, Int<Global::kColStride>>>;

    using BaseTileSharedLayout = tl::SharedLayoutWrapper<
        Shared, traits::AccessBase<DType>::kAccessInBits>::Layout;

    /// @brief returns the lane row of the current thread within a warp.
    DEVICE int lane_row_id() {
        int lane_id = threadIdx.x % WARP_SIZE;
        return lane_id / tl::num_cols<ThreadLayout>;
    }

    /// @brief returns the lane col of the current thread within a warp.
    DEVICE int lane_col_id() {
        int lane_id = threadIdx.x % WARP_SIZE;
        return lane_id % tl::num_cols<ThreadLayout>;
    }
};

}  // namespace tilefusion::cell::copy::atom
