// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "convert.cuh"
#include "copy.cuh"
#include "cuda_utils.cuh"
#include "cutlass/copy.cuh"
#include "cutlass/traits_base.cuh"
#include "reduce.cuh"

namespace benchmarks {
namespace cutlass_wrapper {

using namespace cute;

template <typename Element_, const int kM, const int kN, const int kK,
          const int kP, const int kTM, const int kTN, const int kTK,
          const int kTP, const int kWarpPerRow, const int kWarpPerCol,
          const int SmemKAtom = 64, const int kSwizzle = 3,
          typename Base = AccessBase<Element_>>
struct FATraits : public Base {
    using Element = Element_;

    static_assert(kTP == kP, "The current implementation requires kTP == P.");

    // Declare global to shared memory copy layout.
    using GmemLayoutQ = Layout<Shape<Int<kTM>, Int<kTK>>, Stride<Int<kK>, _1>>;
    using GmemLayoutK = Layout<Shape<Int<kTN>, Int<kTK>>, Stride<Int<kK>, _1>>;
    using GmemLayoutV = Layout<Shape<Int<kTP>, Int<kTN>>, Stride<Int<kN>, _1>>;
    using GmemLayoutO = Layout<Shape<Int<kTM>, Int<kTP>>, Stride<Int<kP>, _1>>;

    static constexpr int kThreads = kWarpPerRow * kWarpPerCol * 32;

    /**
     * Define the atomic layout of shared memory, which is the smallest
     * configuration unit of shared memory. Larger shapes are tiled based on the
     * atomic layout.
     */
    using SmemLayoutAtom = decltype(composition(
        Swizzle<kSwizzle, 3, 3>{},
        Layout<Shape<_8, Int<SmemKAtom>>, Stride<Int<SmemKAtom>, _1>>{}));

    using SmemLayoutQ =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTK>>{}));
    using SmemLayoutK =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTN>, Int<kTK>>{}));
    using SmemLayoutV =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTP>, Int<kTN>>{}));
    using SmemLayoutO =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTP>>{}));

    /**
     * In the Ampere architecture, loading from shared memory to register memory
     * requires the use of the `ldmatrix` instruction, while storing from
     * register memory to shared memory does not have hardware support and uses
     * a default copy instead.”
     */
    using LoadS2RCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using StoreR2SCopyAtom = Copy_Atom<DefaultCopy, Element>;

    static constexpr int kWarps = kThreads / 32;

    using TiledMma =
        TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                 Layout<Shape<Int<kWarpPerRow>, Int<kWarpPerCol>, _1>>,
                 Tile<Int<16 * kWarpPerRow>, Int<16 * kWarpPerCol>, _16>>;

#ifdef CP_ASYNC_SM80_ENABLED
    // for Ampere
    using CopyInstG2S =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>;
#else
    using CopyInstG2S = Copy_Atom<DefaultCopy, Element>;
#endif

    // TODO(KuangjuX): Understand this configuration.
    using GmemCopyLayoutAtom =
        Layout<Shape<Int<kThreads / (SmemKAtom / 8)>, Int<SmemKAtom / 8>>,
               Stride<Int<SmemKAtom / 8>, _1>>;

    using TiledCopyG2S = decltype(make_tiled_copy(
        CopyInstG2S{}, GmemCopyLayoutAtom{}, Layout<Shape<_1, _8>>{}));

    using TiledCopyS2G = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{}, GmemCopyLayoutAtom{},
        Layout<Shape<_1, _8>>{}));
};

template <typename Element, typename KeTraits, const int kM, const int kN,
          const int kK, const int kP, const int kTM, const int kTN,
          const int kTK, const int kTP, const int Nthreads, const int kStagesQK,
          const int kStageV>
__global__ void __launch_bounds__(Nthreads)
    fa_kernel(const Element* dQ, const Element* dK, const Element* dV,
              Element* dO) {
    constexpr float softmax_scale = 1.250000e-01f;
    const bool load_q_once = (kTK == kK);

    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);

    const Element* Q = dQ + blockIdx.z * kTM * kN + blockIdx.x * kTM * kK;
    const Element* K = dK + blockIdx.z * kK * kN;
    const Element* V = dV + blockIdx.z * kP * kN + blockIdx.y * kTP * kN;
    Element* O =
        dO + blockIdx.z * kM * kP + blockIdx.x * (kTM * kP) + blockIdx.y * kTP;

    Element* sQ_ptr = reinterpret_cast<Element*>(buf);
    Element* sK_ptr = sQ_ptr + kTM * kTK * kStagesQK;
    Element* sV_ptr = sK_ptr + kTN * kTK * kStagesQK;
    Element* sO_ptr = sQ_ptr;

    typename KeTraits::TiledMma mma;
    typename KeTraits::TiledCopyG2S tiled_copy_g2s;

    // Build the copy plan for QK from global memory to shared memory.
    auto g2s_copy_qk = make_g2s_qk<
        Element, typename KeTraits::GmemLayoutQ, typename KeTraits::SmemLayoutQ,
        typename KeTraits::GmemLayoutK, typename KeTraits::SmemLayoutK,
        typename KeTraits::TiledCopyG2S>(Q, sQ_ptr, K, sK_ptr, kTK, kTK);

    /**
     * In FractalTensor, The size of the V matrix is [kN, kP], and the size
     * processed in a single SM Block is [kN, kTP]. When split along the N
     * dimension, the size is [kTN, kTP]. Therefore, the stride for global
     * memory should be set to kTN * kP.
     *
     * In the current implementation, the shape of the V matrix is [kP, kN], and
     * the block size processed by a single Block is [kTP, kN]. Therefore, the
     * stride only needs to be set to kTN each time.
     */
    auto g2s_copy_v =
        make_g2s_v<Element, typename KeTraits::GmemLayoutV,
                   typename KeTraits::SmemLayoutV,
                   typename KeTraits::TiledCopyG2S>(V, sV_ptr, kTN);

    auto acc0 = get_acc<kTM, kTN>(mma);
    auto acco = get_acc<kTM, kTP>(mma);

    if (thread0()) {
        printf("acc0 size<0>: %d, size<1>: %d, size<2>: %d\n",
               (int)size<0>(acc0), (int)size<1>(acc0), (int)size<2>(acc0));
        printf("acco size<0>: %d, size<1>: %d, size<2>: %d\n",
               (int)size<0>(acco), (int)size<1>(acco), (int)size<2>(acco));
    }

    /**
     * In TileFusion, we use
     * ```cpp
     *  using RegVec = RegTile<InType, tl::RowMajor<kAccMs, 2>>;
     * ```
     * We need to store the reduce results for both the top row and the bottom
     * row simultaneously.
     */

    auto m_new = make_tensor<float>(Shape<Int<2 * size<1>(acc0)>>{});
    auto lse_new = make_fragment_like(m_new);

    auto s2r_pipeline_qk =
        make_s2r_qk(sQ_ptr, sK_ptr, typename KeTraits::SmemLayoutQ{},
                    typename KeTraits::SmemLayoutK{}, acc0,
                    typename KeTraits::LoadS2RCopyAtom{}, mma);

    auto s2r_pipeline_v =
        make_s2r_v(sV_ptr, typename KeTraits::SmemLayoutV{}, acco,
                   typename KeTraits::LoadS2RCopyAtom{}, mma);

    // Issue global to shared memory copy before the main loop.
    g2s_copy_qk.prologue();

    fill(lse_new, 0.0f);
    fill(m_new, -INFINITY);
    clear(acco);

    /**
     * Flash Attention performs two-level tiling for each SM Block, splitting
     * along the N dimension and the K dimension. The Q matrix is split along
     * the K dimension, the V matrix is split along the N dimension, and the K
     * matrix is split along both dimensions simultaneously.
     */
    int split_n = kN / kTN;
    for (int n = 0; n < split_n; ++n) {
        clear(acc0);

        // When `load_q_once` is true, the folling code is not executed.
        int slice_k = kK / kTK - 1;
        for (int k = 0; k < slice_k; ++k) {
            // Barrier to ensure all data are loaded into shared memory.
            cp_async_wait_flash<0>();
            __syncthreads();
            g2s_copy_qk.body();
            // Load data from shared memory into register and issue MMA.
            s2r_pipeline_qk.body();
        }

        cp_async_wait_flash<0>();
        __syncthreads();
        g2s_copy_v.prologue();
        // When `load_q_once` is true, `g2s_copy_qk.prologue()` is executed only
        // once, and `s2r_pipeline_qk.epilogue()` is executed once as well.
        s2r_pipeline_qk.epilogue();

        // scores = dot(q, k)
        auto scores =
            make_tensor(acc0.data(), convert_layout_scores(acc0.layout()));

        auto m_old = make_fragment_like(m_new);
        copy(m_new, m_old);

        auto scores_max = make_fragment_like(m_new);

        // scores_max = reduce_max(scores, axis=1)
        reduce_max<4, true>(scores, scores_max);

        // Compute new partial max value.
        for (int ax0 = 0; ax0 < size<0>(m_new); ++ax0) {
            m_new(ax0) = max(m_new(ax0), scores_max(ax0));
        }

        // Currently, `acco` stores the results from the previous iteration's
        // computation.
        auto previous_attn_block =
            make_tensor(acco.data(), convert_layout_scores(acco.layout()));

        if (thread0()) {
            printf("scores size<0>: %d, size<1>: %d\n", (int)size<0>(scores),
                   (int)size<1>(scores));
            printf("previous_attn_block size<0>: %d, size<1>: %d\n",
                   (int)size<0>(previous_attn_block),
                   (int)size<1>(previous_attn_block));
        }

        // Renormalization for the previous block.
        for (int ax0 = 0; ax0 < size<0>(previous_attn_block); ++ax0) {
            float scale = exp((m_old(ax0) - m_new(ax0)) * softmax_scale);
            lse_new(ax0) = lse_new(ax0) * scale;
            for (int ax1 = 0; ax1 < size<1>(previous_attn_block); ++ax1) {
                previous_attn_block(ax0, ax1) *= scale;
            }
        }

        for (int ax0 = 0; ax0 < size<0>(scores); ++ax0) {
            float m_scaled = exp((m_old(ax0) - m_new(ax0)) * softmax_scale);
            lse_new(ax0) = lse_new(ax0) * m_scaled;
            for (int ax1 = 0; ax1 < size<1>(scores); ++ax1) {
                scores(ax0, ax1) =
                    exp(scores(ax0, ax1) * softmax_scale - m_scaled);
            }
        }

        auto scores_sum = make_fragment_like(lse_new);
        reduce_sum<4>(scores, scores_sum);

        for (int ax0 = 0; ax0 < size<0>(lse_new); ++ax0) {
            lse_new(ax0) = lse_new(ax0) + scores_sum(ax0);
        }

        // TODO(KuangjuX): Understand the following code.
        auto frag = convert_type<Element>(scores);
        auto rP = make_tensor(make_rmem_ptr<Element>(&frag), scores.layout());
        auto rP_Aregs =
            make_tensor(rP.data(), convert_layout_rowcol_Aregs(rP.layout()));

        /**
         * In FractalTensor, the `kTN` dimension is split again. To simplify the
         * current implementation of rhe pipeline flashattention, the `tile_n`
         * is hardcoded to 0 at this point.
         */
        const int tile_n = 0;
        for (int tile_ = 0; tile_ < tile_n; ++tile_) {
            // Barrier to ensure all data are loaded into shared memory.
            cp_async_wait_flash<0>();
            __syncthreads();
            g2s_copy_v.body();
            s2r_pipeline_v.body(rP_Aregs);
        }

        cp_async_wait_flash<0>();
        __syncthreads();

        if (n < split_n - 1) {
            /**
             * Update K tile because the entire K Block will be processed in a
             * single SM Block.
             *
             * For example, In `TileFusion`:
             * ```cpp
             * for (int n = 0; n < GIteratorV::sc0; ++n) {
             *      load_sv(gVs(n), sV);
             *      for (int k = 0; k < GIteratorQ::sc1; ++k) {
             *          load_sq(gQs(k), sQ);
             *          load_sk(gKs(k, n), sK);
             *      }
             * }
             * ```
             */
            g2s_copy_qk.update_tile_K(kTN, kK);
            /**
             * `load_q_once` means that at this point `kK == kTK`, and the Q is
             * loaded into shared memory in blocks only once. In this case, we
             * only need to update the pointer of K and do not need to update
             * the pointer for Q, because the blocking along the k dimension
             * will not be executed, thus the Q is always reloaded.
             */
            if (load_q_once) {
                g2s_copy_qk.prologue_K();
            }
        }

        s2r_pipeline_v.epilogue(rP_Aregs);
    }

    // Normalize the attention block.
    auto attn_block =
        make_tensor(acco.data(), convert_layout_scores(acco.layout()));
    for (int ax0 = 0; ax0 < size<0>(attn_block); ++ax0) {
        float scale = 1 / lse_new(ax0);
        lse_new(ax0) = m_new(ax0) * softmax_scale + log(lse_new(ax0));
        for (int ax1 = 0; ax1 < size<1>(attn_block); ++ax1) {
            attn_block(ax0, ax1) *= scale;
        }
    }

    // Store O from registers to shared memory and then to global memory.
    store_r2s_o(sO_ptr, typename KeTraits::SmemLayoutO{}, acco,
                typename KeTraits::StoreR2SCopyAtom{}, mma);
    __syncthreads();

    store_s2g_o(O, sO_ptr, typename KeTraits::GmemLayoutO{},
                typename KeTraits::SmemLayoutO{},
                typename KeTraits::TiledCopyS2G{});
}

}  // namespace cutlass_wrapper
}  // namespace benchmarks
