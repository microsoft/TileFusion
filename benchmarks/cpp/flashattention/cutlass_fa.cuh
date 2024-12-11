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

//
/// @brief
/// @tparam Element_
template <typename Element_, const int kM, const int kN, const int kK,
          const int kP, const int kTM, const int kTN, const int kTK,
          const int kTP, const int kWarpPerRow, const int kWarpPerCol,
          const int kThreads, const int SmemKAtom = 64, const int kSwizzle = 3,
          typename Base = AccessBase<Element_>>
struct FATraits : public Base {
    // Q: [kM, kK] --> [length, hidden_qk]
    // K: [kN, kK] --> [length, hidden_qk]
    // V: [kP, kN] --> [length, hidden_v]
    // O: [kM, kP] --> [length, hidden_v]
    // assert(kM == kN)
    using Element = Element_;

    // Declare global to shared memory copy layout.
    using GmemLayoutQ = Layout<Shape<Int<kTM>, Int<kTK>>, Stride<Int<kK>, _1>>;
    using GmemLayoutK = Layout<Shape<Int<kTN>, Int<kTK>>, Stride<Int<kK>, _1>>;
    using GmemLayoutV = Layout<Shape<Int<kTP>, Int<kTN>>, Stride<Int<kN>, _1>>;
    using GmemLayoutO = Layout<Shape<Int<kTM>, Int<kTP>>, Stride<Int<kP>, _1>>;

    // Atom Shared Memory Block.
    using SmemLayoutAtom = decltype(composition(
        Swizzle<kSwizzle, 3, 3>{},
        Layout<Shape<_8, Int<SmemKAtom>>, Stride<Int<SmemKAtom>, _1>>{}));

    // Declare shared memory layout.
    // [kTM / Atom, kTK / Atom, Atom]
    using SmemLayoutQ =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTK>>{}));
    // [kTN / Atom, kTK / Atom, Atom]
    using SmemLayoutK =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTN>, Int<kTK>>{}));
    // [kTP / Atom, kTN / Atom, Atom]
    using SmemLayoutV =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTP>, Int<kTN>>{}));
    using SmemLayoutO =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTP>>{}));

    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;

    static constexpr int kWarps = kThreads / 32;

    // Declare MMA Operation: [16, 8, 16] * [1, 2, 1] -> [16, 16, 16]
    // Legacy code
    // using TiledMma =
    //     TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
    //              Layout<Shape<Int<kWarps>, _1, _1>>, Layout<Shape<_1, _2,
    //              _1>>>;

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

    // Declare Copy plan.
    // Why this configuration?
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
        typename KeTraits::TiledCopyG2S>(Q, sQ_ptr, K, sK_ptr, kK, kTK, kK,
                                         kTK);

    // Build the copy plan for V from global memory to shared memory.
    auto g2s_copy_v =
        make_g2s_v<Element, typename KeTraits::GmemLayoutV,
                   typename KeTraits::SmemLayoutV,
                   typename KeTraits::TiledCopyG2S>(V, sV_ptr, kN, kTN);

#ifdef DEBUG
    g2s_copy_qk.print_gQ();
    g2s_copy_v.print_gV();
    g2s_copy_qk.print_gQ_data(0);
#endif

    auto acc0 = get_acc<kTM, kTN>(mma);
    auto acco = get_acc<kTM, kTP>(mma);

    auto m_new = make_tensor<float>(Shape<Int<2 * size<1>(acc0)>>{});
    auto lse_new = make_fragment_like(m_new);

    if (thread0()) {
        printf("acc0 size<0>: %d, size<1>: %d, size<2>: %d\n",
               (int)size<0>(acc0), (int)size<1>(acc0), (int)size<2>(acc0));
    }

    auto s2r_pipeline_qk =
        make_s2r_qk(sQ_ptr, sK_ptr, typename KeTraits::SmemLayoutQ{},
                    typename KeTraits::SmemLayoutK{}, acc0, kTK, kTK,
                    typename KeTraits::SmemCopyAtom{}, mma);
    s2r_pipeline_qk.print_rQ();

    auto s2r_pipeline_v =
        make_s2r_v(sV_ptr, typename KeTraits::SmemLayoutV{}, acco, kTN,
                   typename KeTraits::SmemCopyAtom{}, mma);

    // Issue global to shared memory copy before the main loop.
    g2s_copy_qk.prologue();

    fill(lse_new, 0.0f);
    fill(m_new, -INFINITY);
    clear(acco);

    int split_n = kN / kTN;
    for (int n = 0; n < split_n; ++n) {
        int split_k = kK / kTK - 1;
        // Pipeline
        for (int k = 0; k < split_k; ++k) {
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
        s2r_pipeline_qk.epilogue();

        // Print acc0 data.
        if (thread0()) {
            printf("acc0: \n");
            print(acc0), print("\n");
        }
        auto scores =
            make_tensor(acc0.data(), convert_layout_scores(acc0.layout()));

        auto m_old = make_fragment_like(m_new);
        copy(m_new, m_old);

        auto scores_max = make_fragment_like(m_new);

        // Compute row max.
        reduce_max<4, true>(scores, scores_max);

        // Compute new max vector.
        for (int ax0 = 0; ax0 < size<0>(m_new); ++ax0) {
            m_new(ax0) = max(m_new(ax0), scores_max(ax0));
        }

        auto acco_rowcol =
            make_tensor(acco.data(), convert_layout_scores(acco.layout()));

        // Renormalizatio for the previous block.
        for (int ax0 = 0; ax0 < size<0>(acco_rowcol); ++ax0) {
            float scale = exp((m_old(ax0) - m_new(ax0)) * softmax_scale);
            lse_new(ax0) = lse_new(ax0) * scale;
            for (int ax1 = 0; ax1 < size<1>(acco_rowcol); ++ax1) {
                acco_rowcol(ax0, ax1) *= scale;
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

        // TODO: Understand the following code.
        auto frag = convert_type<Element>(scores);
        auto rP = make_tensor(make_rmem_ptr<Element>(&frag), scores.layout());
        auto rP_Aregs =
            make_tensor(rP.data(), convert_layout_rowcol_Aregs(rP.layout()));

        // Load V into register and issue MMA.
        int split_n = kN / kTN - 1;
        for (int n = 0; n < split_n; ++n) {
            // Barrier to ensure all data are loaded into shared memory.
            cp_async_wait_flash<0>();
            __syncthreads();
            g2s_copy_v.body();
            s2r_pipeline_v.body(rP_Aregs);
        }

        cp_async_wait_flash<0>();
        __syncthreads();

        s2r_pipeline_v.epilogue(rP_Aregs);
    }

    auto acco_f16 = convert_type<Element>(acco);

    store_r2s_o(sO_ptr, typename KeTraits::SmemLayoutO{}, acco_f16,
                typename KeTraits::SmemCopyAtom{}, mma);
    __syncthreads();

    store_s2g_o(O, sO_ptr, typename KeTraits::GmemLayoutO{},
                typename KeTraits::SmemLayoutO{}, tiled_copy_g2s);
}

}  // namespace cutlass_wrapper
}  // namespace benchmarks
