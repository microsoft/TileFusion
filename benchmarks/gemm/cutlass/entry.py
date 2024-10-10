# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

entry = """#include "../cutlass_gemm.cuh"

extern "C" int kernel_entry(const __half* dA, const __half* dB, __half* dC) {{
    using DType = cutlass::half_t;

    auto* A = reinterpret_cast<const DType*>(dA);
    auto* B = reinterpret_cast<const DType*>(dB);
    auto* C = reinterpret_cast<DType*>(dC);

    cute_gemm<DType, {WarpPerRow}, {WarpPerCol}, {kM}, {kN}, {kK}, {kTM}, {kTN},
              {kTK}>(A, B, C);
    return 0;
}}
"""
