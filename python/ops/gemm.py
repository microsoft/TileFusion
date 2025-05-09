"""GEMM operations for tilefusion."""

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch

__all__ = [
    "gemm",
]


def gemm(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    tensor_c: torch.Tensor,
    num_stages: int,
    pipeline_level: int,
) -> None:
    """GEMM operation.

    Args:
        tensor_a: Input tensor.
        tensor_b: Input tensor.
        tensor_c: Output tensor.
        num_stages: pipeline the loop into this many stages.
        pipeline_level: The GEMM operation is executed in three levels:
                        Level 1: Basic GEMM execution without pipelining.
                        Level 2: Incorporates pipelining of some loops using
                        asynchronous copy instructions from global to shared
                        memory.
                        Level 3: Further pipelining by prefetching data from
                        shared memory to registers for certain loops and
                        kernels.
    """
    matrix_m = tensor_a.shape[0]
    matrix_n = tensor_b.shape[1]
    matrix_k = tensor_a.shape[1]
    torch.ops.tilefusion.gemm(
        tensor_a,
        tensor_b,
        tensor_c,
        matrix_m,
        matrix_n,
        matrix_k,
        num_stages,
        pipeline_level,
    )
