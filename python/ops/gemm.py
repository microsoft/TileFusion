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
    tile_m: int,
    tile_n: int,
    tile_k: int,
    num_stages: int,
    pipeline_level: int,
) -> None:
    """GEMM operation.

    Args:
        tensor_a: Input tensor.
        tensor_b: Input tensor.
        tensor_c: Output tensor.
        tile_m: The tile size for m dimension.
        tile_n: The tile size for n dimension.
        tile_k: The tile size for k dimension.
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
    torch.ops.tilefusion.gemm(
        tensor_a,
        tensor_b,
        tensor_c,
        tile_m,
        tile_n,
        tile_k,
        num_stages,
        pipeline_level,
    )
