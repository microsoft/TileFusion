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
) -> None:
    """GEMM operation.

    Args:
        tensor_a: Input tensor.
        tensor_b: Input tensor.
        tensor_c: Output tensor.
        num_stages: Number of pipeline stages.
    """
    matrix_m = tensor_a.shape[0]
    matrix_n = tensor_b.shape[1]
    matrix_k = tensor_a.shape[1]
    torch.ops.tilefusion.gemm(
        tensor_a, tensor_b, tensor_c, matrix_m, matrix_n, matrix_k, num_stages
    )
