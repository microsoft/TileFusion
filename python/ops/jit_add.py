"""JIT-compiled addition kernel example."""

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch
import torch.autograd


def jit_add(
    input1: torch.Tensor,
    input2: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """JIT-compiled element-wise addition.

    This function performs element-wise addition of two tensors using a
    JIT-compiled CUDA kernel.

    Args:
        input1: First input tensor
        input2: Second input tensor
        output: Output tensor that will store the result

    Raises:
        RuntimeError: If inputs are not CUDA tensors of the same shape and type.
    """
    # Validate inputs
    if not input1.is_cuda or not input2.is_cuda or not output.is_cuda:
        raise RuntimeError("All tensors must be CUDA tensors")

    if input1.shape != input2.shape or input1.shape != output.shape:
        raise RuntimeError("All tensors must have the same shape")

    if (
        input1.dtype != torch.float32
        or input2.dtype != torch.float32
        or output.dtype != torch.float32
    ):
        raise RuntimeError("All tensors must be of type float32")

    # Call the C++ function
    torch.ops.tilefusion.jit_add(input1, input2, output)
