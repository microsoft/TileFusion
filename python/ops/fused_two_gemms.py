"""Fused two gemms implementation for tilefusion."""

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch

__all__ = [
    "fused_two_gemms",
]


def _validate_tensor(
    tensor: torch.Tensor,
    name: str,
    expected_dtype: torch.dtype = torch.float16,
    expected_device: str = "cuda",
) -> None:
    """Validate tensor properties.

    Args:
        tensor: The tensor to validate.
        name: The name of the tensor for error messages.
        expected_dtype: The expected data type.
        expected_device: The expected device type.

    Raises:
        ValueError: If the tensor is not on the expected device.
        TypeError: If the tensor is not of the expected data type.
    """
    if tensor.device.type != expected_device:
        raise ValueError(f"{name} must be a {expected_device} tensor")
    if tensor.dtype != expected_dtype:
        raise TypeError(
            f"{name} must be of type {expected_dtype}, got {tensor.dtype}"
        )


def fused_two_gemms(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    input_c: torch.Tensor,
    output: torch.Tensor,
    tm: int = 64,
    tn: int = 64,
    tk: int = 64,
    tp: int = 64,
) -> None:
    """Fused two gemms implementation.

    This kernel computes the following:

    .. math::
        D = input_a @ input_b @ input_c

    Args:
        input_a: The first input matrix with a shape of (M, K) and is laid out
                 in the row-major fashion.
        input_b: The second input matrix with a shape of (K, N) and is laid out
                 in the column-major fashion.
        input_c: The third input matrix with a shape of (N, P) and is laid out
                 in the column-major fashion.
        output: The output matrix with a shape of (M, P), laid out in the
                row-major fashion.
        tm: The shared memory tile size of the m-dimension.
        tn: The shared memory tile size of the n-dimension.
        tk: The shared memory tile size of the k-dimension.
        tp: The shared memory tile size of the p-dimension.
    """
    _validate_tensor(input_a, "input_a")
    _validate_tensor(input_b, "input_b")
    _validate_tensor(input_c, "input_c")
    _validate_tensor(output, "output")

    torch.ops.tilefusion.fused_two_gemms(
        input_a, input_b, input_c, output, tm, tn, tk, tp
    )
