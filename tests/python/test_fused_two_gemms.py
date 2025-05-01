"""Test for fused two gemms operation."""

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import pytest
import torch

from tilefusion.ops.fused_two_gemms import fused_two_gemms


@pytest.fixture
def dtype() -> torch.dtype:
    """Return the data type for the test."""
    return torch.float16


@pytest.fixture
def device() -> str:
    """Return the device for the test."""
    return "cuda"


@pytest.fixture
def tensor_params() -> dict[str, float]:
    """Return the tensor parameters for the test."""
    return {
        "mean": 5e-3,
        "std": 1e-2,
    }


def create_tensor(
    size: tuple[int, ...],
    mean: float = 5e-2,
    std: float = 1e-2,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> torch.Tensor:
    """Create a tensor with the given parameters."""
    return torch.normal(
        mean=mean,
        std=std,
        size=size,
        device=device,
        dtype=dtype,
    )


@pytest.mark.parametrize(
    "kM, kN, kK, kP, kTM, kTN, kTK, kTP",
    [  # TODO(ying): pass more test cases
        (256, 128, 64, 64, 64, 64, 64, 64),
    ],
)
def test_fused_two_gemms(
    kM: int,
    kN: int,
    kK: int,
    kP: int,
    kTM: int,
    kTN: int,
    kTK: int,
    kTP: int,
    dtype: torch.dtype,
    device: str,
    tensor_params: dict[str, float],
) -> None:
    """Test the fused two gemms operation with different matrix sizes."""
    # torch tensors by default are laid out in row-major fashion
    input_a = create_tensor(
        (kM, kK),
        dtype=dtype,
        device=device,
        **tensor_params,
    )
    input_b = create_tensor(
        (kN, kK),
        dtype=dtype,
        device=device,
        **tensor_params,
    )
    input_c = create_tensor(
        (kP, kN),
        dtype=dtype,
        device=device,
        **tensor_params,
    )
    output = torch.zeros(kM, kP, dtype=dtype, device=device)

    # It is required that:
    # A and D are laid out in row-major fashion
    # B and C are laid out in column-major fashion
    fused_two_gemms(input_a, input_b, input_c, output, kTM, kTN, kTK, kTP)
    ref = input_a @ input_b.t() @ input_c.t()

    print(output)  # noqa: T201
    print(ref)  # noqa: T201

    assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), (
        "Fused two gemms result does not match reference for "
        f"matrix sizes: {kM}x{kK}, {kK}x{kN}, {kN}x{kP}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
