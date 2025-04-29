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
    mean: float = 5e-3,
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


# @pytest.mark.parametrize(
#     "a_rows,a_cols,b_cols,c_cols",
#     [
#         (256, 128, 64, 64),
#         # (1024, 1024, 1024, 1024),
#         # (512, 512, 512, 512),
#     ],
# )
def test_fused_two_gemms(
    a_rows: int,
    a_cols: int,
    b_cols: int,
    c_cols: int,
    dtype: torch.dtype,
    device: str,
    tensor_params: dict[str, float],
) -> None:
    """Test the fused two gemms operation with different matrix sizes."""
    input_a = create_tensor(
        (a_rows, a_cols),
        dtype=dtype,
        device=device,
        **tensor_params,
    )
    input_b = create_tensor(
        (a_cols, b_cols),
        dtype=dtype,
        device=device,
        **tensor_params,
    )
    input_c = create_tensor(
        (b_cols, c_cols),
        dtype=dtype,
        device=device,
        **tensor_params,
    )

    output = torch.zeros(a_rows, c_cols, dtype=dtype, device=device)

    fused_two_gemms(input_a, input_b, input_c, output)

    # ref = input_a @ input_b @ input_c

    # assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), (
    #     "Fused two gemms result does not match reference for "
    #     f"matrix sizes: {a_rows}x{a_cols}, "
    #     f"{a_cols}x{b_cols}, {b_cols}x{c_cols}"
    # )


if __name__ == "__main__":
    # pytest.main([__file__, "-v"])

    test_fused_two_gemms(
        256, 128, 64, 64, torch.float16, "cuda", {"mean": 5e-3, "std": 1e-2}
    )
