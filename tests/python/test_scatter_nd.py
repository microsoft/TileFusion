"""Test scatter operations.

isort:skip_file
"""

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import random

import pytest
import torch

from tilefusion.ops import scatter_nd


def compute_output_shape(
    index_dims: list[int], input_dims: list[int]
) -> list[int]:
    """Compute the output shape for scatter_nd operation.

    Args:
        index_dims: Dimensions of the indices tensor.
        input_dims: Dimensions of the input tensor.

    Returns:
        List[int]: The computed output shape.
    """
    end_size = index_dims[-1]
    out_shape = index_dims[:-1]
    for i in range(len(input_dims) - end_size):
        out_shape.append(input_dims[len(index_dims) + i])
    return out_shape


@pytest.fixture(autouse=True)
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(1234)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ],
    ids=lambda x: str(x).split(".")[-1],
)
def test_scatter_nd(dtype: torch.dtype) -> None:
    """Test scatter_nd operation with different data types.

    Args:
        dtype: The data type to test with
    """
    data_shape = [7, 8, 9, 10]
    data = torch.full(data_shape, 5.0, dtype=dtype, device="cuda")
    scatter_data = data.flatten()

    indices_shape = [5, 2]
    indices = torch.empty(indices_shape, dtype=torch.int64, device="cuda")

    for i in range(indices_shape[0]):
        indices[i][0] = random.randint(0, data_shape[0] - 1)
        indices[i][1] = random.randint(0, data_shape[1] - 1)

    scatter_indices = indices.flatten()

    update_shape = compute_output_shape(indices_shape, data_shape)
    updates = torch.full(update_shape, 10.0, dtype=dtype, device="cuda")
    scatter_updates = updates.flatten()

    scatter_nd(scatter_data, scatter_indices, scatter_updates)

    data[indices[:, 0], indices[:, 1]] = updates

    flattened_data = data.flatten()

    print(scatter_data)  # noqa: T201
    print(flattened_data)  # noqa: T201

    torch.testing.assert_close(
        scatter_data,
        flattened_data,
        rtol=1e-3,
        atol=1e-3,
        msg=f"scatter_nd failed for dtype {dtype}",
    )
