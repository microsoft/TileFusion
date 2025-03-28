"""Scatter operations for tilefusion."""

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch

__all__ = [
    "scatter_nd",
]


def scatter_nd(
    scatter_data: torch.Tensor,
    scatter_indices: torch.Tensor,
    scatter_updates: torch.Tensor,
) -> None:
    """Scatter updates into a tensor at specified indices.

    Args:
        scatter_data: The tensor to scatter updates into.
        scatter_indices: The indices where updates should be scattered.
        scatter_updates: The updates to scatter.
    """
    torch.ops.tilefusion.scatter_nd(
        scatter_data, scatter_updates, scatter_indices
    )
