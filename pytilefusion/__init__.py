"""PyTileFusion: A Python wrapper for tilefusion C++ library."""

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import ctypes
from typing import Any

import torch


def _load_library(filename: str) -> Any:
    """Load the C++ library.

    Args:
        filename: Name of the library file.

    Returns:
        Any: The loaded library.

    Raises:
        RuntimeError: If the library cannot be loaded.
    """
    try:
        return ctypes.CDLL(filename)
    except OSError as e:
        print(f"Failed to load library: {filename}")  # noqa: T201
        print(f"Error: {e}")  # noqa: T201
        raise RuntimeError(f"Failed to load library: {filename}") from e


_load_library("libtilefusion.so")


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


def flash_attention_fwd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_size: int,
    key_size: int,
    query_dim: int,
    value_dim: int,
) -> None:
    """Forward pass of flash attention.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        output: Output tensor.
        query_size: Size of first dimension of query.
        key_size: Size of first dimension of key/value.
        query_dim: Size of second dimension of query/key.
        value_dim: Size of second dimension of value/output.
    """
    torch.ops.tilefusion.flash_attention_fwd(
        query, key, value, output, query_size, key_size, query_dim, value_dim
    )


class TiledFlashAttention:
    """A class implementing tiled flash attention."""

    def __init__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        """Initialize the tiled flash attention.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
        """
        self.m, self.k = query.size(-2), query.size(-1)
        self.n, self.p = value.size(-2), value.size(-1)

        self.query = query.half().flatten()
        self.key = key.half().t().flatten()
        self.value = value.half().t().flatten()

        self.output = torch.empty(
            self.m, self.p, dtype=torch.half, device="cuda"
        ).flatten()

    def forward(self) -> torch.Tensor:
        """Perform the forward pass of tiled flash attention.

        Returns:
            torch.Tensor: The attention output.
        """
        flash_attention_fwd(
            self.query,
            self.key,
            self.value,
            self.output,
            self.m,
            self.n,
            self.k,
            self.p,
        )

        return self.output.view(self.m, self.p)
