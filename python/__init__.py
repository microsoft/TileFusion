"""The Python wrapper for tilefusion C++ library."""

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import ctypes
from pathlib import Path
from typing import Any

from . import ops
from .ops import jit_add

__all__ = [
    "ops",
    "jit_add",
]


def _load_library(filename: str) -> Any:
    """Load the C++ library.

    Args:
        filename: Name of the library file.

    Returns:
        Any: The loaded library.

    Raises:
        RuntimeError: If the library cannot be loaded.
    """
    lib_path = Path(__file__).parent / filename
    print(lib_path)  # noqa: T201

    try:
        return ctypes.CDLL(str(lib_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load library from {lib_path}") from e


_lib = _load_library("libtilefusion.so")
