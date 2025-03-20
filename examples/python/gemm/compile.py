"""Module for compiling CUDA code for matrix multiplication."""

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import ctypes
import importlib.util
import os
import subprocess
from collections import defaultdict
from typing import Optional

import torch

__all__ = [
    "Compile",
]

cutlass_include_dir = os.path.join(
    os.path.dirname(__file__), "../../../3rd-party/cutlass/include"
)
tilefusion_include_dir = os.path.join(
    os.path.dirname(__file__), "../../../include/"
)
csrc_include_dir = os.path.join(os.path.dirname(__file__), "csrc")


class Compile:
    """Class for compiling CUDA code."""

    def __init__(self, file_prefix: str, tmp_dir: str) -> None:
        """Initialize the compiler.

        Args:
            file_prefix: Prefix for generated files.
            tmp_dir: Directory for temporary files.
        """
        self.tmp_dir = tmp_dir
        self.file_prefix = file_prefix

        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        compute_capability = torch.cuda.get_device_capability()
        self.cc = f"{compute_capability[0]}{compute_capability[1]}"

        self.nvcc_path = self._find_nvcc_path()

    def _find_nvcc_path(self) -> str:
        """Find the path to nvcc compiler.

        Returns:
            str: Path to nvcc compiler.

        Raises:
            RuntimeError: If nvcc cannot be found.
        """

        def py_str(input_bytes: bytes) -> str:
            return input_bytes.decode("utf-8")

        if "CUDA_PATH" in os.environ:
            return os.environ["CUDA_PATH"]

        cmd = ["which", "nvcc"]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        (out, _) = proc.communicate()

        if proc.returncode == 0:
            return py_str(out.strip())
        raise RuntimeError("Cannot find cuda path")

    def _create_entry_code(
        self,
        matrix_m: int,
        matrix_n: int,
        matrix_k: int,
        tile_m: int,
        tile_n: int,
        chunk_k: int,
        warp_per_row: int,
        warp_per_col: int,
    ) -> str:
        """Create the entry code for compilation.

        Args:
            matrix_m: Size of first dimension of matrix A.
            matrix_n: Size of second dimension of matrix B.
            matrix_k: Size of second dimension of matrix A and first dimension
                      of matrix B.
            tile_m: Tile size for M dimension.
            tile_n: Tile size for N dimension.
            chunk_k: Chunk size for K dimension.
            warp_per_row: Number of warps per row.
            warp_per_col: Number of warps per column.

        Returns:
            str: The generated entry code.
        """
        entry_code_path = "entry.py"
        spec = importlib.util.spec_from_file_location(
            "entry_code", entry_code_path
        )
        if spec is None:
            raise RuntimeError("Failed to create module spec")

        entry_module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise RuntimeError("Module spec has no loader")
        spec.loader.exec_module(entry_module)

        shape = defaultdict(int)
        shape["kM"] = matrix_m
        shape["kN"] = matrix_n
        shape["kK"] = matrix_k
        shape["kTM"] = tile_m
        shape["kTN"] = tile_n
        shape["kChunkK"] = chunk_k
        shape["warp_per_row"] = warp_per_row
        shape["warp_per_col"] = warp_per_col

        return str(entry_module.types.format_map(shape) + entry_module.entry)

    def compile(
        self,
        matrix_m: int,
        matrix_n: int,
        matrix_k: int,
        tile_m: int,
        tile_n: int,
        chunk_k: int,
        warp_per_row: int,
        warp_per_col: int,
        timeout: Optional[float] = None,
    ) -> Optional[str]:
        """Compile the CUDA code.

        Args:
            matrix_m: Size of first dimension of matrix A.
            matrix_n: Size of second dimension of matrix B.
            matrix_k: Size of second dimension of matrix A and first dimension
                      of matrix B.
            tile_m: Tile size for M dimension.
            tile_n: Tile size for N dimension.
            chunk_k: Chunk size for K dimension.
            warp_per_row: Number of warps per row.
            warp_per_col: Number of warps per column.
            timeout: Timeout for compilation in seconds.

        Returns:
            Optional[str]: Path to the compiled library, or None if compilation
            failed.

        Raises:
            RuntimeError: If compilation fails.
        """
        temp_dir = self.tmp_dir

        file_name = (
            f"{self.file_prefix}_{matrix_m}_{matrix_n}_{matrix_k}"
            f"_{tile_m}_{tile_n}_{warp_per_row}_{warp_per_col}"
        )
        lib_path = os.path.join(temp_dir, f"{file_name}.so")

        if os.path.exists(lib_path):
            return lib_path

        entry_code = self._create_entry_code(
            matrix_m,
            matrix_n,
            matrix_k,
            tile_m,
            tile_n,
            chunk_k,
            warp_per_row,
            warp_per_col,
        )

        source_path = os.path.join(temp_dir, f"{file_name}.cu")
        with open(source_path, "w") as f:
            f.write(entry_code)

        if os.path.exists(lib_path):
            return lib_path

        command = [
            self.nvcc_path,
            "-std=c++20",
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--disable-warnings",
            "--compiler-options",
            "'-fPIC'",
            "--shared",
            source_path,
            "-lcuda",
            f"-gencode=arch=compute_{self.cc},code=sm_{self.cc}",
            f"-I{cutlass_include_dir}",
            f"-I{tilefusion_include_dir}",
            f"-I{csrc_include_dir}",
            "-o",
            lib_path,
        ]
        try:
            ret = subprocess.run(command, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        if ret.returncode == 0:
            return lib_path
        raise RuntimeError("Compilation failed")

    def apply(
        self, lib_path: str, torch_array: list[torch.Tensor], device: int
    ) -> int:
        """Apply the compiled kernel.

        Args:
            lib_path: Path to the compiled library.
            torch_array: List of torch tensors to pass to the kernel.
            device: CUDA device to use.

        Returns:
            int: Return code from the kernel.
        """
        lib = ctypes.CDLL(lib_path)

        lib.kernel_entry.restype = ctypes.c_int
        torch.cuda.set_device(device)

        result = lib.kernel_entry(
            *[ctypes.c_void_p(arr.data_ptr()) for arr in torch_array]
        )
        return int(result)
