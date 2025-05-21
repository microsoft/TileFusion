"""Setup configuration for TileFusion python package."""

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import glob
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from packaging.version import Version, parse
from setuptools import Command, Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from torch.utils.cpp_extension import CUDA_HOME

cur_path = Path(__file__).parent


def get_cuda_bare_metal_version(cuda_dir: str) -> tuple[str, Version]:
    """Get the CUDA version from nvcc.

    Args:
        cuda_dir: Path to CUDA installation directory.

    Returns:
        tuple[str, Version]: Raw nvcc output and parsed version.
    """
    raw_output = subprocess.check_output(
        [os.path.join(cuda_dir, "bin", "nvcc"), "-V"], text=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])
    return raw_output, bare_metal_version


def nvcc_threads() -> int:
    """Get the number of threads for nvcc compilation.

    Returns:
        int: Number of threads to use.
    """
    if CUDA_HOME is None:
        return os.cpu_count() or 1
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version >= Version("11.2"):
        nvcc_threads = os.getenv("NVCC_THREADS")
        if nvcc_threads is not None:
            return int(nvcc_threads)
        return os.cpu_count() or 1
    return os.cpu_count() or 1


class CMakeExtension(Extension):
    """Extension class for CMake-based builds."""

    def __init__(
        self,
        name: str = "tilefusion",
        cmake_lists_dir: str = ".",
        **kwargs: Any,
    ) -> None:
        """Initialize the CMake extension.

        Args:
            name: Name of the extension.
            cmake_lists_dir: Directory containing CMakeLists.txt.
            **kwargs: Additional arguments for Extension.
        """
        Extension.__init__(self, name, sources=[], **kwargs)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)

        if os.path.isdir(".git") and os.path.exists(".gitmodules"):
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                check=True,
            )
        else:
            dependencies = [
                "3rd-party/cutlass/include/cutlass/cutlass.h",
                "3rd-party/googletest/googletest/include/gtest/gtest.h",
            ]
            for dep_file in dependencies:
                if not os.path.exists(dep_file):
                    raise RuntimeError(
                        f"{dep_file} is missing, "
                        "please use source distribution or git clone"
                    )


class CMakeBuildExt(build_ext):
    """Build extension using CMake."""

    def copy_extensions_to_source(self) -> None:
        """Copy built extensions to source directory."""
        pass

    def build_extension(self, ext: CMakeExtension) -> None:
        """Build the extension using CMake.

        Args:
            ext: The extension to build.
        """
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("Cannot find CMake executable") from None

        debug = (
            int(os.environ.get("DEBUG", 0))
            if self.debug is None
            else self.debug
        )
        cfg = "Debug" if debug else "Release"

        # Set CUDA_ARCH_LIST to build the shared library
        # for the specified GPU architectures.
        arch_list = os.environ.get("CUDA_ARCH_LIST")
        if arch_list is not None:
            for arch in arch_list.split(" "):
                arch_num = int(arch.split(".")[0])
                if arch_num < 8:
                    raise ValueError("CUDA_ARCH_LIST must be >= 8.0")

        parallel_level = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL")
        if parallel_level is not None:
            self.parallel = int(parallel_level)
        else:
            self.parallel = os.cpu_count()

        for ext in self.extensions:
            extdir = os.path.abspath(
                os.path.dirname(self.get_ext_fullpath(ext.name))
            )
            extdir = os.path.join(extdir, "tilefusion")

            cmake_args = [
                f"-DCMAKE_BUILD_TYPE={cfg}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}",
                (
                    "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY"
                    f"_{cfg.upper()}={self.build_temp}"
                ),
                f"-DUSER_CUDA_ARCH_LIST={arch_list}" if arch_list else "",
                f"-DNVCC_THREADS={nvcc_threads()}",
            ]

            # Adding CMake arguments set as environment variable
            if "CMAKE_ARGS" in os.environ:
                cmake_args += [
                    item for item in os.environ["CMAKE_ARGS"].split(" ") if item
                ]

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            build_args = []
            build_args += ["--config", cfg]
            # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
            # across all generators.
            if (
                "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ
                and hasattr(self, "parallel")
                and self.parallel
            ):
                build_args += [f"-j{self.parallel}"]

            build_temp = Path(self.build_temp) / ext.name
            if not build_temp.exists():
                build_temp.mkdir(parents=True)

            # Config
            subprocess.check_call(
                ["cmake", ext.cmake_lists_dir] + cmake_args, cwd=self.build_temp
            )

            # Build
            subprocess.check_call(
                ["cmake", "--build", "."] + build_args, cwd=self.build_temp
            )


class Develop(develop):
    """Post-installation for development mode."""

    def run(self) -> None:  # type: ignore[override]
        """Run the develop command."""
        develop.run(self)

        project_root = os.path.dirname(os.path.abspath(__file__))
        python_dir = os.path.join(project_root, "python")
        tilefusion_link = os.path.join(project_root, "tilefusion")

        if os.path.exists(tilefusion_link):
            if os.path.islink(tilefusion_link):
                os.remove(tilefusion_link)
            else:
                shutil.rmtree(tilefusion_link)

        if os.path.exists(python_dir):
            try:
                os.symlink("python", tilefusion_link)
                print("Symlink created successfully")  # noqa: T201

                build_py = self.get_finalized_command("build_py")
                build_lib = build_py.build_lib
                built_lib = os.path.join(
                    build_lib, "tilefusion", "libtilefusion.so"
                )
                target_lib = os.path.join(python_dir, "libtilefusion.so")

                if os.path.exists(built_lib):
                    print(  # noqa: T201
                        f"Copying dynamic library from {built_lib} "
                        f"to {target_lib}"
                    )
                    shutil.copy2(built_lib, target_lib)
                else:
                    print(  # noqa: T201
                        f"Warning: Built library not found at {built_lib}"
                    )
            except Exception as e:
                print(f"Error during setup: {e}")  # noqa: T201
        else:
            print(  # noqa: T201
                f"Warning: python directory not found at {python_dir}"
            )


class Clean(Command):
    """Clean command to remove build artifacts."""

    def initialize_options(self) -> None:
        """Initialize the clean command options."""
        pass

    def finalize_options(self) -> None:
        """Finalize the clean command options."""
        pass

    def run(self) -> None:
        """Run the clean command."""
        # Clean the symlink which is created in the develop mode if it exists
        tilefusion_link = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "tilefusion"
        )
        if os.path.exists(tilefusion_link):
            print(f"cleaning symlink {tilefusion_link}")  # noqa: T201
            if os.path.islink(tilefusion_link):
                os.remove(tilefusion_link)
            else:
                shutil.rmtree(tilefusion_link)

        # Clean the dynamic library in python directory
        # copied in the develop mode if it exists
        python_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "python"
        )
        if os.path.exists(python_dir):
            for so_file in glob.glob(os.path.join(python_dir, "*.so")):
                print(f"cleaning dynamic library {so_file}")  # noqa: T201
                try:
                    os.remove(so_file)
                except OSError as e:
                    print(f"Error removing {so_file}: {e}")  # noqa: T201

        with open(".gitignore") as f:
            ignores = f.read()
            pat = re.compile(r"^#( BEGIN NOT-CLEAN-FILES )?")
            for wildcard in filter(None, ignores.split("\n")):
                match = pat.match(wildcard)
                if match:
                    if match.group(1):
                        # Marker is found and stop reading .gitignore.
                        break
                else:
                    # Don't remove absolute paths from the system
                    wildcard = wildcard.lstrip("./")

                    for filename in glob.glob(wildcard):
                        print(f"cleaning {filename}")  # noqa: T201
                        try:
                            os.remove(filename)
                        except OSError:
                            shutil.rmtree(filename, ignore_errors=True)


class PythonTest(Command):
    """Custom test command to run python unit tests."""

    user_options = [
        ("pytest-args=", "a", "Arguments to pass to pytest"),
    ]

    def initialize_options(self) -> None:
        """Initialize the test command options."""
        self.pytest_args = ""

    def finalize_options(self) -> None:
        """Finalize the test command options."""
        pass

    def run(self) -> None:
        """Run all python unit tests using pytest."""
        errno = pytest.main(["tests/python"] + self.pytest_args.split())
        if errno != 0:
            raise SystemExit(errno)


class CppTest(Command):
    """Custom test command to run C++ unit tests with ctest."""

    user_options = [
        ("ctest-args=", "a", "Arguments to pass to ctest"),
    ]

    def initialize_options(self) -> None:
        """Initialize the test command options."""
        self.ctest_args = ""

    def finalize_options(self) -> None:
        """Finalize the test command options."""
        pass

    def run(self) -> None:
        """Run the C++ tests using ctest."""
        try:
            cmake_path = subprocess.check_output(
                ["which", "cmake"], text=True
            ).strip()
            cmake_dir = os.path.dirname(cmake_path)
            ctest_path = os.path.join(cmake_dir, "ctest")
        except subprocess.CalledProcessError:
            raise RuntimeError("Could not find cmake executable") from None

        build_dir = "build"

        # Reconfigure CMake with testing enabled
        try:
            cmake_path = subprocess.check_output(
                ["which", "cmake"], text=True
            ).strip()
            print("Reconfiguring CMake with testing enabled...")  # noqa: T201
            subprocess.run(
                [cmake_path, "-DWITH_TESTING=ON", ".."],
                cwd=build_dir,
                check=True,
            )
            # Get parallel level from environment or use CPU count
            parallel_level = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL")
            if parallel_level is not None:
                parallel = int(parallel_level)
            else:
                parallel = os.cpu_count() or 1
            # Rebuild after reconfiguration with parallel jobs
            subprocess.run(
                [cmake_path, "--build", ".", f"-j{parallel}"],
                cwd=build_dir,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to configure CMake: {e}") from e

        try:
            # Run ctest in the build directory
            errno = subprocess.call(
                [ctest_path, "--output-on-failure"] + self.ctest_args.split(),
                cwd=build_dir,
            )
            if errno != 0:
                raise SystemExit(errno)
        except OSError as e:
            raise RuntimeError(f"Failed to run ctest: {e}") from e


description = "Python wrapper for tilefusion C++ library."

with open(os.path.join("python", "__version__.py")) as f:
    version_file_content = f.read()
    version_line = next(
        line
        for line in version_file_content.split("\n")
        if line.startswith("__version__")
    )
    __version__ = version_line.split("=")[1].strip().strip("'\"")

setup(
    name="tilefusion",
    version=__version__,
    description=description,
    author="Ying Cao, Chengxiang Qi",
    author_email="ying.cao@microsoft.com",
    url="https://github.com/microsoft/TileFusion",
    packages=["tilefusion"],
    package_dir={"tilefusion": "python"},
    python_requires=">=3.9",
    cmdclass={
        "build_ext": CMakeBuildExt,
        "develop": Develop,
        "clean": Clean,
        "pytests": PythonTest,
        "ctests": CppTest,
    },
    ext_modules=[CMakeExtension()],
    zip_safe=False,
    package_data={
        "tilefusion": ["**/*.py"],
    },
    include_package_data=True,
)
