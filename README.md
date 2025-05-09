<div align="center">
  <img src="assets/TileFusion-logo.png" width="120"/>
  <h1>TileFusion: A High-Level, Modular<br>Tile Processing Library</h1>
  <p>
    <a href="https://tiledtensor.github.io/tilefusion-docs/docs/installation.html"><b>Installation</b></a> |
    <a href="https://tiledtensor.github.io/tilefusion-docs/docs/design/core_concepts"> <b>Getting Started</b></a> |
    <a href="https://tiledtensor.github.io/tilefusion-docs/docs/examples/101_gemm"><b>Examples</b></a> |
    <a href="https://tiledtensor.github.io/tilefusion-docs/"><b>Documentation</b></a>
  </p>
</div>

## Overview

**TileFusion**, derived from the research presented in this [paper](https://dl.acm.org/doi/pdf/10.1145/3694715.3695961), is an efficient C++ macro kernel library designed to elevate the level of abstraction in CUDA C for tile processing. The library offers:

- **Higher-Level Programming Constructs**: TileFusion supports tiles across the three-level GPU memory hierarchy, providing device kernels for transferring tiles between CUDA memory hierarchies and for tile computation.
- **Modularity**: TileFusion enables applications to process larger tiles built out of BaseTiles in both time and space, abstracting away low-level hardware details.
- **Efficiency**: The library's BaseTiles are designed to match TensorCore instruction shapes and encapsulate hardware-specific performance parameters, ensuring optimal utilization of TensorCore capabilities.

A core design goal of **TileFusion** is to allow users to understand and utilize provided primitives using logical concepts, without delving into low-level hardware complexities. The library rigorously separates data flow across the memory hierarchy from the configuration of individual macro kernels. This design choice enables performance enhancements through tuning, which operates in three possible ways:

- **Structural Tuning**: Designs various data flows while keeping kernel configurations unchanged.
- **Parameterized Tuning**: Adjusts kernel configurations while maintaining the same data flow.
- **Combined Tuning**: Integrates both structural and parameterized tuning approaches simultaneously.

In summary, **TileFusion** encourages algorithm developers to focus on designing the data flow of their algorithms using efficient tile primitives. It can be utilized as:

1. A lightweight C++ library with header-only usage, offering superior readability, modifiability, and debuggability.
2. A Python library with pre-existing kernels bound to PyTorch.

## The Basic GEMM Example

TileFusion approaches the efficient implementation of a kernel by:

1. Managing dataflow over memory hierarchies.
2. Configuring tile primitives, such as tile shapes, layouts, and other parameters.

This is an example of a simple GEMM (General Matrix Multiplication) kernel written using TileFusion. For the complete example, please refer to [this directory](examples/101_gemm/01_gemm_global_reg/gemm.hpp).

### Configuration of the Tile Primitives

The core programming constructs in TileFusion are `Tile`, `TileLayout`, `TileIterator`, `Loader`, and `Storer`.

1. **Declare the `Tile`**: [GlobalTile](https://github.com/microsoft/TileFusion/blob/master/include/types/global.hpp) and [RegTile](https://github.com/microsoft/TileFusion/blob/master/include/types/register.hpp) are utilized to customize the shape and layout of 1D (vector) or 2D (matrix) arrays within the GPU's three memory hierarchies, known as a *Tile*.

2. **Declare the `TileIterator`**: Partition the `GlobalTile` into smaller, manageable sub-tiles for efficient processing.

3. **Declare Loader and Storer**: Loaders and Storers use cooperative threads to transfer a tile from the source to the target location. They operate at the CTA level and accept the following inputs:

   - **Warp Layout**
   - **Target Tile**
   - **Source Tile**

   Based on these parameters, they automatically infer a copy plan that partitions the data transfer work among the threads.

```cpp
1  using WarpLayout = RowMajor<2, 2>;
2
3  // operand A
4  using GlobalA = GlobalTile<InType, RowMajor<128, 256>>;
5  using IteratorA = TileIterator<GlobalA, TileShape<128, 32>>;
6  using RegA = RegTile<BaseTileRowMajor<__half>, RowMajor<8, 8>>;
7  using ALoader = GlobalToRegLoader<RegA, WarpLayout, kRowReuseCont>;
8
9  // operand B
10 using GlobalB = GlobalTile<InType, ColMajor<256, 64>>;
11 using IteratorB = TileIterator<GlobalB, TileShape<32, 64>>;
12 using RegB = RegTile<BaseTileColMajor<__half>, ColMajor<8, 4>>;
13 using BLoader = GlobalToRegLoader<RegB, WarpLayout, kColReuseCont>;
14
15 // output C
16 using GlobalC = GlobalTile<AccType, RowMajor<128, 64>>;
17 using RegC = RegTile<BaseTileRowMajor<float>, RowMajor<8, 8>>;
18 using CStorer = RegToGlobalStorer<GlobalC, RegC, WarpLayout>;
```

> **Note**: To simplify the demonstration, this example involves only two memory levels: global memory and registers. TileFusion also applies similar concepts to [SharedTile](https://github.com/microsoft/TileFusion/blob/master/include/types/shared.hpp).

### Dataflow Over Memory Hierarchies

The the kernel is defined as implementing the following dataflow over memory hierarchies:

```cpp
1  template <typename InType, typename AccType,
2            typename IteratorA, typename RegA, typename LoaderA,
3            typename IteratorB, typename RegB, typename LoaderB,
4            typename GlobalC, typename RegC, typename CStorer>
5  __global__ void simple_gemm(const InType* dA, const InType* dB, AccType* dC) {
6      IteratorA gAs(dA);
7      RegA rA;
8      LoaderA loader_a;
9
10     IteratorB gBs(dB);
11     RegB rB;
12     LoaderB loader_b;
13
14     RegC acc;
15
16     for (int k = 0; k < IteratorA::sc1; ++k) {
17         loader_a(gAs(k), rA);
18         loader_b(gBs(k), rB);
19         __syncthreads();
20
21         gemm(rA, rB, acc);
22     }
23     __syncthreads();
24
25     GlobalC gC(dC);
26     CStorer storer_c;
27     storer_c(acc, gC);
28 }
```

The `TileIterator` (`IteratorA`, `IteratorB` in lines 6 and 10) serves as a syntactic interface for defining tile partitions. It is used to divide the `GlobalTile` into smaller sub-tiles and iterate over them.

`Loader` and `Storer` (declared in lines 8, 12, and 26) are efficient methods for loading and storing data, transferring data between memory hierarchies using specialized hardware-accelerated instructions (lines 17, 18, and 27). Tiles of data are cooperatively loaded into the `RegTile`, which is stored in each thread's local register file.

Once the data is loaded into a thread's local register file, `gemm` (in line 21) performs matrix multiplication using TensorCore's warp-level matrix multiply-and-accumulate (WMMA) instruction on the `BaseTile`s. The specialized data distribution required by TensorCore is automatically maintained by TileFusion's `RegTile` layout.

After the `gemm` operation is completed, the data in the `RegTile` is cooperatively stored back from registers to global memory using the `RegToGlobalStorer`.

## Installation

TileFusion can be used as a lightweight C++ library with header-only usage, or it can be built as a Python library. You can choose to build either one.

### Prerequisites

TileFusion requires:

- C++20 host compiler
- CUDA 12.0 or later
- GCC version 10.0 or higher to support C++20 features

Download the repository:

```bash
git clone git@github.com:microsoft/TileFusion.git
cd TileFusion && git submodule update --init --recursive
```

### Building the C++ Library

To build the project using the provided `Makefile`, simply run:

```bash
make
```

To run a single C++ unit test:

```bash
make unit_test_cpp CPP_UT=test_gemm
```

### Building the Python Package

1. Build the wheel:

   ```bash
   python setup.py build bdist_wheel
   ```

2. Clean the build:

   ```bash
   python setup.py clean
   ```

3. Install the Python package in editable mode (recommended for development):

   ```bash
   python setup.py develop
   ```

   This allows you to edit the source code directly without needing to reinstall it repeatedly.

### Running Unit Tests

Before running the Python unit tests, you need to build and install the Python package (see the [Building the Python Package](#building-the-python-package) section).

- **Run a single Python unit test**:

  ```bash
  pytest tests/python/test_scatter_nd.py
  ```

- **Run all Python unit tests**:

  ```bash
  python setup.py pytests
  ```

- **Run all C++ unit tests**:

  ```bash
  python setup.py ctests
  ```

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit <https://cla.opensource.microsoft.com>.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
