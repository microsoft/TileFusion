<div align="center">
  <img src="assets/TileFusion-logo.png" width="120"/>
</div>

# TileFusion: Simplifying Kernel Fusion with Tile Processing

**TileFusion** is a highly efficient C++ macro kernel template library designed to elevate the level of abstraction in CUDA C for processing tiles. It is designed to be:

- **Higher-Level Programming**: TileFusion offers a set of device kernels for transferring tiles between the CUDA device's three memory hierarchies and for computing tiles.
- **Modularity**: TileFusion enables users to construct their applications by processing larger tiles in time and space using the provided BaseTiles.
- **Efficiency**: TileFusion offers highly efficient implementations of these device kernels.

TileFusion adopts a hardware bottom-up approach by building kernels around the core concept of the **BaseTile**. The shapes of these BaseTiles align with TensorCore's instruction shape and encapsulate hardware-dependent performance parameters to optimally utilize TensorCore's capabilities. Serving as building blocks, these BaseTiles are then combined to construct larger tiles in both temporal and spatial dimensions, enabling users to process larger tiles composed of BaseTiles for their applications.

## Basic GEMM Example

TileFusion implements `GlobalTile`, `SharedTile` and `RegTile` to customize the shape and layout of tiles located in the GPU's three memory hierarchies. Here's an example of a simple GEMM kernel written in TileFusion (the complete example can be found in [this directory](examples/cpp/01_gemm/01_gemm_global_reg/gemm.hpp)):

(*To simplify the demonstration, this example only involves two memory levels: global memory and registers. TileFusion also applies a similar concept to shared memory*.)

```cpp
template <typename InType, typename AccType, typename IteratorA, typename RegA,
          typename LoaderA, typename IteratorB, typename RegB, typename LoaderB,
          typename GlobalC, typename RegC, typename CStorer>
__global__ void simple_gemm(const InType* dA, const InType* dB, AccType* dC) {
    IteratorA gAs(dA);
    RegA rA;
    LoaderA loader_a;

    IteratorB gBs(dB);
    RegB rB;
    LoaderB loader_b;

    RegC acc;

    for (int k = 0; k < IteratorA::sc1; ++k) {
        loader_a(gAs(k), rA);
        loader_b(gBs(k), rB);
        __syncthreads();

        gemm(rA, rB, acc);
    }
    __syncthreads();

    GlobalC gC(dC);
    CStorer storer_c;
    storer_c(acc, gC);
}
```

- The `TileIterator` is used to divide the `GlobalTile` into smaller sub-tiles and iterate over them. Various warp reuse methods are provided to support efficient repeated loading of data by warps within a thread block. TileFusion provides efficient loading and storing methods that transfer data between memory hierarchies by utilizing specialized hardware-accelerated instructions. Tiles of data are then cooperatively loaded into the `RegTile`, which is stored in each thread's local register file.

- Once the data is loaded into a thread's local register file, `gemm` performs matrix multiplication using TensorCore's warp-level matrix multiply-and-accumulate (wmma) instruction on the `BaseTile`s. The specialized data distribution required by TensorCore is automatically maintained by TileFusion's `RegTile` layout.

- After the `gemm` operation is completed, data in the `RegTile` is cooperatively stored back from registers to global memory using the `RegToGlobalStorer`.

Here is how to declare the `Tile` at each level of memory, use `TileIterator` to chunk large tiles into sub-tiles, and declare loaders and storers to transfer tiles between memory hierarchies.

```cpp
using WarpLayout = RowMajor<2, 2>;

// operand A
using GlobalA = GlobalTile<InType, RowMajor<128, 256>>;
using IteratorA = TileIterator<GlobalA, TileShape<128, 32>>;
using RegA = RegTile<BaseTileRowMajor<__half>, RowMajor<8, 8>>;
using ALoader = GlobalToRegLoader<RegA, WarpLayout, kRowReuseCont>;

// operand B
using GlobalB = GlobalTile<InType, ColMajor<256, 64>>;
using IteratorB = TileIterator<GlobalB, TileShape<32, 64>>;
using RegB = RegTile<BaseTileColMajor<__half>, ColMajor<8, 4>>;
using BLoader = GlobalToRegLoader<RegB, WarpLayout, kColReuseCont>;

// output C
using GlobalC = GlobalTile<AccType, RowMajor<128, 64>>;
using RegC = RegTile<BaseTileRowMajor<float>, RowMajor<8, 8>>;
using CStorer = RegToGlobalStorer<GlobalC, RegC, WarpLayout>;
```

### Download

```bash
git clone git@github.com:microsoft/TileFusion.git
cd TileFusion && git submodule update --init --recursive
```

### Installation

TileFusion requires a C++20 host compiler, CUDA 12.0 or later, and GCC version 10.0 or higher to support C++20 features.

### Build from Source

#### Building the C++ Library Using Makefile

1. To build the project using the provided `Makefile`, simply run:

   ```bash
   make
   ```

2. Run the C++ unit tests:

   - **Run a single C++ unit test**:
     ```bash
     make unit_test_cpp CPP_UT=test_gemm
     ```
   - **Run all C++ unit tests**:
     ```bash
     make unit_test_cpps
     ```

#### Building the Python Wrapper

1. Build the wheel:

   ```bash
   python3 setup.py build bdist_wheel
   ```

2. Clean the build:

   ```bash
   python3 setup.py clean
   ```

3. Install the Python wrapper in editable mode (recommended for development):

   ```bash
   python3 setup.py develop
   ```

   This allows you to edit the source code directly without needing to reinstall it repeatedly.

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

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
