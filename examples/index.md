---
layout: page
title: Examples
nav_order: 4
has_children: true
---

This page showcases various examples of using TileFusion.

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

## More Examples

Check out our [examples directory](https://github.com/microsoft/TileFusion/tree/main/examples) for more complete examples.
