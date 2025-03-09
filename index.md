---
layout: home
title: Home
nav_order: 1
---

<div align="center">
  <img src="assets/TileFusion-logo.png" width="120"/>
</div>

# TileFusion: Simplifying Kernel Fusion with Tile Processing

**TileFusion** is a highly efficient kernel template library designed to elevate CUDA C’s level of abstraction for processing tiles. It is designed to be:

- **Higher-Level Programming**: TileFusion offers a set of device kernels for transferring tiles between the CUDA device's three memory hierarchies and for computing tiles.
- **Modularity**: TileFusion enables users to construct their applications by processing larger tiles in time and space using the provided BaseTiles.
- **Efficiency**: TileFusion offers highly efficient implementations of these device kernels.

TileFusion adopts a hardware bottom-up approach by building kernels around the core concept of the **BaseTile**. The shapes of these BaseTiles align with TensorCore's instruction shape and encapsulate hardware-dependent performance parameters to optimally utilize TensorCore's capabilities. Serving as building blocks, these BaseTiles are then combined to construct larger tiles in both temporal and spatial dimensions, enabling users to process larger tiles composed of BaseTiles for their applications.

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

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
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
