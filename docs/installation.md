---
layout: page
title: Installation
nav_order: 2
has_children: false
---

TileFusion requires a C++20 host compiler, CUDA 12.0 or later, and GCC version 10.0 or higher to support C++20 features.

## Download

```bash
git clone git@github.com:microsoft/TileFusion.git
cd TileFusion && git submodule update --init --recursive
```

## Build from Source

### Building the C++ Library Using Makefile

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

### Building the Python Wrapper

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
