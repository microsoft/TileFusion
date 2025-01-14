#!/bin/bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

echo "Make sure the Python wrapper is installed first."

for file in $(find tests/python -name "*.py"); do
    echo "Running unit test: $file"
    python3 $file
done
