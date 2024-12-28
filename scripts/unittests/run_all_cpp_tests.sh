#!/bin/bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

BUILD_DIR="build"
TESTS_DIR="$(pwd)/tests/cpp"

if [ ! -d $BUILD_DIR ]; then
    echo "This script should be run from the root of the project."
    exit 0
fi

if [ $(find "$BUILD_DIR/tests/cpp" -name "test_*" | wc -l) -eq 0 ]; then
    echo "No cpp tests are found."
    exit 0
fi

cd $BUILD_DIR

for file in $(find "$TESTS_DIR/cell/" -name "test_*.cu"); do
    test_name=$(basename $file .cu)
    echo "Running test: $test_name"
    ctest -R $test_name
done
